"""Package sample-efficiency study: train the frozen-architecture stack at
nested solved-data budgets, evaluate deployment readiness on the untouched
validation set, and (per the predeclared trigger) run the frozen hybrid
policy with the smallest budget that crosses Spearman >= 0.50.

Reads the Sobol pool (results/pkg_pool_designs.json + pkg_cache), writes
results/pkg_sample_eff_metrics.json and per-budget stacks under
results/pkg_stack_N{N}/.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from emsurr import dataset, models, pkg_task as PT, splits, synth_ext, via_task
from emsurr.risk_cal import spearman
from emsurr.topo_rep import TopoEmbedding

BUDGETS = (16, 32, 64, 128, 256)
REPL = 8
THRESH = 0.50
SYNTH_FREQ = np.linspace(50e6, 20e9, 256)


class DS6(models.DeepSetsModel):
    N_TYPES = 6
    MAX_EL = 12


def pkg_solve_to_sample(d, solver):
    """Training sample from a solved package design (declared conversion)."""
    z = np.load(solver._path(d))
    s96, f96 = z["s"], z["freq"]
    smp = PT.to_sample(d, SYNTH_FREQ)
    s = np.empty((len(SYNTH_FREQ), 2, 2), complex)
    for (a, b), src in (((0, 0), s96[:, 0, 0]), ((1, 0), s96[:, 1, 0])):
        re = np.interp(SYNTH_FREQ, f96, src.real, left=src.real[0])
        im = np.interp(SYNTH_FREQ, f96, src.imag, left=src.imag[0])
        s[:, a, b] = re + 1j * im
    s[:, 0, 1] = s[:, 1, 0]
    s[:, 1, 1] = s[:, 0, 0]
    smp["s"] = s
    return smp


def base_pool():
    cfg = yaml.safe_load(Path("configs/baseline.yaml").read_text())
    samples = dataset.load_h5(cfg["data"]["path"])
    sp = splits.make_splits(samples, seed=0, ood_families=("via_lc", "stub_short"))
    train = sp["ood_topology"]["train"]
    return (train + synth_ext.generate(400, seed=7)
            + via_task.make_via_model_family(400, seed=11)
            + PT.make_chain_model_family(400, seed=13)), cfg["models"]["mlp"]


def train_stack(pool, mk, stack_dir):
    rs = np.random.RandomState(1)
    idx = rs.permutation(len(pool))
    tr = [pool[i] for i in idx[: int(0.85 * len(pool))]]
    va = [pool[i] for i in idx[int(0.85 * len(pool)) :]]
    ens = []
    stack_dir.mkdir(parents=True, exist_ok=True)
    for i in range(5):
        p = stack_dir / f"ds6_{i}.pt"
        m = DS6(hidden=mk["hidden"], depth=mk["depth"], epochs=mk["epochs"],
                lr=mk["lr"], seed=100 + i)
        if p.exists():
            e0 = m.epochs
            m.epochs = 1
            m.fit(tr, va)
            m.epochs = e0
            m.model.load_state_dict(torch.load(p))
        else:
            m.fit(tr, va)
            torch.save(m.model.state_dict(), p)
        ens.append(m)
    enc = TopoEmbedding(ordered=True, objective="resp", n_types=6, max_el=12).fit(tr, va)
    e_ref = enc.embed(tr)
    resp_ref = np.stack([s["s"] for s in tr])

    def gap_of(samples_):
        preds = np.stack([m.predict(samples_) for m in ens])
        mu = preds.mean(0)
        eq = enc.embed(samples_)
        dmat = np.linalg.norm(eq[:, None] - e_ref[None], axis=-1)
        i3 = np.argsort(dmat, 1)[:, :3]
        w = 1.0 / (np.take_along_axis(dmat, i3, 1) + 1e-9)
        tgt = (resp_ref[i3] * w[..., None, None, None]).sum(1) / w.sum(1)[:, None, None, None]
        gap = np.abs(mu - tgt).mean(axis=(1, 2, 3))
        unc = np.abs(preds - mu).mean(axis=(0, 2, 3, 4))
        return mu, gap, unc

    _, gap_va, unc_va = gap_of(va)
    stats = dict(gap_p95=float(np.quantile(gap_va, 0.95)),
                 gap_med=float(np.median(gap_va)),
                 gap_iqr=float(np.quantile(gap_va, .75) - np.quantile(gap_va, .25) + 1e-9),
                 unc_med=float(np.median(unc_va)),
                 unc_iqr=float(np.quantile(unc_va, .75) - np.quantile(unc_va, .25) + 1e-9))
    return ens, enc, gap_of, stats


def evaluate(gap_of, val_designs, val_true):
    mu, gap, unc = gap_of([PT.to_sample(d, SYNTH_FREQ) for d in val_designs])
    surr_J = np.array([PT.surrogate_objective(mu[i], SYNTH_FREQ)
                       for i in range(len(val_designs))])
    true_J = np.array([v["J"] for v in val_true])
    errs = []
    for i, v in enumerate(val_true):
        e = 0.0
        for (a, b) in ((0, 0), (1, 0)):
            ip = np.interp(v["freq"], SYNTH_FREQ, mu[i][:, a, b].real) \
                + 1j * np.interp(v["freq"], SYNTH_FREQ, mu[i][:, a, b].imag)
            e += float(np.abs(ip - v["s"][:, a, b]).mean())
        errs.append(e / 2)
    dj = np.abs(surr_J - true_J)
    return dict(J_spearman=spearman(surr_J, true_J),
                J_mae=float(dj.mean()),
                resp_err_mean=float(np.mean(errs)),
                gap_spearman_err=spearman(gap, errs),
                unc_spearman_err=spearman(unc, errs),
                catastrophic_rate=float(np.mean(dj > 0.3)))


def main():
    designs = json.loads(Path("results/pkg_pool_designs.json").read_text())
    solver = PT.Solver()
    n_ready = sum(1 for d in designs if solver.cached(d))
    print(f"pool solved: {n_ready}/{len(designs)}")

    base, mk = base_pool()
    rngv = np.random.default_rng(99)
    val_designs = [PT.random_design(rngv) for _ in range(24)]
    val_true = solver.solve_batch(val_designs)   # cached from the pkg campaign

    res = dict(budgets={}, threshold=THRESH)
    for N in BUDGETS:
        if n_ready < N:
            print(f"budget {N}: pool not ready, skipping")
            continue
        subset = designs[:N]
        pkg_samples = [pkg_solve_to_sample(d, solver) for d in subset]
        pool = base + pkg_samples * REPL
        t0 = time.perf_counter()
        ens, enc, gap_of, stats = train_stack(pool, mk,
                                              Path(f"results/pkg_stack_N{N}"))
        r = evaluate(gap_of, val_designs, val_true)
        r["train_min"] = round((time.perf_counter() - t0) / 60, 1)
        res["budgets"][N] = r
        print(f"N={N:3d}: Spearman={r['J_spearman']:+.3f} J_MAE={r['J_mae']:.3f} "
              f"resp={r['resp_err_mean']:.3f} gap_sp={r['gap_spearman_err']:+.3f} "
              f"cat={r['catastrophic_rate']:.2f} ({r['train_min']} min)", flush=True)
        Path("results/pkg_sample_eff_metrics.json").write_text(
            json.dumps(res, indent=1))

    ready = [N for N in BUDGETS if N in res["budgets"]
             and res["budgets"][N]["J_spearman"] >= THRESH]
    res["trigger_budget"] = ready[0] if ready else None
    Path("results/pkg_sample_eff_metrics.json").write_text(json.dumps(res, indent=1))
    print("trigger budget:", res["trigger_budget"])


if __name__ == "__main__":
    main()
