"""Design-optimization application: hybrid surrogate + selective-solver vs
solver-only vs surrogate-only on the stub-and-gap interconnect task.

Stack (all frozen recipes, frozen sizes):
  surrogate  5-member DeepSets ensemble (existing architecture, N_TYPES
             extended to 6 so the gap token has a slot; hidden/epochs/lr
             identical to the frozen config), trained on the 5 synthetic
             training families + the analytic gapped_line family
  risk       retrieval-gap (distance-weighted k=3, the frozen train-side
             choice) in the frozen-recipe extended token encoder;
             ens_var from the ensemble; regime via the frozen 2-feature
             logistic deployment switch
  solver     openEMS (authoritative), content-cached

Optimizer: one differential-evolution implementation shared by all methods.
Hybrid verification policy (declared before running): a candidate is solver-
verified iff (a) its surrogate objective beats the current VERIFIED best, or
(b) its retrieval-gap exceeds the train-side p95 AND its surrogate objective
is within 20% of the verified best. The reported best design is always
openEMS-verified. Writes results/design_opt_metrics.json + QA files.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from emsurr import dataset, design_task as DT, models, splits, synth_ext
from emsurr.risk_cal import spearman
from emsurr.topo_rep import TopoEmbedding

STACK = Path("results/design_stack")
POLICY = "b"  # v2b: verified-fitness replacement implemented
RUNS = Path(f"results/design_runs_v{DT.TASK_VERSION}{POLICY}")   # per-run summaries


def _jsonable(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    raise TypeError(type(o))


def save_run(name, r):
    RUNS.mkdir(parents=True, exist_ok=True)
    (RUNS / f"{name}.json").write_text(json.dumps(r, indent=1, default=_jsonable))


def load_run(name):
    p = RUNS / f"{name}.json"
    return json.loads(p.read_text()) if p.exists() else None
import os
SEEDS = tuple(int(x) for x in os.environ.get("DESIGN_SEEDS", "0,1,2").split(","))
BUDGET = 60
POP, GENS = 12, 40
SOLVER_POP = 8
GRID = None  # synth 256-pt grid, set in main


class DS6(models.DeepSetsModel):
    N_TYPES = 6


def train_stack():
    cfg = yaml.safe_load(Path("configs/baseline.yaml").read_text())
    samples = dataset.load_h5(cfg["data"]["path"])
    sp = splits.make_splits(samples, seed=0, ood_families=("via_lc", "stub_short"))
    train = sp["ood_topology"]["train"]
    gapped = synth_ext.generate(400, seed=7)
    rs = np.random.RandomState(1)
    pool = train + gapped
    idx = rs.permutation(len(pool))
    tr = [pool[i] for i in idx[: int(0.85 * len(pool))]]
    va = [pool[i] for i in idx[int(0.85 * len(pool)) :]]

    mk = cfg["models"]["mlp"]
    ens = []
    for i in range(5):
        p = STACK / f"ds6_{i}.pt"
        m = DS6(hidden=mk["hidden"], depth=mk["depth"], epochs=mk["epochs"],
                lr=mk["lr"], seed=100 + i)
        if p.exists():
            # rebuild norms deterministically (1 epoch), then load weights
            real_epochs = m.epochs
            m.epochs = 1
            m.fit(tr, va)
            m.epochs = real_epochs
            m.model.load_state_dict(torch.load(p))
        else:
            print(f"training DS6 member {i}...")
            m.fit(tr, va)
            STACK.mkdir(parents=True, exist_ok=True)
            torch.save(m.model.state_dict(), p)
        ens.append(m)

    enc = TopoEmbedding(ordered=True, objective="resp", n_types=6).fit(tr, va)
    e_ref = enc.embed(tr)
    resp_ref = np.stack([s["s"] for s in tr])
    # train-side stats for gates
    def gap_of(samples_):
        preds = np.stack([m.predict(samples_) for m in ens])
        mu = preds.mean(0)
        eq = enc.embed(samples_)
        d = np.linalg.norm(eq[:, None] - e_ref[None], axis=-1)
        i3 = np.argsort(d, 1)[:, :3]
        w = 1.0 / (np.take_along_axis(d, i3, 1) + 1e-9)
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
    return ens, enc, gap_of, stats, tr[0]["freq"]


def de_step(pop_x, fitness, rng):
    """One DE generation: returns trial vectors."""
    n = len(pop_x)
    trials = []
    for i in range(n):
        a, b, c = rng.choice(n, 3, replace=False)
        mut = pop_x[a] + 0.6 * (pop_x[b] - pop_x[c])
        cross = rng.random(len(DT.KEYS)) < 0.8
        cross[rng.integers(len(DT.KEYS))] = True
        trials.append(np.where(cross, mut, pop_x[i]))
    return trials


def vec(d):
    return np.array([d[k] for k in DT.KEYS])


def unvec(x):
    return DT.clip({k: float(v) for k, v in zip(DT.KEYS, x)})



def run_multistart(solver, seed, gap_of, stats, freq, budget=BUDGET,
                   n_restarts=8, top_per=3):
    """v2c hybrid, aligned with the milestone's C-definition: the surrogate
    explores freely (surrogate evaluations are free), the solver budget is
    spent verifying the pooled best candidates, and nothing unverified is
    ever reported or treated as an incumbent.

    Phase 1: n_restarts independent DE runs on surrogate fitness only.
    Phase 2: pool each restart's top_per candidates, dedupe, verify in
             ascending surrogate-J order (tie-break: lower retrieval-gap,
             i.e. the frozen risk score prioritizes the trustworthy ones)
             using 60% of the budget.
    Phase 3: a short DE polish seeded around the best verified design,
             using the v2b verified-incumbent policy for the remaining
             budget. Constants are mechanical splits declared a priori.
    """
    rng = np.random.default_rng(seed)

    def surr_eval(ds):
        mu, gap, unc = gap_of([DT.to_sample(d, freq) for d in ds])
        return ([DT.objective(mu[i], freq) for i in range(len(ds))], gap, unc)

    # ---- phase 1: free multi-start surrogate search
    pool_c = {}
    n_surr = 0
    for r in range(n_restarts):
        pop = [DT.random_design(rng) for _ in range(POP)]
        fit, gap, _ = surr_eval(pop)
        px = [vec(d) for d in pop]
        n_surr += POP
        for gen in range(GENS):
            tds = [unvec(t) for t in de_step(px, fit, rng)]
            tf, tgap, _ = surr_eval(tds)
            n_surr += POP
            for i in range(POP):
                if tf[i] < fit[i]:
                    fit[i], gap[i], px[i], pop[i] = tf[i], tgap[i], vec(tds[i]), tds[i]
        for i in np.argsort(fit)[:top_per]:
            pool_c[DT.key(pop[i])] = (pop[i], fit[i], gap[i])

    cands = sorted(pool_c.values(), key=lambda t: (t[1], t[2]))
    calls = 0
    best_ver = (2.0, None)
    log, traj = [], []

    def verify_list(items, why):
        nonlocal calls, best_ver
        ds = [c[0] for c in items]
        rs_ = solver.solve_batch(ds)
        for (d, sj, g), r in zip(items, rs_):
            calls += 1
            useful = r["J"] < best_ver[0]
            if useful:
                best_ver = (r["J"], d)
            log.append(dict(call=calls, reason=why, surr_J=float(sj),
                            true_J=float(r["J"]), gap=float(g),
                            err=float(abs(sj - r["J"])), improved=bool(useful)))
            traj.append((calls, best_ver[0]))

    n1 = max(1, int(0.6 * budget))
    verify_list(cands[:n1], "pooled")

    # ---- phase 3: verified polish around the incumbent
    if best_ver[1] is not None and calls < budget:
        center = vec(best_ver[1])
        span = np.array([hi - lo for lo, hi in DT.BOUNDS.values()])
        pop = [unvec(center + rng.normal(0, 0.05, len(center)) * span)
               for _ in range(POP)]
        fit, gap, _ = surr_eval(pop)
        n_surr += POP
        px = [vec(d) for d in pop]
        verified = {}
        for gen in range(15):
            if calls >= budget:
                break
            tds = [unvec(t) for t in de_step(px, fit, rng)]
            tf, tgap, _ = surr_eval(tds)
            n_surr += POP
            for i in range(POP):
                if tf[i] < fit[i]:
                    fit[i], gap[i], px[i], pop[i] = tf[i], tgap[i], vec(tds[i]), tds[i]
            picks = [i for i in np.argsort(fit)
                     if fit[i] < best_ver[0] and DT.key(pop[i]) not in verified][:2]
            items = [(pop[i], fit[i], gap[i]) for i in picks][: budget - calls]
            if items:
                verify_list(items, "polish")
                # v2b policy: verified J permanently replaces fitness
                for j, i in enumerate(picks[: len(items)]):
                    verified[DT.key(pop[i])] = True
                    fit[i] = log[-len(items) + j]["true_J"]

    return dict(best_J=float(best_ver[0]), best_design=best_ver[1],
                calls=calls, traj=traj, log=log, n_surr_evals=n_surr)


def run_solver_only(solver, seed):
    rng = np.random.default_rng(seed)
    pop = [DT.random_design(rng) for _ in range(SOLVER_POP)]
    res = solver.solve_batch(pop)
    fit = [r["J"] for r in res]
    calls = SOLVER_POP
    traj = [(calls, min(fit))]
    px = [vec(d) for d in pop]
    while calls < BUDGET:
        trials = de_step(px, fit, rng)[: BUDGET - calls]
        tds = [unvec(t) for t in trials]
        rs_ = solver.solve_batch(tds)
        for i, (td, r) in enumerate(zip(tds, rs_)):
            calls += 1
            if r["J"] < fit[i]:
                fit[i], px[i], pop[i] = r["J"], vec(td), td
            traj.append((calls, min(fit)))
    best = int(np.argmin(fit))
    return dict(best_J=float(min(fit)), best_design=pop[best], traj=traj)


def run_search(solver, seed, gap_of, stats, freq, mode, budget=BUDGET):
    """mode: 'surrogate' (no fallback), 'hybrid' (gap gate), 'unc' (ens_var
    gate), 'random' (random verification, budget-matched to hybrid)."""
    rng = np.random.default_rng(seed)
    pop = [DT.random_design(rng) for _ in range(POP)]

    def surr_eval(ds):
        mu, gap, unc = gap_of([DT.to_sample(d, freq) for d in ds])
        return ([DT.objective(mu[i], freq) for i in range(len(ds))], gap, unc)

    fit, gap, unc = surr_eval(pop)
    px = [vec(d) for d in pop]
    calls = 0
    verified = {}          # key -> true J
    best_ver = (2.0, None)
    log = []
    traj = []

    def verify(ds, reasons, surr_js, gaps):
        nonlocal calls, best_ver
        if not ds:
            return {}
        rs_ = solver.solve_batch(ds)
        out = {}
        for d, r, why, sj, g in zip(ds, rs_, reasons, surr_js, gaps):
            calls += 1
            verified[DT.key(d)] = r["J"]
            out[DT.key(d)] = r["J"]
            useful = r["J"] < best_ver[0]
            if useful:
                best_ver = (r["J"], d)
            log.append(dict(call=calls, reason=why, surr_J=float(sj),
                            true_J=float(r["J"]), gap=float(g),
                            err=float(abs(sj - r["J"])), improved=bool(useful)))
            traj.append((calls, best_ver[0]))
        return out

    if mode != "surrogate":
        # verify the initial surrogate-best few
        order = np.argsort(fit)[:2]
        out0 = verify([pop[i] for i in order], ["init"] * 2,
                      [fit[i] for i in order], [gap[i] for i in order])
        for i in order:                      # declared policy: verified J
            fit[i] = out0.get(DT.key(pop[i]), fit[i])   # replaces fitness

    for gen in range(GENS):
        trials = de_step(px, fit, rng)
        tds = [unvec(t) for t in trials]
        tf, tgap, tunc = surr_eval(tds)
        # selection on surrogate fitness
        for i in range(POP):
            if tf[i] < fit[i]:
                fit[i], gap[i], unc[i] = tf[i], tgap[i], tunc[i]
                px[i], pop[i] = vec(tds[i]), tds[i]
        if mode == "surrogate" or calls >= budget:
            continue
        # verification set for this generation
        cand, why, sj, gg = [], [], [], []
        risk = tgap if mode in ("hybrid", "random") else tunc
        thresh = stats["gap_p95"] if mode in ("hybrid", "random") else None
        for i in range(POP):
            k = DT.key(tds[i])
            if k in verified or len(cand) >= max(1, budget // GENS + 1):
                continue
            promising = tf[i] < best_ver[0]
            risky = (tgap[i] > stats["gap_p95"] if mode != "unc"
                     else tunc[i] > np.quantile(unc, 0.95))
            if mode == "random":
                if rng.random() < 1.5 / POP:
                    cand.append(tds[i]); why.append("random")
                    sj.append(tf[i]); gg.append(tgap[i])
            elif promising or (risky and tf[i] < 1.2 * best_ver[0]):
                cand.append(tds[i])
                why.append("promising" if promising else "risky")
                sj.append(tf[i]); gg.append(tgap[i])
        n_room = budget - calls
        outg = verify(cand[:n_room], why[:n_room], sj[:n_room], gg[:n_room])
        # Declared policy (POLICY FIX, v2b): a verified candidate's TRUE
        # objective permanently replaces its surrogate fitness wherever it
        # sits in the population, so DE selection stops believing phantom
        # surrogate optima. The v2 runs omitted this step (logged defect).
        for i in range(POP):
            k = DT.key(pop[i])
            if k in outg:
                fit[i] = outg[k]
            elif k in verified:
                fit[i] = verified[k]

    if mode == "surrogate":
        i = int(np.argmin(fit))
        r = solver.solve_batch([pop[i]])[0]
        calls += 1
        best_ver = (r["J"], pop[i])
        log.append(dict(call=calls, reason="final", surr_J=float(fit[i]),
                        true_J=float(r["J"]), gap=float(gap[i]),
                        err=float(abs(fit[i] - r["J"])), improved=True))
        traj.append((calls, best_ver[0]))
    return dict(best_J=float(best_ver[0]),
                best_design=best_ver[1], calls=calls, traj=traj, log=log,
                n_surr_evals=POP * (GENS + 1))


def main():
    t0 = time.perf_counter()
    print("training/loading stack...")
    ens, enc, gap_of, stats, freq = train_stack()
    print("stack stats:", stats)
    solver = DT.Solver()
    res = dict(stats=stats, budget=BUDGET, pop=POP, gens=GENS)

    # ---- validation set (independent; no methodology retuning)
    rngv = np.random.default_rng(99)
    val_designs = [DT.random_design(rngv) for _ in range(24)]
    vres = solver.solve_batch(val_designs)
    mu, vgap, vunc = gap_of([DT.to_sample(d, freq) for d in val_designs])
    surr_J = [DT.objective(mu[i], freq) for i in range(24)]
    true_J = [r["J"] for r in vres]
    # response-space error on the openEMS grid via interpolation
    errs = []
    for i, r in enumerate(vres):
        interp = np.empty((len(r["freq"]), 2, 2), complex)
        for a in range(2):
            for b in range(2):
                interp[:, a, b] = np.interp(r["freq"], freq, mu[i][:, a, b].real) \
                    + 1j * np.interp(r["freq"], freq, mu[i][:, a, b].imag)
        errs.append(float(np.abs(interp - r["s"]).mean()))
    res["validation"] = dict(
        surr_J=surr_J, true_J=true_J,
        J_spearman=spearman(surr_J, true_J),
        J_mae=float(np.mean(np.abs(np.array(surr_J) - true_J))),
        resp_err_mean=float(np.mean(errs)),
        gap_spearman_err=spearman(vgap, errs),
        unc_spearman_err=spearman(vunc, errs),
        gap_over_p95_frac=float(np.mean(vgap > stats["gap_p95"])),
        med_gap_z=float((np.median(vgap) - stats["gap_med"]) / stats["gap_iqr"]),
        med_unc_z=float((np.median(vunc) - stats["unc_med"]) / stats["unc_iqr"]))
    print("validation:", json.dumps({k: v for k, v in res["validation"].items()
                                     if not isinstance(v, list)}, default=float))

    # ---- optimization runs
    res["runs"] = {}
    modes = os.environ.get("DESIGN_MODES", "solver_only,hybrid,surrogate,multistart,ablations").split(",")
    queue = []
    for seed in SEEDS:
        if "solver_only" in modes:
            queue.append((f"solver_only_s{seed}", "solver_only", seed))
        if "hybrid" in modes:
            queue.append((f"hybrid_s{seed}", "hybrid", seed))
        if "surrogate" in modes:
            queue.append((f"surrogate_s{seed}", "surrogate", seed))
    for seed in SEEDS:
        if "multistart" in modes:
            queue.append((f"multistart_s{seed}", "multistart", seed))
    if "ablations" in modes:
        queue += [("unc_s0", "unc", 0), ("random_s0", "random", 0)]
    print("queue:", [q[0] for q in queue])
    for name, mode, seed in queue:
        prev = load_run(name)
        if prev is not None:
            res["runs"][name] = prev
            print(f"{name}: restored from checkpoint (best {prev['best_J']:.3f})")
            continue
        t1 = time.perf_counter()
        sv = DT.Solver()
        if mode == "solver_only":
            r = run_solver_only(sv, seed)
            r["calls"] = sv.calls
            r["n_surr_evals"] = 0
        elif mode == "multistart":
            r = run_multistart(sv, seed, gap_of, stats, freq)
        else:
            r = run_search(sv, seed, gap_of, stats, freq, mode)
        r["wall_min"] = round((time.perf_counter() - t1) / 60, 2)
        r["solver_wall_min"] = round(sv.wall / 60, 2)
        r["new_solver_calls"] = sv.calls
        res["runs"][name] = r
        save_run(name, r)
        print(f"{name}: best {r['best_J']:.3f} calls {r['calls']} "
              f"wall {r['wall_min']} min", flush=True)

    # ---- final openEMS verification of every reported best design + QA
    final = DT.Solver()
    qa = {}
    for name, r in res["runs"].items():
        d = r["best_design"]
        if d is None:
            continue
        rr = final.solve_batch([d])[0]
        mu, g, u = gap_of([DT.to_sample(d, freq)])
        qa[name] = dict(design=d, verified_J=rr["J"], reported_J=r["best_J"],
                        surrogate_J=DT.objective(mu[0], freq),
                        retrieval_gap=float(g[0]), ens_var=float(u[0]),
                        max_sv=float(np.load(final._path(d))["max_sv"]),
                        s11_oems=np.abs(rr["s"][:, 0, 0]).tolist(),
                        s21_oems=np.abs(rr["s"][:, 1, 0]).tolist(),
                        freq_oems=rr["freq"].tolist(),
                        s11_surr=np.abs(mu[0][:, 0, 0]).tolist(),
                        s21_surr=np.abs(mu[0][:, 1, 0]).tolist(),
                        freq_surr=freq.tolist())
    res["final_verification"] = qa
    res["final_verification_new_calls"] = final.calls
    res["total_wall_min"] = round((time.perf_counter() - t0) / 60, 1)
    Path(f"results/design_opt_v{DT.TASK_VERSION}{POLICY}_metrics.json").write_text(
        json.dumps(res, indent=1, default=lambda o: o if not isinstance(o, np.ndarray) else o.tolist()))
    print(f"wrote results/design_opt_v{DT.TASK_VERSION}{POLICY}_metrics.json")


if __name__ == "__main__":
    main()
