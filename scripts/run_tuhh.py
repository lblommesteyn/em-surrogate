"""Phases 2-5 of the TUHH real-data evaluation: frozen splits, frozen
baselines, surrogate-quality metrics, and novelty/solver-fallback evaluation.

Prerequisite: scripts/ingest_tuhh.py has written data/processed/tuhh.h5 and
results/tuhh_manifest.csv.

Usage:
    python scripts/run_tuhh.py [--full] [--all-loso-novelty]

All methodology is frozen from milestones 1/2 (configs/baseline.yaml model
configs, novelty scorer set, pseudo-OOD selection protocol) plus the
a-priori choices in configs/tuhh_run.yaml. Nothing here is tuned on any
held-out super-family.

Outputs: results/tuhh_metrics.json, results/tuhh_*.png
"""

import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from emsurr import dataset, metrics, models, novelty, tuhh, uncertainty


def _split_ids(ids, rng, frac=(0.7, 0.15, 0.15)):
    ids = list(ids)
    rng.shuffle(ids)
    n = len(ids)
    a, b = int(frac[0] * n), int((frac[0] + frac[1]) * n)
    return ids[:a], ids[a:b], ids[b:]


def make_tuhh_splits(samples, fam2sf, seed=0):
    """IID, leave-one-super-family-out (loso_*), leave-one-dataset-family-out
    (lodo_*), and a global parameter-extrapolation split if any raw parameter
    name is shared by every family (else recorded as invalid)."""
    rs = np.random.RandomState(seed)
    by_id = {s["sample_id"]: s for s in samples}
    pick = lambda ids: [by_id[i] for i in ids]
    sp = {}
    tr, va, te = _split_ids(list(by_id), rs)
    sp["iid"] = dict(train=pick(tr), val=pick(va), test=pick(te))

    sfs = sorted({fam2sf[s["topology_family"]] for s in samples})
    for sf in sfs:
        out = [s for s in samples if fam2sf[s["topology_family"]] == sf]
        ins = [s for s in samples if fam2sf[s["topology_family"]] != sf]
        if not out or not ins:
            continue
        tr, va, _ = _split_ids([s["sample_id"] for s in ins], rs, (0.85, 0.15, 0.0))
        sp[f"loso_{sf}"] = dict(train=pick(tr), val=pick(va), test=out)

    fams = sorted({s["topology_family"] for s in samples})
    for fam in fams:
        out = [s for s in samples if s["topology_family"] == fam]
        ins = [s for s in samples if s["topology_family"] != fam]
        if not out or not ins:
            continue
        tr, va, _ = _split_ids([s["sample_id"] for s in ins], rs, (0.85, 0.15, 0.0))
        sp[f"lodo_{fam}"] = dict(train=pick(tr), val=pick(va), test=out)
    return sp


def make_extrap_split(samples, schema, seed=0):
    """Valid only if some raw parameter name exists in every family: hold out
    the top 15% of its (per-family z-scored) value. Returns (name, split) or
    (reason, None)."""
    fams = sorted({s["topology_family"] for s in samples})
    per_fam_names = {
        f: {n.split(":", 1)[1] for n in schema if n.startswith(f + ":")} for f in fams
    }
    shared = set.intersection(*per_fam_names.values()) if per_fam_names else set()
    if not shared:
        return "no raw parameter name shared by all families", None
    name = sorted(shared)[0]
    col = {f: schema.index(f"{f}:{name}") for f in fams}
    vals = np.array([s["params"][col[s["topology_family"]]] for s in samples])
    z = np.empty_like(vals)
    for f in fams:
        m = np.array([s["topology_family"] == f for s in samples])
        z[m] = (vals[m] - vals[m].mean()) / (vals[m].std() + 1e-12)
    hi = np.quantile(z, 0.85)
    ins = [s for s, v in zip(samples, z) if v <= hi]
    out = [s for s, v in zip(samples, z) if v > hi]
    rs = np.random.RandomState(seed)
    by_id = {s["sample_id"]: s for s in ins}
    tr, va, _ = _split_ids(list(by_id), rs, (0.85, 0.15, 0.0))
    return f"extrap_{name}", dict(
        train=[by_id[i] for i in tr], val=[by_id[i] for i in va], test=out
    )


def build(name, mc):
    if name == "knn":
        return models.KNNBaseline(k=mc["knn"]["k"])
    c = mc["mlp" if name == "mlp" else "deepsets"]
    cls = models.TorchRegressor if name == "mlp" else models.DeepSetsModel
    return cls(hidden=c["hidden"], depth=c["depth"], epochs=c["epochs"], lr=c["lr"])


def eval_split(d, split_name, mc, results):
    leak = dataset.check_leakage(d["train"], d["test"])
    assert not leak, (split_name, leak[:3])
    true = np.stack([s["s"] for s in d["test"]])
    res = {"n_train": len(d["train"]), "n_val": len(d["val"]), "n_test": len(d["test"])}
    have_tokens = any(len(s["elements"]) for s in d["train"])
    for mname in ("knn", "mlp", "deepsets"):
        if mname == "deepsets" and not have_tokens:
            # real TUHH layouts have no element-token decomposition, so the
            # geometry-aware model has no input; documented, not simulated
            res[mname] = dict(not_applicable="no element tokens on real data")
            continue
        t0 = time.perf_counter()
        model = build(mname, mc).fit(d["train"], d["val"])
        fit_s = time.perf_counter() - t0
        pred, ms_per = models.timed_predict(model, d["test"])
        rep = metrics.full_report(pred, true)
        rep["latency_ms_per_sample"] = round(ms_per, 3)
        rep["fit_seconds"] = round(fit_s, 1)
        rep["per_family"] = metrics.per_family_report(pred, d["test"], true)
        rep["err_vs_freq"] = metrics.error_vs_frequency(pred, true).tolist()
        res[mname] = rep
        print(
            f"{split_name}/{mname}: cMAE={rep['complex_mae']:.4f} "
            f"dB-MAE={rep['mag_db_mae']:.2f} phase={rep['phase_mae_deg']:.1f}deg "
            f"({ms_per:.2f} ms/sample)"
        )
    results[split_name] = res


def make_scorers(tr, mlp, ens, k, alpha, tag=""):
    knn_in = novelty.KNNInputNovelty(k=k).fit(tr)
    maha = novelty.MahalanobisEmbedding().fit(tr, mlp)
    knn_e = novelty.KNNEmbedding(k=k).fit(tr, mlp)
    ens_s = novelty.EnsembleVariance(ens)
    comb = novelty.CombinedScore(knn_in, ens_s, alpha=alpha)
    scs = [knn_in, maha, knn_e, ens_s, comb]
    if tag:
        for sc in scs:
            sc.name = sc.name + tag
    return scs


def eval_scorers(scorers, id_s, ood_s, errors_pool, budgets):
    pool = id_s + ood_s
    out = {}
    for sc in scorers:
        t0 = time.perf_counter()
        s_all = sc.score(pool)  # single call: rank scores need one pool
        ms = (time.perf_counter() - t0) / len(pool) * 1e3
        s_id, s_ood = s_all[: len(id_s)], s_all[len(id_s):]
        rc = novelty.risk_coverage(s_all, errors_pool, budgets)
        out[sc.name] = dict(
            auroc=novelty.auroc(s_id, s_ood),
            auprc=novelty.auprc(s_id, s_ood),
            spearman_err=uncertainty.spearman(s_all, errors_pool),
            risk_coverage=rc,
            oracle_recovery=novelty.oracle_recovery(rc),
            latency_ms=round(ms, 3),
        )
    return out


def novelty_study(d, holdout, fam2sf, cfg, mc, uc, ncfg):
    """Milestone-2 protocol on one loso split: pseudo-OOD selection inside
    the training super-families, then final evaluation on the held-out
    super-family. Also scores with the synthetic-frozen hyperparameters."""
    budgets = tuple(ncfg["budgets"])
    train, id_pool, ood_pool = d["train"], d["val"], d["test"]
    train_sfs = sorted({fam2sf[s["topology_family"]] for s in train})
    pseudo = train_sfs[:2]  # a-priori rule: alphabetically first two
    if len(train_sfs) < 3:
        pseudo = train_sfs[:1]
    sel_all = [s for s in train if fam2sf[s["topology_family"]] not in pseudo]
    pseudo_ood = [s for s in train if fam2sf[s["topology_family"]] in pseudo]
    rs = np.random.RandomState(1)
    ids = [s["sample_id"] for s in sel_all]
    rs.shuffle(ids)
    cut = int(0.85 * len(ids))
    by_id = {s["sample_id"]: s for s in sel_all}
    sel_train = [by_id[i] for i in ids[:cut]]
    sel_val = [by_id[i] for i in ids[cut:]]

    fit_mlp = lambda tr, va: models.TorchRegressor(
        hidden=mc["mlp"]["hidden"], depth=mc["mlp"]["depth"],
        epochs=mc["mlp"]["epochs"], lr=mc["mlp"]["lr"]).fit(tr, va)
    fit_ens = lambda tr, va: uncertainty.Ensemble(
        n_members=uc["n_members"], hidden=mc["mlp"]["hidden"],
        depth=mc["mlp"]["depth"], epochs=mc["mlp"]["epochs"],
        lr=mc["mlp"]["lr"]).fit(tr, va)

    print(f"== {holdout}: pseudo-OOD selection on {pseudo} ==")
    mlp_s, ens_s = fit_mlp(sel_train, sel_val), fit_ens(sel_train, sel_val)
    pool = sel_val + pseudo_ood
    pred, _ = ens_s.predict_with_uncertainty(pool)
    errs = uncertainty.per_sample_error(pred, np.stack([s["s"] for s in pool]))
    selection = {}
    for k in ncfg["k_grid"]:
        sc = novelty.KNNInputNovelty(k=k).fit(sel_train)
        selection[f"knn_input_k{k}"] = novelty.auroc(sc.score(sel_val), sc.score(pseudo_ood))
    best_k = max(ncfg["k_grid"], key=lambda k: selection[f"knn_input_k{k}"])
    ens_score = novelty.EnsembleVariance(ens_s)
    knn_best = novelty.KNNInputNovelty(k=best_k).fit(sel_train)
    for alpha in ncfg["alpha_grid"]:
        comb = novelty.CombinedScore(knn_best, ens_score, alpha=alpha)
        rc = novelty.risk_coverage(comb.score(pool), errs, (0.2,))
        selection[f"combined_a{alpha}"] = novelty.oracle_recovery(rc)[0.2]
    best_alpha = max(ncfg["alpha_grid"], key=lambda a: selection[f"combined_a{a}"])
    print(f"selected k={best_k}, alpha={best_alpha}")

    print(f"== {holdout}: final evaluation ==")
    mlp_f, ens_f = fit_mlp(train, id_pool), fit_ens(train, id_pool)
    final_pool = id_pool + ood_pool
    pred_f, _ = ens_f.predict_with_uncertainty(final_pool)
    true_f = np.stack([s["s"] for s in final_pool])
    errors_f = uncertainty.per_sample_error(pred_f, true_f)
    sf = ncfg["synth_frozen"]
    scorers = make_scorers(train, mlp_f, ens_f, best_k, best_alpha)
    scorers += [s for s in make_scorers(train, mlp_f, ens_f, sf["k"], sf["alpha"],
                                        tag="_synthfrozen")
                if s.name.startswith(("knn_input", "combined"))]
    res = eval_scorers(scorers, id_pool, ood_pool, errors_f, budgets)

    # 25%-OOD deployment mix
    rs2 = np.random.RandomState(2)
    n_mix = max(1, min(len(ood_pool), round(len(id_pool) / 3)))
    ood_sub = [ood_pool[i] for i in rs2.choice(len(ood_pool), n_mix, replace=False)]
    pred_m, _ = ens_f.predict_with_uncertainty(id_pool + ood_sub)
    errors_m = uncertainty.per_sample_error(
        pred_m, np.stack([s["s"] for s in id_pool + ood_sub]))
    res_mix = eval_scorers(scorers, id_pool, ood_sub, errors_m, budgets)

    fams = np.array([fam2sf[s["topology_family"]] for s in final_pool])
    per_sf_scores = {}
    for sc in scorers:
        s_all = sc.score(final_pool)
        per_sf_scores[sc.name] = {
            g: dict(median=float(np.median(s_all[fams == g])),
                    p90=float(np.quantile(s_all[fams == g], 0.9)))
            for g in sorted(set(fams))
        }
    for tag, r in (("natural", res), ("mix25", res_mix)):
        for name, rr in r.items():
            print(f"{holdout}/{tag}/{name}: AUROC={rr['auroc']:.3f} "
                  f"spearman={rr['spearman_err']:.3f} "
                  f"recovery@20%={rr['oracle_recovery'][0.2]:.2f} "
                  f"cat@20%={rr['risk_coverage'][2]['catastrophic_caught']:.2f}")
    return dict(
        selection=dict(grid=selection, best_k=best_k, best_alpha=best_alpha,
                       pseudo_ood_super_families=pseudo),
        natural=res, deployment_mix=res_mix, per_super_family_scores=per_sf_scores,
        pool=dict(n_id=len(id_pool), n_ood=len(ood_pool),
                  ensemble_mae=float(metrics.complex_mae(pred_f, true_f))),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true", help="no per-family subsample cap")
    ap.add_argument("--all-loso-novelty", action="store_true",
                    help="novelty study on every loso split, not just primary")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path("configs/tuhh_run.yaml").read_text())
    base = yaml.safe_load(Path("configs/baseline.yaml").read_text())
    mc, uc, ncfg = base["models"], base["uncertainty"], cfg["novelty"]
    fam2sf = tuhh.family_map()
    ingest = json.loads(Path("results/tuhh_ingest.json").read_text())
    schema = ingest["meta"]["schema"]

    samples = dataset.load_h5(cfg["data"]["h5"])
    seed = cfg["splits"]["seed"]
    cap = None if args.full else cfg["splits"]["max_per_family"]
    dropped = {}
    if cap:
        rs = np.random.RandomState(seed)
        kept = []
        fams = sorted({s["topology_family"] for s in samples})
        for f in fams:
            fs = [s for s in samples if s["topology_family"] == f]
            if len(fs) > cap:
                idx = rs.choice(len(fs), cap, replace=False)
                dropped[f] = len(fs) - cap
                fs = [fs[i] for i in idx]
            kept.append(fs)
        samples = [s for fs in kept for s in fs]
        if dropped:
            print(f"subsample cap {cap}/family, dropped: {dropped}")

    sanity = dataset.sanity_check(samples, rtol_passivity=0.1)
    sp = make_tuhh_splits(samples, fam2sf, seed=seed)
    ex_name, ex_split = make_extrap_split(samples, schema, seed=seed)
    if ex_split is not None:
        sp[ex_name] = ex_split
    else:
        print(f"parameter-extrapolation split invalid: {ex_name}")

    results = {}
    for split_name, d in sp.items():
        eval_split(d, split_name, mc, results)

    primary = cfg["splits"]["primary_loso"]
    loso_names = [n for n in sp if n.startswith("loso_")]
    nov_targets = loso_names if args.all_loso_novelty else [
        n for n in loso_names if n.removeprefix("loso_") in primary]
    nov = {n: novelty_study(sp[n], n, fam2sf, cfg, mc, uc, ncfg) for n in nov_targets}

    # summary plot: IID vs each loso holdout, per model
    fig, ax = plt.subplots(figsize=(9, 4.5))
    names = ["iid"] + loso_names
    for m in ("knn", "mlp", "deepsets"):
        if any("complex_mae" not in results[n].get(m, {}) for n in names):
            continue
        ax.plot(range(len(names)), [results[n][m]["complex_mae"] for n in names],
                "o-", label=m)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels([n.removeprefix("loso_") for n in names], rotation=30, ha="right")
    ax.set_ylabel("complex MAE")
    ax.set_title("TUHH: IID vs leave-one-super-family-out")
    ax.legend()
    fig.tight_layout()
    fig.savefig("results/tuhh_loso_mae.png", dpi=120)
    plt.close(fig)

    out = dict(
        config=cfg, subsample_dropped=dropped,
        sanity={k: (v if isinstance(v, int) else len(v)) for k, v in sanity.items()},
        extrap_split=(ex_name if ex_split is not None
                      else dict(invalid_reason=ex_name)),
        splits=results, novelty=nov,
    )
    Path("results/tuhh_metrics.json").write_text(json.dumps(out, indent=2))
    print("wrote results/tuhh_metrics.json")


if __name__ == "__main__":
    main()
