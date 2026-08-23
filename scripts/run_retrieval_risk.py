"""Retrieval-gap integration + vocabulary-coverage experiment.

Stage 1: choose the retrieval variant (k=1 / mean-3 / distance-weighted-3)
on the milestone-2 SELECTION stage (models trained without stub_open +
stepped; those two families are the pseudo-OOD) - strictly train-side.

Stage 2: add retrieval_gap as the 11th signal to the frozen within-OOD
ridge, refit on the same train-side pseudo pools, freeze, evaluate on the
frozen benchmarks (synth OOD, planar OOD, SQChip folds, openEMS).

Stage 3: coverage - extend the token vocabulary with EL_SERIES_C, add the
analytically solved `gapped_line` training family (same generator physics),
retrain the SAME-size encoder, and compare old vs extended vocabulary on
openEMS per family.

Writes results/retrieval_risk_metrics.json and results/retrieval_qa.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from emsurr import dataset, planar_windings, splits, sqchip, synth_ext, uncertainty
from emsurr.external_tab import FROZEN, TabEnsemble
from emsurr.novelty import auroc, oracle_recovery, risk_coverage
from emsurr.openems_load import load_openems
from emsurr.risk_cal import budget_to_catch, spearman
from emsurr.risk_signals import SIGNALS, fit_rank_ridge, predict_rank_ridge, rank_features
from emsurr.topo_rep import TopoEmbedding
from riskcal_extract_synth import load_final

RC = Path("results/riskcal")
BUDGETS = (0.05, 0.1, 0.2, 0.3, 0.5)
EK = dict(n_members=FROZEN["n_members"], hidden=FROZEN["hidden"],
          depth=FROZEN["depth"], epochs=FROZEN["epochs"], lr=FROZEN["lr"])


def retrieval_gap(pred_flat, e_q, e_ref, resp_ref_flat, mode):
    d = np.linalg.norm(e_q[:, None] - e_ref[None], axis=-1)
    if mode == "k1":
        nn = d.argmin(1)
        tgt = resp_ref_flat[nn]
    else:
        idx = np.argsort(d, 1)[:, :3]
        nb = resp_ref_flat[idx]                       # (N, 3, D)
        if mode == "mean3":
            tgt = nb.mean(1)
        else:                                          # w3
            w = 1.0 / (np.take_along_axis(d, idx, 1) + 1e-9)
            tgt = (nb * w[..., None]).sum(1) / w.sum(1)[:, None]
    return np.abs(pred_flat - tgt).mean(1)


def sflat(S):
    return np.concatenate([S.real.reshape(len(S), -1), S.imag.reshape(len(S), -1)], 1)


def eval_rank(score, err, is_ood):
    mo = is_ood.astype(bool)
    rc = risk_coverage(score, err, BUDGETS)
    orec = oracle_recovery(rc)
    return dict(spearman_within_ood=spearman(score[mo], err[mo]),
                oracle_recovery={str(b): orec[b] for b in BUDGETS},
                regret_vs_oracle={str(b): 1 - orec[b] for b in BUDGETS},
                budget_catch90=budget_to_catch(score, err, 0.90))


def load_sig(dep):
    z = np.load(RC / f"sig2_{dep}.npz", allow_pickle=True)
    pools = {}
    for key in z.files:
        p, k = key.split("/", 1)
        pools.setdefault(p, {})[k] = z[key]
    return pools


def main():
    res = {}
    cfg = yaml.safe_load(Path("configs/baseline.yaml").read_text())
    samples = dataset.load_h5(cfg["data"]["path"])
    sp = splits.make_splits(samples, seed=cfg["splits"]["seed"],
                            ood_families=tuple(cfg["splits"]["ood_families"]))
    d = sp["ood_topology"]
    train, id_pool, ood_pool = d["train"], d["val"], d["test"]

    # ---------------- stage 1: k-variant selection on the selection stage
    PSEUDO = ("stub_open", "stepped")
    sel_all = [s for s in train if s["topology_family"] not in PSEUDO]
    pseudo_s = [s for s in train if s["topology_family"] in PSEUDO]
    rs = np.random.RandomState(1)
    ids = [s["sample_id"] for s in sel_all]
    rs.shuffle(ids)
    by_id = {s["sample_id"]: s for s in sel_all}
    cut = int(0.85 * len(ids))
    sel_train = [by_id[i] for i in ids[:cut]]
    sel_val = [by_id[i] for i in ids[cut:]]
    _, ens_sel = load_final("selection", sel_train, sel_val)
    enc_sel = TopoEmbedding(ordered=True, objective="resp").fit(sel_train, sel_val)
    pool = sel_val + pseudo_s
    pred, _ = ens_sel.predict_with_uncertainty(pool)
    S_true = np.stack([s["s"] for s in pool])
    err = np.abs(pred - S_true).mean(axis=(1, 2, 3))
    is_ood = np.array([0] * len(sel_val) + [1] * len(pseudo_s))
    e_ref = enc_sel.embed(sel_train)
    resp_ref = sflat(np.stack([s["s"] for s in sel_train]))
    pf = sflat(pred)
    e_q = enc_sel.embed(pool)
    ksel = {}
    for mode in ("k1", "mean3", "w3"):
        g = retrieval_gap(pf, e_q, e_ref, resp_ref, mode)
        ksel[mode] = spearman(g[is_ood == 1], err[is_ood == 1])
    best_mode = max(ksel, key=ksel.get)
    res["k_selection"] = dict(table=ksel, chosen=best_mode)
    print("k-selection (train-side):", ksel, "->", best_mode)

    # ---------------- stage 2: synth ridge integration
    # reproduce sig2 pseudo alignment and per-fold retrieval gaps
    rs2 = np.random.RandomState(1)
    ids2 = [s["sample_id"] for s in train]
    rs2.shuffle(ids2)
    by2 = {s["sample_id"]: s for s in train}
    cut2 = int(0.85 * len(ids2))
    strain = [by2[i] for i in ids2[:cut2]]
    sval = [by2[i] for i in ids2[cut2:]]
    fams = sorted({s["topology_family"] for s in train})
    sig = load_sig("synth")
    gap_parts = []
    for f in fams:
        tr_f = [s for s in strain if s["topology_family"] != f]
        va_f = [s for s in sval if s["topology_family"] != f]
        held = [s for s in strain + sval if s["topology_family"] == f]
        print("fold gap", f)
        enc_f = TopoEmbedding(ordered=True, objective="resp",
                              epochs=100).fit(tr_f, va_f)
        ens_f = uncertainty.Ensemble(n_members=5, hidden=256, depth=3, epochs=200,
                                     lr=1e-3).fit(tr_f, va_f)
        poolf = va_f + held
        predf, _ = ens_f.predict_with_uncertainty(poolf)
        gap_parts.append(retrieval_gap(
            sflat(predf), enc_f.embed(poolf), enc_f.embed(tr_f),
            sflat(np.stack([s["s"] for s in tr_f])), best_mode))
    gap_pseudo = np.concatenate(gap_parts)
    ps = sig["pseudo"]
    assert len(gap_pseudo) == len(ps["err"]), (len(gap_pseudo), len(ps["err"]))

    enc_full = TopoEmbedding(ordered=True, objective="resp").fit(strain, sval)
    _, ens_F = load_final("final", train, id_pool)
    evpool = id_pool + ood_pool
    predE, _ = ens_F.predict_with_uncertainty(evpool)
    gap_eval = retrieval_gap(sflat(predE), enc_full.embed(evpool),
                             enc_full.embed(train),
                             sflat(np.stack([s["s"] for s in train])), best_mode)

    def ridge_block(sig_pools, gap_ps, gap_ev, evname):
        ps = sig_pools["pseudo"]
        ev = sig_pools[evname]
        mo = ps["is_ood"].astype(bool)
        Fp = rank_features(ps, SIGNALS)
        Fp11 = np.c_[Fp, np.argsort(np.argsort(gap_ps)) / max(len(gap_ps) - 1, 1)]
        Fe = rank_features(ev, SIGNALS)
        Fe11 = np.c_[Fe, np.argsort(np.argsort(gap_ev)) / max(len(gap_ev) - 1, 1)]
        w10 = fit_rank_ridge(Fp[mo], ps["err"][mo])
        w11 = fit_rank_ridge(Fp11[mo], ps["err"][mo])
        out = {}
        out["ens_var"] = eval_rank(np.asarray(ev["ens_var"], float), ev["err"], ev["is_ood"])
        out["ridge10"] = eval_rank(predict_rank_ridge(w10, Fe), ev["err"], ev["is_ood"])
        out["gap_only"] = eval_rank(gap_ev, ev["err"], ev["is_ood"])
        out["ridge11"] = eval_rank(predict_rank_ridge(w11, Fe11), ev["err"], ev["is_ood"])
        out["_w11_gap_weight"] = float(w11[-2])
        return out

    res["synth"] = ridge_block(sig, gap_pseudo, gap_eval, "eval_synth")

    # per-family within-OOD spearman for gap and ridge11 on synth
    fams_e = sig["eval_synth"]["family"].astype(str)
    mo_e = sig["eval_synth"]["is_ood"].astype(bool)
    res["synth"]["_per_family"] = {
        f: dict(gap=spearman(gap_eval[mo_e & (fams_e == f)],
                             sig["eval_synth"]["err"][mo_e & (fams_e == f)]))
        for f in sorted(set(fams_e[mo_e]))}

    # ---------------- tabular deployments: retrieval gap in surrogate embedding
    def tab_gap(ens, xtr, ytr, x, mode):
        m0 = ens.members[0]
        pred = np.stack([m.predict(x) for m in ens.members]).mean(0)
        return retrieval_gap(pred, m0.embed(x), m0.embed(xtr), ytr, mode)

    # planar
    rng = np.random.default_rng(0)
    data = planar_windings.load()
    xc, yc = data["CORE"]["x"], np.log10(data["CORE"]["y"])[:, None]
    perm = rng.permutation(len(xc))
    n = len(xc)
    tr, va, te = perm[: int(0.8 * n)], perm[int(0.8 * n) : int(0.9 * n)], perm[int(0.9 * n) :]
    xtr, ytr, xva, yva = xc[tr], yc[tr], xc[va], yc[va]
    xo, yo = data["OOD"]["x"], np.log10(data["OOD"]["y"])[:, None]
    qlo, qhi = np.quantile(xtr, 0.10, 0), np.quantile(xtr, 0.90, 0)
    ctr = ((xtr >= qlo) & (xtr <= qhi)).all(1)
    cva = ((xva >= qlo) & (xva <= qhi)).all(1)
    print("planar ensembles")
    ens_c = TabEnsemble(**EK).fit(xtr[ctr], ytr[ctr], xva[cva], yva[cva])
    ens_f = TabEnsemble(**EK).fit(xtr, ytr, xva, yva)
    sigP = load_sig("planar")
    xp_pool = np.concatenate([xva[cva], np.concatenate([xtr[~ctr], xva[~cva]])])
    gap_ps_P = tab_gap(ens_c, xtr[ctr], ytr[ctr], xp_pool, best_mode)
    xe_pool = np.concatenate([xc[te], xo])
    gap_ev_P = tab_gap(ens_f, xtr, ytr, xe_pool, best_mode)
    res["planar"] = ridge_block(sigP, gap_ps_P, gap_ev_P, "eval_ood")

    # sqchip folds
    recs = sqchip.load_records()
    y = np.stack([r["y"] for r in recs])
    xp, _ = sqchip.param_matrix(recs)
    fam = np.array([r["family"] for r in recs])
    from collections import Counter
    for hold in ["batch300", "batchnear50", "sqchip"]:
        print("sqchip", hold)
        rest = np.where(fam != hold)[0]
        ho = np.where(fam == hold)[0]
        rngh = np.random.default_rng(1)
        rp = rngh.permutation(rest)
        trh, vah = rp[: int(0.9 * len(rp))], rp[int(0.9 * len(rp)) :]
        pf_ = Counter(fam[trh]).most_common(1)[0][0]
        ptr = trh[fam[trh] != pf_]
        pva = vah[fam[vah] != pf_]
        pho = np.concatenate([trh[fam[trh] == pf_], vah[fam[vah] == pf_]])
        ens_ch = TabEnsemble(**EK).fit(xp[ptr], y[ptr], xp[pva], y[pva])
        ens_fh = TabEnsemble(**EK).fit(xp[trh], y[trh], xp[vah], y[vah])
        sigS = load_sig(f"sqchip_{hold}")
        gap_ps_S = tab_gap(ens_ch, xp[ptr], y[ptr], xp[np.concatenate([pva, pho])], best_mode)
        gap_ev_S = tab_gap(ens_fh, xp[trh], y[trh], xp[np.concatenate([vah, ho])], best_mode)
        res[f"sqchip_{hold}"] = ridge_block(sigS, gap_ps_S, gap_ev_S, "eval")

    # ---------------- stage 3: coverage on openEMS
    oems = load_openems()
    z0 = np.load(RC / "synth.npz", allow_pickle=True)
    err_oe = np.asarray(z0["eval_openems/err"], float)[len(id_pool):]
    fam_oe = np.array([s["topology_family"] for s in oems])
    predO, _ = ens_F.predict_with_uncertainty(
        [{**s, "freq": id_pool[0]["freq"]} for s in oems])
    # interp preds onto oems grid for gap computation in oems response space?
    # keep gap in the surrogate's own 256-grid space vs train responses.
    pO = sflat(predO)

    gapped = synth_ext.generate(400, seed=7)
    strain_ext = strain + gapped[:340]
    sval_ext = sval + gapped[340:]
    enc_ext = TopoEmbedding(ordered=True, objective="resp", n_types=6).fit(
        strain_ext, sval_ext)

    def oems_tokens(extended):
        out = []
        for s in oems:
            fam_ = s["topology_family"].removeprefix("oems_")
            if extended and fam_ == "gap":
                names = None  # rebuild tokens from stored params
            out.append(s)
        return out

    # extended tokens for the gap family
    import h5py
    oems_ext = []
    with h5py.File("results/openems_families.h5", "r") as h:
        raw = {sid: dict(names=h[sid]["param_names"][()].decode().split(","),
                         vals=h[sid]["params"][:]) for sid in h}
    for s in oems:
        fam_ = s["topology_family"].removeprefix("oems_")
        if fam_ == "gap":
            p = dict(zip(raw[s["sample_id"]]["names"], raw[s["sample_id"]]["vals"]))
            s2 = {**s, "elements": synth_ext.tokenize_oems_gap(p, 254e-6)}
            oems_ext.append(s2)
        else:
            oems_ext.append(s)

    train_ext_ref = strain_ext + sval_ext
    resp_ref_old = sflat(np.stack([s["s"] for s in train]))
    resp_ref_ext = None
    # extended reference responses are on mixed grids? gapped_line uses the
    # same 256-pt grid as synth -> consistent.
    resp_ref_ext = sflat(np.stack([s["s"] for s in train_ext_ref]))

    gap_old = retrieval_gap(pO, enc_full.embed(oems), enc_full.embed(train),
                            resp_ref_old, best_mode)
    gap_new = retrieval_gap(pO, enc_ext.embed(oems_ext), enc_ext.embed(train_ext_ref),
                            resp_ref_ext, best_mode)
    cov = {}
    for f in sorted(set(fam_oe)):
        m = fam_oe == f
        cov[f] = dict(n=int(m.sum()),
                      gap_old_mean=float(gap_old[m].mean()),
                      gap_new_mean=float(gap_new[m].mean()),
                      err_mean=float(err_oe[m].mean()),
                      sp_old=spearman(gap_old[m], err_oe[m]) if m.sum() >= 5 else None,
                      sp_new=spearman(gap_new[m], err_oe[m]) if m.sum() >= 5 else None)
    cov["all"] = dict(sp_old=spearman(gap_old, err_oe), sp_new=spearman(gap_new, err_oe))
    res["openems_coverage"] = cov
    print("openEMS coverage:", json.dumps(cov["all"]))

    # ---------------- retrieval QA dump
    e_ood = np.asarray(sig["eval_synth"]["err"], float)[mo_e]
    g_ood = gap_eval[mo_e]
    rq, order = [], np.argsort(-g_ood)
    ood_list = [s for s in ood_pool]
    e_ref_full = enc_full.embed(train)
    eq = enc_full.embed(ood_pool)
    nn = np.linalg.norm(eq[:, None] - e_ref_full[None], axis=-1).argmin(1)
    S_tr = np.stack([s["s"] for s in train])
    S_ood = np.stack([s["s"] for s in ood_pool])
    pick = list(order[:3]) + list(order[-3:]) + list(
        np.argsort(np.abs(np.argsort(np.argsort(g_ood)) - np.argsort(np.argsort(e_ood))))[-3:])
    for i in map(int, pick):
        rq.append(dict(
            query=ood_list[i]["sample_id"], family=str(ood_list[i]["topology_family"]),
            retrieved=train[nn[i]]["sample_id"],
            retrieved_family=str(train[nn[i]]["topology_family"]),
            resp_dist_to_retrieved=float(np.abs(S_ood[i] - S_tr[nn[i]]).mean()),
            retrieval_gap=float(g_ood[i]), true_err=float(e_ood[i])))
    Path("results/retrieval_qa.json").write_text(json.dumps(rq, indent=1))

    Path("results/retrieval_risk_metrics.json").write_text(json.dumps(res, indent=1))
    for dep in ("synth", "planar", "sqchip_batch300", "sqchip_batchnear50", "sqchip_sqchip"):
        r = res[dep]
        print(f"[{dep}] " + " ".join(
            f"{m}:sp={r[m]['spearman_within_ood']:+.2f}/orec20={r[m]['oracle_recovery']['0.2']:.2f}"
            for m in ("ens_var", "ridge10", "gap_only", "ridge11")))
    print("wrote results/retrieval_risk_metrics.json")


if __name__ == "__main__":
    main()
