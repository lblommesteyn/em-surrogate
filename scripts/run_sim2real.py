"""Sim-to-real calibration on the 55 planar-winding PCB measurements.

The FEA surrogate is FROZEN: the ensemble is rebuilt with the exact frozen
recipe/seeds (bit-identical training pipeline to the risk-cal extraction);
its weights are never updated here. All correction happens post-hoc on its
log10(L) predictions.

Protocol: for n_cal in {1,2,4,8,16,24,32}, 200 seeded splits of the 55
boards into calibration / untouched measured test. Corrections are fitted on
calibration boards only. Residuals live in log10(L) space (additive there =
multiplicative on L).

Correction methods (deliberately tiny):
  frozen        no correction
  additive_L    subtract mean residual in raw L
  mult          subtract mean log-residual (global multiplicative on L)
  linear        ridge (lam=1) on 9 standardized geometry features -> residual
  ridge_aug     ridge (lam=10) on features + surrogate output + ensemble std
  nearest       subtract residual of nearest calibration board (standardized
                geometry, k = min(3, n_cal) average)

Risk scores for measured test boards (no unmeasured target error used):
  d_cal      distance to nearest calibration board
  pred_res   |linear-predicted residual|
  ens_var    frozen ensemble disagreement
  novelty    input-kNN distance to FEA train set

Active selection (no unmeasured target errors): random, farthest-point
geometry coverage, highest ensemble uncertainty, highest novelty,
diversity+uncertainty greedy. Writes results/sim2real_metrics.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from emsurr import planar_windings
from emsurr.external_tab import FROZEN, TabEnsemble, TabKNNInput
from emsurr.models import Normalizer
from emsurr.risk_cal import spearman

EK = dict(n_members=FROZEN["n_members"], hidden=FROZEN["hidden"],
          depth=FROZEN["depth"], epochs=FROZEN["epochs"], lr=FROZEN["lr"])
N_CALS = (1, 2, 4, 8, 16, 24, 32)
N_SEEDS = 200
SEL_SEEDS = 50


def rel_err_L(pred_log, true_log):
    return np.abs(10 ** pred_log / 10 ** true_log - 1.0)


def fit_corrections(xc, rc_log, Lc_res, mu_c, unc_c, norm):
    """Return dict name -> fn(x, mu_log, unc) -> corrected log10 prediction."""
    fns = {"frozen": lambda x, m, u: m}
    dL = float(np.mean(Lc_res))                      # mean residual in raw L
    fns["additive_L"] = lambda x, m, u: np.log10(np.clip(10 ** m + dL, 1e-12, None))
    dlog = float(np.mean(rc_log))
    fns["mult"] = lambda x, m, u: m + dlog
    n = len(xc)
    if n >= 3:
        A = np.c_[norm(xc), np.ones(n)]
        lam = 1.0
        w = np.linalg.solve(A.T @ A + lam * np.eye(A.shape[1]), A.T @ rc_log)
        fns["linear"] = lambda x, m, u: m + np.c_[norm(x), np.ones(len(x))] @ w
        A2 = np.c_[norm(xc), mu_c, unc_c, np.ones(n)]
        w2 = np.linalg.solve(A2.T @ A2 + 10.0 * np.eye(A2.shape[1]), A2.T @ rc_log)
        fns["ridge_aug"] = lambda x, m, u: m + np.c_[norm(x), m, u, np.ones(len(x))] @ w2
    xcn = norm(xc)
    k = min(3, n)

    def nearest(x, m, u):
        d = np.linalg.norm(norm(x)[:, None] - xcn[None], axis=-1)
        idx = np.argsort(d, 1)[:, :k]
        return m + rc_log[idx].mean(1)

    fns["nearest"] = nearest
    return fns


def select(method, x_pool_n, unc, nov, n, rng):
    if method == "random":
        return rng.choice(len(x_pool_n), n, replace=False)
    if method == "uncertainty":
        return np.argsort(-unc)[:n]
    if method == "novelty":
        return np.argsort(-nov)[:n]
    # greedy farthest-point (optionally uncertainty-weighted)
    w = (np.argsort(np.argsort(unc)) / max(len(unc) - 1, 1) + 0.5
         if method == "div_unc" else np.ones(len(x_pool_n)))
    start = int(np.argmax(unc)) if method == "div_unc" else int(rng.integers(len(x_pool_n)))
    chosen = [start]
    d = np.linalg.norm(x_pool_n - x_pool_n[start], axis=1)
    while len(chosen) < n:
        nxt = int(np.argmax(d * w))
        chosen.append(nxt)
        d = np.minimum(d, np.linalg.norm(x_pool_n - x_pool_n[nxt], axis=1))
        d[chosen] = -1
    return np.array(chosen[:n])


def main():
    rng0 = np.random.default_rng(0)
    data = planar_windings.load()
    xc_all, yc = data["CORE"]["x"], np.log10(data["CORE"]["y"])[:, None]
    perm = rng0.permutation(len(xc_all))
    n = len(xc_all)
    tr, va = perm[: int(0.8 * n)], perm[int(0.8 * n) : int(0.9 * n)]
    xtr, ytr, xva, yva = xc_all[tr], yc[tr], xc_all[va], yc[va]
    xm = data["MEAS"]["x"]
    ym = np.log10(data["MEAS"]["y"])

    print("rebuilding frozen planar ensemble (recipe-identical)...")
    ens = TabEnsemble(**EK).fit(xtr, ytr, xva, yva)
    mu2, unc = ens.predict_with_uncertainty(xm)
    mu = mu2[:, 0]
    nov = TabKNNInput(k=FROZEN["knn_k"]).fit(xtr).score(xm)
    norm = Normalizer().fit(xtr)
    res_log = ym - mu                       # log-space residual (truth - pred)
    res_L = 10 ** ym - 10 ** mu
    rel0 = rel_err_L(mu, ym)
    print(f"zero-shot MEAS: mean rel {rel0.mean():.4f} median {np.median(rel0):.4f}")
    print(f"log-residual: mean {res_log.mean():+.4f} sd {res_log.std():.4f} "
          f"(pure global bias would leave sd unchanged)")

    # residual structure: correlation with features (all 55, descriptive)
    feat_corr = {planar_windings.FEATURES[j]: spearman(xm[:, j], res_log)
                 for j in range(xm.shape[1])}

    out = dict(zero_shot=dict(rel_mean=float(rel0.mean()), rel_median=float(np.median(rel0)),
                              rel_p95=float(np.quantile(rel0, 0.95)),
                              log_res_mean=float(res_log.mean()),
                              log_res_sd=float(res_log.std())),
               residual_feature_spearman=feat_corr, curves={}, risk={}, active={})

    xm_n = norm(xm)
    methods = ["frozen", "additive_L", "mult", "linear", "ridge_aug", "nearest"]

    # ---- 2. correction curves over n_cal (random splits)
    for ncal in N_CALS:
        agg = {m: [] for m in methods}
        risk_sp = {"d_cal": [], "pred_res": [], "ens_var": [], "novelty": []}
        cat_cap = {"d_cal": [], "pred_res": [], "ens_var": [], "novelty": []}
        for s in range(N_SEEDS):
            rng = np.random.default_rng(1000 + s)
            idx = rng.permutation(55)
            cal, te = idx[:ncal], idx[ncal:]
            fns = fit_corrections(xm[cal], res_log[cal], res_L[cal],
                                  mu[cal], unc[cal], norm)
            for m in methods:
                if m not in fns:
                    continue
                pred = fns[m](xm[te], mu[te], unc[te])
                agg[m].append(rel_err_L(pred, ym[te]).mean())
            # ---- 3. risk scores on test boards (corrected by 'mult')
            err_te = rel_err_L(fns["mult"](xm[te], mu[te], unc[te]), ym[te])
            d_cal = np.linalg.norm(xm_n[te][:, None] - xm_n[cal][None], axis=-1).min(1)
            scores = dict(d_cal=d_cal, ens_var=unc[te], novelty=nov[te])
            if "linear" in fns:
                scores["pred_res"] = np.abs(fns["linear"](xm[te], mu[te], unc[te]) - mu[te]
                                            - np.mean(res_log[cal]))
            cat = err_te >= np.quantile(err_te, 0.9)
            for k2, sc in scores.items():
                risk_sp[k2].append(spearman(sc, err_te))
                top = np.argsort(-sc)[: max(1, int(0.2 * len(te)))]
                cat_cap[k2].append(cat[top].sum() / max(cat.sum(), 1))
        out["curves"][ncal] = {m: dict(mean=float(np.mean(v)),
                                       ci=[float(np.quantile(v, 0.025)),
                                           float(np.quantile(v, 0.975))])
                               for m, v in agg.items() if v}
        out["risk"][ncal] = {k2: dict(spearman=float(np.mean(v)),
                                      cat_capture20=float(np.mean(cat_cap[k2])))
                             for k2, v in risk_sp.items() if v}
        best = min(out["curves"][ncal], key=lambda m: out["curves"][ncal][m]["mean"])
        print(f"n_cal={ncal:2d} " + " ".join(
            f"{m}={out['curves'][ncal][m]['mean']:.4f}" for m in methods
            if m in out["curves"][ncal]) + f"  best={best}")

    # ---- 4. active measurement selection (correction fixed to 'mult' for
    # n<8 and 'linear' for n>=8, chosen from the random-split curves' train
    # side, not from these runs)
    for strat in ("random", "farthest", "uncertainty", "novelty", "div_unc"):
        row = {}
        for ncal in N_CALS:
            vals = []
            for s in range(SEL_SEEDS):
                rng = np.random.default_rng(2000 + s)
                cal = select(strat, xm_n, unc, nov, ncal, rng)
                te = np.setdiff1d(np.arange(55), cal)
                fns = fit_corrections(xm[cal], res_log[cal], res_L[cal],
                                      mu[cal], unc[cal], norm)
                m = "linear" if (ncal >= 8 and "linear" in fns) else "mult"
                vals.append(rel_err_L(fns[m](xm[te], mu[te], unc[te]), ym[te]).mean())
                if strat in ("farthest",) or (strat != "random" and ncal > 1):
                    pass  # deterministic strategies repeat; keep loop uniform
            row[ncal] = dict(mean=float(np.mean(vals)),
                             ci=[float(np.quantile(vals, 0.025)),
                                 float(np.quantile(vals, 0.975))])
        out["active"][strat] = row
        print(f"active {strat:11s} " + " ".join(f"{ncal}:{row[ncal]['mean']:.4f}"
                                                for ncal in N_CALS))

    Path("results/sim2real_metrics.json").write_text(json.dumps(out, indent=1))
    print("wrote results/sim2real_metrics.json")


if __name__ == "__main__":
    main()
