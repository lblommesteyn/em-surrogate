"""Dataset A benchmark: planar windings (external FEA + physical measurements).

Mechanical transfer of the frozen milestone-2 methodology; no hyperparameter
retuning, no tuning on MEAS. Outputs results/planar_windings_metrics.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from emsurr import planar_windings
from emsurr.external_tab import (
    FROZEN,
    TabCombined,
    TabEnsemble,
    TabEnsembleVar,
    TabKNNEmb,
    TabKNNInput,
    TabMahalanobisEmb,
    evaluate_pool,
    spearman,
)
from emsurr.models import Normalizer


def rel_err(pred_log, true_log):
    """Per-sample |L_hat - L| / L from log10-space predictions."""
    return np.abs(10 ** pred_log[:, 0] / 10 ** true_log[:, 0] - 1.0)


def summarize(pred_log, true_log):
    e = rel_err(pred_log, true_log)
    return dict(
        log10_mae=float(np.abs(pred_log - true_log).mean()),
        rel_err_mean=float(e.mean()),
        rel_err_median=float(np.median(e)),
        rel_err_p95=float(np.quantile(e, 0.95)),
    )


def main():
    rng = np.random.default_rng(0)
    data = planar_windings.load()
    xc, yc = data["CORE"]["x"], np.log10(data["CORE"]["y"])[:, None]
    perm = rng.permutation(len(xc))
    n = len(xc)
    tr, va, te = perm[: int(0.8 * n)], perm[int(0.8 * n) : int(0.9 * n)], perm[int(0.9 * n) :]
    xtr, ytr, xva, yva, xte, yte = xc[tr], yc[tr], xc[va], yc[va], xc[te], yc[te]
    xo, yo = data["OOD"]["x"], np.log10(data["OOD"]["y"])[:, None]
    xm, ym = data["MEAS"]["x"], np.log10(data["MEAS"]["y"])[:, None]

    res = {"n": dict(train=len(xtr), val=len(xva), test=len(xte), ood=len(xo), meas=len(xm)),
           "frozen_config": FROZEN}

    # ---- simple regression baseline: linear on standardized log-features
    norm = Normalizer().fit(xtr)
    A = np.c_[norm(xtr), np.ones(len(xtr))]
    coef, *_ = np.linalg.lstsq(A, ytr, rcond=None)
    lin = lambda x: np.c_[norm(x), np.ones(len(x))] @ coef
    res["linear"] = {k: summarize(lin(x), y) for k, (x, y) in
                     dict(core_test=(xte, yte), ood=(xo, yo), meas=(xm, ym)).items()}

    # ---- kNN regression baseline (k=3, frozen)
    xn_tr = norm(xtr)
    def knn_pred(x, k=3):
        d = np.linalg.norm(norm(x)[:, None] - xn_tr[None], axis=-1)
        idx = np.argsort(d, 1)[:, :k]
        return ytr[idx].mean(1)
    res["knn"] = {k: summarize(knn_pred(x), y) for k, (x, y) in
                  dict(core_test=(xte, yte), ood=(xo, yo), meas=(xm, ym)).items()}

    # ---- MLP ensemble (frozen config)
    ens = TabEnsemble(n_members=FROZEN["n_members"], hidden=FROZEN["hidden"],
                      depth=FROZEN["depth"], epochs=FROZEN["epochs"], lr=FROZEN["lr"])
    ens.fit(xtr, ytr, xva, yva)
    model = ens.members[0]

    preds = {}
    for k, (x, y) in dict(core_test=(xte, yte), ood=(xo, yo), meas=(xm, ym)).items():
        mu, unc = ens.predict_with_uncertainty(x)
        preds[k] = (mu, unc, y)
        res.setdefault("ensemble", {})[k] = summarize(mu, y)
        res["ensemble"][k]["spearman_unc_err"] = spearman(unc, rel_err(mu, y))

    # ---- novelty scorers (frozen: k=3, alpha=0.3, weights from milestone 2)
    knn_in = TabKNNInput(k=FROZEN["knn_k"]).fit(xtr)
    maha = TabMahalanobisEmb().fit(xtr, model)
    knn_e = TabKNNEmb(k=FROZEN["knn_k"]).fit(xtr, model)
    ens_v = TabEnsembleVar(ens)
    comb = TabCombined(knn_in, ens_v, alpha=FROZEN["alpha"])
    scorers = [knn_in, maha, knn_e, ens_v, comb]

    err_id = rel_err(*[preds["core_test"][i] for i in (0, 2)])
    err_ood = rel_err(*[preds["ood"][i] for i in (0, 2)])
    res["novelty_pool_full"] = evaluate_pool(scorers, xte, err_id, xo, err_ood)

    # deployment-mix pool: all ID test + 25% OOD subsample (mirrors milestone 2 pool B)
    sub = np.random.default_rng(1).choice(len(xo), size=len(xte) // 3, replace=False)
    res["novelty_pool_mix"] = evaluate_pool(scorers, xte, err_id, xo[sub], err_ood[sub])

    # ---- MEAS as the deployment target: are measurements flagged, and does
    # fallback help there? (evaluation only; nothing tuned on MEAS)
    err_meas = rel_err(*[preds["meas"][i] for i in (0, 2)])
    res["meas_vs_core"] = evaluate_pool(scorers, xte, err_id, xm, err_meas)

    out = Path("results/planar_windings_metrics.json")
    out.write_text(json.dumps(res, indent=1))
    print(json.dumps({k: res[k] for k in ["linear", "knn", "ensemble"]}, indent=1))
    print("wrote", out)


if __name__ == "__main__":
    main()
