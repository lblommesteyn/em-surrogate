"""Dataset D: SQChip-EM representation baseline.

Question: does GDS layout geometry carry structural-novelty / prediction
signal beyond the parameter vector? Compared mechanically with the frozen
methodology (no new architectures):
  reps: param vector | geometry-derived features | both concatenated
  evals: random split IID error; held-out generation-family error;
         family-novelty AUROC via kNN-input in each representation.
Outputs results/sqchip_metrics.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from emsurr import sqchip
from emsurr.external_tab import FROZEN, TabEnsemble, TabKNNInput, spearman
from emsurr.models import Normalizer
from emsurr.novelty import auroc

HOLDOUT_MIN = 20  # families this size or larger get a held-out evaluation


def mae_groups(pred, true):
    e = np.abs(pred - true)
    return dict(
        f01_mae_ghz=float(e[:, 0:2].mean()),
        chi_mae_mhz=float(e[:, 2:4].mean()),
        fr_mae_ghz=float(e[:, 4:6].mean()),
    )


def fit_eval(xtr, ytr, xva, yva, xte, yte):
    ens = TabEnsemble(n_members=FROZEN["n_members"], hidden=FROZEN["hidden"],
                      depth=FROZEN["depth"], epochs=FROZEN["epochs"], lr=FROZEN["lr"])
    ens.fit(xtr, ytr, xva, yva)
    mu, unc = ens.predict_with_uncertainty(xte)
    out = mae_groups(mu, yte)
    out["spearman_unc_err"] = spearman(unc, np.abs(mu - yte).mean(1))
    return out


def linear_eval(xtr, ytr, xte, yte):
    norm = Normalizer().fit(xtr)
    A = np.c_[norm(xtr), np.ones(len(xtr))]
    coef, *_ = np.linalg.lstsq(A, ytr, rcond=None)
    return mae_groups(np.c_[norm(xte), np.ones(len(xte))] @ coef, yte)


def main():
    recs = sqchip.load_records()
    fams = [r["family"] for r in recs]
    from collections import Counter

    fam_counts = Counter(fams)
    y = np.stack([r["y"] for r in recs])
    xp, keys = sqchip.param_matrix(recs)
    xg = np.stack([sqchip.geometry_features(r["gds"]) for r in recs])
    xc = np.c_[xp, xg]
    reps = dict(param=xp, geom=xg, both=xc)

    res = dict(n=len(recs), n_param_keys=len(keys), n_geom_feats=xg.shape[1],
               families=dict(fam_counts.most_common()), frozen_config=FROZEN)

    # ---- random-split IID benchmark
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(recs))
    n = len(recs)
    tr, va, te = perm[: int(0.7 * n)], perm[int(0.7 * n) : int(0.8 * n)], perm[int(0.8 * n) :]
    res["random_split"] = {}
    for name, x in reps.items():
        res["random_split"][name] = dict(
            linear=linear_eval(x[tr], y[tr], x[te], y[te]),
            mlp_ensemble=fit_eval(x[tr], y[tr], x[va], y[va], x[te], y[te]),
        )

    # ---- held-out family: error + novelty AUROC per representation
    big = [f for f, c in fam_counts.items() if c >= HOLDOUT_MIN]
    res["family_holdout"] = {}
    fam_arr = np.array(fams)
    for hold in big:
        m_hold = fam_arr == hold
        rest = np.where(~m_hold)[0]
        rng2 = np.random.default_rng(1)
        rp = rng2.permutation(rest)
        tr2, va2 = rp[: int(0.9 * len(rp))], rp[int(0.9 * len(rp)) :]
        te_id = va2  # ID reference for novelty
        ho = np.where(m_hold)[0]
        entry = {}
        for name, x in reps.items():
            r = fit_eval(x[tr2], y[tr2], x[va2], y[va2], x[ho], y[ho])
            nov = TabKNNInput(k=FROZEN["knn_k"]).fit(x[tr2])
            r["novelty_auroc"] = auroc(nov.score(x[te_id]), nov.score(x[ho]))
            entry[name] = r
        res["family_holdout"][hold] = dict(n_holdout=int(m_hold.sum()), **entry)

    out = Path("results/sqchip_metrics.json")
    out.write_text(json.dumps(res, indent=1))
    print(json.dumps(res["random_split"], indent=1))
    print("wrote", out)


if __name__ == "__main__":
    main()
