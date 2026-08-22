"""Risk-calibration signal extraction: SQChip-EM held-out families.

For each frozen held-out family (batch300, batchnear50, sqchip; the same
folds as the external-data milestone), the training pool is everything else.
Calibration pseudo-OOD is built strictly inside that training pool by holding
out its LARGEST family (a-priori rule). Representation: parameter vector
(the frozen milestone's primary representation).

Error metric: mean |dy| across the 6 targets, each normalized by the training
std of that target (scales GHz and MHz comparably).

Writes results/riskcal/sqchip_<fold>.npz with pools:
  val_cal / pseudo / val_final / eval_<fold>
"""

import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from emsurr import sqchip
from emsurr.external_tab import (FROZEN, TabEnsemble, TabKNNEmb, TabKNNInput,
                                 TabMahalanobisEmb)

EK = dict(n_members=FROZEN["n_members"], hidden=FROZEN["hidden"],
          depth=FROZEN["depth"], epochs=FROZEN["epochs"], lr=FROZEN["lr"])
K = FROZEN["knn_k"]
FOLDS = ["batch300", "batchnear50", "sqchip"]


def signals(xtr, ens, x, y, ystd, is_ood):
    model = ens.members[0]
    knn_in = TabKNNInput(k=K).fit(xtr)
    maha = TabMahalanobisEmb().fit(xtr, model)
    knn_e = TabKNNEmb(k=K).fit(xtr, model)
    mu, unc = ens.predict_with_uncertainty(x)
    return dict(knn_input=knn_in.score(x), maha_emb=maha.score(x),
                knn_emb=knn_e.score(x), ens_var=unc,
                err=(np.abs(mu - y) / ystd).mean(1),
                is_ood=np.asarray(is_ood, float))


def main():
    recs = sqchip.load_records()
    y = np.stack([r["y"] for r in recs])
    xp, _ = sqchip.param_matrix(recs)
    fam = np.array([r["family"] for r in recs])

    for hold in FOLDS:
        print("== fold", hold)
        rest = np.where(fam != hold)[0]
        ho = np.where(fam == hold)[0]
        rng = np.random.default_rng(1)
        rp = rng.permutation(rest)
        tr, va = rp[: int(0.9 * len(rp))], rp[int(0.9 * len(rp)) :]
        ystd = y[tr].std(0) + 1e-9
        out = {}

        # calibration: hold out the largest family inside the training pool
        train_fams = Counter(fam[tr])
        pseudo_fam = train_fams.most_common(1)[0][0]
        ptr = tr[fam[tr] != pseudo_fam]
        pva = va[fam[va] != pseudo_fam]
        pho = np.concatenate([tr[fam[tr] == pseudo_fam], va[fam[va] == pseudo_fam]])
        print(f"  pseudo-OOD family: {pseudo_fam} ({len(pho)}), cal train {len(ptr)}")
        ens_c = TabEnsemble(**EK).fit(xp[ptr], y[ptr], xp[pva], y[pva])
        out["val_cal"] = signals(xp[ptr], ens_c, xp[pva], y[pva], ystd, [0] * len(pva))
        pool = np.concatenate([pva, pho])
        out["pseudo"] = signals(xp[ptr], ens_c, xp[pool], y[pool], ystd,
                                [0] * len(pva) + [1] * len(pho))

        # evaluation: frozen fold (train pool -> held-out family)
        ens_f = TabEnsemble(**EK).fit(xp[tr], y[tr], xp[va], y[va])
        out["val_final"] = signals(xp[tr], ens_f, xp[va], y[va], ystd, [0] * len(va))
        ev = np.concatenate([va, ho])
        out["eval"] = signals(xp[tr], ens_f, xp[ev], y[ev], ystd,
                              [0] * len(va) + [1] * len(ho))

        Path("results/riskcal").mkdir(parents=True, exist_ok=True)
        np.savez_compressed(f"results/riskcal/sqchip_{hold}.npz",
                            **{f"{p}/{k}": v for p, t in out.items() for k, v in t.items()})
        print(f"  wrote results/riskcal/sqchip_{hold}.npz")


if __name__ == "__main__":
    main()
