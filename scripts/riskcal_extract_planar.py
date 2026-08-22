"""Risk-calibration signal extraction: planar windings.

Pools written to results/riskcal/planar.npz:
  val_cal    IID val scored by a calibration ensemble trained on the central-
             quantile subset of train (honest train-side errors)
  pseudo     quantile-extrapolation pseudo-OOD: central-val (ID) + rows with
             any parameter outside the train [q10, q90] band (pseudo-OOD),
             scored by the central-subset ensemble. A-priori rule, declared
             before any evaluation; strictly inside CORE train+val.
  val_final  IID val under the full-train final ensemble (z-stat reference)
  eval_ood   CORE test (ID) + provided OOD split, final ensemble
  eval_meas  CORE test (ID) + MEAS (55 physical), final ensemble

Frozen config; provided CORE/OOD/MEAS splits preserved; MEAS never used for
calibration.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from emsurr import planar_windings
from emsurr.external_tab import (FROZEN, TabEnsemble, TabKNNEmb, TabKNNInput,
                                 TabMahalanobisEmb)

EK = dict(n_members=FROZEN["n_members"], hidden=FROZEN["hidden"],
          depth=FROZEN["depth"], epochs=FROZEN["epochs"], lr=FROZEN["lr"])
K = FROZEN["knn_k"]
QLO, QHI = 0.10, 0.90  # central-band rule, fixed a priori


def signals(xtr, ens, x, y=None, is_ood=None):
    model = ens.members[0]
    knn_in = TabKNNInput(k=K).fit(xtr)
    maha = TabMahalanobisEmb().fit(xtr, model)
    knn_e = TabKNNEmb(k=K).fit(xtr, model)
    mu, unc = ens.predict_with_uncertainty(x)
    out = dict(knn_input=knn_in.score(x), maha_emb=maha.score(x),
               knn_emb=knn_e.score(x), ens_var=unc)
    if y is not None:
        out["err"] = np.abs(10 ** mu[:, 0] / 10 ** y[:, 0] - 1.0)
    if is_ood is not None:
        out["is_ood"] = np.asarray(is_ood, float)
    return out


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

    out = {}

    # ---- calibration side: central-quantile ensemble (train data only)
    qlo, qhi = np.quantile(xtr, QLO, 0), np.quantile(xtr, QHI, 0)
    central_tr = ((xtr >= qlo) & (xtr <= qhi)).all(1)
    central_va = ((xva >= qlo) & (xva <= qhi)).all(1)
    x_ctr, y_ctr = xtr[central_tr], ytr[central_tr]
    x_cva, y_cva = xva[central_va], yva[central_va]
    x_ext = np.concatenate([xtr[~central_tr], xva[~central_va]])
    y_ext = np.concatenate([ytr[~central_tr], yva[~central_va]])
    print(f"central train {len(x_ctr)}, central val {len(x_cva)}, extreme {len(x_ext)}")

    ens_c = TabEnsemble(**EK).fit(x_ctr, y_ctr, x_cva, y_cva)
    out["val_cal"] = signals(x_ctr, ens_c, x_cva, y_cva, [0] * len(x_cva))
    xp = np.concatenate([x_cva, x_ext])
    yp = np.concatenate([y_cva, y_ext])
    out["pseudo"] = signals(x_ctr, ens_c, xp, yp,
                            [0] * len(x_cva) + [1] * len(x_ext))

    # ---- evaluation side: full-train final ensemble (frozen recipe)
    ens_f = TabEnsemble(**EK).fit(xtr, ytr, xva, yva)
    out["val_final"] = signals(xtr, ens_f, xva, yva, [0] * len(xva))
    out["eval_ood"] = signals(xtr, ens_f, np.concatenate([xte, xo]),
                              np.concatenate([yte, yo]),
                              [0] * len(xte) + [1] * len(xo))
    out["eval_meas"] = signals(xtr, ens_f, np.concatenate([xte, xm]),
                               np.concatenate([yte, ym]),
                               [0] * len(xte) + [1] * len(xm))

    Path("results/riskcal").mkdir(parents=True, exist_ok=True)
    np.savez_compressed("results/riskcal/planar.npz",
                        **{f"{p}/{k}": v for p, t in out.items() for k, v in t.items()})
    print("wrote results/riskcal/planar.npz")


if __name__ == "__main__":
    main()
