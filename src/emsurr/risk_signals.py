"""Extended per-sample risk signals for within-OOD error ranking.

All signals use only frozen models and training-side data. No target-OOD
error is used to design, scale, or tune any signal.

Signals (per sample):
  ens_var       mean member std (frozen baseline signal)
  ens_range     mean across outputs of (max - min) over members
  knn_input     input-space kNN distance (k=3)
  knn_emb       embedding-space kNN distance (k=3)
  maha_emb      Mahalanobis distance in embedding space
  perturb_sens  mean |f(x+d) - f(x)| over 8 small Gaussian input
                perturbations (1% of per-feature train std; zero-std
                features never perturbed, so one-hot/constant dims stay
                physically valid)
  knn_pred_gap  |f(x) - kNN-regressor prediction| (k=3 train-target mean)
  nbr_target_var  distance-weighted std of the 10 nearest train targets
  jac_norm      input-gradient norm of the mean prediction (autograd)
  rep_gap       |rank(knn_input) - rank(knn_emb)| within the scored pool
                (computed at analysis time from the two kNN signals)
"""

from __future__ import annotations

import numpy as np
import torch

from .external_tab import TabKNNEmb, TabKNNInput, TabMahalanobisEmb, _knn_mean_dist
from .novelty import _rank01

SIGNALS = ["ens_var", "ens_range", "knn_input", "knn_emb", "maha_emb",
           "perturb_sens", "knn_pred_gap", "nbr_target_var", "jac_norm"]


def tab_signals(ens, xtr, ytr, x, k=3, n_pert=8, seed=0):
    """Signal matrix dict for tabular deployments (planar, sqchip)."""
    model = ens.members[0]
    out = {}
    preds = np.stack([m.predict(x) for m in ens.members])   # (M, N, T)
    out["ens_var"] = preds.std(0).mean(-1)
    out["ens_range"] = (preds.max(0) - preds.min(0)).mean(-1)
    out["knn_input"] = TabKNNInput(k=k).fit(xtr).score(x)
    out["knn_emb"] = TabKNNEmb(k=k).fit(xtr, model).score(x)
    out["maha_emb"] = TabMahalanobisEmb().fit(xtr, model).score(x)

    rng = np.random.default_rng(seed)
    sd = xtr.std(0)
    base = preds.mean(0)
    sens = np.zeros(len(x))
    for _ in range(n_pert):
        xp = x + rng.normal(0, 0.01, x.shape) * sd
        pp = np.stack([m.predict(xp) for m in ens.members]).mean(0)
        sens += np.abs(pp - base).mean(-1)
    out["perturb_sens"] = sens / n_pert

    # kNN regressor on the training targets
    norm = model.norm
    xq, xr = norm(x), norm(xtr)
    idx = np.empty((len(x), max(k, 10)), int)
    for i in range(0, len(x), 256):
        dd = np.linalg.norm(xq[i:i + 256, None] - xr[None], axis=-1)
        idx[i:i + 256] = np.argsort(dd, 1)[:, : max(k, 10)]
    knn_pred = ytr[idx[:, :k]].mean(1)
    out["knn_pred_gap"] = np.abs(base - knn_pred).mean(-1)

    # distance-weighted std of 10 nearest train targets
    nb = ytr[idx[:, :10]]                       # (N, 10, T)
    out["nbr_target_var"] = nb.std(1).mean(-1)

    # input-gradient norm of the summed output (autograd, per sample)
    model.model.eval()
    xt2 = torch.tensor(norm(x), dtype=torch.float32, requires_grad=True)
    ys = model.model(xt2).sum(-1)
    g = torch.autograd.grad(ys.sum(), xt2)[0]
    out["jac_norm"] = g.norm(dim=1).detach().numpy()
    return out


def rep_gap(sig_pool):
    return np.abs(_rank01(np.asarray(sig_pool["knn_input"], float))
                  - _rank01(np.asarray(sig_pool["knn_emb"], float)))


def rank_features(sig_pool, keys):
    """Rank-normalized signal matrix (per pool) + rep_gap column."""
    cols = [_rank01(np.asarray(sig_pool[k], float)) for k in keys]
    cols.append(rep_gap(sig_pool))
    return np.stack(cols, 1)


FEATURE_NAMES = SIGNALS + ["rep_gap"]


def fit_rank_ridge(F, err, lam=1.0):
    """Ridge on rank-normalized features -> rank of error."""
    y = _rank01(np.asarray(err, float))
    A = np.c_[F, np.ones(len(F))]
    w = np.linalg.solve(A.T @ A + lam * np.eye(A.shape[1]), A.T @ y)
    return w


def predict_rank_ridge(w, F):
    return np.c_[F, np.ones(len(F))] @ w
