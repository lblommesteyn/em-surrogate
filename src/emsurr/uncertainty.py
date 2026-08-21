"""Deep-ensemble uncertainty and the solver-fallback experiment."""

from __future__ import annotations

import numpy as np

from .metrics import complex_mae
from .models import TorchRegressor


class Ensemble:
    def __init__(self, n_members: int = 5, **kw):
        self.members = [TorchRegressor(seed=100 + i, **kw) for i in range(n_members)]

    def fit(self, train, val):
        for m in self.members:
            m.fit(train, val)
        return self

    def predict_with_uncertainty(self, samples):
        preds = np.stack([m.predict(samples) for m in self.members])  # (M,N,F,2,2)
        mean = preds.mean(0)
        # per-sample epistemic uncertainty: mean complex std across freq/ports
        unc = np.abs(preds - mean).mean(axis=(0, 2, 3, 4))
        return mean, unc


def per_sample_error(pred, true):
    return np.abs(pred - true).mean(axis=(1, 2, 3))


def confidence_curves(pred, unc, true, fracs=np.linspace(0.0, 0.9, 19)):
    """Error on retained (most-confident) subsets as uncertain ones are dropped."""
    err = per_sample_error(pred, true)
    order = np.argsort(unc)  # most confident first
    n = len(err)
    rows = []
    for f in fracs:
        keep = order[: max(1, int(round((1 - f) * n)))]
        rows.append(dict(dropped_frac=float(f), retained_mae=float(err[keep].mean())))
    return rows


def fallback_experiment(pred, unc, true, fallback_fracs=(0.1, 0.2, 0.3)):
    """If the real solver handles the top-p most-uncertain samples, what
    error remains on the rest, and how does that compare to random fallback
    and to oracle (drop truly-worst) fallback?"""
    err = per_sample_error(pred, true)
    n = len(err)
    rng = np.random.default_rng(0)
    out = []
    for p in fallback_fracs:
        k = int(round(p * n))
        by_unc = np.argsort(unc)[::-1][:k]
        by_err = np.argsort(err)[::-1][:k]
        rand = rng.choice(n, size=k, replace=False)
        rem = lambda drop: float(np.delete(err, drop).mean())
        out.append(
            dict(
                fallback_frac=p,
                remaining_mae_uncertainty=rem(by_unc),
                remaining_mae_random=rem(rand),
                remaining_mae_oracle=rem(by_err),
                baseline_mae=float(err.mean()),
                capture_rate=float(len(set(by_unc) & set(by_err)) / max(k, 1)),
            )
        )
    return out


def spearman(unc, err):
    ru = np.argsort(np.argsort(unc)).astype(float)
    re = np.argsort(np.argsort(err)).astype(float)
    return float(np.corrcoef(ru, re)[0, 1])
