"""Topology-novelty scores and OOD-detection / solver-fallback evaluation.

All scores are "higher = more novel". Novelty features never use the
topology-family one-hot: a genuinely new structure has no label, so scores
must come from structure content (design parameters, element tokens) or
learned embeddings of those inputs. Unknown families are encoded with an
all-zero one-hot when a model input requires that block.
"""

from __future__ import annotations

import numpy as np

from .models import Normalizer, features


# ---------------------------------------------------------------- features
def content_features(samples: list[dict]) -> np.ndarray:
    """Design parameters + pooled element-token summary (no family label)."""
    p = features(samples, with_family=False)
    tok = np.zeros((len(samples), 5 + 4 + 4 + 1))  # type hist, mean/max params, count
    for i, s in enumerate(samples):
        els = s["elements"]
        if len(els) == 0:
            continue
        types = els[:, 0].astype(int)
        for t in types:
            if 0 <= t < 5:
                tok[i, t] += 1
        tok[i, 5:9] = els[:, 1:5].mean(0)
        tok[i, 9:13] = els[:, 1:5].max(0)
        tok[i, 13] = len(els)
    return np.concatenate([p, tok], -1)


# ---------------------------------------------------------------- scorers
class KNNInputNovelty:
    """Mean distance to k nearest training samples in normalized content space."""

    name = "knn_input"

    def __init__(self, k: int = 10):
        self.k = k

    def fit(self, train: list[dict], model=None):
        x = content_features(train)
        self.norm = Normalizer().fit(x)
        self.x = self.norm(x)
        return self

    def score(self, samples: list[dict]) -> np.ndarray:
        xq = self.norm(content_features(samples))
        d = np.linalg.norm(xq[:, None] - self.x[None], axis=-1)
        return np.sort(d, 1)[:, : self.k].mean(1)


class MahalanobisEmbedding:
    """Mahalanobis distance to the training distribution in the surrogate's
    penultimate-layer embedding space (shrinkage-regularized covariance)."""

    name = "mahalanobis_emb"

    def __init__(self, shrink: float = 0.1):
        self.shrink = shrink

    def fit(self, train: list[dict], model=None):
        e = model.embed(train)
        self.mu = e.mean(0)
        c = np.cov(e.T)
        c = (1 - self.shrink) * c + self.shrink * np.eye(len(c)) * np.trace(c) / len(c)
        self.prec = np.linalg.inv(c)
        self.model = model
        return self

    def score(self, samples: list[dict]) -> np.ndarray:
        d = self.model.embed(samples) - self.mu
        return np.sqrt(np.einsum("nd,de,ne->n", d, self.prec, d))


class KNNEmbedding:
    """Mean k-NN distance in the surrogate's embedding space."""

    name = "knn_emb"

    def __init__(self, k: int = 10):
        self.k = k

    def fit(self, train: list[dict], model=None):
        self.model = model
        self.e = model.embed(train)
        return self

    def score(self, samples: list[dict]) -> np.ndarray:
        eq = self.model.embed(samples)
        d = np.linalg.norm(eq[:, None] - self.e[None], axis=-1)
        return np.sort(d, 1)[:, : self.k].mean(1)


class EnsembleVariance:
    """Milestone-1 baseline: mean member disagreement of the deep ensemble."""

    name = "ensemble_var"

    def __init__(self, ensemble):
        self.ens = ensemble

    def fit(self, train: list[dict], model=None):
        return self

    def score(self, samples: list[dict]) -> np.ndarray:
        _, unc = self.ens.predict_with_uncertainty(samples)
        return unc


class CombinedScore:
    """Rank-average of a novelty score and ensemble variance."""

    name = "combined"

    def __init__(self, novelty, ens_score, alpha: float = 0.5):
        self.novelty, self.ens_score, self.alpha = novelty, ens_score, alpha

    def fit(self, train: list[dict], model=None):
        return self

    def score(self, samples: list[dict]) -> np.ndarray:
        a = _rank01(self.novelty.score(samples))
        b = _rank01(self.ens_score.score(samples))
        return self.alpha * a + (1 - self.alpha) * b


def _rank01(x: np.ndarray) -> np.ndarray:
    r = np.argsort(np.argsort(x)).astype(float)
    return r / max(len(x) - 1, 1)


# ---------------------------------------------------------------- evaluation
def auroc(scores_id: np.ndarray, scores_ood: np.ndarray) -> float:
    """P(score_ood > score_id) via rank statistic."""
    s = np.concatenate([scores_id, scores_ood])
    y = np.concatenate([np.zeros(len(scores_id)), np.ones(len(scores_ood))])
    order = np.argsort(s)
    ranks = np.empty(len(s))
    ranks[order] = np.arange(1, len(s) + 1)
    n1, n0 = y.sum(), (1 - y).sum()
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def auprc(scores_id: np.ndarray, scores_ood: np.ndarray) -> float:
    s = np.concatenate([scores_id, scores_ood])
    y = np.concatenate([np.zeros(len(scores_id)), np.ones(len(scores_ood))])
    order = np.argsort(-s)
    y = y[order]
    tp = np.cumsum(y)
    prec = tp / np.arange(1, len(y) + 1)
    return float(np.sum(prec * y) / max(y.sum(), 1))


def risk_coverage(scores: np.ndarray, errors: np.ndarray, budgets) -> list[dict]:
    """Defer the top-budget fraction by score; report error on the rest,
    vs random and oracle deferral, plus catastrophic capture."""
    n = len(errors)
    rng = np.random.default_rng(0)
    cat_thresh = np.quantile(errors, 0.95)
    catastrophic = errors >= cat_thresh
    out = []
    for p in budgets:
        k = max(1, int(round(p * n)))
        by_s = np.argsort(scores)[::-1][:k]
        by_e = np.argsort(errors)[::-1][:k]
        rand = rng.choice(n, size=k, replace=False)
        rem = lambda drop: float(np.delete(errors, drop).mean())
        out.append(
            dict(
                budget=float(p),
                remaining_mae=rem(by_s),
                remaining_mae_random=rem(rand),
                remaining_mae_oracle=rem(by_e),
                catastrophic_caught=float(catastrophic[by_s].sum() / max(catastrophic.sum(), 1)),
            )
        )
    return out


def oracle_recovery(rc: list[dict]) -> dict:
    """Fraction of (base - oracle) error reduction achieved, per budget."""
    out = {}
    for r in rc:
        base = r["remaining_mae_random"]
        denom = base - r["remaining_mae_oracle"]
        out[r["budget"]] = float((base - r["remaining_mae"]) / denom) if denom > 1e-12 else 0.0
    return out
