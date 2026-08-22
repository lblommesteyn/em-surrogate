"""Mechanical adaptation of the frozen milestone-2 methodology to plain
tabular regression (feature vector -> scalar/vector target).

Nothing here is retuned: MLP config (hidden=256, depth=3, epochs=200,
lr=1e-3), 5-member deep ensemble, kNN novelty k=3, combined score
alpha=0.3 are carried over verbatim from the frozen synthetic benchmark.
Only the input plumbing changes (numpy arrays instead of sample dicts).
"""

from __future__ import annotations

import time

import numpy as np
import torch
from torch import nn

from .models import MLP, Normalizer


def _knn_mean_dist(q: np.ndarray, ref: np.ndarray, k: int, chunk: int = 256) -> np.ndarray:
    """Mean distance to k nearest refs, computed in query chunks to bound memory."""
    out = np.empty(len(q))
    for i in range(0, len(q), chunk):
        d = np.linalg.norm(q[i : i + chunk, None] - ref[None], axis=-1)
        out[i : i + chunk] = np.sort(d, 1)[:, :k].mean(1)
    return out
from .novelty import _rank01, auprc, auroc, oracle_recovery, risk_coverage

FROZEN = dict(hidden=256, depth=3, epochs=200, lr=1e-3, n_members=5, knn_k=3, alpha=0.3)


class TabRegressor:
    """MLP on standardized features; identical trainer to models.TorchRegressor."""

    def __init__(self, hidden=256, depth=3, epochs=200, lr=1e-3, seed=0):
        self.cfg = dict(hidden=hidden, depth=depth)
        self.epochs, self.lr, self.seed = epochs, lr, seed

    def fit(self, x, y, xv, yv):
        torch.manual_seed(self.seed)
        self.norm = Normalizer().fit(x)
        self.ynorm = Normalizer().fit(y)
        xt = torch.tensor(self.norm(x), dtype=torch.float32)
        yt = torch.tensor(self.ynorm(y), dtype=torch.float32)
        xvt = torch.tensor(self.norm(xv), dtype=torch.float32)
        yvt = torch.tensor(self.ynorm(yv), dtype=torch.float32)
        self.model = MLP(x.shape[1], y.shape[1], **self.cfg)
        opt = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        best, best_state = np.inf, None
        n = len(xt)
        for _ in range(self.epochs):
            self.model.train()
            perm = torch.randperm(n)
            for i in range(0, n, 256):
                b = perm[i : i + 256]
                opt.zero_grad()
                loss = nn.functional.mse_loss(self.model(xt[b]), yt[b])
                loss.backward()
                opt.step()
            self.model.eval()
            with torch.no_grad():
                vl = nn.functional.mse_loss(self.model(xvt), yvt).item()
            if vl < best:
                best = vl
                best_state = {k: v.clone() for k, v in self.model.state_dict().items()}
        self.model.load_state_dict(best_state)
        return self

    def predict(self, x):
        xt = torch.tensor(self.norm(x), dtype=torch.float32)
        self.model.eval()
        with torch.no_grad():
            return self.model(xt).numpy() * self.ynorm.sd + self.ynorm.mu

    def embed(self, x):
        xt = torch.tensor(self.norm(x), dtype=torch.float32)
        self.model.eval()
        with torch.no_grad():
            return self.model.net[:-1](xt).numpy()


class TabEnsemble:
    def __init__(self, n_members=5, **kw):
        self.members = [TabRegressor(seed=s, **kw) for s in range(n_members)]

    def fit(self, x, y, xv, yv):
        for m in self.members:
            m.fit(x, y, xv, yv)
        return self

    def predict_with_uncertainty(self, x):
        ys = np.stack([m.predict(x) for m in self.members])
        return ys.mean(0), ys.std(0).mean(-1)


class TabKNNInput:
    name = "knn_input"

    def __init__(self, k=3):
        self.k = k

    def fit(self, x, model=None):
        self.norm = Normalizer().fit(x)
        self.x = self.norm(x)
        return self

    def score(self, x):
        return _knn_mean_dist(self.norm(x), self.x, self.k)


class TabMahalanobisEmb:
    name = "mahalanobis_emb"

    def __init__(self, shrink=0.1):
        self.shrink = shrink

    def fit(self, x, model=None):
        e = model.embed(x)
        self.mu = e.mean(0)
        c = np.cov(e.T)
        c = (1 - self.shrink) * c + self.shrink * np.eye(len(c)) * np.trace(c) / len(c)
        self.prec = np.linalg.inv(c)
        self.model = model
        return self

    def score(self, x):
        d = self.model.embed(x) - self.mu
        return np.sqrt(np.einsum("nd,de,ne->n", d, self.prec, d))


class TabKNNEmb:
    name = "knn_emb"

    def __init__(self, k=3):
        self.k = k

    def fit(self, x, model=None):
        self.model = model
        self.e = model.embed(x)
        return self

    def score(self, x):
        return _knn_mean_dist(self.model.embed(x), self.e, self.k)


class TabEnsembleVar:
    name = "ensemble_var"

    def __init__(self, ens):
        self.ens = ens

    def fit(self, x, model=None):
        return self

    def score(self, x):
        _, unc = self.ens.predict_with_uncertainty(x)
        return unc


class TabCombined:
    name = "combined"

    def __init__(self, novelty, ens_score, alpha=0.3):
        self.novelty, self.ens_score, self.alpha = novelty, ens_score, alpha

    def fit(self, x, model=None):
        return self

    def score(self, x):
        a = _rank01(self.novelty.score(x))
        b = _rank01(self.ens_score.score(x))
        return self.alpha * a + (1 - self.alpha) * b


def spearman(a, b):
    ra, rb = _rank01(np.asarray(a)), _rank01(np.asarray(b))
    ra, rb = ra - ra.mean(), rb - rb.mean()
    den = np.sqrt((ra**2).sum() * (rb**2).sum())
    return float((ra * rb).sum() / den) if den > 0 else 0.0


def evaluate_pool(scorers, x_id, err_id, x_ood, err_ood, budgets=(0.05, 0.1, 0.2, 0.3, 0.5)):
    """OOD detection + risk-coverage on the ID+OOD pool, per scorer."""
    out = {}
    x_all = np.concatenate([x_id, x_ood])
    err_all = np.concatenate([err_id, err_ood])
    for sc in scorers:
        t0 = time.perf_counter()
        s_all = sc.score(x_all)  # single pass: rank-based scores need the full pool
        ms = (time.perf_counter() - t0) * 1e3 / len(x_all)
        s_id, s_ood = s_all[: len(x_id)], s_all[len(x_id) :]
        rc = risk_coverage(s_all, err_all, budgets)
        out[sc.name] = dict(
            auroc=auroc(s_id, s_ood),
            auprc=auprc(s_id, s_ood),
            spearman_err=spearman(s_all, err_all),
            risk_coverage=rc,
            oracle_recovery=oracle_recovery(rc),
            latency_ms=ms,
        )
    return out
