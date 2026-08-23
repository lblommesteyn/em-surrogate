"""Topology-aware structure embeddings for EM designs.

Small DeepSets-style encoders over element tokens (type one-hot + 4 physical
params + optional cascade position). No topology-family label or dataset ID
is ever encoded. Objectives use only training structures and their solver
responses:

  resp    predict the normalized S-response vector; embedding = pooled layer
  metric  make embedding distance regress the response distance |dS| between
          pairs (batch-pairwise, distances scaled by their train median)
  both    sum of the two

Variants: ordered=True keeps the cascade-position feature (the structures
are 2-port cascades, so position IS the connectivity graph of a path);
ordered=False zeroes it for a purely permutation-invariant set encoding.

Model size: phi 64x2, embedding 32, head 64 -> deliberately small.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from .models import Normalizer, targets

N_TYPES, MAX_EL = 5, 8
EMB = 32
H = 64


def tokens(samples, ordered=True, n_types=N_TYPES):
    d = n_types + 5
    x = np.zeros((len(samples), MAX_EL, d))
    m = np.zeros((len(samples), MAX_EL))
    for i, s in enumerate(samples):
        for j, el in enumerate(s["elements"][:MAX_EL]):
            x[i, j, int(el[0])] = 1
            x[i, j, n_types : n_types + 4] = el[1:5]
            x[i, j, -1] = (j / MAX_EL) if ordered else 0.0
            m[i, j] = 1
    return x, m


def resp_dist(sa, sb):
    return float(np.abs(sa - sb).mean())


def resp_dist_matrix(S):
    n = len(S)
    D = np.zeros((n, n))
    for i in range(n):
        D[i, i + 1 :] = np.abs(S[i + 1 :] - S[i]).mean(axis=(1, 2, 3))
    return D + D.T


class SetEncoder(nn.Module):
    def __init__(self, din, dout):
        super().__init__()
        self.phi = nn.Sequential(nn.Linear(din, H), nn.ReLU(), nn.Linear(H, H), nn.ReLU())
        self.proj = nn.Linear(2 * H, EMB)
        self.head = nn.Sequential(nn.Linear(EMB, H), nn.ReLU(), nn.Linear(H, dout))

    def embed(self, x, m):
        e = self.phi(x) * m[..., None]
        mean = e.sum(1) / m.sum(1, keepdim=True).clamp(min=1)
        mx = e.masked_fill(m[..., None] == 0, -1e9).max(1).values
        return self.proj(torch.cat([mean, mx], -1))

    def forward(self, x, m):
        z = self.embed(x, m)
        return z, self.head(z)


class TopoEmbedding:
    def __init__(self, ordered=True, objective="both", epochs=150, lr=1e-3, seed=0,
                 n_types=N_TYPES):
        self.ordered, self.objective, self.n_types = ordered, objective, n_types
        self.epochs, self.lr, self.seed = epochs, lr, seed

    def _prep(self, samples):
        x, m = tokens(samples, self.ordered, self.n_types)
        return self.tok_norm(x) * m[..., None], m

    def fit(self, train, val):
        torch.manual_seed(self.seed)
        xr, mr = tokens(train, self.ordered, self.n_types)
        flat = xr.reshape(-1, xr.shape[-1])
        self.tok_norm = Normalizer().fit(flat[mr.reshape(-1) > 0])
        self.ynorm = Normalizer().fit(targets(train))
        x, m = self._prep(train)
        y = self.ynorm(targets(train))
        xv, mv = self._prep(val)
        yv = self.ynorm(targets(val))
        S = np.stack([s["s"] for s in train])
        # pair-distance scale from train responses
        rng = np.random.default_rng(0)
        ii, jj = rng.integers(0, len(S), 2000), rng.integers(0, len(S), 2000)
        self.dscale = float(np.median(np.abs(S[ii] - S[jj]).mean(axis=(1, 2, 3))) + 1e-9)

        tt = lambda a: torch.tensor(a, dtype=torch.float32)
        xt, mt, yt = tt(x), tt(m), tt(y)
        xvt, mvt, yvt = tt(xv), tt(mv), tt(yv)
        St = torch.tensor(np.abs(S).reshape(len(S), -1), dtype=torch.float32)
        # complex parts for pair distances
        Sfull = torch.tensor(
            np.stack([S.real, S.imag], -1).reshape(len(S), -1), dtype=torch.float32)

        self.model = SetEncoder(x.shape[-1], y.shape[1])
        opt = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        best, best_state = np.inf, None
        n = len(xt)
        for ep in range(self.epochs):
            self.model.train()
            perm = torch.randperm(n)
            for i in range(0, n, 128):
                b = perm[i : i + 128]
                opt.zero_grad()
                z, pred = self.model(xt[b], mt[b])
                loss = 0.0
                if self.objective in ("resp", "both"):
                    loss = loss + nn.functional.mse_loss(pred, yt[b])
                if self.objective in ("metric", "both"):
                    dz = torch.cdist(z, z)
                    dr = torch.cdist(Sfull[b], Sfull[b]) / (
                        Sfull.shape[1] ** 0.5) / self.dscale
                    iu = torch.triu_indices(len(b), len(b), 1)
                    loss = loss + nn.functional.mse_loss(dz[iu[0], iu[1]],
                                                         dr[iu[0], iu[1]])
                loss.backward()
                opt.step()
            self.model.eval()
            with torch.no_grad():
                zv, pv = self.model(xvt, mvt)
                vl = nn.functional.mse_loss(pv, yvt).item() if self.objective != "metric" else 0.0
                if self.objective in ("metric", "both"):
                    dz = torch.cdist(zv[:200], zv[:200])
                    vl += float(dz.mean()) * 0  # metric val tracked via resp or skipped
            if self.objective == "metric":
                best_state = {k: v.clone() for k, v in self.model.state_dict().items()}
            elif vl < best:
                best = vl
                best_state = {k: v.clone() for k, v in self.model.state_dict().items()}
        self.model.load_state_dict(best_state)
        return self

    def embed(self, samples):
        x, m = self._prep(samples)
        self.model.eval()
        with torch.no_grad():
            return self.model.embed(torch.tensor(x, dtype=torch.float32),
                                    torch.tensor(m, dtype=torch.float32)).numpy()


def knn_dist(qe, re_, k=3):
    d = np.linalg.norm(qe[:, None] - re_[None], axis=-1)
    return np.sort(d, 1)[:, :k].mean(1)
