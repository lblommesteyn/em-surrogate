"""Baseline models.

All models map a sample to complex 2-port S-parameters on the common grid,
represented internally as a real vector of length F*2*2*2 (Re/Im stacked).
Training is deliberately CPU-only (models are tiny; the GPU is shared).
"""

from __future__ import annotations

import time

import numpy as np
import torch
import torch.nn as nn

from .synth import FAMILIES

DEVICE = "cpu"


def s_to_vec(s: np.ndarray) -> np.ndarray:
    return np.concatenate([s.real.reshape(len(s), -1), s.imag.reshape(len(s), -1)], -1)


def vec_to_s(v: np.ndarray, F: int) -> np.ndarray:
    half = v.shape[-1] // 2
    re = v[..., :half].reshape(-1, F, 2, 2)
    im = v[..., half:].reshape(-1, F, 2, 2)
    return re + 1j * im


def features(samples: list[dict], with_family: bool = True) -> np.ndarray:
    x = np.stack([s["params"] for s in samples])
    if with_family:
        fam = np.zeros((len(samples), len(FAMILIES)))
        for i, s in enumerate(samples):
            if s["topology_family"] in FAMILIES:
                fam[i, FAMILIES.index(s["topology_family"])] = 1
        x = np.concatenate([x, fam], -1)
    return x


def targets(samples: list[dict]) -> np.ndarray:
    return s_to_vec(np.stack([s["s"] for s in samples]))


class Normalizer:
    def fit(self, x):
        self.mu = x.mean(0)
        self.sd = x.std(0)
        # columns constant in training (e.g. unseen-family one-hots or unused
        # params) must not explode when nonzero at test time
        self.sd = np.where(self.sd < 1e-9, 1.0, self.sd)
        return self

    def __call__(self, x):
        return (x - self.mu) / self.sd


class KNNBaseline:
    """Inverse-distance-weighted k-nearest-neighbour in normalized param space."""

    def __init__(self, k: int = 5):
        self.k = k

    def fit(self, train: list[dict], val=None):
        self.norm = Normalizer().fit(features(train))
        self.x = self.norm(features(train))
        self.y = targets(train)
        return self

    def predict(self, samples: list[dict]) -> np.ndarray:
        xq = self.norm(features(samples))
        d = np.linalg.norm(xq[:, None] - self.x[None], axis=-1)
        idx = np.argsort(d, 1)[:, : self.k]
        dd = np.take_along_axis(d, idx, 1)
        w = 1.0 / (dd + 1e-9)
        w /= w.sum(1, keepdims=True)
        pred = np.einsum("nk,nkd->nd", w, self.y[idx])
        F = samples[0]["freq"].shape[0]
        return vec_to_s(pred, F)


class MLP(nn.Module):
    def __init__(self, din, dout, hidden=256, depth=3, dropout=0.0):
        super().__init__()
        layers, d = [], din
        for _ in range(depth):
            layers += [nn.Linear(d, hidden), nn.ReLU(), nn.Dropout(dropout)]
            d = hidden
        layers.append(nn.Linear(d, dout))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class TorchRegressor:
    """Shared trainer for MLP-style baselines on flattened parameters."""

    def __init__(self, hidden=256, depth=3, dropout=0.0, epochs=200, lr=1e-3, seed=0):
        self.cfg = dict(hidden=hidden, depth=depth, dropout=dropout)
        self.epochs, self.lr, self.seed = epochs, lr, seed

    def _make_xy(self, samples):
        return self.norm(features(samples)), targets(samples)

    def fit(self, train: list[dict], val: list[dict]):
        torch.manual_seed(self.seed)
        self.norm = Normalizer().fit(features(train))
        self.ynorm = Normalizer().fit(targets(train))
        x, y = self._make_xy(train)
        xv, yv = self._make_xy(val)
        y, yv = self.ynorm(y), self.ynorm(yv)
        xt = torch.tensor(x, dtype=torch.float32)
        yt = torch.tensor(y, dtype=torch.float32)
        xvt = torch.tensor(xv, dtype=torch.float32)
        yvt = torch.tensor(yv, dtype=torch.float32)
        self.model = MLP(x.shape[1], y.shape[1], **self.cfg)
        opt = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        best, best_state = np.inf, None
        n = len(xt)
        for ep in range(self.epochs):
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
                best, best_state = vl, {
                    k: v.clone() for k, v in self.model.state_dict().items()
                }
        self.model.load_state_dict(best_state)
        return self

    def predict(self, samples: list[dict], mc_dropout: int = 0) -> np.ndarray:
        x = torch.tensor(self.norm(features(samples)), dtype=torch.float32)
        F = samples[0]["freq"].shape[0]
        if mc_dropout:
            self.model.train()  # keep dropout active
            with torch.no_grad():
                ys = np.stack(
                    [self.model(x).numpy() for _ in range(mc_dropout)]
                )
            ys = ys * self.ynorm.sd + self.ynorm.mu
            return np.stack([vec_to_s(y, F) for y in ys])  # (T, N, F, 2, 2)
        self.model.eval()
        with torch.no_grad():
            y = self.model(x).numpy() * self.ynorm.sd + self.ynorm.mu
        return vec_to_s(y, F)


class DeepSetsModel(TorchRegressor):
    """Geometry-aware baseline: encodes the element sequence of the structure.

    Each element token (type one-hot + 4 params + normalized position) is
    embedded by a shared MLP; tokens are mean+max pooled and decoded to the
    S-parameter vector. Justified by the data: every structure is a cascade
    of typed elements, so the model sees topology rather than a fixed slot
    vector, and can in principle transfer across families.
    """

    N_TYPES = 5
    MAX_EL = 8

    def _tokens(self, samples):
        d = self.N_TYPES + 5
        x = np.zeros((len(samples), self.MAX_EL, d))
        mask = np.zeros((len(samples), self.MAX_EL))
        for i, s in enumerate(samples):
            els = s["elements"]
            for j, el in enumerate(els[: self.MAX_EL]):
                t = int(el[0])
                x[i, j, t] = 1
                x[i, j, self.N_TYPES : self.N_TYPES + 4] = el[1:5]
                x[i, j, -1] = j / self.MAX_EL
                mask[i, j] = 1
        return x, mask

    def fit(self, train, val):
        torch.manual_seed(self.seed)
        xt_raw, mt = self._tokens(train)
        flat = xt_raw.reshape(-1, xt_raw.shape[-1])
        self.tok_norm = Normalizer().fit(flat[mt.reshape(-1) > 0])
        self.ynorm = Normalizer().fit(targets(train))

        def enc(samples):
            x, m = self._tokens(samples)
            return self.tok_norm(x) * m[..., None], m

        x, m = enc(train)
        xv, mv = enc(val)
        y = self.ynorm(targets(train))
        yv = self.ynorm(targets(val))
        din, dout = x.shape[-1], y.shape[1]
        h = self.cfg["hidden"]

        class DS(nn.Module):
            def __init__(self):
                super().__init__()
                self.phi = nn.Sequential(
                    nn.Linear(din, h), nn.ReLU(), nn.Linear(h, h), nn.ReLU()
                )
                self.rho = nn.Sequential(
                    nn.Linear(2 * h, h), nn.ReLU(), nn.Linear(h, dout)
                )

            def forward(self, x, m):
                e = self.phi(x) * m[..., None]
                mean = e.sum(1) / m.sum(1, keepdim=True).clamp(min=1)
                mx = e.masked_fill(m[..., None] == 0, -1e9).max(1).values
                return self.rho(torch.cat([mean, mx], -1))

        self.model = DS()
        self._enc = enc
        tt = lambda a: torch.tensor(a, dtype=torch.float32)
        xt, mt_, yt = tt(x), tt(m), tt(y)
        xvt, mvt, yvt = tt(xv), tt(mv), tt(yv)
        opt = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        best, best_state = np.inf, None
        n = len(xt)
        for ep in range(self.epochs):
            self.model.train()
            perm = torch.randperm(n)
            for i in range(0, n, 256):
                b = perm[i : i + 256]
                opt.zero_grad()
                loss = nn.functional.mse_loss(self.model(xt[b], mt_[b]), yt[b])
                loss.backward()
                opt.step()
            self.model.eval()
            with torch.no_grad():
                vl = nn.functional.mse_loss(self.model(xvt, mvt), yvt).item()
            if vl < best:
                best, best_state = vl, {
                    k: v.clone() for k, v in self.model.state_dict().items()
                }
        self.model.load_state_dict(best_state)
        return self

    def predict(self, samples, mc_dropout: int = 0):
        x, m = self._enc(samples)
        xt = torch.tensor(x, dtype=torch.float32)
        mt = torch.tensor(m, dtype=torch.float32)
        self.model.eval()
        with torch.no_grad():
            y = self.model(xt, mt).numpy() * self.ynorm.sd + self.ynorm.mu
        return vec_to_s(y, samples[0]["freq"].shape[0])


def timed_predict(model, samples):
    t0 = time.perf_counter()
    pred = model.predict(samples)
    dt = time.perf_counter() - t0
    return pred, dt / len(samples) * 1e3  # ms per sample
