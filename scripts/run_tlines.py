"""Dataset B: measured reverberation-chamber transmission-line networks as an
external measured-physics test.

The frozen surrogate's input representation (synthetic-family design params +
element tokens) cannot describe these physical structures, so no prediction is
attempted; that inapplicability is itself a finding. What we test:

1. physics of the measurements: reciprocity, passivity, finiteness
2. response-space novelty: do the measured S-responses sit outside the
   frozen synthetic benchmark's response support? kNN distance in a shared
   band-limited |S| feature space, referenced against the synthetic ID
   test split's own distances.

Outputs results/tlines_metrics.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from emsurr import tlines
from emsurr.dataset import load_h5, passivity_violations, reciprocity_error
from emsurr.novelty import auroc

BAND = (2.0e8, 1.0e9)  # overlap of tlines (0.2-1 GHz) and synthetic (0.05-20 GHz)
NPTS = 64


def band_features(freq: np.ndarray, s: np.ndarray) -> np.ndarray:
    """Log-magnitude S11/S21/S22 resampled to NPTS points in BAND."""
    grid = np.linspace(*BAND, NPTS)
    out = []
    for (i, j) in [(0, 0), (1, 0), (1, 1)]:
        mag = 20 * np.log10(np.abs(s[:, i, j]) + 1e-12)
        out.append(np.interp(grid, freq, mag))
    return np.concatenate(out)


def knn_dist(q: np.ndarray, ref: np.ndarray, k: int = 3) -> np.ndarray:
    d = np.linalg.norm(q[:, None] - ref[None], axis=-1)
    return np.sort(d, 1)[:, :k].mean(1)


def main():
    res = {}

    # ---- measured physics checks (per configuration, sampled angles)
    angle_sub = list(range(0, 72, 8))  # 9 angles per config; angles are replicates
    phys = {}
    meas_feats, meas_ids = [], []
    for name in tlines.configs():
        cfg = tlines.load_config(name)
        recs, passv, finite = [], [], True
        for a in angle_sub:
            s = tlines.s_at_angle(cfg, a)
            finite &= bool(np.isfinite(s).all())
            recs.append(reciprocity_error(s))
            passv.append(passivity_violations(s, tol=1e-3))
            meas_feats.append(band_features(cfg["freq"], s))
            meas_ids.append(f"{name}@{cfg['alphas'][a]:.0f}deg")
        phys[name] = dict(
            ports=int(cfg["s"].shape[0]),
            n_angles=int(cfg["s"].shape[2]),
            n_freq=int(cfg["s"].shape[3]),
            finite=finite,
            reciprocity_err_max=float(np.max(recs)),
            passivity_violation_freqs_max=int(np.max(passv)),
            passivity_violation_freqs_mean=float(np.mean(passv)),
        )
        del cfg
    res["physics"] = phys
    meas_feats = np.array(meas_feats)

    # ---- response-space novelty vs the frozen synthetic benchmark
    synth = load_h5("data/processed/synth.h5")
    rng = np.random.default_rng(0)
    idx = rng.permutation(len(synth))
    ref_idx, id_idx = idx[:2000], idx[2000:2400]
    feats = lambda ss: np.array([band_features(s["freq"], s["s"]) for s in ss])
    ref = feats([synth[i] for i in ref_idx])
    id_q = feats([synth[i] for i in id_idx])
    mu, sd = ref.mean(0), ref.std(0) + 1e-9
    nz = lambda x: (x - mu) / sd
    d_id = knn_dist(nz(id_q), nz(ref))
    d_meas = knn_dist(nz(meas_feats), nz(ref))
    res["response_novelty"] = dict(
        n_ref=len(ref), n_id=len(d_id), n_meas=len(d_meas),
        auroc_meas_vs_id=auroc(d_id, d_meas),
        id_dist_p95=float(np.quantile(d_id, 0.95)),
        meas_dist_min=float(d_meas.min()),
        meas_dist_median=float(np.median(d_meas)),
        frac_meas_above_id_p95=float((d_meas > np.quantile(d_id, 0.95)).mean()),
    )
    res["note"] = (
        "No surrogate prediction attempted: the frozen input representation "
        "cannot encode these physical structures. Angles/frequencies treated "
        "as replicates of 14 structures, never as independent samples."
    )

    out = Path("results/tlines_metrics.json")
    out.write_text(json.dumps(res, indent=1))
    print(json.dumps(res["response_novelty"], indent=1))
    print("wrote", out)


if __name__ == "__main__":
    main()
