"""Dataset C: IEEE P370 briefcase test cases as a Touchstone-ingestion and
physics sanity benchmark. 14 files (fixtures, 2x-thrus, de-embedded DUTs,
fixture models). Small set: sanity checks and novelty behavior only, no ML
claims. Outputs results/p370_metrics.json."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import skrf

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from emsurr.dataset import load_h5, passivity_violations, reciprocity_error
from emsurr.novelty import auroc

ROOT = Path("data/external/p370/briefcase")
BAND = (5.0e7, 2.0e10)
NPTS = 64

# "model"/"AFR"/"dmbd" files are synthesized or algorithm outputs; the rest are
# VNA measurements of the physical briefcase structures.
SYNTH_MARKERS = ("model", "afr", "dmbd")


def band_features(freq, s):
    grid = np.geomspace(*BAND, NPTS)
    out = []
    for (i, j) in [(0, 0), (1, 0), (1, 1)]:
        mag = 20 * np.log10(np.abs(s[:, i, j]) + 1e-12)
        out.append(np.interp(grid, freq, mag))
    return np.concatenate(out)


def main():
    res = {"files": {}}
    feats, kinds, names = [], [], []
    for p in sorted(ROOT.rglob("*.s*p")):
        nw = skrf.Network(str(p))
        s = nw.s
        kind = "synthesized" if any(m in p.name.lower() for m in SYNTH_MARKERS) else "measured"
        res["files"][str(p.relative_to(ROOT))] = dict(
            kind=kind,
            ports=int(s.shape[1]),
            n_freq=int(s.shape[0]),
            f_min_ghz=float(nw.f[0] / 1e9),
            f_max_ghz=float(nw.f[-1] / 1e9),
            finite=bool(np.isfinite(s).all()),
            reciprocity_err_max=reciprocity_error(s),
            passivity_violation_freqs=passivity_violations(s, tol=1e-3),
        )
        feats.append(band_features(nw.f, s))
        kinds.append(kind)
        names.append(p.name)
    feats = np.array(feats)

    # response-space novelty vs frozen synthetic benchmark (same recipe as tlines)
    synth = load_h5("data/processed/synth.h5")
    rng = np.random.default_rng(0)
    idx = rng.permutation(len(synth))
    fs = lambda ss: np.array([band_features(x["freq"], x["s"]) for x in ss])
    ref, id_q = fs([synth[i] for i in idx[:2000]]), fs([synth[i] for i in idx[2000:2400]])
    mu, sd = ref.mean(0), ref.std(0) + 1e-9
    nz = lambda x: (x - mu) / sd
    knn = lambda q: np.sort(np.linalg.norm(nz(q)[:, None] - nz(ref)[None], axis=-1), 1)[:, :3].mean(1)
    d_id, d_p = knn(id_q), knn(feats)
    res["response_novelty"] = dict(
        auroc_p370_vs_id=auroc(d_id, d_p),
        id_dist_p95=float(np.quantile(d_id, 0.95)),
        frac_above_id_p95=float((d_p > np.quantile(d_id, 0.95)).mean()),
        per_file={n: float(d) for n, d in zip(names, d_p)},
    )

    out = Path("results/p370_metrics.json")
    out.write_text(json.dumps(res, indent=1))
    print(json.dumps(res["response_novelty"], indent=1)[:800])
    print("wrote", out)


if __name__ == "__main__":
    main()
