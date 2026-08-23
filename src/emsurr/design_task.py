"""Design task for the application milestone: a 2-port microstrip
interconnect with two mandatory open stubs and a mandatory series gap, on a
fixed stackup (254 um substrate, er=3.5, the frozen openEMS stackup).

Parameters (um): w in [400,800] line width; s1,s2 in [3000,10000] stub
lengths; sw in [300,700] stub width; x1,x2 in [-8000,-1500] stub positions
(x1 < x2 - 800, both left of the gap); g in [300,700] gap length.

OBJECTIVE (declared before any optimization): passband 2-6 GHz,
  J = max|S11| + max(0, 0.5 - min|S21|)   over the band
lower is better; invalid/failed designs score 2.0. The gap makes low-band
transmission hard; the stubs are the matching degrees of freedom but also
resonators - a genuine tuning problem.

Surrogate-side representation: the extended token vocabulary
(line / stub_open / series-gap), cascade-ordered left to right, evaluated
by a frozen-size DeepSets ensemble; risk by the frozen retrieval-gap +
deployment-switch machinery. openEMS is the authoritative evaluator; every
solver result is content-cached.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np

from . import physics
from .synth_ext import EL_SERIES_C
from .synth import EL_LINE, EL_STUB_OPEN

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "results" / "design_cache"
OEMS_PY = ROOT / "tools" / "oems-venv" / "Scripts" / "python.exe"
EVAL_SCRIPT = ROOT / "scripts" / "oems_eval.py"

BOUNDS = dict(w=(400, 800), s1=(3000, 10000), s2=(3000, 10000), sw=(300, 700),
              x1=(-8000, -2300), x2=(-7200, -1500), g=(300, 700))
KEYS = list(BOUNDS)
EPR, SUB_T, MSL_LEN = 3.5, 254e-6, 18000.0
BAND = (2e9, 6e9)


def clip(d):
    d = {k: float(np.clip(d[k], *BOUNDS[k])) for k in KEYS}
    if d["x2"] < d["x1"] + 800:
        d["x2"] = min(d["x1"] + 800, BOUNDS["x2"][1])
    return d


def random_design(rng):
    d = {k: rng.uniform(*BOUNDS[k]) for k in KEYS}
    return clip(d)


def key(d):
    return hashlib.blake2b(json.dumps({k: round(d[k], 1) for k in KEYS},
                                      sort_keys=True).encode(),
                           digest_size=12).hexdigest()


def to_sample(d, freq):
    """Token cascade for the surrogate/encoder (left -> right)."""
    w, sw = d["w"] * 1e-6, d["sw"] * 1e-6
    z0, eeff = physics.microstrip_z0_eeff(w, SUB_T, EPR)
    z0s, eeffs = physics.microstrip_z0_eeff(sw, SUB_T, EPR)
    L = MSL_LEN * 1e-6
    x1, x2, g = d["x1"] * 1e-6, d["x2"] * 1e-6, d["g"] * 1e-6
    els = [
        (EL_LINE, z0, eeff, 0.02, x1 + L),
        (EL_STUB_OPEN, z0s, eeffs, 0.02, d["s1"] * 1e-6),
        (EL_LINE, z0, eeff, 0.02, x2 - x1),
        (EL_STUB_OPEN, z0s, eeffs, 0.02, d["s2"] * 1e-6),
        (EL_LINE, z0, eeff, 0.02, -x2 - g / 2),
        (EL_SERIES_C, z0, eeff, g, w),
        (EL_LINE, z0, eeff, 0.02, L - g / 2),
    ]
    return dict(sample_id=key(d), topology_family="design", ports=2,
                params=np.zeros(1), elements=np.array(els), freq=freq,
                s=np.zeros((len(freq), 2, 2), complex))


def objective(s, freq):
    m = (freq >= BAND[0]) & (freq <= BAND[1])
    if not np.all(np.isfinite(s[m])):
        return 2.0
    s11 = np.abs(s[m, 0, 0]).max()
    s21 = np.abs(s[m, 1, 0]).min()
    return float(s11 + max(0.0, 0.5 - s21))


class Solver:
    """Batch, content-cached openEMS evaluation via the tools venv."""

    def __init__(self):
        CACHE.mkdir(parents=True, exist_ok=True)
        self.calls = 0
        self.wall = 0.0

    def _path(self, d):
        return CACHE / f"{key(d)}.npz"

    def cached(self, d):
        return self._path(d).exists()

    def solve_batch(self, designs):
        todo = [d for d in designs if not self.cached(d)]
        if todo:
            tmp_in = CACHE / "_batch_in.json"
            tmp_out = CACHE / "_batch_out.npz"
            tmp_in.write_text(json.dumps(todo))
            r = subprocess.run([str(OEMS_PY), str(EVAL_SCRIPT), str(tmp_in),
                                str(tmp_out)], capture_output=True, text=True,
                               cwd=str(ROOT))
            if r.returncode != 0:
                raise RuntimeError("oems_eval failed:\n" + r.stdout[-2000:] +
                                   r.stderr[-2000:])
            z = np.load(tmp_out)
            freq = z["freq"]
            for i, d in enumerate(todo):
                meta = z[f"meta_{i}"]
                np.savez(self._path(d), s=z[str(i)], freq=freq,
                         wall=meta[0], finite=meta[1], max_sv=meta[2])
                self.calls += 1
                self.wall += float(meta[0])
        out = []
        for d in designs:
            z = np.load(self._path(d))
            s, freq = z["s"], z["freq"]
            ok = bool(z["finite"]) and float(z["max_sv"]) <= 1.1
            out.append(dict(s=s, freq=freq, ok=ok,
                            J=objective(s, freq) if ok else 2.0))
        return out
