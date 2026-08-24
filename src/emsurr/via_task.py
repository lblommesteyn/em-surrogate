"""Differential via-transition design task.

Geometry (um), fixed 4-layer stackup (er 3.5, 1000 um, planes at 300/700):
  d_via   drill diameter            200-400
  d_pad   pad diameter              d_via+150 .. d_via+350 (via p_pad)
  d_anti  antipad diameter          d_pad+150 .. d_pad+500 (via p_anti)
  s_via   differential via pitch    800-2000 (>= d_anti+100)
  w_trace trace width               150-400
  n_gnd   stitching-via count       2-8
  r_gnd   stitching ring radius     geometric minimum .. 2200

OBJECTIVE (declared before optimization), band 2-15 GHz on the mixed-mode
response under odd drive:
  J = max|Sdd11| + max(0, 0.7 - min|Sdd21|) + max(0, max(|Scd11|,|Scd21|) - 0.15)
invalid (non-finite or odd-drive power > 1.1) -> 2.0. The parameterization
is y-symmetric, so mode conversion is ~0 by construction; the Scd term
guards asymmetric numerical artifacts rather than driving the search.

Surrogate representation: the classic lumped via model mapped onto the
EXISTING token vocabulary - line(z0_odd) / shunt-C(pad capacitance) /
series-L(barrel + return-path loop inductance) / shunt-C / line. Zero new
primitive types; the via geometry enters through declared physics formulas
(pad capacitance a la Howard-Johnson, loop inductance with ground-ring
term). The matching train-side family `via_model` is generated and solved
by the same analytic ABCD engine.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np

from . import physics
from .synth import EL_LINE, EL_SHUNT_C, EL_SERIES_L

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "results" / "via_cache"
OEMS_PY = ROOT / "tools" / "oems-venv" / "Scripts" / "python.exe"
EVAL_SCRIPT = ROOT / "scripts" / "oems_via_eval.py"

BOUNDS = dict(d_via=(200, 400), p_pad=(150, 350), p_anti=(150, 500),
              s_via=(800, 2000), w_trace=(150, 400), n_gnd=(2, 8),
              r_gnd=(600, 2200))
KEYS = list(BOUNDS)
EPR, SUB_T = 3.5, 1000e-6
H_REF = 300e-6            # trace-to-plane height on both outer layers
BAND = (2e9, 15e9)
LTR = 2500e-6
MU0, EPS0 = 4e-7 * np.pi, 8.854e-12


def clip(d):
    d = {k: float(np.clip(d[k], *BOUNDS[k])) for k in KEYS}
    d["n_gnd"] = float(int(round(d["n_gnd"])))
    dd = derived(d)
    if d["s_via"] < dd["d_anti"] + 100:
        d["s_via"] = min(dd["d_anti"] + 100, BOUNDS["s_via"][1])
    rmin = d["s_via"] / 2 + dd["d_anti"] / 2 + d["d_via"] / 2 + 150
    if d["r_gnd"] < rmin:
        d["r_gnd"] = min(rmin, BOUNDS["r_gnd"][1])
    return d


def derived(d):
    return dict(d_pad=d["d_via"] + d["p_pad"],
                d_anti=d["d_via"] + d["p_pad"] + d["p_anti"])


def geometry(d):
    """Full physical geometry dict for the solver."""
    dd = derived(d)
    return dict(d_via=d["d_via"], d_pad=dd["d_pad"], d_anti=dd["d_anti"],
                s_via=d["s_via"], w_trace=d["w_trace"],
                n_gnd=d["n_gnd"], r_gnd=d["r_gnd"])


def random_design(rng):
    return clip({k: rng.uniform(*BOUNDS[k]) for k in KEYS})


def key(d):
    return hashlib.blake2b(json.dumps({k: round(d[k], 1) for k in KEYS},
                                      sort_keys=True).encode(),
                           digest_size=12).hexdigest()


def via_lumped(d):
    """Declared physics mapping geometry -> (C_pad, L_via)."""
    g = geometry(d)
    dp, da, dv = g["d_pad"] * 1e-6, g["d_anti"] * 1e-6, g["d_via"] * 1e-6
    # pad-to-antipad coaxial capacitance across the two plane layers
    c_pad = 2 * np.pi * EPS0 * EPR * SUB_T * 0.5 / np.log(da / dp)
    # barrel + return-loop inductance; ground ring shortens the loop
    h = SUB_T
    l_via = MU0 * h / (2 * np.pi) * (
        np.log(2 * g["s_via"] * 1e-6 / dv)
        + (2.0 / max(g["n_gnd"], 1)) * np.log(2 * g["r_gnd"] * 1e-6 / dv))
    return c_pad, l_via


def to_sample(d, freq):
    w = d["w_trace"] * 1e-6
    z0, eeff = physics.microstrip_z0_eeff(w, H_REF, EPR)
    c_pad, l_via = via_lumped(d)
    els = np.array([
        (EL_LINE, z0, eeff, 0.02, LTR),
        (EL_SHUNT_C, c_pad, 0, 0, 0),
        (EL_SERIES_L, l_via, 0, 0, 0),
        (EL_SHUNT_C, c_pad, 0, 0, 0),
        (EL_LINE, z0, eeff, 0.02, LTR),
    ])
    return dict(sample_id=key(d), topology_family="via_design", ports=2,
                params=np.zeros(1), elements=els, freq=freq,
                s=np.zeros((len(freq), 2, 2), complex))


def objective(s, freq):
    m = (freq >= BAND[0]) & (freq <= BAND[1])
    if not np.all(np.isfinite(s[m])):
        return 2.0
    sdd11 = np.abs(s[m, 0, 0]).max()
    sdd21 = np.abs(s[m, 1, 0]).min()
    scd = max(np.abs(s[m, 0, 1]).max(), np.abs(s[m, 1, 1]).max())
    return float(sdd11 + max(0.0, 0.7 - sdd21) + max(0.0, scd - 0.15))


def surrogate_objective(s2, freq):
    """Objective from a 2-port surrogate prediction (dd-mode only; the
    surrogate cannot see mode conversion, which is ~0 by symmetry)."""
    m = (freq >= BAND[0]) & (freq <= BAND[1])
    if not np.all(np.isfinite(s2[m])):
        return 2.0
    s11 = np.abs(s2[m, 0, 0]).max()
    s21 = np.abs(s2[m, 1, 0]).min()
    return float(s11 + max(0.0, 0.7 - s21))


class Solver:
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
            tmp_in.write_text(json.dumps([geometry(d) for d in todo]))
            r = subprocess.run([str(OEMS_PY), str(EVAL_SCRIPT), str(tmp_in),
                                str(tmp_out)], capture_output=True, text=True,
                               cwd=str(ROOT))
            if r.returncode != 0:
                raise RuntimeError("oems_via_eval failed:\n" + r.stdout[-2000:]
                                   + r.stderr[-2000:])
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


def make_via_model_family(n=400, seed=11):
    """Analytic train-side family matching the token mapping: solved
    line - shuntC - seriesL - shuntC - line structures over the mapped
    parameter ranges."""
    rng = np.random.default_rng(seed)
    freq = np.linspace(50e6, 20e9, 256)
    out = []
    for _ in range(n):
        d = random_design(rng)
        w = d["w_trace"] * 1e-6
        z0, eeff = physics.microstrip_z0_eeff(w, H_REF, EPR)
        c_pad, l_via = via_lumped(d)
        gamma = physics.line_gamma(freq, eeff, z0, 0.02, 5.8e7, w, 35e-6)
        ln = physics.line_abcd(gamma, z0, LTR)
        wr = 2 * np.pi * freq
        ab = np.einsum("fij,fjk->fik", ln, physics.shunt_abcd(1j * wr * c_pad))
        ab = np.einsum("fij,fjk->fik", ab, physics.series_abcd(1j * wr * l_via))
        ab = np.einsum("fij,fjk->fik", ab, physics.shunt_abcd(1j * wr * c_pad))
        ab = np.einsum("fij,fjk->fik", ab, ln)
        s = physics.abcd_to_s(ab, 50.0)
        smp = to_sample(d, freq)
        smp["s"] = s.astype(complex)
        smp["sample_id"] = f"viamodel_{rng.integers(1 << 60)}"
        smp["topology_family"] = "via_model"
        out.append(smp)
    return out
