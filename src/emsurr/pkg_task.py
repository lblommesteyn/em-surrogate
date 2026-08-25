"""Package/interposer-style channel design task (two coupled via
transitions, asymmetric geometry).

Parameters (um): d_via 200-400; p_pad 150-350; p_anti 150-500 (oversizes);
s1, s2 pitches 800-1600; dy2 lateral offset of the second transition 0-600
(asymmetry: unequal jog lengths for the two lines -> real mode conversion);
l_mid 1500-4000 inter-via routing; w_top, w_bot 150-400; n_gnd 2-8;
r_gnd geometric-min..2200; stub1 50-250 (below M, plane clearance to P2);
stub2 50-300 (above M, clearance to P1).

OBJECTIVE (declared before optimization), band 2-15 GHz, odd drive:
  J = max|Sdd11| + max(0, 0.7 - min|Sdd21|)
      + max(0, max(|Scd11|,|Scd21|) - 0.15)
invalid -> 2.0. Unlike the single-via task the Scd term is ACTIVE here
(prototype: |Scd21| up to 0.12 from dy2 asymmetry).

Representation: ordered 9-token chain on the EXISTING vocabulary -
line(top) / shuntC(2*C1) / seriesL(L1) / open-stub(via stub 1) /
line(mid, length includes the jog) / shuntC(2*C2) / seriesL(L2) /
open-stub(via stub 2) / line(bottom). Via stubs map to EL_STUB_OPEN with
the barrel-in-antipad coax impedance. The only representation extension is
token capacity (MAX_EL 8 -> 12) for ordered multi-via chains; zero new
types, no family IDs. Matching analytic train-side family: `chain_model`.
The surrogate is dd-only: mode conversion is invisible to it by
construction - a declared blind spot that verification must catch.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np

from . import physics
from .synth import EL_LINE, EL_SHUNT_C, EL_SERIES_L, EL_STUB_OPEN

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "results" / "pkg_cache"
OEMS_PY = ROOT / "tools" / "oems-venv" / "Scripts" / "python.exe"
EVAL_SCRIPT = ROOT / "scripts" / "oems_pkg_eval.py"

BOUNDS = dict(d_via=(200, 400), p_pad=(150, 350), p_anti=(150, 500),
              s1=(800, 1600), s2=(800, 1600), dy2=(0, 600),
              l_mid=(1500, 4000), w_top=(150, 400), w_bot=(150, 400),
              n_gnd=(2, 8), r_gnd=(600, 2200), stub1=(50, 250),
              stub2=(50, 300))
KEYS = list(BOUNDS)
EPR = 3.5
SUB_T = 1400e-6
H_TOP = 300e-6
H_MID = 300e-6
BAND = (2e9, 15e9)
LTR = 2200e-6
MU0, EPS0 = 4e-7 * np.pi, 8.854e-12
H_VIA1, H_VIA2 = 700e-6, 700e-6


def derived(d):
    return dict(d_pad=d["d_via"] + d["p_pad"],
                d_anti=d["d_via"] + d["p_pad"] + d["p_anti"])


def clip(d):
    d = {k: float(np.clip(d[k], *BOUNDS[k])) for k in KEYS}
    d["n_gnd"] = float(int(round(d["n_gnd"])))
    dd = derived(d)
    for sk in ("s1", "s2"):
        if d[sk] < dd["d_anti"] + 100:
            d[sk] = min(dd["d_anti"] + 100, BOUNDS[sk][1])
    rmin = max(d["s1"], d["s2"]) / 2 + dd["d_anti"] / 2 + d["d_via"] / 2 + 150
    if d["r_gnd"] < rmin:
        d["r_gnd"] = min(rmin, BOUNDS["r_gnd"][1])
    return d


def geometry(d):
    dd = derived(d)
    return dict(d_via=d["d_via"], d_pad=dd["d_pad"], d_anti=dd["d_anti"],
                s1=d["s1"], s2=d["s2"], dy2=d["dy2"], l_mid=d["l_mid"],
                w_top=d["w_top"], w_bot=d["w_bot"], n_gnd=d["n_gnd"],
                r_gnd=d["r_gnd"], stub1=d["stub1"], stub2=d["stub2"])


def random_design(rng):
    return clip({k: rng.uniform(*BOUNDS[k]) for k in KEYS})


def key(d):
    return hashlib.blake2b(json.dumps({k: round(d[k], 1) for k in KEYS},
                                      sort_keys=True).encode(),
                           digest_size=12).hexdigest()


def lumped(d):
    g = geometry(d)
    dp_, da, dv = g["d_pad"] * 1e-6, g["d_anti"] * 1e-6, g["d_via"] * 1e-6
    c1 = 2 * np.pi * EPS0 * EPR * H_VIA1 * 0.5 / np.log(da / dp_)
    c2 = 2 * np.pi * EPS0 * EPR * H_VIA2 * 0.5 / np.log(da / dp_)

    def loop(s, extra):
        return MU0 * H_VIA1 / (2 * np.pi) * (
            np.log(2 * s * 1e-6 / dv)
            + (2.0 / max(g["n_gnd"], 1)) * np.log(2 * g["r_gnd"] * 1e-6 / dv)
            + extra)

    l1 = loop(g["s1"], 0.0)
    l2 = loop(g["s2"], 0.5 * g["dy2"] / 1000.0)
    z_stub = 60.0 / np.sqrt(EPR) * np.log(da / dv)
    return c1, l1, c2, l2, z_stub


def to_sample(d, freq):
    g = geometry(d)
    wt, wb = g["w_top"] * 1e-6, g["w_bot"] * 1e-6
    z_t, e_t = physics.microstrip_z0_eeff(wt, H_TOP, EPR)
    z_m, e_m = physics.microstrip_z0_eeff(wt, H_MID, EPR)
    z_b, e_b = physics.microstrip_z0_eeff(wb, H_TOP, EPR)
    c1, l1, c2, l2, z_stub = lumped(d)
    lmid = (g["l_mid"] + abs(g["dy2"])) * 1e-6
    els = np.array([
        (EL_LINE, z_t, e_t, 0.02, LTR),
        (EL_SHUNT_C, 2 * c1, 0, 0, 0),
        (EL_SERIES_L, l1, 0, 0, 0),
        (EL_STUB_OPEN, z_stub, EPR, 0.02, g["stub1"] * 1e-6),
        (EL_LINE, z_m, e_m, 0.02, lmid),
        (EL_SHUNT_C, 2 * c2, 0, 0, 0),
        (EL_SERIES_L, l2, 0, 0, 0),
        (EL_STUB_OPEN, z_stub, EPR, 0.02, g["stub2"] * 1e-6),
        (EL_LINE, z_b, e_b, 0.02, LTR),
    ])
    return dict(sample_id=key(d), topology_family="pkg_design", ports=2,
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
    m = (freq >= BAND[0]) & (freq <= BAND[1])
    if not np.all(np.isfinite(s2[m])):
        return 2.0
    return float(np.abs(s2[m, 0, 0]).max()
                 + max(0.0, 0.7 - np.abs(s2[m, 1, 0]).min()))


class Solver:
    def __init__(self):
        CACHE.mkdir(parents=True, exist_ok=True)
        self.calls = 0
        self.wall = 0.0

    def _path(self, d):
        return CACHE / f"{key(d)}.npz"

    def cached(self, d):
        return self._path(d).exists()

    def _run_subset(self, todo):
        tmp_in = CACHE / "_batch_in.json"
        tmp_out = CACHE / "_batch_out.npz"
        tmp_in.write_text(json.dumps([geometry(d) for d in todo]))
        r = subprocess.run([str(OEMS_PY), str(EVAL_SCRIPT), str(tmp_in),
                            str(tmp_out)], capture_output=True, text=True,
                           cwd=str(ROOT))
        if r.returncode != 0:
            return False
        z = np.load(tmp_out)
        freq = z["freq"]
        for i, d in enumerate(todo):
            meta = z[f"meta_{i}"]
            np.savez(self._path(d), s=z[str(i)], freq=freq,
                     wall=meta[0], finite=meta[1], max_sv=meta[2])
            self.calls += 1
            self.wall += float(meta[0])
        return True

    def solve_batch(self, designs):
        todo = [d for d in designs if not self.cached(d)]
        if todo and not self._run_subset(todo):
            # a native openEMS crash kills the whole subprocess; isolate the
            # crashing design(s) one per subprocess, marking crashers invalid
            n_freq = 96
            for d in todo:
                if self.cached(d):
                    continue
                if not self._run_subset([d]):
                    np.savez(self._path(d),
                             s=np.full((n_freq, 2, 2), np.nan, complex),
                             freq=np.linspace(1e9, 20e9, n_freq),
                             wall=0.0, finite=0.0, max_sv=9.9)
                    self.calls += 1
        out = []
        for d in designs:
            z = np.load(self._path(d))
            s, freq = z["s"], z["freq"]
            ok = bool(z["finite"]) and float(z["max_sv"]) <= 1.1
            out.append(dict(s=s, freq=freq, ok=ok,
                            J=objective(s, freq) if ok else 2.0))
        return out


def make_chain_model_family(n=400, seed=13):
    """Analytic train-side family matching the 9-token chain mapping."""
    rng = np.random.default_rng(seed)
    freq = np.linspace(50e6, 20e9, 256)
    out = []
    for _ in range(n):
        d = random_design(rng)
        smp = to_sample(d, freq)
        els = smp["elements"]
        wr = 2 * np.pi * freq
        ab = None
        for el in els:
            t = int(el[0])
            if t == EL_LINE:
                gam = physics.line_gamma(freq, el[2], el[1], el[3], 5.8e7,
                                         200e-6, 35e-6)
                blk = physics.line_abcd(gam, el[1], el[4])
            elif t == EL_SHUNT_C:
                blk = physics.shunt_abcd(1j * wr * el[1])
            elif t == EL_SERIES_L:
                blk = physics.series_abcd(1j * wr * el[1])
            elif t == EL_STUB_OPEN:
                gam = physics.line_gamma(freq, el[2], el[1], el[3], 5.8e7,
                                         200e-6, 35e-6)
                z_in = el[1] / np.tanh(gam * el[4])
                blk = physics.shunt_abcd(1.0 / z_in)
            ab = blk if ab is None else np.einsum("fij,fjk->fik", ab, blk)
        s = physics.abcd_to_s(ab, 50.0)
        smp["s"] = s.astype(complex)
        smp["sample_id"] = f"chainmodel_{rng.integers(1 << 60)}"
        smp["topology_family"] = "chain_model"
        out.append(smp)
    return out
