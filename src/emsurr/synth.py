"""Synthetic multi-family PCB interconnect dataset (stand-in for TUHH data).

Seven topology families, each a cascade of analytic elements. Every sample
records topology_family, the raw design parameters, the element-sequence
representation (for geometry-aware models), and complex 2-port S-parameters
on a common frequency grid.
"""

from __future__ import annotations

import numpy as np

from . import physics as ph

FREQ = np.linspace(0.05e9, 20e9, 256)
ZREF = 50.0

# element token: (type_id, params[4]) -- type ids
EL_LINE = 0  # z0, eeff, tand-scaled loss, length
EL_SHUNT_C = 1
EL_SERIES_L = 2
EL_STUB_OPEN = 3
EL_STUB_SHORT = 4

# shared parameter vector layout (fixed length, family-specific meaning; unused = 0)
PARAM_NAMES = [
    "w1", "w2", "h", "er", "tand", "len1", "len2", "len3", "t", "sigma",
    "stub_len", "via_l", "via_c",
]

FAMILIES = [
    "mstrip_line",
    "sline_line",
    "stub_open",
    "stub_short",
    "stepped",
    "via_lc",
    "mismatch",
]


def _mstrip(rng):
    return dict(
        w=rng.uniform(0.1e-3, 0.6e-3),
        h=rng.uniform(0.1e-3, 0.4e-3),
        er=rng.uniform(2.2, 4.8),
        tand=rng.uniform(0.001, 0.025),
        t=35e-6,
        sigma=5.8e7,
    )


def _line_piece(f, w, h, er, tand, t, sigma, length, strip=False):
    if strip:
        z0 = ph.stripline_z0(w, 2 * h, er)
        eeff = er
    else:
        z0, eeff = ph.microstrip_z0_eeff(w, h, er)
    gamma = ph.line_gamma(f, eeff, z0, tand, sigma, w, t)
    return ph.line_abcd(gamma, z0, length), z0, eeff, gamma


def make_sample(family: str, rng: np.random.Generator) -> dict:
    f = FREQ
    m = _mstrip(rng)
    p = {k: 0.0 for k in PARAM_NAMES}
    p.update(w1=m["w"], h=m["h"], er=m["er"], tand=m["tand"], t=m["t"], sigma=m["sigma"])
    elements = []  # list of (type_id, p1..p4)

    def line_el(w, length, strip=False):
        abcd, z0, eeff, _ = _line_piece(
            f, w, m["h"], m["er"], m["tand"], m["t"], m["sigma"], length, strip
        )
        elements.append((EL_LINE, z0, eeff, m["tand"], length))
        return abcd

    if family == "mstrip_line":
        p["len1"] = rng.uniform(2e-3, 40e-3)
        abcd = line_el(m["w"], p["len1"])
    elif family == "sline_line":
        p["len1"] = rng.uniform(2e-3, 40e-3)
        abcd = line_el(m["w"], p["len1"], strip=True)
    elif family in ("stub_open", "stub_short"):
        p["len1"] = rng.uniform(2e-3, 15e-3)
        p["len2"] = rng.uniform(2e-3, 15e-3)
        p["stub_len"] = rng.uniform(1e-3, 12e-3)
        p["w2"] = rng.uniform(0.1e-3, 0.6e-3)
        a1 = line_el(m["w"], p["len1"])
        _, z0s, eeffs, gs = _line_piece(
            f, p["w2"], m["h"], m["er"], m["tand"], m["t"], m["sigma"], p["stub_len"]
        )
        gl = gs * p["stub_len"]
        if family == "stub_open":
            y = np.tanh(gl) / z0s
            elements.append((EL_STUB_OPEN, z0s, eeffs, m["tand"], p["stub_len"]))
        else:
            y = 1.0 / (z0s * np.tanh(gl))
            elements.append((EL_STUB_SHORT, z0s, eeffs, m["tand"], p["stub_len"]))
        a2 = ph.shunt_abcd(y)
        a3 = line_el(m["w"], p["len2"])
        abcd = ph.cascade(a1, a2, a3)
    elif family == "stepped":
        p["w2"] = rng.uniform(0.05e-3, 1.2e-3)
        p["len1"] = rng.uniform(2e-3, 12e-3)
        p["len2"] = rng.uniform(2e-3, 12e-3)
        p["len3"] = rng.uniform(2e-3, 12e-3)
        abcd = ph.cascade(
            line_el(m["w"], p["len1"]),
            line_el(p["w2"], p["len2"]),
            line_el(m["w"], p["len3"]),
        )
    elif family == "via_lc":
        p["len1"] = rng.uniform(2e-3, 20e-3)
        p["len2"] = rng.uniform(2e-3, 20e-3)
        p["via_l"] = rng.uniform(0.1e-9, 1.2e-9)
        p["via_c"] = rng.uniform(0.05e-12, 0.8e-12)
        w_ang = 2 * np.pi * f
        a1 = line_el(m["w"], p["len1"])
        yc = 1j * w_ang * p["via_c"] / 2
        elements.append((EL_SHUNT_C, p["via_c"] / 2, 0, 0, 0))
        elements.append((EL_SERIES_L, p["via_l"], 0, 0, 0))
        elements.append((EL_SHUNT_C, p["via_c"] / 2, 0, 0, 0))
        avia = ph.cascade(
            ph.shunt_abcd(yc), ph.series_abcd(1j * w_ang * p["via_l"]), ph.shunt_abcd(yc)
        )
        a2 = line_el(m["w"], p["len2"])
        abcd = ph.cascade(a1, avia, a2)
    elif family == "mismatch":
        p["w2"] = rng.uniform(0.05e-3, 1.2e-3)
        p["len1"] = rng.uniform(2e-3, 25e-3)
        p["len2"] = rng.uniform(2e-3, 25e-3)
        abcd = ph.cascade(line_el(m["w"], p["len1"]), line_el(p["w2"], p["len2"]))
    else:
        raise ValueError(family)

    s = ph.abcd_to_s(abcd, ZREF)
    return dict(
        topology_family=family,
        params=np.array([p[k] for k in PARAM_NAMES]),
        elements=np.array(elements, dtype=float),  # (n_el, 5)
        freq=f,
        s=s,  # (F, 2, 2) complex
        ports=2,
    )


def generate(n_per_family: int, seed: int = 0) -> list[dict]:
    rng = np.random.default_rng(seed)
    out = []
    for fam in FAMILIES:
        for i in range(n_per_family):
            smp = make_sample(fam, rng)
            smp["sample_id"] = f"{fam}_{i:05d}"
            out.append(smp)
    return out
