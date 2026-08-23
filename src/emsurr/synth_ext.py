"""Vocabulary extension: series-gap primitive and an analytically solved
training family that exercises it.

EL_SERIES_C (type id 5) models a microstrip series gap as a capacitive pi
network (series Cs with two shunt Cp), the standard quasi-static gap model.
The new family `gapped_line` (line - gap - line) is generated and solved by
the SAME analytic ABCD engine as the frozen families - train-side solver
outputs, no openEMS data, no family IDs encoded in tokens.

Token layout matches the frozen vocabulary: (type, p1, p2, p3, p4) with
p1=z0 of the adjacent line, p2=eeff, p3=gap length (m), p4=line width (m).
The frozen synthetic dataset and benchmark are NOT modified; this family
exists only as additional encoder training data.
"""

from __future__ import annotations

import numpy as np

from . import physics
from .synth import EL_LINE, _mstrip

EL_SERIES_C = 5
EPS0 = 8.854e-12
FREQ = np.linspace(50e6, 20e9, 256)


def gap_caps(w: float, h: float, er: float, g: float) -> tuple[float, float]:
    """Quasi-static microstrip gap: series and shunt capacitance (crude
    closed-form; monotone in the right variables, which is what retrieval
    needs)."""
    cs = 0.5 * EPS0 * er * w * h / max(g, 1e-6)          # series coupling
    cp = 0.2 * EPS0 * er * w * (1.0 + g / max(w, 1e-6))  # fringing to ground
    return cs, cp


def gap_abcd(f: np.ndarray, cs: float, cp: float) -> np.ndarray:
    w_ = 2 * np.pi * f
    a = physics.shunt_abcd(1j * w_ * cp)
    b = physics.series_abcd(1.0 / (1j * w_ * cs))
    return np.einsum("fij,fjk,fkl->fil", a, b, a)


def make_gapped_line(rng: np.random.Generator) -> dict:
    m = _mstrip(rng)
    w, h, er, tand, t, sigma = m["w"], m["h"], m["er"], m["tand"], m["t"], m["sigma"]
    z0, eeff = physics.microstrip_z0_eeff(w, h, er)
    l1 = rng.uniform(2e-3, 15e-3)
    l2 = rng.uniform(2e-3, 15e-3)
    g = rng.uniform(20e-6, 600e-6)
    gamma = physics.line_gamma(FREQ, eeff, z0, tand, sigma, w, t)
    ab = np.einsum("fij,fjk,fkl->fil",
                   physics.line_abcd(gamma, z0, l1),
                   gap_abcd(FREQ, *gap_caps(w, h, er, g)),
                   physics.line_abcd(gamma, z0, l2))
    s = physics.abcd_to_s(ab, 50.0)
    elements = np.array([
        (EL_LINE, z0, eeff, tand, l1),
        (EL_SERIES_C, z0, eeff, g, w),
        (EL_LINE, z0, eeff, tand, l2),
    ])
    return dict(sample_id=f"gapline_{rng.integers(1 << 60)}",
                topology_family="gapped_line", ports=2,
                params=np.zeros(1), elements=elements, freq=FREQ,
                s=s.astype(complex))


def generate(n: int, seed: int = 7) -> list[dict]:
    rng = np.random.default_rng(seed)
    return [make_gapped_line(rng) for _ in range(n)]


def tokenize_oems_gap(p: dict, sub_h: float) -> np.ndarray:
    """Extended-vocabulary tokens for the openEMS gap family."""
    w = p["w"] * 1e-6
    z0, eeff = physics.microstrip_z0_eeff(w, sub_h, p["epr"])
    feed = (EL_LINE, z0, eeff, 0.0, 15e-3)
    return np.array([feed, (EL_SERIES_C, z0, eeff, p["g"] * 1e-6, w), feed])
