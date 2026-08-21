"""Analytic transmission-line physics used by the synthetic dataset.

Closed-form quasi-static models (Hammerstad-Jensen microstrip, Cohn
stripline) with conductor and dielectric loss. Accuracy is adequate for a
surrogate-learning benchmark; no claim of full-wave fidelity is made.
"""

from __future__ import annotations

import numpy as np

ETA0 = 376.730313668
C0 = 299792458.0
MU0 = 4e-7 * np.pi


def microstrip_z0_eeff(w: float, h: float, er: float) -> tuple[float, float]:
    """Hammerstad-Jensen characteristic impedance and effective permittivity."""
    u = w / h
    a = 1 + np.log((u**4 + (u / 52) ** 2) / (u**4 + 0.432)) / 49 + np.log(
        1 + (u / 18.1) ** 3
    ) / 18.7
    b = 0.564 * ((er - 0.9) / (er + 3)) ** 0.053
    eeff = (er + 1) / 2 + (er - 1) / 2 * (1 + 10 / u) ** (-a * b)
    fu = 6 + (2 * np.pi - 6) * np.exp(-((30.666 / u) ** 0.7528))
    z0_air = ETA0 / (2 * np.pi) * np.log(fu / u + np.sqrt(1 + (2 / u) ** 2))
    return z0_air / np.sqrt(eeff), eeff


def stripline_z0(w: float, b: float, er: float) -> float:
    """Cohn's symmetric stripline impedance (thin strip)."""
    we = w / b
    if we < 0.35:
        we = we - (0.35 - we) ** 2
    return 30 * np.pi / np.sqrt(er) / (we + 0.441)


def line_gamma(
    f: np.ndarray, eeff: float, z0: float, tand: float, sigma: float, w: float, t: float
) -> np.ndarray:
    """Propagation constant with dielectric and skin-effect conductor loss."""
    beta = 2 * np.pi * f * np.sqrt(eeff) / C0
    alpha_d = beta * tand / 2
    rs = np.sqrt(np.pi * f * MU0 / sigma)
    alpha_c = rs / (z0 * max(w, t))
    return (alpha_c + alpha_d) + 1j * beta


def line_abcd(gamma: np.ndarray, z0: float, length: float) -> np.ndarray:
    """(F,2,2) ABCD matrix of a uniform line."""
    gl = gamma * length
    ch, sh = np.cosh(gl), np.sinh(gl)
    abcd = np.empty((len(gamma), 2, 2), dtype=complex)
    abcd[:, 0, 0] = ch
    abcd[:, 0, 1] = z0 * sh
    abcd[:, 1, 0] = sh / z0
    abcd[:, 1, 1] = ch
    return abcd


def shunt_abcd(y: np.ndarray) -> np.ndarray:
    abcd = np.zeros((len(y), 2, 2), dtype=complex)
    abcd[:, 0, 0] = 1
    abcd[:, 1, 1] = 1
    abcd[:, 1, 0] = y
    return abcd


def series_abcd(z: np.ndarray) -> np.ndarray:
    abcd = np.zeros((len(z), 2, 2), dtype=complex)
    abcd[:, 0, 0] = 1
    abcd[:, 1, 1] = 1
    abcd[:, 0, 1] = z
    return abcd


def cascade(*mats: np.ndarray) -> np.ndarray:
    out = mats[0]
    for m in mats[1:]:
        out = out @ m
    return out


def abcd_to_s(abcd: np.ndarray, zref: float = 50.0) -> np.ndarray:
    """(F,2,2) ABCD -> (F,2,2) S-parameters."""
    a, b, c, d = abcd[:, 0, 0], abcd[:, 0, 1], abcd[:, 1, 0], abcd[:, 1, 1]
    den = a + b / zref + c * zref + d
    s = np.empty_like(abcd)
    s[:, 0, 0] = (a + b / zref - c * zref - d) / den
    s[:, 0, 1] = 2 * (a * d - b * c) / den
    s[:, 1, 0] = 2 / den
    s[:, 1, 1] = (-a + b / zref - c * zref + d) / den
    return s
