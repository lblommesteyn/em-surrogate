"""Loader for the reverberation-chamber transmission-line network dataset
(Zenodo record 167116, Magdowski & Kasper, OvGU Magdeburg).

14 physical network configurations, each measured at 72 stirrer angles over
5001 frequencies (200 MHz - 1 GHz). `s_supermatrix.mat` per configuration
holds S[i, j, angle, freq] (some configurations are 3-port: two antennas +
network terminal). Loading is lazy per configuration; nothing is expanded
or duplicated on disk.

A (configuration, stirrer angle) pair is ONE measurement of ONE physical
structure. Frequencies and angles are never treated as independent topology
samples.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.io import loadmat

ROOT = Path("data/external/tlines/rawdata")


def configs(root: Path = ROOT) -> list[str]:
    return sorted(p.name for p in root.iterdir() if p.is_dir())


def load_config(name: str, root: Path = ROOT) -> dict:
    d = root / name
    s = loadmat(d / "s_supermatrix.mat")["s_supermatrix"]
    return dict(
        config=name,
        s=s,  # (P, P, A, F) complex
        alphas=np.loadtxt(d / "alphas.out"),
        freq=np.loadtxt(d / "freqs.out"),
    )


def s_at_angle(cfg: dict, angle_idx: int) -> np.ndarray:
    """S-parameter matrix trajectory (F, P, P) for one stirrer angle."""
    return np.transpose(cfg["s"][:, :, angle_idx, :], (2, 0, 1))
