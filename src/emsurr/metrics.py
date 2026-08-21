"""Error metrics on complex S-parameters.

Targets/predictions are (N, F, 2, 2) complex arrays on a shared grid.
"""

from __future__ import annotations

import numpy as np

from .dataset import passivity_violations, reciprocity_error

DB_FLOOR = 1e-8


def complex_mae(pred: np.ndarray, true: np.ndarray) -> float:
    return float(np.mean(np.abs(pred - true)))


def complex_rmse(pred: np.ndarray, true: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.abs(pred - true) ** 2)))


def mag_db_mae(pred: np.ndarray, true: np.ndarray) -> float:
    pd_ = 20 * np.log10(np.abs(pred) + DB_FLOOR)
    td = 20 * np.log10(np.abs(true) + DB_FLOOR)
    return float(np.mean(np.abs(pd_ - td)))


def phase_mae_deg(pred: np.ndarray, true: np.ndarray) -> float:
    dphi = np.angle(pred * np.conj(true))  # wrapped difference
    return float(np.degrees(np.mean(np.abs(dphi))))


def error_vs_frequency(pred: np.ndarray, true: np.ndarray) -> np.ndarray:
    """Complex MAE per frequency point, shape (F,)."""
    return np.mean(np.abs(pred - true), axis=(0, 2, 3))


def physics_violations(pred: np.ndarray) -> dict:
    pv = sum(passivity_violations(p, 1e-3) for p in pred)
    rec = max(reciprocity_error(p) for p in pred)
    return dict(
        passivity_violation_points=int(pv),
        total_points=int(pred.shape[0] * pred.shape[1]),
        max_reciprocity_error=float(rec),
    )


def full_report(pred: np.ndarray, true: np.ndarray) -> dict:
    return dict(
        complex_mae=complex_mae(pred, true),
        complex_rmse=complex_rmse(pred, true),
        mag_db_mae=mag_db_mae(pred, true),
        phase_mae_deg=phase_mae_deg(pred, true),
        **physics_violations(pred),
    )


def per_family_report(pred: np.ndarray, samples: list[dict], true: np.ndarray) -> dict:
    fams = np.array([s["topology_family"] for s in samples])
    return {
        fam: full_report(pred[fams == fam], true[fams == fam])
        for fam in sorted(set(fams))
    }
