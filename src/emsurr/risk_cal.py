"""Risk-calibration layer: decide surrogate-vs-solver from novelty and
ensemble-uncertainty signals, calibrated WITHOUT target-OOD labels.

Signal table convention: dict of 1-D arrays over one pool of samples with
keys 'knn_input', 'maha_emb', 'knn_emb', 'ens_var' (raw scores) and 'err'
(true per-sample error, present only where labels exist).

Calibration inputs allowed: IID validation signals+errors, and pseudo-OOD
pools constructed strictly from training data. Evaluation pools are frozen
benchmarks and are never seen by any calibration routine.
"""

from __future__ import annotations

import numpy as np

from .novelty import _rank01, risk_coverage, oracle_recovery

BUDGETS = (0.05, 0.1, 0.2, 0.3, 0.5)
NOVELTY_SIGNALS = ("knn_input", "knn_emb")
ALPHA_GRID = (0.1, 0.3, 0.5, 0.7, 0.9)


def spearman(a, b):
    ra, rb = _rank01(np.asarray(a, float)), _rank01(np.asarray(b, float))
    ra, rb = ra - ra.mean(), rb - rb.mean()
    den = np.sqrt((ra**2).sum() * (rb**2).sum())
    return float((ra * rb).sum() / den) if den > 0 else 0.0


# ------------------------------------------------------------------ methods
# A method is (name, fn(pool_signals, calib) -> risk score array).
# `calib` holds IID-val statistics and any pseudo-OOD-selected parameters;
# it is built before the evaluation pool is ever seen.


def _z(x, med, iqr):
    return (np.asarray(x, float) - med) / iqr


class Calibration:
    """Container for everything a method may use at deployment time."""

    def __init__(self, val_signals: dict):
        self.val = val_signals
        self.stats = {}
        for k in ("knn_input", "maha_emb", "knn_emb", "ens_var"):
            v = np.asarray(val_signals[k], float)
            iqr = max(np.quantile(v, 0.75) - np.quantile(v, 0.25), 1e-12)
            self.stats[k] = dict(med=float(np.median(v)), iqr=float(iqr),
                                 p95=float(np.quantile(v, 0.95)))
        # filled by selection routines (5/6):
        self.iid_choice = None       # (signal_name, alpha)
        self.pseudo_choice = None    # (signal_name, alpha)
        self.selection_tables = {}


def _rank_combo(pool, sig, alpha):
    return alpha * _rank01(np.asarray(pool[sig], float)) + (1 - alpha) * _rank01(
        np.asarray(pool["ens_var"], float))


def m_novelty_input(pool, cal):
    return np.asarray(pool["knn_input"], float)


def m_novelty_emb(pool, cal):
    return np.asarray(pool["knn_emb"], float)


def m_uncertainty(pool, cal):
    return np.asarray(pool["ens_var"], float)


def m_fixed_alpha(pool, cal):
    return _rank_combo(pool, "knn_input", 0.3)


def m_znorm_mean(pool, cal):
    zs = [_z(pool[k], cal.stats[k]["med"], cal.stats[k]["iqr"])
          for k in ("knn_input", "ens_var")]
    return np.mean(zs, 0)


def m_znorm_max(pool, cal):
    zs = [_z(pool[k], cal.stats[k]["med"], cal.stats[k]["iqr"])
          for k in ("knn_input", "ens_var")]
    return np.max(zs, 0)


def m_iid_calibrated(pool, cal):
    sig, alpha = cal.iid_choice
    return _rank_combo(pool, sig, alpha)


def m_pseudo_ood(pool, cal):
    sig, alpha = cal.pseudo_choice
    return _rank_combo(pool, sig, alpha)


def m_dynamic_support(pool, cal):
    """Local-support gating: outside training support (input-kNN distance above
    the IID-val p95) trust novelty; inside, trust ensemble uncertainty.
    Smooth blend, constants from IID val only."""
    st = cal.stats["knn_input"]
    d = np.asarray(pool["knn_input"], float)
    w = 1.0 / (1.0 + np.exp(-(d - st["p95"]) / st["iqr"]))
    return w * _rank01(d) + (1 - w) * _rank01(np.asarray(pool["ens_var"], float))


METHODS = [
    ("novelty_input", m_novelty_input),
    ("novelty_emb", m_novelty_emb),
    ("uncertainty", m_uncertainty),
    ("fixed_alpha0.3", m_fixed_alpha),
    ("znorm_mean", m_znorm_mean),
    ("znorm_max", m_znorm_max),
    ("iid_calibrated", m_iid_calibrated),
    ("pseudo_ood_calibrated", m_pseudo_ood),
    ("dynamic_support", m_dynamic_support),
]


# ------------------------------------------------------- calibration fitting
def _select(pool, err, criterion):
    """Grid-search (novelty signal, alpha) for a rank combination."""
    table, best, best_v = {}, None, -np.inf
    for sig in NOVELTY_SIGNALS:
        for a in ALPHA_GRID:
            s = _rank_combo(pool, sig, a)
            v = criterion(s, err)
            table[f"{sig}_a{a}"] = float(v)
            if v > best_v:
                best_v, best = v, (sig, a)
    return best, table


def crit_spearman(s, err):
    return spearman(s, err)


def crit_recovery20(s, err):
    rc = risk_coverage(s, err, (0.2,))
    return oracle_recovery(rc)[0.2]


def fit_calibration(val_signals, pseudo_signals=None) -> Calibration:
    """val_signals: IID validation pool WITH 'err'. pseudo_signals: pseudo-OOD
    calibration pool (IID val + pseudo-OOD samples) WITH 'err'."""
    cal = Calibration(val_signals)
    cal.iid_choice, t1 = _select(val_signals, val_signals["err"], crit_spearman)
    cal.selection_tables["iid_spearman"] = t1
    if pseudo_signals is not None:
        cal.pseudo_choice, t2 = _select(pseudo_signals, pseudo_signals["err"],
                                        crit_recovery20)
        cal.selection_tables["pseudo_recovery20"] = t2
    else:
        cal.pseudo_choice = cal.iid_choice
    return cal


# ------------------------------------------------------------- evaluation
def budget_to_catch(scores, errors, frac=0.9):
    """Smallest deferral budget whose top-scored set contains `frac` of
    catastrophic samples (err >= pool p95)."""
    errors = np.asarray(errors, float)
    cat = errors >= np.quantile(errors, 0.95)
    order = np.argsort(-np.asarray(scores, float))
    caught = np.cumsum(cat[order])
    need = frac * cat.sum()
    idx = np.searchsorted(caught, need)
    return float((idx + 1) / len(errors)) if idx < len(errors) else 1.0


def evaluate_methods(pool, cal, methods=METHODS, budgets=BUDGETS):
    """pool: signal table with 'err' over the frozen evaluation pool."""
    err = np.asarray(pool["err"], float)
    out = {}
    for name, fn in methods:
        s = fn(pool, cal)
        rc = risk_coverage(s, err, budgets)
        out[name] = dict(
            spearman_err=spearman(s, err),
            risk_coverage=rc,
            oracle_recovery=oracle_recovery(rc),
            catastrophic_caught={str(b): r["catastrophic_caught"]
                                 for b, r in zip(budgets, rc)},
            budget_catch90=budget_to_catch(s, err, 0.90),
            budget_catch95=budget_to_catch(s, err, 0.95),
        )
    # oracle/random reference rows
    rc_o = risk_coverage(err, err, budgets)
    out["_oracle"] = dict(risk_coverage=rc_o,
                          budget_catch90=budget_to_catch(err, err, 0.90),
                          budget_catch95=budget_to_catch(err, err, 0.95))
    return out


# ------------------------------------------------------------- diagnostics
def diagnose(id_pool, ood_pool, cal):
    """Why do signals differ in value? Scale shifts, within-OOD ranking power,
    and ensemble calibration, computed per pool."""
    d = {}
    for k in ("knn_input", "knn_emb", "ens_var", "maha_emb"):
        st = cal.stats[k]
        zi = _z(id_pool[k], st["med"], st["iqr"])
        zo = _z(ood_pool[k], st["med"], st["iqr"])
        d[k] = dict(
            id_median_z=float(np.median(zi)),
            ood_median_z=float(np.median(zo)),
            shift_ratio=float((np.median(zo) + 1e-9) / (np.abs(np.median(zi)) + 1e-9)),
            spearman_err_within_ood=spearman(ood_pool[k], ood_pool["err"]),
            spearman_err_within_id=spearman(id_pool[k], id_pool["err"]),
        )
    return d
