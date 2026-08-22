"""Pool-level, label-free deployment switch for solver fallback.

Given an UNLABELED batch of queries, decide whether to rank fallback by
plain ensemble uncertainty or by the deployment's frozen pseudo-OOD-
calibrated combination. The decision uses only pool descriptors computed
from signals z-normalized against the deployment's IID-val statistics -
never a target error or OOD label.

Descriptors (per pool):
  med_z_nov / q90_z_nov / q99_z_nov   input-kNN novelty z-shift quantiles
  frac_p95 / frac_p99                 fraction above IID-val p95/p99 novelty
  med_z_unc                            ensemble-disagreement z-shift
  med_z_emb                            embedding-kNN z-shift
  nov_emb_gap                          med_z_nov - med_z_emb (representation
                                       disagreement)
  spread_z_nov                         IQR of novelty z (mixture width)

Switch rules:
  threshold  choose pseudo-calibrated when med_z_nov > tau; tau fitted on
             train-side pseudo-deployments only
  logistic   tiny 2-feature logistic (med_z_nov, med_z_unc) on the same
             pseudo-deployments
"""

from __future__ import annotations

import numpy as np

from .novelty import _rank01, risk_coverage, oracle_recovery
from .risk_cal import budget_to_catch, spearman

BUDGETS = (0.05, 0.1, 0.2, 0.3, 0.5)


def _z(x, med, iqr):
    return (np.asarray(x, float) - med) / iqr


def val_stats(val):
    st = {}
    for k in ("knn_input", "knn_emb", "ens_var"):
        v = np.asarray(val[k], float)
        iqr = max(np.quantile(v, 0.75) - np.quantile(v, 0.25), 1e-12)
        st[k] = dict(med=float(np.median(v)), iqr=float(iqr),
                     p95=float(np.quantile(v, 0.95)), p99=float(np.quantile(v, 0.99)))
    return st


DESC_KEYS = ["med_z_nov", "q90_z_nov", "q99_z_nov", "frac_p95", "frac_p99",
             "med_z_unc", "med_z_emb", "nov_emb_gap", "spread_z_nov"]


def descriptors(pool, st):
    zn = _z(pool["knn_input"], st["knn_input"]["med"], st["knn_input"]["iqr"])
    ze = _z(pool["knn_emb"], st["knn_emb"]["med"], st["knn_emb"]["iqr"])
    zu = _z(pool["ens_var"], st["ens_var"]["med"], st["ens_var"]["iqr"])
    d = dict(
        med_z_nov=float(np.median(zn)),
        q90_z_nov=float(np.quantile(zn, 0.90)),
        q99_z_nov=float(np.quantile(zn, 0.99)),
        frac_p95=float(np.mean(np.asarray(pool["knn_input"], float) > st["knn_input"]["p95"])),
        frac_p99=float(np.mean(np.asarray(pool["knn_input"], float) > st["knn_input"]["p99"])),
        med_z_unc=float(np.median(zu)),
        med_z_emb=float(np.median(ze)),
        nov_emb_gap=float(np.median(zn) - np.median(ze)),
        spread_z_nov=float(np.quantile(zn, 0.75) - np.quantile(zn, 0.25)),
    )
    return d


# ------------------------------------------------------------- strategies
def score_uncertainty(pool, choice):
    return np.asarray(pool["ens_var"], float)


def score_pseudo(pool, choice):
    sig, alpha = choice
    return alpha * _rank01(np.asarray(pool[sig], float)) + (1 - alpha) * _rank01(
        np.asarray(pool["ens_var"], float))


def score_fixed(pool, choice):
    return 0.3 * _rank01(np.asarray(pool["knn_input"], float)) + 0.7 * _rank01(
        np.asarray(pool["ens_var"], float))


def eval_strategy(scores, err, budgets=BUDGETS):
    rc = risk_coverage(np.asarray(scores, float), np.asarray(err, float), budgets)
    orec = oracle_recovery(rc)
    return dict(
        oracle_recovery={str(b): orec[b] for b in budgets},
        remaining_mae={str(r["budget"]): r["remaining_mae"] for r in rc},
        catastrophic_caught={str(r["budget"]): r["catastrophic_caught"] for r in rc},
        budget_catch90=budget_to_catch(scores, err, 0.90),
        budget_catch95=budget_to_catch(scores, err, 0.95),
        spearman=spearman(scores, err),
        score20=float(orec[0.2]),
        score2030=float(0.5 * (orec[0.2] + orec[0.3])),
    )


def strategy_quality(pool, choice, err):
    """orec@{20,30} average for both candidate strategies on a labeled pool."""
    return dict(
        uncertainty=eval_strategy(score_uncertainty(pool, choice), err)["score2030"],
        pseudo=eval_strategy(score_pseudo(pool, choice), err)["score2030"],
    )


# ------------------------------------------------------------- switch fitting
def fit_threshold(rows):
    """rows: list of (descriptor dict, label) where label=1 means pseudo wins.
    Pick tau on med_z_nov maximizing margin-weighted selection quality."""
    xs = np.array([r[0]["med_z_nov"] for r in rows])
    margins = np.array([r[2] for r in rows])  # pseudo_quality - unc_quality
    cand = np.unique(np.round(np.concatenate([xs, np.linspace(0, 20, 81)]), 3))
    best_tau, best_v = 0.0, -np.inf
    for tau in cand:
        pick_pseudo = xs > tau
        v = np.sum(np.where(pick_pseudo, margins, -margins))  # gained quality
        if v > best_v:
            best_v, best_tau = v, float(tau)
    return best_tau


def fit_logistic(rows, keys=("med_z_nov", "med_z_unc"), steps=3000, lr=0.1):
    X = np.array([[r[0][k] for k in keys] for r in rows])
    w_sample = np.abs([r[2] for r in rows])
    y = np.array([r[1] for r in rows], float)
    mu, sd = X.mean(0), X.std(0) + 1e-9
    Xz = (X - mu) / sd
    w = np.zeros(Xz.shape[1] + 1)
    A = np.c_[Xz, np.ones(len(Xz))]
    for _ in range(steps):
        p = 1 / (1 + np.exp(-A @ w))
        g = A.T @ (w_sample * (p - y)) / len(y) + 1e-3 * np.r_[w[:-1], 0]
        w -= lr * g
    return dict(keys=list(keys), mu=mu.tolist(), sd=sd.tolist(), w=w.tolist())


def logistic_choose(model, desc):
    x = np.array([desc[k] for k in model["keys"]])
    xz = (x - np.array(model["mu"])) / np.array(model["sd"])
    p = 1 / (1 + np.exp(-(xz @ np.array(model["w"][:-1]) + model["w"][-1])))
    return p > 0.5, float(p)
