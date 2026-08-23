"""Within-OOD error ranking: fit rank-ridge risk models on train-side
pseudo-OOD errors, freeze, evaluate on frozen benchmark pools.

Reads results/riskcal/sig2_*.npz, writes results/within_ood_metrics.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from emsurr.novelty import _rank01, oracle_recovery, risk_coverage
from emsurr.risk_cal import budget_to_catch, spearman
from emsurr.risk_signals import (FEATURE_NAMES, SIGNALS, fit_rank_ridge,
                                 predict_rank_ridge, rank_features)

RC = Path("results/riskcal")
DEPS = {"planar": ["eval_ood", "eval_meas"],
        "sqchip_batch300": ["eval"], "sqchip_batchnear50": ["eval"],
        "sqchip_sqchip": ["eval"], "synth": ["eval_synth"]}
BUDGETS = (0.05, 0.1, 0.2, 0.3, 0.5)


def load(dep):
    z = np.load(RC / f"sig2_{dep}.npz", allow_pickle=True)
    pools = {}
    for key in z.files:
        p, k = key.split("/", 1)
        pools.setdefault(p, {})[k] = z[key]
    return pools


def topk_capture(risk, err, frac):
    k = max(1, int(round(frac * len(err))))
    top_r = np.argsort(-risk)[:k]
    top_e = set(np.argsort(-err)[:k])
    return float(len(set(top_r) & top_e) / k)


def eval_ranking(risk, err):
    return dict(
        spearman=spearman(risk, err),
        cap5=topk_capture(risk, err, 0.05),
        cap10=topk_capture(risk, err, 0.10),
        cap20=topk_capture(risk, err, 0.20),
        budget_catch90=budget_to_catch(risk, err, 0.90),
    )


def residual_spearman(sig, err, cond):
    """Spearman of sig vs err after linearly removing rank(cond) from both."""
    rs, re, rc = _rank01(sig), _rank01(err), _rank01(cond)
    rs = rs - np.polyval(np.polyfit(rc, rs, 1), rc)
    re = re - np.polyval(np.polyfit(rc, re, 1), rc)
    return spearman(rs, re)


def main():
    res = {}
    for dep, evs in DEPS.items():
        pools = load(dep)
        ps = pools["pseudo"]
        mo = ps["is_ood"].astype(bool)
        Fp = rank_features(ps, SIGNALS)
        # frozen fits: full model + ens_var-only, on pseudo-OOD rows only
        w_full = fit_rank_ridge(Fp[mo], ps["err"][mo])
        w_unc = fit_rank_ridge(Fp[mo][:, [0]], ps["err"][mo])

        # ablation on pseudo pool (5-fold CV, within-OOD)
        idx = np.where(mo)[0]
        rng = np.random.default_rng(0)
        folds = np.array_split(rng.permutation(idx), 5)
        abl = {}
        for extra in [None] + list(range(1, len(FEATURE_NAMES))):
            cols = [0] + ([extra] if extra is not None else [])
            sps = []
            for f in range(5):
                te = folds[f]
                tr = np.concatenate([folds[g] for g in range(5) if g != f])
                w = fit_rank_ridge(Fp[tr][:, cols], ps["err"][tr])
                sps.append(spearman(predict_rank_ridge(w, Fp[te][:, cols]), ps["err"][te]))
            name = "ens_var_only" if extra is None else f"+{FEATURE_NAMES[extra]}"
            abl[name] = float(np.mean(sps))
        entry = dict(ablation_pseudo_cv=abl,
                     weights_full=dict(zip(FEATURE_NAMES + ["bias"],
                                           np.round(w_full, 3).tolist())),
                     evals={})

        for ev in evs:
            pool = pools[ev]
            F = rank_features(pool, SIGNALS)
            err = pool["err"]
            mo_e = pool["is_ood"].astype(bool)
            risk_full = predict_rank_ridge(w_full, F)
            scores = dict(
                ridge_full=risk_full,
                ens_var=np.asarray(pool["ens_var"], float),
                knn_emb=np.asarray(pool["knn_emb"], float),
                ridge_unc=predict_rank_ridge(w_unc, F[:, [0]]),
            )
            ev_out = {}
            for nm, sc in scores.items():
                r = dict(within_ood=eval_ranking(sc[mo_e], err[mo_e]))
                rc = risk_coverage(sc, err, BUDGETS)
                orec = oracle_recovery(rc)
                r["full_pool"] = dict(
                    oracle_recovery={str(b): orec[b] for b in BUDGETS},
                    regret_vs_oracle={str(b): 1 - orec[b] for b in BUDGETS},
                    budget_catch90=budget_to_catch(sc, err, 0.90),
                    budget_catch95=budget_to_catch(sc, err, 0.95))
                ev_out[nm] = r
            # per-family within-OOD spearman (synth only)
            if "family" in pool:
                fams = pool["family"].astype(str)
                pf = {}
                for f in sorted(set(fams[mo_e])):
                    m2 = mo_e & (fams == f)
                    if m2.sum() >= 20:
                        pf[f] = dict(
                            ridge=spearman(risk_full[m2], err[m2]),
                            ens_var=spearman(pool["ens_var"][m2], err[m2]))
                ev_out["_per_family"] = pf
            # hard cases within OOD
            e, nov = err[mo_e], np.asarray(pool["knn_input"], float)[mo_e]
            unc = np.asarray(pool["ens_var"], float)[mo_e]
            rn, re_ = _rank01(nov), _rank01(e)
            cat = e >= np.quantile(e, 0.9)
            silent = (unc <= np.quantile(unc, 0.5)) & cat  # confident+wrong
            hc = dict(
                high_nov_low_err=float(np.mean((rn > 0.8) & (re_ < 0.2))),
                low_nov_high_err=float(np.mean((rn < 0.2) & (re_ > 0.8))),
                confident_but_catastrophic=float(np.mean(silent)),
            )
            if silent.sum() >= 3:
                Fo = F[mo_e]
                hc["signal_rank_on_silent"] = {
                    FEATURE_NAMES[j]: float(Fo[silent, j].mean())
                    for j in range(len(FEATURE_NAMES))}
            hc["residual_spearman_given_novelty"] = {
                SIGNALS[j]: residual_spearman(
                    np.asarray(pool[SIGNALS[j]], float)[mo_e], e, nov)
                for j in range(len(SIGNALS))}
            ev_out["_hard_cases"] = hc
            entry["evals"][ev] = ev_out
        res[dep] = entry
        print(f"== {dep}")
        for ev in evs:
            for nm in ("ridge_full", "ens_var", "knn_emb"):
                w = res[dep]["evals"][ev][nm]["within_ood"]
                fp = res[dep]["evals"][ev][nm]["full_pool"]
                print(f"  [{ev}] {nm:10s} withinOOD sp={w['spearman']:+.3f} "
                      f"cap10={w['cap10']:.2f} b90(pool)={fp['budget_catch90']:.2f} "
                      f"orec20={fp['oracle_recovery']['0.2']:.2f}")

    # ---- hardware QA sweep on planar MEAS (no correction, pure triage)
    pool = load("planar")["eval_meas"]
    m = pool["is_ood"].astype(bool)
    e = pool["err"][m]
    qa = {}
    rng = np.random.default_rng(0)
    for nm, sc in dict(ens_var=pool["ens_var"][m], knn_input=pool["knn_input"][m],
                       ridge_like=pool["ens_var"][m] + pool["knn_input"][m] /
                       (np.abs(pool["knn_input"][m]).max() + 1e-9)).items():
        row = {}
        for b in BUDGETS:
            k = max(1, int(round(b * len(e))))
            keep = np.argsort(-np.asarray(sc, float))[k:]
            row[str(b)] = dict(remaining_mean=float(e[keep].mean()),
                               removed_frac_of_total_err=float(
                                   1 - e[keep].sum() / e.sum()))
        qa[nm] = row
    qa["random"] = {}
    for b in BUDGETS:
        k = max(1, int(round(b * len(e))))
        vals = [np.delete(e, rng.choice(len(e), k, replace=False)).mean()
                for _ in range(500)]
        qa["random"][str(b)] = dict(remaining_mean=float(np.mean(vals)))
    qa["oracle"] = {str(b): dict(remaining_mean=float(
        np.sort(e)[: len(e) - max(1, int(round(b * len(e))))].mean()))
        for b in BUDGETS}
    res["_hardware_qa_meas"] = qa

    Path("results/within_ood_metrics.json").write_text(json.dumps(res, indent=1))
    print("wrote results/within_ood_metrics.json")


if __name__ == "__main__":
    main()
