"""Pool-level label-free deployment switch: fit on train-side pseudo-
deployments, freeze, evaluate on the 7 frozen benchmark pools + composition
stress tests. Reads results/riskcal/*.npz (cached signals; surrogates are
not touched). Writes results/deployment_switch_metrics.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from emsurr import deploy_switch as DS

RC = Path("results/riskcal")
FRACS = (0.02, 0.05, 0.1, 0.25, 0.5, 0.75, 1.0)
SIZES = (30, 100, 300)
N_DRAWS = 3           # per (frac, size) for pseudo-deployment training
STRESS_FRACS = (0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 1.0)
STRESS_SIZES = (10, 30, 100, 300)
N_STRESS = 50

PSEUDO_CHOICE = {  # frozen from risk_calibration_metrics.json
    "synth": ("knn_emb", 0.7), "planar": ("knn_emb", 0.9),
    "sqchip_batch300": ("knn_input", 0.7),
    "sqchip_batchnear50": ("knn_input", 0.1),
    "sqchip_sqchip": ("knn_input", 0.1),
}
EVALS = {  # deployment -> [(pool name in npz, eval label)]
    "synth": [("eval_synth", "synth_ood"), ("eval_openems", "openems")],
    "planar": [("eval_ood", "planar_ood"), ("eval_meas", "planar_meas")],
    "sqchip_batch300": [("eval", "sq_batch300")],
    "sqchip_batchnear50": [("eval", "sq_batchnear50")],
    "sqchip_sqchip": [("eval", "sq_sqchip")],
}
NPZ = {"synth": "synth.npz", "planar": "planar.npz",
       "sqchip_batch300": "sqchip_batch300.npz",
       "sqchip_batchnear50": "sqchip_batchnear50.npz",
       "sqchip_sqchip": "sqchip_sqchip.npz"}
SIG_KEYS = ("knn_input", "knn_emb", "ens_var", "maha_emb", "err")


def load_npz(path):
    z = np.load(path, allow_pickle=True)
    pools = {}
    for key in z.files:
        p, k = key.split("/", 1)
        pools.setdefault(p, {})[k] = z[key]
    return pools


def take(pool, idx):
    return {k: np.asarray(pool[k])[idx] for k in pool if k != "family"}


def mix_pool(pool, frac, size, rng):
    ido = np.where(pool["is_ood"] == 0)[0]
    ood = np.where(pool["is_ood"] == 1)[0]
    n_ood = min(int(round(frac * size)), len(ood))
    n_id = min(size - n_ood, len(ido))
    if n_id + n_ood < 10:
        return None
    idx = np.concatenate([rng.choice(ido, n_id, replace=len(ido) < n_id),
                          rng.choice(ood, n_ood, replace=len(ood) < n_ood)])
    return take(pool, idx)


def pseudo_deployments(dep, pools, choice):
    """Train-side only: mixes of the pseudo pool's ID and pseudo-OOD rows."""
    st = DS.val_stats(pools["val_cal"])
    rows = []
    rng = np.random.default_rng(0)
    ps = pools["pseudo"]
    variants = [("all", ps)]
    if "family" in ps:  # per-family pseudo-OOD variants (synth)
        fam = ps["family"].astype(str)
        for f in sorted(set(fam[ps["is_ood"] == 1])):
            keep = (ps["is_ood"] == 0) | (fam == f)
            variants.append((f, take(ps, np.where(keep)[0])))
    for vname, vp in variants:
        for frac in FRACS:
            for size in SIZES:
                for d in range(N_DRAWS):
                    sub = mix_pool(vp, frac, size, rng)
                    if sub is None:
                        continue
                    desc = DS.descriptors(sub, st)
                    q = DS.strategy_quality(sub, choice, sub["err"])
                    margin = q["pseudo"] - q["uncertainty"]
                    rows.append((desc, int(margin > 0), float(margin),
                                 dict(dep=dep, variant=vname, frac=frac, size=size)))
    return rows


def main():
    data = {dep: load_npz(RC / f) for dep, f in NPZ.items()}

    # ---- train-side switch fitting (frozen before any eval pool is touched)
    train_rows = []
    for dep, pools in data.items():
        train_rows += pseudo_deployments(dep, pools, PSEUDO_CHOICE[dep])
    print(f"{len(train_rows)} pseudo-deployments "
          f"({sum(r[1] for r in train_rows)} pseudo-wins)")
    tau = DS.fit_threshold(train_rows)
    logit = DS.fit_logistic(train_rows)
    print(f"threshold switch: med_z_nov > {tau:.2f} -> pseudo-calibrated")
    print(f"logistic weights: {dict(zip(logit['keys'] + ['bias'], np.round(logit['w'], 2)))}")

    res = dict(tau=tau, logistic=logit,
               n_pseudo_deployments=len(train_rows), evals={}, stress={},
               pseudo_choice={k: list(v) for k, v in PSEUDO_CHOICE.items()})

    # ---- frozen evaluation
    for dep, evs in EVALS.items():
        pools = data[dep]
        st = DS.val_stats(pools["val_final"])
        choice = PSEUDO_CHOICE[dep]
        for pname, label in evs:
            pool = {k: pools[pname][k] for k in SIG_KEYS if k in pools[pname]}
            pool["is_ood"] = pools[pname]["is_ood"]
            err = pool["err"]
            desc = DS.descriptors(pool, st)
            r_unc = DS.eval_strategy(DS.score_uncertainty(pool, choice), err)
            r_pse = DS.eval_strategy(DS.score_pseudo(pool, choice), err)
            r_fix = DS.eval_strategy(DS.score_fixed(pool, choice), err)
            thr_pick = "pseudo" if desc["med_z_nov"] > tau else "uncertainty"
            log_pick_b, p = DS.logistic_choose(logit, desc)
            log_pick = "pseudo" if log_pick_b else "uncertainty"
            better = "pseudo" if r_pse["score2030"] > r_unc["score2030"] else "uncertainty"
            get = {"pseudo": r_pse, "uncertainty": r_unc}
            res["evals"][label] = dict(
                descriptors=desc,
                uncertainty=r_unc, pseudo=r_pse, fixed=r_fix,
                threshold_choice=thr_pick, logistic_choice=log_pick,
                logistic_p=p, better=better,
                regret_threshold=float(get[better]["score2030"] - get[thr_pick]["score2030"]),
                regret_logistic=float(get[better]["score2030"] - get[log_pick]["score2030"]),
                oracle_gap_rem20=float(get[thr_pick]["remaining_mae"]["0.2"]),
            )
            print(f"[{label}] med_z={desc['med_z_nov']:7.2f} better={better:11s} "
                  f"thr={thr_pick:11s} log={log_pick:11s} "
                  f"regret_thr={res['evals'][label]['regret_threshold']:+.3f}")

    # ---- stress tests: composition sweeps on eval pools (labels used only to
    # CONSTRUCT the sub-pools and to score the choice, never to make it)
    for dep, evs in EVALS.items():
        pools = data[dep]
        st = DS.val_stats(pools["val_final"])
        choice = PSEUDO_CHOICE[dep]
        pname, label = evs[0]
        full = pools[pname]
        rng = np.random.default_rng(7)
        tab = {}
        for frac in STRESS_FRACS:
            for size in STRESS_SIZES:
                correct = picks_pseudo = 0
                n = 0
                for _ in range(N_STRESS):
                    sub = mix_pool(full, frac, size, rng)
                    if sub is None or len(sub["err"]) < 10:
                        continue
                    desc = DS.descriptors(sub, st)
                    pick = "pseudo" if desc["med_z_nov"] > tau else "uncertainty"
                    q = DS.strategy_quality(sub, choice, sub["err"])
                    better = "pseudo" if q["pseudo"] > q["uncertainty"] else "uncertainty"
                    correct += int(pick == better)
                    picks_pseudo += int(pick == "pseudo")
                    n += 1
                if n:
                    tab[f"frac{frac}_n{size}"] = dict(
                        n=n, accuracy=correct / n, pick_pseudo_rate=picks_pseudo / n)
        res["stress"][label] = tab

    Path("results/deployment_switch_metrics.json").write_text(json.dumps(res, indent=1))
    print("wrote results/deployment_switch_metrics.json")


if __name__ == "__main__":
    main()
