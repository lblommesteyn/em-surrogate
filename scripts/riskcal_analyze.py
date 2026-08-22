"""Risk-calibration analysis: fit calibrations from train-side pools only,
freeze them, evaluate all methods on every frozen benchmark pool.

Reads results/riskcal/*.npz, writes results/risk_calibration_metrics.json.
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from emsurr import risk_cal


def load_npz(path):
    z = np.load(path, allow_pickle=True)
    pools = {}
    for key in z.files:
        p, k = key.split("/", 1)
        pools.setdefault(p, {})[k] = z[key]
    return pools


def run_deployment(name, pools, eval_names):
    # calibration is fitted BEFORE any eval pool is touched
    cal = risk_cal.fit_calibration(pools["val_cal"], pools.get("pseudo"))
    # z-stats must reference the final models that produced the eval signals
    cal_stats_ref = risk_cal.Calibration(pools["val_final"])
    cal.stats = cal_stats_ref.stats
    entry = dict(
        iid_choice=list(cal.iid_choice),
        pseudo_choice=list(cal.pseudo_choice),
        selection_tables=cal.selection_tables,
        evals={},
    )
    for ev in eval_names:
        pool = pools[ev]
        res = risk_cal.evaluate_methods(pool, cal)
        mask = pool["is_ood"].astype(bool)
        res["_diag"] = risk_cal.diagnose(
            {k: v[~mask] for k, v in pool.items() if k != "family"},
            {k: v[mask] for k, v in pool.items() if k != "family"}, cal)
        res["_pool"] = dict(n_id=int((~mask).sum()), n_ood=int(mask.sum()),
                            err_id=float(pool["err"][~mask].mean()),
                            err_ood=float(pool["err"][mask].mean()))
        entry["evals"][ev] = res
    return entry


def main():
    out = {}
    rc = Path("results/riskcal")
    out["synth"] = run_deployment("synth", load_npz(rc / "synth.npz"),
                                  ["eval_synth", "eval_openems"])
    out["planar"] = run_deployment("planar", load_npz(rc / "planar.npz"),
                                   ["eval_ood", "eval_meas"])
    for fold in ["batch300", "batchnear50", "sqchip"]:
        out[f"sqchip_{fold}"] = run_deployment(
            fold, load_npz(rc / f"sqchip_{fold}.npz"), ["eval"])

    Path("results/risk_calibration_metrics.json").write_text(json.dumps(out, indent=1))

    # compact console summary
    for dep, e in out.items():
        print(f"== {dep}: iid_choice={e['iid_choice']} pseudo_choice={e['pseudo_choice']}")
        for ev, res in e["evals"].items():
            print(f"  [{ev}] id_err={res['_pool']['err_id']:.4f} ood_err={res['_pool']['err_ood']:.4f}")
            for m, r in res.items():
                if m.startswith("_"):
                    continue
                orec = r["oracle_recovery"]
                print(f"    {m:22s} sp={r['spearman_err']:+.3f} "
                      f"orec20={orec[0.2]:.2f} orec30={orec[0.3]:.2f} "
                      f"cat20={r['catastrophic_caught']['0.2']:.2f} "
                      f"b90={r['budget_catch90']:.2f} b95={r['budget_catch95']:.2f}")
    print("wrote results/risk_calibration_metrics.json")


if __name__ == "__main__":
    main()
