"""Train all baselines on all splits, evaluate, run the uncertainty study,
and dump results/metrics.json + plots. Final test sets are only touched at
evaluation time; model selection uses val loss only."""

import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

from emsurr import dataset, metrics, models, splits, uncertainty

cfg = yaml.safe_load(Path("configs/baseline.yaml").read_text())
samples = dataset.load_h5(cfg["data"]["path"])
sp = splits.make_splits(samples, seed=cfg["splits"]["seed"],
                        ood_families=tuple(cfg["splits"]["ood_families"]))

results = {}
mc = cfg["models"]


def build(name):
    if name == "knn":
        return models.KNNBaseline(k=mc["knn"]["k"])
    if name == "mlp":
        c = mc["mlp"]
        return models.TorchRegressor(hidden=c["hidden"], depth=c["depth"],
                                     epochs=c["epochs"], lr=c["lr"])
    if name == "deepsets":
        c = mc["deepsets"]
        return models.DeepSetsModel(hidden=c["hidden"], depth=c["depth"],
                                    epochs=c["epochs"], lr=c["lr"])
    raise ValueError(name)


for split_name, d in sp.items():
    # leakage guard before any training
    assert not dataset.check_leakage(d["train"], d["test"]), split_name
    true = np.stack([s["s"] for s in d["test"]])
    results[split_name] = {"n_train": len(d["train"]), "n_test": len(d["test"])}
    for mname in ("knn", "mlp", "deepsets"):
        t0 = time.perf_counter()
        model = build(mname).fit(d["train"], d["val"])
        fit_s = time.perf_counter() - t0
        pred, ms_per = models.timed_predict(model, d["test"])
        rep = metrics.full_report(pred, true)
        rep["latency_ms_per_sample"] = round(ms_per, 3)
        rep["fit_seconds"] = round(fit_s, 1)
        rep["per_family"] = metrics.per_family_report(pred, d["test"], true)
        rep["err_vs_freq"] = metrics.error_vs_frequency(pred, true).tolist()
        results[split_name][mname] = rep
        print(f"{split_name}/{mname}: cMAE={rep['complex_mae']:.4f} "
              f"dB-MAE={rep['mag_db_mae']:.2f} phase={rep['phase_mae_deg']:.1f}deg "
              f"({ms_per:.2f} ms/sample)")

# ---------------- uncertainty study (ensemble of MLPs) ----------------
unc_results = {}
uc = cfg["uncertainty"]
for split_name in ("iid", "ood_topology"):
    d = sp[split_name]
    ens = uncertainty.Ensemble(n_members=uc["n_members"], hidden=mc["mlp"]["hidden"],
                               depth=mc["mlp"]["depth"], epochs=mc["mlp"]["epochs"],
                               lr=mc["mlp"]["lr"]).fit(d["train"], d["val"])
    true = np.stack([s["s"] for s in d["test"]])
    pred, unc = ens.predict_with_uncertainty(d["test"])
    err = uncertainty.per_sample_error(pred, true)
    unc_results[split_name] = dict(
        ensemble_mae=metrics.complex_mae(pred, true),
        spearman_unc_err=uncertainty.spearman(unc, err),
        confidence_curve=uncertainty.confidence_curves(pred, unc, true),
        fallback=uncertainty.fallback_experiment(pred, unc, true,
                                                 tuple(uc["fallback_fracs"])),
    )
    print(f"unc/{split_name}: spearman={unc_results[split_name]['spearman_unc_err']:.3f}")

    # plots
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].scatter(unc, err, s=8, alpha=0.5)
    ax[0].set_xlabel("ensemble uncertainty"); ax[0].set_ylabel("complex MAE")
    ax[0].set_title(f"error vs confidence ({split_name})")
    cc = unc_results[split_name]["confidence_curve"]
    ax[1].plot([r["dropped_frac"] for r in cc], [r["retained_mae"] for r in cc], "o-")
    ax[1].set_xlabel("fraction deferred to solver"); ax[1].set_ylabel("retained MAE")
    ax[1].set_title("coverage vs error")
    fig.tight_layout()
    fig.savefig(f"results/uncertainty_{split_name}.png", dpi=120)
    plt.close(fig)

# error-vs-frequency plot per split
for split_name in results:
    fig, ax = plt.subplots(figsize=(7, 4))
    f = samples[0]["freq"] / 1e9
    for mname in ("knn", "mlp", "deepsets"):
        ax.plot(f, results[split_name][mname]["err_vs_freq"], label=mname)
    ax.set_xlabel("frequency (GHz)"); ax.set_ylabel("complex MAE"); ax.legend()
    ax.set_title(f"error vs frequency ({split_name})")
    fig.tight_layout()
    fig.savefig(f"results/err_vs_freq_{split_name}.png", dpi=120)
    plt.close(fig)

out = dict(baselines=results, uncertainty=unc_results)
Path("results/metrics.json").write_text(json.dumps(out, indent=2))
print("wrote results/metrics.json")
