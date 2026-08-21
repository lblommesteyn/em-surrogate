"""Novelty-detection study (milestone 2).

Benchmark is frozen: same dataset, same splits (seed 0), same model configs
as milestone 1. Final OOD families (via_lc, stub_short) are never used for
any tuning; novelty hyperparameters are selected on a pseudo-OOD stage built
entirely inside the training families (hold out stub_open + stepped).
"""

import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

from emsurr import dataset, models, novelty, splits, uncertainty
from emsurr.metrics import complex_mae

cfg = yaml.safe_load(Path("configs/baseline.yaml").read_text())
mcfg = cfg["models"]["mlp"]
BUDGETS = (0.05, 0.1, 0.2, 0.3, 0.5)
PSEUDO_OOD = ("stub_open", "stepped")

samples = dataset.load_h5(cfg["data"]["path"])
sp = splits.make_splits(samples, seed=cfg["splits"]["seed"],
                        ood_families=tuple(cfg["splits"]["ood_families"]))
d = sp["ood_topology"]
train, id_pool, ood_pool = d["train"], d["val"], d["test"]


def fit_models(tr, va, tag):
    mlp = models.TorchRegressor(hidden=mcfg["hidden"], depth=mcfg["depth"],
                                epochs=mcfg["epochs"], lr=mcfg["lr"]).fit(tr, va)
    ens = uncertainty.Ensemble(n_members=cfg["uncertainty"]["n_members"],
                               hidden=mcfg["hidden"], depth=mcfg["depth"],
                               epochs=mcfg["epochs"], lr=mcfg["lr"]).fit(tr, va)
    Path("results/checkpoints").mkdir(parents=True, exist_ok=True)
    torch.save(mlp.model.state_dict(), f"results/checkpoints/mlp_{tag}.pt")
    for i, m in enumerate(ens.members):
        torch.save(m.model.state_dict(), f"results/checkpoints/ens{i}_{tag}.pt")
    return mlp, ens


def make_scorers(tr, mlp, ens, k, alpha):
    knn_in = novelty.KNNInputNovelty(k=k).fit(tr)
    maha = novelty.MahalanobisEmbedding().fit(tr, mlp)
    knn_e = novelty.KNNEmbedding(k=k).fit(tr, mlp)
    ens_s = novelty.EnsembleVariance(ens)
    comb = novelty.CombinedScore(knn_in, ens_s, alpha=alpha)
    return [knn_in, maha, knn_e, ens_s, comb]


def eval_scorers(scorers, id_s, ood_s, errors_pool):
    """id_s/ood_s: sample lists. errors_pool: per-sample errors on id+ood."""
    pool = id_s + ood_s
    out = {}
    for sc in scorers:
        t0 = time.perf_counter()
        # score the whole pool in one call: rank-based scorers (combined) need
        # a single consistent ranking over ID and OOD together
        s_all = sc.score(pool)
        ms = (time.perf_counter() - t0) / len(pool) * 1e3
        s_id, s_ood = s_all[: len(id_s)], s_all[len(id_s) :]
        rc = novelty.risk_coverage(s_all, errors_pool, BUDGETS)
        out[sc.name] = dict(
            auroc=novelty.auroc(s_id, s_ood),
            auprc=novelty.auprc(s_id, s_ood),
            spearman_err=uncertainty.spearman(s_all, errors_pool),
            risk_coverage=rc,
            oracle_recovery=novelty.oracle_recovery(rc),
            latency_ms=round(ms, 3),
        )
    return out


# ---------------- stage 1: pseudo-OOD hyperparameter selection ----------------
print("== pseudo-OOD selection stage ==")
sel_train_all = [s for s in train if s["topology_family"] not in PSEUDO_OOD]
pseudo_ood = [s for s in train if s["topology_family"] in PSEUDO_OOD]
rs = np.random.RandomState(1)
ids = [s["sample_id"] for s in sel_train_all]
rs.shuffle(ids)
cut = int(0.85 * len(ids))
by_id = {s["sample_id"]: s for s in sel_train_all}
sel_train = [by_id[i] for i in ids[:cut]]
sel_val = [by_id[i] for i in ids[cut:]]

mlp_s, ens_s_ = fit_models(sel_train, sel_val, "selection")
pool = sel_val + pseudo_ood
pred, _ = ens_s_.predict_with_uncertainty(pool)
errs = uncertainty.per_sample_error(pred, np.stack([s["s"] for s in pool]))

selection = {}
for k in (3, 10, 30):
    sc = novelty.KNNInputNovelty(k=k).fit(sel_train)
    a = novelty.auroc(sc.score(sel_val), sc.score(pseudo_ood))
    selection[f"knn_input_k{k}"] = a
best_k = max((3, 10, 30), key=lambda k: selection[f"knn_input_k{k}"])

ens_score = novelty.EnsembleVariance(ens_s_)
knn_best = novelty.KNNInputNovelty(k=best_k).fit(sel_train)
for alpha in (0.3, 0.5, 0.7):
    comb = novelty.CombinedScore(knn_best, ens_score, alpha=alpha)
    s_all = comb.score(pool)
    rc = novelty.risk_coverage(s_all, errs, (0.2,))
    selection[f"combined_a{alpha}"] = novelty.oracle_recovery(rc)[0.2]
best_alpha = max((0.3, 0.5, 0.7), key=lambda a: selection[f"combined_a{a}"])
print(f"selected k={best_k}, alpha={best_alpha}")
print(json.dumps(selection, indent=2))

# ---------------- stage 2: final evaluation on frozen OOD families ----------------
print("== final stage (frozen OOD: via_lc, stub_short) ==")
mlp_f, ens_f = fit_models(train, id_pool, "final")
final_pool = id_pool + ood_pool
pred_f, _ = ens_f.predict_with_uncertainty(final_pool)
true_f = np.stack([s["s"] for s in final_pool])
errors_f = uncertainty.per_sample_error(pred_f, true_f)
print(f"pool: {len(id_pool)} ID + {len(ood_pool)} OOD, "
      f"ensemble MAE={complex_mae(pred_f, true_f):.3f}")

scorers = make_scorers(train, mlp_f, ens_f, best_k, best_alpha)
results = eval_scorers(scorers, id_pool, ood_pool, errors_f)

# deployment-mix pool: mostly in-distribution traffic with 25% OOD
rs2 = np.random.RandomState(2)
ood_sub = [ood_pool[i] for i in rs2.choice(len(ood_pool), 100, replace=False)]
pred_m, _ = ens_f.predict_with_uncertainty(id_pool + ood_sub)
errors_m = uncertainty.per_sample_error(
    pred_m, np.stack([s["s"] for s in id_pool + ood_sub]))
results_mix = eval_scorers(scorers, id_pool, ood_sub, errors_m)

for tag, res in (("pool73%ood", results), ("pool25%ood", results_mix)):
    for name, r in res.items():
        print(f"{tag}/{name}: AUROC={r['auroc']:.3f} AUPRC={r['auprc']:.3f} "
              f"spearman={r['spearman_err']:.3f} "
              f"recovery@20%={r['oracle_recovery'][0.2]:.2f} "
              f"cat-caught@20%={r['risk_coverage'][2]['catastrophic_caught']:.2f}")

# per-family score distributions
fams = np.array([s["topology_family"] for s in final_pool])
per_family_scores = {}
for sc in scorers:
    s_all = sc.score(final_pool)
    per_family_scores[sc.name] = {
        fam: dict(median=float(np.median(s_all[fams == fam])),
                  p90=float(np.quantile(s_all[fams == fam], 0.9)))
        for fam in sorted(set(fams))
    }

# plots: score distribution by family (best scorer + ensemble) and risk-coverage
fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
for ax, name in zip(axes, ("knn_input", "ensemble_var")):
    sc = next(s for s in scorers if s.name == name)
    s_all = novelty._rank01(sc.score(final_pool))
    fam_list = sorted(set(fams), key=lambda f: np.median(s_all[fams == f]))
    ax.boxplot([s_all[fams == f] for f in fam_list], tick_labels=fam_list)
    ax.set_title(f"{name} score by family (OOD: via_lc, stub_short)")
    ax.tick_params(axis="x", rotation=30)
    ax.set_ylabel("rank-normalized score")
fig.tight_layout()
fig.savefig("results/novelty_scores_by_family.png", dpi=120)
plt.close(fig)

fig, ax = plt.subplots(figsize=(7, 4.5))
for name, r in results.items():
    ax.plot([x["budget"] for x in r["risk_coverage"]],
            [x["remaining_mae"] for x in r["risk_coverage"]], "o-", label=name)
rc0 = results["knn_input"]["risk_coverage"]
ax.plot([x["budget"] for x in rc0], [x["remaining_mae_oracle"] for x in rc0],
        "k--", label="oracle")
ax.plot([x["budget"] for x in rc0], [x["remaining_mae_random"] for x in rc0],
        ":", color="gray", label="random")
ax.set_xlabel("solver budget (fraction deferred)")
ax.set_ylabel("remaining MAE on surrogate-served samples")
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig("results/novelty_risk_coverage.png", dpi=120)
plt.close(fig)

out = dict(
    selection=dict(grid=selection, best_k=best_k, best_alpha=best_alpha,
                   pseudo_ood_families=list(PSEUDO_OOD)),
    final=results,
    final_deployment_mix=results_mix,
    per_family_scores=per_family_scores,
    pool=dict(n_id=len(id_pool), n_ood=len(ood_pool),
              ensemble_mae=float(complex_mae(pred_f, true_f))),
)
Path("results/novelty_metrics.json").write_text(json.dumps(out, indent=2))
print("wrote results/novelty_metrics.json")
