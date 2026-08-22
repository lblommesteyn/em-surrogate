"""Risk-calibration signal extraction: frozen synthetic benchmark + openEMS.

Produces results/riskcal/synth.npz with per-sample signal tables:
  val_cal    IID val scored by a calibration ensemble that early-stopped on it
             (honest-ish errors, train-side only)
  pseudo     leave-one-training-family-out pool: for each of the 5 training
             families, an ensemble trained without it scores internal val (ID)
             + the held-out family (pseudo-OOD). Strictly train-side.
  val_final  IID val scored by the frozen final models (z-stat reference)
  eval_synth frozen id_pool + ood_pool (via_lc, stub_short), final models
  eval_openems  id_pool + openEMS families, final models

Frozen config throughout (MLP 256x3, 5 members, kNN k=3). Final models are
reloaded from the milestone-2 checkpoints; nothing frozen is retrained.
"""

import sys
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from emsurr import dataset, models, novelty, splits, uncertainty

cfg = yaml.safe_load(Path("configs/baseline.yaml").read_text())
mcfg = cfg["models"]["mlp"]
MK = dict(hidden=mcfg["hidden"], depth=mcfg["depth"], epochs=mcfg["epochs"], lr=mcfg["lr"])
K = 3

samples = dataset.load_h5(cfg["data"]["path"])
sp = splits.make_splits(samples, seed=cfg["splits"]["seed"],
                        ood_families=tuple(cfg["splits"]["ood_families"]))
d = sp["ood_topology"]
train, id_pool, ood_pool = d["train"], d["val"], d["test"]


def load_final(tag, train_ref, val_ref):
    """Rebuild frozen models from checkpoints (normalizers from train data)."""
    mlp = models.TorchRegressor(**MK)
    mlp.norm = models.Normalizer().fit(models.features(train_ref))
    mlp.ynorm = models.Normalizer().fit(models.targets(train_ref))
    x = mlp.norm(models.features(train_ref))
    y = mlp.ynorm(models.targets(train_ref))
    mlp.model = models.MLP(x.shape[1], y.shape[1], hidden=MK["hidden"], depth=MK["depth"])
    mlp.model.load_state_dict(torch.load(f"results/checkpoints/mlp_{tag}.pt"))
    ens = uncertainty.Ensemble(n_members=5, **MK)
    for i, m in enumerate(ens.members):
        m.norm, m.ynorm, m.cfg = mlp.norm, mlp.ynorm, mlp.cfg
        m.model = models.MLP(x.shape[1], y.shape[1], hidden=MK["hidden"], depth=MK["depth"])
        m.model.load_state_dict(torch.load(f"results/checkpoints/ens{i}_{tag}.pt"))
    return mlp, ens


def signals(train_ref, mlp, ens, pool, is_ood):
    knn_in = novelty.KNNInputNovelty(k=K).fit(train_ref)
    maha = novelty.MahalanobisEmbedding().fit(train_ref, mlp)
    knn_e = novelty.KNNEmbedding(k=K).fit(train_ref, mlp)
    pred, unc = ens.predict_with_uncertainty(pool)
    err = uncertainty.per_sample_error(pred, np.stack([s["s"] for s in pool]))
    return dict(knn_input=knn_in.score(pool), maha_emb=maha.score(pool),
                knn_emb=knn_e.score(pool), ens_var=unc, err=err,
                is_ood=np.asarray(is_ood, float),
                family=np.array([s["topology_family"] for s in pool]))


out = {}

# ---- calibration side (train data only)
rs = np.random.RandomState(1)
ids = [s["sample_id"] for s in train]
rs.shuffle(ids)
by_id = {s["sample_id"]: s for s in train}
cut = int(0.85 * len(ids))
strain = [by_id[i] for i in ids[:cut]]
sval = [by_id[i] for i in ids[cut:]]

print("calibration ensemble (train-side)...")
mlp_c = models.TorchRegressor(**MK).fit(strain, sval)
ens_c = uncertainty.Ensemble(n_members=5, **MK).fit(strain, sval)
out["val_cal"] = signals(strain, mlp_c, ens_c, sval, [0] * len(sval))

fams = sorted({s["topology_family"] for s in train})
print("LOFO pseudo-OOD over training families:", fams)
pseudo_parts = []
for f in fams:
    tr_f = [s for s in strain if s["topology_family"] != f]
    va_f = [s for s in sval if s["topology_family"] != f]
    held = [s for s in strain + sval if s["topology_family"] == f]
    print(f"  fold {f}: train {len(tr_f)}, id {len(va_f)}, pseudo-ood {len(held)}")
    mlp_f = models.TorchRegressor(**MK).fit(tr_f, va_f)
    ens_f = uncertainty.Ensemble(n_members=5, **MK).fit(tr_f, va_f)
    pool = va_f + held
    pseudo_parts.append(signals(tr_f, mlp_f, ens_f, pool,
                                [0] * len(va_f) + [1] * len(held)))
out["pseudo"] = {k: np.concatenate([p[k] for p in pseudo_parts])
                 for k in pseudo_parts[0]}

# ---- evaluation side (frozen final models from checkpoints)
print("loading frozen final models...")
mlp_F, ens_F = load_final("final", train, id_pool)
out["val_final"] = signals(train, mlp_F, ens_F, sval, [0] * len(sval))
out["eval_synth"] = signals(train, mlp_F, ens_F, id_pool + ood_pool,
                            [0] * len(id_pool) + [1] * len(ood_pool))

from score_openems import load_openems  # noqa: E402

oems = load_openems()
out["eval_openems"] = signals(train, mlp_F, ens_F, id_pool + oems,
                              [0] * len(id_pool) + [1] * len(oems))

Path("results/riskcal").mkdir(parents=True, exist_ok=True)
np.savez_compressed("results/riskcal/synth.npz",
                    **{f"{p}/{k}": v for p, t in out.items() for k, v in t.items()})
print("wrote results/riskcal/synth.npz")
