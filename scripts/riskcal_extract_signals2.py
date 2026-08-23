"""Extended risk-signal extraction for the within-OOD ranking milestone.

For each deployment, writes results/riskcal/sig2_<dep>.npz with pools:
  pseudo   train-side pseudo-OOD pool (LOFO / parameter-extrapolation),
           signals from calibration models, err known (train-side only)
  eval_*   frozen benchmark pools, signals from frozen final models

Frozen recipes throughout; the synth final models are reloaded from the
milestone-2 checkpoints. openEMS is excluded here: its samples cannot pass
through the perturbation/Jacobian machinery unchanged (different freq grid)
and its within-OOD pool is only 20 structures.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from emsurr import dataset, models, planar_windings, splits, sqchip, uncertainty
from emsurr.external_tab import FROZEN, TabEnsemble
from emsurr.risk_signals import SIGNALS, tab_signals

RC = Path("results/riskcal")
EK = dict(n_members=FROZEN["n_members"], hidden=FROZEN["hidden"],
          depth=FROZEN["depth"], epochs=FROZEN["epochs"], lr=FROZEN["lr"])


class VecView:
    """Adapter: a synth TorchRegressor viewed as a tabular model on
    (feature vector -> normalized target vector)."""

    def __init__(self, reg):
        self.reg, self.norm, self.model = reg, reg.norm, reg.model

    def predict(self, x):
        xt = torch.tensor(self.norm(x), dtype=torch.float32)
        self.model.eval()
        with torch.no_grad():
            return self.model(xt).numpy()

    def embed(self, x):
        xt = torch.tensor(self.norm(x), dtype=torch.float32)
        with torch.no_grad():
            return self.model.net[:-1](xt).numpy()


class VecEns:
    def __init__(self, ens):
        self.members = [VecView(m) for m in ens.members]


def synth_signals(ens, train_s, pool_s):
    xtr = models.features(train_s)
    ytr = ens.members[0].ynorm(models.targets(train_s))
    x = models.features(pool_s)
    return tab_signals(VecEns(ens), xtr, ytr, x)


def save(dep, pools):
    RC.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(RC / f"sig2_{dep}.npz",
                        **{f"{p}/{k}": v for p, t in pools.items() for k, v in t.items()})
    print(f"wrote sig2_{dep}.npz")


def synth():
    cfg = yaml.safe_load(Path("configs/baseline.yaml").read_text())
    mcfg = cfg["models"]["mlp"]
    MK = dict(hidden=mcfg["hidden"], depth=mcfg["depth"], epochs=mcfg["epochs"], lr=mcfg["lr"])
    samples = dataset.load_h5(cfg["data"]["path"])
    sp = splits.make_splits(samples, seed=cfg["splits"]["seed"],
                            ood_families=tuple(cfg["splits"]["ood_families"]))
    d = sp["ood_topology"]
    train, id_pool, ood_pool = d["train"], d["val"], d["test"]

    rs = np.random.RandomState(1)
    ids = [s["sample_id"] for s in train]
    rs.shuffle(ids)
    by_id = {s["sample_id"]: s for s in train}
    cut = int(0.85 * len(ids))
    strain = [by_id[i] for i in ids[:cut]]
    sval = [by_id[i] for i in ids[cut:]]

    fams = sorted({s["topology_family"] for s in train})
    parts = []
    for f in fams:
        tr_f = [s for s in strain if s["topology_family"] != f]
        va_f = [s for s in sval if s["topology_family"] != f]
        held = [s for s in strain + sval if s["topology_family"] == f]
        print(f"synth LOFO fold {f}")
        ens_f = uncertainty.Ensemble(n_members=5, **MK).fit(tr_f, va_f)
        pool = va_f + held
        sig = synth_signals(ens_f, tr_f, pool)
        pred, _ = ens_f.predict_with_uncertainty(pool)
        sig["err"] = uncertainty.per_sample_error(pred, np.stack([s["s"] for s in pool]))
        sig["is_ood"] = np.array([0.0] * len(va_f) + [1.0] * len(held))
        sig["family"] = np.array([s["topology_family"] for s in pool])
        parts.append(sig)
    pseudo = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}

    # frozen final models from checkpoints
    sys.path.insert(0, "scripts")
    from riskcal_extract_synth import load_final
    _, ens_F = load_final("final", train, id_pool)
    pool = id_pool + ood_pool
    sig = synth_signals(ens_F, train, pool)
    pred, _ = ens_F.predict_with_uncertainty(pool)
    sig["err"] = uncertainty.per_sample_error(pred, np.stack([s["s"] for s in pool]))
    sig["is_ood"] = np.array([0.0] * len(id_pool) + [1.0] * len(ood_pool))
    sig["family"] = np.array([s["topology_family"] for s in pool])
    save("synth", dict(pseudo=pseudo, eval_synth=sig))


def planar():
    rng = np.random.default_rng(0)
    data = planar_windings.load()
    xc, yc = data["CORE"]["x"], np.log10(data["CORE"]["y"])[:, None]
    perm = rng.permutation(len(xc))
    n = len(xc)
    tr, va, te = perm[: int(0.8 * n)], perm[int(0.8 * n) : int(0.9 * n)], perm[int(0.9 * n) :]
    xtr, ytr, xva, yva = xc[tr], yc[tr], xc[va], yc[va]
    xte, yte = xc[te], yc[te]
    xo, yo = data["OOD"]["x"], np.log10(data["OOD"]["y"])[:, None]
    xm, ym = data["MEAS"]["x"], np.log10(data["MEAS"]["y"])[:, None]

    qlo, qhi = np.quantile(xtr, 0.10, 0), np.quantile(xtr, 0.90, 0)
    ctr = ((xtr >= qlo) & (xtr <= qhi)).all(1)
    cva = ((xva >= qlo) & (xva <= qhi)).all(1)
    x_ctr, y_ctr, x_cva, y_cva = xtr[ctr], ytr[ctr], xva[cva], yva[cva]
    x_ext = np.concatenate([xtr[~ctr], xva[~cva]])
    y_ext = np.concatenate([ytr[~ctr], yva[~cva]])

    print("planar central (pseudo) ensemble")
    ens_c = TabEnsemble(**EK).fit(x_ctr, y_ctr, x_cva, y_cva)
    xp = np.concatenate([x_cva, x_ext])
    yp = np.concatenate([y_cva, y_ext])
    sig = tab_signals(ens_c, x_ctr, y_ctr, xp)
    mu, _ = ens_c.predict_with_uncertainty(xp)
    sig["err"] = np.abs(10 ** mu[:, 0] / 10 ** yp[:, 0] - 1.0)
    sig["is_ood"] = np.array([0.0] * len(x_cva) + [1.0] * len(x_ext))
    pseudo = sig

    print("planar final ensemble")
    ens_f = TabEnsemble(**EK).fit(xtr, ytr, xva, yva)
    pools = dict(pseudo=pseudo)
    for name, (xx, yy, ood) in dict(
            eval_ood=(np.concatenate([xte, xo]), np.concatenate([yte, yo]),
                      [0] * len(xte) + [1] * len(xo)),
            eval_meas=(np.concatenate([xte, xm]), np.concatenate([yte, ym]),
                       [0] * len(xte) + [1] * len(xm))).items():
        s2 = tab_signals(ens_f, xtr, ytr, xx)
        mu, _ = ens_f.predict_with_uncertainty(xx)
        s2["err"] = np.abs(10 ** mu[:, 0] / 10 ** yy[:, 0] - 1.0)
        s2["is_ood"] = np.array(ood, float)
        pools[name] = s2
    save("planar", pools)


def sqchip_dep():
    recs = sqchip.load_records()
    y = np.stack([r["y"] for r in recs])
    xp, _ = sqchip.param_matrix(recs)
    fam = np.array([r["family"] for r in recs])
    from collections import Counter

    for hold in ["batch300", "batchnear50", "sqchip"]:
        print("sqchip fold", hold)
        rest = np.where(fam != hold)[0]
        ho = np.where(fam == hold)[0]
        rng = np.random.default_rng(1)
        rp = rng.permutation(rest)
        tr, va = rp[: int(0.9 * len(rp))], rp[int(0.9 * len(rp)) :]
        ystd = y[tr].std(0) + 1e-9

        pf = Counter(fam[tr]).most_common(1)[0][0]
        ptr = tr[fam[tr] != pf]
        pva = va[fam[va] != pf]
        pho = np.concatenate([tr[fam[tr] == pf], va[fam[va] == pf]])
        ens_c = TabEnsemble(**EK).fit(xp[ptr], y[ptr], xp[pva], y[pva])
        pool = np.concatenate([pva, pho])
        sig = tab_signals(ens_c, xp[ptr], y[ptr], xp[pool])
        mu, _ = ens_c.predict_with_uncertainty(xp[pool])
        sig["err"] = (np.abs(mu - y[pool]) / ystd).mean(1)
        sig["is_ood"] = np.array([0.0] * len(pva) + [1.0] * len(pho))
        pseudo = sig

        ens_f = TabEnsemble(**EK).fit(xp[tr], y[tr], xp[va], y[va])
        ev = np.concatenate([va, ho])
        s2 = tab_signals(ens_f, xp[tr], y[tr], xp[ev])
        mu, _ = ens_f.predict_with_uncertainty(xp[ev])
        s2["err"] = (np.abs(mu - y[ev]) / ystd).mean(1)
        s2["is_ood"] = np.array([0.0] * len(va) + [1.0] * len(ho))
        save(f"sqchip_{hold}", dict(pseudo=pseudo, eval=s2))


if __name__ == "__main__":
    planar()
    sqchip_dep()
    synth()
