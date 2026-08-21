"""Phase 4: score openEMS-generated structure families with novelty methods
fitted only on the synthetic training families.

The openEMS samples are mapped into the canonical representation honestly:
parameters go into the shared layout where a semantic match exists (overlapping
numeric ranges with training, so params alone are not trivially OOD) and
element tokens are built from the existing vocabulary where the structure
allows (dstub is exactly expressible; patch approximately; gap not at all).

`oems_patch` is the untouched final family: scored once, in this script,
for the final report.
"""

import json
from pathlib import Path

import h5py
import numpy as np
import yaml

from emsurr import dataset, models, novelty, physics, splits, synth, uncertainty

cfg = yaml.safe_load(Path("configs/baseline.yaml").read_text())
mcfg = cfg["models"]["mlp"]

SUB_H = 254e-6


def tokenize(fam, p):
    """Element tokens from existing vocabulary, where expressible."""
    w = p["w"] * 1e-6
    er = p["epr"]
    z0, eeff = physics.microstrip_z0_eeff(w, SUB_H, er)
    feed = (synth.EL_LINE, z0, eeff, 0.0, 15e-3)
    if fam == "dstub":
        toks = [feed]
        for key in ("s1", "s2"):
            sw = p["sw"] * 1e-6
            z0s, eeffs = physics.microstrip_z0_eeff(sw, SUB_H, er)
            toks.append((synth.EL_STUB_OPEN, z0s, eeffs, 0.0, p[key] * 1e-6))
        toks.append(feed)
        return np.array(toks)
    if fam == "patch":
        pw = p["pw"] * 1e-6
        z0p, eeffp = physics.microstrip_z0_eeff(pw, SUB_H, er)
        return np.array([feed, (synth.EL_LINE, z0p, eeffp, 0.0, p["pl"] * 1e-6), feed])
    if fam == "gap":
        return np.array([feed, feed])  # the gap itself is not expressible
    raise ValueError(fam)


def load_openems(path="results/openems_families.h5"):
    out = []
    with h5py.File(path, "r") as h:
        for sid in h:
            g = h[sid]
            names = g["param_names"][()].decode().split(",")
            vals = g["params"][:]
            p = dict(zip(names, vals))
            fam = g.attrs["topology_family"].removeprefix("oems_")
            params = {k: 0.0 for k in synth.PARAM_NAMES}
            # non-design constants (conductor sigma/thickness) take the
            # training values: they are not what makes these structures novel,
            # and leaving them 0 trivially separates OOD via constant columns
            params.update(w1=p["w"] * 1e-6, h=SUB_H, er=p["epr"], len1=15e-3,
                          len2=15e-3, sigma=5.8e7, t=35e-6, tand=0.001)
            if fam == "dstub":
                params["w2"] = p["sw"] * 1e-6
                params["stub_len"] = p["s1"] * 1e-6
            if fam == "patch":
                params["w2"] = p["pw"] * 1e-6
                params["len3"] = p["pl"] * 1e-6
            if fam == "gap":
                params["len3"] = p["g"] * 1e-6
            out.append(dict(
                sample_id=sid, topology_family=g.attrs["topology_family"],
                ports=2, params=np.array([params[k] for k in synth.PARAM_NAMES]),
                elements=tokenize(fam, p), freq=g["freq"][:], s=g["s"][:],
            ))
    return out


# refit final-stage models deterministically (same seeds/config as run_novelty)
samples = dataset.load_h5(cfg["data"]["path"])
sp = splits.make_splits(samples, seed=cfg["splits"]["seed"],
                        ood_families=tuple(cfg["splits"]["ood_families"]))
d = sp["ood_topology"]
train, id_pool = d["train"], d["val"]
mlp = models.TorchRegressor(hidden=mcfg["hidden"], depth=mcfg["depth"],
                            epochs=mcfg["epochs"], lr=mcfg["lr"]).fit(train, id_pool)
ens = uncertainty.Ensemble(n_members=cfg["uncertainty"]["n_members"],
                           hidden=mcfg["hidden"], depth=mcfg["depth"],
                           epochs=mcfg["epochs"], lr=mcfg["lr"]).fit(train, id_pool)

sel = json.loads(Path("results/novelty_metrics.json").read_text())["selection"]
k, alpha = sel["best_k"], sel["best_alpha"]

knn_in = novelty.KNNInputNovelty(k=k).fit(train)
maha = novelty.MahalanobisEmbedding().fit(train, mlp)
knn_e = novelty.KNNEmbedding(k=k).fit(train, mlp)
ens_sc = novelty.EnsembleVariance(ens)
scorers = [knn_in, maha, knn_e, ens_sc,
           novelty.CombinedScore(knn_in, ens_sc, alpha=alpha)]

oems = load_openems()
results = {}
for sc in scorers:
    pool = id_pool + oems
    s_all = sc.score(pool)
    s_id, s_o = s_all[: len(id_pool)], s_all[len(id_pool):]
    fams = np.array([s["topology_family"] for s in oems])
    res = dict(auroc_all=novelty.auroc(s_id, s_o))
    for fam in sorted(set(fams)):
        res[f"auroc_{fam}"] = novelty.auroc(s_id, s_o[fams == fam])
        # detection rate at a threshold set to the 95th percentile of ID scores
        thr = np.quantile(s_id, 0.95)
        res[f"deferred_{fam}@id95"] = float((s_o[fams == fam] > thr).mean())
    results[sc.name] = res
    print(sc.name, json.dumps(res))

Path("results/openems_novelty.json").write_text(json.dumps(results, indent=2))
print("wrote results/openems_novelty.json")
