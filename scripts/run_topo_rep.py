"""Topology-aware representation study.

Stage 1 (train-side only): fit 6 encoder variants (ordered/unordered x
resp/metric/both) on the synthetic training families; measure distance-
response correlation, nearest-neighbor response error, and leave-one-
training-family-out metric generalization. PRIMARY variant is pre-declared
as the one with the best WORST-family LOFO pair Spearman (train-side
criterion, chosen before any frozen pool is touched).

Stage 2 (frozen): evaluate embedding-kNN distance of the primary (and all)
variants on the frozen synth OOD pool and openEMS, against cached frozen-
surrogate errors; compare with input-kNN and old surrogate embedding-kNN.
SQChip: compact tabular metric encoder on param+geometry features, frozen
family folds. Planar is skipped: single topology, no structural vocabulary.

Writes results/topo_rep_metrics.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from emsurr import dataset, splits
from emsurr.novelty import _rank01, auroc, content_features, oracle_recovery, risk_coverage
from emsurr.models import Normalizer
from emsurr.risk_cal import budget_to_catch, spearman
from emsurr.topo_rep import TopoEmbedding, knn_dist

RC = Path("results/riskcal")
BUDGETS = (0.05, 0.1, 0.2, 0.3, 0.5)
VARIANTS = [(o, obj) for o in (True, False) for obj in ("resp", "metric", "both")]
LOFO_EPOCHS = 100


def pair_spearman(emb_a, S_a, emb_b, S_b, n_pairs=2000, seed=0):
    rng = np.random.default_rng(seed)
    ii = rng.integers(0, len(emb_a), n_pairs)
    jj = rng.integers(0, len(emb_b), n_pairs)
    de = np.linalg.norm(emb_a[ii] - emb_b[jj], axis=1)
    dr = np.abs(S_a[ii] - S_b[jj]).mean(axis=(1, 2, 3))
    return spearman(de, dr)


def nn_resp_err(emb_q, S_q, emb_r, S_r):
    d = np.linalg.norm(emb_q[:, None] - emb_r[None], axis=-1)
    nn = d.argmin(1)
    return float(np.abs(S_q - S_r[nn]).mean())


def load_sig(dep):
    z = np.load(RC / f"sig2_{dep}.npz", allow_pickle=True)
    pools = {}
    for key in z.files:
        p, k = key.split("/", 1)
        pools.setdefault(p, {})[k] = z[key]
    return pools


def eval_distance(dist_id, dist_ood, err):
    n_id = len(dist_id)
    d_all = np.concatenate([dist_id, dist_ood])
    rc = risk_coverage(d_all, err, BUDGETS)
    orec = oracle_recovery(rc)
    e_ood = err[n_id:]
    return dict(
        auroc=auroc(dist_id, dist_ood),
        spearman_within_ood=spearman(dist_ood, e_ood),
        cap10_within_ood=float(len(set(np.argsort(-dist_ood)[: max(1, len(e_ood) // 10)])
                                   & set(np.argsort(-e_ood)[: max(1, len(e_ood) // 10)]))
                               / max(1, len(e_ood) // 10)),
        budget_catch90=budget_to_catch(d_all, err, 0.90),
        oracle_recovery={str(b): orec[b] for b in BUDGETS},
    )


def main():
    cfg = yaml.safe_load(Path("configs/baseline.yaml").read_text())
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
    S = {id(s): None for s in samples}
    Sv = np.stack([s["s"] for s in sval])
    St = np.stack([s["s"] for s in strain])
    fams = sorted({s["topology_family"] for s in train})

    res = dict(variants={}, selection=None, frozen={}, sqchip={}, hard_cases={})

    # ---------- stage 1: train-side quality + LOFO generalization
    for ordered, obj in VARIANTS:
        tag = f"{'ord' if ordered else 'set'}_{obj}"
        print("== variant", tag)
        enc = TopoEmbedding(ordered=ordered, objective=obj).fit(strain, sval)
        e_tr, e_va = enc.embed(strain), enc.embed(sval)
        v = dict(
            val_pair_spearman=pair_spearman(e_va, Sv, e_va, Sv),
            nn_resp_err=nn_resp_err(e_va, Sv, e_tr, St),
        )
        lofo = {}
        for f in fams:
            tr_f = [s for s in strain if s["topology_family"] != f]
            va_f = [s for s in sval if s["topology_family"] != f]
            held = [s for s in strain if s["topology_family"] == f]
            ef = TopoEmbedding(ordered=ordered, objective=obj,
                               epochs=LOFO_EPOCHS).fit(tr_f, va_f)
            lofo[f] = pair_spearman(ef.embed(held), np.stack([s["s"] for s in held]),
                                    ef.embed(tr_f), np.stack([s["s"] for s in tr_f]))
        v["lofo_pair_spearman"] = lofo
        v["lofo_mean"] = float(np.mean(list(lofo.values())))
        v["lofo_min"] = float(np.min(list(lofo.values())))
        res["variants"][tag] = v
        print(f"  val_sp={v['val_pair_spearman']:.3f} nn_err={v['nn_resp_err']:.3f} "
              f"lofo mean={v['lofo_mean']:.3f} min={v['lofo_min']:.3f}")

    primary = max(res["variants"], key=lambda t: res["variants"][t]["lofo_min"])
    res["selection"] = dict(primary=primary, criterion="max LOFO min pair-Spearman")
    print("PRIMARY:", primary)

    # ---------- baseline: content-feature distance quality on same pairs
    cf = Normalizer().fit(content_features(strain))
    e_tr0 = cf(content_features(strain))
    e_va0 = cf(content_features(sval))
    res["baseline_content"] = dict(
        val_pair_spearman=pair_spearman(e_va0, Sv, e_va0, Sv),
        nn_resp_err=nn_resp_err(e_va0, Sv, e_tr0, St))

    # ---------- stage 2: frozen evaluation (synth OOD + openEMS)
    sig = load_sig("synth")["eval_synth"]
    err_synth = np.asarray(sig["err"], float)
    z0 = np.load(RC / "synth.npz", allow_pickle=True)
    err_oems = np.asarray(z0["eval_openems/err"], float)
    from emsurr.openems_load import load_openems
    oems = load_openems()

    for tag in res["variants"]:
        o = tag.startswith("ord")
        obj = tag.split("_", 1)[1]
        enc = TopoEmbedding(ordered=o, objective=obj).fit(strain, sval)
        e_ref = enc.embed(train)
        d_id = knn_dist(enc.embed(id_pool), e_ref)
        d_ood = knn_dist(enc.embed(ood_pool), e_ref)
        d_oe = knn_dist(enc.embed(oems), e_ref)
        res["frozen"][tag] = dict(
            synth=eval_distance(d_id, d_ood, err_synth),
            openems=eval_distance(d_id, d_oe, err_oems))
        if tag == primary:
            # hard-case retrieval: response distance to NN under old vs new rep
            e_ood_old = cf(content_features(ood_pool))
            S_ood = np.stack([s["s"] for s in ood_pool])
            S_tr = np.stack([s["s"] for s in train])
            nn_old = np.linalg.norm(e_ood_old[:, None] - cf(content_features(train))[None],
                                    axis=-1).argmin(1)
            nn_new = np.linalg.norm(enc.embed(ood_pool)[:, None] - e_ref[None],
                                    axis=-1).argmin(1)
            d_old = np.abs(S_ood - S_tr[nn_old]).mean(axis=(1, 2, 3))
            d_new = np.abs(S_ood - S_tr[nn_new]).mean(axis=(1, 2, 3))
            de = np.linalg.norm(enc.embed(ood_pool)[:, None] - e_ref[None], axis=-1).min(1)
            alias = float(np.mean((_rank01(de) < 0.1)
                                  & (_rank01(np.minimum(d_old, d_new)) > 0.9)))
            res["hard_cases"] = dict(
                nn_resp_dist_old_mean=float(d_old.mean()),
                nn_resp_dist_new_mean=float(d_new.mean()),
                frac_new_retrieves_closer=float(np.mean(d_new < d_old)),
                alias_rate_close_emb_far_resp=alias)
        s1, s2 = res["frozen"][tag]["synth"], res["frozen"][tag]["openems"]
        print(f"[{tag}] synth AUROC={s1['auroc']:.3f} wOOD sp={s1['spearman_within_ood']:+.3f} "
              f"orec20={s1['oracle_recovery']['0.2']:.2f} | "
              f"oems AUROC={s2['auroc']:.3f} sp={s2['spearman_within_ood']:+.3f}")

    # cached baselines for comparison (same pools/errors)
    for nm in ("knn_input", "knn_emb", "ens_var"):
        v = np.asarray(sig[nm], float)
        n_id = int((sig["is_ood"] == 0).sum())
        res["frozen"][f"baseline_{nm}"] = dict(
            synth=eval_distance(v[:n_id], v[n_id:], err_synth))

    Path("results/topo_rep_metrics.json").write_text(json.dumps(res, indent=1))
    print("wrote results/topo_rep_metrics.json")


if __name__ == "__main__":
    main()
