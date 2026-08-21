"""Evaluation splits: IID, OOD-topology, and parameter extrapolation.

Discipline: `val` is for model selection; `test` is final held-out and must
never influence tuning. OOD test families are excluded from both train and
val entirely.
"""

from __future__ import annotations

import numpy as np

from .synth import PARAM_NAMES


def _split_ids(ids: list[str], rng, frac=(0.7, 0.15, 0.15)):
    ids = list(ids)
    rng.shuffle(ids)
    n = len(ids)
    a, b = int(frac[0] * n), int((frac[0] + frac[1]) * n)
    return set(ids[:a]), set(ids[a:b]), set(ids[b:])


def make_splits(samples: list[dict], seed: int = 0, ood_families=("via_lc", "stub_short")) -> dict:
    """Returns {split_name: {"train": [...], "val": [...], "test": [...]}}."""
    rng = np.random.default_rng(seed)
    py_rng = np.random.RandomState(seed)
    by_id = {s["sample_id"]: s for s in samples}
    splits = {}

    # IID: random split over all samples
    tr, va, te = _split_ids(list(by_id), py_rng)
    splits["iid"] = {
        "train": [by_id[i] for i in tr],
        "val": [by_id[i] for i in va],
        "test": [by_id[i] for i in te],
    }

    # OOD-topology: entire families held out for test; train/val on the rest
    in_dom = [s for s in samples if s["topology_family"] not in ood_families]
    out_dom = [s for s in samples if s["topology_family"] in ood_families]
    tr, va, _ = _split_ids([s["sample_id"] for s in in_dom], py_rng, (0.85, 0.15, 0.0))
    splits["ood_topology"] = {
        "train": [by_id[i] for i in tr],
        "val": [by_id[i] for i in va],
        "test": out_dom,
    }

    # Parameter extrapolation: permittivity above the training range
    er_ix = PARAM_NAMES.index("er")
    er = np.array([s["params"][er_ix] for s in samples])
    hi = np.quantile(er, 0.85)
    in_rng = [s for s in samples if s["params"][er_ix] <= hi]
    out_rng = [s for s in samples if s["params"][er_ix] > hi]
    tr, va, _ = _split_ids([s["sample_id"] for s in in_rng], py_rng, (0.85, 0.15, 0.0))
    splits["extrap_er"] = {
        "train": [by_id[i] for i in tr],
        "val": [by_id[i] for i in va],
        "test": out_rng,
    }

    # Geometry extrapolation: total length above training support
    l_ix = [PARAM_NAMES.index(k) for k in ("len1", "len2", "len3", "stub_len")]
    tot = np.array([s["params"][l_ix].sum() for s in samples])
    hi = np.quantile(tot, 0.85)
    in_rng = [s for s, v in zip(samples, tot) if v <= hi]
    out_rng = [s for s, v in zip(samples, tot) if v > hi]
    tr, va, _ = _split_ids([s["sample_id"] for s in in_rng], py_rng, (0.85, 0.15, 0.0))
    splits["extrap_len"] = {
        "train": [by_id[i] for i in tr],
        "val": [by_id[i] for i in va],
        "test": out_rng,
    }
    return splits
