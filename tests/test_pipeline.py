import numpy as np
import pytest

from emsurr import synth, dataset, splits, metrics, models


@pytest.fixture(scope="module")
def small():
    return synth.generate(n_per_family=12, seed=1)


def test_generation_shapes(small):
    s = small[0]
    assert s["s"].shape == (256, 2, 2)
    assert np.iscomplexobj(s["s"])
    assert len({x["topology_family"] for x in small}) == 7


def test_sanity_passes(small):
    rep = dataset.sanity_check(small, rtol_passivity=1e-3)
    assert rep["n"] == len(small)
    assert not rep["nan_samples"]
    assert not rep["grid_mismatch"]
    assert not rep["duplicates"]
    # passive structures must not amplify
    assert not rep["passivity_bad"]
    assert not rep["reciprocity_bad"]


def test_sanity_catches_nan(small):
    bad = dict(small[0])
    bad["s"] = bad["s"].copy()
    bad["s"][0, 0, 0] = np.nan
    bad["sample_id"] = "bad"
    with pytest.raises(ValueError):
        dataset.sanity_check(small + [bad])


def test_h5_roundtrip(tmp_path, small):
    p = str(tmp_path / "d.h5")
    dataset.save_h5(small[:5], p)
    back = dataset.load_h5(p)
    assert len(back) == 5
    a = {s["sample_id"]: s for s in small[:5]}
    for s in back:
        assert np.allclose(s["s"], a[s["sample_id"]]["s"])
        assert s["topology_family"] == a[s["sample_id"]]["topology_family"]


def test_splits_disjoint_and_leakfree(small):
    sp = splits.make_splits(small, seed=0)
    for name, d in sp.items():
        ids = [set(x["sample_id"] for x in d[k]) for k in ("train", "val", "test")]
        assert not (ids[0] & ids[2]) and not (ids[1] & ids[2]) and not (ids[0] & ids[1])
        assert not dataset.check_leakage(d["train"], d["test"])
    # OOD test families absent from train and val
    d = sp["ood_topology"]
    test_fams = {x["topology_family"] for x in d["test"]}
    for k in ("train", "val"):
        assert not test_fams & {x["topology_family"] for x in d[k]}


def test_knn_and_metrics(small):
    sp = splits.make_splits(small, seed=0)["iid"]
    knn = models.KNNBaseline(k=3).fit(sp["train"])
    pred = knn.predict(sp["test"])
    true = np.stack([s["s"] for s in sp["test"]])
    rep = metrics.full_report(pred, true)
    assert 0 < rep["complex_mae"] < 1.0
    assert rep["phase_mae_deg"] < 180


def test_vec_roundtrip(small):
    s = np.stack([x["s"] for x in small[:3]])
    v = models.s_to_vec(s)
    assert np.allclose(models.vec_to_s(v, 256), s)
