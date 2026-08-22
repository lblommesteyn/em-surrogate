"""Unit tests for external-dataset plumbing (no downloads required except
where the data directory already exists)."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from emsurr.external_tab import TabCombined, TabKNNInput, _knn_mean_dist
from emsurr.sqchip import _num


def test_num_parsing():
    assert _num("10.0um") == pytest.approx(1e-5)
    assert _num("3.0mm") == pytest.approx(3e-3)
    assert _num(7) == 7.0
    assert _num(True) == 1.0
    assert _num("12") == 12.0
    assert _num("banana") is None


def test_knn_chunked_matches_direct():
    rng = np.random.default_rng(0)
    ref, q = rng.normal(size=(50, 4)), rng.normal(size=(9, 4))
    direct = np.sort(np.linalg.norm(q[:, None] - ref[None], axis=-1), 1)[:, :3].mean(1)
    assert np.allclose(_knn_mean_dist(q, ref, 3, chunk=4), direct)


def test_knn_input_flags_shifted_points():
    rng = np.random.default_rng(0)
    tr = rng.normal(size=(200, 5))
    sc = TabKNNInput(k=3).fit(tr)
    near, far = rng.normal(size=(20, 5)), rng.normal(size=(20, 5)) + 8
    assert sc.score(far).min() > sc.score(near).max()


def test_combined_is_rank_average():
    class Fixed:
        def __init__(self, v):
            self.v = np.asarray(v, float)

        def score(self, x):
            return self.v

    c = TabCombined(Fixed([0, 1, 2]), Fixed([2, 1, 0]), alpha=0.5)
    assert np.allclose(c.score(np.zeros((3, 1))), [0.5, 0.5, 0.5])


def test_planar_loader_if_present():
    from emsurr import planar_windings

    if not planar_windings.XLSX.exists():
        pytest.skip("dataset not downloaded")
    d = planar_windings.load()
    assert d["CORE"]["x"].shape == (10110, 9)
    assert d["OOD"]["x"].shape == (1124, 9)
    assert d["MEAS"]["x"].shape == (55, 9)
    assert (d["CORE"]["y"] > 0).all()


def test_budget_to_catch():
    from emsurr.risk_cal import budget_to_catch

    err = np.arange(100, dtype=float)
    # perfect score = error itself: 5 catastrophic (>=95), catch all with top-5
    assert budget_to_catch(err, err, 0.9) <= 0.05
    # anti-correlated score: needs nearly the full budget
    assert budget_to_catch(-err, err, 0.9) > 0.9


def test_select_prefers_informative_signal():
    from emsurr.risk_cal import fit_calibration

    rng = np.random.default_rng(0)
    n = 400
    err = rng.gamma(2, 1, n)
    pool = dict(
        knn_input=err + rng.normal(0, 0.1, n),   # informative
        knn_emb=rng.normal(0, 1, n),             # noise
        ens_var=rng.normal(0, 1, n),             # noise
        maha_emb=rng.normal(0, 1, n),
        err=err,
    )
    cal = fit_calibration(pool, pool)
    assert cal.iid_choice[0] == "knn_input"
    assert cal.iid_choice[1] >= 0.7  # leans on the informative signal
