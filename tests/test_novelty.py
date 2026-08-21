import numpy as np
import pytest

from emsurr import novelty, synth


@pytest.fixture(scope="module")
def data():
    return synth.generate(n_per_family=15, seed=3)


def test_auroc_bounds():
    assert novelty.auroc(np.zeros(50), np.ones(50)) == 1.0
    rng = np.random.default_rng(0)
    a = novelty.auroc(rng.normal(size=500), rng.normal(size=500))
    assert 0.4 < a < 0.6


def test_knn_input_separates_family(data):
    train = [s for s in data if s["topology_family"] != "via_lc"]
    ood = [s for s in data if s["topology_family"] == "via_lc"]
    sc = novelty.KNNInputNovelty(k=3).fit(train)
    s_id = sc.score(train[:30])
    s_ood = sc.score(ood)
    assert novelty.auroc(s_id, s_ood) > 0.8


def test_combined_scores_pool_consistently(data):
    # rank-based scores are only meaningful on a jointly-scored pool
    train = data[:60]
    sc = novelty.KNNInputNovelty(k=3).fit(train)
    comb = novelty.CombinedScore(sc, sc, alpha=0.5)
    pool = data[60:90]
    s = comb.score(pool)
    assert s.min() >= 0 and s.max() <= 1
    assert np.allclose(s, novelty._rank01(sc.score(pool)))


def test_risk_coverage_oracle_is_best(data):
    rng = np.random.default_rng(1)
    errors = rng.exponential(size=200)
    noisy_score = errors + rng.normal(scale=0.5, size=200)
    rc = novelty.risk_coverage(noisy_score, errors, (0.2,))[0]
    assert rc["remaining_mae_oracle"] <= rc["remaining_mae"] + 1e-12
    rec = novelty.oracle_recovery([rc])[0.2]
    assert 0.0 < rec <= 1.0
