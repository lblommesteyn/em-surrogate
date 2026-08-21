"""End-to-end tests of the TUHH ingestion/canonicalization pipeline on a
mock fixture in the real TUHH directory layout (parameter.csv + variation/
touchstones). The fixture is synthetic: no TUHH data is stored in the repo."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import skrf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from emsurr import dataset, tuhh_prep
import ingest_tuhh
import run_tuhh

FAM2SF = {"SI-1": "g_a", "SI-2": "g_a", "PI-1": "g_b", "PI-7": "g_c"}


def _write_net(path, f, nports=2, phase=1.0):
    if nports == 2:
        s = np.zeros((len(f), 2, 2), complex)
        s[:, 0, 0] = s[:, 1, 1] = 0.2
        s[:, 0, 1] = s[:, 1, 0] = 0.7 * np.exp(-1j * phase * f / f.max())
    else:
        s = 0.3 * np.exp(-1j * f / f.max()).reshape(-1, 1, 1)
    net = skrf.Network(frequency=skrf.Frequency.from_f(f, unit="hz"), s=s)
    net.write_touchstone(str(path.with_suffix("")))


@pytest.fixture(scope="module")
def mock_root(tmp_path_factory):
    root = tmp_path_factory.mktemp("tuhh")
    f = np.linspace(1e8, 1e10, 40)
    specs = {
        "SI-1": dict(cols=["W", "L"], n=6),
        "SI-2": dict(cols=["W", "D"], n=6),
        "PI-1": dict(cols=["H", "L"], n=6),
        "PI-7": dict(cols=["H", "R"], n=6),
    }
    rng = np.random.default_rng(0)
    for fam, spec in specs.items():
        d = root / fam
        (d / "variation").mkdir(parents=True)
        rows = []
        for i in range(spec["n"]):
            sim = f"run{i}"
            vals = rng.uniform(1, 2, len(spec["cols"]))
            rows.append([sim] + list(vals))
            _write_net(d / "variation" / f"{sim}.s2p", f, phase=1 + vals[0])
        # exclusion cases in SI-1: missing touchstone, single-port, duplicate
        if fam == "SI-1":
            rows.append(["missing", 1.5, 1.5])
            rows.append(["oneport", 1.6, 1.6])
            _write_net(d / "variation" / "oneport.s1p", f, nports=1)
            rows.append(["dup0", *rows[0][1:]])
            _write_net(d / "variation" / "dup0.s2p", f)
        pd.DataFrame(rows, columns=["SIMULATION"] + spec["cols"]).to_csv(
            d / "parameter.csv", index=False
        )
    return root


def test_scan_manifest(mock_root):
    rows, raw = ingest_tuhh.scan(mock_root)
    df = pd.DataFrame(rows)
    assert len(df) == 27  # 24 clean + 3 exclusion rows
    assert df.included.sum() == 24 == len(raw)
    reasons = df[~df.included].set_index("simulation").reason
    assert reasons["missing"] == "missing_touchstone"
    assert reasons["oneport"] == "single_port"
    assert reasons["dup0"] == "duplicate"
    assert df[df.simulation == "dup0"].dup_of.iloc[0] == "SI-1_run0"


def test_canonicalize(mock_root):
    _, raw = ingest_tuhh.scan(mock_root)
    canon, meta = tuhh_prep.canonicalize(raw, n_freq=32)
    assert len(canon) == len(raw)
    assert all(s["s"].shape == (32, 2, 2) for s in canon)
    assert all(len(s["params"]) == len(meta["schema"]) for s in canon)
    # family-prefixed schema: SI-1 params occupy only SI-1 columns
    s0 = next(s for s in canon if s["topology_family"] == "SI-1")
    idx_w = meta["schema"].index("SI-1:W")
    assert s0["params"][idx_w] != 0
    other = [i for i, n in enumerate(meta["schema"]) if not n.startswith("SI-1:")]
    assert np.all(s0["params"][other] == 0)
    dataset.sanity_check(canon)


def test_splits_and_extrap(mock_root):
    _, raw = ingest_tuhh.scan(mock_root)
    canon, meta = tuhh_prep.canonicalize(raw, n_freq=32)
    sp = run_tuhh.make_tuhh_splits(canon, FAM2SF, seed=0)
    assert set(sp) >= {"iid", "loso_g_a", "loso_g_b", "loso_g_c",
                       "lodo_SI-1", "lodo_PI-7"}
    d = sp["loso_g_a"]
    trval_fams = {s["topology_family"] for s in d["train"] + d["val"]}
    assert trval_fams.isdisjoint({"SI-1", "SI-2"})
    assert {s["topology_family"] for s in d["test"]} == {"SI-1", "SI-2"}
    assert not dataset.check_leakage(d["train"], d["test"])
    # L is shared by SI-1 and PI-1 but not all four families -> no global
    # extrapolation split on this fixture
    reason, ex = run_tuhh.make_extrap_split(canon, meta["schema"], seed=0)
    assert ex is None
