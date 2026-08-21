"""Loader for the real TUHH SI/PI-Database layout.

Expected layout (see docs/data_audit.md), placed manually after the access
request is granted:

    data/raw/tuhh/<dataset-id>/parameter.csv
    data/raw/tuhh/<dataset-id>/variation/<SIMULATION>.sNp

The loader converts each row of parameter.csv plus its touchstone file into
the same canonical sample dict used by the synthetic data, so downstream
splits/models/metrics run unchanged. Untested against real archives (data is
form-gated); paths and column names follow the published structure docs.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import skrf


def family_map(config_path: str | Path = "configs/tuhh_families.yaml") -> dict:
    """A-priori dataset-ID -> super_family mapping (never derived from model
    results); used to build leave-one-group-out OOD splits on TUHH data."""
    import yaml

    cfg = yaml.safe_load(Path(config_path).read_text())
    return {k: v["super_family"] for k, v in cfg["families"].items()}


def load_tuhh_family(root: str | Path, family_id: str) -> list[dict]:
    root = Path(root)
    fam_dir = root / family_id
    df = pd.read_csv(fam_dir / "parameter.csv")
    param_cols = [c for c in df.columns if c.upper() != "SIMULATION"]
    var_dir = fam_dir / "variation"
    samples = []
    for _, row in df.iterrows():
        sim = str(row["SIMULATION"])
        matches = list(var_dir.glob(f"{sim}.s*p")) or list(var_dir.glob(f"*{sim}*.s*p"))
        if not matches:
            continue
        net = skrf.Network(str(matches[0]))
        samples.append(
            dict(
                sample_id=f"{family_id}_{sim}",
                topology_family=family_id,
                ports=net.nports,
                params=row[param_cols].to_numpy(dtype=float),
                elements=np.zeros((0, 5)),
                freq=net.f,
                s=net.s,
            )
        )
    return samples
