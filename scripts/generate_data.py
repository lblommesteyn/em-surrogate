"""Generate the synthetic dataset, run sanity checks, save to HDF5."""

import json
import sys
from pathlib import Path

import yaml

from emsurr import synth, dataset

cfg = yaml.safe_load(Path(sys.argv[1] if len(sys.argv) > 1 else "configs/baseline.yaml").read_text())
d = cfg["data"]
samples = synth.generate(d["n_per_family"], seed=d["seed"])
report = dataset.sanity_check(samples, rtol_passivity=1e-3)
Path(d["path"]).parent.mkdir(parents=True, exist_ok=True)
dataset.save_h5(samples, d["path"])
Path("results").mkdir(exist_ok=True)
Path("results/sanity_report.json").write_text(json.dumps(report, indent=2, default=str))
print(f"saved {len(samples)} samples to {d['path']}")
print({k: (len(v) if isinstance(v, list) else v) for k, v in report.items()})
