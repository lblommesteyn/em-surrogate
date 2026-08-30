"""Predeclared space-filling package training pool: 256 scrambled-Sobol
designs (seed 5) over the 13 package bounds, solved in SEQUENCE ORDER so
the budget subsets 16 c 32 c 64 c 128 c 256 are nested by construction.
Content-cached; resumable. Writes results/pkg_pool_designs.json."""
import json, sys
from pathlib import Path
import numpy as np
from scipy.stats import qmc

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from emsurr import pkg_task as PT

N = 256
sob = qmc.Sobol(d=len(PT.KEYS), scramble=True, seed=5)
u = sob.random(N)
designs = []
for row in u:
    d = {k: lo + r * (hi - lo) for r, (k, (lo, hi)) in zip(row, PT.BOUNDS.items())}
    designs.append(PT.clip(d))
Path("results/pkg_pool_designs.json").write_text(json.dumps(designs, indent=0))

import shutil as _sh

def disk_free_mb():
    return _sh.disk_usage("C:\\").free / 1e6

sv = PT.Solver()
B = 8
for i in range(0, N, B):
    if disk_free_mb() < 300:
        print("DISK GUARD: <300MB free, stopping cleanly", flush=True)
        break
    batch = designs[i:i + B]
    todo = sum(0 if sv.cached(d) else 1 for d in batch)
    sv.solve_batch(batch)
    print(f"pool {i + len(batch)}/{N} solved (batch new={todo}, total_calls={sv.calls})",
          flush=True)
print("pool complete; solver calls this run:", sv.calls)
