"""Solve the next K uncached pool designs (foreground chunk)."""
import json, sys, shutil
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from emsurr import pkg_task as PT

K = int(sys.argv[1]) if len(sys.argv) > 1 else 6
designs = [PT.clip(d) for d in json.loads(Path("results/pkg_pool_designs.json").read_text())]
sv = PT.Solver()
remaining = [d for d in designs if not sv.cached(d)]
free_mb = shutil.disk_usage("C:/").free / 1e6
print(f"remaining {len(remaining)}/256, free {free_mb:.0f} MB")
if free_mb < 300:
    print("DISK GUARD"); sys.exit(0)
sv.solve_batch(remaining[:K])
print("cached now", 256 - len([d for d in designs if not sv.cached(d)]))
