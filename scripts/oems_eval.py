"""openEMS evaluator for the design task: reads a JSON list of designs,
solves each, appends results to an .npz cache. Runs inside tools/oems-venv.
Design: microstrip line (width w, er fixed 3.5) with two open stubs
(lengths s1,s2, width sw, positions x1,x2 on the left half) and a series
gap g at x=0. Units um. Reuses the frozen openems_families.simulate().
"""
import json, os, sys, time
import numpy as np
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts"))
from openems_families import simulate, FREQ, sanity  # noqa

EPR = 3.5

def solve(d, tag):
    w, s1, s2, sw, x1, x2, g = (d[k] for k in ("w","s1","s2","sw","x1","x2","g"))
    boxes = [((x1-sw/2, w/2), (x1+sw/2, w/2+s1)),
             ((x2-sw/2, -w/2-s2), (x2+sw/2, -w/2))]
    s, stats = simulate(w, EPR, boxes, max(s1, s2)+w, tag, gap=g, verbose=True)
    return s, stats

if __name__ == "__main__":
    inp, outp = sys.argv[1], sys.argv[2]
    designs = json.loads(open(inp).read())
    out = {}
    for i, d in enumerate(designs):
        t0=time.perf_counter()
        s, stats = solve(d, f"design_{os.getpid()}_{i}")
        chk = sanity(s)
        out[str(i)] = s
        out[f"meta_{i}"] = np.array([stats["wall_s"], float(chk["finite"]), chk["max_sv"]])
        print(f"design {i}: wall={stats['wall_s']}s sv={chk['max_sv']:.3f}", flush=True)
    np.savez(outp, freq=FREQ, **out)
