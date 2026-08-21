# -*- coding: utf-8 -*-
"""Generate small openEMS structure families for the novelty transfer test.

Families (structurally different from every synthetic training family, which
are all single-path cascades of line/stub/LC elements):

  gap    true series gap in the microstrip (series-C discontinuity)
  dstub  double-stub tee: open stubs on BOTH sides at the same plane
  patch  line loaded by a wide rectangular patch (2D low-Z pad)

`oems_patch` is the untouched final family: not inspected or used for any
method adjustment before the final report run.

Lessons from the first (stalled) attempt baked in:
  - domain shortened 30 mm -> 15 mm per side; ports keep >= 5 cells clearance
    from the discontinuity
  - fine mesh lines only local to the discontinuity; SmoothMeshLines grades
    them into the coarse bulk
  - NrTS hard cap: high-Q gap resonators decay slowly and EndCriteria alone
    can run essentially forever
  - the HDF5 file is opened per sample so a crash can't corrupt prior work

Usage:
  tools/oems-venv/Scripts/python scripts/openems_families.py --smoke
  tools/oems-venv/Scripts/python scripts/openems_families.py [gap|dstub|patch]
"""

import os
import sys
import tempfile
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.add_dll_directory(os.path.join(REPO, "tools", "openEMS"))

import numpy as np
import h5py
import psutil

from CSXCAD import ContinuousStructure
from openEMS import openEMS
from openEMS.physical_constants import C0

F_MAX = 7e9
FREQ = np.linspace(0.1e9, F_MAX, 128)
UNIT = 1e-6
SUB_T = 254.0
MSL_LEN = 18000.0
N_TS = 40000  # hard timestep cap


def simulate(msl_width, epr, top_boxes, y_extra, tag, gap=0.0, verbose=False):
    sim_path = os.path.join(tempfile.gettempdir(), f"oems_fam_{tag}")
    fdtd = openEMS(NrTS=N_TS, EndCriteria=1e-3)
    fdtd.SetGaussExcite(F_MAX / 2, F_MAX / 2)
    fdtd.SetBoundaryCond(["PML_8", "PML_8", "MUR", "MUR", "PEC", "MUR"])
    csx = ContinuousStructure()
    fdtd.SetCSX(csx)
    mesh = csx.GetGrid()
    mesh.SetDeltaUnit(UNIT)

    res = C0 / (F_MAX * np.sqrt(epr)) / UNIT / 30
    third = np.array([2 * res / 3, -res / 3]) / 4
    # coarse bulk in x, fine lines only near strip edges / discontinuity
    mesh.AddLine("x", 0)
    mesh.AddLine("x", [-MSL_LEN, MSL_LEN])
    if gap:
        mesh.AddLine("x", [-gap / 2, 0, gap / 2])
    for b in top_boxes:
        mesh.AddLine("x", [b[0][0], b[1][0]])
        mesh.AddLine("y", [b[0][1], b[1][1]])
    mesh.SmoothMeshLines("x", res)
    mesh.AddLine("y", 0)
    mesh.AddLine("y", msl_width / 2 + third)
    mesh.AddLine("y", -msl_width / 2 - third)
    mesh.SmoothMeshLines("y", res / 4)
    y_lo, y_hi = -10 * msl_width - y_extra, 10 * msl_width + y_extra
    mesh.AddLine("y", [y_lo, y_hi])
    mesh.SmoothMeshLines("y", res)
    mesh.AddLine("z", np.linspace(0, SUB_T, 5))
    mesh.AddLine("z", 2500)
    mesh.SmoothMeshLines("z", res)

    sub = csx.AddMaterial("substrate", epsilon=epr)
    sub.AddBox([-MSL_LEN, y_lo, 0], [MSL_LEN, y_hi, SUB_T])
    pec = csx.AddMetal("PEC")
    p1 = fdtd.AddMSLPort(
        1, pec, [-MSL_LEN, -msl_width / 2, SUB_T], [-gap / 2, msl_width / 2, 0],
        # PML_8 is ~8 coarse cells (~6 mm) deep: feed and measurement plane
        # must both sit beyond it, and meas after feed
        "x", "z", excite=-1, FeedShift=10 * res, MeasPlaneShift=MSL_LEN / 2,
        priority=10,
    )
    p2 = fdtd.AddMSLPort(
        2, pec, [MSL_LEN, -msl_width / 2, SUB_T], [gap / 2, msl_width / 2, 0],
        "x", "z", MeasPlaneShift=MSL_LEN / 2, priority=10,
    )
    for start, stop in top_boxes:
        pec.AddBox(list(start) + [SUB_T], list(stop) + [SUB_T], priority=10)

    nx, ny, nz = (mesh.GetQtyLines(d) for d in "xyz")
    stats = dict(nx=nx, ny=ny, nz=nz, cells=nx * ny * nz)
    if verbose:
        print(f"    mesh {nx}x{ny}x{nz} = {stats['cells']} cells")

    t0 = time.perf_counter()
    fdtd.Run(sim_path, cleanup=True)
    stats["wall_s"] = round(time.perf_counter() - t0, 1)
    mi = psutil.Process().memory_info()
    stats["peak_ram_mb"] = round(getattr(mi, "peak_wset", mi.rss) / 1e6, 1)

    for p in (p1, p2):
        p.CalcPort(sim_path, FREQ, ref_impedance=50)
    s = np.empty((len(FREQ), 2, 2), dtype=complex)
    s[:, 0, 0] = s[:, 1, 1] = p1.uf_ref / p1.uf_inc
    s[:, 0, 1] = s[:, 1, 0] = p2.uf_ref / p1.uf_inc
    return s, stats


def build_case(fam, rng):
    w = rng.uniform(400, 800)
    epr = rng.uniform(2.5, 4.5)
    if fam == "gap":
        g = rng.uniform(300, 700)  # >=300 um keeps mesh/timestep sane
        return w, epr, [], 0.0, dict(w=w, epr=epr, g=g), g
    if fam == "dstub":
        s1 = rng.uniform(4000, 10000)
        s2 = rng.uniform(4000, 10000)
        sw = rng.uniform(300, 700)
        boxes = [
            ((-sw / 2, w / 2), (sw / 2, w / 2 + s1)),
            ((-sw / 2, -w / 2 - s2), (sw / 2, -w / 2)),
        ]
        return w, epr, boxes, max(s1, s2) + w, dict(w=w, epr=epr, s1=s1, s2=s2, sw=sw), 0.0
    if fam == "patch":
        pl = rng.uniform(3000, 8000)
        pw = rng.uniform(2000, 5000)
        boxes = [((-pl / 2, -pw / 2), (pl / 2, pw / 2))]
        return w, epr, boxes, pw, dict(w=w, epr=epr, pl=pl, pw=pw), 0.0
    raise ValueError(fam)


def sanity(s):
    ok_finite = bool(np.all(np.isfinite(s)))
    sv = np.linalg.svd(s, compute_uv=False)
    return dict(
        finite=ok_finite,
        max_sv=float(sv.max()),
        s11_mid=float(abs(s[len(FREQ) // 2, 0, 0])),
        s21_mid=float(abs(s[len(FREQ) // 2, 1, 0])),
    )


def run_case(fam, i, case, h5_path):
    w, epr, boxes, y_extra, params, gp = case
    sid = f"oems_{fam}_{i}"
    with h5py.File(h5_path, "a") as h:
        if sid in h:
            return None
    print(f"{sid}: " + " ".join(f"{k}={v:.0f}" if k != "epr" else f"{k}={v:.2f}"
                                for k, v in params.items()))
    s, stats = simulate(w, epr, boxes, y_extra, sid, gap=gp, verbose=True)
    chk = sanity(s)
    print(f"    wall={stats['wall_s']}s ram={stats['peak_ram_mb']}MB "
          f"|S11|={chk['s11_mid']:.3f} |S21|={chk['s21_mid']:.3f} "
          f"max_sv={chk['max_sv']:.3f} finite={chk['finite']}")
    if not chk["finite"] or chk["max_sv"] > 1.1:
        print(f"    !! {sid} failed sanity, not saved")
        return stats
    with h5py.File(h5_path, "a") as h:
        g = h.create_group(sid)
        g.attrs["topology_family"] = f"oems_{fam}"
        g.attrs["ports"] = 2
        g.create_dataset("param_names", data=np.bytes_(",".join(params)))
        g.create_dataset("params", data=np.array(list(params.values())))
        g.create_dataset("freq", data=FREQ)
        g.create_dataset("s", data=s)
    return stats


def main():
    arg = sys.argv[1] if len(sys.argv) > 1 else None
    smoke = arg == "--smoke"
    n_per = 1 if smoke else 8
    rng = np.random.default_rng(7)
    out = os.path.join(REPO, "results",
                       "openems_smoke2.h5" if smoke else "openems_families.h5")
    for fam in ("gap", "dstub", "patch"):
        cases = [build_case(fam, rng) for _ in range(8)][:n_per]
        if arg not in (None, "--smoke") and fam != arg:
            continue
        for i, case in enumerate(cases):
            run_case(fam, i, case, out)
    print("wrote", out)


if __name__ == "__main__":
    main()
