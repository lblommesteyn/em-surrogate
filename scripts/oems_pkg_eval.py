"""openEMS evaluator for the package/interposer-style channel (runs inside
tools/oems-venv).

Stackup (um), er 3.5 throughout, lossy (tand ~0.02):
  z=1400  L1 top differential microstrip (ports 1,2, left side)
  z=1100  P1 ground plane
  z= 700  M  intermediate routing layer (offset stripline)
  z= 400  P2 ground plane
  z=   0  L6 bottom differential microstrip (ports 3,4, right side)

Channel: top pair -> via transition 1 at x=0 (pitch s1, backdrilled from
the bottom leaving stub1 below M) -> mid routing on M with a rectilinear
jog to lateral offset dy2 -> via transition 2 at x=Lmid (pitch s2, stub2
above M) -> bottom pair out. Stitching rings around both transitions.
dy2 != 0 and s1 != s2 make the two lines of the pair genuinely unequal ->
physical mode conversion. Mixed-mode extraction under odd drive as in the
via task. Validity: finite + odd-drive power <= 1.1.
"""
import json, os, shutil, sys, time, tempfile
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.add_dll_directory(os.path.join(REPO, "tools", "openEMS"))

from CSXCAD import ContinuousStructure
from openEMS import openEMS
from openEMS.physical_constants import C0

F_MAX = 20e9
FREQ = np.linspace(1e9, 20e9, 96)
UNIT = 1e-6
EPR = 3.5
Z_TOP, Z_P1, Z_M, Z_P2 = 1400.0, 1100.0, 700.0, 400.0
LTR = 2200.0
N_TS = 80000


def simulate_pkg(p, tag, verbose=False):
    dv, dp_, da = p["d_via"], p["d_pad"], p["d_anti"]
    s1, s2, dy2, lmid = p["s1"], p["s2"], p["dy2"], p["l_mid"]
    wt, wb = p["w_top"], p["w_bot"]
    wm = wt
    ng, rg = int(round(p["n_gnd"])), p["r_gnd"]
    stub1, stub2 = p["stub1"], p["stub2"]
    bd1 = Z_M - stub1              # via1 barrel z bd1..Z_TOP
    td2 = Z_M + stub2              # via2 barrel z 0..td2

    y1a, y1b = s1 / 2, -s1 / 2
    y2a, y2b = dy2 + s2 / 2, dy2 - s2 / 2

    sim_path = os.path.join(tempfile.gettempdir(), f"oems_pkg_{tag}")
    fdtd = openEMS(NrTS=N_TS, EndCriteria=1e-4)
    fdtd.SetGaussExcite(F_MAX / 2, F_MAX / 2)
    fdtd.SetBoundaryCond(["PML_8", "PML_8", "PML_8", "PML_8", "MUR", "MUR"])
    csx = ContinuousStructure()
    fdtd.SetCSX(csx)
    mesh = csx.GetGrid()
    mesh.SetDeltaUnit(UNIT)

    x_lo, x_hi = -LTR - 1200.0, lmid + LTR + 1200.0
    yb = max(rg + abs(dy2) + 1500.0, max(s1, s2) / 2 + da / 2 + abs(dy2) + 1500.0)
    res = C0 / (F_MAX * np.sqrt(EPR)) / UNIT / 25

    KAPPA = 2 * np.pi * 10e9 * 8.854e-12 * EPR * 0.02
    sub = csx.AddMaterial("substrate", epsilon=EPR, kappa=KAPPA)
    sub.AddBox([x_lo, -yb, 0], [x_hi, yb, Z_TOP], priority=1)
    carve = csx.AddMaterial("carve", epsilon=EPR, kappa=KAPPA)
    pec = csx.AddMetal("PEC")

    for zp in (Z_P1, Z_P2):
        pec.AddBox([x_lo, -yb, zp], [x_hi, yb, zp], priority=10)
    # antipads at both plane crossings of both transitions
    for (xc, ys) in ((0.0, (y1a, y1b)), (lmid, (y2a, y2b))):
        for yv in ys:
            for zp in (Z_P1, Z_P2):
                carve.AddCylinder([xc, yv, zp - 60], [xc, yv, zp + 60],
                                  da / 2.0, priority=20)
    # via 1: barrels + pads (top, M)
    for yv in (y1a, y1b):
        pec.AddCylinder([0, yv, bd1], [0, yv, Z_TOP], dv / 2.0, priority=30)
        pec.AddCylinder([0, yv, Z_TOP - 2], [0, yv, Z_TOP], dp_ / 2.0, priority=30)
        pec.AddCylinder([0, yv, Z_M - 1], [0, yv, Z_M + 1], dp_ / 2.0, priority=30)
    # via 2
    for yv in (y2a, y2b):
        pec.AddCylinder([lmid, yv, 0], [lmid, yv, td2], dv / 2.0, priority=30)
        pec.AddCylinder([lmid, yv, 0], [lmid, yv, 2], dp_ / 2.0, priority=30)
        pec.AddCylinder([lmid, yv, Z_M - 1], [lmid, yv, Z_M + 1], dp_ / 2.0,
                        priority=30)
    # stitching rings: plane-to-plane (P2..P1), skipping any post whose
    # footprint (+100 um clearance) would clash with a trace corridor -
    # the deterministic placement rule a real layout follows
    cl = dv / 2.0 + 100.0
    corridors = [  # (x0, x1, ylo, yhi) rectangles metal may occupy
        (-1e9, dp_ / 2, y1a - wt / 2, y1a + wt / 2),
        (-1e9, dp_ / 2, y1b - wt / 2, y1b + wt / 2),
        (0, lmid, min(y1a, y2a) - wm / 2, max(y1a, y2a) + wm / 2),
        (0, lmid, min(y1b, y2b) - wm / 2, max(y1b, y2b) + wm / 2),
        (lmid - dp_ / 2, 1e9, y2a - wb / 2, y2a + wb / 2),
        (lmid - dp_ / 2, 1e9, y2b - wb / 2, y2b + wb / 2),
    ]

    def clashes(gx, gy):
        for (a, bx, lo, hi) in corridors:
            if a - cl <= gx <= bx + cl and lo - cl <= gy <= hi + cl:
                return True
        return False

    for (xc, yc) in ((0.0, 0.0), (lmid, dy2)):
        for k in range(ng):
            ang = 2 * np.pi * k / max(ng, 1) + np.pi / ng
            gx, gy = xc + rg * np.cos(ang), yc + rg * np.sin(ang)
            if clashes(gx, gy):
                continue
            pec.AddCylinder([gx, gy, Z_P2], [gx, gy, Z_P1], dv / 2.0, priority=30)

    # mid routing on M: two rectilinear jogged traces
    xj = lmid / 2.0
    ZT = 40.0   # interior sheets: give the mid traces a resolved thickness
    for (ya, yc2) in ((y1a, y2a), (y1b, y2b)):
        pec.AddBox([0, ya - wm / 2, Z_M - ZT], [xj + wm / 2, ya + wm / 2, Z_M + ZT],
                   priority=12)
        lo, hi = min(ya, yc2) - wm / 2, max(ya, yc2) + wm / 2
        pec.AddBox([xj - wm / 2, lo, Z_M - ZT], [xj + wm / 2, hi, Z_M + ZT], priority=12)
        pec.AddBox([xj - wm / 2, yc2 - wm / 2, Z_M - ZT], [lmid, yc2 + wm / 2, Z_M + ZT],
                   priority=12)

    # mesh seeds
    fine = min(dv / 3.0, 90.0)
    mesh.AddLine("x", [x_lo, 0, xj, lmid, x_hi])
    for xc in (0.0, lmid):
        mesh.AddLine("x", [xc - dp_ / 2, xc + dp_ / 2, xc - da / 2, xc + da / 2,
                           xc - dv / 2, xc + dv / 2])
    for yv in (y1a, y1b, y2a, y2b):
        mesh.AddLine("y", [yv - dp_ / 2, yv, yv + dp_ / 2,
                           yv - da / 2, yv + da / 2,
                           yv - dv / 2, yv + dv / 2])
    for (yv, w) in ((y1a, wt), (y1b, wt), (y2a, wb), (y2b, wb)):
        mesh.AddLine("y", [yv - w / 2, yv + w / 2])
    mesh.AddLine("y", [-yb, 0, dy2, yb])
    mesh.SmoothMeshLines("x", res / 2)
    mesh.SmoothMeshLines("y", res / 2)
    mesh.AddLine("z", [0, Z_P2, Z_M, Z_M - 40, Z_M + 40, Z_P1, Z_TOP, bd1, td2])
    for a, b in ((0, Z_P2), (Z_P2, Z_M), (Z_M, Z_P1), (Z_P1, Z_TOP)):
        mesh.AddLine("z", np.linspace(a, b, 3))
    mesh.AddLine("z", [-900, Z_TOP + 900])
    mesh.SmoothMeshLines("z", res)

    FEED, MEAS = 1500.0, 2300.0
    p1 = fdtd.AddMSLPort(1, pec, [x_lo, y1a - wt / 2, Z_TOP], [0, y1a + wt / 2, Z_P1],
                         "x", "z", excite=-1, FeedShift=FEED, MeasPlaneShift=MEAS,
                         priority=5)
    p2 = fdtd.AddMSLPort(2, pec, [x_lo, y1b - wt / 2, Z_TOP], [0, y1b + wt / 2, Z_P1],
                         "x", "z", excite=1, FeedShift=FEED, MeasPlaneShift=MEAS,
                         priority=5)
    p3 = fdtd.AddMSLPort(3, pec, [x_hi, y2a - wb / 2, 0], [lmid, y2a + wb / 2, Z_P2],
                         "x", "z", MeasPlaneShift=MEAS, priority=5)
    p4 = fdtd.AddMSLPort(4, pec, [x_hi, y2b - wb / 2, 0], [lmid, y2b + wb / 2, Z_P2],
                         "x", "z", MeasPlaneShift=MEAS, priority=5)
    ports = [p1, p2, p3, p4]

    nx, ny, nz = (mesh.GetQtyLines(d) for d in "xyz")
    if verbose:
        print(f"    mesh {nx}x{ny}x{nz} = {nx*ny*nz} cells", flush=True)
    t0 = time.perf_counter()
    fdtd.Run(sim_path, cleanup=True)
    wall = time.perf_counter() - t0

    for pt in ports:
        pt.CalcPort(sim_path, FREQ, ref_impedance=50)
    a1, a2 = p1.uf_inc, p2.uf_inc
    b = [pt.uf_ref for pt in ports]
    ad = a1 - a2
    shutil.rmtree(sim_path, ignore_errors=True)   # keep temp clean (disk!)
    s = np.empty((len(FREQ), 2, 2), complex)
    s[:, 0, 0] = (b[0] - b[1]) / ad
    s[:, 1, 0] = (b[2] - b[3]) / ad
    s[:, 0, 1] = (b[0] + b[1]) / ad
    s[:, 1, 1] = (b[2] + b[3]) / ad
    return s, dict(wall_s=round(wall, 1), cells=nx * ny * nz)


if __name__ == "__main__":
    inp, outp = sys.argv[1], sys.argv[2]
    designs = json.loads(open(inp).read())
    out = {}
    for i, d in enumerate(designs):
        t0 = time.perf_counter()
        try:
            s, stats = simulate_pkg(d, f"{os.getpid()}_{i}", verbose=True)
        except Exception as e:
            print(f"design {i}: SOLVER FAILURE {type(e).__name__}: {e}", flush=True)
            s = np.full((len(FREQ), 2, 2), np.nan, complex)
            stats = {"wall_s": time.perf_counter() - t0}
        pw = np.sqrt((np.abs(s) ** 2).sum(axis=(1, 2)))
        finite = bool(np.all(np.isfinite(s)))
        out[str(i)] = s
        out[f"meta_{i}"] = np.array([stats["wall_s"], float(finite),
                                     float(pw.max() if finite else 9.9)])
        print(f"design {i}: wall={stats['wall_s']}s power_max="
              f"{pw.max() if finite else float('nan'):.3f}", flush=True)
    np.savez(outp, freq=FREQ, **out)
