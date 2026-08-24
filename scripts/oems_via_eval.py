"""openEMS evaluator for the differential via-transition task (runs inside
tools/oems-venv).

Structure (units um): 4-layer board, substrate er=3.5, total 1000 um.
  z=1000  L1: top differential microstrip pair (ports 1,2)
  z= 700  L2: ground plane (antipad holes carved by priority)
  z= 300  L3: ground plane (antipad holes)
  z=   0  L4: bottom differential pair (ports 3,4)
Two signal vias at x=0, y=+/-s_via/2 connect L1->L4 through antipads in
L2/L3. n_gnd stitching vias on a ring of radius r_gnd connect L2-L3.
Traces run 2500 um each side. Differential excitation: port1 +, port2 -.

Mixed-mode extraction under pure odd drive:
  Sdd11=(b1-b2)/(a1-a2), Sdd21=(b3-b4)/(a1-a2),
  Scd11=(b1+b2)/(a1-a2), Scd21=(b3+b4)/(a1-a2)
Stored as s[:,0,0]=Sdd11, s[:,1,0]=Sdd21, s[:,0,1]=Scd11, s[:,1,1]=Scd21.

Validity: finite + odd-drive power balance sqrt(sum|S|^2) <= 1.1 in-band.
"""
import json, os, sys, time, tempfile
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
SUB_T = 1000.0
Z_L2, Z_L3 = 700.0, 300.0
LTR = 2500.0
N_TS = 60000


def simulate_via(p, tag, verbose=False):
    d_via, d_pad, d_anti = p["d_via"], p["d_pad"], p["d_anti"]
    s_via, w = p["s_via"], p["w_trace"]
    n_gnd, r_gnd = int(round(p["n_gnd"])), p["r_gnd"]
    y1, y2 = s_via / 2.0, -s_via / 2.0

    sim_path = os.path.join(tempfile.gettempdir(), f"oems_via_{tag}")
    fdtd = openEMS(NrTS=N_TS, EndCriteria=1e-4)
    fdtd.SetGaussExcite(F_MAX / 2, F_MAX / 2)
    fdtd.SetBoundaryCond(["PML_8", "PML_8", "PML_8", "PML_8", "MUR", "MUR"])
    csx = ContinuousStructure()
    fdtd.SetCSX(csx)
    mesh = csx.GetGrid()
    mesh.SetDeltaUnit(UNIT)

    xb, yb = LTR + 1200.0, max(r_gnd + 1500.0, s_via / 2 + d_anti / 2 + 1500.0)
    res = C0 / (F_MAX * np.sqrt(EPR)) / UNIT / 25          # ~320 um bulk
    fine = min(d_via / 3.0, 90.0)

    # realistic dielectric loss (tand ~0.02 at 10 GHz) damps the plane cavity
    KAPPA = 2 * np.pi * 10e9 * 8.854e-12 * EPR * 0.02
    sub = csx.AddMaterial("substrate", epsilon=EPR, kappa=KAPPA)
    sub.AddBox([-xb, -yb, 0], [xb, yb, SUB_T], priority=1)

    pec = csx.AddMetal("PEC")
    # ground planes with antipad carve (higher-priority substrate cylinders)
    for zp in (Z_L2, Z_L3):
        pec.AddBox([-xb, -yb, zp], [xb, yb, zp], priority=10)
    carve = csx.AddMaterial("carve", epsilon=EPR, kappa=KAPPA)
    for yv in (y1, y2):
        for zp in (Z_L2, Z_L3):
            carve.AddCylinder([0, yv, zp - 1], [0, yv, zp + 1],
                              d_anti / 2.0, priority=20)
    # signal vias + pads
    for yv in (y1, y2):
        pec.AddCylinder([0, yv, 0], [0, yv, SUB_T], d_via / 2.0, priority=30)
        pec.AddCylinder([0, yv, SUB_T - 2], [0, yv, SUB_T], d_pad / 2.0, priority=30)
        pec.AddCylinder([0, yv, 0], [0, yv, 2], d_pad / 2.0, priority=30)
    # stitching vias L2-L3
    for k in range(n_gnd):
        ang = 2 * np.pi * k / max(n_gnd, 1) + np.pi / n_gnd
        gx, gy = r_gnd * np.cos(ang), r_gnd * np.sin(ang)
        pec.AddCylinder([gx, gy, Z_L3], [gx, gy, Z_L2], d_via / 2.0, priority=30)

    # mesh seeds
    third = np.array([2 * fine / 3, -fine / 3])
    mesh.AddLine("x", [-xb, 0, xb])
    mesh.AddLine("x", [-d_pad / 2, d_pad / 2, -d_anti / 2, d_anti / 2,
                       -d_via / 2, d_via / 2, -r_gnd, r_gnd])
    for yv in (y1, y2):
        mesh.AddLine("y", [yv - d_pad / 2, yv, yv + d_pad / 2,
                           yv - d_anti / 2, yv + d_anti / 2,
                           yv - w / 2, yv + w / 2])
        mesh.AddLine("y", yv - w / 2 - third)
        mesh.AddLine("y", yv + w / 2 + third)
    mesh.AddLine("y", [-yb, 0, yb])
    mesh.SmoothMeshLines("x", res / 2)
    mesh.SmoothMeshLines("y", res / 2)
    mesh.AddLine("z", [0, Z_L3, Z_L2, SUB_T])
    mesh.AddLine("z", np.linspace(0, Z_L3, 4))
    mesh.AddLine("z", np.linspace(Z_L2, SUB_T, 4))
    mesh.AddLine("z", np.linspace(Z_L3, Z_L2, 4))
    mesh.AddLine("z", [-900, SUB_T + 900])
    mesh.SmoothMeshLines("z", res)

    # ports: top pair (excite +/-), bottom pair passive.
    # FEED must clear the PML (~8 x-cells) and sit BEFORE the measurement
    # plane; the first prototype had feed beyond measure -> garbage waves.
    FEED, MEAS = 1500.0, 2400.0
    ports = []
    p1 = fdtd.AddMSLPort(1, pec, [-xb, y1 - w / 2, SUB_T], [0, y1 + w / 2, Z_L2],
                         "x", "z", excite=-1, FeedShift=FEED, MeasPlaneShift=MEAS, priority=5)
    p2 = fdtd.AddMSLPort(2, pec, [-xb, y2 - w / 2, SUB_T], [0, y2 + w / 2, Z_L2],
                         "x", "z", excite=1, FeedShift=FEED, MeasPlaneShift=MEAS, priority=5)
    p3 = fdtd.AddMSLPort(3, pec, [xb, y1 - w / 2, 0], [0, y1 + w / 2, Z_L3],
                         "x", "z", MeasPlaneShift=MEAS, priority=5)
    p4 = fdtd.AddMSLPort(4, pec, [xb, y2 - w / 2, 0], [0, y2 + w / 2, Z_L3],
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
    s = np.empty((len(FREQ), 2, 2), complex)
    s[:, 0, 0] = (b[0] - b[1]) / ad          # Sdd11
    s[:, 1, 0] = (b[2] - b[3]) / ad          # Sdd21
    s[:, 0, 1] = (b[0] + b[1]) / ad          # Scd11
    s[:, 1, 1] = (b[2] + b[3]) / ad          # Scd21
    return s, dict(wall_s=round(wall, 1), cells=nx * ny * nz)


if __name__ == "__main__":
    inp, outp = sys.argv[1], sys.argv[2]
    designs = json.loads(open(inp).read())
    out = {}
    for i, d in enumerate(designs):
        t0 = time.perf_counter()
        try:
            s, stats = simulate_via(d, f"{os.getpid()}_{i}", verbose=True)
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
