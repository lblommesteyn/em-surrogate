# -*- coding: utf-8 -*-
"""openEMS smoke test: geometry -> FDTD -> S-parameters -> canonical sample.

Runs 6 tiny microstrip structures (3 plain lines, 3 notch filters) with
varied width / permittivity / stub length, and writes them as canonical
sample dicts to results/openems_smoke.h5. Coarse mesh + relaxed end
criteria keep each run to roughly a minute; this validates the pipeline,
not solver accuracy.

Run with the openEMS venv (Python 3.11, wheels from the Windows build):

    tools/oems-venv/Scripts/python scripts/openems_smoke.py
"""

import os
import sys
import tempfile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.add_dll_directory(os.path.join(REPO, "tools", "openEMS"))

import numpy as np
import h5py

from CSXCAD import ContinuousStructure
from openEMS import openEMS
from openEMS.physical_constants import C0

F_MAX = 7e9
FREQ = np.linspace(0.1e9, F_MAX, 128)
UNIT = 1e-6  # um


def simulate(msl_width, epr, stub_length, tag):
    sim_path = os.path.join(tempfile.gettempdir(), f"oems_smoke_{tag}")
    msl_length = 30000.0
    sub_t = 254.0

    fdtd = openEMS(EndCriteria=1e-3)
    fdtd.SetGaussExcite(F_MAX / 2, F_MAX / 2)
    fdtd.SetBoundaryCond(["PML_8", "PML_8", "MUR", "MUR", "PEC", "MUR"])
    csx = ContinuousStructure()
    fdtd.SetCSX(csx)
    mesh = csx.GetGrid()
    mesh.SetDeltaUnit(UNIT)

    res = C0 / (F_MAX * np.sqrt(epr)) / UNIT / 30  # coarse: lambda/30
    third = np.array([2 * res / 3, -res / 3]) / 4
    mesh.AddLine("x", 0)
    mesh.AddLine("x", msl_width / 2 + third)
    mesh.AddLine("x", -msl_width / 2 - third)
    mesh.SmoothMeshLines("x", res / 4)
    mesh.AddLine("x", [-msl_length, msl_length])
    mesh.SmoothMeshLines("x", res)
    mesh.AddLine("y", 0)
    mesh.AddLine("y", msl_width / 2 + third)
    mesh.AddLine("y", -msl_width / 2 - third)
    mesh.SmoothMeshLines("y", res / 4)
    y_top = 15 * msl_width + stub_length
    mesh.AddLine("y", [-15 * msl_width, y_top])
    if stub_length > 0:
        mesh.AddLine("y", (msl_width / 2 + stub_length) + third)
    mesh.SmoothMeshLines("y", res)
    mesh.AddLine("z", np.linspace(0, sub_t, 5))
    mesh.AddLine("z", 3000)
    mesh.SmoothMeshLines("z", res)

    sub = csx.AddMaterial("substrate", epsilon=epr)
    sub.AddBox([-msl_length, -15 * msl_width, 0], [msl_length, y_top, sub_t])

    pec = csx.AddMetal("PEC")
    p1 = fdtd.AddMSLPort(
        1, pec, [-msl_length, -msl_width / 2, sub_t], [0, msl_width / 2, 0],
        "x", "z", excite=-1, FeedShift=10 * res, MeasPlaneShift=msl_length / 3,
        priority=10,
    )
    p2 = fdtd.AddMSLPort(
        2, pec, [msl_length, -msl_width / 2, sub_t], [0, msl_width / 2, 0],
        "x", "z", MeasPlaneShift=msl_length / 3, priority=10,
    )
    if stub_length > 0:
        pec.AddBox(
            [-msl_width / 2, msl_width / 2, sub_t],
            [msl_width / 2, msl_width / 2 + stub_length, sub_t],
            priority=10,
        )

    fdtd.Run(sim_path, cleanup=True)
    for p in (p1, p2):
        p.CalcPort(sim_path, FREQ, ref_impedance=50)
    s11 = p1.uf_ref / p1.uf_inc
    s21 = p2.uf_ref / p1.uf_inc
    # symmetric 2-port assumption for the smoke test (structure is reciprocal)
    s = np.empty((len(FREQ), 2, 2), dtype=complex)
    s[:, 0, 0] = s11
    s[:, 1, 1] = s11
    s[:, 0, 1] = s21
    s[:, 1, 0] = s21
    return s


def main():
    cases = [
        ("plain", 600, 3.0, 0),
        ("plain", 400, 3.66, 0),
        ("plain", 800, 4.4, 0),
        ("notch", 600, 3.66, 8000),
        ("notch", 600, 3.66, 12000),
        ("notch", 400, 4.4, 10000),
    ]
    out = os.path.join(REPO, "results", "openems_smoke.h5")
    with h5py.File(out, "w") as h:
        for i, (fam, w, epr, stub) in enumerate(cases):
            tag = f"{fam}_{i}"
            print(f"[{i+1}/{len(cases)}] {tag}: w={w}um er={epr} stub={stub}um")
            s = simulate(w, epr, stub, tag)
            assert np.all(np.isfinite(s)), tag
            g = h.create_group(f"openems_{tag}")
            g.attrs["topology_family"] = f"openems_{fam}"
            g.attrs["ports"] = 2
            g.create_dataset("params", data=np.array([w * UNIT, epr, stub * UNIT]))
            g.create_dataset("freq", data=FREQ)
            g.create_dataset("s", data=s)
            print(f"    |S21| @ mid-band: {abs(s[len(FREQ)//2,1,0]):.3f}")
    print("wrote", out)


if __name__ == "__main__":
    sys.exit(main())
