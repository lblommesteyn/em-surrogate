"""Loader for the Multilayer Rectangle-shaped Planar Windings dataset
(Zenodo record 21762502, dataset_planar_windings_v1.1.xlsx).

Subsets (provided split, preserved verbatim):
  CORE  10,110 ANSYS Maxwell 3D FEA samples, in-distribution
  OOD    1,124 FEA samples, seven geometric edge-case classes (unlabeled)
  MEAS      55 fabricated PCB prototypes, laboratory measurements

Features: D1, D2, d1, d2, w, s, NT, NL, O  (O is empty for NL=1 -> 0.0).
Dbar1/Dbar2 are derived quantities and excluded from the input.
Target: self-inductance L in henries; models regress log10(L) because L
spans several orders of magnitude.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import openpyxl

FEATURES = ["D1", "D2", "d1", "d2", "w", "s", "NT", "NL", "O"]
XLSX = Path("data/external/planar_windings/dataset_planar_windings_v1.1.xlsx")


def load(path: Path = XLSX) -> dict[str, dict[str, np.ndarray]]:
    wb = openpyxl.load_workbook(path, read_only=True)
    out = {}
    for sheet in ["CORE", "OOD", "MEAS"]:
        rows = list(wb[sheet].iter_rows(values_only=True))
        header = [c for c in rows[0] if c is not None]
        idx = {name: header.index(name) for name in FEATURES + ["L"]}
        x, y = [], []
        for r in rows[1:]:
            if r[idx["L"]] is None:
                continue
            x.append([float(r[idx[f]] or 0.0) for f in FEATURES])
            y.append(float(r[idx["L"]]))
        out[sheet] = dict(x=np.array(x), y=np.array(y))
    return out
