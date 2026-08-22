"""Loader for SQChip-EM (github.com/Secbrain/SQChip-EM) poor_2q example set:
892 two-qubit layouts with both GDSII geometry and HFSS/Q3D-derived EM
targets, spanning ~30 generation families (filename prefix).

Representations compared downstream:
  1. parameter vector: numeric layout parameters from the JSON annotation
  2. geometry-derived features: cheap statistics computed from the GDS file
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

ROOT = Path("data/external/sqchip/SQChip-EM/examples/poor_2q")

_UNIT = {"um": 1e-6, "mm": 1e-3, "nm": 1e-9, "m": 1.0}


def _win_long(p: Path) -> str:
    """Extended-length path prefix; some filenames exceed the 260-char limit."""
    s = str(p.resolve())
    return "\\\\?\\" + s if len(s) > 240 and not s.startswith("\\\\?\\") else s

TARGETS = ["f01_q1", "f01_q2", "chi_q1", "chi_q2", "fr_1", "fr_2"]


def _num(v):
    if isinstance(v, bool):
        return float(v)
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, str):
        m = re.fullmatch(r"\s*(-?[\d.eE+-]+)\s*([a-zA-Z]*)\s*", v)
        if m:
            try:
                return float(m.group(1)) * _UNIT.get(m.group(2), 1.0)
            except ValueError:
                return None
    return None


def _flatten_layout(lay: dict) -> dict[str, float]:
    out = {}
    for src in [lay.get("shared", {}), {k: v for k, v in lay.items() if k != "shared"}]:
        for k, v in src.items():
            n = _num(v)
            if n is not None:
                out[k] = n
    return out


def family(name: str) -> str:
    return name.split("_")[0]


def load_records(root: Path = ROOT) -> list[dict]:
    gds_stems = {p.stem: p for p in (root / "gds").glob("*.gds")}
    recs = []
    for jp in sorted((root / "json").glob("*.json")):
        if jp.stem not in gds_stems:
            continue
        d = json.loads(open(_win_long(jp), encoding="utf-8", errors="replace").read())
        if d.get("status") != "completed":
            continue
        try:
            q1, q2 = d["qubits"]["Q1"], d["qubits"]["Q2"]
            r1, r2 = d["resonators"]["readout1"], d["resonators"]["readout2"]
            y = [q1["f01_epr_GHz"], q2["f01_epr_GHz"], q1["chi_MHz"], q2["chi_MHz"],
                 r1["f_GHz"], r2["f_GHz"]]
        except (KeyError, TypeError):
            continue
        if any(v is None or not np.isfinite(v) for v in y):
            continue
        recs.append(dict(
            name=jp.stem,
            family=family(jp.stem),
            params=_flatten_layout(d["meta"].get("layout", {})),
            gds=gds_stems[jp.stem],
            y=np.array(y, float),
        ))
    return recs


def param_matrix(recs: list[dict]) -> tuple[np.ndarray, list[str]]:
    """Union of numeric layout keys present in >=80% of records; missing -> 0
    plus a per-key missing indicator."""
    from collections import Counter

    cnt = Counter(k for r in recs for k in r["params"])
    keys = sorted(k for k, c in cnt.items() if c >= 0.8 * len(recs))
    x = np.zeros((len(recs), 2 * len(keys)))
    for i, r in enumerate(recs):
        for j, k in enumerate(keys):
            if k in r["params"]:
                x[i, j] = r["params"][k]
            else:
                x[i, len(keys) + j] = 1.0
    return x, keys


def geometry_features(gds_path: Path) -> np.ndarray:
    """Cheap layout statistics: polygon count/area/perimeter overall and for
    the 4 most common layers, bbox, vertex stats, area histogram."""
    import gdstk

    lib = gdstk.read_gds(_win_long(gds_path))
    top = max(lib.cells, key=lambda c: len(c.polygons)) if lib.cells else None
    polys = top.get_polygons() if top is not None else []
    if not polys:
        return np.zeros(24)
    areas = np.array([abs(p.area()) for p in polys])
    verts = np.array([len(p.points) for p in polys])
    layers = np.array([p.layer for p in polys])
    pts = np.vstack([p.points for p in polys])
    bbox = pts.min(0), pts.max(0)
    feat = [
        len(polys), areas.sum(), np.log10(areas + 1e-12).mean(), areas.max(),
        verts.mean(), verts.max(),
        bbox[1][0] - bbox[0][0], bbox[1][1] - bbox[0][1],
        pts[:, 0].std(), pts[:, 1].std(),
    ]
    from collections import Counter

    common = [l for l, _ in Counter(layers).most_common(4)]
    for li in range(4):
        if li < len(common):
            m = layers == common[li]
            feat += [m.sum(), areas[m].sum(), verts[m].mean()]
        else:
            feat += [0.0, 0.0, 0.0]
    hist, _ = np.histogram(np.log10(areas + 1e-12), bins=2)
    feat += list(hist)
    return np.array(feat, float)
