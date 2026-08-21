"""Mechanical canonicalization of TUHH samples into the frozen pipeline format.

The milestone-1/2 machinery assumes: fixed-length param vector, common
frequency grid, 2x2 complex S block. Real TUHH datasets differ per family in
parameter schema, frequency grid, and port count. Three MECHANICAL (input
plumbing, not research) adaptations are applied here and documented in
results/tuhh_report.md:

1. Union parameter schema: each family keeps its named columns; a global
   union vector is built with family-prefixed names, unused entries 0.0.
   Columns constant in a given training set are neutralized by the existing
   Normalizer sd-floor, so absent parameters cannot fabricate OOD signal by
   scale explosion. (An OOD family's *present-but-untrained* columns still
   legitimately differ, which is exactly the structural signal under test.)
2. Common frequency grid: complex S interpolated (linear, real/imag) onto a
   fixed grid over a band every included sample covers. The band and point
   count are frozen at ingest time in the manifest, before any model run.
3. Port sub-block: the (port_a, port_b) 2x2 block is extracted from N-port
   networks. Default (0, 1): the documented signal path / primary IC ports.
   Single-port samples are excluded at ingest.

TUHH has no element-token decomposition (real layouts, no analytic cascade),
so `elements` stays empty: DeepSets and the token part of content features
degrade to their param-only behaviour. Documented, not compensated.
"""

from __future__ import annotations

import numpy as np


def union_schema(samples: list[dict]) -> list[str]:
    """Family-prefixed union of parameter names, deterministic order."""
    names: list[str] = []
    seen = set()
    for s in sorted(samples, key=lambda s: (s["topology_family"], s["sample_id"])):
        for n in s["param_names"]:
            key = f"{s['topology_family']}:{n}"
            if key not in seen:
                seen.add(key)
                names.append(key)
    return sorted(names)


def common_band(samples: list[dict]) -> tuple[float, float]:
    """Widest band covered by every sample."""
    lo = max(float(s["freq"].min()) for s in samples)
    hi = min(float(s["freq"].max()) for s in samples)
    if not lo < hi:
        raise ValueError(f"no common frequency band (lo={lo}, hi={hi})")
    return lo, hi


def canonicalize(
    samples: list[dict],
    schema: list[str] | None = None,
    band: tuple[float, float] | None = None,
    n_freq: int = 128,
    port_pair: tuple[int, int] = (0, 1),
) -> tuple[list[dict], dict]:
    """Map raw TUHH samples into the canonical dict layout.

    Returns (canonical_samples, meta) where meta records the frozen schema,
    band, grid size, and port pair for the manifest.
    """
    schema = schema if schema is not None else union_schema(samples)
    band = band if band is not None else common_band(samples)
    grid = np.linspace(band[0], band[1], n_freq)
    idx = {n: i for i, n in enumerate(schema)}
    a, b = port_pair
    out = []
    for s in samples:
        p = np.zeros(len(schema))
        for name, val in zip(s["param_names"], s["params"]):
            key = f"{s['topology_family']}:{name}"
            if key in idx:
                p[idx[key]] = float(val)
        sub = s["s"][:, [a, b]][:, :, [a, b]]
        s2 = np.empty((n_freq, 2, 2), dtype=complex)
        for i in range(2):
            for j in range(2):
                s2[:, i, j] = np.interp(grid, s["freq"], sub[:, i, j].real) + 1j * np.interp(
                    grid, s["freq"], sub[:, i, j].imag
                )
        out.append(
            dict(
                sample_id=s["sample_id"],
                topology_family=s["topology_family"],
                ports=2,
                params=p,
                elements=np.zeros((0, 5)),
                freq=grid,
                s=s2,
            )
        )
    meta = dict(
        schema=schema,
        band_hz=[float(band[0]), float(band[1])],
        n_freq=n_freq,
        port_pair=list(port_pair),
    )
    return out, meta
