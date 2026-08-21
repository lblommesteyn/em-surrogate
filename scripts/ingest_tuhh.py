"""Phase 1: ingest + verify the TUHH SI/PI-Database, write a reproducible
manifest of every included/excluded sample and why.

Usage:
    python scripts/ingest_tuhh.py [--root data/raw/tuhh] [--n-freq 128]

Outputs:
    results/tuhh_manifest.csv     one row per parameter.csv row, with
                                  included flag and exclusion reason
    results/tuhh_ingest.json      frozen canonicalization meta (schema,
                                  common band, grid size, port pair) plus
                                  per-family statistics
    data/processed/tuhh.h5        canonical samples (included rows only)

Exclusion reasons (checked in order):
    missing_touchstone, parse_error, non_finite, single_port,
    no_band_overlap (sample does not cover the common band),
    duplicate (same family + identical params as an earlier row; kept row
    listed in dup_of)

Passivity/reciprocity are *recorded* (max singular value, max |S-S^T|), not
exclusion criteria: measured/full-wave data may violate slightly and that is
a property to report, not to censor.
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import skrf

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from emsurr import dataset, tuhh, tuhh_prep


def scan(root: Path) -> tuple[list[dict], list[dict]]:
    """Return (rows, raw_samples): a manifest row per parameter.csv entry,
    and loaded raw samples for rows that parse."""
    fam_map = tuhh.family_map()
    rows, raw = [], []
    fam_dirs = sorted(d for d in root.iterdir() if (d / "parameter.csv").exists())
    if not fam_dirs:
        raise SystemExit(f"no families with parameter.csv under {root}")
    for fam_dir in fam_dirs:
        fam = fam_dir.name
        df = pd.read_csv(fam_dir / "parameter.csv")
        param_cols = [c for c in df.columns if c.upper() != "SIMULATION"]
        var_dir = fam_dir / "variation"
        seen: dict[str, str] = {}
        for _, r in df.iterrows():
            sim = str(r["SIMULATION"])
            sid = f"{fam}_{sim}"
            row = dict(
                family=fam,
                super_family=fam_map.get(fam, "UNMAPPED"),
                simulation=sim,
                sample_id=sid,
                touchstone="",
                ports=0,
                n_freq=0,
                f_min_hz=np.nan,
                f_max_hz=np.nan,
                max_sv=np.nan,
                reciprocity=np.nan,
                dup_of="",
                included=False,
                reason="",
            )
            matches = list(var_dir.glob(f"{sim}.s*p")) or list(var_dir.glob(f"*{sim}*.s*p"))
            if not matches:
                row["reason"] = "missing_touchstone"
                rows.append(row)
                continue
            row["touchstone"] = matches[0].name
            try:
                net = skrf.Network(str(matches[0]))
            except Exception as e:
                row["reason"] = f"parse_error:{type(e).__name__}"
                rows.append(row)
                continue
            row.update(
                ports=net.nports,
                n_freq=len(net.f),
                f_min_hz=float(net.f.min()),
                f_max_hz=float(net.f.max()),
            )
            if not np.all(np.isfinite(net.s)):
                row["reason"] = "non_finite"
                rows.append(row)
                continue
            if net.nports < 2:
                row["reason"] = "single_port"
                rows.append(row)
                continue
            sv = np.linalg.svd(net.s, compute_uv=False)
            row["max_sv"] = float(sv.max())
            row["reciprocity"] = float(np.max(np.abs(net.s - np.transpose(net.s, (0, 2, 1)))))
            params = r[param_cols].to_numpy(dtype=float)
            hsh = hashlib.sha1((fam + params.tobytes().hex()).encode()).hexdigest()
            if hsh in seen:
                row["dup_of"] = seen[hsh]
                row["reason"] = "duplicate"
                rows.append(row)
                continue
            seen[hsh] = sid
            row["included"] = True
            rows.append(row)
            raw.append(
                dict(
                    sample_id=sid,
                    topology_family=fam,
                    ports=net.nports,
                    params=params,
                    param_names=list(param_cols),
                    elements=np.zeros((0, 5)),
                    freq=net.f,
                    s=net.s,
                )
            )
    return rows, raw


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/raw/tuhh")
    ap.add_argument("--n-freq", type=int, default=128)
    ap.add_argument("--out-h5", default="data/processed/tuhh.h5")
    args = ap.parse_args()

    rows, raw = scan(Path(args.root))
    # band check: freeze the common band over parse-clean samples, then
    # exclude any sample that does not cover it (recorded per row)
    lo, hi = tuhh_prep.common_band(raw)
    keep = []
    by_id = {r["sample_id"]: r for r in rows}
    for s in raw:
        if s["freq"].min() > lo + 1e-9 or s["freq"].max() < hi - 1e-9:
            by_id[s["sample_id"]]["included"] = False
            by_id[s["sample_id"]]["reason"] = "no_band_overlap"
        else:
            keep.append(s)
    raw = keep

    canon, meta = tuhh_prep.canonicalize(raw, n_freq=args.n_freq)
    Path("results").mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv("results/tuhh_manifest.csv", index=False)
    Path(args.out_h5).parent.mkdir(parents=True, exist_ok=True)
    dataset.save_h5(canon, args.out_h5)

    df = pd.DataFrame(rows)
    summary = dict(
        meta=meta,
        n_rows=len(df),
        n_included=int(df.included.sum()),
        exclusions=df[~df.included].reason.value_counts().to_dict(),
        per_family={
            fam: dict(
                n=int(g.included.sum()),
                super_family=g.super_family.iloc[0],
                ports=sorted(g[g.included].ports.unique().tolist()),
                max_sv=float(g[g.included].max_sv.max()) if g.included.any() else None,
            )
            for fam, g in df.groupby("family")
        },
    )
    Path("results/tuhh_ingest.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k != "per_family"}, indent=2))
    print(f"wrote results/tuhh_manifest.csv ({len(df)} rows, {summary['n_included']} included)")
    print(f"wrote {args.out_h5}")


if __name__ == "__main__":
    main()
