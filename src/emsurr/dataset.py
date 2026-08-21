"""Canonical dataset representation, HDF5 storage, and sanity checks."""

from __future__ import annotations

import hashlib

import h5py
import numpy as np


def save_h5(samples: list[dict], path: str) -> None:
    with h5py.File(path, "w") as h:
        for s in samples:
            g = h.create_group(s["sample_id"])
            g.attrs["topology_family"] = s["topology_family"]
            g.attrs["ports"] = s["ports"]
            g.create_dataset("params", data=s["params"])
            g.create_dataset("elements", data=s["elements"])
            g.create_dataset("freq", data=s["freq"])
            g.create_dataset("s", data=s["s"], compression="gzip")


def load_h5(path: str) -> list[dict]:
    out = []
    with h5py.File(path, "r") as h:
        for sid in h:
            g = h[sid]
            out.append(
                dict(
                    sample_id=sid,
                    topology_family=g.attrs["topology_family"],
                    ports=int(g.attrs["ports"]),
                    params=g["params"][:],
                    elements=g["elements"][:],
                    freq=g["freq"][:],
                    s=g["s"][:],
                )
            )
    return out


def param_hash(s: dict) -> str:
    key = s["topology_family"] + s["params"].tobytes().hex()
    return hashlib.sha1(key.encode()).hexdigest()


def passivity_violations(s_mat: np.ndarray, tol: float = 1e-6) -> int:
    """Number of frequency points where max singular value of S exceeds 1+tol."""
    sv = np.linalg.svd(s_mat, compute_uv=False)
    return int(np.sum(sv[:, 0] > 1 + tol))


def reciprocity_error(s_mat: np.ndarray) -> float:
    return float(np.max(np.abs(s_mat - np.transpose(s_mat, (0, 2, 1)))))


def sanity_check(samples: list[dict], rtol_passivity: float = 1e-6) -> dict:
    """Validate a dataset; returns a report dict and raises on fatal issues."""
    report = dict(
        n=len(samples),
        nan_samples=[],
        grid_mismatch=[],
        passivity_bad=[],
        reciprocity_bad=[],
        duplicates=[],
    )
    if not samples:
        raise ValueError("empty dataset")
    ref_freq = samples[0]["freq"]
    seen: dict[str, str] = {}
    for s in samples:
        if not np.all(np.isfinite(s["s"])):
            report["nan_samples"].append(s["sample_id"])
        if s["freq"].shape != ref_freq.shape or not np.allclose(s["freq"], ref_freq):
            report["grid_mismatch"].append(s["sample_id"])
        nv = passivity_violations(s["s"], rtol_passivity)
        if nv:
            report["passivity_bad"].append((s["sample_id"], nv))
        if reciprocity_error(s["s"]) > 1e-6:
            report["reciprocity_bad"].append(s["sample_id"])
        hsh = param_hash(s)
        if hsh in seen:
            report["duplicates"].append((seen[hsh], s["sample_id"]))
        else:
            seen[hsh] = s["sample_id"]
    if report["nan_samples"]:
        raise ValueError(f"non-finite S-params in {report['nan_samples'][:5]}")
    return report


def check_leakage(train: list[dict], test: list[dict]) -> list[tuple[str, str]]:
    """Exact-duplicate leakage between two splits (same family + params)."""
    train_hashes = {param_hash(s): s["sample_id"] for s in train}
    return [
        (train_hashes[param_hash(s)], s["sample_id"])
        for s in test
        if param_hash(s) in train_hashes
    ]
