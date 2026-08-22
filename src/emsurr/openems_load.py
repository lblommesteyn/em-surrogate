"""openEMS family loading in the canonical representation (moved out of
scripts/score_openems.py so it can be imported without executing the frozen
scoring pipeline as an import side effect)."""

import h5py
import numpy as np

from . import physics, synth

SUB_H = 254e-6


def tokenize(fam, p):
    """Element tokens from existing vocabulary, where expressible."""
    w = p["w"] * 1e-6
    er = p["epr"]
    z0, eeff = physics.microstrip_z0_eeff(w, SUB_H, er)
    feed = (synth.EL_LINE, z0, eeff, 0.0, 15e-3)
    if fam == "dstub":
        toks = [feed]
        for key in ("s1", "s2"):
            sw = p["sw"] * 1e-6
            z0s, eeffs = physics.microstrip_z0_eeff(sw, SUB_H, er)
            toks.append((synth.EL_STUB_OPEN, z0s, eeffs, 0.0, p[key] * 1e-6))
        toks.append(feed)
        return np.array(toks)
    if fam == "patch":
        pw = p["pw"] * 1e-6
        z0p, eeffp = physics.microstrip_z0_eeff(pw, SUB_H, er)
        return np.array([feed, (synth.EL_LINE, z0p, eeffp, 0.0, p["pl"] * 1e-6), feed])
    if fam == "gap":
        return np.array([feed, feed])  # the gap itself is not expressible
    raise ValueError(fam)


def load_openems(path="results/openems_families.h5"):
    out = []
    with h5py.File(path, "r") as h:
        for sid in h:
            g = h[sid]
            names = g["param_names"][()].decode().split(",")
            vals = g["params"][:]
            p = dict(zip(names, vals))
            fam = g.attrs["topology_family"].removeprefix("oems_")
            params = {k: 0.0 for k in synth.PARAM_NAMES}
            # non-design constants (conductor sigma/thickness) take the
            # training values: they are not what makes these structures novel,
            # and leaving them 0 trivially separates OOD via constant columns
            params.update(w1=p["w"] * 1e-6, h=SUB_H, er=p["epr"], len1=15e-3,
                          len2=15e-3, sigma=5.8e7, t=35e-6, tand=0.001)
            if fam == "dstub":
                params["w2"] = p["sw"] * 1e-6
                params["stub_len"] = p["s1"] * 1e-6
            if fam == "patch":
                params["w2"] = p["pw"] * 1e-6
                params["len3"] = p["pl"] * 1e-6
            if fam == "gap":
                params["len3"] = p["g"] * 1e-6
            out.append(dict(
                sample_id=sid, topology_family=g.attrs["topology_family"],
                ports=2, params=np.array([params[k] for k in synth.PARAM_NAMES]),
                elements=tokenize(fam, p), freq=g["freq"][:], s=g["s"][:],
            ))
    return out
