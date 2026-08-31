"""Slice-level replay of frozen Jalapeno GEMM shapes: Level-2 slice cycles
(results/jalapeno_slice_level2_raw.txt) x P&R-achieved clocks
(results/jphys_slice_achieved_clocks.json). Same composition rule as the
domain-level replay: rounds = ceil(N / (N_ENG*D)); kc = ceil(K/D); m
blocked by ACC with full-block scaling for exact-ACC remainders.
"""
import json, math, re
from pathlib import Path

SHAPES = {
    "attn_qkv_b1": (1, 2880, 80),
    "moe_exp_up_b1": (1, 2880, 90),
    "moe_exp_up_b8": (8, 2880, 90),
    "dense_gateup_b1": (1, 8192, 896),
    "prefill": (128, 2880, 80),
}
DOMS = {"32": dict(D=32, NE=8, ACC=4), "64": dict(D=64, NE=2, ACC=8)}

def load_level2(path):
    rows = {}
    for line in Path(path).read_text().splitlines():
        mm = re.search(r"RESULT D=(\d+) NE=(\d+) ACC=(\d+) BW=(\d+) m=(\d+) kc=(\d+) cycles=(\d+)", line)
        if mm:
            d, ne, acc, bw, m, kc, cyc = map(int, mm.groups())
            rows[(d, bw, m)] = (kc, cyc)
    return rows

def replay(level2, clocks_ns, bws=(64, 128)):
    out = {}
    for bw in bws:
        for dk, dom in DOMS.items():
            D, NE, ACC = dom["D"], dom["NE"], dom["ACC"]
            clk = clocks_ns[(dk, bw)]
            for name, (m, K, N) in SHAPES.items():
                rounds = math.ceil(N / (NE * D))
                kc = math.ceil(K / D)
                m_blocks = math.ceil(m / ACC)
                key_m = 1 if m == 1 else ACC
                kc_ref, cyc_ref = level2[(D, bw, key_m)]
                cyc_round = cyc_ref * kc / kc_ref
                if m_blocks > 1:
                    m_last = m - (m_blocks - 1) * ACC
                    kf, cf = level2[(D, bw, ACC)]
                    full = cf * kc / kf
                    if m_last == ACC:
                        rem = full
                    else:
                        kr, cr = level2[(D, bw, 1)]
                        rem = (cr * kc / kr) * m_last
                    cyc_round = (m_blocks - 1) * full + rem
                total = rounds * cyc_round
                out[(name, dk, bw)] = dict(cycles=total, ns=total * clk,
                                           rounds=rounds, clk_ns=clk)
    return out

if __name__ == "__main__":
    level2 = load_level2("results/jalapeno_slice_level2_raw.txt")
    clocks = json.loads(Path("results/jphys_slice_achieved_clocks.json").read_text())
    clocks = {(k.split("@")[0], int(k.split("@")[1])): v for k, v in clocks.items()}
    res = replay(level2, clocks)
    for bw in (64, 128):
        print(f"\n== BW={bw} B/cyc ==")
        print(f"{'shape':18s} {'fine us':>10s} {'coarse us':>10s} {'coarse/fine':>11s}")
        for name in SHAPES:
            a = res[(name, '32', bw)]["ns"] / 1e3
            b = res[(name, '64', bw)]["ns"] / 1e3
            print(f"{name:18s} {a:10.2f} {b:10.2f} {b/a:11.2f}")
    Path("results/jphys_slice_replay.json").write_text(
        json.dumps({f"{k[0]}|{k[1]}|{k[2]}": v for k, v in res.items()}, indent=1))
