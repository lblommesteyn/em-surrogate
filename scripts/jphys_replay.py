"""Replay frozen Jalapeno GEMM shapes on the two matched domains using
Level-2 measured round cycles and P&R-achieved clocks.

Composition (identical rule for both domains, fixed before results):
  shape (m, K, N): rounds = ceil(N / cols_per_round), cols_per_round =
  N_UNITS*D. Each round runs ceil(K/D) k-chunks; m rows per m_block capped
  by ACC. Round latency comes from the Level-2 RTL sweep (cycles for one
  round at that BW, m, kc), scaled by kc(shape)/kc(sweep) for K != 512.
  Wall time = rounds * round_cycles * achieved_clock_ns.
"""
import json, math, re, sys
from pathlib import Path

SHAPES = {  # (m, K, N) frozen from jalapeno-sim
    "attn_qkv_b1": (1, 2880, 80),
    "moe_exp_up_b1": (1, 2880, 90),
    "moe_exp_up_b8": (8, 2880, 90),
    "dense_gateup_b1": (1, 8192, 896),
    "prefill": (128, 2880, 80),
}
DOMS = {"32": dict(D=32, NU=4, ACC=4), "64": dict(D=64, NU=1, ACC=8)}

def load_level2(path):
    rows = {}
    for line in Path(path).read_text().splitlines():
        m = re.search(r"RESULT D=(\d+) N=(\d+) ACC=(\d+) BW=(\d+) m=(\d+) kc=(\d+) cycles=(\d+)", line)
        if m:
            d, n, acc, bw, mm, kc, cyc = map(int, m.groups())
            rows[(d, bw, mm)] = (kc, cyc)
    return rows

def replay(level2, clocks_ns, bws=(64, 128)):
    out = {}
    for bw in bws:
        for dk, dom in DOMS.items():
            D, NU, ACC = dom["D"], dom["NU"], dom["ACC"]
            clk = clocks_ns[(dk, bw)]
            for name, (m, K, N) in SHAPES.items():
                cols = NU * D
                rounds = math.ceil(N / cols)
                kc = math.ceil(K / D)
                m_blocks = math.ceil(m / ACC)
                # sweep rows measured at m=1 and m=ACC
                key_m = 1 if m == 1 else ACC
                kc_ref, cyc_ref = level2[(D, bw, key_m)]
                cyc_round = cyc_ref * kc / kc_ref
                m_last = m - (m_blocks - 1) * ACC
                if m_blocks > 1:
                    # full blocks at m=ACC plus remainder scaled from m=1/ACC rows
                    kf, cf = level2[(D, bw, ACC)]
                    full = cf * kc / kf
                    if m_last == ACC:
                        rem = full
                    else:
                        kr, cr = level2[(D, bw, 1)]
                        rem = (cr * kc / kr) * m_last  # m=1 rows, per-row approx
                    cyc_round = (m_blocks - 1) * full + rem
                total_cyc = rounds * cyc_round
                out[(name, dk, bw)] = dict(cycles=total_cyc, ns=total_cyc * clk,
                                           rounds=rounds, clk_ns=clk)
    return out

if __name__ == "__main__":
    level2 = load_level2("results/jalapeno_level2_raw.txt")
    clocks = json.loads(Path("results/jphys_achieved_clocks.json").read_text())
    clocks = {(k.split("@")[0], int(k.split("@")[1])): v for k, v in clocks.items()}
    res = replay(level2, clocks)
    for bw in (64, 128):
        print(f"\n== BW={bw} B/cyc ==")
        print(f"{'shape':18s} {'32-dom us':>12s} {'64-dom us':>12s} {'64/32':>7s}")
        for name in SHAPES:
            a = res[(name, '32', bw)]["ns"] / 1e3
            b = res[(name, '64', bw)]["ns"] / 1e3
            print(f"{name:18s} {a:12.2f} {b:12.2f} {b/a:7.2f}")
    Path("results/jphys_replay.json").write_text(
        json.dumps({f"{k[0]}|{k[1]}|{k[2]}": v for k, v in res.items()}, indent=1))
