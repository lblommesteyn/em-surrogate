# Jalapeno physical co-design: 32x32 vs 64x64 granularity after P&R (FINAL)

Date: 2026-08-31. Branch `jalapeno-physical-codesign`. Question: does the
32x32-vs-64x64 matrix-unit granularity verdict survive real placement,
routing, timing, and operand-delivery cost?

## Setup (invariants held throughout)

Two matched compute domains, identical everything except partitioning:
A = 4 units of 32x32 (D=32, ACC=4/unit), B = 1 unit of 64x64 (D=64,
ACC=8). Both: 4096 int8 MACs, 512 int32 accumulator words, identical
FSM/dataflow (`mac_array.v` shared), one shared EXT_BW-byte/cycle external
operand port feeding a round-robin distributor with per-unit double row
buffers - the distributor is the measured operand-delivery cost, same
style both sizes. Fidelity ladder with NO retuning between levels:
analytical (jalapeno-sim) -> RTL cycle sim (iverilog) -> synthesis (yosys
0.56 `abc -fast`, Nangate45 typical) -> P&R (OpenROAD: floorplan 60%
util, global place, repair_design, CTS, global route, SPEF-estimated
timing; identical script, constraints - 2.0 ns clock, 0.2 ns I/O - and
effort for every config). Fairness checks before P&R: equal useful GEMM
work per round (128 cols/round for A = 2x A rounds vs B), matched storage
(FF row buffers both; no SRAM macros anywhere), identical SDC, distributor
inside both netlists, no pruning (dot_col instance counts = 4096 mults
each; obs XOR port), tiling/padding handled in composition.

## P&R results (placed + globally routed, Nangate45)

| metric | 4x32 @BW64 | 1x64 @BW64 | 4x32 @BW128 | 1x64 @BW128 |
|---|---|---|---|---|
| design area (um^2) | 3,674,173 | 3,834,174 (+4.4%) | 3,862,150 | 3,788,173 (-1.9%) |
| total wirelength (um) | 50.1M | 55.1M (+10%) | 55.6M | 55.4M (0%) |
| reg-to-reg critical path (ns) | 4.662 | 4.755 (+2%) | 4.072 | 5.394 (+32%) |
| total power @2ns (W) | 1.12 | 1.38 (+23%) | 1.48 | 1.68 (+14%) |
| clock sinks | 54,103 | 50,390 | 58,199 | 51,414 |
| placement overflow | 0.103 | 0.101 | 0.101 | 0.100 |

Neither domain closes 2.0 ns (both unpipelined by design, disclosed);
achieved clock = reg-to-reg arrival. The 64x64 config could not even be
SYNTHESIZED by yosys 0.33 (hard hashlib abort) and its global placement
diverged once under default RePlAce settings - the flat 64-wide datapath
is the fragile one for the tools, not the 4-unit domain.

## Workload replay (frozen Jalapeno shapes, Level-2 cycles x achieved clock)

64-domain time / 32-domain time (>1 = 32 wins):

| shape | BW=64 | BW=128 |
|---|---|---|
| attn_qkv_b1 (1x2880x80) | 1.00 | 2.59 |
| moe_exp_up_b1 (1x2880x90) | 1.00 | 2.59 |
| moe_exp_up_b8 (8x2880x90) | 0.51 | 1.32 |
| dense_gateup_b1 (1x8192x896) | 1.00 | 2.59 |
| prefill (128x2880x80) | 0.51 | 1.32 |

## Answers

**1. Does physical design favor 32 or 64?** It does not overturn the RTL
verdict; it sharpens it. Area is a near-tie (within +-4%). At BW=64 the
32-domain is modestly better on wirelength (-9%) and power (-19%) at equal
clock; at BW=128 it also clocks 1.32x faster. The system winner remains
bandwidth- and batch-dependent: 64x64 wins batched work at low external
bandwidth (2x), 4x32 wins low-batch work at high bandwidth (2.6x, now
compounded by its clock edge).

**2. Mechanism.** The feared distributor/control wiring cost of fine
granularity is ~1% of domain area (37k of 3.53M um^2 at BW=64) and does
not show up in routed wirelength - four 32x32 tiles place and route MORE
compactly than one 64x64 blob (50.1M vs 55.1M um wire at BW=64). The
64-wide unit's theoretical adder-tree depth advantage (log2 D: 6 vs 7
stages... actually deeper mult+tree) is erased after buffering: real
critical paths are dominated by buffered high-fanout control/operand nets,
which scale with unit width, not helped by it.

**3. Does 32 incur the hypothesized wiring/control penalty?** No. It pays
+2.4% area at BW=128 (bigger aggregate row buffering) and ~7% more clock
sinks; it gains that back in wirelength, power, and (at BW=128) clock.

**4. Can synthesis metrics predict the P&R winner?** Area: yes - the
synthesis ordering (3.53M vs 3.67M um^2, +4% for 64) is exactly the
placed ordering. Timing: catastrophically no - unbuffered post-synthesis
STA gave 201 ns vs 19,134 ns (95x apart, pure fanout artifact); after
P&R buffering the same designs are 2% apart. Any granularity decision
based on synthesis-level timing would be garbage in this flow.

**5. When must hardware search escalate fidelity?** Cycle-accuracy needs
RTL (the analytical model already misses the BW-gated 2x). Area ordering
is stable from synthesis onward - a search may screen on it cheaply.
Timing and power ordering exist only after placement + repair_design +
routing estimates; escalate to P&R exactly when the decision hinges on
achievable clock or energy, and only for the shortlisted candidates.
This mirrors the EM-surrogate recipe: validate each cheap evaluator
against the next level before trusting it to rank.

**6. Public 64x64 Jalapeno reconstruction: support or contradict?**
Refines it. 64x64 is the right choice iff the deployed regime is
operand-bandwidth-poor per domain (<=64 B/cycle) and/or batch-rich -
there it wins 2x on batched shapes and ties elsewhere. If a domain can
be fed >=128 B/cycle, low-batch decode (Jalapeno's stated target) favors
4x32 by 2.6x. The public choice of 64x64 is therefore evidence the real
chip's per-domain operand bandwidth is in the starved regime - consistent
with the reconstruction's SRAM/collectives negatives, and with granularity
being chosen to match feed capability, not compute efficiency.

**7. Smallest next step toward a full slice.** Pipeline the dot product
(1-2 register stages, identical for both sizes) so both domains close a
real clock and the comparison moves from arrival-time ratios to
throughput at closed timing; then detailed routing on the BW=128 pair
(global-route congestion was clean, overflow ~0.10, so no surprise
expected); then replace FF row buffers with fakeram SRAM macros to expose
the SRAM-to-MAC wiring distance term the domain currently hides.

## Provenance

- RTL: `hardware/mac_array.v` (dot_col partitioning, cycle-verified
  identical to jalapeno-sim baseline: 1057/520/520 cycles), `mac_domain.v`.
- Level-2 sweep: `results/jalapeno_level2_raw.txt`.
- Synthesis: `results/stat_*.txt`, `results/sta_*.txt` (yosys 0.56 abc
  -fast; yosys 0.33 aborts on flat D=64 - hashlib limit; its WSL crash
  dumps were the machine-wide disk incidents of Aug 30).
- P&R: `results/pnr_*_{timing,area,power}.txt`, OpenROAD v2.0-17598,
  identical `hardware/pnr_domain.tcl` (gpl `-max_phi_coef 1.03` both after
  one 64x64 divergence under defaults).
- Replay: `scripts/jphys_replay.py` -> `results/jphys_replay.json`,
  achieved clocks `results/jphys_achieved_clocks.json`.
