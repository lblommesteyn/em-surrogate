# Jalapeno slice-level granularity: does slice infrastructure reverse fine granularity? (FINAL)

Date: 2026-09-01. Branch `jalapeno-physical-codesign`. Follow-on to
`jalapeno_physical_granularity_report.md` (bare 4096-MAC domains, where
fine granularity won or tied everything). Question: build two matched
Jalapeno-like SLICES - SRAM macros, slice-level operand distribution,
multiple engines on both sides - and see whether slice-level physical
infrastructure finally reverses the fine-granularity advantage.

## Matched slices (invariants)

A (fine)  = 8 engines of 32x32, ACC=4/engine.
B (coarse) = 2 engines of 64x64, ACC=8/engine.
Equal by construction: 8192 int8 MACs, 1024 int32 accumulator words,
16 identical fakeram45_128x256 SRAM macros = 64 KB operand storage per
slice (fine: 1 wide x 2 deep per engine; coarse: 2 wide x 4 deep), one
shared EXT_BW-byte/cycle external port, round-robin distributor, per-
engine SRAM-FIFO with 2-entry skid (1-cycle SRAM latency), single-port
arbitration read-priority - every policy parameter-generic and identical;
`mac_array.v` reused verbatim. Both slices now carry an inter-engine
distributor level, removing the prior asymmetry where only the fine side
had one. Flow: iverilog cycle sim -> yosys 0.56 synthesis (fakeram
liberty blackbox) -> OpenROAD P&R (floorplan 55% util, gpl, TritonMacroPlace
-halo 4, second gpl, repair_design, CTS, global route metal2-metal7,
SPEF-estimated timing; identical script/constraints/effort both).

## Slice cycle behavior (RTL, results/jalapeno_slice_level2_raw.txt)

The domain-level law survives SRAM banking unchanged: per equal work the
slices tie at BW <= 64 B/c (external-bandwidth-bound) and the fine slice
does ~1.96x better cycles at BW >= 128 (8 engines absorb more parallel
operand streams). Disclosed artifact: single-port read/write conflicts
inflate BOTH slices ~2x over ideal at saturation (~50% starve even at
BW=512) - identical policy, symmetric effect; a real slice would
double-bank or dual-port this away.

## P&R results (placed + globally routed, Nangate45 + fakeram45)

| metric | 8x32 @64 | 2x64 @64 | 8x32 @128 | 2x64 @128 |
|---|---|---|---|---|
| design area (um^2) | 8,174,350 | **8,021,324 (-1.9%)** | 8,175,044 | **8,031,356 (-1.8%)** |
| total wirelength (um) | **103.8M** | 108.5M (+4.5%) | **104.0M** | 107.3M (+3.2%) |
| reg-to-reg critical path (ns) | 6.599 | **5.483 (-17%)** | 6.554 | **5.262 (-20%)** |
| total power @2ns (W) | 1.09 | **1.01** | 1.08 | **1.01** |

**Yes - slice-level infrastructure reverses the physical verdict.** At the
bare-domain level fine granularity was better or equal on every physical
metric; at slice level the coarse slice is smaller, clocks 17-20% faster,
and burns slightly less power at both operand bandwidths. Fine keeps only
its wirelength edge (3-5%). Synthesis foreshadowed the flip (coarse
-3.4% cells/area) and P&R confirmed it - this time synthesis area DID
predict the P&R area winner, again.

Mechanism: the fine slice's slice-level costs scale with engine count,
not engine width - 8 distributor endpoints, 8 SRAM-FIFO controllers, 8
skid buffers, 8 pending/stored counters, and an 8-way reduction tree for
done/obs/counters. The coarse slice amortizes the same infrastructure
over 2 engines; its wider engine-internal buses route slightly worse
(the +3-5% wirelength) but its control perimeter is a quarter the size,
and after buffering that is what sets the critical path. The bare-domain
experiment could not see this because only its fine side had any
distribution infrastructure at all.

## Workload replay (frozen Jalapeno shapes, achieved clocks)

Uniform-round replay (all lanes fed, exactly as the measured sweep runs;
coarse/fine wall-time ratio, >1 = fine wins):

| shape | BW=64 | BW=128 |
|---|---|---|
| attn_qkv_b1 (N=80) | 0.41 | 0.78 |
| moe_exp_up_b1 (N=90) | 0.41 | 0.78 |
| moe_exp_up_b8 | 0.21 | 0.40 |
| dense_gateup_b1 (N=896) | 0.72 | **1.37** |
| prefill (m=128, N=80) | 0.21 | 0.40 |

Two caveats, stated with direction of bias. (1) Jalapeno's inference
shapes are NARROW (N=80-96): one coarse round (128 cols) already covers
them, while the fine slice's 256 lanes sit two-thirds idle yet its
composition still pays 2x the weight-reload chunks (kc = K/32 vs K/64).
That idle-lane penalty is real. (2) The uniform-round model feeds all 8
fine engines; a real controller would feed only the ~3 active ones,
improving their bandwidth share - an active-engine analytical correction
moves the narrow-shape entries at BW=64 from ~0.41 toward ~0.9 (near-tie),
so the table's low-BW narrow-shape numbers are a lower bound for fine,
not a measurement. The batched rows (b8, prefill) and the wide-GEMM row
are robust to this: fine's extra m-block passes and the coarse slice's
full-width utilization are composition facts.

## Verdict

Slice-level physical infrastructure reverses the fine-granularity
advantage. Physically the 2x64 slice is strictly better (smaller, 17-20%
faster clock, less power) except for a few percent of wirelength. On
Jalapeno's actual narrow low-batch shapes the coarse slice wins or ties
everywhere except wide GEMMs (dense_gateup-class, N >= several hundred)
with >= 128 B/cycle operand feed, where the fine slice still nets ~1.4x.
Combined with the bare-domain result, the granularity story is now
three-layered: (i) compute-array physics is granularity-neutral, (ii)
operand bandwidth per domain picks the cycle winner, (iii) slice
integration costs scale with engine COUNT and pull the physical optimum
toward coarser engines. A 64x64 choice for a Jalapeno-like slice is what
this methodology recommends given narrow inference shapes and <= 64 B/c
per-domain feed - the public reconstruction's granularity now has a
physical-design justification, not just a bandwidth one.

Fidelity-escalation lesson (EM-surrogate recipe): the reversal appeared
ONLY at the slice level - domain-level P&R, RTL, and analytical models
all favored fine. Every fidelity level answered the question it could
see; the decision-relevant level is the one containing the costs the
decision trades on (here: integration infrastructure). Escalate until
the cost structure of the decision is inside the model, then stop.

## Provenance

- RTL: `hardware/mac_slice.v`, `hardware/fakeram45_128x256_beh.v`,
  `hardware/tb_slice.v` (mac_array.v verbatim; commit abd55a4).
- Sweep: `results/jalapeno_slice_level2_raw.txt`.
- Synthesis: `results/stat_s*.txt` (fine 7.40M um^2, coarse 7.15M).
- P&R: `results/pnr_s*_{timing,area,power}.txt`; OpenROAD v2.0-17598,
  `hardware/pnr_slice.tcl` (identical both; metal2-metal7 signal layers,
  gpl -max_phi_coef 1.03; one OOM at 36 GB fixed by 40 GB + layer cap).
- Replay: `scripts/jphys_slice_replay.py`,
  `results/jphys_slice_achieved_clocks.json`, `results/jphys_slice_replay.json`.
