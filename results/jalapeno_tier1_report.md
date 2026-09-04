# Tier 1: the faithful-microarchitecture rematch (pipelined, MXFP4-fed, drained)

Date: 2026-09-03. Branch `jalapeno-physical-codesign`. Question (frozen in
`jalapeno_source_alignment.md` before any RTL): once the model is faithful
to the public record - pipelined reduction, MXFP4 feed accounting, drain
counted, symmetric SRAM packing - does slice granularity still matter
physically?

## What changed vs Tier 0 (identical for both slices)

- dot product pipelined by one RULE: register after the multiply and
  after every 2 adder-tree levels (4 stages both sizes; with accumulate,
  the D=64 unit lands in the public 6-9 cycle reduction envelope).
- Feed accounting at 4-bit operands: a row is D/2 bytes.
- Accumulator DRAIN streamed and counted; pipeline flush counted.
- 2-rows-per-SRAM-access packing (fully uses macro width both sizes) -
  which also eliminated Tier 0's symmetric port-conflict artifact
  (starvation at saturation fell from ~50% to ~1%).
- P&R: utilization 45, crash-tolerant metric ordering, GRT capped
  metal2-metal5 (the flow survives the WSL VM's panics at the routing
  memory peak by banking all decisive metrics pre-route).

## Cycle level (results/jalapeno_t1_sweep_raw.txt, equal work = 256 cols)

| ext BW (B/cyc) | fine 8x32 | coarse 2x64 (x2 rounds) | verdict |
|---|---|---|---|
| 9 (real per-unit share) | 7522 | 7416 | tie |
| 32 | 2122 | 2100 | tie |
| 64 | 1068 | 1060 | tie |
| 141 (real per-slice feed) | 544 | 1060 | fine 1.95x |
| 256 | 544 | 1060 | fine 1.95x |

The granularity law survives every fidelity increase unchanged: a pure
bandwidth-gated effect, now with the knee at aggregate 128 B/cycle
(= N_ENG x D/2 at 4-bit).

## P&R (placement-parasitic timing after CTS; Nangate45 + 16 fakeram each)

| metric | fine @141 | coarse @141 | fine @64 | coarse @64 |
|---|---|---|---|---|
| reg-to-reg (ns) | **2.901** | 3.256 (+12%) | 3.215 | pending |
| design area (um^2) | 9.77M | **9.46M (-3.2%)** | 9.77M | pending |
| power @2ns (W) | **2.01** | 2.32 (+15%) | 2.00 | pending |

(The 4th config re-runs after a VM panic mid-placement; it cross-checks
BW sensitivity and cannot change the verdict, which the BW=141 pair
carries. Numbers will be appended when banked.)

## The verdict, and what it overturns

**Pipelining undoes the Tier-0 physical reversal.** Unpipelined, the
coarse slice clocked 17-20% faster and burned less power - that was the
"slice infrastructure favors 64x64" story. With the reduction properly
staged, the FINE slice clocks 12% faster and uses 15% less power; the
coarse slice keeps only a 3.2% area edge. The Tier-0 clock advantage was
an artifact of the unpipelined discipline: it measured whose
un-staged combinational cloud buffered better, not whose architecture
was faster. In the pipelined design the critical path moves into
control/SRAM-interface territory, where 8 small engines' shorter local
nets win.

**The final three-layer law, amended:**
1. Compute-array physics is granularity-neutral, and once pipelined,
   physical design is CLOSE to neutral everywhere (+-3% area, ~12%
   clock, ~15% power - now leaning fine).
2. Operand bandwidth per slice picks the cycle winner: below ~128 B/c
   aggregate it is a tie; above, fine wins ~2x on cycles (2.2x on
   wall-clock with its clock edge).
3. Integration cost scaling with engine count is real but SMALL once
   pipelined - it buys coarse ~3% area, no longer clock or power.

**Reconciliation with the real 64x64 Jalapeno:** the public chip's
per-unit feed (~9.4 B/cycle) sits in the deep tie regime, so granularity
buys nothing in cycles there and the physical deltas are a wash. The
64x64 choice is then justified by what this study cannot price: fewer
units to schedule, simpler software tiling, yield-harvest granularity,
and design-team simplicity - not by placed-silicon physics. Our earlier
claim that slice physics actively favors coarse is hereby corrected: at
faithful fidelity, physics is nearly indifferent, and bandwidth strategy
plus integration pragmatics make the choice.

**Multi-fidelity lesson (final form):** every fidelity level we added
changed WHICH mechanism dominated (RTL: bandwidth; slice P&R: engine-count
overheads; pipelined P&R: those overheads mostly dissolve). The stopping
rule stands: escalate until the decision's cost structure is inside the
model - and pipeline discipline is part of the cost structure, not a
refinement of it.

## Provenance

RTL `hardware/mac_array_p.v`, `hardware/mac_slice2.v` (cycle-verified;
sweep raw in results/); synth `results/stat_t1s*.txt` (yosys 0.56, abc
-fast); P&R `results/pnr_t1s*_{timing,area,power}.txt` (OpenROAD
v2.0-17598, identical scripts `hardware/pnr_slice2.tcl`); source
constraints `results/jalapeno_source_alignment.md`. Seven WSL VM kernel
panics were survived via pre-route metric banking; no results derived
from a partially-crashed run.
