# Jalapeno source alignment: our model vs the public record (Tier 3)

Date: 2026-09-01. Sources: zartbot's architecture analysis
(zartbot.github.io/blog/arch/jalapeno/en.html), OpenAI/Broadcom
announcements, Hot Chips 2026 coverage. Purpose: pin every hard parameter
the public record commits to, and reconcile our matched-slice model
against it - what we got right, what we deviated on, and what Tier 1
must change to stay faithful.

## What the public record commits to

| parameter | public value | status |
|---|---|---|
| Tensor unit shape | **64x64 @ MXFP4** | confirmed |
| Array style | adder-tree reduction, NOT systolic | zartbot, high confidence |
| Stationarity | **output-stationary**; 64 FP32 accumulator columns in-array | zartbot |
| Reduction pipeline depth | **6-9 cycles** | zartbot |
| Units per core slice | ~15-16 (15 active, yield harvest) | zartbot estimate |
| Core slices per chip | 64 | confirmed |
| Clock | 1.7 GHz (TSMC N3P compute die) | confirmed |
| L1 per slice | 512 KB, shared tensor/SIMD; **no global L2** | zartbot |
| HBM | 216 GB HBM4, 15.4 TB/s total; ~240 GB/s local per slice | confirmed / zartbot |
| Precision | MXFP8 x MXFP4 (+BF16 attention) | confirmed |
| Peak | 13.4 PFLOPS MXFP4 @ 700 W | confirmed |
| NoC | 8x8 two-stage collective net + general NoC | zartbot, partial speculation |

## Scorecard: what our campaign already got right

1. **Adder-tree, not systolic** - our mac_array is exactly this. Correct.
2. **In-array FP32 accumulators** - ours live in-array (ACC rows). Correct.
3. **"64x64 is a bandwidth statement"** - the strongest confirmation.
   Per-slice local HBM is ~240 GB/s = ~141 B/cycle at 1.7 GHz across
   ~15 units => ~9.4 B/cycle/unit sustained, far below the per-unit feed
   knee (BW* = D bytes/cycle). The real chip lives deep in the STARVED
   regime and leans on reuse - precisely the regime where our Level-2/
   slice data says fine granularity buys nothing and integration cost
   makes coarse units win. Our slice-level reversal is the public
   choice's mechanism, now with the public bandwidth numbers behind it.
4. **Slice = several coarse engines + shared operand store** - the real
   slice packs ~15 units behind one 512 KB L1; integration cost scaling
   with engine COUNT (our headline) is exactly why you'd pick 15 big
   units over 60 small ones.

## Deviations our model must own (and what Tier 1 changes)

| ours today | real chip | Tier 1 action |
|---|---|---|
| unpipelined combinational dot (5-6 ns) | 6-9 cycle pipelined reduction @ 1.7 GHz | pipeline both engines to the SAME staging rule; target depth ~log2(D)/2+2, lands in the 6-9 range for D=64 |
| weight-stationary (reload per k-chunk) | output-stationary: operands stream, outputs sit | add OS mode: stream both operands, keep acc in-array; kills our 2x weight-reload artifact on narrow shapes |
| int8 operands | MXFP4/MXFP8 | keep int8 MACs (open-cell honesty) but halve BYTES/operand in the feed model: knee shifts from D to D/2 bytes/cycle |
| per-engine private SRAM banks | one 512 KB L1 shared by all units in slice | restructure banks as one shared multi-bank L1 with per-engine ports - the granularity question becomes port/bank arbitration, which is the real fight |
| 2 vs 8 engines per slice | ~15 units per slice | rescale matched pair: 15x(64x64) vs 60x(32x32), equal 61,440 MACs - or keep 2-vs-8 as a scaled proxy and say so |
| EXT_BW 64-128 B/c generous | ~141 B/c per slice across 15 units | add a starved operating point: ~9 B/c/unit with reuse factor R as the swept variable |
| no writeback cost | output-stationary drain phase | count drain cycles (64 FP32 cols/unit) |
| no NoC | 8x8 collective + NoC | out of scope until Tier 4 - documented |

## Honest limits that survive Tier 1

Nangate45 vs N3P (wire/gate balance), fakeram vs real compiled SRAM,
open flow vs Broadcom's, and everything zartbot marks as his own
estimate (unit count per slice, exact L1 banking). Our absolute numbers
will stay 45nm-toy numbers; the deltas and mechanisms are the product.

## Tier 1 experiment definition (frozen before RTL)

Matched pair, same invariants discipline as before:
A = 8x(32x32), B = 2x(64x64) (scaled proxy of 60-vs-15, disclosed),
both: output-stationary streaming, pipelined reduction (identical
staging rule), shared multi-bank L1 (16 fakeram macros, same total),
drain counted, feed swept at {9, 32, 64, 141} B/cycle-per-slice with
4-bit-operand byte accounting. Measure: cycles (RTL), area/clock/wire/
power (synth->P&R, identical flow), replay on the frozen shapes plus a
reuse-swept decode shape. Question unchanged: does granularity still
not matter physically once the model is faithful to the real dataflow?
