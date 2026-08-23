# Design-optimization report: the selective stack on a stub-loaded interconnect

Date: 2026-08-23 (final, 5 seeds). Branch `external-data`. Question: can the
hybrid surrogate + selective-solver system find designs comparable to full
openEMS optimization with fewer full-wave calls?

## Verdict (5 seeds, corrected policy, all numbers openEMS-verified)

| seed | solver-only best @60 | hybrid best | hybrid calls | hybrid beats solver-only's 60-call best at call | surrogate-only |
|---|---|---|---|---|---|
| 0 | 0.579 | **0.289** | **17** | 16 | 0.357 |
| 1 | 0.690 | **0.338** | **36** | 35 | 1.494 |
| 2 | 0.682 | 1.465 | 60 | never | 1.494 |
| 3 | 0.345 | 1.441 | 60 | never | 1.549 |
| 4 | 0.790 | 1.384 | 60 | never | 1.500 |

**The claimed demonstration holds on 2 of 5 seeds and fails on 3.** Where it
holds, it is dramatic: the hybrid surpasses everything solver-only achieves
in 60 calls by call 16 (seed 0) and call 35 (seed 1), and finishes at
roughly half the solver-only objective - a genuine "substantially fewer
calls at better quality" result. Where it fails, the mechanism is single
and fully characterized: the surrogate has no in-support coverage of this
design space (the stack's own validation diagnosed objective Spearman
-0.50 and 100% of designs beyond the training support BEFORE optimization),
it hallucinates pass-bands through series-gap designs, and when the initial
population lands in that basin the verify-and-replace loop burns the budget
confirming phantom optima. The frozen retrieval-gap flags exactly those
candidates (mean gap 3.0-3.1 on every losing seed's verifications; the
fooled designs score >3.0 vs <2.0 for real ones), but the pre-declared
policy only prioritizes by risk - it never refuses - and changing that
after seeing these outcomes would be tuning, so it was not changed.

What IS established across all 5 seeds: the safety contract (no unverified
design was ever reported; surrogate-only was fooled on 4/5 seeds, claiming
~0.9 for designs that verify at 1.49-1.55, and the hybrid's verified output
beat the surrogate's belief every time); correct regime detection (topology
regime on every pool); and the risk signal's diagnostic value (it separates
the trap class cleanly). What is NOT established is robust solver-call
savings with a surrogate that has zero skill on the space - the stack can
refuse to be fooled, but it cannot conjure search efficiency out of a
model it itself measures as untrustworthy.

## Task v2 (declared before optimization)

Microstrip on the frozen 254 um / er 3.5 stackup; line width w 400-800 um;
two mandatory open stubs (3-10 mm long, 300-700 um wide, positions on the
left half); optional series gap g (0 or 300-700 um). Objective over 2-6 GHz:
J = max|S11| + max(0, 0.5 - min|S21|), lower better, invalid -> 2.0.
Optimizer: one differential-evolution implementation for all methods (pop
12, 40 generations for surrogate-driven runs; pop 8 solver-only); solver
budget 60; seeds 0-2; anytime best recorded at every solver call. Hybrid
policy: verify a candidate iff it would become the verified incumbent, or
its retrieval-gap exceeds the train p95 while within 20% of the incumbent.
Stack: frozen-size DeepSets ensemble (6-type tokens) trained on the
synthetic training families + analytic gapped_line; retrieval-gap
(distance-weighted k=3) in the frozen extended-vocabulary encoder;
ensemble variance; frozen logistic deployment switch.

## Validation set (24 random designs; no retuning)

Pool novelty z-shift ~21-24, so the switch selects the topology regime;
100% of designs exceed the train gap-p95. The surrogate's objective ranking
is anti-correlated with openEMS (Spearman -0.50, MAE 0.28; response error
1.60, unphysical |S21| > 1 on gap designs), while both risk signals predict
its error well (retrieval-gap 0.75, ensemble variance 0.77 Spearman). The
stack knew the surrogate was untrustworthy before optimization began.

## Results (v2b, true openEMS objective; anytime best at 5/10/20/40/60 calls)

| run | 5 | 10 | 20 | 40 | 60 | final verified J | calls | surrogate evals |
|---|---|---|---|---|---|---|---|---|
| solver-only s0 | - | .711 | .711 | .711 | .579 | 0.579 | 60 | 0 |
| **hybrid s0** | 1.198 | 1.198 | **.289** | .289 | .289 | **0.289** | **17** | 492 |
| surrogate-only s0 | .357 | | | | | 0.357 | 1 | 492 |
| uncertainty-fallback s0 | 1.198 | 1.198 | .295 | .295 | .295 | 0.295 | 15 | 492 |
| random-fallback s0 | 1.497 | 1.482 | 1.469 | .293 | .293 | 0.293 | 52 | 492 |
| solver-only s1 | - | 1.484 | .702 | .702 | .690 | 0.690 | 59 | 0 |
| **hybrid s1** | 1.44 | 1.44 | 1.44 | **.338** | .338 | **0.338** | **36** | 492 |
| surrogate-only s1 | 1.494 | | | | | 1.494 | 1 | 492 |
| solver-only s2 | - | .826 | .797 | .773 | .682 | 0.682 | 59 | 0 |
| hybrid s2 | 1.496 | 1.496 | 1.493 | 1.465 | 1.465 | 1.465 | 60 | 492 |
| surrogate-only s2 | 1.494 | | | | | 1.494 | 1 | 492 |
| solver-only s3 | - | .345* | .345 | .345 | .345 | 0.345 | 60 | 0 |
| hybrid s3 | 1.496 | 1.496 | 1.493 | 1.441 | 1.441 | 1.441 | 60 | 492 |
| surrogate-only s3 | 1.549 | | | | | 1.549 | 1 | 492 |
| solver-only s4 | - | .790 | .790 | .790 | .790 | 0.790 | 60 | 0 |
| hybrid s4 | 1.496 | 1.494 | 1.494 | 1.384 | 1.384 | 1.384 | 60 | 492 |
| surrogate-only s4 | 1.500 | | | | | 1.500 | 1 | 492 |

(*seed-3 anytime granularity limited by cache replay after the solver-crash
recovery.) Invalid designs: 16 of ~1300 solves failed the measured-column
passivity gate; one openEMS crash on a degenerate geometry is handled as
invalid by the hardened evaluator. Winning-run geometries are gap-free,
wide-line (w 750-780 um), short-stub designs spanning the left half -
physically sensible broadband matches, not artifacts (column norms
1.002-1.058, within the 1.1 gate). Final S-curves, surrogate-vs-openEMS
responses, retrieval gaps and geometries: design_opt_v2b_metrics.json
(final_verification); per-run trajectories and verification logs:
design_runs_v2b/. The v2 (defective-policy) run is preserved in
design_opt_v2_metrics.json for comparison: the policy fix alone moved seed
0 from 0.422@8 to 0.289@17 and seed 1 from a 60-call loss to 0.338@36.

## Answers

**1. Calls saved:** on the winning seeds, the hybrid did not merely match
solver-only cheaper - it surpassed solver-only's ENTIRE 60-call result by
call 16 (seed 0) and call 35 (seed 1) and kept improving (final 0.289 /
0.338 vs 0.579 / 0.690), i.e. >=3.75x and >=1.7x fewer calls for strictly
better designs. On seeds 2-4: no savings; full budget consumed for worse
results. The seed-0 ablations (uncertainty-gate 0.295@15, random 0.293@52)
show the win comes from surrogate pre-screening plus verification, largely
independent of which risk signal gates.

**2. Equal budget:** hybrid wins seeds 0-1 from calls 16-35 onward, loses
seeds 2-4 at every budget. 2/5 - not robust.

**3. Equal quality:** on seeds 0-1 solver-only never reaches the hybrid's
quality at any budget in the sweep, so the reduction is unbounded there; on
seeds 2-4 the question inverts.

**4. Exploitation prevented: yes, 5/5.** Surrogate-only was fooled on 4 of
5 seeds (claims ~0.9, verifies 1.49-1.55, the worst class); every reported
hybrid design is openEMS-verified, and the hybrid's verified result beat
the surrogate's own belief on every seed.

**5. Regime:** topology, correctly, on every pool. The loss mechanism on
seeds 2-4 is verify-and-replace churn: a gap-heavy initial population +
a surrogate that hallucinates gap pass-bands means each verified phantom
(true J ~1.49) is replaced in the population, and the next surrogate-scored
gap trial (~0.9) wins acceptance and gets verified in turn - 58-60
verifications per losing seed, 59-60 of them badly wrong (mean |dJ| ~0.53),
verification gap_mean 3.0-3.1.

**6. Where the surrogate is fooled:** all gap designs (phantom pass-bands;
retrieval-gap > 3.0) and mildly on gap-free ones (under-predicts J by ~0.4
but orders them well enough to drive seeds 0-1 to 0.29-0.34). The frozen
retrieval-gap separates the trap class cleanly (>3.0 vs <2.0) on every
seed - the information needed to avoid all three losses existed at
decision time.

**7. Ready for a realistic PCB/package problem?** Not as a robust
call-saving claim; 2 of 5 seeds is not a demonstration. But the path is now concrete
and cheap: (a) a verification policy that uses the risk signal to refuse,
not just to prioritise (the data show a gap threshold near 2.5 would have
excluded every wasted call on seeds 1-2; recorded here as an observation,
not applied, because selecting it on these outcomes would be tuning); (b) a
surrogate trained on in-support samples of the design space so the hybrid
runs in the extrapolation regime where the stack has demonstrated ranking
skill. With both, the seed-0 behaviour (8 calls, better design) is the
expected case rather than the lucky one.
