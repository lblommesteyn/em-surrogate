# Design-optimization report: the selective stack on a stub-loaded interconnect

Date: 2026-08-23. Branch `external-data`. Question: can the hybrid
surrogate + selective-solver system find designs comparable to full openEMS
optimization with fewer full-wave calls?

**Verdict: demonstrated on one of three seeds, not robustly.** On seed 0 the
hybrid reached a verified objective of 0.422 with **8 openEMS calls**, better
than solver-only's 0.579 after **60 calls** (7.5x fewer calls, better
design). On seeds 1 and 2 the hybrid spent its full 60-call budget verifying
surrogate-favoured gap designs and finished worse than solver-only at equal
budget (0.818 / 0.983 vs 0.690 / 0.682). The selective machinery never
reported an unverified design and correctly flagged the designs that fooled
the surrogate; the failure mode is a verification *policy* that did not act
on a risk signal that had the right answer. All numbers below are true
openEMS objectives; no methodology was retuned on outcomes.

Two campaigns were run. **Task v1** (mandatory 300-700 um series gap) proved
physically infeasible in-band: every method converged to J ~1.49 (|S21|
<= 0.018) and nothing could separate; it is kept as
`design_optimization_report_v1_infeasible.md`. **Task v2** (this report)
makes the gap optional, as the milestone specification allows; nothing else
changed. Both were declared before running. Two evaluator fixes were logged
en route: the frozen simulate() fabricates S22 := S11 (valid only for
symmetric structures), so the passivity gate for these asymmetric designs
uses the measured port-1 column norm; and per-run checkpointing was added
after a PC reboot (solver-only replays bit-identically, DE-driven hybrid
runs do not). Artifacts: results/design_opt_v2_metrics.json,
results/design_runs_v2/, solver cache results/design_cache/ (~900 solves,
content-addressed), scripts/run_design_opt.py, src/emsurr/design_task.py,
scripts/oems_eval.py.

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

## Results (true openEMS objective; anytime best at 5/10/20/40/60 calls)

| run | 5 | 10 | 20 | 40 | 60 | final verified J | calls | surrogate evals | wall |
|---|---|---|---|---|---|---|---|---|---|
| solver-only s0 | - | .711 | .711 | .711 | .579 | 0.579 | 60 | 0 | 26 min |
| **hybrid s0** | .730 | **.422** | .422 | .422 | .422 | **0.422** | **8** | 492 | 6 min |
| surrogate-only s0 | .357 | | | | | 0.357 | 1 | 492 | <1 min |
| uncertainty-fallback s0 | .730 | .347 | .347 | .347 | .347 | 0.347 | 8 | 492 | cache |
| random-fallback s0 | 1.497 | .854 | .593 | .349 | .335 | 0.335 | 53 | 492 | 24 min |
| solver-only s1 | - | 1.484 | .702 | .702 | .690 | 0.690 | 59 | 0 | 35 min |
| hybrid s1 | 1.44 | 1.44 | .818 | .818 | .818 | 0.818 | 60 | 492 | 42 min |
| surrogate-only s1 | 1.494 | | | | | 1.494 | 1 | 492 | 1 min |
| solver-only s2 | - | .826 | .797 | .773 | .682 | 0.682 | 59 | 0 | 36 min |
| hybrid s2 | 1.496 | 1.494 | 1.494 | .983 | .983 | 0.983 | 60 | 492 | 32 min |
| surrogate-only s2 | 1.494 | | | | | 1.494 | 1 | 492 | <1 min |

The uncertainty-gated run chose nearly the same verification set as hybrid
s0 (cache hits). Invalid designs: 16 of ~900 solves failed the column
passivity gate and scored 2.0. Final geometries of every winning run are
gap-free, wide-line (w 700-800 um), shortest-stub designs at the left
boundary: physically sensible broadband-match solutions, not artifacts
(column norms 1.002-1.017, within the 1.1 gate). Final S-curves, surrogate
vs openEMS responses, retrieval gaps and geometries for every run are in
the metrics file (final_verification); trajectories and verification logs
per run in design_runs_v2/.

## Answers

**1. Calls saved:** seed 0: 8 vs 60 (7.5x) with a better design (0.422 vs
0.579); no solver-only run reached 0.422 at any budget. Seeds 1-2: none;
the hybrid used all 60 calls and finished behind solver-only. The
uncertainty-gated and random-fallback ablations on seed 0 reached 0.347 (8
calls) and 0.335 (53 calls): the seed-0 gain comes from surrogate
pre-screening of the gap-free region, not from which risk signal gated
verification.

**2. Equal budget:** hybrid wins on seed 0 at every budget from 10 calls on;
loses on seeds 1-2. Not robust.

**3. Equal quality:** solver-only never matched hybrid s0's 0.422, so the
reduction there is unbounded within the sweep; on seeds 1-2 the question
inverts (hybrid never matched solver-only).

**4. Exploitation prevented: yes.** Surrogate-only was fooled on 2 of 3
seeds: its claimed optima (surrogate J ~0.91, gap designs) verified to
1.494, the worst class of design. The hybrid reported only verified
designs, and on the losing seeds its verified result (0.82-0.98) still beat
the surrogate's own belief.

**5. Regime:** topology, on every pool, correctly. The mechanism of the
seed 1-2 losses is now precise: the surrogate prefers gap designs (it
predicts transmission through them), so its population drifts into the gap
basin; those candidates carry the highest retrieval-gap of the whole
campaign (mean 3.0-3.1 on the losing seeds vs 2.3 on the winning seed;
final gap designs 3.08 vs gap-free 1.66-1.99), i.e. the risk signal
identified the trap, but the declared policy still verified them because
they were "promising" or "risky-but-within-20%": 40-58 wasted calls per
losing seed, 59-60 of 60 verifications badly wrong (mean |dJ| 0.53).

**6. Where the surrogate is fooled:** on every gap design (phantom
pass-bands), and mildly on gap-free ones (it under-predicts J by ~0.4 but
orders them usefully enough that seed 0 won). The retrieval-gap separates
the two classes cleanly (>3.0 vs <2.0).

**7. Ready for a realistic PCB/package problem?** Not as a call-saving
claim; one robust seed is not a demonstration. But the path is now concrete
and cheap: (a) a verification policy that uses the risk signal to refuse,
not just to prioritise (the data show a gap threshold near 2.5 would have
excluded every wasted call on seeds 1-2; recorded here as an observation,
not applied, because selecting it on these outcomes would be tuning); (b) a
surrogate trained on in-support samples of the design space so the hybrid
runs in the extrapolation regime where the stack has demonstrated ranking
skill. With both, the seed-0 behaviour (8 calls, better design) is the
expected case rather than the lucky one.
