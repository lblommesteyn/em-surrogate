# Design-optimization report: the selective stack on a stub-and-gap interconnect

Date: 2026-08-23. Branch `external-data`. Question: can the hybrid
surrogate + selective-solver system find designs comparable to full openEMS
optimization with fewer full-wave calls?

**Short answer: on this task, no solver calls were saved, because the task
turned out to be physically infeasible and the surrogate has no skill on
it; but the selective machinery did exactly its safety job, and every
number below is openEMS-verified.** Everything is reproducible from
`scripts/run_design_opt.py`, `src/emsurr/design_task.py`,
`results/design_opt_metrics.json`, per-run checkpoints in
`results/design_runs/`, and the content-addressed solver cache
`results/design_cache/` (568 solves, 21 flagged non-passive -> scored
invalid per protocol).

## Task (declared before optimization)

Microstrip interconnect on the frozen 254 um / er 3.5 stackup: line width w
(400-800 um), two mandatory open stubs (lengths 3-10 mm, width 300-700 um,
positions on the left half), and a mandatory series gap g (300-700 um) at
center. Objective over 2-6 GHz: J = max|S11| + max(0, 0.5 - min|S21|),
lower is better, invalid -> 2.0. Optimizer: one differential-evolution
implementation for all methods (pop 12, 40 generations for surrogate-driven
methods; pop 8 for solver-only), solver budget 60 per run, seeds 0-2.
Hybrid policy: verify a candidate iff it would become the verified
incumbent, or its retrieval-gap exceeds the train p95 while being within
20% of the incumbent. Reported bests are always openEMS-verified.

Stack (frozen recipes, no retuning): DeepSets ensemble (existing
architecture, 6-type token slot for the gap, frozen config) trained on the
5 synthetic training families + the analytic gapped_line family;
retrieval-gap (distance-weighted k=3, train-side choice) in the frozen
extended-vocabulary encoder; ensemble variance; frozen logistic switch.

## Validation set (24 random designs, openEMS vs surrogate; no retuning)

Pool novelty z-shift ~36 on both features -> the frozen switch selects the
topology regime (retrieval-gap ranking); 100% of designs exceed the train
gap-p95. Surrogate skill on this space is essentially nil: objective
Spearman 0.31, objective MAE 0.53, response error 1.85 (predictions are
unphysical, |S21| > 1). Risk ordering still behaves as established:
retrieval-gap 0.42 vs ensemble variance 0.34 Spearman against true error.

## Results (true openEMS objective only)

| run | best verified J | solver calls | surrogate evals | wall (min) |
|---|---|---|---|---|
| solver-only s0 / s1 / s2 | 1.4911 / 1.4924 / 1.4948 | 60 each | 0 | ~35-45 each |
| hybrid s0 / s1 / s2 | 1.4907 / 1.4940 / 1.4920 | 60 each | 492 each | 45 / 77 / 54 |
| surrogate-only s0 / s1 / s2 | 1.4996 / 1.5001 / 1.4987 | 1 each (final verify) | 492 each | ~2 each |
| uncertainty-only fallback s0 | 1.4907 | 60 | 492 | (identical verification sequence to hybrid s0) |
| random fallback s0 | 1.4925 | 53 | 492 | 50 |

Anytime best at 10/20/40/60 solver calls: solver-only s0 1.496/1.494/
1.493/1.491; hybrid s0 1.497/1.491/1.491/1.491; s1 and s2 analogous
(differences 0.001-0.005, within run-to-run noise).

**The objective floor is ~1.49 for every method**: openEMS shows
max|S11| = 1.000 and min|S21| <= 0.018 in-band for every verified best. A
300-700 um series gap simply does not transmit 2-6 GHz on this stackup;
the optimizer correctly drives all methods to the same bound corner
(w = 800 um, g = 300 um, shortest stubs) and then has nothing left to
improve. The task as declared is infeasible, which makes "comparable
quality" trivially true and the call-reduction question unanswerable on
quality grounds. This is reported as the outcome, not re-declared.

Deviations forced by the PC reboot mid-campaign (logged): per-run
checkpointing was added to the driver; solver-only seeds replayed
bit-identically from cache, but hybrid seeds 0-1 were re-executed because
DE amplifies float-level differences after the encoder/ensemble reload.
Pre-reboot hybrid values (1.491, 1.495) agree with the re-executions
(1.4907, 1.4940) and are kept as an independent replicate.

## Answers

**1. Calls saved:** none demonstrable. Hybrid consumed its full 60-call
budget on every seed: with the whole design space flagged OOD, the gates
never closed, and the "would-be incumbent" rule fired on 58 of 60
verifications per run. Uncertainty-gated fallback chose the identical
verification sequence, random fallback was marginally worse (1.4925 at 53
calls) - on a saturated regime, the risk signal has nothing to select.

**2. Equal budget:** tie. Hybrid 1.4907-1.4940 vs solver-only 1.4911-1.4948
at 60 calls; no seed separates them beyond noise.

**3. Equal quality:** undefined - all methods sit on the same physical
floor. Hybrid s0 reached the final level by ~20 calls vs ~40-60 for
solver-only, but the 0.003 margin is not significant.

**4. Exploitation prevented: yes, decisively.** Surrogate-only's claimed
optimum (surrogate J 0.88-0.92, predicting |S21| ~1.4) verified to J
1.499-1.500 - the WORST designs of the whole campaign (min|S21| 0.0001).
Every one of the 60 hybrid verifications per run was "badly wrong"
(|surrogate J - true J| > 0.15, mean error 0.54), and the hybrid never
reported an unverified design; its verified output matched solver-only.
The safety contract held under a surrogate with zero skill.

**5. Regime:** topology (structural-novelty) on every pool - correct, the
design class is outside the training support in every direction. The
consequence is that per-sample gating degenerates: 100% above p95 means
"verify everything promising", i.e. solver-only with surrogate pre-screening.

**6. Where the surrogate is fooled:** everywhere on this space. The
DeepSets extrapolates the 7-element cascade into unphysical transmission
(|S21| > 1) because no training structure combines two stubs with a series
gap; the optimizer then chases phantom pass-bands. Design QA confirms the
final geometries are legitimate bound-corner designs, not solver artifacts
(max singular value 1.004-1.016, within the frozen 1.1 passivity gate).

**7. Ready for a realistic PCB/package problem?** Not on this evidence - but
the blocker is now precisely located and is not the methodology. Two
protocol changes make the next attempt informative: (a) a feasible task
(optional gap, or a band the gap passes) so methods can separate on
quality; (b) a surrogate trained on samples of the design space itself, so
the hybrid operates in the extrapolation regime where the stack has
demonstrated 0.66 within-OOD ranking and near-oracle budget recovery, rather
than in the fully-OOD regime where it can only act as a verified gate. The
selective stack's guarantees (never trust unverified, rank risk correctly,
detect the regime) all held; what was absent was a surrogate with any skill
to be selective about.
