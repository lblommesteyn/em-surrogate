# Within-OOD risk report: ranking error severity inside a novel pool

Date: 2026-08-22. Branch `external-data`. Question: can train-side signals
predict which already-novel structures produce the largest surrogate errors?

Frozen: all surrogates (synth final models reloaded from checkpoints;
tabular ensembles rebuilt with the bit-identical frozen recipes), all prior
benchmark/risk/switch/sim2real results. No target-OOD error touches signal
design, model fitting, or method selection. Reproduction:
`src/emsurr/risk_signals.py`, `scripts/riskcal_extract_signals2.py`
-> `results/riskcal/sig2_*.npz`, `scripts/run_within_ood.py`
-> `results/within_ood_metrics.json`.

## Setup

Ten signals per sample (frozen models + training data only): ensemble
variance and member range, input/embedding kNN distances, embedding
Mahalanobis, perturbation sensitivity (8 draws at 1% of per-feature train
std), surrogate-vs-kNN-predictor gap, neighborhood target variance,
input-Jacobian norm, and input-vs-embedding rank disagreement.

Risk model: rank-ridge (11 weights) fitted per deployment on the
**pseudo-OOD rows only** of the train-side calibration pools (synthetic:
true leave-one-training-family-out; planar: a-priori parameter
extrapolation; SQChip: largest-family holdout), frozen, then evaluated on
the frozen benchmarks. openEMS is excluded (20 structures on a different
frequency grid cannot pass the perturbation/Jacobian machinery unchanged).
Planar MEAS is reported separately (hardware scatter, frozen finding).

## Frozen evaluation (within-OOD Spearman / top-10% capture / pool b90 / pool orec@20)

| Pool | rank-ridge | ens_var | knn_emb |
|---|---|---|---|
| planar OOD | **+0.66 / 0.64 / 0.13 / 0.75** | +0.12 / 0.00 / 0.53 / 0.34 | +0.47 / 0.61 / 0.24 / 0.55 |
| synth OOD | +0.09 / 0.06 / 0.56 / 0.46 | **+0.36 / 0.26 / 0.30 / 0.74** | +0.23 / 0.20 / 0.35 / 0.62 |
| sq sqchip | +0.43 / 0.17 / 0.87 / 0.30 | +0.34 / 0.24 / 0.42 / 0.36 | **+0.55 / 0.28 / 0.68 / 0.34** |
| sq batch300 | +0.05 | +0.11 | +0.09 (all ~noise; held-out family is easier than ID) |
| sq near50 | +0.13 | +0.12 | -0.06 (60 samples; noise) |
| planar MEAS | +0.32 | +0.36 | +0.30 (hardware scatter; nothing input-side helps) |

## Answers

**1. What predicts error magnitude after OOD is detected?** It depends on
the shift mechanism, and the dependence is now characterized. Under
*parameter extrapolation* (planar), embedding-kNN distance, the
surrogate-vs-kNN-predictor gap, and neighborhood target variance carry
strong within-OOD signal (residual Spearman given novelty +0.44, +0.56,
-0.67), and the learned combination reaches +0.66 within-OOD Spearman —
a 5x improvement over ensemble variance (+0.12) and the largest single
advance on the standing planar gap: 90% catastrophic capture now costs 13%
of the pool (was 24% with the best frozen signal, 53% with uncertainty;
oracle 5%), and oracle-recovery@20% rises 0.60 -> 0.75. Under *topology
novelty* (synthetic families), no distance-type signal survives: within the
truly-new families the learned ranker goes negative (per-family Spearman
-0.18, -0.29) and plain ensemble variance (+0.36) remains the only signal
with the right sign.

**2. Can train-side pseudo-OOD teach within-OOD ranking?** Yes when the
pseudo mechanism matches the deployment mechanism (extrapolation ->
extrapolation: planar), no when it does not (LOFO topology -> new topology:
synth, where pseudo-CV looked healthy at 0.40 yet transfer was negative).
Crucially there is a **train-side tell**: refit the ridge leaving each
pseudo-family out — on synth its transfer Spearman across held-out
pseudo-families is unstable (+0.37, +0.16, -0.17, -0.12, +0.07), while
planar's single-mechanism pseudo pool shows stable CV (0.75). Deployment
rule, fully label-free: use the learned ranker only where its
pseudo-family transfer is stable; otherwise fall back to ensemble variance.

**3. Most information beyond ensemble variance?** Embedding-kNN distance,
everywhere it helps at all (pseudo-CV: planar 0.63->0.75, synth 0.10->0.40,
sqchip 0.03->0.06), followed by the kNN-predictor gap (planar 0.73) and
neighborhood target variance (synth 0.40). Perturbation sensitivity and
Jacobian norm add nothing on any deployment (and flip sign between pools);
local smoothness of the frozen MLP is not a failure indicator here.

**4. Distance to oracle?** Planar OOD: orec@20 0.75, b90 13% vs oracle 5% —
the gap roughly halved. Synth: unchanged (0.74 via uncertainty). SQChip:
ranking remains weak for everything (best +0.55) — with 6 heterogeneous
targets and recipe-skewed families, within-OOD error there is close to
unrankable from input-side signals.

**5. Budget for 90% catastrophic capture?** With the per-deployment best
label-free ranker: planar 13%, synth 30%, sq near50 21%, sq batch300 14%,
sq sqchip 42%, MEAS 79-99% (not capturable). Oracle is ~5% everywhere.

**6. Fundamentally invisible failure modes?** (a) In-support hardware
scatter (MEAS): every signal sits at Spearman ~0.3, capture near random —
confirmed invisible. (b) "Confident-and-wrong" samples: 1.6-6.4% of OOD
rows are below-median uncertainty yet top-decile error; on planar the
learned ridge ranks them high anyway (via knn_pred_gap/knn_emb), on synth
nothing does. (c) Within genuinely-new topology families, all distance
signals lose ordering meaning — only disagreement retains sign.

**7. Representation, surrogate quality, or uncertainty estimation?**
Representation, primarily. Where the input representation is metrically
meaningful for the shift (continuous parameter extrapolation), label-free
ranking gets close to oracle. Where the shift is a new structure the
representation was never built to embed (topology families, mixed-recipe
chips), distances stop ordering error, and no uncertainty machinery on top
of the frozen surrogate recovers it. The next structural improvement must
come from representations whose distances track functional change across
topologies — not from better error estimators on the current one.

## Hardware-QA side result (planar MEAS, triage only, no correction)

Acceptance-testing boards by input-novelty first: measuring the top 5% most
novel boards removes 33% of the total measured-test error mass from the
unmeasured remainder (remaining mean 0.037 vs 0.056 random, oracle 0.036) —
**near-oracle at the 5% budget**. At 20-50% budgets triage stays ahead of
random (0.036/0.030/0.025 vs 0.056) but falls behind oracle (0.021/0.016/
0.010). Practical: a handful of high-novelty boards is the right physical
QA shortlist; deeper sweeps yield diminishing returns.
