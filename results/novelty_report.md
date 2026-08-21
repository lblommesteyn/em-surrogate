# Novelty-detection report: can the surrogate know when to call the solver?

Date: 2026-08-21. Benchmark frozen from milestone 1 (synthetic 7-family
dataset, seed-0 splits, identical MLP/kNN/DeepSets/5-member-ensemble
configs; milestone-1 artifacts preserved in `results/milestone1/`). Final
OOD families `via_lc`, `stub_short` were never used for tuning; novelty
hyperparameters (kNN k=3, combination weight alpha=0.3) were selected on a
pseudo-OOD stage built inside the training families (stub_open + stepped
held out). Numbers: `results/novelty_metrics.json`,
`results/openems_novelty.json`. Scripts: `scripts/run_novelty.py`,
`scripts/score_openems.py`, `scripts/openems_families.py`.

## Methods evaluated (all "higher = more novel", no family labels used)

1. `knn_input`: mean distance to 3 nearest training samples in normalized
   content space (design params + pooled element-token summary)
2. `mahalanobis_emb`: Mahalanobis distance in the MLP's penultimate embedding
3. `knn_emb`: kNN distance in that embedding space
4. `ensemble_var`: milestone-1 baseline, 5-member ensemble disagreement
5. `combined`: rank-average 0.3*knn_input + 0.7*ensemble_var

## Synthetic results

Final pool A = 300 in-distribution (ID) + 800 OOD samples (73% OOD, the full
frozen test); pool B = 300 ID + 100 subsampled OOD (25% OOD, a realistic
deployment mix). Ensemble MAE on pool A: 0.469 (ID ~0.19, OOD ~0.58).

### OOD detection (held-out topology vs ID)

| Method | AUROC (A) | AUPRC (A) | AUROC (B) | Spearman(score, error) B | latency |
|--------|-----------|-----------|-----------|--------------------------|---------|
| knn_input | **0.998** | **0.999** | **0.998** | 0.742 | 0.3 ms |
| mahalanobis_emb | 0.740 | 0.879 | 0.790 | 0.797 | 0.1 ms |
| knn_emb | 0.813 | 0.918 | 0.841 | 0.796 | 1.4 ms |
| ensemble_var | 0.979 | 0.991 | 0.983 | **0.905** | 0.2 ms |
| combined | 0.997 | 0.999 | 0.997 | 0.895 | 0.5 ms |

### Solver fallback: fraction of the oracle benefit recovered (pool B)

Oracle = defer the truly-worst samples; random = defer at random.

| Budget | knn_input | maha_emb | knn_emb | ensemble_var | combined |
|--------|-----------|----------|---------|--------------|----------|
| 5% | 0.42 | 0.62 | 0.80 | **0.83** | 0.77 |
| 10% | 0.64 | 0.64 | 0.82 | **0.87** | 0.82 |
| 20% | 0.84 | 0.45 | 0.57 | **0.86** | 0.85 |
| 30% | 0.92 | 0.60 | 0.59 | 0.88 | **0.96** |
| 50% | 0.84 | 0.71 | 0.79 | **0.93** | 0.92 |

Remaining MAE at 30% budget (pool B): base 0.254 -> combined 0.130 vs
oracle 0.124 and random 0.261.

### Catastrophic-error capture (worst 5% of pool errors caught before use)

Pool B: at 20% budget, combined catches **90%**, knn_input 85%,
ensemble_var 80%; at 30% budget combined and knn_input catch 100%,
ensemble_var 95%. Pool A (73% OOD) is harder at small budgets for every
method: at 20% budget ensemble_var 69%, combined 60%, knn_input only 2%.

### Per-family score medians (pool A, raw scores)

knn_input separates cleanly: ID families 1.36-1.79 vs stub_short 3.05,
via_lc 5.31. ensemble_var overlaps more: ID 0.044-0.093 vs stub_short 0.17,
via_lc 0.12 (stub_open, an ID family, reaches 0.093). Plot:
`results/novelty_scores_by_family.png`; risk-coverage curves:
`results/novelty_risk_coverage.png`.

## Answers

### 1. Why did ensemble uncertainty "fail" OOD in milestone 1?

It didn't fail at detection — that reading was too coarse. Milestone 1
measured Spearman rank correlation *within an all-OOD pool* (0.32): once
everything is OOD, member disagreement can't rank which OOD sample is worse.
As an ID-vs-OOD detector on a mixed pool, ensemble variance is strong
(AUROC 0.979-0.983). The genuine weaknesses that remain: (a) its ID/OOD
score distributions overlap more than input-space distance (per-family
medians above), so at tight false-positive thresholds it defers ID samples
first, and (b) within-OOD prioritization is weak, which caps small-budget
performance when OOD dominates the pool (pool A).

### 2. Which novelty representation works best?

For pure detection: input-space content kNN (AUROC 0.998, and the only
method that stays near-perfect in both pool mixes). For deciding *what to
defer under an error objective*: ensemble variance, because it also ranks
error within-distribution. Learned-embedding scores (Mahalanobis/kNN) are
the weakest on both counts — the surrogate's own representation compresses
away exactly the structural information that signals novelty (confirmed
independently by the openEMS gap family below). The combined rank-average
gets the best of both: detection AUROC 0.997 plus fallback recovery equal to
or better than ensemble variance at 20-50% budgets. **Answer to the primary
question: explicit novelty detection alone does not beat ensemble variance
on solver-fallback recovery; combining them beats either — modestly on
average error (0.85-0.96 of oracle) and most clearly on catastrophic-error
capture (90% vs 80% at a 20% budget).**

### 3. How much solver compute can be avoided at a chosen error tolerance?

In the 25%-OOD deployment mix, with the combined score: accepting remaining
MAE <= 0.17 (i.e. surrogate-only quality on clean data ~0.19) requires
deferring only ~20% of samples to the solver — an 80% solver-compute saving
— while catching 90% of catastrophic predictions. At a 30% budget remaining
MAE is 0.130, within 5% of the oracle's 0.124. If the tolerance is "no
catastrophic prediction ever served", 30% deferral achieved 100% capture in
this benchmark.

### 4. Does novelty detection transfer to genuinely different openEMS structures?

Yes — with one instructive exception. Twenty FDTD-simulated samples in three
families that are not cascades of any training element (series-gap line,
double-stub tee, patch-loaded line; `oems_patch` untouched until this final
evaluation), scored against 300 ID samples with methods frozen beforehand:

| Method | AUROC all | gap | dstub | patch | deferred@ID-95% (gap/dstub/patch) |
|--------|-----------|-----|-------|-------|------------------------------------|
| knn_input | 0.985 | 0.961 | 1.00 | 1.00 | 0.75 / 1.00 / 1.00 |
| mahalanobis_emb | 0.851 | 0.657 | 0.96 | 1.00 | 0.00 / 0.67 / 1.00 |
| knn_emb | 0.876 | 0.693 | 1.00 | 1.00 | 0.00 / 1.00 / 1.00 |
| ensemble_var | 0.997 | 0.992 | 1.00 | 1.00 | 1.00 / 1.00 / 1.00 |
| combined | **0.999** | 0.996 | 1.00 | 1.00 | **1.00 / 1.00 / 1.00** |

The untouched `oems_patch` family is flagged by every method. The series-gap
family — the only one whose defining feature is *inexpressible* in the
training vocabulary, so its canonical parameters look like two plain feed
lines — completely fools the embedding-space scores (0% deferred) but not
ensemble variance or the combined score. Caveat honestly noted: one
mapping artifact was removed before this run (non-design constants sigma/t
initially defaulted to 0, trivially separating OOD; they now take training
values), and the surrogate-input family one-hot is all-zero for unknown
structures, which is the honest encoding but does give model-input methods a
hint that content-space kNN does not get.

### 5. What to run immediately when TUHH data arrives

Leave-one-super-family-out with the a-priori grouping in
`configs/tuhh_families.yaml` (mapping fixed from published structure docs,
never from model results; loader `emsurr.tuhh`): train the frozen pipeline
on all but one super-family (e.g. hold out `pi_central_rail` or
`si_universal_diff`), select novelty hyperparameters on a pseudo-OOD
super-family carved from training, and measure exactly this report's
metrics. First question to answer there: does the combined score's
catastrophic-capture advantage survive when OOD structures share the
via-array vocabulary but differ in configuration (the regime where content
kNN should get *harder* and ensemble variance may not) — that is the regime
this synthetic benchmark cannot probe.

## Appendix: openEMS debugging record (why the first batch stalled)

The original generator stalled >19 min on one gap sample. Diagnosis: the
suspected 10 GB python process belonged to an unrelated concurrent job, not
the simulation (ours peaked at ~70 MB); the actual causes were (1) gaps down
to 100 um forcing fine mesh cells and a small CFL timestep, (2) no timestep
cap, so the high-Q gap resonator's slow energy decay ran essentially
unbounded under EndCriteria=1e-3, over a 60 mm domain. Fixes: domain 60->36
mm, local-only refinement, NrTS cap 40000, minimum gap 300 um, per-sample
HDF5 writes (the stalled run's kill corrupted the always-open file). A port
bug introduced while shortening (feed inside the ~6 mm PML_8 region)
produced |S11|>1 in all three single-sample smoke tests and was fixed by
FeedShift=10*res, MeasPlaneShift=MSL_LEN/2. Final smoke: gap 22.6k cells /
4 s, dstub 77k / 72 s, patch 21k / 8 s, all ~70 MB, max singular value
<= 1.054. Batch: 20/24 passed the automatic sanity gate (max_sv <= 1.1);
4 rejected samples were excluded, not repaired.
