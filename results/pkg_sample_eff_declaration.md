# Package sample-efficiency predeclaration

Written 2026-08-25 while the 256-design Sobol pool is solving; no budget
training or evaluation has been run.

TRAINING RECIPE (single, applied at every budget N in {16,32,64,128,256},
nested prefixes of the scrambled-Sobol sequence, seed 5):
- Retrain the frozen-architecture DS ensemble (5 members, DeepSets,
  N_TYPES=6, MAX_EL=12, hidden/epochs/lr per frozen config) FROM SCRATCH on
  [synthetic train families + gapped_line + via_model + chain_model
   + N package solves replicated x8] with the frozen 85/15 split recipe.
- Retrain the frozen-recipe encoder (ordered, response objective) on the
  same pool; the retrieval corpus (embeddings + true responses) is the same
  pool, so the risk machinery is UNCHANGED in form and simply sees the new
  data.
- Package solve targets: the measured dd-mode 2x2 (Scd slots filled by the
  declared placeholder s01:=Sdd21, s11:=Sdd11) interpolated onto the
  synthetic 256-point grid, flat-extended below 1 GHz.

READINESS THRESHOLD (from prior experiments only): validation objective
Spearman >= +0.50 on the untouched 24-design validation set (rationale:
+0.85 gave 5/5 savings, -0.48/-0.50 gave none; the within-OOD milestone
showed ranking Spearman ~0.55+ already delivers near-oracle low-budget
capture). The SMALLEST budget crossing 0.50 is frozen and taken to
optimization. If no budget crosses, that is the reported answer.

CATASTROPHIC ERROR RATE: fraction of validation designs with
|surrogate_J - true_J| > 0.3.

OPTIMIZATION AT TRIGGER: fresh untouched seeds 5, 6, 7 (package campaign
used 0-4): solver-only (budget 60), original failed hybrid (frozen v2c
multistart, original pkg stack), repaired hybrid (same policy, repaired
stack). All finals openEMS-verified. No methodology change anywhere.
