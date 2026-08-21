# Evaluation protocol

## Task

Predict complex 2-port S-parameters S(f) on a fixed 256-point grid
(0.05-20 GHz for the synthetic data) from design parameters and/or the
element-sequence representation of the structure.

## Splits

All splits are produced by `emsurr.splits.make_splits` from a single seed and
are disjoint by construction; `dataset.check_leakage` verifies no
family+parameter duplicate crosses a train/test boundary (asserted before
every training run).

| Split | Train/val | Test | Question |
|-------|-----------|------|----------|
| `iid` | 70/15% random | 15% random | interpolation difficulty |
| `ood_topology` | all samples from 5 families | **all** samples from 2 held-out families (`via_lc`, `stub_short`) | topology generalization |
| `extrap_er` | permittivity ≤ 85th percentile | permittivity above it | material extrapolation |
| `extrap_len` | total length ≤ 85th percentile | longer structures | geometry extrapolation |

Discipline rules:

- `val` (carved from the training domain) is the only set used for model
  selection/early stopping. Test sets are touched once, at final evaluation.
- No hyperparameter was tuned on any OOD test family.
- Per-family metrics are always reported alongside pooled metrics.
- For the real TUHH data the same machinery applies with families = dataset
  IDs (SI-1..PI-13); a common-grid resampling policy will be needed because
  frequency grids differ across families (documented in
  `docs/data_audit.md`).

## Metrics (`emsurr.metrics`)

- complex MAE and RMSE on S entries
- magnitude error in dB (floor 1e-8 to avoid log blowup)
- phase error in degrees (wrapped difference via angle(pred * conj(true)))
- complex MAE per frequency point
- physical validity of predictions: passivity violations (max singular value
  of S > 1+1e-3, counted per frequency point) and max reciprocity error
- inference latency in ms/sample (CPU)

## Uncertainty protocol (`emsurr.uncertainty`)

Deep ensemble of 5 MLPs (different seeds). Per-sample uncertainty = mean
absolute deviation of member predictions from the ensemble mean.

Reported:

- Spearman rank correlation between uncertainty and true per-sample error
- confidence curve: retained MAE as the most-uncertain fraction is deferred
- solver-fallback experiment: for fallback budgets of 10/20/30%, remaining
  MAE when deferring by uncertainty vs random vs oracle (defer truly-worst),
  plus capture rate (overlap between uncertainty-selected and truly-worst).
