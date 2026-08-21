# TUHH real-data evaluation: pre-registered plan (frozen 2026-08-21)

This file is the a-priori registration for milestone 3, written and committed
BEFORE any TUHH data was available. Nothing below may change after TUHH
results are first inspected; deviations must be logged here with reasons.

## Status

Blocked on data: the TUHH SI/PI-Database is form-gated (manual request at
tet.tuhh.de/en/si-pi-database/, procedure in docs/data_audit.md). The full
pipeline below is implemented and tested end-to-end against a mock fixture
in the real directory layout (tests/test_tuhh.py plus a full rehearsal of
both scripts). When the archives are unpacked to `data/raw/tuhh/<ID>/`, the
entire evaluation is:

    python scripts/ingest_tuhh.py
    python scripts/run_tuhh.py            # add --full to lift the 500/family cap

## Frozen methodology

- Models, ensemble, novelty scorers, and metric definitions are byte-identical
  to milestones 1/2 (`configs/baseline.yaml`, `src/emsurr/*`).
- Splits: IID; leave-one-super-family-out (loso) for every super-family in
  `configs/tuhh_families.yaml` (mapping fixed from published structure docs);
  leave-one-dataset-family-out (lodo); a global parameter-extrapolation split
  only if some raw parameter name exists in every family (auto-detected,
  else recorded invalid).
- Novelty/fallback study runs on the two a-priori primary loso holdouts
  named in results/novelty_report.md before any data existed:
  `pi_central_rail` and `si_universal_diff` (all others via
  `--all-loso-novelty`). Hyperparameters are selected per the milestone-2
  protocol on a pseudo-OOD stage: the two alphabetically-first TRAINING
  super-families. The synthetic-frozen hyperparameters (k=3, alpha=0.3) are
  additionally evaluated as `*_synthfrozen` scorers.
- Compute cap: `max_per_family: 500` seeded subsample (CPU-only training);
  dropped counts are logged in tuhh_metrics.json. `--full` disables it.

## Mechanical input adaptations (documented, not research changes)

1. Union family-prefixed parameter schema; absent columns 0.0 (neutralized
   in-distribution by the Normalizer sd-floor).
2. Complex S linearly interpolated onto a common grid (128 pts over the
   intersection band, frozen at ingest in results/tuhh_ingest.json).
3. 2x2 sub-block of ports (0, 1) from N-port networks; 1-port samples
   excluded at ingest with reason `single_port`.
4. No element tokens exist for real layouts: DeepSets is recorded
   `not_applicable`, and content features reduce to their parameter part.

## Ingest verification (results/tuhh_manifest.csv)

One row per parameter.csv entry with include flag and reason
(missing_touchstone / parse_error / non_finite / single_port /
no_band_overlap / duplicate). Passivity (max singular value) and
reciprocity are recorded per sample but never used for exclusion.

## Report

results/tuhh_report.md will answer, comparing synthetic, openEMS, and TUHH:
persistence of the ~3x OOD degradation; whether input-space novelty stays
near-perfect when families share via/interconnect vocabulary; whether
ensemble uncertainty still ranks dangerous predictions; whether the combined
fallback beats either score alone; solver compute avoidable at a chosen
tolerance; failure modes unique to real data.
