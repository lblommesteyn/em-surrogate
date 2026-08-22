# Storage manifest: external EM datasets

Date: 2026-08-22. Hard cap for all new data + artifacts: 5 GB. Target: <3 GB.
Free disk before downloads: 19 GB (C:, 98% full). Free after: 18 GB.

| Dataset | Source | Remote size | Local path | Local size |
|---|---|---|---|---|
| A. Planar windings (MLRPW) | Zenodo 21762502, `dataset_planar_windings_v1.1.xlsx` | 0.9 MB | `data/external/planar_windings/` | 1 MB |
| B. Reverberation-chamber TL S-params | Zenodo 167116, `rawdata.zip` (697 MB) + `MATLAB.zip` | 697 MB | `data/external/tlines/` (zip deleted after extraction; .mat files barely compress) | 667 MB |
| C. IEEE P370 briefcase test cases | opensource.ieee.org elec-char/ieee-370, path-filtered archive of `IEEE370_Appendix_briefcase_testcases` (full repo NOT cloned) | 7.8 MB zip | `data/external/p370/` | 23 MB |
| D. SQChip-EM | github.com/Secbrain/SQChip-EM, `--depth 1` clone (repo API size 61 MB packed) | ~61 MB packed | `data/external/sqchip/SQChip-EM/` | 221 MB checked out |

**Total external data: ~0.91 GB.** Generated artifacts (processed caches, checkpoints,
figures) budgeted at <0.5 GB; running total tracked here.

Decisions taken to stay under cap:

- Dataset B: kept the extracted per-configuration `.mat` supermatrices and deleted
  the 697 MB zip (compression ratio ~1.00, keeping both would double footprint).
  No expanded/duplicated arrays are written; loaders slice the supermatrix lazily.
- Dataset C: downloaded only the briefcase-testcases subtree via GitLab path-filtered
  archive instead of cloning the full ieee-370 repo (which contains large
  round-robin measurement campaigns).
- Dataset D: shallow clone; public data is 1q (532 records) + 2q (35 records)
  JSON/CSV plus 1065 example GDS files, comfortably small.
- CircuitNet: not downloaded, not touched (per project constraint).

Update log:
- 2026-08-22: initial downloads, 0.91 GB external data, 18 GB disk free.
- 2026-08-22: benchmark artifacts written (`results/planar_windings_metrics.json`,
  `tlines_metrics.json`, `p370_metrics.json`, `sqchip_metrics.json`,
  `external_data_report.md`): <1 MB combined; no model checkpoints persisted.
  Running total ~0.91 GB, well under the 3 GB target.
