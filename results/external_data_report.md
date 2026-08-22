# External-data expansion report

Date: 2026-08-22. Branch `external-data`. Question: does the surrogate +
novelty/uncertainty fallback methodology generalize beyond our synthetic
world to independent simulated and physically measured EM data?

The frozen synthetic/openEMS benchmark (milestones 1-3) was not modified or
retuned. All external runs use the milestone-2 configuration verbatim:
MLP 256x3, 200 epochs, lr 1e-3, 5-member deep ensemble, kNN novelty k=3,
combined score alpha=0.3, fallback budgets 5/10/20/30/50%. Nothing was
tuned on any measured subset. Storage: `results/storage_manifest.md`
(0.91 GB external data, cap 5 GB).

Data provenance tiers used below:

1. **Analytic synthetic** — our frozen 7-family generator (`data/processed/synth.h5`)
2. **openEMS simulation** — frozen milestone-2 transfer test (unchanged, for context)
3. **External FEA simulation** — Dataset A CORE/OOD (ANSYS Maxwell 3D, independent group)
4. **Physical measurements** — Dataset A MEAS (55 fabricated PCBs), Dataset B
   (reverberation chamber VNA), Dataset C measured briefcase files
5. **Superconducting-chip EM** — Dataset D (HFSS/Q3D-derived, layout-grounded)

Runners: `scripts/run_planar_windings.py`, `run_tlines.py`, `run_p370.py`,
`run_sqchip.py`. Loaders: `src/emsurr/planar_windings.py`, `tlines.py`,
`sqchip.py`; shared tabular adaptation `src/emsurr/external_tab.py`.
Numbers: `results/{planar_windings,tlines,p370,sqchip}_metrics.json`.

---

## Dataset A: planar windings (primary benchmark)

Zenodo 21762502: 10,110 CORE ANSYS FEA samples, 1,124 OOD edge-case samples,
55 fabricated + measured PCB prototypes. Provided splits preserved. Task:
9 geometry parameters -> self-inductance L (regressed as log10 L).

### Accuracy (mean relative error on L; median in parentheses)

| Model | CORE test | OOD | MEAS (physical) |
|---|---|---|---|
| linear | 15.0% (12.8%) | 70.6% (45.2%) | 21.0% (6.4%) |
| kNN (k=3) | 5.8% (4.1%) | 43.0% (38.0%) | 24.4% (7.8%) |
| MLP ensemble | **0.41% (0.29%)** | **19.3% (14.3%)** | **5.6% (2.0%)** |

- **OOD degradation persists on independent FEA data**: 47x error growth
  from CORE to OOD for the ensemble, the same qualitative failure as in the
  frozen synthetic benchmark.
- **The surrogate survives physical measurements**: 5.6% mean / 2.0% median
  on the 55 fabricated PCBs with zero tuning on them. MEAS geometries lie
  inside the CORE design region, so this is an in-support hardware test;
  the residual error is FEA-vs-lab mismatch plus fabrication tolerance,
  not extrapolation failure.

### Novelty and fallback (pool: CORE test + OOD)

Full pool (1,011 ID + 1,124 OOD) / deployment mix (1,011 ID + 337 OOD):

| Method | AUROC full | AUROC mix | Spearman(score, err) full | oracle recovery @30% mix |
|---|---|---|---|---|
| knn_input | 0.995 | 0.994 | 0.758 | 0.98 |
| mahalanobis_emb | 0.982 | 0.981 | 0.766 | 0.96 |
| knn_emb | **0.995** | **0.994** | **0.806** | **0.98** |
| ensemble_var | 0.966 | 0.963 | 0.723 | 0.86 |
| combined (frozen alpha=0.3) | 0.985 | 0.983 | 0.750 | 0.94 |

Fallback at 5/10/20/30/50% budget (fraction of oracle benefit recovered,
mix pool): knn_emb 0.70/0.61/0.92/0.98/0.96; combined 0.28/0.38/0.79/0.94/0.93;
ensemble_var 0.27/0.44/0.70/0.86/0.91.

- Novelty detection transfers: OOD is near-perfectly separable, dominated by
  the input-space and embedding kNN scores.
- **The frozen combined weighting does NOT outperform here.** On the synthetic
  benchmark ensemble variance was the strongest single score and combined
  (0.3 knn + 0.7 var) matched the best of both. On this external FEA task
  ensemble variance is the weakest score, so the frozen weighting drags the
  combination below pure kNN-embedding at every budget. Answer to "does
  novelty + uncertainty outperform either alone": **not universally — the
  optimal mix is dataset-dependent, and a fixed alpha inherited from one
  benchmark can hurt.**
- MEAS vs CORE novelty: input-space scores cannot distinguish measurements
  from FEA test samples (AUROC 0.47 — correct behavior, the geometries are
  in-support). Only embedding Mahalanobis mildly flags them (0.77),
  suggesting the embedding is sensitive to something beyond raw geometry.
  Error targeting within MEAS is weak (oracle recovery ~0.5-0.7); with only
  55 points and small errors this is not load-bearing.

## Dataset B: measured transmission-line networks (reverberation chamber)

Zenodo 167116: 14 physical single-wire network configurations, each a 3-port
S-matrix (two antennas + network) at 72 stirrer angles x 5001 frequencies
(0.2-1 GHz). Loader (`emsurr/tlines.py`) keeps configuration, stirrer, and
frequency identity; angles/frequencies are treated as replicates of one
structure, never as independent samples. No expanded arrays on disk (the
697 MB zip was deleted after extraction; .mat files are sliced lazily).

- **Physics of the measurements**: all values finite; zero passivity
  violations (tol 1e-3) in any sampled angle of any configuration;
  reciprocity error max 0.040 across configurations (typical VNA-level
  asymmetry). The measured data passes the same sanity gates we impose on
  synthetic data.
- **Prediction**: not attempted. The frozen input representation (synthetic
  family design parameters + element tokens) cannot encode a wire network in
  a stirred chamber. That a whole class of real measurements is simply
  *unrepresentable* is the first failure mode of moving beyond our synthetic
  world — before accuracy is even in question.
- **Confidence system**: response-space novelty (band-limited |S11|,|S21|,|S22|
  curves, kNN vs 2,000 synthetic references) flags every measurement as far
  outside training support: AUROC 1.00 vs the synthetic ID split; minimum
  measured distance 20.1 vs ID 95th percentile 0.95. The confidence system
  would correctly refuse all of it.

## Dataset C: IEEE P370 briefcase test cases

14 Touchstone files (2- and 4-port, up to ~50 GHz) from the P370 standard's
open repository: fixtures, 2x-thrus, de-embedded DUTs, fixture models.
Downloaded as a path-filtered subtree (23 MB), not the full repo. Small set:
sanity benchmark only, no ML claims.

- Ingestion: all 14 files parse via scikit-rf; all finite.
- **Measured vs synthesized separate cleanly on physics checks**: measured
  files show reciprocity error ~2.4e-3 to 6.6e-3 (real VNA asymmetry);
  synthesized/model files are reciprocal to 1e-12 or better. Passivity:
  measured files are passive (<=1 violating frequency); the AFR de-embedded
  outputs are the exception — `Test8_afterAFR_42p5.s4p` violates passivity at
  326 frequencies and `S2_M9_AFR_2.s2p` at 7. De-embedding algorithms, not
  raw measurements, are what break passivity.
- Novelty: all 14 files score above the synthetic ID 95th percentile
  (AUROC 0.989); fixture *models* score highest (118-124), farther out than
  the measurements they model (7-41).

## Dataset D: SQChip-EM (superconducting-chip EM)

github.com/Secbrain/SQChip-EM (shallow clone, 221 MB). Public 1q/2q releases
carry JSON+CSV only; the `examples/poor_2q` set is the one subset with both
GDSII layouts and HFSS/Q3D-derived EM targets — 892 GDS+JSON pairs, 714
complete records across ~30 generation families. Targets: qubit f01 (GHz),
dispersive chi (MHz), readout fr (GHz), two of each per chip.

Representation comparison (frozen MLP ensemble + linear baseline):
parameter vector (160 dims from JSON layout) vs 24 cheap geometry statistics
computed from the GDS vs both. See `results/sqchip_metrics.json`.

### Random-split IID (MAE: f01 GHz / chi MHz / fr GHz)

| Representation | linear | MLP ensemble | Spearman(unc, err) |
|---|---|---|---|
| param vector | 0.140 / 2.16 / 0.273 | 0.124 / 2.05 / 0.274 | 0.57 |
| geometry features | 0.151 / 2.35 / 0.278 | 0.151 / 2.28 / 0.286 | 0.63 |
| both | 0.149 / 2.47 / 0.328 | **0.120 / 2.04 / 0.274** | **0.71** |

The MLP barely beats linear regression on any representation — with 714
heterogeneous records spanning ~30 generation recipes, the dataset is too
small and too mixed for the surrogate accuracy story that Dataset A
supports. Prediction-wise, geometry features add nothing over the parameter
vector. They do, however, sharpen uncertainty-error correlation (0.57 ->
0.71 when concatenated).

### Held-out generation families (MAE f01 / chi / fr, novelty AUROC)

| Held-out family (n) | param | geom | both |
|---|---|---|---|
| batch300 (300) | 0.094 / 0.46 / 0.051, **AUROC 0.39** | 0.093 / 0.69 / 0.062, AUROC 0.69 | 0.089 / 0.46 / 0.071, AUROC 0.47 |
| batchnear50 (60) | 0.064 / 0.31 / 0.060, AUROC 0.80 | 0.061 / 0.34 / 0.061, AUROC 0.86 | 0.061 / 0.27 / 0.059, AUROC 0.76 |
| sqchip (290) | 0.332 / 3.90 / 0.740, AUROC 0.99 | 0.602 / 10.47 / 0.917, AUROC 0.85 | 0.286 / 4.59 / 0.765, AUROC 0.97 |

- The two representations see *different* novelty: holding out `batch300`,
  the parameter vector scores it as familiar (AUROC 0.39, worse than
  chance) while geometry statistics flag it at 0.69; holding out `sqchip`
  (a genuinely different generation recipe, with 3-6x higher held-out
  error), the parameter vector is near-perfect (0.99) and geometry weaker
  (0.85). Layout geometry carries structural-novelty information the
  parameter vector misses, and vice versa — complementary, not superior.
- A geometry-aware encoder is **not justified** on this data: geometry
  features do not improve prediction, the record pool is small (714) and
  recipe-skewed, and the schema mixes exploratory runs with refined
  batches. Documented and stopped, per plan, rather than forcing a
  benchmark.

---

## Answers to the framing questions

1. **Does novelty + uncertainty outperform either alone?** Not universally.
   On external FEA (Dataset A) ensemble variance — the strong half of the
   frozen combination — collapses to the weakest score, and the frozen
   alpha=0.3 combination underperforms pure embedding-kNN at every fallback
   budget. The combination is only as good as its per-dataset weighting;
   treating alpha as transferable is the part of the frozen methodology
   that did not survive contact with external data.
2. **Does OOD degradation persist on independent datasets?** Yes, and at
   similar magnitude: 47x error growth CORE->OOD on independent ANSYS FEA
   (vs ~3x error ratio on the frozen synthetic benchmark's held-out
   topologies, and ~3x OOD collapse previously observed on TUHH). The
   novelty scores detect it near-perfectly (AUROC 0.99+).
3. **Does the method work on physical measurements?** Where the input
   representation covers the hardware, yes: 5.6% mean / 2.0% median error on
   55 fabricated PCBs, no tuning. Where it does not (reverberation-chamber
   networks), the confidence system correctly identifies every measurement
   as outside training support (AUROC 1.00) instead of failing silently.
4. **Which representation best captures structural novelty?** In input space,
   plain kNN remains the strongest cheap detector (0.995 AUROC on A);
   embedding-space kNN matches it while correlating better with actual error
   (Spearman 0.81), making it the best single fallback score observed.
   On SQChip, parameter-vector and GDS-geometry novelty are complementary:
   each flags held-out families the other misses (0.39 vs 0.69 on one
   family, 0.99 vs 0.85 on another); no single representation wins.
5. **What breaks moving from our synthetic world to external physics?**
   (a) representation coverage — entire measurement classes (Dataset B) are
   unencodable before accuracy matters; (b) the transferability of the
   combined-score weighting; (c) physics guarantees: measured data is
   near-reciprocal but algorithmic post-processing (P370 AFR de-embedding)
   injects passivity violations that a surrogate trained on clean synthetic
   data never sees.
6. **Strongest next research direction?** Dataset A is the result to build
   on: a frozen pipeline transferred mechanically to an independent FEA
   dataset, achieved 0.4% IID error, detected OOD at 0.995 AUROC, and
   validated on physical hardware at 2% median error. The open problem it
   exposes is *adaptive score combination* — choosing or learning the
   novelty/uncertainty mix without OOD labels — which is exactly the piece
   the frozen alpha got wrong. Second candidate: geometry-aware novelty for
   layout-grounded EM (Dataset D) if the representation gap there proves
   real.
