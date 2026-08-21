# Baseline report: learned EM surrogate for PCB interconnects

Date: 2026-08-20. Data: **synthetic stand-in** (7 topology families of
cascaded analytic transmission-line structures, 400 samples each, 2-port
complex S-parameters, 0.05-20 GHz, 256 points). The target TU Hamburg SI/PI
data is form-gated (see `docs/data_audit.md`); nothing here is a claim about
that dataset. All numbers from `results/metrics.json`
(`python scripts/run_baselines.py`, seed 0).

Sanity: 2800/2800 samples pass all checks (no NaN, grid-consistent, passive,
reciprocal, no duplicates); leakage check asserted before every training run.

## Headline table (complex MAE / magnitude MAE in dB / phase MAE in deg)

| Split | kNN | MLP | DeepSets |
|-------|-----|-----|----------|
| IID | 0.432 / 6.09 / 53.5 | **0.205 / 3.15 / 26.4** | 0.226 / 3.55 / 27.5 |
| OOD-topology | **0.581** / 7.87 / 72.1 | 0.622 / **6.20** / 70.8 | 0.670 / 5.83 / 74.2 |
| Extrap-permittivity | 0.467 / 6.40 / 60.6 | **0.240 / 3.57 / 31.6** | 0.257 / 3.84 / 32.2 |
| Extrap-length | 0.544 / 5.29 / 69.3 | **0.456 / 4.27 / 47.5** | 0.504 / 4.43 / 51.3 |

Inference latency: 0.02-0.04 ms/sample for the neural models, ~0.25 ms for
kNN (CPU). All orders of magnitude below any solver.

## Answers to the report questions

### 1. How easy is interpolation?

Moderately easy but not solved. The MLP halves kNN error on IID (0.205 vs
0.432 complex MAE) and reaches ~3 dB magnitude error. Resonant families are
hardest even in-distribution (IID per-family cMAE: plain lines 0.10-0.12,
stubs 0.26-0.29, via_lc 0.33): sharp spectral features under a pointwise MSE
objective remain the bottleneck, not parameter coverage.

### 2. How badly do models degrade on unseen topology families?

Severely. MLP error triples (0.205 → 0.622); DeepSets likewise (0.226 →
0.670). Both learned models fall to, or below, the kNN baseline (0.581) —
on unseen topologies the learned representations currently add nothing over
memory-based lookup. Held-out families: stub_short cMAE 0.680, via_lc 0.564.
A caution from this run: an early version showed a 1e10x "collapse" that was
purely a feature-normalization artifact (constant-zero training columns);
the honest degradation is ~3x, still decisive.

Errors are also unphysical: ~45% of predicted frequency points on IID and
~20% on OOD exceed passivity (max singular value > 1+1e-3), and predictions
carry reciprocity error up to O(0.1). Nothing in these baselines enforces
physics; a structurally passive/reciprocal parameterization is an obvious
lever.

### 3. Which representation generalizes best?

None wins yet. The element-sequence DeepSets was expected to transfer across
families but did not beat the flat MLP on any split, including OOD. Reasons
observed: held-out families contain element *types* absent from training
(short-stub and lumped L/C tokens), so the token encoder itself is OOD; and
mean/max pooling discards element order beyond a scalar position feature.
Conclusion: representation is the open problem. Candidates for the next
iteration: sequence models over element cascades with type dropout /
element-type augmentation at train time, or operator-style models that
predict frequency responses from per-element transfer characteristics.

### 4. Is uncertainty useful for deciding when to call a real solver?

Partially — good in-distribution, weak where it matters most.
Deep ensemble (5 MLPs), per-sample disagreement as confidence:

| Split | Spearman(unc, err) | Base MAE | Remaining MAE after deferring 30% (unc / random / oracle) |
|-------|--------------------|----------|------------------------------------------|
| IID | 0.70 | 0.182 | 0.152 / 0.181 / 0.128 |
| OOD-topology | 0.32 | 0.598 | 0.567 / 0.595 / 0.537 |

On IID, uncertainty-based fallback recovers ~56% of the gap to the oracle at
a 30% solver budget. On OOD, rank correlation drops to 0.32 and deferring
30% removes only ~5% of error — because *every* OOD sample is roughly
equally wrong, per-sample ranking barely helps. Implication: the fallback
trigger for topology-OOD should be a domain/novelty signal (e.g. distance to
training families or a token-coverage check), not per-sample ensemble
variance. That is a concrete, testable next-step hypothesis.

### 5. Strongest next experiment

Get the real TUHH data (manual access request, see `docs/data_audit.md`) and
rerun this exact protocol with families = dataset IDs (train on a subset of
SI-1..SI-7, test on held-out SI families), adding: (a) a
passivity/reciprocity-preserving output parameterization, (b) an OOD
detector (feature-space distance) as the fallback trigger alongside ensemble
variance, and (c) the element/structure-aware model rebuilt so unseen
configurations decompose into seen primitives (vias, cavities, planes),
which the TUHH universal via-array families (SI-5..SI-7, varying layer count
and array size per sample) support directly.

## openEMS smoke test (Phase 6)

Full loop verified on Windows: geometry → openEMS v0.0.36 FDTD → S-parameters
→ canonical dataset sample (`scripts/openems_smoke.py`, local Python 3.11
venv under `tools/`). Six structures (3 plain microstrip lines, 3 notch
filters), <30 s each, results in `results/openems_smoke.h5`. Behavior is
physical: plain lines |S21| ≈ 0.998; notch filters attenuate mid-band with
stub length (0.69 → 0.22 → 0.09). This is the path to generating controlled
full-wave training/eval data later; no large campaign was launched.

## Plots

- `results/err_vs_freq_<split>.png`: error vs frequency per model
- `results/uncertainty_<split>.png`: error-vs-confidence scatter and
  coverage-vs-error curve
