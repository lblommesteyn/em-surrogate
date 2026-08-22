# Deployment-switch report: label-free pool-level fallback selection

Date: 2026-08-22. Branch `external-data`. Question: can an unlabeled
deployment batch's own statistics choose between the two frozen fallback
layers (plain ensemble uncertainty vs pseudo-OOD-calibrated combination)
without any target errors or OOD labels?

Frozen and untouched: all benchmarks, surrogate models (nothing retrained;
everything runs on the cached per-sample signal tables in
`results/riskcal/*.npz`), and the risk-calibration results including each
deployment's pseudo-OOD-calibrated (signal, alpha) choice. No target OOD or
MEAS label enters switch fitting. Reproduction: `src/emsurr/deploy_switch.py`,
`scripts/run_deploy_switch.py` -> `results/deployment_switch_metrics.json`.

## Setup

**Pool descriptors** (unlabeled queries + training statistics only):
novelty z-shift median/q90/q99, fraction above IID-val p95/p99 novelty,
ensemble-disagreement z-shift, embedding-novelty z-shift, input-vs-embedding
disagreement gap, novelty-z IQR.

**Switches fitted on 630 train-side pseudo-deployments** — mixes of each
deployment's pseudo-OOD calibration pool (LOFO families / parameter-
extrapolation subsets) at OOD fractions 2-100% and sizes 30-300, whose known
errors label which strategy wins. Two switches, both frozen before any
benchmark pool was touched:

- **threshold**: choose pseudo-calibrated when pool median novelty z > 0.65
- **logistic**: 2 features (median novelty z, median ensemble z), weights
  (+1.04, +0.91, bias +0.15), margin-weighted fit

## Frozen transfer evaluation (primary metric: selection regret in mean oracle-recovery@{20,30}%)

| Pool | med z-nov | med z-unc | better | unc | pseudo | fixed a=0.3 | thr pick | log pick | thr regret | log regret |
|---|---|---|---|---|---|---|---|---|---|---|
| synth OOD | 6.4 | 3.2 | unc | .73 | .70 | .66 | pseudo | pseudo | +.033 | +.033 |
| openEMS | 2.0 | 0.7 | unc | .87 | .75 | .84 | pseudo | **unc** | +.118 | .000 |
| planar OOD | 4.0 | 1.5 | pseudo | .42 | .60 | .37 | pseudo | pseudo | .000 | .000 |
| planar MEAS | 0.0 | 0.0 | unc | .58 | .54 | .58 | unc | unc | .000 | .000 |
| sq batch300 | -0.1 | -0.2 | unc | .67 | .33 | .39 | unc | unc | .000 | .000 |
| sq near50 | 2.4 | 0.0 | pseudo | .90 | .93 | .90 | pseudo | unc | .000 | +.028 |
| sq sqchip | >>100 | 74 | pseudo | .45 | .50 | .55 | pseudo | pseudo | .000 | .000 |
| **total regret** | | | | **.256** | **.530** | **.603** | | | **.151** | **.061** |

(The "always-X" columns' totals are those policies' own selection regrets.)

## Answers

**1. Can unlabeled pool statistics select the right regime?** Yes. The
2-feature logistic switch picks the better of the two layers on 5 of 7
frozen pools and its two misses are near-ties (regret .033 and .028 where
the strategies score within .03 of each other). Total regret 0.061 —
4x better than the best fixed policy (always-uncertainty, 0.256) and
10x better than the frozen alpha=0.3 baseline (0.603).

**2. Remaining regret vs knowing the best strategy?** 0.061 total across 7
pools (logistic), concentrated entirely in near-tie pools; on every pool
with a material gap between the layers the switch chooses correctly.
Remaining gap to the *oracle deferral* itself is unchanged from the
risk-calibration milestone (the switch selects between existing layers, it
does not improve their internal ranking).

**3. Most predictive statistic?** Median novelty z-shift is the primary
axis, but it is NOT sufficient: the threshold switch fails on openEMS
(z=2.0, structurally novel yet uncertainty-easy) at a cost of 0.118. The
fix is the second feature: when the ensemble's own disagreement has also
shifted (med z-unc high relative to z-nov), the ensemble sees the shift and
can be trusted; when novelty shifts but the ensemble stays asleep
(near50: z-nov 2.4, z-unc 0.0), structural calibration is needed. The
useful statistic is the (novelty shift, ensemble shift) PAIR.

**4. Sensitivity to OOD fraction and batch size?** Composition stress
sweeps (fractions 1-100%, sizes 10-300, 50 draws each): selection is
reliable from ~100 queries where the strategy margin is material (batch300
.92-.98 accuracy, sqchip .73-.89, rising with size); at n=10 the decision
is unstable everywhere (accuracy .24-.92). Mean stress regret stays at
.002-.085 on four of five deployments across all compositions; the synthetic
deployment is the weak case (stress regret .14-.20; its sub-pools favor
uncertainty at compositions where median-z still screams novelty). A switch
trained across mixed fractions (2-100%) transfers across fractions without
per-fraction recalibration; no fraction-specific pattern in the regret.

**5. Better catastrophic capture than plain uncertainty?** Where it
matters, yes: on planar OOD the switch selects the pseudo layer and cuts
the 90%-capture budget from 53% of pool (uncertainty) to 25%. On pools
where uncertainty is the right layer the switch keeps its capture (it picks
uncertainty). The one capture hazard — threshold-switch picking pseudo on
openEMS would have degraded b90 from 7% to 27% — is exactly the case the
logistic's second feature repairs.

**6. Hard switch vs continuous weighting?** A hard 1D threshold is NOT
sufficient (openEMS failure, 0.118). The 2-feature logistic hard switch is:
its residual regret (.061) sits entirely in near-ties where any blend also
gains nothing. Continuous pool-level weighting is not currently justified by
the data; revisit only if intermediate-shift deployments appear where both
layers are simultaneously partially right.

## Failure analysis (nothing averaged away)

- **Large shift, easy OOD** (openEMS z-nov 2.0; synthetic stress mixes):
  novelty says "new", errors say "fine", uncertainty ranks them well.
  Threshold switch fails here; logistic survives openEMS via the z-unc
  feature but still overpicks pseudo on synthetic sub-pools (stress regret
  .14) — synthetic OOD is simply well-covered by ensemble disagreement, and
  a z-nov of 6+ overwhelms the logistic too. Cost is bounded (.033 at full
  pool) because the layers are close there.
- **Small shift, large errors** (planar MEAS: z 0.0, errors 14x IID): both
  switches "correctly" pick uncertainty, but neither layer sees these
  errors (sim-to-lab mismatch, frozen finding). The switch cannot fix a
  failure both its options share.
- **Uncertainty wins despite structural novelty** (sq batch300 was the
  risk-cal example; here its pool z is -0.1, so no conflict at pool level —
  the earlier per-sample confusion dissolves in pool statistics).
- **Pseudo wins despite weak shift** (sq near50: z-nov 2.4 but z-unc 0.0
  and frac_p95 only 0.03): the ensemble is asleep while geometry moved —
  the exact signature where structural calibration pays. The logistic
  misses this one (.028); the 1D threshold catches it.

**7. Remaining obstacle to a deployable hybrid system:** not regime
selection (solved to within near-tie regret, needing ~100 unlabeled
queries), and not OOD detection. Two things remain: (a) within-OOD error
*ranking* — even the correctly-chosen layer recovers only ~0.5-0.9 of the
oracle benefit at 20-30% budgets, and 90% catastrophic capture still costs
3-5x the oracle budget; (b) in-support failure modes invisible to every
input-side signal (planar MEAS sim-to-lab error), which no selection among
input-side layers can address and which needs a small labeled target sample
(explicitly out of scope here).
