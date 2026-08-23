# Sim-to-real calibration report: what 55 measured PCBs can and cannot fix

Date: 2026-08-22. Branch `external-data`. Question: how many measured boards
does it take to estimate simulator bias well enough to improve predictions
and flag dangerous sim-to-real failures?

Frozen: the FEA surrogate (rebuilt with the bit-identical frozen recipe,
never updated), all prior benchmarks and risk/switch results. Corrections
are post-hoc on the frozen log10(L) predictions; the measured test partition
of every split is untouched by fitting. Protocol: calibration sizes
{1,2,4,8,16,24,32}, 200 seeded random splits each (50 per active-selection
strategy), mean + bootstrap 2.5/97.5% CIs. Reproduction:
`scripts/run_sim2real.py` -> `results/sim2real_metrics.json`.

## 1. How much sim-to-real error exists before calibration?

Zero-shot on the 55 boards: **5.6% mean / 2.0% median relative error, 24%
at p95**. In log space the residual has mean -0.0001 and sd 0.046: the
error is a heavy-tailed scatter around an essentially unbiased center.

## 2. Global or geometry-dependent?

**Neither - it is board-idiosyncratic.** There is no global bias to remove
(mean log-residual is 4 orders of magnitude smaller than its sd). Geometry
dependence is weak: the largest |Spearman| between any of the 9 design
features and the residual is 0.29 (trace width), and every feature-based
correction *hurts* out-of-sample. Neighboring boards' residuals do not
predict each other (nearest-calibration correction also loses). The
remaining discrepancy behaves like per-board noise - fabrication tolerance
and measurement error - not like a learnable simulator defect.

## 3-4. Boards needed for improvement; does active selection help?

Measured-test mean relative error (200 splits; frozen surrogate 0.055):

| n_cal | mult (global) | linear (features) | ridge_aug | nearest |
|---|---|---|---|---|
| 1 | 0.093 | - | - | 0.093 |
| 4 | 0.071 | 0.076 | 0.068 | 0.068 |
| 8 | 0.067 | 0.088 | 0.070 | 0.065 |
| 16 | 0.063 | 0.087 | 0.072 | 0.063 |
| 24 | 0.061 | 0.082 | 0.070 | 0.062 |
| 32 | 0.058 | 0.077 | 0.068 | 0.059 |

**No correction beats the frozen surrogate at any budget up to 32 boards**
(frozen 0.055, CI [0.027, 0.087] at n=32; best correction 0.058). The
corrections converge *toward* frozen from above as n grows - the signature
of estimating a bias whose true value is zero: every fitted parameter is
pure variance. Raw-L additive correction is actively harmful (0.19-0.39;
L spans decades, so an additive offset is scale-nonsense - kept as a
cautionary row). **8-16 measured boards do NOT materially outperform
zero-shot simulation; nothing up to 32 does.**

Active selection cannot rescue a correction that has nothing to correct.
The apparent wins of novelty/diversity selection at large budgets (e.g.
novelty@32: remaining-board error 0.034) are a test-set composition effect:
deterministic strategies move the hardest boards INTO the calibration set,
so the untouched remainder is easier. That is not calibration skill - but it
IS operationally useful: measuring the highest-novelty / highest-uncertainty
boards first removes the riskiest hardware from the unmeasured pool
(remainder error 0.034-0.055 vs 0.078 for a random-selection remainder).

## 5. Can measured calibration predict dangerous hardware errors?

Only as well as the label-free signals already could. Risk ranking of
measured-test boards (Spearman vs true error / top-10%-error capture at 20%
budget, stable across n_cal 4-32): ensemble variance 0.29-0.32 / 0.45-0.58,
FEA-train novelty 0.29-0.30 / 0.50-0.55, distance-to-calibration 0.20-0.24 /
0.45-0.53, linear-predicted residual 0.13 / 0.39-0.44. **The two
calibration-derived scores are the weakest**: hardware measurements add no
risk-prediction power beyond the frozen ensemble and novelty signals, whose
~0.3 Spearman (vs ~0.5 capture at 2x random) restates the frozen finding
that most of this error is invisible from the input side.

## 6-7. Verdict

**8-16 boards: no.** At this fabrication/measurement noise level, small
calibration sets cannot improve on the frozen surrogate for in-support
hardware; the 2% median error is already at the scatter floor, and the 24%
p95 tail is board-specific, partially flaggable (2x random capture) but not
correctable from other boards' measurements.

**Simulator discrepancy is NOT the dominant obstacle to deployment.** The
FEA-trained surrogate is unbiased against hardware; its in-support hardware
error looks aleatoric at n=55. The dominant obstacles remain, in order:
(a) within-OOD error ranking (fallback recovers 0.5-0.9 of oracle benefit -
the standing gap from the risk-calibration milestone), and (b) the
board-specific error tail, which is a *hardware QA* problem (measure the
flagged boards) rather than a modeling problem. The practical recipe this
study supports: deploy the frozen surrogate uncorrected, spend the
measurement budget on the highest-novelty/uncertainty boards, and treat
measured deviations as per-board acceptance tests, not as calibration data.

Caveats: single hardware family, 55 boards, one target quantity (L); a
systematic bias could still exist below the ~0.05-log-sd detection floor;
conclusions apply to in-support designs (MEAS geometries sit inside the
CORE design region by construction).
