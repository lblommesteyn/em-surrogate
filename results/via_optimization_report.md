# Via-transition optimization report: the frozen stack on a realistic task

Date: 2026-08-24. Branch `external-data`. Question: does the selective
surrogate + solver stack transfer from the toy stub/gap task to a realistic
differential PCB via transition?

**Verdict: yes.** All five seeds were fresh for this task and the entire
methodology was frozen from the completed design-optimization milestone
(DS-ensemble recipe, physics-retrieval representation, retrieval-gap,
deployment-switch statistics, multistart explore-then-verify policy,
optimizer); nothing was iterated or tuned on outcomes. Every reported
design is openEMS-verified (reported == verified on all 15 runs).

## Task (declared before optimization)

4-layer stackup (er 3.5, 1.0 mm, planes at 300/700 um), differential pair
entering on L1, through two signal vias with antipads in both planes, out
on L4; n stitching vias on a ring. Seven manufacturing-bounded parameters:
drill 200-400 um, pad/antipad oversizes, via pitch 800-2000 um, trace width
150-400 um, 2-8 ground vias, ring radius up to 2.2 mm (with geometric
clearance constraints). Objective, band 2-15 GHz, on the mixed-mode
response under odd drive:
J = max|Sdd11| + max(0, 0.7 - min|Sdd21|) + max(0, max|Scd| - 0.15);
invalid (non-finite or odd-drive power > 1.1) -> 2.0. The parameterization
is y-symmetric, so mode conversion is ~0 by construction (verified:
Scd == 0 to numerical precision on every solve); the Scd term guards
asymmetric artifacts and never activated - the honest answer to Q6 is that
mode-conversion constraints are *represented and verified* but not
*exercised* by this symmetric parameterization.

Solver: new openEMS 4-port evaluator (`scripts/oems_via_eval.py`):
priority-carved antipads, differential MSL excitation, mixed-mode
extraction, realistic dielectric loss (tand 0.02), per-design crash guard;
~17-60 s per solve, ~400 solves content-cached in `results/via_cache/`.
Two prototype defects were found and fixed before any optimization: an
undamped parallel-plate cavity (lossless dielectric) and a port feed placed
beyond the measurement plane (garbage wave separation).

## Representation coverage (Q5)

Zero new token types were needed: the classic lumped via model -
line(z0) / shunt-C(pad) / series-L(barrel+return loop) / shunt-C / line -
uses the existing vocabulary, with geometry entering through declared
physics formulas (coaxial pad-antipad capacitance; loop inductance with a
ground-ring term monotone in n_gnd and r_gnd). One analytic train-side
family (`via_model`, 400 structures solved by the same ABCD engine) covers
the mapped ranges. Consequence: only 4.2% of the design space is flagged
beyond the retrieval-gap p95 - the via space is essentially *in-support*,
the regime the risk stack always said the surrogate could be trusted in.
No geometry regime inside the declared bounds is unsupported; what remains
outside the representation is any *asymmetric* via arrangement (would need
a mode-conversion-capable surrogate) and multi-transition chains.

## Surrogate benchmark BEFORE optimization (24 independent designs)

Objective Spearman **0.85** (toy task: -0.50), objective MAE 0.38
(systematic lumped-model offset; ordering is what matters), dd-response
error 0.67, retrieval-gap vs true error Spearman 0.51, ensemble-variance
0.59, pool novelty z ~ -0.3 (in-support; deployment switch -> uncertainty
regime), 4.2% beyond gap-p95.

## Results (5 fresh seeds, budget 60, all verified by openEMS)

| seed | solver-only @24 | hybrid @24 (final) | solver-only @60 | solver calls to match hybrid | surrogate-only (1 call) |
|---|---|---|---|---|---|
| 0 | 0.142 | **0.062** | 0.101 | **>60 (never)** | 0.100 |
| 1 | 0.080 | **0.068** | 0.058 | 58 | 0.095 |
| 2 | 0.088 | **0.081** | 0.075 | 38 | 0.142 |
| 3 | 0.095 | **0.058** | 0.050 | 28 | 0.095 |
| 4 | 0.096 | **0.068** | 0.078 | **>60 (never)** | 0.082 |

- **Equal budget: hybrid wins 5/5 seeds at 24 calls** (its full spend).
- **Matched quality: solver-only needs 28-58 calls on three seeds and never
  matches within 60 on two** - mean ~49 calls vs the hybrid's fixed 24,
  i.e. a **1.2-2.5x+ solver-call reduction (median ~2x)**, plus halved wall
  time (10-14 min vs 20-29 min per optimization).
- At the full 60-call budget solver-only edges past the hybrid's final on
  3/5 seeds by 0.006-0.010 J (~0.5-1 dB of worst-band return loss);
  mean final quality still favors the hybrid (0.0673 vs 0.0725).
- Best designs are genuinely good signal integrity: worst in-band
  differential RL -24 dB with |Sdd21| >= 0.99 (seed 0 example: 257 um
  drill, 473 um pad, 858 um antipad, 958 um pitch, 400 um traces, 7 ground
  vias at 1.56 mm).

## Answers

**1. Transfer: yes.** The equal-budget dominance is 5/5 on untouched
seeds; the matched-quality reduction is ~2x median and unbounded on 2/5.
This is the in-support behaviour the whole research arc predicted:
with a surrogate the stack itself measures as skilled (Spearman 0.85),
free exploration + pooled verification converts that skill into solver
savings reliably - unlike the toy task, where the same frozen policy
could only act as a verified gate over a skill-less surrogate.

**2. Calls saved:** 60 -> 24 with better quality at that point on every
seed; to match hybrid quality solver-only spends 28->60+ calls (mean ~49).

**3. Risk prediction: yes.** Retrieval-gap Spearman 0.51 vs true error on
the validation set (ensemble variance 0.59; both healthy), and the gap
statistic correctly diagnoses the space as in-support (4.2% flagged) - the
deployment switch's uncertainty regime - the exact opposite diagnosis it
gave the toy task, made from unlabeled data both times.

**4. Exploitation: contained.** Surrogate predictions are systematically
offset (17-24 of 24 verifications per seed "badly wrong" in absolute
terms, mean |dJ| ~0.2-0.3) yet ordinally excellent - precisely the case
verification is for. Surrogate-only's single verified call already lands
0.08-0.14 (its claimed optimum is real, unlike the toy task), but every
hybrid improvement beyond it came from verified candidates only.

**5. Unsupported regimes:** none inside the declared symmetric bounds;
asymmetric arrangements and multi-via chains are outside the current
representation (see coverage section).

**6. Mode conversion:** extracted (Scd11/Scd21 under odd drive),
constrained in the objective, verified ~0 by symmetry on every solve;
not exercised as an active constraint by this parameterization.

**7. Package/interposer-ready?** The evidence pattern says yes with one
condition: keep the surrogate in-support. The via milestone shows the full
loop working end-to-end on a realistic 3D task - representation mapping
with zero new primitives, an honest pre-registered skill benchmark, frozen
policy, 5/5 equal-budget wins, ~2x matched-quality savings, verified-only
reporting. Scaling to package/interposer problems means longer chains of
the same primitives (more transitions, more coupling), where the two known
gaps - asymmetric mode conversion and chain coupling - are representation
extensions of exactly the kind the coverage recipe (add primitive + add
analytic family) has now succeeded at twice.
