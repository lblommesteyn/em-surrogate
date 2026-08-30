# Package sample-efficiency report (FINAL)

Date: 2026-08-30. Branch `external-data`. Question: how many solved package
structures turn the failed package surrogate into one useful for selective
optimization - and does acceleration emerge when a predeclared readiness
threshold is crossed?

Protocol as predeclared (`pkg_sample_eff_declaration.md`, commit ebf0afc,
before any budget result): 256 scrambled-Sobol designs solved in sequence
order (nested budgets 16 c 32 c 64 c 128 c 256), one frozen training
recipe per budget (same architecture/encoder/losses/risk machinery),
untouched 24-design validation set, readiness trigger Spearman >= +0.50,
optimization on fresh untouched seeds 5-7 at trigger. Every reported
design openEMS-verified; reported == verified on all runs. (Campaign
interruptions - background-task kills and a disk-full event external to
the project - were survived losslessly via content caching and per-run
checkpoints; logged, no protocol impact.)

## Scaling of deployment readiness (untouched validation set)

| package solves | objective Spearman | J MAE | resp err | retrieval-gap/err | catastrophic rate (|dJ|>0.3) |
|---|---|---|---|---|---|
| 0 (original) | -0.48 | 0.86 | 0.68 | -0.36 | ~1.0 |
| 16 | +0.03 | 0.90 | 0.64 | -0.62 | 1.00 |
| 32 | -0.05 | 0.78 | 0.52 | -0.25 | 1.00 |
| 64 | +0.33 | 0.50 | 0.37 | -0.36 | 0.83 |
| **128 (trigger)** | **+0.61** | 0.26 | 0.25 | -0.12 | 0.12 |
| 256 | +0.69 | 0.13 | 0.16 | **+0.42** | 0.04 |

Monotone learning from 64 onward; the predeclared trigger fired at
**N=128**. The retrieval/risk stack recovers automatically with the
surrogate: gap/error correlation climbs from -0.36 to +0.42 by 256 with no
change to the risk machinery (Q4: yes).

## Optimization at the trigger (frozen N=128 stack, fresh seeds 5-7)

| seed | solver-only @60 | solver-only @24 | repaired hybrid @24 | original hybrid @24 | surrogate-only |
|---|---|---|---|---|---|
| 5 | 0.221 | 0.240 | 0.323 | 0.322 | 0.389 |
| 6 | 0.211 | 0.276 | 0.299 | 0.309 | 0.385 |
| 7 | 0.242 | 0.283 | 0.322 | 0.331 | 0.457 |

**The repair transformed prediction quality but not optimization
outcome.** Badly-wrong verifications collapsed from 20-22/24 (mean |dJ|
~0.54) with the original stack to 1-3/24 (0.08-0.17) with the repaired one
- the surrogate now tells the truth about its candidates. But its
candidates top out at ~0.30: with Spearman 0.61-0.65 and residual MAE
~0.25 on an objective whose decisive range is 0.20-0.33, the surrogate
cannot resolve the region where the real optima live, while solver-driven
DE can (solver-only matches the repaired hybrid's FINAL quality within 8
calls and reaches 0.21-0.24 by 60). Repaired == original hybrid to within
noise on all three seeds.

## Answers

**1. How many solves to learn the missing coupled physics?** 128 crosses
the predeclared "useful ranking" trigger (Spearman 0.61), and by 256 the
surrogate is well-calibrated (MAE 0.13, catastrophic 4%, risk stack
recovered). But optimization-grade skill was NOT reached by 256: by the
now-bracketed boundary (savings at 0.85, none at 0.65-0.69), the trend
(+0.33 -> +0.61 -> +0.69 for 64 -> 128 -> 256) suggests >= 512-1024
solves, unmeasured.

**2. Scaling:** see table - negative/noise through 32, then roughly
+0.3 per doubling to 128, flattening toward 0.69 at 256.

**3. At what validation skill does optimization begin saving calls?**
Above +0.65-0.69, below +0.85 - measured on opposite sides across two real
tasks. **The predeclared 0.50 trigger is hereby falsified as sufficient**:
it correctly identified "useful ranking" but not "optimization-grade
resolution". The revised, still label-cheap criterion the data supports:
require Spearman >= ~0.75 AND validation J-MAE below roughly half the
objective's decisive range before expecting acceleration.

**4. Does the risk stack recover automatically?** Yes - retrieval-gap/error
went -0.36 -> +0.42 across the sweep with zero changes to the machinery.

**5. Total solver cost:** 256 training + 24 validation (reused from the
prior milestone) + 60x3 solver-only baselines + 24x3 repaired-hybrid +
24x3 original-hybrid + finals = ~535 package solves for the whole study;
the deployable path costs 128 (training) + 24 (validation) + 24
(optimization) = 176 solves per first campaign at trigger level.

**6. Amortization:** at the achieved skill level - never: the repaired
hybrid's designs are worse than a 60-call solver-only run, so each
"accelerated" campaign loses quality rather than saving calls. Projected
at via-task-grade skill (>= 0.85, where the hybrid needed 24 calls vs
solver-only's ~49-60 for worse results): break-even ~ training_cost /
(60 - 24) ~ 512/36 ~ **14+ campaigns** if the >= 512-solve estimate holds
- a materially worse proposition than the via task's, because the package
space is harder to learn.

**7. Does this support validate -> decide -> train if necessary ->
selectively optimize?** The STRUCTURE, emphatically: every decision point
worked as designed - the cheap validation correctly diagnosed the broken
surrogate, the trigger fired exactly per protocol, the trigger-gated
optimization ran on untouched seeds, and the honest outcome (trigger
insufficient) was only discoverable because the pipeline was
pre-registered. The PARAMETER needs revision: the decide step must demand
optimization-grade skill (~0.75+ Spearman with MAE below half the decisive
band), not merely positive ranking. With that revision, the recipe stands;
without cheap-enough training data to reach that bar, the correct decision
for this package class is what the stack originally made - use the solver.
