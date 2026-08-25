# Package sample-efficiency report (IN PROGRESS)

Status 2026-08-25: the predeclared protocol is committed
(`results/pkg_sample_eff_declaration.md`, commit ebf0afc) and the campaign
is partially complete. This file is updated as budgets finish; the
interim state below is honest and reproducible at every step.

## Protocol (predeclared before any budget result)

256 scrambled-Sobol package designs (seed 5) solved in sequence order so
budgets 16 c 32 c 64 c 128 c 256 are nested. Single training recipe:
retrain the frozen-architecture DS ensemble + encoder from scratch per
budget on [all prior analytic families + N package solves x8 replication];
risk machinery unchanged. Readiness trigger: objective Spearman >= +0.50
on the untouched 24-design validation set (from prior regimes: +0.85 gave
5/5 savings, -0.48/-0.50 gave none). At trigger: frozen v2c hybrid vs
solver-only vs original failed hybrid on fresh untouched seeds 5-7, all
openEMS-verified.

## Scaling results so far

| package solves | validation Spearman | J MAE | resp err | gap/err Spearman | catastrophic rate |
|---|---|---|---|---|---|
| 0 (original) | -0.48 | 0.86 | 0.68 | -0.36 | (baseline) |
| 16 | +0.03 | 0.90 | 0.64 | -0.20 | 1.00 |
| 32 | pending | | | | |
| 64 | pending | | | | |
| 128 | pending | | | | |
| 256 | pending | | | | |

Early read: 16 solves move ordinal skill from strongly negative to ~zero -
the data is teaching the right direction but is far from the trigger.

## Campaign interruptions (logged)

The host machine is under external disk pressure (C: fell from ~19 GB to
<1 GB free over three days from consumers outside this project). The pool
solver now carries a 300 MB disk guard and immediate scratch cleanup, and
all solves are content-cached, so interruptions lose at most one batch.
Background tasks were killed twice by session restarts/external control;
the campaign resumes idempotently.

## Remaining protocol

~230 pool solves (~6 h), budget sweeps 32-256 (~2 h CPU), then the
trigger decision and (if triggered) the fresh-seed optimization
comparison (~8 h). The seven pre-registered questions are answered when
those complete; nothing in the protocol changes en route.
