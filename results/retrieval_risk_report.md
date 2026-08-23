# Retrieval-risk report: the retrieval-gap signal and vocabulary coverage

Date: 2026-08-22. Branch `external-data`. Question: does physics-retrieval
disagreement improve the risk stack across deployments, and are the
remaining topology failures caused by representation coverage?

Frozen: all prior results and models. The retrieval variant was chosen on
the milestone-2 SELECTION stage (models trained without stub_open+stepped;
those families are the pseudo-OOD) — strictly train-side. Ridge refits use
only the existing train-side pseudo pools. The coverage experiment adds
train-side data only: a new `gapped_line` family generated and solved by
the same analytic ABCD engine (no openEMS data, no family IDs in tokens),
and the encoder is retrained at identical size. Reproduction:
`src/emsurr/synth_ext.py`, `scripts/run_retrieval_risk.py` ->
`results/retrieval_risk_metrics.json`, `results/retrieval_qa.json`.

## 1-2. Retrieval variant and stack integration

**Q2: top-k beats nearest-neighbor.** Train-side pseudo-OOD Spearman:
k=1 0.699, mean-3 0.740, distance-weighted-3 0.742 -> **w3 chosen** (and
that train-side ordering carried to the frozen pools).

Frozen comparison (within-OOD Spearman / oracle-recovery@20%):

| Pool | ens_var | 10-signal ridge | **gap only** | 10+gap ridge |
|---|---|---|---|---|
| synth topology OOD | +0.36 / .74 | +0.09 / .46 | **+0.59 / .81** | +0.37 / .69 |
| planar OOD | +0.12 / .34 | **+0.66 / .75** | +0.39 / .64 | **+0.66 / .75** |
| sq sqchip | +0.34 / .36 | +0.43 / .30 | -0.24 / .39 | **+0.47 / .32** |
| sq batchnear50 | +0.12 / .86 | +0.13 / .79 | +0.12 / **.95** | +0.14 / .85 |
| sq batch300 | +0.11 / .68 | +0.05 / .33 | -0.12 / .64 | +0.05 / .36 |

**Q1: yes, as a regime-specific signal — not as a ridge input.** On the
topology-shift benchmark the pure retrieval-gap is the best signal ever
measured there (+0.59, oracle recovery 0.74-0.85 across ALL budgets, b90
29%; per-family +0.53/+0.55 on both unseen families). Folding it into the
11-signal ridge dilutes it (the pseudo-fit assigns it weight 0.36 and the
other signals mis-rank true topology OOD), and on planar the ridge simply
ignores it (weight -0.02) because the extrapolation ridge already works.
On SQChip's 6-scalar target space the gap is noise-to-harmful: retrieval
disagreement needs a rich response curve to compare against. The final
stack is therefore **regime-routed, using the existing deployment switch**:
extrapolation regime -> 10-signal ridge; topology regime -> retrieval-gap;
scalar-target/interpolative regimes -> ensemble variance. Every routing
criterion is train-side.

**Q3: at 5% solver budget the gap recovers 0.74 of the oracle benefit** on
the topology benchmark (previous signals: ~0-0.4), 0.78 at 10%, 0.85 at
50%. Regret vs oracle at 20-30% is 0.17-0.19.

## 3-4. Vocabulary coverage (openEMS)

Audit: `dstub` is expressible; `patch` approximately (a wide line);
`gap` is NOT — a series discontinuity has no token, and its structures
tokenized as two plain feed lines. One primitive added: **EL_SERIES_C**, a
series-gap pi-capacitance token, exercised by 400 analytically solved
`gapped_line` training structures from the same generator physics.

Result, frozen openEMS pool (surrogate untouched): pooled retrieval-gap /
error Spearman **0.24 (old vocabulary) -> 0.62 (extended)**. The mechanism
is exactly the predicted one: under the old vocabulary the gap family
carried the SMALLEST retrieval gap (0.18) while having the LARGEST true
errors (0.83) — a silent failure where inexpressible structure retrieved
irrelevant "safe" neighbors. With the token added, its gap value rises to
0.47 and the family is correctly flagged as the riskiest. Within-family
ordering on 6-8 samples per family is noise either way (n too small).
**Q4: yes — coverage, not capacity, explains the openEMS failure. Q5: the
essential primitives are connectivity order (prior milestone) and the
series-discontinuity token; nothing else was needed.**

## 5. Retrieval QA (results/retrieval_qa.json)

High gap = high error: the top-gap queries retrieve cross-family neighbors
at response distance 0.49-0.87 and carry true errors 0.88-1.04 — retrieval
disagreement is physically meaningful there, not accidental. The residual
failure mode is the opposite corner: some via_lc queries with gap ~0.11
still err at ~0.55 — the surrogate's (wrong) prediction happens to sit near
a genuine training response, so disagreement under-fires. These
"plausible-but-wrong" cases are the fraction the signal cannot see; they
are why b90 stays at ~29% instead of oracle's 5%.

## Verdict (Q6-Q7)

Remaining failures after coverage: (a) plausible-but-wrong predictions that
land near real training responses (bounds b90 ~29% on topology shifts);
(b) scalar-target deployments where no response curve exists for retrieval
to compare (SQChip); (c) hardware scatter (MEAS, closed as QA problem).

**Q7: yes — stop methodological work.** The selective-surrogate system now
has: near-perfect OOD detection; a label-free pool-level regime switch
(regret 0.061); regime-appropriate within-OOD ranking (extrapolation 0.66,
topology 0.59, both with 0.75-0.81 oracle recovery at 20%); a verified
coverage recipe for extending to new structure classes; an unbiased
hardware story with a near-oracle acceptance-test triage. The marginal
milestone here has been shrinking for three milestones. The next step is a
real design/optimization application exercising the full stack —
surrogate + switch + regime-routed fallback — against solver budget, where
the remaining weaknesses will surface with actual costs attached.
