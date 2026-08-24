# Package-channel regime predeclaration

Written 2026-08-24 immediately after the validation benchmark and BEFORE
inspecting any optimization output (the campaign log beyond the validation
line has not been read; optimization runs continue in the background).

Pre-optimization signals (24-design independent validation):
- UNLABELED: pool novelty z = -0.49 (nov), -0.73 (unc); 0.0% beyond the
  train gap-p95 -> the frozen deployment-switch diagnosis is IN-SUPPORT /
  uncertainty regime, which in the prior two campaigns predicted savings.
- LABELED (24 solver calls, allowed by protocol): surrogate objective
  Spearman -0.48, J MAE 0.86, dd-response error 0.68; retrieval-gap vs
  error -0.36, ensemble variance -0.18.

Declared predictions for the frozen-policy campaign:
1. PRIMARY: NO robust solver savings. The surrogate is ordinally
   anti-correlated with the true objective, so multistart pooling will
   preferentially verify bad candidates; hybrid final quality should be
   roughly "best of ~24 near-random verifications", likely WORSE than
   solver-only at 60 calls on most seeds, and the safety contract (only
   verified designs reported) should hold on all seeds.
2. Surrogate-only's claimed optimum will verify poorly.
3. Interpretation declared in advance for Q3/Q6 of the milestone: the via
   and toy campaigns showed savings track the surrogate's ORDINAL skill.
   This task splits the two halves of the pre-optimization diagnosis for
   the first time: unlabeled support signals say "trust", the small labeled
   validation says "no skill" (model-form error of the chain mapping, the
   same failure class as planar MEAS hardware scatter: error invisible to
   input-side support). If prediction 1 holds, the arc's law must be stated
   as: unlabeled novelty identifies OUT-of-support failure; a ~24-call
   labeled validation Spearman is the necessary companion check for
   in-support model-form failure - and it costs less than half of one
   optimization run's solver budget.
