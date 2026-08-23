# Topology-representation report: physics-metric embeddings and what they fix

Date: 2026-08-22. Branch `external-data`. Question: can a structure-aware
embedding make electromagnetically similar designs close and dangerous
unseen topologies far — and does that repair within-OOD error ranking?

Frozen: every prior result; the frozen surrogates are reused (checkpoints),
never retrained. Encoders are small (DeepSets phi 64x2, embedding 32, head
64). No topology-family label or dataset ID is encoded. Training uses only
training structures + their solver responses. The primary variant was
selected by a pre-declared train-side criterion (best worst-family LOFO
pair-Spearman) before any frozen pool was touched; the follow-up
retrieval-disagreement construction was likewise defined a priori and run
once. Reproduction: `src/emsurr/topo_rep.py`, `scripts/run_topo_rep.py` ->
`results/topo_rep_metrics.json`, plus `results/topo_nn_gap.json`,
`results/topo_rep_sqchip.json`.

## Stage 1 — representation quality (train-side)

Six variants: {ordered cascade tokens, unordered set tokens} x {response
prediction, metric learning on |dS| pair distances, both}. The synthetic
structures are 2-port cascades, so token position IS the connectivity graph
(a path); "ordered" is the graph representation, "unordered" the pure
permutation-invariant one.

| Variant | val pair-Spearman | NN response err | LOFO mean | LOFO min |
|---|---|---|---|---|
| **ordered + response** | 0.44 | **0.21** | **0.41** | **0.36** |
| ordered + both | 0.46 | 0.28 | 0.39 | 0.33 |
| ordered + metric | 0.43 | 0.36 | 0.30 | 0.21 |
| set + both | 0.41 | 0.35 | 0.31 | 0.16 |
| set + metric | 0.38 | 0.44 | 0.23 | 0.13 |
| set + resp | 0.28 | 0.29 | 0.26 | 0.11 |
| content features (A) | 0.29 | 0.49 | - | - |

**Q1: yes.** The best embedding's distance tracks response similarity far
better than the engineered features (0.44 vs 0.29 pair-Spearman; nearest-
neighbor retrieval lands 2.3x closer in response space, 0.21 vs 0.49), and
— the key property — the metric HOLDS on unseen training families (LOFO
pair-Spearman 0.36-0.48 on every held-out family). **Q3: connectivity/order
dominates**: every ordered variant beats every unordered one on LOFO
transfer; and plain response prediction transfers the metric better than
explicit metric learning (the metric loss overfits pair structure of the
seen families). Primary (pre-declared): **ordered + response**.

## Stage 2 — frozen OOD evaluation

**As a distance, the new embedding detects but still cannot rank.** Synth
topology OOD: detection AUROC up to 0.997, but within-OOD Spearman is
NEGATIVE for every variant (-0.25 to -0.48; openEMS similar for ordered
variants). Old baselines on the same pool: knn_input -0.15, knn_emb +0.23,
ens_var +0.36. The strong hypothesis from the within-OOD milestone is
therefore **refuted**: better metric quality does not fix distance-based
ranking, because inside a genuinely new family the surrogate's error is not
monotone in distance-to-training — near-support samples can carry the worst
errors (mid-band resonances) while far ones saturate into easy responses.
Distance is the wrong *functional*, however good the metric.

**As a retrieval mechanism, it fixes the ranking.** Define the
retrieval-disagreement signal: |frozen surrogate prediction - true response
of the nearest TRAINING structure in the new embedding| (label-free: uses
training solver outputs only). On the frozen synth OOD pool:

| Signal | within-OOD Spearman | orec@5/10/20/30/50% | b90 |
|---|---|---|---|
| **topo-retrieval gap** | **+0.56** | **0.76/0.75/0.76/0.80/0.85** | 0.37 |
| ensemble variance (prev. best) | +0.36 | -/-/0.74/-/- | 0.30 |
| same gap w/ old features | +0.12 | - | 0.56 |
| new-embedding distance | -0.36 | 0.08@20 | 0.74 |

The same construction with the OLD representation's retrieval scores +0.12,
so the gain is specifically the physics-trained embedding retrieving
functionally relevant neighbors. **Q2/Q4/Q5: the representation does fix
unseen-topology ranking — through retrieval-disagreement, not distance** —
beating every previous signal on the benchmark where all of them failed,
and lifting small-budget oracle recovery (0.76 at 5% vs ~0.0-0.4 for prior
signals). Catastrophic b90 is the one metric where ensemble variance keeps
a slight edge (0.30 vs 0.37).

Hard cases: the new retrieval finds a physically closer neighbor for 69% of
OOD structures; embedding aliasing (close in embedding, far in response) is
1.8%. SQChip (compact metric encoder on param+geometry): detection AUROC
0.90 on the sqchip fold with within-OOD +0.46 — comparable to, not better
than, the existing embedding-kNN (+0.55); batch300/near50 stay noise, as
every signal there has. Planar was skipped: one topology, no structural
vocabulary to embed.

## Answers 6-7

**Remaining failure modes:** (a) openEMS within-OOD ranking stays negative
even for ordered encoders — the token vocabulary cannot express the `gap`
discontinuity, so its retrievals are structurally wrong; missing
*vocabulary coverage*, not missing capacity. (b) Unordered encoders fail
across the board: connectivity is not optional information. (c) SQChip's
heterogeneous generation recipes defeat the 24 coarse geometry statistics;
a richer layout tokenization would be needed. (d) Hardware scatter (MEAS)
is untouched, as expected.

**Q7 — next step:** neither a bigger surrogate nor more metric-learning
machinery. The productive direction is operational: (1) add the
topo-retrieval-gap to the frozen within-OOD ridge as an eleventh signal and
re-freeze (mechanism-matched: it is the first signal with the right sign on
topology shifts); (2) extend the token vocabulary to cover the structures
that currently cannot be expressed (openEMS gap; richer SQChip layout
primitives). Both keep model sizes frozen; both attack the two failure
modes actually observed.
