# GNN reproduction-gap diagnostics (Milestone 10, continued investigation)

**Date started:** 2026-08-13
**Status:** CLOSED (2026-08-13) — reopened at explicit user request with a
7-part controlled-diagnostic protocol, after Milestone 10's original
investigation was closed with the gap unresolved
(`notes/deviations_from_paper.md` §1.4); all 7 diagnostics run, gap remains
unresolved, refrozen per explicit instruction. See "CLOSING STATUS" below.

**Scope guardrails (unchanged from the approved plan):** no edits to
`src/polyllm/models/gnn.py`, `train_gnn.py`, `evaluate_gnn.py`, or
`graph/build_graph.py`; no overwrite of `outputs/polyllm/gnn/` (the official
result). All new code lives in `src/polyllm/diagnostics/`; all new outputs
live under `outputs/polyllm/gnn_diagnostics/`. No diagnostic run here is a
new official result unless explicitly stated otherwise.

Recap of the unresolved gap this reopens: test AUC=0.4113 (paper 0.9228,
worse than random), AUPRC=0.5297 (paper 0.8944). Prior investigation found
the model learns a real magnitude/popularity signal (embedding-norm-only
AUC=0.82) but no angular/relational signal (cosine-only AUC=0.51) at the
*final* trained checkpoint, without identifying why.

**CLOSING STATUS (2026-08-13, after Diagnostic 7):** GNN performance
reproduction unresolved despite architecture, data, and RNG fidelity.
Software-environment version fidelity (PyTorch 2.4.1/PyTorch Geometric
2.6.1, matching the paper's stated Sec 3.1 versions) produced a partial,
non-trivial improvement (AUC 0.41→0.60) but did not reach the paper's
reported performance (0.9228) — the version gap narrows the gap, it does
not close it. This branch of investigation is now frozen: no further
tuning, ablation, or environment work without a new, specific hypothesis
and explicit request. See [[project-ddi-nlp]] / [[gnn-authors-source-verified]]
memory for the full multi-session history.

---

## Diagnostic 1 — node/edge/embedding alignment audit

**Script:** `src/polyllm/diagnostics/verify_node_alignment.py`
**Result:** `outputs/polyllm/gnn_diagnostics/node_alignment_audit.json`

20 randomly sampled positive edges (fixed seed 20260813) from the full
graph, each checked on 9 independent conditions: pair-index row existence,
pair-embedding-row == node-ID, bitwise match between the graph's stored
`pdrugs.x` and the source `chemberta_pair_embeddings.npy` row, label-mapping
row existence, seffect-embedding-row == node-ID, seffect label-index ==
node-ID, side-effect ID cross-match between the BERT index and label
mapping, **independent SHA-256 re-hash of `side_effect_name` matching the
hash recorded at BERT-embedding-generation time**, bitwise match between the
graph's stored `seffect.x` and the source `bert_side_effect_embeddings.npy`
row, and a cross-check that the sampled edge is genuinely a positive entry
in `polyllm_labels.npy`.

**All 20/20 sampled edges passed all 9 checks with zero failures.**

**Interpretation:** rules out a silent node/edge/embedding misalignment
(e.g. a reordering between Milestone 1's label mapping and Milestone 10b's
BERT embedding generation, or between Milestone 5's pair embeddings and the
graph's node ordering) as a contributing cause. Node IDs, edge endpoints,
embedding rows, and semantic identities are all consistent with each other.

---

## Diagnostic 3 — layer order verification

**Script:** `src/polyllm/diagnostics/verify_layer_order.py`
**Result:** `outputs/polyllm/gnn_diagnostics/layer_order_audit.json`

Static source reading (`src/polyllm/models/gnn.py:96-109`) plus a runtime
forward-hook trace on a freshly-initialized `GNNEncoder`.

**All checks passed:**
- `named_children()` contains exactly `conv1/bn1/dropout1/conv2/bn2/dropout2/conv3/lin1` — no `bn3`/`bn4`/stray dropout modules.
- Recorded hook call order == `[conv1, bn1, dropout1, conv2, bn2, dropout2, conv3, lin1]`.
- `leaky_relu` confirmed applied exactly between `bn1`→`dropout1` and `bn2`→`dropout2` (dropout's hooked input == `F.leaky_relu(bn output)` elementwise, model in `eval()` mode so dropout is an identity and directly comparable).
- `conv3`'s output feeds `lin1`'s input directly, bit-for-bit — no BN/activation/dropout in between.
- The model's final return value equals `lin1`'s output directly — nothing after it.

**Interpretation:** the implemented layer order is exactly
`conv1→bn1→leaky_relu→dropout(0.8)→conv2→bn2→leaky_relu→dropout(0.8)→conv3→lin1`,
confirmed both statically and at runtime, with no BN/activation/dropout
after `conv3`. Matches the authors' `GNN.py` exactly (their declared but
unused `bn3`/`bn4`/`dropout_rate` remain confirmed dead code, correctly
omitted). Rules out a layer-order transcription bug.

---

## Diagnostic 2 (part 1) & 6 — RNG/reseeding mechanism check

**Script:** `src/polyllm/diagnostics/rng_reseed_comparison.py`
**Result:** `outputs/polyllm/gnn_diagnostics/rng_comparison.json`

Two checks against a small synthetic hetero graph (no real data needed),
comparing this repo's actual behavior (seed once, no reseeding inside the
loop) against a literal inline reproduction of the authors' per-forward
reseed (`set_seed(12)` full reseed + `torch.manual_seed(42)` before negative
sampling) — reproduced only inline in the diagnostic script, never imported
into production code.

**Check 1 (encoder dropout determinism):**
| Regime | Repeated forward calls identical? | Max abs diff |
|---|---|---|
| repo (seed once) | **No** | 8.88 |
| authors (reseed every call) | **Yes** | 0.0 |

Confirms: the authors' per-forward reseed makes dropout masks **identical
across every forward call within a run**, defeating dropout's regularizing
purpose. This repo's seed-once approach retains normal per-call dropout
stochasticity, as intended.

**Check 2 (negative-sampling determinism) — an unexpected, more interesting result:**
| Regime | Repeated sampling identical? |
|---|---|
| repo (seed once) | No |
| **authors (`torch.manual_seed(42)` before each call)** | **No** |
| control: `random.seed(42)` alone before each call | **Yes** |

At first glance this looks like the authors' reseed *doesn't* have the
expected effect on negative sampling — but reading PyG's actual source
(`torch_geometric/utils/_negative_sampling.py::sample()`) explains why:
```python
return torch.tensor(random.sample(range(population), k), device=device)
```
**PyG's `negative_sampling()` draws its candidate pool via Python's builtin
`random.sample()`, not torch's RNG at all.** Empirically confirmed above:
`random.seed(42)` alone makes it perfectly reproducible; `torch.manual_seed`
alone (what the authors actually call) does not move the needle at all.

**Interpretation:** the authors' `torch.manual_seed(42)` call, placed
immediately before negative sampling in their code, is very likely a
**no-op for its apparent intended purpose** — a second, independently
discovered RNG-handling quirk in the authors' own code, distinct from (but
alongside) the dropout-reseeding issue in Check 1. Note, however, that the
authors' *broader* `set_seed(12)` call (which the negative-sampling call
follows) DOES include a `random.seed(12)` component — so in the authors'
actual code, negative sampling is very likely made deterministic by the
earlier `set_seed(12)` call, not by the later, ineffective
`torch.manual_seed(42)` call that appears (from the code) to be intended for
that purpose. Net effect on the authors' actual training: dropout AND
negative sampling are both effectively deterministic and repeated
identically on every single forward pass of every batch of every epoch.

---

## Diagnostic 4 — per-epoch representation diagnostics

**Script:** `src/polyllm/diagnostics/representation_diagnostics.py`
**Result:** `outputs/polyllm/gnn_diagnostics/representation_by_epoch.csv`, `decomposition_by_epoch.csv`, `epoch_checkpoints/epoch_{0..9}.pt`

Ran the **unchanged official recipe** end to end (`DEFAULT_CONFIG` from
`train_gnn.py`, same graph split, same seed), saving a checkpoint every
epoch (including epoch 0 = initialization) and evaluating each on a fixed
20,000-positive + 20,000-negative sample of the test split. Early stopping
triggered after epoch 9 (patience=2, best val_loss at epoch 7 = 0.6678).

**IMPORTANT METHODOLOGY CAVEAT (read before comparing to the official 0.41):**
for speed, this diagnostic scores each epoch via a single **full-graph
(non-mini-batched)** forward pass — encoding using `test_data`'s complete
message-passing edge set directly — instead of the official
`evaluate_gnn.py` protocol, which mini-batches through `LinkNeighborLoader`
with `num_neighbors=[20,10]` (capping each node's sampled neighborhood).
Full-graph propagation lets every node aggregate its ENTIRE true neighborhood
with no cap, which the prior investigation already flagged as pushing
activations far outside the mini-batched regime (activation std≈208,610 in
a full-graph pass). **So the absolute AUC numbers below (e.g. dot AUC≈0.15)
are not directly comparable to the official 0.41** — but the qualitative
mechanism they reveal is new and, based on the internal consistency described
below, very likely the same mechanism operating (at different magnitude) in
the mini-batched official evaluation too.

| Epoch | dot AUC | norm AUC | cos AUC | mean cos (pos) | mean cos (neg) | val_loss |
|---|---|---|---|---|---|---|
| 0 (init) | 0.845 | 0.845 | 0.460 | +0.061 | +0.061 | — |
| 1 | 0.846 | 0.846 | 0.693 | +0.057 | +0.057 | 19.15 |
| 2 | **0.154** | 0.846 | 0.702 | **−0.160** | **−0.161** | 0.800 |
| 3 | 0.154 | 0.846 | 0.722 | −0.076 | −0.076 | 0.712 |
| 7 (best) | 0.154 | 0.846 | 0.737 | −0.046 | −0.046 | 0.668 |
| 9 | 0.154 | 0.846 | 0.724 | −0.025 | −0.026 | 0.686 |

**Two sharp new findings, both more precise than anything found in the
original Milestone 10 investigation:**

1. **The magnitude/degree signal is present at random initialization,
   before any training (`norm AUC=0.845` at epoch 0).** It is not learned —
   it is an architectural property of unnormalized `GraphConv` sum-aggregation
   interacting with this graph's heavy-tailed degree distribution (high-degree
   "hub" nodes sum more neighbor terms → larger output norm, and hub nodes
   are disproportionately true positives simply by having more true edges —
   the same mechanism behind the earlier-found "degree-only baseline gets
   AUC=0.787" result). Training barely moves `norm AUC` at all (0.845→0.846
   across all 10 epochs) — it is essentially a fixed, graph-structural
   quantity that the optimizer neither improves nor destroys.

2. **The dot-product inversion happens abruptly between epoch 1 and epoch
   2**, exactly when mean cosine similarity flips from positive (+0.057,
   both classes) to negative (−0.16, both classes) — and stays negative and
   nearly identical for positive vs. negative edges for the rest of training.
   Once cosine is a near-uniform negative multiplier with no separating
   information between classes, `dot = norm × cos` becomes almost exactly
   `−1 × norm`: **dot AUC (0.154) ≈ 1 − norm AUC (0.845)** at every epoch
   from 2 onward, to 3 decimal places. This is a precise, mechanistic,
   reproducible relationship, not a rough description — training is
   actively driving the embedding geometry into a regime where the
   otherwise-good magnitude signal gets sign-flipped, every single epoch,
   coincident with (not preceding) the AUC collapse.

Also notable: `val_loss` plateaus at 0.668–0.686 from epoch 7 onward — close
to `ln(2)=0.693`, the loss of predicting `p≈0.5` for every edge. Loss keeps
"improving" per early-stopping's bookkeeping while the ranking (AUC) is
catastrophically inverted, consistent with the model settling into a
low-but-uninformative-loss regime (near-constant output near 0.5) rather
than learning genuine discriminative structure via the sign-sensitive dot
product.

## Diagnostic 5 — semantic-projection vs. node-ID-embedding decomposition

**Script:** `src/polyllm/diagnostics/representation_diagnostics.py` (same run as Diagnostic 4)
**Result:** `outputs/polyllm/gnn_diagnostics/decomposition_by_epoch.csv`

| Epoch | full dot | semantic-only dot | id-only dot | semantic-only cos | id-only cos |
|---|---|---|---|---|---|
| 0 (init) | 0.845 | 0.846 | 0.844 | 0.614 | 0.268 |
| 1 | 0.846 | **0.846** | **0.161** | 0.686 | 0.496 |
| 2 | 0.154 | **0.154** | 0.157 | 0.693 | 0.253 |
| 9 | 0.154 | 0.154 | 0.157 | 0.729 | 0.427 |

**Findings:**
- At initialization, **both** the semantic (ChemBERTa/BERT-content)
  projection and the freshly-initialized, content-free node-ID `nn.Embedding`
  independently carry the same strong magnitude/degree signal (dot AUC
  ≈0.845 for both branches alone). This confirms the signal is purely
  structural (a `GraphConv`-on-degree-skewed-graph artifact) — it has
  nothing to do with drug/side-effect semantic content, since it appears
  identically for a branch that starts as pure random per-node noise.
- The **id-only** branch inverts one full epoch *before* the semantic-only
  branch (epoch 1 vs. epoch 2) — the node-identity embedding pathway is the
  first to be driven into the negative-cosine regime by training, and from
  epoch 2 onward `full` and `semantic-only` become numerically almost
  identical, meaning the semantic branch dominates the combined
  representation's behavior for the rest of training (the id-only branch's
  contribution becomes comparatively small once both are summed).
- `id-only cos AUC` is consistently noisier and lower (0.25–0.50, jumping
  around) than `semantic-only cos AUC` (0.61 at init, climbing steadily to
  0.73), suggesting the semantic content, once engaged, does carry some real
  angular/relational signal that the pure identity-embedding pathway does
  not — this signal is simply overwhelmed by the sign-flip mechanism from
  Diagnostic 4, not absent.

**Combined interpretation (4+5):** the reproduction's root cause is now
characterized far more precisely than before. It is not "the encoder fails
to learn angular structure" in a vague sense — it is a **specific, abrupt,
reproducible training dynamic**: an architecturally-inherent, unlearned
magnitude/degree signal (present from initialization, driven by unnormalized
`GraphConv` on a heavy-tailed bipartite graph) survives training completely
intact, while training simultaneously and rapidly (within 1–2 epochs) drives
the embedding geometry's angular component to a near-uniform negative
cosine that is indistinguishable between true and false edges — and because
the raw dot-product decoder multiplies these two effects together, the
combination inverts the one part of the signal (magnitude) that was
actually good. The semantic (content) and node-ID (identity) pathways both
independently exhibit this same failure mode, just on slightly different
epoch timelines.

## Diagnostic 2 (part 2) & 6 — RNG-regime downstream-effect comparison

**Script:** `src/polyllm/diagnostics/rng_regime_downstream_retrain.py`
**Result:** `outputs/polyllm/gnn_diagnostics/rng_regime_downstream_comparison.csv`

Two 3-epoch trainings from the same initialization seed, identical
`DEFAULT_CONFIG`, identical data — the only difference is whether the RNG is
reseeded every batch (a literal, inline reproduction of the authors'
`set_seed(12)` behavior, confirmed by Diagnostic 2 part 1 to make dropout
deterministic across calls) or left alone (this repo's actual, current
behavior).

| Epoch | repo: dot AUC | repo: cos(pos) | authors: dot AUC | authors: cos(pos) |
|---|---|---|---|---|
| 0 (init) | 0.854 | +0.061 | 0.854 | +0.061 |
| 1 | **0.146** | −0.039 | **0.855** | +0.058 |
| 2 | 0.854 | +0.045 | **0.145** | −0.083 |
| 3 | **0.144** | −0.004 | 0.145 | −0.082 |

**Headline result: reverting to the authors' RNG-reseeding behavior does
NOT fix the inversion.** By epoch 3, both regimes have landed in the same
inverted state (dot AUC ≈0.145, negative cosine for both classes). The
authors' RNG bug is not the cause of the core metrics gap — this directly
answers the causal question asked in this session: fixing that bug is not
what produced the sub-random AUC.

**A second, unplanned but important finding: the "repo" regime here does
NOT match Diagnostic 4's earlier "repo" trajectory**, despite nominally
identical config and seed. Diagnostic 4 (separate process, 20k-edge sample)
showed a clean one-way transition — good through epoch 1, inverted from
epoch 2 onward, and stable thereafter. This script's "repo" regime (same
process, 5k-edge sample) instead **oscillates**: inverted at epoch 1, back
to good at epoch 2, inverted again at epoch 3. The "authors" regime, by
contrast, shows the same clean one-way transition pattern as Diagnostic 4
(good at epoch 1, inverted from epoch 2 on, stable). This suggests the
authors' per-batch reseeding has an incidental *stabilizing* effect on the
epoch-to-epoch trajectory (by removing batch-to-batch stochastic variation),
even though it does not change the ultimate outcome — and, more importantly,
that **this codebase's training is not fully reproducible run-to-run even
at a fixed top-level seed**, most likely because `LinkNeighborLoader`'s
internal neighbor-sampling and/or floating-point summation order in
`GraphConv`'s unnormalized aggregation (activations reach magnitudes on the
order of 10^5, where float32 summation is not strictly order-independent)
introduce additional, unseeded variability. This is a genuine caveat on
every single-run diagnostic in this investigation, including the official
0.41 result — not yet quantified, flagged here as a real open question
rather than something already resolved.

**Answer to "did fixing the authors' bugs cause the metrics issue?"**
No — empirically tested now, not just argued from documentation. The
inversion occurs under BOTH RNG regimes; the authors' own (buggy) behavior
produces the identical failure mode, just via a more stable path. Combined
with Diagnostic 4/5's finding that the inversion is driven by an
architecturally-inherent magnitude signal colliding with a training-induced
uniform-negative-cosine collapse, the RNG-reseeding deviation can now be
ruled out as a contributing cause.

## Diagnostic 7 — legacy PyTorch 2.4.1 / PyG 2.6.1 environment

**Script:** `src/polyllm/diagnostics/legacy_env_rerun.py`
**Environment:** new `.venv-polyllm-legacy/` (Python 3.11.9, `torch==2.4.1+cpu`,
`torch_geometric==2.6.1`, `pyg-lib==0.4.0+pt24cpu` — a wheel exists for this
exact combination), added to `.gitignore` alongside the other venvs.
**Result:** `outputs/polyllm/gnn_legacy_env/test_metrics.json`

Completely unchanged architecture, `DEFAULT_CONFIG` hyperparameters, data,
and RNG handling — the graph was rebuilt fresh from the same source feature/
label arrays (environment-independent NumPy files) to avoid any cross-version
`HeteroData` pickling issues, and the same `build_model`/`build_link_loader`/
`EarlyStopping`/`run_train_epoch`/`run_eval_epoch`/`collect_test_predictions`/
`compute_test_metrics` functions were imported and called unchanged from
`train_gnn.py`/`evaluate_gnn.py`. No gradient clipping, no LR changes, no
extra epochs, no aggregation/decoder changes, per explicit instruction.

**One blocking issue encountered and resolved, documented in full because
it is NOT a computational change:** under `torch==2.4.1`, `to_hetero`'s
FX-based code generation raised `AssertionError` in
`torch/fx/graph.py::add_global` before any model code ran. Isolated via a
minimal repro (`GraphConv` + `to_hetero` + a type-annotated `forward()`
under `from __future__ import annotations`) — confirmed the bug requires
BOTH the postponed-annotations directive (present in `models/gnn.py`) AND
type hints on `forward()`; removing either one avoids it. It reproduced
identically under both `torch_geometric==2.6.1` and `2.6.0`, so it's a
`torch.fx` behavior under 2.4.1, not a PyG regression. **Fix: resolve
`GNNEncoder.forward`'s postponed string annotations into real objects via
`typing.get_type_hints()` once, before `to_hetero` traces it** — a pure
Python type-introspection step, monkeypatched only inside
`legacy_env_rerun.py` (never written to `models/gnn.py` on disk), with zero
effect on any tensor computation, weight, or forward-pass value (verified:
the same minimal repro produces identical output with or without the type
annotations present at all — annotations affect only FX's pretty-printer,
never actual execution).

**Test-set result:**

| Metric | Legacy env (2.4.1/2.6.1) | Official (2.12.1/2.8.0.post1) | Diff | Paper | Std devs from paper (legacy) |
|---|---|---|---|---|---|
| AUC | **0.6009** | 0.4113 | **+0.1896** | 0.9228±0.0039 | −82.5 |
| AUPRC | **0.6866** | 0.5297 | **+0.1569** | 0.8944±0.0025 | −83.1 |
| AP@50 | **1.0000** | 0.9507 | +0.0493 | 0.9599±0.0044 | +9.1 |

**Interpretation — a real, partial effect, not a null result and not a
resolution:** the PyTorch/PyTorch-Geometric version gap is a genuine,
measurable contributor to the reproduction's poor performance — AUC moves
from *worse than random* (0.41) to *meaningfully above random* (0.60), a
+0.19 absolute change with zero architecture/hyperparameter/data changes.
This is the single largest swing found from any one variable in the entire
investigation. However, it does **not** reach the paper's reported
performance (still −82.5 std devs from the paper's mean AUC) — the software
version gap explains part of the picture but leaves most of the gap to the
paper unexplained. AP@50=1.0000 should be read cautiously — likely a
ceiling/saturation effect at this sample's top-50, not necessarily evidence
of dramatically better ranking quality than AP@50=0.9507's already-high
baseline.

**What this does and doesn't tell us:** it's plausible that some internal
behavior of `GraphConv`, `RandomLinkSplit`, `negative_sampling`, or
`to_hetero` genuinely differs in a way that matters between these library
versions (beyond the FX-codegen bug already isolated and worked around) —
but this diagnostic does not isolate WHICH specific internal difference is
responsible, only that the aggregate effect of the version swap is real and
substantial. Not investigated further, per the explicit scope limit on this
diagnostic (software-environment swap only, no architecture/hyperparameter/
decoder-level follow-up).

**Evidence reference:** `outputs/polyllm/gnn_legacy_env/{test_metrics.json,
training_history.csv, training_config.json, gnn_graph_audit.json,
checkpoints/best_model.pt}`; `src/polyllm/diagnostics/legacy_env_rerun.py`.
