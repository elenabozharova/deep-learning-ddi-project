# GNN reproduction-gap diagnostics (Milestone 10, continued investigation)

**Date started:** 2026-08-13
**Status:** CLOSED (2026-08-13), reopened once (2026-08-13, 7-part protocol),
reopened again and re-CLOSED 2026-08-16 with a source-equivalence/
methodology-equivalence audit, then reopened once more for a checkpoint-
selection diagnostic (2026-08-16, positive lead: historical AUC-based
checkpointing reproduces the paper's AUC within noise) and a follow-up
epoch-0/structural sanity check, **permanently CLOSED 2026-08-17**. See
"2026-08-17 — final sanity check" at the bottom for the final round. All
rounds: gap remains unresolved as originally scoped (loss-based
checkpointing), though the historical AUC-based checkpointing lead reproduces
the paper's number within noise as a point estimate — no further tuning
without a new specific hypothesis and explicit request.

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

---

## 2026-08-16 (continued) — checkpoint-selection criterion diagnostic (loss vs. AUC) — REOPENED, POSITIVE LEAD FOUND

**Trigger:** a historical finding in the PolyLLM authors' git history —
commit `785556660a62ad0aa772f1adccf1b08b039f779b` ("Updated Early Stopping
to stop by loss") changed `EarlyStopping` to expect validation loss, but
`graph.py`'s call site was not updated in that same commit. User asked
whether checkpointing by best validation AUC instead of best validation loss
(everything else held identical, single training trajectory) explains part
of the reproduction gap.

**Scope guardrails:** no edits to `train_gnn.py` / `evaluate_gnn.py` /
`models/gnn.py`; no LR/dropout/architecture/negative-sampling/split changes;
new code only in `src/polyllm/diagnostics/early_stopping_criterion.py`; new
outputs only in `outputs/polyllm/gnn_diagnostics/early_stopping/`; the
official `outputs/polyllm/gnn/` result untouched.

### Git history — confirmed exactly as hypothesized

Diffed commit `7855566` against its parent `b3d4fc57` (via GitHub API) and
against the next commit touching `graph.py`, `21bef01b` (85 seconds later):

| Version | Commit | `EarlyStopping.__call__` | `graph.py` call site | Consistent? |
|---|---|---|---|---|
| A (before) | `b3d4fc57` | `(val_auc, model)`, `score = val_auc`, higher better | `earlystopping(val_auc1, model)` | Yes |
| B (`7855566`) | `785556660a62` | `(val_loss, model)`, `score = -val_loss`, lower better | `earlystopping(val_auc1, model)` — **unchanged**, `graph.py` byte-identical to A | **No** |
| C (after) | `21bef01b` (+85s) | same as B | `earlystopping(val_total_loss / val_total_examples, model)` — old AUC line commented, not deleted | Yes — matches this repo's current `EarlyStopping` |

At B, `score = -val_auc1`: since the algorithm checkpoints whenever score
*increases*, and score increases exactly when `val_auc1` *decreases*, this
combination checkpoints on the **worst**-AUC epochs — mathematically
confirmed, not just plausible. Whether this ever ran in a real training job
is unverifiable from git history alone (superseded 85 seconds later). In
both A and B, `graph.py` only reloads the saved checkpoint before testing
**if `early_stop` actually triggers** (patience exhausted) — otherwise the
last epoch's weights go to test, uncorrected.

### Diagnostic design

Single training run (`data/graph/gnn_link_split.pt`, seed=42, identical
`DEFAULT_CONFIG`) with two non-interfering checkpoint trackers on the same
trajectory:
- **ES-LOSS** — `train_gnn.EarlyStopping` reused unchanged (patience=2 on
  val loss) — this both decides when the run stops (identical stopping rule
  to the official run) and produces the exact checkpoint the official
  pipeline would.
- **ES-AUC** — a simple max-tracker on val AUC, updated every epoch, saved
  to its own checkpoint file, never affects when training stops.
- A checkpoint is also saved at *every* epoch (incl. epoch 0 = init) so
  Phase 6 ("was there ever a good epoch before collapse?") needs no second
  run.
- Train/val AUC/AUPRC reuse the same per-batch predictions already computed
  for the loss (no extra forward pass) — mirrors the authors' own
  `graph.py`, which logs metrics from the same batch used for backprop.
- All test-set numbers use the actual `evaluate_gnn.py` functions
  (`collect_test_predictions`/`compute_test_metrics`), i.e. the real
  mini-batched `LinkNeighborLoader` protocol — not the non-standard
  full-graph propagation used by the earlier (2026-08-13) geometry
  diagnostics.

### Full trajectory (`outputs/polyllm/gnn_diagnostics/early_stopping/trajectory.csv` + `per_epoch_test_metrics.csv`)

Early stopping (loss-based) triggered after epoch 7 (best val loss = epoch 5).

| Epoch | train_loss | train_auc | val_loss | val_auc | val_auprc | val_ap@50 | test_auc | test_auprc | test_ap@50 |
|---|---|---|---|---|---|---|---|---|---|
| 0 (init) | — | — | — | — | — | — | **0.9391** | 0.9126 | 0.8478 |
| 1 | 109.66 | 0.505 | 1.6166 | 0.5422 | 0.578 | 0.829 | 0.4713 | 0.5086 | 0.9225 |
| 2 | 8.14 | 0.525 | 1.5998 | **0.9272** | 0.908 | 0.999 | **0.9201** | 0.8878 | 0.8770 |
| 3 | 3.48 | 0.565 | 0.9976 | 0.1022 | 0.348 | 0.240 | 0.1591 | 0.3616 | 0.5928 |
| 4 | 2.50 | 0.576 | 0.6492 | 0.6512 | 0.738 | 1.000 | 0.5763 | 0.6691 | 0.9659 |
| 5 (best loss) | 2.01 | 0.578 | **0.6282** | 0.7359 | 0.801 | 0.997 | 0.6264 | 0.7148 | 0.9698 |
| 6 | 1.75 | 0.585 | 0.6302 | 0.7220 | 0.794 | 1.000 | 0.5845 | 0.6838 | 0.9758 |
| 7 | 1.52 | 0.589 | 0.6568 | 0.6436 | 0.740 | 0.994 | 0.4946 | 0.6103 | 0.9096 |

**Best-loss epoch: 5** (val_loss=0.6282). **Best-AUC epoch: 2**
(val_auc=0.9272). **ES-BUGGY-HISTORICAL (min val_auc, sanity check per Phase
8): epoch 3** (val_auc=0.1022) — confirms the interpretation of commit
`7855566` behaviorally: it would have chased the worst epoch.

Validation AUC and test AUC track each other closely at every single epoch
(both peak together at epoch 2, both crash together at epoch 3) — this is
genuine validation-set signal, not test-set cherry-picking.

### Checkpoint comparison (identical frozen test set, official protocol)

| Checkpoint | Selected epoch | Val loss | Val AUC | Test AUC | Test AUPRC | Test AP@50 |
|---|---:|---:|---:|---:|---:|---:|
| **ES-LOSS (current/official mechanism)** | 5 | 0.6282 | 0.7359 | 0.6267 | 0.7145 | 0.9636 |
| **ES-AUC (historical, pre-`7855566`)** | 2 | 1.5998 | 0.9272 | **0.9202** | **0.8877** | 0.9065 |
| ES-BUGGY-HISTORICAL (commit `7855566` literal behavior) | 3 | 0.9976 | 0.1022 | 0.1593 | 0.3614 | 0.7232 |

Paper target: AUC 0.9228±0.0039, AUPRC 0.8944±0.0025, AP@50 0.9599±0.0044.
**ES-AUC's test AUC (0.9202) is 0.7 std devs from the paper's mean — inside
the noise band.** AUPRC (0.8877) is 2.7 std devs away — close but not
inside. AP@50 (0.9065) is more off (12.1 std devs) — the current ES-LOSS
checkpoint's AP@50 (0.9636) is actually closer to the paper's on that one
metric specifically, an interesting asymmetry.

**Confirms Version A's literal behavior would independently have arrived at
the ES-AUC checkpoint**: replaying the true historical stopping rule
(patience=2, counted against AUC non-improvement) on this same trajectory —
epoch 1 first best, epoch 2 improves (counter=0), epoch 3 worse (counter=1),
epoch 4 still below epoch 2's best (counter=2 → stop) — early-stops after
epoch 4 and restores epoch 2's weights. Same checkpoint as ES-AUC above,
reached via the actual pre-`7855566` mechanism, not just the max-tracker.

### Phase 6 — did the model ever look good before collapsing?

Yes, and more dramatically than expected: **the single best test AUC across
all 8 evaluated checkpoints (0.9391) occurs at epoch 0 — before any training
step at all.** Epoch 2 is the best *trained* checkpoint (0.9201). This is
flagged as an important open caveat, not popped as a clean success: an
untrained, randomly-initialized model should not plausibly score 0.94 AUC
through genuine learned structure — this strongly suggests an architectural
signal (in the same family as the previously-documented degree/magnitude
baseline effect, see 2026-08-11/13 findings) rather than fully genuine
learned discriminative structure, now apparently present under the
*mini-batched* evaluation protocol too, not only the full-graph one it was
originally characterized under. **Not investigated further here** — would
require new hypothesis work explicitly out of this diagnostic's scope.

### Phase 7 — geometry at best-loss vs. best-AUC epoch: an important methodology split, not resolution

Reused `representation_diagnostics.measure_representation` (full-graph,
non-mini-batched propagation — same methodology as the 2026-08-13
Diagnostics 4/5) at both selected epochs:

| | Epoch | dot AUC | norm AUC | cos AUC | cos(pos) | cos(neg) | pdrugs norm mean |
|---|---|---|---|---|---|---|---|
| best_loss | 5 | 0.148 | 0.852 | 0.519 | −0.070 | −0.070 | 1,181,626 |
| best_auc | 2 | 0.148 | 0.852 | 0.669 | −0.086 | −0.087 | 2,270,536 |

**Both checkpoints look identically "collapsed" under full-graph
propagation** — near-uniform negative cosine, dot AUC≈0.15 for both, despite
epoch 2 scoring 0.92 and epoch 5 scoring 0.63 under the actual official
mini-batched protocol. **This is a genuinely new finding**: the full-graph
propagation regime (unbounded neighborhood aggregation, activations reaching
astronomical norms — millions, vs. the mini-batched regime's bounded
`num_neighbors=[20,10]` sampling) is evidently a different computational
regime from what the paper-comparable metric actually uses, and the
"geometry collapse" characterized in the 2026-08-13 round does **not**
directly explain what's happening in the number that matters (0.41 official,
or 0.92 at ES-AUC here). The earlier root-cause narrative (magnitude/cosine
sign-flip mechanism, full-graph propagation) should be read as characterizing
a related-but-distinct phenomenon, not a direct explanation of the
mini-batched official metric's behavior. This reframes — but does not
invalidate — the 2026-08-13 findings.

### Known confound — run-to-run non-determinism (already documented, re-observed here)

This diagnostic's own ES-LOSS checkpoint (epoch 5, test AUC=0.6267) does
**not** match the frozen official checkpoint's number (epoch ~6, test
AUC=0.4113) despite identical seed/config/code. This is the same
already-documented (2026-08-13) `LinkNeighborLoader`/float32-order
non-reproducibility, re-confirmed here, not a new bug. It means: **the
ES-LOSS-vs-ES-AUC comparison within this one run is internally valid**
(both checkpoints come from the literal same weight trajectory), but the
absolute numbers should not be expected to reproduce exactly on a re-run,
and the true magnitude of the loss-vs-AUC effect likely has real run-to-run
variance not yet quantified (would require a seed sweep — explicitly out of
scope per the hard stopping rule).

### Classification: **A — strong explanation**

`best_auc_test_auc (0.9202) >= 0.80` per the pre-registered thresholds.
ES-AUC's test AUC lands inside noise of the paper's reported mean. This is
the single most significant positive lead found across the entire GNN
investigation (2026-08-11 through 2026-08-16). It is **not** presented as a
resolved fix — the epoch-0 anomaly (Phase 6) and the run-to-run
non-determinism confound both require further, explicitly-scoped
investigation before treating AUC-based checkpointing as a validated
correction. No further GNN modifications made pending user review of this
report, per the hard stopping rule.

**Evidence reference:** `outputs/polyllm/gnn_diagnostics/early_stopping/{trajectory.csv,
per_epoch_test_metrics.csv, summary.json, epoch_checkpoints/epoch_{0..7}.pt,
checkpoint_best_loss.pt, checkpoint_best_auc.pt}`;
`src/polyllm/diagnostics/early_stopping_criterion.py`.

---

## 2026-08-16 — source-equivalence audit (final round for this branch)

**Scope:** a methodology-equivalence audit against the paper's stated GNN
protocol and the authors' released source (not new hyperparameter tuning).
No architecture/config edits. No new official training run — the frozen
`outputs/polyllm/gnn/` result (commit `6e436fb`, tests 629/629 passing
beforehand) was used throughout; all new evidence is either read from
already-saved artifacts or computed fresh via ad hoc read-only scripts
against the existing saved graph/checkpoint (not saved as permanent
diagnostic modules — reproducible from the commands below).

**Phase 1 — frozen state:** commit `6e436fb` ("Add PubMedBERT and SapBERT
side-effect representation experiment"), working tree clean, Experiment 1
already committed. SHA-256 of `bert_side_effect_embeddings.npy`
(`4b001910…e5f52`), `chemberta_pair_embeddings.npy` (`85053b56…19de1`),
`polyllm_label_mapping.csv` (`243ba454…f89b8`), `polyllm_labels.npy`
(`4a65b213…cb980`), `gnn_link_split.pt` (`37bc9077…20ad3`),
`checkpoints/best_model.pt` (`29d26c26…4516f`) all recorded. Official metrics
unchanged: AUC=0.4113, AUPRC=0.5297, AP@50=0.9507. Full suite 629/629 passed
before any new work.

Most of the paper/source/reproduction comparison was **not re-derived from
scratch** — it was already established with primary-source rigor across the
2026-08-11 and 2026-08-13 rounds (verbatim-fetched authors' source, static +
runtime layer-order trace, node/edge/embedding alignment audit, RNG-reseeding
causal tests, legacy-environment rerun). This round's new work targeted only
what was still open: the 963/964 discrepancy, `to_hetero` parameter parity,
literal set-based split proofs, and a concrete quantification of the
negative-sampling coordinate bug's impact.

### New finding 1 — the 963/964 and 498-edge discrepancy, fully explained (Phase 3)

Recomputed the Milestone 1 frequency table (`count_se_frequency`) on the raw
Decagon file before the `>=500` filter. Exactly one side effect sits closest
to (and just under) the threshold in the *entire* 1,317-label universe:

```
side_effect_id = C0221247, name = "avascular necrosis", unique_pair_count = 498
```

This single label exactly and uniquely accounts for both discrepancies
simultaneously: `963 + 1 = 964` labels, `4,576,287 + 498 = 4,576,785` edges —
matching the paper's stated totals to the exact integer, not approximately.

Traced through every preprocessing stage individually for this one label
(raw row count → post exact-duplicate-removal → post canonicalization → post
self-pair exclusion → post canonical-triple dedup): **498 at every single
stage, with zero duplicates or self-pairs affecting it anywhere.** This rules
out our own deduplication/self-pair logic as the cause — the count is
genuinely 498 in the raw `ChChSe-Decagon_polypharmacy.csv.gz` file we
downloaded, under any reasonable counting method.

**Conclusion:** the discrepancy is not a bug in `prepare_multilabel_dataset.py`
— it correctly applies the paper-stated `>=500` threshold to the data we
have. The most plausible remaining explanation is a minor version/snapshot
difference in the underlying Decagon/TWOSIDES source file between what we
downloaded and what the authors used (a difference of 2 associations for one
borderline side effect is well within the range of known TWOSIDES republish
drift). **This does not explain the AUC gap** — 498 of 4,576,785 edges is
0.011% of the graph; restoring this one label was not attempted (no genuine
bug to fix, and doing so would mean hand-picking a data cutoff to hit a
target count rather than fixing a process).

### New finding 2 — `to_hetero` produces genuinely relation-specific parameters (Phase 9)

Inspected `GNNLinkPredictor.state_dict()` directly (48 tensors). Confirmed
`to_hetero` gives **distinct** `GraphConv` weights per relation
(`encoder.conv{1,2,3}.pdrugs__associated__seffect.{lin_rel,lin_root}` vs.
`...seffect__rev_associated__pdrugs...` — 6 separate tensors per layer) and
**distinct** per-node-type `BatchNorm`/final `Linear`
(`encoder.bn{1,2}.pdrugs` vs. `...seffect`, `encoder.lin1.pdrugs` vs.
`...seffect`). This is genuinely heterogeneous parameterization, not a
homogeneous graph with doubled edges sharing one weight set — matches the
authors' own use of `to_hetero()` on the same base `GNN`/`GNNEncoder`. No
discrepancy found.

### New finding 3 — literal set-based proof of the transductive split (Phases 4–7)

Loaded the actual saved `gnn_link_split.pt` and computed real
`edge_index`/`edge_label_index` sets (not just count arithmetic):

```
val_mp  == train_mp ∪ train_sup                    → True  (exact set equality)
test_mp == train_mp ∪ train_sup ∪ val_sup          → True  (exact set equality)
train_sup, val_sup, test_sup mutually disjoint      → True
train_mp ∪ train_sup ∪ val_sup ∪ test_sup == full_pos → True (exact partition)
REV_EDGE_TYPE has edge_label_index in any split?    → False (all 3 splits)
forward message-passing count == reverse count      → True (all 3 splits, exactly)

train fraction = 0.8000, val = 0.1000, test = 0.1000
train supervision/train_total = 0.3000, message-passing/train_total = 0.7000
```

This is an exact, literal confirmation — not an inference from
`RandomLinkSplit`'s documented parameters — that: validation message-passes
over ALL training edges, testing message-passes over training+validation
edges, reverse edges are never a supervision/prediction target in any split,
and the 80/10/10 and 30/70 splits are exact (to 4 decimal places, on real
integer counts). **Zero discrepancy vs. the paper's stated protocol.**

### New finding 4 — negative-sampling coordinate bug, now precisely quantified (Phase 12)

The local/global index bug (found and fixed in Milestone 10d, documented in
the `gnn-authors-source-verified` memory) was re-examined with an actual
number, not just "would fail to filter almost all collisions." On one real
training batch (65,536 sampled negative candidates):

| | count |
|---|--:|
| Real false negatives among sampled candidates (ground truth, global identity) | 6,730 (10.27%) |
| **A/C — our fix** (compares global-mapped pairs): leaked through after filtering | **0** |
| **B — literal authors' code** (compares local pairs against the global set): correctly caught | 1,091 / 6,730 |
| **B — leaked through as mislabeled negatives** | **5,639 / 6,730 (83.79%)** |
| B — good negatives incorrectly discarded (spurious local-index collisions) | 4,167 |

Running the authors' code exactly as released would inject mislabeled
true-positive-as-negative examples into `BCEWithLogitsLoss` for roughly
**8.6% of every batch's negative labels** (5,639/65,536) — a real, sizeable
label-corruption source in their literal implementation, now quantified
rather than qualitatively described. **This is a genuine `paper != source`
finding** (the paper describes correctly excluding true positives; the
literal source, under real mini-batching, does not reliably do so) but it
is **not a `source != reproduction` gap that could explain our AUC shortfall**
— our reproduction already implements the fix (0 leaked through, by
construction, regression-tested since Milestone 10d).

### Feasibility check — Phase 15 (layer-by-layer parity against the authors' actual model)

Checked the authors' GitHub repo (`sadrahkm/PolyLLM`) directly via the GitHub
API for released weights: repo root contains only `.gitignore`, `README.md`,
`environment.yml`, `src/`, `data/` (which itself only contains
`poly_pubchem/`, `synergy_pubchem/` — data, not weights); zero GitHub
Releases. **No trained checkpoint was ever published.** Phase 15's literal
request (instantiate the authors' actual trained model and diff tensors
layer-by-layer) is **infeasible** — only their source code is available, not
their weights. All comparisons in this and prior rounds are necessarily
against their *source code's* prescribed computation, not a numerically
verified reference output.

### Phase 14 — DeepChem ChemBERTa parity

Already established with primary-source rigor in earlier milestones (not
re-derived here): checkpoint `DeepChem/ChemBERTa-77M-MLM`
(`notes/chemberta_embedding_findings.md`), mean pooling over non-padding
tokens, pair fusion = element-wise sum
(`notes/deviations_from_paper.md` §0.2, `pair_matrix = drug_matrix[drug1] +
drug_matrix[drug2]`) — matches the paper's selected fusion strategy exactly
(`params.py`'s only active `model_names` entry, `chemberta_deepchem_sum`, is
this repo's exact existing backbone). No discrepancy found; nothing new to
add.

### Final audit table

| Check | Paper | Author source | Current reproduction | Match? | Impact |
|---|---|---|---|---|---|
| graph counts | 63,472 / 964 / 4,576,785 edges | (same, their data) | 63,472 / **963** / **4,576,287** edges | **NO** (source==repro: no; fully explained, see Finding 1) | Negligible (0.011% of edges) |
| reverse edges | MP only, never a prediction target | `T.ToUndirected()` | verified identical via literal set proof | YES | None |
| split | 80/10/10 transductive | `RandomLinkSplit(0.1,0.1)` | verified exact (0.8000/0.1000/0.1000) | YES | None |
| 30/70 disjoint split | 30% supervision / 70% MP | `disjoint_train_ratio=0.3` | verified exact (0.3000/0.7000) | YES | None |
| validation graph | all training edges | RandomLinkSplit behavior | verified exact set equality | YES | None |
| test graph | training + validation edges | RandomLinkSplit behavior | verified exact set equality | YES | None |
| negative sampling | 1:1, true positives excluded, global identity | **local/global bug: 83.79% of real false negatives leak through per batch** (quantified this round) + `torch.manual_seed` no-op (2026-08-13) | fixed: 0 leaked through, regression-tested | paper vs source: **NO**; source vs repro: **NO** (intentional fix) | Real for literal source; not present in our repro |
| ID embeddings | trainable 64-dim, added to projected features | `Linear`+`nn.Embedding`, added | identical; global node_id indexing confirmed correct under neighbor sampling | YES | None |
| `to_hetero` | not detailed | wraps `GNN` | **confirmed via state_dict**: genuinely relation- and node-type-specific params | YES | None |
| GNN layers | 3×GraphConv+BN+LeakyReLU+Dropout(0.8)+Linear | verbatim-confirmed identical, bn3/bn4/dropout_rate dead code | identical (static + runtime hook trace, 2026-08-13) | YES | None |
| neighbor loader | unspecified in paper | `num_neighbors=[20,10]`, batch 65536/2048/2048, shuffle=False | identical | YES | None |
| loss | BCEWithLogitsLoss | same | same | YES | None |
| early stopping | "early stopping" (underspecified) | patience=2, val-loss-monitored | identical | YES | None |
| RNG reseeding | not discussed | reseeds every forward call (defeats dropout; empirically ruled out as AUC-gap cause 2026-08-13) | seeded once at start | source vs repro: intentional NO | Empirically confirmed NOT the cause |
| software versions | PyTorch 2.4.1 / PyG 2.6.1 | (same, their env) | PyTorch 2.12.1 / PyG 2.8.0.post1 (legacy env tested 2026-08-13: AUC 0.41→0.60, real but partial) | NO | Real, partial (+0.19 AUC), does not close the gap |

### Answers to the required questions

1. **Do we implement the paper-described transductive split correctly?** Yes — proven via literal set equality on real saved tensors, not just parameter-arithmetic inference.
2. **Does validation use all training edges for message passing?** Yes, exactly (`val_mp == train_mp ∪ train_sup`, proven).
3. **Does testing use training + validation edges for message passing?** Yes, exactly (`test_mp == train_mp ∪ train_sup ∪ val_sup`, proven).
4. **Are reverse edges used only for message passing?** Yes — confirmed no split's reverse relation ever carries an `edge_label_index`.
5. **Are node-ID embeddings implemented/indexed correctly?** Yes — global vs. local indexing was the exact subject of the Milestone 10d bug fix, which this round's Finding 4 re-confirms and quantifies as correctly resolved in our code (0 leakage).
6. **Is our `to_hetero` architecture equivalent to the authors'?** Yes — now confirmed via literal `state_dict` key/shape inspection, not just "both call `to_hetero()`".
7. **Can the 963/964 and 498-edge discrepancy be explained exactly?** Yes, fully — a single identified side effect (`C0221247`, "avascular necrosis", 498 associations, stable at every preprocessing stage) exactly accounts for both numbers. Root cause of *why* the authors' copy of the raw data apparently counts it at ≥500 remains unresolved (their raw data snapshot isn't available to us), but the discrepancy itself is fully and precisely explained, and confirmed negligible in scale.
8. **Does literal author-source negative sampling differ from paper-intended sampling?** Yes, substantially — quantified this round at 83.79% of real false negatives leaking through per batch under the literal source's local/global comparison bug. This is a real `paper != source` finding, but does not apply to our reproduction, which already implements the fix.
9. **Where is the first layer/tensor divergence between source and reproduction?** None found. Every stage checked (layer order, `to_hetero` parameterization, ID embedding indexing, split construction, loss, optimizer, hyperparameters) matches. No authors' trained checkpoint exists to diff against numerically (Phase 15, confirmed infeasible via GitHub API), so "divergence" here means divergence in prescribed computation, not verified output values — the strongest form of evidence actually obtainable.
10. **Is there a newly identified discrepancy capable of plausibly explaining a large part of the 0.92 → 0.41 AUC gap?** **No.** The two genuinely new discrepancies found this round (963/964 label gap, literal-source negative-sampling bug) are both real but both immaterial to our reproduction's AUC: the label gap is 0.011% of the graph, and the negative-sampling bug is a `paper != source` issue that does not exist in our already-fixed code. Every other checked component (`to_hetero` parameterization, split construction, reverse-edge handling) matches exactly, with no gap-relevant discrepancy found.

### Outcome classification

**D. No meaningful discrepancy found; GNN gap remains unresolved.**

(Two real, previously-uncharacterized discrepancies *were* found and fully
quantified — the label/edge-count gap and the literal source's
negative-sampling bug severity — but both are classified as immaterial to
the AUC gap for the reasons in Q10, not as "meaningful" in the sense the
classification asks about, i.e. capable of explaining the gap. Neither
warranted a code change: the label gap is not a bug in our code, and the
negative-sampling bug is already fixed in our code.)

### Hard stopping rule — applied

Per the explicit instruction: no specific, evidence-backed discrepancy
capable of explaining the AUC gap was found. **The GNN investigation is
frozen again as of 2026-08-16.** No further tuning, ablation, or
environment work should be undertaken without a new, specific hypothesis and
an explicit user request — same rule as the 2026-08-11 and 2026-08-13
closures. Next work should move to Experiment 2.

---

## 2026-08-17 — final sanity check: is epoch-0's 0.9391 AUC systematic or a fluke, and does simple degree explain it?

**Trigger:** the 2026-08-16 checkpoint-selection diagnostic's Phase 6 found
that a completely **untrained** model (epoch 0, seed=42) scores test
AUC=0.9391 under the official mini-batched evaluation protocol — higher than
the trained, paper-matching checkpoint (epoch 2, 0.9202). Before treating
that checkpoint-selection result as the closing word on the GNN
reproduction, the user asked for two narrowly-scoped sanity checks: (1) does
epoch-0's high AUC recur across independent random initializations, or was
seed=42 a one-off; (2) can simple, parameter-free node-degree information
explain it. **No training. No architecture/split/negative-sampling/
hyperparameter change.**

**Scope guardrails:** no edits to `train_gnn.py` / `evaluate_gnn.py` /
`models/gnn.py` / `build_graph.py`; new code only in
`src/polyllm/diagnostics/epoch0_structural_diagnostic.py`; new outputs only
under `outputs/polyllm/gnn_diagnostics/epoch0_structural/`; the frozen
`outputs/polyllm/gnn/` result untouched.

**Methodology note — "same test edges" guaranteed by construction, not by
seed-matching:** this repo already documents (2026-08-16, "known confound"
above) that `LinkNeighborLoader` is not perfectly seed-reproducible run to
run, and negative sampling draws from Python's global `random` state on
every forward pass. Re-iterating the test loader once per initialization
seed would therefore silently vary the edge set between seeds — exactly
what the request prohibits ("do not resample negatives separately for each
baseline"). Instead, the official mini-batched test loader
(`build_link_loader` + `assemble_supervision_edges`, imported unchanged) was
iterated **exactly once** and every batch's tensors (message-passing
subgraph, features, local→global `node_id` maps, assembled
`edge_label_index`/`edge_label`) were frozen in memory. Every seed and every
baseline below reads from this one frozen batch list and never re-samples —
855,294 test edges (457,628 positive / 397,666 negative).

### 1. Epoch-0 initialization table

| Init seed | Test AUC | Test AUPRC | Test AP@50 |
|---:|---:|---:|---:|
| 1 | 0.0558 | 0.3433 | 0.0000 |
| 2 | 0.9442 | 0.9137 | 0.7985 |
| 3 | 0.8706 | 0.8641 | 0.9600 |
| 4 | 0.0589 | 0.3436 | 0.3500 |
| 5 | 0.8663 | 0.8603 | 0.9653 |

mean AUC = **0.5592**, std AUC = **0.4106**, min = 0.0558, max = 0.9442.

**Not a single outlier and not uniformly systematic — bimodal.** 3 of 5
seeds land at 0.87–0.94 (matching or exceeding the trained, paper-matching
checkpoint); 2 of 5 land at 0.056–0.059 (far *worse* than random — the same
magnitude of separation, opposite sign). The historically-flagged seed=42
result (0.9391) sits squarely inside the "high" mode observed here (3/5
seeds), not as an isolated fluke — but neither is "high" the only outcome a
random draw produces.

### 2. Structural baseline table

Degree computed from `test_data[EDGE_TYPE].edge_index` — proven by exact set
membership check (0/457,628 test-positive edges present) to be exactly
train-message-passing ∪ train-supervision ∪ val-supervision, i.e. the graph
legitimately available for message passing at test time, **not** the held-out
test positives.

| Baseline | AUC | AUPRC |
|---|---:|---:|
| Drug-pair degree | 0.5240 | 0.5651 |
| Side-effect degree | 0.7863 | 0.7926 |
| Degree product | 0.7596 | 0.7568 |
| Degree geometric mean | 0.7596 | 0.7568 | (rank-identical to product — proven, `sqrt` is strictly increasing on x≥0)
| Degree sum | 0.7881 | 0.7936 |

Best baseline (degree sum) reaches AUC≈0.79 — real and substantial, well
above chance, but **below every "high" epoch-0 seed** (0.87–0.94) and
obviously cannot by itself explain the "low" seeds (0.056–0.059): a
positively-correlated popularity feature is mathematically incapable of
producing sub-0.5 AUC on its own.

### 3. Degree-distribution comparison — positive vs. negative test edges

| Feature | Class | mean | median | std | p25 | p75 |
|---|---|---:|---:|---:|---:|---:|
| drug-pair degree | positive | 115.2 | 100 | 74.6 | 57 | 160 |
| drug-pair degree | negative | 107.7 | 94 | 68.2 | 55 | 148 |
| side-effect degree | positive | 8,830 | 7,539 | 5,960 | 3,978 | 12,668 |
| side-effect degree | negative | 3,733 | 2,328 | 3,899 | 1,030 | 5,001 |
| degree product | positive | 911,296 | 646,740 | 876,441 | 323,532 | 1,190,772 |
| degree product | negative | 386,164 | 200,408 | 516,815 | 88,200 | 468,028 |

**Yes, sampled negatives are structurally different from positives** — driven
almost entirely by side-effect popularity (positive mean ≈2.4× the negative
mean), not drug-pair popularity (only ≈7% higher for positives). This
directly explains why side-effect degree (0.7863) is a far stronger baseline
than drug-pair degree (0.5240): popular side effects are disproportionately
true positives simply because they have more true edges, and negative
sampling under-represents them by construction (uniform random sampling over
a heavy-tailed side-effect degree distribution mostly hits unpopular ones).

### 4. Optional norm baseline — computed for all 5 seeds (see rationale below)

The pre-registered gate ("only if degree baseline ≈0.50–0.60 and epoch-0
GNN ≈0.90+") never cleanly fired, because the actual result — bimodal, not
uniformly high — wasn't one of the two anticipated patterns. Per-seed,
though, 3 of 5 seeds' epoch-0 AUC (0.866–0.944) already exceeds the best
degree baseline (0.788), which is a per-seed instance of the same "degree
alone is insufficient" trigger. Computing the norm baseline directly tests
the leading candidate mechanism already documented in this file (2026-08-11
post-10f deep-dive / 2026-08-13 Diagnostics 4–5): a real, degree-driven
magnitude signal survives training/initialization essentially unlearned,
while the dot product's *angular* (cosine) component is close to random
noise and can independently land net-positive or net-negative, sign-flipping
the otherwise-good magnitude ranking. No training, no config change — reuses
the exact same frozen batches.

| Seed | Initial-projected-embedding-norm AUC | Actual epoch-0 (post-GraphConv) AUC |
|---:|---:|---:|
| 1 | 0.4988 | 0.0558 |
| 2 | 0.5068 | 0.9442 |
| 3 | 0.5065 | 0.8706 |
| 4 | 0.5029 | 0.0589 |
| 5 | 0.4993 | 0.8663 |

**The pre-message-passing embedding norm is uninformative (AUC≈0.50 for
every seed, regardless of whether that seed's actual epoch-0 AUC is 0.06 or
0.94).** This rules out "raw embedding-initialization scale" as the source
of either the magnitude signal or the bimodal sign split, and **confirms**
(does not newly discover — consistent with the 2026-08-13 finding that
norm-product AUC≈0.845 is a property of the *aggregated* representation, not
the pre-aggregation one) that the entire epoch-0 phenomenon, in both
directions, is generated specifically by the 3-layer `GraphConv`
message-passing step over this graph's heavy-tailed degree distribution —
not by node-embedding initialization scale in isolation.

### 5. Final interpretation

1. **Is epoch-0 AUC systematically high across random initializations?**
   No — it is **bimodal**, not systematically high and not a one-off
   anomaly either. 3/5 seeds land high (0.87–0.94), 2/5 land low
   (0.056–0.059), roughly mirrored around 0.5. The historical seed=42
   finding (0.9391) is representative of the majority-but-not-universal
   "high" mode, not an isolated fluke.
2. **Can simple node-degree information explain it?** Only partially. Real
   degree-correlated separability exists (best baseline AUC≈0.79, driven
   almost entirely by side-effect popularity) but tops out well below the
   "high" seeds' 0.87–0.94, and — by construction, since it's a positively
   correlated feature — cannot explain the "low" seeds' sub-0.5 AUC at all.
   Simple degree explains part of the achievable separation budget, not the
   mechanism that flips its sign.
3. **Are positive and sampled-negative test edges structurally different?**
   Yes, clearly, and specifically through side-effect degree (≈2.4× higher
   for positives) rather than drug-pair degree (≈7% higher) — negative
   sampling under-represents popular (high-degree) side effects relative to
   their true prevalence.
4. **Does this undermine the historical AUC-based reproduction result?** It
   does not directly test epoch 2 (this diagnostic is strictly about epoch
   0, no training performed), so the point estimate (test AUC=0.9202 at the
   historical AUC-selected checkpoint) is not contradicted here. But it adds
   a concrete, independently-derived caution: the exact same official
   evaluation protocol swings from 0.056 to 0.944 AUC before any training
   step, purely from initialization. This reinforces (does not newly
   discover) the already-documented 2026-08-16 "known confound" — that
   single-seed training results in this pipeline carry real, unquantified
   run-to-run variance — now demonstrated concretely at the initialization
   stage rather than only inferred from two mismatched training runs.
5. **Can we now state the reported GNN AUC is reproducible under the
   authors' historical AUC-based checkpoint-selection behavior?** As a
   **point estimate**, yes, unchanged from the 2026-08-16 finding: one
   specific run (seed=42) lands at test AUC=0.9202, 0.7 std devs from the
   paper's 0.9228±0.0039. This diagnostic does not overturn that number.
   What it removes is confidence in *why* that number is high.
6. **What caution should be attached?** Do not present the trained
   checkpoint's 0.92 AUC as clean evidence of learned pharmacological
   structure. An **untrained** model, on the identical test edges, already
   reaches 0.87–0.94 AUC in 3 of 5 random draws — via a demonstrated
   graph-topology artifact (unnormalized `GraphConv` aggregation interacting
   with a heavy-tailed, side-effect-driven degree distribution), not
   learning. The trained checkpoint's high AUC is plausibly this same
   architectural mechanism landing on the "positive-sign" side of an
   initialization-driven coin flip, possibly reinforced rather than created
   by training — not disentangled from genuine learned drug/side-effect
   relational structure by anything measured in this or prior rounds.

### Outcome classification

**C. Historical AUC match confirmed, but epoch-0 anomaly remains
unexplained** — more precisely characterized than before (bimodal, not
uniformly high; partially but not fully attributable to side-effect-degree
correlation; ruled out as arising from pre-aggregation embedding-norm scale;
mechanistically consistent with the already-documented dot=norm×cos
sign-flip effect) but not reduced to one fully isolated, single cause. The
historical checkpoint-selection point estimate (epoch 2, test AUC=0.9202)
stands as previously reported and is not contradicted by anything found
here. Neither Case A ("fully explained by degree") nor Case B ("systematic
but unexplained by degree") from the pre-registered decision tree cleanly
fits, because the actual result — bimodal — was not one of the two
anticipated shapes.

**Evidence reference:**
`outputs/polyllm/gnn_diagnostics/epoch0_structural/{epoch0_seed_sweep.csv,
structural_baselines.csv, degree_distributions.csv, summary.json}`;
`src/polyllm/diagnostics/epoch0_structural_diagnostic.py`.

### Hard stopping rule — applied again

Per the explicit instruction, this is the **final** sanity check for the GNN
reproduction. No implementation bug was discovered during it (the
mini-batched evaluation, negative sampling, and degree computation were all
verified against the existing, already-audited code paths; the one runtime
leakage check included ran clean). **The GNN reproduction work is frozen
permanently as of 2026-08-17.** No further seed sweeps, retraining,
architecture changes, or new diagnostics without an explicit new user
request and a specific, evidence-backed hypothesis — same rule as every
prior closure in this file.

---

## 2026-09-09 — REOPENED at explicit user request: `bn3` (the authors' declared-but-unused BatchNorm) + legacy env + AUC checkpoint, combined, seed sweep

**Trigger:** while drafting the reproduction paper, a fresh read of the
training log surfaced the point that had been recorded but not acted on: the
official GNN run **never fits its own training data** — mean BCE ≈ 120 at
epoch 1, train loss stuck at 1.44 (above the 0.693 random-guess baseline)
after 8 epochs, train AUC 0.589. Every prior geometry/checkpoint diagnostic
was analysing a model that never trained. Root-cause hypothesis: the encoder
ends `conv3 → lin1` with **no normalisation and no activation**, and
`GraphConv` sums over side-effect node degrees of 502–28,568 with no degree
normalisation → unbounded final activations → logits in the hundreds. The
authors' `GNN.py` *declares* `self.bn3` / `self.bn4` (`nn.BatchNorm1d`) but
`forward()` never calls them — previously classified (correctly, against the
released source) as dead code and omitted. Hypothesis: the released `GNN.py`
is not the version that produced Table 5.

**Diagnostic (`src/polyllm/diagnostics/gnn_final_combined.py`,
`outputs/polyllm/gnn_final_diagnostic/`):** all three tractable untested
variables changed at once, on one trajectory per seed —
1. legacy env (torch 2.4.1 / PyG 2.6.1) — Diagnostic 7 tested this alone (0.41→0.60);
2. `bn3` restored: `conv3 → bn3 → lin1` (the one surgical insertion; +256 params);
3. validation-AUC checkpoint tracking alongside the loss-based rule.
Graph split held fixed (seed 42); model-init / negative-sampling / loader
seed swept over 42, 0, 1, 2. Isolated — official `outputs/polyllm/gnn/` untouched.

### Result 1 — `bn3` fixes the training non-convergence. Solid, all 4 seeds.

Train loss goes **0.71 → 0.54** (monotone, every epoch below the 0.693
baseline) vs. the official run's 120 → 1.44 (never below). The scale
explosion is gone. `bn3` is very likely the reason the released code does not
converge; the authors clearly intended it (declared in the class).

### Result 2 — but it does NOT reproduce the paper's AUC under the paper's method.

| Seed | untrained (epoch 0) AUC | **patience-early-stop rule** (paper's stated criterion) | global best val-loss | best val-AUC ckpt |
|---|---|---|---|---|
| 42 | 0.056 | 0.663 | 0.802 | 0.938 |
| 0  | **0.945** | 0.731 | 0.893 | 0.942 |
| 1  | 0.717 | 0.677 | 0.676 | 0.942 |
| 2  | **0.938** | 0.705 | 0.704 | 0.939 |
| paper | — | — | — | AUC 0.9228 ± 0.0039, AUPRC 0.8944 ± 0.0025 |

- **Under the paper's early-stopping criterion, `bn3` gives AUC 0.66–0.73** (4/4 seeds). Better than the official 0.41, still ~0.2 short. Not a reproduction.
- **The untrained-model artefact is NOT fixed by `bn3`** — seeds 0 and 2 give an *untrained* network AUC 0.94–0.95, matching the paper before any gradient step. Same topology artefact documented 2026-08-17, still present.
- **The ~0.94 "best val-AUC" number is that artefact, not a trained model.** Selected at epoch 1 or 3 of 10; val AUC at epoch 1 is 0.936–0.937 on *every* seed regardless of init; an untrained model reaches the same value. `bn3` bounds the activation scale so the artefact stays stable across epochs instead of being destroyed by the scale blow-up — that is why "best val-AUC" is now *consistent* (~0.94, 4/4) where before it was noisy.
- **Run-to-run non-determinism persists:** seed 42 gave best-val-loss 0.80 here vs. 0.86 on the first (pre-sweep) run of the same script — `LinkNeighborLoader` / float32-order, already documented.

### Conclusion (for the paper)

The released GNN code fails to converge because it omits a BatchNorm layer it
declares (`bn3`). Restoring it makes training converge, raising test AUC from
0.41 to **0.66–0.73** under the paper's early-stopping criterion (4 seeds).
The paper's reported 0.92 is reachable only by selecting a very-early-epoch
checkpoint whose performance an **untrained** network matches on 2 of 4
seeds. **The released code does not reproduce the reported GNN performance,
and the reported metric on this negative-sampling protocol substantially
reflects graph topology** (positives concentrate on high-degree side-effect
nodes; negatives sampled uniformly) **rather than learned structure.** This
is a stronger reproducibility finding than a clean match: a concrete
released-code bug *plus* direct evidence (untrained model = paper's number)
that the headline metric is partly an evaluation artefact.

**Evidence:** `outputs/polyllm/gnn_final_diagnostic/results.json` (+ `seed_{0,1,2}/`),
`training_history.csv` per seed, `checkpoints/` per seed;
`src/polyllm/diagnostics/gnn_final_combined.py`;
console log `outputs/polyllm/gnn_final_diagnostic_console.log`.

### Frozen again

GNN work re-frozen 2026-09-09. `bn3` is not adopted into the official
`models/gnn.py` — the reproduction faithfully implements the *released* code,
and `bn3`'s absence there is itself the finding. No further GNN runs without
a new explicit request.
