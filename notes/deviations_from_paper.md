# Milestone 9 — Deviations from the PolyLLM Paper

**Date:** 2026-07-01 (Milestone 9); updated 2026-08-08 (Milestone 9b, AP@50 axis fix), 2026-08-09 (Milestone 9c, LeakyReLU slope fix + retrain), and 2026-08-11 (Milestone 10, GNN reproduction attempt — see §1.4)  
**Paper:** "PolyLLM: polypharmacy side effect prediction via LLM-based SMILES encodings"  
**DOI:** 10.3389/fphar.2025.1617142  

**2026-08-09 note:** Milestone 9c retrained both MLPs after correcting `LeakyReLU` negative slope to match the paper (§1.7). Every reproduction number in this document from before that date reflects the old, paper-mismatched slope=0.01 checkpoints, now archived at `outputs/baseline/morgan_slope0.01_backup/` and `outputs/polyllm/chemberta_slope0.01_backup/`. Sections below are annotated inline where the retrain changed a number; sections not mentioning the retrain are unaffected by it.

This document separates deviations into three categories:
- **Confirmed:** deviation is directly evidenced by available artifacts and paper text.
- **Possible:** deviation is plausible but cannot be confirmed without paper code or intermediate data.
- **Reproduction-specific improvements:** additions in the reproduction that are not in the paper (these are not deviations in a negative sense).

---

## 0. Confirmed Matches to Paper-Specified Design

### 0.1 Multi-hot label vector

| Property | Paper | Reproduction |
|---|---|---|
| Label representation | Binary vector per drug pair, one element per side effect | `numpy.uint8` array, shape `(n_pairs, n_labels)`, one row per pair |
| Vector size | 964 | 963 (see §1.1) |

**Evidence:**
Paper (quoted by user, exact section number not yet located): "we construct a binary vector of size 964 for each drug pair, where each element in the vector indicates the presence (1) or absence (0) of a specific side effect... This transformation enables efficient handling of the multi-label classification problem."
Reproduction: `build_label_matrix()`, `src/polyllm/data/prepare_multilabel_dataset.py:407`; output `data/processed/polyllm_labels.npy`.

**Assessment:** Confirmed match, not a deviation — the reproduction's multi-hot design directly implements what the paper specifies. The only difference is vector length (963 vs 964), already covered as a confirmed deviation in §1.1.

---

### 0.2 Pair embedding fusion strategy (element-wise sum)

| Property | Paper | Reproduction |
|---|---|---|
| Fusion method selected | Summation (chosen after evaluating four strategies; concatenation performed similarly but summation was more computationally efficient) | Summation only |
| Formula | `pair = drug_1 + drug_2` (element-wise) | Same: `pair_matrix = drug_matrix[drug1_indices] + drug_matrix[drug2_indices]` |

**Evidence:**
Paper (quoted by user): "we evaluate four distinct strategies to obtain a comprehensive representation for each drug pair. Our experiments showed that both concatenation and summation yielded similar performance. However, to optimize computational efficiency, we selected summation as our fusion method... we sum the embeddings of the two interacting drugs to produce a unified vector representation for each drug pair."
Reproduction: `build_pair_matrix()`, `src/polyllm/features/build_pair_embeddings.py:137-155`; commutativity independently verified via `verify_symmetry()` (line 162), `notes/pair_embedding_findings.md` §4.

**Assessment:** Confirmed match on the final method and its outcome. Not reproduced: the paper's own ablation across all four strategies — this reproduction adopted summation directly as a fixed design choice rather than independently comparing it against alternatives.

**Future scope — candidate alternative fusion strategies not yet implemented:**
The paper names only two of its four evaluated strategies explicitly (concatenation, summation). `notes/pair_embedding_findings.md` §10 independently flagged three unexplored alternatives, before this paper quote was available:
- **Difference** — element-wise subtraction, `drug_1 - drug_2`
- **Concatenation** — `[drug_1 ; drug_2]`, doubling dimensionality to 768
- **Hadamard product** — element-wise multiplication, `drug_1 * drug_2`

These three are the standard remaining members of the classic four-way vector-pair combination set used widely in sentence-pair NLP literature, and are plausible candidates for the paper's other two unnamed strategies. Implementing and comparing all three against the current sum-based pipeline would be a natural future extension to more fully reproduce the paper's fusion-method ablation, and could also be re-run against the Morgan fingerprint pipeline for a fuller Milestone 8 comparison.

**Status:** Not implemented. Flagged as future scope, not a current deviation requiring correction.

---

## 1. Confirmed Deviations

### 1.1 Label count: 963 vs 964

| Property | Paper | Reproduction |
|---|---|---|
| Original side effect types | 1,318 | 1,317 |
| Retained after ≥500-pair filter | 964 | 963 |

**Evidence:**  
Paper: Section 2.1, "964 commonly occurring types of polypharmacy side effects, each present in at least 500 drug combinations."  
Reproduction: `outputs/polyllm/data_audit.json` field `retained_side_effect_count: 963`.

**Consequence:** The MLP output dimension is 963 instead of 964. All per-label and macro metrics are computed over a different (one label smaller) label set.

**Hypothesis:** The off-by-one is consistent across both raw (1,318 vs 1,317) and filtered (964 vs 963) counts. Most likely explanation: one side effect identifier present in the paper's copy of the dataset is absent or differently canonicalized in our downloaded version, or the paper counts a UMLS CUI that our pipeline normalizes to the same string as another entry. Cannot be resolved without the paper's intermediate data.

---

### 1.2 Evaluation protocol: 10-fold cross-validation vs single fixed split

| Property | Paper | Reproduction |
|---|---|---|
| Cross-validation | 10-fold CV on train+val sets | None |
| Number of training runs | 10 (one per fold) | 1 |
| Results | Mean ± std across 10 folds | Point estimate from single run |

**Evidence:**  
Paper: Section 2.1, "10-fold cross-validation was performed on the training and validation sets."  
Reproduction: `notes/split_findings.md`, single split seed=42; `notes/morgan_baseline_findings.md` and `notes/chemberta_mlp_findings.md`, one run each.

**Consequence:** The paper's ± standard deviations reflect fold-to-fold variance. The reproduction produces a single point estimate with no within-experiment variance estimate. Direct numerical comparison is limited — the reproduction's AUROC may be from any point in the paper's fold distribution.

---

### 1.3 AP@50 formula: incompatible values — RESOLVED 2026-08-08 (Milestone 9b)

**Status: the axis-mismatch hypothesis below is now confirmed.** The authors' exact
reference implementation was found in their public code repository
(`github.com/sadrahkm/PolyLLM`, `src/functions.py` — the same repo later used verbatim
for the GNN reproduction, §1.4). The authors were **not** contacted; this is their
published `functions.py` as committed:

```python
def average_precision_at_k_multi_label(y_true, y_pred, k=50):
    ap_at_k_list = []
    for i in range(y_true.shape[1]):
        sorted_indices = np.argsort(y_pred[:, i])[::-1][:k]
        sorted_true = y_true.iloc[sorted_indices, i]
        ap_at_k_label = average_precision_score(sorted_true, y_pred[sorted_indices, i])
        ap_at_k_list.append(ap_at_k_label)
    return np.mean(ap_at_k_list)
```

This loops over **labels** (`range(y_true.shape[1])`, 963 iterations) and, for each label, ranks all **pairs** by that label's score, keeping the top-k pairs. The reproduction's original `sample_mean_ap_at_50()` (`src/polyllm/metrics.py`) loops over **pairs** and ranks the 963 **labels** within each pair — the opposite axis. This is exactly the "different aggregation direction (per-pair vs per-label)" hypothesis recorded below at the time this section was first written.

The authors' exact function (from their `src/functions.py`) is now implemented as `average_precision_at_k_multi_label()` in `src/polyllm/metrics.py`, and recomputed against the already-saved Milestone 6B/7 test predictions (no retraining, no checkpoint reloaded) by `src/polyllm/recompute_ap_at_k.py` (Milestone 9b). Results:

| Property | Paper (DeepChem ChemBERTa MLP) | Reproduction — original per-pair axis | Reproduction — corrected per-label axis |
|---|---|---|---|
| AP@50 value | 0.7557 ± 0.0120 | 0.3795 | **0.8491** |
| Difference from paper mean | — | −0.3762 (−49.8%, −31.4 std devs) | **+0.0934 (+12.3%, +7.8 std devs)** |

**Evidence:** `outputs/polyllm/chemberta/ap_at_k_recompute.json`, `outputs/baseline/morgan/ap_at_k_recompute.json` (Morgan, for context only — paper does not report a Morgan value: 0.8950, up from 0.4220). Both frozen `test_metrics.json` and `test_predictions.npz` files were confirmed byte-identical (SHA-256) before and after this recomputation — see `outputs/comparison/comparison_audit.json` `source_artifact_hashes`.

**Consequence:** The corrected per-label-axis AP@50 closes roughly 75% of the original 0.3762 gap. The metric is far more comparable now than before, though not identical — the residual +0.0934 (reproduction above paper) is directionally consistent with the already-observed pattern that this reproduction's AUROC (+1.2 std) and AUPRC (+6.5 std) both sit above the paper's reported means (§2.6), so the remaining gap most likely shares whatever cause underlies those two, rather than being a new, AP@50-specific issue.

**One remaining caveat:** 69/963 ChemBERTa labels (7.2%; 199/963 for Morgan) have all 50 top-ranked pairs positive within that label's own top-50, which makes `average_precision_score` return exactly 1.0 for those columns by construction. This is a property of the formula itself (confirmed present in the authors' `functions.py`, not a reproduction bug), not filtered out, and inflates the mean somewhat for very high-prevalence or well-separated labels. Zero labels had the opposite (all-negative-in-top-50) degeneracy for either model.

**What is still not verified:** whether `average_precision_score`'s sklearn interpolated-AP formula is bit-for-bit what the paper's own code used internally (the authors' `functions.py` does call `sklearn.metrics.average_precision_score`, so this is now a much safer assumption than before), and whether the paper's own 10-fold protocol changes this number materially (§1.2 remains a separate, still-open deviation).

**Update 2026-08-09 (Milestone 9c):** after retraining both MLPs with the paper-matched LeakyReLU slope=0.1 (§1.7), this metric was recomputed against the new predictions. ChemBERTa's paper-axis AP@50 moved from 0.8491 to **0.8146** — a further narrowing from +7.8 to **+4.9 std devs** above the paper mean (0.7557 ± 0.0120), on top of the axis fix above. Degenerate all-positive-top-50 columns dropped from 69/963 (7.2%) to 28/963 (2.9%), consistent with the slope=0.1 model being somewhat less confident/less separated overall (see the parallel drop in raw AUPRC in §1.7). Morgan (context only): 0.8950 → 0.8716. Current numbers: `outputs/polyllm/chemberta/ap_at_k_recompute.json`, `outputs/baseline/morgan/ap_at_k_recompute.json`; the pre-retrain values above remain historically accurate for the archived slope=0.01 checkpoints.

---

### 1.4 GNN path — attempted (Milestone 10, 2026-08-11); architecture verified faithful, but does not reproduce the paper's reported performance

| Property | Paper | Reproduction |
|---|---|---|
| MLP with ChemBERTa embeddings | Yes | Yes (reproduced, §0–§1.9) |
| GNN with ChemBERTa node features | Yes (best results, Table 5) | Implemented and trained; test-set metrics far below paper (see below) |

**What was implemented, all traced verbatim from the authors' actual source** (`github.com/sadrahkm/PolyLLM`, `src/graph/{graph,helpers,GNN,Model,Classifier,eval,EarlyStopping}.py`, `src/functions.py`, `src/embed/Embedding.py`, fetched character-for-character on 2026-08-11, not AI-summarized):
- Bipartite `HeteroData` graph: 63,472 `pdrugs` nodes (reusing Milestone 5's 384-dim ChemBERTa pair embeddings) + 963 `seffect` nodes (new 768-dim `bert-base-uncased` embeddings of side-effect names, masked mean-pooled — the exact recipe traced from `Embedding.get_embeddings('bert', ...)`), 4,576,287 positive edges (paper: 4,576,785 — the already-known §1.1 label off-by-one, not a new gap). `T.ToUndirected()` adds the reverse relation for message passing only, matching the paper's Section 2.5.2.1 text exactly (doubles the edge count for message passing; decoder still only scores the forward direction).
- `T.RandomLinkSplit(num_val=0.1, num_test=0.1, disjoint_train_ratio=0.3, is_undirected=True, neg_sampling_ratio=0.0)` — exact authors' parameters, and the resulting split arithmetic (train supervision=1,098,309, val=457,628, test=457,628) matches the paper's stated 80/10/10 split with 30% of the training portion held out for supervision exactly.
- 3-layer `GraphConv` encoder (`torch_geometric.nn.GraphConv`) + BatchNorm + LeakyReLU (default slope 0.01 — the authors' `GNN.py` never overrides it, unlike the MLP path) + Dropout(0.8), wrapped heterogeneously via `to_hetero`. Verified against the paper's own Equation 1 (`h_i^(k) = W_1^(k)·h_i^(k-1) + W_2^(k)·Σ_{j∈N(i)} h_j^(k-1)`) — an unnormalized self-transform-plus-neighbor-sum, exactly what `GraphConv` implements; the paper specifies no degree normalization.
- Node features lifted via `Linear(pdrugs_dim/768, hidden_channels=64) + Embedding(num_nodes, 64)` (learned per-node ID embedding summed with the projected precomputed features) — confirmed against the paper's own text ("node IDs are first mapped into a latent space of size 64 and are then combined with the precomputed features... projected... using a simple one-layer neural network").
- Dot-product decoder (`(h_i · h_j)`, raw, no scaling) — confirmed against the paper's Equation 2 and the authors' `Classifier.py`, character-for-character identical.
- Negative sampling: `negative_sampling()` on the message-passing edges, filtered against a global true-positive set — confirmed against the paper's text ("exclude already-connected pairs, then randomly select others, labeled zero... each mini-batch contains both positive and negative edges").
- Training: Adam(lr=0.01), `BCEWithLogitsLoss`, `LinkNeighborLoader(num_neighbors=[20,10])`, train batch_size=65536 / val+test batch_size=2048, `EarlyStopping(patience=2, monitors mean validation loss)`, hidden_channels=64 — all from the authors' `params.py`/`graph.py`, traced verbatim.
- Metrics: `roc_auc_score`, `average_precision_score`, and the authors' exact (non-standard) `average_precision_at_k` formula from `src/functions.py` — flattened over the whole test set (positive + freshly-sampled-negative edges), not the MLP path's per-label macro framing. Implemented as `polyllm.metrics.edge_level_average_precision_at_k`, unit-tested against a literal line-by-line translation of the authors' snippet.

**One real bug found in the authors' own code, and fixed:** `LinkNeighborLoader` mini-batches renumber sampled nodes to a local 0-indexed space per batch (empirically confirmed: a real batch's `node_id` was `[358, 1440, 2711, ...]`, not `arange`). The authors' `Model.valid_negative_sampling` builds its true-positive exclusion set once from the *full* graph in *global* indices, then compares it directly against *locally*-indexed sampled negative edges from every mini-batch — two different index spaces, which would fail to filter almost all true-positive collisions in a real (non-full-graph) batch. Fixed in `src/polyllm/models/gnn.py::sample_negative_edges` by translating sampled negatives' local indices to global via each node's `node_id` before filtering, keeping local indices for the edges that survive (required for indexing into the batch's own embeddings). Proven with a deterministic regression test (`tests/test_gnn_model.py::TestNegativeSampling::test_local_batch_indices_filtered_via_global_node_id_not_local_index`).

**Two confirmed-dead code paths omitted, not reproduced as inert clutter:** `GNN.__init__` declares `bn3`/`bn4` (`BatchNorm1d`) that are never referenced in `forward()`, and a `dropout_rate` constructor parameter that is never used (dropout is hardcoded to 0.8 regardless). Both confirmed dead by direct verbatim reading, not inference.

**One RNG-reseeding bug not reproduced:** the authors' `Model.forward` calls a full reseed (`set_seed(12)`, then `torch.manual_seed(42)` again before negative sampling) on *every single forward call* — this would make dropout masks identical across every forward pass within a run, which looks like an unintentional bug (it defeats dropout's purpose) rather than a deliberate design choice. Not reproduced; seeded once at run start instead, consistent with every other script in this repository.

**Scoped out, by design:** the authors' optional "hard negative sampling" path (`manual_negative_sampling`, targeting a curated `dangerous_seffects_ids` list of severe side effects) and the `integrate=True` elementwise-product test-time variant — both auxiliary to the paper's Table 5 headline metric (tied to commented-out downstream isotonic-calibration analysis in the authors' own `graph.py`, not the reported AUC/AUPRC/AP@50), same category of decision as Milestone 9e's scoped-out 10-fold protocol.

**Training result:** the model trained stably by the usual signal (loss decreased monotonically, early-stopped cleanly at epoch 8 of 10, best epoch 6, no instability of the kind seen in the MLP path's §1.8/M9d ablation) — full history in `outputs/polyllm/gnn/training_history.csv`.

**Test-set result — does NOT reproduce the paper's reported performance:**

| Metric | Paper (DeepChem ChemBERTa row) | Reproduction | Std devs from paper |
|---|---|---|---|
| AUC | 0.9228 ± 0.0039 | **0.4113** | **−131.1** |
| AUPRC | 0.8944 ± 0.0025 | **0.5297** | **−145.9** |
| AP@50 | 0.9599 ± 0.0044 | **0.9507** | **−2.1** |

AUC below 0.5 (worse than random) is the key signal, not just a large gap. Confirmed via diagnostics that this is not a train/test generalization issue: the same near-chance behavior (AUC 0.493) appears when evaluating the trained model on its *own training* supervision edges.

**Root-cause investigation (code-level bugs ruled out, not assumed away):**
1. **Parameter updates** — compared the trained checkpoint against a freshly-initialized model, layer by layer. Every parameter group changed (embeddings, linear projections, all three `GraphConv` layers). Gradients reach every layer; this isn't a frozen-parameter bug.
2. **BatchNorm train/eval mismatch** — re-ran inference with BatchNorm forced into train-mode (live batch statistics) instead of eval-mode (running statistics, which had drifted to an extreme running variance of ~14,461 on `bn1`). AUC was unchanged (0.493 either way). Ruled out as the (sole) cause.
3. **Message-passing wiring** — compared node embeddings computed with the real graph vs. an empty edge set. Message passing is unambiguously active (large, non-trivial difference) — in fact so active that full-graph (non-mini-batched) message passing pushes activation std to ~208,610, since side-effect nodes average ~4,700 edges each and `GraphConv`'s sum-aggregation (confirmed faithful to the paper's Eq. 1, §above) has no degree normalization.
4. **`negative_sampling(force_undirected=True)` on a bipartite graph** — inspected PyG's actual source (`torch_geometric/utils/_negative_sampling.py`). When `num_nodes` is a tuple (our bipartite case, and the authors' too), the function *silently forces `force_undirected=False`* regardless of what's passed — so this argument, present in both our code and the authors', has zero effect. Confirmed the underlying bipartite index flatten/unflatten math (`edge_index_to_vector`/`vector_to_edge_index`) is a standard, correct row-major encoding — no bug there. A real, verified quirk (present in the authors' code too), but a dead end for the inversion.
5. **Sign/orientation check** — computed `roc_auc_score(y, -logits)` alongside the normal direction: **0.588 vs. 0.412**. Label alignment was independently spot-checked directly against the true global positive-edge set (5 positive + 5 negative test edges, all correctly assigned) — ruling out a label-swap bug. The inversion is real but not explained by mislabeling.
6. **Decomposing the dot-product decoder** — scored test edges using only embedding *magnitude* (`‖pdrugs_emb‖ × ‖seffect_emb‖`, discarding direction): **AUC = 0.821**, close to the paper's target. Scored using only *direction* (cosine similarity, discarding magnitude): **AUC = 0.507**, uninformative, with near-identical mean cosine similarity for positive and negative edges alike (−0.0595 vs. −0.0590). Mechanistically: the model learned a real, useful magnitude signal (larger embeddings ≈ more likely a true association — consistent with a per-degree-decile breakdown showing true positive rate climbing from 13% to 89% across the seffect-degree range) but never learned meaningful angular/relational structure. Since dot product = magnitude × cos(angle), and cos(angle) sits at a roughly constant *negative* value regardless of the true label, it acts as a near-uniform sign-flip on top of the (otherwise useful) magnitude signal — precisely inverting the ranking the magnitude alone would have produced correctly.
7. **Tested whether a cosine-similarity decoder (normalizing both embeddings before the dot product, with a learned scale) fixes it — it does not.** Retrained from scratch with the identical recipe, changing only the decoder. Result: AUC = 0.4143 (unchanged), AP@50 = 0.0000 (worse — none of the top-50 ranked test predictions out of 855,448 were true positives). This is an important negative result: it refutes magnitude/angle decomposition as the *root* cause (finding 6 was an accurate description of how the dot-product model fails, not the underlying reason) — forcing the model to rely purely on direction still produced a badly broken result, pointing the root cause back upstream into what the `GraphConv` encoder learns during training, not the choice of decoder.

**Independent confirmation this failure mode is real and not reproduction-specific:** the paper's own §4.1 ablation states — "We also trained the GNN model with zero-filled features (size 200) to evaluate the impact of informative and meaningful embeddings. The results show that zero features performed the worst, **with an AUC of 0.07**" — i.e. the authors' own model, in their own hands, produced an AUC far below random under a degraded configuration. This is strong independent evidence that this exact architecture (`GraphConv` + raw dot-product decoder + BCE) is capable of collapsing to badly sub-random AUC, not something specific to a translation error in this reproduction.

**Two further paper-text confirmations, no new leads:**
- §2.6.2 states the validation/test message-passing protocol precisely: "during validation, all training edges... are used to predict validation edges... in testing, the model uses all training and validation edges to predict test edges." Checked directly against the Milestone 10c graph audit — already matches exactly (val message-passing edges = 3,661,031 = train's full edge set; test message-passing = 4,118,659 = val's full edge set) — `RandomLinkSplit`'s default behavior already does this correctly.
- §3.5.2 confirms lr=0.01 and dropout=0.8 were the paper's own empirically-selected optimal values for the GNN (not arbitrary) — both already matched exactly in this reproduction. Rules out "wrong hyperparameter" as an explanation.

**One newly-discovered, unresolved environment difference:** §3.1 states the paper's GNN experiments used `PyTorch 2.4.1` + `PyTorch Geometric 2.6.1`; this reproduction uses `PyTorch 2.12.1` + `PyTorch Geometric 2.8.0.post1`. Plausible that `GraphConv`/`to_hetero`/`negative_sampling` internals differ subtly across a version gap this large, but not independently verified (would require a separate, older environment to test) and not considered a likely primary explanation given the magnitude of the observed gap.

**Status: investigated in depth twice (2026-08-11 and 2026-08-13), root cause narrowed considerably, gap remains unresolved. CLOSED/FROZEN as of 2026-08-13.** Per explicit user instruction, no hyperparameter/architecture search was performed in either round — every experiment (BN-mode check, sign check, magnitude/angle decomposition, cosine-decoder retrain, and the 2026-08-13 round below) is a targeted diagnostic reusing the existing trained checkpoint or an isolated single retrain/rerun to test one specific hypothesis, consistent with this repository's permanent "no hyperparameter search" constraint and the same controlled-ablation pattern as §1.8/§1.9.

**2026-08-13 follow-up investigation (full detail: `notes/gnn_gap_diagnostics.md`)** — a 7-part controlled-diagnostic protocol, reopened at explicit user request:
1. **Node/edge/embedding alignment**: 20 sampled edges, 9 independent checks each (including an independent SHA-256 re-hash of the BERT embedding source name) — all passed. No alignment bug.
2. **Layer order**: static + runtime forward-hook trace — confirmed exactly `conv1→bn1→leaky_relu→dropout→conv2→bn2→leaky_relu→dropout→conv3→lin1`, nothing after `lin1`. No transcription bug.
3. **RNG reseeding mechanism**: confirmed the authors' per-forward `set_seed(12)` makes dropout identical across every call (as suspected), plus a newly-found second quirk — their `torch.manual_seed(42)` before negative sampling is very likely a no-op, since PyG's `negative_sampling()` draws from Python's `random` module, not torch's RNG.
4. **Per-epoch representation diagnostics** (new, previously only checked at the final checkpoint): the magnitude/degree signal is present **at random initialization**, before any training (norm-product AUC≈0.845 at epoch 0) — it is architectural (unnormalized `GraphConv` on a heavy-tailed degree distribution), not learned. The dot-product inversion happens **abruptly between epoch 1 and 2**, exactly when mean cosine similarity flips from positive to a near-uniform negative value for both classes; from that point, `dot_AUC ≈ 1 − norm_AUC` almost exactly.
5. **Semantic vs. node-ID decomposition**: both the content (ChemBERTa/BERT) branch and the content-free node-identity-embedding branch independently exhibit the same magnitude signal at init and the same inversion during training (identity branch inverts one epoch earlier) — confirms the effect is structural, not content-driven.
6. **RNG-regime downstream retrain**: reproducing the authors' literal RNG-reseeding behavior for 3 epochs produces the **same inversion** as this repo's actual (non-reseeding) behavior — rules out the RNG deviation as the cause. (Side finding: training trajectories are not fully run-to-run reproducible even at a fixed seed, likely due to `LinkNeighborLoader` internals and/or float32 summation order at the extreme activation magnitudes this architecture reaches — an open caveat on every single-run number in this investigation.)
7. **Legacy PyTorch 2.4.1/PyTorch Geometric 2.6.1 environment**: required one non-computational compatibility shim (`to_hetero`'s FX codegen hits an `AssertionError` under torch 2.4.1 when tracing a `from __future__ import annotations` + type-hinted `forward()`; fixed by resolving the postponed annotations via `typing.get_type_hints()` before tracing — verified to have zero effect on any tensor computation). Result: AUC 0.4113→**0.6009**, AUPRC 0.5297→**0.6866** — a real, substantial improvement, the single largest effect of any one variable tested in this whole investigation — but still **−82.5 std devs** from the paper's mean AUC (0.9228). The software-environment gap is a genuine partial contributor, not the full explanation.

**Final honest statement (2026-08-13): GNN performance reproduction unresolved despite architecture, data, and RNG fidelity. Software-environment version fidelity produced a partial, non-trivial improvement but did not reach the paper's reported performance — the version gap narrows the gap, it does not close it.** This branch is now frozen: no further tuning, ablation, or environment work without a new, specific hypothesis and an explicit request.

**Evidence reference:** `src/polyllm/graph/build_graph.py`, `src/polyllm/models/gnn.py`, `src/polyllm/train_gnn.py`, `src/polyllm/evaluate_gnn.py`; `data/graph/gnn_link_split.pt`; `outputs/polyllm/gnn/{training_history.csv,training_config.json,test_metrics.json,checkpoints/best_model.pt}`; `outputs/polyllm/side_effect_embedding_audit.json`, `outputs/polyllm/gnn_graph_audit.json`; test files `tests/test_build_graph.py`, `tests/test_gnn_model.py`, `tests/test_train_gnn.py`, `tests/test_evaluate_gnn.py`, `tests/test_generate_side_effect_embeddings.py`. Diagnostic experiments (sign check, magnitude/angle decomposition, cosine-decoder retrain, degree-only baselines) were run ad hoc during the investigation and are not saved as separate scripts; exact commands are reproducible from this section's description if needed.

**Consequence:** The paper's strongest reported results (GNN path) are not achieved by this reproduction, despite a mechanically-verified, formula-faithful implementation. The MLP-vs-Morgan comparison (§0–§1.9) remains the reliable, validated part of this reproduction; the GNN result should be read as "faithfully implemented, training does not currently converge to the paper's reported performance level" rather than either "reproduced" or "not attempted."

---

### 1.5 Morgan fingerprint baseline not in the paper

The paper does not report any Morgan fingerprint MLP results. The reproduction adds this as a conventional chemoinformatics baseline. This is an addition, not a gap — documented here for completeness.

**Evidence:**  
Paper: Morgan fingerprints are mentioned once in passing (Rogers and Hahn, 2010 citation) but no Morgan MLP is reported in any table.  
Reproduction: `notes/morgan_baseline_findings.md`, `outputs/baseline/morgan/test_metrics.json`.

---

### 1.6 Only one embedding backbone reproduced; paper's broader model comparison out of scope

| Property | Paper | Reproduction |
|---|---|---|
| Embedding backbones evaluated | BERT, Sentence-BERT (Reimers and Gurevych, 2019), fine-tuned ChemBERTa (Xu et al., 2023), OpenAI GPT, Mol2vec (Jaeger et al., 2018), Doc2vec (Le and Mikolov, 2014) — at least these six, in addition to whatever produces the "DeepChem ChemBERTa" row in Table 5 | One: frozen `DeepChem/ChemBERTa-77M-MLM` only |

**Evidence:**  
Paper (quoted by user, exact section not yet located): "BERT, Sentence-BERT (SBERT) Reimers and Gurevych (2019), Fine-tuned ChemBERTa Xu et al. (2023), OpenAI's GPT, Mol2vec Jaeger et al. (2018), and Doc2vec Le and Mikolov (2014)."

Paper : "Mol2vec... produces substructure based embeddings that emphasize local chemical environments. Additionally, we use the Application Programming Interface (API) provided by OpenAI to encode drug representations using their advanced and most powerful third-generation embedding model. The text-embedding-3-small version, with an embedding dimension..." [quote truncated by user].

Paper: "we employ a fine-tuned variant of ChemBERTa developed by Xu et al. (2023), which uses SimCSE, a contrastive learning approach. This fine-tuning process is conducted using the GuacaMol benchmark dataset Brown et al. (2019)... with dropout introduced as noise to further enhance embedding quality." This confirms "Fine-tuned ChemBERTa" is not the paper's own in-house fine-tuning of the base checkpoint — it is a separately published encoder (Xu et al., 2023) trained via SimCSE contrastive learning on GuacaMol, structurally unrelated to this reproduction's plain frozen encoder.

Reproduction: confirmed by exhaustive search — no reference to `SBERT`, `Sentence-BERT`, `Mol2vec`, `Doc2vec`, `GPT`, `sentence_transformers`, `openai`, or `text-embedding-3` anywhere in `src/`. Only encoder implemented: `src/polyllm/features/generate_chemberta_embeddings.py`, targeting the frozen `DeepChem/ChemBERTa-77M-MLM` checkpoint (see §2.1).

**Consequence:** The reproduction cannot speak to how the paper's chosen encoder (whichever produces the Table 5 "DeepChem ChemBERTa" number) compares against the other encoders the paper evaluated. Our single implemented model does not correspond to any of the six listed alternatives — it most likely targets the paper's separate, primary "DeepChem ChemBERTa" configuration instead (see §2.1, now largely corroborated by the paper's own description of that base checkpoint).

**Classification:** Confirmed scope limitation — not a bug, a deliberate reduction of the paper's full comparison to a single backbone (consistent with `notes/polyllm_reproduction_scope.md`'s single-embedding-model plan).

---

### 1.7 LeakyReLU negative slope: 0.1 (paper) vs 0.01 (reproduction) — RESOLVED 2026-08-09 (Milestone 9c), BOTH MODELS RETRAINED

| Property | Paper | Reproduction (current) | Reproduction (archived, pre-2026-08-09) |
|---|---|---|---|
| Activation function | Leaky ReLU (Xu et al., 2015) | Leaky ReLU (`torch.nn.LeakyReLU`) | Leaky ReLU |
| Negative slope | **0.1** | **0.1** | 0.01 |

**Evidence:**
Paper (quoted directly by user from Section 2.2): "Additionally, we evaluated several activation functions and selected the Leaky Rectified Linear Unit (Leaky ReLU) Xu et al. (2015) with a negative slope of 0.1 for each hidden layer. Leaky ReLU was selected for its ability to address the vanishing gradient problem, which is a common issue in activations functions such as tanh or sigmoid, and also for its capacity to enhance learning stability." The same excerpt also confirms the three hidden-layer sizes (512, 1024, 2048), BatchNorm after the first hidden layer only, and dropout 0.2 — all already matched in the reproduction (see §0 and the architecture table elsewhere in this document).

**Root cause:** The reproduction originally adopted PyTorch's own out-of-the-box default for `nn.LeakyReLU` (0.01) rather than a value derived from the paper — confirmed by the code comments crediting "PyTorch default," not the paper, as the source of this number. The paper's specific value (0.1) was not known at the time this code was first written.

**Fix and retrain (2026-08-09, Milestone 9c):**
- `src/polyllm/models/mlp.py`: `MultilabelMLP.NEGATIVE_SLOPE` changed from `0.01` to `0.1`.
- `src/polyllm/train_morgan_baseline.py` and `src/polyllm/train_chemberta_mlp.py`: `DEFAULT_CONFIG["leaky_relu_negative_slope"]` changed from `0.01` to `0.1`.
- Pre-fix (slope=0.01) outputs backed up in full to `outputs/baseline/morgan_slope0.01_backup/` and `outputs/polyllm/chemberta_slope0.01_backup/` before retraining.
- Both `train_morgan_baseline.py --overwrite` and `train_chemberta_mlp.py --overwrite` re-run from scratch (100 epochs each, CPU), followed by both `evaluate_*.py --overwrite` scripts, `compare_models.py`, and `recompute_ap_at_k.py`. All of `outputs/baseline/morgan/`, `outputs/polyllm/chemberta/`, and `outputs/comparison/*` now reflect slope=0.1.

**Result — training dynamics, both models:**

| | Morgan (slope=0.01 → 0.1) | ChemBERTa (slope=0.01 → 0.1) |
|---|---|---|
| Best epoch | 99 → 91 | 99 → 99 |
| Best val macro AUPRC | 0.4966 → 0.4805 | 0.4519 → 0.4381 |
| Selected threshold | 0.31 → 0.28 | 0.28 → 0.26 |

**Result — test set, both models:**

| Metric | Morgan 0.01 | Morgan 0.1 | ChemBERTa 0.01 | ChemBERTa 0.1 | Paper (ChemBERTa) |
|---|---|---|---|---|---|
| Macro AUROC | 0.8980 | 0.8986 | 0.8881 | 0.8880 | 0.8859 ± 0.0019 |
| Macro AUPRC | 0.4969 | 0.4819 | 0.4490 | 0.4328 | 0.3979 ± 0.0079 |
| Micro AUPRC | 0.5784 | 0.5549 | 0.5329 | 0.5092 | not reported |
| Micro F1 (sel. thr.) | 0.5225 | 0.5135 | 0.4967 | 0.4889 | not reported |
| AP@50 (paper axis, §1.3) | 0.8950 | 0.8716 | 0.8491 | 0.8146 | 0.7557 ± 0.0120 |

**Consequence — the interesting part:** fixing the slope did **not** simply make the reproduction "better." It slightly *reduced* both models' own absolute AUPRC/F1/AP@50 (e.g. Morgan macro AUPRC 0.4969 → 0.4819), including Morgan, which has no paper counterpart to move toward — evidence the effect is a general property of the slope=0.1 training dynamics on this task, not something specific to matching the paper. At the same time, it moved ChemBERTa's two metrics that previously showed the largest unexplained gaps *closer* to the paper: macro AUPRC's gap narrowed from 6.47 to 4.42 std devs, and AP@50's (paper-axis, §1.3) gap narrowed from 7.78 to 4.91 std devs. AUROC, which was already close, stayed close (1.16 → 1.11 std devs). This is genuine evidence — not proof — that the paper-specified hyperparameters explain part of the previously "unconfirmed cause" excess noted in §2.6, without fully explaining it. The residual gap (~4.4–4.9 std devs on both metrics) remains open; loss function (§2.3, focal vs. BCE) and LR schedule are the next candidate causes.

**Evidence reference:** `src/polyllm/models/mlp.py`; `outputs/polyllm/chemberta/training_config.json` and `outputs/baseline/morgan/training_config.json` now record `leaky_relu_negative_slope: 0.1`; `outputs/comparison/paper_comparison_audit.json` `retrain_event_2026_08_09` for the full before/after audit trail including SHA-256 hashes of both the new and archived checkpoints/predictions.

---

### 1.8 Loss function (BCE vs. Focal) and LR schedule alignment attempt — tested and reverted (Milestone 9d, 2026-08-09)

**Research question (user-specified):** "When I align the remaining training configuration more closely with PolyLLM, do my results move toward the authors' reported values?" This tests the two candidates left open at the end of §1.7 and named in §2.3/§2.6: loss function (paper possibly Binary Focal Cross-Entropy vs. reproduction's BCE) and learning-rate schedule (paper possibly ~0.005 with 0.96 exponential decay vs. reproduction's fixed 0.001).

**Important caveat before the result:** unlike the LeakyReLU slope in §1.7, neither the focal-loss claim nor the lr=0.005/decay=0.96 claim has ever been confirmed against the paper's primary text in this repository — both originated from an AI-generated summary of the paper. This experiment tests "what happens if we adopt those unconfirmed hypothesized values," not "what happens when we match the paper exactly."

**Scope:** ChemBERTa only (Morgan has no paper target to test against). Same data, same frozen ChemBERTa-77M-MLM embeddings, same `MultilabelMLP` architecture (including the §1.7 slope=0.1 fix), same 80/10/10 split, same seed=42, same batch size as Milestone 7/9c. Exactly two changes:
1. Loss: `BCEWithLogitsLoss` → `BinaryFocalLossWithLogits(gamma=2.0, alpha=0.25)` — new module `src/polyllm/losses.py`, 13 unit tests in `tests/test_losses.py`. gamma/alpha are literature-standard defaults (Lin et al., 2017), not paper-derived.
2. Learning rate: fixed `0.001` → `0.005` initial, with `torch.optim.lr_scheduler.ExponentialLR(gamma=0.96)` stepped once per epoch. Per-epoch stepping is an assumption; the paper (per the unconfirmed summary) names a decay rate but not a decay-step granularity.

Written to a **new, separate** output directory (`outputs/polyllm/chemberta_focal_lrdecay/`) via new scripts `train_chemberta_mlp_focal_lrdecay.py` / `evaluate_chemberta_mlp_focal_lrdecay.py`. The Milestone 7/9c "official" `outputs/polyllm/chemberta/` directory was never touched.

**Result: training instability, not improvement.**

| Epoch | train_loss | val_loss | val macro AUPRC |
|---|---|---|---|
| 1–6 | 0.0321 → 0.0199 (smooth decrease) | 0.0217 → 0.0204 | 0.1287 → **0.2149** (best) |
| 7 | **0.2115** (10× spike) | 0.0284 | 0.0906 (crash) |
| 8–16 | 0.0280 → 0.0219 (slow recovery) | 0.0231 → 0.0211 | 0.0997 → 0.1462 (never regains epoch-6 peak) |

Early stopping (patience=10) correctly triggered at epoch 16; best checkpoint is epoch 6, far short of a converged model.

**Test-set result — far worse than both the paper and the run it was meant to improve on:**

| Metric | Paper | M7/9c (unchanged) | M9d (Focal + LR decay) | M9d std devs from paper |
|---|---|---|---|---|
| Macro AUROC | 0.8859 ± 0.0019 | 0.8880 | **0.7643** | −64.0 |
| Macro AUPRC | 0.3979 ± 0.0079 | 0.4328 | **0.2085** | −24.0 |
| AP@50 (paper axis) | 0.7557 ± 0.0120 | 0.8146 | **0.4654** | −24.2 |

**Conclusion:** for this specific implementation, aligning loss and LR schedule to the hypothesized paper values did **not** move results toward the paper — it destabilized training and produced the worst result of any experiment in this repository. This does **not** prove the paper's actual configuration (whatever it precisely is) is wrong; it shows this reproduction's particular implementation of that hypothesis (assumed per-epoch decay granularity, literature-default focal gamma/alpha, no gradient clipping, no warmup) is unstable in this codebase's training loop. The single most likely cause is the 5× higher initial learning rate, not focal loss itself — but this has not been isolated by a controlled sub-ablation.

**The Milestone 7/9c configuration (BCE, fixed lr=0.001, slope=0.1) remained the reproduction's official ChemBERTa result at the time.** This closed the "reconcile remaining hyperparameters" line of investigation for then, per the user's own framing ("after that I would consider the MLP definitively finished"). **Superseded 2026-09-01** — see §1.9's redesignation note: Milestone 9e is now official.

**Candidate follow-ups, not run (out of scope for now):**
- Isolate focal loss alone at lr=0.001 (unchanged) — is focal loss itself destabilizing, independent of the LR change?
- Isolate lr=0.005+decay alone with plain BCE (unchanged) — is the LR change alone destabilizing, independent of focal loss?
- Add gradient clipping (`torch.nn.utils.clip_grad_norm_`), a standard stabilizer for higher learning rates, and retry.
- Try an initial LR closer to 0.001 with the 0.96 decay schedule to isolate whether the instability is purely a magnitude effect.

**Evidence reference:** `outputs/comparison/paper_comparison_audit.json` key `ablation_event_2026_08_09_milestone_9d`; `outputs/polyllm/chemberta_focal_lrdecay/{training_history.csv,test_metrics.json,ap_at_k_recompute.json}`; `outputs/comparison/paper_comparison.csv` rows tagged "Milestone 9d".

---

### 1.9 Faithful-training ablation (Milestone 9e, 2026-08-09) — closest agreement with the paper in this repository

**Trigger:** after §1.8's instability, the PolyLLM authors' actual source was located and inspected directly (`src/mlp/MLPModel.py`, `src/mlp/mlp.py`, `src/params.py` at `github.com/sadrahkm/PolyLLM`, environment pinned to `keras=3.6.0`, `tensorflow=2.17.0` via their `environment.yml`), rather than continuing to guess at unconfirmed hyperparameters.

**What the authors' source actually shows** (all directly quoted/verified, not inferred):
- Loss: `losses.BinaryFocalCrossentropy(label_smoothing=0.2)`, every other Keras default left unchanged — critically `apply_class_balancing=False`, so **no alpha weighting is ever applied**, despite `alpha=0.25` being Keras' nominal default. §1.8's M9d run had used alpha=0.25 (active) and no label smoothing — the *opposite* configuration on both axes.
- Batch size: never set explicitly (`# batch_size=128` is commented out in `mlp.py`) → Keras `model.fit()` default of **32**. M9d used 256.
- LR schedule: `keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=0.005, decay_steps=1000, decay_rate=0.96, staircase=True)`, evaluated **per optimizer step** (mini-batch), not per epoch. M9d applied the 0.96 decay once per **epoch** via `torch.optim.lr_scheduler.ExponentialLR` — a fundamentally different mechanism, and even a correct per-step implementation at M9d's batch size (256, ~199 steps/epoch) would under-decay by roughly 8× relative to the authors' true schedule at batch=32 (~1607 steps/epoch).
- Not reproduced by design: the authors' sequential 10-fold protocol (single model/optimizer object reused across all 10 folds via repeated `.fit()` calls, `patience=0`, `monitor='val_AUC'`, `restore_best_weights=False`, 10 outer iterations averaged) — explicitly out of scope per the user's framing for this experiment, which isolates loss + batch size + LR schedule only.

**Implementation (all new/backward-compatible, nothing existing modified):**
- `src/polyllm/losses.py`: added `label_smoothing` parameter to `BinaryFocalLossWithLogits` (default 0.0 — Milestone 9d's calls are numerically unchanged). 11 new unit tests (`tests/test_losses.py::TestLabelSmoothing`) verify hand-derived values at target=1/p=0.9, target=0/p=0.1, harder examples at p=0.3/0.7, correct smoothing (1→0.9, 0→0.1), confirmed no-alpha-weighting when disabled, and correct gamma=2 focusing.
- `src/polyllm/lr_schedules.py` (new): `keras_exponential_decay_lr(step, initial_lr, decay_steps, decay_rate, staircase)`, the exact Keras `ExponentialDecay` formula. 13 unit tests (`tests/test_lr_schedules.py`) verify exact values at steps 0, 999, 1000, 1999, 2000 and staircase/monotonicity properties.
- `src/polyllm/train_chemberta_mlp_faithful.py` / `evaluate_chemberta_mlp_faithful.py` (new): batch_size=32, `Adam(lr=0.005, betas=(0.9,0.999), eps=1e-7)` (Keras' epsilon, not PyTorch's 1e-8 default), `BinaryFocalLossWithLogits(gamma=2.0, alpha=-1 [disabled], label_smoothing=0.2)`, LR set from `keras_exponential_decay_lr` before every `optimizer.step()` call. Early stopping **intentionally kept** at the existing methodology (patience=10, monitor=`val_macro_auprc`, best checkpoint restored) rather than the authors' patience=0/val_AUC/restore-last, per explicit user instruction to isolate only loss+batch+LR. Written to a third, separate directory, `outputs/polyllm/chemberta_faithful_training/` — neither `outputs/polyllm/chemberta/` (M7/9c) nor `outputs/polyllm/chemberta_focal_lrdecay/` (M9d) were touched.

**Result — completely stable, and the closest match to the paper in this repository:**

Training ran the full 100 epochs with no early stop, no loss spikes, monotonic (with normal minor noise) improvement in val macro AUPRC from 0.1266 (epoch 1) to a best of 0.3990 (epoch 94) — a direct contrast to M9d's 10× loss spike and crash at epoch 7, under the *same* gamma=2.0.

| Metric | Paper | M7/9c (BCE) | M9d (approx. focal) | **M9e (faithful)** | M9e std devs from paper |
|---|---|---|---|---|---|
| Macro AUROC | 0.8859 ± 0.0019 | 0.8880 | 0.7643 | **0.8847** | **−0.71** |
| Macro AUPRC | 0.3979 ± 0.0079 | 0.4328 | 0.2085 | **0.3943** | **−0.46** |
| AP@50 (paper axis) | 0.7557 ± 0.0120 | 0.8146 | 0.4654 | **0.7548** | **−0.07** |

All three headline metrics land within **less than 1 standard deviation** of the paper's reported 10-fold mean. Macro AUPRC (0.3943) falls **inside** the paper's own ±1 std band (0.3900–0.4058). AP@50 (0.7548 vs. 0.7557) is essentially an exact match — the single closest metric-to-paper agreement found anywhere in this reproduction, by a wide margin.

**Interpretation — evidence, not proof, of the LR-schedule/batch-size hypothesis:** gamma=2.0 was held constant between M9d (unstable) and M9e (stable); stability was restored purely by correcting batch size, LR mechanism, alpha, and label smoothing together. This is *consistent with and supportive of* the hypothesis (from §1.8's ranked-causes analysis) that the LR-schedule/batch-size mismatch was the primary driver of M9d's instability, especially combined with the mechanistic reasoning that a sudden one-epoch spike is a classic LR-instability signature, not a signature of static loss-formula differences. It is **not a strict isolation**: three things changed simultaneously between M9d and M9e (alpha/label-smoothing, batch size, LR mechanism), so the individual contribution of each cannot be attributed without further controlled sub-ablations (not run — see candidates in §1.8, still applicable).

**Notably, M9e's absolute metrics are slightly *below* the official M7/9c (BCE) run** (e.g. macro AUPRC 0.3943 vs. 0.4328) — read as M9e reproducing the paper's actual reported performance *level* almost exactly, while M7/9c happens to somewhat outperform what the paper itself reports. Underperforming M7/9c here is evidence of fidelity, not of a worse model.

**Status (original, 2026-08-09): diagnostic ablation, not a redesignation.** Per explicit user instruction at the time, `outputs/polyllm/chemberta/` (Milestone 7/9c: BCE, fixed lr=0.001, slope=0.1) remained the reproduction's **official** ChemBERTa MLP result, despite M9e's closer paper agreement. The remaining ~0.5–0.7 std dev gap under the faithful configuration is most plausibly attributable to the still-unreplicated single-split-vs-10-fold-CV protocol difference (§1.2) rather than to any further hyperparameter tuning.

**REDESIGNATED OFFICIAL — 2026-09-01.** Per explicit user decision (made while drafting the reproduction paper), the above status is superseded: **`outputs/polyllm/chemberta_faithful_training/` (Milestone 9e) is now the official ChemBERTa MLP result.** Rationale given: a paper reproduction should report the paper-matching configuration, not the historically-first one chosen before the paper's actual loss/LR was confirmed. `outputs/polyllm/chemberta/` (Milestone 7/9c) is retained on disk unmodified, as a secondary/historical reference — it is no longer the number to cite as "the reproduction." This redesignation was made together with the matching Morgan swap (Milestone 9f now official for the same reason) — see `notes/morgan_vs_chemberta_faithful_ablation.md`, redesignated the same day. Both models still beat the same relative comparison under this recipe (Morgan > ChemBERTa on all three headline metrics), so the Milestone 8 conclusion is unaffected by this redesignation.

**What is still not replicated, even under the faithful configuration:** the authors' sequential 10-fold-with-weight-carryover protocol, `patience=0`, `monitor='val_AUC'`, `restore_best_weights=False`, 10 outer iterations averaged to a mean±std, He-normal initialization (vs. PyTorch's default), and the authors' explicit-sigmoid-output-then-probability-based loss (vs. our logits-based internal sigmoid — mathematically reconcilable in exact arithmetic, not bit-verified against Keras' numerical-stability epsilon handling). The 963-vs-964-label dataset-level deviation (§1.1) also remains unresolved.

**Evidence reference:** `outputs/comparison/paper_comparison_audit.json` key `faithful_training_event_2026_08_09_milestone_9e`; `outputs/polyllm/chemberta_faithful_training/{training_history.csv,test_metrics.json,ap_at_k_recompute.json}`; `outputs/comparison/paper_comparison.csv` rows tagged "Milestone 9e"; `src/polyllm/lr_schedules.py`, `tests/test_lr_schedules.py`, `tests/test_losses.py::TestLabelSmoothing`.

---

## 2. Possible Deviations (Unconfirmed)

### 2.1 ChemBERTa model revision

**Confirmed:** The paper's base checkpoint name matches the reproduction's. Paper (quoted by user): "we use the ChemBERTa model pre-trained on the Masked Language Modeling (MLM) task using a dataset size of 77 million samples (ChemBERTa-77M-MLM)." Reproduction: `DeepChem/ChemBERTa-77M-MLM`, `notes/chemberta_embedding_findings.md:26`. Exact name, pretraining objective (MLM), and dataset size (77M) all match.

**Still unconfirmed — exact revision hash:** The reproduction pins Hugging Face revision `ed8a5374f2024ec8da53760af91a33fb8f6a15ff` (SHA resolved via `huggingface_hub.model_info()`). The paper does not specify a revision hash, so if the checkpoint was updated on the Hub after the paper was written, weights could differ slightly. The small AUROC agreement (+0.0022) makes a major weights difference unlikely but cannot be ruled out.

**Largely corroborated — frozen vs. fine-tuned regime:** The paper's own text distinguishes the base `ChemBERTa-77M-MLM` ("we use...") from a separately-described "fine-tuned variant of ChemBERTa developed by Xu et al. (2023)" trained via SimCSE on GuacaMol (see §1.6) — implying the base checkpoint is used directly (consistent with this reproduction's frozen-encoder approach) while the SimCSE-fine-tuned version is a distinct, separately-evaluated alternative. The paper never uses the word "frozen" explicitly, so this remains an inference rather than a direct confirmation, but it is now better supported than before.

**Evidence reference:** `notes/chemberta_embedding_findings.md`; §1.6 of this document.

---

### 2.2 ChemBERTa pooling method

**Claim:** The reproduction uses attention-mask-aware mean pooling over the final hidden layer, including the [CLS] and [SEP] special tokens.

**Why unconfirmed:** The paper does not describe how the token-level ChemBERTa output is reduced to a fixed-length embedding. Mean pooling is the most common choice for frozen encoders; CLS pooling is another option. The result would differ depending on which is used.

**Evidence reference:** `notes/chemberta_embedding_findings.md`.

---

### 2.3 Optimizer and hyperparameters

**Claim:** The reproduction uses Adam, lr=0.001, batch=256, max_epochs=100, patience=10.

**Why unconfirmed:** The paper does not specify the optimizer, learning rate, batch size, or early-stopping configuration for the MLP. These choices are standard but the paper does not confirm them.

**Note:** The activation function and its negative-slope parameter, previously bundled into this general "unspecified hyperparameters" claim, are now confirmed and moved to their own entry — see §1.7 (confirmed 0.1 vs 0.01 mismatch). Loss function and LR schedule were tested directly 2026-08-09 (§1.8) — result was training instability, not improvement; the reproduction's original Adam/lr=0.001/BCE configuration remains in place.

**Evidence reference:** `outputs/polyllm/chemberta/training_config.json`.

---

### 2.4 Threshold selection method

**Claim:** The reproduction selects thresholds on the validation set by maximizing micro F1 over a 0.05–0.95 grid (step 0.01). Selected thresholds: Morgan=0.31, ChemBERTa=0.28.

**Why unconfirmed:** The paper does not describe a threshold selection procedure. The paper may use a default threshold (e.g., 0.50) or a different method. This affects all F1 metrics.

**Evidence reference:** `outputs/baseline/morgan/validation_threshold.json`, `outputs/polyllm/chemberta/validation_threshold.json`.

---

### 2.5 SMILES source and canonicalization

**Claim:** The reproduction uses PubChem-sourced SMILES canonicalized with RDKit (Milestone 2). The paper does not describe its SMILES source.

**Why unconfirmed:** If the paper uses a different SMILES source (e.g., directly from STITCH or another database), the SMILES strings and resulting embeddings may differ.

**Evidence reference:** `notes/smiles_mapping_findings.md`.

---

### 2.6 AUPRC higher than paper (cause partially explained 2026-08-09; investigation closed for now)

**Original claim (pre-2026-08-09):** The reproduction's macro AUPRC (0.4490) was 0.0511 above the paper's 10-fold mean (0.3979), which is 6.5 standard deviations.

**Why possible:** The difference is substantially larger than the fold-to-fold variation reported by the paper. The label count difference (963 vs 964) is not a plausible explanation — changing one label can alter a macro average by at most approximately 0.001. Possible causes considered: differences in dataset preprocessing, split or cross-validation methodology, SMILES processing, pooling and truncation, model selection, metric implementation, training variability, and — as of this update — hyperparameter fidelity to the paper.

**Update 2026-08-09 (Milestone 9c):** one candidate cause has now been tested directly. The reproduction's LeakyReLU negative slope was corrected from 0.01 (PyTorch's own default) to the paper's specified 0.1 (§1.7), and both MLPs were retrained under the corrected value. Result: macro AUPRC moved from 0.4490 to **0.4328**, narrowing the gap from 6.47 to **4.42 standard deviations** — roughly a third of the original excess. This is evidence, not proof, that hyperparameter fidelity was part of the cause. It is not the whole cause: a ~4.4 std dev gap remains. Remaining untested candidates: loss function (§2.3 — paper possibly Binary Focal Cross-Entropy vs. reproduction's plain BCE, unconfirmed against primary source), learning-rate schedule (possibly paper ~0.005 with exponential decay vs. reproduction's fixed 0.001), and the single-split-vs-10-fold-CV protocol difference (§1.2), which remains the most structurally significant unresolved difference.

**Update 2026-08-09 (Milestone 9d):** the two remaining candidates named above (loss function, LR schedule) were tested directly, together, the same day. Result: the attempt made every metric dramatically *worse*, not better — a training instability (10× loss spike, early stopping at epoch 16 with best epoch 6) drove macro AUPRC down to 0.2085, 24 standard deviations *below* the paper, far past the original 4.4 std dev excess in the opposite direction. See §1.8 for the full result. This does not identify the cause of the original 4.4 std dev gap — it only rules out "trivially raise loss/LR fidelity" as a fix, given this implementation's specific assumptions (per-epoch decay, literature-default focal gamma/alpha, no gradient clipping).

**Corrective action:** partially taken (§1.7 retrain, which helped) and partially attempted-and-reverted (§1.8 loss/LR ablation, which hurt) — then completed (§1.9, M9e, succeeded). Per the user's explicit direction at the time, this closed the "reconcile remaining MLP hyperparameters" investigation for then — the Milestone 7/9c configuration (BCE, fixed lr=0.001, slope=0.1) was retained as the reproduction's official result, and remaining unexplained variance (the ~4.4 std dev AUPRC gap under that configuration) was attributed most plausibly to the still-open single-split-vs-10-fold-CV protocol difference (§1.2) rather than to loss/LR. **Superseded 2026-09-01**: per a later explicit user decision (see §1.9), Milestone 9e (the faithful loss/LR/batch configuration) is now the official ChemBERTa result, and the sub-1-std-dev-remaining gap is attributed to the still-open 10-fold-CV difference.

---

## 3. Reproduction-Specific Improvements and Additions

These are features of the reproduction that go beyond the paper but are not deviations in a negative sense.

### 3.1 Morgan fingerprint MLP baseline

A fully trained ECFP4-style (radius=2, 2048-bit, includeChirality=True) MLP baseline was added to establish a conventional chemoinformatics reference point. The paper does not report this. See `notes/morgan_baseline_findings.md`.

### 3.2 Paired bootstrap statistical comparison

500 paired bootstrap resamples (seed=42) were performed over the test set to quantify sampling variability of the Morgan-minus-ChemBERTa difference. Morgan exceeded ChemBERTa in all 500 resamples for three metrics. Finite-sample one-sided upper bound: 1/501 ≈ 0.002. See `notes/model_comparison_findings.md` Section 6 and `outputs/comparison/comparison_audit.json`.

### 3.3 Explicit threshold selection with documentation

Thresholds were selected by maximising validation-set micro F1 and documented in `outputs/baseline/morgan/validation_threshold.json` and `outputs/polyllm/chemberta/validation_threshold.json`. The paper does not document threshold selection.

### 3.4 Per-label analysis

Individual AUROC, AUPRC, and F1 were computed for all 963 labels and compared across models. Win counts: Morgan leads on 831/963 labels for AUROC, 888/963 for AUPRC, 819/963 for F1. See `outputs/comparison/per_label_comparison.csv`.

### 3.5 Prediction agreement analysis

Thresholded agreement (95.06%), probability correlation (Pearson r=0.871, Spearman rho=0.893), and top-10 Jaccard similarity (mean=0.3493) were computed. These characterize how similarly the two models behave on individual predictions, beyond aggregate metric comparison. See `outputs/comparison/prediction_agreement.json`.

### 3.6 Complete artifact hashing and audit trail

SHA-256 hashes are recorded for all major intermediate and output artifacts across all milestones. Reproducibility of each step is documented in per-milestone findings notes and audit JSON files.

### 3.7 Micro AUPRC reported

The reproduction reports both macro and micro AUPRC. The paper reports only (macro) AUPRC. Micro AUPRC evaluates performance at the flattened label-instance level and is less sensitive to the performance distribution across labels.

### 3.8 Source-verified AP@k recomputation (Milestone 9b)

`average_precision_at_k_multi_label()` (`src/polyllm/metrics.py`) and `src/polyllm/recompute_ap_at_k.py` implement and run the authors' own reference AP@k function (a verbatim port of `average_precision_at_k_multi_label` from their public `src/functions.py` — authors not contacted) against the already-saved Milestone 6B/7 predictions, with no retraining and no modification of any existing artifact (verified by SHA-256 comparison before/after). This resolved §1.3 from a complete mismatch to a much narrower, directionally-explicable gap. See `outputs/baseline/morgan/ap_at_k_recompute.json` and `outputs/polyllm/chemberta/ap_at_k_recompute.json`.

---

## 4. Summary Table

| Deviation | Category | Impact on comparison |
|---|---|---|
| 963 vs 964 labels | Confirmed | Small; shifts macro metrics slightly |
| Single run vs 10-fold CV | Confirmed | Moderate; single-point vs distribution; largest remaining structural difference after §1.7 |
| AP@50 formula incompatible | Confirmed, then resolved (§1.3, Milestone 9b); further narrowed by retrain (Milestone 9c) | Was complete mismatch (−49.8%); axis fix narrowed to +12.3%; slope=0.1 retrain further narrowed to +7.8% (4.9 std devs, was 7.8) |
| LeakyReLU negative slope 0.01 vs 0.1 | Confirmed, then resolved (§1.7, Milestone 9c) — both MLPs retrained 2026-08-09 | Moderate; closed ~1/3 of the AUPRC gap (6.47→4.42 std) and a similar fraction of the AP@50 gap (7.78→4.91 std) without changing AUROC; also slightly reduced both models' own absolute AUPRC/F1 |
| Loss function (focal vs BCE) / LR schedule unspecified | Tested twice, 2026-08-09: approximate (§1.8, M9d) reverted, then faithful-to-authors'-source (§1.9, M9e) succeeded; **M9e redesignated OFFICIAL 2026-09-01** | M9d's attempted fix made results dramatically WORSE (AUROC −64 std, AUPRC −24 std, AP@50 −24 std vs paper) via training instability. M9e (exact focal formula from authors' source, batch=32, per-step Keras LR decay) was completely stable and landed ALL THREE headline metrics within <1 std dev of paper — closest match in this repo. **2026-09-01: per explicit user decision for the reproduction paper, M9e (ChemBERTa) and its matched Morgan counterpart M9f are now the official MLP results; BCE/lr=0.001/batch=256 (M7/9c) is retained as a secondary/historical reference** |
| GNN implemented but doesn't reproduce paper performance | Confirmed, investigated twice, CLOSED/FROZEN (§1.4, Milestone 10 2026-08-11 + follow-up 2026-08-13, full detail `notes/gnn_gap_diagnostics.md`) | Architecture verified faithful to paper Eq. 1/2 and authors' source (one real bug found+fixed: negative-sampling local/global index mismatch); trains stably but test AUC 0.41 (paper 0.92, worse than random), AUPRC 0.53 (paper 0.89). Root cause narrowed precisely: an architecturally-inherent magnitude/degree signal (present at random init, unnormalized `GraphConv` on a heavy-tailed graph) survives training intact, while training abruptly (epoch 1→2) drives cosine similarity to a near-uniform negative value indistinguishable between classes — `dot_AUC ≈ 1 − norm_AUC` once this happens. Ruled out as causes: node/edge/embedding misalignment, layer-order bug, RNG-reseeding deviation (authors' own buggy behavior produces the identical inversion). Legacy PyTorch 2.4.1/PyG 2.6.1 environment (paper's exact versions) produced a real partial improvement (AUC 0.41→0.60) but still −82.5 std devs from paper — version gap narrows but does not close it. **Final: unresolved despite architecture, data, RNG, and (partial) software-environment fidelity; frozen, no further work without a new hypothesis.** |
| Morgan MLP added | Reproduction addition | N/A |
| ChemBERTa revision unspecified | Possible | Minor; agreement on AUROC suggests small effect |
| Pooling method unspecified | Possible | Minor to moderate |
| Threshold selection unspecified | Possible | Moderate for F1 metrics |
| SMILES source unspecified | Possible | Minor |
