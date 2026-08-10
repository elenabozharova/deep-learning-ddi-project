# Milestone 9 — Deviations from the PolyLLM Paper

**Date:** 2026-07-01 (Milestone 9); updated 2026-08-08 (Milestone 9b, AP@50 axis fix) and 2026-08-09 (Milestone 9c, LeakyReLU slope fix + retrain)  
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

**Status: the axis-mismatch hypothesis below is now confirmed.** The paper author supplied their exact reference implementation directly:

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

The author's exact function is now implemented as `average_precision_at_k_multi_label()` in `src/polyllm/metrics.py`, and recomputed against the already-saved Milestone 6B/7 test predictions (no retraining, no checkpoint reloaded) by `src/polyllm/recompute_ap_at_k.py` (Milestone 9b). Results:

| Property | Paper (DeepChem ChemBERTa MLP) | Reproduction — original per-pair axis | Reproduction — corrected per-label axis |
|---|---|---|---|
| AP@50 value | 0.7557 ± 0.0120 | 0.3795 | **0.8491** |
| Difference from paper mean | — | −0.3762 (−49.8%, −31.4 std devs) | **+0.0934 (+12.3%, +7.8 std devs)** |

**Evidence:** `outputs/polyllm/chemberta/ap_at_k_recompute.json`, `outputs/baseline/morgan/ap_at_k_recompute.json` (Morgan, for context only — paper does not report a Morgan value: 0.8950, up from 0.4220). Both frozen `test_metrics.json` and `test_predictions.npz` files were confirmed byte-identical (SHA-256) before and after this recomputation — see `outputs/comparison/comparison_audit.json` `source_artifact_hashes`.

**Consequence:** The corrected per-label-axis AP@50 closes roughly 75% of the original 0.3762 gap. The metric is far more comparable now than before, though not identical — the residual +0.0934 (reproduction above paper) is directionally consistent with the already-observed pattern that this reproduction's AUROC (+1.2 std) and AUPRC (+6.5 std) both sit above the paper's reported means (§2.6), so the remaining gap most likely shares whatever cause underlies those two, rather than being a new, AP@50-specific issue.

**One remaining caveat:** 69/963 ChemBERTa labels (7.2%; 199/963 for Morgan) have all 50 top-ranked pairs positive within that label's own top-50, which makes `average_precision_score` return exactly 1.0 for those columns by construction. This is a property of the formula itself (confirmed present in the author's snippet, not a reproduction bug), not filtered out, and inflates the mean somewhat for very high-prevalence or well-separated labels. Zero labels had the opposite (all-negative-in-top-50) degeneracy for either model.

**What is still not verified:** whether `average_precision_score`'s sklearn interpolated-AP formula is bit-for-bit what the paper's own code used internally (the author's snippet does use `sklearn.metrics.average_precision_score`, so this is now a much safer assumption than before), and whether the paper's own 10-fold protocol changes this number materially (§1.2 remains a separate, still-open deviation).

**Update 2026-08-09 (Milestone 9c):** after retraining both MLPs with the paper-matched LeakyReLU slope=0.1 (§1.7), this metric was recomputed against the new predictions. ChemBERTa's paper-axis AP@50 moved from 0.8491 to **0.8146** — a further narrowing from +7.8 to **+4.9 std devs** above the paper mean (0.7557 ± 0.0120), on top of the axis fix above. Degenerate all-positive-top-50 columns dropped from 69/963 (7.2%) to 28/963 (2.9%), consistent with the slope=0.1 model being somewhat less confident/less separated overall (see the parallel drop in raw AUPRC in §1.7). Morgan (context only): 0.8950 → 0.8716. Current numbers: `outputs/polyllm/chemberta/ap_at_k_recompute.json`, `outputs/baseline/morgan/ap_at_k_recompute.json`; the pre-retrain values above remain historically accurate for the archived slope=0.01 checkpoints.

---

### 1.4 GNN path not reproduced

| Property | Paper | Reproduction |
|---|---|---|
| MLP with ChemBERTa embeddings | Yes | Yes (reproduced) |
| GNN with ChemBERTa node features | Yes (best results) | Not implemented |

**Evidence:**  
Paper: Sections 2.3–2.4 and Table 5 report GNN results (AUROC 0.92, AUPRC 0.89).  
Reproduction scope: `notes/polyllm_reproduction_scope.md`, "GNN not in scope."

**Consequence:** The paper's strongest results (GNN path) are not available in this reproduction. The MLP-only comparison reflects only the secondary component of the paper's contribution.

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

**The Milestone 7/9c configuration (BCE, fixed lr=0.001, slope=0.1) remains the reproduction's best and official ChemBERTa result.** This closes the "reconcile remaining hyperparameters" line of investigation for now, per the user's own framing ("after that I would consider the MLP definitively finished").

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

**Status: diagnostic ablation, not a redesignation.** Per explicit user instruction, `outputs/polyllm/chemberta/` (Milestone 7/9c: BCE, fixed lr=0.001, slope=0.1) remains the reproduction's **official** ChemBERTa MLP result, despite M9e's closer paper agreement. The remaining ~0.5–0.7 std dev gap under the faithful configuration is most plausibly attributable to the still-unreplicated single-split-vs-10-fold-CV protocol difference (§1.2) rather than to any further hyperparameter tuning.

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

**Corrective action:** partially taken (§1.7 retrain, which helped) and partially attempted-and-reverted (§1.8 loss/LR ablation, which hurt). Per the user's explicit direction, this closes the "reconcile remaining MLP hyperparameters" investigation for now — the Milestone 7/9c configuration (BCE, fixed lr=0.001, slope=0.1) is retained as the reproduction's official result, and remaining unexplained variance (the ~4.4 std dev AUPRC gap under that configuration) is attributed most plausibly to the still-open single-split-vs-10-fold-CV protocol difference (§1.2) rather than to loss/LR, pending any future, more carefully isolated sub-ablation (candidates listed in §1.8).

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

### 3.8 Author-verified AP@k recomputation (Milestone 9b)

`average_precision_at_k_multi_label()` (`src/polyllm/metrics.py`) and `src/polyllm/recompute_ap_at_k.py` implement and run the paper author's own reference AP@k snippet against the already-saved Milestone 6B/7 predictions, with no retraining and no modification of any existing artifact (verified by SHA-256 comparison before/after). This resolved §1.3 from a complete mismatch to a much narrower, directionally-explicable gap. See `outputs/baseline/morgan/ap_at_k_recompute.json` and `outputs/polyllm/chemberta/ap_at_k_recompute.json`.

---

## 4. Summary Table

| Deviation | Category | Impact on comparison |
|---|---|---|
| 963 vs 964 labels | Confirmed | Small; shifts macro metrics slightly |
| Single run vs 10-fold CV | Confirmed | Moderate; single-point vs distribution; largest remaining structural difference after §1.7 |
| AP@50 formula incompatible | Confirmed, then resolved (§1.3, Milestone 9b); further narrowed by retrain (Milestone 9c) | Was complete mismatch (−49.8%); axis fix narrowed to +12.3%; slope=0.1 retrain further narrowed to +7.8% (4.9 std devs, was 7.8) |
| LeakyReLU negative slope 0.01 vs 0.1 | Confirmed, then resolved (§1.7, Milestone 9c) — both MLPs retrained 2026-08-09 | Moderate; closed ~1/3 of the AUPRC gap (6.47→4.42 std) and a similar fraction of the AP@50 gap (7.78→4.91 std) without changing AUROC; also slightly reduced both models' own absolute AUPRC/F1 |
| Loss function (focal vs BCE) / LR schedule unspecified | Tested twice, 2026-08-09: approximate (§1.8, M9d) reverted, then faithful-to-authors'-source (§1.9, M9e) succeeded | M9d's attempted fix made results dramatically WORSE (AUROC −64 std, AUPRC −24 std, AP@50 −24 std vs paper) via training instability. M9e (exact focal formula from authors' source, batch=32, per-step Keras LR decay) was completely stable and landed ALL THREE headline metrics within <1 std dev of paper — closest match in this repo. Diagnostic ablation only; official result unchanged (BCE/lr=0.001/batch=256 retained per explicit user instruction) |
| GNN not reproduced | Confirmed | Scope limit; best paper results missing |
| Morgan MLP added | Reproduction addition | N/A |
| ChemBERTa revision unspecified | Possible | Minor; agreement on AUROC suggests small effect |
| Pooling method unspecified | Possible | Minor to moderate |
| Threshold selection unspecified | Possible | Moderate for F1 metrics |
| SMILES source unspecified | Possible | Minor |
