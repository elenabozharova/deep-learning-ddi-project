# Milestone 9 — Deviations from the PolyLLM Paper

**Date:** 2026-07-01  
**Paper:** "PolyLLM: polypharmacy side effect prediction via LLM-based SMILES encodings"  
**DOI:** 10.3389/fphar.2025.1617142  

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

### 1.3 AP@50 formula: incompatible values

| Property | Paper (DeepChem ChemBERTa MLP) | Reproduction (ChemBERTa MLP) |
|---|---|---|
| AP@50 value | 0.7557 ± 0.0120 | 0.3795 |
| Difference | — | −0.3762 (−49.8%) |

**Evidence:**  
Paper: Table 5. Reproduction: `outputs/polyllm/chemberta/test_metrics.json` field `sample_mean_ap_at_50`.

**Consequence:** AP@50 cannot be used for numerical comparison between this reproduction and the paper. The reproduction's AP@50 values (ChemBERTa: 0.3795; Morgan: 0.4220) and the paper's AP@50 values almost certainly measure different quantities.

**Hypothesis:** The most likely cause is a different denominator or aggregation direction (per-pair vs per-label). The paper cites Zitnik et al. (2018) as the source of this metric; the paper's own description ("precision calculated for the top elements") is ambiguous. Without the paper's code, the exact formula cannot be verified.

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

### 1.7 LeakyReLU negative slope: 0.1 (paper) vs 0.01 (reproduction)

| Property | Paper | Reproduction |
|---|---|---|
| Activation function | Leaky ReLU (Xu et al., 2015) | Leaky ReLU (`torch.nn.LeakyReLU`) |
| Negative slope | **0.1** | **0.01** |

**Evidence:**  
Paper (quoted by user): "we evaluated several activation functions and selected the Leaky Rectified Linear Unit (Leaky ReLU) Xu et al. (2015) with a negative slope of 0.1 for each hidden layer. Leaky ReLU was selected for its ability to address the vanishing gradient problem... and also for its capacity to enhance learning stability."  
Reproduction: `src/polyllm/models/mlp.py:25`, `NEGATIVE_SLOPE: float = 0.01  # PyTorch LeakyReLU default`; threaded through both training configs unchanged — `train_morgan_baseline.py:83` and `train_chemberta_mlp.py:90`, `"leaky_relu_negative_slope": 0.01,   # PyTorch default for LeakyReLU`.

**Root cause:** The reproduction adopted PyTorch's own out-of-the-box default for `nn.LeakyReLU` (0.01) rather than a value derived from the paper — confirmed by the code comments crediting "PyTorch default," not the paper, as the source of this number. The paper's specific value (0.1) was not known at the time this code was written.

**Consequence:** A 10× difference in how much negative-input signal each hidden-layer neuron passes through during both the forward pass and backpropagation. The paper's stated rationale (mitigating vanishing gradients, improving learning stability) is only partially realized at 0.01 versus 0.1. Effect on final metrics is unquantified without retraining — same caveat as other hyperparameter deviations in this document.

**Evidence reference:** `src/polyllm/models/mlp.py`; `outputs/polyllm/chemberta/training_config.json` and `outputs/baseline/morgan/training_config.json` (both record `leaky_relu_negative_slope: 0.01`).

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

**Note:** The activation function and its negative-slope parameter, previously bundled into this general "unspecified hyperparameters" claim, are now confirmed and moved to their own entry — see §1.7 (confirmed 0.1 vs 0.01 mismatch).

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

### 2.6 AUPRC higher than paper (cause unconfirmed)

**Claim:** The reproduction's macro AUPRC (0.4490) is 0.0511 above the paper's 10-fold mean (0.3979), which is 6.5 standard deviations.

**Why possible:** The difference is substantially larger than the fold-to-fold variation reported by the paper. The label count difference (963 vs 964) is not a plausible explanation — changing one label can alter a macro average by at most approximately 0.001. Possible causes include differences in dataset preprocessing, split or cross-validation methodology, SMILES processing, pooling and truncation, model selection, metric implementation, and training variability. The available evidence does not identify a single confirmed cause.

**No corrective action required or possible without retraining.**

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

---

## 4. Summary Table

| Deviation | Category | Impact on comparison |
|---|---|---|
| 963 vs 964 labels | Confirmed | Small; shifts macro metrics slightly |
| Single run vs 10-fold CV | Confirmed | Moderate; single-point vs distribution |
| AP@50 formula incompatible | Confirmed | Complete; metric not comparable |
| GNN not reproduced | Confirmed | Scope limit; best paper results missing |
| Morgan MLP added | Reproduction addition | N/A |
| ChemBERTa revision unspecified | Possible | Minor; agreement on AUROC suggests small effect |
| Pooling method unspecified | Possible | Minor to moderate |
| Optimizer/hyperparameters unspecified | Possible | Unknown |
| Threshold selection unspecified | Possible | Moderate for F1 metrics |
| SMILES source unspecified | Possible | Minor |
