# Milestone 9 — PolyLLM Paper vs Reproduction Comparison

**Date:** 2026-07-01  
**Status:** Complete — documentation and evidence synthesis only  
**Scope:** No models were retrained, no features regenerated, no thresholds changed, no predictions rerun, and no prior artifacts modified.

---

## 1. Reproduction Classification

Relative to the predefined project scope, the independent reduced reproduction is **complete**. It reconstructs the central frozen ChemBERTa–MLP pipeline, evaluates it on a permanent held-out pair split, and compares it with a matched Morgan-fingerprint baseline.

Relative to the complete PolyLLM study, it is a **partial reproduction**. It does not reproduce the graph-based model, all molecular representations, the complete hyperparameter search, or the paper's full cross-validation procedure. An exact or complete replication is therefore not claimed.

---

## 2. Paper Metadata

| Field | Value |
|---|---|
| Title | PolyLLM: polypharmacy side effect prediction via LLM-based SMILES encodings |
| Authors | Sadra Hakim, Alioune Ngom |
| Journal | Frontiers in Pharmacology |
| Year | 2025 |
| DOI | 10.3389/fphar.2025.1617142 |
| Source | Paper Table 5 for numerical results; Sections 2.1–2.2 for methodology |

---

## 3. Dataset Comparison

| Property | Paper | Reproduction | Match? |
|---|---|---|---|
| Source dataset | Decagon / TWOSIDES | Same | Yes |
| Raw rows | 4,649,441 | 4,649,441 | Yes |
| Drug count | 645 | 645 | Yes |
| Original side effect types | 1,318 | 1,317 | **No — off by 1** |
| Filtering threshold | ≥500 drug combinations | ≥500 canonical pairs | Yes |
| Retained side effect types | 964 | 963 | **No — off by 1** |
| Final drug pair count | 63,472 | 63,472 | Yes |

**Label count deviation:** The paper retains 964 labels; the reproduction retains 963. Both original counts also differ by 1 (1,318 vs 1,317). The off-by-one is consistent across both raw and filtered counts. Most likely cause: one side effect identifier is absent or differently canonicalized in our downloaded dataset version, or the paper resolves an exact-500-pair boundary differently. This cannot be resolved without access to the paper's intermediate processing output.

All dataset claims reference `outputs/polyllm/data_audit.json`.

---

## 4. ChemBERTa Configuration Comparison

| Property | Paper | Reproduction | Match? |
|---|---|---|---|
| Model | DeepChem/ChemBERTa-77M-MLM | DeepChem/ChemBERTa-77M-MLM, revision ed8a5374 | Probable match |
| Backbone frozen | Yes (implied) | Yes — model.requires_grad_(False) | Confirmed match |
| Max token length | 64 (described as "characters") | 64 tokens (HuggingFace max_length) | Probable match |
| Drugs truncated | Not reported | 86 / 645 (13.3%) | Not directly comparable |
| Pooling method | Not described | Attention-mask-aware mean pool (incl. special tokens) | Unverifiable |
| Pair fusion | Element-wise sum | Element-wise sum | Confirmed match |
| L2 normalization after fusion | Not described | None | Unverifiable |

The paper describes the token limit as "64 characters" but in practice Hugging Face tokenizers use token units. For SMILES strings, characters and BPE tokens are approximately 1:1 for simple molecules, so "64 characters" and "64 tokens" likely resolve to the same setting. The paper's reported mean SMILES length of 54.25 characters is consistent with our finding that 86/645 drugs exceed 64 tokens.

All ChemBERTa claims reference `notes/chemberta_embedding_findings.md` and `outputs/polyllm/embedding_audit.json`.

---

## 5. MLP Architecture Comparison

| Property | Paper | Reproduction | Match? |
|---|---|---|---|
| Input dim (ChemBERTa) | 384 | 384 | Confirmed match |
| Hidden layers | [512, 1024, 2048] | [512, 1024, 2048] | Confirmed match |
| Activation | Leaky ReLU | LeakyReLU (slope=0.01) | Confirmed match |
| Batch norm | After first hidden layer | BatchNorm1d(512) after layer 1 | Confirmed match |
| Dropout | 0.2 | 0.2 | Confirmed match |
| Output dim | 964 | 963 | Deviated (label count) |
| Output activation | Not specified | None (BCEWithLogitsLoss applies sigmoid) | Probable match |

The architecture is a confirmed match in all specified dimensions. The output dimension deviation follows directly from the 963 vs 964 label count difference.

Architecture claims reference `notes/chemberta_mlp_findings.md` and `outputs/polyllm/chemberta/training_config.json`.

---

## 6. Training and Evaluation Protocol Comparison

| Property | Paper | Reproduction | Match? |
|---|---|---|---|
| Split ratios | 80 / 10 / 10 | 80 / 10 / 10 | Confirmed match |
| Split unit | Drug pairs | Drug pairs (pair-random) | Confirmed match |
| Split seed | Not specified | 42 | Unverifiable |
| Cross-validation | 10-fold CV on train+val sets | None — single fixed split | **Confirmed deviation** |
| Optimizer | Not specified | Adam, lr=0.001 | Unverifiable |
| Loss function | Not specified | BCEWithLogitsLoss | Unverifiable |
| Class weights | Not specified | None | Unverifiable |
| Batch size | Not specified | 256 | Unverifiable |
| Max epochs | Not specified | 100 | Unverifiable |
| Threshold selection | Not described | Val micro F1 grid search 0.05–0.95 | Unverifiable |

**Critical deviation: evaluation protocol.** The paper performs 10-fold cross-validation and reports mean ± std over 10 runs. The reproduction uses a single fixed split. This difference directly limits numerical comparability: the paper's ± standard deviations cannot be used to construct confidence intervals around the reproduction's single-run values in any standard way.

Split claims reference `notes/split_findings.md` and `outputs/polyllm/split_audit.json`.

---

## 7. Quantitative Result Comparison

### 7.1 ChemBERTa MLP (the shared experiment)

Paper results are for the "DeepChem ChemBERTa" MLP model from Table 5, using 10-fold CV on a fixed 80/10/10 split over 964 labels. Reproduction results are from a single run on a fixed split (seed=42) over 963 labels.

| Metric | Paper (DeepChem ChemBERTa MLP) | Reproduction (ChemBERTa MLP) | Diff (repro − paper) | Comparability |
|---|---|---|---|---|
| AUROC / Macro AUROC | 0.8859 ± 0.0019 | **0.8881** | +0.0022 | Partial |
| AUPRC / Macro AUPRC | 0.3979 ± 0.0079 | **0.4490** | +0.0511 | Partial |
| AP@50 | 0.7557 ± 0.0120 | **0.3795** | −0.3762 | Likely incomparable |

**AUROC:** The reproduction value (0.8881) is +0.0022 above the paper mean and within 1.2 standard deviations of the paper's 10-fold CI. This should be interpreted descriptively rather than as evidence of statistical equivalence, because the reproduction used a different split procedure, a single seed, and independently reconstructed preprocessing.

**AUPRC:** The reproduction value (0.4490) is +0.0511 above the paper mean, which is 6.5 standard deviations — substantially larger than the fold-to-fold variance reported by the paper. The difference cannot reasonably be explained by retaining 963 instead of 964 labels, because changing one label can alter a macro average by at most approximately 0.001. Possible explanations include differences in dataset preprocessing, split or cross-validation methodology, SMILES processing, pooling and truncation, model selection, metric implementation, and training variability. The available evidence does not identify a single confirmed cause.

**AP@50:** The reproduction value of 0.3795 should not be compared directly with the paper's reported value of 0.7557 because the exact AP@50 definition used by the paper could not be verified. The reproduction uses sample-mean average precision over the top 50 ranked labels per pair. Until the authors' precise implementation is confirmed, the difference should be treated primarily as a metric-definition compatibility issue rather than a model-performance discrepancy.

### 7.2 Morgan Fingerprint MLP (reproduction only)

The Morgan fingerprint MLP baseline is not reported in the PolyLLM paper. It was added in the reproduction to provide a conventional chemoinformatics comparison point.

| Metric | Morgan MLP (repro only) | ChemBERTa MLP (repro) | Morgan advantage |
|---|---|---|---|
| Macro AUROC | 0.8980 | 0.8881 | +0.0099 (+1.1%) |
| Macro AUPRC | 0.4969 | 0.4490 | +0.0479 (+10.7%) |
| Micro AUPRC | 0.5784 | 0.5329 | +0.0455 (+8.5%) |
| Micro F1 (sel. thr.) | 0.5225 | 0.4967 | +0.0258 (+5.2%) |
| Sample mean AP@50 | 0.4220 | 0.3795 | +0.0425 (+11.2%) |

Morgan outperforms frozen ChemBERTa across all metrics. See `notes/model_comparison_findings.md` for the full analysis.

---

## 8. AP@50 Formula Analysis

The reproduction's AP@50 (0.3795 for ChemBERTa, 0.4220 for Morgan) is approximately half the paper's reported value (0.7557 for DeepChem ChemBERTa MLP). A discrepancy this large rules out random variance or label count effects.

**Reproduction formula** (`sample_mean_ap_at_50` in `src/polyllm/metrics.py`):

```
For each drug pair p:
1. Rank all 963 labels by predicted probability (descending).
2. Consider ranks 1..50.
3. For each rank k where rank k is a true label:
   precision@k = (# true labels in positions 1..k) / k
4. AP_p = sum(precision@k for true ranks k ≤ 50) / min(# true labels for p, 50)
5. AP@50 = mean(AP_p) over all 6,347 test pairs.
```

**Paper formula (Paper Equations 5a-5b):** "predictions are first ordered by their confidence scores, and precision is calculated separately for the top elements." The exact denominator and aggregation direction (per-pair vs per-label) are not stated. The paper cites Zitnik et al. (2018) as the source of this metric; without the authors' code, the precise implementation cannot be verified.

**Status: metric-definition compatibility unresolved.** The AP@50 values from this reproduction should not be compared numerically to the paper. The difference should be treated as a metric-definition compatibility issue rather than a model-performance discrepancy.

---

## 9. Scientific Interpretation

**What the reproduction supports:**

1. The PolyLLM MLP architecture with frozen ChemBERTa embeddings achieves AUROC ≈ 0.888–0.889 on the Decagon dataset with a 963/964-label, 63,472-pair setup. The reproduction AUROC (0.8881) matches the paper value (0.8859) well.

2. Frozen ChemBERTa embeddings provide a stronger signal than random features (AUROC >> 0.5) but are outperformed by Morgan fingerprints in this reproduction's pair-random split. Whether this holds in the paper's 10-fold setting cannot be assessed because the paper does not report Morgan fingerprint MLP results.

3. The pair-random split means every drug in the test set also appeared in training. Performance on fully unseen drugs would likely be lower. This is an acknowledged limitation shared by both the paper and the reproduction.

**What the reproduction does not support:**

1. The paper's primary contribution is the GNN architecture, which achieves AUROC 0.92 and AUPRC 0.89. The reproduction does not implement or evaluate a GNN. The MLP results are a secondary component of the paper.

2. The paper's AP@50 claims cannot be assessed in the reproduction because the formula is incompatible.

3. No statistical claim about the gap between Morgan and ChemBERTa can be generalized beyond this single fixed split. The paired bootstrap (Milestone 8) quantifies sampling variability only — not training or split variability.

---

## 10. Definition-of-Done Assessment

| Criterion | Status |
|---|---|
| Paper metadata identified | Done |
| Dataset comparison complete | Done |
| Architecture comparison complete | Done |
| Split protocol comparison complete | Done |
| Quantitative comparison (AUROC, AUPRC, AP@50) | Done |
| AP@50 formula discrepancy documented | Done |
| Morgan baseline added as reproduction contribution | Done |
| All deviations documented in deviations_from_paper.md | Done |
| paper_comparison.csv created | Done |
| paper_comparison_audit.json created | Done |
| No models retrained, no artifacts modified | Confirmed |

**Overall:** Milestone 9 complete. Relative to the predefined project scope, the reproduction is **complete**. Relative to the full PolyLLM study, it is **partial** — GNN path not reproduced, full cross-validation not performed, AP@50 metric-definition compatibility unresolved.

---

## 11. Artifact Summary

| File | Description |
|---|---|
| `outputs/comparison/paper_comparison.csv` | Row-per-metric comparison with comparability classification |
| `outputs/comparison/paper_comparison_audit.json` | Full audit: paper metadata, sources, method match, artifact hashes |
| `notes/deviations_from_paper.md` | Confirmed deviations, possible deviations, reproduction additions |
| `requirements-milestone9.txt` | No new packages required |
