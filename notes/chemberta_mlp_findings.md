# Milestone 7 — Frozen ChemBERTa Pair-Embedding MLP Findings

**Date:** 2026-06-30  
**Status:** Complete — ready for review

**2026-08-09 update:** this document's numbers (checkpoint/prediction hashes, best epoch 99, val macro AUPRC 0.4519, test metrics) describe the *original* run trained with LeakyReLU negative_slope=0.01 (PyTorch's default). That run was superseded 2026-08-09 (Milestone 9c) by a retrain under the paper-specified slope=0.1 (best epoch 99, val macro AUPRC 0.4381) — see `notes/deviations_from_paper.md` §1.7 for the full before/after comparison, including why the AUPRC/AP@50 gap to the paper narrowed. The original slope=0.01 artifacts described below are archived at `outputs/polyllm/chemberta_slope0.01_backup/`; this file is left as-is as the historical record of that run.

---

## 1. Purpose

This milestone trains a multi-label MLP using **frozen ChemBERTa pair embeddings**
(Milestone 5) to predict polypharmacy side effects under exactly the same protocol
used for the Morgan fingerprint baseline (Milestone 6B).

The ChemBERTa backbone (`DeepChem/ChemBERTa-77M-MLM`) was loaded and frozen in
Milestone 4 and is **not loaded, executed, or updated in this milestone**.  Only
the MLP head is trained.

All hyperparameters are identical to the Morgan baseline.  No values were adjusted
after seeing Morgan test results.

Do not yet draw a full scientific conclusion comparing Morgan and ChemBERTa.
That comparison belongs to the next milestone.

---

## 2. Input features

| Property | Value |
|---|---|
| Feature source | `data/features/chemberta_pair_embeddings.npy` |
| Feature shape | (63,472, 384), float32 |
| Feature encoding | Element-wise sum of two frozen ChemBERTa drug embeddings (Milestone 5) |
| Pooling method | Attention-mask-aware mean pool over final hidden layer (Milestone 4) |
| Special tokens | Included in pooling |
| Transformation | None — embeddings used as-is; no normalization, rescaling, or L2-norm |
| Memory mapping | `np.load(path, mmap_mode="r")` — no full copy created |
| Pre-validation | dtype=float32, all values finite, no all-zero rows — verified before training |

---

## 3. Model architecture

Same hidden structure as the Morgan MLP — only the first linear layer changes:

```
input (384)
  -> Linear(384, 512)          # smaller input dim than Morgan (2048)
  -> LeakyReLU (slope=0.01)
  -> BatchNorm1d(512)
  -> Dropout(0.2)
  -> Linear(512, 1024)
  -> LeakyReLU (slope=0.01)
  -> Dropout(0.2)
  -> Linear(1024, 2048)
  -> LeakyReLU (slope=0.01)
  -> Dropout(0.2)
  -> Linear(2048, 963)
  -> [raw logits — no sigmoid]
```

**Total parameters: 4,795,843**

Breakdown:
- Linear(384, 512) + bias: 197,120
- BatchNorm1d(512): 1,024
- Linear(512, 1024) + bias: 525,312
- Linear(1024, 2048) + bias: 2,099,200
- Linear(2048, 963) + bias: 1,973,187

**Why the parameter count differs from Morgan (5,647,811):**  
The first linear layer is 384 × 512 instead of 2048 × 512.  All hidden layers
and the output layer are identical.  The difference is:
(2048 − 384) × 512 = 851,968 fewer parameters.

No sigmoid is placed inside the model; `BCEWithLogitsLoss` applies it during
training, and `torch.sigmoid()` is applied during inference.

---

## 4. Fixed training configuration

All values match the Morgan baseline (Milestone 6B) exactly:

| Parameter | Value |
|---|---|
| Seed | 42 |
| Optimizer | Adam |
| Learning rate | 0.001 |
| Loss | BCEWithLogitsLoss |
| Class weights | None |
| Batch size | 256 |
| Maximum epochs | 100 |
| Early-stopping patience | 10 |
| Early-stopping min improvement | 0.0001 |
| Model-selection metric | Validation macro AUPRC |
| Dropout | 0.2 |
| LeakyReLU negative slope | 0.01 (PyTorch default) |
| Mixed precision | Disabled |
| num_workers | 0 (reliable on Windows) |
| Cross-hardware reproducibility | Not guaranteed |

No hyperparameter search was performed.  No values were changed after seeing
Morgan test results.

---

## 5. Device and software versions

| Item | Value |
|---|---|
| Device | CPU |
| Python | 3.11.9 |
| PyTorch | 2.12.1 |
| NumPy | 2.4.6 |
| pandas | 3.0.3 |
| scikit-learn | 1.6.1 |

No CUDA was available during this run.  No new packages were required.

---

## 6. Smoke-test result

Command:
```
.venv-polyllm/Scripts/python.exe src/polyllm/train_chemberta_mlp.py --smoke-test
```

Settings: first 2,000 training pair IDs; 20 most frequent labels measured
from training data only (never validation or test); 3 epochs;
output to `outputs/polyllm/chemberta/smoke_test/`.

All nine mechanics confirmed:

- Input dimension 384 verified
- Output dimension 20 (smoke mode)
- Forward pass
- BCE loss computation
- Backward pass and optimizer update (parameters changed)
- Validation prediction collection (shape, dtype, range, finite)
- Metric calculation (macro AUPRC)
- Checkpoint write
- Checkpoint load and prediction reproducibility (atol 1e-5)

Smoke-test metrics are **not** scientific results.

---

## 7. Official training

Command:
```
.venv-polyllm/Scripts/python.exe src/polyllm/train_chemberta_mlp.py
```

| Property | Value |
|---|---|
| Epochs completed | 100 (reached maximum) |
| Best epoch | 99 |
| Best validation macro AUPRC | 0.4519 |
| Early-stopping result | Not triggered — metric improved through epoch 99 |
| Final train loss (ep 100) | 0.1457 |
| Final val loss (ep 100) | 0.1645 |
| Training wall time | ~13 minutes on CPU |

The metric continued improving at epoch 99/100 (epoch 100 improved by 0.0001
exactly at the floating-point boundary but did not exceed `min_delta=0.0001`,
so the epoch-99 checkpoint was kept as the best).

---

## 8. Threshold selection

Performed on **validation set only** after restoring the best checkpoint.

| Property | Value |
|---|---|
| Grid | 0.05, 0.06, …, 0.95 (step 0.01; 91 thresholds) |
| Optimisation target | Validation micro F1 |
| Tie-break rule | Smallest threshold among equal values |
| **Selected threshold** | **0.28** |
| Validation micro F1 at 0.28 | 0.5006 |
| Validation micro F1 at 0.50 | 0.4254 |

The lower optimal threshold (0.28) reflects the class imbalance across 963
labels.  The threshold selection used the same grid and rule as the Morgan
baseline.

---

## 9. Test-set evaluation

Evaluated exactly once using the validation-selected threshold.  The test
split was never loaded during training, early stopping, or threshold selection.

### Primary metrics

| Metric | Value |
|---|---|
| Macro AUROC | **0.8881** |
| Macro AUPRC | **0.4490** |
| Micro AUPRC | **0.5329** |
| Micro F1 (thr=0.28) | **0.4967** |
| Macro F1 (thr=0.28) | **0.4287** |
| Micro F1 (thr=0.50) | 0.4193 |
| Macro F1 (thr=0.50) | 0.3512 |
| Sample mean AP@50 | **0.3795** |

### Skipped-label counts

All 963 labels had at least one positive test example; no label was skipped.

| Metric | Included | Skipped |
|---|---|---|
| Macro AUROC | 963 | 0 |
| Macro AUPRC | 963 | 0 |
| Macro F1 (sel) | 963 | 0 |
| Macro F1 (0.5) | 963 | 0 |

---

## 10. Per-label performance summary

Computed at the selected threshold 0.28.

| Metric | Min | Median | Max |
|---|---|---|---|
| Per-label AUROC | 0.6995 | 0.8991 | 0.9898 |
| Per-label AUPRC | 0.0926 | 0.4558 | 0.8394 |
| Per-label F1 (thr=0.28) | 0.0606 | 0.4385 | 0.7732 |

Full per-label results are in
`outputs/polyllm/chemberta/per_label_metrics.csv` (963 rows).

---

## 11. AP@50 operational definition

```
For each drug pair:
1. Rank all 963 labels by predicted probability (descending).
2. Keep the top 50.
3. For every rank k (1-based) containing a true label:
   precision@k = (# true labels in positions 1..k) / k
4. Sum those precision values.
5. Divide by min(number of true labels for this pair, 50).
6. Average across all 6,347 test pairs.
```

Pairs with zero true labels contribute 0.0.

**Result: 0.3795**

Function name: `sample_mean_ap_at_50` in `src/polyllm/metrics.py` (unchanged
from Morgan baseline).

> **paper_ap50_definition_compatibility: unresolved.**  
> This definition was fixed before seeing ChemBERTa results and has not been
> independently confirmed against the PolyLLM paper's formula.

---

## 12. Artifact hashes

| File | SHA-256 |
|---|---|
| `outputs/polyllm/chemberta/checkpoints/best_model.pt` | `742f0f014f7c199111528692d0d42bc7ddb298f87b7568165c1060344c765625` |
| `outputs/polyllm/chemberta/test_predictions.npz` | `15e89aec16f88836137643f147c7871c857a22599c4f2689c4929f6f69b472b9` |

Test predictions:
- Shape: (6,347, 963), float32
- All values finite, range [0, 1]
- Pair IDs sorted numerically

---

## 13. Data split interpretation

A **pair-random split** (seed 42, 80/10/10) was used, identical to the Morgan
baseline.  Pairs were assigned to splits at the pair level, not the drug level.
Every drug that appears in the test set also appeared in the training set
(with different partner drugs).  All 645 drugs were seen during training.

This is not a drug-level generalisation test.

---

## 14. 64-token truncation limitation

ChemBERTa tokenises SMILES with a maximum length of 64 tokens.  86 of the
645 drugs (13.3%) exceeded this limit and were silently truncated (Milestone 4).
The truncated suffix of a SMILES string encodes structural information about
atoms further from the centre of the molecule.  Truncation may reduce the
quality of embeddings for structurally complex drugs.

Because embeddings were computed once and frozen in Milestone 4, this
limitation is fixed and cannot be corrected at the MLP training stage.

---

## 15. Confirmation

- **ChemBERTa backbone was not fine-tuned.**  The `DeepChem/ChemBERTa-77M-MLM`
  model was frozen in Milestone 4 (`model.requires_grad_(False)`).  It was not
  loaded in this milestone.  Only the MLP head received gradient updates.

- **No Morgan artifact was modified.**  All files in `data/features/morgan_*`,
  `outputs/baseline/morgan/`, and `notes/morgan_*` are unchanged.
