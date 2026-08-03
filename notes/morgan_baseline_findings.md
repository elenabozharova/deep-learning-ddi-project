# Milestone 6B — Morgan Fingerprint MLP Baseline Findings

**Date:** 2026-06-30  
**Status:** Complete — ready for review

---

## 1. Purpose

This baseline trains a multi-label MLP using **Morgan fingerprint pair features**
(Milestone 6A) to predict polypharmacy side effects.  Morgan fingerprints are
deterministic, handcrafted molecular descriptors, not learned language-model
embeddings.  The baseline serves two purposes:

1. Establish a conventional chemoinformatics reference point before the
   ChemBERTa MLP (Milestone 7).
2. Verify that the shared training infrastructure (`training.py`, `metrics.py`,
   `models/mlp.py`) works end-to-end with real data.

No comparison to ChemBERTa performance is made here because the ChemBERTa MLP
has not been trained.

---

## 2. Input features

| Property | Value |
|---|---|
| Feature source | `data/features/morgan_pair_fingerprints.npy` |
| Feature shape | (63,472, 2,048), uint8 |
| Feature encoding | Element-wise sum of two drug Morgan fingerprints; values in {0, 1, 2} |
| Transformation | None — values 0, 1, 2 passed directly to the model |
| Label source | `data/processed/polyllm_labels.npy` |
| Label shape | (63,472, 963), uint8 |
| Memory mapping | `np.load(path, mmap_mode="r")` — no full float32 copy created |

Batches are converted to float32 inside `PairDataset.__getitem__`.

---

## 3. Model architecture

```
input (2,048)
  -> Linear(2048, 512)
  -> LeakyReLU (negative_slope=0.01)
  -> BatchNorm1d(512)
  -> Dropout(0.2)
  -> Linear(512, 1024)
  -> LeakyReLU (negative_slope=0.01)
  -> Dropout(0.2)
  -> Linear(1024, 2048)
  -> LeakyReLU (negative_slope=0.01)
  -> Dropout(0.2)
  -> Linear(2048, 963)
  -> [raw logits — no sigmoid]
```

**Total parameters: 5,647,811**

Breakdown:
- Linear(2048, 512) + bias: 1,049,088
- BatchNorm1d(512): 1,024
- Linear(512, 1024) + bias: 525,312
- Linear(1024, 2048) + bias: 2,099,200
- Linear(2048, 963) + bias: 1,973,187

BatchNorm appears only after the first hidden layer.  No sigmoid is placed
inside the model; `BCEWithLogitsLoss` applies it numerically stably during
training, and `torch.sigmoid()` is applied during inference.

The same `MultilabelMLP` class accepts `input_dim=384` for the future
ChemBERTa experiment without any hidden-layer changes.

---

## 4. Fixed training configuration

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

No hyperparameter search was performed.  These values are fixed for the
official run.

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

No CUDA was available during this run.

---

## 6. Smoke-test result

Command:
```
.venv-polyllm\Scripts\python.exe src/polyllm/train_morgan_baseline.py --smoke-test
```

Settings: first 2,000 training pair IDs; 20 most frequent labels measured from
training data only; 3 epochs; output to `outputs/baseline/morgan/smoke_test/`.

All seven mechanics confirmed:

- Forward pass
- BCE loss computation
- Backward pass and optimizer update (parameters changed)
- Validation prediction collection (shape, dtype, range, finite values)
- Metric calculation (macro AUPRC)
- Checkpoint write
- Checkpoint load and prediction reproducibility (atol 1e-5)

Smoke-test metrics are **not** scientific results.

---

## 7. Official training

Command:
```
.venv-polyllm\Scripts\python.exe src/polyllm/train_morgan_baseline.py
```

| Property | Value |
|---|---|
| Epochs completed | 100 (reached maximum) |
| Best epoch | 99 |
| Best validation macro AUPRC | 0.4966 |
| Early-stopping result | Not triggered — metric improved continuously through epoch 99 |
| Final train loss | 0.1356 |
| Final val loss | 0.1575 |

Training converged steadily without plateauing within the epoch budget.
With more epochs, the metric would likely continue improving slightly.

---

## 8. Threshold selection

Performed on **validation set only** after restoring the best checkpoint.

| Property | Value |
|---|---|
| Grid | 0.05, 0.06, ..., 0.95 (step 0.01; 91 thresholds) |
| Optimisation target | Validation micro F1 |
| Tie-break rule | Smallest threshold among equal values |
| **Selected threshold** | **0.31** |
| Validation micro F1 at 0.31 | 0.5241 |
| Validation micro F1 at 0.50 | 0.4762 |

The lower optimal threshold (0.31 vs 0.50) reflects the class imbalance:
most labels are rare, so a lower threshold recovers more true positives.

---

## 9. Test-set evaluation

Evaluated exactly once using the validation-selected threshold.
The test split was **never** loaded during training, early stopping,
threshold selection, architecture choice, or debugging.

### Primary metrics

| Metric | Value |
|---|---|
| Macro AUROC | **0.8980** |
| Macro AUPRC | **0.4969** |
| Micro AUPRC | **0.5784** |
| Micro F1 (thr=0.31) | **0.5225** |
| Macro F1 (thr=0.31) | **0.4614** |
| Micro F1 (thr=0.50) | 0.4753 |
| Macro F1 (thr=0.50) | 0.4081 |
| Sample mean AP@50 | **0.4220** |

### Skipped-label counts

All 963 labels had at least one positive example in the test set, so no
label was skipped from any macro metric.

| Metric | Included | Skipped |
|---|---|---|
| Macro AUROC | 963 | 0 |
| Macro AUPRC | 963 | 0 |
| Macro F1 (sel) | 963 | 0 |
| Macro F1 (0.5) | 963 | 0 |

---

## 10. Per-label performance summary

Computed at the selected threshold 0.31.

| Metric | Min | Median | Max |
|---|---|---|---|
| Per-label AUROC | 0.7218 | 0.909 | 0.994 |
| Per-label AUPRC | 0.0975 | 0.5106 | 0.872 |
| Per-label F1 (thr=0.31) | 0.0571 | 0.4743 | 0.8072 |

Full per-label results are in `outputs/baseline/morgan/per_label_metrics.csv`
(963 rows, columns: `label_index, side_effect_id, side_effect_name,
test_positive_count, test_prevalence, auroc, auprc, f1_at_selected_threshold`).

---

## 11. AP@50 operational definition

```
For each drug pair:
1. Rank all 963 labels by predicted probability (descending).
2. Keep the top 50.
3. For every rank k (1-based) that contains a true label:
   precision@k = (# true labels in positions 1..k) / k
4. Sum those precision values.
5. Divide by min(number of true labels for this pair, 50).
6. Average across all 6,347 test pairs.
```

Pairs with zero true labels contribute 0.0.

**Result: 0.4220**

This function is called `sample_mean_ap_at_50` in `src/polyllm/metrics.py`.

> **paper_ap50_definition_compatibility: unresolved.**  
> This definition has not been independently confirmed against the PolyLLM
> paper's formula.  The result should not be compared to the paper's AP@50
> without verifying that the definitions match.

---

## 12. Artifact hashes

| File | SHA-256 |
|---|---|
| `outputs/baseline/morgan/checkpoints/best_model.pt` | `69666e3a5da0749c713da233d6730109a445aeee221b1919de121c492dc625b5` |
| `outputs/baseline/morgan/test_predictions.npz` | `3c3ce71f135c5877e0d9c0b0a6f083c55d2b4b8b23752e9fa522683ff38ff773` |

Test predictions:
- Shape: (6,347, 963), float32
- All values finite, range [0, 1]
- Pair IDs sorted numerically (0-indexed)

---

## 13. Data split interpretation

A **pair-random split** (seed 42, 80/10/10) was used.  Pairs were assigned
to splits at the pair level, not the drug level.  This means every drug
that appears in the test set also appears in the training set (possibly with
different partner drugs).  The model has seen all 645 drugs during training.

This is not a drug-level generalisation test.  Performance on drug pairs
involving completely unseen drugs is not assessed here.

---

## 14. Class-imbalance limitations

The 963 side-effect labels are highly imbalanced.  Most labels are positive
for only a small fraction of the 63,472 drug pairs.  The model was trained
with no class weights and BCEWithLogitsLoss averaged over all labels.  The
optimised threshold (0.31 < 0.50) compensates partially for this imbalance
at prediction time.

Labels with very few positive test examples have lower and more variable
AUROC and AUPRC.  The minimum per-label AUPRC (0.0975) reflects a rare
side effect that is difficult to predict from Morgan fingerprints alone.

---

## 15. Confirmation

No ChemBERTa model was trained in this milestone.  No model training of any
kind was performed in Milestones 1–6A.  The Morgan fingerprints (Milestone 6A)
are deterministic handcrafted descriptors generated by RDKit with no neural
network component.
