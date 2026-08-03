# Milestone 8 — Morgan vs ChemBERTa MLP Comparison Findings

**Date:** 2026-07-01  
**Status:** Complete — analysis only; no retraining or artifact modification

---

## 1. Purpose

This milestone compares the two completed experiments:

| Model | Milestone | Input | Parameters |
|---|---|---|---|
| Morgan fingerprint MLP | 6B | 2048-bit Morgan fingerprints | 5,647,811 |
| ChemBERTa MLP (frozen) | 7 | 384-dim ChemBERTa pair embeddings | 4,795,843 |

Both models used identical training protocols: seed=42, Adam lr=0.001,
BCEWithLogitsLoss, batch=256, max 100 epochs, patience=10.  All comparisons
are against the held-out test set (6,347 pairs) evaluated exactly once.

---

## 2. Comparability validation

17 config fields were verified to match:
`random_seed`, `optimizer`, `learning_rate`, `loss`, `batch_size`,
`max_epochs`, `patience`, `min_delta`, `model_selection_metric`, `dropout`,
`leaky_relu_negative_slope`, `num_workers`, `output_dim`, `hidden_dims`,
`batch_norm_after_layer`, `train_pairs`, `val_pairs`.

Intentional differences (excluded from validation):
`input_dim` (2048 vs 384), `total_parameters`, `feature_source`.

Validation status: **PASSED**

---

## 3. Aggregate metric comparison

All 8 metrics favour Morgan.

| Metric | Morgan | ChemBERTa | Diff (M−C) | Rel. diff | Winner |
|---|---|---|---|---|---|
| Macro AUROC | 0.8980 | 0.8881 | +0.0099 | +1.11% | Morgan |
| Macro AUPRC | 0.4969 | 0.4490 | +0.0479 | +10.68% | Morgan |
| Micro AUPRC | 0.5784 | 0.5329 | +0.0455 | +8.54% | Morgan |
| Micro F1 (sel. thr.) | 0.5225 | 0.4967 | +0.0258 | +5.20% | Morgan |
| Macro F1 (sel. thr.) | 0.4614 | 0.4287 | +0.0328 | +7.64% | Morgan |
| Micro F1 (thr=0.50) | 0.4753 | 0.4193 | +0.0560 | +13.36% | Morgan |
| Macro F1 (thr=0.50) | 0.4081 | 0.3512 | +0.0569 | +16.19% | Morgan |
| Sample mean AP@50 | 0.4220 | 0.3795 | +0.0425 | +11.20% | Morgan |

Selected thresholds: Morgan=0.31, ChemBERTa=0.28 (both chosen to maximise
validation micro F1 before any test evaluation).

---

## 4. Per-label comparison (963 labels)

Win counts at tolerance 1×10⁻¹² (|diff| > tol required to declare a winner):

| Metric | Morgan wins | ChemBERTa wins | Ties |
|---|---|---|---|
| Per-label AUROC | 831 (86.3%) | 132 (13.7%) | 0 |
| Per-label AUPRC | 888 (92.2%) | 75 (7.8%) | 0 |
| Per-label F1 | 819 (85.0%) | 143 (14.8%) | 1 (0.1%) |

Morgan wins on the majority of individual labels for every metric.  The
advantage is most pronounced for AUPRC, where Morgan leads on 92.2% of labels.

Full per-label results are in `outputs/comparison/per_label_comparison.csv`
(963 rows, columns: label_index, side_effect_id, side_effect_name,
test_positive_count, test_prevalence, morgan_\*, chemberta_\*, \*_diff, \*_winner).

---

## 5. Prediction agreement

Thresholded agreement (Morgan thr=0.31, ChemBERTa thr=0.28):

| Statistic | Value |
|---|---|
| Total pair-label decisions | 6,112,161 |
| Agreement | 5,810,355 (95.06%) |
| Both positive | 342,247 |
| Both negative | 5,468,108 |
| Morgan-only positive | 135,777 |
| ChemBERTa-only positive | 166,029 |

Probability correlation (all 6,112,161 pair-label scores):

| Statistic | Value |
|---|---|
| Pearson r | 0.8710 (p < 1e-300) |
| Spearman ρ | 0.8932 (deterministic 100k sample, seed=42) |

Top-10 label overlap per test pair (Jaccard):

| Statistic | Value |
|---|---|
| Mean Jaccard | 0.3493 |
| Median Jaccard | 0.3333 |
| Std | 0.1895 |

Both models produce strongly correlated probability estimates (Pearson
r=0.871, Spearman ρ=0.893) and agree on 95% of binary decisions.  However,
top-10 label set overlap per pair is only ~35%, indicating that the models
identify overlapping but not identical sets of high-confidence predictions.

---

## 6. Paired bootstrap (95% confidence intervals)

### 6.1 Setup and scope

500 paired resamples, seed=42.  For each replicate a single set of 6,347
test-pair indices was drawn with replacement (using `numpy.random.default_rng(42).integers`)
and applied identically to both models.  Fixed thresholds (Morgan=0.31,
ChemBERTa=0.28) were used without reselection inside each replicate.

**What the bootstrap measures:** Sampling variability over the fixed test set
for two fixed trained models.  It quantifies uncertainty due to which specific
test pairs were drawn, given these exact model weights and split.

**What the bootstrap does NOT account for:**
- Training-seed variability (both models trained once at seed=42).
- Split-assignment uncertainty (the pair-random split is fixed from Milestone 3).
- Model-selection uncertainty (threshold and checkpoint selected before test evaluation).
- Dataset uncertainty (the Decagon/TWOSIDES dataset is treated as fixed).

A second independent training run with a different seed would be required to
characterise run-to-run variability.

### 6.2 Deviation from planned metric set

The approved bootstrap contract specified: **micro AUPRC**, micro F1 at each
model's fixed threshold, and sample mean AP@50.

`micro_auprc` was **replaced with `macro_auprc`** in the executed bootstrap.
Computing `average_precision_score` on the full 6.1M-element flattened
prediction array requires ~1 s/sort per replicate per model; 500 replicates
× 2 models ≈ 17 minutes.  The vectorised per-label implementation
(`macro_auprc`, one `argsort` per label column) reduces this to ~0.43 s per
replicate per model, completing all 500 replicates in ~7 minutes.

The exact metrics stored in `bootstrap_differences.csv` are:

| Metric name in CSV | Contract metric | Match? |
|---|---|---|
| `macro_auprc` | micro AUPRC | **No — macro substituted for micro** |
| `micro_f1_selected_threshold` | micro F1 at fixed thresholds | Yes |
| `sample_mean_ap_at_50` | sample mean AP@50 | Yes |

The direction of the macro AUPRC bootstrap difference is consistent with the
observed test-set direction (Morgan macro AUPRC 0.4969, ChemBERTa 0.4490,
diff +0.0479).  This substitution is recorded in the `bootstrap_config.note`
field of `comparison_audit.json` and in the docstring of `run_bootstrap` in
`compare_models.py`.

### 6.3 Row-count verification

`bootstrap_differences.csv`: 1,500 rows total; 500 rows per metric; rep IDs
0–499 present for each metric.  Verified in `test_compare_models.py`
(TestBootstrapDeterminism).

### 6.4 Results

| Metric (CSV name) | Observed diff (M−C) | 95% percentile interval | Positive reps |
|---|---|---|---|
| `macro_auprc` | +0.04793 | [+0.04431, +0.05124] | 500 / 500 |
| `micro_f1_selected_threshold` | +0.02582 | [+0.02386, +0.02737] | 500 / 500 |
| `sample_mean_ap_at_50` | +0.04250 | [+0.03920, +0.04521] | 500 / 500 |

The Morgan-minus-ChemBERTa difference was positive in all 500 paired bootstrap
resamples for every metric.  Because no replicate produced a zero or negative
difference, no conventional empirical p-value can be computed from the
distribution.  Using the finite-sample correction `(k + 1) / (B + 1)` for the
one-sided upper bound (where k = 0 non-positive differences and B = 500
replicates), the estimated one-sided upper bound is 1/501 ≈ 0.002.  This bound
reflects the resolution limit of 500 resamples and should not be interpreted
as a precise probability.

Full bootstrap distribution: `outputs/comparison/bootstrap_differences.csv`
(1,500 rows: 500 reps × 3 metrics).  Full CI and summary in
`outputs/comparison/comparison_audit.json` under `bootstrap_summary`.

### 6.5 Reconfirmation

No model was retrained, no predictions were regenerated, and no thresholds
were changed during Milestone 8.  All analysis uses the fixed checkpoint and
prediction artifacts from Milestones 6B and 7.

---

## 7. Training history comparison

Both runs hit the 100-epoch maximum without early stopping.  Best checkpoint
was from epoch 99 in both cases.

| Property | Morgan | ChemBERTa |
|---|---|---|
| Best epoch | 99 | 99 |
| Best val macro AUPRC | 0.4966 | 0.4519 |
| Final train loss (ep 100) | 0.1357 | 0.1457 |
| Final val loss (ep 100) | ~0.1630 | 0.1645 |
| Early stopping triggered | No | No |

Both models continued improving through epoch 99/100, suggesting neither
fully converged within 100 epochs.  Additional epochs might close or widen
the gap; this is outside the scope of this analysis.

---

## 8. Scientific interpretation

### 8.1 Representation quality

Morgan circular fingerprints outperform frozen ChemBERTa pair embeddings on
every aggregate metric and on the majority of individual labels.  The margin
ranges from ~1.1% (AUROC) to ~16% (macro F1 at 0.50 threshold), with
AUPRC gaps of ~10%.

### 8.2 Why might Morgan outperform ChemBERTa here?

Several task-specific factors may explain the result:

1. **Fixed pair aggregation.** ChemBERTa pair embeddings are element-wise sums
   of two drug embeddings.  This discards information about which drug is which
   and provides no interaction-specific signal.  Morgan fingerprints are also
   combined by OR/concatenation but encode structural co-occurrence patterns
   that may correlate more directly with polypharmacy effects.

2. **64-token truncation.** 86/645 drugs (13.3%) had SMILES exceeding 64
   tokens and were silently truncated in Milestone 4.  This may degrade
   ChemBERTa embeddings for structurally complex drugs.

3. **Pre-training objective mismatch.** ChemBERTa was pre-trained with masked
   language modelling on SMILES strings, which is not directly related to the
   polypharmacy prediction task.  The 384-dim embeddings may not capture the
   pharmacological interaction signals that drive side-effect co-occurrence.

4. **Input dimensionality.** Morgan fingerprints at 2048 bits provide a richer
   explicit feature space for the MLP to learn from.  The 384-dim ChemBERTa
   embeddings are a more compact representation, and the MLP has ~851k fewer
   parameters in its first layer as a consequence.

### 8.3 Limitations of this comparison

- Both models were trained on a pair-random split: every drug in the test set
  also appears in training (with different drug partners).  Results may not
  generalise to unseen drugs (drug-level split).
- Only one run per model (no multi-seed averaging); variability is unquantified
  at the individual-run level.
- Neither model was tuned; additional hyperparameter search might alter the
  relative ranking.
- 100-epoch training may be insufficient for either or both models to converge.

### 8.4 Reproduction conclusion

> Results in this repository were produced with a fixed random seed (42),
> a deterministic pair-random split from Milestone 3, and identical training
> protocols for both models.  Given identical software versions (PyTorch
> 2.12.1, scikit-learn 1.6.1, NumPy 2.4.6, Python 3.11.9) and the same CPU
> hardware, re-running the training scripts from the saved checkpoints should
> reproduce all test metrics to within floating-point rounding.
> Cross-hardware reproducibility is not guaranteed because no GPU was used and
> PyTorch's CPU determinism depends on the underlying BLAS implementation.

---

## 9. Artifact summary

| File | Description |
|---|---|
| `outputs/comparison/aggregate_metrics.csv` | 8 metrics × 2 models with differences and winners |
| `outputs/comparison/per_label_comparison.csv` | 963-row per-label join with diffs and winners |
| `outputs/comparison/prediction_agreement.json` | Thresholded agreement, probability correlations, top-10 Jaccard |
| `outputs/comparison/bootstrap_differences.csv` | 1500 rows (500 reps × 3 metrics) |
| `outputs/comparison/comparison_audit.json` | Config hashes, bootstrap CIs, output file hashes |
| `src/polyllm/compare_models.py` | Comparison script (analysis only) |
| `tests/test_compare_models.py` | 46 tests covering all comparison components |
