# Milestone 3 — Split Findings

**Date:** 2026-06-27  
**Status:** Complete — ready for review

---

## 1. Input and configuration

| Property | Value |
|---|---|
| Input: pair table | `data/processed/polyllm_pairs.parquet` |
| Input: label matrix | `data/processed/polyllm_labels.npy` |
| Input: label mapping | `data/processed/polyllm_label_mapping.csv` |
| Total pairs | 63,472 |
| pair_id range | 0 → 63,471 (sequential, verified) |
| Label matrix shape | (63,472, 963) dtype uint8 |
| Split method | pair-random |
| Random seed | 42 (`np.random.default_rng(42)`) |
| Target fractions | train 80% / validation 10% / test 10% |

---

## 2. Exact split sizes

**Rounding rule:**

```
n_test       = round(0.10 × 63,472) = round(6,347.2) = 6,347
n_validation = round(0.10 × 63,472) = round(6,347.2) = 6,347
n_train      = 63,472 − 6,347 − 6,347 = 50,778
```

| Split | Pairs | Percentage |
|---|---|---|
| **train** | **50,778** | 80.0006% |
| **validation** | **6,347** | 9.9997% |
| **test** | **6,347** | 9.9997% |
| **total** | **63,472** | 100% |

**Procedure:**

1. Copy the full sequential pair-ID array `[0, 1, …, 63,471]`.
2. Shuffle in-place with `rng = np.random.default_rng(42)`.
3. Assign: first 50,778 IDs → train; next 6,347 → validation; remaining 6,347 → test.
4. Sort each split's IDs numerically (cosmetic only — does not change membership).
5. Write each split to a single-column CSV (`pair_id`).

---

## 3. Confirmation of no overlap and complete coverage

All 9 validation checks passed in memory and after disk reload:

| Check | Result |
|---|---|
| No duplicate pair IDs within train | ✓ |
| No duplicate pair IDs within validation | ✓ |
| No duplicate pair IDs within test | ✓ |
| train ∩ validation = empty | ✓ |
| train ∩ test = empty | ✓ |
| validation ∩ test = empty | ✓ |
| union(train, validation, test) = all 63,472 pair IDs | ✓ |
| All split IDs exist in the pair table | ✓ |
| Every pair ID appears exactly once across splits | ✓ |

Disk-reload validation: the three CSVs were reloaded and re-checked with the same 9 conditions — all passed.

---

## 4. Label-distribution findings

### 4.1 Per-split summary

| Metric | Full dataset | Train | Validation | Test |
|---|---|---|---|---|
| Pair count | 63,472 | 50,778 | 6,347 | 6,347 |
| Total positive entries | 4,576,287 | 3,670,586 | 456,004 | 449,697 |
| Mean labels per pair | 72.10 | 72.29 | 71.85 | 70.85 |
| Median labels per pair | 52.0 | 52.0 | 52.0 | 51.0 |
| Min labels per pair | 1 | 1 | 1 | 1 |
| Max labels per pair | 505 | 505 | 435 | 443 |
| Labels with ≥ 1 positive | 963 | 963 | **963** | **963** |
| Labels with 0 positives | **0** | **0** | **0** | **0** |
| Min positives per label | 502 | 398 | 40 | 36 |
| Median positives per label | 2,970 | 2,365 | 294 | 295 |
| Max positives per label | 28,568 | 22,891 | 2,848 | 2,829 |

### 4.2 Labels absent from validation or test

**Zero.** All 963 retained labels have at least one positive example in the validation split and at least one positive example in the test split. No label-absence issues to report.

Note: the minimum positives per label in the validation and test splits are 40 and 36 respectively — small but non-zero. The rarest labels in validation/test are those with ~500 total examples in the full dataset (the minimum across all labels is 502).

### 4.3 Prevalence comparison with full dataset

Prevalence = (positive examples for label j in split) / (pairs in split).

| Comparison | Max absolute prevalence diff | Mean absolute prevalence diff |
|---|---|---|
| Train vs full dataset | 0.002189 | 0.000381 |
| Validation vs full dataset | 0.01332 | 0.002125 |
| Test vs full dataset | 0.017006 | 0.002352 |

**Most deviant label per split:**

| Split | Label index | Full prevalence | Split prevalence | Abs diff |
|---|---|---|---|---|
| Train | 957 | 0.1226 | 0.1248 | 0.0022 |
| Validation | 366 | 0.2003 | 0.2136 | 0.0133 |
| Test | 169 | 0.3009 | 0.2839 | 0.0170 |

The maximum absolute prevalence difference across all labels and splits is **0.017** (label 169, test split). This level of imbalance is expected for a random pair split with 10% test fraction and ~6,347 test pairs. The split was not modified to improve label balance.

---

## 5. Drug-overlap findings

| Metric | Value |
|---|---|
| Unique drugs in train | 645 / 645 (100%) |
| Unique drugs in validation | 622 / 645 (96.4%) |
| Unique drugs in test | 619 / 645 (95.8%) |
| Total unique drugs across all splits | 645 |
| Drugs appearing in all three splits | 609 |
| Drugs unique to train (not in val or test) | 13 |
| Drugs unique to validation (not in train or test) | 0 |
| Drugs unique to test (not in train or validation) | 0 |
| Test drugs also in train | 619 / 619 (100%) |

**Test-pair breakdown by drug familiarity:**

| Category | Pairs | Percentage |
|---|---|---|
| Both drugs seen in train | **6,347** | **100.0%** |
| Exactly one drug seen in train | 0 | 0.0% |
| Neither drug seen in train | 0 | 0.0% |

**Interpretation:**  
Every test pair consists of two drugs that also appear in training. 13 drugs appear only in train (paired only with partners that ended up entirely in train). No drug appears exclusively in validation or test. This split therefore evaluates **unseen drug combinations involving known drugs**, not generalization to completely novel drugs. A model evaluated on this split will not be tested on its ability to predict interactions for a drug it has never seen.

---

## 6. Output files

| File | Description |
|---|---|
| `data/splits/train_pair_ids.csv` | 50,778 rows; single column `pair_id`; sorted numerically |
| `data/splits/validation_pair_ids.csv` | 6,347 rows; single column `pair_id`; sorted numerically |
| `data/splits/test_pair_ids.csv` | 6,347 rows; single column `pair_id`; sorted numerically |
| `outputs/polyllm/split_audit.json` | Full audit: sizes, validation checks, label distribution, drug overlap, software versions |
| `notes/split_findings.md` | This document |
| `src/polyllm/data/create_splits.py` | Implementation |
| `tests/test_create_splits.py` | 37 tests; all synthetic data |

The split CSVs contain **only pair IDs**. Features, labels, SMILES, and drug names are not copied into the split files. To access labels for a split, index `polyllm_labels.npy` with the loaded pair IDs.

---

## 7. Tests

**Command:** `.venv-polyllm\Scripts\python.exe -m pytest tests/ -q`  
**Result:** **125 / 125 tests passed** (37 Milestone 3 + 88 Milestone 1–2)

Milestone 3 test coverage:

| Class | Topics covered |
|---|---|
| `TestSplitSizes` | Exact sizes for n=100 and n=63,472; total always equals n; non-negative |
| `TestDeterministicSplit` | Seed 42 fixed output; sorted IDs; shuffled ≠ sequential |
| `TestCoverageAndOverlap` | Union = all; three pairwise disjoint checks; validate_splits passes |
| `TestNoDuplicates` | No duplicate IDs within any split |
| `TestInputValidation` | Non-sequential IDs raise; duplicates raise; missing column raises; wrong start index raises |
| `TestLabelMatrixValidation` | Row count mismatch raises; 1-D array raises; valid matrix passes |
| `TestDiskValidation` | CSV roundtrip validates; only `pair_id` column; values sorted; wrong columns raise |
| `TestDeterminism` | Same seed → identical membership; different seed → different |
| `TestSplitFileSchema` | Full pipeline: all three CSVs have only `pair_id` column |
| `TestLabelAndDrugStats` | Pair counts sum to total; labels_with_positives ≤ n_labels; prevalence length; drug overlap totals; test-pair categories sum to n_test; drugs_in_all_three ≤ min split; audit keys; train+val+test=total; overlap detection; missing-ids detection |

---

## 8. Software versions

| Package | Version |
|---|---|
| Python | 3.11.9 |
| NumPy | 2.4.6 |
| pandas | 3.0.3 |
| PyArrow | 24.0.0 (Parquet read) |

---

## 9. Known limitations

- **Pair-random split, not drug-disjoint.** All 619 test drugs also appear in training. The split measures generalization to new drug-pair *combinations*, not to new drugs. A drug-disjoint split would prevent any test drug from appearing in training, at the cost of substantially reduced training data. The user has not requested a drug-disjoint split.

- **No iterative multilabel stratification.** Label prevalences differ from the full-dataset baseline by up to 0.017 (label 169, test split). This is the expected variance from random assignment. If future evaluation requires tighter label balance, a stratified split (e.g. iterative stratification) could reduce this but would require an additional implementation step.

- **Split is fixed and permanent.** The CSVs were produced with seed 42. Any subsequent re-generation with the same seed produces identical membership. If the pair table changes (e.g. after reprocessing), the splits must be regenerated.

- **No feature or model work performed.** This milestone produces only pair ID assignments. ChemBERTa embeddings, Morgan fingerprints, pair features, negative sampling, and model training are all deferred.

- **The original TWOSIDES dataset source URL remains unresolved.** This provenance gap (D10 from Milestone 0) carries forward.
