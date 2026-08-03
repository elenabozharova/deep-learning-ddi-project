# Milestone 5 — Pair Embedding Findings

**Date:** 2026-06-28  
**Status:** Complete — ready for review

---

## 1. Input and configuration

| Property | Value |
|---|---|
| Drug embeddings input | `data/features/chemberta_drug_embeddings.npy` |
| Drug index input | `data/features/chemberta_drug_index.csv` |
| Pairs input | `data/processed/polyllm_pairs.parquet` |
| Drug matrix shape | (645, 384), float32 |
| Total drug pairs | 63,472 |
| pair_id range | 0 → 63,471 (sequential, verified) |

---

## 2. Combination operation

Each pair embedding is the **element-wise sum** of the two drug embeddings:

```
pair_embedding[i] = drug_embedding[drug_1] + drug_embedding[drug_2]
```

Addition is commutative, so the result is the same regardless of which drug is labelled `drug_1` vs `drug_2`. The pairs are already canonical from Milestone 1 (`drug_1 < drug_2` lexicographically), but the sum would be identical in either order.

Row `i` of the output matrix corresponds to `pair_id = i` (pairs are sorted by `pair_id` before the index lookup, so the row ordering is deterministic and independent of the order in the input parquet).

The pair embeddings are not L2-normalized.

---

## 3. Results

| Metric | Value |
|---|---|
| Output shape | **(63,472 × 384)** |
| Dtype | float32 |
| File size | ~97.5 MB |
| All values finite | ✓ |
| All-zero rows | 0 |
| L2 norm min | 4.978 |
| L2 norm median | 5.758 |
| L2 norm mean | 5.843 |
| L2 norm max | 10.595 |

The L2 norms are greater than zero for all 63,472 pairs. Norms substantially larger than the median (approaching the max of 10.60) correspond to pairs where both constituent drug embeddings happen to point in similar directions. No degenerate vectors were found.

---

## 4. Symmetry verification

A 5-pair spot-check confirmed that swapping `drug_1` and `drug_2` produces numerically identical pair embeddings (tolerance `atol=1e-6`). All 5 spot-check pairs passed.

---

## 5. Correctness spot-checks

Three pairs were independently verified:

| pair_id | drug_1 | drug_2 | Sum matches row? |
|---|---|---|---|
| 0 | CID000000085 | CID000000206 | ✓ |
| 1,000 | CID000000271 | CID000004829 | ✓ |
| 63,471 | CID005329102 | CID006398525 | ✓ |

---

## 6. Post-run disk validation

All checks passed after reloading `chemberta_pair_embeddings.npy` and `chemberta_pair_index.csv` from disk:

| Check | Result |
|---|---|
| Matrix shape == (63472, 384) | ✓ |
| Matrix dtype == float32 | ✓ |
| All values finite | ✓ |
| No all-zero rows | ✓ |
| Pair index row count == 63,472 | ✓ |
| pair_id column == 0..63,471 | ✓ |
| Row 0 of index drug_1 matches pairs_df | ✓ |

---

## 7. Output files

| File | Description |
|---|---|
| `data/features/chemberta_pair_embeddings.npy` | float32 numpy array, shape (63,472 × 384); row i = pair_id i |
| `data/features/chemberta_pair_index.csv` | 63,472 rows; columns: `pair_id, drug_1, drug_2, embedding_row`; `embedding_row == pair_id` |
| `outputs/polyllm/pair_embedding_audit.json` | Full audit: operation description, shapes, norm stats, SHA-256 hashes, software versions |
| `notes/pair_embedding_findings.md` | This document |
| `src/polyllm/features/build_pair_embeddings.py` | Implementation |
| `tests/test_build_pair_embeddings.py` | 39 tests, all synthetic data |
| `requirements-milestone5.txt` | Cumulative pinned requirements (M1–M5); no new packages |

**Artifact SHA-256:**

| File | SHA-256 |
|---|---|
| `chemberta_pair_embeddings.npy` | `85053b56b13ffb9c38a98addd19a388061c94ba9ad6185efa2d4e844c3719de1` |
| `chemberta_pair_index.csv` | `890e1f76af07ac735d21cb2e7a19b1231cad4de3ec7f0a988374043033097cd4` |

---

## 8. Tests

**Command:** `.venv-polyllm\Scripts\python.exe -m pytest tests/ -q`  
**Result:** **209 / 209 tests passed** (39 Milestone 5 + 170 Milestones 1–4)

Milestone 5 test coverage:

| Class | Topics covered |
|---|---|
| `TestValidateDrugInputs` | Missing column; 1-D matrix; wrong dtype; row mismatch; duplicate stitch_ids; non-sequential embedding_index; valid inputs pass |
| `TestValidatePairsDf` | Missing column; duplicate pair_ids; non-sequential pair_ids; unknown drug; valid pairs pass |
| `TestSumOperation` | Pair is sum of drug embeddings; output shape; output dtype float32; unsorted input gives same result as sorted |
| `TestSymmetry` | Swapping drug_1/drug_2 gives identical embedding; `verify_symmetry` passes |
| `TestInMemoryValidation` | Wrong shape; wrong dtype; NaN; all-zero row; valid matrix passes |
| `TestDiskValidation` | Valid artifacts pass; wrong shape; non-finite; wrong index rows |
| `TestPairIndex` | Schema; sorted by pair_id; embedding_row == pair_id; row count |
| `TestAuditFields` | Required keys present; `symmetric` flag true; operation mentions sum |
| `TestDeterminism` | Identical output on repeated calls |
| `TestOverwriteGuard` | Skips full pipeline when artifacts exist; result contains `pair_matrix_shape` |
| `TestSha256` | Deterministic; known value |
| `TestPipelineIntegration` | End-to-end with synthetic data: correct shape, dtype, row spot-check |

---

## 9. Software versions

| Package | Version |
|---|---|
| Python | 3.11.9 |
| NumPy | 2.4.6 |
| pandas | 3.0.3 |
| PyArrow | 24.0.0 (Parquet read) |

No new packages were required for Milestone 5 beyond those already installed in Milestones 1–4.

---

## 10. Known limitations

- **Sum is a fixed representation choice.** The element-wise sum is symmetric and preserves the same dimensionality as the per-drug embeddings. It does not model asymmetric interactions. Alternative combination methods (difference, concatenation, Hadamard product) were not explored; they are out of scope for this milestone.

- **Pair embeddings inherit truncation effects.** 86 of 645 drugs (13.33%) were truncated at 64 tokens in Milestone 4. Their pair embeddings are derived from incomplete SMILES representations; this limitation carries forward.

- **Pair embeddings are not normalized.** L2 norms range from 4.98 to 10.60. If a downstream model is sensitive to vector magnitude, normalization or standardization may be appropriate at that stage.

- **No labels are attached to this file.** The label matrix (`data/processed/polyllm_labels.npy`) and the split assignments (`data/splits/`) are separate artifacts. Row i of the pair embedding matrix corresponds to pair_id i in the pairs table, which can be joined with the labels using the pair_id as the key.
