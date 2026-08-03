# Milestone 6A — Morgan Fingerprint Feature Findings

**Date:** 2026-06-28  
**Status:** Complete — ready for review

---

## 1. Purpose and baseline rationale

Morgan fingerprints are **deterministic, handcrafted molecular descriptors** computed directly from a SMILES string using a circular hashing algorithm. They are not learned embeddings; no model training is involved in their construction. Each bit encodes the presence of a specific circular substructure centred on a given atom up to a specified bond radius.

They serve as the **conventional molecular baseline** in this project for the following reasons:

- They are among the most widely used molecular features in computational chemistry and cheminformatics.
- They are fully reproducible given the SMILES string and hyperparameters; there is no stochastic component.
- They capture local chemical environments (functional groups, ring systems, short connectivity paths) in a fixed-length binary vector.
- Comparing them against the ChemBERTa learned embeddings (Milestone 4) provides a controlled assessment of how much the language-model representation adds beyond classical molecular descriptors.

---

## 2. RDKit fingerprint parameters

| Parameter | Value |
|---|---|
| Generator | `rdFingerprintGenerator.GetMorganGenerator` |
| `radius` | **2** (each bit encodes atoms within 2 bonds of a centre) |
| Diameter | **4** (radius × 2; ECFP4-style convention) |
| `fpSize` | **2048** bits |
| `includeChirality` | **True** |
| SMILES column | `standardized_smiles` (RDKit canonical isomeric SMILES from Milestone 2) |
| RDKit version | 2026.03.3 |

The radius-2 setting is equivalent to ECFP4 in the original extended-connectivity fingerprint (ECFP) nomenclature, where the number denotes the diameter. Chirality is included because the Decagon drug set contains stereoisomers whose chiral centers are pharmacologically relevant.

Each SMILES is re-validated with `Chem.MolFromSmiles()` before fingerprint generation. A `None` return raises an error. All 645 drugs passed validation.

---

## 3. Drug fingerprints

| Metric | Value |
|---|---|
| Output shape | **(645, 2048)** |
| Dtype | uint8 |
| Allowed values | {0, 1} — binary |
| All-zero drug rows | 0 |
| Active bits per drug — min | 1 |
| Active bits per drug — median | 44 |
| Active bits per drug — mean | 44.75 |
| Active bits per drug — max | 141 |
| Drug ordering | Sorted lexicographically by `stitch_id` → `fingerprint_index` 0–644 |

The minimum active-bit count of 1 belongs to a very small molecule (e.g. selenium sulfide, CID000024011, SMILES `S=[Se]`). The maximum of 141 belongs to a structurally complex molecule.

---

## 4. Pair fingerprints

### 4.1 Combination operation

```
pair_fingerprint[i] = fingerprint[drug_1] + fingerprint[drug_2]
```

This is an element-wise integer **sum** over the two binary drug fingerprints. It is **not** bitwise OR (which would collapse the 1+1 case to 1), and it is **not** concatenation (which would double the vector length). The sum was chosen because:

- It is symmetric: drug_1 + drug_2 = drug_2 + drug_1.
- It preserves the count of drugs that carry each bit: a value of 0 means neither drug has the substructure; 1 means exactly one does; 2 means both do.
- It keeps the same 2048-dimensional space as the drug fingerprints, matching the dimensionality of the ChemBERTa pair embeddings.
- It is consistent with the element-wise sum used for ChemBERTa pair embeddings in Milestone 5.

### 4.2 Pair matrix properties

| Metric | Value |
|---|---|
| Output shape | **(63,472, 2048)** |
| Dtype | uint8 |
| Allowed values | {0, 1, 2} |
| All-zero pair rows | 0 |
| Row alignment | Row i = pair_id i |
| Pair vectors not L2-normalized | ✓ |

### 4.3 Pair value distribution

| Value | Count | Proportion |
|---|---|---|
| 0 (bit absent in both drugs) | 124,852,144 | 96.047% |
| 1 (bit present in exactly one drug) | 4,641,797 | 3.571% |
| 2 (bit present in both drugs) | 496,715 | 0.382% |

The sparsity of value 2 (0.38%) reflects that two randomly paired drugs share few identical circular substructures, which is expected for structurally diverse drug pairs.

---

## 5. Validation

### 5.1 In-memory validation

| Check | Result |
|---|---|
| Drug matrix shape == (645, 2048) | ✓ |
| Drug matrix dtype == uint8 | ✓ |
| All drug values in {0, 1} | ✓ |
| No all-zero drug rows | ✓ |
| Pair matrix shape == (63,472, 2048) | ✓ |
| Pair matrix dtype == uint8 | ✓ |
| All pair values in {0, 1, 2} | ✓ |
| No all-zero pair rows | ✓ |

### 5.2 Post-run disk validation

All five artifact files were reloaded from disk and re-checked against the same conditions. All checks passed.

### 5.3 Spot-checks

Five pairs were independently verified by directly summing the saved drug fingerprints and comparing with the corresponding pair matrix row:

| pair_id | drug_1 | drug_2 | Match? |
|---|---|---|---|
| 0 | (first pair) | | ✓ |
| 1,000 | | | ✓ |
| 10,000 | | | ✓ |
| 50,000 | | | ✓ |
| 63,471 | (last pair) | | ✓ |

An additional 5-pair symmetry check (swap drug_1/drug_2) passed.

---

## 6. Output files

| File | Description |
|---|---|
| `data/features/morgan_drug_fingerprints.npy` | uint8 array, shape (645, 2048), values {0, 1} |
| `data/features/morgan_drug_index.csv` | 645 rows; columns: `fingerprint_index, stitch_id, parsed_pubchem_cid, smiles_sha256, active_bit_count` |
| `data/features/morgan_pair_fingerprints.npy` | uint8 array, shape (63,472, 2048), values {0, 1, 2} |
| `data/features/morgan_pair_index.csv` | 63,472 rows; columns: `pair_id, drug_1, drug_2, drug_1_fingerprint_index, drug_2_fingerprint_index` |
| `outputs/polyllm/morgan_feature_audit.json` | Full audit: fingerprint config, shapes, value stats, SHA-256 hashes |
| `notes/morgan_feature_findings.md` | This document |
| `src/polyllm/features/build_morgan_features.py` | Implementation |
| `tests/test_build_morgan_features.py` | 32 tests, all synthetic data |
| `requirements-milestone6a.txt` | Cumulative requirements; no new packages |

**Artifact SHA-256:**

| File | SHA-256 |
|---|---|
| `morgan_drug_fingerprints.npy` | `f315429eff6ee9e2d4518613521e91e51485334bfeeee8ee13a11ec8ede0e4e2` |
| `morgan_drug_index.csv` | `e3f2c71d81a1e9a9d42ecd8ef4d1d625786102769158f6d65bf418d41cd8e0a4` |
| `morgan_pair_fingerprints.npy` | `5bb5a57575e585aaa95540849b9c56398002239b98b5d20b9c7bac0a98e2c9d2` |
| `morgan_pair_index.csv` | `403427097a13c5ea9ebe75d7fc9cd1dbd962a286b6420fcdc79a1b0189341f62` |

---

## 7. Overwrite protection

Re-running without `--overwrite` when all four artifact files are present and consistent leaves the files with identical SHA-256 hashes and unchanged modification times. Confirmed.

---

## 8. Tests

**Command:** `.venv-polyllm\Scripts\python.exe -m pytest tests/ -q`  
**Result:** **241 / 241 tests passed** (32 Milestone 6A + 209 Milestones 1–5)

| Class | Scenarios covered |
|---|---|
| `TestSmilesValidation` | Invalid SMILES raises; null SMILES raises; valid SMILES pass |
| `TestDuplicateStitchId` | Duplicate stitch_id in mapping raises |
| `TestMissingPairDrug` | Drug in pairs absent from mapping raises; missing column raises |
| `TestDeterministicDrugOrdering` | Drugs sorted by stitch_id; shuffled input gives same matrix |
| `TestDeterministicPairOrdering` | Reversed pair input gives same matrix |
| `TestFingerprintSize` | Drug fp size 2048; pair fp size 2048 |
| `TestDrugValuesBinary` | All values 0 or 1; dtype uint8 |
| `TestPairValues` | Values in {0,1,2}; dtype uint8; invalid value raises |
| `TestSymmetry` | Swapped drug order gives same fingerprint; verify_symmetry passes |
| `TestPairEqualsDirectAddition` | All pairs verified against direct drug fp sum |
| `TestRowAlignment` | Drug row i = fingerprint_index i; pair row i = pair_id i |
| `TestDtype` | Drug uint8; pair uint8; wrong dtypes raise |
| `TestOutputDirCreation` | Nested output directory created atomically |
| `TestDiskValidation` | Valid artifacts pass; bad drug shape raises; non-binary values raise |
| `TestOverwriteProtection` | Pipeline skips when all artifacts exist |
| `TestReproducibility` | Two runs with overwrite=True produce identical SHA-256 hashes |

---

## 9. Software versions

| Package | Version |
|---|---|
| Python | 3.11.9 |
| RDKit | 2026.03.3 |
| NumPy | 2.4.6 |
| pandas | 3.0.3 |

No new packages were required for Milestone 6A.

---

## 10. Known limitations

- **Fixed hyperparameters.** Radius, fingerprint size, and chirality setting were specified and were not tuned. Different radii (e.g. radius=3 for ECFP6) or larger sizes (e.g. 4096 bits) were not explored.

- **No folding or hashing collision analysis.** At 2048 bits with radius-2 features from 645 diverse drugs, bit collisions are possible but not quantified.

- **Minimum active-bit count is 1.** One drug has only 1 active bit. This is chemically valid (very small molecule) but means its fingerprint carries very limited structural information.

- **Pair sum does not distinguish which drug carries which bit.** For bits with pair value 2, the information that both drugs share that substructure is preserved, but it is not possible to recover which drug is drug_1 and which is drug_2 from the pair fingerprint alone (which is by design — the representation is symmetric).

- **Not a learned representation.** Morgan fingerprints encode fixed structural patterns and cannot capture context-dependent or semantically abstract molecular properties the way a language model might. Their comparison with ChemBERTa embeddings is the purpose of the baseline.
