# Milestone 2 — SMILES Mapping Findings

**Date:** 2026-06-27  
**Status:** Complete — ready for review

---

## 1. Input and STITCH format

**Input:** `data/processed/polyllm_pairs.parquet` (Milestone 1 output, not modified)  
**Pairs loaded:** 63,472  
**Unique STITCH IDs extracted:** 645

All 645 STITCH identifiers match the pattern `^CID(\d{9})$` exactly.

- **Format:** literal prefix `CID` followed by exactly 9 decimal digits, zero-padded (e.g. `CID000000085`, `CID009571074`)
- **Unrecognized formats:** 0
- **CID integer range:** 85 → 9,571,074

**Conversion rule:**  
Remove the 3-character prefix `CID`, then call `int()` on the remaining 9-digit string. `int()` drops leading zeros automatically; no numeric information is lost. Example: `CID000000085` → `int("000000085")` → `85`.

---

## 2. Representative validation

Before the batch query, 15 STITCH IDs were selected deterministically to validate the parsing rule. Selection method: indices 0, n−1, and 13 evenly spaced interior indices across the 645 sorted IDs.

| Outcome | Count |
|---|---|
| Found in PubChem with valid SMILES | 15 |
| Not found in PubChem | 0 |
| Request failures | 0 |

For each sample, the integer CID computed by the parsing rule was used as the GET URL path. PubChem returned a record for all 15. The returned `CID` field in each JSON response matched the parsed CID exactly. No contradictions with the rule were observed.

Representative results (stored in `outputs/polyllm/smiles_mapping_audit.json` → `representative_validation.representative_results`):

| STITCH ID | Parsed CID | PubChem name (truncated) |
|---|---|---|
| CID000000085 | 85 | (3-carboxy-2-hydroxypropyl)-trimethylazanium |
| CID000003657 | 3657 | hydroxyurea |
| CID000004873 | 4873 | potassium chloride |
| CID000051263 | 51263 | N-[1-[2-(4-ethyl-5-oxotetrazol-1-yl)...] |
| CID009571074 | 9571074 | (6R)-7-[...cefdinir...] |

---

## 3. Mapping results

| Metric | Value |
|---|---|
| Total unique STITCH IDs | 645 |
| Successfully mapped | **645 (100%)** |
| `pubchem_not_found` | 0 |
| `invalid_smiles` | 0 |
| `missing_smiles` | 0 |
| `invalid_stitch_format` | 0 |
| `request_failed` | 0 |

All 645 drugs have status `mapped`. Every row has a non-null `pubchem_smiles`, `standardized_smiles`, and `inchikey`. Two drugs (CID000024011 and CID006398525) have a null `preferred_name`; PubChem does not provide an IUPAC name for these compounds (selenium sulfide and sucralfate complex). Both are fully mapped with valid SMILES and InChIKey.

---

## 4. PubChem API behavior

**Requested properties:** `IsomericSMILES, IUPACName, InChIKey`  
**URL pattern:** `https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/property/IsomericSMILES,IUPACName,InChIKey/JSON`

**Observed response:** The JSON property key for the SMILES field is `"SMILES"`, not `"IsomericSMILES"`.

```json
{
  "PropertyTable": {
    "Properties": [{
      "CID": 2244,
      "SMILES": "CC(=O)OC1=CC=CC=C1C(=O)O",
      "InChIKey": "BSYNRYMUTXBXSQ-UHFFFAOYSA-N",
      "IUPACName": "2-acetyloxybenzoic acid"
    }]
  }
}
```

This was observed for both the individual GET endpoint and the batch POST endpoint, across all tested compound types. The value is the SMILES string for the PubChem parent compound (the default isomeric representation where stereo information is available). This behavior was verified on 2026-06-27.

**Note:** The renaming of the response key from `IsomericSMILES` to `SMILES` does not necessarily mean stereochemistry was removed. For compounds where the parent compound record in PubChem has no defined stereocenters, the SMILES naturally contains no `@` or `/\` markers regardless of the key name. Separate verification of stereospecific forms (via SID or stereoisomer CIDs) was not performed.

The implementation and tests were updated to look up `"SMILES"` in all response dicts. The first pipeline run used the old key name and reported 0% mapping; after the fix, the second run achieved 100%.

---

## 5. RDKit validation and canonicalization

Each raw `pubchem_smiles` value was processed as follows:

1. `mol = Chem.MolFromSmiles(smiles)` — validate the SMILES string
2. If `mol is None`, record status `invalid_smiles`
3. Otherwise: `canonical = Chem.MolToSmiles(mol, isomericSmiles=True)` — produce canonical form

The result stored in `standardized_smiles` is the **RDKit canonical isomeric SMILES** (`isomericSmiles=True`). RDKit C++ warnings were suppressed via `RDLogger.DisableLog("rdApp.*")`.

| Metric | Value |
|---|---|
| Passed RDKit validation | **645 / 645** |
| Failed RDKit validation | 0 |
| SMILES length — minimum | 5 characters |
| SMILES length — mean | 49.8 characters |
| SMILES length — maximum | 382 characters |

**What this canonicalization does and does not do:**  
RDKit canonicalization produces a unique SMILES string under a fixed atom-ordering algorithm and preserves existing stereochemistry markers. It does **not** perform full chemical standardization: no salt stripping, no tautomer normalization, no charge normalization, no parent compound selection, and no merging of stereoisomers. The `standardized_smiles` values reflect the PubChem parent compound structure, passed through RDKit's SMILES canonicalizer only.

---

## 6. Structure uniqueness

All uniqueness checks were performed on the 645 `mapped` rows only.

| Check | Unique groups | Duplicate groups (2+ members) |
|---|---|---|
| PubChem CID | 645 | **0** |
| InChIKey | 645 | **0** |
| RDKit canonical SMILES | 645 | **0** |

Each of the 645 STITCH IDs maps to a distinct PubChem record by all three identifiers. No two STITCH IDs represent the same compound under any of these criteria.

**Interpretation:** This result shows the absence of exact duplicates under the PubChem parent compound representation and the RDKit canonical form. It does not prove that all compounds are pharmacologically unrelated, nor that no two drugs share a scaffold, pharmacophore, or mechanism. Drugs with the same active moiety but different salts or esters could resolve to different CIDs and different SMILES.

---

## 7. Pair eligibility

| Metric | Value |
|---|---|
| Total canonical pairs (from Milestone 1) | 63,472 |
| Pairs with both drugs mapped | **63,472 (100%)** |
| Pairs with at least one unmapped drug | 0 |

Since all 645 unique drugs are mapped, every pair in the Milestone 1 pair table has a SMILES string for both members. No pairs need to be excluded for missing molecular structures.

**The Milestone 1 pair table has not been modified.** Pair features (fingerprints, graph embeddings, etc.) are deferred to later milestones.

---

## 8. Cache and network behavior

| Property | Value |
|---|---|
| Cache file | `outputs/polyllm/pubchem_cache.json` |
| Cache key | PubChem CID as a string (e.g. `"2244"`) |
| Cache value | Raw property dict returned by PubChem for that CID |
| Write strategy | Atomic: temp file + `os.replace()` after each batch |

**First run (incorrect key):** 7 batches queried, 645 CIDs fetched, 0 cache hits (cold start). Responses cached. Mapping reported 0% due to `IsomericSMILES` key mismatch.

**Cache cleared** manually after the key-name fix.

**Second run (correct key):** 7 batches queried, 645 CIDs fetched, 0 cache hits (cache was cleared). All 645 mapped. Cache now fully warm.

**Rerun behavior:** Any subsequent rerun will read all 645 responses from the cache and skip all HTTP requests. The representative validation still queries PubChem individually (not cached).

**Tests:** All tests use `unittest.mock.patch` to intercept `requests.get` and `requests.post`. No live network access occurs during `pytest`.

---

## 9. Tests

**Command:** `.venv-polyllm\Scripts\python.exe -m pytest tests/ -q`  
**Result:** **88 / 88 tests passed** (42 Milestone 2 + 46 Milestone 1)

Milestone 2 test coverage by class:

| Class | Topics covered |
|---|---|
| `TestExtractUniqueDrugs` | Sorted unique extraction from both columns; empty DataFrame |
| `TestStitchFormat` | Valid pattern detection; invalid patterns counted; CID integer range |
| `TestParseCID` | Leading-zero stripping via `int()`; 9-digit boundary; returns `int` not `None` |
| `TestDeterministicOrdering` | `extract_unique_drugs` is sorted; mapping output is sorted by `stitch_id` |
| `TestSuccessfulParsing` | GET response parsed correctly; batch POST parsed correctly |
| `TestMissingRecords` | HTTP 404 → `{}`; `pubchem_not_found` status in mapping |
| `TestRetries` | HTTP 503 → retry → success; all retries exhausted → `None`; `Timeout` → retry |
| `TestInvalidSMILES` | Bad/empty/None SMILES → `(False, None)`; `invalid_smiles` status recorded |
| `TestRDKitCanonicalization` | Valid SMILES → canonical string; tautomer identity; stereo markers preserved |
| `TestDuplicateDetection` | Duplicate CID/InChIKey/SMILES groups detected; no-dup case; unmapped excluded |
| `TestCacheReuse` | Cache hit prevents HTTP call; cache saved after fetch; roundtrip; missing file → `{}` |
| `TestOutputSchema` | Required CSV columns present; sorted order; audit keys; unrecognized format raises |
| `TestPairEligibility` | Both mapped counted; one unmapped excludes pair |

All mocked HTTP responses use `"SMILES"` as the property key, matching the actual PubChem REST API.

---

## 10. Software versions

| Package | Version |
|---|---|
| Python | 3.11.9 |
| pandas | 3.0.3 |
| NumPy | 2.4.6 |
| PyArrow | 24.0.0 |
| requests | 2.34.2 |
| RDKit | 2026.03.3 |

---

## 11. Outputs

| File | Description |
|---|---|
| `data/processed/drug_smiles_mapping.csv` | 645 rows × 9 columns; sorted by `stitch_id`; all statuses `mapped` |
| `outputs/polyllm/smiles_mapping_audit.json` | Full audit: format analysis, representative validation, status counts, duplicate groups, pair eligibility, software versions |
| `outputs/polyllm/pubchem_cache.json` | Raw PubChem responses keyed by CID string; warm for all 645 CIDs |
| `notes/smiles_mapping_findings.md` | This document |
| `src/polyllm/data/map_drugs_to_smiles.py` | Implementation (~430 lines); CLI entry point |
| `tests/test_map_drugs_to_smiles.py` | 42 tests; all HTTP mocked |
| `requirements-milestone2.txt` | Pinned dependency set for the Milestone 2 environment |

---

## 12. Known limitations

- **Mapping depends on current PubChem records.** PubChem compound records may be updated or reorganized after the query date (2026-06-27). The on-disk cache preserves a snapshot of the responses but is not version-stamped with the PubChem data release.

- **PubChem compound records may encode salts, mixtures, parents, or unspecified stereochemistry.** The parent compound CID for a drug in the Decagon dataset may represent a salt form, a racemic mixture, or a structure without defined stereocenters. For example, CID 6398525 (sucralfate) returns a large aluminum-sucrose sulfate complex as its SMILES. No normalization policy has been applied beyond RDKit SMILES canonicalization.

- **No further structure normalization.** The `standardized_smiles` column reflects RDKit canonical isomeric SMILES only. No salt stripping, tautomer normalization, charge normalization, parent fragment selection, or stereoisomer enumeration has been performed. These steps are deferred to a dedicated standardization stage if required.

- **Two drugs lack preferred names.** CID000024011 (selenium sulfide, `S=[Se]`) and CID006398525 (sucralfate complex) have no IUPAC name in PubChem. Their SMILES and InChIKey are valid.

- **The original TWOSIDES dataset source URL remains unresolved provenance metadata.** This gap was identified in Milestone 0 (decision D10) and carries over to Milestone 2. The PubChem mapping is internally consistent but the provenance of the upstream dataset is not fully documented.

- **No ChemBERTa embeddings, pair fingerprints, data splits, or models have been created.** This milestone produced only SMILES strings and validated their chemical correctness. Molecular feature computation and model training are deferred to later milestones.

---

## Post-run validation summary

All 9 checks passed after the second run:

| Check | Result |
|---|---|
| Exactly 645 rows in `drug_smiles_mapping.csv` | ✓ |
| One row per unique `stitch_id` | ✓ |
| All `mapping_status` values are `mapped` | ✓ |
| `parsed_pubchem_cid` equals `int(stitch_id[3:])` for every row | ✓ |
| No null `pubchem_smiles`, `standardized_smiles`, or `inchikey` | ✓ |
| Every `standardized_smiles` re-parses with RDKit | ✓ |
| `stitch_id` column is sorted lexicographically | ✓ |
| Audit JSON totals match CSV row counts | ✓ |
| All 63,472 pairs resolve to two `mapped` STITCH IDs | ✓ |

**Milestone 2 is ready for review.**
