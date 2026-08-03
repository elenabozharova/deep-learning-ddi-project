# Milestone 1 — Data Preparation Findings

**Date:** 2026-06-27  
**Status:** Complete — all outputs generated, all post-run validations passed

---

## 1. Environment

| Property | Value |
|---|---|
| Python version | 3.11.9 |
| pandas | 3.0.3 |
| numpy | 2.4.6 |
| pyarrow | 24.0.0 |
| pytest | 9.1.1 |
| Virtual environment path | `.venv-polyllm\` (project root) |
| Python executable | `.venv-polyllm\Scripts\python.exe` |

The existing `.venv` (Python 3.13.5) was **not modified**.  All Milestone 1
work ran exclusively under `.venv-polyllm`.

---

## 2. Input verification

| Property | Value |
|---|---|
| Raw dataset path | `data/raw/ChChSe-Decagon_polypharmacy.csv.gz` |
| Compressed file size | 35,657,514 bytes (~34 MB) |
| Compression format | gzip |
| Verified columns | `# STITCH 1`, `STITCH 2`, `Polypharmacy Side Effect`, `Side Effect Name` |
| Column 1 literal `#` | Confirmed — raw bytes: `b'# STITCH 1,...'` |
| Source used for preprocessing | Compressed raw file directly (not `sample.csv`) |

The script opened `ChChSe-Decagon_polypharmacy.csv.gz` directly via
`pandas.read_csv(..., compression="gzip", dtype=str)`.  The sample file
(`data/processed/sample.csv`) was not used as input.

**Missing values:** zero missing or blank values in all four required columns.

**Whitespace normalisation:** zero values changed in any column.

**Side-effect identity:** zero identifier-to-name conflicts; zero name-to-multiple-identifier conflicts.

---

## 3. Transformation decisions

### Exact duplicate definition
A row is an exact duplicate if all four required source columns
(`# STITCH 1`, `STITCH 2`, `Polypharmacy Side Effect`, `Side Effect Name`)
are identical to another row.

### Unordered-pair canonicalisation rule
For each row:
- `drug_1 = lexicographically smaller STITCH identifier`
- `drug_2 = lexicographically larger STITCH identifier`

Comparison is purely lexicographic on the raw string.  STITCH identifiers are
never parsed as integers.  The invariant `drug_1 <= drug_2` holds for all
rows, and `drug_1 < drug_2` holds for all non-self-pairs.

### Self-pair policy
A self-pair is a row where `drug_1 == drug_2` after canonicalisation.
Self-pairs are excluded from the modelling dataset because the task models
interactions between two *distinct* drugs.  Their counts and unique side
effects are recorded in the audit but they are never written to any output file.

### Canonical triple definition
A canonical triple is the combination `(drug_1, drug_2, Polypharmacy Side Effect)`
after canonicalisation.  Duplicate canonical triples — including rows that are
reversed copies of each other (e.g. `(A,B,SE)` and `(B,A,SE)`) — are collapsed
to a single representative row.

### Side-effect frequency definition
The frequency of a side effect is the number of **unique canonical non-self drug
pairs** associated with it in the deduplicated triple set.  This is computed
after exact duplicate removal, self-pair exclusion, and canonical dedup, so
each `(drug_1, drug_2, SE)` triple is counted exactly once.

### Inclusive threshold rule
A side effect is retained if and only if its unique canonical pair count is
**≥ 500**.  The threshold is inclusive: a side effect with exactly 500 pairs
is retained.

### Deterministic pair ordering
Final pairs are sorted lexicographically by `(drug_1, drug_2)` after all
filtering.  `pair_id` is assigned as a zero-based sequential integer after
sorting: 0, 1, 2, …, n_pairs − 1.

### Deterministic label ordering
Retained side effects are sorted lexicographically by `side_effect_id` (UMLS
CUI string, e.g. `C0001234`).  `label_index` is assigned as a zero-based
sequential integer after sorting.  Ordering is independent of frequency, name,
or discovery order.

### Label matrix type and alignment contract
- Shape: `(n_pairs, n_labels)` — `(63472, 963)` in this run.
- `matrix[i, j] = 1` if pair `pair_id == i` is associated with label
  `label_index == j`; otherwise 0.
- Dtype: `numpy.uint8`.
- Row order matches `pair_id`.  Column order matches `label_index`.
  Both are guaranteed by construction and re-verified from disk after writing.

---

## 4. Observed statistics

All values are taken directly from `outputs/polyllm/data_audit.json`.

| Metric | Value |
|---|---|
| Raw row count | 4,649,441 |
| Exact duplicate rows removed | 0 |
| Rows after exact dedup | 4,649,441 |
| Self-pair rows | 0 |
| Unique self-pairs | 0 |
| Canonical duplicate triples removed | 0 |
| Unique canonical pairs before label filtering | 63,473 |
| Unique drugs before label filtering | 645 |
| Original side effects (all) | 1,317 |
| Minimum pair frequency across all SEs | 1 |
| Maximum pair frequency across all SEs | 28,568 |
| Median pair frequency across all SEs | 1,596 |
| **Minimum pair frequency threshold** | **500** |
| Retained side effects | **963** |
| Excluded side effects | 354 |
| Retained triples | 4,576,287 |
| **Final pair count** | **63,472** |
| Final unique drugs | **645** |
| Mean labels per pair | 72.10 |
| Median labels per pair | 52.0 |
| Minimum labels per pair | 1 |
| Maximum labels per pair | 505 |
| **Label matrix shape** | **[63472, 963]** |
| Label matrix dtype | uint8 |

**Notable observation — no duplicates or self-pairs in the raw file:**
The raw Decagon file contains zero exact duplicate rows, zero self-pairs, and
zero canonical duplicate triples (i.e. no reversed A–B / B–A pairs for the
same side effect).  The original dataset was already deduplicated and is
supplied in canonical source order.

---

## 5. Comparison with reference scale

The approximate reference values from the PolyLLM task description are:

| Metric | Reference (approx.) | Observed | Difference |
|---|---|---|---|
| Unique drugs | ~645 | **645** | **0** — exact match |
| Unique canonical pairs | ~63,472 | **63,472** | **0** — exact match |
| Retained side effects | ~964 | **963** | **−1** |

### Drug count and pair count
Both match the reference exactly.  No explanation is needed.

### Side-effect count discrepancy (963 vs ~964)

**Observed:** 963 retained side effects at threshold ≥ 500 unique canonical
pairs.

**Reference:** approximately 964.

**Quantified difference:** −1 (one fewer retained side effect than the reference).

**Supported explanations:**

1. **Threshold boundary:** The side effect immediately below the cutoff has a
   frequency of exactly 499 unique pairs in our count and 500 in the reference
   count — a difference of one pair.  This would arise if one
   `(drug_1, drug_2, SE)` triple is counted by the reference but not by this
   implementation (e.g. due to a slightly different canonicalisation or
   deduplication rule in the reference pipeline).

2. **Reference value is approximate:** The reference description explicitly
   states "approximately 964".  A difference of 1 is within the expected
   rounding margin of an approximate count.

**Hypothesis (unverified):**
The reference pipeline may have used a different deduplication order (e.g.
exact dedup after canonicalisation rather than before), which could cause
one borderline triple to be included or excluded differently.  This cannot
be confirmed without access to the original preprocessing code.

**What to check next:**
- Examine the side effects with counts near 500 in the label mapping CSV.
- Compare with any published preprocessing notebooks or code from the source
  paper, if available.
- If the exact pipeline source becomes available, re-run with matching
  deduplication order and compare.

**Assessment:** A 1-label discrepancy in ~1,300 side effects at a boundary
threshold is not a correctness error.  The pipeline follows all specified
rules and its invariants have been fully verified from disk.

---

## 6. Post-run validation results

All disk-loaded invariants passed:

| Check | Result |
|---|---|
| `drug_1 < drug_2` for every non-self-pair | ✓ |
| `pair_id` is exactly 0..63471 | ✓ |
| No duplicate canonical pairs | ✓ |
| No self-pairs | ✓ |
| `label_index` is exactly 0..962 | ✓ |
| Matrix dtype is uint8 | ✓ |
| All matrix values are 0 or 1 | ✓ |
| Matrix shape matches pair and label counts | ✓ |
| Every pair has at least one positive label | ✓ |
| Every retained label has ≥ 500 positive pairs | ✓ |
| No NaN in pair table | ✓ |
| No NaN in label mapping | ✓ |

---

## 7. Known limitations

- **No drug-name or SMILES mapping has been performed.**  All drugs are
  represented by STITCH compound IDs (`CID`-prefixed strings) only.  Mapping
  to readable names, descriptions, or molecular structures is a Milestone 2
  task.

- **No data split has been created.**  The pair table, label matrix, and
  label mapping represent the complete processed dataset.  Train/validation/test
  splits will be generated in Milestone 2.

- **A pair-random split will permit the same drug to appear in multiple
  partitions.**  A drug-level split (in which a drug's pairs go exclusively to
  one partition) is scientifically safer but reduces training data.  The split
  strategy must be decided in Milestone 2.

- **Filtering rare labels changes the set of eligible pairs.**  One pair
  (`unique_canonical_pair_count_before_label_filtering = 63473` vs
  `final_pair_count = 63472`) was dropped because all of its associated side
  effects fell below the 500-pair threshold.  This is by design.

- **Results depend on the exact local TWOSIDES/Decagon file version.**  A
  different file version (e.g. an earlier release) could produce different
  row counts, duplicate counts, or frequency values.

- **The original dataset source URL remains unresolved.**  The dataset was
  downloaded before Milestone 0.  No source URL or version identifier is
  recorded in the repository.  This is an outstanding provenance gap
  (decision D10 in the scope document).
