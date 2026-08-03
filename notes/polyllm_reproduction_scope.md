# PolyLLM Reproduction — Project Scope (Milestone 0)

**Date:** 2026-06-27  
**Status:** Corrections applied — pending Milestone 0 approval before Milestone 1 begins

---

## 1. Project goal

Reproduce, and where needed extend, an NLP-based approach to the polypharmacy
side-effect prediction task first framed by the Decagon paper
(Zitnik et al., 2018).  The original Decagon model is a graph neural network.
This project replaces the GNN with a text-based model (hence "PolyLLM") that
represents drug pairs and side effects as natural-language inputs to a
pre-trained biomedical language model.

The dataset is the TWOSIDES-derived Decagon release:
`ChChSe-Decagon_polypharmacy.csv.gz`.

---

## 2. Milestone 0 deliverable

This document — `notes/polyllm_reproduction_scope.md` — is the only output of
Milestone 0.  No code, no preprocessing, no model work is performed at this
stage.

---

## 3. Software environment

### 3.1 Declared dependencies (`requirements.txt`)

The file `requirements.txt` at the repository root contains two entries with
**no version pins**:

```
pandas
scikit-learn
```

No other packages are declared.  The absence of version pins means the file
is not a reproducible dependency specification.  Installing from this file on a
different machine may produce different package versions.

### 3.2 Active virtual environment (`.venv`)

A virtual environment is present at `.venv/`.  Its `pyvenv.cfg` records:

| Property | Value |
|---|---|
| Python executable | `C:\Python313\python.exe` |
| Python version | **3.13.5** |
| Created with | `python -m venv .venv` |

The following packages are installed in the environment (verified from
`.dist-info` directories):

| Package | Installed version |
|---|---|
| pandas | 3.0.3 |
| scikit-learn | 1.9.0 |
| NumPy | 2.4.6 |
| SciPy | 1.18.0 |
| joblib | 1.5.3 |
| threadpoolctl | 3.6.0 |
| narwhals | 2.22.1 |
| python-dateutil | 2.9.0.post0 |
| six | 1.17.0 |
| certifi | 2026.6.17 |
| charset-normalizer | 3.4.7 |
| idna | 3.18 |
| requests | 2.34.2 |
| urllib3 | 2.7.0 |
| tzdata | 2026.2 |
| pip | 26.1.2 |

**Important:** the packages listed above are installed in the current `.venv`
under Python 3.13.5.  This is not equivalent to a reproducible dependency
specification.  They have not been tested under Python 3.11, and their versions
are not recorded in any lockfile.

No NLP or deep-learning packages are installed:
`torch`, `transformers`, `tokenizers`, `huggingface-hub`,
`ChemBERTa`, `rdkit`, `pyarrow`, `networkx`, and `tqdm` are all absent.

### 3.3 Python version and planned environment

**Verified current state:** the repository's `.venv` was created with
**Python 3.13.5** (`C:\Python313\python.exe`).

**Planned reproduction environment:** the project targets **Python 3.11**.
The reason is that later milestones require PyTorch, Hugging Face Transformers,
RDKit, and PyArrow, and compatibility between those packages and Python 3.13 is
less established than for 3.11, particularly for CUDA-enabled PyTorch wheels.
Using 3.11 reduces the risk of unresolvable dependency conflicts in Milestones 2
and 3.

**Required actions before Milestone 1 implementation begins:**

- A separate Python 3.11 virtual environment must be created.
- The existing `.venv` (Python 3.13.5) must **not** be silently modified or
  reused for the 3.11 reproduction environment.  An explicit decision is required
  if the existing environment is to be kept, replaced, or supplemented.
- Minimal preprocessing dependencies (at minimum: `pandas`, `pyarrow`) must be
  installed and pinned in the 3.11 environment before any Milestone 1 code runs.

**Note:** Python 3.13 is not scientifically incompatible with pair-level
preprocessing operations (duplicate removal, canonicalization, label grouping).
However, continuing with Python 3.13 would deviate from the agreed 3.11
environment contract and may produce conflicts when Milestone 2 ML packages
are added.  This is an **operational decision**, not a dataset or scientific
blocker.

**Classification:** Operational decision to resolve before Milestone 1
implementation begins.

### 3.4 Planned later dependencies (not yet installed)

These packages are required for Milestones 2 and 3 and have not been
installed or version-pinned yet:

| Package | Purpose | Status |
|---|---|---|
| PyTorch | Model training | TBD — verify when the Python 3.11 environment is created |
| Hugging Face Transformers | Pre-trained LM access | TBD — verify when the Python 3.11 environment is created |
| RDKit | SMILES handling (if molecular features used) | TBD — verify when the Python 3.11 environment is created |
| PyArrow | Parquet output from preprocessing | TBD — verify when the Python 3.11 environment is created |
| tqdm | Progress reporting | TBD — verify when the Python 3.11 environment is created |

### 3.5 Known dependency risks

- **No lockfile** — neither `requirements.lock`, `poetry.lock`, nor
  `conda-lock.yml` was found.  Reproducing the environment on another machine
  requires a lockfile.
- **`requirements.txt` has no versions** — the two declared packages are unpinned.
- **Python 3.13 vs 3.11** — PyTorch wheel availability for Windows + CUDA under
  3.13 is less comprehensive than under 3.11 as of mid-2026.

### Verified repository state (software environment)

| Item | Verdict |
|---|---|
| `requirements.txt` present with `pandas`, `scikit-learn` (unpinned) | Verified from repository |
| `.venv` present, built with Python 3.13.5 | Verified from repository |
| Installed packages listed in §3.2 | Verified from repository (dist-info directories) |
| `pyproject.toml`, `environment.yml`, `setup.cfg`, `Dockerfile` | Unresolved — none found |
| Python 3.11 as target environment for reproduction | Planned reproduction decision |
| NLP/ML packages (torch, transformers, rdkit, pyarrow, etc.) | Unresolved — none declared or installed |

---

## 4. Dataset

### 4.1 Expected file

The task specifies:

```
data/raw/ChChSe-Decagon_polypharmacy.csv.gz
```

### 4.2 Dataset discovery

The expected file **is present** at the exact expected path.

| Property | Value |
|---|---|
| Path | `data/raw/ChChSe-Decagon_polypharmacy.csv.gz` |
| Compression | gzip (`.csv.gz`) |
| File size | 35,657,514 bytes (~34 MB compressed) |
| Last modified | 2026-06-21 16:01 |

No duplicate or alternative dataset file was found under `data/` or elsewhere
in the repository.

### 4.3 Schema verification

**Two independent sources were checked.  They agree.**

#### 4.3.1 Direct verification from the raw compressed file

The gzip stream was opened with Python's `gzip` module.  Only the first three
lines were read; no full load, grouping, or statistical analysis was performed.

Raw bytes extracted:

```
HEADER : b'# STITCH 1,STITCH 2,Polypharmacy Side Effect,Side Effect Name\n'
ROW 1  : b'CID000002173,CID000003345,C0151714,hypermagnesemia\n'
ROW 2  : b'CID000002173,CID000003345,C0035344,retinopathy of prematurity\n'
```

Decoded:

```
# STITCH 1,STITCH 2,Polypharmacy Side Effect,Side Effect Name
CID000002173,CID000003345,C0151714,hypermagnesemia
CID000002173,CID000003345,C0035344,retinopathy of prematurity
```

The two data rows shown also illustrate the multilabel structure: the same drug
pair (`CID000002173`, `CID000003345`) appears twice with different side effects.

**This is the primary schema verification.**  The schema is now confirmed
directly from the raw source file, not inferred from a derived artefact.

#### 4.3.2 Corroborating evidence from the processed sample

`data/processed/sample.csv` is a 500-row reproducible sample produced by
`src/inspect_dataset.py`.  Its header matches the raw file exactly:

```
# STITCH 1,STITCH 2,Polypharmacy Side Effect,Side Effect Name
CID000003121,CID000003640,C0013604,edema
CID000002250,CID000005002,C0028084,nightmare
```

This file is useful for fast iterative checks but is not the authoritative
schema source.

#### 4.3.3 Confirmed column schema

| Column | Format | Example (from raw file) |
|---|---|---|
| `# STITCH 1` | STITCH compound ID | `CID000002173` |
| `STITCH 2` | STITCH compound ID | `CID000003345` |
| `Polypharmacy Side Effect` | UMLS CUI | `C0151714` |
| `Side Effect Name` | Readable English | `hypermagnesemia` |

**Critical note on column-1 name:** the first column is named `# STITCH 1`,
with a **literal `#` character** as part of the string.  This is confirmed in
the raw bytes: `b'# STITCH 1,...'`.  When loaded with pandas, the column will
be registered as the Python string `'# STITCH 1'`.  Any preprocessing code
that references this column must either use that exact string or explicitly
rename the column at load time.  Failing to handle the `#` will produce a
`KeyError` at runtime.

### 4.4 Dataset interpretation (from prior inspection)

The following facts were established during the prior `src/inspect_dataset.py`
run, documented in `notes/dataset_observations.md`:

- One row = one positive (drug₁, drug₂, side-effect) triple.
- The same drug pair appears in multiple rows with different side effects.
- No negative examples are present — only positive associations.
- Drug identifiers are STITCH compound IDs, not readable names.
- Side-effect identifiers are UMLS CUIs; readable names are also present.
- The task is **multilabel** at the drug-pair level, or equivalently **binary
  classification** over (drug, drug, side-effect) triples.

### Verified repository state (dataset)

| Item | Verdict |
|---|---|
| `data/raw/ChChSe-Decagon_polypharmacy.csv.gz` present at expected path | Verified from repository |
| File size ~34 MB compressed | Verified from repository |
| Columns verified **directly from raw compressed file** (gzip header read, no full load) | Verified from repository |
| Columns corroborated by `data/processed/sample.csv` | Verified from repository |
| Column 1 is `# STITCH 1` with literal `#` — confirmed in raw bytes | Verified from repository |
| Drug IDs are STITCH CIDs (`CID`-prefixed) | Verified from repository |
| Side-effect names are readable English | Verified from repository |
| Dataset is the authoritative TWOSIDES/Decagon release | Unresolved — source URL not recorded in the repo |
| Full row count | Unresolved — not counted during Milestone 0 |
| Number of unique drug pairs | Unresolved — not counted during Milestone 0 |
| Number of unique side effects | Unresolved — not counted during Milestone 0 |

---

## 5. Existing project artefacts

### 5.1 Files present at Milestone 0

| Path | Purpose |
|---|---|
| `requirements.txt` | Minimal dependency declaration (unpinned) |
| `.venv/` | Python 3.13.5 virtual environment |
| `data/raw/ChChSe-Decagon_polypharmacy.csv.gz` | Raw dataset (do not modify) |
| `data/processed/sample.csv` | 500-row reproducible sample |
| `src/inspect_dataset.py` | Dataset inspection script (already run) |
| `notes/dataset_observations.md` | Structured observations from the inspection run |
| `notes/polyllm_reproduction_scope.md` | This document |

### 5.2 What the inspection script confirmed

`src/inspect_dataset.py` was written and run before Milestone 0.  Its findings
are recorded in `notes/dataset_observations.md`.  That document is the
authoritative record of the dataset structure; key conclusions are summarised
in §4.4 above.

---

## 6. Planned reproduction approach (high-level decisions)

These are planned decisions, not yet implemented.

### 6.1 Task formulation

**Planned:** binary classification over (drug₁, drug₂, side-effect) triples.

Input to the model: a text prompt such as:
```
Drug 1: <name1>  Drug 2: <name2>  Side effect: <side_effect_name>
```
Label: 1 = observed polypharmacy association, 0 = unobserved (sampled negative).

The multilabel formulation (predict all side effects for a pair) is deferred
as a potential extension.

### 6.2 Drug name mapping (Milestone 2 dependency)

STITCH compound IDs must be converted to drug names or SMILES before a
language model or molecular encoder can use them.  STITCH IDs are based on
PubChem CIDs.  Candidate sources include PubChem and DrugBank.

**This mapping does not block Milestone 1.**  Milestone 1 operates entirely
on the STITCH identifiers as they appear in the dataset and does not require
readable names.  Milestone 1 work includes: duplicate removal, unordered-pair
canonicalization, self-pair reporting, side-effect frequency counting, filtering,
label mapping, pair-level grouping, and multi-hot label creation — none of which
require drug names.

The STITCH-to-PubChem/name/SMILES mapping decision blocks Milestone 2 and all
molecular-language-model feature generation, but it does not block Milestone 1.

**Status: unresolved** — to be decided before Milestone 2 begins.

### 6.3 Text representation (Milestone 2 planning)

| Feature | Source | Status |
|---|---|---|
| Drug 1 name | STITCH → PubChem/DrugBank mapping | Unresolved (Milestone 2) |
| Drug 2 name | STITCH → PubChem/DrugBank mapping | Unresolved (Milestone 2) |
| Side-effect name | `Side Effect Name` column directly | Ready |
| Drug SMILES / molecular features | Optional — PubChem or ChemBERTa encoder | Deferred |

### 6.4 Negative sampling

Unobserved (drug, drug, side-effect) triples are not confirmed safe — they are
simply not reported in pharmacovigilance data.  Negatives will be sampled from
unobserved triples and must be documented as *unobserved*, not as *safe*.

### 6.5 Train / test split

Row-level random splitting risks data leakage because the same drug pair can
appear in both splits with different side effects.  Planned strategy:
**drug-pair-level split** (all triples for a given pair go entirely to train
or test, not both).

### 6.6 Model

A pre-trained biomedical language model (e.g., PubMedBERT) or a general-purpose
LLM will encode the text prompt and output a binary classification score.
Model selection is deferred to Milestone 2.

---

## 7. Open decisions

### 7.1 Must be resolved before Milestone 1 implementation begins

| # | Decision | Notes |
|---|---|---|
| D1 | Create and use the planned Python 3.11 environment | Existing `.venv` is 3.13.5; must not be silently reused |
| D2 | Exact minimal dependency versions for preprocessing | At minimum: `pandas`, `pyarrow`; versions must be pinned |
| D3 | Whether PyArrow must be installed immediately for Parquet output | Parquet is the planned output format for processed data |

### 7.2 Must be resolved before Milestone 2

| # | Decision | Notes |
|---|---|---|
| D4 | STITCH identifier format and conversion rules | Leading zeros, `CID` prefix handling |
| D5 | Authoritative STITCH-to-PubChem mapping source | PubChem REST API or local dump |
| D6 | Preferred-name retrieval source | IUPAC name, synonyms, or common name |
| D7 | Standardized versus canonical SMILES | Affects ChemBERTa tokenization |
| D8 | Salt, stereoisomer and duplicate-structure handling | Affects unique drug count and embeddings |
| D9 | Fallback and exclusion policy for unmapped IDs | What to do when a STITCH ID has no PubChem entry |
| D10 | Confirm authoritative dataset source URL | Required for reproducibility documentation |

### 7.3 Must be resolved before model implementation

| # | Decision | Notes |
|---|---|---|
| D11 | Exact ChemBERTa pooling alignment with the paper | CLS token vs mean pooling |
| D12 | Loss function | Binary cross-entropy vs focal loss |
| D13 | Focal-loss parameters (if used) | γ and α values |
| D14 | Learning rate | |
| D15 | Batch size | |
| D16 | Early-stopping patience | |
| D17 | F1 threshold selection | Fixed 0.5 vs tuned on validation |
| D18 | Exact AP@50 definition | Whether @50 refers to side effects or triples |
| D19 | Baseline classifier | Logistic regression on TF-IDF, random, or frequency-prior |

---

## 8. Milestone plan (overview)

| Milestone | Goal | Prerequisites |
|---|---|---|
| 0 (this) | Scope document, repository inspection | — |
| 1 | Preprocessing: canonicalization, filtering, label grouping, split, Parquet output | D1, D2, D3 resolved |
| 2 | Drug mapping, text/molecular feature generation, model training, baseline evaluation | Milestone 1 complete; D4–D10 resolved |
| 3 | Ablations, analysis, final report | Milestone 2 complete; D11–D19 resolved |

---

## 9. Scientific limitations

- Missing associations are **unobserved**, not confirmed absent or safe.
- The TWOSIDES dataset is derived from pharmacovigilance spontaneous reports,
  which contain noise and reporting bias.
- Negatives sampled from unobserved triples do not imply those combinations
  are medically safe.
- Any external biomedical text used as drug context must not incorporate
  information that could leak test-set target labels.
- STITCH IDs are not interpretable by language models without an explicit
  mapping step; this step is a Milestone 2 concern, not Milestone 1.

---

## 10. Milestone 0 approval status

Milestone 0 may be marked complete when all of the following are confirmed:

| Criterion | Status |
|---|---|
| Raw input header directly verified (not only via derived sample) | **Met** — gzip stream read directly; header confirmed in raw bytes |
| Reduced reproduction scope is explicit | **Met** — §6.2 clarifies mapping does not block Milestone 1 |
| Repository facts separated from planned choices | **Met** — all "Verified repository state" tables use three-way labels |
| STITCH mapping correctly identified as a Milestone 2 dependency | **Met** — §6.2 and §7.2 |
| Python environment decision identified as unresolved operational decision | **Met** — §3.3, classified as operational, assigned to D1 |
| No Milestone 1 code or outputs created | **Met** — no preprocessing code, no model code, no additional files created |

**Milestone 0 is ready for approval.**
