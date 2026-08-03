# Dataset Observations — ChChSe-Decagon_polypharmacy

## Status

- [x] Inspection script run (`src/inspect_dataset.py`)
- [x] Open questions answered
- [ ] Task formulation confirmed
- [ ] Drug ID mapping strategy decided

---

## Basic structure (confirmed)

| Property | Value |
|---|---|
| One row represents | One positive (drug₁, drug₂, side-effect) association |
| Drug identifier type | STITCH compound IDs — not readable names |
| Side-effect identifier type | UMLS concept IDs **and** readable names (both present) |
| Same drug pair in multiple rows | Yes — one row per side effect |
| Task level | Multilabel at the drug-pair level |
| Negative examples | Not present in the file |

---

## Confirmed answers to open questions

1. **Drug identifiers** — STITCH compound IDs, not names. A mapping step is required before any text model.
2. **Side-effect identifiers** — UMLS CUIs with readable names also present. Readable names can be used directly as text.
3. **Row structure** — one row = one positive (drug, drug, side-effect) triple.
4. **Negative examples** — not included. Negatives must be constructed (sampled from unobserved pairs).
5. **Drug-pair repetition** — yes, the same pair appears many times with different side effects.
6. **Task type** — multilabel at the drug-pair level. Also expressible as binary classification over (drug, drug, side-effect) triples.
7. **Text features** — side-effect names are usable directly. Drug names are not available without a STITCH → name mapping.
8. **Unique side effects** — to be read from the script output.

---

## Open decisions

### 1. Drug ID mapping
STITCH IDs must be mapped to something a language model can use — drug names, descriptions,
mechanisms of action, or known targets. Possible sources:
- PubChem (STITCH IDs are based on PubChem CIDs)
- DrugBank (names, descriptions, targets)
- A side file that may accompany the Decagon release

This mapping decision gates the entire NLP approach.

### 2. Task formulation
Two options are compatible with this dataset:

| Formulation | Input | Output |
|---|---|---|
| **Binary per triple** | (drug₁, drug₂, side-effect candidate) | 1 = observed, 0 = unobserved |
| **Multilabel per pair** | (drug₁, drug₂) | set of associated side effects |

Binary per triple is simpler to frame as an NLP task and maps naturally to the
text template: `Drug 1: X [SEP] Drug 2: Y [SEP] Side effect: Z`.
Multilabel per pair is the natural scientific task but harder for sequence models.

### 3. Negative sampling
Unobserved (drug, drug, side-effect) triples are not confirmed safe — they are simply
not reported. Any negative set must be described as *unobserved* in all documentation.

### 4. Train/test split strategy
Random row-level splitting risks leakage: the same drug pair could appear in both
train and test with different side effects. A drug-pair-level or drug-level split
is safer but reduces the available training data.

---

## Implications for text features

| Feature type | Feasibility |
|---|---|
| Drug names | Not directly available — need STITCH → name mapping |
| Drug descriptions / mechanisms | Not in dataset — need external source (e.g. DrugBank) |
| Side-effect names | Available — readable UMLS names present in the file |
| TF-IDF over drug text | Blocked until drug mapping is resolved |
| Biomedical LM (BioBERT) on side-effect names | Feasible once drug names are available |

---

## Scientific limitations (dataset-specific)

- Missing associations are not medically impossible — they are unobserved or unreported.
- The dataset is derived from pharmacovigilance reports (TWOSIDES), which contain noise and reporting bias.
- Constructing negative samples from unobserved triples does not imply those combinations are safe.
- Random row-level splitting risks leakage if the same drug pair appears in train and test.
- Any external biomedical information added as context must not leak test-set target labels.
- STITCH IDs are not interpretable by language models without a mapping to names or descriptions.
