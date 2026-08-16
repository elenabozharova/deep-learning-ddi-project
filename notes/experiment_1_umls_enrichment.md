# Experiment 1 — UMLS Metadata Enrichment for Side-Effect Concepts

**Date:** 2026-08-16
**Status:** Pipeline built, mocked-tested, and smoke-tested end-to-end on all 963 real side effects. **Live UMLS retrieval not yet run — `UMLS_API_KEY` is not configured in this environment.**

**Scope note (2026-08-16, later same day):** Experiment 1 was subsequently narrowed to a pure encoder comparison (SE0/SE1/SE2, same side-effect name text for all three) to keep the controlled variable to just the pretrained model — see [`experiment_1_encoder_comparison.md`](experiment_1_encoder_comparison.md), the primary result. This UMLS pipeline is preserved as exploratory groundwork and possible future work (SE3/SE4), not a dependency of the completed experiment. This is a scope decision, not a failed experiment.

---

## Objective

PolyLLM represents every side-effect node with a single frozen `bert-base-uncased` embedding of its *name only*:

```
side-effect name → bert-base-uncased → 768-d embedding → GNN
```

**RQ1:** Does using domain-specific biomedical language models and richer textual context improve the representation of polypharmacy side effects compared with generic BERT embeddings based only on side-effect names?

Before generating any new embeddings (SE1–SE4 in the planned ablation), we need richer text to feed those encoders — preferred names, synonyms, and definitions. TWOSIDES/Decagon already ships a UMLS CUI for every retained side effect, so the first milestone is a **UMLS enrichment pipeline** that turns those CUIs into canonical metadata, with no change to embeddings, drug features, splits, or the GNN.

## Baseline

```
side-effect name → bert-base-uncased → embedding
```
(`src/polyllm/features/generate_side_effect_embeddings.py`, unchanged.)

## Proposed extension

```
CUI (already in polyllm_label_mapping.csv)
    → UMLS REST API (concept, atoms, definitions)
    → preferred name / synonyms / definition
    → [later] biomedical LM (PubMedBERT / SapBERT)
    → embedding
```

## Important methodological distinction

UMLS is the **biomedical knowledge/terminology source** used here only to enrich *text*. PubMedBERT and SapBERT are **representation models** that will consume this text in a later, separate experiment (SE1–SE4). No embeddings are generated in this milestone.

---

## Repository audit (Phase 1)

- **Source of the 963 retained side effects:** [`data/processed/polyllm_label_mapping.csv`](../data/processed/polyllm_label_mapping.csv) — columns `label_index, side_effect_id, side_effect_name, unique_pair_count`. `side_effect_id` **is** the UMLS CUI (format `C\d{7}`), inherited directly from the Decagon `Polypharmacy Side Effect` column in [`prepare_multilabel_dataset.py`](../src/polyllm/data/prepare_multilabel_dataset.py).
- Verified: 963/963 rows have valid, unique, well-formed CUIs. 0 missing, 0 malformed, 0 duplicates, 0 duplicate names.
- `label_index` is exactly `0..962` in file order; this order is the contract consumed by `generate_side_effect_embeddings.py` (`validate_mapping_df` asserts it) and therefore by `bert_side_effect_embeddings.npy` / `bert_side_effect_index.csv`, and transitively by the GNN's `seffect` node ordering.

## UMLS API access strategy (Phase 2)

Confirmed against the official NLM documentation (`documentation.uts.nlm.nih.gov`), not a third-party source:

- Base URL: `https://uts-ws.nlm.nih.gov/rest`
- Auth: a single `apiKey` query parameter per request (no ticket-granting flow needed under the current auth scheme).
- Endpoints used:
  - `GET /content/{version}/CUI/{cui}` — preferred name, semantic types
  - `GET /content/{version}/CUI/{cui}/atoms?language=ENG&pageSize=200` — English atoms/synonyms
  - `GET /content/{version}/CUI/{cui}/definitions` — source-attributed definitions

**`UMLS_API_KEY` is not set in this environment.** The pipeline reads it via `os.environ["UMLS_API_KEY"]` ([`get_api_key()`](../src/polyllm/data/enrich_side_effects_umls.py)), never hard-codes it, never logs it, and raises a clear `RuntimeError` with setup instructions when missing:

1. Create a free UTS account: https://uts.nlm.nih.gov/uts/signup-login
2. Copy the API key from the UTS profile page.
3. Set it locally, e.g. (PowerShell): `$env:UMLS_API_KEY = "your-key"`, or add `UMLS_API_KEY=your-key` to a local `.env` (already gitignored).

Because no key was available, the pipeline was validated two ways instead of a live run:
1. **Unit tests** (`tests/test_enrich_side_effects_umls.py`, 30 tests) mock every `requests.get` call — valid/missing/unavailable CUI, retries, synonym/definition cleaning, caching, determinism, ordering, and a check that the API key never appears in logs, cache, or output.
2. **Full-scale mocked smoke test**: `run_enrichment()` was run against the real 963-row `polyllm_label_mapping.csv` with `requests.get` mocked to return synthetic (obviously fake, e.g. `"Preferred[C0000731]"`) concept/atom/definition payloads. This is **not** real biomedical content — it only proves the pipeline runs to completion on the true CUI/name/label_index set, that all 963 rows succeed, and that label ordering survives (row 0 → `C0000731` / `abdominal distension`, matching `label_index` 0 exactly). Output was written to a scratch directory outside the repo, not committed.

## Coverage (Phase 9)

**Not available yet** — requires a live `UMLS_API_KEY`. Once the user supplies one, running:

```
python src/polyllm/data/enrich_side_effects_umls.py
```

will populate:
- `outputs/polyllm/umls_enrichment_audit.json` — machine-readable coverage
- `outputs/polyllm/umls_coverage_report.md` — human-readable coverage (counts, percentages, mean/median/max synonym counts, content-level breakdown, failure lists, preferred-name divergences)
- `data/processed/umls_side_effect_metadata.parquet` / `.csv` — the canonical enriched dataset

This note should be updated with the real numbers after that run, before proceeding to SE1–SE4 embedding generation.

## Missing-data policy

- A CUI with no UMLS record → `retrieval_status = "not_found"`, all UMLS fields `null`/empty, `has_umls_concept = False`. Not treated as a pipeline error — recorded per-row and counted in coverage.
- A CUI whose fetch failed after retries → `retrieval_status = "request_failed"`; such rows are automatically retried on the next run (cache is not considered final until success).
- No definitions are ever invented; `umls_definitions = "[]"` and `has_definition = False` explicitly mark absence.
- Synonym/definition counts are never capped in this pipeline; any token-budget cap for PubMedBERT/SapBERT input happens later, during text construction for SE3/SE4 — not here.

## Next planned experiments (not implemented yet)

```
SE0 = BERT(name)                              — PolyLLM baseline (existing)
SE1 = PubMedBERT(name)
SE2 = SapBERT(name)
SE3 = SapBERT(name + synonyms)
SE4 = PubMedBERT(name + synonyms + definition)
```

These require this milestone's canonical metadata artifact as input and are explicitly out of scope until this enrichment step is reviewed and (once a key is available) actually run against live UMLS data.

---

## One concrete example (illustrative — from the mocked smoke test, not live UMLS data)

```
Original label:
abdominal distension  (label_index 0)

CUI:
C0000731

[SYNTHETIC — from mocked smoke test, not real UMLS content]
UMLS preferred name:
Preferred[C0000731]

Synonyms:
["synonym A"]

Definition:
[{"value": "A mock UMLS definition for smoke-testing.", "source": "MSH"}]

Semantic types:
["Sign or Symptom"]
```

Once a live `UMLS_API_KEY` is supplied, rerunning the pipeline will replace this with the actual UMLS record for `C0000731` (real preferred name, e.g. "Abdominal Distension", real MSH/SNOMEDCT synonyms and definitions).

## Confirmation: label ordering unchanged

`validate_label_alignment()` runs automatically at the end of every pipeline execution and raises `ValueError` if `metadata_df` row *i* does not match `mapping_df` row *i* on `label_index`, `side_effect_id`, and `side_effect_name`. The full 963-row mocked run completed this check without error; `tests/test_enrich_side_effects_umls.py::TestLabelAlignment` covers both the passing case and a deliberately shuffled case that must raise.
