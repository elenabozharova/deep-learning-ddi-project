# Experiment 1 — Wikidata Coverage & Feasibility Audit

**Date:** 2026-08-16
**Status:** Complete. Live audit run against all 963 side effects; real Wikidata data throughout this note (no mocked numbers).

**Scope note (2026-08-16, later same day):** Experiment 1 was subsequently narrowed to a pure encoder comparison (SE0/SE1/SE2, same side-effect name text for all three) — see [`experiment_1_encoder_comparison.md`](experiment_1_encoder_comparison.md), the primary result. This audit's coverage gap (21.5% not-found/ambiguous) was one input to that decision; the pipeline is preserved as exploratory groundwork and possible future work, not a dependency of the completed experiment. This is a scope decision, not a failed experiment.

---

## Objective

Determine whether open, unauthenticated Wikidata metadata can support Experiment 1 (RQ1: does richer biomedical text improve side-effect node representations?) while UMLS UTS API-key access remains blocked on account eligibility — either as a stand-in text source or as a bridge from our existing UMLS CUIs to other open biomedical ontologies.

## Data

963 retained PolyLLM side effects, each already carrying a valid, unique UMLS CUI in `data/processed/polyllm_label_mapping.csv` (`side_effect_id` column) — confirmed unchanged from the Experiment 1 UMLS-pipeline audit (see [`experiment_1_umls_enrichment.md`](experiment_1_umls_enrichment.md)).

## Repository audit (Phase 1)

- Re-confirmed `polyllm_label_mapping.csv` structure: `label_index, side_effect_id, side_effect_name, unique_pair_count`; still 963 rows, `label_index` exactly `0..962`, CUIs still unique/valid.
- Reused rather than reimplemented: `validate_mapping_df` (`polyllm.features.generate_side_effect_embeddings`) and `validate_label_alignment` + `CUI_PATTERN` (`polyllm.data.enrich_side_effects_umls`) are imported directly by the new module. The existing UMLS pipeline file was not modified.
- New module: [`src/polyllm/data/enrich_side_effects_wikidata.py`](../src/polyllm/data/enrich_side_effects_wikidata.py). New artifacts only — `polyllm_label_mapping.csv` and all existing reproduction outputs are untouched (verified by test `test_original_mapping_file_untouched`).

## Mapping method

```
UMLS CUI  →  Wikidata property P2892  →  Wikidata QID(s)  →  entity metadata
```

Two public, unauthenticated endpoints (no API key required):
- **SPARQL** `https://query.wikidata.org/sparql` — batched `VALUES` query (50 CUIs/request) to find every QID with `wdt:P2892` equal to one of our CUIs. This directly surfaces zero/one/many matches per CUI without guessing.
- **Action API** `https://www.wikidata.org/w/api.php` (`wbgetentities`) — batched (50 QIDs/request) fetch of English label, aliases, description, and claims for every **uniquely** matched QID only. Ambiguous CUIs are never resolved by picking one candidate.

Cross-reference properties used, verified live against Wikidata's own property-search API (not assumed):

| Field | Property |
|---|---|
| MeSH descriptor ID | P486 |
| MONDO ID | P5270 |
| SNOMED CT ID | P5806 |
| Disease Ontology ID | P699 |
| Human Phenotype Ontology ID | P3841 |
| ICD-10 / ICD-10-CM / ICD-9 / ICD-9-CM | P494 / P4229 / P493 / P1692 |
| instance of / subclass of (structural, one hop only) | P31 / P279 |

Total live HTTP requests for the full 963-CUI run: ~40 (match phase batches + entity-fetch batches + one bounded structural-label-resolution pass), each with a descriptive `User-Agent` and ~1s inter-batch pacing — well within normal public-endpoint etiquette.

## Coverage results (real, full 963-CUI run)

| Outcome | Count | % of 963 |
|---|---|---|
| Mapped, unique QID | **756** | **78.5%** |
| Not found | 163 | 16.93% |
| Multiple matches (ambiguous) | 44 | 4.57% |
| Request failed | 0 | 0.0% |

## Metadata coverage (among the 756 uniquely mapped concepts)

| Field | Count | % |
|---|---|---|
| English label | 756 | 100.0% |
| ≥1 English alias | 638 | 84.39% |
| English description | 747 | 98.81% |
| MeSH ID | 662 | 87.57% |
| MONDO ID | 409 | 54.1% |
| SNOMED CT ID | 112 | 14.81% |
| ICD ID (any variant) | 656 | 86.77% |
| Disease Ontology ID | 459 | 60.71% |
| HPO ID | 402 | 53.17% |
| Structural info (instance/subclass of) | 755 | 99.87% |

Alias counts (mapped concepts): mean 3.90, median 3.0, p75 5, p90 9, p95 12, max 37.

### Text richness for Experiment 1 (out of all 963 — not just the mapped subset)

| Level | N | % |
|---|---|---|
| A. name only | 5 | 0.52% |
| B. name + aliases (no description) | 4 | 0.42% |
| C. name + description (no aliases) | 113 | 11.73% |
| **D. name + aliases + description** | **634** | **65.84%** |

A Wikidata description is a short community-written gloss (e.g. *"physical symptom"*), **not** a formal biomedical definition — kept terminologically distinct from UMLS `MDEF`-style definitions throughout this audit and its code (`wikidata_description`, never `umls_definitions`).

## Examples

**Correctly mapped (label_index, CUI, QID, label, aliases, description, cross-refs):**

```
[0] abdominal distension | C0000731 | Q2536390
  label: "abdominal distention"
  aliases: ["abdominal distension", "swollen abdomen"]
  description: "physical symptom"
  HPO: HP:0003270   (no MeSH/MONDO/SNOMED/DOID in this case)

[1] abdominal pain | C0000737 | Q183425
  label: "abdominal pain"
  aliases: ["stomach ache", "stomach pain", "tummy ache"]
  description: "pain in the abdomen"
  MeSH: D015746   SNOMED: 271681002   HPO: HP:0002027

[3] abortion spontaneous | C0000786 | Q28693
  label: "miscarriage"
  aliases: ["abortion", "spontaneous abortion"]
  description: "natural death of an embryo or fetus before it is able to survive independently"
  MeSH: D000022   DOID: DOID:722   HPO: HP:0005268

[4] abscess | C0000833 | Q164655
  label: "abscess"
  aliases: ["boil", "ulcer"]
  description: "localized collection of pus that has built up within the tissue of the body"
  MeSH: D000038   MONDO: MONDO_0005227
```

Note `[3]`: the UMLS/TWOSIDES name `abortion spontaneous` maps to the Wikidata preferred label `miscarriage` — a materially different surface form for the same concept (worth flagging if names are ever compared textually downstream).

**Not found (representative):**

```
[10] adjustment disorder | C0001546 — no Wikidata entity carries this CUI via P2892.
```

**Ambiguous (representative — full list of 44 in `outputs/polyllm/wikidata_coverage_report.md`):**

```
[2] birth defect | C0000768 → Q2852249 "congenital abnormality" (alias: "birth defect")
                              Q3281904 "congenital physical abnormality" (a narrower subclass of Q2852249)
```
Genuinely overlapping Wikidata items (one is a subclass of the other) rather than a data-quality bug — resolving this automatically would require either a preference rule (e.g. prefer the broader/first-created item) or manual curation, deliberately not done here per the "no silent choice" constraint.

## Interpretation

**Is Wikidata alone sufficient for SE3/SE4?**
Not fully. 65.84% of the 963 side effects get the full `name + aliases + description` bundle SE3/SE4 would want, but 21.5% (163 not-found + 44 ambiguous) currently get *nothing* beyond the name without extra work (manual disambiguation rules for the 44, and no recourse at all for the 163 not-found besides falling back to name-only or another source). A controlled ablation experiment wants uniform treatment across all 963 labels; a ~1-in-5 gap is large enough that Wikidata alone is not a clean substitute for UMLS-sourced text.

**Is Wikidata useful as a bridge from UMLS CUI to open biomedical ontologies?**
Yes, clearly, for the 78.5% it uniquely maps: 87.6% of those also carry a MeSH ID and 86.8% an ICD ID, so Wikidata reliably re-exposes CUIs alongside other identifiers we could use to enrich from MeSH/ICD/MONDO/DOID/HPO sources directly, without needing UMLS UTS access at all. MONDO (54%) and HPO (53%) coverage is moderate; SNOMED CT (14.8%) is low enough that Wikidata should not be relied on as a SNOMED bridge.

### Advantages observed
- No UMLS license/API-key dependency — fully open, ran end-to-end today with zero setup.
- `P2892` gave clean, unambiguous zero/one/many detection — no fuzzy matching needed.
- Genuinely good structured cross-referencing (MeSH/ICD especially) for the mapped subset.
- Deterministic, cheap (~40 HTTP requests total for all 963 CUIs), fast (~4 minutes end-to-end), resumable cache.

### Limitations observed
- 16.93% of CUIs have no Wikidata entity at all (community coverage gap, not a pipeline bug — confirmed via the SPARQL match phase, not an extraction failure).
- 4.57% are genuinely ambiguous (overlapping/nested concepts), each requiring a judgment call this audit deliberately does not make automatically.
- Descriptions are short glosses, not formal biomedical definitions — real semantic content is thinner than a UMLS `MDEF`.
- SNOMED CT bridging coverage (14.8%) is weak.
- One case observed (`abortion spontaneous` → `miscarriage`) where the Wikidata preferred label diverges materially from the TWOSIDES name — worth checking for more such cases before using `wikidata_label` as a name replacement anywhere.

## Recommendation

**B. Wikidata is useful only as a mapping bridge** (and as a *partial* supplementary text source for the ~66% it fully covers) — **not** as a complete substitute for UMLS.

Concretely: for the 756 uniquely mapped concepts, Wikidata's MeSH/ICD/MONDO/DOID/HPO cross-references are strong enough to justify a follow-on open-ontology enrichment step (Phase 11's bridge scenario) once UMLS access is available or if it stays blocked. But for building the actual SE3 (`name + synonyms`) / SE4 (`name + synonyms + definition`) text inputs across *all* 963 side effects uniformly, the 21.5% not-found/ambiguous gap and the description-vs-definition quality difference mean UMLS remains the preferred primary source; Wikidata is best used to backfill or cross-check, not replace it.

## Next planned experiments (unchanged, not implemented here)

```
SE0 = BERT(name)
SE1 = PubMedBERT(name)
SE2 = SapBERT(name)
SE3 = SapBERT(name + synonyms)
SE4 = PubMedBERT(name + synonyms + definition)
```

No embeddings were generated and the GNN was not touched in this milestone.
