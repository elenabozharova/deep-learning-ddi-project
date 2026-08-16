# Experiment 1 — Side-Effect Encoder Comparison (SE0 vs SE1 vs SE2)

**Date:** 2026-08-16
**Status:** Complete. Real GNN training/evaluation runs, not simulated.

This is the primary Experiment 1 result. UMLS/Wikidata enrichment work is preserved separately as exploratory groundwork, superseded for this experiment by the narrower, cleaner design below — see [`experiment_1_umls_enrichment.md`](experiment_1_umls_enrichment.md) and [`experiment_1_wikidata_coverage.md`](experiment_1_wikidata_coverage.md) for their (unchanged) status.

---

## Research Question

> Do domain-specific biomedical language models provide more effective side-effect representations for polypharmacy side-effect prediction than generic BERT when the same side-effect names are used as input?

## Hypothesis

Biomedical-domain and biomedical-entity-specialized pretrained encoders may capture adverse-event terminology more effectively than general-domain BERT.

## Controlled variable

The side-effect encoder only (`seffect.x`, i.e. which pretrained language model produced the 963×768 side-effect node feature matrix).

## Fixed variables (verified, not assumed)

- The 963 retained side-effect labels, names, and `label_index` ordering — unchanged (`data/processed/polyllm_label_mapping.csv` untouched).
- Drug-pair features (`pdrugs.x`) — byte-identical `torch.equal` across all three graphs.
- Graph edges/split — `edge_index` byte-identical across all three graphs; `RandomLinkSplit` produced identical train/val/test edge counts (1,098,309 / 457,628 / 457,628) because the split is seeded on edges, not features.
- GNN architecture, optimizer, learning rate, loss, epoch budget, early-stopping patience, negative sampling, evaluation code — `training_config.json` is **byte-for-byte identical** across SE0/SE1/SE2 (verified programmatically), including `total_parameters: 4,256,064` for all three (no architecture change, since PubMedBERT and SapBERT both happen to be 768-dim like BERT-base).
- No GNN hyperparameter tuning was performed per encoder.

## Audit of the existing SE0 pipeline (Step 1)

Traced from `src/polyllm/features/generate_side_effect_embeddings.py` and `outputs/polyllm/side_effect_embedding_audit.json`:

| Property | Value |
|---|---|
| Model | `bert-base-uncased`, resolved revision `86b5e0934494bd15c9632b12f734a8a67f723594` |
| Tokenizer | `BertTokenizer`, `max_length=64`, dynamic padding, truncation on |
| Pooling | Mean of non-padding token embeddings over the final hidden layer, attention-mask-aware, **special tokens (CLS/SEP) included** — not `[CLS]`, not `pooler_output` |
| Batch size | 32 |
| Model state | `model.eval()`, `requires_grad_(False)` — fully frozen |
| Embedding dim | 768 (confirmed live via smoke test, not assumed) |
| Ordering | Loaded from `polyllm_label_mapping.csv`, sorted/validated `label_index == 0..962`, embedded in that exact order |
| Determinism | Smoke test runs inference twice and asserts `torch.allclose` |
| Existing artifact integrity | `sha256(bert_side_effect_embeddings.npy)` recomputed today and matches the value recorded in `side_effect_embedding_audit.json` exactly — **no drift, SE0 was reused as-is, not regenerated** |
| Downstream, frozen SE0 GNN result reused as-is | `outputs/polyllm/gnn/test_metrics.json` — not retrained for this experiment |

**Methodological note on pooling (reported before generating anything, per plan):** SapBERT's own model card (`cambridgeltl/SapBERT-from-PubMedBERT-fulltext`, verified live) recommends extracting the **`[CLS]` token** of the final layer as the entity embedding, explicitly *not* mean pooling. Since this experiment's only intended variable is the encoder, mean pooling (identical to SE0) was deliberately kept for SE1 *and* SE2 rather than switching to CLS pooling for SE2 — using CLS pooling only for SapBERT would have confounded "encoder" with "pooling strategy," undermining the controlled design the whole experiment is built around. A CLS-pooled SapBERT variant is listed under Future Work, not run here.

## Model selection (Step: Model Selection)

| ID | Model | Source paper | Hidden dim (confirmed live) | Tokenizer |
|---|---|---|---|---|
| SE0 | `bert-base-uncased` | Devlin et al. 2019 | 768 | `BertTokenizer` |
| SE1 | `microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext` rev `e1354b7a3a09615f6aba48dfad4b7a613eef7062` | Gu et al. 2021, *Domain-Specific Language Model Pretraining for Biomedical NLP* | 768 | `BertTokenizer` (WordPiece, PubMed-vocabulary) |
| SE2 | `cambridgeltl/SapBERT-from-PubMedBERT-fulltext` rev `090663c3ae57bf35ffe4d0d468a2a88d03051a4d` | Liu et al. 2021 (NAACL), *Self-Alignment Pretraining for Biomedical Entity Representations* | 768 | inherited from SE1 |

Both are the canonical checkpoints most commonly cited for these two papers (PubMedBERT: the abstract+fulltext variant, the one usually meant by "PubMedBERT" in the literature; SapBERT: the from-PubMedBERT-fulltext variant, the original NAACL release). Confirmed live against the HuggingFace Hub `config.json` (`hidden_size: 768`, `model_type: bert`) and model cards before downloading — not assumed.

## Embedding generation (Steps 2–4)

New module: [`src/polyllm/features/generate_biomedical_side_effect_embeddings.py`](../src/polyllm/features/generate_biomedical_side_effect_embeddings.py). Reuses SE0's actual tokenization/pooling/validation functions by import (`embed_side_effects`, `mean_pool`, `smoke_test`, `tokenization_audit`, `validate_mapping_df`, `build_side_effect_index`, `validate_outputs_from_disk`) rather than reimplementing them — only the encoder and output paths differ.

| Check | SE1 (PubMedBERT) | SE2 (SapBERT) |
|---|---|---|
| Shape | (963, 768) | (963, 768) |
| `label_index` / `side_effect_id` order matches mapping | ✓ | ✓ |
| NaN/Inf | none | none |
| All 963 names processed | ✓ | ✓ |
| Zero-vector rows | 0 | 0 |
| Duplicate row groups | 0 | 0 |
| Embedding norm (mean ± std) | 14.20 ± 0.17 | 15.08 ± 0.18 |
| Pairwise cosine similarity (mean ± std) | **0.928 ± 0.016** | **0.296 ± 0.088** |

The pairwise-similarity gap is a sanity-check observation, not over-interpreted: mean-pooled PubMedBERT vectors sit in a much more anisotropic region of space (near-identical directions) than mean-pooled SapBERT vectors, which is broadly consistent with anisotropy being a known property of mean-pooled un-adapted BERT-family models, and with SapBERT's contrastive training pushing entities apart even under a pooling strategy it wasn't trained for. No causal claim is made from this alone.

Artifacts: `data/features/{pubmedbert,sapbert}_side_effect_embeddings.npy` + matching `_index.csv`; audits at `outputs/polyllm/{pubmedbert,sapbert}_side_effect_embedding_audit.json`.

## Controlled GNN runs (Step 6)

`src/polyllm/train_gnn.py` and `evaluate_gnn.py` were given two new optional CLI flags (`--graph-path`, `--output-dir` / `--metrics-output`), defaulting to their previous hardcoded values — SE0's default invocation is byte-for-byte unchanged. This was the minimum change needed to run SE1/SE2 without overwriting SE0's frozen checkpoint/metrics.

Graphs built with the existing, unmodified `build_graph.py` CLI (`--seffect-features`, `--graph-output`):
- `data/graph/gnn_link_split_pubmedbert.pt`
- `data/graph/gnn_link_split_sapbert.pt`

**Programmatic proof that only `seffect.x` differs** (not just file naming — the audit JSON's `feature_source` field turned out to be a pre-existing cosmetic bug that always prints the SE0 default path regardless of which file was actually used; see Unresolved Issues):

```
pdrugs.x identical across all 3 graphs:      True   (torch.equal)
edge_index identical across all 3 graphs:    True   (torch.equal)
training_config.json SE0 == SE1:             True   (dict equality, incl. total_parameters=4,256,064)
training_config.json SE0 == SE2:             True
seffect.x matches its own source .npy file:  True for SE0, SE1, SE2 individually
seffect.x differs pairwise across SE0/SE1/SE2: True (all three pairs distinct)
```

Training (10 max epochs, patience 2, identical Adam/lr=0.01/BCEWithLogitsLoss for all three):

| Run | Epochs run | Best epoch | Best val_loss | Wall time |
|---|---|---|---|---|
| SE1 (PubMedBERT) | 7 (early-stopped) | 5 | 0.6309 | ~3.3 min |
| SE2 (SapBERT) | 7 (early-stopped) | 5 | 0.7092 | ~3.2 min |

(SE0's own training run predates this experiment and was not repeated — its frozen checkpoint/config were verified, not regenerated, per Step 5.)

## Results (Step 7)

| Side-effect encoder | AUC | AUPRC | AP@50 |
|---|--:|--:|--:|
| SE0 — BERT | 0.4113 | 0.5297 | 0.9507 |
| SE1 — PubMedBERT | **0.5936** | **0.6685** | **0.9897** |
| SE2 — SapBERT | 0.4037 | 0.5569 | 0.8782 |

Relative to SE0:

| Encoder | ΔAUC | ΔAUPRC | ΔAP@50 |
|---|--:|--:|--:|
| SE1 — PubMedBERT | **+0.1823** | **+0.1388** | **+0.0390** |
| SE2 — SapBERT | −0.0076 | +0.0272 | **−0.0725** |

All three test sets carry ~855K edges (~458K positive + freshly-sampled negatives, consistent with the fixed evaluation protocol).

## Interpretation

**1. Does PubMedBERT improve over BERT?** Yes, on all three metrics, and by a margin that is large relative to the metrics' own scale (AUC +0.182, a ~44% relative increase; AUPRC +0.139, a ~26% relative increase; AP@50 +0.039).

**2. Does SapBERT improve over BERT?** Not clearly. AUC is essentially flat (−0.008), AUPRC improves modestly (+0.027), but AP@50 — the metric closest to the paper's own headline number and already near-ceiling for SE0/SE1 — drops meaningfully (−0.073, from 0.951 to 0.878). Net effect across the three metrics is mixed, not a consistent improvement.

**3. Which biomedical representation performs best?** PubMedBERT, unambiguously — it wins on all three metrics and is the only encoder tested that improves consistently over the BERT baseline.

**4. Are the differences substantial or negligible?** PubMedBERT's gains are substantial by the scale of these particular metrics (not negligible noise); SapBERT's mixed/negative changes are also large enough (especially the AP@50 drop) to not be dismissed as noise, but they don't point the same direction across metrics, so "SapBERT helps" is not supported.

**5. Limitations from the unresolved GNN reproduction gap:** All absolute values remain far below the paper's Table 5 target (SE0 was already −131 std devs on AUC before this experiment; SE1's best AUC of 0.59 is still −84 std devs). This is the same documented, unresolved gap covered in `notes/gnn_gap_diagnostics.md` and is **not explained or fixed by changing the side-effect encoder** — nothing here should be read as evidence toward resolving that gap. Because the baseline regime itself is anomalous (an AUC of 0.41 is worse than random for a ranking task), these encoder deltas should be read as a **relative, within-reproduction comparison only** — evidence about how this specific (still-broken) GNN implementation responds to different frozen input representations, not a general claim about PubMedBERT/SapBERT for polypharmacy side-effect prediction at large.

## Future work (moved here, not run in this experiment)

- UMLS enrichment: preferred name, synonyms, definitions, semantic types (pipeline built and tested, live retrieval blocked on UMLS API-key eligibility — see `experiment_1_umls_enrichment.md`).
- Wikidata / open-ontology enrichment and bridging (coverage audit completed live — see `experiment_1_wikidata_coverage.md` — 78.5% unique-CUI mapping, 65.8% full name+aliases+description coverage; judged insufficient as the primary source for a controlled 963-label experiment, useful as a bridge/supplement).
- MedDRA/UMLS ontology hierarchy features.
- SE3 (`SapBERT(name + synonyms)`) / SE4 (`PubMedBERT(name + synonyms + definition)`) once a text-enrichment source is finalized.
- A CLS-pooled SapBERT variant, to isolate whether the mean-pooling deviation from SapBERT's own recommended usage is suppressing its performance here.
- Investigating the pre-existing, unresolved GNN reproduction gap itself (out of scope for Experiment 1).
- Biological/mechanistic drug-side-effect context (genes, proteins, pathways) — explicitly deferred per the original scope decision.

## Unresolved issues

- `build_graph.py`'s `build_graph_audit()` hardcodes `feature_source: str(DEFAULT_SEFFECT_FEATURES)` instead of the path actually passed via `--seffect-features`, so `outputs/polyllm/gnn_graph_audit_{pubmedbert,sapbert}.json` cosmetically mislabel their `seffect.feature_source` as the SE0 BERT file. The graphs themselves are correct — verified directly via `torch.equal`/`np.array_equal` against each source `.npy` file (see table above) — this is a provenance-logging bug only, not fixed here since `build_graph.py`'s logic was kept as a fixed variable for this experiment. Worth a one-line fix in a future, non-Experiment-1 pass.
- SE2 (SapBERT) is evaluated only under mean pooling, which its own authors do not recommend; the CLS-pooled variant (future work above) would be needed to know whether SapBERT is a poor fit for this task or specifically a poor fit for mean pooling.
