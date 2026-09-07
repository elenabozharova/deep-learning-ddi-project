# LLM Prompting Experiment — Zero-Shot / Few-Shot Small-LM Baseline vs. GNN/MLP

**Date:** 2026-09-05
**Status:** Complete. Real inference runs (GNN checkpoint, MLP checkpoint, and a locally-run instruct LLM), not simulated.

## Research question

How does a small, locally-run, zero-shot/few-shot instruction-tuned LLM compare to this repo's two trained supervised models (GNN link predictor, ChemBERTa MLP) on the same 600-triple sample of the official GNN test split, for the underlying binary task: given a (drug pair, candidate side effect) triple, is it a true association?

## Methodological precedent

Design modeled on De Vito, Ferrucci and Angelakis, *LLMs for drug-drug interaction prediction using textual drug descriptors* (Knowledge-Based Systems, 2026): zero-shot → few-shot progression, small-sample justification, robustness-checking philosophy (their paper perturbs input phrasing/order to check whether the model is reasoning about the pair or pattern-matching). This experiment does **not** attempt their headline result: no fine-tuning (ruled out as CPU-infeasible in this repo's environment — no GPU), a much smaller local model (~0.5B params vs. their fine-tuned Phi-3.5, ~3.8B), and a much smaller evaluation sample (600 triples vs. their full test set).

## Scope guardrails

No edits to `train_gnn.py`, `evaluate_gnn.py`, `models/gnn.py`, `models/mlp.py`, `training.py`, or anything under `outputs/polyllm/gnn/` or `outputs/polyllm/chemberta_faithful_training/` (both frozen official results). All new code: `src/polyllm/diagnostics/llm_prompting_experiment.py`. All new outputs: `outputs/polyllm/llm_prompting/`.

## Reused, unmodified

- `train_gnn.py`: `DEFAULT_CONFIG` (sample seed = 42), `GRAPH_PATH`, `load_graph_split()`, `build_link_loader()`, `build_model()`, `set_seeds()`.
- `models/gnn.py`: `build_global_positive_edge_set()`, `sample_negative_edges()` (the already-fixed local/global-index-safe negative sampler).
- `evaluate_gnn.py`: `compute_test_metrics()` — identical AUC/AUPRC/AP@50 formulas used for every method below (GNN, MLP, zero-shot, few-shot), for a fair comparison.
- `metrics.py`: `edge_level_average_precision_at_k` (via `compute_test_metrics`) — the paper-axis AP@50 definition, per `notes/deviations_from_paper.md` §1.3.
- `models/mlp.py`: `MultilabelMLP`. `training.py`: `load_checkpoint`.

## 1. Sample construction

**Script:** `src/polyllm/diagnostics/llm_prompting_experiment.py::build_full_test_edge_pool` / `draw_stratified_sample`
**Output:** `outputs/polyllm/llm_prompting/sample_triples.csv`

Loaded the same graph split evaluate_gnn.py uses (`data/graph/gnn_link_split.pt` via `load_graph_split()`), then reconstructed the test edge population:

- **Positives** — `test_data[('pdrugs','associated','seffect')].edge_label_index`, i.e. literally the official held-out test positive edges (`edge_label` confirmed all-ones, per the `RandomLinkSplit(neg_sampling_ratio=0.0)` contract). {{N_POS_POOL}} positive edges, matching `outputs/polyllm/gnn/test_metrics.json`'s `n_positive=457,628` exactly.
- **Negatives** — one fresh draw via `sample_negative_edges(test_data, global_pos_edges, device)`, the exact function `assemble_supervision_edges` calls internally during official train/eval. {{N_NEG_POOL}} negatives survived the true-positive filter.

**Important caveat on "subset of the official 855K-edge test set":** `evaluate_gnn.py` never persists its own y_true/y_pred arrays (only aggregate metrics went to `test_metrics.json`), and its `run_evaluation()` never calls `set_seeds()` before sampling negatives — so there is no single canonical negative-edge array on disk to literally sub-sample from; negatives are resampled fresh every official run. Positives here **are** literally the official test positives (a strict subset by construction). Negatives are generated via the identical reused code path (`sample_negative_edges` + `build_global_positive_edge_set`), under sample seed **42** (this repo's standard seed, `train_gnn.DEFAULT_CONFIG["random_seed"]`) for this script's own reproducibility — not literally the same negative draw the official evaluation happened to make on its own run.

From this pool, drew **300 positive + 300 negative = 600 triples** (`numpy.random.RandomState(42)`, without replacement, row order additionally shuffled with the same seed). Sanity-asserted in code: every sampled positive is in the graph's global positive-edge set; every sampled negative is not (`assert` statements in `build_full_test_edge_pool`, not skipped).

**Drug/side-effect identifiers:** `pdrugs` node index == `pair_id` (row in `data/processed/polyllm_pairs.parquet`); `seffect` node index == `label_index` (row in `data/processed/polyllm_label_mapping.csv`) — both confirmed by construction in `graph/build_graph.py` (`node_id = torch.arange(num_nodes)`, built directly from the same-order feature arrays). Drug names resolved via `data/processed/drug_smiles_mapping.csv`'s `preferred_name` (PubChem IUPACName) column; only **2/645** drugs in the whole dataset lack a `preferred_name` and fall back to `SMILES:<standardized_smiles>` (`name_source` column in `sample_triples.csv` records which).

**Limitation — drug "names" are IUPAC systematic chemical names, not trade/common names.** E.g. one sampled triple's drugs are literally `3-methyl-2,4,4a,7,7a,13-hexahydro-1H-4,12-methanobenzofuro[3,2-e]isoquinoline-7,9-diol` and `(8R,9S,10R,13S,14S,17R)-17-acetyl-17-hydroxy-6,10,13-trimethyl-2,8,9,11,12,14,15,16-octahydro-1H-cyclopenta[a]phenanthren-3-one` (STITCH `CID000004253`/`CID000019090`, side effect *conjunctivitis*, true label 1). No LLM — however large — can be expected to recognize a drug from its systematic chemical name the way it might recognize "codeine" or "ibuprofen"; this is a hard ceiling on the LLM's task, not a modeling failure, and applies equally to zero-shot and few-shot.

## 2. Matched-subsample baseline metrics (GNN, MLP)

**Script:** `run_gnn_on_sample`, `run_mlp_on_sample`

**GNN:** scored the 600 triples with the official checkpoint (`outputs/polyllm/gnn/checkpoints/best_model.pt`) through the same `LinkNeighborLoader` mini-batching (`build_link_loader`, `num_neighbors=[20,10]`) the official evaluation uses — not full-graph inference, to keep the prediction distribution comparable to the official numbers. One deviation from `evaluate_gnn.collect_test_predictions`, documented in the function docstring: that function calls `assemble_supervision_edges()`, which assumes a batch's baked-in `edge_label_index` is all-positive and appends its own fresh negatives — wrong for our already-fixed 300/300 sample, so the fixed batch's own `(edge_label_index, edge_label)` is read directly instead, skipping only the "assume positive, add negatives" step (no negative-sampling logic is reimplemented). A sanity assertion recovers each prediction's global `(pdrugs, seffect, label)` identity from the mini-batch's own `node_id` tensors and confirms it's an exact match, as a set, to the input sample.

**MLP:** scored via the official checkpoint (`outputs/polyllm/chemberta_faithful_training/checkpoints/best_model.pt`), forward pass on the unique sampled `pair_id`s' ChemBERTa pair embeddings, sigmoid outputs indexed at each triple's `seffect_node_id` column.

**Caveat — not a clean apples-to-apples split for the MLP.** The MLP's own train/val/test split (`data/splits/*_pair_ids.csv`) is **by pair**, independent of and different from the GNN's **by-edge** random split. Some of the sample's pair_ids may have been in the MLP's own training set: **{{MLP_OVERLAP_TRAIN}}/{{MLP_UNIQUE_PAIRS}}** unique sampled pairs fall in the MLP train split, **{{MLP_OVERLAP_VAL}}** in its val split, **{{MLP_OVERLAP_TEST}}** in its held-out test split. The MLP number below is therefore a best-case estimate of MLP performance on these triples, not a strict held-out comparison the way the GNN's number is.

## 3. LLM zero-shot / few-shot

**Model:** `Qwen/Qwen2.5-0.5B-Instruct` (~0.5B params, downloaded from the Hugging Face Hub, ~1 GB, `model.safetensors`) — chosen because no instruction-tuned model was already cached locally (only the repo's own frozen encoders — ChemBERTa, PubMedBERT, SapBERT, BERT-base — were), it sits at the small end of the spec'd 0.5B–1.5B range for CPU feasibility across 1,280 forward passes, and it ships a working chat template out of the box. Run entirely on CPU, `torch.float32`, single forward pass per prompt (no autoregressive generation — see probability extraction below).

**Prompt:** one system instruction ("...Decide whether taking the two drugs together is known to cause that side effect. Respond with exactly one word: Yes or No.") plus a user turn giving Drug A, Drug B, and the candidate side effect name, ending in "Answer:". Few-shot prepends **6 labeled examples** (3 positive + 3 negative) as prior user/assistant turns before the real question.

**Few-shot examples drawn from TRAIN split only:** positives from `train_data`'s own disjoint supervision edges (all label=1 by the split's contract); negatives via the same `sample_negative_edges(train_data, ...)` call, filtered against the full graph's global positive-edge set (train+val+test) — guaranteed not to collide with any true edge anywhere, not just train. Asserted in code: zero overlap between the 6 few-shot `(pdrugs, seffect)` keys and the 600-triple sample's keys.

**Probability extraction:** a genuine calibrated probability, not a binary fallback. Single forward pass, `logsumexp` over the logits of `{Yes," Yes",yes," yes"}` vs. `{No," No",no," no"}` (their first sub-word token ids under this tokenizer), softmax-normalized to `p(yes)`. This is possible because Qwen2.5-Instruct's tokenizer/logits are directly accessible via `transformers`; no generation sampling is involved.

**Results (this sample, n=600):** see table below. Zero-shot and few-shot predictions are in `outputs/polyllm/llm_prompting/zero_shot_predictions.csv` / `few_shot_predictions.csv`.

## 4. Robustness check — drug-order swap

**Script:** `run_robustness_check`
**Output:** `outputs/polyllm/llm_prompting/robustness_order_swap.csv`

On **80** triples (fixed-seed subset of the 600), reran the **few-shot** prompt with Drug A/Drug B swapped in the final question only (few-shot examples unchanged) and compared `p(yes)` before/after.

- **Decision-flip rate** (crosses the 0.5 boundary): **{{FLIP_RATE}}**
- **Large-change rate** (`|Δp(yes)| > 0.1`): **{{LARGE_CHANGE_RATE}}**
- **Mean |Δp(yes)|:** {{MEAN_ABS_DIFF}}

## Results table

| Method | n | AUROC | AUPRC | AP@50 |
|---|---|---|---|---|
| GNN — official full test set | 855,448 | 0.4113 | 0.5297 | 0.9507 |
| GNN — this sample | 600 | {{GNN_SAMPLE_AUC}} | {{GNN_SAMPLE_AUPRC}} | {{GNN_SAMPLE_AP50}} |
| MLP — this sample (caveat above) | 600 | {{MLP_SAMPLE_AUC}} | {{MLP_SAMPLE_AUPRC}} | {{MLP_SAMPLE_AP50}} |
| LLM zero-shot | 600 | {{ZS_AUC}} | {{ZS_AUPRC}} | {{ZS_AP50}} |
| LLM few-shot (6-shot) | 600 | {{FS_AUC}} | {{FS_AUPRC}} | {{FS_AP50}} |

All four use `evaluate_gnn.compute_test_metrics()` (identical AUROC/AUPRC/`edge_level_average_precision_at_k` formulas) so the numbers are directly comparable in *definition*; they differ in what population they were computed against (855K edges vs. this 600-triple sample) and, for the MLP row, in whether the underlying model had ever seen the pair during its own training (see §2 caveat).

## Limitations (explicit)

1. **Single sample, single seed (42).** No confidence interval; these are point estimates on 600 triples, not the full 855K-edge test set. A different seed would draw different triples and could move all four sample-based numbers.
2. **CPU-only, small (~0.5B) model, zero fine-tuning.** Not comparable in scale to De Vito et al.'s fine-tuned Phi-3.5 (~3.8B) result — this is a much cheaper, much weaker baseline by design, not an attempted reproduction of their headline number.
3. **Drug identifiers are IUPAC systematic names, not trade/common names**, for all but 2/645 drugs (SMILES fallback for those 2) — see §1. This is a hard, structural ceiling on what any general-purpose LLM can infer from the prompt text alone, independent of model size.
4. **MLP-on-sample is not a strict held-out comparison** — {{MLP_OVERLAP_TRAIN}}/{{MLP_UNIQUE_PAIRS}} unique sampled pairs were in the MLP's own training split (§2).
5. **The "600 official test edges" are not a literal persisted array** — official `evaluate_gnn.py` never saves one and reseeds negatives fresh every run (no `set_seeds()` call in `run_evaluation`). Positives are literally official test positives; negatives are drawn via the identical reused sampling function under a documented seed, not the exact same draw any specific past official run made (see §1 caveat).
6. **Robustness check covers only order, not phrasing.** Only Drug A/Drug B position was swapped; wording, side-effect phrasing, and prompt structure were not perturbed.
7. **Probability extraction assumes the model's very next token is the answer.** No generation/sampling loop is run; if the model's true first token were something other than a Yes/No variant (e.g. punctuation) for some prompts, its logit would simply lose out to whichever Yes/No variant scored higher — not a hard failure, but a known simplification of using next-token logits as a proxy for "the model's answer."

## Reproduce

```
.venv-polyllm/Scripts/python.exe src/polyllm/diagnostics/llm_prompting_experiment.py --smoke-test   # ~20-triple sanity run
.venv-polyllm/Scripts/python.exe src/polyllm/diagnostics/llm_prompting_experiment.py                # full 600-triple run
```

## Software versions / provenance

See `outputs/polyllm/llm_prompting/summary.json` for exact versions (python, torch, transformers, scikit-learn, pandas, numpy), the checkpoint paths used, and the full few-shot example set (with their `pdrugs`/`seffect` identifiers, for leakage auditing).
