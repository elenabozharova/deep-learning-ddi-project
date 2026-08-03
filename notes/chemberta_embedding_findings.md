# Milestone 4 — ChemBERTa Embedding Findings

**Date:** 2026-06-27  
**Status:** Complete — ready for review

---

## 1. Input and configuration

| Property | Value |
|---|---|
| Input mapping file | `data/processed/drug_smiles_mapping.csv` |
| Input pairs file | `data/processed/polyllm_pairs.parquet` |
| SMILES column used | `standardized_smiles` |
| Reason for column choice | Milestone 2 explicitly documents `standardized_smiles` as RDKit canonical isomeric SMILES (`isomericSmiles=True`); no `rdkit_canonical_isomeric_smiles` column exists |
| Total unique drugs | 645 |
| Drug ordering | Sorted lexicographically by `stitch_id` → `embedding_index` 0–644 |
| Random seed | 42 (`torch.manual_seed(42)` at pipeline start) |

---

## 2. Model and tokenizer

| Property | Value |
|---|---|
| Model name | `DeepChem/ChemBERTa-77M-MLM` |
| Architecture | `RobertaModel` (base encoder; no MLM head) |
| Tokenizer class | `RobertaTokenizer` |
| Resolved HF revision SHA | `ed8a5374f2024ec8da53760af91a33fb8f6a15ff` |
| Revision pinning | SHA resolved via `huggingface_hub.model_info()` before loading; same SHA passed to both `AutoTokenizer.from_pretrained` and `AutoModel.from_pretrained` |
| Hidden dimension | **384** (discovered at runtime from model output; not hardcoded) |
| Parameters frozen | `model.requires_grad_(False)` applied before inference |
| Inference mode | `torch.inference_mode()` context inside each batch |
| Compute device | CPU (CUDA not available) |

### 2.1 Load report — MLM head and pooler warnings

During loading, `transformers` printed a load report with two warning categories:

**UNEXPECTED keys — `lm_head.*` (7 weight tensors):**  
The `DeepChem/ChemBERTa-77M-MLM` checkpoint was trained as a masked language model and therefore contains an MLM head (`lm_head.dense`, `lm_head.decoder`, `lm_head.layer_norm`, `lm_head.bias`). When loaded with `AutoModel` (which maps to `RobertaModel`, the base encoder), these head weights are silently discarded. They are not part of the encoder stack and have no effect on `last_hidden_state`. The warning is expected whenever an MLM checkpoint is loaded through the base-model class.

**MISSING keys — `pooler.dense.weight`, `pooler.dense.bias`:**  
`RobertaModel` includes an optional CLS-token pooler layer that is used in some sequence-classification setups. The checkpoint does not contain those weights, so transformers randomly initializes them. In this pipeline the pooler is never called: the forward pass returns `last_hidden_state` (the full per-token hidden states from the final encoder layer), and our own attention-mask-aware mean pooling is applied on top of that tensor. The randomly initialized pooler weights are loaded into memory but receive no inputs and produce no outputs during inference.

**Summary:** neither warning affects the values in `last_hidden_state` or the mean-pooled embeddings. The encoder weights (all 12 transformer layers) loaded without error.

---

## 3. Tokenization audit

The audit was run without truncation (`truncation=False`) and with special tokens enabled (`add_special_tokens=True`) to measure the natural length of each SMILES before any capping.

| Metric | Value |
|---|---|
| Minimum token length | 4 |
| Median token length | 42.0 |
| Mean token length | 47.45 |
| Maximum token length | 234 |
| Max allowed at inference | 64 |
| Drugs exceeding 64 tokens | **86 / 645 (13.33%)** |

The 86 drugs whose natural token length exceeds 64 have their SMILES truncated at the `max_length=64` boundary during embedding inference. The tokenizer adds special tokens (BOS/EOS) within that budget. Truncated drugs tend to be large, structurally complex molecules. The `was_truncated` flag in the drug index CSV identifies them individually. The token length for each truncated drug is its **natural** length (without truncation), measured in the audit step.

### 3.1 Complete list of truncated drugs

| STITCH ID | Natural token length |
|---|---|
| CID000000772 | 143 |
| CID000001690 | 75 |
| CID000001972 | 114 |
| CID000002019 | 176 |
| CID000002142 | 71 |
| CID000002250 | 73 |
| CID000002269 | 99 |
| CID000002283 | 181 |
| CID000002650 | 74 |
| CID000002656 | 72 |
| CID000002891 | 186 |
| CID000002909 | 169 |
| CID000003062 | 99 |
| CID000003066 | 78 |
| CID000003143 | 114 |
| CID000003161 | 69 |
| CID000003255 | 99 |
| CID000003310 | 73 |
| CID000003381 | 66 |
| CID000003382 | 70 |
| CID000003419 | 68 |
| CID000003706 | 73 |
| CID000003724 | 116 |
| CID000003750 | 79 |
| CID000003793 | 83 |
| CID000003911 | 154 |
| CID000004200 | 68 |
| CID000004451 | 66 |
| CID000004473 | 65 |
| CID000004645 | 72 |
| CID000004666 | 117 |
| CID000004675 | 73 |
| CID000004834 | 71 |
| CID000005040 | 120 |
| CID000005052 | 74 |
| CID000005076 | 85 |
| CID000005297 | 75 |
| CID000005372 | 104 |
| CID000005412 | 69 |
| CID000005651 | 186 |
| CID000005672 | 108 |
| CID000005717 | 71 |
| CID000005771 | 126 |
| CID000005978 | 113 |
| CID000013342 | 111 |
| CID000027991 | 130 |
| CID000039765 | 70 |
| CID000041774 | 79 |
| CID000047320 | 145 |
| CID000047725 | 163 |
| CID000048175 | 66 |
| CID000054374 | 120 |
| CID000054688 | 100 |
| CID000060696 | 66 |
| CID000060787 | 83 |
| CID000065027 | 78 |
| CID000072467 | 175 |
| CID000104741 | 66 |
| CID000104865 | 66 |
| CID000130881 | 71 |
| CID000131536 | 71 |
| CID000147912 | 87 |
| CID000148192 | 92 |
| CID000150610 | 68 |
| CID000151165 | 68 |
| CID000152945 | 68 |
| CID000163742 | 187 |
| CID000213039 | 67 |
| CID000444033 | 72 |
| CID000504578 | 71 |
| CID003002190 | 107 |
| CID003081884 | 79 |
| CID003086258 | 79 |
| CID003468412 | 114 |
| CID004479097 | 183 |
| CID005282044 | 88 |
| CID005311039 | 131 |
| CID005311082 | 97 |
| CID005362420 | 99 |
| CID005381226 | 121 |
| CID005493444 | 67 |
| CID006398525 | 169 |
| CID006435110 | 234 |
| CID006436173 | 114 |
| CID006447131 | 106 |
| CID009571074 | 67 |

The longest SMILES belongs to CID006435110 (234 tokens natural length), which loses 170 tokens to truncation. The shortest truncated drug has a natural length of 65 (CID000004473), exceeding the 64-token cap by a single token.

---

## 4. Pooling method

The embedding for each drug is computed as the **mean of the final hidden-layer vectors for all non-padding tokens**. Formally:

```
mask   = attention_mask.unsqueeze(-1).to(hidden_state.dtype)
pooled = (hidden_state * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
```

- Tokens where `attention_mask == 1` are included (content tokens, BOS, EOS).
- Padding tokens (`attention_mask == 0`) do not contribute.
- The denominator is clamped to 1 to prevent division by zero; this case cannot occur in practice because every valid SMILES produces at least the BOS and EOS special tokens.

The pooling averages all non-padding positions. It does not select a single representative token (such as the CLS vector), and it does not apply L2 normalization.

---

## 5. Embedding results

| Metric | Value |
|---|---|
| Output shape | **(645, 384)** |
| Dtype | float32 |
| All values finite | ✓ |
| All-zero rows | 0 |
| L2 norm range | 2.67 – 5.47 (vectors are not normalized; mean pooling averages token representations without forcing unit norm) |
| Drug index rows | 645 |
| embedding_index range | 0 → 644 (sequential, verified) |
| stitch_ids sorted | ✓ |
| stitch_ids unique | ✓ |
| was_truncated entries | 86 |

All 645 pair-table drugs have an embedding (verified in post-run disk validation).

---

## 6. Post-run disk validation

Nine conditions checked after reloading `chemberta_drug_embeddings.npy` and `chemberta_drug_index.csv` from disk:

| Check | Result |
|---|---|
| Index CSV row count == 645 | ✓ |
| embedding_index is exactly 0..644 | ✓ |
| All stitch_ids unique | ✓ |
| stitch_ids sorted lexicographically | ✓ |
| Matrix rows == index rows | ✓ |
| Matrix columns == hidden dim (384) | ✓ |
| Matrix dtype == float32 | ✓ |
| All values finite (no NaN, no Inf) | ✓ |
| All pair-table drugs present in index | ✓ |

Additionally: no all-zero rows (checked in in-memory pre-write validation); `token_length` and `was_truncated` columns are non-null (checked in disk validation).

---

## 7. Output files

| File | Description |
|---|---|
| `data/features/chemberta_drug_embeddings.npy` | Float32 numpy array, shape (645, 384) |
| `data/features/chemberta_drug_index.csv` | 645 rows, columns: `embedding_index, stitch_id, parsed_pubchem_cid, smiles_source_column, smiles_sha256, token_length, was_truncated` |
| `outputs/polyllm/embedding_audit.json` | Full audit including token-length stats, model revision, artifact SHA-256 hashes, software versions |
| `notes/chemberta_embedding_findings.md` | This document |
| `src/polyllm/features/generate_chemberta_embeddings.py` | Implementation |
| `tests/test_generate_chemberta_embeddings.py` | 45 tests, all mocked |
| `requirements-milestone4.txt` | Cumulative pinned requirements (M1 + M2 + M4) |

**Artifact SHA-256 (at generation time):**

| File | SHA-256 |
|---|---|
| `chemberta_drug_embeddings.npy` | `b1a0cc7ab7acbd1d4e695939d760b8de60b0dc1eb4a92d8168e7a62e76df1297` |
| `chemberta_drug_index.csv` | `0c99fd08d4fb4e96cb56d4e70d4a3296073945a979be3c8daae57ab063b01154` |

---

## 8. Tests

**Command:** `.venv-polyllm\Scripts\python.exe -m pytest tests/ -q`  
**Result:** **170 / 170 tests passed** (45 Milestone 4 + 125 Milestones 1–3)

Milestone 4 test coverage:

| Class | Topics covered |
|---|---|
| `TestInputValidation` | Missing columns; non-mapped status; duplicate stitch_ids; null SMILES; drug absent from pair table; valid case passes |
| `TestTokenizationAudit` | Lengths returned correctly; truncation flagged; no truncation → no IDs listed |
| `TestMeanPool` | All unmasked; padding excluded; batch of two; fully-masked clamp |
| `TestSmokeTest` | Valid mock passes; NaN raises; unfrozen parameter raises |
| `TestEmbedDrugs` | Output shape; dtype float32; batch boundary |
| `TestDrugIndex` | Schema and SHA-256; truncation flag; smiles_source_column recorded |
| `TestDiskValidation` | Valid artifacts pass; wrong row count raises; non-finite raises; missing pair drug raises |
| `TestAuditSchema` | Required fields present; `special_tokens_included_in_pooling` is True |
| `TestOverwriteGuard` | Skips full pipeline when artifacts exist; result has `embedding_shape` |
| `TestDeterministicOrder` | Sort by stitch_id before embedding |
| `TestComputeTokenLengths` | `truncation=False` in call; `add_special_tokens=True`; length matches ids |
| `TestSha256` | Deterministic; matches known value |
| `TestAllZeroRows` | All-zero row raises; valid matrix passes |
| `TestFiniteCheck` | NaN raises; Inf raises |
| `TestDtypeEnforcement` | float64 raises; float32 passes |
| `TestSmilesSourceColumn` | Constant value; index records column; unexpected rdkit col raises |
| `TestModelFrozen` | Frozen model passes smoke test; unfrozen param fails smoke test |

---

## 9. Software versions

| Package | Version |
|---|---|
| Python | 3.11.9 |
| PyTorch | 2.12.1+cpu |
| transformers | 5.12.1 |
| huggingface-hub | 1.21.0 |
| safetensors | 0.8.0 |
| NumPy | 2.4.6 |
| pandas | 3.0.3 |

---

## 10. Known limitations

- **Truncation at 64 tokens.** 86 of 645 drugs (13.33%) have SMILES strings that tokenize to more than 64 tokens without truncation. Their embeddings are derived from a truncated prefix of the SMILES. This affects large, structurally complex molecules. The `was_truncated` column in the index CSV identifies them.

- **CPU-only inference.** No GPU was available. Inference ran on CPU and completed in approximately one second for all 645 drugs at batch size 32.

- **No safetensors checkpoint.** The `DeepChem/ChemBERTa-77M-MLM` repository does not provide a `model.safetensors` file; the pipeline falls back to `pytorch_model.bin`. Safetensors was installed but was not used for this model.

- **Randomly initialized pooler weights.** The `pooler.dense` weights in `RobertaModel` are randomly initialized because the checkpoint does not include them. These weights are never used — our mean pooling bypasses the pooler entirely — so this is harmless.

- **Embeddings are frozen.** The model was not fine-tuned on this dataset. The embeddings reflect the pre-training distribution of ChemBERTa (PubChem SMILES) rather than anything specific to the Decagon drug pairs.

- **Per-drug embeddings only.** This milestone produces one embedding per unique drug. Pair-level features (e.g., concatenation, difference, or element-wise product of drug embeddings) are not computed here. That is deferred to Milestone 5.

- **No salt stripping, tautomer normalization, or stereoisomer merging.** The SMILES passed to the tokenizer are the RDKit canonical isomeric strings from Milestone 2. No additional chemical standardization was applied.
