"""
Milestone 4 — Generate and cache one frozen ChemBERTa embedding per unique drug.

Protocol
--------
Model  : DeepChem/ChemBERTa-77M-MLM  (frozen, eval-mode)
Input  : RDKit canonical isomeric SMILES from Milestone 2
Pooling: attention-mask-aware mean pooling over the final hidden layer
         (all tokens where attention_mask == 1, including special tokens)
Output : (645, hidden_dim) float32 NumPy array + drug index CSV

Run from the project root:
    python src/polyllm/features/generate_chemberta_embeddings.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import platform
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODEL_NAME = "DeepChem/ChemBERTa-77M-MLM"
MAX_TOKEN_LENGTH = 64
DEFAULT_BATCH_SIZE = 32
RANDOM_SEED = 42
SMILES_COLUMN = "standardized_smiles"  # RDKit canonical isomeric SMILES (Milestone 2)

DEFAULT_MAPPING_INPUT = Path("data/processed/drug_smiles_mapping.csv")
DEFAULT_PAIRS_INPUT   = Path("data/processed/polyllm_pairs.parquet")
DEFAULT_EMBED_OUTPUT  = Path("data/features/chemberta_drug_embeddings.npy")
DEFAULT_INDEX_OUTPUT  = Path("data/features/chemberta_drug_index.csv")
DEFAULT_AUDIT_OUTPUT  = Path("outputs/polyllm/embedding_audit.json")

REQUIRED_INDEX_COLUMNS = [
    "embedding_index",
    "stitch_id",
    "parsed_pubchem_cid",
    "smiles_source_column",
    "smiles_sha256",
    "token_length",
    "was_truncated",
]


# ---------------------------------------------------------------------------
# Step 1: Input validation
# ---------------------------------------------------------------------------

def validate_mapping_df(mapping_df: pd.DataFrame, pairs_df: pd.DataFrame) -> None:
    """
    Raise ValueError if the mapping CSV does not meet Milestone 4 preconditions.

    Checks:
    - Required columns present
    - Exactly one SMILES column to use (standardized_smiles)
    - All statuses are 'mapped'
    - No duplicate stitch_ids
    - No null canonical SMILES
    - Every drug in the pair table exists in the mapping
    """
    required_cols = {"stitch_id", "mapping_status", SMILES_COLUMN}
    missing = required_cols - set(mapping_df.columns)
    if missing:
        raise ValueError(f"Mapping CSV missing required columns: {sorted(missing)}")

    # Validate SMILES column selection
    has_rdkit_col = "rdkit_canonical_isomeric_smiles" in mapping_df.columns
    if has_rdkit_col:
        raise ValueError(
            "Unexpected column 'rdkit_canonical_isomeric_smiles' found. "
            "Update SMILES_COLUMN to use it."
        )
    # standardized_smiles is confirmed as RDKit canonical isomeric SMILES in
    # notes/smiles_mapping_findings.md §5.

    non_mapped = mapping_df[mapping_df["mapping_status"] != "mapped"]
    if len(non_mapped):
        raise ValueError(
            f"{len(non_mapped)} rows have non-mapped status: "
            f"{non_mapped['mapping_status'].value_counts().to_dict()}"
        )

    dup_ids = mapping_df["stitch_id"][mapping_df["stitch_id"].duplicated()].tolist()
    if dup_ids:
        raise ValueError(f"Duplicate stitch_ids in mapping: {dup_ids[:5]}")

    null_smiles = mapping_df[SMILES_COLUMN].isna() | (
        mapping_df[SMILES_COLUMN].astype(str).str.strip() == ""
    )
    if null_smiles.any():
        raise ValueError(
            f"{null_smiles.sum()} rows have null/empty '{SMILES_COLUMN}'."
        )

    mapping_ids = set(mapping_df["stitch_id"])
    pair_drugs = set(pairs_df["drug_1"].tolist()) | set(pairs_df["drug_2"].tolist())
    missing_drugs = pair_drugs - mapping_ids
    if missing_drugs:
        raise ValueError(
            f"{len(missing_drugs)} drugs in pair table are absent from mapping: "
            f"{sorted(missing_drugs)[:5]}"
        )


# ---------------------------------------------------------------------------
# Step 2: Tokenization audit
# ---------------------------------------------------------------------------

def compute_token_lengths(
    smiles_list: list[str],
    tokenizer: Any,
    max_length: int = MAX_TOKEN_LENGTH,
) -> list[int]:
    """
    Return the natural token length (with special tokens, without truncation)
    for each SMILES string.
    """
    lengths: list[int] = []
    for smi in smiles_list:
        enc = tokenizer(
            smi,
            add_special_tokens=True,
            truncation=False,
            return_tensors=None,
        )
        lengths.append(len(enc["input_ids"]))
    return lengths


def tokenization_audit(
    smiles_list: list[str],
    stitch_ids: list[str],
    tokenizer: Any,
    max_length: int = MAX_TOKEN_LENGTH,
) -> dict[str, Any]:
    """Compute token-length statistics for a corpus of SMILES strings."""
    lengths = compute_token_lengths(smiles_list, tokenizer, max_length)
    arr = np.array(lengths)
    truncated_mask = arr > max_length
    truncated_ids = [sid for sid, t in zip(stitch_ids, truncated_mask) if t]

    return {
        "min_token_length": int(arr.min()),
        "median_token_length": float(np.median(arr)),
        "mean_token_length": float(round(arr.mean(), 4)),
        "max_token_length": int(arr.max()),
        "count_exceeding_max": int(truncated_mask.sum()),
        "percent_exceeding_max": float(round(100 * truncated_mask.mean(), 4)),
        "truncated_stitch_ids": truncated_ids,
        "per_drug_token_lengths": lengths,
    }


# ---------------------------------------------------------------------------
# Step 3: Mean pooling
# ---------------------------------------------------------------------------

def mean_pool(
    hidden_state: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """
    Average the final hidden-layer embeddings of all non-padding tokens.

    This is attention-mask-aware mean pooling: it sums the hidden vectors for
    every token where attention_mask == 1 (including CLS, EOS/SEP, and content
    tokens) and divides by the count of those tokens.  Padding tokens (where
    attention_mask == 0) do not contribute.

        mask   = attention_mask.unsqueeze(-1).to(hidden_state.dtype)
        pooled = (hidden_state * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)

    Special tokens (CLS, EOS) are included because the ChemBERTa tokenizer adds
    them and their attention_mask is 1.
    """
    mask   = attention_mask.unsqueeze(-1).to(hidden_state.dtype)
    pooled = (hidden_state * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
    return pooled


# ---------------------------------------------------------------------------
# Step 4: Smoke test
# ---------------------------------------------------------------------------

def smoke_test(
    smiles_sample: list[str],
    tokenizer: Any,
    model: Any,
    device: torch.device,
) -> int:
    """
    Embed 3–5 drugs and verify correctness.

    Returns the discovered hidden dimension.
    Raises RuntimeError on any failure.
    """
    assert 3 <= len(smiles_sample) <= 5, "Smoke test requires 3–5 samples."

    # Two independent forward passes on the same input
    results = []
    for _ in range(2):
        enc = tokenizer(
            smiles_sample,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=MAX_TOKEN_LENGTH,
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.inference_mode():
            out = model(**enc)
        pooled = mean_pool(out.last_hidden_state, enc["attention_mask"])
        results.append(pooled.cpu().float())

    # Shape: one vector per drug
    if results[0].shape[0] != len(smiles_sample):
        raise RuntimeError(
            f"Pooling returned {results[0].shape[0]} vectors "
            f"for {len(smiles_sample)} inputs."
        )

    # No NaN or Inf
    for r in results:
        if not torch.isfinite(r).all():
            raise RuntimeError("Smoke test: NaN or Inf in embeddings.")

    # Numerically consistent within floating-point tolerance
    if not torch.allclose(results[0], results[1], atol=1e-5, rtol=1e-5):
        max_diff = (results[0] - results[1]).abs().max().item()
        raise RuntimeError(
            f"Smoke test: repeated inference differs by {max_diff:.2e} (threshold 1e-5)."
        )

    # Model parameters frozen (requires_grad_(False) was called before this point)
    for p in model.parameters():
        if p.requires_grad:
            raise RuntimeError("Smoke test: model parameter found with requires_grad=True.")

    hidden_dim = results[0].shape[1]
    logger.info("Smoke test passed. Hidden dim = %d.", hidden_dim)
    return hidden_dim


# ---------------------------------------------------------------------------
# Step 5: Full embedding generation
# ---------------------------------------------------------------------------

def embed_drugs(
    smiles_list: list[str],
    tokenizer: Any,
    model: Any,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    """
    Embed all SMILES strings in deterministic batch order.

    Returns a float32 NumPy array of shape (n_drugs, hidden_dim).
    """
    all_vecs: list[np.ndarray] = []
    n = len(smiles_list)
    total_batches = (n + batch_size - 1) // batch_size

    for batch_idx, start in enumerate(range(0, n, batch_size), 1):
        batch = smiles_list[start : start + batch_size]
        logger.info("Embedding batch %d/%d (%d SMILES).", batch_idx, total_batches, len(batch))

        enc = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=MAX_TOKEN_LENGTH,
        )
        enc = {k: v.to(device) for k, v in enc.items()}

        with torch.inference_mode():
            out = model(**enc)

        pooled = mean_pool(out.last_hidden_state, enc["attention_mask"])
        all_vecs.append(pooled.cpu().float().numpy())

    return np.concatenate(all_vecs, axis=0).astype(np.float32)


# ---------------------------------------------------------------------------
# Step 6: Build drug index
# ---------------------------------------------------------------------------

def build_drug_index(
    mapping_df: pd.DataFrame,
    token_lengths: list[int],
    smiles_list: list[str],
    max_length: int = MAX_TOKEN_LENGTH,
) -> pd.DataFrame:
    """Build the drug index CSV as a DataFrame."""
    rows = []
    for i, (_, row) in enumerate(mapping_df.iterrows()):
        smi = smiles_list[i]
        sha256 = hashlib.sha256(smi.encode("utf-8")).hexdigest()
        tlen  = token_lengths[i]
        rows.append({
            "embedding_index": i,
            "stitch_id": row["stitch_id"],
            "parsed_pubchem_cid": row.get("parsed_pubchem_cid"),
            "smiles_source_column": SMILES_COLUMN,
            "smiles_sha256": sha256,
            "token_length": tlen,
            "was_truncated": bool(tlen > max_length),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Step 7: Post-run disk validation
# ---------------------------------------------------------------------------

def validate_outputs_from_disk(
    embed_path: Path,
    index_path: Path,
    pairs_df: pd.DataFrame,
    hidden_dim: int,
    expected_drug_count: int = 645,
) -> None:
    """
    Reload the saved files and re-verify all postconditions.

    Raises ValueError on any failure.
    """
    matrix = np.load(embed_path)
    index_df = pd.read_csv(index_path)

    if len(index_df) != expected_drug_count:
        raise ValueError(
            f"Index CSV has {len(index_df)} rows; expected {expected_drug_count}."
        )

    expected_indices = np.arange(expected_drug_count)
    if not np.array_equal(index_df["embedding_index"].to_numpy(), expected_indices):
        raise ValueError(f"embedding_index is not exactly 0..{expected_drug_count - 1}.")

    if index_df["stitch_id"].nunique() != expected_drug_count:
        raise ValueError("stitch_ids in index are not all unique.")

    if list(index_df["stitch_id"]) != sorted(index_df["stitch_id"].tolist()):
        raise ValueError("stitch_ids in index are not sorted.")

    if matrix.shape[0] != len(index_df):
        raise ValueError(
            f"Matrix has {matrix.shape[0]} rows; index has {len(index_df)} rows."
        )

    if matrix.shape[1] != hidden_dim:
        raise ValueError(
            f"Matrix has {matrix.shape[1]} columns; expected {hidden_dim}."
        )

    if matrix.dtype != np.float32:
        raise ValueError(f"Matrix dtype is {matrix.dtype}; expected float32.")

    if not np.isfinite(matrix).all():
        n_bad = (~np.isfinite(matrix)).sum()
        raise ValueError(f"Matrix contains {n_bad} non-finite values.")

    all_zero = (matrix == 0).all(axis=1)
    if all_zero.any():
        raise ValueError(f"{all_zero.sum()} all-zero rows detected in embedding matrix.")

    mapping_ids = set(index_df["stitch_id"])
    pair_drugs = set(pairs_df["drug_1"].tolist()) | set(pairs_df["drug_2"].tolist())
    missing = pair_drugs - mapping_ids
    if missing:
        raise ValueError(f"{len(missing)} pair-table drugs missing from index.")

    if index_df["token_length"].isna().any():
        raise ValueError("token_length column has null values.")

    if index_df["was_truncated"].isna().any():
        raise ValueError("was_truncated column has null values.")


# ---------------------------------------------------------------------------
# Step 8: Artifact SHA-256
# ---------------------------------------------------------------------------

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Step 9: Audit
# ---------------------------------------------------------------------------

def build_embedding_audit(
    mapping_df: pd.DataFrame,
    index_df: pd.DataFrame,
    matrix: np.ndarray,
    tok_audit: dict[str, Any],
    resolved_revision: str | None,
    device_str: str,
    cuda_available: bool,
    batch_size: int,
    embed_path: Path,
    index_path: Path,
    audit_path: Path,
    tokenizer_class: str,
    model_class: str,
) -> dict[str, Any]:
    all_zero_count = int((matrix == 0).all(axis=1).sum())

    return {
        "model_name": MODEL_NAME,
        "resolved_model_revision": resolved_revision,
        "tokenizer_class": tokenizer_class,
        "model_class": model_class,
        "source_smiles_column": SMILES_COLUMN,
        "maximum_token_length": MAX_TOKEN_LENGTH,
        "pooling_method": "mean of non-padding token embeddings (attention-mask-aware, final hidden layer)",
        "special_tokens_included_in_pooling": True,
        "batch_size": batch_size,
        "device": device_str,
        "cuda_available": cuda_available,
        "random_seed": RANDOM_SEED,
        "total_drugs": int(len(index_df)),
        "embedding_dimension": int(matrix.shape[1]),
        "embedding_shape": list(matrix.shape),
        "embedding_dtype": str(matrix.dtype),
        "finite_value_check": bool(np.isfinite(matrix).all()),
        "all_zero_row_count": all_zero_count,
        "token_length_summary": {
            "min": tok_audit["min_token_length"],
            "median": tok_audit["median_token_length"],
            "mean": tok_audit["mean_token_length"],
            "max": tok_audit["max_token_length"],
        },
        "truncated_drug_count": tok_audit["count_exceeding_max"],
        "truncated_drug_ids": tok_audit["truncated_stitch_ids"],
        "software_versions": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": _transformers_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
        "input_paths": {
            "mapping": str(DEFAULT_MAPPING_INPUT),
            "pairs": str(DEFAULT_PAIRS_INPUT),
        },
        "output_paths": {
            "embeddings": str(embed_path),
            "index": str(index_path),
        },
        "artifact_sha256": {
            "embeddings": sha256_file(embed_path),
            "index": sha256_file(index_path),
        },
    }


def _transformers_version() -> str:
    try:
        import transformers as tf
        return tf.__version__
    except ImportError:
        return "unknown"


def _write_atomic(data: bytes, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.parent / (path.name + ".tmp")
    try:
        tmp.write_bytes(data)
        os.replace(str(tmp), str(path))
    except OSError:
        if tmp.exists():
            tmp.unlink(missing_ok=True)
        raise


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def run_embedding_pipeline(
    mapping_input:  Path = DEFAULT_MAPPING_INPUT,
    pairs_input:    Path = DEFAULT_PAIRS_INPUT,
    embed_output:   Path = DEFAULT_EMBED_OUTPUT,
    index_output:   Path = DEFAULT_INDEX_OUTPUT,
    audit_output:   Path = DEFAULT_AUDIT_OUTPUT,
    batch_size:     int  = DEFAULT_BATCH_SIZE,
    overwrite:      bool = False,
) -> dict[str, Any]:
    """
    Full Milestone 4 embedding pipeline.

    Skips generation if valid artifacts already exist and overwrite=False.
    """
    logger.info("=== Milestone 4 ChemBERTa embedding pipeline started ===")

    # 0. Guard: skip if artifacts already valid and overwrite not requested
    if not overwrite and embed_output.exists() and index_output.exists():
        logger.info(
            "Artifacts already exist at %s and %s. "
            "Pass --overwrite to regenerate. Loading existing artifacts.",
            embed_output, index_output,
        )
        matrix   = np.load(embed_output)
        index_df = pd.read_csv(index_output)
        # Accept if matrix rows == index rows (consistent, non-empty)
        if matrix.shape[0] == len(index_df) and len(index_df) > 0:
            logger.info("Existing artifacts valid (shape %s). Skipping generation.", matrix.shape)
            pairs_df = pd.read_parquet(pairs_input)
            return _audit_existing(
                matrix, index_df, pairs_df,
                embed_output, index_output, audit_output,
            )

    # 1. Load inputs
    mapping_df = pd.read_csv(mapping_input)
    pairs_df   = pd.read_parquet(pairs_input)
    logger.info("Loaded mapping: %d rows. Pairs: %d.", len(mapping_df), len(pairs_df))

    # 2. Validate inputs
    validate_mapping_df(mapping_df, pairs_df)
    logger.info("Input validation passed.")

    # 3. Sort by stitch_id (should already be sorted from Milestone 2)
    mapping_df = mapping_df.sort_values("stitch_id").reset_index(drop=True)
    smiles_list = mapping_df[SMILES_COLUMN].tolist()
    stitch_ids  = mapping_df["stitch_id"].tolist()
    logger.info("SMILES column: '%s'. Total drugs: %d.", SMILES_COLUMN, len(smiles_list))

    # 4. Detect device
    torch.manual_seed(RANDOM_SEED)
    cuda_available = torch.cuda.is_available()
    device = torch.device("cuda" if cuda_available else "cpu")
    logger.info("Device: %s. CUDA available: %s.", device, cuda_available)

    # 5. Resolve HuggingFace repository commit SHA before loading
    from transformers import AutoModel, AutoTokenizer  # local import for testability

    resolved_revision = _resolve_repo_sha(MODEL_NAME)
    logger.info("Resolved HF revision SHA: %s", resolved_revision)

    # Load model and tokenizer with the pinned SHA so the loaded artifact is
    # reproducibly tied to a specific commit.
    logger.info("Loading tokenizer and model: %s @ %s ...", MODEL_NAME, resolved_revision)
    load_kwargs: dict[str, Any] = {}
    if resolved_revision:
        load_kwargs["revision"] = resolved_revision

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, **load_kwargs)
    model     = AutoModel.from_pretrained(MODEL_NAME, **load_kwargs)

    # Freeze all parameters: no gradients will be computed
    model.requires_grad_(False)
    model = model.to(device)
    model.eval()

    tokenizer_class = type(tokenizer).__name__
    model_class     = type(model).__name__
    logger.info("Model class: %s. Tokenizer class: %s.", model_class, tokenizer_class)

    # 6. Tokenization audit
    logger.info("Running tokenization audit (no truncation)...")
    tok_audit = tokenization_audit(smiles_list, stitch_ids, tokenizer, MAX_TOKEN_LENGTH)
    logger.info(
        "Token lengths — min: %d, median: %.1f, mean: %.2f, max: %d. "
        "Drugs exceeding %d tokens: %d (%.2f%%).",
        tok_audit["min_token_length"],
        tok_audit["median_token_length"],
        tok_audit["mean_token_length"],
        tok_audit["max_token_length"],
        MAX_TOKEN_LENGTH,
        tok_audit["count_exceeding_max"],
        tok_audit["percent_exceeding_max"],
    )

    # 7. Smoke test on first 5 drugs
    logger.info("Running smoke test on 5 drugs...")
    sample_smiles = smiles_list[:5]
    hidden_dim = smoke_test(sample_smiles, tokenizer, model, device)

    # 8. Embed all drugs
    logger.info("Embedding all %d drugs (batch_size=%d)...", len(smiles_list), batch_size)
    matrix = embed_drugs(smiles_list, tokenizer, model, batch_size, device)

    # 9. Validate in memory before writing
    _validate_matrix_in_memory(matrix, hidden_dim)
    logger.info("In-memory matrix validation passed. Shape: %s.", matrix.shape)

    # 10. Build drug index
    index_df = build_drug_index(
        mapping_df, tok_audit["per_drug_token_lengths"], smiles_list, MAX_TOKEN_LENGTH
    )

    # 11. Write outputs atomically
    for path in [embed_output, index_output, audit_output]:
        path.parent.mkdir(parents=True, exist_ok=True)

    tmp_embed = embed_output.parent / (embed_output.name + ".tmp.npy")
    np.save(tmp_embed, matrix)
    os.replace(str(tmp_embed), str(embed_output))
    logger.info("Saved embedding matrix: %s %s.", embed_output, matrix.shape)

    _write_atomic(
        index_df.to_csv(index=False).encode("utf-8"),
        index_output,
    )
    logger.info("Saved drug index: %s.", index_output)

    # 12. Post-run disk validation
    logger.info("Running post-run disk validation...")
    validate_outputs_from_disk(
        embed_output, index_output, pairs_df, hidden_dim,
        expected_drug_count=len(smiles_list),
    )
    logger.info("Disk validation passed.")

    # 13. Build and write audit
    audit = build_embedding_audit(
        mapping_df, index_df, matrix, tok_audit,
        resolved_revision, str(device), cuda_available, batch_size,
        embed_output, index_output, audit_output,
        tokenizer_class, model_class,
    )
    _write_atomic(
        json.dumps(audit, indent=2, ensure_ascii=False).encode("utf-8"),
        audit_output,
    )
    logger.info("Wrote audit: %s.", audit_output)
    logger.info("=== Milestone 4 complete. Embedding shape: %s ===", matrix.shape)

    return audit


def _resolve_repo_sha(model_name: str) -> str | None:
    """
    Resolve the current HEAD commit SHA for a HuggingFace model repository.

    Uses huggingface_hub (installed as a transitive dependency of transformers).
    Returns None if the lookup fails (e.g. offline mode), in which case the
    caller loads without pinning a revision.
    """
    try:
        from huggingface_hub import model_info as hf_model_info
        info = hf_model_info(model_name)
        sha: str | None = getattr(info, "sha", None)
        return sha
    except Exception as exc:
        logger.warning("Could not resolve HF revision SHA: %s", exc)
        return None


def _validate_matrix_in_memory(matrix: np.ndarray, hidden_dim: int) -> None:
    if matrix.dtype != np.float32:
        raise ValueError(f"Matrix dtype is {matrix.dtype}; expected float32.")
    if matrix.shape[1] != hidden_dim:
        raise ValueError(f"Unexpected hidden dim: {matrix.shape[1]} (expected {hidden_dim}).")
    if not np.isfinite(matrix).all():
        raise ValueError("Matrix contains NaN or Inf values.")
    all_zero = (matrix == 0).all(axis=1)
    if all_zero.any():
        raise ValueError(f"{all_zero.sum()} all-zero rows in embedding matrix.")


def _audit_existing(
    matrix: np.ndarray,
    index_df: pd.DataFrame,
    pairs_df: pd.DataFrame,
    embed_path: Path,
    index_path: Path,
    audit_path: Path,
) -> dict[str, Any]:
    """Minimal audit dict when reusing existing artifacts."""
    return {
        "note": "Existing artifacts loaded without regeneration.",
        "embedding_shape": list(matrix.shape),
        "embedding_dtype": str(matrix.dtype),
        "total_drugs": len(index_df),
        "artifact_sha256": {
            "embeddings": sha256_file(embed_path),
            "index": sha256_file(index_path),
        },
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )

    p = argparse.ArgumentParser(
        description="Milestone 4 — generate ChemBERTa drug embeddings.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--mapping-input",  type=Path, default=DEFAULT_MAPPING_INPUT)
    p.add_argument("--pairs-input",    type=Path, default=DEFAULT_PAIRS_INPUT)
    p.add_argument("--embed-output",   type=Path, default=DEFAULT_EMBED_OUTPUT)
    p.add_argument("--index-output",   type=Path, default=DEFAULT_INDEX_OUTPUT)
    p.add_argument("--audit-output",   type=Path, default=DEFAULT_AUDIT_OUTPUT)
    p.add_argument("--batch-size",     type=int,  default=DEFAULT_BATCH_SIZE)
    p.add_argument("--overwrite",      action="store_true",
                   help="Regenerate even if artifacts already exist.")
    args = p.parse_args(argv)

    run_embedding_pipeline(
        mapping_input=args.mapping_input,
        pairs_input=args.pairs_input,
        embed_output=args.embed_output,
        index_output=args.index_output,
        audit_output=args.audit_output,
        batch_size=args.batch_size,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
