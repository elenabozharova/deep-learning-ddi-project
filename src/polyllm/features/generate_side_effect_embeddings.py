"""
Milestone 10b — Generate and cache one frozen BERT embedding per side effect.

Protocol (traced from the PolyLLM authors' actual source, `src/embed/Embedding.py`
in github.com/sadrahkm/PolyLLM — `Embedding().get_embeddings('bert', unique_ses)`)
--------
Model  : bert-base-uncased  (frozen, eval-mode)
Input  : side-effect names from Milestone 1 (`polyllm_label_mapping.csv`)
Pooling: attention-mask-aware mean pooling over the final hidden layer
         (all tokens where attention_mask == 1, including special tokens) —
         same "last_avg" pooling the authors use, mirrored here exactly as
         Milestone 4 already mirrors it for ChemBERTa drug embeddings.
Output : (963, 768) float32 NumPy array + side-effect index CSV

These are the 'seffect' node features for the GNN's bipartite graph; the
'pdrugs' node features reuse Milestone 5's `chemberta_pair_embeddings.npy`
directly (no new drug-side embedding work needed).

Run from the project root:
    python src/polyllm/features/generate_side_effect_embeddings.py
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

MODEL_NAME = "bert-base-uncased"
MAX_TOKEN_LENGTH = 64
DEFAULT_BATCH_SIZE = 32
RANDOM_SEED = 42
NAME_COLUMN = "side_effect_name"

DEFAULT_MAPPING_INPUT = Path("data/processed/polyllm_label_mapping.csv")
DEFAULT_EMBED_OUTPUT  = Path("data/features/bert_side_effect_embeddings.npy")
DEFAULT_INDEX_OUTPUT  = Path("data/features/bert_side_effect_index.csv")
DEFAULT_AUDIT_OUTPUT  = Path("outputs/polyllm/side_effect_embedding_audit.json")

REQUIRED_INDEX_COLUMNS = [
    "embedding_index",
    "label_index",
    "side_effect_id",
    "side_effect_name_sha256",
    "token_length",
    "was_truncated",
]


# ---------------------------------------------------------------------------
# Step 1: Input validation
# ---------------------------------------------------------------------------

def validate_mapping_df(mapping_df: pd.DataFrame) -> None:
    """
    Raise ValueError if the label mapping CSV does not meet preconditions.

    Checks:
    - Required columns present
    - label_index is exactly 0..n-1 in order (matches Milestone 1's contract)
    - No duplicate side_effect_id
    - No null/empty side_effect_name
    """
    required_cols = {"label_index", "side_effect_id", NAME_COLUMN}
    missing = required_cols - set(mapping_df.columns)
    if missing:
        raise ValueError(f"Label mapping CSV missing required columns: {sorted(missing)}")

    expected_indices = np.arange(len(mapping_df))
    if not np.array_equal(mapping_df["label_index"].to_numpy(), expected_indices):
        raise ValueError("label_index is not exactly 0..n-1 in order.")

    dup_ids = mapping_df["side_effect_id"][mapping_df["side_effect_id"].duplicated()].tolist()
    if dup_ids:
        raise ValueError(f"Duplicate side_effect_ids in mapping: {dup_ids[:5]}")

    null_names = mapping_df[NAME_COLUMN].isna() | (
        mapping_df[NAME_COLUMN].astype(str).str.strip() == ""
    )
    if null_names.any():
        raise ValueError(f"{null_names.sum()} rows have null/empty '{NAME_COLUMN}'.")


# ---------------------------------------------------------------------------
# Step 2: Tokenization audit
# ---------------------------------------------------------------------------

def compute_token_lengths(
    names: list[str],
    tokenizer: Any,
    max_length: int = MAX_TOKEN_LENGTH,
) -> list[int]:
    """
    Return the natural token length (with special tokens, without truncation)
    for each side-effect name.
    """
    lengths: list[int] = []
    for name in names:
        enc = tokenizer(
            name,
            add_special_tokens=True,
            truncation=False,
            return_tensors=None,
        )
        lengths.append(len(enc["input_ids"]))
    return lengths


def tokenization_audit(
    names: list[str],
    side_effect_ids: list[str],
    tokenizer: Any,
    max_length: int = MAX_TOKEN_LENGTH,
) -> dict[str, Any]:
    """Compute token-length statistics for the side-effect-name corpus."""
    lengths = compute_token_lengths(names, tokenizer, max_length)
    arr = np.array(lengths)
    truncated_mask = arr > max_length
    truncated_ids = [sid for sid, t in zip(side_effect_ids, truncated_mask) if t]

    return {
        "min_token_length": int(arr.min()),
        "median_token_length": float(np.median(arr)),
        "mean_token_length": float(round(arr.mean(), 4)),
        "max_token_length": int(arr.max()),
        "count_exceeding_max": int(truncated_mask.sum()),
        "percent_exceeding_max": float(round(100 * truncated_mask.mean(), 4)),
        "truncated_side_effect_ids": truncated_ids,
        "per_side_effect_token_lengths": lengths,
    }


# ---------------------------------------------------------------------------
# Step 3: Mean pooling (attention-mask-aware, final hidden layer — the
# authors' "last_avg" pooling strategy)
# ---------------------------------------------------------------------------

def mean_pool(
    hidden_state: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """
    Average the final hidden-layer embeddings of all non-padding tokens.

        mask   = attention_mask.unsqueeze(-1).to(hidden_state.dtype)
        pooled = (hidden_state * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)

    Special tokens (CLS, SEP) are included because their attention_mask is 1 —
    same convention as Milestone 4's ChemBERTa pooling and the authors' own
    `get_huggingface_models` masked-mean implementation.
    """
    mask   = attention_mask.unsqueeze(-1).to(hidden_state.dtype)
    pooled = (hidden_state * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
    return pooled


# ---------------------------------------------------------------------------
# Step 4: Smoke test
# ---------------------------------------------------------------------------

def smoke_test(
    name_sample: list[str],
    tokenizer: Any,
    model: Any,
    device: torch.device,
) -> int:
    """
    Embed 3-5 side-effect names and verify correctness.

    Returns the discovered hidden dimension. Raises RuntimeError on failure.
    """
    assert 3 <= len(name_sample) <= 5, "Smoke test requires 3-5 samples."

    results = []
    for _ in range(2):
        enc = tokenizer(
            name_sample,
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

    if results[0].shape[0] != len(name_sample):
        raise RuntimeError(
            f"Pooling returned {results[0].shape[0]} vectors for {len(name_sample)} inputs."
        )

    for r in results:
        if not torch.isfinite(r).all():
            raise RuntimeError("Smoke test: NaN or Inf in embeddings.")

    if not torch.allclose(results[0], results[1], atol=1e-5, rtol=1e-5):
        max_diff = (results[0] - results[1]).abs().max().item()
        raise RuntimeError(
            f"Smoke test: repeated inference differs by {max_diff:.2e} (threshold 1e-5)."
        )

    for p in model.parameters():
        if p.requires_grad:
            raise RuntimeError("Smoke test: model parameter found with requires_grad=True.")

    hidden_dim = results[0].shape[1]
    logger.info("Smoke test passed. Hidden dim = %d.", hidden_dim)
    return hidden_dim


# ---------------------------------------------------------------------------
# Step 5: Full embedding generation
# ---------------------------------------------------------------------------

def embed_side_effects(
    names: list[str],
    tokenizer: Any,
    model: Any,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    """
    Embed all side-effect names in deterministic batch order.

    Returns a float32 NumPy array of shape (n_side_effects, hidden_dim).
    """
    all_vecs: list[np.ndarray] = []
    n = len(names)
    total_batches = (n + batch_size - 1) // batch_size

    for batch_idx, start in enumerate(range(0, n, batch_size), 1):
        batch = names[start : start + batch_size]
        logger.info("Embedding batch %d/%d (%d side effects).", batch_idx, total_batches, len(batch))

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
# Step 6: Build side-effect index
# ---------------------------------------------------------------------------

def build_side_effect_index(
    mapping_df: pd.DataFrame,
    token_lengths: list[int],
    names: list[str],
    max_length: int = MAX_TOKEN_LENGTH,
) -> pd.DataFrame:
    """Build the side-effect index CSV as a DataFrame."""
    rows = []
    for i, (_, row) in enumerate(mapping_df.iterrows()):
        name = names[i]
        sha256 = hashlib.sha256(name.encode("utf-8")).hexdigest()
        tlen = token_lengths[i]
        rows.append({
            "embedding_index": i,
            "label_index": int(row["label_index"]),
            "side_effect_id": row["side_effect_id"],
            "side_effect_name_sha256": sha256,
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
    hidden_dim: int,
    expected_side_effect_count: int,
) -> None:
    """
    Reload the saved files and re-verify all postconditions.

    Raises ValueError on any failure.
    """
    matrix = np.load(embed_path)
    index_df = pd.read_csv(index_path)

    if len(index_df) != expected_side_effect_count:
        raise ValueError(
            f"Index CSV has {len(index_df)} rows; expected {expected_side_effect_count}."
        )

    expected_indices = np.arange(expected_side_effect_count)
    if not np.array_equal(index_df["embedding_index"].to_numpy(), expected_indices):
        raise ValueError(f"embedding_index is not exactly 0..{expected_side_effect_count - 1}.")

    if not np.array_equal(index_df["label_index"].to_numpy(), expected_indices):
        raise ValueError(f"label_index is not exactly 0..{expected_side_effect_count - 1}.")

    if index_df["side_effect_id"].nunique() != expected_side_effect_count:
        raise ValueError("side_effect_ids in index are not all unique.")

    if matrix.shape[0] != len(index_df):
        raise ValueError(f"Matrix has {matrix.shape[0]} rows; index has {len(index_df)} rows.")

    if matrix.shape[1] != hidden_dim:
        raise ValueError(f"Matrix has {matrix.shape[1]} columns; expected {hidden_dim}.")

    if matrix.dtype != np.float32:
        raise ValueError(f"Matrix dtype is {matrix.dtype}; expected float32.")

    if not np.isfinite(matrix).all():
        n_bad = (~np.isfinite(matrix)).sum()
        raise ValueError(f"Matrix contains {n_bad} non-finite values.")

    all_zero = (matrix == 0).all(axis=1)
    if all_zero.any():
        raise ValueError(f"{all_zero.sum()} all-zero rows detected in embedding matrix.")

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
    index_df: pd.DataFrame,
    matrix: np.ndarray,
    tok_audit: dict[str, Any],
    resolved_revision: str | None,
    device_str: str,
    cuda_available: bool,
    batch_size: int,
    embed_path: Path,
    index_path: Path,
    tokenizer_class: str,
    model_class: str,
) -> dict[str, Any]:
    all_zero_count = int((matrix == 0).all(axis=1).sum())

    return {
        "model_name": MODEL_NAME,
        "resolved_model_revision": resolved_revision,
        "tokenizer_class": tokenizer_class,
        "model_class": model_class,
        "source_name_column": NAME_COLUMN,
        "maximum_token_length": MAX_TOKEN_LENGTH,
        "pooling_method": "mean of non-padding token embeddings (attention-mask-aware, final hidden layer)",
        "special_tokens_included_in_pooling": True,
        "provenance": (
            "Traced from PolyLLM authors' src/embed/Embedding.py: "
            "Embedding().get_embeddings('bert', unique_ses) — bert-base-uncased, "
            "encode_plus(max_length=64, padding='max_length'), masked mean over final hidden layer."
        ),
        "batch_size": batch_size,
        "device": device_str,
        "cuda_available": cuda_available,
        "random_seed": RANDOM_SEED,
        "total_side_effects": int(len(index_df)),
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
        "truncated_side_effect_count": tok_audit["count_exceeding_max"],
        "truncated_side_effect_ids": tok_audit["truncated_side_effect_ids"],
        "software_versions": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": _transformers_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
        "input_paths": {
            "mapping": str(DEFAULT_MAPPING_INPUT),
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
    embed_output:   Path = DEFAULT_EMBED_OUTPUT,
    index_output:   Path = DEFAULT_INDEX_OUTPUT,
    audit_output:   Path = DEFAULT_AUDIT_OUTPUT,
    batch_size:     int  = DEFAULT_BATCH_SIZE,
    overwrite:      bool = False,
) -> dict[str, Any]:
    """
    Full side-effect BERT embedding pipeline.

    Skips generation if valid artifacts already exist and overwrite=False.
    """
    logger.info("=== Milestone 10b side-effect BERT embedding pipeline started ===")

    # 0. Guard: skip if artifacts already valid and overwrite not requested
    if not overwrite and embed_output.exists() and index_output.exists():
        logger.info(
            "Artifacts already exist at %s and %s. "
            "Pass --overwrite to regenerate. Loading existing artifacts.",
            embed_output, index_output,
        )
        matrix   = np.load(embed_output)
        index_df = pd.read_csv(index_output)
        if matrix.shape[0] == len(index_df) and len(index_df) > 0:
            logger.info("Existing artifacts valid (shape %s). Skipping generation.", matrix.shape)
            return _audit_existing(matrix, index_df, embed_output, index_output)

    # 1. Load inputs
    mapping_df = pd.read_csv(mapping_input)
    logger.info("Loaded label mapping: %d rows.", len(mapping_df))

    # 2. Validate inputs
    validate_mapping_df(mapping_df)
    logger.info("Input validation passed.")

    # 3. Already ordered by label_index (Milestone 1's contract, checked above)
    mapping_df = mapping_df.sort_values("label_index").reset_index(drop=True)
    names = mapping_df[NAME_COLUMN].tolist()
    side_effect_ids = mapping_df["side_effect_id"].tolist()
    logger.info("Name column: '%s'. Total side effects: %d.", NAME_COLUMN, len(names))

    # 4. Detect device
    torch.manual_seed(RANDOM_SEED)
    cuda_available = torch.cuda.is_available()
    device = torch.device("cuda" if cuda_available else "cpu")
    logger.info("Device: %s. CUDA available: %s.", device, cuda_available)

    # 5. Resolve HuggingFace repository commit SHA before loading
    from transformers import AutoModel, AutoTokenizer  # local import for testability

    resolved_revision = _resolve_repo_sha(MODEL_NAME)
    logger.info("Resolved HF revision SHA: %s", resolved_revision)

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
    tok_audit = tokenization_audit(names, side_effect_ids, tokenizer, MAX_TOKEN_LENGTH)
    logger.info(
        "Token lengths — min: %d, median: %.1f, mean: %.2f, max: %d. "
        "Side effects exceeding %d tokens: %d (%.2f%%).",
        tok_audit["min_token_length"],
        tok_audit["median_token_length"],
        tok_audit["mean_token_length"],
        tok_audit["max_token_length"],
        MAX_TOKEN_LENGTH,
        tok_audit["count_exceeding_max"],
        tok_audit["percent_exceeding_max"],
    )

    # 7. Smoke test on first 5 side effects
    logger.info("Running smoke test on 5 side effects...")
    sample_names = names[:5]
    hidden_dim = smoke_test(sample_names, tokenizer, model, device)

    # 8. Embed all side effects
    logger.info("Embedding all %d side effects (batch_size=%d)...", len(names), batch_size)
    matrix = embed_side_effects(names, tokenizer, model, batch_size, device)

    # 9. Validate in memory before writing
    _validate_matrix_in_memory(matrix, hidden_dim)
    logger.info("In-memory matrix validation passed. Shape: %s.", matrix.shape)

    # 10. Build side-effect index
    index_df = build_side_effect_index(
        mapping_df, tok_audit["per_side_effect_token_lengths"], names, MAX_TOKEN_LENGTH
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
    logger.info("Saved side-effect index: %s.", index_output)

    # 12. Post-run disk validation
    logger.info("Running post-run disk validation...")
    validate_outputs_from_disk(
        embed_output, index_output, hidden_dim,
        expected_side_effect_count=len(names),
    )
    logger.info("Disk validation passed.")

    # 13. Build and write audit
    audit = build_embedding_audit(
        index_df, matrix, tok_audit,
        resolved_revision, str(device), cuda_available, batch_size,
        embed_output, index_output,
        tokenizer_class, model_class,
    )
    _write_atomic(
        json.dumps(audit, indent=2, ensure_ascii=False).encode("utf-8"),
        audit_output,
    )
    logger.info("Wrote audit: %s.", audit_output)
    logger.info("=== Milestone 10b complete. Embedding shape: %s ===", matrix.shape)

    return audit


def _resolve_repo_sha(model_name: str) -> str | None:
    """Resolve the current HEAD commit SHA for a HuggingFace model repository."""
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
    embed_path: Path,
    index_path: Path,
) -> dict[str, Any]:
    """Minimal audit dict when reusing existing artifacts."""
    return {
        "note": "Existing artifacts loaded without regeneration.",
        "embedding_shape": list(matrix.shape),
        "embedding_dtype": str(matrix.dtype),
        "total_side_effects": len(index_df),
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
        description="Milestone 10b — generate BERT side-effect embeddings.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--mapping-input",  type=Path, default=DEFAULT_MAPPING_INPUT)
    p.add_argument("--embed-output",   type=Path, default=DEFAULT_EMBED_OUTPUT)
    p.add_argument("--index-output",   type=Path, default=DEFAULT_INDEX_OUTPUT)
    p.add_argument("--audit-output",   type=Path, default=DEFAULT_AUDIT_OUTPUT)
    p.add_argument("--batch-size",     type=int,  default=DEFAULT_BATCH_SIZE)
    p.add_argument("--overwrite",      action="store_true",
                   help="Regenerate even if artifacts already exist.")
    args = p.parse_args(argv)

    run_embedding_pipeline(
        mapping_input=args.mapping_input,
        embed_output=args.embed_output,
        index_output=args.index_output,
        audit_output=args.audit_output,
        batch_size=args.batch_size,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
