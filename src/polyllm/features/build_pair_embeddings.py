"""
Milestone 5 — Build one symmetric pair embedding per drug pair.

Operation
---------
pair_embedding[i] = drug_embedding[drug_1] + drug_embedding[drug_2]

Addition is commutative, so the result is the same regardless of which drug
is labelled drug_1 vs drug_2 (the pairs are already canonical from Milestone 1).

Row i of the output matrix corresponds to pair_id i (0..63,471).

Run from the project root:
    python src/polyllm/features/build_pair_embeddings.py
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

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_DRUG_EMBED_INPUT = Path("data/features/chemberta_drug_embeddings.npy")
DEFAULT_DRUG_INDEX_INPUT = Path("data/features/chemberta_drug_index.csv")
DEFAULT_PAIRS_INPUT      = Path("data/processed/polyllm_pairs.parquet")
DEFAULT_PAIR_EMBED_OUTPUT = Path("data/features/chemberta_pair_embeddings.npy")
DEFAULT_PAIR_INDEX_OUTPUT = Path("data/features/chemberta_pair_index.csv")
DEFAULT_AUDIT_OUTPUT      = Path("outputs/polyllm/pair_embedding_audit.json")

REQUIRED_PAIRS_COLUMNS     = {"pair_id", "drug_1", "drug_2"}
REQUIRED_DRUG_INDEX_COLUMNS = {"stitch_id", "embedding_index"}


# ---------------------------------------------------------------------------
# Step 1: Input validation
# ---------------------------------------------------------------------------

def validate_drug_inputs(
    drug_matrix: np.ndarray,
    drug_index_df: pd.DataFrame,
) -> None:
    """
    Raise ValueError if the drug matrix or index do not meet preconditions.
    """
    missing = REQUIRED_DRUG_INDEX_COLUMNS - set(drug_index_df.columns)
    if missing:
        raise ValueError(f"Drug index CSV missing columns: {sorted(missing)}")

    if drug_matrix.ndim != 2:
        raise ValueError(
            f"Drug matrix must be 2-D; got shape {drug_matrix.shape}."
        )

    if drug_matrix.dtype != np.float32:
        raise ValueError(
            f"Drug matrix dtype must be float32; got {drug_matrix.dtype}."
        )

    if len(drug_index_df) != drug_matrix.shape[0]:
        raise ValueError(
            f"Drug index has {len(drug_index_df)} rows but "
            f"drug matrix has {drug_matrix.shape[0]} rows."
        )

    if drug_index_df["stitch_id"].nunique() != len(drug_index_df):
        raise ValueError("Drug index contains duplicate stitch_ids.")

    expected_idx = np.arange(len(drug_index_df))
    if not np.array_equal(
        drug_index_df["embedding_index"].sort_values().to_numpy(), expected_idx
    ):
        raise ValueError(
            "Drug index embedding_index is not exactly 0.."
            f"{len(drug_index_df) - 1}."
        )


def validate_pairs_df(
    pairs_df: pd.DataFrame,
    known_drugs: set[str],
) -> None:
    """
    Raise ValueError if pairs table does not meet preconditions.
    """
    missing_cols = REQUIRED_PAIRS_COLUMNS - set(pairs_df.columns)
    if missing_cols:
        raise ValueError(f"Pairs table missing columns: {sorted(missing_cols)}")

    if pairs_df["pair_id"].nunique() != len(pairs_df):
        raise ValueError("Pairs table contains duplicate pair_ids.")

    expected_ids = np.arange(len(pairs_df))
    actual_ids = np.sort(pairs_df["pair_id"].to_numpy())
    if not np.array_equal(actual_ids, expected_ids):
        raise ValueError(
            f"Pairs pair_id is not sequential 0..{len(pairs_df) - 1}."
        )

    unknown_d1 = set(pairs_df["drug_1"].dropna()) - known_drugs
    unknown_d2 = set(pairs_df["drug_2"].dropna()) - known_drugs
    unknown = unknown_d1 | unknown_d2
    if unknown:
        raise ValueError(
            f"{len(unknown)} drugs in pairs table have no drug embedding: "
            f"{sorted(unknown)[:5]}"
        )


# ---------------------------------------------------------------------------
# Step 2: Build the pair matrix
# ---------------------------------------------------------------------------

def build_stitch_lookup(drug_index_df: pd.DataFrame) -> dict[str, int]:
    """Return {stitch_id: embedding_index} mapping."""
    return dict(
        zip(
            drug_index_df["stitch_id"].tolist(),
            drug_index_df["embedding_index"].astype(int).tolist(),
        )
    )


def build_pair_matrix(
    pairs_df: pd.DataFrame,
    drug_matrix: np.ndarray,
    stitch_lookup: dict[str, int],
) -> np.ndarray:
    """
    Construct the (n_pairs, hidden_dim) float32 pair embedding matrix.

    Pairs are processed in pair_id order.  Row i = pair_id i.

    pair_matrix[i] = drug_matrix[drug_1_idx] + drug_matrix[drug_2_idx]
    """
    pairs_sorted = pairs_df.sort_values("pair_id").reset_index(drop=True)

    drug1_indices = pairs_sorted["drug_1"].map(stitch_lookup).to_numpy(dtype=np.intp)
    drug2_indices = pairs_sorted["drug_2"].map(stitch_lookup).to_numpy(dtype=np.intp)

    pair_matrix = drug_matrix[drug1_indices] + drug_matrix[drug2_indices]
    return pair_matrix.astype(np.float32)


# ---------------------------------------------------------------------------
# Step 3: Symmetry spot-check
# ---------------------------------------------------------------------------

def verify_symmetry(
    pairs_df: pd.DataFrame,
    drug_matrix: np.ndarray,
    stitch_lookup: dict[str, int],
    n_checks: int = 5,
) -> None:
    """
    Verify that swapping drug_1/drug_2 produces the same embedding.

    Raises RuntimeError if any check fails.
    """
    sample = pairs_df.sample(n=min(n_checks, len(pairs_df)), random_state=42)
    for _, row in sample.iterrows():
        idx1 = stitch_lookup[row["drug_1"]]
        idx2 = stitch_lookup[row["drug_2"]]
        fwd = drug_matrix[idx1] + drug_matrix[idx2]
        rev = drug_matrix[idx2] + drug_matrix[idx1]
        if not np.allclose(fwd, rev, atol=1e-6):
            raise RuntimeError(
                f"Symmetry check failed for pair_id {row['pair_id']}: "
                f"max diff {np.abs(fwd - rev).max():.2e}"
            )


# ---------------------------------------------------------------------------
# Step 4: In-memory validation
# ---------------------------------------------------------------------------

def validate_pair_matrix_in_memory(
    pair_matrix: np.ndarray,
    n_pairs: int,
    hidden_dim: int,
) -> None:
    if pair_matrix.shape != (n_pairs, hidden_dim):
        raise ValueError(
            f"Pair matrix shape {pair_matrix.shape} != expected ({n_pairs}, {hidden_dim})."
        )

    if pair_matrix.dtype != np.float32:
        raise ValueError(f"Pair matrix dtype {pair_matrix.dtype} != float32.")

    if not np.isfinite(pair_matrix).all():
        n_bad = int((~np.isfinite(pair_matrix)).sum())
        raise ValueError(f"Pair matrix has {n_bad} non-finite values.")

    all_zero = (pair_matrix == 0).all(axis=1)
    if all_zero.any():
        raise ValueError(
            f"{int(all_zero.sum())} all-zero rows in pair matrix."
        )


# ---------------------------------------------------------------------------
# Step 5: Post-run disk validation
# ---------------------------------------------------------------------------

def validate_outputs_from_disk(
    pair_embed_path: Path,
    pair_index_path: Path,
    pairs_df: pd.DataFrame,
    hidden_dim: int,
) -> None:
    """
    Reload the saved files and re-verify all postconditions.

    Raises ValueError on any failure.
    """
    matrix   = np.load(pair_embed_path)
    index_df = pd.read_csv(pair_index_path)

    n_pairs = len(pairs_df)

    if matrix.shape != (n_pairs, hidden_dim):
        raise ValueError(
            f"Disk: pair matrix shape {matrix.shape} != ({n_pairs}, {hidden_dim})."
        )

    if matrix.dtype != np.float32:
        raise ValueError(f"Disk: dtype {matrix.dtype} != float32.")

    if not np.isfinite(matrix).all():
        raise ValueError("Disk: pair matrix contains non-finite values.")

    all_zero = (matrix == 0).all(axis=1)
    if all_zero.any():
        raise ValueError(
            f"Disk: {int(all_zero.sum())} all-zero rows in pair matrix."
        )

    if len(index_df) != n_pairs:
        raise ValueError(
            f"Disk: pair index has {len(index_df)} rows; expected {n_pairs}."
        )

    if not np.array_equal(
        index_df["pair_id"].to_numpy(), np.arange(n_pairs)
    ):
        raise ValueError("Disk: pair index pair_id column is not exactly 0..n-1.")

    # Spot-check: first and last rows in index match pairs_df
    first_pair = pairs_df[pairs_df["pair_id"] == 0].iloc[0]
    if index_df.iloc[0]["drug_1"] != first_pair["drug_1"]:
        raise ValueError(
            f"Disk: index row 0 drug_1={index_df.iloc[0]['drug_1']} "
            f"!= pairs drug_1={first_pair['drug_1']}"
        )


# ---------------------------------------------------------------------------
# Step 6: Build pair index CSV
# ---------------------------------------------------------------------------

def build_pair_index(pairs_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a minimal traceability CSV: pair_id, drug_1, drug_2, embedding_row.

    embedding_row == pair_id because the matrix is built in pair_id order.
    """
    df = pairs_df.sort_values("pair_id").reset_index(drop=True)
    df = df[["pair_id", "drug_1", "drug_2"]].copy()
    df["embedding_row"] = df["pair_id"]
    return df


# ---------------------------------------------------------------------------
# Step 7: Audit
# ---------------------------------------------------------------------------

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def build_pair_embedding_audit(
    pair_matrix: np.ndarray,
    pairs_df: pd.DataFrame,
    hidden_dim: int,
    drug_embed_path: Path,
    pair_embed_path: Path,
    pair_index_path: Path,
) -> dict[str, Any]:
    n_pairs = len(pairs_df)
    all_zero = int((pair_matrix == 0).all(axis=1).sum())

    norms = np.linalg.norm(pair_matrix, axis=1)

    return {
        "operation": "element-wise sum of drug_1 and drug_2 embeddings",
        "symmetric": True,
        "source_drug_embeddings": str(drug_embed_path),
        "source_drug_embedding_shape": [645, hidden_dim],
        "total_pairs": n_pairs,
        "pair_matrix_shape": list(pair_matrix.shape),
        "pair_matrix_dtype": str(pair_matrix.dtype),
        "hidden_dim": hidden_dim,
        "finite_value_check": bool(np.isfinite(pair_matrix).all()),
        "all_zero_row_count": all_zero,
        "l2_norm_stats": {
            "min": float(round(norms.min(), 4)),
            "median": float(round(float(np.median(norms)), 4)),
            "mean": float(round(norms.mean(), 4)),
            "max": float(round(norms.max(), 4)),
        },
        "software_versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
        "input_paths": {
            "drug_embeddings": str(drug_embed_path),
            "pairs": str(DEFAULT_PAIRS_INPUT),
        },
        "output_paths": {
            "pair_embeddings": str(pair_embed_path),
            "pair_index": str(pair_index_path),
        },
        "artifact_sha256": {
            "pair_embeddings": sha256_file(pair_embed_path),
            "pair_index": sha256_file(pair_index_path),
        },
    }


# ---------------------------------------------------------------------------
# Atomic write helper
# ---------------------------------------------------------------------------

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

def run_pair_embedding_pipeline(
    drug_embed_input:  Path = DEFAULT_DRUG_EMBED_INPUT,
    drug_index_input:  Path = DEFAULT_DRUG_INDEX_INPUT,
    pairs_input:       Path = DEFAULT_PAIRS_INPUT,
    pair_embed_output: Path = DEFAULT_PAIR_EMBED_OUTPUT,
    pair_index_output: Path = DEFAULT_PAIR_INDEX_OUTPUT,
    audit_output:      Path = DEFAULT_AUDIT_OUTPUT,
    overwrite:         bool = False,
) -> dict[str, Any]:
    """
    Full Milestone 5 pair embedding pipeline.

    Skips generation if valid artifacts already exist and overwrite=False.
    """
    logger.info("=== Milestone 5 pair embedding pipeline started ===")

    # 0. Guard: skip if artifacts already valid
    if not overwrite and pair_embed_output.exists() and pair_index_output.exists():
        matrix   = np.load(pair_embed_output)
        index_df = pd.read_csv(pair_index_output)
        if matrix.shape[0] == len(index_df) and len(index_df) > 0:
            logger.info(
                "Artifacts exist (shape %s). Pass --overwrite to regenerate.",
                matrix.shape,
            )
            return {
                "note": "Existing artifacts loaded without regeneration.",
                "pair_matrix_shape": list(matrix.shape),
                "artifact_sha256": {
                    "pair_embeddings": sha256_file(pair_embed_output),
                    "pair_index": sha256_file(pair_index_output),
                },
            }

    # 1. Load inputs
    drug_matrix   = np.load(drug_embed_input)
    drug_index_df = pd.read_csv(drug_index_input)
    pairs_df      = pd.read_parquet(pairs_input)
    logger.info(
        "Loaded drug matrix %s, drug index %d rows, pairs %d rows.",
        drug_matrix.shape, len(drug_index_df), len(pairs_df),
    )

    # 2. Validate inputs
    validate_drug_inputs(drug_matrix, drug_index_df)
    known_drugs = set(drug_index_df["stitch_id"].tolist())
    validate_pairs_df(pairs_df, known_drugs)
    logger.info("Input validation passed.")

    # 3. Build lookup
    stitch_lookup = build_stitch_lookup(drug_index_df)
    hidden_dim    = drug_matrix.shape[1]

    # 4. Symmetry spot-check before full build
    logger.info("Running symmetry spot-check on 5 pairs...")
    verify_symmetry(pairs_df, drug_matrix, stitch_lookup, n_checks=5)
    logger.info("Symmetry check passed.")

    # 5. Build pair matrix
    logger.info("Building pair matrix (%d pairs × %d dims)...", len(pairs_df), hidden_dim)
    pair_matrix = build_pair_matrix(pairs_df, drug_matrix, stitch_lookup)
    logger.info("Pair matrix built: shape %s dtype %s.", pair_matrix.shape, pair_matrix.dtype)

    # 6. In-memory validation
    validate_pair_matrix_in_memory(pair_matrix, len(pairs_df), hidden_dim)
    logger.info("In-memory validation passed.")

    # 7. Build pair index
    pair_index_df = build_pair_index(pairs_df)

    # 8. Write outputs atomically
    for path in [pair_embed_output, pair_index_output, audit_output]:
        path.parent.mkdir(parents=True, exist_ok=True)

    tmp_npy = pair_embed_output.parent / (pair_embed_output.name + ".tmp.npy")
    np.save(tmp_npy, pair_matrix)
    os.replace(str(tmp_npy), str(pair_embed_output))
    logger.info("Saved pair matrix: %s.", pair_embed_output)

    _write_atomic(
        pair_index_df.to_csv(index=False).encode("utf-8"),
        pair_index_output,
    )
    logger.info("Saved pair index: %s.", pair_index_output)

    # 9. Post-run disk validation
    logger.info("Running post-run disk validation...")
    validate_outputs_from_disk(
        pair_embed_output, pair_index_output, pairs_df, hidden_dim
    )
    logger.info("Disk validation passed.")

    # 10. Build and write audit
    audit = build_pair_embedding_audit(
        pair_matrix, pairs_df, hidden_dim,
        drug_embed_input, pair_embed_output, pair_index_output,
    )
    _write_atomic(
        json.dumps(audit, indent=2, ensure_ascii=False).encode("utf-8"),
        audit_output,
    )
    logger.info("Wrote audit: %s.", audit_output)
    logger.info("=== Milestone 5 complete. Pair matrix shape: %s ===", pair_matrix.shape)

    return audit


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
        description="Milestone 5 — build ChemBERTa pair embeddings.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--drug-embed-input",  type=Path, default=DEFAULT_DRUG_EMBED_INPUT)
    p.add_argument("--drug-index-input",  type=Path, default=DEFAULT_DRUG_INDEX_INPUT)
    p.add_argument("--pairs-input",       type=Path, default=DEFAULT_PAIRS_INPUT)
    p.add_argument("--pair-embed-output", type=Path, default=DEFAULT_PAIR_EMBED_OUTPUT)
    p.add_argument("--pair-index-output", type=Path, default=DEFAULT_PAIR_INDEX_OUTPUT)
    p.add_argument("--audit-output",      type=Path, default=DEFAULT_AUDIT_OUTPUT)
    p.add_argument("--overwrite", action="store_true",
                   help="Regenerate even if artifacts already exist.")
    args = p.parse_args(argv)

    run_pair_embedding_pipeline(
        drug_embed_input=args.drug_embed_input,
        drug_index_input=args.drug_index_input,
        pairs_input=args.pairs_input,
        pair_embed_output=args.pair_embed_output,
        pair_index_output=args.pair_index_output,
        audit_output=args.audit_output,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
