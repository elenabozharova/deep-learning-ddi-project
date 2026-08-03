"""
Milestone 6A — Generate Morgan fingerprint features (conventional molecular baseline).

Drug fingerprints
-----------------
RDKit MorganGenerator, radius=2 (ECFP4-style), 2048 bits, chirality enabled.
One binary fingerprint per unique drug, sorted lexicographically by stitch_id.
Shape: (645, 2048), dtype uint8, values: {0, 1}.

Pair fingerprints
-----------------
pair_fingerprint[i] = fingerprint[drug_1] + fingerprint[drug_2]

Sum preserves the count of drugs that have each bit active (0, 1, or 2).
Commutative — the result is the same regardless of which drug is labelled
drug_1 vs drug_2.  Not L2-normalized.
Shape: (63,472, 2048), dtype uint8, values: {0, 1, 2}.
Row i corresponds to pair_id i.

Run from project root:
    python src/polyllm/features/build_morgan_features.py
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
from rdkit import Chem, RDLogger
from rdkit.Chem import rdFingerprintGenerator

logger = logging.getLogger(__name__)
RDLogger.DisableLog("rdApp.*")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MORGAN_RADIUS         = 2
MORGAN_FP_SIZE        = 2048
MORGAN_CHIRALITY      = True
SMILES_COLUMN         = "standardized_smiles"

DEFAULT_MAPPING_INPUT     = Path("data/processed/drug_smiles_mapping.csv")
DEFAULT_PAIRS_INPUT       = Path("data/processed/polyllm_pairs.parquet")
DEFAULT_DRUG_FP_OUTPUT    = Path("data/features/morgan_drug_fingerprints.npy")
DEFAULT_DRUG_IDX_OUTPUT   = Path("data/features/morgan_drug_index.csv")
DEFAULT_PAIR_FP_OUTPUT    = Path("data/features/morgan_pair_fingerprints.npy")
DEFAULT_PAIR_IDX_OUTPUT   = Path("data/features/morgan_pair_index.csv")
DEFAULT_AUDIT_OUTPUT      = Path("outputs/polyllm/morgan_feature_audit.json")

REQUIRED_MAPPING_COLUMNS = {"stitch_id", "parsed_pubchem_cid", SMILES_COLUMN, "mapping_status"}
REQUIRED_PAIRS_COLUMNS   = {"pair_id", "drug_1", "drug_2"}


# ---------------------------------------------------------------------------
# Step 1: Morgan generator (singleton)
# ---------------------------------------------------------------------------

def make_generator() -> Any:
    """Return the configured Morgan fingerprint generator."""
    return rdFingerprintGenerator.GetMorganGenerator(
        radius=MORGAN_RADIUS,
        fpSize=MORGAN_FP_SIZE,
        includeChirality=MORGAN_CHIRALITY,
    )


# ---------------------------------------------------------------------------
# Step 2: Input validation
# ---------------------------------------------------------------------------

def validate_mapping_df(
    mapping_df: pd.DataFrame,
    pairs_df: pd.DataFrame,
) -> None:
    """
    Raise ValueError if mapping or pairs do not meet preconditions.
    """
    missing = REQUIRED_MAPPING_COLUMNS - set(mapping_df.columns)
    if missing:
        raise ValueError(f"Mapping CSV missing columns: {sorted(missing)}")

    non_mapped = mapping_df[mapping_df["mapping_status"] != "mapped"]
    if len(non_mapped):
        raise ValueError(
            f"{len(non_mapped)} drugs have non-mapped status."
        )

    dup_ids = mapping_df["stitch_id"][mapping_df["stitch_id"].duplicated()].tolist()
    if dup_ids:
        raise ValueError(f"Duplicate stitch_ids in mapping: {dup_ids[:5]}")

    null_smiles = mapping_df[SMILES_COLUMN].isna() | (
        mapping_df[SMILES_COLUMN].astype(str).str.strip() == ""
    )
    if null_smiles.any():
        raise ValueError(f"{null_smiles.sum()} null/empty '{SMILES_COLUMN}' values.")

    missing_pairs_cols = REQUIRED_PAIRS_COLUMNS - set(pairs_df.columns)
    if missing_pairs_cols:
        raise ValueError(f"Pairs table missing columns: {sorted(missing_pairs_cols)}")

    known = set(mapping_df["stitch_id"].tolist())
    all_pair_drugs = (
        set(pairs_df["drug_1"].dropna().tolist())
        | set(pairs_df["drug_2"].dropna().tolist())
    )
    missing_drugs = all_pair_drugs - known
    if missing_drugs:
        raise ValueError(
            f"{len(missing_drugs)} pair-table drugs absent from mapping: "
            f"{sorted(missing_drugs)[:5]}"
        )

    if pairs_df["pair_id"].nunique() != len(pairs_df):
        raise ValueError("Duplicate pair_ids in pairs table.")

    expected = np.arange(len(pairs_df))
    if not np.array_equal(np.sort(pairs_df["pair_id"].to_numpy()), expected):
        raise ValueError("pair_id column is not sequential 0..n-1.")


# ---------------------------------------------------------------------------
# Step 3: Generate per-drug fingerprints
# ---------------------------------------------------------------------------

def generate_drug_fingerprints(
    mapping_df: pd.DataFrame,
    generator: Any,
) -> tuple[np.ndarray, pd.DataFrame]:
    """
    Generate one binary fingerprint per drug.

    Returns:
        drug_matrix : (n_drugs, fp_size) uint8 array with values {0, 1}
        sorted_df   : mapping_df sorted by stitch_id (fingerprint_index = row position)
    """
    sorted_df = mapping_df.sort_values("stitch_id").reset_index(drop=True)
    fps: list[np.ndarray] = []

    for _, row in sorted_df.iterrows():
        mol = Chem.MolFromSmiles(row[SMILES_COLUMN])
        if mol is None:
            raise ValueError(
                f"RDKit could not parse SMILES for {row['stitch_id']}: "
                f"{row[SMILES_COLUMN]!r}"
            )
        fp = generator.GetFingerprintAsNumPy(mol).astype(np.uint8)
        fps.append(fp)

    drug_matrix = np.stack(fps, axis=0)
    return drug_matrix, sorted_df


# ---------------------------------------------------------------------------
# Step 4: Build per-drug index CSV
# ---------------------------------------------------------------------------

def build_drug_index(
    sorted_df: pd.DataFrame,
    drug_matrix: np.ndarray,
) -> pd.DataFrame:
    rows = []
    for i, (_, row) in enumerate(sorted_df.iterrows()):
        smi = row[SMILES_COLUMN]
        sha256 = hashlib.sha256(smi.encode("utf-8")).hexdigest()
        active = int(drug_matrix[i].sum())
        rows.append({
            "fingerprint_index": i,
            "stitch_id": row["stitch_id"],
            "parsed_pubchem_cid": row.get("parsed_pubchem_cid"),
            "smiles_sha256": sha256,
            "active_bit_count": active,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Step 5: Build pair fingerprints
# ---------------------------------------------------------------------------

def build_stitch_lookup(drug_index_df: pd.DataFrame) -> dict[str, int]:
    return dict(
        zip(
            drug_index_df["stitch_id"].tolist(),
            drug_index_df["fingerprint_index"].astype(int).tolist(),
        )
    )


def build_pair_fingerprints(
    pairs_df: pd.DataFrame,
    drug_matrix: np.ndarray,
    stitch_lookup: dict[str, int],
) -> np.ndarray:
    """
    Return (n_pairs, fp_size) uint8 array with values {0, 1, 2}.

    pair_matrix[i] = drug_matrix[drug_1] + drug_matrix[drug_2]

    Uses uint16 arithmetic during addition to prevent any overflow risk,
    then casts back to uint8 (safe since max value is 2).
    """
    pairs_sorted = pairs_df.sort_values("pair_id").reset_index(drop=True)
    d1_idx = pairs_sorted["drug_1"].map(stitch_lookup).to_numpy(dtype=np.intp)
    d2_idx = pairs_sorted["drug_2"].map(stitch_lookup).to_numpy(dtype=np.intp)

    pair_matrix = (
        drug_matrix[d1_idx].astype(np.uint16)
        + drug_matrix[d2_idx].astype(np.uint16)
    ).astype(np.uint8)

    return pair_matrix


# ---------------------------------------------------------------------------
# Step 6: Build pair index CSV
# ---------------------------------------------------------------------------

def build_pair_index(
    pairs_df: pd.DataFrame,
    stitch_lookup: dict[str, int],
) -> pd.DataFrame:
    sorted_df = pairs_df.sort_values("pair_id").reset_index(drop=True)
    return pd.DataFrame({
        "pair_id":                 sorted_df["pair_id"].tolist(),
        "drug_1":                  sorted_df["drug_1"].tolist(),
        "drug_2":                  sorted_df["drug_2"].tolist(),
        "drug_1_fingerprint_index": sorted_df["drug_1"].map(stitch_lookup).tolist(),
        "drug_2_fingerprint_index": sorted_df["drug_2"].map(stitch_lookup).tolist(),
    })


# ---------------------------------------------------------------------------
# Step 7: Symmetry spot-check
# ---------------------------------------------------------------------------

def verify_symmetry(
    pairs_df: pd.DataFrame,
    drug_matrix: np.ndarray,
    stitch_lookup: dict[str, int],
    n_checks: int = 5,
) -> None:
    sample = pairs_df.sample(n=min(n_checks, len(pairs_df)), random_state=42)
    for _, row in sample.iterrows():
        i1 = stitch_lookup[row["drug_1"]]
        i2 = stitch_lookup[row["drug_2"]]
        fwd = drug_matrix[i1].astype(np.uint16) + drug_matrix[i2].astype(np.uint16)
        rev = drug_matrix[i2].astype(np.uint16) + drug_matrix[i1].astype(np.uint16)
        if not np.array_equal(fwd, rev):
            raise RuntimeError(
                f"Symmetry check failed for pair_id {row['pair_id']}."
            )


# ---------------------------------------------------------------------------
# Step 8: In-memory validation
# ---------------------------------------------------------------------------

def validate_drug_matrix(drug_matrix: np.ndarray, n_drugs: int) -> None:
    if drug_matrix.shape != (n_drugs, MORGAN_FP_SIZE):
        raise ValueError(
            f"Drug matrix shape {drug_matrix.shape} != ({n_drugs}, {MORGAN_FP_SIZE})."
        )
    if drug_matrix.dtype != np.uint8:
        raise ValueError(f"Drug matrix dtype {drug_matrix.dtype} != uint8.")
    if not np.all((drug_matrix == 0) | (drug_matrix == 1)):
        raise ValueError("Drug matrix contains values other than 0 or 1.")
    all_zero = (drug_matrix == 0).all(axis=1)
    if all_zero.any():
        raise ValueError(f"{int(all_zero.sum())} all-zero drug fingerprints.")


def validate_pair_matrix(pair_matrix: np.ndarray, n_pairs: int) -> None:
    if pair_matrix.shape != (n_pairs, MORGAN_FP_SIZE):
        raise ValueError(
            f"Pair matrix shape {pair_matrix.shape} != ({n_pairs}, {MORGAN_FP_SIZE})."
        )
    if pair_matrix.dtype != np.uint8:
        raise ValueError(f"Pair matrix dtype {pair_matrix.dtype} != uint8.")
    bad = ~np.isin(pair_matrix, [0, 1, 2])
    if bad.any():
        bad_vals = np.unique(pair_matrix[bad]).tolist()
        raise ValueError(f"Pair matrix contains invalid values: {bad_vals}.")
    all_zero = (pair_matrix == 0).all(axis=1)
    if all_zero.any():
        raise ValueError(f"{int(all_zero.sum())} all-zero pair fingerprints.")


# ---------------------------------------------------------------------------
# Step 9: Post-run disk validation
# ---------------------------------------------------------------------------

def validate_from_disk(
    drug_fp_path: Path,
    drug_idx_path: Path,
    pair_fp_path: Path,
    pair_idx_path: Path,
    pairs_df: pd.DataFrame,
    n_drugs: int = 645,
) -> None:
    """Reload and independently validate all four output files."""
    drug_matrix = np.load(drug_fp_path)
    drug_idx    = pd.read_csv(drug_idx_path)
    pair_matrix = np.load(pair_fp_path)
    pair_idx    = pd.read_csv(pair_idx_path)

    n_pairs = len(pairs_df)

    # Drug fingerprints
    if drug_matrix.shape != (n_drugs, MORGAN_FP_SIZE):
        raise ValueError(f"Disk: drug matrix shape {drug_matrix.shape}.")
    if drug_matrix.dtype != np.uint8:
        raise ValueError(f"Disk: drug dtype {drug_matrix.dtype}.")
    if not np.all((drug_matrix == 0) | (drug_matrix == 1)):
        raise ValueError("Disk: drug matrix has values outside {0,1}.")
    if (drug_matrix == 0).all(axis=1).any():
        raise ValueError("Disk: all-zero drug fingerprint detected.")
    if len(drug_idx) != n_drugs:
        raise ValueError(f"Disk: drug index has {len(drug_idx)} rows; expected {n_drugs}.")
    if list(drug_idx["fingerprint_index"]) != list(range(n_drugs)):
        raise ValueError("Disk: fingerprint_index is not 0..n-1.")
    if drug_idx["stitch_id"].nunique() != n_drugs:
        raise ValueError("Disk: duplicate stitch_ids in drug index.")

    # Pair fingerprints
    if pair_matrix.shape != (n_pairs, MORGAN_FP_SIZE):
        raise ValueError(f"Disk: pair matrix shape {pair_matrix.shape}.")
    if pair_matrix.dtype != np.uint8:
        raise ValueError(f"Disk: pair dtype {pair_matrix.dtype}.")
    if not np.isin(pair_matrix, [0, 1, 2]).all():
        raise ValueError("Disk: pair matrix has values outside {0,1,2}.")
    if (pair_matrix == 0).all(axis=1).any():
        raise ValueError("Disk: all-zero pair fingerprint detected.")
    if len(pair_idx) != n_pairs:
        raise ValueError(f"Disk: pair index has {len(pair_idx)} rows; expected {n_pairs}.")
    if not np.array_equal(pair_idx["pair_id"].to_numpy(), np.arange(n_pairs)):
        raise ValueError("Disk: pair_id column is not 0..n-1.")

    # Row 0 spot-check
    first_row = pairs_df[pairs_df["pair_id"] == 0].iloc[0]
    if pair_idx.iloc[0]["drug_1"] != first_row["drug_1"]:
        raise ValueError("Disk: pair index row 0 drug_1 mismatch.")


# ---------------------------------------------------------------------------
# Step 10: Spot-check pair rows against drug fingerprints
# ---------------------------------------------------------------------------

def spot_check_pairs(
    drug_matrix: np.ndarray,
    pair_matrix: np.ndarray,
    pairs_df: pd.DataFrame,
    stitch_lookup: dict[str, int],
    n_checks: int = 5,
) -> None:
    sample = pairs_df.sample(n=min(n_checks, len(pairs_df)), random_state=7)
    for _, row in sample.iterrows():
        pid = int(row["pair_id"])
        i1  = stitch_lookup[row["drug_1"]]
        i2  = stitch_lookup[row["drug_2"]]
        expected = (
            drug_matrix[i1].astype(np.uint16) + drug_matrix[i2].astype(np.uint16)
        ).astype(np.uint8)
        if not np.array_equal(pair_matrix[pid], expected):
            raise RuntimeError(
                f"Spot-check failed for pair_id {pid}: "
                f"row does not match saved drug fingerprints."
            )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


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


def _save_npy_atomic(array: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.parent / (path.name + ".tmp.npy")
    np.save(tmp, array)
    os.replace(str(tmp), str(path))


# ---------------------------------------------------------------------------
# Step 11: Audit
# ---------------------------------------------------------------------------

def build_audit(
    drug_matrix: np.ndarray,
    pair_matrix: np.ndarray,
    drug_fp_path: Path,
    drug_idx_path: Path,
    pair_fp_path: Path,
    pair_idx_path: Path,
) -> dict[str, Any]:
    active_per_drug = drug_matrix.sum(axis=1)
    unique_vals, counts = np.unique(pair_matrix, return_counts=True)
    pair_val_counts = {int(v): int(c) for v, c in zip(unique_vals, counts)}

    return {
        "fingerprint_type": "Morgan (ECFP4-style)",
        "radius": MORGAN_RADIUS,
        "diameter": MORGAN_RADIUS * 2,
        "fingerprint_size": MORGAN_FP_SIZE,
        "include_chirality": MORGAN_CHIRALITY,
        "drug_count": int(drug_matrix.shape[0]),
        "pair_count": int(pair_matrix.shape[0]),
        "drug_matrix_shape": list(drug_matrix.shape),
        "pair_matrix_shape": list(pair_matrix.shape),
        "drug_matrix_dtype": str(drug_matrix.dtype),
        "pair_matrix_dtype": str(pair_matrix.dtype),
        "drug_active_bit_summary": {
            "min":    int(active_per_drug.min()),
            "median": float(np.median(active_per_drug)),
            "mean":   float(round(active_per_drug.mean(), 4)),
            "max":    int(active_per_drug.max()),
        },
        "pair_value_counts": pair_val_counts,
        "all_zero_drug_count":  int((drug_matrix == 0).all(axis=1).sum()),
        "all_zero_pair_count":  int((pair_matrix == 0).all(axis=1).sum()),
        "software_versions": {
            "python": platform.python_version(),
            "rdkit":  _rdkit_version(),
            "numpy":  np.__version__,
            "pandas": pd.__version__,
        },
        "input_paths": {
            "mapping": str(DEFAULT_MAPPING_INPUT),
            "pairs":   str(DEFAULT_PAIRS_INPUT),
        },
        "output_paths": {
            "drug_fingerprints": str(drug_fp_path),
            "drug_index":        str(drug_idx_path),
            "pair_fingerprints": str(pair_fp_path),
            "pair_index":        str(pair_idx_path),
        },
        "artifact_sha256": {
            "drug_fingerprints": sha256_file(drug_fp_path),
            "drug_index":        sha256_file(drug_idx_path),
            "pair_fingerprints": sha256_file(pair_fp_path),
            "pair_index":        sha256_file(pair_idx_path),
        },
    }


def _rdkit_version() -> str:
    try:
        from rdkit import __version__ as v
        return v
    except Exception:
        return "unknown"


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def run_morgan_pipeline(
    mapping_input:   Path = DEFAULT_MAPPING_INPUT,
    pairs_input:     Path = DEFAULT_PAIRS_INPUT,
    drug_fp_output:  Path = DEFAULT_DRUG_FP_OUTPUT,
    drug_idx_output: Path = DEFAULT_DRUG_IDX_OUTPUT,
    pair_fp_output:  Path = DEFAULT_PAIR_FP_OUTPUT,
    pair_idx_output: Path = DEFAULT_PAIR_IDX_OUTPUT,
    audit_output:    Path = DEFAULT_AUDIT_OUTPUT,
    overwrite:       bool = False,
) -> dict[str, Any]:
    """Full Milestone 6A Morgan fingerprint pipeline."""
    logger.info("=== Milestone 6A Morgan fingerprint pipeline started ===")

    # 0. Guard
    if (
        not overwrite
        and drug_fp_output.exists()
        and drug_idx_output.exists()
        and pair_fp_output.exists()
        and pair_idx_output.exists()
    ):
        dm = np.load(drug_fp_output)
        pm = np.load(pair_fp_output)
        di = pd.read_csv(drug_idx_output)
        pi = pd.read_csv(pair_idx_output)
        if (
            dm.shape[0] == len(di)
            and pm.shape[0] == len(pi)
            and len(di) > 0
            and len(pi) > 0
        ):
            logger.info("Artifacts exist; pass --overwrite to regenerate.")
            return {
                "note": "Existing artifacts loaded without regeneration.",
                "drug_matrix_shape": list(dm.shape),
                "pair_matrix_shape": list(pm.shape),
                "artifact_sha256": {
                    "drug_fingerprints": sha256_file(drug_fp_output),
                    "pair_fingerprints": sha256_file(pair_fp_output),
                },
            }

    # 1. Load
    mapping_df = pd.read_csv(mapping_input)
    pairs_df   = pd.read_parquet(pairs_input)
    logger.info("Loaded mapping (%d rows) and pairs (%d rows).", len(mapping_df), len(pairs_df))

    # 2. Validate inputs
    validate_mapping_df(mapping_df, pairs_df)
    logger.info("Input validation passed.")

    # 3. Create Morgan generator
    generator = make_generator()
    logger.info(
        "Morgan generator: radius=%d, fpSize=%d, chirality=%s.",
        MORGAN_RADIUS, MORGAN_FP_SIZE, MORGAN_CHIRALITY,
    )

    # 4. Generate drug fingerprints
    logger.info("Generating %d drug fingerprints...", len(mapping_df))
    drug_matrix, sorted_df = generate_drug_fingerprints(mapping_df, generator)
    logger.info("Drug matrix: shape=%s dtype=%s.", drug_matrix.shape, drug_matrix.dtype)

    # 5. Validate drug matrix
    validate_drug_matrix(drug_matrix, n_drugs=len(mapping_df))
    logger.info("Drug matrix validation passed.")

    # 6. Build drug index
    drug_idx_df = build_drug_index(sorted_df, drug_matrix)

    # 7. Build stitch lookup
    stitch_lookup = build_stitch_lookup(drug_idx_df)

    # 8. Symmetry check
    logger.info("Running symmetry spot-check on 5 pairs...")
    verify_symmetry(pairs_df, drug_matrix, stitch_lookup, n_checks=5)
    logger.info("Symmetry check passed.")

    # 9. Build pair fingerprints
    logger.info("Building %d pair fingerprints...", len(pairs_df))
    pair_matrix = build_pair_fingerprints(pairs_df, drug_matrix, stitch_lookup)
    logger.info("Pair matrix: shape=%s dtype=%s.", pair_matrix.shape, pair_matrix.dtype)

    # 10. Validate pair matrix
    validate_pair_matrix(pair_matrix, n_pairs=len(pairs_df))
    logger.info("Pair matrix validation passed.")

    # 11. Build pair index
    pair_idx_df = build_pair_index(pairs_df, stitch_lookup)

    # 12. Write all outputs atomically
    for path in [drug_fp_output, drug_idx_output, pair_fp_output, pair_idx_output, audit_output]:
        path.parent.mkdir(parents=True, exist_ok=True)

    _save_npy_atomic(drug_matrix, drug_fp_output)
    logger.info("Saved drug fingerprints: %s.", drug_fp_output)

    _write_atomic(drug_idx_df.to_csv(index=False).encode("utf-8"), drug_idx_output)
    logger.info("Saved drug index: %s.", drug_idx_output)

    _save_npy_atomic(pair_matrix, pair_fp_output)
    logger.info("Saved pair fingerprints: %s.", pair_fp_output)

    _write_atomic(pair_idx_df.to_csv(index=False).encode("utf-8"), pair_idx_output)
    logger.info("Saved pair index: %s.", pair_idx_output)

    # 13. Disk validation
    logger.info("Running post-run disk validation...")
    validate_from_disk(
        drug_fp_output, drug_idx_output,
        pair_fp_output, pair_idx_output,
        pairs_df, n_drugs=len(mapping_df),
    )
    logger.info("Disk validation passed.")

    # 14. Spot-check pair rows against saved drug fingerprints
    logger.info("Running pair spot-checks (5 pairs)...")
    saved_drug_matrix = np.load(drug_fp_output)
    spot_check_pairs(saved_drug_matrix, pair_matrix, pairs_df, stitch_lookup, n_checks=5)
    logger.info("Spot-checks passed.")

    # 15. Build audit
    audit = build_audit(
        drug_matrix, pair_matrix,
        drug_fp_output, drug_idx_output, pair_fp_output, pair_idx_output,
    )
    _write_atomic(json.dumps(audit, indent=2, ensure_ascii=False).encode("utf-8"), audit_output)
    logger.info("Wrote audit: %s.", audit_output)
    logger.info(
        "=== Milestone 6A complete. Drug %s  Pair %s ===",
        drug_matrix.shape, pair_matrix.shape,
    )

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
        description="Milestone 6A — Morgan fingerprint features.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--mapping-input",   type=Path, default=DEFAULT_MAPPING_INPUT)
    p.add_argument("--pairs-input",     type=Path, default=DEFAULT_PAIRS_INPUT)
    p.add_argument("--drug-fp-output",  type=Path, default=DEFAULT_DRUG_FP_OUTPUT)
    p.add_argument("--drug-idx-output", type=Path, default=DEFAULT_DRUG_IDX_OUTPUT)
    p.add_argument("--pair-fp-output",  type=Path, default=DEFAULT_PAIR_FP_OUTPUT)
    p.add_argument("--pair-idx-output", type=Path, default=DEFAULT_PAIR_IDX_OUTPUT)
    p.add_argument("--audit-output",    type=Path, default=DEFAULT_AUDIT_OUTPUT)
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args(argv)

    run_morgan_pipeline(
        mapping_input=args.mapping_input,
        pairs_input=args.pairs_input,
        drug_fp_output=args.drug_fp_output,
        drug_idx_output=args.drug_idx_output,
        pair_fp_output=args.pair_fp_output,
        pair_idx_output=args.pair_idx_output,
        audit_output=args.audit_output,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
