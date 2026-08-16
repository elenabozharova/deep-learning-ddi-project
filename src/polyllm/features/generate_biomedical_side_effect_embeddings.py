"""
Experiment 1 (narrowed scope) — SE1/SE2 biomedical side-effect embeddings.

RQ1: do domain-specific biomedical language models produce more effective
side-effect representations than generic BERT, when the *same* side-effect
name text is used as input?

To isolate the encoder as the only variable, this module reuses the exact
tokenization/pooling/validation procedure Milestone 10b already uses for
SE0 (`generate_side_effect_embeddings.py`) — same masked-mean pooling over
the final hidden layer (special tokens included), same max_length=64, same
963 side-effect names in the same `label_index` order — and only swaps the
pretrained encoder:

    SE1: microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext
         (Gu et al. 2021, "Domain-Specific Language Model Pretraining for
         Biomedical NLP" — the canonical PubMedBERT checkpoint, pretrained
         from scratch on PubMed abstracts + PMC full text.)
    SE2: cambridgeltl/SapBERT-from-PubMedBERT-fulltext
         (Liu et al. 2021, NAACL, "Self-Alignment Pretraining for Biomedical
         Entity Representations" — SapBERT fine-tuned from SE1's checkpoint
         on UMLS synonym pairs.)

Methodological note — pooling deviation, reported per the experiment plan's
instruction to surface this before changing anything:
    SapBERT's own model card recommends taking the [CLS] token of the final
    hidden layer as the entity embedding, NOT mean pooling (verified against
    huggingface.co/cambridgeltl/SapBERT-from-PubMedBERT-fulltext/README.md).
    We deliberately do NOT follow that recommendation here: this experiment's
    controlled variable is the *encoder*, so pooling is held fixed (mean,
    attention-mask-aware, final layer, special tokens included) across SE0/
    SE1/SE2 — using CLS pooling only for SE2 would confound encoder choice
    with pooling choice. A CLS-pooled SapBERT variant is noted as future work
    in the research note, not run here.

Run from the project root:
    python src/polyllm/features/generate_biomedical_side_effect_embeddings.py
    python src/polyllm/features/generate_biomedical_side_effect_embeddings.py --model sapbert
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

from polyllm.features.generate_side_effect_embeddings import (
    DEFAULT_MAPPING_INPUT,
    MAX_TOKEN_LENGTH,
    NAME_COLUMN,
    RANDOM_SEED,
    build_side_effect_index,
    embed_side_effects,
    smoke_test,
    tokenization_audit,
    validate_mapping_df,
    validate_outputs_from_disk,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Model registry — canonical checkpoints only, verified live against the
# HuggingFace Hub (config.json + model card) before being pinned here.
# ---------------------------------------------------------------------------

MODEL_REGISTRY: dict[str, dict[str, str]] = {
    "pubmedbert": {
        "model_name": "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext",
        "source_paper": "Gu et al. 2021, 'Domain-Specific Language Model Pretraining for Biomedical NLP'",
        "pooling": "mean of non-padding token embeddings (attention-mask-aware, final hidden layer) — identical to SE0",
    },
    "sapbert": {
        "model_name": "cambridgeltl/SapBERT-from-PubMedBERT-fulltext",
        "source_paper": "Liu et al. 2021 (NAACL), 'Self-Alignment Pretraining for Biomedical Entity Representations'",
        "pooling": (
            "mean of non-padding token embeddings (attention-mask-aware, final hidden layer) — "
            "identical to SE0/SE1, deliberately NOT the model card's own recommended [CLS] pooling "
            "(see module docstring)"
        ),
    },
}

DEFAULT_BATCH_SIZE = 32

DEFAULT_FEATURES_DIR = Path("data/features")
DEFAULT_AUDIT_DIR = Path("outputs/polyllm")


# ---------------------------------------------------------------------------
# Provenance helpers (kept local per this repo's per-module convention —
# see e.g. enrich_side_effects_umls.py, map_drugs_to_smiles.py)
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


def _resolve_repo_sha(model_name: str) -> str | None:
    try:
        from huggingface_hub import model_info as hf_model_info
        info = hf_model_info(model_name)
        return getattr(info, "sha", None)
    except Exception as exc:
        logger.warning("Could not resolve HF revision SHA for %s: %s", model_name, exc)
        return None


def _transformers_version() -> str:
    try:
        import transformers as tf
        return tf.__version__
    except ImportError:
        return "unknown"


# ---------------------------------------------------------------------------
# Step 1: Diagnostics (Step 4 of the experiment plan) — sanity checks only
# ---------------------------------------------------------------------------

def compute_embedding_diagnostics(matrix: np.ndarray) -> dict[str, Any]:
    """
    Basic sanity-check diagnostics for a generated embedding matrix.
    Not intended to be over-interpreted — shape/NaN/Inf checks matter more.
    """
    norms = np.linalg.norm(matrix, axis=1)
    normed = matrix / np.clip(norms, 1e-12, None)[:, None]
    sim = normed @ normed.T
    n = sim.shape[0]
    off_diag = sim[~np.eye(n, dtype=bool)]

    _, unique_counts = np.unique(matrix, axis=0, return_counts=True)
    duplicate_row_count = int((unique_counts > 1).sum())

    return {
        "embedding_norm_mean": float(norms.mean()),
        "embedding_norm_std": float(norms.std()),
        "pairwise_cosine_similarity_mean": float(off_diag.mean()),
        "pairwise_cosine_similarity_std": float(off_diag.std()),
        "duplicate_row_groups": duplicate_row_count,
        "zero_vector_count": int((norms == 0).sum()),
    }


# ---------------------------------------------------------------------------
# Orchestration — one model at a time, mirroring Milestone 10b's pipeline
# ---------------------------------------------------------------------------

def run_embedding_pipeline(
    model_key: str,
    mapping_input: Path = DEFAULT_MAPPING_INPUT,
    features_dir: Path = DEFAULT_FEATURES_DIR,
    audit_dir: Path = DEFAULT_AUDIT_DIR,
    batch_size: int = DEFAULT_BATCH_SIZE,
    overwrite: bool = False,
) -> dict[str, Any]:
    if model_key not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model_key {model_key!r}; choices: {sorted(MODEL_REGISTRY)}")

    spec = MODEL_REGISTRY[model_key]
    model_name = spec["model_name"]

    embed_output = features_dir / f"{model_key}_side_effect_embeddings.npy"
    index_output = features_dir / f"{model_key}_side_effect_index.csv"
    audit_output = audit_dir / f"{model_key}_side_effect_embedding_audit.json"

    logger.info("=== Experiment 1 — %s (%s) side-effect embedding pipeline started ===", model_key, model_name)

    if not overwrite and embed_output.exists() and index_output.exists():
        matrix = np.load(embed_output)
        index_df = pd.read_csv(index_output)
        if matrix.shape[0] == len(index_df) and len(index_df) > 0:
            logger.info("Existing artifacts valid (shape %s). Skipping generation.", matrix.shape)
            return {"note": "existing artifacts reused", "embedding_shape": list(matrix.shape)}

    mapping_df = pd.read_csv(mapping_input)
    validate_mapping_df(mapping_df)
    mapping_df = mapping_df.sort_values("label_index").reset_index(drop=True)
    names = mapping_df[NAME_COLUMN].tolist()
    side_effect_ids = mapping_df["side_effect_id"].tolist()
    logger.info("Total side effects: %d.", len(names))

    torch.manual_seed(RANDOM_SEED)
    cuda_available = torch.cuda.is_available()
    device = torch.device("cuda" if cuda_available else "cpu")
    logger.info("Device: %s.", device)

    from transformers import AutoModel, AutoTokenizer

    resolved_revision = _resolve_repo_sha(model_name)
    load_kwargs: dict[str, Any] = {"revision": resolved_revision} if resolved_revision else {}

    logger.info("Loading tokenizer and model: %s @ %s ...", model_name, resolved_revision)
    tokenizer = AutoTokenizer.from_pretrained(model_name, **load_kwargs)
    model = AutoModel.from_pretrained(model_name, **load_kwargs)
    model.requires_grad_(False)
    model = model.to(device)
    model.eval()

    tok_audit = tokenization_audit(names, side_effect_ids, tokenizer, MAX_TOKEN_LENGTH)
    logger.info(
        "Token lengths — min: %d, median: %.1f, mean: %.2f, max: %d. Exceeding %d: %d (%.2f%%).",
        tok_audit["min_token_length"], tok_audit["median_token_length"], tok_audit["mean_token_length"],
        tok_audit["max_token_length"], MAX_TOKEN_LENGTH,
        tok_audit["count_exceeding_max"], tok_audit["percent_exceeding_max"],
    )

    hidden_dim = smoke_test(names[:5], tokenizer, model, device)
    logger.info("Confirmed hidden dimension from live model: %d.", hidden_dim)

    matrix = embed_side_effects(names, tokenizer, model, batch_size, device)

    if matrix.dtype != np.float32 or not np.isfinite(matrix).all():
        raise ValueError("Embedding matrix failed dtype/finiteness check.")

    index_df = build_side_effect_index(mapping_df, tok_audit["per_side_effect_token_lengths"], names, MAX_TOKEN_LENGTH)

    for path in [embed_output, index_output, audit_output]:
        path.parent.mkdir(parents=True, exist_ok=True)

    tmp_embed = embed_output.parent / (embed_output.name + ".tmp.npy")
    np.save(tmp_embed, matrix)
    os.replace(str(tmp_embed), str(embed_output))
    logger.info("Saved embedding matrix: %s %s.", embed_output, matrix.shape)

    _write_atomic(index_df.to_csv(index=False).encode("utf-8"), index_output)
    logger.info("Saved side-effect index: %s.", index_output)

    validate_outputs_from_disk(embed_output, index_output, hidden_dim, expected_side_effect_count=len(names))
    logger.info("Disk validation passed.")

    diagnostics = compute_embedding_diagnostics(matrix)

    audit = {
        "model_key": model_key,
        "model_name": model_name,
        "source_paper": spec["source_paper"],
        "resolved_model_revision": resolved_revision,
        "tokenizer_class": type(tokenizer).__name__,
        "model_class": type(model).__name__,
        "source_name_column": NAME_COLUMN,
        "maximum_token_length": MAX_TOKEN_LENGTH,
        "pooling_method": spec["pooling"],
        "special_tokens_included_in_pooling": True,
        "batch_size": batch_size,
        "device": str(device),
        "cuda_available": cuda_available,
        "random_seed": RANDOM_SEED,
        "total_side_effects": int(len(index_df)),
        "embedding_dimension": int(matrix.shape[1]),
        "embedding_shape": list(matrix.shape),
        "embedding_dtype": str(matrix.dtype),
        "finite_value_check": bool(np.isfinite(matrix).all()),
        "diagnostics": diagnostics,
        "token_length_summary": {
            "min": tok_audit["min_token_length"], "median": tok_audit["median_token_length"],
            "mean": tok_audit["mean_token_length"], "max": tok_audit["max_token_length"],
        },
        "truncated_side_effect_count": tok_audit["count_exceeding_max"],
        "software_versions": {
            "python": platform.python_version(), "torch": torch.__version__,
            "transformers": _transformers_version(), "numpy": np.__version__, "pandas": pd.__version__,
        },
        "output_paths": {"embeddings": str(embed_output), "index": str(index_output)},
        "artifact_sha256": {"embeddings": sha256_file(embed_output), "index": sha256_file(index_output)},
    }
    _write_atomic(json.dumps(audit, indent=2, ensure_ascii=False).encode("utf-8"), audit_output)
    logger.info("Wrote audit: %s.", audit_output)
    logger.info("=== %s pipeline complete. Shape: %s ===", model_key, matrix.shape)

    return audit


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s — %(message)s", datefmt="%H:%M:%S")

    p = argparse.ArgumentParser(description="Experiment 1 — SE1/SE2 biomedical side-effect embeddings.")
    p.add_argument("--model", choices=sorted(MODEL_REGISTRY) + ["all"], default="all")
    p.add_argument("--mapping-input", type=Path, default=DEFAULT_MAPPING_INPUT)
    p.add_argument("--features-dir", type=Path, default=DEFAULT_FEATURES_DIR)
    p.add_argument("--audit-dir", type=Path, default=DEFAULT_AUDIT_DIR)
    p.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args(argv)

    keys = sorted(MODEL_REGISTRY) if args.model == "all" else [args.model]
    for key in keys:
        run_embedding_pipeline(
            key, mapping_input=args.mapping_input, features_dir=args.features_dir,
            audit_dir=args.audit_dir, batch_size=args.batch_size, overwrite=args.overwrite,
        )


if __name__ == "__main__":
    main()
