"""
Joint drug-pair embeddings — feed both drugs' SMILES into frozen ChemBERTa
TOGETHER (one sentence-pair forward pass), instead of encoding them
independently and summing (the official Milestone 5 pipeline,
build_pair_embeddings.py).

Motivation (professor-suggestion #3): the official pair representation
never lets Drug A and Drug B interact inside the encoder -- two independent
forward passes, combined only by addition afterward. This script instead
tokenizes (SMILES_A, SMILES_B) as a standard Hugging Face sentence pair
(RoBERTa-style: <s> A </s></s> B </s>), so self-attention can relate the
two molecules to each other before pooling.

Order-invariance (an explicit requirement of the original suggestion):
plain concatenation is NOT symmetric -- encoding (A,B) and (B,A) differ.
Fixed by encoding both orderings and averaging the two pooled embeddings.

Reuses, unmodified: MODEL_NAME and mean_pool() from
generate_chemberta_embeddings.py (same frozen checkpoint, same
attention-mask-aware mean pooling used for every other embedding in this
project). Output is row-compatible with the existing
data/features/chemberta_pair_index.csv (row i = pair_id i, verified by
build_pair_embeddings.py's own docstring) -- that index file is reused
unchanged, no new index file needed.

Output: data/features/chemberta_joint_pair_embeddings.npy, shape
(63472, 384), float32. Does NOT touch chemberta_pair_embeddings.npy (the
official sum-based embeddings) or any other existing artifact.

Run:
    .venv-polyllm/Scripts/python.exe src/polyllm/features/generate_joint_pair_embeddings.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

_src = Path(__file__).resolve().parents[2]
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

from polyllm.features.generate_chemberta_embeddings import MODEL_NAME, mean_pool  # noqa: E402

PAIRS_PATH = Path("data/processed/polyllm_pairs.parquet")
DRUG_MAP_PATH = Path("data/processed/drug_smiles_mapping.csv")
OUT_PATH = Path("data/features/chemberta_joint_pair_embeddings.npy")

BATCH_SIZE = 32
HIDDEN_DIM = 384
MAX_LENGTH = 128  # two SMILES + separators; official single-SMILES cap is 64


def load_model():
    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return tok, model


@torch.inference_mode()
def encode_batch(tok, model, smiles_a: list[str], smiles_b: list[str]) -> np.ndarray:
    """One sentence-pair forward pass per (a, b) entry. Returns (n, 384)."""
    enc = tok(smiles_a, smiles_b, padding=True, truncation=True, max_length=MAX_LENGTH, return_tensors="pt")
    out = model(**enc)
    pooled = mean_pool(out.last_hidden_state, enc["attention_mask"])
    return pooled.numpy().astype(np.float32)


def main() -> None:
    print(f"Loading {MODEL_NAME} (frozen) ...")
    tok, model = load_model()

    print("Loading pairs and SMILES ...")
    pairs_df = pd.read_parquet(PAIRS_PATH).sort_values("pair_id").reset_index(drop=True)
    assert (pairs_df["pair_id"].to_numpy() == np.arange(len(pairs_df))).all(), \
        "pairs_df is not exactly 0..n-1 in pair_id order -- row-index assumption broken."
    drug_map = pd.read_csv(DRUG_MAP_PATH).set_index("stitch_id")["standardized_smiles"]

    smiles_1 = pairs_df["drug_1"].map(drug_map).tolist()
    smiles_2 = pairs_df["drug_2"].map(drug_map).tolist()
    n_missing = sum(s is None or (isinstance(s, float)) for s in smiles_1 + smiles_2)
    if n_missing:
        raise ValueError(f"{n_missing} drugs have no standardized_smiles -- cannot proceed.")

    n = len(pairs_df)
    print(f"{n} pairs. Encoding both orderings (A,B) and (B,A), batch size {BATCH_SIZE} ...")

    out = np.zeros((n, HIDDEN_DIM), dtype=np.float32)
    t0 = time.time()
    for start in range(0, n, BATCH_SIZE):
        end = min(start + BATCH_SIZE, n)
        a_batch = smiles_1[start:end]
        b_batch = smiles_2[start:end]

        emb_ab = encode_batch(tok, model, a_batch, b_batch)
        emb_ba = encode_batch(tok, model, b_batch, a_batch)
        out[start:end] = (emb_ab + emb_ba) / 2.0

        if (start // BATCH_SIZE) % 100 == 0:
            dt = time.time() - t0
            done = end
            rate = done / max(dt, 1e-9)
            eta = (n - done) / max(rate, 1e-9)
            print(f"  {done}/{n}  ({dt:.0f}s elapsed, {rate:.1f} pairs/s, ETA {eta:.0f}s)")

    # Sanity: order-invariance actually holds (symmetric average, by construction,
    # but confirm the two raw orderings really did differ before averaging --
    # otherwise the "joint" step silently degenerated to something order-blind
    # for a trivial reason, e.g. truncation removing all cross-attention).
    sample_idx = np.random.RandomState(42).choice(n, size=min(50, n), replace=False)
    diffs = []
    for i in sample_idx:
        ab = encode_batch(tok, model, [smiles_1[i]], [smiles_2[i]])[0]
        ba = encode_batch(tok, model, [smiles_2[i]], [smiles_1[i]])[0]
        diffs.append(float(np.abs(ab - ba).mean()))
    print(f"\nSanity check (n=50): mean |embedding(A,B) - embedding(B,A)| before averaging = {np.mean(diffs):.4f} "
          f"(should be > 0 -- confirms the two orderings genuinely differ pre-symmetrization)")

    non_finite = int((~np.isfinite(out)).sum())
    all_zero_rows = int(np.all(out == 0, axis=1).sum())
    if non_finite or all_zero_rows:
        raise ValueError(f"Output validation failed: {non_finite} non-finite values, {all_zero_rows} all-zero rows.")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.save(OUT_PATH, out)
    print(f"\nSaved {out.shape} -> {OUT_PATH}")
    print(f"L2-norm range: {np.linalg.norm(out, axis=1).min():.2f} - {np.linalg.norm(out, axis=1).max():.2f}")


if __name__ == "__main__":
    main()
