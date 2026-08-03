"""
Milestone 7 — Frozen ChemBERTa pair-embedding MLP: test-set evaluator.

Must be run AFTER train_chemberta_mlp.py has produced:
    outputs/polyllm/chemberta/checkpoints/best_model.pt
    outputs/polyllm/chemberta/validation_threshold.json

Usage:
    .venv-polyllm/Scripts/python.exe src/polyllm/evaluate_chemberta_mlp.py
    .venv-polyllm/Scripts/python.exe src/polyllm/evaluate_chemberta_mlp.py --overwrite

The test split is evaluated exactly once.  Metrics are identical to the Morgan
baseline (Milestone 6B) for a fair comparison.
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

_src = Path(__file__).resolve().parent.parent
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

from polyllm.metrics import (
    apply_threshold,
    macro_auprc,
    macro_auroc,
    macro_f1,
    micro_auprc,
    micro_f1,
    per_label_auprc,
    per_label_auroc,
    per_label_f1,
    sample_mean_ap_at_50,
)
from polyllm.models.mlp import MultilabelMLP
from polyllm.training import (
    PairDataset,
    file_sha256,
    load_checkpoint,
    make_seeded_loader,
    run_val_epoch,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

FEATURES_PATH   = Path("data/features/chemberta_pair_embeddings.npy")
LABELS_PATH     = Path("data/processed/polyllm_labels.npy")
PAIR_INDEX_PATH = Path("data/features/chemberta_pair_index.csv")
LABEL_MAP_PATH  = Path("data/processed/polyllm_label_mapping.csv")
PAIRS_PATH      = Path("data/processed/polyllm_pairs.parquet")
TEST_IDS_PATH   = Path("data/splits/test_pair_ids.csv")

OUTPUT_DIR      = Path("outputs/polyllm/chemberta")
CHECKPOINT_PATH = OUTPUT_DIR / "checkpoints/best_model.pt"
CONFIG_PATH     = OUTPUT_DIR / "training_config.json"
THRESHOLD_PATH  = OUTPUT_DIR / "validation_threshold.json"
TEST_METRICS    = OUTPUT_DIR / "test_metrics.json"
PER_LABEL_CSV   = OUTPUT_DIR / "per_label_metrics.csv"
PREDICTIONS_NPZ = OUTPUT_DIR / "test_predictions.npz"
EXAMPLES_CSV    = OUTPUT_DIR / "example_predictions.csv"


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate ChemBERTa MLP on the test set.")
    p.add_argument("--overwrite", action="store_true",
                   help="Overwrite existing test-set outputs.")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Pre-run checks
# ---------------------------------------------------------------------------

def check_prerequisites(overwrite: bool) -> None:
    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            f"Best checkpoint not found: {CHECKPOINT_PATH}\n"
            "Run train_chemberta_mlp.py first."
        )
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(
            f"Training config not found: {CONFIG_PATH}\n"
            "Run train_chemberta_mlp.py first."
        )
    if not THRESHOLD_PATH.exists():
        raise FileNotFoundError(
            f"Validation threshold not found: {THRESHOLD_PATH}\n"
            "Run train_chemberta_mlp.py first."
        )
    if TEST_METRICS.exists() and not overwrite:
        raise FileExistsError(
            f"Test metrics already exist: {TEST_METRICS}\n"
            "Pass --overwrite to re-evaluate."
        )


# ---------------------------------------------------------------------------
# Checkpoint validation
# ---------------------------------------------------------------------------

def validate_checkpoint_config(ckpt: dict, features: np.ndarray, labels: np.ndarray) -> None:
    ckpt_in  = ckpt.get("input_dim")
    ckpt_out = ckpt.get("output_dim")
    if ckpt_in != features.shape[1]:
        raise ValueError(
            f"Checkpoint input_dim={ckpt_in} does not match "
            f"feature dim={features.shape[1]}"
        )
    if ckpt_out != labels.shape[1]:
        raise ValueError(
            f"Checkpoint output_dim={ckpt_out} does not match "
            f"label dim={labels.shape[1]}"
        )


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    check_prerequisites(args.overwrite)

    print("Loading features and labels ...")
    features = np.load(FEATURES_PATH, mmap_mode="r")
    labels   = np.load(LABELS_PATH,   mmap_mode="r")
    print(f"  features {features.shape}  labels {labels.shape}")

    label_map  = pd.read_csv(LABEL_MAP_PATH).sort_values("label_index").reset_index(drop=True)
    pairs_df   = pd.read_parquet(PAIRS_PATH)
    test_ids   = pd.read_csv(TEST_IDS_PATH)["pair_id"].to_numpy()
    print(f"  test pairs: {len(test_ids)}")

    with open(CONFIG_PATH) as f:
        train_config = json.load(f)
    with open(THRESHOLD_PATH) as f:
        thr_data = json.load(f)

    selected_threshold = float(thr_data["selected_threshold"])
    print(f"  selected_threshold={selected_threshold}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  device: {device}")

    model = MultilabelMLP(
        input_dim=train_config["input_dim"],
        output_dim=train_config["output_dim"],
        dropout=train_config["dropout"],
        negative_slope=train_config.get("leaky_relu_negative_slope", 0.01),
    ).to(device)

    ckpt = load_checkpoint(CHECKPOINT_PATH, model, device=device)
    validate_checkpoint_config(ckpt, features, labels)
    print(f"  checkpoint epoch={ckpt['epoch']}  "
          f"val_macro_auprc={ckpt['val_macro_auprc']:.4f}")

    test_ds     = PairDataset(test_ids, features, labels)
    test_loader = make_seeded_loader(
        test_ds,
        batch_size=train_config["batch_size"],
        shuffle=False,
        seed=train_config["random_seed"],
        num_workers=train_config.get("num_workers", 0),
    )
    criterion = torch.nn.BCEWithLogitsLoss()

    print("\nRunning test-set inference ...")
    _, test_probs, test_labels, test_pids = run_val_epoch(
        model, test_loader, criterion, device
    )
    assert test_probs.shape == (len(test_ids), labels.shape[1])
    assert test_probs.dtype == np.float32
    assert np.all(np.isfinite(test_probs))
    assert np.all((test_probs >= 0.0) & (test_probs <= 1.0))
    assert np.array_equal(test_pids, np.sort(test_pids)), \
        "test predictions are not sorted by pair_id"

    print(f"  probs shape: {test_probs.shape}  dtype: {test_probs.dtype}")

    # -----------------------------------------------------------------------
    # Thresholded predictions
    # -----------------------------------------------------------------------
    test_pred_sel = apply_threshold(test_probs, selected_threshold)
    test_pred_05  = apply_threshold(test_probs, 0.5)

    # -----------------------------------------------------------------------
    # Aggregate metrics
    # -----------------------------------------------------------------------
    print("Computing metrics ...")

    auroc_res = macro_auroc(test_labels, test_probs)
    auprc_res = macro_auprc(test_labels, test_probs)
    mu_auprc  = micro_auprc(test_labels, test_probs)
    mu_f1_sel = micro_f1(test_labels, test_pred_sel)
    ma_f1_sel = macro_f1(test_labels, test_pred_sel)
    mu_f1_05  = micro_f1(test_labels, test_pred_05)
    ma_f1_05  = macro_f1(test_labels, test_pred_05)
    ap50      = sample_mean_ap_at_50(test_labels, test_probs)

    import sklearn
    import torch as _torch

    test_metrics = {
        "macro_auroc":                 auroc_res["macro_auroc"],
        "macro_auprc":                 auprc_res["macro_auprc"],
        "micro_auprc":                 mu_auprc,
        "micro_f1_selected_threshold": mu_f1_sel,
        "macro_f1_selected_threshold": ma_f1_sel["macro_f1"],
        "micro_f1_threshold_0_5":      mu_f1_05,
        "macro_f1_threshold_0_5":      ma_f1_05["macro_f1"],
        "sample_mean_ap_at_50":        ap50,
        "paper_ap50_definition_compatibility": "unresolved",
        "selected_threshold":          selected_threshold,
        "evaluated_pair_count":        int(len(test_ids)),
        "evaluated_label_count":       int(labels.shape[1]),
        "skipped_metric_label_counts": {
            "macro_auroc_skipped": auroc_res["skipped_labels"],
            "macro_auprc_skipped": auprc_res["skipped_labels"],
            "macro_f1_sel_skipped": ma_f1_sel["skipped_labels"],
            "macro_f1_05_skipped":  ma_f1_05["skipped_labels"],
        },
        "best_epoch":  ckpt["epoch"],
        "device":      str(device),
        "software_versions": {
            "python":       platform.python_version(),
            "torch":        _torch.__version__,
            "numpy":        np.__version__,
            "pandas":       pd.__version__,
            "scikit_learn": sklearn.__version__,
        },
    }

    print(f"\n  macro AUROC:            {test_metrics['macro_auroc']:.4f}")
    print(f"  macro AUPRC:            {test_metrics['macro_auprc']:.4f}")
    print(f"  micro AUPRC:            {test_metrics['micro_auprc']:.4f}")
    print(f"  micro F1 (thr={selected_threshold:.2f}):  {test_metrics['micro_f1_selected_threshold']:.4f}")
    print(f"  macro F1 (thr={selected_threshold:.2f}):  {test_metrics['macro_f1_selected_threshold']:.4f}")
    print(f"  micro F1 (thr=0.50):    {test_metrics['micro_f1_threshold_0_5']:.4f}")
    print(f"  macro F1 (thr=0.50):    {test_metrics['macro_f1_threshold_0_5']:.4f}")
    print(f"  sample mean AP@50:      {test_metrics['sample_mean_ap_at_50']:.4f}")

    # -----------------------------------------------------------------------
    # Per-label metrics
    # -----------------------------------------------------------------------
    print("\nComputing per-label metrics ...")
    pl_auroc = per_label_auroc(test_labels, test_probs)
    pl_auprc = per_label_auprc(test_labels, test_probs)
    pl_f1    = per_label_f1(test_labels, test_pred_sel)

    per_label_rows = []
    for j in range(labels.shape[1]):
        row_lm    = label_map.iloc[j]
        pos_count = int(test_labels[:, j].sum())
        per_label_rows.append({
            "label_index":              j,
            "side_effect_id":           row_lm["side_effect_id"],
            "side_effect_name":         row_lm["side_effect_name"],
            "test_positive_count":      pos_count,
            "test_prevalence":          round(pos_count / len(test_ids), 6),
            "auroc":                    float(pl_auroc[j]) if not np.isnan(pl_auroc[j]) else None,
            "auprc":                    float(pl_auprc[j]) if not np.isnan(pl_auprc[j]) else None,
            "f1_at_selected_threshold": float(pl_f1[j])   if not np.isnan(pl_f1[j])   else None,
        })

    per_label_df = pd.DataFrame(per_label_rows)

    # -----------------------------------------------------------------------
    # Save predictions NPZ
    # -----------------------------------------------------------------------
    print("Saving predictions ...")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(
        PREDICTIONS_NPZ,
        pair_ids=test_pids,
        probabilities=test_probs.astype(np.float32),
        selected_threshold=np.float32(selected_threshold),
    )
    pred_sha = file_sha256(PREDICTIONS_NPZ)
    ckpt_sha = file_sha256(CHECKPOINT_PATH)

    test_metrics["checkpoint_sha256"]          = ckpt_sha
    test_metrics["prediction_artifact_sha256"] = pred_sha

    # -----------------------------------------------------------------------
    # Save per-label CSV
    # -----------------------------------------------------------------------
    per_label_df.to_csv(PER_LABEL_CSV, index=False)

    # -----------------------------------------------------------------------
    # Example predictions (first 20 sorted test pair IDs, top-10 per pair)
    # -----------------------------------------------------------------------
    print("Building example predictions ...")
    first20_pids  = sorted(test_pids.tolist())[:20]
    pairs_indexed = pairs_df.set_index("pair_id") if "pair_id" in pairs_df.columns else pairs_df

    example_rows = []
    for pid in first20_pids:
        row_idx = int(np.searchsorted(test_pids, pid))
        probs_i = test_probs[row_idx]
        true_i  = test_labels[row_idx]
        top10   = np.argsort(probs_i)[::-1][:10]

        pair_info = pairs_indexed.loc[pid] if pid in pairs_indexed.index else {}
        drug_1 = str(pair_info.get("drug_1", ""))
        drug_2 = str(pair_info.get("drug_2", ""))

        for rank, lbl_idx in enumerate(top10, start=1):
            lm_row = label_map.iloc[lbl_idx]
            example_rows.append({
                "pair_id":               pid,
                "drug_1":                drug_1,
                "drug_2":                drug_2,
                "rank":                  rank,
                "label_index":           int(lbl_idx),
                "side_effect_id":        lm_row["side_effect_id"],
                "side_effect_name":      lm_row["side_effect_name"],
                "predicted_probability": float(probs_i[lbl_idx]),
                "is_true_label":         bool(true_i[lbl_idx]),
            })

    pd.DataFrame(example_rows).to_csv(EXAMPLES_CSV, index=False)

    # -----------------------------------------------------------------------
    # Save test metrics JSON (last — signals completion)
    # -----------------------------------------------------------------------
    TEST_METRICS.write_text(json.dumps(test_metrics, indent=2))

    print(f"\nOutputs saved:")
    print(f"  {TEST_METRICS}")
    print(f"  {PER_LABEL_CSV}")
    print(f"  {PREDICTIONS_NPZ}")
    print(f"  {EXAMPLES_CSV}")
    print(f"\nCheckpoint SHA-256:   {ckpt_sha}")
    print(f"Predictions SHA-256:  {pred_sha}")

    # -----------------------------------------------------------------------
    # Disk-loaded validation
    # -----------------------------------------------------------------------
    print("\nValidating saved artifacts from disk ...")
    loaded = np.load(PREDICTIONS_NPZ)
    assert set(loaded.files) >= {"pair_ids", "probabilities", "selected_threshold"}
    assert loaded["probabilities"].shape == (len(test_ids), labels.shape[1])
    assert loaded["probabilities"].dtype == np.float32
    assert np.all(np.isfinite(loaded["probabilities"]))
    assert np.all((loaded["probabilities"] >= 0) & (loaded["probabilities"] <= 1))
    assert np.array_equal(loaded["pair_ids"], np.sort(loaded["pair_ids"]))
    pl_loaded = pd.read_csv(PER_LABEL_CSV)
    assert len(pl_loaded) == labels.shape[1]
    assert {"label_index", "side_effect_id", "auroc", "auprc",
            "f1_at_selected_threshold"}.issubset(pl_loaded.columns)
    print("  OK test_predictions.npz shape, dtype, range, ordering")
    print("  OK per_label_metrics.csv row count and schema")
    print("\nEvaluation complete.")


if __name__ == "__main__":
    main()
