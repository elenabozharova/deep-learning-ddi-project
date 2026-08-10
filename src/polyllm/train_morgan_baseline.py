"""
Milestone 6B — Morgan fingerprint multi-label MLP: training entry point.

Official run:
    .venv-polyllm\Scripts\python.exe src/polyllm/train_morgan_baseline.py

Engineering smoke test (3 epochs, 2 000 training pairs, 20 labels):
    .venv-polyllm\Scripts\python.exe src/polyllm/train_morgan_baseline.py --smoke-test

After training completes, run the separate evaluator:
    .venv-polyllm\Scripts\python.exe src/polyllm/evaluate_morgan_baseline.py

The test split is never loaded or evaluated in this script.
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

# Project imports — add src/ to path when run as a script
_src = Path(__file__).resolve().parent.parent
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

from polyllm.metrics import macro_auprc
from polyllm.models.mlp import MultilabelMLP
from polyllm.training import (
    EarlyStopping,
    PairDataset,
    file_sha256,
    load_checkpoint,
    make_history_row,
    make_seeded_loader,
    run_train_epoch,
    run_val_epoch,
    save_checkpoint,
    set_seeds,
    validate_input_alignment,
)

# ---------------------------------------------------------------------------
# Fixed paths
# ---------------------------------------------------------------------------

FEATURES_PATH    = Path("data/features/morgan_pair_fingerprints.npy")
LABELS_PATH      = Path("data/processed/polyllm_labels.npy")
PAIR_INDEX_PATH  = Path("data/features/morgan_pair_index.csv")
LABEL_MAP_PATH   = Path("data/processed/polyllm_label_mapping.csv")
TRAIN_IDS_PATH   = Path("data/splits/train_pair_ids.csv")
VAL_IDS_PATH     = Path("data/splits/validation_pair_ids.csv")
TEST_IDS_PATH    = Path("data/splits/test_pair_ids.csv")  # loaded only for alignment check

OUTPUT_DIR       = Path("outputs/baseline/morgan")
CHECKPOINT_PATH  = OUTPUT_DIR / "checkpoints/best_model.pt"
HISTORY_PATH     = OUTPUT_DIR / "training_history.csv"
CONFIG_PATH      = OUTPUT_DIR / "training_config.json"
THRESHOLD_PATH   = OUTPUT_DIR / "validation_threshold.json"
SMOKE_OUTPUT_DIR = OUTPUT_DIR / "smoke_test"

# ---------------------------------------------------------------------------
# Default training configuration
# ---------------------------------------------------------------------------

DEFAULT_CONFIG: dict = {
    "random_seed":              42,
    "optimizer":                "Adam",
    "learning_rate":            0.001,
    "loss":                     "BCEWithLogitsLoss",
    "class_weights":            None,
    "batch_size":               256,
    "max_epochs":               100,
    "patience":                 10,
    "min_delta":                0.0001,
    "model_selection_metric":   "validation_macro_auprc",
    "dropout":                  0.2,
    "leaky_relu_negative_slope": 0.1,    # PolyLLM paper Sec 2.2 (was 0.01, PyTorch default, until 2026-08-08)
    "num_workers":              0,        # reliable on Windows
    "input_dim":                2048,
    "output_dim":               963,
    "hidden_dims":              [512, 1024, 2048],
    "batch_norm_after_layer":   1,        # only layer 1 has BatchNorm
    "mixed_precision":          False,
    "cross_hardware_reproducibility": "not guaranteed",
    "feature_source":           str(FEATURES_PATH),
    "label_source":             str(LABELS_PATH),
}


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train Morgan fingerprint MLP baseline.")
    p.add_argument("--smoke-test", action="store_true",
                   help="Run engineering smoke test only (3 epochs, 2 000 pairs, 20 labels).")
    p.add_argument("--seed",       type=int,   default=DEFAULT_CONFIG["random_seed"])
    p.add_argument("--batch-size", type=int,   default=DEFAULT_CONFIG["batch_size"])
    p.add_argument("--max-epochs", type=int,   default=DEFAULT_CONFIG["max_epochs"])
    p.add_argument("--patience",   type=int,   default=DEFAULT_CONFIG["patience"])
    p.add_argument("--lr",         type=float, default=DEFAULT_CONFIG["learning_rate"])
    p.add_argument("--dropout",    type=float, default=DEFAULT_CONFIG["dropout"])
    p.add_argument("--overwrite",  action="store_true",
                   help="Overwrite existing training outputs.")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data():
    print("Loading data ...")
    features    = np.load(FEATURES_PATH,   mmap_mode="r")
    labels      = np.load(LABELS_PATH,     mmap_mode="r")
    pair_index  = pd.read_csv(PAIR_INDEX_PATH)
    label_map   = pd.read_csv(LABEL_MAP_PATH)
    train_ids   = pd.read_csv(TRAIN_IDS_PATH)["pair_id"].to_numpy()
    val_ids     = pd.read_csv(VAL_IDS_PATH)["pair_id"].to_numpy()
    test_ids    = pd.read_csv(TEST_IDS_PATH)["pair_id"].to_numpy()   # IDs only, no data loaded
    return features, labels, pair_index, label_map, train_ids, val_ids, test_ids


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

def run_smoke_test(
    features: np.ndarray,
    labels: np.ndarray,
    train_ids: np.ndarray,
    val_ids: np.ndarray,
    config: dict,
) -> None:
    print("\n=== ENGINEERING SMOKE TEST ===")
    print(f"Output directory: {SMOKE_OUTPUT_DIR}")
    SMOKE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    seed = config["random_seed"]
    set_seeds(seed)

    # 2 000 training pair IDs
    smoke_train_ids = train_ids[:2000]

    # 20 most frequent labels — measured from training data only
    train_label_counts = (
        np.array(labels[smoke_train_ids, :], dtype=np.int64).sum(axis=0)
    )
    top20_idx = np.sort(np.argsort(train_label_counts)[::-1][:20])
    print(f"Top-20 label indices (from training data): {top20_idx.tolist()}")
    print(f"Their training frequencies: {train_label_counts[top20_idx].tolist()}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = MultilabelMLP(
        input_dim=config["input_dim"],
        output_dim=20,
        dropout=config["dropout"],
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    criterion = torch.nn.BCEWithLogitsLoss()

    train_ds = PairDataset(smoke_train_ids, features, labels, label_indices=top20_idx)
    val_ds   = PairDataset(val_ids,         features, labels, label_indices=top20_idx)
    train_loader = make_seeded_loader(train_ds, config["batch_size"], shuffle=True,  seed=seed)
    val_loader   = make_seeded_loader(val_ds,   config["batch_size"], shuffle=False, seed=seed)

    last_val_auprc = float("nan")
    smoke_ckpt = SMOKE_OUTPUT_DIR / "smoke_checkpoint.pt"

    for epoch in range(1, 4):
        # Snapshot a parameter before training to verify update
        param_snapshot = next(model.parameters()).data.clone()

        train_loss = run_train_epoch(model, train_loader, criterion, optimizer, device)

        # Verify: forward pass, BCE loss, backward, optimizer update
        param_after = next(model.parameters()).data
        assert not torch.equal(param_snapshot, param_after), \
            f"Epoch {epoch}: training did not update model parameters."

        val_loss, val_probs, val_labels, _pids = run_val_epoch(
            model, val_loader, criterion, device
        )

        # Verify: validation prediction collection
        assert val_probs.shape == (len(val_ids), 20), \
            f"Unexpected val_probs shape: {val_probs.shape}"
        assert val_probs.dtype == np.float32
        assert np.all(np.isfinite(val_probs))
        assert np.all((val_probs >= 0.0) & (val_probs <= 1.0))

        # Verify: metric calculation
        last_val_auprc = macro_auprc(val_labels, val_probs)["macro_auprc"]
        print(
            f"  Epoch {epoch}: train_loss={train_loss:.4f}  "
            f"val_loss={val_loss:.4f}  val_macro_auprc={last_val_auprc:.4f}"
        )

    # Verify: checkpoint writing
    smoke_config = {**config, "output_dim": 20}
    save_checkpoint(smoke_ckpt, model, optimizer, epoch=3, score=last_val_auprc, config=smoke_config)
    assert smoke_ckpt.exists(), "Checkpoint file was not written."

    # Verify: checkpoint loading and prediction reproducibility
    model2 = MultilabelMLP(
        input_dim=config["input_dim"],
        output_dim=20,
        dropout=config["dropout"],
    ).to(device)
    ckpt = load_checkpoint(smoke_ckpt, model2, device=device)
    assert ckpt["epoch"] == 3
    assert ckpt["input_dim"] == config["input_dim"]
    assert ckpt["output_dim"] == 20

    _, val_probs2, _, _ = run_val_epoch(model2, val_loader, criterion, device)
    assert np.allclose(val_probs, val_probs2, atol=1e-5), \
        "Loaded checkpoint gives different predictions."

    print("\nSMOKE TEST PASSED")
    print("  OK forward pass")
    print("  OK BCE loss")
    print("  OK backward pass and optimizer update")
    print("  OK validation prediction collection")
    print("  OK metric calculation (macro AUPRC)")
    print("  OK checkpoint write")
    print("  OK checkpoint load and prediction reproducibility")
    print(f"\nNote: smoke-test metrics are NOT scientific results.")
    print(f"      20 labels, 2 000 training pairs, 3 epochs only.")


# ---------------------------------------------------------------------------
# Full training
# ---------------------------------------------------------------------------

def build_config(args: argparse.Namespace) -> dict:
    cfg = dict(DEFAULT_CONFIG)
    cfg["random_seed"]    = args.seed
    cfg["batch_size"]     = args.batch_size
    cfg["max_epochs"]     = args.max_epochs
    cfg["patience"]       = args.patience
    cfg["learning_rate"]  = args.lr
    cfg["dropout"]        = args.dropout
    return cfg


def run_full_training(
    features: np.ndarray,
    labels: np.ndarray,
    train_ids: np.ndarray,
    val_ids: np.ndarray,
    config: dict,
    overwrite: bool = False,
) -> None:
    # Overwrite guard
    if CHECKPOINT_PATH.exists() and THRESHOLD_PATH.exists() and not overwrite:
        print(
            f"Training outputs already exist at {OUTPUT_DIR}.\n"
            "Pass --overwrite to re-train."
        )
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "checkpoints").mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    seed = config["random_seed"]
    set_seeds(seed)

    model = MultilabelMLP(
        input_dim=config["input_dim"],
        output_dim=config["output_dim"],
        dropout=config["dropout"],
        negative_slope=config["leaky_relu_negative_slope"],
    ).to(device)

    n_params = model.count_parameters()
    print(f"Model: MultilabelMLP  parameters: {n_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    criterion = torch.nn.BCEWithLogitsLoss()

    import sklearn
    import torch as _torch
    software_versions = {
        "python":       platform.python_version(),
        "torch":        _torch.__version__,
        "numpy":        np.__version__,
        "pandas":       pd.__version__,
        "scikit_learn": sklearn.__version__,
    }

    full_config = {
        **config,
        "total_parameters": n_params,
        "device":           str(device),
        "train_pairs":      len(train_ids),
        "val_pairs":        len(val_ids),
        "software_versions": software_versions,
    }

    # Save config before training starts
    CONFIG_PATH.write_text(json.dumps(full_config, indent=2))
    print(f"Config saved -> {CONFIG_PATH}")

    train_ds = PairDataset(train_ids, features, labels)
    val_ds   = PairDataset(val_ids,   features, labels)
    train_loader = make_seeded_loader(train_ds, config["batch_size"], shuffle=True,  seed=seed, num_workers=config["num_workers"])
    val_loader   = make_seeded_loader(val_ds,   config["batch_size"], shuffle=False, seed=seed, num_workers=config["num_workers"])

    early_stop = EarlyStopping(patience=config["patience"], min_delta=config["min_delta"])
    history: list[dict] = []

    print(f"\nTraining for up to {config['max_epochs']} epochs "
          f"(patience={config['patience']}, min_delta={config['min_delta']}) ...")

    for epoch in range(1, config["max_epochs"] + 1):
        train_loss = run_train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_probs, val_labels, _pids = run_val_epoch(
            model, val_loader, criterion, device
        )

        val_mauprc = macro_auprc(val_labels, val_probs)["macro_auprc"]
        improved   = early_stop.step(val_mauprc, epoch)

        if improved:
            save_checkpoint(
                CHECKPOINT_PATH, model, optimizer,
                epoch=epoch, score=val_mauprc, config=full_config
            )

        row = make_history_row(epoch, train_loss, val_loss, val_mauprc, improved)
        history.append(row)

        marker = " <- best" if improved else ""
        print(
            f"  Epoch {epoch:3d}/{config['max_epochs']}  "
            f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
            f"val_macro_auprc={val_mauprc:.4f}{marker}"
        )

        if early_stop.should_stop:
            print(f"\nEarly stopping triggered after epoch {epoch} "
                  f"(no improvement for {config['patience']} epochs).")
            break

    pd.DataFrame(history).to_csv(HISTORY_PATH, index=False)
    print(f"\nHistory saved -> {HISTORY_PATH}")
    print(f"Best epoch: {early_stop.best_epoch}  "
          f"val_macro_auprc={early_stop.best_score:.4f}")

    # -----------------------------------------------------------------------
    # Threshold selection on validation (after restoring best checkpoint)
    # -----------------------------------------------------------------------
    print("\nLoading best checkpoint for threshold selection ...")
    best_model = MultilabelMLP(
        input_dim=config["input_dim"],
        output_dim=config["output_dim"],
        dropout=config["dropout"],
        negative_slope=config["leaky_relu_negative_slope"],
    ).to(device)
    ckpt = load_checkpoint(CHECKPOINT_PATH, best_model, device=device)
    print(f"  Restored epoch {ckpt['epoch']}  "
          f"val_macro_auprc={ckpt['val_macro_auprc']:.4f}")

    _, val_probs, val_labels, _ = run_val_epoch(
        best_model, val_loader, criterion, device
    )

    from polyllm.metrics import select_threshold, apply_threshold, micro_f1
    thr_result = select_threshold(val_labels, val_probs)

    # Also compute F1 at fixed 0.5
    val_pred_05 = apply_threshold(val_probs, 0.5)
    val_f1_05   = micro_f1(val_labels, val_pred_05)
    thr_result["validation_micro_f1_at_0_5"] = val_f1_05

    thr_result["best_epoch"] = ckpt["epoch"]
    THRESHOLD_PATH.write_text(json.dumps(thr_result, indent=2))
    print(f"Threshold saved -> {THRESHOLD_PATH}")
    print(f"  Selected threshold: {thr_result['selected_threshold']:.2f}  "
          f"val micro-F1: {thr_result['validation_micro_f1']:.4f}")
    print(f"  F1 at 0.5: {val_f1_05:.4f}")
    print("\nTraining complete. Run evaluate_morgan_baseline.py for test-set metrics.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    features, labels, pair_index, label_map, train_ids, val_ids, test_ids = load_data()
    print(f"Features: {features.shape}  Labels: {labels.shape}")
    print(f"Train: {len(train_ids)}  Val: {len(val_ids)}  Test: {len(test_ids)}")

    print("Validating artifact alignment ...")
    validate_input_alignment(
        features, labels, pair_index, label_map,
        train_ids, val_ids, test_ids,
    )
    print("  OK alignment verified")

    config = build_config(args)

    if args.smoke_test:
        run_smoke_test(features, labels, train_ids, val_ids, config)
    else:
        run_full_training(features, labels, train_ids, val_ids, config, args.overwrite)


if __name__ == "__main__":
    main()
