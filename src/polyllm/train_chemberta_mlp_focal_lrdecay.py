"""
Milestone 9d — ChemBERTa MLP ablation: Focal loss + decayed LR, everything
else held fixed at the Milestone 7 / 9c protocol.

Research question (user-specified, 2026-08-09): "When I align the
remaining training configuration more closely with PolyLLM, do my results
move toward the authors' reported values?"

Exactly two changes from src/polyllm/train_chemberta_mlp.py (which itself
already carries the Milestone 9c LeakyReLU slope=0.1 fix):
    1. Loss:            BCEWithLogitsLoss -> BinaryFocalLossWithLogits
                         (gamma=2.0, alpha=0.25 -- literature defaults,
                         NOT paper-specified; see src/polyllm/losses.py)
    2. Learning rate:    fixed 0.001 -> 0.005 initial, with
                         torch.optim.lr_scheduler.ExponentialLR(gamma=0.96)
                         stepped once per epoch (100 epochs -> final LR
                         approx 0.005 * 0.96^99 approx 8.4e-5). Per-epoch
                         stepping is an assumption -- the paper names a
                         decay rate but not a decay-step granularity.

Everything else is identical to Milestone 7/9c: same data, same frozen
ChemBERTa-77M-MLM embeddings (Milestone 4/5), same MultilabelMLP
architecture (hidden dims, dropout, BatchNorm placement, negative_slope
already at the paper-matched 0.1), same 80/10/10 pair-random split, same
seed=42, same batch size, same max_epochs/patience/min_delta.

Output goes to outputs/polyllm/chemberta_focal_lrdecay/ -- a new,
separate directory. Milestone 7's outputs/polyllm/chemberta/ (the
"official" reproduction result) is never touched by this script.

Official run:
    .venv-polyllm/Scripts/python.exe src/polyllm/train_chemberta_mlp_focal_lrdecay.py

After training completes, run the separate evaluator:
    .venv-polyllm/Scripts/python.exe src/polyllm/evaluate_chemberta_mlp_focal_lrdecay.py

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

_src = Path(__file__).resolve().parent.parent
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

from polyllm.losses import BinaryFocalLossWithLogits
from polyllm.metrics import apply_threshold, macro_auprc, micro_f1, select_threshold
from polyllm.models.mlp import MultilabelMLP
from polyllm.training import (
    EarlyStopping,
    PairDataset,
    load_checkpoint,
    make_seeded_loader,
    run_train_epoch,
    run_val_epoch,
    save_checkpoint,
    set_seeds,
    validate_input_alignment,
)

# ---------------------------------------------------------------------------
# Fixed paths — identical inputs to Milestone 7; output directory is new
# ---------------------------------------------------------------------------

FEATURES_PATH   = Path("data/features/chemberta_pair_embeddings.npy")
LABELS_PATH     = Path("data/processed/polyllm_labels.npy")
PAIR_INDEX_PATH = Path("data/features/chemberta_pair_index.csv")
LABEL_MAP_PATH  = Path("data/processed/polyllm_label_mapping.csv")
TRAIN_IDS_PATH  = Path("data/splits/train_pair_ids.csv")
VAL_IDS_PATH    = Path("data/splits/validation_pair_ids.csv")
TEST_IDS_PATH   = Path("data/splits/test_pair_ids.csv")  # loaded for alignment only

OUTPUT_DIR      = Path("outputs/polyllm/chemberta_focal_lrdecay")
CHECKPOINT_PATH = OUTPUT_DIR / "checkpoints/best_model.pt"
HISTORY_PATH    = OUTPUT_DIR / "training_history.csv"
CONFIG_PATH     = OUTPUT_DIR / "training_config.json"
THRESHOLD_PATH  = OUTPUT_DIR / "validation_threshold.json"

# ---------------------------------------------------------------------------
# Default training configuration
# Identical to Milestone 7/9c (outputs/polyllm/chemberta/training_config.json)
# except learning_rate, lr_schedule*, loss, focal_*.
# ---------------------------------------------------------------------------

DEFAULT_CONFIG: dict = {
    "random_seed":                42,
    "optimizer":                  "Adam",
    "learning_rate":              0.005,     # was 0.001 (M7/9c); paper ~0.005 per unconfirmed summary
    "lr_schedule":                "exponential_decay",
    "lr_decay_gamma":             0.96,
    "lr_decay_step_granularity":  "per_epoch",  # assumption -- paper does not specify
    "loss":                       "BinaryFocalLossWithLogits",
    "focal_gamma":                2.0,       # literature default (Lin et al. 2017), not paper-specified
    "focal_alpha":                0.25,      # literature default (Lin et al. 2017), not paper-specified
    "class_weights":              None,
    "batch_size":                 256,
    "max_epochs":                 100,
    "patience":                   10,
    "min_delta":                  0.0001,
    "model_selection_metric":     "validation_macro_auprc",
    "dropout":                    0.2,
    "leaky_relu_negative_slope":  0.1,       # unchanged from M9c (already paper-matched)
    "num_workers":                0,
    "input_dim":                  384,
    "output_dim":                 963,
    "hidden_dims":                [512, 1024, 2048],
    "batch_norm_after_layer":     1,
    "mixed_precision":            False,
    "cross_hardware_reproducibility": "not guaranteed",
    "feature_source":             str(FEATURES_PATH),
    "label_source":               str(LABELS_PATH),
    "representation":             "ChemBERTa-77M-MLM frozen pair embeddings (element-wise sum, Milestone 5)",
    "experiment_note": (
        "Milestone 9d ablation. Same data, same ChemBERTa embeddings, same "
        "MultilabelMLP architecture, same 80/10/10 split as Milestone 7/9c. "
        "Only loss (BCE->FocalBCE) and learning-rate schedule (fixed 0.001-> "
        "0.005 w/ 0.96 exponential decay) changed, to test whether aligning "
        "the remaining training configuration with the paper narrows the "
        "residual AUPRC/AP@50 gap left after the Milestone 9c slope fix. "
        "focal_gamma/focal_alpha and the per-epoch decay granularity are "
        "literature-standard assumptions, not values confirmed from the "
        "paper's primary text -- see notes/deviations_from_paper.md Sec 2.3."
    ),
}


# ---------------------------------------------------------------------------
# ChemBERTa-specific feature validation (same as train_chemberta_mlp.py)
# ---------------------------------------------------------------------------

def validate_chemberta_features(features: np.ndarray) -> None:
    if features.dtype != np.float32:
        raise ValueError(f"ChemBERTa features must be float32, got {features.dtype}.")
    non_finite = int((~np.isfinite(features)).sum())
    if non_finite > 0:
        raise ValueError(f"ChemBERTa features contain {non_finite} non-finite values (NaN/Inf).")
    all_zero_rows = int(np.all(features == 0, axis=1).sum())
    if all_zero_rows > 0:
        raise ValueError(f"ChemBERTa features have {all_zero_rows} all-zero rows.")


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Milestone 9d ablation: ChemBERTa MLP with Focal loss + decayed LR."
    )
    p.add_argument("--seed",         type=int,   default=DEFAULT_CONFIG["random_seed"])
    p.add_argument("--batch-size",   type=int,   default=DEFAULT_CONFIG["batch_size"])
    p.add_argument("--max-epochs",   type=int,   default=DEFAULT_CONFIG["max_epochs"])
    p.add_argument("--patience",     type=int,   default=DEFAULT_CONFIG["patience"])
    p.add_argument("--lr",           type=float, default=DEFAULT_CONFIG["learning_rate"])
    p.add_argument("--lr-decay",     type=float, default=DEFAULT_CONFIG["lr_decay_gamma"])
    p.add_argument("--focal-gamma",  type=float, default=DEFAULT_CONFIG["focal_gamma"])
    p.add_argument("--focal-alpha",  type=float, default=DEFAULT_CONFIG["focal_alpha"])
    p.add_argument("--dropout",      type=float, default=DEFAULT_CONFIG["dropout"])
    p.add_argument("--overwrite",    action="store_true",
                    help="Overwrite existing training outputs.")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Data loading (identical to Milestone 7)
# ---------------------------------------------------------------------------

def load_data():
    print("Loading data ...")
    features   = np.load(FEATURES_PATH, mmap_mode="r")
    labels     = np.load(LABELS_PATH,   mmap_mode="r")
    pair_index = pd.read_csv(PAIR_INDEX_PATH)
    label_map  = pd.read_csv(LABEL_MAP_PATH)
    train_ids  = pd.read_csv(TRAIN_IDS_PATH)["pair_id"].to_numpy()
    val_ids    = pd.read_csv(VAL_IDS_PATH)["pair_id"].to_numpy()
    test_ids   = pd.read_csv(TEST_IDS_PATH)["pair_id"].to_numpy()  # IDs only
    return features, labels, pair_index, label_map, train_ids, val_ids, test_ids


# ---------------------------------------------------------------------------
# Full training
# ---------------------------------------------------------------------------

def build_config(args: argparse.Namespace) -> dict:
    cfg = dict(DEFAULT_CONFIG)
    cfg["random_seed"]     = args.seed
    cfg["batch_size"]      = args.batch_size
    cfg["max_epochs"]      = args.max_epochs
    cfg["patience"]        = args.patience
    cfg["learning_rate"]   = args.lr
    cfg["lr_decay_gamma"]  = args.lr_decay
    cfg["focal_gamma"]     = args.focal_gamma
    cfg["focal_alpha"]     = args.focal_alpha
    cfg["dropout"]         = args.dropout
    return cfg


def make_history_row_with_lr(
    epoch: int, train_loss: float, val_loss: float,
    val_macro_auprc: float, checkpoint_saved: bool, learning_rate: float,
) -> dict:
    return {
        "epoch":            epoch,
        "train_loss":       round(train_loss, 6),
        "val_loss":         round(val_loss, 6),
        "val_macro_auprc":  round(val_macro_auprc, 6),
        "checkpoint_saved": checkpoint_saved,
        "learning_rate":    round(learning_rate, 8),
    }


def run_full_training(
    features: np.ndarray,
    labels: np.ndarray,
    train_ids: np.ndarray,
    val_ids: np.ndarray,
    config: dict,
    overwrite: bool = False,
) -> None:
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
    print(f"Model: MultilabelMLP(input_dim=384)  parameters: {n_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=config["lr_decay_gamma"])
    criterion = BinaryFocalLossWithLogits(gamma=config["focal_gamma"], alpha=config["focal_alpha"])
    print(f"Loss: BinaryFocalLossWithLogits(gamma={config['focal_gamma']}, alpha={config['focal_alpha']})")
    print(f"LR schedule: Adam(lr={config['learning_rate']}) -> "
          f"ExponentialLR(gamma={config['lr_decay_gamma']}), stepped once per epoch")

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

    CONFIG_PATH.write_text(json.dumps(full_config, indent=2))
    print(f"Config saved -> {CONFIG_PATH}")

    train_ds = PairDataset(train_ids, features, labels)
    val_ds   = PairDataset(val_ids,   features, labels)
    train_loader = make_seeded_loader(
        train_ds, config["batch_size"], shuffle=True,
        seed=seed, num_workers=config["num_workers"]
    )
    val_loader = make_seeded_loader(
        val_ds, config["batch_size"], shuffle=False,
        seed=seed, num_workers=config["num_workers"]
    )

    early_stop = EarlyStopping(patience=config["patience"], min_delta=config["min_delta"])
    history: list[dict] = []

    print(f"\nTraining for up to {config['max_epochs']} epochs "
          f"(patience={config['patience']}, min_delta={config['min_delta']}) ...")

    for epoch in range(1, config["max_epochs"] + 1):
        current_lr = optimizer.param_groups[0]["lr"]
        train_loss = run_train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_probs, val_labels, _pids = run_val_epoch(
            model, val_loader, criterion, device
        )
        scheduler.step()

        val_mauprc = macro_auprc(val_labels, val_probs)["macro_auprc"]
        improved   = early_stop.step(val_mauprc, epoch)

        if improved:
            save_checkpoint(
                CHECKPOINT_PATH, model, optimizer,
                epoch=epoch, score=val_mauprc, config=full_config
            )

        row = make_history_row_with_lr(epoch, train_loss, val_loss, val_mauprc, improved, current_lr)
        history.append(row)

        marker = " <- best" if improved else ""
        print(
            f"  Epoch {epoch:3d}/{config['max_epochs']}  lr={current_lr:.6f}  "
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
    # Threshold selection on validation only
    # -----------------------------------------------------------------------
    print("\nLoading best checkpoint for threshold selection ...")
    best_model = MultilabelMLP(
        input_dim=config["input_dim"],
        output_dim=config["output_dim"],
        dropout=config["dropout"],
        negative_slope=config["leaky_relu_negative_slope"],
    ).to(device)
    ckpt = load_checkpoint(CHECKPOINT_PATH, best_model, device=device)
    print(f"  Restored epoch {ckpt['epoch']}  val_macro_auprc={ckpt['val_macro_auprc']:.4f}")

    _, val_probs, val_labels, _ = run_val_epoch(best_model, val_loader, criterion, device)

    thr_result = select_threshold(val_labels, val_probs)
    val_pred_05 = apply_threshold(val_probs, 0.5)
    val_f1_05   = micro_f1(val_labels, val_pred_05)
    thr_result["validation_micro_f1_at_0_5"] = val_f1_05
    thr_result["best_epoch"] = ckpt["epoch"]

    THRESHOLD_PATH.write_text(json.dumps(thr_result, indent=2))
    print(f"Threshold saved -> {THRESHOLD_PATH}")
    print(f"  Selected threshold: {thr_result['selected_threshold']:.2f}  "
          f"val micro-F1: {thr_result['validation_micro_f1']:.4f}")
    print(f"  F1 at 0.5: {val_f1_05:.4f}")
    print("\nTraining complete. Run evaluate_chemberta_mlp_focal_lrdecay.py for test-set metrics.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    features, labels, pair_index, label_map, train_ids, val_ids, test_ids = load_data()
    print(f"Features: {features.shape}  Labels: {labels.shape}")
    print(f"Train: {len(train_ids)}  Val: {len(val_ids)}  Test: {len(test_ids)}")

    print("Validating ChemBERTa feature properties ...")
    validate_chemberta_features(features)
    print("  OK dtype=float32, all finite, no all-zero rows")

    print("Validating artifact alignment ...")
    validate_input_alignment(
        features, labels, pair_index, label_map,
        train_ids, val_ids, test_ids,
        expected_n_features=384,
    )
    print("  OK alignment verified")

    config = build_config(args)
    run_full_training(features, labels, train_ids, val_ids, config, args.overwrite)


if __name__ == "__main__":
    main()
