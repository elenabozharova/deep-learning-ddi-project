"""
Controlled representation ablation — Morgan fingerprint MLP trained under
the exact Milestone 9e ("faithful-training") recipe.

Goal (user-specified, 2026-08-13): hold the training protocol fixed at the
Milestone 9e ChemBERTa configuration (src/polyllm/train_chemberta_mlp_faithful.py)
and swap only the input representation, to isolate whether Morgan/ECFP4's
apparent advantage over ChemBERTa (Milestone 8, official BCE/lr=0.001/batch=256
runs) survives once loss, batch size, and LR schedule are no longer confounds.

Everything below is copied from train_chemberta_mlp_faithful.py verbatim
except: feature/output paths point at the Morgan pair-fingerprint files,
input_dim=2048 (vs. 384), and the ChemBERTa-specific dtype/finite-value
feature validation is replaced with the same checks the official Morgan
baseline (train_morgan_baseline.py) never needed (Morgan fingerprints are
deterministic RDKit output, not a frozen LM's floating-point embeddings).

This is a controlled ablation/comparison, NOT a replacement for the
official Morgan baseline (outputs/baseline/morgan/, Milestone 6B/9c) or the
official ChemBERTa result (outputs/polyllm/chemberta/, Milestone 7/9c).
Neither of those directories is touched by this script.

Official run:
    .venv-polyllm/Scripts/python.exe src/polyllm/train_morgan_mlp_faithful.py

After training completes, run the separate evaluator:
    .venv-polyllm/Scripts/python.exe src/polyllm/evaluate_morgan_mlp_faithful.py
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
from polyllm.lr_schedules import keras_exponential_decay_lr
from polyllm.metrics import apply_threshold, macro_auprc, micro_f1, select_threshold
from polyllm.models.mlp import MultilabelMLP
from polyllm.training import (
    EarlyStopping,
    PairDataset,
    load_checkpoint,
    make_seeded_loader,
    run_val_epoch,
    save_checkpoint,
    set_seeds,
    validate_input_alignment,
)

# ---------------------------------------------------------------------------
# Fixed paths — same split/label files as every other MLP experiment;
# Morgan features/output directory are the only path changes vs. the
# ChemBERTa faithful-training script.
# ---------------------------------------------------------------------------

FEATURES_PATH   = Path("data/features/morgan_pair_fingerprints.npy")
LABELS_PATH     = Path("data/processed/polyllm_labels.npy")
PAIR_INDEX_PATH = Path("data/features/morgan_pair_index.csv")
LABEL_MAP_PATH  = Path("data/processed/polyllm_label_mapping.csv")
TRAIN_IDS_PATH  = Path("data/splits/train_pair_ids.csv")
VAL_IDS_PATH    = Path("data/splits/validation_pair_ids.csv")
TEST_IDS_PATH   = Path("data/splits/test_pair_ids.csv")  # loaded for alignment only

OUTPUT_DIR      = Path("outputs/baseline/morgan_faithful_training")
CHECKPOINT_PATH = OUTPUT_DIR / "checkpoints/best_model.pt"
HISTORY_PATH    = OUTPUT_DIR / "training_history.csv"
CONFIG_PATH     = OUTPUT_DIR / "training_config.json"
THRESHOLD_PATH  = OUTPUT_DIR / "validation_threshold.json"

# ---------------------------------------------------------------------------
# Default training configuration — identical to Milestone 9e ChemBERTa
# except input_dim, feature_source, representation.
# ---------------------------------------------------------------------------

DEFAULT_CONFIG: dict = {
    "random_seed":                42,
    "optimizer":                  "Adam",
    "adam_beta1":                 0.9,
    "adam_beta2":                 0.999,
    "adam_eps":                   1e-7,      # Keras default; PyTorch default is 1e-8. Explicit for fidelity.
    "learning_rate_initial":      0.005,
    "lr_schedule":                "keras_exponential_decay",
    "lr_decay_steps":             1000,
    "lr_decay_rate":              0.96,
    "lr_staircase":                True,
    "lr_step_granularity":        "per_optimizer_step",
    "loss":                       "BinaryFocalLossWithLogits",
    "focal_gamma":                2.0,
    "focal_alpha":                -1.0,      # disabled -- matches Keras apply_class_balancing=False
    "focal_label_smoothing":      0.2,
    "batch_size":                 32,
    "max_epochs":                 100,
    "patience":                   10,
    "min_delta":                  0.0001,
    "model_selection_metric":     "validation_macro_auprc",
    "restore_best_checkpoint":    True,
    "dropout":                    0.2,
    "leaky_relu_negative_slope":  0.1,
    "num_workers":                0,
    "input_dim":                  2048,
    "output_dim":                 963,
    "hidden_dims":                [512, 1024, 2048],
    "batch_norm_after_layer":     1,
    "feature_source":             str(FEATURES_PATH),
    "label_source":                str(LABELS_PATH),
    "representation":             "Morgan/ECFP4 (radius=2, 2048-bit) pair fingerprints (element-wise sum)",
    "kernel_initializer":         "pytorch_default (kaiming_uniform_, a=sqrt(5))",
    "experiment_note": (
        "Controlled representation ablation. Same split, same MultilabelMLP "
        "hidden architecture, same loss/batch-size/LR-schedule as the "
        "ChemBERTa Milestone 9e faithful-training run "
        "(outputs/polyllm/chemberta_faithful_training/). Only the input "
        "representation and the resulting input_dim/total_parameters "
        "change. See notes/morgan_vs_chemberta_faithful_ablation.md."
    ),
}


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Controlled ablation: Morgan fingerprint MLP under the Milestone 9e faithful-training recipe."
    )
    p.add_argument("--seed",         type=int,   default=DEFAULT_CONFIG["random_seed"])
    p.add_argument("--batch-size",   type=int,   default=DEFAULT_CONFIG["batch_size"])
    p.add_argument("--max-epochs",   type=int,   default=DEFAULT_CONFIG["max_epochs"])
    p.add_argument("--patience",     type=int,   default=DEFAULT_CONFIG["patience"])
    p.add_argument("--lr",           type=float, default=DEFAULT_CONFIG["learning_rate_initial"])
    p.add_argument("--lr-decay-steps", type=int, default=DEFAULT_CONFIG["lr_decay_steps"])
    p.add_argument("--lr-decay-rate", type=float, default=DEFAULT_CONFIG["lr_decay_rate"])
    p.add_argument("--focal-gamma",  type=float, default=DEFAULT_CONFIG["focal_gamma"])
    p.add_argument("--label-smoothing", type=float, default=DEFAULT_CONFIG["focal_label_smoothing"])
    p.add_argument("--dropout",      type=float, default=DEFAULT_CONFIG["dropout"])
    p.add_argument("--overwrite",    action="store_true",
                    help="Overwrite existing training outputs.")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Data loading
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
# Per-step-LR training epoch (identical mechanism to the ChemBERTa faithful
# script; duplicated locally rather than shared to keep each ablation script
# self-contained and independently readable, matching the existing pattern)
# ---------------------------------------------------------------------------

def run_train_epoch_with_step_schedule(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    global_step: int,
    lr_initial: float,
    lr_decay_steps: int,
    lr_decay_rate: float,
    lr_staircase: bool,
) -> tuple[float, int, float, float]:
    model.train()
    total_loss = 0.0
    n_batches = 0
    lr_at_epoch_start = keras_exponential_decay_lr(
        global_step, lr_initial, lr_decay_steps, lr_decay_rate, lr_staircase
    )
    lr_this_step = lr_at_epoch_start

    for _pair_ids, x, y in loader:
        lr_this_step = keras_exponential_decay_lr(
            global_step, lr_initial, lr_decay_steps, lr_decay_rate, lr_staircase
        )
        for pg in optimizer.param_groups:
            pg["lr"] = lr_this_step

        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += float(loss.item())
        n_batches += 1
        global_step += 1

    avg_loss = total_loss / n_batches if n_batches else 0.0
    return avg_loss, global_step, lr_at_epoch_start, lr_this_step


# ---------------------------------------------------------------------------
# Full training
# ---------------------------------------------------------------------------

def build_config(args: argparse.Namespace) -> dict:
    cfg = dict(DEFAULT_CONFIG)
    cfg["random_seed"]              = args.seed
    cfg["batch_size"]                = args.batch_size
    cfg["max_epochs"]                = args.max_epochs
    cfg["patience"]                  = args.patience
    cfg["learning_rate_initial"]     = args.lr
    cfg["lr_decay_steps"]            = args.lr_decay_steps
    cfg["lr_decay_rate"]             = args.lr_decay_rate
    cfg["focal_gamma"]               = args.focal_gamma
    cfg["focal_label_smoothing"]     = args.label_smoothing
    cfg["dropout"]                   = args.dropout
    return cfg


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
    print(f"Model: MultilabelMLP(input_dim=2048)  parameters: {n_params:,}")

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["learning_rate_initial"],
        betas=(config["adam_beta1"], config["adam_beta2"]),
        eps=config["adam_eps"],
    )
    criterion = BinaryFocalLossWithLogits(
        gamma=config["focal_gamma"],
        alpha=config["focal_alpha"],
        label_smoothing=config["focal_label_smoothing"],
    )
    print(f"Loss: BinaryFocalLossWithLogits(gamma={config['focal_gamma']}, "
          f"alpha={config['focal_alpha']} [disabled], "
          f"label_smoothing={config['focal_label_smoothing']})")
    print(f"Optimizer: Adam(lr={config['learning_rate_initial']}, "
          f"betas=({config['adam_beta1']},{config['adam_beta2']}), eps={config['adam_eps']})")

    n_train = len(train_ids)
    steps_per_epoch = -(-n_train // config["batch_size"])  # ceil division
    print(f"Batch size: {config['batch_size']}  ->  ~{steps_per_epoch} optimizer steps/epoch "
          f"({n_train} train pairs)")
    print(f"LR schedule: keras_exponential_decay_lr(step, {config['learning_rate_initial']}, "
          f"decay_steps={config['lr_decay_steps']}, decay_rate={config['lr_decay_rate']}, "
          f"staircase={config['lr_staircase']}), applied per optimizer step")

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
        "approx_steps_per_epoch": steps_per_epoch,
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
    val_criterion = criterion

    early_stop = EarlyStopping(patience=config["patience"], min_delta=config["min_delta"])
    history: list[dict] = []
    global_step = 0

    print(f"\nTraining for up to {config['max_epochs']} epochs "
          f"(patience={config['patience']}, min_delta={config['min_delta']}) ...")

    for epoch in range(1, config["max_epochs"] + 1):
        global_step_start = global_step
        train_loss, global_step, lr_start, lr_end = run_train_epoch_with_step_schedule(
            model, train_loader, criterion, optimizer, device,
            global_step,
            config["learning_rate_initial"], config["lr_decay_steps"],
            config["lr_decay_rate"], config["lr_staircase"],
        )
        val_loss, val_probs, val_labels, _pids = run_val_epoch(
            model, val_loader, val_criterion, device
        )

        val_mauprc = macro_auprc(val_labels, val_probs)["macro_auprc"]
        improved   = early_stop.step(val_mauprc, epoch)

        if improved:
            save_checkpoint(
                CHECKPOINT_PATH, model, optimizer,
                epoch=epoch, score=val_mauprc, config=full_config
            )

        history.append({
            "epoch":              epoch,
            "train_loss":         round(train_loss, 6),
            "val_loss":           round(val_loss, 6),
            "val_macro_auprc":    round(val_mauprc, 6),
            "checkpoint_saved":   improved,
            "global_step_start":  global_step_start,
            "global_step_end":    global_step,
            "lr_at_epoch_start":  round(lr_start, 8),
            "lr_at_epoch_end":    round(lr_end, 8),
        })

        marker = " <- best" if improved else ""
        print(
            f"  Epoch {epoch:3d}/{config['max_epochs']}  "
            f"step={global_step:6d}  lr={lr_start:.6f}->{lr_end:.6f}  "
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

    _, val_probs, val_labels, _ = run_val_epoch(best_model, val_loader, val_criterion, device)

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
    print("\nTraining complete. Run evaluate_morgan_mlp_faithful.py for test-set metrics.")


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
        expected_n_features=2048,
    )
    print("  OK alignment verified")

    config = build_config(args)
    run_full_training(features, labels, train_ids, val_ids, config, args.overwrite)


if __name__ == "__main__":
    main()
