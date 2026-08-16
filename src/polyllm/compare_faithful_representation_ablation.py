"""
Controlled representation ablation — Morgan (faithful recipe) vs ChemBERTa
Milestone 9e, both trained under the identical M9e training protocol.

Research question (user-specified, 2026-08-13): "Under the same training
protocol, does the classical Morgan/ECFP4 molecular representation
outperform the frozen ChemBERTa representation for PolyLLM multi-label
polypharmacy side-effect prediction?"

Reuses the aggregate/per-label/agreement/bootstrap logic already written and
tested in compare_models.py (Milestone 8) rather than re-deriving it, since
those helper functions are generic over their (metrics dict / dataframe)
arguments. Two of them (build_prediction_agreement, run_bootstrap) read
their classification thresholds from that module's own MORGAN_THRESHOLD /
CHEMBERTA_THRESHOLD globals rather than taking them as parameters -- those
globals are monkeypatched to this ablation's own validation-selected
thresholds (0.40 / 0.39) before calling in, and restored after. This is
analysis only: no training, no artifact overwritten. Nothing under
outputs/comparison/ (the Milestone 8 official comparison) is touched --
output goes to a new, separate outputs/comparison/faithful_representation_ablation/.

Usage:
    .venv-polyllm/Scripts/python.exe src/polyllm/compare_faithful_representation_ablation.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import polyllm.compare_models as cm  # noqa: E402

MORGAN_DIR = ROOT / "outputs" / "baseline" / "morgan_faithful_training"
CHEMBERTA_DIR = ROOT / "outputs" / "polyllm" / "chemberta_faithful_training"
OUT_DIR = ROOT / "outputs" / "comparison" / "faithful_representation_ablation"
LABELS_PATH = ROOT / "data" / "processed" / "polyllm_labels.npy"

MORGAN_THRESHOLD = 0.40      # this ablation's own validation-selected threshold
CHEMBERTA_THRESHOLD = 0.39   # M9e's own validation-selected threshold

COMPARABLE_FIELDS = [
    "random_seed", "optimizer", "adam_beta1", "adam_beta2", "adam_eps",
    "learning_rate_initial", "lr_schedule", "lr_decay_steps", "lr_decay_rate",
    "lr_staircase", "lr_step_granularity", "loss", "focal_gamma",
    "focal_alpha", "focal_label_smoothing", "batch_size", "max_epochs",
    "patience", "min_delta", "model_selection_metric",
    "restore_best_checkpoint", "dropout", "leaky_relu_negative_slope",
    "num_workers", "output_dim", "hidden_dims", "batch_norm_after_layer",
    "train_pairs", "val_pairs",
]
INTENTIONAL_DIFFERENCES = ["input_dim", "total_parameters", "feature_source", "representation"]


def validate_comparability(morgan_cfg: dict, chemberta_cfg: dict) -> None:
    errors = [
        f"{f}: morgan={morgan_cfg.get(f)!r} != chemberta={chemberta_cfg.get(f)!r}"
        for f in COMPARABLE_FIELDS
        if morgan_cfg.get(f) != chemberta_cfg.get(f)
    ]
    if errors:
        raise ValueError("Configs are not comparable:\n" + "\n".join(errors))


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    morgan_cfg = json.loads((MORGAN_DIR / "training_config.json").read_text())
    chemberta_cfg = json.loads((CHEMBERTA_DIR / "training_config.json").read_text())
    morgan_metrics = json.loads((MORGAN_DIR / "test_metrics.json").read_text())
    chemberta_metrics = json.loads((CHEMBERTA_DIR / "test_metrics.json").read_text())
    morgan_ap50k = json.loads((MORGAN_DIR / "ap_at_k_recompute.json").read_text())
    chemberta_ap50k = json.loads((CHEMBERTA_DIR / "ap_at_k_recompute.json").read_text())

    print("Step 1: Validating comparability of the matched training protocol ...")
    validate_comparability(morgan_cfg, chemberta_cfg)
    print(f"  OK - {len(COMPARABLE_FIELDS)} config fields match exactly.")

    print("Loading predictions ...")
    morgan_probs, morgan_pids = cm.load_predictions(MORGAN_DIR / "test_predictions.npz")
    chem_probs, chem_pids = cm.load_predictions(CHEMBERTA_DIR / "test_predictions.npz")
    if not np.array_equal(morgan_pids, chem_pids):
        raise ValueError("Test pair IDs differ between the two runs.")
    labels_raw = np.load(LABELS_PATH, mmap_mode="r")
    test_labels = labels_raw[morgan_pids].astype(np.int32)
    print(f"  {len(morgan_pids)} test pairs, {morgan_probs.shape[1]} labels.")

    print("Step 2: Aggregate metrics table ...")
    agg_df = cm.build_aggregate_table(morgan_metrics, chemberta_metrics)
    # Add the paper-compatible per-label-axis AP@50, which isn't in test_metrics.json
    agg_df = pd.concat([agg_df, pd.DataFrame([{
        "metric": "ap_at_50_paper_axis",
        "morgan": morgan_ap50k["mean_ap_at_k"],
        "chemberta": chemberta_ap50k["mean_ap_at_k"],
        "difference": morgan_ap50k["mean_ap_at_k"] - chemberta_ap50k["mean_ap_at_k"],
        "relative_difference_pct": (morgan_ap50k["mean_ap_at_k"] - chemberta_ap50k["mean_ap_at_k"]) / chemberta_ap50k["mean_ap_at_k"] * 100.0,
        "winner": cm._winner(morgan_ap50k["mean_ap_at_k"], chemberta_ap50k["mean_ap_at_k"]),
    }])], ignore_index=True)
    agg_path = OUT_DIR / "aggregate_metrics.csv"
    agg_df.to_csv(agg_path, index=False, float_format="%.8f")
    print(agg_df.to_string(index=False))

    print("\nStep 3: Per-label comparison (win counts) ...")
    morgan_pl = pd.read_csv(MORGAN_DIR / "per_label_metrics.csv")
    chem_pl = pd.read_csv(CHEMBERTA_DIR / "per_label_metrics.csv")
    pl_df = cm.build_per_label_comparison(morgan_pl, chem_pl)
    pl_path = OUT_DIR / "per_label_comparison.csv"
    pl_df.to_csv(pl_path, index=False, float_format="%.8f")
    win_counts = {}
    for col in ("auroc", "auprc", "f1"):
        wc = int((pl_df[f"{col}_winner"] == "morgan").sum())
        cc = int((pl_df[f"{col}_winner"] == "chemberta").sum())
        tc = int((pl_df[f"{col}_winner"] == "tie").sum())
        win_counts[col] = {"morgan_wins": wc, "chemberta_wins": cc, "ties": tc}
        print(f"  {col}: morgan wins={wc}, chemberta wins={cc}, ties={tc} (of {len(pl_df)} labels)")

    print("\nStep 4: Prediction agreement (thresholds 0.40 / 0.39) ...")
    # build_prediction_agreement / run_bootstrap read thresholds from
    # compare_models' own module-level globals, not from parameters --
    # monkeypatch to this ablation's thresholds, restore after.
    orig_thresholds = (cm.MORGAN_THRESHOLD, cm.CHEMBERTA_THRESHOLD)
    cm.MORGAN_THRESHOLD, cm.CHEMBERTA_THRESHOLD = MORGAN_THRESHOLD, CHEMBERTA_THRESHOLD
    try:
        agreement = cm.build_prediction_agreement(morgan_probs, chem_probs)
        agree_path = OUT_DIR / "prediction_agreement.json"
        agree_path.write_text(json.dumps(agreement, indent=2))
        print(f"  Agreement: {agreement['threshold_agreement']['agree_pct']:.2f}%")
        print(f"  Pearson r={agreement['probability_correlation']['pearson_r']:.4f}, "
              f"Spearman rho={agreement['probability_correlation']['spearman_rho']:.4f}")
        print(f"  Top-10 Jaccard mean={agreement['top10_jaccard']['mean']:.4f}")

        print(f"\nStep 5: Paired bootstrap ({cm.BOOTSTRAP_REPS} reps, seed={cm.BOOTSTRAP_SEED}) ...")
        boot_df = cm.run_bootstrap(morgan_probs, chem_probs, test_labels)
    finally:
        cm.MORGAN_THRESHOLD, cm.CHEMBERTA_THRESHOLD = orig_thresholds

    boot_path = OUT_DIR / "bootstrap_differences.csv"
    boot_df.to_csv(boot_path, index=False, float_format="%.8f")
    bsummary = cm._bootstrap_summary(boot_df)
    observed = {
        "macro_auprc": float(morgan_metrics["macro_auprc"]) - float(chemberta_metrics["macro_auprc"]),
        "micro_f1_selected_threshold": float(morgan_metrics["micro_f1_selected_threshold"]) - float(chemberta_metrics["micro_f1_selected_threshold"]),
        "sample_mean_ap_at_50": float(morgan_metrics["sample_mean_ap_at_50"]) - float(chemberta_metrics["sample_mean_ap_at_50"]),
    }
    for metric, obs_diff in observed.items():
        if metric in bsummary:
            bsummary[metric]["observed_difference_morgan_minus_chemberta"] = obs_diff
    for metric in ("macro_auprc", "micro_f1_selected_threshold", "sample_mean_ap_at_50"):
        s = bsummary[metric]
        print(f"  {metric}: 95% CI [{s['ci_lower_2_5']:.4f}, {s['ci_upper_97_5']:.4f}]  "
              f"({s['direction_note']})")

    print("\nStep 6: Writing comparison audit ...")
    audit = {
        "comparison_timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "research_question": (
            "Under the same training protocol (Milestone 9e recipe: focal loss "
            "gamma=2.0/label_smoothing=0.2/alpha disabled, batch=32, Keras "
            "per-step exponential LR decay from 0.005, patience=10), does Morgan/"
            "ECFP4 outperform frozen ChemBERTa for PolyLLM multi-label side-effect "
            "prediction?"
        ),
        "morgan_config": morgan_cfg,
        "chemberta_config": chemberta_cfg,
        "comparability_validated": True,
        "comparable_fields": COMPARABLE_FIELDS,
        "intentional_differences": INTENTIONAL_DIFFERENCES,
        "morgan_thresholds": MORGAN_THRESHOLD,
        "chemberta_thresholds": CHEMBERTA_THRESHOLD,
        "win_counts_per_label": win_counts,
        "bootstrap_summary": bsummary,
        "ap_at_50_paper_axis": {
            "morgan": morgan_ap50k["mean_ap_at_k"],
            "chemberta": chemberta_ap50k["mean_ap_at_k"],
            "difference": morgan_ap50k["mean_ap_at_k"] - chemberta_ap50k["mean_ap_at_k"],
        },
        "source_artifact_hashes": {
            "morgan_test_metrics": cm._sha256(MORGAN_DIR / "test_metrics.json"),
            "chemberta_test_metrics": cm._sha256(CHEMBERTA_DIR / "test_metrics.json"),
            "morgan_predictions": cm._sha256(MORGAN_DIR / "test_predictions.npz"),
            "chemberta_predictions": cm._sha256(CHEMBERTA_DIR / "test_predictions.npz"),
        },
    }
    audit_path = OUT_DIR / "comparison_audit.json"
    audit_path.write_text(json.dumps(audit, indent=2))
    print(f"  Saved {audit_path.name}.")

    print("\n-- Comparison complete --")
    print(f"  Output directory: {OUT_DIR}")


if __name__ == "__main__":
    main()
