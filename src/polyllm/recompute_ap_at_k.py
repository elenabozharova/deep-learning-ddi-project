"""
Milestone 9b — Recompute AP@50 using the PolyLLM paper author's exact
per-label semantics, against the already-saved Milestone 6B/7 test
predictions.

Analysis and documentation only — no training, retraining, inference, or
modification of any existing artifact. This script never opens a model
checkpoint; it reads only the frozen test_predictions.npz files that
evaluate_morgan_baseline.py and evaluate_chemberta_mlp.py already wrote,
plus the frozen label matrix, and writes one new sibling JSON file per
model. test_metrics.json, test_predictions.npz, and both checkpoints are
never opened for writing.

Background
----------
The reproduction's own sample_mean_ap_at_50() (src/polyllm/metrics.py)
loops over PAIRS and ranks the 963 LABELS within each pair, keeping the
top 50 labels per pair. Its value on the ChemBERTa test set (0.3795) sits
49.8% below the paper's reported Table 5 value (0.7557 +/- 0.0120),
flagged as "likely_incomparable" in notes/deviations_from_paper.md Sec 1.3
and outputs/comparison/paper_comparison_audit.json.

The paper author's own reference snippet (average_precision_at_k_multi_label
in src/polyllm/metrics.py) loops over LABELS and ranks all PAIRS within
each label, keeping the top 50 pairs per label — the opposite axis. This
script recomputes AP@50 under that corrected axis for both trained models
and reports the result, resolving (or narrowing) the discrepancy without
touching any frozen artifact.

Usage:
    .venv-polyllm/Scripts/python.exe src/polyllm/recompute_ap_at_k.py
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from polyllm.metrics import average_precision_at_k_multi_label  # noqa: E402

MORGAN_DIR = ROOT / "outputs" / "baseline" / "morgan"
CHEMBERTA_DIR = ROOT / "outputs" / "polyllm" / "chemberta"
CHEMBERTA_FOCAL_LRDECAY_DIR = ROOT / "outputs" / "polyllm" / "chemberta_focal_lrdecay"
CHEMBERTA_FAITHFUL_DIR = ROOT / "outputs" / "polyllm" / "chemberta_faithful_training"
LABELS_PATH = ROOT / "data" / "processed" / "polyllm_labels.npy"

K = 50

MODELS = [
    ("morgan", MORGAN_DIR, "Morgan MLP (Milestone 6B)"),
    ("chemberta", CHEMBERTA_DIR, "ChemBERTa MLP (Milestone 7/9c)"),
    ("chemberta_focal_lrdecay", CHEMBERTA_FOCAL_LRDECAY_DIR, "ChemBERTa MLP - Focal loss + LR decay (Milestone 9d)"),
    ("chemberta_faithful", CHEMBERTA_FAITHFUL_DIR, "ChemBERTa MLP - Faithful training (Milestone 9e)"),
]

# Paper's reported value for the model this reproduction targets
# (Table 5, "DeepChem ChemBERTa MLP", 10-fold cross-validation mean).
PAPER_CHEMBERTA_AP50_MEAN = 0.7557
PAPER_CHEMBERTA_AP50_STD = 0.0120


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_predictions(npz_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Read an already-saved test_predictions.npz. No inference performed."""
    data = np.load(npz_path)
    return data["probabilities"], data["pair_ids"]


def recompute_for_model(model_dir: Path, labels_path: Path, k: int = K) -> dict[str, Any]:
    pred_path = model_dir / "test_predictions.npz"
    old_metrics_path = model_dir / "test_metrics.json"

    probs, pair_ids = load_predictions(pred_path)
    labels = np.load(labels_path, mmap_mode="r")
    y_true = np.asarray(labels[pair_ids]).astype(np.int64)

    if y_true.shape != probs.shape:
        raise ValueError(
            f"Shape mismatch loading {pred_path}: y_true {y_true.shape} vs "
            f"probabilities {probs.shape}"
        )

    result = average_precision_at_k_multi_label(y_true, probs, k=k)
    old_metrics = json.loads(old_metrics_path.read_text())

    return {
        "recomputation_timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "scope_note": (
            "Recompute-only. No model checkpoint was loaded, no inference "
            "was run, and no existing artifact (test_metrics.json, "
            "test_predictions.npz, checkpoints/*) was modified."
        ),
        "metric_name": "average_precision_at_k_multi_label",
        "metric_source": (
            "PolyLLM paper author's reference implementation (per-label "
            "axis: for each label, rank all pairs, keep the top-k, compute "
            "sklearn average_precision_score on those k rows)."
        ),
        "k": k,
        "n_pairs_evaluated": int(y_true.shape[0]),
        "n_labels_evaluated": int(y_true.shape[1]),
        "mean_ap_at_k": result["mean_ap_at_k"],
        "degenerate_all_positive_labels": result["degenerate_all_positive_labels"],
        "degenerate_all_negative_labels": result["degenerate_all_negative_labels"],
        "per_label_ap_at_k_summary": {
            "min": float(result["per_label_ap_at_k"].min()),
            "median": float(np.median(result["per_label_ap_at_k"])),
            "max": float(result["per_label_ap_at_k"].max()),
        },
        "comparison_to_existing_repro_metric": {
            "existing_metric_name": "sample_mean_ap_at_50",
            "existing_metric_value": old_metrics.get("sample_mean_ap_at_50"),
            "existing_metric_axis": "per-pair (ranks labels within each pair)",
            "this_metric_axis": "per-label (ranks pairs within each label)",
        },
        "source_artifact_hashes": {
            "test_predictions_npz": _sha256(pred_path),
            "test_metrics_json": _sha256(old_metrics_path),
            "labels_npy": _sha256(labels_path),
        },
    }


def main() -> None:
    print(f"Recomputing AP@{K} (paper author's per-label semantics) from saved predictions...")
    print("No checkpoints loaded, no inference run, no existing artifact modified.\n")

    all_results: dict[str, dict[str, Any]] = {}

    for key, model_dir, label in MODELS:
        print(f"-- {label} --")
        result = recompute_for_model(model_dir, LABELS_PATH, k=K)
        out_path = model_dir / "ap_at_k_recompute.json"
        out_path.write_text(json.dumps(result, indent=2))
        all_results[key] = result

        print(f"  mean_ap_at_{K}: {result['mean_ap_at_k']:.4f}")
        print(f"  (existing per-pair-axis sample_mean_ap_at_50: "
              f"{result['comparison_to_existing_repro_metric']['existing_metric_value']:.4f})")
        print(f"  degenerate top-{K} columns: "
              f"{result['degenerate_all_positive_labels']} all-positive, "
              f"{result['degenerate_all_negative_labels']} all-negative "
              f"(of {result['n_labels_evaluated']} labels)")
        print(f"  saved -> {out_path}\n")

    print("-- Summary vs. paper (Table 5, DeepChem ChemBERTa MLP) --")
    print(f"  paper (10-fold mean +/- std):            "
          f"{PAPER_CHEMBERTA_AP50_MEAN:.4f} +/- {PAPER_CHEMBERTA_AP50_STD:.4f}")
    for key, _model_dir, label in MODELS:
        if not key.startswith("chemberta"):
            continue  # paper's Table 5 row is ChemBERTa-only; Morgan is context, not a comparison target
        val = all_results[key]["mean_ap_at_k"]
        std_devs = (val - PAPER_CHEMBERTA_AP50_MEAN) / PAPER_CHEMBERTA_AP50_STD
        old_axis = all_results[key]["comparison_to_existing_repro_metric"]["existing_metric_value"]
        print(f"  {label}:")
        print(f"    old per-pair-axis metric:              {old_axis:.4f}")
        print(f"    corrected per-label-axis metric:       {val:.4f}  ({std_devs:+.2f} std devs from paper mean)")


if __name__ == "__main__":
    main()
