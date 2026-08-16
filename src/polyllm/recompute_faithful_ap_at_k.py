"""
Controlled representation ablation — paper-compatible (per-label axis)
AP@50 for the Morgan faithful-training run.

Reuses average_precision_at_k_multi_label() (Milestone 9b) against the
already-saved outputs/baseline/morgan_faithful_training/test_predictions.npz.
Analysis only -- no checkpoint loaded, no inference run. Writes exactly one
new file, inside the new ablation's own output directory. Does not touch
outputs/polyllm/chemberta_faithful_training/ap_at_k_recompute.json, which
already contains the equivalent value for the M9e ChemBERTa run
(computed 2026-08-09) and is left as-is.

Usage:
    .venv-polyllm/Scripts/python.exe src/polyllm/recompute_faithful_ap_at_k.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from polyllm.metrics import average_precision_at_k_multi_label  # noqa: E402
from polyllm.recompute_ap_at_k import _sha256  # noqa: E402

MORGAN_FAITHFUL_DIR = ROOT / "outputs" / "baseline" / "morgan_faithful_training"
CHEMBERTA_FAITHFUL_DIR = ROOT / "outputs" / "polyllm" / "chemberta_faithful_training"
LABELS_PATH = ROOT / "data" / "processed" / "polyllm_labels.npy"
K = 50


def main() -> None:
    pred_path = MORGAN_FAITHFUL_DIR / "test_predictions.npz"
    data = np.load(pred_path)
    probs, pair_ids = data["probabilities"], data["pair_ids"]

    labels = np.load(LABELS_PATH, mmap_mode="r")
    y_true = np.asarray(labels[pair_ids]).astype(np.int64)

    result = average_precision_at_k_multi_label(y_true, probs, k=K)

    chemberta_existing = json.loads(
        (CHEMBERTA_FAITHFUL_DIR / "ap_at_k_recompute.json").read_text()
    )

    out = {
        "metric_name": "average_precision_at_k_multi_label",
        "metric_source": "PolyLLM paper author's reference implementation (per-label axis).",
        "k": K,
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
        "comparison_to_chemberta_m9e": {
            "chemberta_m9e_mean_ap_at_k": chemberta_existing["mean_ap_at_k"],
            "difference_morgan_minus_chemberta": result["mean_ap_at_k"] - chemberta_existing["mean_ap_at_k"],
        },
        "source_artifact_hashes": {
            "test_predictions_npz": _sha256(pred_path),
            "labels_npy": _sha256(LABELS_PATH),
        },
    }

    out_path = MORGAN_FAITHFUL_DIR / "ap_at_k_recompute.json"
    out_path.write_text(json.dumps(out, indent=2))

    print(f"Morgan (faithful recipe) paper-axis AP@{K}: {result['mean_ap_at_k']:.4f}")
    print(f"ChemBERTa M9e paper-axis AP@{K}:            {chemberta_existing['mean_ap_at_k']:.4f}")
    print(f"Difference (Morgan - ChemBERTa):             {out['comparison_to_chemberta_m9e']['difference_morgan_minus_chemberta']:+.4f}")
    print(f"Saved -> {out_path}")


if __name__ == "__main__":
    main()
