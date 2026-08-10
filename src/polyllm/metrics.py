"""
Reusable multi-label classification metrics for Milestones 6B and 7.

All functions accept numpy float32/int arrays.
Macro averages skip labels where the metric is mathematically undefined (NaN),
and report included/skipped counts rather than silently replacing NaN with 0.
"""
from __future__ import annotations

import warnings

import numpy as np
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score


# ---------------------------------------------------------------------------
# Per-label primitives
# ---------------------------------------------------------------------------

def per_label_auroc(y_true: np.ndarray, y_score: np.ndarray) -> np.ndarray:
    """AUROC per label.  NaN when only one class is present in y_true."""
    n_labels = y_true.shape[1]
    out = np.full(n_labels, np.nan)
    for j in range(n_labels):
        col = y_true[:, j]
        pos = int(col.sum())
        if pos == 0 or pos == len(col):
            continue
        out[j] = roc_auc_score(col, y_score[:, j])
    return out


def per_label_auprc(y_true: np.ndarray, y_score: np.ndarray) -> np.ndarray:
    """Average precision (AUPRC) per label.  NaN when no positive examples."""
    n_labels = y_true.shape[1]
    out = np.full(n_labels, np.nan)
    for j in range(n_labels):
        if y_true[:, j].sum() == 0:
            continue
        out[j] = average_precision_score(y_true[:, j], y_score[:, j])
    return out


def per_label_f1(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """F1 per label.  NaN only when TP+FP=0 AND TP+FN=0 (no activity)."""
    n_labels = y_true.shape[1]
    out = np.full(n_labels, np.nan)
    for j in range(n_labels):
        tp = int(((y_pred[:, j] == 1) & (y_true[:, j] == 1)).sum())
        fp = int(((y_pred[:, j] == 1) & (y_true[:, j] == 0)).sum())
        fn = int(((y_pred[:, j] == 0) & (y_true[:, j] == 1)).sum())
        if tp == 0 and fp == 0 and fn == 0:
            continue  # truly undefined: label absent and never predicted
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        out[j] = 0.0 if (prec + rec) == 0 else 2 * prec * rec / (prec + rec)
    return out


# ---------------------------------------------------------------------------
# Macro aggregates
# ---------------------------------------------------------------------------

def macro_auroc(y_true: np.ndarray, y_score: np.ndarray) -> dict:
    per = per_label_auroc(y_true, y_score)
    valid = ~np.isnan(per)
    return {
        "macro_auroc": float(per[valid].mean()) if valid.any() else float("nan"),
        "included_labels": int(valid.sum()),
        "skipped_labels": int((~valid).sum()),
    }


def macro_auprc(y_true: np.ndarray, y_score: np.ndarray) -> dict:
    per = per_label_auprc(y_true, y_score)
    valid = ~np.isnan(per)
    return {
        "macro_auprc": float(per[valid].mean()) if valid.any() else float("nan"),
        "included_labels": int(valid.sum()),
        "skipped_labels": int((~valid).sum()),
    }


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    per = per_label_f1(y_true, y_pred)
    valid = ~np.isnan(per)
    return {
        "macro_f1": float(per[valid].mean()) if valid.any() else float("nan"),
        "included_labels": int(valid.sum()),
        "skipped_labels": int((~valid).sum()),
    }


# ---------------------------------------------------------------------------
# Micro metrics
# ---------------------------------------------------------------------------

def micro_auprc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Micro AUPRC: flatten all labels and treat as a single PR curve."""
    return float(average_precision_score(y_true.ravel(), y_score.ravel()))


def micro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(f1_score(y_true, y_pred, average="micro", zero_division=0))


# ---------------------------------------------------------------------------
# Threshold selection
# ---------------------------------------------------------------------------

def apply_threshold(y_score: np.ndarray, threshold: float) -> np.ndarray:
    return (y_score >= threshold).astype(np.int32)


def select_threshold(
    y_true: np.ndarray,
    y_score: np.ndarray,
    thresholds: np.ndarray | None = None,
) -> dict:
    """
    Select the global threshold that maximises validation micro F1.

    Grid: 0.05, 0.06, …, 0.95 (step 0.01, 91 values).
    Tie-break: smallest threshold among equal micro-F1 values.
    """
    if thresholds is None:
        thresholds = np.round(np.arange(0.05, 0.951, 0.01), 2)

    best_f1 = -1.0
    best_thr = float(thresholds[0])
    grid_results: list[dict] = []

    for thr in thresholds:
        thr_val = float(round(float(thr), 4))
        y_pred = apply_threshold(y_score, thr_val)
        f1_val = float(f1_score(y_true, y_pred, average="micro", zero_division=0))
        grid_results.append({"threshold": thr_val, "micro_f1": f1_val})
        # strictly greater → first (smallest) occurrence of the best is kept
        if f1_val > best_f1:
            best_f1 = f1_val
            best_thr = thr_val

    return {
        "selected_threshold": best_thr,
        "validation_micro_f1": best_f1,
        "threshold_grid": grid_results,
        "tie_break_rule": "smallest threshold among equal micro-F1 values",
    }


# ---------------------------------------------------------------------------
# AP@50
# ---------------------------------------------------------------------------

def sample_mean_ap_at_50(
    y_true: np.ndarray,
    y_score: np.ndarray,
) -> float:
    """
    Operational AP@50 definition for the PolyLLM reproduction.

    For each pair:
    1. Rank all labels by predicted probability (descending).
    2. Keep the top 50.
    3. At every rank k (1-based) that contains a true label, compute
       precision@k = (# true labels in positions 1..k) / k.
    4. Sum those precision values.
    5. Divide by min(n_true_labels_for_pair, 50).
    6. Average across pairs.

    Pairs with zero true labels contribute 0.0.

    paper_ap50_definition_compatibility: unresolved — this definition has not
    been independently confirmed against the PolyLLM paper's formula.
    """
    n_samples = y_true.shape[0]
    ap_values = np.empty(n_samples, dtype=np.float64)

    for i in range(n_samples):
        true_i = y_true[i].astype(np.float64)
        score_i = y_score[i]
        n_true = int(true_i.sum())

        if n_true == 0:
            ap_values[i] = 0.0
            continue

        ranked = np.argsort(score_i)[::-1][:50]
        top_true = true_i[ranked]
        cumsum = np.cumsum(top_true)
        positions = np.arange(1, len(ranked) + 1, dtype=np.float64)
        precision_at_k = cumsum / positions
        ap_values[i] = (precision_at_k * top_true).sum() / min(n_true, 50)

    return float(ap_values.mean())


# ---------------------------------------------------------------------------
# AP@k — paper author's exact per-label semantics
# ---------------------------------------------------------------------------

def average_precision_at_k_multi_label(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    k: int = 50,
) -> dict:
    """Per-label AP@k, matching the PolyLLM paper author's reference snippet
    verbatim (translated from pandas ``.iloc`` row selection to numpy row
    indexing — both are positional, so the translation is exact, not
    approximate, given y_true and y_pred share row order):

        for i in range(y_true.shape[1]):
            sorted_indices = np.argsort(y_pred[:, i])[::-1][:k]
            sorted_true = y_true.iloc[sorted_indices, i]
            ap_at_k_label = average_precision_score(sorted_true, y_pred[sorted_indices, i])
            ap_at_k_list.append(ap_at_k_label)
        return np.mean(ap_at_k_list)

    This ranks over the opposite axis from sample_mean_ap_at_50() above: it
    loops over LABELS and, for each label, ranks all PAIRS by that label's
    score, keeping the top k pairs. sample_mean_ap_at_50() loops over pairs
    and ranks labels within each pair. The two are not interchangeable —
    see notes/deviations_from_paper.md Sec 1.3 for the reproduction history
    of this discrepancy (repro originally implemented the per-pair axis;
    the paper's reference code, given here, uses the per-label axis).

    Degenerate top-k columns (all k selected rows share one class) are not
    filtered out — sklearn's average_precision_score is called on them as
    given, matching the author's snippet exactly: an all-positive top-k
    column returns 1.0, an all-negative top-k column returns 0.0 (both
    verified against this sklearn version directly). Counts of each are
    returned for diagnostic purposes only; they do not alter the mean.

    Returns a dict (richer than the author's bare float return, for
    consistency with the other aggregate metrics in this module):
        mean_ap_at_k                 - float, the paper-semantics AP@k
        k                            - the k used
        n_labels                     - number of label columns
        per_label_ap_at_k            - (n_labels,) float array
        degenerate_all_positive_labels - int, top-k columns with no negatives
        degenerate_all_negative_labels - int, top-k columns with no positives
    """
    n_labels = y_true.shape[1]
    per_label = np.empty(n_labels, dtype=np.float64)
    degenerate_all_positive = 0
    degenerate_all_negative = 0

    for i in range(n_labels):
        top_k_idx = np.argsort(y_pred[:, i])[::-1][:k]
        top_k_true = y_true[top_k_idx, i]
        top_k_score = y_pred[top_k_idx, i]

        n_pos = int(top_k_true.sum())
        if n_pos == len(top_k_true):
            degenerate_all_positive += 1
        elif n_pos == 0:
            degenerate_all_negative += 1

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            per_label[i] = average_precision_score(top_k_true, top_k_score)

    return {
        "mean_ap_at_k": float(per_label.mean()),
        "k": k,
        "n_labels": n_labels,
        "per_label_ap_at_k": per_label,
        "degenerate_all_positive_labels": degenerate_all_positive,
        "degenerate_all_negative_labels": degenerate_all_negative,
    }
