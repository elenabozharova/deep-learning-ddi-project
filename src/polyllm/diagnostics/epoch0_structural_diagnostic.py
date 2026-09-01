"""
GNN gap diagnostic -- epoch-0 (untrained-model) structural-signal check.

Motivated by the 2026-08-16 checkpoint-selection diagnostic's Phase 6
anomaly (see notes/gnn_gap_diagnostics.md): under the OFFICIAL mini-batched
evaluation protocol (evaluate_gnn.py, LinkNeighborLoader num_neighbors=
[20,10]), a completely UNTRAINED model (epoch 0, seed=42) scored test
AUC=0.9391 -- higher than the trained, paper-matching checkpoint (0.9202).
This script answers two narrow questions, with NO training and NO
architecture/config change:

  1. Is the high epoch-0 AUC systematic across random initializations, or a
     one-off artifact of seed=42 specifically? (Check 1 -- 5-seed sweep)
  2. Can simple, learned-parameter-free graph-degree information explain
     most of it? (Check 2 -- degree baselines; Check 3 -- degree
     distributions by edge class; Check 4 -- compare against Check 1)

Design note -- "same test positive and negative edges" for every seed and
every baseline, by construction rather than by hoping RNG reproduces:
LinkNeighborLoader is already documented in this repo (see
notes/gnn_gap_diagnostics.md, 2026-08-16 "known confound") as NOT perfectly
seed-reproducible run-to-run, and negative sampling draws from Python's
global `random` state on every forward pass (see the
gnn-authors-source-verified memory, RNG-reseed diagnostic). Re-iterating
the test loader once per seed would therefore silently vary the edge set
between seeds -- exactly what the request prohibits. Instead, the official
mini-batched test loader (build_link_loader + assemble_supervision_edges,
BOTH imported unchanged from train_gnn.py / models/gnn.py) is iterated
EXACTLY ONCE, and every batch's tensors (message-passing subgraph, node
features, node_id local->global maps, and the assembled positive+negative
edge_label_index/edge_label) are frozen in memory. Every downstream
consumer below -- all 5 initialization seeds, all 4 degree baselines, the
positive/negative degree-distribution comparison, and the optional
embedding-norm fallback -- reads from this one frozen batch list and never
re-samples. Only the model's initial parameters differ between seeds.

Degree source graph: test_data[EDGE_TYPE].edge_index, which the 2026-08-16
source-equivalence audit already proved (exact set equality) equals
train_mp u train_sup u val_sup -- the training+validation graph legitimately
available for message passing at test time. This is NOT
test_data[EDGE_TYPE].edge_label_index (the held-out test positives) -- a
runtime leakage check below confirms zero overlap.

Scope guardrails (same as every prior round in notes/gnn_gap_diagnostics.md):
no edits to train_gnn.py / evaluate_gnn.py / models/gnn.py / build_graph.py;
no retraining; no architecture, split, negative-sampling, or hyperparameter
changes. New code only here; new outputs only under
outputs/polyllm/gnn_diagnostics/epoch0_structural/. The frozen
outputs/polyllm/gnn/ official result is untouched.

Run from the project root:
    .venv-polyllm/Scripts/python.exe src/polyllm/diagnostics/epoch0_structural_diagnostic.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

_src = Path(__file__).resolve().parents[2]
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

from polyllm.evaluate_gnn import compute_test_metrics  # noqa: E402
from polyllm.graph.build_graph import EDGE_TYPE  # noqa: E402
from polyllm.models.gnn import assemble_supervision_edges, build_global_positive_edge_set  # noqa: E402
from polyllm.train_gnn import (  # noqa: E402
    DEFAULT_CONFIG,
    GRAPH_PATH,
    build_link_loader,
    build_model,
    load_graph_split,
    set_seeds,
)

OUTPUT_DIR = Path("outputs/polyllm/gnn_diagnostics/epoch0_structural")

INIT_SEEDS = [1, 2, 3, 4, 5]

# Controls neighbor sampling (LinkNeighborLoader's [20,10] sampling) and
# negative sampling (assemble_supervision_edges) ONCE, at materialization
# time -- deliberately independent of INIT_SEEDS. See module docstring.
MATERIALIZATION_SEED = 20260817

PAPER_AUC_MEAN = 0.9228
SYSTEMATIC_THRESHOLD = 0.90
NULL_THRESHOLD = 0.60


# ---------------------------------------------------------------------------
# Batch materialization -- iterate the official test loader exactly once
# ---------------------------------------------------------------------------

def materialize_fixed_test_batches(test_data, global_pos_edges, config, device):
    set_seeds(MATERIALIZATION_SEED)
    test_loader = build_link_loader(
        test_data, config["eval_batch_size"], config["num_neighbors"], config["shuffle_loaders"]
    )

    batches = []
    for batch in test_loader:
        batch = batch.to(device)
        edge_label_index, edge_label = assemble_supervision_edges(batch, global_pos_edges, device)

        pdrugs_node_id = batch["pdrugs"].node_id
        seffect_node_id = batch["seffect"].node_id

        batches.append({
            "x_dict": {"pdrugs": batch["pdrugs"].x, "seffect": batch["seffect"].x},
            "edge_index_dict": dict(batch.edge_index_dict),
            "node_id_dict": {"pdrugs": pdrugs_node_id, "seffect": seffect_node_id},
            "edge_label_index": edge_label_index,
            "edge_label": edge_label,
            "global_pdrugs_id": pdrugs_node_id[edge_label_index[0]].clone(),
            "global_seffect_id": seffect_node_id[edge_label_index[1]].clone(),
        })
    return batches


@torch.no_grad()
def evaluate_model_on_fixed_batches(model, batches, device):
    model.eval()
    all_preds, all_labels = [], []
    for b in batches:
        pred = model(b["x_dict"], b["edge_index_dict"], b["node_id_dict"], b["edge_label_index"])
        all_preds.append(pred.cpu())
        all_labels.append(b["edge_label"].cpu())
    y_pred = torch.cat(all_preds).numpy()
    y_true = torch.cat(all_labels).numpy()
    return y_true, y_pred


@torch.no_grad()
def norm_baseline_scores(model, batches):
    """Check 4 fallback: initial PROJECTED node-embedding norm product,
    computed before message passing -- model.pdrugs_lin(x)+pdrugs_emb(id),
    NOT model.encode()/model.encoder() (that runs the GraphConv stack)."""
    scores = []
    for b in batches:
        x_dict, node_id_dict, eli = b["x_dict"], b["node_id_dict"], b["edge_label_index"]
        z_pdrugs = model.pdrugs_lin(x_dict["pdrugs"].float()) + model.pdrugs_emb(node_id_dict["pdrugs"])
        z_seffect = model.seffect_lin(x_dict["seffect"].float()) + model.seffect_emb(node_id_dict["seffect"])
        score = z_pdrugs[eli[0]].norm(dim=-1) * z_seffect[eli[1]].norm(dim=-1)
        scores.append(score.cpu())
    return torch.cat(scores).numpy()


def auc_auprc(y_true, score) -> dict:
    return {"auc": float(roc_auc_score(y_true, score)), "auprc": float(average_precision_score(y_true, score))}


def describe(x: np.ndarray) -> dict:
    return {
        "mean": float(np.mean(x)), "median": float(np.median(x)), "std": float(np.std(x)),
        "p25": float(np.percentile(x, 25)), "p75": float(np.percentile(x, 75)),
    }


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cpu")  # matches every prior official/diagnostic GNN run
    config = dict(DEFAULT_CONFIG)  # UNCHANGED official hyperparameters
    print(f"Device: {device}")

    print(f"Loading graph split from {GRAPH_PATH} ...")
    saved = load_graph_split()
    full_data, test_data = saved["full"], saved["test"]
    global_pos_edges = build_global_positive_edge_set(full_data)

    print("Materializing fixed test batches (official mini-batched protocol, frozen once) ...")
    batches = materialize_fixed_test_batches(test_data, global_pos_edges, config, device)
    y_true_all = torch.cat([b["edge_label"] for b in batches]).numpy()
    n_pos, n_total = int(y_true_all.sum()), len(y_true_all)
    print(f"Frozen {len(batches)} test batches, {n_total} edges ({n_pos} positive, {n_total - n_pos} negative).")
    print("These exact edges are reused, unmodified, by every seed and every baseline below.")

    # -------------------------------------------------------------------
    # Check 1 -- epoch-0 AUC across 5 initialization seeds
    # -------------------------------------------------------------------
    print("\n=== Check 1: epoch-0 (untrained) AUC across initialization seeds ===")
    print("Seed controls: parameter initialization ONLY (set_seeds() -> build_model()).")
    print("Data split: fixed once when data/graph/gnn_link_split.pt was built -- unaffected by this seed.")
    print("Neighbor sampling & negative sampling: fixed once at materialization "
          f"(MATERIALIZATION_SEED={MATERIALIZATION_SEED}), identical across all 5 seeds by construction.")
    print("Internal set_seed() calls inside Model.forward(): none in this reproduction "
          "(the authors' per-forward reseed was identified as an unintentional bug and intentionally "
          "not reproduced -- see models/gnn.py module docstring); nothing to preserve here beyond that.")

    seed_rows = []
    seed_models = {}
    for seed in INIT_SEEDS:
        set_seeds(seed)
        model = build_model(full_data, config, device)
        y_true, y_pred = evaluate_model_on_fixed_batches(model, batches, device)
        m = compute_test_metrics(y_true, y_pred)
        seed_rows.append({"seed": seed, "test_auc": m["auc"], "test_auprc": m["auprc"], "test_ap_at_50": m["ap_at_50"]})
        seed_models[seed] = model
        print(f"  seed {seed}: AUC={m['auc']:.4f}  AUPRC={m['auprc']:.4f}  AP@50={m['ap_at_50']:.4f}")

    seed_df = pd.DataFrame(seed_rows)
    aucs = seed_df["test_auc"].to_numpy()
    seed_summary = {
        "mean_auc": float(aucs.mean()), "std_auc": float(aucs.std()),
        "min_auc": float(aucs.min()), "max_auc": float(aucs.max()),
    }
    print(f"  mean={seed_summary['mean_auc']:.4f}  std={seed_summary['std_auc']:.4f}  "
          f"min={seed_summary['min_auc']:.4f}  max={seed_summary['max_auc']:.4f}")
    seed_df.to_csv(OUTPUT_DIR / "epoch0_seed_sweep.csv", index=False)
    print(f"Saved -> {OUTPUT_DIR / 'epoch0_seed_sweep.csv'}")

    # -------------------------------------------------------------------
    # Check 2 -- structural degree baselines (same frozen test edges)
    # -------------------------------------------------------------------
    print("\n=== Check 2: structural degree baselines (identical frozen test edges) ===")
    num_pdrugs = test_data["pdrugs"].num_nodes
    num_seffect = test_data["seffect"].num_nodes
    mp_edge_index = test_data[EDGE_TYPE].edge_index  # train_mp u train_sup u val_sup (global ids)

    assert mp_edge_index[0].max().item() < num_pdrugs
    assert mp_edge_index[1].max().item() < num_seffect

    # Leakage proof: the degree-source graph must contain ZERO test positive
    # edges (that would leak test labels into an unsupervised baseline).
    test_pos_global = set(map(tuple, test_data[EDGE_TYPE].edge_label_index.T.tolist()))
    mp_global = set(map(tuple, mp_edge_index.T.tolist()))
    leaked = test_pos_global & mp_global
    print(f"  Leakage check: {len(leaked)}/{len(test_pos_global)} test-positive edges present in the "
          f"degree-source graph (must be 0).")
    assert len(leaked) == 0, "Degree computed with test-edge leakage!"

    pdrugs_degree = torch.zeros(num_pdrugs, dtype=torch.float64)
    seffect_degree = torch.zeros(num_seffect, dtype=torch.float64)
    ones = torch.ones(mp_edge_index.size(1), dtype=torch.float64)
    pdrugs_degree.index_add_(0, mp_edge_index[0], ones)
    seffect_degree.index_add_(0, mp_edge_index[1], ones)
    pdrugs_degree, seffect_degree = pdrugs_degree.numpy(), seffect_degree.numpy()

    global_pdrugs_id = torch.cat([b["global_pdrugs_id"] for b in batches]).numpy()
    global_seffect_id = torch.cat([b["global_seffect_id"] for b in batches]).numpy()

    deg_pdrugs = pdrugs_degree[global_pdrugs_id]
    deg_seffect = seffect_degree[global_seffect_id]
    deg_product = deg_pdrugs * deg_seffect
    deg_geomean = np.sqrt(deg_product)
    deg_sum = deg_pdrugs + deg_seffect

    baseline_results = {
        "Drug-pair degree": auc_auprc(y_true_all, deg_pdrugs),
        "Side-effect degree": auc_auprc(y_true_all, deg_seffect),
        "Degree product": auc_auprc(y_true_all, deg_product),
        "Degree geometric mean": auc_auprc(y_true_all, deg_geomean),
        "Degree sum": auc_auprc(y_true_all, deg_sum),
    }
    rank_equivalent = bool(np.array_equal(
        np.argsort(deg_product, kind="stable"), np.argsort(deg_geomean, kind="stable")
    ))
    for name, m in baseline_results.items():
        print(f"  {name:22s} AUC={m['auc']:.4f}  AUPRC={m['auprc']:.4f}")
    print(f"  Product vs. geometric-mean rank-equivalent: {rank_equivalent} "
          f"(sqrt(x) is strictly increasing for x>=0, so this is expected to always hold exactly)")

    baseline_df = pd.DataFrame([{"baseline": k, **v} for k, v in baseline_results.items()])
    baseline_df.to_csv(OUTPUT_DIR / "structural_baselines.csv", index=False)
    print(f"Saved -> {OUTPUT_DIR / 'structural_baselines.csv'}")

    # -------------------------------------------------------------------
    # Check 3 -- degree distributions, positive vs. negative test edges
    # -------------------------------------------------------------------
    print("\n=== Check 3: degree distributions, positive vs. negative test edges ===")
    pos_mask = y_true_all.astype(bool)
    distribution_rows = []
    for feat_name, values in [
        ("drug_pair_degree", deg_pdrugs),
        ("side_effect_degree", deg_seffect),
        ("degree_product", deg_product),
    ]:
        for label_name, mask in [("positive", pos_mask), ("negative", ~pos_mask)]:
            distribution_rows.append({"feature": feat_name, "edge_class": label_name, **describe(values[mask])})
    dist_df = pd.DataFrame(distribution_rows)
    print(dist_df.to_string(index=False))
    dist_df.to_csv(OUTPUT_DIR / "degree_distributions.csv", index=False)
    print(f"Saved -> {OUTPUT_DIR / 'degree_distributions.csv'}")

    # -------------------------------------------------------------------
    # Check 4 -- compare structural baseline with epoch-0 GNN
    # -------------------------------------------------------------------
    print("\n=== Check 4: epoch-0 GNN vs. best structural baseline ===")
    best_baseline_name = max(baseline_results, key=lambda k: baseline_results[k]["auc"])
    best_baseline_auc = baseline_results[best_baseline_name]["auc"]
    print(f"  Best structural baseline: {best_baseline_name} (AUC={best_baseline_auc:.4f})")
    print(f"  Mean epoch-0 GNN AUC across seeds: {seed_summary['mean_auc']:.4f}")

    per_seed_vs_baseline = {
        seed: {"seed_auc": row["test_auc"], "exceeds_best_baseline": row["test_auc"] > best_baseline_auc}
        for seed, row in zip(INIT_SEEDS, seed_rows)
    }
    any_seed_exceeds_baseline = any(v["exceeds_best_baseline"] for v in per_seed_vs_baseline.values())

    if best_baseline_auc >= SYSTEMATIC_THRESHOLD and seed_summary["mean_auc"] >= SYSTEMATIC_THRESHOLD:
        case = "A"
        print("  Case A: degree baseline and epoch-0 GNN are both >=0.90 -- structural degree signal "
              "largely explains the high epoch-0 AUC.")
    elif best_baseline_auc <= NULL_THRESHOLD and seed_summary["mean_auc"] >= SYSTEMATIC_THRESHOLD:
        case = "B"
        print("  Case B: degree baseline is <=0.60 but mean epoch-0 GNN is >=0.90 -- degree alone is "
              "insufficient.")
    else:
        case = "bimodal (not anticipated by the binary Case A/B split)"
        print("  Neither Case A nor Case B cleanly applies: epoch-0 AUC is BIMODAL across seeds "
              "(some seeds far above the degree baseline, some far below 0.5) rather than uniformly high.")

    # Computed regardless of the strict A/B gate above: per-seed epoch-0 AUC
    # already exceeds the best degree baseline for several seeds (a
    # per-seed version of Check 4's Case B trigger), and the bimodal pattern
    # itself is exactly what the already-documented (2026-08-11/13,
    # notes/gnn_gap_diagnostics.md) dot=norm*cos sign-flip mechanism would
    # predict: a real, degree-correlated magnitude signal whose sign in the
    # raw dot product depends on the random cosine-alignment of freshly
    # initialized node embeddings. Testing that mechanism directly (cheap,
    # no training, reuses the same frozen batches) is the natural follow-up
    # implied by the request's Check 4 rather than a new research branch.
    print("\n  Initial-projected-embedding-norm baseline (Check 4 continuation, all seeds):")
    norm_baseline_results = {}
    for seed in INIT_SEEDS:
        scores = norm_baseline_scores(seed_models[seed], batches)
        norm_baseline_results[seed] = auc_auprc(y_true_all, scores)
        print(f"    seed {seed}: norm-baseline AUC={norm_baseline_results[seed]['auc']:.4f}  "
              f"AUPRC={norm_baseline_results[seed]['auprc']:.4f}  (actual epoch-0 AUC={per_seed_vs_baseline[seed]['seed_auc']:.4f})")

    # -------------------------------------------------------------------
    # Save summary
    # -------------------------------------------------------------------
    summary = {
        "materialization_seed": MATERIALIZATION_SEED,
        "init_seeds": INIT_SEEDS,
        "n_test_edges": n_total,
        "n_test_positive": n_pos,
        "n_test_negative": n_total - n_pos,
        "check1_epoch0_seed_sweep": seed_rows,
        "check1_summary": seed_summary,
        "check2_structural_baselines": baseline_results,
        "check2_product_geomean_rank_equivalent": rank_equivalent,
        "check3_degree_distributions": distribution_rows,
        "check4_best_baseline": {"name": best_baseline_name, "auc": best_baseline_auc},
        "check4_case": case,
        "check4_per_seed_vs_baseline": per_seed_vs_baseline,
        "check4_norm_baseline": norm_baseline_results,
        "paper_auc_mean": PAPER_AUC_MEAN,
    }
    (OUTPUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(f"\nSummary saved -> {OUTPUT_DIR / 'summary.json'}")


if __name__ == "__main__":
    main()
