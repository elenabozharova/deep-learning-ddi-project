"""
LLM prompting experiment -- zero-shot / few-shot small-LM baseline vs. the
official GNN and MLP checkpoints, on a matched 600-triple sample of the GNN's
official test split.

Methodological precedent: De Vito, Ferrucci & Angelakis, "LLMs for drug-drug
interaction prediction using textual drug descriptors" (Knowledge-Based
Systems, 2026) -- zero-shot -> few-shot progression, small-sample
justification, robustness-checking philosophy. This script does NOT attempt
their fine-tuned-Phi-3.5 result: no fine-tuning (ruled out as CPU-infeasible),
a much smaller local model (~0.5B params), and a much smaller sample (600
triples vs. their full test set).

Reuses, unmodified:
    train_gnn.py        -- DEFAULT_CONFIG, GRAPH_PATH, load_graph_split,
                            build_link_loader, build_model, set_seeds
    models/gnn.py        -- build_global_positive_edge_set, sample_negative_edges
                            (assemble_supervision_edges is deliberately NOT
                            reused for scoring the fixed 600-edge sample --
                            see run_gnn_on_sample docstring for why)
    evaluate_gnn.py       -- compute_test_metrics (identical AUC/AUPRC/AP@50
                            formulas for every method below, for a fair
                            comparison)
    metrics.py            -- edge_level_average_precision_at_k (via
                            compute_test_metrics)
    models/mlp.py         -- MultilabelMLP
    training.py           -- load_checkpoint

Additive only -- does not modify train_gnn.py, evaluate_gnn.py, models/gnn.py,
models/mlp.py, training.py, or anything under outputs/polyllm/gnn/ or
outputs/polyllm/chemberta_faithful_training/ (both frozen official results).

Official run:
    .venv-polyllm/Scripts/python.exe src/polyllm/diagnostics/llm_prompting_experiment.py

Smoke test (10+10 sample, 4 robustness triples, same code path):
    .venv-polyllm/Scripts/python.exe src/polyllm/diagnostics/llm_prompting_experiment.py --smoke-test
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

_src = Path(__file__).resolve().parents[2]
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

from polyllm.evaluate_gnn import compute_test_metrics  # noqa: E402
from polyllm.graph.build_graph import EDGE_TYPE  # noqa: E402
from polyllm.models.gnn import build_global_positive_edge_set, sample_negative_edges  # noqa: E402
from polyllm.models.mlp import MultilabelMLP  # noqa: E402
from polyllm.train_gnn import (  # noqa: E402
    DEFAULT_CONFIG,
    GRAPH_PATH,
    build_link_loader,
    build_model,
    load_graph_split,
    set_seeds,
)
from polyllm.training import load_checkpoint  # noqa: E402

# ---------------------------------------------------------------------------
# Fixed paths
# ---------------------------------------------------------------------------

GNN_OUTPUT_DIR   = Path("outputs/polyllm/gnn")
GNN_CHECKPOINT   = GNN_OUTPUT_DIR / "checkpoints/best_model.pt"
GNN_CONFIG_PATH  = GNN_OUTPUT_DIR / "training_config.json"
GNN_TEST_METRICS = GNN_OUTPUT_DIR / "test_metrics.json"

MLP_OUTPUT_DIR  = Path("outputs/polyllm/chemberta_faithful_training")
MLP_CHECKPOINT  = MLP_OUTPUT_DIR / "checkpoints/best_model.pt"
MLP_CONFIG_PATH = MLP_OUTPUT_DIR / "training_config.json"
MLP_TRAIN_IDS   = Path("data/splits/train_pair_ids.csv")
MLP_VAL_IDS     = Path("data/splits/validation_pair_ids.csv")
MLP_TEST_IDS    = Path("data/splits/test_pair_ids.csv")

FEATURES_PATH  = Path("data/features/chemberta_pair_embeddings.npy")
PAIRS_PATH     = Path("data/processed/polyllm_pairs.parquet")
LABEL_MAP_PATH = Path("data/processed/polyllm_label_mapping.csv")
DRUG_MAP_PATH  = Path("data/processed/drug_smiles_mapping.csv")

OUTPUT_DIR              = Path("outputs/polyllm/llm_prompting")
SAMPLE_TRIPLES_CSV      = OUTPUT_DIR / "sample_triples.csv"
ZERO_SHOT_CSV           = OUTPUT_DIR / "zero_shot_predictions.csv"
FEW_SHOT_CSV            = OUTPUT_DIR / "few_shot_predictions.csv"
ROBUSTNESS_CSV          = OUTPUT_DIR / "robustness_order_swap.csv"
SUMMARY_JSON            = OUTPUT_DIR / "summary.json"

# ---------------------------------------------------------------------------
# Experiment constants
# ---------------------------------------------------------------------------

SAMPLE_SEED = DEFAULT_CONFIG["random_seed"]  # 42 -- this repo's standard seed
N_POS_DEFAULT = 300
N_NEG_DEFAULT = 300
FEW_SHOT_K_POS = 3
FEW_SHOT_K_NEG = 3
ROBUSTNESS_N_DEFAULT = 80
ROBUSTNESS_FLIP_PROB_THRESHOLD = 0.1  # |p_yes delta| beyond this counts as a "large change"

LLM_MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

SYSTEM_PROMPT = (
    "You are a pharmacology assistant. You will be given two drugs and a candidate "
    "side effect. Decide whether taking the two drugs together is known to cause that "
    "side effect. Respond with exactly one word: Yes or No."
)

YES_WORDS = ["Yes", " Yes", "yes", " yes"]
NO_WORDS = ["No", " No", "no", " no"]


# ---------------------------------------------------------------------------
# Step 1 -- reconstruct the official test edge pool and draw the sample
# ---------------------------------------------------------------------------

def build_full_test_edge_pool(test_data, global_pos_edges: set[tuple[int, int]], seed: int):
    """
    Reconstruct the GNN's official test edge population: every positive
    supervision edge in test_data (literally the official held-out test
    positives) plus one fresh draw of negatives via the SAME reused
    `sample_negative_edges` the official evaluate_gnn.py run uses internally.

    NOTE on "true subset of the actual official test edges": evaluate_gnn.py
    never persists the y_true/y_pred arrays from its own run (only aggregate
    metrics went to test_metrics.json), and it samples negatives fresh every
    run with no fixed eval-time seed (no set_seeds() call in
    evaluate_gnn.run_evaluation) -- so there is no single canonical 855K-edge
    array on disk to sub-sample from. Positives here ARE literally the
    official test positive edges (test_data[EDGE_TYPE].edge_label_index, a
    strict subset by construction). Negatives are generated via the identical
    negative-sampling code path (assemble_supervision_edges's own
    sample_negative_edges + build_global_positive_edge_set), under a fixed,
    documented seed for this script's own reproducibility -- not literally
    the same negative draw the official run happened to make.
    """
    pos_edge_index = test_data[EDGE_TYPE].edge_label_index  # [2, n_pos], global indices
    assert torch.all(test_data[EDGE_TYPE].edge_label == 1), "test split contract violated: expected all-positive edge_label"

    set_seeds(seed)
    neg_edge_index = sample_negative_edges(test_data, global_pos_edges, device=torch.device("cpu"))

    pos_pdrugs = pos_edge_index[0].numpy()
    pos_seffect = pos_edge_index[1].numpy()
    neg_pdrugs = neg_edge_index[0].numpy()
    neg_seffect = neg_edge_index[1].numpy()

    # Sanity: positives are true positives, negatives are not, per the exact
    # same global positive-edge set the GNN training/eval pipeline uses.
    assert all((int(p), int(s)) in global_pos_edges for p, s in zip(pos_pdrugs[:1000], pos_seffect[:1000]))
    assert all((int(p), int(s)) not in global_pos_edges for p, s in zip(neg_pdrugs, neg_seffect))

    return pos_pdrugs, pos_seffect, neg_pdrugs, neg_seffect


def draw_stratified_sample(
    pos_pdrugs: np.ndarray, pos_seffect: np.ndarray,
    neg_pdrugs: np.ndarray, neg_seffect: np.ndarray,
    n_pos: int, n_neg: int, seed: int,
) -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    pos_idx = rng.choice(len(pos_pdrugs), size=n_pos, replace=False)
    neg_idx = rng.choice(len(neg_pdrugs), size=n_neg, replace=False)

    rows = []
    for i in pos_idx:
        rows.append({"pdrugs_node_id": int(pos_pdrugs[i]), "seffect_node_id": int(pos_seffect[i]), "true_label": 1})
    for i in neg_idx:
        rows.append({"pdrugs_node_id": int(neg_pdrugs[i]), "seffect_node_id": int(neg_seffect[i]), "true_label": 0})

    df = pd.DataFrame(rows).reset_index(drop=True)
    df.insert(0, "sample_idx", df.index)
    # Shuffle row order (still fully determined by `seed`) so pos/neg aren't block-ordered.
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    df["sample_idx"] = df.index
    return df


def attach_names(df: pd.DataFrame, pairs_df: pd.DataFrame, drug_map: pd.DataFrame, label_map: pd.DataFrame) -> pd.DataFrame:
    """pdrugs_node_id == pair_id (row in polyllm_pairs.parquet); seffect_node_id
    == label_index (row in polyllm_label_mapping.csv) -- see module docstring
    of graph/build_graph.py: node_id is torch.arange(num_nodes), and both
    feature arrays are built in exactly that row order."""
    drug_lookup = drug_map.set_index("stitch_id")

    def _drug_name(stitch_id: str) -> tuple[str, str]:
        row = drug_lookup.loc[stitch_id]
        name = row["preferred_name"]
        if isinstance(name, str) and name.strip():
            return name, "pubchem_name"
        return f"SMILES:{row['standardized_smiles']}", "smiles_fallback"

    pairs_indexed = pairs_df.set_index("pair_id")
    label_indexed = label_map.set_index("label_index")

    out_rows = []
    for _, r in df.iterrows():
        pair_row = pairs_indexed.loc[r["pdrugs_node_id"]]
        se_row = label_indexed.loc[r["seffect_node_id"]]
        drug_1_name, drug_1_src = _drug_name(pair_row["drug_1"])
        drug_2_name, drug_2_src = _drug_name(pair_row["drug_2"])
        out_rows.append({
            "drug_1_stitch": pair_row["drug_1"], "drug_2_stitch": pair_row["drug_2"],
            "drug_1_name": drug_1_name, "drug_1_name_source": drug_1_src,
            "drug_2_name": drug_2_name, "drug_2_name_source": drug_2_src,
            "side_effect_id": se_row["side_effect_id"], "side_effect_name": se_row["side_effect_name"],
        })
    return pd.concat([df.reset_index(drop=True), pd.DataFrame(out_rows)], axis=1)


# ---------------------------------------------------------------------------
# Step 2 -- matched-subsample baseline metrics (GNN, MLP)
# ---------------------------------------------------------------------------

def run_gnn_on_sample(sample_df: pd.DataFrame, full_data, test_data, device) -> tuple[dict, np.ndarray]:
    """
    Score the 600 fixed sample triples with the official trained GNN, via the
    same mini-batched LinkNeighborLoader machinery (build_link_loader,
    build_model) the official evaluate_gnn.py uses -- so predictions reflect
    the same [20,10]-hop neighbor-sampled message passing, not full-graph
    inference.

    Deviation from evaluate_gnn.collect_test_predictions: that function calls
    assemble_supervision_edges(), which treats a batch's baked-in
    edge_label_index as ALL-POSITIVE and appends its own freshly-sampled
    negatives on top (correct for the official test_data, whose edge_label
    IS all-ones). Our 600-edge sample already has a fixed 300/300 pos/neg
    edge_label baked in (constructed in Step 1) -- calling
    assemble_supervision_edges here would incorrectly re-label everything
    positive and inject 600 MORE unwanted random negatives. Instead we read
    the loader's own (edge_label_index, edge_label) for our fixed edges
    directly, which is the same tensor plumbing assemble_supervision_edges
    itself reads from, just skipping its "assume all-positive, add
    negatives" step.
    """
    gnn_config = json.loads(GNN_CONFIG_PATH.read_text())
    model = build_model(full_data, gnn_config, device)
    model.load_state_dict(torch.load(GNN_CHECKPOINT, map_location=device, weights_only=True))
    model.eval()

    sample_small = test_data.clone()
    sample_small[EDGE_TYPE].edge_label_index = torch.tensor(
        np.stack([sample_df["pdrugs_node_id"].to_numpy(), sample_df["seffect_node_id"].to_numpy()]),
        dtype=torch.long,
    )
    sample_small[EDGE_TYPE].edge_label = torch.tensor(sample_df["true_label"].to_numpy(), dtype=torch.float32)

    loader = build_link_loader(
        sample_small, batch_size=gnn_config["eval_batch_size"],
        num_neighbors=gnn_config["num_neighbors"], shuffle=False,
    )

    all_pred, all_label, all_global_pdrugs, all_global_seffect = [], [], [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            x_dict = {"pdrugs": batch["pdrugs"].x, "seffect": batch["seffect"].x}
            node_id_dict = {"pdrugs": batch["pdrugs"].node_id, "seffect": batch["seffect"].node_id}
            local_eli = batch[EDGE_TYPE].edge_label_index
            pred = model(x_dict, batch.edge_index_dict, node_id_dict, local_eli)

            all_pred.append(pred.cpu())
            all_label.append(batch[EDGE_TYPE].edge_label.cpu())
            all_global_pdrugs.append(node_id_dict["pdrugs"][local_eli[0]].cpu())
            all_global_seffect.append(node_id_dict["seffect"][local_eli[1]].cpu())

    y_pred = torch.cat(all_pred).numpy()
    y_true = torch.cat(all_label).numpy()
    global_pdrugs = torch.cat(all_global_pdrugs).numpy()
    global_seffect = torch.cat(all_global_seffect).numpy()

    # Sanity: the loader's recovered (global pdrugs, global seffect, label)
    # triples are exactly our input sample, as a set (order across batches
    # need not match the input order).
    recovered = set(zip(global_pdrugs.tolist(), global_seffect.tolist(), y_true.astype(int).tolist()))
    expected = set(zip(sample_df["pdrugs_node_id"], sample_df["seffect_node_id"], sample_df["true_label"]))
    assert recovered == expected, "GNN mini-batch loader did not preserve the fixed sample edges"

    # Re-order y_pred to match sample_df's row order for CSV/robustness use downstream.
    order_key = {(int(p), int(s), int(l)): idx for idx, (p, s, l) in enumerate(zip(global_pdrugs, global_seffect, y_true))}
    ordered_pred = np.array([
        y_pred[order_key[(row.pdrugs_node_id, row.seffect_node_id, row.true_label)]]
        for row in sample_df.itertuples()
    ])

    metrics = compute_test_metrics(sample_df["true_label"].to_numpy(), ordered_pred)
    return metrics, ordered_pred


def run_mlp_on_sample(sample_df: pd.DataFrame, device) -> tuple[dict, np.ndarray, dict]:
    mlp_config = json.loads(MLP_CONFIG_PATH.read_text())
    model = MultilabelMLP(
        input_dim=mlp_config["input_dim"], output_dim=mlp_config["output_dim"],
        dropout=mlp_config["dropout"], negative_slope=mlp_config.get("leaky_relu_negative_slope", 0.1),
    ).to(device)
    load_checkpoint(MLP_CHECKPOINT, model, device=device)
    model.eval()

    features = np.load(FEATURES_PATH, mmap_mode="r")
    unique_pair_ids = np.unique(sample_df["pdrugs_node_id"].to_numpy())
    x = torch.from_numpy(features[unique_pair_ids].astype(np.float32)).to(device)
    with torch.no_grad():
        probs = torch.sigmoid(model(x)).cpu().numpy()  # (n_unique_pairs, 963)
    row_for_pair = {pid: i for i, pid in enumerate(unique_pair_ids)}

    y_score = np.array([
        probs[row_for_pair[row.pdrugs_node_id], row.seffect_node_id]
        for row in sample_df.itertuples()
    ])
    metrics = compute_test_metrics(sample_df["true_label"].to_numpy(), y_score)

    train_ids = set(pd.read_csv(MLP_TRAIN_IDS)["pair_id"].tolist())
    val_ids = set(pd.read_csv(MLP_VAL_IDS)["pair_id"].tolist())
    test_ids = set(pd.read_csv(MLP_TEST_IDS)["pair_id"].tolist())
    overlap = {
        "sample_pairs_in_mlp_train_split": int(sum(1 for p in unique_pair_ids if p in train_ids)),
        "sample_pairs_in_mlp_val_split": int(sum(1 for p in unique_pair_ids if p in val_ids)),
        "sample_pairs_in_mlp_test_split": int(sum(1 for p in unique_pair_ids if p in test_ids)),
        "n_unique_sample_pairs": int(len(unique_pair_ids)),
    }
    return metrics, y_score, overlap


# ---------------------------------------------------------------------------
# Step 3 -- LLM zero-shot / few-shot prompting
# ---------------------------------------------------------------------------

def build_few_shot_examples(train_data, full_data, global_pos_edges, pairs_df, drug_map, label_map, seed: int):
    """Draw FEW_SHOT_K_POS positive + FEW_SHOT_K_NEG negative examples from
    the TRAIN split only (never test). Positives: train_data's own disjoint
    supervision edges (all label=1 by the split's own contract). Negatives:
    sample_negative_edges() on train_data, filtered against the SAME global
    positive-edge set used everywhere else -- guaranteed not to collide with
    any true edge in train, val, OR test."""
    pos_ei = train_data[EDGE_TYPE].edge_label_index
    assert torch.all(train_data[EDGE_TYPE].edge_label == 1)

    set_seeds(seed)
    neg_ei = sample_negative_edges(train_data, global_pos_edges, device=torch.device("cpu"))

    rng = np.random.RandomState(seed)
    pos_pick = rng.choice(pos_ei.size(1), size=FEW_SHOT_K_POS, replace=False)
    neg_pick = rng.choice(neg_ei.size(1), size=FEW_SHOT_K_NEG, replace=False)

    pairs_indexed = pairs_df.set_index("pair_id")
    label_indexed = label_map.set_index("label_index")
    drug_lookup = drug_map.set_index("stitch_id")

    def _name(stitch_id: str) -> str:
        row = drug_lookup.loc[stitch_id]
        if isinstance(row["preferred_name"], str) and row["preferred_name"].strip():
            return row["preferred_name"]
        return f"SMILES:{row['standardized_smiles']}"

    examples = []
    for idx, label in [(i, 1) for i in pos_pick] + [(i, 0) for i in neg_pick]:
        ei = pos_ei if label == 1 else neg_ei
        pdrugs_id, seffect_id = int(ei[0, idx]), int(ei[1, idx])
        pair_row = pairs_indexed.loc[pdrugs_id]
        se_row = label_indexed.loc[seffect_id]
        examples.append({
            "pdrugs_node_id": pdrugs_id, "seffect_node_id": seffect_id, "label": label,
            "drug_a": _name(pair_row["drug_1"]), "drug_b": _name(pair_row["drug_2"]),
            "side_effect": se_row["side_effect_name"],
        })
    return examples


def question_text(drug_a: str, drug_b: str, side_effect: str) -> str:
    return (
        f"Drug A: {drug_a}\n"
        f"Drug B: {drug_b}\n"
        f"Candidate side effect: {side_effect}\n"
        f"Question: Does taking Drug A and Drug B together cause {side_effect}?\n"
        f"Answer:"
    )


def build_messages(few_shot_examples: list[dict] | None, drug_a: str, drug_b: str, side_effect: str) -> list[dict]:
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    if few_shot_examples:
        for ex in few_shot_examples:
            messages.append({"role": "user", "content": question_text(ex["drug_a"], ex["drug_b"], ex["side_effect"])})
            messages.append({"role": "assistant", "content": "Yes" if ex["label"] == 1 else "No"})
    messages.append({"role": "user", "content": question_text(drug_a, drug_b, side_effect)})
    return messages


def load_llm():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(LLM_MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(LLM_MODEL_NAME, dtype=torch.float32)
    model.eval()
    return tok, model


def _first_token_ids(tok, words: list[str]) -> list[int]:
    ids = set()
    for w in words:
        enc = tok.encode(w, add_special_tokens=False)
        if enc:
            ids.add(enc[0])
    return sorted(ids)


@torch.no_grad()
def score_yes_no(model, tok, messages: list[dict], yes_ids: list[int], no_ids: list[int]) -> float:
    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tok(text, return_tensors="pt")
    logits = model(**inputs).logits[0, -1, :]
    yes_logit = torch.logsumexp(logits[yes_ids], dim=0)
    no_logit = torch.logsumexp(logits[no_ids], dim=0)
    p_yes = F.softmax(torch.stack([yes_logit, no_logit]), dim=0)[0].item()
    return p_yes


def run_llm_pass(
    sample_df: pd.DataFrame, tok, model, yes_ids, no_ids,
    few_shot_examples: list[dict] | None, swap_drugs: bool = False,
    label: str = "zero_shot",
) -> pd.DataFrame:
    rows = []
    t0 = time.time()
    for i, row in enumerate(sample_df.itertuples()):
        drug_a, drug_b = row.drug_1_name, row.drug_2_name
        if swap_drugs:
            drug_a, drug_b = drug_b, drug_a
        messages = build_messages(few_shot_examples, drug_a, drug_b, row.side_effect_name)
        p_yes = score_yes_no(model, tok, messages, yes_ids, no_ids)
        rows.append({
            "sample_idx": row.sample_idx, "pdrugs_node_id": row.pdrugs_node_id,
            "seffect_node_id": row.seffect_node_id, "true_label": row.true_label,
            "p_yes": p_yes, "pred_label": int(p_yes >= 0.5),
        })
        if (i + 1) % 100 == 0:
            dt = time.time() - t0
            print(f"    [{label}] {i + 1}/{len(sample_df)}  ({dt:.0f}s elapsed, {dt / (i + 1):.2f}s/prompt)")
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Step 4 -- robustness check (drug-order swap)
# ---------------------------------------------------------------------------

def run_robustness_check(
    sample_df: pd.DataFrame, few_shot_df: pd.DataFrame, tok, model, yes_ids, no_ids,
    few_shot_examples: list[dict], n: int, seed: int,
) -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    idx = rng.choice(len(sample_df), size=min(n, len(sample_df)), replace=False)
    subset = sample_df.iloc[idx].reset_index(drop=True)

    swapped = run_llm_pass(subset, tok, model, yes_ids, no_ids, few_shot_examples, swap_drugs=True, label="robustness_swap")
    original_lookup = few_shot_df.set_index("sample_idx")["p_yes"]

    rows = []
    for r in swapped.itertuples():
        p_original = float(original_lookup.loc[r.sample_idx])
        p_swapped = r.p_yes
        rows.append({
            "sample_idx": r.sample_idx, "pdrugs_node_id": r.pdrugs_node_id, "seffect_node_id": r.seffect_node_id,
            "true_label": r.true_label, "p_yes_original_order": p_original, "p_yes_swapped_order": p_swapped,
            "abs_diff": abs(p_original - p_swapped),
            "decision_flipped": int((p_original >= 0.5) != (p_swapped >= 0.5)),
            "large_change": int(abs(p_original - p_swapped) > ROBUSTNESS_FLIP_PROB_THRESHOLD),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="LLM prompting experiment vs. official GNN/MLP checkpoints.")
    p.add_argument("--smoke-test", action="store_true", help="Tiny sample (10+10, robustness n=4) for a quick end-to-end check.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    n_pos = 10 if args.smoke_test else N_POS_DEFAULT
    n_neg = 10 if args.smoke_test else N_NEG_DEFAULT
    robustness_n = 4 if args.smoke_test else ROBUSTNESS_N_DEFAULT

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cpu")

    print(f"=== LLM prompting experiment ===  smoke_test={args.smoke_test}")
    print(f"Sample seed: {SAMPLE_SEED}  n_pos={n_pos}  n_neg={n_neg}  robustness_n={robustness_n}")

    print("\nLoading graph split + reference tables ...")
    saved = load_graph_split(GRAPH_PATH)
    full_data, train_data, test_data = saved["full"], saved["train"], saved["test"]
    global_pos_edges = build_global_positive_edge_set(full_data)
    pairs_df = pd.read_parquet(PAIRS_PATH)
    label_map = pd.read_csv(LABEL_MAP_PATH)
    drug_map = pd.read_csv(DRUG_MAP_PATH)

    print("Reconstructing the official test edge pool (positives = literal test "
          "positives; negatives = fresh draw via the reused sample_negative_edges) ...")
    pos_pdrugs, pos_seffect, neg_pdrugs, neg_seffect = build_full_test_edge_pool(test_data, global_pos_edges, SAMPLE_SEED)
    print(f"  test positive pool: {len(pos_pdrugs)}  (official test_metrics.json n_positive="
          f"{json.loads(GNN_TEST_METRICS.read_text())['n_positive']})")
    print(f"  test negative pool (this run's fresh draw): {len(neg_pdrugs)}")

    sample_df = draw_stratified_sample(pos_pdrugs, pos_seffect, neg_pdrugs, neg_seffect, n_pos, n_neg, SAMPLE_SEED)
    sample_df = attach_names(sample_df, pairs_df, drug_map, label_map)
    sample_df.to_csv(SAMPLE_TRIPLES_CSV, index=False)
    print(f"  sample: {len(sample_df)} triples ({sample_df['true_label'].sum()} pos / "
          f"{(sample_df['true_label'] == 0).sum()} neg) -> {SAMPLE_TRIPLES_CSV}")

    print("\nScoring sample with the official GNN checkpoint ...")
    gnn_metrics, gnn_scores = run_gnn_on_sample(sample_df, full_data, test_data, device)
    print(f"  GNN-on-sample: {gnn_metrics}")

    print("\nScoring sample with the official MLP checkpoint ...")
    mlp_metrics, mlp_scores, mlp_overlap = run_mlp_on_sample(sample_df, device)
    print(f"  MLP-on-sample: {mlp_metrics}")
    print(f"  MLP split overlap of sampled pairs: {mlp_overlap}")

    print(f"\nLoading LLM: {LLM_MODEL_NAME} ...")
    tok, model = load_llm()
    yes_ids = _first_token_ids(tok, YES_WORDS)
    no_ids = _first_token_ids(tok, NO_WORDS)
    print(f"  yes_ids={yes_ids}  no_ids={no_ids}")

    print("\nBuilding few-shot examples from TRAIN split only ...")
    few_shot_examples = build_few_shot_examples(train_data, full_data, global_pos_edges, pairs_df, drug_map, label_map, SAMPLE_SEED)
    few_shot_keys = {(e["pdrugs_node_id"], e["seffect_node_id"]) for e in few_shot_examples}
    sample_keys = set(zip(sample_df["pdrugs_node_id"], sample_df["seffect_node_id"]))
    assert not (few_shot_keys & sample_keys), "few-shot example leaked into the 600-triple sample"
    for e in few_shot_examples:
        print(f"  [{e['label']}] {e['drug_a']} + {e['drug_b']} -> {e['side_effect']}")

    print(f"\nRunning zero-shot over {len(sample_df)} triples ...")
    zero_shot_df = run_llm_pass(sample_df, tok, model, yes_ids, no_ids, few_shot_examples=None, label="zero_shot")
    zero_shot_df.to_csv(ZERO_SHOT_CSV, index=False)
    zero_shot_metrics = compute_test_metrics(zero_shot_df["true_label"].to_numpy(), zero_shot_df["p_yes"].to_numpy())
    print(f"  zero-shot: {zero_shot_metrics}")

    print(f"\nRunning few-shot ({len(few_shot_examples)} examples) over {len(sample_df)} triples ...")
    few_shot_df = run_llm_pass(sample_df, tok, model, yes_ids, no_ids, few_shot_examples=few_shot_examples, label="few_shot")
    few_shot_df.to_csv(FEW_SHOT_CSV, index=False)
    few_shot_metrics = compute_test_metrics(few_shot_df["true_label"].to_numpy(), few_shot_df["p_yes"].to_numpy())
    print(f"  few-shot: {few_shot_metrics}")

    print(f"\nRunning drug-order-swap robustness check on {robustness_n} triples (few-shot prompt) ...")
    robustness_df = run_robustness_check(
        sample_df, few_shot_df, tok, model, yes_ids, no_ids, few_shot_examples, robustness_n, SAMPLE_SEED
    )
    robustness_df.to_csv(ROBUSTNESS_CSV, index=False)
    flip_rate = float(robustness_df["decision_flipped"].mean())
    large_change_rate = float(robustness_df["large_change"].mean())
    print(f"  flip rate (0.5 decision boundary): {flip_rate:.3f}")
    print(f"  large-change rate (|delta p_yes| > {ROBUSTNESS_FLIP_PROB_THRESHOLD}): {large_change_rate:.3f}")

    import sklearn
    import transformers as _transformers
    gnn_full_test = json.loads(GNN_TEST_METRICS.read_text())["metrics"]

    summary = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "smoke_test": args.smoke_test,
        "sample": {
            "seed": SAMPLE_SEED, "n_pos": int(n_pos), "n_neg": int(n_neg), "n_total": len(sample_df),
            "source": "official GNN test split (data/graph/gnn_link_split.pt, test_data); positives are literal "
                      "test-split positive edges; negatives are a fresh draw via the reused sample_negative_edges "
                      "(no canonical persisted 855K-edge array exists to sub-sample from -- see build_full_test_edge_pool docstring)",
            "sample_triples_csv": str(SAMPLE_TRIPLES_CSV),
        },
        "metrics": {
            "gnn_full_test_855448_edges": gnn_full_test,
            "gnn_on_sample": gnn_metrics,
            "mlp_on_sample": mlp_metrics,
            "llm_zero_shot": zero_shot_metrics,
            "llm_few_shot": few_shot_metrics,
        },
        "mlp_sample_split_overlap": mlp_overlap,
        "few_shot_examples": few_shot_examples,
        "robustness_order_swap": {
            "n": len(robustness_df), "flip_rate_at_0.5": flip_rate,
            "large_change_rate": large_change_rate, "large_change_threshold": ROBUSTNESS_FLIP_PROB_THRESHOLD,
            "mean_abs_diff": float(robustness_df["abs_diff"].mean()),
            "csv": str(ROBUSTNESS_CSV),
        },
        "llm_model": LLM_MODEL_NAME,
        "probability_extraction": (
            "logsumexp over first-subword-token logits of {Yes,\" Yes\",yes,\" yes\"} vs "
            "{No,\" No\",no,\" no\"}, softmax-normalized to p(yes) -- a genuine calibrated "
            "probability from the model's own next-token distribution, not a binary fallback."
        ),
        "checkpoints_used": {"gnn": str(GNN_CHECKPOINT), "mlp": str(MLP_CHECKPOINT)},
        "software_versions": {
            "python": platform.python_version(), "torch": torch.__version__,
            "transformers": _transformers.__version__, "scikit_learn": sklearn.__version__,
            "pandas": pd.__version__, "numpy": np.__version__,
        },
    }
    SUMMARY_JSON.write_text(json.dumps(summary, indent=2, default=str))
    print(f"\nSummary saved -> {SUMMARY_JSON}")
    print("\nDone.")


if __name__ == "__main__":
    main()
