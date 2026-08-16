"""
GNN gap diagnostic 1 — node/edge/embedding alignment audit.

For a deterministic sample of positive (pdrugs, seffect) edges from the
saved graph split, prove that graph node IDs, the original drug pair /
side-effect identities, the embedding-matrix row IDs, and the exact string
used to generate the side-effect BERT embedding all refer to the same
entity. This is a pure consistency check — no model is loaded, nothing is
trained, and the official pipeline files are not imported or modified.

Alignment this script assumes (verified by hand against the actual CSVs
before writing this script, not guessed):
  - pdrugs node ID == pair_id == row in chemberta_pair_embeddings.npy
    (`data/features/chemberta_pair_index.csv`: embedding_row == pair_id for
    all 63,472 rows, per notes/pair_embedding_findings.md; and
    graph/build_graph.py sets data["pdrugs"].node_id = torch.arange(n) while
    loading pdrugs_x directly from that same npy in row order).
  - seffect node ID == label_index == embedding_index == row in
    bert_side_effect_embeddings.npy (same torch.arange construction in
    graph/build_graph.py; bert_side_effect_index.csv's embedding_index and
    label_index columns were spot-checked equal for rows 0-3 before writing
    this script).
  - The side-effect name actually used to generate the BERT embedding is
    only stored as a SHA-256 hash in bert_side_effect_index.csv
    (side_effect_name_sha256), computed in
    features/generate_side_effect_embeddings.py as
    hashlib.sha256(name.encode("utf-8")).hexdigest() — reproduced exactly
    here to independently verify the name, not re-derive it from index
    position alone.

Run from the project root:
    .venv-polyllm/Scripts/python.exe -m polyllm.diagnostics.verify_node_alignment
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

_src = Path(__file__).resolve().parents[2]
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

from polyllm.graph.build_graph import EDGE_TYPE  # noqa: E402

# ---------------------------------------------------------------------------
# Fixed paths
# ---------------------------------------------------------------------------

GRAPH_PATH = Path("data/graph/gnn_link_split.pt")
PAIR_INDEX_PATH = Path("data/features/chemberta_pair_index.csv")
PAIR_EMBED_PATH = Path("data/features/chemberta_pair_embeddings.npy")
LABEL_MAP_PATH = Path("data/processed/polyllm_label_mapping.csv")
SEFFECT_INDEX_PATH = Path("data/features/bert_side_effect_index.csv")
SEFFECT_EMBED_PATH = Path("data/features/bert_side_effect_embeddings.npy")
LABELS_PATH = Path("data/processed/polyllm_labels.npy")

OUTPUT_PATH = Path("outputs/polyllm/gnn_diagnostics/node_alignment_audit.json")

N_SAMPLES = 20
SAMPLE_SEED = 20260813  # arbitrary, fixed for reproducibility; recorded in output


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def audit_one_edge(
    pdrugs_id: int,
    seffect_id: int,
    full_data,
    pair_index_df: pd.DataFrame,
    label_map_df: pd.DataFrame,
    seffect_index_df: pd.DataFrame,
    pair_embed: np.ndarray,
    seffect_embed: np.ndarray,
    labels: np.ndarray,
) -> dict[str, Any]:
    checks: dict[str, bool] = {}
    row: dict[str, Any] = {"pdrugs_node_id": pdrugs_id, "seffect_node_id": seffect_id}

    # --- pdrugs side -------------------------------------------------------
    pair_row = pair_index_df.loc[pair_index_df["pair_id"] == pdrugs_id]
    checks["pair_index_row_exists"] = len(pair_row) == 1
    pair_row = pair_row.iloc[0]
    row["drug_1"] = str(pair_row["drug_1"])
    row["drug_2"] = str(pair_row["drug_2"])
    row["pair_embedding_row_id"] = int(pair_row["embedding_row"])
    checks["pair_embedding_row_equals_node_id"] = int(pair_row["embedding_row"]) == pdrugs_id

    graph_pdrugs_x = full_data["pdrugs"].x[pdrugs_id].numpy()
    source_pdrugs_x = pair_embed[pdrugs_id]
    checks["pdrugs_feature_bitwise_match"] = bool(np.array_equal(graph_pdrugs_x, source_pdrugs_x))

    # --- seffect side --------------------------------------------------------
    label_row = label_map_df.loc[label_map_df["label_index"] == seffect_id]
    checks["label_mapping_row_exists"] = len(label_row) == 1
    label_row = label_row.iloc[0]
    row["side_effect_id"] = str(label_row["side_effect_id"])
    row["side_effect_name"] = str(label_row["side_effect_name"])

    seffect_idx_row = seffect_index_df.loc[seffect_index_df["embedding_index"] == seffect_id]
    checks["seffect_index_row_exists"] = len(seffect_idx_row) == 1
    seffect_idx_row = seffect_idx_row.iloc[0]
    row["side_effect_embedding_row_id"] = int(seffect_idx_row["embedding_index"])
    checks["seffect_embedding_row_equals_node_id"] = int(seffect_idx_row["embedding_index"]) == seffect_id
    checks["seffect_label_index_equals_node_id"] = int(seffect_idx_row["label_index"]) == seffect_id
    checks["seffect_id_matches_label_mapping"] = str(seffect_idx_row["side_effect_id"]) == str(label_row["side_effect_id"])

    computed_hash = _sha256(str(label_row["side_effect_name"]))
    row["name_sha256_recomputed"] = computed_hash
    row["name_sha256_on_disk"] = str(seffect_idx_row["side_effect_name_sha256"])
    checks["name_used_for_embedding_matches_label_mapping"] = computed_hash == str(seffect_idx_row["side_effect_name_sha256"])

    graph_seffect_x = full_data["seffect"].x[seffect_id].numpy()
    source_seffect_x = seffect_embed[seffect_id]
    checks["seffect_feature_bitwise_match"] = bool(np.array_equal(graph_seffect_x, source_seffect_x))

    # --- cross-check against the Milestone 1 label matrix itself -----------
    checks["edge_is_positive_in_polyllm_labels_npy"] = bool(labels[pdrugs_id, seffect_id] != 0)

    row["checks"] = checks
    row["all_passed"] = all(checks.values())
    return row


def main() -> None:
    print(f"Loading graph split from {GRAPH_PATH} ...")
    saved = torch.load(GRAPH_PATH, weights_only=False)
    full_data = saved["full"]

    print("Loading index/mapping/embedding artifacts ...")
    pair_index_df = pd.read_csv(PAIR_INDEX_PATH)
    label_map_df = pd.read_csv(LABEL_MAP_PATH)
    seffect_index_df = pd.read_csv(SEFFECT_INDEX_PATH)
    pair_embed = np.load(PAIR_EMBED_PATH)
    seffect_embed = np.load(SEFFECT_EMBED_PATH)
    labels = np.load(LABELS_PATH)

    edge_index = full_data[EDGE_TYPE].edge_index
    n_edges = edge_index.size(1)
    print(f"Full graph positive edges: {n_edges}")

    rng = np.random.default_rng(SAMPLE_SEED)
    sample_positions = rng.choice(n_edges, size=N_SAMPLES, replace=False)
    sample_positions.sort()

    rows: list[dict[str, Any]] = []
    for pos in sample_positions:
        pdrugs_id = int(edge_index[0, pos].item())
        seffect_id = int(edge_index[1, pos].item())
        row = audit_one_edge(
            pdrugs_id, seffect_id, full_data,
            pair_index_df, label_map_df, seffect_index_df,
            pair_embed, seffect_embed, labels,
        )
        rows.append(row)
        status = "OK" if row["all_passed"] else "FAIL"
        print(
            f"  [{status}] pdrugs={pdrugs_id} ({row['drug_1']}+{row['drug_2']})  "
            f"<-> seffect={seffect_id} ({row['side_effect_name']!r})"
        )
        if not row["all_passed"]:
            failed = [k for k, v in row["checks"].items() if not v]
            raise AssertionError(
                f"Alignment check failed for edge (pdrugs={pdrugs_id}, seffect={seffect_id}): {failed}\n"
                f"Full row: {json.dumps(row, indent=2, default=str)}"
            )

    result = {
        "n_samples": N_SAMPLES,
        "sample_seed": SAMPLE_SEED,
        "sample_edge_positions": sample_positions.tolist(),
        "all_edges_passed": all(r["all_passed"] for r in rows),
        "rows": rows,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    print(f"\nAll {N_SAMPLES} sampled edges passed every alignment check.")
    print(f"Results saved -> {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
