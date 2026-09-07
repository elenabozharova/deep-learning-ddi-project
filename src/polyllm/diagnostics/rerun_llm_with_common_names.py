"""
Re-run the LLM prompting experiment's zero-shot / few-shot passes on the
EXACT SAME 600-triple sample and 6 few-shot examples as
llm_prompting_experiment.py, substituting PubChem common/trade names
(fetch_common_drug_names.py) for the systematic IUPAC names used originally.

Does not redo GNN/MLP scoring (those don't depend on drug names -- their
numbers from outputs/polyllm/llm_prompting/summary.json are reused as-is)
or the robustness check (out of scope for this specific comparison: does
using a recognizable drug name change the LLM's predictions).

Prerequisite: fetch_common_drug_names.py already run
(outputs/polyllm/llm_prompting/common_name_lookup.csv exists).

Run:
    .venv-polyllm/Scripts/python.exe src/polyllm/diagnostics/rerun_llm_with_common_names.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pandas as pd

_src = Path(__file__).resolve().parents[2]
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

from polyllm.diagnostics.llm_prompting_experiment import (  # noqa: E402
    LLM_MODEL_NAME,
    NO_WORDS,
    YES_WORDS,
    _first_token_ids,
    load_llm,
    run_llm_pass,
)
from polyllm.evaluate_gnn import compute_test_metrics  # noqa: E402

ORIG_DIR = Path("outputs/polyllm/llm_prompting")
SAMPLE_CSV = ORIG_DIR / "sample_triples.csv"
LOOKUP_CSV = ORIG_DIR / "common_name_lookup.csv"
ORIG_SUMMARY_JSON = ORIG_DIR / "summary.json"
PAIRS_PATH = Path("data/processed/polyllm_pairs.parquet")

OUT_DIR = Path("outputs/polyllm/llm_prompting_common_names")
ZERO_SHOT_CSV = OUT_DIR / "zero_shot_predictions.csv"
FEW_SHOT_CSV = OUT_DIR / "few_shot_predictions.csv"
SUMMARY_JSON = OUT_DIR / "summary.json"


def load_common_names() -> dict[str, str]:
    lookup = pd.read_csv(LOOKUP_CSV)
    return dict(zip(lookup["stitch_id"], lookup["common_name"]))


def patch_sample_names(sample_df: pd.DataFrame, common: dict[str, str]) -> pd.DataFrame:
    df = sample_df.copy()
    df["drug_1_name_common"] = df["drug_1_stitch"].map(common).fillna(df["drug_1_name"])
    df["drug_2_name_common"] = df["drug_2_stitch"].map(common).fillna(df["drug_2_name"])
    n_changed_1 = (df["drug_1_name_common"] != df["drug_1_name"]).sum()
    n_changed_2 = (df["drug_2_name_common"] != df["drug_2_name"]).sum()
    print(f"  drug_1 names changed: {n_changed_1}/{len(df)}   drug_2 names changed: {n_changed_2}/{len(df)}")
    # run_llm_pass reads row.drug_1_name / row.drug_2_name directly -- swap in the common names.
    df["drug_1_name_iupac"] = df["drug_1_name"]
    df["drug_2_name_iupac"] = df["drug_2_name"]
    df["drug_1_name"] = df["drug_1_name_common"]
    df["drug_2_name"] = df["drug_2_name_common"]
    return df


def patch_few_shot_examples(orig_examples: list[dict], common: dict[str, str], pairs_df: pd.DataFrame) -> list[dict]:
    pairs_indexed = pairs_df.set_index("pair_id")
    patched = []
    for ex in orig_examples:
        pair_row = pairs_indexed.loc[ex["pdrugs_node_id"]]
        stitch_a, stitch_b = pair_row["drug_1"], pair_row["drug_2"]
        patched.append({
            **ex,
            "drug_a": common.get(stitch_a, ex["drug_a"]),
            "drug_b": common.get(stitch_b, ex["drug_b"]),
        })
    return patched


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("Loading original sample, few-shot examples, and common-name lookup ...")
    sample_df = pd.read_csv(SAMPLE_CSV)
    with open(ORIG_SUMMARY_JSON, encoding="utf-8") as f:
        orig_summary = json.load(f)
    pairs_df = pd.read_parquet(PAIRS_PATH)
    common = load_common_names()

    print("Patching sample names ...")
    sample_df = patch_sample_names(sample_df, common)
    print("Patching few-shot example names ...")
    few_shot_examples = patch_few_shot_examples(orig_summary["few_shot_examples"], common, pairs_df)
    for e in few_shot_examples:
        print(f"  [{e['label']}] {e['drug_a']} + {e['drug_b']} -> {e['side_effect']}")

    print(f"\nLoading LLM: {LLM_MODEL_NAME} ...")
    tok, model = load_llm()
    yes_ids = _first_token_ids(tok, YES_WORDS)
    no_ids = _first_token_ids(tok, NO_WORDS)

    print(f"\nRunning zero-shot (common names) over {len(sample_df)} triples ...")
    zero_shot_df = run_llm_pass(sample_df, tok, model, yes_ids, no_ids, few_shot_examples=None, label="zero_shot_common")
    zero_shot_df.to_csv(ZERO_SHOT_CSV, index=False)
    zero_shot_metrics = compute_test_metrics(zero_shot_df["true_label"].to_numpy(), zero_shot_df["p_yes"].to_numpy())
    print(f"  zero-shot (common names): {zero_shot_metrics}")

    print(f"\nRunning few-shot (common names) over {len(sample_df)} triples ...")
    few_shot_df = run_llm_pass(sample_df, tok, model, yes_ids, no_ids, few_shot_examples=few_shot_examples, label="few_shot_common")
    few_shot_df.to_csv(FEW_SHOT_CSV, index=False)
    few_shot_metrics = compute_test_metrics(few_shot_df["true_label"].to_numpy(), few_shot_df["p_yes"].to_numpy())
    print(f"  few-shot (common names): {few_shot_metrics}")

    orig_metrics = orig_summary["metrics"]
    summary = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "comparison": "Same 600-triple sample and same 6 few-shot examples as outputs/polyllm/llm_prompting/, "
                      "only the drug name strings in the prompt changed (systematic IUPAC name -> PubChem "
                      "synonym-derived common name where found).",
        "n_drug_slots_with_a_common_name_found": int((pd.read_csv(LOOKUP_CSV)["source"] == "pubchem_synonym").sum()),
        "n_drug_slots_total": int(len(pd.read_csv(LOOKUP_CSV))),
        "metrics_iupac_names_original": {
            "llm_zero_shot": orig_metrics["llm_zero_shot"],
            "llm_few_shot": orig_metrics["llm_few_shot"],
        },
        "metrics_common_names": {
            "llm_zero_shot": zero_shot_metrics,
            "llm_few_shot": few_shot_metrics,
        },
        "llm_model": LLM_MODEL_NAME,
    }
    with open(SUMMARY_JSON, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nSaved -> {SUMMARY_JSON}")
    print("\n=== Comparison ===")
    print(f"{'Metric':<10} {'IUPAC 0-shot':>14} {'Common 0-shot':>14} {'IUPAC few-shot':>16} {'Common few-shot':>16}")
    for m in ("auc", "auprc", "ap_at_50"):
        print(f"{m:<10} {orig_metrics['llm_zero_shot'][m]:>14.4f} {zero_shot_metrics[m]:>14.4f} "
              f"{orig_metrics['llm_few_shot'][m]:>16.4f} {few_shot_metrics[m]:>16.4f}")


if __name__ == "__main__":
    main()
