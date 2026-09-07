"""
Fetch PubChem *common/trade* names (via the Synonyms endpoint) for the drugs
actually used in the LLM prompting experiment, to replace the systematic
IUPAC names currently used in its prompts.

Trigger: `map_drugs_to_smiles.py` only ever requested PubChem's `IUPACName`
property (see its PUBCHEM_BATCH_URL / PUBCHEM_SINGLE_URL, "/property/
IsomericSMILES,IUPACName,InChIKey/JSON") -- the systematic chemical name, not
a common/trade name. `llm_prompting_experiment.py`'s notes (Limitation #3)
already flag this as a hard ceiling on what any LLM can recognize. This
script does not touch that pipeline or its output (`drug_smiles_mapping.csv`
is read-only here) -- it produces a small, separate, additive name lookup for
just the ~404 drugs actually used in the LLM experiment's sample + few-shot
examples, via a *different* PubChem endpoint
(/compound/cid/{cid}/synonyms/JSON) that this repo has never called before.

Heuristic for "best" common name: PubChem's synonym lists are unordered
across sources and often lead with CAS numbers, DrugBank IDs, or the same
systematic name -- not reliably the trade name. We filter out registry-number-
shaped and systematic-name-shaped candidates and keep the shortest plausible
plain-word candidate. This is a heuristic, not a guarantee -- spot-checked
manually in the accompanying note, not proven correct for every drug.

Output: outputs/polyllm/llm_prompting/common_name_lookup.csv
        (stitch_id, common_name, source: pubchem_synonym | fallback_iupac)

Run:
    .venv-polyllm/Scripts/python.exe src/polyllm/diagnostics/fetch_common_drug_names.py
"""
from __future__ import annotations

import re
import sys
import time
from pathlib import Path

import pandas as pd

_src = Path(__file__).resolve().parents[2]
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

from polyllm.data.map_drugs_to_smiles import _get_with_retry  # noqa: E402

SAMPLE_CSV = Path("outputs/polyllm/llm_prompting/sample_triples.csv")
PAIRS_PATH = Path("data/processed/polyllm_pairs.parquet")
SUMMARY_JSON = Path("outputs/polyllm/llm_prompting/summary.json")
DRUG_MAP_PATH = Path("data/processed/drug_smiles_mapping.csv")
OUT_CSV = Path("outputs/polyllm/llm_prompting/common_name_lookup.csv")

SYNONYMS_URL = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/synonyms/JSON"
REQUEST_DELAY = 0.25  # seconds between requests -- polite to PubChem, still fast for ~400 CIDs

# Candidate rejected if it looks like a registry number / ID / systematic name.
_REJECT_PATTERNS = [
    re.compile(r"^\d+-\d+-\d+$"),           # CAS number, e.g. 50-78-2
    re.compile(r"^DB\d+$", re.I),            # DrugBank ID
    re.compile(r"^CHEBI[:\-]?\d+$", re.I),   # ChEBI ID
    re.compile(r"^CHEMBL\d+$", re.I),        # ChEMBL ID
    re.compile(r"^UNII[- ]?[A-Z0-9]+$", re.I),
    re.compile(r"^NSC[- ]?\d+$", re.I),
    re.compile(r"^EINECS", re.I),
    re.compile(r"^\d+$"),                    # bare number
    re.compile(r"[\[\]{}]"),                 # brackets -> systematic name
    re.compile(r"\d.*-.*\d"),                # digit-hyphen-digit pattern typical of systematic names
]


def _looks_like_common_name(candidate: str) -> bool:
    if not candidate or len(candidate) > 40:
        return False
    if any(p.search(candidate) for p in _REJECT_PATTERNS):
        return False
    # Systematic names are long strings of alternating letters/numbers/hyphens;
    # a plausible common name is mostly letters, allows spaces/hyphens.
    letters = sum(c.isalpha() for c in candidate)
    if letters / max(len(candidate), 1) < 0.7:
        return False
    return True


def pick_common_name(synonyms: list[str]) -> str | None:
    candidates = [s for s in synonyms if _looks_like_common_name(s)]
    if not candidates:
        return None
    # Shortest plausible candidate tends to be the everyday/trade name rather
    # than a longer semi-systematic synonym.
    return min(candidates, key=len)


def fetch_synonyms(cid: int) -> list[str]:
    resp = _get_with_retry(SYNONYMS_URL.format(cid=cid))
    if resp is None or resp.status_code != 200:
        return []
    try:
        info = resp.json()["InformationList"]["Information"][0]
        return info.get("Synonym", [])
    except (KeyError, IndexError, ValueError):
        return []


def main() -> None:
    sample = pd.read_csv(SAMPLE_CSV)
    sample_cids = set(sample["drug_1_stitch"]) | set(sample["drug_2_stitch"])

    import json
    with open(SUMMARY_JSON, encoding="utf-8") as f:
        summary = json.load(f)
    pairs_df = pd.read_parquet(PAIRS_PATH).set_index("pair_id")
    fewshot_cids = set()
    for ex in summary["few_shot_examples"]:
        row = pairs_df.loc[ex["pdrugs_node_id"]]
        fewshot_cids.add(row["drug_1"])
        fewshot_cids.add(row["drug_2"])

    all_cids = sorted(sample_cids | fewshot_cids)
    print(f"Fetching synonyms for {len(all_cids)} unique drugs "
          f"({len(sample_cids)} from sample, {len(fewshot_cids)} from few-shot examples)...")

    drug_map = pd.read_csv(DRUG_MAP_PATH).set_index("stitch_id")

    rows = []
    for i, stitch_id in enumerate(all_cids):
        cid_num = int(stitch_id.replace("CID", "").lstrip("0") or "0")
        synonyms = fetch_synonyms(cid_num)
        common = pick_common_name(synonyms)
        fallback = drug_map.loc[stitch_id, "preferred_name"] if stitch_id in drug_map.index else None
        rows.append({
            "stitch_id": stitch_id,
            "cid": cid_num,
            "common_name": common or fallback,
            "source": "pubchem_synonym" if common else "fallback_iupac",
            "n_synonyms_returned": len(synonyms),
        })
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(all_cids)}")
        time.sleep(REQUEST_DELAY)

    out_df = pd.DataFrame(rows)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(OUT_CSV, index=False)

    n_found = (out_df["source"] == "pubchem_synonym").sum()
    print(f"\nDone. {n_found}/{len(out_df)} drugs got a plausible common name from PubChem synonyms.")
    print(f"Saved -> {OUT_CSV}")
    print("\nSample of results:")
    print(out_df.sample(min(10, len(out_df)), random_state=42)[["stitch_id", "common_name", "source"]].to_string(index=False))


if __name__ == "__main__":
    main()
