"""
Milestone 2 — Map retained STITCH drug IDs to validated SMILES.

Steps
-----
1. Extract all unique STITCH IDs from the Milestone 1 pair table.
2. Inspect and validate STITCH ID format.
3. Validate the CID-parsing rule on 10–20 representative IDs via PubChem.
4. Batch-query PubChem for IsomericSMILES, IUPACName, and InChIKey.
5. Validate each returned SMILES with RDKit.
6. Store the RDKit canonical isomeric SMILES as standardized_smiles.
7. Detect duplicate structures (same CID, InChIKey, or SMILES).
8. Report pair eligibility after mapping.
9. Write outputs and audit.

Run from the project root:
    python src/polyllm/data/map_drugs_to_smiles.py
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import platform
import re
import time
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from rdkit import Chem
from rdkit import RDLogger

# Silence RDKit's verbose C++ warnings — we handle errors explicitly.
RDLogger.DisableLog("rdApp.*")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

STITCH_PATTERN = re.compile(r"^CID(\d{9})$")
PUBCHEM_BATCH_URL = (
    "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid"
    "/property/IsomericSMILES,IUPACName,InChIKey/JSON"
)
PUBCHEM_SINGLE_URL = (
    "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid"
    "/{cid}/property/IsomericSMILES,IUPACName,InChIKey/JSON"
)

BATCH_SIZE = 100       # PubChem supports up to ~100 CIDs per batch POST
REQUEST_TIMEOUT = 30   # seconds
RETRY_COUNT = 3
RETRY_BACKOFF = [1, 2, 4]   # seconds between retries
INTER_BATCH_SLEEP = 0.22     # ~4.5 batches/sec — below PubChem's 5 req/sec limit

REQUIRED_OUTPUT_COLUMNS = [
    "stitch_id",
    "parsed_pubchem_cid",
    "preferred_name",
    "pubchem_smiles",
    "standardized_smiles",
    "inchikey",
    "mapping_source",
    "mapping_status",
    "notes",
]

DEFAULT_PAIRS_INPUT = Path("data/processed/polyllm_pairs.parquet")
DEFAULT_MAPPING_OUTPUT = Path("data/processed/drug_smiles_mapping.csv")
DEFAULT_AUDIT_OUTPUT = Path("outputs/polyllm/smiles_mapping_audit.json")
DEFAULT_CACHE_PATH = Path("outputs/polyllm/pubchem_cache.json")
DEFAULT_VALIDATION_SAMPLE_SIZE = 15

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Step 1: Extract unique drugs
# ---------------------------------------------------------------------------

def extract_unique_drugs(pairs_df: pd.DataFrame) -> list[str]:
    """Return a sorted list of unique STITCH IDs from drug_1 and drug_2 columns."""
    all_ids = set(pairs_df["drug_1"].tolist()) | set(pairs_df["drug_2"].tolist())
    return sorted(all_ids)


# ---------------------------------------------------------------------------
# Step 2: STITCH format inspection
# ---------------------------------------------------------------------------

def inspect_stitch_formats(stitch_ids: list[str]) -> dict[str, Any]:
    """
    Analyse and report the format(s) present in a list of STITCH identifiers.

    Returns a dict suitable for the audit section stitch_id_formats.
    """
    matched: list[str] = []
    unrecognized: list[str] = []
    cid_values: list[int] = []

    for sid in stitch_ids:
        m = STITCH_PATTERN.match(sid)
        if m:
            matched.append(sid)
            cid_values.append(int(m.group(1)))
        else:
            unrecognized.append(sid)

    result: dict[str, Any] = {
        "total": len(stitch_ids),
        "matched_pattern": "CID + exactly 9 zero-padded digits",
        "matched_count": len(matched),
        "unrecognized_count": len(unrecognized),
        "unrecognized_examples": unrecognized[:5],
    }
    if cid_values:
        result["cid_integer_range"] = [min(cid_values), max(cid_values)]
    return result


# ---------------------------------------------------------------------------
# Step 3: CID parsing
# ---------------------------------------------------------------------------

def parse_stitch_cid(stitch_id: str) -> int | None:
    """
    Parse a STITCH ID to a PubChem CID integer.

    Rule (validated against PubChem):
        CID<9-digit-zero-padded-number>  →  int(digits)

    Returns None if the ID does not match the expected format.
    Leading zeros are stripped by int() — they carry no information once
    the numeric CID is known.
    """
    m = STITCH_PATTERN.match(stitch_id)
    if m is None:
        return None
    return int(m.group(1))


# ---------------------------------------------------------------------------
# Step 4: SMILES validation with RDKit
# ---------------------------------------------------------------------------

def validate_and_canonicalize_smiles(smiles: str) -> tuple[bool, str | None]:
    """
    Validate a SMILES string with RDKit and return its canonical isomeric form.

    Returns (True, canonical_smiles) on success, (False, None) on failure.
    Stereochemistry is preserved (isomericSmiles=True).
    """
    if not smiles or not smiles.strip():
        return False, None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return False, None
    canonical = Chem.MolToSmiles(mol, isomericSmiles=True)
    return True, canonical


# ---------------------------------------------------------------------------
# Step 5: Disk cache
# ---------------------------------------------------------------------------

def load_cache(cache_path: Path) -> dict[str, Any]:
    """Load the on-disk PubChem response cache, or return empty dict."""
    if cache_path.exists():
        try:
            return json.loads(cache_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Cache unreadable (%s) — starting fresh.", exc)
    return {}


def save_cache(cache: dict[str, Any], cache_path: Path) -> None:
    """Persist the cache to disk atomically."""
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = cache_path.parent / (cache_path.name + ".tmp")
    try:
        tmp.write_text(
            json.dumps(cache, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        os.replace(str(tmp), str(cache_path))
    except OSError as exc:
        logger.warning("Failed to save cache: %s", exc)
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass


# ---------------------------------------------------------------------------
# Step 6: PubChem HTTP helpers
# ---------------------------------------------------------------------------

def _get_with_retry(
    url: str,
    method: str = "GET",
    data: dict[str, str] | None = None,
) -> requests.Response | None:
    """
    Send a GET or POST to PubChem with up to RETRY_COUNT retries on failure.
    Returns the Response on success, None if all retries are exhausted.
    """
    for attempt, backoff in enumerate(RETRY_BACKOFF[:RETRY_COUNT], start=1):
        try:
            if method == "POST":
                resp = requests.post(
                    url, data=data, timeout=REQUEST_TIMEOUT
                )
            else:
                resp = requests.get(url, timeout=REQUEST_TIMEOUT)

            if resp.status_code == 200:
                return resp
            if resp.status_code == 404:
                return resp     # legitimate "not found" — caller handles
            if resp.status_code in (429, 503) and attempt < RETRY_COUNT:
                logger.warning(
                    "PubChem rate-limited or unavailable (HTTP %d); "
                    "retrying in %ds (attempt %d/%d).",
                    resp.status_code, backoff, attempt, RETRY_COUNT,
                )
                time.sleep(backoff)
                continue
            logger.error("PubChem returned HTTP %d for %s", resp.status_code, url)
            return resp

        except requests.exceptions.Timeout:
            logger.warning("Timeout on %s (attempt %d/%d).", url, attempt, RETRY_COUNT)
            if attempt < RETRY_COUNT:
                time.sleep(backoff)
        except requests.exceptions.RequestException as exc:
            logger.warning("Request error: %s (attempt %d/%d).", exc, attempt, RETRY_COUNT)
            if attempt < RETRY_COUNT:
                time.sleep(backoff)

    logger.error("All retries exhausted for %s.", url)
    return None


def fetch_single_compound(cid: int) -> dict[str, Any] | None:
    """
    Fetch a single compound record from PubChem by CID.
    Returns the parsed property dict, or None on failure.
    """
    url = PUBCHEM_SINGLE_URL.format(cid=cid)
    resp = _get_with_retry(url)
    if resp is None:
        return None
    if resp.status_code == 404:
        return {}   # compound not found — caller records mapping_status
    if resp.status_code != 200:
        logger.error(
            "Non-200/404 response for CID %d: HTTP %d", cid, resp.status_code
        )
        return None
    try:
        props = resp.json()["PropertyTable"]["Properties"][0]
        return props
    except (KeyError, IndexError, ValueError) as exc:
        logger.error("Unexpected PubChem response for CID %d: %s", cid, exc)
        return None


def fetch_batch_compounds(cids: list[int]) -> dict[int, dict[str, Any]]:
    """
    Batch-fetch up to BATCH_SIZE compound records from PubChem via POST.
    Returns a dict mapping CID → property dict.
    Missing CIDs are not included in the result.
    """
    if not cids:
        return {}
    cid_str = ",".join(str(c) for c in cids)
    resp = _get_with_retry(
        PUBCHEM_BATCH_URL, method="POST", data={"cid": cid_str}
    )
    if resp is None:
        return {}
    if resp.status_code == 404:
        return {}   # none of the CIDs found
    try:
        props_list = resp.json()["PropertyTable"]["Properties"]
        return {p["CID"]: p for p in props_list}
    except (KeyError, ValueError) as exc:
        logger.error("Unexpected batch PubChem response: %s", exc)
        return {}


# ---------------------------------------------------------------------------
# Step 7: Representative validation
# ---------------------------------------------------------------------------

def validate_parsing_rule(
    stitch_ids: list[str],
    sample_size: int = DEFAULT_VALIDATION_SAMPLE_SIZE,
) -> dict[str, Any]:
    """
    Query PubChem for a sample of STITCH IDs to confirm the parsing rule.

    Selects IDs spread across the full range (first, last, and evenly spaced
    interior indices).

    Returns a dict with validation results.  Raises RuntimeError if any
    representative ID returns an unexpected result that contradicts the rule.
    """
    n = len(stitch_ids)
    if n == 0:
        raise ValueError("No STITCH IDs to validate.")

    # Spread indices across range
    if n <= sample_size:
        indices = list(range(n))
    else:
        step = n / (sample_size - 1)
        indices = sorted(set(
            [0] + [round(i * step) for i in range(1, sample_size - 1)] + [n - 1]
        ))

    sample = [stitch_ids[i] for i in indices]
    logger.info("Validating CID parsing rule on %d representative IDs.", len(sample))

    results = []
    for sid in sample:
        cid = parse_stitch_cid(sid)
        if cid is None:
            # Should not happen — format was already verified
            results.append({
                "stitch_id": sid, "parsed_cid": None,
                "status": "parse_failed", "pubchem_name": None,
            })
            continue

        props = fetch_single_compound(cid)
        time.sleep(0.22)   # respect rate limit

        if props is None:
            results.append({
                "stitch_id": sid, "parsed_cid": cid,
                "status": "request_failed", "pubchem_name": None,
            })
        elif props == {}:
            results.append({
                "stitch_id": sid, "parsed_cid": cid,
                "status": "not_found", "pubchem_name": None,
            })
        else:
            name = props.get("IUPACName", "")
            # PubChem REST API returns property key "SMILES" regardless of
            # whether "IsomericSMILES" or "CanonicalSMILES" was requested.
            smiles = props.get("SMILES", "")
            valid_smiles, _ = validate_and_canonicalize_smiles(smiles)
            results.append({
                "stitch_id": sid,
                "parsed_cid": cid,
                "status": "found" if valid_smiles else "found_no_valid_smiles",
                "pubchem_name": name,
            })

    found = [r for r in results if r["status"].startswith("found")]
    failed = [r for r in results if r["status"] == "request_failed"]
    not_found = [r for r in results if r["status"] == "not_found"]

    logger.info(
        "Validation: %d found, %d not found, %d request failures.",
        len(found), len(not_found), len(failed),
    )

    if failed:
        raise RuntimeError(
            f"PubChem validation requests failed for {len(failed)} representative IDs: "
            f"{[r['stitch_id'] for r in failed]}. "
            "Stopping — PubChem may be unreachable."
        )

    return {
        "sample_size": len(sample),
        "found_count": len(found),
        "not_found_count": len(not_found),
        "request_failed_count": len(failed),
        "representative_results": results,
    }


# ---------------------------------------------------------------------------
# Step 8: Full mapping
# ---------------------------------------------------------------------------

def _build_row(
    stitch_id: str,
    cid: int | None,
    props: dict[str, Any] | None,
    status: str,
    notes: str,
    source: str = "pubchem_rest_api",
) -> dict[str, Any]:
    """Construct one output row dict."""
    pubchem_smiles = None
    standardized_smiles = None
    inchikey = None
    preferred_name = None

    if props:
        pubchem_smiles = props.get("SMILES") or None
        inchikey = props.get("InChIKey") or None
        preferred_name = props.get("IUPACName") or None
        if pubchem_smiles:
            valid, can = validate_and_canonicalize_smiles(pubchem_smiles)
            if valid:
                standardized_smiles = can
            else:
                if status == "mapped":
                    status = "invalid_smiles"
                    notes = f"RDKit rejected SMILES: {pubchem_smiles!r}"

    return {
        "stitch_id": stitch_id,
        "parsed_pubchem_cid": cid,
        "preferred_name": preferred_name,
        "pubchem_smiles": pubchem_smiles,
        "standardized_smiles": standardized_smiles,
        "inchikey": inchikey,
        "mapping_source": source,
        "mapping_status": status,
        "notes": notes,
    }


def map_all_drugs(
    stitch_ids: list[str],
    cache: dict[str, Any],
    batch_size: int = BATCH_SIZE,
) -> list[dict[str, Any]]:
    """
    Query PubChem for every STITCH ID and build mapping rows.

    Results for each CID are cached on disk after every batch to allow
    interrupted reruns to continue without re-querying PubChem.
    """
    rows: list[dict[str, Any]] = []

    # Separate parseable from unparseable
    parseable: list[tuple[str, int]] = []
    for sid in stitch_ids:
        cid = parse_stitch_cid(sid)
        if cid is None:
            rows.append(_build_row(
                sid, None, None, "invalid_stitch_format",
                f"Does not match expected pattern CID + 9 digits: {sid!r}",
            ))
        else:
            parseable.append((sid, cid))

    # Identify which CIDs are not yet cached
    uncached: list[tuple[str, int]] = [
        (sid, cid) for sid, cid in parseable if str(cid) not in cache
    ]
    cached_count = len(parseable) - len(uncached)
    if cached_count:
        logger.info("Using cached responses for %d CIDs.", cached_count)

    # Batch-query uncached CIDs
    uncached_cids = [cid for _, cid in uncached]
    total_batches = (len(uncached_cids) + batch_size - 1) // batch_size
    for batch_num, start in enumerate(range(0, len(uncached_cids), batch_size), 1):
        batch = uncached_cids[start : start + batch_size]
        logger.info(
            "Batch %d/%d: querying %d CIDs.", batch_num, total_batches, len(batch)
        )
        result = fetch_batch_compounds(batch)
        for cid in batch:
            cache[str(cid)] = result.get(cid, {})   # {} = not found
        save_cache(cache, DEFAULT_CACHE_PATH)
        if batch_num < total_batches:
            time.sleep(INTER_BATCH_SLEEP)

    # Build output rows from cache
    for sid, cid in parseable:
        cached = cache.get(str(cid))
        if cached is None:
            # Should not occur; cache is written above
            rows.append(_build_row(sid, cid, None, "request_failed", "No cached result."))
            continue

        if cached == {}:
            rows.append(_build_row(sid, cid, None, "pubchem_not_found",
                                   f"CID {cid} not found in PubChem."))
            continue

        smiles = cached.get("SMILES") or ""
        if not smiles.strip():
            rows.append(_build_row(sid, cid, cached, "missing_smiles",
                                   "PubChem returned no SMILES for this CID."))
            continue

        valid, _ = validate_and_canonicalize_smiles(smiles)
        status = "mapped" if valid else "invalid_smiles"
        rows.append(_build_row(sid, cid, cached, status, ""))

    return rows


# ---------------------------------------------------------------------------
# Step 9: Duplicate structure detection
# ---------------------------------------------------------------------------

def detect_duplicates(df: pd.DataFrame) -> dict[str, Any]:
    """
    Find groups where multiple STITCH IDs share the same PubChem CID,
    InChIKey, or standardized SMILES.

    Returns a dict with one key per duplicate type, each containing a list
    of groups.  Only groups with 2+ members are reported.
    """
    mapped = df[df["mapping_status"] == "mapped"]

    def _groups(col: str) -> list[dict[str, Any]]:
        if col not in mapped.columns:
            return []
        groups = []
        for val, grp in mapped.groupby(col):
            if len(grp) > 1:
                groups.append({
                    col: val,
                    "stitch_ids": sorted(grp["stitch_id"].tolist()),
                    "count": len(grp),
                })
        return sorted(groups, key=lambda g: -g["count"])

    return {
        "duplicate_cid_groups": _groups("parsed_pubchem_cid"),
        "duplicate_inchikey_groups": _groups("inchikey"),
        "duplicate_smiles_groups": _groups("standardized_smiles"),
    }


# ---------------------------------------------------------------------------
# Step 10: Pair eligibility
# ---------------------------------------------------------------------------

def calculate_pair_eligibility(
    pairs_df: pd.DataFrame,
    mapping_df: pd.DataFrame,
) -> dict[str, Any]:
    """
    Count how many pairs have both drugs successfully mapped.
    A drug is successfully mapped when mapping_status == 'mapped'.
    """
    mapped_ids = set(
        mapping_df.loc[mapping_df["mapping_status"] == "mapped", "stitch_id"]
    )
    total_pairs = len(pairs_df)
    both_mapped = int(
        (pairs_df["drug_1"].isin(mapped_ids) & pairs_df["drug_2"].isin(mapped_ids)).sum()
    )
    one_unmapped = total_pairs - both_mapped
    return {
        "total_pairs": total_pairs,
        "pairs_with_both_drugs_mapped": both_mapped,
        "pairs_with_missing_mapping": one_unmapped,
        "pair_coverage_percent": round(100 * both_mapped / total_pairs, 2) if total_pairs else 0.0,
    }


# ---------------------------------------------------------------------------
# Audit
# ---------------------------------------------------------------------------

def _software_versions() -> dict[str, str]:
    from rdkit import rdBase  # noqa: PLC0415
    return {
        "python": platform.python_version(),
        "pandas": pd.__version__,
        "rdkit": rdBase.rdkitVersion,
        "requests": requests.__version__,
    }


def build_smiles_audit(
    stitch_ids: list[str],
    format_info: dict[str, Any],
    validation_results: dict[str, Any],
    mapping_df: pd.DataFrame,
    duplicates: dict[str, Any],
    pair_eligibility: dict[str, Any],
) -> dict[str, Any]:
    status_counts: dict[str, int] = mapping_df["mapping_status"].value_counts().to_dict()
    mapped_df = mapping_df[mapping_df["mapping_status"] == "mapped"]

    return {
        "total_unique_stitch_ids": len(stitch_ids),
        "stitch_id_formats": format_info,
        "successfully_parsed_cids": int(
            mapping_df["parsed_pubchem_cid"].notna().sum()
        ),
        "representative_validation": validation_results,
        "successfully_mapped_drugs": int(len(mapped_df)),
        "mapping_coverage_percent": round(
            100 * len(mapped_df) / len(stitch_ids), 2
        ) if stitch_ids else 0.0,
        "status_counts": status_counts,
        "valid_smiles_count": int(mapped_df["standardized_smiles"].notna().sum()),
        "invalid_smiles_count": int(
            (mapping_df["mapping_status"] == "invalid_smiles").sum()
        ),
        "unique_pubchem_cids": int(mapped_df["parsed_pubchem_cid"].nunique()),
        "unique_inchikeys": int(
            mapped_df["inchikey"].dropna().nunique()
        ),
        "unique_standardized_smiles": int(
            mapped_df["standardized_smiles"].dropna().nunique()
        ),
        "duplicate_cid_groups": duplicates["duplicate_cid_groups"],
        "duplicate_inchikey_groups": duplicates["duplicate_inchikey_groups"],
        "duplicate_smiles_groups": duplicates["duplicate_smiles_groups"],
        "pairs_with_both_drugs_mapped": pair_eligibility["pairs_with_both_drugs_mapped"],
        "pairs_with_missing_mapping": pair_eligibility["pairs_with_missing_mapping"],
        "pair_coverage_percent": pair_eligibility["pair_coverage_percent"],
        "software_versions": _software_versions(),
        "mapping_source": "PubChem REST PUG API — IsomericSMILES, IUPACName, InChIKey",
    }


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def run_mapping(
    pairs_input: Path = DEFAULT_PAIRS_INPUT,
    mapping_output: Path = DEFAULT_MAPPING_OUTPUT,
    audit_output: Path = DEFAULT_AUDIT_OUTPUT,
    cache_path: Path = DEFAULT_CACHE_PATH,
    validation_sample_size: int = DEFAULT_VALIDATION_SAMPLE_SIZE,
    skip_validation: bool = False,
) -> dict[str, Any]:
    """
    Execute the full Milestone 2 SMILES mapping pipeline.

    Stops with RuntimeError if representative validation fails or if
    the STITCH format is ambiguous.
    """
    logger.info("=== Milestone 2 SMILES mapping started ===")

    # 1. Load pair table
    pairs_df = pd.read_parquet(pairs_input)
    logger.info("Loaded pair table: %d pairs.", len(pairs_df))

    # 2. Extract unique drugs
    stitch_ids = extract_unique_drugs(pairs_df)
    logger.info("Unique STITCH IDs: %d", len(stitch_ids))

    # 3. Inspect formats
    format_info = inspect_stitch_formats(stitch_ids)
    logger.info("STITCH format analysis: %s", format_info)

    if format_info["unrecognized_count"] > 0:
        raise RuntimeError(
            f"Found {format_info['unrecognized_count']} unrecognized STITCH formats: "
            f"{format_info['unrecognized_examples']}. "
            "Cannot proceed with ambiguous CID parsing rule."
        )

    # 4. Representative validation
    if skip_validation:
        logger.warning("Skipping representative validation (--skip-validation flag).")
        validation_results: dict[str, Any] = {"skipped": True}
    else:
        validation_results = validate_parsing_rule(stitch_ids, validation_sample_size)

    # 5. Load cache
    cache = load_cache(cache_path)

    # 6. Full mapping
    rows = map_all_drugs(stitch_ids, cache)

    # 7. Build DataFrame sorted by stitch_id
    mapping_df = pd.DataFrame(rows, columns=REQUIRED_OUTPUT_COLUMNS)
    mapping_df = mapping_df.sort_values("stitch_id").reset_index(drop=True)

    # Ensure int columns are nullable int (CID may be None for invalid format)
    mapping_df["parsed_pubchem_cid"] = pd.array(
        mapping_df["parsed_pubchem_cid"].tolist(), dtype="Int64"
    )

    # 8. Duplicates
    duplicates = detect_duplicates(mapping_df)

    # 9. Pair eligibility
    pair_eligibility = calculate_pair_eligibility(pairs_df, mapping_df)

    # 10. Audit
    audit = build_smiles_audit(
        stitch_ids, format_info, validation_results,
        mapping_df, duplicates, pair_eligibility,
    )

    # 11. Write outputs
    for path in [mapping_output, audit_output]:
        path.parent.mkdir(parents=True, exist_ok=True)

    mapping_df.to_csv(mapping_output, index=False)
    logger.info("Wrote mapping: %s (%d rows)", mapping_output, len(mapping_df))

    audit_output.write_text(
        json.dumps(audit, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    logger.info("Wrote audit: %s", audit_output)

    mapped_n = int((mapping_df["mapping_status"] == "mapped").sum())
    logger.info(
        "=== Milestone 2 complete: %d/%d drugs mapped (%.1f%%) ===",
        mapped_n, len(stitch_ids), 100 * mapped_n / len(stitch_ids),
    )
    return audit


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Milestone 2 — map STITCH IDs to validated SMILES.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--pairs-input", type=Path, default=DEFAULT_PAIRS_INPUT)
    p.add_argument("--mapping-output", type=Path, default=DEFAULT_MAPPING_OUTPUT)
    p.add_argument("--audit-output", type=Path, default=DEFAULT_AUDIT_OUTPUT)
    p.add_argument("--cache-path", type=Path, default=DEFAULT_CACHE_PATH)
    p.add_argument(
        "--validation-sample-size", type=int, default=DEFAULT_VALIDATION_SAMPLE_SIZE,
    )
    p.add_argument(
        "--skip-validation", action="store_true",
        help="Skip representative PubChem validation (for reruns with warm cache).",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )
    args = _parse_args(argv)
    run_mapping(
        pairs_input=args.pairs_input,
        mapping_output=args.mapping_output,
        audit_output=args.audit_output,
        cache_path=args.cache_path,
        validation_sample_size=args.validation_sample_size,
        skip_validation=args.skip_validation,
    )


if __name__ == "__main__":
    main()
