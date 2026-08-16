"""
Experiment 1 — UMLS metadata enrichment for PolyLLM side-effect concepts.

Every retained side effect in `polyllm_label_mapping.csv` already carries a
UMLS CUI in its `side_effect_id` column (verified: 963/963 valid, unique,
`^C\\d{7}$`). This module retrieves, for each unique CUI, the UMLS preferred
name, English synonyms, definitions, and semantic type(s) via the official
UMLS REST API (https://documentation.uts.nlm.nih.gov/rest/), then writes a
canonical derived metadata artifact aligned row-for-row with `label_index`.

This does NOT generate embeddings, retrain the GNN, or otherwise touch the
existing PolyLLM reproduction artifacts.

Auth: reads the API key from the `UMLS_API_KEY` environment variable. Obtain
a key by creating a free UTS account at https://uts.nlm.nih.gov/uts/signup-login
then set it locally, e.g. (PowerShell): `$env:UMLS_API_KEY = "..."`, or add
it to a local `.env` that is not committed (see `.gitignore`).

Run from the project root:
    python src/polyllm/data/enrich_side_effects_umls.py
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import platform
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from polyllm.features.generate_side_effect_embeddings import validate_mapping_df

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

UMLS_BASE_URL = "https://uts-ws.nlm.nih.gov/rest"
UMLS_VERSION = "current"
CUI_PATTERN = re.compile(r"^C\d{7}$")

REQUEST_TIMEOUT = 20        # seconds
RETRY_COUNT = 3
RETRY_BACKOFF = [1, 2, 4]   # seconds between retries
INTER_REQUEST_SLEEP = 0.2   # polite pacing; no published UMLS rate limit,
                             # matches this repo's PubChem convention (Milestone 2)

DEFAULT_MAPPING_INPUT = Path("data/processed/polyllm_label_mapping.csv")
DEFAULT_METADATA_PARQUET = Path("data/processed/umls_side_effect_metadata.parquet")
DEFAULT_METADATA_CSV = Path("data/processed/umls_side_effect_metadata.csv")
DEFAULT_CACHE_PATH = Path("outputs/polyllm/umls_cache.json")
DEFAULT_AUDIT_OUTPUT = Path("outputs/polyllm/umls_enrichment_audit.json")
DEFAULT_COVERAGE_REPORT = Path("outputs/polyllm/umls_coverage_report.md")

METADATA_COLUMNS = [
    "label_index",
    "side_effect_id",
    "side_effect_name",
    "cui",
    "umls_preferred_name",
    "umls_synonyms",
    "umls_definitions",
    "umls_semantic_types",
    "synonym_count",
    "definition_count",
    "has_umls_concept",
    "has_synonyms",
    "has_definition",
    "has_semantic_type",
    "retrieval_status",
    "retrieval_error",
    "umls_version_or_release",
    "retrieved_at",
]

# retrieval_status values
STATUS_SUCCESS = "success"
STATUS_NOT_FOUND = "not_found"
STATUS_REQUEST_FAILED = "request_failed"
STATUS_INVALID_CUI = "invalid_cui"


# ---------------------------------------------------------------------------
# API key handling
# ---------------------------------------------------------------------------

def get_api_key() -> str:
    """
    Read the UMLS API key from the environment.

    Never hard-coded, never logged. Raises RuntimeError with setup
    instructions if missing.
    """
    key = os.environ.get("UMLS_API_KEY", "").strip()
    if not key:
        raise RuntimeError(
            "UMLS_API_KEY environment variable is not set.\n"
            "1. Create a free UTS account: https://uts.nlm.nih.gov/uts/signup-login\n"
            "2. Copy your API key from your UTS profile.\n"
            "3. Set it locally, e.g. (PowerShell): $env:UMLS_API_KEY = \"your-key\"\n"
            "   or add UMLS_API_KEY=your-key to a local .env file (already gitignored)."
        )
    return key


# ---------------------------------------------------------------------------
# CUI validation
# ---------------------------------------------------------------------------

def validate_cuis(mapping_df: pd.DataFrame) -> dict[str, Any]:
    """
    Inspect side_effect_id values as CUIs.

    Returns a dict with malformed/duplicate/missing CUI diagnostics. Does not
    raise — malformed or duplicate CUIs are recorded per-row as
    retrieval_status == 'invalid_cui' rather than aborting the run.
    """
    ids = mapping_df["side_effect_id"].astype(str)
    missing_mask = mapping_df["side_effect_id"].isna() | (ids.str.strip() == "")
    malformed_mask = ~ids.str.match(CUI_PATTERN) & ~missing_mask
    dup_mask = ids.duplicated(keep=False) & ~missing_mask

    return {
        "total": len(mapping_df),
        "unique_cuis": int(ids[~missing_mask].nunique()),
        "missing_cui_count": int(missing_mask.sum()),
        "malformed_cui_count": int(malformed_mask.sum()),
        "malformed_cui_examples": ids[malformed_mask].tolist()[:10],
        "duplicate_cui_count": int(dup_mask.sum()),
        "duplicate_cui_examples": ids[dup_mask].tolist()[:10],
    }


# ---------------------------------------------------------------------------
# Disk cache (one record per CUI, keyed by CUI)
# ---------------------------------------------------------------------------

def load_cache(cache_path: Path) -> dict[str, Any]:
    """Load the on-disk UMLS response cache, or return an empty dict."""
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
        tmp.write_text(json.dumps(cache, indent=2, ensure_ascii=False), encoding="utf-8")
        os.replace(str(tmp), str(cache_path))
    except OSError as exc:
        logger.warning("Failed to save cache: %s", exc)
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass


# ---------------------------------------------------------------------------
# UMLS HTTP layer
# ---------------------------------------------------------------------------

def _request_with_retry(path: str, params: dict[str, str]) -> requests.Response | None:
    """
    GET a UMLS endpoint with capped retries on transient failure.

    `path` is logged on error; the apiKey-bearing query string never is.
    """
    url = f"{UMLS_BASE_URL}{path}"
    for attempt, backoff in enumerate(RETRY_BACKOFF[:RETRY_COUNT], start=1):
        try:
            resp = requests.get(url, params=params, timeout=REQUEST_TIMEOUT)
            if resp.status_code == 200:
                return resp
            if resp.status_code == 404:
                return resp  # legitimate "not found" — caller handles
            if resp.status_code in (429, 500, 502, 503, 504) and attempt < RETRY_COUNT:
                logger.warning(
                    "UMLS transient error HTTP %d for %s; retrying in %ds (attempt %d/%d).",
                    resp.status_code, path, backoff, attempt, RETRY_COUNT,
                )
                time.sleep(backoff)
                continue
            logger.error("UMLS returned HTTP %d for %s", resp.status_code, path)
            return resp
        except requests.exceptions.Timeout:
            logger.warning("Timeout on %s (attempt %d/%d).", path, attempt, RETRY_COUNT)
            if attempt < RETRY_COUNT:
                time.sleep(backoff)
        except requests.exceptions.RequestException as exc:
            logger.warning("Request error on %s: %s (attempt %d/%d).", path, exc, attempt, RETRY_COUNT)
            if attempt < RETRY_COUNT:
                time.sleep(backoff)

    logger.error("All retries exhausted for %s.", path)
    return None


def fetch_cui_record(cui: str, api_key: str) -> dict[str, Any]:
    """
    Fetch concept, atoms, and definitions for one CUI.

    Returns a raw cache record:
        {"retrieval_status", "retrieval_error", "concept", "atoms",
         "definitions", "umls_version_or_release", "retrieved_at"}

    A concept 404 short-circuits to retrieval_status='not_found' (atoms and
    definitions are not queried for a CUI that doesn't exist). Any other
    network exhaustion yields retrieval_status='request_failed'.
    """
    concept_resp = _request_with_retry(f"/content/{UMLS_VERSION}/CUI/{cui}", {"apiKey": api_key})
    if concept_resp is None:
        return {
            "retrieval_status": STATUS_REQUEST_FAILED,
            "retrieval_error": "concept request failed after retries",
            "concept": None, "atoms": None, "definitions": None,
            "umls_version_or_release": None,
            "retrieved_at": _now_iso(),
        }
    if concept_resp.status_code == 404:
        return {
            "retrieval_status": STATUS_NOT_FOUND,
            "retrieval_error": "CUI not found in UMLS",
            "concept": None, "atoms": None, "definitions": None,
            "umls_version_or_release": None,
            "retrieved_at": _now_iso(),
        }
    if concept_resp.status_code != 200:
        return {
            "retrieval_status": STATUS_REQUEST_FAILED,
            "retrieval_error": f"concept endpoint returned HTTP {concept_resp.status_code}",
            "concept": None, "atoms": None, "definitions": None,
            "umls_version_or_release": None,
            "retrieved_at": _now_iso(),
        }

    concept = concept_resp.json().get("result")
    release = _extract_release(concept)

    time.sleep(INTER_REQUEST_SLEEP)
    atoms_resp = _request_with_retry(
        f"/content/{UMLS_VERSION}/CUI/{cui}/atoms",
        {"apiKey": api_key, "language": "ENG", "pageSize": 200},
    )
    atoms = _extract_list_result(atoms_resp)

    time.sleep(INTER_REQUEST_SLEEP)
    defs_resp = _request_with_retry(
        f"/content/{UMLS_VERSION}/CUI/{cui}/definitions", {"apiKey": api_key}
    )
    definitions = _extract_list_result(defs_resp)

    errors = []
    if atoms_resp is None or atoms_resp.status_code not in (200, 404):
        errors.append("atoms request failed")
    if defs_resp is None or defs_resp.status_code not in (200, 404):
        errors.append("definitions request failed")

    return {
        "retrieval_status": STATUS_SUCCESS,
        "retrieval_error": "; ".join(errors) or None,
        "concept": concept,
        "atoms": atoms,
        "definitions": definitions,
        "umls_version_or_release": release,
        "retrieved_at": _now_iso(),
    }


def _extract_list_result(resp: requests.Response | None) -> list[dict[str, Any]] | None:
    """Normalize a 200 (list result) or 404 (no atoms/definitions) response to a list."""
    if resp is None:
        return None
    if resp.status_code == 404:
        return []
    if resp.status_code != 200:
        return None
    result = resp.json().get("result")
    if result is None:
        return []
    return result if isinstance(result, list) else [result]


def _extract_release(concept: dict[str, Any] | None) -> str | None:
    """Best-effort extraction of the resolved UMLS release from a concept response's atoms URL."""
    if not concept:
        return None
    atoms_url = concept.get("atoms", "")
    m = re.search(r"/content/([0-9]{4}[A-Z]{2})/", atoms_url)
    return m.group(1) if m else UMLS_VERSION


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Phase 4: synonym cleaning
# ---------------------------------------------------------------------------

def clean_synonyms(atoms: list[dict[str, Any]] | None, preferred_name: str | None) -> list[str]:
    """
    Deterministic synonym cleanup from raw UMLS atom records.

    English only, trimmed, exact/case-insensitive de-duplicated, preferred
    name excluded, sorted case-insensitively for reproducibility regardless
    of API response ordering. No count cap.
    """
    if not atoms:
        return []

    preferred_norm = (preferred_name or "").strip().casefold()
    seen: set[str] = set()
    cleaned: list[str] = []
    for atom in atoms:
        if atom.get("language") not in (None, "ENG"):
            continue
        name = (atom.get("name") or "").strip()
        if not name:
            continue
        norm = name.casefold()
        if norm == preferred_norm or norm in seen:
            continue
        seen.add(norm)
        cleaned.append(name)

    return sorted(cleaned, key=str.casefold)


# ---------------------------------------------------------------------------
# Phase 5: definitions
# ---------------------------------------------------------------------------

def clean_definitions(definitions: list[dict[str, Any]] | None) -> list[dict[str, str]]:
    """
    Deterministic definition normalization: {"value", "source"} pairs,
    exact-duplicate (source, value) pairs removed, sorted for reproducibility.
    No definitions are invented; an empty list means none were available.
    """
    if not definitions:
        return []

    seen: set[tuple[str, str]] = set()
    cleaned: list[dict[str, str]] = []
    for d in definitions:
        value = (d.get("value") or "").strip()
        source = (d.get("rootSource") or "").strip()
        if not value:
            continue
        key = (source, value)
        if key in seen:
            continue
        seen.add(key)
        cleaned.append({"value": value, "source": source})

    return sorted(cleaned, key=lambda d: (d["source"], d["value"]))


def extract_semantic_types(concept: dict[str, Any] | None) -> list[str]:
    """Deduplicated, sorted semantic type names from a concept record."""
    if not concept:
        return []
    names = {st.get("name", "").strip() for st in concept.get("semanticTypes", []) if st.get("name")}
    return sorted(names)


# ---------------------------------------------------------------------------
# Cache -> canonical record
# ---------------------------------------------------------------------------

def build_metadata_row(
    label_index: int,
    side_effect_id: str,
    side_effect_name: str,
    cui_record: dict[str, Any] | None,
    invalid_cui_reason: str | None,
) -> dict[str, Any]:
    """Build one canonical metadata row from a cache record (or invalid-CUI reason)."""
    if invalid_cui_reason is not None:
        return {
            "label_index": label_index,
            "side_effect_id": side_effect_id,
            "side_effect_name": side_effect_name,
            "cui": side_effect_id,
            "umls_preferred_name": None,
            "umls_synonyms": json.dumps([]),
            "umls_definitions": json.dumps([]),
            "umls_semantic_types": json.dumps([]),
            "synonym_count": 0,
            "definition_count": 0,
            "has_umls_concept": False,
            "has_synonyms": False,
            "has_definition": False,
            "has_semantic_type": False,
            "retrieval_status": STATUS_INVALID_CUI,
            "retrieval_error": invalid_cui_reason,
            "umls_version_or_release": None,
            "retrieved_at": None,
        }

    status = cui_record["retrieval_status"]
    concept = cui_record.get("concept")
    preferred_name = concept.get("name") if concept else None
    synonyms = clean_synonyms(cui_record.get("atoms"), preferred_name)
    definitions = clean_definitions(cui_record.get("definitions"))
    semantic_types = extract_semantic_types(concept)

    return {
        "label_index": label_index,
        "side_effect_id": side_effect_id,
        "side_effect_name": side_effect_name,
        "cui": side_effect_id,
        "umls_preferred_name": preferred_name,
        "umls_synonyms": json.dumps(synonyms, ensure_ascii=False),
        "umls_definitions": json.dumps(definitions, ensure_ascii=False),
        "umls_semantic_types": json.dumps(semantic_types, ensure_ascii=False),
        "synonym_count": len(synonyms),
        "definition_count": len(definitions),
        "has_umls_concept": status == STATUS_SUCCESS and concept is not None,
        "has_synonyms": len(synonyms) > 0,
        "has_definition": len(definitions) > 0,
        "has_semantic_type": len(semantic_types) > 0,
        "retrieval_status": status,
        "retrieval_error": cui_record.get("retrieval_error"),
        "umls_version_or_release": cui_record.get("umls_version_or_release"),
        "retrieved_at": cui_record.get("retrieved_at"),
    }


# ---------------------------------------------------------------------------
# Phase 10: label alignment validation
# ---------------------------------------------------------------------------

def validate_label_alignment(mapping_df: pd.DataFrame, metadata_df: pd.DataFrame) -> None:
    """
    Prove metadata_df row i refers to the exact same side effect as
    mapping_df row i. Raises ValueError on any mismatch.
    """
    if len(mapping_df) != len(metadata_df):
        raise ValueError(
            f"Row count mismatch: mapping has {len(mapping_df)}, metadata has {len(metadata_df)}."
        )

    expected_index = mapping_df["label_index"].to_numpy()
    if not (metadata_df["label_index"].to_numpy() == expected_index).all():
        raise ValueError("metadata_df.label_index does not match mapping_df.label_index in order.")

    if not (metadata_df["side_effect_id"].to_numpy() == mapping_df["side_effect_id"].to_numpy()).all():
        raise ValueError("metadata_df.side_effect_id does not match mapping_df.side_effect_id in order.")

    if not (metadata_df["side_effect_name"].to_numpy() == mapping_df["side_effect_name"].to_numpy()).all():
        raise ValueError("metadata_df.side_effect_name does not match mapping_df.side_effect_name in order.")


# ---------------------------------------------------------------------------
# Phase 9: coverage report
# ---------------------------------------------------------------------------

def build_coverage_report(metadata_df: pd.DataFrame, cui_diagnostics: dict[str, Any]) -> dict[str, Any]:
    total = len(metadata_df)
    successful = int((metadata_df["retrieval_status"] == STATUS_SUCCESS).sum())

    def pct(n: int) -> float:
        return round(100 * n / total, 2) if total else 0.0

    with_name = int(metadata_df["has_umls_concept"].sum())
    with_syn = int(metadata_df["has_synonyms"].sum())
    with_def = int(metadata_df["has_definition"].sum())
    with_sem = int(metadata_df["has_semantic_type"].sum())

    name_only = int(((~metadata_df["has_synonyms"]) & (~metadata_df["has_definition"]) & metadata_df["has_umls_concept"]).sum())
    name_syn = int((metadata_df["has_synonyms"] & (~metadata_df["has_definition"])).sum())
    name_def = int(((~metadata_df["has_synonyms"]) & metadata_df["has_definition"]).sum())
    name_syn_def = int((metadata_df["has_synonyms"] & metadata_df["has_definition"]).sum())

    syn_counts = metadata_df["synonym_count"]

    failed = metadata_df[metadata_df["retrieval_status"] != STATUS_SUCCESS]
    no_syn = metadata_df[~metadata_df["has_synonyms"]]
    no_def = metadata_df[~metadata_df["has_definition"]]

    name_diffs = []
    for _, row in metadata_df.iterrows():
        pref = row["umls_preferred_name"]
        if isinstance(pref, str) and pref.strip() and pref.strip().casefold() != str(row["side_effect_name"]).strip().casefold():
            name_diffs.append({
                "label_index": int(row["label_index"]),
                "side_effect_name": row["side_effect_name"],
                "umls_preferred_name": pref,
            })

    return {
        "total_retained_side_effects": total,
        "unique_cuis": cui_diagnostics["unique_cuis"],
        "valid_cuis": total - cui_diagnostics["malformed_cui_count"] - cui_diagnostics["missing_cui_count"],
        "missing_cuis": cui_diagnostics["missing_cui_count"],
        "malformed_cuis": cui_diagnostics["malformed_cui_count"],
        "duplicate_cuis": cui_diagnostics["duplicate_cui_count"],
        "successful_umls_concept_retrievals": successful,
        "with_preferred_name": with_name,
        "with_preferred_name_percent": pct(with_name),
        "with_ge1_synonym": with_syn,
        "with_ge1_synonym_percent": pct(with_syn),
        "with_ge1_definition": with_def,
        "with_ge1_definition_percent": pct(with_def),
        "with_semantic_type": with_sem,
        "with_semantic_type_percent": pct(with_sem),
        "mean_synonyms_per_side_effect": float(round(syn_counts.mean(), 3)) if total else 0.0,
        "median_synonyms_per_side_effect": float(syn_counts.median()) if total else 0.0,
        "max_synonyms_per_side_effect": int(syn_counts.max()) if total else 0,
        "name_only_count": name_only,
        "name_plus_synonyms_count": name_syn,
        "name_plus_definition_count": name_def,
        "name_plus_synonyms_plus_definition_count": name_syn_def,
        "retrieval_status_counts": metadata_df["retrieval_status"].value_counts().to_dict(),
        "umls_lookup_failed": failed[["label_index", "side_effect_id", "side_effect_name", "retrieval_status", "retrieval_error"]].to_dict("records"),
        "no_synonym_available": no_syn[["label_index", "side_effect_id", "side_effect_name"]].to_dict("records"),
        "no_definition_available": no_def[["label_index", "side_effect_id", "side_effect_name"]].to_dict("records"),
        "preferred_name_differs_materially": name_diffs,
    }


def render_coverage_markdown(coverage: dict[str, Any]) -> str:
    lines = [
        "# UMLS Enrichment Coverage Report",
        "",
        f"Generated: {_now_iso()}",
        "",
        "## Summary",
        "",
        f"- Total retained side effects: **{coverage['total_retained_side_effects']}**",
        f"- Unique CUIs: **{coverage['unique_cuis']}**",
        f"- Valid CUIs: **{coverage['valid_cuis']}**",
        f"- Missing CUIs: **{coverage['missing_cuis']}**",
        f"- Malformed CUIs: **{coverage['malformed_cuis']}**",
        f"- Duplicate CUIs: **{coverage['duplicate_cuis']}**",
        f"- Successful UMLS concept retrievals: **{coverage['successful_umls_concept_retrievals']}**",
        "",
        "## Field coverage",
        "",
        "| Field | Count | Percent |",
        "|---|---|---|",
        f"| Preferred name | {coverage['with_preferred_name']} | {coverage['with_preferred_name_percent']}% |",
        f"| >=1 synonym | {coverage['with_ge1_synonym']} | {coverage['with_ge1_synonym_percent']}% |",
        f"| >=1 definition | {coverage['with_ge1_definition']} | {coverage['with_ge1_definition_percent']}% |",
        f"| Semantic type | {coverage['with_semantic_type']} | {coverage['with_semantic_type_percent']}% |",
        "",
        "## Synonym counts",
        "",
        f"- Mean: {coverage['mean_synonyms_per_side_effect']}",
        f"- Median: {coverage['median_synonyms_per_side_effect']}",
        f"- Max: {coverage['max_synonyms_per_side_effect']}",
        "",
        "## Content-level breakdown",
        "",
        f"- Name only: {coverage['name_only_count']}",
        f"- Name + synonyms: {coverage['name_plus_synonyms_count']}",
        f"- Name + definition: {coverage['name_plus_definition_count']}",
        f"- Name + synonyms + definition: {coverage['name_plus_synonyms_plus_definition_count']}",
        "",
        "## Retrieval status counts",
        "",
    ]
    for status, count in coverage["retrieval_status_counts"].items():
        lines.append(f"- `{status}`: {count}")

    lines += ["", "## UMLS lookup failures", ""]
    if coverage["umls_lookup_failed"]:
        for r in coverage["umls_lookup_failed"]:
            lines.append(f"- [{r['label_index']}] {r['side_effect_id']} `{r['side_effect_name']}` — {r['retrieval_status']}: {r['retrieval_error']}")
    else:
        lines.append("None.")

    lines += ["", "## No synonym available", ""]
    lines.append(f"{len(coverage['no_synonym_available'])} side effects.")

    lines += ["", "## No definition available", ""]
    lines.append(f"{len(coverage['no_definition_available'])} side effects.")

    lines += ["", "## Preferred name differs materially from PolyLLM/TWOSIDES name", ""]
    if coverage["preferred_name_differs_materially"]:
        for r in coverage["preferred_name_differs_materially"]:
            lines.append(f"- [{r['label_index']}] `{r['side_effect_name']}` → UMLS: `{r['umls_preferred_name']}`")
    else:
        lines.append("None.")

    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Atomic write helper
# ---------------------------------------------------------------------------

def _write_atomic(data: bytes, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.parent / (path.name + ".tmp")
    try:
        tmp.write_bytes(data)
        os.replace(str(tmp), str(path))
    except OSError:
        if tmp.exists():
            tmp.unlink(missing_ok=True)
        raise


def _software_versions() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "pandas": pd.__version__,
        "requests": requests.__version__,
    }


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def run_enrichment(
    mapping_input: Path = DEFAULT_MAPPING_INPUT,
    metadata_parquet_output: Path = DEFAULT_METADATA_PARQUET,
    metadata_csv_output: Path = DEFAULT_METADATA_CSV,
    cache_path: Path = DEFAULT_CACHE_PATH,
    audit_output: Path = DEFAULT_AUDIT_OUTPUT,
    coverage_report_output: Path = DEFAULT_COVERAGE_REPORT,
    refresh: bool = False,
    api_key: str | None = None,
) -> dict[str, Any]:
    """
    Full UMLS enrichment pipeline. Idempotent: reruns reuse successful cache
    entries unless refresh=True (which re-fetches everything).
    """
    logger.info("=== Experiment 1 — UMLS side-effect enrichment started ===")

    mapping_df = pd.read_csv(mapping_input)
    logger.info("Loaded label mapping: %d rows.", len(mapping_df))

    validate_mapping_df(mapping_df)
    mapping_df = mapping_df.sort_values("label_index").reset_index(drop=True)

    cui_diagnostics = validate_cuis(mapping_df)
    logger.info("CUI diagnostics: %s", cui_diagnostics)

    if api_key is None:
        api_key = get_api_key()

    cache = load_cache(cache_path)

    rows: list[dict[str, Any]] = []
    total = len(mapping_df)
    for i, row in mapping_df.iterrows():
        cui = str(row["side_effect_id"])

        if not CUI_PATTERN.match(cui):
            rows.append(build_metadata_row(
                int(row["label_index"]), cui, row["side_effect_name"],
                None, f"CUI does not match expected pattern: {cui!r}",
            ))
            continue

        cached = cache.get(cui)
        needs_fetch = (
            refresh
            or cached is None
            or cached.get("retrieval_status") == STATUS_REQUEST_FAILED
        )
        if needs_fetch:
            logger.info("Fetching UMLS record %d/%d: %s", i + 1, total, cui)
            record = fetch_cui_record(cui, api_key)
            cache[cui] = record
            save_cache(cache, cache_path)
            time.sleep(INTER_REQUEST_SLEEP)
        else:
            record = cached

        rows.append(build_metadata_row(
            int(row["label_index"]), cui, row["side_effect_name"], record, None,
        ))

    metadata_df = pd.DataFrame(rows, columns=METADATA_COLUMNS)

    validate_label_alignment(mapping_df, metadata_df)
    logger.info("Label alignment validated: metadata rows match mapping rows 1:1 in order.")

    for path in [metadata_parquet_output, metadata_csv_output, audit_output, coverage_report_output]:
        path.parent.mkdir(parents=True, exist_ok=True)

    metadata_df.to_parquet(metadata_parquet_output, index=False)
    metadata_df.to_csv(metadata_csv_output, index=False)
    logger.info("Wrote canonical metadata: %s, %s (%d rows)",
                metadata_parquet_output, metadata_csv_output, len(metadata_df))

    coverage = build_coverage_report(metadata_df, cui_diagnostics)
    audit = {
        "generated_at": _now_iso(),
        "cui_diagnostics": cui_diagnostics,
        "coverage": coverage,
        "software_versions": _software_versions(),
        "umls_base_url": UMLS_BASE_URL,
        "umls_version_requested": UMLS_VERSION,
        "input_paths": {"mapping": str(mapping_input)},
        "output_paths": {
            "metadata_parquet": str(metadata_parquet_output),
            "metadata_csv": str(metadata_csv_output),
            "cache": str(cache_path),
        },
    }
    _write_atomic(json.dumps(audit, indent=2, ensure_ascii=False).encode("utf-8"), audit_output)
    logger.info("Wrote audit: %s", audit_output)

    _write_atomic(render_coverage_markdown(coverage).encode("utf-8"), coverage_report_output)
    logger.info("Wrote coverage report: %s", coverage_report_output)

    logger.info(
        "=== Experiment 1 enrichment complete: %d/%d successful concept retrievals ===",
        coverage["successful_umls_concept_retrievals"], total,
    )
    return audit


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Experiment 1 — UMLS metadata enrichment for PolyLLM side effects.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--mapping-input", type=Path, default=DEFAULT_MAPPING_INPUT)
    p.add_argument("--metadata-parquet-output", type=Path, default=DEFAULT_METADATA_PARQUET)
    p.add_argument("--metadata-csv-output", type=Path, default=DEFAULT_METADATA_CSV)
    p.add_argument("--cache-path", type=Path, default=DEFAULT_CACHE_PATH)
    p.add_argument("--audit-output", type=Path, default=DEFAULT_AUDIT_OUTPUT)
    p.add_argument("--coverage-report-output", type=Path, default=DEFAULT_COVERAGE_REPORT)
    p.add_argument("--refresh", action="store_true", help="Re-fetch every CUI, ignoring the cache.")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )
    args = _parse_args(argv)
    run_enrichment(
        mapping_input=args.mapping_input,
        metadata_parquet_output=args.metadata_parquet_output,
        metadata_csv_output=args.metadata_csv_output,
        cache_path=args.cache_path,
        audit_output=args.audit_output,
        coverage_report_output=args.coverage_report_output,
        refresh=args.refresh,
    )


if __name__ == "__main__":
    main()
