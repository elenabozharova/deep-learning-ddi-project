"""
Experiment 1 — Wikidata coverage/feasibility audit for PolyLLM side-effect
concepts, using the existing UMLS CUI (Wikidata property P2892) as the join
key.

Context: the UMLS REST enrichment pipeline
(`polyllm.data.enrich_side_effects_umls`) is blocked on UMLS API-key
eligibility. This module asks a narrower question first — can the *public,
unauthenticated* Wikidata endpoints supply enough text (label/aliases/
description) and biomedical cross-references to be useful for Experiment 1,
either as a stand-in or as a bridge to open ontologies?

This module does NOT touch the UMLS pipeline, does not generate embeddings,
and does not modify any existing PolyLLM artifact. It reuses
`validate_mapping_df` and `validate_label_alignment` from the existing
enrichment modules rather than reimplementing them.

Public endpoints used (both unauthenticated, per Wikimedia usage policy —
a descriptive User-Agent is sent, no API key required):
  - SPARQL:    https://query.wikidata.org/sparql            (CUI -> QID matching)
  - Action API: https://www.wikidata.org/w/api.php           (QID -> entity metadata)

Run from the project root:
    python src/polyllm/data/enrich_side_effects_wikidata.py
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import platform
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from polyllm.data.enrich_side_effects_umls import CUI_PATTERN, validate_label_alignment
from polyllm.features.generate_side_effect_embeddings import validate_mapping_df

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SPARQL_ENDPOINT = "https://query.wikidata.org/sparql"
WIKIDATA_API = "https://www.wikidata.org/w/api.php"
USER_AGENT = "ddi-nlp-project-experiment1/1.0 (research; contact: bozharov.petar@gmail.com)"

UMLS_CUI_PROPERTY = "P2892"

# Biomedical cross-reference properties, verified live against Wikidata's own
# property-search API (see notes/experiment_1_wikidata_coverage.md, Phase 1).
CROSSREF_PROPERTIES: dict[str, str] = {
    "mesh_id": "P486",
    "mondo_id": "P5270",
    "snomed_id": "P5806",
    "doid": "P699",
    "hpo_id": "P3841",
}
ICD_PROPERTIES: dict[str, str] = {
    "P494": "ICD-10 ID",
    "P4229": "ICD-10-CM",
    "P493": "ICD-9 ID",
    "P1692": "ICD-9-CM",
}
STRUCTURAL_PROPERTIES: dict[str, str] = {
    "instance_of": "P31",
    "subclass_of": "P279",
}

MATCH_BATCH_SIZE = 50     # CUIs per SPARQL VALUES query
ENTITY_BATCH_SIZE = 50    # QIDs per wbgetentities call (API max is 50)

REQUEST_TIMEOUT = 30
RETRY_COUNT = 3
RETRY_BACKOFF = [1, 2, 4]
INTER_BATCH_SLEEP = 1.0   # polite pacing on shared public infrastructure

DEFAULT_MAPPING_INPUT = Path("data/processed/polyllm_label_mapping.csv")
DEFAULT_METADATA_PARQUET = Path("data/processed/wikidata_side_effect_metadata.parquet")
DEFAULT_METADATA_CSV = Path("data/processed/wikidata_side_effect_metadata.csv")
DEFAULT_CACHE_PATH = Path("outputs/polyllm/wikidata_cache.json")
DEFAULT_AUDIT_OUTPUT = Path("outputs/polyllm/wikidata_enrichment_audit.json")
DEFAULT_COVERAGE_REPORT = Path("outputs/polyllm/wikidata_coverage_report.md")

METADATA_COLUMNS = [
    "label_index",
    "side_effect_name",
    "cui",
    "wikidata_qid",
    "wikidata_label",
    "wikidata_aliases",
    "wikidata_description",
    "mesh_id",
    "mondo_id",
    "snomed_id",
    "icd_ids",
    "doid",
    "hpo_id",
    "wikidata_instance_of",
    "wikidata_subclass_of",
    "wikidata_match_count",
    "ambiguous_qids",
    "alias_count",
    "has_label",
    "has_aliases",
    "has_description",
    "has_any_crossref",
    "has_structural_info",
    "mapping_status",
    "mapping_error",
    "retrieved_at",
]

# mapping_status values
STATUS_MAPPED_UNIQUE = "mapped_unique"
STATUS_NOT_FOUND = "not_found"
STATUS_MULTIPLE_MATCHES = "multiple_matches"
STATUS_REQUEST_FAILED = "request_failed"
STATUS_INVALID_CUI = "invalid_cui"


# ---------------------------------------------------------------------------
# HTTP layer (shared retry helper for both public endpoints)
# ---------------------------------------------------------------------------

def _get_with_retry(url: str, params: dict[str, Any]) -> requests.Response | None:
    headers = {"User-Agent": USER_AGENT}
    for attempt, backoff in enumerate(RETRY_BACKOFF[:RETRY_COUNT], start=1):
        try:
            resp = requests.get(url, params=params, headers=headers, timeout=REQUEST_TIMEOUT)
            if resp.status_code == 200:
                return resp
            if resp.status_code in (429, 500, 502, 503, 504) and attempt < RETRY_COUNT:
                logger.warning(
                    "Transient HTTP %d from %s; retrying in %ds (attempt %d/%d).",
                    resp.status_code, url, backoff, attempt, RETRY_COUNT,
                )
                time.sleep(backoff)
                continue
            logger.error("Non-200 response HTTP %d from %s", resp.status_code, url)
            return resp
        except requests.exceptions.Timeout:
            logger.warning("Timeout on %s (attempt %d/%d).", url, attempt, RETRY_COUNT)
            if attempt < RETRY_COUNT:
                time.sleep(backoff)
        except requests.exceptions.RequestException as exc:
            logger.warning("Request error on %s: %s (attempt %d/%d).", url, exc, attempt, RETRY_COUNT)
            if attempt < RETRY_COUNT:
                time.sleep(backoff)

    logger.error("All retries exhausted for %s.", url)
    return None


def _build_match_query(cuis: list[str]) -> str:
    values = " ".join(json.dumps(c) for c in cuis)  # safe literal quoting
    return (
        "SELECT ?cui ?item WHERE { "
        f"VALUES ?cui {{ {values} }} "
        f"?item wdt:{UMLS_CUI_PROPERTY} ?cui . "
        "}"
    )


def fetch_matches_batch(cuis: list[str]) -> dict[str, list[str]] | None:
    """
    Query Wikidata for every QID whose P2892 (UMLS CUI) equals one of `cuis`.

    Returns {cui: [qid, ...]} for every requested CUI (empty list = no match),
    or None if the request failed after retries.
    """
    query = _build_match_query(cuis)
    resp = _get_with_retry(SPARQL_ENDPOINT, {"query": query, "format": "json"})
    if resp is None or resp.status_code != 200:
        return None

    matches: dict[str, list[str]] = {c: [] for c in cuis}
    for binding in resp.json()["results"]["bindings"]:
        cui = binding["cui"]["value"]
        qid = binding["item"]["value"].rsplit("/", 1)[-1]
        if qid not in matches.setdefault(cui, []):
            matches[cui].append(qid)
    return matches


def fetch_entities_batch(qids: list[str]) -> dict[str, dict[str, Any]] | None:
    """Batch-fetch full entity records (labels/aliases/descriptions/claims) for up to 50 QIDs."""
    if not qids:
        return {}
    resp = _get_with_retry(WIKIDATA_API, {
        "action": "wbgetentities",
        "ids": "|".join(qids),
        "languages": "en",
        "props": "labels|aliases|descriptions|claims",
        "format": "json",
    })
    if resp is None or resp.status_code != 200:
        return None
    return resp.json().get("entities", {})


def fetch_entity_labels_batch(qids: list[str]) -> dict[str, str] | None:
    """Batch-fetch just the English label for up to 50 QIDs (used for structural targets)."""
    if not qids:
        return {}
    resp = _get_with_retry(WIKIDATA_API, {
        "action": "wbgetentities",
        "ids": "|".join(qids),
        "languages": "en",
        "props": "labels",
        "format": "json",
    })
    if resp is None or resp.status_code != 200:
        return None
    entities = resp.json().get("entities", {})
    return {qid: e.get("labels", {}).get("en", {}).get("value", "") for qid, e in entities.items()}


# ---------------------------------------------------------------------------
# Entity field extraction
# ---------------------------------------------------------------------------

def extract_label(entity: dict[str, Any]) -> str | None:
    return entity.get("labels", {}).get("en", {}).get("value")


def extract_description(entity: dict[str, Any]) -> str | None:
    return entity.get("descriptions", {}).get("en", {}).get("value")


def extract_raw_aliases(entity: dict[str, Any]) -> list[str]:
    return [a["value"] for a in entity.get("aliases", {}).get("en", [])]


def clean_aliases(raw_aliases: list[str], preferred_label: str | None) -> list[str]:
    """
    Deterministic alias cleanup (Phase 6): trim, drop empty, case-insensitive
    de-duplicate, drop aliases identical to the preferred label, sort for
    reproducibility. No aggressive normalization of wording.
    """
    if not raw_aliases:
        return []
    preferred_norm = (preferred_label or "").strip().casefold()
    seen: set[str] = set()
    cleaned: list[str] = []
    for alias in raw_aliases:
        name = (alias or "").strip()
        if not name:
            continue
        norm = name.casefold()
        if norm == preferred_norm or norm in seen:
            continue
        seen.add(norm)
        cleaned.append(name)
    return sorted(cleaned, key=str.casefold)


def extract_claim_values(entity: dict[str, Any], property_id: str) -> list[str]:
    """Extract all string/entity-id values for a given property from an entity's claims."""
    values: list[str] = []
    for claim in entity.get("claims", {}).get(property_id, []):
        datavalue = claim.get("mainsnak", {}).get("datavalue", {})
        value = datavalue.get("value")
        if isinstance(value, dict) and "id" in value:
            values.append(value["id"])
        elif isinstance(value, str):
            values.append(value)
    return sorted(set(values))


def extract_icd_ids(entity: dict[str, Any]) -> list[dict[str, str]]:
    """Collect values across all tracked ICD property variants (Phase 3: 'icd_ids' plural)."""
    out: list[dict[str, str]] = []
    for pid, label in ICD_PROPERTIES.items():
        for value in extract_claim_values(entity, pid):
            out.append({"scheme": label, "property": pid, "value": value})
    return out


# ---------------------------------------------------------------------------
# Disk cache
# ---------------------------------------------------------------------------

def load_cache(cache_path: Path) -> dict[str, Any]:
    if cache_path.exists():
        try:
            return json.loads(cache_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Cache unreadable (%s) — starting fresh.", exc)
    return {"cuis": {}, "entities": {}, "structural_labels": {}}


def save_cache(cache: dict[str, Any], cache_path: Path) -> None:
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


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Cache population passes
# ---------------------------------------------------------------------------

def populate_match_cache(cuis: list[str], cache: dict[str, Any], cache_path: Path, refresh: bool) -> None:
    """Pass A: resolve CUI -> QID list for every CUI not yet cached (or refresh=True)."""
    pending = [
        c for c in cuis
        if refresh or c not in cache["cuis"] or cache["cuis"][c]["match_status"] == STATUS_REQUEST_FAILED
    ]
    logger.info("Match phase: %d/%d CUIs need querying.", len(pending), len(cuis))

    for start in range(0, len(pending), MATCH_BATCH_SIZE):
        batch = pending[start:start + MATCH_BATCH_SIZE]
        logger.info("Match batch %d-%d/%d", start + 1, start + len(batch), len(pending))
        result = fetch_matches_batch(batch)
        if result is None:
            for c in batch:
                cache["cuis"][c] = {"match_status": STATUS_REQUEST_FAILED, "qids": [], "matched_at": _now_iso()}
        else:
            for c, qids in result.items():
                if len(qids) == 0:
                    status = STATUS_NOT_FOUND
                elif len(qids) == 1:
                    status = STATUS_MAPPED_UNIQUE
                else:
                    status = STATUS_MULTIPLE_MATCHES
                cache["cuis"][c] = {"match_status": status, "qids": qids, "matched_at": _now_iso()}
        save_cache(cache, cache_path)
        if start + MATCH_BATCH_SIZE < len(pending):
            time.sleep(INTER_BATCH_SLEEP)


def populate_entity_cache(cache: dict[str, Any], cache_path: Path, refresh: bool) -> None:
    """Pass B: fetch full entity metadata for every uniquely-matched QID."""
    unique_qids = sorted({
        rec["qids"][0]
        for rec in cache["cuis"].values()
        if rec["match_status"] == STATUS_MAPPED_UNIQUE
    })
    pending = [q for q in unique_qids if refresh or q not in cache["entities"]]
    logger.info("Entity phase: %d/%d unique QIDs need fetching.", len(pending), len(unique_qids))

    for start in range(0, len(pending), ENTITY_BATCH_SIZE):
        batch = pending[start:start + ENTITY_BATCH_SIZE]
        logger.info("Entity batch %d-%d/%d", start + 1, start + len(batch), len(pending))
        result = fetch_entities_batch(batch)
        if result is None:
            for q in batch:
                cache["entities"][q] = {"_fetch_failed": True}
        else:
            for q in batch:
                cache["entities"][q] = result.get(q, {"_fetch_failed": True})
        save_cache(cache, cache_path)
        if start + ENTITY_BATCH_SIZE < len(pending):
            time.sleep(INTER_BATCH_SLEEP)


def populate_structural_labels(cache: dict[str, Any], cache_path: Path, refresh: bool) -> None:
    """Pass C: one bounded (non-recursive) hop to resolve instance_of/subclass_of target labels."""
    target_qids: set[str] = set()
    for entity in cache["entities"].values():
        if entity.get("_fetch_failed"):
            continue
        for pid in STRUCTURAL_PROPERTIES.values():
            target_qids.update(extract_claim_values(entity, pid))

    pending = sorted(q for q in target_qids if refresh or q not in cache["structural_labels"])
    logger.info("Structural-label phase: %d/%d target QIDs need resolving.", len(pending), len(target_qids))

    for start in range(0, len(pending), ENTITY_BATCH_SIZE):
        batch = pending[start:start + ENTITY_BATCH_SIZE]
        result = fetch_entity_labels_batch(batch)
        if result is None:
            for q in batch:
                cache["structural_labels"][q] = ""
        else:
            cache["structural_labels"].update(result)
        save_cache(cache, cache_path)
        if start + ENTITY_BATCH_SIZE < len(pending):
            time.sleep(INTER_BATCH_SLEEP)


# ---------------------------------------------------------------------------
# Cache -> canonical row
# ---------------------------------------------------------------------------

def _structural_list(entity: dict[str, Any], property_id: str, label_lookup: dict[str, str]) -> list[dict[str, str]]:
    return [
        {"qid": qid, "label": label_lookup.get(qid, "")}
        for qid in extract_claim_values(entity, property_id)
    ]


def build_metadata_row(
    label_index: int,
    cui: str,
    side_effect_name: str,
    cache: dict[str, Any],
    invalid_cui_reason: str | None,
) -> dict[str, Any]:
    if invalid_cui_reason is not None:
        return _empty_row(label_index, cui, side_effect_name, STATUS_INVALID_CUI, invalid_cui_reason, [], 0)

    match_rec = cache["cuis"].get(cui)
    if match_rec is None:
        return _empty_row(label_index, cui, side_effect_name, STATUS_REQUEST_FAILED, "no cached match record", [], 0)

    status = match_rec["match_status"]
    qids = match_rec["qids"]

    if status in (STATUS_NOT_FOUND, STATUS_REQUEST_FAILED):
        return _empty_row(label_index, cui, side_effect_name, status, None, [], len(qids))

    if status == STATUS_MULTIPLE_MATCHES:
        return _empty_row(label_index, cui, side_effect_name, status, None, qids, len(qids))

    # mapped_unique
    qid = qids[0]
    entity = cache["entities"].get(qid, {})
    if entity.get("_fetch_failed"):
        return _empty_row(label_index, cui, side_effect_name, STATUS_REQUEST_FAILED, "entity fetch failed", [], 1)

    label = extract_label(entity)
    description = extract_description(entity)
    aliases = clean_aliases(extract_raw_aliases(entity), label)
    crossrefs = {field: extract_claim_values(entity, pid) for field, pid in CROSSREF_PROPERTIES.items()}
    icd_ids = extract_icd_ids(entity)
    labels_lookup = cache.get("structural_labels", {})
    instance_of = _structural_list(entity, STRUCTURAL_PROPERTIES["instance_of"], labels_lookup)
    subclass_of = _structural_list(entity, STRUCTURAL_PROPERTIES["subclass_of"], labels_lookup)

    has_any_crossref = any(crossrefs.values()) or bool(icd_ids)

    return {
        "label_index": label_index,
        "side_effect_name": side_effect_name,
        "cui": cui,
        "wikidata_qid": qid,
        "wikidata_label": label,
        "wikidata_aliases": json.dumps(aliases, ensure_ascii=False),
        "wikidata_description": description,
        "mesh_id": json.dumps(crossrefs["mesh_id"], ensure_ascii=False),
        "mondo_id": json.dumps(crossrefs["mondo_id"], ensure_ascii=False),
        "snomed_id": json.dumps(crossrefs["snomed_id"], ensure_ascii=False),
        "icd_ids": json.dumps(icd_ids, ensure_ascii=False),
        "doid": json.dumps(crossrefs["doid"], ensure_ascii=False),
        "hpo_id": json.dumps(crossrefs["hpo_id"], ensure_ascii=False),
        "wikidata_instance_of": json.dumps(instance_of, ensure_ascii=False),
        "wikidata_subclass_of": json.dumps(subclass_of, ensure_ascii=False),
        "wikidata_match_count": 1,
        "ambiguous_qids": json.dumps([], ensure_ascii=False),
        "alias_count": len(aliases),
        "has_label": label is not None,
        "has_aliases": len(aliases) > 0,
        "has_description": description is not None,
        "has_any_crossref": has_any_crossref,
        "has_structural_info": bool(instance_of or subclass_of),
        "mapping_status": STATUS_MAPPED_UNIQUE,
        "mapping_error": None,
        "retrieved_at": _now_iso(),
    }


def _empty_row(
    label_index: int, cui: str, side_effect_name: str,
    status: str, error: str | None, ambiguous_qids: list[str], match_count: int,
) -> dict[str, Any]:
    return {
        "label_index": label_index,
        "side_effect_name": side_effect_name,
        "cui": cui,
        "wikidata_qid": None,
        "wikidata_label": None,
        "wikidata_aliases": json.dumps([]),
        "wikidata_description": None,
        "mesh_id": json.dumps([]),
        "mondo_id": json.dumps([]),
        "snomed_id": json.dumps([]),
        "icd_ids": json.dumps([]),
        "doid": json.dumps([]),
        "hpo_id": json.dumps([]),
        "wikidata_instance_of": json.dumps([]),
        "wikidata_subclass_of": json.dumps([]),
        "wikidata_match_count": match_count,
        "ambiguous_qids": json.dumps(ambiguous_qids, ensure_ascii=False),
        "alias_count": 0,
        "has_label": False,
        "has_aliases": False,
        "has_description": False,
        "has_any_crossref": False,
        "has_structural_info": False,
        "mapping_status": status,
        "mapping_error": error,
        "retrieved_at": _now_iso() if status != STATUS_INVALID_CUI else None,
    }


# ---------------------------------------------------------------------------
# Coverage report (Phases 8 & 9)
# ---------------------------------------------------------------------------

def build_coverage_report(metadata_df: pd.DataFrame) -> dict[str, Any]:
    total = len(metadata_df)

    def pct(n: int) -> float:
        return round(100 * n / total, 2) if total else 0.0

    status_counts = metadata_df["mapping_status"].value_counts().to_dict()
    unique_n = status_counts.get(STATUS_MAPPED_UNIQUE, 0)
    not_found_n = status_counts.get(STATUS_NOT_FOUND, 0)
    multi_n = status_counts.get(STATUS_MULTIPLE_MATCHES, 0)
    failed_n = status_counts.get(STATUS_REQUEST_FAILED, 0)

    mapped = metadata_df[metadata_df["mapping_status"] == STATUS_MAPPED_UNIQUE]
    mapped_n = len(mapped)

    def mapped_pct(n: int) -> float:
        return round(100 * n / mapped_n, 2) if mapped_n else 0.0

    alias_counts = mapped["alias_count"]
    has_label = int(mapped["has_label"].sum())
    has_aliases = int(mapped["has_aliases"].sum())
    has_desc = int(mapped["has_description"].sum())
    has_struct = int(mapped["has_structural_info"].sum())

    def crossref_coverage(col: str) -> int:
        return int(mapped[col].apply(lambda v: len(json.loads(v)) > 0).sum())

    name_only = int(((~mapped["has_aliases"]) & (~mapped["has_description"])).sum())
    name_aliases = int((mapped["has_aliases"] & (~mapped["has_description"])).sum())
    name_desc = int(((~mapped["has_aliases"]) & mapped["has_description"]).sum())
    name_aliases_desc = int((mapped["has_aliases"] & mapped["has_description"]).sum())

    ambiguous = metadata_df[metadata_df["mapping_status"] == STATUS_MULTIPLE_MATCHES]
    not_found = metadata_df[metadata_df["mapping_status"] == STATUS_NOT_FOUND]
    failed = metadata_df[metadata_df["mapping_status"] == STATUS_REQUEST_FAILED]

    return {
        "total_side_effects": total,
        "unique_cuis": int(metadata_df["cui"].nunique()),
        "mapped_unique_count": unique_n,
        "mapped_unique_percent": pct(unique_n),
        "not_found_count": not_found_n,
        "not_found_percent": pct(not_found_n),
        "multiple_matches_count": multi_n,
        "multiple_matches_percent": pct(multi_n),
        "request_failed_count": failed_n,
        "request_failed_percent": pct(failed_n),
        "mapped_field_coverage": {
            "english_label": {"count": has_label, "percent": mapped_pct(has_label)},
            "ge1_alias": {"count": has_aliases, "percent": mapped_pct(has_aliases)},
            "english_description": {"count": has_desc, "percent": mapped_pct(has_desc)},
            "mesh_id": {"count": crossref_coverage("mesh_id"), "percent": mapped_pct(crossref_coverage("mesh_id"))},
            "mondo_id": {"count": crossref_coverage("mondo_id"), "percent": mapped_pct(crossref_coverage("mondo_id"))},
            "snomed_id": {"count": crossref_coverage("snomed_id"), "percent": mapped_pct(crossref_coverage("snomed_id"))},
            "icd_id": {"count": crossref_coverage("icd_ids"), "percent": mapped_pct(crossref_coverage("icd_ids"))},
            "doid": {"count": crossref_coverage("doid"), "percent": mapped_pct(crossref_coverage("doid"))},
            "hpo_id": {"count": crossref_coverage("hpo_id"), "percent": mapped_pct(crossref_coverage("hpo_id"))},
            "structural_info": {"count": has_struct, "percent": mapped_pct(has_struct)},
        },
        "alias_count_stats": {
            "mean": float(round(alias_counts.mean(), 3)) if mapped_n else 0.0,
            "median": float(alias_counts.median()) if mapped_n else 0.0,
            "p75": float(alias_counts.quantile(0.75)) if mapped_n else 0.0,
            "p90": float(alias_counts.quantile(0.90)) if mapped_n else 0.0,
            "p95": float(alias_counts.quantile(0.95)) if mapped_n else 0.0,
            "max": int(alias_counts.max()) if mapped_n else 0,
        },
        "experiment1_text_richness": {
            "A_name_only": {"n": name_only, "of_963": total, "percent": pct(name_only)},
            "B_name_plus_aliases": {"n": name_aliases, "of_963": total, "percent": pct(name_aliases)},
            "C_name_plus_description": {"n": name_desc, "of_963": total, "percent": pct(name_desc)},
            "D_name_plus_aliases_plus_description": {"n": name_aliases_desc, "of_963": total, "percent": pct(name_aliases_desc)},
        },
        "ambiguous_matches": ambiguous[["label_index", "cui", "side_effect_name", "ambiguous_qids"]].to_dict("records"),
        "not_found_examples": not_found[["label_index", "cui", "side_effect_name"]].to_dict("records"),
        "request_failed_examples": failed[["label_index", "cui", "side_effect_name"]].to_dict("records"),
    }


def render_coverage_markdown(coverage: dict[str, Any]) -> str:
    fc = coverage["mapped_field_coverage"]
    ac = coverage["alias_count_stats"]
    tr = coverage["experiment1_text_richness"]
    lines = [
        "# Wikidata Coverage Report",
        "",
        f"Generated: {_now_iso()}",
        "",
        "## Mapping outcome (963 total)",
        "",
        f"- Mapped, unique QID: **{coverage['mapped_unique_count']}** ({coverage['mapped_unique_percent']}%)",
        f"- Not found: **{coverage['not_found_count']}** ({coverage['not_found_percent']}%)",
        f"- Multiple matches (ambiguous): **{coverage['multiple_matches_count']}** ({coverage['multiple_matches_percent']}%)",
        f"- Request failed: **{coverage['request_failed_count']}** ({coverage['request_failed_percent']}%)",
        "",
        "## Field coverage among uniquely mapped concepts",
        "",
        "| Field | Count | Percent |",
        "|---|---|---|",
    ]
    for name, key in [
        ("English label", "english_label"), (">=1 alias", "ge1_alias"), ("English description", "english_description"),
        ("MeSH ID", "mesh_id"), ("MONDO ID", "mondo_id"), ("SNOMED CT ID", "snomed_id"),
        ("ICD ID (any variant)", "icd_id"), ("Disease Ontology ID", "doid"), ("HPO ID", "hpo_id"),
        ("Structural info (instance/subclass of)", "structural_info"),
    ]:
        v = fc[key]
        lines.append(f"| {name} | {v['count']} | {v['percent']}% |")

    lines += [
        "",
        "## Alias count distribution (mapped concepts)",
        "",
        f"- Mean: {ac['mean']}, Median: {ac['median']}, p75: {ac['p75']}, p90: {ac['p90']}, p95: {ac['p95']}, Max: {ac['max']}",
        "",
        "## Text richness for Experiment 1 (out of 963)",
        "",
        f"- A. name only: {tr['A_name_only']['n']} ({tr['A_name_only']['percent']}%)",
        f"- B. name + aliases: {tr['B_name_plus_aliases']['n']} ({tr['B_name_plus_aliases']['percent']}%)",
        f"- C. name + description: {tr['C_name_plus_description']['n']} ({tr['C_name_plus_description']['percent']}%)",
        f"- D. name + aliases + description: {tr['D_name_plus_aliases_plus_description']['n']} ({tr['D_name_plus_aliases_plus_description']['percent']}%)",
        "",
        "Note: a Wikidata description is a short community-written gloss, not a formal biomedical definition.",
        "",
        "## Ambiguous matches (multiple Wikidata entities share the same CUI)",
        "",
    ]
    if coverage["ambiguous_matches"]:
        for r in coverage["ambiguous_matches"]:
            lines.append(f"- [{r['label_index']}] {r['cui']} `{r['side_effect_name']}` — {r['ambiguous_qids']}")
    else:
        lines.append("None.")

    lines += ["", "## Not found", ""]
    lines.append(f"{len(coverage['not_found_examples'])} side effects had no Wikidata entity with this CUI.")

    lines += ["", "## Request failures", ""]
    if coverage["request_failed_examples"]:
        for r in coverage["request_failed_examples"]:
            lines.append(f"- [{r['label_index']}] {r['cui']} `{r['side_effect_name']}`")
    else:
        lines.append("None.")

    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Atomic write + software versions
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
    return {"python": platform.python_version(), "pandas": pd.__version__, "requests": requests.__version__}


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
) -> dict[str, Any]:
    """Full Wikidata coverage audit. Idempotent: reruns reuse cached results unless refresh=True."""
    logger.info("=== Experiment 1 — Wikidata coverage audit started ===")

    mapping_df = pd.read_csv(mapping_input)
    validate_mapping_df(mapping_df)
    mapping_df = mapping_df.sort_values("label_index").reset_index(drop=True)
    logger.info("Loaded label mapping: %d rows.", len(mapping_df))

    cache = load_cache(cache_path)
    valid_cuis = [c for c in mapping_df["side_effect_id"].astype(str) if CUI_PATTERN.match(c)]

    populate_match_cache(valid_cuis, cache, cache_path, refresh)
    populate_entity_cache(cache, cache_path, refresh)
    populate_structural_labels(cache, cache_path, refresh)

    rows: list[dict[str, Any]] = []
    for _, row in mapping_df.iterrows():
        cui = str(row["side_effect_id"])
        if not CUI_PATTERN.match(cui):
            rows.append(build_metadata_row(
                int(row["label_index"]), cui, row["side_effect_name"],
                cache, f"CUI does not match expected pattern: {cui!r}",
            ))
        else:
            rows.append(build_metadata_row(
                int(row["label_index"]), cui, row["side_effect_name"], cache, None,
            ))

    metadata_df = pd.DataFrame(rows, columns=METADATA_COLUMNS)

    # Reuse the generic alignment check from the UMLS module (label_index /
    # side_effect_id / side_effect_name columns only — this module's "cui"
    # column is exactly that same side_effect_id, so we alias it for the check.
    check_df = metadata_df.rename(columns={"cui": "side_effect_id"})
    validate_label_alignment(mapping_df, check_df)
    logger.info("Label alignment validated: metadata rows match mapping rows 1:1 in order.")

    for path in [metadata_parquet_output, metadata_csv_output, audit_output, coverage_report_output]:
        path.parent.mkdir(parents=True, exist_ok=True)

    metadata_df.to_parquet(metadata_parquet_output, index=False)
    metadata_df.to_csv(metadata_csv_output, index=False)
    logger.info("Wrote canonical metadata: %s, %s (%d rows)", metadata_parquet_output, metadata_csv_output, len(metadata_df))

    coverage = build_coverage_report(metadata_df)
    audit = {
        "generated_at": _now_iso(),
        "coverage": coverage,
        "software_versions": _software_versions(),
        "sparql_endpoint": SPARQL_ENDPOINT,
        "wikidata_api": WIKIDATA_API,
        "umls_cui_property": UMLS_CUI_PROPERTY,
        "crossref_properties": CROSSREF_PROPERTIES,
        "icd_properties": ICD_PROPERTIES,
        "structural_properties": STRUCTURAL_PROPERTIES,
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
        "=== Experiment 1 Wikidata audit complete: %d/%d mapped_unique ===",
        coverage["mapped_unique_count"], coverage["total_side_effects"],
    )
    return audit


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Experiment 1 — Wikidata coverage audit for PolyLLM side effects.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--mapping-input", type=Path, default=DEFAULT_MAPPING_INPUT)
    p.add_argument("--metadata-parquet-output", type=Path, default=DEFAULT_METADATA_PARQUET)
    p.add_argument("--metadata-csv-output", type=Path, default=DEFAULT_METADATA_CSV)
    p.add_argument("--cache-path", type=Path, default=DEFAULT_CACHE_PATH)
    p.add_argument("--audit-output", type=Path, default=DEFAULT_AUDIT_OUTPUT)
    p.add_argument("--coverage-report-output", type=Path, default=DEFAULT_COVERAGE_REPORT)
    p.add_argument("--refresh", action="store_true", help="Re-fetch everything, ignoring the cache.")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s — %(message)s", datefmt="%H:%M:%S")
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
