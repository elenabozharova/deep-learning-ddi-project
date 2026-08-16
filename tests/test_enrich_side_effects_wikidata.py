"""
Tests for polyllm.data.enrich_side_effects_wikidata — Experiment 1 (Wikidata
coverage/feasibility audit).

All HTTP calls are mocked. No live Wikidata access.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from polyllm.data.enrich_side_effects_wikidata import (
    STATUS_INVALID_CUI,
    STATUS_MAPPED_UNIQUE,
    STATUS_MULTIPLE_MATCHES,
    STATUS_NOT_FOUND,
    STATUS_REQUEST_FAILED,
    build_coverage_report,
    build_metadata_row,
    clean_aliases,
    extract_claim_values,
    extract_description,
    extract_icd_ids,
    extract_label,
    extract_raw_aliases,
    fetch_entities_batch,
    fetch_matches_batch,
    load_cache,
    run_enrichment,
    save_cache,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mapping_df(n: int = 3) -> pd.DataFrame:
    rows = [
        {
            "label_index": i,
            "side_effect_id": f"C{str(i).zfill(7)}",
            "side_effect_name": f"condition_{i}",
            "unique_pair_count": 100 + i,
        }
        for i in range(n)
    ]
    return pd.DataFrame(rows)


def _mock_response(status_code: int, json_data: dict | None = None) -> MagicMock:
    resp = MagicMock()
    resp.status_code = status_code
    if json_data is not None:
        resp.json.return_value = json_data
    return resp


def _sparql_result(bindings: list[dict]) -> dict:
    return {"head": {"vars": ["cui", "item"]}, "results": {"bindings": bindings}}


def _binding(cui: str, qid: str) -> dict:
    return {"cui": {"type": "literal", "value": cui}, "item": {"type": "uri", "value": f"http://www.wikidata.org/entity/{qid}"}}


ENTITY_JSON = {
    "labels": {"en": {"language": "en", "value": "abdominal pain"}},
    "descriptions": {"en": {"language": "en", "value": "pain in the abdomen"}},
    "aliases": {"en": [
        {"language": "en", "value": "stomach pain"},
        {"language": "en", "value": "Stomach Pain"},  # case-dup
        {"language": "en", "value": "abdominal pain"},  # same as label
    ]},
    "claims": {
        "P486": [{"mainsnak": {"datavalue": {"value": "D015746"}}}],
        "P5806": [{"mainsnak": {"datavalue": {"value": "271681002"}}}],
        "P31": [{"mainsnak": {"datavalue": {"value": {"id": "Q112965645"}}}}],
        "P279": [{"mainsnak": {"datavalue": {"value": {"id": "Q81938"}}}}],
    },
}


# ---------------------------------------------------------------------------
# Field extraction
# ---------------------------------------------------------------------------

class TestExtraction:
    def test_label_description(self):
        assert extract_label(ENTITY_JSON) == "abdominal pain"
        assert extract_description(ENTITY_JSON) == "pain in the abdomen"

    def test_missing_description_returns_none(self):
        assert extract_description({"labels": {}, "descriptions": {}}) is None

    def test_raw_aliases(self):
        assert extract_raw_aliases(ENTITY_JSON) == ["stomach pain", "Stomach Pain", "abdominal pain"]

    def test_claim_values_string_and_entity(self):
        assert extract_claim_values(ENTITY_JSON, "P486") == ["D015746"]
        assert extract_claim_values(ENTITY_JSON, "P31") == ["Q112965645"]

    def test_claim_values_missing_property(self):
        assert extract_claim_values(ENTITY_JSON, "P9999999") == []

    def test_icd_ids_extraction(self):
        entity = {"claims": {"P494": [{"mainsnak": {"datavalue": {"value": "R10"}}}]}}
        result = extract_icd_ids(entity)
        assert result == [{"scheme": "ICD-10 ID", "property": "P494", "value": "R10"}]


# ---------------------------------------------------------------------------
# Phase 6: alias cleaning
# ---------------------------------------------------------------------------

class TestAliasCleaning:
    def test_dedup_and_exclude_preferred_label(self):
        cleaned = clean_aliases(extract_raw_aliases(ENTITY_JSON), "abdominal pain")
        assert cleaned == ["stomach pain"]

    def test_empty_input(self):
        assert clean_aliases([], "x") == []
        assert clean_aliases(None, "x") == []

    def test_deterministic_ordering(self):
        assert clean_aliases(["Zeta", "alpha", "Beta"], "pref") == ["alpha", "Beta", "Zeta"]

    def test_whitespace_trimmed(self):
        assert clean_aliases(["  padded  "], "x") == ["padded"]


# ---------------------------------------------------------------------------
# Phase 2: matching (unique / none / multiple)
# ---------------------------------------------------------------------------

class TestMatching:
    def test_unique_match(self):
        with patch("requests.get") as mock_get:
            mock_get.return_value = _mock_response(200, _sparql_result([_binding("C0000731", "Q1")]))
            result = fetch_matches_batch(["C0000731"])
        assert result == {"C0000731": ["Q1"]}

    def test_no_match(self):
        with patch("requests.get") as mock_get:
            mock_get.return_value = _mock_response(200, _sparql_result([]))
            result = fetch_matches_batch(["C9999999"])
        assert result == {"C9999999": []}

    def test_multiple_matches(self):
        with patch("requests.get") as mock_get:
            mock_get.return_value = _mock_response(
                200, _sparql_result([_binding("C0000768", "Q1"), _binding("C0000768", "Q2")])
            )
            result = fetch_matches_batch(["C0000768"])
        assert result == {"C0000768": ["Q1", "Q2"]}

    def test_request_failure_returns_none(self):
        with patch("requests.get") as mock_get:
            mock_get.return_value = _mock_response(503)
            with patch("time.sleep"):
                result = fetch_matches_batch(["C0000731"])
        assert result is None

    def test_retry_then_success(self):
        with patch("requests.get") as mock_get:
            mock_get.side_effect = [_mock_response(503), _mock_response(200, _sparql_result([_binding("C0000731", "Q1")]))]
            with patch("time.sleep"):
                result = fetch_matches_batch(["C0000731"])
        assert result == {"C0000731": ["Q1"]}


class TestEntityFetch:
    def test_batch_fetch(self):
        with patch("requests.get") as mock_get:
            mock_get.return_value = _mock_response(200, {"entities": {"Q1": ENTITY_JSON}})
            result = fetch_entities_batch(["Q1"])
        assert result["Q1"]["labels"]["en"]["value"] == "abdominal pain"

    def test_empty_qid_list_short_circuits(self):
        with patch("requests.get") as mock_get:
            result = fetch_entities_batch([])
        assert result == {}
        mock_get.assert_not_called()


# ---------------------------------------------------------------------------
# Caching
# ---------------------------------------------------------------------------

class TestCaching:
    def test_roundtrip(self, tmp_path):
        cache_path = tmp_path / "cache.json"
        cache = {"cuis": {"C0000731": {"match_status": "mapped_unique", "qids": ["Q1"]}}, "entities": {}, "structural_labels": {}}
        save_cache(cache, cache_path)
        assert load_cache(cache_path) == cache

    def test_missing_file_returns_default_shape(self, tmp_path):
        loaded = load_cache(tmp_path / "nope.json")
        assert loaded == {"cuis": {}, "entities": {}, "structural_labels": {}}

    def test_rerun_reuses_cache_without_new_requests(self, tmp_path):
        mapping_path = tmp_path / "mapping.csv"
        _mapping_df(1).to_csv(mapping_path, index=False)
        kwargs = dict(
            mapping_input=mapping_path,
            metadata_parquet_output=tmp_path / "meta.parquet",
            metadata_csv_output=tmp_path / "meta.csv",
            cache_path=tmp_path / "cache.json",
            audit_output=tmp_path / "audit.json",
            coverage_report_output=tmp_path / "coverage.md",
        )
        with patch("requests.get") as mock_get:
            mock_get.side_effect = [
                _mock_response(200, _sparql_result([_binding("C0000000", "Q1")])),
                _mock_response(200, {"entities": {"Q1": ENTITY_JSON}}),
                _mock_response(200, {"entities": {"Q112965645": {"labels": {"en": {"value": "x"}}}, "Q81938": {"labels": {"en": {"value": "y"}}}}}),
            ]
            run_enrichment(**kwargs)

        with patch("requests.get") as mock_get2:
            run_enrichment(**kwargs)
            assert mock_get2.call_count == 0


# ---------------------------------------------------------------------------
# End-to-end row building
# ---------------------------------------------------------------------------

class TestBuildMetadataRow:
    def test_invalid_cui_skips_lookup(self):
        row = build_metadata_row(0, "NOT-A-CUI", "x", {"cuis": {}, "entities": {}, "structural_labels": {}}, "bad format")
        assert row["mapping_status"] == STATUS_INVALID_CUI
        assert row["wikidata_qid"] is None

    def test_not_found(self):
        cache = {"cuis": {"C1": {"match_status": STATUS_NOT_FOUND, "qids": []}}, "entities": {}, "structural_labels": {}}
        row = build_metadata_row(0, "C1", "x", cache, None)
        assert row["mapping_status"] == STATUS_NOT_FOUND
        assert row["wikidata_match_count"] == 0

    def test_multiple_matches_no_metadata_chosen(self):
        cache = {"cuis": {"C1": {"match_status": STATUS_MULTIPLE_MATCHES, "qids": ["Q1", "Q2"]}}, "entities": {}, "structural_labels": {}}
        row = build_metadata_row(0, "C1", "x", cache, None)
        assert row["mapping_status"] == STATUS_MULTIPLE_MATCHES
        assert row["wikidata_qid"] is None
        assert json.loads(row["ambiguous_qids"]) == ["Q1", "Q2"]

    def test_mapped_unique_full_row(self):
        cache = {
            "cuis": {"C1": {"match_status": STATUS_MAPPED_UNIQUE, "qids": ["Q1"]}},
            "entities": {"Q1": ENTITY_JSON},
            "structural_labels": {"Q112965645": "disease", "Q81938": "pain"},
        }
        row = build_metadata_row(0, "C1", "abdominal pain", cache, None)
        assert row["mapping_status"] == STATUS_MAPPED_UNIQUE
        assert row["wikidata_qid"] == "Q1"
        assert row["wikidata_label"] == "abdominal pain"
        assert json.loads(row["wikidata_aliases"]) == ["stomach pain"]
        assert json.loads(row["mesh_id"]) == ["D015746"]
        assert json.loads(row["wikidata_instance_of"]) == [{"qid": "Q112965645", "label": "disease"}]
        assert row["has_any_crossref"] is True


# ---------------------------------------------------------------------------
# Coverage report
# ---------------------------------------------------------------------------

class TestCoverageReport:
    def test_counts(self):
        cache = {
            "cuis": {
                "C0": {"match_status": STATUS_MAPPED_UNIQUE, "qids": ["Q1"]},
                "C1": {"match_status": STATUS_NOT_FOUND, "qids": []},
            },
            "entities": {"Q1": ENTITY_JSON},
            "structural_labels": {},
        }
        rows = [
            build_metadata_row(0, "C0", "a", cache, None),
            build_metadata_row(1, "C1", "b", cache, None),
        ]
        metadata_df = pd.DataFrame(rows)
        coverage = build_coverage_report(metadata_df)
        assert coverage["total_side_effects"] == 2
        assert coverage["mapped_unique_count"] == 1
        assert coverage["not_found_count"] == 1
        assert coverage["mapped_field_coverage"]["english_label"]["count"] == 1


# ---------------------------------------------------------------------------
# Full pipeline: alignment, determinism, no mutation of originals
# ---------------------------------------------------------------------------

class TestPipeline:
    def test_label_index_and_full_alignment_preserved(self, tmp_path):
        mapping_path = tmp_path / "mapping.csv"
        mapping_df = _mapping_df(3)
        mapping_df.to_csv(mapping_path, index=False)

        def fake_get(url, params=None, headers=None, timeout=None):
            if "sparql" in url:
                cui = params["query"].split('"')[1]
                return _mock_response(200, _sparql_result([_binding(cui, "Q1")]))
            if params.get("action") == "wbgetentities" and "claims" in params.get("props", ""):
                ids = params["ids"].split("|")
                return _mock_response(200, {"entities": {q: ENTITY_JSON for q in ids}})
            return _mock_response(200, {"entities": {}})

        with patch("requests.get", side_effect=fake_get):
            run_enrichment(
                mapping_input=mapping_path,
                metadata_parquet_output=tmp_path / "meta.parquet",
                metadata_csv_output=tmp_path / "meta.csv",
                cache_path=tmp_path / "cache.json",
                audit_output=tmp_path / "audit.json",
                coverage_report_output=tmp_path / "coverage.md",
            )

        metadata_df = pd.read_csv(tmp_path / "meta.csv")
        assert (metadata_df["label_index"].to_numpy() == mapping_df["label_index"].to_numpy()).all()
        assert (metadata_df["cui"].to_numpy() == mapping_df["side_effect_id"].to_numpy()).all()

    def test_original_mapping_file_untouched(self, tmp_path):
        mapping_path = tmp_path / "mapping.csv"
        _mapping_df(1).to_csv(mapping_path, index=False)
        original_bytes = mapping_path.read_bytes()

        with patch("requests.get") as mock_get:
            mock_get.side_effect = [
                _mock_response(200, _sparql_result([_binding("C0000000", "Q1")])),
                _mock_response(200, {"entities": {"Q1": ENTITY_JSON}}),
                _mock_response(200, {"entities": {}}),
            ]
            run_enrichment(
                mapping_input=mapping_path,
                metadata_parquet_output=tmp_path / "meta.parquet",
                metadata_csv_output=tmp_path / "meta.csv",
                cache_path=tmp_path / "cache.json",
                audit_output=tmp_path / "audit.json",
                coverage_report_output=tmp_path / "coverage.md",
            )

        assert mapping_path.read_bytes() == original_bytes

    def test_deterministic_output(self, tmp_path):
        mapping_path = tmp_path / "mapping.csv"
        _mapping_df(1).to_csv(mapping_path, index=False)

        def _run(suffix):
            with patch("requests.get") as mock_get:
                mock_get.side_effect = [
                    _mock_response(200, _sparql_result([_binding("C0000000", "Q1")])),
                    _mock_response(200, {"entities": {"Q1": ENTITY_JSON}}),
                    _mock_response(200, {"entities": {}}),
                ]
                run_enrichment(
                    mapping_input=mapping_path,
                    metadata_parquet_output=tmp_path / f"meta{suffix}.parquet",
                    metadata_csv_output=tmp_path / f"meta{suffix}.csv",
                    cache_path=tmp_path / f"cache{suffix}.json",
                    audit_output=tmp_path / f"audit{suffix}.json",
                    coverage_report_output=tmp_path / f"coverage{suffix}.md",
                )
            df = pd.read_csv(tmp_path / f"meta{suffix}.csv")
            return df.drop(columns=["retrieved_at"])

        pd.testing.assert_frame_equal(_run("_a"), _run("_b"))
