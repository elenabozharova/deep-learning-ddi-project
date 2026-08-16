"""
Tests for polyllm.data.enrich_side_effects_umls — Experiment 1.

All HTTP calls are mocked. No live UMLS access.
"""

from __future__ import annotations

import io
import json
import logging
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from polyllm.data.enrich_side_effects_umls import (
    STATUS_INVALID_CUI,
    STATUS_NOT_FOUND,
    STATUS_REQUEST_FAILED,
    STATUS_SUCCESS,
    build_coverage_report,
    build_metadata_row,
    clean_definitions,
    clean_synonyms,
    extract_semantic_types,
    fetch_cui_record,
    get_api_key,
    load_cache,
    run_enrichment,
    save_cache,
    validate_cuis,
    validate_label_alignment,
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


CONCEPT_JSON = {
    "result": {
        "classType": "Concept",
        "ui": "C0000731",
        "name": "Abdominal Distension",
        "semanticTypes": [{"name": "Sign or Symptom"}],
        "atoms": "https://uts-ws.nlm.nih.gov/rest/content/2024AB/CUI/C0000731/atoms",
    }
}
ATOMS_JSON = {
    "pageSize": 200, "pageCount": 1, "pageNumber": 1,
    "result": [
        {"name": "Abdominal Distension", "language": "ENG", "rootSource": "MSH"},
        {"name": "Bloating", "language": "ENG", "rootSource": "MSH"},
        {"name": "bloating", "language": "ENG", "rootSource": "SNOMEDCT_US"},  # case-dup
        {"name": "Distension abdominale", "language": "FRE", "rootSource": "MSHFRE"},  # non-English
    ],
}
DEFINITIONS_JSON = {
    "pageSize": 25, "pageCount": 1, "pageNumber": 1,
    "result": [
        {"rootSource": "MSH", "value": "Increased volume of the abdomen."},
        {"rootSource": "NCI", "value": "A finding of abdominal enlargement."},
    ],
}


# ---------------------------------------------------------------------------
# API key handling
# ---------------------------------------------------------------------------

class TestApiKey:
    def test_missing_key_raises_helpful_error(self, monkeypatch):
        monkeypatch.delenv("UMLS_API_KEY", raising=False)
        with pytest.raises(RuntimeError, match="UMLS_API_KEY"):
            get_api_key()

    def test_present_key_returned(self, monkeypatch):
        monkeypatch.setenv("UMLS_API_KEY", "secret-value-123")
        assert get_api_key() == "secret-value-123"

    def test_api_key_never_logged(self, monkeypatch, caplog):
        """Fetching a CUI must never write the API key to logs."""
        monkeypatch.setenv("UMLS_API_KEY", "super-secret-key")
        with patch("requests.get") as mock_get:
            mock_get.side_effect = [
                _mock_response(200, CONCEPT_JSON),
                _mock_response(200, ATOMS_JSON),
                _mock_response(200, DEFINITIONS_JSON),
            ]
            with caplog.at_level(logging.DEBUG):
                fetch_cui_record("C0000731", "super-secret-key")
        assert "super-secret-key" not in caplog.text

    def test_api_key_never_in_cache_or_metadata(self, monkeypatch, tmp_path):
        """The persisted cache/metadata must not contain the raw API key."""
        monkeypatch.setenv("UMLS_API_KEY", "super-secret-key")
        mapping_path = tmp_path / "mapping.csv"
        _mapping_df(1).to_csv(mapping_path, index=False)

        with patch("requests.get") as mock_get:
            mock_get.side_effect = [
                _mock_response(200, CONCEPT_JSON),
                _mock_response(200, ATOMS_JSON),
                _mock_response(200, DEFINITIONS_JSON),
            ]
            run_enrichment(
                mapping_input=mapping_path,
                metadata_parquet_output=tmp_path / "meta.parquet",
                metadata_csv_output=tmp_path / "meta.csv",
                cache_path=tmp_path / "cache.json",
                audit_output=tmp_path / "audit.json",
                coverage_report_output=tmp_path / "coverage.md",
            )

        cache_text = (tmp_path / "cache.json").read_text(encoding="utf-8")
        meta_text = (tmp_path / "meta.csv").read_text(encoding="utf-8")
        audit_text = (tmp_path / "audit.json").read_text(encoding="utf-8")
        assert "super-secret-key" not in cache_text
        assert "super-secret-key" not in meta_text
        assert "super-secret-key" not in audit_text


# ---------------------------------------------------------------------------
# CUI validation
# ---------------------------------------------------------------------------

class TestCuiValidation:
    def test_all_valid(self):
        result = validate_cuis(_mapping_df(5))
        assert result["missing_cui_count"] == 0
        assert result["malformed_cui_count"] == 0
        assert result["duplicate_cui_count"] == 0
        assert result["unique_cuis"] == 5

    def test_malformed_cui_detected(self):
        df = _mapping_df(2)
        df.loc[0, "side_effect_id"] = "NOTACUI"
        result = validate_cuis(df)
        assert result["malformed_cui_count"] == 1
        assert "NOTACUI" in result["malformed_cui_examples"]

    def test_duplicate_cui_detected(self):
        df = _mapping_df(2)
        df.loc[1, "side_effect_id"] = df.loc[0, "side_effect_id"]
        result = validate_cuis(df)
        assert result["duplicate_cui_count"] == 2


# ---------------------------------------------------------------------------
# HTTP retrieval scenarios
# ---------------------------------------------------------------------------

class TestFetchCuiRecord:
    def test_valid_cui_retrieval(self):
        with patch("requests.get") as mock_get:
            mock_get.side_effect = [
                _mock_response(200, CONCEPT_JSON),
                _mock_response(200, ATOMS_JSON),
                _mock_response(200, DEFINITIONS_JSON),
            ]
            record = fetch_cui_record("C0000731", "key")
        assert record["retrieval_status"] == STATUS_SUCCESS
        assert record["concept"]["name"] == "Abdominal Distension"
        assert len(record["atoms"]) == 4
        assert len(record["definitions"]) == 2
        assert record["umls_version_or_release"] == "2024AB"

    def test_unavailable_cui_404(self):
        with patch("requests.get") as mock_get:
            mock_get.return_value = _mock_response(404)
            record = fetch_cui_record("C9999999", "key")
        assert record["retrieval_status"] == STATUS_NOT_FOUND
        assert record["concept"] is None

    def test_retry_then_success(self):
        """Transient 503s are retried before succeeding."""
        with patch("requests.get") as mock_get:
            mock_get.side_effect = [
                _mock_response(503),
                _mock_response(200, CONCEPT_JSON),
                _mock_response(200, ATOMS_JSON),
                _mock_response(200, DEFINITIONS_JSON),
            ]
            with patch("time.sleep"):
                record = fetch_cui_record("C0000731", "key")
        assert record["retrieval_status"] == STATUS_SUCCESS

    def test_retries_exhausted_marks_request_failed(self):
        with patch("requests.get") as mock_get:
            mock_get.return_value = _mock_response(503)
            with patch("time.sleep"):
                record = fetch_cui_record("C0000731", "key")
        assert record["retrieval_status"] == STATUS_REQUEST_FAILED

    def test_network_exception_retried_then_failed(self):
        import requests as requests_module
        with patch("requests.get") as mock_get:
            mock_get.side_effect = requests_module.exceptions.ConnectionError("boom")
            with patch("time.sleep"):
                record = fetch_cui_record("C0000731", "key")
        assert record["retrieval_status"] == STATUS_REQUEST_FAILED


# ---------------------------------------------------------------------------
# Phase 4: synonym cleaning
# ---------------------------------------------------------------------------

class TestSynonymCleaning:
    def test_multiple_synonyms_deduped_and_english_only(self):
        cleaned = clean_synonyms(ATOMS_JSON["result"], "Abdominal Distension")
        # "Distension abdominale" (FRE) excluded, preferred name excluded,
        # "bloating"/"Bloating" case-insensitive duplicate collapsed to one
        assert cleaned == ["Bloating"]

    def test_empty_atoms_returns_empty_list(self):
        assert clean_synonyms(None, "X") == []
        assert clean_synonyms([], "X") == []

    def test_deterministic_ordering(self):
        atoms = [
            {"name": "Zeta", "language": "ENG"},
            {"name": "Alpha", "language": "ENG"},
            {"name": "beta", "language": "ENG"},
        ]
        assert clean_synonyms(atoms, "Preferred") == ["Alpha", "beta", "Zeta"]


# ---------------------------------------------------------------------------
# Phase 5: definitions
# ---------------------------------------------------------------------------

class TestDefinitionCleaning:
    def test_multiple_definitions_preserved(self):
        cleaned = clean_definitions(DEFINITIONS_JSON["result"])
        assert len(cleaned) == 2
        assert {"value": "Increased volume of the abdomen.", "source": "MSH"} in cleaned

    def test_no_definition_returns_empty_list(self):
        assert clean_definitions(None) == []
        assert clean_definitions([]) == []

    def test_exact_duplicate_removed(self):
        dupes = [
            {"rootSource": "MSH", "value": "Same text."},
            {"rootSource": "MSH", "value": "Same text."},
        ]
        assert len(clean_definitions(dupes)) == 1


# ---------------------------------------------------------------------------
# Semantic types
# ---------------------------------------------------------------------------

class TestSemanticTypes:
    def test_extraction(self):
        assert extract_semantic_types(CONCEPT_JSON["result"]) == ["Sign or Symptom"]

    def test_missing_concept_returns_empty(self):
        assert extract_semantic_types(None) == []


# ---------------------------------------------------------------------------
# Caching
# ---------------------------------------------------------------------------

class TestCaching:
    def test_save_and_load_roundtrip(self, tmp_path):
        cache_path = tmp_path / "cache.json"
        cache = {"C0000731": {"retrieval_status": "success"}}
        save_cache(cache, cache_path)
        loaded = load_cache(cache_path)
        assert loaded == cache

    def test_missing_cache_file_returns_empty_dict(self, tmp_path):
        assert load_cache(tmp_path / "nope.json") == {}

    def test_rerun_reuses_cache_without_new_requests(self, monkeypatch, tmp_path):
        monkeypatch.setenv("UMLS_API_KEY", "key")
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
                _mock_response(200, CONCEPT_JSON),
                _mock_response(200, ATOMS_JSON),
                _mock_response(200, DEFINITIONS_JSON),
            ]
            run_enrichment(**kwargs)
            first_call_count = mock_get.call_count

        with patch("requests.get") as mock_get2:
            run_enrichment(**kwargs)
            assert mock_get2.call_count == 0  # fully served from cache

        assert first_call_count == 3


# ---------------------------------------------------------------------------
# Label ordering / alignment (Phase 10)
# ---------------------------------------------------------------------------

class TestLabelAlignment:
    def test_matching_order_passes(self):
        mapping = _mapping_df(4)
        metadata = mapping.rename(columns={}).copy()
        metadata["label_index"] = mapping["label_index"]
        validate_label_alignment(mapping, metadata)  # no raise

    def test_shuffled_order_raises(self):
        """label_index stays 0..n-1 but the rows behind it were resorted (e.g. by CUI/name)."""
        mapping = _mapping_df(4)
        metadata = mapping.copy()
        metadata["side_effect_id"] = mapping["side_effect_id"].iloc[::-1].reset_index(drop=True)
        metadata["side_effect_name"] = mapping["side_effect_name"].iloc[::-1].reset_index(drop=True)
        with pytest.raises(ValueError, match="side_effect_id"):
            validate_label_alignment(mapping, metadata)

    def test_row_count_mismatch_raises(self):
        mapping = _mapping_df(4)
        metadata = mapping.iloc[:3]
        with pytest.raises(ValueError, match="Row count mismatch"):
            validate_label_alignment(mapping, metadata)

    def test_end_to_end_ordering_preserved(self, monkeypatch, tmp_path):
        """After a full pipeline run, metadata row i matches mapping row i exactly."""
        monkeypatch.setenv("UMLS_API_KEY", "key")
        mapping_path = tmp_path / "mapping.csv"
        mapping_df = _mapping_df(3)
        mapping_df.to_csv(mapping_path, index=False)

        def response_cycle(*args, **kwargs):
            return _mock_response(200, CONCEPT_JSON)

        with patch("requests.get") as mock_get:
            mock_get.side_effect = [
                _mock_response(200, CONCEPT_JSON), _mock_response(200, ATOMS_JSON), _mock_response(200, DEFINITIONS_JSON),
                _mock_response(200, CONCEPT_JSON), _mock_response(200, ATOMS_JSON), _mock_response(200, DEFINITIONS_JSON),
                _mock_response(200, CONCEPT_JSON), _mock_response(200, ATOMS_JSON), _mock_response(200, DEFINITIONS_JSON),
            ]
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
        assert (metadata_df["side_effect_id"].to_numpy() == mapping_df["side_effect_id"].to_numpy()).all()


# ---------------------------------------------------------------------------
# Invalid CUI row handling
# ---------------------------------------------------------------------------

class TestInvalidCuiRow:
    def test_malformed_cui_skips_network_call(self):
        row = build_metadata_row(0, "NOT-A-CUI", "some condition", None, "CUI does not match expected pattern: 'NOT-A-CUI'")
        assert row["retrieval_status"] == STATUS_INVALID_CUI
        assert row["has_umls_concept"] is False
        assert row["synonym_count"] == 0


# ---------------------------------------------------------------------------
# Deterministic output
# ---------------------------------------------------------------------------

class TestDeterministicOutput:
    def test_same_input_same_output(self, monkeypatch, tmp_path):
        monkeypatch.setenv("UMLS_API_KEY", "key")
        mapping_path = tmp_path / "mapping.csv"
        _mapping_df(1).to_csv(mapping_path, index=False)

        def _run(suffix: str):
            with patch("requests.get") as mock_get:
                mock_get.side_effect = [
                    _mock_response(200, CONCEPT_JSON),
                    _mock_response(200, ATOMS_JSON),
                    _mock_response(200, DEFINITIONS_JSON),
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
            return df.drop(columns=["retrieved_at"])  # wall-clock timestamp, not content

        out1 = _run("_a")
        out2 = _run("_b")
        pd.testing.assert_frame_equal(out1, out2)


# ---------------------------------------------------------------------------
# Coverage report
# ---------------------------------------------------------------------------

class TestCoverageReport:
    def test_coverage_counts(self):
        mapping = _mapping_df(2)
        metadata = pd.DataFrame([
            build_metadata_row(0, "C0000000", "a", {
                "retrieval_status": STATUS_SUCCESS, "concept": {"name": "A", "semanticTypes": []},
                "atoms": [], "definitions": [], "umls_version_or_release": "2024AB", "retrieved_at": "t",
            }, None),
            build_metadata_row(1, "C0000001", "b", None, "invalid"),
        ])
        cui_diag = validate_cuis(mapping)
        report = build_coverage_report(metadata, cui_diag)
        assert report["total_retained_side_effects"] == 2
        assert report["successful_umls_concept_retrievals"] == 1
        assert report["with_ge1_synonym"] == 0
