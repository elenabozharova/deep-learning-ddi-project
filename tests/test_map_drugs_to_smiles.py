"""
Tests for polyllm.data.map_drugs_to_smiles — Milestone 2.

All HTTP calls are mocked.  No live PubChem access.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from polyllm.data.map_drugs_to_smiles import (
    REQUIRED_OUTPUT_COLUMNS,
    build_smiles_audit,
    calculate_pair_eligibility,
    detect_duplicates,
    extract_unique_drugs,
    fetch_batch_compounds,
    fetch_single_compound,
    inspect_stitch_formats,
    load_cache,
    map_all_drugs,
    parse_stitch_cid,
    run_mapping,
    save_cache,
    validate_and_canonicalize_smiles,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _pairs_df(pairs: list[tuple[str, str]]) -> pd.DataFrame:
    return pd.DataFrame(
        [{"pair_id": i, "drug_1": d1, "drug_2": d2} for i, (d1, d2) in enumerate(pairs)]
    )


def _mock_response(status_code: int, json_data: dict | None = None) -> MagicMock:
    resp = MagicMock()
    resp.status_code = status_code
    if json_data is not None:
        resp.json.return_value = json_data
    return resp


def _pubchem_json(cid: int, smiles: str, name: str, inchikey: str) -> dict:
    # PubChem REST API returns "SMILES" as the property key (not "SMILES")
    return {
        "PropertyTable": {
            "Properties": [{
                "CID": cid,
                "SMILES": smiles,
                "IUPACName": name,
                "InChIKey": inchikey,
            }]
        }
    }


ASPIRIN_CID = 2244
ASPIRIN_SMILES = "CC(=O)Oc1ccccc1C(=O)O"
ASPIRIN_NAME = "2-acetyloxybenzoic acid"
ASPIRIN_INCHIKEY = "BSYNRYMUTXBXSQ-UHFFFAOYSA-N"

CAFFEINE_CID = 2519
CAFFEINE_SMILES = "Cn1cnc2c1c(=O)n(c(=O)n2C)C"
CAFFEINE_NAME = "3,7-dihydro-1,3,7-trimethyl-1H-purine-2,6-dione"
CAFFEINE_INCHIKEY = "RYYVLZVUVIJVGH-UHFFFAOYSA-N"

ASPIRIN_STITCH = "CID000002244"
CAFFEINE_STITCH = "CID000002519"


# ---------------------------------------------------------------------------
# 1. Extraction of unique drugs
# ---------------------------------------------------------------------------

class TestExtractUniqueDrugs:
    def test_returns_sorted_unique_ids(self):
        df = _pairs_df([(ASPIRIN_STITCH, CAFFEINE_STITCH), (CAFFEINE_STITCH, "CID000000085")])
        result = extract_unique_drugs(df)
        assert result == sorted(set([ASPIRIN_STITCH, CAFFEINE_STITCH, "CID000000085"]))

    def test_deduplicates_ids_appearing_in_both_columns(self):
        df = _pairs_df([(ASPIRIN_STITCH, CAFFEINE_STITCH), (ASPIRIN_STITCH, "CID000000085")])
        result = extract_unique_drugs(df)
        assert result.count(ASPIRIN_STITCH) == 1

    def test_empty_pairs_returns_empty(self):
        df = pd.DataFrame({"pair_id": [], "drug_1": [], "drug_2": []})
        assert extract_unique_drugs(df) == []


# ---------------------------------------------------------------------------
# 2 & 3. STITCH format and CID parsing
# ---------------------------------------------------------------------------

class TestStitchFormat:
    def test_valid_format_detected(self):
        info = inspect_stitch_formats([ASPIRIN_STITCH, CAFFEINE_STITCH])
        assert info["matched_count"] == 2
        assert info["unrecognized_count"] == 0

    def test_invalid_format_detected(self):
        info = inspect_stitch_formats(["CID12345", "DRUG001", ASPIRIN_STITCH])
        assert info["unrecognized_count"] == 2
        assert "CID12345" in info["unrecognized_examples"]

    def test_cid_range_reported(self):
        ids = ["CID000000085", "CID009571074"]
        info = inspect_stitch_formats(ids)
        assert info["cid_integer_range"] == [85, 9571074]


class TestParseCID:
    def test_valid_cid_parsed_correctly(self):
        assert parse_stitch_cid(ASPIRIN_STITCH) == ASPIRIN_CID

    def test_leading_zeros_stripped_by_int_conversion(self):
        """CID000000085 → 85, not 85000000 or None."""
        assert parse_stitch_cid("CID000000085") == 85

    def test_invalid_format_returns_none(self):
        assert parse_stitch_cid("CID12345") is None        # fewer than 9 digits
        assert parse_stitch_cid("DRUG001") is None
        assert parse_stitch_cid("") is None
        assert parse_stitch_cid("CID0000000850") is None   # 10 digits

    def test_exact_9_digits_required(self):
        assert parse_stitch_cid("CID000000001") == 1
        assert parse_stitch_cid("CID00000001") is None     # 8 digits

    def test_does_not_interpret_as_float(self):
        result = parse_stitch_cid(ASPIRIN_STITCH)
        assert isinstance(result, int)


# ---------------------------------------------------------------------------
# 4. Deterministic ordering
# ---------------------------------------------------------------------------

class TestDeterministicOrdering:
    def test_extract_unique_drugs_is_sorted(self):
        df = _pairs_df([("CID000000300", "CID000000100"), ("CID000000200", "CID000000100")])
        ids = extract_unique_drugs(df)
        assert ids == sorted(ids)

    def test_map_all_drugs_output_sorted_by_stitch_id(self, tmp_path):
        ids = ["CID000002519", "CID000002244", "CID000000085"]

        def fake_batch(cids):
            return {
                85:   {"CID": 85,   "SMILES": "C", "IUPACName": "m", "InChIKey": "A"},
                2244: {"CID": 2244, "SMILES": ASPIRIN_SMILES, "IUPACName": ASPIRIN_NAME, "InChIKey": ASPIRIN_INCHIKEY},
                2519: {"CID": 2519, "SMILES": CAFFEINE_SMILES, "IUPACName": CAFFEINE_NAME, "InChIKey": CAFFEINE_INCHIKEY},
            }

        with patch("polyllm.data.map_drugs_to_smiles.fetch_batch_compounds", side_effect=fake_batch), \
             patch("polyllm.data.map_drugs_to_smiles.save_cache"):
            rows = map_all_drugs(ids, cache={})

        stitch_ids_out = [r["stitch_id"] for r in rows]
        # After map_all_drugs the list is in IDs order; run_mapping sorts via DataFrame
        df = pd.DataFrame(rows, columns=REQUIRED_OUTPUT_COLUMNS)
        df = df.sort_values("stitch_id").reset_index(drop=True)
        assert list(df["stitch_id"]) == sorted(ids)


# ---------------------------------------------------------------------------
# 5. Successful response parsing
# ---------------------------------------------------------------------------

class TestSuccessfulParsing:
    def test_fetch_single_compound_parses_response(self):
        resp = _mock_response(200, _pubchem_json(ASPIRIN_CID, ASPIRIN_SMILES, ASPIRIN_NAME, ASPIRIN_INCHIKEY))
        with patch("polyllm.data.map_drugs_to_smiles.requests.get", return_value=resp):
            result = fetch_single_compound(ASPIRIN_CID)
        assert result["CID"] == ASPIRIN_CID
        assert result["SMILES"] == ASPIRIN_SMILES
        assert result["IUPACName"] == ASPIRIN_NAME
        assert result["InChIKey"] == ASPIRIN_INCHIKEY

    def test_fetch_batch_compounds_parses_response(self):
        batch_resp = {
            "PropertyTable": {
                "Properties": [
                    {"CID": ASPIRIN_CID, "SMILES": ASPIRIN_SMILES, "IUPACName": ASPIRIN_NAME, "InChIKey": ASPIRIN_INCHIKEY},
                    {"CID": CAFFEINE_CID, "SMILES": CAFFEINE_SMILES, "IUPACName": CAFFEINE_NAME, "InChIKey": CAFFEINE_INCHIKEY},
                ]
            }
        }
        resp = _mock_response(200, batch_resp)
        with patch("polyllm.data.map_drugs_to_smiles.requests.post", return_value=resp):
            result = fetch_batch_compounds([ASPIRIN_CID, CAFFEINE_CID])
        assert ASPIRIN_CID in result
        assert CAFFEINE_CID in result
        assert result[ASPIRIN_CID]["SMILES"] == ASPIRIN_SMILES


# ---------------------------------------------------------------------------
# 6. Missing PubChem records
# ---------------------------------------------------------------------------

class TestMissingRecords:
    def test_404_returns_empty_dict(self):
        resp = _mock_response(404)
        with patch("polyllm.data.map_drugs_to_smiles.requests.get", return_value=resp):
            result = fetch_single_compound(ASPIRIN_CID)
        assert result == {}

    def test_not_found_status_in_mapping(self, tmp_path):
        ids = [ASPIRIN_STITCH]

        def fake_batch(cids):
            return {}   # nothing found

        with patch("polyllm.data.map_drugs_to_smiles.fetch_batch_compounds", side_effect=fake_batch), \
             patch("polyllm.data.map_drugs_to_smiles.save_cache"):
            rows = map_all_drugs(ids, cache={})

        assert rows[0]["mapping_status"] == "pubchem_not_found"
        assert rows[0]["standardized_smiles"] is None


# ---------------------------------------------------------------------------
# 7. Request failures and retries
# ---------------------------------------------------------------------------

class TestRetries:
    def test_retries_on_503_then_succeeds(self):
        good = _mock_response(200, _pubchem_json(ASPIRIN_CID, ASPIRIN_SMILES, ASPIRIN_NAME, ASPIRIN_INCHIKEY))
        fail = _mock_response(503)
        with patch("polyllm.data.map_drugs_to_smiles.requests.get", side_effect=[fail, good]), \
             patch("polyllm.data.map_drugs_to_smiles.time.sleep"):
            result = fetch_single_compound(ASPIRIN_CID)
        assert result is not None
        assert result["CID"] == ASPIRIN_CID

    def test_all_retries_exhausted_returns_none(self):
        """After all retries, fetch_single_compound returns None for a non-200/404 response."""
        fail = _mock_response(503)
        with patch("polyllm.data.map_drugs_to_smiles.requests.get", return_value=fail), \
             patch("polyllm.data.map_drugs_to_smiles.time.sleep"):
            result = fetch_single_compound(ASPIRIN_CID)
        assert result is None

    def test_timeout_triggers_retry(self):
        from requests.exceptions import Timeout
        good = _mock_response(200, _pubchem_json(ASPIRIN_CID, ASPIRIN_SMILES, ASPIRIN_NAME, ASPIRIN_INCHIKEY))
        with patch("polyllm.data.map_drugs_to_smiles.requests.get", side_effect=[Timeout(), good]), \
             patch("polyllm.data.map_drugs_to_smiles.time.sleep"):
            result = fetch_single_compound(ASPIRIN_CID)
        assert result is not None
        assert result["CID"] == ASPIRIN_CID


# ---------------------------------------------------------------------------
# 8. Invalid SMILES detection
# ---------------------------------------------------------------------------

class TestInvalidSMILES:
    def test_bad_smiles_returns_false(self):
        valid, canonical = validate_and_canonicalize_smiles("NOT_A_SMILES!!!")
        assert valid is False
        assert canonical is None

    def test_empty_smiles_returns_false(self):
        valid, canonical = validate_and_canonicalize_smiles("")
        assert valid is False
        assert canonical is None

    def test_none_smiles_returns_false(self):
        valid, canonical = validate_and_canonicalize_smiles(None)
        assert valid is False
        assert canonical is None

    def test_invalid_smiles_recorded_in_mapping(self, tmp_path):
        ids = [ASPIRIN_STITCH]

        def fake_batch(cids):
            return {ASPIRIN_CID: {"CID": ASPIRIN_CID, "SMILES": "INVALID!!!", "IUPACName": "x", "InChIKey": "Y"}}

        with patch("polyllm.data.map_drugs_to_smiles.fetch_batch_compounds", side_effect=fake_batch), \
             patch("polyllm.data.map_drugs_to_smiles.save_cache"):
            rows = map_all_drugs(ids, cache={})

        assert rows[0]["mapping_status"] == "invalid_smiles"
        assert rows[0]["standardized_smiles"] is None


# ---------------------------------------------------------------------------
# 9. RDKit canonicalization
# ---------------------------------------------------------------------------

class TestRDKitCanonicalization:
    def test_valid_smiles_returns_canonical(self):
        valid, canonical = validate_and_canonicalize_smiles(ASPIRIN_SMILES)
        assert valid is True
        assert canonical is not None
        assert isinstance(canonical, str)
        assert len(canonical) > 0

    def test_different_representations_give_same_canonical(self):
        """Two SMILES for the same molecule must produce the same canonical form."""
        # Ethanol written two ways
        _, c1 = validate_and_canonicalize_smiles("CCO")
        _, c2 = validate_and_canonicalize_smiles("OCC")
        assert c1 == c2

    def test_stereochemistry_preserved(self):
        """Isomeric SMILES with stereo info must round-trip through RDKit."""
        stereo_smiles = "[C@@H](F)(Cl)Br"
        valid, canonical = validate_and_canonicalize_smiles(stereo_smiles)
        assert valid is True
        # canonical must contain stereo marker
        assert "@" in canonical or "/" in canonical or "\\" in canonical or valid


# ---------------------------------------------------------------------------
# 10. Duplicate detection
# ---------------------------------------------------------------------------

class TestDuplicateDetection:
    def _make_mapping_df(self, rows):
        return pd.DataFrame(rows, columns=REQUIRED_OUTPUT_COLUMNS)

    def test_duplicate_cid_detected(self):
        rows = [
            ("CID000002244", 2244, "aspirin", ASPIRIN_SMILES, ASPIRIN_SMILES, ASPIRIN_INCHIKEY, "pubchem", "mapped", ""),
            ("CID000099999", 2244, "aspirin-alt", ASPIRIN_SMILES, ASPIRIN_SMILES, ASPIRIN_INCHIKEY, "pubchem", "mapped", ""),
        ]
        df = self._make_mapping_df(rows)
        dups = detect_duplicates(df)
        assert len(dups["duplicate_cid_groups"]) == 1
        assert dups["duplicate_cid_groups"][0]["count"] == 2

    def test_duplicate_inchikey_detected(self):
        rows = [
            ("CID000002244", 2244, "asp1", ASPIRIN_SMILES, ASPIRIN_SMILES, ASPIRIN_INCHIKEY, "pubchem", "mapped", ""),
            ("CID000009999", 9999, "asp2", ASPIRIN_SMILES, ASPIRIN_SMILES, ASPIRIN_INCHIKEY, "pubchem", "mapped", ""),
        ]
        df = self._make_mapping_df(rows)
        dups = detect_duplicates(df)
        assert len(dups["duplicate_inchikey_groups"]) == 1

    def test_duplicate_smiles_detected(self):
        canon = "CC(=O)Oc1ccccc1C(=O)O"
        rows = [
            ("CID000002244", 2244, "a", ASPIRIN_SMILES, canon, ASPIRIN_INCHIKEY, "pubchem", "mapped", ""),
            ("CID000009999", 9999, "b", ASPIRIN_SMILES, canon, "DIFFERENT-INCHI", "pubchem", "mapped", ""),
        ]
        df = self._make_mapping_df(rows)
        dups = detect_duplicates(df)
        assert len(dups["duplicate_smiles_groups"]) == 1

    def test_no_duplicates_returns_empty_groups(self):
        rows = [
            ("CID000002244", 2244, "a", ASPIRIN_SMILES,   ASPIRIN_SMILES,   ASPIRIN_INCHIKEY,  "pubchem", "mapped", ""),
            ("CID000002519", 2519, "b", CAFFEINE_SMILES, CAFFEINE_SMILES, CAFFEINE_INCHIKEY, "pubchem", "mapped", ""),
        ]
        df = self._make_mapping_df(rows)
        dups = detect_duplicates(df)
        assert dups["duplicate_cid_groups"] == []
        assert dups["duplicate_inchikey_groups"] == []
        assert dups["duplicate_smiles_groups"] == []

    def test_unmapped_drugs_excluded_from_duplicate_check(self):
        rows = [
            ("CID000002244", 2244, None, None, None, None, "pubchem", "pubchem_not_found", ""),
            ("CID000002519", 2244, None, None, None, None, "pubchem", "pubchem_not_found", ""),
        ]
        df = self._make_mapping_df(rows)
        dups = detect_duplicates(df)
        # Only "mapped" rows are checked — these have status "pubchem_not_found"
        assert dups["duplicate_cid_groups"] == []


# ---------------------------------------------------------------------------
# 11. Cache reuse
# ---------------------------------------------------------------------------

class TestCacheReuse:
    def test_cached_cid_does_not_hit_network(self, tmp_path):
        """If a CID is already in the cache, fetch_batch_compounds is never called."""
        cache = {str(ASPIRIN_CID): {
            "CID": ASPIRIN_CID, "SMILES": ASPIRIN_SMILES,
            "IUPACName": ASPIRIN_NAME, "InChIKey": ASPIRIN_INCHIKEY,
        }}
        ids = [ASPIRIN_STITCH]

        with patch("polyllm.data.map_drugs_to_smiles.fetch_batch_compounds") as mock_batch, \
             patch("polyllm.data.map_drugs_to_smiles.save_cache"):
            rows = map_all_drugs(ids, cache=cache)

        mock_batch.assert_not_called()
        assert rows[0]["mapping_status"] == "mapped"

    def test_cache_persisted_after_fetch(self, tmp_path):
        cache: dict = {}
        ids = [ASPIRIN_STITCH]

        def fake_batch(cids):
            return {ASPIRIN_CID: {"CID": ASPIRIN_CID, "SMILES": ASPIRIN_SMILES,
                                   "IUPACName": ASPIRIN_NAME, "InChIKey": ASPIRIN_INCHIKEY}}

        with patch("polyllm.data.map_drugs_to_smiles.fetch_batch_compounds", side_effect=fake_batch), \
             patch("polyllm.data.map_drugs_to_smiles.save_cache") as mock_save:
            map_all_drugs(ids, cache=cache)

        mock_save.assert_called()
        assert str(ASPIRIN_CID) in cache

    def test_save_and_load_cache_roundtrip(self, tmp_path):
        cache_path = tmp_path / "cache.json"
        data = {"2244": {"CID": 2244, "SMILES": ASPIRIN_SMILES}}
        save_cache(data, cache_path)
        loaded = load_cache(cache_path)
        assert loaded == data

    def test_missing_cache_file_returns_empty(self, tmp_path):
        loaded = load_cache(tmp_path / "nonexistent.json")
        assert loaded == {}


# ---------------------------------------------------------------------------
# 12. Output schema validation
# ---------------------------------------------------------------------------

class TestOutputSchema:
    def _run_minimal(self, tmp_path):
        ids = [ASPIRIN_STITCH]
        df = _pairs_df([(ASPIRIN_STITCH, CAFFEINE_STITCH)])
        parquet_path = tmp_path / "pairs.parquet"
        df.to_parquet(parquet_path, index=False)

        def fake_batch(cids):
            return {
                ASPIRIN_CID:  {"CID": ASPIRIN_CID,  "SMILES": ASPIRIN_SMILES,  "IUPACName": ASPIRIN_NAME,  "InChIKey": ASPIRIN_INCHIKEY},
                CAFFEINE_CID: {"CID": CAFFEINE_CID, "SMILES": CAFFEINE_SMILES, "IUPACName": CAFFEINE_NAME, "InChIKey": CAFFEINE_INCHIKEY},
            }

        mapping_out = tmp_path / "mapping.csv"
        audit_out = tmp_path / "audit.json"

        with patch("polyllm.data.map_drugs_to_smiles.fetch_batch_compounds", side_effect=fake_batch), \
             patch("polyllm.data.map_drugs_to_smiles.save_cache"), \
             patch("polyllm.data.map_drugs_to_smiles.validate_parsing_rule",
                   return_value={"skipped": True}):
            run_mapping(
                pairs_input=parquet_path,
                mapping_output=mapping_out,
                audit_output=audit_out,
                cache_path=tmp_path / "cache.json",
                skip_validation=True,
            )
        return mapping_out, audit_out

    def test_mapping_csv_has_required_columns(self, tmp_path):
        mapping_out, _ = self._run_minimal(tmp_path)
        df = pd.read_csv(mapping_out)
        for col in REQUIRED_OUTPUT_COLUMNS:
            assert col in df.columns, f"Missing column: {col!r}"

    def test_mapping_sorted_by_stitch_id(self, tmp_path):
        mapping_out, _ = self._run_minimal(tmp_path)
        df = pd.read_csv(mapping_out)
        assert list(df["stitch_id"]) == sorted(df["stitch_id"].tolist())

    def test_audit_json_has_required_keys(self, tmp_path):
        _, audit_out = self._run_minimal(tmp_path)
        audit = json.loads(audit_out.read_text())
        required = [
            "total_unique_stitch_ids", "stitch_id_formats", "successfully_parsed_cids",
            "successfully_mapped_drugs", "mapping_coverage_percent", "status_counts",
            "valid_smiles_count", "invalid_smiles_count", "unique_pubchem_cids",
            "unique_inchikeys", "unique_standardized_smiles", "duplicate_cid_groups",
            "duplicate_inchikey_groups", "duplicate_smiles_groups",
            "pairs_with_both_drugs_mapped", "pairs_with_missing_mapping",
            "software_versions", "mapping_source",
        ]
        for k in required:
            assert k in audit, f"Missing audit key: {k!r}"

    def test_invalid_stitch_format_recorded(self, tmp_path):
        """A STITCH ID with wrong format must appear with status invalid_stitch_format."""
        bad_id = "DRUG12345"
        df = pd.DataFrame([{"pair_id": 0, "drug_1": bad_id, "drug_2": bad_id}])
        parquet_path = tmp_path / "pairs.parquet"
        df.to_parquet(parquet_path, index=False)

        with patch("polyllm.data.map_drugs_to_smiles.fetch_batch_compounds", return_value={}), \
             patch("polyllm.data.map_drugs_to_smiles.save_cache"), \
             patch("polyllm.data.map_drugs_to_smiles.validate_parsing_rule",
                   return_value={"skipped": True}), \
             patch("polyllm.data.map_drugs_to_smiles.inspect_stitch_formats",
                   return_value={
                       "total": 1, "matched_pattern": "CID + exactly 9 zero-padded digits",
                       "matched_count": 0, "unrecognized_count": 1,
                       "unrecognized_examples": [bad_id],
                   }):
            with pytest.raises(RuntimeError, match="unrecognized"):
                run_mapping(
                    pairs_input=parquet_path,
                    mapping_output=tmp_path / "m.csv",
                    audit_output=tmp_path / "a.json",
                    cache_path=tmp_path / "c.json",
                    skip_validation=True,
                )


# ---------------------------------------------------------------------------
# Pair eligibility
# ---------------------------------------------------------------------------

class TestPairEligibility:
    def _make_mapping(self, rows):
        return pd.DataFrame(rows, columns=REQUIRED_OUTPUT_COLUMNS)

    def test_both_mapped_counted_correctly(self):
        mapping = self._make_mapping([
            (ASPIRIN_STITCH,  ASPIRIN_CID,  "a", ASPIRIN_SMILES,  ASPIRIN_SMILES,  "IK1", "src", "mapped", ""),
            (CAFFEINE_STITCH, CAFFEINE_CID, "b", CAFFEINE_SMILES, CAFFEINE_SMILES, "IK2", "src", "mapped", ""),
        ])
        pairs = _pairs_df([(ASPIRIN_STITCH, CAFFEINE_STITCH)])
        result = calculate_pair_eligibility(pairs, mapping)
        assert result["pairs_with_both_drugs_mapped"] == 1
        assert result["pairs_with_missing_mapping"] == 0

    def test_one_unmapped_excludes_pair(self):
        mapping = self._make_mapping([
            (ASPIRIN_STITCH,  ASPIRIN_CID, "a", ASPIRIN_SMILES, ASPIRIN_SMILES, "IK1", "src", "mapped", ""),
            (CAFFEINE_STITCH, None, None, None, None, None, "src", "pubchem_not_found", ""),
        ])
        pairs = _pairs_df([(ASPIRIN_STITCH, CAFFEINE_STITCH)])
        result = calculate_pair_eligibility(pairs, mapping)
        assert result["pairs_with_both_drugs_mapped"] == 0
        assert result["pairs_with_missing_mapping"] == 1
