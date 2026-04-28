"""
Unit tests for entity_postprocessor module.
"""

import os
import sys
import tempfile
import pytest
import pandas as pd

# Add src to path
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, src_path)

from utils.entity_postprocessor import (
    normalize_drug_name,
    fuzzy_match_plan_name,
    normalize_entity_name,
    process_entities,
    _plan_names_cache,
)
import utils.entity_postprocessor as ep


# ── Drug name normalization ──────────────────────────────────────────────────

class TestNormalizeDrugName:

    def test_strips_dosage(self):
        assert normalize_drug_name("zolpidem 10mg tablet") == "zolpidem"

    def test_strips_parenthetical(self):
        assert normalize_drug_name("ambien (zolpidem)") == "ambien"

    def test_strips_salt_form(self):
        assert normalize_drug_name("zolpidem tartrate 12.5 mg er") == "zolpidem"

    def test_lowercases(self):
        assert normalize_drug_name("Eliquis") == "eliquis"

    def test_strips_complex_dosage(self):
        assert normalize_drug_name("metformin hydrochloride 500mg tablets") == "metformin"

    def test_strips_percentage(self):
        assert normalize_drug_name("tretinoin 0.025% cream") == "tretinoin"

    def test_returns_none_for_empty(self):
        assert normalize_drug_name("") is None
        assert normalize_drug_name(None) is None
        assert normalize_drug_name("null") is None
        assert normalize_drug_name("N/A") is None
        assert normalize_drug_name("none") is None

    def test_whitespace_cleanup(self):
        assert normalize_drug_name("  lisinopril  ") == "lisinopril"

    def test_multiple_forms(self):
        assert normalize_drug_name("amlodipine besylate 5mg oral tablet") == "amlodipine"


# ── Entity name normalization ────────────────────────────────────────────────

class TestNormalizeEntityName:

    def test_strips_whitespace(self):
        assert normalize_entity_name("  CVS Pharmacy  ") == "CVS Pharmacy"

    def test_returns_none_for_empty(self):
        assert normalize_entity_name("") is None
        assert normalize_entity_name(None) is None
        assert normalize_entity_name("null") is None
        assert normalize_entity_name("N/A") is None

    def test_preserves_original(self):
        assert normalize_entity_name("Mayo Clinic") == "Mayo Clinic"


# ── Fuzzy plan name matching ─────────────────────────────────────────────────

class TestFuzzyMatchPlanName:

    @pytest.fixture(autouse=True)
    def reset_cache(self):
        """Reset the module-level cache before each test."""
        ep._plan_names_cache = None
        yield
        ep._plan_names_cache = None

    @pytest.fixture
    def plan_csv(self, tmp_path):
        """Create a temporary plan names CSV."""
        plans = [
            "AARP Medicare Advantage Choice Plan 1",
            "AARP Medicare Advantage Choice Plan 2",
            "UnitedHealthcare Dual Complete",
            "Medica Prime Solution",
            "Wellcare Value Script",
        ]
        df = pd.DataFrame({"planid": range(len(plans)), "plan_name": plans})
        csv_path = tmp_path / "plan_names.csv"
        df.to_csv(csv_path, index=False)
        return str(csv_path)

    def test_exact_match(self, plan_csv):
        result = fuzzy_match_plan_name("AARP Medicare Advantage Choice Plan 1", plan_csv)
        assert result == "AARP Medicare Advantage Choice Plan 1"

    def test_partial_match(self, plan_csv):
        result = fuzzy_match_plan_name("AARP Medicare Advantage", plan_csv)
        assert "AARP Medicare Advantage" in result

    def test_case_insensitive_match(self, plan_csv):
        result = fuzzy_match_plan_name("unitedHealthcare dual complete", plan_csv)
        assert result == "UnitedHealthcare Dual Complete"

    def test_no_match_returns_original(self, plan_csv):
        result = fuzzy_match_plan_name("Completely Unknown Plan XYZ", plan_csv, score_threshold=95)
        assert result == "Completely Unknown Plan XYZ"

    def test_returns_none_for_empty(self, plan_csv):
        assert fuzzy_match_plan_name("", plan_csv) is None
        assert fuzzy_match_plan_name(None, plan_csv) is None
        assert fuzzy_match_plan_name("null", plan_csv) is None


# ── Process entities (integration) ───────────────────────────────────────────

class TestProcessEntities:

    @pytest.fixture(autouse=True)
    def reset_cache(self):
        ep._plan_names_cache = None
        yield
        ep._plan_names_cache = None

    @pytest.fixture
    def plan_csv(self, tmp_path):
        plans = ["AARP Medicare Advantage Choice Plan 1", "UnitedHealthcare Dual Complete"]
        df = pd.DataFrame({"planid": range(len(plans)), "plan_name": plans})
        csv_path = tmp_path / "plan_names.csv"
        df.to_csv(csv_path, index=False)
        return str(csv_path)

    def test_drug_name_entity(self, plan_csv):
        entities = {"drug_names": "zolpidem 10mg tablet"}
        result = process_entities(entities, plan_csv)
        assert result == ["zolpidem"]

    def test_plan_name_entity(self, plan_csv):
        entities = {"plan_names": "AARP Medicare Advantage Choice Plan 1"}
        result = process_entities(entities, plan_csv)
        assert result == ["AARP Medicare Advantage Choice Plan 1"]

    def test_pharmacy_name_entity(self, plan_csv):
        entities = {"pharmacy_names": "CVS Pharmacy"}
        result = process_entities(entities, plan_csv)
        assert result == ["CVS Pharmacy"]

    def test_hospital_facility_entity(self, plan_csv):
        entities = {"hospital/facility_names": "Mayo Clinic"}
        result = process_entities(entities, plan_csv)
        assert result == ["Mayo Clinic"]

    def test_dental_plan_name_entity(self, plan_csv):
        entities = {"dental_plan_name": "Delta Dental PPO"}
        result = process_entities(entities, plan_csv)
        assert result == ["Delta Dental PPO"]

    def test_multiple_entities(self, plan_csv):
        entities = {
            "drug_names": "metformin 500mg",
            "pharmacy_names": "Walgreens"
        }
        result = process_entities(entities, plan_csv)
        assert "metformin" in result
        assert "Walgreens" in result
        assert len(result) == 2

    def test_skips_null_values(self, plan_csv):
        entities = {"drug_names": "null", "pharmacy_names": "N/A"}
        result = process_entities(entities, plan_csv)
        assert result == []

    def test_empty_dict(self, plan_csv):
        assert process_entities({}, plan_csv) == []

    def test_none_input(self, plan_csv):
        assert process_entities(None, plan_csv) == []

    def test_unknown_entity_type_still_normalized(self, plan_csv):
        entities = {"some_new_type": "  Some Value  "}
        result = process_entities(entities, plan_csv)
        assert result == ["Some Value"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
