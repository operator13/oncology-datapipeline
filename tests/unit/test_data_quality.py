"""
Unit tests for data quality module.
"""

import pytest
import pandas as pd

from src.data_quality import (
    ExpectationBuilder,
    ValidationRunner,
    ValidationResult,
    build_patient_suite,
)


class TestExpectationBuilder:
    """Tests for ExpectationBuilder."""

    @pytest.mark.unit
    def test_build_empty_suite(self):
        """Test building an empty suite."""
        builder = ExpectationBuilder("test_suite")
        suite = builder.to_json()

        assert suite["expectation_suite_name"] == "test_suite"
        assert len(suite["expectations"]) == 0

    @pytest.mark.unit
    def test_expect_not_null(self):
        """Test adding not null expectation."""
        builder = ExpectationBuilder("test_suite")
        builder.expect_not_null("patient_id")
        suite = builder.to_json()

        assert len(suite["expectations"]) == 1
        exp = suite["expectations"][0]
        assert exp["expectation_type"] == "expect_column_values_to_not_be_null"
        assert exp["kwargs"]["column"] == "patient_id"

    @pytest.mark.unit
    def test_expect_unique(self):
        """Test adding uniqueness expectation."""
        builder = ExpectationBuilder("test_suite")
        builder.expect_unique("mrn")
        suite = builder.to_json()

        exp = suite["expectations"][0]
        assert exp["expectation_type"] == "expect_column_values_to_be_unique"

    @pytest.mark.unit
    def test_expect_values_in_set(self):
        """Test adding values in set expectation."""
        builder = ExpectationBuilder("test_suite")
        builder.expect_values_in_set("gender", ["M", "F", "O", "U"])
        suite = builder.to_json()

        exp = suite["expectations"][0]
        assert exp["expectation_type"] == "expect_column_values_to_be_in_set"
        assert "M" in exp["kwargs"]["value_set"]

    @pytest.mark.unit
    def test_expect_regex(self):
        """Test adding regex expectation."""
        builder = ExpectationBuilder("test_suite")
        builder.expect_regex("mrn", r"^MRN\d{8}$")
        suite = builder.to_json()

        exp = suite["expectations"][0]
        assert exp["expectation_type"] == "expect_column_values_to_match_regex"
        assert exp["kwargs"]["regex"] == r"^MRN\d{8}$"

    @pytest.mark.unit
    def test_expect_between(self):
        """Test adding between expectation."""
        builder = ExpectationBuilder("test_suite")
        builder.expect_between("age", min_value=0, max_value=120)
        suite = builder.to_json()

        exp = suite["expectations"][0]
        assert exp["expectation_type"] == "expect_column_values_to_be_between"
        assert exp["kwargs"]["min_value"] == 0
        assert exp["kwargs"]["max_value"] == 120

    @pytest.mark.unit
    def test_chaining(self):
        """Test fluent interface chaining."""
        builder = ExpectationBuilder("test_suite")
        result = (
            builder
            .expect_not_null("id")
            .expect_unique("id")
            .expect_between("value", min_value=0)
        )

        assert result is builder
        suite = builder.to_json()
        assert len(suite["expectations"]) == 3

    @pytest.mark.unit
    def test_icd10_oncology_code(self):
        """Test ICD-10 oncology code expectation."""
        builder = ExpectationBuilder("test_suite")
        builder.expect_icd10_oncology_code("diagnosis_code")
        suite = builder.to_json()

        exp = suite["expectations"][0]
        assert "C\\d{2}" in exp["kwargs"]["regex"]

    @pytest.mark.unit
    def test_ndc_code(self):
        """Test NDC code expectation."""
        builder = ExpectationBuilder("test_suite")
        builder.expect_ndc_code("drug_code")
        suite = builder.to_json()

        exp = suite["expectations"][0]
        assert "\\d{5}-\\d{4}-\\d{2}" in exp["kwargs"]["regex"]


class TestPrebuiltSuites:
    """Tests for pre-built expectation suites."""

    @pytest.mark.unit
    def test_patient_suite_structure(self):
        """Test that patient suite has expected structure."""
        builder = build_patient_suite()
        suite = builder.to_json()

        assert suite["expectation_suite_name"] == "oncology_patients_suite"
        assert len(suite["expectations"]) > 10  # Should have many expectations

        # Check for key expectations
        exp_types = [e["expectation_type"] for e in suite["expectations"]]
        assert "expect_column_values_to_not_be_null" in exp_types
        assert "expect_column_values_to_be_unique" in exp_types


class TestValidationResult:
    """Tests for ValidationResult."""

    @pytest.mark.unit
    def test_success_percent(self):
        """Test success percentage calculation."""
        result = ValidationResult(
            success=True,
            suite_name="test",
            run_time=None,
            statistics={
                "evaluated_expectations": 10,
                "successful_expectations": 8,
            },
            results=[],
        )

        assert result.success_percent == 80.0

    @pytest.mark.unit
    def test_failed_expectations_filter(self):
        """Test filtering failed expectations."""
        result = ValidationResult(
            success=False,
            suite_name="test",
            run_time=None,
            statistics={},
            results=[
                {"success": True, "expectation_type": "test1"},
                {"success": False, "expectation_type": "test2"},
                {"success": True, "expectation_type": "test3"},
                {"success": False, "expectation_type": "test4"},
            ],
        )

        failed = result.failed_expectations
        assert len(failed) == 2
        assert failed[0]["expectation_type"] == "test2"
