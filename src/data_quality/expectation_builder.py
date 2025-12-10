"""
Programmatic expectation suite builder for oncology data.

This module provides utilities for creating Great Expectations
expectation suites programmatically, enabling dynamic and
reusable data quality rules.
"""

from typing import Any

import pandas as pd
import structlog

try:
    import great_expectations as gx
    from great_expectations.core import ExpectationSuite
    from great_expectations.core.expectation_configuration import ExpectationConfiguration
    from great_expectations.expectations.expectation import Expectation

    GE_AVAILABLE = True
except ImportError:
    GE_AVAILABLE = False
    ExpectationSuite = Any
    ExpectationConfiguration = Any

from src.synthetic_data.schemas.oncology_schemas import (
    ICD10_ONCOLOGY_CODES,
    CancerType,
    Gender,
    LabTestCategory,
    TreatmentStatus,
    TreatmentType,
)

logger = structlog.get_logger(__name__)


class ExpectationBuilder:
    """Builder for creating Great Expectations suites programmatically.

    This class provides a fluent interface for building expectation
    suites with common oncology data quality rules.

    Attributes:
        suite_name: Name of the expectation suite.
        expectations: List of expectation configurations.

    Example:
        >>> builder = ExpectationBuilder("patient_suite")
        >>> builder.expect_column_to_exist("patient_id")
        >>> builder.expect_unique("patient_id")
        >>> suite = builder.build()
    """

    def __init__(self, suite_name: str) -> None:
        """Initialize the expectation builder.

        Args:
            suite_name: Name for the expectation suite.
        """
        if not GE_AVAILABLE:
            logger.warning("Great Expectations not installed, builder in mock mode")

        self.suite_name = suite_name
        self.expectations: list[dict[str, Any]] = []
        self._logger = logger.bind(suite_name=suite_name)

    def expect_column_to_exist(self, column: str) -> "ExpectationBuilder":
        """Add expectation that column exists.

        Args:
            column: Column name.

        Returns:
            Self for chaining.
        """
        self._add_expectation("expect_column_to_exist", {"column": column})
        return self

    def expect_columns_to_match_set(
        self,
        columns: list[str],
        exact_match: bool = False,
    ) -> "ExpectationBuilder":
        """Add expectation for column set validation.

        Args:
            columns: Expected columns.
            exact_match: Require exact match.

        Returns:
            Self for chaining.
        """
        self._add_expectation(
            "expect_table_columns_to_match_set",
            {"column_set": columns, "exact_match": exact_match},
        )
        return self

    def expect_not_null(self, column: str) -> "ExpectationBuilder":
        """Add expectation that column has no null values.

        Args:
            column: Column name.

        Returns:
            Self for chaining.
        """
        self._add_expectation(
            "expect_column_values_to_not_be_null",
            {"column": column},
        )
        return self

    def expect_unique(self, column: str) -> "ExpectationBuilder":
        """Add expectation that column values are unique.

        Args:
            column: Column name.

        Returns:
            Self for chaining.
        """
        self._add_expectation(
            "expect_column_values_to_be_unique",
            {"column": column},
        )
        return self

    def expect_values_in_set(
        self,
        column: str,
        value_set: list[Any],
        mostly: float = 1.0,
    ) -> "ExpectationBuilder":
        """Add expectation that values are from a specific set.

        Args:
            column: Column name.
            value_set: Allowed values.
            mostly: Minimum proportion (0.0-1.0).

        Returns:
            Self for chaining.
        """
        kwargs: dict[str, Any] = {"column": column, "value_set": value_set}
        if mostly < 1.0:
            kwargs["mostly"] = mostly
        self._add_expectation("expect_column_values_to_be_in_set", kwargs)
        return self

    def expect_regex(
        self,
        column: str,
        regex: str,
        mostly: float = 1.0,
    ) -> "ExpectationBuilder":
        """Add expectation that values match a regex pattern.

        Args:
            column: Column name.
            regex: Regular expression pattern.
            mostly: Minimum proportion (0.0-1.0).

        Returns:
            Self for chaining.
        """
        kwargs: dict[str, Any] = {"column": column, "regex": regex}
        if mostly < 1.0:
            kwargs["mostly"] = mostly
        self._add_expectation("expect_column_values_to_match_regex", kwargs)
        return self

    def expect_between(
        self,
        column: str,
        min_value: float | None = None,
        max_value: float | None = None,
        mostly: float = 1.0,
    ) -> "ExpectationBuilder":
        """Add expectation that values are within a range.

        Args:
            column: Column name.
            min_value: Minimum allowed value.
            max_value: Maximum allowed value.
            mostly: Minimum proportion (0.0-1.0).

        Returns:
            Self for chaining.
        """
        kwargs: dict[str, Any] = {"column": column}
        if min_value is not None:
            kwargs["min_value"] = min_value
        if max_value is not None:
            kwargs["max_value"] = max_value
        if mostly < 1.0:
            kwargs["mostly"] = mostly
        self._add_expectation("expect_column_values_to_be_between", kwargs)
        return self

    def expect_row_count_between(
        self,
        min_value: int | None = None,
        max_value: int | None = None,
    ) -> "ExpectationBuilder":
        """Add expectation for table row count.

        Args:
            min_value: Minimum rows.
            max_value: Maximum rows.

        Returns:
            Self for chaining.
        """
        kwargs: dict[str, Any] = {}
        if min_value is not None:
            kwargs["min_value"] = min_value
        if max_value is not None:
            kwargs["max_value"] = max_value
        self._add_expectation("expect_table_row_count_to_be_between", kwargs)
        return self

    def expect_column_pair_a_greater_than_b(
        self,
        column_a: str,
        column_b: str,
        or_equal: bool = False,
        parse_as_dates: bool = False,
    ) -> "ExpectationBuilder":
        """Add expectation that column A > column B.

        Args:
            column_a: First column.
            column_b: Second column.
            or_equal: Allow equal values.
            parse_as_dates: Parse strings as dates.

        Returns:
            Self for chaining.
        """
        kwargs: dict[str, Any] = {
            "column_A": column_a,
            "column_B": column_b,
            "or_equal": or_equal,
        }
        if parse_as_dates:
            kwargs["parse_strings_as_datetimes"] = True
        self._add_expectation(
            "expect_column_pair_values_A_to_be_greater_than_B",
            kwargs,
        )
        return self

    def expect_icd10_oncology_code(
        self,
        column: str,
        mostly: float = 1.0,
    ) -> "ExpectationBuilder":
        """Add expectation for valid ICD-10 oncology codes.

        Args:
            column: Column containing ICD-10 codes.
            mostly: Minimum proportion (0.0-1.0).

        Returns:
            Self for chaining.
        """
        # ICD-10 oncology codes: C00-D49
        self.expect_regex(
            column,
            regex=r"^C\d{2}(\.\d{1,4})?$|^D0[0-4](\.\d{1,4})?$",
            mostly=mostly,
        )
        return self

    def expect_ndc_code(
        self,
        column: str,
        mostly: float = 0.8,
    ) -> "ExpectationBuilder":
        """Add expectation for valid NDC (drug) codes.

        Args:
            column: Column containing NDC codes.
            mostly: Minimum proportion (0.0-1.0).

        Returns:
            Self for chaining.
        """
        # NDC format: XXXXX-XXXX-XX
        self.expect_regex(column, regex=r"^\d{5}-\d{4}-\d{2}$", mostly=mostly)
        return self

    def expect_loinc_code(
        self,
        column: str,
        mostly: float = 1.0,
    ) -> "ExpectationBuilder":
        """Add expectation for valid LOINC codes.

        Args:
            column: Column containing LOINC codes.
            mostly: Minimum proportion (0.0-1.0).

        Returns:
            Self for chaining.
        """
        # LOINC format: NNNNN-N or NNNN-N
        self.expect_regex(column, regex=r"^\d{4,5}-\d$", mostly=mostly)
        return self

    def _add_expectation(
        self,
        expectation_type: str,
        kwargs: dict[str, Any],
        meta: dict[str, Any] | None = None,
    ) -> None:
        """Add an expectation to the suite.

        Args:
            expectation_type: Type of expectation.
            kwargs: Expectation parameters.
            meta: Optional metadata.
        """
        expectation = {
            "expectation_type": expectation_type,
            "kwargs": kwargs,
            "meta": meta or {},
        }
        self.expectations.append(expectation)
        self._logger.debug("Added expectation", type=expectation_type)

    def build(self) -> ExpectationSuite | dict[str, Any]:
        """Build the expectation suite.

        Returns:
            ExpectationSuite if GE available, otherwise dict representation.
        """
        self._logger.info("Building expectation suite", expectation_count=len(self.expectations))

        if GE_AVAILABLE:
            suite = ExpectationSuite(expectation_suite_name=self.suite_name)
            for exp in self.expectations:
                config = ExpectationConfiguration(
                    expectation_type=exp["expectation_type"],
                    kwargs=exp["kwargs"],
                    meta=exp["meta"],
                )
                suite.add_expectation(config)
            return suite
        else:
            # Return dict representation when GE not available
            return {
                "expectation_suite_name": self.suite_name,
                "expectations": self.expectations,
            }

    def to_json(self) -> dict[str, Any]:
        """Export suite as JSON-compatible dictionary.

        Returns:
            Dictionary representation of the suite.
        """
        return {
            "expectation_suite_name": self.suite_name,
            "ge_cloud_id": None,
            "meta": {
                "great_expectations_version": "0.18.0",
            },
            "expectations": self.expectations,
        }


# Pre-built suite factories


def build_patient_suite() -> ExpectationBuilder:
    """Build standard patient data expectation suite.

    Returns:
        Configured ExpectationBuilder for patient data.
    """
    builder = ExpectationBuilder("oncology_patients_suite")

    # Required columns
    builder.expect_columns_to_match_set(
        [
            "patient_id", "mrn", "first_name", "last_name", "date_of_birth",
            "gender", "race", "ethnicity", "address_line1", "city", "state",
            "zip_code", "primary_diagnosis_code", "cancer_type", "cancer_stage",
        ],
        exact_match=False,
    )

    # Primary key
    builder.expect_not_null("patient_id")
    builder.expect_unique("patient_id")
    builder.expect_not_null("mrn")
    builder.expect_unique("mrn")
    builder.expect_regex("mrn", r"^MRN\d{8}$")

    # Demographics
    builder.expect_not_null("first_name")
    builder.expect_not_null("last_name")
    builder.expect_not_null("date_of_birth")

    # Code validations
    builder.expect_values_in_set("gender", [g.value for g in Gender])
    builder.expect_regex("state", r"^[A-Z]{2}$")
    builder.expect_regex("zip_code", r"^\d{5}(-\d{4})?$")

    # Oncology specific
    builder.expect_icd10_oncology_code("primary_diagnosis_code")
    builder.expect_values_in_set("cancer_type", [c.value for c in CancerType])
    builder.expect_regex("cancer_stage", r"^(I|II|III|IV)(A|B|C)?$")

    # Date logic
    builder.expect_column_pair_a_greater_than_b(
        "primary_diagnosis_date",
        "date_of_birth",
        parse_as_dates=True,
    )

    builder.expect_row_count_between(min_value=1)

    return builder


def build_treatment_suite() -> ExpectationBuilder:
    """Build standard treatment data expectation suite.

    Returns:
        Configured ExpectationBuilder for treatment data.
    """
    builder = ExpectationBuilder("oncology_treatments_suite")

    # Primary key
    builder.expect_not_null("treatment_id")
    builder.expect_unique("treatment_id")
    builder.expect_not_null("patient_id")

    # Treatment type
    builder.expect_values_in_set("treatment_type", [t.value for t in TreatmentType])
    builder.expect_values_in_set("status", [s.value for s in TreatmentStatus])

    # Drug codes (for chemo treatments)
    builder.expect_ndc_code("drug_code", mostly=0.8)

    # Dosage validation
    builder.expect_between("dosage", min_value=0, mostly=0.95)

    # Cycle validation
    builder.expect_between("cycles_planned", min_value=1, max_value=50, mostly=0.9)
    builder.expect_between("cycles_completed", min_value=0, max_value=50, mostly=0.9)

    builder.expect_row_count_between(min_value=1)

    return builder


def build_lab_results_suite() -> ExpectationBuilder:
    """Build standard lab results data expectation suite.

    Returns:
        Configured ExpectationBuilder for lab results data.
    """
    builder = ExpectationBuilder("oncology_lab_results_suite")

    # Primary key
    builder.expect_not_null("result_id")
    builder.expect_unique("result_id")
    builder.expect_not_null("patient_id")

    # Test identification
    builder.expect_not_null("test_code")
    builder.expect_not_null("test_name")
    builder.expect_values_in_set("test_category", [c.value for c in LabTestCategory])

    # Result validation
    builder.expect_between("result_value", min_value=0, mostly=0.98)

    # Specimen and timing
    builder.expect_not_null("specimen_type")
    builder.expect_not_null("collection_datetime")
    builder.expect_not_null("result_datetime")

    # Result must be after collection
    builder.expect_column_pair_a_greater_than_b(
        "result_datetime",
        "collection_datetime",
        or_equal=True,
        parse_as_dates=True,
    )

    builder.expect_row_count_between(min_value=1)

    return builder
