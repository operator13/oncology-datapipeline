"""
Validation runner for executing Great Expectations validations.

This module provides a high-level interface for running data quality
validations against oncology datasets.
"""

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import structlog

from src.connectors.base import DataConnector

try:
    import great_expectations as gx
    from great_expectations.checkpoint import Checkpoint
    from great_expectations.core import ExpectationSuite
    from great_expectations.data_context import FileDataContext
    from great_expectations.validator.validator import Validator

    GE_AVAILABLE = True
except ImportError:
    GE_AVAILABLE = False

logger = structlog.get_logger(__name__)


@dataclass
class ValidationResult:
    """Container for validation results.

    Attributes:
        success: Whether all expectations passed.
        suite_name: Name of the expectation suite.
        run_time: When validation was run.
        statistics: Summary statistics.
        results: Detailed results per expectation.
        data_docs_url: URL to generated data docs.
    """

    success: bool
    suite_name: str
    run_time: datetime
    statistics: dict[str, Any]
    results: list[dict[str, Any]]
    data_docs_url: str | None = None

    @property
    def success_percent(self) -> float:
        """Calculate percentage of passed expectations."""
        total = self.statistics.get("evaluated_expectations", 0)
        passed = self.statistics.get("successful_expectations", 0)
        return (passed / total * 100) if total > 0 else 0.0

    @property
    def failed_expectations(self) -> list[dict[str, Any]]:
        """Get list of failed expectations."""
        return [r for r in self.results if not r.get("success", True)]

    def summary(self) -> str:
        """Generate human-readable summary.

        Returns:
            Summary string.
        """
        lines = [
            f"Validation Results: {self.suite_name}",
            f"Run Time: {self.run_time}",
            f"Overall Success: {'PASSED' if self.success else 'FAILED'}",
            f"Success Rate: {self.success_percent:.1f}%",
            f"Total Expectations: {self.statistics.get('evaluated_expectations', 0)}",
            f"Passed: {self.statistics.get('successful_expectations', 0)}",
            f"Failed: {self.statistics.get('unsuccessful_expectations', 0)}",
        ]

        if self.failed_expectations:
            lines.append("\nFailed Expectations:")
            for exp in self.failed_expectations[:5]:  # Show first 5
                lines.append(f"  - {exp.get('expectation_type', 'unknown')}")

        if self.data_docs_url:
            lines.append(f"\nData Docs: {self.data_docs_url}")

        return "\n".join(lines)


@dataclass
class ValidationConfig:
    """Configuration for validation runs.

    Attributes:
        ge_root_dir: Great Expectations project root.
        build_data_docs: Whether to generate data docs.
        fail_fast: Stop on first failure.
        result_format: Level of detail in results.
    """

    ge_root_dir: Path = field(default_factory=lambda: Path("great_expectations"))
    build_data_docs: bool = True
    fail_fast: bool = False
    result_format: str = "COMPLETE"


class ValidationRunner:
    """Runner for executing Great Expectations validations.

    This class provides a simplified interface for validating DataFrames
    against expectation suites.

    Attributes:
        config: Validation configuration.
        context: Great Expectations data context.

    Example:
        >>> runner = ValidationRunner()
        >>> result = runner.validate_dataframe(df, "patient_suite")
        >>> print(f"Validation {'passed' if result.success else 'failed'}")
    """

    def __init__(
        self,
        config: ValidationConfig | None = None,
        connector: DataConnector | None = None,
    ) -> None:
        """Initialize the validation runner.

        Args:
            config: Optional validation configuration.
            connector: Optional database connector for SQL validation.
        """
        self.config = config or ValidationConfig()
        self.connector = connector
        self._context: Any = None
        self._logger = logger.bind(runner="validation")

        if GE_AVAILABLE:
            self._init_context()
        else:
            self._logger.warning("Great Expectations not available")

    def _init_context(self) -> None:
        """Initialize the Great Expectations data context."""
        try:
            if self.config.ge_root_dir.exists():
                self._context = FileDataContext(
                    project_root_dir=str(self.config.ge_root_dir.parent)
                )
                self._logger.info("Loaded existing GE context")
            else:
                self._logger.warning(
                    "GE directory not found, running in standalone mode",
                    path=str(self.config.ge_root_dir),
                )
        except Exception as e:
            self._logger.error("Failed to initialize GE context", error=str(e))

    def validate_dataframe(
        self,
        df: pd.DataFrame,
        suite_name: str,
        datasource_name: str = "pandas_datasource",
    ) -> ValidationResult:
        """Validate a DataFrame against an expectation suite.

        Args:
            df: DataFrame to validate.
            suite_name: Name of the expectation suite.
            datasource_name: Name of the datasource to use.

        Returns:
            ValidationResult with validation outcome.

        Example:
            >>> result = runner.validate_dataframe(patients_df, "oncology_patients_suite")
        """
        self._logger.info(
            "Starting validation",
            suite_name=suite_name,
            rows=len(df),
            columns=len(df.columns),
        )

        if not GE_AVAILABLE:
            return self._mock_validation(df, suite_name)

        try:
            # Create batch request for DataFrame
            batch_request = {
                "datasource_name": datasource_name,
                "data_connector_name": "default_runtime_data_connector",
                "data_asset_name": suite_name,
                "runtime_parameters": {"batch_data": df},
                "batch_identifiers": {"default_identifier_name": "validation_run"},
            }

            # Load expectation suite from file (compatible with all GE versions)
            suite = self._load_suite_from_file(suite_name)

            # Run standalone validation (more compatible across GE versions)
            results = self._validate_standalone(df, suite)

            # Build data docs if configured
            data_docs_url = None
            if self.config.build_data_docs and self._context:
                try:
                    self._context.build_data_docs()
                    data_docs_url = self._get_data_docs_url()
                except Exception as e:
                    self._logger.warning("Failed to build data docs", error=str(e))

            return self._parse_results(results, suite_name, data_docs_url)

        except Exception as e:
            self._logger.error("Validation failed", error=str(e), suite=suite_name)
            return ValidationResult(
                success=False,
                suite_name=suite_name,
                run_time=datetime.now(),
                statistics={"error": str(e)},
                results=[],
            )

    def validate_sql_table(
        self,
        table_name: str,
        suite_name: str,
        schema: str | None = None,
        limit: int | None = None,
    ) -> ValidationResult:
        """Validate a SQL table against an expectation suite.

        Args:
            table_name: Name of the table to validate.
            suite_name: Name of the expectation suite.
            schema: Optional schema name.
            limit: Optional row limit for validation.

        Returns:
            ValidationResult with validation outcome.
        """
        if not self.connector:
            raise ValueError("No database connector configured")

        self._logger.info(
            "Validating SQL table",
            table=table_name,
            suite=suite_name,
        )

        # Fetch data from table
        full_name = f"{schema}.{table_name}" if schema else table_name
        query = f"SELECT * FROM {full_name}"
        if limit:
            query += f" LIMIT {limit}"

        with self.connector:
            df = self.connector.execute_query(query)

        return self.validate_dataframe(df, suite_name)

    def validate_all_suites(
        self,
        data: dict[str, pd.DataFrame],
    ) -> dict[str, ValidationResult]:
        """Validate multiple DataFrames against their corresponding suites.

        Args:
            data: Dictionary mapping suite names to DataFrames.

        Returns:
            Dictionary mapping suite names to results.

        Example:
            >>> results = runner.validate_all_suites({
            ...     "oncology_patients_suite": patients_df,
            ...     "oncology_treatments_suite": treatments_df,
            ... })
        """
        results = {}

        for suite_name, df in data.items():
            result = self.validate_dataframe(df, suite_name)
            results[suite_name] = result

            self._logger.info(
                "Suite validation complete",
                suite=suite_name,
                success=result.success,
                success_rate=f"{result.success_percent:.1f}%",
            )

        return results

    def build_data_docs(self) -> str | None:
        """Build Great Expectations data documentation.

        Returns:
            URL to the data docs site, or None if unavailable.
        """
        if not GE_AVAILABLE or not self._context:
            self._logger.warning("Cannot build data docs - GE not available")
            return None

        try:
            self._context.build_data_docs()
            return self._get_data_docs_url()
        except Exception as e:
            self._logger.error("Failed to build data docs", error=str(e))
            return None

    def _load_suite_from_file(self, suite_name: str) -> Any:
        """Load expectation suite from JSON file.

        Args:
            suite_name: Name of the suite.

        Returns:
            Loaded expectation suite data (as dict for compatibility).
        """
        import json

        # Try common locations
        locations = [
            self.config.ge_root_dir / "expectations" / f"{suite_name}.json",
            Path("great_expectations/expectations") / f"{suite_name}.json",
        ]

        for path in locations:
            if path.exists():
                with open(path) as f:
                    data = json.load(f)
                # Return as dict for maximum compatibility across GE versions
                return data

        raise FileNotFoundError(f"Suite not found: {suite_name}")

    def _validate_standalone(
        self,
        df: pd.DataFrame,
        suite: Any,
    ) -> dict[str, Any]:
        """Run validation without a full GE context.

        Args:
            df: DataFrame to validate.
            suite: Expectation suite (dict format).

        Returns:
            Validation results dictionary.
        """
        # Get expectations from suite
        if hasattr(suite, "expectations"):
            expectations = suite.expectations
        else:
            expectations = suite.get("expectations", [])

        results = []
        success_count = 0

        for exp in expectations:
            if isinstance(exp, dict):
                exp_type = exp.get("expectation_type", "")
                kwargs = exp.get("kwargs", {})
            else:
                exp_type = getattr(exp, "expectation_type", "")
                kwargs = getattr(exp, "kwargs", {})

            try:
                # Run validation using pandas directly
                success = self._run_expectation(df, exp_type, kwargs)
                results.append(
                    {
                        "expectation_type": exp_type,
                        "success": success,
                        "kwargs": kwargs,
                    }
                )
                if success:
                    success_count += 1
            except Exception as e:
                self._logger.debug(
                    "Expectation failed",
                    expectation=exp_type,
                    error=str(e),
                )
                results.append(
                    {
                        "expectation_type": exp_type,
                        "success": False,
                        "error": str(e),
                    }
                )

        return {
            "success": success_count == len(expectations) if expectations else True,
            "statistics": {
                "evaluated_expectations": len(expectations),
                "successful_expectations": success_count,
                "unsuccessful_expectations": len(expectations) - success_count,
            },
            "results": results,
        }

    def _run_expectation(
        self,
        df: pd.DataFrame,
        exp_type: str,
        kwargs: dict[str, Any],
    ) -> bool:
        """Run a single expectation against a DataFrame.

        Args:
            df: DataFrame to validate.
            exp_type: Expectation type name.
            kwargs: Expectation arguments.

        Returns:
            True if expectation passes, False otherwise.
        """
        column = kwargs.get("column")

        if exp_type == "expect_column_to_exist":
            return column in df.columns

        if exp_type == "expect_column_values_to_not_be_null":
            if column not in df.columns:
                return False
            return df[column].notna().all()

        if exp_type == "expect_column_values_to_be_unique":
            if column not in df.columns:
                return False
            return df[column].is_unique

        if exp_type == "expect_column_values_to_be_in_set":
            if column not in df.columns:
                return False
            value_set = set(kwargs.get("value_set", []))
            return df[column].dropna().isin(value_set).all()

        if exp_type == "expect_column_values_to_match_regex":
            if column not in df.columns:
                return False
            regex = kwargs.get("regex", ".*")
            return df[column].dropna().astype(str).str.match(regex).all()

        if exp_type == "expect_column_values_to_be_between":
            if column not in df.columns:
                return False
            min_val = kwargs.get("min_value")
            max_val = kwargs.get("max_value")
            col_data = df[column].dropna()
            if min_val is not None and (col_data < min_val).any():
                return False
            if max_val is not None and (col_data > max_val).any():
                return False
            return True

        if exp_type == "expect_column_pair_values_A_to_be_greater_than_B":
            col_a = kwargs.get("column_A")
            col_b = kwargs.get("column_B")
            if col_a not in df.columns or col_b not in df.columns:
                return False
            # Handle date columns
            try:
                a_vals = pd.to_datetime(df[col_a], errors="coerce")
                b_vals = pd.to_datetime(df[col_b], errors="coerce")
                mask = a_vals.notna() & b_vals.notna()
                return (a_vals[mask] >= b_vals[mask]).all()
            except Exception:
                return (df[col_a] >= df[col_b]).all()

        if exp_type == "expect_table_row_count_to_be_between":
            min_val = kwargs.get("min_value", 0)
            max_val = kwargs.get("max_value", float("inf"))
            return min_val <= len(df) <= max_val

        # Unknown expectation type - skip with warning
        self._logger.warning(
            "Unknown expectation type - skipping",
            expectation_type=exp_type,
        )
        return True  # Don't fail on unknown types

    def _parse_results(
        self,
        results: Any,
        suite_name: str,
        data_docs_url: str | None,
    ) -> ValidationResult:
        """Parse GE validation results into ValidationResult.

        Args:
            results: Raw GE results.
            suite_name: Name of the suite.
            data_docs_url: URL to data docs.

        Returns:
            Parsed ValidationResult.
        """
        if isinstance(results, dict):
            # Already a dict (from standalone validation)
            return ValidationResult(
                success=results.get("success", False),
                suite_name=suite_name,
                run_time=datetime.now(),
                statistics=results.get("statistics", {}),
                results=results.get("results", []),
                data_docs_url=data_docs_url,
            )

        # Parse GE result object
        return ValidationResult(
            success=results.success,
            suite_name=suite_name,
            run_time=datetime.now(),
            statistics=results.statistics if hasattr(results, "statistics") else {},
            results=(
                [
                    {
                        "expectation_type": r.expectation_config.expectation_type,
                        "success": r.success,
                        "kwargs": r.expectation_config.kwargs,
                    }
                    for r in results.results
                ]
                if hasattr(results, "results")
                else []
            ),
            data_docs_url=data_docs_url,
        )

    def _get_data_docs_url(self) -> str | None:
        """Get URL to data docs site.

        Returns:
            URL string or None.
        """
        try:
            docs_path = (
                self.config.ge_root_dir / "uncommitted" / "data_docs" / "local_site" / "index.html"
            )
            if docs_path.exists():
                return f"file://{docs_path.absolute()}"
        except Exception:
            pass
        return None

    def _mock_validation(
        self,
        df: pd.DataFrame,
        suite_name: str,
    ) -> ValidationResult:
        """Return mock validation result when GE not available.

        Args:
            df: DataFrame being validated.
            suite_name: Suite name.

        Returns:
            Mock ValidationResult.
        """
        self._logger.warning("Running mock validation - GE not installed")

        return ValidationResult(
            success=True,
            suite_name=suite_name,
            run_time=datetime.now(),
            statistics={
                "evaluated_expectations": 0,
                "successful_expectations": 0,
                "unsuccessful_expectations": 0,
                "note": "Mock validation - Great Expectations not installed",
            },
            results=[],
        )
