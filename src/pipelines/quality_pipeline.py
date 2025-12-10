"""
Data quality pipeline for oncology datasets.

This module provides a pipeline that orchestrates data quality
checks using Great Expectations and profiling tools.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Any

import pandas as pd
import structlog

from src.data_quality import (
    ValidationResult,
    ValidationRunner,
    build_lab_results_suite,
    build_patient_suite,
    build_treatment_suite,
)
from src.pipelines.base_pipeline import DataFramePipeline, PipelineResult, PipelineStatus
from src.profiling import AnomalyDetector, DataProfiler

logger = structlog.get_logger(__name__)


@dataclass
class QualityPipelineResult:
    """Result of quality pipeline execution.

    Attributes:
        pipeline_result: Base pipeline result.
        validation_results: GE validation results per suite.
        profile_summary: Data profile summary.
        anomaly_count: Total anomalies detected.
        quality_score: Overall quality score (0-100).
    """

    pipeline_result: PipelineResult
    validation_results: dict[str, ValidationResult]
    profile_summary: dict[str, Any]
    anomaly_count: int
    quality_score: float

    @property
    def passed(self) -> bool:
        """Check if all validations passed."""
        return all(r.success for r in self.validation_results.values())

    def summary(self) -> str:
        """Generate quality summary."""
        lines = [
            "=" * 50,
            "DATA QUALITY PIPELINE RESULTS",
            "=" * 50,
            f"Overall Status: {'PASSED' if self.passed else 'FAILED'}",
            f"Quality Score: {self.quality_score:.1f}/100",
            f"Anomalies Detected: {self.anomaly_count}",
            "",
            "Validation Results:",
        ]

        for suite, result in self.validation_results.items():
            status = "PASS" if result.success else "FAIL"
            lines.append(f"  [{status}] {suite}: {result.success_percent:.1f}%")

        return "\n".join(lines)


class QualityPipeline:
    """Pipeline for running comprehensive data quality checks.

    This pipeline orchestrates:
    - Great Expectations validations
    - Data profiling
    - Anomaly detection
    - Quality score calculation

    Example:
        >>> pipeline = QualityPipeline()
        >>> result = pipeline.run(
        ...     patients_df=patients,
        ...     treatments_df=treatments,
        ...     lab_results_df=lab_results,
        ... )
        >>> print(result.summary())
    """

    def __init__(
        self,
        validation_runner: ValidationRunner | None = None,
        profiler: DataProfiler | None = None,
        anomaly_detector: AnomalyDetector | None = None,
    ) -> None:
        """Initialize the quality pipeline.

        Args:
            validation_runner: Optional validation runner.
            profiler: Optional data profiler.
            anomaly_detector: Optional anomaly detector.
        """
        self.validation_runner = validation_runner or ValidationRunner()
        self.profiler = profiler or DataProfiler()
        self.anomaly_detector = anomaly_detector or AnomalyDetector()
        self._logger = logger.bind(pipeline="quality")

    def run(
        self,
        patients_df: pd.DataFrame | None = None,
        treatments_df: pd.DataFrame | None = None,
        lab_results_df: pd.DataFrame | None = None,
    ) -> QualityPipelineResult:
        """Run the complete quality pipeline.

        Args:
            patients_df: Patient data DataFrame.
            treatments_df: Treatment data DataFrame.
            lab_results_df: Lab results DataFrame.

        Returns:
            QualityPipelineResult with all quality metrics.
        """
        self._logger.info("Starting quality pipeline")
        start_time = datetime.now()

        validation_results: dict[str, ValidationResult] = {}
        profiles: dict[str, Any] = {}
        total_anomalies = 0

        # Validate patients
        if patients_df is not None:
            self._logger.info("Validating patient data", rows=len(patients_df))
            result = self.validation_runner.validate_dataframe(
                patients_df, "oncology_patients_suite"
            )
            validation_results["patients"] = result

            profile = self.profiler.profile(patients_df, "patients")
            profiles["patients"] = profile.summary()

        # Validate treatments
        if treatments_df is not None:
            self._logger.info("Validating treatment data", rows=len(treatments_df))
            result = self.validation_runner.validate_dataframe(
                treatments_df, "oncology_treatments_suite"
            )
            validation_results["treatments"] = result

            profile = self.profiler.profile(treatments_df, "treatments")
            profiles["treatments"] = profile.summary()

        # Validate lab results
        if lab_results_df is not None:
            self._logger.info("Validating lab results", rows=len(lab_results_df))
            result = self.validation_runner.validate_dataframe(
                lab_results_df, "oncology_lab_results_suite"
            )
            validation_results["lab_results"] = result

            profile = self.profiler.profile(lab_results_df, "lab_results")
            profiles["lab_results"] = profile.summary()

            # Detect anomalies in lab values
            numeric_cols = lab_results_df.select_dtypes(include=["number"]).columns.tolist()
            if numeric_cols:
                anomalies = self.anomaly_detector.detect(lab_results_df, columns=numeric_cols)
                total_anomalies = sum(r.anomaly_count for r in anomalies.values())

        # Calculate quality score
        quality_score = self._calculate_quality_score(validation_results)

        # Create pipeline result
        pipeline_result = PipelineResult(
            pipeline_name="quality_pipeline",
            status=PipelineStatus.COMPLETED,
            start_time=start_time,
            end_time=datetime.now(),
            records_processed=sum(
                len(df) for df in [patients_df, treatments_df, lab_results_df] if df is not None
            ),
        )

        result = QualityPipelineResult(
            pipeline_result=pipeline_result,
            validation_results=validation_results,
            profile_summary=profiles,
            anomaly_count=total_anomalies,
            quality_score=quality_score,
        )

        self._logger.info(
            "Quality pipeline complete",
            quality_score=quality_score,
            passed=result.passed,
        )

        return result

    def _calculate_quality_score(
        self,
        validation_results: dict[str, ValidationResult],
    ) -> float:
        """Calculate overall quality score (0-100).

        Args:
            validation_results: Validation results per dataset.

        Returns:
            Quality score between 0 and 100.
        """
        if not validation_results:
            return 0.0

        # Average of success percentages
        scores = [r.success_percent for r in validation_results.values()]
        return sum(scores) / len(scores)

    def run_single_dataset(
        self,
        df: pd.DataFrame,
        suite_name: str,
        dataset_name: str,
    ) -> ValidationResult:
        """Run quality checks on a single dataset.

        Args:
            df: DataFrame to validate.
            suite_name: Name of expectation suite.
            dataset_name: Name for logging.

        Returns:
            ValidationResult for the dataset.
        """
        self._logger.info(f"Validating {dataset_name}", rows=len(df))
        return self.validation_runner.validate_dataframe(df, suite_name)


def run_quick_quality_check(
    df: pd.DataFrame,
    dataset_type: str = "patients",
) -> ValidationResult:
    """Run a quick quality check on a DataFrame.

    Convenience function for quick validations.

    Args:
        df: DataFrame to validate.
        dataset_type: Type of data ('patients', 'treatments', 'lab_results').

    Returns:
        ValidationResult with validation outcome.
    """
    suite_map = {
        "patients": "oncology_patients_suite",
        "treatments": "oncology_treatments_suite",
        "lab_results": "oncology_lab_results_suite",
    }

    suite_name = suite_map.get(dataset_type, "oncology_patients_suite")
    runner = ValidationRunner()
    return runner.validate_dataframe(df, suite_name)
