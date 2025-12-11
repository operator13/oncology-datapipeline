"""
Data quality module for the Oncology Data Pipeline.

This package provides Great Expectations integration for data validation,
including expectation builders, validation runners, and checkpoint management.

Example:
    >>> from src.data_quality import ValidationRunner, ExpectationBuilder
    >>>
    >>> # Validate a DataFrame
    >>> runner = ValidationRunner()
    >>> result = runner.validate_dataframe(df, "oncology_patients_suite")
    >>> print(f"Validation {'passed' if result.success else 'failed'}")
    >>>
    >>> # Build custom expectations
    >>> builder = ExpectationBuilder("custom_suite")
    >>> builder.expect_not_null("patient_id")
    >>> builder.expect_unique("mrn")
    >>> suite = builder.build()
"""

from src.data_quality.checkpoint_manager import (
    CheckpointConfig,
    CheckpointManager,
    CheckpointResult,
    create_oncology_checkpoints,
    run_oncology_validation_pipeline,
)
from src.data_quality.expectation_builder import (
    ExpectationBuilder,
    build_lab_results_suite,
    build_patient_suite,
    build_treatment_suite,
)
from src.data_quality.validation_runner import ValidationConfig, ValidationResult, ValidationRunner

__all__ = [
    # Validation
    "ValidationRunner",
    "ValidationResult",
    "ValidationConfig",
    # Expectation Builder
    "ExpectationBuilder",
    "build_patient_suite",
    "build_treatment_suite",
    "build_lab_results_suite",
    # Checkpoint Management
    "CheckpointManager",
    "CheckpointConfig",
    "CheckpointResult",
    "create_oncology_checkpoints",
    "run_oncology_validation_pipeline",
]
