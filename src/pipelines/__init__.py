"""
Pipeline definitions for oncology data processing.

This package provides reusable pipeline classes for ETL operations,
data quality checks, and data transformations.

Example:
    >>> from src.pipelines import QualityPipeline, run_quick_quality_check
    >>>
    >>> # Run comprehensive quality checks
    >>> pipeline = QualityPipeline()
    >>> result = pipeline.run(patients_df=df)
    >>> print(result.summary())
    >>>
    >>> # Quick validation
    >>> result = run_quick_quality_check(df, "patients")
    >>> print(f"Passed: {result.success}")
"""

from src.pipelines.base_pipeline import (
    BasePipeline,
    DataFramePipeline,
    PipelineResult,
    PipelineStatus,
)
from src.pipelines.quality_pipeline import (
    QualityPipeline,
    QualityPipelineResult,
    run_quick_quality_check,
)

__all__ = [
    # Base classes
    "BasePipeline",
    "DataFramePipeline",
    "PipelineResult",
    "PipelineStatus",
    # Quality Pipeline
    "QualityPipeline",
    "QualityPipelineResult",
    "run_quick_quality_check",
]
