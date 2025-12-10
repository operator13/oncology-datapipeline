"""
Data profiling and anomaly detection for oncology datasets.

This package provides tools for analyzing data quality, detecting
anomalies, and monitoring data drift.

Example:
    >>> from src.profiling import DataProfiler, AnomalyDetector, DriftDetector
    >>>
    >>> # Profile a dataset
    >>> profiler = DataProfiler()
    >>> profile = profiler.profile(df, "patients")
    >>> print(profile.summary())
    >>>
    >>> # Detect anomalies
    >>> detector = AnomalyDetector(method="iqr")
    >>> anomalies = detector.detect(df, columns=["lab_value"])
    >>>
    >>> # Check for drift
    >>> drift = DriftDetector()
    >>> result = drift.detect_schema_drift(old_df, new_df)
"""

from src.profiling.anomaly_detector import (
    AnomalyDetector,
    AnomalyMethod,
    AnomalyResult,
    detect_lab_value_anomalies,
)
from src.profiling.data_profiler import (
    ColumnProfile,
    DataProfile,
    DataProfiler,
)
from src.profiling.drift_detector import (
    DriftDetector,
    DriftResult,
    DriftType,
)

__all__ = [
    # Profiler
    "DataProfiler",
    "DataProfile",
    "ColumnProfile",
    # Anomaly Detection
    "AnomalyDetector",
    "AnomalyMethod",
    "AnomalyResult",
    "detect_lab_value_anomalies",
    # Drift Detection
    "DriftDetector",
    "DriftResult",
    "DriftType",
]
