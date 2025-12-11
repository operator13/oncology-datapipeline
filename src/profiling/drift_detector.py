"""
Data drift detection for oncology datasets.

This module provides capabilities for detecting schema changes
and data distribution drift between dataset versions.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

import numpy as np
import pandas as pd
import structlog

logger = structlog.get_logger(__name__)


class DriftType(str, Enum):
    """Types of data drift."""

    SCHEMA = "schema"
    DISTRIBUTION = "distribution"
    STATISTICAL = "statistical"


@dataclass
class DriftResult:
    """Result of drift detection.

    Attributes:
        drift_detected: Whether drift was detected.
        drift_type: Type of drift.
        drift_score: Quantified drift measure (0-1).
        details: Detailed drift information.
        timestamp: When detection was run.
    """

    drift_detected: bool
    drift_type: str
    drift_score: float
    details: dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.now)

    def summary(self) -> str:
        """Generate summary string."""
        status = "DETECTED" if self.drift_detected else "NOT DETECTED"
        return f"{self.drift_type} drift {status} (score: {self.drift_score:.3f})"


class DriftDetector:
    """Detector for schema and data distribution drift.

    This class compares datasets to detect changes in structure
    and statistical properties over time.

    Example:
        >>> detector = DriftDetector()
        >>> result = detector.detect_schema_drift(old_df, new_df)
        >>> if result.drift_detected:
        ...     print(f"Schema changed: {result.details}")
    """

    def __init__(self, threshold: float = 0.1) -> None:
        """Initialize the drift detector.

        Args:
            threshold: Drift score threshold for detection (0-1).
        """
        self.threshold = threshold
        self._logger = logger.bind(detector="drift")

    def detect_schema_drift(
        self,
        reference_df: pd.DataFrame,
        current_df: pd.DataFrame,
    ) -> DriftResult:
        """Detect schema changes between DataFrames.

        Args:
            reference_df: Reference (baseline) DataFrame.
            current_df: Current DataFrame to compare.

        Returns:
            DriftResult with schema drift details.
        """
        self._logger.info("Detecting schema drift")

        ref_cols = set(reference_df.columns)
        cur_cols = set(current_df.columns)

        added = cur_cols - ref_cols
        removed = ref_cols - cur_cols
        common = ref_cols & cur_cols

        # Check dtype changes
        dtype_changes = {}
        for col in common:
            ref_dtype = str(reference_df[col].dtype)
            cur_dtype = str(current_df[col].dtype)
            if ref_dtype != cur_dtype:
                dtype_changes[col] = {"from": ref_dtype, "to": cur_dtype}

        # Calculate drift score
        total_changes = len(added) + len(removed) + len(dtype_changes)
        drift_score = total_changes / max(len(ref_cols), 1)

        drift_detected = drift_score > 0 or bool(dtype_changes)

        return DriftResult(
            drift_detected=drift_detected,
            drift_type=DriftType.SCHEMA.value,
            drift_score=drift_score,
            details={
                "columns_added": list(added),
                "columns_removed": list(removed),
                "dtype_changes": dtype_changes,
                "reference_column_count": len(ref_cols),
                "current_column_count": len(cur_cols),
            },
        )

    def detect_distribution_drift(
        self,
        reference_df: pd.DataFrame,
        current_df: pd.DataFrame,
        columns: list[str] | None = None,
    ) -> dict[str, DriftResult]:
        """Detect distribution drift in numeric columns.

        Args:
            reference_df: Reference DataFrame.
            current_df: Current DataFrame.
            columns: Columns to check. Uses all numeric if None.

        Returns:
            Dictionary mapping columns to DriftResults.
        """
        if columns is None:
            common_cols = set(reference_df.columns) & set(current_df.columns)
            columns = [
                c
                for c in common_cols
                if pd.api.types.is_numeric_dtype(reference_df[c])
                and pd.api.types.is_numeric_dtype(current_df[c])
            ]

        self._logger.info("Detecting distribution drift", columns=len(columns))

        results = {}
        for col in columns:
            result = self._detect_column_drift(
                reference_df[col],
                current_df[col],
                col,
            )
            results[col] = result

        return results

    def detect_statistical_drift(
        self,
        reference_df: pd.DataFrame,
        current_df: pd.DataFrame,
        columns: list[str] | None = None,
    ) -> dict[str, DriftResult]:
        """Detect drift in statistical measures.

        Args:
            reference_df: Reference DataFrame.
            current_df: Current DataFrame.
            columns: Columns to check.

        Returns:
            Dictionary of DriftResults per column.
        """
        if columns is None:
            common_cols = set(reference_df.columns) & set(current_df.columns)
            columns = [c for c in common_cols if pd.api.types.is_numeric_dtype(reference_df[c])]

        self._logger.info("Detecting statistical drift", columns=len(columns))

        results = {}
        for col in columns:
            result = self._detect_statistical_column_drift(
                reference_df[col],
                current_df[col],
                col,
            )
            results[col] = result

        return results

    def _detect_column_drift(
        self,
        reference: pd.Series,
        current: pd.Series,
        name: str,
    ) -> DriftResult:
        """Detect distribution drift for a single column.

        Uses Kolmogorov-Smirnov test for continuous data.

        Args:
            reference: Reference column data.
            current: Current column data.
            name: Column name.

        Returns:
            DriftResult for the column.
        """
        try:
            from scipy import stats

            # Clean data
            ref_clean = reference.dropna()
            cur_clean = current.dropna()

            # KS test
            statistic, p_value = stats.ks_2samp(ref_clean, cur_clean)

            drift_detected = p_value < self.threshold

            return DriftResult(
                drift_detected=drift_detected,
                drift_type=DriftType.DISTRIBUTION.value,
                drift_score=statistic,
                details={
                    "column": name,
                    "test": "kolmogorov_smirnov",
                    "statistic": float(statistic),
                    "p_value": float(p_value),
                    "threshold": self.threshold,
                },
            )

        except ImportError:
            self._logger.warning("scipy not available for KS test")
            return self._detect_statistical_column_drift(reference, current, name)

    def _detect_statistical_column_drift(
        self,
        reference: pd.Series,
        current: pd.Series,
        name: str,
    ) -> DriftResult:
        """Detect drift using statistical measures.

        Args:
            reference: Reference column data.
            current: Current column data.
            name: Column name.

        Returns:
            DriftResult with statistical comparison.
        """
        ref_clean = reference.dropna()
        cur_clean = current.dropna()

        # Calculate statistics
        ref_stats = {
            "mean": float(ref_clean.mean()),
            "std": float(ref_clean.std()),
            "min": float(ref_clean.min()),
            "max": float(ref_clean.max()),
            "median": float(ref_clean.median()),
        }

        cur_stats = {
            "mean": float(cur_clean.mean()),
            "std": float(cur_clean.std()),
            "min": float(cur_clean.min()),
            "max": float(cur_clean.max()),
            "median": float(cur_clean.median()),
        }

        # Calculate relative differences
        mean_diff = abs(ref_stats["mean"] - cur_stats["mean"])
        mean_diff_pct = mean_diff / (abs(ref_stats["mean"]) + 1e-10)

        std_diff = abs(ref_stats["std"] - cur_stats["std"])
        std_diff_pct = std_diff / (abs(ref_stats["std"]) + 1e-10)

        # Composite drift score
        drift_score = (mean_diff_pct + std_diff_pct) / 2

        drift_detected = drift_score > self.threshold

        return DriftResult(
            drift_detected=drift_detected,
            drift_type=DriftType.STATISTICAL.value,
            drift_score=drift_score,
            details={
                "column": name,
                "reference_stats": ref_stats,
                "current_stats": cur_stats,
                "mean_diff_pct": float(mean_diff_pct),
                "std_diff_pct": float(std_diff_pct),
            },
        )

    def compare_datasets(
        self,
        reference_df: pd.DataFrame,
        current_df: pd.DataFrame,
    ) -> dict[str, Any]:
        """Run comprehensive drift comparison.

        Args:
            reference_df: Reference DataFrame.
            current_df: Current DataFrame.

        Returns:
            Dictionary with all drift detection results.
        """
        self._logger.info("Running comprehensive drift comparison")

        schema_drift = self.detect_schema_drift(reference_df, current_df)
        distribution_drift = self.detect_distribution_drift(reference_df, current_df)

        # Overall drift status
        any_drift = schema_drift.drift_detected or any(
            r.drift_detected for r in distribution_drift.values()
        )

        return {
            "drift_detected": any_drift,
            "schema_drift": schema_drift,
            "distribution_drift": distribution_drift,
            "comparison_timestamp": datetime.now().isoformat(),
        }
