"""
Anomaly detection for oncology datasets.

This module provides multiple methods for detecting anomalies
and outliers in data, essential for data quality monitoring.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np
import pandas as pd
import structlog

logger = structlog.get_logger(__name__)


class AnomalyMethod(str, Enum):
    """Anomaly detection methods."""

    IQR = "iqr"
    ZSCORE = "zscore"
    ISOLATION_FOREST = "isolation_forest"
    MAD = "mad"  # Median Absolute Deviation


@dataclass
class AnomalyResult:
    """Result of anomaly detection.

    Attributes:
        method: Detection method used.
        column: Column analyzed (if applicable).
        anomaly_count: Number of anomalies detected.
        anomaly_indices: Indices of anomalous records.
        anomaly_rate: Proportion of anomalies.
        threshold: Threshold used for detection.
        details: Additional method-specific details.
    """

    method: str
    column: str | None
    anomaly_count: int
    anomaly_indices: list[int]
    anomaly_rate: float
    threshold: float | None = None
    details: dict[str, Any] | None = None

    def summary(self) -> str:
        """Generate summary string."""
        col_str = f" in '{self.column}'" if self.column else ""
        return (
            f"Detected {self.anomaly_count} anomalies{col_str} "
            f"({self.anomaly_rate:.2%}) using {self.method}"
        )


class AnomalyDetector:
    """Detector for identifying data anomalies.

    This class provides multiple anomaly detection methods suitable
    for different data types and distributions.

    Attributes:
        method: Default detection method.
        threshold: Sensitivity threshold.

    Example:
        >>> detector = AnomalyDetector(method="iqr")
        >>> result = detector.detect(df, columns=["age", "lab_value"])
        >>> print(f"Found {result.anomaly_count} anomalies")
    """

    def __init__(
        self,
        method: str | AnomalyMethod = AnomalyMethod.IQR,
        threshold: float = 1.5,
    ) -> None:
        """Initialize the anomaly detector.

        Args:
            method: Detection method to use.
            threshold: Sensitivity threshold (method-specific).
        """
        if isinstance(method, str):
            method = AnomalyMethod(method.lower())
        self.method = method
        self.threshold = threshold
        self._logger = logger.bind(detector="anomaly", method=method.value)

    def detect(
        self,
        df: pd.DataFrame,
        columns: list[str] | None = None,
    ) -> dict[str, AnomalyResult]:
        """Detect anomalies in a DataFrame.

        Args:
            df: DataFrame to analyze.
            columns: Specific columns to check. Uses all numeric if None.

        Returns:
            Dictionary mapping column names to AnomalyResults.

        Example:
            >>> results = detector.detect(df, columns=["wbc_count", "hemoglobin"])
            >>> for col, result in results.items():
            ...     print(result.summary())
        """
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()

        self._logger.info(
            "Starting anomaly detection",
            columns=len(columns),
            rows=len(df),
        )

        results = {}
        for col in columns:
            if col not in df.columns:
                self._logger.warning(f"Column not found: {col}")
                continue

            result = self._detect_column(df[col], col)
            results[col] = result

        total_anomalies = sum(r.anomaly_count for r in results.values())
        self._logger.info("Detection complete", total_anomalies=total_anomalies)

        return results

    def detect_multivariate(
        self,
        df: pd.DataFrame,
        columns: list[str] | None = None,
    ) -> AnomalyResult:
        """Detect multivariate anomalies using Isolation Forest.

        Args:
            df: DataFrame to analyze.
            columns: Columns to include.

        Returns:
            AnomalyResult with multivariate anomalies.
        """
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()

        self._logger.info("Running multivariate detection", columns=len(columns))

        try:
            from sklearn.ensemble import IsolationForest

            # Prepare data
            X = df[columns].dropna()

            # Fit Isolation Forest
            clf = IsolationForest(
                contamination=min(0.1, 100 / len(X)),
                random_state=42,
                n_estimators=100,
            )
            predictions = clf.fit_predict(X)

            # -1 indicates anomaly
            anomaly_mask = predictions == -1
            anomaly_indices = X.index[anomaly_mask].tolist()

            return AnomalyResult(
                method="isolation_forest_multivariate",
                column=None,
                anomaly_count=len(anomaly_indices),
                anomaly_indices=anomaly_indices,
                anomaly_rate=len(anomaly_indices) / len(X),
                details={"columns": columns, "n_estimators": 100},
            )

        except ImportError:
            self._logger.error("scikit-learn not available for Isolation Forest")
            return AnomalyResult(
                method="isolation_forest_multivariate",
                column=None,
                anomaly_count=0,
                anomaly_indices=[],
                anomaly_rate=0.0,
                details={"error": "scikit-learn not installed"},
            )

    def _detect_column(self, series: pd.Series, name: str) -> AnomalyResult:
        """Detect anomalies in a single column.

        Args:
            series: Column data.
            name: Column name.

        Returns:
            AnomalyResult for the column.
        """
        if self.method == AnomalyMethod.IQR:
            return self._detect_iqr(series, name)
        elif self.method == AnomalyMethod.ZSCORE:
            return self._detect_zscore(series, name)
        elif self.method == AnomalyMethod.MAD:
            return self._detect_mad(series, name)
        elif self.method == AnomalyMethod.ISOLATION_FOREST:
            return self._detect_isolation_forest(series, name)
        else:
            raise ValueError(f"Unknown method: {self.method}")

    def _detect_iqr(self, series: pd.Series, name: str) -> AnomalyResult:
        """Detect anomalies using IQR method.

        Args:
            series: Column data.
            name: Column name.

        Returns:
            AnomalyResult using IQR bounds.
        """
        clean = series.dropna()
        Q1 = clean.quantile(0.25)
        Q3 = clean.quantile(0.75)
        IQR = Q3 - Q1

        lower_bound = Q1 - self.threshold * IQR
        upper_bound = Q3 + self.threshold * IQR

        anomaly_mask = (series < lower_bound) | (series > upper_bound)
        anomaly_indices = series.index[anomaly_mask].tolist()

        return AnomalyResult(
            method="iqr",
            column=name,
            anomaly_count=len(anomaly_indices),
            anomaly_indices=anomaly_indices,
            anomaly_rate=len(anomaly_indices) / len(series) if len(series) > 0 else 0,
            threshold=self.threshold,
            details={
                "Q1": float(Q1),
                "Q3": float(Q3),
                "IQR": float(IQR),
                "lower_bound": float(lower_bound),
                "upper_bound": float(upper_bound),
            },
        )

    def _detect_zscore(self, series: pd.Series, name: str) -> AnomalyResult:
        """Detect anomalies using Z-score method.

        Args:
            series: Column data.
            name: Column name.

        Returns:
            AnomalyResult using Z-score threshold.
        """
        clean = series.dropna()
        mean = clean.mean()
        std = clean.std()

        if std == 0:
            return AnomalyResult(
                method="zscore",
                column=name,
                anomaly_count=0,
                anomaly_indices=[],
                anomaly_rate=0.0,
                details={"error": "Zero standard deviation"},
            )

        z_scores = np.abs((series - mean) / std)
        anomaly_mask = z_scores > self.threshold
        anomaly_indices = series.index[anomaly_mask].tolist()

        return AnomalyResult(
            method="zscore",
            column=name,
            anomaly_count=len(anomaly_indices),
            anomaly_indices=anomaly_indices,
            anomaly_rate=len(anomaly_indices) / len(series) if len(series) > 0 else 0,
            threshold=self.threshold,
            details={
                "mean": float(mean),
                "std": float(std),
                "z_threshold": float(self.threshold),
            },
        )

    def _detect_mad(self, series: pd.Series, name: str) -> AnomalyResult:
        """Detect anomalies using Median Absolute Deviation.

        Args:
            series: Column data.
            name: Column name.

        Returns:
            AnomalyResult using MAD threshold.
        """
        clean = series.dropna()
        median = clean.median()
        mad = np.median(np.abs(clean - median))

        if mad == 0:
            return AnomalyResult(
                method="mad",
                column=name,
                anomaly_count=0,
                anomaly_indices=[],
                anomaly_rate=0.0,
                details={"error": "Zero MAD"},
            )

        # Modified Z-score using MAD
        k = 1.4826  # Consistency constant for normal distribution
        modified_z = np.abs((series - median) / (k * mad))
        anomaly_mask = modified_z > self.threshold
        anomaly_indices = series.index[anomaly_mask].tolist()

        return AnomalyResult(
            method="mad",
            column=name,
            anomaly_count=len(anomaly_indices),
            anomaly_indices=anomaly_indices,
            anomaly_rate=len(anomaly_indices) / len(series) if len(series) > 0 else 0,
            threshold=self.threshold,
            details={
                "median": float(median),
                "mad": float(mad),
                "threshold": float(self.threshold),
            },
        )

    def _detect_isolation_forest(self, series: pd.Series, name: str) -> AnomalyResult:
        """Detect anomalies using Isolation Forest.

        Args:
            series: Column data.
            name: Column name.

        Returns:
            AnomalyResult using Isolation Forest.
        """
        try:
            from sklearn.ensemble import IsolationForest

            clean = series.dropna()
            X = clean.values.reshape(-1, 1)

            clf = IsolationForest(
                contamination=min(0.1, 100 / len(X)),
                random_state=42,
            )
            predictions = clf.fit_predict(X)

            anomaly_mask = predictions == -1
            anomaly_indices = clean.index[anomaly_mask].tolist()

            return AnomalyResult(
                method="isolation_forest",
                column=name,
                anomaly_count=len(anomaly_indices),
                anomaly_indices=anomaly_indices,
                anomaly_rate=len(anomaly_indices) / len(clean) if len(clean) > 0 else 0,
                details={"algorithm": "IsolationForest"},
            )

        except ImportError:
            self._logger.error("scikit-learn not available")
            return AnomalyResult(
                method="isolation_forest",
                column=name,
                anomaly_count=0,
                anomaly_indices=[],
                anomaly_rate=0.0,
                details={"error": "scikit-learn not installed"},
            )


def detect_lab_value_anomalies(
    df: pd.DataFrame,
    value_column: str = "result_value",
    reference_low: str = "reference_range_low",
    reference_high: str = "reference_range_high",
) -> AnomalyResult:
    """Detect lab values outside reference ranges.

    Args:
        df: Lab results DataFrame.
        value_column: Column with result values.
        reference_low: Column with low reference.
        reference_high: Column with high reference.

    Returns:
        AnomalyResult for out-of-range values.
    """
    logger.info("Detecting lab value anomalies")

    # Check for values outside reference ranges
    anomaly_mask = (df[value_column] < df[reference_low]) | (df[value_column] > df[reference_high])

    anomaly_indices = df.index[anomaly_mask].tolist()

    return AnomalyResult(
        method="reference_range",
        column=value_column,
        anomaly_count=len(anomaly_indices),
        anomaly_indices=anomaly_indices,
        anomaly_rate=len(anomaly_indices) / len(df) if len(df) > 0 else 0,
        details={
            "method": "Values outside reference range",
            "reference_columns": [reference_low, reference_high],
        },
    )
