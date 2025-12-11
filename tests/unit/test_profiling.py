"""
Unit tests for profiling module.
"""

import numpy as np
import pandas as pd
import pytest

from src.profiling import AnomalyDetector, AnomalyMethod, DataProfiler, DriftDetector


class TestDataProfiler:
    """Tests for DataProfiler."""

    @pytest.mark.unit
    def test_profile_basic(self):
        """Test basic profiling."""
        df = pd.DataFrame(
            {
                "id": [1, 2, 3, 4, 5],
                "name": ["Alice", "Bob", "Charlie", "Diana", "Eve"],
                "value": [100.0, 200.0, 300.0, 400.0, 500.0],
            }
        )

        profiler = DataProfiler()
        profile = profiler.profile(df, "test_data")

        assert profile.name == "test_data"
        assert profile.row_count == 5
        assert profile.column_count == 3
        assert "id" in profile.columns
        assert "name" in profile.columns
        assert "value" in profile.columns

    @pytest.mark.unit
    def test_profile_numeric_stats(self):
        """Test numeric column statistics."""
        df = pd.DataFrame({"value": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]})

        profiler = DataProfiler()
        profile = profiler.profile(df, "test")

        col_profile = profile.columns["value"]
        assert col_profile.stats["mean"] == 5.5
        assert col_profile.stats["min"] == 1
        assert col_profile.stats["max"] == 10

    @pytest.mark.unit
    def test_profile_with_nulls(self, df_with_nulls):
        """Test profiling with null values."""
        profiler = DataProfiler()
        profile = profiler.profile(df_with_nulls, "test")

        name_profile = profile.columns["name"]
        assert name_profile.null_count == 2
        assert name_profile.null_rate == 0.4

    @pytest.mark.unit
    def test_profile_to_dataframe(self):
        """Test converting profile to DataFrame."""
        df = pd.DataFrame(
            {
                "a": [1, 2, 3],
                "b": ["x", "y", "z"],
            }
        )

        profiler = DataProfiler()
        profile = profiler.profile(df, "test")
        profile_df = profile.to_dataframe()

        assert len(profile_df) == 2
        assert "column" in profile_df.columns
        assert "null_rate" in profile_df.columns


class TestAnomalyDetector:
    """Tests for AnomalyDetector."""

    @pytest.mark.unit
    def test_detect_iqr(self, df_with_outliers):
        """Test IQR anomaly detection."""
        detector = AnomalyDetector(method=AnomalyMethod.IQR)
        results = detector.detect(df_with_outliers, columns=["with_outliers"])

        assert "with_outliers" in results
        result = results["with_outliers"]
        assert result.anomaly_count > 0
        assert result.method == "iqr"

    @pytest.mark.unit
    def test_detect_zscore(self, df_with_outliers):
        """Test Z-score anomaly detection."""
        detector = AnomalyDetector(method=AnomalyMethod.ZSCORE, threshold=3.0)
        results = detector.detect(df_with_outliers, columns=["with_outliers"])

        result = results["with_outliers"]
        assert result.method == "zscore"
        assert "mean" in result.details
        assert "std" in result.details

    @pytest.mark.unit
    def test_detect_mad(self, df_with_outliers):
        """Test MAD anomaly detection."""
        detector = AnomalyDetector(method=AnomalyMethod.MAD, threshold=3.5)
        results = detector.detect(df_with_outliers, columns=["with_outliers"])

        result = results["with_outliers"]
        assert result.method == "mad"
        assert "median" in result.details

    @pytest.mark.unit
    def test_no_anomalies_in_normal_data(self):
        """Test that normal data has few/no anomalies."""
        df = pd.DataFrame({"value": np.random.normal(50, 5, 1000)})

        detector = AnomalyDetector(method=AnomalyMethod.IQR, threshold=1.5)
        results = detector.detect(df, columns=["value"])

        # Should have very few anomalies in normally distributed data
        assert results["value"].anomaly_rate < 0.1

    @pytest.mark.unit
    def test_anomaly_result_summary(self, df_with_outliers):
        """Test AnomalyResult summary generation."""
        detector = AnomalyDetector()
        results = detector.detect(df_with_outliers, columns=["with_outliers"])

        summary = results["with_outliers"].summary()
        assert "anomalies" in summary.lower()
        assert "with_outliers" in summary


class TestDriftDetector:
    """Tests for DriftDetector."""

    @pytest.mark.unit
    def test_detect_schema_drift_no_change(self):
        """Test schema drift with no changes."""
        df1 = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        df2 = pd.DataFrame({"a": [5, 6], "b": [7, 8]})

        detector = DriftDetector()
        result = detector.detect_schema_drift(df1, df2)

        assert not result.drift_detected
        assert result.drift_score == 0

    @pytest.mark.unit
    def test_detect_schema_drift_added_column(self):
        """Test schema drift with added column."""
        df1 = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        df2 = pd.DataFrame({"a": [1, 2], "b": [3, 4], "c": [5, 6]})

        detector = DriftDetector()
        result = detector.detect_schema_drift(df1, df2)

        assert result.drift_detected
        assert "c" in result.details["columns_added"]

    @pytest.mark.unit
    def test_detect_schema_drift_removed_column(self):
        """Test schema drift with removed column."""
        df1 = pd.DataFrame({"a": [1, 2], "b": [3, 4], "c": [5, 6]})
        df2 = pd.DataFrame({"a": [1, 2], "b": [3, 4]})

        detector = DriftDetector()
        result = detector.detect_schema_drift(df1, df2)

        assert result.drift_detected
        assert "c" in result.details["columns_removed"]

    @pytest.mark.unit
    def test_detect_distribution_drift(self):
        """Test distribution drift detection."""
        # Create two distributions with different means
        np.random.seed(42)
        df1 = pd.DataFrame({"value": np.random.normal(50, 5, 1000)})
        df2 = pd.DataFrame({"value": np.random.normal(70, 5, 1000)})  # Shifted mean

        detector = DriftDetector(threshold=0.05)
        results = detector.detect_distribution_drift(df1, df2, columns=["value"])

        result = results["value"]
        assert result.drift_detected  # Should detect the shift

    @pytest.mark.unit
    def test_detect_no_distribution_drift(self):
        """Test no drift in similar distributions."""
        np.random.seed(42)
        df1 = pd.DataFrame({"value": np.random.normal(50, 5, 1000)})
        np.random.seed(43)
        df2 = pd.DataFrame({"value": np.random.normal(50, 5, 1000)})

        detector = DriftDetector(threshold=0.05)
        results = detector.detect_distribution_drift(df1, df2, columns=["value"])

        # Similar distributions should not show significant drift
        # (may vary due to randomness)
        assert results["value"].drift_score < 0.2
