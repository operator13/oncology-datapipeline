"""
Data profiler for oncology datasets.

This module provides statistical profiling capabilities for analyzing
data distributions, patterns, and quality characteristics.
"""

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import structlog

logger = structlog.get_logger(__name__)


@dataclass
class ColumnProfile:
    """Profile for a single column.

    Attributes:
        name: Column name.
        dtype: Data type.
        count: Total values.
        null_count: Number of null values.
        unique_count: Number of unique values.
        stats: Statistical measures.
        top_values: Most frequent values.
    """

    name: str
    dtype: str
    count: int
    null_count: int
    unique_count: int
    stats: dict[str, Any] = field(default_factory=dict)
    top_values: list[tuple[Any, int]] = field(default_factory=list)

    @property
    def null_rate(self) -> float:
        """Calculate null rate."""
        return self.null_count / self.count if self.count > 0 else 0.0

    @property
    def unique_rate(self) -> float:
        """Calculate uniqueness rate."""
        return self.unique_count / self.count if self.count > 0 else 0.0


@dataclass
class DataProfile:
    """Complete profile for a dataset.

    Attributes:
        name: Dataset name.
        row_count: Number of rows.
        column_count: Number of columns.
        profile_time: When profile was generated.
        columns: Per-column profiles.
        correlations: Column correlations (numeric).
        memory_usage: Memory usage in bytes.
    """

    name: str
    row_count: int
    column_count: int
    profile_time: datetime
    columns: dict[str, ColumnProfile]
    correlations: pd.DataFrame | None = None
    memory_usage: int = 0

    def summary(self) -> dict[str, Any]:
        """Generate summary statistics."""
        return {
            "name": self.name,
            "row_count": self.row_count,
            "column_count": self.column_count,
            "profile_time": self.profile_time.isoformat(),
            "memory_usage_mb": round(self.memory_usage / 1024 / 1024, 2),
            "columns_with_nulls": sum(1 for c in self.columns.values() if c.null_count > 0),
            "avg_null_rate": np.mean([c.null_rate for c in self.columns.values()]),
        }

    def to_dataframe(self) -> pd.DataFrame:
        """Convert column profiles to DataFrame."""
        data = []
        for col in self.columns.values():
            row = {
                "column": col.name,
                "dtype": col.dtype,
                "count": col.count,
                "null_count": col.null_count,
                "null_rate": col.null_rate,
                "unique_count": col.unique_count,
                "unique_rate": col.unique_rate,
            }
            row.update(col.stats)
            data.append(row)
        return pd.DataFrame(data)


class DataProfiler:
    """Profiler for generating dataset statistics.

    This class analyzes DataFrames to generate comprehensive
    statistical profiles including distributions, patterns,
    and data quality metrics.

    Example:
        >>> profiler = DataProfiler()
        >>> profile = profiler.profile(df, "patients")
        >>> print(f"Rows: {profile.row_count}")
        >>> print(profile.to_dataframe())
    """

    def __init__(self, sample_size: int | None = None) -> None:
        """Initialize the profiler.

        Args:
            sample_size: Optional sample size for large datasets.
        """
        self.sample_size = sample_size
        self._logger = logger.bind(profiler="data")

    def profile(
        self,
        df: pd.DataFrame,
        name: str = "dataset",
        include_correlations: bool = True,
    ) -> DataProfile:
        """Generate a profile for a DataFrame.

        Args:
            df: DataFrame to profile.
            name: Name for the dataset.
            include_correlations: Whether to compute correlations.

        Returns:
            DataProfile with statistics.
        """
        self._logger.info("Profiling dataset", name=name, rows=len(df), columns=len(df.columns))

        # Sample if needed
        if self.sample_size and len(df) > self.sample_size:
            df = df.sample(n=self.sample_size, random_state=42)
            self._logger.info("Sampled dataset", sample_size=self.sample_size)

        # Profile each column
        columns = {}
        for col in df.columns:
            columns[col] = self._profile_column(df[col])

        # Compute correlations for numeric columns
        correlations = None
        if include_correlations:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 1:
                correlations = df[numeric_cols].corr()

        profile = DataProfile(
            name=name,
            row_count=len(df),
            column_count=len(df.columns),
            profile_time=datetime.now(),
            columns=columns,
            correlations=correlations,
            memory_usage=df.memory_usage(deep=True).sum(),
        )

        self._logger.info("Profile complete", name=name)
        return profile

    def _profile_column(self, series: pd.Series) -> ColumnProfile:
        """Profile a single column.

        Args:
            series: Column data.

        Returns:
            ColumnProfile with statistics.
        """
        name = str(series.name)
        dtype = str(series.dtype)
        count = len(series)
        null_count = series.isna().sum()
        unique_count = series.nunique()

        # Compute statistics based on dtype
        stats: dict[str, Any] = {}
        if pd.api.types.is_numeric_dtype(series):
            stats = self._numeric_stats(series)
        elif pd.api.types.is_datetime64_any_dtype(series):
            stats = self._datetime_stats(series)
        elif pd.api.types.is_object_dtype(series):
            stats = self._string_stats(series)

        # Top values
        top_values = series.value_counts().head(5).items()

        return ColumnProfile(
            name=name,
            dtype=dtype,
            count=count,
            null_count=int(null_count),
            unique_count=int(unique_count),
            stats=stats,
            top_values=[(str(k), int(v)) for k, v in top_values],
        )

    def _numeric_stats(self, series: pd.Series) -> dict[str, Any]:
        """Compute numeric column statistics."""
        clean = series.dropna()
        if len(clean) == 0:
            return {}

        return {
            "mean": float(clean.mean()),
            "std": float(clean.std()),
            "min": float(clean.min()),
            "25%": float(clean.quantile(0.25)),
            "50%": float(clean.quantile(0.50)),
            "75%": float(clean.quantile(0.75)),
            "max": float(clean.max()),
            "skewness": float(clean.skew()),
            "kurtosis": float(clean.kurtosis()),
        }

    def _datetime_stats(self, series: pd.Series) -> dict[str, Any]:
        """Compute datetime column statistics."""
        clean = series.dropna()
        if len(clean) == 0:
            return {}

        return {
            "min": str(clean.min()),
            "max": str(clean.max()),
            "range_days": (clean.max() - clean.min()).days,
        }

    def _string_stats(self, series: pd.Series) -> dict[str, Any]:
        """Compute string column statistics."""
        clean = series.dropna().astype(str)
        if len(clean) == 0:
            return {}

        lengths = clean.str.len()
        return {
            "avg_length": float(lengths.mean()),
            "min_length": int(lengths.min()),
            "max_length": int(lengths.max()),
        }

    def generate_report(
        self,
        profile: DataProfile,
        output_path: Path | str | None = None,
    ) -> str:
        """Generate HTML profile report.

        Args:
            profile: DataProfile to report.
            output_path: Optional path to save report.

        Returns:
            HTML report string.
        """
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Data Profile: {profile.name}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        table {{ border-collapse: collapse; width: 100%; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #4CAF50; color: white; }}
        .summary {{ background-color: #f9f9f9; padding: 15px; margin-bottom: 20px; }}
    </style>
</head>
<body>
    <h1>Data Profile: {profile.name}</h1>
    <div class="summary">
        <p><strong>Rows:</strong> {profile.row_count:,}</p>
        <p><strong>Columns:</strong> {profile.column_count}</p>
        <p><strong>Generated:</strong> {profile.profile_time}</p>
        <p><strong>Memory:</strong> {profile.memory_usage / 1024 / 1024:.2f} MB</p>
    </div>

    <h2>Column Profiles</h2>
    {profile.to_dataframe().to_html(index=False)}
</body>
</html>
"""
        if output_path:
            Path(output_path).write_text(html)
            self._logger.info("Report saved", path=str(output_path))

        return html
