"""
Data quality scorecard for oncology datasets.

This module provides quality scoring based on multiple dimensions:
completeness, accuracy, consistency, timeliness, and validity.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

import pandas as pd
import structlog

logger = structlog.get_logger(__name__)


class QualityDimension(str, Enum):
    """Data quality dimensions."""

    COMPLETENESS = "completeness"
    ACCURACY = "accuracy"
    CONSISTENCY = "consistency"
    TIMELINESS = "timeliness"
    VALIDITY = "validity"
    UNIQUENESS = "uniqueness"


@dataclass
class DimensionScore:
    """Score for a single quality dimension.

    Attributes:
        dimension: Quality dimension name.
        score: Score from 0.0 to 1.0.
        weight: Weight for overall calculation.
        details: Calculation details.
    """

    dimension: QualityDimension
    score: float
    weight: float = 1.0
    details: dict[str, Any] = field(default_factory=dict)

    @property
    def weighted_score(self) -> float:
        """Calculate weighted score."""
        return self.score * self.weight

    @property
    def percentage(self) -> float:
        """Get score as percentage."""
        return self.score * 100


@dataclass
class QualityScorecard:
    """Complete quality scorecard for a dataset.

    Attributes:
        dataset_name: Name of the dataset.
        scores: Scores per dimension.
        overall_score: Weighted overall score.
        grade: Letter grade (A-F).
        assessment_time: When assessment was performed.
        row_count: Number of rows assessed.
    """

    dataset_name: str
    scores: dict[QualityDimension, DimensionScore]
    overall_score: float
    grade: str
    assessment_time: datetime
    row_count: int

    def to_dict(self) -> dict[str, Any]:
        """Convert scorecard to dictionary."""
        return {
            "dataset_name": self.dataset_name,
            "overall_score": self.overall_score,
            "overall_percentage": self.overall_score * 100,
            "grade": self.grade,
            "assessment_time": self.assessment_time.isoformat(),
            "row_count": self.row_count,
            "dimensions": {
                dim.value: {
                    "score": score.score,
                    "percentage": score.percentage,
                    "weight": score.weight,
                    "details": score.details,
                }
                for dim, score in self.scores.items()
            },
        }

    def summary(self) -> str:
        """Generate summary string."""
        lines = [
            "=" * 50,
            f"DATA QUALITY SCORECARD: {self.dataset_name}",
            "=" * 50,
            f"Overall Score: {self.overall_score * 100:.1f}% (Grade: {self.grade})",
            f"Rows Assessed: {self.row_count:,}",
            f"Assessment Time: {self.assessment_time}",
            "",
            "Dimension Scores:",
        ]

        for dim, score in self.scores.items():
            bar = "█" * int(score.score * 20) + "░" * (20 - int(score.score * 20))
            lines.append(f"  {dim.value:15} [{bar}] {score.percentage:.1f}%")

        return "\n".join(lines)


class QualityScorecardCalculator:
    """Calculator for data quality scorecards.

    This class calculates quality scores across multiple dimensions
    to produce a comprehensive quality assessment.

    Example:
        >>> calculator = QualityScorecardCalculator()
        >>> scorecard = calculator.calculate(df, "patients")
        >>> print(scorecard.summary())
    """

    # Default dimension weights
    DEFAULT_WEIGHTS: dict[QualityDimension, float] = {
        QualityDimension.COMPLETENESS: 0.25,
        QualityDimension.ACCURACY: 0.20,
        QualityDimension.CONSISTENCY: 0.20,
        QualityDimension.TIMELINESS: 0.15,
        QualityDimension.VALIDITY: 0.15,
        QualityDimension.UNIQUENESS: 0.05,
    }

    # Grade thresholds
    GRADE_THRESHOLDS: dict[str, float] = {
        "A": 0.95,
        "B": 0.85,
        "C": 0.75,
        "D": 0.65,
        "F": 0.0,
    }

    def __init__(
        self,
        weights: dict[QualityDimension, float] | None = None,
    ) -> None:
        """Initialize the calculator.

        Args:
            weights: Optional custom dimension weights.
        """
        self.weights = weights or self.DEFAULT_WEIGHTS
        self._logger = logger.bind(calculator="scorecard")

    def calculate(
        self,
        df: pd.DataFrame,
        dataset_name: str,
        required_columns: list[str] | None = None,
        unique_columns: list[str] | None = None,
        date_columns: list[str] | None = None,
    ) -> QualityScorecard:
        """Calculate quality scorecard for a DataFrame.

        Args:
            df: DataFrame to assess.
            dataset_name: Name for the dataset.
            required_columns: Columns that should not have nulls.
            unique_columns: Columns that should be unique.
            date_columns: Date columns for timeliness checks.

        Returns:
            QualityScorecard with dimension scores.
        """
        self._logger.info(
            "Calculating quality scorecard",
            dataset=dataset_name,
            rows=len(df),
        )

        scores: dict[QualityDimension, DimensionScore] = {}

        # Calculate each dimension
        scores[QualityDimension.COMPLETENESS] = self._calculate_completeness(df, required_columns)
        scores[QualityDimension.UNIQUENESS] = self._calculate_uniqueness(df, unique_columns)
        scores[QualityDimension.VALIDITY] = self._calculate_validity(df)
        scores[QualityDimension.CONSISTENCY] = self._calculate_consistency(df)
        scores[QualityDimension.TIMELINESS] = self._calculate_timeliness(df, date_columns)
        scores[QualityDimension.ACCURACY] = self._calculate_accuracy(df)

        # Apply weights
        for dim, score in scores.items():
            score.weight = self.weights.get(dim, 1.0)

        # Calculate overall score
        total_weight = sum(s.weight for s in scores.values())
        overall_score = sum(s.weighted_score for s in scores.values()) / total_weight

        # Determine grade
        grade = self._determine_grade(overall_score)

        scorecard = QualityScorecard(
            dataset_name=dataset_name,
            scores=scores,
            overall_score=overall_score,
            grade=grade,
            assessment_time=datetime.now(),
            row_count=len(df),
        )

        self._logger.info(
            "Scorecard calculated",
            overall_score=f"{overall_score:.2%}",
            grade=grade,
        )

        return scorecard

    def _calculate_completeness(
        self,
        df: pd.DataFrame,
        required_columns: list[str] | None,
    ) -> DimensionScore:
        """Calculate completeness score."""
        if required_columns:
            cols_to_check = [c for c in required_columns if c in df.columns]
        else:
            cols_to_check = df.columns.tolist()

        if not cols_to_check:
            return DimensionScore(
                dimension=QualityDimension.COMPLETENESS,
                score=1.0,
                details={"columns_checked": 0},
            )

        completeness_rates = []
        for col in cols_to_check:
            null_rate = df[col].isna().mean()
            completeness_rates.append(1 - null_rate)

        score = sum(completeness_rates) / len(completeness_rates)

        return DimensionScore(
            dimension=QualityDimension.COMPLETENESS,
            score=score,
            details={
                "columns_checked": len(cols_to_check),
                "avg_completeness": score,
            },
        )

    def _calculate_uniqueness(
        self,
        df: pd.DataFrame,
        unique_columns: list[str] | None,
    ) -> DimensionScore:
        """Calculate uniqueness score."""
        if not unique_columns:
            # Check for common unique column names
            unique_columns = [
                c for c in df.columns if any(x in c.lower() for x in ["id", "mrn", "code"])
            ]

        if not unique_columns:
            return DimensionScore(
                dimension=QualityDimension.UNIQUENESS,
                score=1.0,
                details={"columns_checked": 0},
            )

        uniqueness_rates = []
        for col in unique_columns:
            if col in df.columns:
                unique_rate = df[col].nunique() / len(df) if len(df) > 0 else 1.0
                uniqueness_rates.append(unique_rate)

        score = sum(uniqueness_rates) / len(uniqueness_rates) if uniqueness_rates else 1.0

        return DimensionScore(
            dimension=QualityDimension.UNIQUENESS,
            score=score,
            details={
                "columns_checked": len(uniqueness_rates),
                "avg_uniqueness": score,
            },
        )

    def _calculate_validity(self, df: pd.DataFrame) -> DimensionScore:
        """Calculate validity score based on data types."""
        # Check for valid data in each column
        valid_rates = []

        for col in df.columns:
            # Check for unexpected values
            if df[col].dtype == "object":
                # String columns - check for empty strings
                non_empty = (df[col].astype(str).str.strip() != "").mean()
                valid_rates.append(non_empty)
            else:
                # Numeric - check for finite values
                if pd.api.types.is_numeric_dtype(df[col]):
                    finite_rate = (
                        df[col]
                        .apply(
                            lambda x: pd.notna(x) and not (pd.notna(x) and abs(x) == float("inf"))
                        )
                        .mean()
                    )
                    valid_rates.append(finite_rate)
                else:
                    valid_rates.append(1.0)

        score = sum(valid_rates) / len(valid_rates) if valid_rates else 1.0

        return DimensionScore(
            dimension=QualityDimension.VALIDITY,
            score=score,
            details={"columns_checked": len(valid_rates)},
        )

    def _calculate_consistency(self, df: pd.DataFrame) -> DimensionScore:
        """Calculate consistency score."""
        # Check for consistent formatting
        consistency_checks = []

        for col in df.columns:
            if df[col].dtype == "object":
                # Check case consistency
                values = df[col].dropna().astype(str)
                if len(values) > 0:
                    upper_ratio = (values == values.str.upper()).mean()
                    lower_ratio = (values == values.str.lower()).mean()
                    title_ratio = (values == values.str.title()).mean()
                    max_consistency = max(upper_ratio, lower_ratio, title_ratio)
                    consistency_checks.append(max_consistency)

        score = sum(consistency_checks) / len(consistency_checks) if consistency_checks else 1.0

        return DimensionScore(
            dimension=QualityDimension.CONSISTENCY,
            score=score,
            details={"string_columns_checked": len(consistency_checks)},
        )

    def _calculate_timeliness(
        self,
        df: pd.DataFrame,
        date_columns: list[str] | None,
    ) -> DimensionScore:
        """Calculate timeliness score."""
        if not date_columns:
            # Auto-detect date columns
            date_columns = [
                c
                for c in df.columns
                if any(x in c.lower() for x in ["date", "time", "created", "updated"])
            ]

        if not date_columns:
            return DimensionScore(
                dimension=QualityDimension.TIMELINESS,
                score=1.0,
                details={"columns_checked": 0},
            )

        timeliness_scores = []
        now = datetime.now()

        for col in date_columns:
            if col not in df.columns:
                continue

            try:
                dates = pd.to_datetime(df[col], errors="coerce")
                valid_dates = dates.dropna()

                if len(valid_dates) > 0:
                    # Check for future dates (invalid)
                    future_rate = (valid_dates > now).mean()
                    # Check for very old dates (suspicious)
                    very_old = (valid_dates < datetime(2000, 1, 1)).mean()

                    timeliness_scores.append(1 - future_rate - very_old)
            except Exception:
                pass

        score = sum(timeliness_scores) / len(timeliness_scores) if timeliness_scores else 1.0
        score = max(0, min(1, score))  # Clamp to [0, 1]

        return DimensionScore(
            dimension=QualityDimension.TIMELINESS,
            score=score,
            details={"date_columns_checked": len(timeliness_scores)},
        )

    def _calculate_accuracy(self, df: pd.DataFrame) -> DimensionScore:
        """Calculate accuracy score (proxy metrics)."""
        # Use range checks as proxy for accuracy
        accuracy_scores = []

        for col in df.select_dtypes(include=["number"]).columns:
            values = df[col].dropna()
            if len(values) > 0:
                # Check for values within reasonable range (no extreme outliers)
                q1, q3 = values.quantile([0.25, 0.75])
                iqr = q3 - q1
                lower = q1 - 3 * iqr
                upper = q3 + 3 * iqr
                in_range = ((values >= lower) & (values <= upper)).mean()
                accuracy_scores.append(in_range)

        score = sum(accuracy_scores) / len(accuracy_scores) if accuracy_scores else 1.0

        return DimensionScore(
            dimension=QualityDimension.ACCURACY,
            score=score,
            details={"numeric_columns_checked": len(accuracy_scores)},
        )

    def _determine_grade(self, score: float) -> str:
        """Determine letter grade from score."""
        for grade, threshold in self.GRADE_THRESHOLDS.items():
            if score >= threshold:
                return grade
        return "F"
