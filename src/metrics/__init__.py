"""
Quality metrics and reporting for oncology datasets.

This package provides tools for calculating and tracking
data quality metrics across multiple dimensions.

Example:
    >>> from src.metrics import QualityScorecardCalculator
    >>>
    >>> calculator = QualityScorecardCalculator()
    >>> scorecard = calculator.calculate(df, "patients")
    >>> print(scorecard.summary())
"""

from src.metrics.quality_scorecard import (
    DimensionScore,
    QualityDimension,
    QualityScorecard,
    QualityScorecardCalculator,
)

__all__ = [
    "QualityScorecardCalculator",
    "QualityScorecard",
    "QualityDimension",
    "DimensionScore",
]
