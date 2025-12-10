"""
Base pipeline class for oncology data pipelines.

This module provides the abstract base class and common utilities
for building data pipelines using the Template Method pattern.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Generic, TypeVar

import pandas as pd
import structlog

logger = structlog.get_logger(__name__)

T = TypeVar("T")


class PipelineStatus(str, Enum):
    """Pipeline execution status."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class PipelineResult:
    """Result of pipeline execution.

    Attributes:
        pipeline_name: Name of the pipeline.
        status: Execution status.
        start_time: When execution started.
        end_time: When execution ended.
        records_processed: Number of records processed.
        records_failed: Number of failed records.
        errors: List of error messages.
        metadata: Additional execution metadata.
    """

    pipeline_name: str
    status: PipelineStatus
    start_time: datetime
    end_time: datetime | None = None
    records_processed: int = 0
    records_failed: int = 0
    errors: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def duration_seconds(self) -> float | None:
        """Calculate execution duration."""
        if self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return None

    @property
    def success_rate(self) -> float:
        """Calculate record success rate."""
        total = self.records_processed + self.records_failed
        return self.records_processed / total if total > 0 else 0.0

    def summary(self) -> str:
        """Generate execution summary."""
        duration = f"{self.duration_seconds:.2f}s" if self.duration_seconds else "N/A"
        return (
            f"Pipeline: {self.pipeline_name}\n"
            f"Status: {self.status.value}\n"
            f"Duration: {duration}\n"
            f"Processed: {self.records_processed}\n"
            f"Failed: {self.records_failed}\n"
            f"Success Rate: {self.success_rate:.1%}"
        )


class BasePipeline(ABC, Generic[T]):
    """Abstract base class for data pipelines.

    This class implements the Template Method pattern for building
    data pipelines with standardized steps.

    Subclasses must implement:
        - extract(): Get data from source
        - transform(): Transform the data
        - load(): Load data to destination
        - validate(): Validate data quality

    Example:
        >>> class MyPipeline(BasePipeline):
        ...     def extract(self) -> pd.DataFrame:
        ...         return pd.read_csv("data.csv")
        ...     def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        ...         return data.dropna()
        ...     def load(self, data: pd.DataFrame) -> None:
        ...         data.to_parquet("output.parquet")
    """

    def __init__(self, name: str) -> None:
        """Initialize the pipeline.

        Args:
            name: Pipeline name for logging and tracking.
        """
        self.name = name
        self._logger = logger.bind(pipeline=name)
        self._result: PipelineResult | None = None

    @abstractmethod
    def extract(self) -> T:
        """Extract data from source.

        Returns:
            Extracted data.
        """
        ...

    @abstractmethod
    def transform(self, data: T) -> T:
        """Transform the extracted data.

        Args:
            data: Data to transform.

        Returns:
            Transformed data.
        """
        ...

    @abstractmethod
    def load(self, data: T) -> None:
        """Load data to destination.

        Args:
            data: Data to load.
        """
        ...

    def validate(self, data: T) -> bool:
        """Validate data quality.

        Override this method to add custom validation.

        Args:
            data: Data to validate.

        Returns:
            True if validation passes.
        """
        return True

    def pre_execute(self) -> None:
        """Hook called before pipeline execution.

        Override for custom pre-execution logic.
        """
        self._logger.info("Starting pipeline execution")

    def post_execute(self, result: PipelineResult) -> None:
        """Hook called after pipeline execution.

        Override for custom post-execution logic.

        Args:
            result: Pipeline execution result.
        """
        self._logger.info(
            "Pipeline execution complete",
            status=result.status.value,
            duration=result.duration_seconds,
        )

    def on_error(self, error: Exception, step: str) -> None:
        """Hook called when an error occurs.

        Override for custom error handling.

        Args:
            error: The exception that occurred.
            step: The step where error occurred.
        """
        self._logger.error(f"Pipeline failed at {step}", error=str(error))

    def run(self) -> PipelineResult:
        """Execute the complete pipeline.

        This method orchestrates the ETL process:
        1. pre_execute hook
        2. extract
        3. validate (optional)
        4. transform
        5. validate (optional)
        6. load
        7. post_execute hook

        Returns:
            PipelineResult with execution details.
        """
        result = PipelineResult(
            pipeline_name=self.name,
            status=PipelineStatus.RUNNING,
            start_time=datetime.now(),
        )

        try:
            self.pre_execute()

            # Extract
            self._logger.info("Extracting data")
            data = self.extract()
            if isinstance(data, pd.DataFrame):
                result.metadata["extracted_rows"] = len(data)

            # Validate extracted data
            if not self.validate(data):
                raise ValueError("Extracted data validation failed")

            # Transform
            self._logger.info("Transforming data")
            transformed = self.transform(data)
            if isinstance(transformed, pd.DataFrame):
                result.metadata["transformed_rows"] = len(transformed)

            # Validate transformed data
            if not self.validate(transformed):
                raise ValueError("Transformed data validation failed")

            # Load
            self._logger.info("Loading data")
            self.load(transformed)

            # Success
            if isinstance(transformed, pd.DataFrame):
                result.records_processed = len(transformed)

            result.status = PipelineStatus.COMPLETED
            result.end_time = datetime.now()

        except Exception as e:
            result.status = PipelineStatus.FAILED
            result.errors.append(str(e))
            result.end_time = datetime.now()
            self.on_error(e, "unknown")

        finally:
            self.post_execute(result)
            self._result = result

        return result

    @property
    def last_result(self) -> PipelineResult | None:
        """Get the last execution result."""
        return self._result


class DataFramePipeline(BasePipeline[pd.DataFrame]):
    """Base pipeline specifically for DataFrame processing.

    Provides additional utilities for DataFrame-based ETL operations.
    """

    def __init__(self, name: str) -> None:
        super().__init__(name)
        self._schema: dict[str, str] | None = None

    def set_expected_schema(self, schema: dict[str, str]) -> None:
        """Set expected DataFrame schema for validation.

        Args:
            schema: Dictionary mapping column names to dtypes.
        """
        self._schema = schema

    def validate(self, data: pd.DataFrame) -> bool:
        """Validate DataFrame against expected schema.

        Args:
            data: DataFrame to validate.

        Returns:
            True if validation passes.
        """
        if self._schema is None:
            return True

        # Check for required columns
        missing = set(self._schema.keys()) - set(data.columns)
        if missing:
            self._logger.warning("Missing columns", columns=list(missing))
            return False

        return True

    def add_metadata_columns(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add standard metadata columns to DataFrame.

        Args:
            data: Input DataFrame.

        Returns:
            DataFrame with metadata columns.
        """
        data = data.copy()
        data["_pipeline_name"] = self.name
        data["_processed_at"] = datetime.now()
        return data
