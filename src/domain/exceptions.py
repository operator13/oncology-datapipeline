"""
Domain-specific exceptions for the Oncology Data Pipeline.

This module defines custom exceptions used throughout the application,
providing clear error messages and context for debugging.
"""

from typing import Any


class OncologyPipelineError(Exception):
    """Base exception for all oncology pipeline errors."""

    def __init__(self, message: str, context: dict[str, Any] | None = None) -> None:
        """Initialize the exception.

        Args:
            message: Human-readable error message.
            context: Optional dictionary with additional context.
        """
        self.message = message
        self.context = context or {}
        super().__init__(self._format_message())

    def _format_message(self) -> str:
        """Format the exception message with context."""
        if self.context:
            context_str = ", ".join(f"{k}={v}" for k, v in self.context.items())
            return f"{self.message} [{context_str}]"
        return self.message


# =============================================================================
# Connection Errors
# =============================================================================


class ConnectionError(OncologyPipelineError):
    """Base exception for connection-related errors."""

    pass


class ConnectionNotEstablishedError(ConnectionError):
    """Raised when attempting to use a connection that hasn't been established."""

    def __init__(self, connector_type: str) -> None:
        super().__init__(
            f"Connection not established. Call connect() first.",
            context={"connector_type": connector_type},
        )


class ConnectionFailedError(ConnectionError):
    """Raised when a connection attempt fails."""

    def __init__(
        self,
        connector_type: str,
        host: str,
        reason: str,
    ) -> None:
        super().__init__(
            f"Failed to connect to {connector_type}",
            context={"host": host, "reason": reason},
        )


class ConnectionTimeoutError(ConnectionError):
    """Raised when a connection times out."""

    def __init__(
        self,
        connector_type: str,
        host: str,
        timeout_seconds: int,
    ) -> None:
        super().__init__(
            f"Connection timed out after {timeout_seconds} seconds",
            context={"connector_type": connector_type, "host": host},
        )


class ConnectionPoolExhaustedError(ConnectionError):
    """Raised when the connection pool is exhausted."""

    def __init__(self, connector_type: str, pool_size: int) -> None:
        super().__init__(
            f"Connection pool exhausted",
            context={"connector_type": connector_type, "pool_size": pool_size},
        )


# =============================================================================
# Query Errors
# =============================================================================


class QueryError(OncologyPipelineError):
    """Base exception for query-related errors."""

    pass


class QueryExecutionError(QueryError):
    """Raised when a query fails to execute."""

    def __init__(self, query: str, reason: str) -> None:
        # Truncate query for display
        display_query = query[:200] + "..." if len(query) > 200 else query
        super().__init__(
            f"Query execution failed: {reason}",
            context={"query": display_query},
        )


class QueryTimeoutError(QueryError):
    """Raised when a query times out."""

    def __init__(self, query: str, timeout_seconds: int) -> None:
        display_query = query[:200] + "..." if len(query) > 200 else query
        super().__init__(
            f"Query timed out after {timeout_seconds} seconds",
            context={"query": display_query},
        )


class InvalidQueryError(QueryError):
    """Raised when a query is invalid."""

    def __init__(self, query: str, reason: str) -> None:
        display_query = query[:200] + "..." if len(query) > 200 else query
        super().__init__(
            f"Invalid query: {reason}",
            context={"query": display_query},
        )


# =============================================================================
# Configuration Errors
# =============================================================================


class ConfigurationError(OncologyPipelineError):
    """Base exception for configuration-related errors."""

    pass


class MissingConfigurationError(ConfigurationError):
    """Raised when required configuration is missing."""

    def __init__(self, config_key: str, source: str = "environment") -> None:
        super().__init__(
            f"Missing required configuration: {config_key}",
            context={"source": source},
        )


class InvalidConfigurationError(ConfigurationError):
    """Raised when configuration value is invalid."""

    def __init__(self, config_key: str, value: Any, reason: str) -> None:
        super().__init__(
            f"Invalid configuration for {config_key}: {reason}",
            context={"value": str(value)[:100]},
        )


# =============================================================================
# Validation Errors
# =============================================================================


class ValidationError(OncologyPipelineError):
    """Base exception for data validation errors."""

    pass


class DataValidationError(ValidationError):
    """Raised when data fails validation."""

    def __init__(
        self,
        message: str,
        field: str | None = None,
        value: Any | None = None,
        expectation: str | None = None,
    ) -> None:
        context: dict[str, Any] = {}
        if field:
            context["field"] = field
        if value is not None:
            context["value"] = str(value)[:100]
        if expectation:
            context["expectation"] = expectation
        super().__init__(message, context=context)


class SchemaValidationError(ValidationError):
    """Raised when data doesn't match expected schema."""

    def __init__(
        self,
        message: str,
        expected_columns: list[str] | None = None,
        actual_columns: list[str] | None = None,
    ) -> None:
        context: dict[str, Any] = {}
        if expected_columns:
            context["expected"] = expected_columns[:10]
        if actual_columns:
            context["actual"] = actual_columns[:10]
        super().__init__(message, context=context)


class ExpectationSuiteNotFoundError(ValidationError):
    """Raised when an expectation suite is not found."""

    def __init__(self, suite_name: str) -> None:
        super().__init__(
            f"Expectation suite not found: {suite_name}",
            context={"suite_name": suite_name},
        )


# =============================================================================
# Pipeline Errors
# =============================================================================


class PipelineError(OncologyPipelineError):
    """Base exception for pipeline-related errors."""

    pass


class PipelineExecutionError(PipelineError):
    """Raised when a pipeline fails during execution."""

    def __init__(self, pipeline_name: str, step: str, reason: str) -> None:
        super().__init__(
            f"Pipeline '{pipeline_name}' failed at step '{step}': {reason}",
            context={"pipeline": pipeline_name, "step": step},
        )


class PipelineConfigurationError(PipelineError):
    """Raised when pipeline configuration is invalid."""

    def __init__(self, pipeline_name: str, reason: str) -> None:
        super().__init__(
            f"Invalid pipeline configuration: {reason}",
            context={"pipeline": pipeline_name},
        )


# =============================================================================
# Data Generation Errors
# =============================================================================


class DataGenerationError(OncologyPipelineError):
    """Base exception for data generation errors."""

    pass


class InvalidGeneratorConfigError(DataGenerationError):
    """Raised when generator configuration is invalid."""

    def __init__(self, generator_name: str, reason: str) -> None:
        super().__init__(
            f"Invalid generator configuration: {reason}",
            context={"generator": generator_name},
        )


# =============================================================================
# Profiling Errors
# =============================================================================


class ProfilingError(OncologyPipelineError):
    """Base exception for data profiling errors."""

    pass


class ProfileGenerationError(ProfilingError):
    """Raised when profile generation fails."""

    def __init__(self, dataset: str, reason: str) -> None:
        super().__init__(
            f"Failed to generate profile for dataset: {reason}",
            context={"dataset": dataset},
        )


class AnomalyDetectionError(ProfilingError):
    """Raised when anomaly detection fails."""

    def __init__(self, method: str, reason: str) -> None:
        super().__init__(
            f"Anomaly detection failed using method '{method}': {reason}",
            context={"method": method},
        )
