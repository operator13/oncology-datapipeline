"""
Domain layer for the Oncology Data Pipeline.

This module contains domain entities, value objects, and domain-specific
exceptions that form the core business logic of the application.
"""

from src.domain.exceptions import (
    AnomalyDetectionError,
    ConfigurationError,
    ConnectionError,
    ConnectionFailedError,
    ConnectionNotEstablishedError,
    ConnectionPoolExhaustedError,
    ConnectionTimeoutError,
    DataGenerationError,
    DataValidationError,
    ExpectationSuiteNotFoundError,
    InvalidConfigurationError,
    InvalidGeneratorConfigError,
    InvalidQueryError,
    MissingConfigurationError,
    OncologyPipelineError,
    PipelineConfigurationError,
    PipelineError,
    PipelineExecutionError,
    ProfileGenerationError,
    ProfilingError,
    QueryError,
    QueryExecutionError,
    QueryTimeoutError,
    SchemaValidationError,
    ValidationError,
)

__all__ = [
    # Base
    "OncologyPipelineError",
    # Connection
    "ConnectionError",
    "ConnectionNotEstablishedError",
    "ConnectionFailedError",
    "ConnectionTimeoutError",
    "ConnectionPoolExhaustedError",
    # Query
    "QueryError",
    "QueryExecutionError",
    "QueryTimeoutError",
    "InvalidQueryError",
    # Configuration
    "ConfigurationError",
    "MissingConfigurationError",
    "InvalidConfigurationError",
    # Validation
    "ValidationError",
    "DataValidationError",
    "SchemaValidationError",
    "ExpectationSuiteNotFoundError",
    # Pipeline
    "PipelineError",
    "PipelineExecutionError",
    "PipelineConfigurationError",
    # Data Generation
    "DataGenerationError",
    "InvalidGeneratorConfigError",
    # Profiling
    "ProfilingError",
    "ProfileGenerationError",
    "AnomalyDetectionError",
]
