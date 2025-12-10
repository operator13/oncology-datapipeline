"""
Oncology Data Pipeline - Data Quality Automation Framework.

A production-grade framework for automating data quality checks,
profiling, and validation for oncology data pipelines. Built with
Great Expectations, supporting both Databricks and SQL Server.

Features:
    - Great Expectations integration for data validation
    - Multi-platform connectors (Databricks, SQL Server)
    - Synthetic oncology data generation
    - Data profiling and anomaly detection
    - Quality metrics and dashboards
    - CI/CD integration with GitHub Actions

Example:
    >>> from src.data_quality import ValidationRunner
    >>> from src.connectors import ConnectionFactory
    >>>
    >>> connector = ConnectionFactory.create("databricks")
    >>> runner = ValidationRunner(connector)
    >>> results = runner.validate("patients")
    >>> print(f"Validation passed: {results.success}")
"""

__version__ = "0.1.0"
__author__ = "Data Quality Engineer"
__email__ = "dq-engineer@example.com"

from typing import Final

# Package metadata
PACKAGE_NAME: Final[str] = "oncology-datapipeline"
DESCRIPTION: Final[str] = "Data Quality Automation Framework for Oncology Data Pipeline"

# Supported data domains
SUPPORTED_DOMAINS: Final[tuple[str, ...]] = (
    "patients",
    "treatments",
    "lab_results",
)

# Supported database backends
SUPPORTED_BACKENDS: Final[tuple[str, ...]] = (
    "databricks",
    "sqlserver",
)

__all__ = [
    "__version__",
    "__author__",
    "__email__",
    "PACKAGE_NAME",
    "DESCRIPTION",
    "SUPPORTED_DOMAINS",
    "SUPPORTED_BACKENDS",
]
