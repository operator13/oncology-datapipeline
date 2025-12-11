"""
Database connectors for the Oncology Data Pipeline.

This package provides connectors for various database backends,
implementing a consistent interface for data access operations.

Supported Backends:
    - Databricks SQL (Delta Lake)
    - Microsoft SQL Server

Example:
    >>> from src.connectors import ConnectionFactory, get_connector
    >>>
    >>> # Using the factory
    >>> connector = ConnectionFactory.create("databricks")
    >>> with connector:
    ...     df = connector.execute_query("SELECT * FROM patients")
    >>>
    >>> # Using convenience function
    >>> with get_connector("sqlserver") as conn:
    ...     df = conn.execute_query("SELECT TOP 10 * FROM patients")
"""

from src.connectors.base import BatchConnector, DataConnector, TransactionalConnector
from src.connectors.connection_factory import ConnectionFactory, ConnectorType, get_connector
from src.connectors.databricks_connector import DatabricksConnector
from src.connectors.sqlserver_connector import SqlServerConnector

__all__ = [
    # Base classes
    "DataConnector",
    "TransactionalConnector",
    "BatchConnector",
    # Implementations
    "DatabricksConnector",
    "SqlServerConnector",
    # Factory
    "ConnectionFactory",
    "ConnectorType",
    "get_connector",
]
