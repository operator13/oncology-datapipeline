"""
Factory for creating database connectors.

This module provides a factory pattern implementation for creating
database connectors, supporting multiple backends with a unified interface.
"""

from enum import Enum
from typing import TypeVar

import structlog

from src.connectors.base import DataConnector
from src.connectors.databricks_connector import DatabricksConnector
from src.connectors.sqlserver_connector import SqlServerConnector
from src.domain.exceptions import InvalidConfigurationError
from src.utils.config import (
    DatabricksSettings,
    SqlServerSettings,
    get_databricks_settings,
    get_sqlserver_settings,
)

logger = structlog.get_logger(__name__)

T = TypeVar("T", bound=DataConnector)


class ConnectorType(str, Enum):
    """Enumeration of supported connector types."""

    DATABRICKS = "databricks"
    SQLSERVER = "sqlserver"


class ConnectionFactory:
    """Factory for creating database connectors.

    This factory provides a centralized way to create database connectors,
    supporting multiple backends and configuration options.

    Example:
        >>> # Create connector by type
        >>> connector = ConnectionFactory.create("databricks")
        >>> with connector:
        ...     df = connector.execute_query("SELECT 1")
        >>>
        >>> # Create with custom settings
        >>> settings = DatabricksSettings(host="...", token="...")
        >>> connector = ConnectionFactory.create_databricks(settings)
    """

    # Registry of connector types to their implementations
    _registry: dict[ConnectorType, type[DataConnector]] = {
        ConnectorType.DATABRICKS: DatabricksConnector,
        ConnectorType.SQLSERVER: SqlServerConnector,
    }

    @classmethod
    def create(
        cls,
        connector_type: str | ConnectorType,
        **kwargs: object,
    ) -> DataConnector:
        """Create a connector instance by type.

        Args:
            connector_type: Type of connector to create ('databricks' or 'sqlserver').
            **kwargs: Additional arguments passed to the connector constructor.

        Returns:
            Configured DataConnector instance.

        Raises:
            InvalidConfigurationError: If connector type is not supported.

        Example:
            >>> connector = ConnectionFactory.create("databricks")
            >>> connector = ConnectionFactory.create(ConnectorType.SQLSERVER)
        """
        # Normalize connector type
        if isinstance(connector_type, str):
            try:
                connector_type = ConnectorType(connector_type.lower())
            except ValueError:
                valid_types = [t.value for t in ConnectorType]
                raise InvalidConfigurationError(
                    config_key="connector_type",
                    value=connector_type,
                    reason=f"Must be one of: {valid_types}",
                )

        connector_class = cls._registry.get(connector_type)
        if not connector_class:
            raise InvalidConfigurationError(
                config_key="connector_type",
                value=connector_type.value,
                reason="Connector type not registered",
            )

        logger.info(
            "Creating connector",
            connector_type=connector_type.value,
        )

        return connector_class(**kwargs)

    @classmethod
    def create_databricks(
        cls,
        settings: DatabricksSettings | None = None,
    ) -> DatabricksConnector:
        """Create a Databricks connector.

        Args:
            settings: Optional settings override. Uses environment settings if None.

        Returns:
            Configured DatabricksConnector instance.

        Example:
            >>> connector = ConnectionFactory.create_databricks()
            >>> with connector:
            ...     df = connector.execute_query("SELECT * FROM oncology.patients")
        """
        settings = settings or get_databricks_settings()

        if not settings.is_configured:
            logger.warning(
                "Databricks not fully configured",
                has_host=bool(settings.host),
                has_http_path=bool(settings.http_path),
                has_token=bool(settings.token.get_secret_value()),
            )

        return DatabricksConnector(settings=settings)

    @classmethod
    def create_sqlserver(
        cls,
        settings: SqlServerSettings | None = None,
    ) -> SqlServerConnector:
        """Create a SQL Server connector.

        Args:
            settings: Optional settings override. Uses environment settings if None.

        Returns:
            Configured SqlServerConnector instance.

        Example:
            >>> connector = ConnectionFactory.create_sqlserver()
            >>> with connector:
            ...     df = connector.execute_query("SELECT TOP 10 * FROM patients")
        """
        settings = settings or get_sqlserver_settings()

        if not settings.is_configured:
            logger.warning(
                "SQL Server not fully configured",
                has_host=bool(settings.host),
                has_database=bool(settings.database),
                has_username=bool(settings.username),
                has_password=bool(settings.password.get_secret_value()),
            )

        return SqlServerConnector(settings=settings)

    @classmethod
    def register(
        cls,
        connector_type: ConnectorType,
        connector_class: type[DataConnector],
    ) -> None:
        """Register a new connector type.

        This allows extending the factory with custom connector implementations.

        Args:
            connector_type: Type identifier for the connector.
            connector_class: Connector class to register.

        Example:
            >>> class CustomConnector(DataConnector):
            ...     pass
            >>> ConnectionFactory.register(ConnectorType.CUSTOM, CustomConnector)
        """
        cls._registry[connector_type] = connector_class
        logger.info(
            "Registered connector",
            connector_type=connector_type.value,
            connector_class=connector_class.__name__,
        )

    @classmethod
    def get_available_connectors(cls) -> list[str]:
        """Get list of available connector types.

        Returns:
            List of connector type names.
        """
        return [t.value for t in cls._registry.keys()]

    @classmethod
    def test_connection(cls, connector_type: str | ConnectorType) -> bool:
        """Test if a connection can be established.

        Args:
            connector_type: Type of connector to test.

        Returns:
            True if connection succeeds, False otherwise.
        """
        try:
            connector = cls.create(connector_type)
            with connector:
                # Execute simple query to verify connection
                if connector.backend_name == "databricks":
                    connector.execute_query("SELECT 1")
                else:
                    connector.execute_query("SELECT 1 AS test")
            return True
        except Exception as e:
            logger.error(
                "Connection test failed",
                connector_type=str(connector_type),
                error=str(e),
            )
            return False


def get_connector(connector_type: str = "databricks") -> DataConnector:
    """Convenience function to get a connector instance.

    Args:
        connector_type: Type of connector ('databricks' or 'sqlserver').

    Returns:
        DataConnector instance.

    Example:
        >>> connector = get_connector("sqlserver")
        >>> with connector:
        ...     df = connector.execute_query("SELECT 1")
    """
    return ConnectionFactory.create(connector_type)
