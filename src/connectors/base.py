"""
Abstract base class for database connectors.

This module defines the interface that all database connectors must implement,
ensuring consistent behavior across different backends (Databricks, SQL Server, etc.).
"""

from abc import ABC, abstractmethod
from contextlib import contextmanager
from typing import Any, Generator, Iterator, Protocol, TypeVar

import pandas as pd
import structlog

from src.domain.exceptions import ConnectionNotEstablishedError

logger = structlog.get_logger(__name__)

# Type variable for generic connector operations
T = TypeVar("T", bound="DataConnector")


class QueryResult(Protocol):
    """Protocol for query results that can be converted to DataFrame."""

    def fetchall(self) -> list[tuple[Any, ...]]:
        """Fetch all rows from the result."""
        ...

    def fetchone(self) -> tuple[Any, ...] | None:
        """Fetch a single row from the result."""
        ...

    def fetchmany(self, size: int) -> list[tuple[Any, ...]]:
        """Fetch specified number of rows."""
        ...


class DataConnector(ABC):
    """Abstract base class for all data connectors.

    This class defines the interface for database connections, providing
    a consistent API for connecting, querying, and managing database
    resources across different backends.

    Attributes:
        _is_connected: Boolean indicating if connection is established.
        _connection: The underlying database connection object.

    Example:
        >>> class MyConnector(DataConnector):
        ...     def connect(self) -> None:
        ...         self._connection = create_connection()
        ...         self._is_connected = True
        ...
        >>> with MyConnector() as conn:
        ...     df = conn.execute_query("SELECT * FROM patients")
    """

    def __init__(self) -> None:
        """Initialize the connector in disconnected state."""
        self._is_connected: bool = False
        self._connection: Any = None
        self._logger = logger.bind(connector_type=self.__class__.__name__)

    @property
    def is_connected(self) -> bool:
        """Check if the connector is currently connected."""
        return self._is_connected

    @property
    @abstractmethod
    def backend_name(self) -> str:
        """Return the name of the database backend.

        Returns:
            String identifier for the backend (e.g., 'databricks', 'sqlserver').
        """
        ...

    @abstractmethod
    def connect(self) -> None:
        """Establish connection to the data source.

        Raises:
            ConnectionFailedError: If connection cannot be established.
            ConnectionTimeoutError: If connection times out.
        """
        ...

    @abstractmethod
    def close(self) -> None:
        """Close the connection and release resources.

        This method should be idempotent - calling it multiple times
        should not raise an error.
        """
        ...

    @abstractmethod
    def execute_query(self, query: str, params: dict[str, Any] | None = None) -> pd.DataFrame:
        """Execute a query and return results as a DataFrame.

        Args:
            query: SQL query string to execute.
            params: Optional dictionary of query parameters for parameterized queries.

        Returns:
            pandas DataFrame containing the query results.

        Raises:
            ConnectionNotEstablishedError: If not connected.
            QueryExecutionError: If query fails.
            QueryTimeoutError: If query times out.
        """
        ...

    @abstractmethod
    def execute_query_iterator(
        self,
        query: str,
        chunk_size: int = 10000,
        params: dict[str, Any] | None = None,
    ) -> Iterator[pd.DataFrame]:
        """Execute a query and return results as an iterator of DataFrames.

        This method is useful for processing large result sets without
        loading everything into memory.

        Args:
            query: SQL query string to execute.
            chunk_size: Number of rows per chunk.
            params: Optional dictionary of query parameters.

        Yields:
            pandas DataFrames containing chunks of results.

        Raises:
            ConnectionNotEstablishedError: If not connected.
            QueryExecutionError: If query fails.
        """
        ...

    @abstractmethod
    def execute_statement(self, statement: str, params: dict[str, Any] | None = None) -> int:
        """Execute a SQL statement (INSERT, UPDATE, DELETE) and return affected rows.

        Args:
            statement: SQL statement to execute.
            params: Optional dictionary of query parameters.

        Returns:
            Number of rows affected by the statement.

        Raises:
            ConnectionNotEstablishedError: If not connected.
            QueryExecutionError: If statement fails.
        """
        ...

    @abstractmethod
    def table_exists(self, table_name: str, schema: str | None = None) -> bool:
        """Check if a table exists in the database.

        Args:
            table_name: Name of the table to check.
            schema: Optional schema name.

        Returns:
            True if table exists, False otherwise.
        """
        ...

    @abstractmethod
    def get_table_schema(
        self, table_name: str, schema: str | None = None
    ) -> list[dict[str, Any]]:
        """Get the schema (column definitions) for a table.

        Args:
            table_name: Name of the table.
            schema: Optional schema name.

        Returns:
            List of dictionaries with column information:
            - name: Column name
            - type: Data type
            - nullable: Whether column allows nulls
            - default: Default value if any
        """
        ...

    @abstractmethod
    def get_row_count(self, table_name: str, schema: str | None = None) -> int:
        """Get the total number of rows in a table.

        Args:
            table_name: Name of the table.
            schema: Optional schema name.

        Returns:
            Number of rows in the table.
        """
        ...

    def _ensure_connected(self) -> None:
        """Ensure the connector is connected before operations.

        Raises:
            ConnectionNotEstablishedError: If not connected.
        """
        if not self._is_connected:
            raise ConnectionNotEstablishedError(self.backend_name)

    def __enter__(self: T) -> T:
        """Enter context manager and establish connection."""
        self.connect()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> None:
        """Exit context manager and close connection."""
        self.close()

    def __repr__(self) -> str:
        """Return string representation of connector."""
        status = "connected" if self._is_connected else "disconnected"
        return f"<{self.__class__.__name__} ({status})>"


class TransactionalConnector(DataConnector):
    """Extended connector interface supporting transactions.

    This class extends DataConnector to add transaction support for
    databases that support ACID transactions.
    """

    @abstractmethod
    def begin_transaction(self) -> None:
        """Begin a new transaction."""
        ...

    @abstractmethod
    def commit(self) -> None:
        """Commit the current transaction."""
        ...

    @abstractmethod
    def rollback(self) -> None:
        """Rollback the current transaction."""
        ...

    @contextmanager
    def transaction(self) -> Generator[None, None, None]:
        """Context manager for transaction handling.

        Example:
            >>> with connector.transaction():
            ...     connector.execute_statement("INSERT INTO ...")
            ...     connector.execute_statement("UPDATE ...")
            ...     # Commits on success, rollback on exception
        """
        self.begin_transaction()
        try:
            yield
            self.commit()
        except Exception:
            self.rollback()
            raise


class BatchConnector(DataConnector):
    """Extended connector interface supporting batch operations.

    This class extends DataConnector to add batch insert/update
    capabilities for efficient bulk data loading.
    """

    @abstractmethod
    def batch_insert(
        self,
        table_name: str,
        data: pd.DataFrame,
        schema: str | None = None,
        chunk_size: int = 1000,
    ) -> int:
        """Insert data in batches.

        Args:
            table_name: Target table name.
            data: DataFrame containing data to insert.
            schema: Optional schema name.
            chunk_size: Number of rows per batch.

        Returns:
            Total number of rows inserted.
        """
        ...

    @abstractmethod
    def truncate_table(self, table_name: str, schema: str | None = None) -> None:
        """Truncate (delete all rows from) a table.

        Args:
            table_name: Name of the table to truncate.
            schema: Optional schema name.
        """
        ...
