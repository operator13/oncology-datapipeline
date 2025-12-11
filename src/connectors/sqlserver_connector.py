"""
SQL Server connector for the Oncology Data Pipeline.

This module provides a connector for Microsoft SQL Server databases,
supporting query execution, transactions, and batch operations.
"""

from typing import Any, Iterator

import pandas as pd
import structlog

from src.connectors.base import BatchConnector, TransactionalConnector
from src.domain.exceptions import ConnectionFailedError, QueryExecutionError
from src.utils.config import SqlServerSettings, get_sqlserver_settings

logger = structlog.get_logger(__name__)


class SqlServerConnector(TransactionalConnector, BatchConnector):
    """Connector for Microsoft SQL Server databases.

    This connector uses pyodbc to connect to SQL Server, supporting
    ACID transactions, parameterized queries, and batch operations.

    Attributes:
        settings: SqlServerSettings instance with connection configuration.
        _cursor: Database cursor for query execution.
        _in_transaction: Flag indicating if a transaction is active.

    Example:
        >>> from src.connectors import SqlServerConnector
        >>> with SqlServerConnector() as conn:
        ...     df = conn.execute_query("SELECT TOP 10 * FROM patients")
        ...     print(f"Found {len(df)} patients")
    """

    def __init__(self, settings: SqlServerSettings | None = None) -> None:
        """Initialize the SQL Server connector.

        Args:
            settings: Optional settings override. Uses environment settings if None.
        """
        super().__init__()
        self.settings = settings or get_sqlserver_settings()
        self._cursor: Any = None
        self._in_transaction: bool = False
        self._logger = logger.bind(
            connector_type="sqlserver",
            host=self.settings.host,
            database=self.settings.database,
        )

    @property
    def backend_name(self) -> str:
        """Return the backend name."""
        return "sqlserver"

    def connect(self) -> None:
        """Establish connection to SQL Server.

        Raises:
            ConnectionFailedError: If connection fails.
        """
        if self._is_connected:
            self._logger.debug("Already connected to SQL Server")
            return

        try:
            import pyodbc

            self._logger.info("Connecting to SQL Server")

            self._connection = pyodbc.connect(
                self.settings.connection_string,
                timeout=self.settings.connection_timeout,
            )
            self._connection.autocommit = True  # Default to autocommit
            self._cursor = self._connection.cursor()
            self._is_connected = True

            self._logger.info(
                "Successfully connected to SQL Server",
                database=self.settings.database,
            )

        except ImportError as e:
            raise ConnectionFailedError(
                connector_type="sqlserver",
                host=self.settings.host,
                reason="pyodbc not installed. Install with: pip install pyodbc",
            ) from e
        except Exception as e:
            raise ConnectionFailedError(
                connector_type="sqlserver",
                host=self.settings.host,
                reason=str(e),
            ) from e

    def close(self) -> None:
        """Close the SQL Server connection."""
        if self._in_transaction:
            self._logger.warning("Closing connection with active transaction, rolling back")
            try:
                self.rollback()
            except Exception as e:
                self._logger.error("Error during rollback on close", error=str(e))

        if self._cursor:
            try:
                self._cursor.close()
            except Exception as e:
                self._logger.warning("Error closing cursor", error=str(e))
            self._cursor = None

        if self._connection:
            try:
                self._connection.close()
            except Exception as e:
                self._logger.warning("Error closing connection", error=str(e))
            self._connection = None

        self._is_connected = False
        self._logger.info("Disconnected from SQL Server")

    def execute_query(self, query: str, params: dict[str, Any] | None = None) -> pd.DataFrame:
        """Execute a query and return results as DataFrame.

        Args:
            query: SQL query to execute.
            params: Optional query parameters (uses named parameters).

        Returns:
            DataFrame with query results.

        Raises:
            ConnectionNotEstablishedError: If not connected.
            QueryExecutionError: If query fails.
        """
        self._ensure_connected()

        try:
            self._logger.debug("Executing query", query=query[:200])

            # Convert dict params to tuple for pyodbc
            if params:
                # Replace named params with ? placeholders
                processed_query = query
                param_values = []
                for key, value in params.items():
                    processed_query = processed_query.replace(f":{key}", "?")
                    processed_query = processed_query.replace(f"@{key}", "?")
                    param_values.append(value)
                self._cursor.execute(processed_query, param_values)
            else:
                self._cursor.execute(query)

            # Get column names
            columns = (
                [desc[0] for desc in self._cursor.description] if self._cursor.description else []
            )

            # Fetch all results
            rows = self._cursor.fetchall()

            # Convert pyodbc rows to list of tuples
            df = pd.DataFrame([tuple(row) for row in rows], columns=columns)
            self._logger.debug("Query completed", rows_returned=len(df))

            return df

        except Exception as e:
            raise QueryExecutionError(query=query, reason=str(e)) from e

    def execute_query_iterator(
        self,
        query: str,
        chunk_size: int = 10000,
        params: dict[str, Any] | None = None,
    ) -> Iterator[pd.DataFrame]:
        """Execute a query and return results as chunks.

        Args:
            query: SQL query to execute.
            chunk_size: Number of rows per chunk.
            params: Optional query parameters.

        Yields:
            DataFrames containing chunks of results.
        """
        self._ensure_connected()

        try:
            self._logger.debug("Executing chunked query", query=query[:200], chunk_size=chunk_size)

            if params:
                processed_query = query
                param_values = []
                for key, value in params.items():
                    processed_query = processed_query.replace(f":{key}", "?")
                    processed_query = processed_query.replace(f"@{key}", "?")
                    param_values.append(value)
                self._cursor.execute(processed_query, param_values)
            else:
                self._cursor.execute(query)

            columns = (
                [desc[0] for desc in self._cursor.description] if self._cursor.description else []
            )

            while True:
                rows = self._cursor.fetchmany(chunk_size)
                if not rows:
                    break
                yield pd.DataFrame([tuple(row) for row in rows], columns=columns)

        except Exception as e:
            raise QueryExecutionError(query=query, reason=str(e)) from e

    def execute_statement(self, statement: str, params: dict[str, Any] | None = None) -> int:
        """Execute a SQL statement and return affected rows.

        Args:
            statement: SQL statement (INSERT, UPDATE, DELETE).
            params: Optional statement parameters.

        Returns:
            Number of affected rows.
        """
        self._ensure_connected()

        try:
            self._logger.debug("Executing statement", statement=statement[:200])

            if params:
                processed_stmt = statement
                param_values = []
                for key, value in params.items():
                    processed_stmt = processed_stmt.replace(f":{key}", "?")
                    processed_stmt = processed_stmt.replace(f"@{key}", "?")
                    param_values.append(value)
                self._cursor.execute(processed_stmt, param_values)
            else:
                self._cursor.execute(statement)

            affected = self._cursor.rowcount
            self._logger.debug("Statement executed", rows_affected=affected)

            return affected

        except Exception as e:
            raise QueryExecutionError(query=statement, reason=str(e)) from e

    def table_exists(self, table_name: str, schema: str | None = None) -> bool:
        """Check if a table exists.

        Args:
            table_name: Name of the table.
            schema: Optional schema name (default: dbo).

        Returns:
            True if table exists.
        """
        self._ensure_connected()

        schema = schema or "dbo"
        query = """
            SELECT COUNT(*) as cnt
            FROM INFORMATION_SCHEMA.TABLES
            WHERE TABLE_SCHEMA = ?
            AND TABLE_NAME = ?
        """

        self._cursor.execute(query, (schema, table_name))
        result = self._cursor.fetchone()
        return result[0] > 0 if result else False

    def get_table_schema(self, table_name: str, schema: str | None = None) -> list[dict[str, Any]]:
        """Get table schema information.

        Args:
            table_name: Name of the table.
            schema: Optional schema name (default: dbo).

        Returns:
            List of column definitions.
        """
        self._ensure_connected()

        schema = schema or "dbo"
        query = """
            SELECT
                c.COLUMN_NAME as name,
                c.DATA_TYPE as type,
                CASE WHEN c.IS_NULLABLE = 'YES' THEN 1 ELSE 0 END as nullable,
                c.COLUMN_DEFAULT as default_value
            FROM INFORMATION_SCHEMA.COLUMNS c
            WHERE c.TABLE_SCHEMA = ?
            AND c.TABLE_NAME = ?
            ORDER BY c.ORDINAL_POSITION
        """

        df = self.execute_query(query, {"schema": schema, "table": table_name})

        # Re-execute with positional params since our execute_query converts them
        self._cursor.execute(query, (schema, table_name))
        rows = self._cursor.fetchall()

        columns = []
        for row in rows:
            columns.append(
                {
                    "name": row[0],
                    "type": row[1],
                    "nullable": bool(row[2]),
                    "default": row[3],
                }
            )

        return columns

    def get_row_count(self, table_name: str, schema: str | None = None) -> int:
        """Get row count for a table.

        Args:
            table_name: Name of the table.
            schema: Optional schema name (default: dbo).

        Returns:
            Number of rows.
        """
        schema = schema or "dbo"
        full_name = f"[{schema}].[{table_name}]"
        df = self.execute_query(f"SELECT COUNT(*) as cnt FROM {full_name}")
        return int(df["cnt"].iloc[0])

    # Transaction methods

    def begin_transaction(self) -> None:
        """Begin a new transaction."""
        self._ensure_connected()

        if self._in_transaction:
            self._logger.warning("Transaction already active")
            return

        self._connection.autocommit = False
        self._in_transaction = True
        self._logger.debug("Transaction started")

    def commit(self) -> None:
        """Commit the current transaction."""
        self._ensure_connected()

        if not self._in_transaction:
            self._logger.warning("No active transaction to commit")
            return

        self._connection.commit()
        self._connection.autocommit = True
        self._in_transaction = False
        self._logger.debug("Transaction committed")

    def rollback(self) -> None:
        """Rollback the current transaction."""
        self._ensure_connected()

        if not self._in_transaction:
            self._logger.warning("No active transaction to rollback")
            return

        self._connection.rollback()
        self._connection.autocommit = True
        self._in_transaction = False
        self._logger.debug("Transaction rolled back")

    # Batch methods

    def batch_insert(
        self,
        table_name: str,
        data: pd.DataFrame,
        schema: str | None = None,
        chunk_size: int = 1000,
    ) -> int:
        """Insert data in batches using executemany.

        Args:
            table_name: Target table name.
            data: DataFrame to insert.
            schema: Optional schema name (default: dbo).
            chunk_size: Rows per batch.

        Returns:
            Total rows inserted.
        """
        self._ensure_connected()

        schema = schema or "dbo"
        full_name = f"[{schema}].[{table_name}]"
        total_inserted = 0

        columns = ", ".join([f"[{col}]" for col in data.columns])
        placeholders = ", ".join(["?"] * len(data.columns))

        insert_stmt = f"INSERT INTO {full_name} ({columns}) VALUES ({placeholders})"

        self._logger.info(
            "Starting batch insert",
            table=full_name,
            total_rows=len(data),
            chunk_size=chunk_size,
        )

        # Use fast_executemany for better performance
        self._cursor.fast_executemany = True

        try:
            for start in range(0, len(data), chunk_size):
                chunk = data.iloc[start : start + chunk_size]
                values = [tuple(row) for row in chunk.values]

                self._cursor.executemany(insert_stmt, values)
                total_inserted += len(values)

                self._logger.debug(
                    "Batch inserted",
                    rows=len(values),
                    progress=f"{start + len(chunk)}/{len(data)}",
                )

        finally:
            self._cursor.fast_executemany = False

        self._logger.info("Batch insert complete", total_inserted=total_inserted)
        return total_inserted

    def truncate_table(self, table_name: str, schema: str | None = None) -> None:
        """Truncate a table.

        Args:
            table_name: Name of the table.
            schema: Optional schema name (default: dbo).
        """
        schema = schema or "dbo"
        full_name = f"[{schema}].[{table_name}]"
        self.execute_statement(f"TRUNCATE TABLE {full_name}")
        self._logger.info("Table truncated", table=full_name)

    # SQL Server-specific methods

    def get_database_size(self) -> pd.DataFrame:
        """Get database size information.

        Returns:
            DataFrame with database file sizes.
        """
        query = """
            SELECT
                name AS file_name,
                type_desc AS file_type,
                CAST(size * 8.0 / 1024 AS DECIMAL(10,2)) AS size_mb,
                CAST(FILEPROPERTY(name, 'SpaceUsed') * 8.0 / 1024 AS DECIMAL(10,2)) AS used_mb
            FROM sys.database_files
        """
        return self.execute_query(query)

    def get_index_fragmentation(self, table_name: str, schema: str | None = None) -> pd.DataFrame:
        """Get index fragmentation for a table.

        Args:
            table_name: Name of the table.
            schema: Optional schema name (default: dbo).

        Returns:
            DataFrame with index fragmentation information.
        """
        schema = schema or "dbo"
        query = f"""
            SELECT
                i.name AS index_name,
                ips.avg_fragmentation_in_percent,
                ips.page_count
            FROM sys.dm_db_index_physical_stats(
                DB_ID(), OBJECT_ID('{schema}.{table_name}'), NULL, NULL, 'LIMITED'
            ) ips
            JOIN sys.indexes i ON ips.object_id = i.object_id AND ips.index_id = i.index_id
            WHERE i.name IS NOT NULL
        """
        return self.execute_query(query)
