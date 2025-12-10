"""
Databricks SQL connector for the Oncology Data Pipeline.

This module provides a connector for Databricks SQL warehouses,
supporting query execution, batch operations, and Delta Lake features.
"""

from typing import Any, Iterator

import pandas as pd
import structlog

from src.connectors.base import BatchConnector
from src.domain.exceptions import (
    ConnectionFailedError,
    QueryExecutionError,
)
from src.utils.config import DatabricksSettings, get_databricks_settings

logger = structlog.get_logger(__name__)


class DatabricksConnector(BatchConnector):
    """Connector for Databricks SQL warehouses.

    This connector uses the Databricks SQL connector to execute queries
    against Databricks SQL warehouses, supporting both ad-hoc queries
    and batch operations.

    Attributes:
        settings: DatabricksSettings instance with connection configuration.
        _cursor: Database cursor for query execution.

    Example:
        >>> from src.connectors import DatabricksConnector
        >>> with DatabricksConnector() as conn:
        ...     df = conn.execute_query("SELECT * FROM oncology.patients LIMIT 10")
        ...     print(f"Found {len(df)} patients")
    """

    def __init__(self, settings: DatabricksSettings | None = None) -> None:
        """Initialize the Databricks connector.

        Args:
            settings: Optional settings override. Uses environment settings if None.
        """
        super().__init__()
        self.settings = settings or get_databricks_settings()
        self._cursor: Any = None
        self._logger = logger.bind(
            connector_type="databricks",
            host=self.settings.host,
        )

    @property
    def backend_name(self) -> str:
        """Return the backend name."""
        return "databricks"

    def connect(self) -> None:
        """Establish connection to Databricks SQL warehouse.

        Raises:
            ConnectionFailedError: If connection fails.
        """
        if self._is_connected:
            self._logger.debug("Already connected to Databricks")
            return

        try:
            from databricks import sql as databricks_sql

            self._logger.info("Connecting to Databricks SQL warehouse")

            self._connection = databricks_sql.connect(
                server_hostname=self.settings.host.replace("https://", ""),
                http_path=self.settings.http_path,
                access_token=self.settings.token.get_secret_value(),
            )
            self._cursor = self._connection.cursor()
            self._is_connected = True

            self._logger.info("Successfully connected to Databricks")

        except ImportError as e:
            raise ConnectionFailedError(
                connector_type="databricks",
                host=self.settings.host,
                reason="databricks-sql-connector not installed. Install with: pip install databricks-sql-connector",
            ) from e
        except Exception as e:
            raise ConnectionFailedError(
                connector_type="databricks",
                host=self.settings.host,
                reason=str(e),
            ) from e

    def close(self) -> None:
        """Close the Databricks connection."""
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
        self._logger.info("Disconnected from Databricks")

    def execute_query(self, query: str, params: dict[str, Any] | None = None) -> pd.DataFrame:
        """Execute a query and return results as DataFrame.

        Args:
            query: SQL query to execute.
            params: Optional query parameters.

        Returns:
            DataFrame with query results.

        Raises:
            ConnectionNotEstablishedError: If not connected.
            QueryExecutionError: If query fails.
        """
        self._ensure_connected()

        try:
            self._logger.debug("Executing query", query=query[:200])

            if params:
                self._cursor.execute(query, params)
            else:
                self._cursor.execute(query)

            # Get column names from cursor description
            columns = [desc[0] for desc in self._cursor.description] if self._cursor.description else []

            # Fetch all results
            rows = self._cursor.fetchall()

            df = pd.DataFrame(rows, columns=columns)
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
                self._cursor.execute(query, params)
            else:
                self._cursor.execute(query)

            columns = [desc[0] for desc in self._cursor.description] if self._cursor.description else []

            while True:
                rows = self._cursor.fetchmany(chunk_size)
                if not rows:
                    break
                yield pd.DataFrame(rows, columns=columns)

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
                self._cursor.execute(statement, params)
            else:
                self._cursor.execute(statement)

            # Databricks SQL connector doesn't have rowcount for all operations
            affected = getattr(self._cursor, "rowcount", -1)
            self._logger.debug("Statement executed", rows_affected=affected)

            return affected if affected >= 0 else 0

        except Exception as e:
            raise QueryExecutionError(query=statement, reason=str(e)) from e

    def table_exists(self, table_name: str, schema: str | None = None) -> bool:
        """Check if a table exists.

        Args:
            table_name: Name of the table.
            schema: Optional schema/database name.

        Returns:
            True if table exists.
        """
        self._ensure_connected()

        full_name = f"{schema}.{table_name}" if schema else table_name

        try:
            query = f"DESCRIBE TABLE {full_name}"
            self._cursor.execute(query)
            self._cursor.fetchall()
            return True
        except Exception:
            return False

    def get_table_schema(
        self, table_name: str, schema: str | None = None
    ) -> list[dict[str, Any]]:
        """Get table schema information.

        Args:
            table_name: Name of the table.
            schema: Optional schema/database name.

        Returns:
            List of column definitions.
        """
        self._ensure_connected()

        full_name = f"{schema}.{table_name}" if schema else table_name

        df = self.execute_query(f"DESCRIBE TABLE {full_name}")

        columns = []
        for _, row in df.iterrows():
            columns.append(
                {
                    "name": row.get("col_name", row.iloc[0]),
                    "type": row.get("data_type", row.iloc[1] if len(row) > 1 else "unknown"),
                    "nullable": True,  # Databricks doesn't easily expose this
                    "default": None,
                }
            )

        return columns

    def get_row_count(self, table_name: str, schema: str | None = None) -> int:
        """Get row count for a table.

        Args:
            table_name: Name of the table.
            schema: Optional schema/database name.

        Returns:
            Number of rows.
        """
        full_name = f"{schema}.{table_name}" if schema else table_name
        df = self.execute_query(f"SELECT COUNT(*) as cnt FROM {full_name}")
        return int(df["cnt"].iloc[0])

    def batch_insert(
        self,
        table_name: str,
        data: pd.DataFrame,
        schema: str | None = None,
        chunk_size: int = 1000,
    ) -> int:
        """Insert data in batches using Databricks SQL.

        Args:
            table_name: Target table name.
            data: DataFrame to insert.
            schema: Optional schema name.
            chunk_size: Rows per batch.

        Returns:
            Total rows inserted.
        """
        self._ensure_connected()

        full_name = f"{schema}.{table_name}" if schema else table_name
        total_inserted = 0

        columns = ", ".join(data.columns)
        placeholders = ", ".join(["?"] * len(data.columns))

        insert_stmt = f"INSERT INTO {full_name} ({columns}) VALUES ({placeholders})"

        self._logger.info(
            "Starting batch insert",
            table=full_name,
            total_rows=len(data),
            chunk_size=chunk_size,
        )

        for start in range(0, len(data), chunk_size):
            chunk = data.iloc[start : start + chunk_size]

            for _, row in chunk.iterrows():
                values = tuple(row)
                self._cursor.execute(insert_stmt, values)
                total_inserted += 1

            self._logger.debug(
                "Batch inserted",
                rows=len(chunk),
                progress=f"{start + len(chunk)}/{len(data)}",
            )

        self._logger.info("Batch insert complete", total_inserted=total_inserted)
        return total_inserted

    def truncate_table(self, table_name: str, schema: str | None = None) -> None:
        """Truncate a table.

        Args:
            table_name: Name of the table.
            schema: Optional schema name.
        """
        full_name = f"{schema}.{table_name}" if schema else table_name
        self.execute_statement(f"TRUNCATE TABLE {full_name}")
        self._logger.info("Table truncated", table=full_name)

    # Databricks-specific methods

    def optimize_table(self, table_name: str, schema: str | None = None) -> None:
        """Optimize a Delta table.

        Args:
            table_name: Name of the Delta table.
            schema: Optional schema name.
        """
        full_name = f"{schema}.{table_name}" if schema else table_name
        self.execute_statement(f"OPTIMIZE {full_name}")
        self._logger.info("Table optimized", table=full_name)

    def vacuum_table(
        self, table_name: str, schema: str | None = None, retention_hours: int = 168
    ) -> None:
        """Vacuum a Delta table to remove old files.

        Args:
            table_name: Name of the Delta table.
            schema: Optional schema name.
            retention_hours: Hours of history to retain (default 168 = 7 days).
        """
        full_name = f"{schema}.{table_name}" if schema else table_name
        self.execute_statement(f"VACUUM {full_name} RETAIN {retention_hours} HOURS")
        self._logger.info("Table vacuumed", table=full_name, retention_hours=retention_hours)

    def get_table_history(
        self, table_name: str, schema: str | None = None, limit: int = 10
    ) -> pd.DataFrame:
        """Get Delta table history.

        Args:
            table_name: Name of the Delta table.
            schema: Optional schema name.
            limit: Maximum number of history entries.

        Returns:
            DataFrame with table history.
        """
        full_name = f"{schema}.{table_name}" if schema else table_name
        return self.execute_query(f"DESCRIBE HISTORY {full_name} LIMIT {limit}")
