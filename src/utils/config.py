"""
Configuration management for the Oncology Data Pipeline.

This module provides centralized configuration management using Pydantic
settings, supporting environment variables and .env files.
"""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class DatabricksSettings(BaseSettings):
    """Databricks connection settings."""

    model_config = SettingsConfigDict(
        env_prefix="DATABRICKS_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    host: str = Field(
        default="",
        description="Databricks workspace URL",
    )
    http_path: str = Field(
        default="",
        description="SQL warehouse HTTP path",
    )
    token: SecretStr = Field(
        default=SecretStr(""),
        description="Databricks personal access token",
    )
    catalog: str = Field(
        default="hive_metastore",
        description="Unity Catalog name",
    )
    schema_name: str = Field(
        default="default",
        alias="schema",
        description="Database schema name",
    )

    @field_validator("host")
    @classmethod
    def validate_host(cls, v: str) -> str:
        """Ensure host has proper format."""
        if v and not v.startswith("https://"):
            return f"https://{v}"
        return v

    @property
    def is_configured(self) -> bool:
        """Check if Databricks is properly configured."""
        return bool(self.host and self.http_path and self.token.get_secret_value())


class SqlServerSettings(BaseSettings):
    """SQL Server connection settings."""

    model_config = SettingsConfigDict(
        env_prefix="SQLSERVER_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    host: str = Field(
        default="",
        description="SQL Server hostname",
    )
    port: int = Field(
        default=1433,
        description="SQL Server port",
    )
    database: str = Field(
        default="",
        description="Database name",
    )
    username: str = Field(
        default="",
        description="Database username",
    )
    password: SecretStr = Field(
        default=SecretStr(""),
        description="Database password",
    )
    driver: str = Field(
        default="ODBC Driver 18 for SQL Server",
        description="ODBC driver name",
    )
    encrypt: bool = Field(
        default=True,
        description="Enable encryption",
    )
    trust_server_certificate: bool = Field(
        default=False,
        description="Trust server certificate",
    )
    connection_timeout: int = Field(
        default=30,
        description="Connection timeout in seconds",
    )
    query_timeout: int = Field(
        default=300,
        description="Query timeout in seconds",
    )

    @property
    def is_configured(self) -> bool:
        """Check if SQL Server is properly configured."""
        return bool(
            self.host and self.database and self.username and self.password.get_secret_value()
        )

    @property
    def connection_string(self) -> str:
        """Generate ODBC connection string."""
        return (
            f"DRIVER={{{self.driver}}};"
            f"SERVER={self.host},{self.port};"
            f"DATABASE={self.database};"
            f"UID={self.username};"
            f"PWD={self.password.get_secret_value()};"
            f"Encrypt={'yes' if self.encrypt else 'no'};"
            f"TrustServerCertificate={'yes' if self.trust_server_certificate else 'no'};"
            f"Connection Timeout={self.connection_timeout};"
        )


class GreatExpectationsSettings(BaseSettings):
    """Great Expectations configuration settings."""

    model_config = SettingsConfigDict(
        env_prefix="GE_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    root_dir: Path = Field(
        default=Path("great_expectations"),
        description="Great Expectations project root directory",
    )
    data_docs_site: str = Field(
        default="local_site",
        description="Data docs site name",
    )
    checkpoint_store: str = Field(
        default="checkpoint_store",
        description="Checkpoint store name",
    )


class LoggingSettings(BaseSettings):
    """Logging configuration settings."""

    model_config = SettingsConfigDict(
        env_prefix="LOG_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="INFO",
        description="Logging level",
    )
    format: Literal["json", "console"] = Field(
        default="console",
        description="Log output format",
    )
    file_path: Path | None = Field(
        default=None,
        description="Optional log file path",
    )


class AppSettings(BaseSettings):
    """Main application settings."""

    model_config = SettingsConfigDict(
        env_prefix="APP_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    name: str = Field(
        default="oncology-datapipeline",
        description="Application name",
    )
    environment: Literal["development", "staging", "production"] = Field(
        default="development",
        description="Deployment environment",
    )
    debug: bool = Field(
        default=False,
        description="Enable debug mode",
    )

    # Nested settings
    databricks: DatabricksSettings = Field(default_factory=DatabricksSettings)
    sqlserver: SqlServerSettings = Field(default_factory=SqlServerSettings)
    great_expectations: GreatExpectationsSettings = Field(default_factory=GreatExpectationsSettings)
    logging: LoggingSettings = Field(default_factory=LoggingSettings)

    @property
    def is_production(self) -> bool:
        """Check if running in production environment."""
        return self.environment == "production"

    @property
    def is_development(self) -> bool:
        """Check if running in development environment."""
        return self.environment == "development"


@lru_cache
def get_settings() -> AppSettings:
    """Get cached application settings.

    Returns:
        AppSettings instance with all configuration loaded.

    Example:
        >>> settings = get_settings()
        >>> print(settings.environment)
        'development'
    """
    return AppSettings()


def get_databricks_settings() -> DatabricksSettings:
    """Get Databricks settings."""
    return get_settings().databricks


def get_sqlserver_settings() -> SqlServerSettings:
    """Get SQL Server settings."""
    return get_settings().sqlserver


def get_ge_settings() -> GreatExpectationsSettings:
    """Get Great Expectations settings."""
    return get_settings().great_expectations
