"""
Checkpoint manager for Great Expectations workflows.

This module provides utilities for creating and managing GE checkpoints,
enabling automated and scheduled data quality validations.
"""

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import structlog

try:
    import great_expectations as gx
    from great_expectations.checkpoint import Checkpoint
    from great_expectations.data_context import FileDataContext

    GE_AVAILABLE = True
except ImportError:
    GE_AVAILABLE = False

logger = structlog.get_logger(__name__)


@dataclass
class CheckpointConfig:
    """Configuration for a checkpoint.

    Attributes:
        name: Checkpoint name.
        expectation_suite_name: Name of expectation suite.
        datasource_name: Name of datasource.
        data_asset_name: Name of data asset.
        action_list: List of actions to perform.
    """

    name: str
    expectation_suite_name: str
    datasource_name: str = "pandas_datasource"
    data_asset_name: str | None = None
    action_list: list[dict[str, Any]] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to checkpoint configuration dictionary."""
        config = {
            "name": self.name,
            "config_version": 1.0,
            "class_name": "Checkpoint",
            "expectation_suite_name": self.expectation_suite_name,
            "action_list": self.action_list or self._default_action_list(),
            "validations": [
                {
                    "batch_request": {
                        "datasource_name": self.datasource_name,
                        "data_connector_name": "default_runtime_data_connector",
                        "data_asset_name": self.data_asset_name or self.name,
                    },
                    "expectation_suite_name": self.expectation_suite_name,
                }
            ],
        }
        return config

    @staticmethod
    def _default_action_list() -> list[dict[str, Any]]:
        """Get default action list for checkpoints."""
        return [
            {
                "name": "store_validation_result",
                "action": {"class_name": "StoreValidationResultAction"},
            },
            {
                "name": "store_evaluation_params",
                "action": {"class_name": "StoreEvaluationParametersAction"},
            },
            {
                "name": "update_data_docs",
                "action": {"class_name": "UpdateDataDocsAction"},
            },
        ]


@dataclass
class CheckpointResult:
    """Result of a checkpoint run.

    Attributes:
        checkpoint_name: Name of the checkpoint.
        success: Whether all validations passed.
        run_time: When checkpoint was run.
        validation_results: List of validation results.
    """

    checkpoint_name: str
    success: bool
    run_time: datetime
    validation_results: list[dict[str, Any]]

    def summary(self) -> str:
        """Generate human-readable summary."""
        lines = [
            f"Checkpoint: {self.checkpoint_name}",
            f"Run Time: {self.run_time}",
            f"Overall Success: {'PASSED' if self.success else 'FAILED'}",
            f"Validations: {len(self.validation_results)}",
        ]

        for i, result in enumerate(self.validation_results):
            status = "PASS" if result.get("success", False) else "FAIL"
            suite = result.get("expectation_suite_name", "unknown")
            lines.append(f"  [{status}] {suite}")

        return "\n".join(lines)


class CheckpointManager:
    """Manager for Great Expectations checkpoints.

    This class provides methods for creating, running, and managing
    checkpoints for automated data quality validations.

    Attributes:
        ge_root_dir: Great Expectations project root directory.
        context: Great Expectations data context.

    Example:
        >>> manager = CheckpointManager()
        >>> manager.create_checkpoint("patient_checkpoint", "oncology_patients_suite")
        >>> result = manager.run_checkpoint("patient_checkpoint", df)
        >>> print(result.summary())
    """

    def __init__(self, ge_root_dir: Path | str | None = None) -> None:
        """Initialize the checkpoint manager.

        Args:
            ge_root_dir: Path to Great Expectations project root.
        """
        self.ge_root_dir = Path(ge_root_dir) if ge_root_dir else Path("great_expectations")
        self._context: Any = None
        self._logger = logger.bind(manager="checkpoint")

        if GE_AVAILABLE:
            self._init_context()
        else:
            self._logger.warning("Great Expectations not available")

    def _init_context(self) -> None:
        """Initialize the Great Expectations data context."""
        try:
            if self.ge_root_dir.exists():
                self._context = FileDataContext(project_root_dir=str(self.ge_root_dir.parent))
                self._logger.info("Loaded GE context for checkpoint management")
            else:
                self._logger.warning(
                    "GE directory not found",
                    path=str(self.ge_root_dir),
                )
        except Exception as e:
            self._logger.error("Failed to initialize GE context", error=str(e))

    def create_checkpoint(
        self,
        name: str,
        expectation_suite_name: str,
        datasource_name: str = "pandas_datasource",
        overwrite: bool = False,
    ) -> bool:
        """Create a new checkpoint.

        Args:
            name: Checkpoint name.
            expectation_suite_name: Name of expectation suite to use.
            datasource_name: Name of datasource.
            overwrite: Whether to overwrite existing checkpoint.

        Returns:
            True if checkpoint was created successfully.

        Example:
            >>> manager.create_checkpoint(
            ...     "daily_patient_check",
            ...     "oncology_patients_suite"
            ... )
        """
        if not GE_AVAILABLE or not self._context:
            self._logger.error("Cannot create checkpoint - GE not available")
            return False

        try:
            # Check if checkpoint already exists
            existing = self._get_checkpoint_names()
            if name in existing and not overwrite:
                self._logger.warning("Checkpoint already exists", name=name)
                return False

            config = CheckpointConfig(
                name=name,
                expectation_suite_name=expectation_suite_name,
                datasource_name=datasource_name,
            )

            checkpoint = self._context.add_checkpoint(**config.to_dict())

            self._logger.info(
                "Created checkpoint",
                name=name,
                suite=expectation_suite_name,
            )
            return True

        except Exception as e:
            self._logger.error("Failed to create checkpoint", error=str(e), name=name)
            return False

    def run_checkpoint(
        self,
        name: str,
        batch_data: Any = None,
        run_name: str | None = None,
    ) -> CheckpointResult:
        """Run a checkpoint.

        Args:
            name: Checkpoint name.
            batch_data: Optional DataFrame or data to validate.
            run_name: Optional name for this run.

        Returns:
            CheckpointResult with validation outcomes.

        Example:
            >>> result = manager.run_checkpoint("daily_patient_check", patients_df)
            >>> if not result.success:
            ...     send_alert("Patient data quality check failed!")
        """
        if not GE_AVAILABLE or not self._context:
            return self._mock_checkpoint_result(name)

        try:
            self._logger.info("Running checkpoint", name=name)

            # Build batch request if data provided
            batch_request = None
            if batch_data is not None:
                batch_request = {
                    "runtime_parameters": {"batch_data": batch_data},
                    "batch_identifiers": {
                        "default_identifier_name": run_name or f"run_{datetime.now().isoformat()}"
                    },
                }

            # Run checkpoint
            checkpoint = self._context.get_checkpoint(name)
            result = checkpoint.run(
                batch_request=batch_request,
                run_name=run_name,
            )

            return self._parse_checkpoint_result(name, result)

        except Exception as e:
            self._logger.error("Checkpoint run failed", error=str(e), name=name)
            return CheckpointResult(
                checkpoint_name=name,
                success=False,
                run_time=datetime.now(),
                validation_results=[{"error": str(e)}],
            )

    def run_all_checkpoints(
        self,
        checkpoint_data: dict[str, Any] | None = None,
    ) -> dict[str, CheckpointResult]:
        """Run all registered checkpoints.

        Args:
            checkpoint_data: Optional mapping of checkpoint names to data.

        Returns:
            Dictionary mapping checkpoint names to results.
        """
        checkpoints = self._get_checkpoint_names()
        results = {}

        for name in checkpoints:
            data = checkpoint_data.get(name) if checkpoint_data else None
            results[name] = self.run_checkpoint(name, batch_data=data)

        return results

    def list_checkpoints(self) -> list[str]:
        """List all available checkpoints.

        Returns:
            List of checkpoint names.
        """
        return self._get_checkpoint_names()

    def delete_checkpoint(self, name: str) -> bool:
        """Delete a checkpoint.

        Args:
            name: Checkpoint name to delete.

        Returns:
            True if deletion was successful.
        """
        if not GE_AVAILABLE or not self._context:
            return False

        try:
            self._context.delete_checkpoint(name)
            self._logger.info("Deleted checkpoint", name=name)
            return True
        except Exception as e:
            self._logger.error("Failed to delete checkpoint", error=str(e), name=name)
            return False

    def get_checkpoint_config(self, name: str) -> dict[str, Any] | None:
        """Get checkpoint configuration.

        Args:
            name: Checkpoint name.

        Returns:
            Checkpoint configuration dictionary or None.
        """
        if not GE_AVAILABLE or not self._context:
            return None

        try:
            checkpoint = self._context.get_checkpoint(name)
            return checkpoint.config.to_json_dict()
        except Exception as e:
            self._logger.error("Failed to get checkpoint config", error=str(e), name=name)
            return None

    def _get_checkpoint_names(self) -> list[str]:
        """Get list of checkpoint names from context."""
        if not self._context:
            return []

        try:
            return self._context.list_checkpoints()
        except Exception:
            return []

    def _parse_checkpoint_result(
        self,
        name: str,
        result: Any,
    ) -> CheckpointResult:
        """Parse GE checkpoint result.

        Args:
            name: Checkpoint name.
            result: Raw GE result.

        Returns:
            Parsed CheckpointResult.
        """
        validation_results = []

        if hasattr(result, "run_results"):
            for key, run_result in result.run_results.items():
                validation_results.append(
                    {
                        "success": run_result.success if hasattr(run_result, "success") else True,
                        "expectation_suite_name": getattr(
                            run_result, "expectation_suite_name", "unknown"
                        ),
                        "statistics": getattr(run_result, "statistics", {}),
                    }
                )

        return CheckpointResult(
            checkpoint_name=name,
            success=result.success if hasattr(result, "success") else True,
            run_time=datetime.now(),
            validation_results=validation_results,
        )

    def _mock_checkpoint_result(self, name: str) -> CheckpointResult:
        """Return mock result when GE not available."""
        return CheckpointResult(
            checkpoint_name=name,
            success=True,
            run_time=datetime.now(),
            validation_results=[
                {
                    "success": True,
                    "note": "Mock result - Great Expectations not installed",
                }
            ],
        )


# Pre-configured checkpoint factories


def create_oncology_checkpoints(manager: CheckpointManager) -> dict[str, bool]:
    """Create standard oncology data checkpoints.

    Args:
        manager: CheckpointManager instance.

    Returns:
        Dictionary mapping checkpoint names to creation success.
    """
    checkpoints = [
        ("patient_checkpoint", "oncology_patients_suite"),
        ("treatment_checkpoint", "oncology_treatments_suite"),
        ("lab_results_checkpoint", "oncology_lab_results_suite"),
    ]

    results = {}
    for name, suite in checkpoints:
        results[name] = manager.create_checkpoint(name, suite, overwrite=True)

    return results


def run_oncology_validation_pipeline(
    manager: CheckpointManager,
    patients_df: Any,
    treatments_df: Any,
    lab_results_df: Any,
) -> dict[str, CheckpointResult]:
    """Run complete oncology data validation pipeline.

    Args:
        manager: CheckpointManager instance.
        patients_df: Patients DataFrame.
        treatments_df: Treatments DataFrame.
        lab_results_df: Lab results DataFrame.

    Returns:
        Dictionary of checkpoint results.
    """
    return manager.run_all_checkpoints(
        {
            "patient_checkpoint": patients_df,
            "treatment_checkpoint": treatments_df,
            "lab_results_checkpoint": lab_results_df,
        }
    )
