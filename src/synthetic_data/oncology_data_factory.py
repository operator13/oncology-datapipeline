"""
Oncology Data Factory - Main orchestrator for synthetic data generation.

This module provides a unified interface for generating complete
synthetic oncology datasets including patients, treatments, and lab results.
"""

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import structlog

from src.synthetic_data.generators.lab_results_generator import LabResultsGenerator
from src.synthetic_data.generators.patient_generator import PatientGenerator
from src.synthetic_data.generators.treatment_generator import TreatmentGenerator
from src.synthetic_data.schemas.oncology_schemas import (
    CancerType,
    LabResultSchema,
    PatientSchema,
    TreatmentSchema,
)

logger = structlog.get_logger(__name__)


@dataclass
class GenerationConfig:
    """Configuration for synthetic data generation.

    Attributes:
        num_patients: Number of patients to generate.
        avg_treatments_per_patient: Average treatments per patient.
        avg_lab_results_per_patient: Average lab results per patient.
        cancer_types: List of cancer types to include.
        seed: Random seed for reproducibility.
        include_abnormal_patterns: Whether to include data quality issues.
        abnormal_rate: Rate of abnormal/erroneous records (0.0-1.0).
    """

    num_patients: int = 1000
    avg_treatments_per_patient: int = 3
    avg_lab_results_per_patient: int = 15
    cancer_types: list[CancerType] | None = None
    seed: int | None = None
    include_abnormal_patterns: bool = False
    abnormal_rate: float = 0.05


@dataclass
class GeneratedDataset:
    """Container for generated synthetic dataset.

    Attributes:
        patients: List of patient records.
        treatments: List of treatment records.
        lab_results: List of lab result records.
        generation_timestamp: When the data was generated.
        config: Configuration used for generation.
    """

    patients: list[PatientSchema]
    treatments: list[TreatmentSchema]
    lab_results: list[LabResultSchema]
    generation_timestamp: datetime
    config: GenerationConfig

    @property
    def patient_count(self) -> int:
        """Get number of patients."""
        return len(self.patients)

    @property
    def treatment_count(self) -> int:
        """Get number of treatments."""
        return len(self.treatments)

    @property
    def lab_result_count(self) -> int:
        """Get number of lab results."""
        return len(self.lab_results)

    def to_dataframes(self) -> dict[str, pd.DataFrame]:
        """Convert to pandas DataFrames.

        Returns:
            Dictionary with 'patients', 'treatments', 'lab_results' DataFrames.
        """
        return {
            "patients": pd.DataFrame([p.to_dict() for p in self.patients]),
            "treatments": pd.DataFrame([t.to_dict() for t in self.treatments]),
            "lab_results": pd.DataFrame([l.to_dict() for l in self.lab_results]),
        }

    def summary(self) -> dict[str, Any]:
        """Get summary statistics for the dataset.

        Returns:
            Dictionary with dataset statistics.
        """
        dfs = self.to_dataframes()

        return {
            "generation_timestamp": self.generation_timestamp.isoformat(),
            "patient_count": self.patient_count,
            "treatment_count": self.treatment_count,
            "lab_result_count": self.lab_result_count,
            "avg_treatments_per_patient": self.treatment_count / max(1, self.patient_count),
            "avg_lab_results_per_patient": self.lab_result_count / max(1, self.patient_count),
            "cancer_type_distribution": dfs["patients"]["cancer_type"].value_counts().to_dict(),
            "treatment_type_distribution": dfs["treatments"]["treatment_type"]
            .value_counts()
            .to_dict(),
            "abnormal_lab_rate": (
                dfs["lab_results"]["is_abnormal"].mean() if len(dfs["lab_results"]) > 0 else 0
            ),
        }


class OncologyDataFactory:
    """Factory for generating complete oncology datasets.

    This class orchestrates the generation of patients, treatments,
    and lab results to create realistic synthetic oncology datasets.

    Attributes:
        config: Generation configuration.
        patient_generator: Patient data generator.
        treatment_generator: Treatment data generator.
        lab_results_generator: Lab results data generator.

    Example:
        >>> factory = OncologyDataFactory(num_patients=500, seed=42)
        >>> dataset = factory.generate()
        >>> print(f"Generated {dataset.patient_count} patients")
        >>>
        >>> # Export to files
        >>> factory.export_to_csv(dataset, output_dir="data/synthetic")
    """

    def __init__(
        self,
        num_patients: int = 1000,
        avg_treatments: int = 3,
        avg_lab_results: int = 15,
        cancer_types: list[CancerType] | None = None,
        seed: int | None = None,
        include_abnormal_patterns: bool = False,
        abnormal_rate: float = 0.05,
    ) -> None:
        """Initialize the data factory.

        Args:
            num_patients: Number of patients to generate.
            avg_treatments: Average treatments per patient.
            avg_lab_results: Average lab results per patient.
            cancer_types: Cancer types to include (all if None).
            seed: Random seed for reproducibility.
            include_abnormal_patterns: Include data quality issues.
            abnormal_rate: Rate of abnormal records.
        """
        self.config = GenerationConfig(
            num_patients=num_patients,
            avg_treatments_per_patient=avg_treatments,
            avg_lab_results_per_patient=avg_lab_results,
            cancer_types=cancer_types,
            seed=seed,
            include_abnormal_patterns=include_abnormal_patterns,
            abnormal_rate=abnormal_rate,
        )

        self.patient_generator = PatientGenerator(seed=seed)
        self.treatment_generator = TreatmentGenerator(seed=seed)
        self.lab_results_generator = LabResultsGenerator(seed=seed)

        self._logger = logger.bind(factory="oncology_data", seed=seed)
        self._logger.info("OncologyDataFactory initialized", config=self.config)

    def generate(self) -> GeneratedDataset:
        """Generate a complete synthetic oncology dataset.

        Returns:
            GeneratedDataset containing all generated records.

        Example:
            >>> factory = OncologyDataFactory(num_patients=100)
            >>> dataset = factory.generate()
            >>> dfs = dataset.to_dataframes()
            >>> dfs["patients"].to_csv("patients.csv")
        """
        self._logger.info(
            "Starting dataset generation",
            num_patients=self.config.num_patients,
        )

        # Generate patients
        patients = self.patient_generator.generate(
            count=self.config.num_patients,
            cancer_types=self.config.cancer_types,
        )

        # Generate treatments for patients
        treatments = self.treatment_generator.generate_for_patients(
            patients=patients,
            avg_treatments=self.config.avg_treatments_per_patient,
        )

        # Generate lab results for patients
        lab_results = self.lab_results_generator.generate_for_patients(
            patients=patients,
            avg_results=self.config.avg_lab_results_per_patient,
        )

        # Optionally introduce data quality issues for testing
        if self.config.include_abnormal_patterns:
            patients, treatments, lab_results = self._introduce_data_issues(
                patients, treatments, lab_results
            )

        dataset = GeneratedDataset(
            patients=patients,
            treatments=treatments,
            lab_results=lab_results,
            generation_timestamp=datetime.now(),
            config=self.config,
        )

        self._logger.info(
            "Dataset generation complete",
            patients=dataset.patient_count,
            treatments=dataset.treatment_count,
            lab_results=dataset.lab_result_count,
        )

        return dataset

    def export_to_csv(
        self,
        dataset: GeneratedDataset,
        output_dir: str | Path,
        include_summary: bool = True,
    ) -> dict[str, Path]:
        """Export dataset to CSV files.

        Args:
            dataset: Generated dataset to export.
            output_dir: Directory to write files.
            include_summary: Whether to write summary JSON.

        Returns:
            Dictionary mapping entity names to file paths.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        dfs = dataset.to_dataframes()
        paths: dict[str, Path] = {}

        for name, df in dfs.items():
            file_path = output_dir / f"{name}.csv"
            df.to_csv(file_path, index=False)
            paths[name] = file_path
            self._logger.info(f"Exported {name}", path=str(file_path), rows=len(df))

        if include_summary:
            import json

            summary_path = output_dir / "generation_summary.json"
            with open(summary_path, "w") as f:
                json.dump(dataset.summary(), f, indent=2, default=str)
            paths["summary"] = summary_path
            self._logger.info("Exported summary", path=str(summary_path))

        return paths

    def export_to_parquet(
        self,
        dataset: GeneratedDataset,
        output_dir: str | Path,
    ) -> dict[str, Path]:
        """Export dataset to Parquet files.

        Args:
            dataset: Generated dataset to export.
            output_dir: Directory to write files.

        Returns:
            Dictionary mapping entity names to file paths.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        dfs = dataset.to_dataframes()
        paths: dict[str, Path] = {}

        for name, df in dfs.items():
            file_path = output_dir / f"{name}.parquet"
            df.to_parquet(file_path, index=False)
            paths[name] = file_path
            self._logger.info(f"Exported {name}", path=str(file_path), rows=len(df))

        return paths

    def _introduce_data_issues(
        self,
        patients: list[PatientSchema],
        treatments: list[TreatmentSchema],
        lab_results: list[LabResultSchema],
    ) -> tuple[list[PatientSchema], list[TreatmentSchema], list[LabResultSchema]]:
        """Introduce intentional data quality issues for testing.

        This method creates realistic data quality issues that can be
        caught by Great Expectations validations.

        Args:
            patients: Original patient list.
            treatments: Original treatment list.
            lab_results: Original lab results list.

        Returns:
            Tuple of modified lists with data issues.
        """
        import copy
        import random

        rate = self.config.abnormal_rate
        self._logger.info("Introducing data quality issues", rate=rate)

        # Create mutable copies
        patients = [copy.copy(p) for p in patients]
        treatments = [copy.copy(t) for t in treatments]
        lab_results = [copy.copy(l) for l in lab_results]

        issues_introduced = 0

        # Patient data issues
        for patient in patients:
            if random.random() < rate:
                issue_type = random.choice(["invalid_mrn", "future_date", "invalid_code"])

                if issue_type == "invalid_mrn":
                    # Create invalid MRN format
                    object.__setattr__(patient, "mrn", "INVALID")
                elif issue_type == "future_date":
                    # Future diagnosis date
                    from datetime import date, timedelta

                    future_date = date.today() + timedelta(days=30)
                    object.__setattr__(patient, "primary_diagnosis_date", future_date)
                elif issue_type == "invalid_code":
                    # Invalid ICD-10 code
                    object.__setattr__(patient, "primary_diagnosis_code", "INVALID")

                issues_introduced += 1

        # Treatment data issues
        for treatment in treatments:
            if random.random() < rate:
                issue_type = random.choice(["negative_dosage", "end_before_start"])

                if issue_type == "negative_dosage":
                    object.__setattr__(treatment, "dosage", -100.0)
                elif issue_type == "end_before_start":
                    # End date before start date
                    from datetime import timedelta

                    bad_end = treatment.start_date - timedelta(days=10)
                    object.__setattr__(treatment, "end_date", bad_end)

                issues_introduced += 1

        # Lab result data issues
        for result in lab_results:
            if random.random() < rate:
                issue_type = random.choice(["negative_value", "result_before_collection"])

                if issue_type == "negative_value":
                    object.__setattr__(result, "result_value", -999.0)
                elif issue_type == "result_before_collection":
                    # Result before collection
                    from datetime import timedelta

                    bad_result_time = result.collection_datetime - timedelta(hours=5)
                    object.__setattr__(result, "result_datetime", bad_result_time)

                issues_introduced += 1

        self._logger.info("Data issues introduced", count=issues_introduced)

        return patients, treatments, lab_results


# Convenience function for quick generation
def generate_oncology_dataset(
    num_patients: int = 1000,
    seed: int | None = None,
    output_dir: str | Path | None = None,
) -> GeneratedDataset:
    """Generate a synthetic oncology dataset.

    Convenience function for quick dataset generation.

    Args:
        num_patients: Number of patients to generate.
        seed: Random seed for reproducibility.
        output_dir: Optional directory to export CSV files.

    Returns:
        GeneratedDataset containing all generated records.

    Example:
        >>> dataset = generate_oncology_dataset(num_patients=500, seed=42)
        >>> print(f"Generated {dataset.patient_count} patients")
    """
    factory = OncologyDataFactory(num_patients=num_patients, seed=seed)
    dataset = factory.generate()

    if output_dir:
        factory.export_to_csv(dataset, output_dir)

    return dataset
