"""
Synthetic data generation for oncology datasets.

This package provides generators for creating realistic synthetic
oncology data for testing data quality pipelines.

Example:
    >>> from src.synthetic_data import OncologyDataFactory, generate_oncology_dataset
    >>>
    >>> # Quick generation
    >>> dataset = generate_oncology_dataset(num_patients=100, seed=42)
    >>> print(f"Generated {dataset.patient_count} patients")
    >>>
    >>> # Using factory for more control
    >>> factory = OncologyDataFactory(
    ...     num_patients=1000,
    ...     avg_treatments=3,
    ...     avg_lab_results=15,
    ...     seed=42,
    ... )
    >>> dataset = factory.generate()
    >>> factory.export_to_csv(dataset, "data/synthetic")
"""

from src.synthetic_data.generators.lab_results_generator import LabResultsGenerator
from src.synthetic_data.generators.patient_generator import PatientGenerator
from src.synthetic_data.generators.treatment_generator import TreatmentGenerator
from src.synthetic_data.oncology_data_factory import (
    GeneratedDataset,
    GenerationConfig,
    OncologyDataFactory,
    generate_oncology_dataset,
)
from src.synthetic_data.schemas.oncology_schemas import (
    CHEMOTHERAPY_DRUGS,
    ICD10_ONCOLOGY_CODES,
    ONCOLOGY_LAB_TESTS,
    CancerType,
    Ethnicity,
    Gender,
    LabResultSchema,
    LabTestCategory,
    PatientSchema,
    Race,
    TreatmentSchema,
    TreatmentStatus,
    TreatmentType,
)

__all__ = [
    # Factory
    "OncologyDataFactory",
    "GeneratedDataset",
    "GenerationConfig",
    "generate_oncology_dataset",
    # Generators
    "PatientGenerator",
    "TreatmentGenerator",
    "LabResultsGenerator",
    # Schemas
    "PatientSchema",
    "TreatmentSchema",
    "LabResultSchema",
    # Enums
    "Gender",
    "Ethnicity",
    "Race",
    "CancerType",
    "TreatmentType",
    "TreatmentStatus",
    "LabTestCategory",
    # Reference data
    "ICD10_ONCOLOGY_CODES",
    "CHEMOTHERAPY_DRUGS",
    "ONCOLOGY_LAB_TESTS",
]
