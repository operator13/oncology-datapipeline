"""
Data schemas for synthetic oncology data generation.

This package contains Pydantic models and reference data for
generating realistic oncology data.
"""

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
    # Enums
    "Gender",
    "Ethnicity",
    "Race",
    "CancerType",
    "TreatmentType",
    "TreatmentStatus",
    "LabTestCategory",
    # Schemas
    "PatientSchema",
    "TreatmentSchema",
    "LabResultSchema",
    # Reference data
    "ICD10_ONCOLOGY_CODES",
    "CHEMOTHERAPY_DRUGS",
    "ONCOLOGY_LAB_TESTS",
]
