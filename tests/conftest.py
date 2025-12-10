"""
Pytest configuration and fixtures for oncology data pipeline tests.

This module provides reusable fixtures for unit, integration,
and end-to-end tests.
"""

from datetime import date, datetime
from typing import Generator
from uuid import uuid4

import pandas as pd
import pytest

from src.synthetic_data import (
    CancerType,
    Gender,
    OncologyDataFactory,
    PatientGenerator,
    TreatmentGenerator,
    LabResultsGenerator,
)


# =============================================================================
# Session-scoped fixtures (created once per test session)
# =============================================================================


@pytest.fixture(scope="session")
def synthetic_factory() -> OncologyDataFactory:
    """Create a synthetic data factory with fixed seed."""
    return OncologyDataFactory(num_patients=100, seed=42)


@pytest.fixture(scope="session")
def sample_dataset(synthetic_factory: OncologyDataFactory):
    """Generate a sample dataset for testing."""
    return synthetic_factory.generate()


# =============================================================================
# Function-scoped fixtures (created fresh for each test)
# =============================================================================


@pytest.fixture
def patient_generator() -> PatientGenerator:
    """Create a patient generator with fixed seed."""
    return PatientGenerator(seed=42)


@pytest.fixture
def treatment_generator() -> TreatmentGenerator:
    """Create a treatment generator with fixed seed."""
    return TreatmentGenerator(seed=42)


@pytest.fixture
def lab_results_generator() -> LabResultsGenerator:
    """Create a lab results generator with fixed seed."""
    return LabResultsGenerator(seed=42)


@pytest.fixture
def sample_patients_df(patient_generator: PatientGenerator) -> pd.DataFrame:
    """Generate sample patient DataFrame."""
    patients = patient_generator.generate(count=50)
    return pd.DataFrame([p.to_dict() for p in patients])


@pytest.fixture
def sample_treatments_df(
    patient_generator: PatientGenerator,
    treatment_generator: TreatmentGenerator,
) -> pd.DataFrame:
    """Generate sample treatments DataFrame."""
    patients = patient_generator.generate(count=20)
    treatments = treatment_generator.generate_for_patients(patients, avg_treatments=2)
    return pd.DataFrame([t.to_dict() for t in treatments])


@pytest.fixture
def sample_lab_results_df(
    patient_generator: PatientGenerator,
    lab_results_generator: LabResultsGenerator,
) -> pd.DataFrame:
    """Generate sample lab results DataFrame."""
    patients = patient_generator.generate(count=20)
    results = lab_results_generator.generate_for_patients(patients, avg_results=10)
    return pd.DataFrame([r.to_dict() for r in results])


# =============================================================================
# Minimal data fixtures
# =============================================================================


@pytest.fixture
def minimal_patient_data() -> dict:
    """Minimal valid patient data."""
    return {
        "patient_id": str(uuid4()),
        "mrn": "MRN12345678",
        "first_name": "John",
        "last_name": "Doe",
        "date_of_birth": date(1960, 1, 15),
        "gender": "M",
        "race": "W",
        "ethnicity": "N",
        "address_line1": "123 Main St",
        "city": "Boston",
        "state": "MA",
        "zip_code": "02101",
        "phone": "555-123-4567",
        "email": "john.doe@example.com",
        "primary_diagnosis_code": "C50.1",
        "primary_diagnosis_date": date(2023, 6, 15),
        "cancer_type": "breast",
        "cancer_stage": "II",
        "created_at": datetime.now(),
        "updated_at": datetime.now(),
    }


@pytest.fixture
def minimal_treatment_data() -> dict:
    """Minimal valid treatment data."""
    return {
        "treatment_id": str(uuid4()),
        "patient_id": str(uuid4()),
        "treatment_type": "chemotherapy",
        "treatment_name": "AC-T Protocol",
        "drug_code": "00015-3475-30",
        "drug_name": "Paclitaxel",
        "dosage": 175.0,
        "dosage_unit": "mg/m2",
        "route": "IV",
        "start_date": date(2023, 7, 1),
        "end_date": date(2023, 10, 1),
        "status": "completed",
        "cycles_planned": 6,
        "cycles_completed": 6,
        "treating_physician": "Dr. Smith",
        "facility_name": "Cancer Center",
        "notes": None,
        "created_at": datetime.now(),
        "updated_at": datetime.now(),
    }


@pytest.fixture
def minimal_lab_result_data() -> dict:
    """Minimal valid lab result data."""
    return {
        "result_id": str(uuid4()),
        "patient_id": str(uuid4()),
        "test_code": "6690-2",
        "test_name": "WBC",
        "test_category": "hematology",
        "result_value": 7.5,
        "result_unit": "10^3/uL",
        "result_text": None,
        "reference_range_low": 4.5,
        "reference_range_high": 11.0,
        "is_abnormal": False,
        "is_critical": False,
        "specimen_type": "Whole Blood",
        "collection_datetime": datetime(2023, 8, 1, 9, 0),
        "result_datetime": datetime(2023, 8, 1, 12, 0),
        "performing_lab": "Quest Diagnostics",
        "ordering_physician": "Dr. Johnson",
        "created_at": datetime.now(),
    }


# =============================================================================
# Invalid data fixtures for negative testing
# =============================================================================


@pytest.fixture
def invalid_patient_data() -> dict:
    """Invalid patient data for testing validation."""
    return {
        "patient_id": str(uuid4()),
        "mrn": "INVALID",  # Invalid format
        "first_name": "",  # Empty
        "last_name": "Doe",
        "date_of_birth": date(2099, 1, 1),  # Future date
        "gender": "X",  # Invalid
        "race": "W",
        "ethnicity": "N",
        "address_line1": "123 Main St",
        "city": "Boston",
        "state": "Massachusetts",  # Should be 2 letters
        "zip_code": "invalid",  # Invalid format
        "primary_diagnosis_code": "INVALID",  # Invalid ICD-10
        "primary_diagnosis_date": date(2099, 12, 31),  # Future
        "cancer_type": "unknown",  # Invalid
        "cancer_stage": "V",  # Invalid
        "created_at": datetime.now(),
        "updated_at": datetime.now(),
    }


# =============================================================================
# DataFrame fixtures with specific scenarios
# =============================================================================


@pytest.fixture
def df_with_nulls() -> pd.DataFrame:
    """DataFrame with null values for completeness testing."""
    return pd.DataFrame({
        "id": [1, 2, 3, 4, 5],
        "name": ["Alice", None, "Charlie", None, "Eve"],
        "value": [100, 200, None, 400, None],
        "date": [date(2023, 1, 1), date(2023, 2, 1), None, date(2023, 4, 1), date(2023, 5, 1)],
    })


@pytest.fixture
def df_with_duplicates() -> pd.DataFrame:
    """DataFrame with duplicate values for uniqueness testing."""
    return pd.DataFrame({
        "id": [1, 2, 2, 4, 5],  # Duplicate ID
        "name": ["Alice", "Bob", "Bob", "Diana", "Eve"],
        "value": [100, 200, 200, 400, 500],
    })


@pytest.fixture
def df_with_outliers() -> pd.DataFrame:
    """DataFrame with outliers for anomaly detection testing."""
    return pd.DataFrame({
        "id": range(100),
        "normal_value": [50 + i % 10 for i in range(100)],
        "with_outliers": [50 + i % 10 if i < 95 else 1000 for i in range(100)],
    })


# =============================================================================
# Mock connector fixtures
# =============================================================================


@pytest.fixture
def mock_connector():
    """Create a mock database connector."""
    from unittest.mock import MagicMock

    connector = MagicMock()
    connector.is_connected = True
    connector.backend_name = "mock"
    connector.execute_query.return_value = pd.DataFrame({"test": [1, 2, 3]})
    connector.table_exists.return_value = True
    return connector


# =============================================================================
# Pytest markers
# =============================================================================


def pytest_configure(config):
    """Configure custom pytest markers."""
    config.addinivalue_line("markers", "unit: Unit tests")
    config.addinivalue_line("markers", "integration: Integration tests")
    config.addinivalue_line("markers", "e2e: End-to-end tests")
    config.addinivalue_line("markers", "slow: Slow tests")
    config.addinivalue_line("markers", "databricks: Requires Databricks")
    config.addinivalue_line("markers", "sqlserver: Requires SQL Server")
