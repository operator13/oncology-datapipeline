"""
Data schemas for oncology domain entities.

This module defines Pydantic models for all oncology data entities,
providing validation, serialization, and documentation.
"""

from datetime import date, datetime
from enum import Enum
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator


# =============================================================================
# Enumerations
# =============================================================================


class Gender(str, Enum):
    """Patient gender enumeration."""

    MALE = "M"
    FEMALE = "F"
    OTHER = "O"
    UNKNOWN = "U"


class Ethnicity(str, Enum):
    """Patient ethnicity enumeration (HL7 standard)."""

    HISPANIC = "H"
    NON_HISPANIC = "N"
    UNKNOWN = "U"


class Race(str, Enum):
    """Patient race enumeration (HL7 standard)."""

    WHITE = "W"
    BLACK = "B"
    ASIAN = "A"
    NATIVE_AMERICAN = "N"
    PACIFIC_ISLANDER = "P"
    OTHER = "O"
    UNKNOWN = "U"


class CancerType(str, Enum):
    """Common oncology cancer types."""

    BREAST = "breast"
    LUNG = "lung"
    COLORECTAL = "colorectal"
    PROSTATE = "prostate"
    MELANOMA = "melanoma"
    LYMPHOMA = "lymphoma"
    LEUKEMIA = "leukemia"
    PANCREATIC = "pancreatic"
    OVARIAN = "ovarian"
    BLADDER = "bladder"


class TreatmentType(str, Enum):
    """Types of oncology treatments."""

    CHEMOTHERAPY = "chemotherapy"
    RADIATION = "radiation"
    SURGERY = "surgery"
    IMMUNOTHERAPY = "immunotherapy"
    TARGETED_THERAPY = "targeted_therapy"
    HORMONE_THERAPY = "hormone_therapy"


class TreatmentStatus(str, Enum):
    """Treatment status enumeration."""

    PLANNED = "planned"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    ON_HOLD = "on_hold"


class LabTestCategory(str, Enum):
    """Categories of laboratory tests."""

    HEMATOLOGY = "hematology"
    CHEMISTRY = "chemistry"
    TUMOR_MARKER = "tumor_marker"
    COAGULATION = "coagulation"
    URINALYSIS = "urinalysis"


# =============================================================================
# Patient Schema
# =============================================================================


class PatientSchema(BaseModel):
    """Schema for patient demographic data.

    Attributes:
        patient_id: Unique patient identifier (UUID).
        mrn: Medical Record Number.
        first_name: Patient's first name.
        last_name: Patient's last name.
        date_of_birth: Patient's date of birth.
        gender: Patient's gender.
        race: Patient's race.
        ethnicity: Patient's ethnicity.
        address_line1: Street address.
        city: City.
        state: State code (2 letters).
        zip_code: ZIP code.
        phone: Phone number.
        email: Email address.
        primary_diagnosis_code: ICD-10 code for primary cancer diagnosis.
        primary_diagnosis_date: Date of primary diagnosis.
        cancer_type: Type of cancer.
        cancer_stage: Cancer stage (I, II, III, IV).
        created_at: Record creation timestamp.
        updated_at: Record update timestamp.
    """

    model_config = ConfigDict(str_strip_whitespace=True)

    patient_id: UUID = Field(default_factory=uuid4, description="Unique patient identifier")
    mrn: str = Field(..., min_length=6, max_length=20, description="Medical Record Number")
    first_name: str = Field(..., min_length=1, max_length=100)
    last_name: str = Field(..., min_length=1, max_length=100)
    date_of_birth: date = Field(..., description="Patient date of birth")
    gender: Gender = Field(..., description="Patient gender")
    race: Race = Field(default=Race.UNKNOWN)
    ethnicity: Ethnicity = Field(default=Ethnicity.UNKNOWN)

    # Address
    address_line1: str = Field(..., max_length=200)
    city: str = Field(..., max_length=100)
    state: str = Field(..., min_length=2, max_length=2)
    zip_code: str = Field(..., pattern=r"^\d{5}(-\d{4})?$")

    # Contact
    phone: str | None = Field(default=None, pattern=r"^\d{3}-\d{3}-\d{4}$")
    email: str | None = Field(default=None, max_length=255)

    # Diagnosis
    primary_diagnosis_code: str = Field(
        ..., pattern=r"^[A-Z]\d{2}(\.\d{1,4})?$", description="ICD-10 diagnosis code"
    )
    primary_diagnosis_date: date = Field(..., description="Date of primary diagnosis")
    cancer_type: CancerType = Field(..., description="Type of cancer")
    cancer_stage: str = Field(..., pattern=r"^(I|II|III|IV)(A|B|C)?$")

    # Metadata
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    @field_validator("date_of_birth")
    @classmethod
    def validate_dob(cls, v: date) -> date:
        """Ensure date of birth is not in the future."""
        if v > date.today():
            raise ValueError("Date of birth cannot be in the future")
        return v

    @field_validator("primary_diagnosis_date")
    @classmethod
    def validate_diagnosis_date(cls, v: date) -> date:
        """Ensure diagnosis date is not in the future."""
        if v > date.today():
            raise ValueError("Diagnosis date cannot be in the future")
        return v

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary with string UUIDs."""
        data = self.model_dump()
        data["patient_id"] = str(data["patient_id"])
        data["gender"] = data["gender"].value
        data["race"] = data["race"].value
        data["ethnicity"] = data["ethnicity"].value
        data["cancer_type"] = data["cancer_type"].value
        return data


# =============================================================================
# Treatment Schema
# =============================================================================


class TreatmentSchema(BaseModel):
    """Schema for treatment records.

    Attributes:
        treatment_id: Unique treatment identifier.
        patient_id: Reference to patient.
        treatment_type: Type of treatment.
        treatment_name: Specific treatment name/protocol.
        drug_code: NDC code for drugs (if applicable).
        drug_name: Drug name (if applicable).
        dosage: Dosage amount.
        dosage_unit: Unit of dosage.
        route: Administration route.
        start_date: Treatment start date.
        end_date: Treatment end date.
        status: Current treatment status.
        cycles_planned: Number of planned cycles.
        cycles_completed: Number of completed cycles.
        treating_physician: Physician name.
        facility_name: Treatment facility.
        notes: Additional notes.
        created_at: Record creation timestamp.
        updated_at: Record update timestamp.
    """

    model_config = ConfigDict(str_strip_whitespace=True)

    treatment_id: UUID = Field(default_factory=uuid4)
    patient_id: UUID = Field(..., description="Reference to patient")
    treatment_type: TreatmentType = Field(...)
    treatment_name: str = Field(..., max_length=200)

    # Drug information (for chemotherapy/targeted therapy)
    drug_code: str | None = Field(
        default=None, pattern=r"^\d{5}-\d{4}-\d{2}$", description="NDC code"
    )
    drug_name: str | None = Field(default=None, max_length=200)
    dosage: float | None = Field(default=None, ge=0)
    dosage_unit: str | None = Field(default=None, max_length=20)
    route: str | None = Field(default=None, max_length=50)

    # Treatment timeline
    start_date: date = Field(...)
    end_date: date | None = Field(default=None)
    status: TreatmentStatus = Field(default=TreatmentStatus.PLANNED)

    # Cycle information
    cycles_planned: int | None = Field(default=None, ge=1)
    cycles_completed: int | None = Field(default=None, ge=0)

    # Provider information
    treating_physician: str = Field(..., max_length=200)
    facility_name: str = Field(..., max_length=200)

    notes: str | None = Field(default=None, max_length=2000)

    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    @field_validator("end_date")
    @classmethod
    def validate_end_date(cls, v: date | None, info: Any) -> date | None:
        """Ensure end date is after start date."""
        if v is not None and "start_date" in info.data:
            if v < info.data["start_date"]:
                raise ValueError("End date must be after start date")
        return v

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary with string UUIDs."""
        data = self.model_dump()
        data["treatment_id"] = str(data["treatment_id"])
        data["patient_id"] = str(data["patient_id"])
        data["treatment_type"] = data["treatment_type"].value
        data["status"] = data["status"].value
        return data


# =============================================================================
# Lab Results Schema
# =============================================================================


class LabResultSchema(BaseModel):
    """Schema for laboratory test results.

    Attributes:
        result_id: Unique result identifier.
        patient_id: Reference to patient.
        test_code: LOINC code for the test.
        test_name: Name of the test.
        test_category: Category of test.
        result_value: Numeric result value.
        result_unit: Unit of measurement.
        result_text: Text result (for non-numeric).
        reference_range_low: Lower bound of normal range.
        reference_range_high: Upper bound of normal range.
        is_abnormal: Flag for abnormal results.
        is_critical: Flag for critical results.
        specimen_type: Type of specimen.
        collection_datetime: When specimen was collected.
        result_datetime: When result was reported.
        performing_lab: Laboratory name.
        ordering_physician: Ordering physician name.
        created_at: Record creation timestamp.
    """

    model_config = ConfigDict(str_strip_whitespace=True)

    result_id: UUID = Field(default_factory=uuid4)
    patient_id: UUID = Field(..., description="Reference to patient")

    # Test identification
    test_code: str = Field(..., max_length=20, description="LOINC code")
    test_name: str = Field(..., max_length=200)
    test_category: LabTestCategory = Field(...)

    # Result values
    result_value: float | None = Field(default=None)
    result_unit: str | None = Field(default=None, max_length=50)
    result_text: str | None = Field(default=None, max_length=500)

    # Reference ranges
    reference_range_low: float | None = Field(default=None)
    reference_range_high: float | None = Field(default=None)

    # Flags
    is_abnormal: bool = Field(default=False)
    is_critical: bool = Field(default=False)

    # Specimen and timing
    specimen_type: str = Field(..., max_length=100)
    collection_datetime: datetime = Field(...)
    result_datetime: datetime = Field(...)

    # Provider information
    performing_lab: str = Field(..., max_length=200)
    ordering_physician: str = Field(..., max_length=200)

    created_at: datetime = Field(default_factory=datetime.utcnow)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary with string UUIDs."""
        data = self.model_dump()
        data["result_id"] = str(data["result_id"])
        data["patient_id"] = str(data["patient_id"])
        data["test_category"] = data["test_category"].value
        return data


# =============================================================================
# Reference Data
# =============================================================================

# ICD-10 codes for common oncology diagnoses
ICD10_ONCOLOGY_CODES: dict[CancerType, list[str]] = {
    CancerType.BREAST: ["C50.0", "C50.1", "C50.2", "C50.3", "C50.4", "C50.5", "C50.9"],
    CancerType.LUNG: ["C34.0", "C34.1", "C34.2", "C34.3", "C34.9"],
    CancerType.COLORECTAL: ["C18.0", "C18.2", "C18.7", "C19", "C20"],
    CancerType.PROSTATE: ["C61"],
    CancerType.MELANOMA: ["C43.0", "C43.3", "C43.4", "C43.5", "C43.9"],
    CancerType.LYMPHOMA: ["C81.0", "C81.1", "C82.0", "C83.0", "C85.9"],
    CancerType.LEUKEMIA: ["C91.0", "C91.1", "C92.0", "C92.1", "C95.0"],
    CancerType.PANCREATIC: ["C25.0", "C25.1", "C25.2", "C25.9"],
    CancerType.OVARIAN: ["C56.1", "C56.2", "C56.9"],
    CancerType.BLADDER: ["C67.0", "C67.1", "C67.2", "C67.9"],
}

# Common chemotherapy drugs with NDC codes
CHEMOTHERAPY_DRUGS: list[dict[str, str]] = [
    {"name": "Paclitaxel", "ndc": "00015-3475-30", "route": "IV"},
    {"name": "Carboplatin", "ndc": "00015-3214-30", "route": "IV"},
    {"name": "Doxorubicin", "ndc": "00069-3030-20", "route": "IV"},
    {"name": "Cyclophosphamide", "ndc": "00015-0503-41", "route": "IV"},
    {"name": "Cisplatin", "ndc": "00015-3220-22", "route": "IV"},
    {"name": "Gemcitabine", "ndc": "00002-7501-01", "route": "IV"},
    {"name": "Fluorouracil", "ndc": "63323-0117-10", "route": "IV"},
    {"name": "Methotrexate", "ndc": "00703-4250-01", "route": "IV"},
    {"name": "Vincristine", "ndc": "61703-0309-18", "route": "IV"},
    {"name": "Etoposide", "ndc": "00015-3063-20", "route": "IV"},
]

# Common oncology lab tests with LOINC codes and reference ranges
ONCOLOGY_LAB_TESTS: list[dict[str, Any]] = [
    # Hematology
    {"code": "6690-2", "name": "WBC", "category": "hematology", "unit": "10^3/uL", "low": 4.5, "high": 11.0},
    {"code": "789-8", "name": "RBC", "category": "hematology", "unit": "10^6/uL", "low": 4.5, "high": 5.5},
    {"code": "718-7", "name": "Hemoglobin", "category": "hematology", "unit": "g/dL", "low": 12.0, "high": 17.5},
    {"code": "4544-3", "name": "Hematocrit", "category": "hematology", "unit": "%", "low": 36.0, "high": 50.0},
    {"code": "777-3", "name": "Platelets", "category": "hematology", "unit": "10^3/uL", "low": 150.0, "high": 400.0},
    {"code": "751-8", "name": "Neutrophils", "category": "hematology", "unit": "%", "low": 40.0, "high": 70.0},
    {"code": "731-0", "name": "Lymphocytes", "category": "hematology", "unit": "%", "low": 20.0, "high": 40.0},
    # Chemistry
    {"code": "2160-0", "name": "Creatinine", "category": "chemistry", "unit": "mg/dL", "low": 0.6, "high": 1.2},
    {"code": "3094-0", "name": "BUN", "category": "chemistry", "unit": "mg/dL", "low": 7.0, "high": 20.0},
    {"code": "1742-6", "name": "ALT", "category": "chemistry", "unit": "U/L", "low": 7.0, "high": 56.0},
    {"code": "1920-8", "name": "AST", "category": "chemistry", "unit": "U/L", "low": 10.0, "high": 40.0},
    {"code": "1975-2", "name": "Bilirubin Total", "category": "chemistry", "unit": "mg/dL", "low": 0.1, "high": 1.2},
    {"code": "2951-2", "name": "Sodium", "category": "chemistry", "unit": "mEq/L", "low": 136.0, "high": 145.0},
    {"code": "2823-3", "name": "Potassium", "category": "chemistry", "unit": "mEq/L", "low": 3.5, "high": 5.0},
    # Tumor markers
    {"code": "2039-6", "name": "CEA", "category": "tumor_marker", "unit": "ng/mL", "low": 0.0, "high": 3.0},
    {"code": "10834-0", "name": "CA-125", "category": "tumor_marker", "unit": "U/mL", "low": 0.0, "high": 35.0},
    {"code": "19195-7", "name": "PSA", "category": "tumor_marker", "unit": "ng/mL", "low": 0.0, "high": 4.0},
    {"code": "83112-5", "name": "CA 19-9", "category": "tumor_marker", "unit": "U/mL", "low": 0.0, "high": 37.0},
    {"code": "31993-7", "name": "AFP", "category": "tumor_marker", "unit": "ng/mL", "low": 0.0, "high": 10.0},
]
