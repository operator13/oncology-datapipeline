"""
Synthetic patient data generator for oncology datasets.

This module generates realistic patient demographic and diagnosis
data for testing oncology data pipelines.
"""

import random
from datetime import date, timedelta
from typing import Any
from uuid import UUID

import structlog
from faker import Faker

from src.synthetic_data.schemas.oncology_schemas import (
    ICD10_ONCOLOGY_CODES,
    CancerType,
    Ethnicity,
    Gender,
    PatientSchema,
    Race,
)

logger = structlog.get_logger(__name__)


class PatientGenerator:
    """Generator for synthetic oncology patient data.

    This generator creates realistic patient records with demographics,
    contact information, and oncology diagnosis details.

    Attributes:
        faker: Faker instance for generating random data.
        seed: Random seed for reproducibility.

    Example:
        >>> generator = PatientGenerator(seed=42)
        >>> patients = generator.generate(count=100)
        >>> print(f"Generated {len(patients)} patients")
    """

    # Age distribution weights for oncology patients (skewed toward older ages)
    AGE_WEIGHTS: list[tuple[int, int, float]] = [
        (18, 30, 0.05),   # 5% - Young adults
        (31, 45, 0.10),   # 10% - Middle-aged
        (46, 60, 0.25),   # 25% - Late middle-aged
        (61, 75, 0.40),   # 40% - Senior
        (76, 90, 0.20),   # 20% - Elderly
    ]

    # Cancer stage distribution
    STAGE_DISTRIBUTION: dict[str, float] = {
        "I": 0.20,
        "IA": 0.05,
        "IB": 0.05,
        "II": 0.15,
        "IIA": 0.05,
        "IIB": 0.05,
        "III": 0.15,
        "IIIA": 0.05,
        "IIIB": 0.05,
        "IV": 0.15,
        "IVA": 0.03,
        "IVB": 0.02,
    }

    # Gender distribution by cancer type
    CANCER_GENDER_DISTRIBUTION: dict[CancerType, dict[Gender, float]] = {
        CancerType.BREAST: {Gender.FEMALE: 0.99, Gender.MALE: 0.01},
        CancerType.PROSTATE: {Gender.MALE: 1.0},
        CancerType.OVARIAN: {Gender.FEMALE: 1.0},
        CancerType.LUNG: {Gender.MALE: 0.55, Gender.FEMALE: 0.45},
        CancerType.COLORECTAL: {Gender.MALE: 0.52, Gender.FEMALE: 0.48},
        CancerType.MELANOMA: {Gender.MALE: 0.55, Gender.FEMALE: 0.45},
        CancerType.LYMPHOMA: {Gender.MALE: 0.55, Gender.FEMALE: 0.45},
        CancerType.LEUKEMIA: {Gender.MALE: 0.55, Gender.FEMALE: 0.45},
        CancerType.PANCREATIC: {Gender.MALE: 0.52, Gender.FEMALE: 0.48},
        CancerType.BLADDER: {Gender.MALE: 0.75, Gender.FEMALE: 0.25},
    }

    def __init__(self, seed: int | None = None, locale: str = "en_US") -> None:
        """Initialize the patient generator.

        Args:
            seed: Random seed for reproducibility.
            locale: Faker locale for generating names/addresses.
        """
        self.seed = seed
        self.faker = Faker(locale)

        if seed is not None:
            Faker.seed(seed)
            random.seed(seed)

        self._logger = logger.bind(generator="patient", seed=seed)
        self._logger.info("Patient generator initialized")

    def generate(
        self,
        count: int = 100,
        cancer_types: list[CancerType] | None = None,
    ) -> list[PatientSchema]:
        """Generate synthetic patient records.

        Args:
            count: Number of patients to generate.
            cancer_types: Optional list of cancer types to use.
                         Uses all types if None.

        Returns:
            List of PatientSchema objects.

        Example:
            >>> generator = PatientGenerator()
            >>> patients = generator.generate(count=50, cancer_types=[CancerType.BREAST])
        """
        if cancer_types is None:
            cancer_types = list(CancerType)

        self._logger.info("Generating patients", count=count, cancer_types=len(cancer_types))

        patients: list[PatientSchema] = []

        for i in range(count):
            try:
                patient = self._generate_single_patient(cancer_types)
                patients.append(patient)

                if (i + 1) % 100 == 0:
                    self._logger.debug("Generation progress", generated=i + 1, total=count)

            except Exception as e:
                self._logger.error("Failed to generate patient", index=i, error=str(e))
                raise

        self._logger.info("Patient generation complete", count=len(patients))
        return patients

    def generate_as_dicts(
        self,
        count: int = 100,
        cancer_types: list[CancerType] | None = None,
    ) -> list[dict[str, Any]]:
        """Generate patients and return as list of dictionaries.

        Args:
            count: Number of patients to generate.
            cancer_types: Optional list of cancer types.

        Returns:
            List of dictionaries suitable for DataFrame creation.
        """
        patients = self.generate(count=count, cancer_types=cancer_types)
        return [p.to_dict() for p in patients]

    def _generate_single_patient(self, cancer_types: list[CancerType]) -> PatientSchema:
        """Generate a single patient record.

        Args:
            cancer_types: List of cancer types to choose from.

        Returns:
            PatientSchema instance.
        """
        # Select cancer type first (affects gender distribution)
        cancer_type = random.choice(cancer_types)

        # Generate gender based on cancer type
        gender = self._generate_gender(cancer_type)

        # Generate age based on oncology demographics
        age = self._generate_age()
        date_of_birth = date.today() - timedelta(days=age * 365 + random.randint(0, 364))

        # Generate diagnosis date (within last 5 years)
        days_since_diagnosis = random.randint(30, 5 * 365)
        diagnosis_date = date.today() - timedelta(days=days_since_diagnosis)

        # Ensure diagnosis date is after birth
        if diagnosis_date <= date_of_birth:
            diagnosis_date = date_of_birth + timedelta(days=365 * 18)  # At least 18 years old

        # Generate ICD-10 code for cancer type
        icd10_codes = ICD10_ONCOLOGY_CODES.get(cancer_type, ["C80.1"])
        diagnosis_code = random.choice(icd10_codes)

        # Generate cancer stage
        stage = self._generate_stage()

        # Generate demographics
        race = self._generate_race()
        ethnicity = self._generate_ethnicity()

        # Generate name based on gender
        if gender == Gender.MALE:
            first_name = self.faker.first_name_male()
        elif gender == Gender.FEMALE:
            first_name = self.faker.first_name_female()
        else:
            first_name = self.faker.first_name()

        return PatientSchema(
            mrn=self._generate_mrn(),
            first_name=first_name,
            last_name=self.faker.last_name(),
            date_of_birth=date_of_birth,
            gender=gender,
            race=race,
            ethnicity=ethnicity,
            address_line1=self.faker.street_address(),
            city=self.faker.city(),
            state=self.faker.state_abbr(),
            zip_code=self.faker.zipcode(),
            phone=self.faker.numerify("###-###-####"),
            email=self.faker.email(),
            primary_diagnosis_code=diagnosis_code,
            primary_diagnosis_date=diagnosis_date,
            cancer_type=cancer_type,
            cancer_stage=stage,
        )

    def _generate_mrn(self) -> str:
        """Generate a Medical Record Number."""
        # Format: MRN + 8 digits
        return f"MRN{self.faker.numerify('########')}"

    def _generate_age(self) -> int:
        """Generate age based on oncology demographics."""
        # Select age range based on weights
        ranges = [(r[0], r[1]) for r in self.AGE_WEIGHTS]
        weights = [r[2] for r in self.AGE_WEIGHTS]

        selected_range = random.choices(ranges, weights=weights, k=1)[0]
        return random.randint(selected_range[0], selected_range[1])

    def _generate_gender(self, cancer_type: CancerType) -> Gender:
        """Generate gender based on cancer type distribution."""
        distribution = self.CANCER_GENDER_DISTRIBUTION.get(
            cancer_type,
            {Gender.MALE: 0.5, Gender.FEMALE: 0.5},
        )

        genders = list(distribution.keys())
        weights = list(distribution.values())

        return random.choices(genders, weights=weights, k=1)[0]

    def _generate_stage(self) -> str:
        """Generate cancer stage based on distribution."""
        stages = list(self.STAGE_DISTRIBUTION.keys())
        weights = list(self.STAGE_DISTRIBUTION.values())
        return random.choices(stages, weights=weights, k=1)[0]

    def _generate_race(self) -> Race:
        """Generate race based on US demographics."""
        distribution = {
            Race.WHITE: 0.60,
            Race.BLACK: 0.13,
            Race.ASIAN: 0.06,
            Race.NATIVE_AMERICAN: 0.01,
            Race.PACIFIC_ISLANDER: 0.002,
            Race.OTHER: 0.05,
            Race.UNKNOWN: 0.048,
        }

        races = list(distribution.keys())
        weights = list(distribution.values())
        return random.choices(races, weights=weights, k=1)[0]

    def _generate_ethnicity(self) -> Ethnicity:
        """Generate ethnicity based on US demographics."""
        distribution = {
            Ethnicity.NON_HISPANIC: 0.82,
            Ethnicity.HISPANIC: 0.15,
            Ethnicity.UNKNOWN: 0.03,
        }

        ethnicities = list(distribution.keys())
        weights = list(distribution.values())
        return random.choices(ethnicities, weights=weights, k=1)[0]

    def get_patient_ids(self, patients: list[PatientSchema]) -> list[UUID]:
        """Extract patient IDs from a list of patients.

        Args:
            patients: List of PatientSchema objects.

        Returns:
            List of patient UUIDs.
        """
        return [p.patient_id for p in patients]
