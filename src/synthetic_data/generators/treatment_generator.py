"""
Synthetic treatment data generator for oncology datasets.

This module generates realistic treatment records including chemotherapy,
radiation, surgery, and other oncology treatments.
"""

import random
from datetime import date, timedelta
from typing import Any
from uuid import UUID

import structlog
from faker import Faker

from src.synthetic_data.schemas.oncology_schemas import (
    CHEMOTHERAPY_DRUGS,
    CancerType,
    PatientSchema,
    TreatmentSchema,
    TreatmentStatus,
    TreatmentType,
)

logger = structlog.get_logger(__name__)


class TreatmentGenerator:
    """Generator for synthetic oncology treatment data.

    This generator creates realistic treatment records including
    chemotherapy regimens, radiation therapy, and surgical procedures.

    Attributes:
        faker: Faker instance for generating random data.
        seed: Random seed for reproducibility.

    Example:
        >>> generator = TreatmentGenerator(seed=42)
        >>> treatments = generator.generate_for_patients(patients, avg_treatments=3)
    """

    # Treatment type distribution by cancer type
    TREATMENT_DISTRIBUTION: dict[CancerType, dict[TreatmentType, float]] = {
        CancerType.BREAST: {
            TreatmentType.SURGERY: 0.30,
            TreatmentType.CHEMOTHERAPY: 0.30,
            TreatmentType.RADIATION: 0.20,
            TreatmentType.HORMONE_THERAPY: 0.15,
            TreatmentType.TARGETED_THERAPY: 0.05,
        },
        CancerType.LUNG: {
            TreatmentType.CHEMOTHERAPY: 0.35,
            TreatmentType.RADIATION: 0.25,
            TreatmentType.SURGERY: 0.20,
            TreatmentType.IMMUNOTHERAPY: 0.15,
            TreatmentType.TARGETED_THERAPY: 0.05,
        },
        CancerType.COLORECTAL: {
            TreatmentType.SURGERY: 0.35,
            TreatmentType.CHEMOTHERAPY: 0.35,
            TreatmentType.RADIATION: 0.15,
            TreatmentType.TARGETED_THERAPY: 0.10,
            TreatmentType.IMMUNOTHERAPY: 0.05,
        },
        CancerType.PROSTATE: {
            TreatmentType.SURGERY: 0.30,
            TreatmentType.RADIATION: 0.30,
            TreatmentType.HORMONE_THERAPY: 0.30,
            TreatmentType.CHEMOTHERAPY: 0.10,
        },
        CancerType.LYMPHOMA: {
            TreatmentType.CHEMOTHERAPY: 0.50,
            TreatmentType.IMMUNOTHERAPY: 0.20,
            TreatmentType.RADIATION: 0.15,
            TreatmentType.TARGETED_THERAPY: 0.10,
            TreatmentType.SURGERY: 0.05,
        },
    }

    # Chemotherapy protocol names by cancer type
    CHEMO_PROTOCOLS: dict[CancerType, list[str]] = {
        CancerType.BREAST: ["AC-T", "TC", "TAC", "CMF", "FEC"],
        CancerType.LUNG: ["Carboplatin/Paclitaxel", "Cisplatin/Etoposide", "Gemcitabine/Cisplatin"],
        CancerType.COLORECTAL: ["FOLFOX", "FOLFIRI", "CAPOX", "5-FU/Leucovorin"],
        CancerType.LYMPHOMA: ["R-CHOP", "ABVD", "R-CVP", "BEACOPP"],
        CancerType.LEUKEMIA: ["7+3", "HiDAC", "FLAG", "Hyper-CVAD"],
    }

    # Radiation protocols
    RADIATION_PROTOCOLS: list[str] = [
        "External Beam Radiation",
        "IMRT",
        "SBRT",
        "Proton Therapy",
        "Brachytherapy",
    ]

    # Surgical procedures by cancer type
    SURGICAL_PROCEDURES: dict[CancerType, list[str]] = {
        CancerType.BREAST: [
            "Lumpectomy",
            "Mastectomy",
            "Sentinel Node Biopsy",
            "Axillary Dissection",
        ],
        CancerType.LUNG: ["Lobectomy", "Pneumonectomy", "Wedge Resection", "VATS"],
        CancerType.COLORECTAL: ["Colectomy", "LAR", "APR", "Polypectomy"],
        CancerType.PROSTATE: ["Prostatectomy", "TURP", "Robotic Prostatectomy"],
        CancerType.MELANOMA: ["Wide Local Excision", "SLNB", "Lymph Node Dissection"],
    }

    # Facility names
    FACILITY_NAMES: list[str] = [
        "Memorial Cancer Center",
        "University Oncology Institute",
        "Regional Medical Center",
        "Cancer Treatment Associates",
        "Comprehensive Cancer Care",
        "Oncology Specialists Group",
        "Advanced Cancer Treatment Center",
        "Community Cancer Center",
    ]

    def __init__(self, seed: int | None = None, locale: str = "en_US") -> None:
        """Initialize the treatment generator.

        Args:
            seed: Random seed for reproducibility.
            locale: Faker locale for generating names.
        """
        self.seed = seed
        self.faker = Faker(locale)

        if seed is not None:
            Faker.seed(seed)
            random.seed(seed)

        self._logger = logger.bind(generator="treatment", seed=seed)
        self._logger.info("Treatment generator initialized")

    def generate_for_patients(
        self,
        patients: list[PatientSchema],
        avg_treatments: int = 3,
        variance: int = 2,
    ) -> list[TreatmentSchema]:
        """Generate treatments for a list of patients.

        Args:
            patients: List of patients to generate treatments for.
            avg_treatments: Average number of treatments per patient.
            variance: Variance in number of treatments.

        Returns:
            List of TreatmentSchema objects.

        Example:
            >>> treatments = generator.generate_for_patients(patients, avg_treatments=3)
        """
        self._logger.info(
            "Generating treatments",
            patient_count=len(patients),
            avg_treatments=avg_treatments,
        )

        treatments: list[TreatmentSchema] = []

        for patient in patients:
            # Determine number of treatments for this patient
            num_treatments = max(1, avg_treatments + random.randint(-variance, variance))

            patient_treatments = self._generate_patient_treatments(patient, num_treatments)
            treatments.extend(patient_treatments)

        self._logger.info("Treatment generation complete", count=len(treatments))
        return treatments

    def generate_as_dicts(
        self,
        patients: list[PatientSchema],
        avg_treatments: int = 3,
        variance: int = 2,
    ) -> list[dict[str, Any]]:
        """Generate treatments and return as list of dictionaries.

        Args:
            patients: List of patients.
            avg_treatments: Average treatments per patient.
            variance: Variance in treatment count.

        Returns:
            List of dictionaries suitable for DataFrame creation.
        """
        treatments = self.generate_for_patients(
            patients=patients,
            avg_treatments=avg_treatments,
            variance=variance,
        )
        return [t.to_dict() for t in treatments]

    def _generate_patient_treatments(
        self,
        patient: PatientSchema,
        num_treatments: int,
    ) -> list[TreatmentSchema]:
        """Generate treatments for a single patient.

        Args:
            patient: Patient to generate treatments for.
            num_treatments: Number of treatments to generate.

        Returns:
            List of treatments for the patient.
        """
        treatments: list[TreatmentSchema] = []

        # Get treatment distribution for cancer type
        distribution = self.TREATMENT_DISTRIBUTION.get(
            patient.cancer_type,
            {
                TreatmentType.CHEMOTHERAPY: 0.4,
                TreatmentType.RADIATION: 0.3,
                TreatmentType.SURGERY: 0.3,
            },
        )

        treatment_types = list(distribution.keys())
        weights = list(distribution.values())

        # Generate each treatment
        current_date = patient.primary_diagnosis_date + timedelta(days=random.randint(7, 30))

        for _ in range(num_treatments):
            treatment_type = random.choices(treatment_types, weights=weights, k=1)[0]

            treatment = self._generate_single_treatment(
                patient=patient,
                treatment_type=treatment_type,
                start_date=current_date,
            )
            treatments.append(treatment)

            # Next treatment starts after current one ends (or after a gap)
            if treatment.end_date:
                current_date = treatment.end_date + timedelta(days=random.randint(14, 60))
            else:
                current_date = treatment.start_date + timedelta(days=random.randint(30, 90))

        return treatments

    def _generate_single_treatment(
        self,
        patient: PatientSchema,
        treatment_type: TreatmentType,
        start_date: date,
    ) -> TreatmentSchema:
        """Generate a single treatment record.

        Args:
            patient: Patient receiving treatment.
            treatment_type: Type of treatment.
            start_date: Treatment start date.

        Returns:
            TreatmentSchema instance.
        """
        # Generate treatment-specific details
        treatment_name, drug_info, cycles_info = self._get_treatment_details(
            patient.cancer_type, treatment_type
        )

        # Calculate end date and status
        duration_days = self._get_treatment_duration(treatment_type)
        end_date = start_date + timedelta(days=duration_days)

        # Determine status based on dates
        today = date.today()
        if end_date < today:
            status = TreatmentStatus.COMPLETED
            cycles_completed = cycles_info.get("planned", 0) if cycles_info else None
        elif start_date > today:
            status = TreatmentStatus.PLANNED
            cycles_completed = 0
        else:
            status = TreatmentStatus.IN_PROGRESS
            if cycles_info:
                total_duration = (end_date - start_date).days
                elapsed = (today - start_date).days
                cycles_completed = int((elapsed / total_duration) * cycles_info.get("planned", 0))
            else:
                cycles_completed = None

        return TreatmentSchema(
            patient_id=patient.patient_id,
            treatment_type=treatment_type,
            treatment_name=treatment_name,
            drug_code=drug_info.get("ndc") if drug_info else None,
            drug_name=drug_info.get("name") if drug_info else None,
            dosage=drug_info.get("dosage") if drug_info else None,
            dosage_unit=drug_info.get("unit") if drug_info else None,
            route=drug_info.get("route") if drug_info else None,
            start_date=start_date,
            end_date=end_date if status == TreatmentStatus.COMPLETED else None,
            status=status,
            cycles_planned=cycles_info.get("planned") if cycles_info else None,
            cycles_completed=cycles_completed,
            treating_physician=self._generate_physician_name(),
            facility_name=random.choice(self.FACILITY_NAMES),
            notes=self._generate_treatment_notes(treatment_type),
        )

    def _get_treatment_details(
        self,
        cancer_type: CancerType,
        treatment_type: TreatmentType,
    ) -> tuple[str, dict[str, Any] | None, dict[str, int] | None]:
        """Get treatment-specific details.

        Args:
            cancer_type: Type of cancer.
            treatment_type: Type of treatment.

        Returns:
            Tuple of (treatment_name, drug_info, cycles_info).
        """
        drug_info: dict[str, Any] | None = None
        cycles_info: dict[str, int] | None = None

        if treatment_type == TreatmentType.CHEMOTHERAPY:
            protocols = self.CHEMO_PROTOCOLS.get(cancer_type, ["Standard Chemotherapy"])
            treatment_name = random.choice(protocols)

            # Select a drug
            drug = random.choice(CHEMOTHERAPY_DRUGS)
            drug_info = {
                "ndc": drug["ndc"],
                "name": drug["name"],
                "route": drug["route"],
                "dosage": round(random.uniform(50, 200), 1),
                "unit": "mg/m2",
            }
            cycles_info = {"planned": random.randint(4, 8)}

        elif treatment_type == TreatmentType.RADIATION:
            treatment_name = random.choice(self.RADIATION_PROTOCOLS)
            cycles_info = {"planned": random.randint(20, 35)}  # Fractions

        elif treatment_type == TreatmentType.SURGERY:
            procedures = self.SURGICAL_PROCEDURES.get(cancer_type, ["Surgical Resection"])
            treatment_name = random.choice(procedures)

        elif treatment_type == TreatmentType.IMMUNOTHERAPY:
            treatment_name = random.choice(
                [
                    "Pembrolizumab",
                    "Nivolumab",
                    "Atezolizumab",
                    "Ipilimumab",
                ]
            )
            drug_info = {
                "ndc": self.faker.numerify("#####-####-##"),
                "name": treatment_name,
                "route": "IV",
                "dosage": round(random.uniform(100, 400), 1),
                "unit": "mg",
            }
            cycles_info = {"planned": random.randint(6, 24)}

        elif treatment_type == TreatmentType.TARGETED_THERAPY:
            treatment_name = random.choice(
                [
                    "Trastuzumab",
                    "Bevacizumab",
                    "Cetuximab",
                    "Rituximab",
                ]
            )
            drug_info = {
                "ndc": self.faker.numerify("#####-####-##"),
                "name": treatment_name,
                "route": "IV",
                "dosage": round(random.uniform(300, 600), 1),
                "unit": "mg",
            }
            cycles_info = {"planned": random.randint(6, 18)}

        elif treatment_type == TreatmentType.HORMONE_THERAPY:
            treatment_name = random.choice(
                [
                    "Tamoxifen",
                    "Letrozole",
                    "Anastrozole",
                    "Lupron",
                ]
            )
            drug_info = {
                "ndc": self.faker.numerify("#####-####-##"),
                "name": treatment_name,
                "route": (
                    "PO" if treatment_name in ["Tamoxifen", "Letrozole", "Anastrozole"] else "IM"
                ),
                "dosage": round(random.uniform(10, 100), 1),
                "unit": "mg",
            }
        else:
            treatment_name = "Supportive Care"

        return treatment_name, drug_info, cycles_info

    def _get_treatment_duration(self, treatment_type: TreatmentType) -> int:
        """Get typical treatment duration in days.

        Args:
            treatment_type: Type of treatment.

        Returns:
            Duration in days.
        """
        durations = {
            TreatmentType.CHEMOTHERAPY: random.randint(90, 180),
            TreatmentType.RADIATION: random.randint(30, 50),
            TreatmentType.SURGERY: random.randint(1, 7),
            TreatmentType.IMMUNOTHERAPY: random.randint(180, 365),
            TreatmentType.TARGETED_THERAPY: random.randint(180, 365),
            TreatmentType.HORMONE_THERAPY: random.randint(365, 1825),  # 1-5 years
        }
        return durations.get(treatment_type, 30)

    def _generate_physician_name(self) -> str:
        """Generate a physician name with title."""
        return f"Dr. {self.faker.last_name()}"

    def _generate_treatment_notes(self, treatment_type: TreatmentType) -> str | None:
        """Generate optional treatment notes.

        Args:
            treatment_type: Type of treatment.

        Returns:
            Notes string or None.
        """
        if random.random() > 0.7:  # 30% chance of notes
            return None

        notes_templates = {
            TreatmentType.CHEMOTHERAPY: [
                "Patient tolerated treatment well.",
                "Dose reduction due to neutropenia.",
                "Antiemetics administered pre-treatment.",
                "No significant adverse events.",
            ],
            TreatmentType.RADIATION: [
                "Daily treatment completed as scheduled.",
                "Mild skin reaction noted.",
                "Good tumor response on imaging.",
            ],
            TreatmentType.SURGERY: [
                "Procedure completed without complications.",
                "Clear surgical margins achieved.",
                "Patient recovered well post-operatively.",
            ],
        }

        templates = notes_templates.get(treatment_type, ["Treatment ongoing."])
        return random.choice(templates)
