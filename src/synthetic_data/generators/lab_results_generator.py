"""
Synthetic lab results generator for oncology datasets.

This module generates realistic laboratory test results including
hematology panels, chemistry panels, and tumor markers.
"""

import random
from datetime import datetime, timedelta
from typing import Any
from uuid import UUID

import structlog
from faker import Faker

from src.synthetic_data.schemas.oncology_schemas import (
    ONCOLOGY_LAB_TESTS,
    LabResultSchema,
    LabTestCategory,
    PatientSchema,
)

logger = structlog.get_logger(__name__)


class LabResultsGenerator:
    """Generator for synthetic oncology lab results.

    This generator creates realistic laboratory test results with
    clinically valid values and appropriate abnormality patterns.

    Attributes:
        faker: Faker instance for generating random data.
        seed: Random seed for reproducibility.

    Example:
        >>> generator = LabResultsGenerator(seed=42)
        >>> results = generator.generate_for_patients(patients, avg_results=10)
    """

    # Lab names
    LAB_NAMES: list[str] = [
        "Quest Diagnostics",
        "LabCorp",
        "Hospital Clinical Laboratory",
        "Regional Medical Lab",
        "University Hospital Lab",
        "Cancer Center Laboratory",
    ]

    # Specimen types by test category
    SPECIMEN_TYPES: dict[str, list[str]] = {
        "hematology": ["Whole Blood", "EDTA Blood"],
        "chemistry": ["Serum", "Plasma"],
        "tumor_marker": ["Serum", "Plasma"],
        "coagulation": ["Citrated Plasma"],
        "urinalysis": ["Urine", "24-Hour Urine"],
    }

    # Probability of abnormal results for oncology patients (higher than general population)
    ABNORMAL_PROBABILITY: float = 0.25

    # Probability of critical results
    CRITICAL_PROBABILITY: float = 0.05

    def __init__(self, seed: int | None = None, locale: str = "en_US") -> None:
        """Initialize the lab results generator.

        Args:
            seed: Random seed for reproducibility.
            locale: Faker locale for generating names.
        """
        self.seed = seed
        self.faker = Faker(locale)

        if seed is not None:
            Faker.seed(seed)
            random.seed(seed)

        self._logger = logger.bind(generator="lab_results", seed=seed)
        self._logger.info("Lab results generator initialized")

    def generate_for_patients(
        self,
        patients: list[PatientSchema],
        avg_results: int = 15,
        variance: int = 5,
        time_span_days: int = 365,
    ) -> list[LabResultSchema]:
        """Generate lab results for a list of patients.

        Args:
            patients: List of patients to generate results for.
            avg_results: Average number of lab results per patient.
            variance: Variance in number of results.
            time_span_days: Time span for results from diagnosis date.

        Returns:
            List of LabResultSchema objects.

        Example:
            >>> results = generator.generate_for_patients(patients, avg_results=15)
        """
        self._logger.info(
            "Generating lab results",
            patient_count=len(patients),
            avg_results=avg_results,
        )

        results: list[LabResultSchema] = []

        for patient in patients:
            num_results = max(1, avg_results + random.randint(-variance, variance))
            patient_results = self._generate_patient_results(patient, num_results, time_span_days)
            results.extend(patient_results)

        self._logger.info("Lab results generation complete", count=len(results))
        return results

    def generate_as_dicts(
        self,
        patients: list[PatientSchema],
        avg_results: int = 15,
        variance: int = 5,
        time_span_days: int = 365,
    ) -> list[dict[str, Any]]:
        """Generate lab results and return as list of dictionaries.

        Args:
            patients: List of patients.
            avg_results: Average results per patient.
            variance: Variance in result count.
            time_span_days: Time span for results.

        Returns:
            List of dictionaries suitable for DataFrame creation.
        """
        results = self.generate_for_patients(
            patients=patients,
            avg_results=avg_results,
            variance=variance,
            time_span_days=time_span_days,
        )
        return [r.to_dict() for r in results]

    def generate_panel(
        self,
        patient: PatientSchema,
        panel_type: str,
        collection_datetime: datetime | None = None,
    ) -> list[LabResultSchema]:
        """Generate a complete lab panel for a patient.

        Args:
            patient: Patient for the panel.
            panel_type: Type of panel ('hematology', 'chemistry', 'tumor_marker').
            collection_datetime: When specimen was collected.

        Returns:
            List of lab results for the panel.

        Example:
            >>> results = generator.generate_panel(patient, "hematology")
        """
        if collection_datetime is None:
            collection_datetime = datetime.now() - timedelta(days=random.randint(1, 30))

        # Filter tests by panel type
        panel_tests = [t for t in ONCOLOGY_LAB_TESTS if t["category"] == panel_type]

        results: list[LabResultSchema] = []
        for test in panel_tests:
            result = self._generate_single_result(patient, test, collection_datetime)
            results.append(result)

        return results

    def _generate_patient_results(
        self,
        patient: PatientSchema,
        num_results: int,
        time_span_days: int,
    ) -> list[LabResultSchema]:
        """Generate lab results for a single patient.

        Args:
            patient: Patient to generate results for.
            num_results: Number of results to generate.
            time_span_days: Time span for results.

        Returns:
            List of lab results for the patient.
        """
        results: list[LabResultSchema] = []

        # Calculate date range
        start_date = datetime.combine(patient.primary_diagnosis_date, datetime.min.time())
        end_date = start_date + timedelta(days=time_span_days)

        # Don't generate results in the future
        if end_date > datetime.now():
            end_date = datetime.now()

        # Generate results
        for _ in range(num_results):
            # Random collection time within range
            time_offset = random.randint(0, int((end_date - start_date).total_seconds()))
            collection_datetime = start_date + timedelta(seconds=time_offset)

            # Random test selection (with bias toward common tests)
            test = self._select_test()

            result = self._generate_single_result(patient, test, collection_datetime)
            results.append(result)

        return results

    def _select_test(self) -> dict[str, Any]:
        """Select a test with appropriate weighting.

        Returns:
            Test definition dictionary.
        """
        # Weight hematology and chemistry tests higher (more common in oncology)
        weights = []
        for test in ONCOLOGY_LAB_TESTS:
            if test["category"] == "hematology":
                weights.append(3.0)
            elif test["category"] == "chemistry":
                weights.append(2.5)
            elif test["category"] == "tumor_marker":
                weights.append(1.5)
            else:
                weights.append(1.0)

        return random.choices(ONCOLOGY_LAB_TESTS, weights=weights, k=1)[0]

    def _generate_single_result(
        self,
        patient: PatientSchema,
        test: dict[str, Any],
        collection_datetime: datetime,
    ) -> LabResultSchema:
        """Generate a single lab result.

        Args:
            patient: Patient for the result.
            test: Test definition dictionary.
            collection_datetime: When specimen was collected.

        Returns:
            LabResultSchema instance.
        """
        # Generate result value
        result_value, is_abnormal, is_critical = self._generate_result_value(test)

        # Result reporting time (1-24 hours after collection)
        result_datetime = collection_datetime + timedelta(hours=random.randint(1, 24))

        # Get category as enum
        category = LabTestCategory(test["category"])

        # Get specimen type
        specimen_types = self.SPECIMEN_TYPES.get(test["category"], ["Blood"])
        specimen_type = random.choice(specimen_types)

        return LabResultSchema(
            patient_id=patient.patient_id,
            test_code=test["code"],
            test_name=test["name"],
            test_category=category,
            result_value=result_value,
            result_unit=test["unit"],
            result_text=None,
            reference_range_low=test["low"],
            reference_range_high=test["high"],
            is_abnormal=is_abnormal,
            is_critical=is_critical,
            specimen_type=specimen_type,
            collection_datetime=collection_datetime,
            result_datetime=result_datetime,
            performing_lab=random.choice(self.LAB_NAMES),
            ordering_physician=f"Dr. {self.faker.last_name()}",
        )

    def _generate_result_value(
        self,
        test: dict[str, Any],
    ) -> tuple[float, bool, bool]:
        """Generate a result value with appropriate abnormality.

        Args:
            test: Test definition dictionary.

        Returns:
            Tuple of (value, is_abnormal, is_critical).
        """
        low = test["low"]
        high = test["high"]
        normal_range = high - low
        normal_mid = (high + low) / 2

        # Determine if this result should be abnormal
        is_abnormal = random.random() < self.ABNORMAL_PROBABILITY
        is_critical = False

        if is_abnormal:
            # Determine if critical
            is_critical = random.random() < (self.CRITICAL_PROBABILITY / self.ABNORMAL_PROBABILITY)

            if is_critical:
                # Critical values - significantly outside normal
                if random.random() < 0.5:
                    # Low critical
                    value = low * random.uniform(0.3, 0.6)
                else:
                    # High critical
                    value = high * random.uniform(1.5, 2.5)
            else:
                # Abnormal but not critical
                if random.random() < 0.5:
                    # Low abnormal
                    value = low * random.uniform(0.7, 0.95)
                else:
                    # High abnormal
                    value = high * random.uniform(1.05, 1.4)
        else:
            # Normal value with slight variation
            variation = normal_range * random.uniform(-0.3, 0.3)
            value = normal_mid + variation

            # Ensure within normal range for truly normal values
            value = max(low, min(high, value))

        # Round to appropriate precision
        if value < 1:
            value = round(value, 3)
        elif value < 10:
            value = round(value, 2)
        elif value < 100:
            value = round(value, 1)
        else:
            value = round(value, 0)

        return value, is_abnormal, is_critical

    def generate_trending_results(
        self,
        patient: PatientSchema,
        test: dict[str, Any],
        num_results: int = 6,
        start_datetime: datetime | None = None,
        trend: str = "stable",
    ) -> list[LabResultSchema]:
        """Generate a series of results showing a trend.

        Args:
            patient: Patient for the results.
            test: Test definition.
            num_results: Number of results in the series.
            start_datetime: Starting datetime.
            trend: Trend type ('improving', 'worsening', 'stable').

        Returns:
            List of trending lab results.

        Example:
            >>> test = ONCOLOGY_LAB_TESTS[0]  # WBC
            >>> results = generator.generate_trending_results(patient, test, trend="improving")
        """
        if start_datetime is None:
            start_datetime = datetime.now() - timedelta(days=180)

        low = test["low"]
        high = test["high"]
        normal_mid = (high + low) / 2

        results: list[LabResultSchema] = []

        # Generate starting value based on trend
        if trend == "improving":
            current_value = low * 0.6  # Start abnormally low
            target_value = normal_mid
        elif trend == "worsening":
            current_value = normal_mid
            target_value = high * 1.5  # End abnormally high
        else:  # stable
            current_value = normal_mid
            target_value = normal_mid

        value_step = (target_value - current_value) / num_results

        current_datetime = start_datetime

        for i in range(num_results):
            # Add some noise to the trend
            noise = value_step * random.uniform(-0.2, 0.2)
            result_value = current_value + noise

            # Determine abnormality
            is_abnormal = result_value < low or result_value > high
            is_critical = result_value < low * 0.5 or result_value > high * 2

            # Round appropriately
            if result_value < 1:
                result_value = round(result_value, 3)
            elif result_value < 10:
                result_value = round(result_value, 2)
            else:
                result_value = round(result_value, 1)

            # Get category and specimen
            category = LabTestCategory(test["category"])
            specimen_types = self.SPECIMEN_TYPES.get(test["category"], ["Blood"])
            specimen_type = random.choice(specimen_types)

            result = LabResultSchema(
                patient_id=patient.patient_id,
                test_code=test["code"],
                test_name=test["name"],
                test_category=category,
                result_value=result_value,
                result_unit=test["unit"],
                reference_range_low=test["low"],
                reference_range_high=test["high"],
                is_abnormal=is_abnormal,
                is_critical=is_critical,
                specimen_type=specimen_type,
                collection_datetime=current_datetime,
                result_datetime=current_datetime + timedelta(hours=random.randint(1, 6)),
                performing_lab=random.choice(self.LAB_NAMES),
                ordering_physician=f"Dr. {self.faker.last_name()}",
            )

            results.append(result)

            # Move to next time point and value
            current_value += value_step
            current_datetime += timedelta(days=random.randint(14, 30))

        return results
