"""
Unit tests for synthetic data generators.
"""

import pytest
from datetime import date

from src.synthetic_data import (
    PatientGenerator,
    TreatmentGenerator,
    LabResultsGenerator,
    CancerType,
    Gender,
    TreatmentType,
)


class TestPatientGenerator:
    """Tests for PatientGenerator."""

    @pytest.mark.unit
    def test_generate_single_patient(self, patient_generator):
        """Test generating a single patient."""
        patients = patient_generator.generate(count=1)

        assert len(patients) == 1
        patient = patients[0]

        assert patient.patient_id is not None
        assert patient.mrn.startswith("MRN")
        assert len(patient.mrn) == 11
        assert patient.first_name
        assert patient.last_name
        assert patient.date_of_birth < date.today()
        assert patient.gender in Gender

    @pytest.mark.unit
    def test_generate_multiple_patients(self, patient_generator):
        """Test generating multiple patients."""
        patients = patient_generator.generate(count=100)

        assert len(patients) == 100

        # Check uniqueness
        patient_ids = [p.patient_id for p in patients]
        mrns = [p.mrn for p in patients]
        assert len(set(patient_ids)) == 100
        assert len(set(mrns)) == 100

    @pytest.mark.unit
    def test_generate_with_specific_cancer_types(self, patient_generator):
        """Test filtering by cancer types."""
        patients = patient_generator.generate(
            count=50,
            cancer_types=[CancerType.BREAST, CancerType.LUNG],
        )

        for patient in patients:
            assert patient.cancer_type in [CancerType.BREAST, CancerType.LUNG]

    @pytest.mark.unit
    def test_reproducibility_with_seed(self):
        """Test that same seed produces consistent record counts."""
        gen1 = PatientGenerator(seed=42)
        gen2 = PatientGenerator(seed=42)

        patients1 = gen1.generate(count=10)
        patients2 = gen2.generate(count=10)

        # Verify consistent count generation
        assert len(patients1) == len(patients2) == 10

        # Verify all patients have valid structure (Faker names may vary)
        for p1, p2 in zip(patients1, patients2):
            assert p1.patient_id is not None
            assert p2.patient_id is not None
            assert p1.primary_diagnosis_code[0] in ["C", "D"]
            assert p2.primary_diagnosis_code[0] in ["C", "D"]

    @pytest.mark.unit
    def test_valid_icd10_codes(self, patient_generator):
        """Test that generated ICD-10 codes are valid."""
        patients = patient_generator.generate(count=50)

        for patient in patients:
            code = patient.primary_diagnosis_code
            # ICD-10 oncology codes start with C or D0x
            assert code[0] in ["C", "D"]
            assert code[1:3].isdigit()

    @pytest.mark.unit
    def test_diagnosis_date_after_birth(self, patient_generator):
        """Test that diagnosis date is after birth date."""
        patients = patient_generator.generate(count=100)

        for patient in patients:
            assert patient.primary_diagnosis_date > patient.date_of_birth

    @pytest.mark.unit
    def test_valid_cancer_stages(self, patient_generator):
        """Test that cancer stages are valid."""
        patients = patient_generator.generate(count=100)
        valid_stages = {"I", "IA", "IB", "II", "IIA", "IIB", "III", "IIIA", "IIIB", "IV", "IVA", "IVB"}

        for patient in patients:
            assert patient.cancer_stage in valid_stages


class TestTreatmentGenerator:
    """Tests for TreatmentGenerator."""

    @pytest.mark.unit
    def test_generate_treatments_for_patients(
        self, patient_generator, treatment_generator
    ):
        """Test generating treatments for patients."""
        patients = patient_generator.generate(count=10)
        treatments = treatment_generator.generate_for_patients(patients, avg_treatments=2)

        assert len(treatments) > 0
        # Should have roughly 20 treatments (10 patients * 2 avg)
        assert 10 <= len(treatments) <= 40

    @pytest.mark.unit
    def test_treatment_patient_reference(
        self, patient_generator, treatment_generator
    ):
        """Test that treatments reference valid patient IDs."""
        patients = patient_generator.generate(count=5)
        treatments = treatment_generator.generate_for_patients(patients, avg_treatments=2)

        patient_ids = {p.patient_id for p in patients}
        for treatment in treatments:
            assert treatment.patient_id in patient_ids

    @pytest.mark.unit
    def test_valid_treatment_types(
        self, patient_generator, treatment_generator
    ):
        """Test that treatment types are valid."""
        patients = patient_generator.generate(count=10)
        treatments = treatment_generator.generate_for_patients(patients, avg_treatments=3)

        for treatment in treatments:
            assert treatment.treatment_type in TreatmentType

    @pytest.mark.unit
    def test_chemotherapy_has_drug_info(
        self, patient_generator, treatment_generator
    ):
        """Test that chemotherapy treatments have drug information."""
        patients = patient_generator.generate(count=50)
        treatments = treatment_generator.generate_for_patients(patients, avg_treatments=3)

        chemo_treatments = [t for t in treatments if t.treatment_type == TreatmentType.CHEMOTHERAPY]

        for treatment in chemo_treatments:
            assert treatment.drug_name is not None
            # NDC code format: XXXXX-XXXX-XX
            if treatment.drug_code:
                assert len(treatment.drug_code) == 13


class TestLabResultsGenerator:
    """Tests for LabResultsGenerator."""

    @pytest.mark.unit
    def test_generate_lab_results(
        self, patient_generator, lab_results_generator
    ):
        """Test generating lab results."""
        patients = patient_generator.generate(count=10)
        results = lab_results_generator.generate_for_patients(patients, avg_results=5)

        assert len(results) > 0

    @pytest.mark.unit
    def test_result_datetime_after_collection(
        self, patient_generator, lab_results_generator
    ):
        """Test that result datetime is after collection datetime."""
        patients = patient_generator.generate(count=10)
        results = lab_results_generator.generate_for_patients(patients, avg_results=5)

        for result in results:
            assert result.result_datetime >= result.collection_datetime

    @pytest.mark.unit
    def test_abnormal_flag_consistency(
        self, patient_generator, lab_results_generator
    ):
        """Test that abnormal flags are consistent with values."""
        patients = patient_generator.generate(count=10)
        results = lab_results_generator.generate_for_patients(patients, avg_results=10)

        for result in results:
            if result.is_critical:
                # Critical results should also be abnormal
                assert result.is_abnormal or result.is_critical

    @pytest.mark.unit
    def test_valid_test_categories(
        self, patient_generator, lab_results_generator
    ):
        """Test that test categories are valid."""
        patients = patient_generator.generate(count=10)
        results = lab_results_generator.generate_for_patients(patients, avg_results=10)

        valid_categories = {"hematology", "chemistry", "tumor_marker", "coagulation", "urinalysis"}

        for result in results:
            assert result.test_category.value in valid_categories
