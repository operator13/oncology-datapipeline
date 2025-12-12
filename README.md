# Oncology Data Pipeline - Data Quality Automation Framework

[![Tests](https://img.shields.io/badge/tests-41%20passing-brightgreen)](https://github.com)
[![Data Quality](https://img.shields.io/badge/DQ%20Dimensions-6%2F6-blue)](https://github.com)
[![Great Expectations](https://img.shields.io/badge/Great%20Expectations-0.18+-orange)](https://greatexpectations.io/)
[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)

A **production-ready data quality framework** demonstrating enterprise-grade data validation for healthcare oncology data. Built as a portfolio project showcasing expertise in **data engineering**, **data quality**, and **healthcare data validation**.

## Key Highlights

- **41 automated tests** covering data quality, profiling, and synthetic data generation
- **6 Data Quality Dimensions** with real-time scoring and letter grades (A-F)
- **12,300+ lines of code** across 54 files with comprehensive documentation
- **Great Expectations** integration for enterprise-grade validation pipelines
- **Interactive Streamlit dashboard** for DQ visualization and monitoring
- **Healthcare-specific validations** including ICD-10, NDC, and LOINC codes

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                      ONCOLOGY DATA QUALITY PIPELINE                             │
└─────────────────────────────────────────────────────────────────────────────────┘

    ┌──────────────┐     ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
    │   SYNTHETIC  │     │    DATA      │     │    GREAT     │     │  STREAMLIT   │
    │     DATA     │────▶│   PROFILER   │────▶│ EXPECTATIONS │────▶│  DASHBOARD   │
    │  (Generate)  │     │  (Analyze)   │     │  (Validate)  │     │  (Visualize) │
    └──────────────┘     └──────────────┘     └──────────────┘     └──────────────┘
           │                    │                    │                    │
           ▼                    ▼                    ▼                    ▼
    ┌─────────────────────────────────────────────────────────────────────────────┐
    │                           DATA ACCESS LAYER                                 │
    │  ┌─────────────────────────────────┐  ┌─────────────────────────────────┐   │
    │  │      DATABRICKS CONNECTOR       │  │     SQL SERVER CONNECTOR        │   │
    │  │   Delta Lake  │  SQL Warehouse  │  │  Connection Pool  │  Azure SQL  │   │
    │  └─────────────────────────────────┘  └─────────────────────────────────┘   │
    │                                                                             │
    │  ┌─────────────────────────────────────────────────────────────────────┐    │
    │  │                    41 AUTOMATED TESTS (6 DQ DIMENSIONS)             │    │
    │  │  ✓ Completeness  ✓ Accuracy  ✓ Consistency  ✓ Timeliness            │    │
    │  │  ✓ Validity      ✓ Uniqueness                                       │    │
    │  └─────────────────────────────────────────────────────────────────────┘    │
    └─────────────────────────────────────────────────────────────────────────────┘
```

### Data Flow

```
┌────────────────────────────────────────────────────────────────────────────────┐
│                           QUALITY PIPELINE FLOW                                │
├────────────────────────────────────────────────────────────────────────────────┤
│                                                                                │
│  ┌───────┐    ┌─────────┐    ┌─────────┐    ┌───────────────────────────────┐  │
│  │ START │───▶│ INGEST  │───▶│ PROFILE │───▶│       VALIDATION TASKS        │  │
│  └───────┘    └─────────┘    └─────────┘    │  ┌───────┐ ┌───────┐ ┌─────┐  │  │
│                   │              │          │  │Patient│▶│Treat- │▶│Labs │  │  │
│                   ▼              ▼          │  │ Suite │ │ments  │ │Suite│  │  │
│              Load from       Statistical    │  └───────┘ └───────┘ └─────┘  │  │
│              CSV/DB/API      Analysis       │         │                     │  │
│                                             │         ▼                     │  │
│                                             │    ┌─────────┐                │  │
│                                             │    │41 TESTS │                │  │
│                                             │    └─────────┘                │  │
│                                             └───────────────────────────────┘  │
│                                                          │                     │
│                                                          ▼                     │
│                                            ┌─────────────────────────┐         │
│                                            │     QUALITY GATE        │         │
│                                            │  ┌─────────┐ ┌────────┐ │         │
│                                            │  │ PASSED  │ │ FAILED │ │         │
│                                            │  └────┬────┘ └────┬───┘ │         │
│                                            └───────┼───────────┼─────┘         │
│                                                    ▼           ▼               │
│                                            ┌───────────┐  ┌─────────┐          │
│                                            │ SCORECARD │  │  ALERT  │          │
│                                            │  (A-F)    │  │ & LOG   │          │
│                                            └───────────┘  └─────────┘          │
│                                                    │                           │
│                                                    ▼                           │
│                                               ┌───────┐                        │
│                                               │  END  │                        │
│                                               └───────┘                        │
└────────────────────────────────────────────────────────────────────────────────┘
```

---

## Tech Stack

| Layer | Technology | Purpose |
|-------|------------|---------|
| **Data Quality** | Great Expectations 0.18+ | Expectation suites, validation runners, data docs |
| **Connectors** | Databricks SQL, SQL Server | Multi-platform data access with Factory pattern |
| **Profiling** | Custom Python (pandas, numpy) | Statistical analysis, anomaly detection, drift monitoring |
| **Dashboard** | Streamlit 1.28+ | Interactive DQ visualization and monitoring |
| **Synthetic Data** | Faker, Pydantic | Realistic oncology test data generation |
| **Testing** | pytest, pytest-cov | 41 automated tests with fixtures |
| **CLI** | Typer | Command-line interface for all operations |
| **CI/CD** | GitHub Actions | Automated testing, linting, type checking |

---

## Data Quality Framework

### 6 Dimensions Covered

| Dimension | Description | # Tests | Example Validation |
|-----------|-------------|---------|-------------------|
| **Completeness** | Non-null required values | 8 | `expect_column_values_to_not_be_null` |
| **Uniqueness** | No duplicate keys | 6 | `expect_column_values_to_be_unique` |
| **Validity** | Values match expected format | 12 | ICD-10 regex `^C[0-9]{2}`, NDC format |
| **Consistency** | Cross-field logic holds | 5 | diagnosis_date > birth_date |
| **Accuracy** | Values within valid ranges | 7 | Lab values within clinical ranges |
| **Timeliness** | Data freshness | 3 | Recency checks, SLA compliance |

### Quality Scorecard Output

```
==================================================
DATA QUALITY SCORECARD: patients
==================================================
Overall Score: 88.2% (Grade: B)

Dimension Scores:
  completeness    [████████████████████] 100.0%
  uniqueness      [████████████████░░░░] 84.5%
  validity        [████████████████████] 100.0%
  consistency     [████████████████████] 100.0%
  timeliness      [█████░░░░░░░░░░░░░░░] 26.2%
  accuracy        [████████████████████] 100.0%
```

| Grade | Score Range | Interpretation |
|-------|-------------|----------------|
| A | 90-100% | Excellent - Production ready |
| B | 80-89% | Good - Minor issues |
| C | 70-79% | Acceptable - Needs attention |
| D | 60-69% | Poor - Significant issues |
| F | <60% | Failing - Do not use |

### Dashboard Preview

![Oncology Data Quality Dashboard](docs/images/dashboard-overview.png)

*Interactive dashboard showing quality scores, validation results, and anomaly detection*

---

## 41 Tests - Complete Breakdown

### Test Summary by Category

| Category | Count | Description |
|----------|-------|-------------|
| **Data Quality Tests** | 12 | Expectation builder, validation runner, suites |
| **Data Profiling Tests** | 14 | Profiler, anomaly detection, drift monitoring |
| **Synthetic Data Tests** | 15 | Generators, schema validation, referential integrity |
| **TOTAL** | **41** | |

### Detailed Test Inventory

<details>
<summary><b>Data Quality Tests (12 tests)</b></summary>

| Test | What It Validates |
|------|-------------------|
| `test_expect_not_null` | Null value detection in required fields |
| `test_expect_unique` | Uniqueness constraints (e.g., patient IDs) |
| `test_expect_between` | Numeric range validation (e.g., age 0-120) |
| `test_expect_regex` | Pattern matching (e.g., ICD-10 codes) |
| `test_expect_values_in_set` | Categorical value validation |
| `test_icd10_oncology_code` | ICD-10 oncology code format (C00-D49) |
| `test_ndc_code` | NDC drug identifier format |
| `test_chaining` | Fluent API for building expectations |
| `test_build_empty_suite` | Empty suite initialization |
| `test_patient_suite_structure` | JSON expectation suite structure |
| `test_success_percent` | Validation pass rate calculation |
| `test_failed_expectations_filter` | Failed expectation filtering |

</details>

<details>
<summary><b>Data Profiling Tests (14 tests)</b></summary>

| Test | What It Validates |
|------|-------------------|
| `test_profile_basic` | Basic profiling (row count, columns) |
| `test_profile_numeric_stats` | Statistical measures (mean, std, quartiles) |
| `test_profile_with_nulls` | Null value handling in profiles |
| `test_profile_to_dataframe` | Profile export to DataFrame |
| `test_detect_iqr` | IQR-based anomaly detection |
| `test_detect_zscore` | Z-score anomaly detection |
| `test_detect_mad` | Median Absolute Deviation detection |
| `test_no_anomalies_in_normal_data` | False positive prevention |
| `test_anomaly_result_summary` | Anomaly result aggregation |
| `test_detect_schema_drift_no_change` | Schema stability verification |
| `test_detect_schema_drift_added_column` | New column detection |
| `test_detect_schema_drift_removed_column` | Missing column detection |
| `test_detect_distribution_drift` | Statistical distribution changes |
| `test_detect_no_distribution_drift` | Distribution stability |

</details>

<details>
<summary><b>Synthetic Data Tests (15 tests)</b></summary>

| Test | What It Validates |
|------|-------------------|
| `test_generate_single_patient` | Patient record generation |
| `test_generate_multiple_patients` | Batch patient generation |
| `test_generate_treatments_for_patients` | Treatment record linking |
| `test_generate_lab_results` | Lab result generation |
| `test_generate_with_specific_cancer_types` | Cancer type filtering |
| `test_reproducibility_with_seed` | Deterministic generation with seeds |
| `test_valid_icd10_codes` | Generated ICD-10 code validity |
| `test_valid_treatment_types` | Treatment type enumeration |
| `test_valid_cancer_stages` | Cancer staging (I-IV) validity |
| `test_valid_test_categories` | Lab test category validity |
| `test_chemotherapy_has_drug_info` | Drug info for chemo treatments |
| `test_treatment_patient_reference` | Referential integrity |
| `test_diagnosis_date_after_birth` | Date logic validation |
| `test_result_datetime_after_collection` | Temporal ordering |
| `test_abnormal_flag_consistency` | Flag/value consistency |

</details>

---

## Great Expectations: How It Works

Great Expectations (GE) is an open-source data quality framework that enables teams to validate, document, and profile data. This project implements GE following production best practices.

### Core Concepts

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        GREAT EXPECTATIONS WORKFLOW                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   1. EXPECTATIONS          2. SUITES              3. VALIDATION             │
│   ┌─────────────────┐     ┌─────────────────┐    ┌─────────────────┐        │
│   │ Individual      │     │ Collection of   │    │ Run suites      │        │
│   │ data quality    │────▶│ expectations    │───▶│ against actual  │        │
│   │ rules           │     │ for a dataset   │    │ data            │        │
│   └─────────────────┘     └─────────────────┘    └────────┬────────┘        │
│                                                           │                 │
│   4. CHECKPOINTS          5. DATA DOCS           6. ACTIONS                 │
│   ┌─────────────────┐     ┌─────────────────┐    ┌───────▼─────────┐        │
│   │ Orchestrate     │     │ Auto-generated  │    │ Alerts, reports │        │
│   │ validation      │◀────│ documentation   │◀───│ CI/CD gates     │        │
│   │ runs            │     │ & results       │    │                 │        │
│   └─────────────────┘     └─────────────────┘    └─────────────────┘        │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Expectations (Data Quality Rules)

Expectations are declarative statements about data. This project uses healthcare-specific expectations:

```python
# Example: Validate patient MRN format
expect_column_values_to_match_regex(
    column="mrn",
    regex="^MRN-[A-Z]{2}-\\d{6}$"
)

# Example: Validate ICD-10 oncology codes (C00-D49)
expect_column_values_to_match_regex(
    column="diagnosis_code",
    regex="^(C[0-9]{2}|D[0-4][0-9]).*$"
)

# Example: Ensure diagnosis date is after birth date
expect_column_pair_values_A_to_be_greater_than_B(
    column_A="diagnosis_date",
    column_B="date_of_birth"
)
```

### Expectation Suites

Suites group related expectations. Located in `great_expectations/expectations/`:

| Suite | File | Purpose |
|-------|------|---------|
| Patients | `oncology_patients_suite.json` | Demographics, MRN, ICD-10 codes, staging |
| Treatments | `oncology_treatments_suite.json` | NDC codes, dosages, date ranges |
| Lab Results | `oncology_lab_results_suite.json` | LOINC codes, reference ranges, values |

### Healthcare-Specific Validations

| Category | Validation | Why It Matters |
|----------|------------|----------------|
| **ICD-10 Codes** | Match pattern `^C[0-9]{2}` or `^D[0-4][0-9]` | Ensures valid oncology diagnosis codes |
| **NDC Codes** | Match 10/11-digit drug identifier format | Required for treatment accuracy |
| **LOINC Codes** | Valid lab test identifiers | Interoperability standard |
| **Tumor Staging** | Values in {I, II, III, IV, Unknown} | Clinical staging validation |
| **Date Logic** | diagnosis_date > birth_date | Prevents impossible records |
| **Lab Ranges** | Values within clinical reference ranges | Flags potentially erroneous results |

---

## Quick Start

### Prerequisites

- Python 3.10, 3.11, or 3.12
- Git
- Virtual environment (recommended)

### 1. Clone and Setup

```bash
git clone https://github.com/operator13/oncology-datapipeline.git
cd oncology-datapipeline

# Create virtual environment (recommended: Python 3.11)
python3.11 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 2. Install Dependencies

```bash
# Option 1: Using requirements.txt
pip install -r requirements.txt

# Option 2: Using pyproject.toml (includes dev dependencies)
pip install -e ".[dev]"
```

### 3. Generate Synthetic Data

```bash
# Generate 1000 patients with treatments and lab results
make generate-data

# Or with custom options
python -m src.cli generate --patients 5000 --output data/synthetic/
```

### 4. Run Data Quality Validations

```bash
# Run all validations
make validate

# Or run specific domain
make validate-patients
make validate-treatments
make validate-labs
```

### 5. Launch Dashboard

```bash
make dashboard
# Opens at http://localhost:8501
```

### CLI Commands

| Command | Description |
|---------|-------------|
| `python -m src.cli generate` | Generate synthetic oncology data |
| `python -m src.cli validate` | Run Great Expectations validations |
| `python -m src.cli scorecard` | Calculate quality score (A-F grade) |
| `python -m src.cli dashboard` | Launch Streamlit dashboard |
| `python -m src.cli test-connection` | Test database connectivity |

---

## Project Structure

```
oncology_datapipeline/
├── src/
│   ├── cli.py                         # Typer CLI entry point
│   ├── connectors/
│   │   ├── base.py                    # Abstract DataConnector interface
│   │   ├── databricks_connector.py    # Databricks SQL connector
│   │   ├── sqlserver_connector.py     # SQL Server connector
│   │   └── connection_factory.py      # Factory pattern implementation
│   ├── data_quality/
│   │   ├── expectation_builder.py     # Fluent API for expectations
│   │   ├── validation_runner.py       # Execute GE validations
│   │   └── checkpoint_manager.py      # Orchestrate validation runs
│   ├── pipelines/
│   │   ├── base_pipeline.py           # Template Method pattern
│   │   └── quality_pipeline.py        # Orchestrated quality checks
│   ├── profiling/
│   │   ├── data_profiler.py           # Statistical profiling
│   │   ├── anomaly_detector.py        # IQR, Z-score, MAD, Isolation Forest
│   │   └── drift_detector.py          # Schema & distribution drift
│   ├── synthetic_data/
│   │   ├── generators/                # Patient, Treatment, Lab generators
│   │   ├── schemas/                   # Pydantic models
│   │   └── oncology_data_factory.py   # Main orchestrator
│   └── metrics/
│       └── quality_scorecard.py       # 6-dimension scoring (A-F)
├── great_expectations/
│   ├── great_expectations.yml         # GE configuration
│   └── expectations/                  # JSON expectation suites
├── dashboards/
│   └── quality_dashboard.py           # Streamlit dashboard
├── tests/
│   ├── conftest.py                    # Pytest fixtures
│   └── unit/                          # 41 unit tests
├── .github/workflows/
│   └── ci.yml                         # GitHub Actions CI pipeline
├── CLAUDE.md                          # Code quality guidelines
├── Makefile                           # Development commands
└── pyproject.toml                     # Package configuration
```

---

## Make Commands

Run `make help` to see all available commands.

### Data Generation
```bash
make generate-data       # Generate synthetic oncology data (1000 patients)
```

### Data Quality
```bash
make validate            # Run all validations
make validate-patients   # Validate patient data
make validate-treatments # Validate treatment data
make validate-labs       # Validate lab results
```

### Testing
```bash
make test               # Run all 41 tests
make test-unit          # Run unit tests only
make test-coverage      # Run with coverage report
```

### Code Quality
```bash
make lint               # Run flake8 and mypy
make format             # Auto-format with black and isort
make typecheck          # Run mypy type checking
make check              # Run all checks (lint + typecheck + test)
```

### Development
```bash
make dashboard          # Launch Streamlit dashboard
make clean              # Clean build artifacts
```

---

## Dashboard Panels

| Page | Features |
|------|----------|
| **Overview** | Quality scores, dimension metrics, validation history, trend charts |
| **Data Profiling** | Upload CSV files for statistical analysis, column distributions |
| **Validation Results** | Pass/fail status per expectation, detailed failure reports |
| **Anomaly Detection** | Configure detection methods (IQR, Z-score, MAD, Isolation Forest) |
| **Generate Data** | Create synthetic oncology data with configurable parameters |

---

## CI/CD Pipeline

### Pull Request Checks
- **Linting**: flake8 for critical errors
- **Type Checking**: mypy strict mode
- **Testing**: 41 tests with pytest
- **Coverage**: Minimum threshold enforcement

### Workflow
```yaml
on: [push, pull_request]
jobs:
  - lint           # Code style checks
  - typecheck      # Static type analysis
  - test           # Run 41 automated tests
  - coverage       # Generate coverage report
```

---

## Skills Demonstrated

| Category | Skills |
|----------|--------|
| **Data Engineering** | ELT pipelines, Factory pattern, SOLID principles, multi-platform connectors |
| **Data Quality** | 6 DQ dimensions, Great Expectations, anomaly detection, drift monitoring |
| **Healthcare Domain** | ICD-10, NDC, LOINC validation, oncology-specific rules, HIPAA awareness |
| **Python** | Type hints, Pydantic, pytest, Typer CLI, Streamlit, pandas |
| **Testing & QA** | 41 automated tests, fixtures, parameterization, coverage reporting |
| **DevOps** | CI/CD with GitHub Actions, code quality gates, automated checks |
| **Documentation** | Comprehensive docstrings, CLAUDE.md guidelines, README |

---

## Author

**Data Engineering Portfolio Project**

Demonstrating enterprise-grade data quality practices for **Data Engineering** and **QA Lead** roles in healthcare/life sciences.

---

## License

MIT License - See [LICENSE](LICENSE) for details.
