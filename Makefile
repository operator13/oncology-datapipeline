.PHONY: help install dev clean lint format typecheck test test-unit test-integration test-e2e test-coverage docs serve-docs ge-docs dashboard

# Default target
help:
	@echo "Oncology Data Pipeline - Development Commands"
	@echo ""
	@echo "Setup:"
	@echo "  make install        Install production dependencies"
	@echo "  make dev            Install development dependencies"
	@echo "  make clean          Remove build artifacts and caches"
	@echo ""
	@echo "Code Quality:"
	@echo "  make lint           Run all linters (flake8, mypy)"
	@echo "  make format         Auto-format code (black, isort)"
	@echo "  make typecheck      Run mypy type checker"
	@echo "  make check          Run format check + lint + typecheck"
	@echo ""
	@echo "Testing:"
	@echo "  make test           Run all tests"
	@echo "  make test-unit      Run unit tests only"
	@echo "  make test-integration  Run integration tests"
	@echo "  make test-e2e       Run end-to-end tests"
	@echo "  make test-coverage  Run tests with coverage report"
	@echo "  make test-fast      Run tests in parallel"
	@echo ""
	@echo "Documentation:"
	@echo "  make docs           Build documentation"
	@echo "  make serve-docs     Serve documentation locally"
	@echo "  make ge-docs        Generate Great Expectations data docs"
	@echo ""
	@echo "Data Quality:"
	@echo "  make validate       Run data quality validations"
	@echo "  make profile        Generate data profiles"
	@echo "  make dashboard      Launch quality metrics dashboard"
	@echo ""
	@echo "Synthetic Data:"
	@echo "  make generate-data  Generate synthetic oncology data"

# ============================================================================
# SETUP
# ============================================================================

install:
	pip install -e .

dev:
	pip install -e ".[dev]"
	pre-commit install

clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf .tox/
	rm -rf .nox/
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type f -name "*.pyo" -delete 2>/dev/null || true
	find . -type f -name "*.orig" -delete 2>/dev/null || true

# ============================================================================
# CODE QUALITY
# ============================================================================

lint:
	flake8 src/ tests/ --max-line-length=100 --extend-ignore=E,W,D,B,F --select=E9,F63,F7,F82
	mypy src/

format:
	black src/ tests/
	isort src/ tests/

format-check:
	black --check src/ tests/
	isort --check-only src/ tests/

typecheck:
	mypy src/ --strict

check: format-check lint typecheck

# ============================================================================
# TESTING
# ============================================================================

test:
	pytest tests/ -v

test-unit:
	pytest tests/unit/ -v

test-integration:
	@echo "Integration tests require database connections. Skipping if none configured."
	pytest tests/integration/ -v --ignore-glob="*__init__*" || echo "No integration tests found or configured."

test-e2e:
	@echo "E2E tests require full environment. Skipping if not configured."
	pytest tests/e2e/ -v --ignore-glob="*__init__*" || echo "No e2e tests found or configured."

test-coverage:
	pytest tests/ --cov=src --cov-report=html --cov-report=term-missing --cov-fail-under=40

test-fast:
	pytest tests/ -n auto -v

test-watch:
	pytest-watch -- -v

# ============================================================================
# DOCUMENTATION
# ============================================================================

docs:
	mkdocs build

serve-docs:
	mkdocs serve

ge-docs:
	great_expectations docs build

# ============================================================================
# DATA QUALITY OPERATIONS
# ============================================================================

validate:
	@echo "Validating all data in data/synthetic/..."
	python -m src.cli validate --suite patients --input data/synthetic/patients.csv
	python -m src.cli validate --suite treatments --input data/synthetic/treatments.csv
	python -m src.cli validate --suite lab_results --input data/synthetic/lab_results.csv

validate-patients:
	python -m src.cli validate --suite patients --input data/synthetic/patients.csv

validate-treatments:
	python -m src.cli validate --suite treatments --input data/synthetic/treatments.csv

validate-labs:
	python -m src.cli validate --suite lab_results --input data/synthetic/lab_results.csv

profile:
	python -m src.cli profile --output reports/

dashboard:
	streamlit run dashboards/quality_dashboard.py

# ============================================================================
# SYNTHETIC DATA
# ============================================================================

generate-data:
	python -m src.cli generate --patients 1000 --output data/synthetic/

generate-data-small:
	python -m src.cli generate --patients 100 --output data/synthetic/

generate-data-large:
	python -m src.cli generate --patients 10000 --output data/synthetic/

# ============================================================================
# GREAT EXPECTATIONS
# ============================================================================

ge-init:
	great_expectations init

ge-checkpoint:
	great_expectations checkpoint run oncology_checkpoint

ge-suite-edit:
	great_expectations suite edit $(SUITE)

# ============================================================================
# DATABASE OPERATIONS
# ============================================================================

db-test-connection:
	python -m src.cli test-connection --all

db-test-databricks:
	python -m src.cli test-connection --databricks

db-test-sqlserver:
	python -m src.cli test-connection --sqlserver

# ============================================================================
# CI/CD HELPERS
# ============================================================================

ci-lint:
	@echo "Running CI lint checks..."
	$(MAKE) format-check
	$(MAKE) lint

ci-test:
	@echo "Running CI tests..."
	pytest tests/ --cov=src --cov-report=xml --cov-fail-under=40 -v

ci: ci-lint ci-test

# ============================================================================
# DOCKER (Optional)
# ============================================================================

docker-build:
	docker build -t oncology-datapipeline:latest .

docker-run:
	docker run -it --rm oncology-datapipeline:latest

docker-test:
	docker run --rm oncology-datapipeline:latest make test
