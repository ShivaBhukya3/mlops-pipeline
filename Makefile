# ═══════════════════════════════════════════════════════════
#  MLOps Fraud Detection Pipeline — Makefile
# ═══════════════════════════════════════════════════════════

.PHONY: help setup install lint format test train serve dashboard \
        docker-build docker-up docker-down docker-logs clean

PYTHON := python
PIP    := pip

# ─── Help ──────────────────────────────────────────────────
help:
	@echo ""
	@echo "  MLOps Fraud Detection Pipeline"
	@echo "  ────────────────────────────────────────────"
	@echo "  make setup        Install all dependencies"
	@echo "  make install      pip install -r requirements.txt"
	@echo "  make lint         Run ruff + black --check"
	@echo "  make format       Auto-format with black + ruff --fix"
	@echo "  make test         Run pytest with coverage"
	@echo ""
	@echo "  make ingest       Generate raw dataset"
	@echo "  make preprocess   Feature engineering & split"
	@echo "  make train        Train model with MLflow tracking"
	@echo "  make serve        Start FastAPI serving (port 8000)"
	@echo "  make dashboard    Start dashboard (port 8050)"
	@echo ""
	@echo "  make docker-build Build all Docker images"
	@echo "  make docker-up    Start core services (postgres, mlflow, api, dashboard)"
	@echo "  make docker-down  Stop all services"
	@echo "  make docker-logs  Tail service logs"
	@echo "  make docker-train Run training container"
	@echo ""
	@echo "  make mlflow-ui    Open MLflow UI (port 5000)"
	@echo "  make clean        Remove generated data & build artifacts"
	@echo ""

# ─── Setup ─────────────────────────────────────────────────
setup: install
	@mkdir -p data/raw data/processed data/reference data/metrics mlruns

install:
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt

# ─── Code quality ──────────────────────────────────────────
lint:
	ruff check src/ dashboard/ scripts/ tests/
	black --check src/ dashboard/ scripts/ tests/

format:
	ruff check --fix src/ dashboard/ scripts/ tests/
	black src/ dashboard/ scripts/ tests/

# ─── Testing ───────────────────────────────────────────────
test:
	pytest tests/ \
		--cov=src \
		--cov-report=term-missing \
		--cov-report=html:htmlcov \
		-v \
		--tb=short

test-fast:
	pytest tests/ -x -q --tb=short

# ─── Pipeline steps ────────────────────────────────────────
ingest:
	$(PYTHON) scripts/train.py ingest --force

preprocess:
	$(PYTHON) scripts/train.py preprocess-cmd

train:
	$(PYTHON) scripts/train.py train --promote

pipeline: ingest preprocess train

# ─── Serving ───────────────────────────────────────────────
serve:
	uvicorn src.serving.api:app --host 0.0.0.0 --port 8000 --reload

dashboard:
	uvicorn dashboard.main:app --host 0.0.0.0 --port 8050 --reload

mlflow-ui:
	mlflow ui --host 0.0.0.0 --port 5000 --backend-store-uri mlruns

# ─── Docker ────────────────────────────────────────────────
docker-build:
	docker compose build --parallel

docker-up:
	docker compose up -d postgres mlflow api dashboard

docker-train:
	docker compose --profile train up --build training

docker-airflow:
	docker compose --profile airflow up -d

docker-monitoring:
	docker compose --profile monitoring up -d

docker-down:
	docker compose down

docker-logs:
	docker compose logs -f --tail=100

docker-clean:
	docker compose down -v --remove-orphans

# ─── Clean ─────────────────────────────────────────────────
clean:
	rm -rf htmlcov .coverage coverage.xml __pycache__ .pytest_cache
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete

clean-data:
	rm -rf data/raw/* data/processed/* data/reference/* data/metrics/*
	@echo "Data cleared (mlruns preserved)"

clean-all: clean clean-data
	rm -rf mlruns
	@echo "Full clean complete"
