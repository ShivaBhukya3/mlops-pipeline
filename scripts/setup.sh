#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════
# Quick-start setup script for MLOps Fraud Detection Pipeline
# ═══════════════════════════════════════════════════════════
set -euo pipefail

GREEN='\033[0;32m' CYAN='\033[0;36m' YELLOW='\033[1;33m' NC='\033[0m'

echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${CYAN}  MLOps Fraud Detection Pipeline — Quick Start             ${NC}"
echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

# Check Python version
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
echo -e "${GREEN}✓${NC} Python $PYTHON_VERSION detected"

# Create virtual environment
if [ ! -d ".venv" ]; then
    echo -e "${YELLOW}→${NC} Creating virtual environment..."
    python3 -m venv .venv
fi

source .venv/bin/activate
echo -e "${GREEN}✓${NC} Virtual environment activated"

# Install dependencies
echo -e "${YELLOW}→${NC} Installing dependencies..."
pip install --upgrade pip -q
pip install \
    fastapi uvicorn[standard] pydantic pydantic-settings \
    xgboost scikit-learn imbalanced-learn numpy pandas scipy joblib \
    mlflow click rich pyyaml aiofiles jinja2 httpx \
    prometheus-client pytest pytest-asyncio pytest-cov faker \
    -q

echo -e "${GREEN}✓${NC} Dependencies installed"

# Create directories
mkdir -p data/raw data/processed data/reference data/metrics mlruns
echo -e "${GREEN}✓${NC} Directories created"

# Run pipeline
echo -e "${YELLOW}→${NC} Generating dataset..."
python scripts/train.py ingest

echo -e "${YELLOW}→${NC} Preprocessing..."
python scripts/train.py preprocess-cmd

echo -e "${YELLOW}→${NC} Training model (this may take a minute)..."
python scripts/train.py train

echo ""
echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}  ✓ Setup complete! Start services:${NC}"
echo ""
echo -e "  ${YELLOW}Dashboard:${NC}  uvicorn dashboard.main:app --port 8050 --reload"
echo -e "  ${YELLOW}ML API:${NC}     uvicorn src.serving.api:app --port 8000 --reload"
echo -e "  ${YELLOW}MLflow UI:${NC}  mlflow ui --port 5000 --backend-store-uri mlruns"
echo ""
echo -e "  Open: ${CYAN}http://localhost:8050${NC}"
echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
