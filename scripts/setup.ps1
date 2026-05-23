# ═══════════════════════════════════════════════════════════
# MLOps Fraud Detection Pipeline — Windows PowerShell Setup
# Run: powershell -ExecutionPolicy Bypass -File scripts\setup.ps1
# ═══════════════════════════════════════════════════════════

$ErrorActionPreference = "Stop"

function Write-Step($msg)  { Write-Host "→ $msg" -ForegroundColor Cyan }
function Write-Ok($msg)    { Write-Host "✓ $msg" -ForegroundColor Green }
function Write-Warn($msg)  { Write-Host "! $msg" -ForegroundColor Yellow }

Write-Host ""
Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor Cyan
Write-Host "  MLOps Fraud Detection Pipeline — Quick Start    " -ForegroundColor Cyan
Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor Cyan
Write-Host ""

# ── Check Python ──────────────────────────────────────────
Write-Step "Checking Python..."
try {
    $pyver = python --version 2>&1
    Write-Ok "$pyver detected"
} catch {
    Write-Host "ERROR: Python not found. Install from https://python.org" -ForegroundColor Red
    exit 1
}

# ── Virtual environment ───────────────────────────────────
if (-not (Test-Path ".venv")) {
    Write-Step "Creating virtual environment..."
    python -m venv .venv
    Write-Ok "Virtual environment created"
} else {
    Write-Ok "Virtual environment already exists"
}

Write-Step "Activating virtual environment..."
& ".venv\Scripts\Activate.ps1"
Write-Ok "Virtual environment activated"

# ── Upgrade pip ───────────────────────────────────────────
Write-Step "Upgrading pip..."
python -m pip install --upgrade pip -q

# ── Install core dependencies ─────────────────────────────
Write-Step "Installing dependencies (this may take 2-3 minutes)..."
pip install `
    fastapi "uvicorn[standard]" pydantic pydantic-settings `
    xgboost scikit-learn numpy pandas scipy joblib imbalanced-learn `
    mlflow click rich pyyaml aiofiles jinja2 httpx `
    prometheus-client `
    pytest pytest-asyncio pytest-cov faker `
    pyarrow -q

Write-Ok "Dependencies installed"

# ── Create directories ────────────────────────────────────
Write-Step "Creating data directories..."
$dirs = @("data\raw","data\processed","data\reference","data\metrics","mlruns")
foreach ($d in $dirs) {
    if (-not (Test-Path $d)) { New-Item -ItemType Directory -Path $d -Force | Out-Null }
}
Write-Ok "Directories ready"

# ── Run pipeline ──────────────────────────────────────────
Write-Step "Generating synthetic fraud dataset (100k transactions)..."
python scripts\train.py ingest
Write-Ok "Dataset generated"

Write-Step "Preprocessing & splitting data..."
python scripts\train.py preprocess-cmd
Write-Ok "Data split: train / val / test"

Write-Step "Training XGBoost model with MLflow tracking..."
python scripts\train.py train
Write-Ok "Model trained and registered in MLflow"

# ── Done ──────────────────────────────────────────────────
Write-Host ""
Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor Green
Write-Host "  Setup complete! Start services with:" -ForegroundColor Green
Write-Host ""
Write-Host "  Dashboard (UI):" -ForegroundColor Yellow
Write-Host "    uvicorn dashboard.main:app --port 8050 --reload"
Write-Host ""
Write-Host "  ML API:"  -ForegroundColor Yellow
Write-Host "    uvicorn src.serving.api:app --port 8000 --reload"
Write-Host ""
Write-Host "  MLflow UI:" -ForegroundColor Yellow
Write-Host "    mlflow ui --port 5000 --backend-store-uri mlruns"
Write-Host ""
Write-Host "  Open dashboard: http://localhost:8050" -ForegroundColor Cyan
Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor Green
Write-Host ""
