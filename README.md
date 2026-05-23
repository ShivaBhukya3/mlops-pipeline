# MLOps Fraud Detection Pipeline

> Real-time credit card fraud detection with end-to-end MLOps — training, serving, monitoring, and CI/CD.

[![CI](https://github.com/ShivaBhukya3/mlops-pipeline/actions/workflows/ci.yml/badge.svg)](https://github.com/ShivaBhukya3/mlops-pipeline/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.11-blue.svg)](https://python.org)
[![XGBoost](https://img.shields.io/badge/model-XGBoost-orange.svg)](https://xgboost.readthedocs.io)
[![MLflow](https://img.shields.io/badge/tracking-MLflow-blue.svg)](https://mlflow.org)
[![FastAPI](https://img.shields.io/badge/API-FastAPI-009688.svg)](https://fastapi.tiangolo.com)
[![Docker](https://img.shields.io/badge/infra-Docker-2496ED.svg)](https://docker.com)
[![Render](https://img.shields.io/badge/deploy-Render-46E3B7.svg)](https://render.com)

---

## What This Project Does

Trains an XGBoost model on 100,000 synthetic credit card transactions (2% fraud rate), exposes predictions through a production-grade FastAPI, monitors for data drift in real time, and orchestrates the entire lifecycle through Airflow DAGs — with a dark glassmorphism dashboard to visualize it all.

**Live demo** → `https://mlops-dashboard.onrender.com`

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Airflow Orchestration                     │
│  ┌──────────────┐   ┌──────────────────┐   ┌─────────────────┐ │
│  │ Ingest DAG   │   │  Training DAG    │   │ Monitoring DAG  │ │
│  │  every 6h    │──▶│  every Sunday    │   │  every hour     │ │
│  └──────────────┘   └──────────────────┘   └─────────────────┘ │
└───────────────────────────┬─────────────────────────┬───────────┘
                            │                         │
                     ┌──────▼──────┐          ┌───────▼──────┐
                     │   MLflow    │          │  Drift Alert │
                     │  Tracking   │          │  PSI + KS    │
                     └──────┬──────┘          └──────────────┘
                            │
              ┌─────────────▼──────────────┐
              │      FastAPI Serving       │
              │  /predict  /batch-predict  │
              │  /health   /metrics        │
              └─────────────┬──────────────┘
                            │
              ┌─────────────▼──────────────┐
              │   Dashboard (port 8050)    │
              │  Charts · Metrics · Drift  │
              └────────────────────────────┘
```

---

## Stack

| Layer | Technology |
|---|---|
| Model | XGBoost (`scale_pos_weight=49` for class imbalance) |
| Tracking | MLflow (experiments, model registry, artifacts) |
| Serving | FastAPI + Uvicorn |
| Orchestration | Apache Airflow (3 DAGs) |
| Monitoring | Prometheus + Grafana + custom drift detector |
| Dashboard | FastAPI + Jinja2 + Chart.js (dark glassmorphism UI) |
| Infrastructure | Docker Compose (9 services) |
| CI/CD | GitHub Actions (lint → test → build → deploy) |
| Deployment | Render (free tier, `render.yaml` blueprint) |

---

## Project Structure

```
mlops-pipeline/
├── src/
│   ├── data/
│   │   ├── ingestion.py        # Synthetic transaction generator
│   │   ├── preprocessing.py    # Feature engineering + splits
│   │   └── validation.py       # Schema & null checks
│   ├── models/
│   │   ├── trainer.py          # XGBoost + MLflow logging
│   │   ├── evaluator.py        # ROC/PR curves, business impact
│   │   └── registry.py         # Model promotion logic
│   ├── serving/
│   │   ├── api.py              # FastAPI endpoints
│   │   └── schemas.py          # Pydantic v2 request/response
│   └── monitoring/
│       ├── drift_detector.py   # PSI + KS-test + Chi-squared
│       └── metrics_collector.py# Real-time metrics (deque-based)
├── dashboard/
│   ├── main.py                 # Dashboard FastAPI backend
│   ├── templates/index.html    # 7-page SPA
│   └── static/
│       ├── css/style.css       # Dark glassmorphism design
│       └── js/dashboard.js     # Chart.js + 10s live refresh
├── airflow/dags/
│   ├── data_ingestion_dag.py   # Ingest → validate → preprocess
│   ├── training_pipeline_dag.py# Train → evaluate → auto-promote
│   └── monitoring_dag.py       # Drift check → trigger retrain
├── docker/
│   ├── Dockerfile.api
│   ├── Dockerfile.dashboard
│   ├── Dockerfile.mlflow
│   └── Dockerfile.training
├── config/config.yaml          # Central config
├── models/fraud_detector.pkl   # Trained model (baked into Docker)
├── render.yaml                 # Render deployment blueprint
├── docker-compose.yml          # Full local stack
└── .github/workflows/          # CI + CD pipelines
```

---

## Quickstart (Local)

### Prerequisites
- Python 3.11
- Docker Desktop (optional, for full stack)

### 1. Install dependencies

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 2. Run the full pipeline

```powershell
# Generate data
python scripts/train.py ingest

# Feature engineering
python scripts/train.py preprocess-cmd

# Train model + log to MLflow
python scripts/train.py train

# Start serving API
uvicorn src.serving.api:app --reload --port 8000

# Start dashboard (separate terminal)
uvicorn dashboard.main:app --reload --port 8050
```

Open **http://localhost:8050** for the dashboard.
Open **http://localhost:8000/docs** for the API playground.

### 3. Full Docker stack

```bash
docker compose up --build
```

Services started:
| Service | URL |
|---|---|
| Dashboard | http://localhost:8050 |
| ML API | http://localhost:8000 |
| MLflow UI | http://localhost:5000 |
| Prometheus | http://localhost:9090 |
| Grafana | http://localhost:3000 |
| Airflow | http://localhost:8080 |

---

## API Reference

### `POST /predict`
```json
{
  "transaction_id": "txn_001",
  "features": {
    "amount": 250.00,
    "merchant_category": "grocery",
    "distance_from_home": 2.5,
    "hour": 14,
    "day_of_week": 2,
    "is_chip": true,
    "is_pin_used": true,
    "credit_limit": 5000.0,
    "transaction_count_1h": 1
  }
}
```

**Response**:
```json
{
  "transaction_id": "txn_001",
  "fraud_probability": 0.032,
  "is_fraud": false,
  "confidence": "high",
  "model_version": "1"
}
```

### Other endpoints
| Endpoint | Method | Description |
|---|---|---|
| `/batch-predict` | POST | Predict on list of transactions |
| `/health` | GET | Service health + model info |
| `/metrics` | GET | Prometheus metrics |
| `/reload` | POST | Hot-reload model from disk |

---

## Features Engineered

| Feature | Description |
|---|---|
| `log_amount` | Log-transformed transaction amount |
| `is_night` | Transaction between 22:00–06:00 |
| `is_weekend` | Saturday or Sunday |
| `chip_and_pin` | Both chip and PIN used |
| `high_ratio` | Amount > 40% of credit limit |
| `far_from_home` | Distance > 50km from home |

---

## Model Performance

| Metric | Value |
|---|---|
| AUC-ROC | ~0.97 |
| AUC-PR | ~0.89 |
| Optimal threshold | ~0.35 |
| False positive rate | < 3% |

---

## Monitoring & Drift Detection

The monitoring DAG runs hourly and checks:
- **PSI** (Population Stability Index) for numeric features
- **KS-test** for distribution shift
- **Chi-squared** for categorical features

If the aggregate drift score exceeds **0.3**, the monitoring DAG automatically triggers the training pipeline to retrain with fresh data.

---

## Deploy to Render (Free)

1. Fork this repo
2. Go to **dashboard.render.com** → New → Blueprint
3. Connect the forked repo — Render detects `render.yaml` automatically
4. Click **Apply**

Render deploys:
- Free PostgreSQL database (90-day free trial)
- MLflow tracking server (backed by PostgreSQL)
- ML serving API
- Dashboard

No credit card required for the free tier.

---

## CI/CD

**CI** (on every push):
1. Ruff lint + Black format check
2. pytest test suite
3. Docker build verification

**CD** (on push to `main`):
1. Build and push Docker images to ECR
2. Update ECS service (zero-downtime rolling deploy)
3. Promote best MLflow model to production

---

## License

MIT
