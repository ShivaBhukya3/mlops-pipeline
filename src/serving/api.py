"""FastAPI ML serving application with full observability."""

import logging
import os
import time
from contextlib import asynccontextmanager
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import uvicorn
import yaml
from fastapi import FastAPI, HTTPException, Request, Response, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST

from src.serving.schemas import (
    PredictionRequest, PredictionResponse,
    BatchPredictionRequest, BatchPredictionResponse,
    HealthResponse, ModelInfoResponse,
)
from src.data.preprocessing import get_feature_cols, engineer_features

logger = logging.getLogger(__name__)

# ── Prometheus metrics ────────────────────────────────────────────────────────
PREDICTION_COUNT = Counter("predictions_total", "Total predictions", ["result"])
PREDICTION_LATENCY = Histogram("prediction_latency_seconds", "Prediction latency", buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0])
FRAUD_PROBABILITY = Histogram("fraud_probability", "Distribution of fraud probabilities", buckets=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
MODEL_LOADED = Gauge("model_loaded", "Whether model is loaded (1=yes)")
ACTIVE_REQUESTS = Gauge("active_requests", "Number of active requests")


# ── App state ─────────────────────────────────────────────────────────────────
class AppState:
    model: Any = None
    model_version: str = "unknown"
    model_stage: str = "unknown"
    threshold: float = 0.5
    start_time: float = time.time()
    config: dict = {}
    features: list = []


state = AppState()


def load_config() -> dict:
    config_path = os.environ.get("CONFIG_PATH", "config/config.yaml")
    with open(config_path) as f:
        return yaml.safe_load(f)


def load_model_from_registry(cfg: dict) -> tuple:
    """Load model from MLflow registry or fall back to local pickle."""
    mlflow_cfg = cfg.get("mlflow", {})
    tracking_uri = mlflow_cfg.get("tracking_uri", "mlruns")
    model_name = mlflow_cfg.get("model_name", "fraud-detector")

    try:
        import mlflow
        import mlflow.sklearn
        mlflow.set_tracking_uri(tracking_uri)
        client = mlflow.tracking.MlflowClient()
        stage = cfg["serving"].get("model_stage", "Production")
        versions = client.get_latest_versions(model_name, stages=[stage])
        if not versions:
            versions = client.get_latest_versions(model_name, stages=["None"])
        if versions:
            ver = versions[0]
            model = mlflow.sklearn.load_model(f"models:/{model_name}/{ver.version}")
            return model, str(ver.version), stage
    except Exception as e:
        logger.warning("MLflow load failed (%s), trying local fallback", e)

    local_path = "models/fraud_detector.pkl"
    if os.path.exists(local_path):
        import joblib
        model = joblib.load(local_path)
        return model, "local", "local"

    raise RuntimeError("No model found. Run `python scripts/train.py train` first.")


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting MLOps serving API...")
    state.config = load_config()
    state.threshold = state.config["serving"].get("prediction_threshold", 0.5)
    state.features = get_feature_cols()

    try:
        state.model, state.model_version, state.model_stage = load_model_from_registry(state.config)
        MODEL_LOADED.set(1)
        logger.info("Model v%s (%s) loaded successfully", state.model_version, state.model_stage)
    except Exception as e:
        logger.error("Model load failed: %s", e)
        MODEL_LOADED.set(0)

    yield

    logger.info("Shutting down API")


app = FastAPI(
    title="MLOps Fraud Detection API",
    description="Real-time fraud detection powered by XGBoost + MLflow",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(GZipMiddleware, minimum_size=1000)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def track_requests(request: Request, call_next):
    ACTIVE_REQUESTS.inc()
    t0 = time.time()
    try:
        response = await call_next(request)
        return response
    finally:
        ACTIVE_REQUESTS.dec()
        elapsed = time.time() - t0
        if request.url.path in ["/predict", "/batch-predict"]:
            PREDICTION_LATENCY.observe(elapsed)


def _prepare_features(features_dict: dict) -> pd.DataFrame:
    df = pd.DataFrame([features_dict])
    df = engineer_features(df)
    available = [c for c in state.features if c in df.columns]
    return df[available]


def _build_response(
    transaction_id: Optional[str],
    prob: float,
    latency_ms: float,
) -> PredictionResponse:
    is_fraud = prob >= state.threshold
    if prob < 0.3:
        confidence = "high"
    elif prob < 0.6:
        confidence = "medium"
    else:
        confidence = "high" if prob > 0.8 else "medium"

    PREDICTION_COUNT.labels(result="fraud" if is_fraud else "legitimate").inc()
    FRAUD_PROBABILITY.observe(prob)

    return PredictionResponse(
        transaction_id=transaction_id,
        fraud_probability=round(prob, 6),
        is_fraud=is_fraud,
        confidence=confidence,
        model_version=state.model_version,
        latency_ms=round(latency_ms, 2),
    )


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["ops"])
async def health():
    return HealthResponse(
        status="healthy" if state.model else "degraded",
        model_loaded=state.model is not None,
        model_version=state.model_version if state.model else None,
        uptime_seconds=round(time.time() - state.start_time, 1),
    )


@app.get("/metrics", tags=["ops"])
async def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/model/info", response_model=ModelInfoResponse, tags=["model"])
async def model_info():
    if not state.model:
        raise HTTPException(503, "Model not loaded")
    return ModelInfoResponse(
        model_name=state.config["mlflow"]["model_name"],
        model_version=state.model_version,
        stage=state.model_stage,
        metrics={},
        features=state.features,
        threshold=state.threshold,
    )


@app.post("/predict", response_model=PredictionResponse, tags=["inference"])
async def predict(request: PredictionRequest):
    if not state.model:
        raise HTTPException(503, "Model not loaded")

    t0 = time.perf_counter()
    try:
        X = _prepare_features(request.features.model_dump())
        prob = float(state.model.predict_proba(X)[0, 1])
    except Exception as e:
        logger.exception("Prediction error: %s", e)
        raise HTTPException(500, f"Prediction failed: {e}")

    latency_ms = (time.perf_counter() - t0) * 1000
    return _build_response(request.transaction_id, prob, latency_ms)


@app.post("/batch-predict", response_model=BatchPredictionResponse, tags=["inference"])
async def batch_predict(request: BatchPredictionRequest):
    if not state.model:
        raise HTTPException(503, "Model not loaded")

    t0 = time.perf_counter()
    rows = [r.features.model_dump() for r in request.transactions]
    df = pd.DataFrame(rows)
    df = engineer_features(df)
    available = [c for c in state.features if c in df.columns]
    X = df[available]

    try:
        probs = state.model.predict_proba(X)[:, 1]
    except Exception as e:
        raise HTTPException(500, f"Batch prediction failed: {e}")

    total_ms = (time.perf_counter() - t0) * 1000
    per_ms = total_ms / len(probs)

    predictions = []
    for i, (txn, prob) in enumerate(zip(request.transactions, probs)):
        predictions.append(_build_response(txn.transaction_id, float(prob), per_ms))

    fraud_count = sum(1 for p in predictions if p.is_fraud)
    return BatchPredictionResponse(
        predictions=predictions,
        total_transactions=len(predictions),
        fraud_detected=fraud_count,
        processing_time_ms=round(total_ms, 2),
    )


@app.post("/reload", tags=["ops"])
async def reload_model():
    """Hot-reload the model from registry."""
    try:
        state.model, state.model_version, state.model_stage = load_model_from_registry(state.config)
        MODEL_LOADED.set(1)
        return {"status": "reloaded", "version": state.model_version}
    except Exception as e:
        MODEL_LOADED.set(0)
        raise HTTPException(500, f"Reload failed: {e}")


if __name__ == "__main__":
    uvicorn.run("src.serving.api:app", host="0.0.0.0", port=8000, reload=True)
