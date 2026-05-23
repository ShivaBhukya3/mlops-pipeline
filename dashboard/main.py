"""Dashboard FastAPI backend — serves the UI and proxies data from the ML API."""

import asyncio
import logging
import os
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx
import numpy as np
import yaml
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.requests import Request

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

BASE_DIR = Path(__file__).resolve().parent
TEMPLATES_DIR = BASE_DIR / "templates"
STATIC_DIR = BASE_DIR / "static"

CONFIG_PATH = os.environ.get("CONFIG_PATH", "config/config.yaml")
ML_API_URL = os.environ.get("ML_API_URL", "http://localhost:8000")


def load_config():
    if Path(CONFIG_PATH).exists():
        with open(CONFIG_PATH) as f:
            return yaml.safe_load(f)
    return {}


app = FastAPI(
    title="MLOps Dashboard",
    description="Control Tower for the Fraud Detection ML Pipeline",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))


# ── Mock data generators ──────────────────────────────────────────────────────

def _mock_summary() -> Dict:
    rng = np.random.RandomState(int(time.time()) % 1000)
    total = int(rng.randint(800, 1200))
    fraud = int(rng.binomial(total, 0.022))
    return {
        "total_predictions": total,
        "fraud_detected": fraud,
        "fraud_rate_pct": round(fraud / total * 100, 2),
        "avg_fraud_probability": round(float(rng.uniform(0.04, 0.06)), 4),
        "avg_latency_ms": round(float(rng.uniform(2.8, 4.2)), 2),
        "p95_latency_ms": round(float(rng.uniform(7.0, 10.0)), 2),
        "p99_latency_ms": round(float(rng.uniform(12.0, 18.0)), 2),
        "requests_per_minute": round(float(rng.uniform(7.0, 12.0)), 2),
        "error_rate_pct": 0.0,
    }


def _mock_timeseries() -> Dict:
    rng = np.random.RandomState(int(time.time() // 60) % 100)
    buckets = []
    now = datetime.utcnow()
    for i in range(11, -1, -1):
        t = now - timedelta(minutes=i * 5)
        count = int(rng.randint(20, 80))
        fraud = int(rng.binomial(count, 0.022))
        buckets.append({
            "timestamp": t.isoformat(),
            "count": count,
            "fraud_count": fraud,
            "fraud_rate": round(fraud / count, 4),
            "avg_latency_ms": round(float(rng.uniform(2.5, 5.0)), 2),
        })
    return {"buckets": buckets, "bucket_minutes": 5}


def _mock_prob_dist() -> Dict:
    rng = np.random.RandomState(42)
    probs = np.concatenate([
        rng.beta(1, 20, 900),
        rng.beta(5, 2, 100),
    ])
    counts, edges = np.histogram(probs, bins=10, range=(0, 1))
    return {
        "bin_edges": edges.tolist(),
        "counts": counts.tolist(),
        "labels": [f"{edges[i]:.1f}-{edges[i+1]:.1f}" for i in range(len(edges) - 1)],
    }


def _mock_drift_report() -> Dict:
    features = ["amount", "hour", "distance_from_home", "distance_from_last_transaction",
                "ratio_to_median_purchase_price", "online_order", "repeat_retailer",
                "used_chip", "used_pin_number"]
    rng = np.random.RandomState(int(time.time() // 3600))
    results = []
    for feat in features:
        stat = float(rng.uniform(0.01, 0.18))
        pval = float(rng.uniform(0.06, 0.85))
        results.append({
            "feature": feat,
            "method": "chi2" if feat in ("online_order", "repeat_retailer", "used_chip", "used_pin_number") else "PSI+KS",
            "statistic": round(stat, 4),
            "p_value": round(pval, 4),
            "drifted": False,
            "severity": "none",
            "reference_mean": round(float(rng.uniform(5, 50)), 3),
            "current_mean": round(float(rng.uniform(5, 50)), 3),
            "relative_change_pct": round(float(rng.uniform(0, 8)), 2),
        })
    return {
        "timestamp": datetime.utcnow().isoformat(),
        "n_features": len(features),
        "n_drifted": 0,
        "overall_drift": False,
        "drift_score": 0.07,
        "feature_results": results,
    }


def _mock_model_versions() -> List[Dict]:
    return [
        {"version": "3", "stage": "Production", "run_id": "abc123def456", "creation_timestamp": int((datetime.utcnow() - timedelta(days=3)).timestamp() * 1000)},
        {"version": "2", "stage": "Staging",    "run_id": "def456ghi789", "creation_timestamp": int((datetime.utcnow() - timedelta(days=10)).timestamp() * 1000)},
        {"version": "1", "stage": "None",       "run_id": "ghi789jkl012", "creation_timestamp": int((datetime.utcnow() - timedelta(days=30)).timestamp() * 1000)},
    ]


def _mock_experiments() -> List[Dict]:
    return [
        {"run_name": "airflow-weekly-20240519", "status": "FINISHED", "auc": 0.9812, "f1": 0.8763, "precision": 0.9012, "recall": 0.8531, "train_time": 47.3, "model_type": "xgboost", "start_time": "2024-05-19 02:14:38"},
        {"run_name": "airflow-weekly-20240512", "status": "FINISHED", "auc": 0.9778, "f1": 0.8691, "precision": 0.8934, "recall": 0.8461, "train_time": 44.1, "model_type": "xgboost", "start_time": "2024-05-12 02:13:22"},
        {"run_name": "manual-rf-baseline",      "status": "FINISHED", "auc": 0.9540, "f1": 0.8321, "precision": 0.8789, "recall": 0.7894, "train_time": 38.9, "model_type": "random_forest", "start_time": "2024-05-10 15:32:11"},
        {"run_name": "logistic-baseline",       "status": "FINISHED", "auc": 0.9102, "f1": 0.7843, "precision": 0.8234, "recall": 0.7490, "train_time": 12.2, "model_type": "logistic", "start_time": "2024-05-08 09:11:43"},
    ]


def _mock_recent_predictions() -> List[Dict]:
    rng = np.random.RandomState(int(time.time()) % 500)
    rows = []
    for i in range(10):
        prob = float(rng.beta(1, 20)) if rng.random() > 0.03 else float(rng.beta(8, 2))
        rows.append({
            "transaction_id": f"TXN{rng.randint(1_000_000_000, 9_999_999_999)}",
            "timestamp": (datetime.utcnow() - timedelta(seconds=i * 12)).isoformat(),
            "fraud_probability": round(prob, 6),
            "is_fraud": prob >= 0.5,
            "confidence": "high" if prob > 0.8 or prob < 0.2 else "medium",
            "latency_ms": round(float(rng.uniform(2.0, 6.0)), 2),
            "model_version": "v3",
        })
    return rows


# ── Helper: proxy or fall back ────────────────────────────────────────────────
async def proxy_or_mock(path: str, fallback_fn):
    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            r = await client.get(f"{ML_API_URL}{path}")
            if r.status_code == 200:
                return r.json()
    except Exception:
        pass
    return fallback_fn()


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    try:
        return templates.TemplateResponse(request=request, name="index.html")
    except Exception:
        pass
    try:
        html = (TEMPLATES_DIR / "index.html").read_text(encoding="utf-8")
        return HTMLResponse(content=html)
    except Exception as e:
        return HTMLResponse(
            content=f"<h1>Dashboard load error</h1><pre>{e}</pre><p>TEMPLATES_DIR: {TEMPLATES_DIR}</p>",
            status_code=500,
        )


@app.get("/health")
async def health():
    return await proxy_or_mock("/health", lambda: {
        "status": "healthy", "model_loaded": True, "model_version": "v3", "uptime_seconds": 86400
    })


@app.get("/dashboard/summary")
async def summary():
    return await proxy_or_mock("/dashboard/summary", _mock_summary)


@app.get("/dashboard/timeseries")
async def timeseries():
    return _mock_timeseries()


@app.get("/dashboard/prob-distribution")
async def prob_distribution():
    return _mock_prob_dist()


@app.get("/dashboard/drift")
async def drift():
    try:
        from src.monitoring.drift_detector import DriftDetector
        import pandas as pd
        from pathlib import Path as P
        ref = P("data/reference/reference.parquet")
        cur = P("data/processed/test.parquet")
        if ref.exists() and cur.exists():
            ref_df = pd.read_parquet(ref)
            cur_df = pd.read_parquet(cur)
            feats = [c for c in ["amount", "hour", "distance_from_home"] if c in ref_df.columns]
            detector = DriftDetector()
            detector.fit_reference(ref_df[feats])
            report = detector.detect(cur_df[feats])
            return {
                "timestamp": report.timestamp,
                "overall_drift": report.overall_drift,
                "drift_score": report.drift_score,
                "n_features": len(report.feature_results),
                "n_drifted": len(report.drifted_features),
                "feature_results": [vars(r) for r in report.feature_results],
            }
    except Exception as e:
        logger.debug("Live drift detection failed: %s", e)
    return _mock_drift_report()


@app.get("/dashboard/model-versions")
async def model_versions():
    try:
        import mlflow
        cfg = load_config()
        mlflow.set_tracking_uri(cfg.get("mlflow", {}).get("tracking_uri", "mlruns"))
        client = mlflow.tracking.MlflowClient()
        model_name = cfg.get("mlflow", {}).get("model_name", "fraud-detector")
        versions = client.search_model_versions(f"name='{model_name}'")
        return [{"version": v.version, "stage": v.current_stage, "run_id": v.run_id, "creation_timestamp": v.creation_timestamp} for v in versions]
    except Exception as e:
        logger.debug("MLflow unavailable: %s", e)
    return _mock_model_versions()


@app.get("/dashboard/experiments")
async def experiments():
    try:
        import mlflow
        cfg = load_config()
        mlflow.set_tracking_uri(cfg.get("mlflow", {}).get("tracking_uri", "mlruns"))
        client = mlflow.tracking.MlflowClient()
        exp = client.get_experiment_by_name(cfg.get("mlflow", {}).get("experiment_name", "fraud-detection"))
        if exp:
            runs = client.search_runs(
                experiment_ids=[exp.experiment_id],
                order_by=["metrics.test_roc_auc DESC"],
                max_results=20,
            )
            return [
                {
                    "run_name": r.data.tags.get("mlflow.runName", r.info.run_id[:8]),
                    "status": r.info.status,
                    "auc": r.data.metrics.get("test_roc_auc", 0),
                    "f1": r.data.metrics.get("test_f1", 0),
                    "precision": r.data.metrics.get("test_precision", 0),
                    "recall": r.data.metrics.get("test_recall", 0),
                    "train_time": r.data.metrics.get("train_time_seconds", 0),
                    "model_type": r.data.params.get("model_type", "unknown"),
                    "start_time": datetime.fromtimestamp(r.info.start_time / 1000).strftime("%Y-%m-%d %H:%M:%S"),
                }
                for r in runs
            ]
    except Exception as e:
        logger.debug("MLflow experiments unavailable: %s", e)
    return _mock_experiments()


@app.get("/dashboard/recent-predictions")
async def recent_predictions():
    return _mock_recent_predictions()


@app.post("/predict")
async def predict_proxy(request: Request):
    body = await request.json()
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            r = await client.post(f"{ML_API_URL}/predict", json=body)
            if r.status_code == 200:
                return r.json()
    except Exception:
        pass

    features = body.get("features", {})
    prob = _simulate_prob(features)
    return {
        "transaction_id": body.get("transaction_id"),
        "fraud_probability": round(prob, 6),
        "is_fraud": prob >= 0.5,
        "confidence": "high" if prob > 0.8 or prob < 0.2 else "medium",
        "model_version": "v3-demo",
        "latency_ms": round(3.2 + float(np.random.uniform(0, 1.5)), 2),
    }


def _simulate_prob(f: dict) -> float:
    score = 0.0
    score += f.get("online_order", 0) * 0.25
    score += 0.20 if f.get("distance_from_home", 0) > 100 else 0
    score += 0.20 if f.get("distance_from_last_transaction", 0) > 100 else 0
    score += 0.20 if f.get("ratio_to_median_purchase_price", 1) > 5 else 0
    score += 0.10 if not f.get("used_chip", 1) else 0
    score += 0.10 if not f.get("repeat_retailer", 1) else 0
    hour = f.get("hour", 12)
    score += 0.10 if hour < 5 or hour > 22 else 0
    score += 0.10 if f.get("amount", 0) > 1000 else 0
    noise = float(np.random.uniform(-0.05, 0.05))
    return float(np.clip(score + noise, 0.01, 0.99))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("dashboard.main:app", host="0.0.0.0", port=8050, reload=True)
