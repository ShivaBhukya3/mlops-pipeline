"""Real-time metrics collection and storage."""

import json
import logging
import time
from collections import deque
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Deque, Dict, List, Optional

logger = logging.getLogger(__name__)

MAX_WINDOW = 10_000


@dataclass
class PredictionRecord:
    timestamp: str
    transaction_id: Optional[str]
    fraud_probability: float
    is_fraud: bool
    latency_ms: float
    model_version: str


@dataclass
class SystemMetrics:
    timestamp: str
    cpu_pct: float
    memory_pct: float
    requests_per_second: float
    avg_latency_ms: float
    p99_latency_ms: float
    error_rate: float


class MetricsCollector:
    def __init__(self, persist_path: str = "data/metrics"):
        self.predictions: Deque[PredictionRecord] = deque(maxlen=MAX_WINDOW)
        self.system_metrics: Deque[SystemMetrics] = deque(maxlen=1440)  # 24h at 1/min
        self.persist_path = Path(persist_path)
        self.persist_path.mkdir(parents=True, exist_ok=True)
        self._request_times: Deque[float] = deque(maxlen=1000)
        self._error_count: int = 0
        self._total_requests: int = 0

    def record_prediction(
        self,
        transaction_id: Optional[str],
        fraud_probability: float,
        is_fraud: bool,
        latency_ms: float,
        model_version: str,
    ) -> None:
        record = PredictionRecord(
            timestamp=datetime.utcnow().isoformat(),
            transaction_id=transaction_id,
            fraud_probability=fraud_probability,
            is_fraud=is_fraud,
            latency_ms=latency_ms,
            model_version=model_version,
        )
        self.predictions.append(record)
        self._request_times.append(latency_ms)
        self._total_requests += 1

    def record_error(self) -> None:
        self._error_count += 1
        self._total_requests += 1

    def get_summary(self, window_minutes: int = 60) -> Dict:
        cutoff = datetime.utcnow() - timedelta(minutes=window_minutes)
        recent = [
            p for p in self.predictions
            if datetime.fromisoformat(p.timestamp) >= cutoff
        ]

        if not recent:
            return self._empty_summary(window_minutes)

        probs = [p.fraud_probability for p in recent]
        latencies = [p.latency_ms for p in recent]
        fraud_count = sum(1 for p in recent if p.is_fraud)

        return {
            "window_minutes": window_minutes,
            "total_predictions": len(recent),
            "fraud_detected": fraud_count,
            "fraud_rate_pct": round(fraud_count / len(recent) * 100, 2),
            "avg_fraud_probability": round(float(sum(probs) / len(probs)), 4),
            "avg_latency_ms": round(float(sum(latencies) / len(latencies)), 2),
            "p95_latency_ms": round(sorted(latencies)[int(len(latencies) * 0.95)], 2),
            "p99_latency_ms": round(sorted(latencies)[int(len(latencies) * 0.99)], 2),
            "requests_per_minute": round(len(recent) / window_minutes, 2),
            "error_rate_pct": round(self._error_count / max(self._total_requests, 1) * 100, 2),
        }

    def get_timeseries(self, window_minutes: int = 60, bucket_minutes: int = 5) -> Dict:
        cutoff = datetime.utcnow() - timedelta(minutes=window_minutes)
        recent = [
            p for p in self.predictions
            if datetime.fromisoformat(p.timestamp) >= cutoff
        ]

        buckets: Dict[str, List] = {}
        for p in recent:
            dt = datetime.fromisoformat(p.timestamp)
            bucket_key = dt.replace(
                minute=(dt.minute // bucket_minutes) * bucket_minutes,
                second=0, microsecond=0
            ).isoformat()
            buckets.setdefault(bucket_key, []).append(p)

        timeseries = []
        for ts, records in sorted(buckets.items()):
            fraud = sum(1 for r in records if r.is_fraud)
            latencies = [r.latency_ms for r in records]
            timeseries.append({
                "timestamp": ts,
                "count": len(records),
                "fraud_count": fraud,
                "fraud_rate": round(fraud / len(records), 4),
                "avg_latency_ms": round(sum(latencies) / len(latencies), 2),
            })

        return {"buckets": timeseries, "bucket_minutes": bucket_minutes}

    def _empty_summary(self, window_minutes: int) -> Dict:
        return {
            "window_minutes": window_minutes,
            "total_predictions": 0,
            "fraud_detected": 0,
            "fraud_rate_pct": 0,
            "avg_fraud_probability": 0,
            "avg_latency_ms": 0,
            "p95_latency_ms": 0,
            "p99_latency_ms": 0,
            "requests_per_minute": 0,
            "error_rate_pct": 0,
        }

    def persist_snapshot(self) -> None:
        snapshot_path = self.persist_path / f"snapshot_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
        summary = self.get_summary(window_minutes=60)
        with open(snapshot_path, "w") as f:
            json.dump(summary, f, indent=2)
        logger.info("Metrics snapshot saved to %s", snapshot_path)

    def get_fraud_probability_distribution(self, bins: int = 10) -> Dict:
        probs = [p.fraud_probability for p in self.predictions]
        if not probs:
            return {"bins": [], "counts": []}
        import numpy as np
        counts, edges = np.histogram(probs, bins=bins, range=(0, 1))
        return {
            "bin_edges": edges.tolist(),
            "counts": counts.tolist(),
            "labels": [f"{edges[i]:.1f}-{edges[i+1]:.1f}" for i in range(len(edges) - 1)],
        }


# Global singleton
_collector: Optional[MetricsCollector] = None


def get_collector() -> MetricsCollector:
    global _collector
    if _collector is None:
        _collector = MetricsCollector()
    return _collector
