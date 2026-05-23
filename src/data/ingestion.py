"""Data ingestion: synthetic fraud dataset generation and loading."""

import logging
import os
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import yaml

logger = logging.getLogger(__name__)


def load_config(config_path: str = "config/config.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def generate_fraud_dataset(
    n_samples: int = 100_000,
    fraud_ratio: float = 0.02,
    random_state: int = 42,
) -> pd.DataFrame:
    """Generate a realistic synthetic fraud detection dataset."""
    rng = np.random.RandomState(random_state)
    n_fraud = int(n_samples * fraud_ratio)
    n_legit = n_samples - n_fraud

    def make_legit(n: int) -> dict:
        return {
            "amount": rng.lognormal(mean=3.5, sigma=1.2, size=n).clip(0.5, 5000),
            "hour": rng.choice(range(24), size=n, p=_hour_probs(rng)),
            "day_of_week": rng.randint(0, 7, size=n),
            "merchant_category": rng.randint(0, 20, size=n),
            "distance_from_home": rng.exponential(scale=20, size=n).clip(0, 500),
            "distance_from_last_transaction": rng.exponential(scale=5, size=n).clip(0, 200),
            "ratio_to_median_purchase_price": rng.lognormal(mean=0, sigma=0.5, size=n).clip(0.1, 10),
            "repeat_retailer": rng.binomial(1, 0.85, size=n),
            "used_chip": rng.binomial(1, 0.75, size=n),
            "used_pin_number": rng.binomial(1, 0.55, size=n),
            "online_order": rng.binomial(1, 0.35, size=n),
            "fraud": np.zeros(n, dtype=int),
        }

    def make_fraud(n: int) -> dict:
        return {
            "amount": rng.lognormal(mean=4.5, sigma=1.8, size=n).clip(1, 10000),
            "hour": rng.choice(range(24), size=n, p=_fraud_hour_probs()),
            "day_of_week": rng.randint(0, 7, size=n),
            "merchant_category": rng.randint(0, 20, size=n),
            "distance_from_home": rng.exponential(scale=80, size=n).clip(0, 2000),
            "distance_from_last_transaction": rng.exponential(scale=60, size=n).clip(0, 2000),
            "ratio_to_median_purchase_price": rng.lognormal(mean=1.5, sigma=1.2, size=n).clip(0.5, 50),
            "repeat_retailer": rng.binomial(1, 0.25, size=n),
            "used_chip": rng.binomial(1, 0.15, size=n),
            "used_pin_number": rng.binomial(1, 0.10, size=n),
            "online_order": rng.binomial(1, 0.80, size=n),
            "fraud": np.ones(n, dtype=int),
        }

    legit = pd.DataFrame(make_legit(n_legit))
    fraud_df = pd.DataFrame(make_fraud(n_fraud))
    df = pd.concat([legit, fraud_df], ignore_index=True)
    df = df.sample(frac=1, random_state=random_state).reset_index(drop=True)

    df["transaction_id"] = [f"TXN{i:010d}" for i in range(len(df))]
    df["timestamp"] = pd.date_range(
        start="2024-01-01", periods=len(df), freq="1min"
    )

    logger.info(
        "Generated %d transactions (%d fraud, %.2f%% fraud rate)",
        len(df), fraud_df.shape[0], fraud_ratio * 100,
    )
    return df


def _hour_probs(rng: np.random.RandomState) -> np.ndarray:
    base = np.ones(24)
    base[9:18] = 3.0
    base[18:22] = 2.0
    base[0:6] = 0.3
    return base / base.sum()


def _fraud_hour_probs() -> np.ndarray:
    base = np.ones(24)
    base[0:6] = 3.5
    base[22:24] = 2.5
    base[9:17] = 0.5
    return base / base.sum()


def ingest_data(
    output_dir: str = "data/raw",
    config_path: str = "config/config.yaml",
    force: bool = False,
) -> str:
    """Generate (or reload) dataset and save to parquet."""
    cfg = load_config(config_path)
    data_cfg = cfg["data"]
    output_path = Path(output_dir) / "transactions.parquet"

    if output_path.exists() and not force:
        logger.info("Raw data already exists at %s, skipping generation.", output_path)
        return str(output_path)

    df = generate_fraud_dataset(
        n_samples=data_cfg["n_samples"],
        fraud_ratio=data_cfg["fraud_ratio"],
        random_state=data_cfg["random_state"],
    )

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    logger.info("Saved raw data to %s (%d rows)", output_path, len(df))
    return str(output_path)


def load_raw_data(path: str = "data/raw/transactions.parquet") -> pd.DataFrame:
    df = pd.read_parquet(path)
    logger.info("Loaded %d rows from %s", len(df), path)
    return df


def stream_batch(
    df: pd.DataFrame,
    batch_size: int = 100,
    start_idx: int = 0,
) -> Tuple[pd.DataFrame, int]:
    """Yield the next batch for simulating real-time inference."""
    end_idx = min(start_idx + batch_size, len(df))
    batch = df.iloc[start_idx:end_idx].copy()
    return batch, end_idx
