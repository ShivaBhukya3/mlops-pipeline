"""Feature engineering and preprocessing pipeline."""

import logging
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib

logger = logging.getLogger(__name__)

FEATURE_COLS = [
    "amount",
    "hour",
    "day_of_week",
    "merchant_category",
    "distance_from_home",
    "distance_from_last_transaction",
    "ratio_to_median_purchase_price",
    "repeat_retailer",
    "used_chip",
    "used_pin_number",
    "online_order",
]
TARGET_COL = "fraud"
META_COLS = ["transaction_id", "timestamp"]


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived features on top of raw columns."""
    df = df.copy()

    df["log_amount"] = np.log1p(df["amount"])
    df["log_distance_home"] = np.log1p(df["distance_from_home"])
    df["log_distance_last"] = np.log1p(df["distance_from_last_transaction"])

    df["is_night"] = df["hour"].apply(lambda h: 1 if h < 6 or h >= 22 else 0)
    df["is_weekend"] = df["day_of_week"].apply(lambda d: 1 if d >= 5 else 0)

    df["chip_and_pin"] = (df["used_chip"] & df["used_pin_number"]).astype(int)
    df["no_security"] = (~df["used_chip"].astype(bool) & ~df["used_pin_number"].astype(bool)).astype(int)

    df["high_ratio"] = (df["ratio_to_median_purchase_price"] > 3).astype(int)
    df["far_from_home"] = (df["distance_from_home"] > 100).astype(int)

    return df


def get_feature_cols() -> List[str]:
    return FEATURE_COLS + [
        "log_amount",
        "log_distance_home",
        "log_distance_last",
        "is_night",
        "is_weekend",
        "chip_and_pin",
        "no_security",
        "high_ratio",
        "far_from_home",
    ]


def preprocess(
    df: pd.DataFrame,
    scaler: Optional[StandardScaler] = None,
    fit_scaler: bool = True,
) -> Tuple[pd.DataFrame, StandardScaler]:
    df = engineer_features(df)
    features = get_feature_cols()
    X = df[features].copy()

    numeric_scale = ["amount", "log_amount", "distance_from_home", "distance_from_last_transaction",
                     "log_distance_home", "log_distance_last", "ratio_to_median_purchase_price"]

    if fit_scaler:
        scaler = StandardScaler()
        X[numeric_scale] = scaler.fit_transform(X[numeric_scale])
    else:
        X[numeric_scale] = scaler.transform(X[numeric_scale])

    return X, scaler


def split_data(
    df: pd.DataFrame,
    test_size: float = 0.2,
    val_size: float = 0.1,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return train, val, test splits."""
    features = get_feature_cols()
    df = engineer_features(df)

    train_val, test = train_test_split(
        df, test_size=test_size, random_state=random_state, stratify=df[TARGET_COL]
    )
    val_ratio = val_size / (1 - test_size)
    train, val = train_test_split(
        train_val, test_size=val_ratio, random_state=random_state, stratify=train_val[TARGET_COL]
    )

    logger.info("Split: train=%d, val=%d, test=%d", len(train), len(val), len(test))
    return train, val, test


def save_splits(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    output_dir: str = "data/processed",
    scaler: Optional[StandardScaler] = None,
) -> None:
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    train.to_parquet(f"{output_dir}/train.parquet", index=False)
    val.to_parquet(f"{output_dir}/val.parquet", index=False)
    test.to_parquet(f"{output_dir}/test.parquet", index=False)
    if scaler is not None:
        joblib.dump(scaler, f"{output_dir}/scaler.pkl")
    logger.info("Saved splits to %s", output_dir)


def load_splits(processed_dir: str = "data/processed"):
    train = pd.read_parquet(f"{processed_dir}/train.parquet")
    val = pd.read_parquet(f"{processed_dir}/val.parquet")
    test = pd.read_parquet(f"{processed_dir}/test.parquet")
    scaler_path = Path(processed_dir) / "scaler.pkl"
    scaler = joblib.load(scaler_path) if scaler_path.exists() else None
    return train, val, test, scaler


def prepare_xy(df: pd.DataFrame):
    features = get_feature_cols()
    available = [c for c in features if c in df.columns]
    X = df[available]
    y = df[TARGET_COL] if TARGET_COL in df.columns else None
    return X, y
