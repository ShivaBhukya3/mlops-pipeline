"""Data quality validation using Great Expectations."""

import logging
from typing import Dict, Any, List

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


SCHEMA = {
    "amount": {"type": float, "min": 0, "max": 50_000, "nullable": False},
    "hour": {"type": int, "min": 0, "max": 23, "nullable": False},
    "day_of_week": {"type": int, "min": 0, "max": 6, "nullable": False},
    "merchant_category": {"type": int, "min": 0, "max": 50, "nullable": False},
    "distance_from_home": {"type": float, "min": 0, "nullable": False},
    "distance_from_last_transaction": {"type": float, "min": 0, "nullable": False},
    "ratio_to_median_purchase_price": {"type": float, "min": 0, "nullable": False},
    "repeat_retailer": {"type": int, "min": 0, "max": 1, "nullable": False},
    "used_chip": {"type": int, "min": 0, "max": 1, "nullable": False},
    "used_pin_number": {"type": int, "min": 0, "max": 1, "nullable": False},
    "online_order": {"type": int, "min": 0, "max": 1, "nullable": False},
}


class DataValidationError(Exception):
    pass


class DataValidator:
    def __init__(self, schema: Dict[str, Any] = SCHEMA):
        self.schema = schema
        self.results: List[Dict] = []

    def validate(self, df: pd.DataFrame, raise_on_error: bool = False) -> bool:
        self.results = []
        passed = True

        for col, rules in self.schema.items():
            if col not in df.columns:
                self.results.append({"check": f"{col}_exists", "status": "FAIL", "detail": "Column missing"})
                passed = False
                continue

            series = df[col]

            if not rules.get("nullable", True) and series.isnull().any():
                n_null = series.isnull().sum()
                self.results.append({"check": f"{col}_null", "status": "FAIL", "detail": f"{n_null} nulls"})
                passed = False
            else:
                self.results.append({"check": f"{col}_null", "status": "PASS", "detail": ""})

            if "min" in rules and series.min() < rules["min"]:
                self.results.append({"check": f"{col}_min", "status": "FAIL", "detail": f"min={series.min():.4f} < {rules['min']}"})
                passed = False
            elif "min" in rules:
                self.results.append({"check": f"{col}_min", "status": "PASS", "detail": ""})

            if "max" in rules and series.max() > rules["max"]:
                self.results.append({"check": f"{col}_max", "status": "FAIL", "detail": f"max={series.max():.4f} > {rules['max']}"})
                passed = False
            elif "max" in rules:
                self.results.append({"check": f"{col}_max", "status": "PASS", "detail": ""})

        if passed:
            logger.info("Data validation PASSED (%d checks)", len(self.results))
        else:
            fails = [r for r in self.results if r["status"] == "FAIL"]
            logger.warning("Data validation FAILED: %d issues", len(fails))
            if raise_on_error:
                raise DataValidationError(f"Validation failed: {fails}")

        return passed

    def summary(self) -> pd.DataFrame:
        return pd.DataFrame(self.results)

    def validate_schema_drift(
        self, reference: pd.DataFrame, current: pd.DataFrame
    ) -> Dict[str, Any]:
        new_cols = set(current.columns) - set(reference.columns)
        removed_cols = set(reference.columns) - set(current.columns)
        type_changes = {}

        for col in reference.columns & current.columns:
            if reference[col].dtype != current[col].dtype:
                type_changes[col] = {
                    "reference": str(reference[col].dtype),
                    "current": str(current[col].dtype),
                }

        return {
            "new_columns": list(new_cols),
            "removed_columns": list(removed_cols),
            "type_changes": type_changes,
            "has_drift": bool(new_cols or removed_cols or type_changes),
        }
