"""Data drift detection using PSI, KS test, and statistical methods."""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


@dataclass
class DriftResult:
    feature: str
    method: str
    statistic: float
    p_value: Optional[float]
    drifted: bool
    severity: str  # "none" | "low" | "medium" | "high"
    reference_mean: float
    current_mean: float
    relative_change_pct: float


@dataclass
class DriftReport:
    timestamp: str
    n_reference: int
    n_current: int
    feature_results: List[DriftResult] = field(default_factory=list)
    overall_drift: bool = False
    drift_score: float = 0.0
    drifted_features: List[str] = field(default_factory=list)
    model_performance_degraded: bool = False


def compute_psi(reference: np.ndarray, current: np.ndarray, bins: int = 10) -> float:
    """Population Stability Index (PSI)."""
    ref_min, ref_max = reference.min(), reference.max()
    if ref_max == ref_min:
        return 0.0

    bin_edges = np.linspace(ref_min, ref_max, bins + 1)
    ref_counts = np.histogram(reference, bins=bin_edges)[0]
    cur_counts = np.histogram(current, bins=bin_edges)[0]

    ref_pct = (ref_counts / len(reference)).clip(min=1e-4)
    cur_pct = (cur_counts / len(current)).clip(min=1e-4)

    psi = np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct))
    return float(psi)


def ks_test(reference: np.ndarray, current: np.ndarray) -> Tuple[float, float]:
    stat, p_val = stats.ks_2samp(reference, current)
    return float(stat), float(p_val)


def chi2_test(reference: np.ndarray, current: np.ndarray) -> Tuple[float, float]:
    """Chi-squared test for categorical features."""
    all_cats = np.union1d(np.unique(reference), np.unique(current))
    ref_counts = np.array([np.sum(reference == c) for c in all_cats], dtype=float)
    cur_counts = np.array([np.sum(current == c) for c in all_cats], dtype=float)
    ref_counts = (ref_counts / ref_counts.sum() * len(current)).clip(min=0.5)
    stat, p_val = stats.chisquare(cur_counts, f_exp=ref_counts)
    return float(stat), float(p_val)


def psi_severity(psi: float) -> str:
    if psi < 0.1:
        return "none"
    elif psi < 0.2:
        return "low"
    elif psi < 0.25:
        return "medium"
    return "high"


class DriftDetector:
    def __init__(
        self,
        psi_threshold: float = 0.2,
        ks_threshold: float = 0.05,
        categorical_features: Optional[List[str]] = None,
    ):
        self.psi_threshold = psi_threshold
        self.ks_threshold = ks_threshold
        self.categorical_features = categorical_features or [
            "merchant_category", "day_of_week", "hour",
            "repeat_retailer", "used_chip", "used_pin_number", "online_order",
        ]
        self.reference_data: Optional[pd.DataFrame] = None

    def fit_reference(self, reference: pd.DataFrame) -> None:
        self.reference_data = reference.copy()
        logger.info("Reference data fitted: %d rows, %d cols", len(reference), len(reference.columns))

    def detect(
        self,
        current: pd.DataFrame,
        features: Optional[List[str]] = None,
    ) -> DriftReport:
        if self.reference_data is None:
            raise ValueError("Call fit_reference() first")

        from datetime import datetime
        report = DriftReport(
            timestamp=datetime.utcnow().isoformat(),
            n_reference=len(self.reference_data),
            n_current=len(current),
        )

        features = features or [c for c in self.reference_data.columns if c in current.columns]
        numeric_features = [f for f in features if f not in self.categorical_features]

        for feat in numeric_features:
            if feat not in current.columns:
                continue
            ref_arr = self.reference_data[feat].dropna().values
            cur_arr = current[feat].dropna().values

            psi = compute_psi(ref_arr, cur_arr)
            ks_stat, ks_pval = ks_test(ref_arr, cur_arr)
            drifted = psi > self.psi_threshold or ks_pval < self.ks_threshold

            ref_mean = float(ref_arr.mean())
            cur_mean = float(cur_arr.mean())
            rel_change = abs(cur_mean - ref_mean) / (abs(ref_mean) + 1e-8) * 100

            result = DriftResult(
                feature=feat,
                method="PSI+KS",
                statistic=psi,
                p_value=ks_pval,
                drifted=drifted,
                severity=psi_severity(psi),
                reference_mean=ref_mean,
                current_mean=cur_mean,
                relative_change_pct=round(rel_change, 2),
            )
            report.feature_results.append(result)

        for feat in self.categorical_features:
            if feat not in current.columns or feat not in self.reference_data.columns:
                continue
            ref_arr = self.reference_data[feat].dropna().values
            cur_arr = current[feat].dropna().values

            try:
                chi_stat, chi_pval = chi2_test(ref_arr, cur_arr)
            except Exception:
                chi_stat, chi_pval = 0.0, 1.0

            drifted = chi_pval < self.ks_threshold
            ref_mode = float(stats.mode(ref_arr, keepdims=True).mode[0])
            cur_mode = float(stats.mode(cur_arr, keepdims=True).mode[0])

            result = DriftResult(
                feature=feat,
                method="chi2",
                statistic=chi_stat,
                p_value=chi_pval,
                drifted=drifted,
                severity="high" if drifted else "none",
                reference_mean=ref_mode,
                current_mean=cur_mode,
                relative_change_pct=0.0,
            )
            report.feature_results.append(result)

        drifted = [r for r in report.feature_results if r.drifted]
        report.drifted_features = [r.feature for r in drifted]
        report.overall_drift = len(drifted) > 0
        report.drift_score = len(drifted) / max(len(report.feature_results), 1)

        if report.overall_drift:
            logger.warning(
                "DRIFT DETECTED: %d/%d features drifted (score=%.2f)",
                len(drifted), len(report.feature_results), report.drift_score,
            )
        else:
            logger.info("No significant drift detected (score=%.2f)", report.drift_score)

        return report

    def to_dataframe(self, report: DriftReport) -> pd.DataFrame:
        return pd.DataFrame([
            {
                "feature": r.feature,
                "method": r.method,
                "statistic": r.statistic,
                "p_value": r.p_value,
                "drifted": r.drifted,
                "severity": r.severity,
                "reference_mean": r.reference_mean,
                "current_mean": r.current_mean,
                "relative_change_pct": r.relative_change_pct,
            }
            for r in report.feature_results
        ])
