"""Model evaluation utilities."""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score,
    precision_recall_curve, roc_curve, confusion_matrix,
)

logger = logging.getLogger(__name__)


def find_optimal_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Find threshold that maximises F1."""
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    f1_scores = 2 * precision * recall / (precision + recall + 1e-8)
    idx = np.argmax(f1_scores)
    return float(thresholds[idx]) if idx < len(thresholds) else 0.5


def evaluate_at_threshold(
    y_true: np.ndarray, y_prob: np.ndarray, threshold: float
) -> Dict[str, float]:
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "threshold": threshold,
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "avg_precision": float(average_precision_score(y_true, y_prob)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "precision": float(tp / (tp + fp + 1e-8)),
        "recall": float(tp / (tp + fn + 1e-8)),
        "specificity": float(tn / (tn + fp + 1e-8)),
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
        "fraud_detected_pct": float(tp / (tp + fn + 1e-8) * 100),
    }


def compute_business_impact(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    amounts: Optional[np.ndarray] = None,
    threshold: float = 0.5,
    review_cost: float = 5.0,
    avg_fraud_amount: float = 500.0,
) -> Dict[str, float]:
    """Estimate financial impact of the model vs no model."""
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    if amounts is None:
        amounts = np.full(len(y_true), avg_fraud_amount)

    fraud_idx = y_true == 1
    fraud_amounts = amounts[fraud_idx]

    detected_fraud_value = float(amounts[(y_true == 1) & (y_pred == 1)].sum())
    missed_fraud_value = float(amounts[(y_true == 1) & (y_pred == 0)].sum())
    false_alarm_cost = float(fp * review_cost)
    review_cost_total = float((tp + fp) * review_cost)

    net_savings = detected_fraud_value - review_cost_total

    return {
        "detected_fraud_value": detected_fraud_value,
        "missed_fraud_value": missed_fraud_value,
        "false_alarm_cost": false_alarm_cost,
        "net_savings": net_savings,
        "total_fraud_value": float(fraud_amounts.sum()),
        "fraud_capture_rate_pct": float(tp / (tp + fn + 1e-8) * 100),
    }


def get_roc_curve_data(y_true: np.ndarray, y_prob: np.ndarray) -> Dict:
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)
    return {"fpr": fpr.tolist(), "tpr": tpr.tolist(), "thresholds": thresholds.tolist(), "auc": auc}


def get_pr_curve_data(y_true: np.ndarray, y_prob: np.ndarray) -> Dict:
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    ap = average_precision_score(y_true, y_prob)
    return {
        "precision": precision.tolist(),
        "recall": recall.tolist(),
        "thresholds": thresholds.tolist(),
        "ap": ap,
    }
