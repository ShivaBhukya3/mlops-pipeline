"""Tests for model training, evaluation, and registry."""

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier

from src.data.ingestion import generate_fraud_dataset
from src.data.preprocessing import engineer_features, get_feature_cols, prepare_xy, split_data
from src.models.evaluator import (
    compute_business_impact,
    evaluate_at_threshold,
    find_optimal_threshold,
    get_pr_curve_data,
    get_roc_curve_data,
)


@pytest.fixture(scope="module")
def trained_rf():
    df = generate_fraud_dataset(n_samples=5_000, fraud_ratio=0.05, random_state=42)
    train, _, test = split_data(df, test_size=0.2, val_size=0.1)
    X_train, y_train = prepare_xy(train)
    X_test, y_test = prepare_xy(test)
    model = RandomForestClassifier(n_estimators=10, class_weight="balanced", random_state=42)
    model.fit(X_train, y_train)
    y_prob = model.predict_proba(X_test)[:, 1]
    return model, X_test, y_test.values, y_prob


class TestEvaluator:
    def test_optimal_threshold_in_range(self, trained_rf):
        _, _, y_test, y_prob = trained_rf
        threshold = find_optimal_threshold(y_test, y_prob)
        assert 0.0 <= threshold <= 1.0

    def test_evaluate_at_threshold_keys(self, trained_rf):
        _, _, y_test, y_prob = trained_rf
        metrics = evaluate_at_threshold(y_test, y_prob, threshold=0.5)
        required = ["roc_auc", "avg_precision", "f1", "precision", "recall", "tp", "fp", "tn", "fn"]
        for key in required:
            assert key in metrics

    def test_roc_auc_above_random(self, trained_rf):
        _, _, y_test, y_prob = trained_rf
        metrics = evaluate_at_threshold(y_test, y_prob, threshold=0.5)
        assert metrics["roc_auc"] > 0.7, "Model should beat random"

    def test_business_impact_keys(self, trained_rf):
        _, _, y_test, y_prob = trained_rf
        impact = compute_business_impact(y_test, y_prob, threshold=0.5)
        assert "net_savings" in impact
        assert "fraud_capture_rate_pct" in impact

    def test_roc_curve_data(self, trained_rf):
        _, _, y_test, y_prob = trained_rf
        data = get_roc_curve_data(y_test, y_prob)
        assert "fpr" in data and "tpr" in data and "auc" in data
        assert 0.5 <= data["auc"] <= 1.0

    def test_pr_curve_data(self, trained_rf):
        _, _, y_test, y_prob = trained_rf
        data = get_pr_curve_data(y_test, y_prob)
        assert "precision" in data and "recall" in data and "ap" in data
        assert 0 <= data["ap"] <= 1.0


class TestDriftDetector:
    def test_no_drift_on_same_data(self):
        from src.monitoring.drift_detector import DriftDetector
        rng = np.random.RandomState(0)
        df = pd.DataFrame({"amount": rng.lognormal(3.5, 1.2, 500), "hour": rng.randint(0, 24, 500)})
        detector = DriftDetector(psi_threshold=0.2)
        detector.fit_reference(df)
        report = detector.detect(df.copy(), features=["amount", "hour"])
        assert not report.overall_drift

    def test_drift_detected_on_shifted_data(self):
        from src.monitoring.drift_detector import DriftDetector
        rng = np.random.RandomState(0)
        ref = pd.DataFrame({"amount": rng.lognormal(3.5, 1.2, 500)})
        cur = pd.DataFrame({"amount": rng.lognormal(6.0, 1.5, 500)})  # severely shifted
        detector = DriftDetector(psi_threshold=0.1)
        detector.fit_reference(ref)
        report = detector.detect(cur, features=["amount"])
        assert report.overall_drift

    def test_report_has_all_features(self):
        from src.monitoring.drift_detector import DriftDetector
        rng = np.random.RandomState(1)
        df = pd.DataFrame({
            "a": rng.normal(0, 1, 200),
            "b": rng.normal(5, 2, 200),
        })
        detector = DriftDetector()
        detector.fit_reference(df)
        report = detector.detect(df.copy(), features=["a", "b"])
        assert len(report.feature_results) == 2
