"""Tests for data ingestion, preprocessing, and validation."""

import numpy as np
import pandas as pd
import pytest

from src.data.ingestion import generate_fraud_dataset
from src.data.preprocessing import engineer_features, get_feature_cols, split_data, prepare_xy
from src.data.validation import DataValidator


@pytest.fixture
def small_df():
    return generate_fraud_dataset(n_samples=1_000, fraud_ratio=0.05, random_state=0)


class TestDataIngestion:
    def test_generates_correct_shape(self, small_df):
        assert len(small_df) == 1_000

    def test_fraud_ratio_approximate(self, small_df):
        ratio = small_df["fraud"].mean()
        assert 0.02 <= ratio <= 0.10

    def test_required_columns(self, small_df):
        required = [
            "amount", "hour", "day_of_week", "merchant_category",
            "distance_from_home", "distance_from_last_transaction",
            "ratio_to_median_purchase_price", "repeat_retailer",
            "used_chip", "used_pin_number", "online_order", "fraud",
            "transaction_id", "timestamp",
        ]
        for col in required:
            assert col in small_df.columns, f"Missing column: {col}"

    def test_hour_range(self, small_df):
        assert small_df["hour"].between(0, 23).all()

    def test_day_of_week_range(self, small_df):
        assert small_df["day_of_week"].between(0, 6).all()

    def test_amounts_positive(self, small_df):
        assert (small_df["amount"] > 0).all()

    def test_binary_columns(self, small_df):
        for col in ["repeat_retailer", "used_chip", "used_pin_number", "online_order", "fraud"]:
            assert set(small_df[col].unique()).issubset({0, 1})


class TestPreprocessing:
    def test_feature_engineering_adds_columns(self, small_df):
        df = engineer_features(small_df)
        new_cols = ["log_amount", "is_night", "is_weekend", "chip_and_pin", "high_ratio"]
        for col in new_cols:
            assert col in df.columns

    def test_log_amount_non_negative(self, small_df):
        df = engineer_features(small_df)
        assert (df["log_amount"] >= 0).all()

    def test_is_night_binary(self, small_df):
        df = engineer_features(small_df)
        assert set(df["is_night"].unique()).issubset({0, 1})

    def test_split_sizes(self, small_df):
        train, val, test = split_data(small_df, test_size=0.2, val_size=0.1)
        total = len(train) + len(val) + len(test)
        assert total == len(small_df)
        assert len(test) / total == pytest.approx(0.2, abs=0.02)

    def test_split_stratified(self, small_df):
        train, val, test = split_data(small_df)
        for split in [train, val, test]:
            rate = split["fraud"].mean()
            assert 0.01 <= rate <= 0.15, f"Unexpected fraud rate in split: {rate}"

    def test_prepare_xy(self, small_df):
        df = engineer_features(small_df)
        X, y = prepare_xy(df)
        assert X.shape[0] == len(df)
        assert y is not None
        assert X.shape[1] == len(get_feature_cols())


class TestDataValidation:
    def test_valid_data_passes(self, small_df):
        validator = DataValidator()
        assert validator.validate(small_df) is True

    def test_null_values_detected(self, small_df):
        df = small_df.copy()
        df.loc[0, "amount"] = np.nan
        validator = DataValidator()
        passed = validator.validate(df)
        assert not passed
        summary = validator.summary()
        assert (summary["status"] == "FAIL").any()

    def test_out_of_range_detected(self, small_df):
        df = small_df.copy()
        df.loc[0, "hour"] = 25
        validator = DataValidator()
        passed = validator.validate(df)
        assert not passed

    def test_summary_returns_dataframe(self, small_df):
        validator = DataValidator()
        validator.validate(small_df)
        summary = validator.summary()
        assert isinstance(summary, pd.DataFrame)
        assert "status" in summary.columns
