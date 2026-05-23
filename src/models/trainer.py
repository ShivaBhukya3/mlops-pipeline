"""ML model training with MLflow experiment tracking."""

import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import mlflow
import mlflow.sklearn
import mlflow.xgboost
import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score,
    precision_score, recall_score, confusion_matrix,
    classification_report,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

logger = logging.getLogger(__name__)


def load_config(path: str = "config/config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def get_model(model_type: str, params: Dict[str, Any]):
    if model_type == "xgboost" and HAS_XGB:
        clean = {k: v for k, v in params.items() if k not in ("use_label_encoder", "eval_metric")}
        return xgb.XGBClassifier(**clean, random_state=42, n_jobs=-1)
    elif model_type == "random_forest":
        return RandomForestClassifier(
            n_estimators=params.get("n_estimators", 200),
            max_depth=params.get("max_depth", 10),
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )
    elif model_type == "logistic":
        return LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)
    else:
        logger.warning("Unknown model type %s, using RandomForest", model_type)
        return RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42)


def compute_metrics(y_true, y_pred, y_prob) -> Dict[str, float]:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return {
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "avg_precision": float(average_precision_score(y_true, y_prob)),
        "f1": float(f1_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
    }


class ModelTrainer:
    def __init__(self, config_path: str = "config/config.yaml"):
        self.cfg = load_config(config_path)
        self.training_cfg = self.cfg["training"]
        self.mlflow_cfg = self.cfg["mlflow"]
        self.model = None
        self.run_id = None

    def setup_mlflow(self) -> None:
        mlflow.set_tracking_uri(self.mlflow_cfg.get("tracking_uri", "mlruns"))
        mlflow.set_experiment(self.mlflow_cfg["experiment_name"])

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        run_name: Optional[str] = None,
    ) -> Tuple[Any, Dict[str, float], str]:
        self.setup_mlflow()

        run_name = run_name or f"{self.training_cfg['model_type']}-{int(time.time())}"

        with mlflow.start_run(run_name=run_name) as run:
            self.run_id = run.info.run_id
            logger.info("MLflow run: %s", self.run_id)

            mlflow.log_params({
                "model_type": self.training_cfg["model_type"],
                "n_train": len(X_train),
                "n_val": len(X_val),
                "n_test": len(X_test),
                "n_features": X_train.shape[1],
                "fraud_rate_train": float(y_train.mean()),
                **self.training_cfg["hyperparameters"],
            })
            mlflow.set_tags({
                "project": self.cfg["project"]["name"],
                "version": self.cfg["project"]["version"],
                "framework": "xgboost" if HAS_XGB else "sklearn",
            })

            self.model = get_model(
                self.training_cfg["model_type"],
                self.training_cfg["hyperparameters"],
            )

            t0 = time.time()
            self.model.fit(X_train, y_train)
            train_time = time.time() - t0
            mlflow.log_metric("train_time_seconds", train_time)
            logger.info("Training completed in %.1fs", train_time)

            threshold = self.cfg["serving"]["prediction_threshold"]
            for split_name, X_s, y_s in [("val", X_val, y_val), ("test", X_test, y_test)]:
                y_prob = self.model.predict_proba(X_s)[:, 1]
                y_pred = (y_prob >= threshold).astype(int)
                metrics = compute_metrics(y_s, y_pred, y_prob)
                mlflow.log_metrics({f"{split_name}_{k}": v for k, v in metrics.items()})
                logger.info("%s metrics: AUC=%.4f F1=%.4f Precision=%.4f Recall=%.4f",
                            split_name, metrics["roc_auc"], metrics["f1"],
                            metrics["precision"], metrics["recall"])

            test_prob = self.model.predict_proba(X_test)[:, 1]
            test_pred = (test_prob >= threshold).astype(int)
            final_metrics = compute_metrics(y_test, test_pred, test_prob)

            feature_importance = self._get_feature_importance(X_train.columns.tolist())
            if feature_importance:
                import tempfile
                fi_df = pd.DataFrame(list(feature_importance.items()), columns=["feature", "importance"])
                fi_df = fi_df.sort_values("importance", ascending=False)
                fi_path = os.path.join(tempfile.gettempdir(), "feature_importance.csv")
                fi_df.to_csv(fi_path, index=False)
                mlflow.log_artifact(fi_path)

            # Save model via joblib to avoid mlflow.pyfunc / langchain_core import
            # issues on Python 3.14. Falls back to mlflow.sklearn.log_model on
            # a proper tracking server where the registry is available.
            import tempfile, joblib as jl
            tracking_uri = self.mlflow_cfg.get("tracking_uri", "mlruns")
            use_registry = tracking_uri.startswith(("http://", "https://", "sqlite://", "postgresql://", "mysql://"))
            if use_registry:
                model_info = mlflow.sklearn.log_model(
                    self.model,
                    artifact_path="model",
                    registered_model_name=self.mlflow_cfg["model_name"],
                )
            else:
                model_dir = os.path.join(tempfile.gettempdir(), "mlops_model")
                os.makedirs(model_dir, exist_ok=True)
                model_path = os.path.join(model_dir, "model.pkl")
                jl.dump(self.model, model_path)
                mlflow.log_artifact(model_path, artifact_path="model")
                # Also persist to a stable local path for the serving API
                os.makedirs("models", exist_ok=True)
                jl.dump(self.model, "models/fraud_detector.pkl")
                mlflow.log_param("local_model_path", "models/fraud_detector.pkl")

            logger.info(
                "Model registered: AUC=%.4f, AP=%.4f, F1=%.4f",
                final_metrics["roc_auc"],
                final_metrics["avg_precision"],
                final_metrics["f1"],
            )

        return self.model, final_metrics, self.run_id

    def _get_feature_importance(self, feature_names) -> Dict[str, float]:
        try:
            if hasattr(self.model, "feature_importances_"):
                return dict(zip(feature_names, self.model.feature_importances_.tolist()))
            if hasattr(self.model, "coef_"):
                return dict(zip(feature_names, abs(self.model.coef_[0]).tolist()))
        except Exception:
            pass
        return {}

    def cross_validate(self, X: pd.DataFrame, y: pd.Series, cv: int = 5) -> Dict[str, float]:
        model = get_model(self.training_cfg["model_type"], self.training_cfg["hyperparameters"])
        skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
        scores = cross_val_score(model, X, y, cv=skf, scoring="roc_auc", n_jobs=-1)
        return {"cv_auc_mean": float(scores.mean()), "cv_auc_std": float(scores.std())}
