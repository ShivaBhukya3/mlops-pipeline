"""MLflow model registry operations."""

import logging
from typing import Any, Dict, List, Optional

import mlflow
from mlflow.tracking import MlflowClient

logger = logging.getLogger(__name__)


class ModelRegistry:
    def __init__(self, tracking_uri: str = "mlruns", model_name: str = "fraud-detector"):
        mlflow.set_tracking_uri(tracking_uri)
        self.client = MlflowClient()
        self.model_name = model_name

    def get_latest_version(self, stage: str = "None") -> Optional[str]:
        try:
            versions = self.client.get_latest_versions(self.model_name, stages=[stage])
            if versions:
                return versions[0].version
        except Exception as e:
            logger.warning("Could not get latest version: %s", e)
        return None

    def promote_to_staging(self, version: str) -> None:
        self.client.transition_model_version_stage(
            name=self.model_name, version=version, stage="Staging",
            archive_existing_versions=True,
        )
        logger.info("Promoted v%s to Staging", version)

    def promote_to_production(self, version: str) -> None:
        self.client.transition_model_version_stage(
            name=self.model_name, version=version, stage="Production",
            archive_existing_versions=True,
        )
        logger.info("Promoted v%s to Production", version)

    def archive_version(self, version: str) -> None:
        self.client.transition_model_version_stage(
            name=self.model_name, version=version, stage="Archived"
        )

    def load_production_model(self) -> Any:
        uri = f"models:/{self.model_name}/Production"
        model = mlflow.sklearn.load_model(uri)
        logger.info("Loaded Production model from %s", uri)
        return model

    def load_model_by_version(self, version: str) -> Any:
        uri = f"models:/{self.model_name}/{version}"
        return mlflow.sklearn.load_model(uri)

    def get_model_versions(self) -> List[Dict]:
        try:
            versions = self.client.search_model_versions(f"name='{self.model_name}'")
            return [
                {
                    "version": v.version,
                    "stage": v.current_stage,
                    "run_id": v.run_id,
                    "creation_timestamp": v.creation_timestamp,
                    "status": v.status,
                }
                for v in versions
            ]
        except Exception as e:
            logger.warning("Could not fetch model versions: %s", e)
            return []

    def add_model_description(self, version: str, description: str) -> None:
        self.client.update_model_version(
            name=self.model_name, version=version, description=description
        )

    def compare_versions(
        self, version_a: str, version_b: str, metric: str = "test_roc_auc"
    ) -> Dict:
        def get_metrics(version: str) -> Dict:
            versions = self.client.get_model_version(self.model_name, version)
            run = self.client.get_run(versions.run_id)
            return run.data.metrics

        metrics_a = get_metrics(version_a)
        metrics_b = get_metrics(version_b)

        val_a = metrics_a.get(metric, 0)
        val_b = metrics_b.get(metric, 0)

        return {
            "version_a": version_a,
            "version_b": version_b,
            "metric": metric,
            "value_a": val_a,
            "value_b": val_b,
            "winner": version_a if val_a >= val_b else version_b,
        }
