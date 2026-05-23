"""Airflow DAG: Hourly Monitoring & Drift Detection."""

from datetime import timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.bash import BashOperator
from airflow.utils.dates import days_ago

default_args = {
    "owner": "mlops-team",
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=2),
    "execution_timeout": timedelta(minutes=30),
}

dag = DAG(
    dag_id="monitoring_pipeline",
    description="Hourly drift detection and model performance monitoring",
    schedule_interval="0 * * * *",
    start_date=days_ago(1),
    default_args=default_args,
    catchup=False,
    tags=["mlops", "monitoring", "drift"],
)

DRIFT_SCORE_THRESHOLD = 0.3


def _collect_recent_predictions(**context):
    """Simulate collecting recent predictions from production."""
    import sys
    sys.path.insert(0, "/opt/airflow/project")
    import numpy as np, pandas as pd
    from src.data.ingestion import generate_fraud_dataset

    df = generate_fraud_dataset(n_samples=1000, random_state=int(context["ts_nodash"][-6:] or 42))
    df.to_parquet("/tmp/recent_predictions.parquet", index=False)
    context["ti"].xcom_push(key="n_recent", value=len(df))
    return len(df)


def _run_drift_detection(**context):
    import sys
    sys.path.insert(0, "/opt/airflow/project")
    import pandas as pd
    from pathlib import Path
    from src.monitoring.drift_detector import DriftDetector
    from src.data.preprocessing import get_feature_cols

    reference_path = "data/reference/reference.parquet"
    if not Path(reference_path).exists():
        reference_path = "data/processed/test.parquet"
    if not Path(reference_path).exists():
        return "no_drift"

    reference = pd.read_parquet(reference_path)
    current = pd.read_parquet("/tmp/recent_predictions.parquet")

    features = [f for f in get_feature_cols()[:11] if f in reference.columns and f in current.columns]

    detector = DriftDetector()
    detector.fit_reference(reference[features])
    report = detector.detect(current[features], features=features)

    context["ti"].xcom_push(key="drift_score", value=report.drift_score)
    context["ti"].xcom_push(key="drifted_features", value=report.drifted_features)
    context["ti"].xcom_push(key="overall_drift", value=report.overall_drift)

    if report.drift_score > DRIFT_SCORE_THRESHOLD:
        return "trigger_retraining"
    elif report.overall_drift:
        return "send_drift_alert"
    return "log_metrics"


def _send_drift_alert(**context):
    drift_score = context["ti"].xcom_pull(task_ids="run_drift_detection", key="drift_score")
    drifted = context["ti"].xcom_pull(task_ids="run_drift_detection", key="drifted_features")
    print(f"ALERT: Drift detected! Score={drift_score:.2f}, Features={drifted}")
    return "Alert sent"


def _trigger_retraining(**context):
    drift_score = context["ti"].xcom_pull(task_ids="run_drift_detection", key="drift_score")
    print(f"CRITICAL: High drift (score={drift_score:.2f}), triggering retraining DAG...")
    from airflow.api.common.trigger_dag import trigger_dag
    try:
        trigger_dag("training_pipeline", run_id=f"drift-triggered-{context['ts_nodash']}")
    except Exception as e:
        print(f"Could not trigger retraining: {e}")
    return "Retraining triggered"


def _log_metrics(**context):
    drift_score = context["ti"].xcom_pull(task_ids="run_drift_detection", key="drift_score") or 0
    n_recent = context["ti"].xcom_pull(task_ids="collect_predictions", key="n_recent") or 0
    print(f"Monitoring OK: drift_score={drift_score:.4f}, n_predictions={n_recent}")
    return "Metrics logged"


with dag:
    collect = PythonOperator(
        task_id="collect_predictions",
        python_callable=_collect_recent_predictions,
    )

    drift = BranchPythonOperator(
        task_id="run_drift_detection",
        python_callable=_run_drift_detection,
    )

    alert = PythonOperator(
        task_id="send_drift_alert",
        python_callable=_send_drift_alert,
    )

    retrain = PythonOperator(
        task_id="trigger_retraining",
        python_callable=_trigger_retraining,
    )

    log = PythonOperator(
        task_id="log_metrics",
        python_callable=_log_metrics,
    )

    collect >> drift
    drift >> alert
    drift >> retrain
    drift >> log
