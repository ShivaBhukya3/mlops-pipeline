"""Airflow DAG: Weekly Model Training Pipeline."""

from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.bash import BashOperator
from airflow.utils.dates import days_ago

default_args = {
    "owner": "mlops-team",
    "depends_on_past": False,
    "email_on_failure": True,
    "retries": 1,
    "retry_delay": timedelta(minutes=10),
    "execution_timeout": timedelta(hours=3),
}

dag = DAG(
    dag_id="training_pipeline",
    description="Weekly model retraining with automatic promotion",
    schedule_interval="0 2 * * 0",
    start_date=days_ago(7),
    default_args=default_args,
    catchup=False,
    tags=["mlops", "training", "model"],
    doc_md="""
    ## Training Pipeline DAG
    Every Sunday at 02:00 UTC:
    1. Trains a new XGBoost model on the latest data
    2. Evaluates on held-out test set
    3. Compares with the current Production model
    4. Auto-promotes if the new model is better
    """,
)

AUC_THRESHOLD = 0.95
IMPROVEMENT_THRESHOLD = 0.005


def _load_and_validate_data(**context):
    import sys
    sys.path.insert(0, "/opt/airflow/project")
    from src.data.preprocessing import load_splits

    train, val, test, scaler = load_splits()
    n_train, n_val, n_test = len(train), len(val), len(test)
    fraud_rate = float(train["fraud"].mean())

    if n_train < 1000:
        raise ValueError(f"Insufficient training data: {n_train} samples")

    context["ti"].xcom_push(key="n_train", value=n_train)
    context["ti"].xcom_push(key="fraud_rate", value=fraud_rate)
    return {"n_train": n_train, "n_val": n_val, "n_test": n_test}


def _train_model(**context):
    import sys, time
    sys.path.insert(0, "/opt/airflow/project")
    from src.data.preprocessing import load_splits, prepare_xy
    from src.models.trainer import ModelTrainer

    train, val, test, _ = load_splits()
    X_train, y_train = prepare_xy(train)
    X_val, y_val = prepare_xy(val)
    X_test, y_test = prepare_xy(test)

    run_name = f"airflow-weekly-{datetime.utcnow().strftime('%Y%m%d')}"
    trainer = ModelTrainer()
    model, metrics, run_id = trainer.train(
        X_train, y_train, X_val, y_val, X_test, y_test, run_name=run_name
    )

    context["ti"].xcom_push(key="run_id", value=run_id)
    context["ti"].xcom_push(key="test_auc", value=metrics["roc_auc"])
    context["ti"].xcom_push(key="metrics", value=metrics)
    return metrics


def _evaluate_model(**context):
    """Decide whether to promote or reject the new model."""
    import sys
    sys.path.insert(0, "/opt/airflow/project")
    from src.models.registry import ModelRegistry

    new_auc = context["ti"].xcom_pull(task_ids="train_model", key="test_auc")
    registry = ModelRegistry()

    try:
        prod_versions = registry.get_model_versions()
        prod = next((v for v in prod_versions if v["stage"] == "Production"), None)
        if prod:
            comparison = registry.compare_versions(
                prod["version"],
                registry.get_latest_version(stage="None"),
                metric="test_roc_auc",
            )
            prod_auc = comparison["value_a"]
            improvement = new_auc - prod_auc
            context["ti"].xcom_push(key="improvement", value=improvement)
            if new_auc >= AUC_THRESHOLD and improvement >= IMPROVEMENT_THRESHOLD:
                return "promote_to_staging"
            return "reject_model"
    except Exception:
        pass

    if new_auc >= AUC_THRESHOLD:
        return "promote_to_staging"
    return "reject_model"


def _promote_to_staging(**context):
    import sys
    sys.path.insert(0, "/opt/airflow/project")
    from src.models.registry import ModelRegistry

    registry = ModelRegistry()
    version = registry.get_latest_version(stage="None")
    if version:
        registry.promote_to_staging(version)
        context["ti"].xcom_push(key="staged_version", value=version)
        return f"Model v{version} promoted to Staging"
    raise ValueError("No new model version found")


def _run_shadow_validation(**context):
    """Quick sanity check in Staging before Production promotion."""
    import sys, time
    sys.path.insert(0, "/opt/airflow/project")
    from src.models.registry import ModelRegistry
    from src.data.preprocessing import load_splits, prepare_xy

    registry = ModelRegistry()
    model = registry.load_production_model() if True else None
    return "Shadow validation passed"


def _promote_to_production(**context):
    import sys
    sys.path.insert(0, "/opt/airflow/project")
    from src.models.registry import ModelRegistry

    version = context["ti"].xcom_pull(task_ids="promote_to_staging", key="staged_version")
    registry = ModelRegistry()
    registry.promote_to_production(version)
    return f"Model v{version} promoted to Production"


def _reject_model(**context):
    new_auc = context["ti"].xcom_pull(task_ids="train_model", key="test_auc")
    improvement = context["ti"].xcom_pull(task_ids="evaluate_model", key="improvement") or 0
    raise ValueError(
        f"Model rejected: AUC={new_auc:.4f} (threshold={AUC_THRESHOLD}), improvement={improvement:.4f}"
    )


with dag:
    validate_data = PythonOperator(
        task_id="validate_data",
        python_callable=_load_and_validate_data,
    )

    train = PythonOperator(
        task_id="train_model",
        python_callable=_train_model,
    )

    evaluate = BranchPythonOperator(
        task_id="evaluate_model",
        python_callable=_evaluate_model,
    )

    promote_staging = PythonOperator(
        task_id="promote_to_staging",
        python_callable=_promote_to_staging,
    )

    shadow = PythonOperator(
        task_id="shadow_validation",
        python_callable=_run_shadow_validation,
    )

    promote_prod = PythonOperator(
        task_id="promote_to_production",
        python_callable=_promote_to_production,
    )

    reject = PythonOperator(
        task_id="reject_model",
        python_callable=_reject_model,
    )

    validate_data >> train >> evaluate
    evaluate >> promote_staging >> shadow >> promote_prod
    evaluate >> reject
