"""Airflow DAG: Data Ingestion Pipeline."""

from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from airflow.utils.dates import days_ago

default_args = {
    "owner": "mlops-team",
    "depends_on_past": False,
    "email_on_failure": True,
    "email_on_retry": False,
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
    "execution_timeout": timedelta(hours=1),
}

dag = DAG(
    dag_id="data_ingestion",
    description="Ingest and validate transaction data every 6 hours",
    schedule_interval="0 */6 * * *",
    start_date=days_ago(1),
    default_args=default_args,
    catchup=False,
    tags=["mlops", "data", "ingestion"],
    doc_md="""
    ## Data Ingestion DAG
    Runs every 6 hours to pull fresh transaction data, validate it,
    and save to the processed data store.
    """,
)


def _ingest(**context):
    import sys
    sys.path.insert(0, "/opt/airflow/project")
    from src.data.ingestion import ingest_data
    path = ingest_data(force=True)
    context["ti"].xcom_push(key="raw_data_path", value=path)
    return path


def _validate(**context):
    import sys
    sys.path.insert(0, "/opt/airflow/project")
    import pandas as pd
    from src.data.validation import DataValidator

    path = context["ti"].xcom_pull(task_ids="ingest_data", key="raw_data_path")
    df = pd.read_parquet(path)
    validator = DataValidator()
    passed = validator.validate(df)

    summary = validator.summary()
    fails = summary[summary["status"] == "FAIL"]
    if not fails.empty:
        raise ValueError(f"Validation failed:\n{fails.to_string()}")

    context["ti"].xcom_push(key="validation_passed", value=passed)
    return passed


def _preprocess(**context):
    import sys
    sys.path.insert(0, "/opt/airflow/project")
    from src.data.ingestion import load_raw_data
    from src.data.preprocessing import split_data, save_splits, prepare_xy, preprocess

    df = load_raw_data("data/raw/transactions.parquet")
    train, val, test = split_data(df)
    X_train, _ = prepare_xy(train)
    _, scaler = preprocess(X_train, fit_scaler=True)
    save_splits(train, val, test, scaler=scaler)

    context["ti"].xcom_push(key="n_train", value=len(train))
    context["ti"].xcom_push(key="n_test", value=len(test))
    return {"n_train": len(train), "n_val": len(val), "n_test": len(test)}


def _update_reference(**context):
    """Copy current test set as new reference for drift detection."""
    import sys, shutil
    sys.path.insert(0, "/opt/airflow/project")
    shutil.copy("data/processed/test.parquet", "data/reference/reference.parquet")
    return "Reference data updated"


with dag:
    ingest = PythonOperator(
        task_id="ingest_data",
        python_callable=_ingest,
    )

    validate = PythonOperator(
        task_id="validate_data",
        python_callable=_validate,
    )

    preprocess = PythonOperator(
        task_id="preprocess_data",
        python_callable=_preprocess,
    )

    update_reference = PythonOperator(
        task_id="update_reference_data",
        python_callable=_update_reference,
    )

    notify = BashOperator(
        task_id="notify_success",
        bash_command='echo "Data ingestion complete at $(date)" >> /opt/airflow/logs/ingestion.log',
    )

    ingest >> validate >> preprocess >> update_reference >> notify
