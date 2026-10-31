"""
Airflow DAG: student_performance_pipeline

Orchestrates: ingest -> validate/clean -> transform -> data-quality gate -> load to Postgres
Scheduled to run daily; can also be triggered manually for the live demo.
"""
import sys
from datetime import datetime, timedelta
from pathlib import Path

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.exceptions import AirflowFailException

# Make the project's /scripts package importable from the DAG file
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from scripts import ingest, validate_clean, transform, load_to_postgres  # noqa: E402

default_args = {
    "owner": "student",
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
    "email_on_failure": False,
}


def task_ingest(**context):
    results = ingest.run()
    failed = [r for r in results if r["status"] == "FAILED"]
    if failed and len(failed) == len(results):
        raise AirflowFailException(f"All ingestion sources failed: {failed}")
    context["ti"].xcom_push(key="ingestion_results", value=results)


def task_validate_clean(**context):
    summary = validate_clean.run()
    if not summary:
        raise AirflowFailException("Validation/cleaning produced no output — check raw layer.")
    context["ti"].xcom_push(key="clean_summary", value=summary)


def task_transform(**context):
    summary = transform.run()
    if not summary:
        raise AirflowFailException("Transformation produced no analytical tables.")
    context["ti"].xcom_push(key="transform_summary", value=summary)


def task_quality_gate(**context):
    """
    Simple data-quality gate: fail the DAG run (and therefore skip the load
    step) if the cleaned fact table is empty or the rejected-record rate is
    implausibly high (>50%), so bad data never reaches PostgreSQL.
    """
    clean_summary = context["ti"].xcom_pull(key="clean_summary", task_ids="validate_and_clean")
    for source, stats in clean_summary.items():
        input_rows = stats["input_rows"]
        rejected = stats["rejected_rows"]
        if input_rows == 0:
            raise AirflowFailException(f"Source '{source}' had zero input rows.")
        reject_rate = rejected / input_rows
        if reject_rate > 0.5:
            raise AirflowFailException(
                f"Source '{source}' rejected {reject_rate:.0%} of rows — exceeds 50% threshold."
            )
    print("Data-quality gate passed.")


def task_load(**context):
    results = load_to_postgres.run()
    if sum(results.values()) == 0:
        raise AirflowFailException("No rows were loaded into PostgreSQL.")


with DAG(
    dag_id="student_performance_pipeline",
    description="Ingest -> Validate -> Transform -> Quality Gate -> Load student performance data",
    default_args=default_args,
    schedule="@daily",
    start_date=datetime(2026, 1, 1),
    catchup=False,
    tags=["student-performance", "mlops-project", "part1"],
) as dag:

    ingest_task = PythonOperator(
        task_id="ingest_data",
        python_callable=task_ingest,
    )

    validate_clean_task = PythonOperator(
        task_id="validate_and_clean",
        python_callable=task_validate_clean,
    )

    transform_task = PythonOperator(
        task_id="transform_and_aggregate",
        python_callable=task_transform,
    )

    quality_gate_task = PythonOperator(
        task_id="data_quality_gate",
        python_callable=task_quality_gate,
    )

    load_task = PythonOperator(
        task_id="load_to_postgres",
        python_callable=task_load,
    )

    ingest_task >> validate_clean_task >> quality_gate_task >> transform_task >> load_task
