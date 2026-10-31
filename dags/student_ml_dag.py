"""
Airflow DAG: student_ml_lifecycle   (Part 2)

  score_batch -> monitor -> branch ─┬─ retrain -> promote_gate_export -> score_batch_after  (decision == RETRAIN)
                                    ├─ notify_investigate                                    (decision == INVESTIGATE)
                                    └─ no_action                                             (decision == NO_ACTION)

`monitor` runs monitoring.report, whose exit code encodes the decision (0/1/2).
Runs after the Part-1 DAG has refreshed data/analytical/.
"""
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path

from airflow import DAG
from airflow.operators.empty import EmptyOperator
from airflow.operators.python import BranchPythonOperator, PythonOperator

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

default_args = {"owner": "student", "retries": 1, "retry_delay": timedelta(minutes=5)}


def _run(*args: str) -> int:
    proc = subprocess.run([sys.executable, "-m", *args], cwd=ROOT, capture_output=True, text=True)
    print(proc.stdout)
    print(proc.stderr, file=sys.stderr)
    return proc.returncode


def task_monitor(**ctx):
    rc = _run("monitoring.report", "--window-hours", "168")
    if rc not in (0, 1, 2):
        raise RuntimeError("monitoring.report crashed")
    ctx["ti"].xcom_push(key="decision", value={0: "no_action", 1: "notify_investigate", 2: "retrain"}[rc])


def choose(**ctx):
    return ctx["ti"].xcom_pull(key="decision", task_ids="monitor")


def task_retrain(**ctx):
    # ml.train registers a challenger, applies the promotion gate and re-exports models/champion when promoted
    if _run("ml.data") or _run("ml.train"):
        raise RuntimeError("retraining failed")


with DAG(
    dag_id="student_ml_lifecycle",
    description="Batch scoring, monitoring and gated retraining of the student failure-risk model",
    default_args=default_args,
    schedule="@weekly",
    start_date=datetime(2026, 1, 1),
    catchup=False,
    tags=["student-performance", "mlops-project", "part2"],
) as dag:
    score = PythonOperator(task_id="score_batch", python_callable=lambda: _run("ml.batch_score", "--no-postgres"))
    monitor = PythonOperator(task_id="monitor", python_callable=task_monitor)
    branch = BranchPythonOperator(task_id="branch", python_callable=choose)
    retrain = PythonOperator(task_id="retrain", python_callable=task_retrain)
    rescore = PythonOperator(task_id="score_batch_after", python_callable=lambda: _run("ml.batch_score", "--no-postgres"))
    investigate = EmptyOperator(task_id="notify_investigate")   # wire to e-mail / Slack operator in production
    nothing = EmptyOperator(task_id="no_action")

    score >> monitor >> branch >> [retrain, investigate, nothing]
    retrain >> rescore
