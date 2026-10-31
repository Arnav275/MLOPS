"""
STAGE 4 - LOAD
Creates the warehouse schema (if not present) and loads the analytical
CSV tables produced by transform.py into PostgreSQL.
"""
from pathlib import Path

import pandas as pd
from sqlalchemy import create_engine, text

from scripts.config import PG_CONN_STRING, ANALYTICAL_DATA_DIR
from scripts.pipeline_logger import get_logger

logger = get_logger("load_to_postgres")

SCHEMA_SQL_PATH = Path(__file__).resolve().parent.parent / "sql" / "schema.sql"

TABLE_MAP = {
    "dim_student.csv": "dim_student",
    "fact_student_performance.csv": "fact_student_performance",
    "mart_subject_pass_fail.csv": "mart_subject_pass_fail",
    "mart_department_semester.csv": "mart_department_semester",
    "mart_high_risk_students.csv": "mart_high_risk_students",
}

# fact table columns are lower-case in Postgres; normalize CSV headers to match
FACT_RENAME = {"G1": "g1", "G2": "g2", "G3": "g3", "Pstatus": "pstatus",
               "Medu": "medu", "Fedu": "fedu", "Mjob": "mjob", "Fjob": "fjob"}


def create_schema(engine) -> None:
    ddl = SCHEMA_SQL_PATH.read_text()
    with engine.begin() as conn:
        for statement in ddl.split(";"):
            statement = statement.strip()
            if statement:
                conn.execute(text(statement))
    logger.info("Schema ensured (student_analytics.*)")


def load_table(engine, csv_name: str, table_name: str) -> int:
    csv_path = ANALYTICAL_DATA_DIR / csv_name
    if not csv_path.exists():
        logger.warning(
            f"Skipping {table_name}: {csv_path} not found. Run transform.py first."
        )
        return 0

    df = pd.read_csv(csv_path)

    df = df.rename(
        columns={k: v for k, v in FACT_RENAME.items() if k in df.columns}
    )
    df.columns = [c.lower() for c in df.columns]
    if table_name == "mart_subject_pass_fail":
        df = df.rename(columns={
            "fail": "fail_count",
            "pass": "pass_count",
        })
    if table_name == "mart_department_semester":
        df = df.rename(columns={
            "school": "department",
        })
    TABLE_COLUMNS = {
        "dim_student": [
            "student_id", "source_system", "school", "sex", "age",
            "address", "famsize", "pstatus", "medu", "fedu",
            "mjob", "fjob", "internet"
        ],
        "fact_student_performance": [
            "student_id", "subject", "g1", "g2", "g3", "absences",
            "attendance_pct", "avg_internal_marks",
            "assignment_completion_rate",
            "previous_semester_performance", "performance_trend",
            "result_status", "risk_score", "risk_category"
        ],
        "mart_subject_pass_fail": [
            "subject",
            "pass_count",
            "fail_count",
            "pass_rate_pct"
        ],
        "mart_department_semester": [
            "department", "subject", "avg_final_marks",
            "avg_attendance_pct", "avg_risk_score", "student_count"
        ],
        "mart_high_risk_students": [
            "student_id", "source_system", "subject", "attendance_pct",
            "avg_internal_marks", "assignment_completion_rate",
            "risk_score", "risk_category", "result_status"
        ],
    }

    if table_name not in TABLE_COLUMNS:
        raise ValueError(f"No column mapping defined for {table_name}")

    expected_columns = TABLE_COLUMNS[table_name]

    missing = [c for c in expected_columns if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing columns for {table_name}: {missing}"
        )

    # Keep only columns that actually belong to the destination table
    df = df[expected_columns]

    df.to_sql(
        table_name,
        engine,
        schema="student_analytics",
        if_exists="append",
        index=False,
        method="multi",
        chunksize=500,
    )

    logger.info(
        f"Loaded {len(df)} rows into student_analytics.{table_name}"
    )
    return len(df)


def run() -> dict:
    engine = create_engine(PG_CONN_STRING)
    create_schema(engine)
    with engine.begin() as conn:
        conn.execute(text("""
            TRUNCATE TABLE
                student_analytics.fact_student_performance,
                student_analytics.mart_subject_pass_fail,
                student_analytics.mart_department_semester,
                student_analytics.mart_high_risk_students,
                student_analytics.dim_student
            RESTART IDENTITY CASCADE
        """))
    results = {}
    # Load dim_student first (fact/marts reference it via FK)
    ordered = ["dim_student.csv", "fact_student_performance.csv",
               "mart_subject_pass_fail.csv", "mart_department_semester.csv",
               "mart_high_risk_students.csv"]
    for csv_name in ordered:
        table_name = TABLE_MAP[csv_name]
        results[table_name] = load_table(engine, csv_name, table_name)

    logger.info(f"Load complete: {results}")
    return results


if __name__ == "__main__":
    run()
