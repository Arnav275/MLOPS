"""
STAGE 3 - TRANSFORMATION
Builds semester/subject/student-level aggregates and engineered features on
top of the cleaned layer, then writes analytical data-mart tables (CSV here;
loaded into PostgreSQL by load_to_postgres.py):

    fact_student_performance.csv   - one row per student per subject/source
    dim_student.csv                - student-level dimension
    mart_subject_pass_fail.csv     - subject-wise pass/fail distribution
    mart_department_semester.csv   - department/semester performance summary
    mart_high_risk_students.csv    - high-risk student list
"""
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.config import CLEANED_DATA_DIR, ANALYTICAL_DATA_DIR, MAX_ABSENCES_OBSERVED, MAX_MARK
from scripts.pipeline_logger import get_logger

logger = get_logger("transform")

PASS_THRESHOLD = 10  # UCI grades are 0-20; >=10 is a conventional pass mark


def engineer_features(df: pd.DataFrame, subject_label: str) -> pd.DataFrame:
    df = df.copy()
    df["subject"] = subject_label

    # Attendance percentage: fewer absences -> higher attendance
    if "absences" in df.columns:
        df["attendance_pct"] = (
            100 * (1 - (df["absences"].clip(upper=MAX_ABSENCES_OBSERVED) / MAX_ABSENCES_OBSERVED))
        ).round(2)
    else:
        df["attendance_pct"] = np.nan

    # Average internal marks (G1 = first period, G2 = second period)
    internal_cols = [c for c in ["G1", "G2"] if c in df.columns]
    df["avg_internal_marks"] = df[internal_cols].mean(axis=1).round(2) if internal_cols else np.nan

    # Assignment completion rate proxy: normalized studytime (1-4 scale in UCI)
    if "studytime" in df.columns:
        df["assignment_completion_rate"] = (df["studytime"] / 4 * 100).round(2)
    else:
        df["assignment_completion_rate"] = np.nan

    # Previous-semester performance = G1 relative to G3 (final)
    if {"G1", "G3"}.issubset(df.columns):
        df["previous_semester_performance"] = df["G1"]
        df["performance_trend"] = np.where(
            df["G3"] > df["G1"], "improving",
            np.where(df["G3"] < df["G1"], "declining", "stable")
        )

    # Final pass/fail flag and risk score
    if "G3" in df.columns:
        df["result_status"] = np.where(df["G3"] >= PASS_THRESHOLD, "PASS", "FAIL")
    else:
        logger.warning(
            f"Missing G3 column for {subject_label}. "
            f"Available columns: {list(df.columns)}"
        )

    df["risk_score"] = (
        (100 - df["attendance_pct"].fillna(0)) * 0.4
        + (100 - (df.get("avg_internal_marks", 0).fillna(0) / MAX_MARK * 100)) * 0.4
        + (100 - df["assignment_completion_rate"].fillna(0)) * 0.2
    ).round(2)

    df["risk_category"] = pd.cut(
        df["risk_score"],
        bins=[-1, 33, 66, 101],
        labels=["Low", "Medium", "High"],
    )

    return df


def build_fact_table() -> pd.DataFrame:
    frames = []
    for path in sorted(CLEANED_DATA_DIR.glob("*_cleaned.csv")):
        source_name = path.stem.replace("_cleaned", "")
        df = pd.read_csv(path)
        frames.append(engineer_features(df, subject_label=source_name))

    if not frames:
        logger.warning("No cleaned files found. Run validate_clean.py first.")
        return pd.DataFrame()

    fact = pd.concat(frames, ignore_index=True)
    fact_path = ANALYTICAL_DATA_DIR / "fact_student_performance.csv"
    fact.to_csv(fact_path, index=False)
    logger.info(f"Wrote fact table -> {fact_path} ({len(fact)} rows)")
    return fact


def build_dim_student(fact: pd.DataFrame) -> pd.DataFrame:
    demo_cols = [c for c in [
        "student_id", "source_system", "school", "sex", "age", "address",
        "famsize", "Pstatus", "Medu", "Fedu", "Mjob", "Fjob", "internet"
    ] if c in fact.columns]
    dim = fact[demo_cols].drop_duplicates(subset="student_id").reset_index(drop=True)
    out_path = ANALYTICAL_DATA_DIR / "dim_student.csv"
    dim.to_csv(out_path, index=False)
    logger.info(f"Wrote dim_student -> {out_path} ({len(dim)} rows)")
    return dim


def build_subject_pass_fail_mart(fact: pd.DataFrame) -> pd.DataFrame:
    mart = (
        fact.groupby(["subject", "result_status"])
        .size()
        .reset_index(name="student_count")
        .pivot(index="subject", columns="result_status", values="student_count")
        .fillna(0)
        .reset_index()
    )
    for col in ["PASS", "FAIL"]:
        if col not in mart.columns:
            mart[col] = 0
    mart["pass_rate_pct"] = (mart["PASS"] / (mart["PASS"] + mart["FAIL"]) * 100).round(2)
    out_path = ANALYTICAL_DATA_DIR / "mart_subject_pass_fail.csv"
    mart.to_csv(out_path, index=False)
    logger.info(f"Wrote subject pass/fail mart -> {out_path}")
    return mart


def build_department_semester_mart(fact: pd.DataFrame) -> pd.DataFrame:
    group_col = "school" if "school" in fact.columns else "subject"
    mart = (
        fact.groupby([group_col, "subject"])
        .agg(
            avg_final_marks=("G3", "mean"),
            avg_attendance_pct=("attendance_pct", "mean"),
            avg_risk_score=("risk_score", "mean"),
            student_count=("student_id", "count"),
        )
        .round(2)
        .reset_index()
    )
    out_path = ANALYTICAL_DATA_DIR / "mart_department_semester.csv"
    mart.to_csv(out_path, index=False)
    logger.info(f"Wrote department/semester mart -> {out_path}")
    return mart


def build_high_risk_mart(fact: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in [
        "student_id", "source_system", "subject", "attendance_pct",
        "avg_internal_marks", "assignment_completion_rate", "risk_score",
        "risk_category", "result_status"
    ] if c in fact.columns]
    high_risk = fact[fact["risk_category"] == "High"][cols].sort_values(
        "risk_score", ascending=False
    ).reset_index(drop=True)
    out_path = ANALYTICAL_DATA_DIR / "mart_high_risk_students.csv"
    high_risk.to_csv(out_path, index=False)
    logger.info(f"Wrote high-risk student mart -> {out_path} ({len(high_risk)} rows)")
    return high_risk


def run() -> dict:
    fact = build_fact_table()
    if fact.empty:
        return {}

    dim_student = build_dim_student(fact)
    subject_mart = build_subject_pass_fail_mart(fact)
    dept_mart = build_department_semester_mart(fact)
    high_risk_mart = build_high_risk_mart(fact)

    return {
        "fact_rows": len(fact),
        "dim_student_rows": len(dim_student),
        "subject_mart_rows": len(subject_mart),
        "department_mart_rows": len(dept_mart),
        "high_risk_rows": len(high_risk_mart),
    }


if __name__ == "__main__":
    print(run())
