"""
STAGE 2 - VALIDATION & CLEANING
Reads the most recent raw extract(s), standardizes a student identifier,
removes duplicates and invalid marks, imputes missing attendance/assignment
fields, and writes:
    - data/staging/<source>_staged.csv   (standardized, still un-aggregated)
    - data/cleaned/<source>_cleaned.csv  (fully validated analytical-ready)
    - data/logs/rejected_records_<ts>.csv (rows dropped, with reason)
"""
import glob
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.config import RAW_DATA_DIR, STAGING_DATA_DIR, CLEANED_DATA_DIR, LOG_DIR, MAX_MARK
from scripts.pipeline_logger import get_logger

logger = get_logger("validate_clean")


def _latest_raw_files() -> dict[str, Path]:
    """Returns the latest timestamped raw extract for each source."""
    latest = {}
    for path in sorted(RAW_DATA_DIR.glob("*.csv")):
        stem = path.stem
        parts = stem.rsplit("_", 1)
        # Only process timestamped files created by Stage 1 ingestion.
        # Example: math_20260902T125437Z.csv
        if len(parts) != 2:
            continue
        source_name, timestamp = parts
        if not (len(timestamp) == 16 and timestamp.endswith("Z")):
            continue
        latest[source_name] = path
    return latest


def standardize_student_id(df: pd.DataFrame, source_name: str) -> pd.DataFrame:
    """
    The raw UCI dataset has no student ID column. We synthesize a stable,
    source-prefixed identifier so records can be joined/deduped consistently
    across sources (as required: 'standardize student identifiers').
    """
    df = df.reset_index(drop=True)
    df.insert(0, "student_id", [f"{source_name.upper()[:3]}-{i+1:05d}" for i in df.index])
    df.insert(1, "source_system", source_name)
    return df


def validate_and_clean(df: pd.DataFrame, source_name: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    rejects = []

    # 1. Standardize identifiers
    df = standardize_student_id(df, source_name)

    # 2. Remove exact duplicate rows (excluding the just-generated student_id)
    dup_mask = df.drop(columns=["student_id"]).duplicated(keep="first")
    if dup_mask.any():
        rej = df[dup_mask].copy()
        rej["reject_reason"] = "duplicate_record"
        rejects.append(rej)
        df = df[~dup_mask]

    # 3. Validate marks columns (G1, G2, G3 expected 0-MAX_MARK)
    grade_cols = [c for c in ["G1", "G2", "G3"] if c in df.columns]
    for col in grade_cols:
        invalid_mask = ~df[col].between(0, MAX_MARK) | df[col].isna()
        if invalid_mask.any():
            rej = df[invalid_mask].copy()
            rej["reject_reason"] = f"invalid_mark_{col}"
            rejects.append(rej)
            df = df[~invalid_mask]

    # 4. Handle missing attendance (absences) -> impute with source median
    if "absences" in df.columns:
        missing_att = df["absences"].isna()
        if missing_att.any():
            median_val = df["absences"].median()
            df.loc[missing_att, "absences"] = median_val
            logger.info(
                f"[{source_name}] Imputed {missing_att.sum()} missing 'absences' "
                f"values with median={median_val}"
            )
        df["absences"] = df["absences"].clip(lower=0)

    # 5. Handle missing assignment/study fields (studytime is UCI's closest proxy)
    if "studytime" in df.columns:
        df["studytime"] = df["studytime"].fillna(df["studytime"].median())

    # 6. Drop rows missing every grade (cannot be used for analysis at all)
    if grade_cols:
        all_grades_missing = df[grade_cols].isna().all(axis=1)
        if all_grades_missing.any():
            rej = df[all_grades_missing].copy()
            rej["reject_reason"] = "all_grades_missing"
            rejects.append(rej)
            df = df[~all_grades_missing]

    reject_df = pd.concat(rejects, ignore_index=True) if rejects else pd.DataFrame()
    return df.reset_index(drop=True), reject_df


def run() -> dict:
    latest_files = _latest_raw_files()
    if not latest_files:
        logger.warning("No raw files found. Run ingest.py first.")
        return {}

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    all_rejects = []
    summary = {}

    for source_name, path in latest_files.items():
        logger.info(f"Validating/cleaning source '{source_name}' from {path.name}")
        raw_df = pd.read_csv(path, sep=";")

        staged_path = STAGING_DATA_DIR / f"{source_name}_staged.csv"
        raw_df_std = standardize_student_id(raw_df.copy(), source_name)
        raw_df_std.to_csv(staged_path, index=False)

        cleaned_df, rejected_df = validate_and_clean(raw_df.copy(), source_name)

        cleaned_path = CLEANED_DATA_DIR / f"{source_name}_cleaned.csv"
        cleaned_df.to_csv(cleaned_path, index=False)

        if not rejected_df.empty:
            rejected_df["source_name"] = source_name
            all_rejects.append(rejected_df)

        summary[source_name] = {
            "input_rows": len(raw_df),
            "cleaned_rows": len(cleaned_df),
            "rejected_rows": len(rejected_df),
        }
        logger.info(f"'{source_name}' summary: {summary[source_name]}")

    if all_rejects:
        reject_log_path = LOG_DIR / f"rejected_records_{ts}.csv"
        pd.concat(all_rejects, ignore_index=True).to_csv(reject_log_path, index=False)
        logger.info(f"Wrote rejected-record log -> {reject_log_path}")

    return summary


if __name__ == "__main__":
    run()
