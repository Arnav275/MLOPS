"""
Batch-score the Part-1 fact table with the champion and publish a model-driven risk mart:

    data/analytical/mart_student_risk_predictions.csv   (Tableau / Streamlit friendly)
    student_analytics.mart_student_risk_predictions     (PostgreSQL, best effort)

    python -m ml.batch_score [--no-postgres]
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone

import pandas as pd

from ml.bundle import load_bundle
from ml.config import load_params, path


def score_fact(bundle=None, params: dict | None = None) -> pd.DataFrame:
    params = params or load_params()
    bundle = bundle or load_bundle(path(params["serving"]["champion_dir"]))
    fact = pd.read_csv(path(params["data"]["fact_table"]))
    fact.columns = [c.lower() for c in fact.columns]
    fact = fact[fact["source_system"].isin(params["data"]["subjects"])].copy()
    proba = bundle.predict_proba(fact[bundle.input_features])
    split = pd.read_csv(path(params["data"]["modelling_dataset"]), usecols=["student_id", "split"]) \
        if path(params["data"]["modelling_dataset"]).exists() else pd.DataFrame(columns=["student_id", "split"])
    fact = fact.merge(split.rename(columns={"split": "data_split"}), on="student_id", how="left")
    out = pd.DataFrame({
        "student_id": fact["student_id"].values,
        "subject": fact["subject"].values,
        "school": fact["school"].values,
        "sex": fact["sex"].values,
        "g1": fact["g1"].values,
        "g2": fact["g2"].values,
        "absences": fact["absences"].values,
        "fail_probability": proba.round(4),
        "predicted_fail": (proba >= bundle.threshold).astype(int),
        "risk_band": [bundle.band(p) for p in proba],
        # rows with data_split = train/val were seen during training -> their scores are optimistic
        "data_split": fact["data_split"].fillna("unseen").values,
        "model_version": bundle.version,
        "model_scenario": bundle.scenario,
        "threshold": bundle.threshold,
        "scored_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    })
    return out.sort_values("fail_probability", ascending=False).reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-postgres", action="store_true")
    a = ap.parse_args()
    df = score_fact()
    out = path("data/analytical/mart_student_risk_predictions.csv")
    df.to_csv(out, index=False)
    print(f"wrote {out} ({len(df)} rows; {df.risk_band.value_counts().to_dict()})")
    if not a.no_postgres:
        try:
            from sqlalchemy import create_engine

            from scripts.config import PG_CONN_STRING
            df.to_sql("mart_student_risk_predictions", create_engine(PG_CONN_STRING), schema="student_analytics",
                      if_exists="replace", index=False)
            print("loaded to PostgreSQL student_analytics.mart_student_risk_predictions")
        except Exception as exc:
            print(f"PostgreSQL load skipped ({type(exc).__name__}); CSV mart is available for the dashboard / Tableau.")


if __name__ == "__main__":
    main()
