"""
Build a full monitoring report and retraining decision.

    python -m monitoring.report                              # API prediction log (SQLite)
    python -m monitoring.report --window-hours 168           # last 7 days only
    python -m monitoring.report --current-data new_term.csv  # batch mode: modelling-format CSV (+ optional `fail` label)

Writes monitoring/reports/latest.json (+ timestamped copy).  Exit code: 0 NO_ACTION, 1 INVESTIGATE, 2 RETRAIN
(so Airflow / cron / CI can branch on it).
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

from ml.bundle import ModelBundle, load_bundle
from ml.config import load_params, path
from monitoring import drift, performance, quality, retrain
from monitoring.store import MonitoringStore


def build_report(bundle: ModelBundle, cfg: dict, preds: pd.DataFrame, labelled: pd.DataFrame,
                 requests: pd.DataFrame, source: str) -> dict:
    ref, feats = bundle.reference, bundle.input_features
    fd = drift.feature_drift(ref, preds, cfg) if len(preds) else pd.DataFrame()
    rep = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": source,
        "model": {"version": bundle.version, "family": bundle.metadata["model_family"], "scenario": bundle.scenario,
                  "trained_at": bundle.metadata["trained_at"], "threshold": bundle.threshold},
        "n_predictions": int(len(preds)),
        "feature_drift": fd.replace({np.nan: None}).to_dict("records") if len(fd) else [],
        "score_drift": drift.score_drift(ref, preds["proba"], cfg) if len(preds) else {"status": drift.NODATA},
        "class_distribution": drift.class_distribution(ref, bundle.threshold, preds, labelled, cfg) if len(preds) else {},
        "data_quality": quality.check_quality(ref, preds[[c for c in feats if c in preds.columns]], cfg) if len(preds) else {"status": drift.NODATA, "issues": []},
        "performance": performance.performance(labelled, bundle.metadata, cfg),
        "service": performance.service_health(requests, cfg),
        "new_labelled_rows": int(len(labelled)),
    }
    statuses = [rep["score_drift"].get("status"), rep["data_quality"].get("status"), rep["performance"].get("status"),
                rep["service"].get("status")] + [d["status"] for d in rep["feature_drift"]]
    rep["overall_status"] = drift.ALERT if drift.ALERT in statuses else drift.WARN if drift.WARN in statuses else \
        drift.OK if drift.OK in statuses else drift.NODATA
    rep["retraining"] = retrain.decide(rep, bundle.metadata, cfg)
    return rep


def _from_store(bundle, cfg, window_hours):
    store = MonitoringStore()
    since = (datetime.now(timezone.utc) - timedelta(hours=window_hours)).isoformat() if window_hours else None
    preds = store.predictions(since=since, model_version=bundle.version)
    lab = store.labelled()
    lab = lab[lab.model_version.astype(str) == bundle.version] if len(lab) else lab
    return preds, lab, store.requests(since=since), "api_prediction_log"


def _from_csv(bundle, csv_path):
    df = pd.read_csv(csv_path)
    df.columns = [c.lower() for c in df.columns]
    proba = bundle.predict_proba(df[bundle.input_features])
    preds = df[bundle.input_features].copy()
    preds["proba"], preds["predicted_fail"] = proba, (proba >= bundle.threshold).astype(int)
    preds["risk_band"] = [bundle.band(p) for p in proba]
    lab = pd.DataFrame()
    if "fail" in df.columns:
        lab = pd.DataFrame({"proba": proba, "actual_fail": df["fail"].values})
    return preds, lab, pd.DataFrame(), f"batch_csv:{csv_path}"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir")
    ap.add_argument("--window-hours", type=float)
    ap.add_argument("--current-data")
    ap.add_argument("--no-save", action="store_true")
    a = ap.parse_args(argv)
    params = load_params()
    cfg = params["monitoring"]
    bundle = load_bundle(a.model_dir or path(params["serving"]["champion_dir"]))
    preds, lab, reqs, src = _from_csv(bundle, a.current_data) if a.current_data else _from_store(bundle, cfg, a.window_hours)
    rep = build_report(bundle, cfg, preds, lab, reqs, src)

    if not a.no_save:
        out = path(cfg["reports_dir"]); out.mkdir(parents=True, exist_ok=True)
        text = json.dumps(rep, indent=2, default=float)
        (out / "latest.json").write_text(text)
        (out / f"report_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}.json").write_text(text)
    d = rep["retraining"]
    print(f"[monitoring] source={src} n={rep['n_predictions']} overall={rep['overall_status']} -> {d['action']}")
    for t in d["triggers"]:
        print(f"  TRIGGER {t['id']}: {t['detail']}")
    for i in d["investigate"]:
        print(f"  INVESTIGATE: {i}")
    return {"NO_ACTION": 0, "INVESTIGATE": 1, "RETRAIN": 2}[d["action"]]


if __name__ == "__main__":
    sys.exit(main())
