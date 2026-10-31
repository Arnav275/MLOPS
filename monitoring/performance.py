"""Live model performance from delayed ground truth + service health (latency / failures)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ml.metrics import compute_metrics
from monitoring.drift import ALERT, NODATA, OK, WARN


def performance(labelled: pd.DataFrame, metadata: dict, cfg: dict) -> dict:
    """labelled: columns proba, actual_fail.  Compared with the held-out TEST metrics at training time."""
    base = metadata["metrics"]["test"]
    n = len(labelled)
    out = {"n_labels": n, "baseline": {k: round(base[k], 4) for k in ["recall", "precision", "pr_auc", "brier"]}}
    if n < cfg["retrain"]["min_labels"] or labelled["actual_fail"].nunique() < 2:
        out["status"] = NODATA
        return out
    m = compute_metrics(labelled["actual_fail"], labelled["proba"], float(metadata["threshold"]))
    out["current"] = {k: round(m[k], 4) for k in ["recall", "precision", "pr_auc", "brier", "accuracy", "f2"]}
    out["confusion"] = {k: m[k] for k in ["tn", "fp", "fn", "tp"]}
    out["recall_drop"] = round(base["recall"] - m["recall"], 4)
    out["pr_auc_drop"] = round(base["pr_auc"] - m["pr_auc"], 4)
    r = cfg["retrain"]
    if out["recall_drop"] > r["max_recall_drop"] or out["pr_auc_drop"] > r["max_pr_auc_drop"]:
        out["status"] = ALERT
    elif out["recall_drop"] > r["max_recall_drop"] / 2 or out["pr_auc_drop"] > r["max_pr_auc_drop"] / 2:
        out["status"] = WARN
    else:
        out["status"] = OK
    return out


def service_health(requests: pd.DataFrame, cfg: dict) -> dict:
    if requests.empty:
        return {"status": NODATA, "requests": 0}
    r = requests[requests.endpoint.isin(["/predict", "/predict/batch", "/feedback"])]
    if r.empty:
        return {"status": NODATA, "requests": 0}
    lat = r.latency_ms.values
    e5 = float((r.status_code >= 500).mean())
    e4 = float(((r.status_code >= 400) & (r.status_code < 500)).mean())
    out = {
        "requests": int(len(r)),
        "error_rate_5xx": round(e5, 4),
        "rejected_rate_4xx": round(e4, 4),
        "latency_ms": {k: round(float(np.percentile(lat, q)), 2) for k, q in {"p50": 50, "p95": 95, "p99": 99}.items()},
        "by_endpoint": r.groupby("endpoint").agg(n=("id", "count"), p95_ms=("latency_ms", lambda x: float(np.percentile(x, 95)))).round(2).to_dict("index"),
    }
    lt, et = cfg["latency_ms"], cfg["error_rate"]
    status = OK
    if len(r) >= cfg["min_samples"]:
        if out["latency_ms"]["p95"] >= lt["p95_alert"] or e5 >= et["alert"]:
            status = ALERT
        elif out["latency_ms"]["p95"] >= lt["p95_warn"] or e5 >= et["warn"]:
            status = WARN
        if e4 >= 0.25 and status == OK:
            status = WARN                      # a quarter of calls rejected -> upstream is sending bad data
    else:
        status = NODATA
    out["status"] = status
    return out
