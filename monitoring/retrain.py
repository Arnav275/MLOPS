"""
Retraining decision from a monitoring report.  Criteria (all thresholds in params.yaml):

 RETRAIN if ANY of
   R1 data drift      : >= max_alert_features features with PSI alert, or any key feature with PSI alert
   R2 performance     : recall drop > max_recall_drop OR PR-AUC drop > max_pr_auc_drop  (>= min_labels outcomes)
   R3 class prior     : actual fail rate moved > max_prevalence_shift from training  (>= min_labels outcomes)
   R4 model age       : older than max_model_age_days (one academic term)
   R5 new data volume : new labelled records >= new_data_fraction * training rows
 INVESTIGATE (no retrain, engineering action) if service latency/error or input-quality status is ALERT,
   or prediction-score drift is ALERT without a data-drift/performance cause.
"""
from __future__ import annotations

from datetime import datetime, timezone

from monitoring.drift import ALERT


def decide(report: dict, metadata: dict, cfg: dict) -> dict:
    r = cfg["retrain"]
    triggers, investigate = [], []

    drift = report.get("feature_drift", [])
    alerts = [d["feature"] for d in drift if d.get("status") == ALERT]
    key_alerts = [f for f in alerts if f in r["key_features"]]
    if len(alerts) >= r["max_alert_features"] or key_alerts:
        triggers.append({"id": "R1_data_drift", "detail": f"PSI alert on {alerts}" + (f" (key: {key_alerts})" if key_alerts else "")})

    perf = report.get("performance", {})
    if perf.get("status") == ALERT:
        triggers.append({"id": "R2_performance", "detail": f"recall drop {perf.get('recall_drop')}, PR-AUC drop {perf.get('pr_auc_drop')} "
                                                           f"on {perf.get('n_labels')} outcomes"})

    cd = report.get("class_distribution", {})
    shift = cd.get("actual_prevalence_shift")
    if shift is not None and abs(shift) > r["max_prevalence_shift"]:
        triggers.append({"id": "R3_class_prior", "detail": f"actual fail rate {cd['actual_fail_rate']:.1%} vs {cd['reference_fail_rate']:.1%} in training"})

    trained = datetime.fromisoformat(str(metadata["trained_at"]).replace("Z", "+00:00"))
    age = (datetime.now(timezone.utc) - trained).days
    if age > r["max_model_age_days"]:
        triggers.append({"id": "R4_model_age", "detail": f"{age} days old (limit {r['max_model_age_days']})"})

    n_new = report.get("new_labelled_rows", 0)
    if n_new >= r["new_data_fraction"] * metadata["n_train"]:
        triggers.append({"id": "R5_new_data", "detail": f"{n_new} new labelled rows >= {r['new_data_fraction']:.0%} of {metadata['n_train']} training rows"})

    if report.get("service", {}).get("status") == ALERT:
        investigate.append("service latency/error rate in ALERT")
    if report.get("data_quality", {}).get("status") == ALERT:
        investigate.append("input data-quality ALERT: " + "; ".join(f"{i['feature']}: {i['issue']}" for i in report["data_quality"]["issues"][:4]))
    if report.get("score_drift", {}).get("status") == ALERT and not triggers:
        investigate.append("prediction-score drift without feature drift / label evidence")

    action = "RETRAIN" if triggers else "INVESTIGATE" if investigate else "NO_ACTION"
    return {"action": action, "triggers": triggers, "investigate": investigate, "model_age_days": age}
