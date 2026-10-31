import numpy as np
import pandas as pd

from monitoring import drift, quality, retrain
from monitoring.performance import performance, service_health
from monitoring.report import build_report


def _cfg(params):
    return {**params["monitoring"], "min_samples": 30}


def test_psi_basics():
    p = np.array([.25, .25, .25, .25])
    assert drift.psi(p, p) == 0
    assert drift.psi(p, np.array([.7, .1, .1, .1])) > 0.25


def test_no_drift_when_sampling_from_training(bundle, modelling, params):
    train = modelling[modelling.split == "train"].sample(300, random_state=1)
    fd = drift.feature_drift(bundle.reference, train, _cfg(params))
    assert (fd.status != "ALERT").all()


def test_detects_shifted_grades(bundle, modelling, params):
    cur = modelling[modelling.split == "test"].copy()
    cur["g1"] = (cur.g1 - 4).clip(0, 20); cur["g2"] = (cur.g2 - 5).clip(0, 20)
    fd = drift.feature_drift(bundle.reference, cur, _cfg(params)).set_index("feature")
    assert fd.loc["g2", "status"] == "ALERT" and fd.loc["g1", "status"] == "ALERT"


def test_insufficient_data_is_not_judged(bundle, modelling, params):
    fd = drift.feature_drift(bundle.reference, modelling.head(5), _cfg(params))
    assert (fd.status == drift.NODATA).all()


def test_quality_flags_range_missing_and_unseen(bundle, modelling, params):
    cur = modelling[modelling.split == "test"].copy().reset_index(drop=True)
    cur.loc[:20, "absences"] = 500
    cur["g2"] = np.nan
    cur.loc[:15, "mjob"] = "pirate"
    q = quality.check_quality(bundle.reference, cur[bundle.input_features], _cfg(params))
    kinds = " ".join(f"{i['feature']}:{i['issue']}" for i in q["issues"])
    assert q["status"] == "ALERT" and "absences" in kinds and "g2" in kinds and "mjob" in kinds


def test_performance_alert_on_degradation(bundle, params):
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 200)
    lab = pd.DataFrame({"proba": rng.random(200), "actual_fail": y})       # random scores -> bad model
    assert performance(lab, bundle.metadata, _cfg(params))["status"] == "ALERT"
    assert performance(lab.head(10), bundle.metadata, _cfg(params))["status"] == drift.NODATA


def test_service_health_thresholds(params):
    n = 100
    ok = pd.DataFrame({"id": range(n), "endpoint": "/predict", "status_code": 200, "latency_ms": 20.0})
    assert service_health(ok, _cfg(params))["status"] == "OK"
    slow = ok.assign(latency_ms=900.0)
    assert service_health(slow, _cfg(params))["status"] == "ALERT"
    err = ok.assign(status_code=[500] * 5 + [200] * 95)
    assert service_health(err, _cfg(params))["status"] == "ALERT"


def test_retrain_rules(bundle, params):
    cfg = _cfg(params)
    base = {"feature_drift": [], "performance": {"status": "OK"}, "class_distribution": {}, "service": {}, "data_quality": {}, "score_drift": {}}
    assert retrain.decide(base, bundle.metadata, cfg)["action"] == "NO_ACTION"
    key = {**base, "feature_drift": [{"feature": "g2", "status": "ALERT"}]}
    assert retrain.decide(key, bundle.metadata, cfg)["action"] == "RETRAIN"
    minor = {**base, "feature_drift": [{"feature": "walc", "status": "ALERT"}]}
    assert retrain.decide(minor, bundle.metadata, cfg)["action"] == "NO_ACTION"        # one non-key feature is not enough
    perf = {**base, "performance": {"status": "ALERT", "recall_drop": .3, "pr_auc_drop": .2, "n_labels": 90}}
    assert retrain.decide(perf, bundle.metadata, cfg)["action"] == "RETRAIN"
    prior = {**base, "class_distribution": {"actual_prevalence_shift": .2, "actual_fail_rate": .42, "reference_fail_rate": .22}}
    assert retrain.decide(prior, bundle.metadata, cfg)["action"] == "RETRAIN"
    infra = {**base, "service": {"status": "ALERT"}}
    assert retrain.decide(infra, bundle.metadata, cfg)["action"] == "INVESTIGATE"
    old = {**bundle.metadata, "trained_at": "2020-01-01T00:00:00+00:00"}
    assert retrain.decide(base, old, cfg)["action"] == "RETRAIN"
    newdata = {**base, "new_labelled_rows": 10_000}
    assert retrain.decide(newdata, bundle.metadata, cfg)["action"] == "RETRAIN"


def _simulate(client, mode, n=150):
    from monitoring import simulate

    def post(ep, body):
        r = client.post(ep, json=body)
        return r.json() if r.status_code < 300 else None
    return simulate.run(post, mode, n)


def _report(client, bundle, params):
    store = client.app.state.store
    lab = store.labelled()
    return build_report(bundle, _cfg(params), store.predictions(model_version=bundle.version), lab, store.requests(), "test")


def test_end_to_end_normal_traffic_is_healthy(client, bundle, params):
    assert _simulate(client, "normal")["failed"] == 0
    rep = _report(client, bundle, params)
    assert rep["retraining"]["action"] == "NO_ACTION", rep["retraining"]
    assert rep["performance"]["status"] in ("OK", "WARN")


def test_end_to_end_covariate_drift_triggers_retrain(client, bundle, params):
    _simulate(client, "covariate_drift")
    rep = _report(client, bundle, params)
    assert rep["retraining"]["action"] == "RETRAIN"
    assert any(t["id"] == "R1_data_drift" for t in rep["retraining"]["triggers"])


def test_end_to_end_concept_drift_triggers_retrain(client, bundle, params):
    _simulate(client, "concept_drift")
    rep = _report(client, bundle, params)
    assert any(t["id"] == "R2_performance" for t in rep["retraining"]["triggers"])
    assert not any(t["id"] == "R1_data_drift" for t in rep["retraining"]["triggers"])   # inputs did not move
