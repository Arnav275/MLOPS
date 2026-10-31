import pytest
from fastapi.testclient import TestClient

from api.main import create_app


def test_health_ready_info(client, bundle):
    assert client.get("/health").json() == {"status": "ok"}
    assert client.get("/ready").json()["model_version"] == bundle.version
    info = client.get("/model/info").json()
    assert info["scenario"] == bundle.scenario and "metrics" in info


def test_predict_contract(client, sample_record, bundle):
    r = client.post("/predict", json=sample_record)
    assert r.status_code == 200, r.text
    body = r.json()
    assert 0 <= body["fail_probability"] <= 1
    assert body["risk_band"] in ("Low", "Medium", "High")
    assert body["predicted_fail"] == (body["fail_probability"] >= bundle.threshold)
    assert (body["risk_band"] != "Low") == body["predicted_fail"]          # band and decision never disagree
    assert body["imputed_fields"] == []


def test_minimal_record_is_imputed(client):
    r = client.post("/predict", json={"subject": "math", "g1": 6, "g2": 4, "absences": 10})
    assert r.status_code == 200 and len(r.json()["imputed_fields"]) > 10


def test_obviously_failing_vs_strong_student(client):
    weak = client.post("/predict", json={"subject": "math", "g1": 4, "g2": 3, "absences": 8, "failures": 2}).json()
    strong = client.post("/predict", json={"subject": "portuguese", "g1": 16, "g2": 17, "absences": 0}).json()
    assert weak["fail_probability"] > 0.8 > 0.2 > strong["fail_probability"]


@pytest.mark.parametrize("bad", [
    {"subject": "math", "g1": 25, "g2": 5, "absences": 1},          # out of range
    {"subject": "art", "g1": 5, "g2": 5, "absences": 1},            # unknown subject
    {"subject": "math", "g1": 5, "g2": 5},                          # missing required
    {"subject": "math", "g1": 5, "g2": 5, "absences": 1, "hacker": 1},   # unknown field
])
def test_validation_rejects_bad_input(client, bad):
    assert client.post("/predict", json=bad).status_code == 422


def test_g2_required_for_mid_term(client):
    r = client.post("/predict", json={"subject": "math", "g1": 5, "absences": 1})
    assert r.status_code == 422 and "g2" in r.text


def test_batch_and_limit(client, sample_record):
    r = client.post("/predict/batch", json={"records": [sample_record] * 5})
    assert r.status_code == 200 and r.json()["count"] == 5
    assert client.post("/predict/batch", json={"records": []}).status_code == 422


def test_feedback_roundtrip_and_logging(client, sample_record):
    pred = client.post("/predict", json=sample_record).json()
    fb = client.post("/feedback", json={"request_id": pred["request_id"], "actual_final_grade": 7.5})
    assert fb.status_code == 200 and fb.json()["actual_fail"] is True
    by_student = client.post("/feedback", json={"student_id": sample_record["student_id"], "actual_fail": False})
    assert by_student.json()["request_id"] == pred["request_id"]
    assert client.post("/feedback", json={"request_id": "nope", "actual_fail": True}).status_code == 404
    store = client.app.state.store
    assert len(store.predictions()) == 1 and len(store.labelled()) == 1
    assert (store.requests().endpoint == "/predict").sum() == 1


def test_prometheus_metrics(client, sample_record):
    client.post("/predict", json=sample_record)
    text = client.get("/metrics").text
    assert "student_predictions_total" in text and "http_request_latency_seconds" in text


def test_service_survives_missing_model(tmp_path):
    with TestClient(create_app(model_dir=tmp_path / "missing", db_path=tmp_path / "m.db")) as c:
        assert c.get("/health").status_code == 200
        assert c.get("/ready").status_code == 503
        assert c.post("/predict", json={"subject": "math", "g1": 5, "g2": 5, "absences": 1}).status_code == 503


def test_api_key_enforced(monkeypatch, tmp_path, sample_record):
    monkeypatch.setenv("API_KEY", "s3cret")
    with TestClient(create_app(db_path=tmp_path / "k.db")) as c:
        assert c.post("/predict", json=sample_record).status_code == 401
        assert c.post("/predict", json=sample_record, headers={"X-API-Key": "s3cret"}).status_code == 200
        assert c.get("/health").status_code == 200


def test_latency_budget(client, sample_record):
    import time
    client.post("/predict", json=sample_record)                    # warm-up
    t = []
    for _ in range(30):
        t0 = time.perf_counter(); client.post("/predict", json=sample_record); t.append((time.perf_counter() - t0) * 1000)
    assert sorted(t)[int(.95 * len(t)) - 1] < 200                  # p95 < 200 ms in-process
