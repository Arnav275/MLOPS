"""
FastAPI inference service for student failure-risk prediction.

    uvicorn api.main:app --host 0.0.0.0 --port 8000

Endpoints
  GET  /health            liveness
  GET  /ready             readiness (model loaded)
  GET  /model/info        champion metadata (version, threshold, metrics, lineage)
  POST /predict           score one student
  POST /predict/batch     score many students
  POST /feedback          attach the real outcome later -> live performance monitoring
  GET  /monitoring/summary  quick service-health snapshot
  GET  /metrics           Prometheus exposition

Env: MODEL_DIR, MONITORING_DB, API_KEY (optional; if set, required as X-API-Key header)
"""
from __future__ import annotations

import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse, Response
from prometheus_client import CONTENT_TYPE_LATEST, CollectorRegistry, Counter, Gauge, Histogram, generate_latest

from api.schemas import (BatchRequest, BatchResponse, Feedback, FeedbackResponse, Prediction, StudentRecord)
from ml.bundle import ModelBundle, load_bundle
from ml.config import load_params, path
from monitoring.store import MonitoringStore

log = logging.getLogger("student-risk-api")
UNLOGGED = {"/metrics", "/health", "/ready"}

ACTIONS = {
    "High": "Priority intervention: advisor meeting and tutoring this week.",
    "Medium": "Monitor closely: schedule a check-in and offer study support.",
    "Low": "No action needed; continue routine monitoring.",
}


def create_app(model_dir: str | Path | None = None, db_path: str | Path | None = None) -> FastAPI:
    params = load_params()
    registry = CollectorRegistry()
    m_pred = Counter("student_predictions_total", "Scored records", ["risk_band"], registry=registry)
    m_req = Counter("http_requests_total", "HTTP requests", ["endpoint", "status"], registry=registry)
    m_lat = Histogram("http_request_latency_seconds", "Request latency", ["endpoint"], registry=registry,
                      buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5))
    m_info = Gauge("model_info", "Loaded model", ["version", "family", "scenario"], registry=registry)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        mdir = Path(model_dir or os.getenv("MODEL_DIR") or path(params["serving"]["champion_dir"]))
        app.state.bundle = None
        try:
            app.state.bundle: ModelBundle = load_bundle(mdir)
            b = app.state.bundle
            m_info.labels(b.version, b.metadata["model_family"], b.scenario).set(1)
            log.info("loaded model v%s (%s, %s)", b.version, b.metadata["model_family"], b.scenario)
        except Exception as exc:                         # service stays up; /ready reports 503
            log.error("model load failed from %s: %s", mdir, exc)
        app.state.store = MonitoringStore(db_path)
        yield
        app.state.store.close()

    app = FastAPI(title="Student Failure-Risk API", version="1.0.0", lifespan=lifespan,
                  description="Predicts probability that a student finishes below the pass mark (G3 < 10).")
    expected_key = os.getenv("API_KEY")

    def auth(x_api_key: str | None = Header(default=None)):
        if expected_key and x_api_key != expected_key:
            raise HTTPException(status_code=401, detail="invalid or missing X-API-Key")

    def bundle_or_503(request: Request) -> ModelBundle:
        b = request.app.state.bundle
        if b is None:
            raise HTTPException(status_code=503, detail="model not loaded")
        return b

    # ---------------- instrumentation middleware ----------------
    @app.middleware("http")
    async def instrument(request: Request, call_next):
        t0 = time.perf_counter()
        err = None
        try:
            response = await call_next(request)
            status = response.status_code
        except Exception as exc:                         # unhandled -> 500, still measured
            err, status = repr(exc)[:300], 500
            response = JSONResponse({"detail": "internal error"}, status_code=500)
        ms = (time.perf_counter() - t0) * 1000
        ep = request.url.path
        m_req.labels(ep, str(status)).inc()
        m_lat.labels(ep).observe(ms / 1000)
        if ep not in UNLOGGED:
            try:
                request.app.state.store.log_request(ep, status, ms, err)
            except Exception:
                log.exception("request logging failed")
        response.headers["X-Process-Time-ms"] = f"{ms:.2f}"
        return response

    # ---------------- scoring ----------------
    def score(request: Request, records: list[StudentRecord]) -> list[Prediction]:
        b = bundle_or_503(request)
        cols = b.input_features
        rows = [r.model_dump() for r in records]
        if "g2" in cols and any(r["g2"] is None for r in rows):
            raise HTTPException(status_code=422, detail="g2 is required by the mid_term model")
        frame = pd.DataFrame(rows).reindex(columns=cols)
        t0 = time.perf_counter()
        proba = b.predict_proba(frame)
        per_row_ms = (time.perf_counter() - t0) * 1000 / len(rows)
        out = []
        for r, rec, p in zip(rows, frame.to_dict("records"), proba):
            p = float(p)
            band = b.band(p)
            rid = str(uuid.uuid4())
            imputed = [c for c in cols if r.get(c) is None]
            clean = {k: (None if (isinstance(v, float) and np.isnan(v)) else v) for k, v in rec.items()}
            try:
                request.app.state.store.log_prediction(rid, r["student_id"], b.version, clean, p, p >= b.threshold, band, per_row_ms)
            except Exception:
                log.exception("prediction logging failed")      # never fail a prediction because logging failed
            m_pred.labels(band).inc()
            out.append(Prediction(
                request_id=rid, student_id=r["student_id"], fail_probability=round(p, 4),
                predicted_fail=bool(p >= b.threshold), risk_band=band, recommended_action=ACTIONS[band],
                threshold=b.threshold, model_version=b.version, model_scenario=b.scenario, imputed_fields=imputed))
        return out

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.get("/ready")
    def ready(request: Request):
        b = bundle_or_503(request)
        return {"status": "ready", "model_version": b.version}

    @app.get("/model/info")
    def model_info(request: Request):
        md = dict(bundle_or_503(request).metadata)
        return md

    @app.post("/predict", response_model=Prediction, dependencies=[Depends(auth)])
    def predict(rec: StudentRecord, request: Request):
        return score(request, [rec])[0]

    @app.post("/predict/batch", response_model=BatchResponse, dependencies=[Depends(auth)])
    def predict_batch(body: BatchRequest, request: Request):
        limit = params["serving"]["max_batch_size"]
        if len(body.records) > limit:
            raise HTTPException(status_code=413, detail=f"batch too large (max {limit})")
        preds = score(request, body.records)
        return BatchResponse(count=len(preds), predictions=preds)

    @app.post("/feedback", response_model=FeedbackResponse, dependencies=[Depends(auth)])
    def feedback(fb: Feedback, request: Request):
        store: MonitoringStore = request.app.state.store
        if fb.actual_fail is None and fb.actual_final_grade is None:
            raise HTTPException(422, "provide actual_fail or actual_final_grade")
        actual = fb.actual_fail if fb.actual_fail is not None else fb.actual_final_grade < params["data"]["pass_mark"]
        rid = fb.request_id
        if rid is None and fb.student_id:
            rid = store.latest_request_for_student(fb.student_id)
        if rid is None or not store.request_exists(rid):
            raise HTTPException(404, "no matching prediction found")
        store.log_label(rid, fb.student_id, bool(actual))
        return FeedbackResponse(recorded=True, request_id=rid, actual_fail=bool(actual))

    @app.get("/monitoring/summary")
    def summary(request: Request):
        reqs = request.app.state.store.requests()
        if reqs.empty:
            return {"requests": 0}
        return {
            "requests": int(len(reqs)),
            "error_rate_5xx": float((reqs.status_code >= 500).mean()),
            "invalid_rate_4xx": float(((reqs.status_code >= 400) & (reqs.status_code < 500)).mean()),
            "latency_ms": {q: float(np.percentile(reqs.latency_ms, v)) for q, v in {"p50": 50, "p95": 95, "p99": 99}.items()},
        }

    @app.get("/metrics")
    def metrics():
        return Response(generate_latest(registry), media_type=CONTENT_TYPE_LATEST)

    return app


app = create_app()
