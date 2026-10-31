"""Thin client used by the Streamlit pages: prefers the REST API, falls back to the local bundle."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ml.bundle import load_bundle  # noqa: E402
from ml.config import load_params, path  # noqa: E402

API_URL = os.getenv("API_URL", "http://localhost:8000").rstrip("/")
HEADERS = {"X-API-Key": os.environ["API_KEY"]} if os.getenv("API_KEY") else {}


def api_available() -> bool:
    try:
        return requests.get(f"{API_URL}/ready", timeout=1.5).ok
    except Exception:
        return False


def local_bundle():
    return load_bundle(path(load_params()["serving"]["champion_dir"]))


def model_info() -> tuple[dict, str]:
    if api_available():
        return requests.get(f"{API_URL}/model/info", timeout=5).json(), f"REST API ({API_URL})"
    return local_bundle().metadata, "local model bundle (API not reachable)"


def predict(record: dict) -> tuple[dict, str]:
    """Returns (prediction dict, source label)."""
    if api_available():
        r = requests.post(f"{API_URL}/predict", json=record, headers=HEADERS, timeout=10)
        if r.ok:
            return r.json(), "REST API"
        raise RuntimeError(f"API rejected the record: {r.status_code} {r.text}")
    b = local_bundle()
    frame = pd.DataFrame([record]).reindex(columns=b.input_features)
    p = float(b.predict_proba(frame)[0])
    return {"fail_probability": round(p, 4), "predicted_fail": p >= b.threshold, "risk_band": b.band(p),
            "threshold": b.threshold, "model_version": b.version, "model_scenario": b.scenario,
            "imputed_fields": [c for c in b.input_features if record.get(c) is None],
            "recommended_action": "(local fallback - not logged for monitoring)"}, "local bundle (not logged)"
