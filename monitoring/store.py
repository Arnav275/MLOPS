"""
SQLite store shared by the API (writer) and the monitors / dashboard (readers).

Tables
  predictions : one row per scored record  (raw features, probability, decision, model version)
  requests    : one row per HTTP call      (endpoint, status, latency)  -> latency / failure monitoring
  labels      : delayed ground truth       (actual outcome joined to a prediction)  -> performance monitoring
SQLite in WAL mode is plenty for this workload; swap for PostgreSQL (already in the
Part-1 stack) by replacing this module only - readers use the helper functions below.
"""
from __future__ import annotations

import json
import os
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from ml.config import load_params, path

SCHEMA = """
CREATE TABLE IF NOT EXISTS predictions (
    request_id TEXT PRIMARY KEY, ts TEXT NOT NULL, student_id TEXT, model_version TEXT,
    features TEXT NOT NULL, proba REAL NOT NULL, predicted_fail INTEGER NOT NULL,
    risk_band TEXT NOT NULL, latency_ms REAL
);
CREATE TABLE IF NOT EXISTS requests (
    id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT NOT NULL, endpoint TEXT NOT NULL,
    status_code INTEGER NOT NULL, latency_ms REAL NOT NULL, error TEXT
);
CREATE TABLE IF NOT EXISTS labels (
    id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT NOT NULL, request_id TEXT NOT NULL,
    student_id TEXT, actual_fail INTEGER NOT NULL, source TEXT
);
CREATE INDEX IF NOT EXISTS ix_pred_ts ON predictions(ts);
CREATE INDEX IF NOT EXISTS ix_req_ts ON requests(ts);
CREATE INDEX IF NOT EXISTS ix_lab_req ON labels(request_id);
"""


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def default_db_path() -> Path:
    return path(os.getenv("MONITORING_DB", load_params()["monitoring"]["db_path"]))


class MonitoringStore:
    def __init__(self, db_path: str | Path | None = None):
        self.db_path = Path(db_path) if db_path else default_db_path()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False, isolation_level=None)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.executescript(SCHEMA)

    # ---------------- writers (never raise into the request path) ----------------
    def _exec(self, sql: str, args: tuple = ()):
        with self._lock:
            return self._conn.execute(sql, args)

    def log_prediction(self, request_id, student_id, model_version, features: dict, proba, pred, band, latency_ms):
        self._exec(
            "INSERT OR REPLACE INTO predictions VALUES (?,?,?,?,?,?,?,?,?)",
            (request_id, now(), student_id, model_version, json.dumps(features, default=str),
             float(proba), int(pred), band, float(latency_ms)),
        )

    def log_request(self, endpoint: str, status_code: int, latency_ms: float, error: str | None = None):
        self._exec("INSERT INTO requests(ts,endpoint,status_code,latency_ms,error) VALUES (?,?,?,?,?)",
                   (now(), endpoint, int(status_code), float(latency_ms), error))

    def log_label(self, request_id, student_id, actual_fail: bool, source="api"):
        self._exec("INSERT INTO labels(ts,request_id,student_id,actual_fail,source) VALUES (?,?,?,?,?)",
                   (now(), request_id, student_id, int(actual_fail), source))

    def latest_request_for_student(self, student_id: str) -> str | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT request_id FROM predictions WHERE student_id=? ORDER BY ts DESC LIMIT 1", (student_id,)
            ).fetchone()
        return row[0] if row else None

    def request_exists(self, request_id: str) -> bool:
        with self._lock:
            return self._conn.execute("SELECT 1 FROM predictions WHERE request_id=?", (request_id,)).fetchone() is not None

    # ---------------- readers ----------------
    def _df(self, sql: str, args: tuple = ()) -> pd.DataFrame:
        with self._lock:
            return pd.read_sql_query(sql, self._conn, params=args)

    def predictions(self, since: str | None = None, model_version: str | None = None) -> pd.DataFrame:
        sql, args = "SELECT * FROM predictions WHERE 1=1", []
        if since:
            sql += " AND ts>=?"; args.append(since)
        if model_version:
            sql += " AND model_version=?"; args.append(str(model_version))
        df = self._df(sql + " ORDER BY ts", tuple(args))
        if not df.empty:
            feats = pd.json_normalize(df["features"].map(json.loads))
            feats.index = df.index
            df = pd.concat([df.drop(columns=["features"]), feats], axis=1)
        return df

    def requests(self, since: str | None = None) -> pd.DataFrame:
        if since:
            return self._df("SELECT * FROM requests WHERE ts>=? ORDER BY ts", (since,))
        return self._df("SELECT * FROM requests ORDER BY ts")

    def labelled(self) -> pd.DataFrame:
        """Predictions joined with the most recent ground-truth label."""
        return self._df(
            """SELECT p.request_id, p.ts, p.student_id, p.model_version, p.proba, p.predicted_fail,
                      p.risk_band, l.actual_fail
               FROM predictions p JOIN (SELECT request_id, actual_fail, MAX(id) FROM labels GROUP BY request_id) l
               ON p.request_id = l.request_id ORDER BY p.ts"""
        )

    def close(self):
        self._conn.close()
