"""
Generate API traffic for demos / tests, including deliberately degraded scenarios.

    python -m monitoring.simulate --mode normal --n 150 --url http://localhost:8000
    python -m monitoring.simulate --mode covariate_drift --n 150
    python -m monitoring.simulate --mode concept_drift --n 150

normal           resampled held-out TEST rows with their true outcomes (healthy system)
covariate_drift  grades fall, absences rise, more math students (input distribution shifts)
concept_drift    inputs unchanged but the grade->outcome relationship breaks (labels are SYNTHETIC)
All scenarios are synthetic stress tests, clearly NOT real production behaviour.
"""
from __future__ import annotations

import argparse
from typing import Callable

import numpy as np
import pandas as pd

from ml.config import load_params, path


def make_traffic(mode: str, n: int, seed: int = 0) -> list[tuple[dict, bool]]:
    df = pd.read_csv(path(load_params()["data"]["modelling_dataset"]))
    pool = df[df.split == "test"].reset_index(drop=True)
    rng = np.random.default_rng(seed)
    s = pool.sample(n, replace=True, random_state=seed).reset_index(drop=True)
    if mode == "covariate_drift":
        math = s[s.subject == "math"]
        s = pd.concat([s, math.sample(n // 2, replace=True, random_state=seed)]).head(n).reset_index(drop=True)
        s["g1"] = (s.g1 - rng.integers(2, 5, len(s))).clip(0, 20)
        s["g2"] = (s.g2 - rng.integers(2, 6, len(s))).clip(0, 20)
        s["absences"] = (s.absences * 2.5 + rng.integers(0, 6, len(s))).astype(int).clip(0, 100)
        s["fail"] = (s.g2 < 10).astype(int) ^ (rng.random(len(s)) < 0.08)
    elif mode == "concept_drift":
        s["fail"] = np.where(rng.random(len(s)) < 0.45, 1 - s.fail, s.fail)
    out = []
    for r in s.to_dict("records"):
        label = bool(r.pop("fail"))
        for k in ("group_id", "split"):
            r.pop(k, None)
        out.append(({k: (v.item() if hasattr(v, "item") else v) for k, v in r.items()}, label))
    return out


def run(post: Callable[[str, dict], dict], mode: str, n: int, with_labels: bool = True, seed: int = 0) -> dict:
    sent = failed = 0
    for rec, label in make_traffic(mode, n, seed):
        rid = rec["student_id"] = f"SIM-{mode[:3]}-{sent:05d}"
        resp = post("/predict", rec)
        sent += 1
        if resp is None:
            failed += 1
        elif with_labels:
            post("/feedback", {"request_id": resp["request_id"], "actual_fail": label})
    return {"mode": mode, "sent": sent, "failed": failed}


if __name__ == "__main__":
    import requests

    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["normal", "covariate_drift", "concept_drift"], default="normal")
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--no-labels", action="store_true")
    a = ap.parse_args()

    def _post(ep, body):
        r = requests.post(a.url + ep, json=body, timeout=10)
        return r.json() if r.ok else None

    print(run(_post, a.mode, a.n, not a.no_labels))
