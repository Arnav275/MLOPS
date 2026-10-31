"""
Stage `prepare`: Part-1 analytical fact table -> modelling dataset with split.

* Uses only canonical subject sources (math / portuguese); the Part-1 fact table
  currently also contains stale `student-mat` / `student-por` copies of the same
  rows, which would otherwise duplicate every record and leak across splits.
* Target `fail` = G3 < pass_mark.
* Split: stratified, GROUP-aware (the same student can appear in both subject
  files; both rows go to the same split).  The UCI data has no timestamp, so a
  chronological split is impossible; G1 -> G2 -> G3 is the only temporal axis
  and is handled by the `scenario` setting (what is known when we predict).

Run:  python -m ml.data
"""
from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from ml.config import load_params, path
from ml.features import RAW_FEATURES

ID_COLS = ["student_id", "group_id", "split"]


def file_sha256(p) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def make_group_id(df: pd.DataFrame, keys: list[str]) -> pd.Series:
    key = df[keys].astype(str).agg("|".join, axis=1)
    return key.map(lambda s: hashlib.md5(s.encode()).hexdigest()[:10])


def build_modelling_frame(params: dict | None = None) -> pd.DataFrame:
    params = params or load_params()
    d = params["data"]
    fact = pd.read_csv(path(d["fact_table"]))
    fact.columns = [c.lower() for c in fact.columns]

    fact = fact[fact["source_system"].isin(d["subjects"])].copy()
    assert fact["student_id"].is_unique, "student_id must be unique after filtering to canonical sources"

    fact["fail"] = (fact["g3"] < d["pass_mark"]).astype(int)
    fact["group_id"] = make_group_id(fact, d["group_keys"])
    cols = ["student_id", "group_id"] + RAW_FEATURES + ["fail"]
    return fact[cols].reset_index(drop=True)


def assign_splits(df: pd.DataFrame, params: dict | None = None) -> pd.Series:
    """Stratified (subject x label), group-aware train/val/test assignment."""
    params = params or load_params()
    s, seed = params["split"], params["seed"]
    strat = df["subject"].astype(str) + "_" + df["fail"].astype(str)

    n_test = round(1 / s["test_size"])
    outer = StratifiedGroupKFold(n_splits=n_test, shuffle=True, random_state=seed)
    dev_idx, test_idx = next(outer.split(df, strat, df["group_id"]))

    dev = df.iloc[dev_idx]
    n_val = round(1 / (s["val_size"] / (1 - s["test_size"])))
    inner = StratifiedGroupKFold(n_splits=n_val, shuffle=True, random_state=seed)
    tr_rel, va_rel = next(inner.split(dev, strat.iloc[dev_idx], dev["group_id"]))

    split = pd.Series("train", index=df.index)
    split.iloc[test_idx] = "test"
    split.iloc[dev_idx[va_rel]] = "val"
    return split


def prepare(params: dict | None = None) -> pd.DataFrame:
    params = params or load_params()
    df = build_modelling_frame(params)
    df["split"] = assign_splits(df, params)

    out = path(params["data"]["modelling_dataset"])
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)

    profile = {
        "rows": int(len(df)),
        "fail_rate": round(float(df["fail"].mean()), 4),
        "sha256": file_sha256(out),
        "splits": {
            k: {
                "rows": int(len(g)),
                "fail_rate": round(float(g["fail"].mean()), 4),
                "students": int(g["group_id"].nunique()),
            }
            for k, g in df.groupby("split")
        },
        "group_leakage": int(
            (df.groupby("group_id")["split"].nunique() > 1).sum()
        ),
        "missing_cells": int(df[RAW_FEATURES].isna().sum().sum()),
    }
    rep = path(params.get("reports_dir", "reports")) / "data_profile.json"
    rep.parent.mkdir(parents=True, exist_ok=True)
    rep.write_text(json.dumps(profile, indent=2))
    return df


def load_modelling(params: dict | None = None) -> pd.DataFrame:
    params = params or load_params()
    p = path(params["data"]["modelling_dataset"])
    if not p.exists():
        return prepare(params)
    return pd.read_csv(p)


if __name__ == "__main__":
    out = prepare()
    print(json.dumps(json.loads((path(load_params().get("reports_dir", "reports")) / "data_profile.json").read_text()), indent=2))
