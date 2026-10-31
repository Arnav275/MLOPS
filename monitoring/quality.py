"""Input data-quality checks (schema, missingness, ranges, unseen categories) vs. training."""
from __future__ import annotations

import pandas as pd

from ml.features import CATEGORICAL_VALUES, NUMERIC_RANGES
from monitoring.drift import ALERT, NODATA, OK, WARN


def check_quality(reference: dict, current: pd.DataFrame, cfg: dict) -> dict:
    q = cfg["quality"]
    n = len(current)
    if n < cfg["min_samples"]:
        return {"status": NODATA, "n": n, "issues": []}
    issues, rows = [], []
    for name, ref in reference["features"].items():
        if name not in current.columns:
            issues.append({"feature": name, "issue": "column_missing", "severity": ALERT})
            continue
        s = current[name]
        miss = float(s.isna().mean())
        row = {"feature": name, "missing_rate": round(miss, 4), "ref_missing_rate": round(ref["missing_rate"], 4)}
        if miss - ref["missing_rate"] > q["max_missing_rate_increase"]:
            issues.append({"feature": name, "issue": f"missing rate {miss:.1%} vs {ref['missing_rate']:.1%} in training",
                           "severity": ALERT if name in ("g1", "g2", "absences") else WARN})
        if ref["type"] == "numeric" and name in NUMERIC_RANGES:
            lo, hi = NUMERIC_RANGES[name]
            v = pd.to_numeric(s, errors="coerce").dropna()
            oor = float(((v < lo) | (v > hi)).mean()) if len(v) else 0.0
            row["out_of_range_rate"] = round(oor, 4)
            if oor > q["max_out_of_range_rate"]:
                issues.append({"feature": name, "issue": f"{oor:.1%} outside [{lo},{hi}]", "severity": ALERT})
        if ref["type"] == "categorical":
            known = set(CATEGORICAL_VALUES.get(name, [])) | set(ref["proportions"])
            v = s.dropna().astype(str)
            unseen = float((~v.isin(known)).mean()) if len(v) else 0.0
            row["unseen_category_rate"] = round(unseen, 4)
            if unseen > q["max_unseen_category_rate"]:
                issues.append({"feature": name, "issue": f"{unseen:.1%} unseen categories", "severity": WARN})
        rows.append(row)
    status = ALERT if any(i["severity"] == ALERT for i in issues) else WARN if issues else OK
    return {"status": status, "n": n, "issues": issues, "per_feature": rows}
