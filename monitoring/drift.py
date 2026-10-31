"""
Feature, score and class-distribution drift against the training reference.

* PSI (population stability index) - main, thresholded signal: <0.10 stable, 0.10-0.25 moderate, >0.25 major.
* KS test (numeric) / chi-square (categorical) - significance, reported alongside.
PSI is biased upward on small samples, so nothing is judged below `monitoring.min_samples`.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

EPS = 1e-4
OK, WARN, ALERT, NODATA = "OK", "WARN", "ALERT", "INSUFFICIENT_DATA"


def psi(ref_p: np.ndarray, cur_p: np.ndarray) -> float:
    r, c = np.clip(np.asarray(ref_p, float), EPS, None), np.clip(np.asarray(cur_p, float), EPS, None)
    r, c = r / r.sum(), c / c.sum()
    return float(np.sum((c - r) * np.log(c / r)))


def psi_status(value: float, cfg: dict) -> str:
    return ALERT if value >= cfg["alert"] else WARN if value >= cfg["warn"] else OK


def drift_status(psi_value: float, pvalue: float | None, cfg: dict) -> str:
    """ALERT needs a large PSI AND statistical significance: PSI alone is noisy at small n
    (empirically ~13% false alarms at n=150 across 33 features without the significance guard)."""
    sig = pvalue is not None and not np.isnan(pvalue) and pvalue < cfg["ks_pvalue_alert"]
    if psi_value >= cfg["psi"]["alert"]:
        return ALERT if sig else WARN
    return WARN if psi_value >= cfg["psi"]["warn"] and sig else OK


def _proportions(keys: list[str], series: pd.Series) -> np.ndarray:
    vc = series.astype(str).value_counts(normalize=True)
    return np.array([vc.get(k, 0.0) for k in keys] + [max(0.0, 1 - sum(vc.get(k, 0.0) for k in keys))])


def numeric_drift(ref: dict, cur: pd.Series) -> dict:
    cur = pd.to_numeric(cur, errors="coerce").dropna()
    ks = stats.ks_2samp(ref["values"], cur.values)
    if ref.get("discrete"):
        keys = list(ref["proportions"].keys())
        ref_p = np.array(list(ref["proportions"].values()) + [0.0])
        value = psi(ref_p, _proportions(keys, cur.map(float)))
    else:
        edges = [-np.inf if e is None and i == 0 else np.inf if e is None else e for i, e in enumerate(ref["bin_edges"])]
        counts, _ = np.histogram(cur, bins=edges)
        value = psi(np.array(ref["bin_proportions"]), counts / max(counts.sum(), 1))
    return {"psi": round(value, 4), "ks_stat": round(float(ks.statistic), 4), "ks_pvalue": float(ks.pvalue),
            "ref_mean": round(ref["mean"], 3), "cur_mean": round(float(cur.mean()), 3)}


def categorical_drift(ref: dict, cur: pd.Series, n_ref: int) -> dict:
    cur = cur.dropna().astype(str)
    keys = list(ref["proportions"].keys())
    ref_p = np.array(list(ref["proportions"].values()) + [0.0])
    cur_p = _proportions(keys, cur)
    value = psi(ref_p, cur_p)
    table = np.vstack([ref_p * n_ref, cur_p * len(cur)])
    table = table[:, table.sum(axis=0) > 0]
    try:                                    # 2 x k homogeneity test (reference sample vs current sample)
        chi_p = float(stats.chi2_contingency(table)[1]) if table.shape[1] > 1 else 1.0
    except ValueError:
        chi_p = float("nan")
    return {"psi": round(value, 4), "chi2_pvalue": chi_p}


def feature_drift(reference: dict, current: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    rows = []
    for name, ref in reference["features"].items():
        if name not in current.columns:
            rows.append({"feature": name, "type": ref["type"], "psi": np.nan, "status": NODATA})
            continue
        s = current[name]
        if s.notna().sum() < cfg["min_samples"]:
            rows.append({"feature": name, "type": ref["type"], "psi": np.nan, "status": NODATA, "n": int(s.notna().sum())})
            continue
        res = numeric_drift(ref, s) if ref["type"] == "numeric" else categorical_drift(ref, s, reference["n_rows"])
        p = res.get("ks_pvalue", res.get("chi2_pvalue"))
        res.update({"feature": name, "type": ref["type"], "n": int(s.notna().sum()),
                    "status": drift_status(res["psi"], p, cfg)})
        rows.append(res)
    df = pd.DataFrame(rows)
    return df.sort_values("psi", ascending=False, na_position="last").reset_index(drop=True) if not df.empty else df


def score_drift(reference: dict, scores: pd.Series, cfg: dict) -> dict:
    """Prediction drift: distribution of predicted fail probability vs. validation scores."""
    if "score_values" not in reference or len(scores) < cfg["min_samples"]:
        return {"status": NODATA, "n": int(len(scores))}
    ref = np.array(reference["score_values"])
    # Fixed probability bins: quantile bins collapse because most scores sit at ~0 or ~1.
    edges = np.array([0, .05, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1.0 + 1e-9])
    rc, _ = np.histogram(ref, bins=edges)
    cc, _ = np.histogram(scores, bins=edges)
    v = psi(rc / rc.sum(), cc / max(cc.sum(), 1))
    ks = stats.ks_2samp(ref, scores)
    return {"psi": round(v, 4), "ks_pvalue": float(ks.pvalue), "n": int(len(scores)),
            "ref_mean_score": round(float(ref.mean()), 4), "cur_mean_score": round(float(np.mean(scores)), 4),
            "status": psi_status(v, cfg["psi"])}


def class_distribution(reference: dict, threshold: float, preds: pd.DataFrame, labelled: pd.DataFrame, cfg: dict) -> dict:
    """Predicted-positive rate and (when outcomes exist) the actual fail rate vs. training prevalence."""
    out = {"reference_fail_rate": round(reference["target_prevalence"], 4)}
    if len(preds) >= cfg["min_samples"]:
        out["predicted_fail_rate"] = round(float(preds["predicted_fail"].mean()), 4)
        out["predicted_fail_rate_shift"] = round(out["predicted_fail_rate"] - float(np.mean(np.array(reference["score_values"]) >= threshold)), 4) \
            if "score_values" in reference else None
        out["band_mix"] = preds["risk_band"].value_counts(normalize=True).round(4).to_dict()
    else:
        out["predicted_fail_rate"] = None
    if len(labelled) >= cfg["retrain"]["min_labels"]:
        out["actual_fail_rate"] = round(float(labelled["actual_fail"].mean()), 4)
        out["actual_prevalence_shift"] = round(out["actual_fail_rate"] - reference["target_prevalence"], 4)
        out["n_labels"] = int(len(labelled))
    else:
        out["actual_fail_rate"] = None
        out["n_labels"] = int(len(labelled))
    return out
