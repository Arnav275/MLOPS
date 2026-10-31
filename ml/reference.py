"""Training-time reference statistics used by the drift / data-quality monitors."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ml.features import CATEGORICAL_RAW, NUMERIC_RAW, raw_features_for

DISCRETE_MAX_UNIQUE = 12


def build_reference(train: pd.DataFrame, scenario: str, val_scores=None) -> dict:
    feats: dict = {}
    for c in raw_features_for(scenario):
        s = train[c]
        miss = float(s.isna().mean())
        if c in NUMERIC_RAW:
            v = pd.to_numeric(s, errors="coerce").dropna()
            discrete = int(v.nunique()) <= DISCRETE_MAX_UNIQUE
            entry = {
                "type": "numeric", "discrete": discrete, "missing_rate": miss,
                "mean": float(v.mean()), "std": float(v.std(ddof=0)),
                "min": float(v.min()), "max": float(v.max()),
                "values": [float(x) for x in v.tolist()],        # reference sample for KS
            }
            if discrete:
                vc = v.value_counts(normalize=True)
                entry["proportions"] = {str(float(k)): float(p) for k, p in vc.items()}
            else:
                edges = np.unique(np.quantile(v, np.linspace(0, 1, 11)))
                edges[0], edges[-1] = -np.inf, np.inf
                counts, _ = np.histogram(v, bins=edges)
                entry["bin_edges"] = [None if np.isinf(e) else float(e) for e in edges]
                entry["bin_proportions"] = (counts / counts.sum()).tolist()
            feats[c] = entry
        else:
            vc = s.dropna().astype(str).value_counts(normalize=True)
            feats[c] = {
                "type": "categorical", "missing_rate": miss,
                "proportions": {k: float(p) for k, p in vc.items()},
            }
    ref = {
        "scenario": scenario,
        "n_rows": int(len(train)),
        "target_prevalence": float(train["fail"].mean()),
        "features": feats,
    }
    if val_scores is not None:
        ref["score_quantiles"] = [float(q) for q in np.quantile(val_scores, np.linspace(0, 1, 11))]
        ref["score_values"] = [round(float(x), 6) for x in val_scores]
    return ref
