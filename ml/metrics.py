"""Classification metrics shared by training, promotion and monitoring."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score, average_precision_score, brier_score_loss, confusion_matrix,
    f1_score, fbeta_score, precision_score, recall_score, roc_auc_score,
)


def best_threshold(y_true, proba, strategy: str = "f2") -> float:
    """Pick the decision threshold on VALIDATION data (never on test)."""
    beta = {"f1": 1.0, "f2": 2.0}[strategy]
    grid = np.unique(np.concatenate([np.linspace(0.05, 0.95, 91), np.round(proba, 4)]))
    best_t, best_s = 0.5, -1.0
    for t in grid:
        s = fbeta_score(y_true, (proba >= t).astype(int), beta=beta, zero_division=0)
        if s > best_s + 1e-12 or (abs(s - best_s) <= 1e-12 and t > best_t):
            best_t, best_s = float(t), float(s)
    return round(best_t, 4)


def compute_metrics(y_true, proba, threshold: float = 0.5) -> dict:
    y_true = np.asarray(y_true)
    proba = np.asarray(proba)
    pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    one_class = len(np.unique(y_true)) < 2
    return {
        "n": int(len(y_true)),
        "positives": int(y_true.sum()),
        "threshold": float(threshold),
        "pr_auc": float("nan") if one_class else float(average_precision_score(y_true, proba)),
        "roc_auc": float("nan") if one_class else float(roc_auc_score(y_true, proba)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "f2": float(fbeta_score(y_true, pred, beta=2, zero_division=0)),
        "accuracy": float(accuracy_score(y_true, pred)),
        "brier": float(brier_score_loss(y_true, proba)),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def bootstrap_ci(y_true, proba, threshold, n_boot: int = 1000, seed: int = 0) -> dict:
    """95% percentile CI - essential here because the test set has only ~45 positives."""
    rng = np.random.default_rng(seed)
    y_true, proba = np.asarray(y_true), np.asarray(proba)
    out = {"pr_auc": [], "roc_auc": [], "recall": [], "precision": []}
    for _ in range(n_boot):
        idx = rng.integers(0, len(y_true), len(y_true))
        if len(np.unique(y_true[idx])) < 2:
            continue
        m = compute_metrics(y_true[idx], proba[idx], threshold)
        for k in out:
            out[k].append(m[k])
    return {k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] for k, v in out.items()}
