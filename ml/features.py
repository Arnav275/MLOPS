"""
Feature definitions and the sklearn pipeline used for BOTH training and serving.

Keeping feature engineering inside the pipeline object means the API cannot
drift from training (no train/serve skew): the serialized artefact takes the
*raw* student record and does everything itself.

Leakage policy
--------------
Target = (G3 < pass_mark).  The following Part-1 columns are derived from G3
(or from the target-adjacent risk formula) and are therefore NEVER features:
  g3, result_status, performance_trend, risk_score, risk_category
and the Part-1 convenience columns attendance_pct, avg_internal_marks,
assignment_completion_rate, previous_semester_performance are redundant
re-encodings of absences / g1 / g2 / studytime, so we rebuild what we need here.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

LEAKY_COLUMNS = ["g3", "result_status", "performance_trend", "risk_score", "risk_category"]

NUMERIC_RAW = [
    "age", "medu", "fedu", "traveltime", "studytime", "failures", "famrel",
    "freetime", "goout", "dalc", "walc", "health", "absences", "g1", "g2",
]
CATEGORICAL_RAW = [
    "subject", "school", "sex", "address", "famsize", "pstatus", "mjob", "fjob",
    "reason", "guardian", "schoolsup", "famsup", "paid", "activities", "nursery",
    "higher", "internet", "romantic",
]
RAW_FEATURES = NUMERIC_RAW + CATEGORICAL_RAW

# Plausible ranges (UCI documentation) - used by the API schema and data-quality monitor.
NUMERIC_RANGES = {
    "age": (10, 30), "medu": (0, 4), "fedu": (0, 4), "traveltime": (1, 4), "studytime": (1, 4),
    "failures": (0, 4), "famrel": (1, 5), "freetime": (1, 5), "goout": (1, 5), "dalc": (1, 5),
    "walc": (1, 5), "health": (1, 5), "absences": (0, 100), "g1": (0, 20), "g2": (0, 20),
}
CATEGORICAL_VALUES = {
    "subject": ["math", "portuguese"], "school": ["GP", "MS"], "sex": ["F", "M"],
    "address": ["U", "R"], "famsize": ["LE3", "GT3"], "pstatus": ["T", "A"],
    "mjob": ["teacher", "health", "services", "at_home", "other"],
    "fjob": ["teacher", "health", "services", "at_home", "other"],
    "reason": ["home", "reputation", "course", "other"],
    "guardian": ["mother", "father", "other"],
    "schoolsup": ["yes", "no"], "famsup": ["yes", "no"], "paid": ["yes", "no"],
    "activities": ["yes", "no"], "nursery": ["yes", "no"], "higher": ["yes", "no"],
    "internet": ["yes", "no"], "romantic": ["yes", "no"],
}

# Features whose values are only known after the corresponding period ends.
SCENARIO_EXCLUDED = {"early": ["g2"], "mid_term": []}


def raw_features_for(scenario: str) -> list[str]:
    excl = set(SCENARIO_EXCLUDED[scenario])
    return [c for c in RAW_FEATURES if c not in excl]


class StudentFeatureEngineer(BaseEstimator, TransformerMixin):
    """Stateless feature construction from raw student fields."""

    def __init__(self, scenario: str = "mid_term"):
        self.scenario = scenario

    def fit(self, X, y=None):
        self.feature_names_in_ = list(raw_features_for(self.scenario))
        return self

    def transform(self, X):
        X = pd.DataFrame(X).copy()
        for c in self.feature_names_in_:                   # tolerate missing optional columns
            if c not in X.columns:
                X[c] = np.nan
        X = X[self.feature_names_in_].copy()
        for c in NUMERIC_RAW:
            if c in X.columns:
                X[c] = pd.to_numeric(X[c], errors="coerce")

        X["parent_edu_mean"] = X[["medu", "fedu"]].mean(axis=1)
        X["alcohol_total"] = X[["dalc", "walc"]].sum(axis=1, min_count=2)
        X["absences_log1p"] = np.log1p(X["absences"].clip(lower=0))
        X["g1_below_pass"] = (X["g1"] < 10).astype(float).where(X["g1"].notna())
        if self.scenario == "mid_term":
            X["avg_prior_grade"] = X[["g1", "g2"]].mean(axis=1)
            X["grade_delta"] = X["g2"] - X["g1"]
            X["g2_below_pass"] = (X["g2"] < 10).astype(float).where(X["g2"].notna())
            X["g2_is_zero"] = (X["g2"] == 0).astype(float).where(X["g2"].notna())
        else:
            X["g1_is_zero"] = (X["g1"] == 0).astype(float).where(X["g1"].notna())
        return X

    def get_feature_names_out(self, input_features=None):
        return np.array(self.transform(pd.DataFrame(columns=self.feature_names_in_)).columns)


def engineered_numeric(scenario: str) -> list[str]:
    base = [c for c in NUMERIC_RAW if c in raw_features_for(scenario)]
    extra = ["parent_edu_mean", "alcohol_total", "absences_log1p", "g1_below_pass"]
    extra += (["avg_prior_grade", "grade_delta", "g2_below_pass", "g2_is_zero"]
              if scenario == "mid_term" else ["g1_is_zero"])
    return base + extra


def build_pipeline(estimator, scenario: str = "mid_term", scale: bool = False) -> Pipeline:
    """raw record -> engineer -> impute/scale/one-hot -> estimator."""
    num_steps = [("impute", SimpleImputer(strategy="median"))]
    if scale:
        num_steps.append(("scale", StandardScaler()))
    pre = ColumnTransformer(
        [
            ("num", Pipeline(num_steps), engineered_numeric(scenario)),
            ("cat", Pipeline([
                ("impute", SimpleImputer(strategy="most_frequent")),
                ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]), CATEGORICAL_RAW),
        ],
        verbose_feature_names_out=False,
    )
    pre.set_output(transform="pandas")
    return Pipeline([
        ("engineer", StudentFeatureEngineer(scenario)),
        ("prep", pre),
        ("model", estimator),
    ])
