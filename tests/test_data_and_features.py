import numpy as np
import pandas as pd

from ml.features import LEAKY_COLUMNS, RAW_FEATURES, StudentFeatureEngineer, build_pipeline, raw_features_for
from ml.metrics import best_threshold, compute_metrics


def test_no_target_leakage_in_feature_list():
    assert not set(LEAKY_COLUMNS) & set(RAW_FEATURES)
    assert "fail" not in RAW_FEATURES


def test_modelling_frame_is_clean(modelling, params):
    assert modelling.student_id.is_unique
    assert len(modelling) == 1044                      # 395 math + 649 portuguese, stale copies excluded
    assert set(modelling.subject) == {"math", "portuguese"}
    assert 0.15 < modelling.fail.mean() < 0.30


def test_split_is_group_aware_and_stratified(modelling):
    assert (modelling.groupby("group_id").split.nunique() == 1).all()      # same student never straddles splits
    rates = modelling.groupby("split").fail.mean()
    assert rates.max() - rates.min() < 0.03
    sizes = modelling.split.value_counts(normalize=True)
    assert abs(sizes["test"] - 0.2) < 0.04 and abs(sizes["val"] - 0.2) < 0.04


def test_split_is_deterministic(modelling, params):
    from ml.data import assign_splits, build_modelling_frame
    again = assign_splits(build_modelling_frame(params), params)
    assert (again.values == modelling.split.values).all()


def test_early_scenario_drops_g2():
    assert "g2" not in raw_features_for("early") and "g2" in raw_features_for("mid_term")
    out = StudentFeatureEngineer("early").fit(None).transform(pd.DataFrame({"g1": [5], "absences": [3]}))
    assert "grade_delta" not in out.columns


def test_pipeline_handles_missing_optional_and_unseen_category(modelling):
    cols = raw_features_for("mid_term")
    train = modelling[modelling.split == "train"]
    from sklearn.linear_model import LogisticRegression
    pipe = build_pipeline(LogisticRegression(max_iter=500), "mid_term", scale=True).fit(train[cols], train.fail)
    row = train[cols].head(2).copy()
    row.loc[row.index[0], ["famrel", "mjob"]] = np.nan
    row.loc[row.index[1], "mjob"] = "astronaut"
    p = pipe.predict_proba(row)[:, 1]
    assert np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all()


def test_threshold_prefers_recall_for_f2():
    y = np.array([0] * 80 + [1] * 20)
    p = np.concatenate([np.linspace(0, .5, 80), np.linspace(.3, 1, 20)])
    t2 = best_threshold(y, p, "f2"); t1 = best_threshold(y, p, "f1")
    assert t2 <= t1
    assert compute_metrics(y, p, t2)["recall"] >= compute_metrics(y, p, t1)["recall"]
