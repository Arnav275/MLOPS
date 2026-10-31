"""Slow smoke test: full train -> register -> promote -> export in an isolated temp workspace."""
import copy

import pytest

from ml.bundle import load_bundle


@pytest.mark.slow
def test_train_register_promote_export(tmp_path, params):
    from ml import registry, train
    from ml.data import prepare

    p = copy.deepcopy(params)
    p["data"]["modelling_dataset"] = str(tmp_path / "model.csv")
    p["reports_dir"] = str(tmp_path / "reports")
    p["serving"]["champion_dir"] = str(tmp_path / "champion")
    p["mlflow"]["tracking_uri"] = f"sqlite:///{tmp_path}/mlflow.db"
    p["mlflow"]["experiment"] = "pytest"
    p["model"]["candidates"] = {
        "logistic_regression": {"grid": {"C": [1.0]}},
        "random_forest": {"grid": {"n_estimators": [50], "max_depth": [4], "min_samples_leaf": [5]}},
        "xgboost": {"grid": {"n_estimators": [50], "max_depth": [2], "learning_rate": [0.1]}},
    }
    prepare(p)
    s1 = train.run(params=p)
    assert s1["promotion"]["promote"] and s1["champion"]["registry_version"] == 1
    assert s1["test"]["recall"] >= p["model"]["promotion"]["min_recall"]
    assert s1["test"]["pr_auc"] > s1["baselines_test"]["prior_only"]["pr_auc"] + 0.3      # beats trivial baseline by a lot
    b1 = load_bundle(tmp_path / "champion")
    assert b1.version == "1"

    s2 = train.run(params=p)                         # identical data/seed -> challenger tied -> promoted (within epsilon)
    assert s2["champion"]["registry_version"] == 2 and s2["promotion"]["promote"]

    # a clearly worse challenger must be blocked by the gate
    client = registry.configure(p)
    worse = {**s2["test"], "pr_auc": s2["test"]["pr_auc"] - 0.2}
    assert not registry.promotion_decision(client, p, 3, worse)["promote"]
    low_recall = {**s2["test"], "recall": 0.1}
    assert "min_recall" in registry.promotion_decision(client, p, 3, low_recall)["reason"]

    # rollback: alias back to v1 and re-export
    client.set_registered_model_alias(p["mlflow"]["registered_model"], "champion", "1")
    assert registry.export_champion(client, p).version == "1"
