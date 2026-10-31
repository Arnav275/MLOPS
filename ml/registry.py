"""
MLflow Model Registry helpers: register, champion/challenger aliases, gated
promotion, and export of the champion to the serving bundle.

Lifecycle (aliases, not deprecated stages):
    training run -> new model version -> alias `challenger`
        -> promotion gate (re-scored on the current test split vs. champion)
        -> alias `champion` -> export to models/champion/ -> docker build
Rollback = point alias `champion` back at the previous version and re-export.

CLI:
    python -m ml.registry list
    python -m ml.registry export                  # registry champion -> models/champion
    python -m ml.registry rollback --version 3
"""
from __future__ import annotations

import argparse
import json
import tempfile

import mlflow
import pandas as pd
from mlflow import MlflowClient

from ml.bundle import ModelBundle, save_bundle
from ml.config import load_params, path
from ml.data import load_modelling
from ml.metrics import compute_metrics


def configure(params: dict | None = None) -> MlflowClient:
    params = params or load_params()
    uri = params["mlflow"]["tracking_uri"]
    if uri.startswith("sqlite:///") and not uri.startswith("sqlite:////"):
        db = path(uri.replace("sqlite:///", ""))
        db.parent.mkdir(parents=True, exist_ok=True)
        uri = f"sqlite:///{db}"
    mlflow.set_tracking_uri(uri)
    client = MlflowClient()
    # Keep artefacts next to the tracking DB (instead of ./mlruns) so one folder holds the whole store.
    if client.get_experiment_by_name(params["mlflow"]["experiment"]) is None:
        art = path("mlflow/artifacts")
        art.mkdir(parents=True, exist_ok=True)
        client.create_experiment(params["mlflow"]["experiment"], artifact_location=art.as_uri())
    return client


def champion_version(client: MlflowClient, params: dict):
    name, alias = params["mlflow"]["registered_model"], params["mlflow"]["champion_alias"]
    try:
        return client.get_model_version_by_alias(name, alias)
    except Exception:
        return None


def _load_pipeline(version: str, params: dict):
    return mlflow.sklearn.load_model(f"models:/{params['mlflow']['registered_model']}/{version}")


def rescore_on_test(pipeline, threshold: float, scenario: str, params: dict) -> dict:
    df = load_modelling(params)
    test = df[df["split"] == "test"]
    proba = pipeline.predict_proba(test.drop(columns=["fail", "split", "group_id", "student_id"]))[:, 1]
    return compute_metrics(test["fail"], proba, threshold)


def promotion_decision(client, params, challenger_version, challenger_metrics: dict) -> dict:
    """Gate: promote if no champion exists, or the challenger is not worse by more than epsilon
    on PR-AUC (both re-scored on the CURRENT test split) AND meets the minimum recall."""
    promo = params["model"]["promotion"]
    champ = champion_version(client, params)
    if challenger_metrics["recall"] < promo["min_recall"]:
        return {"promote": False, "reason": f"recall {challenger_metrics['recall']:.3f} < min_recall {promo['min_recall']}"}
    if champ is None:
        return {"promote": True, "reason": "no existing champion"}
    champ_tags = champ.tags
    champ_metrics = rescore_on_test(
        _load_pipeline(champ.version, params), float(champ_tags["threshold"]), champ_tags["scenario"], params
    )
    delta = challenger_metrics["pr_auc"] - champ_metrics["pr_auc"]
    ok = delta >= -promo["epsilon"]
    return {
        "promote": bool(ok),
        "reason": f"challenger PR-AUC {challenger_metrics['pr_auc']:.4f} vs champion v{champ.version} "
                  f"{champ_metrics['pr_auc']:.4f} (delta {delta:+.4f}, epsilon {promo['epsilon']})",
        "champion_version": champ.version,
    }


def export_champion(client: MlflowClient | None = None, params: dict | None = None) -> ModelBundle:
    """Registry champion -> models/champion/ (the artefact baked into the Docker image)."""
    params = params or load_params()
    client = client or configure(params)
    mv = champion_version(client, params)
    if mv is None:
        raise SystemExit("No champion registered. Run: python -m ml.train")
    pipeline = _load_pipeline(mv.version, params)
    with tempfile.TemporaryDirectory() as tmp:
        ref_p = client.download_artifacts(mv.run_id, "bundle/reference_stats.json", tmp)
        meta_p = client.download_artifacts(mv.run_id, "bundle/metadata.json", tmp)
        reference = json.loads(open(ref_p).read())
        metadata = json.loads(open(meta_p).read())
    metadata["model_version"] = mv.version
    metadata["registered_model"] = params["mlflow"]["registered_model"]
    metadata["alias"] = params["mlflow"]["champion_alias"]
    save_bundle(path(params["serving"]["champion_dir"]), pipeline, metadata, reference)
    return ModelBundle(pipeline, metadata, reference)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["list", "export", "rollback"])
    ap.add_argument("--version")
    a = ap.parse_args()
    params = load_params()
    client = configure(params)
    name = params["mlflow"]["registered_model"]
    if a.cmd == "list":
        champ = champion_version(client, params)
        for mv in sorted(client.search_model_versions(f"name='{name}'"), key=lambda m: int(m.version)):
            aliases = [al for al, v in client.get_registered_model(name).aliases.items() if v == mv.version]
            print(f"v{mv.version}  aliases={aliases}  tags={dict(mv.tags)}")
        print("champion:", champ.version if champ else None)
    elif a.cmd == "export":
        b = export_champion(client, params)
        print(f"exported champion v{b.version} -> {params['serving']['champion_dir']}")
    elif a.cmd == "rollback":
        client.set_registered_model_alias(name, params["mlflow"]["champion_alias"], a.version)
        b = export_champion(client, params)
        print(f"champion alias -> v{a.version}; bundle re-exported")


if __name__ == "__main__":
    main()
