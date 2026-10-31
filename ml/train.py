"""
Stage `train`: baselines + Logistic Regression / Random Forest / XGBoost,
tracked in MLflow, champion selected on VALIDATION, evaluated ONCE on TEST,
registered, gated-promoted and exported as a serving bundle.

    python -m ml.train                       # uses params.yaml
    python -m ml.train --scenario early      # comparison run (not registered)
    python -m ml.train --no-register
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import mlflow  # noqa: E402
import mlflow.sklearn  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from mlflow.models import infer_signature  # noqa: E402
from sklearn.calibration import calibration_curve  # noqa: E402
from sklearn.dummy import DummyClassifier  # noqa: E402
from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.inspection import permutation_importance  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import ConfusionMatrixDisplay, PrecisionRecallDisplay, RocCurveDisplay  # noqa: E402
from sklearn.model_selection import ParameterGrid  # noqa: E402
from xgboost import XGBClassifier  # noqa: E402

from ml import registry  # noqa: E402
from ml.bundle import save_bundle  # noqa: E402
from ml.config import PARAMS_PATH, git_commit, load_params, path  # noqa: E402
from ml.data import file_sha256, load_modelling  # noqa: E402
from ml.features import build_pipeline, raw_features_for  # noqa: E402
from ml.metrics import best_threshold, bootstrap_ci, compute_metrics  # noqa: E402
from ml.reference import build_reference  # noqa: E402

FAMILY_ORDER = ["logistic_regression", "random_forest", "xgboost"]   # simple -> complex


def make_pipeline(family: str, hp: dict, scenario: str, seed: int, pos_weight: float):
    if family == "logistic_regression":
        est = LogisticRegression(C=hp["C"], class_weight="balanced", max_iter=5000, random_state=seed)
        return build_pipeline(est, scenario, scale=True)
    if family == "random_forest":
        est = RandomForestClassifier(class_weight="balanced_subsample", random_state=seed, n_jobs=-1, **hp)
        return build_pipeline(est, scenario)
    if family == "xgboost":
        est = XGBClassifier(
            eval_metric="logloss", random_state=seed, n_jobs=1, tree_method="hist",
            scale_pos_weight=pos_weight, **hp,
        )
        return build_pipeline(est, scenario)
    raise ValueError(family)


def _log_metrics(prefix: str, m: dict):
    mlflow.log_metrics({f"{prefix}_{k}": v for k, v in m.items() if isinstance(v, (int, float)) and not np.isnan(v)})


def _slice_metrics(test: pd.DataFrame, proba, thr) -> dict:
    out = {}
    for col in ["subject", "sex", "school"]:
        for val, g in test.groupby(col):
            m = compute_metrics(g["fail"], proba[g.index.map(test.index.get_loc)], thr)
            out[f"{col}={val}"] = {k: m[k] for k in ["n", "positives", "recall", "precision", "pr_auc"]}
    return out


def run(scenario: str | None = None, register: bool = True, params: dict | None = None) -> dict:
    params = params or load_params()
    scenario = scenario or params["features"]["scenario"]
    if scenario != params["features"]["scenario"]:
        register = False                       # only the configured scenario feeds the registry
    seed = params["seed"]
    rep_dir = path(params.get("reports_dir", "reports")); rep_dir.mkdir(parents=True, exist_ok=True)
    sfx = "" if scenario == params["features"]["scenario"] else f"_{scenario}"   # comparison runs never overwrite the main reports
    mcfg = params["model"]

    df = load_modelling(params)
    cols = raw_features_for(scenario)
    tr, va, te = (df[df["split"] == s].reset_index(drop=True) for s in ("train", "val", "test"))
    Xtr, ytr, Xva, yva, Xte, yte = tr[cols], tr["fail"], va[cols], va["fail"], te[cols], te["fail"]
    pos_weight = float((ytr == 0).sum() / max((ytr == 1).sum(), 1))

    client = registry.configure(params)
    mlflow.set_experiment(params["mlflow"]["experiment"])
    data_sha = file_sha256(path(params["data"]["modelling_dataset"]))
    commit = git_commit()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    with mlflow.start_run(run_name=f"train_{scenario}_{stamp}") as parent:
        mlflow.set_tags({
            "scenario": scenario, "dataset_sha256": data_sha, "git_commit": commit,
            "target": f"fail = G3 < {params['data']['pass_mark']}",
            "split_method": params["split"]["method"],
        })
        mlflow.log_params({
            "seed": seed, "scenario": scenario, "n_train": len(tr), "n_val": len(va), "n_test": len(te),
            "train_fail_rate": round(float(ytr.mean()), 4), "n_input_features": len(cols),
            "selection_metric": mcfg["selection_metric"], "threshold_strategy": mcfg["threshold_strategy"],
        })
        mlflow.log_artifact(str(PARAMS_PATH), "config")
        mlflow.log_artifact(str(path(params.get("reports_dir", "reports")) / "data_profile.json"), "data")

        # ---------------- baselines -----------------
        baselines = {}
        prior = DummyClassifier(strategy="prior").fit(Xtr, ytr)
        baselines["prior_only"] = compute_metrics(yva, prior.predict_proba(Xva)[:, 1], 0.5)
        rule_col = "g2" if scenario == "mid_term" else "g1"      # "last known grade below pass mark"
        rule_val = (Xva[rule_col] < params["data"]["pass_mark"]).astype(float).values
        rule_te = (Xte[rule_col] < params["data"]["pass_mark"]).astype(float).values
        baselines[f"rule_{rule_col}_below_pass"] = compute_metrics(yva, rule_val, 0.5)
        test_baselines = {
            "prior_only": compute_metrics(yte, prior.predict_proba(Xte)[:, 1], 0.5),
            f"rule_{rule_col}_below_pass": compute_metrics(yte, rule_te, 0.5),
        }
        for name, m in baselines.items():
            with mlflow.start_run(run_name=f"baseline_{name}", nested=True):
                mlflow.set_tags({"stage": "baseline", "family": name})
                _log_metrics("val", m)

        # ---------------- candidate search -----------------
        results = []
        for family in FAMILY_ORDER:
            for hp in ParameterGrid(mcfg["candidates"][family]["grid"]):
                with mlflow.start_run(run_name=f"{family}", nested=True):
                    mlflow.set_tags({"stage": "candidate", "family": family})
                    mlflow.log_params(hp)
                    t0 = time.time()
                    pipe = make_pipeline(family, hp, scenario, seed, pos_weight).fit(Xtr, ytr)
                    fit_s = time.time() - t0
                    p_tr, p_va = pipe.predict_proba(Xtr)[:, 1], pipe.predict_proba(Xva)[:, 1]
                    thr = best_threshold(yva, p_va, mcfg["threshold_strategy"])
                    m_va, m_tr = compute_metrics(yva, p_va, thr), compute_metrics(ytr, p_tr, thr)
                    _log_metrics("val", m_va)
                    _log_metrics("train", m_tr)
                    mlflow.log_metric("fit_seconds", fit_s)
                    results.append({"family": family, "hp": hp, "pipe": pipe, "thr": thr,
                                    "val": m_va, "train": m_tr, "p_va": p_va})

        # ---------------- champion selection -----------------
        sel = mcfg["selection_metric"]
        best_per_family = {}
        for r in results:
            cur = best_per_family.get(r["family"])
            if cur is None or r["val"][sel] > cur["val"][sel]:
                best_per_family[r["family"]] = r
        overall_best = max(r["val"][sel] for r in best_per_family.values())
        champion = next(
            best_per_family[f] for f in FAMILY_ORDER
            if f in best_per_family and best_per_family[f]["val"][sel] >= overall_best - mcfg["simplicity_tolerance"]
        )
        leaderboard = {f: {"hp": r["hp"], f"val_{sel}": round(r["val"][sel], 4), "val_recall": round(r["val"]["recall"], 4),
                           "train_pr_auc": round(r["train"]["pr_auc"], 4)} for f, r in best_per_family.items()}
        mlflow.log_dict(leaderboard, "selection/leaderboard.json")
        mlflow.set_tags({"champion_family": champion["family"]})
        mlflow.log_params({f"champion_{k}": v for k, v in champion["hp"].items()} | {"champion_family": champion["family"]})

        # ---------------- final, single test evaluation -----------------
        pipe, thr = champion["pipe"], champion["thr"]
        p_te = pipe.predict_proba(Xte)[:, 1]
        m_te = compute_metrics(yte, p_te, thr)
        ci = bootstrap_ci(yte, p_te, thr, seed=seed)
        # Every candidate family on test too, for the report (selection already happened on val).
        family_test = {f: compute_metrics(yte, r["pipe"].predict_proba(Xte)[:, 1], r["thr"])
                       for f, r in best_per_family.items()}
        _log_metrics("val", champion["val"])
        _log_metrics("test", m_te)
        for k, (lo, hi) in ci.items():
            mlflow.log_metrics({f"test_{k}_ci_lo": lo, f"test_{k}_ci_hi": hi})
        slices = _slice_metrics(te, p_te, thr)
        mlflow.log_dict(slices, "evaluation/slice_metrics.json")

        # ---------------- plots -----------------
        fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
        PrecisionRecallDisplay.from_predictions(yte, p_te, ax=ax[0], name=champion["family"])
        ax[0].axhline(yte.mean(), ls=":", c="grey", label="prevalence"); ax[0].legend(); ax[0].set_title("Precision-Recall (test)")
        RocCurveDisplay.from_predictions(yte, p_te, ax=ax[1], name=champion["family"]); ax[1].set_title("ROC (test)")
        fr, mp = calibration_curve(yte, p_te, n_bins=6, strategy="quantile")
        ax[2].plot(mp, fr, "o-"); ax[2].plot([0, 1], [0, 1], "k:"); ax[2].set_title("Calibration (test)")
        ax[2].set_xlabel("mean predicted"); ax[2].set_ylabel("observed fail rate")
        fig.tight_layout(); mlflow.log_figure(fig, "evaluation/curves.png"); fig.savefig(rep_dir / f"curves{sfx}.png", dpi=130); plt.close(fig)

        fig, ax = plt.subplots(figsize=(4.2, 4))
        ConfusionMatrixDisplay.from_predictions(yte, (p_te >= thr).astype(int), ax=ax, display_labels=["pass", "fail"], colorbar=False)
        ax.set_title(f"Confusion (test, thr={thr})"); fig.tight_layout()
        mlflow.log_figure(fig, "evaluation/confusion_matrix.png"); fig.savefig(rep_dir / f"confusion_matrix{sfx}.png", dpi=130); plt.close(fig)

        pi = permutation_importance(pipe, Xva, yva, scoring="average_precision", n_repeats=20, random_state=seed)
        imp = pd.Series(pi.importances_mean, index=cols).sort_values(ascending=False)
        mlflow.log_dict(imp.round(5).to_dict(), "evaluation/permutation_importance.json")
        fig, ax = plt.subplots(figsize=(6, 5))
        imp.head(12)[::-1].plot.barh(ax=ax); ax.set_title("Permutation importance (val, PR-AUC drop)")
        fig.tight_layout(); mlflow.log_figure(fig, "evaluation/feature_importance.png"); fig.savefig(rep_dir / f"feature_importance{sfx}.png", dpi=130); plt.close(fig)

        # ---------------- bundle metadata & reference -----------------
        reference = build_reference(tr, scenario, val_scores=champion["p_va"])
        metadata = {
            "model_name": params["mlflow"]["registered_model"],
            "model_family": champion["family"],
            "hyperparameters": champion["hp"],
            "scenario": scenario,
            "target": f"fail = G3 < {params['data']['pass_mark']}",
            "threshold": thr,
            "risk_bands": mcfg["risk_bands"],
            "trained_at": datetime.now(timezone.utc).isoformat(),
            "run_id": parent.info.run_id,
            "dataset_sha256": data_sha,
            "git_commit": commit,
            "n_train": len(tr),
            "metrics": {"validation": champion["val"], "test": m_te, "test_ci95": ci},
            "input_features": cols,
        }
        mlflow.log_dict(metadata, "bundle/metadata.json")
        mlflow.log_dict(reference, "bundle/reference_stats.json")

        sig = infer_signature(Xtr.head(50), pipe.predict_proba(Xtr.head(50)))
        info = mlflow.sklearn.log_model(pipe, name="model", signature=sig, input_example=Xtr.head(3),
                                      serialization_format="cloudpickle")

        decision = {"promote": False, "reason": "registration disabled"}
        version = None
        if register:
            name = params["mlflow"]["registered_model"]
            mv = mlflow.register_model(info.model_uri, name)
            version = mv.version
            for k, v in {
                "scenario": scenario, "family": champion["family"], "threshold": thr,
                "val_pr_auc": round(champion["val"]["pr_auc"], 4), "test_pr_auc": round(m_te["pr_auc"], 4),
                "test_recall": round(m_te["recall"], 4), "dataset_sha256": data_sha[:16], "git_commit": commit,
            }.items():
                client.set_model_version_tag(name, version, k, str(v))
            client.set_registered_model_alias(name, params["mlflow"]["challenger_alias"], version)
            decision = registry.promotion_decision(client, params, version, m_te)
            if decision["promote"]:
                client.set_registered_model_alias(name, params["mlflow"]["champion_alias"], version)
                metadata.update({"model_version": version, "registered_model": name, "alias": "champion"})
                save_bundle(path(params["serving"]["champion_dir"]), pipe, metadata, reference)
            client.set_model_version_tag(name, version, "promotion", json.dumps(decision))
            mlflow.set_tag("registered_version", version)
            mlflow.set_tag("promoted", str(decision["promote"]))

        summary = {
            "scenario": scenario,
            "champion": {"family": champion["family"], "hp": champion["hp"], "threshold": thr, "registry_version": version},
            "validation": champion["val"], "test": m_te, "test_ci95": ci,
            "leaderboard_val": leaderboard,
            "test_by_family": {f: {k: round(v[k], 4) for k in ["pr_auc", "roc_auc", "recall", "precision", "f2"]} for f, v in family_test.items()},
            "baselines_test": test_baselines,
            "baselines_val": baselines,
            "slices_test": slices,
            "top_features": imp.head(10).round(4).to_dict(),
            "promotion": decision,
            "run_id": parent.info.run_id,
        }
        (rep_dir / ("evaluation.json" if scenario == params["features"]["scenario"] else f"evaluation_{scenario}.json")).write_text(
            json.dumps(summary, indent=2, default=float))
        # compact, flat file for `dvc metrics show/diff`
        compact = {"threshold": thr, **{f"val_{k}": round(champion["val"][k], 4) for k in ["pr_auc", "recall", "precision"]},
                   **{f"test_{k}": round(m_te[k], 4) for k in ["pr_auc", "roc_auc", "recall", "precision", "f2", "brier"]}}
        out = rep_dir / ("metrics.json" if scenario == params["features"]["scenario"] else f"metrics_{scenario}.json")
        out.write_text(json.dumps(compact, indent=2))
        mlflow.log_artifact(str(rep_dir / f"evaluation{'' if scenario == params['features']['scenario'] else '_' + scenario}.json"), "evaluation")
        return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", choices=["early", "mid_term"])
    ap.add_argument("--no-register", action="store_true")
    a = ap.parse_args()
    s = run(a.scenario, register=not a.no_register)
    print(json.dumps({k: s[k] for k in ["scenario", "champion", "test", "test_ci95", "test_by_family", "baselines_test", "promotion"]}, indent=2, default=float))
