# Model Lifecycle & Retraining Policy

```
 Part 1 pipeline                       Part 2
 ───────────────                       ───────────────────────────────────────────────────────────────
 ingest → clean → transform ─► fact_student_performance.csv
                                   │
                         [DVC stage: prepare]   ml/data.py        → data/modelling/student_modelling.csv  (+ split, group_id, sha256)
                                   │
                         [DVC stage: train]     ml/train.py       baselines + LR/RF/XGB grid, MLflow runs
                                   │                               select on VALIDATION → one-shot TEST eval
                                   ▼
                  MLflow Registry  student-fail-risk   v_N  ──alias──► challenger
                                   │  promotion gate (re-score champion & challenger on current test split)
                                   ▼
                              alias `champion` ──► ml.registry export ──► models/champion/ (joblib + metadata + reference stats)
                                   │                                              │
                    [DVC stage: score] ml/batch_score.py                  docker build → FastAPI /predict
                                   │                                              │ logs every request/prediction (SQLite)
              data/analytical/mart_student_risk_predictions.csv          POST /feedback (real outcome, later)
               └► Streamlit "Risk Prediction"  └► Tableau                          │
                                                                                   ▼
                                   monitoring.report  (drift · quality · performance · latency · failures)
                                                       │
                                   retrain decision: NO_ACTION │ INVESTIGATE │ RETRAIN ─► back to `prepare`
```

## 1. Stages and owners
| Stage | What happens | Gate to leave the stage |
|---|---|---|
| **Data** | Part 1 produces the fact table; `prepare` filters to canonical sources, builds target, assigns the grouped split | `group_leakage == 0`, no duplicate `student_id`, fail-rate within 15–30 % (tests) |
| **Experiment** | Every candidate is an MLflow run (params, val/train metrics, fit time); the parent run holds config, dataset sha256, git commit, curves, permutation importance, slice metrics, the model | – |
| **Selection** | Best config per family by validation PR-AUC; the **simplest** family within `simplicity_tolerance` (0.01) of the best wins | – |
| **Registered (challenger)** | Version created, tags: scenario, family, threshold, val/test metrics, dataset hash, git SHA | recall ≥ `min_recall` (0.70) |
| **Champion** | Alias `champion` moved to the version, serving bundle exported | challenger PR-AUC ≥ champion PR-AUC − `epsilon` (0.005), both **re-scored on the current test split** |
| **Serving** | Docker image built from `models/champion/`; `/model/info` reports version, threshold, metrics, lineage | `/ready` returns 200 (container HEALTHCHECK) |
| **Monitoring** | Weekly (Airflow `student_ml_lifecycle`) or on demand | see §3 |
| **Retired** | Older versions stay in the registry (no alias); rollback = move the alias back | – |

## 2. Versioning
* **Data**: `dvc.yaml` (`prepare`, `train`, `score`) — `dvc repro` rebuilds only what changed; `dvc metrics diff` / `dvc params diff` compare runs. Independently of DVC, every run records the dataset **sha256** and **git commit** (MLflow tags, registry-version tags, `models/champion/metadata.json`) — the Git-based metadata route, which works with plain `git`.
* **Model**: MLflow Model Registry versions + aliases (`challenger`, `champion`). Stage names are deprecated in MLflow, so aliases are used.
* **Config**: `params.yaml` is the single source of truth (splits, grids, thresholds, monitoring and retraining limits) and is logged with each run.
* **Code/serving**: the Docker image bundles code + the exact exported model; pinned `requirements-serve.txt` must match the training environment's scikit-learn/xgboost.

Promote / roll back
```bash
python -m ml.train                       # → new version, gate decides promotion, bundle re-exported when promoted
python -m ml.registry list
python -m ml.registry rollback --version 1   # alias champion → v1 and re-export bundle
docker compose up --build api            # redeploy
```

## 3. Monitoring (what is measured, how, thresholds in `params.yaml → monitoring`)
| Area | Signal | Method | WARN / ALERT |
|---|---|---|---|
| **Input data quality** | missing rate, out-of-range, unseen categories, missing columns | compare with training reference | missing +10 pp · out-of-range > 1 % · unseen > 2 % |
| **Feature drift** | each of 32 model inputs | PSI vs. training + KS (numeric) / chi-square (categorical) | PSI ≥ 0.10 / PSI ≥ 0.25 **and** p < 0.01 |
| **Prediction drift** | score distribution vs. validation | PSI on fixed score bins | same |
| **Class distribution** | predicted-fail rate and band mix always; *actual* fail rate once outcomes arrive | vs. training prevalence (21.9 %) | shift > 10 pp → retrain |
| **Performance** | recall, precision, PR-AUC, Brier on outcomes posted to `/feedback` | vs. held-out test metrics | recall drop > 0.10 or PR-AUC drop > 0.10 (≥ 50 outcomes) |
| **Latency** | p50 / p95 / p99 per endpoint | request log + Prometheus histogram | p95 ≥ 200 ms / ≥ 500 ms |
| **Failures** | 5xx rate; 4xx (rejected inputs) rate | request log + counters | 5xx ≥ 1 % / ≥ 2 %; 4xx ≥ 25 % → WARN |

Ground truth arrives late (the final grade, G3, exists only at term end), so performance monitoring is delayed by design; drift and score monitoring are the early-warning signals until labels land.
PSI is noisy on small samples: with ≥ 3 features' worth of noise and n = 150 drawn from the same distribution, an unguarded PSI ≥ 0.25 rule false-alarmed on ~13 % of runs; requiring statistical significance as well cut that to ~5 % (any feature) and ~1.5 % (a key feature). At n = 50 it is ~26 %, hence `min_samples: 50` and the advice to evaluate with ≥ 150 predictions.

## 4. Retraining criteria
`monitoring.report` returns **RETRAIN** if any of:

| ID | Trigger | Default |
|---|---|---|
| R1 | Data drift: PSI ALERT on ≥ 3 features, **or** on any key feature (`g1`, `g2`, `absences`, `failures`) | – |
| R2 | Performance: recall drop > 0.10 or PR-AUC drop > 0.10 vs. test baseline, with ≥ 50 outcomes | – |
| R3 | Class prior: actual fail rate moved > 10 pp from training | 21.9 % ± 10 pp |
| R4 | Model age > 180 days (one academic term) | – |
| R5 | New labelled records ≥ 25 % of the training set | 156 rows |

**INVESTIGATE** (engineering, not retraining): service p95/5xx ALERT, input-quality ALERT, or score drift with no feature/label evidence.
Exit codes 0 / 1 / 2 (`NO_ACTION` / `INVESTIGATE` / `RETRAIN`) let cron, CI or Airflow branch on the result.

**Retraining procedure**: append new terms to the Part 1 fact table → `dvc repro` (or `python -m ml.data && python -m ml.train`) → gate promotes only if the challenger is not worse than the champion on the same test split and keeps recall ≥ 0.70 → export → rebuild image → verify `/model/info` → keep the previous version for rollback. Because the test set is small (212 rows), a human should read the MLflow comparison before a *non*-automatic promotion; the weekly DAG retrains automatically but never deploys the container by itself.

## 5. Validated behaviour (synthetic stress tests, `tests/test_monitoring.py`, `docs/examples/`)
| Scenario (150 requests) | Overall | Decision | Fired |
|---|---|---|---|
| normal (held-out rows + true outcomes) | WARN (minor PSI) | NO_ACTION | – |
| covariate drift (grades −2…−5, absences ×2.5) | ALERT | RETRAIN | R1 (g1, g2, absences), R3 |
| concept drift (45 % of outcomes flipped) | ALERT | RETRAIN | R2 (recall −0.51), R3; **no** R1 |

These traffic patterns and labels are simulated; they show the detectors work, not how the real cohort will behave.
