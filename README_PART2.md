# Part 2 — MLOps Extension: Student Failure-Risk Prediction

Builds on the Part 1 pipeline (CSV → Airflow → clean → PostgreSQL → marts → Streamlit). **Part 1 code is untouched**; Part 2 adds `ml/`, `api/`, `monitoring/`, `tests/`, `docs/`, two dashboard pages and a few config files.

| Assignment item | Where |
|---|---|
| Prediction target | `fail = G3 < 10` — rationale in `docs/MODEL_CARD.md` |
| Baseline + advanced models (LR, RF, XGBoost) + trivial baselines | `ml/train.py`, grids in `params.yaml` |
| Reproducible feature-engineering & training pipeline | `ml/features.py` (sklearn Pipeline, raw record → probability), `dvc.yaml`, seeded |
| Train/val/test split | `ml/data.py` — stratified, **student-grouped** (no timestamp exists; see model card) |
| MLflow tracking | `ml/train.py` → `mlflow/` store; `mlflow ui --backend-store-uri sqlite:///mlflow/mlflow.db` |
| Register & version model | `ml/registry.py` (aliases `challenger` / `champion`, gated promotion, rollback) |
| Dataset/model versioning | `dvc.yaml` + Git-based metadata (dataset sha256, git SHA in `metadata.json`/MLflow tags) |
| FastAPI service | `api/main.py` (`/predict`, `/predict/batch`, `/feedback`, `/model/info`, `/health`, `/ready`, `/metrics`) |
| Docker | `Dockerfile`, `docker-compose.yml` (api + MLflow UI + dashboard) |
| Streamlit / Tableau integration | `dashboard/pages/1_Risk_Prediction.py`, `data/analytical/mart_student_risk_predictions.csv` |
| Monitoring: quality, drift, performance, latency, failures | `monitoring/` + `dashboard/pages/2_Model_Monitoring.py` |
| Retraining criteria & lifecycle | `docs/MODEL_LIFECYCLE.md`, `monitoring/retrain.py`, `dags/student_ml_lifecycle` |

## Quick start
```bash
pip install -r requirements-ml.txt
python -m ml.data                      # modelling dataset + split   (or:  dvc init && dvc repro)
python -m ml.train                     # tracks in MLflow, registers v1, promotes it, exports models/champion/
python -m ml.train --scenario early    # optional comparison run (G1 only), not registered
python -m ml.batch_score               # model-driven risk mart → Streamlit / Tableau (Postgres load is best-effort)

uvicorn api.main:app --port 8000       # http://localhost:8000/docs
streamlit run dashboard/app.py         # sidebar: Risk Prediction, Model Monitoring
mlflow ui --backend-store-uri sqlite:///mlflow/mlflow.db

python -m monitoring.simulate --mode normal --n 150     # demo traffic (also: covariate_drift, concept_drift)
python -m monitoring.report                             # report + retrain decision (exit code 0/1/2)
pytest                                                  # 34 tests (~2 min, includes an end-to-end training run)
```
Docker: `python -m ml.train && docker compose up --build` (API :8000, MLflow :5000, dashboard :8501).

Example call
```bash
curl -s localhost:8000/predict -H 'content-type: application/json' \
  -d '{"student_id":"S1","subject":"math","g1":6,"g2":4,"absences":12,"failures":1}'
# later, when the final grade is known:
curl -s localhost:8000/feedback -H 'content-type: application/json' -d '{"student_id":"S1","actual_final_grade":7}'
```

## Headline result (details & caveats: `docs/MODEL_CARD.md`)
Champion = Logistic Regression, `mid_term` scenario (G1+G2): test PR-AUC 0.85, recall 0.94, precision 0.58 on 212 held-out rows (47 failing). LR, RF and XGBoost are statistically indistinguishable, and with G2 known the rule "G2 < 10" matches the model's recall with higher precision — the model adds ranking and the `early` (G1-only) use case, where it clearly beats the rule (PR-AUC 0.74 vs 0.56).

## Things found in Part 1 that affect Part 2 (not changed)
1. `transform.build_fact_table` globs every `*_cleaned.csv`, so the stale `student-mat_cleaned.csv` / `student-por_cleaned.csv` double every record (fact table = 2,088 rows instead of 1,044; the dashboard shows 1,693 "students"). Part 2 filters to `math` / `portuguese`. Fix: delete the two stale cleaned/staged/raw files, or restrict the glob.
2. `risk_category` is never "High" (max risk_score 63), so `mart_high_risk_students` is empty. The Part 2 mart replaces it with model-driven bands.
3. `dashboard/app.py` uses `trendline="ols"` but `statsmodels` is not in `requirements.txt` (added to `requirements-ml.txt`).

## Not verified in this environment
* `docker build` / `docker compose up` — no Docker daemon was available. The serving layout was instead run in a clean venv with only `requirements-serve.txt` (uvicorn + `/predict` + `/metrics` OK).
* PostgreSQL load in `ml.batch_score`, and the Airflow DAG `student_ml_lifecycle` (syntax-checked only; Airflow not installed).
* The MLflow store is not shipped (it records absolute paths); `python -m ml.train` recreates it in about a minute.
