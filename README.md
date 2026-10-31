# Student Performance & Dropout-Risk Prediction System — Part 1

> **Part 2 (MLOps extension: modelling, MLflow, API, Docker, monitoring) → see [README_PART2.md](README_PART2.md).**

Data pipeline: CSV (UCI dataset) → Python ingestion → Airflow orchestration →
validation/cleaning → PostgreSQL warehouse → analytics marts → Streamlit dashboard.

## 1. Setup

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env            # then fill in your PostgreSQL credentials
```

Create the database once (if it doesn't exist yet):
```bash
createdb student_analytics
```

## 2. Get the source data
Either let `ingest.py` auto-download the public UCI archive, or manually place
`student-mat.csv` and `student-por.csv` (from
https://archive.ics.uci.edu/dataset/320/student+performance) into `data/raw/`.

## 3. Run the pipeline

### Option A — quick local run (no Airflow needed for a demo)
```bash
python run_pipeline.py
```

### Option B — via Airflow (required orchestration deliverable)
```bash
export AIRFLOW_HOME=$(pwd)/airflow_home
airflow db init
cp dags/student_performance_dag.py $AIRFLOW_HOME/dags/
airflow webserver -p 8080 &
airflow scheduler &
# then trigger student_performance_pipeline from the Airflow UI, or:
airflow dags trigger student_performance_pipeline
```

Each stage can also be run individually for debugging:
```bash
python -m scripts.ingest
python -m scripts.validate_clean
python -m scripts.transform
python -m scripts.load_to_postgres
```

## 4. Launch the dashboard
```bash
streamlit run dashboard/app.py
```
The dashboard reads from PostgreSQL and automatically falls back to the local
`data/analytical/*.csv` files if the database isn't reachable.

## 5. Project layout
```
student_pipeline/
├── data/
│   ├── raw/            # untouched extracted copies
│   ├── staging/        # standardized, pre-validation
│   ├── cleaned/         # validated, analysis-ready per source
│   ├── analytical/      # fact/dim/mart tables (also loaded to Postgres)
│   └── logs/            # ingestion manifest, rejected-record logs, pipeline.log
├── scripts/
│   ├── config.py         # env-based configuration
│   ├── pipeline_logger.py
│   ├── ingest.py         # Stage 1
│   ├── validate_clean.py # Stage 2
│   ├── transform.py      # Stage 3
│   └── load_to_postgres.py # Stage 4
├── sql/schema.sql        # star-schema DDL
├── dags/student_performance_dag.py
├── dashboard/app.py       # Stage 5 — Streamlit
├── run_pipeline.py        # end-to-end runner (non-Airflow)
├── data_dictionary.md
└── requirements.txt
```

## 6. Data source
UCI Student Performance Dataset (public, free, no auth):
https://archive.ics.uci.edu/dataset/320/student+performance
Files used: `student-mat.csv`, `student-por.csv`.

## 7. Data-quality checks implemented
See `data_dictionary.md` → "Validation rules applied" for the full list
(duplicate removal, grade-range validation, missing-value imputation,
rejected-record logging).
