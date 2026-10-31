# Inference service image: FastAPI + the exported champion bundle (models/champion).
# Build context = project root.  Build AFTER `python -m ml.train` / `python -m ml.registry export`.
FROM python:3.12-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 \
    MODEL_DIR=/app/models/champion MONITORING_DB=/app/monitoring/data/monitoring.db

# xgboost needs the OpenMP runtime
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements-serve.txt .
RUN pip install -r requirements-serve.txt

# Only what serving needs (no training data, no MLflow store)
COPY params.yaml .
COPY ml/__init__.py ml/config.py ml/features.py ml/bundle.py ml/
COPY monitoring/__init__.py monitoring/store.py monitoring/
COPY api/ api/
COPY models/champion/ models/champion/

RUN useradd -m -u 10001 appuser && mkdir -p /app/monitoring/data && chown -R appuser /app
USER appuser

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
  CMD curl -fsS http://localhost:8000/ready || exit 1

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
