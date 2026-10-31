"""
Self-contained, MLflow-free model bundle: what the API container loads.

models/champion/
    model.joblib          sklearn Pipeline (raw record -> probability)
    metadata.json         version, scenario, threshold, bands, metrics, lineage
    reference_stats.json  training distribution for drift / quality monitoring
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from ml.features import raw_features_for


@dataclass
class ModelBundle:
    pipeline: object
    metadata: dict
    reference: dict = field(default_factory=dict)

    @property
    def scenario(self) -> str:
        return self.metadata["scenario"]

    @property
    def threshold(self) -> float:
        return float(self.metadata["threshold"])

    @property
    def version(self) -> str:
        return str(self.metadata.get("model_version", "local"))

    @property
    def input_features(self) -> list[str]:
        return raw_features_for(self.scenario)

    def predict_proba(self, records: pd.DataFrame) -> np.ndarray:
        return self.pipeline.predict_proba(records)[:, 1]

    def band(self, p: float) -> str:
        """Low = below the decision threshold (not flagged); Medium = flagged; High = flagged with high confidence."""
        return "High" if p >= self.metadata["risk_bands"]["high"] else "Medium" if p >= self.threshold else "Low"


def save_bundle(dest: Path, pipeline, metadata: dict, reference: dict) -> Path:
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, dest / "model.joblib", compress=3)
    (dest / "metadata.json").write_text(json.dumps(metadata, indent=2, default=str))
    (dest / "reference_stats.json").write_text(json.dumps(reference))
    return dest


def load_bundle(src: Path) -> ModelBundle:
    src = Path(src)
    pipeline = joblib.load(src / "model.joblib")
    metadata = json.loads((src / "metadata.json").read_text())
    ref_path = src / "reference_stats.json"
    reference = json.loads(ref_path.read_text()) if ref_path.exists() else {}
    return ModelBundle(pipeline, metadata, reference)
