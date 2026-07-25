"""Serving-time inference.

:class:`StrokePredictor` loads the scaler and the production model once and
turns a validated :class:`~stroke_prediction.schema.PatientRecord` into a
prediction. Feature vectors are built *by name* and reindexed to the model's
expected order, which removes an entire class of positional-ordering bugs.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .config import PATHS, SCHEMA, TRAINING
from .features import glucose_category, weight_category
from .io import load_joblib
from .logger import get_logger
from .schema import PatientRecord

logger = get_logger(__name__)


@dataclass(frozen=True)
class Prediction:
    """Result of a single inference call."""

    label: int          # 1 = higher risk, 0 = lower risk
    probability: float   # calibrated-ish P(stroke) in [0, 1]

    def as_dict(self) -> dict:
        return {"label": self.label, "probability": self.probability}


def _encode(record: PatientRecord) -> dict[str, float]:
    """Turn a validated record into a {feature_name: value} mapping."""
    features = {name: 0.0 for name in SCHEMA.feature_order}
    features["gender"] = SCHEMA.gender_map[record.gender]
    features["age"] = record.age
    features["hypertension"] = record.hypertension
    features["heart_disease"] = record.heart_disease
    features["ever_married"] = SCHEMA.married_map[record.ever_married]
    features["avg_glucose_level"] = record.avg_glucose_level
    features["bmi"] = record.bmi
    features[f"work_type_{record.work_type}"] = 1.0
    features[f"smoking_status_{record.smoking_status}"] = 1.0
    features[f"Weight_Category_{weight_category(record.bmi)}"] = 1.0
    features[f"Glucose_Category_{glucose_category(record.avg_glucose_level)}"] = 1.0
    return features


class StrokePredictor:
    """Loads artifacts lazily and predicts stroke risk for one patient."""

    def __init__(self, model_name: str = TRAINING.production_model) -> None:
        self._model_path = PATHS.model(model_name)
        self._scaler = None
        self._model = None

    def _ensure_loaded(self) -> None:
        if self._model is None:
            self._scaler = load_joblib(PATHS.scaler)
            self._model = load_joblib(self._model_path)
            logger.info("Loaded predictor: %s", self._model_path.name)

    def predict(self, record: PatientRecord) -> Prediction:
        """Predict stroke risk for a single validated patient record."""
        self._ensure_loaded()
        features = _encode(record)
        row = pd.DataFrame([[features[name] for name in SCHEMA.feature_order]],
                           columns=list(SCHEMA.feature_order))
        scaled = self._scaler.transform(row)
        proba = float(self._model.predict_proba(scaled)[0, 1])
        return Prediction(label=int(proba >= 0.5), probability=round(proba, 4))
