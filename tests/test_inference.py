"""Tests for the serving-time predictor (uses the real saved artifacts)."""
import pytest

from stroke_prediction import PatientRecord, StrokePredictor
from stroke_prediction.config import PATHS, TRAINING

_ARTIFACTS = PATHS.scaler.exists() and PATHS.model(TRAINING.production_model).exists()

pytestmark = pytest.mark.skipif(not _ARTIFACTS, reason="Model artifacts missing")


def _record(**overrides):
    base = dict(
        gender="Male", age=79, hypertension=1, heart_disease=1,
        ever_married="Yes", work_type="Private", avg_glucose_level=220,
        bmi=33, smoking_status="formerly smoked",
    )
    base.update(overrides)
    return PatientRecord(**base)


def test_prediction_shape_and_range():
    result = StrokePredictor().predict(_record())
    d = result.as_dict()
    assert set(d) == {"label", "probability"}
    assert d["label"] in (0, 1)
    assert 0.0 <= d["probability"] <= 1.0


def test_high_and_low_risk_separation():
    predictor = StrokePredictor()
    high = predictor.predict(_record())
    low = predictor.predict(_record(
        age=25, hypertension=0, heart_disease=0, ever_married="No",
        avg_glucose_level=85, bmi=22, smoking_status="never smoked",
    ))
    assert high.probability > low.probability


def test_invalid_input_rejected():
    with pytest.raises(ValueError):
        _record(gender="Robot")
