"""Tests for cleaning and feature-matrix assembly."""
import pandas as pd

from stroke_prediction.config import SCHEMA
from stroke_prediction.preprocessing import build_feature_matrix, clean


def _sample() -> pd.DataFrame:
    # 'never smoked' is the unambiguous mode, mirroring the real dataset.
    return pd.DataFrame({
        "id": [1, 2, 3, 4],
        "gender": ["Male", "Female", "Male", "Female"],
        "age": [67, 45, 30, 52],
        "hypertension": [0, 1, 0, 0],
        "heart_disease": [1, 0, 0, 0],
        "ever_married": ["Yes", "No", "Yes", "Yes"],
        "work_type": ["Private", "Govt_job", "Self-employed", "Private"],
        "Residence_type": ["Urban", "Rural", "Urban", "Rural"],
        "avg_glucose_level": [228.0, 105.0, 90.0, 110.0],
        "bmi": [36.6, None, 24.0, 28.0],
        "smoking_status": ["formerly smoked", "Unknown", "never smoked", "never smoked"],
        "stroke": [1, 0, 0, 0],
    })


def test_clean_imputes_and_drops():
    out = clean(_sample())
    assert "id" not in out.columns
    assert "Residence_type" not in out.columns
    assert out["bmi"].isna().sum() == 0
    assert "Unknown" not in out["smoking_status"].values


def test_feature_matrix_shape_and_order():
    X, y = build_feature_matrix(_sample())
    assert list(X.columns) == list(SCHEMA.feature_order)
    assert X.shape[1] == 23
    assert set(y.unique()) <= {0, 1}
