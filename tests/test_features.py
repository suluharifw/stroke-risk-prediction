"""Tests for derived clinical categories."""
from stroke_prediction.features import glucose_category, weight_category


def test_weight_category_bands():
    assert weight_category(17.0) == "Underweight"
    assert weight_category(22.0) == "Normal"
    assert weight_category(27.0) == "Overweight"
    assert weight_category(35.0) == "Obesity"


def test_glucose_category_bands():
    assert glucose_category(60.0) == "Hipoglikemia"
    assert glucose_category(100.0) == "Normal"
    assert glucose_category(180.0) == "prediabetic"
    assert glucose_category(250.0) == "Hiperglikemia"
