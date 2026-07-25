"""Feature engineering: derived clinical categories.

Two domain features are derived from continuous measurements, following the
bands used in the original analysis:

* ``Weight_Category``  — WHO BMI classification.
* ``Glucose_Category`` — glucose-tolerance-test bands.
"""
from __future__ import annotations

import pandas as pd

# WHO BMI bands.
UNDERWEIGHT_MAX = 18.5
NORMAL_MAX = 24.9
OVERWEIGHT_MAX = 29.9

# Glucose bands (mg/dL).
HYPOGLYCEMIA_MAX = 70.0
NORMAL_GLUCOSE_MAX = 140.0
PREDIABETIC_MAX = 199.0


def weight_category(bmi: float) -> str:
    """Classify BMI into a WHO weight category."""
    if bmi < UNDERWEIGHT_MAX:
        return "Underweight"
    if bmi <= NORMAL_MAX:
        return "Normal"
    if bmi <= OVERWEIGHT_MAX:
        return "Overweight"
    return "Obesity"


def glucose_category(glucose: float) -> str:
    """Classify an average glucose level into a clinical band."""
    if glucose < HYPOGLYCEMIA_MAX:
        return "Hipoglikemia"
    if glucose <= NORMAL_GLUCOSE_MAX:
        return "Normal"
    if glucose <= PREDIABETIC_MAX:
        return "prediabetic"
    return "Hiperglikemia"


def add_clinical_categories(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of ``frame`` with weight and glucose categories added."""
    out = frame.copy()
    out["Weight_Category"] = out["bmi"].apply(weight_category)
    out["Glucose_Category"] = out["avg_glucose_level"].apply(glucose_category)
    return out
