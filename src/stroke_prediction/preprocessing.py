"""Preprocessing: cleaning, encoding, and assembling the model matrix.

The steps mirror the original analysis, in order:

1. Impute missing ``bmi`` with the column mean.
2. Drop identifier / unused columns (``id``, ``Residence_type``).
3. Replace the ``Unknown`` smoking status with the column mode.
4. Cap ``avg_glucose_level`` and ``bmi`` outliers using the IQR rule.
5. Derive clinical categories (see :mod:`features`).
6. Label-encode binary categoricals and one-hot-encode the rest.
7. Reindex to the exact :data:`config.SCHEMA.feature_order`.
"""
from __future__ import annotations

import pandas as pd

from .config import SCHEMA
from .features import add_clinical_categories
from .logger import get_logger

logger = get_logger(__name__)

_ONE_HOT_COLUMNS = ["work_type", "smoking_status", "Weight_Category", "Glucose_Category"]
_OUTLIER_COLUMNS = ["avg_glucose_level", "bmi"]


def _iqr_bounds(series: pd.Series) -> tuple[float, float]:
    """Return the lower/upper IQR fences for outlier capping."""
    q1, q3 = series.quantile(0.25), series.quantile(0.75)
    iqr = q3 - q1
    return q1 - 1.5 * iqr, q3 + 1.5 * iqr


def clean(frame: pd.DataFrame) -> pd.DataFrame:
    """Apply steps 1-4: imputation, column drops, mode fill, outlier capping."""
    out = frame.copy()

    out["bmi"] = out["bmi"].fillna(out["bmi"].mean())
    out = out.drop(columns=[c for c in SCHEMA.drop_columns if c in out.columns])

    smoking_mode = out["smoking_status"].mode()[0]
    out["smoking_status"] = out["smoking_status"].replace("Unknown", smoking_mode)

    for column in _OUTLIER_COLUMNS:
        lower, upper = _iqr_bounds(out[column])
        out[column] = out[column].clip(lower, upper)

    return out


def encode(frame: pd.DataFrame) -> pd.DataFrame:
    """Apply step 6: label-encode binaries and one-hot-encode categoricals."""
    out = frame.copy()
    out["gender"] = out["gender"].map(SCHEMA.gender_map)
    out["ever_married"] = out["ever_married"].map(SCHEMA.married_map)
    return pd.get_dummies(out, columns=_ONE_HOT_COLUMNS, drop_first=False)


def build_feature_matrix(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series | None]:
    """Run the full pipeline and align columns to the model's feature order.

    Returns:
        A tuple ``(X, y)`` where ``X`` has exactly the 23 model features in
        the expected order and ``y`` is the target (or ``None`` if absent).
    """
    cleaned = clean(frame)
    with_categories = add_clinical_categories(cleaned)
    encoded = encode(with_categories)

    target = (
        encoded[SCHEMA.target].astype("int64")
        if SCHEMA.target in encoded.columns
        else None
    )
    features = encoded.reindex(columns=SCHEMA.feature_order, fill_value=0)
    logger.info("Built feature matrix: %d rows x %d features", *features.shape)
    return features, target
