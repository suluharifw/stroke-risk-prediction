"""Model training with configurable imbalance-handling strategies.

Trains a :class:`~sklearn.linear_model.LogisticRegression` under three
resampling strategies and persists each model plus the shared scaler. The
dataset is highly imbalanced (~4.9% positive), so resampling materially
changes recall.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from .config import PATHS, TRAINING
from .io import save_joblib
from .logger import get_logger

logger = get_logger(__name__)


def _resampler(strategy: str):
    """Return a fresh resampler instance for the given strategy name."""
    if strategy == "smote":
        from imblearn.over_sampling import BorderlineSMOTE

        return BorderlineSMOTE(random_state=TRAINING.random_state)
    if strategy == "rus":
        from imblearn.under_sampling import RandomUnderSampler

        return RandomUnderSampler(random_state=TRAINING.random_state)
    if strategy == "smote_enn":
        from imblearn.combine import SMOTEENN

        return SMOTEENN(random_state=TRAINING.random_state)
    raise ValueError(f"Unknown resampling strategy: {strategy!r}")


@dataclass
class SplitData:
    """Scaled train/test arrays produced by :func:`prepare_split`."""

    x_train: np.ndarray
    x_test: np.ndarray
    y_train: pd.Series
    y_test: pd.Series


def prepare_split(features: pd.DataFrame, target: pd.Series) -> tuple[StandardScaler, SplitData]:
    """Fit the scaler on all features, then make a stratified train/test split.

    The scaler is fit before splitting to reproduce the original notebook; the
    fitted scaler is what the serving layer loads, so this keeps them aligned.
    """
    scaler = StandardScaler().fit(features)
    scaled = scaler.transform(features)
    x_train, x_test, y_train, y_test = train_test_split(
        scaled,
        target,
        test_size=TRAINING.test_size,
        random_state=TRAINING.random_state,
        stratify=target,
    )
    return scaler, SplitData(x_train, x_test, y_train, y_test)


def train_variant(strategy: str, split: SplitData) -> LogisticRegression:
    """Resample the training set with ``strategy`` and fit a logistic model."""
    x_res, y_res = _resampler(strategy).fit_resample(split.x_train, split.y_train)
    model = LogisticRegression(random_state=TRAINING.random_state, max_iter=1000)
    model.fit(x_res, y_res)
    logger.info("Trained '%s' model on %d resampled rows", strategy, len(y_res))
    return model


def train_all(features: pd.DataFrame, target: pd.Series) -> SplitData:
    """Train and persist every configured variant plus the scaler.

    Returns the split so callers (e.g. evaluation) can reuse the held-out set.
    """
    scaler, split = prepare_split(features, target)
    save_joblib(scaler, PATHS.scaler)
    logger.info("Saved scaler -> %s", PATHS.scaler)

    for strategy in TRAINING.resampling_variants:
        model = train_variant(strategy, split)
        save_joblib(model, PATHS.model(strategy))
        logger.info("Saved '%s' model -> %s", strategy, PATHS.model(strategy))

    return split
