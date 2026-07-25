"""Model evaluation: metrics and confusion matrices on the held-out test set."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from .config import PATHS, TRAINING
from .data import load_raw
from .io import load_joblib, save_json
from .logger import get_logger
from .preprocessing import build_feature_matrix

logger = get_logger(__name__)


def compute_metrics(model, x_test: np.ndarray, y_test: pd.Series) -> dict:
    """Return a dict of classification metrics for a fitted model."""
    y_pred = model.predict(x_test)
    y_proba = model.predict_proba(x_test)[:, 1]
    return {
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y_test, y_pred, zero_division=0), 4),
        "f1": round(f1_score(y_test, y_pred, zero_division=0), 4),
        "roc_auc": round(roc_auc_score(y_test, y_proba), 4),
        "pr_auc": round(average_precision_score(y_test, y_proba), 4),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
    }


def reconstruct_test_set() -> tuple[np.ndarray, pd.Series]:
    """Rebuild the exact held-out test set using the *saved* scaler.

    Mirrors the training split (same features, same scaler, same stratified
    70/30 split with the configured seed) so metrics reflect the shipped models.
    """
    features, target = build_feature_matrix(load_raw())
    scaler = load_joblib(PATHS.scaler)
    scaled = scaler.transform(features)
    _, x_test, _, y_test = train_test_split(
        scaled,
        target,
        test_size=TRAINING.test_size,
        random_state=TRAINING.random_state,
        stratify=target,
    )
    return x_test, y_test


def evaluate_all() -> dict:
    """Evaluate every saved model variant and write ``models/metrics.json``."""
    x_test, y_test = reconstruct_test_set()

    results: dict[str, dict] = {}
    for strategy in TRAINING.resampling_variants:
        path = PATHS.model(strategy)
        if not path.exists():
            logger.warning("Skipping '%s' (artifact missing: %s)", strategy, path)
            continue
        metrics = compute_metrics(load_joblib(path), x_test, y_test)
        results[strategy] = metrics
        logger.info(
            "%-10s | recall=%.3f roc_auc=%.3f f1=%.3f",
            strategy, metrics["recall"], metrics["roc_auc"], metrics["f1"],
        )

    save_json(results, PATHS.metrics)
    logger.info("Saved metrics -> %s", PATHS.metrics)
    return results
