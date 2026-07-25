"""Stage 04 — Model Training.

Converted from ``TRAINING_MODEL.ipynb``.

Scales features, makes a stratified 70/30 split, then trains Logistic
Regression under three imbalance strategies (BorderlineSMOTE, RandomUnderSampler,
SMOTE-ENN) and persists each model plus the shared scaler to ``models/``.

Run:
    python pipeline/stage_04_model_training.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from stroke_prediction.data import load_raw  # noqa: E402
from stroke_prediction.logger import get_logger  # noqa: E402
from stroke_prediction.preprocessing import build_feature_matrix  # noqa: E402
from stroke_prediction.training import train_all  # noqa: E402

logger = get_logger("stage_04_training")


def main() -> None:
    # Step 1 — Build the model matrix from raw data.
    features, target = build_feature_matrix(load_raw())

    # Step 2 — Fit scaler, split, resample, train, and persist all variants.
    train_all(features, target)
    logger.info("Training complete. Artifacts written to models/.")


if __name__ == "__main__":
    main()
