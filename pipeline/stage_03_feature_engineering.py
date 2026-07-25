"""Stage 03 — Feature Engineering & Encoding.

Converted from ``STROKE_PREDICTION.ipynb`` (feature-engineering section).

Derives clinical categories, encodes categoricals, and produces the final
23-feature model matrix aligned to the order the models expect. Saves the
matrix to ``data/processed/features.parquet``.

Run:
    python pipeline/stage_03_feature_engineering.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from stroke_prediction.config import PATHS  # noqa: E402
from stroke_prediction.data import load_raw  # noqa: E402
from stroke_prediction.logger import get_logger  # noqa: E402
from stroke_prediction.preprocessing import build_feature_matrix  # noqa: E402

logger = get_logger("stage_03_features")


def main() -> None:
    df = load_raw()

    # Step 1 — Clean + derive Weight_Category / Glucose_Category, then encode
    # and reindex to the canonical feature order (all inside build_feature_matrix).
    features, target = build_feature_matrix(df)

    logger.info("Feature matrix: %d rows x %d features", *features.shape)
    logger.info("Positive rate: %.2f%%", 100 * target.mean())

    PATHS.processed_dir.mkdir(parents=True, exist_ok=True)
    combined = features.copy()
    combined["stroke"] = target.values
    try:
        combined.to_parquet(PATHS.processed_data, index=False)
        out = PATHS.processed_data
    except Exception:  # parquet engine not installed -> fall back to CSV
        out = PATHS.processed_data.with_suffix(".csv")
        combined.to_csv(out, index=False)
    logger.info("Saved feature matrix -> %s", out)


if __name__ == "__main__":
    main()
