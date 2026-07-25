"""Stage 02 — Data Cleaning.

Converted from ``STROKE_PREDICTION.ipynb`` (data-preprocessing section).

Applies the cleaning steps that the EDA motivated and writes an intermediate
cleaned dataset to ``data/processed/cleaned.csv``. Each step is intentionally
explicit so the transformation is auditable.

Run:
    python pipeline/stage_02_preprocessing.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from stroke_prediction.config import PATHS  # noqa: E402
from stroke_prediction.data import load_raw  # noqa: E402
from stroke_prediction.logger import get_logger  # noqa: E402
from stroke_prediction.preprocessing import clean  # noqa: E402

logger = get_logger("stage_02_preprocessing")


def main() -> None:
    df = load_raw()

    # Step 1-4 (impute bmi, drop id/Residence_type, fix 'Unknown' smoking,
    # cap outliers) are implemented in stroke_prediction.preprocessing.clean.
    cleaned = clean(df)

    logger.info("Cleaned shape: %d rows x %d columns", *cleaned.shape)
    logger.info("Remaining NaNs: %d", int(cleaned.isna().sum().sum()))

    PATHS.processed_dir.mkdir(parents=True, exist_ok=True)
    out = PATHS.processed_dir / "cleaned.csv"
    cleaned.to_csv(out, index=False)
    logger.info("Saved cleaned dataset -> %s", out)


if __name__ == "__main__":
    main()
