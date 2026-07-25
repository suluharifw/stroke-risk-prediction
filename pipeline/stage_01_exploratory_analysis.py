"""Stage 01 — Exploratory Data Analysis.

Converted from ``STROKE_PREDICTION.ipynb`` (exploration section).

This stage does not modify data. It loads the raw dataset, reports its
structure and quality, and saves distribution figures to ``reports/figures``
so the findings that motivate later preprocessing are reproducible.

Run:
    python pipeline/stage_01_exploratory_analysis.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless backend for script execution
import matplotlib.pyplot as plt
import seaborn as sns

# Allow running as a plain script without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from stroke_prediction.config import PATHS  # noqa: E402
from stroke_prediction.data import load_raw  # noqa: E402
from stroke_prediction.logger import get_logger  # noqa: E402

logger = get_logger("stage_01_eda")


def main() -> None:
    PATHS.figures_dir.mkdir(parents=True, exist_ok=True)
    df = load_raw()

    # Step 1 — Structure: dtypes and non-null counts.
    logger.info("Columns: %s", list(df.columns))
    logger.info("Shape: %d rows x %d columns", *df.shape)

    # Step 2 — Missing values. Only 'bmi' has NaNs; this justifies imputation.
    missing = df.isna().sum()
    logger.info("Missing values:\n%s", missing[missing > 0].to_string() or "none")

    # Step 3 — Target balance. The dataset is strongly imbalanced (~4.9% stroke),
    # which is why later stages compare resampling strategies.
    balance = df["stroke"].value_counts(normalize=True).mul(100).round(2)
    logger.info("Target balance (%%):\n%s", balance.to_string())

    # Step 4 — Class distribution figure.
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    df["stroke"].value_counts().plot.pie(
        explode=[0, 0.1], autopct="%1.1f%%", ax=axes[0],
        labels=["No stroke", "Stroke"], colors=["#0E7C86", "#E4572E"],
    )
    axes[0].set_ylabel("")
    axes[0].set_title("Stroke class distribution")
    sns.countplot(x="stroke", data=df, ax=axes[1], palette=["#0E7C86", "#E4572E"])
    axes[1].set_title("Stroke counts")
    fig.tight_layout()
    fig.savefig(PATHS.figures_dir / "01_class_distribution.png", dpi=120)
    plt.close(fig)

    # Step 5 — Continuous-feature distributions (age, glucose, BMI).
    fig, axes = plt.subplots(3, 1, figsize=(7, 9))
    for ax, col, title in zip(
        axes, ["age", "avg_glucose_level", "bmi"],
        ["Age", "Average glucose level", "BMI"],
    ):
        sns.histplot(df[col], bins=25, kde=True, ax=ax, color="#0E7C86")
        ax.set_title(f"Distribution of {title}")
    fig.tight_layout()
    fig.savefig(PATHS.figures_dir / "02_continuous_distributions.png", dpi=120)
    plt.close(fig)

    logger.info("Saved EDA figures to %s", PATHS.figures_dir)


if __name__ == "__main__":
    main()
