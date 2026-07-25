"""Stage 05 — Evaluation.

Converted from ``TRAINING_MODEL.ipynb`` (evaluation cells).

Scores every saved model on the reconstructed held-out test set, writes
``models/metrics.json``, and saves a confusion-matrix figure per model.

Run:
    python pipeline/stage_05_evaluation.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from stroke_prediction.config import PATHS  # noqa: E402
from stroke_prediction.evaluation import evaluate_all  # noqa: E402
from stroke_prediction.logger import get_logger  # noqa: E402

logger = get_logger("stage_05_evaluation")


def _plot_confusion(name: str, matrix: list[list[int]]) -> None:
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        matrix, annot=True, fmt="d", cmap="Blues", cbar=False,
        xticklabels=["No stroke", "Stroke"],
        yticklabels=["No stroke", "Stroke"], ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion matrix — {name}")
    fig.tight_layout()
    fig.savefig(PATHS.figures_dir / f"cm_{name}.png", dpi=120)
    plt.close(fig)


def main() -> None:
    PATHS.figures_dir.mkdir(parents=True, exist_ok=True)

    # Step 1 — Compute metrics for every saved model variant.
    results = evaluate_all()

    # Step 2 — Save a confusion-matrix figure per model.
    for name, metrics in results.items():
        _plot_confusion(name, metrics["confusion_matrix"])
    logger.info("Saved confusion-matrix figures to %s", PATHS.figures_dir)


if __name__ == "__main__":
    main()
