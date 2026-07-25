"""Project configuration.

All paths, feature definitions, category rules, and training hyper-parameters
live here so that the training pipeline and the serving layer stay in sync.
Values are derived from the original analysis notebooks and the fitted scaler.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

# Repository root: src/stroke_prediction/config.py -> parents[2]
ROOT_DIR: Path = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Paths:
    """Canonical filesystem locations for data and artifacts."""

    root: Path = ROOT_DIR
    data: Path = ROOT_DIR / "data"
    raw_data: Path = ROOT_DIR / "data" / "raw" / "healthcare-dataset-stroke-data.csv"
    processed_dir: Path = ROOT_DIR / "data" / "processed"
    processed_data: Path = ROOT_DIR / "data" / "processed" / "features.parquet"

    models_dir: Path = ROOT_DIR / "models"
    scaler: Path = ROOT_DIR / "models" / "scaler.joblib"
    metrics: Path = ROOT_DIR / "models" / "metrics.json"
    reports_dir: Path = ROOT_DIR / "reports"
    figures_dir: Path = ROOT_DIR / "reports" / "figures"

    def model(self, name: str) -> Path:
        """Path to a named model artifact (e.g. 'smote')."""
        return self.models_dir / f"logreg_model_{name}.joblib"


@dataclass(frozen=True)
class Schema:
    """Column names and the exact feature order the models expect."""

    target: str = "stroke"
    drop_columns: tuple[str, ...] = ("id", "Residence_type")

    # The 23-feature order the fitted scaler and models were trained on.
    # This ordering is load-bearing: do not change it without retraining.
    feature_order: tuple[str, ...] = (
        "gender",
        "age",
        "hypertension",
        "heart_disease",
        "ever_married",
        "avg_glucose_level",
        "bmi",
        "work_type_Govt_job",
        "work_type_Never_worked",
        "work_type_Private",
        "work_type_Self-employed",
        "work_type_children",
        "smoking_status_formerly smoked",
        "smoking_status_never smoked",
        "smoking_status_smokes",
        "Weight_Category_Normal",
        "Weight_Category_Obesity",
        "Weight_Category_Overweight",
        "Weight_Category_Underweight",
        "Glucose_Category_Hiperglikemia",
        "Glucose_Category_Hipoglikemia",
        "Glucose_Category_Normal",
        "Glucose_Category_prediabetic",
    )

    # Label encodings (alphabetical, matching sklearn LabelEncoder in the notebook).
    gender_map: dict = field(default_factory=lambda: {"Female": 0, "Male": 1})
    married_map: dict = field(default_factory=lambda: {"No": 0, "Yes": 1})


@dataclass(frozen=True)
class TrainingConfig:
    """Hyper-parameters and split settings for reproducible training."""

    random_state: int = 42
    test_size: float = 0.30
    # The model served in production and the resampling variants to benchmark.
    production_model: str = "smote"
    resampling_variants: tuple[str, ...] = ("smote", "rus", "smote_enn")


PATHS = Paths()
SCHEMA = Schema()
TRAINING = TrainingConfig()
