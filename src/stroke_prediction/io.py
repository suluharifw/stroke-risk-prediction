"""Small, typed helpers for reading and writing artifacts."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib


def save_joblib(obj: Any, path: Path) -> None:
    """Persist a Python object (model, scaler) to ``path``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(obj, path)


def load_joblib(path: Path) -> Any:
    """Load a joblib artifact, raising a clear error if it is missing."""
    if not path.exists():
        raise FileNotFoundError(
            f"Artifact not found: {path}. Run the training pipeline first "
            f"(`python pipeline/stage_04_model_training.py`)."
        )
    return joblib.load(path)


def save_json(data: dict, path: Path) -> None:
    """Write ``data`` to ``path`` as pretty-printed JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2))
