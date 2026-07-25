"""Dataset loading."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from .config import PATHS
from .logger import get_logger

logger = get_logger(__name__)


def load_raw(path: Path = PATHS.raw_data) -> pd.DataFrame:
    """Load the raw stroke dataset from CSV."""
    logger.info("Loading raw dataset: %s", path)
    frame = pd.read_csv(path)
    logger.info("Loaded %d rows x %d columns", *frame.shape)
    return frame
