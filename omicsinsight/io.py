"""File I/O utilities for OmicsInsight."""

import json
import logging
from pathlib import Path
from typing import Any, Dict

import joblib
import pandas as pd

logger = logging.getLogger("omicsinsight")


def ensure_dir(path: str) -> Path:
    """Create directory (and parents) if it does not exist."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_csv(df: pd.DataFrame, path: str, **kwargs: Any) -> None:
    """Save a DataFrame to CSV."""
    df.to_csv(path, **kwargs)
    logger.info("Saved CSV: %s", path)


def save_json(data: Dict[str, Any], path: str) -> None:
    """Save a dictionary as JSON."""
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, default=str)
    logger.info("Saved JSON: %s", path)


def load_json(path: str) -> Dict[str, Any]:
    """Load a JSON file into a dictionary."""
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def save_model(model: Any, path: str) -> None:
    """Persist a model/object with joblib."""
    joblib.dump(model, path)
    logger.info("Saved model: %s", path)


def save_text(text: str, path: str) -> None:
    """Write plain text to a file."""
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text)
    logger.info("Saved text: %s", path)
