"""Input validation for OmicsInsight pipeline."""

import logging
from pathlib import Path
from typing import List

import pandas as pd

logger = logging.getLogger("omicsinsight")


class ValidationError(Exception):
    """Raised when a data-integrity check fails."""


def validate_file_exists(path: str, label: str = "File") -> None:
    """Assert that *path* points to a non-empty, readable file."""
    p = Path(path)
    if not p.exists():
        raise ValidationError(f"{label} not found: {path}")
    if not p.is_file():
        raise ValidationError(f"{label} is not a file: {path}")
    if p.stat().st_size == 0:
        raise ValidationError(f"{label} is empty: {path}")


def validate_counts(counts: pd.DataFrame) -> List[str]:
    """Validate the count matrix.  Returns a list of warning strings."""
    warnings: List[str] = []

    if counts.shape[0] == 0:
        raise ValidationError("Count matrix has no genes (rows).")
    if counts.shape[1] == 0:
        raise ValidationError("Count matrix has no samples (columns).")

    dup_genes = counts.index[counts.index.duplicated()]
    if len(dup_genes) > 0:
        raise ValidationError(f"Duplicate gene IDs found: {list(dup_genes[:10])}")

    dup_samples = counts.columns[counts.columns.duplicated()]
    if len(dup_samples) > 0:
        raise ValidationError(f"Duplicate sample IDs found: {list(dup_samples[:10])}")

    if (counts < 0).any().any():
        n_neg = int((counts < 0).sum().sum())
        warnings.append(f"Found {n_neg} negative values in count matrix.")

    zero_genes = int((counts.sum(axis=1) == 0).sum())
    if zero_genes > 0:
        warnings.append(f"{zero_genes} genes have zero counts across all samples.")

    logger.info("Count matrix validation passed. Warnings: %d", len(warnings))
    for w in warnings:
        logger.warning(w)
    return warnings


def validate_metadata(
    metadata: pd.DataFrame,
    target_column: str,
    sample_id_column: str,
) -> List[str]:
    """Validate metadata.  Returns a list of warning strings."""
    warnings: List[str] = []

    if metadata.shape[0] == 0:
        raise ValidationError("Metadata has no samples (rows).")

    if sample_id_column not in metadata.columns:
        raise ValidationError(
            f"Sample ID column '{sample_id_column}' not found. "
            f"Available: {list(metadata.columns)}"
        )

    dup_ids = metadata[sample_id_column][metadata[sample_id_column].duplicated()]
    if len(dup_ids) > 0:
        raise ValidationError(f"Duplicate sample IDs: {list(dup_ids[:10])}")

    if target_column not in metadata.columns:
        warnings.append(
            f"Target column '{target_column}' not found. "
            f"Classification will be skipped. "
            f"Available: {list(metadata.columns)}"
        )
    else:
        dist = metadata[target_column].value_counts()
        logger.info("Target distribution:\n%s", dist.to_string())
        if dist.min() < 2:
            warnings.append(
                f"Some classes in '{target_column}' have fewer than 2 samples."
            )

    for col in [sample_id_column, target_column]:
        if col in metadata.columns:
            n_miss = int(metadata[col].isna().sum())
            if n_miss > 0:
                warnings.append(f"Column '{col}' has {n_miss} missing values.")

    logger.info("Metadata validation passed. Warnings: %d", len(warnings))
    for w in warnings:
        logger.warning(w)
    return warnings


def validate_alignment(
    counts: pd.DataFrame,
    metadata: pd.DataFrame,
    sample_id_column: str,
) -> None:
    """Verify that count matrix columns match metadata sample IDs in order."""
    count_ids = list(counts.columns)
    meta_ids = list(metadata[sample_id_column])
    if count_ids != meta_ids:
        raise ValidationError(
            "Count matrix columns and metadata sample IDs are not aligned."
        )
    logger.info("Alignment validation passed.")
