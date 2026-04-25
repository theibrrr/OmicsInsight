"""Preprocessing functions for transcriptomics count data."""

import logging
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger("omicsinsight")


def transpose_to_samples(counts: pd.DataFrame) -> pd.DataFrame:
    """Transpose a genes × samples matrix to samples × genes."""
    return counts.T


def log_transform(df: pd.DataFrame, pseudo_count: float = 1.0) -> pd.DataFrame:
    """Apply log2(x + pseudo_count) element-wise."""
    logger.info("Applying log2(x + %s) transformation", pseudo_count)
    return np.log2(df + pseudo_count)


def filter_low_counts(
    df: pd.DataFrame, min_total_count: int = 10
) -> pd.DataFrame:
    """Remove features whose total count across samples is below *min_total_count*.

    Expects a **samples × genes** DataFrame.
    """
    totals = df.sum(axis=0)
    keep = totals >= min_total_count
    n_removed = int((~keep).sum())
    logger.info(
        "Low-count filter: removed %d, kept %d of %d features",
        n_removed, int(keep.sum()), len(keep),
    )
    return df.loc[:, keep]


def filter_low_variance(
    df: pd.DataFrame, min_variance: float = 0.0
) -> pd.DataFrame:
    """Remove features with variance at or below *min_variance*."""
    variances = df.var(axis=0)
    keep = variances > min_variance
    n_removed = int((~keep).sum())
    logger.info(
        "Low-variance filter: removed %d, kept %d of %d features",
        n_removed, int(keep.sum()), len(keep),
    )
    return df.loc[:, keep]


def select_top_features(df: pd.DataFrame, max_features: int) -> pd.DataFrame:
    """Keep the *max_features* features with the highest variance."""
    if max_features >= df.shape[1]:
        logger.info(
            "max_features=%d >= available features (%d) — keeping all.",
            max_features, df.shape[1],
        )
        return df
    top_idx = df.var(axis=0).nlargest(max_features).index
    logger.info("Selected top %d features by variance", max_features)
    return df[top_idx]


def scale_features(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, StandardScaler]:
    """Standardise each feature to zero-mean, unit-variance.

    Returns (scaled_DataFrame, fitted_scaler).
    """
    scaler = StandardScaler()
    scaled = scaler.fit_transform(df)
    scaled_df = pd.DataFrame(scaled, index=df.index, columns=df.columns)
    logger.info("Applied StandardScaler to features")
    return scaled_df, scaler


def preprocess_pipeline(
    counts: pd.DataFrame,
    *,
    log: bool = True,
    min_total_count: int = 10,
    min_variance: float = 0.0,
    max_features: int = 500,
) -> Tuple[pd.DataFrame, pd.DataFrame, StandardScaler]:
    """Run the complete preprocessing chain.

    Parameters
    ----------
    counts : DataFrame
        Genes × samples raw count matrix.

    Returns
    -------
    scaled : DataFrame
        Samples × features, StandardScaler-transformed.
    unscaled : DataFrame
        Samples × features, log-transformed but **not** scaled —
        used for LOO-CV where scaling is done inside each fold.
    scaler : StandardScaler
        Fitted scaler (for later inverse transforms if needed).
    """
    df = transpose_to_samples(counts)
    logger.info("Transposed to %d samples × %d features", *df.shape)

    df = filter_low_counts(df, min_total_count=min_total_count)

    if log:
        df = log_transform(df)

    df = filter_low_variance(df, min_variance=min_variance)
    df = select_top_features(df, max_features=max_features)

    unscaled = df.copy()
    scaled, scaler = scale_features(df)
    return scaled, unscaled, scaler
