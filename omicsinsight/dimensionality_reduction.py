"""Dimensionality reduction: PCA and optional UMAP."""

import logging
from typing import Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

logger = logging.getLogger("omicsinsight")


# ---------------------------------------------------------------------------
# PCA
# ---------------------------------------------------------------------------

def run_pca(
    df: pd.DataFrame, n_components: int = 10
) -> Tuple[pd.DataFrame, PCA]:
    """Fit PCA and return (scores_df, fitted_pca)."""
    n_components = min(n_components, df.shape[0], df.shape[1])
    pca = PCA(n_components=n_components)
    scores = pca.fit_transform(df.values)

    cols = [f"PC{i+1}" for i in range(n_components)]
    pca_df = pd.DataFrame(scores, index=df.index, columns=cols)

    logger.info(
        "PCA: %d components, explained variance (top 3): %s",
        n_components,
        np.round(pca.explained_variance_ratio_[:3], 4).tolist(),
    )
    return pca_df, pca


def plot_pca(
    pca_df: pd.DataFrame,
    pca: PCA,
    labels: Optional[pd.Series] = None,
    output_path: str = "pca_scatter.png",
    title: str = "PCA — OmicsInsight",
) -> str:
    """Save a PC1 vs PC2 scatter plot coloured by *labels*."""
    fig, ax = plt.subplots(figsize=(8, 6))

    ev1 = pca.explained_variance_ratio_[0] * 100
    ev2 = pca.explained_variance_ratio_[1] * 100

    if labels is not None:
        unique = sorted(labels.unique())
        cmap = matplotlib.colormaps["tab10"]
        for i, lab in enumerate(unique):
            mask = labels.values == lab
            ax.scatter(
                pca_df.iloc[:, 0][mask],
                pca_df.iloc[:, 1][mask],
                label=str(lab),
                color=cmap(i / max(len(unique) - 1, 1)),
                s=60, edgecolors="k", linewidth=0.5, alpha=0.85,
            )
        ax.legend(title="Group", fontsize=9, title_fontsize=10)
    else:
        ax.scatter(
            pca_df.iloc[:, 0], pca_df.iloc[:, 1],
            s=60, edgecolors="k", linewidth=0.5,
        )

    ax.set_xlabel(f"PC1 ({ev1:.1f}% variance)")
    ax.set_ylabel(f"PC2 ({ev2:.1f}% variance)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    logger.info("PCA plot saved: %s", output_path)
    return output_path


# ---------------------------------------------------------------------------
# UMAP (optional dependency)
# ---------------------------------------------------------------------------

def run_umap(
    df: pd.DataFrame,
    n_components: int = 2,
    random_state: int = 42,
) -> Optional[pd.DataFrame]:
    """Run UMAP if *umap-learn* is installed; return ``None`` otherwise."""
    try:
        from umap import UMAP  # type: ignore[import-untyped]
    except ImportError:
        logger.warning("umap-learn not installed — skipping UMAP.")
        return None

    reducer = UMAP(n_components=n_components, random_state=random_state)
    transformed = reducer.fit_transform(df.values)

    cols = [f"UMAP{i+1}" for i in range(n_components)]
    umap_df = pd.DataFrame(transformed, index=df.index, columns=cols)
    logger.info("UMAP: reduced to %d components", n_components)
    return umap_df


def plot_umap(
    umap_df: pd.DataFrame,
    labels: Optional[pd.Series] = None,
    output_path: str = "umap_scatter.png",
    title: str = "UMAP — OmicsInsight",
) -> str:
    """Save a UMAP scatter plot coloured by *labels*."""
    fig, ax = plt.subplots(figsize=(8, 6))

    if labels is not None:
        unique = sorted(labels.unique())
        cmap = matplotlib.colormaps["tab10"]
        for i, lab in enumerate(unique):
            mask = labels.values == lab
            ax.scatter(
                umap_df.iloc[:, 0][mask],
                umap_df.iloc[:, 1][mask],
                label=str(lab),
                color=cmap(i / max(len(unique) - 1, 1)),
                s=60, edgecolors="k", linewidth=0.5, alpha=0.85,
            )
        ax.legend(title="Group", fontsize=9, title_fontsize=10)
    else:
        ax.scatter(
            umap_df.iloc[:, 0], umap_df.iloc[:, 1],
            s=60, edgecolors="k", linewidth=0.5,
        )

    ax.set_xlabel("UMAP1")
    ax.set_ylabel("UMAP2")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    logger.info("UMAP plot saved: %s", output_path)
    return output_path
