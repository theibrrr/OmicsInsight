"""Clustering analysis: KMeans, Agglomerative, evaluation, and heatmap."""

import logging
from typing import Dict, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score

logger = logging.getLogger("omicsinsight")


def run_kmeans(
    df: pd.DataFrame, n_clusters: int = 3, random_state: int = 42
) -> np.ndarray:
    """Fit KMeans and return cluster labels."""
    km = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    labels = km.fit_predict(df.values)
    logger.info("KMeans: %d clusters", n_clusters)
    return labels


def run_agglomerative(
    df: pd.DataFrame, n_clusters: int = 3
) -> np.ndarray:
    """Fit Agglomerative clustering (Ward linkage) and return labels."""
    agg = AgglomerativeClustering(n_clusters=n_clusters, linkage="ward")
    labels = agg.fit_predict(df.values)
    logger.info("Agglomerative: %d clusters (Ward)", n_clusters)
    return labels


def evaluate_clustering(
    labels: np.ndarray,
    data: pd.DataFrame,
    true_labels: Optional[np.ndarray] = None,
) -> Dict:
    """Compute silhouette score and (optionally) Adjusted Rand Index."""
    results: Dict = {}

    n_unique = len(set(labels))
    if 1 < n_unique < len(labels):
        results["silhouette_score"] = round(
            float(silhouette_score(data.values, labels)), 4
        )

    if true_labels is not None:
        results["adjusted_rand_index"] = round(
            float(adjusted_rand_score(true_labels, labels)), 4
        )

    unique, counts = np.unique(labels, return_counts=True)
    results["cluster_sizes"] = {int(k): int(v) for k, v in zip(unique, counts)}
    return results


def plot_cluster_heatmap(
    df: pd.DataFrame,
    cluster_labels: np.ndarray,
    output_path: str = "cluster_heatmap.png",
    top_n: int = 50,
    title: str = "Top Variable Features — Heatmap",
) -> str:
    """Heatmap of the *top_n* most-variable features, rows ordered by cluster."""
    variances = df.var(axis=0)
    top_features = variances.nlargest(min(top_n, len(variances))).index
    subset = df[top_features].copy()

    order = np.argsort(cluster_labels)
    subset = subset.iloc[order]

    fig, ax = plt.subplots(figsize=(12, 8))
    im = ax.imshow(
        subset.values, aspect="auto", cmap="RdBu_r", interpolation="nearest"
    )
    ax.set_xlabel("Features (top variable)")
    ax.set_ylabel("Samples (ordered by cluster)")
    ax.set_title(title)

    cbar = fig.colorbar(im, ax=ax, fraction=0.02, pad=0.02)
    cbar.set_label("Scaled expression")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    logger.info("Cluster heatmap saved: %s", output_path)
    return output_path
