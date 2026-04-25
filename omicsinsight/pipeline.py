"""Main analysis pipeline orchestrator — ties every module together."""

import logging
from typing import Any, Dict

import pandas as pd
from sklearn.preprocessing import LabelEncoder

from omicsinsight.clustering import (
    evaluate_clustering,
    plot_cluster_heatmap,
    run_agglomerative,
    run_kmeans,
)
from omicsinsight.config import PipelineConfig
from omicsinsight.dataset_parser import align_data, parse_count_matrix, parse_metadata
from omicsinsight.dimensionality_reduction import plot_pca, plot_umap, run_pca, run_umap
from omicsinsight.feature_ranking import (
    combine_rankings,
    get_top_features,
    rank_by_logreg_coef,
    rank_by_rf_importance,
    rank_by_variance,
)
from omicsinsight.io import ensure_dir, save_csv, save_json, save_model, save_text
from omicsinsight.modeling import fit_final_models, run_loo_classification
from omicsinsight.preprocessing import (
    filter_low_counts,
    filter_low_variance,
    log_transform,
    scale_features,
    select_top_features,
    transpose_to_samples,
)
from omicsinsight.reporting import build_report_md, build_summary
from omicsinsight.validation import (
    validate_alignment,
    validate_counts,
    validate_file_exists,
    validate_metadata,
)

logger = logging.getLogger("omicsinsight")


def run_pipeline(config: PipelineConfig) -> Dict[str, Any]:
    """Execute the complete OmicsInsight analysis pipeline.

    Parameters
    ----------
    config : PipelineConfig
        All run settings (paths, parameters, flags).

    Returns
    -------
    dict
        Full analysis summary (also written as JSON to ``output_dir``).
    """
    warnings_list: list = []

    # ── Setup ──────────────────────────────────────────────────────────
    output_dir = ensure_dir(config.output_dir)
    logger.info("Output directory: %s", output_dir)

    # ── Step 1  Validate input files ──────────────────────────────────
    logger.info("Step 1/10: Validating input files")
    validate_file_exists(config.counts_path, "Count data file")
    validate_file_exists(config.metadata_path, "Metadata file")

    # ── Step 2  Parse files ───────────────────────────────────────────
    logger.info("Step 2/10: Parsing dataset files")
    counts_raw = parse_count_matrix(config.counts_path)
    metadata = parse_metadata(config.metadata_path, sample_id_col=config.sample_id_column)
    n_genes_raw = counts_raw.shape[0]

    # ── Step 3  Align samples ─────────────────────────────────────────
    logger.info("Step 3/10: Aligning count matrix and metadata")
    counts, metadata = align_data(
        counts_raw, metadata, sample_id_col=config.sample_id_column,
    )

    # ── Step 4  Validate aligned data ─────────────────────────────────
    logger.info("Step 4/10: Validating data integrity")
    warnings_list.extend(validate_counts(counts))

    has_target = config.target_column in metadata.columns
    warnings_list.extend(
        validate_metadata(metadata, config.target_column, config.sample_id_column)
    )
    validate_alignment(counts, metadata, config.sample_id_column)

    save_csv(metadata, str(output_dir / "metadata.csv"), index=False)

    # ── Step 5  Preprocess ────────────────────────────────────────────
    logger.info("Step 5/10: Preprocessing")
    df = transpose_to_samples(counts)

    df = filter_low_counts(df, min_total_count=config.min_total_count)
    n_after_count_filter = df.shape[1]

    if config.log_transform:
        df = log_transform(df)

    df = filter_low_variance(df, min_variance=config.min_variance)
    n_after_variance_filter = df.shape[1]

    df = select_top_features(df, max_features=config.max_features)
    n_final_features = df.shape[1]

    unscaled = df.copy()
    scaled, scaler = scale_features(df)

    preprocessing_info = {
        "log_transform": config.log_transform,
        "min_total_count": config.min_total_count,
        "n_after_count_filter": n_after_count_filter,
        "min_variance": config.min_variance,
        "n_after_variance_filter": n_after_variance_filter,
        "max_features": config.max_features,
        "n_final_features": n_final_features,
    }

    # ── Step 6  Dimensionality reduction ──────────────────────────────
    logger.info("Step 6/10: Dimensionality reduction")
    labels_series = metadata[config.target_column] if has_target else None

    pca_df, pca_model = run_pca(scaled, n_components=config.n_pca_components)
    plot_pca(
        pca_df, pca_model,
        labels=labels_series,
        output_path=str(output_dir / "pca_scatter.png"),
    )
    save_csv(pca_df, str(output_dir / "pca_components.csv"))

    if config.umap_enabled:
        umap_df = run_umap(scaled, random_state=config.random_state)
        if umap_df is not None:
            plot_umap(
                umap_df,
                labels=labels_series,
                output_path=str(output_dir / "umap_scatter.png"),
            )
            save_csv(umap_df, str(output_dir / "umap_components.csv"))

    # ── Step 7  Clustering ────────────────────────────────────────────
    logger.info("Step 7/10: Clustering")
    true_labels = None
    if has_target:
        le_clust = LabelEncoder()
        true_labels = le_clust.fit_transform(metadata[config.target_column])

    km_labels = run_kmeans(
        scaled, n_clusters=config.n_clusters, random_state=config.random_state,
    )
    km_eval = evaluate_clustering(km_labels, scaled, true_labels)

    agg_labels = run_agglomerative(scaled, n_clusters=config.n_clusters)
    agg_eval = evaluate_clustering(agg_labels, scaled, true_labels)

    clustering_results = {"KMeans": km_eval, "Agglomerative": agg_eval}

    cluster_df = pd.DataFrame({
        "sample_id": metadata[config.sample_id_column].values,
        "kmeans_cluster": km_labels,
        "agglomerative_cluster": agg_labels,
    })
    if has_target:
        cluster_df["true_label"] = metadata[config.target_column].values
    save_csv(cluster_df, str(output_dir / "cluster_labels.csv"), index=False)

    plot_cluster_heatmap(
        scaled, km_labels,
        output_path=str(output_dir / "cluster_heatmap.png"),
    )

    # ── Step 8  Classification ────────────────────────────────────────
    logger.info("Step 8/10: Classification")
    classification_results = None
    final_models = None

    if has_target:
        target = metadata[config.target_column]
        n_classes = target.nunique()
        min_class_size = int(target.value_counts().min())

        if n_classes >= 2 and min_class_size >= 2:
            classification_results = run_loo_classification(
                unscaled, target, random_state=config.random_state,
            )
            final_models, final_scaler = fit_final_models(
                unscaled, target, random_state=config.random_state,
            )
            for name, model in final_models.items():
                if not name.startswith("_"):
                    save_model(model, str(output_dir / f"model_{name}.joblib"))
            save_model(final_scaler, str(output_dir / "scaler.joblib"))
        else:
            msg = (
                f"Classification skipped: {n_classes} classes, "
                f"min class size {min_class_size}."
            )
            logger.warning(msg)
            warnings_list.append(msg)
    else:
        msg = f"Target column '{config.target_column}' not found — skipping classification."
        logger.warning(msg)
        warnings_list.append(msg)

    # ── Step 9  Feature ranking ───────────────────────────────────────
    logger.info("Step 9/10: Feature ranking")
    feature_names = list(unscaled.columns)
    var_ranking = rank_by_variance(unscaled)

    lr_ranking = None
    rf_ranking = None
    if final_models:
        if "LogisticRegression" in final_models:
            lr_ranking = rank_by_logreg_coef(
                final_models["LogisticRegression"], feature_names,
            )
        if "RandomForest" in final_models:
            rf_ranking = rank_by_rf_importance(
                final_models["RandomForest"], feature_names,
            )

    combined_ranking = combine_rankings(var_ranking, lr_ranking, rf_ranking)
    save_csv(combined_ranking, str(output_dir / "ranked_features.csv"), index=False)

    top_features = get_top_features(combined_ranking, top_n=20)
    save_json(top_features, str(output_dir / "top_features.json"))

    # ── Step 10  Reporting ────────────────────────────────────────────
    logger.info("Step 10/10: Generating report")
    summary = build_summary(
        config=config.to_dict(),
        n_samples=len(metadata),
        n_genes_raw=n_genes_raw,
        n_features_final=n_final_features,
        metadata_columns=list(metadata.columns),
        preprocessing_info=preprocessing_info,
        pca_variance=pca_model.explained_variance_ratio_.tolist(),
        clustering_results=clustering_results,
        classification_results=classification_results,
        top_features=top_features,
        warnings=warnings_list,
    )
    save_json(summary, str(output_dir / "analysis_summary.json"))

    report_md = build_report_md(summary)
    save_text(report_md, str(output_dir / "report.md"))

    logger.info("Pipeline completed. Outputs saved to %s", output_dir)
    return summary
