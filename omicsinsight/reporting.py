"""Report generation: JSON summary and Markdown report."""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger("omicsinsight")


def build_summary(
    *,
    config: Dict,
    n_samples: int,
    n_genes_raw: int,
    n_features_final: int,
    metadata_columns: List[str],
    preprocessing_info: Dict,
    pca_variance: List[float],
    clustering_results: Dict,
    classification_results: Optional[Dict],
    top_features: Dict,
    warnings: List[str],
) -> Dict[str, Any]:
    """Assemble the machine-readable analysis summary."""
    return {
        "project": "OmicsInsight",
        "timestamp": datetime.now().isoformat(),
        "config": config,
        "dataset": {
            "n_samples": n_samples,
            "n_genes_raw": n_genes_raw,
            "n_features_after_preprocessing": n_features_final,
            "metadata_columns": metadata_columns,
        },
        "preprocessing": preprocessing_info,
        "pca": {
            "explained_variance_ratio": [round(v, 4) for v in pca_variance],
            "cumulative_variance": [
                round(sum(pca_variance[: i + 1]), 4)
                for i in range(len(pca_variance))
            ],
        },
        "clustering": clustering_results,
        "classification": classification_results,
        "top_features": top_features,
        "warnings": warnings,
    }


def build_report_md(summary: Dict[str, Any]) -> str:
    """Generate a human-readable Markdown analysis report."""
    s = summary
    L: List[str] = []  # noqa: N806 — intentional short alias

    L.append("# OmicsInsight — Analysis Report\n")
    L.append(f"**Generated:** {s['timestamp']}\n")

    # ---- Dataset overview ----
    L.append("## Dataset Overview\n")
    L.append(f"- **Samples:** {s['dataset']['n_samples']}")
    L.append(f"- **Genes (raw):** {s['dataset']['n_genes_raw']}")
    L.append(f"- **Features (after preprocessing):** {s['dataset']['n_features_after_preprocessing']}")
    L.append(f"- **Metadata columns:** {', '.join(s['dataset']['metadata_columns'])}")
    L.append(f"- **Count data file:** `{s['config'].get('counts_path', 'N/A')}`")
    L.append(f"- **Metadata file:** `{s['config'].get('metadata_path', 'N/A')}`\n")

    # ---- Preprocessing ----
    L.append("## Preprocessing Summary\n")
    pre = s.get("preprocessing", {})
    L.append(f"- **Log2 transform:** {pre.get('log_transform', 'N/A')}")
    L.append(f"- **Min total count filter:** {pre.get('min_total_count', 'N/A')}")
    L.append(f"- **Genes after count filter:** {pre.get('n_after_count_filter', 'N/A')}")
    L.append(f"- **Min variance filter:** {pre.get('min_variance', 'N/A')}")
    L.append(f"- **Genes after variance filter:** {pre.get('n_after_variance_filter', 'N/A')}")
    L.append(f"- **Max features selected:** {pre.get('max_features', 'N/A')}")
    L.append(f"- **Final feature count:** {pre.get('n_final_features', 'N/A')}\n")

    # ---- PCA ----
    L.append("## PCA Results\n")
    pca = s.get("pca", {})
    ev = pca.get("explained_variance_ratio", [])
    cv = pca.get("cumulative_variance", [])
    if ev:
        L.append("| Component | Explained Variance | Cumulative |")
        L.append("|-----------|-------------------|------------|")
        for i, (e, c) in enumerate(zip(ev[:10], cv[:10])):
            L.append(f"| PC{i+1} | {e:.4f} | {c:.4f} |")
    L.append("")

    # ---- Clustering ----
    L.append("## Clustering Results\n")
    for method, info in s.get("clustering", {}).items():
        L.append(f"### {method}\n")
        if "silhouette_score" in info:
            L.append(f"- **Silhouette score:** {info['silhouette_score']}")
        if "adjusted_rand_index" in info:
            L.append(f"- **Adjusted Rand Index (vs true labels):** {info['adjusted_rand_index']}")
        if "cluster_sizes" in info:
            L.append(f"- **Cluster sizes:** {info['cluster_sizes']}")
        L.append("")

    # ---- Classification ----
    n = s["dataset"]["n_samples"]
    L.append("## Classification Results (Exploratory)\n")
    L.append(
        f"> **Note:** With only {n} samples and Leave-One-Out cross-validation, "
        "these results are exploratory and should not be interpreted as "
        "generalizable predictive performance.\n"
    )

    clf = s.get("classification")
    if clf:
        for model_name, info in clf.items():
            L.append(f"### {model_name}\n")
            L.append(f"- **Accuracy (LOO-CV):** {info['accuracy']}")
            L.append(f"- **Macro F1 (LOO-CV):** {info['macro_f1']}")
            L.append(f"- **Classes:** {info.get('class_names', 'N/A')}")

            cm = info.get("confusion_matrix")
            cnames = info.get("class_names", [])
            if cm and cnames:
                L.append("\n**Confusion Matrix:**\n")
                L.append("| | " + " | ".join(f"Pred: {c}" for c in cnames) + " |")
                L.append("|" + "|".join(["---"] * (len(cnames) + 1)) + "|")
                for i, row in enumerate(cm):
                    L.append(f"| **{cnames[i]}** | " + " | ".join(str(v) for v in row) + " |")
            L.append("")
    else:
        L.append("Classification was skipped (no valid target column).\n")

    # ---- Top features ----
    L.append("## Top Ranked Features\n")
    tf = s.get("top_features", {})
    features = tf.get("features", [])
    if features:
        L.append(f"Top {tf.get('top_n', 'N')} features by average rank across methods:\n")
        L.append("| Rank | Feature | Avg Rank | Variance |")
        L.append("|------|---------|----------|----------|")
        for i, fi in enumerate(features[:20], 1):
            L.append(
                f"| {i} | `{fi['feature']}` | "
                f"{fi.get('avg_rank', 'N/A')} | "
                f"{fi.get('variance', 'N/A')} |"
            )
    L.append("")

    # ---- Biological interpretation ----
    L.append("## Biological Interpretation\n")
    L.append(
        "This analysis explored the transcriptomic response of rice (*Oryza sativa*) "
        "suspension cultures to two safener compounds (metcamifen and cyprosulfamide) "
        "versus a DMSO control, sampled at three time points (30, 90, and 240 minutes).\n"
    )
    L.append(
        "PCA and clustering analyses reveal the major axes of transcriptomic variation "
        "across treatment groups. Features ranked as important by both supervised "
        "(classification coefficient / importance) and unsupervised (variance) methods "
        "may represent genes whose expression is most responsive to safener treatment. "
        "These candidate genes warrant further investigation through differential "
        "expression analysis, functional annotation, and pathway enrichment using "
        "dedicated statistical frameworks (e.g., DESeq2, edgeR).\n"
    )
    L.append(
        "The balanced experimental design (3 treatments × 3 time points × 3 replicates "
        "= 27 samples) supports exploratory pattern discovery but is insufficient for "
        "robust biomarker validation.\n"
    )

    # ---- Limitations ----
    L.append("## Limitations and Assumptions\n")
    L.append(
        "1. **Small sample size (n=27):** All machine-learning results are exploratory. "
        "With 9 samples per treatment class and Leave-One-Out CV, metrics may exhibit "
        "high variance and should not be over-interpreted.\n"
    )
    L.append(
        "2. **No formal normalization:** Log2(x+1) transformation is applied to raw "
        "counts. For publication-grade differential expression, proper normalization "
        "(TMM, DESeq2 size factors, VST) is recommended.\n"
    )
    L.append(
        "3. **Feature selection bias:** Top features are selected by variance across "
        "all samples *before* classification. For rigorous biomarker discovery, "
        "feature selection should be embedded within the cross-validation loop.\n"
    )
    L.append(
        "4. **No pathway or functional enrichment:** Gene identifiers are reported "
        "without annotation. Downstream analysis should map them to GO terms, "
        "KEGG pathways, or rice-specific databases.\n"
    )
    L.append(
        "5. **Read counting caveat:** The original count data allowed multi-mapping "
        "(up to 6 gene regions per read), which may inflate counts for gene families "
        "with high sequence similarity.\n"
    )
    L.append(
        "6. **Exploratory scope:** This project is a reusable computational analysis "
        "workflow, not a replacement for rigorous statistical genomics pipelines. "
        "Results are hypothesis-generating.\n"
    )

    # ---- Warnings ----
    if s.get("warnings"):
        L.append("## Warnings\n")
        for w in s["warnings"]:
            L.append(f"- {w}")
        L.append("")

    return "\n".join(L)
