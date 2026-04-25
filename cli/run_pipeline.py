"""CLI entry point for the OmicsInsight analysis pipeline."""

import argparse
import sys

from omicsinsight.config import PipelineConfig
from omicsinsight.pipeline import run_pipeline
from omicsinsight.utils import setup_logging


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="OmicsInsight",
        description=(
            "Transcriptomics-focused analysis pipeline — "
            "exploratory ML, dimensionality reduction, and visualization."
        ),
    )
    parser.add_argument(
        "--counts", type=str,
        help="Path to tab-separated count data file.",
    )
    parser.add_argument(
        "--metadata", type=str,
        help="Path to GEO series matrix or metadata file.",
    )
    parser.add_argument(
        "--target", type=str, default="treatment",
        help="Target column for classification (default: treatment).",
    )
    parser.add_argument(
        "--sample-id", type=str, default="sample_id",
        help="Sample ID column name (default: sample_id).",
    )
    parser.add_argument(
        "--output", type=str, default="outputs/run_01",
        help="Output directory (default: outputs/run_01).",
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to YAML config file (overrides other arguments).",
    )
    parser.add_argument(
        "--max-features", type=int, default=500,
        help="Max features to select by variance (default: 500).",
    )
    parser.add_argument(
        "--n-clusters", type=int, default=3,
        help="Number of clusters (default: 3).",
    )
    parser.add_argument(
        "--no-log-transform", action="store_true",
        help="Disable log2 transformation.",
    )
    parser.add_argument(
        "--no-umap", action="store_true",
        help="Disable UMAP even if umap-learn is installed.",
    )
    parser.add_argument(
        "--min-total-count", type=int, default=10,
        help="Min total count threshold for gene filtering (default: 10).",
    )
    parser.add_argument(
        "--random-state", type=int, default=42,
        help="Random seed for reproducibility (default: 42).",
    )
    parser.add_argument(
        "--log-level", type=str, default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO).",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    log = setup_logging(args.log_level)

    # --- Resolve configuration ---
    if args.config:
        log.info("Loading config from %s", args.config)
        config = PipelineConfig.from_yaml(args.config)
        # Allow CLI flags to override YAML values
        if args.counts:
            config.counts_path = args.counts
        if args.metadata:
            config.metadata_path = args.metadata
        if args.output != "outputs/run_01":
            config.output_dir = args.output
    else:
        if not args.counts or not args.metadata:
            parser.error("--counts and --metadata are required (or provide --config)")
        config = PipelineConfig(
            counts_path=args.counts,
            metadata_path=args.metadata,
            output_dir=args.output,
            target_column=args.target,
            sample_id_column=args.sample_id,
            max_features=args.max_features,
            n_clusters=args.n_clusters,
            log_transform=not args.no_log_transform,
            min_total_count=args.min_total_count,
            random_state=args.random_state,
            umap_enabled=not args.no_umap,
        )

    log.info("Starting OmicsInsight pipeline")
    log.info("Config: %s", config.to_dict())

    try:
        run_pipeline(config)
        log.info("Pipeline completed successfully.")
    except Exception as exc:
        log.error("Pipeline failed: %s", exc, exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
