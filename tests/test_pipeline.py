"""Integration test: full pipeline happy path on synthetic data."""

from pathlib import Path

from omicsinsight.config import PipelineConfig
from omicsinsight.pipeline import run_pipeline


class TestPipelineHappyPath:
    def test_full_run(self, synthetic_counts, synthetic_metadata, tmp_path):
        """Run the entire pipeline on 6-sample synthetic data."""
        output_dir = str(tmp_path / "outputs")

        config = PipelineConfig(
            counts_path=synthetic_counts,
            metadata_path=synthetic_metadata,
            output_dir=output_dir,
            target_column="treatment",
            sample_id_column="sample_id",
            max_features=500,
            n_clusters=2,
            log_transform=True,
            min_total_count=5,
            min_variance=0.0,
            random_state=42,
            n_pca_components=5,
            umap_enabled=False,
        )

        summary = run_pipeline(config)

        # Basic structure checks
        assert summary["dataset"]["n_samples"] == 6
        assert summary["dataset"]["n_features_after_preprocessing"] > 0

        # Output files exist
        out = Path(output_dir)
        assert (out / "analysis_summary.json").exists()
        assert (out / "report.md").exists()
        assert (out / "pca_scatter.png").exists()
        assert (out / "ranked_features.csv").exists()
        assert (out / "top_features.json").exists()
        assert (out / "cluster_labels.csv").exists()
        assert (out / "metadata.csv").exists()

        # Classification should have run (2 classes, 3 per class)
        assert summary["classification"] is not None
        assert "LogisticRegression" in summary["classification"]
        assert "RandomForest" in summary["classification"]

        # Model files saved
        assert (out / "model_LogisticRegression.joblib").exists()
        assert (out / "model_RandomForest.joblib").exists()
        assert (out / "scaler.joblib").exists()

        # Clustering includes ARI
        for method in ("KMeans", "Agglomerative"):
            assert "adjusted_rand_index" in summary["clustering"][method]
