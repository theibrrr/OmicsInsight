"""Configuration management for OmicsInsight pipeline."""

from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Dict

import yaml


@dataclass
class PipelineConfig:
    """All settings for a single pipeline run."""

    counts_path: str = ""
    metadata_path: str = ""
    output_dir: str = "outputs/run_01"
    target_column: str = "treatment"
    sample_id_column: str = "sample_id"
    max_features: int = 500
    n_clusters: int = 3
    log_transform: bool = True
    min_total_count: int = 10
    min_variance: float = 0.0
    random_state: int = 42
    n_pca_components: int = 10
    umap_enabled: bool = True

    @classmethod
    def from_yaml(cls, path: str) -> "PipelineConfig":
        """Load configuration from a YAML file."""
        with open(path, "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
        if data is None:
            data = {}
        valid_keys = {f.name for f in fields(cls)}
        filtered = {k: v for k, v in data.items() if k in valid_keys}
        return cls(**filtered)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a plain dictionary."""
        return {f.name: getattr(self, f.name) for f in fields(self)}
