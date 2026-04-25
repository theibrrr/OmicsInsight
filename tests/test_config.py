"""Tests for configuration loading."""

import pytest

from omicsinsight.config import PipelineConfig


class TestPipelineConfig:
    def test_defaults(self):
        cfg = PipelineConfig()
        assert cfg.counts_path == ""
        assert cfg.max_features == 500
        assert cfg.log_transform is True

    def test_from_yaml(self, synthetic_config_yaml):
        cfg = PipelineConfig.from_yaml(synthetic_config_yaml)
        assert cfg.target_column == "treatment"
        assert cfg.n_clusters == 2
        assert cfg.umap_enabled is False

    def test_to_dict_roundtrip(self):
        cfg = PipelineConfig(counts_path="a.txt", max_features=100)
        d = cfg.to_dict()
        assert d["counts_path"] == "a.txt"
        assert d["max_features"] == 100

    def test_from_yaml_ignores_unknown_keys(self, tmp_path):
        p = tmp_path / "bad.yaml"
        p.write_text("counts_path: x.txt\nfoo_unknown: 99\n")
        cfg = PipelineConfig.from_yaml(str(p))
        assert cfg.counts_path == "x.txt"

    def test_from_yaml_empty_file(self, tmp_path):
        p = tmp_path / "empty.yaml"
        p.write_text("")
        cfg = PipelineConfig.from_yaml(str(p))
        assert cfg.counts_path == ""
