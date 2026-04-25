"""Tests for the preprocessing module."""

import numpy as np
import pandas as pd

from omicsinsight.preprocessing import (
    filter_low_counts,
    filter_low_variance,
    log_transform,
    select_top_features,
    transpose_to_samples,
)


def _make_df():
    """Helper: small genes × samples DataFrame."""
    return pd.DataFrame(
        {"S1": [100, 0, 50], "S2": [200, 0, 55], "S3": [150, 1, 52]},
        index=["G1", "G2", "G3"],
    )


class TestTranspose:
    def test_shape(self):
        df = _make_df()
        t = transpose_to_samples(df)
        assert t.shape == (3, 3)
        assert list(t.index) == ["S1", "S2", "S3"]


class TestLogTransform:
    def test_values(self):
        df = pd.DataFrame({"a": [0, 1, 3]})
        result = log_transform(df, pseudo_count=1.0)
        expected = np.log2(df + 1)
        pd.testing.assert_frame_equal(result, expected)


class TestFilterLowCounts:
    def test_removes_zero_gene(self):
        df = transpose_to_samples(_make_df())
        out = filter_low_counts(df, min_total_count=10)
        # G2 total across samples = 0+0+1 = 1 < 10 → removed
        assert "G2" not in out.columns
        assert "G1" in out.columns


class TestFilterLowVariance:
    def test_removes_constant(self):
        df = pd.DataFrame({"A": [5, 5, 5], "B": [1, 2, 3]})
        out = filter_low_variance(df, min_variance=0.0)
        assert "A" not in out.columns
        assert "B" in out.columns


class TestSelectTopFeatures:
    def test_selects_correct_number(self):
        df = pd.DataFrame({
            "A": [1, 2, 3],
            "B": [10, 20, 30],
            "C": [5, 5, 5],
        })
        out = select_top_features(df, max_features=2)
        assert out.shape[1] == 2

    def test_keeps_all_when_max_exceeds(self):
        df = pd.DataFrame({"A": [1, 2], "B": [3, 4]})
        out = select_top_features(df, max_features=100)
        assert out.shape[1] == 2
