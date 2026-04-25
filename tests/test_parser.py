"""Tests for dataset_parser module."""

import pandas as pd
import pytest

from omicsinsight.dataset_parser import (
    _clean_column_name,
    _parse_characteristic,
    _strip_quotes,
    align_data,
    parse_count_matrix,
    parse_series_matrix,
)


class TestStripQuotes:
    def test_removes_surrounding_quotes(self):
        assert _strip_quotes('"hello"') == "hello"

    def test_no_quotes(self):
        assert _strip_quotes("hello") == "hello"

    def test_empty_string(self):
        assert _strip_quotes("") == ""

    def test_strips_whitespace(self):
        assert _strip_quotes('  "test"  ') == "test"


class TestParseCharacteristic:
    def test_normal_format(self):
        trait, val = _parse_characteristic('"treatment: DMSO"')
        assert trait == "treatment"
        assert val == "DMSO"

    def test_no_colon(self):
        trait, val = _parse_characteristic("no_colon_here")
        assert trait == "unknown"

    def test_colon_in_value(self):
        trait, val = _parse_characteristic('"info: a:b:c"')
        assert trait == "info"
        assert val == "a:b:c"


class TestCleanColumnName:
    def test_basic(self):
        assert _clean_column_name("time point (minutes)") == "time_point_minutes"

    def test_already_clean(self):
        assert _clean_column_name("treatment") == "treatment"


class TestParseCountMatrix:
    def test_shape(self, synthetic_counts):
        df = parse_count_matrix(synthetic_counts)
        assert df.shape == (10, 6)
        assert list(df.columns) == [f"Sample{i}" for i in range(1, 7)]

    def test_dtype(self, synthetic_counts):
        df = parse_count_matrix(synthetic_counts)
        assert (df.dtypes == int).all()

    def test_missing_file(self):
        with pytest.raises(FileNotFoundError):
            parse_count_matrix("/nonexistent/path.txt")


class TestParseSeriesMatrix:
    def test_columns_present(self, synthetic_metadata):
        df = parse_series_matrix(synthetic_metadata)
        assert "sample_id" in df.columns
        assert "treatment" in df.columns
        assert "replicate" in df.columns

    def test_sample_count(self, synthetic_metadata):
        df = parse_series_matrix(synthetic_metadata)
        assert len(df) == 6

    def test_treatment_values(self, synthetic_metadata):
        df = parse_series_matrix(synthetic_metadata)
        assert set(df["treatment"]) == {"Control", "Treated"}

    def test_missing_file(self):
        with pytest.raises(FileNotFoundError):
            parse_series_matrix("/nonexistent/path.txt")


class TestAlignData:
    def test_alignment(self, synthetic_counts, synthetic_metadata):
        counts = parse_count_matrix(synthetic_counts)
        meta = parse_series_matrix(synthetic_metadata)
        c_aligned, m_aligned = align_data(counts, meta)
        assert list(c_aligned.columns) == list(m_aligned["sample_id"])

    def test_no_overlap_raises(self, synthetic_counts, synthetic_metadata):
        counts = parse_count_matrix(synthetic_counts)
        meta = parse_series_matrix(synthetic_metadata)
        meta["sample_id"] = ["X1", "X2", "X3", "X4", "X5", "X6"]
        with pytest.raises(ValueError, match="No common sample"):
            align_data(counts, meta)
