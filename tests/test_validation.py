"""Tests for the validation module."""

import pandas as pd
import pytest

from omicsinsight.validation import (
    ValidationError,
    validate_counts,
    validate_file_exists,
    validate_metadata,
)


class TestValidateFileExists:
    def test_valid_file(self, synthetic_counts):
        # Should not raise
        validate_file_exists(synthetic_counts, "Test file")

    def test_missing_file(self):
        with pytest.raises(ValidationError, match="not found"):
            validate_file_exists("/nope/absent.txt")

    def test_empty_file(self, tmp_path):
        p = tmp_path / "empty.txt"
        p.write_text("")
        with pytest.raises(ValidationError, match="empty"):
            validate_file_exists(str(p))


class TestValidateCounts:
    def test_valid(self):
        df = pd.DataFrame({"S1": [1, 2], "S2": [3, 4]}, index=["G1", "G2"])
        warnings = validate_counts(df)
        assert isinstance(warnings, list)

    def test_no_genes(self):
        df = pd.DataFrame(columns=["S1"])
        with pytest.raises(ValidationError, match="no genes"):
            validate_counts(df)

    def test_duplicate_genes(self):
        df = pd.DataFrame({"S1": [1, 2]}, index=["G1", "G1"])
        with pytest.raises(ValidationError, match="Duplicate gene"):
            validate_counts(df)

    def test_negative_values_warning(self):
        df = pd.DataFrame({"S1": [-1, 2], "S2": [3, 4]}, index=["G1", "G2"])
        warnings = validate_counts(df)
        assert any("negative" in w for w in warnings)

    def test_zero_genes_warning(self):
        df = pd.DataFrame({"S1": [0, 2], "S2": [0, 4]}, index=["G1", "G2"])
        warnings = validate_counts(df)
        assert any("zero counts" in w for w in warnings)


class TestValidateMetadata:
    def test_valid(self):
        df = pd.DataFrame({
            "sample_id": ["S1", "S2"],
            "treatment": ["A", "B"],
        })
        warnings = validate_metadata(df, "treatment", "sample_id")
        assert isinstance(warnings, list)

    def test_missing_sample_id_col(self):
        df = pd.DataFrame({"treatment": ["A", "B"]})
        with pytest.raises(ValidationError, match="Sample ID column"):
            validate_metadata(df, "treatment", "sample_id")

    def test_missing_target_warning(self):
        df = pd.DataFrame({"sample_id": ["S1", "S2"]})
        warnings = validate_metadata(df, "treatment", "sample_id")
        assert any("not found" in w for w in warnings)

    def test_duplicate_ids(self):
        df = pd.DataFrame({
            "sample_id": ["S1", "S1"],
            "treatment": ["A", "B"],
        })
        with pytest.raises(ValidationError, match="Duplicate"):
            validate_metadata(df, "treatment", "sample_id")
