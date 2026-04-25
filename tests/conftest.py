"""Shared pytest fixtures for OmicsInsight tests."""

import pytest


@pytest.fixture
def synthetic_counts(tmp_path):
    """Create a small synthetic count-data file (genes × samples)."""
    path = tmp_path / "counts.txt"
    lines = [
        "\tSample1\tSample2\tSample3\tSample4\tSample5\tSample6",
        "Gene1\t100\t200\t150\t50\t60\t70",
        "Gene2\t0\t0\t0\t0\t0\t0",
        "Gene3\t50\t55\t52\t10\t12\t11",
        "Gene4\t300\t280\t310\t500\t520\t490",
        "Gene5\t5\t3\t4\t200\t210\t190",
        "Gene6\t10\t12\t11\t10\t11\t10",
        "Gene7\t1\t2\t1\t1\t1\t2",
        "Gene8\t80\t90\t85\t30\t35\t32",
        "Gene9\t20\t22\t21\t45\t50\t47",
        "Gene10\t0\t1\t0\t0\t0\t1",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
    return str(path)


@pytest.fixture
def synthetic_metadata(tmp_path):
    """Create a small synthetic GEO series-matrix file."""
    path = tmp_path / "series_matrix.txt"
    lines = [
        '!Series_title\t"Synthetic Test Experiment"',
        '!Series_geo_accession\t"GSE000001"',
        '!Sample_title\t"ctrl_1"\t"ctrl_2"\t"ctrl_3"\t"treat_1"\t"treat_2"\t"treat_3"',
        '!Sample_geo_accession\t"GSM1"\t"GSM2"\t"GSM3"\t"GSM4"\t"GSM5"\t"GSM6"',
        '!Sample_description\t"Sample1"\t"Sample2"\t"Sample3"\t"Sample4"\t"Sample5"\t"Sample6"',
        '!Sample_characteristics_ch1\t"treatment: Control"\t"treatment: Control"\t"treatment: Control"\t"treatment: Treated"\t"treatment: Treated"\t"treatment: Treated"',
        '!Sample_characteristics_ch1\t"replicate: 1"\t"replicate: 2"\t"replicate: 3"\t"replicate: 1"\t"replicate: 2"\t"replicate: 3"',
        "!series_matrix_table_begin",
        '"ID_REF"\t"GSM1"\t"GSM2"\t"GSM3"\t"GSM4"\t"GSM5"\t"GSM6"',
        "!series_matrix_table_end",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
    return str(path)


@pytest.fixture
def synthetic_config_yaml(tmp_path, synthetic_counts, synthetic_metadata):
    """Create a YAML config pointing at the synthetic files."""
    path = tmp_path / "config.yaml"
    # Use forward slashes: backslashes in YAML double-quoted strings are
    # treated as escape sequences, causing ScannerError on Windows paths.
    counts_p = synthetic_counts.replace("\\", "/")
    metadata_p = synthetic_metadata.replace("\\", "/")
    output_p = str(tmp_path / "outputs").replace("\\", "/")
    content = (
        f"counts_path: \"{counts_p}\"\n"
        f"metadata_path: \"{metadata_p}\"\n"
        f"output_dir: \"{output_p}\"\n"
        "target_column: \"treatment\"\n"
        "sample_id_column: \"sample_id\"\n"
        "max_features: 500\n"
        "n_clusters: 2\n"
        "log_transform: true\n"
        "min_total_count: 5\n"
        "min_variance: 0.0\n"
        "random_state: 42\n"
        "n_pca_components: 5\n"
        "umap_enabled: false\n"
    )
    path.write_text(content, encoding="utf-8")
    return str(path)
