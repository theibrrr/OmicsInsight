"""Parsers for GEO dataset files (count matrix and series matrix metadata)."""

import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

logger = logging.getLogger("omicsinsight")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _strip_quotes(value: str) -> str:
    """Remove surrounding double-quotes and whitespace."""
    value = value.strip()
    if len(value) >= 2 and value[0] == '"' and value[-1] == '"':
        return value[1:-1]
    return value


def _parse_characteristic(value: str) -> Tuple[str, str]:
    """Parse a GEO characteristic cell such as ``"treatment: DMSO"``.

    Returns (trait_name, trait_value).  Falls back to ("unknown", value)
    when no colon delimiter is present.
    """
    value = _strip_quotes(value)
    if ":" in value:
        trait, val = value.split(":", 1)
        return trait.strip(), val.strip()
    return "unknown", value


def _clean_column_name(name: str) -> str:
    """Convert a free-text trait name to a clean snake_case column name."""
    name = name.lower().strip()
    name = re.sub(r"[^a-z0-9]+", "_", name)
    return name.strip("_")


# ---------------------------------------------------------------------------
# Public parsers
# ---------------------------------------------------------------------------

def parse_count_matrix(path: str) -> pd.DataFrame:
    """Read a tab-separated count data file.

    Returns a **genes × samples** DataFrame (gene IDs as the index,
    sample IDs as column headers).  All values are coerced to integers.
    """
    logger.info("Parsing count matrix from %s", path)
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Count data file not found: {path}")

    df = pd.read_csv(path, sep="\t", index_col=0)
    df.index.name = "gene_id"

    # Coerce to numeric; replace any non-numeric cells with 0
    non_numeric = df.apply(
        lambda col: pd.to_numeric(col, errors="coerce").isna().sum()
    )
    bad_cols = non_numeric[non_numeric > 0]
    if len(bad_cols) > 0:
        logger.warning("Non-numeric values in columns: %s", list(bad_cols.index))

    df = df.apply(pd.to_numeric, errors="coerce").fillna(0).astype(int)

    logger.info("Count matrix loaded: %d genes × %d samples", df.shape[0], df.shape[1])
    return df


def parse_series_matrix(path: str) -> pd.DataFrame:
    """Parse a GEO series matrix file to extract per-sample metadata.

    Handles:
    * ``!Sample_*`` rows with one value per sample (tab-separated, quoted).
    * Multiple ``!Sample_characteristics_ch1`` rows where each cell has
      the ``"trait: value"`` format — each trait becomes its own column.
    * Rows with unexpected column counts (padded / truncated with a warning).
    * Missing or empty files raise informative errors.

    A ``sample_id`` column is added from ``!Sample_description`` (preferred)
    or ``!Sample_geo_accession`` as fallback.
    """
    logger.info("Parsing series matrix from %s", path)
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Series matrix file not found: {path}")

    sample_fields: Dict[str, List[str]] = {}
    characteristics: List[List[str]] = []
    n_samples: Optional[int] = None

    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.rstrip("\n\r")

            if not line.startswith("!Sample_"):
                continue

            parts = line.split("\t")
            if len(parts) < 2:
                continue

            key = parts[0]
            values = [_strip_quotes(v) for v in parts[1:]]

            # Establish expected sample count from the first !Sample_ row
            if n_samples is None:
                n_samples = len(values)

            # Pad / truncate mismatched rows
            if len(values) < n_samples:
                logger.warning(
                    "Row '%s' has %d values, expected %d — padding with ''.",
                    key, len(values), n_samples,
                )
                values.extend([""] * (n_samples - len(values)))
            elif len(values) > n_samples:
                logger.warning(
                    "Row '%s' has %d values, expected %d — truncating.",
                    key, len(values), n_samples,
                )
                values = values[:n_samples]

            if key == "!Sample_characteristics_ch1":
                characteristics.append(values)
            else:
                col_name = key.replace("!Sample_", "")
                if col_name not in sample_fields:
                    sample_fields[col_name] = values
                # silently skip duplicate non-characteristics rows

    if n_samples is None or n_samples == 0:
        raise ValueError(f"No sample metadata found in {path}")

    # Build base DataFrame (rows = samples, each key = column)
    df = pd.DataFrame(sample_fields)

    # Expand characteristics into separate columns
    for char_row in characteristics:
        trait_name: Optional[str] = None
        for v in char_row:
            if v and ":" in v:
                raw_trait, _ = _parse_characteristic(v)
                trait_name = _clean_column_name(raw_trait)
                break

        if trait_name is None:
            logger.warning("Could not determine trait name from a characteristics row — skipping.")
            continue

        parsed_values = []
        for v in char_row:
            if ":" in v:
                _, val = _parse_characteristic(v)
                parsed_values.append(val)
            else:
                parsed_values.append(_strip_quotes(v))

        df[trait_name] = parsed_values

    # Derive canonical sample_id column
    if "description" in df.columns:
        df["sample_id"] = df["description"]
    elif "geo_accession" in df.columns:
        df["sample_id"] = df["geo_accession"]
        logger.warning("No Sample_description found — using geo_accession as sample_id.")
    else:
        df["sample_id"] = [f"sample_{i}" for i in range(len(df))]
        logger.warning("No suitable sample ID column found — using generated IDs.")

    logger.info(
        "Metadata parsed: %d samples, columns: %s",
        len(df), list(df.columns),
    )
    return df


def align_data(
    counts: pd.DataFrame,
    metadata: pd.DataFrame,
    sample_id_col: str = "sample_id",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Align count matrix columns with metadata rows by sample ID.

    Returns (counts_aligned, metadata_aligned) sharing the same sample order.
    """
    count_samples = set(counts.columns)
    meta_samples = set(metadata[sample_id_col])

    common = count_samples & meta_samples
    only_counts = count_samples - meta_samples
    only_meta = meta_samples - count_samples

    if only_counts:
        logger.warning("Samples in counts but not metadata: %s", only_counts)
    if only_meta:
        logger.warning("Samples in metadata but not counts: %s", only_meta)

    if not common:
        raise ValueError(
            "No common sample IDs between count matrix and metadata. "
            f"Count columns (first 5): {list(counts.columns)[:5]}, "
            f"Metadata IDs (first 5): {list(metadata[sample_id_col])[:5]}"
        )

    # Keep metadata row order for the common samples
    meta_aligned = (
        metadata[metadata[sample_id_col].isin(common)]
        .copy()
        .reset_index(drop=True)
    )
    ordered_ids = list(meta_aligned[sample_id_col])
    counts_aligned = counts[ordered_ids]

    logger.info("Aligned %d samples between counts and metadata", len(common))
    return counts_aligned, meta_aligned


def parse_csv_metadata(path: str, sample_id_col: Optional[str] = None) -> pd.DataFrame:
    """Parse a plain CSV or TSV metadata file.

    The file must have a header row.  One column must contain sample IDs that
    match the count matrix column headers.

    If *sample_id_col* is provided and present in the file, it is used as-is.
    Otherwise the function looks for a column named ``sample_id`` and, failing
    that, promotes the first column to ``sample_id``.

    Parameters
    ----------
    path : str
        Path to a ``.csv`` or ``.tsv`` metadata file.
    sample_id_col : str, optional
        Name of the column that holds sample identifiers.

    Returns
    -------
    pd.DataFrame
        Metadata table with at least a ``sample_id`` column.
    """
    logger.info("Parsing CSV metadata from %s", path)
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Metadata file not found: {path}")

    sep = "\t" if p.suffix.lower() in {".tsv", ".txt"} else ","
    df = pd.read_csv(path, sep=sep, dtype=str)
    df.columns = [c.strip() for c in df.columns]

    # Resolve sample_id column
    if sample_id_col and sample_id_col in df.columns:
        if sample_id_col != "sample_id":
            df = df.rename(columns={sample_id_col: "sample_id"})
    elif "sample_id" in df.columns:
        pass  # already present
    else:
        first_col = df.columns[0]
        logger.warning(
            "No 'sample_id' column found in CSV metadata — using first column '%s'.",
            first_col,
        )
        df = df.rename(columns={first_col: "sample_id"})

    logger.info("CSV metadata loaded: %d samples, columns: %s", len(df), list(df.columns))
    return df


def parse_metadata(
    path: str,
    sample_id_col: Optional[str] = None,
) -> pd.DataFrame:
    """Auto-detect and parse a metadata file.

    Dispatches to :func:`parse_csv_metadata` for ``.csv`` / ``.tsv`` / ``.txt``
    files that do **not** start with a ``!``-prefixed GEO header, and to
    :func:`parse_series_matrix` otherwise.

    This is the recommended entry point for reusable metadata loading.
    """
    p = Path(path)
    suffix = p.suffix.lower()

    # Peek at the first non-empty line to detect GEO format
    first_line = ""
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    first_line = line
                    break
    except OSError:
        pass  # let the individual parsers raise a proper error

    is_geo = first_line.startswith("!")

    if is_geo:
        logger.info("Detected GEO series matrix format — using parse_series_matrix.")
        return parse_series_matrix(path)

    if suffix in {".csv", ".tsv", ".txt"}:
        logger.info("Detected tabular metadata format — using parse_csv_metadata.")
        return parse_csv_metadata(path, sample_id_col=sample_id_col)

    # Unknown extension but no GEO header — try CSV as fallback
    logger.warning(
        "Unknown metadata file extension '%s' — attempting CSV parse.", suffix
    )
    return parse_csv_metadata(path, sample_id_col=sample_id_col)

