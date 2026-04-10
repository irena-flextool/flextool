"""Lean parquet I/O: strip bulky pandas metadata, store compact MultiIndex info.

Standard ``df.to_parquet()`` embeds a large JSON blob (often 10-20 KB) in the
parquet footer so that pandas can perfectly round-trip MultiIndex structures.
For small summary tables this metadata can be 10-20x larger than the actual
data.

These helpers replace that blob with a ~60-byte ``flextool`` metadata entry
that records only the index and column level names — enough to reconstruct
both row and column MultiIndex on read.  The resulting DataFrames are
completely normal pandas objects.

Backward-compatible: :func:`read_lean_parquet` detects old files that have
standard pandas metadata and falls back to ``pd.read_parquet()``.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


def write_lean_parquet(
    df: pd.DataFrame, path: str | Path, *, index: bool = True,
) -> None:
    """Write *df* to parquet with minimal metadata.

    Parameters
    ----------
    df : DataFrame
        The data to write.  May have MultiIndex on rows, columns, or both.
    path : str or Path
        Destination file path.
    index : bool
        If ``True`` (default), preserve the row index.  Pass ``False`` for
        flat tables where the index is meaningless (e.g. timeline_breaks).
    """
    # Don't preserve RangeIndex as a physical column — it carries no data
    # and would create a stray ``__index_level_0__`` column without the
    # pandas metadata blob to tell the reader to drop it.
    preserve = index and not isinstance(df.index, pd.RangeIndex)
    table = pa.Table.from_pandas(df, preserve_index=preserve)

    info: dict[str, list | None] = {}

    # Record row-index level names so we can reconstruct on read.
    if preserve:
        if isinstance(df.index, pd.MultiIndex):
            info["idx"] = list(df.index.names)
        else:
            info["idx"] = [df.index.name]

    # Record column-MultiIndex level names.
    if isinstance(df.columns, pd.MultiIndex):
        info["col"] = list(df.columns.names)

    # Swap the bulky pandas blob for our compact one.
    meta = dict(table.schema.metadata or {})
    meta.pop(b"pandas", None)
    meta[b"flextool"] = json.dumps(info).encode()
    table = table.replace_schema_metadata(meta)

    pq.write_table(table, str(path))


def read_lean_parquet(path: str | Path) -> pd.DataFrame:
    """Read a parquet file, reconstructing any MultiIndex.

    If the file contains ``flextool`` metadata (written by
    :func:`write_lean_parquet`), uses the compact reconstruction logic.
    Otherwise falls back to ``pd.read_parquet()`` for full backward
    compatibility with files written by plain ``df.to_parquet()``.
    """
    path_str = str(path)

    # Peek at file-level key-value metadata (cheap — reads only the footer).
    pf = pq.ParquetFile(path_str)
    file_meta = pf.schema_arrow.metadata or {}

    if b"flextool" not in file_meta:
        # Old-format file with standard pandas metadata.
        return pd.read_parquet(path_str)

    info: dict = json.loads(file_meta[b"flextool"])

    # Read all columns as a flat DataFrame (no pandas metadata to guide it).
    # pyarrow will create a default RangeIndex.
    table = pq.read_table(path_str)
    df = table.to_pandas()

    # --- Reconstruct row index ---
    idx_names: list | None = info.get("idx")
    if idx_names:
        # Map None level names to the __index_level_N__ placeholders that
        # pyarrow uses when preserve_index=True encounters unnamed levels.
        col_lookup: list[str] = []
        for i, name in enumerate(idx_names):
            if name is None:
                col_lookup.append(f"__index_level_{i}__")
            else:
                col_lookup.append(name)

        present = [c for c in col_lookup if c in df.columns]
        if present:
            df = df.set_index(present)
            # Restore the original names (including None).
            if isinstance(df.index, pd.MultiIndex):
                df.index.names = idx_names
            else:
                df.index.name = idx_names[0]

    # --- Reconstruct column MultiIndex ---
    col_level_names: list | None = info.get("col")
    if col_level_names:
        # pyarrow stores MultiIndex column names as string representations
        # of tuples — ``"('a', 'b')"`` for 2+ levels, ``"('a',)"`` for 1.
        tuples = [ast.literal_eval(c) for c in df.columns]
        df.columns = pd.MultiIndex.from_tuples(
            tuples, names=col_level_names,
        )

    return df
