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
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

if TYPE_CHECKING:
    import polars as pl

#: The footer key of :func:`write_lean_parquet`'s *write_id*.
WRITE_ID_KEY = b"flextool_write_id"
#: Footer key of the ``{period: solve}`` labels of a table whose frame does not carry them (#210).
SOLVE_LABELS_KEY = b"flextool_solve_labels"


def write_lean_parquet(
    df: pd.DataFrame, path: str | Path, *, index: bool = True,
    column_metadata: "dict | None" = None, write_id: str | None = None,
    solve_labels: "dict[str, str] | None" = None,
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
    column_metadata : dict, optional
        ``{leaf_column_name: {"unit", "semantics", "tooltip", "formula",
        "docs"}}`` describing each logical column.  Stored as a file-level
        ``flextool_columns`` JSON blob (keyed by *leaf* name, robust to the
        MultiIndex/scenario flattening that arrow applies to field names).
        Read it back with :func:`read_lean_parquet_column_meta`.
    write_id : str, optional
        The id of the write this file belongs to (the comparison files: every table and ``_metadata.json`` of
        one write carry the same id), stored as ``flextool_write_id``.  Read it back with
        :func:`read_lean_parquet_write_id` or :attr:`LeanLayout.write_id`.
    solve_labels : dict, optional
        ``{period: solve}``: the solve label of each period that the table's own frame does not carry (the
        investment tables: the solve that realized the period's investments, #210), stored as
        ``flextool_solve_labels`` so a parquet replay labels the table as the native writers did.  Read it back
        with :func:`read_lean_parquet_solve_labels`.
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

    # Record column index level names.  For a MultiIndex we store the
    # full list; for a plain single-level ``Index`` we store the single
    # ``.name`` under a distinct key so the old MultiIndex reconstruction
    # path is untouched.  ``None`` names are skipped (nothing to restore).
    if isinstance(df.columns, pd.MultiIndex):
        info["col"] = list(df.columns.names)
    elif df.columns.name is not None:
        info["col_name"] = df.columns.name

    # Swap the bulky pandas blob for our compact one.
    meta = dict(table.schema.metadata or {})
    meta.pop(b"pandas", None)
    meta[b"flextool"] = json.dumps(info).encode()
    if column_metadata:
        meta[b"flextool_columns"] = json.dumps(column_metadata).encode()
    if write_id is not None:
        meta[WRITE_ID_KEY] = write_id.encode()
    if solve_labels:
        meta[SOLVE_LABELS_KEY] = json.dumps(solve_labels).encode()
    table = table.replace_schema_metadata(meta)

    pq.write_table(table, str(path))


def read_lean_parquet_column_meta(path: str | Path) -> dict | None:
    """Return the per-column metadata blob written by :func:`write_lean_parquet`.

    Reads only the file footer (cheap).  Returns ``{leaf_column: {...}}`` or
    ``None`` when the file carries no ``flextool_columns`` metadata.
    """
    file_meta = pq.ParquetFile(str(path)).schema_arrow.metadata or {}
    blob = file_meta.get(b"flextool_columns")
    return json.loads(blob) if blob else None


def read_lean_parquet_solve_labels(path: str | Path) -> dict[str, str]:
    """The ``{period: solve}`` labels :func:`write_lean_parquet` stored in the file at *path* (``{}`` without
    them).  Reads only the footer."""
    file_meta = pq.ParquetFile(str(path)).schema_arrow.metadata or {}
    blob = file_meta.get(SOLVE_LABELS_KEY)
    return {str(k): str(v) for k, v in json.loads(blob).items()} if blob else {}


def read_lean_parquet_write_id(path: str | Path) -> str | None:
    """The ``write_id`` :func:`write_lean_parquet` stored in the file at *path* (``None`` without one).  Reads
    only the footer."""
    with pq.ParquetFile(_open_source(Path(path))) as pf:
        meta = pf.schema_arrow.metadata or {}
    return _write_id_of(meta)


def _write_id_of(meta: dict) -> str | None:
    value = meta.get(WRITE_ID_KEY)
    return value.decode() if value else None


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
    else:
        # Single-level column Index — restore the ``.name`` if one was
        # recorded at write time.  Only touches files that actually carry
        # the ``col_name`` field, so older parquets are unaffected.
        single_col_name = info.get("col_name")
        if single_col_name is not None:
            df.columns.name = single_col_name

    return df


# ---------------------------------------------------------------------------------------------------------
# Reading a table without pandas: the footer's layout, the wide frame, the long view
#
# A lean parquet stores a table "wide": the row index levels as columns of their own, and one physical column
# per column key, named by arrow's text of the key tuple (``"('scenario', 'unit', 'node')"``; the bare key for a
# single column level).  This module is the one place that knows those names.  The functions below read only
# what is asked for: :func:`lean_parquet_layout` the footer, :func:`read_lean_parquet_wide` the index columns and
# the kept data columns, and :func:`to_long` turns an already reduced wide frame into rows with one column per
# level.

#: Files up to this size are read into memory in one go and closed before they are parsed.
IN_MEMORY_MAX_BYTES = 64 * 1024 * 1024


@dataclass(frozen=True)
class LeanLayout:
    """The layout of a parquet table, from its footer.

    *idx* are the row level names (an unnamed level is ``level_<i>``) and *index_physical* their physical
    columns; *col* the column level names (empty for a flat table, whose columns are all row levels).  Each data
    column has its key (one value per column level) in *columns* and its physical name in *physical*, in file
    order.  *column_meta* is the ``flextool_columns`` blob, *num_rows* the row count, and *lean* whether the file
    carries FlexTool's footer (else the pandas metadata, or none, described it).  *write_id* is the id of the
    write the file belongs to (``write_lean_parquet(write_id=...)``), ``None`` without one.
    """

    idx: tuple[str, ...]
    index_physical: tuple[str, ...]
    col: tuple[str, ...]
    columns: tuple[tuple[str, ...], ...]
    physical: tuple[str, ...]
    column_meta: dict | None
    num_rows: int
    lean: bool
    write_id: str | None = None

    def level_values(self, level: str) -> list[str]:
        """The distinct keys of the column level *level*, in file order."""
        i = self.col.index(level)
        return list(dict.fromkeys(key[i] for key in self.columns))

    def select(self, filters: "dict[str, Callable[[str], bool]] | None" = None) -> list[str]:
        """The physical data columns whose key passes every ``{column level: predicate}`` of *filters*."""
        if not filters:
            return list(self.physical)
        tests = [(self.col.index(level), test) for level, test in filters.items()]
        return [name for name, key in zip(self.physical, self.columns)
                if all(test(key[i]) for i, test in tests)]


def _level_names(names: list, prefix: str) -> tuple[str, ...]:
    return tuple(str(n) if n is not None else f"{prefix}_{i}" for i, n in enumerate(names))


def _tuple_of(physical: str, n_levels: int) -> tuple[str, ...] | None:
    """The key in arrow's tuple text *physical* when it is a tuple of *n_levels* values, else ``None``."""
    try:
        value = ast.literal_eval(physical)
    except (ValueError, SyntaxError):
        return None
    if not isinstance(value, tuple) or len(value) != n_levels:
        return None
    return tuple(str(v) for v in value)


def _key_of(physical: str, n_levels: int, *, tuple_text: bool) -> tuple[str, ...]:
    """The column key of a physical column name: arrow's text of the key tuple when *tuple_text* (a
    ``MultiIndex`` column axis, one level included: ``"('s1',)"``), else the bare name of a single level."""
    if not tuple_text:
        return (physical,)
    key = _tuple_of(physical, n_levels)
    if key is None:
        raise ValueError(f"the column {physical!r} is not a key of {n_levels} levels")
    return key


def _open_source(path: Path) -> "pa.BufferReader | str":
    """What pyarrow reads *path* from: its bytes when the file is small (the file is closed before parsing),
    else the path itself (pyarrow opens and closes it within the call)."""
    if path.stat().st_size <= IN_MEMORY_MAX_BYTES:
        return pa.BufferReader(path.read_bytes())
    return str(path)


def lean_parquet_layout(path: str | Path) -> LeanLayout:
    """The layout of the parquet table at *path*, from its footer only.

    Reads FlexTool's ``flextool`` footer (``write_lean_parquet``), else the pandas metadata of a file written by
    ``DataFrame.to_parquet``, else treats every column as a row level (a flat table).  Raises
    ``pyarrow.ArrowInvalid`` for a file that is not (or not yet) a complete parquet file.
    """
    path = Path(path)
    with pq.ParquetFile(_open_source(path)) as pf:
        schema = pf.schema_arrow
        num_rows = pf.metadata.num_rows
    names = list(schema.names)
    meta = schema.metadata or {}
    column_meta = json.loads(meta[b"flextool_columns"]) if meta.get(b"flextool_columns") else None
    if b"flextool" in meta:
        info = json.loads(meta[b"flextool"])
        raw_idx = list(info.get("idx") or [])
        index_physical = [n if n is not None else f"__index_level_{i}__" for i, n in enumerate(raw_idx)]
        idx = _level_names(raw_idx, "level")
        tuple_text = bool(info.get("col"))                # written from a MultiIndex column axis
        if info.get("col"):
            col = _level_names(list(info["col"]), "column_level")
        elif info.get("col_name") is not None:
            col = (str(info["col_name"]),)
        elif raw_idx:
            col = ("column",)
        else:
            col = ()
        lean = True
    elif b"pandas" in meta:
        pandas_meta = json.loads(meta[b"pandas"])
        by_field = {c.get("field_name"): c.get("name") for c in pandas_meta.get("columns", [])}
        index_physical = [c for c in pandas_meta.get("index_columns", []) if isinstance(c, str)]
        idx = _level_names([by_field.get(c) for c in index_physical], "level")
        col = _level_names([c.get("name") for c in pandas_meta.get("column_indexes", [])], "column_level")
        if not col:
            col = ("column",)
        tuple_text = None                                 # decided from the column names below
        lean = False
    else:
        index_physical, idx, col, lean, tuple_text = [], (), (), False, False
    present = [p for p in index_physical if p in names]
    idx = tuple(level for level, p in zip(idx, index_physical) if p in names)
    if not col:                                       # a flat table: every column is a row level
        return LeanLayout(tuple(names), tuple(names), (), (), (), column_meta, num_rows, lean, _write_id_of(meta))
    data = [n for n in names if n not in set(present)]
    if tuple_text is None:
        # pandas metadata cannot tell a one-level MultiIndex from a plain Index: the names decide, for the whole
        # file at once (every data column the tuple text of one key, or none of them), so the keys never mix.
        tuple_text = len(col) > 1 or (bool(data) and all(_tuple_of(n, len(col)) is not None for n in data))
    columns = tuple(_key_of(n, len(col), tuple_text=tuple_text) for n in data)
    return LeanLayout(idx, tuple(present), col, columns, tuple(data), column_meta, num_rows, lean,
                      _write_id_of(meta))


def read_lean_parquet_wide(path: str | Path, *, keep: "list[str] | None" = None,
                           layout: LeanLayout | None = None) -> "pl.DataFrame":
    """The table at *path* as a wide polars frame: the row levels (named as in the layout) and the physical data
    columns *keep* (all when ``None``), in file order.  Only those columns are read."""
    import polars as pl

    path = Path(path)
    layout = layout if layout is not None else lean_parquet_layout(path)
    wanted = set(layout.physical if keep is None else keep)
    unknown = wanted - set(layout.physical)
    if unknown:
        raise KeyError(f"not data columns of {path.name}: {sorted(unknown)[:5]}")
    data = [p for p in layout.physical if p in wanted]
    table = pq.read_table(_open_source(path), columns=list(layout.index_physical) + data)
    frame = pl.from_arrow(table)
    renames = {p: level for p, level in zip(layout.index_physical, layout.idx)
               if p != level and level not in frame.columns}
    return frame.rename(renames) if renames else frame


def _value_kind(dtype) -> str | None:
    """What kind of values a data column of *dtype* holds: ``numbers``, ``text``, the dtype's name for any other,
    ``None`` for a column with no values (``Null``)."""
    import polars as pl

    if dtype == pl.Null:
        return None
    if dtype.is_numeric():
        return "numbers"
    if dtype == pl.String:
        return "text"
    return str(dtype.base_type())


def to_long(wide: "pl.DataFrame", layout: LeanLayout, *, value_name: str = "value") -> "pl.DataFrame":
    """*wide* (columns from :func:`read_lean_parquet_wide`, possibly reduced) with one row per (row, data column):
    its row levels, one column per column level, and *value_name*.  Meant for frames already reduced or
    filtered: a large table's long form has rows × columns rows.  Numbers come out as ``Float64``, text as
    text; data columns that mix kinds (numbers and text) or a level that is both a row and a column level raise
    ``ValueError``."""
    import polars as pl

    if not layout.col:
        return wide
    keys = dict(zip(layout.physical, layout.columns))
    data = [c for c in wide.columns if c in keys]
    index = [c for c in wide.columns if c not in keys]
    clash = [level for level in layout.col if level in index]
    if clash:
        raise ValueError(f"the level {clash[0]!r} is both a row level and a column level")
    kinds: dict[str, list[str]] = {}
    for c in data:
        kind = _value_kind(wide.schema[c])
        if kind is not None:
            kinds.setdefault(kind, []).append(c)
    if len(kinds) > 1:
        parts = "; ".join(f"{kind}: {', '.join(names[:3])}" for kind, names in kinds.items())
        raise ValueError(f"the data columns mix {' and '.join(kinds)} ({parts}); select columns of one kind")
    if data and set(kinds) <= {"numbers"}:
        wide = wide.with_columns(pl.col(data).cast(pl.Float64))
    long = wide.unpivot(index=index, on=data, variable_name="__column", value_name=value_name)
    mapping = pl.DataFrame(
        {"__column": data, **{level: [keys[c][i] for c in data] for i, level in enumerate(layout.col)}},
        schema={"__column": pl.String, **{level: pl.String for level in layout.col}},
    )
    long = long.join(mapping, on="__column", how="left", maintain_order="left").drop("__column")
    return long.select([*index, *layout.col, value_name])
