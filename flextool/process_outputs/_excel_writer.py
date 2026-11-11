"""The result workbooks (scenario and comparison), with the unit of measure of every column on the sheets.

* A sheet whose columns have two or more levels (a wide table: ``unit, node``; or any table of a scenario
  comparison, which adds the ``scenario`` level) gets one more column level, :data:`UNIT_LEVEL`, as its last
  header row: the unit of each column right above its data.
* A sheet with one column level keeps one header row at ``A1`` with the column names unchanged.  Its data is an
  Excel Table (``tbl_<sheet>``, written with repeated, not merged, index values) that Power Query can read by
  name, and a small block to the right of the Table, one empty column apart, gives the units: one line when every
  value column has the same unit, else one line per value column.
* The first sheet ``Units`` lists every data sheet with its table, units, semantics and description (the sheet
  names link to the sheets); ``_output_metadata`` follows with one row per value column.

The units come from :func:`flextool.process_outputs._output_meta.resolve_columns`.
"""
from __future__ import annotations

import re

import pandas as pd

from flextool.process_outputs._output_meta import (
    OUTPUT_DESCRIPTION,
    ColumnMeta,
    _scenario_level,
    Semantics,
    display_unit,
    output_metadata_rows,
    resolve_columns,
    summarise_units,
)

#: The name of the column level (wide sheets) and of the header (unit blocks, ``Units`` sheet) that holds the
#: unit of measure.  Not ``unit``: that is already a column level of the unit tables.
UNIT_LEVEL = "unit of measure"
UNITS_SHEET = "Units"
METADATA_SHEET = "_output_metadata"
#: Sheet names the writer uses itself, lower-case (as :func:`excel_sheet_name` keeps its ``used`` set).
RESERVED_SHEET_NAMES: frozenset[str] = frozenset({UNITS_SHEET.lower(), METADATA_SHEET.lower()})
UNITS_SHEET_COLUMNS = ["sheet", "table", UNIT_LEVEL, "semantics", "description"]
METADATA_COLUMNS = ["sheet", "output", "column", "unit", "semantics", "tooltip", "long", "formula"]
#: ``_output_metadata`` of a scenario comparison: the scenario of a column in its own column.
COMPARISON_METADATA_COLUMNS = ["sheet", "output", "scenario", "column", "unit", "semantics", "tooltip", "long",
                               "formula"]
#: Column widths (characters) of the ``Units`` and ``_output_metadata`` sheets.
_UNITS_WIDTHS = {"sheet": 32, "table": 32, UNIT_LEVEL: 24, "semantics": 16, "description": 80}
_METADATA_WIDTHS = {"sheet": 32, "output": 32, "scenario": 20, "column": 32, "unit": 14, "semantics": 14,
                    "tooltip": 60, "long": 60, "formula": 40}
_TABLE_STYLE = "Table Style Light 1"
_MAX_TABLE_HEADER = 255


def reserved_sheet_names() -> set[str]:
    """A fresh ``used`` set for :func:`excel_sheet_name` with the writer's own sheet names taken."""
    return set(RESERVED_SHEET_NAMES)


def _as_frame(df) -> pd.DataFrame:
    """A Series as a one-column frame (named ``value`` when unnamed), a frame unchanged."""
    if isinstance(df, pd.Series):
        return df.to_frame(name=df.name if df.name is not None else "value")
    return df


def _measures(raw_key: str, df: pd.DataFrame) -> list[ColumnMeta | None]:
    """The :class:`ColumnMeta` of every column of *df*, ``None`` for a dimension column or an undeclared
    table."""
    resolved = resolve_columns(raw_key, df.columns)
    if resolved is None:
        return [None] * df.shape[1]
    return [m if m.semantics is not Semantics.DIMENSION else None for m in resolved]


def table_name(sheet_name: str, used: set[str]) -> str:
    """A valid, workbook-unique Excel Table name for *sheet_name*: ``tbl_`` and the sheet name with every
    character other than a letter, digit, ``_`` or ``.`` replaced by ``_`` (the prefix keeps it from reading as
    a cell reference such as ``CO2`` or ``R1C1``), ``_<n>`` added when taken; added to *used* (lower-case)."""
    base = "tbl_" + re.sub(r"[^A-Za-z0-9_.]", "_", sheet_name)
    name, n = base, 1
    while name.lower() in used:
        n += 1
        name = f"{base}_{n}"
    used.add(name.lower())
    return name


def table_headers(labels) -> list[str]:
    """Valid Table headers for *labels*: as text, at most 255 characters, an empty one named ``column_<i>``,
    and each distinct without regard to case (a repeat gets ``_<n>``)."""
    out: list[str] = []
    seen: set[str] = set()
    for i, label in enumerate(labels, start=1):
        text = "" if label is None else str(label)
        text = text[:_MAX_TABLE_HEADER] or f"column_{i}"
        candidate, n = text, 1
        while candidate.lower() in seen:
            n += 1
            tag = f"_{n}"
            candidate = text[:_MAX_TABLE_HEADER - len(tag)] + tag
        seen.add(candidate.lower())
        out.append(candidate)
    return out


def _flat_table(df: pd.DataFrame) -> pd.DataFrame:
    """*df* with its row index as leading columns (an unnamed default row number is dropped) and valid Table
    headers."""
    idx = df.index
    data = df.reset_index(drop=True)
    if isinstance(idx, pd.RangeIndex) and idx.name is None:
        flat = data
        labels = list(df.columns)
    else:
        index_frame = idx.to_frame(index=False)
        flat = pd.concat([index_frame, data], axis=1)
        labels = [n if n is not None else ("index" if idx.nlevels == 1 else f"level_{i}")
                  for i, n in enumerate(idx.names)] + list(df.columns)
    flat.columns = table_headers(labels)
    return flat


def _unit_block_rows(df: pd.DataFrame, measures: list[ColumnMeta | None]) -> list[tuple[str, str, str]]:
    """``(column, unit, semantics)`` lines of a Table sheet's unit block: one ``all value columns`` line when
    every value column has the same unit and semantics, else one line per value column."""
    lines = [(str(col), display_unit(m), m.semantics.value)
             for col, m in zip(df.columns, measures) if m is not None]
    if not lines:
        return []
    if len({(u, s) for _, u, s in lines}) == 1:
        return [("all value columns", lines[0][1], lines[0][2])]
    return lines


def _write_wide_sheet(writer, sheet_name: str, df: pd.DataFrame, measures: list[ColumnMeta | None]) -> None:
    """A sheet with two or more column levels, with :data:`UNIT_LEVEL` appended as the last header row."""
    units = [display_unit(m) if m is not None else "" for m in measures]
    if any(units):
        df = df.copy(deep=False)
        df.columns = pd.MultiIndex.from_tuples(
            [(*col, unit) for col, unit in zip(df.columns, units)],
            names=[*df.columns.names, UNIT_LEVEL])
    df.to_excel(writer, sheet_name=sheet_name)


def _write_table_sheet(writer, sheet_name: str, df: pd.DataFrame,
                       measures: list[ColumnMeta | None], tables: set[str]) -> None:
    """A sheet with one column level: an Excel Table at ``A1`` and the unit block beside it."""
    flat = _flat_table(df)
    flat.to_excel(writer, sheet_name=sheet_name, index=False, merge_cells=False)
    worksheet = writer.sheets[sheet_name]
    n_rows, n_cols = flat.shape
    worksheet.add_table(0, 0, n_rows, n_cols - 1, {
        "name": table_name(sheet_name, tables),
        "columns": [{"header": h} for h in flat.columns],
        "style": _TABLE_STYLE,
    })
    block = _unit_block_rows(df, measures)
    if block:
        start = n_cols + 1
        bold = writer.book.add_format({"bold": True})
        for j, header in enumerate(("column", UNIT_LEVEL, "semantics")):
            worksheet.write_string(0, start + j, header, bold)
        for i, line in enumerate(block, start=1):
            for j, text in enumerate(line):
                worksheet.write_string(i, start + j, text)


def units_sheet_row(sheet_name: str, raw_key: str, df: pd.DataFrame,
                    measures: list[ColumnMeta | None]) -> dict[str, str]:
    """The ``Units`` sheet row of one data sheet."""
    present = [m for m in measures if m is not None]
    units = [display_unit(m) for m in present]
    unit = summarise_units(units)
    if len(set(units)) > 1:
        unit += " (by column)"
    semantics = "; ".join(sorted({m.semantics.value for m in present})) if present else "dimension"
    return {"sheet": sheet_name, "table": raw_key, UNIT_LEVEL: unit, "semantics": semantics,
            "description": OUTPUT_DESCRIPTION.get(raw_key, "")}


def _metadata_rows(sheet_name: str, raw_key: str, df: pd.DataFrame) -> list[dict]:
    """The ``_output_metadata`` rows of one data sheet.  In a scenario comparison the scenario of a column is
    given in its own ``scenario`` cell and ``column`` holds the column's own label."""
    rows = output_metadata_rows(raw_key, list(df.columns))
    scen_pos = _scenario_level(raw_key, df.columns.nlevels, list(df.columns.names)
                               if any(n is not None for n in df.columns.names) else None)
    if scen_pos is None or df.columns.nlevels < 2:
        return [{"sheet": sheet_name, **r} for r in rows]
    by_name = {str(col): col for col in df.columns}
    out = []
    for r in rows:
        col = by_name[r["column"]]
        rest = col[:scen_pos] + col[scen_pos + 1:]
        out.append({"sheet": sheet_name, **r, "scenario": str(col[scen_pos]),
                    "column": str(rest[0]) if len(rest) == 1 else str(rest)})
    return out


def _format_list_sheet(worksheet, columns: list[str], widths: dict[str, int]) -> None:
    """Readable column widths and a frozen header row on a list sheet (``Units``, ``_output_metadata``)."""
    for j, name in enumerate(columns):
        worksheet.set_column(j, j, widths.get(name, 16))
    worksheet.freeze_panes(1, 0)


def write_excel_with_metadata(excel_path, sheets) -> None:
    """Write ``sheets``, a list of ``(sheet_name, raw_output_key, df)``, to the workbook ``excel_path``.

    The workbook opens on the ``Units`` sheet, then ``_output_metadata`` (one row per value column with its
    unit, semantics, tooltip and formula), then the data sheets in the given order.  The sheet names must not
    clash with :data:`RESERVED_SHEET_NAMES` (start the ``used`` set of :func:`excel_sheet_name` from
    :func:`reserved_sheet_names`).  See the module docstring for the layout of a data sheet.
    """
    prepared: list[tuple[str, str, pd.DataFrame, list[ColumnMeta | None]]] = []
    meta_rows: list[dict] = []
    units_rows: list[dict] = []
    for sheet_name, raw_key, df in sheets:
        if sheet_name.lower() in RESERVED_SHEET_NAMES:
            raise ValueError(f"The sheet name '{sheet_name}' is used by the workbook itself.")
        df = _as_frame(df)
        measures = _measures(raw_key, df)
        prepared.append((sheet_name, raw_key, df, measures))
        meta_rows.extend(_metadata_rows(sheet_name, raw_key, df))
        units_rows.append(units_sheet_row(sheet_name, raw_key, df, measures))

    with pd.ExcelWriter(excel_path, engine="xlsxwriter") as writer:
        pd.DataFrame(units_rows, columns=UNITS_SHEET_COLUMNS).to_excel(
            writer, sheet_name=UNITS_SHEET, index=False)
        _format_list_sheet(writer.sheets[UNITS_SHEET], UNITS_SHEET_COLUMNS, _UNITS_WIDTHS)
        if meta_rows:
            columns = (COMPARISON_METADATA_COLUMNS if any("scenario" in r for r in meta_rows)
                       else METADATA_COLUMNS)
            pd.DataFrame(meta_rows, columns=columns).to_excel(
                writer, sheet_name=METADATA_SHEET, index=False)
            _format_list_sheet(writer.sheets[METADATA_SHEET], columns, _METADATA_WIDTHS)
        tables: set[str] = set()
        for sheet_name, raw_key, df, measures in prepared:
            if df.columns.nlevels >= 2:
                _write_wide_sheet(writer, sheet_name, df, measures)
            else:
                _write_table_sheet(writer, sheet_name, df, measures, tables)
        units_ws = writer.sheets[UNITS_SHEET]
        link = writer.book.add_format({"font_color": "blue", "underline": 1})
        for i, (sheet_name, *_rest) in enumerate(prepared, start=1):
            target = sheet_name.replace("'", "''")
            units_ws.write_url(i, 0, f"internal:'{target}'!A1", link, string=sheet_name)
