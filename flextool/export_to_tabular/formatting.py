"""Cell formatting constants and helper functions for the Excel export.

Provides fill colours, fonts, and per-layout formatting functions that match
the existing FlexTool Excel template.  All theme colours reference the Office
default colour theme shipped with openpyxl workbooks.
"""

from __future__ import annotations

from collections.abc import Collection

from openpyxl.comments import Comment
from openpyxl.formatting.rule import CellIsRule
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.styles.colors import Color
from openpyxl.utils import get_column_letter
from openpyxl.worksheet.worksheet import Worksheet

# ---------------------------------------------------------------------------
# Colour / font constants
# ---------------------------------------------------------------------------

# Header row (row 2 in constant/periodic sheets) fills
FILL_ALT_HEADER = PatternFill(
    patternType="solid", fgColor=Color(theme=9, tint=0.3999755851924192)
)
FILL_ENTITY_HEADER = PatternFill(
    patternType="solid", fgColor=Color(theme=7, tint=0.5999938962981048)
)
FILL_PARAM_HEADER = PatternFill(
    patternType="solid", fgColor=Color(theme=4, tint=0.5999938962981048)
)

# Data row fills (rows 3+ in constant/periodic sheets)
FILL_ALT_DATA = PatternFill(
    patternType="solid", fgColor=Color(theme=9, tint=0.5999938962981048)
)
FILL_ENTITY_DATA = PatternFill(
    patternType="solid", fgColor=Color(theme=7, tint=0.7999816888943144)
)

# Timeseries-specific fills
FILL_TIME_HEADER = PatternFill(
    patternType="solid", fgColor=Color(theme=0, tint=-0.249977111117893)
)
FILL_TIME_DATA = PatternFill(
    patternType="solid", fgColor=Color(theme=0, tint=-0.1499984740745262)
)

# Description/data-type label cells (row 1-2, definition column) – medium grey, black font
FILL_DESC_ROW = PatternFill(
    patternType="solid", fgColor=Color(theme=0, tint=-0.3499862666707358)
)
FONT_DESC_ROW = Font(color=Color(theme=1, tint=0.0))

# Description/data-type data cells (row 1-2, parameter columns) – lighter grey, black font
FILL_DESC_DATA = PatternFill(
    patternType="solid", fgColor=Color(theme=0, tint=-0.249977111117893)
)
FONT_DESC_DATA = Font(color=Color(theme=1, tint=0.0))

# Definition column background (darker grey) — conveys "do not fill here"
FILL_DEF_COL = PatternFill(
    patternType="solid", fgColor=Color(theme=0, tint=-0.3499862666707358)
)

# 'navigate' label in row 1 of each sheet (blue link colour)
FONT_NAVIGATE_LINK = Font(color=Color(theme=10, tint=0.0))

# Parameter *label* cell (e.g. the word "parameter" in timeseries sheets)
FILL_PARAM_LABEL = PatternFill(
    patternType="solid", fgColor=Color(theme=4, tint=0.3999755851924192)
)

# Parameter data columns — slightly darker grey than default white
FILL_PARAM_DATA = PatternFill(
    patternType="solid", fgColor=Color(theme=0, tint=-0.0499893185216834)
)

# Every other parameter column in the data rows (next to FILL_PARAM_DATA, F2F2F2).
# Conditional-format fills take the colour from bgColor in Excel and fgColor in
# LibreOffice, so both are set.
FILL_PARAM_DATA_ALT = PatternFill(patternType="solid", fgColor="E4E9F1", bgColor="E4E9F1")
# Every other data column of a transposed (time series, array, stochastic) sheet; the others stay white.
FILL_DATA_ALT_TRANSPOSED = PatternFill(patternType="solid", fgColor="EEF1F6", bgColor="EEF1F6")

# Description row: pale yellow (the colour of notes), apart from the grey data-type row.
FILL_DESCRIPTION = PatternFill(patternType="solid", fgColor="FFF2CC")
FILL_DESCRIPTION_LABEL = PatternFill(patternType="solid", fgColor="FFE699")
DESCRIPTION_ROW_HEIGHT = 60  # points: four lines of Calibri 11
ALIGN_DESCRIPTION = Alignment(wrap_text=True, vertical="top")

# Index dimension columns (period, time, constraint) — light green
FILL_INDEX_HEADER = PatternFill(
    patternType="solid", fgColor="C6EFCE"
)
FILL_INDEX_DATA = PatternFill(
    patternType="solid", fgColor="E2EFDA"
)


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------


def format_constant_sheet(
    ws: Worksheet,
    n_entity_cols: int,
    n_extra_cols: int = 0,
    has_entity_alt: bool = True,
) -> None:
    """Apply formatting to a constant-layout sheet.

    Column layout (1-based):
        1                       alternative
        2 .. 1+n_entity_cols    entity dimension columns
        next n_extra_cols       extra columns (e.g. left_node, right_node)
        remaining               parameter columns (+ optional Entity Alternative)

    Row layout:
        1   description / navigate
        2   headers
        3+  data
    """
    max_row = ws.max_row
    max_col = ws.max_column

    1 + n_entity_cols + n_extra_cols + 1  # 1-based

    for col in range(1, max_col + 1):
        # --- Row 1: description row ---
        cell_r1 = ws.cell(row=1, column=col)
        if col == 1:
            # 'navigate' cell gets link font, no dark fill
            pass
        elif cell_r1.value is not None:
            cell_r1.fill = FILL_DESC_ROW
            cell_r1.font = FONT_DESC_ROW

        # --- Row 2: header row ---
        cell_r2 = ws.cell(row=2, column=col)
        if col == 1:
            cell_r2.fill = FILL_ALT_HEADER
        elif col <= 1 + n_entity_cols + n_extra_cols:
            cell_r2.fill = FILL_ENTITY_HEADER
        else:
            cell_r2.fill = FILL_PARAM_HEADER

        # --- Rows 3+: data rows ---
        for row in range(3, max_row + 1):
            cell = ws.cell(row=row, column=col)
            if col == 1:
                cell.fill = FILL_ALT_DATA
            elif col <= 1 + n_entity_cols + n_extra_cols:
                cell.fill = FILL_ENTITY_DATA
            # parameter data cells: no fill (default white)


def format_periodic_sheet(
    ws: Worksheet,
    n_entity_cols: int,
    n_extra_cols: int = 0,
) -> None:
    """Apply formatting to a periodic-layout sheet.

    Same as constant but:
    - The period / index column after entity columns gets entity colouring.
    - No Entity Alternative column.

    Column layout (1-based):
        1                       alternative
        2 .. 1+n_entity_cols    entity dimension columns
        next n_extra_cols       extra columns
        remaining               parameter columns
    """
    max_row = ws.max_row
    max_col = ws.max_column

    entity_end_col = 1 + n_entity_cols + n_extra_cols  # last entity-ish column

    for col in range(1, max_col + 1):
        # --- Row 1: description row ---
        cell_r1 = ws.cell(row=1, column=col)
        if col == 1:
            pass  # navigate link
        elif cell_r1.value is not None:
            cell_r1.fill = FILL_DESC_ROW
            cell_r1.font = FONT_DESC_ROW

        # --- Row 2: header row ---
        cell_r2 = ws.cell(row=2, column=col)
        if col == 1:
            cell_r2.fill = FILL_ALT_HEADER
        elif col <= entity_end_col:
            cell_r2.fill = FILL_ENTITY_HEADER
        else:
            cell_r2.fill = FILL_PARAM_HEADER

        # --- Rows 3+: data rows ---
        for row in range(3, max_row + 1):
            cell = ws.cell(row=row, column=col)
            if col == 1:
                cell.fill = FILL_ALT_DATA
            elif col <= entity_end_col:
                cell.fill = FILL_ENTITY_DATA
            # parameter data cells: no fill


def format_timeseries_sheet(
    ws: Worksheet,
    n_header_rows: int,
) -> None:
    """Apply formatting to a transposed timeseries-layout sheet.

    Row layout (1-based):
        1                   alternative row   (B='alternative', C+= alt values)
        2                   parameter row     (B='parameter',   C+= param names)
        3 .. n_header_rows-1  entity dim rows (B=dim name,      C+= entity names)
        n_header_rows       last entity/time  (A='time', B=dim, C+= entity names)
        n_header_rows+1 ..  data rows         (A=time values,   C+= numeric data)

    Column layout:
        A   navigate / empty / 'time' label / time values
        B   row labels ('alternative', 'parameter', entity dim names)
        C+  data columns
    """
    max_row = ws.max_row
    max_col = ws.max_column

    for row in range(1, max_row + 1):
        for col in range(1, max_col + 1):
            cell = ws.cell(row=row, column=col)

            if col == 1:
                # Column A
                if row == 1:
                    # navigate cell – handled by add_navigate_link
                    pass
                elif row < n_header_rows:
                    # empty cells above the time row
                    pass
                elif row == n_header_rows:
                    cell.fill = FILL_TIME_HEADER
                else:
                    # time data values
                    cell.fill = FILL_TIME_DATA

            elif col == 2:
                # Column B – row labels
                if row == 1:
                    cell.fill = FILL_ALT_HEADER
                elif row == 2:
                    cell.fill = FILL_PARAM_LABEL
                elif row <= n_header_rows:
                    cell.fill = FILL_ENTITY_HEADER
                # data rows in col B: no fill

            else:
                # Columns C+ – data columns
                if row == 1:
                    cell.fill = FILL_ALT_DATA
                elif row == 2:
                    cell.fill = FILL_PARAM_HEADER
                elif row <= n_header_rows:
                    cell.fill = FILL_ENTITY_DATA
                # data rows: no fill


def format_link_sheet(ws: Worksheet) -> None:
    """Apply formatting to a link-only (relationship) sheet.

    Row 1: entity headers.  Rows 2+: entity data.
    """
    max_row = ws.max_row
    max_col = ws.max_column

    for col in range(1, max_col + 1):
        ws.cell(row=1, column=col).fill = FILL_ENTITY_HEADER
        for row in range(2, max_row + 1):
            ws.cell(row=row, column=col).fill = FILL_ENTITY_DATA


def add_navigate_link(ws: Worksheet) -> None:
    """Write a 'navigate' hyperlink in cell A1."""
    cell = ws.cell(row=1, column=1)
    cell.value = "navigate"
    cell.hyperlink = "#navigate!A1"
    cell.font = FONT_NAVIGATE_LINK


# ---------------------------------------------------------------------------
# v2 format formatting helpers
# ---------------------------------------------------------------------------


def format_constant_sheet_v2(
    ws: Worksheet,
    n_entity_cols: int,
    n_extra_cols: int = 0,
    def_col: int = 3,
    index_cols: set[int] | None = None,
    last_data_col: int = 0,
    def_row: int = 3,
) -> None:
    """Apply formatting to a v2 constant-layout sheet.

    Row layout (1-based):
        1 .. def_row-1      metadata rows (description, data type, optional
                            default — same dark/light fill pair regardless
                            of which slot they occupy)
        def_row             definition row (alternative, entity: X, parameter,
                            param names)
        def_row+1 ..        data

    Column layout (1-based):
        1                       alternative
        2 .. 1+n_entity_cols    entity dimension columns
        next n_extra_cols       extra columns (direction, index, etc.)
        def_col                 definition column (description/data type/parameter)
        def_col+1 ..            parameter columns

    Args:
        index_cols: 1-based column numbers that are index dimensions
            (period, time, constraint) — these get a light green fill
            instead of entity fill.
        def_row: Row of the definition / parameter-name header.  Each
            metadata row above it (description, data type, optionally
            default) shares the same fill pair.
    """
    if index_cols is None:
        index_cols = set()

    max_row = ws.max_row
    max_col = last_data_col if last_data_col > 0 else ws.max_column

    # The description row (labelled ``description`` in the definition column) wraps to four lines; a units row
    # (``unit``) has the same colour on one line.
    labels = {r: ws.cell(row=r, column=def_col).value for r in range(1, def_row)}
    description_rows = {r for r, label in labels.items() if label == "description"}
    yellow_rows = description_rows | {r for r, label in labels.items() if label == "unit"}
    for r in description_rows:
        ws.row_dimensions[r].height = DESCRIPTION_ROW_HEIGHT

    for col in range(1, max_col + 1):
        is_index = col in index_cols

        # --- Metadata rows: 1..(def_row-1) ---
        # Each one paints a label cell at def_col and data cells from
        # def_col+1 onward.  Row 1 column 1 is reserved for the navigate
        # hyperlink and stays unfilled.
        for meta_row in range(1, def_row):
            cell = ws.cell(row=meta_row, column=col)
            if col == 1 and meta_row == 1:
                continue  # navigate link
            is_yellow = meta_row in yellow_rows
            if col == def_col and cell.value is not None:
                cell.fill = FILL_DESCRIPTION_LABEL if is_yellow else FILL_DESC_ROW
                cell.font = FONT_DESC_ROW
            elif col > def_col and cell.value is not None:
                cell.fill = FILL_DESCRIPTION if is_yellow else FILL_DESC_DATA
                cell.font = FONT_DESC_DATA
                if meta_row in description_rows:
                    cell.alignment = ALIGN_DESCRIPTION

        # --- Definition row: def_row ---
        cell_def = ws.cell(row=def_row, column=col)
        if col == 1:
            cell_def.fill = FILL_ALT_HEADER
        elif is_index:
            cell_def.fill = FILL_INDEX_HEADER
        elif col < def_col:
            cell_def.fill = FILL_ENTITY_HEADER
        elif col == def_col:
            cell_def.fill = FILL_PARAM_LABEL
        else:
            cell_def.fill = FILL_PARAM_HEADER

    # --- Data rows: from def_row+1 onward ---
    # Conditional formatting paints a 100-row buffer past the actual data.
    data_start = def_row + 1
    if max_row >= data_start:
        _always = ['"§§§NEVER§§§"']
        data_range_end = max_row + 100

        for col in range(1, (last_data_col if last_data_col > 0 else max_col) + 1):
            is_index = col in index_cols
            cl = get_column_letter(col)
            rng = f"{cl}{data_start}:{cl}{data_range_end}"

            if col == 1:
                fill = FILL_ALT_DATA
            elif is_index:
                fill = FILL_INDEX_DATA
            elif col == def_col:
                fill = FILL_DEF_COL
            elif col < def_col:
                fill = FILL_ENTITY_DATA
            elif col > def_col:
                # Every other column right of the definition column gets the second shade.
                fill = FILL_PARAM_DATA if (col - def_col) % 2 == 1 else FILL_PARAM_DATA_ALT
            else:
                continue

            ws.conditional_formatting.add(
                rng, CellIsRule(operator="notEqual", formula=_always, fill=fill)
            )


def format_periodic_sheet_v2(
    ws: Worksheet,
    n_entity_cols: int,
    n_extra_cols: int = 0,
    def_col: int = 3,
    index_cols: set[int] | None = None,
    last_data_col: int = 0,
    def_row: int = 3,
) -> None:
    """Apply formatting to a v2 periodic-layout sheet.

    Same structure as format_constant_sheet_v2 but for periodic sheets.
    """
    format_constant_sheet_v2(
        ws, n_entity_cols, n_extra_cols, def_col, index_cols,
        last_data_col, def_row=def_row,
    )


def format_timeseries_sheet_v2(
    ws: Worksheet,
    n_header_rows: int,
    single_param: bool = False,
    row_types: dict[int, str] | None = None,
    last_data_col: int = 0,
) -> None:
    """Apply formatting to a v2 transposed timeseries-layout sheet.

    Formatting is based on the *row_types* dict which maps 1-based row
    numbers to their semantic type: 'description', 'data_type', 'alternative',
    'parameter', 'entity', 'param_info' (grey single-param row).

    If *row_types* is not provided, it is inferred from *single_param*.
    """
    max_col = ws.max_column

    if row_types is None:
        row_types = {}
        if single_param:
            row_types = {1: "param_info", 2: "alternative", 3: "entity"}
        else:
            row_types = {1: "alternative", 2: "parameter", 3: "entity"}

    # Map row type to (label_fill, data_fill) for col 2 and cols 3+
    type_fills = {
        "description": (FILL_DESC_ROW, FILL_DESC_DATA, FONT_DESC_ROW, FONT_DESC_DATA),
        "data_type":   (FILL_DESC_ROW, FILL_DESC_DATA, FONT_DESC_ROW, FONT_DESC_DATA),
        "default":     (FILL_DESC_ROW, FILL_DESC_DATA, FONT_DESC_ROW, FONT_DESC_DATA),
        "alternative": (FILL_ALT_HEADER, FILL_ALT_DATA, None, None),
        "parameter":   (FILL_PARAM_LABEL, FILL_PARAM_HEADER, None, None),
        "entity":      (FILL_ENTITY_HEADER, FILL_ENTITY_DATA, None, None),
        "param_info":  (FILL_DEF_COL, FILL_DEF_COL, None, None),
    }

    # Format header rows (per-cell — only a few rows, only data columns)
    header_col_end = last_data_col if last_data_col > 0 else max_col
    for row in range(1, n_header_rows + 1):
        rtype = row_types.get(row)

        for col in range(1, header_col_end + 1):
            cell = ws.cell(row=row, column=col)

            if col == 1:
                if row == n_header_rows:
                    cell.fill = FILL_INDEX_HEADER
                # else: navigate link or empty — no fill

            elif rtype in type_fills:
                label_fill, data_fill, label_font, data_font = type_fills[rtype]
                if col == 2:
                    cell.fill = label_fill
                    if label_font:
                        cell.font = label_font
                else:
                    cell.fill = data_fill
                    if data_font:
                        cell.font = data_font

    # Note: Column A data fill is NOT applied here — each writer adds its own
    # (FILL_TIME_DATA for timeseries, FILL_INDEX_DATA for array sheets).


def format_link_sheet_v2(ws: Worksheet) -> None:
    """Apply formatting to a v2 link-only (relationship) sheet.

    Row 1: navigate
    Row 2: entity definition row (entity: X, Y | dim1 | dim2)
    Rows 3+: entity data
    """
    max_row = ws.max_row
    max_col = ws.max_column

    for col in range(1, max_col + 1):
        # Row 2: entity header row
        ws.cell(row=2, column=col).fill = FILL_ENTITY_HEADER
        # Rows 3+: entity data
        for row in range(3, max_row + 1):
            ws.cell(row=row, column=col).fill = FILL_ENTITY_DATA


def auto_column_width(ws: Worksheet, min_width: float = 22, max_width: float = 40) -> None:
    """Size every column to its longest value, between *min_width* and *max_width*.

    Used by the navigate sheet and the v1 writers; the v2 sheets use :func:`fit_column_widths`.
    """
    for col_idx in range(1, (ws.max_column or 0) + 1):
        longest = min_width
        for row_idx in range(1, min((ws.max_row or 0) + 1, 200)):
            value = ws.cell(row=row_idx, column=col_idx).value
            if value is not None:
                longest = max(longest, len(str(value)) + 1)
        ws.column_dimensions[get_column_letter(col_idx)].width = min(longest, max_width)


# Width bounds (in characters of Calibri 11) per column role, see fit_column_widths.
WIDTH_BOUNDS: dict[str, tuple[float, float]] = {
    "key": (8, 32),          # alternative, entity, dimension, filter and index columns
    "definition": (10, 16),  # the definition column (description / data type / parameter labels)
    "data": (8, 30),         # parameter columns and the data columns of transposed sheets
    "scenario": (10, 32),    # scenario sheet columns
}
_WIDTH_PADDING = 2
# Excel's General number format shows at most 11 characters.
_NUMBER_DISPLAY_CHARS = 11


def _display_len(value: object) -> int:
    """Characters a value needs on screen; 0 for texts meant to run over into the next cells."""
    if value is None:
        return 0
    if isinstance(value, bool):
        return 5
    if isinstance(value, (int, float)):
        return min(len(repr(value)), _NUMBER_DISPLAY_CHARS)
    text = str(value)
    lowered = text.lower()
    # INFO rows and single-parameter ``parameter: x | data type: y | ...`` triplets are long reader labels.
    if lowered.startswith("info:") or lowered.startswith("parameter:"):
        return 0
    return max(len(line) for line in text.split("\n"))


def fit_column_widths(
    ws: Worksheet,
    columns: dict[int, tuple[str, int]],
    skip_rows: Collection[int] = (),
    skip_cells: Collection[tuple[int, int]] = (),
) -> None:
    """Size columns to their content: ``clamp(longest + 2, lo, hi)`` with the bounds of the column's role.

    Args:
        columns: 1-based column -> (role in WIDTH_BOUNDS, first row whose cells count).
        skip_rows: Rows that never count (description and data-type rows).
        skip_cells: (row, column) cells that never count (reader labels such as ``filter: {...}``).
    """
    skip_rows = set(skip_rows)
    skip_cells = set(skip_cells)
    longest: dict[int, int] = dict.fromkeys(columns, 0)
    # ``_cells`` holds only the written cells (ws.cell would create the empty ones).
    for (row, col), cell in ws._cells.items():
        spec = columns.get(col)
        if spec is None or row < spec[1] or row in skip_rows or (row, col) in skip_cells:
            continue
        longest[col] = max(longest[col], _display_len(cell.value))
    for col, (role, _first_row) in columns.items():
        lo, hi = WIDTH_BOUNDS[role]
        ws.column_dimensions[get_column_letter(col)].width = max(lo, min(hi, longest[col] + _WIDTH_PADDING))


def freeze_panes_at(ws: Worksheet, first_scrolling_row: int, first_scrolling_col: int) -> None:
    """Freeze the rows above *first_scrolling_row* and the columns left of *first_scrolling_col*."""
    ws.freeze_panes = f"{get_column_letter(first_scrolling_col)}{first_scrolling_row}"


def shade_alternate_columns(
    ws: Worksheet, first_col: int, last_col: int, first_row: int, last_row: int, fill: PatternFill,
) -> None:
    """Paint *fill* on every other column from ``first_col + 1``, rows *first_row*..*last_row*.

    Conditional formatting, so no cells are created (the reader's row count stays the data's).
    """
    for col in range(first_col + 1, last_col + 1, 2):
        letter = get_column_letter(col)
        ws.conditional_formatting.add(
            f"{letter}{first_row}:{letter}{last_row}",
            CellIsRule(operator="notEqual", formula=['"§§§NEVER§§§"'], fill=fill),
        )


_NOTE_WIDTH_PX = 420
_NOTE_CHARS_PER_LINE = 62
_NOTE_LINE_PX = 18


def add_description_note(ws: Worksheet, row: int, col: int, name: str, description: str) -> None:
    """Attach a note (comment) with the parameter name and its full description, sized to the text."""
    text = f"{name}\n\n{description}"
    lines = sum(max(1, -(-len(part) // _NOTE_CHARS_PER_LINE)) for part in text.split("\n"))
    height = max(40, min(520, _NOTE_LINE_PX * lines + 12))
    ws.cell(row=row, column=col).comment = Comment(text, "FlexTool", height=height, width=_NOTE_WIDTH_PX)


def outline_column_groups(ws: Worksheet, spans: list[tuple[int, int, bool]]) -> None:
    """Put each (first column, last column, collapsed) span in an outline group of level 1.

    Each column is set on its own: openpyxl's ``column_dimensions.group`` writes one ``<col>`` element with one
    width for the whole span.  The +/- sits on the column left of each group (``summaryRight=False``), which
    carries ``collapsed`` when the group starts collapsed.
    """
    if not spans:
        return
    for first, last, collapsed in spans:
        for col in range(first, last + 1):
            dim = ws.column_dimensions[get_column_letter(col)]
            dim.outline_level = 1
            dim.hidden = collapsed
        if collapsed:
            ws.column_dimensions[get_column_letter(first - 1)].collapsed = True
    ws.sheet_properties.outlinePr.summaryRight = False
    # openpyxl writes <sheetFormatPr> before it computes the deepest outline level, so on a first save
    # outlineLevelCol would be left out and Excel would give the outline bar no room.
    ws.column_dimensions.max_outline = 1


def group_header_fill(color_hex: str) -> PatternFill:
    """A light tint of a parameter group's colour for the parameter-name cell."""
    r, g, b = (int(color_hex[i:i + 2], 16) for i in (0, 2, 4))
    tint = "".join(f"{int(c + (255 - c) * 0.55):02X}" for c in (r, g, b))
    return PatternFill(patternType="solid", fgColor=tint)


def _cell_text(ws: Worksheet, row: int, col: int) -> str:
    """The text of a written cell, without creating the cell when it is empty."""
    cell = ws._cells.get((row, col))
    return "" if cell is None or cell.value is None else str(cell.value)


def finish_standard_sheet(
    ws: Worksheet,
    *,
    def_row: int,
    def_col: int,
    last_data_col: int,
    notes: dict[int, tuple[str, str]],
    label_only_cols: set[int] | None = None,
) -> None:
    """Widths, freeze panes and description notes of a standard (constant / periodic / ladder) v2 sheet.

    The metadata rows and the key columns left of the definition column stay frozen; the definition column
    scrolls.  *label_only_cols* are key columns whose definition-row label is reader metadata (the multi-dim
    ``entity: (...)`` label, ``filter: {...}``) and does not size the column.
    """
    first_counted = def_row - 1 if _cell_text(ws, def_row - 1, def_col) == "default" else def_row
    columns: dict[int, tuple[str, int]] = {c: ("key", first_counted) for c in range(1, def_col)}
    columns[def_col] = ("definition", 1)
    for c in range(def_col + 1, last_data_col + 1):
        columns[c] = ("data", first_counted)
    fit_column_widths(ws, columns, skip_cells={(def_row, c) for c in (label_only_cols or ())})
    freeze_panes_at(ws, def_row + 1, def_col)
    for col, (name, desc) in notes.items():
        add_description_note(ws, def_row, col, name, desc)


def finish_transposed_sheet(
    ws: Worksheet,
    *,
    n_header_rows: int,
    def_col: int,
    last_data_col: int,
    skip_rows: set[int],
    notes: dict[tuple[int, int], tuple[str, str]],
) -> None:
    """Widths, freeze panes, alternating shading and notes of a transposed (time series / array / stochastic) sheet.

    The header rows and the columns up to the definition column (which labels the header rows) stay frozen.
    *skip_rows* (data-type and description rows) do not size the columns.
    """
    columns: dict[int, tuple[str, int]] = {c: ("key", 1) for c in range(1, def_col)}
    columns[def_col] = ("definition", 1)
    for c in range(def_col + 1, last_data_col + 1):
        columns[c] = ("data", 1)
    fit_column_widths(ws, columns, skip_rows=skip_rows)
    freeze_panes_at(ws, n_header_rows + 1, def_col + 1)
    shade_alternate_columns(ws, def_col + 1, last_data_col, n_header_rows + 1,
                            max(ws.max_row, n_header_rows + 1) + 100, FILL_DATA_ALT_TRANSPOSED)
    for (row, col), (name, desc) in notes.items():
        add_description_note(ws, row, col, name, desc)


def finish_link_sheet(ws: Worksheet, n_dims: int) -> None:
    """Widths and freeze panes of a link sheet: row 2 is the header; column A holds only reader labels."""
    columns: dict[int, tuple[str, int]] = {1: ("key", 1)}
    for c in range(2, 2 + n_dims):
        columns[c] = ("key", 2)
    fit_column_widths(ws, columns, skip_cells={(2, 1)})
    freeze_panes_at(ws, 3, 1)
