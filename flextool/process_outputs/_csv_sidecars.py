"""The sidecar files of the CSV result folder: ``units.csv`` (the unit of every value column) and
``datapackage.json`` (a Frictionless descriptor of every CSV file).  The CSV files themselves are written as
before; the sidecars sit next to them.

``units.csv`` has one row per value column: ``file, position, header_1 … header_n, unit, semantics``.
``position`` is the 0-based column of the file; ``header_i`` are the column's labels in the file's header rows
(empty when the file has fewer header rows).  ``summary_solve.csv`` (free form) gets one row for its costs.

When some node of the run is not MWh, a table whose unit varies by row (its node, unit, connection or group is
a row level: ``unit_capacity_ed_p``, ``nodeGroup_gdt_p``, …) gets one row per value column and row entity, and
``units.csv`` gains a ``row`` column (after ``position``) with the entity; the column's own row keeps ``row``
empty and gives the units of its rows joined.  A run whose nodes are all MWh writes no ``row`` column.
"""
from __future__ import annotations

import csv
import json
import os

from flextool.process_outputs._output_meta import Semantics, UNITLESS, display_unit, resolve_columns, row_units

UNITS_CSV = "units.csv"
DATAPACKAGE_JSON = "datapackage.json"
SUMMARY_SOLVE_CSV = "summary_solve.csv"
#: File names of the CSV folder that no output may take (lower-case, the ``used`` keys of ``_csv_filename``).
RESERVED_CSV_NAMES: dict[str, str] = {
    UNITS_CSV: "units.csv sidecar",
    DATAPACKAGE_JSON: "datapackage.json sidecar",
}
#: The minimum number of ``header_i`` columns of ``units.csv`` (more when a file has more header rows).
MIN_HEADER_COLUMNS = 3
#: The unit and semantics of the cost rows of ``summary_solve.csv``.
SUMMARY_SOLVE_UNIT = ("M CUR", "")


def reserved_csv_names(*, with_summary: bool) -> dict[str, str]:
    """A fresh ``used`` map for ``_csv_filename`` with the sidecar names (and ``summary_solve.csv`` when that file
    is written) taken."""
    used = dict(RESERVED_CSV_NAMES)
    if with_summary:
        used[SUMMARY_SOLVE_CSV] = "summary_solve"
    return used


def _labels(column) -> tuple[str, ...]:
    label = column if isinstance(column, tuple) else (column,)
    return tuple("" if v is None else str(v) for v in label)


def units_rows(output_key: str, file_name: str, header, n_index: int, measure_columns, *, ctx=None,
               rows=None) -> list[dict]:
    """The ``units.csv`` rows of one written file.

    ``header`` is the list of column labels as written (the *n_index* index columns first), ``measure_columns``
    the frame's value columns before the index was reset (they resolve the units).  *ctx* is the run's
    :class:`~flextool.process_outputs._output_meta.UnitContext` and *rows* the frame's row index: a row-keyed
    table then gets one more row per value column and row entity (with a ``row`` key)."""
    resolved = resolve_columns(output_key, measure_columns, ctx=ctx, rows=rows)
    if resolved is None:
        return []
    by_row = row_units(output_key, rows, measure_columns, ctx) if rows is not None else None
    out: list[dict] = []
    for i, meta in enumerate(resolved):
        if meta.semantics is Semantics.DIMENSION:
            continue
        pos = n_index + i
        base = {"file": file_name, "position": pos, "header": _labels(header[pos]),
                "semantics": meta.semantics.value}
        out.append({**base, "unit": display_unit(meta)})
        for entity, units in (by_row or {}).items():
            out.append({**base, "row": entity, "unit": units.get(meta.name) or UNITLESS})
    return out


def summary_solve_row() -> dict:
    unit, semantics = SUMMARY_SOLVE_UNIT
    return {"file": SUMMARY_SOLVE_CSV, "position": "", "header": (), "unit": unit, "semantics": semantics}


def _read_units_csv(path: str) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        out = []
        for rec in reader:
            headers = tuple(rec[k] for k in reader.fieldnames if k.startswith("header_"))
            while headers and headers[-1] == "":
                headers = headers[:-1]
            item = {"file": rec["file"], "position": int(rec["position"]) if rec["position"] else "",
                    "header": headers, "unit": rec["unit"], "semantics": rec["semantics"]}
            if rec.get("row"):
                item["row"] = rec["row"]
            out.append(item)
        return out


def write_units_csv(csv_dir: str, rows: list[dict], *, merge: bool = False) -> None:
    """Write ``units.csv``; with *merge*, keep the rows of the files not in *rows* from the existing file (a
    single-result write updates one table, not the whole folder)."""
    path = os.path.join(csv_dir, UNITS_CSV)
    if merge and os.path.exists(path):
        written = {r["file"] for r in rows}
        rows = [r for r in _read_units_csv(path) if r["file"] not in written] + rows
    # Stable: a column's own row first, then its row entities in the order the table has them.
    rows = sorted(rows, key=lambda r: (r["file"], r["position"] if r["position"] != "" else -1))
    n_headers = max([MIN_HEADER_COLUMNS, *(len(r["header"]) for r in rows)])
    header_cols = [f"header_{i}" for i in range(1, n_headers + 1)]
    with_row = any("row" in r for r in rows)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["file", "position", *(["row"] if with_row else []), *header_cols, "unit", "semantics"])
        for r in rows:
            labels = list(r["header"]) + [""] * (n_headers - len(r["header"]))
            row = [r.get("row", "")] if with_row else []
            writer.writerow([r["file"], r["position"], *row, *labels, r["unit"], r["semantics"]])


def write_datapackage(csv_dir: str, resources: list[dict], *, merge: bool = False) -> None:
    """Write ``datapackage.json``; with *merge*, keep the resources of the files not in *resources* from the
    existing file."""
    path = os.path.join(csv_dir, DATAPACKAGE_JSON)
    if merge and os.path.exists(path):
        written = {r["path"] for r in resources}
        try:
            with open(path, encoding="utf-8") as fh:
                old = json.load(fh).get("resources", [])
        except (OSError, ValueError):
            old = []
        resources = [r for r in old if r.get("path") not in written] + resources
    dp = {
        "name": "flextool-output-csv",
        "profile": "tabular-data-package",
        "title": "FlexTool processed outputs",
        "flextool:units": UNITS_CSV,
        "resources": sorted(resources, key=lambda r: r["path"]),
    }
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(dp, fh, indent=2)
