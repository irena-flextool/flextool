"""Period sets projected from per-class param-table CSVs.

Migrated from flextool.mod:124-127 — four single-dimension sets that
project the ``period`` column out of the ``pd_*.csv`` files written by
``input_writer.write_parameter`` for the per-period ``pdParam`` tables:

    set period_group     := setof {(g, param, d) in group__param__period}     (d);
    set period_node      := setof {(n, param, d) in node__param__period}      (d);
    set period_commodity := setof {(c, param, d) in commodity__param__period} (d);
    set period_process   := setof {(p, param, d) in process__param__period}   (d);

We project here in Python so glpsol no longer iterates the full
(entity, paramName, period) cross-product of each table just to
collect distinct periods. Order of periods follows CSV row order
(insertion-ordered via ``dict.fromkeys``).
"""
from __future__ import annotations

import csv
from pathlib import Path


def _project_period_column(csv_path: Path) -> list[str]:
    """Return the ordered, deduplicated ``period`` values from a 4-col CSV.

    Columns are ``(entity, paramName, period, value)``. We don't validate
    schema beyond reading column index 2.
    """
    if not csv_path.exists():
        return []
    seen: dict[str, None] = {}
    with csv_path.open() as fh:
        reader = csv.reader(fh)
        next(reader, None)  # header
        for row in reader:
            if len(row) >= 3 and row[2]:
                seen.setdefault(row[2], None)
    return list(seen.keys())


def write_period_param_sets(input_dir: Path, solve_data_dir: Path) -> None:
    """Write four single-column ``period`` CSVs for ``flextool.mod``.

    Reads from already-written ``input/pd_<class>.csv`` files
    (``input_writer.write_parameter`` populates them earlier in
    ``write_input``).
    """
    for source_csv, target_name in (
        ("pd_group.csv",     "period_group.csv"),
        ("pd_node.csv",      "period_node.csv"),
        ("pd_commodity.csv", "period_commodity.csv"),
        ("pd_process.csv",   "period_process.csv"),
    ):
        periods = _project_period_column(input_dir / source_csv)
        (solve_data_dir / target_name).write_text(
            "period\n" + "".join(p + "\n" for p in periods)
        )
