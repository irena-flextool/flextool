"""Per-solve preprocessing — sets and params that depend on solve_data CSVs.

These sets can't be computed at write_input time because their inputs
(``steps_in_use.csv``, ``period__branch.csv``, ``rp_weights.csv``,
``period_block_time.csv``, ``fix_storage_*.csv``, ``step_previous.csv``,
…) are written per-solve by ``orchestration.py`` and ``solve_writers.py``.

Migrated from flextool.mod:
    L41   set branch                   = setof b from period__branch
    L43   set year                     = setof y from period__year
    L34   set period_from_period_time  = setof d from period_time
    L233  set period_block             = setof (d, b) from period_block_time
    L222  set rp_base_period           = setof b from rp_base__rep
    L223  set rp_rep_period            = setof r from rp_base__rep
    L354  set dtt                      = setof (d, t, t_previous) from dtttdt
    L359  set time_in_use              = setof t from dt
    L360  set period_in_use            = setof d from dt
    L370  set complete_time_in_use     = setof t from dt_complete
    L377  set d_fix_storage_period     = setof d from dt_fix_storage_timesteps
    L381  set n_fix_storage_quantity   = setof n from ndt_fix_storage_quantity
    L382  set n_fix_storage_price      = setof n from ndt_fix_storage_price
    L383  set n_fix_storage_usage      = setof n from ndt_fix_storage_usage

All run after ``write_block_data_for_solve`` and before ``solver.run`` —
see preprocessing.solve_time.run.
"""
from __future__ import annotations

import csv
from pathlib import Path


def _read_csv_columns(path: Path) -> list[list[str]]:
    if not path.exists():
        return []
    with path.open() as fh:
        reader = csv.reader(fh)
        next(reader, None)
        return [r for r in reader if any(c for c in r)]


def _project_column(rows: list[list[str]], col_idx: int) -> list[str]:
    seen: dict[str, None] = {}
    for r in rows:
        if len(r) > col_idx and r[col_idx]:
            seen.setdefault(r[col_idx], None)
    return list(seen.keys())


def _project_columns(rows: list[list[str]], col_idxs: tuple[int, ...]) -> list[tuple]:
    seen: dict[tuple, None] = {}
    for r in rows:
        if len(r) > max(col_idxs) and all(r[i] for i in col_idxs):
            seen.setdefault(tuple(r[i] for i in col_idxs), None)
    return list(seen.keys())


def _write_singles(path: Path, header: str, rows: list[str]) -> None:
    path.write_text(header + "\n" + "".join(r + "\n" for r in rows))


def _write_tuples(path: Path, header: tuple[str, ...], rows: list[tuple]) -> None:
    path.write_text(",".join(header) + "\n"
                    + "".join(",".join(r) + "\n" for r in rows))


def write_per_solve_sets(solve_data_dir: Path) -> None:
    """Compute and write all per-solve derived sets in one pass.

    Reads upstream CSVs that ``orchestration.py`` /
    ``solve_writers.py`` / ``blocks.py`` have already written for the
    current solve.
    """
    # branch ← period__branch (period, branch)
    rows = _read_csv_columns(solve_data_dir / "period__branch.csv")
    _write_singles(
        solve_data_dir / "branch_set.csv",
        "branch",
        _project_column(rows, 1),
    )
    # year ← period__year (period, year) loaded from p_years_represented.csv
    rows = _read_csv_columns(solve_data_dir / "p_years_represented.csv")
    _write_singles(
        solve_data_dir / "year_set.csv",
        "year",
        _project_column(rows, 1),
    )
    # period_from_period_time ← period_time loaded from steps_in_timeline.csv
    rows = _read_csv_columns(solve_data_dir / "steps_in_timeline.csv")
    _write_singles(
        solve_data_dir / "period_from_period_time_set.csv",
        "period",
        _project_column(rows, 0),
    )
    # period_in_use, time_in_use ← dt loaded from steps_in_use.csv
    rows = _read_csv_columns(solve_data_dir / "steps_in_use.csv")
    _write_singles(
        solve_data_dir / "period_in_use_set.csv",
        "period",
        _project_column(rows, 0),
    )
    _write_singles(
        solve_data_dir / "time_in_use_set.csv",
        "time",
        _project_column(rows, 1),
    )
    # complete_time_in_use ← dt_complete loaded from steps_complete_solve.csv
    rows = _read_csv_columns(solve_data_dir / "steps_complete_solve.csv")
    _write_singles(
        solve_data_dir / "complete_time_in_use_set.csv",
        "time",
        _project_column(rows, 1),
    )
    # rp_base_period, rp_rep_period ← rp_base__rep loaded from rp_weights.csv
    rows = _read_csv_columns(solve_data_dir / "rp_weights.csv")
    _write_singles(
        solve_data_dir / "rp_base_period_set.csv",
        "period",
        _project_column(rows, 0),
    )
    _write_singles(
        solve_data_dir / "rp_rep_period_set.csv",
        "period",
        _project_column(rows, 1),
    )
    # period_block ← period_block_time (period, block_first, step) — project (d, b)
    rows = _read_csv_columns(solve_data_dir / "period_block_time.csv")
    _write_tuples(
        solve_data_dir / "period_block_set.csv",
        ("period", "block_first"),
        _project_columns(rows, (0, 1)),
    )
    # dtt ← dtttdt loaded from step_previous.csv (period, time, previous, ...)
    # Project (period, time, previous) — first 3 columns
    rows = _read_csv_columns(solve_data_dir / "step_previous.csv")
    _write_tuples(
        solve_data_dir / "dtt_set.csv",
        ("period", "time", "time_previous"),
        _project_columns(rows, (0, 1, 2)),
    )
    # d_fix_storage_period ← dt_fix_storage_timesteps loaded from
    # solve_data/fix_storage_timesteps.csv (period, step) — project d
    rows = _read_csv_columns(solve_data_dir / "fix_storage_timesteps.csv")
    _write_singles(
        solve_data_dir / "d_fix_storage_period_set.csv",
        "period",
        _project_column(rows, 0),
    )
    # n_fix_storage_* ← ndt_fix_storage_* loaded from fix_storage_*.csv
    # Header layout: (period, step, node, value) per the writer in
    # solve_writers.py (note the swapped node/period order — the file
    # writer comments show "[period, step, node]" via table data IN).
    for src, dst in (
        ("fix_storage_quantity.csv", "n_fix_storage_quantity_set.csv"),
        ("fix_storage_price.csv",    "n_fix_storage_price_set.csv"),
        ("fix_storage_usage.csv",    "n_fix_storage_usage_set.csv"),
    ):
        rows = _read_csv_columns(solve_data_dir / src)
        # node is column 2 (0-indexed)
        _write_singles(
            solve_data_dir / dst,
            "node",
            _project_column(rows, 2),
        )
