"""Reserve-method partition sets.

Migrated:
    flextool.mod:113  reserve = setof r from reserve__upDown__group__method
    flextool.mod:1178 reserve__upDown__group__method_timeseries
    flextool.mod:1183 reserve__upDown__group__method_dynamic
    flextool.mod:1188 reserve__upDown__group__method_n_1

Each method partition selects (r, ud, g, method) rows where the method
matches one of a small literal set.
"""
from __future__ import annotations

import csv
from pathlib import Path
from collections.abc import Iterable


_TIMESERIES_METHODS = frozenset((
    "timeseries_only", "timeseries_and_dynamic",
    "timeseries_and_large_failure", "all",
))
_DYNAMIC_METHODS = frozenset((
    "dynamic_only", "timeseries_and_dynamic",
    "dynamic_and_large_failure", "all",
))
_N_1_METHODS = frozenset((
    "large_failure_only", "timeseries_and_large_failure",
    "dynamic_and_large_failure", "all",
))


def _read_quad(path: Path) -> list[tuple[str, str, str, str]]:
    if not path.exists():
        return []
    out: list[tuple[str, str, str, str]] = []
    with path.open() as fh:
        reader = csv.reader(fh)
        next(reader, None)
        for row in reader:
            if len(row) >= 4 and all(row[:4]):
                out.append((row[0], row[1], row[2], row[3]))
    return out


def _write_singles(path: Path, header: str, rows: Iterable[str]) -> None:
    path.write_text(header + "\n" + "".join(r + "\n" for r in rows))


def _write_quads(path: Path, header: tuple[str, str, str, str],
                 rows: Iterable[tuple[str, str, str, str]]) -> None:
    path.write_text(",".join(header) + "\n"
                    + "".join(",".join(r) + "\n" for r in rows))


def write_reserve_partitions(input_dir: Path, solve_data_dir: Path) -> None:
    rows = _read_quad(input_dir / "reserve__upDown__group__method.csv")
    # reserve = setof r — single column projection
    reserves = list(dict.fromkeys(r for r, _, _, _ in rows))
    _write_singles(solve_data_dir / "reserve.csv", "reserve", reserves)
    # Three method partitions — keep all 4 cols so mod's existing usage
    # of (r, ud, g, m) bindings works unchanged.
    for fname, allowed in (
        ("reserve__upDown__group__method_timeseries.csv", _TIMESERIES_METHODS),
        ("reserve__upDown__group__method_dynamic.csv",    _DYNAMIC_METHODS),
        ("reserve__upDown__group__method_n_1.csv",        _N_1_METHODS),
    ):
        filtered = [r for r in rows if r[3] in allowed]
        _write_quads(
            solve_data_dir / fname,
            ("reserve", "upDown", "group", "method"),
            list(dict.fromkeys(filtered)),
        )
