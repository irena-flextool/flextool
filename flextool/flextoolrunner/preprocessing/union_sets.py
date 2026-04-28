"""Union sets — multiple setof projections combined.

Migrated from flextool.mod:

    L287  set group_entity := group_process union group_node;
    L950  set process_delayed__duration :=
                  process_delay_weighted__delay_duration
            union process_delay_single__delay_duration;

These are pure unions of already-loaded 2-tuple sets — no filters.
We dedupe on union while preserving the order of first occurrence
across the input streams (mod's iteration order would visit the first
set then the second).
"""
from __future__ import annotations

import csv
from pathlib import Path


def _read_pairs(path: Path) -> list[tuple[str, str]]:
    if not path.exists():
        return []
    out: list[tuple[str, str]] = []
    with path.open() as fh:
        reader = csv.reader(fh)
        next(reader, None)
        for row in reader:
            if len(row) >= 2 and row[0] and row[1]:
                out.append((row[0], row[1]))
    return out


def _write_pairs(path: Path, header: tuple[str, str],
                 rows: list[tuple[str, str]]) -> None:
    path.write_text(
        ",".join(header) + "\n"
        + "".join(f"{a},{b}\n" for a, b in rows)
    )


def _ordered_union(*sources: list[tuple[str, str]]) -> list[tuple[str, str]]:
    seen: dict[tuple[str, str], None] = {}
    for src in sources:
        for pair in src:
            seen.setdefault(pair, None)
    return list(seen.keys())


def write_group_entity(input_dir: Path, solve_data_dir: Path) -> None:
    """flextool.mod:287 — group_process ∪ group_node, both 2-tuple sets."""
    gp = _read_pairs(input_dir / "group__process.csv")
    gn = _read_pairs(input_dir / "group__node.csv")
    _write_pairs(
        solve_data_dir / "group_entity.csv",
        ("group", "entity"),
        _ordered_union(gp, gn),
    )


def write_process_delayed__duration(input_dir: Path, solve_data_dir: Path) -> None:
    """flextool.mod:950."""
    weighted = _read_pairs(input_dir / "p_process_delay_weighted.csv")
    single = _read_pairs(input_dir / "process_delay_single.csv")
    _write_pairs(
        solve_data_dir / "process_delayed__duration.csv",
        ("process", "delay_duration"),
        _ordered_union(weighted, single),
    )
