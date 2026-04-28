"""Simple single-key projections that read one CSV and project one column.

A grab-bag of L0 leaf sets that share the same trivial shape — read a
2- or 4-column CSV, optionally filter by one column's value, project a
single column, write a 1-column CSV. Each is a one-line setof in
flextool.mod.

Migrated:
    flextool.mod:107  set optional_yes := setof{(output,value) in optional_outputs: value == 'yes'}(output);
    flextool.mod:112  set reserve__upDown__group := setof {(r, ud, g, m) in reserve__upDown__group__method : m <> 'no_reserve'} (r, ud, g);
    flextool.mod:293  set group_loss_share := setof {(g, lst) in group__loss_share_type} (g);
"""
from __future__ import annotations

import csv
from pathlib import Path


def _read_csv(path: Path) -> list[list[str]]:
    if not path.exists():
        return []
    with path.open() as fh:
        reader = csv.reader(fh)
        next(reader, None)  # header
        return [row for row in reader if any(c for c in row)]


def write_optional_yes(input_dir: Path, solve_data_dir: Path) -> None:
    rows = _read_csv(input_dir / "optional_outputs.csv")
    seen: dict[str, None] = {}
    for r in rows:
        if len(r) >= 2 and r[1] == "yes":
            seen.setdefault(r[0], None)
    (solve_data_dir / "optional_yes.csv").write_text(
        "output\n" + "".join(o + "\n" for o in seen.keys())
    )


def write_reserve_upDown_group(input_dir: Path, solve_data_dir: Path) -> None:
    """3-tuple set (reserve, upDown, group) for methods != 'no_reserve'."""
    rows = _read_csv(input_dir / "reserve__upDown__group__method.csv")
    seen: dict[tuple[str, str, str], None] = {}
    for r in rows:
        if len(r) >= 4 and r[3] != "no_reserve":
            seen.setdefault((r[0], r[1], r[2]), None)
    out = (solve_data_dir / "reserve__upDown__group.csv")
    out.write_text(
        "reserve,upDown,group\n"
        + "".join(",".join(t) + "\n" for t in seen.keys())
    )


def write_group_loss_share(input_dir: Path, solve_data_dir: Path) -> None:
    rows = _read_csv(input_dir / "group__loss_share_type.csv")
    seen: dict[str, None] = {}
    for r in rows:
        if r and r[0]:
            seen.setdefault(r[0], None)
    (solve_data_dir / "group_loss_share.csv").write_text(
        "group\n" + "".join(g + "\n" for g in seen.keys())
    )
