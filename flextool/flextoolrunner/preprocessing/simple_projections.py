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


def write_def_optional_yes(input_dir: Path, solve_data_dir: Path) -> None:
    """flextool.mod:108
        set def_optional_yes := setof{(output, value) in def_optional_outputs
            : value == 'yes' && (output, 'no') not in optional_outputs} (output);
    """
    explicit = _read_csv(input_dir / "optional_outputs.csv")
    explicit_no = frozenset(r[0] for r in explicit if len(r) >= 2 and r[1] == "no")
    defaults = _read_csv(input_dir / "def_optional_outputs.csv")
    seen: dict[str, None] = {}
    for r in defaults:
        if len(r) >= 2 and r[1] == "yes" and r[0] not in explicit_no:
            seen.setdefault(r[0], None)
    (solve_data_dir / "def_optional_yes.csv").write_text(
        "output\n" + "".join(o + "\n" for o in seen.keys())
    )


def write_process_delayed(input_dir: Path, solve_data_dir: Path) -> None:
    """flextool.mod L956
        set process_delayed := setof {(p, td) in process_delayed__duration} (p);

    Reads the (already-Python-driven) process_delayed__duration set from
    solve_data — projection writes early-stage compatibility.
    """
    # The upstream is in solve_data (we wrote it in batch 3).
    rows = _read_csv(solve_data_dir / "process_delayed__duration.csv")
    seen: dict[str, None] = {}
    for r in rows:
        if r and r[0]:
            seen.setdefault(r[0], None)
    (solve_data_dir / "process_delayed.csv").write_text(
        "process\n" + "".join(p + "\n" for p in seen.keys())
    )


def write_process_side(solve_data_dir: Path) -> None:
    """flextool.mod:245  set process_side := {'source', 'sink'};

    A literal 2-element constant. We write it as a CSV for the migration's
    universal "everything passes through CSV" rule.
    """
    (solve_data_dir / "process_side.csv").write_text(
        "side\nsource\nsink\n"
    )


def write_simple_setof_projections(input_dir: Path, solve_data_dir: Path) -> None:
    """Trivial single-key setof projections from already-loaded CSVs.

    Each projects a single column out of an N-column input CSV,
    deduplicating in row order. Bundled here because each is one line
    of code and the modules would otherwise be too granular.
    """
    # solve_period (s, d) from solve_period_timeset (s, d, tb)
    rows = _read_csv(input_dir / "timesets_in_use.csv")
    seen: dict[tuple[str, str], None] = {}
    for r in rows:
        if len(r) >= 2 and r[0] and r[1]:
            seen.setdefault((r[0], r[1]), None)
    (solve_data_dir / "solve_period.csv").write_text(
        "solve,period\n" + "".join(",".join(t) + "\n" for t in seen.keys())
    )
    # timeline (tl) from timeset__timeline (tb, tl)
    rows = _read_csv(input_dir / "timesets__timeline.csv")
    seen2: dict[str, None] = {}
    for r in rows:
        if len(r) >= 2 and r[1]:
            seen2.setdefault(r[1], None)
    (solve_data_dir / "timeline.csv").write_text(
        "timeline\n" + "".join(t + "\n" for t in seen2.keys())
    )
    # timeline_steps (tl, t) from timeline__timestep__duration (tl, t, d)
    rows = _read_csv(input_dir / "timeline.csv")
    seen3: dict[tuple[str, str], None] = {}
    for r in rows:
        if len(r) >= 2 and r[0] and r[1]:
            seen3.setdefault((r[0], r[1]), None)
    (solve_data_dir / "timeline_steps.csv").write_text(
        "timeline,step\n" + "".join(",".join(t) + "\n" for t in seen3.keys())
    )
    # commodity__tier_ann (c, i) from commodity__tier__period_ann (c, i, d)
    rows = _read_csv(input_dir / "commodity_ladder_annual.csv")
    seen4: dict[tuple[str, str], None] = {}
    for r in rows:
        if len(r) >= 2 and r[0] and r[1]:
            seen4.setdefault((r[0], r[1]), None)
    (solve_data_dir / "commodity__tier_ann.csv").write_text(
        "commodity,tier\n" + "".join(",".join(t) + "\n" for t in seen4.keys())
    )
