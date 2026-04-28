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
    # period ← period_from_model (input) ∪ period_from_period_time (this module)
    pfm = _read_csv_columns(solve_data_dir.parent / "input" / "periods_available.csv")
    pfpt = _read_csv_columns(solve_data_dir / "period_from_period_time_set.csv")
    seen: dict[str, None] = {}
    for r in pfm + pfpt:
        if r and r[0]:
            seen.setdefault(r[0], None)
    _write_singles(solve_data_dir / "period_set.csv", "period",
                   list(seen.keys()))

    # periodAll = period_group ∪ period_node ∪ period_commodity ∪
    #             period_process ∪ period_solve ∪ branch
    seen = {}
    for fname in (
        "period_group.csv", "period_node.csv", "period_commodity.csv",
        "period_process.csv", "period_solve.csv",
    ):
        for r in _read_csv_columns(solve_data_dir / fname):
            if r and r[0]:
                seen.setdefault(r[0], None)
    for r in _read_csv_columns(solve_data_dir / "branch_set.csv"):
        if r and r[0]:
            seen.setdefault(r[0], None)
    _write_singles(solve_data_dir / "periodAll_set.csv", "period",
                   list(seen.keys()))

    # block = ⋃ b across node__block, process__side__block, process__block,
    #         block__period__step, overlap (twice — coarse + fine)
    seen = {}
    # entity_block.csv: (entity, block) — col 1
    for r in _read_csv_columns(solve_data_dir / "entity_block.csv"):
        if len(r) >= 2 and r[1]:
            seen.setdefault(r[1], None)
    # process_side_block.csv: (process, side, block) — col 2
    for r in _read_csv_columns(solve_data_dir / "process_side_block.csv"):
        if len(r) >= 3 and r[2]:
            seen.setdefault(r[2], None)
    # process_block.csv: (process, block) — col 1
    for r in _read_csv_columns(solve_data_dir / "process_block.csv"):
        if len(r) >= 2 and r[1]:
            seen.setdefault(r[1], None)
    # block_step_duration.csv: (block, period, step, ...) — col 0
    for r in _read_csv_columns(solve_data_dir / "block_step_duration.csv"):
        if r and r[0]:
            seen.setdefault(r[0], None)
    # overlap_set.csv: (period, b_coarse, t_coarse, b_fine, t_fine, fraction)
    # Both coarse (col 1) and fine (col 3).
    for r in _read_csv_columns(solve_data_dir / "overlap_set.csv"):
        if len(r) >= 4:
            if r[1]:
                seen.setdefault(r[1], None)
            if r[3]:
                seen.setdefault(r[3], None)
    _write_singles(solve_data_dir / "block_set.csv", "block",
                   list(seen.keys()))

    # period__timeline = {(d, tl) : ∃ (s, d, tb) in solve_period_timeset
    #     where s in solve_current AND (tb, tl) in timeset__timeline}
    spt = _read_csv_columns(solve_data_dir.parent / "input" / "timesets_in_use.csv")
    cur_solve = _read_csv_columns(solve_data_dir / "solve_current.csv")
    cur_solve_set = frozenset(r[0] for r in cur_solve if r and r[0])
    tt = _read_csv_columns(solve_data_dir.parent / "input" / "timesets__timeline.csv")
    # tb → list of tl
    tb_to_tl: dict[str, list[str]] = {}
    for r in tt:
        if len(r) >= 2 and r[0] and r[1]:
            tb_to_tl.setdefault(r[0], []).append(r[1])
    pt_seen: dict[tuple[str, str], None] = {}
    for r in spt:
        if len(r) >= 3 and r[0] in cur_solve_set:
            d = r[1]
            tb = r[2]
            for tl in tb_to_tl.get(tb, ()):
                pt_seen.setdefault((d, tl), None)
    _write_tuples(solve_data_dir / "period__timeline_set.csv",
                  ("period", "timeline"),
                  list(pt_seen.keys()))

    # dt_realize_dispatch = if 'output_horizon' in enable_optional_outputs
    #                       then dt else dt_realize_dispatch_input
    enable = _read_csv_columns(solve_data_dir / "enable_optional_outputs.csv")
    enable_set = frozenset(r[0] for r in enable if r and r[0])
    if "output_horizon" in enable_set:
        # dt has columns (period, time) from steps_in_use.csv
        rows = _read_csv_columns(solve_data_dir / "steps_in_use.csv")
    else:
        # dt_realize_dispatch_input ← solve_data/realized_dispatch.csv (period, time)
        rows = _read_csv_columns(solve_data_dir / "realized_dispatch.csv")
    drd_pairs = _project_columns(rows, (0, 1))
    _write_tuples(
        solve_data_dir / "dt_realize_dispatch_set.csv",
        ("period", "time"), drd_pairs,
    )
    # d_realized_period = setof d from dt_realize_dispatch
    drp_seen: dict[str, None] = {}
    for d, _t in drd_pairs:
        drp_seen.setdefault(d, None)
    _write_singles(solve_data_dir / "d_realized_period_set.csv", "period",
                   list(drp_seen.keys()))
    # d_realize_dispatch_or_invest = d_realized_period ∪ d_realize_invest.
    # d_realize_invest is loaded by mod from
    # solve_data/realized_invest_periods_of_current_solve.csv (single
    # `period` column). The differently-named solve_data/d_realize_invest.csv
    # also exists but it's a (solve, period) two-column file that's unrelated
    # — column 0 there is the solve name, not a period; reading it would
    # poison this set with a solve name.
    drealize = _read_csv_columns(
        solve_data_dir / "realized_invest_periods_of_current_solve.csv"
    )
    union_seen: dict[str, None] = dict(drp_seen)
    for r in drealize:
        if r and r[0]:
            union_seen.setdefault(r[0], None)
    _write_singles(
        solve_data_dir / "d_realize_dispatch_or_invest_set.csv", "period",
        list(union_seen.keys()),
    )

    # dt_non_anticipativity = dt_realize_dispatch_input ∪ dt_fix_storage_timesteps
    a = _read_csv_columns(solve_data_dir / "realized_dispatch.csv")
    b = _read_csv_columns(solve_data_dir / "fix_storage_timesteps.csv")
    dtna_seen: dict[tuple[str, str], None] = {}
    for r in a + b:
        if len(r) >= 2 and r[0] and r[1]:
            dtna_seen.setdefault((r[0], r[1]), None)
    _write_tuples(solve_data_dir / "dt_non_anticipativity_set.csv",
                  ("period", "time"),
                  list(dtna_seen.keys()))

    # pdt_uptime / pdt_downtime = setof (p, d, t) from uptime_lookback /
    # downtime_lookback. Both have (process, period, time, period_back,
    # time_back) — project (process, period, time).
    for src, dst in (
        ("uptime_lookback.csv", "pdt_uptime_set.csv"),
        ("downtime_lookback.csv", "pdt_downtime_set.csv"),
    ):
        rows = _read_csv_columns(solve_data_dir / src)
        triples = _project_columns(rows, (0, 1, 2))
        _write_tuples(solve_data_dir / dst,
                      ("process", "period", "time"), triples)

    # cnd_ladder family — uses period_in_use (per-solve) crossed with
    # commodity_node × commodity_with_ladder*.
    cn_rows = _read_csv_columns(solve_data_dir.parent / "input" / "commodity__node.csv")
    cn_pairs = [(r[0], r[1]) for r in cn_rows if len(r) >= 2 and r[0] and r[1]]
    pin_use = [r[0] for r in _read_csv_columns(solve_data_dir / "period_in_use_set.csv")
               if r and r[0]]
    with_ladder = frozenset(
        r[0] for r in _read_csv_columns(solve_data_dir / "commodity_with_ladder.csv")
        if r and r[0]
    )
    with_ladder_cum = frozenset(
        r[0] for r in _read_csv_columns(solve_data_dir / "commodity_with_ladder_cumulative.csv")
        if r and r[0]
    )
    with_ladder_ann = frozenset(
        r[0] for r in _read_csv_columns(solve_data_dir / "commodity_with_ladder_annual.csv")
        if r and r[0]
    )
    cum_tiers = _read_csv_columns(solve_data_dir.parent / "input" / "commodity_ladder_cumulative.csv")
    ann_tiers = _read_csv_columns(solve_data_dir / "commodity__tier_ann.csv")
    tiers_for_cum: dict[str, list[str]] = {}
    for r in cum_tiers:
        if len(r) >= 2 and r[0] and r[1]:
            tiers_for_cum.setdefault(r[0], []).append(r[1])
    tiers_for_ann: dict[str, list[str]] = {}
    for r in ann_tiers:
        if len(r) >= 2 and r[0] and r[1]:
            tiers_for_ann.setdefault(r[0], []).append(r[1])
    # cnd_ladder: (c, n, d) where c ∈ commodity_with_ladder
    cnd_rows = [
        (c, n, d) for (c, n) in cn_pairs for d in pin_use
        if c in with_ladder
    ]
    _write_tuples(solve_data_dir / "cnd_ladder_set.csv",
                  ("commodity", "node", "period"),
                  list(dict.fromkeys(cnd_rows)))
    # cndi_ladder_cum: (c, n, d, i) where c ∈ ladder_cumulative AND (c, i) ∈ tier_cum
    cum_rows_out: list[tuple[str, str, str, str]] = []
    for (c, n) in cn_pairs:
        if c not in with_ladder_cum:
            continue
        tiers = tiers_for_cum.get(c, ())
        for d in pin_use:
            for i in tiers:
                cum_rows_out.append((c, n, d, i))
    _write_tuples(solve_data_dir / "cndi_ladder_cum_set.csv",
                  ("commodity", "node", "period", "tier"),
                  list(dict.fromkeys(cum_rows_out)))
    ann_rows_out: list[tuple[str, str, str, str]] = []
    for (c, n) in cn_pairs:
        if c not in with_ladder_ann:
            continue
        tiers = tiers_for_ann.get(c, ())
        for d in pin_use:
            for i in tiers:
                ann_rows_out.append((c, n, d, i))
    _write_tuples(solve_data_dir / "cndi_ladder_ann_set.csv",
                  ("commodity", "node", "period", "tier"),
                  list(dict.fromkeys(ann_rows_out)))
    # cndi_ladder = cum ∪ ann
    union: dict[tuple[str, str, str, str], None] = {}
    for r in cum_rows_out + ann_rows_out:
        union.setdefault(r, None)
    _write_tuples(solve_data_dir / "cndi_ladder_set.csv",
                  ("commodity", "node", "period", "tier"),
                  list(union.keys()))

    # dtdt_next = setof (d_prev, t_prev_solve, d, t) from dtttdt (step_previous.csv)
    # dtttdt cols: period, time, t_previous, t_previous_within_timeset,
    #             period_previous, t_previous_within_solve
    # → project (period_previous, t_previous_within_solve, period, time)
    rows = _read_csv_columns(solve_data_dir / "step_previous.csv")
    quads = _project_columns(rows, (4, 5, 0, 1))
    _write_tuples(solve_data_dir / "dtdt_next_set.csv",
                  ("period_prev", "time_prev_solve", "period", "time"),
                  quads)

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
