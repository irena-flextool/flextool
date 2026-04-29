"""Per-period calculated params — sums and projections of per-step values.

Migrated from flextool.mod (line numbers from current head; the mod
derivations are short single-line `:=` expressions):

    L368   p_timeline_duration_in_years{tl} = sum step_duration / 8760
    L1391  hours_in_period{d} = sum over (d, t) in dt of step_duration
    L1393  period_share_of_year{d} = hours_in_period[d] / 8760
    L1395  p_years_d{d} = p_period_from_solve[d]
    L1396  p_years_represented_d{d} = sum over y in year of p_years_represented[d, y]
    L1398  complete_hours_in_period{d} = sum over dt_complete with branch
    L1399  complete_period_share_of_year{d} = complete_hours_in_period[d] / 8760

All read per-solve solve_data/* CSVs and write per-solve outputs.
Float values formatted with `repr()` for round-trip-exact serialization
within Python; MPS parity uses 7-sig-fig comparison so the formula's
intermediate float ordering need not bit-match mod's evaluation.
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


def _write_keyed(path: Path, header: tuple[str, str], rows: list[tuple[str, float]]) -> None:
    path.write_text(",".join(header) + "\n"
                    + "".join(f"{k},{repr(v)}\n" for k, v in rows))


def _write_keyed_2(path: Path, header: tuple[str, str, float | str], rows: list[tuple[str, str, float]]) -> None:
    path.write_text(",".join(map(str, header)) + "\n"
                    + "".join(f"{a},{b},{repr(v)}\n" for a, b, v in rows))


def write_period_calculated_params(input_dir: Path, solve_data_dir: Path) -> None:
    """Compute and write all 7 per-period calc params for the current solve."""
    # ---- Sources --------------------------------------------------------
    # steps_in_use.csv: (period, step, step_duration)
    steps_in_use = _read_csv_columns(solve_data_dir / "steps_in_use.csv")
    step_duration: dict[tuple[str, str], float] = {}
    period_set: dict[str, None] = {}
    for r in steps_in_use:
        if len(r) >= 3 and r[0] and r[1]:
            try:
                step_duration[(r[0], r[1])] = float(r[2])
                period_set.setdefault(r[0], None)
            except ValueError:
                continue

    # steps_complete_solve.csv: (period, step, complete_step_duration)
    complete_steps = _read_csv_columns(solve_data_dir / "steps_complete_solve.csv")
    complete_step_duration: dict[tuple[str, str], float] = {}
    for r in complete_steps:
        if len(r) >= 3 and r[0] and r[1]:
            try:
                complete_step_duration[(r[0], r[1])] = float(r[2])
            except ValueError:
                continue

    # period__branch.csv: (period, branch)
    pb = _read_csv_columns(solve_data_dir / "period__branch.csv")
    branches_for_d: dict[str, list[str]] = {}
    for r in pb:
        if len(r) >= 2 and r[0] and r[1]:
            branches_for_d.setdefault(r[0], []).append(r[1])

    # input/timeline.csv: (timeline, timestep, duration)
    timeline_rows = _read_csv_columns(input_dir / "timeline.csv")
    timeline_step_duration: dict[tuple[str, str], float] = {}
    timelines: dict[str, None] = {}
    for r in timeline_rows:
        if len(r) >= 3 and r[0] and r[1]:
            try:
                timeline_step_duration[(r[0], r[1])] = float(r[2])
                timelines.setdefault(r[0], None)
            except ValueError:
                continue

    # solve_data/period_with_history.csv: (period, param) → param is the year value
    pwh = _read_csv_columns(solve_data_dir / "period_with_history.csv")
    p_period_from_solve: dict[str, float] = {}
    for r in pwh:
        if len(r) >= 2 and r[0]:
            try:
                p_period_from_solve[r[0]] = float(r[1])
            except ValueError:
                continue

    # solve_data/p_years_represented.csv: (period, years_from_solve, p_years_represented, p_years_from_solve)
    pyr = _read_csv_columns(solve_data_dir / "p_years_represented.csv")
    p_years_represented: dict[tuple[str, str], float] = {}
    years_for_period: dict[str, list[str]] = {}
    for r in pyr:
        if len(r) >= 3 and r[0] and r[1]:
            try:
                p_years_represented[(r[0], r[1])] = float(r[2])
                years_for_period.setdefault(r[0], []).append(r[1])
            except ValueError:
                continue

    # period_in_use_set.csv (Python output earlier in this same run)
    period_in_use = [r[0] for r in _read_csv_columns(
        solve_data_dir / "period_in_use_set.csv"
    ) if r and r[0]]

    # period_with_history.csv key list
    period_with_history = [r[0] for r in pwh if r and r[0]]

    # periodAll_set.csv
    periodAll = [r[0] for r in _read_csv_columns(
        solve_data_dir / "periodAll_set.csv"
    ) if r and r[0]]

    # ---- Per-row aggregations ------------------------------------------
    # Pre-aggregate sums by leading key (period or timeline) for O(N).
    sum_step_dur_by_period: dict[str, float] = {}
    for (d, _t), v in step_duration.items():
        sum_step_dur_by_period[d] = sum_step_dur_by_period.get(d, 0.0) + v

    sum_timeline_dur: dict[str, float] = {}
    for (tl, _t), v in timeline_step_duration.items():
        sum_timeline_dur[tl] = sum_timeline_dur.get(tl, 0.0) + v

    # period__branch as a frozenset of pairs for O(1) membership.
    pb_pairs = frozenset(
        (r[0], r[1]) for r in pb if len(r) >= 2 and r[0] and r[1]
    )

    # complete_step_duration sum by branch (the d2 in mod's iteration).
    sum_complete_by_d2: dict[str, float] = {}
    for (d2, _t), v in complete_step_duration.items():
        sum_complete_by_d2[d2] = sum_complete_by_d2.get(d2, 0.0) + v

    # ---- Outputs --------------------------------------------------------

    # p_timeline_duration_in_years[tl] = sum_t step_duration[tl, t] / 8760
    p_tdy = [(tl, sum_timeline_dur.get(tl, 0.0) / 8760.0) for tl in timelines]
    _write_keyed(solve_data_dir / "p_timeline_duration_in_years.csv",
                 ("timeline", "value"), p_tdy)

    # hours_in_period[d]
    hours_in_period = [
        (d, sum_step_dur_by_period.get(d, 0.0)) for d in period_in_use
    ]
    _write_keyed(solve_data_dir / "hours_in_period.csv",
                 ("period", "value"), hours_in_period)

    # period_share_of_year[d]
    period_share = [(d, h / 8760.0) for d, h in hours_in_period]
    _write_keyed(solve_data_dir / "period_share_of_year.csv",
                 ("period", "value"), period_share)

    # p_years_d[d]
    p_years_d_rows = [
        (d, p_period_from_solve.get(d, 0.0)) for d in period_with_history
    ]
    _write_keyed(solve_data_dir / "p_years_d.csv",
                 ("period", "value"), p_years_d_rows)

    # p_years_represented_d[d] — written with _calc suffix to avoid collision
    # with mod's printf at flextool.mod:4974 which targets the same CSV
    # name in solveFirst.
    p_years_rep_d = []
    for d in periodAll:
        years = years_for_period.get(d, ())
        s = sum(p_years_represented.get((d, y), 1.0) for y in years)
        p_years_rep_d.append((d, s))
    _write_keyed(solve_data_dir / "p_years_represented_d_calc.csv",
                 ("period", "value"), p_years_rep_d)

    # complete_hours_in_period[d] = sum over (d2, t) in dt_complete
    # restricted to (d2, d) ∈ period__branch.
    complete_hours = []
    # For each (d2, total_completes) and each d s.t. (d2, d) ∈ pb, add total.
    # Equivalent: per-d, sum across all d2 with (d2, d) ∈ pb.
    branches_for_period: dict[str, list[str]] = {}
    for d2, d in pb_pairs:
        branches_for_period.setdefault(d, []).append(d2)
    for d in period_in_use:
        total = sum(sum_complete_by_d2.get(d2, 0.0)
                    for d2 in branches_for_period.get(d, ()))
        complete_hours.append((d, total))
    _write_keyed(solve_data_dir / "complete_hours_in_period.csv",
                 ("period", "value"), complete_hours)

    # complete_period_share_of_year[d] — same _calc-suffix rationale.
    complete_share = [(d, h / 8760.0) for d, h in complete_hours]
    _write_keyed(solve_data_dir / "complete_period_share_of_year_calc.csv",
                 ("period", "value"), complete_share)
