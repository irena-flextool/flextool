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

    # solve_data/p_years_represented.csv columns:
    #   period, years_from_solve, p_years_from_solve, p_years_represented
    # Mod's table data IN at flextool.mod:751 binds the param
    # ``p_years_represented`` to the LAST column (index 3), NOT
    # ``p_years_from_solve`` at index 2. Reading index 2 here was a
    # bug that mismatched MPS coefficient values for any d using
    # multiple represented-years per (d, y) pair.
    pyr = _read_csv_columns(solve_data_dir / "p_years_represented.csv")
    p_years_represented: dict[tuple[str, str], float] = {}
    years_for_period: dict[str, list[str]] = {}
    for r in pyr:
        if len(r) >= 4 and r[0] and r[1]:
            try:
                p_years_represented[(r[0], r[1])] = float(r[3])
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

    # ---- Inflation-related params --------------------------------------
    # Scalar lookups: max over model entries (mod L1594-1596).
    def _scalar_max(csv_path: Path, default: float) -> float:
        rows = _read_csv_columns(csv_path)
        vals: list[float] = []
        for r in rows:
            if len(r) >= 2 and r[1]:
                try:
                    vals.append(float(r[1]))
                except ValueError:
                    continue
        if not vals:
            return default
        return max(vals)

    p_inflation = _scalar_max(input_dir / "p_inflation_rate.csv", 0.0)
    p_infl_offset_investment = _scalar_max(
        input_dir / "p_inflation_offset_investment.csv", 0.0,
    )
    p_infl_offset_operations = _scalar_max(
        input_dir / "p_inflation_offset_operations.csv", 0.5,
    )

    # p_years_until_invest[(d, y)]:
    #   mod L1545: + sum{y2 in year : y2 < y} p_years_represented[d, y2]
    #             + p_years_represented[d, y] * p_infl_offset_investment
    #
    # The inner sum is over the GLOBAL `year` set (union of all y across
    # all periods), NOT just years bound to d. ``p_years_represented`` has
    # ``default 1`` so unbound (d, y2) pairs contribute 1.
    # p_inflation_factor_*_yearly[d] = if any p_years_represented[d, ·] > 0
    #   then sum_{y in years_bound_to_d} pyr[d, y] * (1+inflation)^(-until[d, y])
    #   else 1.

    # Global year universe (union of years across all periods).
    global_years_set: dict[str, None] = {}
    for years_list in years_for_period.values():
        for y in years_list:
            global_years_set.setdefault(y, None)
    try:
        sorted_global_years = sorted(global_years_set.keys(), key=lambda y: float(y))
    except ValueError:
        sorted_global_years = sorted(global_years_set.keys())

    pyy_invest: list[tuple[str, str, float]] = []
    pyy_dispatch: list[tuple[str, str, float]] = []
    inflation_invest: dict[str, float] = {}
    inflation_ops: dict[str, float] = {}
    one_plus_inflation_inv = (
        1.0 / (1.0 + p_inflation) if p_inflation != -1.0 else 1.0
    )
    for d in periodAll:
        years_for_d = years_for_period.get(d, ())
        try:
            sorted_d_years = sorted(years_for_d, key=lambda y: float(y))
        except ValueError:
            sorted_d_years = sorted(years_for_d)
        try:
            d_years_set = frozenset(years_for_d)
        except TypeError:
            d_years_set = frozenset()

        # For each (d, y), inner sum walks the GLOBAL year set with
        # numerical y2 < y comparison. Cumulative state per d is built
        # by iterating sorted_global_years.
        cumulative = 0.0
        global_pos: dict[str, float] = {}  # y → cumulative-up-to-y
        for y2 in sorted_global_years:
            global_pos[y2] = cumulative
            cumulative += p_years_represented.get((d, y2), 1.0)

        per_year: list[tuple[str, float, float]] = []
        for y in sorted_d_years:
            pyr = p_years_represented.get((d, y), 1.0)
            base = global_pos.get(y, 0.0)  # sum_{y2 < y} pyr[d, y2]
            until_invest = base + pyr * p_infl_offset_investment
            until_dispatch = base + pyr * p_infl_offset_operations
            pyy_invest.append((d, y, until_invest))
            pyy_dispatch.append((d, y, until_dispatch))
            per_year.append((y, until_invest, until_dispatch))

        sum_p_years_for_d = sum(
            p_years_represented.get((d, y), 1.0) for y in sorted_d_years
        )
        if sum_p_years_for_d > 0:
            inv_factor = 0.0
            ops_factor = 0.0
            for y, until_inv, until_op in per_year:
                pyr = p_years_represented.get((d, y), 1.0)
                inv_factor += pyr * (one_plus_inflation_inv ** until_inv)
                ops_factor += pyr * (one_plus_inflation_inv ** until_op)
            inflation_invest[d] = inv_factor
            inflation_ops[d] = ops_factor
        else:
            inflation_invest[d] = 1.0
            inflation_ops[d] = 1.0

    _write_keyed_2(solve_data_dir / "p_years_until_invest.csv",
                   ("period", "year", "value"), pyy_invest)
    _write_keyed_2(solve_data_dir / "p_years_until_dispatch.csv",
                   ("period", "year", "value"), pyy_dispatch)

    # p_inflation_factor_*_yearly[d] is per-period (not periodAll for ops);
    # mod L1551 declares investment over `period`, operations over period_in_use.
    period_universe = [r[0] for r in _read_csv_columns(
        solve_data_dir / "period_set.csv"
    ) if r and r[0]]
    inv_yearly: list[tuple[str, float]] = [
        (d, inflation_invest.get(d, 1.0)) for d in period_universe
    ]
    ops_yearly: list[tuple[str, float]] = [
        (d, inflation_ops.get(d, 1.0)) for d in period_in_use
    ]
    _write_keyed(solve_data_dir / "p_inflation_factor_investment_yearly.csv",
                 ("period", "value"), inv_yearly)
    _write_keyed(solve_data_dir / "p_inflation_factor_operations_yearly.csv",
                 ("period", "value"), ops_yearly)

    # f_d_k[d] = (p_ladder_cum_sim_hours[d] + sum_t step_duration[d,t])
    #            / (complete_period_share_of_year[d] * 8760)
    # mod L1386. Domain: period_in_use. p_ladder_cum_sim_hours has
    # default 0 (mod L452) so missing rows imply zero. Division-by-zero
    # would only occur if complete_period_share_of_year[d] == 0, which
    # mod's reciprocal would already crash on — so we mirror that.
    p_ladder_cum: dict[str, float] = {}
    for r in _read_csv_columns(solve_data_dir / "ladder_cum_sim_hours.csv"):
        if len(r) >= 2 and r[0] and r[1]:
            try:
                p_ladder_cum[r[0]] = float(r[1])
            except ValueError:
                continue
    complete_share_lookup = dict(complete_share)
    f_d_k_rows: list[tuple[str, float]] = []
    for d in period_in_use:
        num = p_ladder_cum.get(d, 0.0) + sum_step_dur_by_period.get(d, 0.0)
        denom = complete_share_lookup.get(d, 0.0) * 8760.0
        f_d_k_rows.append((d, num / denom))
    _write_keyed(solve_data_dir / "f_d_k.csv",
                 ("period", "value"), f_d_k_rows)

    # ---- pdtConversion_rate ---------------------------------------------
    # mod L1686: round(1 / pdtProcess[p, 'efficiency', d, t], 6)
    # pdtProcess is itself a calc param that derives from per-solve sources.
    # For the migration step that doesn't migrate pdtProcess, we read its
    # printf'd output (mod writes solve_data/pdtProcess_*.csv files).
    # Simpler: read efficiency directly from the inputs that pdtProcess
    # reads — process_param_t / process_param_period — and apply the same
    # precedence. However mod's pdtProcess has many fallback branches, so
    # the safest bet is to read its printed output.
    #
    # Postpone pdtConversion_rate to a later batch where we migrate
    # pdtProcess itself. The L0 batch's matrix-gen impact is minimal
    # (it's a pre-evaluated coefficient lookup, not a constraint
    # iteration).
    # (pdtConversion_rate, section, slope landed in batch 57 inside
    # entity_period_calc_params.py.)


def write_branch_weights(input_dir: Path, solve_data_dir: Path) -> None:
    """flextool.mod L563-567 — pd_branch_weight + pdt_branch_weight.

    Both normalize p_branch_weight_input[d] by the sum of input weights
    across sibling branches.

        pd_branch_weight[d] = w[d] / sum w[b] over branches b such that
            (d2, b) ∈ period__branch
            AND (b, ts) ∈ period__time_first
            AND (d, ts) ∈ period__time_first    # same first-time as d
            AND (d2, d) ∈ period__branch        # same parent d2

        pdt_branch_weight[d, t] = w[d] / sum w[b] over branches b such that
            (d2, b) ∈ period__branch
            AND (b, t) ∈ dt                     # b's dt contains t
            AND (d2, d) ∈ period__branch        # same parent d2

    Reads solve_data/period__branch.csv (period, branch),
    solve_data/solve_branch_weight.csv (branch, value),
    solve_data/first_timesteps.csv (period, step) for period__time_first,
    solve_data/steps_in_use.csv (period, step, step_duration) for dt,
    solve_data/period_in_use_set.csv for the pd output domain.

    Float values formatted with `repr()` for round-trip-exact
    serialization within Python; MPS parity uses 7-sig-fig comparison.
    """
    pb_rows: list[tuple[str, str]] = []
    for r in _read_csv_columns(solve_data_dir / "period__branch.csv"):
        if len(r) >= 2 and r[0] and r[1]:
            pb_rows.append((r[0], r[1]))
    parents_of: dict[str, set[str]] = {}  # b -> {d2 : (d2, b) ∈ pb}
    for d2, b in pb_rows:
        parents_of.setdefault(b, set()).add(d2)

    branch_weight: dict[str, float] = {}
    for r in _read_csv_columns(solve_data_dir / "solve_branch_weight.csv"):
        if len(r) >= 2 and r[0]:
            try:
                branch_weight[r[0]] = float(r[1])
            except ValueError:
                continue

    def w(b: str) -> float:
        return branch_weight.get(b, 1.0)

    # period__time_first (period, step) — read solve_data/first_timesteps.csv
    times_with_first_set: dict[str, set[str]] = {}
    first_time_for_d: dict[str, str] = {}
    for r in _read_csv_columns(solve_data_dir / "first_timesteps.csv"):
        if len(r) >= 2 and r[0] and r[1]:
            first_time_for_d[r[0]] = r[1]
            times_with_first_set.setdefault(r[1], set()).add(r[0])

    # dt as (period, time)
    dt_pairs: list[tuple[str, str]] = []
    branches_for_t: dict[str, set[str]] = {}
    for r in _read_csv_columns(solve_data_dir / "steps_in_use.csv"):
        if len(r) >= 2 and r[0] and r[1]:
            dt_pairs.append((r[0], r[1]))
            branches_for_t.setdefault(r[1], set()).add(r[0])

    period_in_use = [r[0] for r in _read_csv_columns(
        solve_data_dir / "period_in_use_set.csv"
    ) if r and r[0]]

    # Iterate exactly as mod does — over (d2, b) tuples in pb — so a
    # branch b that has multiple parents d2 sharing parenthood with d is
    # counted once per (d2, b) tuple, matching MathProg's sum.
    pb_set = frozenset(pb_rows)

    pd_rows: list[tuple[str, float]] = []
    for d in period_in_use:
        ts = first_time_for_d.get(d)
        if ts is None:
            continue
        branches_at_ts = times_with_first_set.get(ts, set())
        denom = 0.0
        for d2, b in pb_rows:
            if b not in branches_at_ts:
                continue
            if (d2, d) not in pb_set:
                continue
            denom += w(b)
        if denom == 0.0:
            continue
        pd_rows.append((d, w(d) / denom))
    _write_keyed(solve_data_dir / "pd_branch_weight.csv",
                 ("period", "value"), pd_rows)

    pdt_rows: list[tuple[str, str, float]] = []
    for d, t in dt_pairs:
        branches_with_t = branches_for_t.get(t, set())
        denom = 0.0
        for d2, b in pb_rows:
            if b not in branches_with_t:
                continue
            if (d2, d) not in pb_set:
                continue
            denom += w(b)
        if denom == 0.0:
            continue
        pdt_rows.append((d, t, w(d) / denom))
    _write_keyed_2(solve_data_dir / "pdt_branch_weight.csv",
                   ("period", "time", "value"), pdt_rows)
