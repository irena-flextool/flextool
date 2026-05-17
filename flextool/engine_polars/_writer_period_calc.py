"""Writer-port Phase 2 (sub-dispatch 3) — per-period calculated params.

Native polars port of
``flextool.flextoolrunner.preprocessing.period_calculated_params``
(legacy ~467 LOC).  Called per-solve from
``flextool.flextoolrunner.preprocessing.solve_time.run`` via
``write_period_calculated_params`` (batch 13) and
``write_branch_weights`` (batch 63).

Output CSVs:

* ``write_period_calculated_params`` (12 CSVs):

  - ``p_timeline_duration_in_years.csv``        — sum_t dur / 8760
  - ``hours_in_period.csv``                     — sum_t step_duration
  - ``period_share_of_year.csv``                — hours / 8760
  - ``p_years_d.csv``                           — period_with_history scan
  - ``p_years_represented_d_calc.csv``          — sum across years
  - ``complete_hours_in_period.csv``            — branch-summed complete hours
  - ``complete_period_share_of_year_calc.csv``  — complete hours / 8760
  - ``p_years_until_invest.csv``                — cumulative years + invest offset
  - ``p_years_until_dispatch.csv``              — cumulative years + ops offset
  - ``p_inflation_factor_investment_yearly.csv``— inflation discount factor
  - ``p_inflation_factor_operations_yearly.csv``— inflation discount factor
  - ``f_d_k.csv``                               — ladder fraction

* ``write_branch_weights`` (2 CSVs):

  - ``pd_branch_weight.csv``  — per-(period) sibling-normalised weight
  - ``pdt_branch_weight.csv`` — per-(period, time) sibling-normalised weight

Reuse note (Phase 2 sub-dispatch 3 brief):

``flextool.engine_polars._per_solve_sets.PerSolveAggregates`` already
derives ``complete_period_share_of_year`` natively from
:class:`InputSource` (cluster-D / NPV path) — but that helper is
upstream-bound: it works on solve-config objects rather than the
per-solve ``solve_data/*.csv`` files that this writer reads.  Sharing
its lazy-LF pipeline would force a second data path through this
writer; instead we mirror the legacy CSV-reading shape (which is the
authoritative byte-for-byte parity surface) and keep the formula
expression aligned with ``PerSolveAggregates``.

Float values formatted with ``repr(float(v))`` for byte-identical
parity with the legacy emitter.  Where the legacy ``sum(())`` lands as
``int 0`` we mirror by not pre-casting to float (see
``p_years_represented_d_calc`` empty-domain branch in the legacy code).
"""
from __future__ import annotations

from pathlib import Path

import polars as pl


# ---------------------------------------------------------------------------
# CSV I/O helpers — same conventions as ``_writer_per_solve`` /
# ``_writer_lp_scaling``.
# ---------------------------------------------------------------------------


def _read_csv(path: Path, columns: list[str],
              *, provider: "object | None" = None) -> pl.DataFrame:
    """Eager-read a tiny flextool CSV via the Provider.

    Returns the Provider's frame sliced to *columns* with positional
    rename; returns an empty all-``Utf8`` frame when the Provider
    misses the key.  Step 2.5 Phase C dropped the disk-fallback arm —
    cascade modules read the Provider only.
    """
    from flextool.engine_polars._writer_provider_io import (
        _provider_key,
        _provider_lookup_positional,
    )
    seeded = _provider_lookup_positional(
        provider, _provider_key(path), path, columns,
    )
    if seeded is not None:
        return seeded
    return pl.DataFrame(
        {c: [] for c in columns},
        schema={c: pl.Utf8 for c in columns},
    )


def _read_singles(path: Path,
                  *, provider: "object | None" = None) -> list[str]:
    """List the first column of a header+rows CSV, dropping blanks."""
    df = _read_csv(path, ["v"], provider=provider)
    return [v for v in df["v"].to_list() if v]


def _to_utf8_frame(
    headers: tuple[str, ...],
    rows: list[tuple],
) -> pl.DataFrame:
    """Build an all-``Utf8`` polars frame from a header tuple + row list.

    Each cell is taken verbatim when already a string, otherwise via
    ``str(v)``.  Float cells are pre-rendered with ``repr(float(v))``
    by the ``derive_*`` builders so polars' ``write_csv`` does not
    re-format numbers (preserves byte parity with the legacy emitter).
    """
    cols: dict[str, list[str]] = {h: [] for h in headers}
    for row in rows:
        for h, v in zip(headers, row):
            cols[h].append(v if isinstance(v, str) else str(v))
    return pl.DataFrame(cols, schema={h: pl.Utf8 for h in headers})


def _write(df: pl.DataFrame, path: Path) -> None:
    """Canonical emitter — funnels every CSV through a single helper so
    :mod:`._flex_data_accumulator` can capture frames via monkey-patch.

    """
    path.parent.mkdir(parents=True, exist_ok=True)
    df.write_csv(path)


def _keyed_frame(header: tuple[str, str],
                 rows: list[tuple[str, float]]) -> pl.DataFrame:
    """Build a 2-col Utf8 frame with ``repr(v)`` on the value cell."""
    return _to_utf8_frame(
        header, [(k, repr(v)) for k, v in rows],
    )


def _keyed_frame_2(header: tuple[str, str, str],
                   rows: list[tuple[str, str, float]]) -> pl.DataFrame:
    """Build a 3-col Utf8 frame with ``repr(v)`` on the value cell."""
    return _to_utf8_frame(
        header, [(a, b, repr(v)) for a, b, v in rows],
    )


# Thin compatibility wrappers — funnel every emit through ``_write`` so
# :mod:`._flex_data_accumulator` captures the polars frame via its
# monkey-patched ``_write`` hook.  Kept as named helpers so the call
# sites read as ``_write_keyed(path, header, rows)`` (matches the
# legacy emitter's intent) while still routing through the canonical
# ``_write(df, path)`` shape.


def _write_keyed(path: Path, header: tuple[str, str],
                 rows: list[tuple[str, float]]) -> None:
    _write(_keyed_frame(header, rows), path)


def _write_keyed_2(path: Path, header: tuple[str, str, str],
                   rows: list[tuple[str, str, float]]) -> None:
    _write(_keyed_frame_2(header, rows), path)


def _read_step_duration(
    path: Path,
    *, provider: "object | None" = None,
) -> tuple[dict[tuple[str, str], float], dict[str, None]]:
    """Read (period, step, duration) CSV → (dict, ordered-set of periods)."""
    df = _read_csv(path, ["period", "step", "duration"], provider=provider)
    out: dict[tuple[str, str], float] = {}
    periods: dict[str, None] = {}
    for d, t, v in zip(df["period"].to_list(),
                       df["step"].to_list(),
                       df["duration"].to_list()):
        if not d or not t:
            continue
        try:
            out[(d, t)] = float(v)
            periods.setdefault(d, None)
        except (ValueError, TypeError):
            continue
    return out, periods


def _read_pb_pairs(path: Path,
                   *, provider: "object | None" = None) -> list[tuple[str, str]]:
    """Read period__branch.csv → list of (period, branch) tuples."""
    df = _read_csv(path, ["period", "branch"], provider=provider)
    return [(d, b) for d, b in zip(df["period"].to_list(),
                                   df["branch"].to_list())
            if d and b]


# ---------------------------------------------------------------------------
# Family A — write_period_calculated_params
# ---------------------------------------------------------------------------


def write_period_calculated_params(
    input_dir: Path, solve_data_dir: Path,
    *, provider: "object | None" = None,
) -> None:
    """Native port of
    ``period_calculated_params.write_period_calculated_params``.

    Emits 12 CSVs covering per-period hour aggregates, year coverage
    counts, inflation discount factors and ladder fractions.
    """
    # ── Sources ────────────────────────────────────────────────────────
    step_duration, _period_set = _read_step_duration(
        solve_data_dir / "steps_in_use.csv", provider=provider,
    )
    complete_step_duration, _ = _read_step_duration(
        solve_data_dir / "steps_complete_solve.csv", provider=provider,
    )

    pb_rows = _read_pb_pairs(solve_data_dir / "period__branch.csv",
                             provider=provider)
    branches_for_d: dict[str, list[str]] = {}
    for d, b in pb_rows:
        branches_for_d.setdefault(d, []).append(b)

    timeline_step_duration, timelines = _read_step_duration(
        input_dir / "timeline.csv", provider=provider,
    )

    # period_with_history: (period, year_value)
    pwh_df = _read_csv(solve_data_dir / "period_with_history.csv",
                       ["period", "value"], provider=provider)
    p_period_from_solve: dict[str, float] = {}
    period_with_history: list[str] = []
    for d, v in zip(pwh_df["period"].to_list(),
                    pwh_df["value"].to_list()):
        if not d:
            continue
        period_with_history.append(d)
        try:
            p_period_from_solve[d] = float(v)
        except (ValueError, TypeError):
            continue

    # p_years_represented: cols (period, years_from_solve,
    # p_years_from_solve, p_years_represented).  Bind to col index 3
    # — mod's table data IN at flextool.mod:751 uses the last column.
    pyr_df = _read_csv(
        solve_data_dir / "p_years_represented.csv",
        ["period", "year", "p_years_from_solve", "value"],
        provider=provider,
    )
    p_years_represented: dict[tuple[str, str], float] = {}
    years_for_period: dict[str, list[str]] = {}
    for d, y, v in zip(pyr_df["period"].to_list(),
                       pyr_df["year"].to_list(),
                       pyr_df["value"].to_list()):
        if not d or not y:
            continue
        try:
            p_years_represented[(d, y)] = float(v)
            years_for_period.setdefault(d, []).append(y)
        except (ValueError, TypeError):
            continue

    period_in_use = _read_singles(solve_data_dir / "period_in_use_set.csv",
                                   provider=provider)
    periodAll = _read_singles(solve_data_dir / "periodAll_set.csv",
                              provider=provider)

    # ── Per-row aggregations ───────────────────────────────────────────
    sum_step_dur_by_period: dict[str, float] = {}
    for (d, _t), v in step_duration.items():
        sum_step_dur_by_period[d] = sum_step_dur_by_period.get(d, 0.0) + v

    sum_timeline_dur: dict[str, float] = {}
    for (tl, _t), v in timeline_step_duration.items():
        sum_timeline_dur[tl] = sum_timeline_dur.get(tl, 0.0) + v

    pb_pairs_set = frozenset(pb_rows)

    sum_complete_by_d2: dict[str, float] = {}
    for (d2, _t), v in complete_step_duration.items():
        sum_complete_by_d2[d2] = sum_complete_by_d2.get(d2, 0.0) + v

    # ── Outputs ────────────────────────────────────────────────────────

    # p_timeline_duration_in_years
    p_tdy = [(tl, sum_timeline_dur.get(tl, 0.0) / 8760.0) for tl in timelines]
    _write_keyed(
        solve_data_dir / "p_timeline_duration_in_years.csv",
        ("timeline", "value"), p_tdy,
    )

    # hours_in_period
    hours_in_period = [
        (d, sum_step_dur_by_period.get(d, 0.0)) for d in period_in_use
    ]
    _write_keyed(
        solve_data_dir / "hours_in_period.csv",
        ("period", "value"), hours_in_period,
    )

    # period_share_of_year
    period_share = [(d, h / 8760.0) for d, h in hours_in_period]
    _write_keyed(
        solve_data_dir / "period_share_of_year.csv",
        ("period", "value"), period_share,
    )

    # p_years_d
    p_years_d_rows = [
        (d, p_period_from_solve.get(d, 0.0)) for d in period_with_history
    ]
    _write_keyed(
        solve_data_dir / "p_years_d.csv",
        ("period", "value"), p_years_d_rows,
    )

    # p_years_represented_d_calc — sum across years bound to d.
    # NOTE: legacy preserves ``int 0`` for empty domain by computing
    # ``sum(())`` (== 0), not 0.0.  We mirror by NOT pre-casting.
    p_years_rep_d: list[tuple[str, float]] = []
    for d in periodAll:
        years = years_for_period.get(d, ())
        s = sum(p_years_represented.get((d, y), 1.0) for y in years)
        p_years_rep_d.append((d, s))
    _write_keyed(
        solve_data_dir / "p_years_represented_d_calc.csv",
        ("period", "value"), p_years_rep_d,
    )

    # complete_hours_in_period
    # For each d in period_in_use, sum over d2 s.t. (d2, d) ∈ pb of
    # ``sum_t complete_step_duration[d2, t]``.
    branches_for_period: dict[str, list[str]] = {}
    for d2, d in pb_pairs_set:
        branches_for_period.setdefault(d, []).append(d2)
    complete_hours: list[tuple[str, float]] = []
    for d in period_in_use:
        total = sum(
            sum_complete_by_d2.get(d2, 0.0)
            for d2 in branches_for_period.get(d, ())
        )
        complete_hours.append((d, total))
    _write_keyed(
        solve_data_dir / "complete_hours_in_period.csv",
        ("period", "value"), complete_hours,
    )

    # complete_period_share_of_year_calc
    complete_share = [(d, h / 8760.0) for d, h in complete_hours]
    _write_keyed(
        solve_data_dir / "complete_period_share_of_year_calc.csv",
        ("period", "value"), complete_share,
    )

    # ── Inflation factors ──────────────────────────────────────────────
    def _scalar_max(csv_path: Path, default: float) -> float:
        rows_df = _read_csv(csv_path, ["key", "value"], provider=provider)
        vals: list[float] = []
        for v in rows_df["value"].to_list():
            if v is None or v == "":
                continue
            try:
                vals.append(float(v))
            except ValueError:
                continue
        return max(vals) if vals else default

    p_inflation = _scalar_max(input_dir / "p_inflation_rate.csv", 0.0)
    p_infl_offset_investment = _scalar_max(
        input_dir / "p_inflation_offset_investment.csv", 0.0,
    )
    p_infl_offset_operations = _scalar_max(
        input_dir / "p_inflation_offset_operations.csv", 0.5,
    )

    # Global year universe — union across all periods, sorted numerically.
    global_years_set: dict[str, None] = {}
    for years_list in years_for_period.values():
        for y in years_list:
            global_years_set.setdefault(y, None)
    try:
        sorted_global_years = sorted(
            global_years_set.keys(), key=lambda y: float(y),
        )
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

        # Cumulative-up-to-y across the GLOBAL year axis (p_years_represented
        # defaults to 1 for unbound (d, y2) pairs).
        cumulative = 0.0
        global_pos: dict[str, float] = {}
        for y2 in sorted_global_years:
            global_pos[y2] = cumulative
            cumulative += p_years_represented.get((d, y2), 1.0)

        per_year: list[tuple[str, float, float]] = []
        for y in sorted_d_years:
            pyr = p_years_represented.get((d, y), 1.0)
            base = global_pos.get(y, 0.0)
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

    _write_keyed_2(
        solve_data_dir / "p_years_until_invest.csv",
        ("period", "year", "value"), pyy_invest,
    )
    _write_keyed_2(
        solve_data_dir / "p_years_until_dispatch.csv",
        ("period", "year", "value"), pyy_dispatch,
    )

    # p_inflation_factor_*_yearly — investment over period_set,
    # operations over period_in_use.
    period_universe = _read_singles(solve_data_dir / "period_set.csv",
                                     provider=provider)
    inv_yearly = [
        (d, inflation_invest.get(d, 1.0)) for d in period_universe
    ]
    ops_yearly = [
        (d, inflation_ops.get(d, 1.0)) for d in period_in_use
    ]
    _write_keyed(
        solve_data_dir / "p_inflation_factor_investment_yearly.csv",
        ("period", "value"), inv_yearly,
    )
    _write_keyed(
        solve_data_dir / "p_inflation_factor_operations_yearly.csv",
        ("period", "value"), ops_yearly,
    )

    # ── f_d_k ──────────────────────────────────────────────────────────
    # = (p_ladder_cum_sim_hours[d] + sum_t step_duration[d, t]) /
    #   (complete_period_share_of_year[d] * 8760)
    ladder_df = _read_csv(
        solve_data_dir / "ladder_cum_sim_hours.csv",
        ["period", "value"],
        provider=provider,
    )
    p_ladder_cum: dict[str, float] = {}
    for d, v in zip(ladder_df["period"].to_list(),
                    ladder_df["value"].to_list()):
        if not d:
            continue
        try:
            p_ladder_cum[d] = float(v)
        except (ValueError, TypeError):
            continue
    complete_share_lookup = dict(complete_share)
    f_d_k_rows: list[tuple[str, float]] = []
    for d in period_in_use:
        num = p_ladder_cum.get(d, 0.0) + sum_step_dur_by_period.get(d, 0.0)
        denom = complete_share_lookup.get(d, 0.0) * 8760.0
        f_d_k_rows.append((d, num / denom))
    _write_keyed(
        solve_data_dir / "f_d_k.csv",
        ("period", "value"), f_d_k_rows,
    )


# ---------------------------------------------------------------------------
# Family B — write_branch_weights
# ---------------------------------------------------------------------------


def write_branch_weights(input_dir: Path, solve_data_dir: Path,
                          *, provider: "object | None" = None) -> None:
    """Native port of ``period_calculated_params.write_branch_weights``.

    Both ``pd_branch_weight`` and ``pdt_branch_weight`` normalise
    ``solve_branch_weight[d]`` against the sum across sibling branches
    (matching first-step / time and sharing a common parent ``d2``).
    """
    del input_dir  # legacy signature parity; no input/ reads here.

    pb_rows = _read_pb_pairs(solve_data_dir / "period__branch.csv",
                             provider=provider)
    pb_set = frozenset(pb_rows)

    # solve_branch_weight (branch, value) → default 1.0 per mod default.
    sbw_df = _read_csv(
        solve_data_dir / "solve_branch_weight.csv",
        ["branch", "value"],
        provider=provider,
    )
    branch_weight: dict[str, float] = {}
    for b, v in zip(sbw_df["branch"].to_list(), sbw_df["value"].to_list()):
        if not b:
            continue
        try:
            branch_weight[b] = float(v)
        except (ValueError, TypeError):
            continue

    def w(b: str) -> float:
        return branch_weight.get(b, 1.0)

    # first_timesteps (period, step) — for the pd_branch_weight gate.
    ft_df = _read_csv(
        solve_data_dir / "first_timesteps.csv",
        ["period", "step"],
        provider=provider,
    )
    times_with_first_set: dict[str, set[str]] = {}
    first_time_for_d: dict[str, str] = {}
    for d, t in zip(ft_df["period"].to_list(), ft_df["step"].to_list()):
        if not d or not t:
            continue
        first_time_for_d[d] = t
        times_with_first_set.setdefault(t, set()).add(d)

    # steps_in_use as (period, time) — for the pdt_branch_weight iteration.
    su_df = _read_csv(
        solve_data_dir / "steps_in_use.csv",
        ["period", "step"],
        provider=provider,
    )
    dt_pairs: list[tuple[str, str]] = []
    branches_for_t: dict[str, set[str]] = {}
    for d, t in zip(su_df["period"].to_list(), su_df["step"].to_list()):
        if not d or not t:
            continue
        dt_pairs.append((d, t))
        branches_for_t.setdefault(t, set()).add(d)

    period_in_use = _read_singles(solve_data_dir / "period_in_use_set.csv",
                                   provider=provider)

    # ── pd_branch_weight ───────────────────────────────────────────────
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
    _write_keyed(
        solve_data_dir / "pd_branch_weight.csv",
        ("period", "value"), pd_rows,
    )

    # ── pdt_branch_weight ──────────────────────────────────────────────
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
    _write_keyed_2(
        solve_data_dir / "pdt_branch_weight.csv",
        ("period", "time", "value"), pdt_rows,
    )
