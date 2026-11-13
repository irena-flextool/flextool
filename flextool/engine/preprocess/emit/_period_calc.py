"""Per-period calculated params.

Called per-solve via ``write_period_calculated_params`` and
``write_branch_weights``.

Output CSVs:

* ``write_period_calculated_params`` (11 CSVs):

  - ``p_timeline_duration_in_years.csv``        — sum_t dur / 8760
  - ``hours_in_period.csv``                     — sum_t step_duration
  - ``period_share_of_year.csv``                — hours / 8760
  - ``p_years_d.csv``                           — period_with_history scan
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

``flextool.engine.preprocess.params._per_solve_sets.PerSolveAggregates`` already
derives ``complete_period_share_of_year`` natively from
:class:`InputSource` (cluster-D / NPV path) — but that helper is
upstream-bound: it works on solve-config objects rather than the
per-solve ``solve_data/*.csv`` files that this writer reads.  Sharing
its lazy-LF pipeline would force a second data path through this
writer; instead we mirror the legacy CSV-reading shape (which is the
authoritative byte-for-byte parity surface) and keep the formula
expression aligned with ``PerSolveAggregates``.

Float values formatted with ``repr(float(v))`` for byte-identical
parity with the legacy emitter.
"""
from __future__ import annotations

from pathlib import Path

import polars as pl

from flextool.engine.input._constants import HOURS_PER_YEAR
from flextool.engine.preprocess._provider_io import _emit
from flextool.engine.preprocess.params._branch import branch_periods_inherit_carry


# ---------------------------------------------------------------------------
# CSV I/O helpers — same conventions as ``_per_solve`` /
# ``_lp_scaling``.
# ---------------------------------------------------------------------------


def _read_csv(path: Path, columns: list[str],
              *, provider: "object | None" = None) -> pl.DataFrame:
    """Eager-read a tiny flextool CSV via the Provider.

    Returns the Provider's frame sliced to *columns* with positional
    rename; returns an empty all-``Utf8`` frame when the Provider
    misses the key.  Step 2.5 Phase C dropped the disk-fallback arm —
    cascade modules read the Provider only.
    """
    from flextool.engine.preprocess._provider_io import (
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


def _emit_keyed(provider, key: str, header: tuple[str, str],
                rows: list[tuple[str, float]]) -> None:
    _emit(provider, key, _keyed_frame(header, rows))


def _emit_keyed_2(provider, key: str, header: tuple[str, str, str],
                  rows: list[tuple[str, str, float]]) -> None:
    _emit(provider, key, _keyed_frame_2(header, rows))


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


def emit_period_calculated_params(
    input_dir: Path, solve_data_dir: Path,
    *, provider,
) -> None:
    """Emit ``period_calculated_params`` to the Provider.
    Emits the same 12 frames under ``solve_data/<basename>`` keys via
    :func:`_emit` (dual-key registration).
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

    p_tdy = [(tl, sum_timeline_dur.get(tl, 0.0) / HOURS_PER_YEAR)
             for tl in timelines]
    _emit_keyed(provider, "solve_data/p_timeline_duration_in_years.csv",
                ("timeline", "value"), p_tdy)

    hours_in_period = [
        (d, sum_step_dur_by_period.get(d, 0.0)) for d in period_in_use
    ]
    _emit_keyed(provider, "solve_data/hours_in_period.csv",
                ("period", "value"), hours_in_period)

    period_share = [(d, h / HOURS_PER_YEAR) for d, h in hours_in_period]
    _emit_keyed(provider, "solve_data/period_share_of_year.csv",
                ("period", "value"), period_share)

    p_years_d_rows = [
        (d, p_period_from_solve.get(d, 0.0)) for d in period_with_history
    ]
    _emit_keyed(provider, "solve_data/p_years_d.csv",
                ("period", "value"), p_years_d_rows)

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
    _emit_keyed(provider, "solve_data/complete_hours_in_period.csv",
                ("period", "value"), complete_hours)

    complete_share = [(d, h / HOURS_PER_YEAR) for d, h in complete_hours]
    _emit_keyed(provider, "solve_data/complete_period_share_of_year_calc.csv",
                ("period", "value"), complete_share)

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

    # The years from the start of the run (``_npv`` "The discount base of a solve's own factors"), when the
    # solve's own count does not give them: a roll or nested sub-solve of a solve that starts after the run
    # start, or fractional years.  None keeps the own count below.
    from flextool.engine.preprocess.params._npv import notice_discount_from_run_start, year_rows_from_history
    history: dict[str, float] = {}
    for d, v in zip(pwh_df["period"].to_list(), pwh_df["value"].to_list()):
        if d and d not in history:
            try:
                history[d] = float(v)
            except (ValueError, TypeError):
                continue
    solve_current = (_read_singles(solve_data_dir / "solve_current.csv", provider=provider) or [None])[0]
    year_rows = year_rows_from_history(
        history,
        {d: [(y, p_years_represented[(d, y)]) for y in ys] for d, ys in years_for_period.items()},
        solve=solve_current)
    if year_rows is not None:
        notice_discount_from_run_start(solve_current, p_inflation)
        in_all = set(periodAll)
        for d, y, pyr, base in year_rows.iter_rows():
            if d not in in_all:
                continue
            until_invest = base + pyr * p_infl_offset_investment
            until_dispatch = base + pyr * p_infl_offset_operations
            pyy_invest.append((d, y, until_invest))
            pyy_dispatch.append((d, y, until_dispatch))
            inflation_invest[d] = inflation_invest.get(d, 0.0) + pyr * (one_plus_inflation_inv ** until_invest)
            inflation_ops[d] = inflation_ops.get(d, 0.0) + pyr * (one_plus_inflation_inv ** until_dispatch)

    for d in ([] if year_rows is not None else periodAll):
        years_for_d = years_for_period.get(d, ())
        try:
            sorted_d_years = sorted(years_for_d, key=lambda y: float(y))
        except ValueError:
            sorted_d_years = sorted(years_for_d)

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

    _emit_keyed_2(provider, "solve_data/p_years_until_invest.csv",
                  ("period", "year", "value"), pyy_invest)
    _emit_keyed_2(provider, "solve_data/p_years_until_dispatch.csv",
                  ("period", "year", "value"), pyy_dispatch)

    period_universe = _read_singles(solve_data_dir / "period_set.csv",
                                     provider=provider)
    inv_yearly = [
        (d, inflation_invest.get(d, 1.0)) for d in period_universe
    ]
    ops_yearly = [
        (d, inflation_ops.get(d, 1.0)) for d in period_in_use
    ]
    _emit_keyed(provider, "solve_data/p_inflation_factor_investment_yearly.csv",
                ("period", "value"), inv_yearly)
    _emit_keyed(provider, "solve_data/p_inflation_factor_operations_yearly.csv",
                ("period", "value"), ops_yearly)

    # Phase 4.1a / 4.2-1g — read the cross-roll sim-hour accumulator from
    # the handoff translator's ``handoff/cum_sim_hours`` Provider key
    # (canonical schema ``[period, p_ladder_cum_sim_hours]``).  Empty /
    # missing handoff (first solve) collapses ``p_ladder_cum`` to {}
    # so ``f_d_k`` reduces to ``sum_step_dur / (share * 8760)``.  The
    # legacy ``solve_data/ladder_cum_sim_hours`` CSV fallback was dropped
    # in 4.2-1g — the translator pipeline always populates the handoff
    # key now, so the fallback was dead in production.
    from flextool.engine.preprocess import _provider_keys as K
    from flextool.engine.preprocess._provider_translators import read_handoff_frame
    p_ladder_cum: dict[str, float] = {}
    handoff_df = read_handoff_frame(provider, K.HANDOFF_CUM_SIM_HOURS)
    if handoff_df is not None and "p_ladder_cum_sim_hours" in handoff_df.columns:
        for d, v in zip(handoff_df["period"].to_list(),
                        handoff_df["p_ladder_cum_sim_hours"].to_list()):
            if not d:
                continue
            try:
                p_ladder_cum[d] = float(v)
            except (ValueError, TypeError):
                continue
    complete_share_lookup = dict(complete_share)
    # A forecast branch period continues from its period's realized hours: the realization solve wrote
    # them under the period label, so each forecast takes the label's carry (f_d_k[a_b] = (H[a] + W[a_b]) /
    # T[a]); the label keeps H[a] / T[a].  The fold of finished periods below reads the raw carry keys.
    cum_of = p_ladder_cum
    if p_ladder_cum and pb_pairs_set:
        carry = branch_periods_inherit_carry(
            pl.DataFrame({"period": list(p_ladder_cum), "value": list(p_ladder_cum.values())},
                         schema={"period": pl.Utf8, "value": pl.Float64}),
            pb_pairs_set, period_col="period")
        cum_of = dict(zip(carry["period"].to_list(), carry["value"].to_list()))
    f_d_k_rows: list[tuple[str, float]] = []
    for d in period_in_use:
        num = cum_of.get(d, 0.0) + sum_step_dur_by_period.get(d, 0.0)
        denom = complete_share_lookup.get(d, 0.0) * HOURS_PER_YEAR
        f_d_k_rows.append((d, num / denom))
    # A period only leaves `period_in_use` once the rolling chain has moved past it, which (periods
    # are visited in order, never revisited) only happens once it is fully realised — at that point
    # its frozen p_ladder_cum entry already equals its own complete hours, i.e. its true f_d_k is
    # exactly 1.0.  Fold those periods back in instead of letting them drop out of the domain: the
    # cumulative-ladder cap (`_eq_tier_cap_cumulative`) sums f_d_k over every period the chain has
    # ever touched, not just the ones still active this roll, so dropping a finished period here
    # made that sum reset to the new period's own local progress each time the chain advanced,
    # while the realized-purchases carrier kept climbing — eventually driving the cap's row negative
    # (#143's root cause).
    period_in_use_set = set(period_in_use)
    for d in p_ladder_cum:
        if d not in period_in_use_set:
            f_d_k_rows.append((d, 1.0))
    _emit_keyed(provider, "solve_data/f_d_k.csv",
                ("period", "value"), f_d_k_rows)


# ---------------------------------------------------------------------------
# Family B — write_branch_weights
# ---------------------------------------------------------------------------


def emit_branch_weights(input_dir: Path, solve_data_dir: Path,
                         *, provider) -> None:
    """Emit ``branch_weights`` to the Provider.
    Emits ``pd_branch_weight.csv`` and ``pdt_branch_weight.csv`` under
    ``solve_data/<basename>`` keys via :func:`_emit`.
    """
    del input_dir  # legacy signature parity; no input/ reads here.

    pb_rows = _read_pb_pairs(solve_data_dir / "period__branch.csv",
                             provider=provider)
    pb_set = frozenset(pb_rows)

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

    # Forecasts-only cohort rule (unconditional) — a period label is a
    # self-row ``(d, d) ∈ pb`` that ALSO fans out to a forecast member;
    # it carries no probability mass → weight 0, excluded from every forecast
    # denominator, and has no steps (so it is tested before the first-step
    # lookup).  A pre-reveal / deterministic period is not a label and
    # keeps 1.0.  Kept in lock-step with
    # ``_branch.pdt_branch_weight_lf`` (the authoritative LP weight) via
    # the shared :func:`_branch.fan_period_labels` helper, so this emitted
    # CSV stays byte-parity with the LP Param.  Applied to the pd (invest)
    # loop as well as pdt.
    from flextool.engine.preprocess.params._branch import fan_period_labels
    phantom = fan_period_labels(pb_set)

    pd_rows: list[tuple[str, float]] = []
    for d in period_in_use:
        if d in phantom:
            pd_rows.append((d, 0.0))
            continue
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
            if b in phantom:
                continue
            denom += w(b)
        if denom == 0.0:
            continue
        pd_rows.append((d, w(d) / denom))
    _emit_keyed(provider, "solve_data/pd_branch_weight.csv",
                ("period", "value"), pd_rows)

    pdt_rows: list[tuple[str, str, float]] = []
    for d, t in dt_pairs:
        if d in phantom:
            # Period-label operational weight → 0 (forecasts carry the mass).
            pdt_rows.append((d, t, 0.0))
            continue
        branches_with_t = branches_for_t.get(t, set())
        denom = 0.0
        for d2, b in pb_rows:
            if b not in branches_with_t:
                continue
            if (d2, d) not in pb_set:
                continue
            if b in phantom:
                # Exclude the period label from the forecast denominator.
                continue
            denom += w(b)
        if denom == 0.0:
            continue
        pdt_rows.append((d, t, w(d) / denom))
    _emit_keyed_2(provider, "solve_data/pdt_branch_weight.csv",
                  ("period", "time", "value"), pdt_rows)
