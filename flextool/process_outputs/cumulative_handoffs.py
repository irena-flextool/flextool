"""
Rolling per-period accumulator writers for price-ladder tiers.

These writers maintain two accumulators across rolling solves so the
LP's ladder caps (annual and cumulative alike) can partition a period's
quota across rolls that share that period:

* ``solve_data/ladder_cum_realized_mwh.csv`` — ``(commodity, tier, period)
  → realized sim-MWh of that tier of that commodity that fell into that
  period across all prior rolls``.  Loaded by the mod into
  ``p_ladder_cum_realized_mwh[c, i, d]``.

* ``solve_data/ladder_cum_sim_hours.csv`` — ``period → realized sim-hours
  of that period across all prior rolls``.  Loaded by the mod into
  ``p_ladder_cum_sim_hours[d]``.

Together with the current roll's horizon these let the mod compute
``f_d_k[d]``, the fraction of period d "filled" by prior-realized hours
plus this-roll hours, and cap v_trade on a rolling-partition basis (see
``flextool.mod`` ladder_tier_cap_annual_roll / _cumulative_roll /
_annual_overspent / _cumulative_overspent).

Uniform-split assumption (Bug #1 fix, commit #7)
------------------------------------------------
``v_trade[c, n, d, i]`` is period-level in the LP (no time index — the
LP makes one trade decision per (commodity, node, period, tier)).  A
roll's horizon covers only some of period d's hours, realizes a subset
(the dispatch window), and looks ahead at the rest.  When attributing
a roll's v_trade MWh to the "realized" slice, we assume the LP's
period-level decision distributes uniformly over the horizon's hours:

    realized_mwh_this_roll[c, i, d] =
        sum_n v_trade[c, n, d, i] * p_commodity_unitsize[c]
          * (realized_sim_hours_d / horizon_sim_hours_d)

This is a modelling choice — the LP itself cannot distinguish "realized"
from "lookahead" hours within a single v_trade number — but it matches
how cost and dispatch are already attributed at the period-level in the
rest of the model.  Under it the accumulators form a proper partition:
summing across all rolls of a run reconstructs the full realized MWh.

Hooked into ``solver_runner._run_highs_or_cplex`` after
:func:`flextool.process_outputs.handoff_writers.write_all_handoffs`.
HiGHS-only.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

from flextool.process_outputs.handoff_writers import (
    _is_first_solve,
    _load_step_duration,
)
from flextool.process_outputs.read_highs_solution import (
    _load_realized_set,
    extract_variable,
)

if TYPE_CHECKING:
    import highspy

_logger = logging.getLogger(__name__)

# Matches flextool.mod's "infinite / inactive" sentinel.  Tiers with
# total_cap at or above this value are the infinite tail — they have no
# cap, so no accumulator is needed for them.
_INFINITE_TIER_THRESHOLD = 1e29

# Commodities that need accumulator tracking: anything using a ladder
# price method (both annual and cumulative).  `price` commodities are
# skipped entirely.
_LADDER_PRICE_METHODS = frozenset(
    {"price_ladder_annual", "price_ladder_cumulative"}
)


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------


def _load_price_methods(work_folder: Path) -> dict[str, str]:
    """Return ``{commodity: price_method}`` from
    ``input/p_commodity_price_method.csv``.  Missing file → empty
    (every commodity defaults to ``'price'`` in the mod)."""
    path = work_folder / "input" / "p_commodity_price_method.csv"
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    if df.empty or "commodity" not in df.columns:
        return {}
    return {
        str(r["commodity"]): str(r["p_commodity_price_method"])
        for _, r in df.iterrows()
    }


def _load_ladder_commodities(work_folder: Path) -> set[str]:
    """Return the set of commodities using any ``price_ladder_*`` method.

    Both annual and cumulative tiers need rolling accumulators now — the
    mod's per-period cap partitions a period's quota across rolls that
    share that period.  A non-ladder commodity uses scalar price and is
    skipped.
    """
    return {
        c for c, m in _load_price_methods(work_folder).items()
        if m in _LADDER_PRICE_METHODS
    }


def _load_finite_ladder_tiers(
    work_folder: Path,
) -> dict[tuple[str, int], float]:
    """Return ``{(commodity, tier): quantity}`` for every ladder tier of
    any ``price_ladder_*`` commodity with a finite cap.

    Infinite tiers (quantity >= 1e29 sentinel) are dropped — they never
    bind and their accumulator row would contribute nothing.
    """
    ladder_commodities = _load_ladder_commodities(work_folder)
    if not ladder_commodities:
        return {}
    path = work_folder / "input" / "commodity_ladder.csv"
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    if df.empty:
        return {}
    out: dict[tuple[str, int], float] = {}
    for _, row in df.iterrows():
        c = str(row["commodity"])
        if c not in ladder_commodities:
            continue
        try:
            tier = int(row["tier"])
            q = float(row["quantity"])
        except (ValueError, TypeError):
            continue
        if q != q or q >= _INFINITE_TIER_THRESHOLD:  # NaN guard + sentinel
            continue
        out[(c, tier)] = q
    return out


def _load_commodity_unitsize(work_folder: Path) -> dict[str, float]:
    """Return ``{commodity: unitsize}`` from
    ``input/p_commodity_unitsize.csv``.  Unknown commodities default to
    1.0 at lookup time (matches the mod's default)."""
    path = work_folder / "input" / "p_commodity_unitsize.csv"
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    if df.empty or "commodity" not in df.columns:
        return {}
    return {
        str(r["commodity"]): float(r["p_commodity_unitsize"])
        for _, r in df.iterrows()
    }


def _load_prior_cum_realized_mwh(
    path: Path,
) -> dict[tuple[str, int, str], float]:
    """Return ``{(commodity, tier, period): cum_mwh}`` from the previous
    roll's accumulator CSV.  Header-only seed → empty.
    """
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    needed = {"commodity", "tier", "period", "p_ladder_cum_realized_mwh"}
    if df.empty or not needed.issubset(df.columns):
        return {}
    out: dict[tuple[str, int, str], float] = {}
    for _, row in df.iterrows():
        try:
            tier = int(row["tier"])
            val = float(row["p_ladder_cum_realized_mwh"])
        except (ValueError, TypeError):
            continue
        out[(str(row["commodity"]), tier, str(row["period"]))] = val
    return out


def _load_prior_cum_sim_hours(path: Path) -> dict[str, float]:
    """Return ``{period: cum_hours}`` from the previous roll's accumulator."""
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    needed = {"period", "p_ladder_cum_sim_hours"}
    if df.empty or not needed.issubset(df.columns):
        return {}
    return {
        str(r["period"]): float(r["p_ladder_cum_sim_hours"])
        for _, r in df.iterrows()
    }


# ---------------------------------------------------------------------------
# Writer
# ---------------------------------------------------------------------------


def _horizon_and_realized_hours(
    step_duration: dict[tuple[str, str], float],
    realized_set: set[tuple[str, str]],
) -> tuple[dict[str, float], dict[str, float]]:
    """Compute per-period ``(horizon_hours, realized_hours)``.

    Horizon hours sum ``step_duration`` over every (d, t) this roll's
    dt loaded from ``steps_in_use.csv``.  Realized hours sum only over
    the (d, t) pairs listed in ``realized_dispatch.csv``.
    """
    horizon: dict[str, float] = {}
    realized: dict[str, float] = {}
    for (d, t), dur in step_duration.items():
        horizon[d] = horizon.get(d, 0.0) + dur
        if (d, t) in realized_set:
            realized[d] = realized.get(d, 0.0) + dur
    # Make sure every horizon period has a realized entry (possibly 0.0)
    # so iteration below never hits a missing key.
    for d in horizon:
        realized.setdefault(d, 0.0)
    return horizon, realized


def _v_trade_realized_mwh_this_roll(
    v_trade_df: pd.DataFrame,
    unitsize: dict[str, float],
    horizon_hours: dict[str, float],
    realized_hours: dict[str, float],
    ladder_commodities: set[str],
) -> dict[tuple[str, int, str], float]:
    """Sum realized MWh contributions for THIS roll per (commodity, tier, period).

    Uniform-split assumption: v_trade's period-level decision attributes
    proportionally to realized hours of the roll's view of the period.
    ``v_trade_df`` row index is ``(solve, period)``; columns are
    ``(commodity, node, tier)``.  Empty frame → empty dict.
    """
    out: dict[tuple[str, int, str], float] = {}
    if v_trade_df.empty:
        return out
    for row_key, row in v_trade_df.iterrows():
        period = str(row_key[-1]) if isinstance(row_key, tuple) else str(row_key)
        hz = horizon_hours.get(period, 0.0)
        if hz <= 0.0:
            continue
        rz = realized_hours.get(period, 0.0)
        if rz <= 0.0:
            # This roll didn't realize any of period d — v_trade in d is
            # lookahead only, so no MWh to accumulate.
            continue
        fraction = rz / hz
        for col_key, val in row.items():
            if pd.isna(val) or val == 0.0:
                continue
            if not isinstance(col_key, tuple) or len(col_key) < 3:
                continue
            commodity = str(col_key[0])
            if commodity not in ladder_commodities:
                continue
            try:
                tier = int(col_key[-1])
            except (ValueError, TypeError):
                continue
            us = unitsize.get(commodity, 1.0)
            key = (commodity, tier, period)
            out[key] = out.get(key, 0.0) + float(val) * us * fraction
    return out


def write_ladder_rolling_accumulators(
    h: "highspy.Highs",
    *,
    solve_name: str,
    work_folder: Path,
) -> list[Path]:
    """Write both ladder rolling accumulator CSVs.

    Uses uniform-split assumption: realized MWh of period d in this roll =
    v_trade_k[c, n, d, i] × unitsize × (realized_sim_hours_d / horizon_sim_hours_d)
    — i.e., the LP's period-level v_trade decision is assumed to distribute
    uniformly across the horizon's hours, and the realized portion is a
    proportional slice.

    Returns [path_to_cum_realized_mwh_csv, path_to_cum_sim_hours_csv].
    """
    mwh_path = work_folder / "solve_data" / "ladder_cum_realized_mwh.csv"
    hrs_path = work_folder / "solve_data" / "ladder_cum_sim_hours.csv"

    ladder_tiers = _load_finite_ladder_tiers(work_folder)
    ladder_commodities = {c for (c, _i) in ladder_tiers}

    # Always write at least header-only CSVs so the mod's table-data-in
    # blocks always find the files (mod defaults cover the empty case).
    if not ladder_commodities:
        mwh_path.write_text(
            "commodity,tier,period,p_ladder_cum_realized_mwh\n"
        )
        # Still need cum_sim_hours — even without ladders the file must
        # exist.  Zero rows → mod default 0 → f_d_k[d] = horizon / (share
        # * 8760), which is 1.0 on a full single solve.
        hrs_path.write_text("period,p_ladder_cum_sim_hours\n")
        return [mwh_path, hrs_path]

    first_solve = _is_first_solve(work_folder)
    prior_mwh = {} if first_solve else _load_prior_cum_realized_mwh(mwh_path)
    prior_hrs = {} if first_solve else _load_prior_cum_sim_hours(hrs_path)

    step_duration = _load_step_duration(work_folder)
    realized_set = _load_realized_set(
        work_folder / "solve_data" / "realized_dispatch.csv"
    ) or set()
    horizon_hours, realized_hours = _horizon_and_realized_hours(
        step_duration, realized_set
    )

    unitsize = _load_commodity_unitsize(work_folder)

    v_trade_df = extract_variable(
        h,
        "v_trade",
        ("commodity", "node"),
        solve_name=solve_name,
        has_time=False,
        trailing_col_names=("tier",),
    )
    this_roll_mwh = _v_trade_realized_mwh_this_roll(
        v_trade_df, unitsize, horizon_hours, realized_hours,
        ladder_commodities,
    )

    # --- Build updated cum_realized_mwh table -----------------------------
    # Seed with prior rows, then bump by this roll's contributions.  Only
    # emit rows for (c, i) actually in ladder_tiers (finite tiers of
    # ladder commodities) — infinite / non-ladder tiers don't need a row.
    combined: dict[tuple[str, int, str], float] = dict(prior_mwh)
    for key, val in this_roll_mwh.items():
        # Only write rows for finite tiers; this_roll_mwh is already
        # filtered to ladder commodities via ladder_commodities.
        if (key[0], key[1]) not in ladder_tiers:
            continue
        combined[key] = combined.get(key, 0.0) + val

    mwh_rows = sorted(
        ((c, i, d, v) for (c, i, d), v in combined.items()
         if (c, i) in ladder_tiers),
        key=lambda r: (r[0], r[1], r[2]),
    )
    mwh_df = pd.DataFrame(
        mwh_rows,
        columns=["commodity", "tier", "period", "p_ladder_cum_realized_mwh"],
    )
    mwh_df.to_csv(mwh_path, index=False, float_format="%.8g")

    # --- Build updated cum_sim_hours table --------------------------------
    # Prior + realized hours from THIS roll.  Include every period
    # appearing in either.
    updated_hrs: dict[str, float] = dict(prior_hrs)
    for d, hrs in realized_hours.items():
        updated_hrs[d] = updated_hrs.get(d, 0.0) + hrs
    hrs_rows = sorted(updated_hrs.items(), key=lambda kv: kv[0])
    hrs_df = pd.DataFrame(
        hrs_rows, columns=["period", "p_ladder_cum_sim_hours"],
    )
    hrs_df.to_csv(hrs_path, index=False, float_format="%.8g")

    _logger.info(
        "wrote %s (%d rows) and %s (%d rows)",
        mwh_path, len(mwh_df), hrs_path, len(hrs_df),
    )
    return [mwh_path, hrs_path]


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def write_cumulative_handoffs(
    h: "highspy.Highs",
    *,
    solve_name: str,
    work_folder: Path,
) -> list[Path]:
    """Write every cumulative-quota handoff CSV.

    Parallels :func:`flextool.process_outputs.handoff_writers.write_all_handoffs`.
    Currently just the ladder rolling accumulators; the CO2 cumulative
    handoff (``co2_max_total``) will mirror this structure in a
    follow-up commit.
    """
    written: list[Path] = []
    try:
        written.extend(
            write_ladder_rolling_accumulators(
                h, solve_name=solve_name, work_folder=work_folder,
            )
        )
    except Exception as exc:  # noqa: BLE001
        _logger.warning(
            "write_ladder_rolling_accumulators failed: %s", exc
        )
    return written
