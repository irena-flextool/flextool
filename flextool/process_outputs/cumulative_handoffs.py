"""
Rolling cumulative-quota handoff writers.

These writers maintain a running balance across rolling solves for
commodity-ladder tiers whose ``price_method == 'price_ladder_cumulative'``.
The cap on such a tier is a single MWh total across the whole model
horizon; in a rolling run each solve only sees a window, so the
per-roll constraint RHS must subtract what prior rolls already
consumed and add this roll's share of the total cap.

Formula (per (commodity, tier) with a finite total cap)::

    allot[c, i]          = total_cap[c, i] * span_weight / total_weight
    new_remaining[c, i]  = prior_remaining[c, i] + allot[c, i]
                         - consumption_in_realized_span[c, i]

Where:

* ``total_weight`` — written once on the first solve by ``flextool.mod``
  into ``solve_data/cumulative_weight_total.csv``.  Equals
  ``sum{d in distinct periods of dt_complete}
     p_years_represented_d[d] / complete_period_share_of_year[d]``.
* ``span_weight`` — sum of the same per-period weight across the
  realized span of THIS roll (the periods emitted in
  ``solve_data/realized_dispatch.csv``).
* ``prior_remaining`` — value in the previous roll's
  ``solve_data/cumulative_ladder_remaining.csv``.  Absent → 0.0 on
  the first solve; the seed CSV there is header-only.
* ``consumption_in_realized_span`` —
  ``sum over (n, d) in realized-periods, i in tier of
      v_trade[c, n, d, i] * p_commodity_unitsize[c]
        * p_years_represented_d[d] / complete_period_share_of_year[d]``.

Because ``v_trade`` is period-level (no time, no branch), the
consumption formula has **no** ``step_duration`` and **no**
``p_rp_cost_weight`` — the plan document predates the v_trade design
and its per-(d,t) formulas must be re-derived to match this period
indexing.

Hooked into ``solver_runner._run_highs_or_cplex`` after
:func:`flextool.process_outputs.handoff_writers.write_all_handoffs`.
HiGHS-only — the CPLEX fallback still writes cumulative CSVs via the
legacy glpsol phase-3 pipeline.
"""
from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

from flextool.process_outputs.handoff_writers import (
    _is_first_solve,
    _load_complete_period_share_of_year,
)
from flextool.process_outputs.read_highs_solution import (
    _load_realized_set,
    extract_variable,
)

if TYPE_CHECKING:
    import highspy

_logger = logging.getLogger(__name__)

# Matches flextool.mod's "infinite / inactive" sentinel
# (see NOTES_commit2_ladder_decisions.md).  Tiers with total_cap at or
# above this value are skipped entirely — they're the infinite tail.
_INFINITE_TIER_THRESHOLD = 1e29


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


def _load_price_ladder_cumulative(
    work_folder: Path,
) -> dict[tuple[str, int], float]:
    """Return ``{(commodity, tier): total_cap}`` for every
    ``price_ladder_cumulative`` tier with a finite cap.

    Infinite tiers (quantity >= 1e29 sentinel) are dropped — they're
    the tail that absorbs overflow and never binds the cumulative
    constraint.
    """
    methods = _load_price_methods(work_folder)
    cumulative_commodities = {
        c for c, m in methods.items() if m == "price_ladder_cumulative"
    }
    if not cumulative_commodities:
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
        if c not in cumulative_commodities:
            continue
        try:
            tier = int(row["tier"])
        except (ValueError, TypeError):
            continue
        try:
            q = float(row["quantity"])
        except (ValueError, TypeError):
            continue
        if not math.isfinite(q) or q >= _INFINITE_TIER_THRESHOLD:
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


def _load_years_represented_d(work_folder: Path) -> dict[str, float]:
    """Return ``{period: p_years_represented_d}`` from
    ``solve_data/p_years_represented_d.csv``.  The file is written by
    the mod's phase-1 printf once (first solve) and left alone on
    later solves, so it covers every period in
    ``d_realize_dispatch_or_invest`` across the full horizon."""
    path = work_folder / "solve_data" / "p_years_represented_d.csv"
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    if df.empty or "period" not in df.columns or "value" not in df.columns:
        return {}
    # Multiple rolls may have appended rows for the same period — dedup
    # by keeping the last value (they must match anyway; the parameter
    # is model-wide constant).
    return dict(zip(df["period"].astype(str), df["value"].astype(float)))


def _load_cumulative_weight_total(work_folder: Path) -> float | None:
    """Return the scalar ``total_weight`` from
    ``solve_data/cumulative_weight_total.csv`` (one-line file written
    by the mod's phase-1 first-solve block).  Missing / empty / zero
    → None (caller treats this as "skip, no arithmetic possible")."""
    path = work_folder / "solve_data" / "cumulative_weight_total.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    if df.empty or "total_weight" not in df.columns:
        return None
    try:
        val = float(df["total_weight"].iloc[0])
    except (ValueError, TypeError):
        return None
    if val <= 0.0 or not math.isfinite(val):
        return None
    return val


def _load_prior_cumulative_ladder_remaining(
    path: Path,
) -> dict[tuple[str, int], float]:
    """Return ``{(commodity, tier): remaining}`` from the previous
    roll's output CSV (or the header-only seed on first solve → empty).

    Graceful on any shape issue — missing columns / unparseable tiers
    are skipped silently (the caller then treats the entry as prior=0).
    """
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    needed = {"commodity", "tier", "p_cumulative_ladder_remaining"}
    if df.empty or not needed.issubset(df.columns):
        return {}
    out: dict[tuple[str, int], float] = {}
    for _, row in df.iterrows():
        try:
            tier = int(row["tier"])
            val = float(row["p_cumulative_ladder_remaining"])
        except (ValueError, TypeError):
            continue
        out[(str(row["commodity"]), tier)] = val
    return out


# ---------------------------------------------------------------------------
# Arithmetic helpers
# ---------------------------------------------------------------------------


def _span_weight(
    realized_periods: set[str],
    years_map: dict[str, float],
    period_share: dict[str, float],
) -> float:
    """Sum ``years_represented[d] / period_share[d]`` over realized
    periods.  The denominator of the allotment ratio (``total_weight``)
    uses the same per-period weight summed over ``dt_complete``'s
    distinct periods, so the ratio is a proper proportional share."""
    total = 0.0
    for d in realized_periods:
        share = period_share.get(d)
        if share is None or share <= 0.0:
            continue
        yrs = years_map.get(d, 0.0)
        total += yrs / share
    return total


def _ladder_consumption(
    v_trade_df: pd.DataFrame,
    realized_periods: set[str],
    years_map: dict[str, float],
    period_share: dict[str, float],
    unitsize: dict[str, float],
) -> dict[tuple[str, int], float]:
    """Sum v_trade × unitsize × (years / share) over realized periods
    and all nodes for each (commodity, tier).

    ``v_trade_df`` row index: ``(solve, period)``.  Column MultiIndex:
    ``(commodity, node, tier)`` — the ``tier`` level sits last because
    ``v_trade`` is declared ``v_trade[c, n, d, i]`` in the mod and the
    extractor drops ``d`` into the row index (see VariableSpec in
    ``read_highs_solution.py``).  Empty frame → zero consumption for
    every key (handled by ``dict.get(..., 0.0)`` downstream).
    """
    out: dict[tuple[str, int], float] = {}
    if v_trade_df.empty:
        return out
    for row_key, row in v_trade_df.iterrows():
        # row_key is (solve, period) for has_period=True, has_time=False.
        if isinstance(row_key, tuple):
            period = str(row_key[-1])
        else:
            period = str(row_key)
        if period not in realized_periods:
            continue
        share = period_share.get(period)
        if share is None or share <= 0.0:
            continue
        yrs = years_map.get(period, 0.0)
        factor_d = yrs / share
        if factor_d == 0.0:
            continue
        for col_key, val in row.items():
            if pd.isna(val) or val == 0.0:
                continue
            # col_key: (commodity, node, tier).  ``tier`` emerges from
            # ``trailing_col_names`` in the VariableSpec — often a str
            # in the wide frame; coerce defensively.
            if not isinstance(col_key, tuple) or len(col_key) < 3:
                continue
            commodity = str(col_key[0])
            try:
                tier = int(col_key[-1])
            except (ValueError, TypeError):
                continue
            us = unitsize.get(commodity, 1.0)
            key = (commodity, tier)
            out[key] = out.get(key, 0.0) + float(val) * us * factor_d
    return out


# ---------------------------------------------------------------------------
# Writer
# ---------------------------------------------------------------------------


def write_cumulative_ladder_remaining(
    h: "highspy.Highs",
    *,
    solve_name: str,
    work_folder: Path,
) -> Path:
    """Write ``solve_data/cumulative_ladder_remaining.csv`` from the
    solved HiGHS instance.

    See module docstring for the running-balance formula.  Called once
    per roll by :func:`write_cumulative_handoffs`; overwrites the file
    whole (same pattern as :func:`handoff_writers.write_p_entity_divested`).
    """
    out_path = work_folder / "solve_data" / "cumulative_ladder_remaining.csv"

    ladder_total = _load_price_ladder_cumulative(work_folder)
    if not ladder_total:
        # No cumulative tiers in this model → keep the seed header so
        # the mod's ``table data IN`` still finds the file.
        out_path.write_text(
            "commodity,tier,p_cumulative_ladder_remaining\n"
        )
        return out_path

    total_weight = _load_cumulative_weight_total(work_folder)
    if total_weight is None:
        _logger.warning(
            "total_weight missing / zero in %s — emitting seed header only "
            "(constraint stays inactive on next roll)",
            work_folder / "solve_data" / "cumulative_weight_total.csv",
        )
        out_path.write_text(
            "commodity,tier,p_cumulative_ladder_remaining\n"
        )
        return out_path

    prior = (
        {}
        if _is_first_solve(work_folder)
        else _load_prior_cumulative_ladder_remaining(out_path)
    )
    realized_set = _load_realized_set(
        work_folder / "solve_data" / "realized_dispatch.csv"
    )
    realized_periods: set[str] = (
        {p for (p, _t) in realized_set} if realized_set else set()
    )
    years_map = _load_years_represented_d(work_folder)
    period_share = _load_complete_period_share_of_year(work_folder)
    unitsize = _load_commodity_unitsize(work_folder)

    span_weight = _span_weight(realized_periods, years_map, period_share)

    v_trade_df = extract_variable(
        h,
        "v_trade",
        ("commodity", "node"),
        solve_name=solve_name,
        has_time=False,
        trailing_col_names=("tier",),
    )
    consumption = _ladder_consumption(
        v_trade_df, realized_periods, years_map, period_share, unitsize
    )

    rows: list[tuple[str, int, float]] = []
    for (c, tier), total_cap in sorted(ladder_total.items()):
        prior_q = prior.get((c, tier), 0.0)
        allot = total_cap * span_weight / total_weight
        consumed = consumption.get((c, tier), 0.0)
        # Negative remaining is legal — it means an earlier roll
        # overspent.  The next roll's LP will then strictly exclude
        # the tier (constraint RHS < 0 with LHS >= 0).
        rows.append((c, tier, prior_q + allot - consumed))

    out = pd.DataFrame(
        rows, columns=["commodity", "tier", "p_cumulative_ladder_remaining"]
    )
    out.to_csv(out_path, index=False, float_format="%.8g")
    _logger.info(
        "wrote %s (%d rows, span_weight=%.6g / total_weight=%.6g)",
        out_path, len(out), span_weight, total_weight,
    )
    return out_path


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
    Currently just the ladder writer; the CO2 cumulative handoff
    (``co2_max_total``) is on the roadmap but deferred — see
    ``project_commodity_ladder.md`` step 4f.
    """
    written: list[Path] = []
    for fn in (write_cumulative_ladder_remaining,):
        try:
            written.append(
                fn(h, solve_name=solve_name, work_folder=work_folder)
            )
        except Exception as exc:  # noqa: BLE001
            _logger.warning("%s failed: %s", fn.__name__, exc)
    return written
