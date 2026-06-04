"""Per-roll realized-slice persistence (multi-solve output union, stage 2).

Multi-sub-solve runs (rolling / nested / multi-period) used to write
outputs covering only the LAST sub-solve's realized window: the
per-roll ``flex_data`` / ``solution`` are nulled by memory slimming on
prior steps, so by output time only the last roll survives.

Variables already avoid this: each roll writes a realized-filtered
per-solve parquet to ``output_raw/`` (``write_variable_parquet``,
filename ``{name}__{solve}.parquet``) and the reader concatenates every
roll.  This module extends that exact pattern to **parameters** (stage
2a) — and, in stage 2b, to **sets** — so the per-roll realized slice is
persisted off-heap while the data is still live.

Design (orchestrator decisions D1/D2/D3):

* **D1** — each per-roll attribute slice is written as parquet at
  ``output_raw/<attr>__<solve>.parquet``, byte-for-byte mirroring the
  variable convention (``read_highs_solution.write_variable_parquet``).
  No size-based CSV fallback.
* **D2** — :func:`read_parameters` is re-run once per roll at the
  per-roll write hook (it already calls every slice builder, including
  the live-solution ``entity_all_capacity``); the resulting solve-keyed
  attributes are realized-filtered and persisted.  We do NOT factor the
  individual builders out of the 1786-line monolith — re-running it per
  roll is runtime-equivalent to the proven ``keep_solutions`` reference
  (which also calls ``read_parameters`` per step).
* **D3** — ``entity_all_capacity`` rides the same parquet route via the
  live-solution builder, filtered to the solve's realized periods.

Realized filter (per :mod:`...read_parameters` ``read_parameters_multi``
pre-concat logic, replicated here at persist time so the stage-3 union
is a clean disjoint concat):

* ``(period, time)``-keyed attrs → keep rows whose ``(period, time)`` is
  in the roll's realized dispatch set (``flex_data.realized_dispatch``).
* ``(period)``-keyed attrs → keep rows whose ``period`` is in the roll's
  realized-invest set.
* Three attr-specific hacks are preserved verbatim
  (``entity_lifetime_fixed_cost[_divest]``, ``entity_all_existing``,
  and the ``entity_annual_discounted`` emptiness gate) — see
  :func:`_apply_realized_filter`.

The 6 output-dead solve-keyed params (``flow_min``, ``flow_max``,
``years_from_start_d``, ``entity_max_units``, ``node_annual_flow``,
``group_capacity_margin``) are NOT persisted (spec §2a / Resolved C).
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

from flextool.lean_parquet import write_lean_parquet

if TYPE_CHECKING:
    from polar_high import Solution

    from flextool.engine_polars.input import FlexData

_logger = logging.getLogger(__name__)


# Solve-keyed params that are built but read by no ``out_*`` / ``calc_*``
# consumer — not persisted (spec §2a / Resolved C).
_DEAD_PARAMS: frozenset[str] = frozenset(
    {
        "flow_min",
        "flow_max",
        "years_from_start_d",
        "entity_max_units",
        "node_annual_flow",
        "group_capacity_margin",
    }
)

# Attr-specific realized-filter hack families (preserved from
# ``read_parameters_multi``).
_LIFETIME_ATTRS: frozenset[str] = frozenset(
    {"entity_lifetime_fixed_cost", "entity_lifetime_fixed_cost_divest"}
)
_EXISTING_ATTRS: frozenset[str] = frozenset({"entity_all_existing"})
# Emptiness-gate attrs and the FlexData carrier that decides each.
_ANNUAL_SRC_FIELD: dict[str, str] = {
    "entity_annual_discounted": "ed_entity_annual_discounted",
    "entity_annuity": "ed_entity_annual_discounted",
    "entity_annual_divest_discounted": "ed_entity_annual_divest_discounted",
}


# ---------------------------------------------------------------------------
# Shared parquet writer helper (reused by the stage-2b set writer)
# ---------------------------------------------------------------------------


def write_realized_slice_parquet(
    frame: "pd.DataFrame | pd.Series",
    *,
    attr: str,
    solve_name: str,
    output_dir: Path | str,
) -> Path:
    """Persist one per-roll realized slice to ``output_dir`` as parquet.

    Filename is ``{attr}__{solve_name}.parquet`` — the same convention
    :func:`...read_highs_solution.write_variable_parquet` uses for
    variables, so the stage-3 reader can union params, sets and
    variables uniformly.  The ``solve`` identity lives only in the
    filename (the on-disk frame keeps whatever ``solve`` index level it
    already carries — stage 3 drops it at union time).

    A pandas ``Series`` is converted to a one-column DataFrame (its
    ``name`` becomes the column label) so the lean-parquet round-trip is
    well defined; the stage-3 reader restores the Series shape.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if isinstance(frame, pd.Series):
        df = frame.to_frame()
    else:
        df = frame
    path = output_dir / f"{attr}__{solve_name}.parquet"
    write_lean_parquet(df, path)
    _logger.debug(
        "Wrote realized slice %s for solve '%s' -> %s (shape %s)",
        attr, solve_name, path, df.shape,
    )
    return path


# ---------------------------------------------------------------------------
# Realized-set extraction from the live FlexData
# ---------------------------------------------------------------------------


def _realized_dt_set(flex_data: "FlexData") -> set[tuple[str, str]]:
    """Return ``{(period, time), …}`` realized by this roll.

    Source is the in-memory ``flex_data.realized_dispatch`` ``(period,
    step)`` frame — the same carrier the variable writer prefers
    (``read_highs_solution._load_realized_set``).  Empty / absent →
    empty set (this roll realized nothing on the dispatch axis).
    """
    rd = getattr(flex_data, "realized_dispatch", None)
    if rd is None or getattr(rd, "height", 0) == 0:
        return set()
    pdf = rd.select("period", "step").to_pandas()
    return set(
        zip(
            pdf["period"].astype(str).tolist(),
            pdf["step"].astype(str).tolist(),
        )
    )


def _realized_dispatch_periods(flex_data: "FlexData") -> set[str]:
    """Return ``{period, …}`` appearing in the roll's realized dispatch."""
    rd = getattr(flex_data, "realized_dispatch", None)
    if rd is None or getattr(rd, "height", 0) == 0:
        return set()
    return set(
        rd.select("period").unique().to_pandas()["period"].astype(str).tolist()
    )


def _existing_filter_periods(flex_data: "FlexData") -> set[str]:
    """``entity_all_existing`` hack periods: realized-dispatch periods ∪
    the ``"d"`` values of ``ed_invest_set`` / ``ed_divest_set``.

    Replicates ``read_parameters_multi`` (read_parameters.py:1681-1705).
    """
    periods = _realized_dispatch_periods(flex_data)
    for src_attr in ("ed_invest_set", "ed_divest_set"):
        src = getattr(flex_data, src_attr, None)
        if src is not None and getattr(src, "height", 0) > 0 and "d" in src.columns:
            periods.update(
                src.select("d").unique().to_pandas()["d"].astype(str).tolist()
            )
    return periods


def _annual_carrier_empty(flex_data: "FlexData", src_attr: str) -> bool:
    """True iff the annual-NPV carrier ``src_attr`` is empty/None — i.e.
    this is a dispatch-only step (read_parameters.py:1707-1752)."""
    src_param = getattr(flex_data, src_attr, None)
    return (
        src_param is None
        or getattr(src_param, "frame", None) is None
        or getattr(src_param.frame, "height", 0) == 0
    )


# ---------------------------------------------------------------------------
# Per-attribute realized filter
# ---------------------------------------------------------------------------


def _index_has_time(obj: "pd.DataFrame | pd.Series") -> bool:
    idx = obj.index
    return isinstance(idx, pd.MultiIndex) and "time" in (idx.names or ())


def _filter_by_dt(
    obj: "pd.DataFrame | pd.Series",
    realized_dt: set[tuple[str, str]],
) -> "pd.DataFrame | pd.Series":
    """Keep rows whose ``(period, time)`` is realized."""
    periods = obj.index.get_level_values("period").astype(str)
    times = obj.index.get_level_values("time").astype(str)
    mask = pd.Series(
        [pt in realized_dt for pt in zip(periods, times)],
        index=obj.index,
    )
    return obj[mask.to_numpy()]


def _filter_by_period(
    obj: "pd.DataFrame | pd.Series",
    realized_periods: set[str],
) -> "pd.DataFrame | pd.Series":
    """Keep rows whose ``period`` is realized."""
    periods = obj.index.get_level_values("period").astype(str)
    mask = periods.isin(realized_periods)
    return obj[mask]


def _apply_realized_filter(
    attr: str,
    obj: "pd.DataFrame | pd.Series",
    flex_data: "FlexData",
    *,
    realized_dt: set[tuple[str, str]],
    realized_invest_periods: set[str],
) -> "pd.DataFrame | pd.Series":
    """Return the roll's realized slice of ``obj`` for attribute ``attr``.

    Dispatches the three preserved attr-specific hacks first, then the
    general realized intersection for every other solve-keyed attr.
    The frame's ``solve`` index level is left intact (stage 3 drops it).
    """
    # Hack 3 — emptiness gate (NOT a period filter): a dispatch-only
    # step's densified-zero annual frame is cleared so it can't compete
    # with the parent invest step's real values at union time.
    if attr in _ANNUAL_SRC_FIELD:
        if _annual_carrier_empty(flex_data, _ANNUAL_SRC_FIELD[attr]):
            return obj.iloc[0:0]
        return obj

    # Hack 1 — entity_lifetime_fixed_cost[_divest]: filter by the roll's
    # realized-DISPATCH periods (not realized-invest), so the committing
    # step's active per-period value is the sole contributor.
    if attr in _LIFETIME_ATTRS:
        return _filter_by_period(obj, _realized_dispatch_periods(flex_data))

    # Hack 2 — entity_all_existing: realized-dispatch periods ∪ the "d"s
    # of ed_invest_set / ed_divest_set.
    if attr in _EXISTING_ATTRS:
        step_periods = _existing_filter_periods(flex_data)
        if not step_periods:
            # Step realizes nothing — legacy behaviour: leave as-is.
            return obj
        return _filter_by_period(obj, step_periods)

    # General realized intersection for every other solve-keyed attr.
    if _index_has_time(obj):
        return _filter_by_dt(obj, realized_dt)
    return _filter_by_period(obj, realized_invest_periods)


# ---------------------------------------------------------------------------
# Top-level per-roll param persistence
# ---------------------------------------------------------------------------


def write_all_params_realized_slice(
    flex_data: "FlexData",
    solution: "Solution",
    *,
    solve_name: str,
    output_dir: Path | str,
    realized_invest_periods: "set[str] | None" = None,
) -> list[Path]:
    """Persist this roll's realized slice of every per-roll-VARYING param.

    Calls :func:`...read_parameters.read_parameters` once (D2), selects
    the solve-keyed (varying) attributes via the structural
    ``_has_solve_level`` test — minus the 6 output-dead params — applies
    the realized filter + the three preserved hacks (incl. the
    live-solution ``entity_all_capacity``, already built inside
    ``read_parameters``), and writes each realized slice to parquet.

    Static (solve-invariant) attrs carry no ``solve`` index level and
    are skipped here — stage 3 takes them once.

    Parameters
    ----------
    realized_invest_periods : set[str] | None
        The roll's realized-invest periods (``period``-keyed attrs).
        When ``None``, falls back to the realized-dispatch periods —
        the same fallback the per-solve realized-invest writer uses
        when no explicit invest set exists.

    Returns the list of parquet paths written.
    """
    from flextool.process_outputs.read_parameters import (
        _has_solve_level,
        read_parameters,
    )

    output_dir = Path(output_dir)

    par = read_parameters(flex_data, solution, solve_name=solve_name)

    realized_dt = _realized_dt_set(flex_data)
    if realized_invest_periods is None:
        realized_invest_periods = _realized_dispatch_periods(flex_data)

    written: list[Path] = []
    for attr, obj in vars(par).items():
        if attr in _DEAD_PARAMS:
            continue
        if not _has_solve_level(obj):
            # Static / invariant — taken once at union time (stage 3).
            continue
        sliced = _apply_realized_filter(
            attr, obj, flex_data,
            realized_dt=realized_dt,
            realized_invest_periods=realized_invest_periods,
        )
        path = write_realized_slice_parquet(
            sliced, attr=attr, solve_name=solve_name, output_dir=output_dir,
        )
        written.append(path)

    _logger.debug(
        "Persisted %d realized param slices for solve '%s' to %s",
        len(written), solve_name, output_dir,
    )
    return written


__all__ = [
    "write_all_params_realized_slice",
    "write_realized_slice_parquet",
]
