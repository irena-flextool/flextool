"""
Direct HiGHS solution → parquet extractor.

Pipeline: HiGHS (in-memory) → VARIABLE_SPECS harvest →
``output_parquet/*.parquet`` → downstream pandas.

Design
------
HiGHS exposes the MPS column names through
``highspy.Highs.allVariableNames()`` and the solution through
``Highs.getSolution().col_value``.  The two arrays are index-aligned.

MPS variable names look like::

    <var_name>[<idx1>,<idx2>,...,<period>,<time>]   # time-indexed vars
    <var_name>[<idx1>,...,<period>]                 # period-only vars (e.g. v_invest)

Splitting on '[', ',' and ']' recovers the indices.  This assumes entity
names never contain commas or brackets — a requirement for unambiguous
MPS emission.

We retain only the ``dt_realize_dispatch`` timesteps — rolling-window
solves compute a longer horizon but only realize the first chunk.  The
filter runs inside the single pass over HiGHS names (O(1) set lookup)
so non-realized rows are never materialised.

Storage layout: wide.  Row MultiIndex = ``(solve, period, time)`` (or
``(solve, period)`` for non-time vars), column (Multi)Index = the
remaining variable indices.  Matches the shape of
``read_variables.read_variables`` outputs exactly, so downstream
``calc_*.py`` code is unchanged.  Persisted via
``flextool.lean_parquet.write_lean_parquet`` — compact level-name
metadata in the parquet footer.

Public API
----------
* :data:`VARIABLE_SPECS` — list of :class:`VariableSpec` describing every
  variable that has parquet coverage.  Add a new line here to add a new
  variable.
* :func:`extract_variable` — single variable → wide DataFrame.
* :func:`write_variable_parquet` — single variable → parquet file.
* :func:`write_all_variables` — iterate :data:`VARIABLE_SPECS` and write
  one parquet per variable for a given solve.
* Loading: use :func:`flextool.lean_parquet.read_lean_parquet` directly.

Usage from solver_runner (after a successful ``h.run()``)::

    from flextool.process_outputs.read_highs_solution import write_all_variables
    write_all_variables(
        h, solve_name=current_solve,
        output_dir=wf / "output_raw",
        realized_dispatch_csv=wf / "solve_data/realized_dispatch.csv",
    )
"""
from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import NamedTuple, Sequence, TYPE_CHECKING

import pandas as pd
import polars as pl

from flextool.lean_parquet import write_lean_parquet

if TYPE_CHECKING:
    import highspy

    from flextool.engine.preprocess.flex_data import FlexData

_logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Provider-aware lookup helper — Provider-first, then ``None`` (caller
# falls back to its own disk read).  Provider key uses the
# parent-qualified convention (``"<parent>/<basename>"`` without
# ``.csv``).

def _provider_lookup(provider: "object | None", path: "Path | str"):
    """Return the polars frame for *path* sourced from the Provider, or
    ``None`` when the Provider doesn't carry it (caller falls back to
    its own disk read).
    """
    p = Path(path)
    parent = p.parent.name
    stem = p.stem
    name = f"{parent}/{stem}" if parent else stem
    if provider is not None and provider.has(name):
        return provider.get(name)
    return None


# ---------------------------------------------------------------------------
# Variable registry
# ---------------------------------------------------------------------------


class VariableSpec(NamedTuple):
    """Describes one HiGHS result quantity for parquet extraction.

    Covers three source arrays on the solved ``Highs`` instance:

    * ``col_value`` (default) — primal variable values, keyed by
      ``allVariableNames()``.
    * ``col_dual`` — variable reduced costs, same keying as
      ``col_value``.  Used for ``v_invest.dual`` / ``v_divest.dual``.
    * ``row_dual`` — constraint dual values, keyed by
      ``getLp().row_names_``.  Used for investment-cap duals,
      ``nodeBalance_eq``, CO2-limit duals, …

    Attributes
    ----------
    name:
        Name as it appears in the MPS.  Variable name for
        ``col_value``/``col_dual`` sources; constraint name for
        ``row_dual``.  Must match the prefix in ``<name>[idx1,...]``.
    col_names:
        Names of the leading "column" indices before the trailing
        ``period[, time]`` (or just ``period``, or nothing — see
        :attr:`has_time` and :attr:`has_period`).
    has_time:
        True when indexed by both period and time.  False for
        period-only quantities.
    has_period:
        True (default) when the trailing index (after ``col_names``)
        includes at least a period.  False for quantities indexed only
        by the column fields (e.g. ``co2_max_total[g]``) — in that case
        the row index collapses to just ``(solve,)``.
    source:
        Which HiGHS array the values come from — one of
        ``"col_value"`` (default), ``"col_dual"``, ``"row_dual"``.
    value_scale:
        Multiplier applied to every raw value.  Typically ``1e6``
        (``1 / scale_the_objective``) for duals.
    output_name:
        Parquet file prefix.  Defaults to :attr:`name`.  Set when the
        constraint name differs from the desired output identifier —
        e.g. constraint ``maxInvest_entity_period`` is written as
        ``v_dual_maxInvest_period``.
    """

    name: str
    col_names: tuple[str, ...]
    has_time: bool = True
    has_period: bool = True
    source: str = "col_value"
    value_scale: float = 1.0
    output_name: str | None = None
    # Column fields that appear AFTER the period (and time) in the
    # bracket list.  Needed for variables whose declared subscript order
    # puts a column index after the period — e.g. ``v_trade[c, n, d, i]``
    # where ``i`` (tier) is a column but sits to the right of ``d``
    # (period).  Parsed-out values are concatenated onto ``col_names`` to
    # form the full column tuple.
    trailing_col_names: tuple[str, ...] = ()
    # Multi-source fan-out: when the output quantity is the sum of two or
    # more HiGHS variables (e.g. a two-tier slack split into
    # ``vq_foo_primary`` + ``vq_foo_escape``), list the source variable
    # names here.  ``name`` then becomes a pure logical identifier used
    # for the parquet file name; the extractor reads each source
    # separately, aligns on the row+column MultiIndex, and adds them.
    # ``None`` (default) preserves the legacy single-source behaviour.
    # All sources must share the same index shape — ``col_names``,
    # ``has_time``, ``has_period``, ``trailing_col_names`` apply
    # uniformly to every source in the tuple.
    derived_from: tuple[str, ...] | None = None
    # Agent 9 — row-scaling un-scaling.  When non-None, the extracted
    # frame is multiplied element-wise by the corresponding row scaler
    # read from ``solve_data/{node,group}_capacity_for_scaling.csv``
    # before parquet emission.  Column names in ``col_names`` supply
    # the entity axis of the scaler; scalers are per-(entity, period).
    # Recognised values:
    #   * "node_cap"  — multiply cell[(d, t), n] by node_cap[n, d].
    #                   Used for vq_state_up / vq_state_down.  Also
    #                   divides by node_cap when the quantity is a dual
    #                   of a row-scaled balance constraint (see
    #                   ``unscale_dual=True`` below for that case).
    #   * "group_cap" — multiply by group_cap[g, d]; used for
    #                   vq_non_synchronous, vq_state_up_group,
    #                   vq_capacity_margin (no t axis).
    # Mode A (flag off): node_cap / group_cap CSVs default to 1 so this
    # is a no-op.  Mode B: recovers absolute CSV magnitudes matching the
    # pre-row-scaling (Agent 1) baselines.
    unscale_by: str | None = None
    # Canonical-row period axis for period-only (``has_period and not
    # has_time``) variables.  ``"dispatch"`` (default) sources the row
    # order from the roll's realized-DISPATCH periods
    # (``flex_data.realized_dispatch`` / ``p_years_from_start_d.csv``);
    # ``"invest"`` sources it from the roll's realized-INVEST periods
    # (``realized_invest_periods_of_current_solve.csv`` / provider).  Only
    # ``v_invest`` / ``v_divest`` use ``"invest"`` — they are decided on
    # the investment axis, so a dispatch-only roll must contribute no
    # rows (see the registry note on those specs).  ``"invest_branches"``
    # is ``"invest"`` plus the forecast branch periods of the realized-
    # invest periods (``vq_capacity_margin``: one slack per forecast branch
    # period of a fan, which no dispatch axis holds).  Ignored for
    # ``has_time`` and ``has_period=False`` variables.
    period_source: str = "dispatch"


# Registry — add a new line here to add a new variable to the pipeline.
#
# ``scale_the_objective`` was a hardcoded ``1e-6`` in ``flextool_base.dat``
# (legacy); Agent 12 centralised it in Python — the value is written per
# solve to ``solve_data/scale_the_objective.csv`` from the Agent-8
# ScaleTable.  Every dual of an objective-scaled constraint needs
# multiplication by ``1/scale_the_objective`` to undo the scaling.
#
# ``_INV_SCALE_THE_OBJECTIVE`` is retained as the **default** multiplier
# (matches the legacy 1e-6 scalar) — used as the sentinel value wired
# into ``VariableSpec.value_scale`` for dual specs.  At write time,
# :func:`_resolve_inv_scale_the_objective` reads the current solve's
# ``solve_data/scale_the_objective.csv`` and replaces this default with
# the live reciprocal so per-solve scalar changes propagate correctly.
_INV_SCALE_THE_OBJECTIVE = 1e6

_DEFAULT_SCALE_THE_OBJECTIVE = 1e-6


def _resolve_inv_scale_the_objective(
    scale_the_objective: float | None = None,
) -> float:
    """Return ``1 / scale_the_objective`` for the current solve.

    Item 163: the output writers never read the work folder, so the disk
    arm (``<work_folder>/solve_data/scale_the_objective.csv``) is gone —
    every caller is in-process and already threads the cascade's live
    ``scale_the_objective`` (Agent 12; emitted by
    :func:`flextool.engine.preprocess.emit._solve_writers.write_scale_the_objective`).
    A ``None`` or non-positive value falls back to ``1 / 1e-6`` — mirrors
    the ``default 1e-6`` clause on ``param scale_the_objective`` in the
    retired GMPL model.
    """
    if scale_the_objective is not None and scale_the_objective > 0:
        return 1.0 / float(scale_the_objective)
    return 1.0 / _DEFAULT_SCALE_THE_OBJECTIVE


# Helper — every investment-constraint-dual entry uses the same scale to
# undo ``scale_the_objective`` and writes its parquet under ``v_dual_<…>``.
def _invest_dual(
    mps_name: str,
    col: str,
    output_suffix: str,
    *,
    has_period: bool = True,
    derived_from: tuple[str, ...] | None = None,
) -> VariableSpec:
    # ``derived_from`` is set for invest-cap families whose producer
    # splits the LHS into process-side (``..._p``) and node-side
    # (``..._n``) constraints (model/core.py / _cumulative_invest.py).  The
    # bracketed entity columns of the two sides are disjoint (process
    # names vs. node names), so ``df.add(fill_value=0.0)`` cleanly
    # unions them into one wide frame at the output boundary.
    return VariableSpec(
        name=mps_name,
        col_names=(col,),
        has_time=False,
        has_period=has_period,
        source="row_dual",
        value_scale=_INV_SCALE_THE_OBJECTIVE,
        output_name=f"v_dual_{output_suffix}",
        derived_from=derived_from,
    )


VARIABLE_SPECS: list[VariableSpec] = [
    # -- Time-indexed decision variables ------------------------------------
    # The variables of _coarse_flows.COARSE_VARS live on coarse steps for their coarse entities; extract_variable
    # copies them to every step.  Every other variable lives on the solve's steps (a coarse node's state is
    # constant within its coarse step by stateConstantCoarse_eq).
    VariableSpec("v_flow",             ("process", "source", "sink")),
    # Reverse-flow auxiliary for method_2way_1var_off arcs.  The signed
    # net flow on such an arc is ``v_flow - v_flow_back``; the output layer
    # folds it into ``r.flow_dt`` (calc_capacity_flows) so the reported
    # connection flow carries the correct sign (negative = sink→source).
    VariableSpec("v_flow_back",        ("process", "source", "sink")),
    # The increase / decrease of v_flow from its ramp predecessor on the arcs with a ramp cost (units of
    # unitsize, as v_flow); a coarse process's change is at the first fine step of its coarse step.
    VariableSpec("v_ramp_up",          ("process", "source", "sink")),
    VariableSpec("v_ramp_down",        ("process", "source", "sink")),
    VariableSpec("v_reserve",          ("process", "reserve", "updown", "node")),
    VariableSpec("v_state",            ("node",)),
    VariableSpec("v_online_linear",    ("process",)),
    VariableSpec("v_startup_linear",   ("process",)),
    VariableSpec("v_shutdown_linear",  ("process",)),
    VariableSpec("v_online_integer",   ("process",)),
    VariableSpec("v_startup_integer",  ("process",)),
    VariableSpec("v_shutdown_integer", ("process",)),
    VariableSpec("v_angle",            ("node",)),

    # -- Time-indexed slack / penalty variables -----------------------------
    # Agent 9 ``unscale_by="node_cap"`` / ``"group_cap"`` un-scales row
    # scaling when ``use_row_scaling=yes`` (no-op in Mode A where the
    # scaler defaults to 1).  See flextool/SLACK_CONVENTION.md for the
    # single-variable slack convention.
    VariableSpec(
        "vq_state_up", ("node",),
        unscale_by="node_cap",
    ),
    VariableSpec(
        "vq_state_down", ("node",),
        unscale_by="node_cap",
    ),
    # Column level names must match the CSV reader
    # (``read_variables._read_from_csv``) so cross-reader mul aligns
    # cleanly — both readers use ('reserve', 'updown', 'node_group').
    # Reserves + inertia pinned to default block (Agent 1.7 V1) — no
    # expansion needed.
    VariableSpec("vq_reserve", ("reserve", "updown", "node_group")),
    VariableSpec("vq_inertia", ("group",)),
    VariableSpec(
        "vq_non_synchronous", ("group",),
        unscale_by="group_cap",
    ),
    # group_loss_share_constraint is emitted at every fine (d, t), so
    # vq_state_up_group lives on the fine timeline directly.
    VariableSpec(
        "vq_state_up_group", ("group",),
        unscale_by="group_cap",
    ),

    # -- Period-only (no time) decision / slack variables -------------------
    # ``v_invest`` / ``v_divest`` are realized on the INVEST axis, not the
    # dispatch axis.  In nested / rolling-dispatch cascades the dispatch
    # rolls realize a period for dispatch but NO investment, yet the
    # densified column union (8e2de938 / a9058c66) still materialises a
    # zero-valued ``(roll, dispatch_period)`` row for every invest-eligible
    # entity.  After ``drop_levels`` strips the ``solve`` level and dedups
    # ``keep='last'``, that trailing dispatch-roll zero overwrites the
    # invest step's real value for the shared period, zeroing the
    # ``invested`` breakdown in ``unit_capacity__d`` (and friends).  Pin
    # the canonical row axis to the roll's realized-INVEST periods so a
    # dispatch-only roll emits no spurious invest rows and the invest
    # step's value is the sole contributor at union time.
    VariableSpec("v_invest", ("entity",), has_time=False, period_source="invest"),
    VariableSpec("v_divest", ("entity",), has_time=False, period_source="invest"),
    # No t axis; the row scaler is still keyed by (g, d).  Item 85: the slack lives on the investment axis, as
    # ``v_invest`` does.  With the dispatch axis a dispatch roll's densified zero row for the period won the
    # ``keep='last'`` union over the investment solve's shortfall, so an invest → dispatch chain reported no
    # shortfall and no penalty.  A fan keeps one slack per forecast branch period (``invest_branches``).
    VariableSpec(
        "vq_capacity_margin", ("group",), has_time=False,
        unscale_by="group_cap", period_source="invest_branches",
    ),

    # -- Commodity-ladder period-level trade ---------------------------------
    # ``v_trade[c, n, d, i]`` — no time, no branch.  ``tier`` sits after
    # the period in the MPS bracket order so it's declared via
    # ``trailing_col_names`` (as opposed to ``col_names`` which are
    # parsed from the leading positions).  Output column MultiIndex is
    # ``(commodity, node, tier)`` — the logical column tuple.
    VariableSpec(
        "v_trade", ("commodity", "node"),
        has_time=False,
        trailing_col_names=("tier",),
    ),

    # -- Investment-cap duals (period-only, simple 1/scale transform) -------
    # Most of these families are emitted by a split process/node producer
    # — ``maxInvest_entity_period_p`` (model._eq_invest_period_caps) for the process arm
    # and ``maxInvest_entity_period_n`` (same function) for the node arm
    # etc. — so the reader matches both via ``derived_from`` and unions
    # the resulting frames.
    _invest_dual("maxInvest_entity_period",          "entity", "maxInvest_period",
                 derived_from=("maxInvest_entity_period_p",
                                "maxInvest_entity_period_n")),
    # ``maxInvest_entity_total`` is asymmetric: process side keeps the
    # bare name, node side gets the ``_n`` suffix
    # (model._eq_invest_total_caps).  Period is summed out inside ``Sum(over=("d",))``
    # — no period component in either constraint name.
    _invest_dual("maxInvest_entity_total",           "entity", "maxInvest_total",
                 has_period=False,
                 derived_from=("maxInvest_entity_total",
                                "maxInvest_entity_total_n")),
    _invest_dual("maxCumulative_capacity",           "entity", "maxCumulative",
                 derived_from=("maxCumulative_capacity_p",
                                "maxCumulative_capacity_n")),
    _invest_dual("maxInvestGroup_entity_period",     "group",  "maxInvestGroup_period",
                 derived_from=("maxInvestGroup_entity_period_p",
                                "maxInvestGroup_entity_period_n")),
    # maxInvestGroup_entity_total is now one row per group over the
    # chain horizon (period summed out, like maxInvest_entity_total) —
    # has_period=False.  Under recourse it is renamed to _path (not read
    # here, same as the entity-level _path rows).
    _invest_dual("maxInvestGroup_entity_total",      "group",  "maxInvestGroup_total",
                 has_period=False,
                 derived_from=("maxInvestGroup_entity_total_p",
                                "maxInvestGroup_entity_total_n")),
    # ``maxInvestGroup_entity_cumulative`` keeps a single non-suffixed
    # name (_cumulative_invest._eq_group_invest_cumulative).
    _invest_dual("maxInvestGroup_entity_cumulative", "group",  "maxInvestGroup_cumulative"),

    # -- Investment-floor (min-side) duals ----------------------------------
    # Mirror of the maxInvest families above, reading the row duals of the
    # ``>=`` lower-floor constraints emitted by ``_eq_min_*`` /
    # ``_eq_group_*`` (flextool/engine/model/_cumulative_invest.py).
    # Purely additive (Increment 3): makes ``v_dual_min*`` available for the
    # later synthesis increment; nothing consumes these yet.  Same scale
    # sentinel + ``source="row_dual"`` as the max-side; absent families
    # degrade to an empty frame identically.
    #
    # ``minInvest_entity_period`` splits process (``_p``) / node (``_n``)
    # arms (_cumulative_invest._eq_min_entity_period) — same shape as the max-side.
    _invest_dual("minInvest_entity_period",          "entity", "minInvest_period",
                 derived_from=("minInvest_entity_period_p",
                                "minInvest_entity_period_n")),
    # ``minInvest_entity_total_path_{p,n}`` is now one row per entity
    # over the whole chain horizon (period summed out, like the max-side
    # ``maxInvest_entity_total`` — model._eq_invest_total_caps), always named
    # ``_path`` even with a single scenario leaf (_cumulative_invest._eq_min_invest_total).
    # ``has_period=False`` to match the row bracket; the old per-(entity,
    # period) "invested and still alive" shape is gone.  Under recourse this
    # row is per (entity, leaf) — not read for duals here, same as the
    # existing entity ``_path`` max rows (CHANGELOG: the group min-total
    # shadow price under recourse is dropped the same way, see below).
    _invest_dual("minInvest_entity_total_path",      "entity", "minInvest_total",
                 has_period=False,
                 derived_from=("minInvest_entity_total_path_p",
                                "minInvest_entity_total_path_n")),
    _invest_dual("minCumulative_capacity",           "entity", "minCumulative",
                 derived_from=("minCumulative_capacity_p",
                                "minCumulative_capacity_n")),
    _invest_dual("minInvestGroup_entity_period",     "group",  "minInvestGroup_period",
                 derived_from=("minInvestGroup_entity_period_p",
                                "minInvestGroup_entity_period_n")),
    # minInvestGroup_entity_total_path_{p,n} is now one row per group
    # over the chain horizon (period summed out, always named _path -- see
    # minInvest_entity_total_path above); has_period=False.
    _invest_dual("minInvestGroup_entity_total_path", "group",  "minInvestGroup_total",
                 has_period=False,
                 derived_from=("minInvestGroup_entity_total_path_p",
                                "minInvestGroup_entity_total_path_n")),
    # ``minInvestGroup_entity_cumulative`` keeps a single non-suffixed
    # name (_cumulative_invest._eq_group_invest_cumulative), like the max-side cumulative.
    _invest_dual("minInvestGroup_entity_cumulative", "group",  "minInvestGroup_cumulative"),

    # -- CO2 emission-cap duals ---------------------------------------------
    # The model writes ``co2_max_*.dual / scale_the_objective``.  Downstream
    # Python processing applies the extra ``/1000`` (scaled RHS) and
    # ``/inflation`` corrections — not our concern here.
    VariableSpec(
        name="co2_max_period", col_names=("group",),
        has_time=False, source="row_dual",
        value_scale=_INV_SCALE_THE_OBJECTIVE,
        output_name="v_dual_co2_max_period",
    ),
    VariableSpec(
        name="co2_max_total", col_names=("group",),
        has_time=False, has_period=False, source="row_dual",
        value_scale=_INV_SCALE_THE_OBJECTIVE,
        output_name="v_dual_co2_max_total",
    ),

    # Not in VARIABLE_SPECS (handled by dedicated writers below):
    # * v_dual_node_balance — per-period inflation scaling needs a
    #   custom writer (see write_v_dual_node_balance below).
    # * v_dual_reserve__upDown__group__period__t — the sum of the duals of the
    #   group's requirement rows per (r,ud,g,d,t) (see write_v_dual_reserve_balance).
    # * v_dual_invest_{unit,connection,node} — v_invest.dual split by
    #   entity class (see write_v_dual_invest_by_class below).
    # * v_obj — scalar, see write_v_obj below.
]


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------


def _load_realized_set(
    realized_dispatch_csv: Path | str | None,
    *,
    provider: "object | None" = None,
) -> set[tuple[str, str]] | None:
    """Return ``{(period, time), …}`` from the provider's ``realized_dispatch`` frame, or None.

    Used for O(1) membership checks in :func:`extract_variable`.  For
    synthesis of empty-frame row order (which must match the canonical
    iteration order in phase-1 printfs), use
    :func:`_load_realized_list` instead — an ordered variant.

    Item 163: ``realized_dispatch_csv`` is a key selector only (its
    ``<parent>/<stem>`` names the provider key) — no disk read.  A
    ``None`` provider or a missing key gives ``None`` (caller treats
    every (period, time) as realized).
    """
    if realized_dispatch_csv is None:
        return None
    seeded = _provider_lookup(provider, realized_dispatch_csv)
    if seeded is None:
        return None
    period_col = "period"
    time_col = "step" if "step" in seeded.columns else "time"
    return set(zip(
        seeded[period_col].cast(str).to_list(),
        seeded[time_col].cast(str).to_list(),
    ))


def _load_realized_list(
    realized_dispatch_csv: Path | str | None,
    *,
    provider: "object | None" = None,
) -> list[tuple[str, str]] | None:
    """Return ``[(period, time), …]`` from the provider's ``realized_dispatch`` frame, in frame order.

    The frame's row order matches the canonical
    ``for {(d, t) in dt_realize_dispatch}`` iteration order that every
    dt-indexed phase-1 printf used — so synthesizing empty-frame rows
    in this order reproduces the CSV-read parameter's row order exactly.

    Item 163: key selector only, no disk read (see :func:`_load_realized_set`).
    """
    if realized_dispatch_csv is None:
        return None
    seeded = _provider_lookup(provider, realized_dispatch_csv)
    if seeded is None:
        return None
    period_col = "period"
    time_col = "step" if "step" in seeded.columns else "time"
    return list(zip(
        seeded[period_col].cast(str).to_list(),
        seeded[time_col].cast(str).to_list(),
    ))


def _load_realized_periods(
    realized_periods_csv: Path | str | None,
    *,
    provider: "object | None" = None,
) -> set[str] | None:
    """Return ``{period, …}`` from the provider's frame at *realized_periods_csv*'s key, or None.

    Item 163: key selector only, no disk read."""
    if realized_periods_csv is None:
        return None
    seeded = _provider_lookup(provider, realized_periods_csv)
    if seeded is None:
        return None
    return set(seeded["period"].cast(str).to_list())


def _load_realized_invest_periods_list(
    realized_periods_csv: Path | str | None,
    *,
    provider: "object | None" = None,
) -> list[str] | None:
    """Return the roll's realized-INVEST ``[period, …]`` in source order.

    Canonical row source for the investment-axis variables
    (``v_invest`` / ``v_divest``).  Sourced from the in-memory provider frame
    (``solve_data/realized_invest_periods_of_current_solve``) — item 163: no disk
    fallback.  Returns the (possibly EMPTY) ordered period list when the
    source is present — an empty list means this roll realizes no
    investment, so the variable extractor must emit no rows for it.
    Returns ``None`` only when no source is available at all (a ``None``
    provider or a missing key), signalling the caller to fall back to the
    dispatch-axis canonical order (e.g. single-solve fixtures that never
    emit the invest-period set).
    """
    if realized_periods_csv is None:
        return None
    seeded = _provider_lookup(provider, realized_periods_csv)
    if seeded is None:
        return None
    return list(seeded["period"].cast(str).to_list())


def _name_regex(var_name: str) -> re.Pattern[str]:
    """Return a compiled regex matching ``<var_name>[...]``."""
    return re.compile(rf"^{re.escape(var_name)}\[(.+)\]$")


def _load_canonical_dt_order(
    solve_name: str,
    *,
    flex_data: "FlexData | None" = None,
) -> list[tuple[str, str]] | None:
    """Return ``[(period, time), …]`` in the canonical iteration order, or ``None``.

    Item 163: the work folder is never read here any more.  The source is the
    in-memory ``flex_data.realized_dispatch`` (already a polars frame of
    ``(period, step)``) — the cascade's authoritative per-roll dispatch order,
    which every dt-indexed output follows.  ``None`` (no ``flex_data``, or
    ``flex_data.realized_dispatch`` not populated this roll) means the caller
    falls back to its own ``realized_dispatch_csv``-keyed provider lookup
    (:func:`_load_realized_list`) or, failing that, the defensive first-
    appearance order in :func:`extract_variable`.
    """
    if flex_data is not None and getattr(flex_data, "realized_dispatch", None) is not None:
        try:
            rd = flex_data.realized_dispatch
            cols = rd.columns
            time_col = "step" if "step" in cols else ("time" if "time" in cols else cols[1])
            return list(zip(
                rd["period"].cast(str).to_list(),
                rd[time_col].cast(str).to_list(),
            ))
        except Exception:  # noqa: BLE001
            pass
    return None


def _fan_labels(flex_data) -> set[str]:
    """The period labels of the solve's stochastic fans (``fan_period_labels`` of ``period_branch_full``)."""
    pbf = getattr(flex_data, "period_branch_full", None) if flex_data is not None else None
    if pbf is None or pbf.height == 0:
        return set()
    from flextool.engine.preprocess.params._branch import fan_period_labels
    return fan_period_labels({(str(a), str(b)) for a, b in zip(pbf["d"].to_list(), pbf["b"].to_list())})


def _invest_branch_periods(flex_data, realized_invest: list[str]) -> list[str]:
    """The forecast branch periods ``b ≠ d`` of ``period_branch_full`` whose period ``d`` is in
    ``realized_invest``, in the frame's order; empty for a deterministic solve."""
    pbf = getattr(flex_data, "period_branch_full", None) if flex_data is not None else None
    if pbf is None or pbf.height == 0:
        return []
    keep = set(realized_invest)
    out: list[str] = []
    for a, b in zip(pbf["d"].cast(str).to_list(), pbf["b"].cast(str).to_list()):
        if a in keep and b != a and b not in out:
            out.append(b)
    return out


def _load_canonical_d_order(
    solve_name: str,
    *,
    flex_data: "FlexData | None" = None,
) -> list[str] | None:
    """Return ``[period, …]`` in the canonical iteration order, or ``None``.

    Item 163: no disk fallback.  Derives the ordered distinct period list
    from the in-memory ``flex_data.realized_dispatch`` when populated this
    roll; ``None`` otherwise (see :func:`_load_canonical_dt_order`).
    """
    if flex_data is not None and getattr(flex_data, "realized_dispatch", None) is not None:
        try:
            rd = flex_data.realized_dispatch
            seen: list[str] = []
            seen_set: set[str] = set()
            for p in rd["period"].cast(str).to_list():
                if p not in seen_set:
                    seen.append(p)
                    seen_set.add(p)
            return seen
        except Exception:  # noqa: BLE001
            pass
    return None


def _row_index_names(*, has_period: bool, has_time: bool) -> list[str]:
    if not has_period:
        return ["solve"]
    if has_time:
        return ["solve", "period", "time"]
    return ["solve", "period"]


def _empty_columns(col_names: Sequence[str]) -> pd.Index:
    if len(col_names) >= 2:
        return pd.MultiIndex.from_tuples([], names=list(col_names))
    return pd.Index([], name=col_names[0])


def empty_variable_frame(
    solve_name: str,
    col_names: Sequence[str],
    *,
    has_period: bool = True,
    has_time: bool = True,
    realized_dt: "list[tuple[str, str]] | set[tuple[str, str]] | None" = None,
    realized_p: "list[str] | set[str] | None" = None,
) -> pd.DataFrame:
    """Same-shape empty frame: full ``(solve, period[, time])`` row index, zero columns.

    Built so downstream pandas ops (``DataFrame.mul(axis=1, level=0)``)
    don't see a ``(0, 0)`` operand on one side and a populated row index
    on the other.  Construction is a single ``DataFrame`` call — no
    Python loops over rows.

    ``realized_dt`` / ``realized_p`` should be **ordered** (from
    :func:`_load_realized_list` / :func:`_load_realized_periods_list`)
    so the synthesised row order matches the canonical
    ``for {(d, t) in dt_realize_dispatch}`` iteration order.  A ``set``
    is accepted (for back-compat) but yields arbitrary iteration order
    — only safe when the caller doesn't need cross-reader row alignment.
    """
    row_index_names = _row_index_names(has_period=has_period, has_time=has_time)
    empty_cols = _empty_columns(col_names)

    if has_period and has_time and realized_dt is not None:
        rows = pd.MultiIndex.from_tuples(
            [(solve_name, d, t) for (d, t) in realized_dt],
            names=row_index_names,
        )
    elif has_period and not has_time and realized_p is not None:
        rows = pd.MultiIndex.from_tuples(
            [(solve_name, d) for d in realized_p],
            names=row_index_names,
        )
    elif not has_period:
        rows = pd.Index([solve_name], name=row_index_names[0])
    else:
        if len(row_index_names) == 1:
            rows = pd.Index([], name=row_index_names[0])
        else:
            rows = pd.MultiIndex.from_tuples([], names=row_index_names)

    return pd.DataFrame(index=rows, columns=empty_cols, dtype=float)


def handle_col_dual(h: "object", solution: "object | None" = None) -> "object":
    """The reduced costs of the solve on ``h``, in the model's own units.

    After an autoscale Layer-2 unscale, HiGHS's ``getSolution().col_dual``
    is in scaled coordinates (``setSolution`` recomputes it from the
    handle's scaled cost and matrix), so the unscale keeps the true values
    on the handle as ``_flextool_unscaled_col_dual`` (item 239).  They are
    used while the handle has not been solved again since (model status
    ``kNotset``, which ``setSolution`` leaves and a ``run`` replaces).
    ``solution`` is an already fetched ``h.getSolution()``.
    """
    sol = solution if solution is not None else h.getSolution()
    stashed = getattr(h, "_flextool_unscaled_col_dual", None)
    if stashed is not None:
        try:
            import highspy

            if h.getModelStatus() == highspy.HighsModelStatus.kNotset and len(stashed) == len(sol.col_dual):
                return stashed
        except Exception:  # pragma: no cover — duck-typed handle without a model status
            pass
    return sol.col_dual


def extract_variable(
    h: "highspy.Highs",
    name: str,
    col_names: Sequence[str],
    *,
    solve_name: str,
    has_time: bool = True,
    has_period: bool = True,
    source: str = "col_value",
    value_scale: float = 1.0,
    realized_dispatch_csv: Path | str | None = None,
    realized_periods_csv: Path | str | None = None,
    trailing_col_names: Sequence[str] = (),
    flex_data: "FlexData | None" = None,
    provider: "object | None" = None,
    col_names_cache: Sequence[str] | None = None,
    row_names_cache: Sequence[str] | None = None,
    col_value: "object | None" = None,
    col_dual: "object | None" = None,
    row_dual: "object | None" = None,
    period_source: str = "dispatch",
    coarse_steps: "object | None" = None,
) -> pd.DataFrame:
    """Extract one quantity from a solved HiGHS instance as a wide DataFrame.

    Parameters mirror :class:`VariableSpec` plus ``solve_name`` (tag
    inserted into the row MultiIndex — typically the current solve name).
    ``source`` selects which aligned array HiGHS exposes —
    ``"col_value"``, ``"col_dual"`` (both keyed by
    ``allVariableNames()``) or ``"row_dual"`` (keyed by
    ``getLp().row_names_``).  ``value_scale`` is applied once per raw
    value (typically ``1e6`` for dual sources, to undo
    ``scale_the_objective``).

    ``realized_dispatch_csv`` filters (period, time) pairs for time-
    indexed quantities; ``realized_periods_csv`` filters periods for
    period-only quantities.  ``has_period=False`` means the quantity
    has no period index at all — row index collapses to just
    ``(solve,)`` (used for ``co2_max_total[g]``).

    The variables of coarse entities (``_coarse_flows.COARSE_VARS``: flows,
    units online, start-ups, shut-downs, reserves of coarse processes, the
    angles of coarse-angle nodes, the inertia / non-synchronous slacks of
    averaged group rows) have one LP column per coarse step; a level is
    copied to every fine step of the step, an event stays at its first fine
    step (:mod:`flextool.engine.output._coarse_flows`).  The coarse steps come
    from ``coarse_steps`` (``(key, d, b_first, t)``) or else from
    ``flex_data`` (the variable's own steps).

    Returns
    -------
    DataFrame
        Wide layout — row index ``(solve,)`` / ``(solve, period)`` /
        ``(solve, period, time)`` depending on the has_* flags.  Column
        MultiIndex when ``len(col_names) >= 2``, a single-level ``Index``
        otherwise.  Missing combinations are filled with 0.0.
    """
    # Cached arrays (hoisted out of the per-spec loop by
    # ``write_all_variables``) are used when provided; otherwise fall
    # back to fetching from the live HiGHS instance so the standalone /
    # single-spec code path keeps working unchanged.
    if source == "row_dual":
        # Constraint names are stored on the LP struct, not exposed via
        # a bulk getter on Highs itself — ``getLp().row_names_`` is the
        # fast path (no per-row Python call).
        names = row_names_cache if row_names_cache is not None else h.getLp().row_names_
        values = row_dual if row_dual is not None else h.getSolution().row_dual
    elif source == "col_dual":
        names = col_names_cache if col_names_cache is not None else h.allVariableNames()
        values = col_dual if col_dual is not None else handle_col_dual(h)
    elif source == "col_value":
        names = col_names_cache if col_names_cache is not None else h.allVariableNames()
        values = col_value if col_value is not None else h.getSolution().col_value
    else:
        raise ValueError(
            f"Unknown source '{source}' — expected one of "
            "'col_value', 'col_dual', 'row_dual'"
        )
    if len(names) != len(values):
        raise RuntimeError(
            f"HiGHS name / value length mismatch for '{name}' "
            f"(source={source}): {len(names)} names vs {len(values)} values"
        )

    prefix = f"{name}["
    pattern = _name_regex(name)
    trailing = (2 if has_time else 1) if has_period else 0
    n_trailing_cols = len(trailing_col_names)
    expected_arity = len(col_names) + trailing + n_trailing_cols
    row_index_names = _row_index_names(has_period=has_period, has_time=has_time)
    # Full column-name tuple — leading cols (before period) + trailing
    # cols (after period/time).  Used for the column (Multi)Index and
    # for the empty-frame shape.
    full_col_names: tuple[str, ...] = tuple(col_names) + tuple(trailing_col_names)

    # Canonical row order: read directly from a phase-1 printf CSV that
    # iterates the same set the per-solve parameter CSVs do.  Building
    # the wide frame against this order from the get-go means no
    # post-hoc sort/reindex — both readers produce the exact same row
    # sequence because both ultimately come from the same phase-1
    # ``for {s, (d, t) in dt_realize_dispatch}`` iteration.
    if has_period and has_time:
        canonical_rows: list[tuple[str, ...]] | None = (
            _load_canonical_dt_order(solve_name, flex_data=flex_data)
        )
        # Fallback for callers whose ``flex_data.realized_dispatch`` is
        # absent this roll (e.g. a custom-writer call with no ``flex_data``,
        # or a unit test) — the provider-keyed frame at ``realized_dispatch_csv``.
        if canonical_rows is None and realized_dispatch_csv is not None:
            canonical_rows = _load_realized_list(
                realized_dispatch_csv, provider=provider,
            )
    elif has_period:
        canonical_d = _load_canonical_d_order(
            solve_name, flex_data=flex_data,
        )
        if period_source in ("invest", "invest_branches"):
            # Investment-axis variables (``v_invest`` / ``v_divest``): a
            # roll that realizes DISPATCH but NO investment (the dispatch
            # children of a nested invest cascade) must emit NO rows.
            # Otherwise the densified column union (8e2de938 / a9058c66)
            # materialises a zero-valued ``(roll, dispatch_period)`` row
            # for every invest-eligible entity, and the ``drop_levels``
            # keep='last' dedup lets that trailing dispatch-roll zero
            # overwrite the invest step's real value for the shared period
            # (zeroing the ``invested`` breakdown in ``unit_capacity__d``).
            #
            # The discriminator is the roll's realized-INVEST set: when it
            # is empty this is a dispatch-only roll → emit nothing; when it
            # is non-empty (the invest-committing step, or a rolling-invest
            # solve) keep the FULL realized-dispatch period axis so periods
            # whose investment is legitimately 0 (e.g. y2020 p2025, no NEW
            # decision) still produce their zero row.  When no realized-
            # invest source is available at all (single-solve fixtures that
            # never emit the set), fall back to the dispatch axis unchanged.
            realized_invest = _load_realized_invest_periods_list(
                realized_periods_csv, provider=provider,
            )
            no_invest = realized_invest is not None and not realized_invest
            # A fan label holds no capacity-margin slack (its forecasts do): no label rows for invest_branches.
            labels = _fan_labels(flex_data) if realized_invest and period_source == "invest" else set()
            if no_invest:
                canonical_d = []
            elif labels or (realized_invest and canonical_d is not None):
                # Every realized-invest period gets its row, also one that is not on the solve's dispatch axis.
                # A stochastic fan solve: a period label of the fan has no dispatch but is where the solve
                # realizes its here-and-now investment (stochastic_invest_method = none; no column under
                # recourse, so a 0 row); with a realized branch the realization dispatch solve holds the
                # dispatch of every period, so the stochastic solve has no dispatch axis and its pre-reveal
                # (real-named) investment period had no row (item 258).  Any other solve: its realized-invest
                # periods need not all be realized-dispatch periods (realized_invest_periods not within
                # realized_periods); those had no row either.  A non-fan solve with NO dispatch axis keeps the
                # solution-order fallback (None) -- the invest-only steps of a cascade rely on it.
                base = list(canonical_d or [])
                canonical_d = base + [p for p in realized_invest if p not in base]
            if period_source == "invest_branches" and realized_invest:
                # The forecast branch periods b ≠ d of the realized-invest periods d (period_branch_full): a
                # fan's planning slack has one value per forecast, and no dispatch axis holds them.  The output
                # tables keep them where the run realizes them (drop_levels._drop_unrealized_forecast_periods).
                branches = _invest_branch_periods(flex_data, realized_invest)
                if branches:
                    base = list(canonical_d or [])
                    canonical_d = base + [p for p in branches if p not in base]
        canonical_rows = [(d,) for d in canonical_d] if canonical_d is not None else None
    else:
        canonical_rows = [()]  # one row: just (solve,)

    # Single pass over HiGHS: dict ``(d[, t], *col_vals) → value`` plus
    # first-appearance unique col_vals tracking.
    values_by_key: dict[tuple[str, ...], float] = {}
    seen_cols_set: set[tuple[str, ...]] = set()
    seen_cols: list[tuple[str, ...]] = []

    for item_name, val in zip(names, values):
        if not item_name.startswith(prefix):
            continue
        m = pattern.match(item_name)
        if not m:
            _logger.warning("Unrecognised %s name: %s", name, item_name)
            continue
        # GLPSOL/MPS quoting: any symbolic name containing a colon (e.g.
        # ISO 8601 timestamps like ``2050-01-01T00:00:00``) is wrapped in
        # single quotes when written to the .mps and HiGHS preserves
        # those quotes verbatim in ``allVariableNames()``.  The canonical
        # row order (read from ``solve_data/p_step_duration.csv``) has
        # bare timestamps, so without stripping here every time-indexed
        # row_key would silently miss the canonical lookup at line ~773
        # and the resulting parquet would be filled with zeros.
        parts = [p.strip("'") for p in m.group(1).split(",")]
        if len(parts) != expected_arity:
            _logger.warning(
                "Unexpected %s arity (%d, expected %d): %s",
                name, len(parts), expected_arity, item_name,
            )
            continue
        col_end = len(col_names)
        leading_col_vals = tuple(parts[:col_end])
        if has_period and has_time:
            row_key: tuple[str, ...] = (parts[col_end], parts[col_end + 1])
            row_len = 2
        elif has_period:
            row_key = (parts[col_end],)
            row_len = 1
        else:
            row_key = ()
            row_len = 0
        trailing_start = col_end + row_len
        trailing_col_vals = tuple(
            parts[trailing_start:trailing_start + n_trailing_cols]
        )
        col_vals = leading_col_vals + trailing_col_vals
        if col_vals not in seen_cols_set:
            seen_cols.append(col_vals)
            seen_cols_set.add(col_vals)
        values_by_key[row_key + col_vals] = float(val) * value_scale

    # Imported here: flextool.engine pulls in polar_high and its solver backends, which `import flextool`
    # must not load.
    from flextool.engine.output import _coarse_flows

    if name in _coarse_flows.COARSE_VARS and has_period and has_time:
        steps = coarse_steps if coarse_steps is not None else _coarse_flows.coarse_steps(flex_data, name)
        _coarse_flows.expand_value_keys(values_by_key, steps, mode=_coarse_flows.COARSE_VARS[name])

    # No variable values matched — produce the same N_rows × 0-cols
    # frame phase-3 CSV writers would have produced (with the canonical
    # row index), so downstream ``DataFrame.mul(axis=1, level=0)``
    # against a populated parameter frame doesn't get an empty operand.
    if not seen_cols:
        return empty_variable_frame(
            solve_name, full_col_names,
            has_period=has_period, has_time=has_time,
            realized_dt=canonical_rows if (has_period and has_time) else None,
            realized_p=(
                [r[0] for r in canonical_rows]
                if (has_period and not has_time and canonical_rows is not None)
                else None
            ),
        )

    # Build the wide matrix by canonical row × first-appearance col
    # position lookup.  Single dict iteration; the lookup itself is
    # O(1) per entry.
    n_cols_total = len(full_col_names)
    if canonical_rows is None:
        # Defensive: no canonical source available — fall back to
        # first-appearance row order from the HiGHS scan.
        seen_rows_set: set[tuple[str, ...]] = set()
        canonical_rows = []
        for k in values_by_key:
            row_key = k[: -n_cols_total] if n_cols_total else k
            if row_key not in seen_rows_set:
                canonical_rows.append(row_key)
                seen_rows_set.add(row_key)

    import numpy as np
    row_pos = {r: i for i, r in enumerate(canonical_rows)}
    col_pos = {c: j for j, c in enumerate(seen_cols)}
    matrix = np.zeros((len(canonical_rows), len(seen_cols)), dtype=float)
    for key, val in values_by_key.items():
        if n_cols_total:
            row_key = key[: -n_cols_total]
            col_key = key[-n_cols_total:]
        else:
            row_key = key
            col_key = ()
        i = row_pos.get(row_key)
        if i is None:
            continue  # row not in canonical (e.g. storage-reference timestep)
        matrix[i, col_pos[col_key]] = val
    # Normalise IEEE negative zeros to positive zero — HiGHS occasionally
    # returns ``-0.0`` for variables pinned at the lower bound, and pandas
    # ``assert_frame_equal`` distinguishes ``-0.0`` from ``0.0``.
    matrix += 0.0

    if n_cols_total >= 2:
        col_idx: pd.Index = pd.MultiIndex.from_tuples(
            seen_cols, names=list(full_col_names),
        )
    else:
        col_idx = pd.Index(
            [c[0] for c in seen_cols], name=full_col_names[0],
        )

    if not has_period:
        row_idx: pd.Index = pd.Index([solve_name], name=row_index_names[0])
    else:
        row_idx = pd.MultiIndex.from_tuples(
            [(solve_name, *r) for r in canonical_rows], names=row_index_names,
        )
    return pd.DataFrame(matrix, index=row_idx, columns=col_idx)


def write_variable_parquet(
    h: "highspy.Highs",
    spec: VariableSpec,
    *,
    solve_name: str,
    output_dir: Path | str,
    realized_dispatch_csv: Path | str | None = None,
    realized_periods_csv: Path | str | None = None,
    file_name: str | None = None,
    flex_data: "FlexData | None" = None,
    scale_the_objective: float | None = None,
    provider: "object | None" = None,
    col_names_cache: Sequence[str] | None = None,
    row_names_cache: Sequence[str] | None = None,
    col_value: "object | None" = None,
    col_dual: "object | None" = None,
    row_dual: "object | None" = None,
) -> Path:
    """Extract the quantity described by *spec* and write a per-solve parquet.

    File name defaults to ``{spec.output_name or spec.name}__{solve}.parquet``
    so parallel / rolling / nested solves don't collide.  Merging
    per-solve files into a single ``{output_name}.parquet`` is a cheap
    post-processing step (one ``pd.concat``).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    # Agent 12: when the spec uses ``_INV_SCALE_THE_OBJECTIVE`` as a
    # sentinel default (value_scale == 1e6, matching the legacy 1e-6
    # hardcoded objective scalar), substitute the live reciprocal of the
    # current solve's ``scale_the_objective``.  Specs with an explicit
    # non-sentinel ``value_scale`` (e.g. ``-1e6`` for node balance duals)
    # still honour that — see :func:`write_v_dual_node_balance`.
    effective_scale = spec.value_scale
    if spec.value_scale == _INV_SCALE_THE_OBJECTIVE:
        effective_scale = _resolve_inv_scale_the_objective(
            scale_the_objective=scale_the_objective,
        )
    # Multi-source fan-out: when the output quantity is the sum of two or
    # more HiGHS variables (e.g. two-tier slack), extract each source and
    # add them.  Per-source frames share the same index shape, so
    # ``DataFrame.add(fill_value=0.0)`` is safe and correct: columns that
    # appear only in one source contribute that source's value plus 0.
    if spec.derived_from:
        df: pd.DataFrame | None = None
        for src_name in spec.derived_from:
            src_df = extract_variable(
                h, src_name, spec.col_names,
                solve_name=solve_name,
                has_time=spec.has_time,
                has_period=spec.has_period,
                source=spec.source,
                value_scale=effective_scale,
                realized_dispatch_csv=realized_dispatch_csv,
                realized_periods_csv=realized_periods_csv,
                trailing_col_names=spec.trailing_col_names,
                flex_data=flex_data,
                provider=provider,
                col_names_cache=col_names_cache,
                row_names_cache=row_names_cache,
                col_value=col_value,
                col_dual=col_dual,
                row_dual=row_dual,
                period_source=spec.period_source,
            )
            df = src_df if df is None else df.add(src_df, fill_value=0.0)
        assert df is not None  # guaranteed: derived_from is non-empty
    else:
        df = extract_variable(
            h, spec.name, spec.col_names,
            solve_name=solve_name,
            has_time=spec.has_time,
            has_period=spec.has_period,
            source=spec.source,
            value_scale=effective_scale,
            realized_dispatch_csv=realized_dispatch_csv,
            realized_periods_csv=realized_periods_csv,
            trailing_col_names=spec.trailing_col_names,
            flex_data=flex_data,
            provider=provider,
            col_names_cache=col_names_cache,
            row_names_cache=row_names_cache,
            col_value=col_value,
            col_dual=col_dual,
            row_dual=row_dual,
            period_source=spec.period_source,
        )
    # Agent 9 — row-scaling un-scaling applied at the output boundary.
    # Item 163: the scaler comes from ``flex_data`` only (no work-folder CSV).
    if spec.unscale_by is not None:
        df = _apply_unscale(df, spec.unscale_by, solve_name, flex_data=flex_data)
    if file_name is None:
        file_name = f"{spec.output_name or spec.name}__{solve_name}.parquet"
    path = output_dir / file_name
    write_lean_parquet(df, path)
    _logger.debug(
        "Wrote %s for solve '%s' -> %s (shape %s)",
        spec.output_name or spec.name, solve_name, path, df.shape,
    )
    return path


# ---------------------------------------------------------------------------
# Custom writers for outputs that don't fit the plain VariableSpec pattern
# (scalar, entity-class split, per-period transform).
# ---------------------------------------------------------------------------


def _entity_class_members(provider: "object | None", set_name: str) -> set[str]:
    """Members of entity class *set_name* (``"process_unit"``, ``"process_connection"``, ``"node"``),
    from the sub-solve's :class:`FlexDataProvider` snapshot under key ``f"input/{set_name}"``.

    Item 163: a normal run never writes ``input/{set_name}.csv``, so reading it from disk left
    every fresh run's entity-class split empty.  The provider carries the same class sets GMPL used
    to write.  A None provider or a missing key gives an empty set — never reads disk.
    """
    if provider is None:
        return set()
    frame = provider.get(f"input/{set_name}")
    if frame is None or frame.is_empty() or frame.width == 0:
        return set()
    return set(frame.to_series(0).cast(pl.Utf8).to_list())


# ---------------------------------------------------------------------------
# Row-scaler (Agent 9)
# ---------------------------------------------------------------------------


def _load_row_scaler(
    kind: str,
    solve_name: str,
    *,
    flex_data: "FlexData | None" = None,
) -> pd.DataFrame | None:
    """The sub-solve's ``p_{node,group}_capacity_for_scaling`` Param, from ``flex_data`` only.

    Phase G — prefers ``flex_data.p_{node,group}_capacity_for_scaling``
    Returns ``None`` when ``flex_data`` is absent or carries no such Param —
    callers then treat the scaler as 1 everywhere (no-op); this is every
    fresh run today (Mode A — the DB-level row scaling was removed).

    ``kind`` is ``"node"`` or ``"group"``.
    """
    if flex_data is None:
        return None
    attr = f"p_{kind}_capacity_for_scaling"
    param = getattr(flex_data, attr, None)
    if param is None:
        return None
    try:
        # Param frame: columns (entity, "d", "value").  Pivot to
        # wide ``(solve, period)``-row × entity-column shape used
        # downstream.  ``solve`` is the active solve_name.
        f = param.frame
        cols = f.columns
        # Drop value column from the pivot index set.
        if "value" not in cols:
            return None
        entity_col = cols[0]
        period_col = cols[1]
        wide = f.pivot(
            on=entity_col, index=period_col, values="value",
        )
        pdf = wide.to_pandas()
        pdf = pdf.set_index(period_col)
        pdf.index = pd.MultiIndex.from_tuples(
            [(str(solve_name), str(p)) for p in pdf.index],
            names=["solve", "period"],
        )
        pdf = pdf.apply(pd.to_numeric, errors="coerce")
        return pdf
    except Exception:  # noqa: BLE001
        return None


def _apply_unscale(
    df: pd.DataFrame,
    unscale_by: str,
    solve_name: str,
    *,
    flex_data: "FlexData | None" = None,
) -> pd.DataFrame:
    """Multiply *df* by the row scaler identified by *unscale_by*.

    Handles two scaler kinds, both sourced from ``flex_data`` (item 163 —
    no work-folder CSV):

    * ``"node_cap"`` — ``flex_data.p_node_capacity_for_scaling``, keyed
      by (period, node).  ``df`` row index is ``(solve, period[, time])``
      and columns are node names; we broadcast the period row of the
      scaler across the time dimension.
    * ``"group_cap"`` — ``flex_data.p_group_capacity_for_scaling``, keyed
      by (period, group).  Columns = group names.  Same
      period-broadcast for time-indexed frames; for the no-t case (only
      ``vq_capacity_margin`` today) the row index is just ``(solve,
      period)`` and the element-wise multiply aligns directly.

    A missing Param / unknown columns / Mode A (scaler = 1) all collapse to
    a safe no-op: rows without a matching scaler are unchanged.
    """
    if df.empty or df.shape[1] == 0:
        return df
    kind = {"node_cap": "node", "group_cap": "group"}.get(unscale_by)
    if kind is None:
        return df
    scaler = _load_row_scaler(kind, solve_name, flex_data=flex_data)
    if scaler is None or scaler.empty:
        return df

    # Re-key the scaler by period alone (drop the solve level — we already
    # filtered on solve_name).  Columns = entity.
    scaler_by_period = scaler.droplevel("solve") if "solve" in (scaler.index.names or []) else scaler

    # Build an aligned multiplier with the same shape as ``df``.
    # 1) Keep only columns of ``scaler_by_period`` that appear in ``df``.
    #    Data column name is the entity — for MultiIndex column frames
    #    the first level holds the entity name used in the CSV header.
    entity_level = 0  # col_names[0] by construction for unscaled slacks
    if isinstance(df.columns, pd.MultiIndex):
        entity_names = df.columns.get_level_values(entity_level).astype(str).tolist()
    else:
        entity_names = df.columns.astype(str).tolist()
    present = [e for e in entity_names if e in scaler_by_period.columns]
    if not present:
        return df

    # 2) For each row of ``df``, fetch the scaler row for that period.
    #    Rows whose period has no scaler row get multiplier = 1
    #    (no-op).  Time-indexed df: broadcast the same period row across
    #    all timesteps of that period.
    if isinstance(df.index, pd.MultiIndex) and "period" in (df.index.names or []):
        periods = df.index.get_level_values("period").astype(str)
    else:
        # Shouldn't happen for un-scaled slacks (all have period at least),
        # but guard for the no-period edge.
        return df

    # Construct the multiplier frame: same rows as df, columns = entity_names
    # order.  Use reindex on scaler_by_period to align columns → NaN for
    # missing entities → fill with 1 so they remain unchanged.
    mult = scaler_by_period.reindex(columns=entity_names).astype(float)
    mult = mult.reindex(periods.astype(str)).fillna(1.0)
    mult.index = df.index
    # Preserve the df column index (could be MultiIndex); numpy-level
    # multiply keeps the dtype and index.
    mult.columns = df.columns

    return df * mult


def write_v_obj(
    h: "highspy.Highs",
    *,
    solve_name: str,
    output_dir: Path | str,
    flex_data: "FlexData | None" = None,
    scale_the_objective: float | None = None,
    obj_offset: float = 0.0,
) -> Path:
    """Write ``v_obj__{solve}.parquet`` — objective value for this solve.

    Model writes ``total_cost.val / scale_the_objective``.  HiGHS's
    ``getObjectiveValue()`` returns the raw (scaled) value; we undo the
    scaling.  ``scale_the_objective`` is per-solve, cascade-threaded
    (item 163: no work-folder CSV fallback); ``None``/non-positive falls
    back to the legacy ``1e-6`` scalar.  ``obj_offset`` is the objective
    constant (``Problem.add_obj_constant``, scaled) that the handle's
    objective leaves out (``Solution.obj_offset``: polar-high >= 3.14.0
    keeps it out of HiGHS, and an MPS file never carries it); it is added
    back.  The autoscale-stashed objective already includes it.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    inv_scale = _resolve_inv_scale_the_objective(
        scale_the_objective=scale_the_objective,
    )
    # Prefer the autoscale-stashed objective when present: Layer 2's
    # ``_push_unscaled_to_highs`` calls ``h.setSolution`` to mirror the
    # unscaled primal back onto the live solver handle, but that call
    # zeroes HiGHS's cached ``getObjectiveValue()`` (verified against
    # highspy 1.14.0).  ``_flextool_unscaled_objective`` is the obj
    # captured immediately before that ``setSolution`` and is the
    # post-Layer-2-substitution / post-user_bound_scale-unscale value.
    raw_obj = getattr(h, "_flextool_unscaled_objective", None)
    if raw_obj is None:
        raw_obj = float(h.getObjectiveValue()) + float(obj_offset)
    obj = float(raw_obj) * inv_scale
    df = pd.DataFrame(
        {"objective": [obj]},
        index=pd.Index([solve_name], name="solve"),
    )
    path = output_dir / f"v_obj__{solve_name}.parquet"
    write_lean_parquet(df, path)
    # Emit the canonical ``total_cost.val`` stdout line so that callers
    # parsing the FlexTool stdout (e.g. test_representative_periods)
    # still see the objective value.
    print(f"total_cost.val = {obj:.12g}")
    _logger.debug("Wrote v_obj for solve '%s' -> %s (%.10g)", solve_name, path, obj)
    return path


def write_v_dual_invest_by_class(
    h: "highspy.Highs",
    *,
    solve_name: str,
    output_dir: Path | str,
    provider: "object | None" = None,
    realized_periods_csv: Path | str | None = None,
    flex_data: "FlexData | None" = None,
    scale_the_objective: float | None = None,
) -> list[Path]:
    """Write v_invest reduced costs split by entity class.

    Produces three parquet files — ``v_dual_invest_unit__{solve}``,
    ``v_dual_invest_connection__{solve}``, ``v_dual_invest_node__{solve}``
    — each containing the ``v_invest.dual`` values for entities in the
    corresponding class (``process_unit``, ``process_connection``,
    ``node``).  Matches the three separate CSVs phase 3 writes.

    HiGHS returns the column reduced cost in *scaled-objective* units
    (the objective is multiplied by ``scale_the_objective`` at build
    time, default 1e-6).  Multiply by ``1 / scale_the_objective`` so the
    written reduced cost is in true NPV currency per v_invest unit —
    consistent with every row-dual writer (cf.
    :func:`write_v_dual_node_balance`).

    ``provider`` is required: it supplies the entity-class membership
    (item 163 — a normal run never writes ``input/{process_unit,
    process_connection,node}.csv``, so that split is fixed from the
    sub-solve's :class:`FlexDataProvider` snapshot instead of disk).
    """
    if provider is None:
        raise ValueError(
            "write_v_dual_invest_by_class needs the sub-solve's FlexDataProvider (entity classes)"
        )
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ``_resolve_inv_scale_the_objective`` returns ``1 / scale_the_objective``
    # (guarding None / non-positive → default 1e-6), exactly as the
    # node-balance writer uses it.
    inv_scale = _resolve_inv_scale_the_objective(
        scale_the_objective=scale_the_objective,
    )
    duals = extract_variable(
        h, "v_invest", ("entity",),
        solve_name=solve_name, has_time=False, source="col_dual",
        value_scale=inv_scale,
        realized_periods_csv=realized_periods_csv,
        flex_data=flex_data,
    )

    classes = {
        "process_unit": _entity_class_members(provider, "process_unit"),
        "process_connection": _entity_class_members(provider, "process_connection"),
        "node": _entity_class_members(provider, "node"),
    }
    output_suffixes = {
        "process_unit": "unit",
        "process_connection": "connection",
        "node": "node",
    }

    paths: list[Path] = []
    for cls_name, members in classes.items():
        keep = [c for c in duals.columns if c in members]
        subset = duals[keep].copy() if keep else duals.iloc[:, :0]
        # Preserve the single-level column index name for round-trip
        subset.columns.name = "entity"
        fname = f"v_dual_invest_{output_suffixes[cls_name]}__{solve_name}.parquet"
        path = output_dir / fname
        write_lean_parquet(subset, path)
        _logger.debug(
            "Wrote v_dual_invest_%s for solve '%s' -> %s (shape %s)",
            output_suffixes[cls_name], solve_name, path, subset.shape,
        )
        paths.append(path)
    return paths


def _price_weight(flex_data: "FlexData | None", *, per_hour: bool = False) -> pl.DataFrame:
    """``(d, t, w)``, the objective weight of one unit of a per-(d,t) balance quantity
    (:func:`flextool.engine.output._prices.price_weight_dt`).

    Item 163: no work-folder CSV fallback and no caller-supplied ``rows`` domain any more — every
    production caller has ``flex_data``; a missing component on ``flex_data`` (including ``flex_data``
    itself being ``None``) counts as weight 1 (:func:`price_weight_dt` / :func:`getattr` default), not a
    disk read.
    """
    from flextool.engine.output._prices import price_weight_dt

    return price_weight_dt(flex_data, per_hour=per_hour)


def _wide_to_price(df: "pd.DataFrame", weight: pl.DataFrame, *, sign: float, inv_scale: float) -> "pd.DataFrame":
    """Apply :func:`flextool.engine.output._prices.to_price` to a wide ``(solve, period, time)`` × columns frame
    of raw duals; the row and column order of ``df`` is kept (a null price becomes NaN)."""
    from flextool.engine.output._prices import to_price

    if df.empty:
        return df
    flat = df.copy()
    flat.columns = range(df.shape[1])
    long = flat.stack(future_stack=True).rename("dual").reset_index()
    long.columns = ["solve", "d", "t", "__col", "dual"]
    long[["solve", "d", "t"]] = long[["solve", "d", "t"]].astype(str)
    priced = to_price(pl.from_pandas(long), weight, sign=sign, inv_scale=inv_scale).sort("__col", "solve", "d", "t")
    values = priced.to_pandas().pivot(index=["solve", "d", "t"], columns="__col", values="price")
    index = pd.MultiIndex.from_tuples([tuple(str(x) for x in i) for i in df.index], names=df.index.names)
    out = values.reindex(index=index, columns=range(df.shape[1]))
    out.index = df.index
    out.columns = df.columns
    return out.astype(float)


def _divide_by_row_scaler(
    df: "pd.DataFrame",
    *,
    solve_name: str,
    flex_data: "FlexData | None",
) -> "pd.DataFrame":
    """Divide each ``(period, node)`` cell by ``node_capacity_for_scaling`` (1 for every solve today: the DB-level
    row scaling was removed, so this is a no-op kept for the parameter's contract)."""
    if df.empty:
        return df
    scaler = _load_row_scaler("node", solve_name, flex_data=flex_data)
    if scaler is not None and not scaler.empty:
        scaler_by_period = (
            scaler.droplevel("solve")
            if "solve" in (scaler.index.names or [])
            else scaler
        )
        # Align columns to df's node names; missing → 1.
        node_names = df.columns.astype(str).tolist()
        mult = scaler_by_period.reindex(columns=node_names).astype(float)
        periods = df.index.get_level_values("period").astype(str)
        mult = mult.reindex(periods.astype(str)).fillna(1.0)
        mult.index = df.index
        mult.columns = df.columns
        df = df.div(mult)
    return df


def _long_duals(df: "pd.DataFrame", step_col: str) -> pl.DataFrame:
    """A wide ``(solve, period, time)`` × node dual frame → ``(n, d, <step_col>, dual)`` (Utf8 keys)."""
    if df.empty:
        return pl.DataFrame(schema={"n": pl.Utf8, "d": pl.Utf8, step_col: pl.Utf8, "dual": pl.Float64})
    long = df.stack(future_stack=True).rename("dual").reset_index()
    long = long.rename(columns={"period": "d", "time": step_col, "node": "n"})[["n", "d", step_col, "dual"]]
    long[["n", "d", step_col]] = long[["n", "d", step_col]].astype(str)
    return pl.from_pandas(long)


def write_v_dual_node_balance(
    h: "highspy.Highs",
    *,
    solve_name: str,
    output_dir: Path | str,
    realized_dispatch_csv: Path | str | None = None,
    flex_data: "FlexData | None" = None,
    scale_the_objective: float | None = None,
    provider: "object | None" = None,
) -> Path:
    """Write ``v_dual_node_balance__{solve}.parquet`` (``price_t``).

    # ─── PRICE. ─── price_t (v_dual_node_balance) ──────────────────────────────
    # price[n,d,t] = σ_row · dual[row(n,d,t)] / s / w[d,t]
    #   w[d,t] = timestep_weight[d,t] · inflation_op[d] / period_share[d] · branch_weight[d,t]
    #   row(n,d,t) = nodeBalance_eq[n,d,t]                      σ = +1
    #              | nodeBalanceCoarse_eq[n,d,b], t ∈ b         σ = +1   (node_coarse_step)
    #              | nodeBalanceBlock_eq[n,d,b],  t ∈ b         σ = +1   (period_block_time)
    #   s = scale_the_objective (live, per solve);   price = null where w = 0
    #   then ÷ node_capacity_for_scaling[n,d] (1 today)
    # The cost of one more MWh of demand at (n,d,t), in CUR/MWh in the price base of the cost inputs (inflation
    # and the years the period represents removed), not weighted by the timestep weight or by the branch
    # probability.  Definition: :mod:`flextool.engine.output._prices`.

    Nodes of coarse new_stepduration groups (``nodeCoarse``) and ``bind_intraperiod_blocks`` nodes
    (``nodeStateBlock``) have no per-step row; their coarse-step / block dual is broadcast to every ``(d, t)`` of
    the step (without them those nodes would have no column and ``node_prices_dt_e`` would raise a KeyError).
    The rows are the realized ``(d, t)`` of the solve in canonical order.
    """
    from flextool.engine.output._prices import node_prices

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    inv_scale = _resolve_inv_scale_the_objective(scale_the_objective=scale_the_objective)
    kw = dict(solve_name=solve_name, has_time=True, source="row_dual", value_scale=1.0,
              realized_dispatch_csv=realized_dispatch_csv, flex_data=flex_data, provider=provider)

    # Raw duals of the per-step rows: (solve, period, time) × node, in the canonical realized row order.
    df = extract_variable(h, "nodeBalance_eq", ("node",), **kw)
    duals: dict[str, pl.DataFrame | None] = {"nodeBalance_eq": _long_duals(df, "t")}
    block_cols: list[str] = []
    if flex_data is not None:
        coarse = getattr(flex_data, "node_coarse_step", None)
        if coarse is not None and coarse.height > 0:
            raw = extract_variable(h, "nodeBalanceCoarse_eq", ("node",), **kw)
            duals["nodeBalanceCoarse_eq"] = _long_duals(raw, "b_first")
            block_cols += [str(c) for c in raw.columns]
        nsb = getattr(flex_data, "nodeStateBlock", None)
        pbt = getattr(flex_data, "period_block_time", None)
        if nsb is not None and nsb.height > 0 and pbt is not None and pbt.height > 0:
            raw = extract_variable(h, "nodeBalanceBlock_eq", ("node",), **kw)
            duals["nodeBalanceBlock_eq"] = _long_duals(raw, "b_first")
            block_cols += [str(c) for c in raw.columns]

    prices = node_prices(duals, flex_data, inv_scale=inv_scale, weight=_price_weight(flex_data))
    nodes = [str(c) for c in df.columns]
    nodes += [c for c in block_cols if c not in set(nodes)]
    if block_cols:
        nodes = sorted(nodes)              # deterministic node-column order once several families are present
    if nodes and prices.height > 0 and len(df.index) > 0:
        wide = (prices.to_pandas()
                .pivot_table(index=["d", "t"], columns="n", values="price", aggfunc="first", dropna=False))
        rows = [(str(d), str(t)) for _s, d, t in df.index]
        wide = wide.reindex(index=pd.MultiIndex.from_tuples(rows, names=["d", "t"]), columns=nodes)
        out = pd.DataFrame(wide.to_numpy(dtype=float), index=df.index, columns=pd.Index(nodes, name="node"))
        out = _divide_by_row_scaler(out, solve_name=solve_name, flex_data=flex_data)
    else:
        out = df

    path = output_dir / f"v_dual_node_balance__{solve_name}.parquet"
    write_lean_parquet(out, path)
    _logger.debug(
        "Wrote v_dual_node_balance for solve '%s' -> %s (shape %s)",
        solve_name, path, out.shape,
    )
    return path


def _solution_frame(names: Sequence[str], values, var: str, dims: tuple[str, ...]) -> pl.DataFrame:
    """``(*dims, value)`` (Utf8 keys) of every column ``var[…]`` of the solved LP."""
    prefix = f"{var}["
    rows = [tuple(p.strip("'") for p in name[len(prefix):-1].split(",")) + (float(v),)
            for name, v in zip(names, values) if name.startswith(prefix)]
    rows = [r for r in rows if len(r) == len(dims) + 1]
    return pl.DataFrame(rows, schema={**{c: pl.Utf8 for c in dims}, "value": pl.Float64}, orient="row")


def write_v_self_discharge(
    h: "highspy.Highs",
    *,
    solve_name: str,
    output_dir: Path | str,
    flex_data: "FlexData | None",
    realized_dispatch_csv: Path | str | None = None,
    realized_periods_csv: Path | str | None = None,
    provider: "object | None" = None,
    col_names_cache: Sequence[str] | None = None,
    col_value: "object | None" = None,
) -> "Path | None":
    """Write ``v_self_discharge__{solve}.parquet``: the self-discharge of each storage node, MWh per step.

    # ─── OUTPUT. ─── v_self_discharge ─────────────────────────────────────────
    # loss[n,d,t] = (α[n,d,t]·v_state[n,d,t] + β[n,d,t]·S⁻[n,d,t])·state_unitsize[n]
    #   α, β, S⁻ as in nodeBalance_eq (flextool.engine.model._self_discharge); coarse steps / blocks: the step's
    #   total spread over its fine steps by Δt/Δ_b
    # The loss the solve's balance rows used, computed from the solve's own solution (S⁻ can be a look-ahead state
    # of the same solve, the carry or the invested start state, which the realized union of the outputs cannot
    # rebuild).  Rows: the realized (d, t) of v_state; columns: the storage nodes with a self_discharge_loss.

    None (no file) without FlexData or without such a node.
    """
    from flextool.engine.model import _self_discharge as sd

    if flex_data is None or flex_data.p_state_self_discharge is None or flex_data.nodeState is None:
        return None
    fd = flex_data
    storage = {str(x) for x in fd.nodeState["n"].cast(pl.Utf8).to_list()}
    nodes = sorted({str(x) for x in fd.p_state_self_discharge.frame["n"].cast(pl.Utf8).to_list()} & storage)
    if not nodes:
        return None
    names = col_names_cache if col_names_cache is not None else h.allVariableNames()
    values = col_value if col_value is not None else h.getSolution().col_value
    state = _solution_frame(names, values, "v_state", ("n", "d", "t"))

    def at(frame: "pl.DataFrame | None") -> "pl.DataFrame | None":
        return None if frame is None else frame.select(pl.col(c).cast(pl.Utf8) for c in ("n", "d", "t"))

    rp_start = None
    if fd.nodeState_rp is not None and fd.nodeState_rp.height > 0:
        rp_start = _solution_frame(names, values, "v_state_rp_start", ("n", "d", "t"))
    start = None
    fix_first = at(sd.fix_start_first(fd))
    if fix_first is not None:
        su = fd.p_state_unitsize.frame.select(pl.col("n").cast(pl.Utf8), pl.col("value").alias("su"))
        cap = (fd.p_state_existing_capacity.frame
               .select(pl.col("n").cast(pl.Utf8), pl.col("d").cast(pl.Utf8), pl.col("value").alias("cap")))
        built = pl.DataFrame(schema={"n": pl.Utf8, "d": pl.Utf8, "built": pl.Float64})
        for var, frame, ent, d_col, sign in (("v_invest", fd.edd_invest_set, "e", "d_invest", 1.0),
                                             ("v_divest", fd.ndd_divest_active, "n", "d_divest", -1.0)):
            if frame is None or frame.height == 0:
                continue
            chosen = _solution_frame(names, values, var, ("n", d_col))
            alive = frame.select(pl.col(ent).cast(pl.Utf8).alias("n"), pl.col(d_col).cast(pl.Utf8),
                                 pl.col("d").cast(pl.Utf8))
            part = (alive.join(chosen, on=["n", d_col], how="inner").group_by("n", "d")
                    .agg((pl.col("value").sum() * sign).alias("built")))
            built = pl.concat([built, part], how="vertical")
        built = built.group_by("n", "d").agg(pl.col("built").sum())
        start = (fix_first
                 .join(fd.p_state_start.frame.select(pl.col("n").cast(pl.Utf8), pl.col("value").alias("ss")),
                       on="n", how="inner")
                 .join(cap, on=["n", "d"], how="left").join(su, on="n", how="left")
                 .join(built, on=["n", "d"], how="left")
                 .select("n", "d", "t", value=pl.col("ss") * (pl.col("cap").fill_null(0.0)
                                                              + pl.col("built").fill_null(0.0) * pl.col("su"))))
    carry = None
    roll = at(sd.roll_first(fd))
    open_start = sd.open_start_nodes(fd)
    if roll is not None and open_start is not None:
        carry = (roll.join(open_start.select(pl.col("n").cast(pl.Utf8)), on="n", how="semi")
                 .join(fd.p_roll_continue_state.frame.select(pl.col("n").cast(pl.Utf8), "value"), on="n",
                       how="inner"))
    energy = sd.self_discharge_energy(fd, state=state, rp_start=rp_start, start=start, carry=carry,
                                      open_first=at(sd.open_first(fd)))

    # Rows and their order: those of the v_state output of this solve.
    state_df = extract_variable(h, "v_state", ("node",), solve_name=solve_name, has_time=True,
                                realized_dispatch_csv=realized_dispatch_csv,
                                realized_periods_csv=realized_periods_csv, flex_data=flex_data, provider=provider,
                                col_names_cache=names, col_value=values)
    rows = [(str(d), str(t)) for _s, d, t in state_df.index]
    out = pd.DataFrame(0.0, index=state_df.index, columns=pd.Index(nodes, name="node"))
    if energy is not None and len(rows) > 0:
        wide = (energy.filter(pl.col("n").is_in(nodes)).to_pandas()
                .pivot(index=["d", "t"], columns="n", values="value"))   # (n, d, t) unique: self_discharge_energy
        wide = wide.reindex(index=pd.MultiIndex.from_tuples(rows, names=["d", "t"]), columns=nodes)
        out = pd.DataFrame(wide.to_numpy(dtype=float), index=state_df.index,
                           columns=pd.Index(nodes, name="node")).fillna(0.0)
    path = Path(output_dir) / f"v_self_discharge__{solve_name}.parquet"
    write_lean_parquet(out, path)
    return path


def write_v_dual_reserve_balance(
    h: "highspy.Highs",
    *,
    solve_name: str,
    output_dir: Path | str,
    realized_dispatch_csv: Path | str | None = None,
    flex_data: "FlexData | None" = None,
    scale_the_objective: float | None = None,
    provider: "object | None" = None,
) -> Path:
    """Write ``v_dual_reserve__upDown__group__period__t__{solve}.parquet`` (``reserve_price_t``).

    # ─── PRICE. ─── reserve_price_t ─────────────────────────────────────────────
    # reserve_price[r,ud,g,d,t] = ( dual[reserveBalance_timeseries_eq[r,ud,g,d,t]]
    #                             + dual[reserveBalance_dynamic_eq[r,ud,g,d,t]]
    #                             + Σ_f dual[reserveBalance_{up,down}_n_1_eq[r,ud,g,f,d,t]] ) / s / (Δt[d,t] · w[d,t])
    #   w[d,t] = timestep_weight[d,t] · inflation_op[d] / period_share[d] · branch_weight[d,t];   null where w = 0
    #   a row the group does not have contributes 0;  f = failing process (row dim p_fail)
    # CUR per MW held for one hour (= CUR/MWh).  Definition: :mod:`flextool.engine.output._prices`.

    The price of a group is the cost of one more MW of every requirement of the group at once: the sum of
    the duals of all its requirement rows at the step (timeseries, dynamic, and one large-failure row per
    failing process).  The rows share the group's shortfall ``vq_reserve``, so the sum equals
    ``penalty_reserve`` when the shortfall is positive; when the group total ``v_reserve_total`` is positive
    it equals ``−dual[reserve_total_upper]``: the value of one MW from a provider that is not failing.
    The sum does not depend on how a degenerate LP splits the dual between equally large failures.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / (
        f"v_dual_reserve__upDown__group__period__t__{solve_name}.parquet"
    )

    # The requirement rows of a reserve group: reserveBalance_timeseries_eq and reserveBalance_dynamic_eq
    # are indexed (r, ud, g, d, t); reserveBalance_{up,down}_n_1_eq (r, ud, g, p_fail, d, t), one row per
    # failing process, summed over p_fail.  reserve_total_upper (the group total v_reserve_total) is not a
    # requirement row: its dual is minus the sum of the duals of the rows that use the total.
    group_cols = ("reserve", "updown", "node_group")
    df: pd.DataFrame | None = None
    for row_name, cols in (("reserveBalance_timeseries_eq", group_cols),
                           ("reserveBalance_dynamic_eq", group_cols),
                           ("reserveBalance_up_n_1_eq", group_cols + ("p_fail",)),
                           ("reserveBalance_down_n_1_eq", group_cols + ("p_fail",))):
        part = extract_variable(
            h, row_name, cols,
            solve_name=solve_name, has_time=True, source="row_dual",
            realized_dispatch_csv=realized_dispatch_csv,
            flex_data=flex_data,
            provider=provider,
        )
        if df is None:
            df = part      # the timeseries frame, also the shape of an empty output
            continue
        if part.empty:
            continue
        if len(cols) > len(group_cols):
            part = part.T.groupby(level=list(range(len(group_cols))), sort=False).sum().T
            part.columns = part.columns.set_names(list(group_cols))
        df = part if df.empty else df.add(part, fill_value=0.0)
    assert df is not None
    if not df.empty:
        df = df.sort_index(axis=1)

    if df.empty:
        write_lean_parquet(df, out_path)
        _logger.debug(
            "Wrote v_dual_reserve_balance for solve '%s' -> %s (empty)",
            solve_name, out_path,
        )
        return out_path

    # Summed raw duals (0.0 where a group has no row of a family) → price per MW held for one hour; a
    # zero-weight step stays empty.
    inv_scale = _resolve_inv_scale_the_objective(scale_the_objective=scale_the_objective)
    df = _wide_to_price(df, _price_weight(flex_data, per_hour=True), sign=1.0,
                        inv_scale=inv_scale)

    write_lean_parquet(df, out_path)
    _logger.debug(
        "Wrote v_dual_reserve_balance for solve '%s' -> %s (shape %s)",
        solve_name, out_path, df.shape,
    )
    return out_path


def _actual_solve_name(fallback: str, *, provider: "object | None" = None) -> str:
    """Return the roll-level solve name from the provider's ``solve_data/solve_current`` frame.

    In rolling-window scenarios, ``solver.run(complete_solve[solve])`` is
    invoked with the PARENT solve name (the ``complete_solve``) while
    every per-solve frame stores the child ROLL name under its ``solve``
    column (the set ``solve_current`` in the retired GMPL model).  When
    filtering those frames or emitting rows whose format the model
    produced, use the ROLL name — otherwise Python output won't line up
    with the model's.

    Item 163: ``provider`` is the sub-solve's in-memory source; there is no
    disk fallback.  A ``None`` provider or a missing/empty key gives
    ``fallback`` (the ``complete_solve`` name the caller already has).
    """
    seeded = _provider_lookup(provider, "solve_data/solve_current.csv")
    if seeded is None or seeded.height == 0 or len(seeded.columns) == 0:
        return fallback
    return str(seeded[0, 0])




def write_v_invest_fan(
    h: "highspy.Highs",
    *,
    solve_name: str,
    output_dir: Path | str,
    flex_data: "FlexData | None",
    realized_periods_csv: Path | str | None = None,
    provider: "object | None" = None,
    col_names_cache: Sequence[str] | None = None,
    col_value: "object | None" = None,
) -> "Path | None":
    """Write ``v_invest_fan__{solve}.parquet``: the investment of a fan's branch periods, weighted to its label.

    # ─── OUTPUT. ─── v_invest_fan ─────────────────────────────────────────────
    # invest_fan[e,L] = Σ_{b ∈ B(L)} pd_branch_weight[b]·v_invest[e,b]
    #   L a period label of a stochastic fan the solve realizes the investment of (``realized_invest_periods``),
    #   B(L) its forecast-branch periods (``period__branch`` rows (L, b), b ≠ L)
    # Under ``stochastic_invest_method = recourse`` every branch decides its own investment: this is the
    # probability-weighted investment of the fan, the amount whose cost the LP objective carries.  The reported
    # costs add it to the label's own (shared) investment, which is 0 under recourse.  Rows: the realized labels;
    # columns: the entities with a branch investment.

    None (no file) without FlexData, without a fan or without an investment in a branch period.
    """
    labels = _fan_labels(flex_data)
    if not labels:
        return None
    pbf = flex_data.period_branch_full
    branch_of = {str(b): str(d) for d, b in zip(pbf["d"].to_list(), pbf["b"].to_list())
                 if str(d) in labels and str(b) != str(d)}
    weights = flex_data.pd_branch_weight
    if not branch_of or weights is None:
        return None
    names = col_names_cache if col_names_cache is not None else h.allVariableNames()
    values = col_value if col_value is not None else h.getSolution().col_value
    chosen = _solution_frame(names, values, "v_invest", ("e", "b")).filter(pl.col("b").is_in(list(branch_of)))
    if chosen.height == 0:
        return None
    w = weights.frame.select(pl.col("d").cast(pl.Utf8).alias("b"), pl.col("value").alias("w"))
    mapping = pl.DataFrame({"b": list(branch_of), "d": list(branch_of.values())})
    weighted = (chosen.join(w, on="b", how="inner").join(mapping, on="b", how="inner")
                .group_by("e", "d").agg((pl.col("value") * pl.col("w")).sum().alias("value")))
    realized = _load_realized_invest_periods_list(realized_periods_csv, provider=provider)
    rows = [d for d in (realized if realized is not None else sorted(labels)) if d in labels]
    entities = list(dict.fromkeys(chosen["e"].to_list()))
    wide = weighted.to_pandas().pivot(index="d", columns="e", values="value").reindex(index=rows, columns=entities)
    out = pd.DataFrame(wide.to_numpy(dtype=float), columns=pd.Index(entities, name="entity"),
                       index=pd.MultiIndex.from_tuples([(solve_name, d) for d in rows],
                                                       names=["solve", "period"])).fillna(0.0)
    out += 0.0
    path = Path(output_dir) / f"v_invest_fan__{solve_name}.parquet"
    write_lean_parquet(out, path)
    return path


def write_all_variables(
    h: "highspy.Highs",
    *,
    solve_name: str,
    output_dir: Path | str,
    realized_dispatch_csv: Path | str | None = None,
    realized_periods_csv: Path | str | None = None,
    specs: Sequence[VariableSpec] | None = None,
    flex_data: "FlexData | None" = None,
    scale_the_objective: float | None = None,
    provider: "object | None" = None,
    obj_offset: float = 0.0,
) -> list[Path]:
    """Iterate :data:`VARIABLE_SPECS` (or a custom list) and write parquets.

    Returns the list of paths written, one per variable.  Each variable
    is independent — failure on one is logged and does not abort the
    remaining ones, so one bad variable can't lose the whole run.
    ``obj_offset``: the objective constant ``h.getObjectiveValue()`` leaves
    out (``Solution.obj_offset``; see :func:`write_v_obj`).
    """
    specs = specs if specs is not None else VARIABLE_SPECS
    written: list[Path] = []
    # Hoist the expensive HiGHS bulk fetches out of the per-spec loop.
    # extract_variable() otherwise re-materialises ``allVariableNames()``
    # (a fresh multi-million-element Python list), ``getSolution()`` (full
    # col_value/col_dual/row_dual array copies) and ``getLp().row_names_``
    # (a full LP copy incl. all row names) ONCE PER SPEC (×len(specs)).
    # Fetching once here and threading the cached arrays through
    # write_variable_parquet -> extract_variable collapses that to a
    # single fetch per solve.  Output content is unchanged — the same
    # name/value arrays are indexed, just shared.  Set
    # ``FLEXTOOL_DISABLE_OUTPUT_HOIST=1`` to keep the old per-spec fetch
    # (A/B comparison / quick revert).
    col_names_cache: Sequence[str] | None = None
    row_names_cache: Sequence[str] | None = None
    col_value = col_dual = row_dual = None
    if os.environ.get("FLEXTOOL_DISABLE_OUTPUT_HOIST") != "1":
        try:
            col_names_cache = h.allVariableNames()
            _sol = h.getSolution()
            col_value = _sol.col_value
            col_dual = handle_col_dual(h, _sol)
            row_dual = _sol.row_dual
            row_names_cache = h.getLp().row_names_
        except Exception as exc:  # pragma: no cover - defensive fallback
            _logger.warning(
                "output-hoist pre-fetch failed (solve '%s'): %s — "
                "falling back to per-spec fetch", solve_name, exc,
            )
            col_names_cache = row_names_cache = None
            col_value = col_dual = row_dual = None

    for spec in specs:
        try:
            path = write_variable_parquet(
                h, spec,
                solve_name=solve_name,
                output_dir=output_dir,
                realized_dispatch_csv=realized_dispatch_csv,
                realized_periods_csv=realized_periods_csv,
                flex_data=flex_data,
                scale_the_objective=scale_the_objective,
                provider=provider,
                col_names_cache=col_names_cache,
                row_names_cache=row_names_cache,
                col_value=col_value,
                col_dual=col_dual,
                row_dual=row_dual,
            )
            written.append(path)
        except Exception as exc:
            _logger.warning(
                "parquet extraction failed for %s (solve '%s'): %s",
                spec.output_name or spec.name, solve_name, exc,
            )

    # Custom writers for outputs that don't fit a plain VariableSpec.
    _custom_writers = (
        ("v_obj", lambda: write_v_obj(
            h, solve_name=solve_name, output_dir=output_dir,
            flex_data=flex_data,
            scale_the_objective=scale_the_objective,
            obj_offset=obj_offset,
        )),
        ("v_dual_invest_{unit,connection,node}", lambda: write_v_dual_invest_by_class(
            h, solve_name=solve_name, output_dir=output_dir,
            provider=provider,
            realized_periods_csv=realized_periods_csv,
            flex_data=flex_data,
            scale_the_objective=scale_the_objective,
        )),
        ("v_dual_node_balance", lambda: write_v_dual_node_balance(
            h, solve_name=solve_name, output_dir=output_dir,
            realized_dispatch_csv=realized_dispatch_csv,
            flex_data=flex_data,
            scale_the_objective=scale_the_objective,
            provider=provider,
        )),
        ("v_self_discharge", lambda: write_v_self_discharge(
            h, solve_name=solve_name, output_dir=output_dir,
            flex_data=flex_data,
            realized_dispatch_csv=realized_dispatch_csv,
            realized_periods_csv=realized_periods_csv,
            provider=provider,
            col_names_cache=col_names_cache,
            col_value=col_value,
        )),
        ("v_invest_fan", lambda: write_v_invest_fan(
            h, solve_name=solve_name, output_dir=output_dir,
            flex_data=flex_data,
            realized_periods_csv=realized_periods_csv,
            provider=provider,
            col_names_cache=col_names_cache,
            col_value=col_value,
        )),
        ("v_dual_reserve_balance", lambda: write_v_dual_reserve_balance(
            h, solve_name=solve_name, output_dir=output_dir,
            realized_dispatch_csv=realized_dispatch_csv,
            flex_data=flex_data,
            scale_the_objective=scale_the_objective,
            provider=provider,
        )),
        # ``entity_all_capacity`` moved to ``handoff_writers.py`` — it's
        # conceptually a solve-to-solve handoff (accumulates across
        # solves) and needs the same CSV layout phase 3 produces, not a
        # per-solve parquet.
    )
    for label, fn in _custom_writers:
        try:
            result = fn()
            if isinstance(result, list):
                written.extend(result)
            elif result is not None:
                written.append(result)
        except Exception as exc:
            _logger.warning(
                "custom writer failed for %s (solve '%s'): %s",
                label, solve_name, exc,
            )
    _logger.info(
        "Wrote %d output variables for solve '%s' -> %s",
        len(written), solve_name, output_dir,
    )
    return written
