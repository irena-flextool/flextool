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

import argparse
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

    from flextool.engine_polars.input import FlexData

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
    # Agent 1.8 — block-aware output expansion.  When non-None, the
    # extracted frame is broadcast from the variable's temporal-resolution
    # block down to the finest timeline.  The raw MPS only emits values
    # at the block's coarse ``(period, step)`` pairs; fine steps covered
    # by a coarse step receive the same value so downstream CSV / parquet
    # readers always see a rectangular fine-grid frame (design rule
    # from Agent 1.8: "print all at finest resolution and drop the block
    # dimension").
    # Recognised values:
    #   * "process_block" — lookup per ``col_names[0]`` in
    #                       ``solve_data/process_block.csv``.  Used for
    #                       v_flow, v_ramp, v_online_*, v_startup_*,
    #                       v_shutdown_* (every process-scoped var).
    #   * "node_block"    — lookup in ``solve_data/entity_block.csv``.
    #                       Used for v_state, v_angle.
    # In the degenerate case every entity maps to ``"default"`` so the
    # overlap lookup is identity and the broadcast is a no-op.
    expand_by: str | None = None


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
    work_folder: Path | str | None,
    scale_the_objective: float | None = None,
) -> float:
    """Return ``1 / scale_the_objective`` for the current solve.

    Phase G — when ``scale_the_objective`` is supplied directly (cascade-
    threaded), use it and skip the disk read entirely.  Otherwise reads
    ``<work_folder>/solve_data/scale_the_objective.csv`` (Agent 12;
    emitted by :func:`flextool.engine_polars._emit_solve_writers.write_scale_the_objective`).
    Falls back to ``1 / 1e-6`` when the file is missing / empty /
    unreadable — mirrors the ``default 1e-6`` clause on
    ``param scale_the_objective`` in ``flextool.mod``.
    """
    if scale_the_objective is not None and scale_the_objective > 0:
        return 1.0 / float(scale_the_objective)
    if work_folder is None:
        return 1.0 / _DEFAULT_SCALE_THE_OBJECTIVE
    path = Path(work_folder) / "solve_data" / "scale_the_objective.csv"
    if not path.exists():
        return 1.0 / _DEFAULT_SCALE_THE_OBJECTIVE
    try:
        df = pd.read_csv(path)
    except Exception:
        return 1.0 / _DEFAULT_SCALE_THE_OBJECTIVE
    if df.empty or "value" not in df.columns:
        return 1.0 / _DEFAULT_SCALE_THE_OBJECTIVE
    try:
        val = float(df["value"].iloc[0])
    except (ValueError, TypeError):
        return 1.0 / _DEFAULT_SCALE_THE_OBJECTIVE
    if not (val > 0):
        return 1.0 / _DEFAULT_SCALE_THE_OBJECTIVE
    return 1.0 / val


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
    # (``..._n``) constraints (model.py / _cumulative_invest.py).  The
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
    # Agent 1.8: ``expand_by`` broadcasts coarse-block values to every
    # covered fine timestep.  Degenerate (every entity on 'default'): no-op.
    VariableSpec("v_flow",             ("process", "source", "sink"), expand_by="process_block"),
    # Reverse-flow auxiliary for method_2way_1var_off arcs.  The signed
    # net flow on such an arc is ``v_flow - v_flow_back``; the output layer
    # folds it into ``r.flow_dt`` (calc_capacity_flows) so the reported
    # connection flow carries the correct sign (negative = sink→source).
    VariableSpec("v_flow_back",        ("process", "source", "sink"), expand_by="process_block"),
    VariableSpec("v_ramp",             ("process", "source", "sink"), expand_by="process_block"),
    # Reserve participants are pinned to the default block in V1 (Agent
    # 1.7), so v_reserve effectively needs no expansion — but the
    # broadcast is still safely identity there.
    VariableSpec("v_reserve",          ("process", "reserve", "updown", "node"), expand_by="process_block"),
    VariableSpec("v_state",            ("node",), expand_by="node_block"),
    VariableSpec("v_online_linear",    ("process",), expand_by="process_block"),
    VariableSpec("v_startup_linear",   ("process",), expand_by="process_block"),
    VariableSpec("v_shutdown_linear",  ("process",), expand_by="process_block"),
    VariableSpec("v_online_integer",   ("process",), expand_by="process_block"),
    VariableSpec("v_startup_integer",  ("process",), expand_by="process_block"),
    VariableSpec("v_shutdown_integer", ("process",), expand_by="process_block"),
    VariableSpec("v_angle",            ("node",), expand_by="node_block"),

    # -- Time-indexed slack / penalty variables -----------------------------
    # Agent 9 ``unscale_by="node_cap"`` / ``"group_cap"`` un-scales row
    # scaling when ``use_row_scaling=yes`` (no-op in Mode A where the
    # scaler defaults to 1).  See flextool/SLACK_CONVENTION.md for the
    # single-variable slack convention.
    # Agent 1.8: vq_state_up / vq_state_down appear in the node balance,
    # which is emitted at the node's block — broadcast via node_block.
    VariableSpec(
        "vq_state_up", ("node",),
        unscale_by="node_cap", expand_by="node_block",
    ),
    VariableSpec(
        "vq_state_down", ("node",),
        unscale_by="node_cap", expand_by="node_block",
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
    VariableSpec("v_invest",           ("entity",), has_time=False),
    VariableSpec("v_divest",           ("entity",), has_time=False),
    # No t axis; the row scaler is still keyed by (g, d).
    VariableSpec(
        "vq_capacity_margin", ("group",), has_time=False,
        unscale_by="group_cap",
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
    # — ``maxInvest_entity_period_p`` (model.py:2319) for the process arm
    # and ``maxInvest_entity_period_n`` (model.py:2336) for the node arm
    # etc. — so the reader matches both via ``derived_from`` and unions
    # the resulting frames.
    _invest_dual("maxInvest_entity_period",          "entity", "maxInvest_period",
                 derived_from=("maxInvest_entity_period_p",
                                "maxInvest_entity_period_n")),
    # ``maxInvest_entity_total`` is asymmetric: process side keeps the
    # bare name (model.py:2442), node side gets the ``_n`` suffix
    # (model.py:2494).  Period is summed out inside ``Sum(over=("d",))``
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
    _invest_dual("maxInvestGroup_entity_total",      "group",  "maxInvestGroup_total",
                 derived_from=("maxInvestGroup_entity_total_p",
                                "maxInvestGroup_entity_total_n")),
    # ``maxInvestGroup_entity_cumulative`` keeps a single non-suffixed
    # name (_cumulative_invest.py:990).
    _invest_dual("maxInvestGroup_entity_cumulative", "group",  "maxInvestGroup_cumulative"),

    # -- Investment-floor (min-side) duals ----------------------------------
    # Mirror of the maxInvest families above, reading the row duals of the
    # ``>=`` lower-floor constraints emitted by ``_emit_*_minmax`` /
    # ``_emit_group_invest_*`` (flextool/engine_polars/_cumulative_invest.py).
    # Purely additive (Increment 3): makes ``v_dual_min*`` available for the
    # later synthesis increment; nothing consumes these yet.  Same scale
    # sentinel + ``source="row_dual"`` as the max-side; absent families
    # degrade to an empty frame identically.
    #
    # ``minInvest_entity_period`` splits process (``_p``) / node (``_n``)
    # arms (_cumulative_invest.py:383,397) — same shape as the max-side.
    _invest_dual("minInvest_entity_period",          "entity", "minInvest_period",
                 derived_from=("minInvest_entity_period_p",
                                "minInvest_entity_period_n")),
    # ``minInvest_entity_total`` is NOT name-symmetric with its max-side
    # counterpart: the process arm carries the ``_p`` suffix
    # (``minInvest_entity_total_p``, _cumulative_invest.py:460) rather than
    # the bare name ``maxInvest_entity_total`` keeps.  More importantly, the
    # min invest-total constraint is indexed per ``(entity, period)`` —
    # ``over = (p|n, d)`` (_cumulative_invest.py:450,490) with the LHS
    # summing over ``d_invest`` — so it KEEPS a period axis.  The max-side
    # total sums the period out (``Sum(over=("d",))``, model.py:2578) and is
    # therefore ``has_period=False``.  We use the default ``has_period=True``
    # to match the actual min constraint's row bracket.
    _invest_dual("minInvest_entity_total",           "entity", "minInvest_total",
                 derived_from=("minInvest_entity_total_p",
                                "minInvest_entity_total_n")),
    _invest_dual("minCumulative_capacity",           "entity", "minCumulative",
                 derived_from=("minCumulative_capacity_p",
                                "minCumulative_capacity_n")),
    _invest_dual("minInvestGroup_entity_period",     "group",  "minInvestGroup_period",
                 derived_from=("minInvestGroup_entity_period_p",
                                "minInvestGroup_entity_period_n")),
    _invest_dual("minInvestGroup_entity_total",      "group",  "minInvestGroup_total",
                 derived_from=("minInvestGroup_entity_total_p",
                                "minInvestGroup_entity_total_n")),
    # ``minInvestGroup_entity_cumulative`` keeps a single non-suffixed
    # name (_cumulative_invest.py:1001), like the max-side cumulative.
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
    # * v_dual_reserve__upDown__group__period__t — max() across up to 3
    #   constraint duals per (r,ud,g,d,t).
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
    """Return ``{(period, time), …}`` from ``realized_dispatch.csv``, or None.

    Used for O(1) membership checks in :func:`extract_variable`.  For
    synthesis of empty-frame row order (which must match the canonical
    iteration order in phase-1 printfs), use
    :func:`_load_realized_list` instead — an ordered variant.
    """
    if realized_dispatch_csv is None:
        return None
    path = Path(realized_dispatch_csv)
    # Step 1-e — Provider-aware: under the in-memory cascade the file
    # isn't on disk but the per-sub-solve Provider has the frame.  The
    # transitional seed-funnel fallback in :func:`_provider_lookup`
    # keeps unplumbed callers working during the dual-write window.
    seeded = _provider_lookup(provider, path)
    if seeded is not None:
        period_col = "period"
        time_col = "step" if "step" in seeded.columns else "time"
        return set(zip(
            seeded[period_col].cast(str).to_list(),
            seeded[time_col].cast(str).to_list(),
        ))
    if not path.exists():
        # Expected fallback path on every solve where the previous
        # iteration didn't realise any timesteps yet -- write all.
        # Not a warning; just informational.
        _logger.debug("realized_dispatch file missing, writing all timesteps: %s", path)
        return None
    realized = pd.read_csv(path)
    period_col = "period"
    time_col = "step" if "step" in realized.columns else "time"
    return set(
        zip(
            realized[period_col].astype(str).to_list(),
            realized[time_col].astype(str).to_list(),
        )
    )


def _load_realized_list(
    realized_dispatch_csv: Path | str | None,
    *,
    provider: "object | None" = None,
) -> list[tuple[str, str]] | None:
    """Return ``[(period, time), …]`` from ``realized_dispatch.csv`` in file order.

    The CSV file order matches the canonical
    ``for {(d, t) in dt_realize_dispatch}`` iteration order that every
    dt-indexed phase-1 printf uses — so synthesizing empty-frame rows
    in this order reproduces the CSV-read parameter's row order exactly.
    """
    if realized_dispatch_csv is None:
        return None
    path = Path(realized_dispatch_csv)
    seeded = _provider_lookup(provider, path)
    if seeded is not None:
        period_col = "period"
        time_col = "step" if "step" in seeded.columns else "time"
        return list(zip(
            seeded[period_col].cast(str).to_list(),
            seeded[time_col].cast(str).to_list(),
        ))
    if not path.exists():
        return None
    realized = pd.read_csv(path)
    period_col = "period"
    time_col = "step" if "step" in realized.columns else "time"
    return list(
        zip(
            realized[period_col].astype(str).to_list(),
            realized[time_col].astype(str).to_list(),
        )
    )


def _load_realized_periods(
    realized_periods_csv: Path | str | None,
    *,
    provider: "object | None" = None,
) -> set[str] | None:
    """Return ``{period, …}`` from a ``period``-only CSV, or None."""
    if realized_periods_csv is None:
        return None
    path = Path(realized_periods_csv)
    seeded = _provider_lookup(provider, path)
    if seeded is not None:
        return set(seeded["period"].cast(str).to_list())
    if not path.exists():
        _logger.debug("realized periods file missing, writing all periods: %s", path)
        return None
    realized = pd.read_csv(path)
    return set(realized["period"].astype(str).to_list())


def _load_realized_periods_list(
    realized_periods_csv: Path | str | None,
) -> list[str] | None:
    """Return ``[period, …]`` in CSV file order."""
    if realized_periods_csv is None:
        return None
    path = Path(realized_periods_csv)
    if not path.exists():
        return None
    realized = pd.read_csv(path)
    return list(realized["period"].astype(str).to_list())


def _name_regex(var_name: str) -> re.Pattern[str]:
    """Return a compiled regex matching ``<var_name>[...]``."""
    return re.compile(rf"^{re.escape(var_name)}\[(.+)\]$")


def _load_canonical_dt_order(
    work_folder: Path | str | None,
    solve_name: str,
    *,
    flex_data: "FlexData | None" = None,
) -> list[tuple[str, str]] | None:
    """Return ``[(period, time), …]`` in the canonical iteration order.

    When ``flex_data`` is supplied, prefer the in-memory
    ``flex_data.realized_dispatch`` (already a polars frame of
    ``(period, step)``); skip the CSV read entirely.

    Source priority (file fallback):
      1. ``solve_data/p_step_duration.csv`` — written by the phase-1
         printf ``for {s in solve_current, (d, t) in dt_realize_dispatch}``.
      2. ``solve_data/dt_realize_dispatch_set.csv`` — the polars
         cascade's authoritative emission set (mirrors the .mod's
         ``dt_realize_dispatch`` after the ``output_horizon`` toggle).
         Carries forecast-branch rows for stochastic scenarios where
         ``realized_dispatch.csv`` is anchor-only by design.

    Every dt-indexed phase-1 printf in ``flextool.mod`` uses the same
    set iteration, so this sequence is the row order ALL parameter
    CSVs of dt arity have.

    Filtered to ``solve_name`` when the CSV carries a ``solve`` column;
    ``dt_realize_dispatch_set.csv`` is per-solve already (no solve col).
    Returns ``None`` when neither file is present.
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
    if work_folder is None:
        return None
    sd = Path(work_folder) / "solve_data"
    psd = sd / "p_step_duration.csv"
    if psd.exists():
        df = pd.read_csv(psd, usecols=["solve", "period", "time"], dtype=str)
        df = df[df["solve"] == str(solve_name)]
        return list(zip(df["period"].to_list(), df["time"].to_list()))
    drd = sd / "dt_realize_dispatch_set.csv"
    if drd.exists():
        df = pd.read_csv(drd, usecols=["period", "time"], dtype=str)
        return list(zip(df["period"].to_list(), df["time"].to_list()))
    return None


def _load_canonical_d_order(
    work_folder: Path | str | None,
    solve_name: str,
    *,
    flex_data: "FlexData | None" = None,
) -> list[str] | None:
    """Return ``[period, …]`` in the canonical iteration order.

    When ``flex_data.realized_dispatch`` is in memory, derive the
    ordered distinct period list directly (skipping the disk read).

    Source (file fallback): ``solve_data/p_years_from_start_d.csv`` —
    written by the phase-1 printf ``for {s in solve_current, d in
    d_realize_dispatch_or_invest}``.  Period-indexed parameter CSVs
    use the same iteration.  Filtered to ``solve_name``.  Returns
    ``None`` if the file is absent.
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
    if work_folder is None:
        return None
    path = Path(work_folder) / "solve_data" / "p_years_from_start_d.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path, usecols=["solve", "period"], dtype=str)
    df = df[df["solve"] == str(solve_name)]
    return list(df["period"].to_list())


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
        values = col_dual if col_dual is not None else h.getSolution().col_dual
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
    work_folder = (
        Path(realized_dispatch_csv).parent.parent
        if realized_dispatch_csv is not None
        else (
            Path(realized_periods_csv).parent.parent
            if realized_periods_csv is not None
            else None
        )
    )
    if has_period and has_time:
        canonical_rows: list[tuple[str, ...]] | None = (
            _load_canonical_dt_order(work_folder, solve_name, flex_data=flex_data)
        )
        # Fallback for callers that only provide ``realized_dispatch_csv``
        # (e.g. unit tests with a synthetic CSV outside ``solve_data/``).
        if canonical_rows is None and realized_dispatch_csv is not None:
            canonical_rows = _load_realized_list(
                realized_dispatch_csv, provider=provider,
            )
    elif has_period:
        canonical_d = _load_canonical_d_order(work_folder, solve_name, flex_data=flex_data)
        if canonical_d is None and realized_periods_csv is not None:
            canonical_d = _load_realized_periods_list(realized_periods_csv)
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
    work_folder = (
        Path(realized_dispatch_csv).parent.parent
        if realized_dispatch_csv is not None
        else (
            Path(realized_periods_csv).parent.parent
            if realized_periods_csv is not None
            else None
        )
    )
    effective_scale = spec.value_scale
    if spec.value_scale == _INV_SCALE_THE_OBJECTIVE:
        effective_scale = _resolve_inv_scale_the_objective(
            work_folder, scale_the_objective=scale_the_objective,
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
        )
    # Agent 1.8 — block-aware output expansion.  Broadcast coarse-block
    # values to every covered fine timestep so parquet output stays
    # rectangular at the finest resolution.  Degenerate case (every
    # entity on 'default'): no-op, bit-identical to pre-Agent-1.8.
    # Apply BEFORE unscale so the row scaler (keyed at the fine grid)
    # multiplies the broadcasted values consistently.
    if spec.expand_by is not None:
        df = _apply_block_expand(
            df, spec.expand_by, work_folder,
            flex_data=flex_data, provider=provider,
        )
    # Agent 9 — row-scaling un-scaling applied at the output boundary.
    # Source CSV comes from the same work folder as realized_*_csv
    # (``work_folder`` was inferred above for the scale_the_objective
    # resolution — reuse it).
    if spec.unscale_by is not None:
        df = _apply_unscale(df, spec.unscale_by, work_folder, solve_name, flex_data=flex_data)
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


def _load_entity_class(work_folder: Path, set_name: str) -> set[str]:
    """Load the members of an entity class directly from ``input/``.

    Sourcing from ``input/`` keeps the Category C custom writers
    independent of phase 3 (which runs AFTER our writers inside
    ``_run_highs``) — important for the planned phase-3 retirement.

    ``set_name`` maps to files like ``input/process_unit.csv``,
    ``input/process_connection.csv``, ``input/node.csv``.  The input
    files are long-format with a single column named after the set.
    """
    path = work_folder / "input" / f"{set_name}.csv"
    if not path.exists():
        return set()
    df = pd.read_csv(path)
    if df.empty or len(df.columns) == 0:
        return set()
    return set(df.iloc[:, 0].astype(str).tolist())


def _load_inflation_factor(
    work_folder: Path,
    flex_data: "FlexData | None" = None,
) -> dict[str, float]:
    """``{period: p_inflation_factor_operations_yearly}``.

    Phase G — when ``flex_data`` is supplied, trust the in-memory carrier
    (``None`` ⇒ empty dict, matching the disk fallback's missing-file
    branch).  CSV fallback retained for callers without FlexData.

    Written by the model during phase 1 (derived parameter moved above
    ``solve;``).
    """
    if flex_data is not None:
        param = getattr(flex_data, "p_inflation_op", None)
        if param is None:
            return {}
        try:
            f = param.frame
            period_col = f.columns[0]
            return dict(zip(
                f[period_col].cast(str).to_list(),
                f["value"].cast(float).to_list(),
            ))
        except Exception:  # noqa: BLE001
            pass
    path = work_folder / "solve_data" / "solve__p_inflation_factor_operations_yearly.csv"
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    period_col = "period"
    value_col = [c for c in df.columns if c not in ("solve", "period")][0]
    return dict(zip(df[period_col].astype(str), df[value_col].astype(float)))


def _load_complete_period_share_of_year(
    work_folder: Path,
    flex_data: "FlexData | None" = None,
) -> dict[str, float]:
    """``{period: complete_period_share_of_year}`` — Phase G prefers
    ``flex_data.p_period_share`` (trusted when supplied; ``None`` ⇒ {}).
    CSV fallback retained."""
    if flex_data is not None:
        param = getattr(flex_data, "p_period_share", None)
        if param is None:
            return {}
        try:
            f = param.frame
            period_col = f.columns[0]
            return dict(zip(
                f[period_col].cast(str).to_list(),
                f["value"].cast(float).to_list(),
            ))
        except Exception:  # noqa: BLE001
            pass
    path = work_folder / "solve_data" / "complete_period_share_of_year.csv"
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    period_col = "period"
    value_col = [c for c in df.columns if c not in ("solve", "period")][0]
    return dict(zip(df[period_col].astype(str), df[value_col].astype(float)))


# ---------------------------------------------------------------------------
# Row-scaler CSVs (Agent 9)
# ---------------------------------------------------------------------------


def _load_row_scaler(
    work_folder: Path | str | None,
    kind: str,
    solve_name: str,
    *,
    flex_data: "FlexData | None" = None,
) -> pd.DataFrame | None:
    """Read ``solve_data/solve__{node,group}_capacity_for_scaling.csv``.

    Phase G — prefers ``flex_data.p_{node,group}_capacity_for_scaling``
    (Param, already in memory).  CSV fallback retained.

    Format (wide, produced by the AMPL phase-1 printf block at
    ``flextool.mod:4805``)::

        solve,period,entity1,entity2,...
        <solve>,<period>,<scaler>,...

    Returned frame has ``(solve, period)`` as row MultiIndex and the
    entity names as columns.  Filtered to ``solve_name`` on read.

    Returns ``None`` when the CSV is missing or empty — callers then
    treat the scaler as 1 everywhere (no-op).

    ``kind`` is ``"node"`` or ``"group"``.
    """
    if flex_data is not None:
        attr = f"p_{kind}_capacity_for_scaling"
        param = getattr(flex_data, attr, None)
        if param is not None:
            try:
                # Param frame: columns (entity, "d", "value").  Pivot to
                # wide ``(solve, period)``-row × entity-column shape used
                # downstream.  ``solve`` is the active solve_name.
                f = param.frame
                cols = f.columns
                # Drop value column from the pivot index set.
                if "value" in cols:
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
                pass
    if work_folder is None:
        return None
    path = Path(work_folder) / "solve_data" / f"solve__{kind}_capacity_for_scaling.csv"
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path)
    except Exception:
        return None
    if df.empty or "solve" not in df.columns or "period" not in df.columns:
        return None
    df = df[df["solve"].astype(str) == str(solve_name)]
    if df.empty:
        return None
    df = df.set_index(["solve", "period"])
    # Column dtype: float.  Empty-entity columns possible if the model
    # emitted headers but no values; ignore parsing failures.
    df = df.apply(pd.to_numeric, errors="coerce")
    return df


# ---------------------------------------------------------------------------
# Agent 1.8 — block-aware output expansion
# ---------------------------------------------------------------------------


def _load_entity_block_map(
    work_folder: Path | str | None, kind: str,
    *,
    provider: "object | None" = None,
) -> dict[str, str]:
    """Return ``{entity: block}`` read from ``solve_data/{entity,process}_block.csv``.

    * ``kind="node_block"``  → reads ``entity_block.csv`` (columns
      ``entity, block``) — every node maps to its temporal-resolution
      block.
    * ``kind="process_block"`` → reads ``process_block.csv`` (columns
      ``process, block``) — per-process unified block (Agent 1.6).

    When *provider* is supplied and carries the frame (cascade path —
    block CSVs are kept in memory rather than flushed to disk), the
    Provider lookup wins over the disk read.  Missing both →
    empty / default fall-through (caller treats every entity as on the
    ``"default"`` block, i.e. identity overlap).
    """
    if work_folder is None and provider is None:
        return {}
    if kind == "node_block":
        path = (Path(work_folder) if work_folder is not None else Path()) / "solve_data" / "entity_block.csv"
        key_col = "entity"
    elif kind == "process_block":
        path = (Path(work_folder) if work_folder is not None else Path()) / "solve_data" / "process_block.csv"
        key_col = "process"
    else:
        return {}
    # Δ.31 — provider-first: the cascade keeps block frames in-memory
    # because ``emit_block_data_for_solve`` registers them on the
    # Provider rather than flushing to disk.  Without this lookup the
    # output writer would broadcast nothing for daily-block fixtures
    # (lh2_three_region).
    pframe = _provider_lookup(provider, path)
    if pframe is not None and pframe.height > 0:
        cols = pframe.columns
        if key_col in cols and "block" in cols:
            return dict(zip(
                pframe[key_col].cast(pl.Utf8).to_list(),
                pframe["block"].cast(pl.Utf8).to_list(),
            ))
    if not path.exists():
        return {}
    try:
        df = pd.read_csv(path, dtype=str)
    except Exception:
        return {}
    if df.empty or key_col not in df.columns or "block" not in df.columns:
        return {}
    return dict(zip(df[key_col].astype(str), df["block"].astype(str)))


def _load_overlap_fine_to_coarse(
    work_folder: Path | str | None,
    *,
    provider: "object | None" = None,
) -> dict[tuple[str, str, str], str]:
    """Return ``{(period, block_coarse, step_fine): step_coarse}``.

    Read from ``solve_data/overlap_set.csv`` (columns ``period,
    block_coarse, step_coarse, block_fine, step_fine, fraction``).
    Only rows where ``block_fine == 'default'`` are kept — those are the
    rows used to broadcast a coarse-block value to every fine timestep
    it covers.

    Provider-first: when *provider* carries the frame, prefer it over
    the disk read (cascade path keeps the block CSVs in memory).
    Missing both → empty dict (caller treats every entity as on the
    default block → identity broadcast).
    """
    if work_folder is None and provider is None:
        return {}
    path = (Path(work_folder) if work_folder is not None else Path()) / "solve_data" / "overlap_set.csv"
    pframe = _provider_lookup(provider, path)
    df: pd.DataFrame | None = None
    if pframe is not None and pframe.height > 0:
        df = pframe.to_pandas()
    elif path.exists():
        try:
            df = pd.read_csv(path, dtype=str)
        except Exception:
            return {}
    if df is None:
        return {}
    required = {"period", "block_coarse", "step_coarse", "block_fine", "step_fine"}
    if not required.issubset(df.columns):
        return {}
    df = df[df["block_fine"].astype(str) == "default"]
    if df.empty:
        return {}
    return {
        (str(p), str(bc), str(sf)): str(sc)
        for p, bc, sf, sc in zip(
            df["period"], df["block_coarse"],
            df["step_fine"], df["step_coarse"],
        )
    }


def _apply_block_expand(
    df: pd.DataFrame,
    expand_by: str,
    work_folder: Path | str | None,
    *,
    flex_data: "FlexData | None" = None,
    provider: "object | None" = None,
) -> pd.DataFrame:
    """Broadcast coarse-block variable values to covered fine timesteps.

    For each column ``e`` whose entity maps to a non-default block ``b``,
    every fine row ``(d, tf)`` has its value replaced with the coarse
    value at ``(d, tc)`` where ``tc = overlap[(d, b, tf)]``.  Entities on
    the default block are left untouched (identity broadcast).

    The DataFrame's row index must be ``(solve, period, time)``; the
    column (Multi)Index's first level is the entity name.

    Degenerate case (every entity on ``'default'``): no columns trigger
    the broadcast → returns *df* unchanged, bit-identical to pre-Agent-
    1.8 state.
    """
    if df.empty or df.shape[1] == 0:
        return df
    if expand_by not in ("process_block", "node_block"):
        return df
    if not isinstance(df.index, pd.MultiIndex):
        return df
    level_names = df.index.names or []
    if "period" not in level_names or "time" not in level_names:
        return df

    entity_block = _load_entity_block_map(
        work_folder, expand_by, provider=provider,
    )
    # Fast path: no entity on a non-default block → nothing to do.
    if not any(v != "default" for v in entity_block.values()):
        return df

    overlap = _load_overlap_fine_to_coarse(work_folder, provider=provider)
    if not overlap:
        return df

    # Entity level on the column index (row 0 of MultiIndex; the only
    # level otherwise).  Agent 1.8's expand_by is always keyed on
    # ``col_names[0]`` by construction.
    if isinstance(df.columns, pd.MultiIndex):
        entity_names = df.columns.get_level_values(0).astype(str).tolist()
    else:
        entity_names = df.columns.astype(str).tolist()

    # Working in-place would mutate the caller's frame — copy once.
    out = df.copy()

    # Row tuples (period, time) in the frame's order — we'll vector-assign
    # into each expand-target column via numpy positional writes.
    periods = df.index.get_level_values("period").astype(str).to_numpy()
    times = df.index.get_level_values("time").astype(str).to_numpy()
    row_count = len(periods)

    # Column axis for block-aware expansion: entity per column position.
    for col_pos, entity in enumerate(entity_names):
        block = entity_block.get(entity, "default")
        if block == "default":
            continue  # identity broadcast → leave column alone
        # Build a per-row source index — position of the coarse row to copy
        # from.  For each (d, tf), find tc via overlap; then locate the row
        # in the frame.  Rows whose (d, tf) has no overlap entry keep their
        # own value (defensive — shouldn't happen with consistent data).
        source_pos = list(range(row_count))
        # Build a (period, time) → row_pos map once.
        row_pos_map: dict[tuple[str, str], int] = {
            (p, t): i for i, (p, t) in enumerate(zip(periods, times))
        }
        for i in range(row_count):
            d = periods[i]
            tf = times[i]
            tc = overlap.get((d, block, tf))
            if tc is None:
                continue
            src = row_pos_map.get((d, tc))
            if src is None:
                continue
            source_pos[i] = src
        # Apply the column-scoped rewrite: new values = existing values
        # indexed by source_pos.  ``out.iloc[:, col_pos]`` returns a view;
        # reassign so pandas records the update without triggering a
        # SettingWithCopy warning.
        col_values = out.iloc[:, col_pos].to_numpy()
        out.iloc[:, col_pos] = col_values[source_pos]
    return out


def _apply_unscale(
    df: pd.DataFrame,
    unscale_by: str,
    work_folder: Path | str | None,
    solve_name: str,
    *,
    flex_data: "FlexData | None" = None,
) -> pd.DataFrame:
    """Multiply *df* by the row scaler identified by *unscale_by*.

    Handles two scaler kinds:

    * ``"node_cap"`` — ``solve_data/solve__node_capacity_for_scaling.csv`` keyed
      by (period, node).  ``df`` row index is ``(solve, period[, time])``
      and columns are node names; we broadcast the period row of the
      scaler across the time dimension.
    * ``"group_cap"`` — ``solve_data/solve__group_capacity_for_scaling.csv``
      keyed by (period, group).  Columns = group names.  Same
      period-broadcast for time-indexed frames; for the no-t case (only
      ``vq_capacity_margin`` today) the row index is just ``(solve,
      period)`` and the element-wise multiply aligns directly.

    Missing CSV / unknown columns / Mode A (scaler = 1) all collapse to
    a safe no-op: rows without a matching scaler are unchanged.
    """
    if df.empty or df.shape[1] == 0:
        return df
    kind = {"node_cap": "node", "group_cap": "group"}.get(unscale_by)
    if kind is None:
        return df
    scaler = _load_row_scaler(work_folder, kind, solve_name, flex_data=flex_data)
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
    work_folder: Path | str | None = None,
    flex_data: "FlexData | None" = None,
    scale_the_objective: float | None = None,
) -> Path:
    """Write ``v_obj__{solve}.parquet`` — objective value for this solve.

    Model writes ``total_cost.val / scale_the_objective``.  HiGHS's
    ``getObjectiveValue()`` returns the raw (scaled) value; we undo the
    scaling.  Agent 12: ``scale_the_objective`` is now per-solve
    (``solve_data/scale_the_objective.csv``); pass ``work_folder`` so
    the live value is read, else fall back to the legacy ``1e-6`` scalar.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    wf = Path(work_folder) if work_folder is not None else output_dir.parent
    inv_scale = _resolve_inv_scale_the_objective(
        wf, scale_the_objective=scale_the_objective,
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
        raw_obj = float(h.getObjectiveValue())
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
    realized_periods_csv: Path | str | None = None,
    work_folder: Path | str | None = None,
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
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    wf = Path(work_folder) if work_folder is not None else output_dir.parent

    # ``_resolve_inv_scale_the_objective`` returns ``1 / scale_the_objective``
    # (guarding None / non-positive → default 1e-6), exactly as the
    # node-balance writer uses it.
    inv_scale = _resolve_inv_scale_the_objective(
        wf, scale_the_objective=scale_the_objective,
    )
    duals = extract_variable(
        h, "v_invest", ("entity",),
        solve_name=solve_name, has_time=False, source="col_dual",
        value_scale=inv_scale,
        realized_periods_csv=realized_periods_csv,
        flex_data=flex_data,
    )

    classes = {
        "process_unit": _load_entity_class(wf, "process_unit"),
        "process_connection": _load_entity_class(wf, "process_connection"),
        "node": _load_entity_class(wf, "node"),
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


def write_v_dual_node_balance(
    h: "highspy.Highs",
    *,
    solve_name: str,
    output_dir: Path | str,
    realized_dispatch_csv: Path | str | None = None,
    work_folder: Path | str | None = None,
    flex_data: "FlexData | None" = None,
    scale_the_objective: float | None = None,
) -> Path:
    """Write ``v_dual_node_balance__{solve}.parquet``.

    Model formula (for n not in nodeStateBlock)::

        -nodeBalance_eq[n, d, t].dual
         / p_inflation_factor_operations_yearly[d]
         / scale_the_objective

    Equivalently, raw-dual × (−1e6 / inflation[d]).  Nodes in
    ``nodeStateBlock`` use ``nodeBalanceBlock_eq`` summed over
    period_block_time — NOT YET IMPLEMENTED HERE.  Scenarios using
    representative periods with block storage will see missing/zero
    entries for those nodes.

    The polars LP emits ``nodeBalance_eq`` with arity-3
    ``(node, period, time)``; this writer reads that arity directly.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    wf = Path(work_folder) if work_folder is not None else output_dir.parent

    # Raw duals × -(1 / scale_the_objective); per-period inflation
    # division applied after.  Agent 12: resolve the live scalar so a
    # non-default scale_the_objective propagates.
    inv_scale = _resolve_inv_scale_the_objective(
        wf, scale_the_objective=scale_the_objective,
    )
    df = extract_variable(
        h, "nodeBalance_eq", ("node",),
        solve_name=solve_name, has_time=True, source="row_dual",
        value_scale=-inv_scale,
        realized_dispatch_csv=realized_dispatch_csv,
        flex_data=flex_data,
    )

    if not df.empty:
        inflation = _load_inflation_factor(wf, flex_data=flex_data)
        if inflation:
            # Divide each row by its period's inflation factor.  Rows
            # whose period isn't in the dict keep a unity divisor.
            periods = df.index.get_level_values("period")
            divisors = pd.Series(
                [inflation.get(str(p), 1.0) for p in periods],
                index=df.index,
                dtype=float,
            )
            df = df.div(divisors, axis=0)

        # Agent 9 — un-scale row scaling on nodeBalance_eq.  Divide each
        # (period, node) cell by node_capacity_for_scaling[n, d] so the
        # nodal price returns to user-facing EUR/MWh.  Mode A: scaler = 1
        # everywhere → no effect.
        scaler = _load_row_scaler(wf, "node", solve_name, flex_data=flex_data)
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
            # dual / node_cap  ⇒  element-wise divide
            df = df.div(mult)

    path = output_dir / f"v_dual_node_balance__{solve_name}.parquet"
    write_lean_parquet(df, path)
    _logger.debug(
        "Wrote v_dual_node_balance for solve '%s' -> %s (shape %s)",
        solve_name, path, df.shape,
    )
    return path


def write_v_dual_reserve_balance(
    h: "highspy.Highs",
    *,
    solve_name: str,
    output_dir: Path | str,
    realized_dispatch_csv: Path | str | None = None,
    work_folder: Path | str | None = None,
    flex_data: "FlexData | None" = None,
) -> Path:
    """Write ``v_dual_reserve__upDown__group__period__t__{solve}.parquet``.

    Model formula takes ``max()`` over up to three constraint duals per
    ``(r, ud, g, r_m, d, t)``:

        * ``reserveBalance_timeseries_eq`` (most common method)
        * ``reserveBalance_dynamic_eq``
        * ``reserveBalance_up_n_1_eq`` / ``reserveBalance_down_n_1_eq``
          (N-1 security)

    applied after period scaling
    ``× complete_period_share_of_year[d] / p_inflation_factor_operations_yearly[d]``.

    SIMPLIFIED HERE: currently extracts only the timeseries-equation
    duals.  For ``(r, ud, g)`` that use the ``timeseries_only`` method
    (the common case covered by all the test scenarios) this is exact.
    Groups using ``dynamic`` or ``n_1`` methods will be under-reported
    until the full ``max()`` logic is ported.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    wf = Path(work_folder) if work_folder is not None else output_dir.parent
    out_path = output_dir / (
        f"v_dual_reserve__upDown__group__period__t__{solve_name}.parquet"
    )

    # reserveBalance_timeseries_eq indices: r, ud, g, d, t (5).  The
    # method is implicit in the constraint name prefix — there is no
    # ``r_m`` column inside the brackets.  Sibling constraints
    # ``reserveBalance_dynamic_eq`` / ``reserveBalance_*_n_1_eq`` carry
    # the same (r, ud, g, d, t) axes; the ``max() over methods`` combine
    # documented above is a future extension.
    df = extract_variable(
        h, "reserveBalance_timeseries_eq",
        ("reserve", "updown", "node_group"),
        solve_name=solve_name, has_time=True, source="row_dual",
        realized_dispatch_csv=realized_dispatch_csv,
        flex_data=flex_data,
    )

    if df.empty:
        write_lean_parquet(df, out_path)
        _logger.debug(
            "Wrote v_dual_reserve_balance for solve '%s' -> %s (empty)",
            solve_name, out_path,
        )
        return out_path

    # Apply × period_share / inflation per period.
    inflation = _load_inflation_factor(wf, flex_data=flex_data)
    period_share = _load_complete_period_share_of_year(wf, flex_data=flex_data)
    periods = df.index.get_level_values("period")
    factor = pd.Series(
        [
            period_share.get(str(p), 1.0) / inflation.get(str(p), 1.0)
            for p in periods
        ],
        index=df.index, dtype=float,
    )
    df = df.mul(factor, axis=0)

    write_lean_parquet(df, out_path)
    _logger.debug(
        "Wrote v_dual_reserve_balance for solve '%s' -> %s (shape %s)",
        solve_name, out_path, df.shape,
    )
    return out_path


def _is_first_solve_from_p_model(work_folder: Path) -> bool:
    """True iff ``solve_data/p_model.csv`` says ``solveFirst`` is 1 (or missing)."""
    path = work_folder / "solve_data" / "p_model.csv"
    if not path.exists():
        return True
    df = pd.read_csv(path)
    matches = df.loc[df["modelParam"] == "solveFirst", "p_model"]
    if matches.empty:
        return True
    return bool(int(matches.iloc[0]))


def _actual_solve_name(work_folder: Path, fallback: str,
                        *, provider: "object | None" = None) -> str:
    """Return the roll-level solve name from ``solve_data/solve_current.csv``.

    In rolling-window scenarios, ``solver.run(complete_solve[solve])`` is
    invoked with the PARENT solve name (the ``complete_solve``) while
    every phase-1 CSV in ``solve_data/`` stores the child ROLL name
    under its ``solve`` column (the set ``solve_current`` in the model).
    When filtering those CSVs or emitting CSV rows whose format the model
    produced, use the ROLL name — otherwise Python output won't line up
    with phase 3's.
    """
    path = work_folder / "solve_data" / "solve_current.csv"
    # Step 1-e — Provider-aware: under the in-memory cascade the file
    # isn't on disk, but the per-sub-solve Provider has the frame.  The
    # transitional seed-funnel fallback in :func:`_provider_lookup`
    # keeps unplumbed callers working during the dual-write window.
    seeded = _provider_lookup(provider, path)
    if seeded is not None:
        if seeded.height == 0 or len(seeded.columns) == 0:
            return fallback
        return str(seeded[0, 0])
    if not path.exists():
        return fallback
    df = pd.read_csv(path)
    if df.empty or len(df.columns) == 0:
        return fallback
    return str(df.iloc[0, 0])




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
) -> list[Path]:
    """Iterate :data:`VARIABLE_SPECS` (or a custom list) and write parquets.

    Returns the list of paths written, one per variable.  Each variable
    is independent — failure on one is logged and does not abort the
    remaining ones, so one bad variable can't lose the whole run.
    """
    specs = specs if specs is not None else VARIABLE_SPECS
    written: list[Path] = []
    # Derive the work folder from the realized-dispatch CSV (preferred)
    # so custom writers (esp. write_v_obj) can find
    # ``solve_data/scale_the_objective.csv`` at the live path.
    _derived_wf: Path | None = (
        Path(realized_dispatch_csv).parent.parent
        if realized_dispatch_csv is not None
        else (
            Path(realized_periods_csv).parent.parent
            if realized_periods_csv is not None
            else None
        )
    )
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
            col_dual = _sol.col_dual
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
            work_folder=_derived_wf,
            flex_data=flex_data,
            scale_the_objective=scale_the_objective,
        )),
        ("v_dual_invest_{unit,connection,node}", lambda: write_v_dual_invest_by_class(
            h, solve_name=solve_name, output_dir=output_dir,
            realized_periods_csv=realized_periods_csv,
            flex_data=flex_data,
            scale_the_objective=scale_the_objective,
        )),
        ("v_dual_node_balance", lambda: write_v_dual_node_balance(
            h, solve_name=solve_name, output_dir=output_dir,
            realized_dispatch_csv=realized_dispatch_csv,
            flex_data=flex_data,
            scale_the_objective=scale_the_objective,
        )),
        ("v_dual_reserve_balance", lambda: write_v_dual_reserve_balance(
            h, solve_name=solve_name, output_dir=output_dir,
            realized_dispatch_csv=realized_dispatch_csv,
            flex_data=flex_data,
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
            else:
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


# ---------------------------------------------------------------------------
# Standalone / offline test entry point
# ---------------------------------------------------------------------------


def _standalone_from_files(
    mps_file: Path, sol_file: Path, solve_name: str, out_dir: Path
) -> list[Path]:
    """Re-read a HiGHS model + solution offline and write all registered vars."""
    import highspy  # local import — the runtime path imports lazily

    h = highspy.Highs()
    status = h.readModel(str(mps_file))
    if status != highspy.HighsStatus.kOk:
        raise RuntimeError(f"HiGHS failed to read MPS: {mps_file}")
    status = h.readSolution(str(sol_file), 0)
    if status != highspy.HighsStatus.kOk:
        raise RuntimeError(
            f"HiGHS failed to read solution file: {sol_file} "
            f"(status={status}).  Try re-solving instead."
        )
    return write_all_variables(h, solve_name=solve_name, output_dir=out_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--mps", type=Path, required=True, help="MPS file")
    parser.add_argument(
        "--sol", type=Path, required=True,
        help="HiGHS solution file (write_solution_style=0)",
    )
    parser.add_argument("--solve", required=True, help="Solve name tag")
    parser.add_argument("--out", type=Path, required=True, help="Output dir")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    paths = _standalone_from_files(args.mps, args.sol, args.solve, args.out)
    for p in paths:
        print(f"Wrote {p}")


if __name__ == "__main__":
    main()
