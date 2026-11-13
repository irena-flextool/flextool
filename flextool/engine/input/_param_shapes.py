"""Δ.17c — centralized parameter-shape resolver.

The user's authoritative directive (Δ.17c dispatch, Gap C):

    There is no guarantee that e.g. 2d_map is (period, time). There is
    multi-level reliance on different factors. The source of truth
    should be three things:
    1. read from database the dimensionality of the parameter.
    2. list of parameter specific allowed dimensionalities e.g.
       p_efficiency[period,time].
    3. If parameter has multiple choices for e.g. 1d-map like
       p_efficiency[period], p_efficiency[time], then you need to read
       the dimension index label from the database. If it says period,
       it is period; if it says time, it is time. If does not match,
       error to the user.

This module implements those three things:

* :data:`PARAM_ALLOWED_SHAPES` — per-(entity_class, parameter_name)
  registry of valid shapes.  Each entry lists the shapes preprocessing
  accepts (mirrors the ``write_pdtX`` / ``write_pdX`` cascade domains;
  see docstring on each entry).
* :func:`resolve_param_shape` — DB-driven shape detection: reads the
  parameter's actual nesting depth and per-level index_name labels
  from the DB via :meth:`InputSource.parameter_shape_info`, validates
  against the allow-list, raises :class:`FlexToolConfigError` on
  mismatch.
* :func:`broadcast_to_period_time` — produces ``[entity, d, t, value]``
  by broadcasting the parameter frame over the active solve's
  ``(d, t)`` axis according to the resolved shape.
* :func:`broadcast_to_period` — produces ``[entity, d, value]`` for
  parameters whose allowed shapes are scalar / 1d_map[period] only
  (e.g. ``co2_max_period``).

Why a centralized resolver and not per-helper column checks?
The previous Δ.17b approach inferred the shape from the column names
of the source frame.  That's flawed when:

* A 2d_map column ordering depends on which dim was outermost in the
  Map — *e.g.* ``2d_map[time, period]`` writes a column ``[t, period]``
  but the helper expected ``[period, t]``.  Column-shape detection
  silently produced the wrong broadcast.
* The DB carries an unexpected (non-period, non-time) index_name —
  e.g. ``branch`` — and the column was renamed to the canonical
  default ``period`` by :class:`SpineDbReader._discover_index_cols`.
  The helper saw a "period" column and broadcast over (d, t) when the
  authoring intent was different.

The data-driven resolver avoids both: it inspects the DB-reported
``index_name`` per level and validates against the per-parameter
allow-list.  Mismatches surface as :class:`FlexToolConfigError`.

Computational efficiency
------------------------

:func:`resolve_param_shape` is called once per (entity_class,
parameter_name) per ``apply_direct_params`` invocation — at most ~10
calls per ``load_flextool``, all cached at the source-plugin level.
The resolver itself does no per-row work; it only inspects the
parameter's structural metadata.

Stochastic branch periods
-------------------------

A branch period (``period1_upper``, ``p2035_high``, …) is minted per solve, so no input value is
keyed by it.  Both broadcasters first copy the period-keyed rows of an anchor period to its branch
periods (from ``period__branch``, restricted to the solve's in-use periods), unless the branch period
authors a row of its own for that entity: a period map ``{anchor: v}`` then gives the branch periods
the same rows as the scalar ``v``.  The step is active only inside ``load_flextool`` (which arms the
pairs when it reads the DB); a direct call outside it returns the authored periods only.

Stochastic time
---------------

A stochastic-time value is a three-level map ``branch → analysis_time → time`` (labelled, or with Spine's
silent ``x`` labels).  For the parameters of :data:`STOCHASTIC_TIME_PARAMS` the resolver takes the entities
authoring one out of its view (the period / time classifier never sees them) and :func:`broadcast_to_period_time`
gives them the values the solve's emit folded from ``pbt_*`` onto its ``(d, t)``; every other entity is read as
before.  The emit frames are armed by ``load_flextool`` (:func:`activate_stochastic_time_source`).

Solve-step aggregation
----------------------

In a solve with ``new_stepduration`` a solve step combines several time steps of the timeline.
:func:`broadcast_to_period_time` gives the time-keyed rows one value per solve step, by the parameter's kind in
``_solve_step_aggregation.TIME_AGGREGATION`` (mean, harmonic mean, last) through
``_solve_step_aggregation.aggregate_if_active`` (keyed by :attr:`ResolvedShape.key`), before the zero filter and
the axis cast.  The base steps are armed per load by ``load_flextool``; without ``new_stepduration`` nothing runs.
"""
from __future__ import annotations

import contextvars
import dataclasses
import logging
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

import polars as pl

from polar_high import Param

from flextool.engine.input._axis_enums import (
    cast_frame_axes,
    get_global_axis_enums,
    rename_to_axis,
)
from flextool.engine.input._solve_state import FlexToolConfigError
from flextool.engine.input._solve_step_aggregation import active_base_steps, aggregate_if_active
from flextool.spinedb_backend._axis_enums import FlexDataIntegrityError

_LOGGER = logging.getLogger(__name__)

if TYPE_CHECKING:
    from flextool.engine.input._input_source import InputSource


# ---------------------------------------------------------------------------
# Shape enum + per-parameter allow-list
# ---------------------------------------------------------------------------


class AxisName(Enum):
    """Canonical axis identifiers for parameter-shape declarations.

    Sourced from ``flextool/schemas/flextool_axis_contract.json`` axes —
    each member's value is the axis's `name` in that contract.  Phase A
    introduces only the axes that participate in declared parameter
    shapes today (period ``d``, time ``t``, tier ``i``).  Adding
    ``block`` / ``branch`` / etc. is a follow-up when the registry
    grows entries that use them.

    Note: the model-internal ``ROLL`` axis is intentionally NOT included
    — rolling boundaries are an engine concern, never authored as a
    parameter index in input/output data.
    """

    D = "d"     # period
    T = "t"     # time
    I = "i"     # tier (price-ladder)  # noqa: E741 — axis name from contract


class LeafKind(Enum):
    """Leaf value kind for parameter shape declarations.

    Phase A introduces three kinds:

    * ``NUMERIC`` — the historical default; a scalar Float (or Int that
      casts cleanly).  Every pre-existing :class:`Shape` member declares
      this leaf kind.
    * ``STR_FROM_LIST`` — a scalar string constrained by a
      ``parameter_value_list`` (e.g. ``invest_methods``).  The reader
      writes whichever token the DB carries; the registry just records
      that this is a constrained-string leaf.
    * ``FACET_PRICE_QUANTITY`` — a depth-1 ``Map(facet -> value)`` whose
      facet keys are exactly ``{"price", "quantity"}``.  Used by the
      commodity price-ladder shapes — Spine encodes the per-tier
      ``{price, quantity}`` record as a third Map level.

    Phase A consumers only need the leaf kind for documentation and
    Phase B reader-routing.  ``resolve_param_shape`` accepts the new
    leaf-kind shapes structurally; the reader/writer wiring is Phase B.
    """

    NUMERIC = "numeric"
    STR_FROM_LIST = "str_from_list"
    FACET_PRICE_QUANTITY = "facet[price,quantity]"


# Frozen facet sets — kept here so ``ParamAxes`` and any Phase B reader
# can share a single source of truth without re-importing across modules.
_FACET_PRICE_QUANTITY: frozenset[str] = frozenset(("price", "quantity"))


class Shape(Enum):
    """Recognised parameter shapes.

    The string values mirror the user's notation in the dispatch /
    open-issues doc (e.g. ``"1d_map[period]"``).  Each member maps to a
    ``ParamAxes(map_levels, leaf)`` view via :data:`_SHAPE_AXES` for
    callers that prefer the (axes, leaf) decomposition over the enum tag.

    The original (period, time)-broadcast family — ``SCALAR``,
    ``MAP_PERIOD``, ``MAP_TIME``, ``MAP_PERIOD_TIME``, ``MAP_TIME_PERIOD``
    — covers every entry routed through ``broadcast_to_period[_time]``
    today.  Phase A adds three new members:

    * ``SCALAR_STR`` — terminal scalar string (no map levels), value
      constrained by a ``parameter_value_list``.  Used by the
      ``{node, unit, connection}.invest_method`` family.
    * ``MAP_TIER_FACET`` — depth-2 Map ``Map(tier -> {price, quantity})``
      via :data:`LeafKind.FACET_PRICE_QUANTITY`.  Used by
      ``commodity.price_ladder_cumulative`` and the depth-2 variant of
      ``commodity.price_ladder_annual``.
    * ``MAP_PERIOD_TIER_FACET`` — depth-3 Map
      ``Map(period -> Map(tier -> {price, quantity}))``.  Used by the
      depth-3 variant of ``commodity.price_ladder_annual``.

    The Phase A choice was (a) — extend this enum in-place — because the
    existing ``Shape`` use sites (3 test files + ``_direct.py``
    imports only the resolver / broadcasters, not the enum members
    directly) confine the blast radius to this module.  A natural
    follow-up (option b in the Phase A design doc) is to make ``Shape``
    a derived view over ``ParamAxes(map_levels, leaf)`` and let
    registry entries quote axis tuples directly — useful when the
    contract grows axes like ``block`` / ``branch``.  For now the
    enum-extension keeps the diff minimal.
    """

    SCALAR = "scalar"
    SCALAR_STR = "scalar[str_from_list]"
    MAP_PERIOD = "1d_map[period]"
    MAP_TIME = "1d_map[time]"
    MAP_PERIOD_TIME = "2d_map[period,time]"
    MAP_TIME_PERIOD = "2d_map[time,period]"
    MAP_TIER_FACET = "2d_map[tier,facet{price,quantity}]"
    MAP_PERIOD_TIER_FACET = "3d_map[period,tier,facet{price,quantity}]"


@dataclass(frozen=True)
class ParamAxes:
    """Decomposed (axes, leaf) view of a :class:`Shape`.

    ``map_levels`` are the outer Map index axes from outer to inner; for
    a ``FACET_PRICE_QUANTITY`` leaf the trailing facet level is NOT
    included in ``map_levels`` — it's carried in ``leaf`` instead, so
    the axis tuple stays a pure list of named axes from
    :class:`AxisName`.

    Example: ``Shape.MAP_PERIOD_TIER_FACET`` decomposes to
    ``ParamAxes(map_levels=(AxisName.D, AxisName.I),
                leaf=LeafKind.FACET_PRICE_QUANTITY)``.

    Phase A does not consume ``ParamAxes`` from the resolver — it's a
    pure documentation/derived view, accessible via
    :func:`shape_to_axes`.  Phase B will use it to route the new
    parameter families through reader-side shape probing.
    """

    map_levels: tuple[AxisName, ...]
    leaf: LeafKind


# Per-(entity_class, parameter_name) allow-list.  Each entry lists the
# shapes preprocessing accepts for that parameter.
#
# The cascade in `write_pdtX` (mod L1227 etc.) is:
#     pbt → pd → pt → p → def1 → 0
# meaning: 2d_map(period, time) preferred, then 1d_map(period), then
# 1d_map(time), then scalar.  All four shapes are valid authoring options
# for parameters in the relevant ``X_TIME_PARAM`` taxonomy.
#
# `write_pdX` cascade (mod L1115) is:
#     pd → period__branch fold → p → 5000-default-set → 0
# meaning: 1d_map(period) or scalar, no time variant.  Used for
# parameters in ``X_PERIOD_PARAM`` but NOT ``X_TIME_PARAM``.
#
# `write_pdtCommodity` cascade is pt → pd → p → 0 (no pbt).
PARAM_ALLOWED_SHAPES: dict[tuple[str, str], set[Shape]] = {
    # ─── group: (period, time) Params ────────────────────────────────────
    # GROUP_TIME_PARAM = {co2_price, max_instant_flow, min_instant_flow}.
    # Cascade: pbt → pd → pt → p (entity_period_calc_params:write_pdtGroup).
    ("group", "co2_price"): {
        Shape.SCALAR, Shape.MAP_PERIOD, Shape.MAP_TIME, Shape.MAP_PERIOD_TIME,
    },
    # The inertia and non-synchronous limits apply per time step: constant, period, time or period + time
    # (read by broadcast_to_period_time; the Param keeps the authored dims ⊆ (g, d, t)).
    ("group", "inertia_limit"): {
        Shape.SCALAR, Shape.MAP_PERIOD, Shape.MAP_TIME, Shape.MAP_PERIOD_TIME,
    },
    ("group", "non_synchronous_limit"): {
        Shape.SCALAR, Shape.MAP_PERIOD, Shape.MAP_TIME, Shape.MAP_PERIOD_TIME,
    },
    # v58: the four flow-limit params live on the ``flowGroup`` class.
    ("flowGroup", "max_instant_flow"): {
        Shape.SCALAR, Shape.MAP_PERIOD, Shape.MAP_TIME, Shape.MAP_PERIOD_TIME,
    },
    ("flowGroup", "min_instant_flow"): {
        Shape.SCALAR, Shape.MAP_PERIOD, Shape.MAP_TIME, Shape.MAP_PERIOD_TIME,
    },
    # ─── group: period-only Params (no time variant) ─────────────────────
    # GROUP_PERIOD_PARAM \ GROUP_TIME_PARAM ⊇ {co2_max_period}.
    # Cascade: pd → p (entity_period_calc_params:write_pdGroup).
    ("group", "co2_max_period"): {Shape.SCALAR, Shape.MAP_PERIOD},
    # Tier-1 silent-default migration (pdGroup_* / p_group_* 1d_map[period]).
    # Each parameter was previously routed through
    # ``_entity_period_scalar`` and so cross-joined silent-default Maps
    # (index_name="x") against the period filter, producing duplicate
    # (g, d) rows.  Migrating to ``resolve_param_shape`` +
    # ``broadcast_to_period`` carries the silent-default disambiguation
    # via ``_infer_silent_default_labels`` (allow-list {SCALAR,
    # MAP_PERIOD} ⇒ depth-1 silent default unambiguously resolves to
    # ``period``).  Cascade: pd → p (entity_period_calc_params:write_pdGroup
    # — same as ``co2_max_period``).
    ("group", "capacity_margin"): {Shape.SCALAR, Shape.MAP_PERIOD},
    ("group", "capacity_margin_share"): {Shape.SCALAR, Shape.MAP_PERIOD},
    ("group", "penalty_capacity_margin"): {Shape.SCALAR, Shape.MAP_PERIOD},
    ("group", "penalty_inertia"): {Shape.SCALAR, Shape.MAP_PERIOD},
    ("group", "penalty_non_synchronous"): {Shape.SCALAR, Shape.MAP_PERIOD},
    ("group", "invest_max_period"): {Shape.SCALAR, Shape.MAP_PERIOD},
    ("group", "invest_min_period"): {Shape.SCALAR, Shape.MAP_PERIOD},
    ("group", "retire_max_period"): {Shape.SCALAR, Shape.MAP_PERIOD},
    ("group", "retire_min_period"): {Shape.SCALAR, Shape.MAP_PERIOD},
    # group.cumulative_{max,min}_capacity — schema "Constant or period", same shapes as
    # the entity-level cumulative_max_capacity / cumulative_min_capacity.
    ("group", "cumulative_max_capacity"): {Shape.SCALAR, Shape.MAP_PERIOD},
    ("group", "cumulative_min_capacity"): {Shape.SCALAR, Shape.MAP_PERIOD},
    ("flowGroup", "max_cumulative_flow"): {Shape.SCALAR, Shape.MAP_PERIOD},
    ("flowGroup", "min_cumulative_flow"): {Shape.SCALAR, Shape.MAP_PERIOD},
    # ─── unit: period-only Params (no time variant) ──────────────────────
    # UC startup cost — Spine schema declares ``unit.startup_cost`` as
    # scalar OR 1d_map(period).  Cascade: pd → p (write_pUnit / startup
    # cost wiring in ``_load_online``).
    # User-constraint right-hand side: constant, or a period map that covers every period of the solve.
    ("constraint", "constant"): {Shape.SCALAR, Shape.MAP_PERIOD},
    ("unit", "startup_cost"): {Shape.SCALAR, Shape.MAP_PERIOD},
    # Connection startup cost (connections with a startup_method): same shapes as the unit's.
    ("connection", "startup_cost"): {Shape.SCALAR, Shape.MAP_PERIOD},
    # Start-up input energy [MWh per MW started] of a unit__inputNode arc with a startup_input_method other than
    # just_input: constant or period, as startup_cost (read by params/_direct.p_startup_energy_from_source).
    ("unit__inputNode", "startup_energy"): {Shape.SCALAR, Shape.MAP_PERIOD},
    # Tier-2 silent-default migration (ed_* multi-class union 1d_map(period)).
    # These six parameters are declared on each of unit/node/connection and
    # are unioned in :func:`_e_period_param_union` into a single
    # ``Param(("e", "d"))`` frame.  Pre-migration the helper had a partial
    # ``"x" → "period"`` workaround but silently dropped scalar authoring
    # and didn't value-domain probe.  Routing each class through
    # ``resolve_param_shape`` + ``broadcast_to_period`` fixes both: the
    # registry's {SCALAR, MAP_PERIOD} allow-list resolves silent-default
    # ``index_name`` at depth-1 structurally, and SCALAR rows are
    # broadcast across the active solve's periods instead of being
    # dropped.  Cascade: pd → p (write_ed* via apply_direct_params_b).
    ("unit",       "invest_max_period"): {Shape.SCALAR, Shape.MAP_PERIOD},
    ("node",       "invest_max_period"): {Shape.SCALAR, Shape.MAP_PERIOD},
    ("connection", "invest_max_period"): {Shape.SCALAR, Shape.MAP_PERIOD},
    ("unit",       "invest_min_period"): {Shape.SCALAR, Shape.MAP_PERIOD},
    ("node",       "invest_min_period"): {Shape.SCALAR, Shape.MAP_PERIOD},
    ("connection", "invest_min_period"): {Shape.SCALAR, Shape.MAP_PERIOD},
    ("unit",       "retire_max_period"): {Shape.SCALAR, Shape.MAP_PERIOD},
    ("node",       "retire_max_period"): {Shape.SCALAR, Shape.MAP_PERIOD},
    ("connection", "retire_max_period"): {Shape.SCALAR, Shape.MAP_PERIOD},
    ("unit",       "retire_min_period"): {Shape.SCALAR, Shape.MAP_PERIOD},
    ("node",       "retire_min_period"): {Shape.SCALAR, Shape.MAP_PERIOD},
    ("connection", "retire_min_period"): {Shape.SCALAR, Shape.MAP_PERIOD},
    ("unit",       "cumulative_max_capacity"): {Shape.SCALAR, Shape.MAP_PERIOD},
    ("node",       "cumulative_max_capacity"): {Shape.SCALAR, Shape.MAP_PERIOD},
    ("connection", "cumulative_max_capacity"): {Shape.SCALAR, Shape.MAP_PERIOD},
    ("unit",       "cumulative_min_capacity"): {Shape.SCALAR, Shape.MAP_PERIOD},
    ("node",       "cumulative_min_capacity"): {Shape.SCALAR, Shape.MAP_PERIOD},
    ("connection", "cumulative_min_capacity"): {Shape.SCALAR, Shape.MAP_PERIOD},
    # Investment cost / salvage value: constant or 1d_map(period) per the docs; read through
    # _e_period_param_union (the annuities, and the notice of investment offered without a cost).
    ("unit",       "invest_cost"):   {Shape.SCALAR, Shape.MAP_PERIOD},
    ("node",       "invest_cost"):   {Shape.SCALAR, Shape.MAP_PERIOD},
    ("connection", "invest_cost"):   {Shape.SCALAR, Shape.MAP_PERIOD},
    ("unit",       "salvage_value"): {Shape.SCALAR, Shape.MAP_PERIOD},
    ("node",       "salvage_value"): {Shape.SCALAR, Shape.MAP_PERIOD},
    ("connection", "salvage_value"): {Shape.SCALAR, Shape.MAP_PERIOD},
    # ─── node: (period, time) Params ─────────────────────────────────────
    # NODE_TIME_PARAM ⊇ {availability, storage_state_reference_value, self_discharge_loss}.
    # Cascade: pbt → pt → pd → p (write_pdtNode, time_first_priority=True).
    ("node", "availability"): {
        Shape.SCALAR, Shape.MAP_PERIOD, Shape.MAP_TIME, Shape.MAP_PERIOD_TIME,
    },
    ("node", "storage_state_reference_value"): {
        Shape.SCALAR, Shape.MAP_PERIOD, Shape.MAP_TIME, Shape.MAP_PERIOD_TIME,
    },
    # Additive energy margin [MWh per time step] of the investment solve: a constant, a period map or a period + time
    # map (ingested by input_derivation/_specs.py, applied by emit/_energy_margin_adder.py).  A time-only map has
    # no ingestion spec and would be silently ignored: the allow-list refuses it
    # (params/_direct.check_energy_margin_adder_shape).
    ("node", "energy_margin_adder"): {
        Shape.SCALAR, Shape.MAP_PERIOD, Shape.MAP_PERIOD_TIME,
    },
    # Self-discharge loss per hour (schema: "Constant or time"; GMPL NODE_TIME_PARAM).  Consumer: the per-step
    # loss weights of model/_self_discharge.py.
    ("node", "self_discharge_loss"): {
        Shape.SCALAR, Shape.MAP_PERIOD, Shape.MAP_TIME, Shape.MAP_PERIOD_TIME,
    },
    # Balance slack penalties.  Cascade: pbt → pt → pd → p → class default (write_pdtNode).  Rank 3
    # (stochastic time: branch, analysis_time, time) is not a resolver shape: an entity authoring it is
    # read from the solve's emit fold (STOCHASTIC_TIME_PARAMS).
    ("node", "penalty_up"): {
        Shape.SCALAR, Shape.MAP_PERIOD, Shape.MAP_TIME, Shape.MAP_PERIOD_TIME,
    },
    ("node", "penalty_down"): {
        Shape.SCALAR, Shape.MAP_PERIOD, Shape.MAP_TIME, Shape.MAP_PERIOD_TIME,
    },
    # ─── unit / connection: (period, time) Params ────────────────────────
    # PROCESS_TIME_PARAM ⊇ {availability}.
    # Cascade: pbt → pd → pt → p (write_pdtProcess).  ``processes`` in
    # process_arc_unions.py:write_param_in_use_sets reads the ``process.csv``
    # union of unit + connection, so both classes participate in the LP's
    # ``p_process_availability`` (the Spine schema declares the parameter
    # on each class independently).
    ("unit", "availability"): {
        Shape.SCALAR, Shape.MAP_PERIOD, Shape.MAP_TIME, Shape.MAP_PERIOD_TIME,
    },
    ("connection", "availability"): {
        Shape.SCALAR, Shape.MAP_PERIOD, Shape.MAP_TIME, Shape.MAP_PERIOD_TIME,
    },
    # ─── unit / connection: efficiency-curve (period, time) Params ────────
    # ``efficiency``, ``efficiency_at_min_load`` and ``min_load`` are all
    # members of ``_PROCESS_TIME_PARAM``
    # (_arc_unions.py:146-149 — flextool_base.dat:153), so each admits
    # the full (period, time) shape family exactly like ``availability``.
    # Consumed by :func:`._derived.p_slope_from_source` /
    # :func:`._derived.p_section_from_source` via
    # :func:`._derived._eff_param_to_pdt`.  ``efficiency`` is
    # authored on both unit and connection; ``efficiency_at_min_load`` /
    # ``min_load`` only feed the unit ``min_load_efficiency`` linearisation.
    ("unit", "efficiency"): {
        Shape.SCALAR, Shape.MAP_PERIOD, Shape.MAP_TIME, Shape.MAP_PERIOD_TIME,
    },
    ("connection", "efficiency"): {
        Shape.SCALAR, Shape.MAP_PERIOD, Shape.MAP_TIME, Shape.MAP_PERIOD_TIME,
    },
    ("unit", "efficiency_at_min_load"): {
        Shape.SCALAR, Shape.MAP_PERIOD, Shape.MAP_TIME, Shape.MAP_PERIOD_TIME,
    },
    ("unit", "min_load"): {
        Shape.SCALAR, Shape.MAP_PERIOD, Shape.MAP_TIME, Shape.MAP_PERIOD_TIME,
    },
    # ─── commodity: time Params ──────────────────────────────────────────
    # commodityTimeParam = {price}.
    # Cascade: pt → pd → p → 0 (write_pdtCommodity, no pbt).
    ("commodity", "price"): {
        Shape.SCALAR, Shape.MAP_PERIOD, Shape.MAP_TIME,
    },
    # ─── Tier-3 silent-default migration ─────────────────────────────────
    # Map(period → time) variable-cost / reservation parameters previously
    # routed through inline helpers in ``_direct.py`` that gated on
    # ``{"period", "t"}.issubset(cols)`` — silently dropping silent-default
    # Maps whose ``index_name`` carried spinedb_api's ``"x" / "" / None``
    # instead of the canonical ``"period" / "t"``.  Routing through
    # ``resolve_param_shape`` + ``broadcast_to_period_time`` carries the
    # silent-default disambiguation structurally (allow-list
    # {SCALAR, MAP_PERIOD, MAP_TIME, MAP_PERIOD_TIME} ⇒ value-domain
    # probing for ambiguous depth-1 cases; depth-2 ambiguity falls back
    # to a Map(period → time) reading).
    #
    # Three are on relationship classes (multi-dim entity columns) — the
    # Tier-3 ``broadcast_to_period_time`` extension accepts a per-source-
    # column rename dict to map those into the LP's canonical
    # ``(p / source / sink / r, ud, g)`` axes.
    ("unit__inputNode",        "other_operational_cost"): {
        Shape.SCALAR, Shape.MAP_PERIOD, Shape.MAP_TIME, Shape.MAP_PERIOD_TIME,
    },
    ("unit__outputNode",       "other_operational_cost"): {
        Shape.SCALAR, Shape.MAP_PERIOD, Shape.MAP_TIME, Shape.MAP_PERIOD_TIME,
    },
    ("connection",             "other_operational_cost"): {
        Shape.SCALAR, Shape.MAP_PERIOD, Shape.MAP_TIME, Shape.MAP_PERIOD_TIME,
    },
    ("reserve__upDown__group", "reservation"): {
        Shape.SCALAR, Shape.MAP_PERIOD, Shape.MAP_TIME, Shape.MAP_PERIOD_TIME,
    },
    # ─── Phase A: commodity price-ladder Maps (tier-keyed, facet leaf) ───
    # ``commodity.price_ladder_cumulative`` — depth-2 Map keyed by ``tier``
    # with a facet-dict leaf ``{price, quantity}``.  Per the schema
    # description (spinedb_schema.json L874) and the input-derivation
    # contract (_commodity_ladder.py:151-156), this is the only valid
    # authoring shape — period-agnostic, one row per tier.  Spine's
    # ``parameter_value_types`` block locks this at ``("commodity",
    # "price_ladder_cumulative", "map", 2)``.
    #
    # ``commodity.price_ladder_annual`` — TWO valid variants per the
    # schema description (L869): a depth-2 form identical to cumulative
    # (the same ``Map(tier -> {price, quantity})`` is expanded across
    # every model period), and a depth-3 form
    # ``Map(period -> Map(tier -> {price, quantity}))`` that varies the
    # cap per period.  Spine's ``parameter_value_types`` declares both:
    # ``("commodity", "price_ladder_annual", "map", 2)`` and
    # ``("commodity", "price_ladder_annual", "map", 3)`` — matching the
    # auto-detect branch in :func:`derive_commodity_ladder_annual`
    # (input_derivation/_commodity_ladder.py:237-243).
    #
    # Phase A is contract-only.  Phase B will wire these through the
    # readers/writers; today the entries just unblock
    # :func:`resolve_param_shape` from raising on the registry probe.
    ("commodity", "price_ladder_cumulative"): {Shape.MAP_TIER_FACET},
    ("commodity", "price_ladder_annual"): {
        Shape.MAP_TIER_FACET, Shape.MAP_PERIOD_TIER_FACET,
    },
    # ─── Phase A: scalar string constrained by parameter_value_list ──────
    # ``{node, unit, connection}.invest_method`` — terminal scalar string
    # whose value is constrained by the ``invest_methods`` value list
    # (spinedb_schema.json L358-414 enumerate the allowed tokens).
    # Spine's ``parameter_value_types`` locks each at ``("...",
    # "invest_method", "str", 0)``.
    #
    # Phase A registers these as :class:`Shape.SCALAR_STR` so the
    # registry can express that the leaf is a constrained string, not a
    # numeric.  Phase B will wire the readers to honour the leaf-kind
    # tag — today these parameters are read by feature-gate logic that
    # bypasses :func:`resolve_param_shape`.
    ("node",       "invest_method"): {Shape.SCALAR_STR},
    ("unit",       "invest_method"): {Shape.SCALAR_STR},
    ("connection", "invest_method"): {Shape.SCALAR_STR},
    # ─── timeset: per-timestep weight map ────────────────────────────────
    # ``timeset.timeset_weights`` — a 1-D Map timestep -> weight (schema: map rank 1).  The index is
    # ``time`` (or ``t``), or Spine's silent ``x`` (unique at depth 1, so it resolves to ``time``); any
    # other shape (scalar, a period map, deeper maps) is an authoring error.  Read by
    # ``_derived.p_timestep_weight_from_source`` and validated for the writer by
    # :func:`check_index_name_path` (``_timeline.TimelineConfig.load_from_db``).
    ("timeset", "timeset_weights"): {Shape.MAP_TIME},
    # ─── solve: years each period represents ────────────────────────────
    # ``solve.years_represented`` — a 1-D Map period -> years (schema: map rank 1).  The index is ``period``, or
    # Spine's silent ``x`` (the only shape, so it resolves to ``period``).  Read through
    # :func:`solve_years_represented` by the investment annuities, the period start years and the inflation factors
    # (``params/_derived.py``, ``params/_existing.py``); the operations side reads the SolveConfig-written
    # ``p_years_represented``.  Before this entry the readers tested for a ``period`` column, so a map without an
    # index name gave investments one year per period.
    ("solve", "years_represented"): {Shape.MAP_PERIOD},
}


# ---------------------------------------------------------------------------
# PERIOD_TIME_PARAMS — keys that admit a period/time Map shape
# ---------------------------------------------------------------------------
#
# Spec ``fix_map_reading.md`` §3.1.  Every ``PARAM_ALLOWED_SHAPES`` key
# whose allow-list contains at least one of ``MAP_PERIOD`` / ``MAP_TIME``
# / ``MAP_PERIOD_TIME`` — the 16 dual-axis keys PLUS the 38 period-only
# keys ({SCALAR, MAP_PERIOD}).  Period-only keys are included on purpose:
# classifying their rows is what lets the §3.3 per-variant allow-list
# check catch a stray ``Map(time)`` and raise (today silently dropped).
#
# Derived from the registry so the two cannot drift.  Uses the three
# EXACT Shape enum members — NOT substring matching — so facet shapes
# such as ``MAP_PERIOD_TIER_FACET`` are correctly EXCLUDED.
_PERIOD_TIME_SHAPE_MEMBERS: "frozenset[Shape]" = frozenset(
    (Shape.MAP_PERIOD, Shape.MAP_TIME, Shape.MAP_PERIOD_TIME)
)
PERIOD_TIME_PARAMS: "frozenset[tuple[str, str]]" = frozenset(
    key
    for key, shapes in PARAM_ALLOWED_SHAPES.items()
    if _PERIOD_TIME_SHAPE_MEMBERS & set(shapes)
)


# ---------------------------------------------------------------------------
# STOCHASTIC_TIME_PARAMS — period/time parameters that also take a stochastic-time map
# ---------------------------------------------------------------------------
#
# The PERIOD_TIME_PARAMS keys whose schema type list has map rank 3 (stochastic time: branch,
# analysis_time, time; ``spinedb_schema.json`` ``parameter_types``, docs/reference/ "stochastic time").
# An entity authoring such a map gets its values from the per-solve emit, which folds ``pbt_*`` onto the
# solve's (d, t) (emit ``_pdt_params`` PdtLookup / ``_reserve_sets._resolve_pdtReserve``).  Value: (emit
# provider key, emit entity columns in class-dim order).  Pinned against the schema and the pbt specs by
# tests/engine/test_param_shapes_stochastic_time.py.
#
# Emit domain vs the DB-direct consumers' domain (an entity outside the emit domain has no folded rows, so
# the reader warns and the entity gets no value; the domain must cover every entity a consumer reads):
#   node.availability            pdtNode over node__TimeParam_in_use: nodeState nodes; consumer: the
#                                storage-state rows of nodeState nodes                      covered
#   node.penalty_up / _down      pdtNode: balance nodes; consumer: the balance-node grid      covered
#   unit.availability / efficiency
#                                pdtProcess over process_TimeParam_in_use: required for every process;
#                                consumers: all processes                                     covered
#   unit.min_load                pdtProcess: min_load_efficiency units and every online unit
#                                (emit ``_arc_unions._derive_process_time_param_in_use``); consumers:
#                                the minFlow_minload floor of the units with min_load > 0 (no value
#                                = 0), the min_load_efficiency section                       covered
#   unit__inputNode / unit__outputNode.other_operational_cost
#                                pdtProcess_source / _sink: required on every arc; consumer: every arc
#                                                                                             covered
#   reserve__upDown__group.reservation
#                                pdtReserve_upDown_group: every reserve__upDown__group row; consumer:
#                                the same rows                                                covered
# Connection parameters have no rank 3 in the schema and stay out (a labelled rank-3 connection value keeps
# the classifier's "not a valid axis" error).
STOCHASTIC_TIME_PARAMS: "dict[tuple[str, str], tuple[str, tuple[str, ...]]]" = {
    ("node", "availability"): ("solve_data/pdtNode", ("node",)),
    ("node", "penalty_up"): ("solve_data/pdtNode", ("node",)),
    ("node", "penalty_down"): ("solve_data/pdtNode", ("node",)),
    ("unit", "availability"): ("solve_data/pdtProcess", ("process",)),
    ("unit", "efficiency"): ("solve_data/pdtProcess", ("process",)),
    ("unit", "min_load"): ("solve_data/pdtProcess", ("process",)),
    ("unit__inputNode", "other_operational_cost"): ("solve_data/pdtProcess_source", ("process", "source")),
    ("unit__outputNode", "other_operational_cost"): ("solve_data/pdtProcess_sink", ("process", "sink")),
    ("reserve__upDown__group", "reservation"): ("solve_data/pdtReserve_upDown_group",
                                                ("reserve", "upDown", "group")),
}

# Accepted index_name per level of a stochastic-time map (a silent ``x`` / empty label is accepted at any
# level); any other label, or a deeper map, is an authoring error.
_STOCHASTIC_TIME_LABELS: "tuple[frozenset[str], ...]" = (
    frozenset(("branch",)),
    frozenset(("analysis_time", "time_start")),
    frozenset(("time", "t")),
)


# ---------------------------------------------------------------------------
# ParamAxes — derived (map_levels, leaf) view per :class:`Shape`
# ---------------------------------------------------------------------------


# Per-shape ``ParamAxes`` view.  Phase A consumers can read this to
# decompose a :class:`Shape` into its outer-axis tuple + leaf kind
# without inspecting the enum tag string.  Kept in sync with the
# :class:`Shape` enum — every member must have an entry here.
_SHAPE_AXES: "dict[Shape, ParamAxes]" = {
    Shape.SCALAR:               ParamAxes((), LeafKind.NUMERIC),
    Shape.SCALAR_STR:           ParamAxes((), LeafKind.STR_FROM_LIST),
    Shape.MAP_PERIOD:           ParamAxes((AxisName.D,), LeafKind.NUMERIC),
    Shape.MAP_TIME:             ParamAxes((AxisName.T,), LeafKind.NUMERIC),
    Shape.MAP_PERIOD_TIME:      ParamAxes((AxisName.D, AxisName.T),
                                          LeafKind.NUMERIC),
    Shape.MAP_TIME_PERIOD:      ParamAxes((AxisName.T, AxisName.D),
                                          LeafKind.NUMERIC),
    Shape.MAP_TIER_FACET:       ParamAxes((AxisName.I,),
                                          LeafKind.FACET_PRICE_QUANTITY),
    Shape.MAP_PERIOD_TIER_FACET: ParamAxes(
        (AxisName.D, AxisName.I), LeafKind.FACET_PRICE_QUANTITY),
}


def shape_to_axes(shape: Shape) -> ParamAxes:
    """Return the :class:`ParamAxes` view of *shape*.

    Stable read-only accessor — consumers should prefer this over
    indexing :data:`_SHAPE_AXES` directly so the module stays free to
    refactor the storage representation later (e.g. derive Shape from
    ParamAxes if option (b) in the Phase A design is adopted).
    """
    return _SHAPE_AXES[shape]


def facet_keys(leaf: LeafKind) -> "frozenset[str]":
    """Return the facet-key set for a facet-dict :class:`LeafKind`.

    Raises :class:`KeyError` when *leaf* isn't a facet leaf.  Phase B
    readers use this when projecting facet-keyed leaves into columnar
    form (one column per facet) — exposing the contract here keeps the
    facet vocabulary in a single place.
    """
    if leaf is LeafKind.FACET_PRICE_QUANTITY:
        return _FACET_PRICE_QUANTITY
    raise KeyError(
        f"facet_keys: LeafKind {leaf!r} is not a facet leaf"
    )


# Allowed shapes for the (entity, period) family — used by
# :func:`broadcast_to_period`.  These parameters never authorise a time
# axis.  Shapes outside ``{SCALAR, MAP_PERIOD}`` raise on detection.
_PERIOD_ONLY_SHAPES: frozenset[Shape] = frozenset(
    (Shape.SCALAR, Shape.MAP_PERIOD)
)


# ---------------------------------------------------------------------------
# ResolvedShape — the resolver's output
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ResolvedShape:
    """Outcome of :func:`resolve_param_shape`.

    Carries everything a broadcast helper needs:

    * :attr:`shape` — the canonical :class:`Shape`.
    * :attr:`frame` — the parameter frame from
      :meth:`InputSource.parameter_explicit` (or :meth:`parameter`),
      or ``None`` when the parameter has no explicit rows.
    * :attr:`entity_dim_columns` — the entity's dim columns (for 0-dim
      classes this is ``["name"]``).
    * :attr:`period_index_column` — the column holding the period index
      (or ``None`` for scalar / time-only shapes).  Always ``"period"``
      when present (the SpineDbReader normalises ``index_name`` to the
      Map's authored name; the registry validates it's authored as
      "period").
    * :attr:`time_index_column` — the column holding the time index
      (or ``None`` for scalar / period-only shapes).  Always ``"t"``
      when present (per ``SpineDbReader._discover_index_cols`` which
      maps "time" → "t").
    * :attr:`stochastic_entities` — the entities (class-dim value tuples)
      authoring a stochastic-time map; :attr:`frame` and :attr:`shape`
      describe the other entities only.  Their values come from the
      solve's emit (:func:`broadcast_to_period_time`).
    * :attr:`stochastic_key` — the ``(entity_class, parameter_name)`` of
      those entities (a :data:`STOCHASTIC_TIME_PARAMS` key), or ``None``.
    * :attr:`key` — the ``(entity_class, parameter_name)`` it was resolved
      for (set by :func:`resolve_param_shape`); the solve-step aggregation
      of a solve with ``new_stepduration`` reads its kind by it.
    """

    shape: Shape
    frame: "pl.DataFrame | None"
    entity_dim_columns: tuple[str, ...]
    period_index_column: "str | None"
    time_index_column: "str | None"
    stochastic_entities: "frozenset[tuple[str, ...]]" = frozenset()
    stochastic_key: "tuple[str, str] | None" = None
    key: "tuple[str, str] | None" = None


# ---------------------------------------------------------------------------
# resolve_param_shape — DB-driven shape detection + validation
# ---------------------------------------------------------------------------


def _allowed_shape_names(allowed: "set[Shape]") -> str:
    """Render a stable, sorted comma-separated name list for error messages."""
    return ", ".join(sorted(s.value for s in allowed))


# Labels that the spinedb_api default-fills when the author didn't
# specify ``index_name`` on a Map.  Treated as "silent default" — when the whole parameter stays
# ambiguous even after value-domain probing, the resolver no longer gives up on every entity: each
# silent entity is classified on its own (:func:`_silent_entity_axes`), and only an entity with no key
# in the solve falls through to the seed / leaves the field unset (the same thing an un-authored entity
# does).  The caller never sees this per-entity work directly — it only ever gets back the same
# ``ResolvedShape | None`` contract.  Per the user advice, the error path (raising) applies to
# *explicit* mismatches, not authoring oversights — that still holds per entity (one entity's keys
# matching both axes raises, naming it, instead of silently returning None for the whole parameter).
_SILENT_DEFAULT_LABELS: frozenset[str] = frozenset(("x", ""))


def _normalise_label(n: "str | None") -> "str | None":
    """Lowercase / strip / collapse silent defaults to ``None``."""
    if n is None:
        return None
    if not isinstance(n, str):
        return None
    s = n.strip().lower()
    if not s or s in _SILENT_DEFAULT_LABELS:
        return None
    return s


# Per-shape depth + per-level canonical label, used to disambiguate
# silent-default ``index_name`` labels by consulting the per-parameter
# allow-list.  Each entry: shape → tuple of canonical labels per depth
# level (length 0/1/2).  Mirrors the enum membership.
# For facet-leaf shapes every slot is ``None`` — the author chooses the
# ``index_name`` per parameter value (``"tier"``, ``"i"``, ``"x"``, ...
# are all valid) so structural detection MUST NOT rely on a canonical
# label.  Phase A routes facet recognition through the registry
# allow-list in :func:`resolve_param_shape` instead (see
# :func:`_recognise_facet_shape`).  Listing the slots as ``None`` here
# documents "label-agnostic" semantics — and keeps
# :func:`_infer_silent_default_labels` consistent for these depths
# (the helper only fills slots when the candidate set agrees on a
# canonical label; ``None`` won't promote to anything).
_SHAPE_LABELS: "dict[Shape, tuple[str | None, ...]]" = {
    Shape.SCALAR:                   (),
    Shape.SCALAR_STR:               (),
    Shape.MAP_PERIOD:               ("period",),
    Shape.MAP_TIME:                 ("time",),
    Shape.MAP_PERIOD_TIME:          ("period", "time"),
    Shape.MAP_TIME_PERIOD:          ("time", "period"),
    Shape.MAP_TIER_FACET:           (None, None),
    Shape.MAP_PERIOD_TIER_FACET:    (None, None, None),
}


def _infer_silent_default_labels(
    raw_labels: "list[str | None]",
    allowed: "set[Shape]",
) -> "list[str | None]":
    """Fill silent-default ``index_name`` slots from the allow-list.

    When the DB authored a Map without ``index_name`` (the spinedb_api
    silent default ``"x"``), the raw label collapses to ``None`` through
    :func:`_normalise_label`.  For parameters whose allow-list has a
    *unique* choice at the observed depth + position, the silent default
    is unambiguous: the registry only permits one shape with that
    depth/position, so the author's intent is recoverable without
    rewriting the DB.

    Example: ``("group", "co2_max_period")`` admits
    ``{SCALAR, MAP_PERIOD}``.  Depth=1 admits only ``MAP_PERIOD`` →
    position-0 label is unambiguously ``"period"``.  Depth=1 for
    ``("commodity", "price")`` admits both ``MAP_PERIOD`` and
    ``MAP_TIME``: position-0 stays ``None`` (genuinely ambiguous; the
    caller falls back).

    Returns a fresh list with disambiguated labels filled in; original
    non-silent labels are passed through unchanged.  The output has the
    same length as ``raw_labels``.
    """
    normalised = [_normalise_label(n) for n in raw_labels]
    depth = len(normalised)
    if all(n is not None for n in normalised):
        # No silent defaults to disambiguate.
        return normalised
    # Per the allow-list, which canonical-label tuples have matching
    # depth?  Only shapes at the observed depth.
    candidates = [
        _SHAPE_LABELS[s] for s in allowed if len(_SHAPE_LABELS[s]) == depth
    ]
    if not candidates:
        # No allowed shape at this depth — leave it for the post-resolve
        # allow-list check / unresolved-shape branch.
        return normalised
    # For each position, the unique label across all candidates fills
    # silent defaults.  Mixed positions stay ambiguous.
    filled: "list[str | None]" = list(normalised)
    for pos in range(depth):
        if filled[pos] is not None:
            continue
        labels_at_pos = {cand[pos] for cand in candidates}
        if len(labels_at_pos) == 1:
            filled[pos] = next(iter(labels_at_pos))
    return filled


def _shape_from_indices(index_names: "list[str | None]",
                        ) -> "Shape | None":
    """Map a depth-ordered list of raw ``index_name`` labels to a
    :class:`Shape`.

    The user advice "If it says period, it is period; if it says time,
    it is time. If does not match, error to the user." is enforced
    here for *explicit* mismatches.  Empty / ``None`` / spinedb_api's
    ``"x"`` silent default propagate as ambiguity → ``return None``,
    letting the caller fall through to whatever non-resolver pathway
    the seed used to fill (see Δ.18+ punch-list to clean up fixtures
    that author Maps without ``index_name``).

    Returns ``None`` when at least one level is a silent default
    (cannot disambiguate without DB-side authoring fix).
    Returns the resolved :class:`Shape` when every level is explicitly
    "period" or "time" / "t".
    Raises :class:`_UnrecognisedIndex` when at least one level carries
    an explicit label outside that set (e.g. "tier", "branch").
    """
    # Canonical normalisation: silent defaults collapse to None.
    normalised = [_normalise_label(n) for n in index_names]

    # Silent defaults at any level → return None (ambiguous; caller
    # falls back).  We DON'T raise here because the spinedb_api default
    # is a silent oversight, not a deliberate mismatch.
    #
    # Phase A note: registry-driven recognition of facet-leaf shapes
    # (``MAP_TIER_FACET``, ``MAP_PERIOD_TIER_FACET``) does NOT live
    # here — :func:`resolve_param_shape` handles it via the allow-list
    # because the user can author any ``index_name`` they want for the
    # outer Map levels (``index_name`` is user-set; ``"x"`` is the
    # spinedb_api default when unset).  The shape + position is what
    # identifies the parameter family; label inspection only
    # disambiguates the legacy depth-1 / depth-2 period-vs-time case
    # below.
    if any(n is None for n in normalised):
        return None

    if len(normalised) == 0:
        return Shape.SCALAR
    if len(normalised) == 1:
        n0 = normalised[0]
        if n0 == "period":
            return Shape.MAP_PERIOD
        if n0 in ("time", "t"):
            return Shape.MAP_TIME
        raise _UnrecognisedIndex(n0, depth=0)
    if len(normalised) == 2:
        n0, n1 = normalised[0], normalised[1]
        if n0 == "period" and n1 in ("time", "t"):
            return Shape.MAP_PERIOD_TIME
        if n0 in ("time", "t") and n1 == "period":
            return Shape.MAP_TIME_PERIOD
        raise _UnrecognisedIndex(f"({n0!r}, {n1!r})", depth=0)
    # Higher depth — not supported here.
    raise _UnrecognisedIndex(
        f"depth {len(normalised)} (got {normalised})", depth=0,
    )


def check_index_name_path(entity_class: str, parameter_name: str, entity_name: str,
                          index_names: "list[str | None]") -> Shape:
    """The :class:`Shape` of one value with raw per-level ``index_name`` labels ``index_names``
    (``[]`` = scalar), validated against :data:`PARAM_ALLOWED_SHAPES`.

    The same label rules as :func:`resolve_param_shape`: a silent label (``x`` / empty / None) takes
    the allow-list's unique label at its depth.  For readers that see the raw Spine values without an
    :class:`InputSource` (``_timeline.TimelineConfig``).  Raises :class:`FlexToolConfigError` when the
    labels are not a valid axis, stay ambiguous, or give a shape outside the allow-list.
    """
    key = (entity_class, parameter_name)
    allowed = PARAM_ALLOWED_SHAPES.get(key)
    if allowed is None:
        raise FlexToolConfigError(
            f"check_index_name_path: parameter {key!r} is not in PARAM_ALLOWED_SHAPES.")
    labels = _infer_silent_default_labels(list(index_names), allowed)
    try:
        shape = _shape_from_indices(labels)
    except _UnrecognisedIndex as exc:
        raise FlexToolConfigError(
            f"Parameter ({entity_class!r}, {parameter_name!r}) of {entity_name!r} carries an unrecognised "
            f"index_name {exc.label}.  Allowed shapes: {_allowed_shape_names(allowed)}.") from None
    if shape is None or shape not in allowed:
        observed = shape.value if shape is not None else f"index_names={list(index_names)}"
        raise FlexToolConfigError(
            f"Parameter ({entity_class!r}, {parameter_name!r}) of {entity_name!r} was authored as {observed} "
            f"but allowed shapes are: {_allowed_shape_names(allowed)}.")
    return shape


def shape_rank(shape: Shape) -> int:
    """The number of index levels of a value of *shape* (0 for a constant): the length of its
    :data:`_SHAPE_LABELS` entry.  The single source of a shape's depth (also the schema's map rank)."""
    return len(_SHAPE_LABELS[shape])


#: What each map shape is over, in user words (:func:`allowed_shapes_in_words`).
_SHAPE_WORDS: "dict[Shape, str]" = {
    Shape.MAP_PERIOD: "periods",
    Shape.MAP_TIME: "time",
    Shape.MAP_PERIOD_TIME: "periods and time",
    Shape.MAP_TIME_PERIOD: "time and periods",
    Shape.MAP_TIER_FACET: "tiers and price / quantity",
    Shape.MAP_PERIOD_TIER_FACET: "periods, tiers and price / quantity",
}
_FACET_SHAPES: "frozenset[Shape]" = frozenset((Shape.MAP_TIER_FACET, Shape.MAP_PERIOD_TIER_FACET))


def _levels(n: int) -> str:
    return f"{n} level" if n == 1 else f"{n} levels"


def allowed_shapes_in_words(allowed: "set[Shape]") -> str:
    """*allowed* for a user message: ``a constant, or a map over periods or over time (1 level)``."""
    parts: list[str] = []
    if allowed & {Shape.SCALAR, Shape.SCALAR_STR}:
        parts.append("a constant")
    by_rank: "dict[int, list[Shape]]" = {}
    for shape in allowed:
        if shape_rank(shape):
            by_rank.setdefault(shape_rank(shape), []).append(shape)
    for rank in sorted(by_rank):
        over = " or over ".join(_SHAPE_WORDS[s] for s in sorted(by_rank[rank], key=lambda s: s.value))
        parts.append(f"a map over {over} ({_levels(rank)})")
    return ", or ".join(parts)


def _entity_text(entity: "tuple[str, ...]") -> str:
    return "/".join(str(e) for e in entity)


def _label_path_problem(path: "tuple[str | None, ...]", allowed: "set[Shape]",
                        period_time_param: bool) -> "str | None":
    """Why one value with raw per-level labels *path* cannot be one of the *allowed* shapes, or None.

    (1) its depth is not the rank of an allowed shape; (2) an explicit label is one that no allowed shape of
    that depth has at that level (a period / time parameter places its levels by label, so there the order of
    ``period`` and ``time`` is free).  Silent labels (``x`` / empty) are accepted: their axis is decided from
    the value's keys later.  Price-ladder levels: level 1 of a 3-level ladder is the period (named ``period``
    or unnamed); the tier and price / quantity levels may have any name except ``period`` / ``time``.
    """
    norm = ["time" if n == "t" else n for n in (_normalise_label(lbl) for lbl in path)]
    depth = len(norm)
    candidates = [s for s in allowed if shape_rank(s) == depth]
    if not candidates:
        return ("the value is a constant" if depth == 0
                else f"the value has {depth} index level{'' if depth == 1 else 's'}")
    explicit = [(pos, n) for pos, n in enumerate(norm) if n is not None]
    if not explicit:
        return None
    if all(s in _FACET_SHAPES for s in candidates):
        for pos, n in explicit:
            if depth == 3 and pos == 0:
                if n != "period":
                    return (f"level 1 of the value is named {path[pos]!r}; level 1 of a 3-level ladder is the "
                            "period (name it 'period' or leave it unnamed)")
                continue
            if n in ("period", "time"):
                return (f"level {pos + 1} of the value is named {path[pos]!r}, but that is a tier or a "
                        "price / quantity level")
        return None
    for shape in candidates:
        labels = _SHAPE_LABELS[shape]
        if period_time_param:
            names = [n for _pos, n in explicit]
            if all(n in labels for n in names) and len(set(names)) == len(names):
                return None
        elif all(labels[pos] == n for pos, n in explicit):
            return None
    named = ", ".join(f"level {pos + 1} {path[pos]!r}" for pos, _n in explicit)
    return f"the index levels of the value are named {named}"


def _check_entity_label_paths(source: "InputSource", entity_class: str, parameter_name: str,
                              allowed: "set[Shape]") -> None:
    """Check every entity's raw label path (:meth:`InputSource.parameter_entity_label_paths`) against
    *allowed* (:func:`_label_path_problem`) and raise :class:`FlexToolConfigError` naming the entities whose
    value cannot be one of the allowed shapes.  A source without per-entity paths is not checked.  Memoised on
    the run's reader (the paths do not depend on the solve)."""
    fn = getattr(source, "parameter_entity_label_paths", None)
    if fn is None:
        return

    def check() -> bool:
        try:
            paths = fn(entity_class, parameter_name)
        except KeyError:
            return True
        pt = (entity_class, parameter_name) in PERIOD_TIME_PARAMS
        problems = []
        for entity, path in sorted(paths.items()):
            problem = _label_path_problem(tuple(path), allowed, pt)
            if problem is not None:
                problems.append(f"{entity_class} {_entity_text(entity)!r}, parameter {parameter_name!r}: "
                                f"{problem}")
        if problems:
            more = f" (and {len(problems) - 5} more)" if len(problems) > 5 else ""
            raise FlexToolConfigError(
                f"{'; '.join(problems[:5])}{more}. This parameter takes {allowed_shapes_in_words(allowed)}. "
                "Name the map's index levels 'period' / 'time' as the parameter expects, or leave them unnamed.")
        return True

    _run_memo(source, ("entity_label_path_check", entity_class, parameter_name), check)


class _UnrecognisedIndex(Exception):
    """Internal sentinel for :func:`_shape_from_indices` — caller maps
    to :class:`FlexToolConfigError` with full context."""

    def __init__(self, label: str, depth: int):
        super().__init__(label)
        self.label = label
        self.depth = depth


def _period_time_vocab(
    period_filter: "pl.DataFrame | None",
) -> "tuple[set[str] | None, set[str] | None]":
    """The period / timestep vocabularies a silent-default Map's keys are classified against.

    Periods are the active solve's own ``d`` plus, when a stochastic load has armed them
    (:func:`flextool.engine.preprocess.params._branch.active_branch_anchor_pairs`), the anchor periods of
    the active branch-anchor pairs — a labelled ``{anchor: v}`` row keeps its value for the branch periods
    through :func:`_with_branch_periods` / ``inherit_active_branch_periods`` even when the anchor itself
    has left ``d`` (a non-realized forecast branch under the forecasts-only stochastic fan).  Classifying
    a silent-label row against the same, wider universe keeps the silent and the labelled path from
    diverging.  Timesteps are the active solve's own ``t``, unchanged.

    Shared by :func:`_disambiguate_shape_by_value_domain` (the whole-parameter probe) and
    :func:`_silent_entity_axes` (the per-entity fallback) so the two cannot disagree.  Either element is
    ``None`` when *period_filter* is ``None`` or lacks that column.
    """
    if period_filter is None:
        return None, None
    pf_cols = period_filter.columns
    periods_set: "set[str] | None" = None
    timesteps_set: "set[str] | None" = None
    if "d" in pf_cols:
        periods_set = set(
            period_filter.lazy()
                         .select(pl.col("d").cast(pl.Utf8))
                         .unique()
                         .collect()
                         .get_column("d")
                         .to_list()
        )
        # Function-local import: input -> preprocess, kept out of the module import order (mirrors
        # ``_with_branch_periods``).
        from flextool.engine.preprocess.params._branch import active_branch_anchor_pairs
        pairs = active_branch_anchor_pairs()
        if pairs is not None and pairs.height > 0:
            periods_set = periods_set | set(pairs.get_column("anchor").cast(pl.Utf8).to_list())
    if "t" in pf_cols:
        timesteps_set = set(
            period_filter.lazy()
                         .select(pl.col("t").cast(pl.Utf8))
                         .unique()
                         .collect()
                         .get_column("t")
                         .to_list()
        )
    return periods_set, timesteps_set


def _disambiguate_shape_by_value_domain(
    df: "pl.DataFrame",
    ent_cols: "list[str]",
    resolved_labels: "list[str | None]",
    allowed: "set[Shape]",
    period_filter: "pl.DataFrame | None",
) -> "Shape | None":
    """Value-domain probing fallback for silent-default ``index_name`` Maps.

    When the registry permits multiple shapes at the observed depth
    (e.g. ``("group", "co2_price")`` admits both ``MAP_PERIOD`` and
    ``MAP_TIME``) and the DB carried no ``index_name`` label, we
    cannot decide structurally.  Mirroring the legacy CSV pipeline
    (``_timeline.separate_period_and_timeseries_data``), we look at
    the Map's actual index values: a Map keyed by ``y2019, y2020, ...``
    is :class:`Shape.MAP_PERIOD`; a Map keyed by ``t0001, t0002, ...``
    is :class:`Shape.MAP_TIME`.

    Requires *period_filter* with columns ``d`` (periods) and ``t``
    (timesteps); the period universe also includes the active stochastic
    branch-anchor periods (:func:`_period_time_vocab`), so a labelled and
    a silent-default map of the same entity never disagree.  Returns
    ``None`` when probing is impossible (no filter, no candidates, mixed
    values matching neither set).

    Currently only handles depth-1 ambiguity (a single Map level).  For
    depth-2 Maps we don't yet probe — the registry's allow-list usually
    fixes the ordering by allowing only one 2d shape at a time, so the
    silent-default case there is rare; the standard structural pathway
    handles every fixture observed in the cascade gate as of 2026-05.
    """
    if period_filter is None:
        return None
    depth = len(resolved_labels)
    if depth != 1:
        return None
    # Identify the silent-default level (the only position with None
    # in resolved_labels — every other depth-1 case yielded a definite
    # shape above and never reached this fallback).
    if resolved_labels[0] is not None:
        return None
    # The candidate set at depth 1 is the intersection of *allowed* with
    # depth-1 shapes: at most MAP_PERIOD and MAP_TIME.
    candidates = {s for s in allowed
                  if len(_SHAPE_LABELS[s]) == 1
                  and s in (Shape.MAP_PERIOD, Shape.MAP_TIME)}
    if not candidates:
        return None
    # Locate the index column (the first non-entity, non-value column).
    non_ent_cols = [
        c for c in df.columns if c not in ent_cols and c != "value"
    ]
    if len(non_ent_cols) != 1:
        # Defensive: depth-1 frame should carry exactly one index column.
        return None
    idx_col = non_ent_cols[0]
    # Get distinct index values from the frame.
    try:
        idx_values = (
            df.lazy()
              .select(pl.col(idx_col).cast(pl.Utf8))
              .unique()
              .collect()
              .get_column(idx_col)
              .to_list()
        )
    except Exception:
        return None
    if not idx_values:
        return None
    idx_set = {str(v) for v in idx_values if v is not None}
    # Build the period / timestep universes from period_filter (+ active branch anchors).
    periods_set, timesteps_set = _period_time_vocab(period_filter)
    # Intersects-and-not-the-other: the index must overlap exactly one
    # universe and not the other.  We use intersection (not subset) so a
    # Map authored with more periods than the active solve realises still
    # resolves cleanly — extra (inactive-period) keys are dropped
    # downstream by ``_filter_param_by_periods``.  Mixed (overlaps both)
    # or no-overlap → genuinely ambiguous → don't guess.
    hits_periods   = bool(periods_set)   and bool(idx_set & periods_set)
    hits_timesteps = bool(timesteps_set) and bool(idx_set & timesteps_set)
    if hits_periods and not hits_timesteps and Shape.MAP_PERIOD in candidates:
        return Shape.MAP_PERIOD
    if hits_timesteps and not hits_periods and Shape.MAP_TIME in candidates:
        return Shape.MAP_TIME
    return None


def _read_index_names_from_source(
    source: "InputSource",
    entity_class: str,
    parameter_name: str,
) -> "list[str | None]":
    """Read the raw per-level ``index_name`` labels from the source.

    Honours the user advice "read from database the dimensionality of
    the parameter" and "read the dimension index label from the
    database".

    Implementation:

    * If the source provides :meth:`parameter_shape_info`, use it (the
      canonical path — see :class:`SpineDbReader.parameter_shape_info`).
    * Otherwise (e.g. :class:`InMemoryReader` in unit tests), inspect
      the parameter frame's columns directly.  This is the legacy path;
      InMemoryReader test fixtures author frames with column names
      already matching the resolved shape.

    Returns a list of raw labels per nesting depth (length 0 for
    scalar; length 1/2 for maps).  Labels may be ``None`` when the DB
    didn't author one — the caller will raise via the
    :class:`_UnrecognisedIndex` path.
    """
    fn = getattr(source, "parameter_shape_info", None)
    if fn is not None:
        return list(fn(entity_class, parameter_name))
    # Fallback path (InMemoryReader): infer from the frame's column
    # names, treating the first non-entity column as depth-0 etc.
    df = _try_parameter_frame(source, entity_class, parameter_name)
    if df is None or df.height == 0:
        return []
    ent_cols = _entity_dim_columns_for_frame(df, source, entity_class)
    cols = [c for c in df.columns if c not in ent_cols and c != "value"]
    # The frame's column names are the post-normalisation labels
    # (period stays "period"; time → "t" via SpineDbReader).  We translate
    # back to canonical labels for the shape resolver.
    out: list[str | None] = []
    for c in cols:
        if c == "period":
            out.append("period")
        elif c in ("t", "time"):
            out.append("time")
        else:
            out.append(c)
    return out


def parameter_index_label_paths(
    source: "InputSource",
    entity_class: str,
    parameter_name: str,
) -> "set[tuple[str | None, ...]]":
    """The distinct raw ``index_name`` paths of the parameter's values (``()`` = scalar), read without
    classifying any level.

    The period / time classifier (``parameter_shape_info`` on a ``PERIOD_TIME_PARAMS`` key) raises on an
    explicit label that is not ``period`` / ``time`` (e.g. ``branch`` of a stochastic-time map), so a
    caller that must recognise rank 3 before resolving asks this first.  Uses the source's own
    ``parameter_index_label_paths`` when it has one (``SpineDbReader``, ``InMemoryReader``); other
    sources fall back to the raw label list.  An unknown parameter has no paths.
    """
    fn = getattr(source, "parameter_index_label_paths", None)
    try:
        if fn is not None:
            return set(fn(entity_class, parameter_name))
        labels = _read_index_names_from_source(source, entity_class, parameter_name)
    except (KeyError, AttributeError):
        return set()
    return {tuple(labels)}


def has_silent_labels(paths: "set[tuple[str | None, ...]]") -> bool:
    """Whether some Map level carries Spine's silent default label (``x`` / empty / None), whose axis the
    resolver may decide by probing the solve's periods and timesteps."""
    return any(_normalise_label(lbl) is None for p in paths for lbl in p)


# ---------------------------------------------------------------------------
# Stochastic-time entities (STOCHASTIC_TIME_PARAMS)
# ---------------------------------------------------------------------------


def _run_memo(source: "InputSource", key: tuple, factory):
    """``factory()`` memoised on the run's reader when it has ``run_memo`` (data-only values)."""
    memo = getattr(source, "run_memo", None)
    return memo(key, factory) if memo is not None else factory()


def stochastic_time_entities(
    source: "InputSource",
    entity_class: str,
    parameter_name: str,
) -> "frozenset[tuple[str, ...]]":
    """The entities (class-dim value tuples) whose value of a :data:`STOCHASTIC_TIME_PARAMS` parameter is a
    stochastic-time map, read from the raw index labels (no period / time classifier).

    A depth-3 value must carry the stochastic-time labels ``branch / analysis_time / time`` (``time_start``
    for the second level, ``t`` for the third, or Spine's silent ``x`` on any level); other labels, or a
    deeper map, raise :class:`FlexToolConfigError` naming the entity.  Empty for other parameters and for
    sources without ``parameter_entity_label_paths``.  Memoised on the run's reader.
    """
    key = (entity_class, parameter_name)
    if key not in STOCHASTIC_TIME_PARAMS:
        return frozenset()
    fn = getattr(source, "parameter_entity_label_paths", None)
    if fn is None:
        return frozenset()

    def compute() -> "frozenset[tuple[str, ...]]":
        try:
            paths = fn(entity_class, parameter_name)
        except KeyError:  # unknown class or parameter
            return frozenset()
        found: set[tuple[str, ...]] = set()
        for ent, path in sorted(paths.items()):
            if len(path) <= 2:
                continue
            labels = [_normalise_label(lbl) for lbl in path]
            if len(path) > 3 or any(lbl is not None and lbl not in ok
                                    for lbl, ok in zip(labels, _STOCHASTIC_TIME_LABELS)):
                raise FlexToolConfigError(
                    f"Parameter ({entity_class!r}, {parameter_name!r}) of {'/'.join(ent)!r} is a {len(path)}-level "
                    f"map with index names {list(path)}.  A map deeper than period / time is a stochastic-time "
                    "value: branch → analysis_time → time (or without index names).  Edit the source database.")
            found.add(tuple(ent))
        return frozenset(found)

    return _run_memo(source, ("stochastic_time_entities", entity_class, parameter_name), compute)


def _view_without(source: "InputSource", entity_class: str, parameter_name: str,
                  entities: "frozenset[tuple[str, ...]]", tag: str) -> "InputSource":
    """``source`` without ``entities``' values of ``(entity_class, parameter_name)``, memoised on the run's
    reader under ``(tag, entity_class, parameter_name, entities)``.

    ``tag`` separates callers with different memo-validity assumptions:

    * ``"stochastic_time_view"`` (:func:`_stochastic_view`) — the entity set is solve-agnostic (every
      stochastic-time entity of the parameter, independent of the active roll), so one view serves every
      call of a run.
    * ``"no_key_in_solve"`` (the silent-label fallback, :func:`resolve_param_shape`) — the entity set is
      the active solve's own drop set (:func:`_silent_entity_axes`), so it is recomputed, and a fresh view
      memoised, on every roll; the number of distinct views is bounded by the number of distinct drop sets,
      at most one per solve.

    Raises :class:`FlexToolConfigError` when the source has no ``without_entities`` (every production
    reader does; a test stub that doesn't is not expected to reach either caller).
    """
    fn = getattr(source, "without_entities", None)
    if fn is None:
        raise FlexToolConfigError(
            f"{entity_class}.{parameter_name}: the input source {type(source).__name__} reports values to hide "
            "from the period / time reader but cannot (no without_entities).")
    return _run_memo(source, (tag, entity_class, parameter_name, frozenset(entities)),
                     lambda: fn(entity_class, parameter_name, sorted(entities)))


def _stochastic_view(source: "InputSource", entity_class: str, parameter_name: str,
                     entities: "frozenset[tuple[str, ...]]") -> "InputSource":
    """``source`` without the stochastic-time entities' values of that one parameter (memoised on the run's
    reader: the entity set is solve-agnostic)."""
    return _view_without(source, entity_class, parameter_name, entities, "stochastic_time_view")


class _StochasticTimeSource:
    """The load's emit frames that fold ``pbt_*`` onto the solve's (d, t); one per ``load_flextool``.

    ``warned`` is the run's set of (kind, class, parameter, entity) warnings already logged; ``load_flextool``
    passes one kept on the run's reader, so the rolls of one run warn once and a later run warns again.
    """

    def __init__(self, provider: object, warned: "set | None" = None) -> None:
        self._provider = provider
        self._folds: dict[tuple[str, str], pl.DataFrame] = {}
        self.warned: set = warned if warned is not None else set()

    def has_realized_branch_rows(self) -> bool:
        """Whether ``solve_data/solve_branch__time_branch`` carries any row for the active solve.

        A realized (or forecast) branch assignment can exist even when this solve has no
        branch-anchor pairs of its own: an auto-spawned realized-dispatch sibling
        (``dispatch_realization_method=separate_dispatch``) leaves its own ``period__branch``
        empty, but ``_run_model.project_realized_branch_time_branch_rows`` projects its parent's
        already-realized ``solve_branch__time_branch`` rows onto it, so the fold below reads the
        realized branch's own value instead of silently falling through to the class default.
        """
        provider = self._provider
        if provider is None or not provider.has("solve_data/solve_branch__time_branch"):
            return False
        df = provider.get("solve_data/solve_branch__time_branch")
        return df is not None and df.height > 0

    def fold(self, key: "tuple[str, str]") -> pl.DataFrame:
        """``[<emit entity columns>, d, t, value]`` of parameter ``key[1]`` in its emit frame (Utf8 keys,
        Float64 values), memoised for the load."""
        emit_key, emit_cols = STOCHASTIC_TIME_PARAMS[key]
        memo_key = (emit_key, key[1])
        if memo_key not in self._folds:
            provider = self._provider
            if provider is None or not provider.has(emit_key):
                raise FlexToolConfigError(
                    f"{key[0]}.{key[1]}: stochastic-time values are read from the solve's emit frame "
                    f"{emit_key!r}, which this load does not have.")
            frame = provider.get(emit_key)
            self._folds[memo_key] = (
                frame.lazy()
                .filter(pl.col("param").cast(pl.Utf8) == key[1])
                .select(*(pl.col(c).cast(pl.Utf8) for c in emit_cols),
                        pl.col("period").cast(pl.Utf8).alias("d"),
                        pl.col("time").cast(pl.Utf8).alias("t"),
                        pl.col("value").cast(pl.Float64))
                .collect())
        return self._folds[memo_key]


_STOCHASTIC_TIME_SOURCE: "contextvars.ContextVar[_StochasticTimeSource | None]" = contextvars.ContextVar(
    "flextool_stochastic_time_source", default=None)


def activate_stochastic_time_source(provider: object, warned: "set | None" = None) -> "contextvars.Token":
    """Make ``provider``'s emit frames the source of stochastic-time values for the load being built; returns
    the token for :func:`deactivate_stochastic_time_source`.  ``load_flextool`` arms it when it reads the DB
    (the emit frames are in the provider before any DB-direct reader runs).  ``warned``: the run's
    warning-dedupe set (a fresh one when omitted)."""
    return _STOCHASTIC_TIME_SOURCE.set(_StochasticTimeSource(provider, warned))


def deactivate_stochastic_time_source(token: "contextvars.Token") -> None:
    """Restore the stochastic-time source to what it was before the matching activation."""
    _STOCHASTIC_TIME_SOURCE.reset(token)


def _warn_once(kind: str, key: "tuple[str, str]", entities, message: str) -> None:
    """One WARNING per (kind, class, parameter, entity) and run, naming the not yet warned entities sorted."""
    scope = _STOCHASTIC_TIME_SOURCE.get()
    warned = scope.warned if scope is not None else set()
    new = [tuple(e) for e in sorted(entities) if (kind, key[0], key[1], tuple(e)) not in warned]
    if not new:
        return
    warned.update((kind, key[0], key[1], e) for e in new)
    _LOGGER.warning("%s.%s: the stochastic-time value of %s %s", key[0], key[1],
                    ", ".join("/".join(e) for e in new), message)


def _warn_entities_without_fold(key: "tuple[str, str]", entities, covered: pl.DataFrame) -> None:
    """Warn (once per run) for the stochastic-time entities with no folded row: the entity is outside the emit's
    domain, so the solve does not use the parameter for it."""
    have = {tuple(r) for r in covered.unique().iter_rows()}
    _warn_once("no_fold", key, [e for e in entities if tuple(e) not in have],
               "has no folded value in this solve (outside the solve's use of the parameter); it is not applied.")


def stochastic_time_values(
    key: "tuple[str, str]",
    entities: "frozenset[tuple[str, ...]]",
    entity_dim_columns: "tuple[str, ...] | list[str]",
) -> pl.DataFrame:
    """``[*entity_dim_columns, d, t, value]``: the solve's folded stochastic-time values of ``entities``.

        value[e, d, t] = the branch series of e for the solve's analysis time    d a branch period of e's
                                                                                 include_stochastics group
                       = the realized branch's series                           otherwise (including a
                                                                                 solve without its own
                                                                                 stochastic branches, e.g. an
                                                                                 auto-spawned realized-dispatch
                                                                                 sibling)
        ∀ (d, t) of the solve's emit (steps_in_use)

    This is what the per-solve emit computes from ``pbt_*`` (and what the CSV seeds carry when the labels are
    silent).  A solve without its own stochastic branch periods (no active branch-anchor pairs) still takes
    the realized branch's value when one is designated for it -- a realized-dispatch sibling has no
    ``period__branch`` of its own, but its parent's realized ``solve_branch__time_branch`` rows are projected
    onto it (:meth:`_StochasticTimeSource.has_realized_branch_rows`), so the fold below still resolves the
    realized branch rather than the class default.  Only when NO realized branch is designated anywhere for
    this solve does owner decision #82(4) raise instead of silently falling back to the class default.
    Raises :class:`FlexToolConfigError` outside ``load_flextool`` (no active emit), when the emit frame is
    missing, and in that no-realized-branch case.  Rows sorted by the keys; Utf8 keys, Float64 values.
    """
    scope = _STOCHASTIC_TIME_SOURCE.get()
    if scope is None:
        raise FlexToolConfigError(
            f"{key[0]}.{key[1]}: stochastic-time values of "
            f"{', '.join('/'.join(e) for e in sorted(entities))} are folded per solve; they can only be read "
            "during load_flextool.")
    # Function-local import: input -> preprocess, kept out of the module import order.
    from flextool.engine.preprocess.params._branch import active_branch_anchor_pairs

    if active_branch_anchor_pairs() is None and not scope.has_realized_branch_rows():
        names = ", ".join("'" + "/".join(e) + "'" for e in sorted(entities))
        raise FlexToolConfigError(
            f"{key[0]} {names}, parameter '{key[1]}': the value is a stochastic forecast (a map over branch, "
            "analysis time and time), but this solve has no realized branch to take the value from: no row of "
            "its stochastic_branches says realized: yes, and no other solve hands its realized branch to this "
            "one. Either mark one branch realized: yes in this solve's stochastic_branches, or give the "
            "parameter a plain value instead of a forecast map. Check also which alternative of the scenario "
            "gives the value.")
    emit_cols = list(STOCHASTIC_TIME_PARAMS[key][1])
    dim_cols = list(entity_dim_columns)
    ents = pl.DataFrame([list(e) for e in sorted(entities)], schema={c: pl.Utf8 for c in emit_cols},
                        orient="row")
    return (scope.fold(key)
            .join(ents, on=emit_cols, how="semi")
            .rename(dict(zip(emit_cols, dim_cols)))
            .select(*dim_cols, "d", "t", "value")
            .sort(*dim_cols, "d", "t"))


def parameter_has_values(source: "InputSource", entity_class: str, parameter_name: str) -> bool:
    """Whether ``entity_class.parameter_name`` has any value (explicit, default, or a stochastic-time map):
    the shape-agnostic presence check for readers that only test for a value."""
    if stochastic_time_entities(source, entity_class, parameter_name):
        return True
    df = _try_parameter_frame(source, entity_class, parameter_name)
    return df is not None and df.height > 0


def _try_parameter_frame(
    source: "InputSource",
    entity_class: str,
    parameter_name: str,
) -> "pl.DataFrame | None":
    """Wrap ``source.parameter_explicit`` / ``source.parameter`` with
    None-on-KeyError semantics.  Used by both the InMemory shape-info
    fallback and by :func:`resolve_param_shape` to fetch the data.

    Δ.28 — when ``parameter_explicit`` returns an empty frame (no
    explicit overrides for any entity), fall through to ``parameter``
    so the schema default (e.g. ``availability = 1.0``) is consulted.
    Without this fall-through, the resolver dropped fields whose
    Spine value is purely the default (``p_process_availability`` on
    fixtures where every unit/connection inherits the 1.0 default,
    e.g. ``work_lh2_three_region``) — the slow path's CSV
    preprocessing always writes the default-broadcast rows so the
    fast path was missing data.

    Parameters with ``default_value=None`` (the §4.5 None-skip family)
    behave unchanged: ``parameter`` still returns the explicit-only
    frame, so an empty result remains an empty result and the resolver
    correctly drops the field.
    """
    try:
        explicit = source.parameter_explicit(entity_class, parameter_name)
    except (KeyError, AttributeError):
        explicit = None
    if explicit is not None and explicit.height > 0:
        return explicit
    try:
        return source.parameter(entity_class, parameter_name)
    except (KeyError, AttributeError):
        # AttributeError covers stub ``InputSource`` implementations in
        # unit tests that only override ``parameter_explicit`` (e.g.
        # ``_ZeroStartupSource`` in ``test_a04_a05_online_ramp.py``).
        return explicit


def _recognise_facet_shape(
    raw_labels: "list[str | None]",
    allowed: "set[Shape]",
) -> "Shape | None":
    """Registry-driven detection for facet-leaf shapes (Phase A).

    Spine-side ``index_name`` labels are user-set per parameter value —
    the user may author ``Map(<anything>)`` with any label they like
    (the spinedb_api default is ``"x"``).  Phase A therefore identifies
    facet-leaf shapes (``MAP_TIER_FACET``, ``MAP_PERIOD_TIER_FACET``)
    from the **shape + position** of the value, not from the labels:

    * Depth of the raw shape info matches the candidate facet shape's
      depth in :data:`_SHAPE_AXES` (``len(map_levels) + 1`` because the
      facet itself sits at the innermost level).
    * The allow-list contains exactly one facet shape at that depth —
      so the structural match is unambiguous.

    When both conditions hold the helper returns the matching facet
    shape.  Otherwise ``None`` and the caller falls back to the legacy
    label-based detection via :func:`_shape_from_indices`.

    The helper does NOT inspect *raw_labels* values — that's the whole
    point: facet recognition is label-agnostic.  We only need the depth
    (``len(raw_labels)``).
    """
    depth = len(raw_labels)
    facet_candidates = [
        s for s in allowed
        if _SHAPE_AXES[s].leaf is LeafKind.FACET_PRICE_QUANTITY
        and len(_SHAPE_AXES[s].map_levels) + 1 == depth
    ]
    if len(facet_candidates) == 1:
        return facet_candidates[0]
    # Zero candidates → not a facet shape; >1 candidates → registry has
    # multiple facet shapes at this depth (no current entries do, but
    # be defensive — fall back to the legacy path which will surface
    # the ambiguity through its own checks).
    return None


def _entity_dim_columns_for_frame(
    df: "pl.DataFrame",
    source: "InputSource",
    entity_class: str,
) -> list[str]:
    """Best-effort recovery of the entity dim columns of *df* by
    consulting :meth:`InputSource.entities` for the class.

    For 0-dim classes the column name is ``"name"``; for n-relationships
    it's the dim-class names with repeats disambiguated.  When
    :meth:`entities` raises KeyError (unknown class), fall back to
    ``["name"]`` — defensive; the registry only references known
    classes.
    """
    try:
        ent = source.entities(entity_class)
        cols = ent.columns
        if cols:
            return list(cols)
    except (KeyError, AttributeError):
        # AttributeError covers stub InputSource implementations in
        # unit-tests that only override ``parameter_explicit`` and don't
        # implement ``entities`` (e.g. the ``_ZeroStartupSource`` stub
        # in ``test_a04_a05_online_ramp.py``).
        pass
    return ["name"]


# ---------------------------------------------------------------------------
# Per-variant allow-list check (spec fix_map_reading.md §3.3)
# ---------------------------------------------------------------------------


# Dedupe the per-(class, param) INFO line to "one per run" (spec §3.3).
# resolve_param_shape may be called once per sub-solve; this keeps the log
# to a single line for a mixed-shape parameter across the whole process.
_MIXED_SHAPE_LOGGED: "set[tuple[str, str]]" = set()


def _variant_to_shape(variant: "tuple[str, ...]") -> "Shape | None":
    """Map one classified axis path (e.g. ``("period",)``) to a Shape.

    Returns ``None`` for a path that does not correspond to a recognised
    period/time shape (treated as a violation by the caller)."""
    try:
        return _shape_from_indices(list(variant))
    except _UnrecognisedIndex:
        return None


def _entity_axes_map(
    df: "pl.DataFrame",
    ent_cols: "tuple[str, ...]",
) -> "dict[tuple[str, ...], tuple[str, ...]]":
    """Group *df* rows by entity and return each entity's classified axis
    path (``("period",)`` / ``("time",)`` / ``("period", "time")`` / ``()``)
    derived from the non-null ``period`` / ``t`` columns.

    Used only on the error / mixed-shape-logging paths to name entities
    and count per-shape multiplicities — never on the hot path.
    """
    axis_cols = [c for c in ("period", "t") if c in df.columns]
    ent_present = [c for c in ent_cols if c in df.columns]
    out: "dict[tuple[str, ...], set[str]]" = {}
    if not ent_present:
        return {}
    for row in df.select(ent_present + axis_cols).iter_rows(named=True):
        ent = tuple(str(row[c]) for c in ent_present)
        axes = out.setdefault(ent, set())
        if "period" in axis_cols and row.get("period") is not None:
            axes.add("period")
        if "t" in axis_cols and row.get("t") is not None:
            axes.add("time")
    canonical: "dict[tuple[str, ...], tuple[str, ...]]" = {}
    for ent, axes in out.items():
        path: list[str] = []
        if "period" in axes:
            path.append("period")
        if "time" in axes:
            path.append("time")
        canonical[ent] = tuple(path)
    return canonical


def _validate_shape_variants(
    variants: "set[tuple[str, ...]]",
    allowed: "set[Shape]",
    df: "pl.DataFrame",
    ent_cols: "tuple[str, ...]",
    entity_class: str,
    parameter_name: str,
) -> None:
    """Check each per-row shape variant against *allowed* (spec §3.3).

    A combined ``MAP_PERIOD_TIME`` is accepted for a parameter whose
    allow-list lacks it (e.g. ``commodity.price``) as long as every
    individual row shape is allowed.  A genuine 2-D row, or a time map
    on a period-only parameter, is rejected with a
    :class:`FlexToolConfigError` naming the offending entities and the
    shape each was authored with.  Logs one INFO line per (class, param)
    per run when there is more than one variant.
    """
    bad_variants = {
        var for var in variants
        if (_variant_to_shape(var) is None
            or _variant_to_shape(var) not in allowed)
    }
    if bad_variants:
        ent_axes = _entity_axes_map(df, ent_cols)
        parts: list[str] = []
        for var in sorted(bad_variants, key=lambda t: "+".join(t)):
            offenders = sorted(
                "/".join(ent) for ent, path in ent_axes.items()
                if path == var
            )
            shape_name = (
                _variant_to_shape(var).value
                if _variant_to_shape(var) is not None
                else ("+".join(var) if var else "scalar")
            )
            who = ", ".join(offenders[:10]) if offenders else "(unknown)"
            parts.append(f"{shape_name} authored by {who}")
        raise FlexToolConfigError(
            f"Parameter ({entity_class!r}, {parameter_name!r}) has rows "
            f"whose shape is not allowed: {'; '.join(parts)}.  Allowed "
            f"shapes: {_allowed_shape_names(allowed)}.  Edit the source "
            "database so every entity uses an allowed shape, or extend "
            "PARAM_ALLOWED_SHAPES."
        )
    key = (entity_class, parameter_name)
    if len(variants) > 1 and key not in _MIXED_SHAPE_LOGGED:
        _MIXED_SHAPE_LOGGED.add(key)
        from collections import Counter
        ent_axes = _entity_axes_map(df, ent_cols)
        counts = Counter(ent_axes.values())
        rendered = ", ".join(
            f"{('+'.join(path) if path else 'scalar')}: {n}"
            for path, n in sorted(counts.items(), key=lambda kv: "+".join(kv[0]))
        )
        _LOGGER.info(
            "%s.%s: mixed shapes {%s} -> (d,t)",
            entity_class, parameter_name, rendered,
        )


# ---------------------------------------------------------------------------
# Per-entity silent-label fallback (#167, #168) — Δ.17c-silent-per-entity
# ---------------------------------------------------------------------------


def _silent_entity_axes(
    source: "InputSource",
    entity_class: str,
    parameter_name: str,
    df: "pl.DataFrame",
    ent_cols: "list[str]",
    period_filter: "pl.DataFrame | None",
) -> "tuple[frozenset[tuple[str, ...]], dict[tuple[str, ...], str]]":
    """Per-entity classification of a depth-1 silent-label Map's axis, read from
    :meth:`InputSource.parameter_entity_label_paths`.

    Called by :func:`resolve_param_shape` only after the whole-frame probe
    (:func:`_disambiguate_shape_by_value_domain`) stayed ambiguous.  Returns ``(drop, axis_of)``:

    * ``drop`` — silent entities whose keys match NEITHER the active solve's periods nor its timesteps
      (:func:`_period_time_vocab`): they have no value in this solve, exactly like an entity that
      authored nothing.  The caller re-resolves the parameter on a view without them.
    * ``axis_of`` — for every OTHER silent entity, the axis its keys matched (``"period"`` or ``"t"``).
      A labelled entity (an explicit, non-silent ``index_name``) is never a key of this dict: its path
      length is not 1-and-silent, so it is outside this fallback entirely and keeps resolving through the
      normal structural / label path once the drop set (if any) has been removed.

    Both are empty when the source has no ``parameter_entity_label_paths`` (no fallback: the caller keeps
    today's ``None``), when no entity's value is a silent depth-1 map, or when the solve has neither a
    period nor a timestep vocabulary to classify against (*period_filter* without ``d`` or ``t``).

    One vectorized pass over *df* — a ``group_by(ent_cols)`` plus a value-domain membership test, never a
    Python loop per entity: the fallback runs once per roll of a rolling / stochastic solve, and a
    silent-label time map can carry thousands of rows across thousands of entities.  Entity dim columns
    are compared as Utf8 (the frame may be Enum-cast by the axis contract).

    Raises :class:`FlexDataIntegrityError` naming the entity when a silent entity's keys match BOTH
    vocabularies — the ambiguous case :meth:`SpineDbReader._classify_pt_level` already raises on for a
    single row; here the same rule is applied once per entity, over the union of its (possibly many) keys.
    """
    fn = getattr(source, "parameter_entity_label_paths", None)
    if fn is None:
        return frozenset(), {}
    try:
        paths = fn(entity_class, parameter_name)
    except KeyError:
        return frozenset(), {}
    silent_entities = [ent for ent, path in paths.items()
                       if len(path) == 1 and _normalise_label(path[0]) is None]
    if not silent_entities:
        return frozenset(), {}
    periods_set, timesteps_set = _period_time_vocab(period_filter)
    if periods_set is None and timesteps_set is None:
        return frozenset(), {}

    non_val_cols = [c for c in df.columns if c not in ent_cols and c != "value"]
    if not non_val_cols:
        return frozenset(), {}
    ent_schema = {c: pl.Utf8 for c in ent_cols}
    ent_df = pl.DataFrame([list(e) for e in silent_entities], schema=ent_schema, orient="row")
    keys = (
        df.lazy()
          .select(*(pl.col(c).cast(pl.Utf8) for c in ent_cols),
                  pl.coalesce([pl.col(c).cast(pl.Utf8) for c in non_val_cols]).alias("__key"))
          .join(ent_df.lazy(), on=ent_cols, how="semi")
          .filter(pl.col("__key").is_not_null())
          .group_by(ent_cols, maintain_order=True)
          .agg(pl.col("__key").unique().alias("__keys"))
          .collect()
    )
    periods = sorted(periods_set) if periods_set else []
    timesteps = sorted(timesteps_set) if timesteps_set else []
    classified = keys.with_columns(
        pl.col("__keys").list.eval(pl.element().is_in(periods)).list.any().fill_null(False).alias("__hits_d"),
        pl.col("__keys").list.eval(pl.element().is_in(timesteps)).list.any().fill_null(False).alias("__hits_t"),
    )
    both = classified.filter(pl.col("__hits_d") & pl.col("__hits_t"))
    if both.height > 0:
        bad = "/".join(str(v) for v in both.row(0)[:len(ent_cols)])
        raise FlexDataIntegrityError(
            f"Parameter ({entity_class!r}, {parameter_name!r}): entity {bad!r} has a silent-label Map "
            "whose keys match BOTH the period and the timestep vocabulary of this solve; cannot classify "
            "the axis.  Author the Map's index_name as 'period' or 'time'."
        )
    axis_of: "dict[tuple[str, ...], str]" = {}
    for row in classified.filter(pl.col("__hits_d") & ~pl.col("__hits_t")).select(ent_cols).iter_rows():
        axis_of[tuple(str(v) for v in row)] = "period"
    for row in classified.filter(pl.col("__hits_t") & ~pl.col("__hits_d")).select(ent_cols).iter_rows():
        axis_of[tuple(str(v) for v in row)] = "t"
    drop = frozenset(
        tuple(str(v) for v in row)
        for row in classified.filter(~pl.col("__hits_d") & ~pl.col("__hits_t")).select(ent_cols).iter_rows()
    )
    return drop, axis_of


def _per_entity_resolved(
    df: "pl.DataFrame",
    ent_cols: "list[str]",
    axis_of: "dict[tuple[str, ...], str]",
    allowed: "set[Shape]",
    entity_class: str,
    parameter_name: str,
) -> "ResolvedShape":
    """Split a depth-1 parameter frame's single, Spine-silent index column into ``period`` / ``t`` per
    entity, using ``axis_of`` (:func:`_silent_entity_axes`'s per-entity classification).

    Only reached without the reader's own axis Enums (:class:`InMemoryReader`; the reader-less
    ``_csv_reload`` fallback): with axis Enums, the recursive retry in :func:`resolve_param_shape` (after
    any ``drop`` entities are removed) has the reader classify every remaining row itself, through its own
    per-row classifier against the reader's GLOBAL vocabulary — which is at least as wide as the active
    solve's — so this helper never runs there.  By the time control reaches here, ``resolve_param_shape``
    has already established that the WHOLE parameter carries exactly one non-entity, non-value column
    (``len(resolved_labels) == 1``): every entity's value, silent or not, was unrolled into that single
    column by the legacy positional path, so a genuinely labelled entity (an explicit, differently-named
    index column) cannot coexist with a silent one here — it would make a second column and the resolver
    would never have reached this branch.  An entity outside ``axis_of`` is therefore always a plain
    scalar (no map at all): its one column is null and its own ``value`` is untouched.

    Builds the per-entity shape ``variants`` exactly as :meth:`SpineDbReader.parameter_shape_info` reports
    them on the reader path (``()`` / ``("period",)`` / ``("time",)``), validates them with
    :func:`_validate_shape_variants` and returns the union :class:`ResolvedShape` (``MAP_PERIOD_TIME``
    when both axes occur among the entities, otherwise the lone one), with the canonical column order
    (``period`` before ``t``) — never ``MAP_TIME_PERIOD``.

    Row order of *df* is preserved throughout (a ``maintain_order`` left join plus ``with_columns`` /
    ``select`` only — the LP-determinism memo); the caller's eventual ``.sort(...)`` in
    ``broadcast_to_period[_time]`` is unaffected either way.
    """
    non_val_cols = [c for c in df.columns if c not in ent_cols and c != "value"]
    shared_col = non_val_cols[0] if non_val_cols else None

    ent_schema = {c: pl.Utf8 for c in ent_cols}
    axis_rows = [[*e, a] for e, a in sorted(axis_of.items())]
    axis_lookup = (
        pl.DataFrame(axis_rows, schema={**ent_schema, "__axis": pl.Utf8}, orient="row")
        if axis_rows else pl.DataFrame(schema={**ent_schema, "__axis": pl.Utf8})
    )
    joined = (
        df.with_columns(pl.col(c).cast(pl.Utf8) for c in ent_cols)
          .join(axis_lookup, on=ent_cols, how="left", maintain_order="left")
    )
    shared = pl.col(shared_col) if shared_col is not None else pl.lit(None, dtype=pl.Utf8)
    out = joined.with_columns(
        period=pl.when(pl.col("__axis") == "period").then(shared).otherwise(None),
        t=pl.when(pl.col("__axis") == "t").then(shared).otherwise(None),
    ).select(*ent_cols, "period", "t", "value")

    return _axis_frame_resolved(out, ent_cols, allowed, entity_class, parameter_name)


def _axis_frame_resolved(
    out: "pl.DataFrame",
    ent_cols: "list[str]",
    allowed: "set[Shape]",
    entity_class: str,
    parameter_name: str,
) -> "ResolvedShape":
    """The :class:`ResolvedShape` of a frame ``[*ent_cols, period, t, value]`` whose rows carry a null
    ``period`` and / or ``t`` where their entity's value has no such level: each entity's shape (its non-null
    axes) is checked with :func:`_validate_shape_variants`, and the result is ``MAP_PERIOD_TIME`` when both
    axes occur, otherwise the lone one (or ``SCALAR``), with the canonical column order (``period`` before
    ``t``)."""
    variants: "set[tuple[str, ...]]" = set()
    for row in out.select(*ent_cols, "period", "t").unique(subset=ent_cols, maintain_order=True).iter_rows(
            named=True):
        path: list[str] = []
        if row["period"] is not None:
            path.append("period")
        if row["t"] is not None:
            path.append("time")
        variants.add(tuple(path))
    _validate_shape_variants(variants, allowed, out, tuple(ent_cols), entity_class, parameter_name)

    any_period = any(v[:1] == ("period",) for v in variants)
    any_time = any("time" in v for v in variants)
    if any_period and any_time:
        shape = Shape.MAP_PERIOD_TIME
        frame = out
    elif any_period:
        shape = Shape.MAP_PERIOD
        frame = out.select(*ent_cols, "period", "value")
    elif any_time:
        shape = Shape.MAP_TIME
        frame = out.select(*ent_cols, "t", "value")
    else:
        shape = Shape.SCALAR
        frame = out.select(*ent_cols, "value")

    return ResolvedShape(
        shape=shape,
        frame=frame,
        entity_dim_columns=tuple(ent_cols),
        period_index_column="period" if shape in (Shape.MAP_PERIOD, Shape.MAP_PERIOD_TIME) else None,
        time_index_column="t" if shape in (Shape.MAP_TIME, Shape.MAP_PERIOD_TIME) else None,
    )


def _unnamed_mixed_depths(source: "InputSource", entity_class: str,
                          parameter_name: str) -> "dict[int, list[tuple[str, ...]]] | None":
    """The entities by map depth when their maps have more than one depth (constants aside) and some level is
    unnamed -- the case the reader's single positional unroll cannot place; None otherwise (also for a source
    without per-entity label paths).  With every level named, the reader places each row by its labels."""
    fn = getattr(source, "parameter_entity_label_paths", None)
    if fn is None:
        return None
    try:
        paths = fn(entity_class, parameter_name)
    except KeyError:
        return None
    by_depth: "dict[int, list[tuple[str, ...]]]" = {}
    for entity, path in paths.items():
        by_depth.setdefault(len(path), []).append(entity)
    if len([d for d in by_depth if d > 0]) < 2 or not has_silent_labels(set(paths.values())):
        return None
    return by_depth


def _resolve_by_depth(
    source: "InputSource",
    entity_class: str,
    parameter_name: str,
    allowed: "set[Shape]",
    by_depth: "dict[int, list[tuple[str, ...]]]",
    period_filter: "pl.DataFrame | None",
) -> "ResolvedShape | None":
    """Resolve each depth group of *by_depth* on a view without the other groups' values (the existing rules
    per group: allow-list labels, value-domain probe, per-entity split) and union the groups into one
    ``[entities, period, t, value]`` :class:`ResolvedShape` (:func:`_axis_frame_resolved`).

    A group that resolves to nothing has no value in this solve (its keys are not periods or time steps of
    the solve), except without *period_filter*, where its unnamed levels cannot be told apart: that raises
    :class:`FlexDataIntegrityError` naming the entities.  When a one-level group is read over time next to
    deeper maps, the behaviour-change notice ``unnamed_maps_read_per_depth`` names those entities (before,
    their keys were read as periods)."""
    parts: list[pl.DataFrame] = []
    ent_cols: "list[str] | None" = None
    for depth in sorted(by_depth):
        others = frozenset(e for d, ents in by_depth.items() if d != depth for e in ents)
        view = _view_without(source, entity_class, parameter_name, others, "shape_depth")
        resolved = _resolve_param_shape(view, entity_class, parameter_name, period_filter)
        if resolved is None:
            if period_filter is None:
                names = ", ".join(repr(_entity_text(e)) for e in sorted(by_depth[depth])[:5])
                raise FlexDataIntegrityError(
                    f"{entity_class} {names}, parameter {parameter_name!r}: the value is a map with "
                    f"{_levels(depth)} without index names next to maps of other depths, and here there are no "
                    "solve periods or time steps to tell its levels apart. Name the index levels "
                    "('period' / 'time').")
            continue
        ent_cols = list(resolved.entity_dim_columns)
        frame = resolved.frame
        parts.append(frame.select(
            *(pl.col(c).cast(pl.Utf8) for c in ent_cols),
            (pl.col(resolved.period_index_column).cast(pl.Utf8) if resolved.period_index_column
             else pl.lit(None, dtype=pl.Utf8)).alias("period"),
            (pl.col(resolved.time_index_column).cast(pl.Utf8) if resolved.time_index_column
             else pl.lit(None, dtype=pl.Utf8)).alias("t"),
            "value"))
    if not parts or ent_cols is None:
        return None
    out = pl.concat(parts, how="vertical_relaxed")
    deeper = [d for d in by_depth if d > 1]
    if 1 in by_depth and deeper:
        one_level = pl.DataFrame([list(e) for e in by_depth[1]], schema={c: pl.Utf8 for c in ent_cols},
                                 orient="row")
        over_time = (out.join(one_level, on=ent_cols, how="semi")
                        .filter(pl.col("t").is_not_null() & pl.col("period").is_null())
                        .select(ent_cols).unique().sort(ent_cols))
        if over_time.height > 0:
            from flextool.engine import _behaviour_changes
            _behaviour_changes.notice(
                "unnamed_maps_read_per_depth", f"{entity_class} {parameter_name} values",
                [_entity_text(row) for row in over_time.iter_rows()])
    return _axis_frame_resolved(out, ent_cols, allowed, entity_class, parameter_name)


_AXIS_WORDS: "dict[tuple[bool, bool], str]" = {
    (True, True): "periods and time", (True, False): "periods", (False, True): "time",
}


def _shape_read_notice(source: "InputSource", entity_class: str, parameter_name: str,
                       resolved: "ResolvedShape") -> None:
    """Print once per (class, parameter) and run which axes the unnamed maps of the parameter were read
    over (their shape was taken from their depth or keys).  Price ladders are left out: their levels are
    placed by depth by definition."""
    if resolved.shape in _FACET_SHAPES or resolved.frame is None:
        return
    from flextool.engine import _behaviour_changes
    key = ("shape_read_from_depth", entity_class, parameter_name)
    if _behaviour_changes.already_warned(key):
        return
    fn = getattr(source, "parameter_entity_label_paths", None)
    if fn is None:
        return
    try:
        paths = fn(entity_class, parameter_name)
    except KeyError:
        return
    unnamed = [e for e, p in paths.items() if p and any(_normalise_label(lbl) is None for lbl in p)]
    if not unnamed:
        return
    ent_cols = list(resolved.entity_dim_columns)
    frame = resolved.frame
    has_p = (pl.col(resolved.period_index_column).is_not_null().any() if resolved.period_index_column
             else pl.lit(False))
    has_t = (pl.col(resolved.time_index_column).is_not_null().any() if resolved.time_index_column
             else pl.lit(False))
    wanted = pl.DataFrame([list(e) for e in unnamed], schema={c: pl.Utf8 for c in ent_cols}, orient="row")
    axes = (frame.lazy()
                 .with_columns(pl.col(c).cast(pl.Utf8) for c in ent_cols)
                 .join(wanted.lazy(), on=ent_cols, how="semi")
                 .group_by(ent_cols).agg(has_p.alias("p"), has_t.alias("t"))
                 .sort(ent_cols).collect())
    groups: "dict[str, list[str]]" = {}
    for row in axes.iter_rows(named=True):
        words = _AXIS_WORDS.get((bool(row["p"]), bool(row["t"])))
        if words is not None:
            groups.setdefault(words, []).append(_entity_text(tuple(row[c] for c in ent_cols)))
    if not groups:
        return
    read = []
    for words, names in groups.items():
        more = f" and {len(names) - 3} more" if len(names) > 3 else ""
        read.append(f"{', '.join(repr(n) for n in names[:3])}{more} over {words}")
    _behaviour_changes.warn_once(
        key, f"Note: {entity_class} '{parameter_name}' has maps without index names; read {'; '.join(read)} from "
             "their depth and keys. Name the index levels to make this explicit.")


def resolve_param_shape(
    source: "InputSource",
    entity_class: str,
    parameter_name: str,
    period_filter: "pl.DataFrame | None" = None,
) -> "ResolvedShape | None":
    """DB-driven shape detection + per-parameter allow-list validation.

    Steps:

    1. Look up :data:`PARAM_ALLOWED_SHAPES` for ``(entity_class,
       parameter_name)``.  Missing entry → :class:`FlexToolConfigError`
       (the registry must explicitly list every parameter routed
       through the resolver).
    2. Fetch the parameter frame via
       :meth:`InputSource.parameter_explicit` (falling back to
       :meth:`parameter`).  Empty / None → return ``None`` (the caller
       drops this parameter from FlexData).
    3. Read the raw per-level ``index_name`` labels from the DB.
    4. Map the labels to a :class:`Shape`.  Unrecognised labels OR a
       shape outside the allow-list → :class:`FlexToolConfigError`
       with full context (parameter name, observed labels, allowed
       shapes).

    When the index_name labels are silent-default (e.g. spinedb_api's
    ``"x"``) AND ``period_filter`` is supplied, the resolver probes the
    index column values against the active solve's known periods /
    timesteps (mirroring the legacy CSV pipeline's value-domain
    discrimination in
    :func:`flextool.engine.input._timeline.separate_period_and_timeseries_data`).
    This recovers Rivendell-style fixtures where ``group.co2_price`` is
    authored as ``Map(period→value)`` but the author omitted
    ``index_name`` so the DB stores the silent ``"x"`` label.  Without
    this probing the resolver returns ``None`` and ``p_co2_price`` is
    silently dropped while the feature gate (driven by topology, not
    data) stays active — :func:`flextool.engine.model.core.build_flextool`
    then skips the CO2 cost term with a warning.

    For a :data:`STOCHASTIC_TIME_PARAMS` parameter, the entities authoring a stochastic-time map
    (:func:`stochastic_time_entities`) are left out of the classification: the result describes the
    other entities (an empty ``SCALAR`` frame when there are none) and carries the stochastic ones in
    :attr:`ResolvedShape.stochastic_entities`.

    Per-entity silent-label fallback (#167, #168) — when the whole-frame value-domain probe above
    stays ambiguous (depth 1, the allow-list admits both ``MAP_PERIOD`` and ``MAP_TIME``, and
    *period_filter* is given), the resolver does not give up on the whole parameter: each silent entity
    is classified on its own against the solve's own periods / timesteps
    (:func:`_silent_entity_axes`).  An entity with no key in the solve has no value in this solve — the
    same as an entity that authored nothing at all — and is dropped (re-resolved on a view without it,
    recursively, until nothing more can be dropped); an entity whose keys match BOTH axes raises
    :class:`FlexDataIntegrityError` naming it, the same rule a single ambiguous row already raises on.
    This never changes the ``ResolvedShape | None`` contract below — it only changes which entities the
    returned shape (or the ``None``) describes.  Re-evaluated on every call (every roll of a rolling or
    stochastic solve may drop a different set, since *period_filter* is the roll's own window); only the
    view per distinct drop set is memoised (:func:`_view_without`), never the drop set itself.

    Returns ``None`` only when the parameter has no explicit rows AND no scalar default to broadcast
    (the source plugin's None-skip per §4.5), or when every entity was dropped by the per-entity fallback
    and the source has no default to broadcast over the rest.  Otherwise returns a :class:`ResolvedShape`
    carrying everything :func:`broadcast_to_period_time` / :func:`broadcast_to_period` need.

    Every entity's own label path is checked first (:func:`_check_entity_label_paths`): a depth the
    parameter does not take, or a ``period`` / ``time`` name on the wrong level, raises
    :class:`FlexToolConfigError` naming the entity.  When the entities' maps have different depths and some
    level is unnamed, each depth is read on its own (:func:`_resolve_by_depth`).  When a shape was read from
    an unnamed map's depth or keys, a one-line note says so (once per parameter and run).
    """
    resolved = _resolve_param_shape(source, entity_class, parameter_name, period_filter)
    if resolved is not None:
        _shape_read_notice(source, entity_class, parameter_name, resolved)
        if resolved.key != (entity_class, parameter_name):
            resolved = dataclasses.replace(resolved, key=(entity_class, parameter_name))
    return resolved


def _resolve_param_shape(
    source: "InputSource",
    entity_class: str,
    parameter_name: str,
    period_filter: "pl.DataFrame | None",
) -> "ResolvedShape | None":
    """:func:`resolve_param_shape` without the note."""
    key = (entity_class, parameter_name)
    allowed = PARAM_ALLOWED_SHAPES.get(key)
    if allowed is None:
        raise FlexToolConfigError(
            f"resolve_param_shape: parameter {key!r} is not in "
            f"PARAM_ALLOWED_SHAPES.  Add an explicit allow-list entry "
            f"to flextool/engine/input/_param_shapes.py.")

    stochastic = stochastic_time_entities(source, entity_class, parameter_name)
    if stochastic:
        # Stochastic-time entities come from the solve's emit; the other entities are resolved on a view
        # without them, so the period / time classifier never sees a depth-3 value.
        view = _stochastic_view(source, entity_class, parameter_name, stochastic)
        view_df = _try_parameter_frame(view, entity_class, parameter_name)
        if view_df is None or view_df.height == 0:
            ent_cols = _entity_dim_columns_for_frame(pl.DataFrame(), source, entity_class)
            resolved = ResolvedShape(
                shape=Shape.SCALAR,
                frame=pl.DataFrame(schema={**{c: pl.Utf8 for c in ent_cols}, "value": pl.Float64}),
                entity_dim_columns=tuple(ent_cols), period_index_column=None, time_index_column=None)
        else:
            resolved = resolve_param_shape(view, entity_class, parameter_name, period_filter=period_filter)
            if resolved is None:
                return None
        return dataclasses.replace(resolved, stochastic_entities=stochastic, stochastic_key=key)

    _check_entity_label_paths(source, entity_class, parameter_name, allowed)
    if not allowed & _FACET_SHAPES:
        by_depth = _unnamed_mixed_depths(source, entity_class, parameter_name)
        if by_depth:
            return _resolve_by_depth(source, entity_class, parameter_name, allowed, by_depth, period_filter)

    df = _try_parameter_frame(source, entity_class, parameter_name)
    if df is None or df.height == 0:
        return None

    ent_cols = _entity_dim_columns_for_frame(df, source, entity_class)
    raw_labels = _read_index_names_from_source(
        source, entity_class, parameter_name)

    # Phase A — registry-driven facet recognition.  The legacy
    # label-based ``_shape_from_indices`` can't see the new facet
    # shapes structurally (and shouldn't: facet outer-axis labels are
    # user-set per parameter value).  Try a label-agnostic depth match
    # against the allow-list first; on a hit, build the ResolvedShape
    # straight away — facet shapes share no columns with the legacy
    # period/time rename block, so the rest of the function (which
    # exists for legacy period/time disambiguation) does not apply.
    facet_shape = _recognise_facet_shape(raw_labels, allowed)
    if facet_shape is not None:
        # No column rename needed: the frame's outer-axis column names
        # are user-set and will be projected by the Phase B reader
        # using :func:`facet_keys` against the inner facet level.
        period_col = "period" if "period" in df.columns else None
        time_col = "t" if "t" in df.columns else None
        return ResolvedShape(
            shape=facet_shape,
            frame=df,
            entity_dim_columns=tuple(ent_cols),
            period_index_column=period_col,
            time_index_column=time_col,
        )

    # Δ.17c follow-up — disambiguate silent-default ``index_name`` labels
    # against the per-parameter allow-list.  For parameters whose
    # registry entry permits a unique shape at the observed (depth,
    # position), the silent default is unambiguous and we can recover
    # the author's intent without rewriting the DB.  Required to keep
    # the Spine path on parity with the legacy CSV pipeline, which
    # picks index dimensionality from value-domain probing (period
    # tokens vs. timestep tokens) rather than the index_name label —
    # see :func:`flextool.engine.input._timeline.separate_period_and_timeseries_data`.
    resolved_labels = _infer_silent_default_labels(raw_labels, allowed)
    try:
        shape = _shape_from_indices(resolved_labels)
    except _UnrecognisedIndex as exc:
        # User advice: "If does not match, error to the user."
        # We're strict only about *explicit* mismatches (e.g. ``branch`` on
        # a parameter whose allow-list is {scalar, period, time}).  Silent
        # defaults (empty / None / ``x``) collapse to ``shape=None`` below
        # via :func:`_shape_from_indices` and trigger the soft-fallback
        # path instead.
        raise FlexToolConfigError(
            f"Parameter ({entity_class!r}, {parameter_name!r}) carries an "
            f"unrecognised dimension index label {exc.label} (depth "
            f"{exc.depth}).  Allowed shapes: {_allowed_shape_names(allowed)}. "
            "Edit the source database so the Map's index_name reads "
            "'period' or 'time' (matching one of the allowed shapes), or "
            "extend PARAM_ALLOWED_SHAPES if a new shape is intended."
        ) from None

    if shape is None:
        # Ambiguous shape — at least one Map level carries the spinedb_api
        # silent default and the per-parameter allow-list permits multiple
        # interpretations at that depth.  Try value-domain probing against
        # the active solve's known periods / timesteps when *period_filter*
        # is supplied — this mirrors the legacy CSV pipeline's
        # ``separate_period_and_timeseries_data`` discriminator (a Map's
        # index values reveal whether it indexes by period or timestep,
        # regardless of the silent ``"x"`` index_name).
        shape = _disambiguate_shape_by_value_domain(
            df, ent_cols, resolved_labels, allowed, period_filter,
        )
        if shape is None:
            # Still ambiguous after the whole-frame probe.  Per-entity fallback (#167, #168): an entity
            # whose silent-label map has no key that is a period or a timestep of this solve has no value
            # in this solve — the same as an entity that authored nothing — so it is dropped and the rest
            # of the parameter is re-resolved without it; when the remaining silent entities split across
            # both axes (only reachable without the reader's own axis Enums — see
            # :func:`_per_entity_resolved`), the frame is built directly.  Runs only for the parameters
            # whose allow-list admits both ``MAP_PERIOD`` and ``MAP_TIME`` at depth 1 (the 21 keys of
            # :data:`PARAM_ALLOWED_SHAPES` that can reach this branch — Δ.17c-silent-per-entity) and only
            # with an active *period_filter* (the probe above already requires one; spelled out again here
            # because this is where the fallback actually changes behaviour).
            if period_filter is not None and {Shape.MAP_PERIOD, Shape.MAP_TIME} <= allowed:
                drop, axis_of = _silent_entity_axes(
                    source, entity_class, parameter_name, df, ent_cols, period_filter,
                )
                if drop:
                    view = _view_without(
                        source, entity_class, parameter_name, drop, "no_key_in_solve",
                    )
                    return resolve_param_shape(
                        view, entity_class, parameter_name, period_filter=period_filter,
                    )
                if axis_of:
                    return _per_entity_resolved(
                        df, ent_cols, axis_of, allowed, entity_class, parameter_name,
                    )
            # Nothing silent was classifiable (no period_filter, no ``parameter_entity_label_paths`` on
            # the source, or no entity at all matches one axis and not the other).  The depth is an allowed
            # one: a depth the parameter does not take was raised by _check_entity_label_paths above.
            # Per Δ.17c policy:
            # don't raise; return None so the caller falls back to its non-resolver pathway (typically the
            # seed CSV).  Δ.18+ punch-list: re-author fixtures so every Map level carries an explicit
            # ``index_name``.
            return None
        # Update the resolved labels so the downstream rename block sees
        # the disambiguated label and renames the silent-default column
        # (typically ``"x"``) to ``"period"`` or ``"t"``.
        resolved_labels = list(_SHAPE_LABELS[shape])

    # Phase A: depth-0 string-leaf promotion.  ``_shape_from_indices``
    # only sees the Map-nesting structure — every scalar (numeric or
    # string) collapses to :class:`Shape.SCALAR`.  When the registry
    # entry permits :class:`Shape.SCALAR_STR` but not :class:`Shape.SCALAR`,
    # the author's intent is a constrained-string leaf (per the
    # ``parameter_value_list``); promote so the allow-list check below
    # accepts it.  Leaf-type confirmation against the DB column dtype is
    # Phase B — today the structural depth-0 match plus the registry's
    # exclusivity unambiguously identifies these params.
    if (shape is Shape.SCALAR
            and Shape.SCALAR_STR in allowed
            and Shape.SCALAR not in allowed):
        shape = Shape.SCALAR_STR

    # Spec fix_map_reading.md §3.3 — per-variant allow-list check.  When
    # the source exposes ``parameter_shape_variants`` (SpineDbReader /
    # InMemoryReader, for PERIOD_TIME_PARAMS keys), validate each
    # per-row shape against ``allowed`` instead of the combined shape.
    # So a mixed period+time frame (combined ``MAP_PERIOD_TIME``) is
    # accepted for e.g. ``commodity.price`` — whose allow-list lacks
    # ``MAP_PERIOD_TIME`` — as long as every individual row shape is
    # allowed; a genuine 2-D row, or a time map on a period-only
    # parameter, is still rejected.  Falls back to the combined check
    # when the source doesn't expose variants (test stubs) or returns an
    # empty set (non-PERIOD_TIME keys).
    variants: "set[tuple[str, ...]] | None" = None
    variants_fn = getattr(source, "parameter_shape_variants", None)
    if variants_fn is not None:
        try:
            raw_variants = variants_fn(entity_class, parameter_name)
        except (KeyError, AttributeError):
            raw_variants = None
        if raw_variants:
            variants = {tuple(v) for v in raw_variants}
    if variants:
        _validate_shape_variants(
            variants, allowed, df, tuple(ent_cols),
            entity_class, parameter_name,
        )
    elif shape not in allowed:
        # Render observed shape for the message — labels can be empty
        # so reconstruct from the labels list.
        observed = (shape.value if shape is not None
                     else f"index_names={resolved_labels}")
        raise FlexToolConfigError(
            f"Parameter ({entity_class!r}, {parameter_name!r}) was authored "
            f"as {observed} but allowed shapes are: "
            f"{_allowed_shape_names(allowed)}.  Edit the source database "
            "or extend PARAM_ALLOWED_SHAPES."
        )

    # Δ.17c follow-up — when the resolver filled silent-default
    # ``index_name`` labels from the allow-list, the source frame's
    # columns still carry the silent-default names (e.g. ``"x"``) used
    # by :meth:`SpineDbReader._discover_index_cols`.  Rename them to the
    # canonical broadcast keys (``"period"`` / ``"t"``) so downstream
    # broadcasters can locate the index columns.
    df_for_broadcast = df
    if any(_normalise_label(r) is None for r in raw_labels):
        rename_map: "dict[str, str]" = {}
        # raw_labels and resolved_labels are aligned per depth.  The
        # SpineDbReader put non-entity, non-value columns in depth order;
        # walk them in parallel.
        non_ent_cols = [
            c for c in df.columns if c not in ent_cols and c != "value"
        ]
        for col, raw, resolved_lbl in zip(
            non_ent_cols, raw_labels, resolved_labels,
        ):
            if _normalise_label(raw) is not None:
                continue
            if resolved_lbl == "period" and col != "period":
                rename_map[col] = "period"
            elif resolved_lbl in ("time", "t") and col != "t":
                rename_map[col] = "t"
        if rename_map:
            df_for_broadcast = df.rename(rename_map)

    # Resolve the column names per the (possibly renamed) frame.
    period_col = "period" if "period" in df_for_broadcast.columns else None
    time_col = "t" if "t" in df_for_broadcast.columns else None
    return ResolvedShape(
        shape=shape,
        frame=df_for_broadcast,
        entity_dim_columns=tuple(ent_cols),
        period_index_column=period_col,
        time_index_column=time_col,
    )


# ---------------------------------------------------------------------------
# Facet-leaf multi-depth resolution (#169) — the commodity price ladders
# ---------------------------------------------------------------------------


def resolve_facet_groups(
    source: "InputSource",
    entity_class: str,
    parameter_name: str,
) -> "list[ResolvedShape]":
    """Resolve a facet-leaf parameter (the commodity price ladders) whose entities may author it at more
    than one depth: ``commodity.price_ladder_annual`` admits both the depth-2 form (``tier -> {price,
    quantity}``, the same ladder in every period) and the depth-3 form (``period -> tier -> {price,
    quantity}``).

    Reads each entity's raw label path via :meth:`InputSource.parameter_entity_label_paths` and groups
    entities by path LENGTH (depth) -- never by label, which is user-set per ladder value and tells
    nothing about depth.  With a single depth, returns ``[resolve_param_shape(source, ...)]`` unchanged
    (no view is built).  With several depths, each group is resolved on
    ``_view_without(source, ..., <every OTHER group's entities>, "facet_depth")``, so every view's frame
    is unrolled from exactly one depth -- the reader's "deepest-row-wins" positional unroll
    (:meth:`SpineDbReader._discover_index_cols`) never has to arbitrate between a depth-2 and a depth-3
    row of the SAME parameter, which is what made the depth-2 and ``x``-labelled forms unreadable before
    (the reader always discovered the depth-3 columns from the one deepest row and placed every other
    row's values into them positionally).

    Each element of the result is a facet :class:`ResolvedShape` (``Shape.MAP_TIER_FACET`` /
    ``Shape.MAP_PERIOD_TIER_FACET``).  Each entity's label path is checked first
    (:func:`_check_entity_label_paths`): a depth outside :data:`PARAM_ALLOWED_SHAPES`' allow-list, a level 1 of
    a 3-level ladder named other than ``period``, or a ``period`` / ``time`` name on a tier or price / quantity
    level raises ``FlexToolConfigError`` naming the entity.  Different depths are fine (split here).  A source without
    ``parameter_entity_label_paths`` (or that raises ``KeyError`` for this key) falls back to a single
    whole-frame :func:`resolve_param_shape` call -- the pre-existing behaviour for every reader that
    cannot report per-entity depths.
    """
    fn = getattr(source, "parameter_entity_label_paths", None)
    if fn is not None:
        try:
            paths = fn(entity_class, parameter_name)
        except KeyError:
            paths = None
    else:
        paths = None
    if paths is None:
        resolved = resolve_param_shape(source, entity_class, parameter_name)
        return [resolved] if resolved is not None else []
    allowed = PARAM_ALLOWED_SHAPES.get((entity_class, parameter_name))
    if allowed is not None:
        _check_entity_label_paths(source, entity_class, parameter_name, allowed)

    by_depth: "dict[int, list[tuple[str, ...]]]" = {}
    for ent, path in paths.items():
        by_depth.setdefault(len(path), []).append(ent)
    if len(by_depth) <= 1:
        resolved = resolve_param_shape(source, entity_class, parameter_name)
        return [resolved] if resolved is not None else []

    out: "list[ResolvedShape]" = []
    for depth in sorted(by_depth):
        others = frozenset(e for d, ents in by_depth.items() if d != depth for e in ents)
        view = _view_without(source, entity_class, parameter_name, others, "facet_depth")
        resolved = resolve_param_shape(view, entity_class, parameter_name)
        if resolved is not None:
            out.append(resolved)
    return out


def facet_long_frame(
    resolved: "ResolvedShape",
    period_filter: "pl.DataFrame | None",
) -> pl.DataFrame:
    """The long-form frame of one facet-leaf :class:`ResolvedShape`: ``[<entity cols>, (d), i, facet,
    value]`` -- the ``d`` column is present only for :class:`Shape.MAP_PERIOD_TIER_FACET`, or when
    *period_filter* cross-joins one in for a :class:`Shape.MAP_TIER_FACET` group (below).

    The axes are taken BY POSITION, never by label: the frame's non-entity, non-value columns are in
    depth order (the reader builds ``ent_cols + index_cols + value``, and nothing downstream reorders
    them).  For ``MAP_PERIOD_TIER_FACET`` the first three are ``(period, tier, facet)``; for
    ``MAP_TIER_FACET`` the first two are ``(tier, facet)``.  A facet-leaf Map's outer-level
    ``index_name`` is user-set per value (see :func:`_recognise_facet_shape`'s own docstring), so labels
    are never consulted here either.  Raises :class:`FlexToolConfigError` (an internal-error wording,
    not a user-input one) when the column count does not match the shape's expected depth.

    ``d`` and ``i`` are cast to Utf8 so a depth-2 group's cross-joined ``d`` (below) and a depth-3
    group's own, raw ``d`` carry the SAME dtype when a caller concatenates several groups' frames
    (:func:`flextool.engine.preprocess.params._direct._ladder_split` does, for a commodity that authors
    both depths on different tiers).

    When *period_filter* is given and the shape is ``MAP_TIER_FACET`` (depth 2: "the same ladder in
    every period"), the frame is cross-joined with *period_filter*'s own periods, EXCLUDING the active
    stochastic branch periods (:func:`flextool.engine.preprocess.params._branch.active_branch_anchor_pairs`)
    -- the emit's "same ladder in every period" rule, restricted to the solve's periods the way
    :func:`._direct._filter_param_by_periods` restricts it later anyway.  Branch periods are excluded so a
    depth-2 ladder matches its depth-3 equivalent and the vectorized emit seed: neither of those gives a
    ladder a row for a branch period today (tracked, not fixed, as design.md's O-2).
    """
    ent_cols = list(resolved.entity_dim_columns)
    df = resolved.frame
    if df is None:
        df = pl.DataFrame(schema={c: pl.Utf8 for c in ent_cols})
    non_val_cols = [c for c in df.columns if c not in ent_cols and c != "value"]

    if resolved.shape is Shape.MAP_PERIOD_TIER_FACET:
        if len(non_val_cols) != 3:
            raise FlexToolConfigError(
                f"facet_long_frame: internal error -- a MAP_PERIOD_TIER_FACET frame has "
                f"{len(non_val_cols)} non-entity column(s) (expected 3: period, tier, facet); "
                f"columns={non_val_cols!r}.")
        out = df.select(
            *ent_cols,
            pl.col(non_val_cols[0]).cast(pl.Utf8).alias("d"),
            pl.col(non_val_cols[1]).cast(pl.Utf8).alias("i"),
            pl.col(non_val_cols[2]).alias("facet"),
            "value",
        )
        return out

    if resolved.shape is Shape.MAP_TIER_FACET:
        if len(non_val_cols) != 2:
            raise FlexToolConfigError(
                f"facet_long_frame: internal error -- a MAP_TIER_FACET frame has {len(non_val_cols)} "
                f"non-entity column(s) (expected 2: tier, facet); columns={non_val_cols!r}.")
        out = df.select(
            *ent_cols,
            pl.col(non_val_cols[0]).cast(pl.Utf8).alias("i"),
            pl.col(non_val_cols[1]).alias("facet"),
            "value",
        )
        if period_filter is None or period_filter.height == 0:
            return out
        periods = period_filter.select(pl.col("d").cast(pl.Utf8)).unique(maintain_order=True)
        # Function-local import: input -> preprocess, kept out of the module import order (mirrors
        # ``_with_branch_periods`` / ``_period_time_vocab``).
        from flextool.engine.preprocess.params._branch import active_branch_anchor_pairs
        pairs = active_branch_anchor_pairs()
        if pairs is not None and pairs.height > 0:
            branch_members = pairs.get_column("br").cast(pl.Utf8).to_list()
            periods = periods.filter(~pl.col("d").is_in(branch_members))
        return (out.join(periods, how="cross")
                   .select(*ent_cols, "d", "i", "facet", "value"))

    raise FlexToolConfigError(
        f"facet_long_frame: unsupported shape {resolved.shape!r} (expected a facet-leaf shape).")


# ---------------------------------------------------------------------------
# Broadcast helpers — produce per-(entity, d, t) and per-(entity, d) frames
# ---------------------------------------------------------------------------


def _resolve_entity_dim_aliases(
    resolved: "ResolvedShape",
    entity_dim_aliases: "str | dict[str, str]",
    helper_name: str,
) -> "tuple[dict[str, str], tuple[str, ...]]":
    """Normalise the ``entity_dim_aliases`` argument into a
    ``(rename_map, target_keys)`` pair.

    Δ.17c-Tier3 — both :func:`broadcast_to_period_time` and
    :func:`broadcast_to_period` accept either a single string (the
    0-dim 1-source-column case — historical signature) or a per-source
    column rename dict (multi-dim relationship classes such as
    ``unit__inputNode``, ``reserve__upDown__group``).

    * ``str`` → renames the entity's sole dim column to that string.
      Defensive when ``resolved.entity_dim_columns`` is multi-dim.
    * ``dict[str, str]`` → per-source-column rename; the helper
      validates every entity dim column has an entry, then orders the
      target keys to match :attr:`ResolvedShape.entity_dim_columns` so
      the output Param's dim tuple is stable across runs (which is
      important for the deterministic LP-column assignment in
      :mod:`polar_high`).

    The returned ``target_keys`` are the post-rename column names in
    source-column order — the helper uses them as the first
    ``len(entity_dim_columns)`` columns of every ``select(...)`` and as
    the leading Param dims.
    """
    ent_cols = resolved.entity_dim_columns
    if isinstance(entity_dim_aliases, str):
        # 0-dim entity class — single "name" column → single alias.
        if ent_cols != ("name",):
            raise FlexToolConfigError(
                f"{helper_name}: entity_dim_aliases was passed as a single "
                f"string {entity_dim_aliases!r} but the entity class has "
                f"multi-dim columns {ent_cols}.  Pass a per-source-column "
                "rename dict instead."
            )
        return ({"name": entity_dim_aliases}, (entity_dim_aliases,))
    if not isinstance(entity_dim_aliases, dict):
        raise FlexToolConfigError(
            f"{helper_name}: entity_dim_aliases must be str or dict; got "
            f"{type(entity_dim_aliases).__name__}.")
    # Multi-dim: every source dim column must have a target name.
    missing = [c for c in ent_cols if c not in entity_dim_aliases]
    if missing:
        raise FlexToolConfigError(
            f"{helper_name}: entity_dim_aliases dict is missing rename "
            f"target(s) for source column(s) {missing}; entity dim "
            f"columns are {ent_cols}.")
    extra = [k for k in entity_dim_aliases if k not in ent_cols]
    if extra:
        raise FlexToolConfigError(
            f"{helper_name}: entity_dim_aliases dict has rename target(s) "
            f"for unknown source column(s) {extra}; entity dim columns "
            f"are {ent_cols}.")
    # Order target keys by source-column order for stable Param dims.
    target_keys = tuple(entity_dim_aliases[c] for c in ent_cols)
    return (dict(entity_dim_aliases), target_keys)


def _with_branch_periods(resolved: ResolvedShape) -> ResolvedShape:
    """``resolved`` with its period-keyed rows copied to the stochastic branch periods of their period.

    Runs on the raw source frame, before the null / zero filters, the period-filter join and the
    axis cast, so "a branch period's own row wins" is judged on authored rows.  Scalar and time-only
    shapes, and calls outside ``load_flextool`` (no active pairs), return ``resolved`` unchanged.
    See :func:`flextool.engine.preprocess.params._branch.inherit_active_branch_periods`.
    """
    if resolved.period_index_column is None:
        return resolved
    # Function-local import: input -> preprocess, kept out of the module import order.
    from flextool.engine.preprocess.params._branch import inherit_active_branch_periods

    frame = inherit_active_branch_periods(
        resolved.frame, group_keys=resolved.entity_dim_columns,
        period_col=resolved.period_index_column)
    if frame is resolved.frame:
        return resolved
    return dataclasses.replace(resolved, frame=frame)


def broadcast_to_period_time(
    resolved: "ResolvedShape | None",
    entity_dim_aliases: "str | dict[str, str]",
    period_filter: "pl.DataFrame | None",
    *,
    filter_zero: bool = False,
) -> "Param | None":
    """Resolve a source frame into a ``Param`` whose dims match the
    authored shape — **without** materialising the (entity, d, t)
    cross-product for sources that don't carry that axis natively.

    Phase E.1: the returned Param's dims depend on the resolved shape:

    ============================  ====================================
    Source shape                  Returned Param dims
    ============================  ====================================
    ``Shape.SCALAR``              ``(*entity_keys,)``
    ``Shape.MAP_PERIOD``          ``(*entity_keys, "d")``
    ``Shape.MAP_TIME``            ``(*entity_keys, "t")``
    ``Shape.MAP_PERIOD_TIME``     ``(*entity_keys, "d", "t")``
    ``Shape.MAP_TIME_PERIOD``     ``(*entity_keys, "d", "t")``
    ============================  ====================================

    ``entity_keys`` is derived from *entity_dim_aliases*:

    * ``str`` — the entity class is 0-dim (sole column ``"name"``); the
      string becomes the only entity key (backwards-compatible).
    * ``dict[str, str]`` — per-source-column rename for n-dim
      relationship classes; ``entity_keys`` is ordered by
      :attr:`ResolvedShape.entity_dim_columns` so the Param's dim tuple
      is deterministic across runs.

    The returned Param is fully lazy — no ``.collect()`` happens inside
    this helper.  ``polar_high.Param`` broadcasts the smaller-dim Params
    against ``(entity, d, t)``-keyed Vars at constraint-emission time
    via shared-dim inner joins on LazyFrames, so the dense cross-product
    only ever lives in the per-term collect that polar_high already
    streams to HiGHS one row at a time.

    *period_filter* must carry ``[d, t]`` columns (typically the active
    solve's ``flex_data.dt`` frame).  For the two-axis shapes
    (``MAP_PERIOD_TIME`` / ``MAP_TIME_PERIOD``) it restricts to active
    ``(d, t)`` pairs.  For ``MAP_PERIOD`` / ``MAP_TIME`` it restricts to
    the active periods / times respectively.  For ``SCALAR`` the filter
    is consulted only as the *gating* signal: when no period is active
    we return ``None`` (no Params produced for an empty solve), but no
    join is needed since SCALAR doesn't carry a d/t axis.

    Period-keyed rows of an anchor period also apply to its stochastic
    branch periods (module docstring, "Stochastic branch periods").

    *filter_zero* mirrors the CSV cascade's "drop rows where the
    explicit value is 0" semantic for fields like ``co2_price``
    (preprocessing emits zero-defaults but downstream gates filter
    them out).

    Stochastic-time entities (:attr:`ResolvedShape.stochastic_entities`) take the solve's folded values:

        out[k, d, t] = the result above for the other entities, on the solve's (d, t)   k not stochastic
                     = stochastic_time_values(...)[k, d, t]                             k stochastic
        ∀ (d, t) ∈ period_filter pairs;  filter_zero applies to both parts

    and the Param is then always ``(*entity_keys, "d", "t")`` (eager, sorted by its keys).  Without
    stochastic entities nothing changes.

    Returns ``None`` when:

    * *resolved* is ``None`` or its frame is empty (and it has no stochastic entities),
    * a required ``period_filter`` is missing or empty.
    """
    if resolved is not None and resolved.stochastic_entities:
        return _broadcast_with_stochastic_entities(resolved, entity_dim_aliases, period_filter,
                                                   filter_zero=filter_zero)
    if resolved is None or resolved.frame is None:
        return None
    resolved = _with_branch_periods(resolved)
    df = resolved.frame
    if df.height == 0:
        return None
    if period_filter is None or period_filter.height == 0:
        return None
    pf_cols = set(period_filter.columns)
    if not {"d", "t"}.issubset(pf_cols):
        return None

    # Δ.17c-Tier3 — multi-dim relationship classes (unit__inputNode,
    # reserve__upDown__group, …) supply a per-source-column rename dict;
    # 0-dim object classes keep the historical single-string signature.
    rename_map, entity_keys = _resolve_entity_dim_aliases(
        resolved, entity_dim_aliases, "broadcast_to_period_time")
    lf = (df.lazy()
            .pipe(rename_to_axis, rename_map)
            .filter(pl.col("value").is_not_null()))
    # A solve with new_stepduration: the time-keyed rows take one value per solve step by the parameter's
    # aggregation kind (_solve_step_aggregation).  Before the zero filter (a 0 hour is part of the step mean)
    # and before the axis cast (the original steps that are not solve-step labels are not in the vocabulary).
    # Only the steps of this solve's dt are aggregated (a roll pays for its own steps).
    if resolved.time_index_column is not None and active_base_steps() is not None:
        lf = aggregate_if_active(lf.collect(), resolved.key, time_col=resolved.time_index_column,
                                 steps=period_filter.get_column("t")).lazy()
    if filter_zero:
        lf = lf.filter(pl.col("value") != 0.0)
    # Phase 4 — align producer-side dim dtypes to the canonical Enum
    # vocabulary so the downstream join against ``period_filter`` (which
    # carries Enum d/t after :func:`apply_derived_a` runs) doesn't trip
    # over Utf8-vs-Enum on the ``t`` (or entity) axis.  The cascade
    # contract is "Enum when global vocabulary is active"; the resolver
    # reads frames in String land via ``InputSource.parameter*``, so the
    # cast happens here at the broadcast boundary.
    #
    # Failed-cast guard (spec fix_map_reading.md §3.4).  ``cast_frame_axes``
    # runs with ``strict=False``, so a period/time token that is NOT in the
    # model's Enum vocabulary (an unknown or inactive period / timestep)
    # silently becomes ``null`` on cast.  The null-pattern split below reads
    # a ``null`` axis token as a "missing axis" default and broadcasts that
    # row across the WHOLE axis — spreading a bogus value over every (d, t)
    # and producing duplicate / wrong keys (reproduced: a MAP_TIME row for
    # ``t01`` ended up carrying both 50 and 60).  Guard it: record which
    # rows carried a NON-null token BEFORE the cast, then after the cast
    # DROP the rows whose token was non-null before but null after (the cast
    # failures), and drive the null-pattern split off the PRE-cast masks —
    # never the post-cast nulls.  For data whose tokens are all valid/active
    # (the normal case) nothing is dropped and the pre-cast masks equal the
    # post-cast nulls, so this is a byte-for-byte no-op.
    _axis_cols = [c for c in ("period", "t") if c in lf.collect_schema().names()]
    _mask_cols = {c: f"__{c}_was_null" for c in _axis_cols}
    if _axis_cols:
        lf = lf.with_columns(
            pl.col(c).is_null().alias(_mask_cols[c]) for c in _axis_cols)
    _enums = get_global_axis_enums()
    if _enums is not None:
        lf = cast_frame_axes(lf, _enums)
        for c in _axis_cols:
            # Non-null before the cast, null after → unknown/inactive token
            # outside the model's vocabulary.  Drop it (do not let the split
            # treat it as a missing-axis default).
            _failed = (~pl.col(_mask_cols[c])) & pl.col(c).is_null()
            if _LOGGER.isEnabledFor(logging.DEBUG):
                _n = lf.select(_failed.sum().alias("n")).collect().item()
                if _n:
                    _LOGGER.debug(
                        "broadcast_to_period_time: dropped %d row(s) with "
                        "unknown/inactive '%s' token (failed Enum cast)",
                        _n, c)
            lf = lf.filter(~_failed)

    shape = resolved.shape
    if shape in (Shape.MAP_PERIOD_TIME, Shape.MAP_TIME_PERIOD):
        # (d, t)-keyed fill.  Both authoring orders unroll to the same
        # flat frame carrying ``period`` + ``t`` columns (the
        # SpineDbReader flattens a 2d_map regardless of index order), so
        # the two shapes share one code path.  Inner-join on dt to
        # restrict to the active solve's (d, t) grid.
        #
        # Mixed-authoring guard (mirrors the MAP_PERIOD / MAP_TIME
        # branches below): when SOME flowGroups author a full 2d
        # Map(period, time) while OTHERS author only a 1d Map(period)
        # (or a scalar) for the same parameter, the whole frame's
        # resolved shape becomes MAP_PERIOD_TIME — SpineDbReader's
        # ``parameter_shape_info`` reports the DEEPEST row's nesting and
        # ``_unroll_rows`` discovers index columns from the widest row,
        # so the shallower rows arrive with a NULL ``t`` (period-only
        # default) or NULL ``d`` (time-only default).  A plain inner-join
        # on (d, t) SILENTLY DROPS every such row — annihilating those
        # entities' floors/caps entirely (the reported ``min_instant_flow``
        # correctness bug: adding one period-time floor on a new
        # flowGroup made the pre-existing period-scalar floors on other
        # flowGroups vanish, LOWERING the optimum below the un-floored
        # baseline).  Split by which index axes are populated and
        # broadcast the missing axis across the active grid instead of
        # dropping.  For a non-mixed (pure 2d) frame every row is fully
        # keyed, so only the ``lf_full`` branch is non-empty and the
        # result is byte-identical to the old inner-join.
        dt_lf = period_filter.lazy().select("d", "t").unique()
        # Drive the four-way split off the PRE-cast null masks (§3.4) — a
        # post-cast null may be a failed cast, not an authored default.
        _d_null = pl.col(_mask_cols["period"])
        _t_null = pl.col(_mask_cols["t"])
        lf_dt = (lf.pipe(rename_to_axis, {"period": "d"})
                    .select(*entity_keys, "d", "t", "value",
                            _mask_cols["period"], _mask_cols["t"]))
        # Fully-keyed rows: exact (d, t) match against the active grid.
        lf_full = (lf_dt.filter(~_d_null & ~_t_null)
                        .join(dt_lf, on=["d", "t"], how="inner")
                        .select(*entity_keys, "d", "t", "value"))
        # Period-only default (null t): broadcast across every active
        # timestep of the matching active period.
        lf_period = (lf_dt.filter(~_d_null & _t_null)
                          .select(*entity_keys, "d", "value")
                          .join(dt_lf, on="d", how="inner")
                          .select(*entity_keys, "d", "t", "value"))
        # Time-only default (null d): broadcast across every active
        # period of the matching active timestep.
        lf_time = (lf_dt.filter(_d_null & ~_t_null)
                        .select(*entity_keys, "t", "value")
                        .join(dt_lf, on="t", how="inner")
                        .select(*entity_keys, "d", "t", "value"))
        # Scalar default (both null): broadcast across the whole grid.
        lf_scalar = (lf_dt.filter(_d_null & _t_null)
                          .select(*entity_keys, "value")
                          .join(dt_lf, how="cross")
                          .select(*entity_keys, "d", "t", "value"))
        out_lf = pl.concat([lf_full, lf_period, lf_time, lf_scalar])
        return Param((*entity_keys, "d", "t"), out_lf)
    elif shape == Shape.MAP_PERIOD:
        # 1d_map(period) → (entity, d) Param.  Phase E.1: do NOT
        # broadcast across ``t``; polar_high handles the (entity, d) →
        # (entity, d, t) broadcast lazily at constraint emission.
        #
        # Mixed authoring: some entities may carry a scalar default
        # (period column null) while others carry an explicit map.
        # SpineDbReader unifies these into a single frame with a
        # nullable index column; rows with null index represent the
        # entity's scalar default and must be broadcast across all
        # active periods — INNER JOIN on a null index would drop them
        # silently.  Split into scalar-default and explicit branches,
        # broadcast independently across the active-period universe,
        # then concatenate.
        d_lf = period_filter.lazy().select("d").unique()
        # Split off the PRE-cast null mask (§3.4): a post-cast null may be
        # an unknown/inactive period that failed the cast, not a scalar
        # default — those rows were already dropped above.
        _d_null = pl.col(_mask_cols["period"])
        lf_p = lf.pipe(rename_to_axis, {"period": "d"})
        lf_scalar = (lf_p.filter(_d_null)
                          .select(*entity_keys, "value")
                          .join(d_lf, how="cross")
                          .select(*entity_keys, "d", "value"))
        lf_explicit = (lf_p.filter(~_d_null)
                            .select(*entity_keys, "d", "value")
                            .join(d_lf, on="d", how="inner")
                            .select(*entity_keys, "d", "value"))
        out_lf = pl.concat([lf_explicit, lf_scalar])
        return Param((*entity_keys, "d"), out_lf)
    elif shape == Shape.MAP_TIME:
        # 1d_map(time) → (entity, t) Param.  Phase E.1: do NOT
        # broadcast across ``d``; polar_high handles the (entity, t) →
        # (entity, d, t) broadcast lazily at constraint emission.
        #
        # Same mixed-authoring guard as MAP_PERIOD above: rows whose
        # ``t`` index is null are scalar defaults for that entity and
        # must broadcast across all active times instead of being
        # dropped by the inner-join on ``t``.  Covers fixtures like
        # ``network_coal_wind_battery_co2_fullYear_availability`` where
        # ``coal_plant`` is MAP_TIME but ``wind_plant`` is scalar 0.7.
        t_lf = period_filter.lazy().select("t").unique()
        # Split off the PRE-cast null mask (§3.4): a post-cast null may be
        # an unknown/inactive timestep that failed the cast, not a scalar
        # default — those rows were already dropped above.
        _t_null = pl.col(_mask_cols["t"])
        lf_scalar = (lf.filter(_t_null)
                          .select(*entity_keys, "value")
                          .join(t_lf, how="cross")
                          .select(*entity_keys, "t", "value"))
        lf_explicit = (lf.filter(~_t_null)
                            .select(*entity_keys, "t", "value")
                            .join(t_lf, on="t", how="inner")
                            .select(*entity_keys, "t", "value"))
        out_lf = pl.concat([lf_explicit, lf_scalar])
        return Param((*entity_keys, "t"), out_lf)
    elif shape == Shape.SCALAR:
        # Phase E.1: scalar stays scalar — one row per entity, no
        # cross-join with (d, t).  polar_high broadcasts the (entity,)
        # Param against (entity, d, t)-keyed Vars at constraint
        # emission via a shared-dim inner-join on ``entity``.
        # ``period_filter`` already gated us above as the active-solve
        # signal; no further join needed here.
        out_lf = lf.select(*entity_keys, "value")
        return Param(tuple(entity_keys), out_lf)
    else:  # pragma: no cover — guarded by allow-list check.
        raise FlexToolConfigError(
            f"broadcast_to_period_time: unhandled shape {shape!r}")


def _broadcast_with_stochastic_entities(
    resolved: ResolvedShape,
    entity_dim_aliases: "str | dict[str, str]",
    period_filter: "pl.DataFrame | None",
    *,
    filter_zero: bool,
) -> "Param | None":
    """:func:`broadcast_to_period_time` for a resolved shape with stochastic-time entities (see there)."""
    if period_filter is None or period_filter.height == 0 or not {"d", "t"}.issubset(period_filter.columns):
        return None
    key = resolved.stochastic_key
    dim_cols = list(resolved.entity_dim_columns)
    _rename, target = _resolve_entity_dim_aliases(resolved, entity_dim_aliases, "broadcast_to_period_time")
    keys = list(target)
    pairs = period_filter.select("d", "t").unique(maintain_order=True)
    pairs_utf8 = pairs.select(pl.col("d").cast(pl.Utf8), pl.col("t").cast(pl.Utf8))
    stochastic = pl.DataFrame([list(e) for e in sorted(resolved.stochastic_entities)],
                              schema={c: pl.Utf8 for c in keys}, orient="row")
    parts: list[pl.DataFrame] = []
    direct = broadcast_to_period_time(
        dataclasses.replace(resolved, stochastic_entities=frozenset(), stochastic_key=None),
        entity_dim_aliases, period_filter, filter_zero=filter_zero)
    if direct is not None:
        parts.append(promote_param_to_dt(direct, pairs)
                     .select(*(pl.col(c).cast(pl.Utf8) for c in (*keys, "d", "t")),
                             pl.col("value").cast(pl.Float64))
                     .join(stochastic.lazy(), on=keys, how="anti")
                     .collect())
    folded = (stochastic_time_values(key, resolved.stochastic_entities, dim_cols)
              .rename(dict(zip(dim_cols, keys)))
              .join(pairs_utf8, on=["d", "t"], how="semi"))
    _warn_entities_without_fold(key, resolved.stochastic_entities, folded.select(keys))
    folded = folded.filter(pl.col("value").is_not_null())
    if filter_zero:
        folded = folded.filter(pl.col("value") != 0.0)
    parts.append(folded)
    out = pl.concat(parts)
    enums = get_global_axis_enums()
    if enums is not None:
        out = cast_frame_axes(out, enums)
    out = out.sort(*keys, "d", "t")
    if out.height == 0:
        return None
    return Param((*keys, "d", "t"), out)


def solve_years_represented(source: "InputSource") -> "pl.DataFrame | None":
    """``[name, period, value]`` of ``solve.years_represented`` (Utf8 ``name``, ``period`` in the source's dtype,
    Float64 ``value``) in the source's row order, whatever the map's index name (``period`` or Spine's silent ``x``); ``None`` when no
    solve has a value.

    The one reader of the parameter for the DB-direct consumers.  It resolves the index through
    :func:`resolve_param_shape` (allow-list ``{MAP_PERIOD}``), so a map left with the default index name counts
    the same as one named ``period``; a level named something else stops the run with the resolver's message.
    """
    resolved = resolve_param_shape(source, "solve", "years_represented")
    if resolved is None or resolved.frame is None or resolved.period_index_column is None:
        return None
    return resolved.frame.select(
        pl.col("name").cast(pl.Utf8),
        pl.col(resolved.period_index_column).alias("period"),
        pl.col("value").cast(pl.Float64, strict=False),
    )


def broadcast_to_period(
    resolved: "ResolvedShape | None",
    entity_dim_aliases: "str | dict[str, str]",
    period_filter: "pl.DataFrame | None",
    *,
    filter_zero: bool = False,
    filter_null: bool = True,
) -> "Param | None":
    """Resolve a source frame into a ``Param`` for parameters whose
    allow-list excludes time — keeping dims at the authored level.

    Phase E.1 (Δ.17c-Tier3 multi-dim extension):

    ====================  ====================================
    Source shape          Returned Param dims
    ====================  ====================================
    ``Shape.SCALAR``      ``(*entity_keys,)`` / ``(*entity_keys, "d")``
                          (Δ.17c-Tier1 cross-join — see below)
    ``Shape.MAP_PERIOD``  ``(*entity_keys, "d")``
    ====================  ====================================

    ``entity_keys`` is derived from *entity_dim_aliases* (see
    :func:`_resolve_entity_dim_aliases`):

    * ``str`` — the entity class is 0-dim (sole column ``"name"``); the
      string becomes the only entity key (backwards-compatible with all
      pre-Tier3 callers).
    * ``dict[str, str]`` — per-source-column rename for n-dim
      relationship classes; ``entity_keys`` is ordered by
      :attr:`ResolvedShape.entity_dim_columns` so the Param's dim tuple
      is deterministic across runs.

    The returned Param is fully lazy.  For ``MAP_PERIOD`` we inner-join
    on ``period_filter['d']`` to restrict to the active solve's periods.
    For ``SCALAR`` the filter is only consulted as the gating signal —
    no join is needed since a scalar doesn't carry a ``d`` axis.

    Period-keyed rows of an anchor period also apply to its stochastic
    branch periods (module docstring, "Stochastic branch periods").

    Other shapes raise :class:`FlexToolConfigError` (the resolver should
    have rejected them already; this is a defensive guard), as do
    stochastic-time entities (no period-only parameter takes them).
    """
    if resolved is not None and resolved.stochastic_entities:
        raise FlexToolConfigError(
            f"broadcast_to_period: {resolved.stochastic_key} has stochastic-time values, which have a time "
            "axis; read them with broadcast_to_period_time.")
    if resolved is None or resolved.frame is None:
        return None
    resolved = _with_branch_periods(resolved)
    df = resolved.frame
    if df.height == 0:
        return None

    # Δ.17c-Tier3 — multi-dim relationship classes supply a per-source-
    # column rename dict; 0-dim object classes keep the historical
    # single-string signature.  See :func:`_resolve_entity_dim_aliases`.
    rename_map, entity_keys = _resolve_entity_dim_aliases(
        resolved, entity_dim_aliases, "broadcast_to_period")
    lf = df.lazy().pipe(rename_to_axis, rename_map)
    if filter_null:
        lf = lf.filter(pl.col("value").is_not_null())
    if filter_zero:
        lf = lf.filter(pl.col("value") != 0.0)
    # Phase 4 — align producer-side dim dtypes to the canonical Enum
    # vocabulary so the downstream join against ``period_filter`` (which
    # carries Enum d after :func:`apply_derived_a` runs) doesn't trip
    # over Utf8-vs-Enum on the ``d`` (or entity) axis.  See the matching
    # comment in :func:`broadcast_to_period_time`.
    #
    # Failed-cast guard (item 65; mirrors the §3.4 guard of
    # :func:`broadcast_to_period_time`): ``cast_frame_axes`` runs with
    # ``strict=False``, so a ``period`` token outside the model's Enum
    # vocabulary becomes ``null``, and the MAP_PERIOD split below reads a
    # null ``d`` as a scalar default and broadcasts it over every active
    # period.  Drop the rows whose ``period`` was non-null before the cast
    # and null after (DEBUG log); never treat them as a scalar.
    #
    # The cast runs only while an axis-Enum vocabulary is live
    # (``provider.axis_enums``; production loads leave it None, so there
    # the explicit rows meet the solve's periods in the inner join below
    # and an inactive key drops out the same way).  The input check
    # (``validate_period_keys``) stops a mistyped key before this runs,
    # but valid data can still carry a key outside the Enum: a period of
    # another scenario's solve, or one only in ``model.periods_available``
    # (the Enum is the scenario's periods and branch periods).  Without
    # the guard such a row was spread as a scalar over every period, next
    # to the entity's own rows (duplicate (entity, d) keys).
    _has_period = "period" in lf.collect_schema().names()
    if _has_period:
        lf = lf.with_columns(pl.col("period").is_null().alias("__period_was_null"))
    _enums = get_global_axis_enums()
    if _enums is not None:
        lf = cast_frame_axes(lf, _enums)
        if _has_period:
            _failed = (~pl.col("__period_was_null")) & pl.col("period").is_null()
            if _LOGGER.isEnabledFor(logging.DEBUG):
                _n = lf.select(_failed.sum().alias("n")).collect().item()
                if _n:
                    _LOGGER.debug(
                        "broadcast_to_period: dropped %d row(s) with unknown/inactive "
                        "'period' token (failed Enum cast)", _n)
            lf = lf.filter(~_failed)
    if _has_period:
        lf = lf.drop("__period_was_null")

    shape = resolved.shape
    if shape == Shape.SCALAR:
        # Δ.17c-Tier1 — emit ``(entity, d)`` even for scalar sources by
        # cross-joining with the active solve's period axis.  Mirrors
        # the legacy ``_entity_period_scalar`` semantic (one
        # (entity, d, value) row per active period) so eager consumers
        # like :mod:`flextool.engine.model._cumulative_invest` (which
        # does ``cap_param.frame.select("g", "d")``) keep working.
        #
        # Phase E.1's "scalar stays (entity,)" optimisation was designed
        # for Vars/Params consumed exclusively through polar_high's
        # lazy broadcast.  The (entity, d) parameters declared on
        # :class:`FlexData` (e.g. ``p_co2_max_period``,
        # ``pd_max_cumulative_flow``) are documented with a ``(g, d)``
        # contract — emitting ``(entity, d)`` keeps the contract
        # consistent across SCALAR / MAP_PERIOD source shapes.
        if period_filter is None or period_filter.height == 0:
            return None
        d_lf = period_filter.lazy().select("d").unique()
        out_lf = (lf.select(*entity_keys, "value")
                    .join(d_lf, how="cross")
                    .select(*entity_keys, "d", "value"))
        return Param((*entity_keys, "d"), out_lf)
    elif shape == Shape.MAP_PERIOD:
        # Mixed authoring: some entities may carry a scalar default
        # (period column null) while others carry an explicit map.
        # SpineDbReader unifies these into a single frame with a
        # nullable index column; rows with null index represent the
        # entity's scalar default and must be broadcast across all
        # active periods — an INNER JOIN on a null index would drop
        # them silently.  Mirrors the matching split in
        # :func:`broadcast_to_period_time`'s MAP_PERIOD branch.
        #
        # Observed in practice: within a single group parameter one
        # entity authors its value as Map(period→…) while a sibling
        # carries a scalar (Map index null) — without this split the
        # scalar entity drops from the LP.
        lf_p = lf.pipe(rename_to_axis, {"period": "d"})
        if period_filter is not None and period_filter.height > 0:
            d_lf = period_filter.lazy().select("d").unique()
            lf_scalar = (lf_p.filter(pl.col("d").is_null())
                              .select(*entity_keys, "value")
                              .join(d_lf, how="cross")
                              .select(*entity_keys, "d", "value"))
            lf_explicit = (lf_p.filter(pl.col("d").is_not_null())
                                .select(*entity_keys, "d", "value")
                                .join(d_lf, on="d", how="inner")
                                .select(*entity_keys, "d", "value"))
            out_lf = pl.concat([lf_explicit, lf_scalar])
        else:
            # No period_filter — only explicit-period rows can be
            # carried; null-index scalars have no axis to broadcast on.
            out_lf = (lf_p.filter(pl.col("d").is_not_null())
                            .select(*entity_keys, "d", "value"))
        return Param((*entity_keys, "d"), out_lf)
    else:
        raise FlexToolConfigError(
            f"broadcast_to_period: shape {shape.value} is not supported "
            f"for (entity, period) parameters.  Allowed: "
            f"{_allowed_shape_names(_PERIOD_ONLY_SHAPES)}")


def promote_param_to_dt(
    param: "Param",
    dt: "pl.DataFrame | pl.LazyFrame",
) -> "pl.LazyFrame":
    """Return a LazyFrame view of ``param`` whose columns include both
    ``"d"`` and ``"t"`` — promoting via lazy joins on ``dt`` when the
    Param's authored shape is narrower.

    Phase E.1 makes flex_data Params keep their authored dims
    (``(entity,)`` / ``(entity, d)`` / ``(entity, t)`` / ``(entity, d,
    t)``).  Polar_high algebra handles the broadcast lazily at
    constraint emission, but a small number of consumers in the cascade
    do eager ``.frame.join(..., on=["x", "d", "t"], how="left")`` style
    densification (e.g. ``model/core.py`` flow_upper_rhs availability fold,
    ``_region_filter.py`` half-flow injection).  Those consumers need a
    (entity, d, t)-shaped LazyFrame on the right-hand side; this helper
    gives it to them without forcing the source Param to materialise
    eagerly.

    * ``(entity, d, t)`` Params — returned unchanged.
    * ``(entity, d)`` — inner-join on ``d`` against ``dt[d, t].unique()``.
    * ``(entity, t)`` — inner-join on ``t`` against ``dt[d, t].unique()``.
    * ``(entity,)`` — cross-join against ``dt[d, t].unique()``.
    """
    lf = param.lazy
    has_d = "d" in param.dims
    has_t = "t" in param.dims
    dt_lf = dt.lazy() if isinstance(dt, pl.DataFrame) else dt
    if has_d and has_t:
        return lf
    dt_sel = dt_lf.select("d", "t").unique()
    if has_d:  # missing t
        return lf.join(dt_sel, on="d", how="inner")
    if has_t:  # missing d
        return lf.join(dt_sel, on="t", how="inner")
    # missing both — scalar-per-entity broadcast over (d, t).
    return lf.join(dt_sel, how="cross")


def fill_default_over_grid(
    param: "Param | None",
    entities: pl.DataFrame,
    steps: pl.DataFrame,
    default: object,
    *,
    name: str,
) -> "Param | None":
    """``param`` over every cell of ``entities × steps``: the authored value where ``param`` has one, else
    ``default``.

        value[k] = param[k restricted to param.dims]    where param has a row
                 = default                              otherwise                ∀ k ∈ entities × steps

    ``entities`` holds the entity key column(s), ``steps`` the solve's ``(d, t)`` **pairs**
    (``dt.select("d", "t")``) or periods (``.select("d")``) — never ``d.unique() × t.unique()``: that product
    is not the timeline and explodes on long runs (CLAUDE.md invariant #2).  The grid is the unique sorted
    entities × the unique sorted steps (after the axis-enum cast; an entity-major cross join that keeps
    order), so the result is already in key order and needs no dedupe or sort of the product.
    ``param.dims`` must be a subset of the grid columns (any shape from :func:`broadcast_to_period` /
    :func:`broadcast_to_period_time`), so one left join on ``param.dims`` broadcasts the narrower shapes.
    ``param=None`` means nothing is authored.  A non-numeric or None ``default`` (no schema default)
    keeps only the grid cells that have an authored value (inner join).

    The authored rows must be unique on ``param.dims``: a duplicate would duplicate grid rows and every
    term built on them would count twice, so it raises :class:`FlexDataIntegrityError` naming ``name``.

    Returns an eager Param (dims = the entity columns then the step columns, rows sorted by them;
    consumers such as the Benders hosted-node checks and the warm-start compare read ``.frame`` as a
    DataFrame), or None when the result is empty.
    """
    from flextool.spinedb_backend._axis_enums import FlexDataIntegrityError

    enums = get_global_axis_enums()
    if enums is not None:
        entities = cast_frame_axes(entities, enums)
        steps = cast_frame_axes(steps, enums)
    entity_keys, step_keys = list(entities.columns), list(steps.columns)
    keys = entity_keys + step_keys
    grid = (entities.unique().sort(entity_keys)
            .join(steps.unique().sort(step_keys), how="cross", maintain_order="left_right"))
    numeric = isinstance(default, (int, float)) and not isinstance(default, bool)
    if param is None:
        if not numeric:
            return None
        out = grid.with_columns(pl.lit(float(default), dtype=pl.Float64).alias("value"))
    else:
        dims = list(param.dims)
        missing = [c for c in dims if c not in keys]
        if missing:
            raise ValueError(f"fill_default_over_grid({name}): Param dims {dims} are not all grid columns {keys}")
        authored = param.lazy.select(*dims, pl.col("value").cast(pl.Float64)).collect()
        dup = authored.select(dims).is_duplicated()
        if dup.any():
            first = authored.filter(dup).row(0, named=True)
            raise FlexDataIntegrityError(
                f"{name}: {int(dup.sum())} authored rows share their key on {dims} (e.g. {first}); each "
                "cell may carry one value.")
        if enums is not None:
            authored = cast_frame_axes(authored, enums)
        mismatched = [c for c in dims if authored.schema[c] != grid.schema[c]]
        if mismatched:
            # Cross-vocabulary dtypes on a key: compare on the grid's dtype; values outside it join nothing.
            authored = authored.with_columns(
                pl.col(c).cast(pl.Utf8).cast(grid.schema[c], strict=False) for c in mismatched)
        out = grid.join(authored, on=dims, how="left" if numeric else "inner", maintain_order="left")
        if numeric:
            out = out.with_columns(pl.col("value").fill_null(float(default)))
    out = out.select(*keys, "value")
    if out.height == 0:
        return None
    return Param(tuple(keys), out)


# ---------------------------------------------------------------------------
# Period coverage — a period map must give a value for every period of the solve
# ---------------------------------------------------------------------------

#: The periods a :class:`PeriodNeed` lists, by ``kind``: the noun of the error message and the closing rule.
_PERIOD_NEED_WORDING: dict[str, tuple[str, str]] = {
    "solve": ("period(s) {periods} used by {solve} (including look-ahead periods)",
              "every period of the solve, including its look-ahead periods"),
    "invest": ("investment period(s) {periods} of {solve}",
               "every investment period of the solve"),
    "online": ("period(s) {periods} in which investments of {solve} start to operate (construction lead time)",
               "every period in which an investment of the solve starts to operate"),
}


@dataclass(frozen=True)
class PeriodNeed:
    """The periods a limit's period map must cover in one solve (:func:`require_period_coverage`).

    * ``periods`` — one set for every entity, or ``{entity label: periods}`` when the periods differ per entity
      (an entity missing from the mapping needs none);
    * ``kind`` — what the periods are, for the error message: ``"solve"`` (every period of the solve, look-ahead
      included), ``"invest"`` (the investment periods) or ``"online"`` (the periods in which investments start
      to operate, with a construction lead time);
    * ``solve`` — the name of the active solve (a roll ``<solve>_roll_<n>`` is named as solve and roll), or
      ``None`` when not known.

    Falsy when no entity needs a period, which skips the check.
    """

    periods: "frozenset[str] | dict[str, frozenset[str]]"
    kind: str = "solve"
    solve: "str | None" = None

    def __post_init__(self) -> None:
        if self.kind not in _PERIOD_NEED_WORDING:
            raise ValueError(f"PeriodNeed.kind must be one of {sorted(_PERIOD_NEED_WORDING)}, not {self.kind!r}")

    def __bool__(self) -> bool:
        if isinstance(self.periods, dict):
            return any(self.periods.values())
        return bool(self.periods)

    def of(self, entity: str) -> "frozenset[str]":
        """The periods ``entity`` (an entity label) must cover."""
        if isinstance(self.periods, dict):
            return self.periods.get(entity, frozenset())
        return self.periods

    def solve_words(self) -> str:
        """``"solve 'a'"``, ``"solve 'a', roll 3"`` or ``"this solve"``."""
        return _solve_words(self.solve)


def _period_coverage_problems(
    resolved: ResolvedShape,
    out: "Param | None",
    entity_keys: "tuple[str, ...]",
    period_filter: pl.DataFrame,
    *,
    need: PeriodNeed,
    active: "set[str] | None",
) -> "list[tuple[str, list[str], list[str]]]":
    """``[(entity, missing periods, periods without a time step)]`` of the entities whose period map leaves out
    a needed period.

        need[k]    = { d ∈ period_filter.d : d ∈ need.of(k) }                  (period_filter order)
        mapped     = { k : resolved has a row of k with a non-null period } ∩ active
        stepped[k] = { d : a (period, time) map of k has a number in period d }
        have[k]    = { d : out[k, d(, t)] is not null and not NaN }
        lacking[k] = need[k] − have[k]                                          ∀ k ∈ mapped, lacking[k] ≠ ∅
        missing[k] = lacking[k] − stepped[k];  no_step[k] = lacking[k] ∩ stepped[k]

    ``no_step`` is a period a (period, time) map gives but none of whose time steps is in the solve.  A
    constant (a null-period row) applies to every period and is never checked.  ``out`` is the broadcast
    without the zero filter, so an authored 0 counts as a value.  Entity labels are the entity's dim values
    joined with ``", "`` (the bare name for a 0-dim class); ``active`` holds the same labels.
    """
    if resolved.frame is None or resolved.period_index_column is None:
        return []
    order = [str(d) for d in period_filter.get_column("d").cast(pl.Utf8).unique(maintain_order=True).to_list()]
    ent_cols = list(resolved.entity_dim_columns)
    label = pl.concat_str([pl.col(c).cast(pl.Utf8) for c in ent_cols], separator=", ").alias("__k")
    mapped = set(resolved.frame.filter(pl.col(resolved.period_index_column).is_not_null())
                 .select(label).get_column("__k").to_list())
    if active is not None:
        mapped &= set(active)
    mapped = {k for k in mapped if need.of(k)}
    if not mapped:
        return []
    # A period a (period, time) map gives with a number in it, but at none of the solve's time steps.
    stepped: dict[str, set[str]] = {}
    if resolved.time_index_column is not None and "value" in resolved.frame.columns:
        num = pl.col("value").cast(pl.Float64, strict=False)
        for k, d in (resolved.frame.filter(pl.col(resolved.period_index_column).is_not_null()
                                           & num.is_not_null() & num.is_not_nan())
                     .select(label, pl.col(resolved.period_index_column).cast(pl.Utf8).alias("__d"))
                     .iter_rows()):
            stepped.setdefault(k, set()).add(d)
    have: dict[str, set[str]] = {}
    if out is not None and "d" in out.dims:
        value = pl.col("value").cast(pl.Float64, strict=False)
        rows = (out.lazy
                .filter(value.is_not_null() & value.is_not_nan())
                .select(pl.concat_str([pl.col(c).cast(pl.Utf8) for c in entity_keys], separator=", ").alias("__k"),
                        pl.col("d").cast(pl.Utf8))
                .unique()
                .collect())
        for k, d in rows.iter_rows():
            have.setdefault(k, set()).add(d)
    problems = []
    for k in sorted(mapped):
        wanted = need.of(k)
        lacking = [d for d in order if d in wanted and d not in have.get(k, set())]
        if lacking:
            given = stepped.get(k, set())
            problems.append((k, [d for d in lacking if d not in given], [d for d in lacking if d in given]))
    return problems


def _raise_period_coverage(problems: "list[tuple[str, list[str], list[str]]]", *, entity_class: str,
                           parameter_name: str, need: PeriodNeed) -> None:
    """Raise one :class:`FlexToolConfigError` listing every entity of ``problems`` (no-op when empty).

    The message names the class, the entity, the parameter, the missing periods with what they are
    (``need.kind``) and the solve (and roll)."""
    if not problems:
        return
    noun, rule = _PERIOD_NEED_WORDING[need.kind]
    solve = need.solve_words()
    parts = []
    for k, missing, no_step in problems:
        what = []
        if missing:
            what.append("the period map has no value for " + noun.format(periods=", ".join(missing), solve=solve))
        if no_step:
            what.append(f"the period map gives period(s) {', '.join(no_step)} but none of their time steps is in "
                        f"{solve}")
        parts.append(f"{entity_class} '{k}', parameter '{parameter_name}': " + "; ".join(what))
    raise FlexToolConfigError(
        "; ".join(parts) + f". A period map must give a value for {rule} (a constant applies to all periods).")


def _drop_zero_rows(out: "Param | None") -> "Param | None":
    """``out`` without its rows of value 0 (None stays None)."""
    if out is None:
        return None
    return Param(out.dims, out.lazy.filter(pl.col("value") != 0.0))


def require_period_coverage(
    resolved: "ResolvedShape | None",
    entity_dim_aliases: "str | dict[str, str]",
    period_filter: "pl.DataFrame | None",
    *,
    entity_class: str,
    parameter_name: str,
    need_periods: "PeriodNeed | None",
    active: "set[str] | None" = None,
    filter_zero: bool = False,
) -> "Param | None":
    """:func:`broadcast_to_period`, after checking that every period map covers the periods it must.

        out        = broadcast_to_period(resolved, aliases, period_filter, filter_zero=False)
        missing[k] = (need_periods.of(k) ∩ period_filter.d) − { d : out[k, d] is a number }
                     ∀ entity k with a period map (∩ active)
        raise FlexToolConfigError naming entity_class, k, parameter_name, missing[k] and the solve
                                                                                    if any missing[k] ≠ ∅
        return out, without the rows of value 0 when filter_zero

    A period a limit applies to but its period map leaves out would otherwise get no row, or a null that a
    consumer reads as 0, so it is an error.  A constant applies to every period and is never checked.  A
    stochastic branch period without a value of its own inherits the value of its anchor in the broadcast
    (:func:`_with_branch_periods`), so it counts as covered.  The zero filter runs after the check, so an
    authored 0 is never read as a missing period.  ``need_periods`` (:class:`PeriodNeed`) is intersected with
    ``period_filter``'s periods, because the broadcast only keeps those; None or an empty need skips the check
    (e.g. a solve with no investment periods).  ``active`` (entity labels, see
    :func:`_period_coverage_problems`) limits the check to the entities whose limit is switched on; None
    checks every entity.  Pure: the caller works out the period sets, the gate and the solve name.
    """
    if not need_periods or period_filter is None or period_filter.height == 0:
        return broadcast_to_period(resolved, entity_dim_aliases, period_filter, filter_zero=filter_zero)
    out = broadcast_to_period(resolved, entity_dim_aliases, period_filter, filter_zero=False)
    if resolved is not None and resolved.frame is not None:
        _rename, keys = _resolve_entity_dim_aliases(resolved, entity_dim_aliases, "require_period_coverage")
        _raise_period_coverage(
            _period_coverage_problems(resolved, out, keys, period_filter, need=need_periods, active=active),
            entity_class=entity_class, parameter_name=parameter_name, need=need_periods)
    return _drop_zero_rows(out) if filter_zero else out


def require_period_coverage_time(
    resolved: "ResolvedShape | None",
    entity_dim_aliases: "str | dict[str, str]",
    period_filter: "pl.DataFrame | None",
    *,
    entity_class: str,
    parameter_name: str,
    need_periods: "PeriodNeed | None",
    active: "set[str] | None" = None,
    filter_zero: bool = False,
) -> "Param | None":
    """:func:`broadcast_to_period_time` with the period check of :func:`require_period_coverage`.

    Only the entities with a period level (a period map, or the period level of a period + time map) are
    checked, and only against the periods: a time map, a constant and the time steps inside a period are
    not.  The value of a (period, time) map covers a period when it has at least one number in it; a period
    the map gives but with none of the solve's time steps is reported as such.
    """
    if not need_periods or period_filter is None or period_filter.height == 0:
        return broadcast_to_period_time(resolved, entity_dim_aliases, period_filter, filter_zero=filter_zero)
    out = broadcast_to_period_time(resolved, entity_dim_aliases, period_filter, filter_zero=False)
    if resolved is not None and resolved.frame is not None:
        _rename, keys = _resolve_entity_dim_aliases(resolved, entity_dim_aliases, "require_period_coverage_time")
        _raise_period_coverage(
            _period_coverage_problems(resolved, out, keys, period_filter, need=need_periods, active=active),
            entity_class=entity_class, parameter_name=parameter_name, need=need_periods)
    return _drop_zero_rows(out) if filter_zero else out


# ---------------------------------------------------------------------------
# Time coverage — a time map must give a value for every time step of the solve
# ---------------------------------------------------------------------------

#: How many missing time steps an error message names (the rest are counted).
_TIME_COVERAGE_SHOWN = 5


def _solve_words(solve: "str | None") -> str:
    """``"solve 'a'"``, ``"solve 'a', roll 3"`` or ``"this solve"``."""
    if not solve:
        return "this solve"
    base, sep, roll = solve.rpartition("_roll_")
    if sep and base and roll.isdigit():
        return f"solve '{base}', roll {roll}"
    return f"solve '{solve}'"


def _time_coverage_problems(
    resolved: "ResolvedShape | None",
    steps: "pl.DataFrame | None",
    *,
    active: "set[str] | None",
) -> "list[tuple[str, list[str]]]":
    """``[(entity, missing steps)]`` of the entities whose time map leaves out a time step of ``steps``.

        need       = the (d, t) pairs of steps                                   (steps order)
        timed      = { k : resolved has a row of k with a time step } ∩ active
        miss[k]    = { t : (d, t) ∈ need } − { t : k's time map has a number at t }      k has a time map
                   = { (d, t) ∈ need : k's map gives d } − { (d, t) : a number at (d, t) }  period + time map
        ∀ k ∈ timed, miss[k] ≠ ∅

    A constant and a period map have no time level and are never checked.  A period that a period + time map
    does not give at all is left to the period check (:func:`require_period_coverage_time`, where a stochastic
    branch period inherits its anchor's value).  An authored 0 counts as a value, a null or NaN does not.
    ``steps`` are the original time steps (a solve with ``new_stepduration`` lists the time steps its solve
    steps combine).  A missing pair of a period + time map is named ``period/time``.  The time steps of a period
    that a period + time map does not give are not checked here (the inner join on ``d``): a stochastic branch
    period it leaves out has its anchor's time steps, which the anchor's own entry covers.
    """
    if (resolved is None or resolved.frame is None or resolved.time_index_column is None
            or steps is None or steps.height == 0):
        return []
    frame = resolved.frame
    tcol, pcol = resolved.time_index_column, resolved.period_index_column
    label = pl.concat_str([pl.col(c).cast(pl.Utf8) for c in resolved.entity_dim_columns], separator=", ")
    num = pl.col("value").cast(pl.Float64, strict=False) if "value" in frame.columns else pl.lit(None, pl.Float64)
    rows = (frame.filter(pl.col(tcol).is_not_null())
            .select(label.alias("__k"),
                    (pl.col(pcol).cast(pl.Utf8) if pcol is not None else pl.lit(None, pl.Utf8)).alias("__d"),
                    pl.col(tcol).cast(pl.Utf8).alias("__t"),
                    num.alias("__v")))
    if active is not None:
        rows = rows.filter(pl.col("__k").is_in(sorted(active)))
    if rows.height == 0:
        return []
    need = (steps.select(pl.col("d").cast(pl.Utf8).alias("__d"), pl.col("t").cast(pl.Utf8).alias("__t"))
            .unique(maintain_order=True).with_row_index("__i"))
    given = rows.filter(pl.col("__v").is_not_null() & pl.col("__v").is_not_nan())
    time_only = rows.filter(pl.col("__d").is_null()).select("__k").unique()
    missing_t = (time_only.join(need.select("__i", "__t"), how="cross")
                 .join(given.filter(pl.col("__d").is_null()).select("__k", "__t").unique(), on=["__k", "__t"],
                       how="anti")
                 .group_by("__k", "__t").agg(pl.col("__i").min())
                 .select("__k", "__i", pl.col("__t").alias("__what")))
    periods = rows.filter(pl.col("__d").is_not_null()).select("__k", "__d").unique()
    missing_dt = (periods.join(need, on="__d", how="inner")
                  .join(given.filter(pl.col("__d").is_not_null()).select("__k", "__d", "__t").unique(),
                        on=["__k", "__d", "__t"], how="anti")
                  .select("__k", "__i", pl.concat_str(["__d", "__t"], separator="/").alias("__what")))
    missing = pl.concat([missing_t, missing_dt]).sort("__k", "__i")
    return [(k, list(group.get_column("__what")))
            for (k,), group in missing.group_by("__k", maintain_order=True)]


def require_time_coverage(
    resolved: "ResolvedShape | None",
    steps: "pl.DataFrame | None",
    *,
    entity_class: str,
    parameter_name: str,
    active: "set[str] | None" = None,
    solve: "str | None" = None,
    hint: str = "",
) -> None:
    """Raise :class:`FlexToolConfigError` when a time map (or the time level of a period + time map) leaves out a
    time step of ``steps`` (``(d, t)``, the original time steps the solve uses; :func:`_time_coverage_problems`).

    The message names the class, each entity, the parameter, the first missing time steps with their count and
    the solve (``solve``: the solve or roll name, None for "this solve").  ``active`` (entity labels) limits the
    check to the entities whose limit is switched on; None checks every entity.  ``hint``: an extra clause of the
    message (what to give for a time step without a limit).  Pure: the caller works out the
    steps, the gate and the solve name.
    """
    problems = _time_coverage_problems(resolved, steps, active=active)
    if not problems:
        return
    where = _solve_words(solve)
    parts = []
    for k, missing in problems:
        shown = ", ".join(missing[:_TIME_COVERAGE_SHOWN])
        more = f" and {len(missing) - _TIME_COVERAGE_SHOWN} more" if len(missing) > _TIME_COVERAGE_SHOWN else ""
        parts.append(f"{entity_class} '{k}', parameter '{parameter_name}': the map has no value for "
                     f"{len(missing)} time step(s) used by {where}: {shown}{more}")
    raise FlexToolConfigError(
        "; ".join(parts) + ". A time map (or the time level of a period+time map) must give a value for every "
        "time step of the solve, look-ahead included (a constant applies to every time step"
        + (f"; {hint}" if hint else "") + ").")
