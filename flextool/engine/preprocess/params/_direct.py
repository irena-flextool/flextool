"""First wave of Direct Param helpers (Γ.1).

Each function in this module takes an :class:`InputSource` and returns
a single :class:`polar_high.Param` (or, for entity-only sets, a
:class:`polars.DataFrame`) — the trivial Direct port of one Param's
loader logic.  The body of each helper is at most:

* ``source.parameter(class, name)`` → polars frame,
* a column rename / dtype cast,
* return.

Lazy-evaluation pattern
-----------------------
``InputSource.parameter()`` already collects internally (the source
plugin is the boundary).  Helpers that compose multiple
``source.parameter()`` calls use ``.lazy()`` to chain operations
before a single ``.collect()`` at the end.  Single-call helpers stay
eager (no compositional benefit from going lazy).

The full sweep into ``flex_data.py`` is Γ.2/Γ.3; Γ.1 wires only the chosen
representative subset.  Each helper is here so a future helper-by-
helper migration replaces the corresponding CSV branch in ``flex_data.py``
with a one-line call to the function below.
"""
from __future__ import annotations

import contextlib
import contextvars
from typing import TYPE_CHECKING

import polars as pl

from polar_high import Param

from flextool.engine.input._axis_enums import (cast_frame_axes,
                                                 get_global_axis_enums,
                                                 rename_to_axis)
from flextool.engine.input._param_shapes import (
    PARAM_ALLOWED_SHAPES,
    Shape,
    allowed_shapes_in_words,
    broadcast_to_period,
    broadcast_to_period_time,
    facet_long_frame,
    fill_default_over_grid,
    PeriodNeed,
    promote_param_to_dt,
    require_period_coverage,
    require_period_coverage_time,
    require_time_coverage,
    resolve_facet_groups,
    resolve_param_shape,
)
from flextool.engine.input._solve_state import FlexToolConfigError
from flextool.engine.input._solve_step_aggregation import active_base_steps, base_steps_of
from flextool.engine.preprocess.emit._leaf_sets import _CO2_MAX_PERIOD_METHOD, _CO2_MAX_TOTAL_METHOD
from flextool.engine.preprocess.params._existing import (
    _DIVEST_PERIOD_METHODS,
    _INVEST_PERIOD_METHODS,
    _construction_lag_configured,
)
from flextool.engine.preprocess.params._projection import (
    _CUMULATIVE_METHODS,
    group_entity,
    groupCapacityMarginManual,
    groupCapacityMarginShareOfPeak,
    groupInertia,
    groupNonSync,
    p_ramp_cost_arc_from_source,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

    from flextool.engine.input._input_source import InputSource
    from flextool.engine.input._param_shapes import ResolvedShape


# ---------------------------------------------------------------------------
# §5.2.1 — object-class scalars (Direct, scalar default → broadcast)


def p_co2_content_from_source(source: "InputSource") -> Param | None:
    """``commodity.co2_content`` → ``Param(("c",), [c, value])``.

    Default ``None`` (§5.2.1) — None-skip per §4.5.  Helpers that
    depend on ``p_co2_content`` filter to entities with explicit values.
    """
    df = source.parameter("commodity", "co2_content")
    if df.height == 0:
        return None
    return Param(("c",),
                 df.lazy().rename({"name": "c"}).select("c", "value"))


def _periods_of(period_filter: "pl.DataFrame | None") -> set[str]:
    """The periods of ``period_filter`` (its ``d`` column) as strings; empty without a filter."""
    if period_filter is None or period_filter.height == 0 or "d" not in period_filter.columns:
        return set()
    return {str(d) for d in period_filter.get_column("d").cast(pl.Utf8).unique().to_list()}


def _gate_needed(need_periods: "PeriodNeed | None", resolved: "ResolvedShape | None") -> bool:
    """True when the period check of a reader has something to check: a need and a period map (a row with a
    period) in ``resolved``.  The gate of the limit (a method or flag read from the source) is built only
    then, so a solve or roll without a period map of the parameter does not rebuild it."""
    if not need_periods or resolved is None or resolved.frame is None or resolved.period_index_column is None:
        return False
    return bool(resolved.frame.get_column(resolved.period_index_column).is_not_null().any())


def _names_with_method(source: "InputSource", entity_class: str, parameter_name: str,
                       allowed: "frozenset[str]") -> set[str]:
    """The entities of ``entity_class`` whose ``parameter_name`` (a method) is one of ``allowed``."""
    try:
        df = source.parameter(entity_class, parameter_name)
    except KeyError:
        return set()
    if df.height == 0 or "value" not in df.columns:
        return set()
    return set(df.filter(pl.col("value").cast(pl.Utf8).is_in(list(allowed)))
                 .get_column("name").cast(pl.Utf8).to_list())


def _group_names(frame: pl.DataFrame) -> set[str]:
    """The ``g`` column of a group projection (``groupInertia`` …) as strings."""
    return set(frame.get_column("g").cast(pl.Utf8).to_list())


def p_constraint_constant_from_source(source: "InputSource",
                                      period_filter: pl.DataFrame | None = None,
                                      *,
                                      active_solve: str | None = None,
                                      ) -> Param | None:
    """``constraint.constant`` → ``Param(("cn",))`` when every constraint has a
    constant, else ``Param(("cn", "d"))``.

    Default ``0.0`` — broadcast to every constraint via the source.  A period
    map is resolved per constraint (``resolve_param_shape`` keeps the spread
    scalar default of the other constraints as null-period rows, which
    ``broadcast_to_period`` repeats over the solve's periods).  A period map
    must cover every period of ``period_filter``; a missing period is an
    error, as a user constraint has no other right-hand side for it
    (:func:`require_period_coverage`).

    The constraint axis uses the column name ``cn`` (not ``c``) to avoid
    collision with the commodity axis — see
    ``schemas/flextool_axis_contract.json`` review note ``c_collision``.
    """
    resolved = resolve_param_shape(source, "constraint", "constant", period_filter=period_filter)
    if resolved is None or resolved.frame is None or resolved.frame.height == 0:
        return None
    if resolved.shape == Shape.SCALAR:
        df = source.parameter("constraint", "constant")
        return Param(("cn",),
                     df.lazy().rename({"name": "cn"}).select("cn", "value"))
    out = require_period_coverage(resolved, "cn", period_filter, entity_class="constraint",
                                  parameter_name="constant",
                                  need_periods=PeriodNeed(frozenset(_periods_of(period_filter)), "solve",
                                                          active_solve))
    if out is None or period_filter is None or period_filter.height == 0:
        return out
    frame = out.frame.collect() if isinstance(out.frame, pl.LazyFrame) else out.frame
    return Param(("cn", "d"), frame.lazy())


def constraint_sense_from_source(source: "InputSource") -> pl.DataFrame:
    """Return the constraint sense set as ``[cn, sense]``.  Empty frame
    if no constraints carry a ``sense`` value (None-default policy).
    """
    df = source.parameter("constraint", "sense")
    if df.height == 0:
        return pl.DataFrame(schema={"cn": pl.Utf8, "sense": pl.Utf8})
    return df.lazy().rename({"name": "cn", "value": "sense"}).select("cn", "sense").collect()


# ---------------------------------------------------------------------------
# §5.2.7 — method discriminator scalars (string)


def node_node_type_from_source(source: "InputSource") -> pl.DataFrame:
    """``node.node_type`` → ``[n, value]`` with ``balance`` default
    broadcast to every node.
    """
    df = source.parameter("node", "node_type")
    if df.height == 0:
        return pl.DataFrame(schema={"n": pl.Utf8, "value": pl.Utf8})
    return df.lazy().rename({"name": "n"}).select("n", "value").collect()


def node_inflow_method_from_source(source: "InputSource") -> pl.DataFrame:
    """``node.inflow_method`` → ``[n, value]``.  None default — only
    nodes with an explicit inflow_method are returned.
    """
    df = source.parameter("node", "inflow_method")
    if df.height == 0:
        return pl.DataFrame(schema={"n": pl.Utf8, "value": pl.Utf8})
    return df.lazy().rename({"name": "n"}).select("n", "value").collect()


def unit_conversion_method_from_source(source: "InputSource") -> pl.DataFrame:
    """``unit.conversion_method`` → ``[p, value]``.  None default."""
    df = source.parameter("unit", "conversion_method")
    if df.height == 0:
        return pl.DataFrame(schema={"p": pl.Utf8, "value": pl.Utf8})
    return df.lazy().rename({"name": "p"}).select("p", "value").collect()


def connection_transfer_method_from_source(source: "InputSource") -> pl.DataFrame:
    """``connection.transfer_method`` → ``[p, value]``.  None default."""
    df = source.parameter("connection", "transfer_method")
    if df.height == 0:
        return pl.DataFrame(schema={"p": pl.Utf8, "value": pl.Utf8})
    return df.lazy().rename({"name": "p"}).select("p", "value").collect()


# ---------------------------------------------------------------------------
# Existing capacity (object-class scalars, per-class then unioned for `e`)


def _union_existing(source: "InputSource", classes: list[str]) -> pl.DataFrame:
    """Union ``existing`` across the supplied entity classes into one
    long ``[e, value]`` frame.  Returns an empty frame if no class
    contributes a row.
    """
    parts: list[pl.LazyFrame] = []
    for cls in classes:
        try:
            df = source.parameter(cls, "existing")
        except KeyError:
            continue
        if df.height == 0:
            continue
        parts.append(df.lazy().rename({"name": "e"}).select("e", "value"))
    if not parts:
        return pl.DataFrame(schema={"e": pl.Utf8, "value": pl.Float64})
    return pl.concat(parts).collect().sort("e")


def existing_long_from_source(source: "InputSource") -> pl.DataFrame:
    """Per-(entity, value) ``existing`` across unit/node/connection.
    Mirrors flextool's preprocessed ``p_entity_unitsize.csv`` indirectly
    — but here we expose just the raw scalar.
    """
    return _union_existing(source, ["unit", "node", "connection"])


# ---------------------------------------------------------------------------
# Membership relationships (entities-only, no parameter)


def commodity_node_set_from_source(source: "InputSource") -> pl.DataFrame:
    """``commodity__node`` membership set.  Schema: ``[commodity, node]``."""
    return source.entities("commodity__node")


def unit_input_node_set_from_source(source: "InputSource") -> pl.DataFrame:
    """``unit__inputNode`` membership set.  Schema: ``[unit, node]``."""
    return source.entities("unit__inputNode")


def unit_output_node_set_from_source(source: "InputSource") -> pl.DataFrame:
    """``unit__outputNode`` membership set.  Schema: ``[unit, node]``."""
    return source.entities("unit__outputNode")


def connection_node_node_set_from_source(source: "InputSource") -> pl.DataFrame:
    """``connection__node__node`` membership set.  Schema:
    ``[connection, node_1, node_2]`` (the dim suffix disambiguates the
    repeated ``node`` dim class).
    """
    return source.entities("connection__node__node")


def reserve_upDown_group_set_from_source(source: "InputSource") -> pl.DataFrame:
    """``reserve__upDown__group`` membership set.  Schema:
    ``[reserve, upDown, group]``.
    """
    return source.entities("reserve__upDown__group")


# ---------------------------------------------------------------------------
# §5.2.3 — relationship 1d_map / scalar (conversion_flow_coeff, etc.)


def unit_input_conversion_flow_coeff_from_source(source: "InputSource") -> Param | None:
    """``unit__inputNode.conversion_flow_coeff`` → ``Param(("p","source"), …)``.

    Default 1.0 — broadcast over all (unit, node) entities.
    """
    df = source.parameter("unit__inputNode", "conversion_flow_coeff")
    if df.height == 0:
        return None
    return Param(("p", "source"),
                 df.lazy().rename({"unit": "p", "node": "source"})
                          .select("p", "source", "value"))


def unit_output_conversion_flow_coeff_from_source(source: "InputSource") -> Param | None:
    """``unit__outputNode.conversion_flow_coeff`` → ``Param(("p","sink"), …)``."""
    df = source.parameter("unit__outputNode", "conversion_flow_coeff")
    if df.height == 0:
        return None
    return Param(("p", "sink"),
                 df.lazy().rename({"unit": "p", "node": "sink"})
                          .select("p", "sink", "value"))


# ---------------------------------------------------------------------------
# Module catalog (used by the parity test scaffolding to iterate over
# the first-wave Params).  The shape of each entry is:
#
#   (logical_name, callable_returning_frame, fixture_csv_extractor) where
#
#   - logical_name: stable label for parametrized test ids.
#   - callable_returning_frame: takes an InputSource → frame.
#   - fixture_csv_extractor: takes the CSV-loaded ``FlexData`` →
#     the corresponding frame (or None) for parity comparison.
#
# Only entries whose CSV-side has a clean structural mirror are listed
# here.  Helpers without a direct ``FlexData`` field (e.g. method
# discriminator strings consumed by Projection) are exercised by the
# unit-test suite via :class:`InMemoryReader` instead.

# Imported lazily inside the test module to avoid a circular import.

FIRST_WAVE_PARAMS = (
    "p_co2_content",
    "p_constraint_constant",
    "node_type",
    "inflow_method",
    "conversion_method",
    "transfer_method",
    "penalty_up",
    "penalty_down",
    "commodity_node_set",
    "unit_inputNode_set",
    "unit_outputNode_set",
    "connection_node_node_set",
    "reserve_upDown_group_set",
    "unit_input_flow_coef",
    "unit_output_flow_coef",
)


# ---------------------------------------------------------------------------
# Helpers for FlexData-field-targeted overrides


# Accepted ``Map.index_name`` labels for the per-entity constraint-coef
# parameters.  ``"constraint"`` is what ``model_builder`` writes; ``"x"``
# is spinedb_api's default when a Map is authored without an explicit
# ``index_name``.  Anything else indicates a Spine authoring mistake
# (typo, wrong axis label) and is surfaced loudly with the offending
# entity rows so the user can fix the DB.
_CONSTRAINT_INDEX_LABELS = ("constraint", "x")


def _resolve_constraint_index_col(
    df: pl.DataFrame,
    *,
    parameter_name: str,
    entity_class: str,
) -> str:
    """Identify the Map's constraint-axis column in *df* (the frame
    returned by :meth:`InputSource.parameter`).

    Raises a clear ``ValueError`` naming the parameter, entity_class,
    offending column name, and the entity rows that carry the bad Map
    so the user can locate and fix it in Spine.
    """
    extras = [c for c in df.columns if c not in ("name", "value")]
    if len(extras) == 1 and extras[0] in _CONSTRAINT_INDEX_LABELS:
        return extras[0]
    accepted = ", ".join(repr(s) for s in _CONSTRAINT_INDEX_LABELS)
    entities = df.get_column("name").unique().to_list() if "name" in df.columns else []
    entities_repr = ", ".join(repr(e) for e in entities[:10])
    if len(entities) > 10:
        entities_repr += f", … (+{len(entities) - 10} more)"
    raise ValueError(
        f"Unrecognised Map index column(s) {extras!r} in parameter "
        f"{parameter_name!r} of entity_class {entity_class!r} "
        f"(affected entities: [{entities_repr}]). "
        f"Expected a 1-D Map with index_name in {{{accepted}}}; "
        f"set Map.index_name to 'constraint' in Spine."
    )


def _node_constraint_coef(source: "InputSource", parameter_name: str) -> Param | None:
    """Return ``Param(("n", "cn"), [n, cn, value])`` for a per-(node,
    constraint) coefficient parameter.  ``None`` if the DB has no rows.

    The constraint axis column is named ``cn`` to disambiguate from the
    commodity axis column ``c`` (the contract's ``c_collision`` decision).
    """
    try:
        df = source.parameter("node", parameter_name)
    except KeyError:
        # Older DBs may not define the parameter on the ``node`` class
        # at all (rather than just having zero rows).  Treat the same.
        return None
    if df.height == 0:
        return None
    c_col = _resolve_constraint_index_col(
        df, parameter_name=parameter_name, entity_class="node")
    return Param(
        ("n", "cn"),
        df.lazy()
          .rename({"name": "n", c_col: "cn"})
          .select("n", "cn", "value")
    )


def _process_constraint_coef(source: "InputSource",
                              parameter_name: str) -> Param | None:
    """Union the per-(process, constraint) coefficient across the unit +
    connection classes (the ``process`` superclass in flextool's CSV
    output).  ``None`` if neither contributes a row.
    """
    parts: list[pl.LazyFrame] = []
    for cls in ("unit", "connection"):
        try:
            df = source.parameter(cls, parameter_name)
        except KeyError:
            continue
        if df.height == 0:
            continue
        c_col = _resolve_constraint_index_col(
            df, parameter_name=parameter_name, entity_class=cls)
        parts.append(
            df.lazy()
              .rename({"name": "p", c_col: "cn"})
              .select("p", "cn", "value")
        )
    if not parts:
        return None
    return Param(("p", "cn"), pl.concat(parts).sort("p", "cn"))


def _entity_methods_pairs(source: "InputSource") -> set[tuple[str, str]]:
    """Mirror ``entity__invest_method.csv`` — return ``{(e, method)}``.

    Spine source: ``unit/node/connection.invest_method``.  Empty when
    no entity has an explicit method.
    """
    out: set[tuple[str, str]] = set()
    for cls in ("unit", "node", "connection"):
        try:
            df = source.parameter(cls, "invest_method")
        except KeyError:
            continue
        if df.height == 0:
            continue
        for e, m in df.select("name", "value").iter_rows():
            out.add((str(e), str(m)))
    return out


# Mirror ``invest_method_sets.py:22-27``.
_INVEST_METHOD_NOT_ALLOWED = frozenset((
    "not_allowed", "retire_period", "retire_total", "retire_no_limit",
))
_DIVEST_METHOD_NOT_ALLOWED = frozenset((
    "not_allowed", "invest_period", "invest_total", "invest_no_limit",
))


def _entity_invest_universe(source: "InputSource",
                              kind: str = "invest") -> list[str]:
    """Return ``entityInvest`` (or ``entityDivest``) — entities whose
    invest_method is NOT in the not-allowed set.  Ordered by class
    encounter (unit → node → connection), deduplicated, matching
    ``invest_method_sets.py:30-46``.
    """
    not_allowed = (_INVEST_METHOD_NOT_ALLOWED if kind == "invest"
                   else _DIVEST_METHOD_NOT_ALLOWED)
    seen: dict[str, None] = {}
    for cls in ("unit", "node", "connection"):
        try:
            df = source.parameter(cls, "invest_method")
        except KeyError:
            continue
        if df.height == 0:
            continue
        for e, m in df.select("name", "value").iter_rows():
            if str(m) not in not_allowed:
                seen.setdefault(str(e), None)
    return list(seen.keys())


def _e_total_param(source: "InputSource", parameter_name: str,
                    kind: str = "invest",
                    filter_zero: bool = False) -> Param | None:
    """Per-entity scalar parameter (e.g. ``invest_max_total``) keyed on
    ``entityInvest`` (or ``entityDivest`` for divest variants), with 0
    default for entities with no explicit row.

    Mirrors flextool's ``entity_total_caps.py:_compute_entity_total`` —
    a row is emitted for every entity in ``entityInvest`` (resp.
    ``entityDivest``), value defaults to 0 when neither
    ``unit/node/connection`` carries an explicit row.

    When ``filter_zero=True``, rows with ``value == 0`` are dropped
    (mirrors ``flex_data.py::_read_e_param`` for ``invest_min_total`` /
    ``retire_min_total`` which use a different reader that filters
    zeros).  Returns ``None`` when no rows survive (or
    ``entityInvest`` is empty).
    """
    keys = _entity_invest_universe(source, kind=kind)
    if not keys:
        return None
    explicit: dict[str, float] = {}
    for cls in ("unit", "node", "connection"):
        try:
            df = source.parameter_explicit(cls, parameter_name)
        except (KeyError, AttributeError):
            try:
                df = source.parameter(cls, parameter_name)
            except KeyError:
                continue
        if df is None or df.height == 0:
            continue
        for e, v in df.select("name", "value").iter_rows():
            explicit[str(e)] = float(v) if v is not None else 0.0
    rows = [(e, explicit.get(e, 0.0)) for e in keys]
    if filter_zero:
        rows = [(e, v) for e, v in rows if v != 0.0]
    if not rows:
        return None
    out = pl.DataFrame(rows, schema=["e", "value"], orient="row").sort("e")
    return Param(("e",), out)


# ---------------------------------------------------------------------------
# Δ.4 — second-wave Direct Param helpers.
#
# Each helper below covers a single FlexData field whose CSV-loader path
# in ``flex_data.py`` is a one-CSV ``rename + select`` of an entity scalar /
# Map / 1d_map parameter.  Helpers return ``Param | None`` (None when
# the DB-side has no explicit row, matching the CSV path's "absent →
# None" contract).  The CSV-side behaviour preserved verbatim — the
# parity tests must continue to pass.

def _entity_scalar_explicit(source: "InputSource", entity_class: str,
                              parameter_name: str,
                              dim: str) -> Param | None:
    """Return ``Param((dim,), [<dim>, value])`` for an entity-class
    scalar, mirroring the CSV path's "explicit rows only" semantics.

    Tolerates a Spine schema that doesn't declare the parameter on the
    class (older fixtures): ``KeyError`` from
    :meth:`InputSource.parameter_explicit` / :meth:`parameter` is
    treated as "no rows".
    """
    try:
        df = source.parameter_explicit(entity_class, parameter_name)
    except (KeyError, AttributeError):
        try:
            df = source.parameter(entity_class, parameter_name)
        except KeyError:
            return None
    if df is None or df.height == 0:
        return None
    return Param((dim,),
                 df.lazy().rename({"name": dim}).select(dim, "value"))


# §5.2.5 — node scalars sliced from p_node.csv
def p_state_self_discharge_from_source(source: "InputSource",
                                      period_filter: pl.DataFrame | None = None,
                                      ) -> Param | None:
    """``node.self_discharge_loss`` → the loss per hour of each storage step, a ``Param`` over ``n`` and the
    authored axes (scalar ``(n,)``, period map ``(n, d)``, time map ``(n, t)``, period-time ``(n, d, t)``;
    ``_self_discharge.loss_coefficients`` joins it on the axes it has).  None without an active period.

    Uses :func:`._param_shapes.resolve_param_shape` → ``broadcast_to_period_time`` (shapes: scalar /
    1d_map[period] / 1d_map[time] / 2d_map[period,time]; stochastic branch periods inherit their anchor).
    Explicit rows only (the schema default is None: no loss); a step a map leaves out has no row (no
    loss), and an entity with no key in the active solve (the silent-label per-entity fallback, #168)
    is absent entirely, which is the same "no loss" outcome.
    """
    resolved = resolve_param_shape(source, "node", "self_discharge_loss", period_filter=period_filter)
    return broadcast_to_period_time(resolved, "n", period_filter)


def p_state_start_from_source(source: "InputSource") -> Param | None:
    """``node.storage_state_start`` → ``Param(("n",), [n, value])``.

    Default ``None`` (schema).  Returns explicit rows only.
    """
    return _entity_scalar_explicit(source, "node", "storage_state_start", "n")


def _storage_behaviour_notices(source: "InputSource", flex_data) -> None:
    """The database version 77 storage notices, for the storage nodes of this solve (each needs its marker).

    * ``storage_default_fix_nothing``: a storage node without its own ``storage_start_end_method`` whose results
      change — when some node had a ``storage_state_start``, the old default ``fix_start`` fixed it at its own
      start value, or at 0 without one (with no start value anywhere it did nothing, the same as ``fix_nothing``
      now).  The upgrade writes ``fix_start`` on the nodes with their own start value, so they are rarely left.
    * ``storage_fix_method_rewritten``: a storage node whose method the upgrade rewrote, gave a 0 start / end, or
      wrote the old default on.
    """
    from flextool.engine import _behaviour_changes as bc
    default_on = bc.has_marker("storage_default_fix_nothing")
    rewritten = bc.marker_json("storage_fix_method_rewritten").get("rewrites") or []
    if not default_on and not rewritten:
        return
    state = getattr(flex_data, "nodeState", None)
    if state is None or state.height == 0:
        return
    storage = [str(n) for n in state.get_column("n").to_list()]
    if default_on and flex_data.p_state_start is not None and flex_data.p_state_start.frame.height > 0:
        explicit = _entity_scalar_explicit(source, "node", "storage_start_end_method", "n")
        authored = set() if explicit is None else {str(n) for n in explicit.frame.get_column("n").to_list()}
        bc.notice("storage_default_fix_nothing", "storage nodes", sorted(n for n in storage if n not in authored))
    in_solve = set(storage)
    lines = [f"{n} (alternative {alt}: {old} -> {new})" for n, alt, old, new in rewritten if n in in_solve]
    if lines:
        bc.notice("storage_fix_method_rewritten", "storage nodes", lines)


def p_state_end_from_source(source: "InputSource") -> Param | None:
    """``node.storage_state_end`` → ``Param(("n",), [n, value])``: the end state of a fix_end / fix_start_end node,
    a share of its storage capacity.

    Default ``None`` (schema).  Returns explicit rows only.
    """
    return _entity_scalar_explicit(source, "node", "storage_state_end", "n")


# §5.2.7 — process scalars sliced from p_process.csv
# NOTE: ``unit.min_load`` is NOT a direct (p,) scalar.  It is produced as a
# shape-correct (p, d, t) Param by ``_derived.apply_derived_b`` (via
# ``p_min_load_pdt_from_source``) so a period/time Map keeps its per-(d, t)
# semantics instead of being summed in the minFlow_minload constraint.
# See ``specs/fix_map_reading.md`` §5.


# §5.18 — connection scalars
def p_connection_susceptance_from_source(source: "InputSource") -> Param | None:
    """``connection.susceptance`` → ``Param(("p",), [p, value])``.

    Default ``None``.  Used only for DC-power-flow scenarios.  Older
    fixtures may not define ``susceptance`` on the class — treated as
    "no rows".
    """
    return _entity_scalar_explicit(source, "connection", "susceptance", "p")


# §5.19 — commodity scalars
def p_commodity_unitsize_from_source(source: "InputSource") -> Param | None:
    """``commodity.unitsize`` → ``Param(("c",), [c, value])``.

    Default 1.0 (schema).  Used by the commodity-price-ladder feature
    only.  CSV path emits explicit rows only — mirror that via
    ``parameter_explicit``.
    """
    return _entity_scalar_explicit(source, "commodity", "unitsize", "c")


# ---------------------------------------------------------------------------
# Δ.4b — third wave Direct Param helpers.
#
# Patterns covered below:
#
#   1. Indexed (1d_map) entity-class scalars — e.g. ``unit.invest_max_period``
#      (``ed_invest_max_period``) which unrolls to ``[entity, period, value]``.
#   2. Indexed Map(period→time) parameters — e.g. ``node.availability``
#      (``p_node_availability``) which unrolls to ``[entity, period, t,
#      value]``.
#   3. Per-(entity, side) relationship scalars — e.g.
#      ``unit__inputNode.ramp_speed_up`` (``p_ramp_speed_up_source``).
#   4. Multi-class union 1d_map — e.g. invest/divest period caps spanning
#      ``unit`` ∪ ``node`` ∪ ``connection``.
#   5. Multi-class union Map — e.g. ``other_operational_cost`` for var-cost.
#
# All helpers honour the CSV path's "drop zero rows" + "explicit rows
# only" semantics — failing loudly if the underlying parameter shape
# doesn't match the expected schema.

def _filter_param_by_periods(p: Param | None,
                                period_filter: pl.DataFrame | None
                                ) -> Param | None:
    """Restrict ``p`` to rows whose ``d`` (and optionally ``t``) column
    matches the active solve's ``period_filter``.  Returns the Param
    with the join chained onto its existing LazyFrame — no eager
    collect.  Returns ``p`` unchanged when there's nothing to filter
    on (``p`` is None, filter is empty, or the Param has no ``d``/
    ``t`` axes to restrict).

    Shape-aware (Phase E.1): the join is on whichever of ``d`` / ``t``
    are present in ``p.dims``.  Under the lazified broadcast contract
    a MAP_TIME-authored field is ``(entity, t)`` and a SCALAR-authored
    field is ``(entity,)``; both are handled — the former joins on
    ``t``, the latter passes through unchanged.

    Mirrors the CSV path's ``solve_data/pdtNode.csv inner-join with
    steps_in_use`` semantic when both ``d`` and ``t`` are present.

    Reads ``p.lazy`` directly — does NOT trigger ``polar_high.Param``'s
    eager ``.frame`` cache.
    """
    if p is None or period_filter is None or period_filter.height == 0:
        return p
    p_dims = set(p.dims)
    has_d = "d" in p_dims
    has_t = "t" in p_dims
    if not has_d and not has_t:
        # SCALAR-shape Param (e.g. ``(entity,)``) — nothing to filter on.
        return p
    # Phase 4.8g: the Param's underlying LazyFrame may carry Utf8
    # ``d`` / ``t`` columns from the Direct-Param CSV/scalar path while
    # ``period_filter`` (the ``dt`` frame) is fully Enum-cast under
    # activation.  Defensively cast both sides to the live axis enums
    # so the join keys match — without this the join raises
    # ``SchemaError`` (Utf8 ↔ Enum).
    lf = p.lazy
    _enums = get_global_axis_enums()
    if _enums is not None:
        lf = cast_frame_axes(lf, _enums)
        period_filter = cast_frame_axes(period_filter, _enums)
    pf_cols = set(period_filter.columns)
    if has_d and has_t and {"d", "t"}.issubset(pf_cols):
        keep = period_filter.lazy().select("d", "t").unique()
        out_lf = lf.join(keep, on=["d", "t"], how="inner")
    elif has_d and "d" in pf_cols:
        keep = period_filter.lazy().select("d").unique()
        out_lf = lf.join(keep, on="d", how="inner")
    elif has_t and "t" in pf_cols:
        # Phase E.1: MAP_TIME-authored field is ``(entity, t)`` — no
        # ``d`` to filter on.  Restrict by ``t`` instead.
        keep = period_filter.lazy().select("t").unique()
        out_lf = lf.join(keep, on="t", how="inner")
    else:
        # Param has a d/t axis but the period_filter doesn't carry the
        # matching column — nothing to filter on.
        return p
    return Param(p.dims, out_lf)


def _entity_period_scalar(source: "InputSource", entity_class: str,
                            parameter_name: str,
                            entity_dim: str,
                            *,
                            filter_zero: bool = False,
                            filter_null: bool = True,
                            period_filter: pl.DataFrame | None = None
                            ) -> Param | None:
    """Return ``Param((entity_dim, "d"), [<entity_dim>, d, value])`` for a
    ``1d_map(period)`` OR scalar parameter on the given entity class.

    Mirrors the CSV path's slice of ``pdGroup.csv`` / ``pdProcess.csv``
    style files: explicit rows only, optionally filtering out zero /
    null values.  Returns ``None`` when no rows survive.

    Δ.12c-fix gap #3: scalar broadcast.  When the source returns a
    scalar shape (``[name, value]`` without a ``period`` column) we
    broadcast the scalar over the periods supplied in ``period_filter``
    — mirroring flextool's preprocessing which expands a group/process
    scalar to one row per (entity, period) before writing the
    ``pdGroup.csv`` / ``pd_process.csv`` family.  Without
    ``period_filter`` we cannot infer the broadcast axis and fall
    through to ``None``.

    ``period_filter``: optional ``[d]`` frame restricting the output to
    a subset of periods (mirrors flextool preprocessing's per-solve
    period filter — Spine ``invest_max_period`` / ``co2_max_period``
    Maps cover ALL declared periods, but the CSV path's
    ``pd_group.csv`` etc. is pre-filtered to the active solve's
    periods).  Pass ``flex_data.dt`` (already restricted to the active
    solve) to mirror that semantic.  Also used as the broadcast axis
    for scalar-shape values.
    """
    try:
        df = source.parameter_explicit(entity_class, parameter_name)
    except (KeyError, AttributeError):
        try:
            df = source.parameter(entity_class, parameter_name)
        except KeyError:
            return None
    if df is None or df.height == 0:
        return None
    cols = df.columns
    if "value" not in cols:
        return None
    if "period" not in cols:
        # Scalar broadcast cascade — Δ.12c-fix gap #3.  Mirrors
        # flextool preprocessing which writes one (entity, period, value)
        # row per period when the source value is a scalar.
        if period_filter is None or period_filter.height == 0:
            return None
        if "name" not in cols:
            return None
        lf = df.lazy().pipe(rename_to_axis, {"name": entity_dim})
        if filter_null:
            lf = lf.filter(pl.col("value").is_not_null())
        if filter_zero:
            lf = lf.filter(pl.col("value") != 0.0)
        periods = period_filter.lazy().select("d").unique()
        out = (lf.select(entity_dim, "value")
                  .join(periods, how="cross")
                  .select(entity_dim, "d", "value")
                  .collect())
        if out.height == 0:
            return None
        return Param((entity_dim, "d"), out.lazy())
    lf = df.lazy().pipe(rename_to_axis, {"name": entity_dim, "period": "d"})
    if filter_null:
        lf = lf.filter(pl.col("value").is_not_null())
    if filter_zero:
        lf = lf.filter(pl.col("value") != 0.0)
    if period_filter is not None and period_filter.height > 0:
        lf = lf.join(period_filter.lazy().select("d").unique(), on="d",
                       how="inner")
    out = lf.select(entity_dim, "d", "value").collect()
    if out.height == 0:
        return None
    return Param((entity_dim, "d"), out.lazy())


def _entity_period_time_param(source: "InputSource", entity_class: str,
                                parameter_name: str,
                                entity_dim: str,
                                *,
                                filter_zero: bool = False,
                                period_filter: pl.DataFrame | None = None
                                ) -> Param | None:
    """Return ``Param((entity_dim, "d", "t"), [...])`` for a
    ``Map(period→time)`` parameter.  CSV slice of ``pdtNode.csv`` /
    ``pdtCommodity.csv`` / ``pdtGroup.csv`` files.

    ``period_filter`` restricts the output to a subset of periods (per
    the active solve), mirroring CSV preprocessing.

    Δ.12c-fix gap #1 — broadcast cascade.  In addition to the explicit
    ``Map(period→time)`` shape (columns ``[name, period, t, value]``)
    the helper now broadcasts:
    * 1d_map(time)  — ``[name, t, value]``  → cross-join with periods.
    * 1d_map(period) — ``[name, period, value]`` → cross-join with t.
    * scalar — ``[name, value]`` → cross-join with (period, t).
    Mirrors flextool preprocessing which writes one row per (entity,
    period, t) into the ``pdtX.csv`` family.

    Broadcast requires ``period_filter`` to carry the (d, t) pairs to
    expand against (typically the active solve's ``dt`` frame).  Without
    it the broadcast paths fall through to ``None``.
    """
    try:
        df = source.parameter_explicit(entity_class, parameter_name)
    except (KeyError, AttributeError):
        try:
            df = source.parameter(entity_class, parameter_name)
        except KeyError:
            return None
    if df is None or df.height == 0:
        return None
    cols = df.columns
    if "value" not in cols or "name" not in cols:
        return None
    has_period = "period" in cols
    has_t = "t" in cols
    if has_period and has_t:
        lf = (df.lazy()
                .pipe(rename_to_axis, {"name": entity_dim, "period": "d"})
                .filter(pl.col("value").is_not_null()))
        if filter_zero:
            lf = lf.filter(pl.col("value") != 0.0)
        if period_filter is not None and period_filter.height > 0:
            lf = lf.join(period_filter.lazy().select("d").unique(), on="d",
                           how="inner")
        out = lf.select(entity_dim, "d", "t", "value").collect()
        if out.height == 0:
            return None
        return Param((entity_dim, "d", "t"), out.lazy())
    # Broadcast paths require period_filter to know the (d, t) axis.
    if period_filter is None or period_filter.height == 0:
        return None
    pf_cols = set(period_filter.columns)
    if not {"d", "t"}.issubset(pf_cols):
        return None
    lf = df.lazy().pipe(rename_to_axis, {"name": entity_dim}).filter(
        pl.col("value").is_not_null())
    if filter_zero:
        lf = lf.filter(pl.col("value") != 0.0)
    dt_lf = period_filter.lazy().select("d", "t").unique()
    if has_period:
        # 1d_map(period) → broadcast across t per (entity, d).
        lf2 = lf.pipe(rename_to_axis, {"period": "d"}).select(entity_dim, "d", "value")
        out = (lf2.join(dt_lf, on="d", how="inner")
                  .select(entity_dim, "d", "t", "value")
                  .collect())
    elif has_t:
        # 1d_map(time) → broadcast across d per (entity, t).
        lf2 = lf.select(entity_dim, "t", "value")
        out = (lf2.join(dt_lf, on="t", how="inner")
                  .select(entity_dim, "d", "t", "value")
                  .collect())
    else:
        # Scalar → broadcast across (d, t) per entity.
        lf2 = lf.select(entity_dim, "value")
        out = (lf2.join(dt_lf, how="cross")
                  .select(entity_dim, "d", "t", "value")
                  .collect())
    if out.height == 0:
        return None
    return Param((entity_dim, "d", "t"), out.lazy())


# §5.14 — group scalars sliced from input/p_group.csv
def _g_scalar(source: "InputSource", parameter_name: str,
                *, filter_zero: bool = True,
                entity_class: str = "group") -> Param | None:
    """Group scalar Param ``Param(("g",), [g, value])``, dropping zero
    rows by default (mirrors ``_slice_pgroup``'s ``!= 0`` filter).

    ``entity_class`` defaults to ``"group"`` (all non-flow callers); the
    v58 cumulative-flow callers pass ``"flowGroup"`` since those four
    flow params now live on the dedicated ``flowGroup`` class.
    """
    try:
        df = source.parameter_explicit(entity_class, parameter_name)
    except (KeyError, AttributeError):
        try:
            df = source.parameter(entity_class, parameter_name)
        except KeyError:
            return None
    if df is None or df.height == 0:
        return None
    if "value" not in df.columns:
        return None
    lf = df.lazy().pipe(rename_to_axis, {"name": "g"}).filter(pl.col("value").is_not_null())
    if filter_zero:
        lf = lf.filter(pl.col("value") != 0.0)
    # Indexed (period) shapes don't belong here — guard.
    if "period" in df.columns or "t" in df.columns:
        return None
    out = lf.select("g", "value").collect()
    if out.height == 0:
        return None
    return Param(("g",), out.lazy())


# §1.16 — pdGroup_* (1d_map period) — dropping zeros and nulls
#
# Δ.17c-Tier1 — silent-default index migration.  These helpers were
# previously routed through ``_entity_period_scalar``, which keyed the
# scalar/explicit branch off ``"period" not in cols``.  When the source
# DB authored the Map without ``index_name`` (spinedb_api's silent
# default ``"x"``), the helper fell into the scalar branch and
# cross-joined the Map's rows against the period filter — producing
# ``N_map_rows × N_periods`` duplicate (g, d) keys per entity and
# destroying the period → value association.
#
# Migration template: ``resolve_param_shape`` reads the DB-side shape
# (with silent-default disambiguation via ``_infer_silent_default_labels``
# — the {SCALAR, MAP_PERIOD} allow-list resolves the silent default
# structurally at depth 1) and ``broadcast_to_period`` produces the
# (entity, d) Param.  ``filter_zero=True`` is supported natively by
# ``broadcast_to_period``, mirroring the legacy
# ``filter_null=True, filter_zero=True`` semantic.
def _only_groups(param: "Param | None", groups: set[str], *, keep: bool = True) -> "Param | None":
    """``param`` (g, d) restricted to the groups ``groups`` (``keep=False``: without them); None when no row is
    left."""
    if param is None:
        return None
    frame = param.frame
    if isinstance(frame, pl.LazyFrame):
        frame = frame.collect()
    member = pl.col("g").cast(pl.Utf8).is_in(sorted(groups))
    frame = frame.filter(member if keep else ~member)
    return Param(param.dims, frame) if frame.height > 0 else None


def pdGroup_capacity_margin_from_source(source: "InputSource",
                                         period_filter: pl.DataFrame | None = None,
                                         *,
                                         need_periods: "PeriodNeed | None" = None,
                                         ) -> Param | None:
    """``group.capacity_margin`` → ``Param(("g","d"))`` [MW]: the margin of the groups with
    ``capacity_margin_method = manual``.

    Δ.17c-Tier1 — uses :func:`._param_shapes.resolve_param_shape`.
    Allowed shapes: scalar / 1d_map[period].  CSV path drops zero
    rows; we mirror via ``filter_zero=True``.  A period map of a manual group must cover ``need_periods`` (the
    solve's investment periods; :func:`require_period_coverage`).  The value of a group with another method is
    not the margin of a ``share_of_peak`` group (that is ``capacity_margin_share``; the validators warn that it is
    ignored), so those rows are left out.
    """
    resolved = resolve_param_shape(
        source, "group", "capacity_margin", period_filter=period_filter)
    if resolved is None or resolved.frame is None:
        return None
    active = _group_names(groupCapacityMarginManual(source)) if _gate_needed(need_periods, resolved) else None
    out = require_period_coverage(
        resolved, "g", period_filter, entity_class="group", parameter_name="capacity_margin",
        need_periods=need_periods, active=active, filter_zero=True)
    share = _group_names(groupCapacityMarginShareOfPeak(source))
    return _only_groups(out, share, keep=False) if share else out


def pdGroup_capacity_margin_share_from_source(source: "InputSource",
                                               period_filter: pl.DataFrame | None = None,
                                               *,
                                               need_periods: "PeriodNeed | None" = None,
                                               ) -> Param | None:
    """``group.capacity_margin_share`` → ``Param(("g","d"))`` [share] of the groups with ``capacity_margin_method =
    share_of_peak``; the margin is the share times the group's peak demand
    (:func:`flextool.engine.model._group_slack.resolve_capacity_margin`).

    Scalar or 1d_map[period] (:func:`resolve_param_shape` / :func:`require_period_coverage`, invariant #2); zero
    rows dropped (margin 0).  A period map must cover ``need_periods`` (the solve's investment periods).
    """
    resolved = resolve_param_shape(
        source, "group", "capacity_margin_share", period_filter=period_filter)
    if resolved is None or resolved.frame is None:
        return None
    share = _group_names(groupCapacityMarginShareOfPeak(source))
    out = require_period_coverage(
        resolved, "g", period_filter, entity_class="group", parameter_name="capacity_margin_share",
        need_periods=need_periods, active=share if _gate_needed(need_periods, resolved) else None,
        filter_zero=True)
    return _only_groups(out, share)


def _group_penalty_from_source(source: "InputSource", parameter_name: str,
                               period_filter: pl.DataFrame | None) -> Param | None:
    """``group.<parameter_name>`` → ``Param(("g", "d"))`` on every group and active period.

        penalty[g, d] = authored value of g at d         scalar or period map
                      = schema default (5000)            g has no value, or its map leaves out d
        ∀ g ∈ group, d ∈ period_filter;  rows with penalty 0 are dropped (no slack term)

    ``_try_parameter_frame`` returns the explicit rows only once any group authors a value, so without the
    fill a group with a limit but no penalty of its own had a free slack (the objective term is an
    inner-join product).  The zero filter runs after the fill: an explicit 0 stays "no term" (free
    slack, the authored meaning) instead of becoming the default.  Without a period filter the authored
    rows are returned as before (no grid to fill).
    """
    resolved = resolve_param_shape(source, "group", parameter_name, period_filter=period_filter)
    if period_filter is None or period_filter.height == 0:
        return broadcast_to_period(resolved, "g", period_filter, filter_zero=True)
    authored = broadcast_to_period(resolved, "g", period_filter)
    groups = source.entities("group").lazy().pipe(rename_to_axis, {"name": "g"}).select("g").collect()
    filled = fill_default_over_grid(authored, groups, period_filter.select("d"),
                                    source.parameter_default("group", parameter_name),
                                    name=f"group.{parameter_name}")
    if filled is None:
        return None
    out = filled.frame.filter(pl.col("value") != 0.0)
    if out.height == 0:
        return None
    return Param(("g", "d"), out)


def pdGroup_penalty_capacity_margin_from_source(source: "InputSource",
                                                 period_filter: pl.DataFrame | None = None,
                                                 ) -> Param | None:
    """``group.penalty_capacity_margin`` → ``Param(("g","d"))``: authored value, else the schema default; zero rows
    dropped.  See :func:`_group_penalty_from_source`.
    """
    return _group_penalty_from_source(source, "penalty_capacity_margin", period_filter)


#: The group limits whose time map must cover every time step of the solve: (parameter, the groups it applies to,
#: the ``reduce_system_detail_method`` part that leaves the limit out of a solve).
GROUP_TIME_LIMITS: tuple[tuple[str, "Callable[[InputSource], pl.DataFrame]", str], ...] = (
    ("inertia_limit", groupInertia, "inertia"),
    ("non_synchronous_limit", groupNonSync, "non_synchronous"),
)

#: What a time map gives for a time step without a limit (the hint of the coverage error).
_TIME_LIMIT_HINT: dict[str, str] = {
    "inertia_limit": "give 0 for a time step without a requirement",
    "non_synchronous_limit": "give 1 for a time step without a limit",
}

#: The ``reduce_system_detail_method`` parts of the solve being loaded (:func:`reduced_system_parts`): a solve that
#: leaves a group limit out does not need its time map to cover the solve's steps.
_REDUCED_SYSTEM_PARTS: "contextvars.ContextVar[frozenset[str]]" = contextvars.ContextVar(
    "flextool_reduced_system_parts", default=frozenset())


@contextlib.contextmanager
def reduced_system_parts(parts: "frozenset[str]") -> "Iterator[None]":
    """Arm the ``reduce_system_detail_method`` parts of the solve being loaded for the per-solve time-coverage check
    (:func:`_require_solve_time_coverage`) while the block runs."""
    token = _REDUCED_SYSTEM_PARTS.set(frozenset(parts))
    try:
        yield
    finally:
        _REDUCED_SYSTEM_PARTS.reset(token)


def _require_solve_time_coverage(source: "InputSource", resolved: "ResolvedShape | None",
                                 period_filter: "pl.DataFrame | None", parameter_name: str,
                                 flagged: "Callable[[InputSource], pl.DataFrame]", part: str,
                                 solve: "str | None") -> None:
    """A group limit's time map must give a value for every time step of this solve (:func:`require_time_coverage`).

        steps = { (d, t) ∈ period_filter }                             without new_stepduration
              = { (d, b) : (d, s) ∈ period_filter, b a base step of s }  with it (the original time steps)

    Only the groups ``flagged`` switches on are checked, and nothing when the solve leaves the limit out
    (``part`` in its armed :func:`reduced_system_parts`).  The run checks every solve's time steps once before its
    first solve (:func:`check_group_limit_time_coverage`); this per-solve check covers a load outside a run.
    """
    if (part in _REDUCED_SYSTEM_PARTS.get()
            or resolved is None or resolved.frame is None or resolved.time_index_column is None
            or period_filter is None or period_filter.height == 0
            or not resolved.frame.get_column(resolved.time_index_column).is_not_null().any()):
        return
    steps = period_filter.select(pl.col("d").cast(pl.Utf8), pl.col("t").cast(pl.Utf8))
    base = active_base_steps()
    if base is not None:
        base_of = base_steps_of(base, steps.get_column("t")).select(
            pl.col("new_step").cast(pl.Utf8).alias("t"), pl.col("time").cast(pl.Utf8).alias("__b"))
        steps = steps.join(base_of, on="t", how="inner", maintain_order="left").select("d", pl.col("__b").alias("t"))
    require_time_coverage(resolved, steps, entity_class="group", parameter_name=parameter_name,
                          active=_group_names(flagged(source)), solve=solve,
                          hint=_TIME_LIMIT_HINT.get(parameter_name, ""))


def _authored_map_names(source: "InputSource", parameter_name: str) -> set[str]:
    """The groups that author ``parameter_name`` as a map (a row with an index key and a value)."""
    try:
        df = source.parameter_explicit("group", parameter_name)
    except (KeyError, AttributeError):
        return set()
    if df is None or df.height == 0 or "value" not in df.columns or "name" not in df.columns:
        return set()
    index = [c for c in df.columns if c not in ("name", "value")]
    if not index:
        return set()
    keyed = pl.any_horizontal([pl.col(c).is_not_null() for c in index])
    return set(df.filter(keyed & pl.col("value").is_not_null()).get_column("name").cast(pl.Utf8).to_list())


def check_group_limit_time_coverage(source: "InputSource",
                                    steps_of_run: "Callable[[], dict[str, pl.DataFrame]]",
                                    reduced_parts_of: "Callable[[str], frozenset[str]] | None" = None) -> None:
    """Before the first solve of a run: every group limit of :data:`GROUP_TIME_LIMITS` given as a time map (or a
    period + time map) must give a value for every time step that the run's solves use.

        run      = ∪_s steps[s]
        resolved = resolve_param_shape(parameter, period_filter=run)                    once per parameter
        ∀ solve s with part ∉ reduced_parts_of(s):  require_time_coverage(resolved, steps[s])

    ``steps_of_run()``: solve name -> ``(d, t)`` of the original time steps of all its sub-solves (every roll,
    look-ahead included; a ``new_stepduration`` step lists the time steps it combines); called only when a group
    authors one of the limits as a map.  ``reduced_parts_of(s)``: the ``reduce_system_detail_method`` parts of
    solve ``s``; a solve that leaves a limit out is not checked for it.  The first gap raises
    :class:`FlexToolConfigError` (:func:`require_time_coverage`), so a rolling run stops before its first roll
    instead of at the roll that reaches the gap.

    The shape is resolved against the steps of the whole run, not per solve, so a map with Spine's silent ``x``
    index is read the same way in every solve: keys that are periods make a period map (not checked here), keys
    that are time steps a time map, and a solve whose steps hold none of its keys is a gap, not "no value".  A
    flagged group whose map has no key that is a period or a time step of the run raises as well.
    """
    limits = [(p, flagged, part) for p, flagged, part in GROUP_TIME_LIMITS if _authored_map_names(source, p)]
    if not limits:
        return
    steps_by_solve = {s: f for s, f in steps_of_run().items() if f is not None and f.height > 0}
    if not steps_by_solve:
        return
    run = pl.concat(list(steps_by_solve.values())).unique(maintain_order=True)
    for parameter_name, flagged, part in limits:
        solves = [s for s in steps_by_solve if reduced_parts_of is None or part not in reduced_parts_of(s)]
        active = _group_names(flagged(source)) if solves else set()
        if not active:
            continue
        resolved = resolve_param_shape(source, "group", parameter_name, period_filter=run)
        kept = set()
        if resolved is not None and resolved.frame is not None and resolved.frame.height > 0:
            label = pl.concat_str([pl.col(c).cast(pl.Utf8) for c in resolved.entity_dim_columns], separator=", ")
            kept = set(resolved.frame.select(label.alias("__k")).get_column("__k").to_list())
        lost = sorted((_authored_map_names(source, parameter_name) & active) - kept
                      - set(getattr(resolved, "stochastic_entities", None) or ()))
        if lost:
            raise FlexToolConfigError(
                "; ".join(f"group '{g}', parameter '{parameter_name}'" for g in lost)
                + ": the map has no key that is a period or a time step of the run's solves. Give a value for "
                  "every period or time step the solves use (and name the map's index 'period' or 'time').")
        if (resolved is None or resolved.frame is None or resolved.time_index_column is None
                or not resolved.frame.get_column(resolved.time_index_column).is_not_null().any()):
            continue
        for solve in solves:
            require_time_coverage(resolved, steps_by_solve[solve], entity_class="group",
                                  parameter_name=parameter_name, active=active, solve=solve,
                                  hint=_TIME_LIMIT_HINT.get(parameter_name, ""))


def pdtGroup_inertia_limit_from_source(source: "InputSource",
                                        period_filter: pl.DataFrame | None = None,
                                        *,
                                        need_periods: "PeriodNeed | None" = None,
                                        ) -> Param | None:
    """``group.inertia_limit`` → ``Param`` with the authored dims ⊆ ``(g, d, t)``.

    Constant, period, time or period + time map (:func:`resolve_param_shape` →
    :func:`broadcast_to_period_time`); zero rows dropped (an authored 0 is no requirement).  A period map, or the
    period level of a period + time map, of a group with ``has_inertia = yes`` must cover ``need_periods``
    (:func:`require_period_coverage_time`), and a time map, or the time level of a period + time map, every
    time step of ``period_filter`` (:func:`_require_solve_time_coverage`).
    """
    resolved = resolve_param_shape(
        source, "group", "inertia_limit", period_filter=period_filter)
    _require_solve_time_coverage(source, resolved, period_filter, "inertia_limit", groupInertia, "inertia",
                                 need_periods.solve if need_periods is not None else None)
    active = _group_names(groupInertia(source)) if _gate_needed(need_periods, resolved) else None
    return require_period_coverage_time(
        resolved, "g", period_filter, entity_class="group", parameter_name="inertia_limit",
        need_periods=need_periods, active=active, filter_zero=True)


def pdGroup_penalty_inertia_from_source(source: "InputSource",
                                         period_filter: pl.DataFrame | None = None,
                                         ) -> Param | None:
    """``group.penalty_inertia`` → ``Param(("g","d"))``: authored value, else the schema default; zero rows
    dropped.  See :func:`_group_penalty_from_source`.
    """
    return _group_penalty_from_source(source, "penalty_inertia", period_filter)


def pdtGroup_non_synchronous_limit_from_source(source: "InputSource",
                                                period_filter: pl.DataFrame | None = None,
                                                *,
                                                need_periods: "PeriodNeed | None" = None,
                                                ) -> Param | None:
    """``group.non_synchronous_limit`` → ``Param`` with the authored dims ⊆ ``(g, d, t)``.

    Constant, period, time or period + time map (:func:`resolve_param_shape` →
    :func:`broadcast_to_period_time`).  Zero rows are kept: an authored 0 is a limit of 0.  A period map, or the
    period level of a period + time map, of a group with ``has_non_synchronous = yes`` must cover
    ``need_periods`` (:func:`require_period_coverage_time`), and a time map, or the time level of a period + time
    map, every time step of ``period_filter`` (:func:`_require_solve_time_coverage`).
    """
    resolved = resolve_param_shape(
        source, "group", "non_synchronous_limit",
        period_filter=period_filter)
    _require_solve_time_coverage(source, resolved, period_filter, "non_synchronous_limit", groupNonSync,
                                 "non_synchronous",
                                 need_periods.solve if need_periods is not None else None)
    active = _group_names(groupNonSync(source)) if _gate_needed(need_periods, resolved) else None
    return require_period_coverage_time(
        resolved, "g", period_filter, entity_class="group", parameter_name="non_synchronous_limit",
        need_periods=need_periods, active=active, filter_zero=False)


def check_energy_margin_adder_shape(source: "InputSource", period_filter: pl.DataFrame | None) -> None:
    """Refuse a ``node.energy_margin_adder`` shape outside :data:`PARAM_ALLOWED_SHAPES` (a constant, a period map or
    a period + time map), raising :class:`FlexToolConfigError` that names the nodes and the allowed shapes.

    :func:`resolve_param_shape` refuses a map whose index levels are named outside the allow-list (a ``time`` map).
    A one-level map with Spine's silent label reads as a period map (the only one-level shape); its keys show
    what it is: a node whose keys hold a time step of the model (``period_filter`` or the axis vocabulary) and
    no period is a time map.  The values themselves are applied by the per-solve emit
    (``emit/_energy_margin_adder.py``), where a time map has no ingestion spec and was silently ignored.
    """
    resolved = resolve_param_shape(source, "node", "energy_margin_adder", period_filter=period_filter)
    if resolved is None or resolved.frame is None or resolved.period_index_column is None:
        return
    if resolved.time_index_column is not None and resolved.frame.get_column(resolved.time_index_column).is_not_null().any():
        return
    periods: set[str] = set()
    times: set[str] = set()
    if period_filter is not None and {"d", "t"} <= set(period_filter.columns):
        periods |= set(period_filter.get_column("d").cast(pl.Utf8).to_list())
        times |= set(period_filter.get_column("t").cast(pl.Utf8).to_list())
    enums = get_global_axis_enums() or {}
    if "d" in enums:
        periods |= set(enums["d"].categories.to_list())
    if "t" in enums:
        times |= set(enums["t"].categories.to_list())
    ent = resolved.entity_dim_columns[0]
    keys = (resolved.frame.select(pl.col(ent).cast(pl.Utf8).alias("name"),
                                  pl.col(resolved.period_index_column).cast(pl.Utf8).alias("k"))
            .filter(pl.col("k").is_not_null())
            .group_by("name", maintain_order=True)
            .agg(timed=pl.col("k").is_in(list(times)).any(), periodic=pl.col("k").is_in(list(periods)).any()))
    bad = sorted(keys.filter(pl.col("timed") & ~pl.col("periodic")).get_column("name").to_list())
    if bad:
        raise FlexToolConfigError(
            f"node {', '.join(repr(b) for b in bad[:5])}, parameter 'energy_margin_adder': the value is a map over "
            f"time steps (its keys are time steps, not periods). This parameter takes "
            f"{allowed_shapes_in_words(PARAM_ALLOWED_SHAPES[('node', 'energy_margin_adder')])}; name the map's "
            f"index levels 'period' / 'time'.")


def pdGroup_penalty_non_synchronous_from_source(source: "InputSource",
                                                 period_filter: pl.DataFrame | None = None,
                                                 ) -> Param | None:
    """``group.penalty_non_synchronous`` → ``Param(("g","d"))``: authored value, else the schema default; zero rows
    dropped.  See :func:`_group_penalty_from_source`.
    """
    return _group_penalty_from_source(source, "penalty_non_synchronous", period_filter)


# §1.16 — group invest/divest 1d_map(period) — drop zeros to match CSV
def _group_invest_period_limit(source: "InputSource", resolved, parameter_name: str,
                               methods: "frozenset[str]", period_filter: pl.DataFrame | None,
                               need_periods: "PeriodNeed | None", *, filter_zero: bool) -> Param | None:
    """A group investment limit per period: a period map of a group whose ``invest_method`` is one of
    ``methods`` must cover ``need_periods`` (the solve's investment periods;
    :func:`require_period_coverage`)."""
    active = (_names_with_method(source, "group", "invest_method", methods)
              if _gate_needed(need_periods, resolved) else None)
    return require_period_coverage(
        resolved, "g", period_filter, entity_class="group", parameter_name=parameter_name,
        need_periods=need_periods, active=active, filter_zero=filter_zero)


def p_group_invest_max_period_from_source(source: "InputSource",
                                            period_filter: pl.DataFrame | None = None,
                                            *,
                                            need_periods: "PeriodNeed | None" = None,
                                            ) -> Param | None:
    """``group.invest_max_period`` → ``Param(("g","d"))``.
    Δ.17c-Tier1 — uses ``resolve_param_shape`` + ``broadcast_to_period``.

    Zero rows are kept (``filter_zero=False``) — an explicit 0 is a given value
    (it binds, forbidding investment that period) and must not be read back as "no limit" by
    ``check_max_cap`` (FA-12) once some other group sets ``invest_max_period``.
    """
    resolved = resolve_param_shape(
        source, "group", "invest_max_period", period_filter=period_filter)
    return _group_invest_period_limit(source, resolved, "invest_max_period", _INVEST_PERIOD_METHODS, period_filter,
                                      need_periods, filter_zero=False)


def p_group_invest_min_period_from_source(source: "InputSource",
                                            period_filter: pl.DataFrame | None = None,
                                            *,
                                            need_periods: "PeriodNeed | None" = None,
                                            ) -> Param | None:
    """``group.invest_min_period`` → ``Param(("g","d"))``.
    Δ.17c-Tier1 — uses ``resolve_param_shape`` + ``broadcast_to_period``.
    """
    resolved = resolve_param_shape(
        source, "group", "invest_min_period", period_filter=period_filter)
    return _group_invest_period_limit(source, resolved, "invest_min_period", _INVEST_PERIOD_METHODS, period_filter,
                                      need_periods, filter_zero=True)


def p_group_retire_max_period_from_source(source: "InputSource",
                                            period_filter: pl.DataFrame | None = None,
                                            *,
                                            need_periods: "PeriodNeed | None" = None,
                                            ) -> Param | None:
    """``group.retire_max_period`` → ``Param(("g","d"))``.
    Δ.17c-Tier1 — uses ``resolve_param_shape`` + ``broadcast_to_period``.

    Zero rows are kept (``filter_zero=False``) — see
    :func:`p_group_invest_max_period_from_source`.
    """
    resolved = resolve_param_shape(
        source, "group", "retire_max_period", period_filter=period_filter)
    return _group_invest_period_limit(source, resolved, "retire_max_period", _DIVEST_PERIOD_METHODS, period_filter,
                                      need_periods, filter_zero=False)


def p_group_retire_min_period_from_source(source: "InputSource",
                                            period_filter: pl.DataFrame | None = None,
                                            *,
                                            need_periods: "PeriodNeed | None" = None,
                                            ) -> Param | None:
    """``group.retire_min_period`` → ``Param(("g","d"))``.
    Δ.17c-Tier1 — uses ``resolve_param_shape`` + ``broadcast_to_period``.
    """
    resolved = resolve_param_shape(
        source, "group", "retire_min_period", period_filter=period_filter)
    return _group_invest_period_limit(source, resolved, "retire_min_period", _DIVEST_PERIOD_METHODS, period_filter,
                                      need_periods, filter_zero=True)


def pd_max_cumulative_flow_from_source(source: "InputSource",
                                         period_filter: pl.DataFrame | None = None,
                                         *,
                                         need_periods: "PeriodNeed | None" = None,
                                         ) -> Param | None:
    """``flowGroup.max_cumulative_flow`` → ``Param(("g","d"))``.
    Δ.17c-Tier1 — uses ``resolve_param_shape`` + ``broadcast_to_period``.

    Zero rows are kept: the net flow into a flowGroup is signed, so an explicit 0
    is a limit (no net import / export), not "no limit".  A period map must cover
    ``need_periods`` (the solve's periods; :func:`require_period_coverage`).
    """
    resolved = resolve_param_shape(
        source, "flowGroup", "max_cumulative_flow",
        period_filter=period_filter)
    return require_period_coverage(
        resolved, "g", period_filter, entity_class="flowGroup", parameter_name="max_cumulative_flow",
        need_periods=need_periods)


def pd_min_cumulative_flow_from_source(source: "InputSource",
                                         period_filter: pl.DataFrame | None = None,
                                         *,
                                         need_periods: "PeriodNeed | None" = None,
                                         ) -> Param | None:
    """``flowGroup.min_cumulative_flow`` → ``Param(("g","d"))``.
    Δ.17c-Tier1 — uses ``resolve_param_shape`` + ``broadcast_to_period``.

    Zero rows are kept: the net flow into a flowGroup is signed, so an explicit 0
    is a limit (no net import / export), not "no limit".  A period map must cover
    ``need_periods`` (the solve's periods; :func:`require_period_coverage`).
    """
    resolved = resolve_param_shape(
        source, "flowGroup", "min_cumulative_flow",
        period_filter=period_filter)
    return require_period_coverage(
        resolved, "g", period_filter, entity_class="flowGroup", parameter_name="min_cumulative_flow",
        need_periods=need_periods)


# §1.16 — group scalar Direct Params (no period dimension)
def p_group_invest_max_total_from_source(source: "InputSource") -> Param | None:
    """``filter_zero=False`` — see
    :func:`p_group_invest_max_period_from_source`; same reasoning for the total side."""
    return _g_scalar(source, "invest_max_total", filter_zero=False)


def p_group_invest_min_total_from_source(source: "InputSource") -> Param | None:
    return _g_scalar(source, "invest_min_total")


def p_group_retire_max_total_from_source(source: "InputSource") -> Param | None:
    """``filter_zero=False`` — see
    :func:`p_group_invest_max_period_from_source`; same reasoning for the total side."""
    return _g_scalar(source, "retire_max_total", filter_zero=False)


def p_group_retire_min_total_from_source(source: "InputSource") -> Param | None:
    return _g_scalar(source, "retire_min_total")


def pd_group_cumulative_max_capacity_from_source(source: "InputSource",
                                                   period_filter: pl.DataFrame | None = None,
                                                   *,
                                                   need_periods: "PeriodNeed | None" = None,
                                                   ) -> Param | None:
    """``group.cumulative_max_capacity`` → ``Param(("g","d"))`` (#5: the schema parameter, not the
    off-schema ``invest_max_cumulative`` the engine used to read).  Zero rows are kept
    (``filter_zero=False``, same reasoning as the entity reader): an
    explicit 0 (a phase-out year) must bind, not read back as "no limit".
    """
    resolved = resolve_param_shape(
        source, "group", "cumulative_max_capacity", period_filter=period_filter)
    return _group_invest_period_limit(source, resolved, "cumulative_max_capacity", _CUMULATIVE_METHODS,
                                      period_filter, need_periods, filter_zero=False)


def pd_group_cumulative_min_capacity_from_source(source: "InputSource",
                                                   period_filter: pl.DataFrame | None = None,
                                                   *,
                                                   need_periods: "PeriodNeed | None" = None,
                                                   ) -> Param | None:
    """``group.cumulative_min_capacity`` → ``Param(("g","d"))`` (#5).  Zero rows dropped (a min of 0
    is no floor, as capacity is never negative; same as the entity reader).  A period map must cover
    ``need_periods`` (checked before the zero filter, so an explicit 0 is a value).
    """
    resolved = resolve_param_shape(
        source, "group", "cumulative_min_capacity", period_filter=period_filter)
    return _group_invest_period_limit(source, resolved, "cumulative_min_capacity", _CUMULATIVE_METHODS,
                                      period_filter, need_periods, filter_zero=True)


def p_group_max_cumulative_flow_from_source(source: "InputSource") -> Param | None:
    """Zero rows are kept (``filter_zero=False``): an explicit 0 is a limit, see
    :func:`pd_max_cumulative_flow_from_source`."""
    return _g_scalar(source, "max_cumulative_flow", entity_class="flowGroup", filter_zero=False)


def p_group_min_cumulative_flow_from_source(source: "InputSource") -> Param | None:
    """Zero rows are kept (``filter_zero=False``), see :func:`pd_min_cumulative_flow_from_source`."""
    return _g_scalar(source, "min_cumulative_flow", entity_class="flowGroup", filter_zero=False)


# §1.16 — pdtGroup (Map period→time) instant flow caps
def pdt_max_instant_flow_from_source(source: "InputSource",
                                      period_filter: pl.DataFrame | None = None,
                                      *,
                                      need_periods: "PeriodNeed | None" = None,
                                      ) -> Param | None:
    """``flowGroup.max_instant_flow`` → ``Param(("g", "d", "t"))``.

    Δ.17c — uses :func:`._param_shapes.resolve_param_shape`.  Allowed
    shapes: scalar / 1d_map[period] / 1d_map[time] / 2d_map[period,time].
    Zero rows are kept: an explicit 0 (also one time step of a time map) is a
    limit of 0.  A period map, or the period level of a period + time map, must
    cover ``need_periods`` (:func:`require_period_coverage_time`).
    """
    resolved = resolve_param_shape(
        source, "flowGroup", "max_instant_flow", period_filter=period_filter)
    return require_period_coverage_time(
        resolved, "g", period_filter, entity_class="flowGroup", parameter_name="max_instant_flow",
        need_periods=need_periods)


def pdt_min_instant_flow_from_source(source: "InputSource",
                                      period_filter: pl.DataFrame | None = None,
                                      *,
                                      need_periods: "PeriodNeed | None" = None,
                                      ) -> Param | None:
    """``flowGroup.min_instant_flow`` → ``Param(("g", "d", "t"))``.

    Δ.17c — uses :func:`._param_shapes.resolve_param_shape`.  Allowed
    shapes: scalar / 1d_map[period] / 1d_map[time] / 2d_map[period,time].
    Zero rows are kept: an explicit 0 (also one time step of a time map) is a
    limit of 0.  A period map, or the period level of a period + time map, must
    cover ``need_periods`` (:func:`require_period_coverage_time`).
    """
    resolved = resolve_param_shape(
        source, "flowGroup", "min_instant_flow", period_filter=period_filter)
    return require_period_coverage_time(
        resolved, "g", period_filter, entity_class="flowGroup", parameter_name="min_instant_flow",
        need_periods=need_periods)


# ---------------------------------------------------------------------------
# Multi-class union helpers — params declared on each of unit/node/connection
# (or unit/connection) and merged into one frame keyed on the generic
# ``e`` (entity) or ``p`` (process) dim.

def _e_period_param_union(source: "InputSource",
                            parameter_name: str,
                            *,
                            classes: tuple[str, ...] = ("unit", "node",
                                                          "connection"),
                            filter_zero: bool = False,
                            period_filter: pl.DataFrame | None = None,
                            need_periods: "PeriodNeed | None" = None,
                            methods: "frozenset[str] | None" = None,
                            ) -> Param | None:
    """Union ``parameter_name`` across the supplied entity classes into a
    single ``Param(("e", "d"), [...])`` frame.

    Δ.17c-Tier2 — each per-class call routes through
    :func:`._param_shapes.resolve_param_shape` +
    :func:`._param_shapes.broadcast_to_period`.  This:

    * carries silent-default ``index_name`` disambiguation
      (spinedb_api's ``"x"`` resolves structurally via the
      ``{SCALAR, MAP_PERIOD}`` allow-list at depth 1; value-domain
      probing kicks in if the labels are still ambiguous), and
    * promotes scalar-authored parameters to ``(e, d)`` via the
      period filter (the pre-migration path silently dropped these
      because ``period_col`` was ``None``).

    Each per-class ``broadcast_to_period`` call aliases the source's
    ``name`` column to the union ``e`` axis (``cast_frame_axes``
    resolves ``"e"`` to the entity-union Enum), so the per-class
    Param frames concat cleanly along the union vocabulary.

    ``period_filter`` is REQUIRED for scalar-authored sources (the
    SCALAR branch in :func:`broadcast_to_period` cross-joins with the
    period axis to emit ``(e, d)``).  Callers that don't have a period
    filter on hand should default to passing the active solve's ``dt``
    frame, mirroring Phase 1's pdGroup pattern.

    With ``need_periods``, a period map of an entity whose ``invest_method``
    is one of ``methods`` must cover those periods (the solve's investment
    periods).  The check runs per class, so the error names the right class
    (:func:`require_period_coverage`).
    """
    parts: list[pl.LazyFrame] = []
    for cls in classes:
        resolved = resolve_param_shape(
            source, cls, parameter_name, period_filter=period_filter)
        active = (_names_with_method(source, cls, "invest_method", methods)
                  if methods is not None and _gate_needed(need_periods, resolved) else None)
        part = require_period_coverage(
            resolved, "e", period_filter, entity_class=cls, parameter_name=parameter_name,
            need_periods=need_periods, active=active, filter_zero=filter_zero)
        if part is None:
            continue
        # ``broadcast_to_period`` returns a Param whose dims are either
        # ``(e, d)`` (MAP_PERIOD or SCALAR-broadcast) or ``(e,)`` (SCALAR
        # with no period_filter — guarded against above).  For the
        # Tier-2 union we require the (e, d) shape; the SCALAR-without-
        # period-filter case is caller error (helpers below always pass
        # period_filter=dt).
        if part.dims != ("e", "d"):
            raise FlexToolConfigError(
                f"_e_period_param_union expected ('e', 'd') dims from "
                f"class {cls!r} on parameter {parameter_name!r}; got "
                f"{part.dims}.  Pass a non-empty period_filter when "
                "calling the ed_* helpers."
            )
        lf = part.frame
        if not isinstance(lf, pl.LazyFrame):
            lf = lf.lazy()
        parts.append(lf.select("e", "d", "value"))
    if not parts:
        return None
    out = pl.concat(parts).collect()
    if out.height == 0:
        return None
    return Param(("e", "d"), out.lazy().sort("e", "d"))


def ed_invest_max_period_from_source(source: "InputSource",
                                       period_filter: pl.DataFrame | None = None,
                                       *,
                                       need_periods: "PeriodNeed | None" = None,
                                       ) -> Param | None:
    """``unit/node/connection.invest_max_period`` 1d_map(period) →
    ``Param(("e", "d"))``.  CSV path keeps zero rows (per
    ``_read_period_cap``); we preserve that.

    Δ.17c-Tier2 — uses ``resolve_param_shape`` + ``broadcast_to_period``
    per class via :func:`_e_period_param_union`.
    """
    return _e_period_param_union(source, "invest_max_period",
                                    filter_zero=False,
                                    period_filter=period_filter,
                                    need_periods=need_periods,
                                    methods=_INVEST_PERIOD_METHODS)


def ed_divest_max_period_from_source(source: "InputSource",
                                       period_filter: pl.DataFrame | None = None,
                                       *,
                                       need_periods: "PeriodNeed | None" = None,
                                       ) -> Param | None:
    """Δ.17c-Tier2 — see :func:`ed_invest_max_period_from_source`."""
    return _e_period_param_union(source, "retire_max_period",
                                    filter_zero=False,
                                    period_filter=period_filter,
                                    need_periods=need_periods,
                                    methods=_DIVEST_PERIOD_METHODS)


def ed_invest_min_period_from_source(source: "InputSource",
                                       period_filter: pl.DataFrame | None = None,
                                       *,
                                       need_periods: "PeriodNeed | None" = None,
                                       ) -> Param | None:
    """CSV path drops zero rows (``_read_e_d_param``).
    Δ.17c-Tier2 — see :func:`ed_invest_max_period_from_source`.
    """
    return _e_period_param_union(source, "invest_min_period",
                                    filter_zero=True,
                                    period_filter=period_filter,
                                    need_periods=need_periods,
                                    methods=_INVEST_PERIOD_METHODS)


def ed_divest_min_period_from_source(source: "InputSource",
                                       period_filter: pl.DataFrame | None = None,
                                       *,
                                       need_periods: "PeriodNeed | None" = None,
                                       ) -> Param | None:
    """Δ.17c-Tier2 — see :func:`ed_invest_max_period_from_source`."""
    return _e_period_param_union(source, "retire_min_period",
                                    filter_zero=True,
                                    period_filter=period_filter,
                                    need_periods=need_periods,
                                    methods=_DIVEST_PERIOD_METHODS)


def ed_cumulative_max_capacity_from_source(source: "InputSource",
                                             period_filter: pl.DataFrame | None = None,
                                             *,
                                             need_periods: "PeriodNeed | None" = None,
                                             ) -> Param | None:
    """An explicit ``cumulative_max_capacity = 0`` (a phase-out year) must bind, so zero rows are kept
    (``filter_zero=False``); ``check_max_cap`` (FA-12) then tells a genuinely
    missing period apart from an authored zero.  See :func:`ed_invest_max_period_from_source`."""
    return _e_period_param_union(source, "cumulative_max_capacity",
                                    filter_zero=False,
                                    period_filter=period_filter,
                                    need_periods=need_periods,
                                    methods=_CUMULATIVE_METHODS)


def ed_cumulative_min_capacity_from_source(source: "InputSource",
                                             period_filter: pl.DataFrame | None = None,
                                             *,
                                             need_periods: "PeriodNeed | None" = None,
                                             ) -> Param | None:
    """Δ.17c-Tier2 — see :func:`ed_invest_max_period_from_source`."""
    return _e_period_param_union(source, "cumulative_min_capacity",
                                    filter_zero=True,
                                    period_filter=period_filter,
                                    need_periods=need_periods,
                                    methods=_CUMULATIVE_METHODS)


# ---------------------------------------------------------------------------
# §1.9, §1.14 — relationship scalars (sink / source side params).
#
# CSV path (``_read_p_process_side``) reads ``input/p_process_sink.csv`` /
# ``input/p_process_source.csv`` and filters ``value != 0``.  We mirror.

def _p_side_scalar(source: "InputSource", entity_class: str,
                     parameter_name: str, side_dim: str,
                     entity_col: str = "unit",
                     *,
                     filter_zero: bool = True) -> Param | None:
    """``unit__inputNode`` / ``unit__outputNode`` per-(p, side) scalar.
    ``entity_col`` is the Spine column name for the unit ('unit' or
    'connection'); ``side_dim`` is 'source' or 'sink'.
    """
    try:
        df = source.parameter_explicit(entity_class, parameter_name)
    except (KeyError, AttributeError):
        try:
            df = source.parameter(entity_class, parameter_name)
        except KeyError:
            return None
    if df is None or df.height == 0:
        return None
    if "value" not in df.columns or entity_col not in df.columns \
            or "node" not in df.columns:
        return None
    lf = (df.lazy()
            .rename({entity_col: "p", "node": side_dim})
            .filter(pl.col("value").is_not_null()))
    if filter_zero:
        lf = lf.filter(pl.col("value") != 0.0)
    out = lf.select("p", side_dim, "value").collect()
    if out.height == 0:
        return None
    return Param(("p", side_dim), out.lazy())


# ramp_speed (§1.9) — relationship scalar, CSV filters zero
def p_ramp_speed_up_sink_from_source(source: "InputSource") -> Param | None:
    return _p_side_scalar(source, "unit__outputNode", "ramp_speed_up", "sink")


def p_ramp_speed_down_sink_from_source(source: "InputSource") -> Param | None:
    return _p_side_scalar(source, "unit__outputNode", "ramp_speed_down", "sink")


def p_ramp_speed_up_source_from_source(source: "InputSource") -> Param | None:
    return _p_side_scalar(source, "unit__inputNode", "ramp_speed_up", "source")


def p_ramp_speed_down_source_from_source(source: "InputSource") -> Param | None:
    return _p_side_scalar(source, "unit__inputNode", "ramp_speed_down", "source")


def p_process_sink_min_capacity_coef_from_source(source: "InputSource") -> Param | None:
    """``unit__outputNode.capacity_min_coeff`` → ``Param(("p", "sink"))``.

    Default 0 (schema).  Optional per-output floor of a multi-output unit:
    ``v_flow[p, p, k] ≥ v_online · min_load · capacity_min_coeff_k``
    (``minFlow_output_floor``), on top of the unit floor ``Σ outputs ≥
    v_online · min_load`` which carries no coefficient.  Explicit rows
    only (``filter_zero=False`` keeps an authored ``0.0``); absent arcs
    mean no per-output floor.
    """
    return _p_side_scalar(source, "unit__outputNode", "capacity_min_coeff",
                          "sink", filter_zero=False)


def p_process_sink_max_capacity_coef_from_source(source: "InputSource") -> Param | None:
    """``unit__outputNode.capacity_max_coeff`` → ``Param(("p", "sink"))``.

    Fraction of the unit's capacity available to this output edge
    (default 1.0; 0 = zero cap).  Explicit rows only, INCLUDING an
    authored ``0.0``; consumers densify absent arcs to the 1.0 default.
    """
    return _p_side_scalar(source, "unit__outputNode", "capacity_max_coeff",
                          "sink", filter_zero=False)


def p_process_source_input_share_max_from_source(source: "InputSource") -> Param | None:
    """``unit__inputNode.input_share_max`` → ``Param(("p", "source"))``.

    Input-side counterpart of :func:`p_process_sink_max_capacity_coef_from_source`
    (schema v70 renamed it from ``capacity_max_coeff``).  Explicit rows
    only, including an authored ``0.0``; consumers densify the 1.0
    default.
    """
    return _p_side_scalar(source, "unit__inputNode", "input_share_max",
                          "source", filter_zero=False)


def p_process_source_input_share_min_from_source(source: "InputSource") -> Param | None:
    """``unit__inputNode.input_share_min`` → ``Param(("p", "source"))``.

    Smallest share of the unit's current input energy (flow ×
    ``conversion_flow_coeff``) the input must supply while the unit runs
    (``minInputShare``).  Default 0 (no minimum).  Explicit rows only,
    including an authored ``0.0``.
    """
    return _p_side_scalar(source, "unit__inputNode", "input_share_min",
                          "source", filter_zero=False)


# inertia_constant (§1.14) — relationship scalar, CSV filters zero
def p_process_sink_inertia_constant_from_source(source: "InputSource") -> Param | None:
    return _p_side_scalar(source, "unit__outputNode", "inertia_constant", "sink")


def p_process_source_inertia_constant_from_source(source: "InputSource") -> Param | None:
    return _p_side_scalar(source, "unit__inputNode", "inertia_constant", "source")


# ---------------------------------------------------------------------------
# §1.10 — UC: startup_cost (1d_map period)

def p_startup_cost_from_source(source: "InputSource",
                                period_filter: pl.DataFrame | None = None,
                                ) -> Param | None:
    """``unit.startup_cost`` and ``connection.startup_cost`` → ``Param(("p", "d"))``.

    Δ.17c-Tier1 — uses :func:`._param_shapes.resolve_param_shape`.
    Allowed shapes: scalar / 1d_map[period], per class.  CSV path filters zero rows
    (``_load_online``: ``filter(value != 0)``); we mirror via
    ``filter_zero=True``.  The connection rows follow the unit rows.
    """
    parts = []
    for cls in ("unit", "connection"):
        resolved = resolve_param_shape(source, cls, "startup_cost", period_filter=period_filter)
        param = broadcast_to_period(resolved, "p", period_filter, filter_zero=True)
        if param is not None:
            parts.append(param)
    if len(parts) <= 1:
        return parts[0] if parts else None
    frames = [p.frame.lazy() if isinstance(p.frame, pl.DataFrame) else p.frame for p in parts]
    schema = frames[0].collect_schema()
    frames = [frames[0]] + [f.with_columns(pl.col(c).cast(schema[c]) for c in ("p", "d")) for f in frames[1:]]
    return Param(("p", "d"), pl.concat(frames).select("p", "d", "value"))


# ---------------------------------------------------------------------------
# §1.11 — Storage: Map(period→time) parameters

def p_node_availability_from_source(source: "InputSource",
                                     period_filter: pl.DataFrame | None = None,
                                     ) -> Param | None:
    """``node.availability`` → ``Param(("n", "d", "t"))``.

    Δ.17c — uses :func:`._param_shapes.resolve_param_shape`.  Allowed
    shapes: scalar / 1d_map[period] / 1d_map[time] / 2d_map[period,time].
    CSV path slices ``pdtNode.csv`` and filters to nodeState entities at
    the apply site; the helper here returns ALL availability rows.  The
    nodeState filter is applied downstream by ``_load_storage``.
    """
    resolved = resolve_param_shape(
        source, "node", "availability", period_filter=period_filter)
    return broadcast_to_period_time(resolved, "n", period_filter)


def p_storage_state_reference_value_from_source(source: "InputSource",
                                                period_filter: pl.DataFrame | None = None,
                                                ) -> Param | None:
    """``node.storage_state_reference_value`` → ``Param(("n", "d", "t"))``.

    Δ.17c — uses :func:`._param_shapes.resolve_param_shape`.  Allowed
    shapes: scalar / 1d_map[period] / 1d_map[time] / 2d_map[period,time].
    CSV path (``_load_storage``) gates on ``use_reference_value`` set;
    the gate stays on the consumer side.
    """
    resolved = resolve_param_shape(
        source, "node", "storage_state_reference_value",
        period_filter=period_filter)
    return broadcast_to_period_time(resolved, "n", period_filter)


# ---------------------------------------------------------------------------
# §1.4 — CO2: group price + period cap

def p_co2_price_from_source(source: "InputSource",
                             period_filter: pl.DataFrame | None = None,
                             ) -> Param | None:
    """``group.co2_price`` → ``Param(("g", "d", "t"))``.

    Δ.17c — uses :func:`._param_shapes.resolve_param_shape` to discover
    the actual shape from the DB and validate against the per-parameter
    allow-list (scalar / 1d_map[period] / 1d_map[time] /
    2d_map[period,time]).  CSV path slices ``pdtGroup.csv``
    (param='co2_price').  None default on the schema — explicit rows
    only.
    """
    resolved = resolve_param_shape(
        source, "group", "co2_price", period_filter=period_filter)
    return broadcast_to_period_time(resolved, "g", period_filter)


def p_co2_max_period_from_source(source: "InputSource",
                                  period_filter: pl.DataFrame | None = None,
                                  *,
                                  need_periods: "PeriodNeed | None" = None,
                                  ) -> Param | None:
    """``group.co2_max_period`` → ``Param(("g", "d"))``.

    Δ.17c — uses :func:`._param_shapes.resolve_param_shape`.  Allowed
    shapes: scalar / 1d_map[period].  CSV path keys this off
    ``inp/pd_group.csv`` slice.

    Gate-coupled note: at the consumer site this Param is only emitted
    into the LP when the (g, c, n) join with ``commodity_node_co2`` ×
    ``group__node`` is non-empty (see ``_load_co2_cap``).  We return the
    Param unconditionally — the consumer decides whether to wire it.

    A group whose ``co2_method`` caps the emissions per period must give
    ``co2_max_period`` (:func:`require_co2_cap_values`), and its period map
    must cover ``need_periods`` (the solve's periods;
    :func:`require_period_coverage`): either gap would otherwise be a cap of 0.
    """
    require_co2_cap_values(source, "co2_max_period", _CO2_MAX_PERIOD_METHOD)
    resolved = resolve_param_shape(
        source, "group", "co2_max_period", period_filter=period_filter)
    active = (_names_with_method(source, "group", "co2_method", _CO2_MAX_PERIOD_METHOD)
              if _gate_needed(need_periods, resolved) else None)
    return require_period_coverage(
        resolved, "g", period_filter, entity_class="group", parameter_name="co2_max_period",
        need_periods=need_periods, active=active)


def require_co2_cap_values(source: "InputSource", parameter_name: str, methods: "frozenset[str]") -> None:
    """Raise :class:`FlexToolConfigError` naming every group whose ``co2_method`` is one of ``methods``
    but which gives no ``parameter_name`` (neither a constant nor a map).

    Without the check such a group gets a CO2 cap of 0 in every period (``co2_max_period``) or a
    developer error when no group gives a value at all.
    """
    gated = _names_with_method(source, "group", "co2_method", methods)
    if not gated:
        return
    try:
        df = source.parameter("group", parameter_name)
    except KeyError:
        df = None
    given: set[str] = set()
    if df is not None and df.height > 0 and "value" in df.columns:
        value = pl.col("value").cast(pl.Float64, strict=False)
        given = set(df.filter(value.is_not_null() & value.is_not_nan())
                      .get_column("name").cast(pl.Utf8).to_list())
    lacking = sorted(gated - given)
    if not lacking:
        return
    method = source.parameter("group", "co2_method")
    by_group = {str(n): str(v) for n, v in method.select(pl.col("name").cast(pl.Utf8),
                                                         pl.col("value").cast(pl.Utf8)).iter_rows()}
    raise FlexToolConfigError("; ".join(
        f"group '{g}', co2_method '{by_group.get(g, '')}' needs parameter '{parameter_name}' (no value is given)"
        for g in lacking) + ".")


# ---------------------------------------------------------------------------
# §1.12 — Variable cost (other_operational_cost) Map(period→time)

def p_pdt_varCost_source_from_source(
    source: "InputSource",
    period_filter: pl.DataFrame | None = None,
) -> Param | None:
    """``unit__inputNode.other_operational_cost`` Map(period→time) →
    ``Param(("p", "source", "d", "t"))``.  CSV path filters zero.

    Δ.17c-Tier3 — routes through ``resolve_param_shape`` +
    ``broadcast_to_period_time`` to carry silent-default ``index_name``
    disambiguation; the per-source-column rename dict maps the
    relationship class's ``(unit, node)`` columns to ``(p, source)``.
    """
    resolved = resolve_param_shape(
        source, "unit__inputNode", "other_operational_cost",
        period_filter=period_filter)
    return broadcast_to_period_time(
        resolved, {"unit": "p", "node": "source"},
        period_filter, filter_zero=True)


def p_pdt_varCost_sink_from_source(
    source: "InputSource",
    period_filter: pl.DataFrame | None = None,
) -> Param | None:
    """``unit__outputNode.other_operational_cost`` Map(period→time) →
    ``Param(("p", "sink", "d", "t"))``.

    Δ.17c-Tier3 — see :func:`p_pdt_varCost_source_from_source`.
    """
    resolved = resolve_param_shape(
        source, "unit__outputNode", "other_operational_cost",
        period_filter=period_filter)
    return broadcast_to_period_time(
        resolved, {"unit": "p", "node": "sink"},
        period_filter, filter_zero=True)


def p_pdt_varCost_process_from_source(
    source: "InputSource",
    period_filter: pl.DataFrame | None = None,
) -> Param | None:
    """``connection.other_operational_cost`` Map(period→time) →
    ``Param(("p", "d", "t"))``.  Single-class helper.

    Δ.17c-Tier3 — the call routes through ``resolve_param_shape`` +
    ``broadcast_to_period_time``.  Pre-migration the helper gated on
    ``{"name", "period", "t"}.issubset``, silently dropping silent-
    default-Map authoring (column ``"x"`` not ``"period"``) and
    SCALAR / MAP_PERIOD / MAP_TIME shapes.

    SCALAR / MAP_PERIOD / MAP_TIME branches return Params with fewer
    dims than the canonical ``(p, d, t)``.  Results are broadcast up
    to ``(p, d, t)`` against ``period_filter``'s ``(d, t)`` axis so
    the output frame is dim-uniform.  Callers (the dispatcher) always
    pass a non-empty ``period_filter``; if the Param is a strict
    subset of ``(p, d, t)`` AND no period_filter is on hand, we raise
    rather than emit an incompatible-shape frame.
    """
    parts: list[pl.LazyFrame] = []
    for cls in ("connection",):
        resolved = resolve_param_shape(
            source, cls, "other_operational_cost",
            period_filter=period_filter)
        part = broadcast_to_period_time(
            resolved, {"name": "p"}, period_filter, filter_zero=True)
        if part is None:
            continue
        lf = part.frame
        if not isinstance(lf, pl.LazyFrame):
            lf = lf.lazy()
        if part.dims == ("p", "d", "t"):
            parts.append(lf.select("p", "d", "t", "value"))
            continue
        # Lower-dim Param shapes (SCALAR / MAP_PERIOD / MAP_TIME) — the
        # union concat requires (p, d, t); broadcast the missing axes
        # against ``period_filter`` so the multi-class union stays
        # dim-uniform.  Mirrors the legacy concat semantic (the inline
        # helper accepted only MAP_PERIOD_TIME-shaped sources because
        # of its column gate; Tier-3 widens that to the full allow-list
        # by broadcasting the lower-dim shapes here).
        if period_filter is None or period_filter.height == 0:
            raise FlexToolConfigError(
                f"p_pdt_varCost_process_from_source: class {cls!r} "
                f"resolved to Param dims {part.dims}; cannot broadcast "
                "to ('p', 'd', 't') without a period_filter.  Pass "
                "period_filter=dt from the dispatcher.")
        if part.dims == ("p",):
            # SCALAR — cross-join with full (d, t).
            dt_lf = period_filter.lazy().select("d", "t").unique()
            lf = lf.join(dt_lf, how="cross").select("p", "d", "t", "value")
        elif part.dims == ("p", "d"):
            # MAP_PERIOD — cross-join across t within each d.
            t_lf = period_filter.lazy().select("d", "t").unique()
            lf = (lf.join(t_lf, on="d", how="inner")
                    .select("p", "d", "t", "value"))
        elif part.dims == ("p", "t"):
            # MAP_TIME — cross-join across d within each t.
            d_lf = period_filter.lazy().select("d", "t").unique()
            lf = (lf.join(d_lf, on="t", how="inner")
                    .select("p", "d", "t", "value"))
        else:  # pragma: no cover — guarded by allow-list above.
            raise FlexToolConfigError(
                f"p_pdt_varCost_process_from_source: unexpected per-class "
                f"Param dims {part.dims}")
        parts.append(lf)
    if not parts:
        return None
    out = pl.concat(parts).collect()
    if out.height == 0:
        return None
    return Param(("p", "d", "t"), out.lazy().sort("p", "d", "t"))


_PSS_KEYS = ("p", "source", "sink")
_PSSDT_KEYS = (*_PSS_KEYS, "d", "t")


def _varcost_side(param: "Param | None", keys: tuple[str, ...], pss: pl.DataFrame,
                  dt: pl.DataFrame) -> "pl.DataFrame | None":
    """``param`` on the solve's ``(d, t)`` pairs as ``[*keys, d, t, value]``, rows with a non-zero value only.

    The key columns are cast to the dtypes of ``pss`` (``p`` / ``source`` / ``sink``) and ``dt`` (``d`` / ``t``)
    so the joins below compare like with like: the per-side Params carry the cast of their own read (a Utf8
    seed, an Enum of the read's axis), ``process_source_sink`` the entity-union axis.
    """
    if param is None:
        return None
    target = {**{c: pss.schema[c] for c in keys}, "d": dt.schema["d"], "t": dt.schema["t"]}
    df = (promote_param_to_dt(param, dt)
          .select(*keys, "d", "t", pl.col("value").cast(pl.Float64))
          .filter(pl.col("value") != 0.0)
          .collect())
    mismatched = [c for c, dtype in target.items() if df.schema[c] != dtype]
    if mismatched:
        # A value outside the target vocabulary becomes null and joins nothing (it has no arc / step).
        df = df.with_columns(pl.col(c).cast(pl.Utf8).cast(target[c], strict=False) for c in mismatched)
    # A period × time Param passes promote_param_to_dt unchanged: keep the solve's (d, t) pairs only.
    return df.join(dt, on=["d", "t"], how="semi")


def apply_varcost_params(flex_data: object) -> None:
    """The variable-cost index sets and ``p_pssdt_varCost`` from the three per-side cost Params.

        src[p, s, d, t]  = p_pdt_varCost_source  on the solve's (d, t)     unit__inputNode.other_operational_cost
        snk[p, k, d, t]  = p_pdt_varCost_sink    on the solve's (d, t)     unit__outputNode.other_operational_cost
        proc[p, d, t]    = p_pdt_varCost_process on the solve's (d, t)     connection.other_operational_cost
                           (each: rows with value ≠ 0; every authored shape, broadcast by promote_param_to_dt)

        pssdt_varCost_eff_unit_source = {(p,s,k,d,t) : (p,s,k) ∈ process_source_sink_eff, src[p,s,d,t] exists}
        pssdt_varCost_eff_unit_sink   = {(p,s,k,d,t) : (p,s,k) ∈ process_source_sink_eff, snk[p,k,d,t] exists}
        pssdt_varCost_eff_connection  = {(p,s,k,d,t) : (p,s,k) ∈ process_source_sink_eff, proc[p,d,t] exists}
        p_pssdt_varCost[p,s,k,d,t]    = src[p,s,d,t] + snk[p,k,d,t] + proc[p,d,t]     (a missing side adds 0)
                                        ∀ (p,s,k) ∈ process_source_sink; rows with value ≠ 0
        pssdt_varCost_noEff           = {(p,s,k,d,t) ∈ keys(p_pssdt_varCost) : (p,s,k) ∈ process_source_sink_noEff}

    The emit's rule (``emit/_period_params._derive_varCost_pair`` / ``_derive_pssdt_varCost_filters``) on the
    DB-direct Params: the per-side Params read every shape (scalar, period, time, period × time, stochastic
    time) on every solve type, while the emit seeds have no period × time tier.  The ``(p, s) ∈
    process__source`` / ``(p, k) ∈ process__sink`` gates of the emit follow from the joins (the per-side
    Params are keyed by the unit's own arcs).  :func:`apply_direct_params_b` calls it right after it writes
    the three per-side Params (later steps there write none of the five fields), so on registered solves and
    synthetic sub-solves alike; the loader seeds none of the five, and a reader-less CSV reload calls it on
    the emit seeds of the per-side Params (``_csv_reload._field_seeds``).  Every set is an eager frame
    ``[p, source, sink, d, t]`` sorted by its columns, ``p_pssdt_varCost`` an eager Param sorted the same way;
    an empty result is None (the objective term is skipped).
    """
    for field in ("pssdt_varCost_noEff", "pssdt_varCost_eff_unit_source", "pssdt_varCost_eff_unit_sink",
                  "pssdt_varCost_eff_connection", "p_pssdt_varCost"):
        setattr(flex_data, field, None)
    dt = getattr(flex_data, "dt", None)
    pss = getattr(flex_data, "process_source_sink", None)
    if dt is None or dt.height == 0 or pss is None or pss.height == 0:
        return
    pss = pss.select(_PSS_KEYS)
    dt = dt.select("d", "t").unique(maintain_order=True)
    sides = (
        (_varcost_side(getattr(flex_data, "p_pdt_varCost_source", None), ("p", "source"), pss, dt),
         ("p", "source"), "pssdt_varCost_eff_unit_source"),
        (_varcost_side(getattr(flex_data, "p_pdt_varCost_sink", None), ("p", "sink"), pss, dt),
         ("p", "sink"), "pssdt_varCost_eff_unit_sink"),
        (_varcost_side(getattr(flex_data, "p_pdt_varCost_process", None), ("p",), pss, dt),
         ("p",), "pssdt_varCost_eff_connection"),
    )

    def _keys(frame: "pl.DataFrame | None") -> "pl.DataFrame | None":
        # The arc sets on pss's key dtypes (they come from the same producer; a seed may differ).
        if frame is None or frame.height == 0:
            return None
        frame = frame.select(_PSS_KEYS)
        mismatched = [c for c in _PSS_KEYS if frame.schema[c] != pss.schema[c]]
        if mismatched:
            frame = frame.with_columns(pl.col(c).cast(pl.Utf8).cast(pss.schema[c], strict=False)
                                       for c in mismatched)
        return frame

    def _on(base: pl.DataFrame, part: pl.DataFrame, on: tuple[str, ...]) -> pl.DataFrame:
        return base.join(part, on=list(on), how="inner").select(*_PSSDT_KEYS, "value")

    eff = _keys(getattr(flex_data, "process_source_sink_eff", None))
    summed_parts: list[pl.DataFrame] = []
    for part, on, field in sides:
        if part is None or part.height == 0:
            continue
        summed_parts.append(_on(pss, part, on))
        if eff is not None:
            idx = _on(eff, part, on).select(_PSSDT_KEYS).sort(_PSSDT_KEYS)
            setattr(flex_data, field, idx if idx.height > 0 else None)
    if not summed_parts:
        return
    summed = (pl.concat(summed_parts)
              .group_by(_PSSDT_KEYS).agg(pl.col("value").sum())
              .filter(pl.col("value") != 0.0)
              .sort(_PSSDT_KEYS))
    if summed.height == 0:
        return
    flex_data.p_pssdt_varCost = Param(_PSSDT_KEYS, summed)
    noeff = _keys(getattr(flex_data, "process_source_sink_noEff", None))
    if noeff is not None:
        idx = (summed.select(_PSSDT_KEYS)
               .join(noeff, on=list(_PSS_KEYS), how="semi")
               .sort(_PSSDT_KEYS))
        flex_data.pssdt_varCost_noEff = idx if idx.height > 0 else None


# ---------------------------------------------------------------------------
# §1.15 — Reserves

def pdtReserve_upDown_group_reservation_from_source(
    source: "InputSource",
    period_filter: pl.DataFrame | None = None,
) -> Param | None:
    """``reserve__upDown__group.reservation`` Map(period→time) →
    ``Param(("r", "ud", "g", "d", "t"))``.  The schema default is None (not authored); the CSV path
    emits explicit rows only (zero values are NOT filtered), and the consumer's constraint RHS treats
    an absent entity as 0 (no reservation requirement).

    Δ.17c-Tier3 — routes through ``resolve_param_shape`` +
    ``broadcast_to_period_time`` with a per-source-column rename dict
    mapping the 3-dim relationship class's ``(reserve, upDown, group)``
    columns to ``(r, ud, g)``.  ``filter_zero=False`` because an
    explicit zero row is authoring-meaningful (legacy CSV path
    preserved zero rows for this param); an entity with no key in the
    active solve (the silent-label per-entity fallback, #168) is absent
    entirely, the same as the None default.
    """
    resolved = resolve_param_shape(
        source, "reserve__upDown__group", "reservation",
        period_filter=period_filter)
    return broadcast_to_period_time(
        resolved,
        {"reserve": "r", "upDown": "ud", "group": "g"},
        period_filter, filter_zero=False)


#: Default of ``reserve__upDown__group.penalty_reserve`` [CUR/MW per hour
#: of shortfall].  The GMPL-era ``flextool.mod`` defaulted it to 5000
#: (``reserveParam_defaults``) and the schema carried the same default
#: from v18 until v56 cleared it; v70 restores it in the schema.  The
#: engine applies it to every reserve group without an authored value.
PENALTY_RESERVE_DEFAULT: float = 5000.0


def p_reserve_upDown_group_penalty_reserve_from_source(source: "InputSource") -> Param | None:
    """``reserve__upDown__group.penalty_reserve`` scalar →
    ``Param(("r", "ud", "g"))``, dense over every ``reserve__upDown__group``
    entity: the authored value where one exists, otherwise
    :data:`PENALTY_RESERVE_DEFAULT`.  (Densified here rather than relying on
    the schema default so the reserve shortfall is never free.)
    """
    keys = ["reserve", "upDown", "group"]
    try:
        ents = source.entities("reserve__upDown__group")
    except KeyError:
        ents = None
    try:
        df = source.parameter_explicit("reserve__upDown__group", "penalty_reserve")
    except (KeyError, AttributeError):
        try:
            df = source.parameter("reserve__upDown__group", "penalty_reserve")
        except KeyError:
            df = None
    if df is not None and not {*keys, "value"}.issubset(df.columns):
        df = None
    if ents is None or not set(keys).issubset(ents.columns):
        ents = df.select(keys) if df is not None else None
    if ents is None or ents.height == 0:
        return None
    lf = ents.lazy().select(keys).unique(maintain_order=True)
    if df is not None and df.height > 0:
        authored = (df.lazy()
                    .filter(pl.col("value").is_not_null())
                    .select(*[pl.col(k).cast(pl.Utf8).alias(f"__{k}")
                              for k in keys],
                            pl.col("value").cast(pl.Float64)))
        lf = (lf.with_columns(pl.col(k).cast(pl.Utf8).alias(f"__{k}")
                              for k in keys)
                .join(authored, on=[f"__{k}" for k in keys], how="left")
                .drop([f"__{k}" for k in keys]))
    else:
        lf = lf.with_columns(value=pl.lit(None, dtype=pl.Float64))
    out = (lf.with_columns(pl.col("value").fill_null(PENALTY_RESERVE_DEFAULT))
             .rename({"reserve": "r", "upDown": "ud", "group": "g"})
             .select("r", "ud", "g", "value")
             .collect())
    return Param(("r", "ud", "g"), out.lazy())


def p_reserve_upDown_group_reserve_duration_from_source(source: "InputSource") -> Param | None:
    """``reserve__upDown__group.reserve_duration`` scalar →
    ``Param(("r", "ud", "g"))``.  None default — explicit rows only.

    Hours over which committed reserve power must be sustainable from a
    storage-backed provider.  Absent (or 0) rows leave the LP untouched
    (no storage-energy coupling) — see ``_reserve._add_storage_energy_constraints``.
    """
    try:
        df = source.parameter_explicit("reserve__upDown__group", "reserve_duration")
    except (KeyError, AttributeError):
        try:
            df = source.parameter("reserve__upDown__group", "reserve_duration")
        except KeyError:
            return None
    if df is None or df.height == 0:
        return None
    cols = df.columns
    if not {"reserve", "upDown", "group", "value"}.issubset(cols):
        return None
    lf = (df.lazy()
            .rename({"reserve": "r", "upDown": "ud", "group": "g"})
            .filter(pl.col("value").is_not_null()))
    out = lf.select("r", "ud", "g", "value").collect()
    if out.height == 0:
        return None
    return Param(("r", "ud", "g"), out.lazy())


def p_reserve_upDown_group_increase_reserve_ratio_from_source(source: "InputSource") -> Param | None:
    """``reserve__upDown__group.increase_reserve_ratio`` scalar → ``Param(("r", "ud", "g"))``.

    Explicit rows with a value > 0 only, sorted; None when there is none.  The ratio of the group's
    demand added to its dynamic requirement (``_reserve.group_demand_requirement``).  The schema types it
    ``float`` only, so there is no map shape to resolve over ``(d, t)``.
    """
    try:
        df = source.parameter_explicit("reserve__upDown__group", "increase_reserve_ratio")
    except (KeyError, AttributeError):
        try:
            df = source.parameter("reserve__upDown__group", "increase_reserve_ratio")
        except KeyError:
            return None
    if df is None or df.height == 0 or not {"reserve", "upDown", "group", "value"}.issubset(df.columns):
        return None
    out = (df.lazy()
             .rename({"reserve": "r", "upDown": "ud", "group": "g"})
             .with_columns(pl.col("value").cast(pl.Float64, strict=False))
             .filter(pl.col("value") > 0.0)
             .select("r", "ud", "g", "value")
             .sort("r", "ud", "g")
             .collect())
    if out.height == 0:
        return None
    return Param(("r", "ud", "g"), out.lazy())


def _process_reserve_node_param(source: "InputSource",
                                  parameter_name: str) -> Param | None:
    """Union ``reserve__upDown__unit__node`` ∪
    ``reserve__upDown__connection__node`` for a per-(p, r, ud, n) scalar.

    CSV path (``_read_p_process_reserve_node`` style) emits explicit rows
    only; we preserve via ``parameter_explicit``.
    """
    parts: list[pl.LazyFrame] = []
    for cls, ent in (("reserve__upDown__unit__node", "unit"),
                       ("reserve__upDown__connection__node", "connection")):
        try:
            df = source.parameter_explicit(cls, parameter_name)
        except (KeyError, AttributeError):
            try:
                df = source.parameter(cls, parameter_name)
            except KeyError:
                continue
        if df is None or df.height == 0:
            continue
        cols = df.columns
        if not {"reserve", "upDown", ent, "node", "value"}.issubset(cols):
            continue
        parts.append(df.lazy()
                       .rename({ent: "p", "reserve": "r",
                                 "upDown": "ud", "node": "n"})
                       .filter(pl.col("value").is_not_null())
                       .select("p", "r", "ud", "n", "value"))
    if not parts:
        return None
    out = pl.concat(parts).collect()
    if out.height == 0:
        return None
    return Param(("p", "r", "ud", "n"), out.lazy().sort("p", "r", "ud", "n"))


def p_process_reserve_upDown_node_reliability_from_source(source: "InputSource") -> Param | None:
    """``reserve__upDown__{unit,connection}__node.reliability`` →
    ``Param(("p", "r", "ud", "n"))``.  Default 1.0 (schema) — broadcast.
    The CSV path uses the broadcast default; here we delegate to
    ``parameter`` (not ``parameter_explicit``) so the default-fill kicks
    in.  Result is the same shape as the CSV path's
    ``Param(("p", "r", "ud", "n"))``.
    """
    parts: list[pl.LazyFrame] = []
    for cls, ent in (("reserve__upDown__unit__node", "unit"),
                       ("reserve__upDown__connection__node", "connection")):
        try:
            df = source.parameter(cls, parameter_name="reliability")
        except KeyError:
            continue
        if df is None or df.height == 0:
            continue
        cols = df.columns
        if not {"reserve", "upDown", ent, "node", "value"}.issubset(cols):
            continue
        parts.append(df.lazy()
                       .rename({ent: "p", "reserve": "r",
                                 "upDown": "ud", "node": "n"})
                       .filter(pl.col("value").is_not_null())
                       .select("p", "r", "ud", "n", "value"))
    if not parts:
        return None
    out = pl.concat(parts).collect()
    if out.height == 0:
        return None
    return Param(("p", "r", "ud", "n"), out.lazy().sort("p", "r", "ud", "n"))


def p_process_reserve_upDown_node_max_share_from_source(source: "InputSource") -> Param | None:
    return _process_reserve_node_param(source, "max_share")


def p_process_reserve_upDown_node_large_failure_ratio_value_from_source(
    source: "InputSource") -> Param | None:
    return _process_reserve_node_param(source, "large_failure_ratio")


def p_process_reserve_upDown_node_increase_reserve_ratio_value_from_source(
    source: "InputSource") -> Param | None:
    return _process_reserve_node_param(source, "increase_reserve_ratio")


# ---------------------------------------------------------------------------
# Penalty / availability scalars (§1.2 / §1.7) — schema-default broadcast.
#
# These differ from the "explicit only" pattern: schema sentinel default
# (e.g. 10000.0 for penalty_up) means the CSV path receives one row per
# entity with the broadcast default.  ``SpineDbReader.parameter()``
# already broadcasts when ``default is not None``; we just rename.

def _entity_scalar_with_default(source: "InputSource", entity_class: str,
                                  parameter_name: str,
                                  dim: str) -> Param | None:
    """Variant of :func:`_entity_scalar_explicit` that uses
    :meth:`parameter` (default-broadcast) instead of ``parameter_explicit``.
    """
    try:
        df = source.parameter(entity_class, parameter_name)
    except KeyError:
        return None
    if df is None or df.height == 0:
        return None
    return Param((dim,),
                 df.lazy().rename({"name": dim}).select(dim, "value"))


# ---------------------------------------------------------------------------
# §1.7, §1.3 — additional Map(period→time) scalars on object classes.

def p_process_availability_from_source(source: "InputSource",
                                        period_filter: pl.DataFrame | None = None,
                                        ) -> Param | None:
    """``unit/connection.availability`` → Param keyed on ``"p"`` plus
    whichever of ``"d"`` / ``"t"`` the union of unit + connection
    authoring requires.

    Δ.17c — uses :func:`._param_shapes.resolve_param_shape` for each of
    ``unit.availability`` and ``connection.availability`` independently
    (Spine carries them on separate classes; the LP consumes the union).
    Allowed shapes per class: scalar / 1d_map[period] / 1d_map[time] /
    2d_map[period,time].  CSV path slices ``pdtProcess.csv``
    (param='availability') and emits explicit rows only.

    Phase E.1 — each per-class Param keeps its authored dims under the
    lazified broadcast helpers.  When the two classes carry different
    shapes (e.g. one SCALAR, one MAP_TIME) we promote each part to the
    union of their dims via lazy joins on ``period_filter`` before the
    concat, so the output Param has a single consistent dim shape but
    the upstream LazyFrames remain unmaterialised.
    """
    parts: list[Param] = []
    for cls in ("unit", "connection"):
        resolved = resolve_param_shape(
            source, cls, "availability", period_filter=period_filter)
        param = broadcast_to_period_time(resolved, "p", period_filter)
        if param is None:
            continue
        parts.append(param)
    if not parts:
        return None
    # Union of all per-part dims preserving order: always starts with
    # "p"; "d" / "t" appear only when at least one part authored them.
    union_dims: tuple[str, ...] = ("p",)
    if any("d" in pt.dims for pt in parts):
        union_dims = union_dims + ("d",)
    if any("t" in pt.dims for pt in parts):
        union_dims = union_dims + ("t",)
    # Promote each part to the union dims via lazy joins on
    # period_filter.  No-op when the part already carries union_dims.
    pf_lf = (period_filter.lazy() if period_filter is not None
                                  and period_filter.height > 0 else None)
    part_lfs: list[pl.LazyFrame] = []
    for pt in parts:
        lf = pt.lazy
        missing = [d for d in union_dims if d not in pt.dims]
        if missing:
            if pf_lf is None:
                # Cannot promote without a period_filter to source the
                # missing axis values — defensive guard, the helper's
                # contract requires a non-empty filter for any broadcast.
                return None
            # Build the (missing-dim) universe from period_filter.
            uni = pf_lf.select(missing).unique()
            lf = lf.join(uni, how="cross")
        lf = lf.select(*union_dims, "value")
        part_lfs.append(lf)
    out_lf = pl.concat(part_lfs).sort(*union_dims)
    return Param(union_dims, out_lf)


def p_commodity_price_from_source(source: "InputSource",
                                   period_filter: pl.DataFrame | None = None,
                                   ) -> Param | None:
    """``commodity.price`` → ``Param(("c", "d", "t"))``.

    Δ.17c — uses :func:`._param_shapes.resolve_param_shape`.  Allowed
    shapes: scalar / 1d_map[period] / 1d_map[time] (no 2d_map per
    flextool's ``write_pdtCommodity`` cascade pt → pd → p → 0).  None
    default on the schema; CSV path slices ``pdtCommodity.csv``
    (param='price') and emits explicit rows only.
    """
    resolved = resolve_param_shape(
        source, "commodity", "price", period_filter=period_filter)
    return broadcast_to_period_time(resolved, "c", period_filter)


# ---------------------------------------------------------------------------
# §1.19 — Commodity ladder split (price / quantity sub-frames).  Δ.17c-per-c25:
#
# Spine schema: ``commodity.price_ladder_annual`` is EITHER a depth-2 Map
# ``(tier → {"price","quantity"} → f64)`` -- the same ladder in every model
# period -- OR a depth-3 Map ``(period → tier → {"price","quantity"} → f64)``.
# ``commodity.price_ladder_cumulative`` is depth-2 only, period-agnostic.
# Column-name detection (the pre-c25 code) misread a depth-2 ladder (no
# ``period`` column at all) and any ``x``-labelled ladder (no ``tier`` column
# either -- the reader names the tier level ``x``/``x_2`` too when its own
# ``index_name`` is unset) as unreadable, and a MIX of depths on different
# commodities as positionally scrambled (the reader discovers index columns
# from the single deepest row and places every other row's values there by
# position).  :func:`._param_shapes.resolve_facet_groups` reads each entity's
# raw label PATH (never its text) and groups by depth instead, so every depth
# is read on its own, depth-pure view.  We split the resulting long frame into
# two Params keyed ``(c, i, d)`` (annual) / ``(c, i)`` (cumulative) with
# ``value`` being the price (resp. quantity) only.

def _check_ladder_tiers_have_prices(parameter_name: str, long: pl.LazyFrame, dims: tuple[str, ...]) -> None:
    """Raise when a tier of a price ladder has no price (a tier without a quantity is unlimited, but every tier
    needs a price).  ``long`` is the ladder's long frame ``(c, i, [d], facet, value)``; a depth-2 annual
    ladder already carries the solve's periods in ``d``."""
    off_depth = long.filter(pl.col("facet").is_null()).select("c").unique().sort("c").collect()
    if off_depth.height > 0:
        names = ", ".join(f"'{c}'" for c in off_depth["c"].cast(pl.Utf8).to_list())
        raise FlexToolConfigError(
            f"commodity {names}, parameter '{parameter_name}': the value has keys at different index depths "
            "(some entries have fewer levels than the others), so they cannot be read as tiers. Use one form "
            "for the whole value: tier -> price / quantity, or period -> tier -> price / quantity.")
    missing = (long.group_by(*dims)
                   .agg(pl.col("facet").cast(pl.Utf8).eq("price").any().alias("has_price"),
                        pl.col("facet").cast(pl.Utf8).eq("quantity").any().alias("has_quantity"))
                   .filter(~pl.col("has_price"))
                   .sort(*dims)
                   .collect())
    if missing.height == 0:
        return
    lines = []
    for row in missing.iter_rows(named=True):
        where = f" (period {row['d']})" if "d" in row else ""
        what = "a quantity but no price" if row["has_quantity"] else "no price"
        lines.append(f"tier '{row['i']}' of commodity '{row['c']}'{where} has {what}")
    raise FlexToolConfigError(
        f"commodity.{parameter_name}: every tier needs a price ({'; '.join(lines)}). Add the price, or remove "
        "the tier.")


def _ladder_split(source: "InputSource", parameter_name: str,
                    *, period_filter: "pl.DataFrame | None") -> tuple[Param | None, Param | None]:
    """Return ``(price_param, quantity_param)`` from the named ladder parameter, read by shape at every
    depth it is authored.  ``period_filter`` is the active solve's ``dt`` for the annual ladder (a depth-2
    group is cross-joined with it -- ``._param_shapes.facet_long_frame``'s "same ladder in every period"
    rule) and ``None`` for the cumulative ladder (period-agnostic; a depth-2 group is never cross-joined).

    A depth-2 annual group with ``period_filter`` ``None`` or empty (no active solve periods) gives no
    rows for that group -- the same "nothing to broadcast over" outcome the pre-c25 seed gave in that
    edge case, kept rather than inventing period-less ladder rows.
    """
    groups = resolve_facet_groups(source, "commodity", parameter_name)
    if not groups:
        return None, None
    is_annual = parameter_name == "price_ladder_annual"
    frames: list[pl.DataFrame] = []
    for resolved in groups:
        if (is_annual and resolved.shape is Shape.MAP_TIER_FACET
                and (period_filter is None or period_filter.height == 0)):
            continue
        frames.append(facet_long_frame(resolved, period_filter))
    if not frames:
        return None, None
    long = pl.concat(frames, how="diagonal_relaxed")
    ent_col = groups[0].entity_dim_columns[0]
    base = (long.lazy()
              .rename({ent_col: "c"} if ent_col != "c" else {})
              .filter(pl.col("value").is_not_null())
              .with_columns(pl.col("i").cast(pl.Utf8)))
    out_dims = ("c", "i", "d") if "d" in long.columns else ("c", "i")
    price_lf = base.filter(pl.col("facet") == "price").select(*out_dims, "value")
    qty_lf   = base.filter(pl.col("facet") == "quantity").select(*out_dims, "value")
    price_df = price_lf.collect()
    qty_df = qty_lf.collect()
    _check_ladder_tiers_have_prices(parameter_name, base, out_dims)
    p_price = Param(out_dims, price_df) if price_df.height > 0 else None
    p_qty = Param(out_dims, qty_df) if qty_df.height > 0 else None
    return p_price, p_qty


def apply_direct_params_a(source: "InputSource",
                            flex_data: object) -> None:
    """Pass 1a of Direct Params — the dt-independent assignments.

    Δ.28 — split from the legacy single-pass ``apply_direct_params``
    so the dispatcher can run :func:`apply_derived_a` (which populates
    ``flex_data.dt``) BEFORE the broadcast-needing assignments in
    :func:`apply_direct_params_b` execute.  See module docstring on the
    bug Δ.28 fixes (fast path's ``flex_data.dt`` is empty when
    ``apply_direct_params`` runs, so scalar / 1d_map[period] /
    1d_map[time] values authored on the source can't broadcast across
    the active solve's ``(d, t)`` axis and the Param ends up empty —
    e.g. ``p_commodity_price`` on ``work_lh2_three_region`` was empty
    on the fast path even though Spine carries a scalar 30 for coal).

    The slow path still calls these passes in sequence and the legacy
    behaviour is preserved exactly — slow-path ``flex_data.dt`` is
    already populated from CSVs by ``_load_*`` before
    ``_apply_db_overrides`` runs, so pass-1a / pass-1b ordering is a
    no-op there.

    This pass writes:
        * §5.2.1 scalar Params (``p_co2_content`` etc.).
        * §5.2.3 relationship 1d_map (constraint coefficients).
        * §5.2.1 invest/divest total caps.
        * §1.16 second-wave scalars.

    All of these consume ``source`` only — no ``flex_data.dt``
    dependency.

    Δ.12b — assignment is unconditional for the scalar / 1d_map and
    relationship-1d_map helpers (these read the source unchanged and
    return None only when no upstream row exists, which is a
    legitimate "feature inactive" outcome).
    """
    # ─── §5.2.1 scalar Params with FlexData fields — Δ.12b unconditional
    flex_data.p_co2_content = p_co2_content_from_source(source)

    # ─── §5.2.3 relationship 1d_map (constraint coefficients) ───────────
    flex_data.p_node_constraint_invested_capacity_coeff = (
        _node_constraint_coef(source, "constraint_invested_capacity_coeff"))
    flex_data.p_process_constraint_invested_capacity_coeff = (
        _process_constraint_coef(source, "constraint_invested_capacity_coeff"))
    flex_data.p_node_constraint_state_coeff = (
        _node_constraint_coef(source, "constraint_state_coeff"))
    flex_data.p_node_constraint_prebuilt_capacity_coeff = (
        _node_constraint_coef(source,
                              "constraint_cumulative_pre_built_capacity_coeff"))
    flex_data.p_process_constraint_prebuilt_capacity_coeff = (
        _process_constraint_coef(source,
                                 "constraint_cumulative_pre_built_capacity_coeff"))
    for kind in ("capacity", "retired_capacity", "in_construction_capacity"):
        setattr(flex_data, f"p_node_constraint_{kind}_coeff",
                _node_constraint_coef(source, f"constraint_{kind}_coeff"))
        setattr(flex_data, f"p_process_constraint_{kind}_coeff",
                _process_constraint_coef(source, f"constraint_{kind}_coeff"))
    # Units online (count of unitsize) — constant per (p, cn); no (d,t) axis.
    flex_data.p_process_constraint_online_coeff = (
        _process_constraint_coef(source, "constraint_online_coeff"))

    # ─── §5.2.1 invest/divest total caps (entity-unioned) ───────────────
    # Max variants: keyed on entityInvest (resp. entityDivest), one row
    # per entity with value defaulting to 0 — mirrors
    # ``entity_total_caps.py:_compute_entity_total`` and
    # ``flex_data.py::_e_total_param``.
    flex_data.e_invest_max_total = _e_total_param(
        source, "invest_max_total", kind="invest")
    flex_data.e_divest_max_total = _e_total_param(
        source, "retire_max_total", kind="divest")
    # Min variants: CSV path filters out zero rows (flex_data.py::_read_e_param)
    # — None when no entity has an explicit non-zero min cap.
    flex_data.e_invest_min_total = _e_total_param(
        source, "invest_min_total", kind="invest", filter_zero=True)
    flex_data.e_divest_min_total = _e_total_param(
        source, "retire_min_total", kind="divest", filter_zero=True)

    # ─── Δ.4 second wave — node scalars (storage feature) ───────────────
    flex_data.p_state_start = p_state_start_from_source(source)
    flex_data.p_state_end = p_state_end_from_source(source)
    _storage_behaviour_notices(source, flex_data)

    # ─── Δ.4 second wave — process scalars (online / UC feature) ────────
    # ``p_min_load`` is produced authoritatively as a shape-correct
    # (p, d, t) Param by ``_derived.apply_derived_b`` (via
    # ``p_min_load_pdt_from_source``), so a period/time Map isn't summed in
    # the minFlow_minload constraint.  The synthetic / rolling sub-solve
    # early-return in ``flex_data.py`` re-wires it (passes 3-10 are skipped
    # there).  See ``specs/fix_map_reading.md`` §5.  It is the floor's field
    # only (no unit authoring min_load leaves it None; the online feature does
    # not need it).  NB: ``p_min_load`` is a
    # per-unit floor — the ``minFlow_minload`` constraint (``model/core.py``) sums
    # flow over all of a unit's outputs — and the (p, d, t) shape lets that
    # per-unit floor vary over time; the two compose.
    # Per-output ``capacity_min_coeff`` — the optional per-output floor of
    # multi-output units (minFlow_output_floor).  Assigned in pass 1a so it is
    # present on the synthetic / rolling
    # sub-solve path too (passes 3-10 are skipped there).
    flex_data.p_process_sink_min_capacity_coef = (
        p_process_sink_min_capacity_coef_from_source(source))
    # Per-edge capacity factors (output capacity_max_coeff, input
    # input_share_max) — scale the ramp limits.  Pass 1a for the same
    # reason as the min coefficient.
    flex_data.p_process_sink_max_capacity_coef = (
        p_process_sink_max_capacity_coef_from_source(source))
    flex_data.p_process_source_input_share_max = (
        p_process_source_input_share_max_from_source(source))
    # input_share_min — the minInputShare mixing minimum.  Pass 1a so the
    # rolling / synthetic sub-solve path carries it too.
    flex_data.p_process_source_input_share_min = (
        p_process_source_input_share_min_from_source(source))

    # ─── Δ.4 second wave — connection scalars (DC power flow feature) ───
    # Δ.16 — preserve the CSV-loaded value when the source has no rows.
    # Some fixtures (e.g. ``work_dc_power_flow`` / ``case14.sqlite``)
    # ship a pre-computed CSV but don't carry the parameter on the
    # ``connection`` class in the DB.
    _pcs_src = p_connection_susceptance_from_source(source)
    if _pcs_src is not None:
        flex_data.p_connection_susceptance = _pcs_src

    # ─── Δ.4 second wave — commodity scalars (price ladder feature) ─────
    flex_data.p_commodity_unitsize = p_commodity_unitsize_from_source(source)


def _with_label_steps(dt: "pl.DataFrame | None", label_steps: "pl.DataFrame | None") -> "pl.DataFrame | None":
    """``dt`` plus the ``(d, t)`` rows of the period labels (cast to ``dt``'s dtypes); ``dt`` without labels."""
    if dt is None or label_steps is None or label_steps.height == 0:
        return dt
    base = dt.select("d", "t")
    extra = label_steps.select(pl.col("d").cast(pl.Utf8).cast(base.schema["d"], strict=False),
                               pl.col("t").cast(pl.Utf8).cast(base.schema["t"], strict=False))
    return pl.concat([base, extra]).unique(maintain_order=True)


def _invest_period_need(flex_data: object, dt: "pl.DataFrame | None", dt_invest: "pl.DataFrame | None",
                        active_solve: str | None) -> PeriodNeed:
    """The investment periods of the solve (``flex_data.period_invest``, with the branch periods) that the
    period filter of the investment limits (``dt_invest``) keeps; empty without investment periods."""
    period_invest = getattr(flex_data, "period_invest", None)
    invest_periods = _periods_of(period_invest) if period_invest is not None else set()
    return PeriodNeed(frozenset(invest_periods & _periods_of(dt_invest)), "invest", active_solve)


def require_invest_period_coverage(source: "InputSource", flex_data: object, active_solve: str | None) -> None:
    """The period check of ``invest_max_period`` / ``invest_min_period`` (unit, node, connection and group) of
    a model with a construction lead time, once the solve's lead-time sets are built.

        online[e] = { d ∈ dt_invest.d : (e, o, d) ∈ edd_invest_commission }        (e an entity)
        online[g] = ∪_{e : (g, e) ∈ group_entity} online[e]                     (g a group)

    A per-period investment limit applies in the period ``d`` in which an investment starts to operate
    (``model/_lead_time.period_limit``, ``_cumulative_invest._group_period_limit_online``: the rows are the
    ``d`` of ``edd_invest_commission``, ``(e, o, o)`` for a decision without a lead time), so a period map must
    cover ``online`` and not the decision periods.  Without lead-time rows in the solve (no decision of it has
    a lead time) the rows are the investment periods, as checked in pass 1b of a model without a lead time.
    No-op for a model without a lead time (pass 1b checked it) or a solve without investment periods.
    """
    if not _construction_lag_configured(source):
        return
    dt = getattr(flex_data, "dt", None)
    dt_invest = _with_label_steps(dt, getattr(flex_data, "label_steps", None))
    need_invest = _invest_period_need(flex_data, dt, dt_invest, active_solve)
    if not need_invest:
        return
    commission = getattr(flex_data, "edd_invest_commission", None)
    if commission is None or commission.height == 0:
        need_e = need_g = need_invest
    else:
        keep = _periods_of(dt_invest)
        online: dict[str, set[str]] = {}
        for e, d in commission.select(pl.col("e").cast(pl.Utf8), pl.col("d").cast(pl.Utf8)).unique().iter_rows():
            if d in keep:
                online.setdefault(e, set()).add(d)
        groups: dict[str, set[str]] = {}
        members = group_entity(source)
        if members.height > 0:
            for g, e in members.select(pl.col("g").cast(pl.Utf8), pl.col("e").cast(pl.Utf8)).iter_rows():
                if e in online:
                    groups.setdefault(g, set()).update(online[e])
        need_e = PeriodNeed({e: frozenset(v) for e, v in online.items()}, "online", active_solve)
        need_g = PeriodNeed({g: frozenset(v) for g, v in groups.items()}, "online", active_solve)
    ed_invest_max_period_from_source(source, period_filter=dt_invest, need_periods=need_e)
    ed_invest_min_period_from_source(source, period_filter=dt_invest, need_periods=need_e)
    p_group_invest_max_period_from_source(source, period_filter=dt_invest, need_periods=need_g)
    p_group_invest_min_period_from_source(source, period_filter=dt_invest, need_periods=need_g)


def apply_direct_params_b(source: "InputSource",
                            flex_data: object,
                            *,
                            active_solve: str | None = None) -> None:
    """Pass 1b of Direct Params — the dt-dependent assignments.

    Δ.28 — runs AFTER :func:`apply_derived_a` populates
    ``flex_data.dt`` for the active solve.  Every helper here is a
    broadcast-needing Direct Param (scalar / 1d_map[period] /
    1d_map[time] values authored on the source need ``dt`` to fan out
    across the per-(d, t) axis).

    Helpers in this pass either:
        * consume ``period_filter=dt`` directly (scalar broadcast inside
          ``broadcast_to_period_time`` / ``broadcast_to_period`` /
          ``_entity_period_scalar`` / ``_entity_period_time_param``), or
        * pass ``dt`` to ``_filter_param_by_periods`` to restrict
          authored Map(period→…) values to the active periods.

    On the fast path ``apply_direct_params_a`` runs first (with empty
    ``dt``), then ``apply_derived_a`` populates ``dt`` from the source's
    timeset / period structure, then this pass runs — every broadcast
    helper sees a non-empty ``dt`` and produces the per-(d, t) frame.

    On the slow path ``flex_data.dt`` is already populated by ``_load_*``
    before ``_apply_db_overrides`` runs, so pass-A then pass-B is
    behaviourally equivalent to the legacy single pass.
    """
    # ─── Δ.4b — period filter (mirrors flextool's per-solve preprocessing) ─
    # Spine Map(period→…) parameters cover ALL declared periods, but the
    # CSV path's pd_group.csv / pdtNode.csv etc. is pre-filtered to the
    # active solve's periods via flextool's preprocessing.  We mirror by
    # restricting period-keyed Params to the active dt's periods.
    dt = getattr(flex_data, "dt", None)
    # Period-level investment limits also hold at a period label (the original name of a period that opens a
    # stochastic fan): it has no dispatch, so it is not in dt, but it carries the here-and-now investment.  Their
    # period filter is dt plus the steps the labels had before the expansion (the same periods and steps as the
    # forecast branches of the fan, so the period / timestep vocabulary of the shape resolver is unchanged).
    dt_invest = _with_label_steps(dt, getattr(flex_data, "label_steps", None))
    # Periods a period-map limit must cover (``require_period_coverage``; ``active_solve`` names the solve in
    # the error): the operational limits every period of dt (look-ahead periods included); the capacity margin
    # and the investment limits the solve's investment periods (period_invest, which already has the branch
    # periods) that their period filter keeps.  No check of the investment families when the solve has no
    # investment periods.  The per-period investment limits (invest_max/min_period) apply in the period an
    # investment starts to operate; with a construction lead time that is known only once the lead-time sets
    # are built, so a model with a lead time checks them there (:func:`require_invest_period_coverage`).
    need_dispatch = PeriodNeed(frozenset(_periods_of(dt)), "solve", active_solve)
    need_invest = _invest_period_need(flex_data, dt, dt_invest, active_solve)
    need_margin = PeriodNeed(need_invest.periods & need_dispatch.periods, "invest", active_solve)
    need_online = None if _construction_lag_configured(source) else need_invest

    # ─── Δ.4b — group capacity_margin (period) and the inertia / nonSync limits (period + time) ─
    # Δ.12c-fix gap #3: pass ``dt`` as ``period_filter`` so the values restrict to the active solve's
    # periods (capacity_margin: (g, d)); the inertia / non-synchronous limits keep their authored dims
    # ⊆ (g, d, t) (broadcast_to_period_time).
    for fn, field, need in (
        (pdGroup_capacity_margin_from_source, "pdGroup_capacity_margin", need_margin),
        (pdGroup_capacity_margin_share_from_source, "pdGroup_capacity_margin_share", need_margin),
        (pdtGroup_inertia_limit_from_source, "pdtGroup_inertia_limit", need_dispatch),
        (pdtGroup_non_synchronous_limit_from_source,
            "pdtGroup_non_synchronous_limit", need_dispatch),
    ):
        setattr(flex_data, field,
                _filter_param_by_periods(fn(source, period_filter=dt, need_periods=need), dt))
    check_energy_margin_adder_shape(source, dt)
    for fn, field in (
        (pdGroup_penalty_capacity_margin_from_source,
            "pdGroup_penalty_capacity_margin"),
        (pdGroup_penalty_inertia_from_source, "pdGroup_penalty_inertia"),
        (pdGroup_penalty_non_synchronous_from_source,
            "pdGroup_penalty_non_synchronous"),
    ):
        setattr(flex_data, field,
                _filter_param_by_periods(fn(source, period_filter=dt), dt))

    # ─── Δ.4b — group 1d_map(period) (invest/divest/cumulative) ──────────
    # Δ.17c-Tier1 — helpers now accept ``period_filter`` (consumed by
    # ``resolve_param_shape`` for silent-default disambiguation and by
    # ``broadcast_to_period`` for the active-solve period restriction).
    for fn, field, need in (
        (p_group_invest_max_period_from_source, "p_group_invest_max_period", need_online),
        (p_group_invest_min_period_from_source, "p_group_invest_min_period", need_online),
        (p_group_retire_max_period_from_source, "p_group_retire_max_period", need_invest),
        (p_group_retire_min_period_from_source, "p_group_retire_min_period", need_invest),
        (pd_group_cumulative_max_capacity_from_source, "pd_group_cumulative_max_capacity", need_invest),
        (pd_group_cumulative_min_capacity_from_source, "pd_group_cumulative_min_capacity", need_invest),
    ):
        setattr(flex_data, field,
                _filter_param_by_periods(fn(source, period_filter=dt_invest, need_periods=need), dt_invest))
    # Cumulative flow limits are operational: dispatch periods only.
    for fn, field in (
        (pd_max_cumulative_flow_from_source, "pd_max_cumulative_flow"),
        (pd_min_cumulative_flow_from_source, "pd_min_cumulative_flow"),
    ):
        setattr(flex_data, field,
                _filter_param_by_periods(fn(source, period_filter=dt, need_periods=need_dispatch), dt))

    # User-constraint constant: scalar, or a period map over the dispatch periods (rows are (cn, d, t)).
    flex_data.p_constraint_constant = p_constraint_constant_from_source(source, period_filter=dt,
                                                                        active_solve=active_solve)

    # ─── Δ.4b — group scalar (no period) ─────────────────────────────────
    for fn, field in (
        (p_group_invest_max_total_from_source, "p_group_invest_max_total"),
        (p_group_invest_min_total_from_source, "p_group_invest_min_total"),
        (p_group_retire_max_total_from_source, "p_group_retire_max_total"),
        (p_group_retire_min_total_from_source, "p_group_retire_min_total"),
        (p_group_max_cumulative_flow_from_source, "p_group_max_cumulative_flow"),
        (p_group_min_cumulative_flow_from_source, "p_group_min_cumulative_flow"),
    ):
        setattr(flex_data, field, fn(source))

    # ─── Δ.4b — group Map(period→time) instant flow caps ─────────────────
    # Δ.12c-fix gap #1: helpers now broadcast scalar / 1d_map(time) /
    # 1d_map(period) shapes across the active solve's (d, t) axis when
    # ``period_filter=dt`` is supplied.  Conditional assignment retained
    # for fixtures with no Spine rows at all (helper returns None) where
    # the seed-side preprocessing produces an empty/zero-broadcast Param.
    v = _filter_param_by_periods(
        pdt_max_instant_flow_from_source(source, period_filter=dt, need_periods=need_dispatch), dt)
    if v is not None:
        flex_data.pdt_max_instant_flow = v
    v = _filter_param_by_periods(
        pdt_min_instant_flow_from_source(source, period_filter=dt, need_periods=need_dispatch), dt)
    if v is not None:
        flex_data.pdt_min_instant_flow = v

    # ─── Δ.4b — multi-class union 1d_map(period): invest / divest ────────
    # Δ.17c-Tier2 — helpers now accept ``period_filter`` (consumed by
    # ``resolve_param_shape`` for silent-default disambiguation and by
    # ``broadcast_to_period`` for the active-solve period restriction and
    # scalar-to-(e,d) cross-join).
    for fn, field, need in (
        (ed_invest_max_period_from_source, "ed_invest_max_period", need_online),
        (ed_divest_max_period_from_source, "ed_divest_max_period", need_invest),
        (ed_invest_min_period_from_source, "ed_invest_min_period", need_online),
        (ed_divest_min_period_from_source, "ed_divest_min_period", need_invest),
        (ed_cumulative_max_capacity_from_source, "ed_cumulative_max_capacity", need_invest),
        (ed_cumulative_min_capacity_from_source, "ed_cumulative_min_capacity", need_invest),
    ):
        setattr(flex_data, field,
                _filter_param_by_periods(fn(source, period_filter=dt_invest, need_periods=need), dt_invest))

    # ─── Δ.4b — relationship scalars (ramp + inertia, sink/source) ───────
    for fn, field in (
        (p_ramp_speed_up_sink_from_source, "p_ramp_speed_up_sink"),
        (p_ramp_speed_down_sink_from_source, "p_ramp_speed_down_sink"),
        (p_ramp_speed_up_source_from_source, "p_ramp_speed_up_source"),
        (p_ramp_speed_down_source_from_source, "p_ramp_speed_down_source"),
        # The ramp cost per arc (Σ of its sides with ramp_method ramp_cost / both), read by ramp_change_eq.
        (p_ramp_cost_arc_from_source, "p_ramp_cost_arc"),
        (p_process_sink_inertia_constant_from_source,
            "p_process_sink_inertia_constant"),
        (p_process_source_inertia_constant_from_source,
            "p_process_source_inertia_constant"),
    ):
        setattr(flex_data, field, fn(source))

    # ─── Δ.4b — UC: startup_cost (1d_map period) ─────────────────────────
    # Δ.12c-fix gap #3: ``_entity_period_scalar`` handles 1d_map(period)
    # AND scalar shapes when ``period_filter=dt`` is supplied.
    v = _filter_param_by_periods(
        p_startup_cost_from_source(source, period_filter=dt), dt)
    if v is not None:
        flex_data.p_startup_cost = v

    # ─── Δ.4b — storage Map(period→time) ─────────────────────────────────
    # Δ.12c-fix gap #1: helpers broadcast non-Map shapes via period_filter.
    v = _filter_param_by_periods(
        p_node_availability_from_source(source, period_filter=dt), dt)
    if v is not None:
        flex_data.p_node_availability = v
    v = _filter_param_by_periods(
        p_storage_state_reference_value_from_source(source, period_filter=dt),
        dt)
    if v is not None:
        flex_data.p_storage_state_reference_value = v
    # Self-discharge per (n, d, t) (scalar or period / time map); None when no node authors one.
    flex_data.p_state_self_discharge = _filter_param_by_periods(
        p_state_self_discharge_from_source(source, period_filter=dt), dt)

    # ─── Δ.4b — CO2 (price + cap) ────────────────────────────────────────
    v = _filter_param_by_periods(
        p_co2_price_from_source(source, period_filter=dt), dt)
    if v is not None:
        flex_data.p_co2_price = v
    v = _filter_param_by_periods(
        p_co2_max_period_from_source(source, period_filter=dt, need_periods=need_dispatch), dt)
    if v is not None:
        flex_data.p_co2_max_period = v
    # co2_max_total has no period axis (read from the emitted pdGroup slice); a total method without a value
    # would leave the group without its cap.
    require_co2_cap_values(source, "co2_max_total", _CO2_MAX_TOTAL_METHOD)

    # ─── Δ.4b — variable cost (Map period→time) ──────────────────────────
    # Δ.17c-Tier3 — the three per-side helpers below route through the resolver
    # cascade (resolve_param_shape + broadcast_to_period_time).  Each
    # accepts the active solve's (d, t) frame as ``period_filter`` so
    # SCALAR / MAP_PERIOD / MAP_TIME shapes broadcast up to (p, d, t).
    v = _filter_param_by_periods(
        p_pdt_varCost_source_from_source(source, period_filter=dt), dt)
    if v is not None:
        flex_data.p_pdt_varCost_source = v
    v = _filter_param_by_periods(
        p_pdt_varCost_sink_from_source(source, period_filter=dt), dt)
    if v is not None:
        flex_data.p_pdt_varCost_sink = v
    v = _filter_param_by_periods(
        p_pdt_varCost_process_from_source(source, period_filter=dt), dt)
    if v is not None:
        flex_data.p_pdt_varCost_process = v
    # The four index sets and p_pssdt_varCost the objective's variable-cost terms join on (#74).
    apply_varcost_params(flex_data)

    # ─── Δ.4b — reserves ─────────────────────────────────────────────────
    # Δ.17c-Tier3 — pdtReserve_upDown_group_reservation routes through
    # the resolver cascade with a per-source-column rename dict (3-dim
    # relationship class).  The relationship-scalar reserves below
    # (penalty / reliability / max_share / failure-ratio /
    # increase-reserve-ratio) stay direct — they're SCALAR-only.
    v = _filter_param_by_periods(
        pdtReserve_upDown_group_reservation_from_source(
            source, period_filter=dt), dt)
    if v is not None:
        flex_data.pdtReserve_upDown_group_reservation = v
    flex_data.p_reserve_upDown_group_penalty_reserve = (
        p_reserve_upDown_group_penalty_reserve_from_source(source))
    flex_data.p_reserve_upDown_group_reserve_duration = (
        p_reserve_upDown_group_reserve_duration_from_source(source))
    flex_data.p_reserve_upDown_group_increase_reserve_ratio = (
        p_reserve_upDown_group_increase_reserve_ratio_from_source(source))
    flex_data.p_process_reserve_upDown_node_reliability = (
        p_process_reserve_upDown_node_reliability_from_source(source))
    flex_data.p_process_reserve_upDown_node_max_share = (
        p_process_reserve_upDown_node_max_share_from_source(source))
    flex_data.p_process_reserve_upDown_node_large_failure_ratio_value = (
        p_process_reserve_upDown_node_large_failure_ratio_value_from_source(source))
    flex_data.p_process_reserve_upDown_node_increase_reserve_ratio_value = (
        p_process_reserve_upDown_node_increase_reserve_ratio_value_from_source(source))

    # ``process_delayed__duration`` is produced together with the delay
    # weights and ``dtt__delay_duration`` by
    # ``_derived.apply_delay_params`` (same td labels).

    # ─── Δ.4b — additional Map(period→time) on object classes ───────────
    # Δ.12c-fix gap #1: helpers broadcast non-Map shapes via period_filter.
    v = _filter_param_by_periods(
        p_process_availability_from_source(source, period_filter=dt), dt)
    if v is not None:
        flex_data.p_process_availability = v
    v = _filter_param_by_periods(
        p_commodity_price_from_source(source, period_filter=dt), dt)
    if v is not None:
        flex_data.p_commodity_price = v

    # ─── Δ.4b — commodity ladder split (price / quantity) ────────────────
    # _ladder_split returns (price, quantity) — both can be None.
    # The annual ladder is authored per period: a period label keeps its curve so that the forecast branch periods
    # inherit it (_commodity_ladder.inherit_branch_periods); the label itself buys nothing (no dispatch).
    p_ann_price, p_ann_qty = _ladder_split(source, "price_ladder_annual",
                                              period_filter=dt_invest)
    if p_ann_price is not None:
        flex_data.p_ladder_ann_price = _filter_param_by_periods(p_ann_price, dt_invest)
    if p_ann_qty is not None:
        flex_data.p_ladder_ann_quantity = _filter_param_by_periods(p_ann_qty, dt_invest)
    p_cum_price, p_cum_qty = _ladder_split(source, "price_ladder_cumulative",
                                              period_filter=None)
    if p_cum_price is not None:
        flex_data.p_ladder_cum_price = p_cum_price
    if p_cum_qty is not None:
        flex_data.p_ladder_cum_quantity = p_cum_qty


def apply_direct_params(source: "InputSource",
                          flex_data: object) -> None:
    """Apply the full Direct Param wave in two phases.

    Δ.28 — kept as a single back-compat entry point.  The dispatcher in
    :func:`flextool.engine.preprocess.flex_data._apply_db_overrides` does NOT
    call this function any more; it calls
    :func:`apply_direct_params_a` / :func:`apply_direct_params_b`
    around :func:`apply_derived_a` so the dt-dependent broadcasts in
    pass 1b see a populated ``flex_data.dt``.

    External callers that want the legacy single-pass behaviour (with
    pre-populated ``flex_data.dt``) keep this entry point.  In the slow
    path ``flex_data.dt`` is loaded by ``_load_*`` before this runs;
    in the fast path the dispatcher's two-phase ordering is the only
    correct sequence.
    """
    apply_direct_params_a(source, flex_data)
    apply_direct_params_b(source, flex_data)
