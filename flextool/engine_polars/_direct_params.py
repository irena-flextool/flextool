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

The full sweep into ``input.py`` is Γ.2/Γ.3; Γ.1 wires only the chosen
representative subset.  Each helper is here so a future helper-by-
helper migration replaces the corresponding CSV branch in ``input.py``
with a one-line call to the function below.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl

from polar_high import Param

from flextool.engine_polars._axis_enums import (cast_frame_axes,
                                                 get_global_axis_enums,
                                                 rename_to_axis)
from flextool.engine_polars._param_shapes import (
    broadcast_to_period,
    broadcast_to_period_time,
    resolve_param_shape,
)
from flextool.engine_polars._solve_state import FlexToolConfigError

if TYPE_CHECKING:
    from flextool.engine_polars._input_source import InputSource


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


def p_constraint_constant_from_source(source: "InputSource") -> Param | None:
    """``constraint.constant`` → ``Param(("cn",), [cn, value])``.

    Default ``0.0`` — broadcast to every constraint via the source.

    The constraint axis uses the column name ``cn`` (not ``c``) to avoid
    collision with the commodity axis — see
    ``schemas/flextool_axis_contract.json`` review note ``c_collision``.
    """
    df = source.parameter("constraint", "constant")
    if df.height == 0:
        return None
    return Param(("cn",),
                 df.lazy().rename({"name": "cn"}).select("cn", "value"))


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
# §5.2.1 — penalty (sentinel default ⇒ broadcast)


def penalty_up_scalar_from_source(source: "InputSource") -> pl.DataFrame:
    """``node.penalty_up`` per-node scalar (pre-broadcast over (d, t)).

    The sentinel default (large positive value, e.g. 10000.0) is
    applied at the source level (§4.6).  The downstream Derived helper
    cross-joins this against ``(d, t)`` to produce ``p_penalty_up``.
    """
    df = source.parameter("node", "penalty_up")
    return df.lazy().rename({"name": "n"}).select("n", "value").collect()


def penalty_down_scalar_from_source(source: "InputSource") -> pl.DataFrame:
    df = source.parameter("node", "penalty_down")
    return df.lazy().rename({"name": "n"}).select("n", "value").collect()


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
# §5.2.3 — relationship 1d_map / scalar (flow_coefficient, etc.)


def unit_input_flow_coefficient_from_source(source: "InputSource") -> Param | None:
    """``unit__inputNode.flow_coefficient`` → ``Param(("p","source"), …)``.

    Default 1.0 — broadcast over all (unit, node) entities.
    """
    df = source.parameter("unit__inputNode", "flow_coefficient")
    if df.height == 0:
        return None
    return Param(("p", "source"),
                 df.lazy().rename({"unit": "p", "node": "source"})
                          .select("p", "source", "value"))


def unit_output_flow_coefficient_from_source(source: "InputSource") -> Param | None:
    """``unit__outputNode.flow_coefficient`` → ``Param(("p","sink"), …)``."""
    df = source.parameter("unit__outputNode", "flow_coefficient")
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


def _node_constraint_coef(source: "InputSource", parameter_name: str) -> Param | None:
    """Return ``Param(("n", "cn"), [n, cn, value])`` for a per-(node,
    constraint) coefficient parameter.  ``None`` if the DB has no rows.

    The constraint axis column is named ``cn`` to disambiguate from the
    commodity axis column ``c`` (the contract's ``c_collision`` decision).
    """
    df = source.parameter("node", parameter_name)
    if df.height == 0:
        return None
    return Param(
        ("n", "cn"),
        df.lazy()
          .rename({"name": "n", "constraint": "cn"})
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
        parts.append(
            df.lazy()
              .rename({"name": "p", "constraint": "cn"})
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
    (mirrors ``input.py::_read_e_param`` for ``invest_min_total`` /
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
# in ``input.py`` is a one-CSV ``rename + select`` of an entity scalar /
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
def p_state_self_discharge_from_source(source: "InputSource") -> Param | None:
    """``node.self_discharge_loss`` → ``Param(("n",), [n, value])``.

    Default 0.0 (schema).  CSV path
    (``input.py::_load_storage::_node_param``) returns the explicit
    rows only.  We mirror by reading via ``parameter_explicit``.
    """
    return _entity_scalar_explicit(source, "node", "self_discharge_loss", "n")


def p_state_start_from_source(source: "InputSource") -> Param | None:
    """``node.storage_state_start`` → ``Param(("n",), [n, value])``.

    Default ``None`` (schema).  Returns explicit rows only.
    """
    return _entity_scalar_explicit(source, "node", "storage_state_start", "n")


# §5.2.7 — process scalars sliced from p_process.csv
def p_min_load_from_source(source: "InputSource") -> Param | None:
    """``unit.min_load`` → ``Param(("p",), [p, value])``.

    Default 0.0 (schema).  CSV path filters by processParam=='min_load'
    and emits rows only when explicit; we mirror via ``parameter_explicit``.
    """
    return _entity_scalar_explicit(source, "unit", "min_load", "p")


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
                *, filter_zero: bool = True) -> Param | None:
    """Group scalar Param ``Param(("g",), [g, value])``, dropping zero
    rows by default (mirrors ``_slice_pgroup``'s ``!= 0`` filter).
    """
    try:
        df = source.parameter_explicit("group", parameter_name)
    except (KeyError, AttributeError):
        try:
            df = source.parameter("group", parameter_name)
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
def pdGroup_capacity_margin_from_source(source: "InputSource",
                                         period_filter: pl.DataFrame | None = None,
                                         ) -> Param | None:
    """``group.capacity_margin`` → ``Param(("g","d"))``.

    Δ.17c-Tier1 — uses :func:`._param_shapes.resolve_param_shape`.
    Allowed shapes: scalar / 1d_map[period].  CSV path drops zero
    rows; we mirror via ``filter_zero=True``.
    """
    resolved = resolve_param_shape(
        source, "group", "capacity_margin", period_filter=period_filter)
    return broadcast_to_period(
        resolved, "g", period_filter, filter_zero=True)


def pdGroup_penalty_capacity_margin_from_source(source: "InputSource",
                                                 period_filter: pl.DataFrame | None = None,
                                                 ) -> Param | None:
    """``group.penalty_capacity_margin`` → ``Param(("g","d"))``.
    Δ.17c-Tier1 — uses ``resolve_param_shape`` + ``broadcast_to_period``.
    """
    resolved = resolve_param_shape(
        source, "group", "penalty_capacity_margin",
        period_filter=period_filter)
    return broadcast_to_period(
        resolved, "g", period_filter, filter_zero=True)


def pdGroup_inertia_limit_from_source(source: "InputSource",
                                       period_filter: pl.DataFrame | None = None,
                                       ) -> Param | None:
    """``group.inertia_limit`` → ``Param(("g","d"))``.
    Δ.17c-Tier1 — uses ``resolve_param_shape`` + ``broadcast_to_period``.
    """
    resolved = resolve_param_shape(
        source, "group", "inertia_limit", period_filter=period_filter)
    return broadcast_to_period(
        resolved, "g", period_filter, filter_zero=True)


def pdGroup_penalty_inertia_from_source(source: "InputSource",
                                         period_filter: pl.DataFrame | None = None,
                                         ) -> Param | None:
    """``group.penalty_inertia`` → ``Param(("g","d"))``.
    Δ.17c-Tier1 — uses ``resolve_param_shape`` + ``broadcast_to_period``.
    """
    resolved = resolve_param_shape(
        source, "group", "penalty_inertia", period_filter=period_filter)
    return broadcast_to_period(
        resolved, "g", period_filter, filter_zero=True)


def pdGroup_non_synchronous_limit_from_source(source: "InputSource",
                                               period_filter: pl.DataFrame | None = None,
                                               ) -> Param | None:
    """``group.non_synchronous_limit`` → ``Param(("g","d"))``.
    Δ.17c-Tier1 — uses ``resolve_param_shape`` + ``broadcast_to_period``.
    """
    resolved = resolve_param_shape(
        source, "group", "non_synchronous_limit",
        period_filter=period_filter)
    return broadcast_to_period(
        resolved, "g", period_filter, filter_zero=True)


def pdGroup_penalty_non_synchronous_from_source(source: "InputSource",
                                                 period_filter: pl.DataFrame | None = None,
                                                 ) -> Param | None:
    """``group.penalty_non_synchronous`` → ``Param(("g","d"))``.
    Δ.17c-Tier1 — uses ``resolve_param_shape`` + ``broadcast_to_period``.
    """
    resolved = resolve_param_shape(
        source, "group", "penalty_non_synchronous",
        period_filter=period_filter)
    return broadcast_to_period(
        resolved, "g", period_filter, filter_zero=True)


# §1.16 — group invest/divest 1d_map(period) — drop zeros to match CSV
def p_group_invest_max_period_from_source(source: "InputSource",
                                            period_filter: pl.DataFrame | None = None,
                                            ) -> Param | None:
    """``group.invest_max_period`` → ``Param(("g","d"))``.
    Δ.17c-Tier1 — uses ``resolve_param_shape`` + ``broadcast_to_period``.
    """
    resolved = resolve_param_shape(
        source, "group", "invest_max_period", period_filter=period_filter)
    return broadcast_to_period(
        resolved, "g", period_filter, filter_zero=True)


def p_group_invest_min_period_from_source(source: "InputSource",
                                            period_filter: pl.DataFrame | None = None,
                                            ) -> Param | None:
    """``group.invest_min_period`` → ``Param(("g","d"))``.
    Δ.17c-Tier1 — uses ``resolve_param_shape`` + ``broadcast_to_period``.
    """
    resolved = resolve_param_shape(
        source, "group", "invest_min_period", period_filter=period_filter)
    return broadcast_to_period(
        resolved, "g", period_filter, filter_zero=True)


def p_group_retire_max_period_from_source(source: "InputSource",
                                            period_filter: pl.DataFrame | None = None,
                                            ) -> Param | None:
    """``group.retire_max_period`` → ``Param(("g","d"))``.
    Δ.17c-Tier1 — uses ``resolve_param_shape`` + ``broadcast_to_period``.
    """
    resolved = resolve_param_shape(
        source, "group", "retire_max_period", period_filter=period_filter)
    return broadcast_to_period(
        resolved, "g", period_filter, filter_zero=True)


def p_group_retire_min_period_from_source(source: "InputSource",
                                            period_filter: pl.DataFrame | None = None,
                                            ) -> Param | None:
    """``group.retire_min_period`` → ``Param(("g","d"))``.
    Δ.17c-Tier1 — uses ``resolve_param_shape`` + ``broadcast_to_period``.
    """
    resolved = resolve_param_shape(
        source, "group", "retire_min_period", period_filter=period_filter)
    return broadcast_to_period(
        resolved, "g", period_filter, filter_zero=True)


def pd_max_cumulative_flow_from_source(source: "InputSource",
                                         period_filter: pl.DataFrame | None = None,
                                         ) -> Param | None:
    """``group.max_cumulative_flow`` → ``Param(("g","d"))``.
    Δ.17c-Tier1 — uses ``resolve_param_shape`` + ``broadcast_to_period``.
    """
    resolved = resolve_param_shape(
        source, "group", "max_cumulative_flow",
        period_filter=period_filter)
    return broadcast_to_period(
        resolved, "g", period_filter, filter_zero=True)


def pd_min_cumulative_flow_from_source(source: "InputSource",
                                         period_filter: pl.DataFrame | None = None,
                                         ) -> Param | None:
    """``group.min_cumulative_flow`` → ``Param(("g","d"))``.
    Δ.17c-Tier1 — uses ``resolve_param_shape`` + ``broadcast_to_period``.
    """
    resolved = resolve_param_shape(
        source, "group", "min_cumulative_flow",
        period_filter=period_filter)
    return broadcast_to_period(
        resolved, "g", period_filter, filter_zero=True)


# §1.16 — group scalar Direct Params (no period dimension)
def p_group_invest_max_total_from_source(source: "InputSource") -> Param | None:
    return _g_scalar(source, "invest_max_total")


def p_group_invest_min_total_from_source(source: "InputSource") -> Param | None:
    return _g_scalar(source, "invest_min_total")


def p_group_retire_max_total_from_source(source: "InputSource") -> Param | None:
    return _g_scalar(source, "retire_max_total")


def p_group_retire_min_total_from_source(source: "InputSource") -> Param | None:
    return _g_scalar(source, "retire_min_total")


def p_group_invest_max_cumulative_from_source(source: "InputSource") -> Param | None:
    return _g_scalar(source, "invest_max_cumulative")


def p_group_invest_min_cumulative_from_source(source: "InputSource") -> Param | None:
    return _g_scalar(source, "invest_min_cumulative")


def p_group_max_cumulative_flow_from_source(source: "InputSource") -> Param | None:
    return _g_scalar(source, "max_cumulative_flow")


def p_group_min_cumulative_flow_from_source(source: "InputSource") -> Param | None:
    return _g_scalar(source, "min_cumulative_flow")


# §1.16 — pdtGroup (Map period→time) instant flow caps
def pdt_max_instant_flow_from_source(source: "InputSource",
                                      period_filter: pl.DataFrame | None = None,
                                      ) -> Param | None:
    """``group.max_instant_flow`` → ``Param(("g", "d", "t"))``.

    Δ.17c — uses :func:`._param_shapes.resolve_param_shape`.  Allowed
    shapes: scalar / 1d_map[period] / 1d_map[time] / 2d_map[period,time].
    """
    resolved = resolve_param_shape(
        source, "group", "max_instant_flow", period_filter=period_filter)
    return broadcast_to_period_time(
        resolved, "g", period_filter, filter_zero=True)


def pdt_min_instant_flow_from_source(source: "InputSource",
                                      period_filter: pl.DataFrame | None = None,
                                      ) -> Param | None:
    """``group.min_instant_flow`` → ``Param(("g", "d", "t"))``.

    Δ.17c — uses :func:`._param_shapes.resolve_param_shape`.  Allowed
    shapes: scalar / 1d_map[period] / 1d_map[time] / 2d_map[period,time].
    """
    resolved = resolve_param_shape(
        source, "group", "min_instant_flow", period_filter=period_filter)
    return broadcast_to_period_time(
        resolved, "g", period_filter, filter_zero=True)


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
                            period_filter: pl.DataFrame | None = None
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
    """
    parts: list[pl.LazyFrame] = []
    for cls in classes:
        resolved = resolve_param_shape(
            source, cls, parameter_name, period_filter=period_filter)
        part = broadcast_to_period(
            resolved, "e", period_filter, filter_zero=filter_zero)
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
                                       ) -> Param | None:
    """``unit/node/connection.invest_max_period`` 1d_map(period) →
    ``Param(("e", "d"))``.  CSV path keeps zero rows (per
    ``_read_period_cap``); we preserve that.

    Δ.17c-Tier2 — uses ``resolve_param_shape`` + ``broadcast_to_period``
    per class via :func:`_e_period_param_union`.
    """
    return _e_period_param_union(source, "invest_max_period",
                                    filter_zero=False,
                                    period_filter=period_filter)


def ed_divest_max_period_from_source(source: "InputSource",
                                       period_filter: pl.DataFrame | None = None,
                                       ) -> Param | None:
    """Δ.17c-Tier2 — see :func:`ed_invest_max_period_from_source`."""
    return _e_period_param_union(source, "retire_max_period",
                                    filter_zero=False,
                                    period_filter=period_filter)


def ed_invest_min_period_from_source(source: "InputSource",
                                       period_filter: pl.DataFrame | None = None,
                                       ) -> Param | None:
    """CSV path drops zero rows (``_read_e_d_param``).
    Δ.17c-Tier2 — see :func:`ed_invest_max_period_from_source`.
    """
    return _e_period_param_union(source, "invest_min_period",
                                    filter_zero=True,
                                    period_filter=period_filter)


def ed_divest_min_period_from_source(source: "InputSource",
                                       period_filter: pl.DataFrame | None = None,
                                       ) -> Param | None:
    """Δ.17c-Tier2 — see :func:`ed_invest_max_period_from_source`."""
    return _e_period_param_union(source, "retire_min_period",
                                    filter_zero=True,
                                    period_filter=period_filter)


def ed_cumulative_max_capacity_from_source(source: "InputSource",
                                             period_filter: pl.DataFrame | None = None,
                                             ) -> Param | None:
    """Δ.17c-Tier2 — see :func:`ed_invest_max_period_from_source`."""
    return _e_period_param_union(source, "cumulative_max_capacity",
                                    filter_zero=True,
                                    period_filter=period_filter)


def ed_cumulative_min_capacity_from_source(source: "InputSource",
                                             period_filter: pl.DataFrame | None = None,
                                             ) -> Param | None:
    """Δ.17c-Tier2 — see :func:`ed_invest_max_period_from_source`."""
    return _e_period_param_union(source, "cumulative_min_capacity",
                                    filter_zero=True,
                                    period_filter=period_filter)


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
    """``unit.startup_cost`` → ``Param(("p", "d"))``.

    Δ.17c-Tier1 — uses :func:`._param_shapes.resolve_param_shape`.
    Allowed shapes: scalar / 1d_map[period].  CSV path filters zero rows
    (``_load_online``: ``filter(value != 0)``); we mirror via
    ``filter_zero=True``.
    """
    resolved = resolve_param_shape(
        source, "unit", "startup_cost", period_filter=period_filter)
    return broadcast_to_period(
        resolved, "p", period_filter, filter_zero=True)


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
                                  ) -> Param | None:
    """``group.co2_max_period`` → ``Param(("g", "d"))``.

    Δ.17c — uses :func:`._param_shapes.resolve_param_shape`.  Allowed
    shapes: scalar / 1d_map[period].  CSV path keys this off
    ``inp/pd_group.csv`` slice.

    Gate-coupled note: at the consumer site this Param is only emitted
    into the LP when the (g, c, n) join with ``commodity_node_co2`` ×
    ``group__node`` is non-empty (see ``_load_co2_cap``).  We return the
    Param unconditionally — the consumer decides whether to wire it.
    """
    resolved = resolve_param_shape(
        source, "group", "co2_max_period", period_filter=period_filter)
    return broadcast_to_period(resolved, "g", period_filter)


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
    """``unit/connection.other_operational_cost`` Map(period→time) →
    ``Param(("p", "d", "t"))``.  Union across the two object classes.

    Δ.17c-Tier3 — each per-class call routes through
    ``resolve_param_shape`` + ``broadcast_to_period_time``.  Pre-
    migration the helper gated on ``{"name", "period", "t"}.issubset``,
    silently dropping silent-default-Map authoring (column ``"x"`` not
    ``"period"``) and SCALAR / MAP_PERIOD / MAP_TIME shapes.  Mirrors
    the Phase-2 ``_e_period_param_union`` multi-class concat template.

    SCALAR / MAP_PERIOD / MAP_TIME branches return Params with fewer
    dims than the canonical ``(p, d, t)``.  Per-class results are
    broadcast up to ``(p, d, t)`` against ``period_filter``'s
    ``(d, t)`` axis before concat so the union frame is dim-uniform.
    Callers (the dispatcher) always pass a non-empty ``period_filter``;
    if a per-class Param is a strict subset of ``(p, d, t)`` AND no
    period_filter is on hand, we raise rather than concat
    incompatible-shape frames.
    """
    parts: list[pl.LazyFrame] = []
    for cls in ("unit", "connection"):
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


# ---------------------------------------------------------------------------
# §1.15 — Reserves

def pdtReserve_upDown_group_reservation_from_source(
    source: "InputSource",
    period_filter: pl.DataFrame | None = None,
) -> Param | None:
    """``reserve__upDown__group.reservation`` Map(period→time) →
    ``Param(("r", "ud", "g", "d", "t"))``.  Default 0.0 (schema) — CSV
    path emits explicit rows only (zero values are NOT filtered).

    Δ.17c-Tier3 — routes through ``resolve_param_shape`` +
    ``broadcast_to_period_time`` with a per-source-column rename dict
    mapping the 3-dim relationship class's ``(reserve, upDown, group)``
    columns to ``(r, ud, g)``.  ``filter_zero=False`` because the
    schema default of 0.0 makes explicit zero rows authoring-
    meaningful (legacy CSV path preserved zero rows for this param).
    """
    resolved = resolve_param_shape(
        source, "reserve__upDown__group", "reservation",
        period_filter=period_filter)
    return broadcast_to_period_time(
        resolved,
        {"reserve": "r", "upDown": "ud", "group": "g"},
        period_filter, filter_zero=False)


def p_reserve_upDown_group_penalty_reserve_from_source(source: "InputSource") -> Param | None:
    """``reserve__upDown__group.penalty_reserve`` scalar →
    ``Param(("r", "ud", "g"))``.  None default — explicit rows only.
    """
    try:
        df = source.parameter_explicit("reserve__upDown__group", "penalty_reserve")
    except (KeyError, AttributeError):
        try:
            df = source.parameter("reserve__upDown__group", "penalty_reserve")
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
# §1.17 — Delayed processes (process_delayed__duration)

def process_delayed__duration_from_source(source: "InputSource") -> pl.DataFrame | None:
    """``unit/connection.delay`` 1d_map(td) → ``[p, td]`` set frame.

    CSV path (``_delay.load_data``) reads ``solve_data/process_delayed__duration.csv``
    and returns the (p, td) keys as a DataFrame (not a Param — it's a
    set, since the duration values are 1.0 or absent).

    Note: this is a *set* in FlexData, not a Param — return
    type matches the field declared on FlexData.
    """
    parts: list[pl.LazyFrame] = []
    for cls in ("unit", "connection"):
        try:
            df = source.parameter_explicit(cls, "delay")
        except (KeyError, AttributeError):
            try:
                df = source.parameter(cls, "delay")
            except KeyError:
                continue
        if df is None or df.height == 0:
            continue
        cols = df.columns
        if "name" not in cols:
            continue
        # 1d_map(td) — the index column may be 'td' or default 'period'/'i'.
        # Detect by elimination: any column not in {name, value} is the index.
        idx_cols = [c for c in cols if c not in ("name", "value")]
        if len(idx_cols) != 1:
            continue
        idx = idx_cols[0]
        lf = (df.lazy()
                .rename({"name": "p", idx: "td"})
                .filter(pl.col("value").is_not_null())
                .filter(pl.col("value") != 0.0))
        parts.append(lf.select("p", "td"))
    if not parts:
        return None
    out = pl.concat(parts).unique().collect()
    if out.height == 0:
        return None
    return out.sort("p", "td")


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
# §1.19 — Commodity ladder split (price / quantity sub-frames).
#
# Spine schema: ``commodity.price_ladder_annual`` is a 3-level Map
# ``(period → tier → {"price","quantity"} → f64)`` which the
# SpineDbReader unrolls to ``[name, period, tier, x, value]`` where
# ``x ∈ {"price", "quantity"}``.  We split it into two Params keyed
# (c, i, d) with ``value`` being the price (resp. quantity) only.
# ``commodity.price_ladder_cumulative`` mirrors but without period.

def _ladder_split(source: "InputSource", parameter_name: str,
                    *, with_period: bool) -> tuple[Param | None, Param | None]:
    """Return ``(price_param, quantity_param)`` from the named ladder
    parameter.  ``with_period=True`` means the unroll produces a
    ``period`` column (annual ladder); ``False`` means tier-only
    (cumulative ladder).
    """
    try:
        df = source.parameter_explicit("commodity", parameter_name)
    except (KeyError, AttributeError):
        try:
            df = source.parameter("commodity", parameter_name)
        except KeyError:
            return None, None
    if df is None or df.height == 0:
        return None, None
    cols = df.columns
    needed = {"name", "tier", "value"}
    if with_period:
        needed.add("period")
    if not needed.issubset(cols):
        return None, None
    # Detect the leaf-discriminator column ("x" or whatever name carries
    # the "price"/"quantity" tag).  All non-(name, period, tier, value)
    # columns are candidates; pick the first.
    keep_cols = ({"name", "period", "tier", "value"} if with_period
                  else {"name", "tier", "value"})
    leaf_cols = [c for c in cols if c not in keep_cols]
    if len(leaf_cols) != 1:
        return None, None
    leaf = leaf_cols[0]
    rename = {"name": "c", "tier": "i"}
    if with_period:
        rename["period"] = "d"
    base = (df.lazy()
              .rename(rename)
              .filter(pl.col("value").is_not_null())
              .with_columns(pl.col("i").cast(pl.Utf8)))
    out_dims = ("c", "i", "d") if with_period else ("c", "i")
    price_lf = base.filter(pl.col(leaf) == "price").select(*out_dims, "value")
    qty_lf   = base.filter(pl.col(leaf) == "quantity").select(*out_dims, "value")
    price_df = price_lf.collect()
    qty_df = qty_lf.collect()
    p_price = Param(out_dims, price_df) if price_df.height > 0 else None
    p_qty = Param(out_dims, qty_df) if qty_df.height > 0 else None
    return p_price, p_qty


def p_ladder_ann_price_from_source(source: "InputSource") -> Param | None:
    return _ladder_split(source, "price_ladder_annual", with_period=True)[0]


def p_ladder_ann_quantity_from_source(source: "InputSource") -> Param | None:
    return _ladder_split(source, "price_ladder_annual", with_period=True)[1]


def p_ladder_cum_price_from_source(source: "InputSource") -> Param | None:
    return _ladder_split(source, "price_ladder_cumulative", with_period=False)[0]


def p_ladder_cum_quantity_from_source(source: "InputSource") -> Param | None:
    return _ladder_split(source, "price_ladder_cumulative", with_period=False)[1]


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
    flex_data.p_constraint_constant = p_constraint_constant_from_source(source)

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

    # ─── §5.2.1 invest/divest total caps (entity-unioned) ───────────────
    # Max variants: keyed on entityInvest (resp. entityDivest), one row
    # per entity with value defaulting to 0 — mirrors
    # ``entity_total_caps.py:_compute_entity_total`` and
    # ``input.py::_e_total_param``.
    flex_data.e_invest_max_total = _e_total_param(
        source, "invest_max_total", kind="invest")
    flex_data.e_divest_max_total = _e_total_param(
        source, "retire_max_total", kind="divest")
    # Min variants: CSV path filters out zero rows (input.py::_read_e_param)
    # — None when no entity has an explicit non-zero min cap.
    flex_data.e_invest_min_total = _e_total_param(
        source, "invest_min_total", kind="invest", filter_zero=True)
    flex_data.e_divest_min_total = _e_total_param(
        source, "retire_min_total", kind="divest", filter_zero=True)

    # ─── Δ.4 second wave — node scalars (storage feature) ───────────────
    flex_data.p_state_self_discharge = p_state_self_discharge_from_source(source)
    flex_data.p_state_start = p_state_start_from_source(source)

    # ─── Δ.4 second wave — process scalars (online / UC feature) ────────
    flex_data.p_min_load = p_min_load_from_source(source)

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


def apply_direct_params_b(source: "InputSource",
                            flex_data: object) -> None:
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

    # ─── Δ.4b — group 1d_map(period) (capacity_margin / inertia / nonSync) ─
    # Δ.12b: unconditional — _entity_period_scalar handles 1d_map(period)
    # natively (this is the only Spine shape for these parameters).
    # Δ.12c-fix gap #3: pass ``dt`` as ``period_filter`` so scalar values
    # (single ``[name, value]`` row) broadcast across the active solve's
    # periods, matching flextool's pdGroup.csv preprocessing.
    for fn, field in (
        (pdGroup_capacity_margin_from_source, "pdGroup_capacity_margin"),
        (pdGroup_penalty_capacity_margin_from_source,
            "pdGroup_penalty_capacity_margin"),
        (pdGroup_inertia_limit_from_source, "pdGroup_inertia_limit"),
        (pdGroup_penalty_inertia_from_source, "pdGroup_penalty_inertia"),
        (pdGroup_non_synchronous_limit_from_source,
            "pdGroup_non_synchronous_limit"),
        (pdGroup_penalty_non_synchronous_from_source,
            "pdGroup_penalty_non_synchronous"),
    ):
        setattr(flex_data, field,
                _filter_param_by_periods(fn(source, period_filter=dt), dt))

    # ─── Δ.4b — group 1d_map(period) (invest/divest/cumulative) ──────────
    # Δ.17c-Tier1 — helpers now accept ``period_filter`` (consumed by
    # ``resolve_param_shape`` for silent-default disambiguation and by
    # ``broadcast_to_period`` for the active-solve period restriction).
    for fn, field in (
        (p_group_invest_max_period_from_source, "p_group_invest_max_period"),
        (p_group_invest_min_period_from_source, "p_group_invest_min_period"),
        (p_group_retire_max_period_from_source, "p_group_retire_max_period"),
        (p_group_retire_min_period_from_source, "p_group_retire_min_period"),
        (pd_max_cumulative_flow_from_source, "pd_max_cumulative_flow"),
        (pd_min_cumulative_flow_from_source, "pd_min_cumulative_flow"),
    ):
        setattr(flex_data, field,
                _filter_param_by_periods(fn(source, period_filter=dt), dt))

    # ─── Δ.4b — group scalar (no period) ─────────────────────────────────
    for fn, field in (
        (p_group_invest_max_total_from_source, "p_group_invest_max_total"),
        (p_group_invest_min_total_from_source, "p_group_invest_min_total"),
        (p_group_retire_max_total_from_source, "p_group_retire_max_total"),
        (p_group_retire_min_total_from_source, "p_group_retire_min_total"),
        (p_group_invest_max_cumulative_from_source,
            "p_group_invest_max_cumulative"),
        (p_group_invest_min_cumulative_from_source,
            "p_group_invest_min_cumulative"),
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
        pdt_max_instant_flow_from_source(source, period_filter=dt), dt)
    if v is not None:
        flex_data.pdt_max_instant_flow = v
    v = _filter_param_by_periods(
        pdt_min_instant_flow_from_source(source, period_filter=dt), dt)
    if v is not None:
        flex_data.pdt_min_instant_flow = v

    # ─── Δ.4b — multi-class union 1d_map(period): invest / divest ────────
    # Δ.17c-Tier2 — helpers now accept ``period_filter`` (consumed by
    # ``resolve_param_shape`` for silent-default disambiguation and by
    # ``broadcast_to_period`` for the active-solve period restriction and
    # scalar-to-(e,d) cross-join).
    for fn, field in (
        (ed_invest_max_period_from_source, "ed_invest_max_period"),
        (ed_divest_max_period_from_source, "ed_divest_max_period"),
        (ed_invest_min_period_from_source, "ed_invest_min_period"),
        (ed_divest_min_period_from_source, "ed_divest_min_period"),
        (ed_cumulative_max_capacity_from_source, "ed_cumulative_max_capacity"),
        (ed_cumulative_min_capacity_from_source, "ed_cumulative_min_capacity"),
    ):
        setattr(flex_data, field,
                _filter_param_by_periods(fn(source, period_filter=dt), dt))

    # ─── Δ.4b — relationship scalars (ramp + inertia, sink/source) ───────
    for fn, field in (
        (p_ramp_speed_up_sink_from_source, "p_ramp_speed_up_sink"),
        (p_ramp_speed_down_sink_from_source, "p_ramp_speed_down_sink"),
        (p_ramp_speed_up_source_from_source, "p_ramp_speed_up_source"),
        (p_ramp_speed_down_source_from_source, "p_ramp_speed_down_source"),
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

    # ─── Δ.4b — CO2 (price + cap) ────────────────────────────────────────
    v = _filter_param_by_periods(
        p_co2_price_from_source(source, period_filter=dt), dt)
    if v is not None:
        flex_data.p_co2_price = v
    v = _filter_param_by_periods(
        p_co2_max_period_from_source(source, period_filter=dt), dt)
    if v is not None:
        flex_data.p_co2_max_period = v

    # ─── Δ.4b — variable cost (Map period→time) ──────────────────────────
    # Δ.17c-Tier3 — the four helpers below route through the resolver
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
    flex_data.p_process_reserve_upDown_node_reliability = (
        p_process_reserve_upDown_node_reliability_from_source(source))
    flex_data.p_process_reserve_upDown_node_max_share = (
        p_process_reserve_upDown_node_max_share_from_source(source))
    flex_data.p_process_reserve_upDown_node_large_failure_ratio_value = (
        p_process_reserve_upDown_node_large_failure_ratio_value_from_source(source))
    flex_data.p_process_reserve_upDown_node_increase_reserve_ratio_value = (
        p_process_reserve_upDown_node_increase_reserve_ratio_value_from_source(source))

    # ─── Δ.4b — delayed processes (set, not Param) ───────────────────────
    flex_data.process_delayed__duration = process_delayed__duration_from_source(source)

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
    p_ann_price, p_ann_qty = _ladder_split(source, "price_ladder_annual",
                                              with_period=True)
    if p_ann_price is not None:
        flex_data.p_ladder_ann_price = _filter_param_by_periods(p_ann_price, dt)
    if p_ann_qty is not None:
        flex_data.p_ladder_ann_quantity = _filter_param_by_periods(p_ann_qty, dt)
    p_cum_price, p_cum_qty = _ladder_split(source, "price_ladder_cumulative",
                                              with_period=False)
    if p_cum_price is not None:
        flex_data.p_ladder_cum_price = p_cum_price
    if p_cum_qty is not None:
        flex_data.p_ladder_cum_quantity = p_cum_qty


def apply_direct_params(source: "InputSource",
                          flex_data: object) -> None:
    """Apply the full Direct Param wave in two phases.

    Δ.28 — kept as a single back-compat entry point.  The dispatcher in
    :func:`flextool.engine_polars.input._apply_db_overrides` does NOT
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
