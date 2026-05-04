"""First wave of Direct Param helpers (Γ.1).

Each function in this module takes an :class:`InputSource` and returns
a single :class:`polar_high_opt.Param` (or, for entity-only sets, a
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

from polar_high_opt import Param

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
    """``constraint.constant`` → ``Param(("c",), [c, value])``.

    Default ``0.0`` — broadcast to every constraint via the source.
    """
    df = source.parameter("constraint", "constant")
    if df.height == 0:
        return None
    return Param(("c",),
                 df.lazy().rename({"name": "c"}).select("c", "value"))


def constraint_sense_from_source(source: "InputSource") -> pl.DataFrame:
    """Return the constraint sense set as ``[c, sense]``.  Empty frame
    if no constraints carry a ``sense`` value (None-default policy).
    """
    df = source.parameter("constraint", "sense")
    if df.height == 0:
        return pl.DataFrame(schema={"c": pl.Utf8, "sense": pl.Utf8})
    return df.lazy().rename({"name": "c", "value": "sense"}).select("c", "sense").collect()


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
    """Return ``Param(("n", "c"), [n, c, value])`` for a per-(node,
    constraint) coefficient parameter.  ``None`` if the DB has no rows.
    """
    df = source.parameter("node", parameter_name)
    if df.height == 0:
        return None
    return Param(
        ("n", "c"),
        df.lazy()
          .rename({"name": "n", "constraint": "c"})
          .select("n", "c", "value")
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
              .rename({"name": "p", "constraint": "c"})
              .select("p", "c", "value")
        )
    if not parts:
        return None
    return Param(("p", "c"), pl.concat(parts).sort("p", "c"))


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


def apply_direct_params(source: "InputSource",
                          flex_data: object) -> None:
    """Apply the DB-direct construction for Γ.1 first-wave Direct Params,
    mutating ``flex_data`` in place.

    Each FlexData field listed below is built by exactly one helper.
    When the helper returns ``None`` (no upstream data), the field is
    left untouched; otherwise the helper's result replaces the field.

    Δ.3: this replaces the previous ``first_wave_overrides`` dict-return
    pattern.  The dict-overlay round-trip is gone — each helper writes
    its field directly.  See progress.md (Δ.3 close stanza).
    """
    # ─── §5.2.1 scalar Params with FlexData fields ──────────────────────
    p_co2 = p_co2_content_from_source(source)
    if p_co2 is not None:
        flex_data.p_co2_content = p_co2
    p_const = p_constraint_constant_from_source(source)
    if p_const is not None:
        flex_data.p_constraint_constant = p_const

    # ─── §5.2.3 relationship 1d_map (constraint coefficients) ───────────
    n_inv = _node_constraint_coef(source, "constraint_invested_capacity_coefficient")
    if n_inv is not None:
        flex_data.p_node_constraint_invested_capacity_coefficient = n_inv
    p_inv = _process_constraint_coef(source, "constraint_invested_capacity_coefficient")
    if p_inv is not None:
        flex_data.p_process_constraint_invested_capacity_coefficient = p_inv
    n_state = _node_constraint_coef(source, "constraint_state_coefficient")
    if n_state is not None:
        flex_data.p_node_constraint_state_coefficient = n_state
    n_pre = _node_constraint_coef(source, "constraint_cumulative_pre_built_capacity_coefficient")
    if n_pre is not None:
        flex_data.p_node_constraint_prebuilt_capacity_coefficient = n_pre
    p_pre = _process_constraint_coef(source, "constraint_cumulative_pre_built_capacity_coefficient")
    if p_pre is not None:
        flex_data.p_process_constraint_prebuilt_capacity_coefficient = p_pre

    # ─── §5.2.1 invest/divest total caps (entity-unioned) ───────────────
    # Max variants: keyed on entityInvest (resp. entityDivest), one row
    # per entity with value defaulting to 0 — mirrors
    # ``entity_total_caps.py:_compute_entity_total`` and
    # ``input.py::_e_total_param``.
    e_inv = _e_total_param(source, "invest_max_total", kind="invest")
    if e_inv is not None:
        flex_data.e_invest_max_total = e_inv
    e_div = _e_total_param(source, "retire_max_total", kind="divest")
    if e_div is not None:
        flex_data.e_divest_max_total = e_div
    # Min variants: CSV path filters out zero rows (input.py::_read_e_param)
    # — None when no entity has an explicit non-zero min cap.
    e_inv_min = _e_total_param(source, "invest_min_total", kind="invest",
                                filter_zero=True)
    if e_inv_min is not None:
        flex_data.e_invest_min_total = e_inv_min
    e_div_min = _e_total_param(source, "retire_min_total", kind="divest",
                                filter_zero=True)
    if e_div_min is not None:
        flex_data.e_divest_min_total = e_div_min


# Deprecated alias scheduled for deletion in Δ.4 — preserved for any
# external callers / docstring references.  Calls the new direct path.
def first_wave_overrides(source: "InputSource",
                          flex_data: object) -> dict[str, object | None]:
    """Deprecated.  Use :func:`apply_direct_params` instead.

    Δ.3: this thin pass-through is preserved for one phase to keep
    external callers compiling; it no longer participates in the
    override-chain plumbing.  Prefer ``apply_direct_params`` which
    mutates ``flex_data`` directly.  Slated for deletion in Δ.4.
    """
    apply_direct_params(source, flex_data)
    return {}
