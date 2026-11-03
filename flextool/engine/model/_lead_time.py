"""Construction lead time: the investment terms that count capacity coming online or in construction.

An investment decision ``v_invest[e,o]`` with a construction lead time starts to operate in a later period
``c(o)`` (``edd_invest_commission``, ``(e, o, c(o))``; ``(e, o, o)`` for a decision without a lead time) and is in
construction from ``o`` until ``c(o)`` (``edd_invest_in_construction``).  Investments realized by earlier solves of
a chain enter as constants (``p_entity_carried_commission`` / ``p_entity_carried_in_construction``).

The helpers here are used by the per-period investment limits (model/core.py, model/_cumulative_invest.py) and
the user constraints (model/core.py).  They are active only when the solve has a decision with a lead time
(:func:`active`); without one the rows keep the formulation over ``v_invest[e,d]`` (byte-identical LP).  The
module declares no variables and adds no constraint families of its own.
"""
from __future__ import annotations

import polars as pl
from polar_high import Param, Sum, Where
from polar_high.engine import Var

from flextool.engine.input._axis_enums import align_join_dtypes, rename_to_axis
from flextool.engine.input._solve_state import FlexToolConfigError
from flextool.engine.model._common import _entity_rows, _has_rows, _var_view, dispatch_periods

#: Tolerance of the checks on rows without a variable (MW / MWh).
_TOL = 1e-6


def active(d) -> bool:
    """True when some investment decision of the solve has a construction lead time."""
    return _has_rows(getattr(d, "edd_invest_commission", None))


def commission_rows(d, var: Var, axis: str) -> pl.DataFrame:
    """``[axis, d_invest, d]``: the period ``d`` in which each decision ``(axis, d_invest)`` of ``var`` starts to
    operate."""
    rows = _entity_rows(d.edd_invest_commission, "e", var.frame.select(axis).unique(), axis)
    keys = var.frame.select(pl.col(axis), pl.col("d").alias("d_invest")).unique()
    rows, keys = align_join_dtypes(rows, keys, [axis, "d_invest"])
    return rows.join(keys, on=[axis, "d_invest"], how="semi")


def in_construction_rows(d, var: Var, axis: str) -> pl.DataFrame | None:
    """``[axis, d_invest, d]``: the periods ``d`` in which the capacity of decision ``(axis, d_invest)`` is in
    construction; ``None`` without such rows."""
    frame = getattr(d, "edd_invest_in_construction", None)
    if not _has_rows(frame):
        return None
    rows = _entity_rows(frame, "e", var.frame.select(axis).unique(), axis)
    return rows if rows.height > 0 else None


def decided_sum(var: Var, unitsize, rows: pl.DataFrame, suffix: str, over: tuple[str, ...]):
    """``Σ_{over} v[axis, d_invest]·unitsize`` over ``rows`` (which carry ``d_invest``, ``d`` and the row keys)."""
    view = _var_view(var, suffix, {"d": "d_invest"}, to_axis=True)
    return Sum(Where(view * unitsize, rows), over=over)


def carried(d, kind: str, axis: str, members: pl.DataFrame | None = None) -> pl.DataFrame | None:
    """``[axis, d, value]`` of the capacity earlier solves realized that comes online (``kind="commission"``) or is
    in construction (``kind="in_construction"``) in this solve's periods; ``None`` without such capacity.
    ``members`` (an ``axis`` column) restricts the entities."""
    param = getattr(d, f"p_entity_carried_{kind}", None)
    if param is None or param.frame.height == 0:
        return None
    entities = (members if members is not None
                else (d.process_source_sink if axis == "p" else d.nodeState))
    if not _has_rows(entities):
        return None
    rows = _entity_rows(param.frame, "e", entities.select(axis).unique(), axis).filter(pl.col("value") != 0.0)
    return rows if rows.height > 0 else None


def lead_time_entities(d, axis: str, members: pl.DataFrame) -> pl.DataFrame:
    """``[axis, L]``: the entities of ``members`` with a construction lead time and their longest lead time."""
    lagged = getattr(d, "e_construction_lead_time", None)
    if not _has_rows(lagged):
        return pl.DataFrame(schema={axis: members.schema[axis], "L": pl.Float64})
    return _entity_rows(lagged, "e", members.select(axis).unique(), axis).select(axis, "L").unique()


def check_rows_without_variable(rows: pl.DataFrame, keys: list[str], limit: pl.DataFrame | None,
                                carried_rows: pl.DataFrame | None, sense: str, param_name: str) -> None:
    """Raise :class:`FlexToolConfigError` where a per-period limit row has no investment variable (nothing that
    can come online there) and the capacity carried from earlier solves breaks the limit.

    ``rows`` are the row keys without a variable with the entity's lead time ``L``, ``limit`` the limit
    ``[*keys, value]``, ``carried_rows`` the carried capacity ``[*keys, value]`` (``None``: nothing carried)."""
    if rows.height == 0 or limit is None:
        return
    lim = limit.select(*keys, pl.col("value").alias("limit"))
    rows, lim = align_join_dtypes(rows, lim, keys)
    check = rows.join(lim, on=keys, how="inner")
    if carried_rows is not None:
        car = carried_rows.select(*keys, pl.col("value").alias("carried"))
        check, car = align_join_dtypes(check, car, keys)
        check = check.join(car, on=keys, how="left")
    else:
        check = check.with_columns(carried=pl.lit(0.0))
    check = check.with_columns(pl.col("carried").fill_null(0.0))
    bad = check.filter((pl.col("limit") > pl.col("carried") + _TOL) if sense == ">="
                       else (pl.col("carried") > pl.col("limit") + _TOL))
    if bad.height == 0:
        return
    cases = "; ".join(
        f"{r[keys[0]]}: nothing can come online in {r['d']} with a {r['L']:g}-year construction_lead_time "
        f"({r['carried']:g} from earlier solves, {param_name} {r['limit']:g})"
        for r in bad.sort(keys).iter_rows(named=True))
    raise FlexToolConfigError(
        f"{param_name} cannot be met: {cases}.  An investment starts to operate construction_lead_time years after "
        "its decision, so the solve needs a decision period that much earlier.")


def period_limit(b, name: str, axis: str, var: Var, unitsize, entity_rows: pl.DataFrame,
                 limit: Param | None, sense: str, param_name: str) -> pl.DataFrame | None:
    """Add ``name``: ``Σ_{(e,o,d) ∈ commission} v[e,o]·unitsize + carried[e,d] (sense) limit[e,d]`` over the
    ``(axis, d)`` that have a decision coming online, for the entities of ``entity_rows`` (``[axis]``).  Rows
    without a variable are checked against the carried capacity for the entities with a lead time.  Returns the
    row frame (``None`` when no row)."""
    m, d = b.m, b.d
    if limit is None:
        return None
    rows = commission_rows(d, var, axis)
    rows, ents = align_join_dtypes(rows, entity_rows.select(axis).unique(), [axis])
    rows = rows.join(ents, on=axis, how="semi")
    over = rows.select(axis, "d").unique().sort(axis, "d")
    car = carried(d, "commission", axis, entity_rows)
    limit_frame = limit.frame.pipe(rename_to_axis, {"e": axis})
    # Rows without a variable: the entities with a lead time in this solve's periods.
    lagged = lead_time_entities(d, axis, entity_rows)
    if lagged.height > 0:
        grid = lagged.join(dispatch_periods(d), how="cross")
        grid, over_k = align_join_dtypes(grid, over, [axis, "d"])
        empty = grid.join(over_k, on=[axis, "d"], how="anti")
        check_rows_without_variable(empty, [axis, "d"], limit_frame, car, sense, param_name)
    if over.height == 0:
        return None
    lhs: dict = {"invest": decided_sum(var, unitsize, rows, "__at_online", ("d_invest",))}
    if car is not None:
        car_rows, over_c = align_join_dtypes(car, over, [axis, "d"])
        car_rows = car_rows.join(over_c, on=[axis, "d"], how="semi")
        if car_rows.height > 0:
            lhs["carried"] = Param((axis, "d"), car_rows.select(axis, "d", "value"))
    m.add_cstr(
        name,
        over      = over,
        sense     = sense,
        lhs_terms = lhs,
        rhs_terms = {"cap": Param((axis, "d"), limit_frame.select(axis, "d", "value"))},
    )
    return over


__all__ = [
    "active",
    "carried",
    "check_rows_without_variable",
    "commission_rows",
    "decided_sum",
    "in_construction_rows",
    "lead_time_entities",
    "period_limit",
]
