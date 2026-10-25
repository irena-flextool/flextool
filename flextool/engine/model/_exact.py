"""The ``exact`` transfer method: a connection flows in one direction per time step.

Owns the variable v_exact_forward (one binary per exact connection and step: the connection is set to its forward
direction) and the rows maxFlow_exact_forward / maxFlow_exact_reverse.  The flows' capacity rows (maxFlow,
maxFlow_online) stay in model/core.py and model/_online.py; the connection reserve rows read :func:`exact_arcs` and
:func:`exact_bound` for their direction-switch rows (model/_reserve.py).
"""

from typing import TYPE_CHECKING

import polars as pl
from polar_high import Param

from flextool.engine.input._param_shapes import promote_param_to_dt
from flextool.engine.model._common import _has_rows, _utf8_keys

if TYPE_CHECKING:
    from flextool.engine.model.core import _Build

# Data fields the exact rows need (checked when process_exact has rows).
EXACT: tuple[str, ...] = ("process_exact", "process_source_canonical", "p_flow_upper")


def has_feature(d) -> bool:
    """True when some connection uses the ``exact`` transfer method."""
    return _has_rows(getattr(d, "process_exact", None))


def exact_arcs(d) -> "pl.DataFrame | None":
    """``(p, source, sink, forward)`` of the exact connections' v_flow arcs, in the dtypes of
    ``process_source_sink``; ``forward`` is True on the node_1 → node_2 arc (``process_source_canonical``).

    Only connections with both arcs in process_source_sink (a one-variable arc has no direction to switch).
    None when there is no such connection.
    """
    if not has_feature(d) or not _has_rows(d.process_source_sink):
        return None
    keys = ["p", "source", "sink"]
    pss = _utf8_keys(d.process_source_sink.lazy().select(keys), keys)
    ex = _utf8_keys(d.process_exact.lazy().select("p"), ["p"]).select("_p").unique()
    arcs = pss.join(ex, on="_p", how="semi")
    rev = arcs.select(pl.col("_p"), pl.col("_sink").alias("_source"), pl.col("_source").alias("_sink"))
    arcs = arcs.join(rev, on=["_p", "_source", "_sink"], how="semi")
    canon = (_utf8_keys(d.process_source_canonical.lazy().select("p", "source"), ["p", "source"])
             .select("_p", "_source").unique().with_columns(pl.lit(True).alias("forward")))
    out = (arcs.join(canon, on=["_p", "_source"], how="left")
           .with_columns(pl.col("forward").fill_null(False))
           .select(*keys, "forward").collect())
    # Exactly one forward arc per connection.
    n_fwd = out.filter(pl.col("forward")).select("p").unique().height
    if out.height == 0:
        return None
    if n_fwd != out.select("p").unique().height or out.filter(pl.col("forward")).height != n_fwd:
        raise ValueError(f"_exact: every exact connection needs one forward (node_1 → node_2) arc: {out}")
    return out


def exact_bound(d, arcs: "pl.DataFrame") -> Param:
    """Ū[p,source,sink,d,t] = p_flow_upper · capacity_max_coeff · availability over ``arcs`` × dt (unitsize units;
    capacity_max_coeff and availability default to 1).  p_flow_upper is the structural upper bound of the arc's
    flow (existing + the most the connection can invest by d, max_flow_for_unconstrained_variables for unlimited
    investment), so Ū is a valid big-M for the direction rows; without investment it is the existing capacity."""
    keys = ["p", "source", "sink"]
    grid = _utf8_keys(arcs.select(keys).unique().join(d.dt.select("d", "t"), how="cross").lazy(),
                      [*keys, "d", "t"])
    fu = (_utf8_keys(promote_param_to_dt(d.p_flow_upper, d.dt).select(*keys, "d", "t", "value"),
                     [*keys, "d", "t"])
          .select("_p", "_source", "_sink", "_d", "_t", pl.col("value").alias("__fu")))
    grid = grid.join(fu, on=["_p", "_source", "_sink", "_d", "_t"], how="left")
    coef = getattr(d, "p_arc_max_cap_coef", None)
    if coef is not None:
        grid = grid.join(_utf8_keys(coef.frame.lazy().select(*keys, "value"), keys)
                         .select("_p", "_source", "_sink", pl.col("value").alias("__coef")),
                         on=["_p", "_source", "_sink"], how="left")
    else:
        grid = grid.with_columns(pl.lit(None, pl.Float64).alias("__coef"))
    avail = getattr(d, "p_process_availability", None)
    if avail is not None:
        grid = grid.join(_utf8_keys(promote_param_to_dt(avail, d.dt).select("p", "d", "t", "value"), ["p", "d", "t"])
                         .select("_p", "_d", "_t", pl.col("value").alias("__av")),
                         on=["_p", "_d", "_t"], how="left")
    else:
        grid = grid.with_columns(pl.lit(None, pl.Float64).alias("__av"))
    frame = grid.with_columns(
        value=pl.col("__fu") * pl.col("__coef").fill_null(1.0) * pl.col("__av").fill_null(1.0)).collect()
    missing = frame.filter(pl.col("value").is_null())
    if missing.height > 0:
        raise ValueError("_exact: p_flow_upper has no value for exact arc(s) "
                         f"{missing.select(keys).unique().head(10).to_dicts()}.")
    return Param((*keys, "d", "t"), frame.select(*keys, "d", "t", "value"))


def add_variables(b: "_Build") -> None:
    """Declare v_exact_forward."""
    d = b.d
    if not has_feature(d) or b.v_flow is None:
        return
    for f in EXACT:
        if getattr(d, f, None) is None:
            raise ValueError(f"_exact: the exact transfer method is used but {f} is not populated (None).")
    arcs = exact_arcs(d)
    if arcs is None:
        return
    b.exact_arcs = arcs

    # ─── VAR. ─── v_exact_forward ────────────────────────────────────────────
    # ─────────────────────────────────────────────────────────────────────────
    # v_exact_forward[p,d,t] ∈ {0, 1}   ∀ p ∈ process_exact with both arcs, (d,t) ∈ dt
    # 1: the exact connection is set to its forward direction (node_1 → node_2 of connection__node__node) in the
    # step, 0: to the reverse direction.  One binary per connection: the whole connection transfers in one
    # direction.  No cost, no inter-temporal link.
    idx = arcs.select("p").unique(maintain_order=True).join(d.dt.select("d", "t"), how="cross")
    b.v_exact_forward = b.m.add_var("v_exact_forward", ("p", "d", "t"), idx, lower=0.0, upper=1.0, integer=True)


def add_constraints(b: "_Build") -> None:
    """maxFlow_exact_forward / maxFlow_exact_reverse."""
    if b.v_exact_forward is None:
        return
    m, d = b.m, b.d
    arcs = b.exact_arcs
    fwd = arcs.filter(pl.col("forward")).select("p", "source", "sink")
    rev = arcs.filter(~pl.col("forward")).select("p", "source", "sink")
    y = b.v_exact_forward

    # ─── EQ. ─── maxFlow_exact_forward / maxFlow_exact_reverse ───────────────
    # ─────────────────────────────────────────────────────────────────────────
    # v_flow[p,a,b,d,t] − Ū[p,a,b,d,t] · v_exact_forward[p,d,t]  ≤  0                         (forward)
    # v_flow[p,b,a,d,t] + Ū[p,b,a,d,t] · v_exact_forward[p,d,t]  ≤  Ū[p,b,a,d,t]              (reverse)
    #   ∀ p ∈ process_exact, (a,b) = (node_1, node_2), (d,t) ∈ dt
    #   Ū = p_flow_upper · capacity_max_coeff · availability (structural upper bound, units of unitsize)
    # An exact connection does not flow both ways at once (docs: the exact method needs a binary variable): in
    # each step it carries flow in one direction only.  maxFlow and maxFlow_online still bound each direction by
    # the built and the online capacity; with investment Ū is a valid big-M (GMPL used v_online_integer and the
    # unit count as the switch, which fails for a non-integer number of units).
    m.add_cstr("maxFlow_exact_forward", over=fwd.join(d.dt.select("d", "t"), how="cross"), sense="<=",
               lhs_terms={"flow": b.v_flow, "switch_neg": -(y * exact_bound(d, fwd))},
               rhs_terms={"zero": 0.0})
    u_rev = exact_bound(d, rev)
    m.add_cstr("maxFlow_exact_reverse", over=rev.join(d.dt.select("d", "t"), how="cross"), sense="<=",
               lhs_terms={"flow": b.v_flow, "switch": y * u_rev},
               rhs_terms={"upper": u_rev})
