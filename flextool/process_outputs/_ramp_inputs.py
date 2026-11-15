"""Per-solve inputs of the ramp envelope (``calc_ramp_envelope``), read from a solve's own ``FlexData``.

Each function returns a long polars frame keyed by ``(..., d, t)`` over the solve's steps, so that
``read_parameters`` turns it into a solve-keyed ``(solve, period, time)`` frame: the per-roll realized slices
(``persist_realized_slice``) then carry, after the union, the value of the solve that realized each step.  That
matters for the ramp-limit sets: a solve whose ``reduce_unit_detail_method`` leaves out the ramps has none
(``_reduce_detail``), and the envelope of a transition uses the set of the solve that realized its later step.

The coefficients come from the model's own helpers (never re-derived here):

* the capacity multiplier of an arc's ``maxFlow`` row (``core._max_flow_rhs``: capacity coefficient × availability
  on direct arcs and indirect output arcs, the input limit × x(availability) on indirect input arcs);
* the floor per unit online (``_unit_floors.arc_floor_share``);
* the ramp-limit arc sets with their speeds (``process_source_sink_ramp_limit_{side}_{dir}`` without the
  ``conversion_flow_coeff = 0`` arcs, ``_common._drop_uncapped``) and the ramp row's capacity factor c
  (``core._c_param_plain``; a coarse process's time-varying c is its coarse step's Δt-weighted mean, as
  ``_coarse_flows.at_first_steps`` gives the ramp row);
* the coarse steps of the coarse processes (``FlexData.process_coarse_step``).

Only unit arcs are covered (connections have no ramp limit and their range is read from their capacity and
availability).  Values equal to the default are left out (1 for the capacity multiplier and c, 0 for the floor,
no row = not ramp limited); ``calc_ramp_envelope`` fills the defaults.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl

if TYPE_CHECKING:
    from flextool.engine.preprocess.flex_data import FlexData

_KEYS = ["p", "source", "sink"]
#: The four ramp-limit sets: (label, side, direction).
RAMP_LIMITS: tuple[tuple[str, str, str], ...] = (
    ("sink_up", "sink", "up"),
    ("sink_down", "sink", "down"),
    ("source_up", "source", "up"),
    ("source_down", "source", "down"),
)


def _has_rows(frame) -> bool:
    return frame is not None and getattr(frame, "height", 0) > 0


def _dt(d: "FlexData") -> "pl.DataFrame | None":
    dt = getattr(d, "dt", None)
    if not _has_rows(dt):
        return None
    return dt.select(pl.col("d").cast(pl.Utf8), pl.col("t").cast(pl.Utf8))


def _unit_arcs(d: "FlexData") -> "pl.DataFrame | None":
    """The ``(p, source, sink)`` arcs (Utf8) of the units, without the ``conversion_flow_coeff = 0`` arcs."""
    from flextool.engine.model._common import _drop_uncapped

    pss = getattr(d, "process_source_sink", None)
    units = getattr(d, "process_unit", None)
    if not _has_rows(pss) or not _has_rows(units):
        return None
    arcs = _drop_uncapped(d, pss.select(_KEYS).unique(maintain_order=True))
    arcs = arcs.select(pl.col(k).cast(pl.Utf8) for k in _KEYS)
    arcs = arcs.join(units.select(pl.col("p").cast(pl.Utf8)).unique(), on="p", how="semi")
    return arcs if arcs.height > 0 else None


def _utf8(frame: pl.DataFrame, cols) -> pl.DataFrame:
    return frame.with_columns(pl.col(c).cast(pl.Utf8) for c in cols)


def _on_arcs_dt(param, base: pl.DataFrame, *, default: float) -> pl.DataFrame:
    """``param`` (a Param over any subset of ``p, source, sink, d, t``) read at every row of ``base``
    (``p, source, sink, d, t``, Utf8), ``default`` where it has no value: ``base`` + ``value``."""
    if param is None:
        return base.with_columns(value=pl.lit(default, pl.Float64))
    frame = param.frame.collect() if isinstance(param.frame, pl.LazyFrame) else param.frame
    on = [k for k in (*_KEYS, "d", "t") if k in frame.columns]
    frame = _utf8(frame.select(*on, pl.col("value").cast(pl.Float64)), on)
    return (base.join(frame, on=on, how="left")
            .with_columns(pl.col("value").fill_null(default)))


def arc_capacity_factor(d: "FlexData") -> "pl.DataFrame | None":
    """``(p, source, sink, d, t, value)``: the capacity multiplier of each unit arc's ``maxFlow`` row
    (``core._max_flow_rhs``'s ``cap_mult``: v_flow ≤ value · capacity), rows where it is not 1."""
    from flextool.engine.model.core import _max_flow_rhs

    arcs = _unit_arcs(d)
    if arcs is None or _dt(d) is None:
        return None
    _rhs, cap_mult, _upper, _bound = _max_flow_rhs(d)
    if cap_mult is None or cap_mult.height == 0:
        return None
    out = (_utf8(cap_mult.select(*_KEYS, "d", "t", pl.col("value").cast(pl.Float64)), [*_KEYS, "d", "t"])
           .join(arcs, on=_KEYS, how="semi"))
    return out if out.height > 0 else None


def arc_floor_share(d: "FlexData") -> "pl.DataFrame | None":
    """``(p, source, sink, d, t, value)``: the floor per unit online of each unit arc
    (``_unit_floors.arc_floor_share``), rows where it is > 0."""
    from flextool.engine.model import _unit_floors

    arcs = _unit_arcs(d)
    if arcs is None:
        return None
    share = _unit_floors.arc_floor_share(d, arcs)
    if share is None or share.height == 0:
        return None
    return _utf8(share.select(*_KEYS, "d", "t", pl.col("value").cast(pl.Float64)), [*_KEYS, "d", "t"])


def _coarse_mean(frame: pl.DataFrame, d: "FlexData") -> pl.DataFrame:
    """``frame`` (``p, source, sink, limit, d, t, value``) with the rows of each coarse process replaced by their
    Δt-weighted mean over its coarse step (the ramp row's c of a coarse process, ``_coarse_flows.at_first_steps``),
    written to every fine step of the coarse step."""
    steps = getattr(d, "process_coarse_step", None)
    if not _has_rows(steps):
        return frame
    steps = _utf8(steps.select("p", "d", "b_first", "t", pl.col("w").cast(pl.Float64)), ["p", "d", "b_first", "t"])
    coarse = frame.join(steps, on=["p", "d", "t"], how="inner")
    if coarse.height == 0:
        return frame
    means = (coarse.group_by(*_KEYS, "limit", "d", "b_first")
             .agg((pl.col("value") * pl.col("w")).sum().alias("__mean")))
    coarse = (coarse.join(means, on=[*_KEYS, "limit", "d", "b_first"], how="left")
              .select(*_KEYS, "limit", "d", "t", pl.col("__mean").alias("value")))
    fine = frame.join(steps.select("p", "d", "t"), on=["p", "d", "t"], how="anti")
    return pl.concat([fine, coarse])


def ramp_limits(d: "FlexData") -> "tuple[pl.DataFrame | None, pl.DataFrame | None]":
    """``(speed, c)``, each ``(p, source, sink, limit, d, t, value)`` over the arcs of this solve's ramp-limit
    sets (``limit`` ∈ :data:`RAMP_LIMITS` labels): the ramp speed (share of capacity per minute) and the ramp
    row's capacity factor c (``core._c_param_plain``; the rows where c is not 1).  ``(None, None)`` without
    ramp limits."""
    from flextool.engine.model._common import _drop_uncapped
    from flextool.engine.model.core import _c_param_plain

    dt = _dt(d)
    if dt is None:
        return None, None
    speeds, coefs = [], []
    for label, side, dir_ in RAMP_LIMITS:
        idx_set = getattr(d, f"process_source_sink_ramp_limit_{label}", None)
        speed = getattr(d, f"p_ramp_speed_{dir_}_{side}", None)
        if not _has_rows(idx_set) or speed is None:
            continue
        idx_set = _drop_uncapped(d, idx_set)
        if idx_set.height == 0:
            continue
        arcs = _utf8(idx_set.select(_KEYS).unique(maintain_order=True), _KEYS)
        sp = _utf8(speed.frame.select("p", side, pl.col("value").cast(pl.Float64)), ["p", side])
        rows = arcs.join(sp, on=["p", side], how="inner")
        if rows.height == 0:
            continue
        base = rows.select(_KEYS).join(dt, how="cross")
        speeds.append(rows.join(dt, how="cross")
                      .select(*_KEYS, pl.lit(label).alias("limit"), "d", "t", "value"))
        c = _on_arcs_dt(_c_param_plain(d, idx_set, side), base, default=1.0)
        coefs.append(c.select(*_KEYS, pl.lit(label).alias("limit"), "d", "t", "value"))
    if not speeds:
        return None, None
    coef = _coarse_mean(pl.concat(coefs), d).filter(pl.col("value") != 1.0)
    return pl.concat(speeds), (coef if coef.height > 0 else None)


def coarse_steps(d: "FlexData") -> "tuple[pl.DataFrame | None, pl.DataFrame | None]":
    """``(duration, last)``, each ``(p, d, t, value)`` over the fine steps of the coarse processes
    (``FlexData.process_coarse_step``): the duration Δ_b of the coarse step that holds the step (h), and 1 at
    the last fine step of its coarse step, 0 at the others.  ``(None, None)`` without coarse processes."""
    steps = getattr(d, "process_coarse_step", None)
    dur = getattr(d, "p_step_duration", None)
    if not _has_rows(steps) or dur is None:
        return None, None
    steps = _utf8(steps.select("p", "d", "b_first", "t"), ["p", "d", "b_first", "t"])
    dur = _utf8(dur.frame.select("d", "t", pl.col("value").cast(pl.Float64)), ["d", "t"])
    with_dur = steps.join(dur, on=["d", "t"], how="inner")
    # Solve order of the fine steps (the coarse step's last fine step is its last in the solve's dt).
    order = _dt(d).with_row_index("__pos")
    with_dur = with_dur.join(order, on=["d", "t"], how="inner")
    agg = (with_dur.group_by("p", "d", "b_first")
           .agg(pl.col("value").sum().alias("__dur_b"), pl.col("__pos").max().alias("__last_pos")))
    full = with_dur.join(agg, on=["p", "d", "b_first"], how="left")
    duration = full.select("p", "d", "t", pl.col("__dur_b").alias("value"))
    last = full.select("p", "d", "t", (pl.col("__pos") == pl.col("__last_pos")).cast(pl.Float64).alias("value"))
    return duration, last
