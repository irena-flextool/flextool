"""Per-solve inputs of the ramp envelope (``calc_ramp_envelope``), read from a solve's own ``FlexData``.

Each function returns a long polars frame keyed by ``(..., d, t)`` over the solve's steps (only the rows that differ
from the default), so that ``read_parameters`` turns it into a solve-keyed ``(solve, period, time)`` frame: the
per-roll realized slices (``persist_realized_slice``) then carry, after the union, the value of the solve that
realized each step.  The ramp-limit sets and their speeds have no time axis: :func:`ramp_limits` gives them per
solve, and ``read_parameters`` keys them by ``(solve, period)`` over the solve's periods; the envelope reads the row
of the solve that realized each step (``s.solve_period_time``).  That matters because a solve whose
``reduce_unit_detail_method`` leaves out the ramps has none (``_reduce_detail``), and the envelope of a transition
uses the set of the solve that realized its later step.

The coefficients come from the model's own helpers (never re-derived here):

* the capacity multiplier of an arc's ``maxFlow`` row (``core._max_flow_rhs``: capacity coefficient × availability
  on direct arcs and indirect output arcs, the input limit × x(availability) on indirect input arcs);
* the floor per unit online (``_unit_floors.arc_floor_share``);
* the ramp-limit arc sets with their speeds (``process_source_sink_ramp_limit_{side}_{dir}`` without the
  ``conversion_flow_coeff = 0`` arcs, ``_common._drop_uncapped``) and the ramp row's capacity factor c
  (``core._c_param_plain``);
* the coarse steps of the coarse processes (``FlexData.process_coarse_step``).

A coarse process has one flow per coarse step, and its ramp row between two coarse steps takes the Δt-weighted means
over each coarse step of the time-varying c and floor F (``_coarse_flows.at_first_steps``, the means of c and F in
the ramp row of ``core._eq_ramp``); its capacity multiplier enters the coarse step's capacity row as the same mean
(``_coarse_flows.average``).  So for a coarse process the capacity multiplier, the floor and c are written as their
coarse-step means to every fine step of the coarse step (:func:`_coarse_mean`): the envelope reads them at any fine
step of a coarse step and gets the value of that coarse step.

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


def _on_arcs_dt(param, arcs: pl.DataFrame, dt: pl.DataFrame, *, default: float) -> "pl.DataFrame | None":
    """``(p, source, sink, d, t, value)`` (Utf8 keys): ``param`` (a Param over any subset of ``p, source, sink, d,
    t``) read at the arcs ``arcs`` (``p, source, sink``) and the steps ``dt``, only the rows whose value is not
    ``default`` (no row = ``default``); ``None`` without such a row.  Only those arcs are crossed with the steps a
    time-invariant ``param`` lacks (never the whole arcs × steps)."""
    if param is None:
        return None
    frame = param.frame.collect() if isinstance(param.frame, pl.LazyFrame) else param.frame
    on = [k for k in (*_KEYS, "d", "t") if k in frame.columns]
    frame = _utf8(frame.select(*on, pl.col("value").cast(pl.Float64)), on).filter(pl.col("value") != default)
    arc_on = [k for k in _KEYS if k in on]
    rows = arcs.join(frame, on=arc_on, how="inner") if arc_on else arcs.join(frame, how="cross")
    step_on = [k for k in ("d", "t") if k in on]
    if len(step_on) < 2:
        rows = rows.join(dt, on=step_on, how="inner") if step_on else rows.join(dt, how="cross")
    else:
        rows = rows.join(dt, on=["d", "t"], how="semi")
    rows = rows.select(*_KEYS, "d", "t", "value")
    return rows if rows.height > 0 else None


def arc_capacity_factor(d: "FlexData") -> "pl.DataFrame | None":
    """``(p, source, sink, d, t, value)``: the capacity multiplier of each unit arc's ``maxFlow`` row
    (``core._max_flow_rhs``'s ``cap_mult``: v_flow ≤ value · capacity), rows where it is not 1 (a coarse
    process's: its coarse-step mean, :func:`_coarse_mean`)."""
    from flextool.engine.model.core import _max_flow_rhs

    arcs = _unit_arcs(d)
    if arcs is None or _dt(d) is None:
        return None
    _rhs, cap_mult, _upper, _bound = _max_flow_rhs(d)
    if cap_mult is None or cap_mult.height == 0:
        return None
    out = (_utf8(cap_mult.select(*_KEYS, "d", "t", pl.col("value").cast(pl.Float64)), [*_KEYS, "d", "t"])
           .join(arcs, on=_KEYS, how="semi"))
    out = _coarse_mean(out, d, _KEYS, default=1.0).filter(pl.col("value") != 1.0)
    return out if out.height > 0 else None


def arc_floor_share(d: "FlexData") -> "pl.DataFrame | None":
    """``(p, source, sink, d, t, value)``: the floor per unit online of each unit arc
    (``_unit_floors.arc_floor_share``), rows where it is > 0 (a coarse process's: its coarse-step mean,
    :func:`_coarse_mean`)."""
    from flextool.engine.model import _unit_floors

    arcs = _unit_arcs(d)
    if arcs is None:
        return None
    share = _unit_floors.arc_floor_share(d, arcs)
    if share is None or share.height == 0:
        return None
    out = _utf8(share.select(*_KEYS, "d", "t", pl.col("value").cast(pl.Float64)), [*_KEYS, "d", "t"])
    out = _coarse_mean(out, d, _KEYS, default=0.0).filter(pl.col("value") != 0.0)
    return out if out.height > 0 else None


def _coarse_mean(frame: pl.DataFrame, d: "FlexData", keys: list[str], *, default: float) -> pl.DataFrame:
    """``frame`` (``*keys, d, t, value``, Utf8 keys; ``keys`` start with ``p``; rows where the value is not
    ``default``, no row = ``default``) with the rows of each coarse process replaced by their Δt-weighted mean over
    its coarse step (``FlexData.process_coarse_step`` ``w`` = Δt/Δ_b), written to every fine step of the coarse
    step.  The mean counts the fine steps without a row at ``default``."""
    steps = getattr(d, "process_coarse_step", None)
    if not _has_rows(steps) or frame.height == 0:
        return frame
    steps = _utf8(steps.select("p", "d", "b_first", "t", pl.col("w").cast(pl.Float64)), ["p", "d", "b_first", "t"])
    coarse_keys = frame.join(steps.select("p").unique(), on="p", how="semi").select(keys).unique(maintain_order=True)
    if coarse_keys.height == 0:
        return frame
    # Every fine step of every coarse step of the coarse keys, the missing ones at the default.
    dense = (coarse_keys.join(steps, on="p", how="inner")
             .join(frame, on=[*keys, "d", "t"], how="left")
             .with_columns(pl.col("value").fill_null(default)))
    means = (dense.group_by(*keys, "d", "b_first")
             .agg((pl.col("value") * pl.col("w")).sum().alias("__mean")))
    coarse = (dense.select(*keys, "d", "b_first", "t")
              .join(means, on=[*keys, "d", "b_first"], how="left")
              .select(*keys, "d", "t", pl.col("__mean").alias("value")))
    fine = frame.join(coarse_keys, on=keys, how="anti")
    return pl.concat([fine, coarse])


def ramp_limits(d: "FlexData") -> "tuple[pl.DataFrame | None, pl.DataFrame | None]":
    """``(speed, c)`` of the arcs of this solve's ramp-limit sets (``limit`` ∈ :data:`RAMP_LIMITS` labels):
    ``speed`` ``(p, source, sink, limit, value)``, the ramp speed (share of capacity per minute; a set has no time
    axis), and ``c`` ``(p, source, sink, limit, d, t, value)``, the ramp row's capacity factor
    (``core._c_param_plain``) at the rows where it is not 1 (a coarse process's: its coarse-step mean,
    :func:`_coarse_mean`).  ``(None, None)`` without ramp limits."""
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
        speeds.append(rows.select(*_KEYS, pl.lit(label).alias("limit"), "value"))
        c = _on_arcs_dt(_c_param_plain(d, idx_set, side), rows.select(_KEYS), dt, default=1.0)
        if c is not None:
            coefs.append(c.select(*_KEYS, pl.lit(label).alias("limit"), "d", "t", "value"))
    if not speeds:
        return None, None
    if not coefs:
        return pl.concat(speeds), None
    coef = _coarse_mean(pl.concat(coefs), d, [*_KEYS, "limit"], default=1.0).filter(pl.col("value") != 1.0)
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
