"""§3.8.2 / §3.8.3 — minimum up / down time: the look-back sets, their projections, the roll carry and
:func:`apply_mintime_lookbacks`."""
from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl
from flextool.input_derivation._method_constants import MIN_TIME_ENABLES

if TYPE_CHECKING:
    from flextool.engine.input._input_source import InputSource

from flextool.engine.preprocess.params._derived._scope import _min_time_methods, _try_param


# ---------------------------------------------------------------------------
# §3.8.2 / §3.8.3 — pdt_uptime_set / pdt_downtime_set / lookbacks
# ---------------------------------------------------------------------------


def _process_min_time(source: "InputSource",
                        parameter_name: str) -> list[tuple[str, float]]:
    """``[(unit, min_time)]`` of ``unit.<parameter_name>`` (``min_uptime`` / ``min_downtime``, with its definition
    default): value > 0 and enabled by ``minimum_time_method`` (:data:`MIN_TIME_ENABLES`, read by
    :func:`_min_time_methods`; unset is ``none``, which enables neither), in source row order.  Every consumer of
    the minimum times goes through it: the look-backs and their rows, the roll carry, the cut-short warning and the
    history trim."""
    df = _try_param(source, "unit", parameter_name)
    if df is None:
        return []
    methods = _min_time_methods(source)
    enables = MIN_TIME_ENABLES[parameter_name]
    out: dict[str, float] = {}
    for r in df.iter_rows(named=True):
        try:
            v = float(r["value"])
        except (TypeError, ValueError):
            continue
        if v > 0 and methods.get(str(r["name"])) in enables:
            out[r["name"]] = v
    return list(out.items())


def uptime_lookback_from_source(source: "InputSource",
                                    dt: pl.DataFrame | None,
                                    p_step_duration: object | None = None,
                                    link: pl.DataFrame | None = None,
                                    ) -> pl.DataFrame | None:
    """Build the ``uptime_lookback`` set ``(p, d, t, d_back, t_back)``: each step and its ancestors along the
    ramp predecessor links *link* (``dt_ramp_link``) within ``min_uptime`` (``params._mintime.mintime_lookback``)."""
    from flextool.engine.preprocess.params._mintime import mintime_lookback
    return mintime_lookback(dt, p_step_duration, link, _process_min_time(source, "min_uptime"))


def downtime_lookback_from_source(source: "InputSource",
                                       dt: pl.DataFrame | None,
                                       p_step_duration: object | None = None,
                                       link: pl.DataFrame | None = None,
                                       ) -> pl.DataFrame | None:
    """Build the ``downtime_lookback`` set (as :func:`uptime_lookback_from_source`, with ``min_downtime``)."""
    from flextool.engine.preprocess.params._mintime import mintime_lookback
    return mintime_lookback(dt, p_step_duration, link, _process_min_time(source, "min_downtime"))


def _min_time_written_notice(source: "InputSource") -> None:
    """W-0634 (``minimum_time_method_written``): the units of this solve on which the database upgrade wrote
    ``minimum_time_method``, one line per write ``U (alternative A: method)`` (each unit once per run)."""
    from flextool.engine import _behaviour_changes as bc
    writes = bc.marker_json("minimum_time_method_written").get("writes") or []
    if not writes:
        return
    units = source.entities("unit")
    in_solve = set() if units.height == 0 else {str(u) for u in units.get_column(units.columns[0]).to_list()}
    lines = [f"{u} (alternative {alt}: {method})" for u, alt, method in writes if u in in_solve]
    if lines:
        bc.notice("minimum_time_method_written", "units", lines)


def apply_mintime_lookbacks(flex_data: object, source: "InputSource",
                            first_step_link: pl.DataFrame | None, solve_name: str) -> None:
    """Set ``uptime_lookback`` / ``downtime_lookback`` and the projected ``pdt_*_set`` of ``flex_data`` from its
    ``dt``, ``p_step_duration`` and ``dt_ramp_link``, the roll carries ``uptime_carry`` / ``downtime_carry``
    (:func:`online_mintime_carry_from_source`), and print a warning for each process whose look-back the
    solve boundary cuts short (``params._mintime.cut_short_lookback_warnings``; ``first_step_link`` is the
    complete solve's ``step_previous`` of this solve's period first steps, ``None`` when not written)."""
    from flextool.engine.preprocess.params._mintime import report_cut_short_lookback
    _min_time_written_notice(source)
    dt = getattr(flex_data, "dt", None)
    sd = getattr(flex_data, "p_step_duration", None)
    link = getattr(flex_data, "dt_ramp_link", None)
    ulb = uptime_lookback_from_source(source, dt, sd, link)
    if ulb is not None and ulb.height > 0:
        flex_data.uptime_lookback = ulb
        flex_data.pdt_uptime_set = pdt_uptime_set_from_lookback(ulb)
    dlb = downtime_lookback_from_source(source, dt, sd, link)
    if dlb is not None and dlb.height > 0:
        flex_data.downtime_lookback = dlb
        flex_data.pdt_downtime_set = pdt_downtime_set_from_lookback(dlb)
    if ulb is None and dlb is None:
        return
    flex_data.uptime_carry = (online_mintime_carry_from_source(source, flex_data, "min_uptime")
                              if ulb is not None else None)
    flex_data.downtime_carry = (online_mintime_carry_from_source(source, flex_data, "min_downtime")
                                if dlb is not None else None)
    min_times = {"min_uptime": _process_min_time(source, "min_uptime"),
                 "min_downtime": _process_min_time(source, "min_downtime")}
    report_cut_short_lookback(min_times, dt, first_step_link, getattr(flex_data, "dt_ramp_carry", None),
                              solve_name)


def pdt_uptime_set_from_lookback(uptime_lookback: pl.DataFrame | None
                                     ) -> pl.DataFrame | None:
    """Project ``(p, d, t, d_back, t_back) → (p, d, t)`` distinct."""
    if uptime_lookback is None or uptime_lookback.height == 0:
        return None
    return (uptime_lookback.lazy()
              .select("p", "d", "t").unique()
              .sort("p", "d", "t")
              .collect())


def pdt_downtime_set_from_lookback(downtime_lookback: pl.DataFrame | None
                                       ) -> pl.DataFrame | None:
    if downtime_lookback is None or downtime_lookback.height == 0:
        return None
    return (downtime_lookback.lazy()
              .select("p", "d", "t").unique()
              .sort("p", "d", "t")
              .collect())


def online_mintime_carry_from_source(source: "InputSource", flex_data: object, parameter_name: str
                                     ) -> pl.DataFrame | None:
    """The roll carry of the minimum up / down time ``(p, d, t, value)`` (Utf8) of *flex_data*.

    ``parameter_name`` ``"min_uptime"`` carries the realized ``v_startup`` (``uptime_carry``, on the
    ``minimum_uptime`` RHS), ``"min_downtime"`` the realized ``v_shutdown`` (``downtime_carry``, on the
    ``minimum_downtime`` LHS).  The previous solve's realized on-line history (``online_startup_history``, each step
    with its ``hours_to_end``) reaches the steps that follow the roots ``dt_ramp_carry`` along ``dt_ramp_link``
    (``params._mintime.mintime_roll_carry``): the first steps that continue from the previous solve's last realized
    step on the timeline — a period that continues the previous one and the forecast branches that start with such a
    step included, a period that does not continue it (``jump != 1``) and a step after a gap excluded.  ``L`` is the
    authored parameter.  ``None`` when nothing is carried.

    The up-time carry of a process is capped by its units online carried in (``p_roll_continue_online``, ĉ[p]:
    the previous solve's realized v_online at the carried step): carry_up ≤ ĉ.  Start-ups still within the minimum
    up-time cannot oblige more units online than were online at the carried step.  Without an outage the earlier
    solve's own ``minimum_uptime`` row at that step already holds ĉ ≥ the start-ups it covers (a superset of those
    any later step covers), so the cap binds only after an outage cut the obligation in an earlier solve
    (``flex_data.carried_obligations_yield_to_outages``, W-0592): the outage forced the units offline, ĉ counts only
    the start-ups after it, and the start-ups before it stop counting — the obligation does not resume.  The
    down-time carry is not capped (an outage never conflicts with it: offline is always possible)."""
    from flextool.engine.preprocess.params._mintime import mintime_roll_carry
    history = getattr(flex_data, "online_startup_history", None)
    if history is None or history.height == 0:
        return None
    carry = mintime_roll_carry(getattr(flex_data, "dt", None), getattr(flex_data, "p_step_duration", None),
                               getattr(flex_data, "dt_ramp_link", None), getattr(flex_data, "dt_ramp_carry", None),
                               history, _process_min_time(source, parameter_name),
                               "startup" if parameter_name == "min_uptime" else "shutdown",
                               getattr(flex_data, "period_branch_full", None))
    rco = getattr(flex_data, "p_roll_continue_online", None)
    if carry is None or parameter_name != "min_uptime" or rco is None:
        return carry
    online_in = rco.frame.select(pl.col("p").cast(pl.Utf8), pl.col("value").cast(pl.Float64).alias("__c"))
    carry = (carry.join(online_in, on="p", how="left", maintain_order="left")
             .with_columns(pl.when(pl.col("__c") < pl.col("value")).then(pl.col("__c"))
                           .otherwise(pl.col("value")).alias("value"))
             .filter(pl.col("value") != 0.0)
             .select(carry.columns))
    return carry if carry.height > 0 else None
