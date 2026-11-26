"""Loaders of the ramp limits and the online (unit commitment) feature, and the carried obligations that
yield to outages."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING
import polars as pl
from polar_high import Param

if TYPE_CHECKING:
    from flextool.engine.input._input_source import InputSource

from flextool.engine.input._axis_enums import rename_to_axis
from flextool.engine.preprocess.flex_data._readers import (
    _active_solve_name,
    _provider_has,
    _provider_read,
    _split_period_branch,
)
from flextool.engine.preprocess.flex_data._container import FlexData


def _read_p_process_side(path: Path, side_col: str,
                          *,
                          provider: "object | None" = None) -> dict[str, pl.DataFrame]:
    """Parse ``input/p_process_sink.csv`` or ``input/p_process_source.csv``.

    Canonical (Python-preprocessing-input) format: long, columns
    ``[process, sink_or_source, sourceSinkParam, p_process_sink_or_source]``.
    The .mod also printf's a wide debug-export to ``solve_data/`` with
    a 2-row hierarchical header (process row, side row, then
    param/value rows) — supported as a fallback.  Returns
    ``{param_name: DataFrame(p, side, value)}``."""
    out: dict[str, pl.DataFrame] = {}
    _p = Path(path)
    name = f"{_p.parent.name}/{_p.stem}" if _p.parent.name else _p.stem
    if not _provider_has(provider, name, path):
        return out
    df = _provider_read(provider, name, path)
    # canonical long: ``process, <side>, sourceSinkParam, p_process_<side>``
    if {"process", "sourceSinkParam"}.issubset(df.columns):
        if df.height == 0:
            return out
        # value column is last; rename for uniformity
        value_col = df.columns[-1]
        out_df = (df.filter(pl.col(value_col) != 0)
                    .pipe(rename_to_axis, {"process": "p",
                             "sourceSinkParam": "param",
                             value_col: "value"}))
        for param, sub in out_df.group_by("param", maintain_order=True):
            param_str = param[0] if isinstance(param, tuple) else param
            out[param_str] = sub.select("p", side_col, "value")
        return out
    # The legacy 2-row-header printf-export format used to fall through
    # here; Step 2.5 removed the disk arm and the Provider only carries
    # the canonical long ``(process, <side>, sourceSinkParam, value)``
    # shape, so any other layout is a contract violation — surface it
    # rather than silently parsing a wide CSV that shouldn't exist.
    raise ValueError(
        f"_read_p_process_side: Provider key '{name}' returned a frame "
        f"with unexpected columns {df.columns!r}; expected canonical "
        f"long format with 'process'+'sourceSinkParam'."
    )


def _read_ramp_link(sd: Path, *, provider: "object | None" = None) -> "pl.DataFrame | None":
    """``dt_ramp_link`` from ``solve_data/step_previous`` (its ``jump`` column) and
    ``solve_data/rp_block_first``; ``None`` without ``step_previous``.  ``apply_derived_e`` replaces it with
    the same frame built from the source timeline."""
    from flextool.engine.preprocess.params._ramp_links import ramp_link_from_jump

    name = "solve_data/step_previous"
    path = sd / "step_previous.csv"
    if not _provider_has(provider, name, path):
        return None
    df = _provider_read(provider, name, path)
    if "jump" not in df.columns:
        return None
    steps = df.rename({"period": "d", "time": "t", "previous_period": "d_previous",
                       "previous_within_solve": "t_previous_within_solve"})
    return ramp_link_from_jump(steps, _read_rp_block_first_utf8(sd, provider=provider))


def _read_rp_block_first_utf8(sd: Path, *, provider: "object | None" = None) -> "pl.DataFrame | None":
    """``solve_data/rp_block_first`` as Utf8 ``(d, t)``; ``None`` when absent or empty."""
    from flextool.engine.preprocess import _provider_keys as K

    path = sd / "rp_block_first.csv"
    if not _provider_has(provider, K.SOLVE_DATA_RP_BLOCK_FIRST, path):
        return None
    df = _provider_read(provider, K.SOLVE_DATA_RP_BLOCK_FIRST, path)
    if df.height == 0:
        return None
    return df.select(pl.col("period").cast(pl.Utf8).alias("d"), pl.col("step").cast(pl.Utf8).alias("t"))


def _load_ramp_carry(sd: Path, dt: "pl.DataFrame | None", *,
                     provider: "object | None" = None
                     ) -> "tuple[pl.DataFrame | None, Param | None, Param | None]":
    """``(dt_ramp_carry, p_roll_continue_flow, p_roll_continue_flow_duration)`` of a continuation roll
    (``params._ramp_links.ramp_roll_carry``; the duration Param is each arc's carried step length)
    from the previous roll's ``handoff/roll_end_step`` and ``handoff/roll_end_flow``, the complete solve's
    ``solve_data/roll_first_step_link``, ``solve_data/period__branch`` and ``solve_data/rp_block_first``;
    ``(None, None)`` otherwise.  ``dt_ramp_carry`` exists also without ramp limits (the ramp output measures a
    roll's first step from it); ``p_roll_continue_flow`` only with ramp-limited arcs."""
    from flextool.engine.preprocess import _provider_keys as K
    from flextool.engine.preprocess._provider_translators import read_handoff_frame
    from flextool.engine.preprocess.params._ramp_links import ramp_roll_carry

    step = read_handoff_frame(provider, K.HANDOFF_ROLL_END_STEP)
    if step is None or step.height == 0:
        return None, None, None
    link_path = sd / "roll_first_step_link.csv"
    if not _provider_has(provider, K.SOLVE_DATA_ROLL_FIRST_STEP_LINK, link_path):
        return None, None, None
    link = _provider_read(provider, K.SOLVE_DATA_ROLL_FIRST_STEP_LINK, link_path)
    pb = None
    pb_path = sd / "period__branch.csv"
    if _provider_has(provider, "solve_data/period__branch", pb_path):
        pb = _provider_read(provider, "solve_data/period__branch", pb_path)
    flows = read_handoff_frame(provider, K.HANDOFF_ROLL_END_FLOW)
    rows, flows_out = ramp_roll_carry(dt, step, link, pb, flows=flows,
                                      rp_block_first=_read_rp_block_first_utf8(sd, provider=provider))
    if rows is None:
        return None, None, None
    if flows_out is None:
        return rows, None, None
    flows_out = flows_out.pipe(rename_to_axis, {"p": "p", "source": "source", "sink": "sink", "d": "d", "t": "t"})
    keys = ("p", "source", "sink", "d", "t")
    return (rows, Param(keys, flows_out.select(*keys, "value")),
            Param(keys, flows_out.select(*keys, pl.col("duration").alias("value"))))


def _load_ramp(inp: Path, sd: Path, pss: pl.DataFrame | None,
                *,
                dt: "pl.DataFrame | None" = None,
                provider: "object | None" = None) -> dict:
    """Load ramp-limit sets and ramp_speed params, the ramp predecessor links (``dt_ramp_link``) and, on a
    continuation roll, the previous roll's flow (``dt_ramp_carry``, ``p_roll_continue_flow``).  The sets are empty
    when no process_source_sink_ramp_limit_* row is populated; the carried flow is kept also then (an arc with
    only a ramp cost ramps from it, ``ramp_change_eq``).  ``dt_online_wrap`` is set with ``dt_ramp_link``
    (``params._ramp_links.ramp_link_fields``)."""
    from flextool.engine.preprocess.params._ramp_links import ramp_link_fields

    blank = dict(
        process_source_sink_ramp_limit_sink_up=None,
        process_source_sink_ramp_limit_sink_down=None,
        process_source_sink_ramp_limit_source_up=None,
        process_source_sink_ramp_limit_source_down=None,
        p_ramp_speed_up_sink=None,
        p_ramp_speed_down_sink=None,
        p_ramp_speed_up_source=None,
        p_ramp_speed_down_source=None,
        **ramp_link_fields(dt, _read_ramp_link(sd, provider=provider), _split_period_branch(sd, provider),
                           _read_rp_block_first_utf8(sd, provider=provider), solve=_active_solve_name(sd, provider)),
        dt_ramp_carry=None,
        p_roll_continue_flow=None,
        p_roll_continue_flow_duration=None,
    )
    # One read of the carry (pure on the Provider): dt_ramp_carry is needed also without ramp limits (the ramp
    # output), p_roll_continue_flow by the ramp limits and the ramp cost (the previous roll wrote roll_end_flow
    # only for arcs with either, _roll_end_flow).
    dt_ramp_carry, p_roll_continue_flow, p_roll_continue_flow_duration = _load_ramp_carry(sd, dt, provider=provider)
    blank["dt_ramp_carry"] = dt_ramp_carry
    blank["p_roll_continue_flow"] = p_roll_continue_flow
    blank["p_roll_continue_flow_duration"] = p_roll_continue_flow_duration
    if pss is None:
        return blank

    def _read_set(name: str) -> pl.DataFrame | None:
        p = sd / f"process_source_sink_ramp_limit_{name}.csv"
        key = f"solve_data/process_source_sink_ramp_limit_{name}"
        if not _provider_has(provider, key, p):
            return None
        df = _provider_read(provider, key, p)
        if df.height == 0:
            return None
        return df.pipe(rename_to_axis, {"process": "p"}).select("p", "source", "sink")

    sets = {f"process_source_sink_ramp_limit_{name}": _read_set(name)
            for name in ("sink_up", "sink_down", "source_up", "source_down")}
    if not any(s is not None for s in sets.values()):
        return blank

    # Δ.12-drop: ``p_ramp_speed_*_{sink,source}`` Params produced
    # authoritatively by ``apply_direct_params`` via
    # ``p_ramp_speed_up_sink_from_source`` etc.  Seeds dropped (the
    # ``_read_p_process_side`` reads of ``p_process_sink.csv`` /
    # ``p_process_source.csv`` are removed).  The four ramp-limit set
    # frames above remain CSV-loaded — apply_projection_params doesn't
    # cover those set partitions today.
    return dict(
        **sets,
        p_ramp_speed_up_sink   = None,
        p_ramp_speed_down_sink = None,
        p_ramp_speed_up_source = None,
        p_ramp_speed_down_source = None,
        dt_ramp_link = blank["dt_ramp_link"],
        dt_online_wrap = blank["dt_online_wrap"],
        dt_ramp_carry = dt_ramp_carry,
        p_roll_continue_flow = p_roll_continue_flow,
        p_roll_continue_flow_duration = p_roll_continue_flow_duration,
    )


def _read_process_exact(inp: Path, *, provider: "object | None" = None) -> "pl.DataFrame | None":
    """``process_exact`` (p,): the processes whose internal method is ``method_2way_2var_MIP_exclude`` (the
    ``exact`` transfer method, group overrides applied) in ``input/process_method``.  None when there is
    none."""
    path = inp / "process_method.csv"
    if not _provider_has(provider, "input/process_method", path):
        return None
    pm = _provider_read(provider, "input/process_method", path)
    ex = (pm.filter(pl.col("method").cast(pl.Utf8) == "method_2way_2var_MIP_exclude")
          .select(pl.col("process").cast(pl.Utf8)).unique(maintain_order=True)
          .pipe(rename_to_axis, {"process": "p"}))
    return ex if ex.height > 0 else None


def _load_online(inp: Path, sd: Path, dt: pl.DataFrame,
                  pss: pl.DataFrame | None,
                  *, source: "InputSource | None" = None,
                  provider: "object | None" = None) -> dict:
    """Load online / min_load / startup data.  Empty dict-of-Nones when
    no process is online."""
    blank = dict(
        process_online=None, process_online_linear=None,
        process_online_integer=None, process_minload=None,
        process_min_load_eff=None,
        p_online_dt=None, pdt_online_linear=None, pdt_online_integer=None,
        p_min_load=None, p_startup_cost=None, p_section=None,
        pdt_uptime_set=None, pdt_downtime_set=None,
        uptime_lookback=None, downtime_lookback=None,
        p_roll_continue_online=None,
        dt_online_carry=None,
        p_fix_online=None,
        online_startup_history=None, online_min_time=None,
        uptime_carry=None, downtime_carry=None,
    )
    if pss is None:
        return blank
    online_path = sd / "process_online.csv"
    if not _provider_has(provider, "solve_data/process_online", online_path):
        return blank
    p_online = _provider_read(provider, "solve_data/process_online", online_path).pipe(rename_to_axis, {"process": "p"})
    if p_online.height == 0:
        return blank

    p_online_lin = _provider_read(
        provider, "solve_data/process_online_linear",
        sd / "process_online_linear.csv").pipe(rename_to_axis, {"process": "p"})
    p_online_int_path = sd / "process_online_integer.csv"
    p_online_int = (_provider_read(
                        provider, "solve_data/process_online_integer",
                        p_online_int_path).pipe(rename_to_axis, {"process": "p"})
                    if _provider_has(provider,
                                       "solve_data/process_online_integer",
                                       p_online_int_path) else
                    pl.DataFrame(schema={"p": pl.Utf8}))

    # ct_method: min_load_efficiency rows.  Canonical input file is
    # ``input/process__ct_method.csv`` with columns (process, ct_method);
    # the .mod also printf's a debug-export to ``solve_data/`` with
    # column ``method`` — tolerate either schema/location.
    ctm_path = inp / "process__ct_method.csv"
    ctm_key = "input/process__ct_method"
    if not _provider_has(provider, ctm_key, ctm_path):
        ctm_path = sd / "process__ct_method.csv"
        ctm_key = "solve_data/process__ct_method"
    p_min_load_eff = pl.DataFrame(schema={"p": pl.Utf8})
    if _provider_has(provider, ctm_key, ctm_path):
        ctm = _provider_read(provider, ctm_key, ctm_path).pipe(rename_to_axis, {"process": "p"})
        method_col = "ct_method" if "ct_method" in ctm.columns else "method"
        p_min_load_eff = (ctm.filter(pl.col(method_col) == "min_load_efficiency")
                          .select("p").unique())

    # p_online_dt — block-aware variable indexing (process, period, step)
    p_odt = _provider_read(provider, "solve_data/p_online_dt_set",
                              sd / "p_online_dt_set.csv").pipe(rename_to_axis, {"process": "p", "step": "t"})
    p_odt = p_odt.select("p", "period", "t").pipe(rename_to_axis, {"period": "d"})

    # Δ.12-drop: ``p_min_load`` produced authoritatively as a shape-correct
    # (p, d, t) Param by ``_derived.apply_derived_b`` (via
    # ``p_min_load_pdt_from_source``); the synthetic-solve early return
    # in ``_apply_db_overrides`` re-wires it.  CSV seed dropped.  See
    # ``specs/fix_map_reading.md`` §5.
    p_min_load = None

    # startup_cost is per (p, d) — produced by ``apply_direct_params``
    # via ``p_startup_cost_from_source`` with full scalar / 1d_map(period)
    # broadcast cascade.  Δ.17b Gap C: local seed dropped.
    # However the LOCAL pdt_online_lin / pdt_online_int sets DEPEND on
    # the same (p, d) keying.  We reconstruct sc_long from the override
    # if it has fired by reading flex_data.p_startup_cost — but
    # flex_data isn't available at this scope.  Defensive: rebuild
    # pdt_online_lin / pdt_online_int from the source-side helper too.
    p_startup_cost = None
    pdt_online_lin = pdt_online_int = None
    if source is not None:
        from flextool.engine.preprocess.params._direct import p_startup_cost_from_source
        sc_param = p_startup_cost_from_source(source, period_filter=dt)
        if sc_param is not None and sc_param.frame.height > 0:
            sc_frame = sc_param.frame.filter(pl.col("value") != 0)
            if sc_frame.height > 0:
                p_startup_cost = Param(("p", "d"),
                                          sc_frame.select("p", "d", "value"))
                sc_p = sc_frame.select("p", "d").unique()
                pdt_online_lin = (p_odt.join(p_online_lin, on="p", how="inner")
                                       .join(sc_p, on=["p", "d"], how="inner"))
                if p_online_int.height > 0:
                    pdt_online_int = (p_odt.join(p_online_int, on="p", how="inner")
                                            .join(sc_p, on=["p", "d"], how="inner"))

    # Δ.12-drop: ``p_section`` produced authoritatively by
    # ``apply_derived_c.p_section_from_source`` when dt and classified
    # processes are non-empty.  Seed dropped.
    p_section = None

    # Δ.12-drop: ``pdt_uptime_set`` / ``pdt_downtime_set`` /
    # ``uptime_lookback`` / ``downtime_lookback`` produced authoritatively
    # by ``apply_derived_c`` (helpers ``uptime_lookback_from_source`` /
    # ``downtime_lookback_from_source`` + ``pdt_uptime_set_from_lookback``
    # / ``pdt_downtime_set_from_lookback``).  Seeds dropped.

    # OC1 — rolling online-commitment carry.  Read the committed-jump-end
    # ``v_online`` handed off by the prior roll from the canonical handoff
    # Provider key (``handoff/roll_end_online``, schema ``[process,
    # value]``).  The translator seeds this key from
    # ``SolveHandoff.roll_end_online`` at iteration start; it is empty
    # (→ ``None`` here) on the first roll / non-rolling / no-prior-carrier
    # solves, so the online dynamics keep their byte-identical baseline
    # path.  Mirrors the storage ``p_roll_continue_state`` overlay
    # (:func:`_load_storage`) but lives here because online can exist
    # without storage (``_load_storage`` short-circuits when there is no
    # ``nodeState``, so it would never read this carrier for a UC-only
    # fixture).  The gate that makes the carry fire is carry-PRESENCE
    # (``p_roll_continue_online is not None``), NOT ``not is_solve_first``:
    # on the standalone rolling-window path ``p_nested_solve_first`` is
    # ``None`` every roll, so the storage gate's ``not is_solve_first``
    # clause is inert there and carry-presence is the operative signal (it
    # is exactly co-extensive with "continuation roll" — empty on roll 1).
    from flextool.engine.preprocess import _provider_keys as K
    from flextool.engine.preprocess._provider_translators import read_handoff_frame
    p_roll_continue_online = None
    df_rco = read_handoff_frame(provider, K.HANDOFF_ROLL_END_ONLINE)
    if df_rco is not None and df_rco.height > 0:
        df_rco = (df_rco
                  .pipe(rename_to_axis, {"process": "p"})
                  .with_columns(value=pl.col("value")
                                          .cast(pl.Float64, strict=False)
                                          .fill_null(0.0))
                  .select("p", "value"))
        if df_rco.height > 0:
            p_roll_continue_online = Param(("p",), df_rco)

    # The step(s) of THIS roll that take the carry (``dt_online_carry``) are the ramp's seam roots
    # (``dt_ramp_carry``, set by :func:`load_flextool` once the ramp carry is read): the commitment continues the
    # previous solve only where the timeline continues, and stops at a gap, a representative-period start and a
    # new year, as ramp limits and the minimum up / down look-back do.  The roots are keyed on the carried step,
    # so a carried commitment without it cannot be placed: raise instead of dropping it.
    dt_online_carry = None
    if p_roll_continue_online is not None:
        _rco_step = read_handoff_frame(provider, K.HANDOFF_ROLL_END_STEP)
        if _rco_step is None or _rco_step.height == 0:
            raise ValueError(
                "handoff/roll_end_online: the carried online commitment needs the step it was realized at "
                "(handoff/roll_end_step), which is missing; the commitment cannot be placed on this solve's "
                "first steps.")

    # OC1 #1 sub-slice (b) — read the parent's committed v_online over the
    # jump (``handoff/fix_online``, schema ``[process, period, step,
    # value]``), seeded by the separate_dispatch parent's handoff producer.
    # Empty (→ ``None``) on every solve that is not a realized-dispatch
    # sibling carrying a pin → the online block keeps its baseline path.
    p_fix_online = None
    df_fxo = read_handoff_frame(provider, K.HANDOFF_FIX_ONLINE)
    if df_fxo is not None and df_fxo.height > 0:
        df_fxo = (df_fxo
                  .pipe(rename_to_axis,
                        {"process": "p", "period": "d", "step": "t"})
                  .with_columns(
                      pl.col("p").cast(pl.Utf8, strict=False),
                      pl.col("d").cast(pl.Utf8, strict=False),
                      pl.col("t").cast(pl.Utf8, strict=False),
                      value=pl.col("value").cast(pl.Float64, strict=False)
                                           .fill_null(0.0))
                  .select("p", "d", "t", "value"))
        if df_fxo.height > 0:
            p_fix_online = Param(("p", "d", "t"), df_fxo)

    # The realized on-line history of the solve this one continues from (``handoff/online_history``, routed with
    # the carried step by ``solve._carry_routing``): each step's start-ups / shut-downs with its ``hours_to_end``
    # (hours from its start to the end of that solve's last realized step).  It feeds ``uptime_carry`` /
    # ``downtime_carry`` (``_derived.apply_mintime_lookbacks``, once ``dt`` and the ramp links are in place) and is
    # extended into this solve's own history in its handoff.  ``None`` on the first roll / non-rolling /
    # no-min-up-down solves.
    online_startup_history = None
    df_hist = read_handoff_frame(provider, K.HANDOFF_ONLINE_HISTORY)
    if df_hist is not None and df_hist.height > 0:
        if "hours_to_end" not in df_hist.columns:
            raise ValueError(
                "handoff/online_history: the realized on-line history needs an 'hours_to_end' column (hours from "
                "the start of each step to the end of the last realized step); got columns "
                f"{df_hist.columns}")
        df_hist = (df_hist
                   .pipe(rename_to_axis,
                         {"process": "p", "period": "d", "step": "t"})
                   .with_columns(
                       pl.col("p").cast(pl.Utf8, strict=False),
                       pl.col("d").cast(pl.Utf8, strict=False),
                       pl.col("t").cast(pl.Utf8, strict=False),
                       pl.col("startup").cast(pl.Float64, strict=False)
                                        .fill_null(0.0),
                       pl.col("shutdown").cast(pl.Float64, strict=False)
                                         .fill_null(0.0),
                       pl.col("duration").cast(pl.Float64, strict=False)
                                         .fill_null(1.0),
                       pl.col("hours_to_end").cast(pl.Float64, strict=False))
                   .select("p", "d", "t", "startup", "shutdown", "duration", "hours_to_end"))
        if df_hist["hours_to_end"].null_count() > 0:
            raise ValueError("handoff/online_history: 'hours_to_end' must be a number on every row")
        if df_hist.height > 0:
            online_startup_history = df_hist

    # Authored max(min_uptime, min_downtime) per online process (hours) —
    # used only to trim ``online_history`` in the next handoff.  Solve-
    # agnostic; ``None`` when no process authors min-up/down.
    online_min_time = None
    if source is not None:
        from flextool.engine.preprocess.params import _derived as _drv_mt
        _mt: dict[str, float] = {}
        for _pn in ("min_uptime", "min_downtime"):
            for _p, _v in _drv_mt._process_min_time(source, _pn):
                if _v > _mt.get(_p, 0.0):
                    _mt[_p] = _v
        if _mt:
            online_min_time = pl.DataFrame(
                {"p": list(_mt.keys()),
                 "value": [float(v) for v in _mt.values()]})

    return dict(
        process_online=p_online,
        process_online_linear=p_online_lin,
        process_online_integer=p_online_int,
        process_minload=None,  # load_flextool sets it from the solve's p_min_load
        process_min_load_eff=p_min_load_eff,
        p_online_dt=p_odt,
        pdt_online_linear=pdt_online_lin,
        pdt_online_integer=pdt_online_int,
        p_min_load=p_min_load,
        p_startup_cost=p_startup_cost,
        p_section=p_section,
        pdt_uptime_set=None,
        pdt_downtime_set=None,
        uptime_lookback=None,
        downtime_lookback=None,
        p_roll_continue_online=p_roll_continue_online,
        dt_online_carry=dt_online_carry,
        p_fix_online=p_fix_online,
        online_startup_history=online_startup_history,
        online_min_time=online_min_time,
        uptime_carry=None,
        downtime_carry=None,
    )


def carried_obligations_yield_to_outages(flex_data: "FlexData", solve: str, unit_level: str = "none") -> None:
    """Obligations carried in from an earlier solve yield to an outage that solve did not see (W-0592), on the
    solve's data after its ``reduce_unit_detail_method`` *unit_level* (a roll and a realization take their base /
    parent solve's level).  B = ``params._mintime.online_blocked_cells``: the steps where a unit cannot be online
    (its availability, the realized data, keeps it below its min_load).

    * (a1) the carried minimum up-time (``uptime_carry``) of a unit ends at the first step of B it reaches and does
      not resume (``_mintime.cut_at_blocked``); the carried minimum down-time never conflicts (offline is always
      possible).
    * (a2) a unit with a realization pin (``p_fix_online``) > 0 on B keeps its pins before the first such step and
      loses them there and at every later step it reaches along the timeline (``dt_ramp_link``; a later period that
      does not continue this one keeps its pins): it is decided again under its own rows from the outage on (no
      later pin, so no minimum down-time conflicts with one).  A start-up of the kept pins whose minimum up-time runs into
      that step has it cut there: the ``uptime_lookback`` terms of the rows at and after the step on start-ups at
      kept pin cells go (``pdt_uptime_set`` follows).  The other units keep their pins.

    Nothing yields under a level that leaves out min limits (``min_limits``: no floor, so no step blocks the unit).
    Only an LP that the carried obligation makes infeasible changes.  Called after the coarse process frames, which
    B reads (a coarse process is judged on its coarse-step means)."""
    from flextool.engine.preprocess._reduce_detail import unit_parts
    from flextool.engine.preprocess.params._mintime import (
        cut_at_blocked, note_carried_obligation_cut, online_blocked_cells)
    pin = flex_data.p_fix_online
    carry = flex_data.uptime_carry
    has_pin = pin is not None and pin.frame.height > 0
    if ("min_limits" in unit_parts(unit_level) or not (has_pin or (carry is not None and carry.height > 0))):
        return
    blocked = online_blocked_cells(flex_data)
    if blocked is None:
        return
    timeline = (flex_data.dt, flex_data.p_step_duration, flex_data.dt_ramp_link)
    notes: dict[str, str] = {}
    if carry is not None and carry.height > 0:
        mt = flex_data.online_min_time
        cut: list = []
        flex_data.uptime_carry = cut_at_blocked(
            carry, blocked, *timeline, float(mt["value"].max()) if mt is not None and mt.height > 0 else float("inf"),
            cut)
        notes.update({p: f"solve {solve}: the carried minimum up-time ends at step ({d}, {t})"
                      for p, d, t in reversed(cut)})
    if has_pin:
        keys = ["p", "d", "t"]
        on = (pin.frame.filter(pl.col("value") > 0.0).select(*(pl.col(k).cast(pl.Utf8) for k in keys))
              .join(blocked, on=keys, how="semi"))
        cut, reached = [], []
        kept = cut_at_blocked(pin.frame, on, *timeline, float("inf"), cut, reached)
        if cut:
            flex_data.p_fix_online = Param(pin.dims, kept) if kept is not None else None
            _cut_pinned_uptime(flex_data, reached[0], kept)
            notes.update({p: f"solve {solve}: the units online pinned to the stochastic solve are decided again "
                             f"from step ({d}, {t}) on, where they cannot run" for p, d, t in cut})
    note_carried_obligation_cut(notes)


def _cut_pinned_uptime(flex_data: "FlexData", reached: pl.DataFrame, kept: "pl.DataFrame | None") -> None:
    """Drop the ``uptime_lookback`` terms of the rows at the *reached* cells ``(p, d, t)`` (from a unit's first
    blocked pinned step on) on start-ups at its *kept* pin cells, and project ``pdt_uptime_set`` again."""
    from flextool.engine.preprocess.params._derived import pdt_uptime_set_from_lookback
    lb = flex_data.uptime_lookback
    if lb is None or lb.height == 0 or kept is None:
        return
    utf8 = lb.with_columns(*(pl.col(k).cast(pl.Utf8).alias(f"__{k}") for k in ("p", "d", "t", "d_back", "t_back")))
    pinned = kept.select(pl.col("p").cast(pl.Utf8).alias("__p"), pl.col("d").cast(pl.Utf8).alias("__d_back"),
                         pl.col("t").cast(pl.Utf8).alias("__t_back"))
    voided = (utf8.join(reached.select(pl.col("p").alias("__p"), pl.col("d").alias("__d"), pl.col("t").alias("__t")),
                        on=["__p", "__d", "__t"], how="semi")
              .join(pinned, on=["__p", "__d_back", "__t_back"], how="semi"))
    if voided.height == 0:
        return
    rest = utf8.join(voided.select(c for c in voided.columns if c.startswith("__")),
                     on=[c for c in utf8.columns if c.startswith("__")], how="anti", maintain_order="left")
    rest = rest.select(lb.columns)
    flex_data.uptime_lookback = rest if rest.height > 0 else None
    flex_data.pdt_uptime_set = pdt_uptime_set_from_lookback(flex_data.uptime_lookback)
