"""Shared minimal in-memory fixtures for the engine_polars constraints
audit (Surface B).

All fixtures construct ``FlexData`` directly in memory in the style of
``tests/engine_polars/fixtures/flex_toy_*.py`` — no workdir, no Spine
DB, no cascade.  Each toy is the smallest input that meaningfully
exercises one constraint family / feature switch.

Two sub-scopes:

* The bare data dict / FlexData is built fresh per call (function scope)
  so individual tests can mutate the returned data with
  ``dataclasses.replace`` without leaking across tests.
* The optional ``*_solved`` lookalikes (kept module-scoped within a
  single test file when needed) are not provided here — tests that
  need a solved instance build their own ``Problem`` from the fresh
  data fixture.

Convention
----------
* ``period`` axis values use ``"d1"``, ``"d2"`` … so tests can
  unambiguously pattern-match on period names.
* ``time`` axis values use ``"t01"`` … (zero-padded so lex order is
  numeric).
* ``solver_options()`` returns a HiGHS option dict that yields
  deterministic single-thread runs.

Self-tests at the bottom of this conftest validate every fixture
solves (or xfails with a reason).  Subsequent agents add focused
constraint-by-constraint tests on top of these fixtures.
"""
from __future__ import annotations

from typing import Any

import polars as pl
import pytest

from polar_high import Param, Problem
from flextool.engine_polars import build_flextool
from flextool.engine_polars.input import FlexData


# ---------------------------------------------------------------------------
# Solver options (deterministic single-thread HiGHS).

def solver_options() -> dict[str, Any]:
    """Return HiGHS option dict for deterministic runs.

    polar_high.Problem.solve(options=...) and set_solver_options pass the
    dict straight through to HiGHS, so the keys are HiGHS-native option
    names.  ``random_seed`` pins HiGHS's symmetry-breaking; ``parallel='off'``
    disables threading so tie-breaking is stable across machines.
    """
    return {"random_seed": 42, "parallel": "off"}


def _solve(data: FlexData) -> tuple[Problem, "Any"]:
    """Build + solve ``data`` with deterministic HiGHS options."""
    pb = Problem()
    build_flextool(pb, data)
    sol = pb.solve(options=solver_options())
    return pb, sol


# ---------------------------------------------------------------------------
# Time / node base helpers — every toy starts from these.

def _dt(periods: list[str], steps_per_period: int) -> pl.DataFrame:
    rows = []
    for d in periods:
        for k in range(1, steps_per_period + 1):
            rows.append({"d": d, "t": f"t{k:02d}"})
    return pl.DataFrame(rows)


def _const_param(dims: tuple[str, ...], dt: pl.DataFrame, value: float,
                  *, extra: dict[str, list[str]] | None = None) -> Param:
    """Build a Param of the given dims, broadcast ``value`` over dt × extra."""
    frame = dt
    if extra:
        for col, vals in extra.items():
            frame = pl.DataFrame({col: vals}).join(frame, how="cross")
    if "value" not in frame.columns:
        frame = frame.with_columns(value=pl.lit(value))
    return Param(dims, frame.select(*dims, "value"))


def _time_axes(periods: list[str], steps_per_period: int = 2):
    dt = _dt(periods, steps_per_period)
    p_step_duration = Param(("d", "t"), dt.with_columns(value=pl.lit(1.0)))
    p_rp_cost_weight = Param(("d", "t"), dt.with_columns(value=pl.lit(1.0)))
    p_inflation_op = Param(("d",),
        pl.DataFrame({"d": periods, "value": [1.0] * len(periods)}))
    p_period_share = Param(("d",),
        pl.DataFrame({"d": periods, "value": [1.0] * len(periods)}))
    return dt, p_step_duration, p_rp_cost_weight, p_inflation_op, p_period_share


def _node_axes(node: str, dt: pl.DataFrame,
                inflow: float, penalty: float = 1e6):
    nb = pl.DataFrame({"n": [node]})
    nb_dt = nb.join(dt, how="cross")
    p_inflow = Param(("n", "d", "t"),
        nb_dt.with_columns(value=pl.lit(inflow)).select("n", "d", "t", "value"))
    p_pen_up = Param(("n", "d", "t"),
        nb_dt.with_columns(value=pl.lit(penalty)).select("n", "d", "t", "value"))
    p_pen_dn = Param(("n", "d", "t"),
        nb_dt.with_columns(value=pl.lit(penalty)).select("n", "d", "t", "value"))
    return nb, nb_dt, p_inflow, p_pen_up, p_pen_dn


# ---------------------------------------------------------------------------
# Fixture 1 — toy_1n1p_1d2t : single node + single process, no UC, no
# storage, deterministic dispatch.  Inflow = 10 each step (positive
# inflow → process source-to-sink absorbs into the node).

@pytest.fixture(scope="function")
def toy_1n1p_1d2t() -> FlexData:
    dt, p_step, p_rp, p_infl, p_psh = _time_axes(["d1"], 2)
    nb, nb_dt, p_inflow, p_pen_up, p_pen_dn = _node_axes(
        "n", dt, inflow=-10.0, penalty=1e6)

    pss = pl.DataFrame({"p": ["p"], "source": ["source_n"], "sink": ["n"]})
    pss_eff = pss.clone()
    pss_noEff = pl.DataFrame(
        schema={"p": pl.Utf8, "source": pl.Utf8, "sink": pl.Utf8})
    pss_dt = pss.join(dt, how="cross")
    flow_to_n = pss.with_columns(n=pl.col("sink"))

    flow_from_commodity_eff = pl.DataFrame(
        {"p": ["p"], "source": ["source_n"], "sink": ["n"], "c": ["FUEL"]})
    flow_from_commodity_noEff = pl.DataFrame(
        schema={"p": pl.Utf8, "source": pl.Utf8, "sink": pl.Utf8, "c": pl.Utf8})

    p_unitsize = Param(("p",), pl.DataFrame({"p": ["p"], "value": [1.0]}))
    p_flow_upper = Param(
        ("p", "source", "sink", "d", "t"),
        pss_dt.with_columns(value=pl.lit(100.0))
              .select("p", "source", "sink", "d", "t", "value"))
    p_slope = Param(("p", "d", "t"),
        dt.with_columns(p=pl.lit("p"), value=pl.lit(1.0))
          .select("p", "d", "t", "value"))
    p_commodity_price = Param(("c", "d", "t"),
        dt.with_columns(c=pl.lit("FUEL"), value=pl.lit(1.0))
          .select("c", "d", "t", "value"))

    return FlexData(
        dt=dt, p_step_duration=p_step, p_rp_cost_weight=p_rp,
        p_inflation_op=p_infl, p_period_share=p_psh,
        nodeBalance=nb, nodeBalance_dt=nb_dt,
        p_inflow=p_inflow, p_penalty_up=p_pen_up, p_penalty_down=p_pen_dn,
        process_source_sink=pss,
        process_source_sink_eff=pss_eff,
        process_source_sink_noEff=pss_noEff,
        pss_dt=pss_dt, flow_to_n=flow_to_n,
        flow_from_commodity_eff=flow_from_commodity_eff,
        flow_from_commodity_noEff=flow_from_commodity_noEff,
        p_unitsize=p_unitsize, p_flow_upper=p_flow_upper,
        p_slope=p_slope, p_commodity_price=p_commodity_price,
    )


# ---------------------------------------------------------------------------
# Fixture 2 — toy_storage_2t : add a storage node ``s`` with nodeState.
# 1 period, 2 steps, no blocks.  No additional process — the storage
# node simply absorbs / releases whatever the slack chooses.  Used to
# probe the nodeBalance + nodeState_eq family in isolation.

@pytest.fixture(scope="function")
def toy_storage_2t() -> FlexData:
    dt, p_step, p_rp, p_infl, p_psh = _time_axes(["d1"], 2)
    # nodeBalance covers BOTH nodes (n is regular, s is the storage node).
    nb = pl.DataFrame({"n": ["n", "s"]})
    nb_dt = nb.join(dt, how="cross")
    p_inflow = Param(("n", "d", "t"), nb_dt.with_columns(
        value=pl.when(pl.col("n") == "n").then(pl.lit(-5.0)).otherwise(pl.lit(0.0))
    ).select("n", "d", "t", "value"))
    p_pen_up = Param(("n", "d", "t"),
        nb_dt.with_columns(value=pl.lit(1e6)).select("n", "d", "t", "value"))
    p_pen_dn = Param(("n", "d", "t"),
        nb_dt.with_columns(value=pl.lit(1e6)).select("n", "d", "t", "value"))

    # Storage node ``s`` — minimal nodeState set; no nodeStateBlock.
    nodeState = pl.DataFrame({"n": ["s"]})
    nodeState_dt = nodeState.join(dt, how="cross")
    # First (d, t) per (n, d) — for storage initial conditions.
    nodeState_first_dt = (
        nodeState_dt.sort(["n", "d", "t"])
                    .group_by(["n", "d"], maintain_order=True)
                    .first()
                    .select("n", "d", "t"))
    nodeState_last_dt = (
        nodeState_dt.sort(["n", "d", "t"])
                    .group_by(["n", "d"], maintain_order=True)
                    .last()
                    .select("n", "d", "t"))
    p_state_unitsize = Param(("n",),
        pl.DataFrame({"n": ["s"], "value": [10.0]}))
    p_state_upper = Param(("n", "d"),
        pl.DataFrame({"n": ["s"], "d": ["d1"], "value": [1.0]}))  # cap/unitsize=1
    p_state_self_discharge = Param(("n",),
        pl.DataFrame({"n": ["s"], "value": [0.0]}))
    p_state_existing_capacity = Param(("n", "d"),
        pl.DataFrame({"n": ["s"], "d": ["d1"], "value": [10.0]}))

    # dtttdt: cyclic step_previous over 2 timesteps within d1.
    dtttdt = pl.DataFrame({
        "d": ["d1", "d1"],
        "t": ["t01", "t02"],
        "t_previous": ["t02", "t01"],
        "t_previous_within_timeset": ["t02", "t01"],
        "d_previous": ["d1", "d1"],
        "t_previous_within_solve": ["t02", "t01"],
    })

    return FlexData(
        dt=dt, p_step_duration=p_step, p_rp_cost_weight=p_rp,
        p_inflation_op=p_infl, p_period_share=p_psh,
        nodeBalance=nb, nodeBalance_dt=nb_dt,
        p_inflow=p_inflow, p_penalty_up=p_pen_up, p_penalty_down=p_pen_dn,
        nodeState=nodeState, nodeState_dt=nodeState_dt,
        nodeState_first_dt=nodeState_first_dt,
        nodeState_last_dt=nodeState_last_dt,
        p_state_unitsize=p_state_unitsize,
        p_state_upper=p_state_upper,
        p_state_self_discharge=p_state_self_discharge,
        p_state_existing_capacity=p_state_existing_capacity,
        dtttdt=dtttdt,
    )


# ---------------------------------------------------------------------------
# Fixture 3 — toy_storage_blocks : 1 period × 2 blocks × 2 fine steps each.
# Storage node ``s`` with bind_intraperiod_blocks; activates
# ``stateConstantWithinBlock_eq`` and ``nodeBalanceBlock_eq``.

@pytest.fixture(scope="function")
def toy_storage_blocks() -> FlexData:
    # 4 fine steps in a single period, partitioned into 2 coarse blocks.
    # Block boundaries: b1=t01..t02, b2=t03..t04.  b_first per block is
    # the FIRST fine step of the block.
    dt, p_step, p_rp, p_infl, p_psh = _time_axes(["d1"], 4)
    nb = pl.DataFrame({"n": ["s"]})  # storage node only
    nb_dt = nb.join(dt, how="cross")
    p_inflow = Param(("n", "d", "t"),
        nb_dt.with_columns(value=pl.lit(0.0)).select("n", "d", "t", "value"))
    p_pen_up = Param(("n", "d", "t"),
        nb_dt.with_columns(value=pl.lit(1e6)).select("n", "d", "t", "value"))
    p_pen_dn = Param(("n", "d", "t"),
        nb_dt.with_columns(value=pl.lit(1e6)).select("n", "d", "t", "value"))

    nodeState = pl.DataFrame({"n": ["s"]})
    nodeState_dt = nodeState.join(dt, how="cross")
    nodeState_first_dt = (nodeState_dt.sort(["n", "d", "t"])
        .group_by(["n", "d"], maintain_order=True).first()
        .select("n", "d", "t"))
    nodeState_last_dt = (nodeState_dt.sort(["n", "d", "t"])
        .group_by(["n", "d"], maintain_order=True).last()
        .select("n", "d", "t"))
    p_state_unitsize = Param(("n",),
        pl.DataFrame({"n": ["s"], "value": [10.0]}))
    p_state_upper = Param(("n", "d"),
        pl.DataFrame({"n": ["s"], "d": ["d1"], "value": [1.0]}))
    p_state_self_discharge = Param(("n",),
        pl.DataFrame({"n": ["s"], "value": [0.0]}))
    p_state_existing_capacity = Param(("n", "d"),
        pl.DataFrame({"n": ["s"], "d": ["d1"], "value": [10.0]}))

    # dtttdt: cyclic step_previous over 4 timesteps within d1.  Used by
    # bind_within_timeset constraints when activated; here we only need
    # the within-block-interior subset for stateConstantWithinBlock.
    dtttdt = pl.DataFrame({
        "d": ["d1"]*4,
        "t": ["t01", "t02", "t03", "t04"],
        "t_previous": ["t04", "t01", "t02", "t03"],
        "t_previous_within_timeset": ["t04", "t01", "t02", "t03"],
        "d_previous": ["d1"]*4,
        "t_previous_within_solve": ["t04", "t01", "t02", "t03"],
    })

    # Block frames — 2 coarse blocks per period.
    nodeStateBlock = pl.DataFrame({"n": ["s"]})
    period_block = pl.DataFrame({"d": ["d1", "d1"], "b_first": ["t01", "t03"]})
    # Cyclic succ: t01 → t03, t03 → t01.
    period_block_succ = pl.DataFrame({
        "d": ["d1", "d1"],
        "b_first": ["t01", "t03"],
        "b_next":  ["t03", "t01"],
    })
    # period_block_time: which fine steps belong to each block.
    period_block_time = pl.DataFrame({
        "d": ["d1"]*4,
        "b_first": ["t01", "t01", "t03", "t03"],
        "t":       ["t01", "t02", "t03", "t04"],
    })
    # dtttdt_block_interior: interior-of-block lag rows (within a block,
    # consecutive fine steps).  block 1: (t02 ← t01); block 2: (t04 ← t03).
    dtttdt_block_interior = pl.DataFrame({
        "d": ["d1", "d1"],
        "t": ["t02", "t04"],
        "t_previous": ["t01", "t03"],
    })

    return FlexData(
        dt=dt, p_step_duration=p_step, p_rp_cost_weight=p_rp,
        p_inflation_op=p_infl, p_period_share=p_psh,
        nodeBalance=nb, nodeBalance_dt=nb_dt,
        p_inflow=p_inflow, p_penalty_up=p_pen_up, p_penalty_down=p_pen_dn,
        nodeState=nodeState, nodeState_dt=nodeState_dt,
        nodeState_first_dt=nodeState_first_dt,
        nodeState_last_dt=nodeState_last_dt,
        p_state_unitsize=p_state_unitsize,
        p_state_upper=p_state_upper,
        p_state_self_discharge=p_state_self_discharge,
        p_state_existing_capacity=p_state_existing_capacity,
        dtttdt=dtttdt,
        nodeStateBlock=nodeStateBlock,
        period_block=period_block,
        period_block_succ=period_block_succ,
        period_block_time=period_block_time,
        dtttdt_block_interior=dtttdt_block_interior,
    )


# ---------------------------------------------------------------------------
# Fixture 4 — toy_uc_3t : single online_linear unit over 3 timesteps with
# minload, min_uptime=2, min_downtime=2, startup_cost.  Activates
# online_lin_ub (maxOnline_linear), minimum_uptime/downtime, minToSink_minload,
# and the startup-cost objective term.

@pytest.fixture(scope="function")
def toy_uc_3t() -> FlexData:
    dt, p_step, p_rp, p_infl, p_psh = _time_axes(["d1"], 3)
    nb, nb_dt, p_inflow, p_pen_up, p_pen_dn = _node_axes(
        "n", dt, inflow=-50.0, penalty=1e6)

    pss = pl.DataFrame({"p": ["u"], "source": ["FUEL_n"], "sink": ["n"]})
    pss_eff = pss.clone()
    pss_noEff = pl.DataFrame(
        schema={"p": pl.Utf8, "source": pl.Utf8, "sink": pl.Utf8})
    pss_dt = pss.join(dt, how="cross")
    flow_to_n = pss.with_columns(n=pl.col("sink"))
    flow_from_commodity_eff = pl.DataFrame(
        {"p": ["u"], "source": ["FUEL_n"], "sink": ["n"], "c": ["FUEL"]})
    flow_from_commodity_noEff = pl.DataFrame(
        schema={"p": pl.Utf8, "source": pl.Utf8, "sink": pl.Utf8, "c": pl.Utf8})

    p_unitsize = Param(("p",), pl.DataFrame({"p": ["u"], "value": [100.0]}))
    p_flow_upper = Param(("p", "source", "sink", "d", "t"),
        pss_dt.with_columns(value=pl.lit(1.0))
              .select("p", "source", "sink", "d", "t", "value"))
    p_slope = Param(("p", "d", "t"),
        dt.with_columns(p=pl.lit("u"), value=pl.lit(1.0))
          .select("p", "d", "t", "value"))
    p_commodity_price = Param(("c", "d", "t"),
        dt.with_columns(c=pl.lit("FUEL"), value=pl.lit(1.0))
          .select("c", "d", "t", "value"))

    # UC sets/params.
    process_online = pl.DataFrame({"p": ["u"]})
    process_online_linear = pl.DataFrame({"p": ["u"]})
    process_minload = pl.DataFrame({"p": ["u"]})
    p_online_dt = pss_dt.select("p", "d", "t").unique()
    pdt_online_linear = p_online_dt.clone()
    p_min_load = Param(("p",), pl.DataFrame({"p": ["u"], "value": [0.4]}))
    p_startup_cost = Param(("p", "d"),
        pl.DataFrame({"p": ["u"], "d": ["d1"], "value": [5.0]}))
    p_process_existing_count = Param(("p", "d"),
        pl.DataFrame({"p": ["u"], "d": ["d1"], "value": [1.0]}))

    # dtttdt with cyclic t_previous_within_solve (used by online dynamics).
    dtttdt = pl.DataFrame({
        "d": ["d1"]*3,
        "t": ["t01", "t02", "t03"],
        "t_previous": ["t03", "t01", "t02"],
        "t_previous_within_timeset": ["t03", "t01", "t02"],
        "d_previous": ["d1"]*3,
        "t_previous_within_solve": ["t03", "t01", "t02"],
    })

    # min_uptime/downtime lookback rows: for each (p, d, t), include
    # (d, t) self plus 1 step back (since min_time=2 and step_duration=1
    # means accumulate 1 hour, still <2 → include one predecessor; then
    # accumulate 2 → stop).  The walk uses the natural d/t order;
    # since this is a single period only, predecessors stay inside it.
    # Order: t01 → no back; t02 → back=t01; t03 → back=t02.  At t01
    # the back walk yields nothing (no earlier step).
    # Self rows are always included.
    uptime_lookback = pl.DataFrame({
        "p":      ["u"]*5,
        "d":      ["d1"]*5,
        "t":      ["t01", "t02", "t02", "t03", "t03"],
        "d_back": ["d1"]*5,
        "t_back": ["t01", "t02", "t01", "t03", "t02"],
    })
    downtime_lookback = uptime_lookback.clone()
    pdt_uptime_set = uptime_lookback.select("p", "d", "t").unique()
    pdt_downtime_set = pdt_uptime_set.clone()

    return FlexData(
        dt=dt, p_step_duration=p_step, p_rp_cost_weight=p_rp,
        p_inflation_op=p_infl, p_period_share=p_psh,
        nodeBalance=nb, nodeBalance_dt=nb_dt,
        p_inflow=p_inflow, p_penalty_up=p_pen_up, p_penalty_down=p_pen_dn,
        process_source_sink=pss, process_source_sink_eff=pss_eff,
        process_source_sink_noEff=pss_noEff, pss_dt=pss_dt,
        flow_to_n=flow_to_n,
        flow_from_commodity_eff=flow_from_commodity_eff,
        flow_from_commodity_noEff=flow_from_commodity_noEff,
        p_unitsize=p_unitsize, p_flow_upper=p_flow_upper,
        p_slope=p_slope, p_commodity_price=p_commodity_price,
        process_online=process_online,
        process_online_linear=process_online_linear,
        process_minload=process_minload,
        p_online_dt=p_online_dt,
        pdt_online_linear=pdt_online_linear,
        p_min_load=p_min_load,
        p_startup_cost=p_startup_cost,
        p_process_existing_count=p_process_existing_count,
        dtttdt=dtttdt,
        uptime_lookback=uptime_lookback,
        downtime_lookback=downtime_lookback,
        pdt_uptime_set=pdt_uptime_set,
        pdt_downtime_set=pdt_downtime_set,
    )


# ---------------------------------------------------------------------------
# Fixture 5 — toy_invest_3d : 3 periods, single investable process, no
# operations (zero inflow, zero demand).  Activates maxInvest_var_bound,
# maxInvest_entity_period (per-period cap) and ed_entity_annual_discounted
# objective term.

@pytest.fixture(scope="function")
def toy_invest_3d() -> FlexData:
    dt, p_step, p_rp, p_infl, p_psh = _time_axes(["d1", "d2", "d3"], 1)
    nb, nb_dt, p_inflow, p_pen_up, p_pen_dn = _node_axes(
        "n", dt, inflow=0.0, penalty=1e6)

    # Process source/sink so v_flow domain exists (max bound active).
    pss = pl.DataFrame({"p": ["u"], "source": ["FUEL_n"], "sink": ["n"]})
    pss_eff = pss.clone()
    pss_noEff = pl.DataFrame(
        schema={"p": pl.Utf8, "source": pl.Utf8, "sink": pl.Utf8})
    pss_dt = pss.join(dt, how="cross")
    flow_to_n = pss.with_columns(n=pl.col("sink"))
    flow_from_commodity_eff = pl.DataFrame(
        {"p": ["u"], "source": ["FUEL_n"], "sink": ["n"], "c": ["FUEL"]})
    flow_from_commodity_noEff = pl.DataFrame(
        schema={"p": pl.Utf8, "source": pl.Utf8, "sink": pl.Utf8, "c": pl.Utf8})

    p_unitsize = Param(("p",), pl.DataFrame({"p": ["u"], "value": [1.0]}))
    p_flow_upper = Param(("p", "source", "sink", "d", "t"),
        pss_dt.with_columns(value=pl.lit(0.0))   # no existing capacity
              .select("p", "source", "sink", "d", "t", "value"))
    # Existing-only cap (for maxToSink RHS): explicit zero so flow ≤ 0
    # without invest, ≤ invest with invest.  flexpy uses
    # p_flow_upper_existing when invest is active; provide it explicitly.
    p_flow_upper_existing = Param(("p", "source", "sink", "d"),
        pss.join(pl.DataFrame({"d": ["d1", "d2", "d3"]}), how="cross")
           .with_columns(value=pl.lit(0.0))
           .select("p", "source", "sink", "d", "value"))
    p_slope = Param(("p", "d", "t"),
        dt.with_columns(p=pl.lit("u"), value=pl.lit(1.0))
          .select("p", "d", "t", "value"))
    p_commodity_price = Param(("c", "d", "t"),
        dt.with_columns(c=pl.lit("FUEL"), value=pl.lit(1.0))
          .select("c", "d", "t", "value"))

    # Invest sets/params: unit ``u`` is investable in every period d1..d3.
    periods = ["d1", "d2", "d3"]
    pd_invest_set = pl.DataFrame(
        {"p": ["u"]*3, "d": periods})
    ed_invest_set = pd_invest_set.rename({"p": "e"})
    # edd_invest_set: which past investment is "alive" at each period.
    # invest_period_total: invest decision on each d only counts at that d
    # (no carry-forward beyond d).  Simplest: d_invest == d for each row.
    edd_invest_set = pl.DataFrame(
        {"e": ["u"]*3, "d_invest": periods, "d": periods})
    p_entity_max_units = Param(("e", "d"),
        pl.DataFrame({"e": ["u"]*3, "d": periods, "value": [5.0]*3}))
    ed_invest_period_set = ed_invest_set.clone()
    ed_invest_max_period = Param(("e", "d"),
        pl.DataFrame({"e": ["u"]*3, "d": periods, "value": [5.0]*3}))
    # Annualised objective coefficient — drives invest cost.
    ed_entity_annual_discounted = Param(("e", "d"),
        pl.DataFrame({"e": ["u"]*3, "d": periods, "value": [10.0]*3}))
    ed_lifetime_fixed_cost = Param(("e", "d"),
        pl.DataFrame({"e": ["u"]*3, "d": periods, "value": [0.0]*3}))

    return FlexData(
        dt=dt, p_step_duration=p_step, p_rp_cost_weight=p_rp,
        p_inflation_op=p_infl, p_period_share=p_psh,
        nodeBalance=nb, nodeBalance_dt=nb_dt,
        p_inflow=p_inflow, p_penalty_up=p_pen_up, p_penalty_down=p_pen_dn,
        process_source_sink=pss, process_source_sink_eff=pss_eff,
        process_source_sink_noEff=pss_noEff, pss_dt=pss_dt,
        flow_to_n=flow_to_n,
        flow_from_commodity_eff=flow_from_commodity_eff,
        flow_from_commodity_noEff=flow_from_commodity_noEff,
        p_unitsize=p_unitsize,
        p_flow_upper=p_flow_upper,
        p_flow_upper_existing=p_flow_upper_existing,
        p_slope=p_slope, p_commodity_price=p_commodity_price,
        pd_invest_set=pd_invest_set,
        ed_invest_set=ed_invest_set,
        edd_invest_set=edd_invest_set,
        p_entity_max_units=p_entity_max_units,
        ed_invest_period_set=ed_invest_period_set,
        ed_invest_max_period=ed_invest_max_period,
        ed_entity_annual_discounted=ed_entity_annual_discounted,
        ed_lifetime_fixed_cost=ed_lifetime_fixed_cost,
    )


# ---------------------------------------------------------------------------
# Fixture 6 — toy_2branch_2d : 2-branch stochastic, 2 periods (d1
# deterministic, d2 splits into b1/b2).  1 storage node so
# non_anticipativity_storage_use binds.

@pytest.fixture(scope="function")
def toy_2branch_2d() -> FlexData:
    # Single timestep per "period"; each branch is its own period in
    # the dt frame.  Anchor period: d2; siblings: d2_b1, d2_b2.
    periods = ["d1", "d2", "d2_b1", "d2_b2"]
    dt = pl.DataFrame({"d": periods, "t": ["t01"]*4})
    p_step = Param(("d", "t"), dt.with_columns(value=pl.lit(1.0)))
    p_rp = Param(("d", "t"), dt.with_columns(value=pl.lit(1.0)))
    p_infl = Param(("d",),
        pl.DataFrame({"d": periods, "value": [1.0]*4}))
    p_psh = Param(("d",),
        pl.DataFrame({"d": periods, "value": [1.0]*4}))

    nb = pl.DataFrame({"n": ["s"]})  # storage-only
    nb_dt = nb.join(dt, how="cross")
    p_inflow = Param(("n", "d", "t"),
        nb_dt.with_columns(value=pl.lit(0.0)).select("n", "d", "t", "value"))
    p_pen_up = Param(("n", "d", "t"),
        nb_dt.with_columns(value=pl.lit(1e6)).select("n", "d", "t", "value"))
    p_pen_dn = Param(("n", "d", "t"),
        nb_dt.with_columns(value=pl.lit(1e6)).select("n", "d", "t", "value"))

    nodeState = pl.DataFrame({"n": ["s"]})
    nodeState_dt = nodeState.join(dt, how="cross")
    nodeState_first_dt = (nodeState_dt.sort(["n", "d", "t"])
        .group_by(["n", "d"], maintain_order=True).first()
        .select("n", "d", "t"))
    nodeState_last_dt = (nodeState_dt.sort(["n", "d", "t"])
        .group_by(["n", "d"], maintain_order=True).last()
        .select("n", "d", "t"))
    p_state_unitsize = Param(("n",),
        pl.DataFrame({"n": ["s"], "value": [10.0]}))
    p_state_upper = Param(("n", "d"),
        pl.DataFrame({"n": ["s"]*4, "d": periods, "value": [1.0]*4}))
    p_state_self_discharge = Param(("n",),
        pl.DataFrame({"n": ["s"], "value": [0.0]}))
    p_state_existing_capacity = Param(("n", "d"),
        pl.DataFrame({"n": ["s"]*4, "d": periods, "value": [10.0]*4}))
    # dtttdt with cyclic in-period lag (single timestep per period self-loops).
    dtttdt = pl.DataFrame({
        "d": periods, "t": ["t01"]*4,
        "t_previous": ["t01"]*4,
        "t_previous_within_timeset": ["t01"]*4,
        "d_previous": periods,
        "t_previous_within_solve": ["t01"]*4,
    })

    # Stochastic frames.  period_branch_full has anchor d → siblings.
    period_branch_full = pl.DataFrame({
        "d": ["d2", "d2", "d2"],
        "b": ["d2", "d2_b1", "d2_b2"],
    })
    # Realised-dispatch timesteps (apply non-anticipativity here).
    dt_non_anticipativity = pl.DataFrame({
        "d": ["d2"], "t": ["t01"]})
    # Group ``g_stoch`` covering node ``s`` enables storage non-anticipativity.
    groupStochastic = pl.DataFrame({"g": ["g_stoch"]})
    group_node = pl.DataFrame({"g": ["g_stoch"], "n": ["s"]})
    period_in_use_set = pl.DataFrame({"d": periods})
    pdt_branch_weight = Param(("d", "t"), dt.with_columns(value=pl.lit(1.0)))
    pd_branch_weight = Param(("d",),
        pl.DataFrame({"d": periods, "value": [1.0]*4}))

    return FlexData(
        dt=dt, p_step_duration=p_step, p_rp_cost_weight=p_rp,
        p_inflation_op=p_infl, p_period_share=p_psh,
        nodeBalance=nb, nodeBalance_dt=nb_dt,
        p_inflow=p_inflow, p_penalty_up=p_pen_up, p_penalty_down=p_pen_dn,
        nodeState=nodeState, nodeState_dt=nodeState_dt,
        nodeState_first_dt=nodeState_first_dt,
        nodeState_last_dt=nodeState_last_dt,
        p_state_unitsize=p_state_unitsize,
        p_state_upper=p_state_upper,
        p_state_self_discharge=p_state_self_discharge,
        p_state_existing_capacity=p_state_existing_capacity,
        dtttdt=dtttdt,
        period_branch_full=period_branch_full,
        dt_non_anticipativity=dt_non_anticipativity,
        groupStochastic=groupStochastic,
        group_node=group_node,
        period_in_use_set=period_in_use_set,
        pdt_branch_weight=pdt_branch_weight,
        pd_branch_weight=pd_branch_weight,
    )


# ---------------------------------------------------------------------------
# Fixture 7 — toy_2node_chp : single CHP-style indirect process with one
# fuel input (``fuel``) and two outputs (``heat``, ``elec``), with
# non-default sink coefficients.  Activates ``conversion_indirect``.

@pytest.fixture(scope="function")
def toy_2node_chp() -> FlexData:
    dt, p_step, p_rp, p_infl, p_psh = _time_axes(["d1"], 2)
    # nodeBalance for both demand sinks; ``fuel`` is a commodity node.
    nb = pl.DataFrame({"n": ["heat", "elec"]})
    nb_dt = nb.join(dt, how="cross")
    p_inflow = Param(("n", "d", "t"), nb_dt.with_columns(
        value=pl.when(pl.col("n") == "heat").then(-5.0)
              .otherwise(-10.0)
    ).select("n", "d", "t", "value"))
    p_pen_up = Param(("n", "d", "t"),
        nb_dt.with_columns(value=pl.lit(1e6)).select("n", "d", "t", "value"))
    p_pen_dn = Param(("n", "d", "t"),
        nb_dt.with_columns(value=pl.lit(1e6)).select("n", "d", "t", "value"))

    # Indirect process: arc 1 fuel→chp (input), arcs 2-3 chp→heat/elec (output).
    pss = pl.DataFrame({
        "p": ["chp"]*3,
        "source": ["fuel", "chp", "chp"],
        "sink":   ["chp", "heat", "elec"],
    })
    pss_eff = pss.clone()
    pss_noEff = pl.DataFrame(
        schema={"p": pl.Utf8, "source": pl.Utf8, "sink": pl.Utf8})
    pss_dt = pss.join(dt, how="cross")
    # Only the sink-side (output) arcs feed nodeBalance via flow_to_n.
    flow_to_n = (pss.filter(pl.col("sink").is_in(["heat", "elec"]))
                 .with_columns(n=pl.col("sink")))
    # Fuel commodity hookup on the input arc only.
    flow_from_commodity_eff = pl.DataFrame(
        {"p": ["chp"], "source": ["fuel"], "sink": ["chp"], "c": ["FUEL"]})
    flow_from_commodity_noEff = pl.DataFrame(
        schema={"p": pl.Utf8, "source": pl.Utf8, "sink": pl.Utf8, "c": pl.Utf8})

    p_unitsize = Param(("p",), pl.DataFrame({"p": ["chp"], "value": [100.0]}))
    p_flow_upper = Param(("p", "source", "sink", "d", "t"),
        pss_dt.with_columns(value=pl.lit(1.0))
              .select("p", "source", "sink", "d", "t", "value"))
    p_slope = Param(("p", "d", "t"),
        dt.with_columns(p=pl.lit("chp"), value=pl.lit(1.0))
          .select("p", "d", "t", "value"))
    p_commodity_price = Param(("c", "d", "t"),
        dt.with_columns(c=pl.lit("FUEL"), value=pl.lit(1.0))
          .select("c", "d", "t", "value"))

    process_indirect = pl.DataFrame({"p": ["chp"]})
    process_input_flows = pl.DataFrame(
        {"p": ["chp"], "source": ["fuel"], "sink": ["chp"]})
    process_output_flows = pl.DataFrame(
        {"p": ["chp"]*2, "source": ["chp"]*2, "sink": ["heat", "elec"]})
    process_indirect_dt = process_indirect.join(dt, how="cross")
    # Sink coefficients: heat 0.5, elec 1.0 (non-default to exercise the
    # multiplier path).
    p_process_sink_flow_coef = Param(("p", "sink"),
        pl.DataFrame({"p": ["chp"]*2, "sink": ["heat", "elec"],
                      "value": [0.5, 1.0]}))

    return FlexData(
        dt=dt, p_step_duration=p_step, p_rp_cost_weight=p_rp,
        p_inflation_op=p_infl, p_period_share=p_psh,
        nodeBalance=nb, nodeBalance_dt=nb_dt,
        p_inflow=p_inflow, p_penalty_up=p_pen_up, p_penalty_down=p_pen_dn,
        process_source_sink=pss, process_source_sink_eff=pss_eff,
        process_source_sink_noEff=pss_noEff, pss_dt=pss_dt,
        flow_to_n=flow_to_n,
        flow_from_commodity_eff=flow_from_commodity_eff,
        flow_from_commodity_noEff=flow_from_commodity_noEff,
        p_unitsize=p_unitsize, p_flow_upper=p_flow_upper,
        p_slope=p_slope, p_commodity_price=p_commodity_price,
        process_indirect=process_indirect,
        process_input_flows=process_input_flows,
        process_output_flows=process_output_flows,
        process_indirect_dt=process_indirect_dt,
        p_process_sink_flow_coef=p_process_sink_flow_coef,
    )


# ---------------------------------------------------------------------------
# Fixture 8 — toy_group_reserve : 2 nodes covered by 1 group, 1 reserve
# product (timeseries), 1 capacity_margin floor, 1 producer process.
# Activates ``capacityMargin``, ``reserveBalance_timeseries_eq`` plus
# the per-process reserve upper bound.

@pytest.fixture(scope="function")
def toy_group_reserve() -> FlexData:
    dt, p_step, p_rp, p_infl, p_psh = _time_axes(["d1"], 2)
    # Two demand nodes, 1 fuel commodity node.
    nb = pl.DataFrame({"n": ["n1", "n2"]})
    nb_dt = nb.join(dt, how="cross")
    p_inflow = Param(("n", "d", "t"), nb_dt.with_columns(
        value=pl.when(pl.col("n") == "n1").then(-30.0).otherwise(-20.0)
    ).select("n", "d", "t", "value"))
    p_pen_up = Param(("n", "d", "t"),
        nb_dt.with_columns(value=pl.lit(1e6)).select("n", "d", "t", "value"))
    p_pen_dn = Param(("n", "d", "t"),
        nb_dt.with_columns(value=pl.lit(1e6)).select("n", "d", "t", "value"))

    # 1 producer at n1 (the reserve provider).
    pss = pl.DataFrame({"p": ["u"], "source": ["FUEL_n"], "sink": ["n1"]})
    pss_eff = pss.clone()
    pss_noEff = pl.DataFrame(
        schema={"p": pl.Utf8, "source": pl.Utf8, "sink": pl.Utf8})
    pss_dt = pss.join(dt, how="cross")
    flow_to_n = pss.with_columns(n=pl.col("sink"))
    flow_from_commodity_eff = pl.DataFrame(
        {"p": ["u"], "source": ["FUEL_n"], "sink": ["n1"], "c": ["FUEL"]})
    flow_from_commodity_noEff = pl.DataFrame(
        schema={"p": pl.Utf8, "source": pl.Utf8, "sink": pl.Utf8, "c": pl.Utf8})

    p_unitsize = Param(("p",), pl.DataFrame({"p": ["u"], "value": [100.0]}))
    p_flow_upper = Param(("p", "source", "sink", "d", "t"),
        pss_dt.with_columns(value=pl.lit(1.0))
              .select("p", "source", "sink", "d", "t", "value"))
    p_slope = Param(("p", "d", "t"),
        dt.with_columns(p=pl.lit("u"), value=pl.lit(1.0))
          .select("p", "d", "t", "value"))
    p_commodity_price = Param(("c", "d", "t"),
        dt.with_columns(c=pl.lit("FUEL"), value=pl.lit(1.0))
          .select("c", "d", "t", "value"))
    p_process_existing_count = Param(("p", "d"),
        pl.DataFrame({"p": ["u"], "d": ["d1"], "value": [1.0]}))

    # Group spans both nodes.
    groupCapacityMargin = pl.DataFrame({"g": ["g"]})
    group_node = pl.DataFrame({"g": ["g"]*2, "n": ["n1", "n2"]})
    pdGroup_capacity_margin = Param(("g", "d"),
        pl.DataFrame({"g": ["g"], "d": ["d1"], "value": [10.0]}))
    pdGroup_penalty_capacity_margin = Param(("g", "d"),
        pl.DataFrame({"g": ["g"], "d": ["d1"], "value": [1.0e3]}))
    p_group_capacity_for_scaling = Param(("g", "d"),
        pl.DataFrame({"g": ["g"], "d": ["d1"], "value": [1.0]}))
    p_inv_group_cap = Param(("g", "d"),
        pl.DataFrame({"g": ["g"], "d": ["d1"], "value": [1.0]}))
    pdtNodeInflow_per_step = p_inflow  # step_duration=1 → identical.

    # Reserve subsystem: 1 (r, ud, g) tuple, timeseries method.
    reserve_upDown_group = pl.DataFrame(
        {"r": ["r1"], "ud": ["up"], "g": ["g"]})
    reserve_upDown_group_method_timeseries = pl.DataFrame(
        {"r": ["r1"], "ud": ["up"], "g": ["g"], "method": ["timeseries"]})
    reserve_upDown_group_method_dynamic = pl.DataFrame(
        schema={"r": pl.Utf8, "ud": pl.Utf8, "g": pl.Utf8, "method": pl.Utf8})
    reserve_upDown_group_method_n_1 = pl.DataFrame(
        schema={"r": pl.Utf8, "ud": pl.Utf8, "g": pl.Utf8, "method": pl.Utf8})
    prundt = pl.DataFrame({
        "p": ["u"]*2, "r": ["r1"]*2, "ud": ["up"]*2, "n": ["n1"]*2,
        "d": ["d1"]*2, "t": ["t01", "t02"]})
    process_reserve_upDown_node_active = pl.DataFrame(
        {"p": ["u"], "r": ["r1"], "ud": ["up"], "n": ["n1"]})
    p_process_reserve_upDown_node_reliability = Param(("p", "r", "ud", "n"),
        pl.DataFrame({"p": ["u"], "r": ["r1"], "ud": ["up"], "n": ["n1"],
                      "value": [1.0]}))
    pdtReserve_upDown_group_reservation = Param(("r", "ud", "g", "d", "t"),
        pl.DataFrame({"r": ["r1"]*2, "ud": ["up"]*2, "g": ["g"]*2,
                      "d": ["d1"]*2, "t": ["t01", "t02"],
                      "value": [10.0, 10.0]}))
    p_reserve_upDown_group_penalty_reserve = Param(("r", "ud", "g"),
        pl.DataFrame({"r": ["r1"], "ud": ["up"], "g": ["g"], "value": [500.0]}))
    p_process_reserve_upDown_node_max_share = Param(("p", "r", "ud", "n"),
        pl.DataFrame({"p": ["u"], "r": ["r1"], "ud": ["up"], "n": ["n1"],
                      "value": [1.0]}))

    return FlexData(
        dt=dt, p_step_duration=p_step, p_rp_cost_weight=p_rp,
        p_inflation_op=p_infl, p_period_share=p_psh,
        nodeBalance=nb, nodeBalance_dt=nb_dt,
        p_inflow=p_inflow, p_penalty_up=p_pen_up, p_penalty_down=p_pen_dn,
        process_source_sink=pss, process_source_sink_eff=pss_eff,
        process_source_sink_noEff=pss_noEff, pss_dt=pss_dt,
        flow_to_n=flow_to_n,
        flow_from_commodity_eff=flow_from_commodity_eff,
        flow_from_commodity_noEff=flow_from_commodity_noEff,
        p_unitsize=p_unitsize, p_flow_upper=p_flow_upper,
        p_slope=p_slope, p_commodity_price=p_commodity_price,
        p_process_existing_count=p_process_existing_count,
        # group / capacity margin
        groupCapacityMargin=groupCapacityMargin,
        group_node=group_node,
        pdGroup_capacity_margin=pdGroup_capacity_margin,
        pdGroup_penalty_capacity_margin=pdGroup_penalty_capacity_margin,
        p_group_capacity_for_scaling=p_group_capacity_for_scaling,
        p_inv_group_cap=p_inv_group_cap,
        pdtNodeInflow_per_step=pdtNodeInflow_per_step,
        # reserve
        reserve_upDown_group=reserve_upDown_group,
        reserve_upDown_group_method_timeseries=reserve_upDown_group_method_timeseries,
        reserve_upDown_group_method_dynamic=reserve_upDown_group_method_dynamic,
        reserve_upDown_group_method_n_1=reserve_upDown_group_method_n_1,
        prundt=prundt,
        process_reserve_upDown_node_active=process_reserve_upDown_node_active,
        p_process_reserve_upDown_node_reliability=p_process_reserve_upDown_node_reliability,
        pdtReserve_upDown_group_reservation=pdtReserve_upDown_group_reservation,
        p_reserve_upDown_group_penalty_reserve=p_reserve_upDown_group_penalty_reserve,
        p_process_reserve_upDown_node_max_share=p_process_reserve_upDown_node_max_share,
    )


# Self-tests live in ``test_fixtures_solve.py`` — pytest only collects
# from ``test_*.py`` files, not conftests.
