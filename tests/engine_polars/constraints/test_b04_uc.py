"""Surface B.4 — Unit-Commitment & Startup constraints.

Two focused tests on UC behavior in the polars engine:

* ``test_linear_and_integer_uc_constraint_suffix_isolation`` — covers
  **B4.11**.  When BOTH ``process_online_linear`` and
  ``process_online_integer`` are non-empty, ``_add_online_block`` is
  invoked twice — once per kind — with a ``_<kind>`` suffix on every
  emitted constraint name (``maxOnline_linear`` vs
  ``maxOnline_integer``, etc.).  Without the suffix the second
  ``add_cstr`` call would raise ``ValueError("already declared")``.
  This test builds a fixture with ONE process per kind, asserts both
  suffixed families appear in ``pb.cstr_names()``, that they reference
  distinct LP rows (one per process), and that solving still yields
  optimal.

* ``test_uptime_downtime_invest_interaction`` — covers **B4.12**.
  Regression guard for the formerly-documented gap at
  ``model.py:2920-2923``: the ``minimum_downtime`` constraint now
  applies the same invest/divest LHS-tightening as ``maxOnline`` so the
  .mod's RHS ``existing_count + Σ v_invest_alive − Σ v_divest_alive``
  is faithfully reproduced.  With ``existing_count=0`` and the unit
  invest-eligible, the cost-min optimum invests 0.5 unit (the minimum
  that satisfies demand AND uptime/downtime) and serves all load,
  rather than falling back to all-slack.
"""
from __future__ import annotations

import dataclasses
from typing import Any

import polars as pl
import pytest

from polar_high import Param, Problem
from flextool.engine_polars import build_flextool
from flextool.engine_polars.input import FlexData

from .conftest import solver_options


def _solve(data: FlexData) -> tuple[Problem, Any]:
    pb = Problem()
    build_flextool(pb, data)
    sol = pb.solve(options=solver_options())
    return pb, sol


# ---------------------------------------------------------------------------
# B4.11 — linear vs integer UC suffix isolation.

def _make_two_kind_uc_data() -> FlexData:
    """Build a 1-period × 2-step model with two processes — one
    ``online_linear`` (``u_lin``) and one ``online_integer`` (``u_int``) —
    sharing sink node ``n``.  Demand is small (= 25 MW per step) so
    either process alone can serve it; the test only cares that the
    LP is built without name collisions and remains feasible.
    """
    periods = ["d1"]
    steps = 2
    dt = pl.DataFrame(
        [{"d": d, "t": f"t{k:02d}"} for d in periods for k in range(1, steps + 1)])
    p_step = Param(("d", "t"), dt.with_columns(value=pl.lit(1.0)))
    p_rp = Param(("d", "t"), dt.with_columns(value=pl.lit(1.0)))
    p_infl = Param(("d",), pl.DataFrame({"d": periods, "value": [1.0]}))
    p_psh = Param(("d",), pl.DataFrame({"d": periods, "value": [1.0]}))

    nb = pl.DataFrame({"n": ["n"]})
    nb_dt = nb.join(dt, how="cross")
    p_inflow = Param(("n", "d", "t"),
        nb_dt.with_columns(value=pl.lit(-25.0))
              .select("n", "d", "t", "value"))
    p_pen = Param(("n", "d", "t"),
        nb_dt.with_columns(value=pl.lit(1e6)).select("n", "d", "t", "value"))

    # Two processes — one per UC kind.
    pss = pl.DataFrame({
        "p":      ["u_lin", "u_int"],
        "source": ["FUEL_n", "FUEL_n"],
        "sink":   ["n", "n"],
    })
    pss_eff = pss.clone()
    pss_noEff = pl.DataFrame(
        schema={"p": pl.Utf8, "source": pl.Utf8, "sink": pl.Utf8})
    pss_dt = pss.join(dt, how="cross")
    flow_to_n = pss.with_columns(n=pl.col("sink"))
    flow_from_commodity_eff = pl.DataFrame({
        "p": ["u_lin", "u_int"], "source": ["FUEL_n"]*2,
        "sink": ["n", "n"], "c": ["FUEL", "FUEL"]})
    flow_from_commodity_noEff = pl.DataFrame(
        schema={"p": pl.Utf8, "source": pl.Utf8, "sink": pl.Utf8, "c": pl.Utf8})

    p_unitsize = Param(("p",),
        pl.DataFrame({"p": ["u_lin", "u_int"], "value": [100.0, 100.0]}))
    p_flow_upper = Param(("p", "source", "sink", "d", "t"),
        pss_dt.with_columns(value=pl.lit(1.0))
              .select("p", "source", "sink", "d", "t", "value"))
    p_slope = Param(("p", "d", "t"),
        pss_dt.select("p", "d", "t").with_columns(value=pl.lit(1.0)))
    p_commodity_price = Param(("c", "d", "t"),
        dt.with_columns(c=pl.lit("FUEL"), value=pl.lit(1.0))
          .select("c", "d", "t", "value"))

    process_online = pl.DataFrame({"p": ["u_lin", "u_int"]})
    process_online_linear = pl.DataFrame({"p": ["u_lin"]})
    process_online_integer = pl.DataFrame({"p": ["u_int"]})
    p_online_dt = pss_dt.select("p", "d", "t").unique()
    pdt_online_linear = p_online_dt.filter(pl.col("p") == "u_lin")
    pdt_online_integer = p_online_dt.filter(pl.col("p") == "u_int")

    p_process_existing_count = Param(("p", "d"),
        pl.DataFrame({"p": ["u_lin", "u_int"], "d": ["d1", "d1"],
                      "value": [1.0, 1.0]}))
    # ``p_min_load`` is required by the ONLINE feature gate (model.py:79
    # ``_check``) even when ``process_minload`` is empty — populate as 0.
    p_min_load = Param(("p",),
        pl.DataFrame({"p": ["u_lin", "u_int"], "value": [0.0, 0.0]}))
    p_startup_cost = Param(("p", "d"),
        pl.DataFrame({"p": ["u_lin", "u_int"], "d": ["d1", "d1"],
                      "value": [0.0, 0.0]}))

    # 2-step cyclic dtttdt.
    dtttdt = pl.DataFrame({
        "d": ["d1", "d1"], "t": ["t01", "t02"],
        "t_previous": ["t02", "t01"],
        "t_previous_within_timeset": ["t02", "t01"],
        "d_previous": ["d1", "d1"],
        "t_previous_within_solve": ["t02", "t01"],
    })

    return FlexData(
        dt=dt, p_step_duration=p_step, p_rp_cost_weight=p_rp,
        p_inflation_op=p_infl, p_period_share=p_psh,
        nodeBalance=nb, nodeBalance_dt=nb_dt,
        p_inflow=p_inflow, p_penalty_up=p_pen, p_penalty_down=p_pen,
        process_source_sink=pss, process_source_sink_eff=pss_eff,
        process_source_sink_noEff=pss_noEff, pss_dt=pss_dt,
        flow_to_n=flow_to_n,
        flow_from_commodity_eff=flow_from_commodity_eff,
        flow_from_commodity_noEff=flow_from_commodity_noEff,
        p_unitsize=p_unitsize, p_flow_upper=p_flow_upper,
        p_slope=p_slope, p_commodity_price=p_commodity_price,
        process_online=process_online,
        process_online_linear=process_online_linear,
        process_online_integer=process_online_integer,
        p_online_dt=p_online_dt,
        pdt_online_linear=pdt_online_linear,
        pdt_online_integer=pdt_online_integer,
        p_min_load=p_min_load,
        p_startup_cost=p_startup_cost,
        p_process_existing_count=p_process_existing_count,
        dtttdt=dtttdt,
    )


def test_linear_and_integer_uc_constraint_suffix_isolation():
    """Covers B4.11 — `_linear`/`_integer` constraint-name suffix
    isolation in ``_add_online_block``.

    Hand-calc: two processes (``u_lin`` linear, ``u_int`` integer) over
    1 period × 2 steps.  ``_add_online_block`` is invoked twice — once
    per kind — emitting ``maxOnline_linear`` (1 row × 2 steps) and
    ``maxOnline_integer`` (1 row × 2 steps) as DISTINCT families.
    Without the suffix the second call would raise.
    """
    data = _make_two_kind_uc_data()
    pb, sol = _solve(data)
    assert sol.optimal
    cstr = set(pb.cstr_names())
    # Hand-calc: every UC constraint family is suffix-namespaced.
    for base in ("maxOnline", "maxStartup", "maxShutdown",
                 "online__startup", "online__shutdown", "maxToSink_online"):
        assert f"{base}_linear" in cstr, f"missing {base}_linear"
        assert f"{base}_integer" in cstr, f"missing {base}_integer"
    # And the row counts: each suffixed family covers exactly its own
    # process (1 process × 1 period × 2 steps = 2 rows).
    assert pb.cstr_row_count("maxOnline_linear") == 2
    assert pb.cstr_row_count("maxOnline_integer") == 2


# ---------------------------------------------------------------------------
# B4.12 — uptime/downtime + invest interaction (KNOWN GAP).

def test_uptime_downtime_invest_interaction(toy_uc_3t):
    """Covers B4.12 — uptime/downtime × invest interaction.

    Mutates ``toy_uc_3t``: existing_count=0 and the unit becomes
    invest-eligible (``invest_method=invest_period_total``,
    ``entity_max_units=2``, cheap fixed cost).

    Hand-calc: with cyclic 3-step demand 50 MW at every step
    (inflow=-50, unitsize=100), the cheapest feasible solution is
    v_invest_p[u, d1] = 0.5, v_online = 0.5 across all 3 steps,
    v_flow = 0.5 each step (= 50 MW), v_startup = v_shutdown = 0.
    Cost: invest 0.5 × 10 = 5 + fuel 3 × 0.5 × 100 × 1 = 150 ⇒ obj 155.

    Regression guard: the .mod's ``minimum_downtime`` includes
    ``v_invest`` on the RHS (existing_count + Σ v_invest_alive); the
    polars engine now applies the same invest/divest LHS-tightening as
    ``maxOnline`` so the LP can reach v_online > 0 via investment.
    Prior to the fix this row was un-tightened and the LP fell back to
    all-slack (v_invest_p = 0, vq_state_up = 50 each step).
    """
    d = toy_uc_3t
    # Existing capacity = 0 — we MUST invest to serve any load.
    p_exist_zero = Param(("p", "d"),
        pl.DataFrame({"p": ["u"], "d": ["d1"], "value": [0.0]}))
    # Invest sets/params.
    pd_invest = pl.DataFrame({"p": ["u"], "d": ["d1"]})
    edd_invest = pl.DataFrame(
        {"e": ["u"], "d_invest": ["d1"], "d": ["d1"]})
    p_max_units = Param(("e", "d"),
        pl.DataFrame({"e": ["u"], "d": ["d1"], "value": [2.0]}))
    ed_invest_period = pl.DataFrame({"e": ["u"], "d": ["d1"]})
    # ed_invest_max_period is in absolute capacity units (MW); the engine
    # pre-divides by unitsize=100 so the LP-side cap becomes
    # 200 / 100 = 2 units, matching p_entity_max_units.
    ed_max_period = Param(("e", "d"),
        pl.DataFrame({"e": ["u"], "d": ["d1"], "value": [200.0]}))
    ed_annual = Param(("e", "d"),
        pl.DataFrame({"e": ["u"], "d": ["d1"], "value": [10.0]}))
    ed_lifetime = Param(("e", "d"),
        pl.DataFrame({"e": ["u"], "d": ["d1"], "value": [0.0]}))

    data = dataclasses.replace(d,
        p_process_existing_count=p_exist_zero,
        pd_invest_set=pd_invest,
        edd_invest_set=edd_invest,
        p_entity_max_units=p_max_units,
        ed_invest_period_set=ed_invest_period,
        ed_invest_max_period=ed_max_period,
        ed_entity_annual_discounted=ed_annual,
        ed_lifetime_fixed_cost=ed_lifetime,
    )
    pb, sol = _solve(data)
    assert sol.optimal
    # Expected (post-fix) optimum: v_invest_p = 0.5, v_online = 0.5.
    v_invest = sol.value("v_invest_p").filter(
        (pl.col("p") == "u") & (pl.col("d") == "d1"))
    assert v_invest["value"][0] == pytest.approx(0.5, rel=1e-7)
    v_online = sol.value("v_online_linear").sort("t")
    for v in v_online["value"].to_list():
        assert v == pytest.approx(0.5, rel=1e-7)
