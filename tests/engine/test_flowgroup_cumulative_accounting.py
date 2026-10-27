"""Flow-group cumulative flow limits weight their steps like every other operational measure, and
the flow-group's net flow counts the fuel a min_load_efficiency unit draws for the units online.

Before: ``{max,min}Cumulative_flow_period`` (and ``_solve``) weighted a step by its
``step_duration`` only — a representative-period's ``timestep_weight`` dropped out, so a
half-weight step and a full-weight step counted the same.  The flow group's net flow
(``_group_flow``) also measured only ``v_flow·slope`` on eff sources, leaving out the
``v_online·section`` fuel a ``min_load_efficiency`` unit draws while idling at partial load — the
same term ``nodeBalance_eq``, the CO2 price and the CO2 caps already count (``_eff_fuel_input``).

Under recourse (stochastic branches), the owner decision is that a constraint carries no shared
branch weight: each scenario path must respect the cumulative-flow-over-the-solve limit on its
own, so ``{max,min}Cumulative_flow_solve`` gains a ``_path`` row per (group, leaf) instead of one
row that lets one branch's surplus flow cover another branch's deficit.

Fixtures: ``tests.json`` + the ``weighted_2day`` timestep-weight override, a ``flowGroup`` over
``coal_plant``'s fuel arc from ``coal_market`` (tests 1/2, built from JSON);
``stoch_two_period_hedge.json``'s fan-at-p2035 recourse topology (``base``/``stoch``/``ws``/
``na_window``/``divest``) + a ``flowGroup`` over ``base``'s output arc (test 3).
"""
from __future__ import annotations

import polars as pl
import pytest

from tests.engine._lp_rows import row_bounds, row_terms
from tests.engine._unit_db import FLEXTOOL_ROOT, PLANT, b64, build, make_db, run, solve, uc, utf8

MINLOAD_CHAIN = ["init", "west", "coal", "coal_min_load", "weighted_2day"]
GROUP = "coal_fuel"
FLOW_ENTITIES = [["flowGroup", GROUP, None],
                 ["flowGroup__unit__node", [GROUP, PLANT, "coal_market"], None]]
STOCH_FIXTURE = FLEXTOOL_ROOT / "tests" / "fixtures" / "stoch_two_period_hedge.json"
STOCH_CHAIN = ["init", "base", "stoch", "ws", "na_window", "divest"]


def _value(param, **keys) -> float:
    """The single value of ``param`` at ``keys`` (keys absent from the Param's columns are ignored)."""
    f = utf8(param.frame if hasattr(param, "frame") else param)
    for col, val in keys.items():
        if col in f.columns:
            f = f.filter(pl.col(col) == val)
    assert f.height == 1, (keys, f)
    return float(f["value"][0])


def _flow_group_step(tmp_path_factory, alt: str, values: list):
    """``coal_plant`` drawing fuel from ``coal_market`` through a min_load_efficiency arc, with a
    ``flowGroup coal_fuel`` wrapping that arc, under the ``weighted_2day`` non-uniform timestep
    weights (t0001-t0012 half-weight, t0013-t0024 full-weight, within p2020)."""
    return solve(tmp_path_factory, alt=alt, base_chain=MINLOAD_CHAIN, entities=FLOW_ENTITIES,
                values=uc(PLANT, 0.5, "linear") + values)


def test_cumulative_flow_weights_timestep_weight(tmp_path_factory) -> None:
    """``minCumulative_flow_period[coal_fuel,p2020]`` weights each step by
    ``step_duration · timestep_weight``, not ``step_duration`` alone.

    ``weighted_2day`` gives t0001 a lower ``timestep_weight`` than t0013 (both steps are 1 h, so
    ``step_duration`` is equal); independently of the exact normalised values, the coefficient
    ratio of the two steps' ``v_flow`` columns in the row must equal the ratio of
    ``step_duration·timestep_weight`` read straight from the fixture's own params.  Before the
    fix the row carried ``step_duration`` only, so the ratio was 1.0 regardless of the weights.
    The RHS must equal ``cap · Σ_t step_duration[t]·timestep_weight[t]`` over p2020 (hand-derived
    below from the fixture's params, not copied from the row).
    """
    pmap = {"index_type": "str", "index_name": "period", "data": [["p2020", -50.0]]}
    step = _flow_group_step(
        tmp_path_factory, "flow_period_weight",
        [["flowGroup", GROUP, "min_cumulative_flow", b64(pmap, "map")]])
    fd = step.flex_data
    pb = build(fd)
    row = f"minCumulative_flow_period[{GROUP},p2020]"
    terms = row_terms(pb, "minCumulative_flow_period")[row]

    t_low, t_high = "t0001", "t0013"
    col_low = f"v_flow[{PLANT},coal_market,west,p2020,{t_low}]"
    col_high = f"v_flow[{PLANT},coal_market,west,p2020,{t_high}]"
    assert col_low in terms and col_high in terms

    sd_low = _value(fd.p_step_duration, d="p2020", t=t_low)
    sd_high = _value(fd.p_step_duration, d="p2020", t=t_high)
    tw_low = _value(fd.p_timestep_weight, d="p2020", t=t_low)
    tw_high = _value(fd.p_timestep_weight, d="p2020", t=t_high)
    assert tw_low != tw_high, "weighted_2day must give t0001 and t0013 different timestep weights"

    ratio = terms[col_low] / terms[col_high]
    assert ratio == pytest.approx((sd_low * tw_low) / (sd_high * tw_high), rel=1e-9)

    cap = _value(fd.pd_min_cumulative_flow, g=GROUP, d="p2020")
    dts = utf8(fd.dt).filter(pl.col("d") == "p2020")["t"].to_list()
    weighted_hours = sum(
        _value(fd.p_step_duration, d="p2020", t=t) * _value(fd.p_timestep_weight, d="p2020", t=t)
        for t in dts)
    lower, _upper = row_bounds(pb, "minCumulative_flow_period")[row]
    assert lower == pytest.approx(cap * weighted_hours, rel=1e-9)


def test_group_flow_counts_minload_section(tmp_path_factory) -> None:
    """The same row's ``v_online_linear[coal_plant,d,t]`` coefficient (the section fuel a
    min_load_efficiency unit draws while online) stands in the fixed ratio ``section/slope`` to
    the ``v_flow`` coefficient of the same step — the fuel-draw measure ``_group_flow`` must share
    with ``nodeBalance_eq``.  Before the fix ``v_online`` was absent from the row entirely.
    """
    pmap = {"index_type": "str", "index_name": "period", "data": [["p2020", -50.0]]}
    step = _flow_group_step(
        tmp_path_factory, "flow_minload_section",
        [["flowGroup", GROUP, "min_cumulative_flow", b64(pmap, "map")]])
    fd = step.flex_data
    pb = build(fd)
    terms = row_terms(pb, "minCumulative_flow_period")[f"minCumulative_flow_period[{GROUP},p2020]"]

    seen = 0
    for t in utf8(fd.dt).filter(pl.col("d") == "p2020")["t"].to_list():
        on_col = f"v_online_linear[{PLANT},p2020,{t}]"
        flow_col = f"v_flow[{PLANT},coal_market,west,p2020,{t}]"
        assert on_col in terms, on_col
        section = _value(fd.p_section, p=PLANT, d="p2020", t=t)
        slope = _value(fd.p_slope, p=PLANT, d="p2020", t=t)
        assert terms[on_col] / terms[flow_col] == pytest.approx(section / slope, rel=1e-9)
        seen += 1
    assert seen > 0


def test_cumulative_flow_solve_gets_one_row_per_scenario_path(tmp_path_factory) -> None:
    """Under recourse, ``maxCumulative_flow_solve_path`` carries one row per scenario leaf, and no
    row's ``v_flow`` columns span periods of more than one leaf.

    ``stoch_two_period_hedge.json``'s fan-at-p2035 topology (``ws``) + the invest-NA window
    (``na_window``) + ``divest`` is the same recourse structure the entity/group ``_path`` total
    families already exercise; ``d_leaf`` partitions the periods into the real-named phantom trunk
    (``__anchor``) and the ``high``/``low`` forecast leaves.  Before the fix there was one bare
    ``maxCumulative_flow_solve`` row mixing every leaf's ``v_flow`` together, so one branch's
    surplus flow could cover another branch's deficit against the shared cap.
    """
    entities = [["flowGroup", "base_flow", None],
               ["flowGroup__unit__node", ["base_flow", "base", "elec"], None]]
    values = [["flowGroup", "base_flow", "max_cumulative_flow", b64(100000.0, "float")]]
    url, scen = make_db(tmp_path_factory, alt="flowgrp_solve_path", base_chain=STOCH_CHAIN,
                        entities=entities, values=values, fixture_json=STOCH_FIXTURE)
    steps = run(tmp_path_factory, url, scen)
    step = list(steps.values())[-1]
    fd = step.flex_data
    d_leaf = utf8(fd.d_leaf)
    assert d_leaf["leaf"].n_unique() > 1, "fixture must have more than one scenario leaf"

    pb = build(fd)
    rows = row_terms(pb, "maxCumulative_flow_solve_path")
    assert len(rows) == d_leaf["leaf"].n_unique()

    leaf_of = dict(zip(d_leaf["d"].to_list(), d_leaf["leaf"].to_list(), strict=True))
    for name, terms in rows.items():
        periods = {col.rsplit(",", 2)[-2] for col in terms if col.startswith("v_flow[")}
        assert periods, name
        leaves = {leaf_of[p] for p in periods}
        assert len(leaves) == 1, (name, leaves)
