"""Indirect units with ``conversion_method = min_load_efficiency``.

The fuel of a min_load_efficiency unit is ``slope · output + section ·
online`` (flextool.mod ``conversion_indirect``)::

    Σ_s conv_s · in_s  =  slope · Σ_k conv_k · out_k
                          + section · v_online · unitsize

The section term used to be missing from ``conversion_indirect``: an
indirect unit burnt ``slope · output`` only — at full load 611.1 MW of
coal instead of 888.9 MW for ``coal_chp`` (efficiency 0.9, min_load 0.2,
efficiency_at_min_load 0.4), understating fuel, commodity cost and CO2 by
31 %.  ``conversion_indirect`` is the only place the section enters for an
indirect unit: its input is its own ``v_flow``, so the node balance,
commodity cost, CO2 and the reported input follow from it.

Fixtures: ``tests.json`` + an override alternative, built from JSON.
``coal_chp``: indirect, 1000 MW, efficiency 0.9, coal_market -> west +
heat (its ratio-fixing user constraint is switched off).
"""
from __future__ import annotations

import warnings

import pandas as pd
import polars as pl
import pytest

from tests.engine_polars._unit_db import (
    CHP, CHP_CHAIN, COAL, b64, demand, flow, make_db, no_ratio_constraint,
    online, run, time_map, uc, utf8,
)

pytestmark = pytest.mark.solver

EFF, MIN_LOAD, EFF_MIN = 0.9, 0.2, 0.4
# Fuel per MW of capacity: 1/EFF at full load, MIN_LOAD/EFF_MIN at min load.
SLOPE = (1.0 / EFF - MIN_LOAD / EFF_MIN) / (1.0 - MIN_LOAD)
SECTION = MIN_LOAD / EFF_MIN - MIN_LOAD * SLOPE
COAL_PRICE, CO2_CONTENT, CO2_PRICE = 20.0, 0.34, 50.0
N_STEPS = 48  # the y2020_2day_dispatch solve: two days of hourly steps


def _mle(kind: str = "linear") -> list:
    return no_ratio_constraint() + uc(CHP, MIN_LOAD, kind) + [
        ["unit", CHP, "conversion_method",
         b64("min_load_efficiency", "str")],
        ["unit", CHP, "efficiency_at_min_load", b64(EFF_MIN, "float")],
    ]


def _param(step, name: str) -> pl.Series:
    """``coal_chp``'s (p, d, t) Param ``name`` ordered by (d, t)."""
    frame = getattr(step.flex_data, name).frame
    return (utf8(frame).filter(pl.col("p") == CHP)
            .sort("d", "t")["value"])


def _assert_fuel_identity(step, kind: str) -> pl.Series:
    """coal = slope · (west + heat) + section · online at every t;
    returns the output sum."""
    out = flow(step, CHP, CHP, "west") + flow(step, CHP, CHP, "heat")
    on = online(step, CHP, kind)
    expected = _param(step, "p_slope") * out + _param(step, "p_section") * on
    assert flow(step, CHP, COAL, CHP).to_list() == pytest.approx(
        expected.to_list(), rel=1e-6, abs=1e-4)
    return out


def test_section_params_cover_the_indirect_unit(tmp_path_factory) -> None:
    """The coefficients the test relies on: coal_chp is indirect and
    min_load_efficiency with the analytic slope / section."""
    url, scen = make_db(tmp_path_factory, alt="imle_par", values=_mle()
                        + [demand("west", 500.0), demand("heat", 300.0)],
                        base_chain=CHP_CHAIN)
    step = list(run(tmp_path_factory, url, scen).values())[-1]
    fd = step.flex_data
    assert CHP in utf8(fd.process_indirect)["p"].to_list()
    assert CHP in utf8(fd.process_min_load_eff)["p"].to_list()
    assert _param(step, "p_slope").to_list() == pytest.approx(
        [SLOPE] * N_STEPS, rel=1e-9)
    # The section is rounded to 6 decimals by the engine (as in the .mod).
    assert _param(step, "p_section").to_list() == pytest.approx(
        [SECTION] * N_STEPS, abs=1e-6)


def test_full_load_linear(tmp_path_factory) -> None:
    """Outputs 500 + 300 on linear UC: online 800, coal 888.9 MW
    (= 800 / 0.9; was 611.1 = 800 · slope)."""
    url, scen = make_db(tmp_path_factory, alt="imle_full", values=_mle()
                        + [demand("west", 500.0), demand("heat", 300.0)],
                        base_chain=CHP_CHAIN)
    step = list(run(tmp_path_factory, url, scen).values())[-1]
    out = _assert_fuel_identity(step, "linear")
    assert out.to_list() == pytest.approx([800.0] * N_STEPS, rel=1e-6)
    assert online(step, CHP).to_list() == pytest.approx([800.0] * N_STEPS,
                                                        rel=1e-6)
    assert flow(step, CHP, COAL, CHP).to_list() == pytest.approx(
        [800.0 / EFF] * N_STEPS, rel=1e-6)


def test_part_load_linear(tmp_path_factory) -> None:
    """West demand alternates 800 / 200 MW and starts are expensive: the
    unit stays online at 800 MW and runs at part load, burning
    ``slope · out + section · online`` (200 MW out → 500 MW coal)."""
    low_hours = (f"t{i:04d}" for i in range(2, N_STEPS + 1, 2))
    west_inflow = time_map({t: -200.0 for t in low_hours}, -800.0)
    vals = _mle() + [
        ["node", "west", "inflow", west_inflow],
        demand("heat", 0.0),
        ["unit", CHP, "startup_cost", b64(1.0e4, "float")],
    ]
    url, scen = make_db(tmp_path_factory, alt="imle_part", values=vals,
                        base_chain=CHP_CHAIN)
    step = list(run(tmp_path_factory, url, scen).values())[-1]
    out = _assert_fuel_identity(step, "linear")
    on = online(step, CHP)
    part = [(o, n) for o, n in zip(out, on) if o < n - 1.0]
    assert part, "expected part-load hours (output below online capacity)"
    for o, n in part:
        assert o == pytest.approx(200.0, rel=1e-6)
        assert n == pytest.approx(800.0, rel=1e-6)
    coal = flow(step, CHP, COAL, CHP)
    assert min(coal) == pytest.approx(SLOPE * 200.0 + SECTION * 800.0,
                                      rel=1e-6)


def test_integer_online(tmp_path_factory) -> None:
    """Integer UC (one 1000 MW unit): online 1000 for 800 MW of output,
    coal = slope · 800 + section · 1000."""
    url, scen = make_db(tmp_path_factory, alt="imle_int",
                        values=_mle("binary")
                        + [demand("west", 500.0), demand("heat", 300.0)],
                        base_chain=CHP_CHAIN)
    step = list(run(tmp_path_factory, url, scen).values())[-1]
    _assert_fuel_identity(step, "integer")
    assert online(step, CHP, "integer").to_list() == pytest.approx(
        [1000.0] * N_STEPS, rel=1e-6)
    assert flow(step, CHP, COAL, CHP).to_list() == pytest.approx(
        [SLOPE * 800.0 + SECTION * 1000.0] * N_STEPS, rel=1e-6)


def test_delayed_unit_section_at_output_time(tmp_path_factory) -> None:
    """Delay 1 h (weight 1): the coal at ``t − 1`` supplies
    ``slope · out[t] + section · online[t]`` — the section sits on the
    output side at the sink-side time with that time's online count."""
    delay = b64({"index_type": "float", "rank": 1, "data": [[1.0, 1.0]],
                 "index_name": "constraint", "type": "map"}, "map")
    url, scen = make_db(tmp_path_factory, alt="imle_delay", values=_mle()
                        + [demand("west", 500.0), demand("heat", 300.0),
                           ["unit", CHP, "delay", delay]],
                        base_chain=CHP_CHAIN)
    step = list(run(tmp_path_factory, url, scen).values())[-1]
    out = flow(step, CHP, CHP, "west") + flow(step, CHP, CHP, "heat")
    on = online(step, CHP)
    coal = flow(step, CHP, COAL, CHP)
    expected = _param(step, "p_slope") * out + _param(step, "p_section") * on
    # The cyclic period wraps: output at t is fed by coal at t − 1.
    assert coal.shift(1, fill_value=coal[-1]).to_list() == pytest.approx(
        expected.to_list(), rel=1e-6, abs=1e-4)
    assert out.to_list() == pytest.approx([800.0] * N_STEPS, rel=1e-6)
    assert coal.to_list() == pytest.approx([800.0 / EFF] * N_STEPS, rel=1e-6)


_ROLLING = [
    ["solve", "y2020_2day_dispatch", "solve_mode",
     b64("rolling_window", "str")],
    ["solve", "y2020_2day_dispatch", "rolling_solve_horizon",
     b64(24.0, "float")],
    ["solve", "y2020_2day_dispatch", "rolling_solve_jump",
     b64(12.0, "float")],
]


def test_rolling_every_roll(tmp_path_factory) -> None:
    """Rolling chain (per-roll emit path): every roll carries the section
    Param for coal_chp and burns slope · out + section · online."""
    url, scen = make_db(tmp_path_factory, alt="imle_roll", values=_mle()
                        + [demand("west", 500.0), demand("heat", 300.0)]
                        + _ROLLING, base_chain=CHP_CHAIN)
    steps = run(tmp_path_factory, url, scen)
    assert len(steps) > 1
    for step in steps.values():
        assert _param(step, "p_section").len() > 0
        _assert_fuel_identity(step, "linear")
        assert flow(step, CHP, COAL, CHP).to_list() == pytest.approx(
            [800.0 / EFF] * flow(step, CHP, COAL, CHP).len(), rel=1e-6)


def test_reported_input_cost_and_co2(tmp_path_factory) -> None:
    """The written outputs follow the fuel: reported coal input, commodity
    cost and CO2 cost are 888.9 MW · price per hour."""
    from flextool.process_outputs.write_outputs import write_outputs

    vals = _mle() + [
        demand("west", 500.0), demand("heat", 300.0),
        ["commodity", "coal", "price", b64(COAL_PRICE, "float")],
        ["commodity", "coal", "co2_content", b64(CO2_CONTENT, "float")],
    ]
    url, scen = make_db(tmp_path_factory, alt="imle_out", values=vals,
                        base_chain=CHP_CHAIN + ["co2_price"])
    work = tmp_path_factory.mktemp("imle_out_w")
    from flextool.engine_polars import run_chain_from_db
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        steps = run_chain_from_db(url, scen, work_folder=work,
                                  keep_solutions=True)
        last = next(reversed(steps.values()))
        write_outputs(
            scenario_name=scen, output_location=str(work), subdir=scen,
            write_methods=["csv"], fallback_output_location=str(work),
            raw_output_dir=str(work / "output_raw"),
            solution=last.solution, solve_name=last.solve_name,
            solve_steps=[(s.solve_name, s.flex_data, s.effective_solution)
                         for s in steps.values()],
            flex_data_provider=last.flex_data_provider,
        )
    out_dir = work / "output_csv" / scen
    coal_mw = 800.0 / EFF
    inputs = pd.read_csv(out_dir / "unit__inputNode__dt.csv", header=[0, 1])
    reported = inputs[(CHP, COAL)].astype(float).abs().to_list()
    assert reported == pytest.approx([coal_mw] * N_STEPS, rel=1e-6)
    costs = pd.read_csv(out_dir / "costs__dt.csv")
    assert costs["commodity_cost"].to_list() == pytest.approx(
        [coal_mw * COAL_PRICE] * N_STEPS, rel=1e-6)
    assert costs["co2"].to_list() == pytest.approx(
        [coal_mw * CO2_CONTENT * CO2_PRICE] * N_STEPS, rel=1e-6)
