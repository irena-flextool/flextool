"""Minimum load applies once per unit, on the sum of its outputs.

``minFlow_minload_<kind>`` is one row per ``(p, d, t)``::

    Σ_{output-measure arcs} v_flow  ≥  v_online · min_load

The output measure is the arc set that defines the unit's capacity: the
``conversion_flow_coeff ≠ 0`` outputs of an indirect unit, the single arc
of a direct unit.  Inputs never carry a min-load floor.  Before, every
input arc of an indirect unit got its own floor ``in_s ≥ v_online ·
min_load``, which forced an expensive second input to burn, took the unit
offline when an input could not supply, kept a heat-pump-like unit
(efficiency > 1) from ever running and forced unwanted electricity out of
an extraction CHP in pure-heat mode.

``unit__outputNode.capacity_min_coeff`` is an optional extra per-output
floor of multi-output units (``minFlow_output_floor_<kind>``)::

    v_flow[p, p, k]  ≥  v_online · min_load · capacity_min_coeff_k

It no longer scales (or switches off) the unit floor.

Fixtures: ``tests.json`` + an override alternative, built from JSON.
``coal_chp``: indirect, 1000 MW, efficiency 0.9, coal_market -> west +
heat (its ratio-fixing user constraint is switched off); ``coal_plant``:
direct, 500 MW.
"""
from __future__ import annotations

import logging
import types

import polars as pl
import pytest

from tests.engine_polars._unit_db import (
    CHP, CHP_CHAIN, COAL, GAS, GAS_ENTITIES, PLANT, PLANT_CHAIN, b64, build,
    cstr_over, demand, flow, gas, no_ratio_constraint, online, run, solve,
    time_map, uc, utf8, make_db,
)

# Objective of coal_chp serving the fixture's west + heat demand on coal.
COAL_ONLY_OBJ = 21_097_000.0


def cflow(step, source: str, sink: str) -> pl.Series:
    """coal_chp arc flow in MW."""
    return flow(step, CHP, source, sink)


def _chp(tpf, alt: str, values: list, entities=None):
    return solve(tpf, alt=alt, values=no_ratio_constraint() + values,
                 base_chain=CHP_CHAIN, entities=entities)


# ── measured regressions ────────────────────────────────────────────────
def test_expensive_second_input_is_not_forced(tmp_path_factory) -> None:
    """Two inputs, gas at 100 €/MWh, linear UC, min_load 0.4: gas stays at
    0 and the objective equals the coal-only run (was 330 230 100 with gas
    forced to 0.4 · online)."""
    step = _chp(tmp_path_factory, "ml_gas100", gas(100.0) + uc(CHP, 0.4),
                entities=GAS_ENTITIES)
    assert cflow(step, GAS, CHP).abs().max() == pytest.approx(0.0, abs=1e-6)
    assert step.solution.obj == pytest.approx(COAL_ONLY_OBJ, rel=1e-9)


def test_unit_runs_when_second_input_cannot_supply(tmp_path_factory) -> None:
    """Gas priced out (1e5 €/MWh): the unit runs on coal (was never
    online, all demand to slack, objective 5.04e9)."""
    step = _chp(tmp_path_factory, "ml_gas1e5", gas(1e5) + uc(CHP, 0.4),
                entities=GAS_ENTITIES)
    assert cflow(step, GAS, CHP).abs().max() == pytest.approx(0.0, abs=1e-6)
    assert step.solution.obj == pytest.approx(COAL_ONLY_OBJ, rel=1e-9)


def test_heat_pump_like_unit_runs(tmp_path_factory) -> None:
    """Single-input indirect unit with efficiency 3: the old input floor
    ``in ≥ 0.4 · online`` meant ``Σ out ≥ 1.2 · online`` so the unit never
    ran.  Now it serves west 500 + heat 300 on 800 / 3 MW of input."""
    step = _chp(tmp_path_factory, "ml_hp",
                [demand("west", 500.0), demand("heat", 300.0),
                 ["unit", CHP, "efficiency", b64(3.0, "float")]]
                + uc(CHP, 0.4))
    west, heat = cflow(step, CHP, "west"), cflow(step, CHP, "heat")
    coal = cflow(step, COAL, CHP)
    assert west.to_list() == pytest.approx([500.0] * west.len(), rel=1e-6)
    assert heat.to_list() == pytest.approx([300.0] * heat.len(), rel=1e-6)
    assert coal.to_list() == pytest.approx([800.0 / 3.0] * coal.len(),
                                           rel=1e-6)


def test_extraction_chp_pure_heat_mode(tmp_path_factory) -> None:
    """Extraction CHP (conversion_flow_coeff west 2 / heat 0.2), heat
    demand only: no electricity is forced and coal = 500 · 0.2 / 0.9
    (was 48.8 MW of unwanted electricity and 219.5 MW of coal)."""
    step = _chp(tmp_path_factory, "ml_pureheat",
                [demand("west", 0.0), demand("heat", 500.0),
                 ["unit__outputNode", [CHP, "west"], "conversion_flow_coeff",
                  b64(2.0, "float")],
                 ["unit__outputNode", [CHP, "heat"], "conversion_flow_coeff",
                  b64(0.2, "float")]] + uc(CHP, 0.4))
    assert cflow(step, CHP, "west").abs().max() == pytest.approx(0.0, abs=1e-6)
    coal = cflow(step, COAL, CHP)
    assert coal.to_list() == pytest.approx([500.0 * 0.2 / 0.9] * coal.len(),
                                           rel=1e-6)
    on = online(step, CHP)
    heat = cflow(step, CHP, "heat")
    assert ((heat - 0.4 * on) >= -1e-6).all()


# ── row structure ───────────────────────────────────────────────────────
@pytest.mark.parametrize("method,kind", [("linear", "linear"),
                                         ("binary", "integer")])
def test_one_minload_row_per_unit_and_timestep(tmp_path_factory, method,
                                               kind) -> None:
    """Exactly one ``minFlow_minload_<kind>`` row per ``(p, d, t)`` of the
    two-input unit; no input arc appears in it."""
    step = _chp(tmp_path_factory, f"ml_rows_{kind}",
                gas(100.0) + uc(CHP, 0.4, method), entities=GAS_ENTITIES)
    pb = build(step.flex_data)
    over = cstr_over(pb, f"minFlow_minload_{kind}")
    assert over is not None
    assert set(over.columns) == {"p", "d", "t"}
    assert over.filter(pl.col("p") == CHP).height == step.flex_data.dt.height
    assert over.select("p", "d", "t").is_unique().all()
    # no per-output row without an authored capacity_min_coeff
    assert cstr_over(pb, f"minFlow_output_floor_{kind}") is None


def test_sinkless_single_input_unit_floor(tmp_path_factory) -> None:
    """A sink-less unit (``dr_increase_demand``: input west, no output) is
    direct: its single input arc is its output measure, so the floor is
    ``in ≥ min_load · online`` — one row per ``(p, d, t)``."""
    step = solve(tmp_path_factory, alt="ml_sinkless",
                 base_chain=PLANT_CHAIN + ["dr_increase_demand"],
                 values=uc("dr_increase_demand", 0.5))
    over = cstr_over(build(step.flex_data), "minFlow_minload_linear")
    assert over is not None
    assert over.filter(pl.col("p") == "dr_increase_demand").height == \
        step.flex_data.dt.height
    inflow = flow(step, "dr_increase_demand", "west", "dr_increase_demand")
    on = online(step, "dr_increase_demand")
    assert ((inflow - 0.5 * on) >= -1e-6).all()


def test_sinkless_indirect_unit_without_output_is_internal_error() -> None:
    """A sink-less indirect online min-load unit cannot reach the floor
    emitter today (``_max_flow_rhs`` raises first); should that change it
    must not silently lose its floor."""
    from flextool.engine_polars.model import _assert_minload_units_have_outputs

    d = types.SimpleNamespace(
        process_indirect=pl.DataFrame({"p": ["mix"]}),
        process_source_sink=pl.DataFrame({
            "p": ["mix", "mix"], "source": ["a", "b"],
            "sink": ["mix", "mix"]}))
    on_ml = pl.LazyFrame({"_p": ["mix"]})
    empty = pl.DataFrame(schema={"p": pl.Utf8})
    with pytest.raises(AssertionError, match="internal error"):
        _assert_minload_units_have_outputs(d, on_ml, empty)
    # A unit whose outputs all have conversion_flow_coeff = 0 is skipped
    # (no floor; the validator warns).
    d.process_source_sink = pl.DataFrame({
        "p": ["mix", "mix"], "source": ["a", "mix"], "sink": ["mix", "z"]})
    _assert_minload_units_have_outputs(d, on_ml, empty)


# ── capacity_min_coeff: opt-in per-output floor ─────────────────────────
_EXTRACTION = [
    ["unit__outputNode", [CHP, "west"], "conversion_flow_coeff",
     b64(2.0, "float")],
    ["unit__outputNode", [CHP, "heat"], "conversion_flow_coeff",
     b64(0.2, "float")],
]


def test_per_output_floor_binds(tmp_path_factory) -> None:
    """``capacity_min_coeff = 1`` on west, west demand 0 / heat 500 (west
    may dump its excess cheaply): while online, west = 0.4 · online (the
    documented per-output floor binds), and the only per-output row is on
    (coal_chp, west)."""
    step = _chp(tmp_path_factory, "ml_floor_west",
                [demand("west", 0.0), demand("heat", 500.0),
                 ["node", "west", "penalty_down", b64(1.0, "float")]]
                + _EXTRACTION
                + [["unit__outputNode", [CHP, "west"], "capacity_min_coeff",
                    b64(1.0, "float")]] + uc(CHP, 0.4))
    west, on = cflow(step, CHP, "west"), online(step, CHP)
    assert (on > 1.0).all()
    assert (west - 0.4 * on).to_list() == pytest.approx(
        [0.0] * west.len(), abs=1e-6)
    over = cstr_over(build(step.flex_data), "minFlow_output_floor_linear")
    assert over is not None
    assert set(over.select("p", "source", "sink").unique().iter_rows()) == {
        (CHP, CHP, "west")}


def test_zero_coefficient_adds_no_per_output_row(tmp_path_factory) -> None:
    """Heat ``capacity_min_coeff = 0`` authored, west unauthored: no
    per-output row; the unit floor still applies to west + heat."""
    step = _chp(tmp_path_factory, "ml_floor_zero",
                [demand("west", 0.0), demand("heat", 500.0)] + _EXTRACTION
                + [["unit__outputNode", [CHP, "heat"], "capacity_min_coeff",
                    b64(0.0, "float")]] + uc(CHP, 0.4))
    assert cstr_over(build(step.flex_data),
                     "minFlow_output_floor_linear") is None
    assert cflow(step, CHP, "west").abs().max() == pytest.approx(0.0, abs=1e-6)


def test_all_zero_coefficients_keep_full_floor_and_warn(
        tmp_path_factory, caplog) -> None:
    """Every output ``capacity_min_coeff = 0`` used to switch the minimum
    load off; now the full unit floor applies and a warning tells the user
    to lower min_load instead."""
    caplog.set_level(logging.WARNING)
    step = _chp(tmp_path_factory, "ml_allzero",
                [["unit__outputNode", [CHP, k], "capacity_min_coeff",
                  b64(0.0, "float")] for k in ("west", "heat")]
                + uc(CHP, 0.4))
    over = cstr_over(build(step.flex_data), "minFlow_minload_linear")
    assert over is not None and over.filter(pl.col("p") == CHP).height > 0
    total = cflow(step, CHP, "west") + cflow(step, CHP, "heat")
    assert ((total - 0.4 * online(step, CHP)) >= -1e-6).all()
    assert any("no longer lowers" in r.getMessage() and CHP in r.getMessage()
               for r in caplog.records)


def test_direct_unit_coefficient_no_longer_softens(tmp_path_factory,
                                                   caplog) -> None:
    """Direct ``coal_plant`` (500 MW, linear UC, min_load 0.5, prohibitive
    startup cost keeps it online) with ``capacity_min_coeff = 0.5``: in a
    100 MW demand dip it still runs at the full floor 0.5 · 500 = 250 MW
    (the coefficient used to halve it to 125 MW), plus a warning."""
    caplog.set_level(logging.WARNING)
    dip = "t0010"
    step = solve(tmp_path_factory, alt="ml_direct_half",
                 base_chain=PLANT_CHAIN,
                 values=uc(PLANT, 0.5) + [
                     ["node", "west", "inflow", time_map({dip: -100.0},
                                                         -500.0)],
                     ["unit", PLANT, "startup_cost", b64(1.0e6, "float")],
                     ["unit__outputNode", [PLANT, "west"],
                      "capacity_min_coeff", b64(0.5, "float")]])
    out = flow(step, PLANT, COAL, "west")
    assert out[9] == pytest.approx(250.0, rel=1e-6)   # t0010
    assert online(step, PLANT)[9] == pytest.approx(500.0, rel=1e-6)
    assert any("no longer lowers" in r.getMessage() and PLANT in r.getMessage()
               for r in caplog.records)


def test_out_of_range_output_coefficient_is_an_error(tmp_path_factory) -> None:
    from flextool.engine_polars._solve_state import FlexToolConfigError

    url, scen = make_db(tmp_path_factory, alt="ml_cmin_bad",
                        base_chain=CHP_CHAIN,
                        values=no_ratio_constraint() + uc(CHP, 0.4) + [
                            ["unit__outputNode", [CHP, "west"],
                             "capacity_min_coeff", b64(2.0, "float")]])
    with pytest.raises(FlexToolConfigError, match="capacity_min_coeff"):
        run(tmp_path_factory, url, scen)


# ── rolling (warm path) ─────────────────────────────────────────────────
_ROLLING = [
    ["solve", "y2020_2day_dispatch", "solve_mode", b64("rolling_window",
                                                       "str")],
    ["solve", "y2020_2day_dispatch", "rolling_solve_horizon",
     b64(24.0, "float")],
    ["solve", "y2020_2day_dispatch", "rolling_solve_jump",
     b64(12.0, "float")],
]


def _roll_flows(steps: dict, unit: str, source: str, sink: str) -> list:
    out = []
    for step in steps.values():
        rows = (utf8(step.solution.value("v_flow"))
                .filter((pl.col("p") == unit) & (pl.col("source") == source)
                        & (pl.col("sink") == sink)).sort("d", "t"))
        out.append(rows["value"].to_list())
    return out


def test_rolling_warm_matches_cold(tmp_path_factory) -> None:
    """Rolling solve of the two-input UC unit: every roll keeps gas at 0
    and the warm-started rolls give the same flows as a cold chain."""
    from flextool.engine_polars import run_chain_from_db

    url, scen = make_db(tmp_path_factory, alt="ml_roll",
                        base_chain=CHP_CHAIN,
                        values=no_ratio_constraint() + gas(100.0)
                        + uc(CHP, 0.4) + _ROLLING,
                        entities=GAS_ENTITIES)
    warm = run_chain_from_db(url, scen, keep_solutions=True, warm=True,
                             work_folder=tmp_path_factory.mktemp("roll_w"))
    cold = run_chain_from_db(url, scen, keep_solutions=True, warm=False,
                             work_folder=tmp_path_factory.mktemp("roll_c"))
    assert len(warm) > 1
    for gas_roll in _roll_flows(warm, CHP, GAS, CHP):
        assert max(abs(v) for v in gas_roll) == pytest.approx(0.0, abs=1e-6)
    for arc in ((COAL, CHP), (CHP, "west"), (CHP, "heat")):
        for w, c in zip(_roll_flows(warm, CHP, *arc),
                        _roll_flows(cold, CHP, *arc)):
            assert w == pytest.approx(c, rel=1e-6, abs=1e-6)
