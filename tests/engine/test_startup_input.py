"""Start-up input: ``unit__inputNode.startup_input_method`` / ``startup_energy`` (database version 79).

Each start-up of a unit takes ``startup_energy`` [MWh per MW started] × MW started from the input node of the arc:
``startup_and_input`` keeps the arc's flow as well, ``startup_only`` has no flow (the readers hide the arc from every
arc set).  The draw is a term on ``v_startup`` (an event: no step duration) in the node's balance, the commodity
cost, the CO2 price and caps and the commodity ladder (``core._startup_draw``).

Hand-built model ``coal on oil`` (on ``tests/fixtures/tests.json``, alternative ``init`` only; never a checked-in
``.sqlite``): one period p2020, timeset ``2day`` cut to four 1-h steps t0001..t0004 (period share 4/8760, so a
``_d`` table is the 4-h sum × 2190).  Node ``elec`` with demand 0, 60, 60, 0 MW; commodity nodes ``coal_n`` (coal
10 €/MWh, 0.34 t/MWh) and ``oil_n`` (oil 50 €/MWh, 0.27 t/MWh); unit ``cp`` coal_n -> elec, existing 100 MW
(unitsize 100), efficiency 0.4, min_load 0.4 (keeps the unit off at t0001, where units online need no start-up), startup_method linear; arc (cp,
oil_n) ``startup_only`` with ``startup_energy`` 2 MWh/MW.

Hand values (specs/startup_input_design.md "Oracle", derived before the code): online 0.6 at t0002-t0003 (60 MW of
100), the start-up 0.6 × 100 = 60 MW at t0002 draws oil 2 × 60 = 120 MWh (all at t0002), 6 000 €, 32.4 t CO2; coal
60 / 0.4 = 150 MWh per step, 300 MWh, 3 000 €, 102 t; 9 000 € in the four hours.  Binary start-up: 1 unit, oil
200 MWh.  CO2 price 100 €/t: +3 240 € from oil, +10 200 € from coal.

CSP tank: turbine ``turb`` tank -> elec, 50 MW, efficiency 0.4, startup_cost 1 €/MW (pins the commitment) and
min_load 0.4 (keeps it off at t0001: units online at the first step need no start-up, and an online unit there
would have to produce into a node without demand), ``startup_and_input`` (turb, tank), startup_energy 0.5; tank storage 1000 MWh starting at 0.4 (fix_start), no
inflow; demand 20 MW at t0002-t0003: online 0.4, start-up 20 MW at t0002, draw 10 MWh; heat 20 / 0.4 = 50 MWh per
step; tank state 400, 340, 290, 290 MWh; the tank's reported input at t0002 is 50 + 10 = 60 MWh.

Each test names the table it reads: ``unit_inputNode_dt_ee`` is stored negated (consumption < 0) in MW (× 1 h =
MWh here), ``unit_startup_energy_dt_ee`` holds MWh in the step (positive), ``costs_dt_p`` is currency per step (not
annualized), ``process_co2_d_eee`` is t/a (annualized ÷ 4/8760); the objective is annualized (× 2190).
"""
from __future__ import annotations

import logging
import os
from pathlib import Path

import polars as pl
import pytest

from tests.engine._lp_rows import row_terms
from tests.engine._unit_db import FLEXTOOL_ROOT, RunCache, b64, build, make_db, time_map, utf8

pytestmark = pytest.mark.solver

SOLVE = "y2020_2day_dispatch"
STEPS = ["t0001", "t0002", "t0003", "t0004"]
ANNUAL = 8760.0 / 4.0
STOCH_JSON = FLEXTOOL_ROOT / "tests" / "fixtures" / "stochastics.json"

_RUNS = RunCache()


@pytest.fixture(scope="module", autouse=True)
def _free_runs():
    yield
    _RUNS.clear()


@pytest.fixture(autouse=True)
def _strict_autoscale(monkeypatch) -> None:
    """A missing autoscale registry entry raises instead of falling back to an unscaled LP."""
    monkeypatch.setenv("FLEXTOOL_AUTOSCALE_STRICT", "1")


def _f(x: float) -> list:
    return b64(float(x), "float")


def _s(x: str) -> list:
    return b64(x, "str")


def _pmap(values: dict[str, float]) -> list:
    return b64({"index_type": "str", "index_name": "period", "data": [[k, v] for k, v in values.items()]}, "map")


def _entities(extra=()) -> list:
    return [["node", "elec", None], ["node", "coal_n", None], ["node", "oil_n", None],
            ["commodity", "coal", None], ["commodity", "oil", None],
            ["commodity__node", ["coal", "coal_n"], None], ["commodity__node", ["oil", "oil_n"], None],
            ["unit", "cp", None], ["unit__inputNode", ["cp", "coal_n"], None],
            ["unit__inputNode", ["cp", "oil_n"], None], ["unit__outputNode", ["cp", "elec"], None], *extra]


def _values(*, method: str = "startup_only", startup: str = "linear", energy=2.0, coal_co2: float = 0.34,
            min_load: float | None = 0.4, extra=()) -> list:
    vals = [["timeset", "2day", "timeset_duration", b64({"index_type": "str", "data": [["t0001", 4.0]]}, "map")],
            ["solve", SOLVE, "solver_mip_gap", _f(0.0)],
            ["node", "elec", "node_type", _s("balance")],
            ["node", "elec", "inflow", time_map({"t0002": -60.0, "t0003": -60.0}, 0.0)],
            ["node", "coal_n", "node_type", _s("commodity")], ["node", "oil_n", "node_type", _s("commodity")],
            ["commodity", "coal", "price", _f(10.0)], ["commodity", "coal", "co2_content", _f(coal_co2)],
            ["commodity", "oil", "price", _f(50.0)], ["commodity", "oil", "co2_content", _f(0.27)],
            ["unit", "cp", "existing", _f(100.0)], ["unit", "cp", "efficiency", _f(0.4)],
            ["unit", "cp", "startup_method", _s(startup)],
            ["unit__inputNode", ["cp", "oil_n"], "startup_input_method", _s(method)]]
    if min_load is not None:
        vals.append(["unit", "cp", "min_load", _f(min_load)])
    if energy is not None:
        vals.append(["unit__inputNode", ["cp", "oil_n"], "startup_energy",
                     energy if isinstance(energy, list) else _f(energy)])
    return vals + list(extra)


#: Group co2g over coal_n and oil_n.
CO2_GROUP = [["group", "co2g", None], ["group__node", ["co2g", "coal_n"], None],
             ["group__node", ["co2g", "oil_n"], None]]


def _co2_price(price: float) -> list:
    return [["group", "co2g", "co2_method", _s("price")], ["group", "co2g", "co2_price", _f(price)]]


def _rolling(horizon: float, jump: float) -> list:
    return [["solve", SOLVE, "solve_mode", _s("rolling_window")],
            ["solve", SOLVE, "rolling_solve_horizon", _f(horizon)], ["solve", SOLVE, "rolling_solve_jump", _f(jump)]]


def _steps(tmp_path_factory, alt: str, values: list, entities: list, *, base_chain=("init",),
           fixture_json=None) -> tuple[dict, Path, str]:
    kwargs = {} if fixture_json is None else {"fixture_json": fixture_json}
    steps, wf, scen = _RUNS.get(tmp_path_factory, alt=alt, values=values, base_chain=list(base_chain),
                                entities=entities, **kwargs)
    assert steps and all(st.solution is not None for st in steps.values())
    return steps, wf, scen


def _step(tmp_path_factory, alt: str, values: list, entities: list):
    steps, _wf, _scen = _steps(tmp_path_factory, alt, values, entities)
    assert len(steps) == 1, list(steps)
    return next(iter(steps.values()))


_TABLES: dict[str, Path] = {}


def _table(tmp_path_factory, alt: str, values: list, entities: list, name: str):
    """The output table ``name`` (pandas, the scenario column level dropped) of the run: written once per run with
    the default output settings into a folder of its own (the cached work folder is only read)."""
    from flextool.engine.preprocess._flex_data_provider import FlexDataProvider
    from flextool.lean_parquet import read_lean_parquet
    from flextool.process_outputs.write_outputs import write_outputs

    steps, wf, scen = _steps(tmp_path_factory, alt, values, entities)
    if alt not in _TABLES:
        out = tmp_path_factory.mktemp(f"{alt}_out")
        last = list(steps.values())[-1]
        cwd = os.getcwd()
        try:
            os.chdir(out)
            write_outputs(scenario_name=scen, output_location=str(out), subdir=scen,
                          output_config_path=str(FLEXTOOL_ROOT / "flextool" / "schemas" / "default_plots.yaml"),
                          write_methods=["parquet"], fallback_output_location=str(out),
                          raw_output_dir=str(wf / "output_raw"), solution=last.solution, solve_name=last.solve_name,
                          flex_data_provider=last.flex_data_provider or FlexDataProvider(),
                          solve_steps=[(s.solve_name, s.flex_data, s.effective_solution) for s in steps.values()])
        finally:
            os.chdir(cwd)
        _TABLES[alt] = next(out.rglob("output_parquet")) / scen
    df = read_lean_parquet(_TABLES[alt] / f"{name}.parquet")
    return df.droplevel(0, axis=1) if df.columns.nlevels > 1 else df


def _per_step(df, column) -> list[float]:
    """The values of ``column`` at t0001..t0004 (rows indexed by (period, time))."""
    series = df[column]
    return [float(series.xs(t, level="time").iloc[0]) for t in STEPS]


def _var(step, name: str, p: str) -> list[float]:
    vals = utf8(step.solution.value(name)).filter(pl.col("p") == p).sort("t")
    assert vals["t"].to_list() == STEPS, vals
    return vals["value"].to_list()


# ---------------------------------------------------------------------------
# coal started on oil (startup_only)


def test_coal_started_on_oil(tmp_path_factory) -> None:
    """The start-up takes 120 MWh of oil at t0002 (unit_startup_energy_dt_ee, MWh in the step), reported as the
    unit's input from oil_n (unit_inputNode_dt_ee, stored negated); the oil arc has no flow and the unit stays a
    direct one-input unit; the objective is 9 000 € × 2190."""
    step = _step(tmp_path_factory, "su_oil", _values(), _entities())
    fd = step.flex_data
    assert _var(step, "v_startup_linear", "cp") == pytest.approx([0.0, 0.6, 0.0, 0.0], abs=1e-9)
    arcs = utf8(fd.process_source_sink).filter(pl.col("p") == "cp")
    assert sorted(arcs.select("source", "sink").rows()) == [("coal_n", "elec")]
    assert fd.process_indirect is None or "cp" not in utf8(fd.process_indirect)["p"].to_list()
    assert sorted(utf8(fd.process_startup_node).rows()) == [("cp", "oil_n")]
    assert float(step.obj) == pytest.approx(9000.0 * ANNUAL, rel=1e-9)
    energy = _table(tmp_path_factory, "su_oil", _values(), _entities(), "unit_startup_energy_dt_ee")
    assert _per_step(energy, ("cp", "oil_n")) == pytest.approx([0.0, 120.0, 0.0, 0.0], abs=1e-9)
    energy_d = _table(tmp_path_factory, "su_oil", _values(), _entities(), "unit_startup_energy_d_ee")
    assert float(energy_d[("cp", "oil_n")].iloc[0]) == pytest.approx(120.0 * ANNUAL, rel=1e-9)
    inputs = _table(tmp_path_factory, "su_oil", _values(), _entities(), "unit_inputNode_dt_ee")
    assert _per_step(inputs, ("cp", "oil_n")) == pytest.approx([0.0, -120.0, 0.0, 0.0], abs=1e-9)
    assert _per_step(inputs, ("cp", "coal_n")) == pytest.approx([0.0, -150.0, -150.0, 0.0], abs=1e-9)


def test_start_up_input_is_priced_and_emits(tmp_path_factory) -> None:
    """costs_dt_p (currency per step): commodity cost 1 500 (coal) + 6 000 (oil) at t0002, 1 500 at t0003;
    process_co2_d_eee (t/a): oil 32.4 t, coal 102 t in the four hours, × 2190."""
    costs = _table(tmp_path_factory, "su_oil", _values(), _entities(), "costs_dt_p")
    assert _per_step(costs, "commodity_cost") == pytest.approx([0.0, 7500.0, 1500.0, 0.0], abs=1e-6)
    co2 = _table(tmp_path_factory, "su_oil", _values(), _entities(), "process_co2_d_eee")
    assert float(co2[("unit", "cp", "oil", "oil_n")].iloc[0]) == pytest.approx(32.4 * ANNUAL, rel=1e-9)
    assert float(co2[("unit", "cp", "coal", "coal_n")].iloc[0]) == pytest.approx(102.0 * ANNUAL, rel=1e-9)


def test_results_database_has_the_start_up_energy(tmp_path_factory) -> None:
    """The results database gets unit__node (cp, oil_n) startup_energy_t (120 MWh at t0002) and
    startup_energy_annualized (120 × 2190), also for the startup_only arc, which has no flow arc."""
    import spinedb_api as api

    from flextool.engine.preprocess._flex_data_provider import FlexDataProvider
    from flextool.process_outputs.write_outputs import write_outputs

    steps, wf, scen = _steps(tmp_path_factory, "su_oil", _values(), _entities())
    out = tmp_path_factory.mktemp("su_oil_results_db")
    db_url = f"sqlite:///{out / 'results.sqlite'}"
    last = list(steps.values())[-1]
    cwd = os.getcwd()
    try:
        os.chdir(out)
        write_outputs(scenario_name=scen, output_location=str(out), subdir=scen,
                      output_config_path=str(FLEXTOOL_ROOT / "flextool" / "schemas" / "default_plots.yaml"),
                      write_methods=["spinedb"], fallback_output_location=str(out),
                      raw_output_dir=str(wf / "output_raw"), solution=last.solution, solve_name=last.solve_name,
                      flex_data_provider=last.flex_data_provider or FlexDataProvider(),
                      solve_steps=[(s.solve_name, s.flex_data, s.effective_solution) for s in steps.values()],
                      results_db_url=db_url)
    finally:
        os.chdir(cwd)
    with api.DatabaseMapping(db_url) as db:
        found = {pv["parameter_definition_name"]: api.from_database(pv["value"], pv["type"])
                 for pv in db.get_parameter_value_items(entity_class_name="unit__node")
                 if tuple(pv["entity_byname"]) == ("cp", "oil_n")}
    assert set(found) >= {"startup_energy_t", "startup_energy_annualized", "flow_t"}

    def numbers(value) -> list[float]:
        vals = getattr(value, "values", None)
        if vals is None:
            return [float(value)]
        return [x for v in vals for x in numbers(v)]

    assert numbers(found["startup_energy_t"]) == pytest.approx([0.0, 120.0, 0.0, 0.0], abs=1e-9)
    assert numbers(found["startup_energy_annualized"]) == pytest.approx([120.0 * ANNUAL], rel=1e-9)


def test_binary_start_up_takes_whole_units(tmp_path_factory) -> None:
    """With binary start-up one whole unit starts: 200 MWh of oil at t0002 (unit_startup_energy_dt_ee)."""
    values = _values(startup="binary")
    step = _step(tmp_path_factory, "su_oil_bin", values, _entities())
    assert float(step.obj) == pytest.approx((3000.0 + 10000.0) * ANNUAL, rel=1e-9)
    energy = _table(tmp_path_factory, "su_oil_bin", values, _entities(), "unit_startup_energy_dt_ee")
    assert _per_step(energy, ("cp", "oil_n")) == pytest.approx([0.0, 200.0, 0.0, 0.0], abs=1e-9)


@pytest.mark.parametrize("coal_co2, extra", [(0.34, 3240.0 + 10200.0), (0.0, 3240.0)])
def test_co2_price_of_the_start_up_input(tmp_path_factory, coal_co2, extra) -> None:
    """CO2 price 100 €/t on a group over coal_n and oil_n: the objective grows by 3 240 € (oil) + 10 200 € (coal)
    in the four hours (× 2190); with coal_co2 0 the start-up draw is the group's only emitter and still priced
    (+3 240 €, costs_dt_p ``co2`` 3 240 at t0002)."""
    base = _step(tmp_path_factory, f"su_co2_none_{coal_co2}", _values(coal_co2=coal_co2), _entities(CO2_GROUP))
    priced_values = _values(coal_co2=coal_co2, extra=_co2_price(100.0))
    priced = _step(tmp_path_factory, f"su_co2_price_{coal_co2}", priced_values, _entities(CO2_GROUP))
    assert float(priced.obj) - float(base.obj) == pytest.approx(extra * ANNUAL, rel=1e-9)
    if coal_co2 == 0.0:
        costs = _table(tmp_path_factory, f"su_co2_price_{coal_co2}", priced_values, _entities(CO2_GROUP),
                       "costs_dt_p")
        assert _per_step(costs, "co2") == pytest.approx([0.0, 3240.0, 0.0, 0.0], abs=1e-6)


def test_co2_cap_binds_on_the_start_up_input_alone(tmp_path_factory) -> None:
    """coal_co2 0 and a co2_max_period of 35 000 t/a on the group: the oil draw is the only emitter, so the cap
    allows start-ups of 35 000 / (2190 · 2 · 100 · 0.27) = 0.29596 units (all at t0002, where they serve both
    hours) instead of 0.6."""
    values = _values(coal_co2=0.0, extra=[["group", "co2g", "co2_method", _s("period")],
                                          ["group", "co2g", "co2_max_period", _f(35000.0)]])
    step = _step(tmp_path_factory, "su_co2_cap", values, _entities(CO2_GROUP))
    assert _var(step, "v_startup_linear", "cp") == pytest.approx([0.0, 35000.0 / (ANNUAL * 2 * 100 * 0.27), 0.0, 0.0],
                                                                 abs=1e-7)


# ---------------------------------------------------------------------------
# CSP turbine started on tank heat (startup_and_input)


def _csp_entities() -> list:
    return [["node", "elec", None], ["node", "tank", None], ["unit", "turb", None],
            ["unit__inputNode", ["turb", "tank"], None], ["unit__outputNode", ["turb", "elec"], None]]


def _csp_values() -> list:
    return [["timeset", "2day", "timeset_duration", b64({"index_type": "str", "data": [["t0001", 4.0]]}, "map")],
            ["node", "elec", "node_type", _s("balance")],
            ["node", "elec", "inflow", time_map({"t0002": -20.0, "t0003": -20.0}, 0.0)],
            ["node", "tank", "node_type", _s("storage")], ["node", "tank", "existing", _f(1000.0)],
            ["node", "tank", "storage_start_end_method", _s("fix_start")],
            ["node", "tank", "storage_state_start", _f(0.4)],
            ["unit", "turb", "existing", _f(50.0)], ["unit", "turb", "efficiency", _f(0.4)],
            ["unit", "turb", "startup_method", _s("linear")], ["unit", "turb", "startup_cost", _f(1.0)],
            ["unit", "turb", "min_load", _f(0.4)],
            ["unit__inputNode", ["turb", "tank"], "startup_input_method", _s("startup_and_input")],
            ["unit__inputNode", ["turb", "tank"], "startup_energy", _f(0.5)]]


def test_csp_turbine_starts_on_tank_heat(tmp_path_factory) -> None:
    """The start-up of 20 MW takes 10 MWh of heat at t0002 (unit_startup_energy_dt_ee): node_state_dt_e (MWh)
    400, 340, 290, 290; the tank's input reported in unit_inputNode_dt_ee (negated, MW = MWh in 1 h) is the flow
    plus the draw, 60 MWh at t0002 and 50 at t0003."""
    step = _step(tmp_path_factory, "su_csp", _csp_values(), _csp_entities())
    assert _var(step, "v_startup_linear", "turb") == pytest.approx([0.0, 0.4, 0.0, 0.0], abs=1e-9)
    state = _table(tmp_path_factory, "su_csp", _csp_values(), _csp_entities(), "node_state_dt_e")
    assert _per_step(state, "tank") == pytest.approx([400.0, 340.0, 290.0, 290.0], abs=1e-6)
    energy = _table(tmp_path_factory, "su_csp", _csp_values(), _csp_entities(), "unit_startup_energy_dt_ee")
    assert _per_step(energy, ("turb", "tank")) == pytest.approx([0.0, 10.0, 0.0, 0.0], abs=1e-9)
    inputs = _table(tmp_path_factory, "su_csp", _csp_values(), _csp_entities(), "unit_inputNode_dt_ee")
    assert _per_step(inputs, ("turb", "tank")) == pytest.approx([0.0, -60.0, -50.0, 0.0], abs=1e-6)


def test_tank_balance_reports_the_start_up_heat(tmp_path_factory) -> None:
    """node_dt_ep (MW): the tank's "To units" holds the flow and the draw (−60 at t0002, −50 at t0003).  Two
    heaters elec -> tank without capacity make the tank a multi-use storage node, which the balance table shows."""
    ents = _csp_entities() + [e for h in ("h1", "h2") for e in (["unit", h, None],
                                                              ["unit__inputNode", [h, "elec"], None],
                                                              ["unit__outputNode", [h, "tank"], None])]
    values = _csp_values() + [["unit", h, "existing", _f(0.0)] for h in ("h1", "h2")]
    node = _table(tmp_path_factory, "su_csp_hub", values, ents, "node_dt_ep")
    assert _per_step(node, ("tank", "To units")) == pytest.approx([0.0, -60.0, -50.0, 0.0], abs=1e-6)


def test_ladder_commodity_counts_the_start_up_input(tmp_path_factory) -> None:
    """Oil on a cumulative price ladder (one unlimited tier at 50 €/MWh): commodity_ladder_balance of
    (oil, oil_n, p2020) carries the draw, startup_energy · unitsize · timestep_weight = 200 per unit started, at
    every start-up column; the cost is the flat-price model's (9 000 € × 2190)."""
    import math

    tiers = {"index_type": "str", "rank": 2, "index_name": "tier",
             "data": [["1", {"type": "map", "index_type": "str", "rank": 1,
                             "data": [["price", 50.0], ["quantity", math.inf]]}]]}
    values = [v for v in _values() if v[:3] != ["commodity", "oil", "price"]]
    values += [["commodity", "oil", "price_method", _s("price_ladder_cumulative")],
               ["commodity", "oil", "price_ladder_cumulative", b64(tiers, "map")]]
    step = _step(tmp_path_factory, "su_ladder", values, _entities())
    assert float(step.obj) == pytest.approx(9000.0 * ANNUAL, rel=1e-9)
    rows = row_terms(build(step.flex_data), "commodity_ladder_balance", "oil,")
    assert list(rows) == ["commodity_ladder_balance[oil,oil_n,p2020]"]
    startup = {c: abs(v) for c, v in rows["commodity_ladder_balance[oil,oil_n,p2020]"].items()
               if c.startswith("v_startup")}
    assert startup == {f"v_startup_linear[cp,p2020,{t}]": pytest.approx(200.0) for t in STEPS}


# ---------------------------------------------------------------------------
# constant vs period map: the same LP in single, rolling and stochastic solves


def _lp(pb) -> dict:
    """Every coefficient, bound and objective cost of a built problem, by row and column name."""
    m = pb.canonicalise()
    coefs = {}
    for c in range(m.n_cols):
        for k in range(m.col_ptr[c], m.col_ptr[c + 1]):
            coefs[(m.row_names[m.row_idx[k]], m.col_names[c])] = float(m.val[k])
    return {"coefs": coefs,
            "rows": {m.row_names[i]: (float(m.row_lb[i]), float(m.row_ub[i])) for i in range(m.n_rows)},
            "cols": {m.col_names[c]: (float(m.col_lb[c]), float(m.col_ub[c]), float(m.col_obj[c]))
                     for c in range(m.n_cols)}}


def _stoch_entities() -> list:
    return [["node", "fuel_n", None], ["node", "oil_n", None], ["commodity", "fuelc", None],
            ["commodity", "oil", None], ["commodity__node", ["fuelc", "fuel_n"], None],
            ["commodity__node", ["oil", "oil_n"], None], ["unit", "su_u", None],
            ["unit__inputNode", ["su_u", "fuel_n"], None], ["unit__inputNode", ["su_u", "oil_n"], None],
            ["unit__outputNode", ["su_u", "demand_node"], None]]


def _stoch_values(energy) -> list:
    return [["node", "fuel_n", "node_type", _s("commodity")], ["node", "oil_n", "node_type", _s("commodity")],
            ["commodity", "fuelc", "price", _f(10.0)], ["commodity", "oil", "price", _f(50.0)],
            ["unit", "su_u", "existing", _f(100.0)], ["unit", "su_u", "efficiency", _f(0.4)],
            ["unit", "su_u", "startup_method", _s("linear")],
            ["unit__inputNode", ["su_u", "oil_n"], "startup_input_method", _s("startup_only")],
            ["unit__inputNode", ["su_u", "oil_n"], "startup_energy", energy]]


@pytest.mark.parametrize("case", ["single", "rolling", "stochastic", "rolling_stochastic"])
def test_period_map_equals_constant(tmp_path_factory, case) -> None:
    """A period map of startup_energy that repeats the constant builds the same LP in every solve, roll and
    forecast (the draw's coefficients included)."""
    lps = {}
    for kind, energy in (("const", _f(2.0)), ("map", None)):
        if case in ("single", "rolling"):
            extra = _rolling(3.0, 2.0) if case == "rolling" else []
            values = _values(energy=_pmap({"p2020": 2.0}) if energy is None else energy, extra=extra)
            steps, _wf, _scen = _steps(tmp_path_factory, f"su_cm_{case}_{kind}", values, _entities())
        else:
            chain = ["base", "init", "system", "2day"] if case == "stochastic" else \
                ["init", "base", "system", "1week_rolling"]
            values = _stoch_values(_pmap({"period1": 2.0}) if energy is None else energy)
            steps, _wf, _scen = _steps(tmp_path_factory, f"su_cm_{case}_{kind}", values, _stoch_entities(),
                                       base_chain=chain, fixture_json=STOCH_JSON)
        assert all(st.flex_data.p_startup_energy is not None for st in steps.values())
        lps[kind] = [_lp(build(st.flex_data)) for st in steps.values()]
    assert len(lps["const"]) == len(lps["map"]) and lps["const"]
    if case in ("rolling", "rolling_stochastic"):
        assert len(lps["const"]) > 1
    for const, mapped in zip(lps["const"], lps["map"]):
        assert mapped == const


# ---------------------------------------------------------------------------
# rolling CO2 carry, coarse steps, reduce levels


def test_rolling_co2_carry_counts_the_start_up_input(tmp_path_factory) -> None:
    """Rolling solve, 3-h horizon, 2-h jump: the start-up at t0002 is in roll 1's realized part, so the CO2 carried
    into roll 2 for the capped group over coal_n and oil_n is oil 32.4 t + coal 150 MWh · 0.34 = 51 t = 83.4 t."""
    values = _values(extra=[["group", "co2g", "co2_method", _s("period")],
                            ["group", "co2g", "co2_max_period", _f(1e9)], *_rolling(3.0, 2.0)])
    steps, _wf, _scen = _steps(tmp_path_factory, "su_roll_co2", values, _entities(CO2_GROUP))
    rolls = list(steps.values())
    assert len(rolls) == 2
    assert _var_roll(rolls[0], "v_startup_linear", "cp")["t0002"] == pytest.approx(0.6, abs=1e-9)
    carried = utf8(rolls[1].flex_data.p_co2_cum_realized_tonnes.frame)
    assert carried.filter((pl.col("g") == "co2g") & (pl.col("d") == "p2020"))["value"].to_list() == \
        pytest.approx([32.4 + 51.0], rel=1e-9)


def _var_roll(step, name: str, p: str) -> dict[str, float]:
    vals = utf8(step.solution.value(name)).filter(pl.col("p") == p)
    return dict(vals.select("t", "value").iter_rows())


def test_coarse_unit_draws_at_the_first_fine_step(tmp_path_factory) -> None:
    """elec in a 2-h new_stepduration group: cp is coarse (the start-up node oil_n, a fine balance node, does
    not count); the start-up columns are at t0001 and t0003, and oil_n's nodeBalance_eq has the draw coefficient
    −startup_energy · unitsize = −200 only at those steps."""
    ents = _entities([["group", "two_h", None], ["group__node", ["two_h", "elec"], None]])
    values = [v for v in _values() if v[:3] != ["node", "oil_n", "node_type"]]
    values += [["group", "two_h", "new_stepduration", _f(2.0)], ["node", "oil_n", "node_type", _s("balance")]]
    step = _step(tmp_path_factory, "su_coarse", values, ents)
    fd = step.flex_data
    assert "cp" in utf8(fd.processCoarse)["p"].to_list()
    rows = row_terms(build(fd), "nodeBalance_eq", "oil_n,")
    assert sorted(rows) == [f"nodeBalance_eq[oil_n,p2020,{t}]" for t in STEPS]
    for t in STEPS:
        startup = {c: v for c, v in rows[f"nodeBalance_eq[oil_n,p2020,{t}]"].items() if c.startswith("v_startup")}
        assert startup == ({f"v_startup_linear[cp,p2020,{t}]": pytest.approx(-200.0)} if t in ("t0001", "t0003")
                           else {})


def test_online_level_drops_the_draw(tmp_path_factory) -> None:
    """reduce_unit_detail_method online leaves out the start-ups and with them the draw (removed["startup_energy"]);
    the LP has no start-up column."""
    from flextool.engine.preprocess._reduce_detail import apply_reduce_detail

    fd = _step(tmp_path_factory, "su_oil", _values(), _entities()).flex_data
    reduced, report = apply_reduce_detail(fd, "online")
    assert report.removed["startup_energy"] == ["cp"]
    assert reduced.process_startup_node is None and reduced.p_startup_energy is None
    m = build(reduced).canonicalise()
    assert not [c for c in m.col_names if c.startswith("v_startup")]


def test_build_refuses_a_draw_without_online_variables(tmp_path_factory) -> None:
    """Hand-built data: a start-up node of a unit without online variables is an error of the build (the input
    derivation checks the database)."""
    import dataclasses

    from flextool.engine.input._solve_state import FlexToolConfigError

    fd = _step(tmp_path_factory, "su_oil", _values(), _entities()).flex_data
    bad = dataclasses.replace(fd, process_online=None, process_online_linear=None, p_online_dt=None)
    bad._axis_enums = getattr(fd, "_axis_enums", None)
    with pytest.raises(FlexToolConfigError, match="startup_input_method needs online variables.*: cp"):
        build(bad)


def test_integers_keeps_the_online_variables_of_a_draw(tmp_path_factory) -> None:
    """Binary cp without min_load: the draw is its only use of the online variables, so integers keeps them
    (relaxed) instead of leaving them out as unused."""
    from flextool.engine.preprocess._reduce_detail import apply_reduce_detail, online_uses

    fd = _step(tmp_path_factory, "su_oil_bin_free", _values(startup="binary", min_load=None), _entities()).flex_data
    assert online_uses(fd) == {"startup_energy": {"cp"}}
    reduced, report = apply_reduce_detail(fd, "integers")
    assert "online_unused" not in report.removed
    assert utf8(reduced.process_online_integer)["p"].to_list() == ["cp"]
    assert reduced.relax_integrality is True


# ---------------------------------------------------------------------------
# readers: the startup_only arc is hidden, its start-up values are kept


def test_readers_hide_the_startup_only_arc(tmp_path_factory) -> None:
    """SpineDbReader and SpineDBBackend leave the startup_only arc out of the entities and of its other parameters
    (a conversion_flow_coeff authored on it); startup_arcs() lists both start-up roles; a period-map
    startup_energy on the hidden arc reads as (p, n, d) rows naming the unit and the node."""
    from flextool.engine.input._spinedb_reader import SpineDbReader
    from flextool.engine.preprocess.params._direct import p_startup_energy_from_source
    from flextool.spinedb_backend import SpineDBBackend

    ents = _entities([["node", "heat_n", None], ["unit__inputNode", ["cp", "heat_n"], None]])
    values = _values(energy=_pmap({"p2020": 2.0}),
                     extra=[["unit__inputNode", ["cp", "oil_n"], "conversion_flow_coeff", _f(0.0)],
                            ["unit__inputNode", ["cp", "heat_n"], "startup_input_method", _s("startup_and_input")],
                            ["unit__inputNode", ["cp", "heat_n"], "startup_energy", _f(1.0)]])
    url, scen = make_db(tmp_path_factory, alt="su_reader", values=values, base_chain=["init"], entities=ents)
    reader = SpineDbReader(url, scen)
    arcs = utf8(reader.entities("unit__inputNode")).filter(pl.col("unit") == "cp")
    assert sorted(arcs.rows()) == [("cp", "coal_n"), ("cp", "heat_n")]
    assert sorted(reader.startup_arcs().rows()) == [("cp", "heat_n", "startup_and_input"),
                                                    ("cp", "oil_n", "startup_only")]
    coeff = utf8(reader.parameter("unit__inputNode", "conversion_flow_coeff"))
    assert coeff.filter(pl.col("node") == "oil_n").height == 0
    dt = pl.DataFrame({"d": ["p2020"], "t": ["t0001"]})
    energy = utf8(p_startup_energy_from_source(reader, period_filter=dt).frame.lazy().collect())
    assert sorted(energy.select("p", "n", "d", "value").rows()) == [("cp", "heat_n", "p2020", 1.0),
                                                                     ("cp", "oil_n", "p2020", 2.0)]
    with SpineDBBackend(url, scen) as backend:
        inputs = backend.entities(classes=["unit__inputNode"], header="unit,node")
        assert sorted(tuple(r) for r in inputs.filter(pl.col("unit") == "cp").rows()) == [("cp", "coal_n"),
                                                                                         ("cp", "heat_n")]
        assert sorted(e["entity_byname"][1] for e in backend.find_entities(entity_class_name="unit__inputNode")
                      if e["entity_byname"][0] == "cp") == ["coal_n", "heat_n"]


def test_inmemory_reader_lists_the_start_up_arcs() -> None:
    """InMemoryReader hides nothing; startup_arcs() reads the startup_input_method frame (or the frame given)."""
    from flextool.engine.input._inmemory_reader import InMemoryReader

    method = pl.DataFrame({"unit": ["u", "u", "v"], "node": ["a", "b", "c"],
                           "value": ["just_input", "startup_only", "startup_and_input"]})
    reader = InMemoryReader({}, {("unit__inputNode", "startup_input_method"): method})
    assert reader.startup_arcs().rows() == [("u", "b", "startup_only"), ("v", "c", "startup_and_input")]
    given = pl.DataFrame({"unit": ["w"], "node": ["d"], "method": ["startup_only"]})
    assert InMemoryReader({}, {}, startup_arcs=given).startup_arcs().rows() == [("w", "d", "startup_only")]
    assert InMemoryReader({}, {}).startup_arcs().height == 0


# ---------------------------------------------------------------------------
# input checks


class _Provider:
    """The one input_derivation frame the start-up checks read (input/process_method)."""

    def __init__(self, methods: dict[str, str]) -> None:
        self._frame = pl.DataFrame({"process": list(methods), "method": list(methods.values())})

    def has(self, key: str) -> bool:
        return key == "input/process_method"

    def get(self, key: str) -> pl.DataFrame:
        return self._frame


def _check(tmp_path_factory, alt: str, values: list, entities: list, methods: dict[str, str], check):
    """``check(db, provider)`` on the scenario-filtered database of the model."""
    import spinedb_api as api

    from flextool.update_flextool.db_handle import open_database_mapping

    url, scen = make_db(tmp_path_factory, alt=alt, values=values, base_chain=["init"], entities=entities)
    with open_database_mapping(url) as db:
        db.fetch_all("entity")
        db.fetch_all("parameter_value")
        api.filters.scenario_filter.scenario_filter_from_dict(
            db, api.filters.scenario_filter.scenario_filter_config(scen))
        return check(db, _Provider(methods))


ONLINE = {"cp": "method_1way_1var_LP"}


@pytest.mark.parametrize("case, values, ents, methods, message", [
    ("E1", [["unit__inputNode", ["cp", "coal_n"], "startup_energy", _f(1.0)]], [], ONLINE,
     r"startup_energy is set on arc\(s\) with startup_input_method just_input.*\(cp, coal_n\)"),
    ("E2", [], [], {"cp": "method_1way_1var_off"}, r"needs online variables, but these units have none: cp"),
    ("E3", [["unit__inputNode", ["cp", "oil_n"], "constraint_flow_coeff",
             b64({"index_type": "str", "index_name": "constraint", "data": [["c1", 1.0]]}, "map")]], [], ONLINE,
     r"constraint_flow_coeff is set on startup_only arc\(s\).*\(cp, oil_n\)"),
    ("E4", [["unit__inputNode", ["cp", "oil_n"], "startup_energy", _f(-1.0)]], [], ONLINE,
     r"startup_energy must be 0 or above: \(cp, oil_n\)"),
    ("E5", [], [["profile", "prof", None], ["unit__node__profile", ["cp", "oil_n", "prof"], None]], ONLINE,
     r"name a startup_only arc.*unit__node__profile \(cp, oil_n, prof\)"),
])
def test_input_errors(tmp_path_factory, case, values, ents, methods, message) -> None:
    from flextool.engine.input._solve_state import FlexToolConfigError
    from flextool.input_derivation._validators import validate_startup_inputs

    logger = logging.getLogger("su_check")
    vals = [v for v in _values() if not (case == "E4" and v[2] == "startup_energy")] + values
    with pytest.raises(FlexToolConfigError, match=message):
        _check(tmp_path_factory, f"su_err_{case}", vals, _entities(ents), methods,
               lambda db, prov: validate_startup_inputs(db, prov, logger))


def test_input_error_stops_the_run(tmp_path_factory) -> None:
    """The check runs in the input derivation: startup_method no_startup with a start-up role stops the run."""
    from flextool.engine import run_chain_from_db
    from flextool.engine.input._solve_state import FlexToolConfigError

    url, scen = make_db(tmp_path_factory, alt="su_err_run", values=_values(startup="no_startup", min_load=None),
                        base_chain=["init"], entities=_entities())
    with pytest.raises(FlexToolConfigError, match="needs online variables, but these units have none: cp"):
        run_chain_from_db(url, scen, work_folder=tmp_path_factory.mktemp("su_err_run_w"))


@pytest.mark.parametrize("case, values, ents, methods, message", [
    ("W1", [], [], ONLINE, "startup_input_method startup_only without a startup_energy"),
    ("W2", [["unit__inputNode", ["cp", "oil_n"], "conversion_flow_coeff", _f(0.0)],
            ["unit__inputNode", ["cp", "oil_n"], "input_share_max", _f(0.5)]], [], ONLINE,
     "conversion_flow_coeff, input_share_max has no effect on a startup_only arc"),
    ("W3", [], [], ONLINE, "has no energy balance and no commodity"),
    ("W4", [["unit__inputNode", ["gt", "oil_n"], "startup_input_method", _s("startup_only")],
            ["unit__inputNode", ["gt", "oil_n"], "startup_energy", _f(1.0)]],
     [["unit", "gt", None], ["unit__inputNode", ["gt", "oil_n"], None], ["unit__outputNode", ["gt", "elec"], None]],
     {**ONLINE, "gt": "method_1way_1var_LP"}, "unit 'gt': every input is startup_only"),
    ("W5", [["unit__inputNode", ["cp", "oil_n"], "constraint_flow_coeff",
             b64({"index_type": "str", "index_name": "constraint", "data": [["c1", 1.0]]}, "map")]], [], ONLINE,
     "the start-up input is not in constraint c1"),
    ("W6", [["unit__inputNode", ["gt", "oil_n"], "startup_input_method", _s("startup_only")],
            ["unit__inputNode", ["gt", "oil_n"], "startup_energy", _f(1.0)],
            ["unit", "gt", "conversion_method", _s("min_load_efficiency")]],
     [["unit", "gt", None], ["unit__inputNode", ["gt", "oil_n"], None], ["unit__outputNode", ["gt", "elec"], None]],
     {**ONLINE, "gt": "method_1way_1var_LP"}, "min_load_efficiency with only startup_only inputs"),
])
def test_input_warnings(tmp_path_factory, caplog, case, values, ents, methods, message) -> None:
    from flextool.input_derivation._validators import validate_startup_inputs

    vals = _values(energy=None if case == "W1" else 2.0,
                   method="startup_and_input" if case == "W5" else "startup_only")
    # W3: oil_n is a commodity node of no commodity, so the start-up energy goes nowhere.
    ents_all = [e for e in _entities(ents) if case != "W3" or e[:2] != ["commodity__node", ["oil", "oil_n"]]]
    logger = logging.getLogger("su_check")
    with caplog.at_level(logging.WARNING, logger="su_check"):
        _check(tmp_path_factory, f"su_warn_{case}", vals + values, ents_all, methods,
               lambda db, prov: validate_startup_inputs(db, prov, logger))
    hits = [r.getMessage() for r in caplog.records if message in r.getMessage()]
    assert len(hits) == 1, [r.getMessage() for r in caplog.records]


def test_arc_checks_skip_the_startup_only_arc(tmp_path_factory, capsys, caplog) -> None:
    """#213 (validate_zero_conversion_coeff) and the input-share checks leave the startup_only arc out: a
    conversion_flow_coeff 0 and an input_share_min on it give no zero-coefficient or input-share line."""
    from flextool.input_derivation._validators import validate_unit_input_shares, validate_zero_conversion_coeff

    values = _values(extra=[["unit__inputNode", ["cp", "oil_n"], "conversion_flow_coeff", _f(0.0)],
                            ["unit__inputNode", ["cp", "oil_n"], "input_share_min", _f(0.3)]])
    logger = logging.getLogger("su_check")

    def checks(db, prov) -> None:
        validate_zero_conversion_coeff(db)
        validate_unit_input_shares(db, prov, logger)

    with caplog.at_level(logging.WARNING, logger="su_check"):
        _check(tmp_path_factory, "su_skip", values, _entities(), ONLINE, checks)
    printed = capsys.readouterr().out
    assert "oil_n" not in printed
    assert not [r for r in caplog.records if "oil_n" in r.getMessage()]


# ---------------------------------------------------------------------------
# Benders: the draw is a term on the units started, which live in the region subproblem


def test_straddle_check_counts_the_start_up_node() -> None:
    """A region unit whose start-up node is master-hosted straddles the region / master boundary, although none of
    its arcs touches the master."""
    from types import SimpleNamespace

    from flextool.engine.solve._region_filter import _validate_no_straddling_units

    data = SimpleNamespace(
        process_source_sink=pl.DataFrame({"p": ["u"], "source": ["fuel_A"], "sink": ["elec_A"]}),
        process_unit=pl.DataFrame({"p": ["u"]}), process_indirect=None,
        process_startup_node=pl.DataFrame({"p": ["u"], "n": ["oil_M"]}))
    with pytest.raises(RuntimeError, match=r"unit 'u' straddles.*oil_M"):
        _validate_no_straddling_units(data, {"fuel_A", "elec_A"}, frozenset({"oil_M"}))
    data.process_startup_node = pl.DataFrame({"p": ["u"], "n": ["oil_A"]})
    _validate_no_straddling_units(data, {"fuel_A", "elec_A", "oil_A"}, frozenset({"oil_M"}))


def test_start_up_node_of_another_region_is_refused() -> None:
    """A unit of region A that takes start-up energy from a node of region B cannot be split (the draw would be in
    B's balance, the start-ups in A's subproblem); a node of no region (a commodity node) is fine."""
    from types import SimpleNamespace

    from flextool.engine.solve._region_filter import _validate_startup_nodes

    nodes, procs = {"A": {"elec_A"}, "B": {"heat_B"}}, {"A": {"u"}, "B": {"v"}}
    data = SimpleNamespace(process_startup_node=pl.DataFrame({"p": ["u"], "n": ["heat_B"]}))
    with pytest.raises(RuntimeError, match=r"unit 'u' takes start-up energy from node 'heat_B' of region 'B'"):
        _validate_startup_nodes(data, nodes, procs)
    data.process_startup_node = pl.DataFrame({"p": ["u", "u"], "n": ["elec_A", "fuel"]})
    _validate_startup_nodes(data, nodes, procs)


@pytest.fixture(scope="module")
def lh2_with_startup(scenario_run):
    """The lh2_three_region_trade_invest FlexData (regions A / B / C share the commodity node coal_market) with
    coal_A given linear online variables and a start-up draw of 2 MWh/MW from coal_market."""
    import dataclasses

    from polar_high import Param

    from tests._support.load import load_scenario

    data = load_scenario(scenario_run("lh2_three_region_trade_invest", db_fixture="lh2_trade_invest"))
    p_dtype = data.process_source_sink.schema["p"]
    n_dtype = data.process_source_sink.schema["source"]

    def procs(names) -> pl.DataFrame:
        return pl.DataFrame({"p": names}).with_columns(pl.col("p").cast(p_dtype))

    dt = data.dt.select("d", "t")
    online_dt = dt.with_columns(p=pl.lit("coal_A").cast(p_dtype)).select("p", "d", "t")
    pn = pl.DataFrame({"p": ["coal_A"], "n": ["coal_market"]}).with_columns(pl.col("p").cast(p_dtype),
                                                                           pl.col("n").cast(n_dtype))
    energy = pn.join(dt.select("d").unique(), how="cross").with_columns(value=pl.lit(2.0))
    commodity = pn.with_columns(c=pl.lit("coal").cast(data.flow_from_commodity_eff.schema["c"]))
    new = dataclasses.replace(data, process_online=procs(["coal_A"]), process_online_linear=procs(["coal_A"]),
                              p_online_dt=online_dt, process_startup_node=pn,
                              p_startup_energy=Param(("p", "n", "d"), energy), startup_from_commodity=commodity)
    new._axis_enums = getattr(data, "_axis_enums", None)
    return new


def _startup_costs(pb) -> dict[str, float]:
    m = pb.canonicalise()
    return {m.col_names[c]: float(m.col_obj[c]) for c in range(m.n_cols)
            if m.col_names[c].startswith("v_startup_linear[coal_A,")}


def test_benders_region_keeps_its_units_draw(lh2_with_startup) -> None:
    """The split gives coal_A's start-up draw from the shared commodity node to region A only (the other regions and
    the master have no start-up node), and region A prices the draw as the whole-system LP does."""
    from flextool.engine.solve._region_filter import master_network_data, split

    data = lh2_with_startup
    regions = ["region_A", "region_B", "region_C"]
    splits = {s.region: s.data for s in split(data, regions=regions, benders_uncap_cross_region=True)}
    assert sorted(utf8(splits["region_A"].process_startup_node).rows()) == [("coal_A", "coal_market")]
    assert splits["region_B"].process_startup_node is None and splits["region_C"].process_startup_node is None
    master = master_network_data(data, regions)
    assert all(getattr(master, f) is None for f in ("process_startup_node", "p_startup_energy",
                                                    "startup_from_commodity"))
    whole = _startup_costs(build(data))
    region_a = _startup_costs(build(splits["region_A"]))
    assert whole and all(v > 0.0 for v in whole.values())
    assert region_a == pytest.approx(whole, rel=1e-12)


# ---------------------------------------------------------------------------
# node-group balance views


def test_node_group_view_shows_the_start_up_input(tmp_path_factory) -> None:
    """nodeGroup_flows_dt_gpe (MWh in the step, stored negated for flows to units): a dispatch group over the tank
    shows the turbine's input as flow + draw (−60 at t0002, startup_and_input); a dispatch group over oil_n (a
    balance node here, upward slack at 100 €/MWh) shows the startup_only draw of cp (−120 at t0002), which its
    loss of load covers."""
    ents = _csp_entities() + [["group", "heatg", None], ["group__node", ["heatg", "tank"], None]]
    values = _csp_values() + [["group", "heatg", "print_dispatch", _s("yes")]]
    heat = _table(tmp_path_factory, "su_csp_group", values, ents, "nodeGroup_flows_dt_gpe")
    assert _per_step(heat, ("heatg", "to_unit", "turb")) == pytest.approx([0.0, -60.0, -50.0, 0.0], abs=1e-6)
    ents = _entities([["group", "oilg", None], ["group__node", ["oilg", "oil_n"], None]])
    values = [v for v in _values() if v[:3] != ["node", "oil_n", "node_type"]]
    values += [["node", "oil_n", "node_type", _s("balance")], ["node", "oil_n", "penalty_up", _f(100.0)],
               ["group", "oilg", "print_dispatch", _s("yes")]]
    oil = _table(tmp_path_factory, "su_oil_group", values, ents, "nodeGroup_flows_dt_gpe")
    assert _per_step(oil, ("oilg", "to_unit", "cp")) == pytest.approx([0.0, -120.0, 0.0, 0.0], abs=1e-6)


def test_aggregated_node_group_view_shows_the_startup_only_draw(tmp_path_factory) -> None:
    """A flow aggregator (flowGroup ``agg``, flow_aggregator dispatch_plots_only) of the unit cp (listed at its
    output elec: a flowGroup cannot name the startup_only arc) shows cp's draw from oil_n in the aggregate column of
    nodeGroup_flows_dt_gpe (−120 at t0002, MWh in the step, stored negated), and cp has no column of its own."""
    ents = _entities([["group", "oilg", None], ["group__node", ["oilg", "oil_n"], None],
                      ["flowGroup", "agg", None], ["flowGroup__unit__node", ["agg", "cp", "elec"], None]])
    values = [v for v in _values() if v[:3] != ["node", "oil_n", "node_type"]]
    values += [["node", "oil_n", "node_type", _s("balance")], ["node", "oil_n", "penalty_up", _f(100.0)],
               ["group", "oilg", "print_dispatch", _s("yes")],
               ["flowGroup", "agg", "flow_aggregator", _s("dispatch_plots_only")]]
    oil = _table(tmp_path_factory, "su_oil_agg", values, ents, "nodeGroup_flows_dt_gpe")
    assert _per_step(oil, ("oilg", "to_unitGroup", "agg")) == pytest.approx([0.0, -120.0, 0.0, 0.0], abs=1e-6)
    assert ("oilg", "to_unit", "cp") not in oil.columns


# ---------------------------------------------------------------------------
# Benders with master-hosted nodes: a master-local unit's draw, and the CO2-cap partition

#: Both carrier chains of regions A and B in the master (tests/engine/test_region_filter_master_hosted.py
#: MASTER_BIG): liquefier_A (h2_A -> lh2_A) is a master-local unit.
MASTER_BIG = frozenset({"h2_A", "lh2_A", "h2_B", "lh2_B"})


@pytest.fixture(scope="module")
def lh2_master_local_startup(lh2_with_startup):
    """lh2_with_startup with liquefier_A also online (linear) and drawing 2 MWh/MW of coal from the shared
    commodity node coal_market at each start-up."""
    import dataclasses

    from polar_high import Param

    data = lh2_with_startup
    p_dtype = data.process_source_sink.schema["p"]
    n_dtype = data.process_source_sink.schema["source"]
    c_dtype = data.startup_from_commodity.schema["c"]
    dt = data.dt.select("d", "t")

    def add(frame: pl.DataFrame, rows: pl.DataFrame) -> pl.DataFrame:
        return pl.concat([frame, rows.select(frame.columns).cast(frame.schema)])

    one = pl.DataFrame({"p": ["liquefier_A"]}).with_columns(pl.col("p").cast(p_dtype))
    pn = pl.DataFrame({"p": ["liquefier_A"], "n": ["coal_market"]}).with_columns(
        pl.col("p").cast(p_dtype), pl.col("n").cast(n_dtype))
    energy = pn.join(dt.select("d").unique(), how="cross").with_columns(value=pl.lit(2.0))
    new = dataclasses.replace(
        data, process_online=add(data.process_online, one),
        process_online_linear=add(data.process_online_linear, one),
        p_online_dt=add(data.p_online_dt, dt.join(one, how="cross")),
        process_startup_node=add(data.process_startup_node, pn),
        p_startup_energy=Param(("p", "n", "d"), add(data.p_startup_energy.frame, energy)),
        startup_from_commodity=add(data.startup_from_commodity,
                                   pn.with_columns(c=pl.lit("coal").cast(c_dtype))))
    new._axis_enums = getattr(data, "_axis_enums", None)
    return new


def _region_membership(data, master_nodes: frozenset[str]):
    from flextool.engine.solve._region_filter import load_region_membership

    mem = load_region_membership(data, ["region_A", "region_B", "region_C"])
    for m in mem.values():
        m["nodes"] = m["nodes"] - set(master_nodes)
    return mem


def test_master_keeps_a_master_local_units_draw_from_a_commodity_node(lh2_master_local_startup) -> None:
    """With master-hosted nodes the master keeps the draw of its master-local unit liquefier_A from the commodity
    node coal_market (a node of no region), with its online variables, and prices it as the whole-system LP does;
    coal_A's draw stays in region A only."""
    from flextool.engine.solve._region_filter import master_network_data, split

    data = lh2_master_local_startup
    regions = ["region_A", "region_B", "region_C"]
    mem = _region_membership(data, MASTER_BIG)
    master = master_network_data(data, regions, region_membership=mem, master_hosted_nodes=MASTER_BIG)
    assert sorted(utf8(master.process_startup_node).rows()) == [("liquefier_A", "coal_market")]
    assert sorted(utf8(master.startup_from_commodity).rows()) == [("liquefier_A", "coal_market", "coal")]
    assert "liquefier_A" in set(utf8(master.process_online_linear)["p"].to_list())
    splits = {s.region: s.data for s in split(data, regions=regions, region_membership=mem,
                                              benders_uncap_cross_region=True, master_hosted_nodes=MASTER_BIG)}
    assert sorted(utf8(splits["region_A"].process_startup_node).rows()) == [("coal_A", "coal_market")]

    def costs(pb) -> dict[str, float]:
        m = pb.canonicalise()
        return {m.col_names[c]: float(m.col_obj[c]) for c in range(m.n_cols)
                if m.col_names[c].startswith("v_startup_linear[liquefier_A,")}

    # liquefier_A is coarse (the daily h2 / lh2 steps) in the whole system and on the fine step in the master
    # (every master flow keeps the solve's step), so the columns differ; the cost per MW started is the same.
    whole = costs(build(data))
    got = costs(build(master))
    assert whole and got and all(v > 0.0 for v in whole.values())
    assert sorted(set(got.values())) == pytest.approx(sorted(set(whole.values())), rel=1e-12)


def test_co2_cap_partition_counts_the_start_up_draws() -> None:
    """_co2_master_partition: a draw is on its unit's side (master for a master-local unit, else region).  A group
    capping only master-local flows and master-local draws is hosted in the master; a region unit's draw in a group
    of master-local flows straddles the boundary and is refused."""
    from types import SimpleNamespace

    from flextool.engine.solve._region_filter import _co2_master_partition

    flows = pl.DataFrame({"p": ["liq"], "source": ["h2_M"], "sink": ["lh2_M"], "c": ["c"], "g": ["cap"]})
    ml = {("liq", "h2_M", "lh2_M")}

    def data(*draws: tuple[str, str]):
        su = pl.DataFrame({"p": [p for p, _ in draws], "n": ["fuel"] * len(draws), "c": ["c"] * len(draws),
                           "g": [g for _, g in draws]}, schema={"p": pl.Utf8, "n": pl.Utf8, "c": pl.Utf8, "g": pl.Utf8})
        return SimpleNamespace(flow_from_co2_capped=flows, startup_from_co2_capped=su if draws else None)

    assert _co2_master_partition(data(("liq", "cap")), ml, set(), {"liq"}) == {"cap"}
    assert _co2_master_partition(data(("liq", "drawonly")), ml, set(), {"liq"}) == {"cap", "drawonly"}
    assert _co2_master_partition(data(("u_A", "regiong")), ml, set(), {"liq"}) == {"cap"}
    with pytest.raises(RuntimeError, match=r"CO2-capped group 'cap' straddles.*start-up draw"):
        _co2_master_partition(data(("u_A", "cap")), ml, set(), {"liq"})


def test_co2_cap_loaders_keep_a_start_up_only_emitter_without_arcs() -> None:
    """With no efficiency arcs at all (pss_eff and pss_noEff None), a start-up draw in a capped group still keeps
    the cap's groups (_load_co2_cap / _load_co2_cap_total): the early return does not drop it."""
    from flextool.engine.preprocess._flex_data_provider import FlexDataProvider
    from flextool.engine.preprocess.flex_data import _load_co2_cap, _load_co2_cap_total

    prov = FlexDataProvider()
    prov.put("solve_data/group_co2_max_period", pl.DataFrame({"group": ["co2g"]}))
    prov.put("solve_data/group_co2_max_total", pl.DataFrame({"group": ["co2g"]}))
    prov.put("solve_data/commodity_node_co2", pl.DataFrame({"commodity": ["oil"], "node": ["oil_n"]}))
    prov.put("input/group__node", pl.DataFrame({"group": ["co2g"], "node": ["oil_n"]}))
    prov.put("solve_data/pdGroup", pl.DataFrame({"group": ["co2g"], "param": ["co2_max_total"], "value": [5.0]}))
    dt = pl.DataFrame({"d": ["p2020"], "t": ["t0001"]})
    here = Path(".")
    assert _load_co2_cap(here, here, None, dt, provider=prov)[0] is None
    g_max = _load_co2_cap(here, here, None, dt, provider=prov, startup_emitters=True)[0]
    assert g_max is not None and g_max["g"].to_list() == ["co2g"]
    assert _load_co2_cap_total(here, here, None, provider=prov)[0] is None
    total = _load_co2_cap_total(here, here, None, provider=prov, startup_emitters=True)
    assert total[0]["g"].to_list() == ["co2g"] and total[3] is not None
