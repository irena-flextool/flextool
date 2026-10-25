"""Online variables of connections (``startup_method`` on connections, #50) and the online sets of units.

docs/reference.md: a connection's ``startup_method`` (``linear`` / ``binary``) gives it online capacity; its flow in
each direction is limited by the online capacity × availability and ``startup_cost`` is paid per MW brought online.
GMPL mapped one-way connections with a startup method to online methods; the engine had lost that:

* a two-way connection with a startup method had no method mapping: it silently became one-way, lossless and
  without online variables;
* the DB-path online projections read ``unit.startup_method`` only and filtered on ``"integer"`` (the schema value is
  ``binary``); a non-empty projection replaced the CSV seed, so a one-way online connection lost its online
  variables as soon as a unit had ``startup_method = linear``, a binary unit next to a linear one left
  ``process_online`` (inertia counted twice, item 105), and a unit online only through ``minimum_time_method`` lost
  its online variables and min up / down rows next to a linear unit;
* a connection's ``startup_cost`` never reached the objective on the DB path;
* a DC power-flow line with a startup method would need line switching: rejected.

Fixtures: ``tests.json`` (network chain: west_east 400 MW west → east; lossless_2way: ll 200 MW lw ↔ le),
``stochastics.json``, ``case14_dc_power_flow.json`` — built from JSON with override alternatives.
"""
from __future__ import annotations

import dataclasses
import os

import polars as pl
import pytest

from tests.engine._lp_rows import coef, row_bounds, row_terms, step_of
from tests.engine._unit_db import FLEXTOOL_ROOT, b64, build, make_db, run, utf8

pytestmark = pytest.mark.solver

NET = ["init", "west", "coal", "wind", "network"]
UNI = NET + ["unidirectional"]
LL = ["Base", "lossless_2way"]
WE = "west_east"
STOCH_JSON = FLEXTOOL_ROOT / "tests" / "fixtures" / "stochastics.json"
CASE14 = FLEXTOOL_ROOT / "tests" / "fixtures" / "case14_dc_power_flow.json"
TOL = 1e-6


def _startup(conn: str, method: str, cost=3.0) -> list:
    return [["connection", conn, "startup_method", b64(method, "str")],
            ["connection", conn, "startup_cost", cost if isinstance(cost, list) else b64(cost, "float")]]


def _steps(tpf, alt, values, chain, entities=None, fixture_json=None) -> list:
    kw = {} if fixture_json is None else {"fixture_json": fixture_json}
    url, scen = make_db(tpf, alt=alt, values=values, base_chain=list(chain), entities=entities, **kw)
    return list(run(tpf, url, scen).values())


def _p(frame) -> set[str]:
    return set() if frame is None else set(utf8(frame.select("p"))["p"].to_list())


def _obj_coefs(pb, prefix: str) -> dict[str, float]:
    m = pb.canonicalise()
    return {nm: float(m.col_obj[i]) for i, nm in enumerate(m.col_names) if nm.startswith(prefix)}


def _col_int(pb, prefix: str) -> list[bool]:
    m = pb.canonicalise()
    return [bool(m.col_int[i]) for i, nm in enumerate(m.col_names) if nm.startswith(prefix)]


# ---------------------------------------------------------------------------
# One-way connection next to an online unit; startup cost


@pytest.fixture(scope="module")
def uni_online(tmp_path_factory):
    """UNI + coal_plant linear online (coal_min_load, startup_cost 100) + west_east linear, startup_cost 3."""
    return _steps(tmp_path_factory, "oc_uni", _startup(WE, "linear"), UNI + ["coal_min_load"])[-1]


def test_unidirectional_online_next_to_online_unit(uni_online) -> None:
    """A1: west_east keeps its online variables next to an online unit: v_online_linear columns and
    maxFlow_online_linear rows ``v_flow − M·v_online ≤ 0`` on its arc.  Before: the unit-only projection replaced the
    seed, no west_east column."""
    fd = uni_online.flex_data
    assert _p(fd.process_online_linear) == {"coal_plant", WE}
    pb = build(fd)
    rows = row_terms(pb, "maxFlow_online_linear", f"{WE},west,east,")
    assert len(rows) == 48
    for row, terms in rows.items():
        d, t = step_of(row)
        assert coef(terms, f"v_flow[{WE},west,east,{d},{t}]") == pytest.approx(1.0)
        assert coef(terms, f"v_online_linear[{WE},{d},{t}]") < 0.0


def test_connection_startup_cost(uni_online) -> None:
    """A2: the objective coefficient of v_startup_linear[west_east] is startup_cost × unitsize × the event weight,
    the same weight as coal_plant's (100 × 500).  Before: 0 (connection startup_cost dropped)."""
    pb = build(uni_online.flex_data)
    conn = _obj_coefs(pb, f"v_startup_linear[{WE},")
    unit = _obj_coefs(pb, "v_startup_linear[coal_plant,")
    assert len(conn) == len(unit) == 48
    for col, c in conn.items():
        d, t = step_of(col)
        u = unit[f"v_startup_linear[coal_plant,{d},{t}]"]
        assert u > 0.0
        assert c / (3.0 * 400.0) == pytest.approx(u / (100.0 * 500.0), rel=1e-12)


def test_connection_binary_startup(tmp_path_factory) -> None:
    """A2 binary: a one-way connection with startup_method binary gets integer v_online_integer columns and its
    startup cost on v_startup_integer."""
    step = _steps(tmp_path_factory, "oc_uni_bin", _startup(WE, "binary"), UNI + ["coal_min_load"])[-1]
    assert _p(step.flex_data.process_online_integer) == {WE}
    pb = build(step.flex_data)
    assert all(_col_int(pb, f"v_online_integer[{WE},"))
    assert len(_col_int(pb, f"v_online_integer[{WE},")) == 48
    assert all(c > 0.0 for c in _obj_coefs(pb, f"v_startup_integer[{WE},").values())


# ---------------------------------------------------------------------------
# Two-way connections


@pytest.fixture(scope="module")
def regular_cases(tmp_path_factory) -> dict:
    out = {"none": _steps(tmp_path_factory, "oc_reg_none", [], NET)[-1]}
    for method in ("linear", "binary"):
        out[method] = _steps(tmp_path_factory, f"oc_reg_{method}", _startup(WE, method, 0.0), NET)[-1]
    return out


@pytest.mark.parametrize("method", ["linear", "binary"])
def test_regular_connection_online_keeps_both_directions(regular_cases, method) -> None:
    """A3: a regular connection with a startup method keeps both arcs, both with efficiency, both capped by the
    online capacity; with startup_cost 0 online is free and the objective equals the no-startup run's.  Before: one
    lossless arc, no online variables, objective 507.5 instead of 585.7."""
    base, step = regular_cases["none"], regular_cases[method]
    fd = step.flex_data
    arcs = set(utf8(fd.process_source_sink).filter(pl.col("p") == WE).select("source", "sink").rows())
    assert arcs == {("west", "east"), ("east", "west")}
    eff = set(utf8(fd.process_source_sink_eff).filter(pl.col("p") == WE).select("source", "sink").rows())
    assert eff == arcs
    kind = "linear" if method == "linear" else "integer"
    rows = row_terms(build(fd), f"maxFlow_online_{kind}", f"{WE},")
    assert {r.split("[")[1].split(",")[1:3].__repr__() for r in rows} == {"['west', 'east']", "['east', 'west']"}
    assert step.solution.obj == pytest.approx(base.solution.obj, rel=1e-9)


@pytest.fixture(scope="module")
def onevar(tmp_path_factory):
    """lossless_2way: ll (no_losses_no_variable_cost, 200 MW) linear online, startup_cost 0; ll exports 100 MW from
    lw at t0001 and imports 100 MW at t0002 (v_flow_back)."""
    return _steps(tmp_path_factory, "oc_1var", _startup("ll", "linear", 0.0), LL)[-1]


def test_one_variable_connection_online(onevar) -> None:
    """A4: maxFlow_back_online_linear ``v_flow_back − v_online ≤ 0`` on every step, and the import at t0002
    (v_flow_back 0.5) is covered by v_online ≥ 0.5.  Before: no online variables."""
    fd = onevar.flex_data
    assert _p(fd.process_online_linear) == {"ll"}
    pb = build(fd)
    rows = row_terms(pb, "maxFlow_back_online_linear", "ll,lw,le,")
    assert len(rows) == 2
    bounds = row_bounds(pb, "maxFlow_back_online_linear", "ll,lw,le,")
    for row, terms in rows.items():
        d, t = step_of(row)
        assert coef(terms, f"v_flow_back[ll,lw,le,{d},{t}]") == pytest.approx(1.0)
        assert coef(terms, f"v_online_linear[ll,{d},{t}]") == pytest.approx(-1.0)
        assert bounds[row][1] == pytest.approx(0.0)
    back = {t: x for p, s, k, d, t, x in utf8(onevar.solution.value("v_flow_back")).rows()}
    on = {t: x for p, d, t, x in utf8(onevar.solution.value("v_online_linear")).rows()}
    assert back["t0002"] == pytest.approx(0.5, abs=TOL)
    assert on["t0002"] >= back["t0002"] - TOL


# ---------------------------------------------------------------------------
# DC power flow lines


def test_dc_connection_startup_rejected(tmp_path_factory) -> None:
    """A5: a DC power-flow line with a startup method raises (input derivation), naming the line."""
    from flextool.engine.input._solve_state import FlexToolConfigError

    with pytest.raises(FlexToolConfigError, match="line_10_11"):
        _steps(tmp_path_factory, "oc_dc", [["connection", "line_10_11", "startup_method", b64("linear", "str")]],
               ["base"], fixture_json=CASE14)


def test_dc_connection_online_rejected_by_model(tmp_path_factory) -> None:
    """A5, engine guard: FlexData with a DC line in process_online_linear (built without the input derivation)
    raises in the model builder."""
    from flextool.engine.input._solve_state import FlexToolConfigError

    step = _steps(tmp_path_factory, "oc_dc_ok", [], ["base"], fixture_json=CASE14)[-1]
    fd = step.flex_data
    p_dtype = fd.process_source_sink.schema["p"]
    on = pl.DataFrame({"p": ["line_10_11"]}).with_columns(pl.col("p").cast(p_dtype))
    bad = dataclasses.replace(fd, process_online_linear=on, process_online=on)
    with pytest.raises(FlexToolConfigError, match="line_10_11"):
        build(bad)


# ---------------------------------------------------------------------------
# Online sets of units (item 105)


@pytest.fixture(scope="module")
def bin_lin(tmp_path_factory):
    """coal_plant binary (coal_min_load_MIP) next to coal_chp linear and west_east linear, with an inertia limit
    (coal_plant inertia_constant 5, coal_chp 4)."""
    return _steps(tmp_path_factory, "oc_bin_lin",
                  [["unit", "coal_chp", "startup_method", b64("linear", "str")]] + _startup(WE, "linear"),
                  ["init", "west", "coal", "heat", "coal_chp", "coal_min_load", "coal_min_load_MIP", "network",
                   "inertia"])[-1]


def test_online_sets_both_classes_and_binary(bin_lin) -> None:
    """A6: process_online holds the binary unit, the linear unit and the linear connection.  Before: {coal_chp}."""
    fd = bin_lin.flex_data
    assert _p(fd.process_online) == {"coal_plant", "coal_chp", WE}
    assert _p(fd.process_online_integer) == {"coal_plant"}
    assert _p(fd.process_online_linear) == {"coal_chp", WE}


def test_binary_unit_inertia_counted_once(bin_lin) -> None:
    """A6″: the inertia row has one term per unit: v_online_integer for coal_plant and no v_flow term of coal_plant.
    Before: both (the v_flow term because the anti-join on process_online missed the binary unit)."""
    rows = row_terms(build(bin_lin.flex_data), "inertia_constraint")
    assert rows
    for row, terms in rows.items():
        d, t = step_of(row)
        assert coef(terms, f"v_online_integer[coal_plant,{d},{t}]") > 0.0
        assert not [c for c in terms if c.startswith("v_flow[coal_plant,")], (row, terms)
        assert coef(terms, f"v_online_linear[coal_chp,{d},{t}]") > 0.0


def test_min_time_unit_keeps_online_next_to_linear_unit(tmp_path_factory) -> None:
    """A6′: coal_plant is online only through minimum_time_method (no startup_method); next to coal_chp with
    startup_method linear it keeps its online variables and its minimum up / down rows.  Before: 0 rows."""
    step = _steps(tmp_path_factory, "oc_mt_lin", [["unit", "coal_chp", "startup_method", b64("linear", "str")]],
                  ["init", "west", "coal", "heat", "coal_chp", "coal_min_updowntime"])[-1]
    assert _p(step.flex_data.process_online_linear) == {"coal_plant", "coal_chp"}
    pb = build(step.flex_data)
    assert len(row_terms(pb, "minimum_uptime_linear", "coal_plant,")) == 72
    assert len(row_terms(pb, "minimum_downtime_linear", "coal_plant,")) == 72


# ---------------------------------------------------------------------------
# Projection == emit seed


_MIX_ENTS = [["connection", "en_exact", None], ["connection__node__node", ["en_exact", "east", "north"], None]]
_MIX_VALS = (
    [["unit", "coal_chp", "startup_method", b64("binary", "str")],
     ["unit", "coal_chp", "startup_cost", b64(5.0, "float")],
     ["connection", "en_exact", "transfer_method", b64("exact", "str")],
     ["connection", "en_exact", "existing", b64(50.0, "float")],
     ["connection", "west_north", "transfer_method", b64("unidirectional", "str")],
     ["connection", "east_north", "transfer_method", b64("no_losses_no_variable_cost", "str")]]
    + _startup(WE, "linear") + _startup("west_north", "binary", 2.0) + _startup("east_north", "linear", 0.0))


@pytest.fixture(scope="module")
def mixed(tmp_path_factory):
    """Units linear by minimum_time_method (coal_plant) and binary (coal_chp); connections regular linear
    (west_east), one-way binary (west_north), one-variable linear (east_north), exact without a startup method
    (en_exact)."""
    return _steps(tmp_path_factory, "oc_mixed", _MIX_VALS,
                  ["init", "west", "coal", "heat", "coal_chp", "coal_min_updowntime", "network"],
                  entities=_MIX_ENTS)[-1]


def test_online_projection_equals_emit(mixed) -> None:
    """A7: the DB-path online sets (projection, used by the model) equal the CSV seed the solve_data frames are
    built from, and the seed-derived fields agree with them (critique 8e): p_online_dt covers exactly
    process_online, pdt_online_* the online processes of each class with a startup cost, p_startup_cost has the
    authored costs (connections included; coal_plant has none in this chain).  exact without a startup method is
    in neither set."""
    fd, prov = mixed.flex_data, mixed.flex_data_provider
    lin, intg = {"coal_plant", WE, "east_north"}, {"coal_chp", "west_north"}
    assert _p(fd.process_online_linear) == lin
    assert _p(fd.process_online_integer) == intg
    assert _p(fd.process_online) == lin | intg
    for key, want in (("solve_data/process_online_linear", lin), ("solve_data/process_online_integer", intg),
                      ("solve_data/process_online", lin | intg)):
        seed = utf8(prov.get(key))
        assert set(seed[seed.columns[0]].to_list()) == want, key
    assert _p(fd.p_online_dt) == lin | intg
    costs = {p: v for p, d, v in utf8(fd.p_startup_cost.frame).select("p", "d", "value").rows()}
    assert costs == {WE: 3.0, "coal_chp": 5.0, "west_north": 2.0}
    assert _p(fd.pdt_online_linear) == {WE}
    assert _p(fd.pdt_online_integer) == {"coal_chp", "west_north"}


def test_methods_mapping_copies_equal() -> None:
    """The engine's copy of METHODS_MAPPING equals the input derivation's, and two-way connections with a startup
    method map to their flow-shape method."""
    from flextool.engine.preprocess.params._derived import _METHODS_MAPPING
    from flextool.input_derivation._specs import METHODS_MAPPING

    assert _METHODS_MAPPING == METHODS_MAPPING
    for st in ("linear", "binary"):
        assert METHODS_MAPPING[("regular", st, "fork_no")] == "method_2way_2var_exclude"
        assert METHODS_MAPPING[("no_losses_no_variable_cost", st, "fork_no")] == "method_2way_1var_off"
        assert METHODS_MAPPING[("variable_cost_only", st, "fork_no")] == "method_2way_2var_off"
        assert METHODS_MAPPING[("exact", st, "fork_no")] == "method_2way_2var_MIP_exclude"


# ---------------------------------------------------------------------------
# Rolling and stochastic


ROLL = ["init", "west", "battery", "wind", "coal", "network", "unidirectional", "coal_min_load", "fullYear",
        "dispatch_fullYear_roll"]
_ROLL_VALS = [["solve", "dispatch_fullYear_roll", "rolling_solve_horizon", b64(36.0, "float")],
              ["solve", "dispatch_fullYear_roll", "rolling_solve_jump", b64(24.0, "float")]]


@pytest.fixture(scope="module")
def rolling(tmp_path_factory) -> list:
    return _steps(tmp_path_factory, "oc_roll", _startup(WE, "linear") + _ROLL_VALS, ROLL)


def test_online_connection_rolling(rolling) -> None:
    """A8: every continuation roll carries west_east's committed online state: p_roll_continue_online has the
    previous roll's v_online at its last realized step, and the first-step online__startup row has no wrap
    column.  Before: no online variables for west_east next to the online coal_plant."""
    carried = 0
    for prev, nxt in zip(rolling, rolling[1:]):
        rco = nxt.flex_data.p_roll_continue_online
        assert rco is not None
        got = dict(utf8(rco.frame).select("p", "value").rows())
        assert WE in got
        last = utf8(prev.flex_data.realized_dispatch).rows()[-1][:2]
        on = {(d, t): x for p, d, t, x in utf8(prev.solution.value("v_online_linear")).rows() if p == WE}
        assert got[WE] == pytest.approx(on[last], abs=1e-9)
        carried += 1
    assert carried == len(rolling) - 1 > 0


@pytest.fixture(scope="module")
def stoch(tmp_path_factory):
    ents = [["node", "north", None], ["connection", "dn_north", None], ["connection", "dn_north2", None],
            ["connection__node__node", ["dn_north", "demand_node", "north"], None],
            ["connection__node__node", ["dn_north2", "demand_node", "north"], None]]
    vals = [["node", "north", "node_type", b64("balance", "str")], ["node", "north", "inflow", b64(-150.0, "float")],
            ["node", "north", "penalty_up", b64(5000.0, "float")],
            ["node", "north", "penalty_down", b64(5000.0, "float")],
            ["connection", "dn_north", "transfer_method", b64("unidirectional", "str")],
            ["connection", "dn_north", "existing", b64(100.0, "float")],
            ["connection", "dn_north", "efficiency", b64(0.95, "float")],
            ["connection", "dn_north2", "transfer_method", b64("regular", "str")],
            ["connection", "dn_north2", "existing", b64(100.0, "float")],
            ["connection", "dn_north2", "efficiency", b64(0.9, "float")]]
    vals += _startup("dn_north", "linear") + _startup("dn_north2", "binary", 5.0)
    url, scen = make_db(tmp_path_factory, alt="oc_stoch", values=vals, base_chain=["base", "init", "system", "2day"],
                        entities=ents, fixture_json=STOCH_JSON)
    return list(run(tmp_path_factory, url, scen).values())[-1]


def test_online_connection_stochastic(stoch) -> None:
    """A9: the online status of both connections (one-way linear, regular binary) is tied across the forecast
    branches by the non-anticipativity online rows.  Before: the regular connection had no online variables."""
    pb = build(stoch.flex_data)
    for kind, conn in (("linear", "dn_north"), ("integer", "dn_north2")):
        rows = row_terms(pb, f"non_anticipativity_online_{kind}", f"{conn},")
        assert rows, kind
        for terms in rows.values():
            cols = [c for c in terms if c.startswith(f"v_online_{kind}[{conn},")]
            assert len(cols) == 2 and sorted(terms[c] for c in cols) == [-1.0, 1.0]


# ---------------------------------------------------------------------------
# Constant vs equivalent period map of startup_cost (CLAUDE.md shape rule)


def _startup_obj(steps: list, conn: str, kind: str) -> list[list[tuple[str, float]]]:
    return [sorted(_obj_coefs(build(s.flex_data), f"v_startup_{kind}[{conn},").items()) for s in steps]


def _pmap(values: dict) -> list:
    return b64({"index_type": "str", "index_name": "period", "data": [[k, v] for k, v in values.items()]}, "map")


def test_startup_cost_constant_vs_period_map(tmp_path_factory, uni_online, rolling, stoch) -> None:
    """A connection startup_cost authored as a constant and as the equivalent period map gives the same objective
    coefficients in a single, a rolling and a stochastic solve."""
    single = _steps(tmp_path_factory, "oc_uni_map", _startup(WE, "linear", _pmap({"p2020": 3.0})),
                    UNI + ["coal_min_load"])
    assert _startup_obj(single, WE, "linear") == _startup_obj([uni_online], WE, "linear")
    per = {d: 3.0 for d in ("p2020", "p2025", "p2030", "p2035")}
    roll_map = _steps(tmp_path_factory, "oc_roll_map", _startup(WE, "linear", _pmap(per)) + _ROLL_VALS, ROLL)
    assert len(roll_map) == len(rolling)
    assert _startup_obj(roll_map, WE, "linear") == _startup_obj(rolling, WE, "linear")
    ents = [["node", "north", None], ["connection", "dn_north", None], ["connection", "dn_north2", None],
            ["connection__node__node", ["dn_north", "demand_node", "north"], None],
            ["connection__node__node", ["dn_north2", "demand_node", "north"], None]]
    vals = [["node", "north", "node_type", b64("balance", "str")], ["node", "north", "inflow", b64(-150.0, "float")],
            ["node", "north", "penalty_up", b64(5000.0, "float")],
            ["node", "north", "penalty_down", b64(5000.0, "float")],
            ["connection", "dn_north", "transfer_method", b64("unidirectional", "str")],
            ["connection", "dn_north", "existing", b64(100.0, "float")],
            ["connection", "dn_north", "efficiency", b64(0.95, "float")],
            ["connection", "dn_north2", "transfer_method", b64("regular", "str")],
            ["connection", "dn_north2", "existing", b64(100.0, "float")],
            ["connection", "dn_north2", "efficiency", b64(0.9, "float")]]
    vals += _startup("dn_north", "linear", _pmap({"period1": 3.0})) + _startup("dn_north2", "binary",
                                                                             _pmap({"period1": 5.0}))
    url, scen = make_db(tmp_path_factory, alt="oc_stoch_map", values=vals,
                        base_chain=["base", "init", "system", "2day"], entities=ents, fixture_json=STOCH_JSON)
    smap = list(run(tmp_path_factory, url, scen).values())[-1]
    for conn, kind in (("dn_north", "linear"), ("dn_north2", "integer")):
        got = _startup_obj([smap], conn, kind)
        assert got == _startup_obj([stoch], conn, kind)
        assert any(v > 0.0 for _c, v in got[0])   # the weight-0 trunk period has coefficient 0


# ---------------------------------------------------------------------------
# Outputs


def test_connection_online_outputs(tmp_path_factory) -> None:
    """E: write_outputs reports connection_online__dt, connection_online__d_average and connection_startup__d for
    the online connection, and the unit outputs only for units."""
    import pandas as pd

    from flextool._resources import package_data_path
    from flextool.engine import run_chain_from_db
    from flextool.engine.preprocess._flex_data_provider import FlexDataProvider
    from flextool.process_outputs.write_outputs import write_outputs

    url, scen = make_db(tmp_path_factory, alt="oc_out", values=_startup(WE, "linear"),
                        base_chain=UNI + ["coal_min_load"])
    work = tmp_path_factory.mktemp("oc_out_w")
    cwd = os.getcwd()
    try:
        os.chdir(work)
        steps = run_chain_from_db(url, scen, work_folder=work, keep_solutions=True)
        last = list(steps.values())[-1]
        write_outputs(scenario_name=scen, output_location=str(work), subdir=scen,
                      output_config_path=str(package_data_path("schemas/default_plots.yaml")),
                      write_methods=["csv"], fallback_output_location=str(work),
                      raw_output_dir=str(work / "output_raw"), solution=last.solution, solve_name=last.solve_name,
                      solve_steps=[(s.solve_name, s.flex_data, s.effective_solution) for s in steps.values()],
                      flex_data_provider=FlexDataProvider())
    finally:
        os.chdir(cwd)
    out = work / "output_csv" / scen
    on = pd.read_csv(out / "connection_online__dt.csv")
    assert WE in on.columns and on[WE].notna().all()
    assert WE in pd.read_csv(out / "connection_online__d_average.csv").columns
    assert WE in pd.read_csv(out / "connection_startup__d.csv").columns
    unit_on = pd.read_csv(out / "unit_online__dt.csv")
    assert "coal_plant" in unit_on.columns and WE not in unit_on.columns
