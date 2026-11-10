"""Online variables without a use under the ``integers*`` levels of ``solve.reduce_unit_detail_method``.

After the level's removals, a unit or connection whose online variables nothing in the solve uses loses them, as if
its ``startup_method`` were ``no_startup`` (``preprocess._reduce_detail.online_uses``).  ``online`` / ``online_ramps``
are unchanged, and ``none`` never prunes.

Hand-built model ``uses`` (on ``tests/fixtures/tests.json``, alternative ``init`` only: one period p2020, timeset
``2day`` cut to two 1-h steps; never a checked-in ``.sqlite``): node ``elec`` (demand 150 MW) in group
``electricity``, commodity node ``fuel`` at 20 CUR/MWh, and six units fuel -> elec, each 100 MW, efficiency 0.5,
``startup_method`` binary, with one use each (or none):

* ``gs``: a start-up cost of 10 CUR/MW (only);
* ``gm``: min_load 0.5 (only);
* ``gr``: min_load 0.5 and the upward reserve ``rsv`` of ``electricity`` (10 MW);
* ``gi``: inertia_constant 2 s at elec; ``electricity`` has an inertia limit of 100 MWs;
* ``gp``: a ramp limit (ramp_speed 0.5 per minute up and down);
* ``gn``: nothing.

Expected (derived by hand from the uses in the design, not from the code): under every ``integers*`` level ``gn``
loses its online variables; a level that removes min limits takes ``gm``'s only use (``gr`` keeps its reserve);
a level that removes ramps takes ``gp``'s.  ``gs`` keeps its start-up cost under every ``integers*`` level, ``gi``
its inertia.  With the system level leaving out reserves (inertia), ``gr`` (``gi``) has no use left under
``integers_min_limits`` (``integers``).

The online rows of a unit without a use only repeat its capacity limit, so the LP without them has the same
optimum as the LP that keeps them (the relaxed integers of the same level).
"""
from __future__ import annotations

import ast
import os
import re
from pathlib import Path

import polars as pl
import pytest

from flextool.engine.preprocess import _reduce_detail
from flextool.engine.preprocess._reduce_detail import ONLINE_USES, apply_reduce_detail, online_uses
from tests.engine._unit_db import RunCache, axis_vocabulary, b64, build, make_db

pytestmark = pytest.mark.solver

SOLVE = "y2020_2day_dispatch"
UNITS = ("gs", "gm", "gr", "gi", "gp", "gn")
_RUNS = RunCache()
MODEL_DIR = Path(_reduce_detail.__file__).resolve().parents[1] / "model"


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


def _entities() -> list:
    ents = [["node", "elec", None], ["node", "fuel", None], ["commodity", "fuelc", None],
            ["group__node", ["electricity", "elec"], None], ["commodity__node", ["fuelc", "fuel"], None],
            ["reserve", "rsv", None], ["reserve__upDown__group", ["rsv", "up", "electricity"], None],
            ["reserve__upDown__unit__node", ["rsv", "up", "gr", "elec"], None]]
    for u in UNITS:
        ents += [["unit", u, None], ["unit__inputNode", [u, "fuel"], None], ["unit__outputNode", [u, "elec"], None]]
    return ents


def _values(unit_level: str | None = None) -> list:
    out, grp, key = "unit__outputNode", ["rsv", "up", "electricity"], ["rsv", "up", "gr", "elec"]
    vals = [["timeset", "2day", "timeset_duration", b64({"index_type": "str", "data": [["t0001", 2.0]]}, "map")],
            ["solve", SOLVE, "solver_mip_gap", _f(0.0)],
            ["node", "elec", "node_type", _s("balance")], ["node", "elec", "inflow", _f(-150.0)],
            ["node", "fuel", "node_type", _s("commodity")], ["commodity", "fuelc", "price", _f(20.0)],
            ["unit", "gs", "startup_cost", _f(10.0)],
            ["unit", "gm", "min_load", _f(0.5)], ["unit", "gr", "min_load", _f(0.5)],
            ["reserve__upDown__group", grp, "reserve_method", _s("timeseries_only")],
            ["reserve__upDown__group", grp, "reservation", _f(10.0)],
            ["reserve__upDown__group", grp, "penalty_reserve", _f(1000.0)],
            ["reserve__upDown__unit__node", key, "is_enabled", _s("yes")],
            ["reserve__upDown__unit__node", key, "max_share", _f(1.0)],
            ["reserve__upDown__unit__node", key, "reliability", _f(1.0)],
            [out, ["gi", "elec"], "inertia_constant", _f(2.0)],
            ["group", "electricity", "has_inertia", _s("yes")], ["group", "electricity", "inertia_limit", _f(100.0)],
            ["group", "electricity", "penalty_inertia", _f(10.0)],
            [out, ["gp", "elec"], "ramp_method", _s("ramp_limit")], [out, ["gp", "elec"], "ramp_speed_up", _f(0.5)],
            [out, ["gp", "elec"], "ramp_speed_down", _f(0.5)]]
    for u in UNITS:
        vals += [["unit", u, "existing", _f(100.0)], ["unit", u, "efficiency", _f(0.5)],
                 ["unit", u, "startup_method", _s("binary")]]
    if unit_level is not None:
        vals.append(["solve", SOLVE, "reduce_unit_detail_method", _s(unit_level)])
    return vals


def _step(tmp_path_factory, unit_level: str | None = None):
    steps, _wf, _scen = _RUNS.get(tmp_path_factory, alt=f"ou_{unit_level or 'none'}", values=_values(unit_level),
                                  base_chain=["init"], entities=_entities())
    assert len(steps) == 1, list(steps)
    step = next(iter(steps.values()))
    assert step.solution is not None
    return step


def _online_columns(pb) -> set[str]:
    """The units with an online column (v_online_linear / v_online_integer) in a built problem."""
    m = pb.canonicalise()
    return {n.split("[", 1)[1].split(",", 1)[0] for n in m.col_names if n.startswith("v_online_")}


# ---------------------------------------------------------------------------
# which units keep their online variables


@pytest.mark.parametrize("unit, system, unused", [
    ("integers", "none", ["gn"]),
    ("integers_ramps", "none", ["gn", "gp"]),
    ("integers_min_limits", "none", ["gm", "gn"]),
    ("integers_ramps_min_limits", "none", ["gm", "gn", "gp"]),
    ("integers_min_limits", "reserves", ["gm", "gn", "gr"]),
    ("integers", "inertia", ["gi", "gn"]),
])
def test_units_without_a_use_lose_their_online_variables(tmp_path_factory, unit, system, unused) -> None:
    """A start-up cost (gs) keeps the online variables under integers_min_limits; min load only (gm) loses them
    there and keeps them under integers; a reserve provider (gr) keeps them while the reserve is in the solve; the
    run-log line names the units that lost them."""
    fd = _step(tmp_path_factory).flex_data
    reduced, report = apply_reduce_detail(fd, unit, system)
    assert report.removed["online_unused"] == unused
    kept = sorted(set(UNITS) - set(unused))
    assert report.removed["integer_online"] == kept
    assert _online_columns(build(reduced)) == set(kept)
    assert sorted(reduced.process_online_integer["p"].cast(pl.Utf8).to_list()) == kept
    assert reduced.relax_integrality is True
    line = _reduce_detail.summary_line(report, fd, "s")
    units = f"{len(unused)} unit{'s' if len(unused) > 1 else ''}"
    assert f"online variables left out (nothing in this solve uses them) for {units} ({', '.join(unused)})" in line


def test_none_and_online_levels_do_not_prune(tmp_path_factory) -> None:
    """none leaves the data as it is (the same object: a byte-identical LP); online and online_ramps leave out
    every online variable as before, without an online_unused entry."""
    fd = _step(tmp_path_factory).flex_data
    reduced, report = apply_reduce_detail(fd, "none")
    assert reduced is fd and not report.changed
    for level in ("online", "online_ramps"):
        reduced, report = apply_reduce_detail(fd, level)
        assert "online_unused" not in report.removed
        assert report.removed["online"] == sorted(UNITS)
        assert reduced.process_online is None and _online_columns(build(reduced)) == set()


def test_the_uses_of_the_units(tmp_path_factory) -> None:
    """online_uses of the unreduced data names each unit's own use (gr: min load and the reserve)."""
    uses = online_uses(_step(tmp_path_factory).flex_data)
    assert uses == {"startup_cost": {"gs"}, "min_load": {"gm", "gr"}, "reserves": {"gr"}, "inertia": {"gi"},
                    "ramp_limits": {"gp"}}


def test_left_out_rows_do_not_change_the_optimum(tmp_path_factory, monkeypatch) -> None:
    """Under integers_ramps_min_limits the LP without the online variables of gm, gn and gp has the same objective
    as the LP that keeps them (relaxed) and the same unit flows."""
    fd = _step(tmp_path_factory).flex_data
    pruned, report = apply_reduce_detail(fd, "integers_ramps_min_limits")
    assert report.removed["online_unused"] == ["gm", "gn", "gp"]
    monkeypatch.setattr(_reduce_detail, "online_uses", lambda d, sparts=frozenset(): {"all": set(UNITS)})
    kept, report_kept = apply_reduce_detail(fd, "integers_ramps_min_limits")
    assert "online_unused" not in report_kept.removed
    sols = {}
    for name, data in (("pruned", pruned), ("kept", kept)):
        pb = build(data)
        with axis_vocabulary(data):
            sols[name] = pb.solve()
    assert float(sols["pruned"].obj) == pytest.approx(float(sols["kept"].obj), rel=1e-9)

    def total(sol) -> float:
        return float(sol.value("v_flow")["value"].sum())

    assert total(sols["pruned"]) == pytest.approx(total(sols["kept"]), rel=1e-9)


def test_a_solve_authored_at_integers_min_limits_runs(tmp_path_factory, capsys) -> None:
    """The level authored on the solve: the solve builds and solves without gm's and gn's online variables, and the
    run log names them."""
    step = _step(tmp_path_factory, "integers_min_limits")
    assert _online_columns(build(step.flex_data)) == {"gs", "gr", "gi", "gp"}
    online = step.solution.value("v_online_integer")
    assert {str(p) for p in online["p"].to_list()} == {"gs", "gr", "gi", "gp"}


# ---------------------------------------------------------------------------
# outputs across solves


def _ts(start: str) -> list:
    return b64({"index_type": "str", "data": [[start, 2.0]]}, "map")


def _two_solves(tmp_path_factory, level_b: str):
    """sA (p2020, t0001-t0002, full) then sB (p2025, t0003-t0004, ``level_b``); demand 50 MW; g1 (efficiency 0.5,
    min load 0.5) and g2 (efficiency 0.4, start-up cost 10), both 100 MW and binary."""
    from flextool.engine import run_chain_from_db

    ents = [["node", "elec", None], ["node", "fuel", None], ["commodity", "fuelc", None],
            ["commodity__node", ["fuelc", "fuel"], None], ["timeset", "tsA", None], ["timeset", "tsB", None],
            ["solve", "sA", None], ["solve", "sB", None]]
    vals = [["model", "flexTool", "solves", b64({"value_type": "str", "data": ["sA", "sB"]}, "array")],
            ["timeset", "tsA", "timeline", _s("y2020")], ["timeset", "tsA", "timeset_duration", _ts("t0001")],
            ["timeset", "tsB", "timeline", _s("y2020")], ["timeset", "tsB", "timeset_duration", _ts("t0003")],
            ["node", "elec", "node_type", _s("balance")], ["node", "elec", "inflow", _f(-50.0)],
            ["node", "fuel", "node_type", _s("commodity")], ["commodity", "fuelc", "price", _f(20.0)],
            ["unit", "g1", "min_load", _f(0.5)], ["unit", "g1", "efficiency", _f(0.5)],
            ["unit", "g2", "startup_cost", _f(10.0)], ["unit", "g2", "efficiency", _f(0.4)]]
    for u in ("g1", "g2"):
        ents += [["unit", u, None], ["unit__inputNode", [u, "fuel"], None], ["unit__outputNode", [u, "elec"], None]]
        vals += [["unit", u, "existing", _f(100.0)], ["unit", u, "startup_method", _s("binary")]]
    for solve, ts, period in (("sA", "tsA", "p2020"), ("sB", "tsB", "p2025")):
        vals += [["solve", solve, "period_timeset",
                  b64({"index_type": "str", "index_name": "period", "data": [[period, ts]]}, "map")],
                 ["solve", solve, "realized_periods", b64({"value_type": "str", "data": [period]}, "array")],
                 ["solve", solve, "solve_mode", _s("single_solve")], ["solve", solve, "solver_mip_gap", _f(0.0)]]
    vals.append(["solve", "sB", "reduce_unit_detail_method", _s(level_b)])
    url, scen = make_db(tmp_path_factory, alt=f"ou2_{level_b}", values=vals, base_chain=["init"], entities=ents)
    wf = tmp_path_factory.mktemp(f"ou2_{level_b}_w")
    steps = run_chain_from_db(url, scen, work_folder=wf, keep_solutions=True)
    assert list(steps) == ["sA", "sB"], list(steps)
    return steps, wf, scen


def _write_outputs(steps, wf: Path, scen: str) -> Path:
    from flextool.engine.preprocess._flex_data_provider import FlexDataProvider
    from flextool.process_outputs.write_outputs import write_outputs

    last = list(steps.values())[-1]
    cwd = os.getcwd()
    try:
        os.chdir(wf)
        write_outputs(scenario_name=scen, output_location=str(wf), subdir=scen,
                      output_config_path=str(Path(__file__).resolve().parents[2] / "templates" / "default_plots.yaml"),
                      write_methods=["csv"], fallback_output_location=str(wf), raw_output_dir=str(wf / "output_raw"),
                      solution=last.solution, solve_name=last.solve_name,
                      flex_data_provider=last.flex_data_provider or FlexDataProvider(),
                      solve_steps=[(s.solve_name, s.flex_data, s.effective_solution) for s in steps.values()])
    finally:
        os.chdir(cwd)
    return wf / "output_csv" / scen


def _cells(out: Path, name: str, column: str) -> dict[tuple, str]:
    """``{(period, time): cell}`` of ``column`` in the output CSV ``name`` (an empty cell is ``""``)."""
    lines = [ln.split(",") for ln in (out / name).read_text().splitlines() if ln]
    head = lines[0]
    k, d, t = head.index(column), head.index("period"), head.index("time")
    return {(row[d], row[t]): row[k] for row in lines[1:] if row[0]}


def test_outputs_leave_the_cells_of_the_left_out_unit_empty(tmp_path_factory) -> None:
    """sA full, sB integers_min_limits: g1 (min load only) loses its online variables in sB, g2 (start-up cost)
    keeps them.  g1 serves the 50 MW (g2 burns more fuel): sA has 1 unit of g1 online (binary, at least half a unit
    for 50 MW of 100), g2 none; in sB g1's cells are empty (not modelled), g2 has 0 online."""
    steps, wf, scen = _two_solves(tmp_path_factory, "integers_min_limits")
    assert steps["sB"].flex_data.process_online["p"].cast(pl.Utf8).to_list() == ["g2"]
    out = _write_outputs(steps, wf, scen)
    g1 = _cells(out, "unit_online__dt.csv", "g1")
    g2 = _cells(out, "unit_online__dt.csv", "g2")
    sa, sb = [("p2020", "t0001"), ("p2020", "t0002")], [("p2025", "t0003"), ("p2025", "t0004")]
    assert set(g1) == set(sa + sb) and set(g2) == set(sa + sb)
    assert [float(g1[k]) for k in sa] == pytest.approx([1.0, 1.0])
    assert [g1[k] for k in sb] == ["", ""]
    assert [float(g2[k]) for k in sa + sb] == pytest.approx([0.0] * 4)


# ---------------------------------------------------------------------------
# every model reader of the online sets and variables is classified


#: The names whose reads make a model function a consumer of the units online.
_ONLINE_NAMES = {
    "process_online", "process_online_linear", "process_online_integer", "v_online_linear", "v_online_integer",
    "v_startup_linear", "v_startup_integer", "v_shutdown_linear", "v_shutdown_integer", "uc_classes", "has_online",
    "has_online_lin", "has_online_int", "p_online_dt", "p_olin_idx", "p_oint_idx", "pdt_online_linear",
    "pdt_online_integer", "p_fix_online", "p_roll_continue_online", "dt_online_carry",
}

#: Each model function that reads them (``module:function``) -> the uses of :data:`ONLINE_USES` it holds, or one of
#: the classes that are no use: ``online_model`` (the online variables and their own rows), ``validation`` (input
#: checks), ``non_anticipativity`` (the ties of the units online across branches).
_NOT_A_USE = {"online_model", "validation", "non_anticipativity"}
CONSUMERS: dict[str, set[str]] = {
    "_balance_terms:balance_flow_terms": {"min_load_efficiency"},
    "_group_slack:_eq_capacity_margin": {"min_load_efficiency"},
    "_group_slack:_eq_inertia": {"inertia"},
    "_group_slack:_eq_non_sync": {"min_load_efficiency"},
    "_non_anticipativity:add_dispatch_constraints": {"non_anticipativity"},
    "_online:add_variables": {"online_model"},
    "_online:uc_classes": {"online_model"},
    # Its rows are classified one by one in ONLINE_ROWS.
    "_online:add_constraints": {"online_model", "min_up_down", "reserves", "ramp_limits", "parent_commitment"},
    "_reserve:_drawn_terms": {"min_load_efficiency"},
    "_reserve:_add_connection_reserve_constraints": {"reserves"},
    "_reserve:_with_section": {"min_load_efficiency"},
    "_unit_floors:add_constraints": {"min_load"},
    "core:_prepare": {"online_model"},
    "core:_validate_no_online_dc_lines": {"validation"},
    "core:_validate_online_constraint_coeff": {"validation"},
    "core:_eq_storage_usage_fix": {"min_load_efficiency"},
    "core:_ramp_family": {"ramp_limits"},
    "core:_online_ramp_rhs": {"ramp_limits"},
    "core:_eq_conversion": {"min_load_efficiency"},
    "core:_eff_fuel_input": {"min_load_efficiency"},
    "core:_eq_user_constraints": {"user_constraint"},
    "core:_obj_variable_cost": {"min_load_efficiency"},
    "core:_obj_startup_cost": {"startup_cost"},
    "core:_max_flow_rhs": {"min_load_efficiency"},
}

#: The row families of ``_online.py`` -> the use they hold (``online_model``: only the units online and the capacity
#: limit; maxFlow_online also holds the spinning reserve of a reserve provider, ``reserves``).
ONLINE_ROWS: dict[str, set[str]] = {
    "maxOnline": {"online_model"}, "online_change": {"online_model"}, "maxStartup": {"online_model"},
    "maxShutdown": {"online_model"}, "maxFlow_online": {"online_model", "reserves"},
    "maxFlow_online_shutdown": {"ramp_limits"}, "maxFlow_online_shutdown_branch": {"ramp_limits"},
    "maxFlow_back_online": {"online_model"}, "maxOutputSum_online": {"online_model", "reserves"},
    "minimum_uptime": {"min_up_down"}, "minimum_downtime": {"min_up_down"}, "online_fix_eq_": {"parent_commitment"},
    "maxInputFlow_online": {"reserves", "ramp_limits"}, "maxInputFlow_online_shutdown": {"ramp_limits"},
    "maxInputFlow_online_shutdown_branch": {"ramp_limits"},
}


def _online_readers() -> set[str]:
    out = set()
    for path in sorted(MODEL_DIR.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        module = path.relative_to(MODEL_DIR).with_suffix("").as_posix()
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            for sub in ast.walk(node):
                name = (sub.attr if isinstance(sub, ast.Attribute) else sub.id if isinstance(sub, ast.Name)
                        else sub.value if isinstance(sub, ast.Constant) and isinstance(sub.value, str) else None)
                if name in _ONLINE_NAMES:
                    out.add(f"{module}:{node.name}")
                    break
    return out


def test_every_reader_of_the_online_sets_is_classified() -> None:
    """Every model function that reads the online sets or variables is in CONSUMERS: a new consumer must be
    classified (a use of ONLINE_USES, which online_uses must then check, or no use).  Every use is held somewhere."""
    assert _online_readers() == set(CONSUMERS)
    classes = set().union(*CONSUMERS.values(), *ONLINE_ROWS.values())
    assert classes - _NOT_A_USE == set(ONLINE_USES)


def test_every_online_row_family_is_classified() -> None:
    """The row families _online.py declares are those of ONLINE_ROWS."""
    text = (MODEL_DIR / "_online.py").read_text(encoding="utf-8")
    names = set(re.findall(r'add_(?:cstr|twin)\(\s*(?:b,\s*)?f?"([A-Za-z_]+?)(?:\{|")', text, flags=re.S))
    assert names == set(ONLINE_ROWS)
