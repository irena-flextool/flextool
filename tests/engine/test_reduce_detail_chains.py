"""Solve-level detail reduction across solves: realization solves, the carry between levels, the outputs.

Two-solve sequence on ``tests/fixtures/tests.json`` (alternative ``init`` only; never a checked-in ``.sqlite``):
solve ``sA`` dispatches p2020 steps t0001-t0002 and ``sB`` the next steps t0003-t0004 (timesets ``tsA`` / ``tsB``
of 2 h on timeline y2020); ``sB`` continues ``sA``'s realised end (ramp flow, units online).  Node ``elec`` has
a demand of 50 MW and an upward reserve ``rsv`` of 10 MW (g1 provides it); unit ``g1`` fuel -> elec (one 100 MW
unit, binary startup, min_load 0.5, ``min_load_efficiency`` with efficiency 0.5 and efficiency_at_min_load 0.4, a
ramp limit) burns fuel at 20 CUR/MWh.

Hand values (design solve_relaxation_design.md, Critique 1-2 and brief WP3 test 3, independent of the code):
section 0.5 · (1/0.4 − 1/0.5) / (1 − 0.5) = 0.5, slope 1/0.5 − 0.5 = 1.5.  A full solve runs one unit (100 MW
online) at 50 MW: fuel 1.5 · 50 + 0.5 · 100 = 125 MW.  A solve with ``reduce_unit_detail_method = online`` has no
units online and the full-load line: fuel 2 · 50 = 100 MW.  Its online and reserve cells are empty (NaN), the
full solve's keep their values (1 unit online, 10 MW of reserve).
"""
from __future__ import annotations

import os
from pathlib import Path

import polars as pl
import pytest

from tests.engine._unit_db import b64, make_db, utf8

pytestmark = pytest.mark.solver

OUTPUT_CONFIG = str(Path(__file__).resolve().parents[2] / "flextool" / "schemas" / "default_plots.yaml")


@pytest.fixture(autouse=True)
def _strict_autoscale(monkeypatch) -> None:
    """A missing autoscale registry entry raises instead of falling back to an unscaled LP."""
    monkeypatch.setenv("FLEXTOOL_AUTOSCALE_STRICT", "1")


def _f(x: float) -> list:
    return b64(float(x), "float")


def _s(x: str) -> list:
    return b64(x, "str")


def _ts(start: str) -> list:
    return b64({"index_type": "str", "data": [[start, 2.0]]}, "map")


def _period_map(ts: str, period: str = "p2020") -> list:
    return b64({"index_type": "str", "index_name": "period", "data": [[period, ts]]}, "map")


ENTITIES = [["node", "elec", None], ["node", "fuel", None], ["commodity", "fuelc", None],
            ["group__node", ["electricity", "elec"], None], ["commodity__node", ["fuelc", "fuel"], None],
            ["unit", "g1", None], ["unit__inputNode", ["g1", "fuel"], None], ["unit__outputNode", ["g1", "elec"], None],
            ["reserve", "rsv", None], ["reserve__upDown__group", ["rsv", "up", "electricity"], None],
            ["reserve__upDown__unit__node", ["rsv", "up", "g1", "elec"], None],
            ["timeset", "tsA", None], ["timeset", "tsB", None], ["solve", "sA", None], ["solve", "sB", None]]

#: ``dc``: node elec2 (demand 10 MW) behind DC power-flow line c (elec -> elec2, 100 MW, reactance 0.1 p.u.) and
#: node elec3 (demand 5 MW) behind connection c2 (elec -> elec3, 100 MW, binary start-up; outside the DC group).
DC_ENTITIES = [["node", "elec2", None], ["group", "ac", None], ["group__node", ["ac", "elec"], None],
               ["group__node", ["ac", "elec2"], None], ["connection", "c", None],
               ["connection__node__node", ["c", "elec", "elec2"], None], ["node", "elec3", None],
               ["connection", "c2", None], ["connection__node__node", ["c2", "elec", "elec3"], None]]


PERIOD = {"sA": "p2020", "sB": "p2020"}
PERIODS2 = {"sA": "p2020", "sB": "p2025"}


def _values(levels: dict[str, tuple[str, str]], *, periods: dict[str, str] = PERIOD, inertia: bool = False,
            non_synchronous: bool = False, dc: bool = False, timesets: dict[str, str] | None = None) -> list:
    out, grp, key = "unit__outputNode", ["rsv", "up", "electricity"], ["rsv", "up", "g1", "elec"]
    vals = [["model", "flexTool", "solves", b64({"value_type": "str", "data": ["sA", "sB"]}, "array")],
            ["timeset", "tsA", "timeline", _s("y2020")], ["timeset", "tsA", "timeset_duration", _ts("t0001")],
            ["timeset", "tsB", "timeline", _s("y2020")], ["timeset", "tsB", "timeset_duration", _ts("t0003")],
            ["node", "elec", "node_type", _s("balance")], ["node", "elec", "inflow", _f(-50.0)],
            ["node", "fuel", "node_type", _s("commodity")], ["commodity", "fuelc", "price", _f(20.0)],
            ["unit", "g1", "conversion_method", _s("min_load_efficiency")], ["unit", "g1", "efficiency", _f(0.5)],
            ["unit", "g1", "efficiency_at_min_load", _f(0.4)], ["unit", "g1", "min_load", _f(0.5)],
            ["unit", "g1", "existing", _f(100.0)], ["unit", "g1", "startup_method", _s("binary")],
            [out, ["g1", "elec"], "ramp_method", _s("ramp_limit")], [out, ["g1", "elec"], "ramp_speed_up", _f(0.01)],
            [out, ["g1", "elec"], "ramp_speed_down", _f(0.01)],
            ["reserve__upDown__group", grp, "reserve_method", _s("timeseries_only")],
            ["reserve__upDown__group", grp, "reservation", _f(10.0)],
            ["reserve__upDown__group", grp, "penalty_reserve", _f(1000.0)],
            ["reserve__upDown__unit__node", key, "is_enabled", _s("yes")],
            ["reserve__upDown__unit__node", key, "max_share", _f(1.0)],
            ["reserve__upDown__unit__node", key, "reliability", _f(1.0)]]
    if inertia:
        vals += [["group", "electricity", "has_inertia", _s("yes")], ["group", "electricity", "inertia_limit", _f(50.0)],
                 ["group", "electricity", "penalty_inertia", _f(1000.0)], [out, ["g1", "elec"], "inertia_constant", _f(2.0)]]
    if non_synchronous:
        vals += [["group", "electricity", "has_non_synchronous", _s("yes")],
                 ["group", "electricity", "non_synchronous_limit", _f(0.8)],
                 ["group", "electricity", "penalty_non_synchronous", _f(1000.0)],
                 [out, ["g1", "elec"], "is_non_synchronous", _s("yes")]]
    if dc:
        vals += [["node", "elec2", "node_type", _s("balance")], ["node", "elec2", "inflow", _f(-10.0)],
                 ["group", "ac", "transfer_method", _s("dc_power_flow_with_angles")],
                 ["connection", "c", "existing", _f(100.0)], ["connection", "c", "reactance", _f(0.1)],
                 ["node", "elec3", "node_type", _s("balance")], ["node", "elec3", "inflow", _f(-5.0)],
                 ["connection", "c2", "existing", _f(100.0)], ["connection", "c2", "startup_method", _s("binary")]]
    for solve, ts in (timesets or {"sA": "tsA", "sB": "tsB"}).items():
        vals += [["solve", solve, "period_timeset", _period_map(ts, periods[solve])],
                 ["solve", solve, "realized_periods", b64({"value_type": "str", "data": [periods[solve]]}, "array")],
                 ["solve", solve, "solve_mode", _s("single_solve")], ["solve", solve, "solver_mip_gap", _f(0.0)]]
    for solve, (unit, system) in levels.items():
        vals += [["solve", solve, "reduce_unit_detail_method", _s(unit)],
                 ["solve", solve, "reduce_system_detail_method", _s(system)]]
    return vals


def _run(tmp_path_factory, alt: str, levels: dict, *, keep_solutions: bool = True, **fixture):
    from flextool.engine import run_chain_from_db

    entities = ENTITIES + (DC_ENTITIES if fixture.get("dc") else [])
    url, scen = make_db(tmp_path_factory, alt=alt, values=_values(levels, **fixture), base_chain=["init"],
                        entities=entities)
    wf = tmp_path_factory.mktemp(f"{alt}_w")
    steps = run_chain_from_db(url, scen, work_folder=wf, keep_solutions=keep_solutions)
    assert list(steps) == ["sA", "sB"], list(steps)
    return steps, wf, scen


def _write_outputs(steps, wf: Path, scen: str, *, in_memory: bool) -> Path:
    from flextool.engine.preprocess._flex_data_provider import FlexDataProvider
    from flextool.process_outputs.write_outputs import write_outputs

    last = list(steps.values())[-1]
    kwargs = dict(solve_steps=[(s.solve_name, s.flex_data, s.effective_solution) for s in steps.values()]) \
        if in_memory else dict(flex_data=last.flex_data)
    cwd = os.getcwd()
    try:
        os.chdir(wf)
        write_outputs(scenario_name=scen, output_location=str(wf), subdir=scen, output_config_path=OUTPUT_CONFIG,
                      write_methods=["csv"], fallback_output_location=str(wf),
                      raw_output_dir=str(wf / "output_raw"), solution=last.solution, solve_name=last.solve_name,
                      flex_data_provider=last.flex_data_provider or FlexDataProvider(), **kwargs)
    finally:
        os.chdir(cwd)
    return wf / "output_csv" / scen


def _table(out: Path, name: str, column: str) -> dict[str, float] | None:
    """``{time: value}`` of ``column`` (the first header row) in the output CSV ``name``; None when the column is
    absent.  Header rows below the first have empty index cells."""
    lines = [ln.split(",") for ln in (out / name).read_text().splitlines() if ln]
    head = lines[0]
    if column not in head:
        return None
    k, t = head.index(column), head.index("time")
    return {row[t]: float(row[k]) for row in lines[1:] if row[0] and row[k] != ""}


FUEL = {"full": -125.0, "online": -100.0}


@pytest.mark.parametrize("in_memory", [True, False])
def test_reduced_last_solve_keeps_the_tables_of_the_full_one(tmp_path_factory, in_memory) -> None:
    """sA full, sB online + reserves: the online and reserve tables keep sA's values (1 unit, 10 MW) although the
    last solve has neither; the fuel of the min_load_efficiency unit is 125 MW in sA and 100 MW in sB (not NaN).
    Both output paths: the in-memory steps and the persisted per-roll slices."""
    steps, wf, scen = _run(tmp_path_factory, f"rdc_fwd_{in_memory}", {"sB": ("online", "reserves")},
                           keep_solutions=True)
    out = _write_outputs(steps, wf, scen, in_memory=in_memory)
    fuel = _table(out, "unit__inputNode__dt.csv", "g1")
    assert fuel == pytest.approx({"t0001": FUEL["full"], "t0002": FUEL["full"],
                                  "t0003": FUEL["online"], "t0004": FUEL["online"]}, abs=1e-6)
    # sB's own cells are empty (not 0): _table leaves them out.
    online = _table(out, "unit_online__dt.csv", "g1")
    assert online == pytest.approx({"t0001": 1.0, "t0002": 1.0})
    reserve = _table(out, "process__reserve__upDown__node__dt.csv", "g1")
    assert reserve == pytest.approx({"t0001": 10.0, "t0002": 10.0})


def test_reduced_first_solve_full_last(tmp_path_factory) -> None:
    """sA online + reserves, sB full: the full solve's tables are complete (1 unit online, 10 MW of reserve, fuel
    125 MW), sA's fuel is 100 MW."""
    steps, wf, scen = _run(tmp_path_factory, "rdc_rev", {"sA": ("online", "reserves")})
    out = _write_outputs(steps, wf, scen, in_memory=True)
    assert _table(out, "unit__inputNode__dt.csv", "g1") == pytest.approx(
        {"t0001": FUEL["online"], "t0002": FUEL["online"], "t0003": FUEL["full"], "t0004": FUEL["full"]}, abs=1e-6)
    # sA's own cells are empty (not 0): _table leaves them out.
    assert _table(out, "unit_online__dt.csv", "g1") == pytest.approx({"t0003": 1.0, "t0004": 1.0})
    assert _table(out, "process__reserve__upDown__node__dt.csv", "g1") == pytest.approx({"t0003": 10.0, "t0004": 10.0})


# ---------------------------------------------------------------------------
# a solve with online variables continuing the ramp flow of a solve without them (Critique 2b)


NESTED = ["init", "west", "coal", "coal_invest", "wind", "wind_invest", "battery", "battery_invest",
          "invest_solveSequence_5weeks", "invest_allYears_5weeks", "fullYear", "dispatch_fullYear_roll",
          "storage_fullYear_6h", "invest_24h", "bind_battery_forward", "unlimited_investment"]


def test_full_storage_roll_after_an_online_dispatch_starts_cold(tmp_path_factory, capsys) -> None:
    """Nested invest -> rolling storage (48 h / 24 h) -> rolling dispatch (12 h / 6 h) with coal unit commitment
    (linear, min_load 0.4) and a ramp limit; the dispatch solve has reduce_unit_detail_method = online.  Storage roll
    1 (online variables kept) continues the ramp flow the last dispatch roll realised, which carries no units
    online: coal_plant's ramp and commitment start cold there (one info line per solve), where the online / ramp
    carry check used to raise."""
    import copy

    from polar_high import Param

    from flextool.engine.preprocess._reduce_detail import cold_start_online_flow_carry
    from flextool.engine.preprocess.params._ramp_links import check_online_ramp_carry
    from tests.engine._unit_db import run

    vals = [["unit", "coal_plant", "startup_method", _s("linear")], ["unit", "coal_plant", "min_load", _f(0.4)],
            ["unit__outputNode", ["coal_plant", "west"], "ramp_method", _s("ramp_limit")],
            ["unit__outputNode", ["coal_plant", "west"], "ramp_speed_up", _f(0.002)],
            ["unit__outputNode", ["coal_plant", "west"], "ramp_speed_down", _f(0.002)],
            ["node", "battery", "storage_nested_fix_method", _s("fix_nothing")],
            ["solve", "storage_fullYear_6h", "solve_mode", _s("rolling_window")],
            ["solve", "storage_fullYear_6h", "rolling_solve_horizon", _f(48.0)],
            ["solve", "storage_fullYear_6h", "rolling_solve_jump", _f(24.0)],
            ["solve", "dispatch_fullYear_roll", "rolling_solve_horizon", _f(12.0)],
            ["solve", "dispatch_fullYear_roll", "rolling_solve_jump", _f(6.0)],
            ["solve", "dispatch_fullYear_roll", "reduce_unit_detail_method", _s("online")]]
    url, scen = make_db(tmp_path_factory, alt="rdc_nest", base_chain=NESTED, values=vals)
    steps = run(tmp_path_factory, url, scen)
    cold = [ln for ln in capsys.readouterr().out.splitlines() if "start cold" in ln]
    assert cold == ["Solve storage_fullYear_6h: ramp and commitment of coal_plant start cold (the solve it continues "
                    "from left out online variables, reduce_unit_detail_method)."]
    names = list(steps)
    s1 = names.index("storage_fullYear_6h_roll_1")
    producer = steps[names[s1 - 1]]
    assert names[s1 - 1].startswith("dispatch_fullYear_roll_roll_")
    assert producer.handoff.unit_detail == "online" and producer.handoff.roll_end_online is None
    assert producer.handoff.online_left_out == frozenset({"coal_plant"})
    assert "coal_plant" in producer.handoff.roll_end_flow["process"].to_list()
    fd = steps["storage_fullYear_6h_roll_1"].flex_data
    assert {str(p) for p in fd.process_online["p"].to_list()} == {"coal_plant"}
    assert fd.dt_ramp_carry is not None and fd.dt_ramp_carry.height > 0
    assert fd.p_roll_continue_flow is None and fd.p_roll_continue_online is None
    # With the carried flow kept, the consistency check refuses this data; the cold start drops it.
    d, t = utf8(fd.dt_ramp_carry).select("d", "t").row(0)
    carried = Param(("p", "source", "sink", "d", "t"),
                    pl.DataFrame({"p": ["coal_plant"], "source": ["coal_plant"], "sink": ["west"], "d": [d], "t": [t],
                                  "value": [0.5]}))
    kept = copy.copy(fd)
    kept.p_roll_continue_flow = carried
    with pytest.raises(ValueError, match="no online carry"):
        check_online_ramp_carry(kept)
    started, procs = cold_start_online_flow_carry(kept, producer.handoff.online_left_out)
    assert procs == ["coal_plant"] and started.p_roll_continue_flow is None
    check_online_ramp_carry(started)


def test_carry_router_passes_the_producers_level() -> None:
    """The routed carriers carry the producing solve's reduce_unit_detail_method; none without a producer."""
    from flextool.engine.solve._carry_routing import CarryRouter
    from flextool.engine.solve._handoff import SolveHandoff

    router = CarryRouter({})
    router.start("a")
    assert router.step_carriers("a", None)["unit_detail"] == "none"
    h = SolveHandoff(roll_end_step=pl.DataFrame({"period": ["p"], "step": ["t1"], "duration": [1.0]}))
    h.unit_detail = "online_ramps"
    router.record("a", h)
    router.start("b")
    assert router.step_carriers("b", None)["unit_detail"] == "online_ramps"


def test_every_solve_reduced_has_no_online_or_reserve_table(tmp_path_factory) -> None:
    """Both solves online + reserves: no solve keeps online variables or reserves, so the outputs have no g1
    column for them; fuel is 100 MW throughout."""
    steps, wf, scen = _run(tmp_path_factory, "rdc_both", {"sA": ("online", "reserves"), "sB": ("online", "reserves")})
    out = _write_outputs(steps, wf, scen, in_memory=True)
    assert _table(out, "unit__inputNode__dt.csv", "g1") == pytest.approx(
        {t: FUEL["online"] for t in ("t0001", "t0002", "t0003", "t0004")}, abs=1e-6)
    for name in ("unit_online__dt.csv", "process__reserve__upDown__node__dt.csv"):
        assert not (out / name).exists() or _table(out, name, "g1") is None, name


def test_output_static_view_keeps_fields_reduced_everywhere() -> None:
    """The static view is the pre-transform data, except the fields every sub-solve so far reduced."""
    import copy
    from types import SimpleNamespace

    from flextool.engine.preprocess._reduce_detail import output_static_view

    full = SimpleNamespace(process_online=pl.DataFrame({"p": ["g1"]}),
                           groupInertia=pl.DataFrame({"g": ["electricity"]}))
    reduced = copy.copy(full)
    reduced.process_online = None
    reduced.groupInertia = None
    view = output_static_view(reduced, full, frozenset({"groupInertia"}))
    assert view.process_online is full.process_online and view.groupInertia is None
    assert output_static_view(full, full, frozenset()) is full


# ---------------------------------------------------------------------------
# realization solves take the parent's levels


SEPARATE = Path(__file__).resolve().parents[1] / "fixtures" / "separate_dispatch_noinvest.json"


def test_separate_dispatch_sibling_takes_the_parents_integers(tmp_path_factory) -> None:
    """stoch_rd with reduce_unit_detail_method = integers and a binary peaker: the realized-dispatch sibling resolves
    to integers (continuous units online, no integer column) and solves at the cost of the linear case,
    50 MW × 10 CUR/MWh × 2920 h × 3 steps = 4 380 000 (hand calc of test_separate_dispatch_noinvest_capacity;
    relaxed binary units online behave as linear ones)."""
    from flextool.engine import run_chain_from_db
    from tests.engine._unit_db import build

    vals = [["unit", "peaker", "startup_method", _s("binary")],
            ["solve", "stoch_rd", "reduce_unit_detail_method", _s("integers")]]
    url, scen = make_db(tmp_path_factory, alt="rdc_sibling", values=vals, base_chain=["init", "base", "stoch"],
                        fixture_json=SEPARATE)
    wf = tmp_path_factory.mktemp("rdc_sibling_w")
    steps = run_chain_from_db(url, scen, work_folder=wf, keep_solutions=True, warm=False)
    sibling = steps["stoch_rd__realized_dispatch"]
    assert sibling.optimal is True
    assert sibling.flex_data.relax_integrality is True and steps["stoch_rd"].flex_data.relax_integrality is True
    m = build(sibling.flex_data).canonicalise()
    assert not any(m.col_int)
    vo = utf8(sibling.solution.value("v_online_integer")).filter(pl.col("p") == "peaker")
    assert vo["value"].min() == pytest.approx(2.0, abs=1e-9) and vo["value"].max() == pytest.approx(2.0, abs=1e-9)
    obj = float(pl.read_parquet(wf / "output_raw" / "v_obj__stoch_rd__realized_dispatch.parquet")["objective"][0])
    assert obj == pytest.approx(4_380_000.0, rel=1e-6)


REALIZED = Path(__file__).resolve().parents[1] / "fixtures" / "stoch_realized_dispatch.json"


def test_declared_realization_solve_warning_from_the_run(tmp_path_factory, capsys) -> None:
    """stoch_rd (level none) with its author-declared realization solve stoch_rd_realized_dispatch, which sets
    reduce_system_detail_method = reserves: the run prints one warning that the realization solve uses its
    parent's none, and the realization solve runs at none."""
    from flextool.engine import run_chain_from_db

    vals = [["solve", "stoch_rd_realized_dispatch", "reduce_system_detail_method", _s("reserves")]]
    url, scen = make_db(tmp_path_factory, alt="rdc_declared", values=vals, base_chain=["init", "base", "stoch"],
                        fixture_json=REALIZED)
    steps = run_chain_from_db(url, scen, work_folder=tmp_path_factory.mktemp("rdc_declared_w"), warm=False,
                              keep_solutions=True)
    lines = [ln for ln in capsys.readouterr().out.splitlines() if ln.startswith("Warning: realization solve")]
    assert lines == ["Warning: realization solve 'stoch_rd_realized_dispatch' sets reduce_system_detail_method = "
                     "'reserves', but it uses 'none', the value of its parent solve 'stoch_rd' (it takes the "
                     "parent's units online and reserves)."]
    assert all(s.optimal for s in steps.values()), list(steps)


def test_declared_realization_solve_warns_when_its_own_level_is_overridden() -> None:
    """An author-declared realization solve that sets a different level is told it uses its parent's; the same
    value or none set gives no line."""
    from flextool.engine.input._solve_config import SolveConfig

    sc = object.__new__(SolveConfig)
    sc.realized_dispatch_parent = {"real": "stoch", "same": "stoch", "unset": "stoch"}
    sc.reduce_unit_detail_method = {"stoch": "online", "real": "none", "same": "Online"}
    sc.reduce_system_detail_method = {"real": "reserves"}
    lines = sc.reduce_detail_overridden("real")
    assert len(lines) == 2
    assert "reduce_unit_detail_method = 'none'" in lines[0] and "'online'" in lines[0] and "'stoch'" in lines[0]
    assert "reduce_system_detail_method = 'reserves'" in lines[1] and "'none'" in lines[1]
    assert sc.reduce_detail_overridden("same") == []
    assert sc.reduce_detail_overridden("unset") == []
    assert sc.reduce_detail_overridden("stoch") == []


# ---------------------------------------------------------------------------
# the cells of a feature a solve left out are empty, not 0 (process_outputs.reduced_cells)


def _raw(out: Path, name: str, column: str) -> dict[tuple, str]:
    """``{(period, time): cell}`` (``time`` None in a per-period table) of ``column`` in the output CSV ``name``, the
    cells as written (an empty cell is ``""``)."""
    lines = [ln.split(",") for ln in (out / name).read_text().splitlines() if ln]
    head = lines[0]
    k, d = head.index(column), head.index("period")
    t = head.index("time") if "time" in head else None
    return {(row[d], row[t] if t is not None else None): row[k] for row in lines[1:] if row[0]}


DT = {"sA": [("p2020", "t0001"), ("p2020", "t0002")], "sB": [("p2025", "t0003"), ("p2025", "t0004")]}


def _assert_cells(out: Path, name: str, column: str, full: str, reduced: str, value: float, *,
                  per_period: bool = False) -> None:
    """The cells of solve ``full`` hold ``value``, those of solve ``reduced`` are empty."""
    cells = _raw(out, name, column)
    keys = {s: ([(PERIODS2[s], None)] if per_period else DT[s]) for s in (full, reduced)}
    assert set(cells) == set(keys[full]) | set(keys[reduced]), (name, cells)
    for key in keys[full]:
        assert float(cells[key]) == pytest.approx(value, abs=1e-6), (name, key, cells)
    for key in keys[reduced]:
        assert cells[key] == "", (name, key, cells)


@pytest.mark.parametrize("in_memory", [True, False])
def test_left_out_cells_are_empty_after_a_full_solve(tmp_path_factory, in_memory) -> None:
    """sA full (realises p2020), sB online + reserves + inertia + non-synchronous (realises p2025), with an inertia
    limit of 50 on group electricity and a non-synchronous limit of 0.8 with g1 non-synchronous: the online,
    start-up, reserve, inertia and non-synchronous tables keep sA's values and leave sB's cells empty, per step and
    per period.  sA: 1 unit online, 0 start-ups (its first step links to its last, units.md, and 1 unit is online
    throughout); 10 MW of reserve, 0 reserve shortfall and a reserve price of 0 (50 MW of online headroom); largest
    flow 50 MW, inertia 2 s · 100 MW online = 200 MWs (all of it g1's) and 0 inertia shortfall; non-synchronous
    shortfall 50 − 0.8 · 50 = 10 MW per step, per period 10 MW · 2 h · 8760 h / 2 h = 87600 MWh.  Fuel of the min_load_efficiency unit in p2025 is output / 0.5 = 100 MW (125 MW in p2020).
    Both output paths: the in-memory steps and the persisted per-roll slices."""
    steps, wf, scen = _run(tmp_path_factory, f"rdc_cells_{in_memory}",
                           {"sB": ("online", "reserves_inertia_non_synchronous")}, periods=PERIODS2, inertia=True,
                           non_synchronous=True, keep_solutions=in_memory)
    out = _write_outputs(steps, wf, scen, in_memory=in_memory)
    _assert_cells(out, "unit_online__dt.csv", "g1", "sA", "sB", 1.0)
    _assert_cells(out, "unit_online__d_average.csv", "g1", "sA", "sB", 1.0, per_period=True)
    _assert_cells(out, "unit_startup__d.csv", "g1", "sA", "sB", 0.0, per_period=True)
    _assert_cells(out, "process__reserve__upDown__node__dt.csv", "g1", "sA", "sB", 10.0)
    _assert_cells(out, "process__reserve__upDown__node__d_average.csv", "g1", "sA", "sB", 10.0, per_period=True)
    _assert_cells(out, "slack__reserve__upDown__group__dt.csv", "rsv", "sA", "sB", 0.0)
    _assert_cells(out, "group_inertia_largest_flow__dt.csv", "electricity", "sA", "sB", 50.0)
    _assert_cells(out, "slack__inertia_group__dt.csv", "electricity", "sA", "sB", 0.0)
    _assert_cells(out, "reserve_price__upDown__group__dt.csv", "rsv", "sA", "sB", 0.0)
    _assert_cells(out, "slack__reserve__upDown__group__d.csv", "rsv", "sA", "sB", 0.0, per_period=True)
    _assert_cells(out, "slack__inertia_group__d.csv", "electricity", "sA", "sB", 0.0, per_period=True)
    _assert_cells(out, "group_inertia__dt.csv", "electricity", "sA", "sB", 200.0)
    _assert_cells(out, "group__unit__node_inertia__dt.csv", "electricity", "sA", "sB", 200.0)
    _assert_cells(out, "slack__nonsync_group__dt.csv", "electricity", "sA", "sB", 10.0)
    _assert_cells(out, "slack__nonsync_group__d.csv", "electricity", "sA", "sB", 87600.0, per_period=True)
    fuel = _raw(out, "unit__inputNode__dt.csv", "g1")
    assert {k: float(v) for k, v in fuel.items()} == pytest.approx(
        {**{k: FUEL["full"] for k in DT["sA"]}, **{k: FUEL["online"] for k in DT["sB"]}}, abs=1e-6)


@pytest.mark.parametrize("in_memory", [True, False])
def test_left_out_cells_are_empty_before_a_full_solve(tmp_path_factory, in_memory) -> None:
    """sA online + reserves first, sB full last: sB's tables are complete (1 unit online, 10 MW of reserve), sA's
    cells are empty, its fuel is 100 MW.  Both output paths."""
    steps, wf, scen = _run(tmp_path_factory, f"rdc_cells_rev_{in_memory}", {"sA": ("online", "reserves")},
                           periods=PERIODS2, keep_solutions=in_memory)
    out = _write_outputs(steps, wf, scen, in_memory=in_memory)
    _assert_cells(out, "unit_online__dt.csv", "g1", "sB", "sA", 1.0)
    _assert_cells(out, "unit_online__d_average.csv", "g1", "sB", "sA", 1.0, per_period=True)
    _assert_cells(out, "process__reserve__upDown__node__dt.csv", "g1", "sB", "sA", 10.0)
    fuel = _raw(out, "unit__inputNode__dt.csv", "g1")
    assert {k: float(v) for k, v in fuel.items()} == pytest.approx(
        {**{k: FUEL["online"] for k in DT["sA"]}, **{k: FUEL["full"] for k in DT["sB"]}}, abs=1e-6)


def test_a_period_shared_with_a_full_solve_keeps_its_value(tmp_path_factory) -> None:
    """sA full and sB online + reserves both in p2020: sB's steps are empty, the per-period cell of p2020 keeps
    its value (sA realised the period too)."""
    steps, wf, scen = _run(tmp_path_factory, "rdc_cells_shared", {"sB": ("online", "reserves")})
    out = _write_outputs(steps, wf, scen, in_memory=True)
    online = _raw(out, "unit_online__dt.csv", "g1")
    assert online == {("p2020", "t0001"): "1", ("p2020", "t0002"): "1", ("p2020", "t0003"): "",
                      ("p2020", "t0004"): ""}
    assert _raw(out, "unit_online__d_average.csv", "g1")[("p2020", None)] != ""


def test_reduced_cells_rules() -> None:
    """A step belongs to the last solve that realised it; a period is empty only when every solve that realised
    it left the feature out; tables of other features and unreduced solves are untouched."""
    import numpy as np
    import pandas as pd

    from flextool.process_outputs.reduced_cells import mask_reduced_cells, reduced_cells_of

    dt = pd.MultiIndex.from_tuples([("a", "p1", "t1"), ("a", "p1", "t2"), ("b", "p1", "t2"), ("b", "p2", "t3")],
                                   names=["solve", "period", "time"])
    dp = pd.MultiIndex.from_tuples([("a", "p1"), ("b", "p1"), ("b", "p2")], names=["solve", "period"])
    from types import SimpleNamespace
    cells = reduced_cells_of(SimpleNamespace(dt_realize_dispatch=dt, d_realized_period=dp),
                             {"b": frozenset({"online"})})
    assert {f: set(x) for f, x in cells.steps.items()} == {"online": {("p1", "t2"), ("p2", "t3")}}
    assert {f: set(x) for f, x in cells.periods.items()} == {"online": {"p2"}}
    step_idx = pd.MultiIndex.from_tuples([("p1", "t1"), ("p1", "t2"), ("p2", "t3")], names=["period", "time"])
    period_idx = pd.Index(["p1", "p2"], name="period")
    results = {"unit_online_dt_e": pd.DataFrame({"g1": [1.0, 1.0, 1.0]}, index=step_idx),
               "unit_online_average_d_e": pd.DataFrame({"g1": [1.0, 1.0]}, index=period_idx),
               "process_reserve_upDown_node_dt_eppe": pd.DataFrame({"g1": [5.0, 5.0, 5.0]}, index=step_idx)}
    mask_reduced_cells(results, cells)
    assert np.isnan(results["unit_online_dt_e"]["g1"].to_numpy()).tolist() == [False, True, True]
    assert np.isnan(results["unit_online_average_d_e"]["g1"].to_numpy()).tolist() == [False, True]
    assert results["process_reserve_upDown_node_dt_eppe"]["g1"].tolist() == [5.0, 5.0, 5.0]
    untouched = {"unit_online_dt_e": pd.DataFrame({"g1": [1.0]}, index=step_idx[:1])}
    mask_reduced_cells(untouched, reduced_cells_of(SimpleNamespace(dt_realize_dispatch=dt, d_realized_period=dp), {}))
    assert untouched["unit_online_dt_e"]["g1"].tolist() == [1.0]


SAME_STEPS = {"sA": "tsA", "sB": "tsA"}


@pytest.mark.parametrize("in_memory", [True, False])
@pytest.mark.parametrize("system", ["reserves", "non_synchronous"])
def test_left_out_on_the_steps_of_a_full_solve(tmp_path_factory, system, in_memory) -> None:
    """sA full, then sB on sA's own steps (timeset tsA; sB realises them last) without reserves or the
    non-synchronous limit.  The output static view keeps sA's reserve and non-synchronous groups, sB's
    parameters have none: the write does not raise, and sB owns the steps, so their cells are empty."""
    steps, wf, scen = _run(tmp_path_factory, f"rdc_same_{system}_{in_memory}", {"sB": ("none", system)},
                           timesets=SAME_STEPS, non_synchronous=True, keep_solutions=in_memory)
    out = _write_outputs(steps, wf, scen, in_memory=in_memory)
    tables = {"reserves": [("process__reserve__upDown__node__dt.csv", "g1"),
                           ("slack__reserve__upDown__group__dt.csv", "rsv"),
                           ("reserve_price__upDown__group__dt.csv", "rsv")],
              "non_synchronous": [("slack__nonsync_group__dt.csv", "electricity")]}[system]
    for name, column in tables:
        assert _raw(out, name, column) == {("p2020", "t0001"): "", ("p2020", "t0002"): ""}, name


def test_left_out_connection_online_and_dc_angles(tmp_path_factory) -> None:
    """``dc`` fixture, sA full (p2020), sB online + dc_power_flow (p2025): sA keeps 1 unit of c2 online, 0 start-ups
    (first step linked to the last, 1 unit online throughout), the voltage angle 0 at elec (the reference) and
    −10 MW · 0.1 p.u. / 100 MVA = −0.01 rad at elec2, the angle difference 0.01 rad on c; sB's cells are empty."""
    steps, wf, scen = _run(tmp_path_factory, "rdc_dc", {"sB": ("online", "dc_power_flow")}, periods=PERIODS2, dc=True)
    out = _write_outputs(steps, wf, scen, in_memory=True)
    _assert_cells(out, "connection_online__dt.csv", "c2", "sA", "sB", 1.0)
    _assert_cells(out, "connection_online__d_average.csv", "c2", "sA", "sB", 1.0, per_period=True)
    _assert_cells(out, "connection_startup__d.csv", "c2", "sA", "sB", 0.0, per_period=True)
    _assert_cells(out, "dc_angle_dt_e.csv", "elec", "sA", "sB", 0.0)
    _assert_cells(out, "dc_angle_dt_e.csv", "elec2", "sA", "sB", -0.01)
    _assert_cells(out, "dc_angle_diff_dt_e.csv", "c", "sA", "sB", 0.01)
