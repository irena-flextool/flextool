"""The capacity margin applies in the solve's investment periods only (#69).

    vq_capacity_margin[g,d],  capacityMargin[g,d,t]   ∀ g ∈ groupCapacityMargin, d ∈ D_inv
    D_inv = period_invest ∩ periods of dt
    period_invest = invest_periods_of_current_solve ∪ the branch periods of those anchors

``reference/groups.md`` and the schema say the capacity margin is "used only by the investment mode", and GMPL built it
for ``d ∈ period_invest`` only.  Before the fix it was built in every period of every solve: dispatch solves,
rolling rolls, stochastic dispatch and look-ahead periods paid the margin penalty.  A period label has no dt,
so a stochastic fan contributes its forecast branch periods, one row per forecast.

Every case uses a manual margin of 100 000 MW (far above any capacity), so a built row always has a slack.
Databases are built from JSON (never a checked-in ``.sqlite``).
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from tests.engine._unit_db import b64, build, cstr_over, make_db, utf8

REPO = Path(__file__).resolve().parents[2]
FIXTURES = REPO / "tests" / "fixtures"
TESTS_JSON = FIXTURES / "tests.json"
STOCHASTICS_JSON = FIXTURES / "stochastics.json"
RECOURSE_JSON = FIXTURES / "stoch_two_period_invest.json"
OUTPUT_CONFIG = str(REPO / "flextool" / "schemas" / "default_plots.yaml")
BIG = 100000.0


@pytest.fixture(autouse=True)
def _strict_autoscale(monkeypatch):
    monkeypatch.setenv("FLEXTOOL_AUTOSCALE_STRICT", "1")


def scalar(v: float) -> list:
    return b64(float(v), "float")


def s(v: str) -> list:
    return b64(v, "str")


def periods_array(periods: list[str]) -> list:
    return b64({"value_type": "str", "data": periods, "index_name": "period"}, "array")


def margin(group: str) -> list:
    return [["group", group, "capacity_margin", scalar(BIG)],
            ["group", group, "capacity_margin_method", s("manual")]]


def _chain(fixture_json: Path, scenario: str) -> list[str]:
    rows = [r for r in json.loads(fixture_json.read_text())["scenario_alternatives"] if r[0] == scenario]
    nxt = {a: b for _, a, b in rows}
    alt = next(a for a in nxt if a not in set(nxt.values()))
    chain = []
    while alt is not None:
        chain.append(alt)
        alt = nxt.get(alt)
    return chain


def _deterministic_rolling_json(tmp_path_factory) -> Path:
    """``stochastics.json`` without branches (the c8 / c10 rolling vehicle)."""
    data = json.loads(STOCHASTICS_JSON.read_text())
    data["parameter_values"] = [
        pv for pv in data["parameter_values"]
        if pv[2] not in ("stochastic_branches", "include_stochastics")
        and not (pv[0] == "profile" and pv[1] == "wind" and pv[4] in ("1week_rolling", "2day"))]
    data["entities"] += [["connection", "link", None],
                         ["connection__node__node", ["link", "gas_market", "coal_market"], None]]
    data["entity_alternatives"].append(["connection", ["link"], "init", True])
    out = tmp_path_factory.mktemp("cm_json") / "stochastics_det.json"
    out.write_text(json.dumps(data))
    return out


ELEC_G = [["group", "elec_g", None], ["group__node", ["elec_g", "demand_node"], None]]
CM_G = [["group", "cm_g", None], ["group__node", ["cm_g", "elec"], None]]

_RUNS: dict = {}
_WORK: dict = {}


def _run(tmp_path_factory, veh: str, values: list, *, warm: bool = True) -> dict:
    from flextool.engine import run_chain_from_db

    key = json.dumps([veh, values, warm])
    if key not in _RUNS:
        entities: list = []
        if veh == "roll":
            fj, chain, entities = _deterministic_rolling_json(tmp_path_factory), \
                _chain(STOCHASTICS_JSON, "1_week_rolling_wind"), ELEC_G
        elif veh == "stoch":
            fj, chain, entities = STOCHASTICS_JSON, _chain(STOCHASTICS_JSON, "2_day_stochastic_dispatch"), ELEC_G
        elif veh == "invdisp":
            fj, chain = TESTS_JSON, _chain(TESTS_JSON, "5weeks_invest_fullYear_dispatch_coal_wind")
        elif veh == "lookahead":
            fj = TESTS_JSON
            chain = ["init", "west", "coal", "coal_invest", "wind", "wind_invest", "5weeks"]
        elif veh == "nested":
            fj, chain = TESTS_JSON, _chain(TESTS_JSON, "multi_fullYear_battery_nested_multi_invest")
        elif veh == "recourse":
            fj, chain, entities = RECOURSE_JSON, _chain(RECOURSE_JSON, "recourse"), CM_G
        else:
            raise ValueError(veh)
        alt = f"c11b_{len(_RUNS)}"
        url, scen = make_db(tmp_path_factory, alt=alt, values=values, base_chain=chain, fixture_json=fj,
                            entities=entities)
        work = tmp_path_factory.mktemp(f"{alt}_w")
        steps = run_chain_from_db(url, scen, work_folder=work, keep_solutions=True, warm=warm)
        assert steps, "run_chain_from_db produced no orchestration steps"
        for step in steps.values():
            assert step.solution is not None, "solve produced no solution"
        _RUNS[key] = steps
        _WORK[id(steps)] = work
    return _RUNS[key]


def _rows(step) -> int:
    over = cstr_over(build(step.flex_data), "capacityMargin")
    return 0 if over is None else over.height


def _slack(step) -> dict[str, float]:
    """``period -> vq_capacity_margin`` of the solve ({} when the variable does not exist)."""
    try:
        v = step.solution.value("vq_capacity_margin")
    except KeyError:
        return {}
    if v is None:
        return {}
    return {d: float(x) for d, x in utf8(v).select("d", "value").iter_rows()}


def _objs(steps: dict) -> list[float]:
    return [float(st.solution.obj) for st in steps.values()]


def _assert_same_objs(a: dict, b: dict) -> None:
    oa, ob = _objs(a), _objs(b)
    assert len(oa) == len(ob)
    for x, y in zip(oa, ob):
        assert x == pytest.approx(y, rel=1e-9)


# ---------------------------------------------------------------------------
# Solves without investment periods: no capacity margin


def test_no_margin_in_rolling_dispatch(tmp_path_factory) -> None:
    """Deterministic rolling dispatch: no row or slack in any roll; every roll equals the model without the
    margin.  Before: 96 rows and roll 0 objective 483 152 446 066.18 (vs 21 107 366.18)."""
    steps = _run(tmp_path_factory, "roll", margin("elec_g"))
    assert len(steps) == 7
    for name, st in steps.items():
        assert _rows(st) == 0, name
        assert _slack(st) == {}, name
    _assert_same_objs(steps, _run(tmp_path_factory, "roll", []))


def test_no_margin_in_stochastic_dispatch(tmp_path_factory) -> None:
    """``2_day_stochastic_dispatch``: no rows (before 240), objective of the model without the margin."""
    steps = _run(tmp_path_factory, "stoch", margin("elec_g"))
    st = list(steps.values())[-1]
    assert _rows(st) == 0
    assert _slack(st) == {}
    _assert_same_objs(steps, _run(tmp_path_factory, "stoch", []))


def test_margin_in_invest_solve_not_in_dispatch_solve(tmp_path_factory) -> None:
    """Invest solve → full-year dispatch solve: the invest solve has 20 rows and one slack, the dispatch solve
    none and objective 0.  Before: the dispatch solve had 72 rows and paid 65 192 500 000 of margin penalty."""
    steps = _run(tmp_path_factory, "invdisp", margin("capacity_margin"))
    inv, disp = steps["invest_1year_5weeks"], steps["y2020_fullYear_dispatch"]
    assert _rows(inv) == 20
    assert set(_slack(inv)) == {"p2020"}
    assert _rows(disp) == 0
    assert _slack(disp) == {}
    assert float(disp.solution.obj) == pytest.approx(0.0, abs=1e-6)


def test_margin_only_in_invest_periods(tmp_path_factory) -> None:
    """``y2020_5week`` with dt periods p2020 + p2025 and ``invest_periods [p2020]``: rows and slack only for
    p2020 (before: 40 rows, slacks on both periods)."""
    steps = _run(tmp_path_factory, "lookahead",
                 margin("capacity_margin") + [["solve", "y2020_5week", "invest_periods", periods_array(["p2020"])]])
    st = list(steps.values())[-1]
    over = cstr_over(build(st.flex_data), "capacityMargin")
    assert over.height == 20
    assert set(over["d"].to_list()) == {"p2020"}
    assert set(_slack(st)) == {"p2020"}


def test_nested_chain_margin_in_invest_sub_solves_only(tmp_path_factory) -> None:
    """Nested multi-invest chain: each ``invest_5weeks_<p>`` sub-solve keeps its rows for its own invest
    periods (20 rows per period); the storage and dispatch children have none."""
    steps = _run(tmp_path_factory, "nested",
                 [["group", "capacity_margin", "capacity_margin", scalar(100.0)],
                  ["group", "capacity_margin", "capacity_margin_method", s("manual")]])
    invest = {n: st for n, st in steps.items() if n.startswith("invest_")}
    assert sorted(invest) == ["invest_5weeks_p2020", "invest_5weeks_p2025", "invest_5weeks_p2030",
                              "invest_5weeks_p2035"]
    others = {n: st for n, st in steps.items() if not n.startswith("invest_")}
    assert others
    for name, st in invest.items():
        periods = sorted(set(utf8(st.flex_data.dt)["d"].to_list()))
        assert _rows(st) == 20 * len(periods), name
        assert sorted(_slack(st)) == periods, name
    for name, st in others.items():
        assert _rows(st) == 0, name
        assert _slack(st) == {}, name
    for name, st in invest.items():
        assert sorted(utf8(st.flex_data.period_invest)["d"].to_list()) == sorted(_slack(st)), name
    for name, st in others.items():
        assert st.flex_data.period_invest is not None and st.flex_data.period_invest.height == 0, name


def test_rolling_with_investment_builds_margin_in_every_roll(tmp_path_factory) -> None:
    """A rolling solve with ``invest_periods [period1]``: every roll's dt is in period1, so every roll builds the
    margin (24 rows per day of the roll's horizon) and has its slack."""
    steps = _run(tmp_path_factory, "roll",
                 margin("elec_g") + [["solve", "1week_rolling", "invest_periods", periods_array(["period1"])]])
    assert len(steps) == 7
    for name, st in steps.items():
        assert _rows(st) == st.flex_data.dt.height, name
        assert set(_slack(st)) == {"period1"}, name


def test_recourse_keeps_branch_periods(tmp_path_factory) -> None:
    """Recourse investment: the margin keeps its rows in the forecast branch periods; the period labels p2035 /
    p2040 have no dispatch, so no margin row: 4 branch periods × 3 steps = 12 rows (18 when the labels had
    rows), slacks on the four branch periods."""
    steps = _run(tmp_path_factory, "recourse", margin("cm_g"))
    st = list(steps.values())[-1]
    assert _rows(st) == 12
    assert sorted(_slack(st)) == ["p2035_high", "p2035_low", "p2040_high", "p2040_low"]


# ---------------------------------------------------------------------------
# Outputs


def _write_outputs(steps: dict, name: str, monkeypatch) -> Path:
    from flextool.engine.preprocess._flex_data_provider import FlexDataProvider
    from flextool.process_outputs.write_outputs import write_outputs

    workdir = _WORK[id(steps)]
    monkeypatch.chdir(workdir)
    last = next(reversed(steps.values()))
    provider = getattr(last, "flex_data_provider", None)
    if provider is not None:
        provider.snapshot_processed_inputs(workdir)
    write_outputs(
        scenario_name=name, output_location=str(workdir), subdir=name, output_config_path=OUTPUT_CONFIG,
        write_methods=["csv"], fallback_output_location=str(workdir), raw_output_dir=str(workdir / "output_raw"),
        solution=last.solution, solve_name=last.solve_name,
        solve_steps=[(st.solve_name, st.flex_data, st.effective_solution) for st in steps.values()],
        flex_data_provider=FlexDataProvider())
    return workdir / "output_csv" / name


def _csv_rows(path: Path) -> list[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def test_outputs_look_ahead_period_without_margin(tmp_path_factory, monkeypatch) -> None:
    """Look-ahead p2025 has no margin: the outputs are written, the p2025 slack cell is empty and its capacity
    margin penalty is 0."""
    steps = _run(tmp_path_factory, "lookahead",
                 margin("capacity_margin") + [["solve", "y2020_5week", "invest_periods", periods_array(["p2020"])]])
    out = _write_outputs(steps, "cm_lookahead", monkeypatch)
    slack = {r["period"]: r["capacity_margin"] for r in _csv_rows(out / "slack__capacity_margin__d.csv")}
    assert slack["p2025"] == ""
    assert slack["p2020"] != ""
    costs = {r["period"]: r["capacity margin penalty"] for r in _csv_rows(out / "annualized_costs__d.csv")}
    assert float(costs["p2025"]) == 0.0


def test_outputs_recourse(tmp_path_factory, monkeypatch) -> None:
    """Recourse with the margin: the outputs are written; the period labels p2035 / p2040 have no dispatch, so
    their slack cells are empty; every forecast branch period has its own slack in the LP, and none of those
    unrealized rows reaches the normal outputs (they hold realized rows only)."""
    steps = _run(tmp_path_factory, "recourse", margin("cm_g"))
    out = _write_outputs(steps, "cm_recourse", monkeypatch)
    slack = {r["period"]: r["cm_g"] for r in _csv_rows(out / "slack__capacity_margin__d.csv")}
    assert slack == {"p2035": "", "p2040": ""}
    (step,) = steps.values()
    lp = {r["d"]: r["value"] for r in utf8(step.solution.value("vq_capacity_margin")).iter_rows(named=True)
          if r["g"] == "cm_g"}
    assert all(lp[d] > 0 for d in ("p2035_high", "p2035_low", "p2040_high", "p2040_low"))


# ---------------------------------------------------------------------------
# Warm start


@pytest.mark.parametrize("invest", [False, True], ids=["dispatch", "invest"])
def test_rolling_margin_warm_equals_cold(tmp_path_factory, invest) -> None:
    """Rolling solve with a manual margin group (with and without invest periods): warm rolls equal cold."""
    vals = margin("elec_g")
    if invest:
        vals = vals + [["solve", "1week_rolling", "invest_periods", periods_array(["period1"])]]
    _assert_same_objs(_run(tmp_path_factory, "roll", vals, warm=True), _run(tmp_path_factory, "roll", vals, warm=False))
