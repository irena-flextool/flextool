"""``group.inertia_limit`` and ``group.non_synchronous_limit`` per time step (C-5).

Both limits take a constant, a period map, a time map or a period + time map.  The reader keeps the authored dims
(``pdtGroup_inertia_limit`` / ``pdtGroup_non_synchronous_limit``, dims ⊆ ``(g, d, t)``) and the rows read

    Σ inertia[g,d,t] + vq_inertia[g,d,t] · L[g,d,t] ≥ L[g,d,t]
    in_ns[g,d,t] − vq_non_synchronous[g,d,t] · Δt ≤ ns_limit[g,d,t] · icap[g,d] · out[g,d,t]

The criterion (CLAUDE.md, a parameter that accepts several shapes): a constant and the same value as a period,
time or period + time map (and as a period map with Spine's silent ``x`` index) give the same LP — every row,
column bound and objective coefficient — in a single, a rolling and a stochastic solve.

Vehicles (databases built from JSON, never a checked-in ``.sqlite``):
- ``coal_wind_inertia`` (``tests/fixtures/tests.json``): group electricity (west) with the inertia of coal_plant
  (coal_chp and water_pump also carry an inertia constant, but are not in the scenario); wind_plant made
  non-synchronous.  Single solve (48 h), the same solve rolled
  24 h / 12 h, and with the solve's ``new_stepduration`` = 2 h.
- ``2_day_stochastic_dispatch`` (``stochastics.json``): ``period1`` and four branch periods; a group ``sys``
  (demand_node) with coal_plant's inertia and wind_plant non-synchronous.  The branch periods carry the anchor's
  value.
- The binding-hour toy: ``tests.json`` init / west / coal (coal_plant 500 MW, the only supply) with a demand of
  600 MW, so coal_plant runs at its full 500 MW in every hour and cannot run up to buy inertia (no spare capacity:
  the over-production route through the downward slack is closed).  inertia_constant 1 → 500 MWs of inertia.
  electricity's limit is a time map without t0010; a second group g2 (west) has a constant limit (mixed shapes).
- The online toy: the same with coal_min_load (coal_plant has a linear online variable) and a demand of 300 MW,
  so the inertia comes from the units online, not from the flow.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import polars as pl
import pytest

from flextool.engine.input._param_shapes import promote_param_to_dt
from tests.engine._unit_db import RunCache, b64, build, flow, utf8

pytestmark = pytest.mark.solver

REPO = Path(__file__).resolve().parents[2]
TESTS_JSON = REPO / "tests" / "fixtures" / "tests.json"
STOCHASTICS_JSON = REPO / "tests" / "fixtures" / "stochastics.json"

T48 = [f"t{i:04d}" for i in range(1, 49)]
ANCHOR = "period1"
L_INERTIA = 200.0           # MWs
L_NS = 0.4                  # share
_RUNS = RunCache()


@pytest.fixture(scope="module", autouse=True)
def _free_runs():
    yield
    _RUNS.clear()
    _TOY.clear()


@pytest.fixture(autouse=True)
def _strict_autoscale(monkeypatch) -> None:
    """A missing autoscale registry entry raises instead of falling back to an unscaled LP."""
    monkeypatch.setenv("FLEXTOOL_AUTOSCALE_STRICT", "1")


# ---------------------------------------------------------------------------
# Values


def scalar(v: float) -> list:
    return b64(float(v), "float")


def pmap(values: dict, index_name: str | None = "period") -> list:
    """1-D map (``index_name=None``: Spine's silent ``x`` index)."""
    m = {"index_type": "str", "data": [[k, v] for k, v in values.items()]}
    if index_name is not None:
        m["index_name"] = index_name
    return b64(m, "map")


def tmap(values: dict) -> list:
    return pmap(values, index_name="time")


def ptmap(per: dict) -> list:
    """2-D map period -> time -> value."""
    return b64({"index_type": "str", "index_name": "period", "data": [
        [d, {"type": "map", "index_type": "str", "index_name": "time",
             "data": [[t, v] for t, v in tv.items()]}] for d, tv in per.items()]}, "map")


def _shapes(value: float, period: str, steps: list[str], *, silent: bool) -> dict[str, list]:
    """``value`` as a constant and as each equivalent map."""
    out = {"constant": scalar(value), "period": pmap({period: value}), "time": tmap({t: value for t in steps}),
           "period_time": ptmap({period: {t: value for t in steps}})}
    if silent:
        out["period_silent_x"] = pmap({period: value}, index_name=None)
        out["time_silent_x"] = pmap({t: value for t in steps}, index_name=None)
    return out


def _chain(fixture_json: Path, scenario: str) -> list[str]:
    rows = [r for r in json.loads(fixture_json.read_text())["scenario_alternatives"] if r[0] == scenario]
    nxt = {a: b for _, a, b in rows}
    alt = next(a for a in nxt if a not in set(nxt.values()))
    chain = []
    while alt is not None:
        chain.append(alt)
        alt = nxt.get(alt)
    return chain


# Single / rolling vehicle: coal_wind_inertia (electricity = west) with a non-synchronous limit on wind_plant.
DET_CHAIN = _chain(TESTS_JSON, "coal_wind_inertia")
ROLLING = [["solve", "y2020_2day_dispatch", "solve_mode", b64("rolling_window", "str")],
           ["solve", "y2020_2day_dispatch", "rolling_solve_horizon", scalar(24.0)],
           ["solve", "y2020_2day_dispatch", "rolling_solve_jump", scalar(12.0)]]
STEP2 = [["solve", "y2020_2day_dispatch", "new_stepduration", scalar(2.0)]]


def _det_values(inertia: list, ns: list) -> list:
    return [["group", "electricity", "inertia_limit", inertia],
            ["group", "electricity", "has_non_synchronous", b64("yes", "str")],
            ["group", "electricity", "non_synchronous_limit", ns],
            ["group", "electricity", "penalty_non_synchronous", scalar(4000.0)],
            ["unit__outputNode", ["wind_plant", "west"], "is_non_synchronous", b64("yes", "str")]]


# Stochastic vehicle: 2_day_stochastic_dispatch with a group sys (demand_node).
STOCH_CHAIN = _chain(STOCHASTICS_JSON, "2_day_stochastic_dispatch")
STOCH_ENTITIES = [["group", "sys", None], ["group__node", ["sys", "demand_node"], None]]


def _stoch_values(inertia: list, ns: list) -> list:
    return [["group", "sys", "has_inertia", b64("yes", "str")],
            ["group", "sys", "inertia_limit", inertia],
            ["group", "sys", "penalty_inertia", scalar(5000.0)],
            ["unit__outputNode", ["coal_plant", "demand_node"], "inertia_constant", scalar(4.0)],
            ["group", "sys", "has_non_synchronous", b64("yes", "str")],
            ["group", "sys", "non_synchronous_limit", ns],
            ["group", "sys", "penalty_non_synchronous", scalar(4000.0)],
            ["unit__outputNode", ["wind_plant", "demand_node"], "is_non_synchronous", b64("yes", "str")]]


def _run(tmp_path_factory, alt: str, fixture_json: Path, chain: list[str], values: list,
         entities: list | None = None) -> dict:
    steps, _, _ = _RUNS.get(tmp_path_factory, alt=alt, values=values, base_chain=chain, entities=entities,
                            fixture_json=fixture_json)
    assert steps, "run_chain_from_db produced no orchestration steps"
    for step in steps.values():
        assert step.solution is not None, "solve produced no solution"
    return steps


# ---------------------------------------------------------------------------
# LP and parameter views


def _lp(fd) -> dict:
    """The LP built from ``fd`` keyed by names: rows (bounds and coefficients) and columns (objective and
    bounds)."""
    m = build(fd).canonicalise()
    rows: dict = {nm: [float(m.row_lb[i]), float(m.row_ub[i]), {}] for i, nm in enumerate(m.row_names)}
    cols: dict = {}
    for c in range(m.n_cols):
        name = m.col_names[c]
        cols[name] = (float(m.col_obj[c]), float(m.col_lb[c]), float(m.col_ub[c]))
        for k in range(m.col_ptr[c], m.col_ptr[c + 1]):
            rows[m.row_names[m.row_idx[k]]][2][name] = float(m.val[k])
    return {"rows": rows, "cols": cols}


def _families(lp: dict) -> set[str]:
    return {nm.split("[", 1)[0] for nm in lp["rows"]}


def _cells(fd, field: str) -> dict[tuple[str, str, str], float]:
    """``(g, d, t) -> value`` of a group limit promoted onto the solve's ``dt``; fails on a duplicate cell."""
    f = utf8(promote_param_to_dt(getattr(fd, field), fd.dt).collect()).select("g", "d", "t", "value")
    assert not f.select("g", "d", "t").is_duplicated().any(), f"{field}: duplicate (g, d, t) cells"
    return {(g, d, t): float(v) for g, d, t, v in f.iter_rows()}


def _assert_same_model(steps: dict, ref: dict, label: str) -> None:
    """Same steps, same limit cells and the same LP (every row, column and objective coefficient)."""
    assert list(steps) == list(ref), f"{label}: steps {list(steps)} vs {list(ref)}"
    for name in ref:
        fd, fd_ref = steps[name].flex_data, ref[name].flex_data
        for field in ("pdtGroup_inertia_limit", "pdtGroup_non_synchronous_limit"):
            assert _cells(fd, field) == _cells(fd_ref, field), f"{label} {name}: {field} cells differ"
        lp, lp_ref = _lp(fd), _lp(fd_ref)
        assert {"inertia_constraint", "non_sync_constraint"} <= _families(lp_ref), f"{name}: rows missing"
        assert lp == lp_ref, f"{label} {name}: the LP differs from the constant's"
        assert float(steps[name].solution.obj) == pytest.approx(float(ref[name].solution.obj), rel=1e-9)


# ---------------------------------------------------------------------------
# Constant vs equivalent maps: single, rolling, stochastic


_DET_SHAPES = list(_shapes(L_INERTIA, "p2020", T48, silent=True))


@pytest.mark.parametrize("shape", [s for s in _DET_SHAPES if s != "constant"])
def test_single_solve_map_equals_constant(tmp_path_factory, shape) -> None:
    inertia, ns = (_shapes(v, "p2020", T48, silent=True) for v in (L_INERTIA, L_NS))
    ref = _run(tmp_path_factory, "c5_single_constant", TESTS_JSON, DET_CHAIN,
               _det_values(inertia["constant"], ns["constant"]))
    steps = _run(tmp_path_factory, f"c5_single_{shape}", TESTS_JSON, DET_CHAIN,
                 _det_values(inertia[shape], ns[shape]))
    fd = list(ref.values())[-1].flex_data
    assert set(_cells(fd, "pdtGroup_inertia_limit").values()) == {L_INERTIA}
    assert len(_cells(fd, "pdtGroup_inertia_limit")) == 48
    _assert_same_model(steps, ref, shape)


@pytest.mark.parametrize("shape", [s for s in _DET_SHAPES if s != "constant"])
def test_rolling_solve_map_equals_constant(tmp_path_factory, shape) -> None:
    inertia, ns = (_shapes(v, "p2020", T48, silent=True) for v in (L_INERTIA, L_NS))
    ref = _run(tmp_path_factory, "c5_roll_constant", TESTS_JSON, DET_CHAIN,
               ROLLING + _det_values(inertia["constant"], ns["constant"]))
    rolls = [k for k in ref if "_roll_" in k]
    assert len(rolls) >= 3, f"expected roll sub-solves, got {list(ref)}"
    steps = _run(tmp_path_factory, f"c5_roll_{shape}", TESTS_JSON, DET_CHAIN,
                 ROLLING + _det_values(inertia[shape], ns[shape]))
    _assert_same_model(steps, ref, shape)


@pytest.mark.parametrize("shape", ["period", "time", "period_time"])
def test_stochastic_solve_map_equals_constant(tmp_path_factory, shape) -> None:
    """The branch periods take the anchor's value of a period or period + time map."""
    inertia, ns = (_shapes(v, ANCHOR, T48, silent=False) for v in (L_INERTIA, L_NS))
    ref = _run(tmp_path_factory, "c5_stoch_constant", STOCHASTICS_JSON, STOCH_CHAIN,
               _stoch_values(inertia["constant"], ns["constant"]), STOCH_ENTITIES)
    fd = [s for k, s in ref.items() if "__realized_dispatch" not in k][-1].flex_data
    periods = set(utf8(fd.dt)["d"].to_list())
    assert len(periods) > 1, "expected the branch periods in the stochastic solve"
    assert {d for _, d, _ in _cells(fd, "pdtGroup_inertia_limit")} == periods
    steps = _run(tmp_path_factory, f"c5_stoch_{shape}", STOCHASTICS_JSON, STOCH_CHAIN,
                 _stoch_values(inertia[shape], ns[shape]), STOCH_ENTITIES)
    _assert_same_model(steps, ref, shape)


@pytest.mark.xfail(strict=True, raises=AssertionError, reason=(
    "Shared gap (also group.co2_price and the other time-map parameters read by resolve_param_shape): on the "
    "DB-direct path a solve with new_stepduration samples a time map at the first original step of each new step "
    "instead of taking the step mean; only the provider's averaged frames (profiles, inflow) are averaged."))
def test_solve_step_aggregation_takes_the_step_mean(tmp_path_factory) -> None:
    """With the solve's ``new_stepduration`` = 2 h, an hourly map alternating 100 / 300 MWs (inertia) and
    0.2 / 0.6 (non-synchronous) gives, on each 2 h step, the Δt-weighted mean: (1·100 + 1·300)/2 = 200 MWs and
    (0.2 + 0.6)/2 = 0.4, the LP of the constants 200 MWs / 0.4 (the solve-level new_stepduration description:
    the other time series are averaged).  Today the 2 h step takes the first hour's value (100 MWs / 0.2)."""
    ref = _run(tmp_path_factory, "c5_step2_constant", TESTS_JSON, DET_CHAIN,
               STEP2 + _det_values(scalar(L_INERTIA), scalar(L_NS)))
    alt_i = tmap({t: (100.0 if i % 2 == 0 else 300.0) for i, t in enumerate(T48)})
    alt_ns = tmap({t: (0.2 if i % 2 == 0 else 0.6) for i, t in enumerate(T48)})
    steps = _run(tmp_path_factory, "c5_step2_tmap", TESTS_JSON, DET_CHAIN, STEP2 + _det_values(alt_i, alt_ns))
    fd = list(steps.values())[-1].flex_data
    assert utf8(fd.dt).height == 24
    cells = _cells(fd, "pdtGroup_inertia_limit")
    assert len(cells) == 24 and all(v == pytest.approx(L_INERTIA, rel=1e-12) for v in cells.values())
    ns_cells = _cells(fd, "pdtGroup_non_synchronous_limit")
    assert all(v == pytest.approx(L_NS, rel=1e-12) for v in ns_cells.values())
    name = list(ref)[-1]
    lp, lp_ref = _lp(fd), _lp(ref[name].flex_data)
    assert lp.keys() == lp_ref.keys() and lp["rows"].keys() == lp_ref["rows"].keys()
    for row, (lo, hi, terms) in lp_ref["rows"].items():
        got = lp["rows"][row]
        assert (got[0], got[1]) == pytest.approx((lo, hi), rel=1e-12), row
        assert got[2].keys() == terms.keys(), row
        for col, v in terms.items():
            assert got[2][col] == pytest.approx(v, rel=1e-12), (row, col)
    for col, v in lp_ref["cols"].items():
        assert lp["cols"][col] == pytest.approx(v, rel=1e-12), col


# ---------------------------------------------------------------------------
# Time-varying non-synchronous limit (hand-derived right-hand side)

# west's inflow in tests.json (MW): its negative part is the group's real demand, the right-hand side's only term
# in coal_wind_inertia (no flow leaves west).
WEST_DEMAND = {"t0001": 589.0, "t0002": 537.0, "t0013": 572.0, "t0014": 564.0, "t0025": 436.0, "t0026": 446.0}
NS_MISSING = "t0007"


def _ns_alternating() -> list:
    """0.2 in the odd hours, 0.6 in the even hours, no value at t0007."""
    return tmap({t: (0.2 if i % 2 == 0 else 0.6) for i, t in enumerate(T48) if t != NS_MISSING})


def _ns_expected_ub(t: str) -> float:
    """Upper bound of non_sync_constraint[electricity, p2020, t]: wind's infeed − vq · Δt ≤ ns_limit[t] · icap ·
    demand[t], icap = 1 (the group has no invested capacity to scale by), Δt = 1 h."""
    return (0.2 if int(t[1:]) % 2 == 1 else 0.6) * WEST_DEMAND[t]


@pytest.mark.parametrize("mode", ["single", "rolling"])
def test_non_synchronous_limit_varies_by_hour(tmp_path_factory, mode) -> None:
    """Hand derivation: t0001 0.2 · 589 = 117.8 MWh, t0002 0.6 · 537 = 322.2 MWh, and in the rolls t0013 0.2 · 572 =
    114.4, t0014 0.6 · 564 = 338.4, t0025 0.2 · 436 = 87.2, t0026 0.6 · 446 = 267.6.  A limit joined to the
    wrong hour would swap 0.2 and 0.6.  At t0007 the map has no value, so the limit is 0: the right-hand side is
    0 and any wind infeed there is slack (groups.md, non_synchronous_limit)."""
    extra = ROLLING if mode == "rolling" else []
    steps = _run(tmp_path_factory, f"c5_ns_tmap_{mode}", TESTS_JSON, DET_CHAIN,
                 extra + _det_values(scalar(L_INERTIA), _ns_alternating()))
    checked = set()
    for step in steps.values():
        rows = _lp(step.flex_data)["rows"]
        for t in WEST_DEMAND.keys() | {NS_MISSING}:
            name = f"non_sync_constraint[electricity,p2020,{t}]"
            if name not in rows:
                continue
            lo, hi, terms = rows[name]
            want = 0.0 if t == NS_MISSING else _ns_expected_ub(t)
            assert (lo, hi) == (float("-inf"), pytest.approx(want, rel=1e-12, abs=1e-12)), name
            assert terms[f"v_flow[wind_plant,wind_plant,west,p2020,{t}]"] == pytest.approx(1000.0, rel=1e-12)
            assert terms[f"vq_non_synchronous[electricity,p2020,{t}]"] == pytest.approx(-1.0, rel=1e-12)
            checked.add(t)
    assert checked == WEST_DEMAND.keys() | {NS_MISSING}, checked
    if mode == "rolling":
        assert len([k for k in steps if "_roll_" in k]) >= 3


# ---------------------------------------------------------------------------
# Binding hour (hand-derived)

TOY_CHAIN = ["init", "west", "coal"]
HOUR = "t0005"
NO_LIMIT = "t0010"          # left out of electricity's time map: no requirement there
PENALTY = 5000.0            # CUR/MWs
G2_LIMIT = 300.0            # MWs, constant: met by coal_plant's 500 MWs in every hour
TOY_ENTITIES = [["group", "g2", None], ["group__node", ["g2", "west"], None]]


def _toy_values(hour: str = HOUR, *, missing: tuple[str, ...] = (NO_LIMIT,)) -> list:
    limit = tmap({t: (800.0 if t == hour else 200.0) for t in T48 if t not in missing})
    return [["node", "west", "inflow", scalar(-600.0)],
            ["group", "electricity", "has_inertia", b64("yes", "str")],
            ["group", "electricity", "inertia_limit", limit],
            ["group", "electricity", "penalty_inertia", scalar(PENALTY)],
            ["group", "g2", "has_inertia", b64("yes", "str")],
            ["group", "g2", "inertia_limit", scalar(G2_LIMIT)],
            ["group", "g2", "penalty_inertia", scalar(PENALTY)],
            ["unit__outputNode", ["coal_plant", "west"], "inertia_constant", scalar(1.0)]]


_TOY: dict = {}


def _toy_outputs(tmp_path_factory, kind: str = "toy") -> tuple[dict, Path]:
    """A toy solved once per module with its CSV outputs: ``(steps, folder of the CSV files)``."""
    if kind not in _TOY:
        if kind == "toy":
            _TOY[kind] = _toy_solve(tmp_path_factory, "c5_toy", _toy_values(), TOY_CHAIN, TOY_ENTITIES)
        else:
            _TOY[kind] = _toy_solve(tmp_path_factory, "c5_online", _online_values(), ONLINE_CHAIN, None)
    return _TOY[kind]


def _toy_solve(tmp_path_factory, alt: str, values: list, chain: list[str], entities) -> tuple[dict, Path]:
    from flextool.engine import run_chain_from_db
    from flextool.process_outputs.write_outputs import write_outputs
    from tests.engine._unit_db import make_db

    url, scen = make_db(tmp_path_factory, alt=alt, values=values, base_chain=chain, entities=entities)
    work = tmp_path_factory.mktemp(f"{alt}_out")
    steps = run_chain_from_db(url, scen, work_folder=work, keep_solutions=True)
    last = list(steps.values())[-1]
    write_outputs(scenario_name=scen, output_location=str(work), subdir=scen, write_methods=["csv"],
                  fallback_output_location=str(work), raw_output_dir=str(work / "output_raw"),
                  solution=last.solution, solve_name=last.solve_name,
                  solve_steps=[(s.solve_name, s.flex_data, s.effective_solution) for s in steps.values()],
                  flex_data_provider=last.flex_data_provider)
    return steps, work / "output_csv" / scen


def _csv(folder: Path, name: str) -> list[dict]:
    with open(folder / name, newline="") as fh:
        return list(csv.DictReader(fh))


def _vq(step) -> dict[tuple[str, str], float]:
    return {(g, t): float(v) for g, t, v in
            utf8(step.solution.value("vq_inertia")).select("g", "t", "value").iter_rows()}


def test_time_map_binds_in_one_hour_only(tmp_path_factory) -> None:
    """Hand derivation (oracle in the design's critique): coal_plant gives 1 · 500 MW = 500 MWs in every hour.
    Outside t0005: 500 + vq · 200 ≥ 200 → vq = 0.  At t0005: 500 + vq · 800 ≥ 800 → vq = 300/800 = 0.375, a
    shortfall of 300 MWs, priced 300 · 5000 · Δt (1 h) · timestep_weight (1) = 1.5e6 CUR in that hour.
    At t0010 the map has no value: no requirement, the slack is reported as 0 MWs (not blank) and costs 0.
    g2 (constant 300 MWs ≤ 500 MWs) has no shortfall."""
    steps, out = _toy_outputs(tmp_path_factory)
    step = list(steps.values())[-1]
    coal = flow(step, "coal_plant", "coal_market", "west")
    assert coal.len() == 48 and all(v == pytest.approx(500.0, rel=1e-9) for v in coal.to_list())
    vq = _vq(step)
    assert set(vq) == {(g, t) for g in ("electricity", "g2") for t in T48}
    assert vq[("electricity", HOUR)] == pytest.approx(0.375, rel=1e-9)
    # vq at t0010 is in no row and has no cost (its value is arbitrary): left out here.
    assert all(abs(v) < 1e-9 for (g, t), v in vq.items() if (g, t) not in {("electricity", HOUR),
                                                                             ("electricity", NO_LIMIT)})
    slack = {r["time"]: (float(r["electricity"]), float(r["g2"])) for r in _csv(out, "slack__inertia_group__dt.csv")}
    assert set(slack) == set(T48)
    assert slack[HOUR][0] == pytest.approx(300.0, rel=1e-9)
    assert slack[NO_LIMIT] == (0.0, 0.0)
    assert all(abs(v) < 1e-6 for t, (e, g2) in slack.items() for v in ((e, g2) if t != HOUR else (g2,)))
    cost = {r["time"]: float(r["inertia slack penalty"]) for r in _csv(out, "costs__dt.csv")}
    assert cost[HOUR] == pytest.approx(300.0 * PENALTY * 1.0 * 1.0, rel=1e-9)
    assert cost[NO_LIMIT] == 0.0
    assert all(abs(v) < 1e-6 for t, v in cost.items() if t != HOUR)


def test_mixed_shapes_and_a_time_map_without_an_hour(tmp_path_factory) -> None:
    """One parameter, two shapes: g2's constant reaches every (d, t), electricity's time map its own hours (800 at
    t0005, 200 elsewhere, none at t0010).  The t0010 row of electricity is Σ inertia ≥ 0 without the slack."""
    steps, _ = _toy_outputs(tmp_path_factory)
    fd = list(steps.values())[-1].flex_data
    cells = _cells(fd, "pdtGroup_inertia_limit")
    assert {t: v for (g, _, t), v in cells.items() if g == "g2"} == {t: G2_LIMIT for t in T48}
    assert {t: v for (g, _, t), v in cells.items() if g == "electricity"} == {
        t: (800.0 if t == HOUR else 200.0) for t in T48 if t != NO_LIMIT}
    rows = _lp(fd)["rows"]
    lo, hi, terms = rows[f"inertia_constraint[electricity,p2020,{NO_LIMIT}]"]
    assert (lo, hi) == (0.0, float("inf"))
    assert f"vq_inertia[electricity,p2020,{NO_LIMIT}]" not in terms
    assert terms["v_flow[coal_plant,coal_market,west,p2020,t0010]"] == pytest.approx(500.0, rel=1e-12)
    lo, _, terms = rows[f"inertia_constraint[electricity,p2020,{HOUR}]"]
    assert lo == pytest.approx(800.0, rel=1e-12)
    assert terms[f"vq_inertia[electricity,p2020,{HOUR}]"] == pytest.approx(800.0, rel=1e-12)
    lo, _, terms = rows["inertia_constraint[g2,p2020,t0010]"]
    assert lo == pytest.approx(G2_LIMIT, rel=1e-12)
    assert terms["vq_inertia[g2,p2020,t0010]"] == pytest.approx(G2_LIMIT, rel=1e-12)


ROLL_HOUR = "t0030"


def test_rolling_time_map_binds_in_its_hour(tmp_path_factory) -> None:
    """The toy rolled (24 h horizon, 12 h jump: windows t0001-24, t0013-36, t0025-48, t0037-48) with the 800 MWs
    hour at t0030 (no hole in the map): the two windows holding t0030 (the 2nd and 3rd roll) have vq = 0.375 there
    and 0 elsewhere, the other rolls have no shortfall.  A roll whose t axis were shifted would bind another
    hour."""
    steps = _run(tmp_path_factory, "c5_toy_roll", TESTS_JSON, TOY_CHAIN,
                 ROLLING + _toy_values(ROLL_HOUR, missing=()), TOY_ENTITIES)
    rolls = [k for k in steps if "_roll_" in k]
    assert len(rolls) == 4, list(steps)
    binding = []
    for name in rolls:
        vq = _vq(steps[name])
        if ("electricity", ROLL_HOUR) in vq:
            binding.append(name)
            assert vq[("electricity", ROLL_HOUR)] == pytest.approx(0.375, rel=1e-9), name
        assert all(abs(v) < 1e-9 for k, v in vq.items() if k != ("electricity", ROLL_HOUR)), name
    assert binding == rolls[1:3]


def test_inertia_output_tables(tmp_path_factory) -> None:
    """The group inertia tables hold the inertia the row counts: coal_plant's flow 500 MW · inertia_constant 1 =
    500 MWs in every hour, in group_inertia__dt (electricity, g2) and group__unit__node_inertia__dt
    ((electricity | g2), coal_plant, west).  Before the fix both tables were empty (the (process, node) columns
    were compared with the group's node names)."""
    _steps, out = _toy_outputs(tmp_path_factory)
    for g in ("electricity", "g2"):
        group = {r["time"]: float(r[g]) for r in _csv(out, "group_inertia__dt.csv")}
        assert set(group) == set(T48) and all(v == pytest.approx(500.0, rel=1e-9) for v in group.values()), g
    with open(out / "group__unit__node_inertia__dt.csv", newline="") as fh:
        rows = list(csv.reader(fh))
    cols = list(zip(rows[0][3:], rows[1][3:], rows[2][3:]))
    assert sorted(cols) == [("electricity", "coal_plant", "west"), ("g2", "coal_plant", "west")]
    for j in range(len(cols)):
        data = {r[2]: float(r[3 + j]) for r in rows[3:]}
        assert set(data) == set(T48) and all(v == pytest.approx(500.0, rel=1e-9) for v in data.values())


# ---------------------------------------------------------------------------
# Inertia from units online (hand-derived)

ONLINE_CHAIN = ["init", "west", "coal", "coal_min_load"]
ONLINE_LIMIT = 1500.0       # MWs
ONLINE_CONSTANT = 2.0       # s


def _online_values() -> list:
    return [["node", "west", "inflow", scalar(-300.0)],
            ["group", "electricity", "has_inertia", b64("yes", "str")],
            ["group", "electricity", "inertia_limit", scalar(ONLINE_LIMIT)],
            ["group", "electricity", "penalty_inertia", scalar(PENALTY)],
            ["unit__outputNode", ["coal_plant", "west"], "inertia_constant", scalar(ONLINE_CONSTANT)]]


def test_inertia_output_of_units_online(tmp_path_factory) -> None:
    """coal_plant (500 MW, one unit of 500 MW, linear online) meets a demand of 300 MW.  Its inertia counts the
    capacity online: the shortfall costs 5000 CUR/MWs, so the unit stays fully online (online = 1, 500 MW), far
    cheaper than the shortfall.  Inertia = online 1 · unitsize 500 MW · inertia_constant 2 s = 1000 MWs, in
    group_inertia__dt and group__unit__node_inertia__dt (a flow-based count would give 300 · 2 = 600 MWs);
    the shortfall is 1500 − 1000 = 500 MWs."""
    steps, out = _toy_outputs(tmp_path_factory, "online")
    step = list(steps.values())[-1]
    on = utf8(step.solution.value("v_online_linear")).filter(pl.col("p") == "coal_plant")["value"]
    assert on.len() == 48 and all(v == pytest.approx(1.0, rel=1e-9) for v in on.to_list())
    coal = flow(step, "coal_plant", "coal_market", "west")
    assert all(v == pytest.approx(300.0, rel=1e-9) for v in coal.to_list())
    group = {r["time"]: float(r["electricity"]) for r in _csv(out, "group_inertia__dt.csv")}
    assert set(group) == set(T48) and all(v == pytest.approx(1000.0, rel=1e-9) for v in group.values())
    with open(out / "group__unit__node_inertia__dt.csv", newline="") as fh:
        rows = list(csv.reader(fh))
    assert (rows[0][3:], rows[1][3:], rows[2][3:]) == (["electricity"], ["coal_plant"], ["west"])
    assert all(float(r[3]) == pytest.approx(1000.0, rel=1e-9) for r in rows[3:]) and len(rows) == 3 + 48
    slack = [float(r["electricity"]) for r in _csv(out, "slack__inertia_group__dt.csv")]
    assert all(v == pytest.approx(500.0, rel=1e-9) for v in slack)


# ---------------------------------------------------------------------------
# The output's limit lookup (calc_slacks)


def _frame(times: list, values: list, *, names=("period", "time")):
    import pandas as pd

    idx = pd.MultiIndex.from_tuples([("p2020", t) for t in times], names=list(names))
    return pd.DataFrame({"electricity": values}, index=idx)


def test_slack_limit_lookup_fills_only_the_steps_without_a_limit() -> None:
    """A step the limit leaves out reads 0; a step it gives keeps its value (the shortfall survives)."""
    from flextool.process_outputs.calc_slacks import inertia_limit_on_slack_cells

    q = _frame(["t0001", "t0002", "t0003"], [0.0, 0.375, 0.5])
    got = inertia_limit_on_slack_cells(_frame(["t0002", "t0003"], [800.0, 200.0]), q)
    assert got["electricity"].tolist() == [0.0, 800.0, 200.0]
    assert q.mul(got)["electricity"].tolist() == [0.0, 300.0, 100.0]


@pytest.mark.parametrize("limit", [
    _frame([1, 2], [800.0, 200.0]),                                  # time labels of another type
    _frame(["T0001", "T0002"], [800.0, 200.0]),                      # another spelling
    _frame(["t0001", "t0002"], [800.0, 200.0], names=("d", "t")),    # other level names
], ids=["int_times", "spelling", "level_names"])
def test_slack_limit_lookup_rejects_a_misaligned_limit(limit) -> None:
    """A limit with values whose index does not line up with the slack's raises instead of reading 0 everywhere
    (which would report every shortfall as 0 MWs)."""
    from flextool.process_outputs.calc_slacks import inertia_limit_on_slack_cells

    with pytest.raises(ValueError, match="group_inertia_limit"):
        inertia_limit_on_slack_cells(limit, _frame(["t0001", "t0002"], [0.375, 0.0]))
