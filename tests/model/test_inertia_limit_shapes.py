"""``group.inertia_limit`` and ``group.non_synchronous_limit`` per time step (C-5).

Both limits take a constant, a period map, a time map or a period + time map.  The reader keeps the authored dims
(``pdtGroup_inertia_limit`` / ``pdtGroup_non_synchronous_limit``, dims ⊆ ``(g, d, t)``) and the rows read

    Σ inertia[g,d,t] + vq_inertia[g,d,t] · L[g,d,t] ≥ L[g,d,t]
    in_ns[g,d,t] − vq_non_synchronous[g,d,t] · Δt ≤ ns_limit[g,d,t] · icap[g,d] · out[g,d,t]

The criterion (CLAUDE.md, a parameter that accepts several shapes): a constant and the same value as a period,
time or period + time map (and as a period map with Spine's silent ``x`` index) give the same LP — every row,
column bound and objective coefficient — in a single, a rolling and a stochastic solve.

Vehicles (databases built from JSON, never a checked-in ``.sqlite``):
- ``coal_wind_inertia`` (``tests/fixtures/tests.json``): group electricity (west) with the inertia of coal_plant,
  coal_chp and water_pump; wind_plant made non-synchronous.  Single solve (48 h), the same solve rolled
  24 h / 12 h, and with the solve's ``new_stepduration`` = 2 h.
- ``2_day_stochastic_dispatch`` (``stochastics.json``): ``period1`` and four branch periods; a group ``sys``
  (demand_node) with coal_plant's inertia and wind_plant non-synchronous.  The branch periods carry the anchor's
  value.
- The binding-hour toy: ``tests.json`` init / west / coal (coal_plant 500 MW, the only supply) with a demand of
  600 MW, so coal_plant runs at its full 500 MW in every hour and cannot run up to buy inertia (no spare capacity:
  the over-production route through the downward slack is closed).  inertia_constant 1 → 500 MWs of inertia.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

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


@pytest.mark.xfail(strict=True, reason=(
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
# Binding hour (hand-derived)

TOY_CHAIN = ["init", "west", "coal"]
HOUR = "t0005"
PENALTY = 5000.0            # CUR/MWs


def _toy_values() -> list:
    limit = tmap({t: (800.0 if t == HOUR else 200.0) for t in T48})
    return [["node", "west", "inflow", scalar(-600.0)],
            ["group", "electricity", "has_inertia", b64("yes", "str")],
            ["group", "electricity", "inertia_limit", limit],
            ["group", "electricity", "penalty_inertia", scalar(PENALTY)],
            ["unit__outputNode", ["coal_plant", "west"], "inertia_constant", scalar(1.0)]]


_TOY: dict = {}


def _toy_outputs(tmp_path_factory) -> tuple[dict, Path]:
    """The toy solved once per module with its CSV outputs: ``(steps, folder of the CSV files)``."""
    if "out" not in _TOY:
        _TOY["out"] = _toy_solve(tmp_path_factory)
    return _TOY["out"]


def _toy_solve(tmp_path_factory) -> tuple[dict, Path]:
    from flextool.engine import run_chain_from_db
    from flextool.process_outputs.write_outputs import write_outputs
    from tests.engine._unit_db import make_db

    url, scen = make_db(tmp_path_factory, alt="c5_toy", values=_toy_values(), base_chain=TOY_CHAIN)
    work = tmp_path_factory.mktemp("c5_toy_out")
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


def test_time_map_binds_in_one_hour_only(tmp_path_factory) -> None:
    """Hand derivation (oracle in the design's critique): coal_plant gives 1 · 500 MW = 500 MWs in every hour.
    Outside t0005: 500 + vq · 200 ≥ 200 → vq = 0.  At t0005: 500 + vq · 800 ≥ 800 → vq = 300/800 = 0.375, a
    shortfall of 300 MWs, priced 300 · 5000 · Δt (1 h) · timestep_weight (1) = 1.5e6 CUR in that hour."""
    steps, out = _toy_outputs(tmp_path_factory)
    step = list(steps.values())[-1]
    coal = flow(step, "coal_plant", "coal_market", "west")
    assert coal.len() == 48 and all(v == pytest.approx(500.0, rel=1e-9) for v in coal.to_list())
    vq = {t: float(v) for t, v in utf8(step.solution.value("vq_inertia")).select("t", "value").iter_rows()}
    assert set(vq) == set(T48)
    assert vq[HOUR] == pytest.approx(0.375, rel=1e-9)
    assert all(abs(v) < 1e-9 for t, v in vq.items() if t != HOUR)
    slack = {r["time"]: float(r["electricity"]) for r in _csv(out, "slack__inertia_group__dt.csv")}
    assert set(slack) == set(T48)
    assert slack[HOUR] == pytest.approx(300.0, rel=1e-9)
    assert all(abs(v) < 1e-6 for t, v in slack.items() if t != HOUR)
    cost = {r["time"]: float(r["inertia slack penalty"]) for r in _csv(out, "costs__dt.csv")}
    assert cost[HOUR] == pytest.approx(300.0 * PENALTY * 1.0 * 1.0, rel=1e-9)
    assert all(abs(v) < 1e-6 for t, v in cost.items() if t != HOUR)


def test_inertia_output_tables(tmp_path_factory) -> None:
    """The group inertia tables hold the inertia the row counts: coal_plant's flow 500 MW · inertia_constant 1 =
    500 MWs in every hour, in group_inertia__dt (electricity) and group__unit__node_inertia__dt
    (electricity, coal_plant, west).  Before the fix both tables were empty (the (process, node) columns were
    compared with the group's node names)."""
    _steps, out = _toy_outputs(tmp_path_factory)
    group = {r["time"]: float(r["electricity"]) for r in _csv(out, "group_inertia__dt.csv")}
    assert set(group) == set(T48) and all(v == pytest.approx(500.0, rel=1e-9) for v in group.values())
    with open(out / "group__unit__node_inertia__dt.csv", newline="") as fh:
        rows = list(csv.reader(fh))
    assert rows[0][3:] == ["electricity"] and rows[1][3:] == ["coal_plant"] and rows[2][3:] == ["west"]
    data = {r[2]: float(r[3]) for r in rows[3:]}
    assert set(data) == set(T48) and all(v == pytest.approx(500.0, rel=1e-9) for v in data.values())
