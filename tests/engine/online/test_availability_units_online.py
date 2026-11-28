"""Availability derates a unit's built capacity, not the output of a unit online.

maxFlow: v_flow ≤ κ · availability · capacity (unchanged); maxFlow_online: v_flow ≤ κ · v_online (no availability:
a unit online has its full capacity); maxOnline: v_online ≤ the units built (no availability); minFlow_minload:
Σ v_flow ≥ min_load · v_online (unchanged).  So a derated unit can still be online: its output is limited by the
derated capacity, its units online by min_load · v_online ≤ that output.

Fixture: ``tests.json`` (``coal_plant``, the only unit of the ``init, west, coal`` chain, 48 one-hour steps) with
``virtual_unitsize`` 125 MW and ``existing`` 4 (units): 500 MW, ``startup_cost`` 100 (start-ups cost, so the LP does
not start and stop units for nothing).  West demand 1000 MW in every step, above what the plant can give
(penalty_up 900 €/MWh, far above its fuel and start-up costs), so the plant runs at the most it can.  Flows are in
units of unitsize (125 MW).  Hand-derived oracles:

* linear online, min_load 0.9, availability 0.75 at t0001–t0024 and 1 after: flow = 0.75 · 4 = 3 units (375 MW, the
  derated capacity) at t0001–t0024, 4 after; units online 3 ≤ online ≤ 3 / 0.9 = 3.33 in the outage (flow ≤ online
  and flow ≥ 0.9 · online); the least start-up cost keeps the most online, 10/3, so the return to 4 units at t0025
  starts 2/3 of a unit (the min_load floor forces it).  Before: each unit online was capped at 0.75 of its
  capacity, below its min_load 0.9: no unit could be online, the flow was 0.
* the same with min_load 0.5: the outage forces nothing (0.5 · 4 = 2 ≤ 3): 4 units stay online, no start-up.
* integer online, availability 0.75: flow 3, online 3 (3 ≤ online ≤ 3.33); availability 0.7: flow 2.8, online 3
  (2.8 ≤ online ≤ 3.11).
* one integer unit of 500 MW (no virtual_unitsize, existing 500), availability 0.95, min_load 0.5: online 1, flow
  0.95 (the derated capacity).
* no online variables, availability 0.75: flow 3 units (unchanged).
* availability authored as 1 gives the same LP rows and objective as no availability.
"""
from __future__ import annotations

import polars as pl
import pytest

from tests.engine._lp_rows import row_bounds, row_terms
from tests.engine._unit_db import PLANT, RunCache, b64, build, unitsize, utf8

pytestmark = pytest.mark.solver

CHAIN = ["init", "west", "coal"]
STEPS = [f"t{i:04d}" for i in range(1, 49)]
OUTAGE = {t: (0.75 if t <= "t0024" else 1.0) for t in STEPS}
_RUNS = RunCache()


@pytest.fixture(scope="module", autouse=True)
def _free_runs():
    yield
    _RUNS.clear()


@pytest.fixture(autouse=True)
def _strict_autoscale(monkeypatch):
    monkeypatch.setenv("FLEXTOOL_AUTOSCALE_STRICT", "1")


def _tmap(values: dict[str, float]) -> list:
    return b64({"index_type": "str", "index_name": "time", "data": [[t, v] for t, v in values.items()]}, "map")


def _model(startup: str, availability, min_load: float | None = 0.9, units: float | None = 4.0) -> list:
    vals = [["unit", PLANT, "startup_method", b64(startup, "str")],
            ["unit", PLANT, "startup_cost", b64(100.0, "float")],
            ["node", "west", "inflow", b64(-1000.0, "float")]]
    if units is not None:
        vals += [["unit", PLANT, "virtual_unitsize", b64(125.0, "float")],
                 ["unit", PLANT, "existing", b64(units, "float")]]
    if min_load is not None:
        vals.append(["unit", PLANT, "min_load", b64(min_load, "float")])
    if availability is not None:
        vals.append(["unit", PLANT, "availability",
                     _tmap(availability) if isinstance(availability, dict) else b64(availability, "float")])
    return vals


def _solve(tmp_path_factory, alt: str, values: list):
    steps, _wf, _scen = _RUNS.get(tmp_path_factory, alt=alt, values=values, base_chain=CHAIN)
    (step,) = steps.values()
    assert step.solution.optimal
    return step


def _by_t(step, var: str) -> dict[str, float]:
    df = utf8(step.solution.value(var)).filter(pl.col("p") == PLANT)
    if "sink" in df.columns:
        df = df.filter(pl.col("sink") == "west")
    return {t: v for t, v in zip(df["t"], df["value"])}


def _t(row: str) -> str:
    return row.rsplit(",", 1)[1][:-1]


def test_linear_units_online_run_within_the_derated_capacity(tmp_path_factory) -> None:
    step = _solve(tmp_path_factory, "av_linear", _model("linear", OUTAGE))
    assert unitsize(step, PLANT) == pytest.approx(125.0)
    online, flow = _by_t(step, "v_online_linear"), _by_t(step, "v_flow")
    startup = _by_t(step, "v_startup_linear")
    for t in STEPS:
        if t <= "t0024":
            assert flow[t] == pytest.approx(3.0, abs=1e-6), t
            assert online[t] == pytest.approx(10.0 / 3.0, abs=1e-6), t
        else:
            assert flow[t] == pytest.approx(4.0, abs=1e-6), t
            assert online[t] == pytest.approx(4.0, abs=1e-6), t
    assert sum(startup.values()) == pytest.approx(4.0 - 10.0 / 3.0, abs=1e-6)
    assert startup["t0025"] == pytest.approx(4.0 - 10.0 / 3.0, abs=1e-6)
    # The LP: availability is in maxFlow's bound (0.75 · 4 units), not in maxOnline (4) nor in maxFlow_online.
    pb = build(step.flex_data)
    flow_ub = {_t(r): ub for r, (_lb, ub) in row_bounds(pb, "maxFlow", f"{PLANT},coal_market,west,").items()}
    assert flow_ub == pytest.approx({t: 4.0 * OUTAGE[t] for t in STEPS})
    on_ub = {_t(r): ub for r, (_lb, ub) in row_bounds(pb, "maxOnline_linear", f"{PLANT},").items()}
    assert on_ub == pytest.approx({t: 4.0 for t in STEPS})
    for row, terms in row_terms(pb, "maxFlow_online_linear", f"{PLANT},").items():
        assert {c: v for c, v in terms.items() if c.startswith("v_online")} == {
            f"v_online_linear[{PLANT},p2020,{_t(row)}]": -1.0}, row


def test_outage_above_min_load_starts_no_units(tmp_path_factory) -> None:
    step = _solve(tmp_path_factory, "av_linear_ml5", _model("linear", OUTAGE, min_load=0.5))
    online, flow = _by_t(step, "v_online_linear"), _by_t(step, "v_flow")
    for t in STEPS:
        assert flow[t] == pytest.approx(3.0 if t <= "t0024" else 4.0, abs=1e-6), t
        assert online[t] == pytest.approx(4.0, abs=1e-6), t
    assert sum(_by_t(step, "v_startup_linear").values()) == pytest.approx(0.0, abs=1e-6)
    assert sum(_by_t(step, "v_shutdown_linear").values()) == pytest.approx(0.0, abs=1e-6)


@pytest.mark.parametrize("availability, out, units", [(0.75, 3.0, 3.0), (0.7, 2.8, 3.0)])
def test_integer_units_online_within_the_derated_capacity(tmp_path_factory, availability, out, units) -> None:
    step = _solve(tmp_path_factory, f"av_integer_{availability}", _model("integer", availability))
    online, flow = _by_t(step, "v_online_integer"), _by_t(step, "v_flow")
    for t in STEPS:
        assert online[t] == pytest.approx(units, abs=1e-6), t
        assert flow[t] == pytest.approx(out, abs=1e-6), t


def test_single_integer_unit_stays_online_derated(tmp_path_factory) -> None:
    step = _solve(tmp_path_factory, "av_integer_single", _model("integer", 0.95, min_load=0.5, units=None))
    assert unitsize(step, PLANT) == pytest.approx(500.0)
    online, flow = _by_t(step, "v_online_integer"), _by_t(step, "v_flow")
    for t in STEPS:
        assert online[t] == pytest.approx(1.0, abs=1e-6), t
        assert flow[t] == pytest.approx(0.95, abs=1e-6), t


def test_unit_without_online_variables_is_derated(tmp_path_factory) -> None:
    step = _solve(tmp_path_factory, "av_no_online", _model("no_startup", 0.75, min_load=None))
    online_set = step.flex_data.process_online_linear
    assert online_set is None or PLANT not in utf8(online_set)["p"].to_list()
    flow = _by_t(step, "v_flow")
    for t in STEPS:
        assert flow[t] == pytest.approx(3.0, abs=1e-6), t


@pytest.mark.parametrize("startup", ["linear", "integer"])
def test_availability_one_is_no_availability(tmp_path_factory, startup) -> None:
    with_one = _solve(tmp_path_factory, f"av_one_{startup}", _model(startup, 1.0))
    without = _solve(tmp_path_factory, f"av_none_{startup}", _model(startup, None))
    pb1, pb0 = build(with_one.flex_data), build(without.flex_data)
    for fam in (f"maxOnline_{startup}", f"maxFlow_online_{startup}", f"minFlow_minload_{startup}", "maxFlow"):
        assert row_terms(pb1, fam) == row_terms(pb0, fam), fam
        assert row_bounds(pb1, fam) == row_bounds(pb0, fam), fam
    assert with_one.solution.obj == pytest.approx(without.solution.obj, rel=1e-12)
