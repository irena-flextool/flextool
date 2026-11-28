"""Availability is the share of a unit's units that is available (an outage), not a derating of each unit online.

maxOnline: v_online ≤ availability · (existing_count + invested − divested); maxFlow_online: v_flow ≤ κ · v_online
(no availability); minFlow_minload: Σ v_flow ≥ min_load · v_online.  A unit without online variables keeps
maxFlow: v_flow ≤ availability · capacity.

Fixture: ``tests.json`` (``coal_plant``, 500 MW at west, the only unit of the ``init, west, coal`` chain, 48 one-hour
steps) with ``virtual_unitsize`` 125 MW and ``existing`` 4 (units): 500 MW.  West demand 1000 MW in every step, above what the plant can
give (penalty_up 900 €/MWh, far above its fuel cost), so the plant runs at the most it can.  Flows are in units of
unitsize (125 MW).  Hand-derived oracles:

* linear online, min_load 0.9, availability 0.75 at t0001–t0024 and 1 after: 4 · 0.75 = 3 units online at full
  output (flow 3 units = 375 MW = 0.75 of capacity), 4 units (500 MW) from t0025.  Before: each unit online was
  capped at 0.75 of its capacity, below its min_load 0.9 · online: no unit could be online, the flow was 0.
* integer online, availability 0.75: 3 units online, flow 3; availability 0.7: 4 · 0.7 = 2.8, so 2 whole units,
  flow 2 (250 MW).
* no online variables, availability 0.75: flow ≤ 0.75 · 4 = 3 units (unchanged).
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


def _model(startup: str, availability, min_load: float | None = 0.9) -> list:
    vals = [["unit", PLANT, "virtual_unitsize", b64(125.0, "float")], ["unit", PLANT, "existing", b64(4.0, "float")],
            ["unit", PLANT, "startup_method", b64(startup, "str")],
            ["node", "west", "inflow", b64(-1000.0, "float")]]
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


def test_linear_units_online_run_at_full_output_in_an_outage(tmp_path_factory) -> None:
    av = {t: (0.75 if t <= "t0024" else 1.0) for t in STEPS}
    step = _solve(tmp_path_factory, "av_linear", _model("linear", av))
    assert unitsize(step, PLANT) == pytest.approx(125.0)
    online, flow = _by_t(step, "v_online_linear"), _by_t(step, "v_flow")
    for t in STEPS:
        expected = 3.0 if t <= "t0024" else 4.0
        assert online[t] == pytest.approx(expected, abs=1e-6), t
        assert flow[t] == pytest.approx(expected, abs=1e-6), t
    # The LP: maxOnline holds availability · existing_count as its bound; maxFlow_online none.
    pb = build(step.flex_data)
    bounds = row_bounds(pb, "maxOnline_linear", f"{PLANT},")
    assert {r.rsplit(",", 1)[1][:-1]: ub for r, (_lb, ub) in bounds.items()} == pytest.approx(
        {t: 4.0 * av[t] for t in STEPS})
    for row, terms in row_terms(pb, "maxFlow_online_linear", f"{PLANT},").items():
        assert {c: v for c, v in terms.items() if c.startswith("v_online")} == {
            f"v_online_linear[{PLANT},p2020,{row.rsplit(',', 1)[1][:-1]}]": -1.0}, row


@pytest.mark.parametrize("availability, units", [(0.75, 3.0), (0.7, 2.0)])
def test_integer_units_online_are_whole_available_units(tmp_path_factory, availability, units) -> None:
    step = _solve(tmp_path_factory, f"av_integer_{availability}", _model("integer", availability))
    online, flow = _by_t(step, "v_online_integer"), _by_t(step, "v_flow")
    for t in STEPS:
        assert online[t] == pytest.approx(units, abs=1e-6), t
        assert flow[t] == pytest.approx(units, abs=1e-6), t


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
    for fam in (f"maxOnline_{startup}", f"maxFlow_online_{startup}", f"minFlow_minload_{startup}"):
        assert row_terms(pb1, fam) == row_terms(pb0, fam), fam
        assert row_bounds(pb1, fam) == row_bounds(pb0, fam), fam
    assert with_one.solution.obj == pytest.approx(without.solution.obj, rel=1e-12)
