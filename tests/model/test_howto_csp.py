"""The CSP how-to examples (``docs/how_to/investment.md``, "How to model CSP with thermal storage").

Scenarios ``csp_*`` of ``tests/fixtures/tests.json`` (built from JSON, never a checked-in ``.sqlite``).  Chain
``init, west, coal, csp_system`` (+ alternatives): a constant 150 MW demand at *west*; *coal_plant* (50 CUR/MWh_e)
is the backup.  The field yield *csp_field_yield* is 1.0 for t0007..t0018 (clear day), 0.4 for t0031..t0042
(cloudy day) and 0 otherwise.  *csp_field* (no input) feeds the storage node *csp_heat*; *csp_turbine* (efficiency
0.4) turns its heat into power.  Hand values over the 48 h:

    csp_existing      field 500 MW_th, tank 2500 MWh_th, turbine 100 MW_e: potential 8400 MWh_th, used
                      3000 + 2500 + 2400 = 7900, curtailed 500; turbine 3160 MWh_e; coal 4040 MWh
    csp_fixed_design  invested, field = 5 x turbine and tank = 25 x turbine (solar multiple 2, 10 h at the
                      design efficiency 0.4): 100 / 500 / 2500, the dispatch totals of csp_existing,
                      objective 36.865 + 162.5 M x CRF(7 %, 25) = 50.809209 M CUR
    csp_optimised_design  no ties: the turbine (100 MW_e) and the tank (5000 MWh_th) reach their bounds; the cyclic
                      heat balance needs tank >= 9000 - 4.8 x field and 16.8 x field >= 12 000, and 4.8 MWh_th
                      of tank is cheaper than 1 MW_th of field, so field = (9000 - 5000) / 4.8 = 833.33 MW_th;
                      the turbine runs at 100 MW_e in every hour (4800 MWh_e from 12 000 MWh_th), coal 2400 MWh
    csp_gas_backup    the boiler's heat (12 / 0.9 / 0.4 = 33.3 CUR/MWh_e) is cheaper than coal: turbine 100 MW_e in
                      every hour, field 7900, boiler 4100 MWh_th, coal 2400 MWh

Heat is free and coal costs the same in every hour, so the timing and the storage state are not unique: only
totals and capacities are checked.  The objectives are checked in ``tests/scenarios.yaml``.
"""
from __future__ import annotations

import polars as pl
import pytest

from tests.engine._unit_db import flow, make_db, run, time_map, unitsize, utf8

pytestmark = pytest.mark.solver

FIXED_DESIGN_OBJECTIVE = 50.809209  # M CUR
BOUNDS = {"csp_turbine": 100.0, "csp_field": 1000.0, "csp_heat": 5000.0}
FIXED_CHAIN = ["init", "west", "coal", "csp_system", "csp_plant", "csp_invest_common", "csp_fixed_design"]
_STEPS: dict[str, object] = {}


@pytest.fixture(autouse=True)
def _strict_autoscale(monkeypatch):
    monkeypatch.setenv("FLEXTOOL_AUTOSCALE_STRICT", "1")


@pytest.fixture(scope="module", autouse=True)
def _free_steps():
    yield
    _STEPS.clear()


def _single(steps: dict):
    assert len(steps) == 1, list(steps)
    return next(iter(steps.values()))


def _step(tmp_path_factory, test_db_url: str, scenario: str):
    """The one solve of *scenario* (solved once per module)."""
    if scenario not in _STEPS:
        _STEPS[scenario] = _single(run(tmp_path_factory, test_db_url, scenario))
    return _STEPS[scenario]


def _mwh(step, unit: str, source: str, sink: str) -> float:
    """Energy of the flow over the 48 one-hour steps."""
    return float(flow(step, unit, source, sink).sum())


def _invested(step) -> dict[str, float]:
    """Invested capacity [MW or MWh] of the CSP entities."""
    out = {}
    for unit in ("csp_field", "csp_turbine"):
        rows = utf8(step.solution.value("v_invest_p")).filter(pl.col("p") == unit)
        out[unit] = float(rows["value"].sum()) * unitsize(step, unit)
    rows = utf8(step.solution.value("v_invest_n")).filter(pl.col("n") == "csp_heat")
    state_unitsize = float(utf8(step.flex_data.p_state_unitsize.frame).filter(pl.col("n") == "csp_heat")["value"][0])
    out["csp_heat"] = float(rows["value"].sum()) * state_unitsize
    return out


def _objective(step) -> float:
    """The solver objective in M CUR."""
    return step.solution.obj / 1e6


@pytest.mark.parametrize(
    ("scenario", "field", "turbine", "coal", "boiler"),
    [
        ("csp_existing", 7900.0, 3160.0, 4040.0, None),
        ("csp_fixed_design", 7900.0, 3160.0, 4040.0, None),
        ("csp_gas_backup", 7900.0, 4800.0, 2400.0, 4100.0),
    ],
)
def test_dispatch_totals(tmp_path_factory, test_db_url, scenario, field, turbine, coal, boiler):
    """The field delivers 7900 of its 8400 MWh_th potential (500 curtailed: the tank is full on the clear day), the
    turbine makes 0.4 x its heat, and coal makes the rest of the 7200 MWh demand."""
    step = _step(tmp_path_factory, test_db_url, scenario)
    used = _mwh(step, "csp_field", "csp_field", "csp_heat")
    assert used == pytest.approx(field, rel=1e-9)
    potential = 500.0 * (12 * 1.0 + 12 * 0.4)  # field capacity x yield over the 48 h
    assert potential - used == pytest.approx(500.0, rel=1e-9)
    assert _mwh(step, "csp_turbine", "csp_heat", "west") == pytest.approx(turbine, rel=1e-9)
    assert _mwh(step, "coal_plant", "coal_market", "west") == pytest.approx(coal, rel=1e-9)
    if boiler is not None:
        assert _mwh(step, "csp_backup_boiler", "natural_gas_market", "csp_heat") == pytest.approx(boiler, rel=1e-9)


def test_fixed_design_ratios(tmp_path_factory, test_db_url):
    """The ties hold field = 5 x turbine (solar multiple 2 / 0.4) and tank = 25 x turbine (10 h / 0.4); the turbine
    is built to its 100 MW_e bound."""
    step = _step(tmp_path_factory, test_db_url, "csp_fixed_design")
    cap = _invested(step)
    assert cap["csp_turbine"] == pytest.approx(100.0, rel=1e-9)
    assert cap["csp_field"] == pytest.approx(5.0 * cap["csp_turbine"], rel=1e-9)
    assert cap["csp_heat"] == pytest.approx(25.0 * cap["csp_turbine"], rel=1e-9)
    assert _objective(step) == pytest.approx(FIXED_DESIGN_OBJECTIVE, rel=1e-6)


def test_optimised_design_no_dearer_than_fixed(tmp_path_factory, test_db_url):
    """Without the ties the model may still choose the fixed design, so its optimum costs no more; every capacity
    stays within its investment bound."""
    fixed = _step(tmp_path_factory, test_db_url, "csp_fixed_design")
    optimised = _step(tmp_path_factory, test_db_url, "csp_optimised_design")
    assert _objective(optimised) <= _objective(fixed) + 1e-6
    for entity, cap in _invested(optimised).items():
        assert -1e-6 <= cap <= BOUNDS[entity] + 1e-6, (entity, cap)


def test_optimised_design_capacities(tmp_path_factory, test_db_url):
    """The turbine and the tank are built to their bounds and the tank bound sets the field: (9000 - 5000) / 4.8
    = 833.33 MW_th.  The turbine then runs at 100 MW_e in every hour and coal makes the other 2400 MWh."""
    step = _step(tmp_path_factory, test_db_url, "csp_optimised_design")
    cap = _invested(step)
    assert cap["csp_turbine"] == pytest.approx(100.0, rel=1e-6)
    assert cap["csp_field"] == pytest.approx(2500.0 / 3.0, rel=1e-6)
    assert cap["csp_heat"] == pytest.approx(5000.0, rel=1e-6)
    assert _mwh(step, "csp_field", "csp_field", "csp_heat") == pytest.approx(12000.0, rel=1e-6)
    assert _mwh(step, "csp_turbine", "csp_heat", "west") == pytest.approx(4800.0, rel=1e-6)
    assert _mwh(step, "coal_plant", "coal_market", "west") == pytest.approx(2400.0, rel=1e-6)


def test_turbine_efficiency_constant_or_map(tmp_path_factory, test_db_url):
    """A 72-step map of 0.4 as the turbine efficiency gives the same optimum as the constant 0.4 (the design
    efficiency in the ties stays a constant)."""
    url, scenario = make_db(
        tmp_path_factory, alt="csp_turbine_efficiency_map", base_chain=FIXED_CHAIN,
        values=[["unit", "csp_turbine", "efficiency", time_map({}, 0.4)]],
    )
    as_map = _single(run(tmp_path_factory, url, scenario))
    constant = _step(tmp_path_factory, test_db_url, "csp_fixed_design")
    assert _objective(as_map) == pytest.approx(_objective(constant), rel=1e-9)
    assert _invested(as_map) == pytest.approx(_invested(constant), rel=1e-9)
    assert _mwh(as_map, "csp_turbine", "csp_heat", "west") == pytest.approx(3160.0, rel=1e-9)
