"""The investment annuity of ``reinvest_choice`` / ``no_reinvestment`` capacity is paid for ``payback_time`` years
from its start of operation, in exact years: the period the payback time ends in pays only its years before the end.

Fixture ``tests.json`` (built from JSON), single solve ``y2020_2035_5week``: periods p2020 … p2035 of 5 years each,
run years 0, 5, 10, 15, horizon 20.  coal_plant: invest_cost 400 CUR/kW, discount_rate 0.05, ``invest_no_limit``,
fixed_cost 0.  The model sets ``inflation_rate`` 0.05 and ``inflation_offset_investment`` 1 (both explicitly: the
defaults are 0 and 0), so the investment factor of year k (0-based, from the run start) is 1.05^-(k+1).

Hand derivation.  The annuity over n years is A(n) = 400 · 1000 · 0.05 / (1 − 1.05^-n) and Σ_{j=1..n} 1.05^-j =
(1 − 1.05^-n) / 0.05, so n whole payments from the run start are worth exactly 400 000 and n payments from year 5
on 400 000 · 1.05^-5.  With A12 = 45 130.164 and S(a, b) = Σ_{j=a..b} 1.05^-j (year k paid ↔ j = k + 1):

====================  ==================  ============================  =========================================
decision              years paid          ed_entity_annual_discounted   before (every period starting before end)
====================  ==================  ============================  =========================================
p2020, payback 12     0 … 11              A12 · S(1, 12) = 400 000      A12 · S(1, 15) = 468 435.67
p2025, payback 12     5 … 16              400 000 · 1.05^-5             A12 · S(6, 20) = 367 031.60
p2030, payback 12     10 … 19 (horizon)   A12 · S(11, 20)               unchanged
p2035, payback 12     15 … 19             A12 · S(16, 20)               unchanged
====================  ==================  ============================  =========================================
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import polars as pl
import pytest

from tests.engine._unit_db import PLANT, RunCache, b64, utf8
from tests.engine.test_chain_costs import MEGA, _no_battery_invest, _results
from tests.engine.test_user_constraint_capacity_terms import CHAIN, SINGLE

PERIODS = ["p2020", "p2025", "p2030", "p2035"]
R = 1.05


def _s(a: int, b: int) -> float:
    """Σ_{j=a..b} 1.05^-j: the investment factors of years a−1 … b−1."""
    return sum(R ** -j for j in range(a, b + 1))


def _annuity(n: float) -> float:
    return 400.0 * 1000.0 * 0.05 / (1.0 - R ** -n)


A12 = _annuity(12.0)

_RUNS = RunCache()


@pytest.fixture(scope="module", autouse=True)
def _free_runs():
    yield
    _RUNS.clear()


def _period_map(values: dict[str, float]) -> list:
    return b64({"index_type": "str", "index_name": "period", "data": [[k, v] for k, v in values.items()]}, "map")


def _economics(method: str = "reinvest_choice", technical: float = 12.0, payback=12.0,
               invest_method: str = "invest_no_limit") -> list:
    payback_value = _period_map(payback) if isinstance(payback, dict) else b64(float(payback), "float")
    return [["model", "flexTool", "inflation_rate", b64(0.05, "float")],
            ["model", "flexTool", "inflation_offset_investment", b64(1.0, "float")],
            ["unit", PLANT, "invest_cost", b64(400.0, "float")],
            ["unit", PLANT, "discount_rate", b64(0.05, "float")],
            ["unit", PLANT, "fixed_cost", b64(0.0, "float")],
            ["unit", PLANT, "invest_method", b64(invest_method, "str")],
            ["unit", PLANT, "lifetime_method", b64(method, "str")],
            ["unit", PLANT, "lifetime_technical", b64(technical, "float")],
            ["unit", PLANT, "payback_time", payback_value]]


def _step(tmp_path_factory, alt: str, values: list, chain: list[str] | None = None):
    steps, _, _ = _RUNS.get(tmp_path_factory, alt=alt, values=values,
                            base_chain=_no_battery_invest(SINGLE) if chain is None else chain)
    return list(steps.values())[-1]


def _ed(param, entity: str = PLANT) -> dict[str, float]:
    frame = utf8(param.frame).filter(pl.col("e") == entity)
    return dict(zip(frame["d"].to_list(), frame["value"].to_list()))


EXPECTED_12 = {"p2020": 400_000.0, "p2025": 400_000.0 * R ** -5, "p2030": A12 * _s(11, 20),
               "p2035": A12 * _s(16, 20)}


@pytest.mark.solver
@pytest.mark.parametrize("method", ["reinvest_choice", "no_reinvestment"])
def test_annuity_paid_over_payback_time(tmp_path_factory, method) -> None:
    """Payback time = technical lifetime = 12 (table above).  The p2020 decision pays 12 years and so exactly its
    present value 400 000 (before: three whole periods, 15 years, 468 435.67); the p2025 decision 400 000 ·
    1.05^-5 (before 367 031.60); the later ones are cut by the horizon.  no_reinvestment allows investment only
    within the first technical lifetime (decisions p2020, p2025, p2030), at the same values."""
    got = _ed(_step(tmp_path_factory, f"pb_{method}", _economics(method)).flex_data.ed_entity_annual_discounted)
    periods = PERIODS if method == "reinvest_choice" else PERIODS[:3]
    for d in periods:
        assert got[d] == pytest.approx(EXPECTED_12[d], rel=1e-12), d


@pytest.mark.solver
def test_reinvest_automatic_pays_to_the_horizon(tmp_path_factory) -> None:
    """reinvest_automatic keeps paying the annuity to the end of the horizon: the p2020 decision pays years 0 … 19,
    A12 · S(1, 20) = 562 421.60."""
    got = _ed(_step(tmp_path_factory, "pb_automatic", _economics("reinvest_automatic")).flex_data
              .ed_entity_annual_discounted)
    assert got["p2020"] == pytest.approx(A12 * _s(1, 20), rel=1e-12)
    assert got["p2020"] == pytest.approx(562_421.5967, rel=1e-9)


@pytest.mark.solver
@pytest.mark.parametrize("technical", [30.0, 5.0], ids=["technical_30", "technical_5"])
def test_payback_window_is_not_the_technical_lifetime(tmp_path_factory, technical) -> None:
    """payback_time 12 with lifetime_technical 30 or 5: the annuity is paid over the 12 payback years whatever the
    technical lifetime (before: over the technical window, four whole periods with 30 and only p2020 with 5)."""
    got = _ed(_step(tmp_path_factory, f"pb_technical_{technical:g}", _economics(technical=technical)).flex_data
              .ed_entity_annual_discounted)
    for d in PERIODS:
        assert got[d] == pytest.approx(EXPECTED_12[d], rel=1e-12), d


@pytest.mark.solver
def test_payback_time_of_the_decision_period(tmp_path_factory) -> None:
    """A payback_time map {p2020: 12, p2025: 7, p2030: 12, p2035: 12}: the p2025 decision pays years 5 … 11,
    A(7) · S(6, 12) = 400 000 · 1.05^-5 (seven whole payments from year 5)."""
    payback = {"p2020": 12.0, "p2025": 7.0, "p2030": 12.0, "p2035": 12.0}
    got = _ed(_step(tmp_path_factory, "pb_map", _economics(payback=payback)).flex_data.ed_entity_annual_discounted)
    assert got["p2025"] == pytest.approx(_annuity(7.0) * _s(6, 12), rel=1e-12)
    assert got["p2025"] == pytest.approx(400_000.0 * R ** -5, rel=1e-12)
    assert got["p2020"] == pytest.approx(400_000.0, rel=1e-12)


@pytest.mark.solver
def test_fractional_payback_pays_part_of_a_year(tmp_path_factory) -> None:
    """payback_time 12.5: the p2020 decision pays years 0 … 11 and half of year 12,
    A(12.5) · (S(1, 12) + 0.5 · 1.05^-13) = 399 858.33."""
    got = _ed(_step(tmp_path_factory, "pb_half", _economics(payback=12.5)).flex_data.ed_entity_annual_discounted)
    want = _annuity(12.5) * (_s(1, 12) + 0.5 * R ** -13)
    assert got["p2020"] == pytest.approx(want, rel=1e-12)
    assert want == pytest.approx(399_858.33, rel=1e-8)


@pytest.mark.solver
def test_fractional_period_pays_part_of_its_last_year(tmp_path_factory) -> None:
    """p2020 represents 2.5 years (rows of width 1, 1, 0.5 at years 0, 1, 2; the last discounted
    0.5 · 1.05^-(2 + 0.5)) and payback_time = lifetime_technical = 2.25: the p2020 decision pays years 0, 1 and half
    of the half-year row, A(2.25) · (1.05^-1 + 1.05^-2 + 0.25 · 1.05^-2.5) = 400 262.77 (before: the whole period,
    A(2.25) · (1.05^-1 + 1.05^-2 + 0.5 · 1.05^-2.5) = 442 832.59)."""
    years = _period_map({"p2020": 2.5, "p2025": 5.0, "p2030": 5.0, "p2035": 5.0})
    got = _ed(_step(tmp_path_factory, "pb_frac_period", _economics(technical=2.25, payback=2.25) + [
        ["solve", "y2020_2035_5week", "years_represented", years]]).flex_data.ed_entity_annual_discounted)
    want = _annuity(2.25) * (R ** -1 + R ** -2 + 0.25 * R ** -2.5)
    assert got["p2020"] == pytest.approx(want, rel=1e-12)
    assert want == pytest.approx(400_262.77, rel=1e-8)


@pytest.mark.solver
def test_lead_time_starts_the_payback_window_at_start_of_operation(tmp_path_factory) -> None:
    """construction_lead_time 5: the p2020 decision starts to operate in p2025 and pays years 5 … 16, the value of a
    p2025 decision without a lead time, 400 000 · 1.05^-5."""
    got = _ed(_step(tmp_path_factory, "pb_lead", _economics() + [
        ["unit", PLANT, "construction_lead_time", b64(5.0, "float")]]).flex_data.ed_entity_annual_discounted)
    assert got["p2020"] == pytest.approx(400_000.0 * R ** -5, rel=1e-12)


def _seed(step, name: str) -> dict[str, float]:
    from flextool.engine.preprocess.params._derived import _provider_has_key, _provider_read

    path = Path("solve_data") / f"{name}.csv"
    provider = step.flex_data_provider
    assert _provider_has_key(provider, path), name
    frame = _provider_read(provider, path)
    frame = frame.filter(pl.col(frame.columns[0]) == PLANT)
    return {str(d): float(v) for d, v in zip(frame[frame.columns[1]].to_list(), frame[frame.columns[2]].to_list())}


SEED_CASES = {
    "payback_12": _economics(),
    "payback_map": _economics(payback={"p2020": 12.0, "p2025": 7.0, "p2030": 12.0, "p2035": 12.0}),
    "payback_12_5": _economics(payback=12.5),
    # A construction lead time with a fixed cost: both lumps start at the start of operation.
    "lead_time_fixed_cost": [v for v in _economics() if v[2] != "fixed_cost"] + [
        ["unit", PLANT, "fixed_cost", b64(100.0, "float")],
        ["unit", PLANT, "construction_lead_time", b64(5.0, "float")]],
}


@pytest.mark.solver
@pytest.mark.parametrize("case", list(SEED_CASES))
def test_solve_frames_of_the_lumps_are_the_objective_coefficients(tmp_path_factory, case) -> None:
    """The solve's ``solve_data/ed_entity_annual_discounted`` and ``solve_data/ed_lifetime_fixed_cost`` frames are
    the coefficients the objective uses.  Before, the CSV generation derived them a second time, anchored at the
    decision period: with a 5-year lead time the p2020 decision's frames summed the window from p2020 and the
    annuity of p2020, not from its start of operation p2025."""
    step = _step(tmp_path_factory, f"pb_seed_{case}", SEED_CASES[case])
    for name in ("ed_entity_annual_discounted", "ed_lifetime_fixed_cost"):
        want = _ed(getattr(step.flex_data, name))
        got = _seed(step, name)
        assert set(want) <= set(got), name
        for d, v in want.items():
            assert got[d] == pytest.approx(v, rel=1e-12, abs=1e-9), (name, d)


PIN_100_IN_P2020 = {"p2020": 100.0, "p2025": 0.0, "p2030": 0.0, "p2035": 0.0}


def _pinned(technical: float) -> list:
    return _economics(technical=technical, invest_method="invest_period") + [
        ["unit", PLANT, "invest_min_period", _period_map(PIN_100_IN_P2020)],
        ["unit", PLANT, "invest_max_period", _period_map(PIN_100_IN_P2020)]]


def _yearly_investment(results: dict[str, pd.DataFrame]) -> dict[str, float]:
    unit = results["cost_unit_annualized__d"]
    out = {}
    for d in PERIODS:
        rows = unit[(unit["period"] == d) & (unit["unit"] == PLANT)]
        out[d] = float(rows["investment"].sum()) * MEGA if len(rows) else 0.0
    return out


@pytest.mark.solver
def test_yearly_investment_cost_over_the_payback_years(tmp_path_factory) -> None:
    """100 MW decided in p2020 (pinned), payback_time 12 and lifetime_technical 5: the capacity operates in p2020 only,
    but the annuity is paid for 12 years.  The annualized investment cost is 100 · A12 a year in p2020 and p2025 and
    2/5 of that in p2030 (years 10 and 11 of its five), 0 in p2035, so the undiscounted yearly amounts add up to the
    12 years paid: 5 + 5 + 2.  Before: only the operating periods paid (p2020)."""
    results = _results(tmp_path_factory, "pb_yearly_single", _pinned(5.0), _no_battery_invest(SINGLE))
    got = _yearly_investment(results)
    want = {"p2020": 100.0 * A12, "p2025": 100.0 * A12, "p2030": 100.0 * A12 * 2.0 / 5.0, "p2035": 0.0}
    for d in PERIODS:
        assert got[d] == pytest.approx(want[d], rel=1e-7, abs=1e-6), d


@pytest.mark.solver
def test_chain_pays_the_payback_years_of_the_run(tmp_path_factory) -> None:
    """The chain ``invest_solveSequence_5weeks`` (each solve holds its own period and the next) with 100 MW decided in
    p2020, payback_time = lifetime_technical = 12.  The y2020 solve (p2020, p2025) prices the decision over its own
    years 0 … 9: A12 · S(1, 10) = 348 483.16 per MW.  The reported lump covers the payback years of the run, 0 … 11:
    A12 · S(1, 12) = 400 000 per MW (before: every run period that starts before year 12, 468 435.67).  The
    annualized investment cost is 100 · A12 in p2020 and p2025 and 100 · A12 · 2/5 = 1.805 M in p2030."""
    values = _pinned(12.0)
    steps, _, _ = _RUNS.get(tmp_path_factory, alt="pb_chain_steps", values=values,
                            base_chain=_no_battery_invest(CHAIN))
    fd = steps["y2020_5week"].flex_data
    assert _ed(fd.ed_entity_annual_discounted)["p2020"] == pytest.approx(A12 * _s(1, 10), rel=1e-12)
    assert A12 * _s(1, 10) == pytest.approx(348_483.1638, rel=1e-9)
    assert _ed(fd.ed_entity_annual_discounted_run)["p2020"] == pytest.approx(400_000.0, rel=1e-12)
    got = _yearly_investment(_results(tmp_path_factory, "pb_yearly_chain", values, _no_battery_invest(CHAIN)))
    want = {"p2020": 100.0 * A12, "p2025": 100.0 * A12, "p2030": 100.0 * A12 * 2.0 / 5.0, "p2035": 0.0}
    for d in PERIODS:
        assert got[d] == pytest.approx(want[d], rel=1e-7, abs=1e-6), d
    assert want["p2030"] == pytest.approx(1_805_206.56, rel=1e-8)


NOTICE_ID = "annuity_over_payback_time"


def _notice_lines(tmp_path_factory, capsys, alt: str, values: list) -> list[str]:
    from spinedb_api import DatabaseMapping

    from flextool.engine import run_chain_from_db
    from flextool.engine._behaviour_changes import BEHAVIOUR_CHANGES, MARKER_PREFIX
    from tests.engine._unit_db import make_db

    url, scen = make_db(tmp_path_factory, alt=alt, values=values, base_chain=_no_battery_invest(SINGLE))
    with DatabaseMapping(url) as db:
        db.add_item("metadata", name=MARKER_PREFIX + NOTICE_ID, value='{"since": 81, "entities": ["coal_plant"]}')
        db.commit_session("behaviour-change marker")
    capsys.readouterr()
    run_chain_from_db(url, scen, work_folder=tmp_path_factory.mktemp(f"{alt}_w"))
    summary = BEHAVIOUR_CHANGES[NOTICE_ID].summary
    return [ln for ln in capsys.readouterr().out.splitlines() if summary in ln]


@pytest.mark.solver
def test_notice_names_entities_whose_annuity_changes(tmp_path_factory, capsys) -> None:
    """With the upgrade's marker, coal_plant (payback time 12 ending inside p2030) is named once; with a payback
    time ending on a period boundary (10) nothing changes for it and no notice is printed."""
    lines = _notice_lines(tmp_path_factory, capsys, "pb_notice", _economics())
    assert len(lines) == 1 and "coal_plant" in lines[0]
    assert _notice_lines(tmp_path_factory, capsys, "pb_notice_seam", _economics(technical=10.0, payback=10.0)) == []


def test_notice_is_registered() -> None:
    from flextool.engine._behaviour_changes import BEHAVIOUR_CHANGES

    assert NOTICE_ID in BEHAVIOUR_CHANGES
