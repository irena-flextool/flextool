"""``solve.years_represented`` without an index name counts for investments.

A map left with Spine's default index name ``x`` gave investments one year per period while operations counted the
full years (the readers of the investment annuities, period start years and inflation factors tested for a
``period`` column).  The map is now read through ``_param_shapes.solve_years_represented``: the same LP whatever the
index name, in a single, a rolling and a stochastic solve (the CLAUDE.md constant-vs-map rule; the parameter takes
only a period map, so the two forms are the two index names).

The single-solve investment coefficients are the independent hand calculation (oracle, from the docs: annuity
1000 * 0.05 / (1 - 1.05**-20) = 80.2425872 EUR/MW/a, counted over the years the capacity operates in the horizon):
p1 = 10 a * 80.2425872 = 802.425872, p2 = 5 a = 401.212936 EUR/MW; objective 100 MW * 802.425872 + 100 MW * 4 steps
* 10 EUR/MWh * 21900 h = 87 680 242.59 EUR.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.engine import _solve_copy_models as scm

REPO = Path(__file__).resolve().parents[2]
TESTS_JSON = REPO / "tests" / "fixtures" / "tests.json"
STOCH_INVEST_JSON = REPO / "tests" / "fixtures" / "stoch_two_period_invest.json"


def _model_a(index: str | None) -> dict:
    return scm.model_a("single", years_index=index)


def _stochastic(index: str | None):
    """``stoch_two_period_invest.json`` / recourse (forecasts, recourse investment) with years_represented 5 / 3."""
    def patch(data: dict) -> dict:
        lps = scm.lp_snapshot()
        return lps._overlay(data, name="years_ix", base="recourse", values=[
            ["solve", "stoch_2p_inv", "years_represented",
             scm.pmap([("p2035", 5.0), ("p2040", 3.0)], **({} if index is None else {"index_name": index}))]])
    return patch(json.loads(STOCH_INVEST_JSON.read_text()))


def _rolling(index: str | None):
    """``tests.json`` / fullYear_roll (rolling dispatch over four periods) with years_represented 2 / 3 / 4 / 5 and
    rolls of 48 h every 36 h (two rolls per period)."""
    lps = scm.lp_snapshot()
    years = [("p2020", 2.0), ("p2025", 3.0), ("p2030", 4.0), ("p2035", 5.0)]
    return lps._overlay(json.loads(TESTS_JSON.read_text()), name="years_ix", base="fullYear_roll", values=[
        ["solve", "dispatch_fullYear_roll", "years_represented",
         scm.pmap(years, **({} if index is None else {"index_name": index}))],
        ["solve", "dispatch_fullYear_roll", "rolling_solve_horizon", 48.0],
        ["solve", "dispatch_fullYear_roll", "rolling_solve_jump", 36.0]])


def _run(tmp_path_factory, data: dict, scenario: str, stem: str):
    folder = tmp_path_factory.mktemp(stem)
    return scm.run_capturing(scm.build_db(scm.write_spec(data, folder), folder), scenario, folder)


@pytest.mark.parametrize("build, scenario", [
    pytest.param(_model_a, "base", id="single"),
    pytest.param(_rolling, "years_ix", id="rolling"),
    pytest.param(_stochastic, "years_ix", id="stochastic"),
])
def test_default_index_name_builds_the_period_index_lp(tmp_path_factory, build, scenario):
    unnamed, unnamed_steps = _run(tmp_path_factory, build(None), scenario, "x")
    named, named_steps = _run(tmp_path_factory, build("period"), scenario, "period")
    assert not scm.lp_differences(unnamed, named)
    assert [s.obj for s in unnamed_steps.values()] == pytest.approx([s.obj for s in named_steps.values()],
                                                                       rel=1e-9)


def test_single_solve_investment_counts_the_years_represented(tmp_path_factory):
    built, steps = _run(tmp_path_factory, _model_a(None), "base", "coef")
    (_solve, mps, _offset), = built
    assert scm.objective_coefficient(mps, "v_invest_p[gen,p1]") == pytest.approx(802.425872, rel=1e-8)
    assert scm.objective_coefficient(mps, "v_invest_p[gen,p2]") == pytest.approx(401.212936, rel=1e-8)
    assert steps["inv"].obj == pytest.approx(87_680_242.59, rel=1e-9)
