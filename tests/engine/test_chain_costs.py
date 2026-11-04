"""Reported costs of chained solves: discount factors, lumps and windows from the start of the run.

Fixture ``tests.json`` (built from JSON).  ``model.inflation_rate`` r = 0.04 (alternative
``invest_solveSequence_5weeks``), ``inflation_offset_operations`` 0.5 and ``inflation_offset_investment`` 0 (the
defaults), four periods p2020 … p2035 of 5 years each, so the run years are Y = 0, 5, 10, 15 and the horizon ends at
20.  Hand derivation of the factors (``_npv`` "Run-level fields of the reported costs")::

    F_ops(d) = Σ_{y=0..4} 1.04^-(Y_d + y + 0.5)        F_inv(d) = Σ_{y=0..4} 1.04^-(Y_d + y)

coal_plant: invest_cost 400 CUR/kW, discount_rate 0.05, lifetime 30 (longer than the run), no lifetime_method
(reinvest_automatic: the operating window of a decision is every run period from its start of operation on),
fixed_cost 100 CUR/kW/yr.  Annuity A = 400 · 1000 · 0.05 / (1 − 1.05^−30) CUR/MW/yr.  So, for a decision in d with
start of operation c(d)::

    lump_invest[d] = A · Σ_{d' ≥ c(d)} F_inv(d')        lump_fixed[d] = 100 · 1000 · Σ_{d' ≥ c(d)} F_ops(d')

The chain ``invest_solveSequence_5weeks`` (y2020 … y2035: each solve holds its own period and the next one) gets
these values in every solve, while each solve's own factors restart at its first period.
"""
from __future__ import annotations

import polars as pl
import pytest

from tests.engine._unit_db import b64, make_db, run, utf8
from tests.engine.test_construction_lead_time_calendar import ROLLING, ROLLING_VALUES
from tests.engine.test_user_constraint_capacity_terms import CHAIN, PLANT, SINGLE

PERIODS = ["p2020", "p2025", "p2030", "p2035"]
YEARS = {"p2020": 0.0, "p2025": 5.0, "p2030": 10.0, "p2035": 15.0}


def _f(d: str, offset: float) -> float:
    return sum(1.04 ** -(YEARS[d] + y + offset) for y in range(5))


F_OPS = {d: _f(d, 0.5) for d in PERIODS}
F_INV = {d: _f(d, 0.0) for d in PERIODS}
ANNUITY = 400.0 * 1000.0 * 0.05 / (1.0 - 1.05 ** -30)
FIXED = 100.0 * 1000.0


def _later(d: str) -> list[str]:
    return PERIODS[PERIODS.index(d):]


def _d(param) -> dict[str, float]:
    frame = utf8(param.frame)
    return dict(zip(frame["d"].to_list(), frame["value"].to_list()))


def _ed(param, entity: str = PLANT) -> dict[str, float]:
    frame = utf8(param.frame).filter(pl.col("e") == entity)
    return dict(zip(frame["d"].to_list(), frame["value"].to_list()))


def _window(frame: pl.DataFrame, anchor: str, entity: str = PLANT) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for a, d in utf8(frame).filter(pl.col("e") == entity).sort(anchor, "d").select(anchor, "d").iter_rows():
        out.setdefault(a, []).append(d)
    return out


def _steps(tmp_path_factory, alt: str, values: list, chain: list[str]) -> dict:
    url, scen = make_db(tmp_path_factory, alt=alt, values=values, base_chain=chain)
    return run(tmp_path_factory, url, scen)


def _assert_run_factors(fd) -> None:
    op, inv = _d(fd.p_inflation_op_run), _d(fd.p_inflation_inv_run)
    assert sorted(op) == PERIODS and sorted(inv) == PERIODS
    for d in PERIODS:
        assert op[d] == pytest.approx(F_OPS[d], rel=1e-12), d
        assert inv[d] == pytest.approx(F_INV[d], rel=1e-12), d


def test_chain_reports_from_the_start_of_the_run(tmp_path_factory) -> None:
    """Every solve of the chain gets the factors of all four run periods from the start of the run, and the
    lumps of its decisions over the operating window up to the end of the run (p2035), not only over its own two
    periods; the solve's own factors (the objective's) are unchanged and restart at its first period."""
    steps = _steps(tmp_path_factory, "chain_run_fields", [], CHAIN)
    assert list(steps) == ["y2020_5week", "y2025_5week", "y2030_5week", "y2035_5week"]
    for k, (name, step) in enumerate(steps.items()):
        fd = step.flex_data
        own = PERIODS[k:k + 2]
        _assert_run_factors(fd)
        # The objective's factors: discounted from the solve's own first period.
        assert _d(fd.p_inflation_op) == pytest.approx({d: F_OPS[PERIODS[i]] for i, d in enumerate(own)},
                                                       rel=1e-12), name
        assert _window(fd.edd_operating_window, "d_invest") == {d: _later(d) for d in own}, name
        assert _ed(fd.ed_annuity) == pytest.approx({d: ANNUITY for d in own}, rel=1e-12), name
        assert _ed(fd.ed_entity_annual_discounted_run) == pytest.approx(
            {d: ANNUITY * sum(F_INV[x] for x in _later(d)) for d in own}, rel=1e-12), name
        fixed = _ed(fd.ed_lifetime_fixed_cost_run)
        for d in own:
            assert fixed[d] == pytest.approx(FIXED * sum(F_OPS[x] for x in _later(d)), rel=1e-12), (name, d)
    # 100 MW decided in p2025: fixed cost 100 · 1e5 · (3.7315 + 3.0671 + 2.5209) = 93.19 M (the single solve's).
    assert 100.0 * _ed(steps["y2025_5week"].flex_data.ed_lifetime_fixed_cost_run)["p2025"] == pytest.approx(
        93.1948e6, rel=1e-5)


def test_lead_time_window_starts_at_start_of_operation(tmp_path_factory) -> None:
    """coal_plant with a 5-year construction lead time: the y2020 solve offers only the p2020 decision, which
    starts to operate in p2025.  Its window is p2025 … p2035 and its lumps equal those of a p2025 decision
    without a lead time (same operating years, same invest and fixed cost); the p2025 decision (which would start
    after the solve) is not a vintage of the window."""
    steps = _steps(tmp_path_factory, "chain_run_lead",
                   [["unit", PLANT, "construction_lead_time", b64(5.0, "float")]], CHAIN)
    fd = steps["y2020_5week"].flex_data
    _assert_run_factors(fd)
    assert _window(fd.edd_operating_window, "d_invest") == {"p2020": _later("p2025")}
    assert _ed(fd.ed_entity_annual_discounted_run)["p2020"] == pytest.approx(
        ANNUITY * sum(F_INV[x] for x in _later("p2025")), rel=1e-12)
    assert _ed(fd.ed_lifetime_fixed_cost_run)["p2020"] == pytest.approx(
        FIXED * sum(F_OPS[x] for x in _later("p2025")), rel=1e-12)


_RUN_FIELDS = ("ed_entity_annual_discounted", "ed_entity_annual_divest_discounted", "ed_lifetime_fixed_cost",
               "ed_lifetime_fixed_cost_divest")


def _assert_run_is_solve(fd) -> None:
    assert fd.p_inflation_op_run is fd.p_inflation_op
    for name in _RUN_FIELDS:
        assert getattr(fd, f"{name}_run") is getattr(fd, name), name


def test_single_solve_run_fields_are_its_own(tmp_path_factory) -> None:
    """One solve over the whole run: the run fields ARE the solve's fields (the reported costs do not change),
    and the investment factor and windows are those of the solve."""
    steps = _steps(tmp_path_factory, "single_run_fields", [], SINGLE)
    (fd,) = [s.flex_data for s in steps.values()]
    _assert_run_is_solve(fd)
    _assert_run_factors(fd)
    assert _window(fd.edd_operating_window, "d_invest") == {d: _later(d) for d in PERIODS}
    lump = _ed(fd.ed_entity_annual_discounted_run)
    assert lump == pytest.approx({d: ANNUITY * sum(F_INV[x] for x in _later(d)) for d in PERIODS}, rel=1e-12)


def _rolling(periods: dict[str, str]) -> list:
    def arr(items: list[str]) -> list:
        return b64({"type": "array", "value_type": "str", "data": items, "index_name": "period"}, "array")

    def pmap(data: dict) -> list:
        return b64({"index_type": "str", "index_name": "period", "data": [[k, v] for k, v in data.items()]}, "map")
    return ROLLING_VALUES + [
        ["model", "flexTool", "inflation_rate", b64(0.04, "float")],
        ["solve", "y2020_2day_dispatch", "period_timeset", pmap(periods)],
        ["solve", "y2020_2day_dispatch", "realized_periods", arr(list(periods))],
        ["solve", "y2020_2day_dispatch", "years_represented", pmap({d: 5.0 for d in periods})]]


def test_rolls_of_one_period_use_their_own_factors(tmp_path_factory) -> None:
    """A rolling solve over p2020 alone: every roll holds the whole run, so its run factors are its own."""
    steps = _steps(tmp_path_factory, "roll_one_period", _rolling({"p2020": "2day"}), ROLLING)
    assert len(steps) > 1
    for name, step in steps.items():
        fd = step.flex_data
        _assert_run_is_solve(fd)
        assert _d(fd.p_inflation_op_run) == pytest.approx({"p2020": F_OPS["p2020"]}, rel=1e-12), name
        assert _d(fd.p_inflation_inv_run) == pytest.approx({"p2020": F_INV["p2020"]}, rel=1e-12), name


def test_rolls_across_a_period_boundary_discount_from_the_run_start(tmp_path_factory) -> None:
    """A rolling solve over p2020 and p2025: every roll, also one that holds only p2025, gets the factors of both
    run periods from the start of the run; a roll's own factor (the rolling solve's, which starts the run) is the
    same for the periods it holds."""
    steps = _steps(tmp_path_factory, "roll_two_periods", _rolling({"p2020": "2day", "p2025": "2day"}), ROLLING)
    late = 0
    for name, step in steps.items():
        fd = step.flex_data
        run_op = _d(fd.p_inflation_op_run)
        assert run_op == pytest.approx({d: F_OPS[d] for d in ("p2020", "p2025")}, rel=1e-12), name
        assert _d(fd.p_inflation_inv_run) == pytest.approx({d: F_INV[d] for d in ("p2020", "p2025")},
                                                            rel=1e-12), name
        own = _d(fd.p_inflation_op)
        assert own == pytest.approx({d: run_op[d] for d in own}, rel=1e-12), name
        late += list(own) == ["p2025"]
    assert late > 0, "no roll holds p2025 alone"


def test_run_window_keeps_the_fan_lineage() -> None:
    """The run walk of a stochastic fan counts each calendar period once along a lineage: a decision in the
    branch period p2035_high operates in p2035_high and p2040_high, not in the other branch's p2040_low; a period
    of the run outside the solve (p2045) is in every window; the bounded window stops at the end of the
    lifetime (yr 0 + 10)."""
    from flextool.engine.preprocess.params._npv import _run_window

    yr = {"p2035_high": 0.0, "p2035_low": 0.0, "p2040_high": 5.0, "p2040_low": 5.0, "p2045": 10.0}
    in_use = ["p2035_high", "p2035_low", "p2040_high", "p2040_low"]
    lineage = pl.DataFrame({"d": ["p2035_high", "p2035_high", "p2035_low", "p2035_low"],
                            "d_other": ["p2035_high", "p2040_high", "p2035_low", "p2040_low"]})
    anchors = pl.DataFrame({"e": ["u", "u"], "d": ["p2035_high", "p2035_high"], "kind": ["unbounded", "bounded"],
                            "yr_avail": [0.0, 0.0], "life": [0.0, 10.0]})
    walk = _run_window(anchors, yr, lineage, in_use)
    got = {k: sorted(walk.filter(pl.col("kind") == k)["d_all"].to_list()) for k in ("unbounded", "bounded")}
    assert got == {"unbounded": ["p2035_high", "p2040_high", "p2045"], "bounded": ["p2035_high", "p2040_high"]}
