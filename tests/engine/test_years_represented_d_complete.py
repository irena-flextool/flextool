"""``p_years_represented_d`` covers every period of the solve's ``dt``, with the years it represents.

The years a period represents (``solve.years_represented``) scale the annualised CO2 and costs to horizon
totals (output ``model CO2``, ``years_represented__d``).  Two paths used to lose them:

* stochastic forecast branch periods (``p2035_high`` …) got 0 years, while the canonical
  ``solve_data/p_years_represented.csv`` gives them their anchor period's years;
* rolling and synthetic sub-solves left the Param ``None`` (read as 1 year per period).

Fixtures: ``stoch_two_period.json`` and ``tests.json`` plus an override alternative (built from JSON).
"""
from __future__ import annotations

from tests.engine._unit_db import FLEXTOOL_ROOT, b64, make_db, run, utf8

STOCH_JSON = FLEXTOOL_ROOT / "tests" / "fixtures" / "stoch_two_period.json"


def _years(periods: dict[str, float]) -> list:
    return b64({"index_type": "str", "index_name": "period",
                "data": [[d, y] for d, y in periods.items()]}, "map")


def _years_by_period(fd) -> dict[str, float]:
    assert fd.p_years_represented_d is not None
    frame = utf8(fd.p_years_represented_d.frame)
    assert frame["d"].is_unique().all()
    return dict(frame.select("d", "value").iter_rows())


def _dt_periods(fd) -> list[str]:
    return utf8(fd.dt.select("d"))["d"].unique(maintain_order=True).to_list()


def test_branch_periods_get_anchor_years(tmp_path_factory) -> None:
    """Two-period stochastic solve, 5 years per period: the anchors and every forecast branch get 5.0."""
    url, scen = make_db(tmp_path_factory, alt="yrd_stoch", base_chain=["init", "base", "stoch"],
                        fixture_json=STOCH_JSON,
                        values=[["solve", "stoch_2p", "years_represented", _years({"p2035": 5.0, "p2040": 5.0})]])
    steps = list(run(tmp_path_factory, url, scen).values())
    assert len(steps) == 1
    fd = steps[0].flex_data
    periods = _dt_periods(fd)
    assert {"p2035", "p2035_high", "p2035_low", "p2040", "p2040_high", "p2040_low"} <= set(periods)
    years = _years_by_period(fd)
    assert years == {d: 5.0 for d in periods}
    assert list(years) == periods


def test_rolling_sub_solves_get_parent_years(tmp_path_factory) -> None:
    """Rolling 2-day dispatch (24 h horizon, 12 h jump) with 5 years on p2020: every roll carries
    ``{p2020: 5.0}``."""
    url, scen = make_db(tmp_path_factory, alt="yrd_roll", base_chain=["init", "west", "wind"], values=[
        ["solve", "y2020_2day_dispatch", "solve_mode", b64("rolling_window", "str")],
        ["solve", "y2020_2day_dispatch", "rolling_solve_horizon", b64(24.0, "float")],
        ["solve", "y2020_2day_dispatch", "rolling_solve_jump", b64(12.0, "float")],
        ["solve", "y2020_2day_dispatch", "years_represented", _years({"p2020": 5.0})]])
    steps = list(run(tmp_path_factory, url, scen).values())
    assert len(steps) > 1
    for step in steps:
        assert _dt_periods(step.flex_data) == ["p2020"], step.solve_name
        assert _years_by_period(step.flex_data) == {"p2020": 5.0}, step.solve_name

