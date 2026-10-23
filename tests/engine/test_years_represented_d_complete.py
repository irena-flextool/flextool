"""``p_years_represented_d`` covers every period of the solve's ``dt``, with the years it represents.

The years a period represents (``solve.years_represented``) scale the annualised CO2 and costs to horizon
totals (output ``model CO2``, ``years_represented__d``).  Two paths used to lose them:

* stochastic forecast branch periods (``p2035_high`` …) got 0 years, while the canonical
  ``solve_data/p_years_represented.csv`` gives them their anchor period's years;
* rolling and synthetic sub-solves left the Param ``None`` (read as 1 year per period).

On those sub-solves the load-boundary block builds the Param only from the canonical CSV: without a provider
or the CSV it stays ``None`` (no silent one year per period); a header-only CSV gives one year per period.

Fixtures: ``stoch_two_period.json`` and ``tests.json`` plus an override alternative (built from JSON).
"""
from __future__ import annotations

import polars as pl

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



# ---------------------------------------------------------------------------
# The load-boundary block builds the Param only from the canonical CSV


class _Provider:
    """A minimal FlexDataProvider: frames keyed by their canonical provider key."""

    def __init__(self, frames: dict[str, pl.DataFrame]):
        from flextool.engine.preprocess._provider_io import _provider_key

        self._frames = {_provider_key(path): frame for path, frame in frames.items()}

    def has(self, key: str) -> bool:
        return key in self._frames

    def get(self, key: str) -> pl.DataFrame:
        return self._frames[key]


class _NoParams:
    """An input source without any parameter (``years_represented`` not authored)."""

    def parameter(self, entity_class: str, parameter_name: str) -> pl.DataFrame:
        raise KeyError(parameter_name)


YEARS_CSV = "solve_data/p_years_represented.csv"
DT = pl.DataFrame({"d": ["p2020", "p2020", "p2025"], "t": ["t0001", "t0002", "t0001"]})


def _boundary(provider) -> dict[str, float] | None:
    from flextool.engine.preprocess.flex_data import _years_represented_boundary

    param = _years_represented_boundary(DT, _NoParams(), provider, None)
    return None if param is None else dict(utf8(param.frame).select("d", "value").iter_rows())


def test_boundary_without_provider_or_csv_stays_none() -> None:
    """No provider, or a provider without the canonical CSV: the Param stays None instead of a silent
    one year per period."""
    assert _boundary(None) is None
    assert _boundary(_Provider({})) is None


def test_boundary_reads_the_canonical_csv() -> None:
    """With the canonical CSV, each period gets the sum of its year widths."""
    csv = pl.DataFrame({"period": ["p2020", "p2020", "p2025"], "years_from_solve": ["0", "1", "2"],
                        "p_years_represented": [1.0, 1.0, 0.5]})
    assert _boundary(_Provider({YEARS_CSV: csv})) == {"p2020": 2.0, "p2025": 0.5}


def test_boundary_header_only_csv_gives_one_year_per_period() -> None:
    """A header-only CSV (``years_represented`` not authored) keeps the default of one year per period,
    so a ``co2_max_total`` model without ``years_represented`` still builds."""
    header_only = pl.DataFrame(schema={"period": pl.Utf8, "years_from_solve": pl.Utf8,
                                       "p_years_represented": pl.Float64})
    assert _boundary(_Provider({YEARS_CSV: header_only})) == {"p2020": 1.0, "p2025": 1.0}


def test_rolling_co2_max_total_without_years_represented_builds(tmp_path_factory) -> None:
    """Rolling 2-day dispatch with a ``co2_max_total`` cap and no ``years_represented``: every roll gets
    one year for p2020 and builds its ``co2_max_total`` row."""
    from tests.engine._unit_db import build

    url, scen = make_db(tmp_path_factory, alt="yrd_roll_co2", base_chain=["init", "west", "coal", "co2_limit"],
                        values=[
        ["solve", "y2020_2day_dispatch", "solve_mode", b64("rolling_window", "str")],
        ["solve", "y2020_2day_dispatch", "rolling_solve_horizon", b64(24.0, "float")],
        ["solve", "y2020_2day_dispatch", "rolling_solve_jump", b64(12.0, "float")],
        ["group", "co2_limit", "co2_method", b64("total", "str")],
        ["group", "co2_limit", "co2_max_total", b64(1.0e6, "float")]])
    steps = list(run(tmp_path_factory, url, scen).values())
    assert len(steps) > 1
    for step in steps:
        assert _years_by_period(step.flex_data) == {"p2020": 1.0}, step.solve_name
        assert "co2_max_total" in set(build(step.flex_data).cstr_names()), step.solve_name
