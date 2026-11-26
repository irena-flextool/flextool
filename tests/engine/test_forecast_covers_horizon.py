"""A stochastic forecast must cover every time step that a solve or roll reads it for (item 307).

``docs/how_to/stochastics.md`` "Rolling horizon stochastics": every forecast branch has, for each
``analysis_time`` where a solve or roll branches out, a series that extends to the end of the horizon.  A step of a
branch period that the forecast made at the fan's ``analysis_time`` has no value for is an input error, raised
before the solve is built.  The reader never falls back to the default value (0 for a profile) or to the realized
branch's series.

Vehicles (databases built from JSON): ``tests/fixtures/stochastics.json``

* ``2_day_stochastic_dispatch``: one solve ``2day_dispatch``, period ``period1`` t0001-t0048, the fan at t0001 with
  the forecasts ``realized`` / ``upper`` / ``lower`` / ``mid`` (no ``realized: yes`` branch); ``wind_plant`` reads
  profile ``wind`` and is in group ``add_stochastics``.
* ``1_week_rolling_wind``: rolling ``within_solve``, jump 24, horizon 96, steps t0001-t0168.  Roll 4 starts at t0097,
  realizes t0097-t0120 with branch ``realized`` and branches out at t0121 into the forecasts made there; its horizon
  ends at t0168.  So the forecasts made at t0121 must cover t0121-t0168, and the realized series made at t0121 covers
  the jump of roll 5 (t0121-t0144).

Every override below replaces the whole ``wind`` profile with constant series: realized 0.5, upper 0.9, lower 0.2,
mid 0.4.
"""
from __future__ import annotations

import json
from pathlib import Path

import polars as pl
import pytest

from flextool.engine.input._solve_state import FlexToolConfigError
from tests.engine._unit_db import b64, make_db, utf8

STOCHASTICS_JSON = Path(__file__).resolve().parents[1] / "fixtures" / "stochastics.json"
VALUES = {"realized": 0.5, "upper": 0.9, "lower": 0.2, "mid": 0.4}
FORECASTS = ("upper", "lower", "mid")
ROLL_STARTS = list(range(1, 169, 24))


@pytest.fixture(autouse=True)
def _strict_autoscale(monkeypatch):
    monkeypatch.setenv("FLEXTOOL_AUTOSCALE_STRICT", "1")


def _t(i: int) -> str:
    return f"t{i:04d}"


def _chain(scenario: str) -> list[str]:
    rows = [r for r in json.loads(STOCHASTICS_JSON.read_text(encoding="utf-8"))["scenario_alternatives"]
            if r[0] == scenario]
    nxt = {a: b for _, a, b in rows}
    alt = next(a for a in nxt if a not in set(nxt.values()))
    chain = []
    while alt is not None:
        chain.append(alt)
        alt = nxt.get(alt)
    return chain


def _profile(series: dict[str, dict[str, list[int]]]) -> list:
    """``wind`` as branch -> analysis_time -> time -> the branch's constant (``series[branch][at]`` = step indices)."""
    def level(data: list) -> dict:
        return {"type": "map", "index_type": "str", "data": data}
    top = {"index_type": "str", "data": [
        [b, level([[at, level([[_t(i), VALUES[b]] for i in steps])] for at, steps in by_at.items()])]
        for b, by_at in series.items()]}
    return b64(top, "map")


def _run(tmp_path_factory, alt: str, scenario: str, series: dict) -> dict:
    from flextool.engine import run_chain_from_db

    url, scen = make_db(tmp_path_factory, alt=alt, values=[["profile", "wind", "profile", _profile(series)]],
                        base_chain=_chain(scenario), fixture_json=STOCHASTICS_JSON)
    return run_chain_from_db(url, scen, work_folder=tmp_path_factory.mktemp(f"{alt}_w"), keep_solutions=True)


def _profile_value(step, d: str, t: str) -> float:
    f = step.flex_data.p_profile_value
    f = getattr(f, "frame", f)
    if isinstance(f, pl.LazyFrame):
        f = f.collect()
    f = utf8(f).filter((pl.col("f") == "wind") & (pl.col("d") == d) & (pl.col("t") == t))
    assert f.height == 1, f"wind at ({d}, {t}): {f.height} rows"
    return float(f["value"][0])


# --- single solve: the fan at t0001 covers t0001-t0048 ------------------------------------------------------------

def _two_day(upper_last: int) -> dict:
    series = {b: {"t0001": list(range(1, 49))} for b in VALUES}
    series["upper"] = {"t0001": list(range(1, upper_last + 1))}
    return series


def test_single_solve_forecast_one_step_short_is_an_error(tmp_path_factory) -> None:
    """``upper`` made at t0001 stops at t0047; period1_upper reads t0048 (hand-written message)."""
    with pytest.raises(FlexToolConfigError) as err:
        _run(tmp_path_factory, "fc_2day_short", "2_day_stochastic_dispatch", _two_day(47))
    assert str(err.value) == (
        "profile 'wind', parameter 'profile': solve '2day_dispatch' (time steps 't0001' to 't0048') branches into "
        "its forecasts at time step 't0001' and reads the forecasts made then over its whole horizon, but forecast "
        "'upper' made at analysis_time 't0001' has no value at time step 't0048' (period 'period1_upper').  "
        "FlexTool does not fill a forecast with the default value or with the series of another branch.  Give every "
        "forecast made at analysis_time 't0001' values up to the end of the horizon ('t0048'), or shorten the "
        "horizon (in a rolling solve, rolling_solve_horizon).  Check which alternative of the scenario gives this "
        "value.")


def test_single_solve_forecast_to_the_last_step_is_read(tmp_path_factory) -> None:
    """``upper`` made at t0001 over t0001-t0048: the run completes and period1_upper reads 0.9 at t0048."""
    steps = _run(tmp_path_factory, "fc_2day_full", "2_day_stochastic_dispatch", _two_day(48))
    step = steps["2day_dispatch"]
    assert _profile_value(step, "period1_upper", "t0048") == 0.9
    assert _profile_value(step, "period1_lower", "t0048") == 0.2


# --- rolling: roll 4 branches out at t0121, its horizon ends at t0168 ---------------------------------------------

def _rolling(*, upper_t0121: list[int] | None = None) -> dict:
    """The realized series at each roll start over its jump; every forecast at each roll start a over a .. a + 95,
    cut at t0168; ``upper_t0121`` replaces the steps of ``upper`` made at t0121."""
    series = {"realized": {_t(a): list(range(a, a + 24)) for a in ROLL_STARTS}}
    for b in FORECASTS:
        series[b] = {_t(a): list(range(a, min(a + 95, 168) + 1)) for a in ROLL_STARTS}
    if upper_t0121 is not None:
        series["upper"]["t0121"] = upper_t0121
    return series


def test_roll_forecast_one_step_short_is_an_error(tmp_path_factory) -> None:
    """``upper`` made at t0121 stops at t0167; roll 4 reads period1_upper at t0168."""
    with pytest.raises(FlexToolConfigError) as err:
        _run(tmp_path_factory, "fc_roll_short", "1_week_rolling_wind", _rolling(upper_t0121=list(range(121, 168))))
    msg = str(err.value)
    assert msg.startswith("profile 'wind', parameter 'profile': solve '1week_rolling_roll_4' (time steps 't0097' "
                          "to 't0168') branches into its forecasts at time step 't0121'"), msg
    assert ("forecast 'upper' made at analysis_time 't0121' has no value at time step 't0168' "
            "(period 'period1_upper')") in msg
    assert "values up to the end of the horizon ('t0168'), or shorten the horizon" in msg


def test_roll_forecast_to_the_horizon_end_is_read(tmp_path_factory) -> None:
    """Every forecast covers its roll's horizon: the run completes; roll 4 reads upper 0.9 at t0168."""
    steps = _run(tmp_path_factory, "fc_roll_full", "1_week_rolling_wind", _rolling())
    assert _profile_value(steps["1week_rolling_roll_4"], "period1_upper", "t0168") == 0.9
    assert _profile_value(steps["1week_rolling_roll_4"], "period1_mid", "t0145") == 0.4


def test_forecast_gap_is_not_filled_with_the_realized_series(tmp_path_factory) -> None:
    """``upper`` made at t0121 misses t0130.  The realized series made at t0121 has it (0.5), but a forecast never
    reads the actual data: an error naming the step."""
    steps = [i for i in range(121, 169) if i != 130]
    with pytest.raises(FlexToolConfigError) as err:
        _run(tmp_path_factory, "fc_roll_gap", "1_week_rolling_wind", _rolling(upper_t0121=steps))
    msg = str(err.value)
    assert "solve '1week_rolling_roll_4'" in msg, msg
    assert ("forecast 'upper' made at analysis_time 't0121' has no value at time step 't0130' "
            "(period 'period1_upper')") in msg
