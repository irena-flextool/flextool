"""The demand response how-to (``docs/how_to/demand_response.md``) does what each of its sections says.

``flextool/schemas/canonical_databases/howto_demand_response.json``: 48 one-hour steps, Friday t0001-t0024 and
Saturday t0025-t0048.  Node ``town`` (unserved demand 3000 CUR/MWh); ``base_plant`` 700 MW at 40 and ``peak_plant``
300 MW at 150 CUR/MWh, units without inputs.  Demand (MW): Friday 00-05 680, 06-07 800, 08-16 950, 17-18 1080,
19-20 1130, 21-23 780; Saturday 00-07 550, 08-16 820, 17-20 980, 21-23 650.  The price is 40 below 700 MW, 150
between 700 and 1000 MW and 3000 above, where demand goes unserved (80 MW at Friday 17-18, 130 MW at 19-20).

Hand values per scenario (each test docstring repeats its own):

    no_dr              unserved 80 / 80 / 130 / 130 MW at Friday 17-20, nothing else
    dr_price           dr_reduce (150 MW at 200) cuts 80 / 80 / 130 / 130 at Friday 17-20, 0 elsewhere (price <= 150)
    dr_blocks          dr_reduce_low (50 MW at 100) 50 in every hour above 700 MW; dr_reduce (now 50 MW at 200)
                       30 / 30 / 50 / 50 at Friday 17-20; dr_reduce_high (50 MW at 500) 30 at Friday 19-20
    dr_window          dr_reduce available Friday 08-18 only: 80 at 17-18, nothing at 19-20 (130 MW unserved)
    dr_shift           town_shift (500 MWh) and shift_link (150 MW, 60 CUR/MWh each way): 420 MWh delivered at
                       Friday 17-20 and taken back only in hours at price 40 (round trip 120 > 150 - 40)
    ev_smart_charging  ev_charger (60 MW, efficiency 0.9) only while plugged in (00-06, 18-23); 4 x 40 MWh of
                       driving a day, 320 / 0.9 = 355.56 MWh from town, all of it at price 40
"""
from __future__ import annotations

import contextlib
import io
import json

import polars as pl
import pytest

from flextool._resources import package_data_path
from flextool.update_flextool.initialize_database import initialize_database
from tests.engine._unit_db import flow, run, utf8

pytestmark = pytest.mark.solver

HOWTO_JSON = "schemas/canonical_databases/howto_demand_response.json"
T = [f"t{i:04d}" for i in range(1, 49)]


def hours(day: int, lo: int, hi: int) -> list[int]:
    """Indexes (0-47) of hours ``lo``..``hi`` of day 0 (Friday) or 1 (Saturday)."""
    return [day * 24 + h for h in range(lo, hi + 1)]


DEMAND = [0.0] * 48
for _day, _blocks in ((0, [(0, 5, 680), (6, 7, 800), (8, 16, 950), (17, 18, 1080), (19, 20, 1130), (21, 23, 780)]),
                      (1, [(0, 7, 550), (8, 16, 820), (17, 20, 980), (21, 23, 650)])):
    for _lo, _hi, _mw in _blocks:
        for _i in hours(_day, _lo, _hi):
            DEMAND[_i] = float(_mw)

EVENING_17_18 = hours(0, 17, 18)
EVENING_19_20 = hours(0, 19, 20)
CHEAP = [i for i in range(48) if DEMAND[i] < 700.0]          # price 40
ABOVE_700 = [i for i in range(48) if DEMAND[i] > 700.0]
PLUGGED = [i for d in (0, 1) for i in hours(d, 0, 6) + hours(d, 18, 23)]

_STEPS: dict[str, object] = {}


@pytest.fixture(autouse=True)
def _strict_autoscale(monkeypatch):
    monkeypatch.setenv("FLEXTOOL_AUTOSCALE_STRICT", "1")


@pytest.fixture(scope="module")
def db_url(tmp_path_factory) -> str:
    sqlite = tmp_path_factory.mktemp("howto_demand_response") / "demand_response.sqlite"
    with contextlib.redirect_stdout(io.StringIO()):
        initialize_database(str(package_data_path(HOWTO_JSON)), str(sqlite))
    yield f"sqlite:///{sqlite}"
    _STEPS.clear()


def _step(tmp_path_factory, db_url: str, scenario: str):
    """The one solve of ``scenario`` (solved once per module)."""
    if scenario not in _STEPS:
        steps = run(tmp_path_factory, db_url, scenario)
        assert len(steps) == 1, list(steps)
        _STEPS[scenario] = next(iter(steps.values()))
    return _STEPS[scenario]


def _mw(step, unit: str, source: str, sink: str) -> list[float]:
    values = flow(step, unit, source, sink).to_list()
    assert len(values) == 48
    return values


def _unserved(step) -> list[float]:
    """Upward slack of ``town`` per hour (0 where the solution has no row)."""
    frame = utf8(step.solution.value("vq_state_up")).filter(pl.col("n") == "town")
    by_t = dict(zip(frame["t"], frame["value"]))
    return [float(by_t.get(t, 0.0)) for t in T]


def _expect(values: list[float], expected: dict[int, float], what: str) -> None:
    """``values`` equals ``expected`` at its hours and 0 at every other hour."""
    want = [expected.get(i, 0.0) for i in range(48)]
    assert values == pytest.approx(want, abs=1e-6), what


def test_demand_profile_is_the_documented_one() -> None:
    data = json.loads(package_data_path(HOWTO_JSON).read_text(encoding="utf-8"))
    pv = next(p for p in data["parameter_values"] if p[:3] == ["node", "town", "inflow"])
    inflow = {t: v for t, v in json.loads(pv[3][0])["data"]}
    assert [-inflow[t] for t in T] == DEMAND


def test_no_dr_leaves_the_evening_peak_unserved(tmp_path_factory, db_url) -> None:
    """Demand above the 1000 MW of plants goes unserved: 80 MW at Friday 17-18 and 130 MW at 19-20."""
    step = _step(tmp_path_factory, db_url, "no_dr")
    _expect(_unserved(step), {**dict.fromkeys(EVENING_17_18, 80.0), **dict.fromkeys(EVENING_19_20, 130.0)},
            "unserved")


def test_price_sensitive_reduction_acts_only_above_its_price(tmp_path_factory, db_url) -> None:
    """dr_reduce (150 MW at 200) cuts exactly the unserved demand of no_dr; in the hours at price 40 or 150 it
    stays at 0."""
    step = _step(tmp_path_factory, db_url, "dr_price")
    cut = {**dict.fromkeys(EVENING_17_18, 80.0), **dict.fromkeys(EVENING_19_20, 130.0)}
    _expect(_mw(step, "dr_reduce", "dr_reduce", "town"), cut, "dr_reduce")
    _expect(_unserved(step), {}, "unserved")


def test_price_blocks_form_a_demand_curve(tmp_path_factory, db_url) -> None:
    """dr_reduce_low (at 100, below the peak plant's 150) cuts 50 MW in every hour above 700 MW; the middle block
    (dr_reduce, 50 MW at 200) 30 at Friday 17-18 and 50 at 19-20; dr_reduce_high (at 500) 30 at 19-20 only
    (700 + 300 + 50 + 50 + 30 = 1130).  No demand is unserved."""
    step = _step(tmp_path_factory, db_url, "dr_blocks")
    _expect(_mw(step, "dr_reduce_low", "dr_reduce_low", "town"), dict.fromkeys(ABOVE_700, 50.0), "low block")
    _expect(_mw(step, "dr_reduce", "dr_reduce", "town"),
            {**dict.fromkeys(EVENING_17_18, 30.0), **dict.fromkeys(EVENING_19_20, 50.0)}, "middle block")
    _expect(_mw(step, "dr_reduce_high", "dr_reduce_high", "town"), dict.fromkeys(EVENING_19_20, 30.0),
            "high block")
    _expect(_unserved(step), {}, "unserved")


def test_no_reduction_outside_the_availability_window(tmp_path_factory, db_url) -> None:
    """dr_reduce is available on Friday 08-18 only: it cuts 80 MW at 17-18, nothing at 19-20 although 130 MW goes
    unserved there, and nothing on Saturday."""
    step = _step(tmp_path_factory, db_url, "dr_window")
    _expect(_mw(step, "dr_reduce", "dr_reduce", "town"), dict.fromkeys(EVENING_17_18, 80.0), "dr_reduce")
    _expect(_unserved(step), dict.fromkeys(EVENING_19_20, 130.0), "unserved")


def test_load_shift_moves_the_evening_peak_to_cheap_hours(tmp_path_factory, db_url) -> None:
    """The shift storage delivers 80 / 80 / 130 / 130 MW at Friday 17-20 (no demand unserved) and takes the 420 MWh
    back only in hours at price 40, where the base plant has room (20 MW at Friday 00-05, 150 at Saturday 00-07, 50
    at Saturday 21-23).  It shifts nothing out of the hours at price 150: 150 - 40 = 110 is less than the 120 of a
    round trip.  The link is lossless and the storage cycles over the solve, so the net shift is 0."""
    step = _step(tmp_path_factory, db_url, "dr_shift")
    out = _mw(step, "shift_link", "town_shift", "town")
    back = _mw(step, "shift_link", "town", "town_shift")
    _expect(out, {**dict.fromkeys(EVENING_17_18, 80.0), **dict.fromkeys(EVENING_19_20, 130.0)}, "cut")
    _expect(_unserved(step), {}, "unserved")
    room = {i: min(700.0 - DEMAND[i], 150.0) for i in CHEAP}
    for i in range(48):
        assert -1e-6 <= back[i] <= room.get(i, 0.0) + 1e-6, (T[i], back[i])
    assert sum(back) == pytest.approx(420.0, abs=1e-6)
    assert sum(out) - sum(back) == pytest.approx(0.0, abs=1e-6)


def test_ev_charges_only_while_plugged_in_and_at_the_cheap_hours(tmp_path_factory, db_url) -> None:
    """ev_charger has no flow at 07-17 (availability 0); it charges 320 MWh into ev_battery (160 MWh of driving a
    day), 320 / 0.9 = 355.56 MWh from town, all of it in plugged-in hours at price 40 (none in the scarce Friday
    evening).  Charging at price 40 only is possible: Saturday night and Friday 00-05 give (150 + 120) x 0.9 = 243
    MWh before Friday's 160 MWh of driving."""
    step = _step(tmp_path_factory, db_url, "ev_smart_charging")
    into_battery = _mw(step, "ev_charger", "town", "ev_battery")
    allowed = set(PLUGGED) & set(CHEAP)
    for i in range(48):
        if i not in allowed:
            assert into_battery[i] == pytest.approx(0.0, abs=1e-6), T[i]
        else:
            assert into_battery[i] / 0.9 <= 700.0 - DEMAND[i] + 1e-6, T[i]
    assert sum(into_battery) == pytest.approx(320.0, abs=1e-6)
    assert sum(into_battery) / 0.9 == pytest.approx(355.5556, abs=1e-4)
    _expect(_unserved(step), {**dict.fromkeys(EVENING_17_18, 80.0), **dict.fromkeys(EVENING_19_20, 130.0)},
            "unserved")
