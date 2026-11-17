"""The cohort table (``preprocess/params/_cohorts.py``): end of life, periods in service, pre-built capacity and the
reach of a retirement, on hand-derived cases.

Model: ``tests.json``, the single solve ``y2020_2035_5week`` (scenario chain ``SINGLE``) over p2020 … p2035 of 5
years each.  Its years are 0, 5, 10, 15 and its horizon ends at 20 (no run calendar: the solve is the run), so the
seams are {0, 5, 10, 15, 20}.  coal_plant has 500 MW existing.  An end of life ``Y`` is rounded to the closest seam
(the default ``end_of_life_method`` closest_seam; a tie goes to the later seam); a cohort is in service in ``d`` when
``y_on <= y[d] < y_eol``.

* existing, reinvest_choice, lifetime 12: Y = 0 + 12 = 12, 12 − 10 = 2 < 15 − 12 = 3 → 10: in service p2020, p2025
  (y 0, 5 < 10), gone from p2030; existing capacity 500 / 500 / 0 / 0;
* lifetime 10: Y = 10 → 10: p2020, p2025; gone from p2030; 500 / 500 / 0 / 0;
* reinvest_automatic: renewed, no end of life (+inf), in service in every period, no end-of-life period;
* an investment decided in p2025, reinvest_choice, lifetime 10: on at 5, Y = 15 → 15: p2025, p2030; in service
  from the start of the period (pre-built) only in p2030 (y 10 > 5);
* decided in p2020 with a construction lead time of 5 years: starts at 0 + 5 = 5 (a seam: every method gives
  p2025), lifetime 10: Y = 15: p2025, p2030; pre-built in p2030;
* no lifetime_technical (reinvest_choice): Y = y_on, in service nowhere (existing and investment);
* a retirement decided in p2025 under lifetime 12 with end_of_life_method next_seam (Y = 12 → 15) counts in p2025
  and p2030 (p2035 is past the end of life).
"""
from __future__ import annotations

import math

import polars as pl
import pytest

from flextool.engine.preprocess.params._cohorts import (
    build_cohort_frames,
    existing_eol,
    existing_expired,
    forward_pairs,
    invest_alive,
    invest_prebuilt,
    retire_active,
)
from flextool.engine.preprocess.params._run_calendar import YEAR_TOL, snap_to_seam
from tests.engine._unit_db import PLANT, b64, make_db
from tests.engine.test_user_constraint_capacity_terms import SINGLE

SOLVE = "y2020_2035_5week"
PERIODS = ["p2020", "p2025", "p2030", "p2035"]
YEARS = {"p2020": 0.0, "p2025": 5.0, "p2030": 10.0, "p2035": 15.0}

_READERS: dict[str, object] = {}


#: The single-solve chain without coal_invest, which gives coal_plant a lifetime_technical of 30.
_WITHOUT_LIFETIME = [a for a in SINGLE if a != "coal_invest"]


def _reader(tmp_path_factory, alt: str, values: list):
    """A reader of ``tests.json`` + ``values`` on the single-solve chain (one DB per case, built once); the
    ``*_nolife`` cases leave out the alternative that sets coal_plant's lifetime."""
    from flextool.engine import SpineDbReader
    if alt not in _READERS:
        chain = _WITHOUT_LIFETIME if alt.endswith("_nolife") else SINGLE
        url, scen = make_db(tmp_path_factory, alt=alt, values=values, base_chain=chain)
        _READERS[alt] = SpineDbReader(url, scen)
    return _READERS[alt]


@pytest.fixture(scope="module", autouse=True)
def _free_readers():
    yield
    _READERS.clear()


def _lifetime(method: str, lifetime: float | None) -> list:
    values = [["unit", PLANT, "lifetime_method", b64(method, "str")]]
    if lifetime is not None:
        values.append(["unit", PLANT, "lifetime_technical", b64(lifetime, "float")])
    return values


def _frames(reader, ed_invest=None, commission_lf=None):
    return build_cohort_frames(reader, SOLVE, ed_invest, PERIODS, None, commission_lf=commission_lf)


def _rows(frame: pl.DataFrame, entity: str = PLANT) -> pl.DataFrame:
    return frame.filter(pl.col("e").cast(pl.Utf8) == entity)


def _alive(frames, kind: str) -> list[str]:
    rows = _rows(frames.cohort_alive).filter(pl.col("kind") == kind)
    return [str(d) for d in rows["d"].to_list()]


def _ed(pairs: list[tuple[str, str]]) -> pl.DataFrame:
    return pl.DataFrame({"e": [e for e, _ in pairs], "d": [d for _, d in pairs]})


# ---------------------------------------------------------------------------
# Rounding to the next seam
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("y_end,want", [
    (10.0, 10.0),            # on a seam: that seam
    (12.0, 15.0),            # between two seams: the next one
    (20.0, 20.0),            # the end of the horizon is a seam
    (21.0, math.inf),        # past the last seam: never ends inside the model
    (10.0 + 1e-12, 15.0),    # exact comparison: just past a seam is the next seam
])
def test_end_of_life_rounds_to_the_next_seam(y_end, want) -> None:
    assert snap_to_seam(y_end, "next_seam", [0.0, 5.0, 10.0, 15.0, 20.0], tol=0.0) == want


def test_rounding_tolerance_merges_a_year_just_past_a_seam() -> None:
    """With the year tolerance, 1e-12 past a seam is that seam (the construction lead time's rounding)."""
    assert snap_to_seam(10.0 + 1e-12, "next_seam", [0.0, 5.0, 10.0, 15.0, 20.0], tol=YEAR_TOL) == 10.0


# Periods of 5, 5 and 10 years: seams {0, 5, 10, 20}, the horizon ends at 20.  Each row: the end of life Y and its
# rounding by closest_seam / previous_seam / next_seam, derived by hand:
#   9  (lifetime 12, age 3): 9 − 5 = 4 > 10 − 9 = 1 → closest 10; previous 5; next 10
#   12 (lifetime 12):        12 − 10 = 2 < 20 − 12 = 8 → closest 10; previous 10; next 20
#   7.5:                     a tie (2.5, 2.5) goes to the later seam → closest 10; previous 5; next 10
#   16:                      16 − 10 = 6 > 20 − 16 = 4 → closest 20; previous 10; next 20
#   23 (lifetime 25, age 2): past the horizon end: previous / closest 20 (in service to the end), next +inf
#   −2 (lifetime 10, age 12): ended before the model start → the model start 0 under every method
#   3:                       3 − 0 = 3 > 5 − 3 = 2 → closest 5; previous 0; next 5
#   5, 5 ± 1e-12:            on a seam (within the year tolerance) → 5 under every method
_L_SEAMS = [0.0, 5.0, 10.0, 20.0]


@pytest.mark.parametrize("y_end,closest,previous,nxt", [
    (9.0, 10.0, 5.0, 10.0),
    (12.0, 10.0, 10.0, 20.0),
    (7.5, 10.0, 5.0, 10.0),
    (16.0, 20.0, 10.0, 20.0),
    (23.0, 20.0, 20.0, math.inf),
    (-2.0, 0.0, 0.0, 0.0),
    (3.0, 5.0, 0.0, 5.0),
    (5.0, 5.0, 5.0, 5.0),
    (5.0 + 1e-12, 5.0, 5.0, 5.0),
    (5.0 - 1e-12, 5.0, 5.0, 5.0),
    (math.inf, math.inf, math.inf, math.inf),
])
def test_end_of_life_rounding_by_method(y_end, closest, previous, nxt) -> None:
    from flextool.engine.preprocess.params._run_calendar import round_end_of_life
    for method, want in (("closest_seam", closest), ("previous_seam", previous), ("next_seam", nxt)):
        assert round_end_of_life(y_end, method, _L_SEAMS) == want, method


@pytest.mark.parametrize("y_end,method,alive", [
    (9.0, "closest_seam", ["p1", "p2"]), (9.0, "next_seam", ["p1", "p2"]), (9.0, "previous_seam", ["p1"]),
    (12.0, "closest_seam", ["p1", "p2"]), (12.0, "previous_seam", ["p1", "p2"]),
    (12.0, "next_seam", ["p1", "p2", "p3"]),
    (23.0, "closest_seam", ["p1", "p2", "p3"]), (23.0, "previous_seam", ["p1", "p2", "p3"]),
    (23.0, "next_seam", ["p1", "p2", "p3"]),
])
def test_alive_periods_and_end_of_life_period_by_method(y_end, method, alive) -> None:
    """In service in ``d`` while ``y[d] < y_eol`` (period years 0, 5, 10); the end-of-life period is the period that
    starts at ``y_eol`` (none at the horizon end or +inf): 10 → p3, 5 → p2, 20 / +inf → none."""
    from flextool.engine.preprocess.params._cohorts import cohort_calendar
    from flextool.engine.preprocess.params._run_calendar import RunCalendar, round_end_of_life
    years = {"p1": 0.0, "p2": 5.0, "p3": 10.0}
    rc = RunCalendar(years=dict(years), horizon_end=20.0, years_represented={"p1": 5.0, "p2": 5.0, "p3": 10.0})
    cal = cohort_calendar(rc, years)
    assert cal.seams == _L_SEAMS
    y_eol = cal.eol(y_end, method)
    assert y_eol == round_end_of_life(y_end, method, _L_SEAMS)
    assert [d for d, y in years.items() if y < y_eol] == alive
    assert cal.label(y_eol) == {10.0: "p3", 5.0: "p2"}.get(y_eol)


# ---------------------------------------------------------------------------
# Existing capacity
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("alt,method,life,y_eol,d_eol,alive,capacity", [
    ("coh_choice12", "reinvest_choice", 12.0, 10.0, "p2030", ["p2020", "p2025"], [500.0, 500.0, 0.0, 0.0]),
    ("coh_choice10", "reinvest_choice", 10.0, 10.0, "p2030", ["p2020", "p2025"], [500.0, 500.0, 0.0, 0.0]),
    ("coh_auto", "reinvest_automatic", 12.0, math.inf, None, PERIODS, [500.0, 500.0, 500.0, 500.0]),
    ("coh_nolife", "reinvest_choice", None, 0.0, "p2020", [], [0.0, 0.0, 0.0, 0.0]),
])
def test_existing_capacity_end_of_life(tmp_path_factory, alt, method, life, y_eol, d_eol, alive, capacity) -> None:
    from flextool.engine.preprocess.params._existing import p_entity_pre_existing_lf
    reader = _reader(tmp_path_factory, alt, _lifetime(method, life))
    frames = _frames(reader)
    row = _rows(frames.cohort).filter(pl.col("kind") == "existing")
    assert row.height == 1
    assert row["cohort"][0] == "existing:p2020"
    assert row["y_eol"][0] == y_eol
    assert row["d_eol"][0] == d_eol
    assert _alive(frames, "existing") == alive
    gone = {(PLANT, d) for d in PERIODS if d not in alive}
    assert {p for p in existing_expired(frames.cohort, YEARS, PERIODS) if p[0] == PLANT} == (
        gone if method != "reinvest_automatic" else set())
    eol = existing_eol(frames.cohort)
    if method == "reinvest_automatic":
        assert PLANT not in eol
    else:
        assert eol[PLANT] == (life or 0.0)
    pre = (p_entity_pre_existing_lf(reader, SOLVE, PERIODS)
           .filter(pl.col("e").cast(pl.Utf8) == PLANT).sort("d").collect())
    assert pre["value"].to_list() == capacity


# ---------------------------------------------------------------------------
# Investments
# ---------------------------------------------------------------------------


def test_investment_in_service_for_its_lifetime(tmp_path_factory) -> None:
    reader = _reader(tmp_path_factory, "coh_choice10", _lifetime("reinvest_choice", 10.0))
    frames = _frames(reader, _ed([(PLANT, "p2025")]))
    row = _rows(frames.cohort).filter(pl.col("kind") == "invest")
    assert row.select("cohort", "y_on", "y_end", "y_eol", "d_eol").rows() == [("invest:p2025", 5.0, 15.0, 15.0,
                                                                                "p2035")]
    assert _alive(frames, "invest") == ["p2025", "p2030"]
    prebuilt = invest_prebuilt(frames.cohort, frames.cohort_alive)
    assert [str(d) for d in _rows(prebuilt)["d"].to_list()] == ["p2030"]
    assert [(str(a), str(b)) for _, a, b in _rows(invest_alive(frames.cohort_alive)).rows()] == [
        ("p2025", "p2025"), ("p2025", "p2030")]


def test_investment_with_lead_time_starts_at_its_start_of_operation(tmp_path_factory) -> None:
    from flextool.engine.preprocess.params._existing import commissioning_year_lf
    values = _lifetime("reinvest_choice", 10.0) + [
        ["unit", PLANT, "construction_lead_time", b64(5.0, "float")]]
    reader = _reader(tmp_path_factory, "coh_lead5", values)
    ed = _ed([(PLANT, "p2020")])
    commission = commissioning_year_lf(reader, SOLVE, ed.lazy(), PERIODS)
    frames = _frames(reader, ed, commission)
    row = _rows(frames.cohort).filter(pl.col("kind") == "invest")
    assert row.select("d_build", "y_on", "y_eol").rows() == [("p2020", 5.0, 15.0)]
    assert _alive(frames, "invest") == ["p2025", "p2030"]
    prebuilt = invest_prebuilt(frames.cohort, frames.cohort_alive)
    assert [str(d) for d in _rows(prebuilt)["d"].to_list()] == ["p2030"]


def test_investment_without_lifetime_is_never_in_service(tmp_path_factory) -> None:
    reader = _reader(tmp_path_factory, "coh_nolife", _lifetime("reinvest_choice", None))
    frames = _frames(reader, _ed([(PLANT, "p2025")]))
    assert _alive(frames, "invest") == []
    assert _rows(invest_prebuilt(frames.cohort, frames.cohort_alive)).height == 0


# ---------------------------------------------------------------------------
# Retirements
# ---------------------------------------------------------------------------


def test_retirement_counts_until_the_end_of_life(tmp_path_factory) -> None:
    reader = _reader(tmp_path_factory, "coh_choice12_next", _lifetime("reinvest_choice", 12.0) + [
        ["unit", PLANT, "end_of_life_method", b64("next_seam", "str")]])
    frames = _frames(reader)
    active = retire_active(_ed([(PLANT, "p2025")]), frames.cohort_alive, frames.dd_forward)
    assert [(str(a), str(b)) for _, a, b in active.rows()] == [("p2025", "p2025"), ("p2025", "p2030")]


def test_retirement_reach_follows_the_scenario_lineage() -> None:
    """A fan of two branches (a, b) from p2025: a retirement decided in a branch counts in that branch only; one
    decided in p2020, before the fan, counts in both."""
    years = {"p2020": 0.0, "p2025_a": 5.0, "p2025_b": 5.0, "p2030_a": 10.0, "p2030_b": 10.0}
    periods = list(years)
    same = [("p2020", d) for d in periods] + [(d, "p2020") for d in periods]
    for branch in ("a", "b"):
        own = [f"p2025_{branch}", f"p2030_{branch}"]
        same += [(x, y) for x in own for y in own]
    lineage = pl.DataFrame({"d": [x for x, _ in same], "d_other": [y for _, y in same]})
    forward = forward_pairs(years, periods, lineage=lineage)
    got = sorted((str(a), str(b)) for a, b in forward.rows())
    assert got == sorted([("p2020", d) for d in periods]
                         + [("p2025_a", "p2025_a"), ("p2025_a", "p2030_a"), ("p2030_a", "p2030_a"),
                            ("p2025_b", "p2025_b"), ("p2025_b", "p2030_b"), ("p2030_b", "p2030_b")])


# ---------------------------------------------------------------------------
# A solve without years_represented and without a run calendar
# ---------------------------------------------------------------------------


def test_solve_calendar_without_years_represented_ends_one_year_after_the_last_start() -> None:
    """Each period represents one year (as the run calendar counts it): the horizon ends at the last start + 1."""
    from flextool.engine.preprocess.params._run_calendar import solve_calendar
    rc = solve_calendar({"p1": 0.0, "p2": 1.0, "p3": 2.0}, None)
    assert rc.horizon_end == 3.0 and rc.seams() == [0.0, 1.0, 2.0, 3.0]
    assert solve_calendar({"p1": 0.0, "p2": 5.0}, {"p1": 5.0}).seams() == [0.0, 5.0, 6.0]


@pytest.mark.parametrize("alt,method,life,alive", [
    # y2020_2day_dispatch: p2020 only, no years_represented: year 0, one year, the horizon ends at 1; seams {0, 1}.
    ("coh_1y_closest", "closest_seam", 0.6, ["p2020"]),     # 1 − 0.6 = 0.4 < 0.6 − 0 → 1: in service in p2020
    ("coh_1y_previous", "previous_seam", 1.5, ["p2020"]),   # past the horizon end → 1: in service to the end
    ("coh_1y_previous_short", "previous_seam", 0.6, []),    # 0.6 → 0: gone from the start
    ("coh_1y_next", "next_seam", 0.6, ["p2020"]),           # 0.6 → 1
])
def test_end_of_life_on_a_solve_without_years_represented(tmp_path_factory, alt, method, life, alive) -> None:
    """Without a run calendar and without years_represented the seams still end at the end of the horizon, so a
    previous_seam / closest_seam end of life after the last period start keeps that period."""
    from flextool.engine import SpineDbReader
    url, scen = make_db(tmp_path_factory, alt=alt, base_chain=["init", "west", "coal"], values=[
        ["unit", PLANT, "lifetime_method", b64("reinvest_choice", "str")],
        ["unit", PLANT, "lifetime_technical", b64(life, "float")],
        ["unit", PLANT, "end_of_life_method", b64(method, "str")]])
    reader = SpineDbReader(url, scen)
    frames = build_cohort_frames(reader, "y2020_2day_dispatch", None, ["p2020"], None)
    assert _alive(frames, "existing") == alive


@pytest.mark.parametrize("alt,life,alive,named", [
    # y2020_2day_dispatch (p2020 only, no years_represented): seams {0, 1}, the horizon ends at 1.
    ("coh_1y_default", 0.6, ["p2020"], False),       # 1 − 0.6 = 0.4 < 0.6 → 1, the horizon end: no period lost
    ("coh_1y_default_past", 1.5, ["p2020"], False),  # past the horizon end → 1 (next_seam: +inf): in service alike
    ("coh_1y_default_short", 0.4, [], True),         # 0.4 < 1 − 0.4 → 0 (next_seam: 1): p2020 is lost
])
def test_default_end_of_life_on_a_solve_without_years_represented(tmp_path_factory, capsys, alt, life, alive,
                                                                   named) -> None:
    """The default end_of_life_method (closest_seam) on a solve without a run calendar and without
    years_represented: an end of life after the last period start rounds to the end of the horizon, so that period
    stays in service and the upgrade's notice is silent; only an end of life that loses a period is named."""
    from flextool.engine import SpineDbReader
    from flextool.engine import _behaviour_changes as bc
    url, scen = make_db(tmp_path_factory, alt=alt, base_chain=["init", "west", "coal"], values=[
        ["unit", PLANT, "lifetime_method", b64("reinvest_choice", "str")],
        ["unit", PLANT, "lifetime_technical", b64(life, "float")]])
    reader = SpineDbReader(url, scen)
    bc.start_run({"end_of_life_closest_seam": '{"since": 81, "entities": ["coal_plant"]}'})
    try:
        capsys.readouterr()
        frames = build_cohort_frames(reader, "y2020_2day_dispatch", None, ["p2020"], None)
        printed = [ln for ln in capsys.readouterr().out.splitlines()
                   if bc.BEHAVIOUR_CHANGES["end_of_life_closest_seam"].summary in ln]
    finally:
        bc.start_run(None)
    assert _alive(frames, "existing") == alive
    if named:
        assert len(printed) == 1 and "coal_plant (existing): end of life at year 0.4 of the run, out of service " \
            "from p2020 (next_seam: in service to the end of the horizon)" in printed[0]
    else:
        assert printed == []
