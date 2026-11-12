"""Construction lead time on the run calendar.

The start of operation of an investment decision (decision year + ``construction_lead_time``) is rounded to a
period start of the whole run, or to the end of the model horizon, by ``construction_lead_time_method``.  A
decision whose capacity would start to operate after the end of its solve or of the model horizon is not
offered, with a warning.

Built from ``tests/fixtures/tests.json`` (coal_plant, periods p2020 … p2035 five years apart, each representing
5 years: the run calendar puts them at 0, 5, 10 and 15 years and the model horizon ends at 20) and
``stoch_two_period_invest.json`` (peaker, p2035 and p2040 of one year each: horizon end 2).

* single solve ``y2020_2035_5week``: p2020 … p2035, all realized, investments in every period;
* chain ``y2020_5week`` (p2020 + p2025, realizes p2020), ``y2025_5week`` (p2025 + p2030, realizes p2025),
  ``y2030_5week`` (p2030 + p2035, realizes p2030), ``y2035_5week`` (p2035).
"""
from __future__ import annotations

import polars as pl
import pytest

from flextool.engine.input._solve_state import CommissioningLagError
from flextool.engine.preprocess.params._existing import commissioning_year_lf
from tests.engine._unit_db import b64, build, make_db, run, utf8
from tests.engine.test_user_constraint_capacity_terms import (
    CHAIN,
    PLANT,
    SINGLE,
    STOCH_CHAIN,
    STOCH_JSON,
    _add_values,
)

LEAD = "construction_lead_time"


def _lead(value, cls: str = "unit", entity: str = PLANT) -> list:
    return [cls, entity, LEAD, value]


def _method(method: str) -> list:
    return ["unit", PLANT, "construction_lead_time_method", b64(method, "str")]


def _period_map(data: dict[str, float]) -> list:
    return b64({"index_type": "str", "index_name": "period", "data": [[k, v] for k, v in data.items()]}, "map")


def _run(tmp_path_factory, alt: str, values: list, chain: list, *, fixture_json=None) -> dict:
    if fixture_json is None:
        url, scen = make_db(tmp_path_factory, alt=alt, values=values, base_chain=chain)
    else:
        # The parameter is newer than the fixture's database version: add it after the upgrade.
        url, scen = make_db(tmp_path_factory, alt=alt, values=[], base_chain=chain, fixture_json=fixture_json)
        _add_values(url, alt, values)
    return run(tmp_path_factory, url, scen)


def _commission(step, entity: str = PLANT) -> dict[str, str]:
    """``{decision period: start of operation}`` of the offered decisions of ``entity`` with a lead time."""
    commission = getattr(step.flex_data, "ed_invest_commission", None)
    if commission is None:
        return {}
    rows = utf8(commission).filter(pl.col("e") == entity)
    return dict(zip(rows["d"].to_list(), rows["d_c"].to_list()))


def _offered(step, entity: str = PLANT) -> list[str]:
    fd = step.flex_data
    if fd.ed_invest_set is None:
        return []
    return sorted(utf8(fd.ed_invest_set).filter(pl.col("e") == entity)["d"].to_list())


def _invest_columns(step, entity: str = PLANT) -> list[str]:
    if "v_invest_p" not in step.solution._vars:
        return []
    return sorted(utf8(step.solution.value("v_invest_p")).filter(pl.col("p") == entity)["d"].to_list())


def _solve_of(steps: dict, name: str):
    return next(s for k, s in steps.items() if k.startswith(name))


# ---------------------------------------------------------------------------
# Rounding on the run calendar


@pytest.mark.parametrize(("lead", "method", "start"), [
    # p2020 (0) + 8 = 8: previous seam 5 (p2025), next seam 10 (p2030), closest |8-5| = 3 > |10-8| = 2 -> 10.
    (8.0, "closest_seam", "p2030"),
    (8.0, "previous_seam", "p2025"),
    (8.0, "next_seam", "p2030"),
    # p2020 + 18 = 18: previous seam 15 (p2035); closest |18-15| = 3 > |20-18| = 2 -> 20, the end of the horizon:
    # never operates, not offered.
    (18.0, "previous_seam", "p2035"),
    (18.0, "closest_seam", None),
])
def test_rounding_on_run_calendar_single_solve(tmp_path_factory, lead, method, start) -> None:
    values = [_lead(b64(lead, "float")), _method(method)]
    step = list(_run(tmp_path_factory, f"lt_round_{method[:4]}_{int(lead)}", values, SINGLE).values())[-1]
    if start is None:
        assert "p2020" not in _offered(step)
        assert "p2020" not in _invest_columns(step)
    else:
        assert _commission(step)["p2020"] == start
        assert "p2020" in _invest_columns(step)


def test_rounding_on_run_calendar_chain(tmp_path_factory, capsys) -> None:
    """L = 8.  Solve y2020_5week (p2020 + p2025): with previous_seam the p2020 decision starts to operate in
    p2025 (5), inside the solve, and is offered; its p2025 decision (5 + 8 = 13 -> 10, p2030) starts after the
    solve and is not.  With closest_seam the p2020 decision rounds to p2030 (10), after the solve: not offered,
    and the warning names the solve, its last period and the lead time."""
    prev = _run(tmp_path_factory, "lt_chain_prev", [_lead(b64(8.0, "float")), _method("previous_seam")], CHAIN)
    first = _solve_of(prev, "y2020_5week")
    assert _commission(first) == {"p2020": "p2025"}
    assert _offered(first) == ["p2020"]
    capsys.readouterr()
    closest = _run(tmp_path_factory, "lt_chain_close", [_lead(b64(8.0, "float")), _method("closest_seam")], CHAIN)
    out = capsys.readouterr().out
    first = _solve_of(closest, "y2020_5week")
    assert _offered(first) == []
    assert _invest_columns(first) == []
    lines = [ln for ln in out.splitlines()
             if "unit coal_plant: construction_lead_time" in ln and "after the end of solve y2020_5week " in ln]
    assert len(lines) == 1, out
    assert "construction_lead_time 8 years" in lines[0]
    assert "investments decided in p2020, p2025" in lines[0]
    assert "after the end of solve y2020_5week (p2025)" in lines[0]
    assert "construction_lead_time_method immediate" in lines[0]


def test_myopic_chain_shorter_than_lead_time(tmp_path_factory, capsys) -> None:
    """L = 10 with two-period solves: every decision of every solve starts to operate after its solve (or after
    the model horizon for the decisions from p2030 on: 10 + 10 = 20 is the end of the horizon), so no solve has a
    coal_plant investment variable, and each solve warns once."""
    capsys.readouterr()
    steps = _run(tmp_path_factory, "lt_myopic", [_lead(b64(10.0, "float"))], CHAIN)
    out = capsys.readouterr().out
    for name in ("y2020_5week", "y2025_5week", "y2030_5week", "y2035_5week"):
        step = _solve_of(steps, name)
        assert _invest_columns(step) == [], name
    for name in ("y2020_5week", "y2025_5week"):
        lines = [ln for ln in out.splitlines() if f"after the end of solve {name} " in ln and "coal_plant" in ln]
        assert len(lines) == 1, (name, out)
        assert "construction_lead_time 10 years" in lines[0]
    horizon = [ln for ln in out.splitlines() if "after the end of the model horizon" in ln and "coal_plant" in ln]
    assert horizon, out
    assert all("(the end of period p2035)" in ln for ln in horizon)


# ---------------------------------------------------------------------------
# Calendar disagreement


class _StubSource:
    def __init__(self, params):
        self._params = params

    def entities(self, entity_class: str) -> pl.DataFrame:
        if entity_class == "unit":
            return pl.DataFrame({"name": ["nuke"]})
        raise KeyError(entity_class)

    def parameter(self, entity_class: str, name: str) -> pl.DataFrame:
        try:
            return self._params[(entity_class, name)]
        except KeyError:
            raise KeyError((entity_class, name)) from None

    def parameter_default(self, entity_class: str, name: str):
        raise KeyError((entity_class, name))


def _stub(years: dict[str, float], lead: float) -> _StubSource:
    return _StubSource({
        ("solve", "years_from_start"): pl.DataFrame(
            {"name": ["b"] * len(years), "period": list(years), "value": list(years.values())}),
        ("unit", LEAD): pl.DataFrame({"name": ["nuke"], "value": [lead]}),
    })


def test_solves_disagree_on_a_period_year() -> None:
    """Solve a puts p2025 at 5 years (p2020 represents 5), solve b at 10 (its history gives p2020 10 years):
    with a lead time this is an error naming the period and both solves; without one it is not used."""
    from flextool.engine.preprocess.params._run_calendar import build_run_calendar, run_calendar_from_frame

    frame = build_run_calendar({"a": [("p2020", 5), ("p2025", 5)],
                                "b": [("p2020", 10), ("p2025", 5), ("p2030", 5)]}, ["a", "b"])
    rc = run_calendar_from_frame(frame)
    assert rc.conflicts == [("p2025", "a", 5.0, "b", 10.0)]
    anchor = pl.LazyFrame({"e": ["nuke"], "d": ["p2025"]})
    piu = ["p2025", "p2030"]
    years = {"p2020": 0.0, "p2025": 10.0, "p2030": 15.0}
    with pytest.raises(CommissioningLagError, match="'a' puts period 'p2025'.*'b'"):
        commissioning_year_lf(_stub(years, 5.0), "b", anchor, piu, run_calendar=rc)
    assert commissioning_year_lf(_stub(years, 0.0), "b", anchor, piu, run_calendar=rc) is None


def test_solve_years_off_the_run_calendar() -> None:
    """The solve's own years must be the run calendar's shifted by one constant: p2025 at 0 and p2030 at 3 do
    not match the run calendar's 5 and 10."""
    from flextool.engine.preprocess.params._run_calendar import build_run_calendar, run_calendar_from_frame

    rc = run_calendar_from_frame(build_run_calendar({"a": [("p2020", 5), ("p2025", 5), ("p2030", 5)]}, ["a"]))
    with pytest.raises(CommissioningLagError, match="'p2030'"):
        commissioning_year_lf(_stub({"p2025": 0.0, "p2030": 3.0}, 5.0), "b",
                              pl.LazyFrame({"e": ["nuke"], "d": ["p2025"]}), ["p2025", "p2030"], run_calendar=rc)


def test_solve_counting_from_its_own_start() -> None:
    """A solve that counts years from its own first period (p2025 at 0, p2030 at 5) is the run calendar shifted
    by 5: the p2025 decision with L = 5 starts to operate in p2030, at the solve's year 5."""
    from flextool.engine.preprocess.params._run_calendar import build_run_calendar, run_calendar_from_frame

    rc = run_calendar_from_frame(build_run_calendar({"a": [("p2020", 5), ("p2025", 5), ("p2030", 5)]}, ["a"]))
    out = commissioning_year_lf(_stub({"p2025": 0.0, "p2030": 5.0}, 5.0), "b",
                                pl.LazyFrame({"e": ["nuke"], "d": ["p2025"]}), ["p2025", "p2030"],
                                run_calendar=rc).collect()
    assert out.select("d", "d_c", "yr_c", "yr_c_run").rows() == [("p2025", "p2030", 5.0, 10.0)]


# ---------------------------------------------------------------------------
# A constant lead time and the equivalent period map give the same LP


def _same_lp(a, b) -> None:
    ma, mb = build(a.flex_data).canonicalise(), build(b.flex_data).canonicalise()
    assert list(ma.row_names) == list(mb.row_names)
    assert list(ma.col_names) == list(mb.col_names)
    for attr in ("col_ptr", "row_idx", "val", "row_lb", "row_ub", "col_obj", "col_lb", "col_ub"):
        assert list(getattr(ma, attr)) == list(getattr(mb, attr)), attr


ROLLING = ["init", "west", "coal"]
# y2020_2day_dispatch as a rolling solve: 24 h windows, 12 h jumps (no investments; the rolls must build alike).
ROLLING_VALUES = [["solve", "y2020_2day_dispatch", "solve_mode", b64("rolling_window", "str")],
                  ["solve", "y2020_2day_dispatch", "rolling_solve_horizon", b64(24.0, "float")],
                  ["solve", "y2020_2day_dispatch", "rolling_solve_jump", b64(12.0, "float")]]


@pytest.mark.parametrize("case", ["single", "chain", "rolling", "stochastic"])
def test_constant_equals_period_map(tmp_path_factory, case) -> None:
    if case == "stochastic":
        kw = {"fixture_json": STOCH_JSON}
        chain, cls, entity, periods = STOCH_CHAIN, "unit", "peaker", ["p2035", "p2040"]
        lead = 1.0
    else:
        kw = {}
        chain = {"single": SINGLE, "chain": CHAIN, "rolling": ROLLING}[case]
        cls, entity, periods, lead = "unit", PLANT, ["p2020", "p2025", "p2030", "p2035"], 5.0
    extra = ROLLING_VALUES if case == "rolling" else []
    if case == "rolling":
        periods = ["p2020"]
    const = _run(tmp_path_factory, f"lt_const_{case}", [_lead(b64(lead, "float"), cls, entity)] + extra, chain,
                 **kw)
    as_map = _run(tmp_path_factory, f"lt_map_{case}",
                  [_lead(_period_map({d: lead for d in periods}), cls, entity)] + extra, chain, **kw)
    assert list(const) == list(as_map)
    for name in const:
        _same_lp(const[name], as_map[name])
    if case in ("single", "stochastic"):
        step = list(const.values())[0]
        assert _commission(step, entity), "the lead time must move the start of operation"


# ---------------------------------------------------------------------------
# The capacity carried to later solves of a chain


def test_carry_counts_lifetime_from_start_of_operation(tmp_path_factory) -> None:
    """coal_plant (existing 500 MW, unit size 500 MW) with reinvest_choice, lifetime 10, L = 5 and exactly 100 MW
    invested over the chain (invest_min_total = invest_max_total = 100).  Solve y2020_5week offers only the p2020
    decision (it starts in p2025, inside the solve), so it builds 100 MW then.  The capacity operates over
    [5, 15): p2025 and p2030.  The existing 500 MW operate over [0, 10): p2020 and p2025.  So the later solves see
    p2025: 600, p2030: 100, p2035: 0.  Counting the lifetime from the decision year ([0, 10)) would drop the
    100 MW from p2030."""
    values = [_lead(b64(5.0, "float")),
              ["unit", PLANT, "lifetime_method", b64("reinvest_choice", "str")],
              ["unit", PLANT, "lifetime", b64(10.0, "float")],
              ["unit", PLANT, "invest_min_total", b64(100.0, "float")],
              ["unit", PLANT, "invest_max_total", b64(100.0, "float")]]
    steps = _run(tmp_path_factory, "lt_carry", values, CHAIN)
    first = _solve_of(steps, "y2020_5week")
    us = float(utf8(first.flex_data.p_unitsize.frame).filter(pl.col("p") == PLANT)["value"][0])
    inv = utf8(first.solution.value("v_invest_p")).filter(pl.col("p") == PLANT)
    assert dict(zip(inv["d"].to_list(), (inv["value"] * us).to_list())) == pytest.approx({"p2020": 100.0})

    def existing(name: str) -> dict[str, float]:
        frame = utf8(_solve_of(steps, name).flex_data.p_entity_all_existing.frame).filter(pl.col("e") == PLANT)
        return dict(zip(frame["d"].to_list(), frame["value"].to_list()))

    assert existing("y2025_5week") == pytest.approx({"p2025": 600.0, "p2030": 100.0})
    assert existing("y2030_5week") == pytest.approx({"p2030": 100.0, "p2035": 0.0})
