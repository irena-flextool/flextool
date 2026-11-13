"""A solve split by a 2-D period map is the same solve as the one written by hand.

The docs (``how_to/solve_sequences.md``) say that a 2-D map of ``invest_periods`` / ``realized_invest_periods`` /
``realized_periods`` "generates these solves, with the periods given".  Before, the split solves lost the
parameters their database solve had besides the period lists: ``new_stepduration`` (they ran on the aggregated
steps with hourly data), ``stochastic_branches`` and ``stochastic_invest_method`` (they ran deterministic),
``dispatch_realization_method`` (no realization solve), ``shared_*`` and a 1-D ``fix_storage_periods``.

Each case builds the split and the hand-written model in separate databases (``_solve_copy_models``) and compares
every LP of the chain (canonical, up to row / column order) and the solve list.  The objectives are the
independent hand calculations (oracles, from the docs):

* model A with ``new_stepduration`` 2 h: the merged step sums the demand (-200 MWh per 2-h step), so the
  objectives stay 43 840 121.29 EUR (100 MW invested at the 5-year annuity 401.212936 EUR/MW plus
  100 MW * 2 h * 10 EUR/MWh * 21 900) and 43 800 000 EUR.  Before the fix the split ran at half the demand
  (21 920 060.65 / 21 900 000).
* model A as a look-ahead ladder (``LOOK_AHEAD``) with a 5-year ``construction_lead_time``: inv_p1 can invest only
  in p1 (a p2 decision would come online after the model horizon), the capacity comes online in p2, so p1 is
  served by the penalty: 100 MW * 2 h * 10 000 EUR/MWh * 21 900 + 100 MW * 2 h * 10 EUR/MWh * 21 900 + 100 MW *
  401.212936 EUR/MW = 43 843 840 121.29 EUR; inv_p2 43 800 000 EUR.  Before the fix the split refused the lead
  time.
* model A with one row of the 2-D maps holding both periods (``ONE_ROW``): one split solve, the single solve
  over both periods: 100 MW * 802.425872 EUR/MW + 100 MW * 4 steps * 10 EUR/MWh * 21 900 = 87 680 242.59 EUR.
  Before the fix the split stopped with an IndexError.
* model B (forecasts high / low with weights 1 / 3 in both periods): 6 570 000 EUR and 7 227 000 EUR.
* ``separate_dispatch_noinvest.json`` (one period) split by a 2-D ``realized_periods``: 6 570 000 EUR and the
  realization solve's 4 380 000 EUR (the hand calculation of ``test_separate_dispatch_noinvest_capacity.py``).
* model A with ``inflation_rate`` 0.05: see ``test_years_represented_counts_from_the_split_solves_first_period``.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.engine import _solve_copy_models as scm

REPO = Path(__file__).resolve().parents[2]
SEPARATE_DISPATCH_JSON = REPO / "tests" / "fixtures" / "separate_dispatch_noinvest.json"


def _nsd(mode: str) -> dict:
    return scm.model_a(mode, solve_rows=[["new_stepduration", scm.pack(2.0, "float")]])


def _separate_dispatch(mode: str) -> dict:
    data = json.loads(SEPARATE_DISPATCH_JSON.read_text())
    if mode == "hand":
        return scm.rename_solve(data, "stoch_rd", "stoch_rd_p2035")
    lps = scm.lp_snapshot()
    data = lps._overlay(data, name="split", base="separate_dispatch_noinvest", values=[
        ["solve", "stoch_rd", "realized_periods", scm.pmap([("p2035", scm.nested([("p2035", "yes")]))])]])
    # The overlay adds the scenario "split"; run both variants under one scenario name.
    data["scenarios"] = [s for s in data["scenarios"] if s[0] == "split"]
    data["scenario_alternatives"] = [r for r in data["scenario_alternatives"] if r[0] == "split"]
    return data


#: A ladder whose first solve looks one period ahead: invest_periods {p1: [p1, p2], p2: [p2]}, realized one period.
LOOK_AHEAD = {"invest_periods": {"p1": ["p1", "p2"], "p2": ["p2"]},
              "realized_invest_periods": {"p1": ["p1"], "p2": ["p2"]},
              "realized_periods": {"p1": ["p1"], "p2": ["p2"]}}


def _lead_time(mode: str) -> dict:
    return scm.model_a_ladder(mode, LOOK_AHEAD, unit_rows=[["construction_lead_time", scm.pack(5.0, "float")]])


#: One row of the 2-D maps with both periods: one split solve inv_p1 over both periods (before: IndexError).
ONE_ROW = {"invest_periods": {"p1": ["p1", "p2"]}, "realized_invest_periods": {"p1": ["p1", "p2"]},
           "realized_periods": {"p1": ["p1", "p2"]}}

#: Model A's per-period ladder as written by model_a("split").
PER_PERIOD = {par: {p: [p] for p in scm.PERIODS} for par in scm.SPLIT_PARAMS}


def _scenario(data: dict) -> str:
    return data["scenarios"][0][0]


CASES = [
    pytest.param(lambda mode: scm.model_a(mode), {"inv_p1": 43_840_121.29, "inv_p2": 43_800_000.0},
                 id="invest-ladder"),
    pytest.param(_nsd, {"inv_p1": 43_840_121.29, "inv_p2": 43_800_000.0}, id="new_stepduration"),
    pytest.param(_lead_time, {"inv_p1": 43_843_840_121.29, "inv_p2": 43_800_000.0}, id="construction_lead_time"),
    pytest.param(lambda mode: scm.model_a_ladder(mode, ONE_ROW), {"inv_p1": 87_680_242.59}, id="one-row"),
    pytest.param(lambda mode: scm.model_b(mode), {"st_p1": 6_570_000.0, "st_p2": 7_227_000.0},
                 id="stochastic_branches"),
    pytest.param(_separate_dispatch,
                 {"stoch_rd_p2035": 6_570_000.0, "stoch_rd_p2035__realized_dispatch": 4_380_000.0},
                 id="separate_dispatch"),
]


@pytest.mark.parametrize("build, objectives", CASES)
def test_split_solve_builds_the_hand_written_lp(tmp_path_factory, build, objectives):
    split_data, hand_data = build("split"), build("hand")
    split, split_steps = scm.run_spec(split_data, tmp_path_factory.mktemp("split"), _scenario(split_data))
    hand, hand_steps = scm.run_spec(hand_data, tmp_path_factory.mktemp("hand"), _scenario(hand_data))
    assert list(split_steps) == list(hand_steps) == list(objectives)
    assert not scm.lp_differences(split, hand)
    for solve, obj in objectives.items():
        assert split_steps[solve].obj == pytest.approx(obj, rel=1e-9), solve


@pytest.mark.parametrize("outer_index", ["solve", "1"])
def test_outer_index_name_does_not_matter(tmp_path_factory, outer_index):
    """The outer level of a 2-D period map can have any index name (the docs: the first column tells the solve).
    Before, a name other than Spine's default ``x`` gave the split solves no investment (the reader tested for an
    ``x`` column).  The split solves are reader solves with period arrays now."""
    named, _ = scm.run_spec(scm.model_a_ladder("split", PER_PERIOD, outer_index=outer_index),
                            tmp_path_factory.mktemp("named"))
    default, _ = scm.run_spec(scm.model_a_ladder("split", PER_PERIOD), tmp_path_factory.mktemp("default"))
    assert not scm.lp_differences(named, default)


def _nested(rows: tuple[str, str]) -> dict:
    """Model A's per-period invest ladder (rows *rows* for p1 / p2) containing the dispatch solve ``disp``
    (``realized_periods`` [p1, p2]), as ``invest_5weeks`` contains its storage solve in the shipped example."""
    first, second = rows
    spec = scm.model_a_ladder("split", {par: {first: ["p1"], second: ["p2"]}
                                        for par in ("invest_periods", "realized_invest_periods")})
    spec["entities"].append(["solve", "disp", None])
    spec["parameter_values"] += [
        ["solve", "inv", "contains_solves", scm.array(["disp"]), "init"],
        ["solve", "disp", "solve_mode", scm.pack("single_solve", "str"), "init"],
        ["solve", "disp", "period_timeset", scm.pack(scm.pmap(list(scm.PERIODS.items())), "map"), "init"],
        ["solve", "disp", "realized_periods", scm.array(list(scm.PERIODS)), "init"],
        ["solve", "disp", "years_represented", scm.pack(scm.pmap([(p, 5.0) for p in scm.PERIODS]), "map"),
         "init"]]
    return spec


def test_a_solve_contained_in_a_split_solve_gets_the_period_whatever_the_row_name(tmp_path_factory):
    """The docs let the first column of a 2-D map name the split solve freely.  A solve contained in a split
    solve is renamed after the period the split solve realizes (``disp_p1`` / ``disp_p2``).  Before, the rename
    read the row name as the period: with rows ``first`` / ``second`` the contained solve was not renamed, ran over
    both periods' scope under one name, and the second run replaced the results of the first."""
    lps = scm.lp_snapshot()
    named, named_steps = scm.run_spec(_nested(("first", "second")), tmp_path_factory.mktemp("named"))
    periods, period_steps = scm.run_spec(_nested(("p1", "p2")), tmp_path_factory.mktemp("periods"))
    assert list(period_steps) == ["inv_p1", "disp_p1", "inv_p2", "disp_p2"]
    assert list(named_steps) == ["inv_first", "disp_p1", "inv_second", "disp_p2"]
    assert len(named) == len(periods) == 4
    for (_, pa, off_a), (_, pb, off_b) in zip(named, periods):
        assert not lps.canonical_diff(lps.parse_mps(pa), lps.parse_mps(pb), 1e-12, off_a, off_b)
    # Hand calculation: the dispatch solves see the invested 100 MW, so they pay only operations:
    # 100 MW * 2 h * 10 EUR/MWh * 21 900 = 43 800 000 EUR each.
    assert named_steps["disp_p1"].obj == pytest.approx(43_800_000.0, rel=1e-9)
    assert named_steps["disp_p2"].obj == pytest.approx(43_800_000.0, rel=1e-9)


def test_years_represented_counts_from_the_split_solves_first_period(tmp_path_factory):
    """Decision D1: a split solve counts ``years_represented`` from its own first period, so with
    ``inflation_rate`` 0.05 it is the hand-written solve that carries only its own period (with inflation 0 every
    discount base gives the same LP, so only this case pins the rule).  The coefficients are the independent hand
    calculation from the docs (``advanced_concepts.md``: investments discounted at the start of each year,
    operations at mid-year): annuity A = 1000 * 0.05 / (1 - 1.05**-20) = 80.2425872 EUR/MW/a; investment
    A * sum_{y=0..4} 1.05**-y = 364.778830 EUR/MW in both solves; operations 10 EUR/MWh * 4380 h *
    sum_{y=0..4} 1.05**-(y + 0.5) = 194 314.032 EUR per MW and step.  Counted from the run start, p2 would get
    285.813758 EUR/MW."""
    inflation = [["model", "flextool", "inflation_rate", scm.pack(0.05, "float")]]
    split, _ = scm.run_spec(scm.model_a("split", model_rows=inflation), tmp_path_factory.mktemp("split"))
    hand, _ = scm.run_spec(scm.model_a("hand", model_rows=inflation, years_own_periods=True),
                           tmp_path_factory.mktemp("hand"))
    assert not scm.lp_differences(split, hand)
    for (solve, mps, _offset), (period, step) in zip(split, (("p1", "t01"), ("p2", "t03"))):
        assert solve == f"inv_{period}"
        assert scm.objective_coefficient(mps, f"v_invest_p[gen,{period}]") == pytest.approx(364.778830, rel=1e-8)
        assert scm.objective_coefficient(mps, f"v_flow[gen,gen,elec,{period},{step}]") == pytest.approx(
            194_314.032, rel=1e-8)


# ---------------------------------------------------------------------------
# What a 2-D map splits, and what it refuses (SolveConfig level, no solve)
# ---------------------------------------------------------------------------


def _config(spec: dict, folder):
    from flextool.engine.input._solve_config import SolveConfig

    return SolveConfig.load_from_db_url(scm.build_db(scm.write_spec(spec, folder), folder), "base")


def _with_solves(spec: dict, solves: list[str], extra: list | None = None) -> dict:
    """*spec* with the solves *solves* in model.solves, extra solve entities and *extra* parameter rows."""
    for name in solves:
        if ["solve", name, None] not in spec["entities"]:
            spec["entities"].append(["solve", name, None])
    for row in spec["parameter_values"]:
        if (row[0], row[2]) == ("model", "solves"):
            row[3] = scm.array(solves)
    spec["parameter_values"] += extra or []
    return spec


def _dispatch_rows(name: str) -> list:
    return [["solve", name, "period_timeset", scm.pack(scm.pmap([("p1", "ts_a")]), "map"), "init"],
            ["solve", name, "realized_periods", scm.array(["p1"]), "init"]]


def test_split_solves_take_the_place_of_their_solve(tmp_path):
    """The split solves run where their solve was in model.solves, so a later solve still runs after them
    (before they were appended at the end)."""
    spec = _with_solves(scm.model_a("split"), ["inv", "after"], _dispatch_rows("after"))
    assert _config(spec, tmp_path).model_solve["flextool"] == ["inv_p1", "inv_p2", "after"]


def test_a_solve_the_model_does_not_run_is_not_split(tmp_path):
    """A 2-D map on a solve outside the model adds no solves (before its split solves were added to
    model.solves and ran)."""
    spec = _with_solves(scm.model_a("split"), ["after"], _dispatch_rows("after"))
    sc = _config(spec, tmp_path)
    assert sc.model_solve["flextool"] == ["after"]
    assert sc.map2d_duplicate_of == {}


def _raises(spec: dict, folder, *parts: str) -> None:
    from flextool.engine.input._solve_state import FlexToolConfigError

    with pytest.raises(FlexToolConfigError) as err:
        _config(spec, folder)
    for part in parts:
        assert part in str(err.value), str(err.value)


def test_a_contained_solve_cannot_be_split(tmp_path):
    spec = _with_solves(scm.model_a("split"), ["top"], _dispatch_rows("top") + [
        ["solve", "top", "contains_solves", scm.array(["inv"]), "init"]])
    _raises(spec, tmp_path, "solve 'inv'", "only contained in another solve")


def test_a_split_solve_name_that_is_a_solve_is_an_error(tmp_path):
    spec = scm.model_a("split")
    spec["entities"].append(["solve", "inv_p2", None])
    _raises(spec, tmp_path, "solve 'inv'", "['inv_p2']", "has a solve of that name already")


def test_two_split_solves_realizing_one_period_is_an_error(tmp_path):
    """A 1-D realized_periods [p1, p2] next to an overlapping invest ladder: inv_p1 (p1, p2) and inv_p2 (p2) would
    both realize p2 (decision D3: an error, not last-wins)."""
    spec = scm.model_a_ladder("split", {"invest_periods": LOOK_AHEAD["invest_periods"],
                                        "realized_invest_periods": LOOK_AHEAD["realized_invest_periods"]},
                              lists={"realized_periods": ["p1", "p2"]})
    _raises(spec, tmp_path, "'inv_p1' and 'inv_p2' both realize period 'p2' (realized_periods)",
            "realized_periods of 'inv' is a 1-D list", "'inv_p1' realizes p1, p2; 'inv_p2' realizes p2",
            "Make realized_periods a 2-D map", "for example {p1: [p1], p2: [p2]}")


def test_two_rows_of_a_2d_map_realizing_one_period_is_an_error(tmp_path):
    """The same overlap written in the 2-D map itself: the message names the two rows."""
    spec = scm.model_a_ladder("split", {**LOOK_AHEAD, "realized_periods": {"p1": ["p1", "p2"], "p2": ["p2"]}})
    _raises(spec, tmp_path, "'inv_p1' and 'inv_p2' both realize period 'p2' (realized_periods)",
            "rows 'p1' and 'p2' both list 'p2': remove 'p2' from one of the two rows")


def test_a_2d_period_timeset_is_an_error(tmp_path):
    """Decision D4: period_timeset takes one timeset per period (before a 2-D one read the map's diagonal)."""
    spec = scm.model_a("single")
    for row in spec["parameter_values"]:
        if row[:3] == ["solve", "inv", "period_timeset"]:
            row[3] = scm.pack(scm.pmap([(p, scm.nested([(p, ts)])) for p, ts in scm.PERIODS.items()]), "map")
    _raises(spec, tmp_path, "solve 'inv': period_timeset is a 2-D map")


def test_forecasts_of_no_period_of_a_split_solve_stop_the_run(tmp_path_factory):
    """Decision D2: forecasts declared at p1 only; the split solve st_p2 has the solve's stochastic_branches but
    none of their periods, so it stops the run as the hand-written st_p2 does, and the message names the 2-D map
    entry and its solve."""
    from flextool.engine.input._solve_state import FlexToolConfigError

    for mode in ("split", "hand"):
        with pytest.raises(FlexToolConfigError) as err:
            scm.run_spec(scm.model_b(mode, fan_periods=("p1",)), tmp_path_factory.mktemp(mode))
        assert "none reveal at a reachable step" in str(err.value) and "'st_p2'" in str(err.value)
        advice = ("the 2-D period map row 'p2' of solve 'st'", "start in the period(s) p1",
                  "'st_p2' has only the period(s) p2", "at its first time step: p2/t03",
                  "leave stochastic_branches out of the solve for p2")
        assert [part in str(err.value) for part in advice] == [mode == "split"] * len(advice), str(err.value)
