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
* model B with look-ahead (st_p1 over p1 + p2, fan at p1; st_p2 fan at p2; 4380 h per step, gen 10 EUR/MWh,
  demand 100 MW, wind 100 MW at CF high 0.4 / low 0.2 in p1 and 0.1 in p2): per hour p1 0.25 * 600 + 0.75 * 800 =
  750 EUR, p2 0.25 * 600 + 0.75 * 900 = 825 EUR; st_p1 (2 * 750 + 2 * 825) * 4380 = 13 797 000 EUR, st_p2
  2 * 825 * 4380 = 7 227 000 EUR.  Before the fix st_p1 had both fans and stopped ("fans out only once").
* model A ladder with rows ``first`` / ``second``: 43 840 121.29 EUR and 43 800 000 EUR (the capacity of
  inv_first carried).  Before the fix inv_second lost it and re-invested.
* ``separate_dispatch_noinvest.json`` (one period) split by a 2-D ``realized_periods``: 6 570 000 EUR and the
  realization solve's 4 380 000 EUR (the hand calculation of ``test_separate_dispatch_noinvest_capacity.py``).
* model A's look-ahead ladder (``LOOK_AHEAD``) containing a dispatch solve ``disp`` over both periods (item
  243): inv_p1 (over p1 + p2) invests 100 MW in p1 and pays 100 MW * 80.2425872 EUR/MW/a * 10 years + 100 MW *
  4 steps * 10 EUR/MWh * 21 900 = 87 680 242.59 EUR (the ``one-row`` value); disp_p1, inv_p2 and disp_p2 run one
  period on the existing 100 MW: 43 800 000 EUR each.  disp_p1 = 43.8 M (not 87.6 M) pins that the contained
  solve runs only the period its containing solve realizes.  Before the fix the hand-written ladder ran
  ``disp`` (not ``disp_p1``) under inv_p1 and the examples' hand-written ladder stopped with a KeyError.
* model A with ``inflation_rate`` 0.05: see ``test_years_represented_counts_from_the_model_start`` and the tests
  after it (item 242: every solve discounts from the start of the run).
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


def _nested(rows: tuple[str, str]) -> dict:
    """Model A's per-period invest ladder (rows *rows* for p1 / p2) containing the dispatch solve ``disp``
    (``realized_periods`` [p1, p2]), as ``invest_5weeks`` contains its storage solve in the shipped example."""
    first, second = rows
    return scm.model_a_ladder("split", {par: {first: ["p1"], second: ["p2"]}
                                        for par in ("invest_periods", "realized_invest_periods")},
                              contains=["disp"])


def _nested_look_ahead(mode: str) -> dict:
    """The look-ahead ladder (``LOOK_AHEAD``) whose solves contain the dispatch solve ``disp`` over both periods:
    split, or the hand-written inv_p1 (over p1 + p2, realizing p1) and inv_p2 (item 243)."""
    return scm.model_a_ladder(mode, LOOK_AHEAD, contains=["disp"])


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
    pytest.param(lambda mode: scm.model_b(mode, look_ahead=True), {"st_p1": 13_797_000.0, "st_p2": 7_227_000.0},
                 id="stochastic_branches-look-ahead"),
    pytest.param(lambda mode: scm.model_a_ladder(mode, {par: {"first": ["p1"], "second": ["p2"]}
                                                        for par in scm.SPLIT_PARAMS}),
                 {"inv_first": 43_840_121.29, "inv_second": 43_800_000.0}, id="row-names"),
    pytest.param(_separate_dispatch,
                 {"stoch_rd_p2035": 6_570_000.0, "stoch_rd_p2035__realized_dispatch": 4_380_000.0},
                 id="separate_dispatch"),
    # Item 243: before, the hand-written inv_p1 stood for both periods it looks ahead to, so ``disp`` kept its
    # name under it (steps inv_p1, disp, inv_p2, disp_p2) and the hand-written examples' ladder stopped.
    pytest.param(_nested_look_ahead, {"inv_p1": 87_680_242.59, "disp_p1": 43_800_000.0, "inv_p2": 43_800_000.0,
                                      "disp_p2": 43_800_000.0}, id="nested-look-ahead"),
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


def test_row_names_give_the_lp_of_period_names(tmp_path_factory):
    """Rows ``first`` / ``second`` (item 256) and, with look-ahead forecasts, ``y1`` / ``y2`` (items 256 + 257, as
    in a year-by-year chain) build the LPs of the period-named rows."""
    named, named_steps = scm.run_spec(scm.model_a_ladder("split", {par: {"first": ["p1"], "second": ["p2"]}
                                                                   for par in scm.SPLIT_PARAMS}),
                                      tmp_path_factory.mktemp("named"))
    periods, _ = scm.run_spec(scm.model_a_ladder("split", PER_PERIOD), tmp_path_factory.mktemp("periods"))
    assert list(named_steps) == ["inv_first", "inv_second"]
    assert not scm.lp_differences(_renamed(named, {"inv_first": "inv_p1", "inv_second": "inv_p2"}), periods)

    years, year_steps = scm.run_spec(scm.model_b("split", look_ahead=True, rows=("y1", "y2")),
                                     tmp_path_factory.mktemp("years"))
    stoch, _ = scm.run_spec(scm.model_b("split", look_ahead=True), tmp_path_factory.mktemp("stoch"))
    assert list(year_steps) == ["st_y1", "st_y2"]
    assert not scm.lp_differences(_renamed(years, {"st_y1": "st_p1", "st_y2": "st_p2"}), stoch)
    assert year_steps["st_y1"].obj == pytest.approx(13_797_000.0, rel=1e-9)
    assert year_steps["st_y2"].obj == pytest.approx(7_227_000.0, rel=1e-9)


def test_year_rows_with_a_realized_branch_run_the_documented_chain(tmp_path_factory):
    """The one-solve form of the year-by-year recipe (stochastics.md "Investing year by year", step 5): rows y1 / y2
    of the 2-D ``realized_periods``, a fan per year with a ``realized: yes`` branch ``real``.  Each split solve
    fans out and is followed by its realization dispatch (review 256/257).  Hand values (dispatch only, 100 MW
    demand, 100 MW wind, gen 10 EUR/MWh, one year represented, no discounting): the stochastic solves as without
    the realized branch (st_y1: 0.25·60 + 0.75·80 = 75 MW of gen, ·10·8760 = 6 570 000; st_y2: 0.25·60 + 0.75·90
    = 82.5 MW, 7 227 000); the realizations read real = 0.3 in p1 (gen 70 MW, 6 132 000) and real = 0.5 in p2
    (gen 50 MW, 4 380 000)."""
    _, steps = scm.run_spec(scm.model_b("split", rows=("y1", "y2"), real_branch={"p1": 0.3, "p2": 0.5}),
                            tmp_path_factory.mktemp("recipe"))
    assert list(steps) == ["st_y1", "st_y1__realized_dispatch", "st_y2", "st_y2__realized_dispatch"]
    objectives = {name: step.obj for name, step in steps.items()}
    assert objectives == pytest.approx({"st_y1": 6_570_000.0, "st_y1__realized_dispatch": 6_132_000.0,
                                        "st_y2": 7_227_000.0, "st_y2__realized_dispatch": 4_380_000.0}, rel=1e-9)


def _renamed(built: list, names: dict[str, str]) -> list:
    return [(names.get(solve, solve), mps, offset) for solve, mps, offset in built]


def test_a_1d_fix_storage_periods_does_not_give_a_split_solve_the_next_fan(tmp_path_factory):
    """A 1-D ``fix_storage_periods`` [p1, p2] is kept over every period of each split solve, so it must not make
    st_p1 stand for its look-ahead period p2 too: st_p1 keeps the p1 fan only (else item 229's "fans out only
    once" stops the run).  The objectives are those of the look-ahead case (model B has no storage)."""
    spec = scm.model_b("split", look_ahead=True, solve_rows=[["fix_storage_periods", scm.array(list(scm.PERIODS))]])
    _, steps = scm.run_spec(spec, tmp_path_factory.mktemp("fix"))
    assert list(steps) == ["st_p1", "st_p2"]
    assert steps["st_p1"].obj == pytest.approx(13_797_000.0, rel=1e-9)
    assert steps["st_p2"].obj == pytest.approx(7_227_000.0, rel=1e-9)


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


#: ``inflation_rate`` 0.05 on the model.
INFLATION = [["model", "flextool", "inflation_rate", scm.pack(0.05, "float")]]

# Hand calculation (``advanced_concepts.md``: every solve discounts from the start of the run, investments at the
# start of each year, operations at mid-year; 4380 h per step and year; offsets inv 0 / ops 0.5):
# annuity A = 1000 * 0.05 / (1 - 1.05**-20) = 80.2425872 EUR/MW/a; F_inv(p1) = sum_{y=0..4} 1.05**-y = 4.54595050,
# F_inv(p2) = sum_{y=5..9} 1.05**-y = 3.56187117; F_ops(p1) = sum_{y=0..4} 1.05**-(y + 0.5) = 4.43639343,
# F_ops(p2) = sum_{y=5..9} 1.05**-(y + 0.5) = 3.47603034.
INV_P1 = 364.778830          # A * F_inv(p1)
INV_P2 = 285.813758          # A * F_inv(p2)
INV_P1_WINDOW = 650.592588   # A * (F_inv(p1) + F_inv(p2)): a p1 decision of a solve over p1 and p2
OPS_P1 = 194_314.032         # 10 EUR/MWh * 4380 h * F_ops(p1), per MW and step
OPS_P2 = 152_250.129         # 10 EUR/MWh * 4380 h * F_ops(p2)


def _coefficients(mps, wanted: dict[str, float]) -> None:
    for column, value in wanted.items():
        assert scm.objective_coefficient(mps, column) == pytest.approx(value, rel=1e-8), column


def _with_inflation(spec: dict) -> dict:
    spec["parameter_values"] += [[*row, "init"] for row in INFLATION]
    return spec


def test_years_represented_counts_from_the_model_start(tmp_path_factory):
    """Item 242 (owner: discount investments and operations from the model start, as a single solve does): the
    split solves, the hand-written solves with only their own period in ``years_represented`` and the
    hand-written solves with the whole map build the same LPs.  inv_p2 discounts both its investment and its
    operations from the start of the run: 285.813758 EUR/MW and 152 250.129 EUR per MW and step.  Before, the
    split and own-map solves gave 364.778830 / 194 314.032 (both from the solve's start) and the whole-map solves
    285.813758 / 194 314.032 (investments from the run start, operations from the solve's start)."""
    split, _ = scm.run_spec(scm.model_a("split", model_rows=INFLATION), tmp_path_factory.mktemp("split"))
    own, _ = scm.run_spec(scm.model_a("hand", model_rows=INFLATION, years_own_periods=True),
                          tmp_path_factory.mktemp("own"))
    whole, _ = scm.run_spec(scm.model_a("hand", model_rows=INFLATION), tmp_path_factory.mktemp("whole"))
    assert not scm.lp_differences(split, own)
    assert not scm.lp_differences(split, whole)
    assert [solve for solve, _, _ in split] == ["inv_p1", "inv_p2"]
    _coefficients(split[0][1], {"v_invest_p[gen,p1]": INV_P1, "v_flow[gen,gen,elec,p1,t01]": OPS_P1})
    _coefficients(split[1][1], {"v_invest_p[gen,p2]": INV_P2, "v_flow[gen,gen,elec,p2,t03]": OPS_P2})


def test_a_look_ahead_split_counts_from_the_model_start(tmp_path_factory):
    """The look-ahead ladder (inv_p1 over p1 + p2, inv_p2 over p2) with inflation 0.05: inv_p1 starts the run
    (its p1 decision is paid over p1 and p2: 650.592588 EUR/MW, a p2 decision 285.813758 EUR/MW, operations in p2
    152 250.129); inv_p2 has the same p2 coefficients.  Before, inv_p2 had 364.778830 / 194 314.032.  The split
    builds the hand-written LPs."""
    split, _ = scm.run_spec(_with_inflation(scm.model_a_ladder("split", LOOK_AHEAD)), tmp_path_factory.mktemp("s"))
    hand, _ = scm.run_spec(_with_inflation(scm.model_a_ladder("hand", LOOK_AHEAD)), tmp_path_factory.mktemp("h"))
    assert not scm.lp_differences(split, hand)
    assert [solve for solve, _, _ in split] == ["inv_p1", "inv_p2"]
    _coefficients(split[0][1], {"v_invest_p[gen,p1]": INV_P1_WINDOW, "v_invest_p[gen,p2]": INV_P2,
                                "v_flow[gen,gen,elec,p1,t01]": OPS_P1, "v_flow[gen,gen,elec,p2,t03]": OPS_P2})
    _coefficients(split[1][1], {"v_invest_p[gen,p2]": INV_P2, "v_flow[gen,gen,elec,p2,t03]": OPS_P2})


def test_a_nested_split_counts_from_the_model_start(tmp_path_factory):
    """The split ladder containing the dispatch solve ``disp`` (:func:`_nested`), with inflation 0.05: the split
    solve inv_p2 discounts from the start of the run (285.813758 / 152 250.129; before 364.778830 / 194 314.032).
    The contained ``disp_p1`` / ``disp_p2`` are sub-solves of ``disp``, which starts the run, so their operations
    were already discounted from the run start and stay 194 314.032 / 152 250.129."""
    built, steps = scm.run_spec(_with_inflation(_nested(("p1", "p2"))), tmp_path_factory.mktemp("nested"))
    assert list(steps) == ["inv_p1", "disp_p1", "inv_p2", "disp_p2"] and len(built) == 4
    mps = {solve: path for solve, (_, path, _) in zip(steps, built)}  # build order is the solve order
    _coefficients(mps["inv_p1"], {"v_invest_p[gen,p1]": INV_P1, "v_flow[gen,gen,elec,p1,t01]": OPS_P1})
    _coefficients(mps["inv_p2"], {"v_invest_p[gen,p2]": INV_P2, "v_flow[gen,gen,elec,p2,t03]": OPS_P2})
    _coefficients(mps["disp_p1"], {"v_flow[gen,gen,elec,p1,t01]": OPS_P1})
    _coefficients(mps["disp_p2"], {"v_flow[gen,gen,elec,p2,t03]": OPS_P2})


def test_a_nested_look_ahead_ladder_written_by_hand_is_the_split_one_with_inflation(tmp_path_factory):
    """Item 243 with inflation 0.05: the hand-written look-ahead ladder containing ``disp`` builds the split
    ladder's LPs, solve by solve, because every solve discounts from the start of the run (item 242).  Before
    item 242 the hand-written inv_p2 discounted its operations from its own start (and before item 243 it did not
    run)."""
    split, split_steps = scm.run_spec(_with_inflation(_nested_look_ahead("split")), tmp_path_factory.mktemp("s"))
    hand, hand_steps = scm.run_spec(_with_inflation(_nested_look_ahead("hand")), tmp_path_factory.mktemp("h"))
    assert list(split_steps) == list(hand_steps) == ["inv_p1", "disp_p1", "inv_p2", "disp_p2"]
    assert not scm.lp_differences(split, hand)


def test_a_single_solve_in_rolls_realizing_different_periods_is_an_error(tmp_path):
    """A rolling solve over p1 and p2 (2 h windows and jumps: one roll per period) contains the single solve
    ``disp`` over both periods: ``disp`` keeps its name in every roll, so the p2 roll's run would replace the p1
    roll's (the item 243 mechanism).  The run stops and names the rolls' periods."""
    from flextool.engine.input._solve_state import FlexToolConfigError

    spec = scm.model_a_ladder("hand", {"realized_periods": {"r": ["p1", "p2"]}}, contains=["disp"])
    pv = [r for r in spec["parameter_values"] if (r[0], r[1], r[2]) != ("solve", "inv_r", "solve_mode")]
    pv += [["solve", "inv_r", "solve_mode", scm.pack("rolling_window", "str"), "init"],
           ["solve", "inv_r", "rolling_solve_horizon", scm.pack(2.0, "float"), "init"],
           ["solve", "inv_r", "rolling_solve_jump", scm.pack(2.0, "float"), "init"]]
    spec["parameter_values"] = pv
    with pytest.raises(FlexToolConfigError, match=r"'disp' is contained in the rolling solve 'inv_r', whose rolls "
                                                  r"realize different periods \(p1 and p2\)"):
        scm.run_spec(spec, tmp_path)


def _kept_name_guard(renamed_from: dict[str, str] | None = None):
    """A bare :class:`RecursiveSolveBuilder` with only the state :meth:`_check_kept_name` reads."""
    import logging

    from flextool.engine.solve._recursive_solve import RecursiveSolveBuilder

    builder = object.__new__(RecursiveSolveBuilder)
    builder._kept_name_scopes = {}
    builder._renamed_from = dict(renamed_from or {})
    builder.logger = logging.getLogger("test_kept_name_guard")
    return builder


def test_a_single_solve_in_two_solves_realizing_different_periods_is_an_error():
    """Item 243, the two-parent case: ``disp`` is contained in ``inv_a`` (realizing p1, p2) and in ``inv_b``
    (realizing p3, p4) and shares more than one period with each, so it keeps its name in both and the second run
    would replace the first one's results.  The message lists each parent's periods and the remedy."""
    from flextool.engine.input._solve_state import FlexToolConfigError

    guard = _kept_name_guard()
    guard._check_kept_name("disp", "inv_a", {"p1", "p2"}, {"p1", "p2"})
    guard._check_kept_name("disp", "inv_a", {"p1", "p2"}, {"p1", "p2"})  # the same scope again: no error
    with pytest.raises(FlexToolConfigError) as err:
        guard._check_kept_name("disp", "inv_b", {"p3", "p4"}, {"p3", "p4"})
    assert str(err.value) == (
        "Solve 'disp' is contained in 'inv_a' (realizing p1, p2) and in 'inv_b' (realizing p3, p4). 'disp' shares "
        "more than one of these periods with each of them, so it keeps its name in both and the runs would replace "
        "each other's results. Let each containing solve realize one period that 'disp' has (it then runs as "
        "'disp_<period>'), or contain a different solve in each.")


def test_a_single_solve_below_a_copy_renamed_for_one_period_is_an_error():
    """Item 243, the third-level case: ``mid_p1`` and ``mid_p2`` are the copies of ``mid`` for one period each, but
    each keeps all periods of ``mid`` (p1, p2, p3), so the single solve ``disp`` below them keeps its name under
    both.  The message says the copies stand for the base solve's periods and gives the rolling_window remedy
    (letting the copy realize one period would not help: it already does)."""
    from flextool.engine.input._solve_state import FlexToolConfigError

    guard = _kept_name_guard({"mid_p1": "mid", "mid_p2": "mid"})
    guard._check_kept_name("disp", "mid_p1", {"p1", "p2", "p3"}, {"p1", "p2"})
    with pytest.raises(FlexToolConfigError) as err:
        guard._check_kept_name("disp", "mid_p2", {"p1", "p2", "p3"}, {"p2", "p3"})
    assert str(err.value) == (
        "Solve 'disp' is contained in 'mid_p1' (standing for p1, p2, p3) and in 'mid_p2' (standing for p1, p2, "
        "p3). 'disp' shares more than one of these periods with each of them, so it keeps its name in both and the "
        "runs would replace each other's results. 'mid_p1' is the copy of 'mid' for one period, but it keeps all "
        "periods of 'mid'. 'mid_p2' is the copy of 'mid' for one period, but it keeps all periods of 'mid'. A "
        "single solve contained in such a copy is not renamed per period: make 'disp' a rolling_window solve (its "
        "rolls get their own names), or contain it directly in the solves that realize one period each.")


def _invest_then_rolling() -> dict:
    """Model A: the hand-written inv_p1 (whole ``years_represented`` map), then the rolling dispatch ``roll_p2``
    over p2 (1 h windows and jumps, its own map {p2: 5}) on the capacity inv_p1 built."""
    spec = scm.model_a("hand", model_rows=INFLATION)
    spec["parameter_values"] = [r for r in spec["parameter_values"]
                                if not (r[0] == "solve" and r[1] == "inv_p2")]
    spec["entities"] = [e for e in spec["entities"] if e != ["solve", "inv_p2", None]] + [["solve", "roll_p2", None]]
    for row in spec["parameter_values"]:
        if (row[0], row[2]) == ("model", "solves"):
            row[3] = scm.array(["inv_p1", "roll_p2"])
    spec["parameter_values"] += [
        ["solve", "roll_p2", "solve_mode", scm.pack("rolling_window", "str"), "init"],
        ["solve", "roll_p2", "rolling_solve_horizon", scm.pack(1.0, "float"), "init"],
        ["solve", "roll_p2", "rolling_solve_jump", scm.pack(1.0, "float"), "init"],
        ["solve", "roll_p2", "period_timeset", scm.pack(scm.pmap([("p2", "ts_b")]), "map"), "init"],
        ["solve", "roll_p2", "realized_periods", scm.array(["p2"]), "init"],
        ["solve", "roll_p2", "years_represented", scm.pack(scm.pmap([("p2", 5.0)]), "map"), "init"]]
    return spec


def test_rolls_of_a_later_solve_count_from_the_model_start(tmp_path_factory):
    """The rolls of a rolling solve that starts after the run start (a synthetic sub-solve each, discounted by
    the per-solve seeds of ``emit/_period_calc``): every roll of roll_p2 discounts its operations from the start
    of the run, 152 250.129 EUR per MW and step (before: 194 314.032, from the start of roll_p2)."""
    built, steps = scm.run_spec(_invest_then_rolling(), tmp_path_factory.mktemp("roll"))
    rolls = [(solve, path) for solve, path, _ in built if solve != "inv_p1"]
    assert len(rolls) == 2, [solve for solve, _, _ in built]
    for (solve, path), step in zip(rolls, ("t03", "t04")):
        _coefficients(path, {f"v_flow[gen,gen,elec,p2,{step}]": OPS_P2})


def test_fractional_years_represented_count_the_fraction(tmp_path_factory):
    """Item 259: one solve over p1 (2.5 years) and p2 (5 years) with inflation 0.05.  p2 starts 2.5 years after
    the run start, so (hand calculation, as above) its investment is A * 1.05**-2.5 * F_inv(p1) = 322.891326 EUR/MW
    and its operations 10 * 4380 * 1.05**-2.5 * F_ops(p1) = 172 000.978 EUR per MW and step.  Before, the half
    year of p1 counted as a whole year: p2 started at year 3 (315.109668 / 167 855.767)."""
    built, _ = scm.run_spec(scm.model_a("single", years=(2.5, 5.0), model_rows=INFLATION),
                            tmp_path_factory.mktemp("frac"))
    ((solve, mps, _),) = built
    _coefficients(mps, {"v_invest_p[gen,p2]": 322.891326, "v_flow[gen,gen,elec,p2,t03]": 172_000.978})


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


def _fan_periods(sc, solve: str) -> list[str]:
    return list(dict.fromkeys(str(row[0]) for row in sc.stochastic_branches.get(solve, [])))


def test_each_split_solve_keeps_the_fans_of_the_periods_it_realizes(tmp_path):
    """Look-ahead: st_p1 (p1, p2) realizes p1 and keeps the p1 fan; st_p2 keeps the p2 fan.  A split solve that
    realizes a period without a fan takes the fan of its first period that has one (a deterministic first year
    with a stochastic look-ahead, decision 250 D1)."""
    sc = _config(scm.model_b("split", look_ahead=True), tmp_path / "both")
    assert (_fan_periods(sc, "st_p1"), _fan_periods(sc, "st_p2")) == (["p1"], ["p2"])
    sc = _config(scm.model_b("split", look_ahead=True, fan_periods=("p2",)), tmp_path / "later")
    assert (_fan_periods(sc, "st_p1"), _fan_periods(sc, "st_p2")) == (["p2"], ["p2"])


def test_a_fan_no_split_solve_uses_is_an_error(tmp_path):
    """Forecasts at p1 and p2, but the 2-D map has the row of p1 only: no split solve has p2, and its fan would be
    silently ignored (before, the run went on without it)."""
    spec = scm.model_b("split")
    for row in spec["parameter_values"]:
        if row[:3] == ["solve", "st", "realized_periods"]:
            row[3] = scm.two_d({"p1": ["p1"]})
    _raises(spec, tmp_path, "solve 'st': the stochastic_branches rows of period(s) p2 are used by none of the "
            "solves its 2-D period maps make (st_p1)", "period 'p2' (high at t03, low at t03) is a period of none",
            "add 'p2' to the 2-D maps, in the row whose solve should realize it (realized_periods or "
            "realized_invest_periods). Or remove these rows.")


def test_a_fan_of_a_look_ahead_period_no_split_solve_realizes_is_an_error(tmp_path):
    """Forecasts at p1 and p2 and one row (p1) of the 2-D maps over p1 and p2 that realizes p1 only: st_p1 keeps the
    p1 fan, p2 is in its look-ahead but realized by no split solve, so its fan rows are unused.  The advice names
    the realized lists of a row (review 256/257)."""
    spec = scm.model_b("split", look_ahead=True)
    for row in spec["parameter_values"]:
        if row[:2] == ["solve", "st"] and row[2] in ("realized_periods", "realized_invest_periods"):
            row[3] = scm.two_d({"p1": ["p1"]})
        if row[:3] == ["solve", "st", "invest_periods"]:
            row[3] = scm.two_d({"p1": ["p1", "p2"]})
    _raises(spec, tmp_path, "solve 'st': the stochastic_branches rows of period(s) p2 are used by none of the "
            "solves its 2-D period maps make (st_p1)", "period 'p2' (high at t03, low at t03) is a period of these "
            "solves, but each of them takes the forecasts of the periods it realizes", "add 'p2' to "
            "realized_periods (or realized_invest_periods) in the row of the 2-D maps whose solve should realize "
            "it. Or remove these rows.")


# ---------------------------------------------------------------------------
# The solver settings of a split solve (feat/solver-security merged with the copy step)
# ---------------------------------------------------------------------------


def _gurobi_split(time_limit_name: str) -> dict:
    """Model A split, on Gurobi with ``solver_mip_gap`` 0.2 and ``solver_arguments`` {*time_limit_name*: 800}."""
    return scm.model_a("split", solve_rows=[
        ["solver", scm.pack("gurobi", "str")],
        ["solver_mip_gap", scm.pack(0.2, "float")],
        ["solver_arguments", scm.pack(scm.pmap([(time_limit_name, 800.0)]), "map")]])


def test_a_split_solve_runs_on_its_solves_solver_and_is_checked_as_it(tmp_path):
    """The split solves are the solves the run checks before its first solve (item 240 B1 / B2): each has the
    database solve's solver, ``solver_arguments`` and ``solver_mip_gap`` (``solver_config_for`` agrees with
    ``solver_settings.solvers``), and a refused option is reported once, under the database solve's name."""
    from flextool.engine.solve._solver_dispatch import FlexToolUserError, check_run_solver_options

    sc = _config(_gurobi_split("TimeLimit"), tmp_path / "ok")
    assert sc.solves_in_run() == ["inv_p1", "inv_p2"]
    assert sc.database_solve_names() == {"inv_p1": "inv", "inv_p2": "inv"}
    parent = sc.solver_config_for("inv")
    assert (parent.name, parent.mip_gap) == ("gurobi", 0.2)
    for solve in ("inv_p1", "inv_p2"):
        assert sc.solver_config_for(solve) == parent
        assert sc.solver_settings.arguments[solve] == sc.solver_settings.arguments["inv"]
    check_run_solver_options(sc.solver_settings, solves=sc.solves_in_run(), shown_as=sc.database_solve_names())

    bad = _config(_gurobi_split("NoSuchParameter"), tmp_path / "bad")
    with pytest.raises(FlexToolUserError, match="solver_arguments of solve inv: ") as err:
        check_run_solver_options(bad.solver_settings, solves=bad.solves_in_run(), shown_as=bad.database_solve_names())
    assert "inv_p1" not in str(err.value) and "inv_p2" not in str(err.value), str(err.value)


def test_solves_in_run_reads_a_contained_solve_given_as_one_name():
    """``contains_solves`` may hold one name instead of a list (a partly built config); both are followed, in
    order, each solve once."""
    from types import SimpleNamespace

    from flextool.engine.input._solve_config import SolveConfig

    cfg = SimpleNamespace(model_solve={"m": ["invest", "dispatch"]},
                          contains_solves={"invest": "month", "month": ["week", "invest"]},
                          database_solves=["invest", "month", "week", "dispatch", "unused"])
    assert SolveConfig.solves_in_run(cfg) == ["invest", "dispatch", "month", "week"]
    cfg.model_solve = {}
    assert SolveConfig.solves_in_run(cfg) == ["invest", "month", "week", "dispatch", "unused"]
