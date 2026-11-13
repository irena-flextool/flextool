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
* model B (forecasts high / low with weights 1 / 3 in both periods): 6 570 000 EUR and 7 227 000 EUR.
* ``separate_dispatch_noinvest.json`` (one period) split by a 2-D ``realized_periods``: 6 570 000 EUR and the
  realization solve's 4 380 000 EUR (the hand calculation of ``test_separate_dispatch_noinvest_capacity.py``).
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


def _scenario(data: dict) -> str:
    return data["scenarios"][0][0]


CASES = [
    pytest.param(lambda mode: scm.model_a(mode), {"inv_p1": 43_840_121.29, "inv_p2": 43_800_000.0},
                 id="invest-ladder"),
    pytest.param(_nsd, {"inv_p1": 43_840_121.29, "inv_p2": 43_800_000.0}, id="new_stepduration"),
    pytest.param(_lead_time, {"inv_p1": 43_843_840_121.29, "inv_p2": 43_800_000.0}, id="construction_lead_time"),
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
