"""A shared investment in a stochastic fan without recourse reaches the fan's branch periods.

Under ``stochastic_invest_method = none`` the investment is one decision on the period label and the fan's
branch periods operate the capacity.  The branch periods take their label's year in every period walk, so a
decision that starts to operate in a later period (a construction lead time, or a decision made only in the
second period) is available in that period's branches.

Fixture ``stoch_two_period_invest.json`` with the alternatives ``init, base, stoch`` (no ``recourse``), by hand:

* periods p2035 (year 0) and p2040 (year 1), ``years_represented`` 1 each; ts1 = t0001..t0003 (p2035),
  ts2 = t0004..t0006 (p2040), 1 h steps; the fan (high weight 1, low weight 3 → 0.25 / 0.75) covers both periods.
* demand 100 MW in p2035, 130 MW in p2040; wind 100 MW, profile high 0.4 / 0.4, low 0.2 (p2035) / 0.1 (p2040).
  The p2040 deficit is 130 − 40 = 90 MW (high) and 130 − 10 = 120 MW (low).  One shared peaker decision has to
  cover the larger, so it builds **120 MW** (``invest_max_total`` 150, invest_cost 1 vs penalty 1e4).
* (i) lead time 1 period: a decision in p2035 (year 0) starts at year 1 = p2040; it is in construction in the
  p2035 branches and operates in the p2040 branches.  A decision in p2040 would start at year 2 = the end of the
  horizon, so it is not offered.  → v_invest_p[peaker, p2035] = 120.
* (ii) no lead time, ``invest_periods = [p2040]``: v_invest_p[peaker, p2040] = 120.
* In both, nothing can operate in p2035: penalty = (0.25·60 + 0.75·80) MW · 3 h · 8760/3 · 1e4 €/MWh
  = 75 · 8760 · 1e4 = 6.570e9 €; the 120 MW peaker adds a small annuity, the same in both cases (invest_cost is
  taken in the period the capacity starts to operate, from which the annuity is paid).
"""
from __future__ import annotations

import base64
import json

import polars as pl
import pytest

from tests.engine._unit_db import FLEXTOOL_ROOT, b64, make_db, run, without_realization

FAN = FLEXTOOL_ROOT / "tests" / "fixtures" / "stoch_two_period_invest.json"
SOLVE = "stoch_2p_inv"
P1_PENALTY = 75.0 * 8760.0 * 1e4  # 6.570e9 €: the p2035 penalty, see the module docstring


def _str_array(xs: list[str]) -> list:
    payload = json.dumps({"value_type": "str", "data": xs}).encode()
    return [base64.b64encode(payload).decode(), "array"]


def _solve(tmp_path_factory, case: str):
    if case == "lead_time":
        values = [["unit", "peaker", "construction_lead_time", b64(1.0, "float")]]
    else:
        values = [["solve", SOLVE, "invest_periods", _str_array(["p2040"])],
                  ["solve", SOLVE, "realized_invest_periods", _str_array(["p2040"])]]
    url, scen = make_db(tmp_path_factory, alt=f"fan_{case}", values=values,
                        base_chain=["init", "base", "stoch"], fixture_json=FAN)
    steps = without_realization(run(tmp_path_factory, url, scen))
    return list(steps.values())[-1]


def _utf8(df: pl.DataFrame) -> pl.DataFrame:
    return df.with_columns(pl.col(c).cast(pl.Utf8) for c in df.columns if c != "value")


def _invest(step) -> dict[str, float]:
    df = _utf8(step.solution.value("v_invest_p")).filter(pl.col("p") == "peaker")  # virtual_unitsize 1: MW
    return {d: float(v) for d, v in zip(df["d"], df["value"]) if abs(v) > 1e-9}


def _peak_flow(step, period: str) -> float:
    df = (_utf8(step.solution.value("v_flow"))
          .filter((pl.col("p") == "peaker") & (pl.col("d") == period)))
    return float(df["value"].max()) if df.height else 0.0


@pytest.fixture(scope="module")
def lead_time(tmp_path_factory):
    return _solve(tmp_path_factory, "lead_time")


@pytest.fixture(scope="module")
def second_period_only(tmp_path_factory):
    return _solve(tmp_path_factory, "second_period_only")


def test_lead_time_decision_operates_in_the_later_branches(lead_time):
    assert _invest(lead_time) == pytest.approx({"p2035": 120.0}, abs=1e-6)
    assert _peak_flow(lead_time, "p2040_low") == pytest.approx(120.0, abs=1e-6)
    # high needs 90 MW; the peaker and wind both run at zero cost, so 90..120 MW are equally optimal.
    assert 90.0 - 1e-6 <= _peak_flow(lead_time, "p2040_high") <= 120.0 + 1e-6
    assert _peak_flow(lead_time, "p2035_low") == pytest.approx(0.0, abs=1e-6)
    fd = lead_time.flex_data
    commission = {d for e, o, d in _utf8(fd.edd_invest_commission).rows() if (e, o) == ("peaker", "p2035")}
    building = {d for e, o, d in _utf8(fd.edd_invest_in_construction).rows() if (e, o) == ("peaker", "p2035")}
    assert {"p2040_high", "p2040_low"} <= commission
    assert not commission & {"p2035", "p2035_high", "p2035_low"}
    assert {"p2035_high", "p2035_low"} <= building
    assert not building & {"p2040", "p2040_high", "p2040_low"}


def test_second_period_decision_operates_in_its_branches(second_period_only):
    assert _invest(second_period_only) == pytest.approx({"p2040": 120.0}, abs=1e-6)
    assert _peak_flow(second_period_only, "p2040_low") == pytest.approx(120.0, abs=1e-6)
    # high needs 90 MW; the peaker and wind both run at zero cost, so 90..120 MW are equally optimal.
    assert 90.0 - 1e-6 <= _peak_flow(second_period_only, "p2040_high") <= 120.0 + 1e-6


def test_lead_time_equals_deciding_in_the_operating_period(lead_time, second_period_only):
    assert lead_time.solution.obj == pytest.approx(second_period_only.solution.obj, rel=1e-9)
    assert lead_time.solution.obj == pytest.approx(P1_PENALTY, rel=1e-3)
    assert lead_time.solution.obj > P1_PENALTY


# ---------------------------------------------------------------------------
# Behaviour-change notice
# ---------------------------------------------------------------------------

NOTICE_ID = "stochastic_branch_period_years"
NOTICE_MARKER = f"flextool_behaviour_change:{NOTICE_ID}"


def _notice_lines(tmp_path_factory, capsys, alt: str, chain: list[str], marker: bool) -> list[str]:
    from spinedb_api import DatabaseMapping

    from flextool.engine import run_chain_from_db
    from flextool.engine._behaviour_changes import BEHAVIOUR_CHANGES

    url, scen = make_db(tmp_path_factory, alt=alt, values=[], base_chain=chain, fixture_json=FAN)
    if marker:
        with DatabaseMapping(url) as db:
            db.add_item("metadata", name=NOTICE_MARKER, value="77")
            db.commit_session("behaviour-change marker")
    capsys.readouterr()
    run_chain_from_db(url, scen, work_folder=tmp_path_factory.mktemp(f"{alt}_w"))
    summary = BEHAVIOUR_CHANGES[NOTICE_ID].summary
    return [ln for ln in capsys.readouterr().out.splitlines() if summary in ln]


def test_notice_names_the_investment_entities(tmp_path_factory, capsys) -> None:
    lines = _notice_lines(tmp_path_factory, capsys, "fan_notice", ["init", "base", "stoch"], marker=True)
    assert len(lines) == 1
    assert "Affected entities: peaker." in lines[0] and NOTICE_MARKER in lines[0]


@pytest.mark.parametrize("chain, marker", [
    (["init", "base", "stoch"], False),
    (["init", "base"], True),
    (["init", "base", "stoch", "recourse"], True),
], ids=["no_marker", "deterministic", "recourse"])
def test_no_notice(tmp_path_factory, capsys, chain, marker) -> None:
    assert _notice_lines(tmp_path_factory, capsys, f"fan_no_notice_{len(chain)}_{marker}", chain, marker) == []
