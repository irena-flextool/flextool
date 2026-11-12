"""Item 85: an investment solve's capacity-margin shortfall reaches the outputs of an invest → dispatch chain.

Vehicle: ``tests.json`` chain ``init, west, coal, fullYear, 5weeks, 5weeks_only_invest``: the model solves
``invest_1year_5weeks`` (``invest_periods [p2020]``, timeset ``5weeks``: steps t0001-t0004, t0009-t0012,
t0013-t0016, t0025-t0028, t0037-t0040) and then ``y2020_fullYear_dispatch`` (no investment periods, so no capacity
margin).  Override: ``coal_plant`` existing 1000 MW (the only unit into west; no investment options), group ``cm`` =
{west} with ``capacity_margin`` 243 MW and penalty 1000 CUR/kW/a.

Hand derivation.  The only counted capacity is coal_plant's 1000 MW (firm, availability and capacity_max_coeff 1);
west has no connection or consumption, and with 1000 MW of coal no step dumps energy.  West's demand over the five
weeks peaks at t0037: 807 MW (the ``west`` inflow of the fixture, read from the JSON below).  So at t0037

    requirement = 243 + 807 = 1050 MW,  counted = 1000 MW,  shortfall = 1050 − 1000 = 50 MW,
    penalty = 50 MW · 1000 CUR/kW/a · 1000 kW/MW · inflation 1 / 1e6 = 50 M CUR (inflation_rate is unset, so
    inflation_op = years_represented, 1 by default).

Before item 85 the dispatch solve's zero row for p2020 overwrote the investment solve's shortfall: the slack table
showed 0 and the costs no penalty.  Databases are built from JSON (never a checked-in ``.sqlite``).
"""
from __future__ import annotations

import base64
import json
from pathlib import Path

import pandas as pd
import polars as pl
import pytest

from tests.engine._unit_db import b64, make_db

REPO = Path(__file__).resolve().parents[2]
FIXTURE = json.loads((REPO / "tests" / "fixtures" / "tests.json").read_text())
CHAIN = ["init", "west", "coal", "fullYear", "5weeks", "5weeks_only_invest"]
FIVE_WEEKS = [f"t{i:04d}" for start in (1, 9, 13, 25, 37) for i in range(start, start + 4)]
MARGIN, COAL, PENALTY = 243.0, 1000.0, 1000.0
INVEST, DISPATCH = "invest_1year_5weeks", "y2020_fullYear_dispatch"


def _west_demand() -> dict[str, float]:
    for row in FIXTURE["parameter_values"]:
        if row[4] == "west" and row[1] == "west" and row[2] == "inflow":
            return {t: -v for t, v in json.loads(base64.b64decode(row[3][0]))["data"]}
    raise KeyError("west inflow")


@pytest.fixture(autouse=True)
def _strict_autoscale(monkeypatch):
    monkeypatch.setenv("FLEXTOOL_AUTOSCALE_STRICT", "1")


@pytest.fixture(scope="module")
def chain_run(tmp_path_factory):
    """``(work folder, csv folder, captured stdout)`` of the invest → dispatch chain and its written outputs."""
    import contextlib
    import io

    from flextool.engine import run_chain_from_db
    from flextool.process_outputs.write_outputs import write_outputs

    entities = [["group", "cm", None], ["group__node", ["cm", "west"], None]]
    values = [["unit", "coal_plant", "existing", b64(COAL, "float")],
              ["group", "cm", "capacity_margin_method", b64("manual", "str")],
              ["group", "cm", "capacity_margin", b64(MARGIN, "float")],
              ["group", "cm", "penalty_capacity_margin", b64(PENALTY, "float")]]
    url, scen = make_db(tmp_path_factory, alt="item85", values=values, base_chain=CHAIN, entities=entities)
    work = tmp_path_factory.mktemp("item85_w")
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        steps = run_chain_from_db(url, scen, work_folder=work)
    assert list(steps) == [INVEST, DISPATCH]
    last = next(reversed(steps.values()))
    write_outputs(scen, output_location=str(work / "out"), subdir=scen,
                  output_config_path=str(REPO / "flextool" / "schemas" / "default_plots.yaml"),
                  write_methods=["csv"], fallback_output_location=str(work), raw_output_dir=str(work / "output_raw"),
                  solution=last.solution, solve_name=last.solve_name, flex_data=last.flex_data,
                  flex_data_provider=last.flex_data_provider)
    return work, work / "out" / "output_csv" / scen, out.getvalue()


def test_the_binding_step_is_the_peak_of_the_five_weeks() -> None:
    """The oracle's inputs: the five-week peak of west is 807 MW at t0037, so the shortfall is 50 MW."""
    demand = _west_demand()
    peak_t = max(FIVE_WEEKS, key=lambda t: demand[t])
    assert (peak_t, demand[peak_t]) == ("t0037", 807.0)
    assert MARGIN + demand[peak_t] - COAL == 50.0


def test_the_slack_table_shows_the_investment_solves_shortfall(chain_run) -> None:
    _, csv, _ = chain_run
    slack = pd.read_csv(csv / "slack__capacity_margin__d.csv")
    assert slack["period"].tolist() == ["p2020"]
    assert slack["cm"].tolist() == pytest.approx([50.0], abs=1e-6)


def test_the_penalty_is_in_the_costs(chain_run) -> None:
    _, csv, _ = chain_run
    costs = pd.read_csv(csv / "costs_discounted.csv").set_index("category").iloc[:, 0]
    assert costs["capacity margin penalty"] == pytest.approx(50.0, rel=1e-9)


def test_the_capacity_margin_table_has_the_investment_solves_row(chain_run) -> None:
    _, csv, _ = chain_run
    table = pd.read_csv(csv / "capacity_margin__d.csv")
    assert table[["solve", "period", "group", "binding_time"]].values.tolist() == [[INVEST, "p2020", "cm", "t0037"]]
    row = table.iloc[0]
    assert row["margin"] == pytest.approx(MARGIN)
    assert row["demand"] == pytest.approx(807.0)
    assert row["requirement"] == pytest.approx(1050.0)
    assert row["firm"] == pytest.approx(1000.0)
    assert row["counted"] == pytest.approx(1000.0)
    assert row["shortfall"] == pytest.approx(50.0, abs=1e-6)
    assert row["penalty_cost"] == pytest.approx(50.0, rel=1e-9)
    # The shadow price of the margin equals the penalty while there is a shortfall.
    assert row["capacity_price"] == pytest.approx(PENALTY, rel=1e-6)


def test_the_entity_breakdown_sums_to_the_left_hand_side(chain_run) -> None:
    """The per-entity table (raw per-solve frame: one row per (class, entity)) sums to ``counted`` = the row's
    left-hand side without the slack: coal_plant's 1000 MW, and west dumps nothing."""
    from flextool.engine.output._capacity_margin import ENTITY_PREFIX, REPORT_PREFIX, read_capacity_margin_parts

    work, _, _ = chain_run
    entity = read_capacity_margin_parts(work / "output_raw", ENTITY_PREFIX)
    summary = read_capacity_margin_parts(work / "output_raw", REPORT_PREFIX)
    assert set(entity["solve"].to_list()) == {INVEST}
    assert entity.filter(pl.col("class") == "firm").select("entity", "value").rows() == [("coal_plant", 1000.0)]
    assert entity["value"].sum() == pytest.approx(1000.0)
    assert entity["value"].sum() == pytest.approx(summary["counted"][0], rel=1e-12)


def test_the_run_log_names_the_shortfall(chain_run) -> None:
    _, _, stdout = chain_run
    lines = [ln for ln in stdout.splitlines() if ln.startswith("capacity margin shortfall")]
    assert lines == [f"capacity margin shortfall: solve {INVEST}, group cm, period p2020: 50 MW at t0037 "
                     "(penalty 50 M CUR)"]


def test_a_coarse_row_is_evaluated_per_coarse_step(tmp_path_factory) -> None:
    """Group cm_sink of ``test_coarse_row_rule`` (h2_sink, demand 5 MW at every step, fed only by the daily
    connection h2_link; margin 10 MW): its row is one capacityMargin_coarse row per day, so the binding step is a
    day's first hour.  No firm or VRE unit feeds h2_sink, so nothing can hold the margin: h2_link counts by its
    flow, 5 MW, the requirement is 10 + 5 = 15 MW and the shortfall the whole margin, 10 MW."""
    from flextool.engine.model._group_slack import capacity_margin_report
    from tests.engine._unit_db import last_stochastic
    from tests.engine.test_coarse_row_rule import _CM_SINK, _CM_SINK_ENTS, _DAYS, _case

    step = last_stochastic(_case(tmp_path_factory, "cm_out_coarse", _CM_SINK, _CM_SINK_ENTS))
    summary, entity = capacity_margin_report(step.flex_data, step.solution)
    row = summary.filter(pl.col("group") == "cm_sink").row(0, named=True)
    assert row["binding_time"] in _DAYS
    assert (row["demand"], row["requirement"]) == (pytest.approx(5.0), pytest.approx(15.0))
    assert (row["firm"], row["vre"]) == (0.0, 0.0)
    assert row["connections_net"] == pytest.approx(5.0)
    assert row["shortfall"] == pytest.approx(10.0, rel=1e-6)
    sink = entity.filter(pl.col("group") == "cm_sink")
    assert sink.filter(pl.col("class") == "connection").select("entity", "value").rows() == [
        ("h2_link", pytest.approx(5.0))]
    assert sink["value"].sum() == pytest.approx(row["counted"], rel=1e-12)
