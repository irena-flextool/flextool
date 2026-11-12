"""Scaled copies of large values (assistant tools batch B1, T6): "≈1.03 TWh/a" beside 1 030 000 MWh/a, for reading.

Each expected text is worked out by hand: the value divided by the larger unit's factor, rounded to three
significant digits.  The exact values are never changed.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from flextool.assistant.context import make_context
from flextool.assistant.registry import call_tool
from flextool.assistant.results._catalog import Measure
from flextool.assistant.unit_wording import scaled, table_scaling
from tests.assistant.conftest import make_project, record_run


@pytest.mark.parametrize(("value", "unit", "expected"), [
    (1_030_000.0, "MWh/a", "≈1.03 TWh/a"),
    (999_960.0, "MWh", "≈1 TWh"),             # rounded first: 0.99996 TWh is 1 TWh, never "1e+03 GWh"
    (999.6, "MWh", "≈1 GWh"),
    (999.4, "MWh", None),                     # under the first larger unit
    (2_500.0, "MW", "≈2.5 GW"),
    (5e9, "MWh", "≈5000 TWh"),                # the largest unit has no "e+03"
    (-20_000.0, "MWh", "≈-20 GWh"),
    (50_000.0, "t", "≈50 kt"),
    (2e6, "t/a", "≈2 Mt/a"),
    (3e6, "CUR", "≈3 M CUR"),
    (5e5, "CUR", None),                       # CUR only from a million on
])
def test_a_copy_in_the_largest_unit_that_rounds_to_one_or_more(value, unit, expected) -> None:
    assert scaled(value, unit) == expected


@pytest.mark.parametrize("unit", ["M CUR", "M CUR/a", "Mt", "CUR/MWh", "CUR/t", "CUR/kW", "MW·s", "MW·s/a", "t/h",
                                  "tCO2", "kg", "units", "", None])
def test_rates_scaled_and_other_units_get_no_copy(unit) -> None:
    assert scaled(1e7, unit) is None


def test_a_carrier_is_scaled_only_in_power_or_energy() -> None:
    """A node's unit_of_measure may be t or a unit of its own: its flows are not scaled; MWh and MW are."""
    assert scaled(5e4, "t/a", carrier=True) is None
    assert scaled(5e4, "MWh/a", carrier=True) == "≈50 GWh/a"
    energy, tonnes = Measure("MWh/a", "flow", carrier=True), Measure("t/a", "flow", carrier=True)
    assert table_scaling({"*": "MWh/a"}, [energy]) == ("MWh/a", True)
    assert table_scaling({"*": "t/a"}, [tonnes]) is None
    assert table_scaling({"*": "t/a"}, [Measure("t/a", "emission")]) == ("t/a", False)
    # Several units in one table, or a unit that varies by row: no column of copies.
    assert table_scaling({"a": "MWh/a", "b": "t/a"}, [energy, tonnes]) is None
    assert table_scaling({"*": "MWh/a"}, [energy], row_units={"n1": {"x": "MWh/a"}}) is None
    assert table_scaling({"*": "CUR/MWh"}, [Measure("CUR/MWh", "price")]) is None


def _big_slack(folder: Path) -> None:
    import pandas as pd

    from tests.assistant.results_tables import PERIODS, write_result

    periods = pd.Index(PERIODS, name="period")
    # node_slack_up_d_e [MWh/a]: west 25 000 and 400 in the two periods, east 1 500 in the first.
    slack = pd.DataFrame({"west": [25_000.0, 400.0], "east": [1_500.0, 0.0]}, index=periods)
    slack.columns.name = "node"
    write_result(folder, "node_slack_up_d_e", slack, "coal")


def test_slack_entries_and_totals_get_their_copies(tmp_path: Path, tests_db: Path) -> None:
    from tests.assistant.results_tables import PERIODS, succeed_writing

    project = make_project(tmp_path, tests_db)
    record_run(project, "coal_1", "coal", succeed_writing(_big_slack))
    out = call_tool(make_context(project), "get_slack_summary", {"run": "coal_1"})
    assert out.ok, out.message
    slack = json.loads(out.text)["slacks"][0]
    first, second = PERIODS
    assert slack["entries"] == [
        {"entity": ["west"], "period": first, "value": 25_000.0, "scaled": "≈25 GWh/a"},
        {"entity": ["east"], "period": first, "value": 1_500.0, "scaled": "≈1.5 GWh/a"},
        {"entity": ["west"], "period": second, "value": 400.0}]
    # Totals over the nodes: 25 000 + 1 500 = 26 500 (≈26.5 GWh/a); 400 gets no copy.
    assert slack["total"] == {first: 26_500.0, second: 400.0}
    assert slack["total_scaled"] == {first: "≈26.5 GWh/a"}
