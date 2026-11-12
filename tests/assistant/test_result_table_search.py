"""find_result_tables' search (assistant tools batch B1, T2): the words users ask in, and the main tables of a few
common questions first, in ``results._catalog.search`` only (the documentation search is not changed).

Evidence (helpfulness analysis 2026-11-12): "fuel" found no table (#28), "invest" only the dual tables (#11),
"unserved energy" put start-up energy first and no slack table, "cost" the per-entity cost tables first.  The
expected tables are the ones the design names for each question; the pins at the end are queries whose ranking
must not move (they trigger no synonym or preference).
"""
from __future__ import annotations

import json

import pytest

from flextool.assistant.registry import call_tool
from flextool.assistant.results import _catalog

CAPACITY = ["unit_capacity_ed_p", "connection_capacity_ed_p", "node_capacity_ed_p"]
SYSTEM_COSTS = ["annualized_costs_d_p", "costs_discounted_p_", "costs_discounted_d_p", "costs_dt_p"]
UNSERVED = ["node_slack_up_d_e", "node_slack_up_dt_e", "node_d_ep"]


def _stems(query: str, n: int = 8) -> list[str]:
    return [r["stem"] for r in _catalog.search(query, n)]


@pytest.mark.parametrize("query,first", [
    # Synonyms + preferences.
    ("fuel", ["costs_dt_p", "annualized_costs_d_p"]),
    ("fuel cost in the first hour", ["costs_dt_p", "annualized_costs_d_p"]),
    ("invest", CAPACITY),
    ("investment", CAPACITY),
    ("how much was built", CAPACITY),
    ("capacity", CAPACITY),
    # Another word of the question that names a table of the group puts it first.
    ("connection capacity", ["connection_capacity_ed_p"]),
    ("storage capacity", ["node_capacity_ed_p"]),
    ("unserved energy", UNSERVED),
    ("loss of load", UNSERVED),
    ("not served demand", UNSERVED),
    # "value" does not lift the dual penalty: the value of lost load is about unserved energy.
    ("value of lost load", UNSERVED),
    ("excess load", ["node_slack_down_dt_e", "node_slack_down_d_e"]),
    ("cost", SYSTEM_COSTS),
    ("objective", SYSTEM_COSTS),
    # Specific phrases outrank the general preferences.
    ("capacity margin", ["nodeGroup_slack_capacity_margin_d_g", "nodeGroup_capacity_margin_entity_d_g",
                         "nodeGroup_capacity_margin_d_g"]),
    ("capacity factor", ["connection_cf_d_e", "unit_inputs_cf_d_ee", "unit_outputs_cf_d_ee"]),
    ("unit cost", ["cost_unit_annualized_d_ec", "cost_unit_discounted_d_ec"]),
    # Asking for duals: no penalty.
    ("dual", ["dual_invest_effective_connection_d_e", "dual_invest_effective_unit_d_e",
              "dual_invest_effective_node_d_e"]),
    ("marginal value of investment", ["dual_invest_effective_connection_d_e", "dual_invest_effective_unit_d_e",
                                      "dual_invest_effective_node_d_e"]),
    # No-regression pins: no synonym, no preference, no dual table near the top.
    ("nodal energy price", ["node_prices_dt_e"]),
    ("co2 price", ["co2_price_period_d_g", "co2_price_total_d_g"]),
    ("emissions", ["CO2__", "CO2_d_g", "process_co2_d_eee"]),
    ("slack", ["nodeGroup_slack_inertia_dt_g", "node_slack_down_dt_e", "node_slack_up_dt_e"]),
    ("curtailment", ["unit_curtailment_share_outputNode_d_ee", "unit_curtailment_outputNode_dt_ee"]),
])
def test_the_main_tables_come_first(query: str, first: list[str]) -> None:
    assert _stems(query)[:len(first)] == first


def test_dual_tables_rank_below_the_capacity_tables_on_invest() -> None:
    """Without dual, marginal or shadow in the question a dual table's score is halved: "invest" lists the three
    capacity tables, then the duals (before: only the duals)."""
    stems = _stems("invest", 6)
    assert stems[:3] == CAPACITY and all(s.startswith("dual_") for s in stems[3:6])


def test_the_tool_lists_the_preferred_tables(ctx) -> None:
    out = call_tool(ctx, "find_result_tables", {"query": "fuel"})
    assert out.ok, out.message
    tables = [t["table"] for t in json.loads(out.text)["tables"]]
    assert tables[:2] == ["costs_dt_p", "annualized_costs_d_p"]


def test_the_documentation_search_is_not_changed() -> None:
    """The synonyms live in the result-table search only: the shared tokenizer adds no terms."""
    from flextool.assistant.textsearch import tokens

    assert tokens("fuel invest unserved") == ["fuel", "invest", "unserved"]
