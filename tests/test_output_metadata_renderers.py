"""Output-metadata renderers (follow-up 2): CSV / Excel / Spine / plot all
surface the unit + semantics from the single source ``_output_meta``.

These are pure / lightweight tests — no full solve.  They lock the contract
of each renderer's metadata feed and ratchet Spine schema-description drift.
"""

from __future__ import annotations

import json

import pandas as pd
import pytest

from flextool.plot_outputs.config import default_ylabel_for
from flextool.process_outputs import write_spinedb
from flextool.process_outputs._output_meta import (
    datapackage_resource,
    output_metadata_rows,
    result_key_summary,
    result_variant_summary,
)
from flextool.process_outputs.write_outputs import write_excel_with_metadata


# ── CSV sidecar: datapackage_resource ───────────────────────────────────────

def test_datapackage_resource_tags_measure_and_dimension_fields():
    res = datapackage_resource(
        "annualized_costs_d_p", "annualized_costs__d.csv",
        index_names=["solve", "period"], measure_columns=["commodity_cost"],
    )
    assert res["path"] == "annualized_costs__d.csv"
    assert res["profile"] == "tabular-data-resource"
    fields = {f["name"]: f for f in res["schema"]["fields"]}
    # dimension index columns
    assert fields["solve"]["flextool:semantics"] == "dimension"
    assert fields["period"]["type"] == "string"
    # measure column carries derived unit + semantics + formula
    cc = fields["commodity_cost"]
    assert cc["unit"] == "M CUR/a"
    assert cc["flextool:semantics"] == "annualized"
    assert cc["type"] == "number"
    assert "description" in cc and cc["flextool:formula"]


def test_datapackage_resource_mixed_unit_table():
    res = datapackage_resource(
        "flowGroup_gd_p", "flowGroup__gd.csv",
        index_names=["group"], measure_columns=["cumulative_flow", "average_flow"],
    )
    fields = {f["name"]: f for f in res["schema"]["fields"]}
    assert fields["cumulative_flow"]["unit"] == "MWh"
    assert fields["average_flow"]["unit"] == "MW"
    assert fields["average_flow"]["flextool:semantics"] == "average"


def test_datapackage_resource_undeclared_output_is_all_strings():
    res = datapackage_resource(
        "totally_undeclared", "x.csv",
        index_names=["a"], measure_columns=["x", "y"],
    )
    fields = res["schema"]["fields"]
    assert all(f["type"] == "string" for f in fields)
    assert all(f["flextool:semantics"] == "dimension" for f in fields)


# ── Excel: rows + workbook round-trip ───────────────────────────────────────

def test_output_metadata_rows_drops_dimensions():
    rows = output_metadata_rows(
        "annualized_costs_d_p", ["solve", "period", "commodity_cost"])
    assert [r["column"] for r in rows] == ["commodity_cost"]
    assert rows[0]["unit"] == "M CUR/a"
    assert rows[0]["semantics"] == "annualized"


def test_output_metadata_rows_undeclared_empty():
    assert output_metadata_rows("nope", ["a", "b"]) == []


def test_excel_writer_embeds_metadata_sheet_and_comments(tmp_path):
    openpyxl = pytest.importorskip("openpyxl")
    df = pd.DataFrame(
        {"commodity_cost": [1.0, 2.0]},
        index=pd.MultiIndex.from_tuples(
            [("s", "p1"), ("s", "p2")], names=["solve", "period"]),
    )
    xlsx = tmp_path / "out.xlsx"
    write_excel_with_metadata(
        str(xlsx), [("annualized costs", "annualized_costs_d_p", df)])

    wb = openpyxl.load_workbook(xlsx)
    assert "_output_metadata" in wb.sheetnames
    meta = wb["_output_metadata"]
    header = [c.value for c in next(meta.iter_rows(max_row=1))]
    assert header == ["sheet", "output", "column", "unit",
                      "semantics", "tooltip", "long", "formula"]
    body = [c.value for c in next(meta.iter_rows(min_row=2, max_row=2))]
    assert body[2] == "commodity_cost" and body[3] == "M CUR/a"

    # Header-cell comment on the measure column (index has 2 levels -> col 2).
    ws = wb["annualized costs"]
    comment = ws.cell(row=1, column=3).comment
    assert comment is not None
    assert "M CUR/a" in comment.text and "annualized" in comment.text


# ── Spine: merge units at DB-create (prose kept, units from _output_meta) ────

def _load_schema():
    with open(write_spinedb._SCHEMA_PATH, encoding="utf-8") as fh:
        return json.load(fh)


def test_spine_merge_fills_and_is_idempotent():
    schema = _load_schema()
    assert any(not e[4] for e in schema["parameter_definitions"])
    write_spinedb._apply_derived_descriptions(schema)
    # every description is now non-empty (empties filled, others unit-merged)
    assert all(e[4] for e in schema["parameter_definitions"])
    # re-applying is a fixed point (strip-and-reprepend the same unit token)
    snapshot = [list(e) for e in schema["parameter_definitions"]]
    write_spinedb._apply_derived_descriptions(schema)
    assert [list(e) for e in schema["parameter_definitions"]] == snapshot


def test_spine_merge_injects_authoritative_units():
    """Units come from `_output_meta`; the hand-written prose is preserved."""
    schema = _load_schema()
    write_spinedb._apply_derived_descriptions(schema)
    desc = {(e[0], e[1]): e[4] for e in schema["parameter_definitions"]}
    # corrected units now lead each description
    assert desc[("group", "flow_t")].startswith("[MW] ")          # rate, not MWh
    assert desc[("node", "balance_t")].startswith("[MW] ")
    assert desc[("unit", "CO2_annualized")].startswith("[t/a] ")   # was [MtCO2]
    assert desc[("group", "CO2_annualized")].startswith("[t/a] ")  # was [Mt]
    assert desc[("unit", "invest_marginal")].startswith("[CUR/kW] ")
    assert desc[("node", "invest_marginal")].startswith("[CUR/kWh] ")
    assert desc[("unit", "startup_cumulative")].startswith("[units/a] ")
    assert desc[("model", "cost_annualized")].startswith("[M CUR/a] ")
    # prose preserved (capacity categories, cost breakdown)
    assert "existing" in desc[("node", "capacity")]
    assert "categor" in desc[("model", "cost_annualized")].lower() \
        or "investment" in desc[("model", "cost_annualized")].lower()
    # params with no mapped transform keep their hand-written description
    assert desc[("connection__node__node", "cf")].lstrip().startswith("[per unit]")


def test_spine_merge_units_match_output_meta():
    """The leading unit token equals the unit `_output_meta` derives."""
    schema = _load_schema()
    write_spinedb._apply_derived_descriptions(schema)
    desc = {(e[0], e[1]): e[4] for e in schema["parameter_definitions"]}
    for key, unit in write_spinedb.derived_param_unit_map().items():
        tag = f"[{unit}]" if unit else "[per unit]"
        assert desc[key].startswith(tag + " ") or desc[key] == tag, \
            f"{key}: expected leading {tag!r}, got {desc[key][:40]!r}"


# ── Plot: default y-axis label ──────────────────────────────────────────────

def test_default_ylabel_uses_unit():
    assert default_ylabel_for("annualized_costs_d_p") == "M CUR/a"
    assert default_ylabel_for("unit_outputNode_dt_ee") == "MW"


def test_default_ylabel_none_for_ratio_and_unknown():
    # a unitless ratio output -> no label
    assert result_key_summary("nodeGroup_VRE_share_d_g")[0] == "ratio"
    assert default_ylabel_for("nodeGroup_VRE_share_d_g") is None
    assert default_ylabel_for("not_an_output") is None


def test_default_ylabel_variant_aware():
    # 'a' (total) strips the /a annual-rate suffix on the value axis
    assert default_ylabel_for("annualized_costs_d_p", "a") == "M CUR"
    # 'w' (weekly) keeps the base unit
    assert default_ylabel_for("annualized_costs_d_p", "w") == "M CUR/a"
    # explicit None / base letters -> base behaviour unchanged
    assert default_ylabel_for("annualized_costs_d_p", "p") == "M CUR/a"
    assert default_ylabel_for("annualized_costs_d_p") == "M CUR/a"


# ── Variant-aware summary (a → total, w → weekly) ───────────────────────────

def test_result_variant_summary_sum_periods_strips_annual_suffix():
    # 'a' sums over the horizon: /a annual-rate suffix dropped, semantics=total
    base = result_key_summary("annualized_costs_d_p")
    assert base[0] == "M CUR/a" and base[1] == "annualized"
    unit, semantics, desc = result_variant_summary("annualized_costs_d_p", "a")
    assert unit == "M CUR"           # /a stripped
    assert semantics == "total"      # horizon total, NOT annual
    assert desc == base[2]           # description carried


def test_result_variant_summary_chunks_keeps_unit_weekly():
    base = result_key_summary("unit_outputNode_dt_ee")
    assert base[0] == "MW"
    unit, semantics, _ = result_variant_summary("unit_outputNode_dt_ee", "w")
    assert unit == "MW"              # unchanged
    assert semantics == "weekly"


def test_result_variant_summary_base_letters_unchanged():
    base = result_key_summary("annualized_costs_d_p")
    assert result_variant_summary("annualized_costs_d_p", "p") == base
    assert result_variant_summary("unit_outputNode_dt_ee", "h") == \
        result_key_summary("unit_outputNode_dt_ee")


def test_result_variant_summary_a_keeps_unit_without_annual_suffix():
    # a base unit that is already absolute (no /a) is untouched by 'a'
    base = result_key_summary("unit_outputNode_dt_ee")  # MW, no /a
    unit, semantics, _ = result_variant_summary("unit_outputNode_dt_ee", "a")
    assert unit == base[0]           # MW kept as-is
    assert semantics == "total"


def test_result_variant_summary_none_base_returns_none():
    # undeclared output / pure membership set -> None for every variant
    assert result_variant_summary("not_an_output", "a") is None
    assert result_variant_summary("nodeGroupDispatch", "w") is None


# ── Corrected units + two-layer description ─────────────────────────────────

def test_dt_group_and_node_flows_are_rate_mw():
    # un-integrated dt flows/balances are MW levels, not per-step MWh
    assert result_key_summary("nodeGroup_flows_dt_g") == \
        ("MW",) + result_key_summary("nodeGroup_flows_dt_g")[1:]
    assert output_metadata_rows("node_dt_ep", ["from_units"])[0]["unit"] == "MW"
    assert output_metadata_rows("node_inflow__dt", ["x"])[0]["unit"] == "MW"


def test_co2_system_total_is_megatonnes_horizon():
    unit, semantics, _ = result_key_summary("CO2__")
    assert unit == "Mt"
    assert semantics == "horizon"          # NOT annualized (×years_represented)


def test_startup_is_units_per_year():
    assert output_metadata_rows("unit_startup_d_e", ["x"])[0]["unit"] == "units/a"


def test_datapackage_uses_long_description_when_present():
    # CO2__ (EMISSION_MT) declares a longer `description`; the datapackage
    # field should carry it, not the short tooltip.
    res = datapackage_resource("CO2__", "CO2.csv", ["model"], ["CO2"])
    field = {f["name"]: f for f in res["schema"]["fields"]}["CO2"]
    assert field["unit"] == "Mt"
    assert "undiscounted horizon total" in field["description"]
