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
                      "semantics", "tooltip", "formula"]
    body = [c.value for c in next(meta.iter_rows(min_row=2, max_row=2))]
    assert body[2] == "commodity_cost" and body[3] == "M CUR/a"

    # Header-cell comment on the measure column (index has 2 levels -> col 2).
    ws = wb["annualized costs"]
    comment = ws.cell(row=1, column=3).comment
    assert comment is not None
    assert "M CUR/a" in comment.text and "annualized" in comment.text


# ── Spine: fill empties + drift ratchet ─────────────────────────────────────

def _load_schema():
    with open(write_spinedb._SCHEMA_PATH, encoding="utf-8") as fh:
        return json.load(fh)


def test_spine_fills_all_empty_descriptions_idempotently():
    schema = _load_schema()
    assert any(not e[4] for e in schema["parameter_definitions"])
    write_spinedb._apply_derived_descriptions(schema)
    assert all(e[4] for e in schema["parameter_definitions"])
    # re-applying changes nothing (non-empty descriptions are never clobbered)
    snapshot = [list(e) for e in schema["parameter_definitions"]]
    write_spinedb._apply_derived_descriptions(schema)
    assert [list(e) for e in schema["parameter_definitions"]] == snapshot


# Known, intentional schema-vs-derived unit differences (notation folded out
# by ``_norm`` below).  Each is a documented convention gap, NOT a bug:
#   * the schema omits the annualized ``/a`` suffix (MWh vs MWh/a, t vs t/a);
#   * CO2 totals are in Mt while per-entity emissions derive in t;
#   * invest duals follow the CUR/kW(h) invest-cost convention (handoff §4),
#     the schema text says CUR/MW(h);
#   * ``_dt`` energy is rate (MW) in the schema, per-step energy (MWh) in the
#     metadata;
#   * startup is an annualized count (1/a) vs the schema's bare "count".
# A NEW disagreement (or a stale entry here) fails the ratchet, forcing a
# reconciliation decision rather than silent drift.
KNOWN_SPINE_UNIT_DIFFERENCES = {
    ("connection", "invest_marginal"),
    ("connection__node__node", "flow_annualized"),
    ("connection__node__node", "flow_to_first_node_annualized"),
    ("connection__node__node", "flow_to_second_node_annualized"),
    ("group", "CO2_annualized"),
    ("group", "flow_annualized"),
    ("group", "flow_t"),
    ("group", "inertia_largest_flow_t"),
    ("group", "slack_capacity_margin"),
    ("group", "slack_nonsync_t"),
    ("group", "sum_flow_annualized"),
    ("node", "balance"),
    ("node", "balance_t"),
    ("node", "invest_marginal"),
    ("unit", "CO2_annualized"),
    ("unit", "invest_marginal"),
    ("unit", "startup_cumulative"),
    ("unit__node", "flow_annualized"),
}


def _norm(unit: str | None) -> str:
    u = (unit or "").lower().replace(" ", "").replace("·", "")
    u = u.replace("co2", "").replace("units", "count")
    for z in ("perunit", "ratio"):
        u = u.replace(z, "")
    return u


def test_spine_description_unit_drift_ratchet():
    schema = _load_schema()
    desc = {(e[0], e[1]): e[4] for e in schema["parameter_definitions"]}
    derived = write_spinedb.derived_param_unit_map()
    live = set()
    for key, du in derived.items():
        eu = write_spinedb.leading_unit(desc.get(key))
        if eu is not None and _norm(eu) != _norm(du):
            live.add(key)
    new = live - KNOWN_SPINE_UNIT_DIFFERENCES
    stale = KNOWN_SPINE_UNIT_DIFFERENCES - live
    assert not new, f"new Spine unit drift (reconcile or allowlist): {new}"
    assert not stale, f"stale allowlist entries (now consistent): {stale}"


# ── Plot: default y-axis label ──────────────────────────────────────────────

def test_default_ylabel_uses_unit():
    assert default_ylabel_for("annualized_costs_d_p") == "M CUR/a"
    assert default_ylabel_for("unit_outputNode_dt_ee") == "MW"


def test_default_ylabel_none_for_ratio_and_unknown():
    # a unitless ratio output -> no label
    assert result_key_summary("nodeGroup_VRE_share_d_g")[0] == "ratio"
    assert default_ylabel_for("nodeGroup_VRE_share_d_g") is None
    assert default_ylabel_for("not_an_output") is None
