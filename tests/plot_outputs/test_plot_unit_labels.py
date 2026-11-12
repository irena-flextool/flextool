"""Plot axis labels follow the nodes' ``unit_of_measure`` (unit_of_measure WP3, ``plot_outputs.unit_labels``).

* The power/energy labels of ``default_plots.yaml`` are unit templates; filled for MWh they give exactly the
  labels of 4.2 (the snapshot below was taken from the literal labels before the change).
* A label has a unit token exactly when its table follows its nodes (``UNIT_SOURCE``, not a power feature of
  ``DEFAULT_UNITS_ONLY``): reserves, inertia, non-synchronous share and capacity margin stay in MW (critique M1).
* With a node in m3, a subplot per node gets that node's unit, a stack over nodes of different units gets
  "Mixed units of measure!", and the direct (PNG) path and the plan (viewer) path agree.
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402
import yaml  # noqa: E402

from flextool import units_of_measure as uom  # noqa: E402
from flextool._resources import package_data_path  # noqa: E402
from flextool.plot_outputs.config import PlotConfig, default_axis_label_for, default_ylabel_for  # noqa: E402
from flextool.plot_outputs.orchestrator import prepare_plot_data  # noqa: E402
from flextool.plot_outputs.plan import compute_live_plan, load_plot_plan, save_plot_plan  # noqa: E402
from flextool.plot_outputs.unit_labels import fill_default, merged_context  # noqa: E402
from flextool.process_outputs._output_meta import DEFAULT_UNITS_ONLY, UNIT_SOURCE, UnitContext  # noqa: E402

MIXED = "Mixed units of measure!"

#: The literal labels of 4.2 that are unit templates now: (table, sub-config, "" or "comparison", axis) -> label.
OLD_LABELS = {
    ('connection_capacity_ed_p', 'default', '', 'xlabel'): 'MW',
    ('connection_d_eee', 'default', '', 'xlabel'): 'MWh (annualized)',
    ('connection_d_eee', 'sum_periods', '', 'xlabel'): 'MWh (horizon total)',
    ('connection_dt_eee', 'chunks', '', 'ylabel'): 'MWh/h',
    ('connection_dt_eee', 'default', '', 'ylabel'): 'MWh/h',
    ('connection_dt_eee', 'time_plots', '', 'ylabel'): 'MWh/h',
    ('connection_leftward_d_eee', 'debug', '', 'xlabel'): 'MWh (annualized)',
    ('connection_leftward_d_eee', 'sum_periods', '', 'xlabel'): 'MWh (horizon total)',
    ('connection_leftward_dt_eee', 'chunks', '', 'ylabel'): 'MWh/h',
    ('connection_leftward_dt_eee', 'debug', '', 'ylabel'): 'MWh/h',
    ('connection_leftward_dt_eee', 'time_plots', '', 'ylabel'): 'MWh/h',
    ('connection_losses_d_eee', 'default', '', 'xlabel'): 'MWh (annualized)',
    ('connection_losses_d_eee', 'sum_periods', '', 'xlabel'): 'MWh (horizon total)',
    ('connection_losses_d_eee', 'sum_periods', 'comparison', 'ylabel'): 'MWh (horizon total)',
    ('connection_losses_dt_eee', 'chunks', '', 'ylabel'): 'MWh/h',
    ('connection_losses_dt_eee', 'chunks', 'comparison', 'ylabel'): 'MWh',
    ('connection_losses_dt_eee', 'default', '', 'ylabel'): 'MWh/h',
    ('connection_rightward_d_eee', 'debug', '', 'xlabel'): 'MWh (annualized)',
    ('connection_rightward_d_eee', 'sum_periods', '', 'xlabel'): 'MWh (horizon total)',
    ('connection_rightward_dt_eee', 'chunks', '', 'ylabel'): 'MWh/h',
    ('connection_rightward_dt_eee', 'debug', '', 'ylabel'): 'MWh/h',
    ('connection_rightward_dt_eee', 'time_plots', '', 'ylabel'): 'MWh/h',
    ('dual_invest_effective_connection_d_e', 'default', '', 'xlabel'): 'CUR/kW  (+ = more investment here would lower total cost)',
    ('dual_invest_effective_node_d_e', 'default', '', 'xlabel'): 'CUR/kWh  (+ = more investment here would lower total cost)',
    ('dual_invest_effective_unit_d_e', 'default', '', 'xlabel'): 'CUR/kW  (+ = more investment here would lower total cost)',
    ('entity_annuity_d_p', 'debug', '', 'xlabel'): 'CUR/MW/a',
    ('nodeGroup_flows_d_g', 'default', '', 'ylabel'): 'MWh (annualized)',
    ('nodeGroup_flows_d_g', 'sum_periods', '', 'ylabel'): 'MWh (horizon total)',
    ('nodeGroup_flows_d_g', 'sum_periods', 'comparison', 'ylabel'): 'MWh (horizon total)',
    ('nodeGroup_flows_d_gpe', 'default', '', 'ylabel'): 'MWh (annualized)',
    ('nodeGroup_flows_d_gpe', 'sum_periods', '', 'ylabel'): 'MWh (horizon total)',
    ('nodeGroup_flows_dt_g', 'chunks', '', 'ylabel'): 'MWh/h',
    ('nodeGroup_flows_dt_g', 'default', '', 'ylabel'): 'MWh/h',
    ('nodeGroup_flows_dt_gpe', 'chunks', '', 'ylabel'): 'MWh',
    ('nodeGroup_flows_dt_gpe', 'chunks', 'comparison', 'ylabel'): 'MWh',
    ('nodeGroup_flows_dt_gpe', 'default', '', 'ylabel'): 'MWh',
    ('nodeGroup_flows_dt_gpe', 'default', 'comparison', 'ylabel'): 'MWh',
    ('nodeGroup_g_p', 'sum_periods', '', 'xlabel'): ['percent (horizon total, energy ratio)', 'percent (horizon total, energy ratio)', 'percent (horizon total, energy ratio)', 'percent (horizon total, energy ratio)', 'MWh (horizon total)', 'percent (horizon total, energy ratio)'],
    ('nodeGroup_gd_p', 'default', '', 'xlabel'): ['percent', 'percent', 'percent', 'percent', 'MWh (annualized)', 'percent'],
    ('nodeGroup_gdt_p', 'chunks', '', 'ylabel'): ['MWh', 'MWh', 'MWh', 'MWh', 'MWh', 'percent', 'MWh', 'percent'],
    ('nodeGroup_gdt_p', 'debug', '', 'ylabel'): ['MWh', 'MWh', 'MWh', 'MWh', 'MWh', 'percent', 'MWh', 'percent'],
    ('node_capacity_ed_p', 'default', '', 'xlabel'): 'MWh',
    ('node_d_ep', 'default', '', 'ylabel'): 'MWh (annualized)',
    ('node_d_ep', 'default', 'comparison', 'xlabel'): 'MWh (annualized)',
    ('node_d_ep', 'sum_periods', '', 'ylabel'): 'MWh (horizon total)',
    ('node_d_ep', 'sum_periods', 'comparison', 'xlabel'): 'MWh (horizon total)',
    ('node_dt_ep', 'chunks', '', 'ylabel'): 'MWh/h',
    ('node_dt_ep', 'chunks', 'comparison', 'ylabel'): 'MWh',
    ('node_dt_ep', 'default', '', 'ylabel'): 'MWh/h',
    ('node_dt_ep', 'default', 'comparison', 'ylabel'): 'MWh',
    ('node_prices_dt_e', 'chunks', '', 'ylabel'): 'Currency/MWh',
    ('node_prices_dt_e', 'chunks', 'comparison', 'ylabel'): 'Currency / MWh',
    ('node_prices_dt_e', 'default', '', 'ylabel'): 'Currency/MWh',
    ('node_prices_dt_e', 'default', 'comparison', 'ylabel'): 'Currency / MWh',
    ('node_slack_down_d_e', 'default', '', 'xlabel'): 'MWh (annualized)',
    ('node_slack_down_d_e', 'sum_periods', '', 'xlabel'): 'MWh (horizon total)',
    ('node_slack_down_dt_e', 'chunks', '', 'xlabel'): 'MWh',
    ('node_slack_down_dt_e', 'chunks', 'comparison', 'ylabel'): 'MWh',
    ('node_slack_down_dt_e', 'default', '', 'xlabel'): 'MWh',
    ('node_slack_down_dt_e', 'default', 'comparison', 'ylabel'): 'MWh',
    ('node_slack_up_d_e', 'default', '', 'xlabel'): 'MWh (annualized)',
    ('node_slack_up_d_e', 'sum_periods', '', 'xlabel'): 'MWh (horizon total)',
    ('node_slack_up_dt_e', 'chunks', '', 'xlabel'): 'MWh',
    ('node_slack_up_dt_e', 'chunks', 'comparison', 'ylabel'): 'MWh',
    ('node_slack_up_dt_e', 'default', '', 'xlabel'): 'MWh',
    ('node_slack_up_dt_e', 'default', 'comparison', 'ylabel'): 'MWh',
    ('node_state_dt_e', 'chunks', '', 'ylabel'): 'MWh',
    ('node_state_dt_e', 'default', '', 'ylabel'): 'MWh',
    ('unit_VRE_potential_outputNode_d_ee', 'debug', '', 'xlabel'): 'MWh (annualized)',
    ('unit_VRE_potential_outputNode_dt_ee', 'chunks', '', 'ylabel'): 'MWh/h',
    ('unit_VRE_potential_outputNode_dt_ee', 'debug', '', 'ylabel'): 'MWh/h',
    ('unit_capacity_ed_p', 'default', '', 'xlabel'): 'MW',
    ('unit_curtailment_outputNode_d_ee', 'default', '', 'xlabel'): 'MWh (annualized)',
    ('unit_curtailment_outputNode_d_ee', 'sum_periods', '', 'xlabel'): 'MWh (horizon total)',
    ('unit_curtailment_outputNode_dt_ee', 'chunks', '', 'ylabel'): 'MWh/h',
    ('unit_curtailment_outputNode_dt_ee', 'chunks', 'comparison', 'ylabel'): 'MWh',
    ('unit_curtailment_outputNode_dt_ee', 'debug', '', 'ylabel'): 'MWh',
    ('unit_curtailment_outputNode_dt_ee', 'default', '', 'ylabel'): 'MWh/h',
    ('unit_inputNode_d_ee', 'default', '', 'xlabel'): 'MWh (annualized)',
    ('unit_inputNode_d_ee', 'sum_periods', '', 'xlabel'): 'MWh (horizon total)',
    ('unit_inputNode_dt_ee', 'chunks', '', 'ylabel'): 'MWh/h',
    ('unit_inputNode_dt_ee', 'default', '', 'ylabel'): 'MWh/h',
    ('unit_inputNode_dt_ee', 'default', 'comparison', 'ylabel'): 'MWh',
    ('unit_inputNode_dt_ee', 'time_plots', '', 'ylabel'): 'MWh/h',
    ('unit_outputNode_d_ee', 'default', '', 'xlabel'): 'MWh (annualized)',
    ('unit_outputNode_d_ee', 'sum_periods', '', 'xlabel'): 'MWh (horizon total)',
    ('unit_outputNode_dt_ee', 'chunks', '', 'ylabel'): 'MWh/h',
    ('unit_outputNode_dt_ee', 'chunks', 'comparison', 'ylabel'): 'MWh',
    ('unit_outputNode_dt_ee', 'default', '', 'ylabel'): 'MWh/h',
    ('unit_outputNode_dt_ee', 'default', 'comparison', 'ylabel'): 'MWh',
    ('unit_outputNode_dt_ee', 'time_plots', '', 'ylabel'): 'MWh/h',
    ('unit_ramp_inputs_dt_ee', 'chunks', '', 'ylabel'): 'MW/h',
    ('unit_ramp_inputs_dt_ee', 'debug', '', 'ylabel'): 'MW/h',
    ('unit_ramp_outputs_dt_ee', 'chunks', '', 'ylabel'): 'MW/h',
    ('unit_ramp_outputs_dt_ee', 'debug', '', 'ylabel'): 'MW/h',
    ('unit_startup_energy_d_ee', 'default', '', 'xlabel'): 'MWh (annualized)',
    ('unit_startup_energy_dt_ee', 'default', '', 'ylabel'): 'MWh',
}


def _labels():
    """(table, sub-config, mode, axis) -> label of every config in default_plots.yaml."""
    with open(package_data_path("schemas/default_plots.yaml"), encoding="utf-8") as f:
        plots = yaml.safe_load(f)["plots"]
    out = {}
    for tables in plots.values():
        for table, variants in tables.items():
            if not isinstance(variants, dict) or table in ("group", "order"):
                continue
            for variant, cfg in variants.items():
                if not isinstance(cfg, dict):
                    continue
                for axis in ("xlabel", "ylabel"):
                    if axis in cfg:
                        out[(table, variant, "", axis)] = cfg[axis]
                    overrides = cfg.get("comparison_overrides") or {}
                    if axis in overrides:
                        out[(table, variant, "comparison", axis)] = overrides[axis]
    return out


def _has_tokens(label) -> bool:
    items = label if isinstance(label, list) else [label]
    return any(isinstance(x, str) and uom.has_tokens(x) for x in items)


def test_default_labels_read_as_before() -> None:
    """Filled for MWh nodes, every templated label is the literal label it replaced (byte-identical)."""
    labels = _labels()
    templated = {k: v for k, v in labels.items() if _has_tokens(v)}
    assert set(templated) == set(OLD_LABELS)
    for key, label in templated.items():
        assert fill_default(label) == OLD_LABELS[key], key


def test_tokens_exactly_on_tables_that_follow_their_nodes() -> None:
    """Critique M1: a label is a template iff its table follows its nodes; the power features stay literal."""
    by_table: dict[str, bool] = {}
    for (table, *_rest), label in _labels().items():
        by_table[table] = by_table.get(table, False) or _has_tokens(label)
    for table, templated in by_table.items():
        follows = table in UNIT_SOURCE and table not in DEFAULT_UNITS_ONLY
        assert templated == follows, table
    assert not any(by_table[t] for t in DEFAULT_UNITS_ONLY if t in by_table)


def test_default_axis_label_is_a_template() -> None:
    assert default_ylabel_for("unit_outputNode_dt_ee") == "MW"
    assert default_axis_label_for("unit_outputNode_dt_ee") == "{R}"
    assert default_axis_label_for("unit_outputNode_d_ee", "a") == "{A}"
    assert default_axis_label_for("process_reserve_upDown_node_dt_eppe") == "MW"
    assert default_axis_label_for("annualized_costs_d_p") == "M CUR/a"


# ─── a water node in m3 ──────────────────────────────────────────────────

CTX = UnitContext(node_units={"water": "m3"},
                  capacity_nodes={"desal": ("water",), "coal": ("elec",)},
                  connection_nodes={"pipe": ("elec", "water"), "line": ("elec", "east")},
                  group_nodes={"all": ("elec", "water"), "power": ("elec", "east")})


def _dt_frame(columns: list[tuple], names: list[str]) -> pd.DataFrame:
    rows = pd.MultiIndex.from_tuples([("p1", f"t{i:02d}") for i in range(4)], names=["period", "time"])
    cols = pd.MultiIndex.from_tuples(columns, names=names)
    return pd.DataFrame(np.arange(4 * len(columns), dtype=float).reshape(4, -1) + 1.0, index=rows, columns=cols)


def _figure_labels(figures) -> list[str]:
    out = []
    for _title, fig in figures:
        out.extend(ax.get_ylabel() for ax in fig.axes if ax.get_ylabel())
        plt.close(fig)
    return out


FLOWS = _dt_frame([("desal", "water"), ("coal", "elec")], ["unit", "node"])
#: unit_outputNode_dt_ee "default": units stacked, one subplot per node.
FLOWS_CFG = PlotConfig(map_dimensions_for_plots=["dt_ee", "tt_su"], ylabel="{A}/h", subplots_per_row=2)


def test_subplot_per_node_takes_the_nodes_unit() -> None:
    plan = compute_live_plan(FLOWS, FLOWS_CFG, "flows", result_key="unit_outputNode_dt_ee", unit_ctx=CTX)
    by_title = dict(zip([t for t, _s in plan.effective_plot_specs], plan.ylabel))
    assert by_title == {"elec": "MWh/h", "water": "m3/h"}
    figures, _n = prepare_plot_data(FLOWS, FLOWS_CFG, "flows", plot_rows=(0, 4),
                                    result_key="unit_outputNode_dt_ee", unit_ctx=CTX)
    assert sorted(_figure_labels(figures)) == ["MWh/h", "m3/h"]


def test_default_context_fills_mwh() -> None:
    for ctx in (None, UnitContext(), {"A": UnitContext()}):
        plan = compute_live_plan(FLOWS, FLOWS_CFG, "flows", result_key="unit_outputNode_dt_ee", unit_ctx=ctx)
        assert plan.ylabel == "MWh/h" and plan.batch_ylabels is None


def test_stack_over_nodes_of_different_units_is_mixed() -> None:
    """A sum or stack over nodes of different units has no unit; lines over them list each unit."""
    chp = _dt_frame([("chp", "water"), ("chp", "elec")], ["unit", "node"])
    stacked = PlotConfig(map_dimensions_for_plots=["dt_ee", "tt_us"], ylabel="{A}/h")
    plan = compute_live_plan(chp, stacked, "flows", result_key="unit_outputNode_dt_ee", unit_ctx=CTX)
    assert plan.ylabel == MIXED
    # One subplot per unit, nodes stacked: each unit's stack is over one node here.
    plan = compute_live_plan(FLOWS, stacked, "flows", result_key="unit_outputNode_dt_ee", unit_ctx=CTX)
    assert dict(zip([t for t, _s in plan.effective_plot_specs], plan.ylabel)) == {"coal": "MWh/h", "desal": "m3/h"}
    summed = PlotConfig(map_dimensions_for_plots=["dt_ee", "tt_mm"], ylabel="{A}/h")
    plan = compute_live_plan(FLOWS, summed, "flows", result_key="unit_outputNode_dt_ee", unit_ctx=CTX)
    assert plan.ylabel == MIXED
    lines = PlotConfig(map_dimensions_for_plots=["dt_ee", "tt_ml"], ylabel="{A}/h")
    plan = compute_live_plan(FLOWS, lines, "flows", result_key="unit_outputNode_dt_ee", unit_ctx=CTX)
    assert plan.ylabel == "MWh/h; m3/h"


def test_group_and_connection_sums() -> None:
    """A nodeGroup over elec and water, and connections stacked across units: mixed; an all-MWh group: MWh/h."""
    groups = _dt_frame([("all",), ("power",)], ["group"])
    cfg = PlotConfig(map_dimensions_for_plots=["dt_g", "tt_u"], ylabel="{A}/h")
    plan = compute_live_plan(groups, cfg, "groups", result_key="nodeGroup_flows_dt_g", unit_ctx=CTX)
    assert dict(zip([t for t, _s in plan.effective_plot_specs], plan.ylabel)) == {"all": MIXED, "power": "MWh/h"}
    conns = _dt_frame([("line",), ("pipe",)], ["process"])
    cfg = PlotConfig(map_dimensions_for_plots=["dt_e", "tt_s"], ylabel="{A}/h")
    plan = compute_live_plan(conns, cfg, "conns", result_key="connection_dt_eee", unit_ctx=CTX)
    assert plan.ylabel == MIXED


def test_power_features_stay_mw() -> None:
    """A reserve table with an m3 node in the context keeps its literal label (M1)."""
    cfg = PlotConfig(map_dimensions_for_plots=["dt_ee", "tt_su"], ylabel="MWh/h")
    plan = compute_live_plan(FLOWS, cfg, "x", result_key="process_reserve_upDown_node_dt_eppe", unit_ctx=CTX)
    assert plan.ylabel == "MWh/h"


def test_capacity_bars_by_row() -> None:
    """unit_capacity_ed_p: the unit is a row level (one bar each), so two units of different capacity units in one
    subplot list both."""
    rows = pd.MultiIndex.from_tuples([("desal", "p1"), ("coal", "p1")], names=["unit", "period"])
    df = pd.DataFrame({"total": [5.0, 7.0]}, index=rows)
    df.columns.name = "parameter"
    cfg = PlotConfig(map_dimensions_for_plots=["ed_p", "bu_g"], xlabel="{R}")
    plan = compute_live_plan(df, cfg, "cap", result_key="unit_capacity_ed_p", unit_ctx=CTX)
    assert plan.xlabel == "MW; m3/h"
    figures, _n = prepare_plot_data(df, cfg, "cap", result_key="unit_capacity_ed_p", unit_ctx=CTX)
    texts = [ax.get_xlabel() for _t, fig in figures for ax in fig.axes if ax.get_xlabel()]
    for _t, fig in figures:
        plt.close(fig)
    assert texts == ["MW; m3/h"]


def test_file_batches_keep_their_labels(tmp_path) -> None:
    """One subplot per file: each file has its own label, kept through save / load of the plan."""
    cfg = PlotConfig(map_dimensions_for_plots=["dt_ee", "tt_su"], ylabel="{A}/h", max_subplots_per_file=1)
    plan = compute_live_plan(FLOWS, cfg, "flows", result_key="unit_outputNode_dt_ee", unit_ctx=CTX)
    assert plan.batch_ylabels == ["MWh/h", "m3/h"]
    save_plot_plan(plan, tmp_path, "unit_outputNode_dt_ee", "default")
    loaded = load_plot_plan(tmp_path, "unit_outputNode_dt_ee", "default")
    assert loaded.batch_ylabels == plan.batch_ylabels
    from flextool.plot_outputs.plan import build_figure_from_plan
    fig = build_figure_from_plan(loaded, 1, (0, 4))
    assert [ax.get_ylabel() for ax in fig.axes if ax.get_ylabel()] == ["m3/h"]
    plt.close(fig)
    figures, _n = prepare_plot_data(FLOWS, cfg, "flows", plot_rows=(0, 4), result_key="unit_outputNode_dt_ee",
                                    unit_ctx=CTX)
    assert _figure_labels(figures) == ["MWh/h", "m3/h"]


def test_comparison_contexts_merge() -> None:
    """Two compared scenarios: a node whose unit differs between them reads as mixed on the shared axis."""
    merged = merged_context({"A": CTX, "B": UnitContext(node_units={"water": "t"})})
    assert merged.node_units["water"] == MIXED
    assert merged_context({"A": UnitContext(), "B": None}) is None
    assert merged_context({"A": CTX, "B": UnitContext()}).node_units == {"water": MIXED}


@pytest.mark.parametrize("label, expected", [
    (["{A}", "percent", "{A} (annualized)"], ["MWh", "percent", "MWh (annualized)"]),
    ("CUR/{kR}  (+ = more)", "CUR/kW  (+ = more)"), ("Node", "Node"), (None, None)])
def test_fill_default(label, expected) -> None:
    assert fill_default(label) == expected
