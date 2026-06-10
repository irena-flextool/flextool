"""Coverage + validity guardrails for derived output-column metadata.

The metadata is *derived* from a small transform catalog (see
``flextool/process_outputs/_output_meta.py``).  These tests keep the
derivation honest and ratchet coverage upward — a NEW processed output may
not land without either a transform declaration or an explicit allowlist
entry, and allowlist entries must be removed once the output is declared.
"""

import pandas as pd

from flextool.engine_polars._parquet_bundle import REGISTRY
from flextool.process_outputs._output_meta import (
    OUTPUT_TRANSFORM,
    FORMULA_OVERRIDE,
    DOCS_ANCHOR,
    Transform,
    Semantics,
    derive_column_meta,
)

# Processed outputs not yet given a transform declaration.  This list may only
# SHRINK: declaring one of these requires removing it here (enforced by
# ``test_allowlist_has_no_stale_entries``), and a new undeclared output trips
# ``test_no_new_undeclared_processed_outputs``.
UNDECLARED_ALLOWLIST = {
    'CO2__', 'co2_price_period_d_g', 'co2_price_total_d_g',
    'connection_capacity_ed_p', 'connection_dc_power_flow',
    'connection_leftward_d_eee', 'connection_leftward_dt_eee',
    'connection_losses_d_eee', 'connection_losses_dt_eee',
    'connection_rightward_d_eee', 'connection_rightward_dt_eee',
    'costs_discounted_p_', 'dc_angle_diff_dt_e', 'dc_angle_dt_e',
    'discountFactors_d_p', 'dual_invest_effective_connection_d_e',
    'dual_invest_effective_node_d_e', 'dual_invest_effective_unit_d_e',
    'entity_annuity_d_p', 'flowGroupIndicators', 'flowGroup_gd_p',
    'flowGroup_gd_t', 'group_node', 'group_process', 'group_process_node',
    'nodeGroupDispatch', 'nodeGroupIndicators', 'nodeGroup_VRE_share_d_g',
    'nodeGroup_VRE_share_dt_g', 'nodeGroup_flows_d_gpe', 'nodeGroup_flows_dt_g',
    'nodeGroup_flows_dt_gpe', 'nodeGroup_gd_p', 'nodeGroup_gdt_p',
    'nodeGroup_inertia_dt_g', 'nodeGroup_inertia_largest_flow_dt_g',
    'nodeGroup_slack_capacity_margin_d_g', 'nodeGroup_slack_inertia_d_g',
    'nodeGroup_slack_inertia_dt_g', 'nodeGroup_slack_nonsync_d_g',
    'nodeGroup_slack_nonsync_dt_g', 'nodeGroup_slack_reserve_d_eeg',
    'nodeGroup_slack_reserve_dt_eeg', 'nodeGroup_total_inflow',
    'nodeGroup_unit_node_inertia_dt_gee', 'node_capacity_ed_p', 'node_d_ep',
    'node_dc_power_flow', 'node_dt_ep', 'node_inflow__dt', 'node_prices_dt_e',
    'node_slack_down_d_e', 'node_slack_down_dt_e', 'node_slack_up_d_e',
    'node_slack_up_dt_e', 'node_state_dt_e', 'process_reserve_average_d_eppe',
    'process_reserve_upDown_node_dt_eppe', 'reserve_prices_dt_ppg',
    'unit_VRE_potential_outputNode_d_ee', 'unit_VRE_potential_outputNode_dt_ee',
    'unit_capacity_ed_p', 'unit_curtailment_outputNode_d_ee',
    'unit_curtailment_outputNode_dt_ee',
    'unit_curtailment_share_outputNode_d_ee',
    'unit_curtailment_share_outputNode_dt_ee', 'unit_online_average_d_e',
    'unit_online_dt_e', 'unit_ramp_inputs_dt_ee', 'unit_ramp_outputs_dt_ee',
    'years_represented__d',
}


def _processed_keys() -> set[str]:
    return {k for k, s in REGISTRY.items() if s.category == "processed"}


def test_declarations_are_valid():
    """Every OUTPUT_TRANSFORM value is a Transform with a valid Semantics."""
    for key, tf in OUTPUT_TRANSFORM.items():
        assert isinstance(tf, Transform), f"{key}: not a Transform"
        assert isinstance(tf.semantics, Semantics), f"{key}: bad semantics"
        assert tf.tooltip, f"{key}: empty tooltip"
        # Measures must carry a unit unless they are dimensionless ratios.
        assert tf.unit or tf.semantics is Semantics.RATIO, f"{key}: missing unit"


def test_overrides_reference_declared_outputs():
    for (key, _col) in FORMULA_OVERRIDE:
        assert key in OUTPUT_TRANSFORM, f"FORMULA_OVERRIDE references undeclared {key}"
    for key in DOCS_ANCHOR:
        assert key in OUTPUT_TRANSFORM, f"DOCS_ANCHOR references undeclared {key}"


def test_derive_marks_dimensions_and_measures():
    cols = pd.Index(
        ["period", "commodity_cost", "co2", "starts"], name="category")
    meta = derive_column_meta("annualized_costs_d_p", cols)
    assert meta is not None
    assert meta["period"].semantics is Semantics.DIMENSION
    assert meta["commodity_cost"].semantics is Semantics.ANNUALIZED
    assert meta["commodity_cost"].unit == "M CUR/a"
    # Per-column formula override is applied.
    assert "period_share" in meta["commodity_cost"].formula
    # No bogus override leaks onto other columns.
    assert meta["co2"].formula == ""


def test_derive_returns_none_for_undeclared():
    assert derive_column_meta("definitely_not_an_output", ["a", "b"]) is None


def test_no_new_undeclared_processed_outputs():
    """A new processed output must be declared or explicitly allowlisted."""
    undeclared = _processed_keys() - set(OUTPUT_TRANSFORM)
    new = undeclared - UNDECLARED_ALLOWLIST
    assert not new, (
        "New undeclared processed outputs — add a Transform to "
        f"OUTPUT_TRANSFORM (or allowlist): {sorted(new)}"
    )


def test_allowlist_has_no_stale_entries():
    """Ratchet: once an output is declared (or removed), drop it from the
    allowlist so coverage can only move forward."""
    undeclared = _processed_keys() - set(OUTPUT_TRANSFORM)
    stale = UNDECLARED_ALLOWLIST - undeclared
    assert not stale, (
        "Stale allowlist entries (now declared or no longer a processed "
        f"output) — remove them: {sorted(stale)}"
    )
