"""Typed dataclasses replacing the raw dicts ``combined_dfs`` and ``combined_mapping_dfs``.

TimeSeriesResults — combined time-series result DataFrames (scenario in column MultiIndex).
DispatchMappings — dispatch mapping DataFrames combined across scenarios (scenario in row index).
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields

import pandas as pd


# Parquet filenames that are dispatch mapping / structural files loaded separately
# by dispatch_mappings.py.  These are NOT time-series results and should be excluded
# from the TimeSeriesResults.from_dict() unknown-key warning.
_DISPATCH_MAPPING_FILENAMES: frozenset[str] = frozenset({
    # Loaded by dispatch_mappings.load_dispatch_mappings():
    'group_node',
    'group_process_node',
    'nodeGroupIndicators',   # → dispatch_groups (dispatch scoping for indicator groups)
    'flowGroupIndicators',
    'nodeGroupDispatch__processGroup_Unit_to_group',
    'nodeGroupDispatch__processGroup_Group_to_unit',
    'nodeGroupDispatch__processGroup_Connection',
    'nodeGroupDispatch__processGroup__process__unit__to_node',
    'nodeGroupDispatch__processGroup__process__node__to_unit',
    'nodeGroupDispatch__processGroup__process__connection__to_node',
    'nodeGroupDispatch__processGroup__process__node__to_connection',
    'nodeGroupDispatch__process__unit__to_node_Not_in_aggregate',
    'nodeGroupDispatch__process__node__to_unit_Not_in_aggregate',
    'nodeGroupDispatch__process__connection__to_node_Not_in_aggregate',
    'nodeGroupDispatch__process__node__to_connection_Not_in_aggregate',
    'nodeGroupDispatch__connection_Not_in_aggregate',
    'nodeGroupDispatch__process_fully_inside',
    # Other structural/metadata files (not time-series):
    'group_process',
    'nodeGroupDispatch',
    # DC power flow structural sets:
    'node_dc_power_flow',
    'connection_dc_power_flow',
})


@dataclass
class TimeSeriesResults:
    """Combined time-series result DataFrames (scenario in column MultiIndex).

    To add a new FlexTool output variable:
      1. Add a field below with type ``pd.DataFrame | None = None``
      2. The field name must exactly match the parquet filename (without .parquet)
      3. from_dict() will populate it automatically; a warning flags unknown files
         (tests/scenario_comparison/test_data_models_registry.py checks every processed parquet table of
         ``engine.output._parquet_bundle.REGISTRY`` has a field)
    """

    # --- Unit outputs ---
    unit_outputNode_dt_ee: pd.DataFrame | None = None
    unit_outputNode_d_ee: pd.DataFrame | None = None
    unit_inputNode_dt_ee: pd.DataFrame | None = None
    unit_inputNode_d_ee: pd.DataFrame | None = None
    unit_outputs_cf_d_ee: pd.DataFrame | None = None
    unit_inputs_cf_d_ee: pd.DataFrame | None = None
    unit_curtailment_outputNode_dt_ee: pd.DataFrame | None = None
    unit_curtailment_outputNode_d_ee: pd.DataFrame | None = None
    unit_curtailment_share_outputNode_dt_ee: pd.DataFrame | None = None
    unit_curtailment_share_outputNode_d_ee: pd.DataFrame | None = None
    unit_VRE_potential_outputNode_dt_ee: pd.DataFrame | None = None
    unit_VRE_potential_outputNode_d_ee: pd.DataFrame | None = None
    unit_ramp_outputs_dt_ee: pd.DataFrame | None = None
    unit_ramp_inputs_dt_ee: pd.DataFrame | None = None
    unit_online_dt_e: pd.DataFrame | None = None
    unit_online_average_d_e: pd.DataFrame | None = None
    unit_startup_d_e: pd.DataFrame | None = None
    unit_startup_energy_dt_ee: pd.DataFrame | None = None
    unit_startup_energy_d_ee: pd.DataFrame | None = None
    unit_capacity_ed_p: pd.DataFrame | None = None

    # --- Connection outputs ---
    connection_online_dt_e: pd.DataFrame | None = None
    connection_online_average_d_e: pd.DataFrame | None = None
    connection_startup_d_e: pd.DataFrame | None = None
    connection_leftward_dt_eee: pd.DataFrame | None = None
    connection_leftward_d_eee: pd.DataFrame | None = None
    connection_rightward_dt_eee: pd.DataFrame | None = None
    connection_rightward_d_eee: pd.DataFrame | None = None
    connection_losses_dt_eee: pd.DataFrame | None = None
    connection_losses_d_eee: pd.DataFrame | None = None
    connection_dt_eee: pd.DataFrame | None = None
    connection_d_eee: pd.DataFrame | None = None
    connection_cf_d_e: pd.DataFrame | None = None
    connection_capacity_ed_p: pd.DataFrame | None = None

    # --- Node outputs ---
    node_d_ep: pd.DataFrame | None = None
    node_dt_ep: pd.DataFrame | None = None
    node_inflow__dt: pd.DataFrame | None = None
    node_prices_dt_e: pd.DataFrame | None = None
    node_state_dt_e: pd.DataFrame | None = None
    node_slack_up_dt_e: pd.DataFrame | None = None
    node_slack_up_d_e: pd.DataFrame | None = None
    node_slack_down_dt_e: pd.DataFrame | None = None
    node_slack_down_d_e: pd.DataFrame | None = None
    node_capacity_ed_p: pd.DataFrame | None = None

    # --- NodeGroup outputs ---
    nodeGroup_flows_d_gpe: pd.DataFrame | None = None
    nodeGroup_flows_dt_gpe: pd.DataFrame | None = None
    nodeGroup_flows_d_g: pd.DataFrame | None = None
    nodeGroup_flows_dt_g: pd.DataFrame | None = None
    nodeGroup_gd_p: pd.DataFrame | None = None
    nodeGroup_gdt_p: pd.DataFrame | None = None
    nodeGroup_g_p: pd.DataFrame | None = None            # horizon totals (energy ratios)
    nodeGroup_VRE_share_d_g: pd.DataFrame | None = None
    nodeGroup_VRE_share_dt_g: pd.DataFrame | None = None
    nodeGroup_VRE_share_g: pd.DataFrame | None = None    # horizon total (energy ratio)
    nodeGroup_inertia_dt_g: pd.DataFrame | None = None
    nodeGroup_inertia_largest_flow_dt_g: pd.DataFrame | None = None
    nodeGroup_unit_node_inertia_dt_gee: pd.DataFrame | None = None
    nodeGroup_slack_capacity_margin_d_g: pd.DataFrame | None = None
    nodeGroup_capacity_margin_d_g: pd.DataFrame | None = None
    nodeGroup_capacity_margin_entity_d_g: pd.DataFrame | None = None
    nodeGroup_slack_inertia_d_g: pd.DataFrame | None = None
    nodeGroup_slack_inertia_dt_g: pd.DataFrame | None = None
    nodeGroup_slack_nonsync_d_g: pd.DataFrame | None = None
    nodeGroup_slack_nonsync_dt_g: pd.DataFrame | None = None
    nodeGroup_slack_reserve_d_eeg: pd.DataFrame | None = None
    nodeGroup_slack_reserve_dt_eeg: pd.DataFrame | None = None
    nodeGroup_reserve_dt_eegp: pd.DataFrame | None = None
    nodeGroup_reserve_d_eegp: pd.DataFrame | None = None
    nodeGroup_reserve_provider_dt_eegee: pd.DataFrame | None = None

    # --- Costs ---
    costs_dt_p: pd.DataFrame | None = None
    costs_discounted_d_p: pd.DataFrame | None = None
    costs_discounted_p_: pd.DataFrame | None = None
    annualized_costs_d_p: pd.DataFrame | None = None
    discountFactors_d_p: pd.DataFrame | None = None      # debug runs only
    entity_annuity_d_p: pd.DataFrame | None = None       # debug runs only

    # --- Per-entity cost break-down (unit/connection/node, period level) ---
    cost_unit_annualized_d_ec: pd.DataFrame | None = None
    cost_connection_annualized_d_ec: pd.DataFrame | None = None
    cost_node_annualized_d_ec: pd.DataFrame | None = None
    cost_unit_discounted_d_ec: pd.DataFrame | None = None
    cost_connection_discounted_d_ec: pd.DataFrame | None = None
    cost_node_discounted_d_ec: pd.DataFrame | None = None

    # --- CO2 and process outputs ---
    CO2__: pd.DataFrame | None = None
    CO2_d_g: pd.DataFrame | None = None
    co2_price_period_d_g: pd.DataFrame | None = None
    co2_price_total_d_g: pd.DataFrame | None = None
    process_co2_d_eee: pd.DataFrame | None = None
    process_reserve_average_d_eppe: pd.DataFrame | None = None
    process_reserve_upDown_node_dt_eppe: pd.DataFrame | None = None

    # --- DC power flow ---
    dc_angle_dt_e: pd.DataFrame | None = None
    dc_angle_diff_dt_e: pd.DataFrame | None = None

    # --- Other ---
    reserve_prices_dt_ppg: pd.DataFrame | None = None
    years_represented__d: pd.DataFrame | None = None
    period_share_of_year__d: pd.DataFrame | None = None
    dual_invest_effective_node_d_e: pd.DataFrame | None = None
    dual_invest_effective_unit_d_e: pd.DataFrame | None = None
    dual_invest_effective_connection_d_e: pd.DataFrame | None = None

    # --- Flow groups (v58) ---
    flowGroup_gd_p: pd.DataFrame | None = None
    flowGroup_gd_t: pd.DataFrame | None = None

    # --- Not a result: each scenario's own rows of each combined frame ---
    # {variable: {scenario: row index of that scenario's file, in file order}}.
    # The combined frames are the scenarios side by side, so their row index is
    # the union of the scenarios' steps in first-seen order; this keeps each
    # scenario's own steps and their order (see combine_parquet_files).
    scenario_rows: dict[str, dict[str, pd.Index]] = field(default_factory=dict, repr=False)

    def to_dict(self) -> dict[str, pd.DataFrame]:
        """All non-None DataFrames as a dict, for plot_dict_of_dataframes()."""
        return {
            f.name: getattr(self, f.name)
            for f in fields(self)
            if f.name != 'scenario_rows' and getattr(self, f.name) is not None
        }

    def own_rows(self, name: str, scenario: str) -> pd.Index | None:
        """The rows of *scenario* in the combined frame *name*, in its file order (None if not recorded)."""
        return self.scenario_rows.get(name, {}).get(scenario)

    @classmethod
    def from_dict(
        cls,
        d: dict[str, pd.DataFrame],
        scenario_rows: dict[str, dict[str, pd.Index]] | None = None,
    ) -> TimeSeriesResults:
        """Build from the raw dict produced by combine_parquet_files().

        *scenario_rows* is the ``row_order`` that combine_parquet_files() filled.
        Warns about unknown keys (signals a field needs to be added).
        Dispatch mapping files (loaded separately) are silently excluded.
        """
        known = {f.name for f in fields(cls)} - {'scenario_rows'}
        unknown = set(d.keys()) - known - _DISPATCH_MAPPING_FILENAMES
        if unknown:
            print(
                f"Warning: Result variables not registered in TimeSeriesResults "
                f"(add fields to data_models.py): {sorted(unknown)}"
            )
        return cls(
            **{k: v for k, v in d.items() if k in known},
            scenario_rows=dict(scenario_rows or {}),
        )


@dataclass
class DispatchMappings:
    """Dispatch mapping DataFrames combined across scenarios (scenario in row index).

    Use get_for_scenario(field_name, scenario) to extract a per-scenario slice.
    """

    dispatch_groups: pd.DataFrame | None = None
    group_node: pd.DataFrame | None = None
    group_process_node: pd.DataFrame | None = None
    processGroup_Unit_to_group: pd.DataFrame | None = None
    processGroup_Group_to_unit: pd.DataFrame | None = None
    processGroup_Connection: pd.DataFrame | None = None
    processGroup_unit_to_node_members: pd.DataFrame | None = None
    processGroup_node_to_unit_members: pd.DataFrame | None = None
    processGroup_connection_to_node_members: pd.DataFrame | None = None
    processGroup_node_to_connection_members: pd.DataFrame | None = None
    not_in_aggregate_unit_to_node: pd.DataFrame | None = None
    not_in_aggregate_node_to_unit: pd.DataFrame | None = None
    not_in_aggregate_connection_to_node: pd.DataFrame | None = None
    not_in_aggregate_node_to_connection: pd.DataFrame | None = None
    not_in_aggregate_connection: pd.DataFrame | None = None
    process_fully_inside: pd.DataFrame | None = None

    def get_for_scenario(self, field_name: str, scenario: str) -> pd.DataFrame | None:
        """Extract per-scenario slice from a mapping field.

        Handles xs() returning Series when only one row matches.
        Returns None if field is absent or scenario not found.
        """
        df = getattr(self, field_name, None)
        if df is None or df.empty:
            return None
        if df.index.name == 'scenario':
            if scenario not in df.index:
                return None
            result = df.xs(scenario)
            if isinstance(result, pd.Series):
                return result.to_frame().T
            return result
        return df
