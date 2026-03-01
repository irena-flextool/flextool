"""Typed dataclasses replacing the raw dicts ``combined_dfs`` and ``combined_mapping_dfs``.

TimeSeriesResults — combined time-series result DataFrames (scenario in column MultiIndex).
DispatchMappings — dispatch mapping DataFrames combined across scenarios (scenario in row index).
"""

from __future__ import annotations

from dataclasses import dataclass, fields

import pandas as pd


@dataclass
class TimeSeriesResults:
    """Combined time-series result DataFrames (scenario in column MultiIndex).

    To add a new FlexTool output variable:
      1. Add a field below with type ``pd.DataFrame | None = None``
      2. The field name must exactly match the parquet filename (without .parquet)
      3. from_dict() will populate it automatically; a warning flags unknown files
    """

    unit_outputNode_dt_ee: pd.DataFrame | None = None
    unit_inputNode_dt_ee: pd.DataFrame | None = None
    connection_leftward_dt_eee: pd.DataFrame | None = None
    connection_rightward_dt_eee: pd.DataFrame | None = None
    connection_losses_dt_eee: pd.DataFrame | None = None
    node_slack_up_dt_e: pd.DataFrame | None = None
    node_d_ep: pd.DataFrame | None = None
    node_inflow__dt: pd.DataFrame | None = None
    nodeGroup_flows_d_gpe: pd.DataFrame | None = None
    nodeGroup_gd_p: pd.DataFrame | None = None
    unit_curtailment_outputNode_dt_ee: pd.DataFrame | None = None

    def to_dict(self) -> dict[str, pd.DataFrame]:
        """All non-None DataFrames as a dict, for plot_dict_of_dataframes()."""
        return {
            f.name: getattr(self, f.name)
            for f in fields(self)
            if getattr(self, f.name) is not None
        }

    @classmethod
    def from_dict(cls, d: dict[str, pd.DataFrame]) -> TimeSeriesResults:
        """Build from the raw dict produced by combine_parquet_files().

        Warns about unknown keys (signals a field needs to be added).
        """
        known = {f.name for f in fields(cls)}
        unknown = set(d.keys()) - known
        if unknown:
            print(
                f"Warning: Result variables not registered in TimeSeriesResults "
                f"(add fields to data_models.py): {sorted(unknown)}"
            )
        return cls(**{k: v for k, v in d.items() if k in known})


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
