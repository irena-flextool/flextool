"""Discover dispatch entity / scenario names and assign palette colors.

Reads dispatch mappings to enumerate the group / unit / connection / scenario
names present in a run, and assigns default palette colors.  These feed the
additive seeding of the project ``plot_settings.yaml`` (the durable colors
file the renderers read); the legacy ``config.yaml`` system has been removed.
"""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from flextool.scenario_comparison.constants import DEFAULT_SPECIAL_COLORS
from flextool.scenario_comparison.data_models import DispatchMappings


def discover_dispatch_entities(
    mappings: DispatchMappings,
    scenarios: list[str],
) -> dict[str, list[str]]:
    """Discover and classify dispatch entity / scenario names from mappings.

    Reads the dispatch-mapping fields and classifies each discovered name
    into the entity class the project ``plot_settings.yaml`` expects:

    * ``group``       — processGroup / unitGroup / connectionGroup aggregate
      names (``group_aggregate`` of ``processGroup_Unit_to_group`` /
      ``processGroup_Group_to_unit`` / ``processGroup_Connection``, plus the
      dispatch ``group`` names);
    * ``unit``        — individual (not-aggregated) unit names, taken as the
      bare ``unit`` / ``process`` entity of the unit member and
      ``not_in_aggregate`` unit fields;
    * ``connection``  — individual (not-aggregated) connection names, taken
      as the bare ``process`` / ``connection`` entity of the connection
      member and ``not_in_aggregate`` connection fields;
    * ``scenarios``   — the run's scenario names.

    The *bare* entity name is recorded (the unit / connection name), matching
    what the dispatch color resolver looks up — never the ``(process, node)``
    composite string.  Nodes are deliberately not discovered (node colors are
    dataset-coupled and are not seeded into the portable settings file).

    Returns a mapping ``{class -> sorted list of names}`` with the keys
    ``group``, ``unit``, ``connection`` and ``scenarios``; empty lists when a
    class has no members.
    """
    groups: set[str] = set()
    units: set[str] = set()
    connections: set[str] = set()

    # --- Group aggregate names (processGroups / unit- and connectionGroups) ---
    dispatch_groups_df = mappings.dispatch_groups
    if dispatch_groups_df is not None and not dispatch_groups_df.empty:
        if 'group' in dispatch_groups_df.columns:
            groups.update(str(g) for g in dispatch_groups_df['group'].unique())

    for pg_attr in (
        'processGroup_Unit_to_group',
        'processGroup_Group_to_unit',
        'processGroup_Connection',
    ):
        pg_df = getattr(mappings, pg_attr, None)
        if pg_df is not None and not pg_df.empty and 'group_aggregate' in pg_df.columns:
            groups.update(str(g) for g in pg_df['group_aggregate'].unique())

    # --- Individual unit names (bare entity, not the composite) ---
    # processGroup_*_members carry the units/connections that DO aggregate but
    # are still useful per-entity colors when shown individually; the
    # not_in_aggregate_* fields carry the units/connections shown un-aggregated.
    for unit_attr in (
        'processGroup_unit_to_node_members',
        'processGroup_node_to_unit_members',
        'not_in_aggregate_unit_to_node',
        'not_in_aggregate_node_to_unit',
    ):
        u_df = getattr(mappings, unit_attr, None)
        if u_df is None or u_df.empty:
            continue
        col = 'unit' if 'unit' in u_df.columns else (
            'process' if 'process' in u_df.columns else None
        )
        if col is not None:
            units.update(str(u) for u in u_df[col].unique())

    # --- Individual connection names (bare entity) ---
    for conn_attr in (
        'processGroup_connection_to_node_members',
        'processGroup_node_to_connection_members',
        'not_in_aggregate_connection_to_node',
        'not_in_aggregate_node_to_connection',
    ):
        c_df = getattr(mappings, conn_attr, None)
        if c_df is None or c_df.empty:
            continue
        col = 'connection' if 'connection' in c_df.columns else (
            'process' if 'process' in c_df.columns else None
        )
        if col is not None:
            connections.update(str(c) for c in c_df[col].unique())

    na_conn_df = mappings.not_in_aggregate_connection
    if na_conn_df is not None and not na_conn_df.empty:
        col = 'connection' if 'connection' in na_conn_df.columns else (
            'process' if 'process' in na_conn_df.columns else None
        )
        if col is not None:
            connections.update(str(c) for c in na_conn_df[col].unique())

    # A name discovered as both a group aggregate and an individual entity is
    # a group (the aggregate is the durable, project-portable color).
    units -= groups
    connections -= groups
    # Guard against a name landing in both individual classes.
    connections -= units

    scenario_names = [str(s) for s in scenarios if s]

    return {
        'group': sorted(groups),
        'unit': sorted(units),
        'connection': sorted(connections),
        'scenarios': sorted(dict.fromkeys(scenario_names)),
    }


def assign_palette_colors(
    names: list[str],
    start_index: int = 0,
    use_tab10: bool = False,
) -> dict[str, str]:
    """Assign default palette colors to *names*, in order.

    Special-token names get their fixed ``DEFAULT_SPECIAL_COLORS`` value;
    everything else cycles the tab20 (or tab10 for scenarios) matplotlib
    palette starting at *start_index* so the assigned colors are stable and
    visually sensible.

    Returns an ordered ``{name -> '#RRGGBB'}`` mapping for *names*.
    """
    palette = plt.cm.tab10(np.linspace(0, 1, 10)) if use_tab10 else \
        plt.cm.tab20(np.linspace(0, 1, 20))
    span = 10 if use_tab10 else 20
    out: dict[str, str] = {}
    idx = start_index
    for name in names:
        if name in DEFAULT_SPECIAL_COLORS:
            out[name] = DEFAULT_SPECIAL_COLORS[name]
            continue
        out[name] = matplotlib.colors.rgb2hex(palette[idx % span])
        idx += 1
    return out
