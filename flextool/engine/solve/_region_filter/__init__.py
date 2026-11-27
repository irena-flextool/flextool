"""Regional filter for Benders decomposition (gaps A4 + A5).

Slices a whole-system :class:`FlexData` into N per-region :class:`FlexData`s.
Cross-region arcs (rows in ``process_source_sink`` whose ``source`` and
``sink`` straddle two regions) are *severed* into two virtual one-way
half-flow arcs, one in each region.

The virtual entities are pure bookkeeping — the half-flow on the export
side and the half-flow on the import side carry the **same flow** at
optimality.  In the Benders scheme that coupling is enforced by the
coordinating master (:mod:`flextool.engine.solve._benders`): the master
holds the inter-regional trade flows and pins each region's forward
cross-region half-flow to its chosen value before solving the region as a
subproblem.  Inside a region's standalone LP, the half-flow column is just
an ordinary ``v_flow`` column with:

* ``p_unitsize = 1`` (same units as the original column);
* ``p_flow_upper`` = original pipe capacity (so dispatch can push flow
  at full pipe capacity; uncapped when the master may pin a positive
  greenfield trade);
* the in-region terminal node enters ``flow_to_n`` (export) or
  ``flow_from_n`` (import) so the half-flow contributes to its
  nodeBalance;
* the *virtual* node sits OUTSIDE ``nodeBalance`` so the half-flow's
  other terminal is free (no balance pin, no penalty).

This module is a no-op when the input :class:`FlexData` has no
``decomposition_method=benders_regional`` group params.

Public surface
--------------
* :class:`HalfFlow`        — one severed arc; pairs across regions.
* :class:`RegionSplit`     — output of :func:`split` for one region.
* :func:`discover_regions` — returns ``[region_name, ...]`` from a
  whole-system FlexData (reads ``p_group_decomposition.csv`` indirectly
  via the populated ``group_entity`` / ``group_node`` frames).
* :func:`split`            — produces ``list[RegionSplit]``.

Modules
-------
* ``_regions``        — :class:`HalfFlow`, :class:`RegionSplit`, decomposition method, region membership,
  master-hosted nodes; the module logger.
* ``_frames``         — row filters of frames / Params by an entity set.
* ``_arcs``           — arc classification, half-flows, master-local processes, the units the master holds.
* ``_master_side``    — the master / region side of user constraints, feature groups, limits and CO2.
* ``_checks``         — the split's hard errors and the integer-region warning.
* ``_region_data``    — one region's :class:`FlexData` and its injected half-flows.
* ``_master_network`` — :func:`master_network_data`, the Benders master's reduced :class:`FlexData`.
* ``_split``          — :func:`split`.
"""
from __future__ import annotations

from flextool.engine.solve._region_filter._split import split
from flextool.engine.solve._region_filter._master_network import master_network_data
from flextool.engine.solve._region_filter._arcs import master_units, master_arc_counts
from flextool.engine.solve._region_filter._regions import (
    HalfFlow,
    RegionSplit,
    load_decomposition_method,
    discover_regions,
    load_region_membership,
    process_region_by_nodes as process_region_by_nodes,
    compute_master_hosted_nodes,
)

__all__ = [
    "HalfFlow",
    "RegionSplit",
    "compute_master_hosted_nodes",
    "master_arc_counts",
    "master_units",
    "discover_regions",
    "split",
    "master_network_data",
    "load_decomposition_method",
    "load_region_membership",
]
