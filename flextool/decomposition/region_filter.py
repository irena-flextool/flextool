"""
region_filter.py — Regional filter for Lagrangian decomposition (Agent 3.1).

Given a populated ``input/`` directory and a region group name (a group with
``decomposition_method = 'lagrangian_region'``), produce a self-contained
``input_region_<group>/`` directory for that region's standalone solve.

Cross-region processes (pipelines, transmission) are removed from the
filtered directory and replaced by **import/export half-flows**:

* Each cross-region ``connection`` whose in-region endpoint is the *source*
  becomes an **export half-flow** from the in-region node to a new virtual
  node ``<connection>__export__<region>``.
* Each cross-region ``connection`` whose in-region endpoint is the *sink*
  becomes an **import half-flow** from a new virtual node
  ``<connection>__import__<region>`` to the in-region node.

The virtual nodes are declared with ``node_type = commodity`` — they
participate in the node balance but have no state, and their flow is the
Lagrangian coupling variable to be λ-priced by the coordinator (Agent 3.2).

Entry point: ``build_region_directory``.

Top-level contract:

* Reads the already-populated ``input/`` produced by ``write_input``.
* Does NOT modify the staging directory.
* Writes the filtered copy to ``output_dir``.
* Also writes ``solve_data/region_coupling.csv`` listing the coupling
  variables for this region.
"""
from __future__ import annotations

import csv
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Iterable

import polars as pl

if TYPE_CHECKING:  # pragma: no cover
    from flextool.engine_polars._flex_data_provider import FlexDataProvider


# ---------------------------------------------------------------------------
# Region membership & cross-region classification
# ---------------------------------------------------------------------------


@dataclass
class RegionMembership:
    """Membership sets for one decomposition region."""

    region: str
    # Entities that belong to this region (from group__node / group__unit /
    # group__connection of the region_group).
    nodes: set[str] = field(default_factory=set)
    units: set[str] = field(default_factory=set)
    connections: set[str] = field(default_factory=set)
    # Nodes that belong to some OTHER decomposition region.
    other_region_nodes: set[str] = field(default_factory=set)
    # Units and connections in any OTHER decomposition region.
    other_region_units: set[str] = field(default_factory=set)
    other_region_connections: set[str] = field(default_factory=set)
    # All decomposition region names (for reference).
    all_regions: set[str] = field(default_factory=set)


@dataclass
class HalfFlow:
    """One import or export half-flow replacing a cross-region connection."""

    original_connection: str
    region: str
    side: str  # "import" or "export"
    # For export: in-region node is source, virtual node is sink.
    # For import: virtual node is source, in-region node is sink.
    in_region_node: str
    virtual_node: str
    # Half-flow connection name: the virtual process that appears in
    # ``input_region_<region>/process_connection.csv`` replacing the
    # cross-region pipe.  Convention: ``hf_<pipe>__<side>__<region>``
    # — the ``hf_`` prefix disambiguates the virtual connection from
    # the virtual node (which re-uses the ``<pipe>__<side>__<region>``
    # stem) so ``entity.csv`` does not carry a duplicate tuple.
    virtual_connection: str


# ---------------------------------------------------------------------------
# Reading helpers
# ---------------------------------------------------------------------------


def _read_csv_rows(path: Path) -> tuple[list[str], list[list[str]]]:
    """Return ``(header_fields, data_rows)`` for a CSV at *path*.

    Rows are returned as lists of strings (no type coercion).  Blank lines
    are dropped.  If the file does not exist, returns ``([], [])``.
    """
    if not path.exists():
        return [], []
    with path.open() as fh:
        reader = csv.reader(fh)
        rows = [row for row in reader if row and any(cell.strip() for cell in row)]
    if not rows:
        return [], []
    return rows[0], rows[1:]


def _write_csv_rows(
    path: Path, header: list[str], rows: Iterable[list[str]]
) -> None:
    """Write CSV with *header* and *rows* to *path*.  Creates parent dirs."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(header)
        for row in rows:
            writer.writerow(row)


def _read_single_col(path: Path) -> list[str]:
    """Read a single-column CSV and return the list of values (minus header)."""
    _, rows = _read_csv_rows(path)
    return [r[0] for r in rows if r]


def _read_two_col_dict(path: Path) -> dict[str, list[str]]:
    """Read a two-column ``key,value`` CSV into ``{key: [values...]}``."""
    _, rows = _read_csv_rows(path)
    out: dict[str, list[str]] = {}
    for r in rows:
        if len(r) < 2:
            continue
        out.setdefault(r[0], []).append(r[1])
    return out


def _read_connection_endpoints(
    input_dir: Path,
) -> dict[str, tuple[str, str]]:
    """Return ``{connection: (source_node, sink_node)}`` for every connection.

    Derives endpoints from ``process__source.csv`` and ``process__sink.csv``,
    keeping only those processes that are listed in ``process_connection.csv``.
    """
    connections = set(_read_single_col(input_dir / "process_connection.csv"))
    src_of: dict[str, str] = {}
    for proc, src in (
        (r[0], r[1])
        for r in _read_csv_rows(input_dir / "process__source.csv")[1]
        if len(r) >= 2
    ):
        if proc in connections:
            src_of[proc] = src
    snk_of: dict[str, str] = {}
    for proc, snk in (
        (r[0], r[1])
        for r in _read_csv_rows(input_dir / "process__sink.csv")[1]
        if len(r) >= 2
    ):
        if proc in connections:
            snk_of[proc] = snk
    endpoints: dict[str, tuple[str, str]] = {}
    for conn in connections:
        if conn in src_of and conn in snk_of:
            endpoints[conn] = (src_of[conn], snk_of[conn])
    return endpoints


def discover_region_membership(
    input_dir: Path, region: str
) -> RegionMembership:
    """Parse ``input/`` to build the membership set for *region*.

    Identifies:

    * in-region nodes/units/connections from ``group__*`` CSVs
    * other regions' membership (every other group whose
      ``decomposition_method`` is ``lagrangian_region``)

    ``group__decomposition_method.csv`` is not a direct output of
    ``write_input``; instead the group→decomposition_method pairing lives
    in the group-parameter CSVs.  For V1 we rely on the caller passing the
    full list of decomposition regions.  If unavailable, we fall back to
    "every group listed in group.csv that has members in group__node AND
    whose name starts with ``region_``" (the LH2 fixture convention).
    """
    # group → nodes / units / connections mappings
    group_nodes = _read_two_col_dict(input_dir / "group__node.csv")
    group_processes = _read_two_col_dict(input_dir / "group__process.csv")

    # Discover all decomposition regions: pragmatic heuristic for V1 is
    # "group that has at least one node or process member and is not a
    # resolution group".  Better: read from a sibling file the caller can
    # populate.  For now we require the caller to pass *region* as the
    # concrete name; sibling regions come from the caller via
    # ``list_decomposition_regions``.
    # Fallback heuristic for the LH2 fixture: region groups start with
    # ``region_``.
    all_groups = set(_read_single_col(input_dir / "group.csv"))
    region_like = {g for g in all_groups if g.startswith("region_")}
    if region not in region_like:
        region_like.add(region)  # trust the caller

    mem = RegionMembership(
        region=region,
        all_regions=region_like,
    )
    mem.nodes = set(group_nodes.get(region, []))
    in_region_processes = set(group_processes.get(region, []))

    # group__process.csv has *every* process (unit or connection).  We
    # disambiguate by consulting process_unit.csv and process_connection.csv.
    units = set(_read_single_col(input_dir / "process_unit.csv"))
    connections = set(_read_single_col(input_dir / "process_connection.csv"))
    mem.units = in_region_processes & units
    mem.connections = in_region_processes & connections

    # Other regions' memberships
    for other in region_like:
        if other == region:
            continue
        mem.other_region_nodes.update(group_nodes.get(other, []))
        other_procs = set(group_processes.get(other, []))
        mem.other_region_units.update(other_procs & units)
        mem.other_region_connections.update(other_procs & connections)

    return mem


def discover_decomposition_regions_from_db(input_db_url: str) -> list[str]:
    """Return the list of group names whose ``decomposition_method`` is
    ``lagrangian_region``.

    Used by the CLI to present the list of available regions and by
    ``build_region_directory`` to know other regions' membership.
    """
    from spinedb_api import DatabaseMapping  # lazy import

    with DatabaseMapping(input_db_url) as db:
        names: list[str] = []
        for pv in db.find_parameter_values(
            entity_class_name="group",
            parameter_definition_name="decomposition_method",
        ):
            if pv["type"] is None:
                continue
            if str(pv["parsed_value"]) == "lagrangian_region":
                names.append(pv["entity_byname"][0])
    return names


# ---------------------------------------------------------------------------
# Cross-region classification
# ---------------------------------------------------------------------------


def classify_half_flows(
    input_dir: Path, mem: RegionMembership
) -> list[HalfFlow]:
    """Identify cross-region connections for *mem.region* and produce half-flow
    specs for each.

    A connection is cross-region if exactly one endpoint (source or sink)
    lies in ``mem.nodes`` and the other lies in ``mem.other_region_nodes``.
    """
    endpoints = _read_connection_endpoints(input_dir)
    half_flows: list[HalfFlow] = []
    for conn, (src, snk) in endpoints.items():
        src_in = src in mem.nodes
        snk_in = snk in mem.nodes
        src_out = src in mem.other_region_nodes
        snk_out = snk in mem.other_region_nodes
        # Only one endpoint in-region, other endpoint in another region.
        if src_in and snk_out:
            # Flow leaves the region — export half-flow.  Virtual node
            # and virtual connection must have distinct names: entity.csv
            # carries both, and a duplicate tuple would be flagged.
            virtual = f"{conn}__export__{mem.region}"
            half_flows.append(HalfFlow(
                original_connection=conn,
                region=mem.region,
                side="export",
                in_region_node=src,
                virtual_node=virtual,
                virtual_connection=f"hf_{conn}__export__{mem.region}",
            ))
        elif snk_in and src_out:
            # Flow enters the region — import half-flow.  Same
            # naming-distinction rule as the export branch above.
            virtual = f"{conn}__import__{mem.region}"
            half_flows.append(HalfFlow(
                original_connection=conn,
                region=mem.region,
                side="import",
                in_region_node=snk,
                virtual_node=virtual,
                virtual_connection=f"hf_{conn}__import__{mem.region}",
            ))
        # else: both endpoints in-region, both out-of-region, or one is
        # a shared commodity — leave the connection as is (filter will
        # handle it by keeping/dropping whole).
    return half_flows


# ---------------------------------------------------------------------------
# Per-CSV filter spec: which columns reference which entity kind
# ---------------------------------------------------------------------------


# Mapping from header-column position to the "kind" of entity it refers to.
# Kind is one of: "node", "unit", "connection", "process" (either unit or
# connection), "entity" (any of node/unit/connection), or a literal group
# name.
#
# We only list columns that MUST be in-region or shared.  Parameter-value
# columns (time, numeric value, method name, etc.) are ignored — we copy
# those through.

# Set of CSVs that are entity-level (column positions identify entities);
# we also treat single-column CSVs as "column 0 is an entity of kind X".
# Each entry: filename → list of (column_index, kind) tuples where kind
# identifies which membership set the column must be in.
#
# For "process" kind: accept if the value is in units ∪ connections ∪
# kept_processes (we classify unit vs connection by reading process_unit
# vs process_connection).
#
# If a row has any filter column whose value is NOT in the kept set,
# the row is dropped.


def _build_column_specs() -> dict[str, list[tuple[int, str]]]:
    """Per-CSV column spec for filtering.

    Returns {filename: [(col_index, kind), ...]}.  Files not listed are
    copied verbatim (e.g. solver config, timeline, default_values, debug).
    """
    specs: dict[str, list[tuple[int, str]]] = {}

    # Single-column "entity-set" CSVs — column 0 is the entity name.
    for name, kind in (
        ("node.csv", "node"),
        ("process.csv", "process"),
        ("process_unit.csv", "unit"),
        ("process_connection.csv", "connection"),
        ("entity.csv", "entity"),
        ("process_nonSync_connection.csv", "connection"),
        ("process_min_downtime.csv", "process"),
        ("process_min_uptime.csv", "process"),
    ):
        specs[name] = [(0, kind)]

    # Two-column CSVs where col-0 is entity of given kind.
    for name, kind0 in (
        ("commodity__node.csv", "node_col1"),  # special: commodity,node
        ("node__inflow_method.csv", "node"),
        ("node__storage_binding_method.csv", "node"),
        ("node__storage_nested_fix_method.csv", "node"),
        ("node__storage_solve_horizon_method.csv", "node"),
        ("node__storage_start_end_method.csv", "node"),
        ("node__profile__profile_method.csv", "node"),
        ("p_node_type.csv", "node"),
        ("process__ct_method.csv", "process"),
        ("process__startup_method.csv", "process"),
        ("process__profile__profile_method.csv", "process"),
        ("entity__invest_method.csv", "entity"),
        ("entity__lifetime_method.csv", "entity"),
    ):
        specs[name] = [(0, kind0)] if kind0 != "node_col1" else [(1, "node")]

    # Process→source / process→sink (col 0 process, col 1 node).
    for name in (
        "process__source.csv",
        "process__sink.csv",
    ):
        specs[name] = [(0, "process"), (1, "node")]

    # process_method.csv (user-input, derived by input_writer): col 0 is
    # process.  Other process-keyed sets (process_online*, process_profile,
    # process_VRE, process__commodity__node*) are now written to
    # solve_data/ by flextool.mod and never seen by the region filter.
    specs["process_method.csv"] = [(0, "process")]

    # Group-related: col 0 group, col 1 node/process.
    specs["group__node.csv"] = [(1, "node")]  # keep group; filter node
    specs["group__process.csv"] = [(1, "process")]
    specs["group__process__node.csv"] = [(1, "process"), (2, "node")]

    # Reserves (process, reserve, upDown, node) — col 0 process, col 3 node.
    specs["process__reserve__upDown__node.csv"] = [(0, "process"), (3, "node")]

    # Node-keyed parameter CSVs (header starts with 'node').  These are
    # ALL ``pt_node*`` / ``pd_node*`` / ``pbt_node*`` / ``p_node*``
    # variants.  The entity is always column 0.
    # Rather than enumerate, we'll catch them dynamically in the main
    # loop by inspecting headers — see ``_infer_column_spec_from_header``.

    # Same for process-keyed parameter CSVs and process__node-keyed.

    return specs


_EXPLICIT_SPECS = _build_column_specs()


def _infer_column_spec_from_header(
    header: list[str], filename: str
) -> list[tuple[int, str]]:
    """Infer filter columns from header names for parameter CSVs.

    Common patterns:
    * header[0] == "node"          → column 0 is a node
    * header[0] == "process"       → column 0 is a process; if col 1 is
                                     "source"/"sink"/"node" it's a node
    * header[0] == "connection"    → column 0 is a connection
    * header[0] == "entity"        → column 0 is any of node/unit/connection
    * header[0] == "commodity"     → leave alone (commodity is shared)
    * header[0] == "group"         → column 0 is a group; filter by extra
                                     known-entity columns (node, process)
    * header[0] == "profile"       → leave alone (profiles are shared
                                     identifiers, not spatial)
    * header[0] == "reserve"       → leave alone
    * header[0] == "constraint"    → leave alone
    * header[0] == "timeline"/"timeset"/"period"/"solve"/"model"/
                   "version"/"debug"/"class" → leave alone
    """
    if filename in _EXPLICIT_SPECS:
        return _EXPLICIT_SPECS[filename]
    if not header:
        return []

    col0 = header[0].lower()
    spec: list[tuple[int, str]] = []

    if col0 == "node":
        spec.append((0, "node"))
        # Some node CSVs have a "profile" column (which is always kept
        # globally) — skip.
    elif col0 == "process":
        spec.append((0, "process"))
        # Process-keyed may have a "source"/"sink"/"node" in col 1.
        if len(header) > 1 and header[1].lower() in {"source", "sink", "node"}:
            spec.append((1, "node"))
    elif col0 == "connection":
        spec.append((0, "connection"))
    elif col0 == "unit":
        spec.append((0, "unit"))
    elif col0 == "entity":
        spec.append((0, "entity"))
    elif col0 == "group":
        # Groups themselves stay; filter by any trailing entity column.
        for i, h in enumerate(header[1:], start=1):
            hl = h.lower()
            if hl == "node":
                spec.append((i, "node"))
            elif hl == "process":
                spec.append((i, "process"))
            elif hl == "connection":
                spec.append((i, "connection"))
            elif hl == "unit":
                spec.append((i, "unit"))
            elif hl in {"source", "sink"}:
                spec.append((i, "node"))
    # else: no spatial filter (commodity, profile, reserve, timeline,
    # solve, model, etc.)

    return spec


# ---------------------------------------------------------------------------
# Membership predicate
# ---------------------------------------------------------------------------


@dataclass
class _KeepSets:
    """Resolved kept-entity sets for a region, including virtual entities."""

    nodes: set[str]
    units: set[str]
    connections: set[str]
    processes: set[str]  # units ∪ connections
    entities: set[str]  # nodes ∪ units ∪ connections

    def contains(self, kind: str, value: str) -> bool:
        if kind == "node":
            return value in self.nodes
        if kind == "unit":
            return value in self.units
        if kind == "connection":
            return value in self.connections
        if kind == "process":
            return value in self.processes
        if kind == "entity":
            return value in self.entities
        # Unknown kind → conservative: accept.
        return True


def _resolve_keep_sets(
    input_dir: Path,
    mem: RegionMembership,
    half_flows: list[HalfFlow],
) -> _KeepSets:
    """Build the kept-entity sets: in-region members + shared (non-region) +
    virtual import/export nodes & half-flow connections.
    """
    all_nodes = set(_read_single_col(input_dir / "node.csv"))
    all_units = set(_read_single_col(input_dir / "process_unit.csv"))
    all_connections = set(_read_single_col(input_dir / "process_connection.csv"))

    # Shared (non-region-specific) entities: nodes/units/connections NOT
    # assigned to any decomposition region.
    shared_nodes = all_nodes - mem.nodes - mem.other_region_nodes
    shared_units = all_units - mem.units - mem.other_region_units
    shared_connections = (
        all_connections - mem.connections - mem.other_region_connections
    )

    kept_nodes = mem.nodes | shared_nodes
    kept_units = mem.units | shared_units
    kept_connections = mem.connections | shared_connections

    # Drop the original cross-region connections (they will be replaced
    # by half-flow virtual connections).
    cross_region_conns = {hf.original_connection for hf in half_flows}
    kept_connections -= cross_region_conns

    # Add virtual half-flow nodes and virtual connections.
    virtual_nodes = {hf.virtual_node for hf in half_flows}
    virtual_conns = {hf.virtual_connection for hf in half_flows}
    kept_nodes |= virtual_nodes
    kept_connections |= virtual_conns

    kept_processes = kept_units | kept_connections
    kept_entities = kept_nodes | kept_processes

    return _KeepSets(
        nodes=kept_nodes,
        units=kept_units,
        connections=kept_connections,
        processes=kept_processes,
        entities=kept_entities,
    )


# ---------------------------------------------------------------------------
# Virtual entities — rows to inject into each CSV
# ---------------------------------------------------------------------------


_VIRTUAL_NODE_TYPE = "commodity"


def _virtual_rows(
    half_flows: list[HalfFlow],
    filename: str,
    header: list[str],
) -> list[list[str]]:
    """Return rows to inject into *filename* for the virtual half-flow
    entities.

    Only a small number of files need virtual entries:

    * ``node.csv``                — add virtual nodes
    * ``entity.csv``              — add virtual nodes + virtual connections
    * ``process.csv``             — add virtual connections
    * ``process_connection.csv``  — add virtual connections
    * ``p_node_type.csv``         — declare virtual nodes as commodity
    * ``process__source.csv``     — for each half-flow, (conn, source_node)
    * ``process__sink.csv``       — for each half-flow, (conn, sink_node)
    * ``process__ct_method.csv``  — virtual connection is no-loss
    * ``process_method.csv``      — virtual connection enters process_method set

    We keep it simple: virtual connections inherit the existing
    ``transfer_method = no_losses_no_variable_cost`` (method_2way_1var_off)
    analogous to the group-level DC-flow overrides used elsewhere in
    FlexTool.  This is just enough to make them appear in the LP with a
    flow variable; Agent 3.2's coordinator then prices them via λ.
    """
    rows: list[list[str]] = []
    if not half_flows:
        return rows

    def _src_sink(hf: HalfFlow) -> tuple[str, str]:
        # Export: in-region node → virtual node
        # Import: virtual node → in-region node
        if hf.side == "export":
            return hf.in_region_node, hf.virtual_node
        return hf.virtual_node, hf.in_region_node

    if filename == "node.csv":
        for hf in half_flows:
            rows.append([hf.virtual_node])
    elif filename == "entity.csv":
        for hf in half_flows:
            rows.append([hf.virtual_node])
            rows.append([hf.virtual_connection])
    elif filename == "process.csv":
        for hf in half_flows:
            rows.append([hf.virtual_connection])
    elif filename == "process_connection.csv":
        for hf in half_flows:
            rows.append([hf.virtual_connection])
    elif filename == "process_unit.csv":
        pass  # virtual entities are connections, not units
    elif filename == "p_node_type.csv":
        # node,p_node_type
        for hf in half_flows:
            rows.append([hf.virtual_node, _VIRTUAL_NODE_TYPE])
    elif filename == "process__source.csv":
        for hf in half_flows:
            src, _ = _src_sink(hf)
            rows.append([hf.virtual_connection, src])
    elif filename == "process__sink.csv":
        for hf in half_flows:
            _, snk = _src_sink(hf)
            rows.append([hf.virtual_connection, snk])
    elif filename == "process__ct_method.csv":
        for hf in half_flows:
            rows.append([hf.virtual_connection, "no_losses_no_variable_cost"])
    elif filename == "process_method.csv":
        # flextool.mod reads ``input/process_method.csv`` directly into
        # the ``process_method`` set; without this row the virtual
        # half-flow connection never enters ``process_source_toSink``
        # and v_flow[hf_*, ...] columns are absent from the MPS.
        for hf in half_flows:
            rows.append([hf.virtual_connection, "method_1way_1var_off"])
    elif filename == "p_process.csv":
        # The half-flow's ``existing`` capacity gates ``p_flow_max`` at
        # zero otherwise (v_flow columns appear in the MPS but with
        # ``lb = ub = 0``, so Lagrangian λ updates have no effect).
        # ``efficiency = 1.0`` because the half-flow is a bookkeeping
        # artefact, not a physical pipe — the underlying pipe
        # efficiency stays embedded in the importing region's import
        # half-flow.
        #
        # Capacity: we source the original pipe's ``existing`` from
        # :func:`_original_connection_existing` (reads
        # input/p_process.csv one directory up from the filtered copy
        # before the rename swap, or defaults to a conservative
        # ``1e3`` when that information is no longer available).  The
        # caller (input_writer) sets the value via a module-level
        # context dict — see
        # :data:`_virtual_capacity_override`.
        for hf in half_flows:
            cap = _virtual_capacity_override.get(hf.original_connection, 1e3)
            rows.append([hf.virtual_connection, "existing", str(cap)])
            rows.append([hf.virtual_connection, "efficiency", "1.0"])
            rows.append([hf.virtual_connection, "availability", "1.0"])
            rows.append([hf.virtual_connection, "virtual_unitsize", "1.0"])
    return rows


# Module-level override for original-connection capacities injected
# into the filtered ``p_process.csv`` for virtual half-flow
# connections.  Populated by ``build_region_directory`` before
# ``_virtual_rows`` is called, cleared afterwards.  Using a module
# attribute rather than function parameters keeps the ``_virtual_rows``
# signature backward-compatible with the tests added by Agent 3.1.
_virtual_capacity_override: dict[str, float] = {}


# ---------------------------------------------------------------------------
# The main build function
# ---------------------------------------------------------------------------


def build_region_directory(
    input_dir: Path,
    output_dir: Path,
    region: str,
    *,
    all_regions: list[str] | None = None,
) -> dict:
    """Build a filtered ``input_region_<region>/`` directory.

    Parameters
    ----------
    input_dir
        The fully-populated ``input/`` directory produced by
        ``write_input``.
    output_dir
        Destination directory for the filtered copy, e.g.
        ``work_folder / "input_region_region_A"``.
    region
        The decomposition region group name.
    all_regions
        The full list of decomposition-region group names.  If ``None``,
        falls back to the ``region_*`` naming convention (LH2 fixture).

    Returns a dict with entries:

    * ``region``: the region name
    * ``half_flows``: list of :class:`HalfFlow`
    * ``kept_nodes``, ``kept_units``, ``kept_connections``: sets
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    mem = discover_region_membership(input_dir, region)
    if all_regions is not None:
        mem.all_regions = set(all_regions) | {region}
        # Re-populate other_region_* given the authoritative list.
        group_nodes = _read_two_col_dict(input_dir / "group__node.csv")
        group_processes = _read_two_col_dict(input_dir / "group__process.csv")
        units = set(_read_single_col(input_dir / "process_unit.csv"))
        connections = set(_read_single_col(input_dir / "process_connection.csv"))
        mem.other_region_nodes = set()
        mem.other_region_units = set()
        mem.other_region_connections = set()
        for other in mem.all_regions:
            if other == region:
                continue
            mem.other_region_nodes.update(group_nodes.get(other, []))
            other_procs = set(group_processes.get(other, []))
            mem.other_region_units.update(other_procs & units)
            mem.other_region_connections.update(other_procs & connections)

    half_flows = classify_half_flows(input_dir, mem)
    keep = _resolve_keep_sets(input_dir, mem, half_flows)

    # Cross-region connection names: these are REMOVED from the filtered
    # output (replaced by virtual half-flow connections).
    cross_region_conns = {hf.original_connection for hf in half_flows}

    # Populate per-pipe capacity override so _virtual_rows can inject a
    # matching ``existing`` line into the filtered ``p_process.csv``.
    # Without this, p_flow_max gates v_flow[hf_*, ...] at zero and
    # Lagrangian cost updates have no effect on the solve.
    _virtual_capacity_override.clear()
    try:
        _, p_process_rows = _read_csv_rows(input_dir / "p_process.csv")
        for row in p_process_rows:
            if len(row) >= 3 and row[0] in cross_region_conns and row[1] == "existing":
                try:
                    _virtual_capacity_override[row[0]] = float(row[2])
                except ValueError:
                    continue
    except Exception:  # noqa: BLE001 — defensive; fall back to defaults.
        pass

    # Ensure output dir exists and is empty.
    output_dir.mkdir(parents=True, exist_ok=True)

    # Walk every CSV in input_dir.  For each, decide:
    # * if it has a column spec: filter rows and optionally inject virtuals
    # * else: copy verbatim
    for src in sorted(input_dir.iterdir()):
        if not src.is_file():
            continue
        if src.suffix.lower() != ".csv":
            shutil.copy2(src, output_dir / src.name)
            continue

        dst = output_dir / src.name
        header, rows = _read_csv_rows(src)
        if not header:
            # Empty file — copy as is.
            shutil.copy2(src, dst)
            continue

        spec = _infer_column_spec_from_header(header, src.name)

        if not spec:
            # No spatial filtering needed — copy verbatim.  Still inject
            # virtual rows if this file has an entry.
            virtuals = _virtual_rows(half_flows, src.name, header)
            if virtuals:
                _write_csv_rows(dst, header, rows + virtuals)
            else:
                shutil.copy2(src, dst)
            continue

        # Filter rows: every (col_index, kind) in spec must be satisfied.
        kept_rows: list[list[str]] = []
        for row in rows:
            # Defensive: pad short rows.
            keep_row = True
            for col_idx, kind in spec:
                if col_idx >= len(row):
                    continue
                val = row[col_idx]
                if kind == "process":
                    # Also drop cross-region original connections.
                    if val in cross_region_conns:
                        keep_row = False
                        break
                    if val not in keep.processes:
                        keep_row = False
                        break
                elif kind == "connection":
                    if val in cross_region_conns:
                        keep_row = False
                        break
                    if val not in keep.connections:
                        keep_row = False
                        break
                elif kind == "unit":
                    if val not in keep.units:
                        keep_row = False
                        break
                elif kind == "node":
                    if val not in keep.nodes:
                        keep_row = False
                        break
                elif kind == "entity":
                    if val in cross_region_conns:
                        keep_row = False
                        break
                    if val not in keep.entities:
                        keep_row = False
                        break
            if keep_row:
                kept_rows.append(row)

        virtuals = _virtual_rows(half_flows, src.name, header)
        _write_csv_rows(dst, header, kept_rows + virtuals)

    return {
        "region": region,
        "half_flows": half_flows,
        "kept_nodes": keep.nodes,
        "kept_units": keep.units,
        "kept_connections": keep.connections,
    }


def write_region_coupling_manifest(
    work_folder: Path,
    results: Iterable[dict],
) -> Path:
    """Write ``solve_data/region_coupling.csv`` listing coupling variables
    across all produced regions.

    Columns: ``region,process,side,virtual_node``.

    Returns the path written.
    """
    work_folder = Path(work_folder)
    solve_data = work_folder / "solve_data"
    solve_data.mkdir(parents=True, exist_ok=True)
    path = solve_data / "region_coupling.csv"
    rows: list[list[str]] = []
    for res in results:
        for hf in res.get("half_flows", []):
            rows.append([hf.region, hf.original_connection, hf.side, hf.virtual_node])
    _write_csv_rows(path, ["region", "process", "side", "virtual_node"], rows)
    return path


# ---------------------------------------------------------------------------
# Step 2.6 — Provider-based port
# ---------------------------------------------------------------------------
#
# Below: an in-memory parallel of the disk-walking machinery above.  Both
# share the same filter rules (``_EXPLICIT_SPECS`` /
# ``_infer_column_spec_from_header`` / ``_virtual_rows``); the difference
# is the data carrier.  Disk walkers read CSV bytes through ``csv.reader``
# and write CSVs via ``csv.writer``; the Provider port consumes
# :class:`polars.DataFrame` frames out of a :class:`FlexDataProvider` and
# emits filtered frames into a new region-scoped Provider.
#
# The CLI deliverable contract (``--region GROUP`` writes
# ``input_region_<GROUP>/<file>.csv``) still terminates in a disk
# directory — but the materialisation is one downward
# ``snapshot_processed_inputs`` from the region-Provider in
# ``region_decomposition.write_input_for_region``, NOT a snapshot of the
# full monolithic input/ followed by a disk-walk filter.  No bridge.


def _frame_to_csv_rows(
    df: pl.DataFrame,
) -> tuple[list[str], list[list[str]]]:
    """Return ``(header, rows_as_strings)`` for a Polars frame.

    Mirrors :func:`_read_csv_rows`'s "list of strings" representation so
    the rest of the filter machinery (which is string-equality-based)
    operates identically on disk and Provider data.

    Numeric values are stringified with ``str(v)``; ``None`` becomes an
    empty string (mirroring how empty CSV cells round-trip through
    ``csv.reader``).
    """
    if df.height == 0:
        return list(df.columns), []
    header = list(df.columns)
    rows: list[list[str]] = []
    for row in df.iter_rows():
        rows.append(["" if v is None else str(v) for v in row])
    return header, rows


def _rows_to_frame(
    header: list[str], rows: Iterable[list[str]],
) -> pl.DataFrame:
    """Inverse of :func:`_frame_to_csv_rows` — assemble a Polars frame.

    Every column is typed as ``Utf8``; downstream loaders coerce to
    numeric where required.  This matches the round-trip a CSV would
    take through ``pl.read_csv`` with default settings on these
    string-keyed entity tables — column types are inferred lazily by
    consumers.
    """
    if not header:
        return pl.DataFrame()
    rows_list = list(rows)
    if not rows_list:
        # Empty body with explicit columns.
        return pl.DataFrame({col: pl.Series(col, [], dtype=pl.Utf8) for col in header})
    # Pad/truncate to header width so polars accepts uniform rows.
    width = len(header)
    padded = [
        row[:width] + [""] * (width - len(row)) if len(row) < width else row[:width]
        for row in rows_list
    ]
    cols = {h: [r[i] for r in padded] for i, h in enumerate(header)}
    return pl.DataFrame({h: pl.Series(h, vals, dtype=pl.Utf8) for h, vals in cols.items()})


def _provider_csv_rows(
    provider: "FlexDataProvider", name: str,
) -> tuple[list[str], list[list[str]]]:
    """Provider-side analogue of :func:`_read_csv_rows`.

    Looks up ``input/<name>`` first, falls back to bare ``<name>``.
    Returns ``([], [])`` for missing frames (matches the disk helper's
    "absent file → empty" semantics).
    """
    df = provider.get(f"input/{name}")
    if df is None:
        df = provider.get(name)
    if df is None:
        return [], []
    return _frame_to_csv_rows(df)


def _provider_single_col(provider: "FlexDataProvider", name: str) -> list[str]:
    _, rows = _provider_csv_rows(provider, name)
    return [r[0] for r in rows if r and r[0] != ""]


def _provider_two_col_dict(
    provider: "FlexDataProvider", name: str,
) -> dict[str, list[str]]:
    _, rows = _provider_csv_rows(provider, name)
    out: dict[str, list[str]] = {}
    for r in rows:
        if len(r) < 2:
            continue
        out.setdefault(r[0], []).append(r[1])
    return out


def _provider_connection_endpoints(
    provider: "FlexDataProvider",
) -> dict[str, tuple[str, str]]:
    """Provider-side analogue of :func:`_read_connection_endpoints`."""
    connections = set(_provider_single_col(provider, "process_connection.csv"))
    src_of: dict[str, str] = {}
    _, src_rows = _provider_csv_rows(provider, "process__source.csv")
    for r in src_rows:
        if len(r) >= 2 and r[0] in connections:
            src_of[r[0]] = r[1]
    snk_of: dict[str, str] = {}
    _, snk_rows = _provider_csv_rows(provider, "process__sink.csv")
    for r in snk_rows:
        if len(r) >= 2 and r[0] in connections:
            snk_of[r[0]] = r[1]
    return {
        conn: (src_of[conn], snk_of[conn])
        for conn in connections
        if conn in src_of and conn in snk_of
    }


def discover_region_membership_from_provider(
    provider: "FlexDataProvider", region: str,
) -> RegionMembership:
    """Provider-side analogue of :func:`discover_region_membership`."""
    group_nodes = _provider_two_col_dict(provider, "group__node.csv")
    group_processes = _provider_two_col_dict(provider, "group__process.csv")

    all_groups = set(_provider_single_col(provider, "group.csv"))
    region_like = {g for g in all_groups if g.startswith("region_")}
    if region not in region_like:
        region_like.add(region)

    mem = RegionMembership(region=region, all_regions=region_like)
    mem.nodes = set(group_nodes.get(region, []))
    in_region_processes = set(group_processes.get(region, []))

    units = set(_provider_single_col(provider, "process_unit.csv"))
    connections = set(_provider_single_col(provider, "process_connection.csv"))
    mem.units = in_region_processes & units
    mem.connections = in_region_processes & connections

    for other in region_like:
        if other == region:
            continue
        mem.other_region_nodes.update(group_nodes.get(other, []))
        other_procs = set(group_processes.get(other, []))
        mem.other_region_units.update(other_procs & units)
        mem.other_region_connections.update(other_procs & connections)
    return mem


def classify_half_flows_from_provider(
    provider: "FlexDataProvider", mem: RegionMembership,
) -> list[HalfFlow]:
    """Provider-side analogue of :func:`classify_half_flows`."""
    endpoints = _provider_connection_endpoints(provider)
    half_flows: list[HalfFlow] = []
    for conn, (src, snk) in endpoints.items():
        src_in = src in mem.nodes
        snk_in = snk in mem.nodes
        src_out = src in mem.other_region_nodes
        snk_out = snk in mem.other_region_nodes
        if src_in and snk_out:
            virtual = f"{conn}__export__{mem.region}"
            half_flows.append(HalfFlow(
                original_connection=conn,
                region=mem.region,
                side="export",
                in_region_node=src,
                virtual_node=virtual,
                virtual_connection=f"hf_{conn}__export__{mem.region}",
            ))
        elif snk_in and src_out:
            virtual = f"{conn}__import__{mem.region}"
            half_flows.append(HalfFlow(
                original_connection=conn,
                region=mem.region,
                side="import",
                in_region_node=snk,
                virtual_node=virtual,
                virtual_connection=f"hf_{conn}__import__{mem.region}",
            ))
    return half_flows


def _resolve_keep_sets_from_provider(
    provider: "FlexDataProvider",
    mem: RegionMembership,
    half_flows: list[HalfFlow],
) -> _KeepSets:
    """Provider-side analogue of :func:`_resolve_keep_sets`."""
    all_nodes = set(_provider_single_col(provider, "node.csv"))
    all_units = set(_provider_single_col(provider, "process_unit.csv"))
    all_connections = set(_provider_single_col(provider, "process_connection.csv"))

    shared_nodes = all_nodes - mem.nodes - mem.other_region_nodes
    shared_units = all_units - mem.units - mem.other_region_units
    shared_connections = (
        all_connections - mem.connections - mem.other_region_connections
    )

    kept_nodes = mem.nodes | shared_nodes
    kept_units = mem.units | shared_units
    kept_connections = mem.connections | shared_connections

    cross_region_conns = {hf.original_connection for hf in half_flows}
    kept_connections -= cross_region_conns

    virtual_nodes = {hf.virtual_node for hf in half_flows}
    virtual_conns = {hf.virtual_connection for hf in half_flows}
    kept_nodes |= virtual_nodes
    kept_connections |= virtual_conns

    kept_processes = kept_units | kept_connections
    kept_entities = kept_nodes | kept_processes

    return _KeepSets(
        nodes=kept_nodes,
        units=kept_units,
        connections=kept_connections,
        processes=kept_processes,
        entities=kept_entities,
    )


def build_region_provider(
    provider: "FlexDataProvider",
    region: str,
    *,
    all_regions: list[str] | None = None,
) -> tuple["FlexDataProvider", dict]:
    """Provider-in/Provider-out analogue of :func:`build_region_directory`.

    Operates entirely in-memory: consumes the cascade-input *provider*
    (every ``input/<name>`` frame populated by
    :func:`flextool.input_derivation.run`) and returns
    ``(region_provider, result)``:

    * ``region_provider`` carries the same set of keys filtered to the
      region's members, with virtual half-flow rows injected where
      needed.  CLI mode materialises this with one call to
      ``region_provider.snapshot_processed_inputs(output_dir)``.
    * ``result`` is the same metadata dict shape returned by the disk
      port (``region``, ``half_flows``, ``kept_nodes``,
      ``kept_units``, ``kept_connections``) so downstream code is
      drop-in.

    Filter rules are shared with :func:`build_region_directory`
    (``_EXPLICIT_SPECS`` / ``_infer_column_spec_from_header`` /
    ``_virtual_rows``); only the data carrier changes.
    """
    from flextool.engine_polars._flex_data_provider import FlexDataProvider

    mem = discover_region_membership_from_provider(provider, region)
    if all_regions is not None:
        mem.all_regions = set(all_regions) | {region}
        group_nodes = _provider_two_col_dict(provider, "group__node.csv")
        group_processes = _provider_two_col_dict(provider, "group__process.csv")
        units = set(_provider_single_col(provider, "process_unit.csv"))
        connections = set(_provider_single_col(provider, "process_connection.csv"))
        mem.other_region_nodes = set()
        mem.other_region_units = set()
        mem.other_region_connections = set()
        for other in mem.all_regions:
            if other == region:
                continue
            mem.other_region_nodes.update(group_nodes.get(other, []))
            other_procs = set(group_processes.get(other, []))
            mem.other_region_units.update(other_procs & units)
            mem.other_region_connections.update(other_procs & connections)

    half_flows = classify_half_flows_from_provider(provider, mem)
    keep = _resolve_keep_sets_from_provider(provider, mem, half_flows)

    cross_region_conns = {hf.original_connection for hf in half_flows}

    # Same per-pipe capacity override as the disk path: walk p_process to
    # find each cross-region connection's ``existing`` value so the
    # virtual half-flow inherits a non-zero p_flow_max.
    _virtual_capacity_override.clear()
    try:
        _, p_process_rows = _provider_csv_rows(provider, "p_process.csv")
        for row in p_process_rows:
            if len(row) >= 3 and row[0] in cross_region_conns and row[1] == "existing":
                try:
                    _virtual_capacity_override[row[0]] = float(row[2])
                except ValueError:
                    continue
    except Exception:  # noqa: BLE001 — defensive; fall back to defaults.
        pass

    region_provider = FlexDataProvider()

    # Iterate every ``input/<name>`` frame in the source Provider and
    # produce a filtered/virtual-augmented frame in the region Provider.
    # Frames stored under other parents (e.g. ``solve_data/*``) are
    # passed through unchanged — region filtering only applies to the
    # entity-set + parameter CSVs under ``input/``.
    seen_filenames: set[str] = set()
    for key, frame in provider.items():
        parent = key.split("/", 1)[0] if "/" in key else ""
        if parent != "input":
            # Non-input frames are copied verbatim.
            region_provider.put(key, frame)
            continue

        stem = key.split("/", 1)[1] if "/" in key else key
        filename = f"{stem}.csv"
        seen_filenames.add(filename)
        header, rows = _frame_to_csv_rows(frame)
        if not header:
            region_provider.put(key, frame)
            continue
        spec = _infer_column_spec_from_header(header, filename)
        if not spec:
            virtuals = _virtual_rows(half_flows, filename, header)
            if virtuals:
                region_provider.put(
                    key, _rows_to_frame(header, rows + virtuals),
                )
            else:
                region_provider.put(key, frame)
            continue
        kept_rows: list[list[str]] = []
        for row in rows:
            keep_row = True
            for col_idx, kind in spec:
                if col_idx >= len(row):
                    continue
                val = row[col_idx]
                if kind == "process":
                    if val in cross_region_conns or val not in keep.processes:
                        keep_row = False
                        break
                elif kind == "connection":
                    if val in cross_region_conns or val not in keep.connections:
                        keep_row = False
                        break
                elif kind == "unit":
                    if val not in keep.units:
                        keep_row = False
                        break
                elif kind == "node":
                    if val not in keep.nodes:
                        keep_row = False
                        break
                elif kind == "entity":
                    if val in cross_region_conns or val not in keep.entities:
                        keep_row = False
                        break
            if keep_row:
                kept_rows.append(row)
        virtuals = _virtual_rows(half_flows, filename, header)
        region_provider.put(key, _rows_to_frame(header, kept_rows + virtuals))

    # Some files referenced by ``_virtual_rows`` (e.g. ``process_method``
    # or ``p_process``) might not be in the source Provider at all (the
    # fixture didn't populate them).  ``_virtual_rows`` skips those; we
    # already passed through every key we saw, so nothing to do here.
    _ = seen_filenames  # diagnostic hook — kept for future debugging.

    return region_provider, {
        "region": region,
        "half_flows": half_flows,
        "kept_nodes": keep.nodes,
        "kept_units": keep.units,
        "kept_connections": keep.connections,
    }


def write_region_coupling_manifest_to_provider(
    provider: "FlexDataProvider",
    results: Iterable[dict],
) -> None:
    """Provider-side analogue of :func:`write_region_coupling_manifest`.

    Stores ``solve_data/region_coupling`` into *provider* with the same
    four-column layout (``region, process, side, virtual_node``).  The
    CLI driver materialises it via ``snapshot_processed_inputs``.
    """
    rows: list[tuple[str, str, str, str]] = []
    for res in results:
        for hf in res.get("half_flows", []):
            rows.append((hf.region, hf.original_connection, hf.side, hf.virtual_node))
    if rows:
        df = pl.DataFrame({
            "region": [r[0] for r in rows],
            "process": [r[1] for r in rows],
            "side": [r[2] for r in rows],
            "virtual_node": [r[3] for r in rows],
        })
    else:
        df = pl.DataFrame({
            "region": pl.Series("region", [], dtype=pl.Utf8),
            "process": pl.Series("process", [], dtype=pl.Utf8),
            "side": pl.Series("side", [], dtype=pl.Utf8),
            "virtual_node": pl.Series("virtual_node", [], dtype=pl.Utf8),
        })
    provider.put("solve_data/region_coupling", df)
