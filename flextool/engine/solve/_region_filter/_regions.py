"""The regions of a Benders decomposition: the :class:`HalfFlow` / :class:`RegionSplit` records, the decomposition
method and region membership read from the solve's frames, the region of a process by its nodes and the
master-hosted nodes; the module logger."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import polars as pl

from flextool.engine.preprocess.flex_data import FlexData


_logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class HalfFlow:
    """One half-flow arc replacing one direction of a cross-region pipe.

    A bi-directional cross-region pipe ``pipe(A→B, B→A)`` produces FOUR
    HalfFlows: ``pipe(A→B)`` becomes an export in A and an import in B;
    ``pipe(B→A)`` becomes an export in B and an import in A.  Each
    coupling pair shares the same ``(original_p, original_source,
    original_sink)`` triple — the :mod:`flextool.engine.solve._benders`
    coordinator pairs export and import on that key.
    """
    region: str
    side: str  # "export" or "import"
    # Original cross-region arc identity (the pairing key).
    original_p: str
    original_source: str
    original_sink: str
    # In-region terminal node — appears in the regional flow_to_n /
    # flow_from_n alongside the half-flow.  For an export this is the
    # original source; for an import it is the original sink.
    in_region_node: str
    # Virtual entities introduced by this half-flow.
    virtual_node: str
    # Virtual connection: the (p, source, sink) row that replaces the
    # original cross-region arc inside the region's frames.
    virtual_p: str
    virtual_arc_source: str
    virtual_arc_sink: str
    # The original arc is a one-variable connection's (process_source_sink_2way_1var): the virtual arc is one
    # too, so its reverse direction is a v_flow_back column coupled like v_flow (item 301).
    one_var: bool = False


@dataclass
class RegionSplit:
    """One region's filtered :class:`FlexData` plus coupling metadata."""
    region: str
    data: FlexData
    half_flows: list[HalfFlow] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Decomposition discovery
# ---------------------------------------------------------------------------


def load_decomposition_method(
    work_dir: "Path | str | None" = None,
    *,
    provider: "object | None" = None,
) -> dict[str, str]:
    """Return ``{group: method}`` from the Provider key
    ``input/p_group_decomposition``.

    *provider* is required (``provider=None`` raises ``ValueError``);
    a provider without the key → ``{}``.  *work_dir* is retained for
    call compatibility and is not read — the work folder on disk is
    never consulted.

    The whole-system loader doesn't surface the decomposition_method
    parameter in :class:`FlexData` because it's only used by the
    decomposition driver itself; this helper plus the Provider keep
    it out of cascade memory until the driver needs it.
    """
    if provider is None:
        raise ValueError(
            "load_decomposition_method requires a FlexDataProvider; it "
            "never reads input/p_group_decomposition.csv on disk."
        )
    key = "input/p_group_decomposition"
    df: pl.DataFrame | None = provider.get(key) if provider.has(key) else None
    if df is None or df.height == 0:
        return {}
    cols = df.columns
    # Expected columns: group, groupParam, p_group
    if "group" not in cols:
        return {}
    val_col = next((c for c in ("p_group", "value") if c in cols), None)
    if val_col is None:
        return {}
    rows = (
        df.filter(pl.col("groupParam") == "decomposition_method")
        if "groupParam" in cols else df
    )
    return {r["group"]: r[val_col] for r in rows.iter_rows(named=True)}


def discover_regions(
    work_dir: "Path | str | None" = None,
    *,
    provider: "object | None" = None,
) -> list[str]:
    """Return the list of group names with
    ``decomposition_method=benders_regional`` (Provider-only; see
    :func:`load_decomposition_method`)."""
    methods = load_decomposition_method(work_dir, provider=provider)
    return sorted(g for g, m in methods.items() if m == "benders_regional")


def load_region_membership(
    data: FlexData, regions: list[str],
) -> dict[str, dict[str, set[str]]]:
    """Return ``{region: {"nodes": set, "processes": set}}`` from
    ``data.group_entity`` / ``data.group_node`` for the given regions.

    Falls back to empty sets when the relevant frames are absent (a
    same-shape stub useful in unit tests).
    """
    out: dict[str, dict[str, set[str]]] = {}
    nodes_by_g: dict[str, set[str]] = {}
    procs_by_g: dict[str, set[str]] = {}
    if data.group_node is not None and data.group_node.height > 0:
        for r in data.group_node.iter_rows(named=True):
            nodes_by_g.setdefault(r["g"], set()).add(r["n"])
    if data.group_entity is not None and data.group_entity.height > 0:
        # group_entity is the union (g, e) — to get processes we'd need
        # to know which e's are processes.  process_unit + connections
        # set is everything in process_source_sink["p"].
        all_procs: set[str] = set()
        if data.process_source_sink is not None:
            all_procs |= set(data.process_source_sink["p"].unique().to_list())
        if data.process_indirect is not None:
            all_procs |= set(data.process_indirect["p"].unique().to_list())
        for r in data.group_entity.iter_rows(named=True):
            if r["e"] in all_procs:
                procs_by_g.setdefault(r["g"], set()).add(r["e"])
    for g in regions:
        out[g] = {
            "nodes": set(nodes_by_g.get(g, set())),
            "processes": set(procs_by_g.get(g, set())),
        }
    return out


def process_region_by_nodes(
    pss: pl.DataFrame | None, region_nodes: dict[str, set[str]],
) -> dict[str, str]:
    """``{process: region}`` for each process whose arcs touch region nodes of exactly ONE region.

    Read from the ``source`` / ``sink`` of the process's ``process_source_sink`` arcs; other tokens (the unit
    itself, shared or master-hosted nodes) are ignored.  A cross-region connection, a unit with nodes in two
    regions and a process with no region node (master-local, shared only) are left out.  Used for the region
    investments of processes not listed in their region group (the documented setup lists nodes only).
    """
    if pss is None or pss.height == 0:
        return {}
    node_region = {n: r for r, ns in region_nodes.items() for n in ns}
    regions_of: dict[str, set[str]] = {}
    for p, source, sink in pss.select(pl.col("p", "source", "sink").cast(pl.Utf8)).iter_rows():
        for n in (source, sink):
            if n in node_region:
                regions_of.setdefault(p, set()).add(node_region[n])
    return {p: next(iter(rs)) for p, rs in regions_of.items() if len(rs) == 1}


def compute_master_hosted_nodes(
    data: FlexData,
    region_membership: dict[str, dict[str, set[str]]],
) -> set[str]:
    """Return the master-hosted node set: every node carrying a balance
    or state row (``data.nodeBalance`` ∪ ``data.nodeState``) that is in
    NO region's membership.

    Nodes with no balance/state row (pure commodity/market nodes) are
    deliberately NOT included — they keep today's shared-replicate
    semantics (replication is safe for them: there is no balance row to
    duplicate), pinned by the existing region-filter tests.
    """
    balance_state: set[str] = set()
    if data.nodeBalance is not None and data.nodeBalance.height > 0:
        balance_state |= set(data.nodeBalance["n"].to_list())
    if data.nodeState is not None and data.nodeState.height > 0:
        balance_state |= set(data.nodeState["n"].to_list())
    region_all: set[str] = set()
    for m in region_membership.values():
        region_all |= m["nodes"]
    return balance_state - region_all
