"""Writer-port Phase 1 (L3-L6) — mid-level set / param projections.

Native polars port of the next batch of preprocessing families invoked
from :func:`flextool.flextoolrunner.input_writer.write_input` (lines
1893-1945).  Each family reads a small handful of ``input/*.csv`` (and,
occasionally, a leaf-level ``solve_data/*.csv`` written earlier by
:mod:`._writer_leaf_sets`) and emits one or more ``solve_data/`` CSVs.

Ported legacy modules (preprocessing/):

* ``node_type_sets.py``             —  77 LOC — 4 partitions of node by p_node_type
* ``union_sets.py``                 —  69 LOC — 2 ordered-union 2-tuples
* ``dc_angle_bounds.py``            —  52 LOC — per-DC-node angle bounds
* ``reserve_method_partitions.py``  —  73 LOC — reserve.csv + 3 method partitions
* ``nonsync_sets.py``               — 153 LOC — process__sink_nonSync + group_inside_group_nonSync
* ``method_with_fallback_sets.py``  — 194 LOC — 5 per-entity fallback method tables
* ``invest_total_sets.py``          — 113 LOC — 5 invest/divest-total filters + ci_ladder_cumulative
* ``structural_filters.py``         — 182 LOC — 6 single-condition filters

Total ~913 LOC of legacy code ported.  Each ``derive_*`` returns a
fresh ``pl.DataFrame`` (the in-memory contract); ``write_*`` wrappers
materialise the frame to the legacy ``solve_data/*.csv`` path so
downstream consumers continue to read identical bytes.

Style mirrors :mod:`._writer_leaf_sets`: eager polars reads of tiny
CSVs, expression chains, ``unique(maintain_order=True)`` for ordered
dedup.  Constants mirror the legacy module's literals one-for-one.
"""
from __future__ import annotations

from pathlib import Path

import polars as pl


# ---------------------------------------------------------------------------
# CSV I/O — same conventions as _writer_leaf_sets:
#   * eager read, missing file → empty frame with requested schema
#   * positional column rename (handle legacy headers that differ in label)
#   * empty frame still writes header line
# ---------------------------------------------------------------------------

def _read_csv(path: Path, columns: list[str],
              *, provider: "object | None" = None) -> pl.DataFrame:
    """Read a tiny flextool CSV with positional column rename.

    Forces every column to ``Utf8`` (via ``infer_schema_length=0``)
    regardless of the CSV's header names or data shape.  This matters
    because the ``columns=`` arg is the *post-rename* target — it
    cannot be used as ``schema_overrides=`` keys, since those match
    CSV header names (e.g. ``p_commodity.csv`` carries the header
    ``commodity,commodityParam,p_commodity`` and gets renamed to
    ``commodity,param,value`` here).  Without forcing Utf8, polars
    type-inference picks Float64 for all-numeric value columns and
    downstream ``pl.col(...) != ""`` filters raise
    ``cannot compare string with numeric type``.
    """
    # Step 1-g — Provider-first; falls back to legacy seed lookup
    # (still installed during the migration window) then disk.
    from flextool.engine_polars._writer_provider_io import (
        _provider_key,
        _provider_lookup_positional,
    )
    seeded = _provider_lookup_positional(
        provider, _provider_key(path), path, columns,
    )
    if seeded is not None:
        return seeded
    if not path.exists() or path.stat().st_size == 0:
        return pl.DataFrame({c: [] for c in columns}, schema={c: pl.Utf8 for c in columns})
    df = pl.read_csv(
        path,
        has_header=True,
        infer_schema_length=0,
        truncate_ragged_lines=True,
    )
    keep = df.columns[: len(columns)]
    df = df.select(keep)
    df.columns = columns
    return df


def _write(df: pl.DataFrame, path: Path) -> None:
    # Phase E-c — gate disk emission behind ``emit_csvs_enabled``.
    from flextool.engine_polars._flex_data_accumulator import emit_csvs_enabled
    if not emit_csvs_enabled():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    df.write_csv(path)


def _drop_blank_rows(df: pl.DataFrame, required_cols: list[str]) -> pl.DataFrame:
    expr = pl.col(required_cols[0]) != ""
    for c in required_cols[1:]:
        expr = expr & (pl.col(c) != "")
    return df.filter(expr)


# ===========================================================================
# Family 5 — node_type_sets (legacy: preprocessing/node_type_sets.py)
# ===========================================================================

# Mirror flextool.mod:192 ``default 'balance'`` clause on p_node_type.
_DEFAULT_NODE_TYPE = "balance"

# (output filename, set of effective types that match the partition)
_NODE_TYPE_PARTITIONS: list[tuple[str, frozenset[str]]] = [
    ("nodeCommodity.csv",     frozenset(("commodity",))),
    ("nodeBalance.csv",       frozenset(("balance", "storage"))),
    ("nodeState.csv",         frozenset(("storage",))),
    ("nodeBalancePeriod.csv", frozenset(("balance_within_period",))),
]


def derive_node_effective_type(input_dir: Path) -> pl.DataFrame:
    """Materialize every node with its effective ``p_node_type``.

    Nodes without an explicit row in ``p_node_type.csv`` get the
    flextool.mod default ``'balance'``.  Order = ``node.csv`` order
    (mod's would-be iteration order).
    """
    nodes = _read_csv(input_dir / "node.csv", ["node"])
    nodes = _drop_blank_rows(nodes, ["node"])
    explicit = _read_csv(input_dir / "p_node_type.csv", ["node", "type"])
    explicit = _drop_blank_rows(explicit, ["node", "type"])
    return (
        nodes.join(explicit, on="node", how="left")
             .with_columns(
                 pl.col("type").fill_null(_DEFAULT_NODE_TYPE),
             )
    )


def write_node_type_sets(input_dir: Path, solve_data_dir: Path) -> None:
    effective = derive_node_effective_type(input_dir)
    for fname, types in _NODE_TYPE_PARTITIONS:
        out = (
            effective.filter(pl.col("type").is_in(list(types)))
                     .select("node")
        )
        _write(out, solve_data_dir / fname)


# ===========================================================================
# Family 6 — union_sets (legacy: preprocessing/union_sets.py)
# ===========================================================================

def _ordered_union_pairs(
    sources: list[pl.DataFrame], columns: list[str],
) -> pl.DataFrame:
    """Vertical-concat then dedupe preserving first-occurrence order."""
    aligned = [df.select(columns) for df in sources]
    return (
        pl.concat(aligned, how="vertical")
          .pipe(_drop_blank_rows, columns)
          .unique(maintain_order=True)
    )


def derive_group_entity(input_dir: Path) -> pl.DataFrame:
    """flextool.mod:287 — ``group_process ∪ group_node``."""
    gp = _read_csv(input_dir / "group__process.csv", ["group", "entity"])
    gn = _read_csv(input_dir / "group__node.csv",    ["group", "entity"])
    return _ordered_union_pairs([gp, gn], ["group", "entity"])


def derive_process_delayed__duration(input_dir: Path) -> pl.DataFrame:
    """flextool.mod:950 — ``process_delay_weighted ∪ process_delay_single``."""
    w = _read_csv(input_dir / "p_process_delay_weighted.csv", ["process", "delay_duration"])
    s = _read_csv(input_dir / "process_delay_single.csv",     ["process", "delay_duration"])
    return _ordered_union_pairs([w, s], ["process", "delay_duration"])


def write_group_entity(input_dir: Path, solve_data_dir: Path) -> None:
    _write(derive_group_entity(input_dir), solve_data_dir / "group_entity.csv")


def write_process_delayed__duration(input_dir: Path, solve_data_dir: Path) -> None:
    _write(
        derive_process_delayed__duration(input_dir),
        solve_data_dir / "process_delayed__duration.csv",
    )


# ===========================================================================
# Family 7 — dc_angle_bounds (legacy: preprocessing/dc_angle_bounds.py)
# ===========================================================================

# 8-digit truncation of π from flextool.mod:2262.  Do NOT replace with
# math.pi — bit-exact MPS parity requires this exact decimal string.
_PI_LITERAL = "3.14159265"


def derive_dc_angle_bounds(input_dir: Path) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Return (lower, upper) frames for nodes participating in DC flow.

    ref nodes pin both bounds to ``0``; other DC nodes get ``±π``.
    """
    dc = _read_csv(input_dir / "node_dc_power_flow.csv", ["node"])
    dc = _drop_blank_rows(dc, ["node"])
    ref = _read_csv(input_dir / "node_reference_angle.csv", ["node"])
    ref_set = ref.filter(pl.col("node") != "").get_column("node").to_list()

    lower_values = pl.when(pl.col("node").is_in(ref_set)).then(pl.lit("0")).otherwise(pl.lit(f"-{_PI_LITERAL}"))
    upper_values = pl.when(pl.col("node").is_in(ref_set)).then(pl.lit("0")).otherwise(pl.lit(_PI_LITERAL))
    lower = dc.with_columns(value=lower_values).select("node", "value")
    upper = dc.with_columns(value=upper_values).select("node", "value")
    return lower, upper


def write_dc_angle_bounds(input_dir: Path, solve_data_dir: Path) -> None:
    lower, upper = derive_dc_angle_bounds(input_dir)
    _write(lower, solve_data_dir / "p_angle_lower.csv")
    _write(upper, solve_data_dir / "p_angle_upper.csv")


# ===========================================================================
# Family 8 — reserve_method_partitions
# (legacy: preprocessing/reserve_method_partitions.py)
# ===========================================================================

_RESERVE_TIMESERIES_METHODS: frozenset[str] = frozenset((
    "timeseries_only", "timeseries_and_dynamic",
    "timeseries_and_large_failure", "all",
))
_RESERVE_DYNAMIC_METHODS: frozenset[str] = frozenset((
    "dynamic_only", "timeseries_and_dynamic",
    "dynamic_and_large_failure", "all",
))
_RESERVE_N_1_METHODS: frozenset[str] = frozenset((
    "large_failure_only", "timeseries_and_large_failure",
    "dynamic_and_large_failure", "all",
))

_RESERVE_QUAD_COLS = ["reserve", "upDown", "group", "method"]


def derive_reserve_universe(input_dir: Path) -> pl.DataFrame:
    """Single-column ``reserve`` projected from the quad CSV."""
    quad = _read_csv(input_dir / "reserve__upDown__group__method.csv", _RESERVE_QUAD_COLS)
    quad = _drop_blank_rows(quad, _RESERVE_QUAD_COLS)
    return quad.select("reserve").unique(maintain_order=True)


def derive_reserve_method_partition(
    input_dir: Path, allowed: frozenset[str],
) -> pl.DataFrame:
    """4-tuple rows whose method ∈ allowed.  Order preserved, deduped."""
    quad = _read_csv(input_dir / "reserve__upDown__group__method.csv", _RESERVE_QUAD_COLS)
    quad = _drop_blank_rows(quad, _RESERVE_QUAD_COLS)
    return (
        quad.filter(pl.col("method").is_in(list(allowed)))
            .unique(maintain_order=True)
    )


def write_reserve_partitions(input_dir: Path, solve_data_dir: Path) -> None:
    _write(derive_reserve_universe(input_dir), solve_data_dir / "reserve.csv")
    for fname, allowed in (
        ("reserve__upDown__group__method_timeseries.csv", _RESERVE_TIMESERIES_METHODS),
        ("reserve__upDown__group__method_dynamic.csv",    _RESERVE_DYNAMIC_METHODS),
        ("reserve__upDown__group__method_n_1.csv",        _RESERVE_N_1_METHODS),
    ):
        _write(
            derive_reserve_method_partition(input_dir, allowed),
            solve_data_dir / fname,
        )


# ===========================================================================
# Family 9 — nonsync_sets (legacy: preprocessing/nonsync_sets.py)
# ===========================================================================

def derive_process__sink_nonSync(input_dir: Path) -> pl.DataFrame:
    """flextool.mod:1980-1985 — 3-branch OR over sink/source membership.

    Output: (process, sink) 2-tuples.  Branch order (deduped):
      1. ``(p, sink) in process_sink`` AND ``(p, sink) in process__sink_nonSync_unit``
      2. ``(p, sink) in process_sink`` AND ``p in process_nonSync_connection``
      3. ``(p, source) in process_source`` AND ``p in process_nonSync_connection``
    """
    sinks = _read_csv(input_dir / "process__sink.csv", ["process", "sink"])
    sinks = _drop_blank_rows(sinks, ["process", "sink"])
    sources = _read_csv(input_dir / "process__source.csv", ["process", "sink"])
    sources = _drop_blank_rows(sources, ["process", "sink"])
    nonsync_units = _read_csv(
        input_dir / "process__sink_nonSync_unit.csv", ["process", "sink"],
    )
    nonsync_conn = _read_csv(
        input_dir / "process_nonSync_connection.csv", ["process"],
    )
    nonsync_conn_set = (
        nonsync_conn.filter(pl.col("process") != "")
                    .get_column("process").to_list()
    )

    # Branch 1+2: walk sinks once, admit if either condition matches.
    sink_matches = sinks.join(
        nonsync_units, on=["process", "sink"], how="inner",
    ).select("process", "sink")
    sink_conn_matches = sinks.filter(
        pl.col("process").is_in(nonsync_conn_set)
    ).select("process", "sink")
    # Branch 3: source rows admitted whenever process ∈ nonSync_connection.
    source_conn_matches = sources.filter(
        pl.col("process").is_in(nonsync_conn_set)
    ).select("process", "sink")

    # Legacy walks sinks first (in order), THEN sources.  Within the sink
    # walk, branch 1 takes precedence over branch 2 but neither emits
    # duplicates because the dict deduplicates.  Reproducing that with
    # vertical-concat-then-unique preserves the same first-seen order.
    combined = pl.concat(
        [sink_matches, sink_conn_matches, source_conn_matches],
        how="vertical",
    )
    return combined.unique(maintain_order=True)


def derive_process__group_inside_group_nonSync(input_dir: Path) -> pl.DataFrame:
    """flextool.mod:2017-2023 — exists (source, sink) ∈ g, source ≠ sink, for p.

    Iterate process × groupNonSync in CSV order to match the order the
    mod's nested loops would produce.
    """
    nonsync_groups = _read_csv(input_dir / "groupNonSync.csv", ["group"])
    nonsync_groups = _drop_blank_rows(nonsync_groups, ["group"])
    if nonsync_groups.height == 0:
        return pl.DataFrame({"process": [], "group": []},
                            schema={"process": pl.Utf8, "group": pl.Utf8})

    processes = _read_csv(input_dir / "process.csv", ["process"])
    processes = _drop_blank_rows(processes, ["process"])
    group_nodes = _read_csv(input_dir / "group__node.csv", ["group", "node"])
    group_nodes = _drop_blank_rows(group_nodes, ["group", "node"])
    sources = _read_csv(input_dir / "process__source.csv", ["process", "node"])
    sources = _drop_blank_rows(sources, ["process", "node"])
    sinks = _read_csv(input_dir / "process__sink.csv", ["process", "node"])
    sinks = _drop_blank_rows(sinks, ["process", "node"])

    # Lookup tables.  We materialise to python dicts because the per-row
    # set-intersection logic ("∃ source ≠ sink both in g") is awkward to
    # express as a pure polars expression and the input sizes are tiny.
    group_node_lookup: dict[str, set[str]] = {}
    for g, n in group_nodes.iter_rows():
        group_node_lookup.setdefault(g, set()).add(n)
    process_sources: dict[str, set[str]] = {}
    for p, n in sources.iter_rows():
        process_sources.setdefault(p, set()).add(n)
    process_sinks: dict[str, set[str]] = {}
    for p, n in sinks.iter_rows():
        process_sinks.setdefault(p, set()).add(n)

    nonsync_group_list = nonsync_groups.get_column("group").to_list()
    rows: list[tuple[str, str]] = []
    for p in processes.get_column("process").to_list():
        psrc = process_sources.get(p)
        psnk = process_sinks.get(p)
        if not psrc or not psnk:
            continue
        for g in nonsync_group_list:
            gnodes = group_node_lookup.get(g)
            if not gnodes:
                continue
            srcs_in = psrc & gnodes
            sinks_in = psnk & gnodes
            if not srcs_in or not sinks_in:
                continue
            # ∃ s ≠ t with s ∈ srcs_in, t ∈ sinks_in iff NOT
            # (|srcs|=1 ∧ |sinks|=1 ∧ srcs==sinks).
            if (len(srcs_in) == 1 and len(sinks_in) == 1
                    and srcs_in == sinks_in):
                continue
            rows.append((p, g))
    return pl.DataFrame(
        {"process": [r[0] for r in rows], "group": [r[1] for r in rows]},
        schema={"process": pl.Utf8, "group": pl.Utf8},
    )


def write_process__sink_nonSync(input_dir: Path, solve_data_dir: Path) -> None:
    _write(
        derive_process__sink_nonSync(input_dir),
        solve_data_dir / "process__sink_nonSync.csv",
    )


def write_process_group_inside_group_nonsync(
    input_dir: Path, solve_data_dir: Path,
) -> None:
    _write(
        derive_process__group_inside_group_nonSync(input_dir),
        solve_data_dir / "process__group_inside_group_nonSync.csv",
    )


# ===========================================================================
# Family 10 — method_with_fallback_sets
# (legacy: preprocessing/method_with_fallback_sets.py)
# ===========================================================================

# Single-element defaults from flextool/flextool_base.dat — mirror exactly.
_LIFETIME_METHOD_DEFAULT = "reinvest_automatic"
_CT_METHOD_REGULAR = "regular"
_CT_METHOD_CONSTANT = "constant_efficiency"
_STARTUP_METHOD_NO = "no_startup"
_INFLOW_METHOD_DEFAULT = "use_original"
_STORAGE_BINDING_METHOD_DEFAULT = "bind_forward_only"


def _per_entity_fallback(
    explicit: pl.DataFrame,
    entities: pl.DataFrame,
    default_for: dict[str, str | None],
    out_columns: tuple[str, str],
) -> pl.DataFrame:
    """Emit explicit rows + a default row for each entity without explicit.

    ``default_for`` maps entity → method-or-None.  ``None`` means
    "skip this entity" (mirrors the ``return ()`` branch in the legacy
    ``process__ct_method`` fallback for non-connection/non-unit
    processes).

    Order is ``entities.csv`` order; within an entity, explicit rows
    preserve their CSV order.
    """
    entity_col, method_col = out_columns
    explicit = explicit.rename({explicit.columns[0]: entity_col,
                                explicit.columns[1]: method_col})

    explicit_by_entity: dict[str, list[str]] = {}
    for e, m in explicit.iter_rows():
        explicit_by_entity.setdefault(e, []).append(m)

    rows: list[tuple[str, str]] = []
    for e in entities.get_column(entities.columns[0]).to_list():
        if e in explicit_by_entity:
            for m in explicit_by_entity[e]:
                rows.append((e, m))
        else:
            default = default_for.get(e)
            if default is not None:
                rows.append((e, default))
    return pl.DataFrame(
        {entity_col: [r[0] for r in rows], method_col: [r[1] for r in rows]},
        schema={entity_col: pl.Utf8, method_col: pl.Utf8},
    )


def derive_entity_lifetime_method(input_dir: Path) -> pl.DataFrame:
    explicit = _read_csv(
        input_dir / "entity__lifetime_method.csv", ["entity", "lifetime_method"],
    )
    explicit = _drop_blank_rows(explicit, ["entity", "lifetime_method"])
    entities = _read_csv(input_dir / "entity.csv", ["entity"])
    entities = _drop_blank_rows(entities, ["entity"])
    defaults = {e: _LIFETIME_METHOD_DEFAULT
                for e in entities.get_column("entity").to_list()}
    return _per_entity_fallback(
        explicit, entities, defaults, ("entity", "lifetime_method"),
    )


def derive_process_ct_method(input_dir: Path) -> pl.DataFrame:
    explicit = _read_csv(
        input_dir / "process__ct_method.csv", ["process", "ct_method"],
    )
    explicit = _drop_blank_rows(explicit, ["process", "ct_method"])
    processes = _read_csv(input_dir / "process.csv", ["process"])
    processes = _drop_blank_rows(processes, ["process"])
    connections = set(
        _drop_blank_rows(
            _read_csv(input_dir / "process_connection.csv", ["process"]), ["process"],
        ).get_column("process").to_list()
    )
    units = set(
        _drop_blank_rows(
            _read_csv(input_dir / "process_unit.csv", ["process"]), ["process"],
        ).get_column("process").to_list()
    )

    defaults: dict[str, str | None] = {}
    for p in processes.get_column("process").to_list():
        if p in connections:
            defaults[p] = _CT_METHOD_REGULAR
        elif p in units:
            defaults[p] = _CT_METHOD_CONSTANT
        else:
            defaults[p] = None
    return _per_entity_fallback(
        explicit, processes, defaults, ("process", "ct_method"),
    )


def derive_node_inflow_method(input_dir: Path) -> pl.DataFrame:
    explicit = _read_csv(
        input_dir / "node__inflow_method.csv", ["node", "inflow_method"],
    )
    explicit = _drop_blank_rows(explicit, ["node", "inflow_method"])
    nodes = _read_csv(input_dir / "node.csv", ["node"])
    nodes = _drop_blank_rows(nodes, ["node"])
    defaults = {n: _INFLOW_METHOD_DEFAULT
                for n in nodes.get_column("node").to_list()}
    return _per_entity_fallback(
        explicit, nodes, defaults, ("node", "inflow_method"),
    )


def derive_node_storage_binding_method(input_dir: Path) -> pl.DataFrame:
    explicit = _read_csv(
        input_dir / "node__storage_binding_method.csv",
        ["node", "storage_binding_method"],
    )
    explicit = _drop_blank_rows(explicit, ["node", "storage_binding_method"])
    nodes = _read_csv(input_dir / "node.csv", ["node"])
    nodes = _drop_blank_rows(nodes, ["node"])
    defaults = {n: _STORAGE_BINDING_METHOD_DEFAULT
                for n in nodes.get_column("node").to_list()}
    return _per_entity_fallback(
        explicit, nodes, defaults, ("node", "storage_binding_method"),
    )


def derive_process_startup_method(input_dir: Path) -> pl.DataFrame:
    explicit = _read_csv(
        input_dir / "process__startup_method.csv", ["process", "startup_method"],
    )
    explicit = _drop_blank_rows(explicit, ["process", "startup_method"])
    processes = _read_csv(input_dir / "process.csv", ["process"])
    processes = _drop_blank_rows(processes, ["process"])
    defaults = {p: _STARTUP_METHOD_NO
                for p in processes.get_column("process").to_list()}
    return _per_entity_fallback(
        explicit, processes, defaults, ("process", "startup_method"),
    )


def write_entity_lifetime_method(input_dir: Path, solve_data_dir: Path) -> None:
    _write(
        derive_entity_lifetime_method(input_dir),
        solve_data_dir / "entity__lifetime_method.csv",
    )


def write_process_ct_method(input_dir: Path, solve_data_dir: Path) -> None:
    _write(derive_process_ct_method(input_dir), solve_data_dir / "process__ct_method.csv")


def write_process_startup_method(input_dir: Path, solve_data_dir: Path) -> None:
    _write(
        derive_process_startup_method(input_dir),
        solve_data_dir / "process__startup_method.csv",
    )


def write_node_inflow_method(input_dir: Path, solve_data_dir: Path) -> None:
    _write(
        derive_node_inflow_method(input_dir),
        solve_data_dir / "node__inflow_method.csv",
    )


def write_node_storage_binding_method(input_dir: Path, solve_data_dir: Path) -> None:
    _write(
        derive_node_storage_binding_method(input_dir),
        solve_data_dir / "node__storage_binding_method.csv",
    )


# ===========================================================================
# Family 11 — invest_total_sets (legacy: preprocessing/invest_total_sets.py)
# ===========================================================================

_INVEST_TOTAL_METHODS: frozenset[str] = frozenset((
    "invest_total", "invest_period_total",
    "invest_retire_total", "invest_retire_period_total",
))
_RETIRE_TOTAL_METHODS: frozenset[str] = frozenset((
    "retire_total", "retire_period_total",
    "invest_retire_total", "invest_retire_period_total",
))
_CUMULATIVE_METHODS: frozenset[str] = frozenset(("cumulative_limits",))


def _entities_with_method_in(
    method_csv: Path, allowed: frozenset[str], col1: str,
) -> set[str]:
    df = _read_csv(method_csv, [col1, "method"])
    df = _drop_blank_rows(df, [col1, "method"])
    return set(
        df.filter(pl.col("method").is_in(list(allowed)))
          .get_column(col1).to_list()
    )


def _filter_singles(
    universe_csv: Path, with_method: set[str], col: str,
) -> pl.DataFrame:
    universe = _read_csv(universe_csv, [col])
    universe = _drop_blank_rows(universe, [col])
    return universe.filter(pl.col(col).is_in(list(with_method)))


def write_invest_total_sets(input_dir: Path, solve_data_dir: Path) -> None:
    entity_methods_csv = input_dir / "entity__invest_method.csv"
    group_methods_csv = input_dir / "group__invest_method.csv"

    e_with_invest = _entities_with_method_in(entity_methods_csv, _INVEST_TOTAL_METHODS, "entity")
    e_with_retire = _entities_with_method_in(entity_methods_csv, _RETIRE_TOTAL_METHODS, "entity")
    g_with_invest = _entities_with_method_in(group_methods_csv,  _INVEST_TOTAL_METHODS, "group")
    g_with_retire = _entities_with_method_in(group_methods_csv,  _RETIRE_TOTAL_METHODS, "group")
    g_with_cum    = _entities_with_method_in(group_methods_csv,  _CUMULATIVE_METHODS,   "group")

    _write(
        _filter_singles(solve_data_dir / "entityInvest.csv", e_with_invest, "entity"),
        solve_data_dir / "e_invest_total.csv",
    )
    _write(
        _filter_singles(solve_data_dir / "entityDivest.csv", e_with_retire, "entity"),
        solve_data_dir / "e_divest_total.csv",
    )
    _write(
        _filter_singles(solve_data_dir / "group_invest.csv", g_with_invest, "group"),
        solve_data_dir / "g_invest_total.csv",
    )
    _write(
        _filter_singles(solve_data_dir / "group_divest.csv", g_with_retire, "group"),
        solve_data_dir / "g_divest_total.csv",
    )
    _write(
        _filter_singles(solve_data_dir / "group_invest.csv", g_with_cum, "group"),
        solve_data_dir / "g_invest_cumulative.csv",
    )


def write_ci_ladder_cumulative(input_dir: Path, solve_data_dir: Path) -> None:
    """flextool.mod:2000 — commodity__tier_cum filtered to ladder-cumulative."""
    cum = _read_csv(
        input_dir / "commodity_ladder_cumulative.csv", ["commodity", "tier"],
    )
    cum = _drop_blank_rows(cum, ["commodity", "tier"])
    with_cum = _read_csv(
        solve_data_dir / "commodity_with_ladder_cumulative.csv", ["commodity"],
    )
    with_cum_set = (
        with_cum.filter(pl.col("commodity") != "")
                .get_column("commodity").to_list()
    )
    filtered = (
        cum.filter(pl.col("commodity").is_in(with_cum_set))
           .unique(maintain_order=True)
    )
    _write(filtered, solve_data_dir / "ci_ladder_cumulative.csv")


# ===========================================================================
# Family 12 — structural_filters (legacy: preprocessing/structural_filters.py)
# ===========================================================================

def derive_connection_param(input_dir: Path) -> pl.DataFrame:
    """connection__param = process__param filtered by p ∈ process_connection.

    p_process.csv layout: (process, processParam, value).  We project the
    first two columns and filter by the connection-process membership.
    """
    pp = _read_csv(input_dir / "p_process.csv", ["process", "processParam"])
    pp = _drop_blank_rows(pp, ["process", "processParam"])
    connections = _read_csv(input_dir / "process_connection.csv", ["process"])
    conn_set = (
        connections.filter(pl.col("process") != "")
                   .get_column("process").to_list()
    )
    return (
        pp.filter(pl.col("process").is_in(conn_set))
          .unique(maintain_order=True)
    )


def derive_nodegroup_dispatch_node(input_dir: Path) -> pl.DataFrame:
    """nodeGroupDispatch ∩ ``{g : ∃ n with (g, n) in group__node}``."""
    groups = _read_csv(input_dir / "nodeGroupDispatch.csv", ["group"])
    groups = _drop_blank_rows(groups, ["group"])
    gn = _read_csv(input_dir / "group__node.csv", ["group", "node"])
    gn = _drop_blank_rows(gn, ["group", "node"])
    groups_with_nodes = gn.select("group").unique()
    return (
        groups.join(groups_with_nodes, on="group", how="semi")
              .unique(maintain_order=True)
    )


def derive_commodity_node_co2(input_dir: Path) -> pl.DataFrame:
    """commodity_node filtered by ``p_commodity[c, 'co2_content'] != 0``.

    flextool.mod:2011.  ``default 0`` on p_commodity means commodities
    without an explicit ``co2_content`` row are excluded (0 is falsy).
    """
    cn = _read_csv(input_dir / "commodity__node.csv", ["commodity", "node"])
    cn = _drop_blank_rows(cn, ["commodity", "node"])
    p_commodity = _read_csv(
        input_dir / "p_commodity.csv", ["commodity", "param", "value"],
    )
    co2 = (
        p_commodity.filter(
            (pl.col("param") == "co2_content")
            & (pl.col("value") != "")
            & (pl.col("value").cast(pl.Float64, strict=False) != 0.0)
        )
        .select("commodity")
        .unique()
    )
    return (
        cn.join(co2, on="commodity", how="semi")
          .unique(maintain_order=True)
    )


def derive_process__commodity__node(input_dir: Path) -> pl.DataFrame:
    """process × commodity_node where (p, n) is an arc endpoint of p.

    flextool.mod:2009.  Iteration order: process.csv × commodity__node.csv.
    """
    processes = _read_csv(input_dir / "process.csv", ["process"])
    processes = _drop_blank_rows(processes, ["process"])
    cn = _read_csv(input_dir / "commodity__node.csv", ["commodity", "node"])
    cn = _drop_blank_rows(cn, ["commodity", "node"])
    sources = _read_csv(input_dir / "process__source.csv", ["process", "node"])
    sources = _drop_blank_rows(sources, ["process", "node"])
    sinks = _read_csv(input_dir / "process__sink.csv", ["process", "node"])
    sinks = _drop_blank_rows(sinks, ["process", "node"])

    # arc_endpoints[p] = set of nodes (source ∪ sink)
    arc_endpoints: dict[str, set[str]] = {}
    for p, n in sources.iter_rows():
        arc_endpoints.setdefault(p, set()).add(n)
    for p, n in sinks.iter_rows():
        arc_endpoints.setdefault(p, set()).add(n)

    cn_rows = list(cn.iter_rows())
    rows: list[tuple[str, str, str]] = []
    seen: set[tuple[str, str, str]] = set()
    for p in processes.get_column("process").to_list():
        nodes_for_p = arc_endpoints.get(p)
        if not nodes_for_p:
            continue
        for c, n in cn_rows:
            if n in nodes_for_p:
                key = (p, c, n)
                if key not in seen:
                    seen.add(key)
                    rows.append(key)
    return pl.DataFrame(
        {
            "process":   [r[0] for r in rows],
            "commodity": [r[1] for r in rows],
            "node":      [r[2] for r in rows],
        },
        schema={"process": pl.Utf8, "commodity": pl.Utf8, "node": pl.Utf8},
    )


def _derive_coeff_zero(
    arc_csv: Path, coef_csv: Path, second_col: str,
) -> pl.DataFrame:
    """(process, source/sink) rows whose max-capacity-coefficient is 0.

    Default of 1 on missing coefficients means only EXPLICITLY-zero rows
    appear in the output (matches the mod's truthy-check behaviour).
    """
    arcs = _read_csv(arc_csv, ["process", second_col])
    arcs = _drop_blank_rows(arcs, ["process", second_col])
    coef = _read_csv(coef_csv, ["process", second_col, "value"])
    zeros = (
        coef.filter(
            (pl.col("process") != "")
            & (pl.col(second_col) != "")
            & (pl.col("value") != "")
            & (pl.col("value").cast(pl.Float64, strict=False) == 0.0)
        )
        .select("process", second_col)
        .unique()
    )
    return (
        arcs.join(zeros, on=["process", second_col], how="semi")
            .unique(maintain_order=True)
    )


def derive_process_source_coeff_zero(input_dir: Path) -> pl.DataFrame:
    return _derive_coeff_zero(
        input_dir / "process__source.csv",
        input_dir / "p_process_source_max_capacity_coefficient.csv",
        "source",
    )


def derive_process_sink_coeff_zero(input_dir: Path) -> pl.DataFrame:
    return _derive_coeff_zero(
        input_dir / "process__sink.csv",
        input_dir / "p_process_sink_max_capacity_coefficient.csv",
        "sink",
    )


def write_connection_param(input_dir: Path, solve_data_dir: Path) -> None:
    _write(derive_connection_param(input_dir), solve_data_dir / "connection__param.csv")


def write_nodegroup_dispatch_node(input_dir: Path, solve_data_dir: Path) -> None:
    _write(
        derive_nodegroup_dispatch_node(input_dir),
        solve_data_dir / "nodeGroupDispatch_node.csv",
    )


def write_commodity_node_co2(input_dir: Path, solve_data_dir: Path) -> None:
    _write(
        derive_commodity_node_co2(input_dir),
        solve_data_dir / "commodity_node_co2.csv",
    )


def write_process__commodity__node(input_dir: Path, solve_data_dir: Path) -> None:
    _write(
        derive_process__commodity__node(input_dir),
        solve_data_dir / "process__commodity__node.csv",
    )


def write_process_coeff_zero_sets(input_dir: Path, solve_data_dir: Path) -> None:
    _write(
        derive_process_source_coeff_zero(input_dir),
        solve_data_dir / "process_source_coeff_zero.csv",
    )
    _write(
        derive_process_sink_coeff_zero(input_dir),
        solve_data_dir / "process_sink_coeff_zero.csv",
    )
