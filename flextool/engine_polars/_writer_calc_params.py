"""Writer-port Phase 1 (L7-L9) — calculated-param + process-method families.

Native polars port of the next batch of preprocessing modules invoked
from :func:`flextool.flextoolrunner.input_writer.write_input`
(lines 1908, 1919-1922).

Ported legacy modules (preprocessing/):

* ``entity_total_caps.py``     — 127 LOC — 4 e_*_total params keyed
  on entityInvest / entityDivest, summed across p_process[e, param]
  and p_node[e, param]; **MPS-precision-sensitive** (legacy emits
  ``repr(float(v))``, so we pre-stringify the value column with the
  same expression and then ``write_csv``).
* ``process_method_sets.py``   — 387 LOC — process-method-driven
  derived sets: 3 method-enum projections, ``process_VRE``, the
  ``process_*_to_*`` family of 10 method-gated cross-products,
  and 2 profile-method joins.

Total ~514 LOC of legacy code ported in this dispatch.

Each ``derive_*`` returns a fresh ``pl.DataFrame`` (in-memory
contract); ``write_*`` wrappers materialise the frame to the legacy
``solve_data/*.csv`` path so downstream consumers continue to read
identical bytes.

Style mirrors :mod:`._writer_leaf_sets` / :mod:`._writer_mid_sets`:
eager polars reads of tiny CSVs, expression chains,
``unique(maintain_order=True)`` for ordered dedup.

Precision-parity pattern (calculated-param families)
----------------------------------------------------

Legacy ``entity_total_caps`` writes ``f"{key},{repr(float(v))}\\n"``
per row.  ``repr(float)`` is round-trip-exact and is what the GMPL
parser then reads.  Polars' default ``write_csv`` float formatting
does *not* match ``repr`` for all values (it strips trailing zeros
differently for some doubles), so we explicitly pre-stringify the
value column with the same ``repr(float(v))`` step before writing.
This is verified to be byte-identical by the parity tests against
fixtures with explicit float values (e.g.
``work_network_coal_wind_battery_invest_cumulative`` has
``e_invest_max_total = 800.0``; legacy writes ``800.0``,
``repr(800.0) == '800.0'``, native matches).
"""
from __future__ import annotations

from pathlib import Path

import polars as pl


# ---------------------------------------------------------------------------
# CSV I/O — same conventions as _writer_{leaf,mid}_sets:
#   * eager read, missing file → empty frame with requested schema
#   * positional column rename (handle legacy headers that differ in label)
#   * empty frame still writes header line
# ---------------------------------------------------------------------------

def _read_csv(path: Path, columns: list[str],
              *, provider: "object | None" = None) -> pl.DataFrame:
    """Read a tiny flextool CSV with positional column rename.

    Step 1-g — Provider-first: when *provider* has a frame keyed on
    ``path``'s basename, return that frame after the same positional
    rename.  Falls back to the legacy seed lookup (still installed as
    the active seed during the migration window) and then disk.
    """
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
        schema_overrides={c: pl.Utf8 for c in columns},
        truncate_ragged_lines=True,
    )
    keep = df.columns[: len(columns)]
    df = df.select(keep)
    df.columns = columns
    return df


def _write(df: pl.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.write_csv(path)


def _drop_blank_rows(df: pl.DataFrame, required_cols: list[str]) -> pl.DataFrame:
    expr = pl.col(required_cols[0]) != ""
    for c in required_cols[1:]:
        expr = expr & (pl.col(c) != "")
    return df.filter(expr)


# ===========================================================================
# Family 13 — entity_total_caps  (legacy: preprocessing/entity_total_caps.py)
# ===========================================================================

# (output filename, source-keys CSV, p_*-table param name)
_ENTITY_TOTAL_SPEC: list[tuple[str, str, str]] = [
    ("e_invest_max_total.csv", "entityInvest.csv", "invest_max_total"),
    ("e_divest_max_total.csv", "entityDivest.csv", "retire_max_total"),
    ("e_invest_min_total.csv", "entityInvest.csv", "invest_min_total"),
    ("e_divest_min_total.csv", "entityDivest.csv", "retire_min_total"),
]


def _read_param_lookup(path: Path,
                        *, provider: "object | None" = None,
                        ) -> dict[tuple[str, str], float]:
    """Read a 3-col (entity, paramName, value) CSV into a python dict.

    Mirrors ``entity_total_caps._read_param_table``: silently skip
    rows whose value isn't parseable as float.  Keeping a python
    dict (rather than a polars frame) keeps the per-entity summation
    below straightforward — input tables are tiny.
    """
    df = _read_csv(path, ["entity", "paramName", "value"], provider=provider)
    out: dict[tuple[str, str], float] = {}
    for e, p, v in df.iter_rows():
        if not e or not p:
            continue
        try:
            out[(e, p)] = float(v)
        except (TypeError, ValueError):
            continue
    return out


def derive_entity_total_cap(
    keys: list[str],
    process_set: frozenset[str],
    node_set: frozenset[str],
    p_process: dict[tuple[str, str], float],
    p_node: dict[tuple[str, str], float],
    param_name: str,
) -> pl.DataFrame:
    """Compute a single ``(entity, value)`` table for one e_*_total param.

    ``value`` = ``p_process[e, param_name]`` (if e ∈ process, default 0)
    + ``p_node[e, param_name]`` (if e ∈ node, default 0).

    Returns a 2-column frame with both columns as ``pl.Utf8`` —
    ``value`` is pre-stringified with ``repr(float(v))`` to preserve
    bit-exact MPS-precision parity with legacy code.  See module
    docstring for the precision-parity rationale.
    """
    entities: list[str] = []
    values: list[str] = []
    for e in keys:
        v = 0.0
        if e in process_set:
            v += p_process.get((e, param_name), 0.0)
        if e in node_set:
            v += p_node.get((e, param_name), 0.0)
        entities.append(e)
        values.append(repr(v))
    return pl.DataFrame(
        {"entity": entities, "value": values},
        schema={"entity": pl.Utf8, "value": pl.Utf8},
    )


def write_entity_total_caps(input_dir: Path, solve_data_dir: Path,
                              *, provider: "object | None" = None,
                              ) -> None:
    """Write all four ``e_*_total`` params keyed on entityInvest/Divest."""
    process_set = frozenset(
        _drop_blank_rows(_read_csv(input_dir / "process.csv", ["process"],
                                     provider=provider), ["process"])
        .get_column("process").to_list()
    )
    node_set = frozenset(
        _drop_blank_rows(_read_csv(input_dir / "node.csv", ["node"],
                                     provider=provider), ["node"])
        .get_column("node").to_list()
    )
    p_process = _read_param_lookup(input_dir / "p_process.csv",
                                     provider=provider)
    p_node = _read_param_lookup(input_dir / "p_node.csv", provider=provider)

    # Cache the entity-key lists per source CSV (entityInvest is reused
    # by both invest_max and invest_min, same for entityDivest).
    key_cache: dict[str, list[str]] = {}
    for _, src, _ in _ENTITY_TOTAL_SPEC:
        if src not in key_cache:
            df = _read_csv(solve_data_dir / src, ["entity"], provider=provider)
            df = _drop_blank_rows(df, ["entity"])
            key_cache[src] = df.get_column("entity").to_list()

    for fname, src, param in _ENTITY_TOTAL_SPEC:
        out = derive_entity_total_cap(
            key_cache[src], process_set, node_set,
            p_process, p_node, param,
        )
        _write(out, solve_data_dir / fname)


# ===========================================================================
# Family 14 — process_method_sets (legacy: preprocessing/process_method_sets.py)
# ===========================================================================

# Method-enum subsets mirrored from
# ``flextool.flextoolrunner.preprocessing._method_constants``.  Pinned
# here as frozensets so the native module has no transitive imports
# from the legacy preprocessing tree.  If the legacy constants change,
# update both sites in lockstep (the parity tests would catch drift).

_METHOD_LP: frozenset[str] = frozenset((
    "method_1way_1var_LP", "method_1way_nvar_LP",
))
_METHOD_MIP: frozenset[str] = frozenset((
    "method_1way_1var_MIP", "method_1way_nvar_MIP",
    "method_2way_2var_MIP_exclude",
))
_METHOD_INDIRECT: frozenset[str] = frozenset((
    "method_1way_nvar_off", "method_1way_nvar_LP",
    "method_1way_nvar_MIP", "method_2way_nvar_off",
))
_METHOD_DIRECT: frozenset[str] = frozenset((
    "method_1way_1var_off", "method_1way_1var_LP", "method_1way_1var_MIP",
    "method_2way_1var_off", "method_2way_2var_off",
    "method_2way_2var_exclude", "method_2way_2var_MIP_exclude",
))
_METHOD_2WAY_1VAR: frozenset[str] = frozenset(("method_2way_1var_off",))
_METHOD_2WAY_2VAR: frozenset[str] = frozenset((
    "method_2way_2var_off", "method_2way_2var_exclude",
    "method_2way_2var_MIP_exclude",
))
_METHOD_2WAY_NVAR: frozenset[str] = frozenset(("method_2way_nvar_off",))
_METHOD_1WAY_1VAR: frozenset[str] = frozenset((
    "method_1way_1var_off", "method_1way_1var_LP", "method_1way_1var_MIP",
))


# ---- process-method projections (mod L1121-L1194) -------------------------

def derive_process_online_linear(input_dir: Path,
                                   *, provider: "object | None" = None,
                                   ) -> pl.DataFrame:
    """``process_online_linear`` = projection of process_method onto
    rows whose method ∈ METHOD_LP."""
    pm = _read_csv(input_dir / "process_method.csv", ["process", "method"],
                    provider=provider)
    pm = _drop_blank_rows(pm, ["process", "method"])
    return (
        pm.filter(pl.col("method").is_in(list(_METHOD_LP)))
          .select("process")
          .unique(maintain_order=True)
    )


def derive_process_online_integer(input_dir: Path,
                                    *, provider: "object | None" = None,
                                    ) -> pl.DataFrame:
    """``process_online_integer`` = METHOD_MIP filter on process_method."""
    pm = _read_csv(input_dir / "process_method.csv", ["process", "method"],
                    provider=provider)
    pm = _drop_blank_rows(pm, ["process", "method"])
    return (
        pm.filter(pl.col("method").is_in(list(_METHOD_MIP)))
          .select("process")
          .unique(maintain_order=True)
    )


def derive_process_method_indirect(input_dir: Path,
                                     *, provider: "object | None" = None,
                                     ) -> pl.DataFrame:
    """``process__method_indirect`` = METHOD_INDIRECT filter, both columns kept."""
    pm = _read_csv(input_dir / "process_method.csv", ["process", "method"],
                    provider=provider)
    pm = _drop_blank_rows(pm, ["process", "method"])
    return (
        pm.filter(pl.col("method").is_in(list(_METHOD_INDIRECT)))
          .unique(maintain_order=True)
    )


def write_process_method_projections(input_dir: Path, solve_data_dir: Path,
                                       *, provider: "object | None" = None,
                                       ) -> None:
    _write(derive_process_online_linear(input_dir, provider=provider),
           solve_data_dir / "process_online_linear.csv")
    _write(derive_process_online_integer(input_dir, provider=provider),
           solve_data_dir / "process_online_integer.csv")
    _write(derive_process_method_indirect(input_dir, provider=provider),
           solve_data_dir / "process__method_indirect.csv")


# ---- process_VRE (mod L2248) ----------------------------------------------

def derive_process_VRE(input_dir: Path,
                         *, provider: "object | None" = None,
                         ) -> pl.DataFrame:
    """``process_VRE`` = process_unit ∩ no-source ∩ has-upper-limit-profile.

    flextool.mod:2248 — VRE units have no source arc (free-energy
    primary input) and at least one ``upper_limit`` profile method.
    """
    units = _read_csv(input_dir / "process_unit.csv", ["process"],
                       provider=provider)
    units = _drop_blank_rows(units, ["process"])
    sources = _read_csv(input_dir / "process__source.csv",
                         ["process", "source"], provider=provider)
    sources = _drop_blank_rows(sources, ["process", "source"])
    profiles = _read_csv(
        input_dir / "process__node__profile__profile_method.csv",
        ["process", "node", "profile", "profile_method"],
        provider=provider,
    )
    profiles = _drop_blank_rows(
        profiles, ["process", "node", "profile", "profile_method"],
    )
    has_source = frozenset(sources.get_column("process").to_list())
    has_upper_limit = frozenset(
        profiles.filter(pl.col("profile_method") == "upper_limit")
                .get_column("process").to_list()
    )
    return (
        units.filter(
            ~pl.col("process").is_in(list(has_source))
            & pl.col("process").is_in(list(has_upper_limit))
        )
        .select("process")
        .unique(maintain_order=True)
    )


def write_process_VRE(input_dir: Path, solve_data_dir: Path,
                        *, provider: "object | None" = None,
                        ) -> None:
    _write(derive_process_VRE(input_dir, provider=provider),
           solve_data_dir / "process_VRE.csv")


# ---- process_*_to_* family (mod L993-L1052) -------------------------------
#
# Each entry below is a method-enum-existence join.  We iterate a base
# 2-tuple set (process_source, process_sink, or process) and admit rows
# whose process has at least one method in a specific enum subset.
#
# Output shape is dimen-3:  (process_outer, process, source/sink) or
# (process, source/sink, process_aux).  The legacy module fixes the
# header column names per output — we mirror those exactly.
# ---------------------------------------------------------------------------


def _processes_with_method_in(
    pm: pl.DataFrame, allowed: frozenset[str],
) -> frozenset[str]:
    return frozenset(
        pm.filter(pl.col("method").is_in(list(allowed)))
          .get_column("process").to_list()
    )


def _build_arc_map(arc_df: pl.DataFrame, key: str, value: str) -> dict[str, list[str]]:
    """Group an arc CSV by ``key`` preserving CSV order."""
    out: dict[str, list[str]] = {}
    for k, v in arc_df.select(key, value).iter_rows():
        out.setdefault(k, []).append(v)
    return out


def _arc_method_inputs(input_dir: Path,
                        *, provider: "object | None" = None,
                        ) -> dict:
    """Shared scan for the 10 process_*_to_* derives.

    Returns the bundle of in-memory sets/lists used by every
    ``derive_process_*`` below.  Each public derive_X also calls this
    helper directly so it remains standalone for accumulator capture.
    """
    pm = _drop_blank_rows(
        _read_csv(input_dir / "process_method.csv", ["process", "method"],
                   provider=provider),
        ["process", "method"],
    )
    sources = _drop_blank_rows(
        _read_csv(input_dir / "process__source.csv", ["process", "source"],
                   provider=provider),
        ["process", "source"],
    )
    sinks = _drop_blank_rows(
        _read_csv(input_dir / "process__sink.csv", ["process", "sink"],
                   provider=provider),
        ["process", "sink"],
    )
    processes = _drop_blank_rows(
        _read_csv(input_dir / "process.csv", ["process"], provider=provider),
        ["process"],
    ).get_column("process").to_list()

    has_source = frozenset(sources.get_column("process").to_list())
    has_sink = frozenset(sinks.get_column("process").to_list())

    return {
        "pm": pm,
        "sources": sources,
        "sinks": sinks,
        "processes": processes,
        "p_with_2way_nvar": _processes_with_method_in(pm, _METHOD_2WAY_NVAR),
        "p_with_direct": _processes_with_method_in(pm, _METHOD_DIRECT),
        "p_with_2way_2var": _processes_with_method_in(pm, _METHOD_2WAY_2VAR),
        "p_with_1way_1var": _processes_with_method_in(pm, _METHOD_1WAY_1VAR),
        "has_source": has_source,
        "has_sink": has_sink,
        "process_no_source": frozenset(p for p in processes if p not in has_source),
        "process_no_sink": frozenset(p for p in processes if p not in has_sink),
        "sinks_by_process": _build_arc_map(sinks, "process", "sink"),
        "sources_by_process": _build_arc_map(sources, "process", "source"),
        "sink_rows": list(sinks.iter_rows()),
        "source_rows": list(sources.iter_rows()),
    }


def _to_frame(rows: list[tuple[str, ...]],
              header: tuple[str, ...]) -> pl.DataFrame:
    """Dedup + materialise to a pl.DataFrame with all-Utf8 columns."""
    deduped = list(dict.fromkeys(rows))
    cols = {h: [r[i] for r in deduped] for i, h in enumerate(header)}
    return pl.DataFrame(cols, schema={h: pl.Utf8 for h in header})


# ---- 10 derive_X for the process_*_to_* family ----

def derive_process_sink_toProcess(input_dir: Path,
                                    *, provider: "object | None" = None,
                                    ) -> pl.DataFrame:
    """``process_sink_toProcess`` — METHOD_2WAY_NVAR filter on (p, sink)."""
    inp = _arc_method_inputs(input_dir, provider=provider)
    rows = [
        (p, sink, p)
        for p, sink in inp["sink_rows"]
        if p in inp["p_with_2way_nvar"]
    ]
    return _to_frame(rows, ("process", "sink", "process_aux"))


def derive_process_process_toSource(input_dir: Path,
                                      *, provider: "object | None" = None,
                                      ) -> pl.DataFrame:
    """``process_process_toSource`` — METHOD_2WAY_NVAR filter on (p, source)."""
    inp = _arc_method_inputs(input_dir, provider=provider)
    rows = [
        (p, p, source)
        for p, source in inp["source_rows"]
        if p in inp["p_with_2way_nvar"]
    ]
    return _to_frame(rows, ("process_outer", "process", "source"))


def derive_process_source_toSink(input_dir: Path,
                                   *, provider: "object | None" = None,
                                   ) -> pl.DataFrame:
    """``process_source_toSink`` — METHOD_DIRECT cross-product
    of source rows and sinks_by_process."""
    inp = _arc_method_inputs(input_dir, provider=provider)
    rows = [
        (p, source, sink)
        for p, source in inp["source_rows"]
        if p in inp["p_with_direct"]
        for sink in inp["sinks_by_process"].get(p, ())
    ]
    return _to_frame(rows, ("process", "source", "sink"))


def derive_process_source_toProcess_direct(input_dir: Path,
                                              *, provider: "object | None" = None,
                                              ) -> pl.DataFrame:
    """``process_source_toProcess_direct`` — METHOD_DIRECT on (p, source)."""
    inp = _arc_method_inputs(input_dir, provider=provider)
    rows = [
        (p, source, p)
        for p, source in inp["source_rows"]
        if p in inp["p_with_direct"]
    ]
    return _to_frame(rows, ("process", "source", "process_aux"))


def derive_process_process_toSink_direct(input_dir: Path,
                                            *, provider: "object | None" = None,
                                            ) -> pl.DataFrame:
    """``process_process_toSink_direct`` — METHOD_DIRECT on (p, sink)."""
    inp = _arc_method_inputs(input_dir, provider=provider)
    rows = [
        (p, p, sink)
        for p, sink in inp["sink_rows"]
        if p in inp["p_with_direct"]
    ]
    return _to_frame(rows, ("process_outer", "process", "sink"))


def derive_process_sink_toProcess_direct(input_dir: Path,
                                            *, provider: "object | None" = None,
                                            ) -> pl.DataFrame:
    """``process_sink_toProcess_direct`` — METHOD_2WAY_2VAR on (p, sink)."""
    inp = _arc_method_inputs(input_dir, provider=provider)
    rows = [
        (p, sink, p)
        for p, sink in inp["sink_rows"]
        if p in inp["p_with_2way_2var"]
    ]
    return _to_frame(rows, ("process", "sink", "process_aux"))


def derive_process_sink_toSource(input_dir: Path,
                                   *, provider: "object | None" = None,
                                   ) -> pl.DataFrame:
    """``process_sink_toSource`` — METHOD_2WAY_2VAR cross-product
    of sink rows and sources_by_process."""
    inp = _arc_method_inputs(input_dir, provider=provider)
    rows = [
        (p, sink, source)
        for p, sink in inp["sink_rows"]
        if p in inp["p_with_2way_2var"]
        for source in inp["sources_by_process"].get(p, ())
    ]
    return _to_frame(rows, ("process", "sink", "source"))


def derive_process_process_toSink_noConversion(input_dir: Path,
                                                  *, provider: "object | None" = None,
                                                  ) -> pl.DataFrame:
    """``process_process_toSink_noConversion`` — METHOD_1WAY_1VAR ∧ no source."""
    inp = _arc_method_inputs(input_dir, provider=provider)
    rows = [
        (p, p, sink)
        for p, sink in inp["sink_rows"]
        if p in inp["p_with_1way_1var"] and p in inp["process_no_source"]
    ]
    return _to_frame(rows, ("process_outer", "process", "sink"))


def derive_process_source_toProcess_noConversion(input_dir: Path,
                                                    *, provider: "object | None" = None,
                                                    ) -> pl.DataFrame:
    """``process_source_toProcess_noConversion`` — METHOD_1WAY_1VAR ∧ no sink."""
    inp = _arc_method_inputs(input_dir, provider=provider)
    rows = [
        (p, source, p)
        for p, source in inp["source_rows"]
        if p in inp["p_with_1way_1var"] and p in inp["process_no_sink"]
    ]
    return _to_frame(rows, ("process", "source", "process_aux"))


def derive_process_process_toSource_direct(input_dir: Path,
                                              *, provider: "object | None" = None,
                                              ) -> pl.DataFrame:
    """``process_process_toSource_direct`` — METHOD_2WAY_2VAR on (p, source)."""
    inp = _arc_method_inputs(input_dir, provider=provider)
    rows = [
        (p, p, source)
        for p, source in inp["source_rows"]
        if p in inp["p_with_2way_2var"]
    ]
    return _to_frame(rows, ("process_outer", "process", "source"))


def write_process_arc_method_joins(input_dir: Path, solve_data_dir: Path,
                                     *, provider: "object | None" = None,
                                     ) -> None:
    """Emit the 10 method-gated process_*_to_* tables.

    Iteration order mirrors the legacy module exactly so that the
    deduped output preserves the same first-occurrence ordering.
    Each emitted CSV goes through ``_write(derive_X(...), path)`` so
    Phase E-b's accumulator captures every frame.
    """
    _write(derive_process_sink_toProcess(input_dir, provider=provider),
           solve_data_dir / "process_sink_toProcess.csv")
    _write(derive_process_process_toSource(input_dir, provider=provider),
           solve_data_dir / "process_process_toSource.csv")
    _write(derive_process_source_toSink(input_dir, provider=provider),
           solve_data_dir / "process_source_toSink.csv")
    _write(derive_process_source_toProcess_direct(input_dir,
                                                    provider=provider),
           solve_data_dir / "process_source_toProcess_direct.csv")
    _write(derive_process_process_toSink_direct(input_dir, provider=provider),
           solve_data_dir / "process_process_toSink_direct.csv")
    _write(derive_process_sink_toProcess_direct(input_dir, provider=provider),
           solve_data_dir / "process_sink_toProcess_direct.csv")
    _write(derive_process_sink_toSource(input_dir, provider=provider),
           solve_data_dir / "process_sink_toSource.csv")
    _write(derive_process_process_toSink_noConversion(input_dir,
                                                        provider=provider),
           solve_data_dir / "process_process_toSink_noConversion.csv")
    _write(derive_process_source_toProcess_noConversion(input_dir,
                                                          provider=provider),
           solve_data_dir / "process_source_toProcess_noConversion.csv")
    _write(derive_process_process_toSource_direct(input_dir,
                                                    provider=provider),
           solve_data_dir / "process_process_toSource_direct.csv")


# ---- profile-method joins (mod L961, L969) --------------------------------

def _profile_method_inputs(input_dir: Path,
                             *, provider: "object | None" = None,
                             ) -> tuple[
    pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame, list[str],
    frozenset[str], frozenset[str], frozenset[str],
    dict[str, set[str]], dict[str, set[str]],
    list[tuple[str, str, str, str]],
]:
    """Shared scan for both profile-method join derives.

    Each public ``derive_*`` calls this so it is standalone for
    accumulator capture; the wrapper :func:`write_process_profile_method_joins`
    calls it once and feeds the same scan to both derives.
    """
    pm = _drop_blank_rows(
        _read_csv(input_dir / "process_method.csv", ["process", "method"],
                   provider=provider),
        ["process", "method"],
    )
    sources = _drop_blank_rows(
        _read_csv(input_dir / "process__source.csv", ["process", "source"],
                   provider=provider),
        ["process", "source"],
    )
    sinks = _drop_blank_rows(
        _read_csv(input_dir / "process__sink.csv", ["process", "sink"],
                   provider=provider),
        ["process", "sink"],
    )
    profiles = _drop_blank_rows(
        _read_csv(
            input_dir / "process__node__profile__profile_method.csv",
            ["process", "node", "profile", "profile_method"],
            provider=provider,
        ),
        ["process", "node", "profile", "profile_method"],
    )
    processes = _drop_blank_rows(
        _read_csv(input_dir / "process.csv", ["process"], provider=provider),
        ["process"],
    ).get_column("process").to_list()

    p_with_indirect = _processes_with_method_in(pm, _METHOD_INDIRECT)
    has_sources = frozenset(sources.get_column("process").to_list())
    has_sinks = frozenset(sinks.get_column("process").to_list())

    sinks_by_process: dict[str, set[str]] = {}
    for p, n in sinks.iter_rows():
        sinks_by_process.setdefault(p, set()).add(n)
    sources_by_process: dict[str, set[str]] = {}
    for p, n in sources.iter_rows():
        sources_by_process.setdefault(p, set()).add(n)

    profiles_rows = list(profiles.iter_rows())  # (p, n, f, fm) tuples

    return (
        pm, sources, sinks, profiles, processes,
        p_with_indirect, has_sources, has_sinks,
        sinks_by_process, sources_by_process,
        profiles_rows,
    )


def derive_process_profileProcess_toSink_profile_profile_method(
    input_dir: Path,
    *, provider: "object | None" = None,
) -> pl.DataFrame:
    """``process__profileProcess__toSink__profile__profile_method``:
    join profile rows against (process, sink) arcs, gated by
    "process has any indirect method OR process has no source rows".

    Iteration order mirrors the legacy module exactly so first-seen
    dedup preserves the legacy CSV row order.
    """
    (_pm, _sources, _sinks, _profiles, processes,
     p_with_indirect, has_sources, _has_sinks,
     sinks_by_process, _sources_by_process,
     profiles_rows) = _profile_method_inputs(input_dir, provider=provider)

    rows_to_sink: list[tuple[str, str, str, str, str]] = []
    for p in processes:
        if not (p in p_with_indirect or p not in has_sources):
            continue
        psinks = sinks_by_process.get(p, set())
        for p2, n, f, fm in profiles_rows:
            if p2 == p and n in psinks:
                rows_to_sink.append((p, p2, n, f, fm))
    deduped = list(dict.fromkeys(rows_to_sink))
    return pl.DataFrame(
        {
            "process_outer":  [r[0] for r in deduped],
            "process":        [r[1] for r in deduped],
            "sink":           [r[2] for r in deduped],
            "profile":        [r[3] for r in deduped],
            "profile_method": [r[4] for r in deduped],
        },
        schema={
            "process_outer": pl.Utf8, "process": pl.Utf8, "sink": pl.Utf8,
            "profile": pl.Utf8, "profile_method": pl.Utf8,
        },
    )


def derive_process_source_toProfileProcess_profile_profile_method(
    input_dir: Path,
    *, provider: "object | None" = None,
) -> pl.DataFrame:
    """``process__source__toProfileProcess__profile__profile_method``:
    join profile rows against (process, source) arcs, gated by
    "process has any indirect method OR process has no sink rows"."""
    (_pm, sources, _sinks, _profiles, _processes,
     p_with_indirect, _has_sources, has_sinks,
     _sinks_by_process, _sources_by_process,
     profiles_rows) = _profile_method_inputs(input_dir, provider=provider)

    rows_to_source: list[tuple[str, str, str, str, str]] = []
    for p, source in sources.iter_rows():
        if not (p in p_with_indirect or p not in has_sinks):
            continue
        for p2, src2, f, fm in profiles_rows:
            if p2 == p and src2 == source:
                rows_to_source.append((p, source, p2, f, fm))
    deduped = list(dict.fromkeys(rows_to_source))
    return pl.DataFrame(
        {
            "process":        [r[0] for r in deduped],
            "source":         [r[1] for r in deduped],
            "process_aux":    [r[2] for r in deduped],
            "profile":        [r[3] for r in deduped],
            "profile_method": [r[4] for r in deduped],
        },
        schema={
            "process": pl.Utf8, "source": pl.Utf8, "process_aux": pl.Utf8,
            "profile": pl.Utf8, "profile_method": pl.Utf8,
        },
    )


def write_process_profile_method_joins(
    input_dir: Path, solve_data_dir: Path,
    *, provider: "object | None" = None,
) -> None:
    """Two profile-method joins:

    * ``process__profileProcess__toSink__profile__profile_method``
    * ``process__source__toProfileProcess__profile__profile_method``
    """
    _write(
        derive_process_profileProcess_toSink_profile_profile_method(
            input_dir, provider=provider),
        solve_data_dir
        / "process__profileProcess__toSink__profile__profile_method.csv",
    )
    _write(
        derive_process_source_toProfileProcess_profile_profile_method(
            input_dir, provider=provider),
        solve_data_dir
        / "process__source__toProfileProcess__profile__profile_method.csv",
    )
