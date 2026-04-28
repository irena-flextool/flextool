"""Process-method-driven derived sets.

A grab-bag of L0 sets that all key off ``process_method`` (the (process,
method) pairs loaded from input/process_method.csv) plus the method-
enum subset constants in _method_constants.py.

Migrated:
    L1121 process_online_linear   = setof p WHERE m in method_LP
    L1122 process_online_integer  = setof p WHERE m in method_MIP
    L1194 process__method_indirect = filter (p, m) WHERE m in method_indirect
    L2248 process_VRE              = process_unit ∩ no-source ∩ has-upper-limit-profile
    L993, 999, 1005, 1010, 1021, 1033, 1046, 1052, plus the L1015 process_process_toSink_direct (look up actual line)
        = the process_*_to_* family — all are cross-products of
          process_source/sink with process or node, gated by a
          method-enum existential check on (p, method).
    L961 process__profileProcess__toSink__profile__profile_method
    L969 process__source__toProfileProcess__profile__profile_method
        = profile-method joins.
"""
from __future__ import annotations

import csv
from pathlib import Path
from collections.abc import Iterable

from flextool.flextoolrunner.preprocessing._method_constants import (
    METHOD_LP, METHOD_MIP, METHOD_INDIRECT,
    METHOD_2WAY_NVAR, METHOD_2WAY_2VAR, METHOD_DIRECT, METHOD_1WAY_1VAR,
)


def _read_pairs(path: Path) -> list[tuple[str, str]]:
    if not path.exists():
        return []
    out: list[tuple[str, str]] = []
    with path.open() as fh:
        reader = csv.reader(fh)
        next(reader, None)
        for row in reader:
            if len(row) >= 2 and row[0] and row[1]:
                out.append((row[0], row[1]))
    return out


def _read_singles(path: Path) -> list[str]:
    if not path.exists():
        return []
    with path.open() as fh:
        reader = csv.reader(fh)
        next(reader, None)
        return [r[0] for r in reader if r and r[0]]


def _read_quad(path: Path) -> list[tuple[str, str, str, str]]:
    """4-col CSV reader for process__node__profile__profile_method-type files."""
    if not path.exists():
        return []
    out: list[tuple[str, str, str, str]] = []
    with path.open() as fh:
        reader = csv.reader(fh)
        next(reader, None)
        for row in reader:
            if len(row) >= 4 and all(row[:4]):
                out.append((row[0], row[1], row[2], row[3]))
    return out


def _write_singles(path: Path, header: str, rows: Iterable[str]) -> None:
    path.write_text(header + "\n" + "".join(r + "\n" for r in rows))


def _write_pairs(path: Path, header: tuple[str, str],
                 rows: Iterable[tuple[str, str]]) -> None:
    path.write_text(",".join(header) + "\n"
                    + "".join(f"{a},{b}\n" for a, b in rows))


def _write_triples(path: Path, header: tuple[str, str, str],
                   rows: Iterable[tuple[str, str, str]]) -> None:
    path.write_text(",".join(header) + "\n"
                    + "".join(f"{a},{b},{c}\n" for a, b, c in rows))


def _write_quads(path: Path, header: tuple[str, str, str, str],
                 rows: Iterable[tuple[str, str, str, str]]) -> None:
    path.write_text(",".join(header) + "\n"
                    + "".join(f"{a},{b},{c},{d}\n" for a, b, c, d in rows))


# ---------------------------------------------------------------------------
# Process-method projections
# ---------------------------------------------------------------------------


def write_process_method_projections(input_dir: Path, solve_data_dir: Path) -> None:
    """process_online_linear / _integer / process__method_indirect."""
    pm = _read_pairs(input_dir / "process_method.csv")
    # process_online_linear
    online_linear = list(dict.fromkeys(p for p, m in pm if m in METHOD_LP))
    _write_singles(solve_data_dir / "process_online_linear.csv",
                   "process", online_linear)
    # process_online_integer
    online_integer = list(dict.fromkeys(p for p, m in pm if m in METHOD_MIP))
    _write_singles(solve_data_dir / "process_online_integer.csv",
                   "process", online_integer)
    # process__method_indirect — keep both columns
    indirect = [(p, m) for p, m in pm if m in METHOD_INDIRECT]
    _write_pairs(solve_data_dir / "process__method_indirect.csv",
                 ("process", "method"),
                 list(dict.fromkeys(indirect)))


# ---------------------------------------------------------------------------
# process_VRE
# ---------------------------------------------------------------------------


def write_process_VRE(input_dir: Path, solve_data_dir: Path) -> None:
    """process in process_unit with no source AND has 'upper_limit' profile.

    flextool.mod:2248:
        set process_VRE := {p in process_unit : sum{(p, source) in process_source} 1 == 0
                                                && (sum{(p, n, prof, m) in process__node__profile__profile_method : m = 'upper_limit'} 1)};
    """
    units = _read_singles(input_dir / "process_unit.csv")
    sources = _read_pairs(input_dir / "process__source.csv")
    profiles = _read_quad(input_dir / "process__node__profile__profile_method.csv")

    has_source: dict[str, bool] = {}
    for p, _ in sources:
        has_source[p] = True
    has_upper_limit_profile: dict[str, bool] = {}
    for p, _, _, m in profiles:
        if m == "upper_limit":
            has_upper_limit_profile[p] = True

    rows = [p for p in units
            if not has_source.get(p) and has_upper_limit_profile.get(p)]
    _write_singles(solve_data_dir / "process_VRE.csv", "process",
                   list(dict.fromkeys(rows)))


# ---------------------------------------------------------------------------
# process_*_to_* family — cross products of arc-side sets gated by a
# method-enum existential check on (p, method).
# ---------------------------------------------------------------------------


def _processes_with_any_method_in(
    pm: list[tuple[str, str]], allowed: frozenset[str]
) -> frozenset[str]:
    return frozenset(p for p, m in pm if m in allowed)


def write_process_arc_method_joins(input_dir: Path, solve_data_dir: Path) -> None:
    """All process_*_to_* sets that are method-enum existence joins.

    Each iterates a base 2-tuple set (process_source, process_sink, or
    process) and admits rows whose process has at least one method in
    a specific enum subset.
    """
    pm = _read_pairs(input_dir / "process_method.csv")
    sources = _read_pairs(input_dir / "process__source.csv")
    sinks = _read_pairs(input_dir / "process__sink.csv")
    processes = _read_singles(input_dir / "process.csv")

    p_with_2way_nvar = _processes_with_any_method_in(pm, METHOD_2WAY_NVAR)
    p_with_direct   = _processes_with_any_method_in(pm, METHOD_DIRECT)
    p_with_2way_2var = _processes_with_any_method_in(pm, METHOD_2WAY_2VAR)
    p_with_1way_1var = _processes_with_any_method_in(pm, METHOD_1WAY_1VAR)

    has_source = frozenset(p for p, _ in sources)
    has_sink = frozenset(p for p, _ in sinks)

    # Helper for "p has no source rows" / "p has no sink rows" used by
    # the noConversion variants.
    process_no_source = frozenset(p for p in processes if p not in has_source)
    process_no_sink = frozenset(p for p in processes if p not in has_sink)

    # ---- process_sink_toProcess (mod L993) ----
    # (p, sink) in process_sink, p2 in process : p == p2 AND p has 2way_nvar method
    # The condition `p == p2` collapses the cross-product so each (p, sink, p)
    # row satisfying the method check appears once. Output is dimen 3:
    # (process, sink, process_aux). To match mod's emission shape exactly, we
    # mirror the order it would have iterated.
    rows_sink_toProcess: list[tuple[str, str, str]] = [
        (p, sink, p)
        for p, sink in sinks
        if p in p_with_2way_nvar
    ]
    _write_triples(solve_data_dir / "process_sink_toProcess.csv",
                   ("process", "sink", "process_aux"),
                   list(dict.fromkeys(rows_sink_toProcess)))

    # ---- process_process_toSource (mod L999) ----
    # p in process, (p2, source) in process_source : p == p2 AND has 2way_nvar
    rows_process_toSource: list[tuple[str, str, str]] = [
        (p, p, source)
        for p, source in sources
        if p in p_with_2way_nvar
    ]
    _write_triples(solve_data_dir / "process_process_toSource.csv",
                   ("process_outer", "process", "source"),
                   list(dict.fromkeys(rows_process_toSource)))

    # ---- process_source_toSink (mod L1005) ----
    # (p, source) in process_source, sink in node : (p, sink) in process_sink
    # AND has direct method. The cross-product over `node` is filtered to
    # those that ARE sinks of p — yielding a (p, source, sink) row only for
    # actual existing arcs of that process.
    sinks_by_process: dict[str, list[str]] = {}
    for p, sink in sinks:
        sinks_by_process.setdefault(p, []).append(sink)
    rows_source_toSink: list[tuple[str, str, str]] = [
        (p, source, sink)
        for p, source in sources
        if p in p_with_direct
        for sink in sinks_by_process.get(p, ())
    ]
    _write_triples(solve_data_dir / "process_source_toSink.csv",
                   ("process", "source", "sink"),
                   list(dict.fromkeys(rows_source_toSink)))

    # ---- process_source_toProcess_direct (mod L1010) ----
    # (p, source) in process_source, p2 in process : p == p2 AND has direct
    rows_source_toProcess_direct: list[tuple[str, str, str]] = [
        (p, source, p)
        for p, source in sources
        if p in p_with_direct
    ]
    _write_triples(solve_data_dir / "process_source_toProcess_direct.csv",
                   ("process", "source", "process_aux"),
                   list(dict.fromkeys(rows_source_toProcess_direct)))

    # ---- process_process_toSink_direct (mod L1015 region) ----
    # p in process, (p2, sink) in process_sink : p == p2 AND has direct
    rows_process_toSink_direct: list[tuple[str, str, str]] = [
        (p, p, sink)
        for p, sink in sinks
        if p in p_with_direct
    ]
    _write_triples(solve_data_dir / "process_process_toSink_direct.csv",
                   ("process_outer", "process", "sink"),
                   list(dict.fromkeys(rows_process_toSink_direct)))

    # ---- process_sink_toProcess_direct (mod L1021) ----
    # (p, sink) in process_sink, p2 in process : p == p2 AND has 2way_2var
    rows_sink_toProcess_direct: list[tuple[str, str, str]] = [
        (p, sink, p)
        for p, sink in sinks
        if p in p_with_2way_2var
    ]
    _write_triples(solve_data_dir / "process_sink_toProcess_direct.csv",
                   ("process", "sink", "process_aux"),
                   list(dict.fromkeys(rows_sink_toProcess_direct)))

    # ---- process_sink_toSource (mod L1033) ----
    # (p, sink) in process_sink, source in node : (p, source) in process_source
    # AND has 2way_2var method. Cross-product node filtered to actual sources.
    sources_by_process: dict[str, list[str]] = {}
    for p, source in sources:
        sources_by_process.setdefault(p, []).append(source)
    rows_sink_toSource: list[tuple[str, str, str]] = [
        (p, sink, source)
        for p, sink in sinks
        if p in p_with_2way_2var
        for source in sources_by_process.get(p, ())
    ]
    _write_triples(solve_data_dir / "process_sink_toSource.csv",
                   ("process", "sink", "source"),
                   list(dict.fromkeys(rows_sink_toSource)))

    # ---- process_process_toSink_noConversion (mod L1046) ----
    # p in process, (p2, sink) in process_sink :
    #   p == p2 AND has 1way_1var AND p has no source rows
    rows_process_toSink_noConv: list[tuple[str, str, str]] = [
        (p, p, sink)
        for p, sink in sinks
        if p in p_with_1way_1var and p in process_no_source
    ]
    _write_triples(solve_data_dir / "process_process_toSink_noConversion.csv",
                   ("process_outer", "process", "sink"),
                   list(dict.fromkeys(rows_process_toSink_noConv)))

    # ---- process_source_toProcess_noConversion (mod L1052) ----
    # (p, source) in process_source, p2 in process :
    #   p == p2 AND has 1way_1var AND p has no sink rows
    rows_source_toProcess_noConv: list[tuple[str, str, str]] = [
        (p, source, p)
        for p, source in sources
        if p in p_with_1way_1var and p in process_no_sink
    ]
    _write_triples(solve_data_dir / "process_source_toProcess_noConversion.csv",
                   ("process", "source", "process_aux"),
                   list(dict.fromkeys(rows_source_toProcess_noConv)))


# ---------------------------------------------------------------------------
# Profile-method process joins
# ---------------------------------------------------------------------------


def write_process_profile_method_joins(
    input_dir: Path, solve_data_dir: Path
) -> None:
    """The two profile-method ``process__*__profile__profile_method`` sets.

    flextool.mod:961  process__profileProcess__toSink__profile__profile_method
    flextool.mod:969  process__source__toProfileProcess__profile__profile_method

    Both are: a join with process__node__profile__profile_method,
    constrained to (p, sink/source) being an actual arc, gated by
    "has any indirect method OR has no sources/sinks".
    """
    pm = _read_pairs(input_dir / "process_method.csv")
    sources = _read_pairs(input_dir / "process__source.csv")
    sinks = _read_pairs(input_dir / "process__sink.csv")
    profiles = _read_quad(input_dir / "process__node__profile__profile_method.csv")
    processes = _read_singles(input_dir / "process.csv")

    p_with_indirect = _processes_with_any_method_in(pm, METHOD_INDIRECT)
    has_sources = frozenset(p for p, _ in sources)
    has_sinks = frozenset(p for p, _ in sinks)
    sinks_by_process: dict[str, frozenset[str]] = {}
    for p, sink in sinks:
        sinks_by_process[p] = (sinks_by_process.get(p, frozenset()) | {sink})
    sources_by_process: dict[str, frozenset[str]] = {}
    for p, source in sources:
        sources_by_process[p] = (sources_by_process.get(p, frozenset()) | {source})

    # process__profileProcess__toSink__profile__profile_method
    # { p in process, (p2, sink, f, fm) in process__node__profile__profile_method
    #   : p == p2 AND (p, sink) in process_sink
    #   AND (has any indirect method OR has fewer than 1 source) }
    rows_to_sink: list[tuple[str, str, str, str, str]] = []
    for p in processes:
        if not (p in p_with_indirect or p not in has_sources):
            continue
        psinks = sinks_by_process.get(p, frozenset())
        for p2, sink, f, fm in profiles:
            if p2 == p and sink in psinks:
                rows_to_sink.append((p, p2, sink, f, fm))

    out = solve_data_dir / "process__profileProcess__toSink__profile__profile_method.csv"
    out.write_text(
        "process_outer,process,sink,profile,profile_method\n"
        + "".join(",".join(r) + "\n" for r in dict.fromkeys(rows_to_sink))
    )

    # process__source__toProfileProcess__profile__profile_method
    # { (p, source) in process_source, (p2, source, f, fm) in process__node__profile__profile_method
    #   : p == p2
    #   AND (has any indirect method OR has fewer than 1 sink) }
    # Tuple is dimen 5: (p, source, p2, f, fm). When the filter passes p == p2.
    rows_to_source: list[tuple[str, str, str, str, str]] = []
    for p, source in sources:
        if not (p in p_with_indirect or p not in has_sinks):
            continue
        for p2, src2, f, fm in profiles:
            if p2 == p and src2 == source:
                rows_to_source.append((p, source, p2, f, fm))

    out2 = solve_data_dir / "process__source__toProfileProcess__profile__profile_method.csv"
    out2.write_text(
        "process,source,process_aux,profile,profile_method\n"
        + "".join(",".join(r) + "\n" for r in dict.fromkeys(rows_to_source))
    )
