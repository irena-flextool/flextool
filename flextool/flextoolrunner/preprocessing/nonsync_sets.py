"""process__group_inside_group_nonSync — quadratic-style join made fast.

Migrated from flextool.mod:2017-2023:

    set process__group_inside_group_nonSync :=
      {p in process, g in groupNonSync:
       sum{source in node, sink in node:
           (p, source) in process_source && (g, source) in group_node
           && (p, sink) in process_sink   && (g, sink)   in group_node
           && source != sink} 1};

In MathProg this generates one (p, g) row only if there is at least
one (source, sink) pair where:
  - source ≠ sink
  - process p has source as one of its sources
  - process p has sink as one of its sinks
  - both source and sink are in group g

The mod expression iterates ``node × node`` per (p, g) — O(|node|²)
per row. In Python we precompute group → nodes / process → sources /
process → sinks once and intersect, dropping the quadratic.

Output: 2-tuple set (process, group).
"""
from __future__ import annotations

import csv
from pathlib import Path


def _read_two_col_pairs(path: Path) -> list[tuple[str, str]]:
    if not path.exists():
        return []
    rows: list[tuple[str, str]] = []
    with path.open() as fh:
        reader = csv.reader(fh)
        next(reader, None)
        for row in reader:
            if len(row) >= 2 and row[0] and row[1]:
                rows.append((row[0], row[1]))
    return rows


def _read_single_col(path: Path) -> list[str]:
    if not path.exists():
        return []
    with path.open() as fh:
        reader = csv.reader(fh)
        next(reader, None)
        return [r[0] for r in reader if r and r[0]]


def write_process_group_inside_group_nonsync(
    input_dir: Path, solve_data_dir: Path
) -> None:
    """Compute (p, g) where p has source≠sink both in g, for g ∈ groupNonSync."""
    # groupNonSync is loaded by mod from input/groupNonSync.csv.
    nonsync_groups = _read_single_col(input_dir / "groupNonSync.csv")
    if not nonsync_groups:
        (solve_data_dir / "process__group_inside_group_nonSync.csv").write_text(
            "process,group\n"
        )
        return

    group_nodes_mut: dict[str, dict[str, None]] = {}
    for g, n in _read_two_col_pairs(input_dir / "group__node.csv"):
        group_nodes_mut.setdefault(g, {})[n] = None
    group_node_lookup: dict[str, frozenset[str]] = {
        g: frozenset(d.keys()) for g, d in group_nodes_mut.items()
    }

    process_sources_mut: dict[str, dict[str, None]] = {}
    for p, n in _read_two_col_pairs(input_dir / "process__source.csv"):
        process_sources_mut.setdefault(p, {})[n] = None
    process_sources: dict[str, frozenset[str]] = {
        p: frozenset(d.keys()) for p, d in process_sources_mut.items()
    }

    process_sinks_mut: dict[str, dict[str, None]] = {}
    for p, n in _read_two_col_pairs(input_dir / "process__sink.csv"):
        process_sinks_mut.setdefault(p, {})[n] = None
    process_sinks: dict[str, frozenset[str]] = {
        p: frozenset(d.keys()) for p, d in process_sinks_mut.items()
    }

    processes = _read_single_col(input_dir / "process.csv")

    # Iterate process × groupNonSync in CSV order to preserve mod's
    # would-be iteration order.
    rows: list[tuple[str, str]] = []
    for p in processes:
        psrc = process_sources.get(p, frozenset())
        psnk = process_sinks.get(p, frozenset())
        if not psrc or not psnk:
            continue
        for g in nonsync_groups:
            gnodes = group_node_lookup.get(g, frozenset())
            if not gnodes:
                continue
            srcs_in = psrc & gnodes
            sinks_in = psnk & gnodes
            if not srcs_in or not sinks_in:
                continue
            # Exists source ≠ sink iff |srcs ∪ sinks| ≥ 2 OR (srcs == sinks
            # but with ≥2 elements). Equivalently: not (|srcs|==1 and
            # |sinks|==1 and srcs==sinks).
            if (len(srcs_in) == 1 and len(sinks_in) == 1
                    and srcs_in == sinks_in):
                continue
            rows.append((p, g))

    out = solve_data_dir / "process__group_inside_group_nonSync.csv"
    out.write_text(
        "process,group\n"
        + "".join(f"{p},{g}\n" for p, g in rows)
    )
