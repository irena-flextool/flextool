"""Structural filter sets — single-condition filters of arc / param tables.

Each is a one-line setof / filter in flextool.mod:

    L405 connection__param        — process__param filtered by p ∈ process_connection
    L294 nodeGroupDispatch_node   — nodeGroupDispatch with at least one (g, n) in group_node
    L2011 commodity_node_co2      — commodity_node with p_commodity[c, 'co2_content'] ≠ 0
    L2009 process__commodity__node — process × commodity_node filtered by arc membership
    L2219 process_source_coeff_zero — process_source with zero max-capacity coefficient
    L2220 process_sink_coeff_zero   — process_sink with zero max-capacity coefficient
"""
from __future__ import annotations

import csv
from pathlib import Path


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


def _read_param_table(path: Path) -> dict[tuple[str, str], float]:
    """3-col: (entity, paramName, value)."""
    if not path.exists():
        return {}
    out: dict[tuple[str, str], float] = {}
    with path.open() as fh:
        reader = csv.reader(fh)
        next(reader, None)
        for row in reader:
            if len(row) >= 3 and row[0] and row[1]:
                try:
                    out[(row[0], row[1])] = float(row[2])
                except ValueError:
                    continue
    return out


def _read_arc_param(path: Path) -> dict[tuple[str, str], float]:
    """3-col: (process, source/sink, value) — coefficient tables."""
    if not path.exists():
        return {}
    out: dict[tuple[str, str], float] = {}
    with path.open() as fh:
        reader = csv.reader(fh)
        next(reader, None)
        for row in reader:
            if len(row) >= 3 and row[0] and row[1]:
                try:
                    out[(row[0], row[1])] = float(row[2])
                except ValueError:
                    continue
    return out


def _write_pairs(path: Path, header: tuple[str, str],
                 rows: list[tuple[str, str]]) -> None:
    path.write_text(",".join(header) + "\n"
                    + "".join(f"{a},{b}\n" for a, b in rows))


def _write_singles(path: Path, header: str, rows: list[str]) -> None:
    path.write_text(header + "\n" + "".join(r + "\n" for r in rows))


def write_connection_param(input_dir: Path, solve_data_dir: Path) -> None:
    """connection__param — process__param filtered by p ∈ process_connection."""
    pp = _read_pairs(input_dir / "p_process.csv")
    # p_process.csv has (process, processParam, value) — but we only need
    # (process, processParam) for set membership. Re-read via _read_pairs
    # would only give first 2 cols, perfect.
    # Wait: _read_pairs reads first 2 cols. p_process.csv columns are
    # (process, processParam, p_process). So pp is [(process, processParam), ...]
    connections = frozenset(_read_singles(input_dir / "process_connection.csv"))
    rows = list(dict.fromkeys((p, pa) for p, pa in pp if p in connections))
    _write_pairs(solve_data_dir / "connection__param.csv",
                 ("process", "processParam"), rows)


def write_nodegroup_dispatch_node(input_dir: Path, solve_data_dir: Path) -> None:
    """nodeGroupDispatch with at least one node in group_node."""
    dispatch_groups = _read_singles(input_dir / "nodeGroupDispatch.csv")
    group_nodes_pairs = _read_pairs(input_dir / "group__node.csv")
    groups_with_nodes = frozenset(g for g, _ in group_nodes_pairs)
    out = list(dict.fromkeys(g for g in dispatch_groups if g in groups_with_nodes))
    _write_singles(solve_data_dir / "nodeGroupDispatch_node.csv", "group", out)


def write_commodity_node_co2(input_dir: Path, solve_data_dir: Path) -> None:
    """commodity_node filtered by p_commodity[c, 'co2_content'] truthy.

    flextool.mod:2011  set commodity_node_co2 := {(c, n) in commodity_node :
                                                  p_commodity[c, 'co2_content']};
    `default 0` on p_commodity → commodities not in CSV give 0 (falsy).
    """
    cn = _read_pairs(input_dir / "commodity__node.csv")
    p_commodity = _read_param_table(input_dir / "p_commodity.csv")
    rows = [
        (c, n) for c, n in cn
        if p_commodity.get((c, "co2_content"), 0.0) != 0.0
    ]
    _write_pairs(solve_data_dir / "commodity_node_co2.csv",
                 ("commodity", "node"),
                 list(dict.fromkeys(rows)))


def write_process__commodity__node(input_dir: Path, solve_data_dir: Path) -> None:
    """process × commodity_node filtered by (p, n) ∈ process_source ∪ process_sink.

    flextool.mod:2009  set process__commodity__node := {p in process, (c, n) in commodity_node :
                                                        (p, n) in process_source || (p, n) in process_sink};
    """
    processes = _read_singles(input_dir / "process.csv")
    cn = _read_pairs(input_dir / "commodity__node.csv")
    sources = _read_pairs(input_dir / "process__source.csv")
    sinks = _read_pairs(input_dir / "process__sink.csv")
    _ep_acc: dict[str, dict[str, None]] = {}
    for p, n in sources + sinks:
        _ep_acc.setdefault(p, {})[n] = None
    arc_endpoints: dict[str, frozenset[str]] = {
        p: frozenset(d.keys()) for p, d in _ep_acc.items()
    }
    rows: list[tuple[str, str, str]] = []
    for p in processes:
        nodes_for_p = arc_endpoints.get(p, frozenset())
        if not nodes_for_p:
            continue
        for c, n in cn:
            if n in nodes_for_p:
                rows.append((p, c, n))
    out = solve_data_dir / "process__commodity__node.csv"
    out.write_text(
        "process,commodity,node\n"
        + "".join(",".join(r) + "\n" for r in dict.fromkeys(rows))
    )


def write_process_coeff_zero_sets(input_dir: Path, solve_data_dir: Path) -> None:
    """process_source_coeff_zero / process_sink_coeff_zero.

    flextool.mod:2219 process_source_coeff_zero := {(p, source) in process_source :
                                                     not p_process_source_max_capacity_coefficient[p, source]};
    flextool.mod:2220 process_sink_coeff_zero   := {(p, sink) in process_sink :
                                                     not p_process_sink_max_capacity_coefficient[p, sink]};
    `default 1` on these coefficients → entries not in CSV count as 1 (truthy).
    Only entities WITH an explicit zero coefficient appear in the result.
    """
    sources = _read_pairs(input_dir / "process__source.csv")
    sinks = _read_pairs(input_dir / "process__sink.csv")
    src_coef = _read_arc_param(
        input_dir / "p_process_source_max_capacity_coefficient.csv"
    )
    sink_coef = _read_arc_param(
        input_dir / "p_process_sink_max_capacity_coefficient.csv"
    )
    # default 1 — missing key is truthy (= 1), so excluded. Only explicit 0 values.
    src_zero = [(p, s) for p, s in sources if src_coef.get((p, s), 1.0) == 0.0]
    sink_zero = [(p, s) for p, s in sinks if sink_coef.get((p, s), 1.0) == 0.0]
    _write_pairs(solve_data_dir / "process_source_coeff_zero.csv",
                 ("process", "source"),
                 list(dict.fromkeys(src_zero)))
    _write_pairs(solve_data_dir / "process_sink_coeff_zero.csv",
                 ("process", "sink"),
                 list(dict.fromkeys(sink_zero)))
