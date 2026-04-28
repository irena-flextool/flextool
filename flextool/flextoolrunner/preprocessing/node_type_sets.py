"""Node-type filtered sets — exercises the schema-default flow.

Migrated from flextool.mod:193-196:

    param p_node_type {n in node} symbolic in node_type, default 'balance';
    set nodeCommodity     := {n in node : p_node_type[n] = 'commodity'};
    set nodeBalance       := {n in node : p_node_type[n] = 'balance' || p_node_type[n] = 'storage'};
    set nodeState         := {n in node : p_node_type[n] = 'storage'};
    set nodeBalancePeriod := {n in node : p_node_type[n] = 'balance_within_period'};

The mod ``default 'balance'`` on p_node_type means a node not present
in input/p_node_type.csv is treated as 'balance'. This is the first
preprocessing module exercising that flow: every node in input/node.csv
is materialized with either its explicit type from p_node_type.csv or
the default 'balance', then partitioned into the four derived sets.

Key invariant: order of nodes in each output CSV matches input/node.csv
(so MathProg iteration over the loaded set follows the same order it
would have followed iterating ``node`` directly).
"""
from __future__ import annotations

import csv
from pathlib import Path


_DEFAULT_NODE_TYPE = "balance"  # mirrors flextool.mod:192 default clause


def _read_nodes_in_order(input_dir: Path) -> list[str]:
    path = input_dir / "node.csv"
    if not path.exists():
        return []
    with path.open() as fh:
        reader = csv.reader(fh)
        next(reader, None)
        return [r[0] for r in reader if r and r[0]]


def _read_explicit_node_types(input_dir: Path) -> dict[str, str]:
    path = input_dir / "p_node_type.csv"
    if not path.exists():
        return {}
    out: dict[str, str] = {}
    with path.open() as fh:
        reader = csv.reader(fh)
        next(reader, None)
        for row in reader:
            if len(row) >= 2 and row[0] and row[1]:
                out[row[0]] = row[1]
    return out


def write_node_type_sets(input_dir: Path, solve_data_dir: Path) -> None:
    """Write four single-column CSVs partitioned by p_node_type."""
    nodes = _read_nodes_in_order(input_dir)
    explicit = _read_explicit_node_types(input_dir)

    # Materialize every node with its effective type (explicit or default).
    effective: list[tuple[str, str]] = [
        (n, explicit.get(n, _DEFAULT_NODE_TYPE)) for n in nodes
    ]

    def _filter(predicate) -> list[str]:
        return [n for (n, t) in effective if predicate(t)]

    targets = (
        ("nodeCommodity.csv",      lambda t: t == "commodity"),
        ("nodeBalance.csv",        lambda t: t in ("balance", "storage")),
        ("nodeState.csv",          lambda t: t == "storage"),
        ("nodeBalancePeriod.csv",  lambda t: t == "balance_within_period"),
    )
    for name, pred in targets:
        out = _filter(pred)
        (solve_data_dir / name).write_text(
            "node\n" + "".join(n + "\n" for n in out)
        )
