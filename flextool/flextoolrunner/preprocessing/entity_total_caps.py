"""Entity invest/divest total caps — first calculated-param migration.

Migrated from flextool.mod:1825-1843:

    param e_invest_max_total{e in entityInvest} :=
      + (if e in process then p_process[e, 'invest_max_total'])
      + (if e in node    then p_node[e, 'invest_max_total']);

    param e_divest_max_total{e in entityDivest} := ... 'retire_max_total' ...
    param e_invest_min_total{e in entityInvest} := ... 'invest_min_total' ...
    param e_divest_min_total{e in entityDivest} := ... 'retire_min_total' ...

Each param is keyed on entityInvest / entityDivest (now Python-driven
themselves — see invest_method_sets.py from L0 batch 1). The value
comes from either p_process[e, paramName] (if entity is a process) or
p_node[e, paramName] (if entity is a node). p_process / p_node
default to 0 (mod L478 area), so missing entries contribute 0.

Float precision: values are read from input/p_process.csv and
input/p_node.csv as written by input_writer.write_parameter; we
parse them with ``float()`` (round-trip exact for IEEE 754 doubles)
and write them back with ``repr()`` so MathProg sees the same bit
pattern it would have read directly from the source CSVs.
"""
from __future__ import annotations

import csv
from pathlib import Path


def _read_param_table(path: Path) -> dict[tuple[str, str], float]:
    """Read a 3-col ``(entity, paramName, value)`` CSV into a lookup dict."""
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


def _read_single_col(path: Path) -> list[str]:
    if not path.exists():
        return []
    with path.open() as fh:
        reader = csv.reader(fh)
        next(reader, None)
        return [r[0] for r in reader if r and r[0]]


def _read_keyed_set(path: Path) -> set[str]:
    """Membership set built from a 1-col CSV. Used only for hot-path
    `e in process` / `e in node` checks — the source CSV already has
    deterministic order so we don't need to preserve it here.
    """
    return frozenset(_read_single_col(path))


def _format_float(v: float) -> str:
    """Round-trip-exact float repr that the GMPL parser also accepts."""
    return repr(v)


def _write_keyed_param(
    path: Path,
    keys_in_order: list[str],
    values: dict[str, float],
    header: tuple[str, str],
) -> None:
    path.write_text(
        ",".join(header) + "\n"
        + "".join(f"{k},{_format_float(values[k])}\n" for k in keys_in_order)
    )


def _compute_entity_total(
    entity_keys: list[str],
    process_set: frozenset[str],
    node_set: frozenset[str],
    p_process: dict[tuple[str, str], float],
    p_node: dict[tuple[str, str], float],
    param_name: str,
) -> dict[str, float]:
    """Sum p_process[e, param] + p_node[e, param], defaulting either to 0
    when the entity is not in that class or has no explicit row.
    """
    out: dict[str, float] = {}
    for e in entity_keys:
        v = 0.0
        if e in process_set:
            v += p_process.get((e, param_name), 0.0)
        if e in node_set:
            v += p_node.get((e, param_name), 0.0)
        out[e] = v
    return out


def write_entity_total_caps(
    input_dir: Path, solve_data_dir: Path
) -> None:
    """Write all four e_*_total params keyed on entityInvest / entityDivest."""
    process_set = _read_keyed_set(input_dir / "process.csv")
    node_set = _read_keyed_set(input_dir / "node.csv")
    p_process = _read_param_table(input_dir / "p_process.csv")
    p_node = _read_param_table(input_dir / "p_node.csv")

    invest_keys = _read_single_col(solve_data_dir / "entityInvest.csv")
    divest_keys = _read_single_col(solve_data_dir / "entityDivest.csv")

    spec = (
        ("e_invest_max_total.csv", invest_keys, "invest_max_total"),
        ("e_divest_max_total.csv", divest_keys, "retire_max_total"),
        ("e_invest_min_total.csv", invest_keys, "invest_min_total"),
        ("e_divest_min_total.csv", divest_keys, "retire_min_total"),
    )
    header = ("entity", "value")
    for fname, keys, param in spec:
        values = _compute_entity_total(
            keys, process_set, node_set, p_process, p_node, param,
        )
        _write_keyed_param(solve_data_dir / fname, keys, values, header)
