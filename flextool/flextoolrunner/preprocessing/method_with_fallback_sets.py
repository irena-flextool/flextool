"""Method-assignment sets with global-empty fallback rules.

Three derived sets in flextool.mod follow the same idiom: read explicit
(entity, method) rows from input/<class>__<method>.csv, and if that
table is GLOBALLY empty, fall back to a constant default-method
assignment per entity / per entity kind.

    flextool.mod:178-179 entity__lifetime_method
        explicit OR (read empty AND m in lifetime_method_default)
        — single fallback class

    flextool.mod:298-302 process__ct_method
        explicit OR (read empty AND p in process_connection AND m in ct_method_regular)
        OR        (read empty AND p in process_unit       AND m in ct_method_constant)
        — two fallback classes split by process kind

    flextool.mod:304-306 process__startup_method
        explicit (where _read is itself defaulted to {(p, 'no_startup')} per
        the set-default on _read at L303) OR (read empty AND m in startup_method_no)
        — for our purposes: when the CSV is empty, emit (p, 'no_startup') for
        every process; when non-empty, emit the CSV literal contents.

The constants below mirror flextool/flextool_base.dat:18,19,26,189.
Update both sites in lockstep if those constants ever change.
"""
from __future__ import annotations

import csv
from pathlib import Path
from collections.abc import Sequence


# Mirror flextool/flextool_base.dat:189 — single-element default
_LIFETIME_METHOD_DEFAULT: tuple[str, ...] = ("reinvest_automatic",)

# Mirror flextool/flextool_base.dat:18-19 — single-element constants
_CT_METHOD_REGULAR: tuple[str, ...] = ("regular",)
_CT_METHOD_CONSTANT: tuple[str, ...] = ("constant_efficiency",)

# Mirror flextool/flextool_base.dat:26 — single-element constant
_STARTUP_METHOD_NO: tuple[str, ...] = ("no_startup",)

# Mirror flextool/flextool_base.dat — single-element defaults
_INFLOW_METHOD_DEFAULT: tuple[str, ...] = ("use_original",)
_STORAGE_BINDING_METHOD_DEFAULT: tuple[str, ...] = ("bind_forward_only",)


def _read_two_col_csv(path: Path) -> list[tuple[str, str]]:
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


def _read_single_col_csv(path: Path) -> list[str]:
    if not path.exists():
        return []
    with path.open() as fh:
        reader = csv.reader(fh)
        next(reader, None)
        return [r[0] for r in reader if r and r[0]]


def _write_two_col(path: Path, header: tuple[str, str],
                   rows: Sequence[tuple[str, str]]) -> None:
    path.write_text(
        ",".join(header) + "\n"
        + "".join(f"{a},{b}\n" for a, b in rows)
    )


def _per_entity_fallback(
    explicit: list[tuple[str, str]],
    entities: list[str],
    default_method_for,
) -> list[tuple[str, str]]:
    """Mod's idiom: emit explicit rows AND default rows for entities lacking
    any explicit row. The mod check ``sum{(e, m2) in _read} 1 = 0`` is a
    PER-ENTITY count (e is bound from outer iterator, m2 is summed) — NOT
    a global emptiness check. Fallback fires per-entity.

    ``default_method_for(entity)`` returns an iterable of methods to assign
    when the entity has no explicit rows; return () to omit the entity
    (e.g. process__ct_method skips processes that are neither connection
    nor unit).
    """
    explicit_by_entity: dict[str, list[str]] = {}
    for e, m in explicit:
        explicit_by_entity.setdefault(e, []).append(m)
    rows: list[tuple[str, str]] = []
    for e in entities:
        if e in explicit_by_entity:
            for m in explicit_by_entity[e]:
                rows.append((e, m))
        else:
            for m in default_method_for(e):
                rows.append((e, m))
    return rows


def write_entity_lifetime_method(input_dir: Path, solve_data_dir: Path) -> None:
    """flextool.mod:178-179 — per-entity fallback to ``lifetime_method_default``."""
    explicit = _read_two_col_csv(input_dir / "entity__lifetime_method.csv")
    entities = _read_single_col_csv(input_dir / "entity.csv")
    rows = _per_entity_fallback(
        explicit, entities, lambda _e: _LIFETIME_METHOD_DEFAULT,
    )
    _write_two_col(
        solve_data_dir / "entity__lifetime_method.csv",
        ("entity", "lifetime_method"),
        rows,
    )


def write_process_ct_method(input_dir: Path, solve_data_dir: Path) -> None:
    """flextool.mod:298-302 — per-process two-class fallback.

    Processes without explicit rows get ct_method_regular (if
    process_connection) or ct_method_constant (if process_unit), no
    fallback otherwise.
    """
    explicit = _read_two_col_csv(input_dir / "process__ct_method.csv")
    processes = _read_single_col_csv(input_dir / "process.csv")
    connections = frozenset(_read_single_col_csv(input_dir / "process_connection.csv"))
    units = frozenset(_read_single_col_csv(input_dir / "process_unit.csv"))

    def _ct_default_for(p: str) -> tuple[str, ...]:
        if p in connections:
            return _CT_METHOD_REGULAR
        if p in units:
            return _CT_METHOD_CONSTANT
        return ()

    rows = _per_entity_fallback(explicit, processes, _ct_default_for)
    _write_two_col(
        solve_data_dir / "process__ct_method.csv",
        ("process", "ct_method"),
        rows,
    )


def write_node_inflow_method(input_dir: Path, solve_data_dir: Path) -> None:
    """flextool.mod:203-204 — per-node fallback to ``inflow_method_default``."""
    explicit = _read_two_col_csv(input_dir / "node__inflow_method.csv")
    nodes = _read_single_col_csv(input_dir / "node.csv")
    rows = _per_entity_fallback(
        explicit, nodes, lambda _n: _INFLOW_METHOD_DEFAULT,
    )
    _write_two_col(
        solve_data_dir / "node__inflow_method.csv",
        ("node", "inflow_method"),
        rows,
    )


def write_node_storage_binding_method(input_dir: Path, solve_data_dir: Path) -> None:
    """flextool.mod:208-209 — per-node fallback to ``storage_binding_method_default``."""
    explicit = _read_two_col_csv(input_dir / "node__storage_binding_method.csv")
    nodes = _read_single_col_csv(input_dir / "node.csv")
    rows = _per_entity_fallback(
        explicit, nodes, lambda _n: _STORAGE_BINDING_METHOD_DEFAULT,
    )
    _write_two_col(
        solve_data_dir / "node__storage_binding_method.csv",
        ("node", "storage_binding_method"),
        rows,
    )


def write_process_startup_method(input_dir: Path, solve_data_dir: Path) -> None:
    """flextool.mod:303-305 — per-process fallback to ``startup_method_no``.

    The _read source set has its own ``default {p in process, 'no_startup'}``
    set-default in mod (L303). For our purposes that set-default fires only
    when _read is GLOBALLY empty; otherwise _read = literal CSV contents.
    The DERIVED set's filter is per-process: include explicit rows plus
    (p, 'no_startup') for processes lacking an explicit row.
    """
    explicit = _read_two_col_csv(input_dir / "process__startup_method.csv")
    processes = _read_single_col_csv(input_dir / "process.csv")
    rows = _per_entity_fallback(
        explicit, processes, lambda _p: _STARTUP_METHOD_NO,
    )
    _write_two_col(
        solve_data_dir / "process__startup_method.csv",
        ("process", "startup_method"),
        rows,
    )
