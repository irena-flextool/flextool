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


def write_entity_lifetime_method(input_dir: Path, solve_data_dir: Path) -> None:
    """flextool.mod:178-179."""
    explicit = _read_two_col_csv(input_dir / "entity__lifetime_method.csv")
    entities = _read_single_col_csv(input_dir / "entity.csv")
    if explicit:
        rows = explicit
    else:
        rows = [(e, m) for e in entities for m in _LIFETIME_METHOD_DEFAULT]
    _write_two_col(
        solve_data_dir / "entity__lifetime_method.csv",
        ("entity", "lifetime_method"),
        rows,
    )


def write_process_ct_method(input_dir: Path, solve_data_dir: Path) -> None:
    """flextool.mod:298-302 — two-class fallback (connection vs unit).

    process_connection / process_unit are loaded as 1-col input CSVs
    by the mod's ``set process_connection within process;`` declaration
    backed by input/process_connection.csv (and similarly process_unit).
    """
    explicit = _read_two_col_csv(input_dir / "process__ct_method.csv")
    if explicit:
        rows = explicit
    else:
        connections = _read_single_col_csv(input_dir / "process_connection.csv")
        units = _read_single_col_csv(input_dir / "process_unit.csv")
        rows = (
            [(p, m) for p in connections for m in _CT_METHOD_REGULAR]
            + [(p, m) for p in units for m in _CT_METHOD_CONSTANT]
        )
    _write_two_col(
        solve_data_dir / "process__ct_method.csv",
        ("process", "ct_method"),
        rows,
    )


def write_process_startup_method(input_dir: Path, solve_data_dir: Path) -> None:
    """flextool.mod:303-305.

    The _read source set has its OWN default in mod (``default {p in process,
    'no_startup'}`` at L303) — when the CSV is empty mod auto-fills it. We
    replicate that: empty CSV → emit (p, 'no_startup') for every process.
    """
    explicit = _read_two_col_csv(input_dir / "process__startup_method.csv")
    if explicit:
        rows = explicit
    else:
        processes = _read_single_col_csv(input_dir / "process.csv")
        rows = [(p, m) for p in processes for m in _STARTUP_METHOD_NO]
    _write_two_col(
        solve_data_dir / "process__startup_method.csv",
        ("process", "startup_method"),
        rows,
    )
