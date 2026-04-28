"""Invest / divest method projections of entity__invest_method and group__invest_method.

Migrated from flextool.mod:175,176,182,183:

    set entityDivest := setof {(e, m) in entity__invest_method : m not in divest_method_not_allowed} (e);
    set entityInvest := setof {(e, m) in entity__invest_method : m not in invest_method_not_allowed} (e);
    set group_invest := setof {(g, m) in group__invest_method : m not in invest_method_not_allowed} (g);
    set group_divest := setof {(g, m) in group__invest_method : m not in divest_method_not_allowed} (g);

The ``*_method_not_allowed`` constant sets are model invariants
hardcoded in flextool/flextool_base.dat:211-212. They are duplicated
here as Python tuples; if those base.dat constants ever change the
constants below MUST be updated in lockstep.
"""
from __future__ import annotations

import csv
from pathlib import Path


# Mirror flextool/flextool_base.dat:211-212 — constants of the model.
_INVEST_METHOD_NOT_ALLOWED = frozenset((
    "not_allowed", "retire_period", "retire_total", "retire_no_limit",
))
_DIVEST_METHOD_NOT_ALLOWED = frozenset((
    "not_allowed", "invest_period", "invest_total", "invest_no_limit",
))


def _project_first_col_filter(
    csv_path: Path, disallowed_methods: frozenset[str]
) -> list[str]:
    """Read a 2-col ``(entity, method)`` CSV; return entities whose
    method is NOT in ``disallowed_methods``. Order = CSV row order,
    deduplicated by first occurrence.
    """
    if not csv_path.exists():
        return []
    seen: dict[str, None] = {}
    with csv_path.open() as fh:
        reader = csv.reader(fh)
        next(reader, None)
        for row in reader:
            if len(row) >= 2 and row[0] and row[1] not in disallowed_methods:
                seen.setdefault(row[0], None)
    return list(seen.keys())


def write_invest_method_sets(input_dir: Path, solve_data_dir: Path) -> None:
    """Write four single-column CSVs for the four derived sets."""
    for source_csv, target_name, disallowed in (
        ("entity__invest_method.csv", "entityInvest.csv", _INVEST_METHOD_NOT_ALLOWED),
        ("entity__invest_method.csv", "entityDivest.csv", _DIVEST_METHOD_NOT_ALLOWED),
        ("group__invest_method.csv",  "group_invest.csv", _INVEST_METHOD_NOT_ALLOWED),
        ("group__invest_method.csv",  "group_divest.csv", _DIVEST_METHOD_NOT_ALLOWED),
    ):
        names = _project_first_col_filter(input_dir / source_csv, disallowed)
        header = "entity" if "entity" in target_name.lower() else "group"
        (solve_data_dir / target_name).write_text(
            header + "\n" + "".join(n + "\n" for n in names)
        )
