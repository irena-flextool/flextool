"""CO2-method projections of group__co2_method.

Migrated from flextool.mod:185-187:

    set group_co2_price      := setof {(g, m) in group__co2_method : m in co2_price_method}      (g);
    set group_co2_max_period := setof {(g, m) in group__co2_method : m in co2_max_period_method} (g);
    set group_co2_max_total  := setof {(g, m) in group__co2_method : m in co2_max_total_method}  (g);

The ``co2_*_method`` membership sets are model invariants from
flextool/flextool_base.dat:191-193 — duplicated here as Python
frozensets. Update both sites in lockstep if they ever change.
"""
from __future__ import annotations

import csv
from pathlib import Path


# Mirror flextool/flextool_base.dat:191-193 — constants of the model.
_CO2_PRICE_METHOD = frozenset((
    "price", "price_period", "price_total", "price_period_total",
))
_CO2_MAX_PERIOD_METHOD = frozenset((
    "period", "price_period", "period_total", "price_period_total",
))
_CO2_MAX_TOTAL_METHOD = frozenset((
    "total", "price_total", "period_total", "price_period_total",
))


def _project_groups_with_method_in(
    csv_path: Path, allowed_methods: frozenset[str]
) -> list[str]:
    if not csv_path.exists():
        return []
    seen: dict[str, None] = {}
    with csv_path.open() as fh:
        reader = csv.reader(fh)
        next(reader, None)
        for row in reader:
            if len(row) >= 2 and row[0] and row[1] in allowed_methods:
                seen.setdefault(row[0], None)
    return list(seen.keys())


def write_co2_method_sets(input_dir: Path, solve_data_dir: Path) -> None:
    src = input_dir / "group__co2_method.csv"
    for target, allowed in (
        ("group_co2_price.csv",      _CO2_PRICE_METHOD),
        ("group_co2_max_period.csv", _CO2_MAX_PERIOD_METHOD),
        ("group_co2_max_total.csv",  _CO2_MAX_TOTAL_METHOD),
    ):
        groups = _project_groups_with_method_in(src, allowed)
        (solve_data_dir / target).write_text(
            "group\n" + "".join(g + "\n" for g in groups)
        )
