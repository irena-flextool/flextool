"""Invest/divest *_total sets — entity and group filters by method enum.

Migrated from flextool.mod (post-batch-9 line numbers in the 1780s):

    set e_invest_total       = entityInvest filtered by 'invest_total' /
                               'invest_period_total' / 'invest_retire_total' /
                               'invest_retire_period_total' in entity__invest_method
    set e_divest_total       = entityDivest filtered by 'retire_total' /
                               'retire_period_total' / 'invest_retire_total' /
                               'invest_retire_period_total'
    set g_invest_total       = group_invest filtered by same invest_total set
    set g_divest_total       = group_divest filtered by same retire_total set
    set g_invest_cumulative  = group_invest filtered by 'cumulative_limits'

All read input/entity__invest_method.csv and input/group__invest_method.csv
plus the corresponding entityInvest / entityDivest / group_invest /
group_divest solve_data CSVs (Python-driven from L0 batch 1).
"""
from __future__ import annotations

import csv
from pathlib import Path


_INVEST_TOTAL = frozenset((
    "invest_total", "invest_period_total",
    "invest_retire_total", "invest_retire_period_total",
))
_RETIRE_TOTAL = frozenset((
    "retire_total", "retire_period_total",
    "invest_retire_total", "invest_retire_period_total",
))
_CUMULATIVE = frozenset(("cumulative_limits",))


def _read_two_col(path: Path) -> list[tuple[str, str]]:
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


def _entities_with_method_in(
    method_csv: Path, allowed: frozenset[str]
) -> frozenset[str]:
    return frozenset(e for e, m in _read_two_col(method_csv) if m in allowed)


def _filter_singles(
    universe: list[str], with_method: frozenset[str]
) -> list[str]:
    return [e for e in universe if e in with_method]


def _write_singles(path: Path, header: str, rows: list[str]) -> None:
    path.write_text(header + "\n" + "".join(r + "\n" for r in rows))


def write_invest_total_sets(input_dir: Path, solve_data_dir: Path) -> None:
    entity_methods_csv = input_dir / "entity__invest_method.csv"
    group_methods_csv = input_dir / "group__invest_method.csv"

    e_with_invest_total = _entities_with_method_in(entity_methods_csv, _INVEST_TOTAL)
    e_with_retire_total = _entities_with_method_in(entity_methods_csv, _RETIRE_TOTAL)
    g_with_invest_total = _entities_with_method_in(group_methods_csv, _INVEST_TOTAL)
    g_with_retire_total = _entities_with_method_in(group_methods_csv, _RETIRE_TOTAL)
    g_with_cumulative = _entities_with_method_in(group_methods_csv, _CUMULATIVE)

    invest_universe = _read_singles(solve_data_dir / "entityInvest.csv")
    divest_universe = _read_singles(solve_data_dir / "entityDivest.csv")
    g_invest_universe = _read_singles(solve_data_dir / "group_invest.csv")
    g_divest_universe = _read_singles(solve_data_dir / "group_divest.csv")

    _write_singles(solve_data_dir / "e_invest_total.csv", "entity",
                   _filter_singles(invest_universe, e_with_invest_total))
    _write_singles(solve_data_dir / "e_divest_total.csv", "entity",
                   _filter_singles(divest_universe, e_with_retire_total))
    _write_singles(solve_data_dir / "g_invest_total.csv", "group",
                   _filter_singles(g_invest_universe, g_with_invest_total))
    _write_singles(solve_data_dir / "g_divest_total.csv", "group",
                   _filter_singles(g_divest_universe, g_with_retire_total))
    _write_singles(solve_data_dir / "g_invest_cumulative.csv", "group",
                   _filter_singles(g_invest_universe, g_with_cumulative))


def write_ci_ladder_cumulative(input_dir: Path, solve_data_dir: Path) -> None:
    """flextool.mod:2000 — commodity__tier_cum filtered to commodities with
    cumulative-ladder pricing.
    """
    cum_rows = _read_two_col(input_dir / "commodity_ladder_cumulative.csv")
    with_cum = frozenset(_read_singles(
        solve_data_dir / "commodity_with_ladder_cumulative.csv"
    ))
    filtered = [(c, i) for c, i in cum_rows if c in with_cum]
    (solve_data_dir / "ci_ladder_cumulative.csv").write_text(
        "commodity,tier\n"
        + "".join(f"{c},{i}\n" for c, i in dict.fromkeys(filtered))
    )
