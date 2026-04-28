"""Commodity-ladder filtered subsets of ``commodity``.

Migrated from flextool.mod:468-470 — three single-dimension sets
selected by ``p_commodity_price_method``:

    set commodity_with_ladder            := {c : price_method[c] != 'price'};
    set commodity_with_ladder_annual     := {c : price_method[c] == 'price_ladder_annual'};
    set commodity_with_ladder_cumulative := {c : price_method[c] == 'price_ladder_cumulative'};

The mod ``default 'price'`` on ``p_commodity_price_method`` means
commodities not present in the price-method CSV are treated as
``'price'`` and therefore excluded from all three sets.
"""
from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path


def compute_commodity_ladder_sets(
    price_methods: Mapping[str, str],
) -> tuple[list[str], list[str], list[str]]:
    """Return (with_ladder, with_ladder_annual, with_ladder_cumulative).

    Each list preserves insertion order of ``price_methods``. Use
    ``dict.fromkeys`` to dedupe while keeping order — the input mapping
    should already be deduplicated, but the wrapper makes the order-
    preserving intent explicit.
    """
    with_ladder = list(dict.fromkeys(
        c for c, m in price_methods.items() if m != "price"
    ))
    with_ladder_annual = list(dict.fromkeys(
        c for c, m in price_methods.items() if m == "price_ladder_annual"
    ))
    with_ladder_cumulative = list(dict.fromkeys(
        c for c, m in price_methods.items() if m == "price_ladder_cumulative"
    ))
    return with_ladder, with_ladder_annual, with_ladder_cumulative


def write_commodity_ladder_sets(
    price_methods: Mapping[str, str],
    solve_data_dir: Path,
) -> None:
    """Write three single-column CSVs for ``flextool.mod`` to load."""
    with_ladder, with_annual, with_cum = compute_commodity_ladder_sets(price_methods)
    _write_single_col(solve_data_dir / "commodity_with_ladder.csv",
                      "commodity", with_ladder)
    _write_single_col(solve_data_dir / "commodity_with_ladder_annual.csv",
                      "commodity", with_annual)
    _write_single_col(solve_data_dir / "commodity_with_ladder_cumulative.csv",
                      "commodity", with_cum)


def _write_single_col(path: Path, header: str, rows: list[str]) -> None:
    path.write_text(header + "\n" + "".join(r + "\n" for r in rows))
