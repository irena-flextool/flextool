"""Solve creation-order helper.

Reads ``output_raw/_solve_order.txt``, the per-roll manifest
:func:`append_solve_order` writes — one line per sub-solve, in execution
order — to recover the order in which solves were executed.

Used by :mod:`read_variables._read_from_parquet` to concatenate per-solve
parquets in solve creation order.  Matching row order at the source
avoids needing any post-concat sort or per-call-site reindex.

Item 163: the manifest is the sole source.  It is authoritative on the
default ``keep_solutions=False`` flow and is always present after
:func:`append_solve_order` has run at least once; there is no disk
fallback to a legacy per-solve CSV.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd


# Creation-order manifest written incrementally by the per-roll output
# hook (one line per sub-solve, in execution order).  This is the
# authoritative order source on the default ``keep_solutions=False`` flow,
# where ``solve_data/solve__p_entity_pre_existing.csv`` is absent.
_MANIFEST_NAME = "_solve_order.txt"


def append_solve_order(output_dir: Path | str, solve_name: str) -> None:
    """Append ``solve_name`` to the creation-order manifest in ``output_dir``.

    Idempotent within a run only in the sense that re-running the cascade
    into the same folder must start from a clean manifest — callers that
    re-run must clear ``output_dir`` first (the cascade does).  Called
    once per sub-solve, in execution order, so the file's line order is
    the cascade creation order both the variable reader and the stage-3
    param/set union read back via :func:`load_solve_order`.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    with (out / _MANIFEST_NAME).open("a", encoding="utf-8") as f:
        f.write(f"{solve_name}\n")


def _load_from_manifest(output_dir: Path) -> dict[str, int]:
    path = output_dir / _MANIFEST_NAME
    if not path.exists():
        return {}
    seen: list[str] = []
    seen_set: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        name = line.strip()
        if name and name not in seen_set:
            seen_set.add(name)
            seen.append(name)
    return {s: i for i, s in enumerate(seen)}


def load_solve_order(work_folder: Path | str) -> dict[str, int]:
    """Map solve names to their creation order (0-indexed).

    Source: ``output_raw/_solve_order.txt`` (:func:`append_solve_order`).
    Returns ``{}`` when the manifest is absent or empty.
    """
    work = Path(work_folder)
    return _load_from_manifest(work / "output_raw")


# ---------------------------------------------------------------------------
# Solve labels of the per-period output tables (#210)
# ---------------------------------------------------------------------------

#: A rolling solve's rolls are named ``<solve>_roll_<k>`` (``solve._recursive_solve``).
_ROLL_SUFFIX = re.compile(r"_roll_\d+$")

#: The per-period output tables (raw result keys) whose values come from the investment solves: the capacities,
#: the investment marginal values and the capacity-margin shortfall (a constraint of the investment periods).  Their ``solve`` column names the solve that realized the period's investments
#: (``d_realize_invest``); a period without realized investments falls back to the dispatch label.
INVESTMENT_PERIOD_TABLES: frozenset[str] = frozenset({
    "unit_capacity_ed_p",
    "connection_capacity_ed_p",
    "node_capacity_ed_p",
    "dual_invest_effective_unit_d_e",
    "dual_invest_effective_connection_d_e",
    "dual_invest_effective_node_d_e",
    "nodeGroup_slack_capacity_margin_d_g",
})


def complete_solve_name(name: str) -> str:
    """``name`` without a rolling ``_roll_<k>`` suffix: the name of the complete solve."""
    return _ROLL_SUFFIX.sub("", str(name))


def period_solve_labels(
    solve_period: "pd.MultiIndex | None",
    invest_solve_period: "pd.MultiIndex | None" = None,
    *,
    table: str | None = None,
) -> dict[str, str]:
    """``{period: solve label}`` for the ``solve`` column of a per-period output table.

    ``solve_period`` holds one ``(solve, period)`` row for every solve (or roll) that realized a part of the
    period's dispatch, in cascade order.  A period's label is the complete solve (roll suffix removed) of its last
    row: the last complete solve that realized the period's dispatch.  For a table of
    :data:`INVESTMENT_PERIOD_TABLES`, the periods of ``invest_solve_period`` (``(solve, period)`` rows of the
    realized investments) are labelled by the last complete solve that realized the period's investments instead.
    """
    labels: dict[str, str] = {}
    for source in (solve_period,
                   invest_solve_period if table in INVESTMENT_PERIOD_TABLES else None):
        if source is None or len(source) == 0:
            continue
        for solve, period in zip(source.get_level_values("solve"),
                                 source.get_level_values("period")):
            labels[period] = complete_solve_name(solve)
    return labels
