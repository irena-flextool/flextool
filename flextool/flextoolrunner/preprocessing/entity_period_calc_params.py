"""Per-entity per-period calculated params — pdProcess / pdNode + ed_* family.

Migrated from flextool.mod:

    L1216 pdNode      (4-branch resolution from pd_node / p_node)
    L1252 pdProcess   (4-branch resolution from pd_process / p_process)
    L1358 edEntity_lifetime  = pdProcess['lifetime', d] OR pdNode['lifetime', d]
    L1699 ed_fixed_cost      = (node ? pdNode['fixed_cost', d]   : 0) * 1000
                               + (process ? pdProcess['fixed_cost', d] : 0) * 1000
    L1845 ed_invest_max_period {(e, d) in ed_invest}  = pdProcess/pdNode 'invest_max_period'
    L1850 ed_invest_min_period                          'invest_min_period'
    L1855 ed_divest_max_period {(e, d) in ed_divest}  = pdProcess/pdNode 'retire_max_period'
    L1860 ed_divest_min_period                          'retire_min_period'
    L1865 ed_cumulative_max_capacity                    'cumulative_max_capacity'
    L1870 ed_cumulative_min_capacity                    'cumulative_min_capacity'

Each follows the if-process-then-pdProcess[…]-else-if-node-then-pdNode[…]-else-0
shape; processes and nodes are disjoint so exactly one branch fires.
"""
from __future__ import annotations

import csv
from pathlib import Path

from flextool.flextoolrunner.preprocessing.pd_lookups import PdLookup


def _read_singles(path: Path) -> list[str]:
    if not path.exists():
        return []
    with path.open() as fh:
        reader = csv.reader(fh)
        next(reader, None)
        return [r[0] for r in reader if r and r[0]]


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


def _write_keyed_2(path: Path, header: tuple[str, str, str],
                   rows: list[tuple[str, str, float]]) -> None:
    path.write_text(",".join(header) + "\n"
                    + "".join(f"{a},{b},{repr(v)}\n" for a, b, v in rows))


def write_entity_period_calc_params(input_dir: Path, solve_data_dir: Path) -> None:
    """Migrate pdProcess/pdNode + 8 ed_* family params in one pass."""
    pp = PdLookup(
        pd_csv=input_dir / "pd_process.csv",
        p_csv=input_dir / "p_process.csv",
        period_branch_csv=solve_data_dir / "period__branch.csv",
    )
    pn = PdLookup(
        pd_csv=input_dir / "pd_node.csv",
        p_csv=input_dir / "p_node.csv",
        period_branch_csv=solve_data_dir / "period__branch.csv",
    )

    process_set = frozenset(_read_singles(input_dir / "process.csv"))
    node_set = frozenset(_read_singles(input_dir / "node.csv"))

    # Universes (computed in earlier preprocessing batches)
    process_period_in_use = _read_pairs(
        solve_data_dir / "process__PeriodParam_in_use.csv"
    )  # list of (process, param)
    node_period_in_use = _read_pairs(
        solve_data_dir / "node__PeriodParam_in_use.csv"
    )
    period_with_history = _read_singles(solve_data_dir / "period_with_history.csv")

    # ---- pdProcess ------------------------------------------------------
    pdProcess_rows: list[tuple[str, str, str, float]] = []
    for (p, param) in process_period_in_use:
        for d in period_with_history:
            v = pp.get(p, param, d)
            pdProcess_rows.append((p, param, d, v))

    out = solve_data_dir / "pdProcess.csv"
    out.write_text(
        "process,param,period,value\n"
        + "".join(f"{p},{pa},{d},{repr(v)}\n" for p, pa, d, v in pdProcess_rows)
    )

    # ---- pdNode ---------------------------------------------------------
    pdNode_rows: list[tuple[str, str, str, float]] = []
    for (n, param) in node_period_in_use:
        for d in period_with_history:
            v = pn.get(n, param, d)
            pdNode_rows.append((n, param, d, v))

    out = solve_data_dir / "pdNode.csv"
    out.write_text(
        "node,param,period,value\n"
        + "".join(f"{n},{pa},{d},{repr(v)}\n" for n, pa, d, v in pdNode_rows)
    )

    # ---- ed_* family ----------------------------------------------------
    # Each is keyed on (e in entity, d in period_with_history) for the
    # broad ones, or (e, d) in ed_invest / ed_divest for the per-d ones.
    # ed_invest / ed_divest are derived sets we haven't migrated yet —
    # they depend on ed_entity_annual which itself depends on these
    # very params (chicken-egg). For NOW, key on the same universe mod
    # uses: period_with_history × (entity from process ∪ node).
    #
    # Mod's domain `{(e, d) in ed_invest}` only generates a value for
    # entries actually in ed_invest. Since ed_invest is still derived in
    # mod (and reads my migrated pdProcess/pdNode), mod will only look
    # up ed_invest_max_period[e, d] for those (e, d). My output covers a
    # superset — extra rows are harmless because mod's table data IN
    # only loads the ones it indexes.

    def _ed_value(e: str, param_proc: str, d: str) -> float:
        if e in process_set:
            return pp.get(e, param_proc, d)
        if e in node_set:
            return pn.get(e, param_proc, d)
        return 0.0

    # edEntity_lifetime{e in entity, d in period_with_history}
    rows: list[tuple[str, str, float]] = []
    entities = _read_singles(input_dir / "entity.csv")
    for e in entities:
        for d in period_with_history:
            rows.append((e, d, _ed_value(e, "lifetime", d)))
    _write_keyed_2(solve_data_dir / "edEntity_lifetime.csv",
                   ("entity", "period", "value"), rows)

    # ed_fixed_cost{e in entity, d in period_with_history}: each side ×1000.
    rows = []
    for e in entities:
        for d in period_with_history:
            v = (1000.0 if e in node_set else 0.0) * pn.get(e, "fixed_cost", d) \
                + (1000.0 if e in process_set else 0.0) * pp.get(e, "fixed_cost", d)
            rows.append((e, d, v))
    _write_keyed_2(solve_data_dir / "ed_fixed_cost.csv",
                   ("entity", "period", "value"), rows)

    # ed_*_period and ed_cumulative_* deferred until ed_invest / ed_divest
    # are themselves Python-driven. Their mod domain is
    # ``{(e, d) in ed_invest}`` and writing extra rows with
    # ``table data IN [entity, period]`` may not be silently tolerated by
    # MathProg's loader. Once ed_invest moves to Python (later batch
    # after ed_entity_annual lands), these can follow.
