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

from flextool.flextoolrunner.preprocessing.pd_lookups import (
    PdLookup,
    PdtLookup,
    PdtLookupPerSide,
    PROCESS_PARAM_DEF1,
)


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

    # ---- p_entity_unitsize{e in entity} (write_input scope) ------------
    # mod L1279: process branch prefers virtual_unitsize, then existing,
    # then 1000. Same for nodes.
    p_process: dict[tuple[str, str], float] = {}
    pp_path = input_dir / "p_process.csv"
    if pp_path.exists():
        with pp_path.open() as fh:
            reader = csv.reader(fh)
            next(reader, None)
            for row in reader:
                if len(row) >= 3 and row[0] and row[1]:
                    try:
                        p_process[(row[0], row[1])] = float(row[2])
                    except ValueError:
                        continue
    p_node: dict[tuple[str, str], float] = {}
    pn_path = input_dir / "p_node.csv"
    if pn_path.exists():
        with pn_path.open() as fh:
            reader = csv.reader(fh)
            next(reader, None)
            for row in reader:
                if len(row) >= 3 and row[0] and row[1]:
                    try:
                        p_node[(row[0], row[1])] = float(row[2])
                    except ValueError:
                        continue
    unitsize_rows: list[tuple[str, float]] = []
    for e in entities:
        if e in process_set:
            v = (p_process.get((e, "virtual_unitsize"), 0.0)
                 or p_process.get((e, "existing"), 0.0)
                 or 1000.0)
        elif e in node_set:
            v = (p_node.get((e, "virtual_unitsize"), 0.0)
                 or p_node.get((e, "existing"), 0.0)
                 or 1000.0)
        else:
            v = 0.0
        unitsize_rows.append((e, v))
    (solve_data_dir / "p_entity_unitsize.csv").write_text(
        "entity,value\n"
        + "".join(f"{e},{repr(v)}\n" for e, v in unitsize_rows)
    )


def write_p_entity_pre_existing(
    input_dir: Path, solve_data_dir: Path
) -> None:
    """flextool.mod L1886-1895 — pre-existing capacity per (entity, period).

    12-branch sum, but exactly one branch fires per (e, d) given the
    method/kind/virtual_unitsize trichotomy. Equivalent simplified form:

        Let method = entity__lifetime_method[e]
        Let v_existing = pdProcess[e,'existing',d] if e in process
                         else pdNode[e,'existing',d] if e in node
                         else 0
        Let v_unit = p_process[e,'virtual_unitsize'] if e in process
                     else p_node[e,'virtual_unitsize'] if e in node
                     else 0
        if method not in {reinvest_automatic, reinvest_choice, no_investment}: 0
        if method in {reinvest_choice, no_investment} and not (
              p_years_d[d] < sum_{d_first in period_first}
                              (p_years_d[d_first] + edEntity_lifetime[e, d_first])
            ): 0
        else: v_existing * v_unit  if v_unit else  v_existing

    Output covers entity × period_in_use (matches mod's index domain) so
    every key the mod's `param p_entity_pre_existing` table loader expects
    is present. Reads pdProcess/pdNode and edEntity_lifetime CSVs that
    write_entity_period_calc_params just wrote in the same per-solve
    pass.
    """
    process_set = frozenset(_read_singles(input_dir / "process.csv"))
    node_set = frozenset(_read_singles(input_dir / "node.csv"))
    entities = _read_singles(input_dir / "entity.csv")
    period_in_use = _read_singles(solve_data_dir / "period_in_use_set.csv")
    period_first = _read_singles(solve_data_dir / "period_first.csv")

    lifetime_method: dict[str, str] = {}
    for e_, m in _read_pairs(solve_data_dir / "entity__lifetime_method.csv"):
        lifetime_method[e_] = m

    p_years_d: dict[str, float] = {}
    for r in _read_pairs(solve_data_dir / "p_years_d.csv"):
        try:
            p_years_d[r[0]] = float(r[1])
        except ValueError:
            continue

    ed_lifetime: dict[tuple[str, str], float] = {}
    elf = solve_data_dir / "edEntity_lifetime.csv"
    if elf.exists():
        with elf.open() as fh:
            reader = csv.reader(fh)
            next(reader, None)
            for r in reader:
                if len(r) >= 3 and r[0] and r[1]:
                    try:
                        ed_lifetime[(r[0], r[1])] = float(r[2])
                    except ValueError:
                        continue

    pd_existing_proc: dict[tuple[str, str], float] = {}
    pd_existing_node: dict[tuple[str, str], float] = {}
    pdp = solve_data_dir / "pdProcess.csv"
    if pdp.exists():
        with pdp.open() as fh:
            reader = csv.reader(fh)
            next(reader, None)
            for r in reader:
                if len(r) >= 4 and r[0] and r[1] == "existing" and r[2]:
                    try:
                        pd_existing_proc[(r[0], r[2])] = float(r[3])
                    except ValueError:
                        continue
    pdn = solve_data_dir / "pdNode.csv"
    if pdn.exists():
        with pdn.open() as fh:
            reader = csv.reader(fh)
            next(reader, None)
            for r in reader:
                if len(r) >= 4 and r[0] and r[1] == "existing" and r[2]:
                    try:
                        pd_existing_node[(r[0], r[2])] = float(r[3])
                    except ValueError:
                        continue

    p_process_vu: dict[str, float] = {}
    p_node_vu: dict[str, float] = {}
    pp_path = input_dir / "p_process.csv"
    if pp_path.exists():
        with pp_path.open() as fh:
            reader = csv.reader(fh)
            next(reader, None)
            for r in reader:
                if len(r) >= 3 and r[0] and r[1] == "virtual_unitsize":
                    try:
                        p_process_vu[r[0]] = float(r[2])
                    except ValueError:
                        continue
    pn_path = input_dir / "p_node.csv"
    if pn_path.exists():
        with pn_path.open() as fh:
            reader = csv.reader(fh)
            next(reader, None)
            for r in reader:
                if len(r) >= 3 and r[0] and r[1] == "virtual_unitsize":
                    try:
                        p_node_vu[r[0]] = float(r[2])
                    except ValueError:
                        continue

    # Per-entity lifetime gate sum: sum_{d_first in period_first}
    #                                 (p_years_d[d_first] + edEntity_lifetime[e, d_first])
    def _life_sum(e: str) -> float:
        return sum(
            p_years_d.get(d_first, 0.0) + ed_lifetime.get((e, d_first), 0.0)
            for d_first in period_first
        )

    rows: list[tuple[str, str, float]] = []
    for e in entities:
        method = lifetime_method.get(e, "")
        is_proc = e in process_set
        is_node = e in node_set
        v_unit = (
            p_process_vu.get(e, 0.0) if is_proc
            else p_node_vu.get(e, 0.0) if is_node
            else 0.0
        )
        gate_sum = (
            _life_sum(e)
            if method in ("reinvest_choice", "no_investment") else None
        )
        for d in period_in_use:
            v: float = 0.0
            if method in ("reinvest_automatic", "reinvest_choice", "no_investment"):
                if method == "reinvest_automatic" or (
                    gate_sum is not None and p_years_d.get(d, 0.0) < gate_sum
                ):
                    if is_proc:
                        pd_e = pd_existing_proc.get((e, d), 0.0)
                    elif is_node:
                        pd_e = pd_existing_node.get((e, d), 0.0)
                    else:
                        pd_e = 0.0
                    v = pd_e * v_unit if v_unit else pd_e
            rows.append((e, d, v))

    _write_keyed_2(solve_data_dir / "p_entity_pre_existing.csv",
                   ("entity", "period", "value"), rows)


def write_p_entity_divest_cumulative_max(
    input_dir: Path, solve_data_dir: Path
) -> None:
    """flextool.mod L1920-1933 — cumulative ceiling on v_divest summed
    by dispatch period d.

    Three-branch sum (only one fires per (e, d) given the e_divest_total
    membership / cardinality split):

        if e ∉ e_divest_total:
            sum_{(e, d_div) in ed_divest_period, p_years_d[d_div] ≤ p_years_d[d]}
                ed_divest_max_period[e, d_div]
        elif e ∈ e_divest_total AND ed_divest_period has no rows for e:
            e_divest_max_total[e]
        else:  # e ∈ e_divest_total AND ed_divest_period has ≥1 row for e
            max(period_sum, e_divest_max_total[e])

    Domain: entityDivest × period_in_use. Reads ed_divest_period.csv
    and ed_divest_max_period.csv that batches 19/20 produce.
    """
    entityDivest = _read_singles(solve_data_dir / "entityDivest.csv")
    e_divest_total = frozenset(
        _read_singles(solve_data_dir / "e_divest_total.csv")
    )
    period_in_use = _read_singles(solve_data_dir / "period_in_use_set.csv")

    ed_divest_period = _read_pairs(solve_data_dir / "ed_divest_period.csv")
    div_periods_for_e: dict[str, list[str]] = {}
    for e_, d_ in ed_divest_period:
        div_periods_for_e.setdefault(e_, []).append(d_)

    p_years_d: dict[str, float] = {}
    for r in _read_pairs(solve_data_dir / "p_years_d.csv"):
        try:
            p_years_d[r[0]] = float(r[1])
        except ValueError:
            continue

    ed_divest_max: dict[tuple[str, str], float] = {}
    edmp = solve_data_dir / "ed_divest_max_period.csv"
    if edmp.exists():
        with edmp.open() as fh:
            reader = csv.reader(fh)
            next(reader, None)
            for r in reader:
                if len(r) >= 3 and r[0] and r[1]:
                    try:
                        ed_divest_max[(r[0], r[1])] = float(r[2])
                    except ValueError:
                        continue

    e_divest_max_total: dict[str, float] = {}
    edmt = solve_data_dir / "e_divest_max_total.csv"
    if edmt.exists():
        with edmt.open() as fh:
            reader = csv.reader(fh)
            next(reader, None)
            for r in reader:
                if len(r) >= 2 and r[0]:
                    try:
                        e_divest_max_total[r[0]] = float(r[1])
                    except ValueError:
                        continue

    rows: list[tuple[str, str, float]] = []
    for e in entityDivest:
        in_total = e in e_divest_total
        e_div_periods = div_periods_for_e.get(e, [])
        e_total_max = e_divest_max_total.get(e, 0.0)
        for d in period_in_use:
            d_years = p_years_d.get(d, 0.0)
            period_sum = sum(
                ed_divest_max.get((e, d_div), 0.0)
                for d_div in e_div_periods
                if p_years_d.get(d_div, 0.0) <= d_years
            )
            if not in_total:
                v = period_sum
            elif not e_div_periods:
                v = e_total_max
            else:
                v = max(period_sum, e_total_max)
            rows.append((e, d, v))
    _write_keyed_2(solve_data_dir / "p_entity_divest_cumulative_max.csv",
                   ("entity", "period", "value"), rows)


def write_ed_period_params(input_dir: Path, solve_data_dir: Path) -> None:
    """ed_*_period / ed_cumulative_* family — keyed on ed_invest / ed_divest.

    Must run AFTER invest_divest_sets has produced solve_data/ed_invest.csv
    and ed_divest.csv. Mod's `if e in process then pdProcess[e, P, d]` and
    similar for node; processes and nodes are disjoint so exactly one
    branch fires.
    """
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

    ed_invest_pairs = _read_ed_pairs(solve_data_dir / "ed_invest.csv")
    ed_divest_pairs = _read_ed_pairs(solve_data_dir / "ed_divest.csv")

    for fname, src_pairs, mod_param in (
        ("ed_invest_max_period.csv",     ed_invest_pairs, "invest_max_period"),
        ("ed_invest_min_period.csv",     ed_invest_pairs, "invest_min_period"),
        ("ed_divest_max_period.csv",     ed_divest_pairs, "retire_max_period"),
        ("ed_divest_min_period.csv",     ed_divest_pairs, "retire_min_period"),
        ("ed_cumulative_max_capacity.csv", ed_invest_pairs, "cumulative_max_capacity"),
        ("ed_cumulative_min_capacity.csv", ed_invest_pairs, "cumulative_min_capacity"),
    ):
        rows: list[tuple[str, str, float]] = []
        for e, d in src_pairs:
            if e in process_set:
                v = pp.get(e, mod_param, d)
            elif e in node_set:
                v = pn.get(e, mod_param, d)
            else:
                v = 0.0
            rows.append((e, d, v))
        _write_keyed_2(solve_data_dir / fname,
                       ("entity", "period", "value"), rows)


def _read_ed_pairs(path: Path) -> list[tuple[str, str]]:
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


def write_pdtProcess(input_dir: Path, solve_data_dir: Path) -> None:
    """flextool.mod L1227 — pdtProcess: 7-branch fallback over pbt/pd/pt/p
    + processParam_def1 default-1 + 0.

    Output: ``solve_data/pdtProcess.csv`` indexed by (process, param, period, time).
    Domain: ``process_TimeParam_in_use × dt``.
    """
    lookup = PdtLookup(
        pbt_csv=input_dir / "pbt_process.csv",
        pd_csv=input_dir / "pd_process.csv",
        pt_csv=input_dir / "pt_process.csv",
        p_csv=input_dir / "p_process.csv",
        period_time_first_csv=solve_data_dir / "first_timesteps.csv",
        solve_branch_csv=solve_data_dir / "solve_branch__time_branch.csv",
        period_branch_csv=solve_data_dir / "period__branch.csv",
        group_entity_csv=solve_data_dir / "group_process.csv",
        group_stochastic_csv=input_dir / "groupIncludeStochastics.csv",
        param_def1=PROCESS_PARAM_DEF1,
    )
    domain = _read_pairs(solve_data_dir / "process_TimeParam_in_use.csv")
    dt = _read_pairs(solve_data_dir / "steps_in_use.csv")
    out_path = solve_data_dir / "pdtProcess.csv"
    with out_path.open("w") as fh:
        fh.write("process,param,period,time,value\n")
        for (p, param) in domain:
            for (d, t) in dt:
                v = lookup.get(p, param, d, t)
                fh.write(f"{p},{param},{d},{t},{repr(v)}\n")


def _read_triples(path: Path) -> list[tuple[str, str, str]]:
    if not path.exists():
        return []
    out: list[tuple[str, str, str]] = []
    with path.open() as fh:
        reader = csv.reader(fh)
        next(reader, None)
        for row in reader:
            if len(row) >= 3 and row[0] and row[1] and row[2]:
                out.append((row[0], row[1], row[2]))
    return out


def write_pdtProcess_source(input_dir: Path, solve_data_dir: Path) -> None:
    """flextool.mod L1265 — pdtProcess_source: 6-branch fallback (no def1).

    Domain: ``process_source_sourceSinkTimeParam_in_use × dt``.
    """
    lookup = PdtLookupPerSide(
        pbt_csv=input_dir / "pbt_process_source.csv",
        pd_csv=input_dir / "pd_process_source.csv",
        pt_csv=input_dir / "pt_process_source.csv",
        p_csv=input_dir / "p_process_source.csv",
        period_time_first_csv=solve_data_dir / "first_timesteps.csv",
        solve_branch_csv=solve_data_dir / "solve_branch__time_branch.csv",
        period_branch_csv=solve_data_dir / "period__branch.csv",
        group_process_csv=solve_data_dir / "group_process.csv",
        group_stochastic_csv=input_dir / "groupIncludeStochastics.csv",
    )
    domain = _read_triples(solve_data_dir / "process_source_sourceSinkTimeParam_in_use.csv")
    dt = _read_pairs(solve_data_dir / "steps_in_use.csv")
    out_path = solve_data_dir / "pdtProcess_source.csv"
    with out_path.open("w") as fh:
        fh.write("process,source,param,period,time,value\n")
        for (p, src, param) in domain:
            for (d, t) in dt:
                v = lookup.get(p, src, param, d, t)
                fh.write(f"{p},{src},{param},{d},{t},{repr(v)}\n")


def write_pdtProcess_sink(input_dir: Path, solve_data_dir: Path) -> None:
    """flextool.mod L1279 — pdtProcess_sink: 6-branch fallback (no def1).

    Domain: ``process_sink_sourceSinkTimeParam_in_use × dt``.
    """
    lookup = PdtLookupPerSide(
        pbt_csv=input_dir / "pbt_process_sink.csv",
        pd_csv=input_dir / "pd_process_sink.csv",
        pt_csv=input_dir / "pt_process_sink.csv",
        p_csv=input_dir / "p_process_sink.csv",
        period_time_first_csv=solve_data_dir / "first_timesteps.csv",
        solve_branch_csv=solve_data_dir / "solve_branch__time_branch.csv",
        period_branch_csv=solve_data_dir / "period__branch.csv",
        group_process_csv=solve_data_dir / "group_process.csv",
        group_stochastic_csv=input_dir / "groupIncludeStochastics.csv",
    )
    domain = _read_triples(solve_data_dir / "process_sink_sourceSinkTimeParam_in_use.csv")
    dt = _read_pairs(solve_data_dir / "steps_in_use.csv")
    out_path = solve_data_dir / "pdtProcess_sink.csv"
    with out_path.open("w") as fh:
        fh.write("process,sink,param,period,time,value\n")
        for (p, snk, param) in domain:
            for (d, t) in dt:
                v = lookup.get(p, snk, param, d, t)
                fh.write(f"{p},{snk},{param},{d},{t},{repr(v)}\n")


def _read_pdt_at_param(path: Path, param_col: int, param_value: str,
                       key_cols: tuple[int, ...], val_col: int) -> dict[tuple, float]:
    """Read a long-format pdtX CSV, filter rows where col[param_col] == param_value,
    return dict[tuple(row[c] for c in key_cols)] = float(row[val_col]).
    """
    out: dict[tuple, float] = {}
    if not path.exists():
        return out
    with path.open() as fh:
        reader = csv.reader(fh)
        next(reader, None)
        for row in reader:
            if len(row) > max(param_col, val_col, *key_cols) and row[param_col] == param_value:
                try:
                    out[tuple(row[c] for c in key_cols)] = float(row[val_col])
                except ValueError:
                    continue
    return out


def write_pdtProcess__source__sink__dt_varCost_pair(
    input_dir: Path, solve_data_dir: Path
) -> None:
    """flextool.mod L1493, L1502 — two `varCost` calc params keyed on
    process_source_sink and process_source_sink_alwaysProcess.

    Both sum: per-side ``other_operational_cost`` (gated by
    process_source / process_sink membership) plus ``pdtProcess['OOC']``.
    The ``_alwaysProcess`` variant additionally gates the third term on
    ``(p, sink) in process_sink || (p, sink) in process_source``.
    """
    pdt = _read_pdt_at_param(
        solve_data_dir / "pdtProcess.csv",
        param_col=1, param_value="other_operational_cost",
        key_cols=(0, 2, 3), val_col=4,
    )  # (process, period, time) → value
    pdt_src = _read_pdt_at_param(
        solve_data_dir / "pdtProcess_source.csv",
        param_col=2, param_value="other_operational_cost",
        key_cols=(0, 1, 3, 4), val_col=5,
    )  # (process, source, period, time) → value
    pdt_snk = _read_pdt_at_param(
        solve_data_dir / "pdtProcess_sink.csv",
        param_col=2, param_value="other_operational_cost",
        key_cols=(0, 1, 3, 4), val_col=5,
    )  # (process, sink, period, time) → value
    proc_src = frozenset(_read_pairs(input_dir / "process__source.csv"))
    proc_snk = frozenset(_read_pairs(input_dir / "process__sink.csv"))
    pss = _read_triples(solve_data_dir / "process_source_sink.csv")
    pss_always = _read_triples(solve_data_dir / "process_source_sink_alwaysProcess.csv")
    dt = _read_pairs(solve_data_dir / "steps_in_use.csv")

    out_basic = solve_data_dir / "pdtProcess__source__sink__dt_varCost.csv"
    with out_basic.open("w") as fh:
        fh.write("process,source,sink,period,time,value\n")
        for (p, src, snk) in pss:
            for (d, t) in dt:
                v = 0.0
                if (p, src) in proc_src:
                    v += pdt_src.get((p, src, d, t), 0.0)
                if (p, snk) in proc_snk:
                    v += pdt_snk.get((p, snk, d, t), 0.0)
                v += pdt.get((p, d, t), 0.0)
                fh.write(f"{p},{src},{snk},{d},{t},{repr(v)}\n")

    out_always = solve_data_dir / "pdtProcess__source__sink__dt_varCost_alwaysProcess.csv"
    with out_always.open("w") as fh:
        fh.write("process,source,sink,period,time,value\n")
        for (p, src, snk) in pss_always:
            for (d, t) in dt:
                v = 0.0
                if (p, src) in proc_src:
                    v += pdt_src.get((p, src, d, t), 0.0)
                if (p, snk) in proc_snk:
                    v += pdt_snk.get((p, snk, d, t), 0.0)
                # mod gate: ((p, sink) in process_sink || (p, sink) in process_source)
                if (p, snk) in proc_snk or (p, snk) in proc_src:
                    v += pdt.get((p, d, t), 0.0)
                fh.write(f"{p},{src},{snk},{d},{t},{repr(v)}\n")

