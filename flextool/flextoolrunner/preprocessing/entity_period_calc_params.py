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
    NODE_PARAM_DEF1,
    read_class_defaults,
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


def write_pdtNode(input_dir: Path, solve_data_dir: Path) -> None:
    """flextool.mod L1176 — pdtNode: 9-branch fallback over pbt/pt/pd/p
    + nodeParam_def1 default-1 + class_paramName_default fallback + 0.

    Differs from pdtProcess in two ways:
      * time axis is checked BEFORE period axis (mod L1182-1185)
      * extra ``('node', param) in class_paramName_default`` branch
        before the final 0 fallback (input/default_values.csv)

    Output: ``solve_data/pdtNode.csv`` indexed by (node, param, period, time).
    Domain: ``node__TimeParam_in_use × dt``.
    """
    lookup = PdtLookup(
        pbt_csv=input_dir / "pbt_node.csv",
        pd_csv=input_dir / "pd_node.csv",
        pt_csv=input_dir / "pt_node.csv",
        p_csv=input_dir / "p_node.csv",
        period_time_first_csv=solve_data_dir / "first_timesteps.csv",
        solve_branch_csv=solve_data_dir / "solve_branch__time_branch.csv",
        period_branch_csv=solve_data_dir / "period__branch.csv",
        group_entity_csv=solve_data_dir / "group_node.csv",
        group_stochastic_csv=input_dir / "groupIncludeStochastics.csv",
        param_def1=NODE_PARAM_DEF1,
        time_first_priority=True,
        class_default_values=read_class_defaults(
            input_dir / "default_values.csv", "node"
        ),
    )
    domain = _read_pairs(solve_data_dir / "node__TimeParam_in_use.csv")
    dt = _read_pairs(solve_data_dir / "steps_in_use.csv")
    out_path = solve_data_dir / "pdtNode.csv"
    with out_path.open("w") as fh:
        fh.write("node,param,period,time,value\n")
        for (n, param) in domain:
            for (d, t) in dt:
                v = lookup.get(n, param, d, t)
                fh.write(f"{n},{param},{d},{t},{repr(v)}\n")


def write_pdtNodeInflow(input_dir: Path, solve_data_dir: Path) -> None:
    """flextool.mod L1325 — pdtNodeInflow: stochastic / parent-branch fold-in
    OR additive sum over the 4 inflow scaling methods.

    Domain: {n in node, (d, t) in dt : (n, 'no_inflow') not in node__inflow_method}.

    Branch 1 (stochastic): when n belongs to a stochastic group, fold pbt
    inflow rows over (tb in solve_branch[d], ts in period_time_first[d]).

    Branch 2 (parent-period): for each parent period pe of d, fold pbt
    inflow rows over (tb in solve_branch[pe], ts in period_time_first[d]).

    Branch 3 (deterministic): sum of contributions from whichever of the 4
    nodeBalance(∪Period)-gated methods are active for n:
      * scale_to_annual_flow      → period_flow_annual_multiplier[n,d] * ptNode_inflow[n,t]
      * scale_in_proportion       → period_flow_proportional_multiplier[n,d] * ptNode_inflow[n,t]
      * scale_to_annual_and_peak_flow → new_old_slope[n,d] * ptNode_inflow[n,t] - new_old_section[n,d]
      * use_original              → ptNode_inflow[n,t]
    """
    nodes = _read_singles(input_dir / "node.csv")
    dt = _read_pairs(solve_data_dir / "steps_in_use.csv")

    # node__inflow_method (canonical, post-fallback) — set of (node, method)
    inflow_method_pairs = frozenset(
        _read_pairs(solve_data_dir / "node__inflow_method.csv")
    )
    n_balance = frozenset(_read_singles(solve_data_dir / "nodeBalance.csv"))
    n_balance_period = frozenset(_read_singles(solve_data_dir / "nodeBalancePeriod.csv"))
    balance_union = n_balance | n_balance_period

    # Stochastic gate (n via group_node × groupIncludeStochastics)
    groups_stoch = frozenset(_read_singles(input_dir / "groupIncludeStochastics.csv"))
    stoch_node: set[str] = set()
    gn_path = solve_data_dir / "group_node.csv"
    if gn_path.exists():
        with gn_path.open() as fh:
            reader = csv.reader(fh)
            next(reader, None)
            for row in reader:
                if len(row) >= 2 and row[0] in groups_stoch and row[1]:
                    stoch_node.add(row[1])

    # Branch indices (same shape as PdtLookup)
    ts_for_d: dict[str, list[str]] = {}
    fts_path = solve_data_dir / "first_timesteps.csv"
    if fts_path.exists():
        with fts_path.open() as fh:
            reader = csv.reader(fh)
            next(reader, None)
            for row in reader:
                if len(row) >= 2 and row[0] and row[1]:
                    ts_for_d.setdefault(row[0], []).append(row[1])
    tb_for_d: dict[str, list[str]] = {}
    sb_path = solve_data_dir / "solve_branch__time_branch.csv"
    if sb_path.exists():
        with sb_path.open() as fh:
            reader = csv.reader(fh)
            next(reader, None)
            for row in reader:
                if len(row) >= 2 and row[0] and row[1]:
                    tb_for_d.setdefault(row[0], []).append(row[1])
    pe_for_d: dict[str, list[str]] = {}
    pb_path = solve_data_dir / "period__branch.csv"
    if pb_path.exists():
        with pb_path.open() as fh:
            reader = csv.reader(fh)
            next(reader, None)
            for row in reader:
                if len(row) >= 2 and row[0] and row[1]:
                    pe_for_d.setdefault(row[1], []).append(row[0])

    # pbt_node_inflow{(n, branch, ts, t) → value}
    pbt_inflow: dict[tuple[str, str, str, str], float] = {}
    pbt_path = input_dir / "pbt_node_inflow.csv"
    if pbt_path.exists():
        with pbt_path.open() as fh:
            reader = csv.reader(fh)
            next(reader, None)
            for row in reader:
                if len(row) >= 5 and row[0] and row[1] and row[2] and row[3]:
                    try:
                        pbt_inflow[(row[0], row[1], row[2], row[3])] = float(row[4])
                    except ValueError:
                        continue

    # ptNode_inflow{(n, t) → value}
    pt_inflow: dict[tuple[str, str], float] = {}
    pti_path = solve_data_dir / "ptNode_inflow.csv"
    if pti_path.exists():
        with pti_path.open() as fh:
            reader = csv.reader(fh)
            next(reader, None)
            for row in reader:
                if len(row) >= 3 and row[0] and row[1]:
                    try:
                        pt_inflow[(row[0], row[1])] = float(row[2])
                    except ValueError:
                        continue

    # pdNode lookup limited to (annual_flow, peak_inflow) — keyed by (n, param) → {d: value}
    pdNode_af: dict[tuple[str, str], float] = {}
    pdNode_pk: dict[tuple[str, str], float] = {}
    pdn_path = solve_data_dir / "pdNode.csv"
    if pdn_path.exists():
        with pdn_path.open() as fh:
            reader = csv.reader(fh)
            next(reader, None)
            for row in reader:
                if len(row) >= 4 and row[0]:
                    try:
                        v = float(row[3])
                    except ValueError:
                        continue
                    if row[1] == "annual_flow":
                        pdNode_af[(row[0], row[2])] = v
                    elif row[1] == "peak_inflow":
                        pdNode_pk[(row[0], row[2])] = v

    def _read_2_keyed_value(path: Path) -> dict[tuple[str, str], float]:
        out: dict[tuple[str, str], float] = {}
        if not path.exists():
            return out
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

    pfa = _read_2_keyed_value(solve_data_dir / "period_flow_annual_multiplier.csv")
    pfp = _read_2_keyed_value(solve_data_dir / "period_flow_proportional_multiplier.csv")
    nos_slope = _read_2_keyed_value(solve_data_dir / "new_old_slope.csv")
    nos_section = _read_2_keyed_value(solve_data_dir / "new_old_section.csv")

    # Domain filter — skip nodes with 'no_inflow' method
    eligible_nodes = [n for n in nodes if (n, "no_inflow") not in inflow_method_pairs]

    out_path = solve_data_dir / "pdtNodeInflow.csv"
    with out_path.open("w") as fh:
        fh.write("node,period,time,value\n")
        for n in eligible_nodes:
            is_stoch = n in stoch_node
            in_balance = n in balance_union
            has_scale_annual = (n, "scale_to_annual_flow") in inflow_method_pairs
            has_scale_proportion = (n, "scale_in_proportion") in inflow_method_pairs
            has_scale_peak = (n, "scale_to_annual_and_peak_flow") in inflow_method_pairs
            has_use_original = (n, "use_original") in inflow_method_pairs
            for (d, t) in dt:
                # Branch 1: stochastic fold-in
                if is_stoch:
                    total = 0.0
                    hit = False
                    for tb in tb_for_d.get(d, ()):
                        for ts in ts_for_d.get(d, ()):
                            v = pbt_inflow.get((n, tb, ts, t))
                            if v is not None:
                                total += v
                                hit = True
                    if hit:
                        fh.write(f"{n},{d},{t},{repr(total)}\n")
                        continue
                # Branch 2: parent-period fold-in
                pe_list = pe_for_d.get(d, ())
                ts_list = ts_for_d.get(d, ())
                if pe_list and ts_list:
                    total = 0.0
                    hit = False
                    for pe in pe_list:
                        for tb in tb_for_d.get(pe, ()):
                            for ts in ts_list:
                                v = pbt_inflow.get((n, tb, ts, t))
                                if v is not None:
                                    total += v
                                    hit = True
                    if hit:
                        fh.write(f"{n},{d},{t},{repr(total)}\n")
                        continue
                # Branch 3: deterministic additive sum
                value = 0.0
                if in_balance:
                    pti = pt_inflow.get((n, t), 0.0)
                    if has_scale_annual and pdNode_af.get((n, d), 0.0):
                        value += pfa.get((n, d), 0.0) * pti
                    if has_scale_proportion and pdNode_af.get((n, d), 0.0):
                        value += pfp.get((n, d), 0.0) * pti
                    if has_scale_peak \
                            and pdNode_af.get((n, d), 0.0) \
                            and pdNode_pk.get((n, d), 0.0):
                        value += nos_slope.get((n, d), 0.0) * pti \
                                 - nos_section.get((n, d), 0.0)
                    if has_use_original:
                        value += pti
                fh.write(f"{n},{d},{t},{repr(value)}\n")


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


def write_pProcess_source_sink(input_dir: Path, solve_data_dir: Path) -> None:
    """flextool.mod L1249 — pProcess_source_sink: prefer p_process_source,
    fall back to p_process_sink, then 0.

    Domain: process__source__sink__param.
    """
    p_src: dict[tuple[str, str, str], float] = {}
    src_path = input_dir / "p_process_source.csv"
    if src_path.exists():
        with src_path.open() as fh:
            reader = csv.reader(fh)
            next(reader, None)
            for row in reader:
                if len(row) >= 4 and all(row[i] for i in range(3)):
                    try:
                        p_src[(row[0], row[1], row[2])] = float(row[3])
                    except ValueError:
                        continue
    p_snk: dict[tuple[str, str, str], float] = {}
    snk_path = input_dir / "p_process_sink.csv"
    if snk_path.exists():
        with snk_path.open() as fh:
            reader = csv.reader(fh)
            next(reader, None)
            for row in reader:
                if len(row) >= 4 and all(row[i] for i in range(3)):
                    try:
                        p_snk[(row[0], row[1], row[2])] = float(row[3])
                    except ValueError:
                        continue
    proc_src_keys = frozenset(p_src.keys())
    proc_snk_keys = frozenset(p_snk.keys())

    domain: list[tuple[str, str, str, str]] = []
    pssp = solve_data_dir / "process__source__sink__param.csv"
    if pssp.exists():
        with pssp.open() as fh:
            reader = csv.reader(fh)
            next(reader, None)
            for row in reader:
                if len(row) >= 4 and all(row[i] for i in range(4)):
                    domain.append((row[0], row[1], row[2], row[3]))

    out_path = solve_data_dir / "pProcess_source_sink.csv"
    with out_path.open("w") as fh:
        fh.write("process,source,sink,param,value\n")
        for (p, src, snk, param) in domain:
            if (p, src, param) in proc_src_keys:
                v = p_src[(p, src, param)]
            elif (p, snk, param) in proc_snk_keys:
                v = p_snk[(p, snk, param)]
            else:
                v = 0.0
            fh.write(f"{p},{src},{snk},{param},{repr(v)}\n")


def _read_p_2(path: Path) -> dict[tuple[str, str], float]:
    out: dict[tuple[str, str], float] = {}
    if not path.exists():
        return out
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


def _read_pd_2(path: Path) -> dict[tuple[str, str, str], float]:
    out: dict[tuple[str, str, str], float] = {}
    if not path.exists():
        return out
    with path.open() as fh:
        reader = csv.reader(fh)
        next(reader, None)
        for row in reader:
            if len(row) >= 4 and all(row[i] for i in range(3)):
                try:
                    out[(row[0], row[1], row[2])] = float(row[3])
                except ValueError:
                    continue
    return out


def _read_pt_2(path: Path) -> dict[tuple[str, str, str], float]:
    return _read_pd_2(path)  # same shape: (e, param, t/d, value)


# flextool_base.dat L196-L201, L204-L209, L210
GROUP_PARAM = frozenset((
    "has_capacity_margin", "capacity_margin", "has_inertia", "inertia_limit",
    "invest_max_total", "invest_min_total", "invest_max_period", "invest_min_period",
    "retire_max_total", "retire_min_total", "retire_max_period", "retire_min_period",
    "non_synchronous_limit", "co2_price", "co2_max_period", "co2_max_total",
    "penalty_inertia", "penalty_non_synchronous", "max_cumulative_flow", "min_cumulative_flow",
    "max_instant_flow", "min_instant_flow", "output_nodeGroup_indicators",
    "output_flowGroup_indicators", "penalty_capacity_margin",
    "cumulative_max_capacity", "cumulative_min_capacity", "new_stepduration",
))
GROUP_PERIOD_PARAM = frozenset((
    "capacity_margin", "co2_price", "co2_max_period", "co2_max_total",
    "inertia_limit", "invest_max_period", "invest_min_period",
    "max_cumulative_flow", "min_cumulative_flow", "non_synchronous_limit",
    "penalty_inertia", "penalty_non_synchronous",
    "max_instant_flow", "min_instant_flow", "penalty_capacity_margin",
    "retire_max_period", "retire_min_period",
    "cumulative_max_capacity", "cumulative_min_capacity",
))
GROUP_TIME_PARAM = frozenset(("co2_price", "max_instant_flow", "min_instant_flow"))
# pdGroup specific: param-specific 5000 default (mod L1122-1123)
GROUP_PARAM_DEFAULT_5000 = frozenset((
    "penalty_inertia", "penalty_capacity_margin", "penalty_non_synchronous",
))


def write_pdGroup(input_dir: Path, solve_data_dir: Path) -> None:
    """flextool.mod L1115 — pdGroup: 5-branch fallback.

        if (g, param, d) in group__param__period: pd_group[g, param, d]
        else if exists fold via period__branch: sum pd_group[g, param, db]
        else if (g, param) in group__param: p_group[g, param]
        else if param in {penalty_inertia, penalty_capacity_margin,
                          penalty_non_synchronous}: 5000
        else 0;
    """
    pd_g = _read_pd_2(input_dir / "pd_group.csv")
    p_g = _read_p_2(input_dir / "p_group.csv")
    branches_for_d: dict[str, list[str]] = {}
    pb_path = solve_data_dir / "period__branch.csv"
    if pb_path.exists():
        with pb_path.open() as fh:
            reader = csv.reader(fh)
            next(reader, None)
            for row in reader:
                if len(row) >= 2 and row[0] and row[1]:
                    branches_for_d.setdefault(row[1], []).append(row[0])
    groups = _read_singles(input_dir / "group.csv")
    period_in_use = _read_singles(solve_data_dir / "period_in_use_set.csv")

    out_path = solve_data_dir / "pdGroup.csv"
    with out_path.open("w") as fh:
        fh.write("group,param,period,value\n")
        for g in groups:
            for param in GROUP_PERIOD_PARAM:
                for d in period_in_use:
                    if (g, param, d) in pd_g:
                        v = pd_g[(g, param, d)]
                    else:
                        branched_vals = [
                            pd_g[(g, param, db)] for db in branches_for_d.get(d, ())
                            if (g, param, db) in pd_g
                        ]
                        if branched_vals:
                            v = sum(branched_vals)
                        elif (g, param) in p_g:
                            v = p_g[(g, param)]
                        elif param in GROUP_PARAM_DEFAULT_5000:
                            v = 5000.0
                        else:
                            v = 0.0
                    fh.write(f"{g},{param},{d},{repr(v)}\n")


def write_pdtGroup(input_dir: Path, solve_data_dir: Path) -> None:
    """flextool.mod L1126 — pdtGroup: 4-branch fallback (pt → pd → p → 0)."""
    pt_g = _read_pt_2(input_dir / "pt_group.csv")
    pd_g = _read_pd_2(input_dir / "pd_group.csv")
    p_g = _read_p_2(input_dir / "p_group.csv")
    groups = _read_singles(input_dir / "group.csv")
    dt = _read_pairs(solve_data_dir / "steps_in_use.csv")

    out_path = solve_data_dir / "pdtGroup.csv"
    with out_path.open("w") as fh:
        fh.write("group,param,period,time,value\n")
        for g in groups:
            for param in GROUP_TIME_PARAM:
                for (d, t) in dt:
                    if (g, param, t) in pt_g:
                        v = pt_g[(g, param, t)]
                    elif (g, param, d) in pd_g:
                        v = pd_g[(g, param, d)]
                    elif (g, param) in p_g:
                        v = p_g[(g, param)]
                    else:
                        v = 0.0
                    fh.write(f"{g},{param},{d},{t},{repr(v)}\n")


def write_pdCommodity(input_dir: Path, solve_data_dir: Path) -> None:
    """flextool.mod L1101 — pdCommodity: 3-branch fallback (pd →
    period__branch fold → p_commodity[c,param]).

    Note: mod has no else 0; p_commodity has table default 0.
    """
    pd_c = _read_pd_2(input_dir / "pd_commodity.csv")
    p_c = _read_p_2(input_dir / "p_commodity.csv")
    branches_for_d: dict[str, list[str]] = {}
    pb_path = solve_data_dir / "period__branch.csv"
    if pb_path.exists():
        with pb_path.open() as fh:
            reader = csv.reader(fh)
            next(reader, None)
            for row in reader:
                if len(row) >= 2 and row[0] and row[1]:
                    branches_for_d.setdefault(row[1], []).append(row[0])
    commodities = _read_singles(input_dir / "commodity.csv")
    period_in_use = _read_singles(solve_data_dir / "period_in_use_set.csv")

    out_path = solve_data_dir / "pdCommodity.csv"
    with out_path.open("w") as fh:
        fh.write("commodity,param,period,value\n")
        for c in commodities:
            for param in ("price",):  # commodityPeriodParam = {price}
                for d in period_in_use:
                    if (c, param, d) in pd_c:
                        v = pd_c[(c, param, d)]
                    else:
                        branched_vals = [
                            pd_c[(c, param, db)] for db in branches_for_d.get(d, ())
                            if (c, param, db) in pd_c
                        ]
                        if branched_vals:
                            v = sum(branched_vals)
                        else:
                            v = p_c.get((c, param), 0.0)
                    fh.write(f"{c},{param},{d},{repr(v)}\n")


def write_pdtCommodity(input_dir: Path, solve_data_dir: Path) -> None:
    """flextool.mod L1108 — pdtCommodity: 3-branch fallback (pt → pd → p → 0).

    Domain: commodity × commodityTimeParam × dt.
    commodityTimeParam = {price} (flextool_base.dat L134).
    """
    pt: dict[tuple[str, str, str], float] = {}
    pt_path = input_dir / "pt_commodity.csv"
    if pt_path.exists():
        with pt_path.open() as fh:
            reader = csv.reader(fh)
            next(reader, None)
            for row in reader:
                if len(row) >= 4 and all(row[i] for i in range(3)):
                    try:
                        pt[(row[0], row[1], row[2])] = float(row[3])
                    except ValueError:
                        continue
    pd_: dict[tuple[str, str, str], float] = {}
    pd_path = input_dir / "pd_commodity.csv"
    if pd_path.exists():
        with pd_path.open() as fh:
            reader = csv.reader(fh)
            next(reader, None)
            for row in reader:
                if len(row) >= 4 and all(row[i] for i in range(3)):
                    try:
                        pd_[(row[0], row[1], row[2])] = float(row[3])
                    except ValueError:
                        continue
    p: dict[tuple[str, str], float] = {}
    p_path = input_dir / "p_commodity.csv"
    if p_path.exists():
        with p_path.open() as fh:
            reader = csv.reader(fh)
            next(reader, None)
            for row in reader:
                if len(row) >= 3 and row[0] and row[1]:
                    try:
                        p[(row[0], row[1])] = float(row[2])
                    except ValueError:
                        continue

    commodities = _read_singles(input_dir / "commodity.csv")
    dt = _read_pairs(solve_data_dir / "steps_in_use.csv")

    out_path = solve_data_dir / "pdtCommodity.csv"
    with out_path.open("w") as fh:
        fh.write("commodity,param,period,time,value\n")
        for c in commodities:
            for param in ("price",):  # commodityTimeParam
                for (d, t) in dt:
                    v = pt.get((c, param, t))
                    if v is None:
                        v = pd_.get((c, param, d))
                    if v is None:
                        v = p.get((c, param), 0.0)
                    fh.write(f"{c},{param},{d},{t},{repr(v)}\n")


def write_cap_reduction_params(input_dir: Path, solve_data_dir: Path) -> None:
    """flextool.mod L1637-1663 — Morales-Espana startup/shutdown capacity
    reduction params (4 calc params, 1 per side × startup/shutdown).

        if p_process_<side>[p, side_n, 'ramp_speed_<dir>'] > 0 then
            max(0, 1 - p_process[p, 'min_load']
                    - p_process_<side>[..,'ramp_speed_<dir>'] * 60 * step_duration[d, t])
        else 0;

    Domain: (p, side_n) in process_<side> × dt, restricted to p in
    process_online.
    """
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

    def _read_p_side(path: Path) -> dict[tuple[str, str, str], float]:
        out: dict[tuple[str, str, str], float] = {}
        if not path.exists():
            return out
        with path.open() as fh:
            reader = csv.reader(fh)
            next(reader, None)
            for row in reader:
                if len(row) >= 4 and all(row[i] for i in range(3)):
                    try:
                        out[(row[0], row[1], row[2])] = float(row[3])
                    except ValueError:
                        continue
        return out

    p_process_source = _read_p_side(input_dir / "p_process_source.csv")
    p_process_sink = _read_p_side(input_dir / "p_process_sink.csv")

    process_online = frozenset(_read_singles(solve_data_dir / "process_online.csv"))
    proc_src = _read_pairs(input_dir / "process__source.csv")
    proc_snk = _read_pairs(input_dir / "process__sink.csv")

    # dt with step_duration
    dt_with_dur: list[tuple[str, str, float]] = []
    su_path = solve_data_dir / "steps_in_use.csv"
    if su_path.exists():
        with su_path.open() as fh:
            reader = csv.reader(fh)
            next(reader, None)
            for row in reader:
                if len(row) >= 3 and row[0] and row[1]:
                    try:
                        dt_with_dur.append((row[0], row[1], float(row[2])))
                    except ValueError:
                        continue

    def _compute(p_side: dict, pairs: list[tuple[str, str]],
                 ramp_param: str) -> list[tuple[str, str, str, str, float]]:
        rows: list[tuple[str, str, str, str, float]] = []
        for (p, side_n) in pairs:
            if p not in process_online:
                continue
            ramp = p_side.get((p, side_n, ramp_param), 0.0)
            if ramp <= 0:
                # Branch evaluates to 0 for all (d, t).
                for (d, t, _dur) in dt_with_dur:
                    rows.append((p, side_n, d, t, 0.0))
                continue
            min_load = p_process.get((p, "min_load"), 0.0)
            for (d, t, dur) in dt_with_dur:
                v = max(0.0, 1.0 - min_load - ramp * 60.0 * dur)
                rows.append((p, side_n, d, t, v))
        return rows

    def _write(path: Path, side_label: str,
               rows: list[tuple[str, str, str, str, float]]) -> None:
        with path.open("w") as fh:
            fh.write(f"process,{side_label},period,time,value\n")
            for (p, sn, d, t, v) in rows:
                fh.write(f"{p},{sn},{d},{t},{repr(v)}\n")

    _write(solve_data_dir / "p_startup_cap_reduction_sink.csv", "sink",
           _compute(p_process_sink, proc_snk, "ramp_speed_up"))
    _write(solve_data_dir / "p_shutdown_cap_reduction_sink.csv", "sink",
           _compute(p_process_sink, proc_snk, "ramp_speed_down"))
    _write(solve_data_dir / "p_startup_cap_reduction_source.csv", "source",
           _compute(p_process_source, proc_src, "ramp_speed_up"))
    _write(solve_data_dir / "p_shutdown_cap_reduction_source.csv", "source",
           _compute(p_process_source, proc_src, "ramp_speed_down"))


def write_pssdt_varCost_filters(input_dir: Path, solve_data_dir: Path) -> None:
    """flextool.mod L1498-1501 — four filter sets keyed on pdt-* OOC values.

        set pssdt_varCost_noEff := {process_source_sink_noEff × dt :
                pdtProcess__source__sink__dt_varCost[p,src,snk,d,t]};
        set pssdt_varCost_eff_unit_source := {process_source_sink_eff × dt :
                (p,src) in process_source AND pdtProcess_source[p,src,'OOC',d,t]};
        set pssdt_varCost_eff_unit_sink := {process_source_sink_eff × dt :
                (p,snk) in process_sink AND pdtProcess_sink[p,snk,'OOC',d,t]};
        set pssdt_varCost_eff_connection := {process_source_sink_eff × dt :
                pdtProcess[p,'OOC',d,t]};

    Each row is included only when the gating value is non-zero.
    """
    pdt = _read_pdt_at_param(
        solve_data_dir / "pdtProcess.csv",
        param_col=1, param_value="other_operational_cost",
        key_cols=(0, 2, 3), val_col=4,
    )
    pdt_src = _read_pdt_at_param(
        solve_data_dir / "pdtProcess_source.csv",
        param_col=2, param_value="other_operational_cost",
        key_cols=(0, 1, 3, 4), val_col=5,
    )
    pdt_snk = _read_pdt_at_param(
        solve_data_dir / "pdtProcess_sink.csv",
        param_col=2, param_value="other_operational_cost",
        key_cols=(0, 1, 3, 4), val_col=5,
    )
    varcost: dict[tuple[str, str, str, str, str], float] = {}
    vp = solve_data_dir / "pdtProcess__source__sink__dt_varCost.csv"
    if vp.exists():
        with vp.open() as fh:
            reader = csv.reader(fh)
            next(reader, None)
            for row in reader:
                if len(row) >= 6 and all(row[i] for i in range(5)):
                    try:
                        varcost[(row[0], row[1], row[2], row[3], row[4])] = float(row[5])
                    except ValueError:
                        continue

    proc_src = frozenset(_read_pairs(input_dir / "process__source.csv"))
    proc_snk = frozenset(_read_pairs(input_dir / "process__sink.csv"))
    pss_noEff = _read_triples(solve_data_dir / "process_source_sink_noEff.csv")
    pss_eff = _read_triples(solve_data_dir / "process_source_sink_eff.csv")
    dt = _read_pairs(solve_data_dir / "steps_in_use.csv")

    # pssdt_varCost_noEff
    rows: list[tuple[str, str, str, str, str]] = []
    for (p, src, snk) in pss_noEff:
        for (d, t) in dt:
            if varcost.get((p, src, snk, d, t), 0.0):
                rows.append((p, src, snk, d, t))
    _write_5col(solve_data_dir / "pssdt_varCost_noEff.csv",
                ("process", "source", "sink", "period", "time"), rows)

    # pssdt_varCost_eff_unit_source
    rows = []
    for (p, src, snk) in pss_eff:
        for (d, t) in dt:
            if (p, src) in proc_src and pdt_src.get((p, src, d, t), 0.0):
                rows.append((p, src, snk, d, t))
    _write_5col(solve_data_dir / "pssdt_varCost_eff_unit_source.csv",
                ("process", "source", "sink", "period", "time"), rows)

    # pssdt_varCost_eff_unit_sink
    rows = []
    for (p, src, snk) in pss_eff:
        for (d, t) in dt:
            if (p, snk) in proc_snk and pdt_snk.get((p, snk, d, t), 0.0):
                rows.append((p, src, snk, d, t))
    _write_5col(solve_data_dir / "pssdt_varCost_eff_unit_sink.csv",
                ("process", "source", "sink", "period", "time"), rows)

    # pssdt_varCost_eff_connection
    rows = []
    for (p, src, snk) in pss_eff:
        for (d, t) in dt:
            if pdt.get((p, d, t), 0.0):
                rows.append((p, src, snk, d, t))
    _write_5col(solve_data_dir / "pssdt_varCost_eff_connection.csv",
                ("process", "source", "sink", "period", "time"), rows)


def _write_5col(path: Path, header: tuple[str, ...], rows: list[tuple]) -> None:
    path.write_text(",".join(header) + "\n"
                    + "".join(",".join(r) + "\n" for r in rows))


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

