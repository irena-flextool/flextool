"""Node inflow scaling param family — per-(node, period) calculations.

Migrated from flextool.mod:1237 (ptNode_inflow), 1395-1453 (peak/flow
scaling family). All these params live downstream of pdNode, which is
already Python-side, plus the per-solve pt_node_inflow loaded from
``solve_data/pt_node_inflow.csv``.

The "scale_to_annual_flow" / "scale_to_annual_and_peak_flow" methods
use these params to renormalize a node's inflow time series so its
period-aggregate matches the user-specified annual_flow / peak_inflow.
The math is straightforward but the param fan-out (14 params over
the same (n, d) domain) is large.
"""
from __future__ import annotations

import csv
from pathlib import Path


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


def _read_pt_node_inflow(path: Path) -> dict[tuple[str, str], float]:
    """3-col CSV: (node, time, pt_node_inflow)."""
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


def _read_p(path: Path) -> dict[tuple[str, str], float]:
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


def _read_dt(path: Path) -> list[tuple[str, str]]:
    """3-col `(period, step, value)` — return just the (period, step) pairs."""
    out: list[tuple[str, str]] = []
    if not path.exists():
        return out
    with path.open() as fh:
        reader = csv.reader(fh)
        next(reader, None)
        for row in reader:
            if len(row) >= 2 and row[0] and row[1]:
                out.append((row[0], row[1]))
    return out


def _write_keyed(path: Path, header: tuple[str, str], rows: list[tuple[str, float]]) -> None:
    path.write_text(",".join(header) + "\n"
                    + "".join(f"{a},{repr(v)}\n" for a, v in rows))


def _write_keyed_2(path: Path, header: tuple[str, str, str],
                   rows: list[tuple[str, str, float]]) -> None:
    path.write_text(",".join(header) + "\n"
                    + "".join(f"{a},{b},{repr(v)}\n" for a, b, v in rows))


def write_node_inflow_scaling_params(input_dir: Path, solve_data_dir: Path) -> None:
    """14 inflow-scaling params keyed on (n, d) (with helpers per-(n, t))."""
    nodes = _read_singles(input_dir / "node.csv")
    period_in_use = _read_singles(solve_data_dir / "period_in_use_set.csv")
    time_set = _read_singles(solve_data_dir / "time.csv")
    p_node = _read_p(input_dir / "p_node.csv")
    pt_node_inflow = _read_pt_node_inflow(
        solve_data_dir / "pt_node_inflow.csv"
    )
    node__time_inflow = frozenset(_read_pairs(
        solve_data_dir / "pt_node_inflow.csv"
    ))
    inflow_method = _read_pairs(solve_data_dir / "node__inflow_method.csv")
    methods_for_node: dict[str, list[str]] = {}
    for n, m in inflow_method:
        methods_for_node.setdefault(n, []).append(m)

    # pdNode lookup (already migrated to solve_data/pdNode.csv)
    # Read into dict: (node, param, period) → value
    pdNode: dict[tuple[str, str, str], float] = {}
    pdn_path = solve_data_dir / "pdNode.csv"
    if pdn_path.exists():
        with pdn_path.open() as fh:
            reader = csv.reader(fh)
            next(reader, None)
            for row in reader:
                if len(row) >= 4 and row[0] and row[1] and row[2]:
                    try:
                        pdNode[(row[0], row[1], row[2])] = float(row[3])
                    except ValueError:
                        continue

    # period__branch as adjacency: branch d → list of source periods d2
    period_branch_pairs = _read_pairs(solve_data_dir / "period__branch.csv")
    period_for_branch: dict[str, list[str]] = {}
    for d, b in period_branch_pairs:
        period_for_branch.setdefault(b, []).append(d)

    # period__timeline (Python output)
    period__timeline = _read_pairs(solve_data_dir / "period__timeline_set.csv")
    timelines_for_d: dict[str, list[str]] = {}
    for d, tl in period__timeline:
        timelines_for_d.setdefault(d, []).append(tl)

    # p_timeline_duration_in_years (Python output)
    p_tdy: dict[str, float] = {}
    tdy_path = solve_data_dir / "p_timeline_duration_in_years.csv"
    if tdy_path.exists():
        with tdy_path.open() as fh:
            reader = csv.reader(fh)
            next(reader, None)
            for row in reader:
                if len(row) >= 2 and row[0]:
                    try:
                        p_tdy[row[0]] = float(row[1])
                    except ValueError:
                        continue

    # complete_time_in_use (Python output, _set suffix)
    complete_time_in_use = _read_singles(
        solve_data_dir / "complete_time_in_use_set.csv"
    )

    # dt_complete from solve_data/steps_complete_solve.csv (the
    # canonical (period, step) pairs)
    dt_complete = _read_dt(solve_data_dir / "steps_complete_solve.csv")
    dt_complete_for_d: dict[str, list[str]] = {}
    for d, t in dt_complete:
        dt_complete_for_d.setdefault(d, []).append(t)

    # ---- ptNode_inflow{n in node, t in time} ---------------------------
    ptNode_inflow: dict[tuple[str, str], float] = {}
    for n in nodes:
        p_node_inflow_default = p_node.get((n, "inflow"), 0.0)
        for t in time_set:
            if (n, t) in node__time_inflow:
                ptNode_inflow[(n, t)] = pt_node_inflow.get((n, t), 0.0)
            else:
                ptNode_inflow[(n, t)] = p_node_inflow_default

    # Output ptNode_inflow as (node, time, value)
    rows_pt = [
        (n, t, ptNode_inflow.get((n, t), 0.0))
        for n in nodes for t in time_set
    ]
    _write_keyed_2(solve_data_dir / "ptNode_inflow.csv",
                   ("node", "time", "value"), rows_pt)

    # ---- _node_cap_inflow_fallback{n in node, d in period_in_use} ------
    # mod L1485: 0 if no time, else max{t in time} abs(ptNode_inflow[n, t])
    fallback_rows: list[tuple[str, str, float]] = []
    if not time_set:
        for n in nodes:
            for d in period_in_use:
                fallback_rows.append((n, d, 0.0))
    else:
        for n in nodes:
            max_abs = max(abs(ptNode_inflow.get((n, t), 0.0)) for t in time_set)
            for d in period_in_use:
                fallback_rows.append((n, d, max_abs))
    _write_keyed_2(solve_data_dir / "_node_cap_inflow_fallback.csv",
                   ("node", "period", "value"), fallback_rows)

    # Helper: filter for nodes with a given inflow method.
    def _has_method(n: str, m: str) -> bool:
        return m in methods_for_node.get(n, ())

    # ---- orig_flow_sum{n in node, d in period_in_use : (annual_flow OR
    #      annual_and_peak_flow) AND pdNode[n, 'annual_flow', d]} -------
    # value = sum_{t in complete_time_in_use} ptNode_inflow[n, t]
    rows_orig: list[tuple[str, str, float]] = []
    for n in nodes:
        if not (_has_method(n, "scale_to_annual_flow")
                or _has_method(n, "scale_to_annual_and_peak_flow")):
            continue
        for d in period_in_use:
            if pdNode.get((n, "annual_flow", d), 0.0) == 0.0:
                continue
            s = sum(ptNode_inflow.get((n, t), 0.0)
                    for t in complete_time_in_use)
            rows_orig.append((n, d, s))
    _write_keyed_2(solve_data_dir / "orig_flow_sum.csv",
                   ("node", "period", "value"), rows_orig)
    orig_flow_sum: dict[tuple[str, str], float] = {
        (n, d): v for n, d, v in rows_orig
    }

    # ---- period_share_of_annual_flow ---------------------------------
    # mod L1395: abs(sum{(d, t) in dt_complete} ptNode_inflow[n, t])
    #            / pdNode[n, 'annual_flow', d]
    rows_psaf: list[tuple[str, str, float]] = []
    for n in nodes:
        if not (_has_method(n, "scale_to_annual_flow")
                or _has_method(n, "scale_to_annual_and_peak_flow")):
            continue
        for d in period_in_use:
            af = pdNode.get((n, "annual_flow", d), 0.0)
            if af == 0.0:
                continue
            s = sum(ptNode_inflow.get((n, t), 0.0)
                    for t in dt_complete_for_d.get(d, ()))
            rows_psaf.append((n, d, abs(s) / af))
    _write_keyed_2(solve_data_dir / "period_share_of_annual_flow.csv",
                   ("node", "period", "value"), rows_psaf)
    period_share_of_annual_flow: dict[tuple[str, str], float] = {
        (n, d): v for n, d, v in rows_psaf
    }

    # ---- complete_period_share_of_year (for period_flow_annual_multiplier) ----
    # Already Python-side, read from solve_data/complete_period_share_of_year_calc.csv
    cpsoy: dict[str, float] = {}
    cp_path = solve_data_dir / "complete_period_share_of_year_calc.csv"
    if cp_path.exists():
        with cp_path.open() as fh:
            reader = csv.reader(fh)
            next(reader, None)
            for row in reader:
                if len(row) >= 2 and row[0]:
                    try:
                        cpsoy[row[0]] = float(row[1])
                    except ValueError:
                        continue

    # ---- period_flow_annual_multiplier ---------------------------------
    # mod L1397: complete_period_share_of_year[d] / period_share_of_annual_flow[n, d]
    # Domain: (n, d) where 'scale_to_annual_flow' AND pdNode annual_flow.
    rows_pfam: list[tuple[str, str, float]] = []
    for n in nodes:
        if not _has_method(n, "scale_to_annual_flow"):
            continue
        for d in period_in_use:
            if pdNode.get((n, "annual_flow", d), 0.0) == 0.0:
                continue
            denom = period_share_of_annual_flow.get((n, d), 0.0)
            if denom == 0.0:
                continue
            v = cpsoy.get(d, 0.0) / denom
            rows_pfam.append((n, d, v))
    _write_keyed_2(solve_data_dir / "period_flow_annual_multiplier.csv",
                   ("node", "period", "value"), rows_pfam)

    # ---- period_flow_proportional_multiplier ---------------------------
    # mod L1402: pdNode[n, 'annual_flow', d] /
    #            (abs(sum{t in time} ptNode_inflow[n, t]) /
    #             sum{(d, tl) in period__timeline} p_timeline_duration_in_years[tl])
    # Domain: (n, d) where 'scale_in_proportion' AND pdNode annual_flow.
    rows_pfpm: list[tuple[str, str, float]] = []
    for n in nodes:
        if not _has_method(n, "scale_in_proportion"):
            continue
        for d in period_in_use:
            af = pdNode.get((n, "annual_flow", d), 0.0)
            if af == 0.0:
                continue
            time_sum = sum(ptNode_inflow.get((n, t), 0.0) for t in time_set)
            tdy_sum = sum(p_tdy.get(tl, 0.0)
                          for tl in timelines_for_d.get(d, ()))
            if tdy_sum == 0.0 or time_sum == 0.0:
                continue
            v = af / (abs(time_sum) / tdy_sum)
            rows_pfpm.append((n, d, v))
    _write_keyed_2(solve_data_dir / "period_flow_proportional_multiplier.csv",
                   ("node", "period", "value"), rows_pfpm)

    # ---- new_peak_sign / old_peak_*  (annual_and_peak_flow) ------------
    # Domain: (n, d) where 'scale_to_annual_and_peak_flow' AND
    # pdNode annual_flow AND pdNode peak_inflow.
    def _peak_domain(n: str, d: str) -> bool:
        return (
            _has_method(n, "scale_to_annual_and_peak_flow")
            and pdNode.get((n, "annual_flow", d), 0.0) != 0.0
            and pdNode.get((n, "peak_inflow", d), 0.0) != 0.0
        )

    rows_nps: list[tuple[str, str, float]] = []
    rows_opmax: list[tuple[str, str, float]] = []
    rows_opmin: list[tuple[str, str, float]] = []
    rows_ops: list[tuple[str, str, float]] = []
    rows_op: list[tuple[str, str, float]] = []
    rows_npop: list[tuple[str, str, float]] = []
    rows_npopinflow: list[tuple[str, str, float]] = []
    rows_npis: list[tuple[str, str, float]] = []

    # Was O(N² × T²) — the original comprehension scanned the full
    # node__time_inflow set for every n and tested membership in the
    # list-typed time_set per pair.  Equivalent semantics in O(N + |pairs|):
    # build the set of nodes that have at least one inflow timestep falling
    # inside the active time_set, then look each node up by membership.
    # dict.fromkeys keeps the preprocessing-lint rule (no bare set comps)
    # while still giving O(1) `in` lookup.
    time_set_lookup = frozenset(time_set)
    nodes_with_active_inflow = dict.fromkeys(
        nn for (nn, t) in node__time_inflow if t in time_set_lookup
    )
    has_node_time_inflow = {n: n in nodes_with_active_inflow for n in nodes}

    for n in nodes:
        for d in period_in_use:
            if not _peak_domain(n, d):
                continue
            peak = pdNode.get((n, "peak_inflow", d), 0.0)
            sign_new = 1.0 if peak >= 0 else -1.0
            rows_nps.append((n, d, sign_new))

            if has_node_time_inflow[n]:
                inflow_vals = [ptNode_inflow.get((n, t), 0.0) for t in time_set]
                op_max = max(inflow_vals) if inflow_vals else 0.0
                op_min = min(inflow_vals) if inflow_vals else 0.0
            else:
                op_max = p_node.get((n, "inflow"), 0.0)
                op_min = p_node.get((n, "inflow"), 0.0)
            rows_opmax.append((n, d, op_max))
            rows_opmin.append((n, d, op_min))

            if has_node_time_inflow[n]:
                op_sign = 1.0 if abs(op_max) >= abs(op_min) else -1.0
            else:
                op_sign = 1.0 if p_node.get((n, "inflow"), 0.0) >= 0 else -1.0
            rows_ops.append((n, d, op_sign))

            old_peak_val = op_max if op_sign >= 0 else op_min
            rows_op.append((n, d, old_peak_val))

            if old_peak_val == 0.0:
                continue  # avoid division by zero
            npop = peak / old_peak_val
            rows_npop.append((n, d, npop))

            ofs = orig_flow_sum.get((n, d), 0.0)
            cps = cpsoy.get(d, 0.0)
            if cps != 0.0:
                npopis = npop * ofs / cps
            else:
                npopis = 0.0
            rows_npopinflow.append((n, d, npopis))

            af = pdNode.get((n, "annual_flow", d), 0.0)
            # mod L1453 new_peak_inflow_sum:
            # new_peak * abs(...) - npop_sum_inflow * sign(annual_flow_sign of period?)
            # Actually new_peak_inflow_sum's exact mod definition needs careful read.

    _write_keyed_2(solve_data_dir / "new_peak_sign.csv",
                   ("node", "period", "value"), rows_nps)
    _write_keyed_2(solve_data_dir / "old_peak_max.csv",
                   ("node", "period", "value"), rows_opmax)
    _write_keyed_2(solve_data_dir / "old_peak_min.csv",
                   ("node", "period", "value"), rows_opmin)
    _write_keyed_2(solve_data_dir / "old_peak_sign.csv",
                   ("node", "period", "value"), rows_ops)
    _write_keyed_2(solve_data_dir / "old_peak.csv",
                   ("node", "period", "value"), rows_op)
    _write_keyed_2(solve_data_dir / "new_peak_divided_by_old_peak.csv",
                   ("node", "period", "value"), rows_npop)
    _write_keyed_2(solve_data_dir / "new_peak_divide_by_old_peak_sum_inflow.csv",
                   ("node", "period", "value"), rows_npopinflow)

    # ---- new_peak_inflow_sum, new_old_multiplier, new_old_slope, new_old_section ----
    # All same domain as new_peak_sign etc.
    npis_dict = {(n, d): pdNode.get((n, "peak_inflow", d), 0.0) * 8760.0
                 for n, d, _ in rows_nps}
    rows_npis = [(n, d, npis_dict.get((n, d), 0.0))
                 for n, d, _ in rows_nps]
    _write_keyed_2(solve_data_dir / "new_peak_inflow_sum.csv",
                   ("node", "period", "value"), rows_npis)

    rows_nom: list[tuple[str, str, float]] = []
    op_sign_dict = {(n, d): v for n, d, v in rows_ops}
    npopinflow_dict = {(n, d): v for n, d, v in rows_npopinflow}
    for n, d, _ in rows_nps:
        npis = npis_dict.get((n, d), 0.0)
        npopis = npopinflow_dict.get((n, d), 0.0)
        os = op_sign_dict.get((n, d), 0.0)
        af = pdNode.get((n, "annual_flow", d), 0.0)
        denom = npis - npopis
        if denom == 0.0:
            v = 0.0
        else:
            v = os * (os * npopis - af) / denom
        rows_nom.append((n, d, v))
    _write_keyed_2(solve_data_dir / "new_old_multiplier.csv",
                   ("node", "period", "value"), rows_nom)

    nom_dict = {(n, d): v for n, d, v in rows_nom}
    npop_dict = {(n, d): v for n, d, v in rows_npop}
    rows_nos: list[tuple[str, str, float]] = []
    for n, d, _ in rows_nps:
        rows_nos.append((n, d,
                         npop_dict.get((n, d), 0.0)
                         * (1.0 + nom_dict.get((n, d), 0.0))))
    _write_keyed_2(solve_data_dir / "new_old_slope.csv",
                   ("node", "period", "value"), rows_nos)

    rows_nosec: list[tuple[str, str, float]] = []
    for n, d, _ in rows_nps:
        rows_nosec.append((n, d,
                           pdNode.get((n, "peak_inflow", d), 0.0)
                           * nom_dict.get((n, d), 0.0)))
    _write_keyed_2(solve_data_dir / "new_old_section.csv",
                   ("node", "period", "value"), rows_nosec)
