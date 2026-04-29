"""LP-scaling row-scalers (Agent 5b/5c) — node and group capacities.

Migrated from flextool.mod:1457-1497. The scaling family computes a
power-of-10 capacity proxy per (node, period) and (group, period) so
the LP solver sees coefficient ranges compressed near O(1). When
``p_use_row_scaling[solve] < 0.5`` the scalers collapse to 1
(Mode A, pre-Agent-5 behaviour).

Migrated params (9):
    _node_cap_unitsize_sum  — sum of process_source_sink unitsizes at n
    _node_cap_raw           — unitsize_sum if > 0, else inflow_fallback, else 1
    _node_cap_pow10         — clamp(1e-6, 1e9) of 10^round(log10(_raw))
    node_capacity_for_scaling — pow10 if scaling on else 1
    inv_node_cap            — 1 / node_capacity_for_scaling
    _group_cap_raw          — sum_{(g,n) in group_node} node_capacity_for_scaling[n, d]
    _group_cap_pow10        — clamp / pow10 of group raw
    group_capacity_for_scaling — pow10 if scaling on else 1
    inv_group_cap           — 1 / group_capacity_for_scaling
"""
from __future__ import annotations

import csv
import math
from pathlib import Path


def _read_singles(path: Path) -> list[str]:
    if not path.exists():
        return []
    with path.open() as fh:
        reader = csv.reader(fh)
        next(reader, None)
        return [r[0] for r in reader if r and r[0]]


def _read_keyed_value(path: Path, value_col_idx: int = 1) -> dict[str, float]:
    out: dict[str, float] = {}
    if not path.exists():
        return out
    with path.open() as fh:
        reader = csv.reader(fh)
        next(reader, None)
        for row in reader:
            if len(row) > value_col_idx and row[0]:
                try:
                    out[row[0]] = float(row[value_col_idx])
                except ValueError:
                    continue
    return out


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


def _read_n_col(path: Path, n: int) -> list[tuple[str, ...]]:
    if not path.exists():
        return []
    out: list[tuple[str, ...]] = []
    with path.open() as fh:
        reader = csv.reader(fh)
        next(reader, None)
        for row in reader:
            if len(row) >= n and all(row[i] for i in range(n)):
                out.append(tuple(row[:n]))
    return out


def _write_keyed_2(path: Path, header: tuple[str, str, str],
                   rows: list[tuple[str, str, float]]) -> None:
    path.write_text(",".join(header) + "\n"
                    + "".join(f"{a},{b},{repr(v)}\n" for a, b, v in rows))


def _pow10_round_clamped(v: float) -> float:
    """max(1e-6, min(1e9, 10 ** round(log10(v))))."""
    if v <= 0:
        return 1.0
    return max(1e-6, min(1e9, 10.0 ** round(math.log10(v))))


def write_lp_scaling_params(input_dir: Path, solve_data_dir: Path) -> None:
    nodes = _read_singles(input_dir / "node.csv")
    groups = _read_singles(input_dir / "group.csv")
    period_in_use = _read_singles(solve_data_dir / "period_in_use_set.csv")

    # solve_current → use_row_scaling. The mod check is
    # `sum{c in solve_current} p_use_row_scaling[c] < 0.5` — passes if no
    # solve has scaling enabled.
    p_use_row_scaling: dict[str, float] = {}
    purs_path = solve_data_dir / "p_use_row_scaling.csv"
    if purs_path.exists():
        with purs_path.open() as fh:
            reader = csv.reader(fh)
            next(reader, None)
            for row in reader:
                if len(row) >= 2 and row[0]:
                    try:
                        p_use_row_scaling[row[0]] = float(row[1])
                    except ValueError:
                        continue
    solve_current = _read_singles(solve_data_dir / "solve_current.csv")
    scaling_active = sum(p_use_row_scaling.get(c, 0.0) for c in solve_current) >= 0.5

    # process_source_sink (Python-side now)
    pss = _read_n_col(solve_data_dir / "process_source_sink.csv", 3)

    # p_entity_unitsize (Python output)
    p_entity_unitsize = _read_keyed_value(
        solve_data_dir / "p_entity_unitsize.csv"
    )

    # _node_cap_inflow_fallback (Python output from node_inflow_scaling_params)
    inflow_fallback: dict[tuple[str, str], float] = {}
    fb_path = solve_data_dir / "_node_cap_inflow_fallback.csv"
    if fb_path.exists():
        with fb_path.open() as fh:
            reader = csv.reader(fh)
            next(reader, None)
            for row in reader:
                if len(row) >= 3:
                    try:
                        inflow_fallback[(row[0], row[1])] = float(row[2])
                    except ValueError:
                        continue

    # group_node (input)
    group_node = _read_pairs(input_dir / "group__node.csv")
    nodes_for_group: dict[str, list[str]] = {}
    for g, n in group_node:
        nodes_for_group.setdefault(g, []).append(n)

    # ---- _node_cap_unitsize_sum -----------------------------------------
    # Sum of p_entity_unitsize[p] for each (p, source, n) and (p, n, sink) in process_source_sink.
    cap_unitsize: dict[str, float] = {n: 0.0 for n in nodes}
    for p, source, sink in pss:
        # arc end at n: (p, source, n) where sink == n: incoming arc to n
        # arc end at n: (p, n, sink) where source == n: outgoing arc from n
        if sink in cap_unitsize:
            cap_unitsize[sink] += p_entity_unitsize.get(p, 0.0)
        if source in cap_unitsize:
            cap_unitsize[source] += p_entity_unitsize.get(p, 0.0)

    rows_unitsize: list[tuple[str, str, float]] = [
        (n, d, cap_unitsize.get(n, 0.0)) for n in nodes for d in period_in_use
    ]
    _write_keyed_2(solve_data_dir / "_node_cap_unitsize_sum.csv",
                   ("node", "period", "value"), rows_unitsize)

    # ---- _node_cap_raw --------------------------------------------------
    raw_dict: dict[tuple[str, str], float] = {}
    rows_raw: list[tuple[str, str, float]] = []
    for n in nodes:
        usz = cap_unitsize.get(n, 0.0)
        for d in period_in_use:
            if usz > 0:
                v = usz
            else:
                fb = inflow_fallback.get((n, d), 0.0)
                v = fb if fb > 0 else 1.0
            raw_dict[(n, d)] = v
            rows_raw.append((n, d, v))
    _write_keyed_2(solve_data_dir / "_node_cap_raw.csv",
                   ("node", "period", "value"), rows_raw)

    # ---- _node_cap_pow10 ------------------------------------------------
    pow10_dict: dict[tuple[str, str], float] = {}
    rows_pow10: list[tuple[str, str, float]] = []
    for (n, d), v in raw_dict.items():
        p10 = _pow10_round_clamped(v)
        pow10_dict[(n, d)] = p10
        rows_pow10.append((n, d, p10))
    _write_keyed_2(solve_data_dir / "_node_cap_pow10.csv",
                   ("node", "period", "value"), rows_pow10)

    # ---- node_capacity_for_scaling --------------------------------------
    ncfs_dict: dict[tuple[str, str], float] = {}
    rows_ncfs: list[tuple[str, str, float]] = []
    for n in nodes:
        for d in period_in_use:
            v = pow10_dict.get((n, d), 1.0) if scaling_active else 1.0
            ncfs_dict[(n, d)] = v
            rows_ncfs.append((n, d, v))
    _write_keyed_2(solve_data_dir / "node_capacity_for_scaling.csv",
                   ("node", "period", "value"), rows_ncfs)

    # ---- inv_node_cap ---------------------------------------------------
    rows_inc: list[tuple[str, str, float]] = []
    for n in nodes:
        for d in period_in_use:
            ncfs = ncfs_dict.get((n, d), 1.0)
            rows_inc.append((n, d, 1.0 / ncfs if ncfs != 0 else 0.0))
    _write_keyed_2(solve_data_dir / "inv_node_cap.csv",
                   ("node", "period", "value"), rows_inc)

    # ---- _group_cap_raw -------------------------------------------------
    grp_raw: dict[tuple[str, str], float] = {}
    rows_graw: list[tuple[str, str, float]] = []
    for g in groups:
        for d in period_in_use:
            v = sum(ncfs_dict.get((n, d), 1.0)
                    for n in nodes_for_group.get(g, ()))
            grp_raw[(g, d)] = v
            rows_graw.append((g, d, v))
    _write_keyed_2(solve_data_dir / "_group_cap_raw.csv",
                   ("group", "period", "value"), rows_graw)

    # ---- _group_cap_pow10 -----------------------------------------------
    grp_pow10: dict[tuple[str, str], float] = {}
    rows_gpow10: list[tuple[str, str, float]] = []
    for (g, d), v in grp_raw.items():
        p10 = _pow10_round_clamped(v) if v > 0 else 1.0
        grp_pow10[(g, d)] = p10
        rows_gpow10.append((g, d, p10))
    _write_keyed_2(solve_data_dir / "_group_cap_pow10.csv",
                   ("group", "period", "value"), rows_gpow10)

    # ---- group_capacity_for_scaling -------------------------------------
    gcfs_dict: dict[tuple[str, str], float] = {}
    rows_gcfs: list[tuple[str, str, float]] = []
    for g in groups:
        for d in period_in_use:
            v = grp_pow10.get((g, d), 1.0) if scaling_active else 1.0
            gcfs_dict[(g, d)] = v
            rows_gcfs.append((g, d, v))
    _write_keyed_2(solve_data_dir / "group_capacity_for_scaling.csv",
                   ("group", "period", "value"), rows_gcfs)

    # ---- inv_group_cap --------------------------------------------------
    rows_igc: list[tuple[str, str, float]] = []
    for g in groups:
        for d in period_in_use:
            gcfs = gcfs_dict.get((g, d), 1.0)
            rows_igc.append((g, d, 1.0 / gcfs if gcfs != 0 else 0.0))
    _write_keyed_2(solve_data_dir / "inv_group_cap.csv",
                   ("group", "period", "value"), rows_igc)
