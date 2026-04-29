"""Reserve-side calc params and dependent sets.

Migrated from flextool.mod:

    L1319 pdtReserve_upDown_group   (4-branch fallback over pbt/pt/p
                                     for the reserveTimeParam axis)
    L1327 process_reserve_upDown_node_active
                                    (filter of process_reserve_upDown_node
                                     by non-zero summed reservation
                                     across reserve__upDown__group × dt)
    L1328 prundt                    (process_reserve_upDown_node_active × dt)
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


def _read_pairs_to_dict(path: Path, key_col: int) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    if not path.exists():
        return out
    other_col = 1 - key_col
    with path.open() as fh:
        reader = csv.reader(fh)
        next(reader, None)
        for row in reader:
            if len(row) >= 2 and row[0] and row[1]:
                out.setdefault(row[key_col], []).append(row[other_col])
    return out


# reserveTimeParam from flextool_base.dat L183 — only 'reservation'.
_RESERVE_TIME_PARAMS = ("reservation",)
# reserveParam_defaults (mod L493): reliability=1, penalty_reserve=5000, else 0.
# For reservation specifically, the default is 0.
_RESERVATION_DEFAULT = 0.0


def write_pdtReserve_upDown_group(input_dir: Path, solve_data_dir: Path) -> None:
    """flextool.mod L1319 — 4-branch hourly resolution for reserve groups.

    Branches mirror pdtProcess's branches 1, 2, 4, 5 (no period axis, no
    def1 — and the scalar branch returns the table's default of 0 for
    'reservation' when the (r, ud, g) row is missing).

    Stochastic gate is direct (``g in groupStochastic``); no
    group_process indirection like for pdtProcess.
    """
    pbt = _read_pbt_reserve(input_dir / "pbt_reserve__upDown__group.csv")
    pt = _read_pt_reserve(input_dir / "pt_reserve__upDown__group.csv")
    p = _read_p_reserve(input_dir / "p_reserve__upDown__group.csv")

    ts_for_d = _read_pairs_to_dict(solve_data_dir / "first_timesteps.csv", 0)
    tb_for_d = _read_pairs_to_dict(solve_data_dir / "solve_branch__time_branch.csv", 0)
    pe_for_d = _read_pairs_to_dict(solve_data_dir / "period__branch.csv", 1)
    groups_stoch = frozenset(_read_singles(input_dir / "groupIncludeStochastics.csv"))

    rug = _read_n_col(solve_data_dir / "reserve__upDown__group.csv", 3)
    dt = _read_pairs(solve_data_dir / "steps_in_use.csv")

    out_path = solve_data_dir / "pdtReserve_upDown_group.csv"
    with out_path.open("w") as fh:
        fh.write("reserve,upDown,group,param,period,time,value\n")
        for (r, ud, g) in rug:
            g_stoch = g in groups_stoch
            for param in _RESERVE_TIME_PARAMS:
                for (d, t) in dt:
                    val = _resolve(
                        r, ud, g, param, d, t,
                        pbt=pbt, pt=pt, p=p,
                        ts_for_d=ts_for_d,
                        tb_for_d=tb_for_d,
                        pe_for_d=pe_for_d,
                        g_stoch=g_stoch,
                    )
                    fh.write(f"{r},{ud},{g},{param},{d},{t},{repr(val)}\n")


def _resolve(
    r: str, ud: str, g: str, param: str, d: str, t: str,
    *, pbt, pt, p, ts_for_d, tb_for_d, pe_for_d, g_stoch: bool,
) -> float:
    # Branch 1: stochastic + outer-d's ts/tb
    if g_stoch:
        ts_list = ts_for_d.get(d, ())
        tb_list = tb_for_d.get(d, ())
        total = 0.0
        hit = False
        for tb in tb_list:
            for ts in ts_list:
                v = pbt.get((r, ud, g, param, tb, ts, t))
                if v is not None:
                    total += v
                    hit = True
        if hit:
            return total
    # Branch 2: parent period pe of d, tb from solve_branch[pe], ts from period__time_first[d]
    ts_list = ts_for_d.get(d, ())
    pe_list = pe_for_d.get(d, ())
    if pe_list and ts_list:
        total = 0.0
        hit = False
        for pe in pe_list:
            for tb in tb_for_d.get(pe, ()):
                for ts in ts_list:
                    v = pbt.get((r, ud, g, param, tb, ts, t))
                    if v is not None:
                        total += v
                        hit = True
        if hit:
            return total
    # Branch 3: time axis
    v = pt.get((r, ud, g, param, t))
    if v is not None:
        return v
    # Branch 4: scalar (with table default 0 for 'reservation')
    v = p.get((r, ud, g, param))
    if v is not None:
        return v
    return _RESERVATION_DEFAULT


def _read_pbt_reserve(path: Path) -> dict[tuple[str, str, str, str, str, str, str], float]:
    out: dict[tuple[str, str, str, str, str, str, str], float] = {}
    if not path.exists():
        return out
    with path.open() as fh:
        reader = csv.reader(fh)
        next(reader, None)
        for row in reader:
            if len(row) >= 8 and all(row[i] for i in range(7)):
                try:
                    out[(row[0], row[1], row[2], row[3], row[4], row[5], row[6])] = float(row[7])
                except ValueError:
                    continue
    return out


def _read_pt_reserve(path: Path) -> dict[tuple[str, str, str, str, str], float]:
    out: dict[tuple[str, str, str, str, str], float] = {}
    if not path.exists():
        return out
    with path.open() as fh:
        reader = csv.reader(fh)
        next(reader, None)
        for row in reader:
            if len(row) >= 6 and all(row[i] for i in range(5)):
                try:
                    out[(row[0], row[1], row[2], row[3], row[4])] = float(row[5])
                except ValueError:
                    continue
    return out


def _read_p_reserve(path: Path) -> dict[tuple[str, str, str, str], float]:
    out: dict[tuple[str, str, str, str], float] = {}
    if not path.exists():
        return out
    with path.open() as fh:
        reader = csv.reader(fh)
        next(reader, None)
        for row in reader:
            if len(row) >= 5 and all(row[i] for i in range(4)):
                try:
                    out[(row[0], row[1], row[2], row[3])] = float(row[4])
                except ValueError:
                    continue
    return out


def write_process_reserve_upDown_node_active_and_prundt(
    input_dir: Path, solve_data_dir: Path
) -> None:
    """flextool.mod L1321-1322 — derived sets keyed on pdtReserve_upDown_group.

        set process_reserve_upDown_node_active :=
            {(p, r, ud, n) in process_reserve_upDown_node :
                 sum{(r, ud, g) in reserve__upDown__group, (d,t) in dt}
                     pdtReserve_upDown_group[r, ud, g, 'reservation', d, t]};
        set prundt :=
            {(p, r, ud, n) in process_reserve_upDown_node_active, (d, t) in dt};

    GMPL semantics: in ``sum{(r, ud, g) in reserve__upDown__group, ...}``
    the outer ``(r, ud)`` are pre-bound and filter ``reserve__upDown__group``
    to rows whose first two columns equal them; ``g`` is fresh.
    """
    # Load pdtReserve as (r, ud, g, param, d, t) → value (just-written by batch 43)
    pdt_reserve: dict[tuple[str, str, str, str, str, str], float] = {}
    pdt_path = solve_data_dir / "pdtReserve_upDown_group.csv"
    if pdt_path.exists():
        with pdt_path.open() as fh:
            reader = csv.reader(fh)
            next(reader, None)
            for row in reader:
                if len(row) >= 7 and all(row[i] for i in range(6)):
                    try:
                        pdt_reserve[(row[0], row[1], row[2], row[3], row[4], row[5])] = float(row[6])
                    except ValueError:
                        continue
    # (r, ud) → list[g] from reserve__upDown__group
    rug_by_ru: dict[tuple[str, str], list[str]] = {}
    for r, ud, g in _read_n_col(solve_data_dir / "reserve__upDown__group.csv", 3):
        rug_by_ru.setdefault((r, ud), []).append(g)

    dt = _read_pairs(solve_data_dir / "steps_in_use.csv")
    prun = _read_n_col(input_dir / "process__reserve__upDown__node.csv", 4)

    active_rows: list[tuple[str, str, str, str]] = []
    for (p, r, ud, n) in prun:
        groups = rug_by_ru.get((r, ud), ())
        total = 0.0
        for g in groups:
            for (d, t) in dt:
                total += pdt_reserve.get((r, ud, g, "reservation", d, t), 0.0)
        if total != 0.0:
            active_rows.append((p, r, ud, n))

    out_active = solve_data_dir / "process_reserve_upDown_node_active.csv"
    with out_active.open("w") as fh:
        fh.write("process,reserve,upDown,node\n")
        for row in active_rows:
            fh.write(",".join(row) + "\n")

    out_prundt = solve_data_dir / "prundt.csv"
    with out_prundt.open("w") as fh:
        fh.write("process,reserve,upDown,node,period,time\n")
        for (p, r, ud, n) in active_rows:
            for (d, t) in dt:
                fh.write(f"{p},{r},{ud},{n},{d},{t}\n")
