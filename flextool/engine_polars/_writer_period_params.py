"""Writer-port Phase 1 follow-up 3 — heavy per-(d, t) emission writers.

Native ports of four ``write_pdt*`` helpers in
:mod:`flextool.flextoolrunner.preprocessing.entity_period_calc_params`
that emit per-(entity, period, time) CSVs via mostly-procedural
fallback cascades:

* :func:`write_pdtNodeInflow`                  (legacy L568 — 3-branch).
* :func:`write_pdtProfile`                     (legacy L767 — 5-branch).
* :func:`write_pdtConversion_rate_section_slope` (legacy L1209 — 3 outputs).
* :func:`write_pdtProcess_source_sink`         (legacy L950 — 11-branch).

Each writer mirrors the legacy CSV reads byte-for-byte (header,
delimiter, ``repr(value)`` precision) so the parity tests in
``tests/engine_polars/test_writer_port_phase1.py`` can ``filecmp`` the
outputs against the legacy emitter.

The fallback cascades use simple dict-keyed lookups — the legacy code is
already optimal for the per-row access pattern and we keep that shape
here for code clarity.  See the module docstring on
:mod:`._pdt_lookup` for the broader rationale.

Branches 1 (stochastic fold-in) and 2 (parent-period fold-in) of
``write_pdtNodeInflow`` are mod's stochastic / parent-branch fold-ins
(Gap E in the migration tracker).  No fixture in the repo's test data
carries non-empty ``pbt_node_inflow``, so these branches are inert in
parity tests — but we keep the structure for forward-compatibility.
"""
from __future__ import annotations

import csv
from pathlib import Path


# ---------------------------------------------------------------------------
# Shared CSV readers (mirror legacy helpers byte-for-byte).
# ---------------------------------------------------------------------------


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


def _read_pairs_to_dict(path: Path, key_col: int) -> dict[str, list[str]]:
    """Generic two-col CSV → ``key_col → list[other_col]``."""
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


def _read_stochastic_entities(group_entity_csv: Path,
                              group_stochastic_csv: Path) -> set[str]:
    """``stoch_entity = { e : exists g ∈ groupIncludeStochastics with (g, e) ∈ group__<entity> }``."""
    stoch_groups = frozenset(_read_singles(group_stochastic_csv))
    out: set[str] = set()
    if not group_entity_csv.exists():
        return out
    with group_entity_csv.open() as fh:
        reader = csv.reader(fh)
        next(reader, None)
        for row in reader:
            if len(row) >= 2 and row[0] in stoch_groups and row[1]:
                out.add(row[1])
    return out


# ---------------------------------------------------------------------------
# write_pdtNodeInflow — flextool.mod L1325 (3-branch).
# ---------------------------------------------------------------------------


def write_pdtNodeInflow(input_dir: Path, solve_data_dir: Path) -> None:
    """Emit ``solve_data/pdtNodeInflow.csv``.

    Branches:
      1. Stochastic fold-in (``pbt_node_inflow`` over stochastic node).
      2. Parent-period fold-in (``pbt_node_inflow`` over parent periods).
      3. Deterministic additive sum of the 4 scaling methods:
         * ``scale_to_annual_flow``            — pfa[n,d] * pti[n,t]
         * ``scale_in_proportion``             — pfp[n,d] * pti[n,t]
         * ``scale_to_annual_and_peak_flow``   — slope[n,d] * pti[n,t] - section[n,d]
         * ``use_original``                    — pti[n,t]

    Domain: nodes whose method is anything BUT ``no_inflow``.  Non-
    balance-union nodes get 0 (mod L1280 guard).
    """
    nodes = _read_singles(input_dir / "node.csv")
    dt = _read_pairs(solve_data_dir / "steps_in_use.csv")

    inflow_method_pairs = frozenset(
        _read_pairs(solve_data_dir / "node__inflow_method.csv")
    )
    n_balance = frozenset(_read_singles(solve_data_dir / "nodeBalance.csv"))
    n_balance_period = frozenset(
        _read_singles(solve_data_dir / "nodeBalancePeriod.csv")
    )
    balance_union = n_balance | n_balance_period

    stoch_node = _read_stochastic_entities(
        input_dir / "group__node.csv",
        input_dir / "groupIncludeStochastics.csv",
    )

    ts_for_d = _read_pairs_to_dict(
        solve_data_dir / "first_timesteps.csv", key_col=0,
    )
    tb_for_d = _read_pairs_to_dict(
        solve_data_dir / "solve_branch__time_branch.csv", key_col=0,
    )
    # period__branch.csv stores (db, d) — child key column is 1.
    pe_for_d = _read_pairs_to_dict(
        solve_data_dir / "period__branch.csv", key_col=1,
    )

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

    # pdNode lookup limited to (annual_flow, peak_inflow).
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

    pfa = _read_2_keyed_value(
        solve_data_dir / "period_flow_annual_multiplier.csv"
    )
    pfp = _read_2_keyed_value(
        solve_data_dir / "period_flow_proportional_multiplier.csv"
    )
    nos_slope = _read_2_keyed_value(solve_data_dir / "new_old_slope.csv")
    nos_section = _read_2_keyed_value(solve_data_dir / "new_old_section.csv")

    eligible_nodes = [
        n for n in nodes if (n, "no_inflow") not in inflow_method_pairs
    ]

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
                # Branch 1: stochastic fold-in.
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
                # Branch 2: parent-period fold-in.
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
                # Branch 3: deterministic additive sum.
                value = 0.0
                if in_balance:
                    pti = pt_inflow.get((n, t), 0.0)
                    if has_scale_annual and pdNode_af.get((n, d), 0.0):
                        value += pfa.get((n, d), 0.0) * pti
                    if has_scale_proportion and pdNode_af.get((n, d), 0.0):
                        value += pfp.get((n, d), 0.0) * pti
                    if (has_scale_peak
                            and pdNode_af.get((n, d), 0.0)
                            and pdNode_pk.get((n, d), 0.0)):
                        value += nos_slope.get((n, d), 0.0) * pti \
                                 - nos_section.get((n, d), 0.0)
                    if has_use_original:
                        value += pti
                fh.write(f"{n},{d},{t},{repr(value)}\n")


# ---------------------------------------------------------------------------
# write_pdtProfile — flextool.mod L1192 (5-branch fallback + stochastic UNION).
# ---------------------------------------------------------------------------


def write_pdtProfile(input_dir: Path, solve_data_dir: Path) -> None:
    """Emit ``solve_data/pdtProfile.csv``.

    Branches:
      1. Stochastic fold-in (any of process / node / process_node refs
         the profile under a stochastic group).
      2. Parent-period fold-in.
      3. ``pt_profile[p, t]``.
      4. ``p_profile[p]``.
      5. 0.

    Domain: every profile in ``input/profile.csv`` × ``dt``.
    """
    profiles = _read_singles(input_dir / "profile.csv")
    dt = _read_pairs(solve_data_dir / "steps_in_use.csv")

    # pbt / pt / p loaders.
    pbt_profile: dict[tuple[str, str, str, str], float] = {}
    pbt_path = input_dir / "pbt_profile.csv"
    if pbt_path.exists():
        with pbt_path.open() as fh:
            reader = csv.reader(fh)
            next(reader, None)
            for row in reader:
                if len(row) >= 5 and row[0] and row[1] and row[2] and row[3]:
                    try:
                        pbt_profile[(row[0], row[1], row[2], row[3])] = float(row[4])
                    except ValueError:
                        continue
    pt_profile: dict[tuple[str, str], float] = {}
    pt_path = solve_data_dir / "pt_profile.csv"
    if pt_path.exists():
        with pt_path.open() as fh:
            reader = csv.reader(fh)
            next(reader, None)
            for row in reader:
                if len(row) >= 3 and row[0] and row[1]:
                    try:
                        pt_profile[(row[0], row[1])] = float(row[2])
                    except ValueError:
                        continue
    p_profile: dict[str, float] = {}
    p_path = input_dir / "p_profile.csv"
    if p_path.exists():
        with p_path.open() as fh:
            reader = csv.reader(fh)
            next(reader, None)
            for row in reader:
                if len(row) >= 2 and row[0]:
                    try:
                        p_profile[row[0]] = float(row[1])
                    except ValueError:
                        continue

    # Branch indices.
    ts_for_d = _read_pairs_to_dict(
        solve_data_dir / "first_timesteps.csv", key_col=0,
    )
    tb_for_d = _read_pairs_to_dict(
        solve_data_dir / "solve_branch__time_branch.csv", key_col=0,
    )
    pe_for_d = _read_pairs_to_dict(
        solve_data_dir / "period__branch.csv", key_col=1,
    )

    # Stochastic profile UNION: any profile referenced via a stochastic
    # process / node / process_node binding.
    stoch_processes = _read_stochastic_entities(
        input_dir / "group__process.csv",
        input_dir / "groupIncludeStochastics.csv",
    )
    stoch_nodes = _read_stochastic_entities(
        input_dir / "group__node.csv",
        input_dir / "groupIncludeStochastics.csv",
    )
    stoch_profile: set[str] = set()
    pp_path = input_dir / "process__profile__profile_method.csv"
    if pp_path.exists():
        with pp_path.open() as fh:
            reader = csv.reader(fh)
            next(reader, None)
            for row in reader:
                if len(row) >= 2 and row[0] in stoch_processes and row[1]:
                    stoch_profile.add(row[1])
    np_path = input_dir / "node__profile__profile_method.csv"
    if np_path.exists():
        with np_path.open() as fh:
            reader = csv.reader(fh)
            next(reader, None)
            for row in reader:
                if len(row) >= 2 and row[0] in stoch_nodes and row[1]:
                    stoch_profile.add(row[1])
    pnp_path = input_dir / "process__node__profile__profile_method.csv"
    if pnp_path.exists():
        with pnp_path.open() as fh:
            reader = csv.reader(fh)
            next(reader, None)
            for row in reader:
                if len(row) >= 3 and row[0] in stoch_processes and row[2]:
                    stoch_profile.add(row[2])

    out_path = solve_data_dir / "pdtProfile.csv"
    with out_path.open("w") as fh:
        fh.write("profile,period,time,value\n")
        for p in profiles:
            is_stoch = p in stoch_profile
            for (d, t) in dt:
                # Branch 1: stochastic fold-in.
                if is_stoch:
                    total = 0.0
                    hit = False
                    for tb in tb_for_d.get(d, ()):
                        for ts in ts_for_d.get(d, ()):
                            v = pbt_profile.get((p, tb, ts, t))
                            if v is not None:
                                total += v
                                hit = True
                    if hit:
                        fh.write(f"{p},{d},{t},{repr(total)}\n")
                        continue
                # Branch 2: parent-period fold-in.
                pe_list = pe_for_d.get(d, ())
                ts_list = ts_for_d.get(d, ())
                if pe_list and ts_list:
                    total = 0.0
                    hit = False
                    for pe in pe_list:
                        for tb in tb_for_d.get(pe, ()):
                            for ts in ts_list:
                                v = pbt_profile.get((p, tb, ts, t))
                                if v is not None:
                                    total += v
                                    hit = True
                    if hit:
                        fh.write(f"{p},{d},{t},{repr(total)}\n")
                        continue
                # Branch 3: time axis.
                v = pt_profile.get((p, t))
                if v is not None:
                    fh.write(f"{p},{d},{t},{repr(v)}\n")
                    continue
                # Branch 4: scalar.
                v = p_profile.get(p)
                if v is not None:
                    fh.write(f"{p},{d},{t},{repr(v)}\n")
                    continue
                # Branch 5: 0.
                fh.write(f"{p},{d},{t},0.0\n")


# ---------------------------------------------------------------------------
# write_pdtConversion_rate_section_slope — flextool.mod L1390-1400 (3 outputs).
# ---------------------------------------------------------------------------


def write_pdtConversion_rate_section_slope(
    input_dir: Path, solve_data_dir: Path,
) -> None:
    """Emit ``pdtConversion_rate.csv``, ``pdtProcess_section.csv``,
    ``pdtProcess_slope.csv``.

    Each is per-(process, period, time) and derived from ``pdtProcess``
    (efficiency / min_load / efficiency_at_min_load).  See the legacy
    docstring at
    :func:`flextool.flextoolrunner.preprocessing.entity_period_calc_params.write_pdtConversion_rate_section_slope`
    for the exact formulas (mirrored here as-is).
    """
    processes = _read_singles(input_dir / "process.csv")
    process_minload = frozenset(
        _read_singles(solve_data_dir / "process_minload.csv")
    )
    dt = _read_pairs(solve_data_dir / "steps_in_use.csv")

    eff: dict[tuple[str, str, str], float] = {}
    min_load: dict[tuple[str, str, str], float] = {}
    eff_min: dict[tuple[str, str, str], float] = {}
    pdt_path = solve_data_dir / "pdtProcess.csv"
    if pdt_path.exists():
        with pdt_path.open() as fh:
            reader = csv.reader(fh)
            next(reader, None)
            for row in reader:
                if len(row) < 5 or not row[0]:
                    continue
                try:
                    v = float(row[4])
                except ValueError:
                    continue
                key = (row[0], row[2], row[3])  # (process, period, time)
                if row[1] == "efficiency":
                    eff[key] = v
                elif row[1] == "min_load":
                    min_load[key] = v
                elif row[1] == "efficiency_at_min_load":
                    eff_min[key] = v

    # pdtConversion_rate.
    conv_path = solve_data_dir / "pdtConversion_rate.csv"
    conv_rate: dict[tuple[str, str, str], float] = {}
    with conv_path.open("w") as fh:
        fh.write("process,period,time,value\n")
        for p in processes:
            for (d, t) in dt:
                e = eff.get((p, d, t), 0.0)
                v = round(1.0 / e, 6) if e else 0.0
                conv_rate[(p, d, t)] = v
                fh.write(f"{p},{d},{t},{repr(v)}\n")

    # pdtProcess_section (process_minload only).
    sec_path = solve_data_dir / "pdtProcess_section.csv"
    section: dict[tuple[str, str, str], float] = {}
    with sec_path.open("w") as fh:
        fh.write("process,period,time,value\n")
        for p in processes:
            if p not in process_minload:
                continue
            for (d, t) in dt:
                cr = conv_rate.get((p, d, t), 0.0)
                ml = min_load.get((p, d, t), 0.0)
                em = eff_min.get((p, d, t), 0.0)
                inv_em = (1.0 / em) if em else 0.0
                denom = 1.0 - ml
                rounded = round((cr - ml * inv_em) / denom, 6) if denom else 0.0
                v = cr - rounded
                section[(p, d, t)] = v
                fh.write(f"{p},{d},{t},{repr(v)}\n")

    # pdtProcess_slope.
    slope_path = solve_data_dir / "pdtProcess_slope.csv"
    with slope_path.open("w") as fh:
        fh.write("process,period,time,value\n")
        for p in processes:
            in_min = p in process_minload
            for (d, t) in dt:
                cr = conv_rate.get((p, d, t), 0.0)
                sec = section.get((p, d, t), 0.0) if in_min else 0.0
                v = cr - sec
                fh.write(f"{p},{d},{t},{repr(v)}\n")


# ---------------------------------------------------------------------------
# write_pdtProcess_source_sink — flextool.mod L1219 (11-branch fallback).
# ---------------------------------------------------------------------------


def _read_quads(path: Path) -> list[tuple[str, str, str, str]]:
    if not path.exists():
        return []
    out: list[tuple[str, str, str, str]] = []
    with path.open() as fh:
        reader = csv.reader(fh)
        next(reader, None)
        for row in reader:
            if len(row) >= 4 and all(row[i] for i in range(4)):
                out.append((row[0], row[1], row[2], row[3]))
    return out


def _load_pbt_per_side(path: Path) -> dict[tuple[str, str, str, str, str, str], float]:
    out: dict[tuple[str, str, str, str, str, str], float] = {}
    if not path.exists():
        return out
    with path.open() as fh:
        reader = csv.reader(fh)
        next(reader, None)
        for row in reader:
            if len(row) >= 7 and all(row[i] for i in range(6)):
                try:
                    out[(row[0], row[1], row[2], row[3], row[4], row[5])] = float(row[6])
                except ValueError:
                    continue
    return out


def _load_pt_per_side(path: Path) -> dict[tuple[str, str, str, str], float]:
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


def _load_p_per_side(path: Path) -> dict[tuple[str, str, str], float]:
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


def write_pdtProcess_source_sink(input_dir: Path, solve_data_dir: Path) -> None:
    """Emit ``solve_data/pdtProcess_source_sink.csv`` (11-branch fallback).

    Outer index: ``(p, source, sink, param) ∈ process__source__sink__param_t,
    (d, t) ∈ dt``.

    Branch order:
      1. sink stochastic (pbt_process_sink fold-in)
      2. source stochastic (pbt_process_source fold-in)
      3. sink parent-period fold-in
      4. source parent-period fold-in
      5. ``pt_process_sink[p, sink, param, t]``
      6. ``pt_process_source[p, source, param, t]``
      7. ``pt_process[p, param, t]`` (only when ``p ∈ process_connection``)
      8. ``p_process_source[p, source, param]``
      9. ``p_process_sink[p, sink, param]``
     10. ``p_process[p, param]`` (only when ``p ∈ process_connection``)
     11. 0.
    """
    domain = _read_quads(solve_data_dir / "process__source__sink__param_t.csv")
    dt = _read_pairs(solve_data_dir / "steps_in_use.csv")

    pbt_sink = _load_pbt_per_side(input_dir / "pbt_process_sink.csv")
    pbt_source = _load_pbt_per_side(input_dir / "pbt_process_source.csv")

    pt_sink = _load_pt_per_side(input_dir / "pt_process_sink.csv")
    pt_source = _load_pt_per_side(input_dir / "pt_process_source.csv")

    pt_process: dict[tuple[str, str, str], float] = {}
    ptp_path = input_dir / "pt_process.csv"
    if ptp_path.exists():
        with ptp_path.open() as fh:
            reader = csv.reader(fh)
            next(reader, None)
            for row in reader:
                if len(row) >= 4 and row[0] and row[1] and row[2]:
                    try:
                        pt_process[(row[0], row[1], row[2])] = float(row[3])
                    except ValueError:
                        continue

    p_source = _load_p_per_side(input_dir / "p_process_source.csv")
    p_sink = _load_p_per_side(input_dir / "p_process_sink.csv")

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

    process_connection = frozenset(
        _read_singles(input_dir / "process_connection.csv")
    )

    ts_for_d = _read_pairs_to_dict(
        solve_data_dir / "first_timesteps.csv", key_col=0,
    )
    tb_for_d = _read_pairs_to_dict(
        solve_data_dir / "solve_branch__time_branch.csv", key_col=0,
    )
    pe_for_d = _read_pairs_to_dict(
        solve_data_dir / "period__branch.csv", key_col=1,
    )

    stoch_processes = _read_stochastic_entities(
        input_dir / "group__process.csv",
        input_dir / "groupIncludeStochastics.csv",
    )

    out_path = solve_data_dir / "pdtProcess_source_sink.csv"
    with out_path.open("w") as fh:
        fh.write("process,source,sink,param,period,time,value\n")
        for (p, src, snk, param) in domain:
            is_stoch = p in stoch_processes
            is_conn = p in process_connection
            for (d, t) in dt:
                # Branch 1: sink stochastic.
                if is_stoch:
                    total = 0.0
                    hit = False
                    for tb in tb_for_d.get(d, ()):
                        for ts in ts_for_d.get(d, ()):
                            v = pbt_sink.get((p, snk, param, tb, ts, t))
                            if v is not None:
                                total += v
                                hit = True
                    if hit:
                        fh.write(f"{p},{src},{snk},{param},{d},{t},{repr(total)}\n")
                        continue
                    # Branch 2: source stochastic.
                    total = 0.0
                    hit = False
                    for tb in tb_for_d.get(d, ()):
                        for ts in ts_for_d.get(d, ()):
                            v = pbt_source.get((p, src, param, tb, ts, t))
                            if v is not None:
                                total += v
                                hit = True
                    if hit:
                        fh.write(f"{p},{src},{snk},{param},{d},{t},{repr(total)}\n")
                        continue
                # Branch 3: sink parent-period.
                pe_list = pe_for_d.get(d, ())
                ts_list = ts_for_d.get(d, ())
                if pe_list and ts_list:
                    total = 0.0
                    hit = False
                    for pe in pe_list:
                        for tb in tb_for_d.get(pe, ()):
                            for ts in ts_list:
                                v = pbt_sink.get((p, snk, param, tb, ts, t))
                                if v is not None:
                                    total += v
                                    hit = True
                    if hit:
                        fh.write(f"{p},{src},{snk},{param},{d},{t},{repr(total)}\n")
                        continue
                    # Branch 4: source parent-period.
                    total = 0.0
                    hit = False
                    for pe in pe_list:
                        for tb in tb_for_d.get(pe, ()):
                            for ts in ts_list:
                                v = pbt_source.get((p, src, param, tb, ts, t))
                                if v is not None:
                                    total += v
                                    hit = True
                    if hit:
                        fh.write(f"{p},{src},{snk},{param},{d},{t},{repr(total)}\n")
                        continue
                # Branch 5: pt_process_sink.
                v = pt_sink.get((p, snk, param, t))
                if v is not None:
                    fh.write(f"{p},{src},{snk},{param},{d},{t},{repr(v)}\n")
                    continue
                # Branch 6: pt_process_source.
                v = pt_source.get((p, src, param, t))
                if v is not None:
                    fh.write(f"{p},{src},{snk},{param},{d},{t},{repr(v)}\n")
                    continue
                # Branch 7: pt_process (connection only).
                if is_conn:
                    v = pt_process.get((p, param, t))
                    if v is not None:
                        fh.write(f"{p},{src},{snk},{param},{d},{t},{repr(v)}\n")
                        continue
                # Branch 8: p_process_source.
                v = p_source.get((p, src, param))
                if v is not None:
                    fh.write(f"{p},{src},{snk},{param},{d},{t},{repr(v)}\n")
                    continue
                # Branch 9: p_process_sink.
                v = p_sink.get((p, snk, param))
                if v is not None:
                    fh.write(f"{p},{src},{snk},{param},{d},{t},{repr(v)}\n")
                    continue
                # Branch 10: p_process (connection only).
                if is_conn:
                    v = p_process.get((p, param))
                    if v is not None:
                        fh.write(f"{p},{src},{snk},{param},{d},{t},{repr(v)}\n")
                        continue
                # Branch 11: 0.
                fh.write(f"{p},{src},{snk},{param},{d},{t},0.0\n")
