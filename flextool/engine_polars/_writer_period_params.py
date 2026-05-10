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


# ---------------------------------------------------------------------------
# Writer-port Phase 1 follow-up 4 — group/commodity period-param fallbacks
# and the inflow positive/negative split.
#
# Same procedural shape as the legacy emitters in
# :mod:`flextool.flextoolrunner.preprocessing.entity_period_calc_params`.
# The legacy code is already optimal for the per-row access pattern
# (dict lookups in a nested loop); we mirror it byte-for-byte so the
# parity tests can ``filecmp``.
# ---------------------------------------------------------------------------


# flextool_base.dat L196-201 — group period param taxonomies.
_GROUP_PERIOD_PARAM: frozenset[str] = frozenset((
    "capacity_margin", "co2_price", "co2_max_period", "co2_max_total",
    "inertia_limit", "invest_max_period", "invest_min_period",
    "max_cumulative_flow", "min_cumulative_flow", "non_synchronous_limit",
    "penalty_inertia", "penalty_non_synchronous",
    "max_instant_flow", "min_instant_flow", "penalty_capacity_margin",
    "retire_max_period", "retire_min_period",
    "cumulative_max_capacity", "cumulative_min_capacity",
))
_GROUP_TIME_PARAM: frozenset[str] = frozenset((
    "co2_price", "max_instant_flow", "min_instant_flow",
))
_GROUP_PARAM_DEFAULT_5000: frozenset[str] = frozenset((
    "penalty_inertia", "penalty_capacity_margin", "penalty_non_synchronous",
))


def _read_p_2(path: Path) -> dict[tuple[str, str], float]:
    """Read a 3-col CSV ``(key1, key2, value)`` into a dict.

    Mirrors legacy ``_read_p_2`` (entity_period_calc_params.py L1965).
    """
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
    """Read a 4-col CSV ``(k1, k2, k3, value)`` into a dict."""
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


def _read_branches_for_d(period_branch_csv: Path) -> dict[str, list[str]]:
    """``period__branch.csv`` is ``(branch_period, period)`` — index by
    the child period (column 1) and gather branch list."""
    out: dict[str, list[str]] = {}
    if not period_branch_csv.exists():
        return out
    with period_branch_csv.open() as fh:
        reader = csv.reader(fh)
        next(reader, None)
        for row in reader:
            if len(row) >= 2 and row[0] and row[1]:
                out.setdefault(row[1], []).append(row[0])
    return out


# ---------------------------------------------------------------------------
# write_pdGroup — flextool.mod L1115 (5-branch fallback).
# ---------------------------------------------------------------------------


def write_pdGroup(input_dir: Path, solve_data_dir: Path) -> None:
    """Emit ``solve_data/pdGroup.csv``.

    Branches per (g, param, d):
      1. ``pd_group[g, param, d]``                        — direct.
      2. ``sum_{db ∈ branches_for_d[d]} pd_group[g, param, db]`` — fold.
      3. ``p_group[g, param]``                            — scalar fallback.
      4. ``5000`` when ``param`` is a 5000-default penalty.
      5. ``0``.
    """
    pd_g = _read_pd_2(input_dir / "pd_group.csv")
    p_g = _read_p_2(input_dir / "p_group.csv")
    branches_for_d = _read_branches_for_d(solve_data_dir / "period__branch.csv")
    groups = _read_singles(input_dir / "group.csv")
    period_in_use = _read_singles(solve_data_dir / "period_in_use_set.csv")

    out_path = solve_data_dir / "pdGroup.csv"
    with out_path.open("w") as fh:
        fh.write("group,param,period,value\n")
        for g in groups:
            for param in _GROUP_PERIOD_PARAM:
                for d in period_in_use:
                    if (g, param, d) in pd_g:
                        v = pd_g[(g, param, d)]
                    else:
                        branched = [
                            pd_g[(g, param, db)]
                            for db in branches_for_d.get(d, ())
                            if (g, param, db) in pd_g
                        ]
                        if branched:
                            v = sum(branched)
                        elif (g, param) in p_g:
                            v = p_g[(g, param)]
                        elif param in _GROUP_PARAM_DEFAULT_5000:
                            v = 5000.0
                        else:
                            v = 0.0
                    fh.write(f"{g},{param},{d},{repr(v)}\n")


# ---------------------------------------------------------------------------
# write_pdtGroup — flextool.mod L1126 (4-branch fallback: pt → pd → p → 0).
# ---------------------------------------------------------------------------


def write_pdtGroup(input_dir: Path, solve_data_dir: Path) -> None:
    """Emit ``solve_data/pdtGroup.csv``.

    Branches: ``pt_group[g, param, t]`` → ``pd_group[g, param, d]`` →
    ``p_group[g, param]`` → 0.
    """
    pt_g = _read_pd_2(input_dir / "pt_group.csv")  # same (k1, k2, k3, v) shape
    pd_g = _read_pd_2(input_dir / "pd_group.csv")
    p_g = _read_p_2(input_dir / "p_group.csv")
    groups = _read_singles(input_dir / "group.csv")
    dt = _read_pairs(solve_data_dir / "steps_in_use.csv")

    out_path = solve_data_dir / "pdtGroup.csv"
    with out_path.open("w") as fh:
        fh.write("group,param,period,time,value\n")
        for g in groups:
            for param in _GROUP_TIME_PARAM:
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


# ---------------------------------------------------------------------------
# write_pdCommodity — flextool.mod L1101 (3-branch fallback).
# ---------------------------------------------------------------------------

# commodityPeriodParam = {price} (flextool_base.dat L134)
_COMMODITY_PERIOD_PARAM: tuple[str, ...] = ("price",)


def write_pdCommodity(input_dir: Path, solve_data_dir: Path) -> None:
    """Emit ``solve_data/pdCommodity.csv``.

    Branches per (c, 'price', d):
      1. ``pd_commodity[c, 'price', d]``                 — direct.
      2. ``sum_{db ∈ branches_for_d[d]} pd_commodity[c, 'price', db]`` — fold.
      3. ``p_commodity[c, 'price']``                     — scalar (default 0).
    """
    pd_c = _read_pd_2(input_dir / "pd_commodity.csv")
    p_c = _read_p_2(input_dir / "p_commodity.csv")
    branches_for_d = _read_branches_for_d(solve_data_dir / "period__branch.csv")
    commodities = _read_singles(input_dir / "commodity.csv")
    period_in_use = _read_singles(solve_data_dir / "period_in_use_set.csv")

    out_path = solve_data_dir / "pdCommodity.csv"
    with out_path.open("w") as fh:
        fh.write("commodity,param,period,value\n")
        for c in commodities:
            for param in _COMMODITY_PERIOD_PARAM:
                for d in period_in_use:
                    if (c, param, d) in pd_c:
                        v = pd_c[(c, param, d)]
                    else:
                        branched = [
                            pd_c[(c, param, db)]
                            for db in branches_for_d.get(d, ())
                            if (c, param, db) in pd_c
                        ]
                        if branched:
                            v = sum(branched)
                        else:
                            v = p_c.get((c, param), 0.0)
                    fh.write(f"{c},{param},{d},{repr(v)}\n")


# ---------------------------------------------------------------------------
# write_pdtCommodity — flextool.mod L1108 (3-branch: pt → pd → p → 0).
# ---------------------------------------------------------------------------

# commodityTimeParam = {price} (flextool_base.dat L134)
_COMMODITY_TIME_PARAM: tuple[str, ...] = ("price",)


def write_pdtCommodity(input_dir: Path, solve_data_dir: Path) -> None:
    """Emit ``solve_data/pdtCommodity.csv``.

    Domain: commodity × commodityTimeParam × dt.
    Branches: ``pt_commodity`` → ``pd_commodity`` → ``p_commodity`` → 0.
    """
    pt = _read_pd_2(input_dir / "pt_commodity.csv")
    pd_ = _read_pd_2(input_dir / "pd_commodity.csv")
    p = _read_p_2(input_dir / "p_commodity.csv")
    commodities = _read_singles(input_dir / "commodity.csv")
    dt = _read_pairs(solve_data_dir / "steps_in_use.csv")

    out_path = solve_data_dir / "pdtCommodity.csv"
    with out_path.open("w") as fh:
        fh.write("commodity,param,period,time,value\n")
        for c in commodities:
            for param in _COMMODITY_TIME_PARAM:
                for (d, t) in dt:
                    v = pt.get((c, param, t))
                    if v is None:
                        v = pd_.get((c, param, d))
                    if v is None:
                        v = p.get((c, param), 0.0)
                    fh.write(f"{c},{param},{d},{t},{repr(v)}\n")


# ---------------------------------------------------------------------------
# write_p_positive_negative_inflow — flextool.mod L1672 / L1675.
# ---------------------------------------------------------------------------


def write_p_positive_negative_inflow(
    input_dir: Path, solve_data_dir: Path,
) -> None:
    """Emit ``p_positive_inflow.csv`` and ``p_negative_inflow.csv``
    from ``solve_data/pdtNodeInflow.csv`` (native, written upstream).

    * ``p_positive_inflow`` is restricted to non-``no_inflow`` nodes;
      ``value = max(pdtNodeInflow, 0)``.
    * ``p_negative_inflow`` covers all nodes — ``no_inflow`` nodes emit
      explicit ``0.0`` (matches the mod's all-nodes domain).
    """
    nodes = _read_singles(input_dir / "node.csv")
    dt = _read_pairs(solve_data_dir / "steps_in_use.csv")
    inflow_method_pairs = frozenset(
        _read_pairs(solve_data_dir / "node__inflow_method.csv")
    )
    no_inflow_nodes = frozenset(
        n for n in nodes if (n, "no_inflow") in inflow_method_pairs
    )

    pdt_inflow: dict[tuple[str, str, str], float] = {}
    pdtni_path = solve_data_dir / "pdtNodeInflow.csv"
    if pdtni_path.exists():
        with pdtni_path.open() as fh:
            reader = csv.reader(fh)
            next(reader, None)
            for row in reader:
                if len(row) >= 4 and row[0] and row[1] and row[2]:
                    try:
                        pdt_inflow[(row[0], row[1], row[2])] = float(row[3])
                    except ValueError:
                        continue

    pos_path = solve_data_dir / "p_positive_inflow.csv"
    with pos_path.open("w") as fh:
        fh.write("node,period,time,value\n")
        for n in nodes:
            if n in no_inflow_nodes:
                continue
            for (d, t) in dt:
                v = pdt_inflow.get((n, d, t), 0.0)
                fh.write(f"{n},{d},{t},{repr(v if v >= 0 else 0.0)}\n")

    neg_path = solve_data_dir / "p_negative_inflow.csv"
    with neg_path.open("w") as fh:
        fh.write("node,period,time,value\n")
        for n in nodes:
            for (d, t) in dt:
                if n in no_inflow_nodes:
                    fh.write(f"{n},{d},{t},0.0\n")
                else:
                    v = pdt_inflow.get((n, d, t), 0.0)
                    fh.write(f"{n},{d},{t},{repr(v if v < 0 else 0.0)}\n")


# ---------------------------------------------------------------------------
# Phase 1 follow-up 5 — entity_period_calc_params: varCost + cap_reduction +
# ed_period_params + pssdt_varCost filters.
# ---------------------------------------------------------------------------


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


def _read_pdt_at_param(path: Path, param_col: int, param_value: str,
                       key_cols: tuple[int, ...],
                       val_col: int) -> dict[tuple, float]:
    """Read a long-format pdtX CSV, filter rows where col[param_col] ==
    param_value, return dict[tuple(row[c] for c in key_cols)] = float(row[val_col]).
    """
    out: dict[tuple, float] = {}
    if not path.exists():
        return out
    with path.open() as fh:
        reader = csv.reader(fh)
        next(reader, None)
        for row in reader:
            if (len(row) > max(param_col, val_col, *key_cols)
                    and row[param_col] == param_value):
                try:
                    out[tuple(row[c] for c in key_cols)] = float(row[val_col])
                except ValueError:
                    continue
    return out


def _write_5col(path: Path, header: tuple[str, ...], rows: list[tuple]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(",".join(header) + "\n"
                    + "".join(",".join(r) + "\n" for r in rows))


# ---- write_pdtProcess__source__sink__dt_varCost_pair (mod L1493, L1502) ----

def write_pdtProcess__source__sink__dt_varCost_pair(
    input_dir: Path, solve_data_dir: Path,
) -> None:
    """flextool.mod L1493, L1502 — two ``varCost`` calc params keyed on
    ``process_source_sink`` and ``process_source_sink_alwaysProcess``.

    Both sum per-side ``other_operational_cost`` (gated by process_source
    / process_sink membership) plus ``pdtProcess['OOC']``.  The
    ``_alwaysProcess`` variant additionally gates the third term on
    ``(p, sink) ∈ process_sink ∪ process_source``.
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
    pss_always = _read_triples(
        solve_data_dir / "process_source_sink_alwaysProcess.csv"
    )
    dt = _read_pairs(solve_data_dir / "steps_in_use.csv")

    out_basic = solve_data_dir / "pdtProcess__source__sink__dt_varCost.csv"
    out_basic.parent.mkdir(parents=True, exist_ok=True)
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

    out_always = (
        solve_data_dir / "pdtProcess__source__sink__dt_varCost_alwaysProcess.csv"
    )
    with out_always.open("w") as fh:
        fh.write("process,source,sink,period,time,value\n")
        for (p, src, snk) in pss_always:
            for (d, t) in dt:
                v = 0.0
                if (p, src) in proc_src:
                    v += pdt_src.get((p, src, d, t), 0.0)
                if (p, snk) in proc_snk:
                    v += pdt_snk.get((p, snk, d, t), 0.0)
                if (p, snk) in proc_snk or (p, snk) in proc_src:
                    v += pdt.get((p, d, t), 0.0)
                fh.write(f"{p},{src},{snk},{d},{t},{repr(v)}\n")


# ---- write_pssdt_varCost_filters (mod L1498-1501) -------------------------

def write_pssdt_varCost_filters(
    input_dir: Path, solve_data_dir: Path,
) -> None:
    """flextool.mod L1498-1501 — four filter sets keyed on pdt-* OOC values.

    Depends on ``pdtProcess__source__sink__dt_varCost.csv`` produced by
    :func:`write_pdtProcess__source__sink__dt_varCost_pair`.
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


# ---- write_cap_reduction_params (mod L1637-1663) --------------------------

def write_cap_reduction_params(
    input_dir: Path, solve_data_dir: Path,
) -> None:
    """flextool.mod L1637-1663 — Morales-Espana startup/shutdown capacity
    reduction params (4 calc params, 1 per side × startup/shutdown).
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

    process_online = frozenset(
        _read_singles(solve_data_dir / "process_online.csv")
    )
    proc_src = _read_pairs(input_dir / "process__source.csv")
    proc_snk = _read_pairs(input_dir / "process__sink.csv")

    # dt with step_duration column from steps_in_use.csv.
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
        path.parent.mkdir(parents=True, exist_ok=True)
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


# ---- write_ed_period_params (mod L1252-1255 family, ed_*_period) ----------

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


def _write_keyed_2(path: Path, header: tuple[str, str, str],
                   rows: list[tuple[str, str, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(",".join(header) + "\n"
                    + "".join(f"{a},{b},{repr(v)}\n" for a, b, v in rows))


def write_ed_period_params(
    input_dir: Path, solve_data_dir: Path,
) -> None:
    """``ed_*_period`` / ``ed_cumulative_*`` family — six calc params
    keyed on ``ed_invest`` / ``ed_divest``.

    Must run AFTER ``invest_divest_sets`` has produced
    ``solve_data/ed_invest.csv`` and ``ed_divest.csv``.  Processes and
    nodes are disjoint by construction, so exactly one of the
    ``PdLookup`` branches fires per entity row.
    """
    from flextool.engine_polars._pdt_lookup import PdLookup
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
        ("ed_invest_max_period.csv",       ed_invest_pairs, "invest_max_period"),
        ("ed_invest_min_period.csv",       ed_invest_pairs, "invest_min_period"),
        ("ed_divest_max_period.csv",       ed_divest_pairs, "retire_max_period"),
        ("ed_divest_min_period.csv",       ed_divest_pairs, "retire_min_period"),
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
