"""Heavy per-(d, t) emission writers.

Three ``write_pdt*`` helpers that emit per-(entity, period, time) CSVs
via mostly-procedural fallback cascades:

* :func:`write_pdtNodeInflow`                  (3-branch).
* :func:`write_pdtProfile`                     (5-branch).
* :func:`write_pdtConversion_rate_section_slope` (2 outputs — section + slope; conversion_rate emit pruned, dead).

The fallback cascades use simple dict-keyed lookups — optimal for the
per-row access pattern.  See the module docstring on
:mod:`._pdt_lookup` for the broader rationale.

Branches 1 (stochastic fold-in) and 2 (parent-period fold-in) of
``write_pdtNodeInflow`` are mod's stochastic / parent-branch fold-ins
(Gap E in the migration tracker).  No fixture in the repo's test data
carries non-empty ``pbt_node_inflow``, so these branches are inert in
parity tests — but we keep the structure for forward-compatibility.
"""
from __future__ import annotations

from pathlib import Path

import polars as pl

from flextool.engine_polars._emit_provider_io import (
    _emit,
    _provider_key,
)
from flextool.engine_polars._vectorize import (
    build_entity_dt_grid,
    build_entity_period_grid,
    coalesce_value,
    collect_value_frame,
    lift_dict_to_lookup,
)


def _cell_str(value: "object | None") -> str:
    """Reproduce a ``csv.reader`` cell string for a native frame value.

    ``DataFrame.write_csv`` renders ``null`` as the empty string and every
    other scalar as its textual form; ``csv.reader`` then reads those
    strings back.  Mirror that here so dict keys / structural string
    columns stay byte-identical to the legacy CSV round-trip — ``None`` →
    ``""`` (skipped by the original truthiness guards) while a literal
    ``"0"`` is kept.  ``provider.get`` returns DATA rows only (no header
    row to skip); an empty / missing frame yields the same empty output
    the legacy loop produced.
    """
    return "" if value is None else str(value)


def _utf8_frame(columns: dict[str, list[str]]) -> pl.DataFrame:
    """Build an all-Utf8 ``pl.DataFrame`` from column-name → string-list.

    All columns (including ``value``) are stored as ``Utf8`` so the
    legacy ``f"...,{repr(v)}\\n"`` byte-emission round-trips identically
    through ``pl.DataFrame.write_csv``.  See
    :mod:`._emit_chain_params._ed_value_frame` for the rationale on
    using ``repr(v)`` directly rather than coercing to ``float`` first.
    """
    schema = {name: pl.Utf8 for name in columns}
    return pl.DataFrame(columns, schema=schema)


# ---------------------------------------------------------------------------
# Shared CSV readers (mirror legacy helpers byte-for-byte).
# ---------------------------------------------------------------------------


def _read_singles(path: Path,
                  *, provider: "object | None" = None) -> list[str]:
    df = provider.get(_provider_key(path))
    if df is None:
        return []
    out: list[str] = []
    for row in df.iter_rows():
        c0 = _cell_str(row[0]) if row else ""
        if c0:
            out.append(c0)
    return out


def _read_pairs(path: Path,
                *, provider: "object | None" = None) -> list[tuple[str, str]]:
    df = provider.get(_provider_key(path))
    if df is None:
        return []
    out: list[tuple[str, str]] = []
    for row in df.iter_rows():
        if len(row) < 2:
            continue
        c0, c1 = _cell_str(row[0]), _cell_str(row[1])
        if c0 and c1:
            out.append((c0, c1))
    return out


def _read_pairs_to_dict(path: Path, key_col: int,
                        *, provider: "object | None" = None,
                        ) -> dict[str, list[str]]:
    """Generic two-col CSV → ``key_col → list[other_col]``."""
    out: dict[str, list[str]] = {}
    df = provider.get(_provider_key(path))
    if df is None:
        return out
    other_col = 1 - key_col
    for row in df.iter_rows():
        if len(row) < 2:
            continue
        c0, c1 = _cell_str(row[0]), _cell_str(row[1])
        if c0 and c1:
            cells = (c0, c1)
            out.setdefault(cells[key_col], []).append(cells[other_col])
    return out


def _read_stochastic_entities(group_entity_csv: Path,
                              group_stochastic_csv: Path,
                              *, provider: "object | None" = None,
                              ) -> set[str]:
    """``stoch_entity = { e : exists g ∈ groupIncludeStochastics with (g, e) ∈ group__<entity> }``."""
    stoch_groups = frozenset(
        _read_singles(group_stochastic_csv, provider=provider)
    )
    out: set[str] = set()
    df = provider.get(_provider_key(group_entity_csv))
    if df is None:
        return out
    for row in df.iter_rows():
        if len(row) < 2:
            continue
        c0, c1 = _cell_str(row[0]), _cell_str(row[1])
        if c0 in stoch_groups and c1:
            out.add(c1)
    return out


# ---------------------------------------------------------------------------
# write_pdtNodeInflow — flextool.mod L1325 (3-branch).
# ---------------------------------------------------------------------------


def derive_pdtNodeInflow(input_dir: Path, solve_data_dir: Path,
                         *, provider: "object | None" = None,
                         ) -> pl.DataFrame:
    """Materialise the ``pdtNodeInflow`` frame.

    Columns: ``node, period, time, value`` — all ``Utf8`` (value cells
    are ``repr(v)`` so int/float distinction round-trips byte-identically
    to the legacy ``fh.write(f"{n},{d},{t},{repr(v)}\\n")`` emission).

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
    nodes = _read_singles(input_dir / "node.csv", provider=provider)
    dt = _read_pairs(solve_data_dir / "steps_in_use.csv", provider=provider)

    inflow_method_pairs = frozenset(
        _read_pairs(solve_data_dir / "node__inflow_method.csv",
                    provider=provider)
    )
    n_balance = frozenset(
        _read_singles(solve_data_dir / "nodeBalance.csv", provider=provider)
    )
    n_balance_period = frozenset(
        _read_singles(solve_data_dir / "nodeBalancePeriod.csv",
                      provider=provider)
    )
    balance_union = n_balance | n_balance_period

    stoch_node = _read_stochastic_entities(
        input_dir / "group__node.csv",
        input_dir / "groupIncludeStochastics.csv",
        provider=provider,
    )

    ts_for_d = _read_pairs_to_dict(
        solve_data_dir / "first_timesteps.csv", key_col=0,
        provider=provider,
    )
    tb_for_d = _read_pairs_to_dict(
        solve_data_dir / "solve_branch__time_branch.csv", key_col=0,
        provider=provider,
    )
    # period__branch.csv stores (db, d) — child key column is 1.
    pe_for_d = _read_pairs_to_dict(
        solve_data_dir / "period__branch.csv", key_col=1,
        provider=provider,
    )

    # pbt_node_inflow{(n, branch, ts, t) → value}
    pbt_inflow: dict[tuple[str, str, str, str], float] = {}
    pbt_path = input_dir / "pbt_node_inflow.csv"
    pbt_df = provider.get(_provider_key(pbt_path))
    if pbt_df is not None:
        for row in pbt_df.iter_rows():
            if len(row) < 5:
                continue
            c0, c1, c2, c3 = (_cell_str(row[0]), _cell_str(row[1]),
                              _cell_str(row[2]), _cell_str(row[3]))
            if c0 and c1 and c2 and c3:
                try:
                    pbt_inflow[(c0, c1, c2, c3)] = float(row[4])
                except (ValueError, TypeError):
                    continue

    # ptNode_inflow{(n, t) → value}
    pt_inflow: dict[tuple[str, str], float] = {}
    pti_path = solve_data_dir / "ptNode_inflow.csv"
    pti_df = provider.get(_provider_key(pti_path))
    if pti_df is not None:
        for row in pti_df.iter_rows():
            if len(row) < 3:
                continue
            c0, c1 = _cell_str(row[0]), _cell_str(row[1])
            if c0 and c1:
                try:
                    pt_inflow[(c0, c1)] = float(row[2])
                except (ValueError, TypeError):
                    continue

    # pdNode lookup limited to (annual_flow, peak_inflow).
    pdNode_af: dict[tuple[str, str], float] = {}
    pdNode_pk: dict[tuple[str, str], float] = {}
    pdn_path = solve_data_dir / "pdNode.csv"
    pdn_df = provider.get(_provider_key(pdn_path))
    if pdn_df is not None:
        for row in pdn_df.iter_rows():
            if len(row) < 4:
                continue
            c0 = _cell_str(row[0])
            if c0:
                try:
                    v = float(row[3])
                except (ValueError, TypeError):
                    continue
                c1, c2 = _cell_str(row[1]), _cell_str(row[2])
                if c1 == "annual_flow":
                    pdNode_af[(c0, c2)] = v
                elif c1 == "peak_inflow":
                    pdNode_pk[(c0, c2)] = v

    def _read_2_keyed_value(path: Path) -> dict[tuple[str, str], float]:
        out: dict[tuple[str, str], float] = {}
        df = provider.get(_provider_key(path))
        if df is None:
            return out
        for row in df.iter_rows():
            if len(row) < 3:
                continue
            c0, c1 = _cell_str(row[0]), _cell_str(row[1])
            if c0 and c1:
                try:
                    out[(c0, c1)] = float(row[2])
                except (ValueError, TypeError):
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

    nodes_col: list[str] = []
    periods_col: list[str] = []
    times_col: list[str] = []
    values_col: list[str] = []
    for n in eligible_nodes:
        is_stoch = n in stoch_node
        in_balance = n in balance_union
        has_scale_annual = (n, "scale_to_annual_flow") in inflow_method_pairs
        has_scale_proportion = (n, "scale_in_proportion") in inflow_method_pairs
        has_scale_peak = (n, "scale_to_annual_and_peak_flow") in inflow_method_pairs
        has_use_original = (n, "use_original") in inflow_method_pairs
        for (d, t) in dt:
            emit_v: float | None = None
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
                    emit_v = total
            # Branch 2: parent-period fold-in.
            if emit_v is None:
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
                        emit_v = total
            # Branch 3: deterministic additive sum.
            if emit_v is None:
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
                emit_v = value
            nodes_col.append(n)
            periods_col.append(d)
            times_col.append(t)
            values_col.append(repr(emit_v))
    return _utf8_frame({
        "node": nodes_col,
        "period": periods_col,
        "time": times_col,
        "value": values_col,
    })


def emit_pdtNodeInflow(input_dir: Path, solve_data_dir: Path,
                        *, provider) -> None:
    """Emit ``pdtNodeInflow`` to the Provider."""
    _emit(provider, "solve_data/pdtNodeInflow.csv",
          derive_pdtNodeInflow(input_dir, solve_data_dir, provider=provider))


# ---------------------------------------------------------------------------
# write_pdtProfile — flextool.mod L1192 (5-branch fallback + stochastic UNION).
# ---------------------------------------------------------------------------


def derive_pdtProfile(input_dir: Path, solve_data_dir: Path,
                      *, provider: "object | None" = None) -> pl.DataFrame:
    """Materialise the ``pdtProfile`` frame.

    Branches:
      1. Stochastic fold-in (any of process / node / process_node refs
         the profile under a stochastic group).
      2. Parent-period fold-in.
      3. ``pt_profile[p, t]``.
      4. ``p_profile[p]``.
      5. 0.

    Domain: every profile in ``input/profile.csv`` × ``dt``.
    """
    profiles = _read_singles(input_dir / "profile.csv", provider=provider)
    dt = _read_pairs(solve_data_dir / "steps_in_use.csv", provider=provider)

    # pbt / pt / p loaders.
    pbt_profile: dict[tuple[str, str, str, str], float] = {}
    pbt_path = input_dir / "pbt_profile.csv"
    pbt_df = provider.get(_provider_key(pbt_path))
    if pbt_df is not None:
        for row in pbt_df.iter_rows():
            if len(row) < 5:
                continue
            c0, c1, c2, c3 = (_cell_str(row[0]), _cell_str(row[1]),
                              _cell_str(row[2]), _cell_str(row[3]))
            if c0 and c1 and c2 and c3:
                try:
                    pbt_profile[(c0, c1, c2, c3)] = float(row[4])
                except (ValueError, TypeError):
                    continue
    pt_profile: dict[tuple[str, str], float] = {}
    pt_path = solve_data_dir / "pt_profile.csv"
    pt_df = provider.get(_provider_key(pt_path))
    if pt_df is not None:
        for row in pt_df.iter_rows():
            if len(row) < 3:
                continue
            c0, c1 = _cell_str(row[0]), _cell_str(row[1])
            if c0 and c1:
                try:
                    pt_profile[(c0, c1)] = float(row[2])
                except (ValueError, TypeError):
                    continue
    p_profile: dict[str, float] = {}
    p_path = input_dir / "p_profile.csv"
    p_df = provider.get(_provider_key(p_path))
    if p_df is not None:
        for row in p_df.iter_rows():
            if len(row) < 2:
                continue
            c0 = _cell_str(row[0])
            if c0:
                try:
                    p_profile[c0] = float(row[1])
                except (ValueError, TypeError):
                    continue

    # Branch indices.
    ts_for_d = _read_pairs_to_dict(
        solve_data_dir / "first_timesteps.csv", key_col=0,
        provider=provider,
    )
    tb_for_d = _read_pairs_to_dict(
        solve_data_dir / "solve_branch__time_branch.csv", key_col=0,
        provider=provider,
    )
    pe_for_d = _read_pairs_to_dict(
        solve_data_dir / "period__branch.csv", key_col=1,
        provider=provider,
    )

    # Stochastic profile UNION: any profile referenced via a stochastic
    # process / node / process_node binding.
    stoch_processes = _read_stochastic_entities(
        input_dir / "group__process.csv",
        input_dir / "groupIncludeStochastics.csv",
        provider=provider,
    )
    stoch_nodes = _read_stochastic_entities(
        input_dir / "group__node.csv",
        input_dir / "groupIncludeStochastics.csv",
        provider=provider,
    )
    stoch_profile: set[str] = set()
    pp_path = input_dir / "process__profile__profile_method.csv"
    pp_df = provider.get(_provider_key(pp_path))
    if pp_df is not None:
        for row in pp_df.iter_rows():
            if len(row) < 2:
                continue
            c0, c1 = _cell_str(row[0]), _cell_str(row[1])
            if c0 in stoch_processes and c1:
                stoch_profile.add(c1)
    np_path = input_dir / "node__profile__profile_method.csv"
    np_df = provider.get(_provider_key(np_path))
    if np_df is not None:
        for row in np_df.iter_rows():
            if len(row) < 2:
                continue
            c0, c1 = _cell_str(row[0]), _cell_str(row[1])
            if c0 in stoch_nodes and c1:
                stoch_profile.add(c1)
    pnp_path = input_dir / "process__node__profile__profile_method.csv"
    pnp_df = provider.get(_provider_key(pnp_path))
    if pnp_df is not None:
        for row in pnp_df.iter_rows():
            if len(row) < 3:
                continue
            c0, c2 = _cell_str(row[0]), _cell_str(row[2])
            if c0 in stoch_processes and c2:
                stoch_profile.add(c2)

    profiles_col: list[str] = []
    periods_col: list[str] = []
    times_col: list[str] = []
    values_col: list[str] = []
    for p in profiles:
        is_stoch = p in stoch_profile
        for (d, t) in dt:
            cell: str | None = None
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
                    cell = repr(total)
            # Branch 2: parent-period fold-in.
            if cell is None:
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
                        cell = repr(total)
            # Branch 3: time axis.
            if cell is None:
                v = pt_profile.get((p, t))
                if v is not None:
                    cell = repr(v)
            # Branch 4: scalar.
            if cell is None:
                v = p_profile.get(p)
                if v is not None:
                    cell = repr(v)
            # Branch 5: 0.
            if cell is None:
                cell = "0.0"
            profiles_col.append(p)
            periods_col.append(d)
            times_col.append(t)
            values_col.append(cell)
    return _utf8_frame({
        "profile": profiles_col,
        "period": periods_col,
        "time": times_col,
        "value": values_col,
    })


def emit_pdtProfile(input_dir: Path, solve_data_dir: Path,
                     *, provider) -> None:
    """Emit ``pdtProfile`` to the Provider."""
    _emit(provider, "solve_data/pdtProfile.csv",
          derive_pdtProfile(input_dir, solve_data_dir, provider=provider))


# ---------------------------------------------------------------------------
# write_pdtConversion_rate_section_slope — flextool.mod L1390-1400 (3 outputs).
# ---------------------------------------------------------------------------


def _derive_conversion_trio(
    input_dir: Path, solve_data_dir: Path,
    *, provider: "object | None" = None,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Build the three ``pdtConversion_rate`` / ``pdtProcess_section`` /
    ``pdtProcess_slope`` frames in a single pass.

    The slope formula reuses the ``conv_rate`` and ``section`` values
    computed for the first two frames; we share the intermediate dicts
    here so the work isn't duplicated across three independent
    ``derive_*`` calls.
    """
    processes = _read_singles(input_dir / "process.csv", provider=provider)
    process_minload = frozenset(
        _read_singles(solve_data_dir / "process_minload.csv",
                      provider=provider)
    )
    dt = _read_pairs(solve_data_dir / "steps_in_use.csv", provider=provider)

    eff: dict[tuple[str, str, str], float] = {}
    min_load: dict[tuple[str, str, str], float] = {}
    eff_min: dict[tuple[str, str, str], float] = {}
    pdt_path = solve_data_dir / "pdtProcess.csv"
    pdt_df = provider.get(_provider_key(pdt_path))
    if pdt_df is not None:
        for row in pdt_df.iter_rows():
            if len(row) < 5:
                continue
            c0 = _cell_str(row[0])
            if not c0:
                continue
            try:
                v = float(row[4])
            except (ValueError, TypeError):
                continue
            c1, c2, c3 = _cell_str(row[1]), _cell_str(row[2]), _cell_str(row[3])
            key = (c0, c2, c3)  # (process, period, time)
            if c1 == "efficiency":
                eff[key] = v
            elif c1 == "min_load":
                min_load[key] = v
            elif c1 == "efficiency_at_min_load":
                eff_min[key] = v

    # pdtConversion_rate columns + intermediate conv_rate dict.
    conv_rate: dict[tuple[str, str, str], float] = {}
    cr_p: list[str] = []
    cr_d: list[str] = []
    cr_t: list[str] = []
    cr_v: list[str] = []
    for p in processes:
        for (d, t) in dt:
            e = eff.get((p, d, t), 0.0)
            v = round(1.0 / e, 6) if e else 0.0
            conv_rate[(p, d, t)] = v
            cr_p.append(p)
            cr_d.append(d)
            cr_t.append(t)
            cr_v.append(repr(v))
    conv_frame = _utf8_frame({
        "process": cr_p, "period": cr_d, "time": cr_t, "value": cr_v,
    })

    # pdtProcess_section + intermediate section dict.
    section: dict[tuple[str, str, str], float] = {}
    sec_p: list[str] = []
    sec_d: list[str] = []
    sec_t: list[str] = []
    sec_v: list[str] = []
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
            sec_p.append(p)
            sec_d.append(d)
            sec_t.append(t)
            sec_v.append(repr(v))
    section_frame = _utf8_frame({
        "process": sec_p, "period": sec_d, "time": sec_t, "value": sec_v,
    })

    # pdtProcess_slope.
    sl_p: list[str] = []
    sl_d: list[str] = []
    sl_t: list[str] = []
    sl_v: list[str] = []
    for p in processes:
        in_min = p in process_minload
        for (d, t) in dt:
            cr = conv_rate.get((p, d, t), 0.0)
            sec = section.get((p, d, t), 0.0) if in_min else 0.0
            v = cr - sec
            sl_p.append(p)
            sl_d.append(d)
            sl_t.append(t)
            sl_v.append(repr(v))
    slope_frame = _utf8_frame({
        "process": sl_p, "period": sl_d, "time": sl_t, "value": sl_v,
    })

    return conv_frame, section_frame, slope_frame


def derive_pdtProcess_section(
    input_dir: Path, solve_data_dir: Path,
    *, provider: "object | None" = None,
) -> pl.DataFrame:
    """Materialise just the ``pdtProcess_section`` frame."""
    _conv, sec, _slope = _derive_conversion_trio(
        input_dir, solve_data_dir, provider=provider,
    )
    return sec


def derive_pdtProcess_slope(
    input_dir: Path, solve_data_dir: Path,
    *, provider: "object | None" = None,
) -> pl.DataFrame:
    """Materialise just the ``pdtProcess_slope`` frame."""
    _conv, _sec, slope = _derive_conversion_trio(
        input_dir, solve_data_dir, provider=provider,
    )
    return slope


def emit_pdtConversion_rate_section_slope(
    input_dir: Path, solve_data_dir: Path,
    *, provider,
) -> None:
    """Emit ``pdtConversion_rate_section_slope`` to the Provider."""
    _conv, sec, slope = _derive_conversion_trio(
        input_dir, solve_data_dir, provider=provider,
    )
    _emit(provider, "solve_data/pdtProcess_section.csv", sec)
    _emit(provider, "solve_data/pdtProcess_slope.csv", slope)


# ---------------------------------------------------------------------------
# Group / commodity period-param fallbacks and the inflow
# positive/negative split.
#
# Procedural shape with dict lookups in a nested loop — optimal for the
# per-row access pattern.  Output is byte-for-byte so the
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


def _read_p_2(path: Path,
              *, provider: "object | None" = None,
              ) -> dict[tuple[str, str], float]:
    """Read a 3-col CSV ``(key1, key2, value)`` into a dict.

    Mirrors legacy ``_read_p_2`` (entity_period_calc_params.py L1965).
    """
    out: dict[tuple[str, str], float] = {}
    df = provider.get(_provider_key(path))
    if df is None:
        return out
    for row in df.iter_rows():
        if len(row) < 3:
            continue
        c0, c1 = _cell_str(row[0]), _cell_str(row[1])
        if c0 and c1:
            try:
                out[(c0, c1)] = float(row[2])
            except (ValueError, TypeError):
                continue
    return out


def _read_pd_2(path: Path,
               *, provider: "object | None" = None,
               ) -> dict[tuple[str, str, str], float]:
    """Read a 4-col CSV ``(k1, k2, k3, value)`` into a dict."""
    out: dict[tuple[str, str, str], float] = {}
    df = provider.get(_provider_key(path))
    if df is None:
        return out
    for row in df.iter_rows():
        if len(row) < 4:
            continue
        c = [_cell_str(row[i]) for i in range(3)]
        if all(c):
            try:
                out[(c[0], c[1], c[2])] = float(row[3])
            except (ValueError, TypeError):
                continue
    return out


def _read_branches_for_d(period_branch_csv: Path,
                         *, provider: "object | None" = None,
                         ) -> dict[str, list[str]]:
    """``period__branch.csv`` is ``(branch_period, period)`` — index by
    the child period (column 1) and gather branch list."""
    out: dict[str, list[str]] = {}
    df = provider.get(_provider_key(period_branch_csv))
    if df is None:
        return out
    for row in df.iter_rows():
        if len(row) < 2:
            continue
        c0, c1 = _cell_str(row[0]), _cell_str(row[1])
        if c0 and c1:
            out.setdefault(c1, []).append(c0)
    return out


# ---------------------------------------------------------------------------
# write_pdGroup — flextool.mod L1115 (5-branch fallback).
# ---------------------------------------------------------------------------


def derive_pdGroup(input_dir: Path, solve_data_dir: Path,
                   *, provider: "object | None" = None) -> pl.DataFrame:
    """Materialise the ``pdGroup`` frame (5-branch fallback per (g, param, d)).

    Branches:
      1. ``pd_group[g, param, d]``                        — direct.
      2. ``sum_{db ∈ branches_for_d[d]} pd_group[g, param, db]`` — fold.
      3. ``p_group[g, param]``                            — scalar fallback.
      4. ``5000`` when ``param`` is a 5000-default penalty.
      5. ``0``.
    """
    pd_g = _read_pd_2(input_dir / "pd_group.csv", provider=provider)
    p_g = _read_p_2(input_dir / "p_group.csv", provider=provider)
    branches_for_d = _read_branches_for_d(
        solve_data_dir / "period__branch.csv", provider=provider,
    )
    groups = _read_singles(input_dir / "group.csv", provider=provider)
    period_in_use = _read_singles(
        solve_data_dir / "period_in_use_set.csv", provider=provider,
    )

    g_col: list[str] = []
    p_col: list[str] = []
    d_col: list[str] = []
    v_col: list[str] = []
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
                g_col.append(g)
                p_col.append(param)
                d_col.append(d)
                v_col.append(repr(v))
    return _utf8_frame({
        "group": g_col, "param": p_col, "period": d_col, "value": v_col,
    })


def derive_pdGroup_vectorized(input_dir: Path, solve_data_dir: Path,
                              *,
                              provider: "object | None" = None,
                              engine: str = "eager") -> pl.DataFrame:
    """Vectorized ``pdGroup`` derive — period-only, branch-sum, dedup-safe.

    Replaces the per-cell cascade loop in :func:`derive_pdGroup` with
    vectorized polars (left-joins + a group-by-sum for the branch fold +
    ``coalesce`` in cascade-priority order), still per roll over the
    roll's own window.  Output columns ``group, param, period, value``
    all ``Utf8`` (NO time axis), entity-major row order, ``repr(v)``
    value cells.

    Domain = ``group × _GROUP_PERIOD_PARAM`` (the live module frozenset,
    so iteration order matches legacy — S4) preserving order and
    duplicates; period axis = ``period_in_use`` (order + duplicates
    preserved).

    Cascade per ``(g, param, d)`` (5-branch):

    1. ``pd_group[g, param, d]``                                — direct.
    2. ``sum_{db ∈ branches_for_d[d], (g,param,db)∈pd_group}
        pd_group[g, param, db]``                               — fold,
       only when NON-empty.
    3. ``p_group[g, param]``                                   — scalar.
    4. ``5000`` when ``param ∈ _GROUP_PARAM_DEFAULT_5000``.
    5. ``0``.

    The branch-sum (D3 critique fix) is computed on the **de-duplicated**
    ``(group, param)`` set so a duplicated ``group`` entry in
    ``group.csv`` does NOT double-count the fold; the dup-preserving final
    grid then left-joins the per-``(group,param,period)`` sum back,
    re-expanding to every output row.
    """
    pd_g = _read_pd_2(input_dir / "pd_group.csv", provider=provider)
    p_g = _read_p_2(input_dir / "p_group.csv", provider=provider)
    branches_for_d = _read_branches_for_d(
        solve_data_dir / "period__branch.csv", provider=provider,
    )
    groups = _read_singles(input_dir / "group.csv", provider=provider)
    period_in_use = _read_singles(
        solve_data_dir / "period_in_use_set.csv", provider=provider,
    )

    key_cols = ["group", "param"]
    out_cols = [*key_cols, "period"]

    # Domain = group × _GROUP_PERIOD_PARAM, referencing the LIVE frozenset
    # object so iteration order matches the legacy loop (S4); preserve the
    # group list order + duplicates, never ``.unique()``.
    domain = [
        (g, param) for g in groups for param in list(_GROUP_PERIOD_PARAM)
    ]

    grid = build_entity_period_grid(
        domain, period_in_use, key_cols=key_cols,
    )

    # Branch 1 — direct pd_group[(g, param, d)].
    pd_df = lift_dict_to_lookup(pd_g, ["group", "param", "period"], "v_pd")

    # Branch 3 — scalar p_group[(g, param)].
    p_df = lift_dict_to_lookup(p_g, ["group", "param"], "v_p")

    # Branch 4 — 5000 default for the penalty params.
    def5000_params = list(_GROUP_PARAM_DEFAULT_5000)
    def5000_df = pl.DataFrame(
        {
            "param": def5000_params,
            "v_5000": [5000.0] * len(def5000_params),
        },
        schema={"param": pl.Utf8, "v_5000": pl.Float64},
    )

    # Branch 2 — branch-sum.  Expand (period d → branch period db),
    # preserving duplicate (d, db) rows (D2: duplicates must double-count
    # to match the legacy ``sum(...)`` over the branches list).
    exp_period: list[str] = []
    exp_db: list[str] = []
    for d in period_in_use:
        for db in branches_for_d.get(d, ()):
            exp_period.append(d)
            exp_db.append(db)
    exp = pl.DataFrame(
        {"period": exp_period, "db": exp_db},
        schema={"period": pl.Utf8, "db": pl.Utf8},
    )

    if exp.height > 0:
        pd_db = lift_dict_to_lookup(
            pd_g, ["group", "param", "db"], "v_pddb")
        # D3: de-dup the (group, param) set for the SUM ONLY so a
        # duplicated group does not double-count; the dup-preserving grid
        # re-expands the result back to every output row via the final
        # left-join.
        gp_unique = grid.select(["group", "param"]).unique()
        bsum = (
            gp_unique
            .join(exp, how="cross")
            # INNER join drops non-matching (g, param, db) → reproduces
            # both the ``if (g,param,db) in pd_g`` gate and the
            # ``if branched:`` non-empty gate.
            .join(pd_db, on=["group", "param", "db"], how="inner")
            .group_by(["group", "param", "period"])
            .agg(pl.col("v_pddb").sum().alias("v_branch"))
        )
    else:
        bsum = pl.DataFrame(
            {"group": [], "param": [], "period": [], "v_branch": []},
            schema={
                "group": pl.Utf8,
                "param": pl.Utf8,
                "period": pl.Utf8,
                "v_branch": pl.Float64,
            },
        )

    out = (
        grid
        .join(pd_df, on=["group", "param", "period"], how="left")
        .join(bsum, on=["group", "param", "period"], how="left")
        .join(p_df, on=["group", "param"], how="left")
        .join(def5000_df, on=["param"], how="left")
        .with_columns(
            coalesce_value([
                pl.col("v_pd"),      # branch 1 (direct)
                pl.col("v_branch"),  # branch 2 (branch-sum, non-empty)
                pl.col("v_p"),       # branch 3 (scalar)
                pl.col("v_5000"),    # branch 4 (5000 default set)
                pl.lit(0.0),         # branch 5 (default)
            ])
        )
    )
    return collect_value_frame(
        out, key_cols=out_cols, sort_cols=["__eo", "__po"], engine=engine,
    )


def emit_pdGroup(input_dir: Path, solve_data_dir: Path,
                  *, provider) -> None:
    """Emit ``pdGroup`` to the Provider."""
    _emit(provider, "solve_data/pdGroup.csv",
          derive_pdGroup_vectorized(
              input_dir, solve_data_dir, provider=provider))


# ---------------------------------------------------------------------------
# write_pdtGroup — flextool.mod L1126 (4-branch fallback: pt → pd → p → 0).
# ---------------------------------------------------------------------------


def derive_pdtGroup(input_dir: Path, solve_data_dir: Path,
                    *, provider: "object | None" = None) -> pl.DataFrame:
    """Materialise the ``pdtGroup`` frame.

    Branches: ``pt_group[g, param, t]`` → ``pd_group[g, param, d]`` →
    ``p_group[g, param]`` → 0.
    """
    pt_g = _read_pd_2(input_dir / "pt_group.csv", provider=provider)  # same (k1, k2, k3, v) shape
    pd_g = _read_pd_2(input_dir / "pd_group.csv", provider=provider)
    p_g = _read_p_2(input_dir / "p_group.csv", provider=provider)
    groups = _read_singles(input_dir / "group.csv", provider=provider)
    dt = _read_pairs(solve_data_dir / "steps_in_use.csv", provider=provider)

    g_col: list[str] = []
    p_col: list[str] = []
    d_col: list[str] = []
    t_col: list[str] = []
    v_col: list[str] = []
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
                g_col.append(g)
                p_col.append(param)
                d_col.append(d)
                t_col.append(t)
                v_col.append(repr(v))
    return _utf8_frame({
        "group": g_col, "param": p_col, "period": d_col, "time": t_col,
        "value": v_col,
    })


def derive_pdtGroup_vectorized(input_dir: Path, solve_data_dir: Path,
                               *,
                               provider: "object | None" = None,
                               engine: str = "eager") -> pl.DataFrame:
    """Vectorized ``pdtGroup`` derive — byte-parity with the legacy.

    Replaces the per-cell cascade loop in :func:`derive_pdtGroup` with
    vectorized polars (left-joins + ``coalesce`` in cascade-priority
    order), still per roll over the roll's own window.  Output is
    byte-identical to :func:`derive_pdtGroup`: columns ``group, param,
    period, time, value`` all ``Utf8``, entity-major row order,
    ``repr(v)`` value cells.

    Domain = ``group × _GROUP_TIME_PARAM`` (the live module frozenset, so
    iteration order matches legacy — S4) preserving order and duplicates,
    crossed with ``dt`` from ``steps_in_use``.

    Cascade (inline 4-branch, time-first):
    ``pt_group`` → ``pd_group`` → ``p_group`` → ``0.0``.
    """
    pt_g = _read_pd_2(input_dir / "pt_group.csv", provider=provider)
    pd_g = _read_pd_2(input_dir / "pd_group.csv", provider=provider)
    p_g = _read_p_2(input_dir / "p_group.csv", provider=provider)
    groups = _read_singles(input_dir / "group.csv", provider=provider)
    dt = _read_pairs(solve_data_dir / "steps_in_use.csv", provider=provider)

    key_cols = ["group", "param"]
    out_cols = [*key_cols, "period", "time"]

    # Domain = group × _GROUP_TIME_PARAM, referencing the LIVE frozenset
    # object so iteration order matches the legacy loop (S4); preserve the
    # group list order + duplicates, never ``.unique()``.
    domain = [(g, param) for g in groups for param in list(_GROUP_TIME_PARAM)]

    grid = build_entity_dt_grid(domain, dt, key_cols=key_cols)

    pt_df = lift_dict_to_lookup(pt_g, ["group", "param", "time"], "v_pt")
    pd_df = lift_dict_to_lookup(pd_g, ["group", "param", "period"], "v_pd")
    p_df = lift_dict_to_lookup(p_g, ["group", "param"], "v_p")

    out = (
        grid
        .join(pt_df, on=["group", "param", "time"], how="left")
        .join(pd_df, on=["group", "param", "period"], how="left")
        .join(p_df, on=["group", "param"], how="left")
        .with_columns(
            coalesce_value([
                pl.col("v_pt"),   # branch 1 (time-first)
                pl.col("v_pd"),   # branch 2 (period)
                pl.col("v_p"),    # branch 3
                pl.lit(0.0),      # branch 4 (default)
            ])
        )
    )
    return collect_value_frame(out, key_cols=out_cols, engine=engine)


def emit_pdtGroup(input_dir: Path, solve_data_dir: Path,
                   *, provider) -> None:
    """Emit ``pdtGroup`` to the Provider."""
    _emit(provider, "solve_data/pdtGroup.csv",
          derive_pdtGroup_vectorized(
              input_dir, solve_data_dir, provider=provider))


# ---------------------------------------------------------------------------
# write_pdtCommodity — flextool.mod L1108 (3-branch: pt → pd → p → 0).
# ---------------------------------------------------------------------------

# commodityTimeParam = {price} (flextool_base.dat L134)
_COMMODITY_TIME_PARAM: tuple[str, ...] = ("price",)


def derive_pdtCommodity(input_dir: Path, solve_data_dir: Path,
                        *, provider: "object | None" = None) -> pl.DataFrame:
    """Materialise the ``pdtCommodity`` frame.

    Domain: commodity × commodityTimeParam × dt.
    Branches: ``pt_commodity`` → ``pd_commodity`` → ``p_commodity`` → 0.
    """
    pt = _read_pd_2(input_dir / "pt_commodity.csv", provider=provider)
    pd_ = _read_pd_2(input_dir / "pd_commodity.csv", provider=provider)
    p = _read_p_2(input_dir / "p_commodity.csv", provider=provider)
    commodities = _read_singles(input_dir / "commodity.csv", provider=provider)
    dt = _read_pairs(solve_data_dir / "steps_in_use.csv", provider=provider)

    c_col: list[str] = []
    p_col: list[str] = []
    d_col: list[str] = []
    t_col: list[str] = []
    v_col: list[str] = []
    for c in commodities:
        for param in _COMMODITY_TIME_PARAM:
            for (d, t) in dt:
                v = pt.get((c, param, t))
                if v is None:
                    v = pd_.get((c, param, d))
                if v is None:
                    v = p.get((c, param), 0.0)
                c_col.append(c)
                p_col.append(param)
                d_col.append(d)
                t_col.append(t)
                v_col.append(repr(v))
    return _utf8_frame({
        "commodity": c_col, "param": p_col, "period": d_col, "time": t_col,
        "value": v_col,
    })


def derive_pdtCommodity_vectorized(input_dir: Path, solve_data_dir: Path,
                                   *,
                                   provider: "object | None" = None,
                                   engine: str = "eager") -> pl.DataFrame:
    """Vectorized ``pdtCommodity`` derive — byte-parity with the legacy.

    Replaces the per-cell cascade loop in :func:`derive_pdtCommodity`
    with vectorized polars (left-joins + ``coalesce`` in cascade-priority
    order), still per roll over the roll's own window.  Output is
    byte-identical to :func:`derive_pdtCommodity`: columns ``commodity,
    param, period, time, value`` all ``Utf8``, entity-major row order,
    ``repr(v)`` value cells.

    Cascade (inline 3-branch, time-first):
    ``pt_commodity`` → ``pd_commodity`` → ``p_commodity`` → ``0.0``.
    """
    pt = _read_pd_2(input_dir / "pt_commodity.csv", provider=provider)
    pd_ = _read_pd_2(input_dir / "pd_commodity.csv", provider=provider)
    p = _read_p_2(input_dir / "p_commodity.csv", provider=provider)
    commodities = _read_singles(input_dir / "commodity.csv", provider=provider)
    dt = _read_pairs(solve_data_dir / "steps_in_use.csv", provider=provider)

    key_cols = ["commodity", "param"]
    out_cols = [*key_cols, "period", "time"]

    # Domain = commodity × _COMMODITY_TIME_PARAM, preserving legacy
    # iteration order (commodity-major, param from the tuple) and never
    # ``.unique()``-d.
    domain = [(c, param) for c in commodities for param in _COMMODITY_TIME_PARAM]

    grid = build_entity_dt_grid(domain, dt, key_cols=key_cols)

    pt_df = lift_dict_to_lookup(
        pt, ["commodity", "param", "time"], "v_pt")
    pd_df = lift_dict_to_lookup(
        pd_, ["commodity", "param", "period"], "v_pd")
    p_df = lift_dict_to_lookup(
        p, ["commodity", "param"], "v_p")

    out = (
        grid
        .join(pt_df, on=["commodity", "param", "time"], how="left")
        .join(pd_df, on=["commodity", "param", "period"], how="left")
        .join(p_df, on=["commodity", "param"], how="left")
        .with_columns(
            coalesce_value([
                pl.col("v_pt"),   # branch 1 (time-first)
                pl.col("v_pd"),   # branch 2 (period)
                pl.col("v_p"),    # branch 3
                pl.lit(0.0),      # branch 4 (default)
            ])
        )
    )
    return collect_value_frame(out, key_cols=out_cols, engine=engine)


def emit_pdtCommodity(input_dir: Path, solve_data_dir: Path,
                       *, provider) -> None:
    """Emit ``pdtCommodity`` to the Provider."""
    _emit(provider, "solve_data/pdtCommodity.csv",
          derive_pdtCommodity_vectorized(
              input_dir, solve_data_dir, provider=provider))


# ---------------------------------------------------------------------------
# write_p_positive_negative_inflow — flextool.mod L1672 / L1675.
# ---------------------------------------------------------------------------


def _derive_positive_negative_inflow(
    input_dir: Path, solve_data_dir: Path,
    *, provider: "object | None" = None,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Build both ``p_positive_inflow`` and ``p_negative_inflow`` frames
    from a single read of ``pdtNodeInflow.csv``."""
    nodes = _read_singles(input_dir / "node.csv", provider=provider)
    dt = _read_pairs(solve_data_dir / "steps_in_use.csv", provider=provider)
    inflow_method_pairs = frozenset(
        _read_pairs(solve_data_dir / "node__inflow_method.csv",
                    provider=provider)
    )
    no_inflow_nodes = frozenset(
        n for n in nodes if (n, "no_inflow") in inflow_method_pairs
    )

    pdt_inflow: dict[tuple[str, str, str], float] = {}
    pdtni_path = solve_data_dir / "pdtNodeInflow.csv"
    pdtni_df = provider.get(_provider_key(pdtni_path))
    if pdtni_df is not None:
        for row in pdtni_df.iter_rows():
            if len(row) < 4:
                continue
            c0, c1, c2 = _cell_str(row[0]), _cell_str(row[1]), _cell_str(row[2])
            if c0 and c1 and c2:
                try:
                    pdt_inflow[(c0, c1, c2)] = float(row[3])
                except (ValueError, TypeError):
                    continue

    pos_n: list[str] = []
    pos_d: list[str] = []
    pos_t: list[str] = []
    pos_v: list[str] = []
    for n in nodes:
        if n in no_inflow_nodes:
            continue
        for (d, t) in dt:
            v = pdt_inflow.get((n, d, t), 0.0)
            pos_n.append(n)
            pos_d.append(d)
            pos_t.append(t)
            pos_v.append(repr(v if v >= 0 else 0.0))
    pos_frame = _utf8_frame({
        "node": pos_n, "period": pos_d, "time": pos_t, "value": pos_v,
    })

    neg_n: list[str] = []
    neg_d: list[str] = []
    neg_t: list[str] = []
    neg_v: list[str] = []
    for n in nodes:
        for (d, t) in dt:
            if n in no_inflow_nodes:
                cell = "0.0"
            else:
                v = pdt_inflow.get((n, d, t), 0.0)
                cell = repr(v if v < 0 else 0.0)
            neg_n.append(n)
            neg_d.append(d)
            neg_t.append(t)
            neg_v.append(cell)
    neg_frame = _utf8_frame({
        "node": neg_n, "period": neg_d, "time": neg_t, "value": neg_v,
    })

    return pos_frame, neg_frame


def derive_p_positive_inflow(
    input_dir: Path, solve_data_dir: Path,
    *, provider: "object | None" = None,
) -> pl.DataFrame:
    """Materialise just the ``p_positive_inflow`` frame."""
    pos, _neg = _derive_positive_negative_inflow(
        input_dir, solve_data_dir, provider=provider,
    )
    return pos


def derive_p_negative_inflow(
    input_dir: Path, solve_data_dir: Path,
    *, provider: "object | None" = None,
) -> pl.DataFrame:
    """Materialise just the ``p_negative_inflow`` frame."""
    _pos, neg = _derive_positive_negative_inflow(
        input_dir, solve_data_dir, provider=provider,
    )
    return neg


def emit_p_positive_negative_inflow(
    input_dir: Path, solve_data_dir: Path,
    *, provider,
) -> None:
    """Emit ``p_positive_negative_inflow`` to the Provider."""
    pos, neg = _derive_positive_negative_inflow(
        input_dir, solve_data_dir, provider=provider,
    )
    _emit(provider, "solve_data/p_positive_inflow.csv", pos)
    _emit(provider, "solve_data/p_negative_inflow.csv", neg)


# ---------------------------------------------------------------------------
# Phase 1 follow-up 5 — entity_period_calc_params: varCost + cap_reduction +
# ed_period_params + pssdt_varCost filters.
# ---------------------------------------------------------------------------


def _read_triples(path: Path,
                  *, provider: "object | None" = None,
                  ) -> list[tuple[str, str, str]]:
    df = provider.get(_provider_key(path))
    if df is None:
        return []
    out: list[tuple[str, str, str]] = []
    for row in df.iter_rows():
        if len(row) < 3:
            continue
        c0, c1, c2 = _cell_str(row[0]), _cell_str(row[1]), _cell_str(row[2])
        if c0 and c1 and c2:
            out.append((c0, c1, c2))
    return out


def _read_pdt_at_param(path: Path, param_col: int, param_value: str,
                       key_cols: tuple[int, ...],
                       val_col: int,
                       *, provider: "object | None" = None,
                       ) -> dict[tuple, float]:
    """Read a long-format pdtX CSV, filter rows where col[param_col] ==
    param_value, return dict[tuple(_cell_str(row[c]) for c in key_cols)] =
    float(row[val_col]).
    """
    out: dict[tuple, float] = {}
    df = provider.get(_provider_key(path))
    if df is None:
        return out
    for row in df.iter_rows():
        if (len(row) > max(param_col, val_col, *key_cols)
                and _cell_str(row[param_col]) == param_value):
            try:
                out[tuple(_cell_str(row[c]) for c in key_cols)] = float(row[val_col])
            except (ValueError, TypeError):
                continue
    return out


# ---- write_pdtProcess__source__sink__dt_varCost_pair (mod L1493, L1502) ----

def _derive_varCost_pair(
    input_dir: Path, solve_data_dir: Path,
    *, provider: "object | None" = None,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Build both ``pdtProcess__source__sink__dt_varCost`` frames in one
    pass — the basic and ``_alwaysProcess`` variants share the same
    OOC dictionaries / proc_src / proc_snk lookups.
    """
    pdt = _read_pdt_at_param(
        solve_data_dir / "pdtProcess.csv",
        param_col=1, param_value="other_operational_cost",
        key_cols=(0, 2, 3), val_col=4,
        provider=provider,
    )  # (process, period, time) → value
    pdt_src = _read_pdt_at_param(
        solve_data_dir / "pdtProcess_source.csv",
        param_col=2, param_value="other_operational_cost",
        key_cols=(0, 1, 3, 4), val_col=5,
        provider=provider,
    )  # (process, source, period, time) → value
    pdt_snk = _read_pdt_at_param(
        solve_data_dir / "pdtProcess_sink.csv",
        param_col=2, param_value="other_operational_cost",
        key_cols=(0, 1, 3, 4), val_col=5,
        provider=provider,
    )  # (process, sink, period, time) → value
    proc_src = frozenset(
        _read_pairs(input_dir / "process__source.csv", provider=provider)
    )
    proc_snk = frozenset(
        _read_pairs(input_dir / "process__sink.csv", provider=provider)
    )
    pss = _read_triples(
        solve_data_dir / "process_source_sink.csv", provider=provider,
    )
    pss_always = _read_triples(
        solve_data_dir / "process_source_sink_alwaysProcess.csv",
        provider=provider,
    )
    dt = _read_pairs(solve_data_dir / "steps_in_use.csv", provider=provider)

    def _build(rows_iter: list[tuple[str, str, str]],
               always: bool) -> pl.DataFrame:
        p_col: list[str] = []
        s_col: list[str] = []
        k_col: list[str] = []
        d_col: list[str] = []
        t_col: list[str] = []
        v_col: list[str] = []
        for (p, src, snk) in rows_iter:
            for (d, t) in dt:
                v = 0.0
                if (p, src) in proc_src:
                    v += pdt_src.get((p, src, d, t), 0.0)
                if (p, snk) in proc_snk:
                    v += pdt_snk.get((p, snk, d, t), 0.0)
                if always:
                    if (p, snk) in proc_snk or (p, snk) in proc_src:
                        v += pdt.get((p, d, t), 0.0)
                else:
                    v += pdt.get((p, d, t), 0.0)
                p_col.append(p)
                s_col.append(src)
                k_col.append(snk)
                d_col.append(d)
                t_col.append(t)
                v_col.append(repr(v))
        return _utf8_frame({
            "process": p_col, "source": s_col, "sink": k_col,
            "period": d_col, "time": t_col, "value": v_col,
        })

    return _build(pss, always=False), _build(pss_always, always=True)


def derive_pdtProcess__source__sink__dt_varCost(
    input_dir: Path, solve_data_dir: Path,
    *, provider: "object | None" = None,
) -> pl.DataFrame:
    """Materialise the ``pdtProcess__source__sink__dt_varCost`` frame."""
    basic, _always = _derive_varCost_pair(
        input_dir, solve_data_dir, provider=provider,
    )
    return basic


def derive_pdtProcess__source__sink__dt_varCost_alwaysProcess(
    input_dir: Path, solve_data_dir: Path,
    *, provider: "object | None" = None,
) -> pl.DataFrame:
    """Materialise the ``pdtProcess__source__sink__dt_varCost_alwaysProcess``
    frame."""
    _basic, always = _derive_varCost_pair(
        input_dir, solve_data_dir, provider=provider,
    )
    return always


def emit_pdtProcess__source__sink__dt_varCost_pair(
    input_dir: Path, solve_data_dir: Path,
    *, provider,
) -> None:
    """Emit ``pdtProcess__source__sink__dt_varCost_pair`` to the Provider."""
    basic, always = _derive_varCost_pair(
        input_dir, solve_data_dir, provider=provider,
    )
    _emit(provider, "solve_data/pdtProcess__source__sink__dt_varCost.csv",
          basic)
    _emit(provider,
          "solve_data/pdtProcess__source__sink__dt_varCost_alwaysProcess.csv",
          always)


# ---- write_pssdt_varCost_filters (mod L1498-1501) -------------------------

def _filter_rows_to_frame(
    rows: list[tuple[str, str, str, str, str]],
) -> pl.DataFrame:
    """5-column key-only Utf8 frame for the pssdt_varCost filter outputs."""
    return _utf8_frame({
        "process": [r[0] for r in rows],
        "source":  [r[1] for r in rows],
        "sink":    [r[2] for r in rows],
        "period":  [r[3] for r in rows],
        "time":    [r[4] for r in rows],
    })


def _derive_pssdt_varCost_filters(
    input_dir: Path, solve_data_dir: Path,
    *, provider: "object | None" = None,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Build all four ``pssdt_varCost_*`` filter frames in one pass."""
    pdt = _read_pdt_at_param(
        solve_data_dir / "pdtProcess.csv",
        param_col=1, param_value="other_operational_cost",
        key_cols=(0, 2, 3), val_col=4,
        provider=provider,
    )
    pdt_src = _read_pdt_at_param(
        solve_data_dir / "pdtProcess_source.csv",
        param_col=2, param_value="other_operational_cost",
        key_cols=(0, 1, 3, 4), val_col=5,
        provider=provider,
    )
    pdt_snk = _read_pdt_at_param(
        solve_data_dir / "pdtProcess_sink.csv",
        param_col=2, param_value="other_operational_cost",
        key_cols=(0, 1, 3, 4), val_col=5,
        provider=provider,
    )
    varcost: dict[tuple[str, str, str, str, str], float] = {}
    vp = solve_data_dir / "pdtProcess__source__sink__dt_varCost.csv"
    vp_df = provider.get(_provider_key(vp))
    if vp_df is not None:
        for row in vp_df.iter_rows():
            if len(row) < 6:
                continue
            c = [_cell_str(row[i]) for i in range(5)]
            if all(c):
                try:
                    varcost[(c[0], c[1], c[2], c[3], c[4])] = float(row[5])
                except (ValueError, TypeError):
                    continue

    proc_src = frozenset(
        _read_pairs(input_dir / "process__source.csv", provider=provider)
    )
    proc_snk = frozenset(
        _read_pairs(input_dir / "process__sink.csv", provider=provider)
    )
    pss_noEff = _read_triples(
        solve_data_dir / "process_source_sink_noEff.csv", provider=provider,
    )
    pss_eff = _read_triples(
        solve_data_dir / "process_source_sink_eff.csv", provider=provider,
    )
    dt = _read_pairs(solve_data_dir / "steps_in_use.csv", provider=provider)

    no_eff: list[tuple[str, str, str, str, str]] = []
    for (p, src, snk) in pss_noEff:
        for (d, t) in dt:
            if varcost.get((p, src, snk, d, t), 0.0):
                no_eff.append((p, src, snk, d, t))

    eff_src: list[tuple[str, str, str, str, str]] = []
    for (p, src, snk) in pss_eff:
        for (d, t) in dt:
            if (p, src) in proc_src and pdt_src.get((p, src, d, t), 0.0):
                eff_src.append((p, src, snk, d, t))

    eff_snk: list[tuple[str, str, str, str, str]] = []
    for (p, src, snk) in pss_eff:
        for (d, t) in dt:
            if (p, snk) in proc_snk and pdt_snk.get((p, snk, d, t), 0.0):
                eff_snk.append((p, src, snk, d, t))

    eff_conn: list[tuple[str, str, str, str, str]] = []
    for (p, src, snk) in pss_eff:
        for (d, t) in dt:
            if pdt.get((p, d, t), 0.0):
                eff_conn.append((p, src, snk, d, t))

    return (
        _filter_rows_to_frame(no_eff),
        _filter_rows_to_frame(eff_src),
        _filter_rows_to_frame(eff_snk),
        _filter_rows_to_frame(eff_conn),
    )


def derive_pssdt_varCost_noEff(
    input_dir: Path, solve_data_dir: Path,
    *, provider: "object | None" = None,
) -> pl.DataFrame:
    return _derive_pssdt_varCost_filters(
        input_dir, solve_data_dir, provider=provider,
    )[0]


def derive_pssdt_varCost_eff_unit_source(
    input_dir: Path, solve_data_dir: Path,
    *, provider: "object | None" = None,
) -> pl.DataFrame:
    return _derive_pssdt_varCost_filters(
        input_dir, solve_data_dir, provider=provider,
    )[1]


def derive_pssdt_varCost_eff_unit_sink(
    input_dir: Path, solve_data_dir: Path,
    *, provider: "object | None" = None,
) -> pl.DataFrame:
    return _derive_pssdt_varCost_filters(
        input_dir, solve_data_dir, provider=provider,
    )[2]


def derive_pssdt_varCost_eff_connection(
    input_dir: Path, solve_data_dir: Path,
    *, provider: "object | None" = None,
) -> pl.DataFrame:
    return _derive_pssdt_varCost_filters(
        input_dir, solve_data_dir, provider=provider,
    )[3]


def emit_pssdt_varCost_filters(
    input_dir: Path, solve_data_dir: Path,
    *, provider,
) -> None:
    """Emit ``pssdt_varCost_filters`` to the Provider."""
    no_eff, eff_src, eff_snk, eff_conn = _derive_pssdt_varCost_filters(
        input_dir, solve_data_dir, provider=provider,
    )
    _emit(provider, "solve_data/pssdt_varCost_noEff.csv", no_eff)
    _emit(provider, "solve_data/pssdt_varCost_eff_unit_source.csv", eff_src)
    _emit(provider, "solve_data/pssdt_varCost_eff_unit_sink.csv", eff_snk)
    _emit(provider, "solve_data/pssdt_varCost_eff_connection.csv", eff_conn)


# ---- write_cap_reduction_params (mod L1637-1663) --------------------------

def _read_p_side_3(path: Path,
                   *, provider: "object | None" = None,
                   ) -> dict[tuple[str, str, str], float]:
    """Per-side 4-col CSV ``(k1, k2, k3, value)`` → ``(k1, k2, k3) → v``."""
    out: dict[tuple[str, str, str], float] = {}
    df = provider.get(_provider_key(path))
    if df is None:
        return out
    for row in df.iter_rows():
        if len(row) < 4:
            continue
        c = [_cell_str(row[i]) for i in range(3)]
        if all(c):
            try:
                out[(c[0], c[1], c[2])] = float(row[3])
            except (ValueError, TypeError):
                continue
    return out


# ---- write_ed_period_params (mod L1252-1255 family, ed_*_period) ----------

def _read_ed_pairs(path: Path,
                   *, provider: "object | None" = None,
                   ) -> list[tuple[str, str]]:
    df = provider.get(_provider_key(path))
    if df is None:
        return []
    out: list[tuple[str, str]] = []
    for row in df.iter_rows():
        if len(row) < 2:
            continue
        c0, c1 = _cell_str(row[0]), _cell_str(row[1])
        if c0 and c1:
            out.append((c0, c1))
    return out


def _ed_period_compute(
    src_pairs: list[tuple[str, str]],
    mod_param: str,
    pp,  # PdLookup
    pn,  # PdLookup
    process_set: frozenset[str],
    node_set: frozenset[str],
) -> pl.DataFrame:
    """One ed_period_param frame — 3-col Utf8 ``(entity, period, value)``."""
    e_col: list[str] = []
    d_col: list[str] = []
    v_col: list[str] = []
    for e, d in src_pairs:
        if e in process_set:
            v = pp.get(e, mod_param, d)
        elif e in node_set:
            v = pn.get(e, mod_param, d)
        else:
            v = 0.0
        e_col.append(e)
        d_col.append(d)
        v_col.append(repr(v))
    return _utf8_frame({"entity": e_col, "period": d_col, "value": v_col})


_ED_PERIOD_PARAM_SPECS: tuple[tuple[str, str, str], ...] = (
    # (basename, src-pair-tag ['invest'|'divest'], mod_param)
    ("ed_invest_max_period.csv",       "invest", "invest_max_period"),
    ("ed_invest_min_period.csv",       "invest", "invest_min_period"),
    ("ed_divest_max_period.csv",       "divest", "retire_max_period"),
    ("ed_divest_min_period.csv",       "divest", "retire_min_period"),
    ("ed_cumulative_max_capacity.csv", "invest", "cumulative_max_capacity"),
    ("ed_cumulative_min_capacity.csv", "invest", "cumulative_min_capacity"),
)


def _ed_period_inputs(input_dir: Path, solve_data_dir: Path,
                      *, provider: "object | None" = None):
    from flextool.engine_polars._pdt_lookup import PdLookup
    pp = PdLookup(
        pd_csv=input_dir / "pd_process.csv",
        p_csv=input_dir / "p_process.csv",
        period_branch_csv=solve_data_dir / "period__branch.csv",
        provider=provider,
    )
    pn = PdLookup(
        pd_csv=input_dir / "pd_node.csv",
        p_csv=input_dir / "p_node.csv",
        period_branch_csv=solve_data_dir / "period__branch.csv",
        provider=provider,
    )
    process_set = frozenset(
        _read_singles(input_dir / "process.csv", provider=provider)
    )
    node_set = frozenset(
        _read_singles(input_dir / "node.csv", provider=provider)
    )
    ed_invest_pairs = _read_ed_pairs(
        solve_data_dir / "ed_invest.csv", provider=provider,
    )
    ed_divest_pairs = _read_ed_pairs(
        solve_data_dir / "ed_divest.csv", provider=provider,
    )
    return pp, pn, process_set, node_set, ed_invest_pairs, ed_divest_pairs


def derive_ed_invest_max_period(
    input_dir: Path, solve_data_dir: Path,
    *, provider: "object | None" = None,
) -> pl.DataFrame:
    pp, pn, ps, ns, inv, _div = _ed_period_inputs(
        input_dir, solve_data_dir, provider=provider,
    )
    return _ed_period_compute(inv, "invest_max_period", pp, pn, ps, ns)


def derive_ed_invest_min_period(
    input_dir: Path, solve_data_dir: Path,
    *, provider: "object | None" = None,
) -> pl.DataFrame:
    pp, pn, ps, ns, inv, _div = _ed_period_inputs(
        input_dir, solve_data_dir, provider=provider,
    )
    return _ed_period_compute(inv, "invest_min_period", pp, pn, ps, ns)


def derive_ed_divest_max_period(
    input_dir: Path, solve_data_dir: Path,
    *, provider: "object | None" = None,
) -> pl.DataFrame:
    pp, pn, ps, ns, _inv, div = _ed_period_inputs(
        input_dir, solve_data_dir, provider=provider,
    )
    return _ed_period_compute(div, "retire_max_period", pp, pn, ps, ns)


def derive_ed_divest_min_period(
    input_dir: Path, solve_data_dir: Path,
    *, provider: "object | None" = None,
) -> pl.DataFrame:
    pp, pn, ps, ns, _inv, div = _ed_period_inputs(
        input_dir, solve_data_dir, provider=provider,
    )
    return _ed_period_compute(div, "retire_min_period", pp, pn, ps, ns)


def derive_ed_cumulative_max_capacity(
    input_dir: Path, solve_data_dir: Path,
    *, provider: "object | None" = None,
) -> pl.DataFrame:
    pp, pn, ps, ns, inv, _div = _ed_period_inputs(
        input_dir, solve_data_dir, provider=provider,
    )
    return _ed_period_compute(inv, "cumulative_max_capacity", pp, pn, ps, ns)


def derive_ed_cumulative_min_capacity(
    input_dir: Path, solve_data_dir: Path,
    *, provider: "object | None" = None,
) -> pl.DataFrame:
    pp, pn, ps, ns, inv, _div = _ed_period_inputs(
        input_dir, solve_data_dir, provider=provider,
    )
    return _ed_period_compute(inv, "cumulative_min_capacity", pp, pn, ps, ns)


def emit_ed_period_params(
    input_dir: Path, solve_data_dir: Path,
    *, provider,
) -> None:
    """Emit ``ed_period_params`` to the Provider."""
    pp, pn, ps, ns, inv, div = _ed_period_inputs(
        input_dir, solve_data_dir, provider=provider,
    )
    pair_for = {"invest": inv, "divest": div}
    for fname, tag, mod_param in _ED_PERIOD_PARAM_SPECS:
        frame = _ed_period_compute(pair_for[tag], mod_param, pp, pn, ps, ns)
        _emit(provider, f"solve_data/{fname}", frame)
