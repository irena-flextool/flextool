"""Writer-port Phase 2 (sub-dispatch 1) — per-solve set / invest-divest writers.

Native polars port of two preprocessing families called from
:func:`flextool.flextoolrunner.preprocessing.solve_time.run` (NOT from
``input_writer.write_input`` — these are per-solve, not per-write-input):

* ``per_solve_sets.write_per_solve_sets`` (legacy 416 LOC) — ~24
  set-projection / union outputs covering ``branch_set``, ``year_set``,
  ``period_*_set``, ``rp_*_set``, ``dt*``, ``period__timeline_set``,
  ``cnd_ladder*`` and the small storage-fix / online sets.
* ``invest_divest_sets.write_invest_divest_sets`` +
  ``write_ed_invest_forbidden_no_investment`` (legacy 361 LOC) — the
  ed_invest / ed_divest / edd_history family + the gd_* group
  projections + the no-investment-lifetime gate.

Style mirrors :mod:`._emit_leaf_sets` / :mod:`._emit_mid_sets`:
eager polars CSV reads, expression chains, ``unique(maintain_order=True)``
for legacy-bit-identical row order, missing files treated as empty
frames.

Output bytes are CSV-identical to the legacy emitter — verified by
``tests/engine_polars/test_writer_port_phase1.py`` parity tests.
"""
from __future__ import annotations

from pathlib import Path

import polars as pl

from flextool.engine_polars._emit_provider_io import _emit


# ---------------------------------------------------------------------------
# Method-enum constants — mirror preprocessing/invest_divest_sets.py and
# flextool/flextool_base.dat.  Frozen sets so identity-equality checks at
# join time are stable.
# ---------------------------------------------------------------------------

_INVEST_PERIOD_METHODS: frozenset[str] = frozenset((
    "invest_period", "invest_period_total",
    "invest_retire_period", "invest_retire_period_total",
))
_DIVEST_PERIOD_METHODS: frozenset[str] = frozenset((
    "retire_period", "retire_period_total",
    "invest_retire_period", "invest_retire_period_total",
))


# ---------------------------------------------------------------------------
# CSV I/O helpers — same conventions as _emit_leaf_sets / _emit_mid_sets.
# ---------------------------------------------------------------------------


def _read_csv(path: Path, columns: list[str],
              *, provider: "object | None" = None) -> pl.DataFrame:
    """Provider-only — returns an empty all-Utf8 frame on Provider miss.
    Step 2.5 Phase C dropped the disk-fallback arm.
    """
    from flextool.engine_polars._emit_provider_io import (
        _provider_key,
        _provider_lookup_positional,
    )
    seeded = _provider_lookup_positional(
        provider, _provider_key(path), path, columns,
    )
    if seeded is not None:
        return seeded
    return pl.DataFrame(
        {c: [] for c in columns},
        schema={c: pl.Utf8 for c in columns},
    )


def _to_utf8_frame(
    headers: tuple[str, ...],
    rows: list[tuple],
) -> pl.DataFrame:
    """Build an all-``Utf8`` polars frame from a header tuple + row list."""
    cols: dict[str, list[str]] = {h: [] for h in headers}
    for row in rows:
        for h, v in zip(headers, row):
            cols[h].append(v if isinstance(v, str) else str(v))
    return pl.DataFrame(cols, schema={h: pl.Utf8 for h in headers})


def _emit_singles(provider, key: str, header: str, values: list[str]) -> None:
    """Provider-emit a one-column header + rows frame under *key*."""
    _emit(provider, key, _to_utf8_frame((header,), [(v,) for v in values]))


def _emit_tuples(provider, key: str, header: tuple[str, ...],
                 rows: list[tuple[str, ...]]) -> None:
    """Provider-emit a multi-column header + tuple rows frame under *key*."""
    _emit(provider, key, _to_utf8_frame(header, rows))


def _read_singles(path: Path,
                  *, provider: "object | None" = None) -> list[str]:
    """List the first column of a header+rows CSV, dropping blanks."""
    df = _read_csv(path, ["v"], provider=provider)
    return [v for v in df["v"].to_list() if v]


def _read_pairs(path: Path,
                *, provider: "object | None" = None) -> list[tuple[str, str]]:
    """List 2-tuples from a 2+-column CSV, dropping rows with any blank."""
    df = _read_csv(path, ["a", "b"], provider=provider)
    return [(a, b) for a, b in zip(df["a"].to_list(), df["b"].to_list())
            if a and b]


def _project_column(df: pl.DataFrame, col_idx: int) -> list[str]:
    """Ordered-unique projection of the *col_idx* column, dropping blanks.

    Mirrors the legacy ``_project_column`` helper's dict-of-None semantics
    by using polars' ``unique(maintain_order=True)`` over a single-column
    frame.
    """
    if col_idx >= len(df.columns):
        return []
    col_name = df.columns[col_idx]
    s = df[col_name].cast(pl.Utf8).fill_null("")
    return [v for v in s.unique(maintain_order=True).to_list() if v]


# ---------------------------------------------------------------------------
# Family A — per_solve_sets.write_per_solve_sets
# ---------------------------------------------------------------------------


def emit_per_solve_sets(solve_data_dir: Path, *, provider) -> None:
    """Emit ``per_solve_sets`` to the Provider.
    Emits the same ~24 frames under ``solve_data/<basename>`` keys via
    :func:`_emit` (dual-key registration).  *solve_data_dir* is retained
    because the function still reads sister CSVs from disk via the
    Provider-only read helpers (which derive keys from the path).
    """
    input_dir = solve_data_dir.parent / "input"

    df = _read_csv(solve_data_dir / "period__branch.csv", ["period", "branch"], provider=provider)
    _emit_singles(provider, "solve_data/branch_set.csv", "branch",
                  _project_column(df, 1))

    df = _read_csv(solve_data_dir / "p_years_represented.csv",
                   ["period", "year"], provider=provider)
    _emit_singles(provider, "solve_data/year_set.csv", "year",
                  _project_column(df, 1))

    df = _read_csv(solve_data_dir / "steps_in_timeline.csv",
                   ["period", "step"], provider=provider)
    _emit_singles(provider, "solve_data/period_from_period_time_set.csv",
                  "period", _project_column(df, 0))

    df = _read_csv(solve_data_dir / "steps_in_use.csv",
                   ["period", "time"], provider=provider)
    _emit_singles(provider, "solve_data/period_in_use_set.csv", "period",
                  _project_column(df, 0))
    _emit_singles(provider, "solve_data/time_in_use_set.csv", "time",
                  _project_column(df, 1))

    df = _read_csv(solve_data_dir / "steps_complete_solve.csv",
                   ["period", "time"], provider=provider)
    _emit_singles(provider, "solve_data/complete_time_in_use_set.csv", "time",
                  _project_column(df, 1))

    df = _read_csv(solve_data_dir / "rp_weights.csv",
                   ["base", "rep", "weight"], provider=provider)
    _emit_singles(provider, "solve_data/rp_base_period_set.csv", "period",
                  _project_column(df, 0))
    _emit_singles(provider, "solve_data/rp_rep_period_set.csv", "period",
                  _project_column(df, 1))

    df = _read_csv(solve_data_dir / "period_block_time.csv",
                   ["period", "block_first", "step"], provider=provider)
    pb_pairs = [
        (p, b) for p, b in zip(df["period"].to_list(),
                               df["block_first"].to_list())
        if p and b
    ]
    pb_dedup: list[tuple[str, str]] = []
    seen: dict[tuple[str, str], None] = {}
    for r in pb_pairs:
        if r not in seen:
            seen[r] = None
            pb_dedup.append(r)
    _emit_tuples(provider, "solve_data/period_block_set.csv",
                 ("period", "block_first"), pb_dedup)

    df = _read_csv(
        solve_data_dir / "step_previous.csv",
        ["period", "time", "time_previous", "tprev_within_ts",
         "period_previous", "tprev_within_solve"], provider=provider
    )
    triples = list(zip(
        df["period"].to_list(),
        df["time"].to_list(),
        df["time_previous"].to_list(),
    ))
    seen3: dict[tuple[str, str, str], None] = {}
    dtt_rows: list[tuple[str, str, str]] = []
    for r in triples:
        if all(r) and r not in seen3:
            seen3[r] = None
            dtt_rows.append(r)
    _emit_tuples(provider, "solve_data/dtt_set.csv",
                 ("period", "time", "time_previous"), dtt_rows)

    df = _read_csv(solve_data_dir / "fix_storage_timesteps.csv",
                   ["period", "step"], provider=provider)
    _emit_singles(provider, "solve_data/d_fix_storage_period_set.csv",
                  "period", _project_column(df, 0))

    pa = _read_csv(input_dir / "periods_available.csv", ["period"], provider=provider)
    pfpt = _read_csv(
        solve_data_dir / "period_from_period_time_set.csv", ["period"], provider=provider
    )
    seen_p: dict[str, None] = {}
    for v in pa["period"].to_list() + pfpt["period"].to_list():
        if v and v not in seen_p:
            seen_p[v] = None
    _emit_singles(provider, "solve_data/period_set.csv", "period",
                  list(seen_p.keys()))

    seen_pa: dict[str, None] = {}
    for fname in (
        "period_group.csv", "period_node.csv", "period_commodity.csv",
        "period_process.csv", "period_solve.csv",
    ):
        for v in _read_csv(solve_data_dir / fname, ["period"], provider=provider)["period"].to_list():
            if v and v not in seen_pa:
                seen_pa[v] = None
    for v in _read_csv(solve_data_dir / "branch_set.csv", ["period"], provider=provider)["period"].to_list():
        if v and v not in seen_pa:
            seen_pa[v] = None
    _emit_singles(provider, "solve_data/periodAll_set.csv", "period",
                  list(seen_pa.keys()))

    seen_b: dict[str, None] = {}

    df = _read_csv(solve_data_dir / "entity_block.csv", ["entity", "block"], provider=provider)
    for v in df["block"].to_list():
        if v and v not in seen_b:
            seen_b[v] = None

    df = _read_csv(solve_data_dir / "process_side_block.csv",
                   ["process", "side", "block"], provider=provider)
    for v in df["block"].to_list():
        if v and v not in seen_b:
            seen_b[v] = None

    df = _read_csv(solve_data_dir / "process_block.csv",
                   ["process", "block"], provider=provider)
    for v in df["block"].to_list():
        if v and v not in seen_b:
            seen_b[v] = None

    df = _read_csv(solve_data_dir / "block_step_duration.csv",
                   ["block", "period", "step", "duration"], provider=provider)
    for v in df["block"].to_list():
        if v and v not in seen_b:
            seen_b[v] = None

    df = _read_csv(solve_data_dir / "overlap_set.csv",
                   ["period", "b_coarse", "t_coarse",
                    "b_fine", "t_fine", "fraction"], provider=provider)
    for v in df["b_coarse"].to_list():
        if v and v not in seen_b:
            seen_b[v] = None
    for v in df["b_fine"].to_list():
        if v and v not in seen_b:
            seen_b[v] = None

    _emit_singles(provider, "solve_data/block_set.csv", "block",
                  list(seen_b.keys()))

    spt = _read_csv(input_dir / "timesets_in_use.csv",
                    ["solve", "period", "timeset"], provider=provider)
    cur_solve = _read_csv(solve_data_dir / "solve_current.csv", ["solve"], provider=provider)
    cur_solve_set = frozenset(v for v in cur_solve["solve"].to_list() if v)
    tt = _read_csv(input_dir / "timesets__timeline.csv",
                   ["timeset", "timeline"], provider=provider)
    tb_to_tl: dict[str, list[str]] = {}
    for ts, tl in zip(tt["timeset"].to_list(), tt["timeline"].to_list()):
        if ts and tl:
            tb_to_tl.setdefault(ts, []).append(tl)
    pt_seen: dict[tuple[str, str], None] = {}
    for s, d, tb in zip(spt["solve"].to_list(),
                        spt["period"].to_list(),
                        spt["timeset"].to_list()):
        if s in cur_solve_set:
            for tl in tb_to_tl.get(tb, ()):
                key = (d, tl)
                if key not in pt_seen:
                    pt_seen[key] = None
    _emit_tuples(provider, "solve_data/period__timeline_set.csv",
                 ("period", "timeline"), list(pt_seen.keys()))

    enable = _read_csv(solve_data_dir / "enable_optional_outputs.csv",
                       ["flag"], provider=provider)
    enable_set = frozenset(v for v in enable["flag"].to_list() if v)
    if "output_horizon" in enable_set:
        rows_df = _read_csv(solve_data_dir / "steps_in_use.csv",
                            ["period", "time"], provider=provider)
    else:
        rows_df = _read_csv(solve_data_dir / "realized_dispatch.csv",
                            ["period", "time"], provider=provider)
    drd_pairs_raw = list(zip(rows_df["period"].to_list(),
                             rows_df["time"].to_list()))
    seen_drd: dict[tuple[str, str], None] = {}
    drd_pairs: list[tuple[str, str]] = []
    for r in drd_pairs_raw:
        if all(r) and r not in seen_drd:
            seen_drd[r] = None
            drd_pairs.append(r)
    _emit_tuples(provider, "solve_data/dt_realize_dispatch_set.csv",
                 ("period", "time"), drd_pairs)

    drp_seen: dict[str, None] = {}
    for d, _t in drd_pairs:
        if d not in drp_seen:
            drp_seen[d] = None
    _emit_singles(provider, "solve_data/d_realized_period_set.csv", "period",
                  list(drp_seen.keys()))

    drealize = _read_csv(
        solve_data_dir / "realized_invest_periods_of_current_solve.csv",
        ["period"], provider=provider
    )
    union_seen: dict[str, None] = dict(drp_seen)
    for v in drealize["period"].to_list():
        if v and v not in union_seen:
            union_seen[v] = None
    _emit_singles(provider,
                  "solve_data/d_realize_dispatch_or_invest_set.csv", "period",
                  list(union_seen.keys()))

    a_df = _read_csv(solve_data_dir / "realized_dispatch.csv",
                     ["period", "time"], provider=provider)
    b_df = _read_csv(solve_data_dir / "fix_storage_timesteps.csv",
                     ["period", "time"], provider=provider)
    dtna_seen: dict[tuple[str, str], None] = {}
    for src in (a_df, b_df):
        for d, t in zip(src["period"].to_list(), src["time"].to_list()):
            if d and t and (d, t) not in dtna_seen:
                dtna_seen[(d, t)] = None
    _emit_tuples(provider, "solve_data/dt_non_anticipativity_set.csv",
                 ("period", "time"), list(dtna_seen.keys()))

    cn_df = _read_csv(input_dir / "commodity__node.csv",
                      ["commodity", "node"], provider=provider)
    cn_pairs = [
        (c, n) for c, n in zip(cn_df["commodity"].to_list(),
                               cn_df["node"].to_list())
        if c and n
    ]
    pin_use_df = _read_csv(solve_data_dir / "period_in_use_set.csv",
                           ["period"], provider=provider)
    pin_use = [v for v in pin_use_df["period"].to_list() if v]
    with_ladder = frozenset(
        v for v in _read_csv(solve_data_dir / "commodity_with_ladder.csv",
                             ["commodity"], provider=provider)["commodity"].to_list() if v
    )
    with_ladder_cum = frozenset(
        v for v in _read_csv(
            solve_data_dir / "commodity_with_ladder_cumulative.csv",
            ["commodity"], provider=provider
        )["commodity"].to_list() if v
    )
    with_ladder_ann = frozenset(
        v for v in _read_csv(
            solve_data_dir / "commodity_with_ladder_annual.csv",
            ["commodity"], provider=provider
        )["commodity"].to_list() if v
    )
    cum_df = _read_csv(input_dir / "commodity_ladder_cumulative.csv",
                       ["commodity", "tier"], provider=provider)
    ann_df = _read_csv(solve_data_dir / "commodity__tier_ann.csv",
                       ["commodity", "tier"], provider=provider)
    tiers_for_cum: dict[str, list[str]] = {}
    for c, i in zip(cum_df["commodity"].to_list(), cum_df["tier"].to_list()):
        if c and i:
            tiers_for_cum.setdefault(c, []).append(i)
    tiers_for_ann: dict[str, list[str]] = {}
    for c, i in zip(ann_df["commodity"].to_list(), ann_df["tier"].to_list()):
        if c and i:
            tiers_for_ann.setdefault(c, []).append(i)

    cnd_rows = [
        (c, n, d) for (c, n) in cn_pairs for d in pin_use
        if c in with_ladder
    ]
    _emit_tuples(provider, "solve_data/cnd_ladder_set.csv",
                 ("commodity", "node", "period"),
                 list(dict.fromkeys(cnd_rows)))

    cum_rows_out: list[tuple[str, str, str, str]] = []
    for (c, n) in cn_pairs:
        if c not in with_ladder_cum:
            continue
        tiers = tiers_for_cum.get(c, ())
        for d in pin_use:
            for i in tiers:
                cum_rows_out.append((c, n, d, i))
    _emit_tuples(provider, "solve_data/cndi_ladder_cum_set.csv",
                 ("commodity", "node", "period", "tier"),
                 list(dict.fromkeys(cum_rows_out)))

    ann_rows_out: list[tuple[str, str, str, str]] = []
    for (c, n) in cn_pairs:
        if c not in with_ladder_ann:
            continue
        tiers = tiers_for_ann.get(c, ())
        for d in pin_use:
            for i in tiers:
                ann_rows_out.append((c, n, d, i))
    _emit_tuples(provider, "solve_data/cndi_ladder_ann_set.csv",
                 ("commodity", "node", "period", "tier"),
                 list(dict.fromkeys(ann_rows_out)))

    union4: dict[tuple[str, str, str, str], None] = {}
    for r in cum_rows_out + ann_rows_out:
        if r not in union4:
            union4[r] = None
    _emit_tuples(provider, "solve_data/cndi_ladder_set.csv",
                 ("commodity", "node", "period", "tier"),
                 list(union4.keys()))

    spv_df = _read_csv(
        solve_data_dir / "step_previous.csv",
        ["period", "time", "time_previous", "tprev_within_ts",
         "period_previous", "tprev_within_solve"], provider=provider
    )
    quads_raw = list(zip(
        spv_df["period_previous"].to_list(),
        spv_df["tprev_within_solve"].to_list(),
        spv_df["period"].to_list(),
        spv_df["time"].to_list(),
    ))
    seen_q: dict[tuple[str, str, str, str], None] = {}
    quads: list[tuple[str, str, str, str]] = []
    for r in quads_raw:
        if all(r) and r not in seen_q:
            seen_q[r] = None
            quads.append(r)
    _emit_tuples(provider, "solve_data/dtdt_next_set.csv",
                 ("period_prev", "time_prev_solve", "period", "time"), quads)

    # Phase 4.1g — derive ``n_fix_storage_{quantity,price,usage}_set`` from
    # the canonical handoff Provider keys (``handoff/fix_storage_<metric>``,
    # schema ``[node, period, step, p_fix_storage_<metric>]``).  Only the
    # ``node`` column is consumed; the translator seeds these keys at
    # iteration start, replacing the legacy ``solve_data/fix_storage_*``
    # Provider read path ahead of the wide-field deletion in 4.1k.
    from flextool.engine_polars import _provider_keys as K
    from flextool.engine_polars._provider_translators import read_handoff_frame
    for handoff_key, dst in (
        (K.HANDOFF_FIX_STORAGE_QUANTITY, "n_fix_storage_quantity_set.csv"),
        (K.HANDOFF_FIX_STORAGE_PRICE,    "n_fix_storage_price_set.csv"),
        (K.HANDOFF_FIX_STORAGE_USAGE,    "n_fix_storage_usage_set.csv"),
    ):
        df = read_handoff_frame(provider, handoff_key)
        if df is None:
            nodes: list[str] = []
        else:
            nodes = _project_column(df, 0)
        _emit_singles(provider, f"solve_data/{dst}", "node", nodes)

    proc_online = frozenset(
        v for v in _read_csv(solve_data_dir / "process_online.csv",
                             ["process"], provider=provider)["process"].to_list() if v
    )
    proc_blocks: dict[str, list[str]] = {}
    pb_df = _read_csv(solve_data_dir / "process_block.csv",
                      ["process", "block"], provider=provider)
    for p, b in zip(pb_df["process"].to_list(), pb_df["block"].to_list()):
        if p and b:
            proc_blocks.setdefault(p, []).append(b)
    block_dt: dict[str, set[tuple[str, str]]] = {}
    bsd_df = _read_csv(solve_data_dir / "block_step_duration.csv",
                       ["block", "period", "step", "duration"], provider=provider)
    for b, d, t in zip(bsd_df["block"].to_list(),
                       bsd_df["period"].to_list(),
                       bsd_df["step"].to_list()):
        if b and d and t:
            block_dt.setdefault(b, set()).add((d, t))
    su_df = _read_csv(solve_data_dir / "steps_in_use.csv",
                      ["period", "time"], provider=provider)
    dt_set = frozenset(
        (d, t) for d, t in zip(su_df["period"].to_list(),
                               su_df["time"].to_list())
        if d and t
    )
    online_rows: list[tuple[str, str, str]] = []
    for p in proc_online:
        seen_pdt: set[tuple[str, str]] = set()
        blocks_for_p = proc_blocks.get(p)
        if blocks_for_p:
            for b in blocks_for_p:
                for d, t in block_dt.get(b, ()):
                    if (d, t) in dt_set and (d, t) not in seen_pdt:
                        seen_pdt.add((d, t))
                        online_rows.append((p, d, t))
        else:
            for d, t in zip(su_df["period"].to_list(),
                            su_df["time"].to_list()):
                if d and t and (d, t) not in seen_pdt:
                    seen_pdt.add((d, t))
                    online_rows.append((p, d, t))
    _emit_tuples(provider, "solve_data/p_online_dt_set.csv",
                 ("process", "period", "step"), online_rows)


# ---------------------------------------------------------------------------
# Family B — invest_divest_sets.write_invest_divest_sets
#                + write_ed_invest_forbidden_no_investment
# ---------------------------------------------------------------------------


def _read_keyed_value_kv(path: Path,
                         *, provider: "object | None" = None) -> dict[str, float]:
    """Read a (key, value) CSV into a dict (float values, blanks skipped)."""
    df = _read_csv(path, ["key", "value"], provider=provider)
    out: dict[str, float] = {}
    for k, v in zip(df["key"].to_list(), df["value"].to_list()):
        if not k or v is None or v == "":
            continue
        try:
            out[k] = float(v)
        except ValueError:
            continue
    return out


def _read_pdv(path: Path,
              *, provider: "object | None" = None) -> dict[tuple[str, str], float]:
    """Read a 3-col (entity, period, value) CSV into a dict."""
    df = _read_csv(path, ["entity", "period", "value"], provider=provider)
    out: dict[tuple[str, str], float] = {}
    for e, d, v in zip(df["entity"].to_list(),
                       df["period"].to_list(),
                       df["value"].to_list()):
        if not e or not d:
            continue
        try:
            out[(e, d)] = float(v)
        except (ValueError, TypeError):
            continue
    return out


def emit_invest_divest_sets(input_dir: Path, solve_data_dir: Path,
                             *, provider) -> None:
    """Emit ``invest_divest_sets`` to the Provider.
    Emits the same 15 frames under ``solve_data/<basename>`` keys via
    :func:`_emit` (dual-key registration).
    """
    entityInvest = _read_singles(solve_data_dir / "entityInvest.csv", provider=provider)
    entityDivest = _read_singles(solve_data_dir / "entityDivest.csv", provider=provider)
    period_invest = _read_singles(
        solve_data_dir / "invest_periods_of_current_solve.csv", provider=provider
    )
    period_in_use = _read_singles(solve_data_dir / "period_in_use_set.csv", provider=provider)
    period_with_history = _read_singles(
        solve_data_dir / "period_with_history.csv", provider=provider
    )
    process_set = frozenset(_read_singles(input_dir / "process.csv", provider=provider))
    node_set = frozenset(_read_singles(input_dir / "node.csv", provider=provider))
    entity_set = _read_singles(input_dir / "entity.csv", provider=provider)

    pcc_inv = _read_pairs(
        input_dir / "p_process_constraint_invested_capacity_coefficient.csv", provider=provider
    )
    pcc_pre = _read_pairs(
        input_dir / "p_process_constraint_pre_built_capacity_coefficient.csv", provider=provider
    )
    ncc_inv = _read_pairs(
        input_dir / "p_node_constraint_invested_capacity_coefficient.csv", provider=provider
    )
    ncc_pre = _read_pairs(
        input_dir / "p_node_constraint_pre_built_capacity_coefficient.csv", provider=provider
    )
    has_capacity_constraint = (
        frozenset(p for p, _ in pcc_inv)
        | frozenset(p for p, _ in pcc_pre)
        | frozenset(n for n, _ in ncc_inv)
        | frozenset(n for n, _ in ncc_pre)
    )

    eea = _read_pdv(solve_data_dir / "ed_entity_annual.csv", provider=provider)
    eead = _read_pdv(solve_data_dir / "ed_entity_annual_divest.csv", provider=provider)

    eim = _read_pairs(input_dir / "entity__invest_method.csv", provider=provider)
    methods_for_e: dict[str, set[str]] = {}
    for e, m in eim:
        methods_for_e.setdefault(e, set()).add(m)

    elm = _read_pairs(solve_data_dir / "entity__lifetime_method.csv", provider=provider)
    lm_for_e: dict[str, set[str]] = {}
    for e, m in elm:
        lm_for_e.setdefault(e, set()).add(m)

    p_years_d = _read_keyed_value_kv(solve_data_dir / "p_years_d.csv", provider=provider)
    edEntity_lifetime = _read_pdv(solve_data_dir / "edEntity_lifetime.csv", provider=provider)

    group_invest = _read_singles(solve_data_dir / "group_invest.csv", provider=provider)
    group_divest = _read_singles(solve_data_dir / "group_divest.csv", provider=provider)
    del group_divest
    group_entity = _read_pairs(solve_data_dir / "group_entity.csv", provider=provider)
    entities_for_g: dict[str, list[str]] = {}
    for g, e in group_entity:
        entities_for_g.setdefault(g, []).append(e)
    gim = _read_pairs(input_dir / "group__invest_method.csv", provider=provider)
    methods_for_g: dict[str, set[str]] = {}
    for g, m in gim:
        methods_for_g.setdefault(g, set()).add(m)

    ed_invest_pairs: list[tuple[str, str]] = []
    for e in entityInvest:
        has_cap = e in has_capacity_constraint
        for d in period_invest:
            if eea.get((e, d), 0.0) != 0.0 or has_cap:
                ed_invest_pairs.append((e, d))
    _emit_tuples(provider, "solve_data/ed_invest.csv",
                 ("entity", "period"), ed_invest_pairs)
    ed_invest_set = frozenset(ed_invest_pairs)

    ed_divest_pairs: list[tuple[str, str]] = []
    for e in entityDivest:
        has_cap = e in has_capacity_constraint
        for d in period_invest:
            if eead.get((e, d), 0.0) != 0.0 or has_cap:
                ed_divest_pairs.append((e, d))
    _emit_tuples(provider, "solve_data/ed_divest.csv",
                 ("entity", "period"), ed_divest_pairs)
    ed_divest_set = frozenset(ed_divest_pairs)

    rows = [(e, d) for e, d in ed_invest_pairs
            if methods_for_e.get(e, set()) & _INVEST_PERIOD_METHODS]
    _emit_tuples(provider, "solve_data/ed_invest_period.csv",
                 ("entity", "period"), rows)
    rows = [(e, d) for e, d in ed_invest_pairs
            if methods_for_e.get(e, set()) & _DIVEST_PERIOD_METHODS]
    _emit_tuples(provider, "solve_data/ed_divest_period.csv",
                 ("entity", "period"), rows)
    rows = [(e, d) for e, d in ed_invest_pairs
            if "cumulative_limits" in methods_for_e.get(e, set())]
    _emit_tuples(provider, "solve_data/ed_invest_cumulative.csv",
                 ("entity", "period"), rows)

    _emit_tuples(provider, "solve_data/pd_invest.csv", ("process", "period"),
                 [(e, d) for e, d in ed_invest_pairs if e in process_set])
    _emit_tuples(provider, "solve_data/nd_invest.csv", ("node", "period"),
                 [(e, d) for e, d in ed_invest_pairs if e in node_set])
    _emit_tuples(provider, "solve_data/pd_divest.csv", ("process", "period"),
                 [(e, d) for e, d in ed_divest_pairs if e in process_set])
    _emit_tuples(provider, "solve_data/nd_divest.csv", ("node", "period"),
                 [(e, d) for e, d in ed_divest_pairs if e in node_set])

    edd_choice: list[tuple[str, str, str]] = []
    edd_auto: list[tuple[str, str, str]] = []
    edd_noinv: list[tuple[str, str, str]] = []
    for e in entity_set:
        e_lm = lm_for_e.get(e, set())
        is_choice = "reinvest_choice" in e_lm
        is_auto = "reinvest_automatic" in e_lm
        is_noinv = "no_investment" in e_lm
        if not (is_choice or is_auto or is_noinv):
            continue
        for d_h in period_with_history:
            life = edEntity_lifetime.get((e, d_h), 0.0)
            pdy_dh = p_years_d.get(d_h, 0.0)
            for d in period_in_use:
                pdy_d = p_years_d.get(d, 0.0)
                if is_choice and pdy_d >= pdy_dh and pdy_d < pdy_dh + life:
                    edd_choice.append((e, d_h, d))
                if is_auto and pdy_d >= pdy_dh:
                    edd_auto.append((e, d_h, d))
                if is_noinv and pdy_d >= pdy_dh and pdy_d < pdy_dh + life:
                    edd_noinv.append((e, d_h, d))
    _emit_tuples(provider, "solve_data/edd_history_choice.csv",
                 ("entity", "period_history", "period"), edd_choice)
    _emit_tuples(provider, "solve_data/edd_history_automatic.csv",
                 ("entity", "period_history", "period"), edd_auto)
    _emit_tuples(provider, "solve_data/edd_history_no_investment.csv",
                 ("entity", "period_history", "period"), edd_noinv)

    edd_seen: dict[tuple[str, str, str], None] = {}
    for r in edd_choice + edd_auto + edd_noinv:
        if r not in edd_seen:
            edd_seen[r] = None
    edd_history = list(edd_seen.keys())
    _emit_tuples(provider, "solve_data/edd_history.csv",
                 ("entity", "period_history", "period"), edd_history)
    invest_set = frozenset(entityInvest)
    edd_history_invest = [r for r in edd_history if r[0] in invest_set]
    _emit_tuples(provider, "solve_data/edd_history_invest.csv",
                 ("entity", "period_history", "period"), edd_history_invest)
    edd_invest = [
        (e, d_inv, d) for (e, d_inv, d) in edd_history_invest
        if (e, d_inv) in ed_invest_set
    ]
    _emit_tuples(provider, "solve_data/edd_invest.csv",
                 ("entity", "period_history", "period"), edd_invest)
    _ = ed_divest_set

    gd_invest_pairs: list[tuple[str, str]] = []
    for g in group_invest:
        ents = entities_for_g.get(g, ())
        for d in period_invest:
            if any((e, d) in ed_invest_set for e in ents):
                gd_invest_pairs.append((g, d))
    _emit_tuples(provider, "solve_data/gd_invest.csv",
                 ("group", "period"), gd_invest_pairs)
    _emit_tuples(provider, "solve_data/gd_divest.csv",
                 ("group", "period"), gd_invest_pairs[:])

    rows = [
        (g, d) for g, d in gd_invest_pairs
        if methods_for_g.get(g, set()) & _INVEST_PERIOD_METHODS
    ]
    _emit_tuples(provider, "solve_data/gd_invest_period.csv",
                 ("group", "period"), rows)
    rows = [
        (g, d) for g, d in gd_invest_pairs
        if methods_for_g.get(g, set()) & _DIVEST_PERIOD_METHODS
    ]
    _emit_tuples(provider, "solve_data/gd_divest_period.csv",
                 ("group", "period"), rows)


def emit_ed_invest_forbidden_no_investment(
    input_dir: Path, solve_data_dir: Path,
    *, provider,
) -> None:
    """Emit ``ed_invest_forbidden_no_investment`` to the Provider."""
    del input_dir  # legacy signature parity; no input/ reads here.
    ed_invest_pairs = _read_pairs(solve_data_dir / "ed_invest.csv", provider=provider)
    elm = _read_pairs(solve_data_dir / "entity__lifetime_method.csv", provider=provider)
    no_invest_set = frozenset(e for e, m in elm if m == "no_investment")

    p_years_d: dict[str, float] = {}
    pyd_df = _read_csv(solve_data_dir / "p_years_d.csv",
                       ["period", "value"], provider=provider)
    for d, v in zip(pyd_df["period"].to_list(), pyd_df["value"].to_list()):
        if not d:
            continue
        try:
            p_years_d[d] = float(v)
        except (ValueError, TypeError):
            continue

    ed_lifetime = _read_pdv(solve_data_dir / "edEntity_lifetime.csv", provider=provider)
    period_first = _read_singles(solve_data_dir / "period_first.csv", provider=provider)

    def _life_sum(e: str) -> float:
        return sum(
            p_years_d.get(d_first, 0.0)
            + ed_lifetime.get((e, d_first), 0.0)
            for d_first in period_first
        )

    cached_sum: dict[str, float] = {}
    rows: list[tuple[str, str]] = []
    for e, d in ed_invest_pairs:
        if e not in no_invest_set:
            continue
        s = cached_sum.get(e)
        if s is None:
            s = _life_sum(e)
            cached_sum[e] = s
        if p_years_d.get(d, 0.0) >= s:
            rows.append((e, d))
    _emit_tuples(provider,
                 "solve_data/ed_invest_forbidden_no_investment.csv",
                 ("entity", "period"), rows)
