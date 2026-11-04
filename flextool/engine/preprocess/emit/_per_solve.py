"""Per-solve set / invest-divest writers.

Two preprocessing families called per-solve (not per-write-input):

* ``per_solve_sets.write_per_solve_sets`` — ~24 set-projection / union
  outputs covering ``branch_set``, ``year_set``, ``period_*_set``,
  ``rp_*_set``, ``dt*``, ``period__timeline_set``, ``cnd_ladder*`` and
  the small storage-fix / online sets.
* ``invest_divest_sets.write_invest_divest_sets`` +
  ``write_ed_invest_forbidden_no_investment`` — the
  ed_invest / ed_divest / edd_history family + the gd_* group
  projections + the no-investment-lifetime gate.

Style mirrors :mod:`._leaf_sets` / :mod:`._mid_sets`:
eager polars CSV reads, expression chains, ``unique(maintain_order=True)``
for legacy-bit-identical row order, missing files treated as empty
frames.

Output bytes are CSV-identical to the legacy emitter — verified by
``tests/engine/test_writer_port_phase1.py`` parity tests.
"""
from __future__ import annotations

from pathlib import Path

import polars as pl

from flextool.engine.preprocess import _provider_keys as K
from flextool.engine.preprocess._provider_io import _emit
from flextool.engine.preprocess.params._branch import fan_period_labels
from flextool.engine.preprocess.params._derived import _expand_invest_branch_periods, run_start_period


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
# CSV I/O helpers — same conventions as _leaf_sets / _mid_sets.
# ---------------------------------------------------------------------------


def _read_csv(path: Path, columns: list[str],
              *, provider: "object | None" = None) -> pl.DataFrame:
    """Provider-only — returns an empty all-Utf8 frame on Provider miss.
    Step 2.5 Phase C dropped the disk-fallback arm.
    """
    from flextool.engine.preprocess._provider_io import (
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


def _project_values(df: pl.DataFrame, col_idx: int) -> list[str]:
    """The *col_idx* column as strings (blanks as ``""``), row order kept."""
    if col_idx >= len(df.columns):
        return []
    return df[df.columns[col_idx]].cast(pl.Utf8).fill_null("").to_list()


def _periods_with_labels(periods: list[str], pb_pairs: list[tuple[str, str]]) -> list[str]:
    """*periods* (the periods with steps) with each period label inserted right before its first fan member.

    A period label (:func:`~flextool.engine.preprocess.params._branch.fan_period_labels`) has no steps, but it is
    a period of the solve for the period-level bookkeeping (investment, years, NPV, lineage), so the period set
    keeps it at the position it had while it still had steps: just before the first of its ``period__branch``
    members that has steps (appended at the end when none has).
    """
    labels = fan_period_labels(set(pb_pairs))
    if not labels:
        return periods
    out = list(periods)
    for label in [d for d, b in pb_pairs if d == b and d in labels]:
        if label in out:
            continue
        members = [b for d, b in pb_pairs if d == label and b != label and b in out]
        if members:
            out.insert(min(out.index(m) for m in members), label)
        else:
            out.append(label)
    return out


# ---------------------------------------------------------------------------
# Family A — per_solve_sets.write_per_solve_sets
# ---------------------------------------------------------------------------


def emit_per_solve_sets(
    solve_data_dir: Path,
    *,
    provider,
    na_window: "list | None" = None,
    na_invest_window: "list | None" = None,
) -> None:
    """Emit ``per_solve_sets`` to the Provider.
    Emits the same ~24 frames under ``solve_data/<basename>`` keys via
    :func:`_emit` (dual-key registration).  *solve_data_dir* is retained
    because the function still reads sister CSVs from disk via the
    Provider-only read helpers (which derive keys from the path).

    *na_window* is the resolved Slice-G operational non-anticipativity
    window for this solve (see
    ``SolveConfig.non_anticipativity_periods_to_tristate``).  It drives
    the ``dt_non_anticipativity_set.csv`` content — the single source of
    truth read back by ``_branch.dt_non_anticipativity_lf``:

    * ``None`` (unset)      -> ``realized_dispatch u committed_in_fan u
      fix_storage`` union (``committed_in_fan`` = the cells the solve
      commits inside its fan, keyed on the period label, which has no
      dispatch of its own);
    * ``[]`` (empty)        -> header-only (empty) set -> operations free
      from t0 (all four non_anticipativity_* families off);
    * ``[(p, p), ...]``     -> those periods' timesteps, intersected with
      ``steps_in_use`` u ``label_steps`` (a period label keeps the steps
      it had before the stochastic expansion; the ONLY domain guard for
      the storage family, which joins the (d,t) set on ``d`` alone — F7).

    *na_invest_window* is the resolved v72 invest-NA window
    (``SolveConfig.non_anticipativity_invest_periods``, plain
    ``periods_to_tuples`` — unset == [] == no tie).  When non-empty it
    materialises ``pd_non_anticipativity_invest_periods.csv`` (a
    one-column period list), read back by
    ``_branch.pd_non_anticipativity_lf`` to populate the invest
    tie-set.  When unset/empty NO file is emitted, so the frame builder's
    reader returns ``None`` -> the legacy branch_start path -> a provably
    empty frame -> zero constraints -> byte-identical to prior behaviour
    (and no new CSV appears in any existing fixture's solve_data).
    """
    input_dir = solve_data_dir.parent / "input"

    df = _read_csv(solve_data_dir / "period__branch.csv", ["period", "branch"], provider=provider)
    branch_pairs = [(d, b) for d, b in zip(_project_values(df, 0), _project_values(df, 1)) if d and b]
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
                  _periods_with_labels(_project_column(df, 0), branch_pairs))
    _emit_singles(provider, "solve_data/time_in_use_set.csv", "time",
                  _project_column(df, 1))

    df = _read_csv(solve_data_dir / "steps_complete_solve.csv",
                   ["period", "time"], provider=provider)
    _emit_singles(provider, "solve_data/complete_time_in_use_set.csv", "time",
                  _project_column(df, 1))

    # rp_base_period_set / rp_rep_period_set used to be derived here from
    # rp_weights.csv, but emit_per_solve_sets runs BEFORE emit_rp_data in
    # the cascade — the Provider doesn't yet hold ``rp_weights`` at this
    # point, so the derivation always wrote empty frames.  Both sets are
    # now emitted by ``_solve_writers._compute_rp_frames`` (which
    # owns the rp_weights data) so they land populated for the RP path
    # and empty under ``emit_empty_rp_data`` for the non-RP path.

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
    # period__timeline_set names the ORIGINAL timeline (the one input/timeline.csv, p_timeline_duration_in_years
    # and the time set use) also when a solve-level new_stepduration re-pointed the timeset to an aggregated one.
    tt = _read_csv(input_dir / "timesets__timeline.csv",
                   ["timeset", "timeline", "original_timeline"], provider=provider)
    if tt.height == 0:
        tt = _read_csv(input_dir / "timesets__timeline.csv",
                       ["timeset", "timeline"], provider=provider).with_columns(
                           pl.col("timeline").alias("original_timeline"))
    tb_to_tl: dict[str, list[str]] = {}
    for ts, tl in zip(tt["timeset"].to_list(), tt["original_timeline"].to_list()):
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

    # Output rows are the realized steps only: ``model.output_horizon`` never widens them (its raw LP dump is
    # written separately, ``flextool/engine/output/_horizon_dump.py``).
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

    # Slice G — window-driven operational non-anticipativity set.  This
    # is the single source of truth: dt_non_anticipativity_lf reads it
    # back.  The window (na_window) tri-states the content; see the
    # emit_per_solve_sets docstring.
    dtna_seen: dict[tuple[str, str], None] = {}
    if na_window is None:
        # Unset -> the committed window: realized_dispatch u fix_storage
        # plus the cells committed inside the fan (``committed_in_fan``,
        # keyed on the period label as the forecast star expects).  The
        # fan cells are independent of whether a realization solve later
        # takes over the realized window: the ties hold in this LP.
        a_df = _read_csv(solve_data_dir / "realized_dispatch.csv",
                         ["period", "time"], provider=provider)
        c_df = _read_csv(solve_data_dir / "committed_in_fan.csv",
                         ["period", "time"], provider=provider)
        b_df = _read_csv(solve_data_dir / "fix_storage_timesteps.csv",
                         ["period", "time"], provider=provider)
        for src in (a_df, c_df, b_df):
            for d, t in zip(src["period"].to_list(), src["time"].to_list()):
                if d and t and (d, t) not in dtna_seen:
                    dtna_seen[(d, t)] = None
    elif len(na_window) == 0:
        # Explicit empty window -> emit header-only (operations free
        # from t0); dtna_seen stays empty.
        pass
    else:
        # Curated period list -> those periods' timesteps, intersected
        # with steps_in_use (the emit-time domain guard the storage NA
        # family relies on — it joins the set on ``d`` only, F7).
        na_periods = {p for (p, _p_in) in na_window}
        su_df = pl.concat([
            _read_csv(solve_data_dir / name, ["period", "time"], provider=provider).select(pl.all().cast(pl.Utf8))
            for name in ("steps_in_use.csv", "label_steps.csv")])
        for d, t in zip(su_df["period"].to_list(), su_df["time"].to_list()):
            if d and t and d in na_periods and (d, t) not in dtna_seen:
                dtna_seen[(d, t)] = None
    _emit_tuples(provider, "solve_data/dt_non_anticipativity_set.csv",
                 ("period", "time"), list(dtna_seen.keys()))

    # v72 — invest-NA window.  Materialise the tie period list ONLY when
    # the window is non-empty; absent file -> reader returns None ->
    # legacy empty frame -> byte-parity (no new CSV for unset fixtures).
    if na_invest_window:
        na_invest_periods = list(dict.fromkeys(
            p for (p, _p_in) in na_invest_window))
        _emit_singles(
            provider,
            "solve_data/pd_non_anticipativity_invest_periods.csv",
            "period", na_invest_periods)

    cn_df = _read_csv(input_dir / "commodity__node.csv",
                      ["commodity", "node"], provider=provider)
    cn_pairs = [
        (c, n) for c, n in zip(cn_df["commodity"].to_list(),
                               cn_df["node"].to_list())
        if c and n
    ]
    pin_use_df = _read_csv(solve_data_dir / "period_in_use_set.csv",
                           ["period"], provider=provider)
    # Ladder purchases are operational: a period label (no dispatch) has none.
    fan_labels = fan_period_labels(set(branch_pairs))
    pin_use = [v for v in pin_use_df["period"].to_list() if v and v not in fan_labels]
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
    from flextool.engine.preprocess._provider_translators import read_handoff_frame
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
    # Online variables live on every step of the solve, like the flows they bound (a unit or connection in a
    # coarse new_stepduration group too: membership has no effect on time resolution).
    su_df = _read_csv(solve_data_dir / "steps_in_use.csv",
                      ["period", "time"], provider=provider)
    steps = [(d, t) for d, t in zip(su_df["period"].to_list(), su_df["time"].to_list()) if d and t]
    online_rows: list[tuple[str, str, str]] = []
    for p in proc_online:
        seen_pdt: set[tuple[str, str]] = set()
        for d, t in steps:
            if (d, t) not in seen_pdt:
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

    def _coeff_entities(kinds: tuple[str, ...]) -> frozenset[str]:
        """Entities with a user-constraint coefficient of one of ``kinds`` (``p_<process|node>_constraint_<kind>
        _coeff.csv``, written by ``input_derivation._specs``)."""
        out: set[str] = set()
        for cls in ("process", "node"):
            for kind in kinds:
                out.update(e for e, _ in _read_pairs(
                    input_dir / f"p_{cls}_constraint_{kind}_coeff.csv", provider=provider))
        return frozenset(out)

    # Mirror of ``_derived._has_capacity_constraint_invest_set``: an invested, pre-built, capacity or
    # in-construction coefficient links the investment variable; those and a retired-capacity coefficient link the
    # retirement variable.
    invest_kinds = ("invested_capacity", "cumulative_pre_built_capacity", "capacity", "in_construction_capacity")
    has_capacity_constraint = _coeff_entities(invest_kinds)
    has_retire_constraint = _coeff_entities((*invest_kinds, "retired_capacity"))

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

    # Only capacity that existed before the run can be retired: a period without pre-existing capacity (none
    # given, or past its end of life) has no retirement option.
    pre_existing = {(str(e), str(d)) for e, d, v in _read_csv(solve_data_dir / "p_entity_pre_existing.csv",
                                                     ["entity", "period", "value"], provider=provider).iter_rows()
                    if v not in (None, "") and float(v) > 0.0}
    ed_divest_pairs: list[tuple[str, str]] = []
    for e in entityDivest:
        has_cap = e in has_retire_constraint
        for d in period_invest:
            if (eead.get((e, d), 0.0) != 0.0 or has_cap) and (e, d) in pre_existing:
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
    # Under recourse the cumulative-capacity rows sit on the invest axis ``v_invest`` has: each
    # anchor period's branch periods, never the period label (no column there, so a label row
    # would read ``0 ≥ cumulative_min_capacity``).  Flag-off the expansion returns the period
    # itself (byte-parity).
    invest_axis_of: dict[str, list[str]] = {}
    rows = []
    for e, d in ed_invest_pairs:
        if "cumulative_limits" not in methods_for_e.get(e, set()):
            continue
        if d not in invest_axis_of:
            invest_axis_of[d] = _expand_invest_branch_periods(
                [d], solve_data_dir.parent, provider=provider) or []
        rows.extend((e, d_axis) for d_axis in invest_axis_of[d])
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

    # #67 — a group's per-period invest/divest cap must reach the branch periods of its own
    # invest periods too: under recourse a branch period is itself an invest period
    # (``ed_invest_period_set`` already carries it, via ``ed_invest_set_from_source``'s
    # ``_expand_invest_branch_periods``), so ``maxInvestGroup_entity_period_p``'s row domain
    # needs a row there exactly as the entity cap already does — otherwise only the entity's
    # own (looser) cap applies in the branches.  Flag-off, or no branches, is a no-op
    # (byte-parity); ``gd_invest.csv`` / ``gd_divest.csv`` above are untouched.
    #
    # This stage has no per-entity, per-branch-period eligibility data of its own (``eea`` /
    # ``ed_invest_set`` above are anchor-only: the annuity pass that builds them never visits a
    # branch period, so a membership test against them here would always be false and silently
    # drop every branch row again). Listing every branch here and no more is deliberate: whether
    # a given member actually has the variable at a given branch period is decided downstream,
    # where the real (branch-aware, zero-respecting) variable domain is known --
    # ``_eq_group_entity_period`` scopes its constraint rows to that, not to this listing alone
    # (#82-adjacent fix: a branch-native 0 beating the anchor's value must not leave a vacuous
    # 0 ≥ positive-floor row here).
    gd_invest_pairs_period = list(gd_invest_pairs)
    seen_period_pairs = set(gd_invest_pairs_period)
    for g in group_invest:
        anchors_for_g: list[str] = []
        seen_anchor: set[str] = set()
        for gg, d in gd_invest_pairs:
            if gg == g and d not in seen_anchor:
                anchors_for_g.append(d)
                seen_anchor.add(d)
        if not anchors_for_g:
            continue
        expanded = _expand_invest_branch_periods(
            anchors_for_g, solve_data_dir.parent, provider=provider) or []
        for d in expanded:
            if (g, d) not in seen_period_pairs:
                gd_invest_pairs_period.append((g, d))
                seen_period_pairs.add((g, d))

    rows = [
        (g, d) for g, d in gd_invest_pairs_period
        if methods_for_g.get(g, set()) & _INVEST_PERIOD_METHODS
    ]
    _emit_tuples(provider, "solve_data/gd_invest_period.csv",
                 ("group", "period"), rows)
    rows = [
        (g, d) for g, d in gd_invest_pairs_period
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
    # The end of life counts from the start of the run (``p_years_d`` covers the period history).
    d0 = run_start_period(p_years_d)

    def _life_sum(e: str) -> float:
        return 0.0 if d0 is None else p_years_d.get(d0, 0.0) + ed_lifetime.get((e, d0), 0.0)

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
