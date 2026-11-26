""":func:`build_handoff_from_solution`: the handoff of a solved solve to the next one."""

from __future__ import annotations

import logging
from pathlib import Path
import polars as pl
from flextool.engine.input._axis_enums import alias_to_axis
from flextool.engine.preprocess.emit import _flowgroup_accumulators as _fg
from flextool.engine.preprocess.flex_data._readers import _provider_has, _provider_read
from flextool.engine.preprocess.flex_data._container import FlexData
from flextool.engine.preprocess.flex_data._handoff_inputs import (
    _committed_fan_online_cells,
    _label_tied_cohorts,
    _last_realized_step,
    _read_realize_invest_periods,
    _read_singles_csv,
    _read_unitsize_long,
    _roll_end_flow,
    _step_duration_at,
    _step_duration_frame,
    _tied_copy_labels,
    round_whole_unit_events,
    snap_carried_online,
)
from flextool.engine.preprocess.flex_data._handoff_commodity import (
    _extract_cum_sim_hours,
    _extract_cumulative_commodity,
)
from flextool.engine.preprocess.flex_data._handoff_storage import _storage_start_and_level, _storage_usage_mwh


def build_handoff_from_solution(
    sol, work_folder: Path, solve_name: str,
    prior_handoff=None,
    *,
    flex_data: "FlexData | None" = None,
    parent_handoff=None,
    provider: "object | None" = None,
    realized_dispatch_parents: "set[str] | None" = None,
    scale_the_objective: float | None = None,
    cohort: "pl.DataFrame | None" = None,
):
    """Build a ``SolveHandoff`` from a polar_high ``Solution`` + the work
    folder's per-solve metadata, in memory.

    Carriers:

    * ``realized_cohort`` — the realized investments as cohorts (decision period, start of operation, end of life,
      size), chain-cumulative.
    * ``realized_retire`` — per-(entity, cohort, decision period) chain-cumulative retirements (realized periods
      only).
    * ``realized_commission`` — the realized investments with a construction lead time, chain-cumulative.
    * ``divest_cumulative`` — per-entity chain-cumulative divest.
    * ``roll_end_state`` — last-step v_state per nodeState node.
    * ``fix_storage`` — wide [node, period, time, quantity, price, usage]
      with NULL columns for inactive metrics.  ``quantity`` is populated
      from v_state at fix_storage_timesteps for fix_quantity nodes; the
      price (dual-based) and usage (flow-based) variants stay NULL until
      a fixture exercises them and the dual / flow extraction lands.
    * ``cumulative_co2`` / ``cumulative_flowgroup`` — per-(group, period) running totals (``emit/_*_accumulators``).
    * ``cumulative_commodity`` — per-(commodity, tier, period),
      derived via :func:`_extract_cumulative_commodity` (this-roll
      v_trade + ``prior_handoff.cumulative_commodity``).
    * ``cum_sim_hours`` — per-period running sim-hour total, derived
      via :func:`_extract_cum_sim_hours` (this-roll realized hours +
      ``prior_handoff.cum_sim_hours``).
    (Δ.1 — ``periods_already_emitted`` was here; it moved to
    ``_writer.OutputWriterState`` since it's a writer-side
    emission gate, not a solver handoff carrier.  The on-disk source
    ``solve_data/period_capacity.csv`` is unchanged.)

    ``scale_the_objective`` is the objective scale this solve was built with (None = 1.0, as
    ``build_flextool``); the ``fix_price`` carrier divides it out of the duals.

    ``cohort`` is the solve's cohort table when ``flex_data`` is ``None`` (the Benders invest solve hands its
    undecomposed ``FlexData.cohort``); with ``flex_data`` its ``cohort`` is used.  ``realized_cohort`` /
    ``realized_retire`` (:func:`._cohort_handoff.realized_cohorts`) carry the realized investments as cohorts and the
    retirements per cohort.

    *provider* (the sub-solve's :class:`FlexDataProvider`) is required:
    ``None`` raises ``ValueError``.  It must already carry the per-solve
    preprocessing frames for ``solve_name`` (``solve_data/period_first``,
    ``solve_data/solve__ed_invest``,
    ``solve_data/realized_invest_periods_of_current_solve``, etc.);
    *work_folder* only forms the provider keys and is never read.
    """
    import polars as pl  # local — keep this helper's import surface narrow
    # Native import — Γ.8.D moved SolveHandoff into engine.
    from flextool.engine.solve._handoff import SolveHandoff
    # Slice C Guard 3 — local imports keep this helper's import surface narrow.
    from flextool.engine.input._solve_state import FlexToolConfigError
    from flextool.engine.preprocess._stochastic_detect import (
        synthetic_branch_tokens,
    )

    if provider is None:
        raise ValueError(
            "build_handoff_from_solution requires the sub-solve's FlexDataProvider; "
            "it never reads the work folder on disk."
        )
    sd = work_folder / "solve_data"
    unitsize = _read_unitsize_long(work_folder, provider=provider)

    # Prior solve's accumulators — sourced from the in-memory handoff
    # carriers when supplied, else empty (multi-solve cascade always
    # passes the parent handoff).
    prior_divested: dict[str, float] = {}
    prior_commission: list[tuple[str, str, str, float]] = []
    if prior_handoff is not None:
        if getattr(prior_handoff, "realized_commission", None) is not None:
            for r in prior_handoff.realized_commission.iter_rows(named=True):
                prior_commission.append((str(r["entity"]), str(r["period"]), str(r["commission_period"]),
                                         float(r["value"])))
        if prior_handoff.divest_cumulative is not None:
            for r in prior_handoff.divest_cumulative.iter_rows(named=True):
                prior_divested[str(r["entity"])] = float(r["value"])

    # ---- v_invest / v_divest from polar_high ----
    invest_by_ed: dict[tuple[str, str], float] = {}
    divest_by_ed: dict[tuple[str, str], float] = {}
    for var_name, entity_col in (("v_invest_p", "p"), ("v_invest_n", "n")):
        if var_name in sol._vars:
            df = sol.value(var_name)
            for r in df.iter_rows(named=True):
                v = float(r["value"])
                if v <= 1e-12:
                    continue
                invest_by_ed[(str(r[entity_col]), str(r["d"]))] = v
    for var_name, entity_col in (("v_divest_p", "p"), ("v_divest_n", "n")):
        if var_name in sol._vars:
            df = sol.value(var_name)
            for r in df.iter_rows(named=True):
                v = float(r["value"])
                if v <= 1e-12:
                    continue
                e = str(r[entity_col])
                dd = str(r["d"])
                divest_by_ed[(e, dd)] = divest_by_ed.get((e, dd), 0.0) + v
    # The retirements of model-built cohorts (v_divest_built), per (entity, cohort, decision period).
    divest_built_by_ecd: dict[tuple[str, str, str], float] = {}
    for var_name, entity_col in (("v_divest_built_p", "p"), ("v_divest_built_n", "n")):
        if var_name in sol._vars:
            for e, c, dd, v in sol.value(var_name).select(entity_col, "cohort", "d", "value").iter_rows():
                if float(v) > 1e-12:
                    key = (str(e), str(c), str(dd))
                    divest_built_by_ecd[key] = divest_built_by_ecd.get(key, 0.0) + float(v)
    # The life extensions (v_extend), per (entity, extended cohort, period).
    extend_by_ecd: dict[tuple[str, str, str], float] = {}
    for var_name, entity_col in (("v_extend_p", "p"), ("v_extend_n", "n")):
        if var_name in sol._vars:
            for e, c, dd, v in sol.value(var_name).select(entity_col, "cohort", "d", "value").iter_rows():
                if float(v) > 1e-12:
                    key = (str(e), str(c), str(dd))
                    extend_by_ecd[key] = extend_by_ecd.get(key, 0.0) + float(v)

    # ---- the decisions tied across the forecasts (``invest_solve_branching_method``
    # ``investments_after_reveal_period``): one decision of every branch copy of the reveal period, the label has
    # none.  It is realized under the label (its real period name), taken from the reference copy. ----
    _tied, _label_of = _tied_copy_labels(flex_data)
    for _ref, _label in _tied.items():
        for _by in (invest_by_ed, divest_by_ed):
            for (_e, _d), _v in [kv for kv in _by.items() if kv[0][1] == _ref]:
                _by[(_e, _label)] = _v
        for _by3 in (divest_built_by_ecd, extend_by_ecd):
            for (_e, _c, _d), _v in [kv for kv in _by3.items() if kv[0][2] == _ref]:
                # A cohort the reference copy built is the label's (``_label_tied_cohorts``).
                _by3[(_e, ":".join(_label if tok == _ref else tok for tok in _c.split(":")), _label)] = _v

    # ---- Slice D (H, §10.1): EXPLICIT non-realized invest drop ----
    # ``invest_by_ed`` collects every solved v_invest row, including
    # branch-named (non-realized-scenario) ones fanned in under recourse.
    # Historically the ``d in realize_invest`` commit gate below dropped
    # those by accident; make it explicit here — partition on synthetic
    # branch membership and debug-log the dropped mass BEFORE the commit
    # loop, so the belt-and-braces Slice C Guard 3 (which asserts no
    # synthetic token survived into the committed rows) is a true
    # invariant check, not the mechanism.  Flag-off / deterministic:
    # ``_synth`` is empty and this block is a no-op (byte-parity).
    _synth_handoff = synthetic_branch_tokens(
        getattr(flex_data, "period_branch_full", None))
    if _synth_handoff:
        _dropped = {k: v for k, v in invest_by_ed.items()
                    if k[1] in _synth_handoff}
        if _dropped:
            logging.getLogger("flextool.engine.preprocess.flex_data").debug(
                "recourse handoff: dropping %d non-realized invest rows, "
                "total capacity %.6g (periods %s)",
                len(_dropped), sum(_dropped.values()),
                sorted({k[1] for k in _dropped}),
            )
        invest_by_ed = {k: v for k, v in invest_by_ed.items()
                        if k[1] not in _synth_handoff}
        _dropped_div = {k: v for k, v in divest_by_ed.items()
                        if k[1] in _synth_handoff}
        if _dropped_div:
            logging.getLogger("flextool.engine.preprocess.flex_data").debug(
                "recourse handoff: dropping %d non-realized divest rows, "
                "total capacity %.6g (periods %s)",
                len(_dropped_div), sum(_dropped_div.values()),
                sorted({k[1] for k in _dropped_div}),
            )
        divest_by_ed = {k: v for k, v in divest_by_ed.items()
                        if k[1] not in _synth_handoff}
        divest_built_by_ecd = {k: v for k, v in divest_built_by_ecd.items() if k[2] not in _synth_handoff}
        extend_by_ecd = {k: v for k, v in extend_by_ecd.items() if k[2] not in _synth_handoff}

    # ---- the realized decision periods of this solve ----
    realize_invest = _read_realize_invest_periods(
        sd / "realized_invest_periods_of_current_solve.csv",
        provider=provider,
    )

    # ---- divest_cumulative input: count only realized periods (mirrors the
    # ``d in realize_invest`` invest gate above) — v_divest
    # rows in non-realized look-ahead periods or alternative recourse branches
    # must not inflate the chain-cumulative divest prior.
    divest_by_e: dict[str, float] = {}
    realized_divest_by_ed: dict[tuple[str, str], float] = {}
    for (_e, _d), _v in divest_by_ed.items():
        if _d in realize_invest:
            divest_by_e[_e] = divest_by_e.get(_e, 0.0) + _v
            realized_divest_by_ed[(_e, _d)] = _v
    realized_divest_built: dict[tuple[str, str, str], float] = {}
    for (_e, _c, _d), _v in divest_built_by_ecd.items():
        if _d in realize_invest:
            divest_by_e[_e] = divest_by_e.get(_e, 0.0) + _v
            realized_divest_built[(_e, _c, _d)] = _v

    # ---- the realized investments with a construction lead time, by decision period and the period they start
    # to operate in (``ed_invest_commission``, found by this solve on the run calendar): a later solve counts them
    # as coming online / in construction (``realized_commission``, chain-cumulative) ----
    commission_of: dict[tuple[str, str], str] = {}
    _commission = getattr(flex_data, "ed_invest_commission", None)
    if _commission is not None and _commission.height > 0:
        for _e, _o, _dc in _commission.select(
                pl.col("e").cast(pl.Utf8), pl.col("d").cast(pl.Utf8), pl.col("d_c").cast(pl.Utf8)).iter_rows():
            if _dc is not None:
                commission_of[(_e, _o)] = _dc
    commission_rows: list[tuple[str, str, str, float]] = list(prior_commission)
    for (_e, _o), _v in invest_by_ed.items():
        if _o in realize_invest and (_e, _o) in commission_of:
            commission_rows.append((_e, _o, commission_of[(_e, _o)], _v * unitsize.get(_e, 1.0)))

    # ---- Slice C Guard 3: no realized investment may be decided in a
    #      synthetic stochastic-branch fan member (recourse plan §3.3).
    #      The token set is the BROAD d != b b-token set of
    #      period_branch_full (shared idiom ``synthetic_branch_tokens``) —
    #      deliberately broader than Guard 2's PIU-intersected set:
    #      committing invest at a metadata mirror (e.g. period1_realized,
    #      the rolling-bookkeeping token) must ALSO be forbidden, and those
    #      rolling metadata tokens never legitimately appear in
    #      realized_invest.  None-degrade: when flex_data /
    #      period_branch_full is unavailable (the Benders TIER-1 snapshot
    #      call passes flex_data=None by design) the check is a
    #      no-op — that path is a Benders invest solve, which Guard 2
    #      already forbids for stochastic models, so no synthetic token can
    #      reach it.  Check (a) here, (b) on the carried cohorts below.
    _pbf = getattr(flex_data, "period_branch_full", None)
    _synth = synthetic_branch_tokens(_pbf)
    if _synth:
        # (a) config-level: realize_invest lists no synthetic token.
        _bad_cfg = sorted(set(map(str, realize_invest)) & _synth)
        if _bad_cfg:
            raise FlexToolConfigError(
                f"Solve '{solve_name}': realized_invest_periods lists "
                f"synthetic stochastic-branch period(s) {_bad_cfg}. "
                f"Only real (realized-branch) period names may be "
                f"realized for investment."
            )

    # ---- divest_cumulative: prior + sum_d v_divest * unitsize ----
    entity_divest = set(_read_singles_csv(sd / "entityDivest.csv",
                                            provider=provider))
    div_rows: list[tuple[str, float]] = []
    for e in sorted(entity_divest):
        cum = prior_divested.get(e, 0.0) + divest_by_e.get(e, 0.0) * unitsize.get(e, 1.0)
        div_rows.append((e, cum))

    # ---- realized_cohort / realized_retire: the realized investments as cohorts (their rows of the cohort table,
    # placed once by this solve) and the retirements per cohort, chain-cumulative ----
    from flextool.engine.preprocess.params._cohort_handoff import realized_cohorts
    _cohort_table = _label_tied_cohorts(getattr(flex_data, "cohort", None) if flex_data is not None else cohort,
                                        _tied, _label_of)
    realized_cohort_df, realized_retire_df = realized_cohorts(
        getattr(prior_handoff, "realized_cohort", None) if prior_handoff is not None else None,
        getattr(prior_handoff, "realized_retire", None) if prior_handoff is not None else None,
        _cohort_table,
        {(e, o): v for (e, o), v in invest_by_ed.items() if o in realize_invest},
        {(e, d): v for (e, d), v in realized_divest_by_ed.items() if e in entity_divest and v != 0.0},
        unitsize,
        {k: v for k, v in realized_divest_built.items() if k[0] in entity_divest},
        {k: v for k, v in extend_by_ecd.items() if k[2] in realize_invest})
    # Slice C Guard 3 (b): no carried cohort is decided in a synthetic branch period.
    if _synth and realized_cohort_df is not None:
        _bad_cohort = sorted(set(realized_cohort_df["period_decided"].drop_nulls().to_list()) & _synth)
        if _bad_cohort:
            raise FlexToolConfigError(
                f"Solve '{solve_name}': handoff would carry investment cohorts decided in synthetic stochastic-branch "
                f"period(s) {_bad_cohort} — these belong to a non-realized scenario and must not enter the "
                f"committed capacity chain (recourse plan §3.3).")

    # ---- roll_end_state: v_state[n, last_t] * unitsize per nodeState node ----
    # Mirrors flextool's ``write_p_roll_continue_state``: takes v_state at
    # the LAST realized (period, time) pair (from ``realized_dispatch.csv``),
    # multiplies by p_entity_unitsize, and emits a (node, value) row per
    # nodeState node.  Skipped when the solve has no nodeState or no
    # realized dispatch.  See flextool/process_outputs/handoff_writers.py:
    # 250-271 (``_load_realized_period_time_last``) and 425-468.
    #
    # NOTE: source MUST be ``realized_dispatch`` (end-of-realized-commitment,
    # e.g. roll_7 jump=4 realizes t0029-t0032 → last = t0032), NOT
    # ``period__time_last`` (end-of-horizon, e.g. t0036 for the same roll).
    # Using the horizon-end variant breaks rolling storage handoff: the next
    # roll starts from a wrong initial v_state.
    roll_end_state_df = None
    nodes_state = _read_singles_csv(sd / "nodeState.csv", provider=provider)
    # Carry the COMMITTED JUMP-END storage state of the realized TRUNK
    # (restores GMPL value parity).  Prefer the Provider anchor
    # ``solve_data/realized_dispatch`` (trunk-only, committed jump-end —
    # emitted by ``emit_realized_dispatch`` before the handoff is built), in
    # its emitted order, over the in-memory ``flex_data.realized_dispatch``
    # (the de-duplicated output row set).  Fall back to the latter only when
    # the Provider key is absent (unit-test synthesis, no real solve).
    # The last realized step in time (``_last_realized_step``: the emitted order of
    # ``solve_data/realized_dispatch``, equivalent to ``_load_realized_period_time_last`` followed by
    # ``last_pairs[-1]`` in v3.32.0's ``write_p_roll_continue_state``), shared by the three roll carriers.
    last_step = _last_realized_step(sd, flex_data, provider=provider)
    # The committed cells of a forecasts-only fan carry unit commitment to the next solve when no realization
    # solve follows this one (a realization solve's own carries continue the run then).
    fan_online = (None if realized_dispatch_parents is not None and solve_name in realized_dispatch_parents
                  else _committed_fan_online_cells(sd, flex_data, provider=provider))
    if nodes_state and last_step is not None and "v_state" in sol._vars:
        last_d, last_t = last_step
        rcs_rows: list[tuple[str, float]] = []
        # Filter the variable's col_id frame to ONLY the last
        # (d, t) keys, then pull values for those rows alone.
        # Avoids materializing the full per-step v_state frame.
        # ``sol.value`` does ``v.frame.select(dims).with_columns(
        # value=col_value[col_id])``; we replicate that against the
        # pre-filtered frame so the result equals filtering
        # ``sol.value("v_state")`` to the last (d, t).
        _v = sol._vars["v_state"]
        _last_frame = _v.frame.filter(
            (pl.col("d") == last_d) & (pl.col("t") == last_t))
        if _last_frame.height > 0:
            _ids = _last_frame["col_id"].to_numpy()
            _vals = sol.col_value[_ids]
            last_state = _last_frame.select(*_v.dims).with_columns(
                value=pl.Series(_vals))
        else:
            last_state = None
        if last_state is not None:
            nodes_state_set = set(nodes_state)
            for r in last_state.iter_rows(named=True):
                n = str(r["n"])
                if n not in nodes_state_set:
                    continue
                v = float(r["value"]) * unitsize.get(n, 1.0)
                rcs_rows.append((n, v))
        # A representative-period node's realised level is its base-period chain at the end of the last realised
        # period (v_state_inter at that period's last base period): the state at the last realised step lies
        # inside a representative block and has a free offset.  The next solve starts its chain from it
        # (model.core rp_inter_period_start_carry, owner decision).
        rp_nodes = getattr(flex_data, "nodeState_rp", None) if flex_data is not None else None
        if rcs_rows and rp_nodes is not None and rp_nodes.height > 0 and "v_state_inter" in sol._vars:
            inter = sol.value("v_state_inter")
            if inter is not None and inter.height > 0:
                inter = (inter.with_columns(pl.col("n").cast(pl.Utf8), pl.col("d").cast(pl.Utf8),
                                            pl.col("b").cast(pl.Utf8))
                         .filter(pl.col("d") == str(last_d)).sort("n", "b")
                         .group_by("n", maintain_order=True).last())
                chain = {n: float(v) * unitsize.get(n, 1.0) for n, v in zip(inter["n"].to_list(),
                                                                           inter["value"].to_list())}
                rp_set = set(rp_nodes["n"].cast(pl.Utf8).to_list())
                rcs_rows = [(n, chain[n]) if n in rp_set and n in chain else (n, v) for n, v in rcs_rows]
        if rcs_rows:
            roll_end_state_df = pl.DataFrame(
                rcs_rows, schema=["node", "value"], orient="row")

    # ---- upward_roll_end_state: routed UPWARD to parent storage's next roll ----
    # Per specs/feature_fixes.md §1: dispatch sub-solves pass their
    # realized end-of-horizon v_state UPWARD to the parent storage
    # solve's next roll, replacing the parent's own previously-
    # predicted state.  Always-on for any storage→dispatch nesting;
    # no opt-in flag.
    #
    # Carrier value is the same as ``roll_end_state``: the LAST realized
    # (period, step) v_state per nodeState node.  The carrier exists as
    # a distinct SolveHandoff field so the parent storage's consumer
    # can prefer it explicitly over the sequential-prior fallback (see
    # flex_data.py:2810-2819).  In the current sequential-prior orchestration,
    # the last completed solve before the parent's next roll IS the
    # child dispatch sub-solve, so sequential prior already fans the
    # dispatch's roll_end_state into the parent's provider key — but
    # adding the explicit upward carrier makes the intent visible and
    # leaves room for future routing logic (e.g. picking a non-last
    # child by topology rather than completion order).
    upward_roll_end_state_df = roll_end_state_df

    # ---- roll_end_online: committed-jump-end v_online per online process ----
    # OC1 — the online analog of ``roll_end_state`` (above).  Carries the
    # realized committed-jump-end ``v_online`` so the next roll's
    # ``online_change`` first-step row (``dt_online_carry``) sees the
    # prior roll's commitment as predecessor instead of the free start of
    # a first step without a carry (no start-up at every boundary).  Kept in
    # unit-count space (NOT ×unitsize): the online dynamics operate on
    # ``v_online ∈ [0, existing_count]``, the same units the cyclic lag
    # carries.  Read at the same last realized step as ``roll_end_state``
    # (``last_step``, computed whether or not the solve has storage).  Keyed on ``p`` alone
    # (a process has one startup_method at a time, so linear/integer never
    # mix for the same ``p``); linear and integer rows are unioned.  Rounded
    # (``snap_carried_online``): an integer column to whole units, a continuous one
    # near 0 to 0; a process moved to 0 carries its ``roll_end_flow`` as 0 too.
    roll_end_online_df = None
    _has_online_var = ("v_online_linear" in sol._vars
                       or "v_online_integer" in sol._vars)
    online_step = last_step
    if fan_online is not None:
        # The last committed step lies in the fan (after every pre-reveal realized step): read the tied value.
        online_step = (fan_online["d_ref"][-1], fan_online["t"][-1])
    # The processes whose carried units online the rounding moved to 0 (from a value that was not 0): their
    # ``roll_end_flow`` is carried as 0 with them (below), so the next roll does not ramp from a flow of units it
    # sees offline.
    snapped_off: set[str] = set()
    if online_step is not None and _has_online_var:
        from flextool.engine.output._coarse_flows import var_is_integer, var_values_at, var_zero_tolerance

        last_d_on, last_t_on = online_step
        rco_rows: list[tuple[str, float]] = []
        for _vname in ("v_online_linear", "v_online_integer"):
            # The committed-jump-end (d, t) alone (a coarse process's units online hold through its coarse step,
            # whose last fine step is the last realized step: the column of that coarse step).
            _lst = var_values_at(sol, flex_data, _vname, str(last_d_on), str(last_t_on))
            if _lst is None:
                continue
            _snapped = snap_carried_online(_lst, var_is_integer(sol, _vname), var_zero_tolerance(sol, _vname))
            for raw, r in zip(_lst["value"], _snapped.iter_rows(named=True)):
                rco_rows.append((str(r["p"]), float(r["value"])))
                if float(r["value"]) == 0.0 and float(raw) != 0.0:
                    snapped_off.add(str(r["p"]))
        if rco_rows:
            roll_end_online_df = pl.DataFrame(
                rco_rows, schema=["process", "value"], orient="row")

    # ---- roll_end_flow: v_flow of the ramp-limited and ramp-cost arcs at the last realized step ----
    # The next roll's first step ramps from this flow (``_load_ramp_carry``; the next roll applies it only
    # where the complete solve's predecessor of its first step is this step and adjacent).  Units of
    # unitsize, like v_flow.  ``duration`` = this step's length (the ramp room uses both step lengths).
    roll_end_flow_df = _roll_end_flow(sol, flex_data, last_step)
    if roll_end_flow_df is not None and snapped_off:
        roll_end_flow_df = roll_end_flow_df.with_columns(
            pl.when(pl.col("process").is_in(sorted(snapped_off))).then(0.0).otherwise(pl.col("value")).alias("value"))
    # ---- roll_end_step: the last realized step itself (with its duration) ----
    # The next roll links its first step to it (ramp rows with roll_end_flow, and the ramp output also in
    # models without ramp limits).
    roll_end_step_df = None
    if fan_online is not None:
        # The committed step under its period label (the name the complete solve's timeline links to).
        roll_end_step_df = pl.DataFrame({"period": [fan_online["label"][-1]], "step": [fan_online["t"][-1]],
                                         "duration": [_step_duration_at(flex_data, online_step)]})
    elif last_step is not None:
        roll_end_step_df = pl.DataFrame({"period": [last_step[0]], "step": [last_step[1]],
                                         "duration": [_step_duration_at(flex_data, last_step)]})

    # Is THIS solve the parent of an auto-spawned realized-dispatch sibling
    # (OC1 separate_dispatch)?  Shared by fix_online below and by the ladder /
    # CO2 rolling-accumulator extraction further down (#59 fix 3): a
    # separate_dispatch parent's own dispatch is a degenerate, weight-0
    # placeholder (the sibling re-solves the same window for real right
    # after it), so neither extraction may add this solve's own realized
    # contribution — the sibling's own extraction call does that instead.
    _is_dispatch_sibling_parent = (
        realized_dispatch_parents is not None
        and solve_name in realized_dispatch_parents
    )

    # ---- fix_online: NA-tied v_online of the _fix_in_uc processes for the realization solve ----
    # For the parent of a realization solve (its name is a registered
    # realization parent) carry the committed ``v_online`` of the processes
    # whose startup_method is linear_fix_in_uc or integer_fix_in_uc
    # (``process_online_fix_in_uc``) so the realization solve PINS it
    # instead of re-deciding commitment (the realization solve inherits the
    # NA-tied units online of those processes, never reserve).  The units
    # online of linear and integer processes are not pinned: the realization
    # decides them again with the realized data.  An integer_fix_in_uc
    # process keeps its pin when the solve relaxes integers
    # (reduce_unit_detail_method integers*): the relaxed values are pinned.
    # The value is the reference forecast's (the first sorted forecast branch of
    # each anchor, the ``d_ref`` of non_anticipativity_online), on the
    # online tie cells only (``online_tie_cells``: the tied steps; nothing
    # when nothing is tied), re-keyed from the forecast branch
    # to its anchor period, the period name the realization solve
    # dispatches.  Unit-count space (NOT ×unitsize), rounded
    # (``snap_carried_online``: an integer column to whole units, a continuous
    # one near 0 to 0, so the (a2) outage cut ``> 0`` and the pin rows are the
    # same on every platform).  ``None`` on every other solve → byte-identical
    # baseline.
    fix_online_df = None
    _fix_in_uc = getattr(flex_data, "process_online_fix_in_uc", None) if flex_data is not None else None
    if (realized_dispatch_parents is not None
            and solve_name in realized_dispatch_parents
            and flex_data is not None
            and getattr(flex_data, "period_branch_full", None) is not None
            and flex_data.period_branch_full.height > 0
            and _fix_in_uc is not None and _fix_in_uc.height > 0
            and _has_online_var):
        from flextool.engine.model._non_anticipativity import online_tie_cells
        _cells = online_tie_cells(flex_data)
        _parts: list[pl.DataFrame] = []
        if _cells is not None:
            from flextool.engine.output._coarse_flows import var_is_integer, var_values, var_zero_tolerance

            _pinned = _fix_in_uc["p"].cast(pl.Utf8).to_list()
            for _vn in ("v_online_linear", "v_online_integer"):
                # Every fine step (a coarse process's units online hold through its coarse step).
                _all = var_values(sol, flex_data, _vn)
                if _all is None:
                    continue
                _fr = (snap_carried_online(_all, var_is_integer(sol, _vn), var_zero_tolerance(sol, _vn))
                       .with_columns(pl.col("p").cast(pl.Utf8), pl.col("d").cast(pl.Utf8), pl.col("t").cast(pl.Utf8))
                       .filter(pl.col("p").is_in(_pinned)))
                _re = (_fr.join(_cells, left_on=["d", "t"],
                                right_on=["d_ref", "t"], how="inner")
                       .select(pl.col("p").alias("process"),
                               pl.col("anchor").alias("period"),
                               pl.col("t").alias("step"),
                               "value"))
                if _re.height > 0:
                    _parts.append(_re)
        if _parts:
            fix_online_df = (pl.concat(_parts)
                             .sort(["process", "period", "step"]))

    # ---- online_history: the realized start-ups / shut-downs a minimum up / down time still covers ----
    # The next solve's first steps that continue from this solve's last realized step (its ``dt_ramp_carry``) owe
    # the obligations opened here (``_derived.apply_mintime_lookbacks``).  Each step keeps its ``hours_to_end``:
    # the hours from its start to the end of the last realized step (``online_step``: the committed end, inside a
    # forecasts-only fan the tied forecast ``d_ref``), walked back along this solve's ``dt_ramp_link`` over its
    # realized cells (``params._mintime.history_hours_to_end``), so it stops where the look-back stops (solve
    # start, a period that does not continue the previous one, a gap, a representative-block start).  The history
    # this solve started from is kept, its ``hours_to_end`` moved by the root's, when the walk reaches this solve's
    # ``dt_ramp_carry`` (the first step that continued from it).  Trimmed per process to ``hours_to_end <`` the
    # authored max min-up/down time (``flex_data.online_min_time``); a process without one is never stored.  The
    # start-ups / shut-downs are rounded: of a process with integer units online to whole units
    # (``round_whole_unit_events``), the others near 0 to 0 (``snap_carried_online``), so solver noise opens no
    # carried obligation.
    online_history_df = None
    last_pairs_on = None
    rd_path_on = sd / "realized_dispatch.csv"
    if _provider_has(provider, "solve_data/realized_dispatch", rd_path_on):
        try:
            last_pairs_on = _provider_read(
                provider, "solve_data/realized_dispatch", rd_path_on)
        except pl.exceptions.NoDataError:
            last_pairs_on = None
    if (last_pairs_on is None and flex_data is not None
            and getattr(flex_data, "realized_dispatch", None) is not None):
        last_pairs_on = flex_data.realized_dispatch
    _has_su_sd = any(
        v in sol._vars for v in
        ("v_startup_linear", "v_startup_integer",
         "v_shutdown_linear", "v_shutdown_integer"))
    online_min_time_df = (getattr(flex_data, "online_min_time", None)
                          if flex_data is not None else None)
    # The realized cells under this solve's step names: the realized dispatch and the committed cells of a
    # forecasts-only fan (the tied forecast ``d_ref``; the period label has no steps of its own).
    _hist_cells = []
    if last_pairs_on is not None and last_pairs_on.height > 0:
        _cols = last_pairs_on.columns
        _dc = "period" if "period" in _cols else "d"
        _tc = "step" if "step" in _cols else "t"
        _hist_cells.append(last_pairs_on.select(pl.col(_dc).cast(pl.Utf8).alias("d"),
                                                pl.col(_tc).cast(pl.Utf8).alias("t")))
    if fan_online is not None:
        _hist_cells.append(fan_online.select(pl.col("d_ref").alias("d"), "t"))
    if (_hist_cells
            and online_step is not None
            and _has_su_sd
            and online_min_time_df is not None
            and online_min_time_df.height > 0):
        from flextool.engine.preprocess.params._mintime import (
            check_history_continues_carried_step, history_hours_to_end)
        lmap = online_min_time_df.select(pl.col("p").cast(pl.Utf8), pl.col("value").cast(pl.Float64).alias("_L"))
        cells = pl.concat(_hist_cells).unique(maintain_order=True)
        _sdf = _step_duration_frame(sd, flex_data, provider=provider)
        dur = None
        if _sdf is not None and _sdf.height > 0:
            _d2 = "period" if "period" in _sdf.columns else "d"
            _t2 = "time" if "time" in _sdf.columns else ("step" if "step" in _sdf.columns else "t")
            _v2 = "value" if "value" in _sdf.columns else "p_step_duration"
            dur = (_sdf.select(pl.col(_d2).cast(pl.Utf8).alias("d"), pl.col(_t2).cast(pl.Utf8).alias("t"),
                               pl.col(_v2).cast(pl.Float64).alias("dur"))
                   .unique(subset=["d", "t"], keep="first"))
        cells = (cells.join(dur, on=["d", "t"], how="left", maintain_order="left") if dur is not None
                 else cells.with_columns(dur=pl.lit(None, dtype=pl.Float64)))
        cells = cells.with_columns(pl.col("dur").fill_null(1.0))
        h2e = history_hours_to_end(cells, getattr(flex_data, "dt_ramp_link", None), online_step,
                                   float(lmap["_L"].max()))
        hist_now = None
        if h2e is not None:
            from flextool.engine.output._coarse_flows import var_is_integer, var_values, var_zero_tolerance

            # Start-ups / shut-downs are continuous columns.  Of a process whose units online are integer they are
            # rounded to whole units together below (round_whole_unit_events); the others go to 0 near 0.
            whole = var_is_integer(sol, "v_online_integer")
            whole_procs = (set(flex_data.process_online_integer["p"].cast(pl.Utf8).to_list())
                           if whole and getattr(flex_data, "process_online_integer", None) is not None else set())

            def _pull_online_metric(vnames):
                # Every fine step: a coarse process's start-up / shut-down stays at its coarse step's first fine
                # step (the event at the step's start, 0 at its other steps; history_hours_to_end).
                frames = []
                for vn in vnames:
                    _all = var_values(sol, flex_data, vn)
                    if _all is None:
                        continue
                    _fr = (_all.with_columns(pl.col("d").cast(pl.Utf8).alias("__d"),
                                             pl.col("t").cast(pl.Utf8).alias("__t"))
                           .join(h2e, left_on=["__d", "__t"], right_on=["d", "t"], how="inner"))
                    if _fr.height == 0:
                        continue
                    _fr = _fr.select(pl.col("p").cast(pl.Utf8), pl.col("__d").alias("d"),
                                     pl.col("__t").alias("t"), pl.col("value"))
                    if not (whole and vn.endswith("_integer")):
                        _fr = snap_carried_online(_fr, False, var_zero_tolerance(sol, vn))
                    frames.append(_fr)
                if not frames:
                    return None
                return pl.concat(frames).group_by(["p", "d", "t"]).agg(pl.col("value").sum())

            su = _pull_online_metric(("v_startup_linear", "v_startup_integer"))
            sh = _pull_online_metric(("v_shutdown_linear", "v_shutdown_integer"))
            if su is not None or sh is not None:
                hist_now = pl.concat([f.select("p", "d", "t") for f in (su, sh) if f is not None]).unique()
                for name, frame in (("startup", su), ("shutdown", sh)):
                    hist_now = (hist_now.join(frame.rename({"value": name}), on=["p", "d", "t"], how="left")
                                if frame is not None else hist_now.with_columns(pl.lit(0.0).alias(name)))
                hist_now = (hist_now
                            .join(cells.select("d", "t", pl.col("dur").alias("duration")), on=["d", "t"],
                                  how="left")
                            .join(h2e, on=["d", "t"], how="inner")
                            .with_columns(pl.col("startup").cast(pl.Float64).fill_null(0.0),
                                          pl.col("shutdown").cast(pl.Float64).fill_null(0.0),
                                          pl.col("duration").cast(pl.Float64).fill_null(1.0),
                                          pl.col("hours_to_end").cast(pl.Float64))
                            .select("p", "d", "t", "startup", "shutdown", "duration", "hours_to_end"))
                if whole_procs:
                    rounded = round_whole_unit_events(hist_now)
                    is_whole = pl.col("p").is_in(sorted(whole_procs))
                    hist_now = hist_now.with_columns(
                        pl.when(is_whole).then(rounded["startup"]).otherwise(pl.col("startup")).alias("startup"),
                        pl.when(is_whole).then(rounded["shutdown"]).otherwise(pl.col("shutdown")).alias("shutdown"))
        if hist_now is not None:
            # The history this solve started from (the routed ``handoff/online_history`` its LP read,
            # solve._carry_routing), continued through the first step of this solve that continued from it.
            from flextool.engine.preprocess import _provider_keys as _K
            from flextool.engine.preprocess._provider_translators import read_handoff_frame as _read_hf
            parts = [hist_now]
            prior_hist = _read_hf(provider, _K.HANDOFF_ONLINE_HISTORY)
            ramp_carry = getattr(flex_data, "dt_ramp_carry", None)
            if (prior_hist is not None and prior_hist.height > 0
                    and ramp_carry is not None and ramp_carry.height > 0):
                root = h2e.join(ramp_carry.select(pl.col("d").cast(pl.Utf8), pl.col("t").cast(pl.Utf8)).unique(),
                                on=["d", "t"], how="semi")
                if root.height > 1:
                    raise ValueError(f"online_history: the realized chain reaches several carried steps "
                                     f"{root.rows()}")
                if root.height == 1:
                    prior = prior_hist.rename({"process": "p", "period": "d", "step": "t"}).select(
                        pl.col("p").cast(pl.Utf8), pl.col("d").cast(pl.Utf8), pl.col("t").cast(pl.Utf8),
                        pl.col("startup").cast(pl.Float64), pl.col("shutdown").cast(pl.Float64),
                        pl.col("duration").cast(pl.Float64), pl.col("hours_to_end").cast(pl.Float64))
                    carried = ramp_carry.select(pl.col("d_previous").cast(pl.Utf8),
                                                pl.col("t_previous").cast(pl.Utf8)).unique()
                    check_history_continues_carried_step(prior, carried.row(0),
                                                         getattr(flex_data, "period_branch_full", None))
                    parts.append(prior.with_columns(pl.col("hours_to_end") + float(root["hours_to_end"][0])))
            combined = (pl.concat(parts).unique(subset=["p", "d", "t"], keep="first", maintain_order=True)
                        .join(lmap, on="p", how="inner")
                        .filter(pl.col("hours_to_end") < pl.col("_L"))
                        .drop("_L"))
            if combined.height > 0:
                online_history_df = (combined.rename({"p": "process", "d": "period", "t": "step"})
                                     .sort(["process", "hours_to_end"], descending=[False, True]))

    # ---- fix_storage_quantity: v_state at fix_quantity timesteps × unitsize ----
    # Mirrors flextool's ``write_fix_storage_quantity`` (handoff_writers.py
    # :380).  Restricted to nodes whose storage_nested_fix_method is
    # ``fix_quantity`` and (period, step) in fix_storage_timesteps.csv.
    # The fix_price (dual-based) and fix_usage (flow-based) variants are
    # left unfilled here — they require nodeBalance_eq dual extraction
    # / per-arc flow summation which is significantly more involved than
    # the quantity case and isn't exercised by the multi_invest fixture.
    # Phase 4.1l — the narrow per-metric carrier is built directly from
    # v_state × unitsize; the legacy wide intermediate frame has been
    # retired (its last consumer, _run_model._fan_out_fix_storage,
    # was deleted in 4.1l).
    fix_storage_quantity_df = None
    fq_nodes: set[str] = set()
    # Gap F final — prefer ``flex_data.node__storage_nested_fix_method``.
    nsfm_df = None
    if flex_data is not None and getattr(
            flex_data, "node__storage_nested_fix_method", None) is not None:
        nsfm_df = flex_data.node__storage_nested_fix_method
    else:
        nsfm_path = sd / "node__storage_nested_fix_method.csv"
        if _provider_has(provider, "solve_data/node__storage_nested_fix_method", nsfm_path):
            try:
                nsfm_df = _provider_read(provider, "solve_data/node__storage_nested_fix_method", nsfm_path)
            except pl.exceptions.NoDataError:
                nsfm_df = None
    if nsfm_df is not None and nsfm_df.height > 0 and "method" in nsfm_df.columns:
        fq_nodes = set(
            nsfm_df.filter(pl.col("method") == "fix_quantity")["node"]
            .cast(pl.Utf8).to_list()
        )
    # Gap F final — prefer the in-memory fix_storage_timesteps carriers:
    # ``parent_handoff.fix_storage_timesteps`` deposits the (period, step)
    # set for child solves; otherwise this solve emits its own set via
    # ``emit_fix_storage_timesteps`` and we read it back from the Provider.
    fs_steps_df = None
    if parent_handoff is not None and getattr(
            parent_handoff, "fix_storage_timesteps", None) is not None:
        fs_steps_df = parent_handoff.fix_storage_timesteps
    else:
        fix_steps_path = sd / "fix_storage_timesteps.csv"
        if _provider_has(provider, "solve_data/fix_storage_timesteps", fix_steps_path):
            try:
                fs_steps_df = _provider_read(provider, "solve_data/fix_storage_timesteps", fix_steps_path)
            except pl.exceptions.NoDataError:
                fs_steps_df = None
    if (fq_nodes
            and fs_steps_df is not None
            and "v_state" in sol._vars):
        if fs_steps_df.height > 0 and {"period", "step"}.issubset(
                fs_steps_df.columns):
            fs_steps = (fs_steps_df
                .select(alias_to_axis("period", "d"),
                         alias_to_axis("step", "t"))
                .unique())
            v_state = sol.value("v_state")
            if v_state is not None and v_state.height > 0:
                fq_rows = (v_state
                    .filter(pl.col("n").is_in(list(fq_nodes)))
                    .join(fs_steps, on=["d", "t"], how="inner"))
                if fq_rows.height > 0:
                    # Multiply by unitsize (per-node) and emit the
                    # canonical narrow schema
                    # ``[node, period, step, p_fix_storage_quantity]``.
                    us_rows = [(n, unitsize.get(n, 1.0))
                                 for n in sorted(fq_nodes)]
                    us_df = pl.DataFrame(
                        us_rows, schema={"n": v_state.schema["n"], "us": pl.Float64}, orient="row")
                    fq_rows = (fq_rows
                        .join(us_df, on="n", how="inner")
                        .with_columns(
                            p_fix_storage_quantity=pl.col("value") * pl.col("us"),
                        )
                        .select(
                            alias_to_axis("n", "node"),
                            alias_to_axis("d", "period"),
                            alias_to_axis("t", "step"),
                            pl.col("p_fix_storage_quantity"),
                        )
                        .sort("node", "period", "step"))  # the joins' row order varies run to run
                    if fq_rows.height > 0:
                        fix_storage_quantity_df = fq_rows

    # ---- cumulative_co2: per-(group, period) running total ----
    # Gap F final close-out — native compute via
    # ``_co2_accumulators.compute_co2_rolling_accumulator`` when
    # ``flex_data`` + ``sol`` are available (cascade path).  The disk-read arm for legacy / test
    # callers that only pass ``sol`` was dropped, c24 S4-6: nothing ever writes
    # ``solve_data/co2_cum_realized_tonnes`` (no producer puts that provider key, and
    # ``dump_csvs`` does not emit the file either), so it was a no-op fallback.  When
    # ``flex_data`` or ``sol`` is absent, or the native compute returns no rows, the
    # ``prior_handoff`` carry-forward set above is the only value.
    cumulative_co2_df = None
    if prior_handoff is not None and prior_handoff.cumulative_co2 is not None:
        cumulative_co2_df = prior_handoff.cumulative_co2
    if flex_data is not None and sol is not None:
        from flextool.engine.preprocess.emit._co2_accumulators import (
            compute_co2_rolling_accumulator,
        )
        prior_df = (prior_handoff.cumulative_co2
                    if prior_handoff is not None else None)
        native_co2 = compute_co2_rolling_accumulator(
            flex_data, sol, work_folder=work_folder,
            prior_cumulative_co2=prior_df,
            provider=provider,
            is_dispatch_sibling_parent=_is_dispatch_sibling_parent,
        )
        if native_co2.height > 0:
            # ``represented``: the tonnes in co2_max_total's units (× years_represented / period_share of
            # the solve that realized them); null where that is unknown, which co2_max_total refuses.
            cumulative_co2_df = (native_co2
                .with_columns(
                    value=pl.col("p_co2_cum_realized_tonnes")
                            .cast(pl.Float64, strict=False)
                            .fill_null(0.0),
                    represented=pl.col("p_co2_cum_realized_represented").cast(pl.Float64, strict=False))
                .select("group", "period", "value", "represented"))

    # ---- cumulative_commodity: per-(commodity, tier, period) running mwh ----
    # Δ.11 — derive from sol when v_trade is in the LP and prior_handoff
    # carries finite-tier ladder commodities.  Algorithm:
    #
    #   this_roll_mwh[c, i, d] = Σ_n v_trade[c, n, d, i] × unitsize[c]
    #                              × (realized_hours[d] / horizon_hours[d])
    #
    # Cumulative across solves: prior_mwh + this_roll_mwh, restricted to
    # finite ladder tiers (``ci_ladder_cumulative.csv``) — non-finite
    # tiers don't need the carry-over since their cap is unbounded.
    #
    # Falls back to the workdir CSV when v_trade isn't in the solution
    # (the LP didn't expose the variable for this fixture) — preserves
    # the legacy propagation path.
    cumulative_commodity_df = _extract_cumulative_commodity(
        sol, sd, prior_handoff=prior_handoff, flex_data=flex_data,
        provider=provider, is_dispatch_sibling_parent=_is_dispatch_sibling_parent)
    if cumulative_commodity_df is None:
        # Phase 4 (Gap F) — disk fallback retired.  When
        # ``_extract_cumulative_commodity`` returns None this solve had no
        # this-roll increment to add (no ``v_trade`` in the LP, or no
        # finite ladder tiers), so the carrier reduces to whatever the
        # prior solve deposited.  The legacy
        # ``solve_data/commodity_ladder_cumulative.csv`` fallback is
        # unreachable in the cascade path: ``write_ladder_rolling_accumulators``
        # writes ``ladder_cum_realized_mwh.csv`` (different name), and
        # ``commodity_ladder_cumulative.csv`` lives under ``input/`` not
        # ``solve_data/`` in production.
        if (prior_handoff is not None
                and prior_handoff.cumulative_commodity is not None):
            cumulative_commodity_df = prior_handoff.cumulative_commodity

    # ---- cum_sim_hours: per-period running sim-hour total ----
    # Δ.11 — derive from the workdir's ``p_step_duration.csv`` +
    # ``realized_dispatch.csv`` (uniform-split assumption):
    #
    #   this_roll_hrs[d] = Σ_t step_duration[d, t] for (d, t) ∈ realized_dispatch
    #
    # Cumulative: prior_hrs + this_roll_hrs.  ``v_trade`` not required —
    # the carrier exists for every chained fixture even when the ladder
    # itself is inactive (CO2-cap normalisation also consumes it).
    cum_sim_hours_df = _extract_cum_sim_hours(
        sd, prior_handoff=prior_handoff, flex_data=flex_data,
        provider=provider, is_dispatch_sibling_parent=_is_dispatch_sibling_parent)
    if cum_sim_hours_df is None:
        # Disk fallback retired.  ``_extract_cum_sim_hours`` returns
        # None only when this solve has zero realized timesteps AND
        # ``prior_handoff`` carries no prior hours.  Propagate the
        # prior carrier as-is when present — the post-solve writer
        # hasn't run yet, so re-reading disk would just round-trip
        # data we already hold in memory.
        if (prior_handoff is not None
                and prior_handoff.cum_sim_hours is not None):
            cum_sim_hours_df = prior_handoff.cum_sim_hours

    # Δ.1 — ``periods_already_emitted`` extraction removed.  The carrier
    # moved to ``_writer.OutputWriterState`` (writer-side state).
    # ``solve_data/period_capacity.csv`` is unchanged (handoff_writers
    # still bumps it post-solve).

    # Phase 4.1h / 4.1l — parent's narrow fix_storage_usage field passes
    # straight through to this solve's narrow field when nested.  The
    # fix_storage_quantity carrier is produced above from v_state ×
    # unitsize; deferred-B Phase B2 (below) produces fix_storage_price
    # from ``nodeBalance_eq`` row duals at this solve's fix_price
    # storage timesteps.  The parent's narrow price carrier remains the
    # fallback when this solve declares no fix_price-method nodes.
    fix_storage_price_df = None
    fix_storage_usage_df = None
    if parent_handoff is not None:
        if parent_handoff.fix_storage_price is not None:
            fix_storage_price_df = parent_handoff.fix_storage_price
        if parent_handoff.fix_storage_usage is not None:
            fix_storage_usage_df = parent_handoff.fix_storage_usage

    # ── deferred-B Phase B2: fix_storage_price from the node balance duals ─
    # For each storage node n with node__storage_nested_fix_method[n] == 'fix_price' and each (period, step) in
    # fix_storage_timesteps, the price handed to the child is this solve's node price there (price_t, the same
    # definition, flextool.engine.output._prices):
    #
    #   p_fix_storage_price[n,d,t] = σ_row · dual[row(n,d,t)] / s / (tw[d,t] · inflation_op[d] / period_share[d]
    #                                · branch_weight[d,t])
    #     row = nodeBalance_eq[n,d,t] (σ +1) | nodeBalanceCoarse_eq[n,d,b ∋ t] (σ +1) | nodeBalanceBlock_eq (σ +1)
    #     s = scale_the_objective of this solve (the ``scale_the_objective`` argument; None = 1, as build_flextool)
    #
    # CUR/MWh, as the child's storage_reference_price term (_obj_storage_reference_price) uses it with its own
    # weights.  polar_high's ``constraint_dual`` is the HiGHS shadow price ∂obj/∂rhs, so a storage node whose
    # energy is worth +C has dual +C·s·w and price +C.  A step with zero weight has no price (dropped).  This
    # overrides the parent passthrough above whenever this solve has a fix_price node with a price at a fix step.
    fp_nodes: set[str] = set()
    if nsfm_df is not None and nsfm_df.height > 0 and "method" in nsfm_df.columns:
        fp_nodes = set(
            nsfm_df.filter(pl.col("method") == "fix_price")["node"]
            .cast(pl.Utf8).to_list()
        )
    if (fp_nodes
            and fs_steps_df is not None
            and fs_steps_df.height > 0
            and {"period", "step"}.issubset(fs_steps_df.columns)
            and sol is not None
            and flex_data is not None):
        from flextool.engine.output._prices import node_prices

        def _node_step_duals(family: str, step_col: str) -> "pl.DataFrame | None":
            # Row names are ``"<family>[n,d,t]"``; ``constraint_dual`` exposes the comma-joined dims as one
            # ``key`` column.  Split into (n, d, <step_col>), all Utf8.
            try:
                raw = sol.constraint_dual(family)
            except KeyError:
                return None
            if raw is None or raw.height == 0 or "key" not in raw.columns:
                return None
            raw = raw.with_columns(pl.col("key").str.split_exact(",", 2).alias("_parts"))
            return raw.select(
                n=pl.col("_parts").struct.field("field_0").cast(pl.Utf8),
                d=pl.col("_parts").struct.field("field_1").cast(pl.Utf8),
                **{step_col: pl.col("_parts").struct.field("field_2").cast(pl.Utf8)},
                dual=pl.col("dual"),
            )

        duals = {"nodeBalance_eq": _node_step_duals("nodeBalance_eq", "t")}
        coarse_steps = getattr(flex_data, "node_coarse_step", None)
        if coarse_steps is not None and coarse_steps.height > 0:
            duals["nodeBalanceCoarse_eq"] = _node_step_duals("nodeBalanceCoarse_eq", "b_first")
        block_time = getattr(flex_data, "period_block_time", None)
        if block_time is not None and block_time.height > 0:
            duals["nodeBalanceBlock_eq"] = _node_step_duals("nodeBalanceBlock_eq", "b_first")
        s_obj = 1.0 if scale_the_objective is None else float(scale_the_objective)
        prices = node_prices(duals, flex_data, inv_scale=1.0 / s_obj)
        fs_steps_fp = (fs_steps_df
                       .select(pl.col("period").cast(pl.Utf8).alias("d"), pl.col("step").cast(pl.Utf8).alias("t"))
                       .unique())
        picked = (prices
                  .filter(pl.col("n").is_in(sorted(fp_nodes)) & pl.col("price").is_not_null())
                  .join(fs_steps_fp, on=["d", "t"], how="inner")
                  .sort("n", "d", "t"))
        if picked.height > 0:
            fix_storage_price_df = picked.select(
                alias_to_axis("n", "node"),
                alias_to_axis("d", "period"),
                alias_to_axis("t", "step"),
                pl.col("price").alias("p_fix_storage_price"),
            )

    # ── deferred-B Phase B3: fix_storage_usage from v_flow primals ──────
    # For each storage node n with
    #   node__storage_nested_fix_method[n] == 'fix_usage'
    # and each (period, step) in fix_storage_timesteps, sum v_flow primals
    # of arcs touching n, weighted by p_entity_unitsize[process] and
    # p_step_duration[d, t].  The formula mirrors the LP constraint
    # ``node_storage_usage_fix_le`` (model._eq_storage_usage_fix) so the
    # producer ↔ constraint round-trip closes exactly:
    #
    #   p_fix_storage_usage[n, d, t]
    #     = ( - Σ_{(p, n, sink) ∈ pss}                                # n-as-sink (raw)
    #             v_flow[p, n, sink, d, t] * unitsize[p]
    #         + Σ_{(p, source, n) ∈ pss_eff, p_slope present}         # n-as-source (eff, slope)
    #             v_flow[p, source, n, d, t] * unitsize[p] * slope[p, d, t]
    #         + Σ_{(p, source, n) ∈ pss_eff ∩ min_load_eff}           # n-as-source (eff, min_load_eff section)
    #             v_online[p, d, t] * section[p, d, t] * unitsize[p]
    #         + Σ_{(p, source, n) ∈ pss_noEff}                        # n-as-source (noEff, raw)
    #             v_flow[p, source, n, d, t] * unitsize[p]
    #       ) × step_duration[d, t]
    #
    # Diverges from the legacy producer (handoff_writers.py:694-766)
    # which always used the simplified formula (sink and source both
    # raw, no slope/section).  The new engine prefers round-trip
    # consistency with its own constraint over legacy fidelity — see
    # specs/feature_fixes.md §3 and the user decision recorded there
    # (option a).  Pairing fix_usage with min_load_efficiency and
    # non-unit slope is now supported.  Per-process flow-coefficient
    # ratio is still deferred (matches model._eq_storage_usage_fix).
    #
    # Overrides the parent passthrough whenever this solve has any
    # fix_usage node × fix-step pair with extractable flows.
    fu_nodes: set[str] = set()
    if nsfm_df is not None and nsfm_df.height > 0 and "method" in nsfm_df.columns:
        fu_nodes = set(
            nsfm_df.filter(pl.col("method") == "fix_usage")["node"]
            .cast(pl.Utf8).to_list()
        )
    weighted = _storage_usage_mwh(sol, flex_data, fu_nodes, fs_steps_df)
    if weighted is not None:
        fix_storage_usage_df = weighted

    # ---- realised_storage_usage: the energy this solve drew in its realised steps from the nodes an upper-level
    # solve caps by fix_usage (not a carrier: the cascade sums it per solve instance, solve._run_model) ----
    realised_storage_usage_df = None
    n_fu = getattr(flex_data, "n_fix_storage_usage", None) if flex_data is not None else None
    rd = getattr(flex_data, "realized_dispatch", None) if flex_data is not None else None
    if n_fu is not None and n_fu.height > 0 and rd is not None and rd.height > 0 and {"period", "step"} <= set(
            rd.columns):
        used = _storage_usage_mwh(sol, flex_data, set(n_fu["n"].cast(pl.Utf8).to_list()),
                                  rd.select(pl.col("period").cast(pl.Utf8), pl.col("step").cast(pl.Utf8)))
        if used is not None and used.height > 0:
            realised_storage_usage_df = (used.select(pl.col("node").cast(pl.Utf8), pl.col("period").cast(pl.Utf8),
                                                     pl.col("p_fix_storage_usage").alias("value"))
                                         .group_by("node", "period").agg(pl.col("value").sum())
                                         .sort("node", "period"))

    # ---- cycle_start_state / storage_level: the start level of each storage node and the planned level of the
    # nodes handed a nested fix (not carriers: the cascade reads them, solve._run_model) ----
    cycle_start_state_df, storage_level_df, scope_start_state_df = _storage_start_and_level(
        sol, flex_data, nsfm_df, unitsize)

    return SolveHandoff(
        realized_commission=pl.DataFrame(
            commission_rows, schema=["entity", "period", "commission_period", "value"], orient="row",
        ) if commission_rows else None,
        divest_cumulative=pl.DataFrame(
            div_rows, schema=["entity", "value"], orient="row",
        ) if div_rows else None,
        realized_cohort=realized_cohort_df,
        realized_retire=realized_retire_df,
        roll_end_state=roll_end_state_df,
        upward_roll_end_state=upward_roll_end_state_df,
        roll_end_online=roll_end_online_df,
        roll_end_flow=roll_end_flow_df,
        roll_end_step=roll_end_step_df,
        online_history=online_history_df,
        fix_online=fix_online_df,
        fix_storage_quantity=fix_storage_quantity_df,
        fix_storage_price=fix_storage_price_df,
        fix_storage_usage=fix_storage_usage_df,
        cumulative_co2=cumulative_co2_df,
        cumulative_commodity=cumulative_commodity_df,
        cum_sim_hours=cum_sim_hours_df,
        cumulative_flowgroup=_fg.flowgroup_handoff(flex_data, sol, prior_handoff, _is_dispatch_sibling_parent),
        cycle_start_state=cycle_start_state_df,
        storage_level=storage_level_df,
        scope_start_state=scope_start_state_df,
        realised_storage_usage=realised_storage_usage_df,
    )
