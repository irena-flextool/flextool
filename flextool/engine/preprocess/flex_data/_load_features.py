"""Loaders of the optional features: indirect units, user constraints, investment, profiles, costs,
cumulative investment, stochastics and the period sets of a sub-solve."""

from __future__ import annotations

from pathlib import Path
import polars as pl
from polar_high import Param
from flextool.engine.input._axis_enums import cast_dim, rename_to_axis
from flextool.engine.preprocess.flex_data._readers import _provider_has, _provider_read, _read_wide_per_entity


def _load_indirect(sd: Path, pss: pl.DataFrame | None, dt: pl.DataFrame,
                    inp: Path | None = None,
                    *,
                    provider: "object | None" = None):
    if pss is None:
        return (None, None, None, None, None, None)
    p = sd / "process__method_indirect.csv"
    if not _provider_has(provider, "solve_data/process__method_indirect", p):
        return (None, None, None, None, None, None)
    raw = _provider_read(provider, "solve_data/process__method_indirect", p).pipe(rename_to_axis, {"process":"p"})
    if raw.height == 0:
        return (None, None, None, None, None, None)
    indirect = raw.select("p").unique()
    # Cross-axis compares: sink/source (e-axis) vs p (p-axis).  Per
    # contract p ⊂ e; up-cast p to e so the compare runs in Enum
    # natively without Utf8 materialisation.
    inputs  = pss.filter((pl.col("p").is_in(indirect["p"])) & (pl.col("sink") == cast_dim(pl.col("p"), None, "e")))
    outputs = pss.filter((pl.col("p").is_in(indirect["p"])) & (pl.col("source") == cast_dim(pl.col("p"), None, "e")))

    # The .mod's conversion_indirect LHS multiplies each source-side
    # v_flow by ``p_process_source_conversion_flow_coeff[p, source]`` and
    # the RHS sum by ``p_process_sink_conversion_flow_coeff[p, sink]``
    # (.mod:2557-2580).  Most scenarios have all coefs = 1 (the default);
    # a zero coefficient effectively drops that flow from the conversion
    # equation; the ``coal_chp_extraction`` scenario uses non-default
    # sink coefficients ({heat: 0.2, west: 2.0}) to encode the iso-fuel
    # relationship via the source-side capacity bound.  Build optional
    # Params restricted to the indirect inputs / outputs sets — only
    # when non-default coefficients are present — and let model/core/
    # multiply them into the conversion equation.  Zero-coefficient
    # rows are still anti-joined out (so they don't survive into
    # ``inputs`` / ``outputs``).
    p_source_flow_coef = None
    p_sink_flow_coef = None
    if inp is not None:
        src_path = inp / "p_process_source_conversion_flow_coeff.csv"
        if _provider_has(provider,
                          "input/p_process_source_conversion_flow_coeff",
                          src_path):
            srcdf = _provider_read(provider,
                                     "input/p_process_source_conversion_flow_coeff",
                                     src_path)
            if srcdf.height > 0 and "p_process_source_conversion_flow_coeff" in srcdf.columns:
                src_long = (srcdf
                    .pipe(rename_to_axis, {"process": "p", "source": "source",
                             "p_process_source_conversion_flow_coeff": "coef"})
                    .with_columns(pl.col("coef").cast(pl.Float64, strict=False))
                    .select("p", "source", "coef"))
                zero_src = src_long.filter(pl.col("coef") == 0.0).select("p", "source")
                if zero_src.height > 0:
                    inputs = inputs.join(zero_src, on=["p", "source"], how="anti")
                # If any non-default, non-zero coefficient applies to a
                # surviving (p, source) row, build a Param covering ALL
                # surviving (p, source) pairs (defaulted to 1.0 where not
                # listed) so the inner-join in v_flow * Param doesn't drop
                # rows.  Zero-coef rows have already been removed.
                nonzero_nondefault = src_long.filter(
                    (pl.col("coef") != 0.0) & (pl.col("coef") != 1.0))
                if nonzero_nondefault.height > 0:
                    in_pair = inputs.select("p", "source").unique()
                    if in_pair.height > 0:
                        merged = (in_pair.join(src_long, on=["p", "source"], how="left")
                                          .with_columns(pl.col("coef").fill_null(1.0))
                                          .rename({"coef": "value"}))
                        p_source_flow_coef = Param(("p", "source"), merged)
        sink_path = inp / "p_process_sink_conversion_flow_coeff.csv"
        if _provider_has(provider,
                          "input/p_process_sink_conversion_flow_coeff",
                          sink_path):
            sinkdf = _provider_read(provider,
                                      "input/p_process_sink_conversion_flow_coeff",
                                      sink_path)
            if sinkdf.height > 0 and "p_process_sink_conversion_flow_coeff" in sinkdf.columns:
                sink_long = (sinkdf
                    .pipe(rename_to_axis, {"process": "p", "sink": "sink",
                             "p_process_sink_conversion_flow_coeff": "coef"})
                    .with_columns(pl.col("coef").cast(pl.Float64, strict=False))
                    .select("p", "sink", "coef"))
                zero_sink = sink_long.filter(pl.col("coef") == 0.0).select("p", "sink")
                if zero_sink.height > 0:
                    outputs = outputs.join(zero_sink, on=["p", "sink"], how="anti")
                nonzero_nondefault = sink_long.filter(
                    (pl.col("coef") != 0.0) & (pl.col("coef") != 1.0))
                if nonzero_nondefault.height > 0:
                    out_pair = outputs.select("p", "sink").unique()
                    if out_pair.height > 0:
                        merged = (out_pair.join(sink_long, on=["p", "sink"], how="left")
                                           .with_columns(pl.col("coef").fill_null(1.0))
                                           .rename({"coef": "value"}))
                        p_sink_flow_coef = Param(("p", "sink"), merged)

    # Phase E.3: ``process_indirect_dt`` no longer materialised;
    # consumers call ``_pdt_join.compute_process_indirect_dt`` on demand.
    return (indirect, inputs, outputs, None,
            p_source_flow_coef, p_sink_flow_coef)


def _load_user_constraints(inp: Path, pss: pl.DataFrame | None, dt: pl.DataFrame,
                            *,
                            provider: "object | None" = None):
    """Returns 12 items:
    flow_cstr_idx, flow_cstr_coef, constraint_constant, cdt_eq, cdt_le, cdt_ge,
    n_inv_cstr_coef, p_inv_cstr_coef, n_state_cstr_coef,
    n_prebuilt_cstr_coef, p_prebuilt_cstr_coef, has_user_cstr.

    The ``*_inv_cstr_coef`` Params carry
    ``p_<entity>_constraint_invested_capacity_coeff`` data;
    ``n_state_cstr_coef`` carries ``p_node_constraint_state_coeff``
    (user-cstr v_state contribution); the ``*_prebuilt_cstr_coef`` Params
    carry ``p_<entity>_constraint_prebuilt_capacity_coeff``
    (existing + prior-period invest)."""
    if pss is None:
        return [None]*12
    cs_path = inp / "constraint__sense.csv"
    if not _provider_has(provider, "input/constraint__sense", cs_path):
        return [None]*12
    cs = _provider_read(provider, "input/constraint__sense", cs_path).pipe(rename_to_axis, {"constraint":"cn"})
    if cs.height == 0:
        return [None]*12
    coef_path = inp / "p_process_node_constraint_flow_coeff.csv"
    flow_cstr_idx = None
    # Δ.12-drop: ``p_flow_constraint_coef`` produced authoritatively by
    # ``apply_derived_b.p_flow_constraint_coef_from_source`` when pss is
    # non-empty.  We retain the CSV read because ``flow_constraint_idx``
    # (the index frame, not a Param) is still needed by downstream
    # constraint emission and isn't produced by an override-chain helper.
    # The constraint axis column is ``cn`` (not ``c``) to disambiguate
    # from the commodity axis — see the c_collision review note in
    # ``schemas/flextool_axis_contract.json``.
    flow_cstr_coef = None
    if _provider_has(provider,
                      "input/p_process_node_constraint_flow_coeff",
                      coef_path):
        coef_long = (_provider_read(
                provider,
                "input/p_process_node_constraint_flow_coeff",
                coef_path)
            .pipe(rename_to_axis, {"process":"p","node":"n","constraint":"cn",
                     "p_process_node_constraint_flow_coeff":"coef"})
            .with_columns(pl.col("coef").cast(pl.Float64, strict=False))
            .select("p","n","cn","coef"))
        # Cross-axis join: pss carries ``source``/``sink`` (e-axis) joined
        # against ``coef_long.n`` (n-axis).  Per the contract n ⊂ e, so
        # up-cast coef_long.n to e-Enum once; the join then composes
        # natively in Enum without any Utf8 materialisation at the join
        # boundary.
        coef_long_e = coef_long.with_columns(
            cast_dim(pl.col("n"), None, "e"))
        src_match = (pss.join(
                            coef_long_e,
                            left_on=["p","source"], right_on=["p","n"],
                            how="inner")
                       .select("p", "source", "sink", "cn", "coef"))
        sink_match = (pss.join(
                            coef_long_e,
                            left_on=["p","sink"], right_on=["p","n"],
                            how="inner")
                        .select("p", "source", "sink", "cn", "coef"))
        if src_match.height + sink_match.height > 0:
            joined = (pl.concat([src_match, sink_match], how="vertical")
                        .group_by(["p","source","sink","cn"])
                        .agg(pl.col("coef").sum())
                        .sort("p", "source", "sink", "cn"))  # group_by order varies run to run
            flow_cstr_idx  = joined.select("p","source","sink","cn")
    # Δ.12-drop: ``p_node_constraint_invested_capacity_coeff`` /
    # ``p_process_constraint_invested_capacity_coeff`` /
    # ``p_node_constraint_state_coeff`` /
    # ``p_node_constraint_prebuilt_capacity_coeff`` /
    # ``p_process_constraint_prebuilt_capacity_coeff`` and
    # ``p_constraint_constant`` produced authoritatively by
    # ``apply_direct_params``.  Seeds dropped.
    n_inv_cstr_coef = None
    p_inv_cstr_coef = None
    n_state_cstr_coef = None
    n_prebuilt_cstr_coef = None
    p_prebuilt_cstr_coef = None
    constraint_constant = None
    cdt_eq = cdt_le = cdt_ge = None
    for s, slot in [("equal","eq"), ("less_than","le"), ("greater_than","ge")]:
        cs_s = cs.filter(pl.col("sense")==s).select("cn")
        if cs_s.height > 0:
            axes = cs_s.join(dt, how="cross")
            if   slot=="eq":
                cdt_eq = axes
            elif slot=="le":
                cdt_le = axes
            else:
                cdt_ge = axes
    return (flow_cstr_idx, flow_cstr_coef, constraint_constant,
            cdt_eq, cdt_le, cdt_ge, n_inv_cstr_coef, p_inv_cstr_coef,
            n_state_cstr_coef, n_prebuilt_cstr_coef, p_prebuilt_cstr_coef,
            True)


def _read_wide_e_d(path: Path,
                    *,
                    provider: "object | None" = None) -> pl.DataFrame:
    """Read a CSV in either long-format (``entity, period, value``) or
    wide-format (``solve, period, e1, e2, …``) and return long form
    ``(e, d, value)``."""
    _p = Path(path)
    name = f"{_p.parent.name}/{_p.stem}" if _p.parent.name else _p.stem
    if not _provider_has(provider, name, path):
        return pl.DataFrame(schema={"e": pl.Utf8, "d": pl.Utf8, "value": pl.Float64})
    df = _provider_read(provider, name, path)
    if "solve" in df.columns:
        df = df.drop("solve")
    if df.height == 0 or "period" not in df.columns:
        return pl.DataFrame(schema={"e": pl.Utf8, "d": pl.Utf8, "value": pl.Float64})
    # Long-format detection: explicit (entity, period, value) columns.
    if {"entity", "period", "value"}.issubset(df.columns):
        return (df.pipe(rename_to_axis, {"entity": "e", "period": "d"})
                  .with_columns(value=pl.col("value")
                                        .cast(pl.Float64, strict=False)
                                        .fill_null(0.0))
                  .select("e", "d", "value"))
    val_cols = [c for c in df.columns if c != "period"]
    return (df.unpivot(on=val_cols, index=["period"], variable_name="e",
                       value_name="value")
              .pipe(rename_to_axis, {"period": "d"})
              .with_columns(value=pl.col("value")
                                  .cast(pl.Float64, strict=False)
                                  .fill_null(0.0)))


def _load_invest(sd: Path, dt: pl.DataFrame, inp: Path,
                  pss: pl.DataFrame | None,
                  *,
                  db_reader: "object | None" = None,
                  provider: "object | None" = None) -> dict:
    """Load invest/divest sets and per-(e, d) cost params.  Empty when
    neither ed_invest nor ed_divest has any row.

    Δ.19 — when ``db_reader`` is supplied AND the active solve is a
    recognised synthetic ``<base>_<anchor>`` (see
    :func:`_derived._resolve_synthetic_solve`), the 8 invest-set
    seed reads (``ed_invest``, ``ed_divest``, ``ed_invest_forbidden``,
    ``pd_invest``, ``pd_divest``, ``nd_invest``, ``nd_divest``,
    ``edd_invest``) are skipped — :func:`apply_synthetic_invest_sets`
    populates these fields from Spine during ``_apply_db_overrides``.
    The cost-param frames (``ed_lifetime_fixed_cost`` etc.) are read here
    when the Provider has them (see the comment above ``_read_e_d``), and
    the per-period cap sets (``ed_*_period_set``) remain on the CSV path.  The cap
    values (``e_*_max_total``, ``ed_*_max_period``) are not seeded: the DB
    overlay writes them on every path (a reader-less reload seeds them in
    ``_csv_reload._field_seeds``).
    """
    blank = dict(
        ed_invest_set=None, ed_divest_set=None,
        pd_invest_set=None, pd_divest_set=None,
        nd_invest_set=None, nd_divest_set=None,
        cohort=None, cohort_alive=None,
        p_entity_max_units=None,
        ed_lifetime_fixed_cost=None,
        ed_lifetime_fixed_cost_divest=None,
        ed_entity_annual_discounted=None,
        e_invest_total=None, e_divest_total=None,
        e_invest_max_total=None, e_divest_max_total=None,
        ed_invest_period_set=None, ed_divest_period_set=None,
        ed_invest_max_period=None, ed_divest_max_period=None,
    )
    # Δ.19 — detect synthetic ``<base>_<anchor>`` solve.  When matched,
    # the apply_synthetic_invest_sets path covers the 8 set frames, so
    # we skip those reads entirely (set frames default to None and the
    # override populates them after this loader returns).
    skip_set_seeds = False
    if db_reader is not None:
        try:
            scc = sd / "solve_current.csv"
            if _provider_has(provider, "solve_data/solve_current", scc):
                from flextool.engine.preprocess.params._derived import (_read_active_solve,
                                                _resolve_synthetic_solve,
                                                _solve_in_spine)
                # workdir = sd.parent (sd is solve_data/)
                active_solve = _read_active_solve(sd.parent, provider=provider)
                if (active_solve is not None
                        and not _solve_in_spine(db_reader, active_solve)
                        and _resolve_synthetic_solve(db_reader, active_solve)
                            is not None):
                    skip_set_seeds = True
        except Exception:  # pragma: no cover — defensive
            skip_set_seeds = False

    # Workdir-CSV seeds for the invest/divest cascade live in
    # ``_invest_seeds.py`` — keeps the synthetic-solve fallback I/O off
    # ``flex_data.py`` so the override chain stays the dominant data path.
    from flextool.engine.preprocess._invest_seeds import (
        read_invest_set as _seed_invest_set,
        read_forbidden_no_investment as _seed_forbidden_ni,
        read_set_seed as _seed_set,
        read_edd_invest as _seed_edd_invest,
    )

    if skip_set_seeds:
        # Set frames are populated by apply_synthetic_invest_sets after
        # this loader returns; the dispatch-only empty case is detected
        # from Spine directly (``solve.invest_periods`` for ``<base>``
        # at the anchor key is empty → no invest activity).  No disk
        # reads needed for the gate.
        from flextool.engine.preprocess.params._derived import _solve_periods as _sp
        invest_periods = _sp(db_reader, active_solve, "invest_periods")
        if not invest_periods:
            return blank
        # Cost-cascade seeds + per-period caps continue below; skip the
        # set-frame seeds (ed_inv/ed_div/forbid/pd/nd/edd).
        ed_inv = None  # type: ignore[assignment]
        ed_div = None  # type: ignore[assignment]
        pd_inv = None  # type: ignore[assignment]
        pd_div = None  # type: ignore[assignment]
        nd_inv = None  # type: ignore[assignment]
        nd_div = None  # type: ignore[assignment]
        edd_inv = None  # type: ignore[assignment]
    else:
        ed_inv = _seed_invest_set(sd, "ed_invest", "e", provider=provider)
        ed_div = _seed_invest_set(sd, "ed_divest", "e", provider=provider)
        if ed_inv.height == 0 and ed_div.height == 0:
            return blank

        forbid = _seed_forbidden_ni(sd, provider=provider)
        if forbid.height > 0:
            ed_inv = ed_inv.join(forbid, on=["e", "d"], how="anti")

        # Δ.18 — CSV-fallback seeds for pd/nd_invest_set, pd/nd_divest_set,
        # the investment cohorts (edd_invest).  The override chain (``apply_derived_c`` via the
        # lazy LFs in ``_existing.py``) overlays these when active.
        # For synthetic per-sub-solve fixtures the snapshot CSV is the only
        # source.
        pd_inv = _seed_set(sd, "pd_invest", "p", provider=provider)
        pd_div = _seed_set(sd, "pd_divest", "p", provider=provider)
        nd_inv = _seed_set(sd, "nd_invest", "n", provider=provider)
        nd_div = _seed_set(sd, "nd_divest", "n", provider=provider)

        edd_inv = _seed_edd_invest(sd, provider=provider)

    # The edd_invest seed as investment cohorts (their years unknown, never pre-built: _cohorts.seed_cohorts);
    # the investment walk of apply_derived_c replaces them when it finds periods in service.
    seed_cohort = seed_alive = None
    if edd_inv is not None and edd_inv.height > 0:
        from flextool.engine.preprocess.params._cohorts import seed_cohorts
        seed_cohort, seed_alive = seed_cohorts(edd_inv)

    # ``p_entity_max_units`` is produced by apply_derived_c; this provider seed is kept for synthetic
    # per-sub-solve names. The DB overlay normally wins.
    p_max_units = Param(("e", "d"),
        _read_wide_e_d(sd / "p_entity_max_units.csv", provider=provider)
            .filter(pl.col("value") > 0)
            .select("e", "d", "value")) if _provider_has(
                provider, "solve_data/p_entity_max_units",
                sd / "p_entity_max_units.csv") else None

    # The lifetime / NPV cost frames as the Provider holds them.  None is
    # emitted: ``_npv.apply_invest_lumps`` derives ``ed_entity_annual_discounted``
    # / ``ed_lifetime_fixed_cost`` for every solve that invests and
    # ``_fixed_costs.apply_retirement_fixed_cost`` ``ed_lifetime_fixed_cost_divest`` for
    # every solve that retires (Spine solve and synthetic sub-solve alike),
    # after this load, and registers its frames: what is read here is a
    # reader-less reload's frame of a dumped workdir, or the frame an earlier
    # solve sharing the Provider registered (replaced when this solve invests
    # or retires).
    # A solve with invest variables and neither frame fails in the objective
    # instead of investing for free.
    def _read_e_d(name: str) -> "Param | None":
        f = sd / f"{name}.csv"
        if not _provider_has(provider, f"solve_data/{name}", f):
            return None
        df = _read_wide_e_d(f, provider=provider)
        if df.height == 0:
            return None
        return Param(("e", "d"), df.select("e", "d", "value"))

    ed_lifetime_fc_seed       = _read_e_d("ed_lifetime_fixed_cost")
    ed_lifetime_fc_div_seed   = _read_e_d("ed_lifetime_fixed_cost_divest")
    ed_annual_disc_seed       = _read_e_d("ed_entity_annual_discounted")

    # ``ed_invest_period_set`` / ``ed_divest_period_set`` (set frames
    # of (e, d) pairs with per-period invest / divest caps) —
    # ``apply_derived_c`` populates these via
    # ``ed_invest_period_set_from_source`` /
    # ``ed_divest_period_set_from_source`` when active_solve is in
    # Spine; the workdir-CSV seeds below cover the synthetic per-sub-
    # solve case.
    from flextool.engine.preprocess._invest_seeds import read_period_set as _seed_period_set

    # Multi-solve handoff state.  These files are written between
    # sub-solves; the .mod uses them as constants on the
    # max/min Invest/Divest_entity_total + cumulative-group
    # constraints.  Empty / missing → no prior-solve activity.
    # Δ.12c — ``_read_handoff_e_d`` and ``_read_handoff_e`` retired:
    # the handoff carriers (``p_entity_previously_invested_capacity`` /
    # ``p_entity_invested`` / ``p_entity_divested``) are now produced by
    # ``apply_derived_f`` BEFORE ``apply_existing_chain`` consumes them,
    # so the CSV seed is no longer load-bearing.

    # Δ.12-drop: CSV seeds for fields whose override-chain helpers are
    # authoritative are set to None below.  The override chain repopulates
    # each field unconditionally:
    #   * ``e_invest_max_total`` / ``e_divest_max_total`` / ``e_invest_min_total``
    #     / ``e_divest_min_total``  ← ``apply_direct_params._e_total_param``.
    #   * ``ed_lifetime_fixed_cost`` / ``ed_lifetime_fixed_cost_divest`` /
    #     ``ed_entity_annual_discounted`` ← ``apply_derived_f`` (npv).
    #   * ``ed_invest_max_period`` / ``ed_divest_max_period``
    #     ← ``apply_direct_params`` via ``ed_*_max_period_from_source``.
    #
    # Δ.12c — handoff carriers (``p_entity_previously_invested_capacity`` /
    # ``p_entity_invested`` / ``p_entity_divested``) dropped: now produced
    # authoritatively by ``apply_derived_f``.  Since Δ.12c moved
    # ``apply_existing_chain`` to run AFTER ``apply_derived_f`` (was inside
    # ``apply_derived_d``), the chain summation sees the carriers populated
    # by the override helper without the seed.
    def _hnz(x):  # height-non-zero predicate that tolerates None
        return x if (x is not None and x.height > 0) else None

    return dict(
        ed_invest_set=_hnz(ed_inv),
        ed_divest_set=_hnz(ed_div),
        pd_invest_set=_hnz(pd_inv),
        pd_divest_set=_hnz(pd_div),
        nd_invest_set=_hnz(nd_inv),
        nd_divest_set=_hnz(nd_div),
        cohort=seed_cohort,
        cohort_alive=seed_alive,
        p_entity_max_units=p_max_units,
        # The cost frames the Provider holds (see ``_read_e_d`` above); the
        # override chain overlays them.
        ed_lifetime_fixed_cost=ed_lifetime_fc_seed,
        ed_lifetime_fixed_cost_divest=ed_lifetime_fc_div_seed,
        ed_entity_annual_discounted=ed_annual_disc_seed,
        e_invest_total=None,
        e_divest_total=None,
        e_invest_max_total=None,
        e_divest_max_total=None,
        ed_invest_period_set=_seed_period_set(sd, "ed_invest_period",
                                                provider=provider),
        ed_divest_period_set=_seed_period_set(sd, "ed_divest_period",
                                                provider=provider),
        ed_invest_max_period=None,
        ed_divest_max_period=None,
        p_entity_previously_invested_capacity=None,
        p_entity_invested=None,
        p_entity_divested=None,
    )


def _load_profiles(inp: Path, sd: Path, pss: pl.DataFrame | None,
                    unitsize: Param | None,
                    cap_pd: pl.DataFrame | None,
                    *,
                    provider: "object | None" = None):
    """Load profile_flow_upper/lower/fixed mappings.  ``cap_pd`` is the
    (p, d, base_cap) frame; combined with unitsize we get the
    ``existing_count`` term used on the RHS."""
    if pss is None or unitsize is None or cap_pd is None:
        return [None]*6
    pp_path = sd / "process__source__sink__profile__profile_method.csv"
    if not _provider_has(provider,
                          "solve_data/process__source__sink__profile__profile_method",
                          pp_path):
        return [None]*6
    pp = _provider_read(provider,
                          "solve_data/process__source__sink__profile__profile_method",
                          pp_path).pipe(rename_to_axis, {"process":"p"})
    if pp.height == 0:
        return [None]*6
    method_col = "method" if "method" in pp.columns else "profile_method"
    upper = pp.filter(pl.col(method_col)=="upper_limit").select("p","source","sink","profile")
    lower = pp.filter(pl.col(method_col)=="lower_limit").select("p","source","sink","profile")
    fixed = pp.filter(pl.col(method_col)=="fixed").select("p","source","sink","profile")

    # profile values - file is solve, period, time, p1, p2... — wide per profile.
    # The CSV seed is kept: ``apply_profile_cascade`` (``apply_derived_a`` step 9) is not
    # always reached — ``dt_and_step_duration_from_source`` bails (and with it the whole
    # of ``apply_derived_a`` past step 1) whenever the source has no ``timeset.timeset_duration``
    # (e.g. a single-timeline model that relies on the legacy "no timeset_durations -> whole
    # timeline" default, ``input/_timeline.py`` Rule 2). That path never reassigns
    # ``p_profile_value``, so a scenario with ``profile_flow`` active and no DB-written
    # timeset_duration needs this seed (c24 stage 4 follow-up, restores f5970cdb's deletion).
    pdt_profile = sd / "pdtProfile.csv"
    profile_value = None
    if _provider_has(provider, "solve_data/pdtProfile", pdt_profile):
        prof_long = _read_wide_per_entity(pdt_profile, rename={"entity":"f"},
                                            provider=provider)
        if "profile" in upper.columns:
            upper = upper.pipe(rename_to_axis, {"profile": "f"})
            lower = lower.pipe(rename_to_axis, {"profile": "f"})
            fixed = fixed.pipe(rename_to_axis, {"profile": "f"})
        profile_value = Param(("f","d","t"), prof_long.select("f","d","t","value"))

    # existing_count = capacity / unitsize per (p, d).  For our scenarios
    # (no investment yet) this equals base_cap_pd.
    existing_count = Param(("p","d"), cap_pd.rename({"base":"value"}))

    # Δ.17c Gap C: ``p_process_availability`` produced authoritatively
    # by ``apply_direct_params`` via ``p_process_availability_from_source``
    # (uses the ``_param_shapes`` resolver — unions ``unit.availability``
    # + ``connection.availability`` with full broadcast cascade).  Local
    # pdtProcess.csv slice dropped.
    availability = None

    return upper, lower, fixed, profile_value, existing_count, availability


# ---------------------------------------------------------------------------
# The single loader

def _load_varcost(sd: Path, pss: pl.DataFrame | None,
                   *,
                   provider: "object | None" = None) -> dict:
    """Load the three per-side variable-cost (other_operational_cost) Params: the emit seeds.

    On every DB path ``_direct.apply_direct_params_b`` replaces each of the three with its DB-direct Param when
    the DB authors one (``if v is not None``), then ``_direct.apply_varcost_params`` builds the four index sets
    and ``p_pssdt_varCost`` from the three (the emit has no period × time tier, #74).  The loader does not seed
    those five: ``apply_varcost_params`` sets them to None before rebuilding them.  A reader-less CSV reload
    reads those five fields unchanged from the dump in ``_csv_reload/_field_seeds.py``; it does not call
    ``apply_varcost_params`` (the production producer).

      p_pdt_varCost_source  — pdtProcess_source[…,'other_operational_cost']  (p, source, d, t)
      p_pdt_varCost_sink    — pdtProcess_sink[…,'other_operational_cost']    (p, sink, d, t)
      p_pdt_varCost_process — pdtProcess[…,'other_operational_cost']         (p, d, t)
    """
    blank = dict(
        p_pdt_varCost_source=None,
        p_pdt_varCost_sink=None,
        p_pdt_varCost_process=None,
    )
    if pss is None:
        return blank

    # pdtProcess_source[p,source,'other_operational_cost',d,t] — wide param file
    def _slice_pds(name: str, side_col: str) -> Param | None:
        f = sd / f"{name}.csv"
        if not _provider_has(provider, f"solve_data/{name}", f):
            return None
        df = _provider_read(provider, f"solve_data/{name}", f)
        if df.height == 0:
            return None
        sliced = df.filter(pl.col("param") == "other_operational_cost") \
                   .drop("param")
        if sliced.height == 0:
            return None
        sliced = (sliced.pipe(rename_to_axis, {"process": "p", "period": "d", "time": "t"})
                          .with_columns(value=pl.col("value").cast(pl.Float64, strict=False)
                                                              .fill_null(0.0))
                          .filter(pl.col("value") != 0))
        if sliced.height == 0:
            return None
        return Param(("p", side_col, "d", "t"),
                     sliced.select("p", side_col, "d", "t", "value"))

    p_var_src  = _slice_pds("pdtProcess_source", "source")
    p_var_sink = _slice_pds("pdtProcess_sink",   "sink")

    # pdtProcess[p,'other_operational_cost',d,t] — process-level (no source/sink dim)
    p_var_proc = None
    pp_path = sd / "pdtProcess.csv"
    if _provider_has(provider, "solve_data/pdtProcess", pp_path):
        df = _provider_read(provider, "solve_data/pdtProcess", pp_path)
        if df.height > 0:
            sliced = df.filter(pl.col("param") == "other_operational_cost") \
                       .drop("param")
            if sliced.height > 0:
                sliced = (sliced.pipe(rename_to_axis, {"process": "p", "period": "d", "time": "t"})
                                  .with_columns(value=pl.col("value").cast(pl.Float64, strict=False)
                                                                      .fill_null(0.0))
                                  .filter(pl.col("value") != 0))
                if sliced.height > 0:
                    p_var_proc = Param(("p", "d", "t"),
                                       sliced.select("p", "d", "t", "value"))

    return dict(
        p_pdt_varCost_source=p_var_src,
        p_pdt_varCost_sink=p_var_sink,
        p_pdt_varCost_process=p_var_proc,
    )


def _load_fixed_cost(sd: Path,
                      *,
                      provider: "object | None" = None) -> dict:
    """Load (e, d) ed_fixed_cost and (e, d) p_entity_all_existing.

    Δ.18 — CSV-fallback seeds.  Override chain (``apply_derived_f`` /
    ``apply_existing_chain``) overlays these when active; for synthetic
    per-sub-solve fixtures the snapshot CSV is the only source.  Empty
    or missing CSV → None (override-only path).  All-zero rows are
    dropped from ``p_ed_fixed_cost`` to match the override's filter — the
    override's ``ed_lifetime_fixed_cost_*`` family filters zero rows; the
    ``p_ed_fixed_cost`` helper similarly skips zero-value rows so the
    "no fixed cost" semantic round-trips as None on both paths.
    """
    def _read_e_d_seed(name: str, drop_zero: bool = False) -> "Param | None":
        f = sd / f"{name}.csv"
        if not _provider_has(provider, f"solve_data/{name}", f):
            return None
        df = _read_wide_e_d(f, provider=provider)
        if df.height == 0:
            return None
        if drop_zero:
            df = df.filter(pl.col("value") != 0.0)
        if df.height == 0:
            return None
        return Param(("e", "d"), df.select("e", "d", "value"))

    return dict(
        p_ed_fixed_cost=_read_e_d_seed("ed_fixed_cost", drop_zero=True),
        # ``p_entity_all_existing`` keeps zero rows — the chain consumer
        # uses them as the "no existing" sentinel, distinct from absent.
        p_entity_all_existing=_read_e_d_seed("p_entity_all_existing"),
    )


def _load_node_capacity_for_scaling(sd: Path,
                                     nb: pl.DataFrame,
                                     nbp: pl.DataFrame | None = None,
                                     *,
                                     provider: "object | None" = None) -> dict:
    """Load node_capacity_for_scaling[n, d] for slack-penalty scaling."""
    blank = dict(p_node_capacity_for_scaling=None)
    f = sd / "node_capacity_for_scaling.csv"
    if not _provider_has(provider, "solve_data/node_capacity_for_scaling", f):
        return blank
    df = _provider_read(provider, "solve_data/node_capacity_for_scaling", f)
    if df.height == 0:
        return blank
    df = df.pipe(rename_to_axis, {"node": "n", "period": "d"}) \
           .with_columns(value=pl.col("value").cast(pl.Float64, strict=False).fill_null(0.0))
    # Restrict to nodes carrying slack vars to avoid spurious rows.  The
    # slack vars span ``nodeBalance ∪ nodeBalancePeriod`` (period nodes
    # also carry ``vq_state_up/down``), so the restriction set is their
    # dtype-safe union.  When ``nbp`` is None/empty the union collapses to
    # exactly ``nb`` (no reorder/dup) → byte-identical legacy behavior.
    if nb is not None and nb.height > 0:
        nodes_restrict = nb
        if nbp is not None and nbp.height > 0:
            nodes_restrict = (pl.concat([nb.select("n"), nbp.select("n")])
                                .unique()
                                .select("n"))
        df = df.join(nodes_restrict, on="n", how="inner")
    if df.height == 0:
        return blank
    return dict(p_node_capacity_for_scaling=Param(("n", "d"), df.select("n", "d", "value")))


def _load_cumulative_invest(inp: Path, sd: Path, dt: pl.DataFrame,
                              *,
                              provider: "object | None" = None) -> dict:
    """Load the new ``FlexData`` fields consumed by ``_cumulative_invest``.

    All fields are independently optional — missing CSV / empty file ⇒ None.
    Sets are filtered to keep only non-empty rows; per-period parameters
    drop all-zero rows so ``has_feature(d)`` won't fire on placeholder
    fixtures whose CSVs exist with all-zero placeholders.
    """
    out: dict = {}

    def _read_set(name: str, src_to_dst: dict[str, str]) -> pl.DataFrame | None:
        f = sd / f"{name}.csv"
        if not _provider_has(provider, f"solve_data/{name}", f):
            return None
        df = _provider_read(provider, f"solve_data/{name}", f)
        if df.height == 0:
            return None
        rename = {s: d for s, d in src_to_dst.items() if s in df.columns}
        out_df = df.pipe(rename_to_axis, rename).select(*src_to_dst.values()).unique(maintain_order=True)
        return out_df if out_df.height > 0 else None

    # Δ.17 — ``_read_set_drop_zeros`` / ``_read_e_d_param`` / ``_read_e_param``
    # / ``_slice_pdgroup`` / ``_slice_pgroup`` were dead-code helpers
    # retained from Δ.12-drop (their consumer Params were retired but
    # the inner-function definitions weren't cleaned up).  Removed; the
    # override chain produces the corresponding fields:
    #   * ``ed_invest_min_period`` etc. ← ``apply_direct_params``.
    #   * ``p_group_invest_*_period`` etc. ← ``apply_direct_params``.
    # 5 dead-code ``_read_csv_file`` calls retired.

    def _slice_pdtgroup(param_name: str) -> pl.DataFrame | None:
        """solve_data/pdtGroup.csv slice → (g, d, t, value), zero dropped."""
        f = sd / "pdtGroup.csv"
        if not _provider_has(provider, "solve_data/pdtGroup", f):
            return None
        df = _provider_read(provider, "solve_data/pdtGroup", f)
        if df.height == 0:
            return None
        sliced = (df.filter(pl.col("param") == param_name)
                    .pipe(rename_to_axis, {"group": "g", "period": "d", "time": "t"})
                    .with_columns(pl.col("value").cast(pl.Float64, strict=False))
                    .filter(pl.col("value").is_not_null() & (pl.col("value") != 0.0))
                    .select("g", "d", "t", "value"))
        return sliced if sliced.height > 0 else None

    # ── Sets (key-only frames) ────────────────────────────────────────────
    out["ed_invest_cumulative"] = _read_set(
        "ed_invest_cumulative", {"entity": "e", "period": "d"})

    # group_entity: solve_data/group_entity.csv (written by every solve via
    # emit._mid_sets.emit_group_entity).  The input/group__entity.csv fallback arm nothing
    # ever populates ("input/group__entity" has no producer) was dropped, c24 S4-6.
    ge = None
    ge_cand, ge_mapping, ge_key = (
        sd / "group_entity.csv", {"group": "g", "entity": "e"}, "solve_data/group_entity")
    if _provider_has(provider, ge_key, ge_cand):
        df = _provider_read(provider, ge_key, ge_cand)
        if df.height > 0:
            ge = df.pipe(rename_to_axis, ge_mapping).select("g", "e").unique()
    out["group_entity"] = ge

    # group_process_node: input/flowGroup__process__node.csv (raw long, v58; written by the
    # input-derivation exporter).  The solve_data/group_process_node.csv candidate nothing
    # ever writes ("solve_data/group_process_node" has no producer) was dropped, c24 S4-6.
    gpn = None
    gpn_cand = inp / "flowGroup__process__node.csv"
    gpn_mapping = {"flowGroup": "g", "process": "p", "node": "n"}
    gpn_key = "input/flowGroup__process__node"
    if _provider_has(provider, gpn_key, gpn_cand):
        df = _provider_read(provider, gpn_key, gpn_cand)
        if df.height > 0:
            gpn = df.pipe(rename_to_axis, gpn_mapping).select("g", "p", "n").unique()
    out["group_process_node"] = gpn

    out["g_invest_total"]      = _read_set("g_invest_total", {"group": "g"})
    out["g_divest_total"]      = _read_set("g_divest_total", {"group": "g"})
    out["g_invest_cumulative"] = _read_set("g_invest_cumulative", {"group": "g"})
    out["gd_invest_period"]    = _read_set("gd_invest_period",
                                            {"group": "g", "period": "d"})
    out["gd_divest_period"]    = _read_set("gd_divest_period",
                                            {"group": "g", "period": "d"})

    # Δ.12-drop: ``ed_invest_min_period`` / ``ed_divest_min_period`` /
    # ``ed_cumulative_max_capacity`` / ``ed_cumulative_min_capacity`` /
    # ``e_invest_min_total`` / ``e_divest_min_total`` /
    # ``p_group_invest_max_period`` / ``p_group_invest_min_period`` /
    # ``p_group_retire_max_period`` / ``p_group_retire_min_period`` /
    # ``p_group_invest_max_total`` / ``p_group_invest_min_total`` /
    # ``p_group_retire_max_total`` / ``p_group_retire_min_total`` /
    # ``pd_group_cumulative_max_capacity`` / ``pd_group_cumulative_min_capacity`` /
    # ``p_group_max_cumulative_flow`` / ``p_group_min_cumulative_flow`` /
    # ``pd_max_cumulative_flow`` / ``pd_min_cumulative_flow``
    # all produced authoritatively by ``apply_direct_params`` (Δ.4b).
    # Seeds dropped.
    out["ed_invest_min_period"]       = None
    out["ed_divest_min_period"]       = None
    out["ed_cumulative_max_capacity"] = None
    out["ed_cumulative_min_capacity"] = None
    out["e_invest_min_total"]         = None
    out["e_divest_min_total"]         = None

    # Δ.17c Gap C: ``pdt_max_instant_flow`` / ``pdt_min_instant_flow``
    # produced authoritatively by ``apply_direct_params`` via
    # ``pdt_max_instant_flow_from_source`` / ``pdt_min_instant_flow_from_source``
    # (use the ``_param_shapes`` resolver — full scalar / 1d_map[period] /
    # 1d_map[time] / 2d_map[period,time] cascade).  Local pdtGroup.csv
    # slices dropped.
    pdt_max = None
    pdt_min = None
    out["p_group_invest_max_period"]      = None
    out["p_group_invest_min_period"]      = None
    out["p_group_retire_max_period"]      = None
    out["p_group_retire_min_period"]      = None
    out["p_group_invest_max_total"]       = None
    out["p_group_invest_min_total"]       = None
    out["p_group_retire_max_total"]       = None
    out["p_group_retire_min_total"]       = None
    out["pd_group_cumulative_max_capacity"] = None
    out["pd_group_cumulative_min_capacity"] = None
    out["p_group_max_cumulative_flow"]    = None
    out["p_group_min_cumulative_flow"]    = None
    out["pd_max_cumulative_flow"]         = None
    out["pd_min_cumulative_flow"]         = None
    out["pdt_max_instant_flow"]           = (Param(("g", "d", "t"), pdt_max)
                                              if pdt_max is not None else None)
    out["pdt_min_instant_flow"]           = (Param(("g", "d", "t"), pdt_min)
                                              if pdt_min is not None else None)
    # NB: the instant-flow constraint support is no longer materialised
    # as a separate ``gdt_*InstantFlow`` field — ``_cumulative_invest._eq_instant_flow``
    # derives the (g, d, t) domain from the resolved cap directly.

    return out


def _load_stochastics(inp: Path, sd: Path, dt: pl.DataFrame,
                       *,
                       provider: "object | None" = None) -> dict:
    """Load multi-branch stochastic operational data.

    Mirrors flextool.mod's stochastic feature (mod:38-41, :562-588,
    :873-895, :988, :1978-2142, :4173-4233).  Reads four CSVs:

    * ``solve_data/pdt_branch_weight.csv`` (period, time, value) →
      :class:`Param` keyed (d, t).  Per-branch operational probability
      that multiplies every dispatch-class objective term.  Defaults to
      1.0 per (d, t) when CSV is empty / missing.
    * ``solve_data/pd_branch_weight.csv`` (period, value) →
      :class:`Param` keyed (d,).  Per-branch period-level probability
      for investment-fixed-cost terms.  Defaults to 1.0.
    * ``solve_data/dt_non_anticipativity_set.csv`` (period, time) →
      ``pl.DataFrame``.  Realised-dispatch + fix-storage timesteps where
      the four ``non_anticipativity_*`` constraints fire.  Empty when
      stochastics inactive.
    * ``input/groupIncludeStochastics.csv`` (group,) → ``pl.DataFrame``.
      Groups whose ``group_node`` membership unlocks the storage
      non-anticipativity coupling (``non_anticipativity_storage_use``).

    Also loads the unfiltered ``period__branch.csv`` (anchor → sibling)
    distinct from the existing ``period_branch`` rolling-handoff field
    (which renames columns to ``d_upper``/``d``).  And the active
    ``period_in_use_set`` from ``solve_data/period_in_use_set.csv`` —
    used by the model layer to filter branch periods that exist in the
    metadata-only ``period_branch`` map but aren't part of the actual
    LP (e.g. ``period1_realized`` in the 2_day_stochastic_dispatch
    fixture).
    """
    # Δ.18 — CSV-fallback seeds for ``pdt_branch_weight`` /
    # ``pd_branch_weight`` / ``period_in_use_set``.  Override chain
    # (``apply_branch_cluster`` in ``apply_derived_g``) overlays these
    # when active; for synthetic per-sub-solve fixtures the snapshot
    # CSV is the only source.  pdt_branch_weight defaults to dense-dt
    # × 1.0 when CSV missing / empty (mirrors mod's
    # ``param pdt_branch_weight {(d,t) in dt}`` declaration).
    pdt_branch_weight = None
    pdt_bw_path = sd / "pdt_branch_weight.csv"
    if _provider_has(provider, "solve_data/pdt_branch_weight", pdt_bw_path):
        df = _provider_read(provider, "solve_data/pdt_branch_weight", pdt_bw_path)
        if df.height > 0:
            df = (df.pipe(rename_to_axis, {"period": "d", "time": "t"})
                    .with_columns(value=pl.col("value")
                                            .cast(pl.Float64, strict=False)
                                            .fill_null(1.0))
                    .select("d", "t", "value"))
            base = dt.with_columns(value=pl.lit(1.0)).select("d", "t", "value")
            base = (base
                    .join(df, on=["d", "t"], how="left", suffix="__r")
                    .with_columns(value=pl.coalesce(
                        pl.col("value__r"), pl.col("value")))
                    .select("d", "t", "value"))
            pdt_branch_weight = Param(("d", "t"), base)

    pd_branch_weight = None
    pd_bw_path = sd / "pd_branch_weight.csv"
    if _provider_has(provider, "solve_data/pd_branch_weight", pd_bw_path):
        df = _provider_read(provider, "solve_data/pd_branch_weight", pd_bw_path)
        if df.height > 0:
            df = (df.pipe(rename_to_axis, {"period": "d"})
                    .with_columns(value=pl.col("value")
                                            .cast(pl.Float64, strict=False)
                                            .fill_null(1.0))
                    .select("d", "value"))
            pd_branch_weight = Param(("d",), df)

    # Δ.12-drop: ``dt_non_anticipativity`` / ``period_branch_full``
    # produced authoritatively by ``apply_branch_cluster`` in
    # ``apply_derived_g``.  Seeds dropped.
    dt_non_anticipativity = None
    period_branch_full = None

    # Δ.18 — CSV-fallback seed for ``period_in_use_set``.
    period_in_use_set = None
    piu_path = sd / "period_in_use_set.csv"
    if _provider_has(provider, "solve_data/period_in_use_set", piu_path):
        df = _provider_read(provider, "solve_data/period_in_use_set", piu_path)
        if df.height > 0 and "period" in df.columns:
            period_in_use_set = df.pipe(rename_to_axis, {"period": "d"}).select("d").unique(maintain_order=True)

    label_steps = None
    ls_path = sd / "label_steps.csv"
    if _provider_has(provider, "solve_data/label_steps", ls_path):
        df = _provider_read(provider, "solve_data/label_steps", ls_path)
        if df.height > 0 and {"period", "step"}.issubset(df.columns):
            label_steps = (df.pipe(rename_to_axis, {"period": "d", "step": "t"})
                           .select("d", "t").unique(maintain_order=True))

    # groupIncludeStochastics: (g,)
    gis_path = inp / "groupIncludeStochastics.csv"
    groupStochastic = None
    if _provider_has(provider, "input/groupIncludeStochastics", gis_path):
        df = _provider_read(provider, "input/groupIncludeStochastics", gis_path)
        if df.height > 0:
            # CSV column is named ``group``; rename to canonical ``g``.
            df = df.pipe(rename_to_axis, {df.columns[0]: "g"})
            groupStochastic = df.select("g").unique()

    return dict(
        pdt_branch_weight=pdt_branch_weight,
        pd_branch_weight=pd_branch_weight,
        dt_non_anticipativity=dt_non_anticipativity,
        groupStochastic=groupStochastic,
        period_branch_full=period_branch_full,
        period_in_use_set=period_in_use_set,
        label_steps=label_steps,
    )


def _load_roll_period_whole(sd: Path, *, provider: "object | None" = None) -> pl.DataFrame | None:
    """``solve_data/roll_period_whole`` (period, whole): a row per period of a rolling sub-solve (``_run_model``);
    None when the solve is not a roll."""
    path = sd / "roll_period_whole.csv"
    if not _provider_has(provider, "solve_data/roll_period_whole", path):
        return None
    rpw = _provider_read(provider, "solve_data/roll_period_whole", path)
    return rpw if rpw.height > 0 else None


def _load_period_invest(sd: Path, *, provider: "object | None" = None) -> pl.DataFrame | None:
    """``period_invest`` (d,): the active solve's investment periods and the branch periods of those anchors.

        period_invest = { d : d ∈ invest_periods_of_current_solve }
                      ∪ { b : (d, b) ∈ period__branch, d ∈ invest_periods_of_current_solve }

    ``solve_data/invest_periods_of_current_solve.csv`` is GMPL's ``period_invest``; the branch periods are
    added as today's model has them (the capacity margin in branch periods is pending the stochastics
    re-grounding).  ``period__branch.csv`` also has identity rows ``(d, d)``, which add nothing.  The model
    intersects the set with the periods of dt.  Sorted and de-duplicated; an empty frame when the solve has
    no investment periods; None when the invest-period file is not there at all.
    """
    from flextool.engine.preprocess.params._branch import period_branch_full_lf

    path = sd / "invest_periods_of_current_solve.csv"
    key = "solve_data/invest_periods_of_current_solve"
    if not _provider_has(provider, key, path):
        return None
    df = _provider_read(provider, key, path)
    anchors: set[str] = set()
    if df.height > 0:
        anchors = {str(v) for v in df.get_column(df.columns[0]).to_list() if v is not None and str(v) != ""}
    periods = set(anchors)
    if anchors:
        pbf = period_branch_full_lf(sd.parent, provider=provider).collect()
        for a, b in pbf.select(pl.col("d").cast(pl.Utf8), pl.col("b").cast(pl.Utf8)).iter_rows():
            if a in anchors and b is not None:
                periods.add(b)
    return (pl.DataFrame({"period": sorted(periods)}, schema={"period": pl.Utf8})
              .pipe(rename_to_axis, {"period": "d"}).select("d"))
