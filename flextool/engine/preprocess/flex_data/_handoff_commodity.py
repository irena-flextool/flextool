"""The cumulative simulated hours and the cumulative commodity a solve hands on."""

from __future__ import annotations

from pathlib import Path
import polars as pl
from flextool.engine.preprocess.flex_data._readers import _provider_has, _provider_read
from flextool.engine.preprocess.flex_data._container import FlexData
from flextool.engine.preprocess.flex_data._handoff_inputs import _realized_dispatch_frame, _step_duration_frame


def _extract_cum_sim_hours(
    sd: Path, *, prior_handoff=None, flex_data: "FlexData | None" = None,
    provider: "object | None" = None, is_dispatch_sibling_parent: bool = False,
) -> "pl.DataFrame | None":
    """Δ.11 — derive ``cum_sim_hours[period]`` from
    ``flex_data.p_step_duration`` (in-memory, when supplied) + the
    workdir's ``realized_dispatch.csv``.

    Algorithm:

      this_roll_hrs[d] = Σ_t step_duration[d, t]   for (d, t) ∈ realized_dispatch
      cum[d]           = prior[d] + this_roll[d]   ∀d ∈ keys(prior) ∪ keys(this_roll)

    Phase 4 (Gap F) — ``p_step_duration.csv`` read is replaced by the
    ``flex_data.p_step_duration`` Param when ``flex_data`` is supplied
    (the cascade always supplies it).  Test fixtures that call this
    helper directly without ``flex_data`` retain the disk fallback.

    ``realized_dispatch`` holds no cell of a stochastic fan: the period
    label (the original period name) has no dispatch and the forecast
    branches are never realized.  A fan's realized hours come only from
    the realization solve's own extraction (its period has the real name);
    a committed window in a forecasts-only fan that a later solve continues
    from raises before the LP (``_realization.check_committed_window_realized``),
    never a silent 0.  ``is_dispatch_sibling_parent`` (#59
    fix 3) is True when THIS solve is the parent of an auto-spawned
    realized-dispatch sibling (``state.solve.realized_dispatch_parent``):
    the sibling re-solves the SAME realized window with its own real
    (non-zero-weight) objective right after this solve and will add its own
    ``this_roll_hrs`` for those same hours (``_run_model``'s
    ``ROLL_FORWARD_FIELDS`` routes the sibling's prior from its OWN previous
    sibling roll, not from this parent) — so this parent must contribute
    nothing of its own here, or the hours would be counted twice.
    """
    this_roll_hrs: dict[str, float] = {}
    realized: set[tuple[str, str]] = set()
    if not is_dispatch_sibling_parent:
        rd_df = _realized_dispatch_frame(sd, flex_data, provider=provider)
        if rd_df is not None and rd_df.height > 0 and {"period", "step"}.issubset(rd_df.columns):
            for r in rd_df.iter_rows(named=True):
                realized.add((str(r["period"]), str(r["step"])))
    if realized:
        sd_df = _step_duration_frame(sd, flex_data, provider=provider)
        if sd_df is not None and sd_df.height > 0:
            d_col = "period" if "period" in sd_df.columns else (
                "d" if "d" in sd_df.columns else None)
            t_col = "time" if "time" in sd_df.columns else (
                "step" if "step" in sd_df.columns else (
                    "t" if "t" in sd_df.columns else None))
            v_col = "value" if "value" in sd_df.columns else (
                "p_step_duration" if "p_step_duration" in sd_df.columns else None
            )
            if d_col is not None and t_col is not None and v_col is not None:
                for r in sd_df.iter_rows(named=True):
                    key = (str(r[d_col]), str(r[t_col]))
                    if key in realized:
                        try:
                            this_roll_hrs[key[0]] = this_roll_hrs.get(key[0], 0.0) + float(r[v_col])
                        except (TypeError, ValueError):
                            continue
    prior_hrs: dict[str, float] = {}
    if prior_handoff is not None and prior_handoff.cum_sim_hours is not None:
        # Canonical schema is [period, p_ladder_cum_sim_hours] (Phase 4.1a).
        for r in prior_handoff.cum_sim_hours.iter_rows(named=True):
            prior_hrs[str(r["period"])] = float(r["p_ladder_cum_sim_hours"])
    if not this_roll_hrs and not prior_hrs:
        return None
    keys = sorted(set(prior_hrs) | set(this_roll_hrs))
    rows = [(d, prior_hrs.get(d, 0.0) + this_roll_hrs.get(d, 0.0)) for d in keys]
    return pl.DataFrame(
        rows, schema=["period", "p_ladder_cum_sim_hours"], orient="row",
    )


def _extract_cumulative_commodity(
    sol, sd: Path, *, prior_handoff=None, flex_data: "FlexData | None" = None,
    provider: "object | None" = None, is_dispatch_sibling_parent: bool = False,
) -> "pl.DataFrame | None":
    """Δ.11 — derive ``cumulative_commodity[c, i, d]`` from ``v_trade``
    on the LP solution + workdir-side accumulator metadata.

    Algorithm:

      this_roll[c, i, d] = Σ_n v_trade[c, n, d, i] × unitsize[c]
                              × (realized_hours[d] / horizon_hours[d])

      cum[c, i, d] = prior[c, i, d] + this_roll[c, i, d]

    Restricted to ``(c, i)`` in ``ci_ladder_cumulative.csv`` (the finite
    tiers; non-finite tiers don't need carry-over).  Returns the wide
    ``[commodity, tier, period, mwh]`` carrier frame, or ``None`` when
    no v_trade variable exists on the solution AND no prior accumulator
    is supplied (caller falls back to the workdir CSV).

    ``realized_hours[d]`` (below) comes from ``realized_dispatch``, which
    holds no cell of a stochastic fan (see :func:`_extract_cum_sim_hours`):
    a fan's purchases are carried only by the realization solve's own
    extraction.
    ``is_dispatch_sibling_parent`` (#59 fix 3) — see
    :func:`_extract_cum_sim_hours`'s docstring for why this solve must
    contribute nothing of its own when a realized-dispatch sibling will
    re-solve (and carry forward) the same hours right after it.
    """
    if sol is None or "v_trade" not in getattr(sol, "_vars", {}):
        # Without v_trade we can only propagate prior — same shape the
        # legacy file path uses.  Returning None lets the caller fall
        # through to the file-based propagation.
        return None

    # Finite ladder tiers — cumulative restriction set.  Phase 4 (Gap F):
    # prefer in-memory ``flex_data.ci_ladder_cumulative`` over the workdir
    # CSV when supplied.
    #
    # B1 — also filter by quantity to mirror the legacy
    # ``_load_finite_ladder_tiers``:
    # tiers with the ``1e30`` infinity sentinel (quantity >= 1e29) are
    # dropped because their cap never binds and a 0-valued accumulator
    # row carries no semantic content.  Without this filter
    # infinity-sentinel tiers like ``coal,2`` leak into the carrier with
    # 0.0 values and breach the test invariant in
    # ``tests/test_commodity_ladder_rolling.py``.
    #
    # B1 — additionally union annual-ladder tiers from
    # ``commodity__tier_ann``.  Annual fixtures may have an EMPTY
    # ``ci_ladder_cumulative`` (no cumulative ladder commodities) but
    # still need accumulator rows — the legacy
    # ``_load_finite_ladder_tiers`` reads BOTH
    # ``commodity_ladder_cumulative`` and ``commodity_ladder_annual``
    # for this same reason.
    finite_tiers: set[tuple[str, str]] = set()
    cilc_df = None
    if flex_data is not None and getattr(flex_data, "ci_ladder_cumulative", None) is not None:
        cilc_df = flex_data.ci_ladder_cumulative
    else:
        cilc_path = sd / "ci_ladder_cumulative.csv"
        if _provider_has(provider, "solve_data/ci_ladder_cumulative", cilc_path):
            try:
                cilc_df = _provider_read(provider, "solve_data/ci_ladder_cumulative", cilc_path)
            except pl.exceptions.NoDataError:
                cilc_df = None
    if cilc_df is not None and cilc_df.height > 0:
        # FlexData carries (c, i); the CSV carries (commodity, tier).
        c_col = "commodity" if "commodity" in cilc_df.columns else "c"
        i_col = "tier" if "tier" in cilc_df.columns else "i"
        if c_col in cilc_df.columns and i_col in cilc_df.columns:
            for r in cilc_df.iter_rows(named=True):
                finite_tiers.add((str(r[c_col]), str(r[i_col])))

    # Annual-ladder tier set (commodity__tier_ann) — additive.
    cta_df = None
    if flex_data is not None and getattr(flex_data, "commodity__tier_ann", None) is not None:
        cta_df = flex_data.commodity__tier_ann
    else:
        cta_path = sd / "commodity__tier_ann.csv"
        if _provider_has(provider, "solve_data/commodity__tier_ann", cta_path):
            try:
                cta_df = _provider_read(provider, "solve_data/commodity__tier_ann", cta_path)
            except pl.exceptions.NoDataError:
                cta_df = None
    if cta_df is not None and cta_df.height > 0:
        c_col = "commodity" if "commodity" in cta_df.columns else "c"
        i_col = "tier" if "tier" in cta_df.columns else "i"
        if c_col in cta_df.columns and i_col in cta_df.columns:
            for r in cta_df.iter_rows(named=True):
                finite_tiers.add((str(r[c_col]), str(r[i_col])))

    if not finite_tiers:
        return None

    # B1 — drop infinity-sentinel tiers (quantity >= 1e29).  The
    # quantity column lives on ``input/commodity_ladder_cumulative`` and
    # ``input/commodity_ladder_annual``; a tier is "finite" if either
    # source carries a finite quantity for it (the annual cap evaluates
    # per period, so any finite-cap period qualifies the tier).
    _INFINITE_TIER_THRESHOLD = 1e29
    qty_finite: set[tuple[str, str]] = set()
    for prov_key, csv_path in (
        ("input/commodity_ladder_cumulative",
         (Path(sd).parent / "input" / "commodity_ladder_cumulative.csv")),
        ("input/commodity_ladder_annual",
         (Path(sd).parent / "input" / "commodity_ladder_annual.csv")),
    ):
        qdf = None
        if _provider_has(provider, prov_key, csv_path):
            try:
                qdf = _provider_read(provider, prov_key, csv_path)
            except pl.exceptions.NoDataError:
                qdf = None
        if qdf is None or qdf.height == 0:
            continue
        if not {"commodity", "tier", "quantity"}.issubset(qdf.columns):
            continue
        for r in qdf.iter_rows(named=True):
            c, i = str(r["commodity"]), str(r["tier"])
            try:
                q = float(r["quantity"])
            except (TypeError, ValueError):
                continue  # not a number: no finite cap
            if q != q or q >= _INFINITE_TIER_THRESHOLD:
                continue
            qty_finite.add((c, i))
    if qty_finite:
        finite_tiers = finite_tiers & qty_finite
    if not finite_tiers:
        return None

    # Per-period horizon vs realized hours (uniform-split fraction).
    # Phase 4 (Gap F) — ``p_step_duration`` sourced from FlexData when
    # supplied.  Gap F final — ``realized_dispatch`` also flows through
    # ``flex_data.realized_dispatch`` via ``_realized_dispatch_frame``.
    horizon_hrs: dict[str, float] = {}
    realized_hrs: dict[str, float] = {}
    sd_df = _step_duration_frame(sd, flex_data, provider=provider)
    if sd_df is not None and sd_df.height > 0:
        d_col = "period" if "period" in sd_df.columns else (
            "d" if "d" in sd_df.columns else None)
        t_col = "time" if "time" in sd_df.columns else (
            "step" if "step" in sd_df.columns else (
                "t" if "t" in sd_df.columns else None))
        v_col = "value" if "value" in sd_df.columns else (
            "p_step_duration" if "p_step_duration" in sd_df.columns else None
        )
        realized_set: set[tuple[str, str]] = set()
        if not is_dispatch_sibling_parent:
            rd_df = _realized_dispatch_frame(sd, flex_data, provider=provider)
            if rd_df is not None and rd_df.height > 0 and {"period", "step"}.issubset(rd_df.columns):
                for r in rd_df.iter_rows(named=True):
                    realized_set.add((str(r["period"]), str(r["step"])))
        if d_col is not None and t_col is not None and v_col is not None:
            for r in sd_df.iter_rows(named=True):
                d_v = str(r[d_col])
                try:
                    dur = float(r[v_col])
                except (TypeError, ValueError):
                    continue
                horizon_hrs[d_v] = horizon_hrs.get(d_v, 0.0) + dur
                if (d_v, str(r[t_col])) in realized_set:
                    realized_hrs[d_v] = realized_hrs.get(d_v, 0.0) + dur

    # Commodity unitsize (defaults 1.0 when absent — flextool default).
    # Phase 4 (Gap F) — in-memory ``flex_data.p_commodity_unitsize``.  The provider disk-fallback arm
    # ("solve_data/p_commodity_unitsize") nothing ever writes was dropped, c24 S4-6: the exporter writes
    # ``input/p_commodity_unitsize.csv`` (a different key, never checked here), and no producer puts
    # the "solve_data/" key the fallback looked for.
    unitsize: dict[str, float] = {}
    cu_df = None
    if flex_data is not None and getattr(flex_data, "p_commodity_unitsize", None) is not None:
        cu_df = flex_data.p_commodity_unitsize.frame
    if cu_df is not None and cu_df.height > 0:
        c_col = "commodity" if "commodity" in cu_df.columns else (
            "c" if "c" in cu_df.columns else "name")
        v_col = "value" if "value" in cu_df.columns else "p_commodity_unitsize"
        if c_col in cu_df.columns and v_col in cu_df.columns:
            for r in cu_df.iter_rows(named=True):
                try:
                    unitsize[str(r[c_col])] = float(r[v_col])
                except (TypeError, ValueError):
                    continue

    # v_trade extraction — schema from _commodity_ladder.add_variables.
    v_trade_df = sol.value("v_trade")
    this_roll: dict[tuple[str, str, str], float] = {}
    if v_trade_df is not None and v_trade_df.height > 0:
        for r in v_trade_df.iter_rows(named=True):
            c, i, d = str(r["c"]), str(r["i"]), str(r["d"])
            v = float(r["value"])
            if (c, i) not in finite_tiers:
                continue
            hz = horizon_hrs.get(d, 0.0)
            rz = realized_hrs.get(d, 0.0)
            if hz <= 0.0 or rz <= 0.0:
                continue
            us = unitsize.get(c, 1.0)
            key = (c, i, d)
            this_roll[key] = this_roll.get(key, 0.0) + v * us * (rz / hz)

    # Prior accumulator (carry across solves).
    # Canonical schema is [commodity, tier, period, p_ladder_cum_realized_mwh]
    # (Phase 4.1a).
    prior: dict[tuple[str, str, str], float] = {}
    if prior_handoff is not None and prior_handoff.cumulative_commodity is not None:
        for r in prior_handoff.cumulative_commodity.iter_rows(named=True):
            key = (str(r["commodity"]), str(r["tier"]), str(r["period"]))
            prior[key] = float(r["p_ladder_cum_realized_mwh"])

    if not this_roll and not prior:
        return None

    keys = sorted(set(prior) | set(this_roll))
    rows = [(c, i, d, prior.get((c, i, d), 0.0) + this_roll.get((c, i, d), 0.0))
                for (c, i, d) in keys]
    return pl.DataFrame(
        rows,
        schema={"commodity": pl.Utf8, "tier": pl.Utf8, "period": pl.Utf8,
                "p_ladder_cum_realized_mwh": pl.Float64},
        orient="row",
    )
