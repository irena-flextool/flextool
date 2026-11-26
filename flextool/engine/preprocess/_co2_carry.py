"""The carry of the CO2 caps across rolls and the solves of a chain (#38, Q22, item 1042): what a solve reads of
the CO2 its chain already realized, and how much of the horizon it reaches (``load_co2_rolling_carry``, the
consumers ``model/core.py`` ``_co2_period_cap_with_carry`` / ``_co2_total_cap_with_carry``).

A period that the solves of a chain split (each solve declares a part of it, item 1042) is known to a solve as the
hours earlier solves realized of it plus this solve's own complete hours of it.  co2_max_total weighs every hour of
such a period by that: share*[d] = period_share[d] + hours_before_solve[d] / 8760 (:func:`split_share`), and its
share reached is f*[d] = (realized hours + this roll's hours) / (share*[d] · 8760) (:func:`split_f_d_k`).  A solve
whose periods no earlier solve realized (every single solve, every rolling solve that holds its periods alone) has
no such period, so its rows are unchanged.
"""
from __future__ import annotations

from pathlib import Path

import polars as pl
from polar_high import Param

from flextool.engine.input._axis_enums import join_aligned, rename_to_axis
from flextool.engine.input._constants import HOURS_PER_YEAR
from flextool.engine.preprocess.params._branch import fan_period_labels, period_branch_pairs_lf


def _hours_before_solve(provider) -> "Param | None":
    """``(d,)`` hours earlier solves of the chain realized of the solve's periods, None without any."""
    if not provider.has("solve_data/hours_realized_by_earlier_solves"):
        return None
    raw = provider.get("solve_data/hours_realized_by_earlier_solves")
    if raw is None or raw.height == 0:
        return None
    return Param(("d",), raw.pipe(rename_to_axis, {"period": "d"})
                 .select("d", pl.col("value").cast(pl.Float64, strict=False)))


def split_share(period_share: pl.DataFrame, before: pl.DataFrame) -> pl.DataFrame:
    """``[d, value]`` share* = period_share + hours_before_solve / 8760 for the periods of *before* (the split
    periods)."""
    return (join_aligned(period_share.select("d", share="value"), before.select("d", hours="value"),
                         on="d", how="inner")
            .select("d", value=pl.col("share") + pl.col("hours") / HOURS_PER_YEAR))


def split_f_d_k(fdk: pl.DataFrame, before: pl.DataFrame, provider) -> pl.DataFrame:
    """*fdk* ``[d, value]`` with each split period's value f_d_k · share / share*: the share reached of the hours
    known so far (its f_d_k counts the realized hours of the earlier solves against this solve's own hours only)."""
    if not provider.has("solve_data/complete_period_share_of_year_calc"):
        return fdk
    share = (provider.get("solve_data/complete_period_share_of_year_calc").pipe(rename_to_axis, {"period": "d"})
             .select("d", pl.col("value").cast(pl.Float64, strict=False)))
    star = split_share(share, before).rename({"value": "star"})
    return (join_aligned(fdk, join_aligned(share.rename({"value": "share"}), star, on="d", how="inner"),
                         on="d", how="left")
            .with_columns(value=pl.when(pl.col("star").is_null()).then(pl.col("value"))
                          .otherwise(pl.col("value") * pl.col("share") / pl.col("star")))
            .select("d", "value"))


def _fan_members(workdir: Path, provider) -> "tuple[set[str], dict[str, str]]":
    """``(members, first_member)`` of the solve's stochastic fans: every fan member period (a forecast branch
    period, a zero-weight forecast, the realized designation) and, per period label, its first sorted member
    with steps in ``steps_in_use``.  Both empty without a fan."""
    from flextool.engine.preprocess.params._branch import steps_in_use_lf
    pb = period_branch_pairs_lf(workdir, provider=provider).collect()
    pairs = {(str(a), str(b)) for a, b in zip(pb["d"].to_list(), pb["b"].to_list())}
    labels = fan_period_labels(pairs)
    if not labels:
        return set(), {}
    with_steps = {str(x) for x in steps_in_use_lf(workdir, provider=provider).collect()["d"].to_list()}
    members = {b for a, b in pairs if a != b and a in labels}
    first: dict[str, str] = {}
    for a in sorted(labels):
        stepped = sorted(b for x, b in pairs if x == a and b != a and b in with_steps)
        if stepped:
            first[a] = stepped[0]
    return members, first


def load_co2_rolling_carry(sd: Path, *, provider: "object | None" = None):
    """``(p_co2_f_d_k, p_co2_cum_realized_tonnes, p_co2_cum_realized_represented, horizon_fraction_reached,
    p_co2_hours_before_solve)`` for the CO2 cap rolling / chained carry (#38, Q22, item 1042).

    ``p_co2_f_d_k`` — the raw ``solve_data/f_d_k.csv`` Param ``(d,)``: the same per-period "share of
    d's hours reached by the end of this roll's horizon" the commodity ladder's annual tier cap uses
    (``_commodity_ladder._eq_tier_cap_annual``), read straight off the Provider — the file is always
    emitted — rather than through ``flex_data.p_f_d_k``, which stays ``None`` without a price-ladder
    commodity (``p_f_d_k_from_source`` gates on that). ``None`` when the file is absent or empty;
    consumers then treat every period as fully reached (1.0), the pre-fix behaviour.

    ``p_co2_cum_realized_tonnes`` — tonnes already realized per ``(group, period)``, net of removal
    credits (#19): the ``cumulative_co2`` handoff carrier (``preprocess/emit/_co2_accumulators.py``).
    ``None`` on the first solve of a chain (no prior carrier); consumers then treat realized as 0.

    ``p_co2_cum_realized_represented`` — the carrier's ``represented`` column: the same tonnes ×
    years_represented[d] / period_share[d] of the solve that realized them (co2_max_total's LHS units);
    null values where the carrier has none.  ``None`` exactly when ``p_co2_cum_realized_tonnes`` is.

    ``horizon_fraction_reached`` — ``Σ_d years_represented[d]·f_d_k[d] / Σ_d years_represented[d]``,
    over every period of the whole chain, each counted once: a stochastic fan's members are left out and its
    period label stands for the period (the f_d_k of its first member with steps: the realized share plus this
    solve's share; in a split realization roll the label holds the jump, which that f_d_k counts) (``solve_data/p_years_represented.csv``, the solve's static period→year map, not dt-pruned — the same un-pruned source the ladder's cumulative tier cap reads
    for its own carried weight, ``p_ladder_cum_weight``). The fraction of the chain's total represented
    years reached by the end of this roll — scales ``co2_max_total``'s fixed whole-horizon budget down
    to "allocated so far". ``1.0`` (the old, full-cap-every-roll behaviour) outside rolling / chained
    solves, or when either CSV is unavailable (a fixture that skips the derived-param pass).  A period that the
    solves of a chain split counts its share reached of the hours known so far (:func:`split_f_d_k`).

    ``p_co2_hours_before_solve`` — ``(d,)`` the hours of the solve's periods that earlier solves of the chain
    realized (``solve_data/hours_realized_by_earlier_solves``, ``emit/_period_calc``); ``None`` without such a
    period (every single solve and every rolling solve that holds its periods alone).
    """
    if provider is None:
        return None, None, None, 1.0, None
    from flextool.engine.preprocess import _provider_keys as K
    from flextool.engine.preprocess._provider_translators import read_handoff_frame

    fdk_df = None
    if provider.has("solve_data/f_d_k"):
        raw = provider.get("solve_data/f_d_k")
        if raw is not None and raw.height > 0 and {"period", "value"}.issubset(raw.columns):
            fdk_df = (raw.pipe(rename_to_axis, {"period": "d"})
                      .with_columns(pl.col("value").cast(pl.Float64, strict=False))
                      .select("d", "value"))
            if fdk_df.height == 0:
                fdk_df = None
    p_fdk = Param(("d",), fdk_df) if fdk_df is not None else None
    before = _hours_before_solve(provider)

    realized_raw = read_handoff_frame(provider, K.HANDOFF_CUMULATIVE_CO2)
    p_realized = None
    p_realized_rep = None
    if realized_raw is not None and realized_raw.height > 0:
        realized_df = (realized_raw.pipe(rename_to_axis, {"group": "g", "period": "d"})
                       .with_columns(pl.col("value").cast(pl.Float64, strict=False)))
        if realized_df.height > 0:
            p_realized = Param(("g", "d"), realized_df.select("g", "d", "value"))
            rep = (realized_df.with_columns(pl.col("represented").cast(pl.Float64, strict=False))
                   if "represented" in realized_df.columns
                   else realized_df.with_columns(represented=pl.lit(None, dtype=pl.Float64)))
            p_realized_rep = Param(("g", "d"), rep.select("g", "d", value=pl.col("represented")))

    fraction_reached = 1.0
    if provider.has("solve_data/p_years_represented"):
        pyr = provider.get("solve_data/p_years_represented")
        if pyr is not None and pyr.height > 0 and {"period", "p_years_represented"}.issubset(pyr.columns):
            # Each calendar period counts once.  The members of a stochastic fan are alternatives of their
            # period, not more calendar, so they are in neither sum; the period label (the original name of
            # the period, no dispatch of its own; in a split realization roll it holds the jump, which its first
            # member's f_d_k counts) stands for the period: its share reached is the f_d_k of a
            # member with steps, which holds the label's realized hours from earlier solves plus this
            # solve's hours (every forecast has the same steps).  A label without such a member keeps its
            # own f_d_k (its realized share).
            members, first_member = _fan_members(sd.parent, provider)
            years = (pyr.pipe(rename_to_axis, {"period": "d"})
                     .filter(~pl.col("d").cast(pl.Utf8).is_in(sorted(members)))
                     .with_columns(pl.col("p_years_represented").cast(pl.Float64, strict=False))
                     .group_by("d").agg(pl.col("p_years_represented").sum().alias("years")))
            total_years = float(years["years"].sum())
            if total_years > 0.0 and fdk_df is not None:
                fdk_frame = fdk_df if before is None else split_f_d_k(fdk_df, before.frame, provider)
                fdk_of = {str(a): float(v) for a, v in zip(fdk_frame["d"].to_list(), fdk_frame["value"].to_list())
                          if v is not None}
                eff = fdk_frame.with_columns(pl.col("d").cast(pl.Utf8)).select("d", pl.col("value").alias("fdk"))
                if first_member:
                    eff = eff.with_columns(pl.col("d").replace_strict(
                        {a: fdk_of.get(b, 0.0) for a, b in first_member.items()}, default=pl.col("fdk"),
                        return_dtype=pl.Float64).alias("fdk"))
                reached = (years.with_columns(pl.col("d").cast(pl.Utf8)).join(eff, on="d", how="left")
                           .with_columns(pl.col("fdk").fill_null(0.0)))
                fraction_reached = min(
                    float((reached["years"] * reached["fdk"]).sum()) / total_years, 1.0)
    return p_fdk, p_realized, p_realized_rep, fraction_reached, before
