"""The capacity of each cohort of a solve, per period: the per-solve part of the capacity-by-cohort tables.

At the per-roll write hook (:func:`flextool.engine.output._writer.write_outputs_for_solve`), while the solution is
live, :func:`write_cohort_capacity` evaluates :func:`cohort_capacity` over the solve's periods in use (look-ahead
and forecast-branch periods included) and writes ``output_raw/cohort_capacity__<solve>.parquet``: one row per
cohort (plant) and period with its capacity, the capacity retired so far and the part retired at the start of the
period, the fan label of the period (the period itself outside a stochastic fan) and the period's branch weight (1
without one).  :func:`read_cohort_capacity`
unions the parts: each label period from the last solve (in solve order) that holds it, as the capacity tables
take each period from the last solve that writes it.
"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Iterable, Mapping

import polars as pl

from flextool.engine.preprocess.params._cohorts import (
    KIND_EXISTING,
    KIND_EXTEND,
    KIND_GENERATION,
    KIND_INVEST,
    as_names,
    within_life,
)
from flextool.engine.preprocess.params._run_calendar import YEAR_TOL

if TYPE_CHECKING:
    from polar_high import Solution

PREFIX = "cohort_capacity"
_INF = float("inf")

#: The reported kind of a cohort (:func:`cohort_capacity`): the existing capacity (and the first plant of an existing
#: ``reinvest_automatic`` chain), an investment (and its chain's first plant), a later plant of any chain and a
#: life extension.
REPORTED_KIND_EXISTING = "existing"
REPORTED_KIND_INVESTED = "invested"
REPORTED_KIND_REPLACEMENT = "replacement"
REPORTED_KIND_EXTENSION = "extension"
_REPORTED_ROOT_KIND = {KIND_EXISTING: REPORTED_KIND_EXISTING, KIND_INVEST: REPORTED_KIND_INVESTED,
                       KIND_EXTEND: REPORTED_KIND_EXTENSION}
#: The columns of :func:`cohort_capacity`.
COHORT_CAPACITY_SCHEMA: dict[str, pl.DataType] = {
    "e": pl.Utf8, "cohort": pl.Utf8, "kind": pl.Utf8, "built": pl.Boolean, "d_on": pl.Utf8, "d_eol": pl.Utf8,
    "d": pl.Utf8, "capacity": pl.Float64, "retired": pl.Float64, "retired_in_period": pl.Float64}


def cohort_capacity(cohort: pl.DataFrame,
                    alive: pl.DataFrame,
                    carried: pl.DataFrame | None,
                    carried_retire: pl.DataFrame | None,
                    pre: pl.DataFrame,
                    sizes: pl.DataFrame,
                    retired: pl.DataFrame,
                    years: Mapping[str, float],
                    periods: Iterable[str],
                    ) -> pl.DataFrame:
    """The capacity of every cohort in service in the solve's ``periods`` (MW, MWh for storage;
    :data:`COHORT_CAPACITY_SCHEMA`, labels Utf8)::

        size[c,d]     = pre[e,d]                 c the existing cohort of e, (c, d) ∈ alive (rows with size > 0)
                      = sizes[c]                 c an investment or life extension of the solve, (c, d) ∈ alive
                      = size_c                   c carried from an earlier solve (``realized_cohort``),
                                                 y_on_c ≤ y[d] < y_eol_c
        retired[c,d]  = Σ_{(c, d′, d) ∈ retired} mw  +  Σ_{(c, d′, v) ∈ carried_retire, y[d′] ≤ y[d] or d′ before
                        the solve} v             (retired so far, still counting in d)
        capacity[c,d] = size[c,d] − retired[c,d]

    ``cohort`` / ``alive`` the solve's cohort table and ``cohort_alive`` (the existing cohort and the solve's own
    decisions), ``carried`` / ``carried_retire`` the handoff ``realized_cohort`` / ``realized_retire`` of the earlier
    solves, ``pre`` ``[e, d, mw]`` the existing cohort's capacity (``p_entity_pre_existing``), ``sizes`` ``[e,
    cohort, mw]`` the MW of the solve's investments and life extensions, ``retired`` ``[e, cohort, d_divest, d, mw]``
    the solve's retirements and the periods each counts in (``retire_active`` / ``cdd_divest_built``), ``years`` the
    run years (:func:`._cohorts.cohort_years`).  ``retired_in_period[c,d]`` is the part retired at the start of ``d``.

    A ``reinvest_automatic`` root holds its capacity through its plants (its ``generation`` rows): each period's
    capacity is reported under the plant in service then (``y_on[g] ≤ y[d] < y_eol[g]``), the first plant with the
    root's kind, a later one as ``replacement``; other cohorts under their own row.  ``kind``: existing /
    invested / replacement / extension; ``built`` false for the existing capacity and its chain; ``d_on`` the period
    the cohort (plant) comes online in, ``d_eol`` the period its life ends at (null at or past the horizon end)."""
    periods = [str(d) for d in periods]
    out_empty = pl.DataFrame(schema=COHORT_CAPACITY_SCHEMA)
    if not periods:
        return out_empty
    year = pl.DataFrame({"d": periods, "y": [float(years.get(d, 0.0)) for d in periods]},
                        schema={"d": pl.Utf8, "y": pl.Float64})
    keep = ["e", "cohort", "kind", "root", "d_on", "d_eol", "y_on", "y_eol"]
    own = as_names(cohort.select(keep), ("e",))
    if carried is not None and carried.height > 0:
        theirs = (carried.select(pl.col("entity").alias("e"), "cohort", "kind", "root",
                                 pl.col("period_online").alias("d_on"), pl.col("period_eol").alias("d_eol"),
                                 "y_on", "y_eol", "size")
                         .join(own.select("e", "cohort"), on=["e", "cohort"], how="anti"))
    else:
        theirs = pl.DataFrame(schema={**{c: own.schema[c] for c in keep}, "size": pl.Float64})
    rows_all = pl.concat([own, theirs.select(keep)], how="vertical_relaxed")
    roots = rows_all.filter(pl.col("kind") != KIND_GENERATION)
    gens = rows_all.filter(pl.col("kind") == KIND_GENERATION)

    # The size of each root in each period it is in service.
    on_alive = as_names(alive.select("e", "cohort", "kind", "d"), ("e", "d"))
    existing = (on_alive.filter(pl.col("kind") == KIND_EXISTING)
                        .join(as_names(pre.select("e", "d", "mw"), ("e", "d")), on=["e", "d"], how="inner")
                        .select("e", "cohort", "d", size=pl.col("mw")))
    decided = (on_alive.filter(pl.col("kind") != KIND_EXISTING)
                       .join(as_names(sizes.select("e", "cohort", "mw"), ("e",)), on=["e", "cohort"], how="inner")
                       .select("e", "cohort", "d", size=pl.col("mw")))
    previous = (theirs.filter(pl.col("kind") != KIND_GENERATION)
                      .join(year, how="cross")
                      .filter(within_life(pl.col("y"), pl.col("y_on").fill_null(-_INF),
                                          pl.col("y_eol").fill_null(_INF)))
                      .select("e", "cohort", "d", size=pl.col("size").fill_null(0.0)))
    sized = pl.concat([existing, decided, previous], how="vertical_relaxed").filter(pl.col("size") > 0.0)

    # The capacity retired so far (and in the period) that still counts.
    own_retired = as_names(retired.select("e", "cohort", "d_divest", "d", "mw"), ("e", "d_divest", "d"))
    parts = [own_retired.select("e", "cohort", "d", "mw",
                                now=pl.when(pl.col("d") == pl.col("d_divest")).then(pl.col("mw")).otherwise(0.0))]
    if carried_retire is not None and carried_retire.height > 0:
        decided_year = pl.DataFrame({"d_r": list(years), "y_r": [float(v) for v in years.values()]},
                                    schema={"d_r": pl.Utf8, "y_r": pl.Float64})
        parts.append(carried_retire.select(pl.col("entity").alias("e"), "cohort", pl.col("period").alias("d_r"),
                                           mw=pl.col("value"))
                     .join(decided_year, on="d_r", how="left")
                     .join(year, how="cross")
                     .filter(pl.col("y_r").is_null() | (pl.col("y_r") <= pl.col("y") + YEAR_TOL))
                     .select("e", "cohort", "d", "mw",
                             now=pl.when(pl.col("d") == pl.col("d_r")).then(pl.col("mw")).otherwise(0.0)))
    gone = (pl.concat(parts, how="vertical_relaxed")
              .group_by("e", "cohort", "d").agg(pl.col("mw").sum().alias("retired"), pl.col("now").sum()))
    held = (sized.join(gone, on=["e", "cohort", "d"], how="left")
                 .with_columns(retired=pl.col("retired").fill_null(0.0), now=pl.col("now").fill_null(0.0))
                 .with_columns(capacity=pl.col("size") - pl.col("retired") + 0.0, retired=pl.col("retired") + 0.0,
                               now=pl.col("now") + 0.0)
                 .join(roots.select("e", "cohort", "kind", "d_on", "d_eol"), on=["e", "cohort"], how="inner"))

    # A chained root is reported under the plant in service in each period.
    chained = held.join(gens.select("e", "root").unique(), left_on=["e", "cohort"], right_on=["e", "root"],
                        how="semi")
    plants = (chained.join(year, on="d", how="inner")
                     .join(gens.select("e", "root", pl.col("cohort").alias("plant"), pl.col("d_on").alias("g_on"),
                                       pl.col("d_eol").alias("g_eol"), pl.col("y_on").alias("g_y_on"),
                                       pl.col("y_eol").alias("g_y_eol")),
                           left_on=["e", "cohort"], right_on=["e", "root"], how="inner")
                     .filter((pl.col("y") >= pl.col("g_y_on") - YEAR_TOL)
                             & (pl.col("y") < pl.col("g_y_eol") - YEAR_TOL))
                     .with_columns(kind=pl.when(pl.col("plant").str.ends_with(":1"))
                                          .then(pl.col("kind").replace_strict(_REPORTED_ROOT_KIND))
                                          .otherwise(pl.lit(REPORTED_KIND_REPLACEMENT)),
                                   built=pl.col("kind") != KIND_EXISTING)
                     .select("e", cohort=pl.col("plant"), kind="kind", built="built", d_on=pl.col("g_on"),
                             d_eol=pl.col("g_eol"), d="d", capacity="capacity", retired="retired",
                             retired_in_period="now"))
    single = (held.join(chained.select("e", "cohort").unique(), on=["e", "cohort"], how="anti")
                  .select("e", "cohort", kind=pl.col("kind").replace_strict(_REPORTED_ROOT_KIND),
                          built=pl.col("kind") != KIND_EXISTING, d_on="d_on", d_eol="d_eol", d="d",
                          capacity="capacity", retired="retired", retired_in_period="now"))
    out = pl.concat([single, plants], how="vertical_relaxed").select(
        *(pl.col(c).cast(t) for c, t in COHORT_CAPACITY_SCHEMA.items()))
    order = {d: i for i, d in enumerate(periods)}
    return (out.with_columns(_o=pl.col("d").replace_strict(order, default=len(order), return_dtype=pl.Int64))
               .sort("e", "_o", "d_on", "cohort").drop("_o"))


def _decision_mw(sol: "Solution", flex_data, name: str, keys: tuple[str, ...]) -> pl.DataFrame:
    """``[e, *keys, mw]`` (labels Utf8) of the decision variable ``name`` + ``_p`` / ``_n`` times the unit size of
    its process / storage node; empty without the variable."""
    frames = []
    for axis, size in (("p", "p_unitsize"), ("n", "p_state_unitsize")):
        var = f"{name}_{axis}"
        param = getattr(flex_data, size, None)
        if var not in getattr(sol, "_vars", {}) or param is None:
            continue
        values = as_names(sol.value(var), (axis, *keys)).rename({axis: "e"})
        us = as_names(param.frame.select(axis, "value"), (axis,)).rename({axis: "e", "value": "us"})
        frames.append(values.join(us, on="e", how="inner").select("e", *keys, mw=pl.col("value") * pl.col("us")))
    if not frames:
        return pl.DataFrame(schema={"e": pl.Utf8, **{k: pl.Utf8 for k in keys}, "mw": pl.Float64})
    return pl.concat(frames, how="vertical_relaxed")


def solve_cohort_capacity(sol: "Solution", flex_data, provider) -> "pl.DataFrame | None":
    """The cohort capacity of a solved solve (:func:`cohort_capacity`) with its fan ``label`` and branch
    ``weight`` columns: its own cohort table, or the run's existing cohorts for a solve without one; None without
    a Provider or without the existing capacity of the solve."""
    from flextool.engine.preprocess import _provider_keys as K
    from flextool.engine.preprocess._provider_translators import read_handoff_frame
    from flextool.engine.preprocess.emit._chain_params import solve_run_years
    from flextool.engine.preprocess.params._cohorts import in_service, run_existing_frames
    from flextool.engine.preprocess.params._extend import extension_id_expr
    from flextool.process_outputs._fan_labels import fan_branch_labels

    pre_param = getattr(flex_data, "p_entity_pre_existing", None) if flex_data is not None else None
    if sol is None or provider is None or pre_param is None:
        return None
    years = solve_run_years(provider)
    periods = as_names(flex_data.p_period_share.frame.select("d"), ("d",))["d"].to_list()
    cohort, alive = getattr(flex_data, "cohort", None), getattr(flex_data, "cohort_alive", None)
    if cohort is None or alive is None:
        # A solve without investment or retirement (a dispatch solve): the run's existing cohorts and the carried
        # ones.
        frames = run_existing_frames(provider, years, periods)
        if frames is None:
            return None
        cohort, alive = frames

    invested = (_decision_mw(sol, flex_data, "v_invest", ("d",))
                .select("e", cohort=pl.concat_str(pl.lit("invest:"), pl.col("d")), mw="mw"))
    extended = (_decision_mw(sol, flex_data, "v_extend", ("cohort", "d"))
                .select("e", cohort=extension_id_expr("d", "cohort"), mw="mw"))
    sizes = pl.concat([invested, extended], how="vertical_relaxed")

    existing_of = as_names(cohort.filter(pl.col("kind") == KIND_EXISTING).select("e", "cohort"), ("e",))
    frames = in_service(flex_data)
    reach = [as_names(frames.retired(axis), (axis, "d_divest", "d")).rename({axis: "e"}) for axis in ("p", "n")]
    divest = (_decision_mw(sol, flex_data, "v_divest", ("d",)).rename({"d": "d_divest"})
              .join(pl.concat(reach, how="vertical_relaxed"), on=["e", "d_divest"], how="inner")
              .join(existing_of, on="e", how="inner")
              .select("e", "cohort", "d_divest", "d", "mw"))
    counting = getattr(flex_data, "cdd_divest_built", None)
    parts = [divest]
    if counting is not None and counting.height > 0:
        parts.append(_decision_mw(sol, flex_data, "v_divest_built", ("cohort", "d")).rename({"d": "d_divest"})
                     .join(as_names(counting.select("e", "cohort", "d_divest", "d"), ("e", "d_divest", "d")),
                           on=["e", "cohort", "d_divest"], how="inner")
                     .select("e", "cohort", "d_divest", "d", "mw"))
    retired = pl.concat(parts, how="vertical_relaxed")

    pre = as_names(pre_param.frame.select("e", "d", "value"), ("e", "d")).rename({"value": "mw"})
    out = cohort_capacity(cohort, alive, read_handoff_frame(provider, K.HANDOFF_REALIZED_COHORT),
                          read_handoff_frame(provider, K.HANDOFF_REALIZED_RETIRE), pre, sizes, retired, years,
                          periods)
    labels = fan_branch_labels(flex_data)
    weights = getattr(flex_data, "pd_branch_weight", None)
    out = out.with_columns(label=pl.col("d").replace(labels) if labels else pl.col("d"))
    if weights is not None:
        w = as_names(weights.frame.select("d", "value"), ("d",)).rename({"value": "weight"})
        out = out.join(w, on="d", how="left").with_columns(weight=pl.col("weight").fill_null(1.0))
    else:
        out = out.with_columns(weight=pl.lit(1.0))
    return out


def write_cohort_capacity(sol: "Solution", flex_data, *, output_dir: Path, solve_name: str,
                          provider) -> "pl.DataFrame | None":
    """Evaluate and persist the cohort capacity of one solve (module docstring); returns the frame with a leading
    ``solve`` column, or None without one."""
    frame = solve_cohort_capacity(sol, flex_data, provider)
    if frame is None:
        return None
    frame = frame.select(pl.lit(solve_name, dtype=pl.Utf8).alias("solve"), pl.all())
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    frame.write_parquet(output_dir / f"{PREFIX}__{solve_name}.parquet")
    return frame


def read_cohort_capacity(output_dir: Path, solve_order: "dict[str, int] | None" = None) -> "pl.DataFrame | None":
    """The per-solve parts of ``output_dir`` unioned: the rows of each label period from the last solve (in solve
    order, ``solve_order``) that holds it; None without a part."""
    from flextool.engine.output._capacity_margin import read_capacity_margin_parts

    frame = read_capacity_margin_parts(output_dir, PREFIX, solve_order)
    if frame is None:
        return None
    solves = frame["solve"].unique(maintain_order=True).to_list()  # already in solve order
    ranked = frame.with_columns(_rank=pl.col("solve").replace_strict({s: i for i, s in enumerate(solves)},
                                                                     return_dtype=pl.Int64))
    last = ranked.group_by("label").agg(pl.col("_rank").max().alias("_last"))
    return (ranked.join(last, on="label", how="inner").filter(pl.col("_rank") == pl.col("_last"))
                  .drop("_rank", "_last"))
