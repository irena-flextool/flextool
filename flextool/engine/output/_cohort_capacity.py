"""The capacity of each cohort of a solve, per period: the per-solve part of the capacity-by-cohort tables.

At the per-roll write hook (:func:`flextool.engine.output._writer.write_outputs_for_solve`), while the solution is
live, :func:`write_cohort_capacity` evaluates :func:`flextool.engine.preprocess.params._cohorts.cohort_capacity`
over the solve's periods in use (look-ahead and forecast-branch periods included) and writes
``output_raw/cohort_capacity__<solve>.parquet``: one row per cohort (plant) and period with its capacity, the
capacity retired so far and the part retired at the start of the period, the fan label of the period (the period
itself outside a stochastic fan) and the period's branch weight (1 without one).  :func:`read_cohort_capacity`
unions the parts: each label period from the last solve (in solve order) that holds it, as the capacity tables
take each period from the last solve that writes it.
"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import polars as pl

if TYPE_CHECKING:
    from polar_high import Solution

PREFIX = "cohort_capacity"


def _decision_mw(sol: "Solution", flex_data, name: str, keys: tuple[str, ...]) -> pl.DataFrame:
    """``[e, *keys, mw]`` (labels Utf8) of the decision variable ``name`` + ``_p`` / ``_n`` times the unit size of
    its process / storage node; empty without the variable."""
    from flextool.engine.preprocess.params._cohorts import as_names

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
    """The cohort capacity of a solved solve (:func:`._cohorts.cohort_capacity`) with its fan ``label`` and branch
    ``weight`` columns: its own cohort table, or the run's existing cohorts for a solve without one; None without
    a Provider or without the existing capacity of the solve."""
    from flextool.engine.preprocess import _provider_keys as K
    from flextool.engine.preprocess._provider_translators import read_handoff_frame
    from flextool.engine.preprocess.emit._chain_params import solve_run_years
    from flextool.engine.preprocess.params._cohorts import (
        KIND_EXISTING,
        as_names,
        cohort_capacity,
        in_service,
        run_existing_frames,
    )
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
