"""Retirement of pre-existing capacity: the minimum retirements no existing capacity can meet.

Only capacity that existed before the run can be retired (``ed_divest_set`` keeps the periods with pre-existing
capacity, :func:`._derived.retirable_existing_pairs`), and a retirement counts in the periods of its window
``edd_divest_active`` = {(e, d′, d) : yr[d′] ≤ yr[d] < EOL[e]} (``maxRetired_existing_{p,n}`` caps the retired
capacity of each such period at the pre-existing capacity).  A minimum retirement the existing capacity cannot
supply would otherwise reach the solver as an infeasible LP; :func:`check_retirement_minimums` names it instead:

* ``retire_min_period[e, d] > 0`` on a retirement option without pre-existing capacity (none given, or past its
  end of life);
* Σ_{d′ : (e, d′, d) ∈ edd_divest_active} retire_min_period[e, d′] + carried[e, d] > pre_existing[e, d] for a
  period d (the required retirements still counting in d exceed what exists there);
* retire_min_total[e] − prior[e] > :func:`max_retirable` (e), the largest total the existing capacity allows.

``carried[e, d]`` is the capacity earlier solves of a chain retired that still counts in d, ``prior[e]`` all of
it (``p_entity_divested``).
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping

import polars as pl

#: Capacities (MW, MWh) closer than this are equal.
_TOL = 1e-6


def _pairs(frame: "pl.DataFrame | None", cols: tuple[str, str] = ("e", "d")) -> list[tuple[str, str]]:
    if frame is None or frame.height == 0:
        return []
    return [(str(a), str(b)) for a, b in frame.select(pl.col(cols[0]).cast(pl.Utf8),
                                                      pl.col(cols[1]).cast(pl.Utf8)).iter_rows()]


def _ed_values(param) -> dict[tuple[str, str], float]:
    if param is None:
        return {}
    frame = param.frame if hasattr(param, "frame") else param
    if frame.height == 0:
        return {}
    return {(str(e), str(d)): float(v) for e, d, v in frame.select(
        pl.col("e").cast(pl.Utf8), pl.col("d").cast(pl.Utf8), pl.col("value").cast(pl.Float64)).iter_rows()
        if v is not None}


def _e_values(param) -> dict[str, float]:
    if param is None:
        return {}
    frame = param.frame if hasattr(param, "frame") else param
    if frame.height == 0:
        return {}
    return {str(e): float(v) for e, v in frame.select(pl.col("e").cast(pl.Utf8),
                                                      pl.col("value").cast(pl.Float64)).iter_rows()
            if v is not None}


def active_windows(frames: Iterable["pl.DataFrame | None"]) -> dict[tuple[str, str], list[str]]:
    """``{(e, d′): [d, …]}`` of ``edd_divest_active`` / ``ndd_divest_active`` frames (entity column ``p`` / ``n``,
    then ``d_divest``, ``d``): the periods a retirement decided in d′ counts in."""
    parts = [frame.select(pl.col(frame.columns[0]).cast(pl.Utf8).alias("e"), pl.col("d_divest").cast(pl.Utf8),
                          pl.col("d").cast(pl.Utf8))
             for frame in frames if frame is not None and frame.height > 0]
    if not parts:
        return {}
    grouped = pl.concat(parts).group_by("e", "d_divest", maintain_order=True).agg("d")
    return {(e, d_r): list(ds) for e, d_r, ds in grouped.iter_rows()}


def max_retirable(entity: str,
                  windows: Mapping[tuple[str, str], list[str]],
                  pre: Mapping[tuple[str, str], float],
                  carried: Mapping[tuple[str, str], float] | None = None,
                  ) -> tuple[float, str | None]:
    """``(capacity, d′)``: the largest total ``entity`` can retire in the solve, max over its decision periods d′ of
    min over the window of d′ of (pre_existing[e, d] − carried[e, d]) — a retirement counts in every period of its
    window, so the total retired by the last decision is capped by the smallest capacity left over the rest of the
    window.  ``(0.0, None)`` without decision periods."""
    carried = carried or {}
    best, best_d = 0.0, None
    for (e, d_r), window in windows.items():
        if e != entity or not window:
            continue
        cap = max(min(pre.get((e, d), 0.0) - carried.get((e, d), 0.0) for d in window), 0.0)
        if best_d is None or cap > best:
            best, best_d = cap, d_r
    return best, best_d


def _eol_phrase(entity: str, eol: Mapping[str, float]) -> str:
    return (f"; its existing capacity ends at year {eol[entity]:g} of the run (lifetime)" if entity in eol else "")


def check_retirement_minimums(*,
                              offered: "pl.DataFrame | None",
                              kept: "pl.DataFrame | None",
                              period_method_pairs: "pl.DataFrame | None",
                              total_method_entities: Iterable[str],
                              windows: Mapping[tuple[str, str], list[str]],
                              pre: Mapping[tuple[str, str], float],
                              min_period,
                              min_total,
                              prior,
                              carried: Mapping[tuple[str, str], float] | None,
                              eol: Mapping[str, float],
                              ) -> None:
    """Raise :class:`FlexToolConfigError` for a minimum retirement the pre-existing capacity cannot supply (the
    three conditions of the module docstring), naming the entity, the period and the existing capacity.

    ``offered`` / ``kept`` are the retirement options ``[e, d]`` before and after keeping only those with
    pre-existing capacity; ``period_method_pairs`` the offered pairs of entities with a per-period retirement
    limit (``retire_min_period`` applies); ``total_method_entities`` those with a total limit; ``windows`` is
    :func:`active_windows`; ``pre`` the pre-existing capacity ``{(e, d): MW}``; ``min_period`` /
    ``min_total`` / ``prior`` the Params ``ed_divest_min_period`` / ``e_divest_min_total`` /
    ``p_entity_divested``; ``eol`` the end-of-life year of the bounded-lifetime entities."""
    from flextool.engine.input._solve_state import FlexToolConfigError
    carried = carried or {}
    mins = {k: v for k, v in _ed_values(min_period).items() if v > _TOL}
    kept_set = set(_pairs(kept))
    errors: list[str] = []
    for e, d in sorted(set(_pairs(period_method_pairs)) - kept_set):
        if (e, d) in mins:
            errors.append(f"{e} must retire at least {mins[(e, d)]:g} (retire_min_period) in {d}, but it has no "
                          f"existing capacity there (only capacity that existed before the run can be retired)"
                          + _eol_phrase(e, eol))
    needed: dict[tuple[str, str], float] = {}
    period_method = set(_pairs(period_method_pairs))
    for (e, d_r), window in windows.items():
        m = mins.get((e, d_r), 0.0)
        if m <= _TOL or (e, d_r) not in period_method:
            continue
        for d in window:
            needed[(e, d)] = needed.get((e, d), 0.0) + m
    for (e, d), need in sorted(needed.items()):
        have = pre.get((e, d), 0.0) - carried.get((e, d), 0.0)
        if need > have + _TOL:
            errors.append(f"{e} must have retired at least {need:g} (retire_min_period of this and earlier periods) "
                          f"by {d}, more than its existing capacity there ({have:g})" + _eol_phrase(e, eol))
    totals = _e_values(min_total)
    prior_e = _e_values(prior)
    offered_e = {e for e, _ in _pairs(offered)}
    for e in sorted(set(total_method_entities) & offered_e):
        required = totals.get(e, 0.0) - prior_e.get(e, 0.0)
        if required <= _TOL:
            continue
        cap, at = max_retirable(e, windows, pre, carried)
        if required > cap + _TOL:
            where = f" (at most {cap:g}, retiring in {at})" if at is not None else " (it has no existing capacity "\
                "in the retirement periods)"
            errors.append(f"{e} must retire at least {required:g} (retire_min_total) in this solve, more than its "
                          f"existing capacity allows{where}" + _eol_phrase(e, eol))
    if errors:
        raise FlexToolConfigError("Minimum retirement cannot be met: " + "; ".join(errors) + ".")


def check_group_retirement_minimums(*,
                                    group_entity: "pl.DataFrame | None",
                                    min_total,
                                    min_period,
                                    windows: Mapping[tuple[str, str], list[str]],
                                    pre: Mapping[tuple[str, str], float],
                                    prior,
                                    carried: Mapping[tuple[str, str], float] | None,
                                    ) -> None:
    """Raise :class:`FlexToolConfigError` when a group's minimum retirement exceeds what its members' pre-existing
    capacity allows: ``retire_min_total[g] − Σ_{e ∈ g} prior[e] > Σ_{e ∈ g} max_retirable(e)``, or
    ``retire_min_period[g, d] > Σ_{e ∈ g} (pre_existing[e, d] − carried[e, d])``.  ``group_entity`` is ``[g, e]``,
    ``min_total`` / ``min_period`` the group Params ``[g, value]`` / ``[g, d, value]``."""
    from flextool.engine.input._solve_state import FlexToolConfigError
    if group_entity is None or group_entity.height == 0:
        return
    members: dict[str, list[str]] = {}
    for g, e in group_entity.select(pl.col("g").cast(pl.Utf8), pl.col("e").cast(pl.Utf8)).iter_rows():
        members.setdefault(g, []).append(e)
    carried = carried or {}
    prior_e = _e_values(prior)
    errors: list[str] = []
    totals = {} if min_total is None else {str(g): float(v) for g, v in min_total.frame.select(
        pl.col("g").cast(pl.Utf8), pl.col("value").cast(pl.Float64)).iter_rows() if v is not None}
    for g, need in sorted(totals.items()):
        required = need - sum(prior_e.get(e, 0.0) for e in members.get(g, []))
        cap = sum(max_retirable(e, windows, pre, carried)[0] for e in members.get(g, []))
        if required > cap + _TOL:
            errors.append(f"group {g} must retire at least {required:g} (retire_min_total) in this solve, more than "
                          f"the existing capacity of its members allows ({cap:g})")
    if min_period is not None:
        for g, d, need in min_period.frame.select(pl.col("g").cast(pl.Utf8), pl.col("d").cast(pl.Utf8),
                                                  pl.col("value").cast(pl.Float64)).iter_rows():
            if need is None or need <= _TOL or g not in members:
                continue
            have = sum(pre.get((e, d), 0.0) - carried.get((e, d), 0.0) for e in members[g])
            if need > have + _TOL:
                errors.append(f"group {g} must retire at least {need:g} (retire_min_period) in {d}, more than the "
                              f"existing capacity of its members there ({have:g})")
    if errors:
        raise FlexToolConfigError("Minimum retirement cannot be met: " + "; ".join(errors) + ".")


def ed_values(param) -> dict[tuple[str, str], float]:
    """``{(e, d): value}`` of an ``[e, d, value]`` Param (or frame)."""
    return _ed_values(param)


def retired_carried(rows: Iterable[tuple[str, str, float]],
                    years: Mapping[str, float],
                    periods: Iterable[str],
                    expired: "set[tuple[str, str]]",
                    ) -> dict[tuple[str, str], float]:
    """``{(e, d): MW}`` of the retirements earlier solves of a chain realized (``rows``: entity, decision period
    d′, capacity) that still count in the solve's periods: Σ_{d′ : yr[d′] ≤ yr[d], (e, d) not past the end of
    life of e's pre-existing capacity} retired[e, d′] (``expired``: the (e, d) pairs past it).  A decision period
    without a year in ``years`` lies before the solve (its history), so it counts."""
    from flextool.engine.preprocess.params._run_calendar import YEAR_TOL
    retired = (pl.DataFrame([(str(e), str(d_r), float(v or 0.0)) for e, d_r, v in rows],
                            schema={"e": pl.Utf8, "d_r": pl.Utf8, "v": pl.Float64}, orient="row")
                 .with_row_index("_row").filter(pl.col("v") != 0.0))
    periods = [str(d) for d in periods]
    if retired.height == 0 or not periods:
        return {}
    decided = pl.DataFrame({"d_r": [str(d) for d in years], "y_r": [float(y) for y in years.values()]},
                           schema={"d_r": pl.Utf8, "y_r": pl.Float64})
    counted = pl.DataFrame({"d": periods, "y": [float(years.get(d, 0.0)) for d in periods]},
                           schema={"d": pl.Utf8, "y": pl.Float64}).with_row_index("_col")
    past = pl.DataFrame(sorted((str(e), str(d)) for e, d in expired), schema={"e": pl.Utf8, "d": pl.Utf8},
                        orient="row")
    out = (retired.join(decided, on="d_r", how="left")
                  .join(counted, how="cross")
                  .filter(pl.col("y_r").is_null() | (pl.col("y_r") <= pl.col("y") + YEAR_TOL))
                  .join(past, on=["e", "d"], how="anti")
                  .sort("_row", "_col")  # the rows' order, then the periods' (the sums add in that order)
                  .group_by("e", "d", maintain_order=True).agg(pl.col("v").sum()))
    return {(e, d): v for e, d, v in out.iter_rows()}


__all__ = ["active_windows", "check_group_retirement_minimums", "check_retirement_minimums", "ed_values",
           "max_retirable", "retired_carried"]
