"""``co2_max_total`` needs a solve that sees every period of the model (D54).

``co2_max_total`` is one limit over all the periods of the model.  A solve allots it by the share of the model's
periods that it reaches (``horizon_fraction_reached``) and subtracts what earlier solves emitted.  A solve that does
not hold every period cannot know how much of the limit belongs to the periods it lacks: an early solve may take the
whole limit and a later one is left a negative remainder (infeasible), or the limit is shared at random.  So a run
in which a solve with a total CO2 limit does not see every period of the model stops before anything solves.

What a solve sees: the periods of its solve instance, look-ahead included.  A roll sees the periods of its rolling
solve instance (the union of its rolls' periods), and a realized-dispatch sibling those of its parent.  A nested
solve is an instance per roll of its parent (each sees its own window), and each solve of a 2-D period map is a solve
of its own.  The periods come from the expanded solve tree, before the stochastic pass names the forecast branch
periods, so a fan's branches never appear: its period stands for them.  The model's periods are the union of what
every solve sees, without a period that represents no years (``years_represented`` 0 in every solve that lists it,
e.g. an initial period): the total limit gives it no weight (``_co2_carry``).  The solves of a chain that each hold a
part of the same period all see that period, so they pass.
"""
from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence

Instance = tuple[str, "str | None"]


def solve_period_views(all_solves: Sequence[str], complete_solve: Mapping[str, str],
                       parent_roll: Mapping[str, "str | None"], sub_solve_periods: Mapping[str, Iterable[str]],
                       realized_dispatch_parent: Mapping[str, str]) -> dict[Instance, set[str]]:
    """``{(complete solve, parent roll): the periods it sees}`` in run order, one entry per solve instance (the
    rolls of one instance share their complete solve and their parent's roll; module docstring)."""
    def instance(s: str) -> Instance:
        return complete_solve.get(s, s), parent_roll.get(s)

    views: dict[Instance, set[str]] = {}
    for s in all_solves:
        views.setdefault(instance(s), set()).update(str(p) for p in sub_solve_periods.get(s, ()))
    for sibling, parent in realized_dispatch_parent.items():
        sib, par = instance(sibling), instance(parent)
        if sib in views and par in views:
            views[sib] |= views[par]
    return views


def weightless_periods(years_represented: Mapping[str, Iterable[tuple[str, float]]]) -> set[str]:
    """The periods that represent no years: listed with ``years_represented`` 0 by every solve that lists them."""
    weight: dict[str, float] = {}
    for rows in years_represented.values():
        for period, years in rows or ():
            weight[str(period)] = max(weight.get(str(period), 0.0), float(years))
    return {p for p, w in weight.items() if w == 0.0}


def check_co2_total_sees_all_periods(views: Mapping[Instance, set[str]], groups: Sequence[str],
                                     weightless: "set[str] | frozenset[str]" = frozenset()) -> None:
    """Raise :class:`FlexToolConfigError` when *groups* (the groups with a ``co2_max_total`` row) is not empty and
    a solve instance of *views* (:func:`solve_period_views`) does not see every period of the model, the
    *weightless* periods (:func:`weightless_periods`) left out."""
    if not groups or not views:
        return
    from flextool.engine.input._solve_state import FlexToolConfigError

    model = set().union(*views.values()) - set(weightless)
    short: dict[str, dict["str | None", list[str]]] = {}
    for (solve, parent), seen in views.items():
        if model - seen:
            short.setdefault(solve, {})[parent] = sorted(model - seen)
    if not short:
        return
    solve, by_parent = next(iter(short.items()))
    lists = {tuple(m) for m in by_parent.values()}
    if len(lists) == 1:
        missing = f"does not have period(s) {_names(next(iter(lists)))}"
    else:
        missing = "does not have, " + "; ".join(
            f"under '{parent}', period(s) {_names(m)}" for parent, m in by_parent.items())
    more = len(short) - 1
    others = f" (and {more} more {'solve' if more == 1 else 'solves'})" if more else ""
    raise FlexToolConfigError(
        f"co2_max_total of group(s) {_names(groups)} limits the emissions over all the periods of the model, but "
        f"solve '{solve}'{others} {missing}: it cannot tell how much of the limit belongs to the periods it lacks, "
        "so the results could be wrong. Use co2_max_period (a limit per period) instead, or give every solve that "
        "has the group all the periods of the model (period_timeset, look-ahead periods included).")


def _names(items: Iterable[str]) -> str:
    return ", ".join(repr(str(x)) for x in items)


def co2_total_groups(provider, source) -> list[str]:
    """The groups that get a ``co2_max_total`` row: a ``co2_method`` that caps the total, a finite non-zero
    ``co2_max_total``, and an emitter: an eff / noEff arc drawing from a node of the group that carries a CO2-bearing
    commodity, or a start-up draw at such a node.  The gate of the row's loader (``flex_data._load_co2_cap_total``
    with :func:`~flextool.engine.preprocess.flex_data.co2_capped_emitter_arcs` and ``_startup_co2_sets``), over the
    model's processes (the run's input frames and its input database reader *source*)."""
    if provider is None or source is None or not provider.has("solve_data/group_co2_max_total"):
        return []
    import polars as pl

    from flextool.engine.input._axis_enums import rename_to_axis
    from flextool.engine.preprocess.flex_data import co2_capped_emitter_arcs
    from flextool.engine.preprocess.flex_data._load_co2 import _startup_co2_sets
    from flextool.engine.preprocess.params._projection import process_source_sink_canonical

    capped = provider.get("solve_data/group_co2_max_total")
    if capped.height == 0 or not provider.has("input/p_group"):
        return []
    params = provider.get("input/p_group")
    with_value = set()
    for g, par, value in zip(params["group"].to_list(), params["groupParam"].to_list(), params["p_group"].to_list()):
        try:
            v = float(value)
        except (TypeError, ValueError):
            continue
        if str(par) == "co2_max_total" and v != 0.0 and not math.isinf(v):
            with_value.add(str(g))
    capped = capped.pipe(rename_to_axis, {"group": "g"}).filter(pl.col("g").cast(pl.Utf8).is_in(sorted(with_value)))
    if capped.height == 0:
        return []
    from pathlib import Path

    canonical = process_source_sink_canonical(source)
    arcs = {m: canonical.filter(pl.col("method") == m).select("p", "source", "sink").unique()
            for m in ("eff", "noEff")} if canonical.height > 0 else {"eff": None, "noEff": None}
    gcn, eff, noeff = co2_capped_emitter_arcs(capped, Path("input"), Path("solve_data"), arcs["eff"],
                                              arcs["noEff"], provider=provider)
    if gcn is None:
        return []
    emitting = {str(g) for f in (eff, noeff) if f is not None for g in f["g"].to_list()}
    starts = source.startup_arcs()
    if starts is not None and starts.height > 0:
        pn = starts.select(pl.col("unit").alias("p"), pl.col("node").alias("n")).unique()
        rows = _startup_co2_sets(Path("input"), Path("solve_data"), pn, provider=provider)[
            "startup_from_co2_capped_total"]
        if rows is not None:
            emitting |= {str(g) for g in rows["g"].to_list()}
    return sorted(emitting & with_value)
