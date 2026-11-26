"""``co2_max_total`` needs a solve that sees every period of the model (D54).

``co2_max_total`` is one limit over all the periods of the model.  A solve allots it by the share of the model's
periods that it reaches (``horizon_fraction_reached``) and subtracts what earlier solves emitted.  A solve that does
not hold every period cannot know how much of the limit belongs to the periods it lacks: an early solve may take the
whole limit and a later one is left a negative remainder (infeasible), or the limit is shared at random.  So a run
in which a solve with a total CO2 limit does not see every period of the model stops before anything solves.

What a solve sees: the periods of its solve instance, look-ahead included.  A roll sees the periods of its rolling
solve instance (the union of its rolls' periods), and a realized-dispatch sibling those of its parent.  A nested
solve is an instance per roll of its parent (each sees its own window), and each solve of a 2-D period map is a solve
of its own.  A
forecast branch period counts as its period.  The model's periods are the union of what every solve sees.  The
solves of a chain that each hold a part of the same period all see that period, so they pass.
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence


def solve_period_views(all_solves: Sequence[str], complete_solve: Mapping[str, str],
                       parent_roll: Mapping[str, "str | None"],
                       sub_solve_periods: Mapping[str, Iterable[str]],
                       period__branch_lists: Mapping[str, Iterable[tuple[str, str]]],
                       realized_dispatch_parent: Mapping[str, str]) -> dict[tuple[str, "str | None"], set[str]]:
    """``{(complete solve, parent roll): the periods it sees}`` in run order, one entry per solve instance (the
    rolls of one instance share their complete solve and their parent's roll; module docstring)."""
    anchor: dict[str, str] = {}
    for pairs in period__branch_lists.values():
        for period, branch in pairs or ():
            if branch != period:
                anchor[str(branch)] = str(period)
    def instance(s: str) -> tuple[str, "str | None"]:
        return complete_solve.get(s, s), parent_roll.get(s)

    views: dict[tuple[str, "str | None"], set[str]] = {}
    for s in all_solves:
        views.setdefault(instance(s), set()).update(
            anchor.get(str(p), str(p)) for p in sub_solve_periods.get(s, ()))
    for sibling, parent in realized_dispatch_parent.items():
        sib, par = instance(sibling), instance(parent)
        if sib in views and par in views:
            views[sib] |= views[par]
    return views


def check_co2_total_sees_all_periods(views: Mapping[tuple[str, "str | None"], set[str]], groups: Sequence[str]) -> None:
    """Raise :class:`FlexToolConfigError` when *groups* (the groups with a ``co2_max_total``) is not empty and a
    solve of *views* (:func:`solve_period_views`) does not see every period of the model."""
    if not groups or not views:
        return
    from flextool.engine.input._solve_state import FlexToolConfigError

    model = set().union(*views.values())
    short: dict[str, list[str]] = {}
    for (solve, _parent), seen in views.items():
        if model - seen and solve not in short:
            short[solve] = sorted(model - seen)
    if not short:
        return
    solve, missing = next(iter(short.items()))
    others = f" (and {len(short) - 1} more solves)" if len(short) > 1 else ""
    raise FlexToolConfigError(
        f"co2_max_total of group(s) {', '.join(repr(g) for g in groups)} limits the emissions over all the periods "
        f"of the model, but solve '{solve}'{others} does not have period(s) {', '.join(repr(p) for p in missing)}: "
        "it cannot tell how much of the limit belongs to the periods it lacks, so the results could be wrong. "
        "Use co2_max_period (a limit per period) instead, or give every solve that has the group all the "
        "periods of the model (period_timeset, look-ahead periods included).")


def co2_total_groups(provider) -> list[str]:
    """The groups whose ``co2_method`` caps the total and whose ``co2_max_total`` is a finite, non-zero value (the
    groups with a ``co2_max_total`` row: a zero or missing value gives none), from the run's input frames."""
    import math

    if provider is None or not provider.has("input/group__co2_method") or not provider.has("input/p_group"):
        return []
    from flextool.engine.preprocess._co2_methods import CO2_MAX_TOTAL_METHODS

    methods = provider.get("input/group__co2_method")
    if methods.height == 0 or {"group", "co2_method"} - set(methods.columns):
        return []
    total = {str(g) for g, m in zip(methods["group"].to_list(), methods["co2_method"].to_list())
             if str(m) in CO2_MAX_TOTAL_METHODS}
    params = provider.get("input/p_group")
    if params.height == 0 or {"group", "groupParam", "p_group"} - set(params.columns):
        return []
    out = set()
    for g, par, value in zip(params["group"].to_list(), params["groupParam"].to_list(), params["p_group"].to_list()):
        if str(par) != "co2_max_total" or str(g) not in total:
            continue
        try:
            v = float(value)
        except (TypeError, ValueError):
            continue
        if v != 0.0 and not math.isinf(v):
            out.add(str(g))
    return sorted(out)
