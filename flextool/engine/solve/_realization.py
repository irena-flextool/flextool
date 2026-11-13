"""Realization of a stochastic fan: which solve realizes the committed window of a stochastic solve.

A stochastic solve optimises over forecasts only; its operational decisions inside a fan have no realized answer of
their own.  Realized operations come from a *realization solve*: a deterministic solve that runs right after the
stochastic solve over the same committed window, with the investments of the stochastic solve fixed and its
non-anticipativity-tied unit commitment and reserves inherited.  The realization solve is either

* engine-spawned (``<solve>__realized_dispatch``): for ``dispatch_realization_method = separate_dispatch``, or for a
  ``realized: yes`` designation in ``stochastic_branches`` whose fan reaches the solve's committed window; or
* author-declared: a deterministic solve that immediately follows the stochastic solve in ``model.solves``, has the
  same solve mode and roll structure, and realizes every committed period of the fan.

Both kinds are registered in ``SolveConfig.realized_dispatch_parent`` (realization solve -> stochastic solve).  This
module holds the config-time predicates that decide the registration and the per-sub-solve
:class:`RealizationPlan`.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Iterable, Mapping

import polars as pl

from flextool.engine.input._solve_state import FlexToolConfigError

from flextool.engine.preprocess.params._branch import fan_period_labels

_MODULE_LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class RealizationPlan:
    """How the fan of one sub-solve is realized.

    ``labels`` are the period labels of the sub-solve: the original names of the periods that open a fan
    (``period__branch`` self-row with at least one forecast member).  A label has no dispatch.
    ``committed_in_fan`` are the ``(period, step)`` cells the sub-solve commits inside its fan (its realized and
    fix-storage cells of the labels before the stochastic expansion), in list order.  ``realization_solve`` is the
    sub-solve that realizes this sub-solve's committed window (the key of ``realized_dispatch_parent`` whose value
    is this sub-solve), or ``None`` when no realization solve exists.
    """

    labels: frozenset[str]
    committed_in_fan: tuple[tuple[str, str], ...]
    realization_solve: str | None


def build_realization_plan(
    solve: str,
    period__branch: Iterable[tuple[str, str]],
    realized_dispatch_parent: Mapping[str, str],
    *,
    committed_in_fan: "pl.DataFrame | None" = None,
) -> RealizationPlan:
    """The :class:`RealizationPlan` of *solve* from its ``period__branch`` rows, its committed cells inside the fan
    (a ``(period, step)`` frame) and the realization registry.

    Per-roll registry entries (``<sibling>_roll_<k> -> <parent>_roll_<k>``) are matched like single-solve ones; when
    several entries name *solve* the first in registry order is taken (one realization solve per sub-solve).
    """
    labels = frozenset(fan_period_labels({(str(d), str(b)) for d, b in period__branch}))
    realization_solve = next((s for s, p in realized_dispatch_parent.items() if p == solve), None)
    cells = tuple() if committed_in_fan is None else tuple(
        (str(p), str(t)) for p, t in zip(committed_in_fan["period"].to_list(), committed_in_fan["step"].to_list()))
    return RealizationPlan(labels=labels, committed_in_fan=cells, realization_solve=realization_solve)


def _branch_rows(solve_cfg: Any, solve: str) -> list:
    rows = (getattr(solve_cfg, "stochastic_branches", None) or {}).get(solve)
    return list(rows or [])


def _period_order(solve_cfg: Any, solve: str) -> list[str]:
    timesets = (getattr(solve_cfg, "timesets_used_by_solves", None) or {}).get(solve) or []
    return [str(p) for p, _ts in timesets]


def _realized_period_names(solve_cfg: Any, solve: str) -> list[str]:
    realized = (getattr(solve_cfg, "realized_periods", None) or {}).get(solve) or []
    return [str(p_in) for _p_from, p_in in realized]


def _committed_from(solve_cfg: Any, solve: str, reveal_periods: set[str]) -> list[str]:
    """Realized periods of *solve* at or after the earliest of *reveal_periods* in the solve's period order.

    A fan opened at a period continues through every later period of the solve, so a realized period at or after
    the reveal period is committed inside that fan.  A reveal period missing from the period order counts only
    itself.
    """
    realized = _realized_period_names(solve_cfg, solve)
    order = _period_order(solve_cfg, solve)
    positions = [order.index(p) for p in reveal_periods if p in order]
    if not positions:
        return [p for p in realized if p in reveal_periods]
    first = min(positions)
    return [p for p in realized if (p in order and order.index(p) >= first) or p in reveal_periods]


def fan_committed_periods(solve_cfg: Any, solve: str) -> list[str]:
    """The realized periods of *solve* that lie inside one of its fans (in realized-period order); [] when
    *solve* declares no ``stochastic_branches``."""
    rows = _branch_rows(solve_cfg, solve)
    if not rows:
        return []
    return _committed_from(solve_cfg, solve, {str(r[0]) for r in rows})


def designation_reaches_committed_window(solve_cfg: Any, solve: str) -> bool:
    """True when a ``realized: yes`` row of *solve*'s ``stochastic_branches`` opens a fan that reaches a committed
    window of the solve.

    A single solve commits its ``realized_periods``; a rolling solve commits each roll's jump, and the jumps of the
    rolls tile the realized periods, so in both modes the fan reaches a committed window when a realized period lies
    at or after the designated row's reveal period.
    """
    for row in _branch_rows(solve_cfg, solve):
        if str(row[3]).strip().lower() != "yes":
            continue
        if _committed_from(solve_cfg, solve, {str(row[0])}):
            return True
    return False


def configured_realization_solve(solve_cfg: Any, parent: str) -> str | None:
    """The author-declared realization solve of *parent*, or ``None``.

    It is the solve right after *parent* in ``model.solves`` when that solve is deterministic (no
    ``stochastic_branches``), contains no solves, has the same ``solve_mode`` and roll structure
    (``rolling_times``) as *parent*, and realizes every committed period of *parent*'s fans.  A *parent* without a
    fan reaching a committed window has no realization solve.
    """
    committed = fan_committed_periods(solve_cfg, parent)
    if not committed:
        return None
    model_solve = getattr(solve_cfg, "model_solve", None) or {}
    follower = None
    for solves in model_solve.values():
        if parent in solves:
            idx = list(solves).index(parent)
            if idx + 1 < len(solves):
                follower = solves[idx + 1]
            break
    if follower is None or _branch_rows(solve_cfg, follower):
        return None
    contains = getattr(solve_cfg, "contains_solves", None) or {}
    if contains.get(follower):
        return None
    modes = getattr(solve_cfg, "solve_modes", None) or {}
    if modes.get(follower, "single_solve") != modes.get(parent, "single_solve"):
        return None
    rolling = getattr(solve_cfg, "rolling_times", None) or {}
    if list(rolling.get(follower) or []) != list(rolling.get(parent) or []):
        return None
    if not set(committed) <= set(_realized_period_names(solve_cfg, follower)):
        return None
    return follower


def lookahead_in_configured_realization(solve_cfg: Any, realization_solve: str, parent: str) -> str | None:
    """The warning for an author-declared *realization_solve* of the single solve *parent* that holds a period
    *parent* does not realize; ``None`` when it holds none (or *parent* rolls: each realization roll is trimmed to
    its jump).

    An engine-added realization solve of a single solve dispatches the realized periods only (item 255).  An
    author-declared one keeps the periods its author gave it, so a period beyond the parent's realized window is
    dispatched with the realized data: information the parent did not have when it decided.
    """
    if (getattr(solve_cfg, "solve_modes", None) or {}).get(parent, "single_solve") == "rolling_window":
        return None
    realized = set(_realized_period_names(solve_cfg, parent))
    extra = [p for p in dict.fromkeys(_period_order(solve_cfg, realization_solve)) if p not in realized]
    if not extra:
        return None
    return (f"Realization solve '{realization_solve}' of solve '{parent}' also dispatches "
            f"{', '.join(repr(p) for p in extra)}, which '{parent}' does not realize: it dispatches "
            f"{'that period' if len(extra) == 1 else 'those periods'} with the realized data, information "
            f"'{parent}' did not have when it decided.  Leave {'it' if len(extra) == 1 else 'them'} out of "
            f"'{realization_solve}' unless this is intended.")


def warn_recourse_designation(solve: str) -> None:
    """Warn that a ``realized: yes`` designation of a recourse-investment solve gets no realization solve.

    Under recourse every forecast branch invests on its own after the reveal, so there is no realized post-reveal
    investment to fix in a realization solve.  The designation still selects the branch a deterministic solve
    reads stochastic data from.
    """
    _MODULE_LOGGER.warning(
        "Solve '%s' has a realized: yes designation in stochastic_branches, but it uses "
        "stochastic_invest_method = recourse: no realization solve is added (there is no realized post-reveal "
        "investment to fix).  The designation only selects the branch deterministic solves read stochastic data "
        "from.", solve)


def _utf8(frame):
    return frame.with_columns(pl.col(c).cast(pl.Utf8) for c in frame.columns if c != "value")


def _rows(frame) -> bool:
    return frame is not None and frame.height > 0


def _frame(param):
    if param is None:
        return None
    frame = param.frame if hasattr(param, "frame") else param
    return _utf8(frame) if frame.height > 0 else None


def _join_value(base: pl.DataFrame, param, name: str, default: float) -> pl.DataFrame:
    """*base* with column *name* = *param*'s value joined on the key columns both share (*default* where absent)."""
    frame = _frame(param)
    if frame is None:
        return base.with_columns(pl.lit(default).alias(name))
    keys = [c for c in frame.columns if c != "value" and c in base.columns]
    if not keys:
        return base.with_columns(pl.lit(default).alias(name))
    vals = frame.select(*keys, pl.col("value").cast(pl.Float64).alias(name)).unique(subset=keys, keep="first")
    return base.join(vals, on=keys, how="left").with_columns(pl.col(name).fill_null(default))


def check_pinned_realization_feasible(d, solve: str, *, rtol: float = 1e-6) -> None:
    """Raise when a pinned realization solve cannot balance a node without balance slack.

    A realization solve pins units online (``p_fix_online``) and reserves (``p_fix_reserve``) to values the
    stochastic solve chose for its forecasts; the realized data was not among them.  On a node with
    ``penalty_method = off`` (no balance slack) the realized net load may then exceed what the supply into the node
    can deliver, and the LP is infeasible.  This check finds that case before the solve and names the node and step.

    Per hard-balance node and step: the realized need ``−inflow`` (MWh) is compared with an upper bound of the
    supply into the node, Σ over the arcs into the node of ``unitsize · Δt · (cap · availability · profile −
    reserve_up)``, where ``cap`` is the pinned online count × the sink's capacity_max_coeff for a pinned process and
    the existing capacity (units of unitsize) × the arc's capacity factor otherwise, ``profile`` the realized upper
    or fixed flow profile (1 without one) and ``reserve_up`` the pinned upward reserve the process holds at the node.
    A node is not checked when the bound is not certain: it has a storage state, a block or coarse balance, a
    connection, an arc without capacity (conversion_flow_coeff = 0) or an arc without an existing-capacity value.
    """
    off = _frame(getattr(d, "nodeBalance_penalty_off", None))
    fix_online = _frame(getattr(d, "p_fix_online", None))
    fix_reserve = _frame(getattr(d, "p_fix_reserve", None))
    flow_to_n = _frame(getattr(d, "flow_to_n", None))
    inflow = _frame(getattr(d, "p_inflow", None))
    if off is None or (fix_online is None and fix_reserve is None) or flow_to_n is None or inflow is None:
        return
    nodes = set(off["n"].to_list())
    for skip in ("nodeState", "nodeStateBlock", "nodeCoarse"):
        frame = _frame(getattr(d, skip, None))
        if frame is not None and "n" in frame.columns:
            nodes -= set(frame["n"].to_list())
    units = _frame(getattr(d, "process_unit", None))
    unit_names = set(units["p"].to_list()) if units is not None else set()
    arcs_all = pl.concat([f.select("p", "n") for f in (flow_to_n, _frame(getattr(d, "flow_from_n", None)))
                          if f is not None])
    nodes -= set(arcs_all.filter(~pl.col("p").is_in(list(unit_names)))["n"].to_list())
    uncapped = _frame(getattr(d, "process_source_sink_uncapped", None))
    arcs = flow_to_n.filter(pl.col("n").is_in(list(nodes))).select("p", "source", "sink", "n").unique()
    if uncapped is not None:
        nodes -= set(arcs.join(uncapped.select("p", "source", "sink"), on=["p", "source", "sink"],
                               how="semi")["n"].to_list())
    arcs = arcs.filter(pl.col("n").is_in(list(nodes)))
    if arcs.height == 0:
        return
    dt = _utf8(d.dt.select("d", "t")).unique()
    cells = arcs.join(dt, how="cross")
    cells = _join_value(cells, d.p_unitsize, "unitsize", 1.0)
    cells = _join_value(cells, getattr(d, "p_process_availability", None), "avail", 1.0)
    cells = _join_value(cells, d.p_step_duration, "dur", 1.0)
    cells = _join_value(cells, getattr(d, "p_process_sink_max_capacity_coef", None), "k_sink", 1.0)
    cells = _join_value(cells, getattr(d, "p_arc_max_cap_coef", None), "k_arc", 1.0)
    existing = _frame(getattr(d, "p_flow_upper_existing", None))
    if existing is None:
        return
    cells = cells.join(existing.select("p", "source", "sink", "d", pl.col("value").alias("existing")),
                       on=["p", "source", "sink", "d"], how="left")
    if fix_online is not None:
        cells = cells.join(fix_online.select("p", "d", "t", pl.col("value").alias("online")),
                           on=["p", "d", "t"], how="left")
    else:
        cells = cells.with_columns(pl.lit(None, dtype=pl.Float64).alias("online"))
    profile_value = _frame(getattr(d, "p_profile_value", None))
    profiles = [f.select("p", "source", "sink", "f") for f in (_frame(getattr(d, "process_profile_upper", None)),
                                                               _frame(getattr(d, "process_profile_fixed", None)))
                if f is not None]
    if profiles and profile_value is not None:
        arc_f = pl.concat(profiles).unique()
        pv_keys = [c for c in ("f", "d", "t") if c in profile_value.columns]
        prof = (arcs.join(arc_f, on=["p", "source", "sink"], how="inner").join(dt, how="cross")
                .join(profile_value.select(*pv_keys, pl.col("value").cast(pl.Float64).alias("profile")),
                      on=pv_keys, how="inner")
                .group_by("p", "source", "sink", "d", "t").agg(pl.col("profile").min()))
        cells = cells.join(prof, on=["p", "source", "sink", "d", "t"], how="left")
    else:
        cells = cells.with_columns(pl.lit(None, dtype=pl.Float64).alias("profile"))
    if fix_reserve is not None:
        res_up = (fix_reserve.filter(pl.col("ud") == "up")
                  .group_by("p", "n", "d", "t").agg(pl.col("value").sum().alias("reserve_up")))
        cells = cells.join(res_up, on=["p", "n", "d", "t"], how="left")
    else:
        cells = cells.with_columns(pl.lit(None, dtype=pl.Float64).alias("reserve_up"))
    unknown = cells.filter(pl.col("online").is_null() & pl.col("existing").is_null())
    nodes_unknown = set(unknown["n"].to_list())
    cells = cells.filter(~pl.col("n").is_in(list(nodes_unknown)))
    if cells.height == 0:
        return
    cap = (pl.when(pl.col("online").is_not_null()).then(pl.col("online") * pl.col("k_sink"))
           .otherwise(pl.col("existing") * pl.col("k_arc")))
    supply = (cells
              .with_columns(mwh=pl.col("unitsize") * pl.col("dur")
                            * (cap * pl.col("avail") * pl.col("profile").fill_null(1.0)
                               - pl.col("reserve_up").fill_null(0.0)))
              .group_by("n", "d", "t").agg(pl.col("mwh").sum().alias("supply")))
    need = inflow.select("n", "d", "t", (-pl.col("value").cast(pl.Float64)).alias("need"))
    short = (supply.join(need, on=["n", "d", "t"], how="inner")
             .filter(pl.col("need") > pl.col("supply") + rtol * pl.max_horizontal(pl.lit(1.0), pl.col("need").abs()))
             .sort("n", "d", "t"))
    if short.height == 0:
        return
    n, period, step, sup, req = short.row(0)
    raise FlexToolConfigError(
        f"Realization solve '{solve}' cannot balance node '{n}' (penalty_method = off) at period '{period}', step "
        f"'{step}': the realized net load needs {req:.6g} MWh but the units online and the reserves pinned from the "
        f"stochastic solve leave at most {sup:.6g} MWh of supply ({short.height} step(s) affected).  Give the node "
        "balance slack (penalty_method = regular), add forecasts that cover the realized conditions, or end the "
        "committed window before the reveal.")


def _untied_families(d, committed: "list[tuple[str, str]]") -> list[str]:
    """The families of *d* (a sub-solve's FlexData) whose decisions over the *committed* fan cells are not tied
    across the forecasts, so that the committed window has no realized value for them."""
    from flextool.engine.model._commodity_ladder import _INF_SENTINEL

    families: list[str] = []
    if _rows(d.nodeState):
        families.append("storage state")
    if any(_rows(getattr(d, f"process_source_sink_ramp_limit_{side}_{ud}", None))
           for side in ("sink", "source") for ud in ("up", "down")):
        families.append("ramp-limited flows")
    tied = set()
    if _rows(d.dt_non_anticipativity):
        tied = {(str(p), str(t)) for p, t in d.dt_non_anticipativity.select("d", "t").iter_rows()}
    untied_cells = [c for c in committed if c not in tied]
    # Units online, like reserves, are tied only over the tied steps.
    if untied_cells and _rows(d.process_online):
        families.append("units online")
    if untied_cells and _rows(d.process_reserve_upDown_node_active):
        families.append("reserves")
    qty = d.p_ladder_cum_quantity
    if _rows(d.ci_ladder_cumulative) and qty is not None and qty.frame.filter(
            pl.col("value") < _INF_SENTINEL).height > 0:
        families.append("cumulative commodity ladder")
    if _rows(d.group_co2_max_total):
        families.append("CO2 total cap")
    committed_labels = {str(p) for p, _t in committed}
    members = set()
    if _rows(d.period_branch_full):
        members = {str(b) for a, b in d.period_branch_full.select("d", "b").iter_rows()
                   if str(a) in committed_labels and str(a) != str(b)}
    fan_periods = sorted(committed_labels | members)
    # The annual ladder tiers and the CO2 period caps continue from the period's realized purchases / emissions
    # (the period label's carry), which a committed window inside the fan leaves per forecast.  Only a cap on the
    # fan's own period (its label or a forecast of it) is affected; a cap on another period of the chain is not.
    ann_qty = d.p_ladder_ann_quantity
    if _rows(d.cndi_ladder_ann) and ann_qty is not None and d.cndi_ladder_ann.join(
            ann_qty.frame.filter(pl.col("value") < _INF_SENTINEL).select("c", "i", "d"),
            on=["c", "i", "d"], how="inner").filter(pl.col("d").cast(pl.Utf8).is_in(fan_periods)).height > 0:
        families.append("annual commodity ladder")
    if _rows(d.group_d_co2_capped) and d.group_d_co2_capped.filter(
            pl.col("d").cast(pl.Utf8).is_in(fan_periods)).height > 0:
        families.append("CO2 period cap")
    if getattr(d, "recourse_invest", False):
        invest_d = set()
        for frame in (d.pd_invest_set, d.nd_invest_set):
            if _rows(frame):
                invest_d |= {str(x) for x in frame["d"].to_list()}
        if members & invest_d:
            families.append("recourse investment")
    return families


def check_committed_window_realized(d, solve: str, plan: "RealizationPlan | None", has_later_solve: bool) -> None:
    """Raise when *solve* commits cells inside a forecasts-only fan that no solve realizes.

    A committed window that reaches into a fan is realized for the families tied across the forecasts (units
    online and reserves on the tied steps); every other family the next solve continues from — storage state,
    ramp-limited flows, the cumulative commodity ladder, the CO2 total cap, the finite annual commodity ladder
    tiers, the CO2 period caps, recourse investment, and units online or reserves outside the tied steps — is
    decided per forecast there.  Without a realization solve (a
    ``realized: yes`` designation or a configured one) nothing realizes it, so a later solve would continue from
    an arbitrary forecast.  Called on the built FlexData before the LP.
    """
    if plan is None or not plan.committed_in_fan or plan.realization_solve is not None or not has_later_solve:
        return
    families = _untied_families(d, list(plan.committed_in_fan))
    if not families:
        return
    periods = sorted({p for p, _t in plan.committed_in_fan})
    raise FlexToolConfigError(
        f"Solve '{solve}' commits {len(plan.committed_in_fan)} step(s) inside a forecasts-only stochastic fan "
        f"(period(s) {', '.join(periods)}) and a later solve continues from them, but {', '.join(families)} "
        "is decided per forecast there and no solve realizes it.  Designate a realized: yes branch in "
        "stochastic_branches, configure a realization solve right after it, or end the committed window before "
        "the reveal.")
