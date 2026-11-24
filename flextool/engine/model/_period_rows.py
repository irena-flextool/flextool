"""Period-level rows and branch weights under a realization roll: the checks G1 and G2 of ``build_flextool``.

Under a split realization roll the base period ``p`` (a period label with steps) dispatches its jump and each branch
period ``p_f`` the steps after it.  A *period-level* row (one per period, summing every step of the period) built
per period name then holds only a fragment of the path ``p`` + ``p_f``.  G1 makes such a row an internal error
instead of a different LP (default-deny):

* a row on a label with steps of a ``path``, ``path_at_step``, ``invest``, ``rp`` or unclassified period-level
  family;
* a row on a branch period of such a label of a ``path`` / ``path_at_step`` family that the path helper has not
  converted (``_Build.path_converted``).

A family is period-level when its ``over`` has ``d`` and no step column, or when ``PERIOD_ROW_KINDS`` names it,
whatever its columns.  G2: every (d, t) cell of the LP has a branch weight > 0.  Without a label with steps (every
solve but a split realization roll) G1 does nothing and G2 holds.
"""
from __future__ import annotations

import polars as pl

from flextool.engine.preprocess._step_relations import labels_with_steps, not_path_aware

PATH, PATH_AT_STEP, FRAGMENT, INVEST, RP = "path", "path_at_step", "fragment", "invest", "rp"
RP_PREFIX = "rp_inter_period_"   # every rp_inter_period_* family is kind ``rp`` (refused in rolls)
STEP_COLUMNS = frozenset({"t", "b_first"})

_INVEST_NAMES = (
    "capacityMargin_slack_energy",
    "divestedBuilt_n", "divestedBuilt_p",
    "extendedCapacity_n", "extendedCapacity_p",
    "maxExtend_cohort_n", "maxExtend_cohort_p",
    "non_anticipativity_extend_n", "non_anticipativity_extend_p",
    "maxCumulative_capacity_n", "maxCumulative_capacity_p",
    "minCumulative_capacity_n", "minCumulative_capacity_p",
    "maxInvestGroup_entity_cumulative", "minInvestGroup_entity_cumulative",
    "maxInvestGroup_entity_period_n", "maxInvestGroup_entity_period_p",
    "minInvestGroup_entity_period_n", "minInvestGroup_entity_period_p",
    "maxDivestGroup_entity_period_n", "maxDivestGroup_entity_period_p",
    "minDivestGroup_entity_period_n", "minDivestGroup_entity_period_p",
    "maxInvest_entity_period_n", "maxInvest_entity_period_p",
    "minInvest_entity_period_n", "minInvest_entity_period_p",
    "maxDivest_entity_period_n", "maxDivest_entity_period_p",
    "minDivest_entity_period_n", "minDivest_entity_period_p",
    "maxDivest_cohort_n", "maxDivest_cohort_p",
    "maxDivest_var_bound", "maxDivest_var_bound_n",
    "maxInvest_var_bound", "maxInvest_var_bound_n",
    "maxRetired_existing_n", "maxRetired_existing_p",
    "non_anticipativity_divest_built_n", "non_anticipativity_divest_built_p",
    "non_anticipativity_divest_n", "non_anticipativity_divest_p",
    "non_anticipativity_invest_n", "non_anticipativity_invest_p",
    # rows of invest and capacity terms only (core._eq_user_constraints)
    "process_constraint_equal_period", "process_constraint_less_than_period",
    "process_constraint_greater_than_period",
)

PERIOD_ROW_KINDS: dict[str, str] = {
    "co2_max_period": PATH,
    "ladder_tier_cap_annual_roll": PATH,
    "maxCumulative_flow_period": PATH,
    "minCumulative_flow_period": PATH,
    "nodeBalancePeriod_eq": PATH,
    "process_constraint_equal_period_total": PATH,
    "process_constraint_less_than_period_total": PATH,
    "process_constraint_greater_than_period_total": PATH,
    "node_storage_usage_fix_le": PATH_AT_STEP,   # a period sum anchored at a step (over has t)
    "commodity_ladder_balance": FRAGMENT,        # v_trade[d] buys d's own cells: right per name
    **dict.fromkeys(_INVEST_NAMES, INVEST),
}


def period_row_kind(name: str) -> str | None:
    """The ``PERIOD_ROW_KINDS`` kind of constraint family ``name`` (``rp`` by prefix), None when unclassified."""
    return RP if name.startswith(RP_PREFIX) else PERIOD_ROW_KINDS.get(name)


def _period_level_columns(over) -> bool:
    return over is not None and "d" in over.columns and not STEP_COLUMNS.intersection(over.columns)


def unclassified_period_families(m) -> list[str]:
    """The constraint families of ``m`` whose ``over`` has ``d`` and no step column and that ``PERIOD_ROW_KINDS``
    does not classify (the registry pin; empty when complete)."""
    return [n for n in m.cstr_names()
            if period_row_kind(n) is None and _period_level_columns(m.cstrs_named(n)[0].over)]


def check_period_rows(b) -> None:
    """G1 on the families of ``b.m`` (module docstring).  No-op without a label with steps."""
    lab = labels_with_steps(b.d)
    if not lab:
        return
    invest = getattr(b.d, "period_invest", None)
    if invest is not None and invest.height > 0:
        # S9: a split realization roll is a roll, and rolls build no investment.
        for p in sorted(set(invest["d"].cast(pl.Utf8).to_list()).intersection(lab)):
            raise not_path_aware("period_invest", p, b.solve_name)
    branches = {f for fs in lab.values() for f in fs}
    for name in b.m.cstr_names():
        over = b.m.cstrs_named(name)[0].over
        kind = period_row_kind(name)
        if over is None or "d" not in over.columns or (kind is None and not _period_level_columns(over)):
            continue
        periods = set(over["d"].cast(pl.Utf8).unique().to_list())
        hit = sorted(periods.intersection(lab)) if kind != FRAGMENT else []
        if kind in (PATH, PATH_AT_STEP) and name not in b.path_converted:
            hit += sorted(periods & branches)
        if hit:
            raise not_path_aware(name, hit[0], b.solve_name)


def check_branch_weights(b) -> None:
    """G2: every (d, t) cell of the LP has a ``pdt_branch_weight`` > 0 (a missing cell has none)."""
    w = b.d.pdt_branch_weight
    if w is None:
        return
    cells = (b.d.dt.select(pl.col("d").cast(pl.Utf8), pl.col("t").cast(pl.Utf8))
             .join(w.frame.select(pl.col("d").cast(pl.Utf8), pl.col("t").cast(pl.Utf8), "value"),
                   on=["d", "t"], how="left"))
    bad = cells.filter(~(pl.col("value").fill_nan(None) > 0).fill_null(False))
    if bad.height > 0:
        raise not_path_aware("pdt_branch_weight", bad["d"][0], b.solve_name)
