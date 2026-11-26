"""The apply boundary of the lifetime costs: :func:`apply_npv` (a Spine solve) and
:func:`apply_run_fields_of_synthetic_solve` (a roll or a nested sub-solve) set the investment price, the salvage and
fixed-cost terms, the life-extension costs and the run-level fields of the reported costs on ``flex_data``, from the
modules that derive them (``_npv``, ``_run_costs``, …).
"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import polars as pl
from polar_high import Param

from flextool.engine.input._axis_enums import alias_to_axis
from flextool.engine.preprocess.params._discount import (
    _solve_inflation_scalars,
    notice_discount_from_run_start,
    p_inflation_op_from_source,
    run_years,
    solve_year_rows,
)
from flextool.engine.preprocess.params._existing_salvage import apply_existing_replacement, apply_retirement_salvage
from flextool.engine.preprocess.params._extend import apply_extension_costs
from flextool.engine.preprocess.params._fixed_costs import apply_retirement_fixed_cost, p_ed_fixed_cost_from_source
from flextool.engine.preprocess.params._npv import apply_invest_lumps
from flextool.engine.preprocess.params._retire_settlement import apply_built_retirement_costs
from flextool.engine.preprocess.params._run_costs import (
    _run_lumps_are_the_solves,
    apply_pre_existing_of_reported_costs,
    apply_run_cost_fields,
    apply_run_factors,
)

if TYPE_CHECKING:
    from flextool.engine.input._input_source import InputSource


def apply_run_fields_of_synthetic_solve(flex_data: object,
                                        source: "InputSource",
                                        workdir: Path,
                                        *,
                                        provider: "object | None" = None,
                                        ) -> None:
    """The investment lumps and the run-level fields of a synthetic sub-solve (a roll of a rolling solve, a nested
    ``<solve>_<period>`` sub-solve), for which ``apply_npv`` does not run.

    A sub-solve with an invest or a divest set gets its investment lumps from :func:`._npv.apply_invest_lumps`, its
    salvage terms from :func:`._existing_salvage.apply_retirement_salvage` and the fixed cost its retirements save from
    :func:`._fixed_costs.apply_retirement_fixed_cost` over its own periods, as a Spine solve does (no CSV emitter
    derives them; the rolls and nested sub-solves of the test fixtures have no divest set); its own discount factors
    stay the loaded ``solve_data`` frames.  Without run years the run fields are the sub-solve's own fields."""
    from flextool.engine.preprocess.params._branch import annuity_window_lineage_df
    from flextool.engine.preprocess.params._derived import (
        _expand_invest_branch_periods,
        _period_in_use_set,
        _periodAll_from_source,
        _provider_has_key,
        _provider_read,
        _read_active_solve,
        _read_period_with_history,
        _solve_periods,
    )
    active_solve = _read_active_solve(workdir, provider=provider)
    period_in_use = list(_period_in_use_set(source, active_solve, workdir, provider=provider))
    apply_pre_existing_of_reported_costs(flex_data, source, active_solve, period_in_use, workdir, provider=provider)
    years = run_years(source, active_solve, period_in_use, workdir, provider=provider)
    seed_inv = None
    path = Path(workdir) / "solve_data" / "p_inflation_factor_investment_yearly.csv"
    if years is None and _provider_has_key(provider, path):
        df = _provider_read(provider, path)
        if df.height > 0 and "period" in df.columns:
            col = "value" if "value" in df.columns else df.columns[-1]
            seed = (df.lazy()
                      .select(pl.col("period").cast(pl.Utf8).alias("d"),
                              pl.col(col).cast(pl.Float64, strict=False).alias("value"))
                      .filter(pl.col("d").is_in([str(d) for d in period_in_use]))
                      .select(alias_to_axis("d", "d"), "value").sort("d").collect())
            seed_inv = Param(("d",), seed) if seed.height > 0 else None
    apply_run_factors(flex_data, source, active_solve, years, seed_inv)
    if active_solve is not None:
        apply_existing_replacement(flex_data, source, active_solve, period_in_use,
                                   lineage=annuity_window_lineage_df(workdir, source, active_solve, provider=provider),
                                   workdir=workdir, provider=provider)
    ed_invest = getattr(flex_data, "ed_invest_set", None)
    ed_divest = getattr(flex_data, "ed_divest_set", None)
    if ((ed_invest is None or ed_invest.height == 0) and (ed_divest is None or ed_divest.height == 0)) \
            or active_solve is None:
        if years is None:
            _run_lumps_are_the_solves(flex_data)
        return
    pi = getattr(flex_data, "period_invest", None)
    period_invest = [] if pi is None or pi.height == 0 else [str(d) for d in pi[pi.columns[0]].to_list()]
    period_with_history = _read_period_with_history(workdir, provider=provider) or list(period_in_use)
    lineage = annuity_window_lineage_df(workdir, source, active_solve, provider=provider)
    # The decision periods as apply_npv and ed_invest_set read them (the sub-solve's own invest periods).
    decision_periods = _expand_invest_branch_periods(
        _solve_periods(source, active_solve, "invest_periods"), workdir, provider=provider) or []
    period_universe = list(_periodAll_from_source(source, active_solve, workdir=workdir, provider=provider))
    year_rows = solve_year_rows(workdir, provider, source, active_solve, period_universe)
    if year_rows is not None:
        notice_discount_from_run_start(active_solve, _solve_inflation_scalars(source)[0])
    if ed_invest is not None and ed_invest.height > 0:
        apply_invest_lumps(flex_data, source, active_solve, decision_periods, period_in_use, period_universe,
                           period_with_history, lineage=lineage, year_rows=year_rows, provider=provider)
    if ed_divest is not None and ed_divest.height > 0:
        apply_retirement_salvage(flex_data, source, active_solve, decision_periods, period_in_use, period_universe,
                                 period_with_history, lineage=lineage, year_rows=year_rows, workdir=workdir,
                                 provider=provider)
        apply_retirement_fixed_cost(flex_data, source, active_solve, decision_periods, period_in_use,
                                    period_universe, lineage=lineage, year_rows=year_rows, workdir=workdir,
                                    provider=provider)
    apply_built_retirement_costs(flex_data, source, active_solve, period_in_use, period_universe,
                                 period_with_history, lineage=lineage, year_rows=year_rows, workdir=workdir,
                                 provider=provider)
    apply_run_cost_fields(flex_data, source, active_solve, period_in_use, period_invest, workdir, lineage=lineage,
                          years=years, provider=provider)


# ---------------------------------------------------------------------------
# Apply boundary
# ---------------------------------------------------------------------------


def apply_npv(flex_data: object,
                  source: "InputSource",
                  workdir: Path,
                  *,
                  provider: "object | None" = None) -> None:
    """Wire the cluster-A NPV helpers into ``flex_data`` (mutates in place).

    Order:
      1. ``p_inflation_op``                 (depends on years_represented)
      2. ``p_ed_fixed_cost``                (depends on period_with_history)
      3. ``ed_entity_annual_discounted``    (depends on inv_factor + ed_invest)
      4. ``ed_divest_salvage`` / ``ed_existing_salvage_credit``
                                            (depends on inv_factor + ed_divest; in apply_invest_lumps)
      5. ``ed_lifetime_fixed_cost``         (depends on ops_factor + period_with_history)
      6. ``ed_lifetime_fixed_cost_divest``  (depends on inv_factor + ed_divest)

    Next to 1 the run-level factors of the reported costs (:func:`._run_costs.apply_run_factors`), after 6 the run-level
    investment lump and cost lines (:func:`._run_costs.apply_run_cost_fields`); no model term reads them.

    Δ.12b — assignment is unconditional (no silent fall-through to a
    seed value).  Each helper raises on hard errors and returns
    ``None`` only when the upstream feature is genuinely inactive
    (no_invest short-circuit, no period_with_history, etc.) — that
    ``None`` is the explicit "field stays default" signal.
    """
    from flextool.engine.preprocess.params._derived import (
        _read_active_solve, _solve_periods, _period_in_use_set,
        _periodAll_from_source, _read_period_with_history,
        _expand_invest_branch_periods, _enter_recourse_anchor_scope,
        _recourse_branch_axis_present,
    )

    # Slice D §4 — arm the per-period anchor-map + walker-Provider scope for
    # the annuity / lifetime / fixed-cost resolution below (no-op flag-off).
    _enter_recourse_anchor_scope(workdir, provider=provider)
    active_solve = _read_active_solve(workdir, provider=provider)
    # Lineage of the four NPV window walks: each calendar period counted once along a lineage, under every
    # invest method (recourse: the branch's own path; none: the period label's bookkeeping path, not every
    # forecast copy of the same calendar years).  Built on demand from the provider:
    # ``flex_data.dd_same_scenario`` is populated only by ``apply_branch_cluster`` (derived_g), after
    # apply_npv.  None without a fan (deterministic / rolling) -> walks unfiltered.
    from flextool.engine.preprocess.params._branch import (
        annuity_window_lineage_df,
        assert_recourse_npv_preconditions,
    )
    if _recourse_branch_axis_present():
        assert_recourse_npv_preconditions(
            workdir, source, active_solve, provider=provider)
    _npv_lineage = annuity_window_lineage_df(
        workdir, source, active_solve, provider=provider)
    period_in_use = _period_in_use_set(source, active_solve, workdir, provider=provider)
    period_universe = _periodAll_from_source(source, active_solve, workdir=workdir, provider=provider)
    period_invest = _expand_invest_branch_periods(
        _solve_periods(source, active_solve, "invest_periods"),
        workdir, provider=provider) or []
    period_with_history = (_read_period_with_history(workdir, provider=provider)
                              or list(period_in_use))
    # The solve's years from the start of the run, when its own count does not give them (a solve after the run
    # start, fractional years): every factor below, the objective's and the walks', discounts from the run start.
    year_rows = solve_year_rows(workdir, provider, source, active_solve, list(period_universe))
    if year_rows is not None:
        notice_discount_from_run_start(active_solve, _solve_inflation_scalars(source)[0])

    # 1. p_inflation_op (None == no inflation data → field stays default).
    flex_data.p_inflation_op = p_inflation_op_from_source(
        source, active_solve, period_in_use, period_universe,
        provider=provider, year_rows=year_rows)
    # Reported costs: the discount factors from the start of the run (outputs only).
    run_yrs = run_years(source, active_solve, list(period_in_use), workdir, provider=provider)
    apply_run_factors(flex_data, source, active_solve, run_yrs,
                      None if run_yrs is not None else p_inflation_op_from_source(
                          source, active_solve, period_in_use, period_universe, provider=provider,
                          factor="inv_factor", year_rows=year_rows))

    apply_pre_existing_of_reported_costs(flex_data, source, active_solve, list(period_in_use), workdir,
                                         provider=provider)
    if active_solve is not None:
        apply_existing_replacement(flex_data, source, active_solve, list(period_in_use), lineage=_npv_lineage,
                                   workdir=workdir, provider=provider)

    # 2. p_ed_fixed_cost — None == no period_with_history (single-solve
    #    fixtures with no historical periods); legitimate "no data" outcome.
    flex_data.p_ed_fixed_cost = p_ed_fixed_cost_from_source(
        source, period_with_history)

    # 3-6. NPV / lifetime cascade family.  Mirrors the "no_invest"
    # short-circuit in :mod:`._derived.ed_entity_annual_family_from_source`
    # and :func:`flextool.engine.preprocess.flex_data._load_features._load_invest`: when
    # neither ed_invest nor ed_divest carries any row, the cluster
    # outputs are all None (the LP has no v_invest / v_divest
    # variables that would consume them).
    ed_invest = getattr(flex_data, "ed_invest_set", None)
    ed_divest = getattr(flex_data, "ed_divest_set", None)
    built = getattr(flex_data, "cd_divest_built", None)
    extend = getattr(flex_data, "cd_extend", None)
    no_invest = (
        (ed_invest is None or ed_invest.height == 0)
        and (ed_divest is None or ed_divest.height == 0)
        and (built is None or built.height == 0)
        and (extend is None or extend.height == 0)
    )
    if no_invest or active_solve is None:
        if run_yrs is None:
            _run_lumps_are_the_solves(flex_data)
        return

    apply_invest_lumps(flex_data, source, active_solve, list(period_invest), list(period_in_use),
                       list(period_universe), list(period_with_history), lineage=_npv_lineage,
                       year_rows=year_rows, provider=provider)
    apply_extension_costs(flex_data, source, active_solve, list(period_in_use), list(period_universe),
                          list(period_with_history), lineage=_npv_lineage, year_rows=year_rows, workdir=workdir,
                          provider=provider)
    apply_retirement_salvage(flex_data, source, active_solve, list(period_invest), list(period_in_use),
                             list(period_universe), list(period_with_history), lineage=_npv_lineage,
                             year_rows=year_rows, workdir=workdir, provider=provider)
    apply_built_retirement_costs(flex_data, source, active_solve, list(period_in_use), list(period_universe),
                                 list(period_with_history), lineage=_npv_lineage, year_rows=year_rows,
                                 workdir=workdir, provider=provider)

    apply_retirement_fixed_cost(flex_data, source, active_solve, list(period_invest), list(period_in_use),
                                list(period_universe), lineage=_npv_lineage, year_rows=year_rows, workdir=workdir,
                                provider=provider)

    # Reported costs: the investment lump from the start of the run over the run's periods and the cost lines
    # (outputs only).
    apply_run_cost_fields(flex_data, source, active_solve, list(period_in_use), list(period_invest), workdir,
                          lineage=_npv_lineage, years=run_yrs,
                          provider=provider)


__all__ = [
    "apply_npv",
    "apply_run_fields_of_synthetic_solve",
]
