""":func:`build_flextool`: the build steps of the FlexTool LP in emit order (the table of contents)."""

from flextool.engine.model import _commodity_ladder
from flextool.engine.model import _cumulative_invest
from flextool.engine.model import _dc_power_flow
from flextool.engine.model import _exact
from flextool.engine.model import _group_slack
from flextool.engine.model import _non_anticipativity
from flextool.engine.model import _online
from flextool.engine.model import _period_rows
from flextool.engine.model import _reserve
from flextool.engine.model import _extend
from flextool.engine.model import _retire_built
from flextool.engine.model import _unit_floors
from flextool.engine.preprocess._step_relations import check_step_relations
from flextool.engine.solve import _objective_options
from flextool.engine.model.core._context import _Build
from flextool.engine.model.core._prepare import _prepare
from flextool.engine.model.core._variables import _declare_variables, _shared_expressions
from flextool.engine.model.core._rp_storage import _eq_rp_chain_ends, _eq_rp_storage_chain
from flextool.engine.model.core._node_balance import (
    _eq_block_coupling,
    _eq_node_balance,
    _eq_node_balance_coarse,
    _eq_node_balance_fix_quantity,
    _eq_node_balance_period,
    _eq_storage_usage_fix,
)
from flextool.engine.model.core._flows import _eq_max_flow, _eq_max_output_sum
from flextool.engine.model.core._ramp import _eq_ramp, _eq_ramp_cost
from flextool.engine.model.core._invest import (
    _eq_invest_max_units,
    _eq_invest_period_caps,
    _eq_invest_total_caps,
    _eq_retired_existing,
)
from flextool.engine.model.core._conversion import _eq_conversion, _eq_conversion_piecewise, _eq_min_input_share
from flextool.engine.model.core._co2 import _eq_co2_max_period, _eq_co2_max_total
from flextool.engine.model.core._user_constraints import _eq_user_constraints
from flextool.engine.model.core._storage_state import (
    _eq_max_state,
    _eq_state_profiles,
    _eq_storage_end_binding,
    _eq_storage_end_match,
    _eq_storage_reference_value,
    _eq_storage_roll_end,
    _eq_storage_split_path_end,
    _eq_storage_start_binding_cyclic,
    _eq_storage_state_fan_start,
    _eq_storage_within_solve_path_end,
)
from flextool.engine.model.core._profiles import _eq_profiles
from flextool.engine.model.core._objective import (
    _obj_balance_slack,
    _obj_co2_price,
    _obj_co2_price_noEff,
    _obj_commodity,
    _obj_existing_fixed_cost,
    _obj_existing_replacement,
    _obj_existing_salvage,
    _obj_invest_divest,
    _obj_ramp_cost,
    _obj_startup_cost,
    _obj_storage_reference_price,
    _obj_variable_cost,
)


def _build_prof(label: str) -> None:
    """Env-gated (FLEXTOOL_BUILD_PROFILE=1) RSS print to stderr. No-op otherwise."""
    import os
    import sys
    if os.environ.get("FLEXTOOL_BUILD_PROFILE") != "1":
        return
    try:
        with open("/proc/self/status", encoding="utf-8") as _f:
            for _ln in _f:
                if _ln.startswith("VmRSS:"):
                    sys.stderr.write(f"[build profile] step={label}\trss_gb={int(_ln.split()[1])/(1024*1024):.3f}\n")
                    sys.stderr.flush()
                    break
    except Exception:
        pass


def build_flextool(m, d, *, include_existing_fixed_cost: bool = False,
                   scale_the_objective: float = 1.0, solve_name: str | None = None) -> None:
    """Build the FlexTool LP into ``m`` from the data ``d``.

    Every block is conditional on its data: a feature with an empty index
    set emits nothing; a feature whose index set is populated but whose
    required fields are ``None`` raises ``ValueError`` (requirement tuples
    above).

    ``include_existing_fixed_cost``: add the fixed cost of existing
    capacity, Σ existing · fixed_cost · inflation_op · pd_branch_weight, to
    the objective as a constant (OBJ. existing_fixed_cost).  It changes the
    objective value, not the solution.  Off by default.
    ``scale_the_objective``: multiplier on the whole objective, constant
    included (1.0 = none).  Declared to polar-high
    (``Problem.set_objective_scale``), which then reads objective-valued
    solver options (targets, bounds, cutoffs, absolute gaps) in unscaled
    currency.
    ``solve_name``: the solve this LP is for; named in the configuration errors and warnings of the build.
    """
    b = _Build(m=m, d=d, include_existing_fixed_cost=include_existing_fixed_cost,
               scale_the_objective=scale_the_objective, solve_name=solve_name)
    # Objective steps return their term (or None); terms are summed in this
    # order.  Constraint steps return None.
    obj = None
    for step in (
        _prepare,                        # flags, data checks, index sets
        _declare_variables,
        _shared_expressions,
        # ── constraints, in emit order ──
        _eq_rp_storage_chain,
        _eq_rp_chain_ends,
        _eq_node_balance,
        _eq_node_balance_coarse,
        _eq_node_balance_period,
        _eq_node_balance_fix_quantity,
        _eq_storage_usage_fix,
        _eq_block_coupling,
        _eq_max_flow,
        _eq_max_output_sum,
        _dc_power_flow.add_constraints,
        _online.add_constraints,         # incl. online_fix_eq
        _exact.add_constraints,
        _unit_floors.add_constraints,
        _eq_ramp,
        _eq_ramp_cost,
        _eq_invest_max_units,
        _eq_invest_period_caps,
        _eq_invest_total_caps,
        _eq_retired_existing,
        _retire_built.add_constraints,
        _extend.add_constraints,
        _eq_conversion,
        _eq_conversion_piecewise,
        _eq_min_input_share,
        _eq_co2_max_period,
        _eq_co2_max_total,
        _eq_user_constraints,
        _eq_max_state,
        _eq_state_profiles,
        _eq_storage_start_binding_cyclic,
        _eq_storage_within_solve_path_end,
        _eq_storage_split_path_end,
        _eq_storage_state_fan_start,
        _eq_storage_end_binding,
        _eq_storage_roll_end,
        _eq_storage_end_match,
        _eq_storage_reference_value,
        _eq_profiles,
        # ── core objective terms ──
        _obj_balance_slack,
        _obj_commodity,
        _obj_co2_price,
        _obj_co2_price_noEff,
        _obj_variable_cost,
        _obj_startup_cost,
        _obj_ramp_cost,
        _obj_storage_reference_price,
        _obj_invest_divest,
        _obj_existing_fixed_cost,
        _obj_existing_salvage,
        _obj_existing_replacement,
        # ── further constraint families and their objective terms ──
        _group_slack.add_constraints,
        _group_slack.objective_terms,
        _reserve.add_constraints,
        _reserve.objective_terms,
        _cumulative_invest.add_constraints,
        _commodity_ladder.add_constraints,
        _commodity_ladder.objective_terms,
        # ── stochastic ties ──
        _non_anticipativity.add_dispatch_constraints,
        _non_anticipativity.add_invest_constraints,
    ):
        _build_prof(f"{step.__module__.rsplit('.', 1)[-1]}.{step.__name__}")
        term = step(b)
        if term is not None:
            obj = term if obj is None else obj + term

    # The realization-roll checks G1-G4 (no-ops without a period label with steps; G2 holds on every LP).
    _period_rows.check_period_rows(b)
    _period_rows.check_branch_weights(b)
    check_step_relations(d, solve_name)
    _period_rows.check_registry_pins(b)   # their registries, on this LP (FLEXTOOL_AUTOSCALE_STRICT=1 only)
    if scale_the_objective != 1.0:
        obj = obj * scale_the_objective
    m.set_objective(obj, sense="min")
    _objective_options.set_objective_scale(m, scale_the_objective)
    _build_prof("build_flextool:return")
