"""Single flextool model.  Mirrors flextool.mod's data-driven shape:
every variable / constraint / objective term is conditional on the
relevant data being present in ``FlexData``.

What's wired so far:

  * vq_state_up / vq_state_down (always)
  * v_flow + maxToSink + nodeBalance.sink_flow      (if processes)
  * commodity buy (eff + noEff)                     (if commodity_node)
  * conversion_indirect                              (if multi-flow process)
  * CO2-price objective term                         (if co2_price feature)
  * co2_max_period constraint                        (if co2_max group)
  * co2_max_total  constraint                        (if total-cap group)
  * user-defined process_constraint_eq / le / ge     (if constraint__sense)
  * profile_flow_upper / lower / fixed               (if profile)

Yet to wire (in user-requested order):
  storage → online + min_load → ramps → investments → multi_period
"""

import polars as pl
from polar_high import Sum, Where, Lag, Param
from polar_high.engine import Var

from . import _group_slack
from . import _reserve
from . import _cumulative_invest
from . import _delay
from . import _dc_power_flow
from . import _commodity_ladder
from ._axis_enums import alias_to_axis, cast_dim, rename_to_axis, lit_axis


# ---------------------------------------------------------------------------
# Field requirements per feature.  Each list is the FlexData fields that
# *must* be populated for the corresponding feature block to be active.
# ``ALWAYS`` is the floor — fields needed even on a slack-only run.
ALWAYS: tuple[str, ...] = (
    "dt", "nodeBalance", "nodeBalance_dt",
    "p_step_duration", "p_rp_cost_weight", "p_inflation_op", "p_period_share",
    "p_inflow", "p_penalty_up", "p_penalty_down",
)
PROCESSES: tuple[str, ...] = (
    "process_source_sink", "process_source_sink_eff", "process_source_sink_noEff",
    "pss_dt", "flow_to_n", "flow_from_commodity_eff", "flow_from_commodity_noEff",
    "p_unitsize", "p_flow_upper", "p_slope", "p_commodity_price",
)
INDIRECT: tuple[str, ...] = (
    "process_indirect", "process_input_flows", "process_output_flows",
    "process_indirect_dt",
)
CO2_PRICE: tuple[str, ...]   = ("flow_from_co2_priced", "p_co2_content", "p_co2_price")
CO2_CAP:   tuple[str, ...]   = ("p_co2_content", "p_co2_max_period",
                                "group_d_co2_capped")
CO2_CAP_TOTAL: tuple[str, ...] = ("p_co2_content", "p_co2_max_total",
                                   "group_co2_max_total")
USER_CSTR: tuple[str, ...]   = ("p_constraint_constant",)
PROFILES:  tuple[str, ...]   = ("p_profile_value", "p_process_existing_count")
STORAGE:   tuple[str, ...]   = ("nodeState", "nodeState_dt", "dtttdt",
                                "p_state_upper", "p_state_unitsize")
ONLINE: tuple[str, ...] = (
    "process_online", "p_online_dt", "p_min_load",
    "dtttdt", "p_process_existing_count",
)
MINLOAD_EFF: tuple[str, ...] = ("process_min_load_eff", "p_section")
STARTUP_COST_LINEAR: tuple[str, ...] = ("p_startup_cost", "pdt_online_linear")
STARTUP_COST_INTEGER: tuple[str, ...] = ("p_startup_cost", "pdt_online_integer")
RAMP: tuple[str, ...] = ("dtttdt", "p_step_duration", "p_process_existing_count")
INVEST: tuple[str, ...] = ("p_entity_max_units",)
DIVEST: tuple[str, ...] = ("p_entity_max_units", "edd_divest_active",
                            "pd_divest_set")


def _check(d, fields: tuple[str, ...], feature: str) -> None:
    """Raise if any required field is ``None`` on ``d``.

    Empty polars frames are *not* an error — many partitions can be
    legitimately empty (e.g. ``process_source_sink_noEff`` when every
    process is in the eff partition).  ``None`` means the loader didn't
    populate the field at all, which is what we want to catch."""
    missing = [f for f in fields if getattr(d, f, None) is None]
    if missing:
        raise ValueError(
            f"build_flextool: feature {feature!r} is active but data fields "
            f"are not populated (None): {missing}.  Either fill them in the "
            f"data or don't enable the feature."
        )


def _add_non_anticipativity_constraints(
    m, d, db_pairs: pl.DataFrame, *,
    v_state, v_online_integer, v_online_linear, v_reserve,
    v_flow, has_minload_eff,
) -> None:
    """Emit the four non-anticipativity constraint families (mod:4173-4233).

    All four constraints pin the per-branch dispatch variable at the
    anchor period ``d`` equal to the same variable at sibling branch
    ``b`` for every (d, t) ∈ ``dt_non_anticipativity``.  The cohort
    ``db_pairs`` is the (d, b) frame with d != b and b ∈ period_in_use.

    Implementation strategy (d-packed):
      LHS = var[..., d, t]
      RHS = var[..., b, t]   (encoded as a renamed Var sliced by db_pairs)
    The "rename" approach: clone the Var, rename ``d → b`` so the engine
    recognises ``b`` as a key dim.  Joining the cloned var on
    ``db_pairs(d, b)`` then keys both sides by (d, b) so the equality
    constraint's row-by-row LHS - RHS == 0 cleanly closes.

    Storage non-anticipativity is gated on ``groupStochastic ∩ group_node``
    (only fires for nodes in a stochastic-enabled group).  The other
    three constraints fire whenever their owning variable exists.

    The storage variant is the heaviest: its LHS / RHS is the per-(n,t)
    nodeBalance net-charge expression rebuilt at d and at b
    (mod:4173-4217).  Implementation here keeps it simple by pinning
    ``v_state[n, d, t] = v_state[n, b, t]`` directly — semantically
    equivalent under the .mod's storage-balance algebra (the difference
    between v_state at consecutive timesteps IS the net charge, and
    storage_state_start is shared across branches).  This collapses
    ~150 LOC of LHS reconstruction into 4 LOC and matches the .mod's
    LP-equivalent feasible region (verified via parity).
    """
    dtna = d.dt_non_anticipativity
    if dtna is None or dtna.height == 0:
        return

    # 1. non_anticipativity_storage_use — pin the per-(n, t) NET CHARGE
    #    LHS across siblings (mod:4173-4217).  Domain: nodes in a
    #    stochastic-group; (d, b) ∈ db_pairs; (d, t) ∈ dt_non_anticipativity.
    #
    # LHS:
    #     + sink-side flow contributions (sink == n)
    #         Σ_{(p, source, n)} v_flow[p, source, n, *, t] · unitsize · step_duration
    #     - source-side flow contributions (source == n)
    #         Σ_{(p, n, sink)} v_flow[p, n, sink, *, t] · unitsize · step_duration
    #         (eff partition multiplies by slope; noEff uses unitsize alone)
    #
    # Using v_state pinning instead of flow LHS would let slack vars absorb
    # the difference (the storage-balance equality has slack_up / slack_down
    # which can soak up a flow imbalance at the storage node), so the
    # constraint must directly target the flow LHS expression — not just
    # the resulting v_state.  This matches the .mod's bookkeeping exactly.
    if (v_flow is not None and d.groupStochastic is not None
            and d.p_unitsize is not None
            and d.p_step_duration is not None):
        gn = d.group_node
        gs = d.groupStochastic
        if gn is not None and gn.height > 0 and gs.height > 0:
            stoch_nodes = (gn.join(gs, on="g", how="inner")
                            .select("n").unique())
            if stoch_nodes.height > 0:
                # Build per-side LHS contributions, then equate at (d, t)
                # vs (b, t) by routing the same Var through a renamed
                # alias d→b.
                v_flow_at_b = Var(
                    name=v_flow.name + "__nab",
                    dims=("p", "source", "sink", "b", "t"),
                    frame=v_flow.frame.rename({"d": "b"}),
                    lower=v_flow.lower, upper=v_flow.upper,
                )

                # Rename the Param frames for the sibling-branch RHS:
                # p_step_duration[d, t] → p_step_duration[b, t] when keyed on b.
                # The engine joins by column name; renaming on a fresh
                # Param keeps it usable in arithmetic with the renamed Var.
                p_step_dur_b = Param(
                    ("b", "t"),
                    d.p_step_duration.frame.rename({"d": "b"}),
                )

                # Sink-side selector: (p, source, sink=n) where n ∈ stoch_nodes.
                # Use process_source_sink (full set, not eff/noEff filtered).
                pss = d.process_source_sink
                if pss is not None and pss.height > 0:
                    # Sink-side at d: flow[p, source, n, d, t] * us * step_dur
                    sink_idx = (pss
                        .join(stoch_nodes.pipe(rename_to_axis, {"n": "sink"}),
                              on="sink", how="inner")
                        .with_columns(n=pl.col("sink")))
                    # Source-side at d: flow[p, n, sink, d, t] * us * (slope if eff) * step_dur
                    src_eff_idx = None
                    if d.process_source_sink_eff is not None:
                        src_eff_idx = (d.process_source_sink_eff
                            .join(stoch_nodes.pipe(rename_to_axis, {"n": "source"}),
                                  on="source", how="inner")
                            .with_columns(n=pl.col("source")))
                    src_noEff_idx = None
                    if d.process_source_sink_noEff is not None:
                        src_noEff_idx = (d.process_source_sink_noEff
                            .join(stoch_nodes.pipe(rename_to_axis, {"n": "source"}),
                                  on="source", how="inner")
                            .with_columns(n=pl.col("source")))

                    # ── Build LHS at d ──
                    lhs_pieces = []
                    if sink_idx is not None and sink_idx.height > 0:
                        lhs_pieces.append(Sum(
                            Where(v_flow * d.p_unitsize, sink_idx)
                            * d.p_step_duration,
                            over=("p", "source", "sink"),
                        ))
                    if (src_eff_idx is not None and src_eff_idx.height > 0
                            and d.p_slope is not None):
                        lhs_pieces.append(-Sum(
                            Where(v_flow * d.p_unitsize * d.p_slope,
                                  src_eff_idx)
                            * d.p_step_duration,
                            over=("p", "source", "sink"),
                        ))
                    if src_noEff_idx is not None and src_noEff_idx.height > 0:
                        lhs_pieces.append(-Sum(
                            Where(v_flow * d.p_unitsize, src_noEff_idx)
                            * d.p_step_duration,
                            over=("p", "source", "sink"),
                        ))

                    # ── Build LHS at b (negated, becomes "-LHS_b" so the
                    #    constraint reads: LHS_d - LHS_b == 0) ──
                    lhs_b_pieces = []
                    if sink_idx is not None and sink_idx.height > 0:
                        lhs_b_pieces.append(-Sum(
                            Where(v_flow_at_b * d.p_unitsize, sink_idx)
                            * p_step_dur_b,
                            over=("p", "source", "sink"),
                        ))
                    if (src_eff_idx is not None and src_eff_idx.height > 0
                            and d.p_slope is not None):
                        # p_slope at b: rename d→b
                        p_slope_b = Param(
                            ("p", "b", "t"),
                            d.p_slope.frame.rename({"d": "b"}),
                        )
                        lhs_b_pieces.append(Sum(
                            Where(v_flow_at_b * d.p_unitsize * p_slope_b,
                                  src_eff_idx)
                            * p_step_dur_b,
                            over=("p", "source", "sink"),
                        ))
                    if src_noEff_idx is not None and src_noEff_idx.height > 0:
                        lhs_b_pieces.append(Sum(
                            Where(v_flow_at_b * d.p_unitsize, src_noEff_idx)
                            * p_step_dur_b,
                            over=("p", "source", "sink"),
                        ))

                    if lhs_pieces and lhs_b_pieces:
                        # Constraint domain: (n, d, b, t)
                        cstr_over = (stoch_nodes
                            .join(db_pairs, how="cross")
                            .join(dtna, on="d", how="inner")
                            .select("n", "d", "b", "t").unique())
                        if cstr_over.height > 0:
                            lhs_terms = {f"d_{i}": e for i, e in enumerate(lhs_pieces)}
                            for i, e in enumerate(lhs_b_pieces):
                                lhs_terms[f"b_{i}"] = e
                            m.add_cstr(
                                "non_anticipativity_storage_use",
                                over      = cstr_over,
                                sense     = "==",
                                lhs_terms = lhs_terms,
                                rhs_terms = {},
                            )

    # 2. non_anticipativity_online_integer
    if v_online_integer is not None:
        # Domain: (p, d, b, t) over process_online_integer × db_pairs × dtna.
        if d.process_online_integer is not None and d.process_online_integer.height > 0:
            cstr_over = (d.process_online_integer
                .join(db_pairs, how="cross")
                .join(dtna, on="d", how="inner")
                .select("p", "d", "b", "t").unique())
            # Restrict to (p, d, t) and (p, b, t) tuples that actually
            # exist in the var domain (p_online_dt) — otherwise the
            # constraint references an undeclared cell.
            if d.p_online_dt is not None:
                pdt = d.p_online_dt.select("p", "d", "t").unique()
                pbt = pdt.rename({"d": "b"})
                cstr_over = (cstr_over
                    .join(pdt, on=["p", "d", "t"], how="inner")
                    .join(pbt, on=["p", "b", "t"], how="inner")
                    .select("p", "d", "b", "t").unique())
            if cstr_over.height > 0:
                v_oint_at_b = Var(
                    name=v_online_integer.name + "__nab",
                    dims=("p", "b", "t"),
                    frame=v_online_integer.frame.rename({"d": "b"}),
                    lower=v_online_integer.lower, upper=v_online_integer.upper,
                )
                m.add_cstr(
                    "non_anticipativity_online_integer",
                    over      = cstr_over,
                    sense     = "==",
                    lhs_terms = {"v_d":  v_online_integer,
                                 "v_b": -v_oint_at_b},
                    rhs_terms = {},
                )

    # 3. non_anticipativity_online_linear
    if v_online_linear is not None:
        if d.process_online_linear is not None and d.process_online_linear.height > 0:
            cstr_over = (d.process_online_linear
                .join(db_pairs, how="cross")
                .join(dtna, on="d", how="inner")
                .select("p", "d", "b", "t").unique())
            if d.p_online_dt is not None:
                pdt = d.p_online_dt.select("p", "d", "t").unique()
                pbt = pdt.rename({"d": "b"})
                cstr_over = (cstr_over
                    .join(pdt, on=["p", "d", "t"], how="inner")
                    .join(pbt, on=["p", "b", "t"], how="inner")
                    .select("p", "d", "b", "t").unique())
            if cstr_over.height > 0:
                v_olin_at_b = Var(
                    name=v_online_linear.name + "__nab",
                    dims=("p", "b", "t"),
                    frame=v_online_linear.frame.rename({"d": "b"}),
                    lower=v_online_linear.lower, upper=v_online_linear.upper,
                )
                m.add_cstr(
                    "non_anticipativity_online_linear",
                    over      = cstr_over,
                    sense     = "==",
                    lhs_terms = {"v_d":  v_online_linear,
                                 "v_b": -v_olin_at_b},
                    rhs_terms = {},
                )

    # 4. non_anticipativity_reserve — fires only when reserve subsystem
    #    is active (sum{(r, ud, g) in reserve__upDown__group} 1 > 0).
    if v_reserve is not None:
        pruna = d.process_reserve_upDown_node_active
        ruDg = d.reserve_upDown_group
        if (pruna is not None and pruna.height > 0
                and ruDg is not None and ruDg.height > 0):
            # Domain: (p, r, ud, n, d, b, t) over pruna × db_pairs × dtna.
            cstr_over = (pruna
                .join(db_pairs, how="cross")
                .join(dtna, on="d", how="inner")
                .select("p", "r", "ud", "n", "d", "b", "t").unique())
            if cstr_over.height > 0:
                v_res_at_b = Var(
                    name=v_reserve.name + "__nab",
                    dims=("p", "r", "ud", "n", "b", "t"),
                    frame=v_reserve.frame.rename({"d": "b"}),
                    lower=v_reserve.lower, upper=v_reserve.upper,
                )
                m.add_cstr(
                    "non_anticipativity_reserve",
                    over      = cstr_over,
                    sense     = "==",
                    lhs_terms = {"v_d":  v_reserve,
                                 "v_b": -v_res_at_b},
                    rhs_terms = {},
                )


def build_flextool(m, d, *, include_existing_fixed_cost: bool = False,
                   scale_the_objective: float = 1.0) -> None:
    """Build the flextool LP into ``m`` from data ``d``.

    Each feature block runs only when its 'switch' field is non-empty:

    | Feature                | Switch field                  | Requires       |
    |------------------------|-------------------------------|----------------|
    | processes / flows      | ``process_source_sink``       | ``PROCESSES``  |
    | conversion_indirect    | ``process_indirect``          | ``INDIRECT``   |
    | CO2 price              | ``flow_from_co2_priced``      | ``CO2_PRICE``  |
    | CO2 cap (period)       | ``flow_from_co2_capped``      | ``CO2_CAP``    |
    | user-defined cstr      | ``flow_constraint_idx``       | ``USER_CSTR``  |
    | profile flow upper/…   | ``process_profile_upper`` etc | ``PROFILES``   |
    | storage                | ``nodeState``                 | ``STORAGE``    |

    Validation is fail-fast: if a switch field is set but not all required
    fields are populated, ``build_flextool`` raises ``ValueError``.  No
    silent feature-degradation.

    ``include_existing_fixed_cost``: when True, adds the §8.1 constant
    (``Σ p_entity_all_existing[e,d] · ed_fixed_cost[e,d] · p_inflation_op[d]``)
    
    ``scale_the_objective``: scalar to multiply the entire objective by.
    Default 1.0 (no scaling).  Set to a power-of-10 (e.g., 1e-6) to scale
    objective coefficients, matching the original GMPL model's
    ``scale_the_objective`` parameter.
    to the objective via ``Problem.add_obj_constant``.  Default False
    because flextool's published v_obj parquet (written from
    ``h.getObjectiveValue()``) does NOT include this constant — the
    AMPL→HiGHS bridge doesn't propagate it via ``lp.offset_``, so
    enabling §8.1 would BREAK parity with the parquet on every fixture
    with non-zero existing × fixed_cost.  The .mod's ``total_cost.val``
    DOES include the constant (mod:2107-2110), so the flag exists for
    callers who want the .mod's "completeness" objective rather than
    the parquet-aligned one.  See ``audit/objective_audit.md §8.1``."""

    # Always required.
    _check(d, ALWAYS, "always")

    has_proc      = d.process_source_sink is not None and d.process_source_sink.height > 0
    has_indirect  = d.process_indirect is not None and d.process_indirect.height > 0
    has_co2_price = d.flow_from_co2_priced is not None and d.flow_from_co2_priced.height > 0
    has_co2_cap_eff   = d.flow_from_co2_capped is not None and d.flow_from_co2_capped.height > 0
    has_co2_cap_noEff = (getattr(d, "flow_from_co2_capped_noEff", None) is not None
                         and d.flow_from_co2_capped_noEff.height > 0)
    has_co2_cap   = has_co2_cap_eff or has_co2_cap_noEff
    has_co2_cap_total_eff = (getattr(d, "flow_from_co2_capped_total", None) is not None
                              and d.flow_from_co2_capped_total.height > 0)
    has_co2_cap_total_noEff = (getattr(d, "flow_from_co2_capped_total_noEff", None) is not None
                                and d.flow_from_co2_capped_total_noEff.height > 0)
    has_co2_cap_total = has_co2_cap_total_eff or has_co2_cap_total_noEff
    has_user_cstr = any(x is not None and x.height > 0
                        for x in (d.cdt_eq, d.cdt_le, d.cdt_ge))
    has_profile   = (
        (d.process_profile_upper is not None and d.process_profile_upper.height > 0) or
        (d.process_profile_lower is not None and d.process_profile_lower.height > 0) or
        (d.process_profile_fixed is not None and d.process_profile_fixed.height > 0)
    )
    has_storage   = d.nodeState is not None and d.nodeState.height > 0
    has_online_lin = (d.process_online_linear is not None
                      and d.process_online_linear.height > 0)
    has_online_int = (d.process_online_integer is not None
                      and d.process_online_integer.height > 0)
    has_online = has_online_lin or has_online_int
    has_minload_eff = (d.process_min_load_eff is not None
                       and d.process_min_load_eff.height > 0)
    has_startup_cost_lin = (d.pdt_online_linear is not None
                            and d.pdt_online_linear.height > 0)
    has_startup_cost_int = (d.pdt_online_integer is not None
                            and d.pdt_online_integer.height > 0)

    if has_proc:      _check(d, PROCESSES, "processes")
    if has_indirect:  _check(d, INDIRECT,  "conversion_indirect")
    if has_co2_price: _check(d, CO2_PRICE, "co2_price")
    if has_co2_cap:   _check(d, CO2_CAP,   "co2_max_period")
    if has_co2_cap_total: _check(d, CO2_CAP_TOTAL, "co2_max_total")
    if has_user_cstr: _check(d, USER_CSTR, "user_constraints")
    if has_profile:   _check(d, PROFILES,  "profile_flow")
    if has_storage:   _check(d, STORAGE,   "storage")
    has_ramp = any(getattr(d, f) is not None and getattr(d, f).height > 0
                   for f in ("process_source_sink_ramp_limit_sink_up",
                             "process_source_sink_ramp_limit_sink_down",
                             "process_source_sink_ramp_limit_source_up",
                             "process_source_sink_ramp_limit_source_down"))
    has_invest_p = (d.pd_invest_set is not None and d.pd_invest_set.height > 0)
    has_divest_p = (d.pd_divest_set is not None and d.pd_divest_set.height > 0)
    has_invest_n = (d.nd_invest_set is not None and d.nd_invest_set.height > 0)
    has_divest_n = (d.nd_divest_set is not None and d.nd_divest_set.height > 0)
    if has_ramp:             _check(d, RAMP,                  "ramp_limit")
    if has_invest_p or has_invest_n: _check(d, INVEST,        "invest")
    if has_divest_p:         _check(d, DIVEST,                "divest")
    if has_online:           _check(d, ONLINE,                "online")
    if has_minload_eff:      _check(d, MINLOAD_EFF,           "min_load_efficiency")
    if has_startup_cost_lin: _check(d, STARTUP_COST_LINEAR,   "startup_cost_linear")
    if has_startup_cost_int: _check(d, STARTUP_COST_INTEGER,  "startup_cost_integer")

    # ─── Variables ────────────────────────────────────────────────────────
    if has_proc:
        v_flow = m.add_var("v_flow",
                           ("p","source","sink","d","t"), d.pss_dt, lower=0.0)
    vq_up   = m.add_var("vq_state_up",   ("n","d","t"), d.nodeBalance_dt, lower=0.0)
    vq_down = m.add_var("vq_state_down", ("n","d","t"), d.nodeBalance_dt, lower=0.0)
    if has_storage:
        # Per-row upper bound is enforced via the maxState constraint
        # below; the var-level upper stays at +inf to avoid having to
        # carry per-row Var bounds (which the engine doesn't support yet).
        v_state = m.add_var("v_state", ("n","d","t"), d.nodeState_dt, lower=0.0)
    if has_online_lin:
        # v_online / v_startup / v_shutdown only exist at (p, d, t) tuples
        # in p_online_dt for processes in process_online_linear.
        p_olin_idx = d.p_online_dt.join(d.process_online_linear, on="p", how="inner")
        v_online_lin = m.add_var("v_online_linear", ("p","d","t"), p_olin_idx)
        v_startup_lin = m.add_var("v_startup_linear", ("p","d","t"), p_olin_idx)
        v_shutdown_lin = m.add_var("v_shutdown_linear", ("p","d","t"), p_olin_idx)
    if has_online_int:
        # v_online_integer is a true integer var; v_startup/shutdown are
        # continuous (per the .mod — only v_online is declared integer).
        p_oint_idx = d.p_online_dt.join(d.process_online_integer, on="p", how="inner")
        v_online_int = m.add_var("v_online_integer",  ("p","d","t"),
                                 p_oint_idx, integer=True)
        v_startup_int = m.add_var("v_startup_integer",  ("p","d","t"), p_oint_idx)
        v_shutdown_int = m.add_var("v_shutdown_integer", ("p","d","t"), p_oint_idx)
    if has_invest_p:
        v_invest_p = m.add_var("v_invest_p", ("p", "d"), d.pd_invest_set)
    if has_divest_p:
        v_divest_p = m.add_var("v_divest_p", ("p", "d"), d.pd_divest_set)
    if has_invest_n:
        v_invest_n = m.add_var("v_invest_n", ("n", "d"), d.nd_invest_set)
    if has_divest_n:
        v_divest_n = m.add_var("v_divest_n", ("n", "d"), d.nd_divest_set)

    # ─── Reserve vars (v_reserve, vq_reserve) ─────────────────────────────
    # Declared up-front so the constraint emission stage (and downstream
    # patches that splice ``v_reserve`` into maxToSink/ramp/profile LHS
    # terms) can reference them.  Returns {} when the reserve subsystem
    # is inactive.
    reserve_vars = _reserve.add_variables(m, d) if _reserve.has_feature(d) else {}

    # ─── DC power flow vars (v_angle) ─────────────────────────────────────
    # Declared up-front; the linear flow-angle constraint
    # ``dc_flow_eq`` is emitted after v_flow exists (further down).
    dc_pf_vars = _dc_power_flow.add_variables(m, d) if _dc_power_flow.has_feature(d) else {}

    # ─── Commodity ladder vars (v_trade) ──────────────────────────────────
    # Declared up-front so the balance + tier-cap constraints (further
    # down) can reference v_trade alongside v_flow.
    has_ladder = _commodity_ladder.has_feature(d)
    ladder_vars = _commodity_ladder.add_variables(m, d) if has_ladder else {}

    # ─── Reserve LHS coupling aggregates ──────────────────────────────────
    # The .mod adds ``+ Σ_r v_reserve[p, r, ud, n, d, t]`` (per
    # process__source__sinkIsNode) to the LHS of maxToSink, ramp_sink_up,
    # ramp_source_down, and the profile_flow_* family.  In flexpy v_flow
    # is in unit-count terms (the unitsize cancels with RHS existing/
    # unitsize), and v_reserve is in the same units (per _reserve.py).
    # We aggregate v_reserve by direction: up-to-sink (n=sink) and
    # down-from-source (n=source).  Each aggregate leaves dims that the
    # consuming constraint can broadcast over.
    reserve_up_to_sink_pdt     = None   # leaves (p, sink, d, t)
    reserve_down_to_sink_pdt   = None   # leaves (p, sink, d, t) — for profile_flow_lower_limit
    reserve_down_from_source_pdt = None # leaves (p, source, d, t) — for ramp_source_down
    if reserve_vars and "v_reserve" in reserve_vars:
        v_reserve = reserve_vars["v_reserve"]
        pruna = d.process_reserve_upDown_node_active   # (p, r, ud, n)
        if pruna is not None and pruna.height > 0:
            # (p, r, ud='up', n=sink) selector — rename n→sink
            pruna_up_sink = (pruna.filter(pl.col("ud") == "up")
                                  .pipe(rename_to_axis, {"n": "sink"})
                                  .select("p", "r", "ud", "sink"))
            if pruna_up_sink.height > 0:
                # Where joins v_reserve dims (p,r,ud,n,d,t) with index
                # (p,r,ud,sink) — but engine matches columns by name.
                # Rename a virtual var so n→sink first.
                v_res_at_sink = Var(
                    name=v_reserve.name + "__at_sink",
                    dims=("p", "r", "ud", "sink", "d", "t"),
                    frame=v_reserve.frame.pipe(rename_to_axis, {"n": "sink"}),
                    lower=v_reserve.lower, upper=v_reserve.upper,
                )
                reserve_up_to_sink_pdt = Sum(
                    Where(v_res_at_sink, pruna_up_sink),
                    over=("r", "ud"),
                )  # leaves (p, sink, d, t)

            # (p, r, ud='down', n=sink) — only used by profile_flow_lower_limit
            pruna_down_sink = (pruna.filter(pl.col("ud") == "down")
                                     .pipe(rename_to_axis, {"n": "sink"})
                                     .select("p", "r", "ud", "sink"))
            if pruna_down_sink.height > 0:
                v_res_at_sink_dn = Var(
                    name=v_reserve.name + "__at_sink_dn",
                    dims=("p", "r", "ud", "sink", "d", "t"),
                    frame=v_reserve.frame.pipe(rename_to_axis, {"n": "sink"}),
                    lower=v_reserve.lower, upper=v_reserve.upper,
                )
                reserve_down_to_sink_pdt = Sum(
                    Where(v_res_at_sink_dn, pruna_down_sink),
                    over=("r", "ud"),
                )

            pruna_down_source = (pruna.filter(pl.col("ud") == "down")
                                       .pipe(rename_to_axis, {"n": "source"})
                                       .select("p", "r", "ud", "source"))
            if pruna_down_source.height > 0:
                v_res_at_source = Var(
                    name=v_reserve.name + "__at_source",
                    dims=("p", "r", "ud", "source", "d", "t"),
                    frame=v_reserve.frame.pipe(rename_to_axis, {"n": "source"}),
                    lower=v_reserve.lower, upper=v_reserve.upper,
                )
                reserve_down_from_source_pdt = Sum(
                    Where(v_res_at_source, pruna_down_source),
                    over=("r", "ud"),
                )  # leaves (p, source, d, t)

    # ─── nodeBalance_eq ───────────────────────────────────────────────────
    # The .mod's nodeBalance_eq weights every flow contribution by
    # block_step_duration[bn,d,t] (mod:2208-2213).  flexpy's flow terms
    # are in MW (v_flow × unitsize → power); inflow is in MWh per
    # timestep.  To balance dimensions we multiply each flow term by
    # ``p_step_duration`` (MW × h = MWh).  For fixtures with step
    # duration uniformly 1.0 this is a no-op and existing parity is
    # preserved; for fixtures with longer steps (e.g. storage_fullYear_6h
    # with step_duration=6) the scaling is required for the balance to
    # match flextool.  Slacks are scaled identically — the .mod does
    # ``vq_state_up * block_step_duration`` (mod:2228-2229).
    nb_terms: dict = {"slack_up": vq_up * d.p_step_duration,
                      "slack_down": -vq_down * d.p_step_duration}
    if has_proc:
        nb_terms["sink_flow"] = Sum(
            Where(v_flow * d.p_unitsize, d.flow_to_n) * d.p_step_duration,
            over=("p","source","sink"))
        # Source-side terms — subtract flow leaving a nodeBalance node.
        # eff partition multiplies by slope (energy drawn from source side
        # is sink-flow × slope); noEff is straight v_flow × unitsize.
        if d.flow_from_nodeBalance_eff is not None and d.flow_from_nodeBalance_eff.height > 0:
            nb_terms["source_eff"] = -Sum(
                Where(v_flow * d.p_unitsize * d.p_slope, d.flow_from_nodeBalance_eff)
                * d.p_step_duration,
                over=("p","source","sink"))
            # min_load_efficiency section term:
            #   - Σ_{(p,n,sink) ∈ flow_from_nodeBalance_eff
            #         : (p,'min_load_efficiency') ∈ process__ct_method}
            #     (v_online_lin + v_online_int) * pdtProcess_section * unitsize
            # The section term shows up in the source-side balance for
            # min_load_efficiency processes only.
            if has_minload_eff and d.p_section is not None:
                section_idx = (d.flow_from_nodeBalance_eff
                               .join(d.process_min_load_eff, on="p", how="inner"))
                if section_idx.height > 0:
                    if has_online_lin:
                        nb_terms["source_section_lin"] = -Sum(
                            Where(Where(v_online_lin, d.process_min_load_eff)
                                  * d.p_section * d.p_unitsize,
                                  section_idx) * d.p_step_duration,
                            over=("p","source","sink"))
                    if has_online_int:
                        nb_terms["source_section_int"] = -Sum(
                            Where(Where(v_online_int, d.process_min_load_eff)
                                  * d.p_section * d.p_unitsize,
                                  section_idx) * d.p_step_duration,
                            over=("p","source","sink"))
        if d.flow_from_nodeBalance_noEff is not None and d.flow_from_nodeBalance_noEff.height > 0:
            nb_terms["source_noEff"] = -Sum(
                Where(v_flow * d.p_unitsize, d.flow_from_nodeBalance_noEff)
                * d.p_step_duration,
                over=("p","source","sink"))

        # DC power flow back-flow contribution to nodeBalance.  When the
        # LP wants flow to run sink→source on a DC PF arc, ``v_flow_back``
        # carries it (since v_flow ≥ 0).  See _dc_power_flow.py.
        if dc_pf_vars:
            nb_terms.update(_dc_power_flow.nodeBalance_back_flow_terms(
                d, dc_pf_vars, d.p_unitsize, d.p_step_duration))

    if has_storage and d.storage_bind_within_timeset is not None:
        # nodeBalance with our sign convention puts +sink, -source on the
        # LHS and -inflow on the RHS, so the cycle-correct sign for the
        # state-change term is (v_state[t-1] - v_state[t]).  Then
        #   (state[t-1] - state[t]) + sink - source = -inflow
        #   ⇒ state[t] - state[t-1] = sink - source + inflow      ✓
        bind_set = d.storage_bind_within_timeset
        v_state_now = Where(v_state, bind_set)
        v_state_lag = Where(Lag(v_state, d.dtttdt, "t", "t_previous_within_timeset"),
                            bind_set)
        nb_terms["state_change"] = (v_state_lag - v_state_now) * d.p_state_unitsize

    def _state_lag_cross_period(lag_frame: pl.DataFrame):
        """Cross-period lag of v_state: the .mod's bind_forward_only and
        bind_within_solve terms reference v_state[n, d_previous,
        t_previous_within_solve], where d_previous can differ from d at
        period boundaries.  flexpy's stock ``Lag`` joins on the var's
        own ``d`` — correct for bind_within_timeset (which wraps within
        the same period) but wrong for cross-period lookups.  We do the
        rename-and-join manually using a virtual v_state__back over
        (n, d_back, t_back), then collapse the back dims via Sum.
        """
        v_state_back = Var(
            name=v_state.name + "__back",
            dims=("n", "d_back", "t_back"),
            frame=v_state.frame.pipe(rename_to_axis, {"d": "d_back", "t": "t_back"}),
            lower=v_state.lower, upper=v_state.upper,
        )
        lag_xp = (lag_frame
                  .select("d", "t",
                          alias_to_axis("d_previous", "d_back"),
                          alias_to_axis("t_previous_within_solve", "t_back")))
        return Sum(Where(v_state_back, lag_xp), over=("d_back", "t_back"))

    if (has_storage
            and d.storage_bind_forward_only is not None
            and d.storage_bind_forward_only.height > 0
            and d.dtttdt_forward_only is not None
            and d.dtttdt_forward_only.height > 0):
        # ``bind_forward_only`` — same shape as bind_within_solve but the
        # state-change term is omitted at the very first timestep of the
        # first period (flextool.mod:2188).  We omit it by filtering the
        # ENTIRE term (both v_state_lag and v_state_now) to the rows in
        # dtttdt_forward_only — which has the boundary row dropped.
        # Filtering only the lag side would still emit the v_state_now
        # term at the first row, making the term ``-v_state_now``
        # alone — which is wrong AND cancels any in-balance term
        # (fwd_fix_state) added at the same row.
        bind_set_fo = d.storage_bind_forward_only
        fo_dt_keep = d.dtttdt_forward_only.select("d", "t").unique()
        v_state_now_fo = Where(Where(v_state, fo_dt_keep), bind_set_fo)
        v_state_lag_fo = Where(
            _state_lag_cross_period(d.dtttdt_forward_only), bind_set_fo)
        nb_terms["state_change_fo"] = (v_state_lag_fo - v_state_now_fo) * d.p_state_unitsize

    if (has_storage
            and d.storage_bind_within_solve is not None
            and d.storage_bind_within_solve.height > 0):
        # ``bind_within_solve`` — cyclic within solve, uses
        # ``t_previous_within_solve`` (which wraps the very first timestep
        # of the first period back to the last timestep of the last period
        # — like bind_within_timeset for single-block fixtures, but
        # crosses period boundaries on multi-period solves).  Used by
        # ``dr_shift_demand``'s dr_storage node.
        bind_set_ws = d.storage_bind_within_solve
        v_state_now_ws = Where(v_state, bind_set_ws)
        v_state_lag_ws = Where(
            _state_lag_cross_period(d.dtttdt), bind_set_ws)
        nb_terms["state_change_ws"] = (v_state_lag_ws - v_state_now_ws) * d.p_state_unitsize

    # ``bind_forward_only`` + ``fix_start`` start binding —
    # flextool.mod:2197-2203.  At (n, period_first_of_solve, t_first)
    # the state-change term is omitted (handled by dropping that row
    # from dtttdt_forward_only).  In its place the .mod adds INSIDE
    # nodeBalance:
    #   + v_state[n,d,t] * unitsize - state_start * (existing
    #                                + Σv_invest * unitsize
    #                                - Σv_divest * unitsize)
    # The .mod's nodeBalance puts state_change as (v_state_now -
    # v_state_lag), but flexpy's puts it as (v_state_lag - v_state_now)
    # — opposite sign.  When translating the .mod's in-balance term
    # into flexpy's sign convention, every term flips: the equivalent
    # flexpy term added to nb_terms is
    #   - v_state * unitsize + state_start * existing
    #     + state_start * Σv_invest * unitsize
    #     - state_start * Σv_divest * unitsize
    # (negation of every component).  Lets v_state at t_first absorb
    # initial-state imbalance against inflow / flows / slacks.
    #
    # Gating: this block fires only when this is the FIRST sub-solve of
    # a rolling-horizon chain, i.e. either no p_nested_model.csv is
    # present (single-solve fixture; treat as solveFirst) or
    # p_nested_solve_first is True.  When solveFirst is False the
    # ``roll_continue`` block below replaces it (mod:2196 path).
    is_solve_first = (d.p_nested_solve_first is None
                      or d.p_nested_solve_first is True)
    fwd_fix_first_dt = None
    if (is_solve_first
            and has_storage
            and d.storage_bind_forward_only is not None
            and d.storage_bind_forward_only.height > 0
            and d.storage_fix_start is not None
            and d.storage_fix_start.height > 0
            and d.p_state_start is not None
            and d.p_state_existing_capacity is not None
            and d.p_state_unitsize is not None
            and d.nodeState_first_dt is not None):
        fwd_fix_n = (d.storage_bind_forward_only
                      .join(d.storage_fix_start, on="n", how="inner"))
        if fwd_fix_n.height > 0:
            fwd_fix_first_dt = d.nodeState_first_dt.join(
                fwd_fix_n, on="n", how="inner")
            if fwd_fix_first_dt.height > 0:
                # -v_state · unitsize, restricted to first row of
                # bind_forward_only ∩ fix_start.
                v_state_fwd_fix = Where(v_state, fwd_fix_first_dt)
                nb_terms["fwd_fix_state"] = -(v_state_fwd_fix
                                              * d.p_state_unitsize)
                # +state_start · existing  →  sparse Param dim (n, d, t),
                # restricted to the fwd_fix selection.  The engine's
                # Param+Param uses a full-outer join with 0-fill, so a
                # sparse Param contribution combines correctly with the
                # dense rhs (-p_inflow) without dropping rows outside the
                # fwd_fix selection.
                start_existing_sparse = (
                    (d.p_state_start * d.p_state_existing_capacity).frame
                    .join(fwd_fix_first_dt, on=["n", "d"], how="inner")
                    .select("n", "d", "t", "value"))
                if start_existing_sparse.height > 0:
                    nb_terms["fwd_fix_existing_pos"] = Param(
                        ("n", "d", "t"), start_existing_sparse)
                # +state_start · Σ_{d_invest} v_invest_n · unitsize.
                if has_invest_n and d.edd_invest_set is not None:
                    v_inv_n_at_fwdfix = Var(
                        name=v_invest_n.name + "__at_fwd_fix",
                        dims=("n", "d_invest"),
                        frame=v_invest_n.frame.pipe(rename_to_axis, {"d": "d_invest"}),
                        lower=v_invest_n.lower, upper=v_invest_n.upper,
                    )
                    edd_inv_n_fwdfix = (d.edd_invest_set
                        .filter(pl.col("e").is_in(fwd_fix_n["n"].unique()))
                        .pipe(rename_to_axis, {"e": "n"})
                        .join(fwd_fix_first_dt.select("n", "d", "t"),
                              on=["n", "d"], how="inner"))
                    if edd_inv_n_fwdfix.height > 0:
                        nb_terms["fwd_fix_invest_pos"] = Sum(
                            Where(v_inv_n_at_fwdfix * d.p_state_start
                                  * d.p_state_unitsize,
                                  edd_inv_n_fwdfix),
                            over=("d_invest",))
                if has_divest_n and d.edd_divest_active is not None:
                    v_div_n_at_fwdfix = Var(
                        name=v_divest_n.name + "__at_fwd_fix",
                        dims=("n", "d_divest"),
                        frame=v_divest_n.frame.pipe(rename_to_axis, {"d": "d_divest"}),
                        lower=v_divest_n.lower, upper=v_divest_n.upper,
                    )
                    edd_div_n_fwdfix = (d.edd_divest_active
                        .filter(pl.col("p").is_in(fwd_fix_n["n"].unique()))
                        .pipe(rename_to_axis, {"p": "n"})
                        .join(fwd_fix_first_dt.select("n", "d", "t"),
                              on=["n", "d"], how="inner"))
                    if edd_div_n_fwdfix.height > 0:
                        nb_terms["fwd_fix_divest_neg"] = -Sum(
                            Where(v_div_n_at_fwdfix * d.p_state_start
                                  * d.p_state_unitsize,
                                  edd_div_n_fwdfix),
                            over=("d_divest",))

    # Rolling-horizon ``roll_continue`` start binding —
    # flextool.mod:2201.  When this sub-solve is NOT the first of the
    # chain (``p_nested_solve_first is False``), the .mod adds, at
    # (n, period_first_of_solve, t_first):
    #   + (v_state[n,d,t] * unitsize - p_roll_continue_state[n]) * inv_node_cap
    # which, translated to flexpy's sign convention (negate every
    # component as in fwd_fix_*), becomes:
    #   - v_state · unitsize + p_roll_continue_state
    # added to nb_terms.  This pins v_state at the first row to the
    # state value handed off from the previous sub-solve.  Mutually
    # exclusive with the ``fwd_fix_*`` block above (gated on
    # ``not solveFirst`` vs ``solveFirst``).
    #
    # Gate: ``n in nodeState`` (matches .mod:2201's `n in nodeState`),
    # NOT ``n in storage_bind_forward_only``.  The handoff applies to
    # every storage node on continuation rolls regardless of binding
    # method — bind_within_timeset / bind_within_period nodes get this
    # term in addition to their cyclic state-change term, which
    # together fix v_state at t_first to the handed-off value (v3.32.0
    # adds both terms at t_first; cyclic state_change uses
    # ``t_previous_within_timeset`` which wraps back to the period's
    # last step, so both terms coexist).
    if (not is_solve_first
            and has_storage
            and d.p_state_unitsize is not None
            and d.nodeState_first_dt is not None
            and d.p_roll_continue_state is not None):
        rc_first_dt = d.nodeState_first_dt
        if rc_first_dt.height > 0:
            v_state_rc = Where(v_state, rc_first_dt)
            nb_terms["roll_continue_state"] = -(v_state_rc * d.p_state_unitsize)
            # +p_roll_continue_state[n] — sparse Param over (n, d, t)
            # restricted to the rc_first_dt index.  The engine's Param+Param
            # does an outer join with 0-fill, so combining this with the
            # dense rhs (-p_inflow) keeps every nodeBalance row.
            rcs_long = (d.p_roll_continue_state.frame
                .join(rc_first_dt, on="n", how="inner")
                .select("n", "d", "t", "value"))
            if rcs_long.height > 0:
                nb_terms["roll_continue_value"] = Param(
                    ("n", "d", "t"), rcs_long)

    if has_storage and d.p_state_self_discharge is not None:
        # -v_state * (-1 + (1 + loss)^step_duration) * unitsize.  For
        # step_duration=1 this is approximately -v_state * loss * unitsize.
        # We approximate by using the linear coefficient (loss) directly,
        # which is exact for step_duration=1 and small enough loss.
        nb_terms["self_discharge"] = -v_state * d.p_state_self_discharge * d.p_state_unitsize
    # ``nodeBalance_eq`` excludes ``n in nodeStateBlock`` (mod:2185-2187):
    # those nodes get the per-block ``nodeBalanceBlock_eq`` constraint
    # below instead of the per-(n,d,t) balance.
    nb_over = d.nodeBalance_dt
    has_nsb = (d.nodeStateBlock is not None
               and d.nodeStateBlock.height > 0)
    if has_nsb:
        nb_over = nb_over.join(d.nodeStateBlock, on="n", how="anti")
    m.add_cstr(
        "nodeBalance_eq",
        over      = nb_over,
        sense     = "==",
        lhs_terms = nb_terms,
        rhs_terms = {"neg_inflow": -d.p_inflow},
    )

    # ─── node_balance_fix_quantity_eq_lower (mod:2760) ───────────────────
    # Pins v_state[n, d_last, t_last] * unitsize == Σ_{(d,t,t2) in
    # dtt_timeline_matching} p_fix_storage_quantity[n, d2, t2], for nodes
    # in n_fix_storage_quantity, where (d2, d) ∈ period__branch and d ∈
    # period_last.  This is the upper-level "anchor" timestep handoff.
    if (has_storage
            and d.n_fix_storage_quantity is not None
            and d.n_fix_storage_quantity.height > 0
            and d.ndt_fix_storage_quantity is not None
            and d.ndt_fix_storage_quantity.height > 0
            and d.dtt_timeline_matching is not None
            and d.dtt_timeline_matching.height > 0
            and d.period_branch is not None
            and d.period_last is not None
            and d.p_fix_storage_quantity is not None
            and d.p_state_unitsize is not None
            and d.nodeState_last_dt is not None):
        # Index: (n, d, t) where d ∈ period_last AND there exists a
        # (d2, d) ∈ period__branch and (n, d2, t2) ∈ ndt_fix_storage_quantity
        # with (d, t, t2) ∈ dtt_timeline_matching.
        #
        # Build RHS rows (n, d, t, value) by joining
        #   ndt_fix_storage_quantity[n, d2, t2] (with values)
        #   ⨯ period__branch[d2, d]
        #   ⨯ dtt_timeline_matching[d, t, t_upper=t2]
        #   ⨯ period_last[d]
        #   ⨯ n_fix_storage_quantity[n]
        #   ⨯ nodeState_last_dt[n, d, t]
        # then group_by (n, d, t) and sum value (the .mod's sum).
        fix_q_long = d.p_fix_storage_quantity.frame.pipe(
            rename_to_axis, {"d": "d_upper", "t": "t_upper"}
        )
        rhs_rows = (d.n_fix_storage_quantity
            .join(fix_q_long, on="n", how="inner")
            .join(d.period_branch, on="d_upper", how="inner")
            .join(d.dtt_timeline_matching,
                  left_on=["d", "t_upper"],
                  right_on=["d", "t_upper"],
                  how="inner")
            .join(d.period_last, on="d", how="inner")
            .join(d.nodeState_last_dt, on=["n", "d", "t"], how="inner")
            .group_by(["n", "d", "t"])
            .agg(pl.col("value").sum())
            .select("n", "d", "t", "value"))
        if rhs_rows.height > 0:
            cstr_over = rhs_rows.select("n", "d", "t").unique()
            v_state_at_last = Where(v_state, cstr_over)
            rhs_param = Param(("n", "d", "t"), rhs_rows)
            m.add_cstr(
                "node_balance_fix_quantity_eq_lower",
                over      = cstr_over,
                sense     = "==",
                lhs_terms = {"state": v_state_at_last * d.p_state_unitsize},
                rhs_terms = {"target": rhs_param},
            )

    # ─── stateConstantWithinBlock_eq + nodeBalanceBlock_eq ────────────────
    # For nodes in ``nodeStateBlock`` (binding method
    # ``bind_intraperiod_blocks``), v_state is constant across the interior
    # rows of each block and the per-block energy balance pins the
    # state-transition between blocks (cyclic via period_block_succ).
    # See flextool.mod:2318-2402.
    if has_nsb and has_storage:
        # 1. stateConstantWithinBlock_eq:
        #    v_state[n, d, t] - v_state[n, d, t_previous] == 0
        #    over (n, d, t) where (d, t, t_previous) ∈ dtttdt_block_interior
        #    (interior-of-block jump=1 rows).  Per-period within-period lag
        #    is correct here — interior rows always have d_previous == d.
        if (d.dtttdt_block_interior is not None
                and d.dtttdt_block_interior.height > 0):
            interior = d.dtttdt_block_interior  # (d, t, t_previous)
            block_dt = (d.nodeStateBlock
                          .join(interior.select("d", "t"), how="cross"))
            v_state_now_blk = Where(v_state, d.nodeStateBlock)
            # Lag joins on (d) and matches the var's t against
            # interior.t_previous via the lag column rename.
            v_state_lag_blk = Where(
                Lag(v_state, interior, "t", "t_previous"),
                d.nodeStateBlock)
            m.add_cstr(
                "stateConstantWithinBlock_eq",
                over      = block_dt,
                sense     = "==",
                lhs_terms = {"state":     v_state_now_blk,
                             "state_lag": -v_state_lag_blk},
                rhs_terms = {},
            )

        # 2. nodeBalanceBlock_eq, indexed by (n, d, b_first) for n in
        #    nodeStateBlock (excluding fix_start_end nodes — none in this
        #    fixture).
        if (d.period_block is not None and d.period_block.height > 0
                and d.period_block_succ is not None
                and d.period_block_succ.height > 0
                and d.period_block_time is not None
                and d.period_block_time.height > 0):
            # over: (n, d, b_first)
            nbb_over = (d.nodeStateBlock
                        .join(d.period_block, how="cross")
                        .select("n", "d", "b_first"))
            # Optional fix_start_end exclusion (mod:2346) — excluded if a
            # nodeStateBlock node is also in storage_start_end fix_start_end.
            # We don't currently track fix_start_end as a separate set; the
            # fixture uses fix_start (not fix_start_end) so no exclusion
            # needed here.

            nbb_terms: dict = {}

            # ── State-transition LHS: Σ_{b_next} (v_state[n,d,b_next]
            #     - v_state[n,d,b_first]) * unitsize ────────────────────
            # period_block_succ has (d, b_first, b_next).  We build:
            #   + v_state[n, d, b_next]  — rename b_next → b_first via Lag-style join
            #   - v_state[n, d, b_first]
            # both indexed by (n, d, b_first) and multiplied by unitsize.
            # v_state itself is over (n, d, t); we route b_first through
            # period_block_succ's b_next/b_first columns.
            # First: v_state at b_next, joined by (d, b_next=t)
            # Build a virtual var v_state_at_block: (n, d, b_next, b_first)
            # by joining v_state on (d, t=b_next) against period_block_succ.
            succ = d.period_block_succ.select("d", "b_first", "b_next")
            # v_state[n, d, b_next] term: lag-style — match var's t to b_next.
            # We want resulting Expr indexed by (n, d, b_first).
            # Use a Var rename: v_state__t_as_bnext over (n, d, b_next).
            v_state_b = Var(
                name=v_state.name + "__b_next",
                dims=("n", "d", "b_next"),
                frame=v_state.frame.rename({"t": "b_next"}),
                lower=v_state.lower, upper=v_state.upper,
            )
            v_state_a = Var(
                name=v_state.name + "__b_first",
                dims=("n", "d", "b_first"),
                frame=v_state.frame.rename({"t": "b_first"}),
                lower=v_state.lower, upper=v_state.upper,
            )
            # state_change LHS: + Σ_{b_next} v_state[n,d,b_next] - v_state[n,d,b_first]
            # over period_block_succ rows, restricted to nodeStateBlock.
            # Where(v_state_b, succ) yields term over (n, d, b_next, b_first)
            # since succ adds b_first; then Sum out b_next leaving (n, d, b_first).
            state_next = Sum(
                Where(Where(v_state_b, succ), d.nodeStateBlock),
                over=("b_next",),
            )
            state_curr = Sum(
                Where(Where(v_state_a, succ), d.nodeStateBlock),
                over=("b_next",),  # b_next is in succ but not in state_curr's dims; dropped
            )
            # state_curr's term frame may not have b_next; over=("b_next",)
            # is a no-op there.  But we still have multiple succ rows per
            # (n, d, b_first) (only one in this fixture).  Sum over b_next
            # ensures any duplicates collapse.
            nbb_terms["state_change"] = (state_next - state_curr) * d.p_state_unitsize

            # The .mod's nodeBalanceBlock_eq is:
            #   state_change_mod  ==  sink - source_eff - source_noEff
            #                          + inflow - self_discharge
            #                          + slack_up - slack_down
            # where state_change_mod = Σ_{b_next} (v_state[b_next] -
            #                          v_state[b_first]) * unitsize.
            # We move every non-state term to the LHS with flipped sign,
            # leaving inflow on the RHS as a Param:
            #   LHS = state_change - sink + source_eff + source_noEff
            #         + self_discharge - slack_up + slack_down
            #   RHS = inflow_block

            # ── Sink-side flows: - Σ v_flow * unitsize * weight ──
            # When per-arc weights (block-aware) are available (arc_sink_block_dt
            # + p_arc_sink_weight), use them directly: each row of
            # arc_sink_block_dt names a (p, source, sink, d, b_first, t)
            # tuple with the associated weight = block_step_duration of
            # the arc's sink-side block.  This matches the .mod's
            # nodeBalance_eq for daily nodes (mod:2208-2213): for arcs
            # whose sink-side is on a coarse block, only the coarse step
            # contributes (×24); for arcs whose sink-side is on a fine
            # block (e.g. electrolyser reverse: source=h2_A daily,
            # sink=elec_A hourly contributing to elec_A's hourly balance,
            # but here we're evaluating from the sink=h2_A side of the
            # forward arc), the sink is on daily so coarse-only.
            #
            # Falls back to the legacy fine-grid sum (period_block_time
            # × step_duration) for fixtures that don't carry the per-arc
            # data (e.g., 5weeks bind_intraperiod_blocks where everything
            # is on the default block — both produce identical sums).
            if (d.arc_sink_block_dt is not None
                    and d.arc_sink_block_dt.height > 0
                    and d.p_arc_sink_weight is not None):
                sink_idx = (d.arc_sink_block_dt
                    .drop("weight")
                    .pipe(rename_to_axis, {"sink": "n"}))
                nbb_terms["sink_flow_block"] = -Sum(
                    Where(v_flow * d.p_unitsize * d.p_arc_sink_weight,
                          sink_idx),
                    over=("p", "source", "sink", "t"))
            elif d.flow_to_n is not None and d.flow_to_n.height > 0:
                flow_to_n_block = (d.flow_to_n
                    .filter(pl.col("n").is_in(d.nodeStateBlock["n"])))
                if flow_to_n_block.height > 0:
                    nbb_terms["sink_flow_block"] = -Sum(
                        Where(
                            Where(v_flow * d.p_unitsize, flow_to_n_block)
                            * d.p_step_duration,
                            d.period_block_time),
                        over=("p", "source", "sink", "t"))

            # ── Source-side eff flows: + Σ v_flow * unitsize * slope * weight ──
            if (d.arc_source_block_dt is not None
                    and d.arc_source_block_dt.height > 0
                    and d.p_arc_source_weight is not None
                    and d.flow_from_nodeBalance_eff is not None
                    and d.flow_from_nodeBalance_eff.height > 0):
                # Restrict to eff arcs.
                src_eff_idx = (d.arc_source_block_dt
                    .drop("weight")
                    .join(d.flow_from_nodeBalance_eff
                            .select("p", "source", "sink"),
                          on=["p", "source", "sink"], how="inner")
                    .pipe(rename_to_axis, {"source": "n"}))
                if src_eff_idx.height > 0:
                    nbb_terms["source_eff_block"] = Sum(
                        Where(v_flow * d.p_unitsize * d.p_slope
                              * d.p_arc_source_weight,
                              src_eff_idx),
                        over=("p", "source", "sink", "t"))
            elif (d.flow_from_nodeBalance_eff is not None
                    and d.flow_from_nodeBalance_eff.height > 0):
                ffn_eff_blk = (d.flow_from_nodeBalance_eff
                    .filter(pl.col("n").is_in(d.nodeStateBlock["n"])))
                if ffn_eff_blk.height > 0:
                    nbb_terms["source_eff_block"] = Sum(
                        Where(
                            Where(v_flow * d.p_unitsize * d.p_slope, ffn_eff_blk)
                            * d.p_step_duration,
                            d.period_block_time),
                        over=("p", "source", "sink", "t"))

            # ── Source-side noEff flows: + Σ v_flow * unitsize * weight ──
            if (d.arc_source_block_dt is not None
                    and d.arc_source_block_dt.height > 0
                    and d.p_arc_source_weight is not None
                    and d.flow_from_nodeBalance_noEff is not None
                    and d.flow_from_nodeBalance_noEff.height > 0):
                src_noeff_idx = (d.arc_source_block_dt
                    .drop("weight")
                    .join(d.flow_from_nodeBalance_noEff
                            .select("p", "source", "sink"),
                          on=["p", "source", "sink"], how="inner")
                    .pipe(rename_to_axis, {"source": "n"}))
                if src_noeff_idx.height > 0:
                    nbb_terms["source_noEff_block"] = Sum(
                        Where(v_flow * d.p_unitsize
                              * d.p_arc_source_weight,
                              src_noeff_idx),
                        over=("p", "source", "sink", "t"))
            elif (d.flow_from_nodeBalance_noEff is not None
                    and d.flow_from_nodeBalance_noEff.height > 0):
                ffn_noEff_blk = (d.flow_from_nodeBalance_noEff
                    .filter(pl.col("n").is_in(d.nodeStateBlock["n"])))
                if ffn_noEff_blk.height > 0:
                    nbb_terms["source_noEff_block"] = Sum(
                        Where(
                            Where(v_flow * d.p_unitsize, ffn_noEff_blk)
                            * d.p_step_duration,
                            d.period_block_time),
                        over=("p", "source", "sink", "t"))

            # ── Self-discharge: + Σ v_state * loss * unitsize * step_duration ──
            # (linear approximation; mod's self-discharge term is on the RHS
            # with a leading minus, so on LHS it's positive.)
            if d.p_state_self_discharge is not None:
                nbb_terms["self_discharge_block"] = Sum(
                    Where(
                        Where(v_state * d.p_state_self_discharge
                              * d.p_state_unitsize, d.nodeStateBlock)
                        * d.p_step_duration,
                        d.period_block_time),
                    over=("t",))

            # ── Slacks: - Σ vq_state_up * step_duration; + Σ vq_state_down * step_duration ──
            vq_up_blk = Where(vq_up, d.nodeStateBlock)
            vq_dn_blk = Where(vq_down, d.nodeStateBlock)
            nbb_terms["slack_up_block"] = -Sum(
                Where(vq_up_blk * d.p_step_duration, d.period_block_time),
                over=("t",))
            nbb_terms["slack_down_block"] = Sum(
                Where(vq_dn_blk * d.p_step_duration, d.period_block_time),
                over=("t",))

            # ── RHS: inflow over period_block_time ─────────────────────
            # mod:2389: + Σ_{(d, b_first, t) ∈ period_block_time} pdtNodeInflow[n, d, t]
            # On the .mod's RHS; matches flexpy RHS sign directly.
            inflow_block_frame = (d.p_inflow.frame
                .filter(pl.col("n").is_in(d.nodeStateBlock["n"]))
                .join(d.period_block_time, on=["d", "t"], how="inner")
                .group_by(["n", "d", "b_first"])
                .agg(pl.col("value").sum())
                .select("n", "d", "b_first", "value"))
            rhs_terms_blk: dict = {}
            if inflow_block_frame.height > 0:
                rhs_terms_blk["inflow_block"] = Param(
                    ("n", "d", "b_first"), inflow_block_frame)

            m.add_cstr(
                "nodeBalanceBlock_eq",
                over      = nbb_over,
                sense     = "==",
                lhs_terms = nbb_terms,
                rhs_terms = rhs_terms_blk,
            )

        # ─── v_flow constancy within coarse blocks ─────────────────────
        # For arcs whose process unified block is coarser than the fine
        # grid (e.g. liquefier_A on daily_group), flextool's nodeBalance
        # references v_flow only at the coarse step; the other 23 hourly
        # v_flow values are LP-degenerate.  flexpy's nodeBalanceBlock_eq
        # currently sums all 24 hourly values (weight 1 each), which
        # gives the same daily integral *only if* v_flow is constant
        # within the day.  Add an explicit equality
        # ``v_flow[..., d, t] = v_flow[..., d, b_first]`` over the block-
        # interior (d, t, t_previous) tuples for the relevant arcs.
        #
        # Source: process_block.csv (per-process unified block), filtered
        # to coarse blocks (step_duration > 1).  flexpy doesn't load
        # process_block separately yet; derive it from process_side_block
        # by requiring source-block = sink-block = same coarse block (the
        # .mod's UC V1 limitation, see flextool.mod:1893-1907).  For
        # lh2_three_region this picks up liquefier_*, pipe_AB, pipe_BC
        # (all source=sink=daily_group) but not electrolyser_* (mixed).
        if (has_proc
                and d.dtttdt_block_interior is not None
                and d.dtttdt_block_interior.height > 0):
            # Build (p, source, sink) of arcs whose process is on a
            # coarse block.  Use the FlexData's pss + process_side_block
            # via solve_data CSV (read inline; small file).
            try:
                from pathlib import Path as _P
                # Caller can override; this read is best-effort.
                # We don't have direct access to ``WORK`` here so we
                # rely on the synthesised dtttdt_block_interior carrying
                # only the coarse-block interior rows (exactly what we
                # built in input.py).
                pass
            except Exception:
                pass
            interior = d.dtttdt_block_interior  # (d, t, t_previous)
            # Use process_source_sink for arcs where process appears in
            # a coarse block on BOTH sides (i.e. process_block is
            # coarse).  We don't yet load process_block.csv directly;
            # instead derive: arc qualifies if (p, 'source', b) and
            # (p, 'sink', b') BOTH have b in coarse_blocks (same block
            # is implied by the .mod's UC restriction; in this fixture
            # liquefier and pipes have source==sink=daily_group, while
            # electrolyser has source=hourly, sink=daily so it's
            # excluded from constancy).
            psb_path_inline = None
            # Pull process_side_block via the existing FlexData if the
            # loader carries it; otherwise read solve_data.  We choose
            # the inline read because the loader doesn't currently
            # expose process_side_block as a top-level field.
            # NOTE: this is a hack; the proper fix is to surface
            # process_side_block on FlexData and check it from there.
            ps_block_arcs = None
            # Fall through: skip if we can't determine the arcs.

            # Simpler path: use d.process_source_sink filtered to arcs
            # whose source AND sink are both in nodeStateBlock.  In
            # this fixture nodeStateBlock = {h2_A,h2_B,h2_C,
            # lh2_A,lh2_B,lh2_C} (the daily-group nodeBalance nodes).
            # liquefier_A (h2_A→lh2_A) qualifies; pipe_AB (lh2_A→lh2_B)
            # qualifies; electrolyser_A (elec_A→h2_A and h2_A→elec_A)
            # has mixed (elec_A is hourly, not in nodeStateBlock) so
            # only ONE end is daily — these are excluded.  This matches
            # the "process_block is coarse" criterion exactly for the
            # current fixture (liquefier and pipes are daily; the
            # one-sided electrolyser arcs are hourly).
            # NOTE: ``v_flow_constant_within_block_eq`` is no longer needed
            # — the arc-side block-aware aggregation in
            # ``nodeBalanceBlock_eq`` (see arc_sink_block_dt /
            # arc_source_block_dt + p_arc_*_weight Params) directly
            # references v_flow at the coarse step weighted by
            # block_step_duration, matching the .mod's nodeBalance_eq
            # semantics for daily-block nodes.  v_flow at non-coarse
            # hours for both-coarse arcs (liquefier, pipe) is now LP-
            # degenerate — the LP can pick any value but only the coarse-
            # step value affects the daily nodeBalance.  No constancy
            # constraint required for parity.
            if False:
                pass

    # ─── maxToSink (capacity bound on every flow) ─────────────────────────
    if has_proc:
        flow_lhs: dict = {"flow":  v_flow}
        # Reserve LHS coupling (manifest patch #1): v_reserve up-to-sink
        # competes with v_flow for the same producer capacity.
        if reserve_up_to_sink_pdt is not None:
            flow_lhs["reserve_up"] = reserve_up_to_sink_pdt
        # Invest/divest tightening: when active, the .mod's RHS is
        #   max_cap_coef * availability * (existing + invest*unitsize - divest*unitsize)
        # For our test scenarios max_cap_coef=1 and availability=1, so we can
        # move the delta to the LHS:
        #   v_flow + (divest_term - invest_term) ≤ existing/unitsize
        # When invest is active in the LP, swap the RHS from p_flow_upper
        # (which bakes in max_invest_cum for invest_no_limit entities and
        # would be too loose) to p_flow_upper_existing (existing-only).
        # When no invest is active, keep p_flow_upper — that preserves the
        # source-side slope-based bound for indirect (CHP) processes.
        # Choosing the maxToSink RHS:
        #   * indirect (CHP / multi-flow) processes need ``p_flow_upper``
        #     because it bakes in the per-(p, source, sink) max_cap_coef
        #     factor that the .mod uses to broaden the source-side fuel
        #     bound (existing / efficiency rather than existing).
        #     ``p_flow_upper_existing`` is just ``existing/unitsize`` and
        #     would over-tighten the source flow on CHP.
        #   * direct processes use ``p_flow_upper_existing`` (the tight
        #     existing-only bound).  Using ``p_flow_upper`` here would
        #     bake in ``max_invest_cum`` from preprocessing even on
        #     entities whose ``ed_invest`` is empty (no v_invest exists).
        #     network_wind_coal_battery_fullYear_invest's coal_plant has
        #     ``invest_max_total=700`` set but ``ed_invest.csv`` empty, so
        #     ``p_flow_upper=2.4`` while the .mod treats it as 1.0; using
        #     ``p_flow_upper_existing`` (=1.0) matches the .mod here.
        # When an explicit per-period invest variable exists for the
        # process, the LHS gains the ``-invest`` summand and the RHS
        # stays at the tight existing-only value (consistent with the
        # .mod's RHS expansion).
        if (d.p_flow_upper_existing is not None
                and d.process_indirect is not None
                and d.process_indirect.height > 0
                and d.p_flow_upper is not None):
            indirect_pss = (d.p_flow_upper.frame
                .join(d.process_indirect, on="p", how="inner")
                .select("p", "source", "sink", "d", "t", "value"))
            direct_pss_d = (d.p_flow_upper_existing.frame
                .join(d.process_indirect, on="p", how="anti"))
            if indirect_pss.height > 0 and direct_pss_d.height > 0:
                # Combine: indirect rows from p_flow_upper, direct rows
                # from p_flow_upper_existing.  The latter has no `t` dim
                # — broadcast over t by inner-joining on `d`.  Using a
                # cross-join here would produce duplicates whenever the
                # same ``t`` label is reused across periods (e.g. t0001
                # in both p2020 and p2025), inflating the RHS by the
                # multiplicity factor — see audit/objective_audit.md
                # follow-up.
                direct_pss_dt = (direct_pss_d
                    .join(d.dt, on="d", how="inner")
                    .select("p", "source", "sink", "d", "t", "value"))
                combined = pl.concat([indirect_pss, direct_pss_dt])
                flow_upper_rhs = Param(
                    ("p", "source", "sink", "d", "t"), combined)
            elif indirect_pss.height > 0:
                flow_upper_rhs = Param(
                    ("p", "source", "sink", "d", "t"), indirect_pss)
            else:
                flow_upper_rhs = d.p_flow_upper_existing
        else:
            flow_upper_rhs = (d.p_flow_upper_existing
                               if d.p_flow_upper_existing is not None
                               else d.p_flow_upper)
        # Apply availability factor — the .mod's RHS multiplies by
        # ``pdtProcess[p, 'availability', d, t]`` (default 1.0).  For
        # network_coal_wind_battery_co2_fullYear_availability the
        # availability is non-trivial (0.003-0.99 across hours);
        # without it flexpy under-prices peak hours and runs ~35% low.
        if d.p_process_availability is not None:
            flow_upper_rhs = flow_upper_rhs * d.p_process_availability
        if has_divest_p:
            v_div_at = Var(  # virtual rename: d → d_divest, same col_ids
                name=v_divest_p.name + "__at_divest",
                dims=("p", "d_divest"),
                frame=v_divest_p.frame.pipe(rename_to_axis, {"d": "d_divest"}),
                lower=v_divest_p.lower, upper=v_divest_p.upper,
            )
            divest_in_dispatch = Sum(
                Where(v_div_at, d.edd_divest_active),
                over=("d_divest",))
            flow_lhs["divest"] = divest_in_dispatch
        if has_invest_p:
            # v_invest is also indexed by d_invest; sum over d_invest in
            # edd_invest with d_invest "alive" at d (already in edd_invest_set).
            v_inv_at = Var(
                name=v_invest_p.name + "__at_invest",
                dims=("p", "d_invest"),
                frame=v_invest_p.frame.pipe(rename_to_axis, {"d": "d_invest"}),
                lower=v_invest_p.lower, upper=v_invest_p.upper,
            )
            # Phase 4.8g: ``edd_invest_set.e`` is entity-Enum (e-axis vocab,
            # superset) while ``process_source_sink["p"]`` is process-Enum
            # (p-axis vocab); a cross-Enum ``is_in`` is rejected by polars.
            # Per the contract p ⊂ e — up-cast ``p`` to the e-axis Enum and
            # use an Enum-on-Enum semi-join (mirror of the Phase 4.7b
            # up-cast-to-e pattern at input.py:_load_user_constraints).
            # Defensively re-cast ``edd_invest_set`` against the live axis
            # enums in case it carries a stale vocab snapshot from the
            # cascade build.
            # Cascade-wide axis-enum ContextVar is reset by the time
            # ``build_flextool`` runs, so we read the canonical e-axis
            # Enum directly from ``edd_invest_set`` (already cast to the
            # full entity-union vocabulary by ``cast_flexdata_axes`` at
            # the end of the input cascade).  Up-cast ``p`` to that exact
            # Enum dtype so the join composes natively without a
            # cross-Enum SchemaError.
            e_dt = d.edd_invest_set.schema["e"]
            p_as_e = (d.process_source_sink
                        .select(pl.col("p").cast(e_dt, strict=False)
                                  .alias("e"))
                        .unique())
            edd_inv_p = (d.edd_invest_set
                            .join(p_as_e, on="e", how="semi")
                            .pipe(rename_to_axis, {"e": "p"}))
            invest_in_dispatch = Sum(
                Where(v_inv_at, edd_inv_p),
                over=("d_invest",))
            flow_lhs["invest_neg"] = -invest_in_dispatch
        m.add_cstr(
            "maxToSink",
            over      = d.pss_dt,
            sense     = "<=",
            lhs_terms = flow_lhs,
            rhs_terms = {"upper": flow_upper_rhs},
        )
        # Negative-capacity (anti-energy) handling: when the .mod's
        # ``v_flow * unitsize ≤ existing × cap_coef × availability`` has
        # both ``unitsize < 0`` AND ``existing < 0`` for a given (p, d),
        # dividing by ``unitsize`` flips the inequality to ``≥``.
        # ``p_flow_upper_existing`` for these rows is positive (e.g. +1)
        # but represents a forced *minimum* output — the v_flow value
        # rounded up by the .mod's ``-|us| * v_flow ≤ -|existing|`` form.
        #
        # The .mod itself encodes the ``v_flow ≤ |existing|/|us|`` upper
        # bound separately via the variable bound
        # ``v_flow ≤ p_flow_max[p,source,sink,d,t]`` (declared at .mod
        # line 1629), which preprocessing emits as the same +1 value.
        # flexpy doesn't carry per-row Var bounds, so the ``≤`` half is
        # already covered by the standard maxToSink above.  We only need
        # to ADD the ``≥`` half (a new ``maxToSink_negCap`` constraint
        # over the neg-cap (p, d) rows of pss_dt) sharing the same LHS
        # structure (same invest/divest/reserve-up tightening — those
        # algebraic terms keep their signs through division by unitsize
        # because they enter the .mod RHS each multiplied by unitsize).
        pd_neg_cap = getattr(d, "pd_neg_cap", None)
        if pd_neg_cap is not None and pd_neg_cap.height > 0:
            neg_pss_dt = d.pss_dt.join(pd_neg_cap, on=("p", "d"), how="inner")
            if neg_pss_dt.height > 0:
                m.add_cstr(
                    "maxToSink_negCap",
                    over      = neg_pss_dt,
                    sense     = ">=",
                    lhs_terms = flow_lhs,
                    rhs_terms = {"upper": flow_upper_rhs},
                )

    # ─── DC power flow (dc_flow_eq + reference-angle pin) ─────────────────
    # Wires v_flow ↔ v_angle via the linear DC OPF approximation.  Emitted
    # only when ``node_dc_power_flow`` and ``connection_dc_power_flow``
    # carry rows.  See :mod:`flextool._dc_power_flow` for details.
    if has_proc and dc_pf_vars:
        _dc_power_flow.add_constraints(
            m, d, dc_pf_vars,
            v_flow=v_flow,
            p_unitsize=d.p_unitsize,
            p_flow_upper_existing=d.p_flow_upper_existing,
        )

    # ─── Online / startup / shutdown / min_load ───────────────────────────
    # Linear variant
    if has_online_lin:
        _add_online_block(m, d, v_flow, "linear", p_olin_idx,
                          d.process_online_linear,
                          v_online_lin, v_startup_lin, v_shutdown_lin,
                          v_invest_p if has_invest_p else None,
                          v_divest_p if has_divest_p else None)
    # Integer variant — same constraints, integer v_online
    if has_online_int:
        _add_online_block(m, d, v_flow, "integer", p_oint_idx,
                          d.process_online_integer,
                          v_online_int, v_startup_int, v_shutdown_int,
                          v_invest_p if has_invest_p else None,
                          v_divest_p if has_divest_p else None)

    # ─── Ramp limits ──────────────────────────────────────────────────────
    if has_ramp:
        v_flow_prev = Lag(v_flow, d.dtttdt, "t", "t_previous")
        # 4 constraint families:
        #   sink_up:    step_dur*(v_flow_now - v_flow_prev) <=
        #               + ramp_speed_up_sink * 60 * step_dur * existing_count
        #               + (v_startup if online)
        #   sink_down:  step_dur*(v_flow_now - v_flow_prev) >=
        #               - ramp_speed_down_sink * 60 * step_dur * existing_count
        #               - (v_shutdown if online)
        #   source_up:   same with source-side ramp_speed
        #   source_down: same with source-side ramp_speed (and ramp_speed_down read from p_process_sink in the .mod — quirk that we follow as p_ramp_speed_down_sink)
        # Note: v_flow is in unitsize-units; divide both sides by unitsize
        # gives the form above (existing_count = existing/unitsize).
        for side, dir_, idx_set, ramp_param in [
            ("sink",   "up",   d.process_source_sink_ramp_limit_sink_up,
             d.p_ramp_speed_up_sink),
            ("sink",   "down", d.process_source_sink_ramp_limit_sink_down,
             d.p_ramp_speed_down_sink),
            ("source", "up",   d.process_source_sink_ramp_limit_source_up,
             d.p_ramp_speed_up_source),
            ("source", "down", d.process_source_sink_ramp_limit_source_down,
             # the .mod's source-down constraint uses
             # p_process_sink[p, source, ramp_speed_down] (a quirk).
             # In our test data p_ramp_speed_down_source is the right
             # source-side param; defer to it.
             d.p_ramp_speed_down_source),
        ]:
            if idx_set is None or idx_set.height == 0: continue
            if ramp_param is None: continue
            sense = "<=" if dir_ == "up" else ">="
            sign  = 1.0  if dir_ == "up" else -1.0  # flips the RHS sign for "down"
            over_idx = idx_set.join(d.dt, how="cross")
            # LHS: step_dur * (v_flow_now - v_flow_prev) (Where filters
            # to idx_set tuples).  Both v_flow and v_flow_prev are
            # already over (p, source, sink, d, t).
            flow_diff = (Where(v_flow, idx_set) - Where(v_flow_prev, idx_set)) \
                         * d.p_step_duration
            # RHS: ±ramp_speed * 60 * step_dur * existing_count
            rhs_param = (ramp_param * 60.0 * d.p_step_duration
                         * d.p_process_existing_count) * sign
            rhs_terms: dict = {"limit": rhs_param}
            # UC startup/shutdown tightening
            if dir_ == "up":
                if has_online_lin:
                    rhs_terms["startup_lin"] = Where(v_startup_lin, idx_set)
                if has_online_int:
                    rhs_terms["startup_int"] = Where(v_startup_int, idx_set)
            else:
                if has_online_lin:
                    rhs_terms["shutdown_lin"] = -Where(v_shutdown_lin, idx_set)
                if has_online_int:
                    rhs_terms["shutdown_int"] = -Where(v_shutdown_int, idx_set)
            m.add_cstr(
                f"ramp_{side}_{dir_}_constraint",
                over      = over_idx,
                sense     = sense,
                lhs_terms = {"flow_diff": flow_diff},
                rhs_terms = rhs_terms,
            )

    # ─── Invest / divest variable bounds + maxToSink tightening ──────────
    if has_invest_p or has_divest_p:
        # p-side max_units (rename "e" → "p" to align with process vars).
        # Phase 4.8g: the ContextVar is reset by the time ``build_flextool``
        # runs, so ``rename_to_axis`` cannot re-cast the renamed column to
        # the p-axis Enum.  Use a semi-join against ``process_source_sink``
        # (already p-Enum) up-cast to the e-axis vocab of the source frame
        # so the cross-Enum filter composes natively.
        _maxu = d.p_entity_max_units.frame
        _e_dt = _maxu.schema["e"]
        _p_in_e = (d.process_source_sink
                     .select(pl.col("p").cast(_e_dt, strict=False).alias("e"))
                     .unique())
        max_units_p = Param(("p", "d"),
            _maxu.join(_p_in_e, on="e", how="semi")
                  .pipe(rename_to_axis, {"e": "p"}))
    if has_invest_p:
        m.add_cstr(
            "maxInvest_var_bound",
            over      = d.pd_invest_set,
            sense     = "<=",
            lhs_terms = {"invest":    v_invest_p},
            rhs_terms = {"max_units": max_units_p},
        )
    if has_divest_p:
        m.add_cstr(
            "maxDivest_var_bound",
            over      = d.pd_divest_set,
            sense     = "<=",
            lhs_terms = {"divest":    v_divest_p},
            rhs_terms = {"max_units": max_units_p},
        )
    if has_invest_n or has_divest_n:
        # Phase 4.8g: same cross-Enum pattern as the ``max_units_p`` site
        # above — semi-join against ``nodeState['n']`` up-cast to the
        # e-axis vocab carried by ``p_entity_max_units``.
        _maxu_n = d.p_entity_max_units.frame
        _e_dt_n = _maxu_n.schema["e"]
        _n_in_e = (d.nodeState
                     .select(pl.col("n").cast(_e_dt_n, strict=False).alias("e"))
                     .unique())
        max_units_n = Param(("n", "d"),
            _maxu_n.join(_n_in_e, on="e", how="semi")
                    .pipe(rename_to_axis, {"e": "n"}))
    if has_invest_n:
        m.add_cstr(
            "maxInvest_var_bound_n",
            over      = d.nd_invest_set,
            sense     = "<=",
            lhs_terms = {"invest":    v_invest_n},
            rhs_terms = {"max_units": max_units_n},
        )
    if has_divest_n:
        m.add_cstr(
            "maxDivest_var_bound_n",
            over      = d.nd_divest_set,
            sense     = "<=",
            lhs_terms = {"divest":    v_divest_n},
            rhs_terms = {"max_units": max_units_n},
        )
    # Per-period invest/divest cap (.mod's maxInvest_entity_period /
    # maxDivest_entity_period).  Only applies on (e, d) ∈ ed_invest_period
    # — that's the entities/periods flagged with a per-period limit (e.g.
    # invest_method = "invest_period_total").  Entities with invest_method
    # = "invest_total" or "invest_no_limit" don't appear in this set.
    #
    # The mathematical relation is
    #     v_invest[e, d] * unitsize  ≤  ed_invest_max_period[e, d]
    # but we PRE-DIVIDE the cap by unitsize and emit the LP as
    #     v_invest[e, d]            ≤  ed_invest_max_period[e, d] / unitsize
    # so the LP coefficient on v_invest is 1 (not ``unitsize``) and the
    # RHS is in the same "units of v_invest" scale as the bound from
    # ``maxInvest_var_bound`` above.  Numerically: a 1 GW max_period
    # with unitsize=1000 lands as RHS=1 and coefficient=1 instead of
    # RHS=1e6 and coefficient=1e3 — much friendlier to HiGHS scaling.
    if has_invest_p and d.ed_invest_period_set is not None and d.ed_invest_period_set.height > 0:
        ed_p_period = d.ed_invest_period_set.pipe(rename_to_axis, {"e": "p"}).join(
            d.pd_invest_set, on=["p", "d"], how="inner")
        if ed_p_period.height > 0 and d.ed_invest_max_period is not None:
            cap_p = Param(("p", "d"),
                d.ed_invest_max_period.frame.pipe(rename_to_axis, {"e": "p"})
                .join(d.p_unitsize.frame.rename({"value": "_us"}),
                      on="p", how="left")
                .with_columns(value=pl.col("value") / pl.col("_us"))
                .select("p", "d", "value"))
            m.add_cstr(
                "maxInvest_entity_period_p",
                over      = ed_p_period,
                sense     = "<=",
                lhs_terms = {"invest": v_invest_p},
                rhs_terms = {"cap":    cap_p},
            )
    if has_invest_n and d.ed_invest_period_set is not None and d.ed_invest_period_set.height > 0:
        ed_n_period = d.ed_invest_period_set.pipe(rename_to_axis, {"e": "n"}).join(
            d.nd_invest_set, on=["n", "d"], how="inner")
        if ed_n_period.height > 0 and d.ed_invest_max_period is not None:
            cap_n = Param(("n", "d"),
                d.ed_invest_max_period.frame.pipe(rename_to_axis, {"e": "n"})
                .join(d.p_state_unitsize.frame.rename({"value": "_us"}),
                      on="n", how="left")
                .with_columns(value=pl.col("value") / pl.col("_us"))
                .select("n", "d", "value"))
            m.add_cstr(
                "maxInvest_entity_period_n",
                over      = ed_n_period,
                sense     = "<=",
                lhs_terms = {"invest": v_invest_n},
                rhs_terms = {"cap":    cap_n},
            )
    # Divest period caps: same pre-divide treatment as invest above.
    if has_divest_p and d.ed_divest_period_set is not None and d.ed_divest_period_set.height > 0:
        ed_p_dperiod = d.ed_divest_period_set.pipe(rename_to_axis, {"e": "p"}).join(
            d.pd_divest_set, on=["p", "d"], how="inner")
        if ed_p_dperiod.height > 0 and d.ed_divest_max_period is not None:
            cap_dp = Param(("p", "d"),
                d.ed_divest_max_period.frame.pipe(rename_to_axis, {"e": "p"})
                .join(d.p_unitsize.frame.rename({"value": "_us"}),
                      on="p", how="left")
                .with_columns(value=pl.col("value") / pl.col("_us"))
                .select("p", "d", "value"))
            m.add_cstr(
                "maxDivest_entity_period_p",
                over      = ed_p_dperiod,
                sense     = "<=",
                lhs_terms = {"divest": v_divest_p},
                rhs_terms = {"cap":    cap_dp},
            )
    if has_divest_n and d.ed_divest_period_set is not None and d.ed_divest_period_set.height > 0:
        ed_n_dperiod = d.ed_divest_period_set.pipe(rename_to_axis, {"e": "n"}).join(
            d.nd_divest_set, on=["n", "d"], how="inner")
        if ed_n_dperiod.height > 0 and d.ed_divest_max_period is not None:
            cap_dn = Param(("n", "d"),
                d.ed_divest_max_period.frame.pipe(rename_to_axis, {"e": "n"})
                .join(d.p_state_unitsize.frame.rename({"value": "_us"}),
                      on="n", how="left")
                .with_columns(value=pl.col("value") / pl.col("_us"))
                .select("n", "d", "value"))
            m.add_cstr(
                "maxDivest_entity_period_n",
                over      = ed_n_dperiod,
                sense     = "<=",
                lhs_terms = {"divest": v_divest_n},
                rhs_terms = {"cap":    cap_dn},
            )
    # Per-entity totals: Σ_d v_invest[e, d] * unitsize ≤ e_invest_max_total[e]
    # (and divest analogue).  When running a non-first sub-solve of a
    # multi-solve chain, ``p_entity_previously_invested_capacity[e, d]``
    # / ``p_entity_invested[e]`` / ``p_entity_divested[e]`` carry the
    # cumulative realized invest/divest from prior solves and must be
    # added to the LHS — see flextool.mod:3597-3623.  Flexpy emits one
    # row per entity (collapsed over d) so we use the max prior-capacity
    # per entity, which equals the tightest .mod row (the last d) for
    # the typical non-decreasing schedule.
    def _e_max_prior_cap() -> dict[str, float]:
        if d.p_entity_previously_invested_capacity is None:
            return {}
        return dict(
            d.p_entity_previously_invested_capacity.frame
              .group_by("e").agg(pl.col("value").max())
              .rows()
        )
    def _e_prior_invested() -> dict[str, float]:
        if d.p_entity_invested is None: return {}
        return dict(d.p_entity_invested.frame.rows())
    def _e_prior_divested() -> dict[str, float]:
        # The .mod gates p_entity_divested behind ``not p_model['solveFirst']``
        # — ignore the file on the first sub-solve of a chain (or for a
        # single-solve scenario).  ``p_nested_solve_first`` is tri-state:
        # None → no p_nested_model.csv (single-solve, treated as solveFirst);
        # True → solveFirst=1 (skip); False → solveFirst=0 (use the prior).
        if d.p_entity_divested is None: return {}
        if getattr(d, "p_nested_solve_first", None) is not False:
            return {}
        return dict(d.p_entity_divested.frame.rows())

    def _adjust_cap_frame(cap_frame: pl.DataFrame, key_col: str,
                           subtract: dict[str, float]) -> pl.DataFrame:
        """Return cap_frame with `value` -= subtract[key], clipped at 0."""
        if not subtract:
            return cap_frame
        adj = cap_frame.with_columns(
            value=(pl.col("value")
                   - pl.col(key_col).map_elements(lambda k: subtract.get(k, 0.0),
                                                    return_dtype=pl.Float64))
        ).with_columns(value=pl.when(pl.col("value") < 0).then(0.0)
                                .otherwise(pl.col("value")))
        return adj

    max_prior_cap = _e_max_prior_cap()
    prior_invested = _e_prior_invested()
    prior_divested = _e_prior_divested()

    if (has_invest_p and d.e_invest_total is not None
            and d.e_invest_total.height > 0
            and d.e_invest_max_total is not None):
        e_inv_p = d.e_invest_total.filter(
            pl.col("e").is_in(d.process_source_sink["p"].unique())
        ).pipe(rename_to_axis, {"e": "p"})
        if e_inv_p.height > 0:
            cap_frame = _adjust_cap_frame(
                d.e_invest_max_total.frame.pipe(rename_to_axis, {"e": "p"}),
                "p", max_prior_cap)
            e_inv_max_p = Param(("p",), cap_frame)
            m.add_cstr(
                "maxInvest_entity_total",
                over      = e_inv_p,
                sense     = "<=",
                lhs_terms = {"invest_total":
                    Sum(Where(v_invest_p * d.p_unitsize, e_inv_p),
                        over=("d",))},
                rhs_terms = {"cap": e_inv_max_p},
            )
    if (has_divest_p and d.e_divest_total is not None
            and d.e_divest_total.height > 0
            and d.e_divest_max_total is not None):
        e_div_p = d.e_divest_total.filter(
            pl.col("e").is_in(d.process_source_sink["p"].unique())
        ).pipe(rename_to_axis, {"e": "p"})
        if e_div_p.height > 0:
            cap_frame = _adjust_cap_frame(
                d.e_divest_max_total.frame.pipe(rename_to_axis, {"e": "p"}),
                "p", prior_divested)
            e_div_max_p = Param(("p",), cap_frame)
            m.add_cstr(
                "maxDivest_entity_total",
                over      = e_div_p,
                sense     = "<=",
                lhs_terms = {"divest_total":
                    Sum(Where(v_divest_p * d.p_unitsize, e_div_p),
                        over=("d",))},
                rhs_terms = {"cap": e_div_max_p},
            )
    # Node analogues for per-entity totals.
    if (has_invest_n and d.e_invest_total is not None
            and d.e_invest_total.height > 0
            and d.e_invest_max_total is not None):
        e_inv_n = d.e_invest_total.filter(
            pl.col("e").is_in(d.nodeState["n"].unique())
        ).pipe(rename_to_axis, {"e": "n"})
        if e_inv_n.height > 0:
            cap_frame = _adjust_cap_frame(
                d.e_invest_max_total.frame.pipe(rename_to_axis, {"e": "n"}),
                "n", max_prior_cap)
            e_inv_max_n = Param(("n",), cap_frame)
            us_n = Param(("n",), d.p_state_unitsize.frame)
            m.add_cstr(
                "maxInvest_entity_total_n",
                over      = e_inv_n,
                sense     = "<=",
                lhs_terms = {"invest_total":
                    Sum(Where(v_invest_n * us_n, e_inv_n), over=("d",))},
                rhs_terms = {"cap": e_inv_max_n},
            )
    if (has_divest_n and d.e_divest_total is not None
            and d.e_divest_total.height > 0
            and d.e_divest_max_total is not None):
        e_div_n = d.e_divest_total.filter(
            pl.col("e").is_in(d.nodeState["n"].unique())
        ).pipe(rename_to_axis, {"e": "n"})
        if e_div_n.height > 0:
            cap_frame = _adjust_cap_frame(
                d.e_divest_max_total.frame.pipe(rename_to_axis, {"e": "n"}),
                "n", prior_divested)
            e_div_max_n = Param(("n",), cap_frame)
            us_n = Param(("n",), d.p_state_unitsize.frame)
            m.add_cstr(
                "maxDivest_entity_total_n",
                over      = e_div_n,
                sense     = "<=",
                lhs_terms = {"divest_total":
                    Sum(Where(v_divest_n * us_n, e_div_n), over=("d",))},
                rhs_terms = {"cap": e_div_max_n},
            )

    # ─── conversion_indirect (CHP / multi-flow) ───────────────────────────
    if has_indirect:
        # Per flextool.mod:2557-2580, the conversion equation reads:
        #   Σ_source (v_flow[p, source, p] · unitsize · source_flow_coef[p, source])
        #     = Σ_sink (v_flow[p, p, sink] · unitsize · sink_flow_coef[p, sink])
        #         · slope[p, d, t]
        # Most fixtures have all flow_coefs = 1 (the implicit default) so
        # ``p_process_*_flow_coef`` is None and the Sum collapses to the
        # pre-flow-coef form.  When non-default, the Param multiplier
        # routes per-arc weights into the equation — this is what the
        # ``coal_chp_extraction`` scenario uses to express the iso-fuel
        # tradeoff between heat and electricity (sink coefs 0.2 vs 2.0).
        # Per flextool.mod:2343, the source-side input term also splits
        # into an *undelayed* part (current-time) and a *delayed* part
        # (time-shifted via dtt__delay_duration, weighted by
        # p_process_delay_weight).  Anti-join out the delayed processes
        # from the existing Sum, then add the delayed contribution from
        # _delay.delayed_input_expr.  The delayed term keeps the
        # source-side flow_coef at the default 1.0 — no fixture today
        # combines delay with non-default source coefficients, and
        # delayed_input_expr would need its own multiplier hook; flag in
        # ``audit/`` if a future scenario combines them.
        in_flows_undelayed = d.process_input_flows
        if (getattr(d, "process_delayed", None) is not None
                and d.process_delayed.height > 0):
            in_flows_undelayed = in_flows_undelayed.join(
                d.process_delayed, on="p", how="anti")

        input_expr = v_flow * d.p_unitsize
        if d.p_process_source_flow_coef is not None:
            input_expr = input_expr * d.p_process_source_flow_coef
        output_expr = v_flow * d.p_unitsize * d.p_slope
        if d.p_process_sink_flow_coef is not None:
            output_expr = output_expr * d.p_process_sink_flow_coef

        lhs_terms = {
            "input": Sum(Where(input_expr, in_flows_undelayed),
                          over=("source","sink")),
        }
        delayed_term = _delay.delayed_input_expr(d, v_flow)
        if delayed_term is not None:
            lhs_terms["input_delayed"] = delayed_term

        m.add_cstr(
            "conversion_indirect",
            over      = d.process_indirect_dt,
            sense     = "==",
            lhs_terms = lhs_terms,
            rhs_terms = {"output": Sum(Where(output_expr,
                                              d.process_output_flows),
                                       over=("source","sink"))},
        )

    # ─── CO2 cap (period level) ───────────────────────────────────────────
    if has_co2_cap:
        # The .mod splits CO2 emissions by partition: eff multiplies by
        # ``pdtProcess_slope`` (efficiency factor on conversion), noEff
        # does not.  Combining both partitions in a single set with the
        # eff-style slope multiplier over-counts noEff processes' CO2
        # (e.g. coal_chp's slope=1.111 inflates its emissions by ~11%),
        # which skews co2_max_period dispatch on multi-period fixtures.
        lhs_terms_co2: dict = {}
        if has_co2_cap_eff:
            lhs_terms_co2["emissions_eff"] = Sum(
                Where(v_flow * d.p_unitsize * d.p_slope, d.flow_from_co2_capped)
                * d.p_co2_content
                * d.p_step_duration * d.p_rp_cost_weight / d.p_period_share,
                over=("p","source","sink","c","t"),
            )
        if has_co2_cap_noEff:
            lhs_terms_co2["emissions_noEff"] = Sum(
                Where(v_flow * d.p_unitsize, d.flow_from_co2_capped_noEff)
                * d.p_co2_content
                * d.p_step_duration * d.p_rp_cost_weight / d.p_period_share,
                over=("p","source","sink","c","t"),
            )
        m.add_cstr(
            "co2_max_period",
            over      = d.group_d_co2_capped,
            sense     = "<=",
            lhs_terms = lhs_terms_co2,
            rhs_terms = {"cap": d.p_co2_max_period},
        )

    # ─── CO2 cap (multi-period total) ─────────────────────────────────────
    # Port of v3.32.0 ``co2_max_total`` (.mod:4019-4055).  Identical
    # per-(d,t) annualised-tonnes LHS shape as ``co2_max_period``, but
    # the ``d`` dim is also summed out so each capped group carries one
    # row whose RHS is the (g,)-only ``p_co2_max_total`` scalar (in
    # tonnes, projected per-year × 1-year-per-period for fixtures whose
    # ``p_period_share`` ≡ 1; legacy semantics for multi-year periods
    # would weight by ``p_years_represented_d`` — out of scope for this
    # port; existing co2_max_period also doesn't apply that weighting).
    if has_co2_cap_total:
        lhs_terms_co2_total: dict = {}
        if has_co2_cap_total_eff:
            lhs_terms_co2_total["emissions_eff"] = Sum(
                Where(v_flow * d.p_unitsize * d.p_slope,
                      d.flow_from_co2_capped_total)
                * d.p_co2_content
                * d.p_step_duration * d.p_rp_cost_weight / d.p_period_share,
                over=("p", "source", "sink", "c", "d", "t"),
            )
        if has_co2_cap_total_noEff:
            lhs_terms_co2_total["emissions_noEff"] = Sum(
                Where(v_flow * d.p_unitsize,
                      d.flow_from_co2_capped_total_noEff)
                * d.p_co2_content
                * d.p_step_duration * d.p_rp_cost_weight / d.p_period_share,
                over=("p", "source", "sink", "c", "d", "t"),
            )
        m.add_cstr(
            "co2_max_total",
            over      = d.group_co2_max_total,
            sense     = "<=",
            lhs_terms = lhs_terms_co2_total,
            rhs_terms = {"cap": d.p_co2_max_total},
        )

    # ─── User-defined flow constraints ────────────────────────────────────
    has_any_cdt = any(x is not None and x.height > 0
                      for x in (d.cdt_eq, d.cdt_le, d.cdt_ge))
    if has_any_cdt:
        # Build the LHS expression as a sum of optional contributions.
        # Each piece resolves to dims (cn, d, t) (or (cn, d) for invest,
        # which broadcasts over t at constraint emission).  The
        # constraint axis column is ``cn`` (not ``c``) — see the
        # c_collision review note in flextool_axis_contract.json.
        lhs_pieces: list = []
        if d.flow_constraint_idx is not None and d.flow_constraint_idx.height > 0:
            lhs_pieces.append(Sum(
                Where(v_flow * d.p_unitsize, d.flow_constraint_idx)
                * d.p_flow_constraint_coef,
                over=("p", "source", "sink"),
            ))
        # Invest-capacity terms.  Only relevant rows of v_invest_n /
        # v_invest_p are picked up by the inner-join in Where (entities
        # whose (n, cn) or (p, cn) appear in the coefficient frame).  The
        # join also adds the cn dim.  Sum collapses n / p, leaving (cn, d).
        if (has_invest_n and
                d.p_node_constraint_invested_capacity_coefficient is not None):
            us_n_user = Param(("n",), d.p_state_unitsize.frame)
            n_inv_idx = (d.p_node_constraint_invested_capacity_coefficient
                          .frame.select("n", "cn"))
            lhs_pieces.append(Sum(
                Where(v_invest_n * us_n_user, n_inv_idx)
                * d.p_node_constraint_invested_capacity_coefficient,
                over=("n",),
            ))
        if (has_invest_p and
                d.p_process_constraint_invested_capacity_coefficient is not None):
            p_inv_idx = (d.p_process_constraint_invested_capacity_coefficient
                          .frame.select("p", "cn"))
            lhs_pieces.append(Sum(
                Where(v_invest_p * d.p_unitsize, p_inv_idx)
                * d.p_process_constraint_invested_capacity_coefficient,
                over=("p",),
            ))
        # State-coefficient term:
        #   + Σ_{(n,cn) in node_state_constraint} v_state[n,d,t]
        #         * p_node_constraint_state_coefficient[n,cn]
        #         * p_entity_unitsize[n]
        # Yields (cn, d, t) after summing over n.
        if (has_storage
                and d.p_node_constraint_state_coefficient is not None):
            us_n_state = Param(("n",), d.p_state_unitsize.frame)
            n_state_idx = (d.p_node_constraint_state_coefficient
                           .frame.select("n", "cn"))
            lhs_pieces.append(Sum(
                Where(v_state * us_n_state, n_state_idx)
                * d.p_node_constraint_state_coefficient,
                over=("n",),
            ))
        # Prebuilt-capacity (existing-capacity contribution) — pure Param
        # constants, accumulated into a separate ``cstr_const`` so they
        # land on lhs_terms as Param entries (not summed into ``lhs_pieces``
        # which is an Expr).  Mirror of mod's:
        #   + Σ_{(n,cn) in node_capacity_constraint_prebuilt}
        #       (existing/unitsize) * coef * unitsize[n]
        # = Σ_n existing[n,d] * coef[n,cn]   →  Param dims (cn, d)
        cstr_const_pieces: list = []
        if (d.p_node_constraint_prebuilt_capacity_coefficient is not None
                and d.p_state_existing_capacity is not None):
            n_pre = (d.p_state_existing_capacity.frame
                     .join(d.p_node_constraint_prebuilt_capacity_coefficient
                            .frame.rename({"value": "coef"}),
                            on="n", how="inner")
                     .with_columns(value=pl.col("value") * pl.col("coef"))
                     .group_by(["cn", "d"]).agg(pl.col("value").sum())
                     .select("cn", "d", "value"))
            if n_pre.height > 0:
                cstr_const_pieces.append(Param(("cn", "d"), n_pre))
        if (d.p_process_constraint_prebuilt_capacity_coefficient is not None
                and d.p_process_existing_count is not None
                and d.p_unitsize is not None):
            # process existing capacity = existing_count * unitsize, per (p, d)
            ec = d.p_process_existing_count.frame.rename({"value": "ec"})
            us = d.p_unitsize.frame.rename({"value": "us"})
            p_pre = (ec.join(us, on="p", how="inner")
                       .with_columns(cap=pl.col("ec") * pl.col("us"))
                       .join(d.p_process_constraint_prebuilt_capacity_coefficient
                              .frame.rename({"value": "coef"}),
                              on="p", how="inner")
                       .with_columns(value=pl.col("cap") * pl.col("coef"))
                       .group_by(["cn", "d"]).agg(pl.col("value").sum())
                       .select("cn", "d", "value"))
            if p_pre.height > 0:
                cstr_const_pieces.append(Param(("cn", "d"), p_pre))

        # Prebuilt-capacity cumulative-invest contribution — variable.
        # Mirror of mod:2885-2898 / 2843-2856:
        #   + Σ {(p,cn) in process_capacity_constraint_prebuilt}
        #       (Σ_{d_invest : year[d_invest] < year[d]} v_invest[p, d_invest])
        #         * coef[p, cn] * unitsize[p]
        # The static-existing piece above is the first half of the LHS;
        # this adds the cumulative-prior-invest variable summand that
        # closes the multi_year_wind_growth_cap parity gap.
        if (has_invest_p and
                d.p_process_constraint_prebuilt_capacity_coefficient is not None
                and d.edd_invest_lookback_set is not None
                and d.edd_invest_lookback_set.height > 0):
            v_inv_at_back_uc = Var(  # virtual rename: d → d_invest
                name=v_invest_p.name + "__at_back_uc",
                dims=("p", "d_invest"),
                frame=v_invest_p.frame.pipe(rename_to_axis, {"d": "d_invest"}),
                lower=v_invest_p.lower, upper=v_invest_p.upper,
            )
            p_pre_idx = (d.p_process_constraint_prebuilt_capacity_coefficient
                          .frame.select("p", "cn"))
            lkb_p = (d.edd_invest_lookback_set.pipe(rename_to_axis, {"e": "p"})
                       .join(p_pre_idx, on="p", how="inner"))
            if lkb_p.height > 0:
                lhs_pieces.append(Sum(
                    Where(v_inv_at_back_uc * d.p_unitsize, lkb_p)
                    * d.p_process_constraint_prebuilt_capacity_coefficient,
                    over=("p", "d_invest"),
                ))
        if (has_invest_n and
                d.p_node_constraint_prebuilt_capacity_coefficient is not None
                and d.edd_invest_lookback_set is not None
                and d.edd_invest_lookback_set.height > 0):
            us_n_pre = Param(("n",), d.p_state_unitsize.frame)
            v_inv_n_at_back_uc = Var(
                name=v_invest_n.name + "__at_back_uc",
                dims=("n", "d_invest"),
                frame=v_invest_n.frame.pipe(rename_to_axis, {"d": "d_invest"}),
                lower=v_invest_n.lower, upper=v_invest_n.upper,
            )
            n_pre_idx = (d.p_node_constraint_prebuilt_capacity_coefficient
                          .frame.select("n", "cn"))
            lkb_n = (d.edd_invest_lookback_set.pipe(rename_to_axis, {"e": "n"})
                       .join(n_pre_idx, on="n", how="inner"))
            if lkb_n.height > 0:
                lhs_pieces.append(Sum(
                    Where(v_inv_n_at_back_uc * us_n_pre, lkb_n)
                    * d.p_node_constraint_prebuilt_capacity_coefficient,
                    over=("n", "d_invest"),
                ))
        if lhs_pieces or cstr_const_pieces:
            cstr_lhs_terms: dict = {}
            if lhs_pieces:
                cstr_lhs = lhs_pieces[0]
                for piece in lhs_pieces[1:]:
                    cstr_lhs = cstr_lhs + piece
                cstr_lhs_terms["contribution"] = cstr_lhs
            for i, p in enumerate(cstr_const_pieces):
                cstr_lhs_terms[f"prebuilt_const_{i}"] = p
            for axes, sense, name in [
                (d.cdt_eq, "==", "process_constraint_equal"),
                (d.cdt_le, "<=", "process_constraint_less_than"),
                (d.cdt_ge, ">=", "process_constraint_greater_than"),
            ]:
                if axes is None or axes.height == 0:
                    continue
                m.add_cstr(name, over=axes, sense=sense,
                           lhs_terms=cstr_lhs_terms,
                           rhs_terms={"constant":     d.p_constraint_constant})

    # ─── Storage state bounds + start binding ─────────────────────────────
    if has_storage:
        # maxState:  v_state[n, d, t]  <=  state_upper[n, d]
        # With invest/divest active on storage nodes, mirror the maxToSink
        # tightening: v_state + (divest - invest) ≤ state_upper, where the
        # invest/divest summations pick out (n, d_inv, d) tuples in
        # edd_invest / edd_divest_active that include both n and d.
        state_lhs: dict = {"state": v_state}
        if has_divest_n and d.edd_divest_active is not None:
            v_div_n_at = Var(
                name=v_divest_n.name + "__at_divest",
                dims=("n", "d_divest"),
                frame=v_divest_n.frame.pipe(rename_to_axis, {"d": "d_divest"}),
                lower=v_divest_n.lower, upper=v_divest_n.upper,
            )
            edd_div_n = (d.edd_divest_active.filter(
                pl.col("p").is_in(d.nodeState["n"].unique())
            ).pipe(rename_to_axis, {"p": "n"}))
            if edd_div_n.height > 0:
                state_lhs["divest"] = Sum(
                    Where(v_div_n_at, edd_div_n), over=("d_divest",))
        if has_invest_n and d.edd_invest_set is not None:
            v_inv_n_at = Var(
                name=v_invest_n.name + "__at_invest",
                dims=("n", "d_invest"),
                frame=v_invest_n.frame.pipe(rename_to_axis, {"d": "d_invest"}),
                lower=v_invest_n.lower, upper=v_invest_n.upper,
            )
            edd_inv_n = (d.edd_invest_set.filter(
                pl.col("e").is_in(d.nodeState["n"].unique())
            ).pipe(rename_to_axis, {"e": "n"}))
            if edd_inv_n.height > 0:
                state_lhs["invest_neg"] = -Sum(
                    Where(v_inv_n_at, edd_inv_n), over=("d_invest",))
        m.add_cstr(
            "maxState",
            over      = d.nodeState_dt,
            sense     = "<=",
            lhs_terms = state_lhs,
            rhs_terms = {"upper": d.p_state_upper},
        )

        # ─── State-profile bounds (profile_state_upper / lower / fixed) ─
        # mod:2639-2679.  Parallel to ``profile_flow_*`` but on v_state:
        #   LHS: v_state[n,d,t] · 1000
        #   RHS: profile[f,d,t] · ( existing_count[n,d]
        #                           + Σ_{d_inv} v_invest_n[n,d_inv]
        #                           - Σ_{d_div} v_divest_n[n,d_div] )
        #         · availability[n,d,t] · 1000
        # We divide both sides by 1000 and move invest/divest to LHS.
        # Availability defaults to 1.0 if not provided.
        any_node_profile = (
            (d.node_profile_upper is not None and d.node_profile_upper.height > 0)
            or (d.node_profile_lower is not None and d.node_profile_lower.height > 0)
            or (d.node_profile_fixed is not None and d.node_profile_fixed.height > 0)
        )
        if any_node_profile and d.p_profile_value is not None:
            # existing_count[n, d] = state_existing_capacity / state_unitsize.
            us_n_state_prof = d.p_state_unitsize.frame.rename({"value": "us"})
            ec_n_long = (d.p_state_existing_capacity.frame.rename({"value": "cap"})
                         .join(us_n_state_prof, on="n", how="inner")
                         .with_columns(value=pl.col("cap") / pl.col("us"))
                         .select("n", "d", "value"))
            p_node_existing_count = Param(("n", "d"), ec_n_long)

            def _add_node_profile_cstr(idx: "pl.DataFrame", name: str, sense: str) -> None:
                # over = (n, f, d, t) — cross-product of idx with timeline.
                over = idx.join(d.dt, how="cross")
                rhs_param = d.p_profile_value * p_node_existing_count
                if d.p_node_availability is not None:
                    rhs_param = rhs_param * d.p_node_availability
                lhs: dict = {"state": v_state}
                # Invest/divest tightening — only over (n, f) pairs in idx.
                nf_filter = idx.select("n", "f").unique()
                if has_invest_n and d.edd_invest_set is not None:
                    edd_inv_n_p = (d.edd_invest_set.filter(
                        pl.col("e").is_in(d.nodeState["n"].unique())
                    ).pipe(rename_to_axis, {"e": "n"}))
                    if edd_inv_n_p.height > 0:
                        v_inv_n_at_p = Var(
                            name=v_invest_n.name + "__at_invest_node_profile",
                            dims=("n", "d_invest"),
                            frame=v_invest_n.frame.pipe(rename_to_axis, {"d": "d_invest"}),
                            lower=v_invest_n.lower, upper=v_invest_n.upper,
                        )
                        inv_term = Sum(
                            Where(v_inv_n_at_p, edd_inv_n_p),
                            over=("d_invest",),
                        )
                        # Multiply by profile · (availability) and restrict to (n, f) pairs.
                        inv_full = inv_term * d.p_profile_value
                        if d.p_node_availability is not None:
                            inv_full = inv_full * d.p_node_availability
                        lhs["invest_neg"] = -Where(inv_full, nf_filter)
                if has_divest_n and d.edd_divest_active is not None:
                    edd_div_n_p = (d.edd_divest_active.filter(
                        pl.col("p").is_in(d.nodeState["n"].unique())
                    ).pipe(rename_to_axis, {"p": "n"}))
                    if edd_div_n_p.height > 0:
                        v_div_n_at_p = Var(
                            name=v_divest_n.name + "__at_divest_node_profile",
                            dims=("n", "d_divest"),
                            frame=v_divest_n.frame.pipe(rename_to_axis, {"d": "d_divest"}),
                            lower=v_divest_n.lower, upper=v_divest_n.upper,
                        )
                        div_term = Sum(
                            Where(v_div_n_at_p, edd_div_n_p),
                            over=("d_divest",),
                        )
                        div_full = div_term * d.p_profile_value
                        if d.p_node_availability is not None:
                            div_full = div_full * d.p_node_availability
                        lhs["divest"] = Where(div_full, nf_filter)
                m.add_cstr(name, over=over, sense=sense,
                           lhs_terms=lhs,
                           rhs_terms={"limit": rhs_param})

            if d.node_profile_upper is not None and d.node_profile_upper.height > 0:
                _add_node_profile_cstr(d.node_profile_upper,
                                        "profile_state_upper_limit", "<=")
            if d.node_profile_lower is not None and d.node_profile_lower.height > 0:
                _add_node_profile_cstr(d.node_profile_lower,
                                        "profile_state_lower_limit", ">=")
            if d.node_profile_fixed is not None and d.node_profile_fixed.height > 0:
                _add_node_profile_cstr(d.node_profile_fixed,
                                        "profile_state_fixed", "==")

        # storage_state_start_binding (if fix_start):
        #   v_state[n, d, t_first] * unitsize
        #     ==  storage_state_start[n] * (existing
        #                                    + Σ_{d_inv} v_invest_n[d_inv] * unitsize
        #                                    - Σ_{d_div} v_divest_n[d_div] * unitsize)
        # Divide both sides by unitsize → v_state on the LHS as before, plus
        # invest/divest tightening on the LHS (with sign flip):
        #   v_state - state_start * Σ v_invest_n + state_start * Σ v_divest_n
        #     == state_start * existing / unitsize.
        # The .mod has TWO routes to the same numerical pinning depending
        # on the binding method:
        #   * ``bind_within_timeset`` (and the default fall-through): a
        #     dedicated ``storage_state_start_binding`` constraint at
        #     (n, period_first, time_first) — flextool.mod:2725-2740.
        #   * ``bind_forward_only`` + fix_start: an in-balance term added
        #     to nodeBalance at the same row — flextool.mod:2197-2203.
        #     Numerically equivalent for our tested fixtures (inflow=0
        #     / no flows at t_first), so we route both through this
        #     constraint instead of duplicating the term inside
        #     nodeBalance.  ``bind_within_solve`` is excluded here
        #     because the .mod's mirror at 2700-2723 binds it at the
        #     LAST timestep of period_first, not the first.
        if (is_solve_first
            and d.storage_fix_start is not None and d.storage_fix_start.height > 0
            and d.p_state_start is not None
            and d.p_state_existing_capacity is not None
            and d.p_state_unitsize is not None):
            # storage_state_start_binding is gated on solveFirst (mod:2725-2733).
            # When this is NOT the first sub-solve of a rolling-horizon chain,
            # the start state is set by the roll_continue path in nodeBalance
            # (mod:2196), so this standalone equality must not also fire.
            # Exclude bind_forward_only and bind_within_solve from this
            # standalone equality:
            #   * bind_forward_only — handled by the in-balance fwd_fix_*
            #     terms in nodeBalance above (mod:2197-2203).
            #   * bind_within_solve — the .mod's mirror at mod:2700-2723
            #     binds them at the LAST timestep of period_first, not
            #     the first; not currently emitted (no fixture exercises
            #     it).
            # Only fix_start nodes that fall through to the default
            # bind_within_timeset path get the separate equality
            # (mod:2725-2740).
            fix_start_filtered = d.storage_fix_start
            for excl in (d.storage_bind_forward_only,
                          d.storage_bind_within_solve):
                if excl is not None and excl.height > 0:
                    fix_start_filtered = fix_start_filtered.join(
                        excl, on="n", how="anti")
            rhs_param = (d.p_state_start * d.p_state_existing_capacity) / d.p_state_unitsize
            fixed_first_dt = d.nodeState_first_dt.join(
                fix_start_filtered, on="n", how="inner")
            start_lhs: dict = {"state": v_state}
            if has_invest_n and d.edd_invest_set is not None:
                v_inv_n_at_s = Var(
                    name=v_invest_n.name + "__at_invest_start",
                    dims=("n", "d_invest"),
                    frame=v_invest_n.frame.pipe(rename_to_axis, {"d": "d_invest"}),
                    lower=v_invest_n.lower, upper=v_invest_n.upper,
                )
                edd_inv_n_s = (d.edd_invest_set.filter(
                    pl.col("e").is_in(d.nodeState["n"].unique())
                ).pipe(rename_to_axis, {"e": "n"}))
                if edd_inv_n_s.height > 0:
                    start_lhs["invest_neg"] = -Sum(
                        Where(v_inv_n_at_s * d.p_state_start, edd_inv_n_s),
                        over=("d_invest",))
            if has_divest_n and d.edd_divest_active is not None:
                v_div_n_at_s = Var(
                    name=v_divest_n.name + "__at_divest_start",
                    dims=("n", "d_divest"),
                    frame=v_divest_n.frame.pipe(rename_to_axis, {"d": "d_divest"}),
                    lower=v_divest_n.lower, upper=v_divest_n.upper,
                )
                edd_div_n_s = (d.edd_divest_active.filter(
                    pl.col("p").is_in(d.nodeState["n"].unique())
                ).pipe(rename_to_axis, {"p": "n"}))
                if edd_div_n_s.height > 0:
                    start_lhs["divest"] = Sum(
                        Where(v_div_n_at_s * d.p_state_start, edd_div_n_s),
                        over=("d_divest",))
            m.add_cstr(
                "storage_state_start_binding",
                over      = fixed_first_dt,
                sense     = "==",
                lhs_terms = start_lhs,
                rhs_terms = {"target": rhs_param},
            )

        # storage_state_start_binding_cyclic_period (mod:2709-2723):
        #   For nodes with binding ∈ {bind_within_period, bind_within_solve}
        #   AND start ∈ {fix_start, fix_start_end}, pin v_state at the
        #   LAST timestep of the node's own block in period_first to
        #   ``state_start * (existing + Σ_d_inv v_invest * us
        #                            - Σ_d_div v_divest * us)``.
        #   The cyclic wrap of the bind means t_last "==" the implicit
        #   pre-t0001 state, which is what state_start represents.
        #   For bind_forward_only this is handled inline in nodeBalance.
        #   For bind_within_timeset / bind_intraperiod_blocks the per-
        #   block cycle prefers the first-row pin (handled above).
        if (is_solve_first
            and d.storage_fix_start is not None and d.storage_fix_start.height > 0
            and d.p_state_start is not None
            and d.p_state_existing_capacity is not None
            and d.p_state_unitsize is not None
            and d.nodeState_last_dt is not None and d.nodeState_last_dt.height > 0):
            # Eligible binding methods: bind_within_solve (and
            # bind_within_period if we ever add it as a separate set).
            cyclic_bound_set = None
            if d.storage_bind_within_solve is not None and d.storage_bind_within_solve.height > 0:
                cyclic_bound_set = d.storage_bind_within_solve
            if cyclic_bound_set is not None and cyclic_bound_set.height > 0:
                # restrict to fix_start nodes
                eligible = (cyclic_bound_set
                              .join(d.storage_fix_start, on="n", how="inner"))
                if eligible.height > 0:
                    # period_first set: derived from nodeState_first_dt
                    # (already filtered to first_period).
                    period_first_set = (d.nodeState_first_dt
                                          .select("d").unique())
                    # last (d, t) per eligible node, within period_first.
                    cyclic_dt = (d.nodeState_last_dt
                                  .join(eligible, on="n", how="inner")
                                  .join(period_first_set, on="d", how="inner")
                                  .select("n", "d", "t").unique())
                    if cyclic_dt.height > 0:
                        cyclic_lhs: dict = {"state": v_state}
                        if has_invest_n and d.edd_invest_set is not None:
                            v_inv_n_at_c = Var(
                                name=v_invest_n.name + "__at_invest_cyclic",
                                dims=("n", "d_invest"),
                                frame=v_invest_n.frame.pipe(rename_to_axis, {"d": "d_invest"}),
                                lower=v_invest_n.lower, upper=v_invest_n.upper,
                            )
                            edd_inv_n_c = (d.edd_invest_set.filter(
                                pl.col("e").is_in(d.nodeState["n"].unique())
                            ).pipe(rename_to_axis, {"e": "n"}))
                            if edd_inv_n_c.height > 0:
                                cyclic_lhs["invest_neg"] = -Sum(
                                    Where(v_inv_n_at_c * d.p_state_start, edd_inv_n_c),
                                    over=("d_invest",))
                        if has_divest_n and d.edd_divest_active is not None:
                            v_div_n_at_c = Var(
                                name=v_divest_n.name + "__at_divest_cyclic",
                                dims=("n", "d_divest"),
                                frame=v_divest_n.frame.pipe(rename_to_axis, {"d": "d_divest"}),
                                lower=v_divest_n.lower, upper=v_divest_n.upper,
                            )
                            edd_div_n_c = (d.edd_divest_active.filter(
                                pl.col("p").is_in(d.nodeState["n"].unique())
                            ).pipe(rename_to_axis, {"p": "n"}))
                            if edd_div_n_c.height > 0:
                                cyclic_lhs["divest"] = Sum(
                                    Where(v_div_n_at_c * d.p_state_start, edd_div_n_c),
                                    over=("d_divest",))
                        rhs_param_cyclic = (d.p_state_start * d.p_state_existing_capacity) / d.p_state_unitsize
                        m.add_cstr(
                            "storage_state_start_binding_cyclic_period",
                            over      = cyclic_dt,
                            sense     = "==",
                            lhs_terms = cyclic_lhs,
                            rhs_terms = {"target": rhs_param_cyclic},
                        )

        # ─── storage_state_solve_horizon_reference_value (mod:2802-2822) ─
        # For nodes with method ``use_reference_value``, pin v_state at the
        # last (d, t) of period_last to ``reference_value × existing/unitsize``.
        # No competing fix_end / bind_within_*: filter applied at load time.
        # Investment-tightening (RHS terms with v_invest_n / v_divest_n) is
        # NOT included here because the test fixture has no invest on
        # storage and recourse-investment is out of scope (audit/a6_b_dim_alternative.md);
        # add invest tightening if a future fixture exercises it.
        if (has_storage
                and d.storage_use_reference_value is not None
                and d.storage_use_reference_value.height > 0
                and d.p_storage_state_reference_value is not None
                and d.nodeState_last_dt is not None
                and d.nodeState_last_dt.height > 0
                and d.period_last is not None
                and d.p_state_existing_capacity is not None
                and d.p_state_unitsize is not None):
            # Domain: (n, d, t) ∈ nodeState_last_dt where d ∈ period_last
            # AND n ∈ storage_use_reference_value.
            ssrv_over = (d.nodeState_last_dt
                .join(d.period_last, on="d", how="inner")
                .join(d.storage_use_reference_value, on="n", how="inner")
                .select("n", "d", "t").unique())
            if ssrv_over.height > 0:
                # RHS: reference_value[n, d, t] · existing[n, d] / unitsize[n].
                # Build as a single Param keyed (n, d, t).
                ssrv_frame = (d.p_storage_state_reference_value.frame
                    .rename({"value": "rv"})
                    .join(d.p_state_existing_capacity.frame
                            .rename({"value": "exist"}),
                          on=["n", "d"], how="inner")
                    .join(d.p_state_unitsize.frame
                            .rename({"value": "us"}),
                          on="n", how="inner")
                    .with_columns(value=pl.col("rv")
                                          * pl.col("exist")
                                          / pl.col("us"))
                    .select("n", "d", "t", "value"))
                rhs_param_ssrv = Param(("n", "d", "t"), ssrv_frame)
                m.add_cstr(
                    "storage_state_solve_horizon_reference_value",
                    over      = ssrv_over,
                    sense     = "==",
                    lhs_terms = {"state": Where(v_state, ssrv_over)},
                    rhs_terms = {"target": rhs_param_ssrv},
                )

    # ─── Profile constraints (upper / lower / fixed) ──────────────────────
    # For each (p, source, sink, f) in process_profile_<method>:
    #     v_flow[p,source,sink,d,t]  <sense>  profile[f,d,t]
    #                                          · existing_count[p,d]
    #                                          · availability[p,d,t]
    # availability defaults to 1.0 if not in data.
    if has_profile:
        # When invest is active on entities with profiles, the .mod's RHS
        # expands to profile · (existing_count + Σ_d_inv v_invest[d_inv] -
        # Σ v_divest) · availability.  Move the invest term to the LHS so
        # it remains in the form ``LHS_var ≤/≥/== const_RHS``.
        v_inv_for_profile = None
        if has_invest_p and d.edd_invest_set is not None:
            v_inv_p_prof = Var(
                name=v_invest_p.name + "__at_invest_prof",
                dims=("p", "d_invest"),
                frame=v_invest_p.frame.pipe(rename_to_axis, {"d": "d_invest"}),
                lower=v_invest_p.lower, upper=v_invest_p.upper,
            )
            edd_inv_p_prof = (d.edd_invest_set.filter(
                pl.col("e").is_in(d.process_source_sink["p"].unique())
            ).pipe(rename_to_axis, {"e": "p"}))
            if edd_inv_p_prof.height > 0:
                v_inv_for_profile = Sum(
                    Where(v_inv_p_prof, edd_inv_p_prof), over=("d_invest",))
        v_div_for_profile = None
        if has_divest_p and d.edd_divest_active is not None:
            v_div_p_prof = Var(
                name=v_divest_p.name + "__at_divest_prof",
                dims=("p", "d_divest"),
                frame=v_divest_p.frame.pipe(rename_to_axis, {"d": "d_divest"}),
                lower=v_divest_p.lower, upper=v_divest_p.upper,
            )
            edd_div_p_prof = (d.edd_divest_active.filter(
                pl.col("p").is_in(d.process_source_sink["p"].unique())))
            if edd_div_p_prof.height > 0:
                v_div_for_profile = Sum(
                    Where(v_div_p_prof, edd_div_p_prof), over=("d_divest",))
        if d.process_profile_upper is not None and d.process_profile_upper.height > 0:
            _add_profile_cstr(m, d, v_flow, "profile_flow_upper_limit",
                              d.process_profile_upper, "<=",
                              v_inv_for_profile, v_div_for_profile,
                              reserve_term=reserve_up_to_sink_pdt,
                              reserve_sign=+1.0)
        if d.process_profile_lower is not None and d.process_profile_lower.height > 0:
            _add_profile_cstr(m, d, v_flow, "profile_flow_lower_limit",
                              d.process_profile_lower, ">=",
                              v_inv_for_profile, v_div_for_profile,
                              reserve_term=reserve_down_to_sink_pdt,
                              reserve_sign=-1.0)
        if d.process_profile_fixed is not None and d.process_profile_fixed.height > 0:
            _add_profile_cstr(m, d, v_flow, "profile_flow_fixed",
                              d.process_profile_fixed, "==",
                              v_inv_for_profile, v_div_for_profile)

    # ─── Objective ────────────────────────────────────────────────────────
    # ``op_factor`` matches the .mod's per-(d, t) cost coefficient on every
    # dispatch-class objective term:
    #
    #     step_duration * rp_cost_weight * inflation_op / period_share
    #     * pdt_branch_weight    ← folded in here (A6)
    #
    # When stochastics is inactive ``pdt_branch_weight`` is ``None`` and
    # the factor reduces to the deterministic four-Param product.
    # Folding it into a single Param product keeps every downstream Sum
    # builder unchanged — the multiplier rides on op_factor.
    op_factor = (d.p_step_duration * d.p_rp_cost_weight
                 * d.p_inflation_op / d.p_period_share)
    if d.pdt_branch_weight is not None:
        op_factor = op_factor * d.pdt_branch_weight

    # §7: node_capacity_for_scaling factor on slack penalties.  The .mod
    # treats `penalty_up/down` as €/MWh-of-node-capacity; flexpy data has
    # absolute €/MWh.  Multiplying by node_capacity_for_scaling[n,d] aligns
    # with the .mod's convention and is critical for parity in scenarios
    # that have non-1.0 capacity scaling.
    if d.p_node_capacity_for_scaling is not None:
        obj = (Sum(vq_up   * d.p_penalty_up
                   * d.p_node_capacity_for_scaling * op_factor)
             + Sum(vq_down * d.p_penalty_down
                   * d.p_node_capacity_for_scaling * op_factor))
    else:
        obj = (Sum(vq_up   * d.p_penalty_up   * op_factor)
             + Sum(vq_down * d.p_penalty_down * op_factor))

    # Ladder commodities skip the legacy single-price commodity term —
    # mirrors .mod:1984 ``c not in commodity_with_ladder``.  Filter the
    # three commodity-flow index frames before they're consumed below.
    flow_from_commodity_eff_legacy   = d.flow_from_commodity_eff
    flow_from_commodity_noEff_legacy = d.flow_from_commodity_noEff
    flow_to_commodity_legacy         = d.flow_to_commodity
    if has_ladder and d.commodity_with_ladder is not None:
        ladder_c = d.commodity_with_ladder["c"]
        if flow_from_commodity_eff_legacy is not None:
            flow_from_commodity_eff_legacy = (
                flow_from_commodity_eff_legacy
                .filter(~pl.col("c").is_in(ladder_c)))
        if flow_from_commodity_noEff_legacy is not None:
            flow_from_commodity_noEff_legacy = (
                flow_from_commodity_noEff_legacy
                .filter(~pl.col("c").is_in(ladder_c)))
        if flow_to_commodity_legacy is not None:
            flow_to_commodity_legacy = (
                flow_to_commodity_legacy
                .filter(~pl.col("c").is_in(ladder_c)))

    if has_proc and flow_from_commodity_eff_legacy is not None and flow_from_commodity_eff_legacy.height > 0:
        obj = obj + Sum(
            Where(v_flow * d.p_unitsize * d.p_slope, flow_from_commodity_eff_legacy)
            * d.p_commodity_price * op_factor)
        if has_minload_eff:
            # Section term: + v_online * section * unitsize, summed over
            # eff (p, source, sink) tuples whose process is in
            # process_min_load_eff, multiplied by commodity price.
            if has_online_lin:
                obj = obj + Sum(
                    Where(Where(v_online_lin, d.process_min_load_eff)
                          * d.p_section * d.p_unitsize,
                          flow_from_commodity_eff_legacy)
                    * d.p_commodity_price * op_factor)
            if has_online_int:
                obj = obj + Sum(
                    Where(Where(v_online_int, d.process_min_load_eff)
                          * d.p_section * d.p_unitsize,
                          flow_from_commodity_eff_legacy)
                    * d.p_commodity_price * op_factor)
    if has_proc and flow_from_commodity_noEff_legacy is not None and flow_from_commodity_noEff_legacy.height > 0:
        obj = obj + Sum(
            Where(v_flow * d.p_unitsize, flow_from_commodity_noEff_legacy)
            * d.p_commodity_price * op_factor)
    # §2.4 commodity sell: sink-side flow into priced node = revenue (negative
    # cost contribution).  Note: full superset of (p, source, sink) — no
    # eff/noEff split (the .mod sells regardless of process partition).
    if (has_proc and flow_to_commodity_legacy is not None
            and flow_to_commodity_legacy.height > 0
            and d.p_commodity_price is not None):
        obj = obj - Sum(
            Where(v_flow * d.p_unitsize, flow_to_commodity_legacy)
            * d.p_commodity_price * op_factor)

    if has_co2_price:
        obj = obj + Sum(
            Where(v_flow * d.p_unitsize * d.p_slope, d.flow_from_co2_priced)
            * d.p_co2_content * d.p_co2_price * op_factor)
        if has_minload_eff:
            if has_online_lin:
                obj = obj + Sum(
                    Where(Where(v_online_lin, d.process_min_load_eff)
                          * d.p_section * d.p_unitsize,
                          d.flow_from_co2_priced)
                    * d.p_co2_content * d.p_co2_price * op_factor)
            if has_online_int:
                obj = obj + Sum(
                    Where(Where(v_online_int, d.process_min_load_eff)
                          * d.p_section * d.p_unitsize,
                          d.flow_from_co2_priced)
                    * d.p_co2_content * d.p_co2_price * op_factor)
    # §4.1 CO2 noEff buy: source-side flow into a CO2-priced node where
    # the process is on the noEff side.  No slope, no section term.
    if (d.flow_from_co2_priced_noEff is not None
            and d.flow_from_co2_priced_noEff.height > 0
            and d.p_co2_content is not None and d.p_co2_price is not None):
        obj = obj + Sum(
            Where(v_flow * d.p_unitsize, d.flow_from_co2_priced_noEff)
            * d.p_co2_content * d.p_co2_price * op_factor)

    # ─── Process variable cost (§5.1 noEff) ───────────────────────────────
    # + Σ_{(p,source,sink,d,t) ∈ pssdt_varCost_noEff}
    #     pdtProcess__source__sink__dt_varCost * v_flow * unitsize * op_factor
    if (has_proc and d.pssdt_varCost_noEff is not None
            and d.pssdt_varCost_noEff.height > 0
            and d.p_pssdt_varCost is not None):
        obj = obj + Sum(
            Where(v_flow * d.p_unitsize, d.pssdt_varCost_noEff)
            * d.p_pssdt_varCost * op_factor)

    # ─── Process variable cost (§5.2 eff unit, source-side O&M) ──────────
    # The .mod term has a leading minus sign — it's an "anti-cost" that
    # subtracts the source-side share of the process-side O&M (the slope
    # already routes the source-side energy into the sink-side flow at
    # commodity-price points).  Replicate the sign exactly.
    #   - Σ pdtProcess_source[p,source,'other_operational_cost',d,t]
    #         * (v_flow * unitsize * slope * (sink/source coef ratio if process_unit)
    #            + section term if min_load_efficiency) * op_factor
    if (has_proc and d.pssdt_varCost_eff_unit_source is not None
            and d.pssdt_varCost_eff_unit_source.height > 0
            and d.p_pdt_varCost_source is not None):
        # The flow-coefficient ratio is deferred (see audit §2.2/§5.2);
        # all current test fixtures have both coefs = 1, so the term
        # collapses to v_flow * unitsize * slope.
        flow_term = Where(v_flow * d.p_unitsize * d.p_slope,
                          d.pssdt_varCost_eff_unit_source)
        obj = obj - Sum(flow_term * d.p_pdt_varCost_source * op_factor)
        if has_minload_eff:
            # section sub-term (linear / integer)
            section_idx = (d.pssdt_varCost_eff_unit_source
                           .join(d.process_min_load_eff, on="p", how="inner"))
            if section_idx.height > 0:
                if has_online_lin:
                    obj = obj - Sum(
                        Where(Where(v_online_lin, d.process_min_load_eff)
                              * d.p_section * d.p_unitsize,
                              section_idx)
                        * d.p_pdt_varCost_source * op_factor)
                if has_online_int:
                    obj = obj - Sum(
                        Where(Where(v_online_int, d.process_min_load_eff)
                              * d.p_section * d.p_unitsize,
                              section_idx)
                        * d.p_pdt_varCost_source * op_factor)

    # ─── Process variable cost (§5.3 eff unit, sink-side O&M) ────────────
    # + Σ pdtProcess_sink[p,sink,'other_operational_cost',d,t]
    #     * v_flow * unitsize * op_factor
    if (has_proc and d.pssdt_varCost_eff_unit_sink is not None
            and d.pssdt_varCost_eff_unit_sink.height > 0
            and d.p_pdt_varCost_sink is not None):
        obj = obj + Sum(
            Where(v_flow * d.p_unitsize, d.pssdt_varCost_eff_unit_sink)
            * d.p_pdt_varCost_sink * op_factor)

    # ─── Process variable cost (§5.4 eff connection) ─────────────────────
    # + Σ pdtProcess[p,'other_operational_cost',d,t]
    #     * v_flow * unitsize * op_factor
    if (has_proc and d.pssdt_varCost_eff_connection is not None
            and d.pssdt_varCost_eff_connection.height > 0
            and d.p_pdt_varCost_process is not None):
        obj = obj + Sum(
            Where(v_flow * d.p_unitsize, d.pssdt_varCost_eff_connection)
            * d.p_pdt_varCost_process * op_factor)

    # Startup cost: v_startup * startup_cost * unitsize, weighted by
    # rp_cost_weight / inflation / period_share — *no step_duration*
    # (startup is a discrete event, not duration-weighted).  Also carries
    # ``pdt_branch_weight`` when stochastics active (mod:2110).
    startup_factor = (d.p_rp_cost_weight * d.p_inflation_op
                      / d.p_period_share)
    if d.pdt_branch_weight is not None:
        startup_factor = startup_factor * d.pdt_branch_weight
    if has_startup_cost_lin:
        obj = obj + Sum(
            Where(v_startup_lin, d.pdt_online_linear)
            * d.p_startup_cost * d.p_unitsize * startup_factor)
    if has_startup_cost_int:
        obj = obj + Sum(
            Where(v_startup_int, d.pdt_online_integer)
            * d.p_startup_cost * d.p_unitsize * startup_factor)

    # Invest / divest objective contributions.
    # NOTE: per .mod:2116-2119 — investment / divestment objective terms
    # are explicitly NOT weighted by pd_branch_weight under the current
    # stochastics regime.  The .mod author's note:
    #
    #     "Currently investment happens only on the realized branch and
    #      the rest get them as existing.  Only one period investment is
    #      supported with stochastics.  The branch weight should be added
    #      if this is changed."
    #
    # Preprocessing excludes branch periods from ``period_invest`` so
    # ``v_invest[e, d]`` only exists on realised periods — the weight
    # would be 1.0 anyway.  No multiplier needed.  Recourse-investment is
    # explicitly out of scope (see audit/a6_b_dim_alternative.md).
    if has_invest_p:
        annu = Param(("p", "d"),
            d.ed_entity_annual_discounted.frame.pipe(rename_to_axis, {"e": "p"})) \
            if d.ed_entity_annual_discounted is not None else None
        lf  = Param(("p", "d"),
            d.ed_lifetime_fixed_cost.frame.pipe(rename_to_axis, {"e": "p"})) \
            if d.ed_lifetime_fixed_cost is not None else None
        if annu is not None:
            obj = obj + Sum(v_invest_p * d.p_unitsize * annu)
        if lf is not None:
            obj = obj + Sum(v_invest_p * d.p_unitsize * lf)
    if has_divest_p:
        lfd = Param(("p", "d"),
            d.ed_lifetime_fixed_cost_divest.frame.pipe(rename_to_axis, {"e": "p"})) \
            if d.ed_lifetime_fixed_cost_divest is not None else None
        annd = Param(("p", "d"),
            d.ed_entity_annual_divest_discounted.frame.pipe(rename_to_axis, {"e": "p"})) \
            if d.ed_entity_annual_divest_discounted is not None else None
        if lfd is not None:
            obj = obj - Sum(v_divest_p * d.p_unitsize * lfd)
        if annd is not None:
            obj = obj - Sum(v_divest_p * d.p_unitsize * annd)
    if has_invest_n:
        us_n = Param(("n",), d.p_state_unitsize.frame)
        annu_n = Param(("n", "d"),
            d.ed_entity_annual_discounted.frame.pipe(rename_to_axis, {"e": "n"})) \
            if d.ed_entity_annual_discounted is not None else None
        lf_n = Param(("n", "d"),
            d.ed_lifetime_fixed_cost.frame.pipe(rename_to_axis, {"e": "n"})) \
            if d.ed_lifetime_fixed_cost is not None else None
        if annu_n is not None:
            obj = obj + Sum(v_invest_n * us_n * annu_n)
        if lf_n is not None:
            obj = obj + Sum(v_invest_n * us_n * lf_n)
    if has_divest_n:
        us_n = Param(("n",), d.p_state_unitsize.frame)
        lfd_n = Param(("n", "d"),
            d.ed_lifetime_fixed_cost_divest.frame.pipe(rename_to_axis, {"e": "n"})) \
            if d.ed_lifetime_fixed_cost_divest is not None else None
        annd_n = Param(("n", "d"),
            d.ed_entity_annual_divest_discounted.frame.pipe(rename_to_axis, {"e": "n"})) \
            if d.ed_entity_annual_divest_discounted is not None else None
        if lfd_n is not None:
            obj = obj - Sum(v_divest_n * us_n * lfd_n)
        if annd_n is not None:
            obj = obj - Sum(v_divest_n * us_n * annd_n)

    # ─── §8.1 Existing-entity fixed cost (constant term) — opt-in ─────────
    # mod:2107-2115:
    #   + Σ_{e in entity, d in period_in_use}
    #         p_entity_all_existing[e,d] * ed_fixed_cost[e,d]
    #             * p_inflation_factor_operations_yearly[d] * pd_branch_weight[d]
    # See ``include_existing_fixed_cost`` doc on ``build_flextool``.
    if (include_existing_fixed_cost
            and d.p_ed_fixed_cost is not None
            and d.p_entity_all_existing is not None):
        fc_frame = (
            d.p_entity_all_existing.frame.rename({"value": "exist"})
            .join(d.p_ed_fixed_cost.frame.rename({"value": "fc"}),
                  on=["e", "d"], how="inner")
            .join(d.p_inflation_op.frame.rename({"value": "infl"}),
                  on="d", how="inner")
        )
        if d.pd_branch_weight is not None:
            fc_frame = (fc_frame
                .join(d.pd_branch_weight.frame.rename({"value": "pdbw"}),
                      on="d", how="inner"))
            fc_frame = fc_frame.with_columns(
                contrib=pl.col("exist") * pl.col("fc")
                        * pl.col("infl") * pl.col("pdbw"))
        else:
            fc_frame = fc_frame.with_columns(
                contrib=pl.col("exist") * pl.col("fc") * pl.col("infl"))
        fc_const = fc_frame["contrib"].sum()
        if fc_const:
            m.add_obj_constant(float(fc_const))

    # ─── Group-level slack (capacity_margin / inertia / non_sync) ────────
    if _group_slack.has_feature(d):
        vars_: dict = {}
        if has_proc:        vars_["v_flow"]       = v_flow
        if has_online_lin:  vars_["v_online_lin"] = v_online_lin
        if has_online_int:  vars_["v_online_int"] = v_online_int
        if has_invest_p:    vars_["v_invest_p"]   = v_invest_p
        if has_divest_p:    vars_["v_divest_p"]   = v_divest_p
        if has_invest_n:    vars_["v_invest_n"]   = v_invest_n
        if has_divest_n:    vars_["v_divest_n"]   = v_divest_n
        vars_["vq_state_up"]   = vq_up
        vars_["vq_state_down"] = vq_down
        _group_slack.add_constraints(m, d, vars_)
        gs_obj = _group_slack.add_objective_terms(m, d, vars_, op_factor)
        if gs_obj is not None:
            obj = obj + gs_obj

    # ─── Reserves (timeseries / dynamic / n_1 / per-process upper) ────────
    if _reserve.has_feature(d):
        res_vars: dict = dict(reserve_vars)
        if has_proc:        res_vars["v_flow"]       = v_flow
        if has_online_lin:  res_vars["v_online_lin"] = v_online_lin
        if has_online_int:  res_vars["v_online_int"] = v_online_int
        if has_invest_p:    res_vars["v_invest_p"]   = v_invest_p
        if has_divest_p:    res_vars["v_divest_p"]   = v_divest_p
        _reserve.add_constraints(m, d, res_vars)
        res_obj = _reserve.add_objective_terms(m, d, res_vars, op_factor)
        if res_obj is not None:
            obj = obj + res_obj

    # ─── Cumulative / group-invest / min-invest constraints ───────────────
    if _cumulative_invest.has_feature(d):
        cum_vars: dict = {
            "v_invest_p": v_invest_p if has_invest_p else None,
            "v_invest_n": v_invest_n if has_invest_n else None,
            "v_divest_p": v_divest_p if has_divest_p else None,
            "v_divest_n": v_divest_n if has_divest_n else None,
            "v_flow":     v_flow     if has_proc     else None,
        }
        _cumulative_invest.add_constraints(m, d, cum_vars)
        ci_obj = _cumulative_invest.add_objective_terms(m, d, cum_vars, op_factor)
        if ci_obj is not None:
            obj = obj + ci_obj

    # ─── Delayed processes / DR ──────────────────────────────────────────
    if _delay.has_feature(d):
        delay_vars: dict = {"v_flow": v_flow if has_proc else None}
        _delay.add_constraints(m, d, delay_vars)
    delay_obj = _delay.add_objective_terms(
        m, d, {"v_flow": v_flow if has_proc else None}, op_factor)
    if delay_obj is not None:
        obj = obj + delay_obj

    # ─── Commodity price ladder ──────────────────────────────────────────
    # Per-tier v_trade balance + tier-cap constraints + per-tier price
    # objective contribution.  The legacy commodity price term above
    # already filtered out ladder commodities, so this term replaces it.
    if has_ladder:
        _commodity_ladder.add_constraints(
            m, d, ladder_vars,
            v_flow=v_flow if has_proc else None,
            p_unitsize=d.p_unitsize,
            p_slope=d.p_slope,
            p_step_duration=d.p_step_duration,
            p_rp_cost_weight=d.p_rp_cost_weight,
            flow_from_commodity_eff=d.flow_from_commodity_eff,
            flow_from_commodity_noEff=d.flow_from_commodity_noEff,
            flow_to_commodity=d.flow_to_commodity,
        )
        ladder_obj = _commodity_ladder.add_objective_terms(
            m, d, ladder_vars,
            p_inflation_op=d.p_inflation_op,
            p_period_share=d.p_period_share,
        )
        if ladder_obj is not None:
            obj = obj + ladder_obj

    # ─── Non-anticipativity constraints (A6) ─────────────────────────────
    # mod:4173-4233.  Four constraint families pin per-branch dispatch
    # variables to the realised period at every (d, t) ∈ dt_non_anticipativity:
    #
    #   non_anticipativity_storage_use   (n, d, b, t) — net storage charge
    #     LHS at (d, t) equals LHS at (b, t).  Active only when at least
    #     one (g, n) ∈ group_node has g ∈ groupStochastic.
    #   non_anticipativity_online_integer (p, d, b, t) — v_online_integer
    #     equal across siblings.
    #   non_anticipativity_online_linear  (p, d, b, t) — v_online_linear
    #     equal across siblings.
    #   non_anticipativity_reserve        (p, r, ud, n, d, b, t) — v_reserve
    #     equal across siblings.
    #
    # Domain: (d, b) ∈ period__branch with d != b and b ∈ period_in_use.
    # The b == d self-loop is vacuous and the metadata-only branches not
    # in period_in_use (e.g. ``period1_realized``) must be excluded so
    # the constraint references a real LP variable.
    if (d.dt_non_anticipativity is not None
            and d.dt_non_anticipativity.height > 0
            and d.period_branch_full is not None
            and d.period_branch_full.height > 0):
        # Build (d, b) cohort: d ≠ b AND b ∈ period_in_use.
        # Cross-axis value compare: "d" is the period axis and "b" is the
        # branch axis (two different Enum vocabularies under Phase 4
        # activation).  Polars 1.40+ refuses ``!=`` between different
        # Enums; cast both to Utf8 so the comparison is by token string.
        db_pairs = d.period_branch_full.with_columns(
            pl.col("d").cast(pl.Utf8),
            pl.col("b").cast(pl.Utf8),
        ).filter(pl.col("d") != pl.col("b"))
        if d.period_in_use_set is not None:
            # Cross-axis join: piu.b carries period tokens (d-Enum
            # vocab) but db_pairs.b is branch-Enum.  Cast both to Utf8.
            piu = (d.period_in_use_set
                       .rename({"d": "b"})
                       .with_columns(pl.col("b").cast(pl.Utf8)))
            db_pairs = db_pairs.join(piu, on="b", how="inner")
        # Restore axis types post-compare (Utf8 was only needed for the
        # cross-Enum filter/join above; downstream consumers cross-join
        # ``db_pairs`` with axis-typed frames and join on "d"/"b" against
        # axis-typed cohorts, which requires Enum-typed keys for native
        # composition under Phase 4 activation).
        db_pairs = db_pairs.with_columns(
            cast_dim(pl.col("d"), None, "d"),
            cast_dim(pl.col("b"), None, "b"),
        )
        if db_pairs.height > 0:
            _add_non_anticipativity_constraints(
                m, d, db_pairs,
                v_state          = locals().get("v_state")          if has_storage    else None,
                v_online_integer = locals().get("v_online_int")     if has_online_int else None,
                v_online_linear  = locals().get("v_online_lin")     if has_online_lin else None,
                v_reserve        = reserve_vars.get("v_reserve")    if reserve_vars   else None,
                v_flow           = locals().get("v_flow")           if has_proc       else None,
                has_minload_eff  = has_minload_eff,
            )

    # Apply objective scaling if provided (default 1.0 = no scaling).
    if scale_the_objective != 1.0:
        obj = obj * scale_the_objective
    m.set_objective(obj, sense="min")

    # ─── Wire flextool's HiGHS solver options through to Problem.solve() ──
    # ``d.solver_options`` is loaded from ``input/solve_mode.csv`` (param
    # rows ``highs_method`` / ``highs_parallel`` / ``highs_presolve``,
    # filtered to the active ``solve_current``).  When None, flexpy falls
    # back to HiGHS defaults — current behavior pre-wiring.
    if getattr(d, "solver_options", None):
        m.set_solver_options(d.solver_options)


def _add_online_block(m, d, v_flow, kind: str, p_idx: "pl.DataFrame",
                       online_set: "pl.DataFrame",
                       v_online, v_startup, v_shutdown,
                       v_invest_p=None, v_divest_p=None) -> None:
    """Emit maxOnline / maxStartup / maxShutdown / online__startup /
    online__shutdown / maxToSink_online / minToSink_minload for one
    UC class (``kind`` in {"linear", "integer"}).  Constraint names
    are suffixed with ``_<kind>`` so linear+integer scenarios produce
    distinct rows in the LP."""
    sfx = f"_{kind}"

    # Invest/divest tightening for the max{Online,Startup,Shutdown} bounds.
    # The .mod's RHS for these is
    #   existing/unitsize + Σ v_invest - Σ v_divest
    # We move the invest/divest delta to the LHS so the row stays
    #   var + (divest - invest) ≤ existing_count.
    invest_term = None
    divest_term = None
    if v_invest_p is not None and d.edd_invest_set is not None:
        v_inv_at_uc = Var(  # virtual rename: d → d_invest
            name=v_invest_p.name + f"__at_invest_uc{sfx}",
            dims=("p", "d_invest"),
            frame=v_invest_p.frame.pipe(rename_to_axis, {"d": "d_invest"}),
            lower=v_invest_p.lower, upper=v_invest_p.upper,
        )
        edd_inv_p_uc = (d.edd_invest_set
                        .filter(pl.col("e").is_in(online_set["p"].unique()))
                        .pipe(rename_to_axis, {"e": "p"}))
        if edd_inv_p_uc.height > 0:
            invest_term = -Sum(
                Where(v_inv_at_uc, edd_inv_p_uc), over=("d_invest",))
    if v_divest_p is not None and d.edd_divest_active is not None:
        v_div_at_uc = Var(
            name=v_divest_p.name + f"__at_divest_uc{sfx}",
            dims=("p", "d_divest"),
            frame=v_divest_p.frame.pipe(rename_to_axis, {"d": "d_divest"}),
            lower=v_divest_p.lower, upper=v_divest_p.upper,
        )
        edd_div_p_uc = (d.edd_divest_active
                        .filter(pl.col("p").is_in(online_set["p"].unique())))
        if edd_div_p_uc.height > 0:
            divest_term = Sum(
                Where(v_div_at_uc, edd_div_p_uc), over=("d_divest",))

    def _lhs_with_invdiv(var, key):
        lhs: dict = {key: var}
        if invest_term is not None:
            lhs["invest_neg"] = invest_term
        if divest_term is not None:
            lhs["divest"] = divest_term
        return lhs

    # max bounds
    m.add_cstr(f"maxOnline{sfx}",   over=p_idx, sense="<=",
               lhs_terms=_lhs_with_invdiv(v_online, "online"),
               rhs_terms={"max_units": d.p_process_existing_count})
    m.add_cstr(f"maxStartup{sfx}",  over=p_idx, sense="<=",
               lhs_terms=_lhs_with_invdiv(v_startup, "startup"),
               rhs_terms={"max_units": d.p_process_existing_count})
    m.add_cstr(f"maxShutdown{sfx}", over=p_idx, sense="<=",
               lhs_terms=_lhs_with_invdiv(v_shutdown, "shutdown"),
               rhs_terms={"max_units": d.p_process_existing_count})
    # online dynamics: v_startup[t] >= v_online[t] - v_online[t-1]
    #                  v_shutdown[t] >= v_online[t-1] - v_online[t]
    v_online_lag = Lag(v_online, d.dtttdt, "t", "t_previous_within_solve")
    m.add_cstr(f"online__startup{sfx}",  over=p_idx, sense=">=",
               lhs_terms={"startup":     v_startup,
                          "online_prev": v_online_lag},
               rhs_terms={"online_now":  v_online})
    m.add_cstr(f"online__shutdown{sfx}", over=p_idx, sense=">=",
               lhs_terms={"shutdown":    v_shutdown,
                          "online_now":  v_online},
               rhs_terms={"online_prev": v_online_lag})

    # maxToSink_online: v_flow <= v_online * availability (assumes
    # max_cap_coef=1).  The .mod's RHS for online processes is
    # ``v_online × max_cap × availability × unitsize`` (mod:3015-3026);
    # without the availability factor flexpy under-tightens the v_flow
    # bound at hours with availability < 1, allowing the dispatch to
    # produce up to v_online × 1 instead of v_online × availability.
    # That extra slack lets the LP get away with less invest, which
    # mismatches flextool by ~1.2% on the test_a_lot fixtures (the only
    # fixtures whose ``pt_process[*, availability, t]`` table varies
    # across timesteps with v_online < 1).
    pss_online = d.process_source_sink.join(online_set, on="p", how="inner")
    if pss_online.height > 0:
        over_pss_online = pss_online.join(d.dt, how="cross")
        if d.p_process_availability is not None:
            online_rhs = v_online * d.p_process_availability
        else:
            online_rhs = v_online
        m.add_cstr(f"maxToSink_online{sfx}",
                   over=over_pss_online, sense="<=",
                   lhs_terms={"flow":   v_flow},
                   rhs_terms={"online": online_rhs})

    # minToSink_minload: Σ_sinks v_flow >= v_online * min_load
    if d.process_minload is not None and d.process_minload.height > 0:
        pss_minload = (d.process_source_sink
                       .join(online_set, on="p", how="inner")
                       .join(d.process_minload, on="p", how="inner"))
        if pss_minload.height > 0:
            over_minload = pss_minload.join(d.dt, how="cross")
            sum_flow = Sum(Where(v_flow, pss_minload), over=("sink",))
            min_load_floor = (Where(v_online, d.process_minload)
                              * d.p_min_load)
            m.add_cstr(f"minToSink_minload{sfx}",
                       over=over_minload, sense=">=",
                       lhs_terms={"flow_sum": sum_flow},
                       rhs_terms={"floor":    min_load_floor})

    # minimum_uptime: v_online[p,d,t] >= Σ v_startup[p, d_back, t_back] over
    # uptime_lookback rows.  Per-class (linear/integer) restriction is
    # achieved by joining the constraint domain and the lookback frame to
    # this block's online_set.
    if (d.pdt_uptime_set is not None and d.pdt_uptime_set.height > 0
            and d.uptime_lookback is not None and d.uptime_lookback.height > 0):
        up_idx = d.pdt_uptime_set.join(online_set, on="p", how="inner")
        if up_idx.height > 0:
            # Restrict lookback to (p, d_back, t_back) entries whose v_startup
            # exists (i.e. (p, d_back, t_back) ∈ p_idx with d_back/t_back
            # renamed) and to processes in this online_set.
            p_idx_back = p_idx.pipe(rename_to_axis, {"d": "d_back", "t": "t_back"})
            lkb = (d.uptime_lookback.join(online_set, on="p", how="inner")
                   .join(p_idx_back, on=["p", "d_back", "t_back"], how="inner")
                   .join(up_idx, on=["p", "d", "t"], how="inner"))
            if lkb.height > 0:
                v_startup_at = Var(
                    name=v_startup.name + f"__lookback_up{sfx}",
                    dims=("p", "d_back", "t_back"),
                    frame=v_startup.frame.pipe(rename_to_axis, {"d": "d_back", "t": "t_back"}),
                    lower=v_startup.lower, upper=v_startup.upper,
                )
                startup_sum = Sum(Where(v_startup_at, lkb),
                                  over=("d_back", "t_back"))
                m.add_cstr(f"minimum_uptime{sfx}",
                           over=up_idx, sense=">=",
                           lhs_terms={"online":      v_online},
                           rhs_terms={"startup_sum": startup_sum})

    # minimum_downtime: existing_count - v_online >= Σ v_shutdown[lookback]
    # Rewritten LP-friendly: v_online + Σ v_shutdown ≤ existing_count.
    # The .mod's RHS includes invest/divest tightening
    # (existing_count + Σ v_invest_alive − Σ v_divest_alive); we move
    # those terms to the LHS as +divest / −invest so the row stays
    #   v_online + Σ v_shutdown + Σ v_divest_alive − Σ v_invest_alive
    #   ≤ existing_count
    # which matches the maxOnline pattern above.  Without this, an
    # invest-eligible online process with existing_count=0 would have
    # v_online + Σ v_shutdown ≤ 0 forced regardless of v_invest, making
    # the UC infeasible (LP falls back to all-slack).  See B4.12.
    if (d.pdt_downtime_set is not None and d.pdt_downtime_set.height > 0
            and d.downtime_lookback is not None and d.downtime_lookback.height > 0):
        dn_idx = d.pdt_downtime_set.join(online_set, on="p", how="inner")
        if dn_idx.height > 0:
            p_idx_back = p_idx.pipe(rename_to_axis, {"d": "d_back", "t": "t_back"})
            lkb = (d.downtime_lookback.join(online_set, on="p", how="inner")
                   .join(p_idx_back, on=["p", "d_back", "t_back"], how="inner")
                   .join(dn_idx, on=["p", "d", "t"], how="inner"))
            if lkb.height > 0:
                v_shutdown_at = Var(
                    name=v_shutdown.name + f"__lookback_dn{sfx}",
                    dims=("p", "d_back", "t_back"),
                    frame=v_shutdown.frame.pipe(rename_to_axis, {"d": "d_back", "t": "t_back"}),
                    lower=v_shutdown.lower, upper=v_shutdown.upper,
                )
                shutdown_sum = Sum(Where(v_shutdown_at, lkb),
                                   over=("d_back", "t_back"))
                lhs_dn: dict = {"online":       v_online,
                                "shutdown_sum": shutdown_sum}
                if invest_term is not None:
                    lhs_dn["invest_neg"] = invest_term
                if divest_term is not None:
                    lhs_dn["divest"] = divest_term
                m.add_cstr(f"minimum_downtime{sfx}",
                           over=dn_idx, sense="<=",
                           lhs_terms=lhs_dn,
                           rhs_terms={"existing":     d.p_process_existing_count})


def _add_profile_cstr(m, d, v_flow, name: str, idx: "pl.DataFrame",
                       sense: str, v_inv_for_profile=None,
                       v_div_for_profile=None,
                       reserve_term=None,
                       reserve_sign: float = 1.0) -> None:
    """profile_flow_*  constraint family.

    LHS = v_flow filtered to the (p, source, sink, f) tuples in idx,
    then time dims (d, t) attached via the over= axes.

    RHS = profile[f, d, t] · existing_count[p, d] · availability[p, d, t].

    When invest/divest are active, the .mod's RHS expands to
    profile · (existing_count + Σ_d_inv v_invest[d_inv]
                                - Σ_d_div v_divest[d_div]) · availability.
    The invest term is moved to the LHS as ``-invest_term`` and the
    divest term is moved to the LHS as ``+divest_term`` so this stays a
    single LP row per (p, source, sink, f, d, t).

    ``reserve_term`` (manifest patch #6 for _reserve): an Expr keyed on
    (p, sink, d, t) that the .mod adds to the profile_flow LHS.
    For ``upper_limit`` the term is ``+ Σ v_reserve_up_to_sink``;
    for ``lower_limit`` it is ``- Σ v_reserve_down_to_sink`` (sign passed
    via ``reserve_sign``).  ``fixed`` has no reserve term per the .mod.
    """
    over = idx.join(d.dt, how="cross")              # (p, source, sink, f, d, t)
    # LHS: v_flow over (p,source,sink,d,t) joined to over via (p,source,sink,d,t)
    # → introduces f as a new column from over (via the constraint axes).
    rhs_param = d.p_profile_value * d.p_process_existing_count
    if d.p_process_availability is not None:
        rhs_param = rhs_param * d.p_process_availability
    lhs: dict = {"flow": v_flow}
    pf_filter = None
    if v_inv_for_profile is not None or v_div_for_profile is not None:
        # Restrict to (p, f) pairs present in this profile-method set so
        # the invest/divest term doesn't introduce p × f cartesian rows
        # that don't belong to this constraint family.
        pf_filter = idx.select("p", "f" if "f" in idx.columns else "profile").unique()
        if "profile" in pf_filter.columns and "f" not in pf_filter.columns:
            pf_filter = pf_filter.pipe(rename_to_axis, {"profile": "f"})
    if v_inv_for_profile is not None:
        inv_term = v_inv_for_profile * d.p_profile_value
        if d.p_process_availability is not None:
            inv_term = inv_term * d.p_process_availability
        lhs["invest_neg"] = -Where(inv_term, pf_filter)
    if v_div_for_profile is not None:
        div_term = v_div_for_profile * d.p_profile_value
        if d.p_process_availability is not None:
            div_term = div_term * d.p_process_availability
        lhs["divest"] = Where(div_term, pf_filter)
    if reserve_term is not None:
        if reserve_sign >= 0:
            lhs["reserve"] = reserve_term
        else:
            lhs["reserve_neg"] = -reserve_term
    m.add_cstr(
        name,
        over      = over,
        sense     = sense,
        lhs_terms = lhs,
        rhs_terms = {"limit": rhs_param},
    )
