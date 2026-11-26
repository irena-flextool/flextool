"""The :class:`FlexData` container (fields, ``dump_csvs``) and its test-only cascade gate."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
import polars as pl
from polar_high import Param
from flextool.engine.preprocess.params._block_layout import BlockLayout


# ---------------------------------------------------------------------------
# Diagnostic gate for `__setattr__` on FlexData.  Production code never
# arms this — it's a tool for ``tests/engine/test_native_cascade_parity.py``
# (and ad-hoc investigation) to compare the cascade's native-override
# objective against the CSV-seed-preserved objective on the same
# fixture.  See ``specs/native_cascade_parity.md`` for the bug class
# this diagnoses.
#
# When ``_CASCADE_GATE_ACTIVE`` is True (set via ``_cascade_gate()``),
# ``FlexData.__setattr__`` drops any reassignment of a field that
# already carries a non-None value.  Construction, handoff overlay,
# and ``apply_existing_chain`` are unaffected — only direct
# assignments inside the gated context window are filtered.

_CASCADE_GATE_ACTIVE = False


@contextmanager
def _cascade_gate():
    """Test-only: while active, ``FlexData.__setattr__`` drops
    reassignments of fields that already have a seeded value.  Used
    by the cascade parity test to A/B the cascade override path
    against the CSV-seed preservation path.
    """
    global _CASCADE_GATE_ACTIVE
    prev = _CASCADE_GATE_ACTIVE
    _CASCADE_GATE_ACTIVE = True
    try:
        yield
    finally:
        _CASCADE_GATE_ACTIVE = prev


# ---------------------------------------------------------------------------
# Single FlexData container

@dataclass
class FlexData:
    """Naming convention:

    * **Sets** (index frames, ``pl.DataFrame``) — no prefix.
      Examples: ``nodeBalance``, ``process_source_sink``, ``flow_to_n``,
      ``cdt_eq``, ``nodeState``.
    * **Parameters** (numeric ``Param``) — ``p_`` prefix.
      Examples: ``p_inflow``, ``p_unitsize``, ``p_commodity_price``.

    Variables created by ``build_flextool`` use ``v_`` for primal,
    ``vq_`` for slack — same convention as flextool.mod.
    """

    # ─── Time / weighting (always present) ────────────────────────────────
    dt: pl.DataFrame                         # set: (d, t)
    p_step_duration: Param                   # (d, t)
    p_timestep_weight: Param                 # (d, t)
    p_inflation_op: Param                    # (d,)
    p_period_share: Param                    # (d,)

    # ─── Nodes (always present in tested scenarios) ───────────────────────
    nodeBalance: pl.DataFrame                # set: (n,)
    p_inflow: Param                          # (n, d, t)
    p_penalty_up: Param                      # (n, d, t)
    p_penalty_down: Param                    # (n, d, t)

    # Companion to ``nodeBalance`` above: the subset of balance nodes whose
    # balance is enforced once per period (legacy ``balance_within_period``)
    # rather than per (d, t).  Defaults to ``None`` (kept here in the
    # default-field region so dataclass ordering stays valid next to the
    # non-default node fields above); ``load_flextool`` always supplies an
    # (empty when the CSV is absent / header-only) frame, so existing
    # FlexData(...) constructors that omit it stay valid and unchanged.
    nodeBalancePeriod: pl.DataFrame | None = None  # set: (n,)

    # Companion to ``nodeBalance``: the subset of balance nodes whose
    # ``penalty_method == 'off'`` — these get NO balance-slack
    # (``vq_state_up`` / ``vq_state_down``) variables, so their node
    # balance is a hard equality.  ``None`` (default) or empty means every
    # balance node keeps its slack (byte-identical legacy behaviour).
    nodeBalance_penalty_off: pl.DataFrame | None = None  # set: (n,)

    # Model-wide small-coefficient cutoff (``model.small_number_threshold``).
    # Any LP constraint-matrix coefficient or RHS term with
    # ``abs(value) < threshold`` is floored to 0.0 inside polar_high at LP
    # assembly time (set on ``Problem.coef_zero_threshold`` by
    # ``build_flextool``).  Default ``0.0001`` mirrors the schema default;
    # ``load_flextool`` overrides it from ``p_small_number_threshold.csv``
    # when present.  A value of ``0.0`` disables the cutoff.
    small_number_threshold: float = 0.0001

    # Per-period years-represented R (e.g. 5.0 for a 5-year invest period).
    # Built from ``solve.years_represented`` via
    # ``p_years_represented_d_from_source``.  None when the source
    # carries no ``solve.years_represented`` rows (single-year fixtures
    # default each period to width 1).
    p_years_represented_d: Param | None = None  # (d,)

    # Share of each period's complete-solve hours that this (sub-)solve's own dt covers:
    # (Σ_t step_duration[d,t] / 8760) / p_period_share[d], snapped to exactly 1.0 when the two
    # agree within 1e-9 relative.  1.0 except in the rolls of a rolling solve, where p_period_share
    # is the complete (parent) solve's share.  Scales the co2_max_period cap to the roll's window.
    # Set by ``apply_derived_a`` (exactly 1.0) or at the load boundary of ``load_flextool``.
    p_window_fraction: Param | None = None  # (d,)

    # ─── Process topology  ───────────────────────────────────────────────
    process_source_sink: pl.DataFrame | None = None
    process_source_sink_eff: pl.DataFrame | None = None
    process_source_sink_noEff: pl.DataFrame | None = None
    # Phase E.3: ``pss_dt`` / ``nodeBalance_dt`` / ``nodeState_dt`` /
    # ``nodeState_first_dt`` / ``process_indirect_dt`` are no longer
    # eager-built up-front.  Consumers call the on-demand helpers in
    # :mod:`flextool.engine.model._pdt_join`.  Fields stay declared
    # (defaulting to ``None``) so callers that defensively read them via
    # ``getattr(d, ..., None)`` continue to work.
    pss_dt: pl.DataFrame | None = None
    nodeBalance_dt: pl.DataFrame | None = None
    # Canonical (p, source) / (p, sink) per process — one row per unit input
    # node / output node, and one row per connection using the original
    # connection__node__node direction (not the added reverse arc).
    process_source_canonical: pl.DataFrame | None = None
    process_sink_canonical: pl.DataFrame | None = None
    flow_to_n: pl.DataFrame | None = None
    flow_from_n: pl.DataFrame | None = None
    flow_from_commodity_eff: pl.DataFrame | None = None
    flow_from_commodity_noEff: pl.DataFrame | None = None
    flow_to_commodity: pl.DataFrame | None = None  # §2.4 sell into priced commodity node
    p_unitsize: Param | None = None              # (p,)
    p_all_entity_unitsize: Param | None = None  # (e,) — all entities (processes + connections + nodes); used by scaling
    p_flow_upper: Param | None = None            # (p, source, sink, d, t) — preprocessed structural max (existing + max_invest_cum)
    p_flow_upper_existing: Param | None = None   # (p, source, sink, d) — existing unit count (capacity/unitsize) only; used by maxFlow
    p_arc_max_cap_coef: Param | None = None      # (p, source, sink) — per-arc capacity factor for DIRECT + indirect OUTPUT arcs (maxToSink: capacity_max_coeff, min'd with an authored input_share_max on direct units; maxFromSource: input_share_max); folded onto p_flow_upper_existing in maxFlow
    p_indirect_input_cap: Param | None = None    # (p, source, sink, d, t) — indirect-unit INPUT arcs: input_share_max × full-load input flow per unit of capacity; × availability × capacity = the arc's limit (v_flow upper bound, or maxFlow row when the unit invests/divests)
    process_source_sink_uncapped: pl.DataFrame | None = None  # (p, source, sink) — conversion_flow_coeff = 0 arcs: outside every per-edge capacity / ramp / min-load constraint
    p_process_sink_max_capacity_coef: Param | None = None    # (p, sink) — unit__outputNode.capacity_max_coeff (explicit rows; default 1.0): ramp / maxFlow_online multiplier
    p_process_source_input_share_max: Param | None = None  # (p, source) — unit__inputNode.input_share_max (explicit rows; default 1.0): source-side ramp multiplier (non-indirect arcs)
    p_process_source_input_share_min: Param | None = None  # (p, source) — unit__inputNode.input_share_min (explicit rows; default 0): minInputShare mixing minimum of indirect units
    p_slope: Param | None = None                 # (p, d, t)
    p_commodity_price: Param | None = None       # (c, d, t)
    pd_neg_cap: pl.DataFrame | None = None       # set: (p, d) where existing<0 AND unitsize<0
                                                  # (anti-energy semantics: forces v_flow ≥ |existing|/|unitsize|)

    # ─── CO2 price ────────────────────────────────────────────────────────
    flow_from_co2_priced: pl.DataFrame | None = None
    flow_from_co2_priced_noEff: pl.DataFrame | None = None
    # (#19) Removal credit: sink-side flow into a CO2-priced commodity node
    # — (p, source, sink, c, g), no eff/noEff split (straight v_flow *
    # unitsize, mirrors flow_to_commodity; GMPL .mod:2047-2050 sums the
    # unsplit process_source_sink, no slope).
    flow_to_co2_priced: pl.DataFrame | None = None
    p_co2_content: Param | None = None           # (c,)
    p_co2_price: Param | None = None             # (g, d, t)

    # ─── CO2 cap (period) ─────────────────────────────────────────────────
    group_co2_max_period: pl.DataFrame | None = None
    flow_from_co2_capped: pl.DataFrame | None = None        # eff partition (slope)
    flow_from_co2_capped_noEff: pl.DataFrame | None = None  # noEff partition (no slope)
    # (#19) Removal credit: sink-side flow into a CO2-capped node of the same
    # group — (p, source, sink, c, g), unsplit (no slope); reduces the cap's
    # LHS (GMPL .mod:3988-3991).
    flow_to_co2_capped: pl.DataFrame | None = None
    p_co2_max_period: Param | None = None        # (g, d)
    group_d_co2_capped: pl.DataFrame | None = None

    # ─── CO2 cap (multi-period total) — port of v3.32.0 co2_max_total ────
    # Mirrors co2_max_period but the cap is a single tonnes value per group
    # spanning the whole horizon.  Active only for groups whose
    # ``group__co2_method`` is ``total`` / ``price_total`` / ``period_total``
    # (set materialised in ``solve_data/group_co2_max_total.csv``).  LHS sums
    # across (d, t); RHS is the per-group ``p_group[g, 'co2_max_total']``
    # scalar (in tonnes).  See .mod:4019-4055 for the legacy formulation.
    group_co2_max_total: pl.DataFrame | None = None              # (g,)
    flow_from_co2_capped_total: pl.DataFrame | None = None       # eff partition
    flow_from_co2_capped_total_noEff: pl.DataFrame | None = None # noEff partition
    # (#19) Removal credit for the total cap, sibling of flow_to_co2_capped
    # (GMPL .mod:4047-4052).
    flow_to_co2_capped_total: pl.DataFrame | None = None
    p_co2_max_total: Param | None = None         # (g,)

    # ─── CO2 cap rolling / chained carry (#38, Q22) ────────────────────────
    # co2_max_period and co2_max_total each got the full cap again on every roll / chained solve (no
    # accumulator fed the LP) — over-allowing roughly by the number of rolls.  The owner's fix (Q15,
    # reused for Q22) mirrors the commodity ladder's realized-based carried budget: this roll's cap is
    # "budget allocated up to the end of this roll's horizon minus what earlier rolls/solves realized",
    # never negative by construction.  p_co2_f_d_k is the raw solve_data/f_d_k.csv Param (d,) — the
    # same per-period "share of d's hours reached by the end of this roll's horizon" the ladder's
    # annual tier cap uses, read independently of the ladder gate (f_d_k.csv is always emitted).
    # p_co2_cum_realized_tonnes (g, d) is the cumulative_co2 handoff carrier, net of removal credits
    # (#19). co2_horizon_fraction_reached is the whole-chain fraction (years_represented-weighted) that
    # p_co2_max_total scales by. See _co2_carry.load_co2_rolling_carry and model/core/'s co2_max_period /
    # co2_max_total EQ headers.
    p_co2_f_d_k: Param | None = None             # (d,)
    p_co2_cum_realized_tonnes: Param | None = None  # (g, d)
    # The same realized tonnes in co2_max_total's units: × years_represented[d] / period_share[d] of the
    # solve that realized them (the weight co2_max_total's LHS gives them); value null where unknown.
    p_co2_cum_realized_represented: Param | None = None  # (g, d)
    co2_horizon_fraction_reached: float = 1.0
    # (d,) hours of the solve's periods that earlier solves of the chain realized (item 1042; _co2_carry).
    p_co2_hours_before_solve: Param | None = None
    flowgroup_realized: pl.DataFrame | None = None  # (g, d) realized flow-group history, _flowgroup_accumulators

    # ─── Indirect-conversion (CHP) ────────────────────────────────────────
    process_indirect: pl.DataFrame | None = None
    process_input_flows: pl.DataFrame | None = None
    process_output_flows: pl.DataFrame | None = None
    process_indirect_dt: pl.DataFrame | None = None
    # Per-arc multipliers on the source / sink side of the
    # ``conversion_indirect`` equation (.mod:2557-2580).  Both default to
    # 1.0 (the absent-Param convention) and are only populated when at
    # least one row in the corresponding
    # ``input/p_process_*_conversion_flow_coeff.csv`` has a non-default,
    # non-zero value.  When populated, the Param covers *all* relevant
    # (p, source) / (p, sink) rows of the indirect inputs / outputs
    # (filled to 1.0 where the CSV is silent), so multiplying
    # ``v_flow * unitsize * Param`` won't drop any flows.
    p_process_source_conversion_flow_coeff: Param | None = None  # (p, source)
    p_process_sink_conversion_flow_coeff: Param | None = None    # (p, sink)

    # ─── User-defined flow constraints ────────────────────────────────────
    flow_constraint_idx: pl.DataFrame | None = None  # (p, source, sink, cn)
    p_flow_constraint_coef: Param | None = None  # (p, source, sink, cn)
    p_constraint_constant: Param | None = None   # (cn,), or (cn, d) when some constraint has a period map
    cdt_eq: pl.DataFrame | None = None  # (cn, d, t)
    cdt_le: pl.DataFrame | None = None  # (cn, d, t)
    cdt_ge: pl.DataFrame | None = None  # (cn, d, t)
    # Constraints with constraint_period_method period_total (one annualised row per period); None = all per_timestep.
    constraint_period_total: pl.DataFrame | None = None  # set: (cn,)
    # A rolling solve whose rolls do not each realize whole periods leaves the period_total rows out
    # (solve_data/user_constraint_period_total_off, written per sub-solve by _run_model).
    user_constraint_period_total_off: bool = False
    p_node_constraint_invested_capacity_coeff: Param | None = None  # (n, cn)
    p_process_constraint_invested_capacity_coeff: Param | None = None  # (p, cn)
    p_node_constraint_state_coeff: Param | None = None  # (n, cn) — user-cstr v_state coefficient
    p_node_constraint_prebuilt_capacity_coeff: Param | None = None  # (n, cn)
    p_process_constraint_prebuilt_capacity_coeff: Param | None = None  # (p, cn)
    p_node_constraint_capacity_coeff: Param | None = None  # (n, cn) — storage capacity in service
    p_process_constraint_capacity_coeff: Param | None = None  # (p, cn) — capacity in service
    p_node_constraint_retired_capacity_coeff: Param | None = None  # (n, cn) — storage capacity retired in d
    p_process_constraint_retired_capacity_coeff: Param | None = None  # (p, cn) — capacity retired in d
    p_node_constraint_in_construction_capacity_coeff: Param | None = None  # (n, cn) — storage in construction
    p_process_constraint_in_construction_capacity_coeff: Param | None = None  # (p, cn) — capacity in construction
    p_process_constraint_online_coeff: Param | None = None  # (p, cn) — units online (count of unitsize)

    # ─── Profiles ─────────────────────────────────────────────────────────
    process_profile_upper: pl.DataFrame | None = None    # (p, source, sink, f)
    process_profile_lower: pl.DataFrame | None = None
    process_profile_fixed: pl.DataFrame | None = None
    p_profile_value: Param | None = None         # (f, d, t)
    p_process_existing_count: Param | None = None  # (p, d) = cap / unitsize
    p_process_availability: Param | None = None  # (p, d, t)

    # ─── Invest / divest ──────────────────────────────────────────────────
    ed_invest_set: pl.DataFrame | None = None        # (e, d) — invest var index
    ed_divest_set: pl.DataFrame | None = None        # (e, d) — divest var index
    pd_invest_set: pl.DataFrame | None = None        # (p, d) — process-side
    pd_divest_set: pl.DataFrame | None = None        # (p, d)
    nd_invest_set: pl.DataFrame | None = None        # (n, d) — node-side
    nd_divest_set: pl.DataFrame | None = None        # (n, d)
    # The cohort table (preprocess/params/_cohorts.py; apply_derived_c / apply_synthetic_invest_sets, None in a
    # solve without investment or retirement; the edd_invest seed of a reload without the investment walk):
    # one row per block of capacity (the existing capacity of an entity, each investment decision) with its start
    # of operation and end of life; the periods each cohort is in service; (d_retire, d) the periods a retirement
    # decided in d_retire counts in.  The model's capacity in service is built from them (model/_capacity.py).
    cohort: pl.DataFrame | None = None
    cohort_alive: pl.DataFrame | None = None
    dd_forward: pl.DataFrame | None = None
    # (e, d, d_c, yr_c) — start of operation d_c (a run period; yr_c its year on the solve calendar) of each
    # offered investment decision d with a construction lead time; None without lead times (apply_derived_c).
    ed_invest_commission: pl.DataFrame | None = None
    # Construction lead time (apply_derived_c; all None when no investment decision of the solve has one):
    # (e, d_invest, d) the period d in which the capacity of decision d_invest starts to operate ((e, o, o) for a
    # decision without a lead time) and the periods d in which it is in construction; (e, L) the entities with a
    # lead time and their longest lead time (years).
    edd_invest_commission: pl.DataFrame | None = None
    edd_invest_in_construction: pl.DataFrame | None = None
    e_construction_lead_time: pl.DataFrame | None = None
    # (e, d) MW (MWh): investments realized by earlier solves of a chain with a construction lead time that start to
    # operate in d / are in construction in d (apply_existing_chain, from the handoff realized_commission).
    p_entity_carried_commission: Param | None = None
    p_entity_carried_in_construction: Param | None = None
    # (e, d) MW (MWh): existing capacity earlier solves of a chain retired that still counts in d (while the existing
    # cohort is in service; emitted from the handoff realized_retire, read by apply_existing_chain).
    p_entity_retired_carried: Param | None = None
    p_entity_max_units: Param | None = None          # (e, d)
    ed_lifetime_fixed_cost: Param | None = None      # (e, d)
    ed_lifetime_fixed_cost_divest: Param | None = None
    ed_entity_annual_discounted: Param | None = None
    # (e, d) the salvage term of a retirement of existing capacity (D(d)·S_d − a_x·S̄_x, CUR/MW) and (e, d) the level
    # salvage credit of existing capacity in d (a_x·inflation_inv[d], the objective constant with
    # include_existing_fixed_cost): ``_existing_salvage.apply_retirement_salvage``.
    ed_divest_salvage: Param | None = None
    ed_existing_salvage_credit: Param | None = None
    # (e, d) the yearly price of the replacement of existing reinvest_automatic capacity in service in d times
    # inflation_inv[d] (CUR/MW; the objective constant with include_existing_fixed_cost):
    # ``_existing_salvage.apply_existing_replacement``.
    ed_existing_replacement: Param | None = None
    # Retirement of model-built capacity (preprocess/params/_retire_built.py; apply_derived_c, None without an
    # option): (e, cohort, d) the retirement options (the columns of v_divest_built), (e, cohort, d_divest, d) the
    # periods a retirement decided in d_divest counts in, one row per retirable cohort (its decision period in this
    # solve or the carried capacity it has left, its bound, years and prices); (e, cohort, d) the settlement of a
    # retired MW less the level price taken back and the fixed cost it saves (CUR/MW, MWh;
    # ``_retire_settlement.apply_built_retirement_costs``).
    cd_divest_built: pl.DataFrame | None = None
    cdd_divest_built: pl.DataFrame | None = None
    cohort_divest_built: pl.DataFrame | None = None
    cd_divest_built_cost: Param | None = None
    cd_divest_built_fixed: Param | None = None
    # Life extension (preprocess/params/_extend.py; apply_derived_c, None without an option): (e, cohort, d) the
    # extension options (the columns of v_extend; cohort the parent), (e, cohort, d_extend, d) the periods the units
    # extended in d_extend are in service in, one row per option (the parent's kind, decision period, last period
    # in service and carried size); (e, cohort, d) the level price of the extension over its periods in service and
    # its fixed cost (CUR/MW, MWh; ``_extend.apply_extension_costs``).
    cd_extend: pl.DataFrame | None = None
    cdd_extend: pl.DataFrame | None = None
    cohort_extend: pl.DataFrame | None = None
    cd_extend_cost: Param | None = None
    cd_extend_fixed: Param | None = None
    # Reported costs only (no model term reads them; ``_run_costs`` "Run-level fields of the reported costs"):
    # discounted from the start of the run over the periods of the whole run.  (d,) factors over the solve's
    # periods and the other run periods; (e, d) the investment lump of the invest decisions.  Equal to the
    # per-solve fields when the run is the solve.
    p_inflation_op_run: Param | None = None
    p_inflation_inv_run: Param | None = None
    ed_entity_annual_discounted_run: Param | None = None
    # The cost lines of the reported costs (``_npv``, :mod:`.params._cost_lines`; no model term reads them):
    # ``[e, d, line, d_report, value]`` (Utf8) per MW of v_invest (the run lump split by event), of v_divest, of
    # the existing capacity in d (its credit; its replacement over the run), and ``[e, cohort, d, line, d_report,
    # value]`` per MW of v_divest_built / v_extend (the parent's cohort).  Each frame's lines add up to its term.
    edl_invest_lines_run: pl.DataFrame | None = None
    edl_divest_lines: pl.DataFrame | None = None
    edl_existing_credit_lines: pl.DataFrame | None = None
    edl_existing_replacement_lines: pl.DataFrame | None = None
    cdl_divest_built_lines: pl.DataFrame | None = None
    cdl_extend_lines: pl.DataFrame | None = None
    # (e, d) existing capacity given by the user, aged over its lifetime (no capacity carried from earlier solves,
    # no retirements): the reported fixed cost of existing capacity.
    p_entity_pre_existing: Param | None = None
    e_invest_total: pl.DataFrame | None = None       # (e,)
    e_divest_total: pl.DataFrame | None = None
    e_invest_max_total: Param | None = None          # (e,)
    e_divest_max_total: Param | None = None
    ed_invest_period_set: pl.DataFrame | None = None  # (e, d) — entities with per-period invest cap
    ed_divest_period_set: pl.DataFrame | None = None  # (e, d)
    ed_invest_max_period: Param | None = None        # (e, d)
    ed_divest_max_period: Param | None = None        # (e, d)

    # Multi-solve handoff state — populated when running a sub-solve of
    # a chain, tells the LP what investment/divestment was realized in
    # prior sub-solves so that cumulative caps stay tight.  See
    # flextool.mod:3597-3623 (maxInvest_entity_total / maxDivest_entity_total
    # / minInvest_entity_total / minDivest_entity_total).
    p_entity_previously_invested_capacity: Param | None = None  # (e, d)
    p_entity_invested: Param | None = None      # (e,)  — cumulative prior-solve invest, used by min/max divest variants when not solveFirst
    p_entity_divested: Param | None = None      # (e,)  — cumulative prior-solve divest, used by max/min divest variants when not solveFirst

    # ─── Ramp limits ──────────────────────────────────────────────────────
    process_source_sink_ramp_limit_sink_up:   pl.DataFrame | None = None
    process_source_sink_ramp_limit_sink_down: pl.DataFrame | None = None
    process_source_sink_ramp_limit_source_up: pl.DataFrame | None = None
    process_source_sink_ramp_limit_source_down: pl.DataFrame | None = None
    # The arcs whose flow change is priced (ramp_change_eq, v_ramp_up / v_ramp_down, the ramp_cost objective
    # term): p_ramp_cost_arc > 0.  Populated by ``apply_projection_params`` (params._projection.ramp_cost_arc_frame).
    process_source_sink_ramp_cost: pl.DataFrame | None = None
    # (p, source, sink): the ramp cost of each arc in CUR/MW of change, the sum of the ramp_cost of its sides whose
    # own ramp_method is ramp_cost or both (> 0 only).  Populated by ``apply_direct_params``.
    p_ramp_cost_arc: Param | None = None
    # method_2way_1var_off arcs (the .mod's signed-flow 2-way method,
    # ``v_flow ∈ [-cap, +cap]``).  The engine emits ``v_flow ≥ 0`` on the
    # forward arc only, so these arcs carry sink→source flow via a
    # non-negative auxiliary ``v_flow_back``.  Union of DC-power-flow and
    # non-DC arcs; ``model/core/`` declares the shared ``v_flow_back`` over it
    # and ``_dc_power_flow`` consumes the same Var on its DC subset.
    process_source_sink_2way_1var: pl.DataFrame | None = None  # (p, source, sink)
    p_ramp_speed_up_sink:   Param | None = None    # (p, sink)
    p_ramp_speed_down_sink: Param | None = None
    p_ramp_speed_up_source:   Param | None = None  # (p, source)
    p_ramp_speed_down_source: Param | None = None
    # (d, t, d_previous, t_previous), Utf8: the steps whose predecessor in the un-rolled solve is adjacent on the
    # timeline (jump == 1), without the first steps of representative blocks (see params._ramp_links).
    dt_ramp_link: pl.DataFrame | None = None
    # (d, t, d_previous, t_previous), Utf8: the online predecessor of the steps without a ramp predecessor (a stretch
    # start: the first step of a solve or period, the step after a gap, a representative-block start) — the last step
    # of its own stretch of the period (params._ramp_links.online_stretch_wrap).  Set with dt_ramp_link
    # (ramp_link_fields).  The model uses it only at the stretch starts of samples (after a gap, a representative-
    # block start): the first step of a continuous period starts free (_ramp_links.online_free_start_cells).
    dt_online_wrap: pl.DataFrame | None = None
    # (d, t, d_previous, t_previous, duration_previous): first steps of a continuation roll that ramp from the
    # previous roll's realized flow at (d_previous, t_previous); p_roll_continue_flow holds that flow.
    dt_ramp_carry: pl.DataFrame | None = None
    p_roll_continue_flow: Param | None = None  # (p, source, sink, d, t) Utf8 keys, units of unitsize
    # (p, source, sink, d, t): the length (h) of the carried step of each arc of p_roll_continue_flow — the
    # previous roll's last realized step, or for a coarse process its last realized coarse step.
    p_roll_continue_flow_duration: Param | None = None

    # ─── Online / min_load (unit commitment) ──────────────────────────────
    process_online: pl.DataFrame | None = None              # set: (p,)
    process_online_linear: pl.DataFrame | None = None        # set: (p,)
    process_online_integer: pl.DataFrame | None = None       # set: (p,)
    # The processes whose startup_method is linear_fix_in_uc or integer_fix_in_uc (params._projection
    # .process_online_fix_in_uc), online or not in this solve: a separate realization solve keeps their units online
    # at the stochastic solve's value (the fix_online handoff, build_handoff_from_solution).  Read there, by
    # solve._realization, by the units-online ties of a solve whose na_online_fix_in_uc is set and by
    # coarse_tie_straddles (which processes such a solve ties).  A reader-less CSV reload seeds it from the dump
    # (_csv_reload._field_seeds).
    process_online_fix_in_uc: pl.DataFrame | None = None     # set: (p,)
    # True in solve (1) of separate_dispatch (``solve_data/na_online_fix_in_uc``): across the forecasts only the
    # units online of process_online_fix_in_uc are tied (model._non_anticipativity.na_role "uc_only").
    na_online_fix_in_uc: bool = False
    # True when the solve's reduce_unit_detail_method relaxes integers (preprocess._reduce_detail): v_online_integer
    # and v_exact_forward are then declared continuous.  False (default): integer as authored.
    relax_integrality: bool = False
    # Connections with the ``exact`` transfer method (internal method_2way_2var_MIP_exclude, override-aware):
    # their direction switch v_exact_forward (model/_exact.py).
    process_exact: pl.DataFrame | None = None                 # set: (p,)
    process_minload: pl.DataFrame | None = None              # set: (p,)
    process_min_load_eff: pl.DataFrame | None = None  # (p,) where ct_method=min_load_efficiency (+ single-line piecewise_linear units with an intercept)
    # piecewise_linear units with two or more lines in some period (params/_piecewise.py, the load boundary):
    process_piecewise: pl.DataFrame | None = None                 # set: (p,) — conversion_piecewise rows
    process_piecewise_needs_online: pl.DataFrame | None = None    # set: (p,) — efficiency rises from the first point
    process_piecewise_remap: pl.DataFrame | None = None           # set: (p,) — one input, one output, not delayed
    p_piecewise_slope: Param | None = None       # (p, seg, d) — input per output of each line (online / capacity form)
    p_piecewise_intercept: Param | None = None   # (p, seg, d) — input per unit of capacity at zero output of each line
    p_piecewise_points: pl.DataFrame | None = None  # (p, d, load, eff) — the solve's curve points (load / eff non-dim)
    p_online_dt: pl.DataFrame | None = None                  # set: (p, d, t) — UC var domain
    pdt_online_linear: pl.DataFrame | None = None  # (p, d, t) — startup-cost obj index, linear
    pdt_online_integer: pl.DataFrame | None = None # (p, d, t) — startup-cost obj index, integer
    p_min_load: Param | None = None                          # (p, d, t) — per-unit min-load floor (minFlow_minload), time-varying
    p_process_sink_min_capacity_coef: Param | None = None    # (p, sink) — capacity_min_coeff: optional per-output floor (minFlow_output_floor), default 0
    p_startup_cost: Param | None = None                      # (p, d)
    p_section: Param | None = None                           # (p, d, t)
    # Start-up input (unit__inputNode.startup_input_method startup_and_input / startup_only, startup_energy):
    # each start-up draws startup_energy · MW started from node n (core._conversion._startup_draw).  A startup_only arc is
    # not an arc: the readers hide it, so it is in none of the arc sets.  All None without start-up roles.
    process_startup_node: pl.DataFrame | None = None         # (p, n): units of the solve, startup_energy ≠ 0
    p_startup_energy: Param | None = None                    # (p, n, d) MWh per MW started
    startup_to_balance: pl.DataFrame | None = None           # (p, n): n has an energy balance
    startup_from_commodity: pl.DataFrame | None = None       # (p, n, c): n is a node of commodity c
    startup_from_co2_priced: pl.DataFrame | None = None      # (p, n, c, g): n in a CO2-priced group, c with CO2
    startup_from_co2_capped: pl.DataFrame | None = None      # (p, n, c, g): co2_max_period groups
    startup_from_co2_capped_total: pl.DataFrame | None = None  # (p, n, c, g): co2_max_total groups
    pdt_uptime_set: pl.DataFrame | None = None               # (p, d, t) — minimum_uptime constraint domain
    pdt_downtime_set: pl.DataFrame | None = None             # (p, d, t) — minimum_downtime constraint domain
    uptime_lookback: pl.DataFrame | None = None              # (p, d, t, d_back, t_back) — startup lookback window
    downtime_lookback: pl.DataFrame | None = None            # (p, d, t, d_back, t_back) — shutdown lookback window
    # Rolling online-commitment carry (online analog of
    # ``p_roll_continue_state``).  Carried committed-jump-end ``v_online``
    # of the prior roll per online process; applied at the continuation
    # roll's first ``online_change`` step (``dt_online_carry``) as the
    # predecessor constant so a unit that stays committed is not re-charged
    # a startup across the roll boundary.  ``None`` on the first roll /
    # non-rolling / no-online solves (→ byte-identical baseline path).
    p_roll_continue_online: Param | None = None              # (p,)
    # (d, t), Utf8: the step(s) of THIS roll that take the ``p_roll_continue_online`` carry as their online
    # predecessor — the ramp's seam roots (``dt_ramp_carry``: the first steps whose complete-solve predecessor is
    # adjacent and is, or ends where, the carried step ends, plus the forecast branches starting with them).
    # ``None`` without a carry or where the seam is at a gap, a representative-period start or a new year.
    dt_online_carry: pl.DataFrame | None = None
    # OC1 #1 sub-slice (b) — separate_dispatch sibling online PIN.  The
    # parent's committed ``v_online`` over the jump (read from the
    # ``fix_online`` handoff), pinned onto the sibling's ``v_online`` by
    # ``online_fix_eq`` so the realized dispatch honours the plan's
    # commitment.  ``None`` on every solve that is not a realized-dispatch
    # sibling with a carried pin (→ byte-identical baseline path).
    p_fix_online: Param | None = None                        # (p, d, t)
    # Roll carry of the minimum up / down time.  ``online_startup_history`` is the realized ``v_startup`` /
    # ``v_shutdown`` of the solve this one continues from (the ``online_history`` handoff), each step with its
    # ``hours_to_end``; ``online_min_time`` is the authored max(min_uptime, min_downtime) per online process
    # (hours), used only to trim this solve's own history.  ``uptime_carry`` / ``downtime_carry`` are the carried
    # obligations per ``(p, d, t)`` at the steps that follow ``dt_ramp_carry`` (``_derived.apply_mintime_lookbacks``),
    # on the ``minimum_uptime`` RHS / ``minimum_downtime`` LHS.  All ``None`` on the first roll / non-rolling /
    # no-min-up-down solves.
    online_startup_history: pl.DataFrame | None = None       # (p, d, t, startup, shutdown, duration, hours_to_end)
    online_min_time: pl.DataFrame | None = None              # (p, value) — max authored min-up/down hours
    uptime_carry: pl.DataFrame | None = None                 # (p, d, t, value) — carried Σ v_startup
    downtime_carry: pl.DataFrame | None = None               # (p, d, t, value) — carried Σ v_shutdown

    # ─── Storage ─────────────────────────────────────────────────────────
    nodeState: pl.DataFrame | None = None
    nodeState_dt: pl.DataFrame | None = None
    nodeState_first_dt: pl.DataFrame | None = None
    storage_bind_within_timeblock: pl.DataFrame | None = None
    storage_bind_forward_only: pl.DataFrame | None = None    # set: (n,)
    storage_bind_within_solve: pl.DataFrame | None = None    # set: (n,)
    storage_bind_within_period: pl.DataFrame | None = None   # set: (n,)
    storage_bind_within_solve_blended_weights: pl.DataFrame | None = None  # set: (n,)
    # Phase C — recognised by the v55 value_list but the constraint
    # implementations land in Phases D / E.  Loaded here so the
    # ``nodeBalance_eq`` guard in :mod:`flextool.engine.model.core`
    # can raise a precise "not yet implemented" error instead of
    # silently emitting zero state-change residuals.
    storage_bind_within_period_blended_weights: pl.DataFrame | None = None  # set: (n,)
    storage_bind_forward_only_blended_weights: pl.DataFrame | None = None   # set: (n,)
    storage_fix_start: pl.DataFrame | None = None
    # Nodes whose end state is fixed (storage_start_end_method fix_end or fix_start_end), at the last step of the
    # entire model, in every sub-solve whose horizon holds it (storage_solve_at_model_end;
    # solve_data/storage_solve_at_model_end, default True).
    storage_fix_end: pl.DataFrame | None = None  # set: (n,)
    storage_solve_at_model_end: bool = True
    # Whether the base-period chain of a representative-period solve ends at the end of the model (the solve holds
    # the model's last period; solve_data/storage_rp_chain_at_model_end, default True).
    storage_rp_chain_at_model_end: bool = True
    # (n, kind, value): the horizon-end condition of the bind_within_solve nodes of a rolling roll that continues
    # from a carried state (solve_data/storage_roll_end, _run_model._anchor_cyclic_in_rolling): kind anchor /
    # close (MWh value), anchor_share / close_share (storage_state_start × capacity) and wrap_end (the first roll's
    # wrap pinned at storage_state_end).  model.core._storage_state._eq_storage_roll_end.
    storage_roll_end: pl.DataFrame | None = None
    dtttdt: pl.DataFrame | None = None           # (d, t, t_previous_*, ...)
    dtttdt_forward_only: pl.DataFrame | None = None  # dtttdt with first (d,t) per solve dropped
    # The bind_within_solve lag (columns of dtttdt) where it differs from dtttdt: with a stochastic fan in the
    # solve's first period, the first step of each root forecast continues from the last step of its own path
    # (_derived.dtttdt_within_solve_from_dtttdt); None otherwise (the lag is dtttdt).
    dtttdt_within_solve: pl.DataFrame | None = None
    # (d, t, t_previous_within_period): the previous step of t in its own period; the first step of each period
    # wraps to the period's own last step (the bind_within_period lag; blocks chain inside the period).
    dtttdt_within_period: pl.DataFrame | None = None
    # Rolling-horizon (nested-solve) framework — flextool.mod:2196 + 2760.
    # ``p_nested_solve_first``: tri-state.  None → no p_nested_model.csv,
    # treat as single-solve (== solveFirst).  True / False — read from
    # ``solve_data/p_nested_model.csv``'s ``solveFirst`` row.
    # When False, the nodeBalance ``fwd_fix_*`` block is *replaced* with a
    # ``roll_continue`` term that pins
    # ``v_state[n, d_first, t_first] * unitsize == p_roll_continue_state[n]``.
    p_nested_solve_first: bool | None = None
    p_roll_continue_state: Param | None = None        # (n,)
    n_fix_storage_quantity: pl.DataFrame | None = None  # (n,)
    ndt_fix_storage_quantity: pl.DataFrame | None = None  # (n, d_upper, t_upper)
    p_fix_storage_quantity: Param | None = None       # (n, d_upper, t_upper)
    # Phase B4-pre — fix_storage_usage loader (constraint added in B4).
    # Mirrors the fix_storage_quantity triplet above; populated from the
    # canonical handoff key ``handoff/fix_storage_usage`` (schema
    # ``[node, period, step, p_fix_storage_usage]``).
    n_fix_storage_usage: pl.DataFrame | None = None     # (n,)
    ndt_fix_storage_usage: pl.DataFrame | None = None   # (n, d, t)
    p_fix_storage_usage: Param | None = None            # (n, d, t)
    # The (node, upper period, upper step) an upper-level solve hands a nested fix_price to (handoff/fix_storage_price).
    ndt_fix_storage_price: pl.DataFrame | None = None   # (n, d, t), upper-level (d, t)
    dtt_timeline_matching: pl.DataFrame | None = None  # (d, t, t_upper) — lower→upper step map
    period_branch: pl.DataFrame | None = None         # (d_upper, d) — period→branch map
    period_last: pl.DataFrame | None = None           # (d,)
    nodeState_last_dt: pl.DataFrame | None = None     # (n, d, t) — block_period_time_last × node__block × nodeState
    # In-memory BlockLayout of the solve (node -> new_stepduration group, group steps).  Built by load_flextool
    # from the solve_data/ block frames; the coarse-step frames below are derived from it.
    block_layout: "BlockLayout | None" = None
    # ─── Intraperiod-block storage (bind_intraperiod_blocks) ─────────────
    nodeStateBlock: pl.DataFrame | None = None             # set: (n,)
    period_block: pl.DataFrame | None = None               # set: (d, b_first)
    period_block_succ: pl.DataFrame | None = None          # set: (d, b_first, b_next)
    period_block_time: pl.DataFrame | None = None          # set: (d, b_first, t)
    dtttdt_block_interior: pl.DataFrame | None = None      # dtttdt rows where t_previous_within_timeset == t_previous (interior-of-block jump=1)
    dtttdt_block_lag: pl.DataFrame | None = None           # (d, t, d_previous, t_previous_within_solve): block state lag
    # Rolling sub-solve (solve_data/roll_period_whole has rows) and the periods the roll holds only in part: their
    # bind_intraperiod_blocks blocks chain forward (dtttdt_block_lag), the blocks of the other periods loop.
    solve_is_rolling: bool = False
    period_block_chain: pl.DataFrame | None = None         # set: (d,)
    # ─── Coarse new_stepduration groups (params._coarse.coarse_node_frames) ─────────
    # Nodes of a group whose new_stepduration is longer than the solve's step: balanced once per coarse step
    # (nodeBalanceCoarse_eq); a coarse storage node keeps one state per coarse step (stateConstantCoarse_eq).
    nodeCoarse: pl.DataFrame | None = None                 # set: (n,)
    node_coarse_step: pl.DataFrame | None = None           # set: (n, d, b_first, t) — fine t in coarse step b_first
    node_coarse_interior: pl.DataFrame | None = None       # set: (n, d, t, t_previous) — steps of one coarse step
    # Units / connections whose every balance node is coarse (params._coarse.coarse_process_frames): one flow per
    # coarse step of their nodes.
    processCoarse: pl.DataFrame | None = None              # set: (p,)
    process_coarse_step: pl.DataFrame | None = None        # (p, d, b_first, t, w) — fine t in coarse step b_first, w = Δt/Δ_b
    # ─── RP-blended-weights storage (bind_within_solve_blended_weights) ─────────
    # Eight per-solve sets / params that drive the intra-period state-change
    # branch for ``nodeState_rp`` plus the three ``rp_inter_period_*``
    # constraints (.mod:2197-2200, .mod:2965-2997).  Populated by
    # ``_load_storage`` from the Phase-1 Provider keys (see
    # ``_provider_keys.SOLVE_DATA_NODE_STATE_RP`` + siblings).  Loader-only
    # at this commit — model/core/ wiring lands in Phase 5+.  When
    # ``nodeState_rp`` is non-empty the loader enforces that the four
    # tightly-coupled fields (``rp_base_period_set``, ``rp_base__rep``,
    # ``rp_block_first``, ``p_rp_last_step``) are also non-empty.
    nodeState_rp: pl.DataFrame | None = None               # set: (n,)
    # Every rp frame carries the owning period d: each period has its own base periods b (base-period start
    # steps), representative blocks r (block start steps) and weights.
    rp_base_period_set: pl.DataFrame | None = None         # set: (d, b)
    rp_base_chain: pl.DataFrame | None = None              # set: (d, b, b_prev) — within-period edges
    rp_base_first: pl.DataFrame | None = None              # set: (d, b)
    rp_base_last: pl.DataFrame | None = None               # set: (d, b)
    rp_block_first: pl.DataFrame | None = None             # set: (d, t)
    # Relation (d, r) → last_step (DataFrame, not Param: v_state[n, d, p_rp_last_step[d, r]] is a relational
    # join on (d, r) → last_step).
    p_rp_last_step: pl.DataFrame | None = None             # set: (d, r, last_step)
    rp_base__rep: Param | None = None                      # (d, b, r) → weight
    rp_period_previous: pl.DataFrame | None = None         # set: (d, d_prev) — across-solve rp period chain
    flow_from_nodeBalance_eff: pl.DataFrame | None = None
    flow_from_nodeBalance_noEff: pl.DataFrame | None = None
    p_state_upper: Param | None = None           # (n, d) — capacity / unitsize
    p_state_unitsize: Param | None = None        # (n,)
    p_state_self_discharge: Param | None = None  # (n, d, t): loss per hour
    p_state_start: Param | None = None           # (n,)
    p_state_end: Param | None = None             # (n,): storage_state_end
    p_state_existing_capacity: Param | None = None  # (n, d)
    # ─── Storage end-state binding (use_reference_value) ─────────────────
    # mod:2802-2822 — pins v_state at the last timestep of period_last to
    # ``reference_value × existing capacity / unitsize`` for nodes with
    # ``storage_solve_horizon_method=use_reference_value`` and no
    # competing fix_end / fix_start_end / bind_within_solve method.
    storage_use_reference_value: pl.DataFrame | None = None  # (n,)
    p_storage_state_reference_value: Param | None = None     # (n, d, t)
    # ─── Storage end-state binding (use_reference_price) ─────────────────
    # B1a — per-(node, period) reference price emitted by
    # ``_arc_unions.emit_p_storage_state_reference_price`` into
    # ``solve_data/p_storage_state_reference_price.csv``.  Consumed by
    # the use_reference_price objective term (B1b) for nodes whose
    # ``storage_solve_horizon_method=use_reference_price``.  Loaded but
    # unused at the B1a commit; B1b wires it into model/core/.
    p_storage_state_reference_price: Param | None = None     # (n, d)
    # ─── State-profile bounds (node__profile__profile_method) ────────────
    # (n, f) tuples for nodes with a profile-method state bound.  Mirrors
    # ``process_profile_*`` (process side) but for ``v_state``.
    node_profile_upper: pl.DataFrame | None = None  # (n, f)
    node_profile_lower: pl.DataFrame | None = None  # (n, f)
    node_profile_fixed: pl.DataFrame | None = None  # (n, f)
    p_node_availability: Param | None = None     # (n, d, t) — slice of pdtNode availability

    # ─── Process variable cost (other_operational_cost) ──────────────────
    pssdt_varCost_noEff: pl.DataFrame | None = None
    pssdt_varCost_eff_unit_source: pl.DataFrame | None = None
    pssdt_varCost_eff_unit_sink: pl.DataFrame | None = None
    pssdt_varCost_eff_connection: pl.DataFrame | None = None
    p_pssdt_varCost: Param | None = None     # (p, source, sink, d, t)
    p_pdt_varCost_source: Param | None = None  # (p, source, d, t) — eff source O&M
    p_pdt_varCost_sink: Param | None = None    # (p, sink, d, t) — eff sink O&M
    p_pdt_varCost_process: Param | None = None # (p, d, t) — connection O&M

    # ─── Existing-entity fixed cost (constant; reported in objective) ─────
    p_ed_fixed_cost: Param | None = None         # (e, d)
    p_entity_all_existing: Param | None = None   # (e, d)

    # ─── Slack penalty scaling ────────────────────────────────────────────
    p_node_capacity_for_scaling: Param | None = None  # (n, d)

    # ─── Group-level slack (capacity_margin / inertia / non_sync) ─────────
    groupCapacityMargin: pl.DataFrame | None = None      # (g,)
    groupInertia: pl.DataFrame | None = None             # (g,)
    groupNonSync: pl.DataFrame | None = None             # (g,)
    group_node: pl.DataFrame | None = None               # (g, n)
    process_unit: pl.DataFrame | None = None             # (p,)  set of unit-typed processes (mod's process_unit set)
    process_sink_inertia: pl.DataFrame | None = None     # (p, sink)
    process_source_inertia: pl.DataFrame | None = None   # (p, source)
    process_sink_nonSync: pl.DataFrame | None = None     # (p, sink)
    process_group_inside_nonSync: pl.DataFrame | None = None  # (p, g)
    p_inv_group_cap: Param | None = None                 # (g, d)
    p_group_capacity_for_scaling: Param | None = None    # (g, d)
    pdGroup_capacity_margin: Param | None = None         # (g, d) MW: absolute, and relative_of_peak_demand once resolved
    pdGroup_capacity_margin_relative: Param | None = None   # (g, d) share of the peak (relative_of_peak_demand groups)
    pdGroup_peak_demand: Param | None = None             # (g, d) MW: peak demand over the period's timeline
    pdGroup_penalty_capacity_margin: Param | None = None # (g, d)
    pdtGroup_inertia_limit: Param | None = None          # authored dims ⊆ (g, d, t)
    pdGroup_penalty_inertia: Param | None = None         # (g, d)
    pdtGroup_non_synchronous_limit: Param | None = None  # authored dims ⊆ (g, d, t)
    pdGroup_penalty_non_synchronous: Param | None = None # (g, d)
    p_process_sink_inertia_constant: Param | None = None    # (p, sink)
    p_process_source_inertia_constant: Param | None = None  # (p, source)
    # max(inflow, 0), min(inflow, 0) and inflow / Δt of the real inflow (before the invest-solve energy
    # margin, p_inflow_unmargined when present): the group rows (non-synchronous limit, capacity margin,
    # dynamic reserve group demand) read the real demand.
    p_positive_inflow: Param | None = None               # (n, d, t)
    p_negative_inflow: Param | None = None               # (n, d, t)
    p_inflow_unmargined: Param | None = None             # (n, d, t) — solve_data/pdtNodeInflow_unmargined, if any
    pdtNodeInflow_per_step: Param | None = None          # (n, d, t)

    # ─── Reserves (timeseries / dynamic / n-1, plus per-process upper) ────
    reserve_upDown_group: pl.DataFrame | None = None                  # (r, ud, g) — gate
    reserve_upDown_group_method_timeseries: pl.DataFrame | None = None  # (r, ud, g, method)
    reserve_upDown_group_method_dynamic: pl.DataFrame | None = None     # (r, ud, g, method)
    reserve_upDown_group_method_n_1: pl.DataFrame | None = None         # (r, ud, g, method)
    prundt: pl.DataFrame | None = None                                  # (p, r, ud, n, d, t) — v_reserve domain
    process_reserve_upDown_node_active: pl.DataFrame | None = None      # (p, r, ud, n)
    process_reserve_upDown_node_increase_reserve_ratio: pl.DataFrame | None = None  # (p, r, ud, n)
    process_reserve_upDown_node_large_failure_ratio: pl.DataFrame | None = None     # (p, r, ud, n)
    p_process_reserve_upDown_node_reliability: Param | None = None      # (p, r, ud, n)
    pdtReserve_upDown_group_reservation: Param | None = None            # (r, ud, g, d, t)
    p_reserve_upDown_group_penalty_reserve: Param | None = None         # (r, ud, g)
    p_reserve_upDown_group_reserve_duration: Param | None = None        # (r, ud, g)
    p_reserve_upDown_group_increase_reserve_ratio: Param | None = None  # (r, ud, g) — group demand ratio
    p_process_reserve_upDown_node_max_share: Param | None = None        # (p, r, ud, n)
    p_process_reserve_upDown_node_large_failure_ratio_value: Param | None = None     # (p, r, ud, n)
    p_process_reserve_upDown_node_increase_reserve_ratio_value: Param | None = None  # (p, r, ud, n)

    # ─── Cumulative / group-invest / min-invest (read by _cumulative_invest) ─
    # Sets
    ed_invest_cumulative: pl.DataFrame | None = None               # (e, d) — cumulative-cap rows
    group_entity: pl.DataFrame | None = None                       # (g, e)
    g_invest_total: pl.DataFrame | None = None                     # (g,)
    g_divest_total: pl.DataFrame | None = None                     # (g,)
    g_invest_cumulative: pl.DataFrame | None = None                # (g,)
    gd_invest_period: pl.DataFrame | None = None                   # (g, d)
    gd_divest_period: pl.DataFrame | None = None                   # (g, d)
    group_process_node: pl.DataFrame | None = None                 # (g, p, n)
    # Parameters
    ed_invest_min_period: Param | None = None             # (e, d)
    ed_divest_min_period: Param | None = None             # (e, d)
    e_invest_min_total: Param | None = None               # (e,)
    e_divest_min_total: Param | None = None               # (e,)
    ed_cumulative_max_capacity: Param | None = None       # (e, d)
    ed_cumulative_min_capacity: Param | None = None       # (e, d)
    p_group_invest_max_period: Param | None = None        # (g, d)
    p_group_invest_min_period: Param | None = None        # (g, d)
    p_group_retire_max_period: Param | None = None        # (g, d)
    p_group_retire_min_period: Param | None = None        # (g, d)
    p_group_invest_max_total: Param | None = None         # (g,)
    p_group_invest_min_total: Param | None = None         # (g,)
    p_group_retire_max_total: Param | None = None         # (g,)
    p_group_retire_min_total: Param | None = None         # (g,)
    pd_group_cumulative_max_capacity: Param | None = None  # (g, d)
    pd_group_cumulative_min_capacity: Param | None = None  # (g, d)
    p_group_max_cumulative_flow: Param | None = None      # (g,)
    p_group_min_cumulative_flow: Param | None = None      # (g,)
    pd_max_cumulative_flow: Param | None = None           # (g, d)
    pd_min_cumulative_flow: Param | None = None           # (g, d)
    pdt_max_instant_flow: Param | None = None             # (g, d, t)
    pdt_min_instant_flow: Param | None = None             # (g, d, t)

    # ─── Delayed processes (read by _delay) ───────────────────────────
    process_delayed: pl.DataFrame | None = None                  # (p,)
    process_delayed__duration: pl.DataFrame | None = None        # (p, td)
    process_source_delayed: pl.DataFrame | None = None           # (p, source)
    process_source_undelayed: pl.DataFrame | None = None         # (p, source)
    process_source_sink_delayed: pl.DataFrame | None = None      # (p, source, sink)
    process_source_sink_undelayed: pl.DataFrame | None = None    # (p, source, sink)
    dtt__delay_duration: pl.DataFrame | None = None              # (d, t_source, t_sink, td)
    p_process_delay_weight: Param | None = None                  # (p, td)

    # ─── DC power flow (read by _dc_power_flow) ──────────────────────────
    # Populated only when ``input/node_dc_power_flow.csv`` and
    # ``connection_dc_power_flow.csv`` carry rows.  See
    # :mod:`flextool._dc_power_flow` for the constraint emission.
    node_dc_power_flow: pl.DataFrame | None = None               # (n,)
    connection_dc_power_flow: pl.DataFrame | None = None         # (p,)
    node_reference_angle: pl.DataFrame | None = None             # (n,)
    p_connection_susceptance: Param | None = None                # (p,)
    # Forward-direction (p, source, sink) mirror of process_source_toSink,
    # used to dedupe DC PF arcs.  process_source_sink doubles up 2-way
    # connections; the .mod's dc_flow_eq indexes on process_source_toSink
    # (one direction per arc).
    process_source_toSink_dc: pl.DataFrame | None = None         # (p, source, sink)

    # ─── Commodity price ladder (read by _commodity_ladder) ─────────────
    # Populated only when at least one commodity has
    # ``price_method = price_ladder_*``.  See
    # :mod:`flextool._commodity_ladder` for the constraint emission.
    commodity_with_ladder: pl.DataFrame | None = None            # (c,)
    commodity_with_ladder_annual: pl.DataFrame | None = None     # (c,)
    commodity_with_ladder_cumulative: pl.DataFrame | None = None # (c,)
    cnd_ladder: pl.DataFrame | None = None                       # (c, n, d)
    cndi_ladder: pl.DataFrame | None = None                      # (c, n, d, i)
    cndi_ladder_ann: pl.DataFrame | None = None                  # (c, n, d, i)
    cndi_ladder_cum: pl.DataFrame | None = None                  # (c, n, d, i)
    ci_ladder_cumulative: pl.DataFrame | None = None             # (c, i)
    commodity__tier_ann: pl.DataFrame | None = None              # (c, i)
    commodity__tier_cum: pl.DataFrame | None = None              # (c, i)
    p_ladder_ann_price: Param | None = None                      # (c, i, d)
    p_ladder_ann_quantity: Param | None = None                   # (c, i, d)
    p_ladder_cum_price: Param | None = None                      # (c, i)
    p_ladder_cum_quantity: Param | None = None                   # (c, i)
    p_commodity_unitsize: Param | None = None                    # (c,)
    p_f_d_k: Param | None = None                                 # (d,)
    p_ladder_cum_realized_mwh: Param | None = None               # (c, i, d)
    p_ladder_cum_weight: Param | None = None                     # (d,)

    # ─── Stochastic / multi-branch operational data (A6) ─────────────────
    # All fields populated only when the active solve actually runs a
    # multi-branch stochastic dispatch (signalled by ``solve_data/
    # pdt_branch_weight.csv`` containing rows where the cohort
    # (anchor period d) has multiple sibling periods b).  When stochastics
    # is inactive every (d, t) carries weight 1.0 and these fields stay
    # ``None`` (the model layer falls back to the deterministic path).
    #
    # ``period_branch_full`` is the unfiltered ``period__branch.csv``
    # (anchor d → sibling b).  Distinct from the existing
    # ``period_branch`` field which is the rolling-handoff helper
    # (renamed columns).  Both share the same source CSV; we keep them
    # separate to avoid disturbing the rolling-handoff consumer.
    pdt_branch_weight: Param | None = None        # (d, t) — operational weight (defaults 1.0)
    pd_branch_weight: Param | None = None         # (d,) — period-level weight (defaults 1.0)
    period_branch_full: pl.DataFrame | None = None  # (d, b) — full anchor→sibling map
    dt_non_anticipativity: pl.DataFrame | None = None  # (d, t) — realised dispatch + fix-storage timesteps
    groupStochastic: pl.DataFrame | None = None   # (g,) — groups enabling storage non-anticipativity
    period_in_use_set: pl.DataFrame | None = None  # (d,) — periods active this solve (filters branches)
    # (d, t) — the steps the period labels of a stochastic fan had before the expansion.  A label has no dispatch
    # (no dt rows); this is only the period filter of period-level parameters (investment limits), so a label's
    # period-level investment keeps its values.  None without a fan.
    label_steps: pl.DataFrame | None = None
    # (d,) — investment periods of the active solve and their stochastic branch periods: the capacity-margin
    # domain.  Empty frame = the solve has no investment periods; None = not loaded.
    period_invest: pl.DataFrame | None = None
    # Scenario-lineage frames (recourse plan §6b Slice A).  Built by
    # apply_branch_cluster for every solve; None only when the derived
    # cascade never ran (hand-built FlexData).  dd_same_scenario:
    # (d, d_other) pairs sharing a scenario leaf-path (all-pairs over
    # period_in_use for deterministic solves).  pd_non_anticipativity:
    # (d, b) invest-NA period pairs — empty until mid-horizon branch
    # points exist (Slice E).
    dd_same_scenario: pl.DataFrame | None = None       # (d, d_other)
    pd_non_anticipativity: pl.DataFrame | None = None  # (d, b)
    # Slice D — per-scenario-leaf partition of period_in_use (design §7.2):
    # (d, leaf) with real-named calendar anchors → "__anchor", synthetic
    # fan members → their time-branch id.  Drives the per-path total caps
    # (maxInvest/maxDivest_entity_total_path, maxDivestGroup_entity_total_path)
    # so a total cap applies once per scenario path, never cross-scenario.
    # Single "__anchor" leaf for deterministic / flag-off solves →
    # caps stay on the legacy single-row shape (byte-parity).
    d_leaf: pl.DataFrame | None = None                 # (d, leaf)

    # Slice D — per-scenario (wait-and-see) stochastic investment gate.
    # Set once in ``apply_derived_c`` from
    # ``solve_data/stochastic_invest_method.csv`` == 'recourse'.  Survives
    # ``dataclasses.replace`` (region filter) like the lineage frames so
    # ``model/core/`` reads ``d.recourse_invest`` without a second CSV read.
    recourse_invest: bool = False

    # ─── Gap F final — handoff-path auxiliaries ───────────────────────────
    # Per-solve in-memory carriers for fields that ``build_handoff_from_solution``
    # would otherwise re-read from ``solve_data/`` to capture the post-solve
    # handoff.  Populated by :func:`load_flextool` from the corresponding
    # CSVs when present; ``None`` falls through to the disk-read fallback in
    # the handoff extractor (preserves test paths that construct FlexData by
    # hand).
    realized_dispatch: pl.DataFrame | None = None         # (period, step)
    # (period, step): the steps of this solve's own fix_storage_periods, at which it hands the nested storage fix
    # down (solve_data/fix_storage_timesteps); None when it has none.  The output set dt_fix_storage_timesteps.
    fix_storage_timesteps: pl.DataFrame | None = None
    # The solve's realized investment periods (``solve_data/realized_invest_periods_of_current_solve``, in
    # emitted order); ``None`` when the key is absent (FlexData built by hand), ``[]`` when it is empty.
    realized_invest_periods: list[str] | None = None
    node__storage_nested_fix_method: pl.DataFrame | None = None  # (node, method)

    def dump_csvs(self,
                   workdir: "Path | str",
                   *,
                   include_heavy: bool | None = None,
                   ) -> "Path":
        """Materialise this FlexData to flextool's CSV layout under ``workdir``.

        See :mod:`flextool.engine.output._dump_csvs` for the full mapping.
        The dump is for inspection.  Reloading it is unsupported and lives
        in ``flextool.engine._csv_reload`` (``copy_runner_meta`` +
        ``load_from_dump``).

        ``include_heavy`` (default ``None``) controls whether the seven
        gigabyte-scale CSVs (``p_flow_max.csv`` and friends) are
        written.  ``None`` honours the ``FLEXTOOL_DUMP_CSVS`` env var
        (off by default); pass ``True`` to force-write them.
        """
        # Local import — avoids a circular import at module-load.
        from flextool.engine.output._dump_csvs import dump_csvs as _impl
        return _impl(self, workdir, include_heavy=include_heavy)

    def __setattr__(self, name: str, value):
        # Diagnostic gate: only armed by ``_cascade_gate()`` (used by
        # the cascade parity test).  Production code paths never see
        # this branch — the module-level flag stays False end-to-end.
        if _CASCADE_GATE_ACTIVE:
            current = self.__dict__.get(name)
            if current is not None and value is not current:
                return
        object.__setattr__(self, name, value)
