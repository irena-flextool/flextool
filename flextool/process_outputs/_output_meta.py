"""Derive per-column output metadata (unit + semantics) from transforms.

The output stage turns LP variables and input parameters into ~95 processed
tables.  Hand-authoring metadata per column would be a slog and rot on the
next schema change.  Instead we *derive* it: every annual/extensive output is
produced by one of a small, named set of transforms (see :class:`Transform`),
and the displayed unit + temporal semantics fall out of the transform applied
to a base quantity.

This is the "derive from transforms" layer.  ``annualize_dt_to_d`` (the single
dt→d aggregation helper all annual outputs now route through, see
``_annualize.py``) is the runtime embodiment of the ``*_ANNUAL`` transforms;
this module is the static counterpart that labels the resulting columns.

A per-output declaration (:data:`OUTPUT_TRANSFORM`) maps an output key to the
transform its (non-dimension) columns underwent.  :func:`derive_column_meta`
expands that to one :class:`ColumnMeta` per column, generating a templated
tooltip and attaching any per-column ``formula`` override.

Coverage is enforced by ``tests/test_output_column_metadata.py`` via a ratchet:
new processed outputs may not land undocumented.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Semantics(str, Enum):
    """The kind of quantity a column holds — the field that disambiguates
    a per-timestep rate from a full-year-annualized total."""

    INSTANTANEOUS = "instantaneous"  # a rate at the timestep (MW). Integrate for energy.
    PER_STEP = "per_step"            # already integrated over the step's duration (cost_dt).
    PER_PERIOD = "per_period"        # summed over the period, as-sampled (NOT scaled to a year).
    ANNUALIZED = "annualized"        # scaled to a full-year equivalent (÷ period_share_of_year).
    HORIZON = "horizon"              # undiscounted total over the represented horizon (×years_represented).
    DISCOUNTED = "discounted"        # NPV over the horizon (inflation + years_represented).
    AVERAGE = "average"              # time-average over the period (÷ period_hours).
    RATIO = "ratio"                  # dimensionless (share, capacity factor).
    COUNT = "count"                  # event count (e.g. start-ups), annualized.
    LEVEL = "level"                  # a stock/snapshot (capacity, storage state) — not summed.
    PRICE = "price"                  # marginal value / shadow price (dual).
    DIMENSION = "dimension"          # index/key column, not a measure.


# Column names that are always index/key dimensions, never measures.
DIMENSION_COLUMNS: frozenset[str] = frozenset(
    {"solve", "period", "time", "scenario", "step", "year", "group", "node",
     "unit", "connection", "process", "source", "sink", "commodity", "reserve",
     "upDown", "category"}
)


@dataclass(frozen=True)
class Transform:
    """A named output transform → the (unit, semantics) it yields.

    ``unit`` is the *displayed* unit string (already including ``/a`` where the
    quantity is annualized), derived by the simple algebra noted in ``algebra``
    (e.g. MW · h / period_share → MWh/a).  Keeping the algebra in a comment
    rather than a full dimensional-analysis engine is deliberate: the transform
    set is tiny and stable, so an explicit table is clearer and just as safe.
    """

    unit: str
    semantics: Semantics
    tooltip: str               # short, one-line (plot/CSV/hover).
    description: str = ""       # optional longer form (docs/datapackage); falls back to tooltip.
    algebra: str = ""


# The transform catalog.  Each entry is the result of applying one output-stage
# transform to its base quantity.  Names mirror the code paths in out_*/calc_*.
RATE = Transform(
    "MW", Semantics.INSTANTANEOUS,
    "Power at the timestep (a rate). Multiply by step_duration for energy.",
    algebra="v_flow",
)
ENERGY_ANNUAL = Transform(
    "MWh/a", Semantics.ANNUALIZED,
    "Energy, scaled from the sampled timeline to a full-year equivalent.",
    algebra="MW · step_duration · timestep_weight / period_share",
)
ENERGY_PERIOD = Transform(
    "MWh", Semantics.PER_PERIOD,
    "Energy summed over the sampled timeline (NOT scaled to a full year).",
    algebra="MW · step_duration · timestep_weight  (no /period_share)",
)
PREENERGY_ANNUAL = Transform(
    "MWh/a", Semantics.ANNUALIZED,
    "Energy (already per-step MWh), scaled to a full-year equivalent.",
    algebra="MWh_step · timestep_weight / period_share",
)
MONEY_PERSTEP = Transform(
    "CUR", Semantics.PER_STEP,
    "Cost incurred during the timestep (step_duration and weight already applied).",
    algebra="price · flow · step_duration · timestep_weight",
)
MONEY_ANNUAL = Transform(
    "M CUR/a", Semantics.ANNUALIZED,
    "Cost, scaled from the sampled timeline to a full-year equivalent.",
    algebra="Σ cost_dt / period_share / 1e6",
)
MONEY_DISCOUNTED = Transform(
    "M CUR", Semantics.DISCOUNTED,
    "Net present cost over the horizon (inflation × years_represented applied).",
    algebra="annual_cost · inflation_factor · years_represented",
)
COST_ENTITY_ANNUAL = Transform(
    "M CUR/a", Semantics.ANNUALIZED,
    "Per-entity annualized cost for this category (full-year equivalent).",
    description="Annualized cost of one (entity, period) cell for this "
                "category — the system annualized cost collapsed per entity "
                "rather than summed system-wide. Fuel/commodity cost is "
                "attributed to the consuming process (the unit/connection that "
                "draws the commodity, not the source node); CO2 cost is summed "
                "over every priced group the process touches. Summing a "
                "category over all entities reproduces the system summary "
                "(annualized_costs_d_p).",
    algebra="Σ entity_cost_dt / period_share / 1e6",
)
COST_ENTITY_DISCOUNTED = Transform(
    "M CUR", Semantics.DISCOUNTED,
    "Per-entity net present cost for this category over the horizon.",
    description="Net present (discounted) cost of one (entity, period) cell "
                "for this category — the system discounted cost collapsed per "
                "entity rather than summed system-wide. Fuel/commodity cost is "
                "attributed to the consuming process (the unit/connection that "
                "draws the commodity, not the source node); CO2 cost is summed "
                "over every priced group the process touches. Summing a "
                "category over all entities reproduces the system summary "
                "(costs_discounted_d_p).",
    algebra="annual_entity_cost · inflation_factor · years_represented",
)
EMISSION_ANNUAL = Transform(
    "t/a", Semantics.ANNUALIZED,
    "Emissions, scaled from the sampled timeline to a full-year equivalent.",
    algebra="t_step · timestep_weight / period_share",
)
COUNT_ANNUAL = Transform(
    "units/a", Semantics.COUNT,
    "Event count (e.g. start-ups), scaled to a full-year equivalent.",
    algebra="count_step · timestep_weight / period_share",
)
AVERAGE_MW = Transform(
    "MW", Semantics.AVERAGE,
    "Time-average power held over the period.",
    algebra="Σ(MW · step_duration) / period_hours",
)
RATIO = Transform(
    "", Semantics.RATIO,
    "Dimensionless ratio (share / capacity factor).",
)
ENERGY_PERSTEP = Transform(
    "MWh", Semantics.PER_STEP,
    "Energy within the timestep (already per-step MWh, not a rate).",
)
RAMP = Transform(
    "MW", Semantics.INSTANTANEOUS,
    "Change in flow between consecutive timesteps (a ramp).",
    algebra="flow[t] − flow[t-1]",
)
CAPACITY_FACTOR = Transform(
    "", Semantics.RATIO,
    "Capacity factor (a fraction): the period's flow divided by the capacity and the period's hours.",
    algebra="Σ(MW · step_duration) / (capacity · period_hours)",
)
CAPACITY_MW = Transform(
    "MW", Semantics.LEVEL,
    "Installed/usable capacity (a stock that holds over the period, not a flow).",
    description="Capacity by period: existing, invested (capacity that comes online in the period), "
                "in_construction (capacity decided but not yet operating; empty without a construction lead "
                "time), divested and total (capacity in service). Capacity in construction is not in the total.",
    algebra="total = existing + invested − divested (cumulative; in_construction excluded)",
)
CAPACITY_MWH = Transform(
    "MWh", Semantics.LEVEL,
    "Storage energy capacity (a stock).",
    description="Storage capacity by period: existing, invested (capacity that comes online in the period), "
                "in_construction (capacity decided but not yet operating; empty without a construction lead "
                "time), divested and total (capacity in service). Capacity in construction is not in the total.",
    algebra="total = existing + invested − divested (cumulative; in_construction excluded)",
)
STORAGE_STATE = Transform(
    "MWh", Semantics.LEVEL,
    "Stored energy at the timestep (state of charge).",
    algebra="v_state · unitsize",
)
PRICE_ENERGY = Transform(
    "CUR/MWh", Semantics.PRICE,
    "Nodal energy price (shadow price of the node balance constraint).",
    description="Nodal energy price — the shadow price (dual) of the node "
                "balance constraint: the marginal system cost of one more unit "
                "of the commodity at that node and timestep, in CUR/MWh in the "
                "price base of the cost inputs (inflation and the years the "
                "period represents removed; not weighted by timestep weight or "
                "branch probability; empty where the objective does not weigh "
                "the step). For a storage "
                "node this is the marginal value of stored energy (the "
                "\"water value\"): a well-defined intertemporal price, but "
                "treat it with care — it can be non-unique when storage sits "
                "at the margin (indifferent between charging and discharging), "
                "spike at empty/full state of charge, differ between the "
                "charge and discharge sides by the round-trip efficiency, and "
                "be shifted by end-of-horizon, rolling and inter-period "
                "(blended-weight) storage linking.",
    algebra="σ·dual(node_balance) / s / (timestep_weight · inflation / period_share · branch_weight)",
)
EMISSION_MT = Transform(
    "Mt", Semantics.HORIZON,
    "System CO2 emissions, total over the represented horizon (NOT per year).",
    description="System CO2 emissions summed across all periods, each scaled "
                "by its years_represented and converted tonnes→megatonnes — an "
                "undiscounted horizon total, not an annual rate.",
    algebra="Σ_periods(t/a · years_represented) / 1e6",
)
FACTOR = Transform(
    "", Semantics.RATIO,
    "Dimensionless discount / inflation factor.",
)
INVEST_COST_DISCOUNTED = Transform(
    "CUR/MW", Semantics.DISCOUNTED,
    "Discounted investment cost per MW of capacity (per MWh for storage nodes): the yearly annuity summed "
    "with the inflation factors over the years of the lifetime inside the horizon.",
    algebra="annuity · Σ inflation_factor over the lifetime window",
)
YEARS = Transform(
    "a", Semantics.COUNT,
    "Calendar years the period represents (annualisation weight).",
)
PRICE_RESERVE = Transform(
    "CUR/MW", Semantics.PRICE,
    "Reserve price (dual of the reserve balance) — CUR per MW of reserve held "
    "for one hour (CUR/MWh).",
    algebra="dual(reserve_balance) / s / (step_duration · timestep_weight · inflation / period_share "
            "· branch_weight)",
)
PRICE_CO2 = Transform(
    "CUR/t", Semantics.PRICE,
    "CO2 price (shadow price of the CO2 cap).",
    algebra="−dual(co2_max_period) / s / inflation",
)
# Investment duals carry the same convention as invest_cost: CUR/kW for
# units & connections (CUR/kWh for storage nodes).
DUAL_INVEST_KW = Transform(
    "CUR/kW", Semantics.PRICE,
    "Effective marginal system value of +1 unit of capacity "
    "(same convention as invest_cost).",
    algebra="dual(invest constraint) / entity_unitsize",
)
DUAL_INVEST_KWH = Transform(
    "CUR/kWh", Semantics.PRICE,
    "Effective marginal system value of +1 unit of storage capacity "
    "(same convention as invest_cost).",
)
INERTIA = Transform(
    "MW·s", Semantics.LEVEL,
    "Inertia provided (rotational energy) at the timestep.",
    algebra="online · unitsize · inertia_constant",
)
INERTIA_ANNUAL = Transform(
    "MW·s/a", Semantics.ANNUALIZED,
    "Inertia-constraint slack, scaled to a full-year equivalent.",
)
ONLINE_COUNT = Transform(
    "units", Semantics.LEVEL,
    "Number of online units at the timestep.",
)
ONLINE_AVG = Transform(
    "units", Semantics.AVERAGE,
    "Average number of online units over the period.",
)
RESERVE_AVG = Transform(
    "MW", Semantics.AVERAGE,
    "Time-step-weighted average MW held over the period's realized steps.",
    algebra="Σₜ w·Δt·MW / Σₜ w·Δt over the realized steps where reserves were modelled (w = timestep_weight)",
)
RESERVE_REQUIREMENT_MW = Transform(
    "MW", Semantics.LEVEL,
    "Reserve requirement term of the group at the time step (MW held, not an energy).",
    algebra="the group's reserveBalance rows evaluated at the solution: need, counted reserve, shortfall",
)
RESERVE_REQUIREMENT_AVG = Transform(
    "MW", Semantics.AVERAGE,
    "Time-step-weighted average over the period's realized steps where reserves were modelled.",
    algebra="Σₜ w·Δt·X / Σₜ w·Δt (w = timestep_weight)",
)
RESERVE_REQUIREMENT_MAX = Transform(
    "MW", Semantics.LEVEL,
    "Largest value of the period's time steps (not weighted).",
    algebra="maxₜ X",
)
RESERVE_PROVIDER_MW = Transform(
    "MW", Semantics.LEVEL,
    "Counted reserve (× reliability) of the provider for the group at the time step.",
    algebra="v_reserve · unitsize · reliability",
)
ANGLE = Transform(
    "rad", Semantics.LEVEL,
    "DC voltage angle (Bθ formulation; reference node pinned to 0). Radians — "
    "flow = susceptance · Δangle with susceptance = base_MVA / reactance.",
)
SLACK_CAP_MARGIN = Transform(
    "MW", Semantics.LEVEL,
    "Capacity-margin slack — per-period MW shortfall (not annualized).",
)
CAPACITY_MARGIN_MW = Transform(
    "MW", Semantics.LEVEL,
    "Capacity-margin term of the group at its binding step (MW, not annualized).",
    algebra="the capacityMargin row's terms at the step with the largest deficit (requirement − counted)",
)
CAPACITY_MARGIN_PENALTY = Transform(
    "M CUR", Semantics.DISCOUNTED,
    "Penalty of the capacity-margin shortfall: shortfall · penalty_capacity_margin · 1000 · inflation / 1e6, "
    "priced once per period.",
    algebra="shortfall · penalty_capacity_margin · 1000 · inflation_op / 1e6",
)
CAPACITY_MARGIN_PRICE = Transform(
    "CUR/kW/a", Semantics.PRICE,
    "Shadow price of the capacity margin (equals penalty_capacity_margin while there is a shortfall).",
    algebra="Σₜ dual(capacityMargin) · inv_group_cap / s / (1000 · inflation_op · branch_weight)",
)
# Sentinel: an output that is a pure membership/index set (no measures).
DIMENSION_TABLE = Transform(
    "", Semantics.DIMENSION,
    "Membership / index set — no measure columns.",
)


@dataclass(frozen=True)
class ColumnMeta:
    """Derived metadata for one output column."""

    name: str
    unit: str
    semantics: Semantics
    tooltip: str               # short, one-line.
    long: str = ""             # optional longer description (falls back to tooltip).
    formula: str = ""
    docs: str = ""

    def as_dict(self) -> dict[str, str]:
        return {
            "unit": self.unit,
            "semantics": self.semantics.value,
            "tooltip": self.tooltip,
            "long": self.long,
            "formula": self.formula,
            "docs": self.docs,
        }


# ── Per-column transform maps (for mixed-unit tables) ───────────────────────
# Some outputs hold columns of different units in one frame.  Their
# OUTPUT_TRANSFORM value is a {column_label: Transform} map instead of a single
# Transform; derive_column_meta applies the per-column entry (columns absent
# from the map fall back to a '*' default if present, else dimension).
_NODEGROUP_GDT_P = {
    '1. Loss of load': ENERGY_PERSTEP,
    '2. VRE generation': ENERGY_PERSTEP,
    '3. Excess load': ENERGY_PERSTEP,
    '4. Curtailed VRE': ENERGY_PERSTEP,
    # The inflow as given: already MWh per time step (inflow is an energy per
    # step), like the slack/VRE columns above, which are integrated to MWh/step.
    '5. Timestep inflow': ENERGY_PERSTEP,
    '6. Curtailed VRE of potential VRE': RATIO,
    '7. Annualized inflow': PREENERGY_ANNUAL,
    '8. VRE share of demand': RATIO,
}
_NODEGROUP_GD_P = {
    '1. Loss of load share': RATIO,
    '2. VRE share of demand': RATIO,
    '3. Excess load share': RATIO,
    '4. Curtailed VRE of demand': RATIO,
    '5. Annualized inflow': PREENERGY_ANNUAL,
    '6. Curtailed VRE of potential VRE': RATIO,
}
_RESERVE_REQUIREMENT_D = {
    'requirement_avg': RESERVE_REQUIREMENT_AVG,
    'provided_avg': RESERVE_REQUIREMENT_AVG,
    'counted_avg': RESERVE_REQUIREMENT_AVG,
    'shortfall_avg': RESERVE_REQUIREMENT_AVG,
    'requirement_max': RESERVE_REQUIREMENT_MAX,
    'shortfall_max': RESERVE_REQUIREMENT_MAX,
}
_CAPACITY_MARGIN_D_G = {
    '*': CAPACITY_MARGIN_MW,
    'penalty_cost': CAPACITY_MARGIN_PENALTY,
    'capacity_price': CAPACITY_MARGIN_PRICE,
}
_FLOWGROUP_GD_P = {
    'cumulative_flow': ENERGY_PERIOD,   # MWh summed over the sample (not annualized)
    'average_flow': AVERAGE_MW,         # MW time-average
}


# ── Per-output declaration ──────────────────────────────────────────────────
# Map each output key to the transform its measure columns underwent.  One line
# per output; the per-column expansion + tooltips are generated.  Seeded with
# the high-traffic outputs; extend as coverage grows (the CI ratchet enforces
# that no NEW output lands here undocumented).
OUTPUT_TRANSFORM: "dict[str, Transform | dict[str, Transform]]" = {
    # Costs
    "costs_dt_p": MONEY_PERSTEP,
    "annualized_costs_d_p": MONEY_ANNUAL,
    "costs_discounted_d_p": MONEY_DISCOUNTED,
    # Per-entity cost break-down (period level).  Category column VALUES are
    # the measures; the entity/period index levels auto-resolve to dimension.
    "cost_unit_annualized_d_ec": COST_ENTITY_ANNUAL,
    "cost_connection_annualized_d_ec": COST_ENTITY_ANNUAL,
    "cost_node_annualized_d_ec": COST_ENTITY_ANNUAL,
    "cost_unit_discounted_d_ec": COST_ENTITY_DISCOUNTED,
    "cost_connection_discounted_d_ec": COST_ENTITY_DISCOUNTED,
    "cost_node_discounted_d_ec": COST_ENTITY_DISCOUNTED,
    # Unit / connection flows
    "unit_outputNode_dt_ee": RATE,
    "unit_inputNode_dt_ee": RATE,
    "unit_outputNode_d_ee": ENERGY_ANNUAL,
    "unit_inputNode_d_ee": ENERGY_ANNUAL,
    "unit_outputs_cf_d_ee": CAPACITY_FACTOR,
    "unit_inputs_cf_d_ee": CAPACITY_FACTOR,
    "connection_cf_d_e": CAPACITY_FACTOR,
    "connection_dt_eee": RATE,
    "connection_d_eee": ENERGY_ANNUAL,
    "connection_leftward_dt_eee": RATE,
    "connection_rightward_dt_eee": RATE,
    "connection_losses_dt_eee": RATE,
    "connection_leftward_d_eee": ENERGY_ANNUAL,
    "connection_rightward_d_eee": ENERGY_ANNUAL,
    "connection_losses_d_eee": ENERGY_ANNUAL,
    # CO2
    "CO2_d_g": EMISSION_ANNUAL,
    "process_co2_d_eee": EMISSION_ANNUAL,
    "CO2__": EMISSION_MT,
    # Node / group flows
    "nodeGroup_flows_d_g": PREENERGY_ANNUAL,
    "nodeGroup_flows_d_gpe": PREENERGY_ANNUAL,
    # Inflow is an energy per time step (MWh per step, not a rate): the group
    # inflow sum and the node inflow are written as given.  sum_flow_t (gpe)
    # integrates its flow columns to MWh/step as well.
    "nodeGroup_flows_dt_g": ENERGY_PERSTEP,
    "nodeGroup_flows_dt_gpe": ENERGY_PERSTEP,
    "node_inflow__dt": ENERGY_PERSTEP,
    "node_state_dt_e": STORAGE_STATE,
    # VRE shares + curtailment/potential.  NOTE: the *_d_ee curtailment and
    # potential are SUM-ONLY over the sample (per_period), NOT annualized —
    # unlike unit_outputNode_d_ee (see Task #4 / Stage-1 finding).
    "nodeGroup_VRE_share_d_g": RATIO,
    "nodeGroup_VRE_share_dt_g": RATIO,
    "unit_VRE_potential_outputNode_dt_ee": RATE,
    "unit_curtailment_outputNode_dt_ee": RATE,
    "unit_VRE_potential_outputNode_d_ee": ENERGY_PERIOD,
    "unit_curtailment_outputNode_d_ee": ENERGY_PERIOD,
    "unit_curtailment_share_outputNode_dt_ee": RATIO,
    "unit_curtailment_share_outputNode_d_ee": RATIO,
    # Ramps
    "unit_ramp_inputs_dt_ee": RAMP,
    "unit_ramp_outputs_dt_ee": RAMP,
    # Node slacks (dt = MW at the timestep; d = annualized MWh/a — out_node:120-126)
    "node_slack_up_dt_e": RATE,
    "node_slack_down_dt_e": RATE,
    "node_slack_up_d_e": ENERGY_ANNUAL,
    "node_slack_down_d_e": ENERGY_ANNUAL,
    "nodeGroup_slack_nonsync_dt_g": RATE,
    "nodeGroup_slack_nonsync_d_g": ENERGY_ANNUAL,
    # Capacity (a stock, not a flow)
    "unit_capacity_ed_p": CAPACITY_MW,
    "connection_capacity_ed_p": CAPACITY_MW,
    "node_capacity_ed_p": CAPACITY_MWH,
    # Costs / factors
    "costs_discounted_p_": MONEY_DISCOUNTED,
    "discountFactors_d_p": FACTOR,
    "entity_annuity_d_p": INVEST_COST_DISCOUNTED,
    "years_represented__d": YEARS,
    # Start-ups
    "unit_startup_d_e": COUNT_ANNUAL,
    "connection_startup_d_e": COUNT_ANNUAL,
    # Prices / duals
    "node_prices_dt_e": PRICE_ENERGY,
    "reserve_prices_dt_ppg": PRICE_RESERVE,
    "co2_price_period_d_g": PRICE_CO2,
    "co2_price_total_d_g": PRICE_CO2,
    "dual_invest_effective_unit_d_e": DUAL_INVEST_KW,
    "dual_invest_effective_connection_d_e": DUAL_INVEST_KW,
    "dual_invest_effective_node_d_e": DUAL_INVEST_KWH,
    # Inertia (MW·s) + reserve provision + online counts
    "nodeGroup_inertia_dt_g": INERTIA,
    "nodeGroup_unit_node_inertia_dt_gee": INERTIA,
    "nodeGroup_inertia_largest_flow_dt_g": RATE,
    "nodeGroup_slack_inertia_dt_g": INERTIA,
    "nodeGroup_slack_inertia_d_g": INERTIA_ANNUAL,
    "process_reserve_upDown_node_dt_eppe": RATE,
    "process_reserve_average_d_eppe": RESERVE_AVG,
    "dc_angle_dt_e": ANGLE,
    "dc_angle_diff_dt_e": ANGLE,
    "unit_online_dt_e": ONLINE_COUNT,
    "unit_online_average_d_e": ONLINE_AVG,
    "connection_online_dt_e": ONLINE_COUNT,
    "connection_online_average_d_e": ONLINE_AVG,
    "flowGroup_gd_t": RATE,
    "nodeGroup_slack_reserve_dt_eeg": RATE,
    "nodeGroup_slack_reserve_d_eeg": ENERGY_ANNUAL,
    "nodeGroup_reserve_dt_eegp": RESERVE_REQUIREMENT_MW,
    "nodeGroup_reserve_d_eegp": _RESERVE_REQUIREMENT_D,
    "nodeGroup_reserve_provider_dt_eegee": RESERVE_PROVIDER_MW,
    "nodeGroup_slack_capacity_margin_d_g": SLACK_CAP_MARGIN,
    "nodeGroup_capacity_margin_d_g": _CAPACITY_MARGIN_D_G,
    "nodeGroup_capacity_margin_entity_d_g": CAPACITY_MARGIN_MW,
    # Node balance tables.  _dt_ep is the un-integrated MW level (assembled
    # from raw flow_dt / inflow / slacks with no step_duration, like the dt
    # group flows); _d_ep integrates + annualizes to MWh/a.
    "node_dt_ep": RATE,
    "node_d_ep": ENERGY_ANNUAL,
    # Mixed-unit tables — per-column transform maps.
    "nodeGroup_gdt_p": _NODEGROUP_GDT_P,
    "nodeGroup_gd_p": _NODEGROUP_GD_P,
    "flowGroup_gd_p": _FLOWGROUP_GD_P,
    # Pure membership / index sets (no measure columns)
    "group_node": DIMENSION_TABLE,
    "group_process": DIMENSION_TABLE,
    "group_process_node": DIMENSION_TABLE,
    "timeline_breaks": DIMENSION_TABLE,
    "nodeGroupDispatch": DIMENSION_TABLE,
    "nodeGroupIndicators": DIMENSION_TABLE,
    "flowGroupIndicators": DIMENSION_TABLE,
    "connection_dc_power_flow": DIMENSION_TABLE,
    "node_dc_power_flow": DIMENSION_TABLE,
}

# Per-(output, column) formula overrides — only where a bespoke derivation
# string helps more than the transform's generic tooltip.
FORMULA_OVERRIDE: dict[tuple[str, str], str] = {
    ("annualized_costs_d_p", "commodity_cost"):
        "Σₜ(costs__dt.commodity_cost) / complete_period_share_of_year / 1e6",
}

DOCS_ANCHOR: dict[str, str] = {
    "annualized_costs_d_p": "results.html#costs",
    "costs_dt_p": "results.html#costs",
    "cost_unit_annualized_d_ec": "results.html#cost-by-entity",
    "cost_connection_annualized_d_ec": "results.html#cost-by-entity",
    "cost_node_annualized_d_ec": "results.html#cost-by-entity",
    "cost_unit_discounted_d_ec": "results.html#cost-by-entity",
    "cost_connection_discounted_d_ec": "results.html#cost-by-entity",
    "cost_node_discounted_d_ec": "results.html#cost-by-entity",
}


#: Row levels that place a value in time.  ``solve`` counts as time, ``period`` as its own kind.
TIME_LIKE_LEVELS: tuple[str, ...] = ("period", "time", "solve")
#: Levels whose values are distinct things of one quantity: adding across them gives a total.
ENTITY_LEVELS: frozenset[str] = frozenset(
    {"unit", "node", "connection", "process", "entity", "sink", "source", "commodity"}
)
#: Levels whose values can share members (a node or a process is in several groups): adding across them can
#: count a member more than once.
OVERLAPPING_LEVELS: frozenset[str] = frozenset({"group", "node_group"})
#: What combining the values of a non-time level means (:meth:`OutputDims.level_kind`):
#:
#: - ``entity``: distinct things of one quantity; every statistic is allowed, the sum is a total;
#: - ``part``: additive parts of one whole (cost categories); the sum is the total, a mean across them is not
#:   a figure anyone means;
#: - ``overlapping``: groups that can share members; a sum may count a member twice, the other statistics hold;
#: - ``quantity``: different quantities (existing / invested / total capacity, up / down reserve); they are
#:   not combined at all.
LEVEL_KINDS: tuple[str, ...] = ("entity", "part", "overlapping", "quantity")


def default_level_kind(level: str) -> str:
    """The kind of *level* in a table that does not declare it: ``time`` / ``period`` for the time levels,
    ``entity`` / ``overlapping`` by name, else ``quantity`` (an undeclared level is never combined)."""
    if level == "period":
        return "period"
    if level in TIME_LIKE_LEVELS:
        return "time"
    if level in ENTITY_LEVELS:
        return "entity"
    if level in OVERLAPPING_LEVELS:
        return "overlapping"
    return "quantity"


@dataclass(frozen=True)
class OutputDims:
    """The layout of one result table: its row index levels and its column levels (below the top
    ``scenario`` level of the parquet), declared per table, not parsed from the table name.

    A membership set (``DIMENSION_TABLE``) has no measure columns: its ``rows`` are the set's columns.
    ``counterpart`` is the same quantity at the other time resolution (a per-step table and its
    per-period table), or ``None``.  ``kinds`` declares, as ``(level, kind)`` pairs, the kind
    (:data:`LEVEL_KINDS`) of each level that is neither a time level nor in :data:`ENTITY_LEVELS` /
    :data:`OVERLAPPING_LEVELS`.
    """

    rows: tuple[str, ...]
    columns: tuple[str, ...] = ()
    counterpart: str | None = None
    kinds: tuple[tuple[str, str], ...] = ()

    def level_kind(self, level: str) -> str:
        """``time``, ``period``, or one of :data:`LEVEL_KINDS`: what combining over *level* means."""
        if level not in TIME_LIKE_LEVELS:
            for name, kind in self.kinds:
                if name == level:
                    return kind
        return default_level_kind(level)

    @property
    def has_time(self) -> bool:
        return "time" in self.rows

    @property
    def has_period(self) -> bool:
        return "period" in self.rows


_DT = ("period", "time")
_D = ("period",)
_CATEGORY_PART = (("category", "part"),)
_PARAMETER_QUANTITY = (("parameter", "quantity"),)
_PARAM_QUANTITY = (("param", "quantity"),)
_RESERVE_QUANTITY = (("reserve", "quantity"), ("updown", "quantity"))
_NODE_CATEGORY = (("category", "quantity"),)              # signed balance terms: their sum is the net (about 0)
_GROUP_FLOW_KINDS = (("type", "quantity"), ("item", "entity"))

# One sentence per result table: what it holds.  Table names are the parquet stems
# (``output_parquet/<scenario>/<stem>.parquet``); every key of OUTPUT_TRANSFORM and of the
# results-database spec (``write_spinedb.SPINEDB_OUTPUT_SPEC``) has one.
OUTPUT_DESCRIPTION: dict[str, str] = {
    # Costs
    "costs_dt_p": "Operational and penalty costs incurred in each time step by cost category, in currency (not "
                  "millions), with the step duration and timestep weight applied and not scaled to a year.",
    "annualized_costs_d_p": "Total cost of each period by cost category, scaled to a full year of operation, "
                            "including the yearly investment, retirement and fixed costs.",
    "costs_discounted_d_p": "Cost of each period by cost category, discounted and multiplied by the years the "
                            "period represents (the net present value the objective counts).",
    "costs_discounted_p_": "Discounted cost by cost category, totalled over all periods of the run.",
    "cost_unit_annualized_d_ec": "Annualized cost of each unit and period by cost category; summed over units, "
                                 "connections and nodes it gives the system annualized costs.",
    "cost_connection_annualized_d_ec": "Annualized cost of each connection and period by cost category.",
    "cost_node_annualized_d_ec": "Annualized cost of each node and period by cost category (storage investment "
                                 "and retirement, fixed costs, balance slack penalties).",
    "cost_unit_discounted_d_ec": "Discounted cost of each unit and period by cost category; summed over units, "
                                 "connections and nodes it gives the system discounted costs.",
    "cost_connection_discounted_d_ec": "Discounted cost of each connection and period by cost category.",
    "cost_node_discounted_d_ec": "Discounted cost of each node and period by cost category.",
    "discountFactors_d_p": "Operations and investment discount factors of each period (written only in a debug "
                           "run).",
    "entity_annuity_d_p": "Discounted investment cost per MW of the capacity of each unit, connection and storage "
                          "node decided in a period (written only in a debug run).",
    "years_represented__d": "Number of years each period represents.",
    # Unit and connection flows
    "unit_outputNode_dt_ee": "Flow from each unit to each of its output nodes in each time step.",
    "unit_outputNode_d_ee": "Flow from each unit to each of its output nodes summed over the period and scaled "
                            "to a full year.",
    "unit_inputNode_dt_ee": "Flow from each input node into each unit in each time step, as a negative value.",
    "unit_inputNode_d_ee": "Flow from each input node into each unit summed over the period and scaled to a full "
                           "year, as a negative value.",
    "unit_outputs_cf_d_ee": "Capacity factor of each unit's output flow in each period: the period's flow divided "
                            "by the unit capacity and the period's hours.",
    "unit_inputs_cf_d_ee": "Capacity factor of each unit's input flow in each period: the period's flow divided "
                           "by the unit capacity and the period's hours.",
    "connection_dt_eee": "Net flow through each connection in each time step, positive from the first to the "
                         "second node.",
    "connection_d_eee": "Net flow through each connection summed over the period and scaled to a full year, "
                        "positive from the first to the second node.",
    "connection_leftward_dt_eee": "Net flow of each connection into its first node in each time step.",
    "connection_leftward_d_eee": "Net flow of each connection into its first node summed over the period and "
                                 "scaled to a full year.",
    "connection_rightward_dt_eee": "Net flow of each connection into its second node in each time step.",
    "connection_rightward_d_eee": "Net flow of each connection into its second node summed over the period and "
                                  "scaled to a full year.",
    "connection_losses_dt_eee": "Losses of each connection in each time step: the flow out minus the flow in, "
                                "negative when energy is lost.",
    "connection_losses_d_eee": "Losses of each connection summed over the period and scaled to a full year, "
                               "negative when energy is lost.",
    "connection_cf_d_e": "Capacity factor of each connection in each period: the absolute flow in either "
                         "direction divided by the connection capacity and the period's hours.",
    # CO2
    "CO2__": "CO2 emissions of the whole model totalled over the years the periods represent (one column per "
             "scenario path when a stochastic run has several).",
    "CO2_d_g": "Annual CO2 emissions of each CO2-priced or CO2-capped group, by period.",
    "process_co2_d_eee": "Annual CO2 emissions of each unit and connection flow by commodity and node, by "
                         "period (negative for a removal).",
    # Nodes and node groups
    "node_dt_ep": "Energy balance of each node in each time step by category: flows from and to units and "
                  "connections, self-discharge, loss of load, excess load and inflow.",
    "node_d_ep": "Energy balance of each node by category, summed over the period and scaled to a full year.",
    "node_state_dt_e": "Stored energy of each storage node at each time step.",
    "node_inflow__dt": "Inflow of each node in each time step as given to the model (demand is negative).",
    "node_prices_dt_e": "Price of energy at each balanced node and time step: the cost of one more unit of demand "
                        "there (the dual of the node balance), in the price base of the cost inputs.",
    "node_slack_up_dt_e": "Upward slack (loss of load) of each balanced node in each time step.",
    "node_slack_down_dt_e": "Downward slack (excess load) of each balanced node in each time step.",
    "node_slack_up_d_e": "Upward slack (loss of load) of each balanced node summed over the period and scaled to "
                         "a full year.",
    "node_slack_down_d_e": "Downward slack (excess load) of each balanced node summed over the period and scaled "
                           "to a full year.",
    "nodeGroup_flows_dt_g": "Inflow summed over the member nodes of each indicator group (nodes with an inflow "
                            "method) in each time step.",
    "nodeGroup_flows_d_g": "Inflow summed over the member nodes of each indicator group, summed over the period "
                           "and scaled to a full year.",
    "nodeGroup_flows_dt_gpe": "Energy in each time step into and out of each dispatch group by type (slack, units, "
                              "unit groups, connections, connection groups, inflow, internal losses) and item.",
    "nodeGroup_flows_d_gpe": "Energy into and out of each dispatch group by type and item, summed over the period "
                             "and scaled to a full year.",
    "nodeGroup_gdt_p": "Indicator building blocks of each indicator group in each time step: loss of load, VRE "
                       "generation, excess load, curtailed VRE, inflow and VRE shares.",
    "nodeGroup_gd_p": "Indicators of each indicator group in each period: loss of load, excess load and "
                      "curtailment as shares of demand, the VRE share of demand and the annualized inflow.",
    "nodeGroup_VRE_share_dt_g": "Share of the inflow (demand) of each indicator group covered by VRE units in "
                                "each time step.",
    "nodeGroup_VRE_share_d_g": "Share of the period's inflow (demand) of each indicator group covered by VRE "
                               "units: the VRE energy over the inflow, both summed with the timestep weights.",
    # VRE curtailment and potential
    "unit_VRE_potential_outputNode_dt_ee": "Potential output of each VRE unit to each output node in each time "
                                           "step: capacity times the upper-limit profile.",
    "unit_VRE_potential_outputNode_d_ee": "Potential output of each VRE unit to each output node summed over the "
                                          "period with the timestep weights, not scaled to a full year.",
    "unit_curtailment_outputNode_dt_ee": "Curtailment of each VRE unit at each output node in each time step: "
                                         "the potential minus the actual output.",
    "unit_curtailment_outputNode_d_ee": "Curtailment of each VRE unit at each output node summed over the period "
                                        "with the timestep weights, not scaled to a full year.",
    "unit_curtailment_share_outputNode_dt_ee": "Curtailment of each VRE unit as a share of its potential output "
                                               "in each time step.",
    "unit_curtailment_share_outputNode_d_ee": "Curtailment of each VRE unit as a share of its potential output "
                                              "over the period.",
    # Ramps
    "unit_ramp_inputs_dt_ee": "Change of each unit's input flow from the previous time step.",
    "unit_ramp_outputs_dt_ee": "Change of each unit's output flow from the previous time step.",
    # Online units and startups
    "unit_online_dt_e": "Number of units online of each unit with online variables, in each time step.",
    "unit_online_average_d_e": "Average number of units online of each unit with online variables over the "
                               "period, weighted by step duration.",
    "unit_startup_d_e": "Startups of each unit with online variables in the period, scaled to a full year.",
    "connection_online_dt_e": "Number of units online of each connection with a startup method, in each time "
                              "step.",
    "connection_online_average_d_e": "Average number of units online of each connection with a startup method "
                                     "over the period, weighted by step duration.",
    "connection_startup_d_e": "Startups of each connection with a startup method in the period, scaled to a full "
                              "year.",
    # Capacity and investment
    "unit_capacity_ed_p": "Capacity of each unit in each period: existing, invested, in construction, divested "
                          "and the total in service.",
    "connection_capacity_ed_p": "Capacity of each connection in each period: existing, invested, in construction, "
                                "divested and the total in service.",
    "node_capacity_ed_p": "Storage capacity of each storage node in each period: existing, invested, in "
                          "construction, divested and the total in service.",
    "dual_invest_effective_unit_d_e": "Marginal value of one more unit of capacity of each investable unit in "
                                      "each period (positive: more capacity would lower the cost).",
    "dual_invest_effective_connection_d_e": "Marginal value of one more unit of capacity of each investable "
                                            "connection in each period (positive: more capacity would lower the "
                                            "cost).",
    "dual_invest_effective_node_d_e": "Marginal value of one more unit of storage capacity of each investable "
                                      "storage node in each period (positive: more capacity would lower the "
                                      "cost).",
    # Prices of constraints
    "co2_price_period_d_g": "Shadow price of the per-period CO2 cap of each group, in the currency of the period "
                            "(positive when the cap binds).",
    "co2_price_total_d_g": "Shadow price of the cumulative CO2 cap of each group, in discounted currency per "
                           "tonne, the same in every period.",
    "reserve_prices_dt_ppg": "Price of each reserve, direction and group in each time step: the cost of one more "
                             "MW of every requirement of the group held for the step.",
    # Reserves
    "process_reserve_upDown_node_dt_eppe": "Reserve held by each unit or connection for each reserve, direction "
                                           "and node in each time step.",
    "process_reserve_average_d_eppe": "Reserve held by each unit or connection for each reserve, direction and "
                                      "node, the time-step-weighted average MW over the period's realized steps.",
    "nodeGroup_reserve_dt_eegp": "Reserve requirement of each reserve, direction and group in each time step: the "
                                 "need of the binding requirement row and the reserve it counts (for an n-1 row "
                                 "without the failing process's own reserve, so requirement − counted is the "
                                 "shortfall), the need of each method, the reserve provided and the shortfall.",
    "nodeGroup_reserve_d_eegp": "Reserve requirement, provision and shortfall of each reserve, direction and group "
                                "in each period: time-step-weighted averages and step maxima.",
    "nodeGroup_reserve_provider_dt_eegee": "Counted reserve (× reliability) of each unit or connection at each "
                                           "node for each reserve, direction and group in each time step, summing "
                                           "to the group's provided reserve.",
    # Inertia
    "nodeGroup_inertia_dt_g": "Inertia available in each inertia group in each time step.",
    "nodeGroup_unit_node_inertia_dt_gee": "Inertia contribution of each unit or connection at each node of each "
                                          "inertia group in each time step.",
    "nodeGroup_inertia_largest_flow_dt_g": "Largest single flow into the nodes of each inertia group in each time "
                                           "step.",
    # Slacks of group constraints
    "nodeGroup_slack_capacity_margin_d_g": "Capacity margin shortfall of each group in each investment period, "
                                           "not scaled to a year.",
    "nodeGroup_capacity_margin_d_g": "Capacity margin of each solve with investment periods, per group and "
                                     "investment period at the step with the largest deficit: the requirement, "
                                     "what firm, VRE, storage, other units, connections and consumption count, "
                                     "the shortfall, its penalty and the shadow price.",
    "nodeGroup_capacity_margin_entity_d_g": "Contribution of each unit, connection and node to the capacity margin "
                                            "of each group at its binding step, per solve and investment period, "
                                            "summing to the counted capacity.",
    "nodeGroup_slack_inertia_dt_g": "Inertia shortfall of each inertia group in each time step.",
    "nodeGroup_slack_inertia_d_g": "Inertia shortfall of each inertia group summed over the period and scaled to "
                                   "a full year.",
    "nodeGroup_slack_nonsync_dt_g": "Non-synchronous generation above the allowed share of each group in each "
                                    "time step.",
    "nodeGroup_slack_nonsync_d_g": "Non-synchronous generation above the allowed share of each group summed over "
                                   "the period and scaled to a full year.",
    "nodeGroup_slack_reserve_dt_eeg": "Reserve shortfall of each reserve, direction and group in each time step.",
    "nodeGroup_slack_reserve_d_eeg": "Reserve shortfall of each reserve, direction and group summed over the "
                                     "period and scaled to a full year.",
    # Flow groups
    "flowGroup_gd_p": "Sum of the absolute flows of the member legs of each flow group over the period "
                      "(cumulative_flow) and its average power (average_flow).",
    "flowGroup_gd_t": "Signed net flow into the nodes of each flow group in each time step (into the group's "
                      "nodes positive).",
    # DC power flow
    "dc_angle_dt_e": "Voltage angle of each DC power-flow node in each time step.",
    "dc_angle_diff_dt_e": "Voltage angle difference between the first and the second node of each DC power-flow "
                          "connection in each time step.",
    # Membership sets
    "group_node": "Membership set: the nodes of each group.",
    "group_process": "Membership set: the units and connections of each group.",
    "group_process_node": "Membership set: the (unit or connection, node) flows of each group.",
    "timeline_breaks": "Realized time steps at which a time-series plot starts a new line (gaps, representative "
                       "period starts, periods that do not continue the previous one).",
    "nodeGroupDispatch": "Set of the node groups with dispatch output (print_dispatch).",
    "nodeGroupIndicators": "Set of the node groups with indicator output (print_indicators).",
    "flowGroupIndicators": "Set of the flow groups with indicator output.",
    "connection_dc_power_flow": "Set of the connections in a DC power-flow network.",
    "node_dc_power_flow": "Set of the nodes in a DC power-flow network.",
}

OUTPUT_DIMS: dict[str, OutputDims] = {
    "costs_dt_p": OutputDims(_DT, ("category",), "annualized_costs_d_p", _CATEGORY_PART),
    "annualized_costs_d_p": OutputDims(_D, ("category",), "costs_dt_p", _CATEGORY_PART),
    "costs_discounted_d_p": OutputDims(_D, ("category",), kinds=_CATEGORY_PART),
    "costs_discounted_p_": OutputDims(("category",), kinds=_CATEGORY_PART),
    "cost_unit_annualized_d_ec": OutputDims(("period", "unit"), ("category",), kinds=_CATEGORY_PART),
    "cost_connection_annualized_d_ec": OutputDims(("period", "connection"), ("category",), kinds=_CATEGORY_PART),
    "cost_node_annualized_d_ec": OutputDims(("period", "node"), ("category",), kinds=_CATEGORY_PART),
    "cost_unit_discounted_d_ec": OutputDims(("period", "unit"), ("category",), kinds=_CATEGORY_PART),
    "cost_connection_discounted_d_ec": OutputDims(("period", "connection"), ("category",), kinds=_CATEGORY_PART),
    "cost_node_discounted_d_ec": OutputDims(("period", "node"), ("category",), kinds=_CATEGORY_PART),
    "discountFactors_d_p": OutputDims(_D, ("param",), kinds=_PARAM_QUANTITY),
    "entity_annuity_d_p": OutputDims(("solve", "period"), ("entity",)),
    "years_represented__d": OutputDims(_D, ("param",), kinds=_PARAM_QUANTITY),
    "unit_outputNode_dt_ee": OutputDims(_DT, ("unit", "node"), "unit_outputNode_d_ee"),
    "unit_outputNode_d_ee": OutputDims(_D, ("unit", "node"), "unit_outputNode_dt_ee"),
    "unit_inputNode_dt_ee": OutputDims(_DT, ("unit", "node"), "unit_inputNode_d_ee"),
    "unit_inputNode_d_ee": OutputDims(_D, ("unit", "node"), "unit_inputNode_dt_ee"),
    "unit_outputs_cf_d_ee": OutputDims(_D, ("unit", "sink")),
    "unit_inputs_cf_d_ee": OutputDims(_D, ("unit", "source")),
    "connection_dt_eee": OutputDims(_DT, ("process",), "connection_d_eee"),
    "connection_d_eee": OutputDims(_D, ("process",), "connection_dt_eee"),
    "connection_leftward_dt_eee": OutputDims(_DT, ("process", "node"), "connection_leftward_d_eee"),
    "connection_leftward_d_eee": OutputDims(_D, ("process", "node"), "connection_leftward_dt_eee"),
    "connection_rightward_dt_eee": OutputDims(_DT, ("process", "node"), "connection_rightward_d_eee"),
    "connection_rightward_d_eee": OutputDims(_D, ("process", "node"), "connection_rightward_dt_eee"),
    "connection_losses_dt_eee": OutputDims(_DT, ("process",), "connection_losses_d_eee"),
    "connection_losses_d_eee": OutputDims(_D, ("process",), "connection_losses_dt_eee"),
    "connection_cf_d_e": OutputDims(_D, ("connection",)),
    "CO2__": OutputDims(("param_CO2",), ("scope",), kinds=(("param_CO2", "quantity"), ("scope", "quantity"))),
    "CO2_d_g": OutputDims(_D, ("group",)),
    "process_co2_d_eee": OutputDims(_D, ("type", "process", "commodity", "node"), kinds=(("type", "entity"),)),
    "node_dt_ep": OutputDims(_DT, ("node", "category"), "node_d_ep", _NODE_CATEGORY),
    "node_d_ep": OutputDims(_D, ("node", "category"), "node_dt_ep", _NODE_CATEGORY),
    "node_state_dt_e": OutputDims(_DT, ("node",)),
    "node_inflow__dt": OutputDims(_DT, ("node",)),
    "node_prices_dt_e": OutputDims(("solve", "period", "time"), ("node",)),
    "node_slack_up_dt_e": OutputDims(_DT, ("node",), "node_slack_up_d_e"),
    "node_slack_down_dt_e": OutputDims(_DT, ("node",), "node_slack_down_d_e"),
    "node_slack_up_d_e": OutputDims(_D, ("node",), "node_slack_up_dt_e"),
    "node_slack_down_d_e": OutputDims(_D, ("node",), "node_slack_down_dt_e"),
    "nodeGroup_flows_dt_g": OutputDims(_DT, ("group",), "nodeGroup_flows_d_g"),
    "nodeGroup_flows_d_g": OutputDims(_D, ("group",), "nodeGroup_flows_dt_g"),
    "nodeGroup_flows_dt_gpe": OutputDims(_DT, ("group", "type", "item"), "nodeGroup_flows_d_gpe",
                                          _GROUP_FLOW_KINDS),
    "nodeGroup_flows_d_gpe": OutputDims(_D, ("group", "type", "item"), "nodeGroup_flows_dt_gpe",
                                         _GROUP_FLOW_KINDS),
    "nodeGroup_gdt_p": OutputDims(("group", "period", "time"), ("parameter",), "nodeGroup_gd_p",
                                   _PARAMETER_QUANTITY),
    "nodeGroup_gd_p": OutputDims(("group", "period"), ("parameter",), "nodeGroup_gdt_p", _PARAMETER_QUANTITY),
    "nodeGroup_VRE_share_dt_g": OutputDims(_DT, ("group",), "nodeGroup_VRE_share_d_g"),
    "nodeGroup_VRE_share_d_g": OutputDims(_D, ("group",), "nodeGroup_VRE_share_dt_g"),
    "unit_VRE_potential_outputNode_dt_ee": OutputDims(_DT, ("unit", "sink"), "unit_VRE_potential_outputNode_d_ee"),
    "unit_VRE_potential_outputNode_d_ee": OutputDims(_D, ("unit", "sink"), "unit_VRE_potential_outputNode_dt_ee"),
    "unit_curtailment_outputNode_dt_ee": OutputDims(_DT, ("unit", "sink"), "unit_curtailment_outputNode_d_ee"),
    "unit_curtailment_outputNode_d_ee": OutputDims(_D, ("unit", "sink"), "unit_curtailment_outputNode_dt_ee"),
    "unit_curtailment_share_outputNode_dt_ee": OutputDims(_DT, ("unit", "sink"),
                                                          "unit_curtailment_share_outputNode_d_ee"),
    "unit_curtailment_share_outputNode_d_ee": OutputDims(_D, ("unit", "sink"),
                                                         "unit_curtailment_share_outputNode_dt_ee"),
    "unit_ramp_inputs_dt_ee": OutputDims(_DT, ("unit", "source")),
    "unit_ramp_outputs_dt_ee": OutputDims(_DT, ("unit", "sink")),
    "unit_online_dt_e": OutputDims(_DT, ("process",), "unit_online_average_d_e"),
    "unit_online_average_d_e": OutputDims(_D, ("process",), "unit_online_dt_e"),
    "unit_startup_d_e": OutputDims(_D, ("process",)),
    "connection_online_dt_e": OutputDims(_DT, ("process",), "connection_online_average_d_e"),
    "connection_online_average_d_e": OutputDims(_D, ("process",), "connection_online_dt_e"),
    "connection_startup_d_e": OutputDims(_D, ("process",)),
    "unit_capacity_ed_p": OutputDims(("unit", "period"), ("parameter",), kinds=_PARAMETER_QUANTITY),
    "connection_capacity_ed_p": OutputDims(("connection", "period"), ("parameter",), kinds=_PARAMETER_QUANTITY),
    "node_capacity_ed_p": OutputDims(("node", "period"), ("parameter",), kinds=_PARAMETER_QUANTITY),
    "dual_invest_effective_unit_d_e": OutputDims(_D, ("entity",)),
    "dual_invest_effective_connection_d_e": OutputDims(_D, ("entity",)),
    "dual_invest_effective_node_d_e": OutputDims(_D, ("entity",)),
    "co2_price_period_d_g": OutputDims(_D, ("group",)),
    "co2_price_total_d_g": OutputDims(_D, ("group",)),
    "reserve_prices_dt_ppg": OutputDims(("solve", "period", "time"), ("reserve", "updown", "node_group"),
                                         kinds=_RESERVE_QUANTITY),
    "process_reserve_upDown_node_dt_eppe": OutputDims(_DT, ("process", "reserve", "updown", "node"),
                                                      "process_reserve_average_d_eppe", _RESERVE_QUANTITY),
    "process_reserve_average_d_eppe": OutputDims(_D, ("process", "reserve", "updown", "node"),
                                                 "process_reserve_upDown_node_dt_eppe", _RESERVE_QUANTITY),
    "nodeGroup_inertia_dt_g": OutputDims(_DT, ("group",)),
    "nodeGroup_unit_node_inertia_dt_gee": OutputDims(_DT, ("group", "process", "node")),
    "nodeGroup_inertia_largest_flow_dt_g": OutputDims(_DT, ("group",)),
    "nodeGroup_slack_capacity_margin_d_g": OutputDims(_D, ("group",)),
    "nodeGroup_capacity_margin_d_g": OutputDims(("solve", "period", "group", "binding_time"), ("parameter",),
                                                kinds=(*_PARAMETER_QUANTITY, ("binding_time", "quantity"))),
    "nodeGroup_capacity_margin_entity_d_g": OutputDims(("solve", "period", "group", "binding_time"),
                                                       ("class", "entity"),
                                                       kinds=(("binding_time", "quantity"), ("class", "part"))),
    "nodeGroup_slack_inertia_dt_g": OutputDims(_DT, ("group",), "nodeGroup_slack_inertia_d_g"),
    "nodeGroup_slack_inertia_d_g": OutputDims(_D, ("group",), "nodeGroup_slack_inertia_dt_g"),
    "nodeGroup_slack_nonsync_dt_g": OutputDims(_DT, ("group",), "nodeGroup_slack_nonsync_d_g"),
    "nodeGroup_slack_nonsync_d_g": OutputDims(_D, ("group",), "nodeGroup_slack_nonsync_dt_g"),
    "nodeGroup_slack_reserve_dt_eeg": OutputDims(_DT, ("reserve", "updown", "node_group"),
                                                 "nodeGroup_slack_reserve_d_eeg", _RESERVE_QUANTITY),
    "nodeGroup_reserve_dt_eegp": OutputDims(_DT, ("reserve", "updown", "node_group", "quantity"),
                                            "nodeGroup_reserve_d_eegp",
                                            (*_RESERVE_QUANTITY, ("quantity", "quantity"))),
    "nodeGroup_reserve_d_eegp": OutputDims(_D, ("reserve", "updown", "node_group", "quantity"),
                                           "nodeGroup_reserve_dt_eegp",
                                           (*_RESERVE_QUANTITY, ("quantity", "quantity"))),
    "nodeGroup_reserve_provider_dt_eegee": OutputDims(_DT, ("reserve", "updown", "node_group", "process", "node"),
                                                      None,
                                                      (*_RESERVE_QUANTITY, ("process", "part"), ("node", "part"))),
    "nodeGroup_slack_reserve_d_eeg": OutputDims(_D, ("reserve", "updown", "node_group"),
                                                "nodeGroup_slack_reserve_dt_eeg", _RESERVE_QUANTITY),
    "flowGroup_gd_p": OutputDims(("group", "period"), ("parameter",), "flowGroup_gd_t", _PARAMETER_QUANTITY),
    "flowGroup_gd_t": OutputDims(("group", "period", "time"), ("parameter",), "flowGroup_gd_p", _PARAMETER_QUANTITY),
    "dc_angle_dt_e": OutputDims(_DT, ("node",)),
    "dc_angle_diff_dt_e": OutputDims(_DT, ("connection",)),
    "group_node": OutputDims(("group", "node")),
    "group_process": OutputDims(("group", "process")),
    "group_process_node": OutputDims(("group", "process", "node")),
    "timeline_breaks": OutputDims(_DT),
    "nodeGroupDispatch": OutputDims(("group",)),
    "nodeGroupIndicators": OutputDims(("group",)),
    "flowGroupIndicators": OutputDims(("group",)),
    "connection_dc_power_flow": OutputDims(("connection",)),
    "node_dc_power_flow": OutputDims(("node",)),
}


def level_kind(stem: str, level: str) -> str:
    """What combining over *level* of the result table *stem* means (:meth:`OutputDims.level_kind`); a table
    that is not declared falls back to :func:`default_level_kind`."""
    dims = OUTPUT_DIMS.get(stem)
    return dims.level_kind(level) if dims is not None else default_level_kind(level)


def is_dimension(column: str) -> bool:
    """True if ``column`` is an index/key dimension rather than a measure."""
    return column in DIMENSION_COLUMNS


def result_key_summary(result_key: str) -> "tuple[str, str, str] | None":
    """Output-level ``(unit, semantics, tooltip)`` for a result key.

    For a uniform output this is its transform's fields; for a mixed-unit
    table the unit lists the distinct measure units and semantics is
    ``"mixed"``.  Returns ``None`` for an undeclared output or a pure
    membership/index set (nothing to annotate).  Intended for UI hovers
    (e.g. the result viewer's plot-tree tooltip), where ``result_key`` is the
    plot variant's key.
    """
    spec = OUTPUT_TRANSFORM.get(result_key)
    if spec is None:
        return None
    if isinstance(spec, dict):
        units = sorted({t.unit for t in spec.values() if t.unit})
        sems = sorted({t.semantics.value for t in spec.values()
                       if t.semantics is not Semantics.DIMENSION})
        if not sems:
            return None
        return ("; ".join(units) if units else "ratio",
                "mixed" if len(sems) > 1 else sems[0],
                "Mixed-unit table — units vary by column.")
    if spec.semantics is Semantics.DIMENSION:
        return None
    return (spec.unit or "ratio", spec.semantics.value, spec.tooltip)


def result_variant_summary(
    result_key: str, variant_letter: str
) -> "tuple[str, str, str] | None":
    """Variant-adjusted ``(unit, semantics, tooltip)`` for a plot variant.

    The ``a`` (sum-over-periods → total) and ``w`` (weekly chunk
    aggregation) plot variants are *on-the-fly* aggregations layered on the
    same base ``result_key`` as the ``p``/``h`` variants — they have no
    output of their own in :data:`OUTPUT_TRANSFORM`.  So their unit and
    semantics differ from the base summary and must be derived here:

    * ``'a'`` — the variant totals the base over the whole horizon.  An
      annual-rate base unit (suffix ``/a``, e.g. ``M CUR/a``, ``MWh/a``)
      becomes the absolute total (``M CUR``, ``MWh``); a base unit without
      ``/a`` (e.g. a discounted ``M CUR``) is already absolute and stays as
      is.  Semantics become ``"total"`` (a horizon total, *not* annual).
    * ``'w'`` — weekly chunk aggregation keeps the base/``h`` unit unchanged;
      semantics become ``"weekly"``.
    * any other letter (``'p'``, ``'h'``, …) — the base summary unchanged.

    Returns ``None`` when the base output is undeclared or a pure
    membership/index set (i.e. when :func:`result_key_summary` returns
    ``None``), so callers can skip it exactly as they do for the base.
    Pure and GUI-free; the single source for variant-aware UI hovers and
    plot value-axis units.
    """
    base = result_key_summary(result_key)
    if base is None:
        return None
    unit, semantics, desc = base
    if variant_letter == 'a':
        if unit and unit.endswith('/a'):
            unit = unit[:-2]
        return (unit, "total", desc)
    if variant_letter == 'w':
        return (unit, "weekly", desc)
    return base


def output_metadata_rows(output_key: str, columns) -> list[dict[str, str]]:
    """Flat ``[{output, column, unit, semantics, tooltip, formula}, …]`` rows
    for the *measure* columns of one output (dimension columns dropped).

    Single source for the tabular renderers that list metadata in a side
    table (the Excel ``_output_metadata`` sheet).  Returns ``[]`` for an
    undeclared output or a pure membership/index set.
    """
    meta = derive_column_meta(output_key, columns)
    if not meta:
        return []
    rows: list[dict[str, str]] = []
    for col in columns:
        cm = meta.get(str(col))
        if cm is None or cm.semantics is Semantics.DIMENSION:
            continue
        rows.append({
            "output": output_key,
            "column": cm.name,
            "unit": cm.unit,
            "semantics": cm.semantics.value,
            "tooltip": cm.tooltip,
            "long": cm.long,
            "formula": cm.formula,
        })
    return rows


def datapackage_resource(
    output_key: str, path: str, index_names, measure_columns,
) -> dict:
    """A Frictionless ``tabular-data-resource`` describing one written CSV.

    ``index_names`` are the (reset-to-column) dimension levels and
    ``measure_columns`` the value columns — together the CSV's header.  Each
    field carries its ``type`` and, for measures, the derived ``unit`` /
    semantics / description / formula (under ``flextool:`` keys so the
    descriptor stays valid Frictionless).  Always returns a resource (an
    undeclared output simply yields string/dimension fields), so the sidecar
    documents the full ``output_csv/`` tree, declared or not.
    """
    meta = derive_column_meta(output_key, measure_columns) or {}
    fields: list[dict[str, str]] = []
    seen: set[str] = set()

    def _dimension_field(name: str) -> dict[str, str]:
        return {"name": name, "type": "string", "flextool:semantics": "dimension"}

    for name in index_names:
        if name is None:
            continue
        key = str(name)
        if key in seen:
            continue
        seen.add(key)
        fields.append(_dimension_field(key))

    for col in measure_columns:
        name = str(col)
        if name in seen:
            continue
        seen.add(name)
        cm = meta.get(name)
        if cm is None or cm.semantics is Semantics.DIMENSION:
            fields.append(_dimension_field(name))
            continue
        spec: dict[str, str] = {
            "name": name,
            "type": "number",
            "flextool:semantics": cm.semantics.value,
        }
        if cm.unit:
            spec["unit"] = cm.unit
        # Frictionless `description` carries the longer form when present.
        if cm.long or cm.tooltip:
            spec["description"] = cm.long or cm.tooltip
        if cm.formula:
            spec["flextool:formula"] = cm.formula
        if cm.docs:
            spec["flextool:docs"] = cm.docs
        fields.append(spec)

    return {
        "name": output_key.replace("_", "-").strip("-").lower() or "resource",
        "path": path,
        "profile": "tabular-data-resource",
        "schema": {"fields": fields},
    }


def derive_column_meta(
    output_key: str, columns,
) -> dict[str, ColumnMeta] | None:
    """Derive ``{column_name: ColumnMeta}`` for one output.

    ``columns`` is the iterable of *leaf* column names (e.g. cost categories),
    typically ``df.columns`` before any scenario-level wrapping.  Dimension
    columns get ``Semantics.DIMENSION``; measure columns get the output's
    declared transform.  The declared value may be a single ``Transform``
    (uniform) or a ``{column_label: Transform}`` map (mixed-unit tables, with
    an optional ``'*'`` default).  Returns ``None`` when the output is not (yet)
    declared in :data:`OUTPUT_TRANSFORM`, so callers can treat absence as
    "no metadata".
    """
    spec = OUTPUT_TRANSFORM.get(output_key)
    if spec is None:
        return None
    docs = DOCS_ANCHOR.get(output_key, "")
    out: dict[str, ColumnMeta] = {}
    for col in columns:
        name = str(col)
        tf = spec.get(name, spec.get('*')) if isinstance(spec, dict) else spec
        # Index/key dimension, a pure membership set, or a column absent from a
        # mixed-table map → dimension.
        if tf is None or tf.semantics is Semantics.DIMENSION or is_dimension(name):
            out[name] = ColumnMeta(name, "", Semantics.DIMENSION,
                                   "Index / key dimension.")
            continue
        out[name] = ColumnMeta(
            name=name,
            unit=tf.unit,
            semantics=tf.semantics,
            tooltip=tf.tooltip,
            long=tf.description,
            formula=FORMULA_OVERRIDE.get((output_key, name), ""),
            docs=docs,
        )
    return out
