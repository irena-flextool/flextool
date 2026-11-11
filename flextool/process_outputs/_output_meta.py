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

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum

from flextool import units_of_measure as uom
from flextool.units_of_measure import UnitSource, UnitSourceKind


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
    #: The unit as a template of the node's unit of measure (``{A}``, ``{R}``, ``{kR}``, ``{kA}``, see
    #: :mod:`flextool.units_of_measure`); ``""`` for a unit that never follows the nodes (costs, CO2, counts,
    #: reserves, inertia).  ``fill(template, "MWh") == unit`` for every transform (a registry test).
    template: str = ""
    #: The template when the column's entity is a node (``entity_annuity_d_p`` holds units, connections and
    #: storage nodes in one table: per MW of capacity, per MWh of storage); ``""`` = :attr:`template`.
    node_template: str = ""


# The transform catalog.  Each entry is the result of applying one output-stage
# transform to its base quantity.  Names mirror the code paths in out_*/calc_*.
RATE = Transform(
    "MW", Semantics.INSTANTANEOUS,
    "Power at the timestep (a rate). Multiply by step_duration for energy.",
    algebra="v_flow",
    template="{R}",
)
ENERGY_ANNUAL = Transform(
    "MWh/a", Semantics.ANNUALIZED,
    "Energy, scaled from the sampled timeline to a full-year equivalent.",
    algebra="MW · step_duration · timestep_weight / period_share",
    template="{A}/a",
)
ENERGY_PERIOD = Transform(
    "MWh", Semantics.PER_PERIOD,
    "Energy summed over the sampled timeline (NOT scaled to a full year).",
    algebra="MW · step_duration · timestep_weight  (no /period_share)",
    template="{A}",
)
PREENERGY_ANNUAL = Transform(
    "MWh/a", Semantics.ANNUALIZED,
    "Energy (already per-step MWh), scaled to a full-year equivalent.",
    algebra="MWh_step · timestep_weight / period_share",
    template="{A}/a",
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
    template="{R}",
)
RATIO = Transform(
    "", Semantics.RATIO,
    "Dimensionless ratio (share / capacity factor).",
)
ENERGY_PERSTEP = Transform(
    "MWh", Semantics.PER_STEP,
    "Energy within the timestep (already per-step MWh, not a rate).",
    template="{A}",
)
RAMP = Transform(
    "MW", Semantics.INSTANTANEOUS,
    "Change in flow between consecutive timesteps (a ramp).",
    algebra="flow[t] − flow[t-1]",
    template="{R}",
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
    template="{R}",
)
CAPACITY_MWH = Transform(
    "MWh", Semantics.LEVEL,
    "Storage energy capacity (a stock).",
    description="Storage capacity by period: existing, invested (capacity that comes online in the period), "
                "in_construction (capacity decided but not yet operating; empty without a construction lead "
                "time), divested and total (capacity in service). Capacity in construction is not in the total.",
    algebra="total = existing + invested − divested (cumulative; in_construction excluded)",
    template="{A}",
)
STORAGE_STATE = Transform(
    "MWh", Semantics.LEVEL,
    "Stored energy at the timestep (state of charge).",
    algebra="v_state · unitsize",
    template="{A}",
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
    template="CUR/{A}",
)
ENERGY_HORIZON = Transform(
    "MWh", Semantics.HORIZON,
    "Energy over the represented horizon: each period's energy per year times its years represented, summed.",
    algebra="Σ_d years_represented · MWh_step · timestep_weight / period_share",
    template="{A}",
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
    template="CUR/{R}", node_template="CUR/{A}",
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
    template="CUR/{kR}",
)
DUAL_INVEST_KWH = Transform(
    "CUR/kWh", Semantics.PRICE,
    "Effective marginal system value of +1 unit of storage capacity "
    "(same convention as invest_cost).",
    template="CUR/{kA}",
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
_NODEGROUP_G_P = {
    '1. Loss of load share': RATIO,
    '2. VRE share of demand': RATIO,
    '3. Excess load share': RATIO,
    '4. Curtailed VRE of demand': RATIO,
    '5. Inflow (horizon total)': ENERGY_HORIZON,
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
    'cumulative_flow': ENERGY_ANNUAL,   # MWh/a: the period's energy scaled to a full year
    'average_flow': AVERAGE_MW,         # MW: the weighted time-average
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
    "nodeGroup_VRE_share_g": RATIO,
    "nodeGroup_VRE_share_dt_g": RATIO,
    "unit_VRE_potential_outputNode_dt_ee": RATE,
    "unit_curtailment_outputNode_dt_ee": RATE,
    "unit_VRE_potential_outputNode_d_ee": ENERGY_ANNUAL,
    "unit_curtailment_outputNode_d_ee": ENERGY_ANNUAL,
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
    "unit_startup_energy_dt_ee": ENERGY_PERSTEP,
    "unit_startup_energy_d_ee": PREENERGY_ANNUAL,
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
    "nodeGroup_g_p": _NODEGROUP_G_P,
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
    "nodeGroup_g_p": "Indicators of each indicator group over the whole horizon: loss of load, excess load and "
                     "curtailment as shares of demand, the VRE share of demand and the curtailed share of the VRE "
                     "potential, each a ratio of the energies over all periods (each period scaled to a year and "
                     "times its years represented), and the inflow over the horizon.",
    "nodeGroup_VRE_share_g": "Share of the inflow (demand) of each indicator group covered by VRE units over the "
                             "whole horizon: the total VRE energy over the total inflow (each period scaled to a "
                             "year and times its years represented).",
    "nodeGroup_VRE_share_dt_g": "Share of the inflow (demand) of each indicator group covered by VRE units in "
                                "each time step.",
    "nodeGroup_VRE_share_d_g": "Share of the period's inflow (demand) of each indicator group covered by VRE "
                               "units: the VRE energy over the inflow, both summed with the timestep weights.",
    # VRE curtailment and potential
    "unit_VRE_potential_outputNode_dt_ee": "Potential output of each VRE unit to each output node in each time "
                                           "step: capacity times the upper-limit profile.",
    "unit_VRE_potential_outputNode_d_ee": "Potential output of each VRE unit to each output node in the period, "
                                          "summed with the timestep weights and scaled to a full year (MWh/a).",
    "unit_curtailment_outputNode_dt_ee": "Curtailment of each VRE unit at each output node in each time step: "
                                         "the potential minus the actual output.",
    "unit_curtailment_outputNode_d_ee": "Curtailment of each VRE unit at each output node in the period, summed "
                                        "with the timestep weights and scaled to a full year (MWh/a).",
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
    "unit_startup_energy_dt_ee": "Energy that the start-ups of each unit take from each of its start-up input "
                                 "nodes (startup_input_method) in each time step: startup_energy times the MW "
                                 "started.  Also in the unit's input from the node, the commodity cost and CO2.",
    "unit_startup_energy_d_ee": "Energy that the start-ups of each unit take from each of its start-up input "
                                "nodes, summed over the period and scaled to a full year.",
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
    "flowGroup_gd_p": "Sum of the absolute flows of the member legs of each flow group in the period, summed "
                      "with the timestep weights and scaled to a full year (cumulative_flow, MWh/a), and its "
                      "average power (average_flow, MW).",
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
    "nodeGroup_g_p": OutputDims(("group",), ("parameter",), kinds=_PARAMETER_QUANTITY),
    "nodeGroup_VRE_share_dt_g": OutputDims(_DT, ("group",), "nodeGroup_VRE_share_d_g"),
    "nodeGroup_VRE_share_d_g": OutputDims(_D, ("group",), "nodeGroup_VRE_share_dt_g"),
    "nodeGroup_VRE_share_g": OutputDims(("horizon",), ("group",), kinds=(("horizon", "quantity"),)),
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
    "unit_startup_energy_dt_ee": OutputDims(_DT, ("unit", "node"), "unit_startup_energy_d_ee"),
    "unit_startup_energy_d_ee": OutputDims(_D, ("unit", "node"), "unit_startup_energy_dt_ee"),
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
        measures = [t for t in spec.values() if t.semantics is not Semantics.DIMENSION]
        sems = sorted({t.semantics.value for t in measures})
        if not sems:
            return None
        # A unitless column counts as ``per unit``, as on the workbook's ``Units`` sheet; a table of only
        # unitless columns gets the ``ratio`` sentinel of a uniform unitless output.
        units = (summarise_units([t.unit or UNITLESS for t in measures])
                 if any(t.unit for t in measures) else "")
        return (units or "ratio",
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


def written_variant_summary(parquet_dir, result_key: str, variant_letter: str) -> "tuple[str, str, str] | None":
    """:func:`result_variant_summary` with the units the run wrote into the footer of *result_key*'s parquet in
    the scenario folder *parquet_dir* (columns and row units joined, :func:`summarise_units`), when the folder has
    a :data:`UNIT_CONTEXT_FILE` (some node is not MWh); ``None`` otherwise, so the caller shows the static
    summary.  For the ``a`` variant each unit drops its ``/a``.  The GUI hover of a result whose units follow
    the nodes."""
    import os

    base = result_variant_summary(result_key, variant_letter)
    if base is None or not os.path.exists(os.path.join(str(parquet_dir), UNIT_CONTEXT_FILE)):
        return None
    written = FooterUnits(parquet_dir).table(result_key)
    if written is None:
        return None
    columns, rows = written
    units = [u for u in columns.values() if u]
    if rows:
        units.extend(u for per_row in rows.values() for u in per_row.values() if u)
    if variant_letter == 'a':
        units = [u[:-2] if u.endswith('/a') else u for u in units]
    text = summarise_units(units)
    return (text, base[1], base[2]) if text else None


def output_metadata_rows(output_key: str, columns, *, ctx=None, rows=None) -> list[dict[str, str]]:
    """Flat ``[{output, column, unit, semantics, tooltip, formula}, …]`` rows
    for the *measure* columns of one output (dimension columns dropped).

    Single source for the tabular renderers that list metadata in a side
    table (the Excel ``_output_metadata`` sheet).  Returns ``[]`` for an
    undeclared output or a pure membership/index set.  *ctx* and *rows* as in :func:`resolve_columns`.
    """
    resolved = resolve_columns(output_key, columns, ctx=ctx, rows=rows)
    meta = None if resolved is None else {m.name: m for m in resolved}
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
            "unit": display_unit(cm),
            "semantics": cm.semantics.value,
            "tooltip": cm.tooltip,
            "long": cm.long,
            "formula": cm.formula,
        })
    return rows


def _joined_label(texts: list[str]) -> str:
    """The field name a Frictionless reader derives from the header-row cells *texts* of one column: the
    cells joined with ``|``, where a cell equal to the one above it is skipped (Frictionless's reading of a
    merged header cell, so ``solve, "", "", ""`` gives ``solve|``)."""
    parts = texts[:1]
    for text in texts[1:]:
        if text != parts[-1]:
            parts.append(text)
    return "|".join(parts)


def datapackage_resource(
    output_key: str, path: str, index_names, measure_columns, *, header=None, ctx=None, rows=None,
) -> dict:
    """A Frictionless ``tabular-data-resource`` describing one written CSV.

    ``index_names`` are the (reset-to-column) dimension levels and
    ``measure_columns`` the value columns — together the CSV's header.  Each
    field carries its ``type`` and, for measures, the derived ``unit`` /
    semantics / description / formula (under ``flextool:`` keys so the
    descriptor stays valid Frictionless).  Always returns a resource (an
    undeclared output simply yields string/dimension fields), so the sidecar
    documents the full ``output_csv/`` tree, declared or not.

    ``header`` is the list of column labels as written to the CSV (index
    columns first, a tuple per column when the file has several header rows).
    When given, the field names are the written header: the labels of a
    column joined with ``|`` the way Frictionless joins them (see :func:`_joined_label`; with ``flextool:header`` = the label list) and a
    ``dialect`` with ``headerRows`` / ``headerJoin`` for a multi-row header,
    so a Frictionless reader derives the same field names from the file.  Header rows that are blank in
    every column (a wide table with no value columns) are ``commentRows`` and left out of the names.  *ctx* and
    *rows* as in :func:`resolve_columns` (a row-keyed table's field unit joins the units of its rows).
    """
    resolved = resolve_columns(output_key, measure_columns, ctx=ctx, rows=rows) or []
    meta = {i: m for i, m in enumerate(resolved)}
    n_index = len(list(index_names))
    fields: list[dict] = []
    seen: set[str] = set()

    def _dimension_field(name: str) -> dict:
        return {"name": name, "type": "string", "flextool:semantics": "dimension"}

    def _measure_field(name: str, cm: ColumnMeta) -> dict:
        spec: dict = {
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
        return spec

    header_rows: list[int] = [1]
    blank_rows: list[int] = []
    if header is not None:
        labels = [c if isinstance(c, tuple) else (c,) for c in header]
        n_header_rows = max((len(t) for t in labels), default=1)
        all_texts = [["" if v is None else str(v) for v in label] + [""] * (n_header_rows - len(label))
                     for label in labels]
        # A header row that is blank in every column (a wide table written without value columns) is not
        # a header row to a reader: Frictionless would skip it as a blank row and take the next data row
        # as the header.  It is described as a comment row, and the field names leave it out.
        kept = [i for i in range(n_header_rows) if i == 0 or any(t[i] for t in all_texts)]
        header_rows = [i + 1 for i in kept]
        blank_rows = [i + 1 for i in range(n_header_rows) if i not in kept]
        for pos, full in enumerate(all_texts):
            texts = [full[i] for i in kept]
            name = _joined_label(texts)
            cm = meta.get(pos - n_index) if pos >= n_index else None
            if cm is None or cm.semantics is Semantics.DIMENSION:
                spec = _dimension_field(name)
            else:
                spec = _measure_field(name, cm)
            if len(header_rows) > 1:
                spec["flextool:header"] = texts
            fields.append(spec)
    else:
        for name in index_names:
            if name is None:
                continue
            key = str(name)
            if key in seen:
                continue
            seen.add(key)
            fields.append(_dimension_field(key))

        for i, col in enumerate(measure_columns):
            name = str(col)
            if name in seen:
                continue
            seen.add(name)
            cm = meta.get(i)
            if cm is None or cm.semantics is Semantics.DIMENSION:
                fields.append(_dimension_field(name))
                continue
            fields.append(_measure_field(name, cm))

    resource = {
        "name": output_key.replace("_", "-").strip("-").lower() or "resource",
        "path": path,
        "profile": "tabular-data-resource",
        "schema": {"fields": fields},
    }
    if len(header_rows) > 1:
        resource["dialect"] = {"headerRows": header_rows, "headerJoin": "|"}
    if blank_rows:
        resource.setdefault("dialect", {})["commentRows"] = blank_rows
    return resource


# ── Column → unit resolver ──────────────────────────────────────────────────
# One function gives the unit of every column of a result table; the parquet footer, the Excel sheets, the CSV
# sidecars (``units.csv``, ``datapackage.json``), the results-database descriptions and the plot labels all
# read it, so they cannot disagree.

#: The name of the column level that a scenario comparison puts above the table's own column levels.
SCENARIO_LEVEL = "scenario"
#: The display word of a measure that has no unit (a share, a capacity factor), as in the results database.
UNITLESS = "per unit"


#: The file of a scenario's parquet folder that holds its :class:`UnitContext` when some node is not MWh (absent
#: = every node is MWh).  Written or deleted by every non-replay ``write_outputs`` call; read only by replays.
UNIT_CONTEXT_FILE = "_unit_context.json"
#: The format version of :data:`UNIT_CONTEXT_FILE`.
UNIT_CONTEXT_VERSION = 1


@dataclass(frozen=True)
class UnitContext:
    """What the units of one scenario's result columns can depend on beyond the table itself.

    ``node_units`` maps a node to its amount unit (``node.unit_of_measure``; only the nodes that are not MWh, see
    :func:`flextool.units_of_measure.read_node_units`).  ``capacity_nodes`` maps a unit to its capacity-side nodes
    (its output nodes with a non-zero ``conversion_flow_coeff``, in arc order, else its input nodes),
    ``shared_capacity`` names the units whose outputs share the capacity
    (:func:`flextool.units_of_measure.shares_capacity`), ``connection_nodes`` maps a connection to its two ends,
    ``group_nodes`` a nodeGroup to its member nodes and ``flowgroup_nodes`` a flowGroup to the nodes of its arcs.
    :data:`UNIT_SOURCE` says which of them a table's unit comes from.  A context without ``node_units`` gives
    the default units, except that ``entity_annuity_d_p`` tells its storage nodes (CUR per amount, ``CUR/MWh``)
    from its units and connections (CUR per rate, ``CUR/MW``) whenever the context knows the units and
    connections (:func:`_follows_nodes`).  ``footer`` holds the units a writer already resolved
    (:class:`FooterUnits`, set by :func:`replay_unit_context`); a replay reads them first.  Holds dicts, so it is
    not hashable: do not key a cache on it."""

    node_units: Mapping[str, str] = field(default_factory=dict)
    capacity_nodes: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    shared_capacity: frozenset[str] = frozenset()
    group_nodes: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    connection_nodes: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    flowgroup_nodes: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    footer: "FooterUnits | None" = field(default=None, compare=False)

    @property
    def is_default(self) -> bool:
        """Whether every node is MWh (the units are then exactly the transforms' default units)."""
        return not self.node_units

    def to_json(self) -> dict:
        """The context as the JSON object of :data:`UNIT_CONTEXT_FILE` (sorted, so the file is deterministic)."""
        def nodes(m: Mapping[str, tuple[str, ...]]) -> dict[str, list[str]]:
            return {k: list(m[k]) for k in sorted(m)}
        return {
            "version": UNIT_CONTEXT_VERSION,
            "node_units": {k: self.node_units[k] for k in sorted(self.node_units)},
            "capacity_nodes": nodes(self.capacity_nodes),
            "shared_capacity": sorted(self.shared_capacity),
            "connection_nodes": nodes(self.connection_nodes),
            "group_nodes": nodes(self.group_nodes),
            "flowgroup_nodes": nodes(self.flowgroup_nodes),
        }

    @classmethod
    def from_json(cls, data: Mapping) -> "UnitContext":
        """The context of a :meth:`to_json` object (missing maps read as empty)."""
        def nodes(key: str) -> dict[str, tuple[str, ...]]:
            return {str(k): tuple(str(n) for n in v) for k, v in (data.get(key) or {}).items()}
        return cls(
            node_units={str(k): str(v) for k, v in (data.get("node_units") or {}).items()},
            capacity_nodes=nodes("capacity_nodes"),
            shared_capacity=frozenset(str(u) for u in data.get("shared_capacity") or ()),
            group_nodes=nodes("group_nodes"),
            connection_nodes=nodes("connection_nodes"),
            flowgroup_nodes=nodes("flowgroup_nodes"),
        )


def write_unit_context(parquet_dir: str, ctx: "UnitContext | None") -> None:
    """Write :data:`UNIT_CONTEXT_FILE` into *parquet_dir* when *ctx* has a node that is not MWh, else delete the
    file (a run whose nodes are all MWh again must not leave a stale context behind).  The folder is created only
    when there is something to write."""
    import json
    import os

    path = os.path.join(parquet_dir, UNIT_CONTEXT_FILE)
    if ctx is None or ctx.is_default:
        if os.path.exists(path):
            os.remove(path)
        return
    os.makedirs(parquet_dir, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(ctx.to_json(), fh, indent=1)


class FooterUnits:
    """The units the writer resolved into the parquet footers of one scenario folder (``units`` of each column
    and ``flextool_row_units``), read from a table's file on first use.  A replay takes a table's units from
    here before it resolves them again, so a replay shows what the run wrote (one source of the units)."""

    def __init__(self, parquet_dir) -> None:
        import os

        self._dir = str(parquet_dir)
        self._cache: dict[str, "tuple[dict[str, str], dict | None] | None"] = {}
        self._exists = os.path.isdir(self._dir)

    def table(self, output_key: str) -> "tuple[dict[str, str], dict | None] | None":
        """``({column: unit}, row units or None)`` of *output_key*'s footer, ``None`` without a file or units."""
        if output_key in self._cache:
            return self._cache[output_key]
        found = None
        if self._exists:
            import os

            path = os.path.join(self._dir, f"{output_key}.parquet")
            if os.path.isfile(path):
                from flextool.lean_parquet import read_lean_parquet_column_meta, read_lean_parquet_row_units
                try:
                    meta = read_lean_parquet_column_meta(path)
                    rows = read_lean_parquet_row_units(path)
                except Exception:  # noqa: BLE001 - an unreadable footer: resolve the units instead
                    meta, rows = None, None
                if meta:
                    found = ({str(k): str(v.get("unit", "")) for k, v in meta.items() if isinstance(v, dict)}, rows)
        self._cache[output_key] = found
        return found


def replay_unit_context(parquet_dir) -> UnitContext:
    """The :class:`UnitContext` of a replay of the scenario folder *parquet_dir*: its
    :data:`UNIT_CONTEXT_FILE` (an empty context without one) with the folder's footers
    (:class:`FooterUnits`), so the replayed units are those the run wrote."""
    from dataclasses import replace

    ctx = load_unit_context(parquet_dir) or UnitContext()
    return replace(ctx, footer=FooterUnits(parquet_dir))


def _footer_of(ctx, output_key: str) -> "tuple[dict[str, str], dict | None] | None":
    """The written units of *output_key* in the footer of *ctx* (one context), ``None`` without them."""
    footer = getattr(ctx, "footer", None) if ctx is not None else None
    return footer.table(output_key) if footer is not None else None


def _leaf_name(label: tuple, scen_pos: "int | None") -> str:
    """The footer key of a column: ``str`` of its label without the comparison ``scenario`` level (a one-level
    label as its value, as the writer keyed it)."""
    leaf = label if scen_pos is None else label[:scen_pos] + label[scen_pos + 1:]
    return str(leaf[0]) if len(leaf) == 1 else str(tuple(leaf))


def load_unit_context(parquet_dir) -> "UnitContext | None":
    """The :class:`UnitContext` of a scenario's parquet folder, ``None`` when it has no
    :data:`UNIT_CONTEXT_FILE` (every node is MWh) or the file cannot be read."""
    import json
    import logging
    import os

    path = os.path.join(str(parquet_dir), UNIT_CONTEXT_FILE)
    if not os.path.exists(path):
        return None
    try:
        with open(path, encoding="utf-8") as fh:
            return UnitContext.from_json(json.load(fh))
    except (OSError, ValueError, AttributeError) as exc:
        logging.warning("Could not read %s (%s): the units are shown for MWh nodes.", path, exc)
        return None


_U = UnitSourceKind

#: Where the unit of a result table that follows its nodes' ``unit_of_measure`` comes from: the level that
#: carries the node(s) and how (:class:`flextool.units_of_measure.UnitSource`).  A ``node`` level beats the
#: "connection -> both ends" rule (``connection_leftward_*`` take the unit of their node).  Every table with a
#: power or energy unit is here or in :data:`DEFAULT_UNITS_ONLY` (pinned by a registry test).  An
#: ``entity`` level (``entity_annuity_d_p``) is read by class: a connection, else a unit, else a node; a node and
#: a unit of the same name cannot be told apart there.
UNIT_SOURCE: dict[str, UnitSource] = {
    "entity_annuity_d_p": UnitSource("entity", _U.ENTITY_BY_CLASS),
    "unit_outputNode_dt_ee": UnitSource("node", _U.NODE),
    "unit_outputNode_d_ee": UnitSource("node", _U.NODE),
    "unit_inputNode_dt_ee": UnitSource("node", _U.NODE),
    "unit_inputNode_d_ee": UnitSource("node", _U.NODE),
    "connection_dt_eee": UnitSource("process", _U.CONNECTION_ENDS),
    "connection_d_eee": UnitSource("process", _U.CONNECTION_ENDS),
    "connection_leftward_dt_eee": UnitSource("node", _U.NODE),
    "connection_leftward_d_eee": UnitSource("node", _U.NODE),
    "connection_rightward_dt_eee": UnitSource("node", _U.NODE),
    "connection_rightward_d_eee": UnitSource("node", _U.NODE),
    "connection_losses_dt_eee": UnitSource("process", _U.CONNECTION_ENDS),
    "connection_losses_d_eee": UnitSource("process", _U.CONNECTION_ENDS),
    "node_dt_ep": UnitSource("node", _U.NODE),
    "node_d_ep": UnitSource("node", _U.NODE),
    "node_state_dt_e": UnitSource("node", _U.NODE),
    "node_inflow__dt": UnitSource("node", _U.NODE),
    "node_prices_dt_e": UnitSource("node", _U.NODE),
    "node_slack_up_dt_e": UnitSource("node", _U.NODE),
    "node_slack_down_dt_e": UnitSource("node", _U.NODE),
    "node_slack_up_d_e": UnitSource("node", _U.NODE),
    "node_slack_down_d_e": UnitSource("node", _U.NODE),
    "nodeGroup_flows_dt_g": UnitSource("group", _U.GROUP_MEMBERS),
    "nodeGroup_flows_d_g": UnitSource("group", _U.GROUP_MEMBERS),
    "nodeGroup_flows_dt_gpe": UnitSource("group", _U.GROUP_MEMBERS),
    "nodeGroup_flows_d_gpe": UnitSource("group", _U.GROUP_MEMBERS),
    "nodeGroup_gdt_p": UnitSource("group", _U.GROUP_MEMBERS, row=True),
    "nodeGroup_gd_p": UnitSource("group", _U.GROUP_MEMBERS, row=True),
    "nodeGroup_g_p": UnitSource("group", _U.GROUP_MEMBERS, row=True),
    "unit_VRE_potential_outputNode_dt_ee": UnitSource("sink", _U.NODE),
    "unit_VRE_potential_outputNode_d_ee": UnitSource("sink", _U.NODE),
    "unit_curtailment_outputNode_dt_ee": UnitSource("sink", _U.NODE),
    "unit_curtailment_outputNode_d_ee": UnitSource("sink", _U.NODE),
    "unit_ramp_inputs_dt_ee": UnitSource("source", _U.NODE),
    "unit_ramp_outputs_dt_ee": UnitSource("sink", _U.NODE),
    "unit_startup_energy_dt_ee": UnitSource("node", _U.NODE),
    "unit_startup_energy_d_ee": UnitSource("node", _U.NODE),
    "unit_capacity_ed_p": UnitSource("unit", _U.CAPACITY_SIDE, row=True),
    "connection_capacity_ed_p": UnitSource("connection", _U.CONNECTION_ENDS, row=True),
    "node_capacity_ed_p": UnitSource("node", _U.NODE, row=True),
    "dual_invest_effective_unit_d_e": UnitSource("entity", _U.CAPACITY_SIDE),
    "dual_invest_effective_connection_d_e": UnitSource("entity", _U.CONNECTION_ENDS),
    "dual_invest_effective_node_d_e": UnitSource("entity", _U.NODE),
    "flowGroup_gd_p": UnitSource("group", _U.FLOWGROUP_ARCS, row=True),
    "flowGroup_gd_t": UnitSource("group", _U.FLOWGROUP_ARCS, row=True),
}

#: Tables with a power or energy unit that stay in MW / MWh whatever their nodes' ``unit_of_measure``: reserves,
#: inertia, non-synchronous share and the capacity margin are power features (the input validators warn about a
#: member node that is not MWh).  The resolver must consult this set before it fills a template from a node or
#: group level (critique M1: several of them use templated transforms such as RATE).
DEFAULT_UNITS_ONLY: frozenset[str] = frozenset({
    "reserve_prices_dt_ppg",
    "process_reserve_upDown_node_dt_eppe",
    "process_reserve_average_d_eppe",
    "nodeGroup_inertia_dt_g",
    "nodeGroup_unit_node_inertia_dt_gee",
    "nodeGroup_inertia_largest_flow_dt_g",
    "nodeGroup_slack_capacity_margin_d_g",
    "nodeGroup_capacity_margin_d_g",
    "nodeGroup_capacity_margin_entity_d_g",
    "nodeGroup_slack_inertia_dt_g",
    "nodeGroup_slack_inertia_d_g",
    "nodeGroup_slack_nonsync_dt_g",
    "nodeGroup_slack_nonsync_d_g",
    "nodeGroup_slack_reserve_dt_eeg",
    "nodeGroup_slack_reserve_d_eeg",
    "nodeGroup_reserve_dt_eegp",
    "nodeGroup_reserve_d_eegp",
    "nodeGroup_reserve_provider_dt_eegee",
})


def _column_labels(columns) -> tuple[list[tuple], list | None]:
    """``(labels, level_names)`` of *columns*: every label as a tuple (one element per column level) and the
    level names when *columns* is a pandas index that names its levels."""
    names = getattr(columns, "names", None)
    labels = [c if isinstance(c, tuple) else (c,) for c in columns]
    if names is not None and all(n is None for n in names):
        names = None
    return labels, (list(names) if names is not None else None)


def _label_level(output_key: str, n_levels: int, level_names) -> int:
    """The position of the column level whose labels a mixed-unit table's map is keyed by.

    The map labels are on the table's innermost declared column level (``parameter`` / ``quantity`` in
    :data:`OUTPUT_DIMS`).  The level is found by its name when the columns name their levels, else by its
    position below an optional leading ``scenario`` level, so an entity or group that happens to be named like
    a map label is never read as one."""
    dims = OUTPUT_DIMS.get(output_key)
    declared = dims.columns if dims is not None else ()
    if level_names is not None and declared and declared[-1] in level_names:
        return list(level_names).index(declared[-1])
    return n_levels - 1


def _scenario_level(output_key: str, n_levels: int, level_names) -> int | None:
    """The position of the comparison's ``scenario`` column level, or ``None`` when there is none."""
    if level_names is not None:
        return list(level_names).index(SCENARIO_LEVEL) if SCENARIO_LEVEL in level_names else None
    dims = OUTPUT_DIMS.get(output_key)
    if dims is not None and dims.columns and n_levels == len(dims.columns) + 1:
        return 0
    return None


def _context_for(ctx, scenario) -> "UnitContext | None":
    """The :class:`UnitContext` of one column: *ctx* itself, or its entry for the column's scenario."""
    if ctx is None or isinstance(ctx, UnitContext):
        return ctx
    return ctx.get(scenario) if scenario is not None else None


def _any_node_units(ctx) -> bool:
    """Whether *ctx* (one context or one per scenario) has a node that is not MWh."""
    if ctx is None:
        return False
    if isinstance(ctx, UnitContext):
        return not ctx.is_default
    return any(c is not None and not c.is_default for c in ctx.values())


def _follows_nodes(output_key: str, tf: Transform, ctx: "UnitContext | None") -> bool:
    """Whether the unit of a column of *output_key* that underwent *tf* follows its nodes in *ctx*: the transform
    has a template, the table is in :data:`UNIT_SOURCE` and not in :data:`DEFAULT_UNITS_ONLY` (checked first:
    the power features stay MW whatever their nodes), and the context has a node that is not MWh.

    Also with every node MWh, a table of entities by class (``entity_annuity_d_p``: storage nodes, units and
    connections in one table) follows when the context knows the units and connections: a storage node's
    capacity is an amount, so its cost is per amount (``CUR/MWh``); a unit's or connection's capacity is a
    rate, so its cost is per rate (``CUR/MW``)."""
    if ctx is None or not tf.template or output_key in DEFAULT_UNITS_ONLY or output_key not in UNIT_SOURCE:
        return False
    if not ctx.is_default:
        return True
    return (bool(tf.node_template) and UNIT_SOURCE[output_key].kind is _U.ENTITY_BY_CLASS
            and bool(ctx.capacity_nodes or ctx.connection_nodes))


def entity_unit(tf: Transform, kind: UnitSourceKind, value, ctx: UnitContext) -> str:
    """The unit of a quantity that underwent *tf* for the level value *value* (a node, unit, connection or group,
    read as *kind* says), with the node units of *ctx*."""
    return entity_label(tf.template, kind, value, ctx, node_template=tf.node_template)


def entity_label(template: str, kind: UnitSourceKind, value, ctx: UnitContext, *, node_template: str = "") -> str:
    """*template* (a unit template, or a plot label with unit tokens) filled for the level value *value* (a node,
    unit, connection or group, read as *kind* says) with the node units of *ctx*.  *node_template* is the
    template of an ``entity`` level value that is a node (:attr:`Transform.node_template`; ``""`` =
    *template*).  The one rule of :func:`entity_unit` and the plot labels."""
    units = ctx.node_units
    name = str(value)
    if kind is _U.ENTITY_BY_CLASS:
        if name in ctx.connection_nodes:
            kind = _U.CONNECTION_ENDS
        elif name in ctx.capacity_nodes:
            kind = _U.CAPACITY_SIDE
        else:
            return uom.fill(node_template or template, uom.node_unit(name, units))
    if kind is _U.NODE:
        return uom.fill(template, uom.node_unit(name, units))
    if kind is _U.CAPACITY_SIDE:
        return uom.capacity_units(template, ctx.capacity_nodes.get(name, ()), units,
                                  shared=name in ctx.shared_capacity)
    nodes = {_U.CONNECTION_ENDS: ctx.connection_nodes, _U.GROUP_MEMBERS: ctx.group_nodes,
             _U.FLOWGROUP_ARCS: ctx.flowgroup_nodes}[kind].get(name, ())
    return uom.fill(template, uom.units_of(nodes, units))


def _unit_of(tf: Transform, ctx: "UnitContext | None", output_key: str = "", value=None) -> str:
    """The unit of a column of *output_key* that underwent *tf*, whose node-bearing level (:data:`UNIT_SOURCE`)
    has the value *value*.  With no context, a context whose nodes are all MWh, a transform without a template
    or a table that does not follow its nodes this is the transform's unit, so the default is byte-identical by
    construction."""
    if value is None or not _follows_nodes(output_key, tf, ctx):
        return tf.unit
    return entity_unit(tf, UNIT_SOURCE[output_key].kind, value, ctx)


def _level_position(output_key: str, level: str, n_levels: int, level_names) -> int | None:
    """The position of the column level *level* of *output_key* in a column label of *n_levels* levels (found
    by name when the columns name their levels, else by its declared position below an optional leading
    ``scenario`` level); ``None`` when it cannot be placed."""
    if level_names is not None:
        names = list(level_names)
        return names.index(level) if level in names else None
    dims = OUTPUT_DIMS.get(output_key)
    if dims is None or level not in dims.columns:
        return None
    if n_levels == len(dims.columns):
        return dims.columns.index(level)
    if n_levels == len(dims.columns) + 1:
        return dims.columns.index(level) + 1
    return None


def _column_transforms(output_key: str, columns, level_names=None):
    """``(column, label, transform or None, scenario)`` for every column of *output_key*: the transform the
    column underwent (``None`` for a dimension column, a membership set and a column absent from a mixed-unit
    map) and the value of its comparison ``scenario`` level (``None`` without one).  Empty for an undeclared
    table."""
    spec = OUTPUT_TRANSFORM.get(output_key)
    if spec is None:
        return [], None
    labels, names = _column_labels(columns)
    if level_names is not None:
        names = list(level_names)
    out = []
    for col, label in zip(columns, labels):
        n = len(label)
        scen_pos = _scenario_level(output_key, n, names)
        if isinstance(spec, dict):
            pos = _label_level(output_key, n, names)
            key = label[pos] if pos != scen_pos else None
            tf = spec.get(str(key), spec.get('*')) if key is not None else spec.get('*')
        else:
            tf = spec
        if tf is None or tf.semantics is Semantics.DIMENSION or is_dimension(str(col)):
            tf = None
        out.append((col, label, tf, label[scen_pos] if scen_pos is not None else None))
    return out, names


def _row_entities(index, level: str) -> list | None:
    """The distinct values of the row level *level* of *index*, in first-seen order (``None`` when *index* has
    no such level)."""
    names = list(getattr(index, "names", None) or [])
    if level not in names:
        return None
    values = index.get_level_values(level) if hasattr(index, "get_level_values") else index
    return list(dict.fromkeys(str(v) for v in values))


def row_units(output_key: str, index, columns, ctx=None, *, level_names=None) -> "dict[str, dict[str, str]] | None":
    """``{row entity: {column: unit}}`` of a row-keyed table whose unit varies by row (its node, unit,
    connection or group is a row level, :data:`UNIT_SOURCE` with ``row=True``), else ``None``.

    ``None`` also when no node in *ctx* (one context, or one per comparison scenario) is non-MWh and for the
    tables of :data:`DEFAULT_UNITS_ONLY`, so the default outputs do not change.  The keys are the row level's
    values (each entity once, never one entry per row) and ``str`` of each measure column; each column's own
    template is filled for the entity, so a mixed-column table (``nodeGroup_gdt_p``: m3, a ratio, m3/a) gets one
    unit per column.  A column that does not follow its nodes keeps its unit (``""`` for a unitless one)."""
    source = UNIT_SOURCE.get(output_key)
    if source is None or not source.row or output_key in DEFAULT_UNITS_ONLY:
        return None
    if not _any_node_units(ctx) and not _any_footer_rows(ctx, output_key):
        return None
    entities = _row_entities(index, source.level)
    if entities is None:
        return None
    cols, names = _column_transforms(output_key, columns, level_names)
    if not cols:
        return None
    measures = []
    for col, label, tf, scen in cols:
        if tf is None:
            continue
        c = _context_for(ctx, scen)
        footer = _footer_of(c, output_key)
        written = footer[1] if footer is not None else None
        leaf = _leaf_name(label, _scenario_level(output_key, len(label), names))
        measures.append((str(col), tf, c, written, leaf))
    out: dict[str, dict[str, str]] = {}
    for entity in entities:
        units: dict[str, str] = {}
        for name, tf, c, written, leaf in measures:
            if written is not None and leaf in written.get(entity, {}):
                units[name] = written[entity][leaf]
            elif _follows_nodes(output_key, tf, c):
                units[name] = entity_unit(tf, source.kind, entity, c)
            else:
                units[name] = tf.unit
        out[entity] = units
    return out


def _any_footer_rows(ctx, output_key: str) -> bool:
    """Whether a footer of *ctx* (one context, or one per scenario) holds row units of *output_key*."""
    if ctx is None:
        return False
    contexts = [ctx] if isinstance(ctx, UnitContext) else list(ctx.values())
    for c in contexts:
        footer = _footer_of(c, output_key)
        if footer is not None and footer[1]:
            return True
    return False


def resolve_columns(
    output_key: str, columns, *, level_names=None,
    ctx: "UnitContext | Mapping[str, UnitContext] | None" = None, rows=None,
) -> list[ColumnMeta] | None:
    """One :class:`ColumnMeta` per column of the result table *output_key*, in column order.

    *columns* is ``df.columns`` (single- or multi-level, with or without the comparison's top ``scenario``
    level) or a list of labels.  A mixed-unit table's per-column map is looked up on the label of its declared
    label level (see :func:`_label_level`), never on the text of the whole tuple.  *level_names* overrides the
    level names of *columns*.  *ctx* is one :class:`UnitContext` or one per scenario (keyed by the
    ``scenario`` level's value): a column whose node-bearing column level names a node of another unit gets
    that unit (:data:`UNIT_SOURCE`).  For a row-keyed table, *rows* (``df.index``) gives each column the units
    of its rows joined (:func:`row_units`, :func:`summarise_units`); without *rows* it keeps the default unit.
    ``ColumnMeta.name`` is ``str`` of the column label, the key of the parquet footer.  Returns ``None`` when the
    output is not declared in :data:`OUTPUT_TRANSFORM`."""
    if output_key not in OUTPUT_TRANSFORM:
        return None
    cols, names = _column_transforms(output_key, columns, level_names)
    docs = DOCS_ANCHOR.get(output_key, "")
    source = UNIT_SOURCE.get(output_key)
    by_row = (row_units(output_key, rows, columns, ctx, level_names=level_names)
              if rows is not None and source is not None and source.row else None)
    out: list[ColumnMeta] = []
    for col, label, tf, scenario in cols:
        name = str(col)
        if tf is None:
            out.append(ColumnMeta(name, "", Semantics.DIMENSION, "Index / key dimension."))
            continue
        c = _context_for(ctx, scenario)
        footer = _footer_of(c, output_key)
        written = (footer[0].get(_leaf_name(label, _scenario_level(output_key, len(label), names)))
                   if footer is not None else None)
        if by_row is not None:
            unit = summarise_units(units[name] for units in by_row.values()) or tf.unit
        elif written is not None:
            unit = written
        elif source is not None and not source.row and _follows_nodes(output_key, tf, c):
            pos = _level_position(output_key, source.level, len(label), names)
            unit = _unit_of(tf, c, output_key, label[pos] if pos is not None else None)
        else:
            unit = tf.unit
        out.append(ColumnMeta(
            name=name,
            unit=unit,
            semantics=tf.semantics,
            tooltip=tf.tooltip,
            long=tf.description,
            formula=FORMULA_OVERRIDE.get((output_key, name), ""),
            docs=docs,
        ))
    return out


def column_units(output_key: str, columns, *, level_names=None, ctx=None) -> list[str]:
    """The unit of every column of *output_key*, in column order (``""`` for a dimension column, a unitless
    measure and an undeclared table).  See :func:`resolve_columns`."""
    resolved = resolve_columns(output_key, columns, level_names=level_names, ctx=ctx)
    if resolved is None:
        return [""] * len(list(columns))
    return [m.unit for m in resolved]


def row_unit_label(units: Mapping[str, str]) -> str:
    """The text of one row's units (a value of :func:`row_units`): its measure columns' units joined
    (:func:`summarise_units`; unitless columns left out)."""
    return summarise_units(units.values())


def display_unit(meta: ColumnMeta) -> str:
    """The unit of a measure column as shown to people: its unit, or ``per unit`` for a unitless measure."""
    if meta.semantics is Semantics.DIMENSION:
        return ""
    return meta.unit or UNITLESS


def summarise_units(units) -> str:
    """One text for a set of units: the unit when all agree, else the distinct units sorted and joined with
    ``"; "``; ``""`` when there is none.  A unit that is itself a ``"; "`` list (a per-arc capacity label,
    ``MW; MW_th``) counts as its parts.  The single reducer of the ``Units`` sheet, the results-database
    descriptions, the GUI hover and the plot labels."""
    distinct = sorted({part for u in units if u for part in u.split("; ")})
    return "; ".join(distinct)


def derive_column_meta(
    output_key: str, columns, *, ctx: "UnitContext | None" = None, rows=None,
) -> dict[str, ColumnMeta] | None:
    """Derive ``{column_name: ColumnMeta}`` for one output.

    ``columns`` is ``df.columns`` before the scenario level is prepended (the keys are ``str`` of each
    column label, the parquet-footer keys).  Dimension columns get ``Semantics.DIMENSION``; measure columns
    get the output's declared transform, for a mixed-unit table the entry of the column's label on its
    declared label level (see :func:`resolve_columns`, which also says what *ctx* and *rows* do).  Returns
    ``None`` when the output is not (yet) declared in :data:`OUTPUT_TRANSFORM`, so callers can treat absence as
    "no metadata".
    """
    resolved = resolve_columns(output_key, columns, ctx=ctx, rows=rows)
    if resolved is None:
        return None
    return {m.name: m for m in resolved}
