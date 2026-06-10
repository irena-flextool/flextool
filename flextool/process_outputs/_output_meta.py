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

from dataclasses import dataclass, field
from enum import Enum


class Semantics(str, Enum):
    """The kind of quantity a column holds — the field that disambiguates
    a per-timestep rate from a full-year-annualized total."""

    INSTANTANEOUS = "instantaneous"  # a rate at the timestep (MW). Integrate for energy.
    PER_STEP = "per_step"            # already integrated over the step's duration (cost_dt).
    PER_PERIOD = "per_period"        # summed over the period, as-sampled (NOT scaled to a year).
    ANNUALIZED = "annualized"        # scaled to a full-year equivalent (÷ period_share_of_year).
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
    tooltip: str
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
EMISSION_ANNUAL = Transform(
    "t/a", Semantics.ANNUALIZED,
    "Emissions, scaled from the sampled timeline to a full-year equivalent.",
    algebra="t_step · timestep_weight / period_share",
)
COUNT_ANNUAL = Transform(
    "1/a", Semantics.COUNT,
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
CAPACITY_MW = Transform(
    "MW", Semantics.LEVEL,
    "Installed/usable capacity (a stock that holds over the period, not a flow).",
    algebra="existing + invested − divested",
)
CAPACITY_MWH = Transform(
    "MWh", Semantics.LEVEL,
    "Storage energy capacity (a stock).",
    algebra="existing + invested − divested",
)
STORAGE_STATE = Transform(
    "MWh", Semantics.LEVEL,
    "Stored energy at the timestep (state of charge).",
    algebra="v_state · unitsize",
)
PRICE_ENERGY = Transform(
    "CUR/MWh", Semantics.PRICE,
    "Nodal energy price (shadow price of the node balance constraint).",
    algebra="dual(node_balance)",
)
EMISSION_MT = Transform(
    "Mt/a", Semantics.ANNUALIZED,
    "System CO2 emissions, full-year equivalent (×years_represented).",
    algebra="t · years_represented / 1e6",
)
FACTOR = Transform(
    "", Semantics.RATIO,
    "Dimensionless discount / inflation factor.",
)
ANNUITY = Transform(
    "1/a", Semantics.RATIO,
    "Annuity factor — annual cost per unit of overnight investment.",
)
YEARS = Transform(
    "a", Semantics.COUNT,
    "Calendar years the period represents (annualisation weight).",
)
PRICE_RESERVE = Transform(
    "CUR/MW", Semantics.PRICE,
    "Reserve price (dual of the reserve balance) — CUR per MW of reserve held "
    "(= CUR/MWh held for one hour).",
    algebra="dual(reserve_balance)",
)
PRICE_CO2 = Transform(
    "CUR/t", Semantics.PRICE,
    "CO2 price (shadow price of the CO2 cap).",
    algebra="dual(co2_cap) / 1000",
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
    "Average reserve held over the period.",
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
    tooltip: str
    formula: str = ""
    docs: str = ""

    def as_dict(self) -> dict[str, str]:
        return {
            "unit": self.unit,
            "semantics": self.semantics.value,
            "tooltip": self.tooltip,
            "formula": self.formula,
            "docs": self.docs,
        }


# ── Per-output declaration ──────────────────────────────────────────────────
# Map each output key to the transform its measure columns underwent.  One line
# per output; the per-column expansion + tooltips are generated.  Seeded with
# the high-traffic outputs; extend as coverage grows (the CI ratchet enforces
# that no NEW output lands here undocumented).
OUTPUT_TRANSFORM: dict[str, Transform] = {
    # Costs
    "costs_dt_p": MONEY_PERSTEP,
    "annualized_costs_d_p": MONEY_ANNUAL,
    "costs_discounted_d_p": MONEY_DISCOUNTED,
    # Unit / connection flows
    "unit_outputNode_dt_ee": RATE,
    "unit_inputNode_dt_ee": RATE,
    "unit_outputNode_d_ee": ENERGY_ANNUAL,
    "unit_inputNode_d_ee": ENERGY_ANNUAL,
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
    # (node_d_ep / node_dt_ep / nodeGroup_gd_p / nodeGroup_gdt_p are MIXED
    #  flows+inflow+share tables — they need per-column transforms, deferred;
    #  they stay in the coverage allowlist for now.)
    "nodeGroup_flows_d_g": PREENERGY_ANNUAL,
    "nodeGroup_flows_d_gpe": PREENERGY_ANNUAL,
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
    "entity_annuity_d_p": ANNUITY,
    "years_represented__d": YEARS,
    # Start-ups
    "unit_startup_d_e": COUNT_ANNUAL,
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
    "unit_online_dt_e": ONLINE_COUNT,
    "unit_online_average_d_e": ONLINE_AVG,
    "flowGroup_gd_t": RATE,
    # Pure membership / index sets (no measure columns)
    "group_node": DIMENSION_TABLE,
    "group_process": DIMENSION_TABLE,
    "group_process_node": DIMENSION_TABLE,
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
    "annualized_costs_d_p": "outputs.html#annualized-costs",
    "costs_dt_p": "outputs.html#per-timestep-costs",
}


def is_dimension(column: str) -> bool:
    """True if ``column`` is an index/key dimension rather than a measure."""
    return column in DIMENSION_COLUMNS


def derive_column_meta(
    output_key: str, columns,
) -> dict[str, ColumnMeta] | None:
    """Derive ``{column_name: ColumnMeta}`` for one output.

    ``columns`` is the iterable of *leaf* column names (e.g. cost categories),
    typically ``df.columns`` before any scenario-level wrapping.  Dimension
    columns get ``Semantics.DIMENSION``; measure columns get the output's
    declared transform.  Returns ``None`` when the output is not (yet) declared
    in :data:`OUTPUT_TRANSFORM`, so callers can treat absence as "no metadata".
    """
    transform = OUTPUT_TRANSFORM.get(output_key)
    if transform is None:
        return None
    docs = DOCS_ANCHOR.get(output_key, "")
    out: dict[str, ColumnMeta] = {}
    for col in columns:
        name = str(col)
        # A pure membership/index set: every column is a dimension.
        if transform.semantics is Semantics.DIMENSION or is_dimension(name):
            out[name] = ColumnMeta(name, "", Semantics.DIMENSION,
                                   "Index / key dimension.")
            continue
        out[name] = ColumnMeta(
            name=name,
            unit=transform.unit,
            semantics=transform.semantics,
            tooltip=transform.tooltip,
            formula=FORMULA_OVERRIDE.get((output_key, name), ""),
            docs=docs,
        )
    return out
