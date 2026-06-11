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
ANNUITY = Transform(
    "CUR/MW/a", Semantics.ANNUALIZED,
    "Annualised investment cost per MW of capacity — the cost charged for "
    "each year of the entity's economic lifetime.",
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
ANGLE = Transform(
    "rad", Semantics.LEVEL,
    "DC voltage angle (Bθ formulation; reference node pinned to 0). Radians — "
    "flow = susceptance · Δangle with susceptance = base_MVA / reactance.",
)
SLACK_CAP_MARGIN = Transform(
    "MW", Semantics.LEVEL,
    "Capacity-margin slack — per-period MW shortfall (not annualized).",
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
    # Raw inflow level at the timestep — no step_duration applied (an MW
    # level, like nodeGroup/node dt flows), unlike the slack/VRE columns
    # above which are integrated to MWh/step.
    '5. Timestep inflow': RATE,
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
    # dt group/node flows show the un-integrated MW level (no step_duration —
    # relatable to installed capacity), per maintainer convention.  Only
    # sum_flow_t (gpe) integrates the flow columns to MWh/step.
    "nodeGroup_flows_dt_g": RATE,
    "nodeGroup_flows_dt_gpe": ENERGY_PERSTEP,
    "node_inflow__dt": RATE,
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
    "dc_angle_dt_e": ANGLE,
    "dc_angle_diff_dt_e": ANGLE,
    "unit_online_dt_e": ONLINE_COUNT,
    "unit_online_average_d_e": ONLINE_AVG,
    "flowGroup_gd_t": RATE,
    "nodeGroup_slack_reserve_dt_eeg": RATE,
    "nodeGroup_slack_reserve_d_eeg": ENERGY_ANNUAL,
    "nodeGroup_slack_capacity_margin_d_g": SLACK_CAP_MARGIN,
    "nodeGroup_total_inflow": PREENERGY_ANNUAL,
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
