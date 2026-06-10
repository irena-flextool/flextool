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
    # CO2
    "CO2_d_g": EMISSION_ANNUAL,
    "process_co2_d_eee": EMISSION_ANNUAL,
    # Node / group flows
    # (node_d_ep is a mixed flows+inflow table — needs per-column transforms,
    #  deferred; it stays in the coverage allowlist for now.)
    "nodeGroup_flows_d_g": PREENERGY_ANNUAL,
    # Start-ups
    "unit_startup_d_e": COUNT_ANNUAL,
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
        if is_dimension(name):
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
