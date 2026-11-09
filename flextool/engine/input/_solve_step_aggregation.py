"""One aggregation rule for a solve's ``new_stepduration``: how a time series becomes one value per solve step.

A solve with ``new_stepduration`` combines the time steps of its timeline into longer steps; each new step is
labelled by its first original (base) step.  Every time-series input then needs one value per new step.  The rule
depends on what the value is, so it is a property of the parameter: :data:`TIME_AGGREGATION` maps each
``(entity_class, parameter_name)`` that can vary in time to its :class:`Agg` kind.

* ``MEAN`` -- a rate or a share per hour (prices, costs, limits, availability, min load, penalties, reserves,
  self-discharge, profiles, time-step weights): the duration-weighted mean ``Σ v·Δ / Σ Δ`` over the base steps.
* ``HARMONIC`` -- an efficiency: the duration-weighted mean of ``1/v`` (the fuel per unit of output, the
  ``p_slope``), returned as its reciprocal.  The fuel use of a flow that is constant over the step is then exact.
* ``SUM`` -- an energy per base step (``inflow``, ``energy_margin_adder``): ``Σ v``; a constant counts once per
  base step (the consumers scale constants themselves).
* ``LAST`` -- a state at the end of a step (``storage_state_reference_value``): the value of the last base step
  (in timeline order) that has one.

Numbers: when every base value of a new step is equal, a MEAN, HARMONIC or LAST step returns the value itself,
without arithmetic (a constant and its equivalent map give the same model); a SUM step always sums.  Otherwise ``MEAN`` and ``HARMONIC`` are rounded to 6
decimals (``HARMONIC`` rounds the mean of ``1/v``, the linear quantity, then takes the reciprocal); equal base
durations use ``Σ v / n``.  The same function serves the provider's CSV frames
(``TimelineConfig.create_averaged_timeseries``) and the DB-direct readers, so both paths give identical bits.

A new step whose base steps have values only for some of its base steps takes the steps it has (one warning per
parameter and run); the defaults that a reader adds later for the missing hours are not part of the step's value.

:data:`NOT_TIME_SERIES` lists the schema's map parameters that have no time axis (period maps, coefficient maps,
the solve / timeline / timeset maps); every map parameter of the schema is in exactly one of the two
(``tests/engine/test_time_aggregation_registry.py``).  A new time parameter needs an entry here.

The base steps of the active solve (``solve_data/new_step_base_steps``: ``time``, ``new_step``, ``value`` = the
number of base steps of the new step, ``duration``, ``pos`` = timeline position) are armed for the DB-direct
readers by ``load_flextool`` (:func:`activate_solve_step_aggregation`); without ``new_stepduration`` the frame is
empty and nothing is aggregated.
"""
from __future__ import annotations

import contextvars
import logging
from enum import Enum
from typing import Iterable

import polars as pl

from flextool.engine.input._solve_state import FlexToolConfigError

_LOGGER = logging.getLogger(__name__)


class Agg(Enum):
    """How the base steps of a solve step combine into its value (module docstring)."""

    MEAN = "mean"
    HARMONIC = "harmonic"
    SUM = "sum"
    LAST = "last"


TIME_AGGREGATION: dict[tuple[str, str], Agg] = {
    ("commodity", "price"): Agg.MEAN,
    ("connection", "availability"): Agg.MEAN,
    ("connection", "efficiency"): Agg.HARMONIC,
    ("connection", "other_operational_cost"): Agg.MEAN,
    ("flowGroup", "max_instant_flow"): Agg.MEAN,
    ("flowGroup", "min_instant_flow"): Agg.MEAN,
    ("group", "co2_price"): Agg.MEAN,
    ("group", "inertia_limit"): Agg.MEAN,
    ("group", "non_synchronous_limit"): Agg.MEAN,
    ("node", "availability"): Agg.MEAN,
    ("node", "energy_margin_adder"): Agg.SUM,
    ("node", "inflow"): Agg.SUM,
    ("node", "penalty_down"): Agg.MEAN,
    ("node", "penalty_up"): Agg.MEAN,
    # Exact would be the mean of ln(1 + loss) (the loss compounds per hour); the mean differs by O(loss²).
    ("node", "self_discharge_loss"): Agg.MEAN,
    ("node", "storage_state_reference_value"): Agg.LAST,
    ("profile", "profile"): Agg.MEAN,
    ("reserve__upDown__group", "reservation"): Agg.MEAN,
    ("timeset", "timeset_weights"): Agg.MEAN,
    ("unit", "availability"): Agg.MEAN,
    ("unit", "efficiency"): Agg.HARMONIC,
    # HARMONIC like efficiency: the min-load section is linear in 1/efficiency and 1/efficiency_at_min_load, so
    # with a min_load that is constant within the step both give the step mean of the section and the slope.
    ("unit", "efficiency_at_min_load"): Agg.HARMONIC,
    ("unit", "min_load"): Agg.MEAN,
    ("unit__inputNode", "other_operational_cost"): Agg.MEAN,
    ("unit__outputNode", "other_operational_cost"): Agg.MEAN,
}

# Schema map parameters without a time axis.  ``peak_inflow`` is a scalar per original time step, scaled by the
# number of base steps in the inflow scaling (``emit/_inflow_scaling_checks.py``), not a time series.
NOT_TIME_SERIES: frozenset[tuple[str, str]] = frozenset({
    ("commodity", "price_ladder_annual"),
    ("commodity", "price_ladder_cumulative"),
    ("connection", "constraint_capacity_coeff"),
    ("connection", "constraint_cumulative_pre_built_capacity_coeff"),
    ("connection", "constraint_in_construction_capacity_coeff"),
    ("connection", "constraint_invested_capacity_coeff"),
    ("connection", "constraint_online_coeff"),
    ("connection", "constraint_retired_capacity_coeff"),
    ("connection", "construction_lead_time"),
    ("connection", "cumulative_max_capacity"),
    ("connection", "cumulative_min_capacity"),
    ("connection", "delay"),
    ("connection", "discount_rate"),
    ("connection", "existing"),
    ("connection", "fixed_cost"),
    ("connection", "invest_cost"),
    ("connection", "invest_max_period"),
    ("connection", "invest_min_period"),
    ("connection", "lifetime"),
    ("connection", "retire_max_period"),
    ("connection", "retire_min_period"),
    ("connection", "salvage_value"),
    ("connection", "startup_cost"),
    ("connection__node", "constraint_flow_coeff"),
    ("constraint", "constant"),
    ("flowGroup", "max_cumulative_flow"),
    ("flowGroup", "min_cumulative_flow"),
    ("group", "capacity_margin"),
    ("group", "co2_max_period"),
    ("group", "cumulative_max_capacity"),
    ("group", "cumulative_min_capacity"),
    ("group", "invest_max_period"),
    ("group", "invest_min_period"),
    ("group", "penalty_capacity_margin"),
    ("group", "penalty_inertia"),
    ("group", "penalty_non_synchronous"),
    ("group", "retire_max_period"),
    ("group", "retire_min_period"),
    ("node", "annual_flow"),
    ("node", "constraint_capacity_coeff"),
    ("node", "constraint_cumulative_pre_built_capacity_coeff"),
    ("node", "constraint_in_construction_capacity_coeff"),
    ("node", "constraint_invested_capacity_coeff"),
    ("node", "constraint_retired_capacity_coeff"),
    ("node", "constraint_state_coeff"),
    ("node", "construction_lead_time"),
    ("node", "cumulative_max_capacity"),
    ("node", "cumulative_min_capacity"),
    ("node", "discount_rate"),
    ("node", "existing"),
    ("node", "fixed_cost"),
    ("node", "invest_cost"),
    ("node", "invest_forced"),
    ("node", "invest_max_period"),
    ("node", "invest_min_period"),
    ("node", "lifetime"),
    ("node", "peak_inflow"),
    ("node", "retire_max_period"),
    ("node", "retire_min_period"),
    ("node", "salvage_value"),
    ("node", "storage_state_reference_price"),
    ("solve", "fix_storage_periods"),
    ("solve", "invest_periods"),
    ("solve", "period_timeset"),
    ("solve", "realized_invest_periods"),
    ("solve", "realized_periods"),
    ("solve", "solver_arguments"),
    ("solve", "stochastic_branches"),
    ("solve", "years_represented"),
    ("timeline", "timestep_duration"),
    ("timeset", "representative_period_weights"),
    ("timeset", "timeset_duration"),
    ("unit", "constraint_capacity_coeff"),
    ("unit", "constraint_cumulative_pre_built_capacity_coeff"),
    ("unit", "constraint_in_construction_capacity_coeff"),
    ("unit", "constraint_invested_capacity_coeff"),
    ("unit", "constraint_online_coeff"),
    ("unit", "constraint_retired_capacity_coeff"),
    ("unit", "construction_lead_time"),
    ("unit", "cumulative_max_capacity"),
    ("unit", "cumulative_min_capacity"),
    ("unit", "delay"),
    ("unit", "discount_rate"),
    ("unit", "existing"),
    ("unit", "fixed_cost"),
    ("unit", "invest_cost"),
    ("unit", "invest_max_period"),
    ("unit", "invest_min_period"),
    ("unit", "lifetime"),
    ("unit", "retire_max_period"),
    ("unit", "retire_min_period"),
    ("unit", "salvage_value"),
    ("unit", "startup_cost"),
    ("unit__inputNode", "constraint_flow_coeff"),
    ("unit__outputNode", "constraint_flow_coeff"),
})

# Parameters whose aggregation is exact only while the value is constant within a solve step: a varying value
# gets one warning per run.
_APPROXIMATE_WHEN_VARYING: dict[tuple[str, str], str] = {
    ("unit", "min_load"): (
        "unit.min_load varies within a solve step of new_stepduration (units: {entities}). The step uses the "
        "mean min_load and the harmonic means of efficiency and efficiency_at_min_load, so the "
        "min_load_efficiency curve (slope and section) of those steps is an approximation of the step mean of "
        "the hourly curves; it is exact for a min_load that is constant within each step."),
}

BASE_STEP_COLUMNS: tuple[str, ...] = ("time", "new_step", "value", "duration", "pos")


def kind_of(key: tuple[str, str]) -> Agg:
    """The :class:`Agg` of ``key``; :class:`FlexToolConfigError` when the registry has no entry."""
    kind = TIME_AGGREGATION.get(key)
    if kind is None:
        raise FlexToolConfigError(
            f"{key[0]}.{key[1]}: a solve with new_stepduration needs to know how this time series combines over "
            f"the time steps of a new step, but the parameter is not in TIME_AGGREGATION "
            f"(flextool/engine/input/_solve_step_aggregation.py).")
    return kind


def combine(kind: Agg, values: list[float], durations: list[float]) -> float:
    """The value of one solve step from the ``values`` of its base steps (timeline order) and their ``durations``.

        SUM               → Σ v                          (an energy per base step: never the value itself)
        all values equal  → the value itself             (MEAN, HARMONIC, LAST)
        LAST              → values[-1]
        MEAN              → round(Σ v / n, 6)            equal durations
                            round(Σ v·Δ / Σ Δ, 6)        otherwise
        HARMONIC          → 1 / round(Σ (1/v) / n, 6)    equal durations
                            1 / round(Σ Δ/v / Σ Δ, 6)    otherwise
    """
    if kind is Agg.SUM:
        return sum(values)
    first = values[0]
    if all(v == first for v in values):
        return first
    if kind is Agg.LAST:
        return values[-1]
    equal = all(d == durations[0] for d in durations)
    if kind is Agg.MEAN:
        if equal:
            return round(sum(values) / len(values), 6)
        return round(sum(v * d for v, d in zip(values, durations)) / sum(durations), 6)
    if any(v <= 0.0 for v in values):
        raise ValueError("a harmonic mean needs positive values")
    if equal:
        return 1.0 / round(sum(1.0 / v for v in values) / len(values), 6)
    return 1.0 / round(sum(d / v for v, d in zip(values, durations)) / sum(durations), 6)


def _what(key: tuple[str, str]) -> str:
    return f"{key[0]}.{key[1]}"


def _note_once(notes: "set | None", tag: tuple, message: str) -> None:
    """One WARNING per ``tag`` within ``notes`` (the run's dedupe set; always logged without one)."""
    if notes is not None:
        if tag in notes:
            return
        notes.add(tag)
    _LOGGER.warning(message)


def aggregate_to_solve_steps(
    frame: pl.DataFrame,
    kind: "Agg | str",
    group_cols: "Iterable[str]",
    base_steps: pl.DataFrame,
    *,
    time_col: str = "t",
    value_col: str = "value",
    keys: "dict[str, tuple[str, str]] | tuple[str, str] | None" = None,
    kind_values: "dict[str, Agg] | None" = None,
    notes: "set | None" = None,
    report_changes: bool = True,
) -> pl.DataFrame:
    """``frame`` (``group_cols``, ``time_col``, ``value_col``, …) with one row per (group, solve step).

        out[k, s] = combine(kind, [v[k, t] : t ∈ base(s), v[k, t] given], [Δt : same t])
        ∀ group k, solve step s with at least one given value;  t ∉ any base(s) is dropped

    ``base_steps`` is ``solve_data/new_step_base_steps`` (:data:`BASE_STEP_COLUMNS`).  ``kind`` is one
    :class:`Agg` for the whole frame, or the name of a column of ``frame`` holding a kind name per row (a frame of
    several parameters, constant within each group); ``kind_values`` maps those names to kinds.  ``keys`` names
    the ``(class, parameter)`` of the rows for the messages (one key, or per kind-column value).  Null values are
    dropped.  The output keeps ``group_cols`` (their dtypes), ``time_col`` (Utf8, the solve step) and
    ``value_col`` (Float64), in the order of each group's first row in ``frame``.

    ``report_changes``: name the entities whose step value differs from the value of the step's first base step
    (what earlier releases used) in the behaviour-change notice ``solve_step_series_combined``; off for the
    series that earlier releases already combined (the time-map profile and inflow, the energy-margin adder,
    which has its own notice).
    """
    group_cols = list(group_cols)
    per_row = isinstance(kind, str)
    kind_col = kind if per_row else None
    extra = [kind_col] if per_row and kind_col not in group_cols else []
    base = base_steps.select(
        pl.col("time").cast(pl.Utf8).alias("__t"),
        pl.col("new_step").cast(pl.Utf8).alias("__s"),
        pl.col("value").cast(pl.Float64).alias("__n"),
        pl.col("duration").cast(pl.Float64).alias("__dur"),
        pl.col("pos").cast(pl.Int64).alias("__pos"))
    rows = (frame.with_row_index("__row")
            .with_columns(pl.col(time_col).cast(pl.Utf8).alias("__t"),
                          pl.col(value_col).cast(pl.Float64, strict=False).alias("__v"))
            .filter(pl.col("__v").is_not_null())
            .join(base, on="__t", how="inner")
            .sort("__pos", maintain_order=True))
    by = [*group_cols, *extra, "__s"]
    grouped = (rows.group_by(by, maintain_order=True)
               .agg(pl.col("__row").min(), pl.col("__v"), pl.col("__dur"), pl.col("__n").first())
               .sort("__row"))
    if per_row:
        kinds = [kind_values[k] for k in grouped.get_column(kind_col).to_list()]
        labels = grouped.get_column(kind_col).to_list()
    else:
        kinds = [kind] * grouped.height
        labels = [None] * grouped.height
    vals = grouped.get_column("__v").to_list()
    durs = grouped.get_column("__dur").to_list()
    counts = grouped.get_column("__n").to_list()

    def key_of(label) -> "tuple[str, str] | None":
        if isinstance(keys, dict):
            return keys.get(label)
        return keys

    out_values: list[float] = []
    partial: dict = {}
    varying: dict = {}
    changed: dict = {}
    for i, (k, vs, ds, n) in enumerate(zip(kinds, vals, durs, counts)):
        try:
            out_values.append(combine(k, vs, ds))
            if out_values[-1] != vs[0]:
                changed.setdefault(labels[i], i)
        except ValueError as exc:
            key = key_of(labels[i])
            raise FlexToolConfigError(
                f"{_what(key) if key else 'A time series'}: {exc}; a value of 0 or less cannot be averaged over "
                f"the time steps of a new_stepduration step.") from None
        if k is not Agg.LAST and len(vs) < n:
            partial.setdefault(labels[i], []).append(i)
        key = key_of(labels[i])
        if key in _APPROXIMATE_WHEN_VARYING and any(v != vs[0] for v in vs):
            varying.setdefault(key, []).append(i)
    entity_cols = [c for c in group_cols if c not in (kind_col,)]
    for label, idx in partial.items():
        key = key_of(label)
        name = _what(key) if key else "a time series"
        sample = grouped.select(entity_cols).slice(idx[0], 1).row(0) if entity_cols else ()
        _note_once(notes, ("partial", key or label), (
            f"{name}: a solve step of new_stepduration has values for only some of its time steps "
            f"(e.g. {', '.join(str(x) for x in sample if x is not None)} at {grouped.get_column('__s')[idx[0]]}); "
            f"the step takes the time steps that have a value (the default of the other time steps is not part "
            f"of it)."))
    for key, idx in varying.items():
        names = sorted({str(grouped.get_column(entity_cols[0])[i]) for i in idx}) if entity_cols else []
        _note_once(notes, ("varying", key), _APPROXIMATE_WHEN_VARYING[key].format(entities=", ".join(names)))
    if report_changes and changed:
        _report_changes(changed, grouped, entity_cols, kinds, vals, out_values, labels, key_of)
    out = grouped.select(*group_cols, pl.col("__s").alias(time_col)).with_columns(
        pl.Series(value_col, out_values, dtype=pl.Float64))
    return out


def _report_changes(changed: dict, grouped: pl.DataFrame, entity_cols: list[str], kinds: list, vals: list,
                    out_values: list[float], labels: list, key_of) -> None:
    """The behaviour-change notice ``solve_step_series_combined`` (printed only while the input database carries
    its marker): the parameters and entities whose solve-step value is no longer the value of the step's first
    base step."""
    from flextool.engine import _behaviour_changes

    if not _behaviour_changes.has_marker("solve_step_series_combined"):
        return
    first_col = grouped.get_column(entity_cols[0]).cast(pl.Utf8).to_list() if entity_cols else None
    names: list[str] = []
    for i, (vs, v) in enumerate(zip(vals, out_values)):
        if v == vs[0]:
            continue
        key = key_of(labels[i])
        what = _what(key) if key else "a time series"
        names.append(f"{what} of {first_col[i]}" if first_col is not None else what)
    _behaviour_changes.notice("solve_step_series_combined", "parameters", sorted(dict.fromkeys(names)))


# ---------------------------------------------------------------------------
# The active solve's base steps (armed by load_flextool for the DB-direct readers)


class _SolveSteps:
    def __init__(self, base_steps: "pl.DataFrame | None", notes: "set | None") -> None:
        self.base_steps = base_steps if base_steps is not None and base_steps.height > 0 else None
        self.notes: set = notes if notes is not None else set()


_SOLVE_STEPS: "contextvars.ContextVar[_SolveSteps | None]" = contextvars.ContextVar(
    "flextool_solve_step_aggregation", default=None)


def activate_solve_step_aggregation(base_steps: "pl.DataFrame | None", notes: "set | None" = None,
                                    ) -> "contextvars.Token":
    """Arm ``base_steps`` (``solve_data/new_step_base_steps`` of the solve being loaded; None or empty: the solve
    has no ``new_stepduration``) for :func:`aggregate_if_active`; returns the token for
    :func:`deactivate_solve_step_aggregation`.  ``notes``: the run's warning-dedupe set."""
    return _SOLVE_STEPS.set(_SolveSteps(base_steps, notes))


def deactivate_solve_step_aggregation(token: "contextvars.Token") -> None:
    """Restore the state before the matching activation."""
    _SOLVE_STEPS.reset(token)


def active_base_steps() -> "pl.DataFrame | None":
    """The armed base steps, or None when no solve step combines several base steps."""
    scope = _SOLVE_STEPS.get()
    return None if scope is None else scope.base_steps


def aggregate_if_active(frame: pl.DataFrame, key: "tuple[str, str] | None", *, time_col: str = "t",
                        value_col: str = "value", allow_sum: bool = False) -> pl.DataFrame:
    """``frame`` aggregated to the armed solve steps (:func:`aggregate_to_solve_steps`, grouped by every column
    other than ``time_col`` and ``value_col``); ``frame`` itself when nothing is armed or it has no time rows.

    Rows with a null ``time_col`` (a period or constant default of a mixed frame) pass through unchanged.  A
    ``key`` without a :data:`TIME_AGGREGATION` entry raises.  A SUM key raises unless ``allow_sum``: its constant
    and period rows would need the base-step count, which the caller has to apply.
    """
    base = active_base_steps()
    if base is None or time_col not in frame.columns or frame.height == 0:
        return frame
    timed = frame.filter(pl.col(time_col).is_not_null())
    if timed.height == 0:
        return frame
    if key is None:
        raise FlexToolConfigError(
            "A time series without a parameter key reached the solve-step aggregation of new_stepduration.")
    kind = kind_of(key)
    if kind is Agg.SUM and not allow_sum:
        raise FlexToolConfigError(
            f"{_what(key)}: an energy per time step (SUM) is aggregated by its own reader, which also counts its "
            f"constant values once per time step; it cannot go through the generic broadcast.")
    scope = _SOLVE_STEPS.get()
    group_cols = [c for c in frame.columns if c not in (time_col, value_col)]
    agg = aggregate_to_solve_steps(timed, kind, group_cols, base, time_col=time_col, value_col=value_col,
                                   keys=key, notes=scope.notes if scope is not None else None)
    agg = agg.select(*(pl.col(c).cast(frame.schema[c]) for c in frame.columns))
    rest = frame.filter(pl.col(time_col).is_null())
    return pl.concat([agg, rest]) if rest.height else agg
