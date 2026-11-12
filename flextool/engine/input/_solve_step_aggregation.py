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
without arithmetic (a constant and its equivalent map give the same model); a SUM step always sums.  Otherwise the
step takes the exact mean (``HARMONIC``: the reciprocal of the duration-weighted mean of ``1/v``), not rounded;
only the profile (:data:`ROUND6_KEYS`) keeps the 6-decimal rounding of its earlier CSV path.  Equal base durations
use ``Σ v / n``; SUM is ``math.fsum`` (n equal values give exactly n·v, the value of a constant), the means
use Neumaier's compensated sum (Python 3.12's ``sum``).  The same vectorised function
serves the provider's CSV frames (``TimelineConfig.create_averaged_timeseries``) and the DB-direct readers, so both
paths give identical bits; :func:`combine` is its per-step reference.

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
import math
from enum import Enum
from typing import Iterable

import numpy as np
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
    ("group", "capacity_margin_share"),
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
    ("unit__inputNode", "startup_energy"),
    ("unit__outputNode", "constraint_flow_coeff"),
})

# Parameters whose aggregation is exact only for some entities while the value is constant within a solve step:
# a varying value of such an entity gets one warning per run.  ``unit.min_load`` matters to the
# min_load_efficiency curve only; for the other units it only bounds the online flow, where the step mean is
# exact.  The callers pass the entities it matters to (``approx_entities``).
_APPROXIMATE_WHEN_VARYING: dict[tuple[str, str], str] = {
    ("unit", "min_load"): (
        "unit.min_load varies within a solve step of new_stepduration (units with min_load_efficiency: "
        "{entities}). The step uses the mean min_load and the harmonic means of efficiency and "
        "efficiency_at_min_load, so the min_load_efficiency curve (slope and section) of those steps is an "
        "approximation of the step mean of the hourly curves; it is exact for a min_load that is constant within "
        "each step."),
}

#: Keys whose MEAN keeps the 6-decimal rounding of earlier releases: the profile, whose CSV path rounded its step
#: mean before (the goldens).  Every other MEAN is unrounded, so small values (self_discharge_loss 1e-7 per hour)
#: keep their precision.
ROUND6_KEYS: frozenset[tuple[str, str]] = frozenset({("profile", "profile")})

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


def _sum(values: "Iterable[float]") -> float:
    """Σ ``values`` by Neumaier's compensated summation, the algorithm of Python 3.12's ``sum`` of floats (the
    same bits on every Python version; :func:`_sum_rows` is its vectorised twin)."""
    it = iter(values)
    f = 0.0 + next(it)
    c = 0.0
    for x in it:
        t = f + x
        c += ((f - t) + x) if abs(f) >= abs(x) else ((x - t) + f)
        f = t
    return f + c if c and math.isfinite(c) else f


def _sum_rows(m: np.ndarray) -> np.ndarray:
    """:func:`_sum` of every row of ``m`` (shape groups x base steps), vectorised over the rows."""
    f = 0.0 + m[:, 0]
    c = np.zeros_like(f)
    for j in range(1, m.shape[1]):
        x = m[:, j]
        t = f + x
        c += np.where(np.abs(f) >= np.abs(x), (f - t) + x, (x - t) + f)
        f = t
    return np.where((c != 0.0) & np.isfinite(c), f + c, f)


def _fsum_rows(m: np.ndarray) -> np.ndarray:
    """``math.fsum`` of every row of ``m`` (the correctly rounded sum).  One or two values per row are summed
    exactly by IEEE arithmetic (``a + b`` is correctly rounded); longer rows take ``math.fsum`` per row."""
    if m.shape[1] == 1:
        return m[:, 0] + 0.0
    if m.shape[1] == 2:
        return (m[:, 0] + m[:, 1]) + 0.0          # fsum never returns -0.0
    return np.array([math.fsum(r) for r in m.tolist()], dtype=np.float64)


def combine(kind: Agg, values: list[float], durations: list[float], *, round6: bool = False) -> float:
    """The value of one solve step from the ``values`` of its base steps (timeline order) and their ``durations``.
    The reference of the vectorised :func:`aggregate_to_solve_steps` (the same bits).

        SUM               → math.fsum(v)                 (an energy per base step: never the value itself; the
                                                         correctly rounded sum, so n equal values give n·v exactly,
                                                         the value of a constant)
        HARMONIC, v ≤ 0   → ValueError                   (also when all values are equal)
        all values equal  → the value itself             (MEAN, HARMONIC, LAST)
        LAST              → values[-1]
        MEAN              → Σ v / n                      equal durations
                            Σ v·Δ / Σ Δ                  otherwise
                            rounded to 6 decimals when ``round6`` (:data:`ROUND6_KEYS`)
        HARMONIC          → 1 / (Σ (1/v) / n)            equal durations
                            1 / (Σ Δ/v / Σ Δ)            otherwise
    Σ is :func:`_sum`.
    """
    if kind is Agg.SUM:
        return math.fsum(values)
    if kind is Agg.HARMONIC and any(v <= 0.0 for v in values):
        raise ValueError("a harmonic mean needs positive values")
    first = values[0]
    if all(v == first for v in values):
        return first
    if kind is Agg.LAST:
        return values[-1]
    equal = all(d == durations[0] for d in durations)
    if kind is Agg.MEAN:
        m = (_sum(values) / len(values) if equal
             else _sum(v * d for v, d in zip(values, durations)) / _sum(durations))
        return round(m, 6) if round6 else m
    try:
        if equal:
            return 1.0 / (_sum(1.0 / v for v in values) / len(values))
        return 1.0 / (_sum(d / v for v, d in zip(values, durations)) / _sum(durations))
    except (ZeroDivisionError, OverflowError):
        raise ValueError("the harmonic mean is not a finite number") from None


def _what(key: tuple[str, str]) -> str:
    return f"{key[0]}.{key[1]}"


def _note_once(notes: "set | None", tag: tuple, message: str) -> None:
    """One WARNING per ``tag`` within ``notes`` (the run's dedupe set; always logged without one)."""
    if notes is not None:
        if tag in notes:
            return
        notes.add(tag)
    _LOGGER.warning(message)


_KIND_CODE: dict[Agg, int] = {Agg.MEAN: 0, Agg.HARMONIC: 1, Agg.SUM: 2, Agg.LAST: 3}


def _step_values(kind: np.ndarray, v: np.ndarray, dur: np.ndarray, cnt: np.ndarray,
                 ) -> "tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]":
    """``(value, all_equal, bad_harmonic, first)`` per group: :func:`combine` vectorised (without the rounding).

    ``kind`` is the :data:`_KIND_CODE` per group, ``v`` / ``dur`` the base rows sorted by group then timeline
    order, ``cnt`` the rows per group.  The groups of one length are summed together as a dense
    (groups x length) block (no padding: every block holds exactly its rows)."""
    n_groups = cnt.shape[0]
    value = np.empty(n_groups)
    all_eq = np.empty(n_groups, dtype=bool)
    bad = np.zeros(n_groups, dtype=bool)
    first = np.empty(n_groups)
    starts = np.zeros(n_groups, dtype=np.int64)
    np.cumsum(cnt[:-1], out=starts[1:])
    for length in np.unique(cnt):
        g = np.flatnonzero(cnt == length)
        idx = starts[g][:, None] + np.arange(length)[None, :]
        mv, md = v[idx], dur[idx]
        k = kind[g]
        f = mv[:, 0]
        first[g] = f
        eq = (mv == f[:, None]).all(axis=1)
        all_eq[g] = eq
        deq = (md == md[:, :1]).all(axis=1)
        harm = k == _KIND_CODE[Agg.HARMONIC]
        bad[g] = harm & (mv <= 0.0).any(axis=1)
        out = np.where(eq, f, mv[:, -1])                     # equal values; LAST
        is_sum = k == _KIND_CODE[Agg.SUM]
        if is_sum.any():
            out = np.where(is_sum, _fsum_rows(mv), out)
        mean = (k == _KIND_CODE[Agg.MEAN]) & ~eq
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            if mean.any():
                out = np.where(mean & deq, _sum_rows(mv) / length, out)
                if (mean & ~deq).any():
                    out = np.where(mean & ~deq, _sum_rows(mv * md) / _sum_rows(md), out)
            hm = harm & ~eq & ~bad[g]
            if hm.any():
                safe = np.where(hm[:, None], mv, 1.0)
                out = np.where(hm & deq, 1.0 / (_sum_rows(1.0 / safe) / length), out)
                if (hm & ~deq).any():
                    out = np.where(hm & ~deq, 1.0 / (_sum_rows(md / safe) / _sum_rows(md)), out)
        value[g] = out
    return value, all_eq, bad, first


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
    approx_entities: "dict[tuple[str, str], frozenset[str]] | None" = None,
) -> pl.DataFrame:
    """``frame`` (``group_cols``, ``time_col``, ``value_col``, …) with one row per (group, solve step).

        out[k, s] = combine(kind, [v[k, t] : t ∈ base(s), v[k, t] given], [Δt : same t], round6=key ∈ ROUND6_KEYS)
        ∀ group k, solve step s with at least one given value;  t ∉ any base(s) is dropped

    Vectorised (no per-step Python loop; the same bits as :func:`combine`).  ``base_steps`` is
    ``solve_data/new_step_base_steps`` (:data:`BASE_STEP_COLUMNS`), or the part of it for the solve steps in
    use.  ``kind`` is one :class:`Agg` for the whole frame, or the name of a column of ``frame`` holding a kind
    name per row (a frame of several parameters, constant within each group); ``kind_values`` maps those names to
    kinds.  ``keys`` names the ``(class, parameter)`` of the rows for the messages and the rounding (one key, or
    per kind-column value).  Null values are dropped.  The output keeps ``group_cols`` (their dtypes),
    ``time_col`` (Utf8, the solve step) and ``value_col`` (Float64), in the order of each group's first row in
    ``frame``.

    ``report_changes``: name the entities whose step value differs from the value of the step's first base step
    (what earlier releases used) in the behaviour-change notice ``solve_step_series_combined``; off for the
    series that earlier releases already combined (the time-map profile and inflow, the energy-margin adder,
    which has its own notice).  ``approx_entities``: per :data:`_APPROXIMATE_WHEN_VARYING` key, the entities
    whose varying value is approximate (no warning for the others); a key without an entry warns for every
    entity.
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
    by = [*group_cols, *extra, "__s"]
    rows = (frame.with_row_index("__row")
            .with_columns(pl.col(time_col).cast(pl.Utf8).alias("__t"),
                          pl.col(value_col).cast(pl.Float64, strict=False).alias("__v"))
            .filter(pl.col("__v").is_not_null())
            .join(base, on="__t", how="inner")
            .sort("__pos", maintain_order=True)
            .with_row_index("__r"))
    grouped = (rows.group_by(by, maintain_order=True)
               .agg(pl.col("__row").min(), pl.col("__r"), pl.col("__n").first(), pl.len().alias("__cnt"),
                    pl.col("__pos").first().alias("__p0"))
               .sort("__row"))
    out_frame = grouped.select(*group_cols, pl.col("__s").alias(time_col))
    if grouped.height == 0:
        return out_frame.with_columns(pl.Series(value_col, [], dtype=pl.Float64))
    order = grouped.get_column("__r").explode().to_numpy()
    v = rows.get_column("__v").to_numpy()[order]
    dur = rows.get_column("__dur").to_numpy()[order]
    cnt = grouped.get_column("__cnt").to_numpy().astype(np.int64)
    n_base = grouped.get_column("__n").to_numpy()

    def key_of(label) -> "tuple[str, str] | None":
        if isinstance(keys, dict):
            return keys.get(label)
        return keys

    if per_row:
        label_arr = grouped.get_column(kind_col).cast(pl.Utf8)
        uniq = label_arr.unique(maintain_order=True).to_list()
        code_of = {lab: _KIND_CODE[kind_values[lab]] for lab in uniq}
        kinds = label_arr.replace_strict(code_of, return_dtype=pl.Int64).to_numpy()
        round_mask = label_arr.is_in([lab for lab in uniq if key_of(lab) in ROUND6_KEYS]).to_numpy()
    else:
        uniq = [None]
        kinds = np.full(grouped.height, _KIND_CODE[kind], dtype=np.int64)
        label_arr = None
        round_mask = np.full(grouped.height, keys in ROUND6_KEYS if isinstance(keys, tuple) else False)
    value, all_eq, bad, first = _step_values(kinds, v, dur, cnt)

    def labels_where(mask: np.ndarray) -> "list":
        if label_arr is None:
            return [None] if mask.any() else []
        return list(dict.fromkeys(label_arr.filter(pl.Series(mask)).to_list()))

    def first_index(mask: np.ndarray, label) -> int:
        if label_arr is not None:
            mask = mask & (label_arr == label).to_numpy()
        return int(np.flatnonzero(mask)[0])

    if bad.any():
        label = labels_where(bad)[0]
        key = key_of(label)
        raise FlexToolConfigError(
            f"{_what(key) if key else 'A time series'}: a harmonic mean needs positive values; a value of 0 or "
            f"less cannot be averaged over the time steps of a new_stepduration step.")
    harmonic = (kinds == _KIND_CODE[Agg.HARMONIC]) & ~all_eq
    broken = harmonic & ~np.isfinite(value) & np.isfinite(first)
    if broken.any():
        key = key_of(labels_where(broken)[0])
        raise FlexToolConfigError(
            f"{_what(key) if key else 'A time series'}: the harmonic mean is not a finite number; a value of 0 or "
            f"less cannot be averaged over the time steps of a new_stepduration step.")
    rounding = round_mask & (kinds == _KIND_CODE[Agg.MEAN]) & ~all_eq
    if rounding.any():
        idx = np.flatnonzero(rounding)
        value[idx] = [round(x, 6) for x in value[idx].tolist()]

    entity_cols = [c for c in group_cols if c not in (kind_col,)]
    partial = (kinds != _KIND_CODE[Agg.LAST]) & (cnt < n_base)
    steps_col = grouped.get_column("__s")
    for label in labels_where(partial):
        key = key_of(label)
        name = _what(key) if key else "a time series"
        i = first_index(partial, label)
        sample = grouped.select(entity_cols).row(i) if entity_cols else ()
        _note_once(notes, ("partial", key or label), (
            f"{name}: a solve step of new_stepduration has values for only some of its time steps "
            f"(e.g. {', '.join(str(x) for x in sample if x is not None)} at {steps_col[i]}); "
            f"the step takes the time steps that have a value (the default of the other time steps is not part "
            f"of it)."))
    varying_labels = [lab for lab in uniq if key_of(lab) in _APPROXIMATE_WHEN_VARYING]
    for label in varying_labels:
        key = key_of(label)
        mask = ~all_eq if label_arr is None else (~all_eq & (label_arr == label).to_numpy())
        if not mask.any() or not entity_cols:
            continue
        names = set(grouped.get_column(entity_cols[0]).cast(pl.Utf8).filter(pl.Series(mask)).to_list())
        if approx_entities is not None and key in approx_entities:
            names &= set(approx_entities[key])
        if names:
            _note_once(notes, ("varying", key),
                       _APPROXIMATE_WHEN_VARYING[key].format(entities=", ".join(sorted(names))))
    # Earlier releases took the step's first base step, its default where that hour has no value: a step whose
    # first authored hour is not its first hour counts as changed too.
    step_p0 = grouped.select("__s").join(base.group_by("__s").agg(pl.col("__pos").min().alias("__sp0")),
                                         on="__s", how="left", maintain_order="left").get_column("__sp0")
    changed = (value != first) | (grouped.get_column("__p0").to_numpy() != step_p0.to_numpy())
    if report_changes and changed.any():
        _report_changes(changed, grouped, entity_cols, label_arr, key_of)
    return out_frame.with_columns(pl.Series(value_col, value, dtype=pl.Float64))


def _report_changes(changed: np.ndarray, grouped: pl.DataFrame, entity_cols: list[str],
                    label_arr: "pl.Series | None", key_of) -> None:
    """The behaviour-change notice ``solve_step_series_combined`` (printed only while the input database carries
    its marker): the parameters and entities whose solve-step value is no longer the value of the step's first
    base step."""
    from flextool.engine import _behaviour_changes

    if not _behaviour_changes.has_marker("solve_step_series_combined"):
        return
    idx = np.flatnonzero(changed)
    first_col = (grouped.get_column(entity_cols[0]).cast(pl.Utf8).gather(idx).to_list()
                 if entity_cols else None)
    labels = label_arr.gather(idx).to_list() if label_arr is not None else [None] * len(idx)
    names: list[str] = []
    for j, label in enumerate(labels):
        key = key_of(label)
        what = _what(key) if key else "a time series"
        names.append(f"{what} of {first_col[j]}" if first_col is not None else what)
    _behaviour_changes.notice("solve_step_series_combined", "parameters", sorted(dict.fromkeys(names)))


# ---------------------------------------------------------------------------
# The active solve's base steps (armed by load_flextool for the DB-direct readers)


class _SolveSteps:
    def __init__(self, base_steps: "pl.DataFrame | None", notes: "set | None",
                 approx_entities: "dict[tuple[str, str], frozenset[str]] | None" = None) -> None:
        self.base_steps = base_steps if base_steps is not None and base_steps.height > 0 else None
        self.notes: set = notes if notes is not None else set()
        self.approx_entities = approx_entities


_SOLVE_STEPS: "contextvars.ContextVar[_SolveSteps | None]" = contextvars.ContextVar(
    "flextool_solve_step_aggregation", default=None)


def activate_solve_step_aggregation(base_steps: "pl.DataFrame | None", notes: "set | None" = None,
                                    approx_entities: "dict[tuple[str, str], frozenset[str]] | None" = None,
                                    ) -> "contextvars.Token":
    """Arm ``base_steps`` (``solve_data/new_step_base_steps`` of the solve being loaded; None or empty: the solve
    has no ``new_stepduration``) for :func:`aggregate_if_active`; returns the token for
    :func:`deactivate_solve_step_aggregation`.  ``notes``: the run's warning-dedupe set; ``approx_entities``:
    see :func:`aggregate_to_solve_steps`."""
    return _SOLVE_STEPS.set(_SolveSteps(base_steps, notes, approx_entities))


def deactivate_solve_step_aggregation(token: "contextvars.Token") -> None:
    """Restore the state before the matching activation."""
    _SOLVE_STEPS.reset(token)


def active_base_steps() -> "pl.DataFrame | None":
    """The armed base steps, or None when no solve step combines several base steps."""
    scope = _SOLVE_STEPS.get()
    return None if scope is None else scope.base_steps


def base_steps_of(base_steps: pl.DataFrame, steps: "pl.Series | None") -> pl.DataFrame:
    """The rows of ``base_steps`` whose solve step is one of ``steps`` (all rows when None): a roll aggregates
    only its own steps."""
    if steps is None:
        return base_steps
    return base_steps.filter(pl.col("new_step").cast(pl.Utf8).is_in(steps.cast(pl.Utf8).unique().implode()))


def aggregate_if_active(frame: pl.DataFrame, key: "tuple[str, str] | None", *, time_col: str = "t",
                        value_col: str = "value", allow_sum: bool = False,
                        steps: "pl.Series | None" = None) -> pl.DataFrame:
    """``frame`` aggregated to the armed solve steps (:func:`aggregate_to_solve_steps`, grouped by every column
    other than ``time_col`` and ``value_col``); ``frame`` itself when nothing is armed or it has no time rows.

    ``steps``: the solve steps in use (the ``t`` of the roll's ``dt``); only their base steps are aggregated,
    so the cost of a roll is proportional to the roll, not to the whole timeline (rows of other steps are
    dropped, as the later join with the roll's ``dt`` would drop them).  Rows with a null ``time_col`` (a period
    or constant default of a mixed frame) pass through unchanged.  A ``key`` without a :data:`TIME_AGGREGATION`
    entry raises.  A SUM key raises unless ``allow_sum``: its constant and period rows would need the base-step
    count, which the caller has to apply.
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
    agg = aggregate_to_solve_steps(timed, kind, group_cols, base_steps_of(base, steps), time_col=time_col,
                                   value_col=value_col, keys=key,
                                   notes=scope.notes if scope is not None else None,
                                   approx_entities=scope.approx_entities if scope is not None else None)
    agg = agg.select(*(pl.col(c).cast(frame.schema[c]) for c in frame.columns))
    rest = frame.filter(pl.col(time_col).is_null())
    return pl.concat([agg, rest]) if rest.height else agg
