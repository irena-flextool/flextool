"""The cohort table: every block of capacity of a solve, when it starts to operate and when its life ends.

A *cohort* is capacity that came into service together: the pre-existing capacity of an entity (the user's
``existing``, one cohort per entity counted from the start of the run) or the capacity of one investment decision
``v_invest[e, d_build]``.  Each cohort has one start of operation ``y_on`` and one end of life ``y_eol``; it is in
service in the periods ``d`` with ``y_on <= y[d] < y_eol``.  Every question "is this capacity still there in
period d" is answered from these frames, so the lifetime rule lives in one place.

Three frames (``FlexData.cohort``, ``cohort_alive``, ``dd_forward``):

* ``cohort`` — one row per cohort: ``cohort`` (Utf8 id, unique per entity: ``existing:<run start period>`` /
  ``invest:<decision period>``; ``existing`` when the solve has no period years; ``gen:<root>:<k>`` for a
  generation; ``extend:<period>:<parent>`` for a life extension), ``e`` (entity-union axis), ``kind``
  (``existing`` / ``invest`` / ``generation`` / ``extend``), ``method``
  (``lifetime_method`` with its default), ``eol_method``
  (``end_of_life_method`` with its default ``closest_seam``: how the end of life is rounded), ``d_build`` (the
  decision period, the key of ``v_invest``; null for existing capacity), ``d_on`` (Utf8: the period the capacity
  starts to operate in: the run start for existing capacity, the decision period or, with a construction lead
  time, the later period of ``ed_invest_commission``), ``y_on`` (start of operation; existing capacity:
  ``y[d0] − age``), ``age`` (the age of existing capacity at the model start, :func:`existing_ages`; 0 for an
  investment), ``y_life`` (``lifetime_technical`` of ``d_on``, the plant as constructed; ``+inf`` for
  ``reinvest_automatic``), ``y_end = y_on + y_life`` (the end of life, unrounded), ``y_first_life_end``
  (existing capacity: ``y[d0] + y_life``, the end of its first life counted from the model start without the age
  and unrounded, the year a ``no_reinvestment`` entity stops investing; null for
  an investment), ``y_eol`` (``y_end`` rounded to a seam by ``eol_method``,
  :func:`._run_calendar.round_end_of_life`: ``+inf`` for ``reinvest_automatic`` and for ``next_seam`` past the last
  seam), ``d_eol`` (Utf8: the run period that starts at ``y_eol``; null at ``+inf`` and at the end of the
  horizon) and the prices (CUR/MW a year, MWh for storage; ``_npv.apply_invest_lumps``, null until the lumps are
  set): ``a_cap`` the capital part of an investment cohort's price per year in service, the level yearly cost of
  the plant over its real lifetime (the annuity for ``reinvest_automatic``; null for existing capacity, which has
  no capital in the model); for a ``reinvest_choice`` / ``no_reinvestment`` investment the loan's yearly payment
  ``pay`` (A), its years ``n_pay`` and interest rate ``r_loan``; ``a_salv`` the level yearly salvage credit (the
  price per year in service is ``a_cap − a_salv``; 0 for ``reinvest_automatic``; for existing capacity its credit
  a_x) and ``nsv_eol`` the salvage value at the end of life (existing capacity: ``r_loan`` its discount_rate at
  the run start, ``nsv_eol`` S_x), ``root`` (Utf8: the cohort the row belongs to; a root row's own id) and
  ``parent`` (Utf8: the cohort a life extension extends, ``_extend``; null on every other row).  An ``extend`` row
  is a cohort of its own: decided (``d_build``) and online in the period that starts at its parent's end of life,
  ``y_life`` its ``life_extension_years``, priced as one plant (``_extend.extension_prices``).

  A ``reinvest_automatic`` root (an investment decision, or the existing capacity) is renewed: its row keeps the
  window of the whole chain (``y_life`` / ``y_end`` / ``y_eol`` ``+inf``), and its *generations* (kind
  ``generation``, one row each, :func:`chain_rows`) are the plants that hold its capacity one after another: g1
  comes online at the root's ``y_on``; ``g_{k+1}`` at ``y_eol[g_k]``; ``y_life`` is ``lifetime_technical`` of the
  period the generation comes online in (``d_on``, the plant as constructed) and ``y_eol`` its end rounded by the
  ``end_of_life_method`` (at least one period: a rounding to its own start gives the next seam), up to the end of
  the horizon.  Generations hold no capacity of their own (no ``cohort_alive`` rows, no size): the chain keeps the
  root's capacity, less its retirements; they carry the prices of each plant.  The existing chain's g1 is the
  plant in service at the run start (its age is the existing capacity's age less the replacements before the
  run, :func:`_existing_first_generation`).
* ``cohort_alive`` — ``[cohort, e, kind, d_build, d, y]``: the periods ``d`` of the solve's ``period_in_use`` in
  which a cohort is in service, ``y`` the year of ``d``.
* ``dd_forward`` — ``[d_retire, d]``: a retirement decided in ``d_retire`` counts in ``d`` when
  ``y[d_retire] <= y[d]`` (along the scenario lineage of a stochastic fan).  Entity-independent: a retirement of
  existing capacity counts in ``d`` when ``(d_retire, d) ∈ dd_forward`` and the existing cohort is alive in ``d``.

Years are years of the run: a solve's period years (``p_years_d``, which covers the period history) on the run
calendar (:func:`cohort_years`, :func:`._run_calendar.run_years`: a period takes its run-calendar year, a branch
period its anchor's, never a year shifted by an offset).  The seams are those years, the start of the run, and the
run calendar's period starts and horizon end, each year once (a run seam within ``YEAR_TOL`` of a solve year is that
year, so ``y_eol`` is a period's own year and ``y[d] < y_eol`` compares exactly).  An end of life is rounded with
the year tolerance ``YEAR_TOL`` under every ``end_of_life_method``; for ``next_seam`` and a period year ``y`` (a
seam; the seams lie more than ``YEAR_TOL`` apart) ``y < next_seam(Y) ⟺ y < Y − YEAR_TOL``.  So a cohort placed by one solve
(the existing cohort, built once for the run: :func:`run_existing_cohorts`; an investment realized by an earlier
solve, carried in the handoff ``realized_cohort``: ``_cohort_handoff``) is in service in a later solve's period
``d`` by the same comparison of ``d``'s run year with the carried ``y_on`` / ``y_eol``, with no rounding repeated.
Where two solves give a period different years (the run calendar's ``conflicts``), the run calendar's year (the
first solve's) decides.

Views (the only lifetime-gate code): :func:`existing_eol`, :func:`existing_expired`,
:func:`no_reinvestment_closed`, :func:`invest_alive`, :func:`invest_prebuilt`, :func:`retire_active`; the test
whether a year lies in a life ``[y_on, y_eol)``, with its tolerance, is :func:`within_life`.

A cohort taken from the ``edd_invest`` seed of a sub-solve without the investment walk has unknown years
(``y_on`` null, ``y_eol = +inf``): its alive periods are the seed's, and it is never pre-built
(:func:`invest_prebuilt` needs ``y > y_on``).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Iterable, Mapping

import polars as pl

from flextool.engine.input._axis_enums import (
    align_join_dtypes,
    axis_lazyframe,
    cast_frame_axes,
    get_global_axis_enums,
    join_aligned,
    rename_to_axis,
    schema_dtype,
)
from flextool.engine.input._lifetime_roles import (
    AGE_AT_MODEL_START,
    BUILD_YEAR,
    END_OF_LIFE_METHOD,
    END_OF_LIFE_METHOD_DEFAULT,
    END_OF_LIFE_METHODS,
    ENTITY_CLASSES,
    LIFETIME_TECHNICAL,
    NO_REINVESTMENT,
    START_YEAR,
    existing_age,
    source_entities_with,
)
from flextool.engine.input._solve_state import CommissioningLagError
from flextool.engine.preprocess.params._period_values import _ed_period_value_map, ed_values_online
from flextool.engine.preprocess.params._run_calendar import YEAR_TOL, eol_moves, round_end_of_life, snap_to_seam

if TYPE_CHECKING:
    from flextool.engine.input._input_source import InputSource
    from flextool.engine.preprocess.params._run_calendar import RunCalendar

KIND_EXISTING = "existing"
KIND_INVEST = "invest"
#: A plant of a ``reinvest_automatic`` chain (:func:`chain_rows`).
KIND_GENERATION = "generation"
#: The life extension of a cohort (``_extend``): a cohort of its own, ``parent`` the cohort it extends.
KIND_EXTEND = "extend"
REINVEST_AUTOMATIC = "reinvest_automatic"
REINVEST_CHOICE = "reinvest_choice"
#: The ``lifetime_method`` values; an entity with another value has no cohort.
LIFETIME_METHODS: tuple[str, ...] = (REINVEST_AUTOMATIC, REINVEST_CHOICE, NO_REINVESTMENT)
#: The methods whose capacity ends its life (``reinvest_automatic`` capacity is renewed).
BOUNDED_METHODS: tuple[str, ...] = (REINVEST_CHOICE, NO_REINVESTMENT)

_INF = float("inf")
#: The behaviour change of the ``closest_seam`` default of ``end_of_life_method`` (``_behaviour_changes``).
END_OF_LIFE_CLOSEST_SEAM = "end_of_life_closest_seam"


def _cohort_schema() -> dict[str, pl.DataType]:
    return {"cohort": pl.Utf8, "e": schema_dtype(None, "e"), "kind": pl.Utf8, "method": pl.Utf8,
            "eol_method": pl.Utf8, "d_build": schema_dtype(None, "d"), "d_on": pl.Utf8, "y_on": pl.Float64,
            "age": pl.Float64, "y_life": pl.Float64,
            "y_end": pl.Float64, "y_first_life_end": pl.Float64, "y_eol": pl.Float64, "d_eol": pl.Utf8,
            **{c: pl.Float64 for c in PRICE_COLUMNS}, "root": pl.Utf8, "parent": pl.Utf8}


#: The price columns of the cohort table (CUR/MW, MWh for storage; ``_npv.apply_invest_lumps`` sets them): the
#: level yearly price ``a_cap``, the loan's yearly payment ``pay``, years ``n_pay`` and interest rate ``r_loan``,
#: the level yearly salvage credit ``a_salv`` and the salvage value ``nsv_eol`` at the end of life.
PRICE_COLUMNS: tuple[str, ...] = ("a_cap", "pay", "n_pay", "r_loan", "a_salv", "nsv_eol")
#: The price cells of a cohort row before the prices are set.
_NO_PRICES: tuple[None, ...] = (None,) * len(PRICE_COLUMNS)


#: The cohort-table columns on an axis (``e`` / ``d_build``); every other column is Utf8 or Float64.
_AXIS_COLUMNS: dict[str, str] = {"e": "e", "d_build": "d_build"}
#: The cohort-table columns that hold a year of the run (the solve's period years on the run calendar).
YEAR_COLUMNS: tuple[str, ...] = ("y_on", "y_end", "y_first_life_end", "y_eol")


def _alive_schema() -> dict[str, pl.DataType]:
    return {"cohort": pl.Utf8, "e": schema_dtype(None, "e"), "kind": pl.Utf8, "d_build": schema_dtype(None, "d"),
            "d": schema_dtype(None, "d"), "y": pl.Float64}


def _forward_schema() -> dict[str, pl.DataType]:
    return {"d_retire": schema_dtype(None, "d"), "d": schema_dtype(None, "d")}


@dataclass(frozen=True)
class CohortFrames:
    """The three cohort frames of a solve (see the module docstring)."""

    cohort: pl.DataFrame
    cohort_alive: pl.DataFrame
    dd_forward: pl.DataFrame | None


@dataclass(frozen=True)
class _Calendar:
    """The years and seams of the run, the run period label of a seam and the years periods start at."""

    years: dict[str, float]
    seams: list[float]
    label: Callable[[float], str | None]
    period_starts: tuple[float, ...] = ()
    #: The end of the run horizon (``None`` when no period has a length).
    horizon: float | None = None
    #: ``{period: start}`` of the run calendar's periods, in run order.
    run_periods: tuple[tuple[str, float], ...] = ()

    def eol(self, y_end: float, method: str = END_OF_LIFE_METHOD_DEFAULT) -> float:
        """``y_end`` rounded to a seam by the ``end_of_life_method`` ``method``
        (:func:`._run_calendar.round_end_of_life`)."""
        return round_end_of_life(y_end, method, self.seams)

    def online_period(self, y: float) -> str | None:
        """The period a plant coming online at the year ``y`` comes online in: the run period that starts there,
        else the solve's period with that year, else the last run period starting before ``y`` (``None`` before
        every period)."""
        at = self.label(y)
        if at is not None:
            return at
        for d, yd in self.years.items():
            if abs(yd - y) <= YEAR_TOL:
                return d
        before = [d for d, yd in self.run_periods if yd <= y + YEAR_TOL]
        return before[-1] if before else None

    def next_seam(self, y: float) -> float:
        """The first seam after the year ``y`` (``+inf`` past the last one)."""
        return min((s for s in self.seams if s > y + YEAR_TOL), default=_INF)


def _run_calendar_of(source: "InputSource",
                     active_solve: str | None,
                     years: Mapping[str, float],
                     provider: "object | None") -> "RunCalendar":
    """The run calendar on ``provider``; without one (a solve built on its own) the solve's own periods and
    ``years_represented`` stand in for it."""
    from flextool.engine.preprocess.params._existing import _solve_years_represented
    from flextool.engine.preprocess.params._run_calendar import read_run_calendar, solve_calendar
    rc = read_run_calendar(provider)
    if rc is None:
        rc = solve_calendar(years, _solve_years_represented(source, active_solve))
    return rc


def cohort_years(source: "InputSource",
                 active_solve: str | None,
                 workdir: Path | None,
                 *,
                 provider: "object | None" = None) -> dict[str, float]:
    """``{period: run year}`` the cohort table compares: the solve's period years (:func:`._derived._periods._p_years_d_lf`:
    with the Provider, the solve's ``p_years_d`` over its period history; without it, the solve's own
    ``years_represented``) on the run calendar (:func:`._run_calendar.run_years`: a period takes its run-calendar
    year, a branch period its anchor's).  A solve that is not handed the run history (its years count from its own
    first period) so compares in the years of the run, where the cohorts of earlier solves were placed."""
    from flextool.engine.preprocess.params._derived._periods import _p_years_d_lf
    from flextool.engine.preprocess.params._run_calendar import read_run_calendar, run_years
    pyd_lf = _p_years_d_lf(source, active_solve, workdir, provider=provider)
    if pyd_lf is None:
        return {}
    own: dict[str, float] = {}
    for d, y in pyd_lf.select("d", "yr").collect().iter_rows():
        own.setdefault(str(d), float(y if y is not None else 0.0))
    return run_years(own, read_run_calendar(provider))


def cohort_calendar(rc: "RunCalendar", years: Mapping[str, float]) -> _Calendar:
    """The seams of the run: the years ``years``, the start of the run (0) and the run calendar's period starts and
    horizon end (a run seam within ``YEAR_TOL`` of a year of ``years`` is that year).  ``label(y)`` is the run
    period that starts at the seam ``y`` (``None`` at ``+inf`` and where no period starts)."""
    from flextool.engine.preprocess.params._run_calendar import YEAR_TOL
    own = sorted({*(float(y) for y in years.values()), 0.0})
    seams = list(own)
    for s in rc.seams():
        if all(abs(s - o) > YEAR_TOL for o in own):
            seams.append(s)
    seams.sort()

    def label(y: float) -> str | None:
        return None if y == _INF else rc.period_at(y)

    starts = tuple(sorted({*(float(y) for y in years.values()), *rc.years.values()}))
    run_periods = tuple(sorted(((str(d), float(y)) for d, y in rc.years.items()), key=lambda dy: dy[1]))
    return _Calendar(years=dict(years), seams=seams, label=label, period_starts=starts, horizon=rc.horizon_end,
                     run_periods=run_periods)


def _entity_methods(source: "InputSource") -> dict[str, str]:
    """``{entity: lifetime_method}`` (default ``reinvest_automatic``) of every unit, node and connection."""
    from flextool.engine.preprocess.params._existing import _all_entities_lf, _lifetime_method_with_default_lf
    df = _lifetime_method_with_default_lf(source, _all_entities_lf(source)).select("e", "method").collect()
    out: dict[str, str] = {}
    for e, m in df.iter_rows():
        out.setdefault(str(e), str(m))
    return out


def end_of_life_methods(source: "InputSource") -> dict[str, str]:
    """``{entity: end_of_life_method}`` of the units, nodes and connections that have one (explicit, or the
    schema default the source broadcasts); an entity without one rounds by :data:`END_OF_LIFE_METHOD_DEFAULT`
    (:func:`eol_methods_of`).  A constant choice (``resolve_param_shape`` refuses a map); a value outside the
    ``end_of_life_methods`` list raises :class:`FlexToolConfigError`."""
    from flextool.engine.input._param_shapes import resolve_param_shape
    from flextool.engine.input._solve_state import FlexToolConfigError
    out: dict[str, str] = {}
    for cls in ENTITY_CLASSES:
        resolved = resolve_param_shape(source, cls, END_OF_LIFE_METHOD)
        if resolved is None or resolved.frame is None:
            continue
        for e, m in resolved.frame.select(pl.col("name").cast(pl.Utf8), pl.col("value").cast(pl.Utf8)).iter_rows():
            if m is None:
                continue
            if m not in END_OF_LIFE_METHODS:
                raise FlexToolConfigError(f"{cls} {e!r} has end_of_life_method {m!r}: use one of "
                                          f"{', '.join(END_OF_LIFE_METHODS)}.")
            out[str(e)] = m
    return out


def _constant_values(source: "InputSource", cls: str, name: str) -> dict[str, float]:
    """``{entity: value}`` of a constant parameter (``resolve_param_shape`` refuses a map) of class ``cls``."""
    from flextool.engine.input._param_shapes import resolve_param_shape
    resolved = resolve_param_shape(source, cls, name)
    if resolved is None or resolved.frame is None:
        return {}
    return {str(e): float(v) for e, v in resolved.frame.select(pl.col("name").cast(pl.Utf8), "value").iter_rows()
            if v is not None}


def model_start_year(source: "InputSource") -> float | None:
    """``model.start_year`` of the scenario's models (``None`` without one); models that give different years raise
    :class:`FlexToolConfigError`."""
    from flextool.engine.input._solve_state import FlexToolConfigError
    years = _constant_values(source, "model", START_YEAR)
    distinct = sorted(set(years.values()))
    if len(distinct) > 1:
        raise FlexToolConfigError("Input values out of range:\n  model.start_year differs between the models of the "
                                  "scenario: " + ", ".join(f"{m} {v:g}" for m, v in sorted(years.items()))
                                  + "; give the models one start year.")
    return distinct[0] if distinct else None


def existing_ages(source: "InputSource") -> dict[str, float]:
    """``{entity: age}`` of the existing capacity at the model start of the units, nodes and connections with an
    ``age_at_model_start`` or a ``build_year`` (:func:`._lifetime_roles.existing_age`: ``age_at_model_start``,
    else ``model.start_year − build_year``); an entity without either is new at the model start (age 0).  Constants;
    an undefined age (both given, a negative age, a build year without or after the start year) raises
    :class:`FlexToolConfigError` (the input checks name it first, with its alternatives)."""
    from flextool.engine.input._solve_state import FlexToolConfigError
    start = model_start_year(source)
    out: dict[str, float] = {}
    errors: list[str] = []
    for cls in ENTITY_CLASSES:
        ages = _constant_values(source, cls, AGE_AT_MODEL_START)
        built = _constant_values(source, cls, BUILD_YEAR)
        for e in sorted(set(ages) | set(built)):
            out[e], error = existing_age(ages.get(e), built.get(e), start)
            if error is not None:
                errors.append(f"{cls} '{e}' {error}.")
    if errors:
        raise FlexToolConfigError("Input values out of range:\n  " + "\n  ".join(errors))
    return out


def eol_methods_of(cohort: pl.DataFrame) -> dict[str, str]:
    """``{entity: end_of_life_method}`` of the existing rows of a cohort frame (Utf8 or axis labels): the run's
    rounding of each entity, read where the run placed its existing cohort."""
    rows = cohort.filter(pl.col("kind") == KIND_EXISTING).select(pl.col("e").cast(pl.Utf8), "eol_method")
    return {str(e): str(m) for e, m in rows.iter_rows() if m is not None}


def run_cohort_calendar(source: "InputSource",
                        active_solve: str | None,
                        years: Mapping[str, float],
                        provider: "object | None") -> _Calendar:
    """The seams of the run for the period years ``years`` (:func:`cohort_calendar` on the run calendar of
    ``provider``; the solve's own periods without one)."""
    return cohort_calendar(_run_calendar_of(source, active_solve, years, provider), years)


def salvage_periods(source: "InputSource",
                    active_solve: str | None,
                    years: Mapping[str, float],
                    provider: "object | None",
                    y_eols: Iterable[float | None]) -> list[str | None]:
    """The period whose ``net_salvage_value`` a salvage at each end of life ``y_eols`` (years of the run) takes: the
    run period that starts there; the last period of the run after the last period start (the horizon end, a life
    past it, ``+inf``) or for an unknown end (``None``) — the value of the period the salvage happens in, after the
    last model period the last value."""
    rc = _run_calendar_of(source, active_solve, years, provider)
    cal = cohort_calendar(rc, years)
    return [(cal.label(y) if y is not None else None) or rc.last_period for y in y_eols]


def end_of_life_years(anchors: pl.DataFrame, cal: _Calendar, eol_methods: Mapping[str, str]) -> pl.Series:
    """``y_eol`` (Float64, one per row of ``anchors`` ``[e, y_on, life]``, years of the run) of capacity starting to
    operate at ``y_on`` with the lifetime ``life`` (a null as 0): ``round_end_of_life(y_on + life,
    end_of_life_method[e])``, the end of the cohort window (:func:`window_end` of a bounded ``lifetime_method``).
    The investment windows that stop with the capacity (its lifetime fixed cost) end there."""
    out = [cal.eol((y_on or 0.0) + (life or 0.0), eol_methods.get(str(e), END_OF_LIFE_METHOD_DEFAULT))
           for e, y_on, life in anchors.select("e", "y_on", "life").iter_rows()]
    return pl.Series("y_eol", out, dtype=pl.Float64)


def _years_frame(years: Mapping[str, float]) -> pl.LazyFrame:
    """``[d, y]`` of ``years`` on the period axis."""
    return (axis_lazyframe({"d": list(years)})
            .with_columns(y=pl.Series("y", [float(v) for v in years.values()], dtype=pl.Float64)))


def window_end(y_on: float, y_life: float, method: str, cal: _Calendar,
               eol_method: str = END_OF_LIFE_METHOD_DEFAULT) -> tuple[float, float, float]:
    """``(y_life, y_end, y_eol)`` of capacity starting to operate at ``y_on`` with the lifetime ``y_life`` under
    ``lifetime_method`` ``method``: ``reinvest_automatic`` capacity is renewed (``+inf`` each); otherwise
    ``y_end = y_on + y_life`` and ``y_eol = round_end_of_life(y_end, eol_method)``.  The one lifetime-window rule:
    the capacity is in service in the periods with ``y_on <= y[d] < y_eol``."""
    if method == REINVEST_AUTOMATIC:
        return _INF, _INF, cal.eol(_INF, eol_method)
    y_end = y_on + y_life
    return y_life, y_end, cal.eol(y_end, eol_method)


def within_life(y: pl.Expr, y_on: pl.Expr | None, y_eol: pl.Expr | None) -> pl.Expr:
    """Whether the year ``y`` lies in the life ``[y_on, y_eol)`` of a plant, the one in-service test of the lifetime
    rule (``y_on`` / ``y_eol`` ``None``: no bound on that side)::

        y_on − YEAR_TOL <= y < y_eol − YEAR_TOL

    Every year it compares is a year of the run (:func:`cohort_years`) and an end of life is rounded to a seam,
    which is a period's own year, so on those years this is ``y_on <= y < y_eol`` exactly; the tolerance only keeps
    a year carried on another calendar (or a sum of years) from falling to the wrong side of a seam by rounding.
    A null bound makes the test null (the row is filtered out, a ``when`` takes its ``otherwise``); a caller that
    means "no bound" for a null fills it (``±inf``)."""
    test = None
    if y_on is not None:
        test = y >= y_on - YEAR_TOL
    if y_eol is not None:
        before_end = y < y_eol - YEAR_TOL
        test = before_end if test is None else test & before_end
    if test is None:
        raise ValueError("within_life needs y_on or y_eol.")
    return test


@dataclass(frozen=True)
class _ChainStart:
    """The first generation of a ``reinvest_automatic`` chain (:func:`chain_rows`)."""

    root: str
    e: str
    method: str
    eol_method: str
    d_build: str | None
    d_on: str
    y_on: float
    age: float
    life: float | None
    #: The end of life of g1 when it is not its rounded lifetime: the existing plant replaced at the run start.
    end: float | None = None


def _rounds_to_run_start(t: float, y0: float, cal: _Calendar, method: str) -> bool:
    """Whether a replacement at the year ``t`` (at or before the run start ``y0``) is rounded to the run start by the
    ``end_of_life_method`` ``method``: the seams before the run start continue those of the run by the length of its
    first period (:func:`._run_calendar.snap_to_seam` on ``[y0 − w, y0]``)."""
    w = cal.next_seam(y0) - y0
    if w == _INF:
        return True
    return snap_to_seam(t, method, [y0 - w, y0]) >= y0 - YEAR_TOL


def _existing_first_generation(age: float, life: float, y0: float, cal: _Calendar,
                               eol_method: str) -> tuple[float, float, float | None]:
    """``(age_eff, y_on, end)`` of the plant of a ``reinvest_automatic`` entity's existing capacity in service at the
    run start ``y0`` (lifetime ``life`` > 0)::

        k = ⌊age / life⌋ (the replacements before the run, unrounded: there are no seams before it)
        age_eff = age − k·life,  y_on = y0 − age_eff
        end = y0  when the plant's rounded end of life is at or before y0 (it serves no period), or when it is a
                  replacement (k ≥ 1) dated y_on that the end_of_life_method rounds to the run start
                  (:func:`_rounds_to_run_start`; age = k·life included): that replacement is in the model
              None otherwise (its rounded lifetime ends it)"""
    reps = int((age + YEAR_TOL) // life)
    age_eff = max(0.0, age - reps * life)
    if age_eff <= YEAR_TOL:
        age_eff = 0.0
    y_on = y0 - age_eff
    gone = cal.eol(y_on + life, eol_method) <= y0 + YEAR_TOL
    replaced = reps >= 1 and _rounds_to_run_start(y_on, y0, cal, eol_method)
    return age_eff, y_on, (y0 if gone or replaced else None)


def chain_rows(source: "InputSource", starts: Iterable[_ChainStart], cal: _Calendar) -> list[tuple]:
    """The ``generation`` rows (:func:`_cohort_schema` order, Utf8 labels) of the ``reinvest_automatic`` chains that
    start at ``starts``, one per generation ``gen:<root>:<k>``::

        g1:      y_on, d_on, life of the start (the existing plant: :func:`_existing_first_generation`)
        g_{k+1}: y_on = y_eol[g_k],  d_on = the period it comes online in (:meth:`_Calendar.online_period`),
                 life = lifetime_technical[e, d_on] (the plant as constructed)
        y_end = y_on + life,  y_eol = round_end_of_life(y_end, end_of_life_method[e]); the next seam after y_on
                when that is at or before y_on (every generation serves at least one period)

    until a generation reaches the end of the horizon (``y_eol`` at or past it, ``+inf`` past the last seam).  A
    generation without a lifetime (missing or not > 0) is the last one, to ``+inf``.  Prices are null (set by
    ``_npv``); ``d_build`` is the root's, ``age`` 0 after g1."""
    rows: list[tuple] = []
    pending = [(st, 1, st.y_on, st.d_on, st.age, st.life, st.end) for st in starts]
    horizon = cal.horizon if cal.horizon is not None else (cal.seams[-1] if cal.seams else None)
    while pending:
        nxt: list[tuple[_ChainStart, int, float, str | None]] = []
        for st, k, y_on, d_on, age, life, end in pending:
            cohort = f"gen:{st.root}:{k}"
            if life is None or not 0.0 < life < _INF:
                rows.append((cohort, st.e, KIND_GENERATION, st.method, st.eol_method, st.d_build, d_on, y_on, age,
                             _INF, _INF, None, _INF, None, *_NO_PRICES, st.root, None))
                continue
            y_end = y_on + life
            y_eol = end if end is not None else cal.eol(y_end, st.eol_method)
            if end is None and y_eol <= y_on + YEAR_TOL:
                y_eol = cal.next_seam(y_on)
            rows.append((cohort, st.e, KIND_GENERATION, st.method, st.eol_method, st.d_build, d_on, y_on, age, life,
                         y_end, None, y_eol, cal.label(y_eol), *_NO_PRICES, st.root, None))
            if y_eol < _INF and horizon is not None and y_eol < horizon - YEAR_TOL:
                nxt.append((st, k + 1, y_eol, cal.online_period(y_eol)))
        periods = sorted({d for _st, _k, _y, d in nxt if d is not None})
        lives = _ed_period_value_map(source, LIFETIME_TECHNICAL, periods) if periods else {}
        pending = [(st, k, y_on, d_on, 0.0, lives.get((st.e, d_on)), None) for st, k, y_on, d_on in nxt]
    return rows


def _existing_rows(source: "InputSource", years: Mapping[str, float], cal: _Calendar) -> list[tuple]:
    """The rows (:func:`_cohort_schema` order, Utf8 labels) of the existing cohorts on the years ``years``, then the
    generations of the ``reinvest_automatic`` existing capacity of the processes and storage nodes
    (:func:`chain_rows`, the existing plant's first generation :func:`_existing_first_generation`)."""
    from flextool.engine.preprocess.params._derived import run_start_period
    from flextool.engine.preprocess.params._derived._scope import _entity_classes_lookup
    start = run_start_period(dict(years))
    eol_methods = end_of_life_methods(source)
    ages = existing_ages(source)
    if start is None:
        return [(KIND_EXISTING, e, KIND_EXISTING, method, eol_methods.get(e, END_OF_LIFE_METHOD_DEFAULT), None,
                 None, None, ages.get(e, 0.0), None, None, None, _INF, None, *_NO_PRICES, KIND_EXISTING, None)
                for e, method in _entity_methods(source).items() if method in LIFETIME_METHODS]
    lifetime = _ed_period_value_map(source, LIFETIME_TECHNICAL, [start])
    process_set, node_set, _ = _entity_classes_lookup(source)
    y0 = years[start]
    rows: list[tuple] = []
    chains: list[_ChainStart] = []
    for e, method in _entity_methods(source).items():
        if method not in LIFETIME_METHODS:
            continue
        eol_method = eol_methods.get(e, END_OF_LIFE_METHOD_DEFAULT)
        age = ages.get(e, 0.0)
        y_on = y0 - age
        root = f"{KIND_EXISTING}:{start}"
        if method != REINVEST_AUTOMATIC and e not in process_set and e not in node_set:
            y_life, y_end, y_first, y_eol = 0.0, 0.0, 0.0, cal.eol(0.0, eol_method)
        else:
            y_life, y_end, y_eol = window_end(y_on, lifetime.get((e, start), 0.0), method, cal, eol_method)
            y_first = y0 + y_life
        rows.append((root, e, KIND_EXISTING, method, eol_method, None, start, y_on, age,
                     y_life, y_end, y_first, y_eol, cal.label(y_eol), *_NO_PRICES, root, None))
        if method == REINVEST_AUTOMATIC and (e in process_set or e in node_set):
            life = lifetime.get((e, start))
            if life is not None and 0.0 < life < _INF:
                age_eff, g1_on, end = _existing_first_generation(age, life, y0, cal, eol_method)
            else:
                age_eff, g1_on, end = age, y_on, None
            chains.append(_ChainStart(root=root, e=e, method=method, eol_method=eol_method, d_build=None,
                                      d_on=start, y_on=g1_on, age=age_eff, life=life, end=end))
    _closest_seam_notice(source, rows, cal)
    return rows + chain_rows(source, chains, cal)


def run_existing_cohorts(source: "InputSource", rc: "RunCalendar | None") -> pl.DataFrame:
    """The ``existing`` rows of the cohort table of the whole run (Utf8 labels, run years): one per entity with a
    ``lifetime_method``, counted from the start of the run ``d0`` (:func:`._derived.run_start_period` of the run
    calendar)::

        y_on = y[d0] − age[e],   y_life = lifetime_technical[e, d0] (0 when missing; +inf for reinvest_automatic)
        y_end = y_on + y_life,   y_eol = round_end_of_life(y_end, end_of_life_method[e])
        y_first_life_end = y[d0] + y_life   (the no_reinvestment investment window: no age, no rounding)

    The existing cohort is fixed when the run starts: ``_run_model`` builds this frame once and puts it on every
    sub-solve's Provider (:data:`._run_calendar.RUN_EXISTING_COHORT_KEY`); :func:`existing_cohorts` reads it there.
    An entity that is neither a process nor a node has no lifetime: its end of life is 0.  Without period years
    (``rc`` ``None`` or empty) the end of life cannot be placed: every row has the cohort id ``existing``, null
    years and ``y_eol = +inf`` (in service in every period, no end of life).  A warning names, once per
    run, each process or storage node whose existing capacity is past its rounded end of life at the model start
    (:func:`_past_end_of_life_warning`)."""
    years = dict(rc.years) if rc is not None else {}
    cal = cohort_calendar(rc, years) if rc is not None else None
    frame = _utf8_cohort_frame(_existing_rows(source, years, cal))
    _past_end_of_life_warning(source, frame, years)
    _existing_replacement_info(source, frame)
    return frame


def _existing_replacement_info(source: "InputSource", frame: pl.DataFrame) -> None:
    """Name, once per run, the processes and storage nodes with existing capacity (> 0 in some period) whose
    reinvest_automatic chain has a replacement inside the horizon (a generation after the first): their existing
    capacity reaches its end of life and is replaced at the values of the period the replacement comes online
    (note W-0520)."""
    from flextool.engine.preprocess.params._existing import _per_entity_param_lf
    replaced = frame.filter((pl.col("kind") == KIND_GENERATION) & pl.col("root").str.starts_with(f"{KIND_EXISTING}:")
                            & ~pl.col("cohort").str.ends_with(":1"))
    if replaced.height == 0:
        return
    first = (replaced.sort("y_on").unique(subset=["e"], keep="first", maintain_order=True)
                     .select(pl.col("e").cast(pl.Utf8), "y_on"))
    with_existing = {str(e) for e, v in _per_entity_param_lf(source, "existing").select("e", "value").collect()
                     .iter_rows() if v is not None and v > 0.0}
    years = {e: y for e, y in first.iter_rows() if e in with_existing}
    if years:
        from flextool import run_warnings
        run_warnings.note("W-0520", sorted(years), per_run=True,
                          details={e: f"replaced from year {years[e]:g} of the run" for e in sorted(years)})


def _roots(frame: pl.DataFrame) -> pl.DataFrame:
    """The root rows of a cohort frame (every row but the ``generation`` rows)."""
    return frame.filter(pl.col("kind") != KIND_GENERATION)


def _generations(frame: pl.DataFrame) -> pl.DataFrame:
    """The ``generation`` rows of a cohort frame."""
    return frame.filter(pl.col("kind") == KIND_GENERATION)


def _past_end_of_life_warning(source: "InputSource", frame: pl.DataFrame, years: Mapping[str, float]) -> None:
    """Warn, once per run and entity, about the existing capacity of a ``reinvest_choice`` / ``no_reinvestment``
    process or storage node with ``existing`` > 0 in some period whose rounded end of life is at or before the model
    start (``y_eol <= y[d0] + YEAR_TOL``): its existing capacity is 0 in every period (W-0521, one line)."""
    from flextool import run_warnings
    from flextool.engine.preprocess.params._derived import run_start_period
    from flextool.engine.preprocess.params._derived._scope import _entity_classes_lookup
    start = run_start_period(dict(years))
    if start is None:
        return
    y0 = years[start]
    gone = frame.filter((pl.col("kind") == KIND_EXISTING) & pl.col("method").is_in(list(BOUNDED_METHODS))
                        & pl.col("y_eol").is_not_null() & (pl.col("y_eol") <= y0 + YEAR_TOL))
    if gone.height == 0:
        return
    process_set, node_set, _ = _entity_classes_lookup(source)
    from flextool.engine.preprocess.params._existing import _per_entity_param_lf
    with_existing = {str(e) for e, v in _per_entity_param_lf(source, "existing").select("e", "value").collect()
                     .iter_rows() if v is not None and v > 0.0}
    classes = _entity_class_names(source)
    built: dict[str, float] = {}
    for cls in ENTITY_CLASSES:
        built.update(_constant_values(source, cls, BUILD_YEAR))
    start_year = model_start_year(source)
    found: dict[str, str] = {}
    for e, age, y_life, y_end, method in gone.select("e", "age", "y_life", "y_end", "eol_method").iter_rows():
        if (e not in process_set and e not in node_set) or e not in with_existing:
            continue
        if not age:
            aged = ""
        elif e in built and start_year is not None:
            aged = f" and age {age:g} years (build_year {built[e]:g}, model.start_year {start_year:g})"
        else:
            aged = f" and age_at_model_start {age:g} years"
        if y_end < y0 - YEAR_TOL:
            where = "before the model start"
        elif y_end <= y0 + YEAR_TOL:
            where = "at the model start"
        else:
            where = f"which end_of_life_method {method} rounds to the model start"
        found[e] = (f"{classes.get(e, 'entity')}: the end of life (lifetime_technical {y_life:g} years{aged}) "
                    f"is at year {y_end:g} of the run, {where}")
    if found:
        run_warnings.warn_grouped("W-0521", list(found), per_run=True, details=found)


def existing_cohorts(source: "InputSource",
                     active_solve: str | None,
                     workdir: Path | None = None,
                     *,
                     provider: "object | None" = None,
                     years: Mapping[str, float] | None = None) -> pl.DataFrame:
    """The ``existing`` rows of the cohort table (:func:`run_existing_cohorts`, without the generations): the frame
    the run put on ``provider`` (:data:`._run_calendar.RUN_EXISTING_COHORT_KEY`), so every sub-solve of the run has
    the same existing cohort; built from ``source`` on the solve's run calendar when the Provider does not carry it
    (a solve built on its own)."""
    return _roots(_run_existing_frame(source, active_solve, workdir, provider=provider, years=years))


def existing_generations(source: "InputSource",
                         active_solve: str | None,
                         workdir: Path | None = None,
                         *,
                         provider: "object | None" = None,
                         years: Mapping[str, float] | None = None) -> pl.DataFrame:
    """The ``generation`` rows of the existing ``reinvest_automatic`` chains (:func:`run_existing_cohorts`), from the
    same frame as :func:`existing_cohorts`."""
    return _generations(_run_existing_frame(source, active_solve, workdir, provider=provider, years=years))


def _run_existing_frame(source: "InputSource",
                        active_solve: str | None,
                        workdir: Path | None,
                        *,
                        provider: "object | None",
                        years: Mapping[str, float] | None) -> pl.DataFrame:
    """The run's existing cohorts and their generations: the frame on ``provider``, else built on the solve's run
    calendar."""
    from flextool.engine.preprocess.params._run_calendar import RUN_EXISTING_COHORT_KEY
    if provider is not None and provider.has(RUN_EXISTING_COHORT_KEY):
        return cohort_frame_from_utf8(provider.get(RUN_EXISTING_COHORT_KEY))
    years = dict(years) if years is not None else cohort_years(source, active_solve, workdir, provider=provider)
    cal = cohort_calendar(_run_calendar_of(source, active_solve, years, provider), years) if years else None
    return _cohort_frame(_existing_rows(source, years, cal))


def _utf8_cohort_frame(rows: list[tuple]) -> pl.DataFrame:
    """The cohort frame of ``rows`` (column order of :func:`_cohort_schema`) with Utf8 labels (the Provider form)."""
    schema = {c: (pl.Utf8 if c in _AXIS_COLUMNS else t) for c, t in _cohort_schema().items()}
    if not rows:
        return pl.DataFrame(schema=schema)
    return pl.DataFrame(rows, schema=schema, orient="row")


def cohort_frame_from_utf8(frame: pl.DataFrame) -> pl.DataFrame:
    """A cohort frame read back from the Provider (or from its csv dump, whose column types are inferred): each
    column on its :func:`_cohort_schema` type, the axis columns on their axes."""
    schema = _cohort_schema()
    out = frame.lazy().select(
        *(pl.col(c).cast(pl.Utf8 if c in _AXIS_COLUMNS else t) for c, t in schema.items()))
    return out.pipe(rename_to_axis, _AXIS_COLUMNS).select(list(schema)).collect()


def _cohort_frame(rows: list[tuple]) -> pl.DataFrame:
    """The cohort frame of ``rows`` (column order of :func:`_cohort_schema`), axis columns on their axes."""
    schema = _cohort_schema()
    if not rows:
        return pl.DataFrame(schema=schema)
    return (_utf8_cohort_frame(rows).lazy()
              .pipe(rename_to_axis, _AXIS_COLUMNS)
              .select(list(schema))
              .collect())


def run_existing_frames(provider: object, years: Mapping[str, float],
                        period_in_use: list[str]) -> tuple[pl.DataFrame, pl.DataFrame] | None:
    """``(cohort, cohort_alive)`` of the run's existing cohorts on ``provider`` (with their generations) over
    ``period_in_use``: the cohort frames of a solve that builds none of its own (a dispatch solve); None without the
    run's existing cohorts."""
    from flextool.engine.preprocess.params._run_calendar import RUN_EXISTING_COHORT_KEY
    if provider is None or not provider.has(RUN_EXISTING_COHORT_KEY):
        return None
    cohort = cohort_frame_from_utf8(provider.get(RUN_EXISTING_COHORT_KEY))
    return cohort, _existing_alive(cohort, years, period_in_use)


def _existing_alive(cohort: pl.DataFrame, years: Mapping[str, float], period_in_use: list[str]) -> pl.DataFrame:
    """``cohort_alive`` rows of the existing cohorts: the periods of ``period_in_use`` with ``y[d] < y_eol``."""
    existing = cohort.filter(pl.col("kind") == KIND_EXISTING)
    if existing.height == 0 or not period_in_use:
        return pl.DataFrame(schema=_alive_schema())
    periods = (axis_lazyframe({"d": list(period_in_use)})
               .with_columns(y=pl.Series("y", [float(years.get(str(d), 0.0)) for d in period_in_use],
                                         dtype=pl.Float64)))
    return (existing.lazy()
              .select("cohort", "e", "kind", "d_build", "y_eol")
              .join(periods, how="cross")
              .filter(within_life(pl.col("y"), None, pl.col("y_eol")))
              .select(list(_alive_schema()))
              .collect())


def _invest_rows(source: "InputSource",
                 active_solve: str | None,
                 ed_invest: pl.DataFrame,
                 years: Mapping[str, float],
                 cal: _Calendar,
                 eol_methods: Mapping[str, str],
                 commission_lf: pl.LazyFrame | None,
                 ) -> tuple[pl.DataFrame, list[tuple], list[_ChainStart]]:
    """``(anchor, rows, chains)``: the decisions ``[e, d]`` of ``ed_invest`` of entities with a ``lifetime_method``,
    their cohort rows (:func:`_cohort_schema` order, Utf8 labels): the start of operation ``d_on`` (``commission_lf``'s
    ``d_c``; the decision period without a construction lead time) and the end of life of :func:`window_end` with
    the lifetime of the period ``d_on`` (the plant as constructed, :func:`._period_values.ed_values_online`), and the
    first generation of each ``reinvest_automatic`` decision (:func:`chain_rows`)."""
    methods = _entity_methods(source)
    with_life = axis_lazyframe({"e": [e for e, m in methods.items() if m in LIFETIME_METHODS]})
    anchor, with_life = align_join_dtypes(ed_invest.lazy().select("e", "d"), with_life, ["e"])
    anchor = anchor.join(with_life, on="e", how="semi").collect()
    if anchor.height == 0:
        return anchor, [], []
    online = (None if commission_lf is None
              else commission_lf.filter(pl.col("yr_c").is_finite()).select("e", "d", "d_c").collect())
    life = ed_values_online(source, LIFETIME_TECHNICAL, anchor.lazy(), None, online).collect()
    starts: dict[tuple[str, str], str] = {}
    if online is not None:
        for e, d, d_c in online.iter_rows():
            starts[(str(e), str(d))] = str(d_c)
    rows: list[tuple] = []
    chains: list[_ChainStart] = []
    for e, d, life_v in life.select("e", "d", "value").iter_rows():
        e, d = str(e), str(d)
        method = methods[e]
        eol_method = eol_methods.get(e, END_OF_LIFE_METHOD_DEFAULT)
        d_on = starts.get((e, d), d)
        if d_on not in years:
            raise CommissioningLagError(
                f"Investment of {e!r} decided in period {d!r} starts to operate in period {d_on!r}, which has no "
                f"year in the solve {active_solve!r}: its end of life cannot be placed.")
        y_on = float(years[d_on])
        y_life, y_end, y_eol = window_end(y_on, float(life_v or 0.0), method, cal, eol_method)
        root = f"{KIND_INVEST}:{d}"
        rows.append((root, e, KIND_INVEST, method, eol_method, d, d_on, y_on, 0.0, y_life, y_end,
                     None, y_eol, cal.label(y_eol), *_NO_PRICES, root, None))
        if method == REINVEST_AUTOMATIC:
            chains.append(_ChainStart(root=root, e=e, method=method, eol_method=eol_method, d_build=d, d_on=d_on,
                                      y_on=y_on, age=0.0, life=life_v))
    return anchor, rows, chains


def never_operating_investments(source: "InputSource",
                                active_solve: str | None,
                                ed_invest: pl.DataFrame | None,
                                workdir: Path | None = None,
                                *,
                                provider: "object | None" = None,
                                commission_lf: pl.LazyFrame | None = None,
                                ) -> pl.DataFrame:
    """``[e, d]`` (Utf8): the investment decisions of ``ed_invest`` that would never operate, so the model does not
    offer them::

        never = { (e, d) : 0 < y_life < +inf,  y_eol <= y_on + YEAR_TOL }

    with ``y_on``, ``y_life`` and ``y_eol`` those of the decision's cohort (:func:`_invest_rows`, on the run's
    seams and the run's ``end_of_life_method``): a lifetime that ``previous_seam`` / ``closest_seam`` rounds to the
    decision's own start of operation (also after a construction lead time, ``commission_lf``).  A warning names
    them once per run and entity (:func:`_never_operates_warning`)."""
    empty = pl.DataFrame(schema={"e": pl.Utf8, "d": pl.Utf8})
    if ed_invest is None or ed_invest.height == 0:
        return empty
    years = cohort_years(source, active_solve, workdir, provider=provider)
    if not years:
        return empty
    cal = cohort_calendar(_run_calendar_of(source, active_solve, years, provider), years)
    eol_methods = eol_methods_of(existing_cohorts(source, active_solve, workdir, provider=provider, years=years))
    _anchor, rows, _chains = _invest_rows(source, active_solve, ed_invest, years, cal, eol_methods, commission_lf)
    never: dict[str, list[tuple[str, float, str]]] = {}
    for _c, e, _k, _m, eol_method, d, _d_on, y_on, _age, y_life, _y_end, _first, y_eol, _d_eol, *_prices in rows:
        if 0.0 < y_life < _INF and y_eol <= y_on + YEAR_TOL:
            never.setdefault(e, []).append((d, y_life, eol_method))
    if not never:
        return empty
    _never_operates_warning(source, never)
    pairs = [(e, d) for e, decisions in never.items() for d, _life, _m in decisions]
    return pl.DataFrame(pairs, schema={"e": pl.Utf8, "d": pl.Utf8}, orient="row")


def invest_cohorts(source: "InputSource",
                   active_solve: str | None,
                   ed_invest: pl.DataFrame,
                   period_in_use: list[str],
                   workdir: Path | None = None,
                   *,
                   provider: "object | None" = None,
                   years: Mapping[str, float] | None = None,
                   lineage: pl.DataFrame | None = None,
                   commission_lf: pl.LazyFrame | None = None,
                   eol_methods: Mapping[str, str] | None = None,
                   ) -> tuple[pl.DataFrame, pl.DataFrame]:
    """``(cohort rows, cohort_alive rows)`` of the investment decisions ``ed_invest`` (``[e, d]``) of entities with a
    ``lifetime_method``::

        y_on  = y[d_on],  d_on = the period the decision starts to operate in (``commission_lf``'s ``d_c``; the
                decision period without a construction lead time)
        y_life = lifetime_technical[e, d_on] (the plant as constructed; 0 when missing; +inf for
                 reinvest_automatic)
        y_end = y_on + y_life,   y_eol = round_end_of_life(y_end, end_of_life_method[e])
        alive = { d ∈ period_in_use : y_walk[d] >= yr_avail (the availability walk, along ``lineage``),
                                       y[d] < y_eol }

    The start of the window is the availability walk of :func:`._walks.period_walk_iterator`
    (``UNBOUNDED_FORWARD``, with the construction lead time of ``commission_lf`` and the scenario ``lineage``).
    ``eol_methods`` is each entity's ``end_of_life_method`` (:func:`eol_methods_of` the run's existing cohort; read
    from ``source`` when not given).  A decision that would never operate is not in ``ed_invest``: the investment
    set leaves it out (:func:`never_operating_investments`).  The cohort rows of a ``reinvest_automatic`` decision
    are followed by its generations (:func:`chain_rows`), which have no ``cohort_alive`` rows."""
    from flextool.engine.preprocess.params._walks import WindowMethod, period_walk_iterator
    empty = (pl.DataFrame(schema=_cohort_schema()), pl.DataFrame(schema=_alive_schema()))
    if ed_invest is None or ed_invest.height == 0 or not period_in_use:
        return empty
    years = dict(years) if years is not None else cohort_years(source, active_solve, workdir, provider=provider)
    cal = cohort_calendar(_run_calendar_of(source, active_solve, years, provider), years)
    if eol_methods is None:
        eol_methods = end_of_life_methods(source)
    anchor, rows, chains = _invest_rows(source, active_solve, ed_invest, years, cal, eol_methods, commission_lf)
    if anchor.height == 0:
        return empty
    _closest_seam_notice(source, rows, cal)
    cohort = _cohort_frame(rows)
    generations = _cohort_frame(chain_rows(source, chains, cal))
    walk = (period_walk_iterator(source, active_solve, anchor.lazy(), list(period_in_use), list(period_in_use),
                                 window_method=WindowMethod.UNBOUNDED_FORWARD, life_lf=None, factor_side=None,
                                 workdir=workdir, lineage=lineage, commission_lf=commission_lf)
              .pipe(rename_to_axis, {"d": "d_build", "d_all": "d"}))
    ends = cohort.lazy().select("e", "d_build", "cohort", "kind", "y_eol")
    walk, ends = align_join_dtypes(walk, ends, ["e", "d_build"])
    walk = walk.join(ends, on=["e", "d_build"], how="inner")
    walk, yrs = align_join_dtypes(walk, _years_frame(years), ["d"])
    alive = (walk.join(yrs, on="d", how="left")
                 .with_columns(y=pl.col("y").fill_null(0.0))
                 .filter(within_life(pl.col("y"), None, pl.col("y_eol")))
                 .select(list(_alive_schema()))
                 .collect())
    return pl.concat([cohort, generations], how="vertical"), alive


def _entity_class_names(source: "InputSource") -> dict[str, str]:
    """``{entity: class}`` of the units, nodes and connections (the class a message names)."""
    from flextool.engine.preprocess.params._existing import _entity_class_lf
    return {str(e): str(c) for e, c in _entity_class_lf(source).select("e", "ec").collect().iter_rows()}


def _never_operates_warning(source: "InputSource", never: Mapping[str, list[tuple[str, float, str]]]) -> None:
    """Warn, once per run and entity, about investment decisions ``never[e] = [(period, lifetime, method)]`` whose
    end of life is rounded to their own start of operation: such a decision would be in service in no period, so
    the investment set leaves it out."""
    if not never:
        return
    from flextool import run_warnings
    classes = _entity_class_names(source)
    found: dict[str, str] = {}
    for e, decisions in sorted(never.items()):
        periods = ", ".join(d for d, _life, _m in decisions)
        _d, life, method = decisions[0]
        found[e] = (f"{classes.get(e, 'entity')}: investments decided in {periods} (lifetime_technical {life:g} "
                    f"years, end_of_life_method {method})")
    run_warnings.warn_grouped("W-0522", list(found), per_run=True, details=found)


def _closest_seam_notice(source: "InputSource", rows: list[tuple], cal: _Calendar) -> None:
    """The ``end_of_life_closest_seam`` notice (an upgraded database whose marker names entities): the entities of
    the cohort ``rows`` (:func:`_cohort_schema` order, Utf8 labels) that the marker names, with a bounded
    ``lifetime_method``, ``end_of_life_method`` closest_seam and no explicit ``end_of_life_method`` (the new default
    rounds them), and an end of life that the default moves (:func:`._run_calendar.eol_moves`: closest_seam takes
    out of service a period next_seam keeps).  An existing cohort counts only with ``existing`` > 0 in some period;
    an investment cohort is any decision the solve offers.  Each entity is named once per run, with its moved
    cohorts in the detail."""
    from flextool.engine import _behaviour_changes as bc
    if not bc.has_marker(END_OF_LIFE_CLOSEST_SEAM):
        return
    named = {str(n) for n in bc.marker_json(END_OF_LIFE_CLOSEST_SEAM).get("entities") or []}
    if not named:
        return
    moved: dict[str, list[tuple[str, str, float, float]]] = {}
    for (_c, e, kind, method, eol_method, d_build, _d_on, _y_on, _age, _life, y_end, _first, y_eol, _d_eol,
         *_prices) in rows:
        if (e in named and method in BOUNDED_METHODS and eol_method == "closest_seam" and y_end is not None
                and eol_moves(y_end, cal.seams, cal.period_starts)):
            moved.setdefault(e, []).append((kind, str(d_build), float(y_end), float(y_eol)))
    if not moved:
        return
    explicit: set[str] = set()
    for cls in ENTITY_CLASSES:
        explicit |= source_entities_with(source, cls)(END_OF_LIFE_METHOD)
    if any(kind == KIND_EXISTING for cohorts in moved.values() for kind, *_ in cohorts):
        from flextool.engine.preprocess.params._existing import _per_entity_param_lf
        with_existing = {str(e) for e, v in _per_entity_param_lf(source, "existing").select("e", "value").collect()
                         .iter_rows() if v is not None and v > 0.0}
        moved = {e: [c for c in cohorts if c[0] != KIND_EXISTING or e in with_existing]
                 for e, cohorts in moved.items()}
    fresh = bc.unannounced(END_OF_LIFE_CLOSEST_SEAM, [e for e in sorted(moved) if moved[e] and e not in explicit])
    if not fresh:
        return

    def phrase(e: str, kind: str, d_build: str, y_end: float, y_eol: float) -> str:
        who = "existing" if kind == KIND_EXISTING else f"decided in {d_build}"
        nxt = cal.label(cal.eol(y_end, "next_seam"))
        before = f"from {nxt}" if nxt is not None else "in service to the end of the horizon"
        return (f"{e} ({who}): end of life at year {y_end:g} of the run, out of service from "
                f"{cal.label(y_eol) or f'year {y_eol:g}'} (next_seam: {before})")

    detail = "; ".join(phrase(e, *c) for e in fresh for c in moved[e]) + "."
    bc.notice(END_OF_LIFE_CLOSEST_SEAM, "entities", fresh, detail)


def seed_cohorts(seed: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    """``(cohort rows, cohort_alive rows)`` of an ``edd_invest`` seed ``[e, d_invest, d]`` (a sub-solve whose
    investment walk was not run): one cohort per decision, its years unknown (``y_on`` null, ``y_eol = +inf``) and
    its alive periods those of the seed."""
    keys = seed.select("e", "d_invest").unique(maintain_order=True)
    rows = [(f"{KIND_INVEST}:{d}", str(e), KIND_INVEST, None, None, str(d), None, None, None, None, None, None,
             _INF, None, *_NO_PRICES, f"{KIND_INVEST}:{d}", None)
            for e, d in keys.iter_rows()]
    cohort = _cohort_frame(rows)
    triples = [(f"{KIND_INVEST}:{di}", str(e), KIND_INVEST, str(di), str(d), None)
               for e, di, d in seed.select("e", "d_invest", "d").iter_rows()]
    alive = (pl.DataFrame(triples, schema={"cohort": pl.Utf8, "e": pl.Utf8, "kind": pl.Utf8, "d_build": pl.Utf8,
                                           "d": pl.Utf8, "y": pl.Float64}, orient="row")
               .lazy()
               .pipe(rename_to_axis, {"e": "e", "d_build": "d_build", "d": "d"})
               .collect())
    return cohort, alive


def forward_pairs(years: Mapping[str, float],
                  period_in_use: list[str],
                  *,
                  lineage: pl.DataFrame | None = None) -> pl.DataFrame:
    """``dd_forward`` ``[d_retire, d]``: ``d_retire`` any period with a year, ``d ∈ period_in_use`` with a year,
    ``y[d_retire] <= y[d]``; with ``lineage`` (a stochastic fan under recourse) only the pairs on one scenario
    lineage (the pairs with either end outside ``period_in_use`` pass)."""
    if not years or not period_in_use:
        return pl.DataFrame(schema=_forward_schema())
    from flextool.engine.preprocess.params._walks import _apply_lineage_filter, _assert_lineage_castable
    yrs = _years_frame(years)
    used = axis_lazyframe({"d": [str(d) for d in period_in_use]})
    used, yrs_used = align_join_dtypes(used, yrs, ["d"])
    pairs = (yrs.pipe(rename_to_axis, {"d": "d_retire"}).rename({"y": "y_retire"})
                .join(used.join(yrs_used, on="d", how="inner"), how="cross")
                .filter(pl.col("y_retire") <= pl.col("y")))
    if lineage is not None:
        d_dtype = pairs.collect_schema()["d"]
        _assert_lineage_castable(lineage, d_dtype)
        pairs = _apply_lineage_filter(pairs, lineage, [str(d) for d in period_in_use], d_dtype,
                                      anchor_col="d_retire", dall_col="d")
    return pairs.select("d_retire", "d").sort("d_retire", "d").collect()


def build_cohort_frames(source: "InputSource",
                        active_solve: str | None,
                        ed_invest: pl.DataFrame | None,
                        period_in_use: list[str],
                        workdir: Path | None = None,
                        *,
                        provider: "object | None" = None,
                        lineage: pl.DataFrame | None = None,
                        commission_lf: pl.LazyFrame | None = None,
                        seed: pl.DataFrame | None = None,
                        with_forward: bool = True,
                        ) -> CohortFrames:
    """The cohort frames of a solve: the existing cohorts, the investment cohorts of ``ed_invest`` (when their walk
    finds no alive period, the ``edd_invest`` ``seed`` when it has rows), their alive periods over
    ``period_in_use``, and ``dd_forward`` (``None`` with ``with_forward=False``: a solve that does not place
    retirements)."""
    years = cohort_years(source, active_solve, workdir, provider=provider)
    run_existing = _run_existing_frame(source, active_solve, workdir, provider=provider, years=years)
    existing = _roots(run_existing)
    inv_rows, inv_alive = invest_cohorts(source, active_solve, ed_invest, period_in_use, workdir, provider=provider,
                                         years=years, lineage=lineage, commission_lf=commission_lf,
                                         eol_methods=eol_methods_of(existing))
    seeded = inv_alive.height == 0 and seed is not None and seed.height > 0
    if seeded:
        inv_rows, inv_alive = seed_cohorts(seed)
    cohort = (pl.concat([existing, _generations(run_existing), inv_rows], how="vertical")
                .sort("e", "kind", "d_build", "y_on", nulls_last=False))
    alive = (pl.concat([_existing_alive(existing, years, period_in_use), inv_alive], how="vertical")
               .sort("e", "kind", "d_build", "d", nulls_last=False))
    forward = forward_pairs(years, period_in_use, lineage=lineage) if with_forward else None
    return CohortFrames(cohort=cohort, cohort_alive=alive, dd_forward=forward)


# ---------------------------------------------------------------------------
# Views
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ExistingEnd:
    """The end of life of an entity's existing capacity (:func:`existing_ends`)."""

    #: The end of life, unrounded (years of the run).
    y_end: float
    #: ``y_end`` rounded by the entity's ``end_of_life_method``: in service while ``y[d] < y_eol``.
    y_eol: float
    #: The run period that starts at ``y_eol`` (``None`` at the end of the horizon and at ``+inf``).
    d_eol: str | None
    #: The age of the existing capacity at the model start (years).
    age: float = 0.0


def existing_ends(cohort: pl.DataFrame) -> dict[str, ExistingEnd]:
    """``{entity: ExistingEnd}`` of the existing capacity of the ``reinvest_choice`` / ``no_reinvestment``
    entities (``reinvest_automatic`` capacity is renewed and has none)."""
    rows = (cohort.filter((pl.col("kind") == KIND_EXISTING) & pl.col("method").is_in(list(BOUNDED_METHODS))
                          & pl.col("y_end").is_not_null())
                  .select("e", "y_end", "y_eol", "d_eol", "age"))
    return {str(e): ExistingEnd(y_end=float(y), y_eol=_INF if y_eol is None else float(y_eol),
                                d_eol=None if d_eol is None else str(d_eol), age=float(age or 0.0))
            for e, y, y_eol, d_eol, age in rows.iter_rows()}


def existing_eol(cohort: pl.DataFrame) -> dict[str, float]:
    """``{entity: y_end}``: the end of life (unrounded) of the existing capacity of the ``reinvest_choice`` /
    ``no_reinvestment`` entities (``reinvest_automatic`` capacity is renewed and has none)."""
    rows = (cohort.filter((pl.col("kind") == KIND_EXISTING) & pl.col("method").is_in(list(BOUNDED_METHODS))
                          & pl.col("y_end").is_not_null())
                  .select("e", "y_end"))
    return {str(e): float(y) for e, y in rows.iter_rows()}


def _expired_where(cohort: pl.DataFrame, years: Mapping[str, float], periods: Iterable[str],
                   methods: Iterable[str], end_col: str) -> set[tuple[str, str]]:
    ends = (cohort.filter((pl.col("kind") == KIND_EXISTING) & pl.col("method").is_in(list(methods))
                          & pl.col(end_col).is_not_null())
                  .select("e", end_col))
    periods = [str(d) for d in periods]
    return {(str(e), d) for e, end in ends.iter_rows() for d in periods if years.get(d, 0.0) >= end}


def existing_expired(cohort: pl.DataFrame,
                     years: Mapping[str, float],
                     periods: Iterable[str],
                     methods: Iterable[str] = BOUNDED_METHODS) -> set[tuple[str, str]]:
    """``{(e, d)}``: the periods of ``periods`` past the end of life of the existing capacity
    (``y[d] >= y_eol``) of the entities whose ``lifetime_method`` is in ``methods``."""
    return _expired_where(cohort, years, periods, methods, "y_eol")


def no_reinvestment_closed(cohort: pl.DataFrame,
                           years: Mapping[str, float],
                           periods: Iterable[str]) -> set[tuple[str, str]]:
    """``{(e, d)}``: the periods of ``periods`` in which a ``no_reinvestment`` entity may no longer invest: at or
    after the end of the first life of its existing capacity (``y[d] >= y_first_life_end``, unrounded)."""
    return _expired_where(cohort, years, periods, (NO_REINVESTMENT,), "y_first_life_end")


def invest_alive(cohort_alive: pl.DataFrame | None) -> pl.DataFrame:
    """``[e, d_invest, d]``: the investment decisions and the periods their capacity is in service, sorted."""
    if cohort_alive is None:
        cohort_alive = pl.DataFrame(schema=_alive_schema())
    return (cohort_alive.lazy()
              .filter(pl.col("kind") == KIND_INVEST)
              .pipe(rename_to_axis, {"d_build": "d_invest"})
              .select("e", "d_invest", "d")
              .sort("e", "d_invest", "d")
              .collect())


def invest_prebuilt(cohort: pl.DataFrame | None, cohort_alive: pl.DataFrame | None) -> pl.DataFrame:
    """``[e, d_invest, d]``: the periods an investment's capacity is in service from the start of the period
    (alive and ``y[d] > y_on``: not starting to operate in ``d``), sorted."""
    if cohort is None or cohort_alive is None:
        return invest_alive(None)
    starts = cohort.lazy().filter(pl.col("kind") == KIND_INVEST).select("e", "cohort", "y_on")
    return (cohort_alive.lazy()
              .filter(pl.col("kind") == KIND_INVEST)
              .join(starts, on=["e", "cohort"], how="inner")
              .filter(pl.col("y") > pl.col("y_on"))
              .pipe(rename_to_axis, {"d_build": "d_invest"})
              .select("e", "d_invest", "d")
              .sort("e", "d_invest", "d")
              .collect())


def retire_active(pairs: pl.DataFrame | None,
                  cohort_alive: pl.DataFrame | None,
                  dd_forward: pl.DataFrame | None,
                  col: str = "e") -> pl.DataFrame:
    """``[col, d_divest, d]``: the periods ``d`` a retirement decided in ``d_divest`` (``pairs`` ``[col, d]``)
    counts in — ``(d_divest, d) ∈ dd_forward`` while the entity's existing capacity is alive in ``d`` — sorted
    ``(col, d_divest, d)``; the entity column first."""
    schema = {col: schema_dtype(None, col), "d_divest": schema_dtype(None, "d_divest"), "d": schema_dtype(None, "d")}
    if pairs is None or pairs.height == 0 or cohort_alive is None or dd_forward is None:
        return pl.DataFrame(schema=schema)
    live = get_global_axis_enums()
    if live is not None:
        pairs = cast_frame_axes(pairs, live)
    existing = (cohort_alive.lazy().filter(pl.col("kind") == KIND_EXISTING)
                .select(pl.col("e").alias(col), "d"))
    chosen = (pairs.lazy().select(col, "d").pipe(rename_to_axis, {"d": "d_retire"})
              .join(dd_forward.lazy(), on="d_retire", how="inner"))
    return (join_aligned(chosen, existing, on=[col, "d"], how="semi")
              .pipe(rename_to_axis, {"d_retire": "d_divest"})
              .select(col, "d_divest", "d")
              .sort(col, "d_divest", "d")
              .collect())


@dataclass(frozen=True)
class InService:
    """The periods each investment and each retirement of existing capacity counts in, from a solve's cohort
    frames (:func:`in_service`)."""

    #: ``[e, d_invest, d]``: the investments in service in ``d`` (:func:`invest_alive`).
    alive: pl.DataFrame
    #: ``[e, d_invest, d]``: the investments in service from the start of ``d`` (:func:`invest_prebuilt`).
    prebuilt: pl.DataFrame
    #: ``[p, d_divest, d]`` / ``[n, d_divest, d]``: the retirements of the process / storage-node options
    #: (``pd_divest_set`` / ``nd_divest_set``) counting in ``d`` (:func:`retire_active`).
    retired_p: pl.DataFrame
    retired_n: pl.DataFrame

    def retired(self, axis: str) -> pl.DataFrame:
        """:attr:`retired_p` (``axis="p"``) or :attr:`retired_n` (``axis="n"``)."""
        return self.retired_p if axis == "p" else self.retired_n


def in_service(data: object) -> InService:
    """The :class:`InService` frames of ``data`` (a ``FlexData``: ``cohort``, ``cohort_alive``, ``dd_forward``,
    ``pd_divest_set``, ``nd_divest_set``).  A solve without ``dd_forward`` (a nested sub-solve, which places no
    retirement reach) has no pre-built capacity and no retirement either; every frame is empty, never ``None``."""
    cohort = getattr(data, "cohort", None)
    alive_rows = getattr(data, "cohort_alive", None)
    forward = getattr(data, "dd_forward", None)
    prebuilt = invest_prebuilt(cohort, alive_rows) if forward is not None else invest_alive(None)
    return InService(
        alive=invest_alive(alive_rows),
        prebuilt=prebuilt,
        retired_p=retire_active(getattr(data, "pd_divest_set", None), alive_rows, forward, "p"),
        retired_n=retire_active(getattr(data, "nd_divest_set", None), alive_rows, forward, "n"))


def as_names(frame: pl.DataFrame, columns: Iterable[str]) -> pl.DataFrame:
    """``frame`` with the label columns ``columns`` as plain strings (``Series.to_list`` of the axis labels, never a
    cast of an axis Enum): the cohort tables of the outputs are keyed by names."""
    return frame.with_columns(pl.Series(c, frame[c].to_list(), dtype=pl.Utf8) for c in columns)


__all__ = [
    "within_life",
    "BOUNDED_METHODS",
    "as_names",
    "CohortFrames",
    "ExistingEnd",
    "KIND_EXISTING",
    "KIND_EXTEND",
    "KIND_GENERATION",
    "KIND_INVEST",
    "LIFETIME_METHODS",
    "YEAR_COLUMNS",
    "build_cohort_frames",
    "chain_rows",
    "cohort_calendar",
    "cohort_frame_from_utf8",
    "cohort_years",
    "existing_cohorts",
    "end_of_life_methods",
    "end_of_life_years",
    "eol_methods_of",
    "existing_ages",
    "existing_ends",
    "existing_eol",
    "existing_generations",
    "existing_expired",
    "InService",
    "forward_pairs",
    "in_service",
    "invest_alive",
    "invest_cohorts",
    "invest_prebuilt",
    "model_start_year",
    "no_reinvestment_closed",
    "retire_active",
    "run_cohort_calendar",
    "run_existing_cohorts",
    "run_existing_frames",
    "seed_cohorts",
    "window_end",
]
