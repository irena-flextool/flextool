"""The cohort table: every block of capacity of a solve, when it starts to operate and when its life ends.

A *cohort* is capacity that came into service together: the pre-existing capacity of an entity (the user's
``existing``, one cohort per entity counted from the start of the run) or the capacity of one investment decision
``v_invest[e, d_build]``.  Each cohort has one start of operation ``y_on`` and one end of life ``y_eol``; it is in
service in the periods ``d`` with ``y_on <= y[d] < y_eol``.  Every question "is this capacity still there in
period d" is answered from these frames, so the lifetime rule lives in one place.

Three frames (``FlexData.cohort``, ``cohort_alive``, ``dd_forward``):

* ``cohort`` — one row per cohort: ``cohort`` (Utf8 id, unique per entity: ``existing:<run start period>`` /
  ``invest:<decision period>``; ``existing`` when the solve has no period years), ``e`` (entity-union axis),
  ``kind`` (``existing`` / ``invest``), ``method`` (``lifetime_method`` with its default), ``d_build`` (the
  decision period, the key of ``v_invest``; null for existing capacity), ``d_on`` (Utf8: the period the capacity
  starts to operate in: the run start for existing capacity, the decision period or, with a construction lead
  time, the later period of ``ed_invest_commission``), ``y_on`` (start of operation), ``y_life``
  (``lifetime_technical``; ``+inf`` for ``reinvest_automatic``), ``y_end = y_on + y_life`` (the end of life,
  unrounded), ``y_first_life_end`` (existing capacity: the end of its first life, the year a ``no_reinvestment``
  entity stops investing; null for an investment), ``y_eol`` (``y_end`` rounded to the next seam, ``+inf`` when
  it lies past the last seam) and ``d_eol`` (Utf8: the run period that starts at ``y_eol``; null at ``+inf`` and
  at the end of the horizon).
* ``cohort_alive`` — ``[cohort, e, kind, d_build, d, y]``: the periods ``d`` of the solve's ``period_in_use`` in
  which a cohort is in service, ``y`` the year of ``d``.
* ``dd_forward`` — ``[d_retire, d]``: a retirement decided in ``d_retire`` counts in ``d`` when
  ``y[d_retire] <= y[d]`` (along the scenario lineage of a stochastic fan).  Entity-independent: a retirement of
  existing capacity counts in ``d`` when ``(d_retire, d) ∈ dd_forward`` and the existing cohort is alive in ``d``.

Years are years of the run: a solve's period years (``p_years_d``, which covers the period history) on the run
calendar (:func:`cohort_years`, :func:`._run_calendar.run_years`: a period takes its run-calendar year, a branch
period its anchor's, never a year shifted by an offset).  The seams are those years, the start of the run, and the
run calendar's period starts and horizon end.  Rounding to the next seam with no tolerance keeps every comparison
exact: for a period year ``y`` (itself a seam), ``y < Y ⟺ y < next_seam(Y)``.  So a cohort placed by one solve
(the existing cohort, built once for the run: :func:`run_existing_cohorts`; an investment realized by an earlier
solve, carried in the handoff ``realized_cohort``) is in service in a later solve's period ``d`` by the same
comparison of ``d``'s run year with the carried ``y_on`` / ``y_eol``, with no rounding repeated.  Where two solves
give a period different years (the run calendar's ``conflicts``), the run calendar's year (the first solve's)
decides.

Views (the only lifetime-gate code): :func:`existing_eol`, :func:`existing_expired`,
:func:`no_reinvestment_closed`, :func:`invest_alive`, :func:`invest_prebuilt`, :func:`retire_active`.

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
from flextool.engine.input._lifetime_roles import LIFETIME_TECHNICAL, NO_REINVESTMENT
from flextool.engine.input._solve_state import CommissioningLagError
from flextool.engine.preprocess.params._period_values import _ed_period_value_map, ed_values_on

if TYPE_CHECKING:
    from flextool.engine.input._input_source import InputSource
    from flextool.engine.preprocess.params._run_calendar import RunCalendar

KIND_EXISTING = "existing"
KIND_INVEST = "invest"
REINVEST_AUTOMATIC = "reinvest_automatic"
REINVEST_CHOICE = "reinvest_choice"
#: The ``lifetime_method`` values; an entity with another value has no cohort.
LIFETIME_METHODS: tuple[str, ...] = (REINVEST_AUTOMATIC, REINVEST_CHOICE, NO_REINVESTMENT)
#: The methods whose capacity ends its life (``reinvest_automatic`` capacity is renewed).
BOUNDED_METHODS: tuple[str, ...] = (REINVEST_CHOICE, NO_REINVESTMENT)
#: The rounding of an end of life to a seam (the next seam, compared exactly).
_EOL_ROUNDING = "next_seam"
_EOL_TOL = 0.0

_INF = float("inf")


def _cohort_schema() -> dict[str, pl.DataType]:
    return {"cohort": pl.Utf8, "e": schema_dtype(None, "e"), "kind": pl.Utf8, "method": pl.Utf8,
            "d_build": schema_dtype(None, "d"), "d_on": pl.Utf8, "y_on": pl.Float64, "y_life": pl.Float64,
            "y_end": pl.Float64, "y_first_life_end": pl.Float64, "y_eol": pl.Float64, "d_eol": pl.Utf8}


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
    """The years and seams of the run, and the run period label of a seam."""

    years: dict[str, float]
    seams: list[float]
    label: Callable[[float], str | None]

    def eol(self, y_end: float) -> float:
        from flextool.engine.preprocess.params._run_calendar import snap_to_seam
        return snap_to_seam(y_end, _EOL_ROUNDING, self.seams, tol=_EOL_TOL)


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
    """``{period: run year}`` the cohort table compares: the solve's period years (:func:`._derived._p_years_d_lf`:
    with the Provider, the solve's ``p_years_d`` over its period history; without it, the solve's own
    ``years_represented``) on the run calendar (:func:`._run_calendar.run_years`: a period takes its run-calendar
    year, a branch period its anchor's).  A solve that is not handed the run history (its years count from its own
    first period) so compares in the years of the run, where the cohorts of earlier solves were placed."""
    from flextool.engine.preprocess.params._derived import _p_years_d_lf
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

    return _Calendar(years=dict(years), seams=seams, label=label)


def _entity_methods(source: "InputSource") -> dict[str, str]:
    """``{entity: lifetime_method}`` (default ``reinvest_automatic``) of every unit, node and connection."""
    from flextool.engine.preprocess.params._existing import _all_entities_lf, _lifetime_method_with_default_lf
    df = _lifetime_method_with_default_lf(source, _all_entities_lf(source)).select("e", "method").collect()
    out: dict[str, str] = {}
    for e, m in df.iter_rows():
        out.setdefault(str(e), str(m))
    return out


def _years_frame(years: Mapping[str, float]) -> pl.LazyFrame:
    """``[d, y]`` of ``years`` on the period axis."""
    return (axis_lazyframe({"d": list(years)})
            .with_columns(y=pl.Series("y", [float(v) for v in years.values()], dtype=pl.Float64)))


def window_end(y_on: float, y_life: float, method: str, cal: _Calendar) -> tuple[float, float, float]:
    """``(y_life, y_end, y_eol)`` of capacity starting to operate at ``y_on`` with the lifetime ``y_life`` under
    ``lifetime_method`` ``method``: ``reinvest_automatic`` capacity is renewed (``+inf`` each); otherwise
    ``y_end = y_on + y_life`` and ``y_eol = next_seam(y_end)``.  The one lifetime-window rule: the capacity is in
    service in the periods with ``y_on <= y[d] < y_eol``."""
    if method == REINVEST_AUTOMATIC:
        return _INF, _INF, cal.eol(_INF)
    y_end = y_on + y_life
    return y_life, y_end, cal.eol(y_end)


def _existing_rows(source: "InputSource", years: Mapping[str, float], cal: _Calendar) -> list[tuple]:
    """The rows (:func:`_cohort_schema` order, Utf8 labels) of the existing cohorts on the years ``years``."""
    from flextool.engine.preprocess.params._derived import _entity_classes_lookup, run_start_period
    start = run_start_period(dict(years))
    if start is None:
        return [(KIND_EXISTING, e, KIND_EXISTING, method, None, None, None, None, None, None, _INF, None)
                for e, method in _entity_methods(source).items() if method in LIFETIME_METHODS]
    lifetime = _ed_period_value_map(source, LIFETIME_TECHNICAL, [start])
    process_set, node_set, _ = _entity_classes_lookup(source)
    y_on = years[start]
    rows: list[tuple] = []
    for e, method in _entity_methods(source).items():
        if method not in LIFETIME_METHODS:
            continue
        if method != REINVEST_AUTOMATIC and e not in process_set and e not in node_set:
            y_life, y_end, y_eol = 0.0, 0.0, cal.eol(0.0)
        else:
            y_life, y_end, y_eol = window_end(y_on, lifetime.get((e, start), 0.0), method, cal)
        rows.append((f"{KIND_EXISTING}:{start}", e, KIND_EXISTING, method, None, start, y_on, y_life, y_end, y_end,
                     y_eol, cal.label(y_eol)))
    return rows


def run_existing_cohorts(source: "InputSource", rc: "RunCalendar | None") -> pl.DataFrame:
    """The ``existing`` rows of the cohort table of the whole run (Utf8 labels, run years): one per entity with a
    ``lifetime_method``, counted from the start of the run ``d0`` (:func:`._derived.run_start_period` of the run
    calendar)::

        y_on = y[d0],   y_life = lifetime_technical[e, d0] (0 when missing; +inf for reinvest_automatic)
        y_end = y_first_life_end = y_on + y_life,   y_eol = next_seam(y_end)

    The existing cohort is fixed when the run starts: ``_run_model`` builds this frame once and puts it on every
    sub-solve's Provider (:data:`._run_calendar.RUN_EXISTING_COHORT_KEY`); :func:`existing_cohorts` reads it there.
    An entity that is neither a process nor a node has no lifetime: its end of life is 0.  Without period years
    (``rc`` ``None`` or empty) the end of life cannot be placed: every row has the cohort id ``existing``, null
    years and ``y_eol = +inf`` (in service in every period, no end of life)."""
    years = dict(rc.years) if rc is not None else {}
    cal = cohort_calendar(rc, years) if rc is not None else None
    return _utf8_cohort_frame(_existing_rows(source, years, cal))


def existing_cohorts(source: "InputSource",
                     active_solve: str | None,
                     workdir: Path | None = None,
                     *,
                     provider: "object | None" = None,
                     years: Mapping[str, float] | None = None) -> pl.DataFrame:
    """The ``existing`` rows of the cohort table (:func:`run_existing_cohorts`): the frame the run put on
    ``provider`` (:data:`._run_calendar.RUN_EXISTING_COHORT_KEY`), so every sub-solve of the run has the same
    existing cohort; built from ``source`` on the solve's run calendar when the Provider does not carry it (a solve
    built on its own)."""
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
              .filter(pl.col("y") < pl.col("y_eol"))
              .select(list(_alive_schema()))
              .collect())


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
                   ) -> tuple[pl.DataFrame, pl.DataFrame]:
    """``(cohort rows, cohort_alive rows)`` of the investment decisions ``ed_invest`` (``[e, d]``) of entities with a
    ``lifetime_method``::

        y_on  = y[d_on],  d_on = the period the decision starts to operate in (``commission_lf``'s ``d_c``; the
                decision period without a construction lead time)
        y_life = lifetime_technical[e, d_build] (0 when missing; +inf for reinvest_automatic)
        y_end = y_on + y_life,   y_eol = next_seam(y_end)
        alive = { d ∈ period_in_use : y_walk[d] >= yr_avail (the availability walk, along ``lineage``),
                                       y[d] < y_eol }

    The start of the window is the availability walk of :func:`._walks.period_walk_iterator`
    (``UNBOUNDED_FORWARD``, with the construction lead time of ``commission_lf`` and the scenario ``lineage``)."""
    from flextool.engine.preprocess.params._walks import WindowMethod, period_walk_iterator
    empty = (pl.DataFrame(schema=_cohort_schema()), pl.DataFrame(schema=_alive_schema()))
    if ed_invest is None or ed_invest.height == 0 or not period_in_use:
        return empty
    years = dict(years) if years is not None else cohort_years(source, active_solve, workdir, provider=provider)
    cal = cohort_calendar(_run_calendar_of(source, active_solve, years, provider), years)
    methods = _entity_methods(source)
    with_life = axis_lazyframe({"e": [e for e, m in methods.items() if m in LIFETIME_METHODS]})
    anchor, with_life = align_join_dtypes(ed_invest.lazy().select("e", "d"), with_life, ["e"])
    anchor = anchor.join(with_life, on="e", how="semi").collect()
    if anchor.height == 0:
        return empty
    life = ed_values_on(source, LIFETIME_TECHNICAL, anchor.lazy(), fill=0.0).collect()
    starts: dict[tuple[str, str], str] = {}
    if commission_lf is not None:
        for e, d, d_c in (commission_lf.filter(pl.col("yr_c").is_finite())
                                       .select("e", "d", "d_c").collect().iter_rows()):
            starts[(str(e), str(d))] = str(d_c)
    rows: list[tuple] = []
    for e, d, life_v in life.select("e", "d", "value").iter_rows():
        e, d = str(e), str(d)
        method = methods[e]
        d_on = starts.get((e, d), d)
        if d_on not in years:
            raise CommissioningLagError(
                f"Investment of {e!r} decided in period {d!r} starts to operate in period {d_on!r}, which has no "
                f"year in the solve {active_solve!r}: its end of life cannot be placed.")
        y_on = float(years[d_on])
        y_life, y_end, y_eol = window_end(y_on, float(life_v), method, cal)
        rows.append((f"{KIND_INVEST}:{d}", e, KIND_INVEST, method, d, d_on, y_on, y_life, y_end, None, y_eol,
                     cal.label(y_eol)))
    cohort = _cohort_frame(rows)
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
                 .filter(pl.col("y") < pl.col("y_eol"))
                 .select(list(_alive_schema()))
                 .collect())
    return cohort, alive


def seed_cohorts(seed: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    """``(cohort rows, cohort_alive rows)`` of an ``edd_invest`` seed ``[e, d_invest, d]`` (a sub-solve whose
    investment walk was not run): one cohort per decision, its years unknown (``y_on`` null, ``y_eol = +inf``) and
    its alive periods those of the seed."""
    keys = seed.select("e", "d_invest").unique(maintain_order=True)
    rows = [(f"{KIND_INVEST}:{d}", str(e), KIND_INVEST, None, str(d), None, None, None, None, None, _INF, None)
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
    existing = existing_cohorts(source, active_solve, workdir, provider=provider, years=years)
    inv_rows, inv_alive = invest_cohorts(source, active_solve, ed_invest, period_in_use, workdir, provider=provider,
                                         years=years, lineage=lineage, commission_lf=commission_lf)
    seeded = inv_alive.height == 0 and seed is not None and seed.height > 0
    if seeded:
        inv_rows, inv_alive = seed_cohorts(seed)
    cohort = pl.concat([existing, inv_rows], how="vertical").sort("e", "kind", "d_build", nulls_last=False)
    alive = (pl.concat([_existing_alive(existing, years, period_in_use), inv_alive], how="vertical")
               .sort("e", "kind", "d_build", "d", nulls_last=False))
    forward = forward_pairs(years, period_in_use, lineage=lineage) if with_forward else None
    return CohortFrames(cohort=cohort, cohort_alive=alive, dd_forward=forward)


# ---------------------------------------------------------------------------
# Views
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# The cohorts carried between the solves of a run
# ---------------------------------------------------------------------------

#: Cohort-table column → its ``realized_cohort`` carrier column (the carrier speaks the handoff's Utf8 names).
#: Any other column of the cohort table is carried under its own name.
_CARRIER_NAMES: dict[str, str] = {"e": "entity", "d_build": "period_decided", "d_on": "period_online",
                                  "d_eol": "period_eol"}
#: The order of the cohort kinds in the carrier (the order the capacity of an entity is summed in).
_KIND_ORDER: dict[str, int] = {KIND_EXISTING: 0, KIND_INVEST: 1}


def realized_cohort_schema(cohort_schema: Mapping[str, pl.DataType] | None = None) -> dict[str, pl.DataType]:
    """The columns of the ``realized_cohort`` carrier: ``entity`` and one per column of the cohort table
    (``cohort_schema``, default :func:`_cohort_schema`; renamed by :data:`_CARRIER_NAMES`; a float column Float64,
    every other Utf8), then ``size`` (MW, MWh for storage)."""
    schema = _cohort_schema() if cohort_schema is None else dict(cohort_schema)
    out: dict[str, pl.DataType] = {"entity": pl.Utf8}
    for c, t in schema.items():
        out[_CARRIER_NAMES.get(c, c)] = pl.Float64 if t.is_float() else pl.Utf8
    out["size"] = pl.Float64
    return out


#: The columns of the ``realized_retire`` carrier: MW (MWh) retired from a cohort in a realized decision period.
REALIZED_RETIRE_SCHEMA: dict[str, pl.DataType] = {"entity": pl.Utf8, "cohort": pl.Utf8, "period": pl.Utf8,
                                                  "value": pl.Float64}


def _cell(value: object, dtype: pl.DataType) -> object:
    if value is None:
        return None
    return float(value) if dtype == pl.Float64 else str(value)


def realized_cohorts(prior_cohort: pl.DataFrame | None,
                     prior_retire: pl.DataFrame | None,
                     cohort: pl.DataFrame | None,
                     invested: Mapping[tuple[str, str], float],
                     retired: Mapping[tuple[str, str], float],
                     unitsize: Mapping[str, float],
                     ) -> tuple[pl.DataFrame | None, pl.DataFrame | None]:
    """``(realized_cohort, realized_retire)``: the cohorts a solve hands to the later solves of the run, and the
    capacity retired from them, both chain-cumulative.

    ``realized_cohort`` — the rows of ``prior_cohort`` (the carrier the solve was handed), unchanged, and one row per
    investment decision ``(e, d) ∈ invested`` (``v_invest`` of a realized, real-named decision period, > 0) with
    the columns of its row of the solve's cohort table ``cohort`` (run years, never recomputed by a later solve) and
    ``size = v_invest · unitsize[e]``.  A decision already carried (the same cohort realized again) adds its size to
    the carried row.  Sorted by entity, kind, start of operation (stable: earlier rows first).

    ``realized_retire`` ``[entity, cohort, period, value]`` — the rows of ``prior_retire`` and
    ``retired[e, d] · unitsize[e]`` (``v_divest`` of a realized decision period, > 0) of the existing cohort of
    ``e``, summed per (entity, cohort, period); sorted, zero rows left out.  A retirement of another cohort is a
    row with that cohort's id.

    Each frame is ``None`` when it has no row.  An investment without a row in ``cohort`` cannot be placed in a
    later solve's periods and raises :class:`ValueError`."""
    by_key: dict[tuple[str, str], dict] = {}
    columns = list(cohort.columns) if cohort is not None else list(_cohort_schema())
    schema = realized_cohort_schema(cohort.schema if cohort is not None else None)
    if prior_cohort is not None and prior_cohort.height > 0:
        for row in prior_cohort.iter_rows(named=True):
            rec = {c: _cell(row.get(c), t) for c, t in schema.items()}
            by_key[(rec["entity"], rec["cohort"])] = rec
    if invested:
        if cohort is None:
            raise ValueError("Realized investments cannot be handed to the later solves: the solve has no cohort "
                             "table.")
        rows = {(str(r["e"]), str(r["cohort"])): r for r in cohort.iter_rows(named=True)}
        for (e, d), v in invested.items():
            key = (e, f"{KIND_INVEST}:{d}")
            size = v * unitsize.get(e, 1.0)
            if key in by_key:
                by_key[key]["size"] = by_key[key]["size"] + size
                continue
            row = rows.get(key)
            if row is None or row.get("y_on") is None:
                raise ValueError(f"The investment of {e!r} decided in period {d!r} has no start of operation in the "
                                 "solve's cohort table: it cannot be handed to the later solves.")
            rec = {_CARRIER_NAMES.get(c, c): _cell(row[c], schema[_CARRIER_NAMES.get(c, c)]) for c in columns}
            rec["size"] = size
            by_key[key] = rec
    ordered = sorted(by_key.values(),
                     key=lambda r: (r["entity"], _KIND_ORDER.get(r["kind"], len(_KIND_ORDER)), r["y_on"]))
    cohort_out = (pl.DataFrame([[r[c] for c in schema] for r in ordered], schema=schema, orient="row")
                  if ordered else None)

    existing_of: dict[str, str] = {}
    if cohort is not None:
        for e, c in cohort.filter(pl.col("kind") == KIND_EXISTING).select("e", "cohort").iter_rows():
            existing_of.setdefault(str(e), str(c))
    sums: dict[tuple[str, str, str], float] = {}
    if prior_retire is not None and prior_retire.height > 0:
        for e, c, d, v in prior_retire.select("entity", "cohort", "period", "value").iter_rows():
            key = (str(e), str(c), str(d))
            sums[key] = sums.get(key, 0.0) + float(v)
    for (e, d), v in retired.items():
        key = (e, existing_of.get(e, KIND_EXISTING), d)
        sums[key] = sums.get(key, 0.0) + v * unitsize.get(e, 1.0)
    retire_rows = [(e, c, d, v) for (e, c, d), v in sorted(sums.items()) if v != 0.0]
    retire_out = (pl.DataFrame(retire_rows, schema=REALIZED_RETIRE_SCHEMA, orient="row")
                  if retire_rows else None)
    return cohort_out, retire_out


@dataclass(frozen=True)
class ExistingCapacity:
    """The capacity a solve starts from, per entity and period of ``period_in_use`` (MW, MWh for storage): the
    existing cohort and the cohorts earlier solves of the run realized (:func:`existing_capacity`)."""

    #: ``pre[e, d]``: the existing cohort (the user's ``existing`` of period d, × virtual_unitsize) in service in d.
    pre: dict[tuple[str, str], float]
    #: ``carried[e, d]``: the carried investment cohorts in service in d.
    carried: dict[tuple[str, str], float]
    #: ``all[e, d] = pre + carried − retired_all`` (``retired_all`` subtracted for the entities that may retire).
    all: dict[tuple[str, str], float]
    #: ``retired[e, d]``: the retirements of the existing cohort that earlier solves realized and still count in d.
    retired: dict[tuple[str, str], float]
    #: ``invested[e]``: every carried investment of e, in service or not.
    invested: dict[str, float]


def _in_service(y: float, y_on: float | None, y_eol: float) -> bool:
    return (y_on is None or y >= y_on) and y < y_eol


def existing_capacity(entities: Iterable[str],
                      periods: Iterable[str],
                      years: Mapping[str, float],
                      existing: pl.DataFrame,
                      existing_mw: Mapping[tuple[str, str], float],
                      carried: pl.DataFrame | None,
                      retire: pl.DataFrame | None,
                      divest_entities: Iterable[str],
                      ) -> ExistingCapacity:
    """The capacity in service at the start of a solve, from the cohorts (one formula for every solve)::

        alive_c(d)    ⟺  y_on_c ≤ y[d] < y_eol_c                          (years of the run)
        pre[e, d]     =  existing_mw[e, d]  if the existing cohort of e is alive in d, else 0
        carried[e, d] =  Σ_{carried invest cohorts c of e alive in d} size_c
        retired_c[e, d] = Σ_{(c, d′, v) ∈ retire : alive_c(d), y[d′] ≤ y[d] or d′ before the solve} v
        all[e, d]     =  (pre[e, d] + carried[e, d]) − [e may retire] · Σ_c retired_c[e, d]
        invested[e]   =  Σ_{carried invest cohorts c of e} size_c

    ``existing`` are the existing cohorts of the run (:func:`run_existing_cohorts`, ``y_on`` / ``y_eol``),
    ``existing_mw[e, d]`` the user's ``existing`` of each period in MW (the map is read in every solve),
    ``carried`` / ``retire`` the handoff ``realized_cohort`` / ``realized_retire``.  The sums add in the order of
    the carried rows (by start of operation), ``all`` starting from ``pre``.  ``retired`` is the existing cohort's
    part (the bound of the retirements of existing capacity); ``all`` subtracts the retirements of every cohort.
    A retirement of a cohort that is neither the existing cohort nor a carried one raises :class:`ValueError`."""
    from flextool.engine.preprocess.params._retire_existing import retired_carried
    entities = [str(e) for e in entities]
    periods = [str(d) for d in periods]
    y_of = {d: float(years.get(d, 0.0)) for d in periods}
    ends: dict[tuple[str, str], tuple[float | None, float]] = {}
    existing_of: dict[str, str] = {}
    for e, c, y_on, y_eol in existing.select("e", "cohort", "y_on", "y_eol").iter_rows():
        existing_of.setdefault(str(e), str(c))
        ends[(str(e), str(c))] = (None, _INF if y_eol is None else float(y_eol))
    sizes: dict[str, list[tuple[str, float]]] = {}
    if carried is not None and carried.height > 0:
        for e, c, y_on, y_eol, size in carried.select("entity", "cohort", "y_on", "y_eol", "size").iter_rows():
            e, c = str(e), str(c)
            ends[(e, c)] = (None if y_on is None else float(y_on), _INF if y_eol is None else float(y_eol))
            sizes.setdefault(e, []).append((c, float(size)))
    alive = {(e, c, d): _in_service(y_of[d], *ends[(e, c)]) for (e, c) in ends for d in periods}

    rows = [] if retire is None else [(str(e), str(c), str(d), float(v)) for e, c, d, v in
                                      retire.select("entity", "cohort", "period", "value").iter_rows()]
    unknown = sorted({(e, c) for e, c, _d, _v in rows if (e, c) not in ends})
    if unknown:
        raise ValueError(f"Carried retirements name cohorts the solve does not have: {unknown}.")
    dead = {k for k, v in alive.items() if not v}
    retired_all = retired_carried(rows, years, periods, dead)
    retired = retired_carried([r for r in rows if existing_of.get(r[0]) == r[1]], years, periods, dead)

    divest = frozenset(str(e) for e in divest_entities)
    pre: dict[tuple[str, str], float] = {}
    carried_mw: dict[tuple[str, str], float] = {}
    all_mw: dict[tuple[str, str], float] = {}
    for e in entities:
        c0 = existing_of.get(e)
        for d in periods:
            v_pre = existing_mw.get((e, d), 0.0) if c0 is not None and alive[(e, c0, d)] else 0.0
            v_all = v_pre
            v_carried = 0.0
            for c, size in sizes.get(e, ()):
                if alive[(e, c, d)]:
                    v_all += size
                    v_carried += size
            if e in divest:
                v_all -= retired_all.get((e, d), 0.0)
            pre[(e, d)], carried_mw[(e, d)], all_mw[(e, d)] = v_pre, v_carried, v_all
    invested: dict[str, float] = {}
    for e, cohorts in sizes.items():
        total = 0.0
        for _c, size in cohorts:
            total += size
        invested[e] = total
    return ExistingCapacity(pre=pre, carried=carried_mw, all=all_mw, retired=retired, invested=invested)


__all__ = [
    "BOUNDED_METHODS",
    "CohortFrames",
    "ExistingCapacity",
    "KIND_EXISTING",
    "KIND_INVEST",
    "LIFETIME_METHODS",
    "REALIZED_RETIRE_SCHEMA",
    "YEAR_COLUMNS",
    "build_cohort_frames",
    "cohort_calendar",
    "cohort_frame_from_utf8",
    "cohort_years",
    "existing_cohorts",
    "existing_capacity",
    "existing_eol",
    "existing_expired",
    "InService",
    "forward_pairs",
    "in_service",
    "invest_alive",
    "invest_cohorts",
    "invest_prebuilt",
    "no_reinvestment_closed",
    "realized_cohort_schema",
    "realized_cohorts",
    "retire_active",
    "run_existing_cohorts",
    "seed_cohorts",
    "window_end",
]
