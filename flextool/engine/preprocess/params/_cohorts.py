"""The cohort table: every block of capacity of a solve, when it starts to operate and when its life ends.

A *cohort* is capacity that came into service together: the pre-existing capacity of an entity (the user's
``existing``, one cohort per entity counted from the start of the run) or the capacity of one investment decision
``v_invest[e, d_build]``.  Each cohort has one start of operation ``y_on`` and one end of life ``y_eol``; it is in
service in the periods ``d`` with ``y_on <= y[d] < y_eol``.  Every question "is this capacity still there in
period d" is answered from these frames, so the lifetime rule lives in one place.

Three frames (``FlexData.cohort``, ``cohort_alive``, ``dd_forward``):

* ``cohort`` — one row per cohort: ``cohort`` (Utf8 id, unique per entity: ``existing:<run start period>`` /
  ``invest:<decision period>``; ``existing`` when the solve has no period years), ``e`` (entity-union axis),
  ``kind`` (``existing`` / ``invest``), ``method`` (``lifetime_method`` with its default), ``d_build`` (the decision period, the key of ``v_invest``; null for
  existing capacity), ``y_on`` (start of operation), ``y_life`` (``lifetime_technical``; ``+inf`` for
  ``reinvest_automatic``), ``y_end = y_on + y_life`` (the end of life, unrounded), ``y_first_life_end`` (existing
  capacity: the end of its first life, the year a ``no_reinvestment`` entity stops investing; null for an
  investment), ``y_eol`` (``y_end`` rounded to the next seam, ``+inf`` when it lies past the last seam) and
  ``d_eol`` (Utf8: the run period that starts at ``y_eol``; null at ``+inf`` and at the end of the horizon).
* ``cohort_alive`` — ``[cohort, e, kind, d_build, d, y]``: the periods ``d`` of the solve's ``period_in_use`` in
  which a cohort is in service, ``y`` the year of ``d``.
* ``dd_forward`` — ``[d_retire, d]``: a retirement decided in ``d_retire`` counts in ``d`` when
  ``y[d_retire] <= y[d]`` (along the scenario lineage of a stochastic fan).  Entity-independent: a retirement of
  existing capacity counts in ``d`` when ``(d_retire, d) ∈ dd_forward`` and the existing cohort is alive in ``d``.

Years are the solve's years from the start of the run (``p_years_d``, which covers the period history).  The
seams are those years, the start of the run, and the run calendar's period starts and horizon end (shifted onto
the solve's years).  Rounding to the next seam with no tolerance keeps every comparison exact: for a period year
``y`` (itself a seam), ``y < Y ⟺ y < next_seam(Y)``.

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
            "d_build": schema_dtype(None, "d"), "y_on": pl.Float64, "y_life": pl.Float64, "y_end": pl.Float64,
            "y_first_life_end": pl.Float64, "y_eol": pl.Float64, "d_eol": pl.Utf8}


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
    #: True when the investment cohorts are those of the ``edd_invest`` seed (:func:`seed_cohorts`).
    seeded: bool = False


@dataclass(frozen=True)
class _Calendar:
    """The solve's years and seams, and the run period label of a seam."""

    years: dict[str, float]
    seams: list[float]
    label: Callable[[float], str | None]

    def eol(self, y_end: float) -> float:
        from flextool.engine.preprocess.params._run_calendar import snap_to_seam
        return snap_to_seam(y_end, _EOL_ROUNDING, self.seams, tol=_EOL_TOL)


def cohort_years(source: "InputSource",
                 active_solve: str | None,
                 workdir: Path | None,
                 *,
                 provider: "object | None" = None) -> dict[str, float]:
    """``{period: year}`` the cohort table compares: :func:`._derived._p_years_d_lf` (with the Provider, the solve's
    ``p_years_d``: years from the start of the run over its period history; without it, the solve's own
    ``years_represented``, the solve being the run)."""
    from flextool.engine.preprocess.params._derived import _p_years_d_lf
    pyd_lf = _p_years_d_lf(source, active_solve, workdir, provider=provider)
    if pyd_lf is None:
        return {}
    out: dict[str, float] = {}
    for d, y in pyd_lf.select("d", "yr").collect().iter_rows():
        out.setdefault(str(d), float(y if y is not None else 0.0))
    return out


def _calendar(source: "InputSource",
              active_solve: str | None,
              years: Mapping[str, float],
              provider: "object | None") -> _Calendar:
    """The seams of the solve: its period years, the start of the run (0) and, when the solve's years sit on the run
    calendar at one offset, the run calendar's period starts and horizon end shifted onto them (a run seam within
    ``YEAR_TOL`` of a solve year is that year).  Without a run calendar the solve's own periods and
    ``years_represented`` give it.  ``label(y)`` is the run period that starts at the seam ``y``."""
    from flextool.engine.preprocess.params._existing import _solve_years_represented
    from flextool.engine.preprocess.params._run_calendar import (
        YEAR_TOL,
        read_run_calendar,
        solve_calendar,
        solve_year_offset,
    )
    own = sorted({*(float(y) for y in years.values()), 0.0})
    rc = read_run_calendar(provider)
    if rc is None:
        rc = solve_calendar(years, _solve_years_represented(source, active_solve))
    offset, _mismatch = solve_year_offset(rc, years)
    seams = list(own)
    if offset is not None:
        for s in rc.seams():
            y = s - offset
            if all(abs(y - o) > YEAR_TOL for o in own):
                seams.append(y)
    seams.sort()

    def label(y: float) -> str | None:
        if y == _INF:
            return None
        if offset is not None:
            return rc.period_at(y + offset)
        for d, yd in years.items():
            if abs(yd - y) <= YEAR_TOL:
                return d
        return None

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


def existing_cohorts(source: "InputSource",
                     active_solve: str | None,
                     workdir: Path | None = None,
                     *,
                     provider: "object | None" = None,
                     years: Mapping[str, float] | None = None) -> pl.DataFrame:
    """The ``existing`` rows of the cohort table: one per entity with a ``lifetime_method``, counted from the start
    of the run ``d0`` (:func:`._derived.run_start_period`)::

        y_on = y[d0],   y_life = lifetime_technical[e, d0] (0 when missing; +inf for reinvest_automatic)
        y_end = y_first_life_end = y_on + y_life,   y_eol = next_seam(y_end)

    An entity that is neither a process nor a node has no lifetime: its end of life is 0.  Without period years
    the end of life cannot be placed: every row has the cohort id ``existing``, null years and ``y_eol = +inf``
    (in service in every period, no end of life)."""
    from flextool.engine.preprocess.params._derived import _entity_classes_lookup, run_start_period
    years = dict(years) if years is not None else cohort_years(source, active_solve, workdir, provider=provider)
    start = run_start_period(years)
    if start is None:
        return _cohort_frame([(KIND_EXISTING, e, KIND_EXISTING, method, None, None, None, None, None, _INF, None)
                              for e, method in _entity_methods(source).items() if method in LIFETIME_METHODS])
    cal = _calendar(source, active_solve, years, provider)
    lifetime = _ed_period_value_map(source, LIFETIME_TECHNICAL, [start])
    process_set, node_set, _ = _entity_classes_lookup(source)
    y_on = years[start]
    rows: list[tuple] = []
    for e, method in _entity_methods(source).items():
        if method not in LIFETIME_METHODS:
            continue
        if method == REINVEST_AUTOMATIC:
            y_life, y_end = _INF, _INF
        elif e not in process_set and e not in node_set:
            y_life, y_end = 0.0, 0.0
        else:
            y_life = lifetime.get((e, start), 0.0)
            y_end = y_on + y_life
        y_eol = cal.eol(y_end)
        rows.append((f"{KIND_EXISTING}:{start}", e, KIND_EXISTING, method, None, y_on, y_life, y_end, y_end,
                     y_eol, cal.label(y_eol)))
    return _cohort_frame(rows)


def _cohort_frame(rows: list[tuple]) -> pl.DataFrame:
    """The cohort frame of ``rows`` (column order of :func:`_cohort_schema`), axis columns on their axes."""
    schema = _cohort_schema()
    if not rows:
        return pl.DataFrame(schema=schema)
    raw = pl.DataFrame(rows, schema={c: (pl.Utf8 if c in ("e", "d_build") else t) for c, t in schema.items()},
                       orient="row")
    return (raw.lazy()
               .pipe(rename_to_axis, {"e": "e", "d_build": "d_build"})
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
    cal = _calendar(source, active_solve, years, provider)
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
        if method == REINVEST_AUTOMATIC:
            y_life, y_end = _INF, _INF
        else:
            y_life = float(life_v)
            y_end = y_on + y_life
        y_eol = cal.eol(y_end)
        rows.append((f"{KIND_INVEST}:{d}", e, KIND_INVEST, method, d, y_on, y_life, y_end, None, y_eol,
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
    rows = [(f"{KIND_INVEST}:{d}", str(e), KIND_INVEST, None, str(d), None, None, None, None, _INF, None)
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
    return CohortFrames(cohort=cohort, cohort_alive=alive, dd_forward=forward, seeded=seeded)


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


__all__ = [
    "BOUNDED_METHODS",
    "CohortFrames",
    "KIND_EXISTING",
    "KIND_INVEST",
    "LIFETIME_METHODS",
    "build_cohort_frames",
    "cohort_years",
    "existing_cohorts",
    "existing_eol",
    "existing_expired",
    "forward_pairs",
    "invest_alive",
    "invest_cohorts",
    "invest_prebuilt",
    "no_reinvestment_closed",
    "retire_active",
    "seed_cohorts",
]
