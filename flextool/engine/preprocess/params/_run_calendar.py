"""The run calendar: the year of every period of the whole model run.

A construction lead time moves the start of operation of an investment to a later period.  That period may lie
beyond the deciding solve's own periods (a look-ahead the solve does not hold, or the end of the model), so the
start of operation is rounded on the periods of the whole run, not on those of one solve.

:func:`build_run_calendar` collects ``(solve, period, yr, years_represented)`` from the per-solve period history
that ``solve._run_model`` accumulates (``yr`` = years from the start of the run; the same numbers the solve's
``period_with_history`` carries).  ``_run_model`` puts the frame under :data:`RUN_CALENDAR_KEY` in every sub-solve
provider.  :func:`read_run_calendar` turns it into a :class:`RunCalendar`: one year per period (the first solve
that holds the period gives it), the end of the model horizon (the latest period end), and the periods on whose
year two solves disagree.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Mapping

import polars as pl

from flextool.engine.input._lifetime_roles import END_OF_LIFE_METHODS

#: Provider key of the run calendar frame ``[solve, period, yr, years_represented]``.
RUN_CALENDAR_KEY = "solve_data/run_period_calendar"
#: Provider key of the existing cohorts of the run (:func:`._cohorts.run_existing_cohorts`: Utf8 labels, run
#: years), built once when the run starts and put on every sub-solve's Provider beside the run calendar.
RUN_EXISTING_COHORT_KEY = "solve_data/run_existing_cohort"

_SCHEMA = {"solve": pl.Utf8, "period": pl.Utf8, "yr": pl.Float64, "years_represented": pl.Float64}

#: Two years closer than this are the same year (cumulative sums of fractional years).
YEAR_TOL = 1e-9


@dataclass(frozen=True)
class RunCalendar:
    """Years of the periods of a model run.

    ``years`` maps each period to its start (years from the start of the run), in run order.  ``horizon_end`` is
    the end of the last period (``None`` when no period has a length).  ``conflicts`` lists
    ``(period, solve_a, yr_a, solve_b, yr_b)`` where two solves give a period different years.
    ``years_represented`` maps each period to the years it represents (given by the same solve as its year).
    """

    years: dict[str, float]
    horizon_end: float | None
    conflicts: list[tuple[str, str, float, str, float]] = field(default_factory=list)
    last_period: str | None = None
    years_represented: dict[str, float] = field(default_factory=dict)

    def seams(self) -> list[float]:
        """The period starts and the end of the horizon, sorted, each year once."""
        out: list[float] = []
        for y in sorted([*self.years.values(), *([self.horizon_end] if self.horizon_end is not None else [])]):
            if not out or y - out[-1] > YEAR_TOL:
                out.append(y)
        return out

    def period_at(self, yr: float) -> str | None:
        """The first period (in run order) that starts at ``yr``; ``None`` when no period does."""
        for d, y in self.years.items():
            if abs(y - yr) <= YEAR_TOL:
                return d
        return None


def snap_to_seam(t_star: float, method: str, seams: list[float], tol: float = YEAR_TOL) -> float:
    """The seam (a period start, or the end of the horizon) that the year ``t_star`` rounds to with ``method``.

    * ``previous_seam`` — the last seam ``<= t_star + tol``;
    * ``next_seam``     — the first seam ``>= t_star - tol``;
    * ``closest_seam``  — the nearer of the two, a tie (within ``tol``) goes to the later seam.

    ``+inf`` when the method finds no seam (``next_seam`` past the last seam, or ``t_star = +inf``).  ``tol`` makes
    two years closer than it the same seam; ``0.0`` compares exactly."""
    lo = max((s for s in seams if s <= t_star + tol), default=None)
    hi = min((s for s in seams if s >= t_star - tol), default=None)
    if method == "previous_seam":
        return lo if lo is not None else float("inf")
    if method == "next_seam":
        return hi if hi is not None else float("inf")
    # closest_seam: the nearer one; a tie goes to the later seam.
    if hi is None:
        return lo if lo is not None else float("inf")
    if lo is None:
        return hi
    return lo if (t_star - lo) < (hi - t_star) - tol else hi


def round_end_of_life(y_end: float, method: str, seams: list[float], tol: float = YEAR_TOL) -> float:
    """The end of life ``y_eol`` of capacity whose life ends at the year ``y_end``, rounded to a seam by the
    ``end_of_life_method`` ``method`` (:func:`snap_to_seam`, two years within ``tol`` the same seam)::

        +inf                              if y_end = +inf (reinvest_automatic: renewed, never ends)
        seams[0]                          if y_end < seams[0] − tol (ended before the model start)
        snap_to_seam(y_end, method)       otherwise

    Past the last seam ``next_seam`` gives ``+inf`` and ``previous_seam`` / ``closest_seam`` the last seam (the end
    of the horizon): in service to the end under all three.  The capacity is in service in the periods with
    ``y[d] < y_eol``."""
    if method not in END_OF_LIFE_METHODS:
        raise ValueError(f"Unknown end_of_life_method {method!r}: use one of {', '.join(END_OF_LIFE_METHODS)}.")
    if y_end == float("inf"):
        return y_end
    if seams and y_end < seams[0] - tol:
        return seams[0]
    return snap_to_seam(y_end, method, seams, tol)


def solve_year_offset(rc: RunCalendar, years: Mapping[str, float]) -> tuple[float | None, str | None]:
    """``(offset, mismatch)``: the constant ``rc.years[d] - years[d]`` over the periods of ``years`` (a solve's own
    year calendar) that are on the run calendar ``rc``.  A solve that is not handed the run history counts years
    from its own first period, so the offset is that period's run year.  ``(None, None)`` when no period of
    ``years`` is on the run calendar; ``(None, d)`` when the offset of period ``d`` differs from the first one by
    more than :data:`YEAR_TOL`."""
    offsets = {d: rc.years[d] - y for d, y in years.items() if d in rc.years}
    if not offsets:
        return None, None
    offset = next(iter(offsets.values()))
    for d, off in offsets.items():
        if abs(off - offset) > YEAR_TOL:
            return None, d
    return offset, None


def run_years(years: Mapping[str, float], rc: RunCalendar | None) -> dict[str, float]:
    """``{period: run year}`` of a solve's periods ``years`` (``{period: year}`` on the solve's own calendar).

    A period on the run calendar ``rc`` takes its run year.  A period that is not (a branch period of a stochastic
    fan) takes the run year of a period of ``years`` that is on the run calendar and has the same solve year within
    :data:`YEAR_TOL` (its anchor period); a period with no such match keeps its own year.  No year is shifted by an
    offset, so a run year compares exactly with the run-calendar seams.  Without a run calendar the solve is the
    run: ``years`` unchanged."""
    if rc is None:
        return {str(d): float(y) for d, y in years.items()}
    on_rc = [(float(y), rc.years[str(d)]) for d, y in years.items() if str(d) in rc.years]
    out: dict[str, float] = {}
    for d, y in years.items():
        d = str(d)
        if d in rc.years:
            out[d] = rc.years[d]
            continue
        match = next((ry for ys, ry in on_rc if abs(ys - float(y)) <= YEAR_TOL), None)
        out[d] = float(y) if match is None else match
    return out


def build_run_calendar(period_history: Mapping[str, list], solves: Iterable[str]) -> pl.DataFrame:
    """``[solve, period, yr, years_represented]`` of every period of every solve in ``solves``.

    ``period_history[solve]`` is the solve's ``(period, years_represented)`` list (history first, then the
    solve's own periods), as ``_run_model`` builds it for ``period_with_history``; ``yr`` is the running sum of
    the years represented before the period.
    """
    rows: list[tuple[str, str, float, float]] = []
    for solve in solves:
        cum = 0.0
        for period, years in period_history.get(solve, []):
            try:
                rep = float(years)
            except (TypeError, ValueError):
                rep = 0.0
            rows.append((str(solve), str(period), cum, rep))
            cum += rep
    if not rows:
        return pl.DataFrame(schema=_SCHEMA)
    return pl.DataFrame(rows, schema=_SCHEMA, orient="row")


def run_calendar_from_frame(df: pl.DataFrame) -> RunCalendar | None:
    """The :class:`RunCalendar` of a :func:`build_run_calendar` frame; ``None`` for an empty frame."""
    if df is None or df.height == 0:
        return None
    years: dict[str, float] = {}
    represented: dict[str, float] = {}
    first_solve: dict[str, str] = {}
    conflicts: list[tuple[str, str, float, str, float]] = []
    horizon_end: float | None = None
    last_period: str | None = None
    for solve, period, yr, rep in df.select("solve", "period", "yr", "years_represented").iter_rows():
        period, yr, rep = str(period), float(yr), float(rep or 0.0)
        if period not in years:
            years[period] = yr
            represented[period] = rep
            first_solve[period] = str(solve)
        elif abs(years[period] - yr) > YEAR_TOL:
            conflicts.append((period, first_solve[period], years[period], str(solve), yr))
        end = yr + rep
        if horizon_end is None or end > horizon_end + YEAR_TOL:
            horizon_end = end
            last_period = period
    return RunCalendar(years=years, horizon_end=horizon_end, conflicts=conflicts, last_period=last_period,
                       years_represented=represented)


def read_run_calendar(provider) -> RunCalendar | None:
    """The run calendar from ``provider``; ``None`` when the provider does not carry it."""
    if provider is None or not provider.has(RUN_CALENDAR_KEY):
        return None
    return run_calendar_from_frame(provider.get(RUN_CALENDAR_KEY))


def solve_calendar(years: Mapping[str, float], years_represented: Mapping[str, float] | None) -> RunCalendar:
    """A :class:`RunCalendar` of one solve's own periods (when no run calendar is available): the horizon ends
    where the solve's last period ends.  A period without a ``years_represented`` value (every period, when the
    solve gives none) represents one year, as the run calendar counts it (``solve._run_model``), so the seams are
    the period starts and the end of the horizon in every case."""
    ordered = dict(sorted(((str(d), float(y)) for d, y in years.items()), key=lambda kv: kv[1]))
    represented = dict(years_represented or {})
    horizon_end: float | None = None
    last_period: str | None = None
    for d, y in ordered.items():
        rep = float(represented.get(d, 1.0))
        if horizon_end is None or y + rep > horizon_end:
            horizon_end, last_period = y + rep, d
    return RunCalendar(years=ordered, horizon_end=horizon_end, last_period=last_period)


__all__ = [
    "RUN_CALENDAR_KEY",
    "RUN_EXISTING_COHORT_KEY",
    "RunCalendar",
    "YEAR_TOL",
    "build_run_calendar",
    "read_run_calendar",
    "round_end_of_life",
    "run_calendar_from_frame",
    "run_years",
    "snap_to_seam",
    "solve_calendar",
    "solve_year_offset",
]
