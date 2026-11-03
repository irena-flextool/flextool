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

#: Provider key of the run calendar frame ``[solve, period, yr, years_represented]``.
RUN_CALENDAR_KEY = "solve_data/run_period_calendar"

_SCHEMA = {"solve": pl.Utf8, "period": pl.Utf8, "yr": pl.Float64, "years_represented": pl.Float64}

#: Two years closer than this are the same year (cumulative sums of fractional years).
YEAR_TOL = 1e-9


@dataclass(frozen=True)
class RunCalendar:
    """Years of the periods of a model run.

    ``years`` maps each period to its start (years from the start of the run), in run order.  ``horizon_end`` is
    the end of the last period (``None`` when no period has a length).  ``conflicts`` lists
    ``(period, solve_a, yr_a, solve_b, yr_b)`` where two solves give a period different years.
    """

    years: dict[str, float]
    horizon_end: float | None
    conflicts: list[tuple[str, str, float, str, float]] = field(default_factory=list)
    last_period: str | None = None

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
    first_solve: dict[str, str] = {}
    conflicts: list[tuple[str, str, float, str, float]] = []
    horizon_end: float | None = None
    last_period: str | None = None
    for solve, period, yr, rep in df.select("solve", "period", "yr", "years_represented").iter_rows():
        period, yr, rep = str(period), float(yr), float(rep or 0.0)
        if period not in years:
            years[period] = yr
            first_solve[period] = str(solve)
        elif abs(years[period] - yr) > YEAR_TOL:
            conflicts.append((period, first_solve[period], years[period], str(solve), yr))
        end = yr + rep
        if horizon_end is None or end > horizon_end + YEAR_TOL:
            horizon_end = end
            last_period = period
    return RunCalendar(years=years, horizon_end=horizon_end, conflicts=conflicts, last_period=last_period)


def read_run_calendar(provider) -> RunCalendar | None:
    """The run calendar from ``provider``; ``None`` when the provider does not carry it."""
    if provider is None or not provider.has(RUN_CALENDAR_KEY):
        return None
    return run_calendar_from_frame(provider.get(RUN_CALENDAR_KEY))


def solve_calendar(years: Mapping[str, float], years_represented: Mapping[str, float] | None) -> RunCalendar:
    """A :class:`RunCalendar` of one solve's own periods (when no run calendar is available): the horizon ends
    where the solve's last period ends, or at its last period start when the solve gives no period lengths."""
    ordered = dict(sorted(((str(d), float(y)) for d, y in years.items()), key=lambda kv: kv[1]))
    horizon_end: float | None = None
    last_period: str | None = None
    for d, y in ordered.items():
        rep = (years_represented or {}).get(d)
        if rep is None:
            continue
        if horizon_end is None or y + float(rep) > horizon_end:
            horizon_end, last_period = y + float(rep), d
    return RunCalendar(years=ordered, horizon_end=horizon_end, last_period=last_period)


__all__ = [
    "RUN_CALENDAR_KEY",
    "RunCalendar",
    "YEAR_TOL",
    "build_run_calendar",
    "read_run_calendar",
    "run_calendar_from_frame",
    "solve_calendar",
]
