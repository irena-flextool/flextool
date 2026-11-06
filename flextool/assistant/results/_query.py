"""Querying a result table: filters, statistics and aggregation on the wide frame, and the aggregation guard.

A result table is stored wide (``flextool.lean_parquet``): row levels (period, time, …) as columns and one data
column per column key (unit and node, …).  A query never expands a large table to rows first (P0-D14):

1. filters on column levels choose the data columns from the footer, so only those are read;
2. filters on row levels and ``time_range`` drop rows of the wide frame;
3. the levels a query does not keep (``group_by``) are combined on the wide frame: column levels across the
   data columns of each kept key (``sum_horizontal`` …), row levels with ``group_by(...).agg``;
4. only the reduced frame is turned into rows (``to_long``), after the budget check
   (:data:`LONG_CELLS_MAX` values).

A table with time steps queried without ``group_by`` / ``agg`` gives one summary per series and period (mean,
min, max, count, the time of the maximum, and the sum where a sum over time is meaningful), largest ``|mean|``
first.

:func:`check_aggregation` refuses a sum that does not mean anything (P0 §5.4): power at each time step summed over
time is not energy (the step durations are not stored), annual values summed over periods are not a horizon
total, levels, averages, ratios and prices do not add up.
"""
from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from flextool.assistant.errors import ToolError, name_list, similar_text
from flextool.assistant.results import _catalog
from flextool.assistant.results._catalog import Measure
from flextool.lean_parquet import LeanLayout, lean_parquet_layout, read_lean_parquet_wide, to_long

#: The most values (rows × kept data columns) a query may turn into rows.
LONG_CELLS_MAX = 5_000_000

AGGS = ("sum", "mean", "min", "max", "count", "time_of_max")
SUMMARY_STATS = ("mean", "min", "max", "count", "sum", "time_of_max")
SCENARIO_LEVEL = "scenario"
#: Row levels that place a value in time; combining over them is combining "over time".
TIME_LEVELS = ("time", "solve")
PERIOD_LEVEL = "period"

#: Semantics whose default statistic is the mean (the others add up over something).
_MEAN_DEFAULT = frozenset({"instantaneous", "average", "level", "ratio", "price", "unknown"})

MEAN_OVER_TIME_NOTE = ("The mean over time assumes equal step lengths and equal weights (in models with "
                       "representative periods the steps carry different weights).")
PER_STEP_SUM_NOTE = ("The sum over time is the total of the period's time steps as sampled; it is not scaled to "
                     "a full year.")


def level_kind(level: str) -> str:
    """``time``, ``period`` or ``entity``: what combining over a level means."""
    if level in TIME_LEVELS:
        return "time"
    if level == PERIOD_LEVEL:
        return "period"
    return "entity"


# ---------------------------------------------------------------------------------------------------------
# The aggregation guard


def default_agg(semantics: str) -> str:
    return "mean" if semantics in _MEAN_DEFAULT else "sum"


def sum_problem(stem: str, semantics: str, over: set[str]) -> tuple[str, dict] | None:
    """``(message key, fields)`` when a sum of a *semantics* table over *over* (``time`` / ``period`` /
    ``entity``) is not meaningful, else ``None``."""
    counterpart = _catalog.counterpart(stem)
    use = f" Use the annual table {counterpart} instead." if counterpart else ""
    if semantics == "instantaneous":
        if "time" in over:
            return "sum_rate_over_time", {"use": use}
        if "period" in over:
            return "sum_rate_over_periods", {"use": use}
        return None
    if semantics in ("annualized", "count"):
        return ("sum_annual_over_periods", {}) if over & {"period", "time"} else None
    if semantics == "per_period":
        return ("sum_per_period_over_periods", {}) if over & {"period", "time"} else None
    if semantics in ("average", "level"):
        span = "time steps" if "time" in over else "periods"
        return ("sum_level", {"span": span}) if over & {"period", "time"} else None
    if semantics in ("ratio", "price"):
        return ("sum_ratio", {"what": "prices" if semantics == "price" else "ratios"}) if over else None
    if semantics in ("per_step", "discounted", "horizon"):
        return None
    return ("sum_unknown", {}) if over else None


def check_aggregation(stem: str, semantics: set[str], over: set[str], agg: str) -> list[str]:
    """Raise :class:`ToolError` when *agg* over *over* is not meaningful for a table of these *semantics*; else
    the notes the result carries."""
    notes: list[str] = []
    if agg == "sum":
        for sem in sorted(semantics):
            problem = sum_problem(stem, sem, over)
            if problem is not None:
                raise ToolError(problem[0], table=stem, **problem[1])
        if "time" in over and "per_step" in semantics:
            notes.append(PER_STEP_SUM_NOTE)
    if agg == "mean" and "time" in over:
        notes.append(MEAN_OVER_TIME_NOTE)
    return notes


# ---------------------------------------------------------------------------------------------------------
# Tables


@dataclass
class Table:
    stem: str
    path: Path
    layout: LeanLayout
    measures: list[Measure]
    membership: bool

    @property
    def row_levels(self) -> list[str]:
        return list(self.layout.idx)

    @property
    def column_levels(self) -> list[str]:
        return [c for c in self.layout.col if c != SCENARIO_LEVEL]

    @property
    def levels(self) -> list[str]:
        return self.row_levels + self.column_levels

    @property
    def has_time(self) -> bool:
        return any(level in TIME_LEVELS for level in self.row_levels)

    def measure_of(self) -> dict[str, Measure]:
        return dict(zip(self.layout.physical, self.measures))


def table_stems(folder: Path) -> list[str]:
    return sorted(p.stem for p in folder.glob("*.parquet") if p.is_file())


def _damaged() -> ToolError:
    return ToolError("result_damaged")


def open_table(folder: Path, stem: str, *, where: str) -> Table:
    """The table *stem* of a result folder (*where* names the folder in messages)."""
    path = folder / f"{stem}.parquet"
    if any(part in stem for part in ("/", "\\", "..")) or not path.is_file():
        raise ToolError("unknown_table", table=stem, run=where, similar=similar_text(stem, table_stems(folder)))
    try:
        layout = lean_parquet_layout(path)
    except OSError:
        raise _damaged() from None
    except Exception as exc:  # noqa: BLE001 — pyarrow.ArrowInvalid and friends: a file being written
        if type(exc).__module__.startswith("pyarrow"):
            raise _damaged() from None
        raise
    return Table(stem, path, layout, _catalog.measures(stem, layout), _catalog.membership_set(stem, layout))


def read_wide(table: Table, keep: list[str]):
    """The row levels (as text) and the data columns *keep* (as floats, NaN read as empty) of *table*."""
    import polars as pl

    try:
        wide = read_lean_parquet_wide(table.path, keep=keep, layout=table.layout)
    except OSError:
        raise _damaged() from None
    except Exception as exc:  # noqa: BLE001
        if type(exc).__module__.startswith("pyarrow"):
            raise _damaged() from None
        raise
    casts = [pl.col(level).cast(pl.String) for level in table.row_levels]
    if keep:
        casts.append(pl.col(keep).cast(pl.Float64, strict=False).fill_nan(None))
    return wide.with_columns(casts)


# ---------------------------------------------------------------------------------------------------------
# Queries


@dataclass(frozen=True)
class Query:
    filters: dict[str, list[str]] = field(default_factory=dict)
    time_range: tuple[str, str] | None = None
    group_by: tuple[str, ...] | None = None
    agg: str | None = None
    sort_by: str | None = None
    descending: bool = True
    top_n: int | None = None


def parse_query(args: dict) -> Query:
    filters: dict[str, list[str]] = {}
    for level, values in (args.get("filters") or {}).items():
        filters[str(level)] = [str(v) for v in (values if isinstance(values, list) else [values])]
    tr = args.get("time_range")
    time_range = (str(tr["from"]), str(tr["to"])) if tr else None
    group_by = tuple(str(g) for g in args["group_by"]) if args.get("group_by") is not None else None
    sort = args.get("sort") or {}
    return Query(filters=filters, time_range=time_range, group_by=group_by, agg=args.get("agg"),
                 sort_by=sort.get("by"), descending=bool(sort.get("descending", True)), top_n=args.get("top_n"))


def _matching(values: list[str], patterns: list[str]) -> list[str]:
    """The *values* that equal a pattern or match it with ``*`` standing for any text."""
    from fnmatch import fnmatchcase

    exact = {p for p in patterns if "*" not in p}
    globs = [p.replace("[", "[[]") for p in patterns if "*" in p]
    return [v for v in values if v in exact or any(fnmatchcase(v, g) for g in globs)]


def _check_level(table: Table, level: str, what: str) -> None:
    if level not in table.levels:
        raise ToolError("unknown_level", table=table.stem, level=level, what=what,
                        levels=", ".join(table.levels) or "(none)")


@dataclass
class Result:
    """The rows of a query (a polars frame), the keys among its columns and what was done."""

    frame: Any
    keys: list[str]
    value_columns: list[str]
    mode: str                         # "values", "aggregate" or "summary"
    agg: str | None
    kept: list[str]
    over: list[str]
    notes: list[str]
    semantics: set[str]
    units: dict[str, str]             # column label (or "*") → unit as written
    empty_dropped: int = 0


def _selected_columns(table: Table, q: Query, scenario: str | None) -> list[str]:
    layout = table.layout
    tests: dict[str, Callable[[str], bool]] = {}
    for level, patterns in q.filters.items():
        if level in table.column_levels:
            allowed = set(_matching(layout.level_values(level), patterns))
            if not allowed:
                raise ToolError("no_match", table=table.stem, level=level,
                                values=name_list(layout.level_values(level), 10))
            tests[level] = allowed.__contains__
    if scenario is not None and SCENARIO_LEVEL in layout.col:
        tests[SCENARIO_LEVEL] = scenario.__eq__
    keep = layout.select(tests)
    if not keep:
        raise ToolError("no_match_combination", table=table.stem)
    return keep


def _filter_rows(table: Table, wide, q: Query):
    import polars as pl

    for level, patterns in q.filters.items():
        if level in table.row_levels:
            values = wide[level].unique(maintain_order=True).to_list()
            allowed = _matching([v for v in values if v is not None], patterns)
            if not allowed:
                raise ToolError("no_match", table=table.stem, level=level, values=name_list(values, 10))
            wide = wide.filter(pl.col(level).is_in(allowed))
    if q.time_range is not None:
        if "time" not in table.row_levels:
            raise ToolError("no_time_steps", table=table.stem, what="time_range")
        times = wide["time"].unique(maintain_order=True).to_list()
        lo, hi = q.time_range
        missing = [t for t in (lo, hi) if t not in times]
        if missing:
            raise ToolError("bad_time_step", table=table.stem, value=missing[0],
                            first=times[0] if times else "-", last=times[-1] if times else "-")
        a, b = times.index(lo), times.index(hi)
        wide = wide.filter(pl.col("time").is_in(times[min(a, b):max(a, b) + 1]))
    return wide


def _units(table: Table, keep: list[str]) -> dict[str, str]:
    """``{"*": unit}`` when every kept column has one unit, else ``{column label: unit}``."""
    measure_of = table.measure_of()
    keys = dict(zip(table.layout.physical, table.layout.columns))
    by_label: dict[str, str] = {}
    for p in keep:
        by_label.setdefault(keys[p][-1], measure_of[p].unit)
    units = set(by_label.values())
    return {"*": units.pop()} if len(units) == 1 else by_label


def _budget(n: int) -> None:
    if n > LONG_CELLS_MAX:
        raise ToolError("budget", n=f"{n:,}".replace(",", " "))


def run_query(table: Table, q: Query, *, scenario: str | None = None, value_only: bool = False) -> Result:
    """The rows *q* asks for from *table*.  *scenario* keeps one value of the ``scenario`` column level (a
    comparison file); *value_only* turns the per-series summary of a time table into its default statistic (one
    value per key, as compare_runs needs)."""
    if table.membership:
        if q.group_by is not None or q.agg is not None:
            raise ToolError("membership_table", table=table.stem)
        return _membership(table, q)
    for level in q.filters:
        _check_level(table, level, "filters")
    for level in q.group_by or ():
        _check_level(table, level, "group_by")
    keep = _selected_columns(table, q, scenario)
    measure_of = table.measure_of()
    semantics = {measure_of[p].semantics for p in keep}
    units = _units(table, keep)
    if q.group_by is None and q.agg is None and not table.has_time:
        return _values(table, q, keep, semantics, units)
    if q.group_by is None and q.agg is None and not value_only:
        return _summary(table, q, keep, semantics, units)
    if q.group_by is None:
        kept = [level for level in table.levels if level_kind(level) != "time"]
    else:
        kept = list(q.group_by)
    agg = q.agg or default_agg(sorted(semantics)[0] if len(semantics) == 1 else "unknown")
    return _aggregate(table, q, keep, kept, agg, semantics, units)


def _membership(table: Table, q: Query) -> Result:
    import polars as pl

    for level in q.filters:
        if level not in table.layout.idx:
            raise ToolError("unknown_level", table=table.stem, level=level, what="filters",
                            levels=", ".join(table.layout.idx))
    wide = read_wide(table, [])
    for level, patterns in q.filters.items():
        values = wide[level].unique(maintain_order=True).to_list()
        wide = wide.filter(pl.col(level).is_in(_matching([v for v in values if v is not None], patterns)))
    keys = [c for c in wide.columns if c != SCENARIO_LEVEL]
    return Result(wide.select(keys), keys, [], "values", None, keys, [], [], set(), {})


def _values(table: Table, q: Query, keep: list[str], semantics: set[str], units: dict[str, str]) -> Result:
    import polars as pl

    _budget(table.layout.num_rows * len(keep))
    wide = _filter_rows(table, read_wide(table, keep), q).with_row_index("__r")
    long = to_long(wide, table.layout).sort("__r", maintain_order=True).drop("__r")
    if SCENARIO_LEVEL in long.columns:
        long = long.drop(SCENARIO_LEVEL)
    before = long.height
    long = long.filter(pl.col("value").is_not_null())
    keys = [c for c in long.columns if c != "value"]
    return Result(long, keys, ["value"], "values", None, keys, [], [], semantics, units,
                  empty_dropped=before - long.height)


def _varying(values) -> bool:
    return len(set(values)) > 1


def _aggregate(table: Table, q: Query, keep: list[str], kept: list[str], agg: str, semantics: set[str],
               units: dict[str, str]) -> Result:
    import polars as pl

    layout = table.layout
    if agg == "time_of_max" and not table.has_time:
        raise ToolError("no_time_steps", table=table.stem, what="time_of_max")
    wide = _filter_rows(table, read_wide(table, keep), q)
    keys_of = dict(zip(layout.physical, layout.columns))
    position = {level: i for i, level in enumerate(layout.col)}
    ck = [level for level in table.column_levels if level in kept]
    cd = [level for level in table.column_levels if level not in kept]
    rk = [level for level in table.row_levels if level in kept]
    rd = [level for level in table.row_levels if level not in kept]

    groups: dict[tuple[str, ...], list[str]] = {}
    for p in keep:
        groups.setdefault(tuple(keys_of[p][position[level]] for level in ck), []).append(p)
    over: set[str] = {level_kind(level) for level in rd if wide[level].n_unique() > 1}
    for level in cd:
        if _varying(keys_of[p][position[level]] for p in keep):
            over.add("entity")
    measure_of = table.measure_of()
    for cols in groups.values():
        group_units = {measure_of[p].unit for p in cols}
        if len(group_units) > 1:
            level = next(lv for lv in cd if _varying(keys_of[p][position[lv]] for p in cols))
            raise ToolError("mixed_units", table=table.stem, units=", ".join(sorted(u or "no unit" for u in
                                                                                    group_units)), level=level)
    notes = check_aggregation(table.stem, semantics, over, agg)

    names = [f"__g{i}" for i in range(len(groups))]
    horizontal: list[Any] = []
    for name, cols in zip(names, groups.values()):
        c = pl.col(cols)
        present = pl.any_horizontal(c.is_not_null())
        if agg in ("sum", "mean"):
            horizontal.append(pl.when(present).then(pl.sum_horizontal(cols)).alias(name + "|s"))
            horizontal.append(pl.sum_horizontal(c.is_not_null().cast(pl.Int64)).alias(name + "|n"))
        elif agg == "count":
            horizontal.append(pl.sum_horizontal(c.is_not_null().cast(pl.Int64)).alias(name + "|n"))
        elif agg == "min":
            horizontal.append(pl.min_horizontal(cols).alias(name))
        else:                                   # max, time_of_max
            horizontal.append(pl.max_horizontal(cols).alias(name))
    frame = wide.select([*(pl.col(level) for level in table.row_levels), *horizontal])

    extras: list[str] = []
    if agg == "time_of_max":
        extras = [level for level in rd if level_kind(level) != "entity"]
    final: list[Any] = []
    for name in names:
        if agg == "sum":
            s = pl.col(name + "|s").sum() if rd else pl.col(name + "|s")
            n = pl.col(name + "|n").sum() if rd else pl.col(name + "|n")
            final.append(pl.when(n > 0).then(s).alias(name))
        elif agg == "mean":
            s = pl.col(name + "|s").sum() if rd else pl.col(name + "|s")
            n = pl.col(name + "|n").sum() if rd else pl.col(name + "|n")
            final.append(pl.when(n > 0).then(s / n).alias(name))
        elif agg == "count":
            final.append((pl.col(name + "|n").sum() if rd else pl.col(name + "|n")).cast(pl.Float64).alias(name))
        elif agg == "min":
            final.append(pl.col(name).min() if rd else pl.col(name))
        else:
            final.append(pl.col(name).max() if rd else pl.col(name))
            for level in extras:
                final.append(pl.col(level).filter(pl.col(name) == pl.col(name).max()).first()
                             .alias(f"{name}|{level}"))
    if not rd:
        reduced = frame.select([*(pl.col(level) for level in rk), *final])
    elif rk:
        reduced = frame.group_by(rk, maintain_order=True).agg(final)
    else:
        reduced = frame.select(final)
    _budget(reduced.height * len(names))

    reduced = reduced.with_row_index("__r")
    long = reduced.unpivot(index=["__r", *rk], on=names, variable_name="__g", value_name="value")
    for level in extras:
        part = reduced.unpivot(index=["__r"], on=[f"{n}|{level}" for n in names], variable_name="__x",
                               value_name=f"{level}_of_max")
        long = long.with_columns(part[f"{level}_of_max"])
    long = long.sort("__r", maintain_order=True).drop("__r")    # row by row, the kept column keys in file order
    mapping = pl.DataFrame({"__g": names, **{level: [key[i] for key in groups] for i, level in enumerate(ck)}},
                           schema={"__g": pl.String, **{level: pl.String for level in ck}})
    long = long.join(mapping, on="__g", how="left", maintain_order="left").drop("__g")
    value_columns = ["value"] + [f"{level}_of_max" for level in extras]
    keys = [level for level in dict.fromkeys(kept) if level in rk or level in ck]     # in group_by order
    long = long.select([*keys, *value_columns])
    before = long.height
    long = long.filter(pl.col("value").is_not_null())
    return Result(long, keys, value_columns, "aggregate", agg, keys, sorted(over), notes, semantics, units,
                  empty_dropped=before - long.height)


def _summary(table: Table, q: Query, keep: list[str], semantics: set[str], units: dict[str, str]) -> Result:
    """One row per (series, period): mean, min, max, count, the time of the maximum, and the sum when a sum over
    time is meaningful for every kept column."""
    import polars as pl

    wide = _filter_rows(table, read_wide(table, keep), q)
    rk = [level for level in table.row_levels if level_kind(level) != "time"]
    tl = [level for level in table.row_levels if level_kind(level) == "time"]
    with_sum = all(sum_problem(table.stem, sem, {"time"}) is None for sem in semantics)
    stats = [s for s in SUMMARY_STATS if s != "sum" or with_sum]
    exprs: list[Any] = []
    for p in keep:
        c = pl.col(p)
        exprs += [c.mean().alias(f"{p}|mean"), c.min().alias(f"{p}|min"), c.max().alias(f"{p}|max"),
                  c.count().cast(pl.Float64).alias(f"{p}|count")]
        if with_sum:
            exprs.append(pl.when(c.count() > 0).then(c.sum()).alias(f"{p}|sum"))
        for level in tl:
            exprs.append(pl.col(level).filter(c == c.max()).first().alias(f"{p}|time_of_max|{level}"))
    reduced = wide.group_by(rk, maintain_order=True).agg(exprs) if rk else wide.select(exprs)
    _budget(reduced.height * len(keep))
    base = None
    value_columns: list[str] = []
    for stat in stats:
        parts = [stat] if stat != "time_of_max" else [f"time_of_max|{level}" for level in tl]
        for part in parts:
            column = stat if stat != "time_of_max" else f"{part.split('|', 1)[1]}_of_max"
            frame = reduced.select([*rk, *(pl.col(f"{p}|{part}").alias(p) for p in keep)])
            if stat == "time_of_max":
                frame = frame.with_columns(pl.col(keep).cast(pl.String))
            long = frame.unpivot(index=rk, on=keep, variable_name="__column", value_name=column)
            if base is None:
                keys_of = dict(zip(table.layout.physical, table.layout.columns))
                level_map = pl.DataFrame(
                    {"__column": keep, **{level: [keys_of[p][i] for p in keep]
                                          for i, level in enumerate(table.layout.col)}},
                    schema={"__column": pl.String, **{level: pl.String for level in table.layout.col}})
                base = long.join(level_map, on="__column", how="left", maintain_order="left").drop("__column")
            else:
                base = base.with_columns(long[column])
            value_columns.append(column)
    assert base is not None
    if SCENARIO_LEVEL in base.columns:
        base = base.drop(SCENARIO_LEVEL)
    keys = rk + table.column_levels
    base = base.select([*keys, *value_columns])
    before = base.height
    base = base.filter(pl.col("count") > 0)
    notes = [MEAN_OVER_TIME_NOTE]
    if with_sum and "per_step" in semantics:
        notes.append(PER_STEP_SUM_NOTE)
    return Result(base, keys, value_columns, "summary", None, keys, ["time"], notes, semantics, units,
                  empty_dropped=before - base.height)


# ---------------------------------------------------------------------------------------------------------
# Sorting and the rows of a result


def ordered(result: Result, q: Query, *, default_by: str | None = None):
    """The result's frame sorted as *q* asks (``sort`` and ``top_n``); by default the summary by ``|mean|`` and
    a ``top_n`` by value, both largest first."""
    import polars as pl

    frame = result.frame
    by = q.sort_by
    if by is None and result.mode == "summary":
        by, descending, absolute = "mean", True, True
    elif by is None and result.value_columns and (q.top_n is not None or default_by is not None):
        by, descending, absolute = (default_by or result.value_columns[0]), True, default_by is not None
    else:
        descending, absolute = q.descending, False
    if by is not None:
        if by not in frame.columns:
            raise ToolError("bad_sort", by=by, columns=", ".join(frame.columns))
        key = pl.col(by).abs() if absolute and frame.schema[by].is_numeric() else pl.col(by)
        frame = frame.sort(key, descending=descending, nulls_last=True, maintain_order=True)
    if q.top_n is not None:
        frame = frame.head(q.top_n)
    return frame


def _number(value: Any) -> Any:
    if isinstance(value, float):
        if not math.isfinite(value):
            return None
        return float(f"{value:.10g}")
    return value


def rows_of(frame, limit: int) -> list[list[Any]]:
    """The first *limit* rows as lists, numbers to ten significant digits."""
    return [[_number(v) for v in row] for row in frame.head(limit).iter_rows()]
