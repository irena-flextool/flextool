"""The cohorts carried between the solves of a run, and the capacity a solve starts from.

A solve hands its realized cohorts to the later solves of the run (:func:`realized_cohorts`: the handoff carriers
``realized_cohort`` / ``realized_retire``, in run years, never recomputed by a later solve), and a later solve sums
the existing cohort and the carried ones into the capacity it starts from (:func:`existing_capacity`).  The cohort
table these rows come from is ``_cohorts``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

import polars as pl

from flextool.engine.preprocess.params._cohorts import (
    KIND_EXISTING,
    KIND_EXTEND,
    KIND_GENERATION,
    KIND_INVEST,
    _cohort_schema,
    _roots,
)

_INF = float("inf")

#: Cohort-table column → its ``realized_cohort`` carrier column (the carrier speaks the handoff's Utf8 names).
#: Any other column of the cohort table is carried under its own name.
_CARRIER_NAMES: dict[str, str] = {"e": "entity", "d_build": "period_decided", "d_on": "period_online",
                                  "d_eol": "period_eol"}
#: The order of the cohort kinds in the carrier (the order the capacity of an entity is summed in).
_KIND_ORDER: dict[str, int] = {KIND_EXISTING: 0, KIND_INVEST: 1, KIND_GENERATION: 2, KIND_EXTEND: 3}


def realized_cohort_schema(cohort_schema: Mapping[str, pl.DataType] | None = None) -> dict[str, pl.DataType]:
    """The columns of the ``realized_cohort`` carrier: ``entity`` and one per column of the cohort table
    (``cohort_schema``, default :func:`._cohorts._cohort_schema`; renamed by :data:`_CARRIER_NAMES`; a float column Float64,
    every other Utf8), then ``size`` (MW, MWh for storage)."""
    schema = _cohort_schema() if cohort_schema is None else dict(cohort_schema)
    out: dict[str, pl.DataType] = {"entity": pl.Utf8}
    for c, t in schema.items():
        out[_CARRIER_NAMES.get(c, c)] = pl.Float64 if t.is_float() else pl.Utf8
    out["size"] = pl.Float64
    return out


#: A retirement (MW, MWh) of a cohort that was not built, at most this, is the solver's tolerance.
_RETIRED_TOL = 1e-6
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
                     retired_built: Mapping[tuple[str, str, str], float] | None = None,
                     extended: Mapping[tuple[str, str, str], float] | None = None,
                     ) -> tuple[pl.DataFrame | None, pl.DataFrame | None]:
    """``(realized_cohort, realized_retire)``: the cohorts a solve hands to the later solves of the run, and the
    capacity retired from them, both chain-cumulative.

    ``realized_cohort`` — the rows of ``prior_cohort`` (the carrier the solve was handed), unchanged, and one row per
    investment decision ``(e, d) ∈ invested`` (``v_invest`` of a realized, real-named decision period, > 0) with
    the columns of its row of the solve's cohort table ``cohort`` (run years, never recomputed by a later solve) and
    ``size = v_invest · unitsize[e]``, followed by the rows of its generations (``root`` the decision's cohort, the
    rows of the solve's table by name, size null: the chain's plants and their prices), and one row per life
    extension ``(e, c, d) ∈ extended`` (``v_extend`` of the cohort ``c`` in a realized, real-named period, > 0):
    its row ``extend:<d>:<c>`` of the cohort table and ``size = v_extend · unitsize[e]``.  A decision already carried
    (the same cohort realized again) adds its size to the carried row.  Sorted by entity, kind, start of operation
    (stable: earlier rows first).

    ``realized_retire`` ``[entity, cohort, period, value]`` — the rows of ``prior_retire`` and
    ``retired[e, d] · unitsize[e]`` (``v_divest`` of a realized decision period, > 0) of the existing cohort of
    ``e`` and ``retired_built[e, c, d] · unitsize[e]`` (``v_divest_built`` of a realized decision period, > 0) of
    the model-built cohort ``c`` (also a life extension), summed per (entity, cohort, period); sorted, zero rows left
    out.  A later solve
    subtracts each from its own cohort only (the existing cohort's retirements bound the existing capacity, a built
    cohort's its carried size).

    Each frame is ``None`` when it has no row.  An investment without a row in ``cohort`` cannot be placed in a
    later solve's periods and raises :class:`ValueError`."""
    by_key: dict[tuple[str, str], dict] = {}
    columns = list(cohort.columns) if cohort is not None else list(_cohort_schema())
    schema = realized_cohort_schema(cohort.schema if cohort is not None else None)
    if prior_cohort is not None and prior_cohort.height > 0:
        for row in prior_cohort.iter_rows(named=True):
            rec = {c: _cell(row.get(c), t) for c, t in schema.items()}
            by_key[(rec["entity"], rec["cohort"])] = rec
    if invested or extended:
        if cohort is None:
            raise ValueError("Realized investments cannot be handed to the later solves: the solve has no cohort "
                             "table.")
        rows = {(str(r["e"]), str(r["cohort"])): r for r in cohort.iter_rows(named=True)}
        chains: dict[tuple[str, str], list[dict]] = {}
        if "root" in columns:
            for r in cohort.filter(pl.col("kind") == KIND_GENERATION).iter_rows(named=True):
                chains.setdefault((str(r["e"]), str(r["root"])), []).append(r)
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
            for gen in chains.get(key, ()):
                rec = {_CARRIER_NAMES.get(c, c): _cell(gen[c], schema[_CARRIER_NAMES.get(c, c)]) for c in columns}
                rec["size"] = None
                by_key[(e, str(gen["cohort"]))] = rec
        for (e, c, d), v in (extended or {}).items():
            key = (e, f"{KIND_EXTEND}:{d}:{c}")
            size = v * unitsize.get(e, 1.0)
            if key in by_key:
                by_key[key]["size"] = by_key[key]["size"] + size
                continue
            row = rows.get(key)
            if row is None or row.get("y_on") is None:
                raise ValueError(f"The life extension of {e!r} cohort {c!r} in period {d!r} has no row in the solve's "
                                 "cohort table: it cannot be handed to the later solves.")
            rec = {_CARRIER_NAMES.get(c_, c_): _cell(row[c_], schema[_CARRIER_NAMES.get(c_, c_)]) for c_ in columns}
            rec["size"] = size
            by_key[key] = rec
    ordered = sorted(by_key.values(),
                     key=lambda r: (r["entity"], _KIND_ORDER.get(r["kind"], len(_KIND_ORDER)), r["y_on"] or 0.0))
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
    for (e, c, d), v in (retired_built or {}).items():
        mw = v * unitsize.get(e, 1.0)
        if (e, c) not in by_key:
            if mw <= _RETIRED_TOL:
                continue  # the solver's tolerance on a cohort that was not built
            raise ValueError(f"The retirement of {e!r} cohort {c!r} in period {d!r} ({mw:g}) names a cohort that is "
                             "not realized: it cannot be handed to the later solves.")
        sums[(e, c, d)] = sums.get((e, c, d), 0.0) + mw
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
    #: ``carried[e, d]``: the carried investment cohorts in service in d, less their carried retirements.
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
        retired_c[e, d] = Σ_{(c, d′, v) ∈ retire : alive_c(d), y[d′] ≤ y[d] or d′ before the solve} v
        carried[e, d] =  Σ_{carried invest cohorts c of e alive in d} size_c
                         − [e may retire] · Σ_{carried invest cohorts c} retired_c[e, d]
        all[e, d]     =  (pre[e, d] + Σ_{carried invest and extension cohorts c alive in d} size_c)
                         − [e may retire] · Σ_c retired_c[e, d]
        invested[e]   =  Σ_{carried invest cohorts c of e} size_c

    ``existing`` are the existing cohorts of the run (:func:`._cohorts.run_existing_cohorts`, ``y_on`` / ``y_eol``),
    ``existing_mw[e, d]`` the user's ``existing`` of each period in MW (the map is read in every solve),
    ``carried`` / ``retire`` the handoff ``realized_cohort`` / ``realized_retire`` (their ``generation`` rows are
    the plants of a chain, not capacity, and are left out).  The sums add in the order of
    the carried rows (by start of operation), ``all`` starting from ``pre``.  ``retired`` is the existing cohort's
    part (the bound of the retirements of existing capacity); ``carried`` subtracts those of the carried cohorts,
    ``all`` those of every cohort (so ``all − carried`` is the existing cohort less its own retirements and plus the
    carried life extensions).  A carried life extension (kind ``extend``) is capacity but never an investment: it
    counts in ``all`` only, not in ``carried`` (the previously invested capacity) or ``invested`` (the investment
    limits).  A retirement of a cohort that is neither the existing cohort nor a carried one raises
    :class:`ValueError`."""
    from flextool.engine.preprocess.params._retire_existing import retired_carried
    entities = [str(e) for e in entities]
    periods = [str(d) for d in periods]
    y_of = {d: float(years.get(d, 0.0)) for d in periods}
    ends: dict[tuple[str, str], tuple[float | None, float]] = {}
    existing_of: dict[str, str] = {}
    existing = _roots(existing)
    if carried is not None:
        carried = _roots(carried)
    for e, c, y_on, y_eol in existing.select("e", "cohort", "y_on", "y_eol").iter_rows():
        existing_of.setdefault(str(e), str(c))
        ends[(str(e), str(c))] = (None, _INF if y_eol is None else float(y_eol))
    sizes: dict[str, list[tuple[str, float]]] = {}
    extensions: set[tuple[str, str]] = set()
    if carried is not None and carried.height > 0:
        for e, c, kind, y_on, y_eol, size in carried.select("entity", "cohort", "kind", "y_on", "y_eol",
                                                            "size").iter_rows():
            e, c = str(e), str(c)
            ends[(e, c)] = (None if y_on is None else float(y_on), _INF if y_eol is None else float(y_eol))
            sizes.setdefault(e, []).append((c, float(size)))
            if kind == KIND_EXTEND:
                extensions.add((e, c))
    alive = {(e, c, d): _in_service(y_of[d], *ends[(e, c)]) for (e, c) in ends for d in periods}

    rows = [] if retire is None else [(str(e), str(c), str(d), float(v)) for e, c, d, v in
                                      retire.select("entity", "cohort", "period", "value").iter_rows()]
    unknown = sorted({(e, c) for e, c, _d, _v in rows if (e, c) not in ends})
    if unknown:
        raise ValueError(f"Carried retirements name cohorts the solve does not have: {unknown}.")
    dead = {k for k, v in alive.items() if not v}
    retired_all = retired_carried(rows, years, periods, dead)
    retired = retired_carried([r for r in rows if existing_of.get(r[0]) == r[1]], years, periods, dead)
    retired_built = retired_carried(
        [r for r in rows if existing_of.get(r[0]) != r[1] and (r[0], r[1]) not in extensions], years, periods, dead)

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
                    if (e, c) not in extensions:
                        v_carried += size
            if e in divest:
                v_all -= retired_all.get((e, d), 0.0)
                v_carried -= retired_built.get((e, d), 0.0)
            pre[(e, d)], carried_mw[(e, d)], all_mw[(e, d)] = v_pre, v_carried, v_all
    invested: dict[str, float] = {}
    for e, cohorts in sizes.items():
        total = 0.0
        for c, size in cohorts:
            if (e, c) not in extensions:
                total += size
        invested[e] = total
    return ExistingCapacity(pre=pre, carried=carried_mw, all=all_mw, retired=retired, invested=invested)


__all__ = [
    "ExistingCapacity",
    "REALIZED_RETIRE_SCHEMA",
    "existing_capacity",
    "realized_cohort_schema",
    "realized_cohorts",
]
