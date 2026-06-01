"""full_dt slicing helper — "solve as picker" infrastructure.

This module is installment 1 of the ``(d,t)``-centric / "solve as picker"
preprocessing refactor (see ``specs/preprocessing_csv_to_db_migration.md``
§9 / §10.1).  The single public function :func:`slice_full_dt` realises
the picker step of the architecture: a parameter family is computed ONCE
over the COMPLETE ``(d,t)`` timeline (``full_dt``), cached per
``complete_solve``, and each rolling sub-solve obtains its values by
*slicing* that full-domain frame down to the roll's ``(period, time)``
window — never by re-deriving per roll.

The correctness invariant (§9.5) that makes the slice value-preserving:
for a pure ``(d,t)`` parameter ``P``,

    broadcast(P, full_dt).filter((d,t) ∈ dt_solve) ≡ broadcast(P, dt_solve)

because a value at ``(c, "price", d, t)`` is a coordinate lookup with ZERO
window aggregation — restricting the ``(d,t)`` grid before vs. after the
derivation yields identical rows.  This helper is therefore only valid for
families *proven* coordinate-stable; aggregating families must compute
their aggregate over the complete timeline and only slice the coordinate
(§9.5 amendment 1).

Laziness is load-bearing (§9.8 Attack 2): the full-domain frame is scanned
lazily and the ``(d,t)`` predicate is pushed down via a semi-join, so a
roll materialises only its slice rather than the whole-timeline frame.
"""
from __future__ import annotations

import polars as pl

__all__ = ["slice_full_dt"]


def slice_full_dt(
    full_frame: "pl.DataFrame | pl.LazyFrame",
    roll_dt: "pl.DataFrame | pl.LazyFrame",
    *,
    dt_cols: tuple[str, str] = ("period", "time"),
    roll_dt_cols: tuple[str, str] = ("period", "step"),
) -> pl.DataFrame:
    """Slice a full-domain ``(d,t)`` frame down to a roll's window.

    Parameters
    ----------
    full_frame:
        The parameter frame computed once over the COMPLETE timeline.
        May be eager or lazy; ``.lazy()`` is called defensively.
    roll_dt:
        The roll's ``(period, step)`` window — typically the frame read
        from ``solve_data/steps_in_use.csv``.  Only ``roll_dt_cols`` are
        used; duplicates are de-duplicated.  May be eager or lazy.
    dt_cols:
        The ``(period, time)`` column names on *full_frame*.
    roll_dt_cols:
        The ``(period, step)`` column names on *roll_dt* (mapped
        positionally onto *dt_cols* for the join).

    Returns
    -------
    pl.DataFrame
        * **strict_static** — if *full_frame* carries neither ``dt_cols``
          column, it is returned unchanged (identity): a frame with no
          ``(d,t)`` axis is solve-invariant and needs no slice.
        * **slice_static** — otherwise, the rows of *full_frame* whose
          ``(period, time)`` coordinate appears in the roll window,
          preserving *full_frame*'s row order (a semi-join keeps the
          left side's order).  An empty *roll_dt* yields an empty frame
          with *full_frame*'s schema.

    Notes
    -----
    The slice is a semi-join, not an inner join: it filters *full_frame*
    by coordinate membership without duplicating rows or reordering, and
    pushes the ``(d,t)`` predicate down through the lazy plan so only the
    matching slice is materialised.
    """
    full_lazy = full_frame.lazy()
    schema_names = full_lazy.collect_schema().names()

    period_col, time_col = dt_cols
    # strict_static: no (d,t) axis at all → solve-invariant, identity.
    if period_col not in schema_names and time_col not in schema_names:
        return full_lazy.collect()

    # slice_static: semi-join on the (d,t) coordinate.  Select + unique
    # the roll window down to just its join keys so the semi-join filters
    # purely by coordinate membership and never widens the result.
    roll_keys = (
        roll_dt.lazy()
        .select(list(roll_dt_cols))
        .unique()
    )
    return (
        full_lazy.join(
            roll_keys,
            left_on=list(dt_cols),
            right_on=list(roll_dt_cols),
            how="semi",
        )
        .collect()
    )
