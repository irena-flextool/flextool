"""Life extension: capacity reaching its end of life kept in service ``life_extension_years`` longer.

An entity that gives ``life_extension_cost`` and ``life_extension_years`` may extend each cohort of its capacity
that ends its life (``reinvest_choice`` / ``no_reinvestment``: the existing cohort, an investment of this solve, or
one an earlier solve of the run realized; never an extension itself) in an investment period ``d`` of the solve
that starts at the cohort's end of life ``t0 = y_eol(c)`` and lies on the cohort's lineage.  The extension is a
cohort of its own (kind ``extend``, id ``extend:<d>:<parent>``, :func:`extension_id`): it comes online at ``t0``
in ``d`` and ends at ``y_eol(x) = round_end_of_life(t0 + life_extension_years[e,d], end_of_life_method[e])``; an
extension that the rounding gives no period of service (``y_eol(x) ≤ t0``) is not offered (a warning names it).
Its prices are ``_npv.extension_prices``' (the one cost point).  The frames (``FlexData`` fields,
:func:`extension_frames`):

* ``cd_extend`` ``[e, cohort, d]`` — the extension options: ``cohort`` the parent, the columns of ``v_extend``;
* ``cdd_extend`` ``[e, cohort, d_extend, d]`` — the units extended in ``d_extend`` are in service in ``d``
  (``d_extend``'s lineage, ``y[d] < y_eol(x)``);
* ``cohort_extend`` ``[e, cohort, kind, d_build, d, d_last, remaining, size]`` — one row per option: the parent's
  kind and decision period (the key of its ``v_invest``; null for the existing cohort and a carried one),
  ``d_last`` the parent's last period in service before ``d`` on ``d``'s lineage (null when that period is not
  in the solve), ``remaining`` (MW, MWh) the size of a carried parent less its realized retirements and
  ``size`` the existing capacity in service at the parent's end of life when ``d_last`` is not in the solve.

The extension cohort's rows (cohort table) and its periods in service (``cohort_alive``, kind ``extend``) are
returned with them; a retirement of an extension is one of model-built capacity (``_retire_built``).

A solve that places no retirement reach (no ``dd_forward``: a roll or a nested sub-solve) has no option.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Mapping

import polars as pl

from flextool.engine.input._axis_enums import align_join_dtypes, rename_to_axis, schema_dtype
from flextool.engine.input._lifetime_roles import (
    ENTITY_CLASSES,
    LIFE_EXTENSION,
    LIFE_EXTENSION_COST,
    LIFE_EXTENSION_YEARS,
    source_entities_with,
)
from flextool.engine.preprocess.params._cohorts import (
    BOUNDED_METHODS,
    KIND_EXISTING,
    KIND_EXTEND,
    KIND_INVEST,
    _alive_schema,
    _cohort_frame,
    _Calendar,
)
from flextool.engine.preprocess.params._period_values import ed_values_on
from flextool.engine.preprocess.params._run_calendar import YEAR_TOL

if TYPE_CHECKING:
    from flextool.engine.input._input_source import InputSource

_INF = float("inf")


def extension_id(d: str, parent: str) -> str:
    """The cohort id of the extension of the cohort ``parent`` decided in (coming online in) the period ``d``."""
    return f"{KIND_EXTEND}:{d}:{parent}"


def extension_id_expr(d: str, parent: str) -> pl.Expr:
    """:func:`extension_id` of the columns ``d`` and ``parent`` (Utf8), as a polars expression."""
    return pl.concat_str(pl.lit(f"{KIND_EXTEND}:"), pl.col(d).cast(pl.Utf8), pl.lit(":"), pl.col(parent))


def extending_entities(source: "InputSource") -> set[str]:
    """The units, nodes and connections that give both ``life_extension_cost`` and ``life_extension_years`` (the
    input checks refuse one without the other)."""
    out: set[str] = set()
    for cls in ENTITY_CLASSES:
        with_value = source_entities_with(source, cls)
        out |= with_value(LIFE_EXTENSION_COST) & with_value(LIFE_EXTENSION_YEARS)
    return out


def cohort_extend_columns() -> dict[str, pl.DataType]:
    """The columns of ``cohort_extend`` (module docstring)."""
    return {"e": schema_dtype(None, "e"), "cohort": pl.Utf8, "kind": pl.Utf8, "d_build": schema_dtype(None, "d_build"),
            "d": schema_dtype(None, "d"), "d_last": schema_dtype(None, "d"), "remaining": pl.Float64,
            "size": pl.Float64}


@dataclass(frozen=True)
class ExtensionFrames:
    """The life-extension frames of a solve (module docstring); every frame has rows."""

    options: pl.DataFrame
    counting: pl.DataFrame
    parents: pl.DataFrame
    #: The extension cohorts' rows of the cohort table (axis columns on their axes, prices null).
    cohort: pl.DataFrame
    #: The extension cohorts' periods in service (``cohort_alive`` rows, kind ``extend``).
    alive: pl.DataFrame


def _parent_rows(cohort: pl.DataFrame, entities: set[str]) -> pl.DataFrame:
    """``[e, cohort, kind, method, eol_method, d_build, y_eol]`` (Utf8 labels) of the cohorts of ``entities`` that
    end their life inside the run: an existing or investment cohort of a bounded ``lifetime_method`` with a
    lifetime (``0 < y_life < +inf``) and a finite end of life."""
    return (cohort.filter((pl.col("kind").is_in([KIND_EXISTING, KIND_INVEST]))
                          & pl.col("method").is_in(list(BOUNDED_METHODS))
                          & pl.col("y_on").is_not_null() & pl.col("y_life").is_not_null()
                          & (pl.col("y_life") > 0.0) & (pl.col("y_life") < _INF)
                          & pl.col("y_eol").is_not_null() & (pl.col("y_eol") < _INF)
                          & pl.col("e").cast(pl.Utf8).is_in(sorted(entities)))
                  .select(pl.col("e").cast(pl.Utf8), "cohort", "kind", "method", "eol_method",
                          pl.col("d_build").cast(pl.Utf8), "y_eol"))


def extension_frames(source: "InputSource",
                     cohort: pl.DataFrame | None,
                     cohort_alive: pl.DataFrame | None,
                     dd_forward: pl.DataFrame | None,
                     period_invest: list[str],
                     years: Mapping[str, float],
                     cal: _Calendar | None,
                     ) -> ExtensionFrames | None:
    """The :class:`ExtensionFrames` of a solve, ``None`` without an extension option.

    ``cohort`` / ``cohort_alive`` / ``dd_forward`` are the solve's cohort frames, ``period_invest`` its investment
    periods, ``years`` the period years of the run (:func:`._cohorts.cohort_years`) and ``cal`` the seams of the
    run (:func:`._cohorts.run_cohort_calendar`)::

        options = {(e, c, d) : e ∈ extending_entities, c a parent cohort of e (:func:`_parent_rows`),
                   d ∈ period_invest, |y[d] − y_eol(c)| ≤ YEAR_TOL, (d_build(c), d) ∈ dd_forward for an
                   investment cohort, E[e,d] and L_x[e,d] given, y_eol(x) > y[d] + YEAR_TOL}
        y_eol(x) = round_end_of_life(y[d] + L_x[e,d], end_of_life_method[e])
        alive(x) = {d′ : (d, d′) ∈ dd_forward, y[d′] < y_eol(x)}
        d_last(c, d) = the period d″ with the largest year y[d″] < y[d] − YEAR_TOL, (d″, d) ∈ dd_forward and c in
                   service in d″ (``cohort_alive``)
    """
    if (cohort is None or cohort.height == 0 or dd_forward is None or dd_forward.height == 0 or not period_invest
            or cal is None):
        return None
    entities = extending_entities(source)
    if not entities:
        return None
    parents = _parent_rows(cohort, entities)
    if parents.height == 0:
        return None
    invest = pl.DataFrame({"d": [str(d) for d in period_invest if str(d) in years],
                           "y_d": [float(years[str(d)]) for d in period_invest if str(d) in years]},
                          schema={"d": pl.Utf8, "y_d": pl.Float64})
    forward = dd_forward.select(pl.col("d_retire").cast(pl.Utf8), pl.col("d").cast(pl.Utf8))
    rows = (parents.join(invest, how="cross")
                   .filter((pl.col("y_d") - pl.col("y_eol")).abs() <= YEAR_TOL))
    # An investment cohort extends on the lineage of its decision.
    built = rows.filter(pl.col("kind") == KIND_INVEST).join(
        forward.rename({"d_retire": "d_build"}), on=["d_build", "d"], how="semi")
    rows = pl.concat([rows.filter(pl.col("kind") != KIND_INVEST), built])
    if rows.height == 0:
        return None
    keys = rows.select("e", "d").unique(maintain_order=True)
    values = {name: (ed_values_on(source, name, keys.lazy(), fill=None).select("e", pl.col("d").cast(pl.Utf8),
                                                                                pl.col("value").alias(name))
                     .collect().with_columns(pl.col("e").cast(pl.Utf8)))
              for name in LIFE_EXTENSION}
    for name in LIFE_EXTENSION:
        rows = rows.join(values[name], on=["e", "d"], how="left")
    rows = rows.filter(pl.col(LIFE_EXTENSION_COST).is_not_null() & pl.col(LIFE_EXTENSION_YEARS).is_not_null()
                       & (pl.col(LIFE_EXTENSION_YEARS) > 0.0))
    if rows.height == 0:
        return None
    ends = [cal.eol(y + life, method) for y, life, method in
            rows.select("y_d", LIFE_EXTENSION_YEARS, "eol_method").iter_rows()]
    rows = rows.with_columns(y_eol_x=pl.Series("y_eol_x", ends, dtype=pl.Float64))
    never = rows.filter(pl.col("y_eol_x") <= pl.col("y_d") + YEAR_TOL)
    _never_operates_warning(never)
    rows = rows.filter(pl.col("y_eol_x") > pl.col("y_d") + YEAR_TOL)
    if rows.height == 0:
        return None
    rows = rows.with_columns(x=extension_id_expr("d", "cohort"))

    yr = pl.DataFrame({"d_all": list(years), "y": [float(v) for v in years.values()]},
                      schema={"d_all": pl.Utf8, "y": pl.Float64})
    alive_x = (rows.select("e", "cohort", "x", "d", "y_eol_x")
                   .join(forward.rename({"d_retire": "d", "d": "d_all"}), on="d", how="inner")
                   .join(yr, on="d_all", how="inner")
                   .filter(pl.col("y") < pl.col("y_eol_x") - YEAR_TOL))
    d_last = _last_periods(rows.select("e", "cohort", "d", "y_d"), cohort_alive, forward, yr)
    rows = rows.join(d_last, on=["e", "cohort", "d"], how="left")

    x_rows = [(x, e, KIND_EXTEND, method, eol_method, d, d, y_d, 0.0, life, y_d + life, None, y_eol,
               cal.label(y_eol), *((None,) * 6), x, c)
              for x, e, method, eol_method, d, y_d, life, y_eol, c in
              rows.select("x", "e", "method", "eol_method", "d", "y_d", LIFE_EXTENSION_YEARS, "y_eol_x",
                          "cohort").iter_rows()]
    x_cohort = _cohort_frame(x_rows)
    alive = (alive_x.select("x", "e", d_build="d", d="d_all", y="y")
                    .with_columns(kind=pl.lit(KIND_EXTEND))
                    .lazy()
                    .pipe(rename_to_axis, {"e": "e", "d_build": "d_build", "d": "d"})
                    .select(pl.col("x").alias("cohort"), "e", "kind", "d_build", "d", "y")
                    .select(list(_alive_schema()))
                    .collect())
    options = (rows.select("e", "cohort", "d").lazy().pipe(rename_to_axis, {"e": "e", "d": "d"})
                   .sort("e", "cohort", "d").collect())
    counting = (alive_x.select("e", "cohort", d_extend="d", d="d_all").lazy()
                       .pipe(rename_to_axis, {"e": "e", "d_extend": "d_extend", "d": "d"})
                       .sort("e", "cohort", "d_extend", "d").collect())
    parents_out = (rows.select("e", "cohort", "kind", "d_build", "d", "d_last",
                               remaining=pl.lit(0.0), size=pl.lit(None, dtype=pl.Float64))
                       .lazy()
                       .pipe(rename_to_axis, {"e": "e", "d_build": "d_build", "d": "d", "d_last": "d_last"})
                       .select(list(cohort_extend_columns()))
                       .sort("e", "cohort", "d").collect())
    return ExtensionFrames(options=options, counting=counting, parents=parents_out, cohort=x_cohort, alive=alive)


def _last_periods(rows: pl.DataFrame, cohort_alive: pl.DataFrame | None, forward: pl.DataFrame,
                  yr: pl.DataFrame) -> pl.DataFrame:
    """``[e, cohort, d, d_last]`` (Utf8): for each option ``(e, c, d)`` of ``rows`` (with ``y_d``) the parent's last
    period in service before ``d`` on ``d``'s lineage (none when the parent serves no period of the solve before
    ``d``)."""
    schema = {"e": pl.Utf8, "cohort": pl.Utf8, "d": pl.Utf8, "d_last": pl.Utf8}
    if cohort_alive is None or cohort_alive.height == 0:
        return pl.DataFrame(schema=schema)
    served = cohort_alive.select(pl.col("e").cast(pl.Utf8), "cohort", pl.col("d").cast(pl.Utf8).alias("d_last"))
    return (rows.join(forward.rename({"d_retire": "d_last"}), on="d", how="inner")
                .join(served, on=["e", "cohort", "d_last"], how="inner")
                .join(yr.rename({"d_all": "d_last", "y": "y_last"}), on="d_last", how="inner")
                .filter(pl.col("y_last") < pl.col("y_d") - YEAR_TOL)
                .sort("e", "cohort", "d", "y_last", "d_last")
                .unique(subset=["e", "cohort", "d"], keep="last", maintain_order=True)
                .select(list(schema)))


def _never_operates_warning(never: pl.DataFrame) -> None:
    """Warn, once per run and entity, about the extensions of ``never`` (``[e, d, life_extension_years,
    eol_method]``) that the ``end_of_life_method`` rounds to their own start: they would be in service in no period,
    so they are not offered."""
    if never.height == 0:
        return
    from flextool.engine import _behaviour_changes
    for e, rows in never.group_by("e", maintain_order=True):
        name = str(e[0])
        periods = ", ".join(dict.fromkeys(str(d) for d in rows["d"].to_list()))
        life, method = rows.select(LIFE_EXTENSION_YEARS, "eol_method").row(0)
        _behaviour_changes.warn_once(
            ("extension_never_operates", name),
            f"Warning: {name}: the life extensions starting in {periods} would never operate "
            f"(life_extension_years {life:g} years, end_of_life_method {method}): their end of life is rounded to "
            "the start of the period they start in. They are not offered.")


def append_extensions(cohort: pl.DataFrame, cohort_alive: pl.DataFrame,
                      frames: ExtensionFrames | None) -> tuple[pl.DataFrame, pl.DataFrame]:
    """The cohort frames with the extension cohorts' rows and periods in service (unchanged without one)."""
    if frames is None:
        return cohort, cohort_alive
    x_cohort, own = align_join_dtypes(frames.cohort, cohort, ["e", "d_build"])
    x_alive, own_alive = align_join_dtypes(frames.alive, cohort_alive, ["e", "d_build", "d"])
    return (pl.concat([own, x_cohort.select(own.columns)], how="vertical"),
            pl.concat([own_alive, x_alive.select(own_alive.columns)], how="vertical"))


__all__ = ["ExtensionFrames", "append_extensions", "cohort_extend_columns", "extending_entities",
           "extension_frames", "extension_id", "extension_id_expr"]
