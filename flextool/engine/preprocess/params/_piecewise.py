"""Load-boundary step of ``piecewise_linear`` units (``specs/piecewise_efficiency_final.md`` §3.2).

:func:`apply_piecewise_curves` is called once per load in ``load_flextool`` (single, rolling, synthetic / nested
and stochastic sub-solves), right after ``_apply_db_overrides`` and its completion block, before the handoff
overlay, the coarse frames, ``process_minload`` and ``apply_reduce_detail``.  It is the one place that turns the
curves (``unit.piecewise_linear_efficiency``) into model parameters; the producers before it (``p_slope`` /
``p_section`` seeds, the rolling emit) treat a ``piecewise_linear`` unit as a constant-efficiency unit and their
values are overwritten here:

1. the points ``(p, d, load, eff)`` of the solve's periods (:func:`solve_points`): one curve for every period is
   used in each period of the solve, a stochastic branch period takes its anchor period's whole curve, and a
   solve period without a curve stops the run (E11);
2. the lines of each curve (:func:`flextool.engine.input._piecewise_curve.curve_lines`): the online form for the
   units with online variables (``process_online``), the capacity form for the others;
3. single-line units (one line in every period, :func:`_piecewise_flags`): handled as ``min_load_efficiency``
   units.  Their ``p_slope`` rows are overwritten in place with the line's slope, and the units whose curve has an
   intercept get ``p_section`` rows (the intercept) and join ``process_min_load_eff``; a direct unit's
   ``conversion_flow_coeff`` ratio is folded as for ``min_load_efficiency`` (``slope · k_sink / k_src``,
   ``section / k_src``);
4. multi-line units (two or more lines in some period): ``process_piecewise``, ``process_piecewise_needs_online``,
   ``process_piecewise_remap``, ``p_piecewise_slope`` / ``p_piecewise_intercept`` ``(p, seg, d)`` and
   ``p_piecewise_points``; their ``p_slope`` is the full-load input per output ``1/η(1)``;
5. the capacity parameters derived from the seeded slope before this step, for every piecewise unit:
   ``p_indirect_input_cap`` and the indirect input-arc rows of ``p_flow_upper``
   (:func:`_recompute_input_caps`).

The model has no rows for multi-line units yet: a solve with one stops here, after the steps above.
"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import polars as pl
from polar_high import Param

from flextool.engine.input._solve_state import FlexToolConfigError

if TYPE_CHECKING:
    from flextool.engine.input._input_source import InputSource

#: The keys of a curve value per unit and period.
_KEYS_PD = ("p", "d")


def piecewise_units(db_reader: "InputSource") -> list[str]:
    """The units whose ``conversion_method`` is ``piecewise_linear`` in the scenario, sorted."""
    from flextool.engine.preprocess.params._derived import _try_param

    cm = _try_param(db_reader, "unit", "conversion_method")
    if cm is None:
        return []
    return sorted(cm.filter(pl.col("value").cast(pl.Utf8) == "piecewise_linear")
                    .get_column("name").cast(pl.Utf8).unique().to_list())


def _names(frame, col: str = "p") -> set[str]:
    """The ``col`` values of a set frame (or a Param's frame) as Utf8; empty for None."""
    if frame is None:
        return set()
    f = frame.frame if isinstance(frame, Param) else frame
    if f.height == 0 or col not in f.columns:
        return set()
    return set(f.get_column(col).cast(pl.Utf8).to_list())


def _solve_periods(dt: pl.DataFrame) -> list[str]:
    """The periods of the solve's ``dt``, in their order."""
    return dt.get_column("d").cast(pl.Utf8).unique(maintain_order=True).to_list()


def solve_points(curves: pl.DataFrame, units: list[str], periods: list[str], pairs: pl.DataFrame | None,
                 solve: str | None) -> pl.DataFrame:
    """The points ``(p, d, load, eff)`` (Utf8 ``p`` / ``d``) of *units* in the solve's *periods* from the whole
    authored curves *curves* (:func:`flextool.engine.input._param_shapes.piecewise_curve_frame`).

    One curve for every period (``d`` null) is used in each period; a curve per period keeps its periods of the
    solve, and a stochastic branch period of *pairs* ``(anchor, br)`` without a curve of its own takes its anchor
    period's whole curve (:func:`flextool.engine.preprocess.params._branch.inherit_anchor_period`).  A solve period
    of a unit without a curve is the E11 error (``FlexToolConfigError``, naming the unit, the period and *solve*).
    Sorted by ``p``, then the solve's period order, then ``load``.
    """
    from flextool.engine.preprocess.params._branch import inherit_anchor_period

    order = pl.DataFrame({"d": periods, "__order": list(range(len(periods)))},
                         schema={"d": pl.Utf8, "__order": pl.Int64})
    own = curves.filter(pl.col("p").is_in(units)).with_columns(pl.col("p").cast(pl.Utf8), pl.col("d").cast(pl.Utf8))
    constant = own.filter(pl.col("d").is_null()).drop("d").join(order.select("d"), how="cross")
    per_period = own.filter(pl.col("d").is_not_null())
    if pairs is not None and per_period.height > 0:
        per_period = inherit_anchor_period(per_period, pairs, group_keys=("p",))
    points = (pl.concat([constant.select("p", "d", "load", "eff"), per_period.select("p", "d", "load", "eff")])
              .join(order, on="d", how="inner")
              .sort("p", "__order", "load")
              .drop("__order"))
    have = points.select("p", "d").unique()
    want = pl.DataFrame({"p": units}, schema={"p": pl.Utf8}).join(order.select("d"), how="cross")
    missing = want.join(have, on=["p", "d"], how="anti").sort("p", "d")
    if missing.height > 0:
        where = f" of solve {solve}" if solve else ""
        raise FlexToolConfigError("; ".join(
            f"unit '{p}': piecewise_linear_efficiency has no curve for period {d}{where}"
            for p, d in missing.iter_rows()) + ".")
    return points


def _with_utf8_keys(frame: pl.DataFrame, keys: tuple[str, ...]) -> pl.DataFrame:
    return frame.with_columns(pl.col(k).cast(pl.Utf8).alias(f"__{k}") for k in keys)


def _overwrite_values(param: Param, values: pl.DataFrame, units: set[str], keys: tuple[str, ...],
                      what: str) -> Param:
    """*param* with the ``value`` of its rows of *units* taken from *values* (Utf8 *keys* + ``value``), joined on
    *keys* (a subset of the dims); row order and dtypes kept.  Every row of *units* must find a value."""
    f = param.frame
    uk = [f"__{k}" for k in keys]
    new = values.select(*(pl.col(k).cast(pl.Utf8).alias(f"__{k}") for k in keys),
                        pl.col("value").cast(pl.Float64).alias("__new"))
    if new.select(uk).is_duplicated().any():
        raise ValueError(f"piecewise_linear: duplicate {keys} cells in the new {what} values.")
    joined = (_with_utf8_keys(f, keys).with_row_index("__row")
              .join(new, on=uk, how="left").sort("__row"))
    mine = pl.col("__p").is_in(sorted(units))
    lost = joined.filter(mine & pl.col("__new").is_null())
    if lost.height > 0:
        raise ValueError(f"piecewise_linear: {what} rows without a curve line: {lost.select(uk).head(5).rows()}.")
    out = joined.with_columns(
        pl.when(mine).then(pl.col("__new")).otherwise(pl.col("value")).cast(f.schema["value"]).alias("value")
    ).select(f.columns)
    return Param(param.dims, out, name=param.name)


def _splice_rows(param: "Param | None", new: "Param | None", units: set[str], what: str) -> "Param | None":
    """*param* with its rows of *units* replaced by *new*'s rows of *units* (same keys; values from *new*), the
    other rows unchanged and in place."""
    if new is None:
        new_rows = None
    else:
        new_rows = new.frame.filter(pl.col("p").cast(pl.Utf8).is_in(sorted(units)))
    if param is None:
        if new_rows is None or new_rows.height == 0:
            return None
        raise ValueError(f"piecewise_linear: {what} has no rows to replace, but the curve gives "
                         f"{new_rows.height} (first {new_rows.head(3).rows()}).")
    keys = tuple(param.dims)
    old_keys = (_with_utf8_keys(param.frame, keys).filter(pl.col("__p").is_in(sorted(units)))
                .select(f"__{k}" for k in keys))
    if new_rows is None:
        new_rows = param.frame.head(0)
    new_keys = _with_utf8_keys(new_rows, keys).select(f"__{k}" for k in keys)
    if (old_keys.height != new_keys.height
            or old_keys.join(new_keys, on=list(old_keys.columns), how="anti").height > 0):
        raise ValueError(f"piecewise_linear: the recomputed {what} rows of {sorted(units)} have other keys than "
                         f"the existing ones ({old_keys.height} vs {new_keys.height} rows).")
    if old_keys.height == 0:
        return param
    return _overwrite_values(param, new_rows.select(*keys, "value"), units, keys, what)


def _direct_ratio(db_reader: "InputSource") -> pl.DataFrame:
    """``(p, k_src, k_sink)`` (Utf8 ``p``) of the direct units: the ratio ``p_slope_from_source`` folds."""
    from flextool.engine.preprocess.params._arithmetic import _direct_flow_coef_ratio

    return _direct_flow_coef_ratio(db_reader, warn=False).with_columns(pl.col("p").cast(pl.Utf8))


def single_line_values(lines: pl.DataFrame, ratio: pl.DataFrame) -> pl.DataFrame:
    """``(p, d, slope, section)`` of single-line units from their one line per ``(p, d)`` (*lines*:
    ``(p, d, seg, slope, intercept)``), with a direct unit's ``conversion_flow_coeff`` ratio folded as
    ``p_slope_from_source`` / ``p_section_from_source`` fold the ``min_load_efficiency`` slope and section:

        slope   = α · k_sink / k_src,   section = β / k_src        (direct units)
        slope   = α,                    section = β                 (others)
    """
    dup = lines.group_by("p", "d").len().filter(pl.col("len") != 1)
    if dup.height > 0:
        raise ValueError(f"piecewise_linear: a single-line unit has several lines in {dup.select('p', 'd').rows()}.")
    return (lines.join(ratio, on="p", how="left")
            .with_columns(
                slope=pl.when(pl.col("k_sink").is_not_null())
                        .then(pl.col("slope") * pl.col("k_sink") / pl.col("k_src"))
                        .otherwise(pl.col("slope")),
                section=pl.when(pl.col("k_src").is_not_null())
                          .then(pl.col("intercept") / pl.col("k_src"))
                          .otherwise(pl.col("intercept")))
            .select("p", "d", "slope", "section"))


def _remap_units(db_reader: "InputSource", units: list[str]) -> list[str]:
    """The *units* that would be ``fork_no`` without their curve: one input arc, one output arc and not delayed
    (the classifier's arc counts, ``_derived._classify_process_method`` step 6)."""
    from flextool.engine.preprocess.params._derived import _try_entities, _try_param

    def counts(cls: str) -> dict[str, int]:
        df = _try_entities(db_reader, cls)
        if df is None or df.height == 0:
            return {}
        return dict(df.group_by(pl.col("unit").cast(pl.Utf8)).len().iter_rows())

    n_src, n_sink = counts("unit__inputNode"), counts("unit__outputNode")
    delay = _try_param(db_reader, "unit", "delay")
    delayed = set() if delay is None else set(delay.get_column("name").cast(pl.Utf8).to_list())
    return [u for u in units if n_src.get(u, 0) == 1 and n_sink.get(u, 0) == 1 and u not in delayed]


def _unit_set(names: list[str], like: "pl.DataFrame | None") -> pl.DataFrame:
    """A sorted ``(p,)`` set of *names*, in the ``p`` dtype of *like* when given."""
    out = pl.DataFrame({"p": sorted(names)}, schema={"p": pl.Utf8})
    if like is not None and "p" in like.columns:
        out = out.with_columns(pl.col("p").cast(like.schema["p"]))
    return out


def _emit_cap_per_unit(provider) -> pl.DataFrame:
    """``(p, d, cpu)`` (Utf8 keys): the capacity count of the rolling emit's ``p_flow_max``
    (``emit/_arc_unions.derive_p_flow_max``: ``p_entity_dispatch_capacity_max[p, d] / p_entity_unitsize[p]``, 1 as
    the unit size where none is given), read from the same provider frames."""
    from flextool.engine.preprocess._provider_io import _provider_key

    def frame(name: str) -> "pl.DataFrame | None":
        key = _provider_key(Path("solve_data") / name)
        return provider.get(key) if provider is not None and provider.has(key) else None

    unitsize: dict[str, float] = {}
    us = frame("p_entity_unitsize.csv")
    if us is not None:
        for row in us.iter_rows():
            if len(row) >= 2 and row[0] not in (None, ""):
                try:
                    unitsize[str(row[0])] = float(row[1])
                except (TypeError, ValueError):
                    continue
    rows: list[tuple[str, str, float]] = []
    dcm = frame("p_entity_dispatch_capacity_max.csv")
    if dcm is not None:
        for row in dcm.iter_rows():
            if len(row) >= 3 and row[0] not in (None, "") and row[1] not in (None, ""):
                try:
                    value = float(row[2])
                except (TypeError, ValueError):
                    continue
                rows.append((str(row[0]), str(row[1]), value / unitsize.get(str(row[0]), 1.0)))
    return pl.DataFrame(rows, schema={"p": pl.Utf8, "d": pl.Utf8, "cpu": pl.Float64}, orient="row")


def _is_synthetic_solve(db_reader: "InputSource", provider) -> bool:
    """The sub-solve is synthetic (a roll or a nested sub-solve, absent from the database): the loader's rule in
    ``_apply_db_overrides`` (its capacity parameters then come from the per-solve emit)."""
    from flextool.engine.preprocess.params import _derived as _drv

    if provider is None:
        return False
    active = _drv._read_active_solve(Path("."), provider=provider)
    return active is not None and not _drv._solve_in_spine(db_reader, active)


def _recompute_input_caps(flex_data, db_reader: "InputSource", provider, units: set[str],
                          multi_lines: pl.DataFrame) -> None:
    """Step 5: ``p_indirect_input_cap`` and the indirect input-arc rows of ``p_flow_upper`` of the piecewise
    *units*, from the overwritten ``p_slope`` / ``p_section`` (single-line units: full-load fuel ``slope · W +
    section``; multi-line units: ``max_g(slope_g · W + intercept_g)`` of *multi_lines*;
    ``_derived._indirect_input_cap_lf``).

        p_indirect_input_cap[p,s,p,d,t] = input_share_max · fuel · delay_factor / conversion_flow_coeff
        p_flow_upper[p,s,p,d,t]         = cap_per_unit[p,d] · p_indirect_input_cap[p,s,p,d,t]

    ``cap_per_unit`` as the producer of the other rows computed it: ``_derived.p_flow_upper_from_source`` (run
    again with the boundary's capacity inputs) on a solve of the database, the rolling emit's
    ``p_entity_dispatch_capacity_max / p_entity_unitsize`` on a synthetic sub-solve.  The other rows are kept.
    """
    from flextool.engine.preprocess.params import _derived as _drv

    pss, dt = flex_data.process_source_sink, flex_data.dt
    indirect = _names(getattr(flex_data, "process_indirect", None)) & units
    if not indirect or pss is None or pss.height == 0 or dt is None or dt.height == 0:
        return
    classified = _drv._classify_process_method(db_reader)
    lines = multi_lines if multi_lines.height > 0 else None
    cap = _drv.p_indirect_input_cap_from_source(db_reader, pss, dt, flex_data.p_slope, flex_data.p_section,
                                                classified, lines)
    flex_data.p_indirect_input_cap = _splice_rows(flex_data.p_indirect_input_cap, cap, indirect,
                                                  "p_indirect_input_cap")
    if cap is None or flex_data.p_flow_upper is None:
        return
    arcs = cap.frame.filter(pl.col("p").cast(pl.Utf8).is_in(sorted(indirect)))
    if _is_synthetic_solve(db_reader, provider):
        cpu = _emit_cap_per_unit(provider)
        upper = (_with_utf8_keys(arcs, ("p", "d"))
                 .join(cpu.rename({"p": "__p", "d": "__d"}), on=["__p", "__d"], how="left")
                 .with_columns(value=pl.col("cpu").fill_null(0.0) * pl.col("value"))
                 .select("p", "source", "sink", "d", "t", "value"))
    else:
        full = _drv.p_flow_upper_from_source(
            db_reader, pss, dt, p_slope=flex_data.p_slope, p_section=flex_data.p_section,
            p_unitsize=flex_data.p_unitsize, p_entity_max_units=flex_data.p_entity_max_units,
            p_entity_all_existing=flex_data.p_entity_all_existing, classified=classified, piecewise_lines=lines)
        if full is None:
            raise ValueError("piecewise_linear: p_flow_upper could not be recomputed for the input arcs of "
                             f"{sorted(indirect)}.")
        keys = ["__p", "__source", "__sink", "__d", "__t"]
        upper = (_with_utf8_keys(full.frame, ("p", "source", "sink", "d", "t"))
                 .join(_with_utf8_keys(arcs, ("p", "source", "sink", "d", "t")).select(keys), on=keys, how="semi")
                 .select("p", "source", "sink", "d", "t", "value"))
    fu = flex_data.p_flow_upper
    in_rows = _with_utf8_keys(upper, ("p", "source", "sink", "d", "t")).select(
        "__p", "__source", "__sink", "__d", "__t", pl.col("value").alias("__new"))
    keys = ["__p", "__source", "__sink", "__d", "__t"]
    joined = (_with_utf8_keys(fu.frame, ("p", "source", "sink", "d", "t")).with_row_index("__row")
              .join(in_rows, on=keys, how="left").sort("__row"))
    if joined.filter(pl.col("__new").is_not_null()).height != in_rows.height:
        raise ValueError("piecewise_linear: p_flow_upper lacks input-arc rows of "
                         f"{sorted(indirect)} ({in_rows.height} expected).")
    out = joined.with_columns(
        pl.when(pl.col("__new").is_not_null()).then(pl.col("__new")).otherwise(pl.col("value"))
        .cast(fu.frame.schema["value"]).alias("value")).select(fu.frame.columns)
    flex_data.p_flow_upper = Param(fu.dims, out, name=fu.name)


def apply_piecewise_curves(flex_data, db_reader: "InputSource", provider=None) -> None:
    """The load-boundary step of the ``piecewise_linear`` units of the solve (module docstring, steps 1-5).  No-op
    without such units.  A solve with a multi-line unit (two or more lines in some period) stops after the steps:
    the model has no rows for it yet."""
    from flextool.engine.input._param_shapes import piecewise_curve_frame
    from flextool.engine.input._piecewise_curve import curve_lines
    from flextool.engine.preprocess.params import _derived as _drv
    from flextool.engine.preprocess.params._branch import branch_anchor_pairs

    pss, dt = getattr(flex_data, "process_source_sink", None), getattr(flex_data, "dt", None)
    units = sorted(set(piecewise_units(db_reader)) & _names(pss))
    if not units or dt is None or dt.height == 0:
        return
    periods = _solve_periods(dt)
    pairs = branch_anchor_pairs(provider, periods) if provider is not None else None
    solve = _drv._read_active_solve(Path("."), provider=provider) if provider is not None else None
    points = solve_points(piecewise_curve_frame(db_reader), units, periods, pairs, solve)
    online = _names(getattr(flex_data, "process_online", None))
    lines = curve_lines(points, online)

    flags = _drv._piecewise_flags(db_reader).filter(pl.col("p").is_in(units))
    multi = sorted(flags.filter(pl.col("multi_line")).get_column("p").to_list())
    single = [u for u in units if u not in set(multi)]
    rising = sorted(flags.filter(pl.col("single_line_rising")).get_column("p").to_list())

    if flex_data.p_slope is None or tuple(flex_data.p_slope.dims) != ("p", "d", "t"):
        raise ValueError(f"piecewise_linear: p_slope must be a (p, d, t) Param at the load boundary; got "
                         f"{None if flex_data.p_slope is None else flex_data.p_slope.dims}.")

    # Step 3 — single-line units: min_load_efficiency slope / section.
    #   p_slope[p,d,t] = α[p,d] (· k_sink / k_src),  p_section[p,d,t] = β[p,d] (/ k_src)
    if single:
        values = single_line_values(lines.filter(pl.col("p").is_in(single)), _direct_ratio(db_reader))
        flex_data.p_slope = _overwrite_values(flex_data.p_slope, values.select("p", "d", pl.col("slope")
                                                                               .alias("value")),
                                              set(single), _KEYS_PD, "p_slope")
        if rising:
            _set_sections(flex_data, values.filter(pl.col("p").is_in(rising)), set(rising))

    # Step 4 — multi-line units: the line sets; p_slope = 1/η(1) (read by full-load measures only).
    multi_lines = lines.filter(pl.col("p").is_in(multi))
    if multi:
        full_load = (points.filter(pl.col("p").is_in(multi) & (pl.col("load") == 1.0))
                     .select("p", "d", value=1.0 / pl.col("eff")))
        flex_data.p_slope = _overwrite_values(flex_data.p_slope, full_load, set(multi), _KEYS_PD, "p_slope")
        like = getattr(flex_data, "process_online", None)
        flex_data.process_piecewise = _unit_set(multi, like)
        flex_data.process_piecewise_needs_online = _unit_set(
            flags.filter(pl.col("needs_online")).get_column("p").to_list(), like)
        flex_data.process_piecewise_remap = _unit_set(_remap_units(db_reader, multi), like)
        flex_data.p_piecewise_slope = Param(("p", "seg", "d"),
                                            multi_lines.select("p", "seg", "d", pl.col("slope").alias("value")))
        flex_data.p_piecewise_intercept = Param(("p", "seg", "d"), multi_lines.select(
            "p", "seg", "d", pl.col("intercept").alias("value")))
        flex_data.p_piecewise_points = points.filter(pl.col("p").is_in(multi))

    # Step 5 — the capacity parameters derived from the seeded slope.
    _recompute_input_caps(flex_data, db_reader, provider, set(units), multi_lines)

    if multi:
        raise FlexToolConfigError(
            f"piecewise_linear with more than one line is not implemented yet (unit "
            f"{', '.join(repr(u) for u in multi)}).")


def _set_sections(flex_data, values: pl.DataFrame, rising: set[str]) -> None:
    """``p_section`` rows ``(p, d, t)`` of the *rising* single-line units (their intercept per ``(p, d)``,
    *values* ``(p, d, slope, section)``, promoted onto the solve's ``dt``) and their membership of
    ``process_min_load_eff``; the rows of other units are kept and the result is sorted by ``(p, d, t)`` as
    ``_derived.p_section_from_source`` sorts it."""
    from flextool.engine.input._param_shapes import promote_param_to_dt

    slope_schema = flex_data.p_slope.frame.schema
    dt = flex_data.dt
    pd_frame = values.select(pl.col("p").cast(slope_schema["p"]), pl.col("d").cast(dt.schema["d"]),
                             pl.col("section").alias("value"))
    new = (promote_param_to_dt(Param(("p", "d"), pd_frame), dt).select("p", "d", "t", "value").collect())
    old = flex_data.p_section
    if old is not None and old.frame.height > 0:
        kept = old.frame.filter(~pl.col("p").cast(pl.Utf8).is_in(sorted(rising)))
        new = pl.concat([kept, new.cast(kept.schema)], how="vertical")
        dims, name = old.dims, old.name
    else:
        dims, name = ("p", "d", "t"), None
    flex_data.p_section = Param(dims, new.sort("p", "d", "t"), name=name)

    mle = getattr(flex_data, "process_min_load_eff", None)
    add = _unit_set(sorted(rising), mle if mle is not None and mle.width > 0 else None)
    if mle is not None and mle.height > 0:
        merged = pl.concat([mle.select("p"), add.cast(mle.select("p").schema)], how="vertical")
    else:
        merged = add
    flex_data.process_min_load_eff = merged.unique().sort("p")
