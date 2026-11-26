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

Multi-line units are modelled by ``conversion_piecewise`` (``model/core/_conversion._eq_conversion_piecewise``), on the
parameters of step 4.
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
    (:func:`flextool.engine.preprocess.params._projection.single_input_curve_arc_remap`)."""
    from flextool.engine.preprocess.params._projection import single_input_curve_arc_remap

    remap = set(single_input_curve_arc_remap(db_reader).get_column("p").to_list())
    return [u for u in units if u in remap]


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
        # The output arc of a one-input unit whose input_share_max is an output cap: cap_per_unit · its
        # capacity_max_coeff as remapped (_projection.remap_capacity_coef; the emit's p_flow_max has the authored one).
        out_rows = _remapped_output_upper(db_reader, pss, dt, cpu, indirect)
        if out_rows is not None:
            upper = pl.concat([upper, out_rows.select(upper.columns).cast(upper.schema)], how="vertical")
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


def _remapped_output_upper(db_reader: "InputSource", pss: pl.DataFrame, dt: pl.DataFrame, cpu: pl.DataFrame,
                           units: set[str]) -> "pl.DataFrame | None":
    """``(p, source, sink, d, t, value)`` (Utf8 keys): the ``p_flow_upper`` of the output arcs ``(p, p, o)`` of the
    one-input *units* with an authored ``input_share_max`` (a remapped output cap), ``cap_per_unit[p,d] ·
    capacity_max_coeff'[p,o]`` (``derive_p_flow_max``'s formula with the remapped coefficient); None without
    such arcs."""
    from flextool.engine.preprocess.params._derived import _arc_max_capacity_coef_lf, _try_param_explicit
    from flextool.engine.preprocess.params._projection import single_input_curve_arc_remap

    remap = single_input_curve_arc_remap(db_reader).filter(pl.col("p").is_in(sorted(units)))
    ism = _try_param_explicit(db_reader, "unit__inputNode", "input_share_max")
    if remap.height == 0 or ism is None or ism.height == 0:
        return None
    capped = remap.join(ism.select(pl.col("unit").cast(pl.Utf8).alias("p"),
                                   pl.col("node").cast(pl.Utf8).alias("source")), on=["p", "source"], how="semi")
    if capped.height == 0:
        return None
    coef = (_arc_max_capacity_coef_lf(db_reader, "sink").collect()
            .select(pl.col("p").cast(pl.Utf8), pl.col("sink").cast(pl.Utf8), pl.col("coef").cast(pl.Float64)))
    arcs = (capped.select("p", pl.col("p").alias("source"), "sink")
            .join(pss.select(*(pl.col(c).cast(pl.Utf8) for c in ("p", "source", "sink"))),
                  on=["p", "source", "sink"], how="semi")
            .join(coef, on=["p", "sink"], how="left")
            .with_columns(pl.col("coef").fill_null(1.0)))
    steps = dt.select(pl.col("d").cast(pl.Utf8), pl.col("t").cast(pl.Utf8))
    return (arcs.join(steps, how="cross")
            .join(cpu, on=["p", "d"], how="left")
            .select("p", "source", "sink", "d", "t", value=pl.col("cpu").fill_null(0.0) * pl.col("coef")))


def _user_solve(db_reader: "InputSource", solve: "str | None") -> "str | None":
    """The user's solve of the sub-solve *solve*: *solve* when it is a solve of the database, else the database
    solve it was made from (a roll ``<solve>_roll_<k>``, a realization ``<solve>__realized_dispatch…``, a nested
    ``<solve>_<period>``: :func:`_derived._resolve_synthetic_solve`)."""
    from flextool.engine.preprocess.params import _derived as _drv

    if solve is None or _drv._solve_in_spine(db_reader, solve):
        return solve
    resolved = _drv._resolve_synthetic_solve(db_reader, solve)
    return resolved[0] if resolved is not None else solve


def _check_user_solve_periods(db_reader: "InputSource", curves: pl.DataFrame, units: list[str],
                              user_solve: str) -> None:
    """E11 over the periods of the user's solve (the keys of its ``period_timeset``), once per run: every
    piecewise unit with a curve per period has one for each of them (:func:`solve_points`).  Stochastic branch
    periods are not keys of ``period_timeset``; they take their anchor's curve in each sub-solve."""
    from flextool.engine.input._param_shapes import _run_memo
    from flextool.engine.preprocess.params._derived import _try_param

    def check() -> bool:
        pt = _try_param(db_reader, "solve", "period_timeset")
        if pt is None or pt.height == 0:
            return True
        keys = [c for c in pt.columns if c not in ("name", "value")]
        if len(keys) != 1:
            return True  # a 2-level period map: its sub-solves are solves of their own (checked as such)
        periods = (pt.filter(pl.col("name").cast(pl.Utf8) == user_solve).get_column(keys[0]).cast(pl.Utf8)
                   .unique(maintain_order=True).to_list())
        if periods:
            solve_points(curves, units, periods, None, user_solve)
        return True

    _run_memo(db_reader, ("piecewise_user_solve_periods", user_solve, tuple(units)), check)


def apply_piecewise_curves(flex_data, db_reader: "InputSource", provider=None) -> None:
    """The load-boundary step of the ``piecewise_linear`` units of the solve (module docstring).  No-op without
    such units."""
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
    user_solve = _user_solve(db_reader, solve)
    curves = piecewise_curve_frame(db_reader)
    if user_solve is not None and user_solve != solve:
        # A roll (or another sub-solve the engine made): E11 is checked once over every period of the user's
        # solve, at its first sub-solve, so a long run does not stop late.
        _check_user_solve_periods(db_reader, curves, units, user_solve)
    points = solve_points(curves, units, periods, pairs, user_solve)
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

    # Step 6 — one-input units: the input's ramp limit, ramp cost, profile and user-constraint coefficient act on
    # the output flow.
    if multi:
        from flextool.engine.preprocess.params._projection import single_input_curve_arc_remap
        remap = single_input_curve_arc_remap(db_reader).filter(pl.col("p").is_in(multi))
        if remap.height > 0:
            remap_input_items(flex_data, remap)



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


# ---------------------------------------------------------------------------
# Step 6: the one-input remap (design §3.4)

_ARC = ("p", "source", "sink")


def _frame_of(x) -> "pl.DataFrame | None":
    if x is None:
        return None
    f = x.frame if isinstance(x, Param) else x
    return f.collect() if isinstance(f, pl.LazyFrame) else f


def _arc_map(remap: pl.DataFrame) -> pl.DataFrame:
    """``(__p, __source, __sink, __to_source, __to_sink)`` Utf8: each remapped unit's input arc ``(p, s, p)`` and
    its output arc ``(p, p, o)``."""
    return remap.select(pl.col("p").alias("__p"), pl.col("source").alias("__source"), pl.col("p").alias("__sink"),
                        pl.col("p").alias("__to_source"), pl.col("sink").alias("__to_sink"))


def _move_arcs(frame: "pl.DataFrame | None", remap: pl.DataFrame,
               like: "pl.DataFrame | None" = None) -> "tuple[pl.DataFrame | None, pl.DataFrame]":
    """``(rest, moved)``: the rows of *frame* (arc columns ``p, source, sink`` + others) without the input arcs of
    the remapped units, and those rows moved to their output arcs (in the dtypes of *like*, else of *frame*)."""
    if frame is None or frame.height == 0:
        return frame, pl.DataFrame()
    u = frame.with_columns(pl.col(k).cast(pl.Utf8).alias(f"__{k}") for k in _ARC)
    amap = _arc_map(remap)
    on = ["__p", "__source", "__sink"]
    hit = u.join(amap, on=on, how="inner")
    if hit.height == 0:
        return frame, pl.DataFrame()
    rest = u.join(amap.select(on), on=on, how="anti").select(frame.columns)
    target = like if like is not None and like.width > 0 else frame
    moved = hit.with_columns(pl.col("__p").alias("p"), pl.col("__to_source").alias("source"),
                             pl.col("__to_sink").alias("sink")).select(frame.columns)
    moved = moved.with_columns(pl.col(k).cast(target.schema[k]) for k in _ARC if k in target.columns)
    return rest, moved


def _union(frame: "pl.DataFrame | None", rows: pl.DataFrame, keys: list[str]) -> pl.DataFrame:
    """*frame* plus *rows* (same columns), unique on *keys*, sorted by *keys*."""
    if frame is None or frame.height == 0:
        return rows.unique(subset=keys, maintain_order=True).sort(keys)
    rows = rows.select(frame.columns).with_columns(pl.col(c).cast(frame.schema[c]) for c in frame.columns)
    return pl.concat([frame, rows], how="vertical").unique(subset=keys, maintain_order=True).sort(keys)


def _none_if_empty(frame: "pl.DataFrame | None") -> "pl.DataFrame | None":
    return None if frame is None or frame.height == 0 else frame


def remap_input_items(flex_data, remap: pl.DataFrame) -> None:
    """The input-side items of the one-input multi-line units *remap* ``(p, source, sink)`` moved from the input
    arc ``(p, s, p)`` to the output arc ``(p, p, o)``, as if authored on ``unit__outputNode (p, o)``
    (``specs/piecewise_efficiency_final.md`` §3.4; their ``input_share_max`` is remapped where it is read,
    ``_projection.remap_capacity_coef``):

    * ramp limits: the source-family arcs go to the sink family (``ramp_sink_{up,down}_constraint`` on the
      output, scaled as the output's capacity); the speed is the input's, or the smaller of the two where the
      output has a limit in that direction too;
          ramp_speed_{dir}_sink'[p,o] = min(ramp_speed_{dir}_sink[p,o], ramp_speed_{dir}_source[p,s])
    * ramp cost: the input arc's cost is added to the output arc's (``ramp_change_eq`` on the output flow);
          p_ramp_cost_arc'[p,p,o] = p_ramp_cost_arc[p,p,o] + p_ramp_cost_arc[p,s,p]
    * profiles (``unit__node__profile`` on the input node): the profile rows act on the output arc;
    * ``constraint_flow_coeff`` on the input: the coefficient of the output arc in that constraint, added to an
      output coefficient there.
          p_flow_constraint_coef'[p,p,o,cn] = p_flow_constraint_coef[p,p,o,cn] + p_flow_constraint_coef[p,s,p,cn]

    The other input-side items keep acting on the input flow: reserves at the input node, the input's operating
    cost, commodity price and CO2, the start-up input, inertia / non-synchronous at the input node, and flowGroup
    membership through the input node (a direct unit's flowGroup term at its input node is its fuel use, which is
    the input flow of the curve unit).  Inertia at the input node of a unit without online variables is counted
    from the flow of its input arc (``_group_slack._eq_inertia``): the actual input, where a direct unit's one flow
    is its output.
    """
    remap = remap.select(*(pl.col(c).cast(pl.Utf8) for c in _ARC))
    fd = flex_data
    # Ramp limits and speeds.
    for dir_ in ("up", "down"):
        src_name, sink_name = (f"process_source_sink_ramp_limit_source_{dir_}",
                               f"process_source_sink_ramp_limit_sink_{dir_}")
        src_speed_name, sink_speed_name = f"p_ramp_speed_{dir_}_source", f"p_ramp_speed_{dir_}_sink"
        src_set, sink_set = _frame_of(getattr(fd, src_name)), _frame_of(getattr(fd, sink_name))
        rest, moved = _move_arcs(src_set, remap, sink_set)
        src_speed = getattr(fd, src_speed_name)
        sf = _frame_of(src_speed)
        if sf is not None and sf.height > 0:
            # The input's speed rows of the remapped units: moved to the output (below), the rest dropped.
            mine = sf.with_columns(pl.col("p").cast(pl.Utf8).alias("__p"), pl.col("source").cast(pl.Utf8)
                                   .alias("__s")).join(remap.select(pl.col("p").alias("__p"),
                                                                    pl.col("source").alias("__s"),
                                                                    pl.col("sink").alias("__o")),
                                                       on=["__p", "__s"], how="left")
            in_speed = mine.filter(pl.col("__o").is_not_null()).select(
                "__p", "__o", pl.col("value").cast(pl.Float64).alias("__in"))
            kept = mine.filter(pl.col("__o").is_null()).select(sf.columns)
            setattr(fd, src_speed_name, None if kept.height == 0 else Param(src_speed.dims, kept,
                                                                            name=src_speed.name))
        else:
            in_speed = pl.DataFrame(schema={"__p": pl.Utf8, "__o": pl.Utf8, "__in": pl.Float64})
        if moved.height == 0:
            continue
        setattr(fd, src_name, _none_if_empty(rest))
        had_limit = (pl.DataFrame(schema={"__p": pl.Utf8, "__o": pl.Utf8}) if sink_set is None else
                     sink_set.select(pl.col("p").cast(pl.Utf8).alias("__p"),
                                     pl.col("source").cast(pl.Utf8).alias("__s"),
                                     pl.col("sink").cast(pl.Utf8).alias("__o"))
                     .filter(pl.col("__s") == pl.col("__p")).select("__p", "__o").unique())
        moved_u = moved.select(pl.col("p").cast(pl.Utf8).alias("__p"), pl.col("sink").cast(pl.Utf8).alias("__o"))
        speeds = moved_u.join(in_speed, on=["__p", "__o"], how="left")
        if speeds.filter(pl.col("__in").is_null()).height > 0:
            raise ValueError(f"piecewise_linear: a ramp-limited input arc without a {src_speed_name} row: "
                             f"{speeds.filter(pl.col('__in').is_null()).rows()}.")
        sink_speed = getattr(fd, sink_speed_name)
        kf = _frame_of(sink_speed)
        if kf is not None and kf.height > 0:
            old = kf.select(pl.col("p").cast(pl.Utf8).alias("__p"), pl.col("sink").cast(pl.Utf8).alias("__o"),
                            pl.col("value").cast(pl.Float64).alias("__out"))
            speeds = speeds.join(old, on=["__p", "__o"], how="left")
        else:
            speeds = speeds.with_columns(__out=pl.lit(None, pl.Float64))
        speeds = (speeds.join(had_limit.with_columns(__lim=pl.lit(True)), on=["__p", "__o"], how="left")
                  .with_columns(pl.col("__lim").fill_null(False)))
        speeds = speeds.select("__p", "__o", value=pl.when(pl.col("__lim") & pl.col("__out").is_not_null())
                               .then(pl.min_horizontal("__out", "__in")).otherwise(pl.col("__in")))
        if kf is not None and kf.height > 0:
            upd = (kf.with_columns(pl.col("p").cast(pl.Utf8).alias("__p"), pl.col("sink").cast(pl.Utf8).alias("__o"))
                   .join(speeds.rename({"value": "__v"}), on=["__p", "__o"], how="left")
                   .with_columns(pl.coalesce("__v", pl.col("value").cast(pl.Float64)).cast(kf.schema["value"])
                                 .alias("value")))
            new_rows = speeds.join(upd.select("__p", "__o"), on=["__p", "__o"], how="anti").select(
                pl.col("__p").cast(kf.schema["p"]).alias("p"), pl.col("__o").cast(kf.schema["sink"]).alias("sink"),
                pl.col("value").cast(kf.schema["value"]))
            frame = pl.concat([upd.select(kf.columns), new_rows.select(kf.columns)], how="vertical")
            setattr(fd, sink_speed_name, Param(sink_speed.dims, frame, name=sink_speed.name))
        else:
            setattr(fd, sink_speed_name, Param(("p", "sink"), speeds.select(
                pl.col("__p").alias("p"), pl.col("__o").alias("sink"), "value")))
        setattr(fd, sink_name, _union(sink_set, moved, list(_ARC)))

    # Ramp cost.
    cost = getattr(fd, "p_ramp_cost_arc", None)
    cf = _frame_of(cost)
    if cf is not None and cf.height > 0:
        rest, moved = _move_arcs(cf, remap)
        if moved.height > 0:
            summed = (pl.concat([rest, moved], how="vertical")
                      .group_by(list(_ARC), maintain_order=True).agg(pl.col("value").sum())
                      .sort(list(_ARC)))
            fd.p_ramp_cost_arc = Param(cost.dims, summed.select(cf.columns), name=cost.name)
            arcs = summed.filter(pl.col("value") > 0.0).select(_ARC)
            rc = _frame_of(getattr(fd, "process_source_sink_ramp_cost", None))
            if rc is not None and rc.width > 0:
                arcs = arcs.with_columns(pl.col(c).cast(rc.schema[c]) for c in _ARC)
            fd.process_source_sink_ramp_cost = _none_if_empty(arcs)

    # Profiles.
    for name in ("process_profile_upper", "process_profile_lower", "process_profile_fixed"):
        prof = _frame_of(getattr(fd, name))
        rest, moved = _move_arcs(prof, remap)
        if moved.height > 0:
            setattr(fd, name, _union(rest, moved, [*_ARC, "f"]))

    # User-constraint flow coefficients.
    coef = getattr(fd, "p_flow_constraint_coef", None)
    kf = _frame_of(coef)
    if kf is not None and kf.height > 0:
        rest, moved = _move_arcs(kf, remap)
        if moved.height > 0:
            summed = (pl.concat([rest, moved], how="vertical")
                      .group_by([*_ARC, "cn"], maintain_order=True).agg(pl.col("value").sum())
                      .sort([*_ARC, "cn"]))
            fd.p_flow_constraint_coef = Param(coef.dims, summed.select(kf.columns), name=coef.name)
    idx = _frame_of(getattr(fd, "flow_constraint_idx", None))
    rest, moved = _move_arcs(idx, remap)
    if moved.height > 0:
        fd.flow_constraint_idx = _union(rest, moved, [*_ARC, "cn"])
