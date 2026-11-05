"""Coarse processes: one value per coarse step of their nodes (``FlexData.processCoarse``).

A unit or connection whose every balance node is coarse (``preprocess/params/_coarse.coarse_process_frames``) has
one ``v_flow`` / ``v_flow_back`` column per coarse step b of its nodes, a rate that stays the same over the fine
steps t of b.  The registered variable has a row only at each coarse step's first fine step ``b_first``
(:func:`domain`); the model reads it through an *alias* (:func:`alias`) that is indexed on every fine step like a
fine variable and maps each fine row to the column of its coarse step.  Every term that reads the alias is
unchanged: a sum over the fine steps of a coarse step (the coarse node balance, costs, CO2, cumulative flow) adds
the fine coefficients on the one column, Σ_t v·Δt·slope_t = v_b·Σ_t Δt·slope_t.  :func:`alias` serves any *level*
of a coarse process keyed ``(p, …, d, t)`` (a value that holds through the step, like a rate): its units online
(``_online``), its reserve (``_reserve``) and the direction of an exact connection (``_exact``) are aliased like its
flows; :func:`alias` with ``steps`` serves a level of another coarse entity: the angle of a coarse-angle node
(``_dc_power_flow``).
The *events* of a coarse process (start-ups, shut-downs) are registered at ``b_first`` only and are not aliased: a
fine row of an interior step has no event term (an alias would count one start-up on every fine step of a sum
over t).

The arc-local rows of a coarse process (capacity, output sum, conversion, input share, profiles, the online
capacity, the minimum load and the reserve limits) are not written per fine step: :func:`add_cstr` writes them once per coarse step as
the twin family ``<family>_coarse``, the fine row's Δt-weighted average over the step (EQ. header in
:func:`add_cstr`).  A constant flow could not meet a per-step equality while a parameter varies within the step;
the average lets time-varying availability, profile, efficiency and capacity coefficients act through their step
means.

Rows that couple several entities follow the *row rule* (:func:`add_row_rule_cstr`): a row whose every member is
coarse (:func:`row_partition`; DC power flow, the storage reserve rows) is one averaged row per coarse step; a row
that mixes coarse and fine members stays per solve step and reads the coarse terms at their constant value.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import polars as pl
from polar_high import Param, Sum
from polar_high.engine import Expr, Var

from flextool.engine.model._common import _has_rows

if TYPE_CHECKING:
    from flextool.engine.model.core import _Build

_KEYS = ("p", "d", "t")


def prepare(b: "_Build") -> None:
    """Set ``b.has_coarse_proc``, ``b.coarse_steps`` ``(p, d, b_first, t, w)`` (p, d, t in the dtypes of
    ``pss_dt``, b_first Utf8) and ``b.coarse_w`` (the Param ``w[p,d,b_first,t] = Δt / Δ_b``)."""
    d = b.d
    steps = getattr(d, "process_coarse_step", None)
    if not b.has_proc or not _has_rows(getattr(d, "processCoarse", None)) or not _has_rows(steps):
        return
    schema = b.pss_dt.schema
    frame = (steps.select(*(pl.col(c).cast(pl.Utf8).cast(schema[c], strict=False) for c in _KEYS),
                          pl.col("b_first").cast(pl.Utf8), pl.col("w").cast(pl.Float64))
             .drop_nulls(list(_KEYS))
             .select("p", "d", "b_first", "t", "w"))
    frame = frame.join(b.pss_dt.select("p").unique(), on="p", how="semi")
    if frame.height == 0:
        return
    b.has_coarse_proc = True
    b.coarse_steps = frame
    b.coarse_w = Param(("p", "d", "b_first", "t"), frame.rename({"w": "value"}))


def coarse_procs(b: "_Build") -> pl.DataFrame:
    """``(p,)`` of the coarse processes, in the dtype of ``pss_dt``."""
    return b.coarse_steps.select("p").unique(maintain_order=True)


def _step_of(b: "_Build", frame: pl.DataFrame, steps: "pl.DataFrame | None" = None) -> pl.DataFrame:
    """``frame`` with ``b_first`` (Utf8) of every row of a coarse entity (null for the other rows).  ``steps``
    ``(key, d, b_first, t)``: the coarse steps of the entities in column ``key`` of ``frame`` (default the coarse
    processes' ``(p, d, b_first, t)``)."""
    if steps is None:
        steps = b.coarse_steps.select("p", "d", "t", "b_first")
    on = [c for c in steps.columns if c != "b_first"]
    steps = steps.select(*on, "b_first")
    left = frame
    for c in on:
        if left.schema[c] != steps.schema[c]:
            steps = steps.with_columns(pl.col(c).cast(left.schema[c], strict=False))
    return left.join(steps, on=on, how="left")


def domain(b: "_Build", index: pl.DataFrame, steps: "pl.DataFrame | None" = None) -> pl.DataFrame:
    """The rows of a variable's ``index`` (with ``p``, ``d``, ``t``; with ``steps``, its key column) that get a
    column: every row of a fine entity, and each coarse step's first fine step of a coarse one (``steps`` as in
    :func:`_step_of`)."""
    if not b.has_coarse_proc or (steps is not None and steps.height == 0):
        return index
    keep = (_step_of(b, index, steps)
            .filter(pl.col("b_first").is_null() | (pl.col("t").cast(pl.Utf8) == pl.col("b_first"))))
    return keep.select(index.columns)


def alias(b: "_Build", registered: Var, index: pl.DataFrame, steps: "pl.DataFrame | None" = None) -> Var:
    """``registered`` (a level declared over :func:`domain` of ``index``, dims ``(key, …, d, t)`` with ``t`` last)
    indexed on every row of ``index``: a coarse entity's fine row carries the column of its coarse step (``steps``
    as in :func:`_step_of`; default the coarse processes).  Not registered on the Problem; sorted by its dims like
    the registered frame."""
    if not b.has_coarse_proc or (steps is not None and steps.height == 0):
        return registered
    dims = registered.dims
    t_dtype = registered.frame.schema["t"]
    rows = (_step_of(b, index.select(dims), steps)
            .with_columns(pl.coalesce(pl.col("b_first").cast(t_dtype, strict=False), pl.col("t")).alias("__t_col")))
    cols = registered.frame.rename({"t": "__t_col"})
    frame = (rows.join(cols, on=[*dims[:-1], "__t_col"], how="inner")
             .select(*dims, "col_id")
             .sort(list(dims)))
    if frame.height != index.height:
        raise ValueError(f"{registered.name}: {index.height - frame.height} fine rows have no column of their "
                         "coarse step (the coarse steps of a coarse process must cover every step of the solve)")
    return Var(registered.name, dims, frame)


def _split(b: "_Build", over: pl.DataFrame,
           partition: "pl.DataFrame | None" = None) -> "tuple[pl.DataFrame, pl.DataFrame]":
    """``over`` (with ``p``, ``d``, ``t``) split into the rows of fine processes and the coarse rows, one per
    coarse step: ``over``'s columns with ``t`` replaced by ``b_first``.  With ``partition`` (:func:`row_partition`:
    ``(key…, d, b_first, t, w)``) the coarse rows are those of its keys instead."""
    if partition is None:
        with_b = _step_of(b, over)
    else:
        keys = [c for c in partition.columns if c not in ("b_first", "w")]
        steps = partition.select(*keys, "b_first")
        for c in keys:
            if over.schema[c] != steps.schema[c]:
                steps = steps.with_columns(pl.col(c).cast(over.schema[c], strict=False))
        with_b = over.join(steps, on=keys, how="left")
    fine = with_b.filter(pl.col("b_first").is_null()).select(over.columns)
    cols = [("b_first" if c == "t" else c) for c in over.columns]
    coarse = (with_b.filter(pl.col("b_first").is_not_null())
              .select(cols)
              .unique(maintain_order=True))
    return fine, coarse


def _ordered_sum(frame: pl.DataFrame, keys: list[str], order: str) -> pl.DataFrame:
    """``(keys…, value)``: ``frame``'s values summed per ``keys`` in ``order`` (and key) order, with
    ``numpy.add.reduceat`` on one contiguous array, so the bits of each sum do not depend on the row order a
    join produced or on how the frame is chunked."""
    srt = frame.sort([*keys, order])
    if srt.height == 0:
        return srt.select(*keys, "value")
    new = pl.any_horizontal([pl.col(c).ne_missing(pl.col(c).shift(1)) for c in keys]).fill_null(True)
    starts = np.flatnonzero(srt.select(new)[:, 0].to_numpy())
    sums = np.add.reduceat(srt["value"].to_numpy().astype(np.float64, copy=False), starts)
    return srt[starts].select(keys).with_columns(value=pl.Series("value", sums, dtype=pl.Float64))


def average(b: "_Build", term, w: "Param | None" = None):
    """``term`` (a row term of an arc-local family, open dims with ``t``) as its Δt-weighted mean over each coarse
    step: ``Σ_{t∈b} w[K,d,b,t] · term[…,t]`` with open dims ``(…, b_first)``; ``w`` the weights of the row's
    partition keyed ``(K, d, b_first, t)`` (default :attr:`coarse_w`, K = ``p``).  A term without ``t`` (and a
    scalar, and an event term on the coarse grid) is the same on every fine step of the step, so it is returned as
    it is (Σ_t w = 1)."""
    w = b.coarse_w if w is None else w
    if isinstance(term, (int, float)):
        return term
    if isinstance(term, Param):
        if "t" not in term.dims:
            return term
        frame = term.frame
        on = [c for c in w.dims if c != "b_first" and c in term.dims]
        wf = w.frame
        for c in on:
            if wf.schema[c] != frame.schema[c]:
                wf = wf.with_columns(pl.col(c).cast(frame.schema[c], strict=False))
        key = [c for c in w.dims if c not in ("d", "b_first", "t")]
        # A term keyed by t without d (a time map) takes the period of the weights: the steps of every period.
        at_t = ("d", "b_first") if "d" in w.dims and "d" not in term.dims else ("b_first",)
        dims = tuple(dict.fromkeys((*key, *(x for c in term.dims for x in (at_t if c == "t" else (c,))))))
        weighted = (frame.join(wf.rename({"value": "__w"}), on=on, how="inner")
                    .with_columns(value=pl.col("value") * pl.col("__w"))
                    .select(*dims, "t", "value"))
        return Param(dims, _ordered_sum(weighted, list(dims), "t"))
    if isinstance(term, Var):
        return Sum(term * w, over=("t",)) if "t" in term.dims else term
    if isinstance(term, Expr):
        with_t = [x for x in term.terms if "t" in x.dims]
        if not with_t:
            return term
        avg = Sum(Expr(with_t) * w, over=("t",))
        rest = [x for x in term.terms if "t" not in x.dims]
        return avg + Expr(rest) if rest else avg
    raise TypeError(f"average: unsupported term {type(term).__name__}")


def add_cstr(b: "_Build", name: str, *, over: pl.DataFrame, sense: str, lhs_terms: dict,
             rhs_terms: dict, fine_rhs: "dict | None" = None, coarse_rhs: "dict | None" = None,
             partition: "pl.DataFrame | None" = None) -> None:
    """``m.add_cstr(name, …)`` for the rows of fine processes and ``<name>_coarse`` for those of coarse ones (the
    EQ. header below).  ``fine_rhs`` holds right-hand-side terms of the fine rows only (after ``rhs_terms``);
    ``coarse_rhs`` the event terms E of the coarse rows, already on the coarse grid (open dims with ``b_first``,
    :func:`at_step`), added to the averaged right-hand side unchanged.  ``over`` may name a coarse row by its
    coarse step's first fine step.  ``partition`` (:func:`row_partition`) makes the rows of its keys coarse instead
    of those of coarse processes: a row key whose every member is coarse (the row rule); the other rows stay per
    solve step and read the coarse terms at their constant value."""
    m = b.m
    if (not b.has_coarse_proc or (partition is not None and partition.height == 0)
            or (partition is None and "p" not in over.columns)):
        m.add_cstr(name, over=over, sense=sense, lhs_terms=lhs_terms, rhs_terms={**rhs_terms, **(fine_rhs or {})})
        return
    fine, coarse = _split(b, over, partition)
    if fine.height > 0:
        m.add_cstr(name, over=fine, sense=sense, lhs_terms=lhs_terms, rhs_terms={**rhs_terms, **(fine_rhs or {})})
    # ─── EQ. ─── <family>_coarse: maxFlow_coarse, maxFlow_negCap_coarse, maxFlow_back_coarse, maxOutputSum_coarse,
    #             maxOutputSum_negCap_coarse, conversion_indirect_coarse, minInputShare_coarse,
    #             profile_flow_{upper_limit,lower_limit,fixed}_coarse, maxFlow_online{sfx}_coarse,
    #             maxFlow_online_shutdown{sfx}_coarse, maxFlow_online_shutdown_branch{sfx}_coarse,
    #             maxFlow_back_online{sfx}_coarse, maxInputFlow_online{sfx}_coarse,
    #             maxInputFlow_online_shutdown{sfx}_coarse, maxInputFlow_online_shutdown_branch{sfx}_coarse,
    #             maxOutputSum_online{sfx}_coarse, minFlow_minload{sfx}_coarse, minFlow_output_floor{sfx}_coarse,
    #             minFlow_reserve_coarse, minFlow_reserve_arc_coarse, reserve_process_{upward,downward}_coarse,
    #             connection_reserve_{up,down}_coarse, connection_reserve_{up,down}_online{sfx}_coarse,
    #             connection_reserve_{up,down}_exact_coarse, reserve_storage_up_floor_coarse,
    #             reserve_storage_down_headroom_coarse, maxFlow_exact_{forward,reverse}_coarse; by the row rule
    #             (add_row_rule_cstr): dc_flow_eq_coarse   ({sfx} = _linear / _integer) ──────────────────────
    # Σ_{t∈b} w[K,d,b,t] · ( LHS_t − RHS_t )  {≤,=,≥}  E[K,d,b]      w = Δt / Δ_b,  Δ_b = Σ_{t∈b} Δt
    #   ∀ coarse row key K (a coarse process p, or a row key whose every member is coarse: row_partition — a
    #   storage node n, a DC arc between coarse-angle nodes), (…, d, b)
    #   its coarse steps (b = its first fine step b_first);  LHS_t / RHS_t = the level terms of the fine row of the
    #   family at (d, t);  E = the row's event terms on the coarse grid (start-ups at b, shut-downs at the coarse
    #   successor b⁺: the capacity reduction of maxFlow_online / maxInputFlow_online and their _shutdown rows)
    # The fine row's Δt-weighted average over the coarse step.  The flow is one column per coarse step, so its own
    # term averages to itself; time-varying availability, profile, efficiency (slope) and capacity coefficients act
    # through their step means (the mean of a product, not the product of means); so do the units online and the
    # reserve (one value per coarse step) times a time-varying availability, profile or minimum load.  With one
    # time-varying limit per arc the averaged row allows exactly the energy the fine rows allow over the step;
    # with several it is their convex relaxation (it can be looser).
    if coarse.height > 0:
        w = None
        if partition is not None:
            keys = [c for c in partition.columns if c != "w"]
            w = Param(tuple(keys), partition.rename({"w": "value"}))
        add_twin(b, name, over=coarse, sense=sense, lhs_terms=lhs_terms, rhs_terms=rhs_terms,
                 coarse_rhs=coarse_rhs, w=w)


def add_twin(b: "_Build", name: str, *, over: pl.DataFrame, sense: str, lhs_terms: dict, rhs_terms: dict,
             coarse_rhs: "dict | None" = None, w: "Param | None" = None) -> None:
    """``<name>_coarse`` over the coarse rows ``over`` (``t`` replaced by ``b_first``): the averaged terms (with the
    weights ``w`` of the rows' partition, default the coarse processes' :attr:`coarse_w`) and the event terms
    ``coarse_rhs`` (EQ. header in :func:`add_cstr`)."""
    b.m.add_cstr(f"{name}_coarse", over=over, sense=sense,
                 lhs_terms={k: average(b, v, w) for k, v in lhs_terms.items()},
                 rhs_terms={**{k: average(b, v, w) for k, v in rhs_terms.items()}, **(coarse_rhs or {})})


def row_partition(b: "_Build", members: pl.DataFrame, like: pl.DataFrame,
                  node_steps: "pl.DataFrame | None" = None) -> "pl.DataFrame | None":
    """The coarse steps ``(key…, d, b_first, t, w)`` of the row keys of ``members`` ``(key…, kind, e)`` whose every
    member is coarse (``params._coarse.coarse_row_steps``: kind ``p`` a process with its coarse steps, kind ``n`` a
    coarse node — of ``node_steps`` ``(n, d, b_first, t)`` when given, else of ``node_coarse_step``), with ``key…``,
    ``d`` and ``t`` in the dtypes of the rows' index ``like`` and ``b_first`` Utf8; ``None`` without such a key."""
    from flextool.engine.preprocess.params._coarse import coarse_row_steps

    if not b.has_coarse_proc or members.height == 0:
        return None
    d = b.d
    psd = d.p_step_duration.frame if d.p_step_duration is not None else None
    nodes = getattr(d, "node_coarse_step", None) if node_steps is None else node_steps
    steps = coarse_row_steps(members, b.coarse_steps, nodes, psd)
    if steps.height == 0:
        return None
    keys = [c for c in members.columns if c not in ("kind", "e")]
    return steps.with_columns(*(pl.col(c).cast(like.schema[c]) for c in (*keys, "d", "t")))


def add_row_rule_cstr(b: "_Build", name: str, *, over: pl.DataFrame, sense: str, lhs_terms: dict,
                      rhs_terms: dict, partition: "pl.DataFrame | None") -> None:
    """A row family under the row rule: the rows of the keys of ``partition`` (:func:`row_partition`, every member
    coarse) as ``<name>_coarse``, one averaged row per coarse step (EQ. header in :func:`add_cstr`); every other row
    (a key with a fine member, or with no time-varying member) per solve step as ``name``, reading the coarse terms
    at their constant value.  Without such a key: ``m.add_cstr(name, …)`` unchanged."""
    if partition is None or partition.height == 0 or not b.has_coarse_proc:
        b.m.add_cstr(name, over=over, sense=sense, lhs_terms=lhs_terms, rhs_terms=rhs_terms)
        return
    add_cstr(b, name, over=over, sense=sense, lhs_terms=lhs_terms, rhs_terms=rhs_terms, partition=partition)


def at_step(b: "_Build", var: Var) -> Var:
    """The coarse-process columns of an event variable ``var`` (start-ups / shut-downs, registered at each coarse
    step's first fine step only, dims ``(p, …, d, t)``) keyed on the coarse grid: dims ``(p, …, d, b_first)``,
    ``b_first`` Utf8 (as the coarse rows of :func:`_split`).  :func:`average` passes it through unchanged."""
    procs = coarse_procs(b).select(pl.col("p").cast(pl.Utf8).alias("__p"))
    frame = (var.frame.with_columns(pl.col("p").cast(pl.Utf8).alias("__p"))
             .join(procs, on="__p", how="semi")
             .with_columns(pl.col("t").cast(pl.Utf8))
             .rename({"t": "b_first"})
             .select(*[("b_first" if c == "t" else c) for c in var.dims], "col_id"))
    return Var(var.name, tuple("b_first" if c == "t" else c for c in var.dims), frame)


def spans(b: "_Build") -> pl.DataFrame:
    """``(_p, _d, _b, __dur)`` (Utf8 keys): the duration Δ_b = Σ_{t∈b} Δt of every coarse step b (named by its first
    fine step) of every coarse process."""
    psd = b.d.p_step_duration.frame.select(pl.col("d").cast(pl.Utf8).alias("_d"),
                                           pl.col("t").cast(pl.Utf8).alias("_t"),
                                           pl.col("value").cast(pl.Float64).alias("__dt"))
    st = b.coarse_steps.select(pl.col("p").cast(pl.Utf8).alias("_p"), pl.col("d").cast(pl.Utf8).alias("_d"),
                               pl.col("b_first").alias("_b"), pl.col("t").cast(pl.Utf8).alias("_t"))
    return _ordered_sum(st.join(psd, on=["_d", "_t"], how="left").select("_p", "_d", "_b", "_t", value="__dt"),
                        ["_p", "_d", "_b"], "_t").rename({"value": "__dur"})


def coarse_links(b: "_Build", link: pl.DataFrame) -> pl.DataFrame:
    """``(_p, _d, _b, __d_prev, __t_prev, __dur, __dur_prev)`` (Utf8 keys): every coarse step b of a coarse process p
    with a predecessor b⁻ — the coarse step that holds the fine predecessor of b's first step (``link``,
    :func:`core.ramp_step_links`); ``__t_prev`` is b⁻'s first fine step, ``__dur`` / ``__dur_prev`` the durations
    Δ_b / Δ_b⁻.  Read the other way round, b is a coarse successor b⁺ of b⁻."""
    span = spans(b)
    st = b.coarse_steps.select(pl.col("p").cast(pl.Utf8).alias("_p"), pl.col("d").cast(pl.Utf8).alias("__d_prev"),
                               pl.col("t").cast(pl.Utf8).alias("__t_fine"), pl.col("b_first").alias("__t_prev"))
    return (span.join(link.select("_d", pl.col("_t").alias("_b"), "__d_prev", pl.col("__t_prev").alias("__t_fine")),
                      on=["_d", "_b"], how="inner")
            .join(st, on=["_p", "__d_prev", "__t_fine"], how="inner")
            .join(span.select("_p", pl.col("_d").alias("__d_prev"), pl.col("_b").alias("__t_prev"),
                              pl.col("__dur").alias("__dur_prev")), on=["_p", "__d_prev", "__t_prev"], how="inner")
            .select("_p", "_d", "_b", "__d_prev", "__t_prev", "__dur", "__dur_prev"))


def step_limit_share(ml: pl.Expr, rho: pl.Expr) -> pl.Expr:
    """The mean output share over a coarse step of a unit that starts at its step's start (or stops at its end):
    synchronised at ``ml`` and ramping at ``rho`` (the share of capacity the ramp speed covers in the whole step,
    ramp_speed · 60 · Δ_b) up to full load, ``ml + ½·ρ`` while the ramp does not reach full load (ρ ≤ 1 − ml) and
    ``1 − (1 − ml)² / (2ρ)`` once it does (full load for the rest of the step); 1 for ml ≥ 1."""
    return (pl.when(ml >= 1.0).then(pl.lit(1.0))
            .when(rho <= 1.0 - ml).then(ml + 0.5 * rho)
            .otherwise(1.0 - (1.0 - ml) * (1.0 - ml) / (2.0 * rho)))


def split_arcs(b: "_Build", arcs: pl.DataFrame) -> "tuple[pl.DataFrame, pl.DataFrame]":
    """``arcs`` (with ``p``) split into the arcs of fine processes and those of coarse processes."""
    coarse = pl.col("p").cast(pl.Utf8).is_in(coarse_procs(b)["p"].cast(pl.Utf8).implode())
    return arcs.filter(~coarse), arcs.filter(coarse)


def at_first_steps(b: "_Build", param: "Param | None") -> "Param | None":
    """``param`` (open dims with ``p``, ``d``, ``t``) with the rows of each coarse process replaced by their
    Δt-weighted mean over each coarse step, keyed at the step's first fine step; other rows unchanged."""
    if param is None or not b.has_coarse_proc or not {"p", "d", "t"}.issubset(param.dims):
        return param
    avg = average(b, param)
    t_dtype = param.frame.schema["t"]
    coarse = avg.frame.with_columns(pl.col("b_first").cast(t_dtype, strict=False).alias("t")).select(
        *param.dims, "value")
    for c in param.dims:
        if coarse.schema[c] != param.frame.schema[c]:
            coarse = coarse.with_columns(pl.col(c).cast(param.frame.schema[c], strict=False))
    fine, _ = split_arcs(b, param.frame)
    return Param(param.dims, pl.concat([fine.select(*param.dims, "value"), coarse]), name=param.name)


def ramp_rows(b: "_Build", arcs: pl.DataFrame, ramp_speed: Param, side: str, link: pl.DataFrame,
              carry: pl.DataFrame) -> pl.DataFrame:
    """The ramp rows of the coarse-process ``arcs`` ``(p, source, sink)``: one per coarse step b with a
    predecessor, in the columns of ``core._ramp_rows`` (``t`` = b_first).

    The predecessor of b is the coarse step b⁻ that holds the fine predecessor of b's first step (``link``,
    ``dt_ramp_link``); ρ̄ = ramp_speed · 60 · (Δ_b⁻ + Δ_b) / 2 with the coarse step durations (the distance
    between the two step centres).  A row inside one period with ρ̄ ≥ 1 cannot bind and is not built, so for a
    24-h step only a speed below 1/1440 per minute (≈ 6.9e-4) leaves in-period rows.  On a continuation roll the
    first coarse step of a period ramps from the previous roll's flow (``carry``, ``p_roll_continue_flow``) over
    that arc's carried step length (``p_roll_continue_flow_duration``: its last realized coarse step).  The reserve
    the arc holds in this direction is in these rows too (with the flow change); the reserve's own room at the solve's
    steps of each coarse step is :func:`reserve_room_rows`."""
    d = b.d
    keys = ["p", "source", "sink"]
    speed = ramp_speed.frame.select(pl.col("p").cast(pl.Utf8).alias("_p"), pl.col(side).cast(pl.Utf8).alias("_n"),
                                    pl.col("value").cast(pl.Float64).alias("__speed"))
    base = (arcs.select(keys).unique(maintain_order=True)
            .with_columns(*(pl.col(k).cast(pl.Utf8).alias(f"_{k}") for k in keys))
            .with_columns(pl.col(f"_{side}").alias("_n"))
            .join(speed, on=["_p", "_n"], how="inner")
            .join(spans(b), on="_p", how="inner"))
    with_link = (base.join(coarse_links(b, link).drop("__dur"), on=["_p", "_d", "_b"], how="inner")
                 .with_columns(__kind=pl.lit("link"), __f_hat=pl.lit(None, pl.Float64)))
    parts = [with_link.filter((pl.col("__d_prev") != pl.col("_d"))
                              | (pl.col("__speed") * 60.0 * (pl.col("__dur") + pl.col("__dur_prev")) / 2.0 < 1.0))]
    f_hat_p = getattr(d, "p_roll_continue_flow", None)
    if carry.height > 0 and f_hat_p is not None:
        f_hat = f_hat_p.frame.select(*(pl.col(k).cast(pl.Utf8).alias(f"_{k}") for k in (*keys, "d", "t")),
                                     pl.col("value").cast(pl.Float64).alias("__f_hat"))
        dur_p = getattr(d, "p_roll_continue_flow_duration", None)
        carried = (base.join(carry.select("_d", pl.col("_t").alias("_b"), "__d_prev", "__t_prev",
                                          pl.col("__dur_prev").alias("__dur_step")), on=["_d", "_b"], how="inner")
                   .join(f_hat.rename({"_t": "_b"}), on=["_p", "_source", "_sink", "_d", "_b"], how="inner"))
        if dur_p is not None:
            durs = dur_p.frame.select(*(pl.col(k).cast(pl.Utf8).alias(f"_{k}") for k in (*keys, "d")),
                                      pl.col("t").cast(pl.Utf8).alias("_b"),
                                      pl.col("value").cast(pl.Float64).alias("__dur_arc"))
            carried = carried.join(durs, on=["_p", "_source", "_sink", "_d", "_b"], how="left")
        else:
            carried = carried.with_columns(pl.lit(None, pl.Float64).alias("__dur_arc"))
        parts.append(carried.with_columns(pl.coalesce(pl.col("__dur_arc"), pl.col("__dur_step")).alias("__dur_prev"),
                                          __kind=pl.lit("carry")))
    t_dtype, d_dtype = b.pss_dt.schema["t"], b.pss_dt.schema["d"]
    cols = [*(pl.col(k) for k in keys), pl.col("_d").cast(d_dtype).alias("d"), pl.col("_b").cast(t_dtype).alias("t"),
            "__kind", "__d_prev", "__t_prev", "__f_hat",
            (pl.col("__speed") * 60.0 * (pl.col("__dur") + pl.col("__dur_prev")) / 2.0).alias("__room")]
    out = pl.concat([p.select(cols) for p in parts])
    return out.with_columns(*(pl.col(k).cast(arcs.schema[k]) for k in keys))


def reserve_room_rows(b: "_Build", arcs: pl.DataFrame, ramp_speed: Param, side: str, dir_: str,
                      link: pl.DataFrame, carry: pl.DataFrame) -> pl.DataFrame:
    """The reserve ramp-room rows of the coarse-process ``arcs`` ``(p, source, sink)`` in ramp direction ``dir_``:
    one per coarse step b on which the arc holds reserve in that direction (upward / downward reserve at its sink
    in the same direction, at its source in the opposite one, ``prundt``), in the columns of ``core._ramp_rows``
    (``__kind`` ``reserve``, ``t`` = b_first) with ``__room`` = min_{t∈b} ρ_t, only where it is < 1.

    ρ_t is the fine rule's room at each fine step t of b: ramp_speed · 60 · (Δt_prev + Δt) / 2 where t has a ramp
    predecessor (``link``; ``carry``: the previous roll's step with its carried duration), ramp_speed · 60 · Δt
    where it has none.  The reserve is one value per coarse step and the flow does not change within it, so the
    fine rows of the step (flow change 0) would each bound the reserve by their own room: the smallest binds.  The
    ramp between coarse steps (:func:`ramp_rows`) still holds the reserve with the flow change."""
    d = b.d
    keys = ["p", "source", "sink"]
    empty_cols = {**{k: arcs.schema[k] for k in keys}, "d": b.pss_dt.schema["d"], "t": b.pss_dt.schema["t"],
                  "__kind": pl.Utf8, "__d_prev": pl.Utf8, "__t_prev": pl.Utf8, "__f_hat": pl.Float64,
                  "__room": pl.Float64}
    prundt = getattr(d, "prundt", None)
    if prundt is None or prundt.height == 0 or arcs.height == 0:
        return pl.DataFrame(schema=empty_cols)
    speed = ramp_speed.frame.select(pl.col("p").cast(pl.Utf8).alias("_p"), pl.col(side).cast(pl.Utf8).alias("_n"),
                                    pl.col("value").cast(pl.Float64).alias("__speed"))
    st = b.coarse_steps.select(pl.col("p").cast(pl.Utf8).alias("_p"), pl.col("d").cast(pl.Utf8).alias("_d"),
                               pl.col("b_first").alias("_b"), pl.col("t").cast(pl.Utf8).alias("_t"))
    psd = d.p_step_duration.frame.select(pl.col("d").cast(pl.Utf8).alias("_d"), pl.col("t").cast(pl.Utf8).alias("_t"),
                                         pl.col("value").cast(pl.Float64).alias("__dt"))
    prev = pl.concat([link.select("_d", "_t", "__dur_prev"), carry.select("_d", "_t", "__dur_prev")]).unique(
        subset=["_d", "_t"], keep="first")
    held = prundt.select(pl.col("p").cast(pl.Utf8).alias("_p"), pl.col("ud").cast(pl.Utf8).alias("_ud"),
                         pl.col("n").cast(pl.Utf8).alias("_rn"), pl.col("d").cast(pl.Utf8).alias("_d"),
                         pl.col("t").cast(pl.Utf8).alias("_t"))
    at_sink = held.filter(pl.col("_ud") == dir_).select("_p", pl.col("_rn").alias("_sink"), "_d", "_t").unique()
    at_source = held.filter(pl.col("_ud") != dir_).select("_p", pl.col("_rn").alias("_source"), "_d", "_t").unique()
    base = (arcs.select(keys).unique(maintain_order=True)
            .with_columns(*(pl.col(k).cast(pl.Utf8).alias(f"_{k}") for k in keys))
            .with_columns(pl.col(f"_{side}").alias("_n"))
            .join(speed, on=["_p", "_n"], how="inner")
            .join(st, on="_p", how="inner"))
    hold = pl.concat([base.join(at_sink, on=["_p", "_sink", "_d", "_t"], how="semi"),
                      base.join(at_source, on=["_p", "_source", "_d", "_t"], how="semi")]).select(
        "_p", "_source", "_sink", "_d", "_b").unique()
    rho = (base.join(psd, on=["_d", "_t"], how="left")
           .join(prev, on=["_d", "_t"], how="left")
           .with_columns(__rho=pl.when(pl.col("__dur_prev").is_not_null())
                         .then(pl.col("__speed") * 60.0 * (pl.col("__dt") + pl.col("__dur_prev")) / 2.0)
                         .otherwise(pl.col("__speed") * 60.0 * pl.col("__dt")))
           .join(hold, on=["_p", "_source", "_sink", "_d", "_b"], how="semi")
           .group_by(*keys, "_d", "_b")
           .agg(pl.col("__rho").min().alias("__room"))
           .filter(pl.col("__room") < 1.0))
    t_dtype, d_dtype = b.pss_dt.schema["t"], b.pss_dt.schema["d"]
    return rho.select(*(pl.col(k).cast(arcs.schema[k]) for k in keys), pl.col("_d").cast(d_dtype).alias("d"),
                      pl.col("_b").cast(t_dtype).alias("t"), __kind=pl.lit("reserve"),
                      __d_prev=pl.lit(None, pl.Utf8), __t_prev=pl.lit(None, pl.Utf8),
                      __f_hat=pl.lit(None, pl.Float64), __room=pl.col("__room"))
