"""The flows of coarse processes on the solve's steps.

A coarse process (``FlexData.processCoarse``, :func:`flextool.engine.preprocess.params._coarse.coarse_process_frames`)
has one ``v_flow`` / ``v_flow_back`` column per coarse step of its nodes: the LP row is keyed by the step's first
fine step ``b_first``.  The value is a rate (MW per unitsize) that holds through the whole coarse step, so every
reader that needs the flows on the solve's steps copies it to each fine step of the coarse step
(``energy = Σ_t flow · Δt`` stays the LP's energy).  Every such reader goes through this module:

* :func:`flow_values` — ``Solution.value`` of a flow variable on every fine step (CO2 accumulator, fix_usage
  handoff, the ramp roll carry);
* :func:`expand_coarse_flows` — the same for an LP-shaped long frame;
* :func:`expand_value_keys` — the HiGHS-name reader of the output parquets
  (:func:`flextool.process_outputs.read_highs_solution.extract_variable`).

A value is only ever copied to a fine step that the LP does not have a column for, so an LP whose flows are all
fine (no coarse process, or a solve that kept the process fine) is returned unchanged.  The raw horizon dump
(:mod:`flextool.engine.output._horizon_dump`) stays LP-shaped: it writes the LP's columns, a coarse process's at
``b_first`` only.  Benders reads only the half-flows of cross-region arcs, which are never coarse.
"""
from __future__ import annotations

import polars as pl

#: The flow variables whose coarse-process columns live on coarse steps.
COARSE_FLOW_VARS: tuple[str, ...] = ("v_flow", "v_flow_back")


def coarse_steps(flex_data) -> "pl.DataFrame | None":
    """``(p, d, b_first, t)`` (Utf8) of the coarse processes of ``flex_data``; ``None`` without any."""
    steps = getattr(flex_data, "process_coarse_step", None) if flex_data is not None else None
    if steps is None or steps.height == 0:
        return None
    return steps.select(*(pl.col(c).cast(pl.Utf8) for c in ("p", "d", "b_first", "t")))


def expand_coarse_flows(frame: "pl.DataFrame | None", flex_data, *, p: str = "p", d: str = "d",
                        t: str = "t", value: str = "value") -> "pl.DataFrame | None":
    """``frame`` (an LP-shaped flow frame: key columns with ``p``, ``d``, ``t``, and ``value``) with each
    coarse-process row at a coarse step's first fine step copied to the other fine steps of the step that the
    frame has no row for.  Rows keep the frame's order (each copied row right after its source row, then in solve
    order) and dtypes; the other columns are copied unchanged."""
    steps = coarse_steps(flex_data)
    if frame is None or steps is None or frame.height == 0:
        return frame
    keys = [c for c in frame.columns if c != value]
    others = (steps.filter(pl.col("t") != pl.col("b_first")).with_row_index("__j", offset=1)
              .rename({"p": "__p", "d": "__d", "b_first": "__b", "t": "__t"}))
    lead = frame.with_row_index("__i")
    copies = (lead.with_columns(pl.col(p).cast(pl.Utf8).alias("__p"), pl.col(d).cast(pl.Utf8).alias("__d"),
                                pl.col(t).cast(pl.Utf8).alias("__b"))
              .join(others, on=["__p", "__d", "__b"], how="inner")
              .with_columns(pl.col("__t").cast(frame.schema[t]).alias(t))
              .select(*frame.columns, "__i", "__j")
              .join(frame.select(keys), on=keys, how="anti"))
    if copies.height == 0:
        return frame
    out = pl.concat([lead.with_columns(pl.lit(0, dtype=pl.UInt32).alias("__j")).select(copies.columns), copies])
    return out.sort("__i", "__j").select(frame.columns)


def flow_values(sol, flex_data, name: str = "v_flow") -> "pl.DataFrame | None":
    """``sol.value(name)`` on every fine step (:func:`expand_coarse_flows`); ``None`` when the solution has no
    such variable."""
    if sol is None or name not in getattr(sol, "_vars", {}):
        return None
    frame = sol.value(name)
    if name not in COARSE_FLOW_VARS:
        return frame
    return expand_coarse_flows(frame, flex_data)


def coarse_step_members(steps: "pl.DataFrame | None") -> "dict[tuple[str, str, str], list[str]]":
    """``(p, d, b_first)`` -> the other fine steps of the coarse step, in solve order (:func:`coarse_steps`)."""
    out: dict[tuple[str, str, str], list[str]] = {}
    if steps is None:
        return out
    for p, d, b, t in steps.iter_rows():
        if t != b:
            out.setdefault((p, d, b), []).append(t)
    return out


def expand_value_keys(values_by_key: "dict[tuple[str, ...], float]", steps: "pl.DataFrame | None") -> None:
    """In place: each key ``(d, t, p, …)`` of a coarse process at its coarse step's first fine step gets a copy
    for every other fine step of the step that has no key of its own (the HiGHS-name reader's row keys: period,
    time, then the process column first)."""
    members = coarse_step_members(steps)
    if not members:
        return
    extra: dict[tuple[str, ...], float] = {}
    for key, val in values_by_key.items():
        others = members.get((key[2], key[0], key[1]))
        if others:
            for t in others:
                copy = (key[0], t, *key[2:])
                if copy not in values_by_key:
                    extra[copy] = val
    values_by_key.update(extra)
