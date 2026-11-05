"""The variables of coarse processes on the solve's steps.

A coarse process (``FlexData.processCoarse``, :func:`flextool.engine.preprocess.params._coarse.coarse_process_frames`)
has one column per coarse step of its nodes for each of its time-indexed variables (:data:`COARSE_VARS`): the LP
row is keyed by the step's first fine step ``b_first``.  So does the angle of a coarse-angle node
(:data:`COARSE_VAR_STEPS`).  Every reader that needs the values on the solve's steps
expands them to every fine step of the coarse step, by the variable's mode:

* ``copy`` — a *level* that holds through the whole step (a flow is a rate in MW per unitsize, units online a
  count, a reserve a capacity held): the value is copied to each fine step, so ``energy = Σ_t flow · Δt`` stays
  the LP's energy and the Δt-weighted means of units online and of the reserve are the LP's;
* ``first`` — an *event* at the step's start (units started / shut down): the whole value stays at the first fine
  step and the other fine steps get 0, so totals and costs (Σ_t) are the LP's, an integer unit shows whole
  start-ups, and the minimum up / down time history (``params._mintime.history_hours_to_end``) has the event at
  the start of its step.

Every such reader goes through this module:

* :func:`var_values` — ``Solution.value`` of a variable on every fine step (CO2 accumulator, fix_usage handoff,
  the ramp and online roll carries, the online pin and history handoffs);
* :func:`expand_coarse` — the same for an LP-shaped long frame;
* :func:`expand_value_keys` — the HiGHS-name reader of the output parquets
  (:func:`flextool.process_outputs.read_highs_solution.extract_variable`).

A value is only ever written to a fine step that the LP does not have a column for, so an LP whose processes are
all fine (no coarse process, or a solve that kept the process fine) is returned unchanged.  The raw horizon dump
(:mod:`flextool.engine.output._horizon_dump`) stays LP-shaped: it writes the LP's columns, a coarse process's at
``b_first`` only.  Benders reads only the half-flows of cross-region arcs, which are never coarse.
"""
from __future__ import annotations

import polars as pl

#: The variables whose coarse columns live on coarse steps -> how a value reaches the other fine steps of its
#: step (module docstring): ``copy`` the value, or keep it at the ``first`` fine step (0 at the others).
COARSE_VARS: dict[str, str] = {
    "v_flow": "copy", "v_flow_back": "copy",
    "v_online_linear": "copy", "v_online_integer": "copy", "v_reserve": "copy", "v_exact_forward": "copy",
    "v_startup_linear": "first", "v_startup_integer": "first",
    "v_shutdown_linear": "first", "v_shutdown_integer": "first",
    "v_angle": "copy",
}

#: The variables of :data:`COARSE_VARS` that are not keyed by a process -> whose coarse steps they take: the
#: coarse-angle nodes (``flex_data.coarse_angle_steps``).  The others take the coarse processes' steps.
COARSE_VAR_STEPS: dict[str, str] = {"v_angle": "angle"}


def coarse_steps(flex_data, name: "str | None" = None) -> "pl.DataFrame | None":
    """``(key, d, b_first, t)`` (Utf8) of the coarse entities of variable ``name`` in ``flex_data``
    (:data:`COARSE_VAR_STEPS`; default and any other name: the coarse processes, key ``p``); ``None`` without
    any."""
    if flex_data is None:
        return None
    kind = COARSE_VAR_STEPS.get(name or "")
    if kind is None:
        steps = getattr(flex_data, "process_coarse_step", None)
        cols = ("p", "d", "b_first", "t")
    elif getattr(flex_data, "process_coarse_step", None) is None or flex_data.process_coarse_step.height == 0:
        return None
    else:
        from flextool.engine.preprocess.flex_data import coarse_angle_steps

        steps, cols = coarse_angle_steps(flex_data), ("n", "d", "b_first", "t")
    if steps is None or steps.height == 0:
        return None
    return steps.select(*(pl.col(c).cast(pl.Utf8) for c in cols))


def expand_coarse(frame: "pl.DataFrame | None", flex_data, *, mode: str = "copy", p: str = "p", d: str = "d",
                  t: str = "t", value: str = "value", name: "str | None" = None) -> "pl.DataFrame | None":
    """``frame`` (an LP-shaped frame of a :data:`COARSE_VARS` variable: key columns with the entity key ``p``,
    ``d``, ``t``, and ``value``) with each coarse row at a coarse step's first fine step written to the other fine
    steps of the step that the frame has no row for: its value (``mode="copy"``) or 0 (``mode="first"``).  The
    coarse steps are those of variable ``name`` (:func:`coarse_steps`).  Rows keep the frame's order (each added
    row right after its source row, then in solve order) and dtypes; the other columns are copied unchanged."""
    if mode not in ("copy", "first"):
        raise ValueError(f"expand_coarse: unknown mode {mode!r}")
    steps = coarse_steps(flex_data, name)
    if frame is None or steps is None or frame.height == 0:
        return frame
    keys = [c for c in frame.columns if c != value]
    others = (steps.filter(pl.col("t") != pl.col("b_first")).with_row_index("__j", offset=1)
              .rename({steps.columns[0]: "__p", "d": "__d", "b_first": "__b", "t": "__t"}))
    lead = frame.with_row_index("__i")
    copies = (lead.with_columns(pl.col(p).cast(pl.Utf8).alias("__p"), pl.col(d).cast(pl.Utf8).alias("__d"),
                                pl.col(t).cast(pl.Utf8).alias("__b"))
              .join(others, on=["__p", "__d", "__b"], how="inner")
              .with_columns(pl.col("__t").cast(frame.schema[t]).alias(t))
              .select(*frame.columns, "__i", "__j")
              .join(frame.select(keys), on=keys, how="anti"))
    if copies.height == 0:
        return frame
    if mode == "first":
        copies = copies.with_columns(pl.lit(0.0).cast(frame.schema[value]).alias(value))
    out = pl.concat([lead.with_columns(pl.lit(0, dtype=pl.UInt32).alias("__j")).select(copies.columns), copies])
    return out.sort("__i", "__j").select(frame.columns)


def var_values(sol, flex_data, name: str) -> "pl.DataFrame | None":
    """``sol.value(name)`` on every fine step (:func:`expand_coarse` by the mode of :data:`COARSE_VARS`; any other
    variable as it is); ``None`` when the solution has no such variable."""
    if sol is None or name not in getattr(sol, "_vars", {}):
        return None
    frame = sol.value(name)
    if name not in COARSE_VARS or frame is None or frame.height == 0:
        return frame
    return expand_coarse(frame, flex_data, mode=COARSE_VARS[name], p=frame.columns[0], name=name)


def coarse_step_members(steps: "pl.DataFrame | None") -> "dict[tuple[str, str, str], list[str]]":
    """``(key, d, b_first)`` -> the other fine steps of the coarse step, in solve order (:func:`coarse_steps`)."""
    out: dict[tuple[str, str, str], list[str]] = {}
    if steps is None:
        return out
    for p, d, b, t in steps.iter_rows():
        if t != b:
            out.setdefault((p, d, b), []).append(t)
    return out


def expand_value_keys(values_by_key: "dict[tuple[str, ...], float]", steps: "pl.DataFrame | None", *,
                      mode: str = "copy") -> None:
    """In place: each key ``(d, t, e, …)`` of a coarse entity at its coarse step's first fine step gets an entry
    for every other fine step of the step that has no key of its own (the HiGHS-name reader's row keys: period,
    time, then the entity column first — the process, or the node of :data:`COARSE_VAR_STEPS`): its value
    (``mode="copy"``) or 0 (``mode="first"``)."""
    if mode not in ("copy", "first"):
        raise ValueError(f"expand_value_keys: unknown mode {mode!r}")
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
                    extra[copy] = val if mode == "copy" else 0.0
    values_by_key.update(extra)
