"""The monolith that a Benders decomposition solves, for tests that compare the two.

A Benders split keeps the coarse step only for the coarse processes inside one region; a process with an arc
that crosses regions keeps the solve's step in every region, and the master keeps every flow at the solve's step
(user docs, ``new_stepduration``).  The undecomposed model makes a process coarse whenever all its balance nodes
are coarse, so on a model with a coarse cross-region connection the two are different LPs: same optimum when
the parameters are constant within each coarse step, but different per-step flows and flow sums.
:func:`benders_reference_data` gives the monolith with exactly the decomposition's coarse processes.
"""
from __future__ import annotations

import copy

import polars as pl

from flextool.engine.input._axis_enums import (
    get_global_axis_enums,
    reset_global_axis_enums,
    set_global_axis_enums,
)
from flextool.engine.solve import _region_filter


def benders_reference_data(data, regions: list[str]):
    """A shallow copy of ``data`` whose coarse processes are those of the Benders regions over ``regions``.

    ``data`` itself is not changed (it may be a shared, session-cached FlexData)."""
    enums = getattr(data, "_axis_enums", None)
    token = set_global_axis_enums(enums) if enums is not None and enums != get_global_axis_enums() else None
    try:
        splits = _region_filter.split(data, regions=regions)
    finally:
        if token is not None:
            reset_global_axis_enums(token)
    region_coarse: set[str] = set()
    for s in splits:
        frame = getattr(s.data, "processCoarse", None)
        if frame is not None:
            region_coarse |= {str(p) for p in frame["p"].to_list()}
    ref = copy.copy(data)
    for field in ("processCoarse", "process_coarse_step"):
        frame = getattr(data, field, None)
        if frame is not None:
            setattr(ref, field, frame.filter(pl.col("p").cast(pl.Utf8).is_in(sorted(region_coarse))))
    return ref
