"""Capacity in service: the investments and retirements counting in each period, built once per solve.

The capacity of a process ``p`` or a storage node ``n`` in period ``d`` is a linear expression over the existing
``v_invest`` / ``v_divest`` columns (no variable, row or parameter of its own), in units of ``unitsize``
(``state_unitsize`` for a node)::

    cap[e,d]      = existing[e,d] + invested[e,d] − retired[e,d]
    invested[e,d] = Σ_{d′ : (e,d′,d) ∈ alive}    v_invest[e,d′]
    retired[e,d]  = Σ_{d′ : (e,d′,d) ∈ retired}  v_divest[e,d′]
    prebuilt[e,d] = Σ_{d′ : (e,d′,d) ∈ prebuilt} v_invest[e,d′]

    alive    = {(e,d′,d) : d in service of the investment cohort (e,d′)}: y_on ≤ y[d] < y_eol, along the scenario
               lineage of a fan (``cohort_alive``)
    prebuilt = {(e,d′,d) ∈ alive : y[d] > y_on}: in service from the start of d, not starting to operate in d
    retired  = {(e,d′,d) : (e,d′) ∈ divest options, (d′,d) ∈ dd_forward (y[d′] ≤ y[d]), d in service of the
               existing cohort of e (y[d] < y_eol)}

``y_on`` / ``y_eol`` are the start of operation and the end of life of a cohort (``preprocess/params/_cohorts.py``,
the only lifetime-gate code).  The sets come from :func:`._cohorts.in_service`; the constants ``existing`` are the
solve's existing capacity Params (``existing_count_p`` units of a process, ``existing_state_n`` MWh of a storage
node, ``existing_mw_e`` MW / MWh of an entity, ``existing_cohort_size_e`` the pre-existing capacity that can be
retired).  Each model site keeps its own domain (members, coefficient, joins) and reads the frames and sums here.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import polars as pl
from polar_high import Param, Sum, Where

from flextool.engine.model._common import _entity_rows, _var_view
from flextool.engine.preprocess.params._cohorts import InService, in_service

if TYPE_CHECKING:
    from flextool.engine.model.core import _Build


@dataclass
class CapacityInService:
    """The capacity-in-service frames and sums of a solve (module docstring)."""

    frames: InService
    #: ``invested[p,d]`` / ``retired[p,d]`` of every process with a ``v_invest_p`` / ``v_divest_p`` column, dims
    #: ``(p, d)``; ``None`` without the variable.  Read by maxFlow, maxOutputSum, the ramp limits, the
    #: piecewise conversion, the profiles and the connection reserves.
    invested_p: Any = None
    retired_p: Any = None
    existing_count_p: Param | None = None
    existing_state_n: Param | None = None
    existing_mw_e: Param | None = None
    existing_cohort_size_e: Param | None = None

    @property
    def alive(self) -> pl.DataFrame:
        """``[e, d_invest, d]``: the investments in service in ``d``."""
        return self.frames.alive

    @property
    def prebuilt(self) -> pl.DataFrame:
        """``[e, d_invest, d]``: the investments in service from the start of ``d``."""
        return self.frames.prebuilt

    def retired_frame(self, axis: str) -> pl.DataFrame:
        """``[p, d_divest, d]`` (``axis="p"``) / ``[n, d_divest, d]`` (``axis="n"``): the retirements of existing
        capacity counting in ``d``."""
        return self.frames.retired(axis)


def build_capacity(b: "_Build") -> CapacityInService:
    """The :class:`CapacityInService` of ``b``: the frames of the solve's cohorts and the cached process sums."""
    d = b.d
    cap = CapacityInService(
        frames=in_service(d),
        existing_count_p=getattr(d, "p_process_existing_count", None),
        existing_state_n=getattr(d, "p_state_existing_capacity", None),
        existing_mw_e=getattr(d, "p_entity_all_existing", None),
        existing_cohort_size_e=getattr(d, "p_entity_pre_existing", None))
    if not b.has_proc:
        return cap
    if b.has_divest_p:
        divested = _var_view(b.v_divest_p, "__at_divest", {"d": "d_divest"}, to_axis=True)
        cap.retired_p = Sum(Where(divested, cap.retired_frame("p")), over=("d_divest",))
    if b.has_invest_p:
        invested = _var_view(b.v_invest_p, "__at_invest", {"d": "d_invest"}, to_axis=True)
        alive = _entity_rows(cap.alive, "e", d.process_source_sink, "p")
        cap.invested_p = Sum(Where(invested, alive), over=("d_invest",))
    return cap


def node_in_service(b: "_Build", nodes: pl.DataFrame, tag: str, coef: "Param | None" = None) -> tuple:
    """``(invested[n,d], retired[n,d])`` of the storage nodes ``nodes``, dims (n, d), each multiplied by ``coef``
    inside the sum; each ``None`` without the variable or without a period in service.  ``tag`` keeps the
    variable-view names of different callers apart."""
    invested = retired = None
    if b.has_invest_n:
        view = _var_view(b.v_invest_n, "__at_invest" + tag, {"d": "d_invest"}, to_axis=True)
        alive = _entity_rows(b.cap.alive, "e", nodes, "n")
        if alive.height > 0:
            invested = Sum(Where(view if coef is None else view * coef, alive), over=("d_invest",))
    if b.has_divest_n:
        view = _var_view(b.v_divest_n, "__at_divest" + tag, {"d": "d_divest"}, to_axis=True)
        active = _entity_rows(b.cap.retired_frame("n"), "n", nodes, "n")
        if active.height > 0:
            retired = Sum(Where(view if coef is None else view * coef, active), over=("d_divest",))
    return invested, retired


def node_capacity_change_terms(b: "_Build", nodes: pl.DataFrame, tag: str) -> dict:
    """``{"divest": retired[n,d], "invest_neg": −invested[n,d]}`` of the storage nodes ``nodes`` (the terms present)."""
    invested, retired = node_in_service(b, nodes, tag)
    terms: dict = {}
    if retired is not None:
        terms["divest"] = retired
    if invested is not None:
        terms["invest_neg"] = -invested
    return terms


__all__ = ["CapacityInService", "build_capacity", "node_capacity_change_terms", "node_in_service"]
