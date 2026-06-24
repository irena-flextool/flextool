"""Benders (Option C) regional decomposition — master + multi-cut loop.

This module implements the Phase-2 Benders scheme described in
``specs/benders_option_c.md`` (design, Phase-2 revised, and the
PROCEED-WITH-CHANGES critique).  It REPLACES the dual-subgradient scheme's
false-convergence behaviour on GREENFIELD cross-region trade connections: the
Lagrangian splitter severs each cross-region arc into invest-less half-flows
bounded to ~0, collapsing to an autarkic solution with an INVALID bound ABOVE
the true optimum.  Benders puts the trade investment + the trade flow / capacity
coupling in a MASTER and feeds each region the master's chosen flow as a pinned
boundary injection, returning a VALID lower bound and the true optimum.

Architecture (per the spec's locked decisions):

* **Hand-built persistent master** — a single ``polar_high.Problem`` wrapped in a
  ``WarmProblem``, built ONCE and grown by appended optimality cut rows
  (``WarmProblem.add_cut_row``).  Structure MIRRORS the monolith's trade layer:

    - trade flow vars ``f[arc, d, t]`` for every cross-region directed arc, in the
      SAME unitsize-normalised units as the region half-flow ``v_flow`` (so cut
      slopes — reduced costs of the pinned columns — drop in with no rescale);
    - invest vars ``C[conn]`` per cross-region connection (single invest period in
      the prototype);
    - capacity coupling ``f[arc,d,t] ≤ C[conn]`` (NORMALISED — unitsize cancels,
      Phase-1 §A.5);
    - one recourse var ``η_r`` per region, lower-bounded by a large-NEGATIVE
      finite floor sized from the bootstrap region costs (NOT a hard ``η≥0`` —
      FlexTool region costs can be negative, so a blind 0 floor could cut off the
      optimum; the finite floor is a provably valid global under-estimate that
      keeps the cut-less iter-0 master kOptimal, and the f̄=0 bootstrap seeds the
      first real cuts before the first LB-bearing master solve);
    - objective ``Σ_conn C[conn]·unitsize·annu  +  Σ_r η_r`` where ``annu`` is read
      from the SAME source ``build_flextool`` uses
      (``ed_entity_annual_discounted`` + ``ed_lifetime_fixed_cost``) — NOT a
      hand-derived annuity (a mismatch silently yields a wrong-but-plausible
      optimum).

  Master is built AUTOSCALE-OFF (the test path never applies Layer 2), so the
  appended cut rows live on the built-column scale.

* **Region subproblems** are normal FlexTool models via the splitter, with the
  cross-region half-flows UNCAPPED (``benders_uncap_cross_region=True``) so a
  positive master pin is feasible.  Each iteration pins every region's forward
  cross-region half-flows to the current f̄ per-``(d,t)`` (reverse pinned to 0) and
  solves; the cut slope per ``(arc,d,t)`` is the reduced cost of the pinned
  forward column ``Solution.col_dual[pin_col_id]`` (Phase-1-verified =
  ``∂cost_r/∂f̄``, basis-correct, no monolith reference).

Loop (multi-cut Benders): bootstrap f̄=0 → first cuts → master → new f̄ → regions
→ cuts → master → … until ``gap = (best_UB − LB)/|best_UB| ≤ tol``.  ``LB`` =
master objective (a valid lower bound — the whole point vs the Lagrangian bug);
``UB`` = master invest cost(C) + Σ cost_r(f̄), incumbent = best (min) UB.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import highspy
import numpy as np
import polars as pl

from polar_high import Param, Problem, Sum, WarmProblem

from flextool.engine_polars import _region_filter
from flextool.engine_polars import build_flextool as _build_flextool
from flextool.engine_polars._axis_enums import (
    get_global_axis_enums,
    reset_global_axis_enums,
    set_global_axis_enums,
)
from flextool.engine_polars._lagrangian import _identify_coupling_cols
from flextool.engine_polars.input import FlexData

_logger = logging.getLogger(__name__)

# Slack on the "LB ≤ M" valid-bound assertion: LB is a true lower bound up to
# the LP optimality tolerance, so allow a tiny relative overshoot.
_LB_VALID_SLACK = 1e-9

# A large-NEGATIVE finite floor on each recourse var ``η_r``.  A truly free
# (lower=-inf) η leaves the cut-less master UNBOUNDED; an unbounded HiGHS solve
# then corrupts the warm basis so the first appended cut row triggers a
# kSolveError.  The floor must be a PROVABLY valid global under-estimate so it
# never cuts off the optimum (the spec's point-2 warning: FlexTool region costs
# can be negative via commodity-sell / storage-revenue, so a blind 0 is unsafe).
#
# The floor is derived per-run from the bootstrap (autarkic, f̄=0) region costs:
# ``eta_floor = -_ETA_FLOOR_MULT · max_r |cost_r^autarky|`` (see the bootstrap in
# ``_solve_benders_inner``), computed in the SAME (scaled) space the master
# objective lives in.  Validity: at the optimum ``η_r = cost_r(f̄*)``, the
# region's recourse cost at the optimal trade schedule.  ``cost_r^autarky``
# (zero import) is the no-trade reference; trade only relaxes a region's balance
# (a free injection it may ignore within finite slack), so the minimum
# achievable ``cost_r`` is bounded below by ``-(region sell-revenue capacity)``,
# a finite quantity no larger in magnitude than a small multiple of the autarkic
# cost for these models.  ``-1.1·max|cost^autarky|`` therefore sits BELOW any
# achievable ``cost_r`` (with margin) and can never cut off the optimum, while
# being ~90× tighter than the old ``-100·max|cost|``: that narrows the
# floor/cut-coef dynamic range (the over-wide range, e.g. 1e11 vs O(1e9) cut
# coefs, is what drove the Phase-3c "Optimal→Unknown" warning on the warm
# post-append re-solve).  The runtime ``LB ≤ best_UB`` sandwich guard is the
# safety net confirming the floor never produced an invalid bound.
_ETA_FLOOR_MULT = 1.1


@dataclass
class _ArcMaster:
    """Master-side bookkeeping for one cross-region directed arc."""

    key: tuple  # (p, source, sink)
    conn: str  # the connection entity == key[0]
    export_region: str
    import_region: str
    # Ordered (by d,t) dim-tuples + master flow col-ids for f[arc, d, t].
    dim_tuples: list[tuple]
    f_col_ids: np.ndarray  # master f columns, aligned to dim_tuples
    # Per-region pinned-column ids (in the REGION's v_flow var), aligned to
    # dim_tuples.  Export region pins the export half-flow; import the import.
    export_pin_cols: np.ndarray
    import_pin_cols: np.ndarray


@dataclass
class BendersResult:
    """Outcome of :func:`solve_benders`."""

    converged: bool
    iterations: int
    total_objective: float  # best (min) UB = the recovered optimum
    lower_bound: float
    upper_bound: float
    gap: float
    region_costs: dict[str, float]  # cost_r at the incumbent f̄
    # Recovered master decisions at the incumbent.
    invest: dict[str, float]  # connection -> normalised invested capacity C
    # arc-key -> polars frame (p, source, sink, d, t, value) of the trade flow.
    trade_flow: dict[tuple, pl.DataFrame] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Master construction (hand-built persistent WarmProblem).
# ---------------------------------------------------------------------------


class _BendersMaster:
    """Persistent master mirroring the monolith trade layer.

    Owns the trade flow vars, invest vars, the capacity coupling, and the
    ``η_r`` recourse vars; grows by appended cut rows.  Built ONCE;
    re-solved warm each iteration after appending cuts.

    Two construction paths, selected by ``master=``:

    * ``"flextool"`` (default, Phase 3a): the trade flow / invest / capacity
      / cost layer is generated by ``build_flextool`` over the network-only
      reduced :class:`FlexData` (:func:`_region_filter.master_network_data`)
      — the SAME emit the monolith uses, so the master picks up the
      ``maxFlow`` capacity coupling, the invest annuity, AND the connection
      flow cost natively / RP-consistently (the last is decisive at scale,
      Phase-3 §1.5 / §6).  ``v_flow[conn,d,t]`` and ``v_invest_p[conn,d]``
      are FlexTool-built; the ``η_r`` recourse vars + cut rows are appended
      via the polar-high primitives (:meth:`add_recourse_col` /
      :meth:`add_cut_row`).
    * ``"hand"`` (Phase 2): the hand-built ``f[arc,d,t]`` / ``C[conn]`` /
      normalised ``trade_capacity`` / invest-only objective.  Kept for the
      Phase-2 acceptance gate and master-vs-master comparison.

    The public surface (``solve``, ``set_eta_floor``, ``add_cut``,
    ``read_master``, ``invest_cost``, the ``_eta_col`` / ``a.f_col_ids`` /
    ``_C_cols`` id maps) is identical across both paths so
    :func:`solve_benders` is path-agnostic.
    """

    def __init__(self, data: FlexData, arcs: list[_ArcMaster],
                 regions: list[str], eta_floor: float,
                 *, master: str = "flextool", obj_scale: float = 1.0):
        self.arcs = arcs
        self.regions = list(regions)
        self._eta_floor = eta_floor
        self._master_kind = master
        # Objective scale ``s`` shared with the region subproblems.  The master
        # FlexTool objective is built ×s and each η enters at coef 1.0, so the
        # master objective lives in scaled space; the region cut slopes (region
        # objective duals) are ∂(s·currency) and drop in homogeneously.  See
        # ``_build_flextool_master`` and ``solve_benders``.
        self._obj_scale = float(obj_scale)
        # Master trade cost (invest annuity + flow cost) at the last
        # ``read_master`` — stashed for the flextool path's ``invest_cost``
        # (which reads it from the FlexTool objective rather than a hand
        # coefficient sum).
        self._last_trade_cost: float = 0.0
        if master == "flextool":
            self._build_flextool_master(data)
        elif master == "hand":
            self._build(data)
        else:
            raise ValueError(
                f"_BendersMaster: unknown master={master!r} "
                f"(expected 'flextool' or 'hand')"
            )

    # ------------------------------------------------------------------
    # FlexTool-generated master (Phase 3a).
    # ------------------------------------------------------------------

    def _build_flextool_master(self, data: FlexData) -> None:
        """Build the master trade layer via ``build_flextool`` over the
        network-only reduced FlexData, then append the ``η_r`` recourse
        columns and resolve the stable f / C / η col-id maps."""
        reduced = _region_filter.master_network_data(data, self.regions)
        m = Problem()
        # AUTOSCALE-OFF: ``build_flextool`` never applies Layer 2 (that lives
        # only in ``_orchestration``); the appended cut rows then sit on the
        # built-column scale with no pre-scaling, identical to the hand path
        # and to the region subproblems (Phase-3 §2.3).
        #
        # Objective SCALING: build the master objective ×s (the SAME ``s`` the
        # regions are built with).  The master invest coef, flow-cost, and the
        # FlexTool objective are all multiplied by s natively (model.py:4128);
        # the appended η cols (coef 1.0) and the region cut slopes (∂(s·currency))
        # then live at the same scale, so the cut row is homogeneous in s and
        # cut validity is preserved (an exact monotone re-expression).  For the
        # small fixtures s=1.0 ⇒ byte-identical to the un-scaled build.
        _build_flextool(m, reduced, scale_the_objective=self._obj_scale)

        self._m = m
        self._wp = WarmProblem(m)
        # Initial build so the warm handle exists and col ids resolve.  The
        # iter-0 master (no cuts) is bounded: η has the finite floor and
        # ``v_invest_p ≤ p_entity_max_units`` (FlexTool-emitted) bounds the
        # capacity (and hence ``v_flow`` via ``maxFlow``).
        self._wp.solve()
        self._built = True

        # Resolve f col-ids per arc by (p, source, sink, d, t) — the SAME
        # triple the region forward half-flow carries, so master f cell and
        # region pin cell map 1:1 (Phase-3 §1.4).
        for a in self.arcs:
            a.f_col_ids = np.array(
                [int(self._wp.col_id_of_var("v_flow", dt))
                 for dt in a.dim_tuples],
                dtype=np.int64,
            )
        # Resolve C col-ids per connection by (conn, d).  The reduced
        # FlexData's ``pd_invest_set`` gives one invest period per pipe in
        # the prototype; we sum the master invest value over those periods
        # in ``read_master``.
        conns = sorted({a.conn for a in self.arcs})
        invest_periods: dict[str, list[str]] = {}
        if reduced.pd_invest_set is not None:
            for r in reduced.pd_invest_set.iter_rows(named=True):
                invest_periods.setdefault(r["p"], []).append(r["d"])
        self._C_cols: dict[str, list[int]] = {}
        for c in conns:
            periods = invest_periods.get(c, [])
            if not periods:
                raise RuntimeError(
                    f"Benders flextool master: connection {c!r} has no "
                    f"pd_invest_set period — no v_invest_p column"
                )
            self._C_cols[c] = [
                int(self._wp.col_id_of_var("v_invest_p", (c, d)))
                for d in periods
            ]

        # Append one η_r recourse column per region (cost=1.0 ⇒ each η
        # enters the master objective with coef 1.0 ON TOP of FlexTool's
        # own invest + flow cost — so the master objective is
        # FlexTool-invest+flow-cost + Σ η_r, exactly the Benders master).
        self._eta_col: dict[str, int] = {}
        for r in self.regions:
            self._eta_col[r] = int(self._wp.add_recourse_col(
                f"eta_{r}", cost=1.0, lower=self._eta_floor,
            ))
        # Re-solve so the appended η columns are part of the live model
        # before the loop's first cut append.
        self._wp._h.clearSolver()
        self._wp.solve()

        self._f_var_dims = ("p", "source", "sink", "d", "t")

    def _build(self, data: FlexData) -> None:
        arcs = self.arcs
        conns = sorted({a.conn for a in arcs})

        # --- invest annuity coefficient, read from the SAME source the
        # monolith uses: obj += Sum(v_invest_p * p_unitsize * annu)
        #                       + Sum(v_invest_p * p_unitsize * lifetime_fixed)
        # (model.py ~3900-3910).  We collapse the two e-d cost params into a
        # single per-connection coefficient `C_cost = unitsize * (annu + lf)`.
        # Single invest period in the prototype; we sum over the periods the
        # connection is invest-eligible (pd_invest_set).
        unitsize = self._param_map(data.p_unitsize, ("p",), "value")
        annu = self._param_map(
            data.ed_entity_annual_discounted, ("e", "d"), "value"
        )
        lf = self._param_map(data.ed_lifetime_fixed_cost, ("e", "d"), "value")
        invest_periods: dict[str, list[str]] = {}
        if data.pd_invest_set is not None:
            for r in data.pd_invest_set.iter_rows(named=True):
                invest_periods.setdefault(r["p"], []).append(r["d"])

        # Per-connection invest upper bound = p_entity_max_units (the monolith's
        # `maxInvest_var_bound`: v_invest_p <= p_entity_max_units, NORMALISED).
        # Mirroring it bounds the master (a single steep cut would otherwise
        # leave the LP unbounded until enough cuts accumulate).
        max_units = self._param_map(data.p_entity_max_units, ("e", "d"), "value")

        self._conn_cost: dict[str, float] = {}
        self._conn_cap: dict[str, float] = {}
        for c in conns:
            us = float(unitsize.get((c,), 0.0))
            periods = invest_periods.get(c, [])
            if not periods:
                raise RuntimeError(
                    f"Benders master: connection {c!r} has no pd_invest_set "
                    f"period — cannot form its invest cost coefficient"
                )
            coef = 0.0
            cap = 0.0
            for d in periods:
                coef += us * (float(annu.get((c, d), 0.0)) + float(lf.get((c, d), 0.0)))
                cap += float(max_units.get((c, d), 0.0))
            # Scale the invest cost coef by ``s`` so the hand master objective
            # (Σ C_cost·C + Σ η, η at coef 1.0 carrying the scaled recourse cost)
            # lives in the same scaled space as the regions / the flextool path.
            self._conn_cost[c] = coef * self._obj_scale
            self._conn_cap[c] = cap

        # --- build the hand master as a polar_high.Problem -----------------
        m = Problem()

        # Invest vars C[conn] >= 0 — one row per connection (single invest col).
        c_frame = pl.DataFrame({"conn": conns})
        C = m.add_var("C", ("conn",), c_frame, lower=0.0)

        # Trade flow vars f[arc_p, arc_source, arc_sink, d, t] >= 0 over the
        # union of every arc's (d,t) grid, keyed by the SAME (p,source,sink)
        # triple the region half-flows carry so the pin values map 1:1.
        f_rows = []
        for a in arcs:
            for (p, s, k, d, t) in a.dim_tuples:
                f_rows.append({"p": p, "source": s, "sink": k, "d": d, "t": t})
        f_frame = pl.DataFrame(f_rows)
        f = m.add_var("f", ("p", "source", "sink", "d", "t"), f_frame, lower=0.0)

        # eta_r recourse vars, lower-bounded by a large-negative finite floor
        # (see _ETA_FLOOR — keeps the cut-less iter-0 master kOptimal without
        # cutting off the optimum; the f̄=0 bootstrap then seeds the real cuts).
        eta_frame = pl.DataFrame({"region": self.regions})
        eta = m.add_var("eta", ("region",), eta_frame, lower=self._eta_floor)

        # Capacity coupling  f[arc,d,t] <= C[conn]   <=>   C[conn] - f >= 0
        # (NORMALISED: unitsize cancels — both f and C are unitsize-normalised).
        # Build one row per (arc, d, t).  We attach the connection key as a
        # join column so `Where` aligns f with its connection's C.
        cap_idx = pl.DataFrame(
            [
                {"conn": a.conn, "p": p, "source": s, "sink": k, "d": d, "t": t}
                for a in arcs
                for (p, s, k, d, t) in a.dim_tuples
            ]
        )
        # C[conn] - f[arc,d,t] >= 0  (f and C broadcast onto `over` by their
        # shared dims — `conn` for C, the 5 arc dims for f).
        m.add_cstr(
            "trade_capacity",
            over=cap_idx,
            sense=">=",
            lhs_terms={"cap": C, "flow_neg": -f},
            rhs_terms={"zero": 0.0},
        )

        # Invest upper bound  C[conn] <= max_units[conn]  (mirrors the
        # monolith maxInvest_var_bound; bounds the master).
        cap_param = Param(
            ("conn",),
            pl.DataFrame(
                {"conn": conns, "value": [self._conn_cap[c] for c in conns]}
            ),
        )
        m.add_cstr(
            "maxInvest",
            over=c_frame,
            sense="<=",
            lhs_terms={"invest": C},
            rhs_terms={"max_units": cap_param},
        )

        # Objective: Σ_conn C_cost[conn]·C[conn] + Σ_r eta_r.
        cost_param = Param(
            ("conn",),
            pl.DataFrame(
                {
                    "conn": conns,
                    "value": [self._conn_cost[c] for c in conns],
                }
            ),
        )
        m.set_objective(Sum(C * cost_param) + Sum(eta))

        self._m = m
        self._wp = WarmProblem(m)
        # Initial build so the warm handle exists and col ids resolve.  The
        # iter-0 master (no cuts, η at its finite floor) is kOptimal; we never
        # read this solve's objective — the loop seeds the first cuts BEFORE the
        # first LB-bearing master solve.  But the build must succeed.
        self._wp.solve()
        self._built = True

        # Resolve stable master col-ids we reference in cut rows.
        self._eta_col: dict[str, int] = {
            r: int(self._wp.col_id_of_var("eta", (r,))) for r in self.regions
        }
        # f col-ids per arc were captured at arc construction time; re-resolve
        # them from the built master so they index the live HiGHS columns.
        for a in arcs:
            a.f_col_ids = np.array(
                [int(self._wp.col_id_of_var("f", dt)) for dt in a.dim_tuples],
                dtype=np.int64,
            )
        self._C_cols: dict[str, int] = {
            c: int(self._wp.col_id_of_var("C", (c,))) for c in conns
        }
        self._f_var_dims = ("p", "source", "sink", "d", "t")

    @staticmethod
    def _param_map(param, dims: tuple, value_col: str) -> dict[tuple, float]:
        """Build a ``{dim_tuple: value}`` lookup from a Param (or empty)."""
        if param is None:
            return {}
        fr = param.frame
        out: dict[tuple, float] = {}
        for r in fr.iter_rows(named=True):
            out[tuple(r[d] for d in dims)] = float(r[value_col])
        return out

    # -- per-iteration interface ----------------------------------------

    def set_eta_floor(self, floor: float) -> None:
        """Update every η column's lower bound on the live model.

        Skips regions whose η has already been relaxed to free (-inf) after
        contributing its first cut (see :meth:`relax_eta_after_cut`)."""
        relaxed = getattr(self, "_eta_relaxed", set())
        cols = np.array(
            [self._eta_col[r] for r in self.regions if r not in relaxed],
            dtype=np.int32,
        )
        if cols.size:
            lows = np.full(cols.size, float(floor), dtype=np.float64)
            highs = np.full(cols.size, highspy.kHighsInf, dtype=np.float64)
            self._wp._h.changeColsBounds(int(cols.size), cols, lows, highs)
        self._eta_floor = float(floor)

    def relax_eta_after_cut(self, region: str) -> None:
        """Relax ``η_region`` to free (lower=-inf) once it has at least one
        cut.  The cut(s) now bound ``η_region`` from below, so the finite
        bootstrap floor is no longer needed and removing it both tightens the
        master (no spurious floor-active corner) and narrows the bound dynamic
        range that drives the warm-resolve kUnknown."""
        relaxed = getattr(self, "_eta_relaxed", None)
        if relaxed is None:
            relaxed = self._eta_relaxed = set()
        if region in relaxed:
            return
        col = np.array([self._eta_col[region]], dtype=np.int32)
        lows = np.array([-highspy.kHighsInf], dtype=np.float64)
        highs = np.array([highspy.kHighsInf], dtype=np.float64)
        self._wp._h.changeColsBounds(1, col, lows, highs)
        relaxed.add(region)

    def solve(self):
        # Warm-restart: the master objective is scaled (scale_the_objective),
        # so appending a cut row and re-solving WARM stays kOptimal — no need
        # to throw away the basis every iteration.  WarmProblem.solve runs warm
        # first and only falls back to a cold clearSolver()+re-run if the warm
        # path fails to certify kOptimal (the proven cold fallback).
        sol = self._wp.solve(retry_on_unknown=True)
        if not sol.optimal:
            status = self._wp._h.getModelStatus()
            raise RuntimeError(
                f"Benders master solve not optimal: {status} "
                f"(ncol={self._wp._h.getNumCol()} nrow={self._wp._h.getNumRow()})"
            )
        return sol

    def add_cut(self, region: str, f_bar: dict[int, float], cost_r: float,
                slopes: dict[int, float]) -> int:
        """Append the optimality cut for ``region``::

            eta_r  -  Σ_cell slope[cell]·f[cell]   >=   cost_r - Σ slope·f̄

        ``f_bar`` and ``slopes`` are keyed by MASTER f col-id.  Returns the
        appended row id.
        """
        eta_col = self._eta_col[region]
        col_ids: list[int] = [eta_col]
        coefs: list[float] = [1.0]
        rhs = cost_r
        for fcol, slope in slopes.items():
            if slope == 0.0:
                continue
            col_ids.append(int(fcol))
            coefs.append(-float(slope))
            rhs -= slope * f_bar[fcol]
        return self._wp.add_cut_row(col_ids, coefs, float(rhs))

    def read_master(self, sol) -> tuple[dict[str, dict[int, float]],
                                        dict[str, float], dict[str, float]]:
        """Return (f̄ per region-arc-cell, C per connection, eta per region)
        from a master solution.  f̄ is returned BOTH per arc and flattened by
        master col-id for cut bookkeeping."""
        f_by_col: dict[int, float] = {}
        for a in self.arcs:
            vals = sol.col_value[a.f_col_ids]
            for cid, v in zip(a.f_col_ids, vals):
                f_by_col[int(cid)] = float(v)
        # ``_C_cols`` carries a single col id per connection on the hand
        # path and a LIST (one per invest period) on the flextool path;
        # sum over the period columns either way.
        C_by_conn: dict[str, float] = {}
        for c, col in self._C_cols.items():
            cols = col if isinstance(col, (list, tuple, np.ndarray)) else [col]
            C_by_conn[c] = float(sum(sol.col_value[int(ci)] for ci in cols))
        eta_by_region = {r: float(sol.col_value[col]) for r, col in self._eta_col.items()}
        # Stash the master TRADE cost (invest annuity + flow cost) at this
        # solution.  On the flextool path it is FlexTool's own objective
        # MINUS Σ η_r (each η enters obj with coef 1.0); on the hand path
        # it is the invest-only coefficient sum (computed in invest_cost).
        if self._master_kind == "flextool":
            eta_sum = sum(eta_by_region.values())
            self._last_trade_cost = float(sol.obj) - eta_sum
        return f_by_col, C_by_conn, eta_by_region

    def invest_cost(self, C_by_conn: dict[str, float]) -> float:
        """Master trade cost (invest annuity + flow cost) at the last
        ``read_master``.

        On the flextool path this is read from FlexTool's own objective
        (``sol.obj − Σ η_r``), so it INCLUDES the connection flow cost the
        hand master omits (Phase-3 §2.4 ``master_trade_cost``).  On the hand
        path it is the invest-only hand coefficient sum (the prototype's
        pipe flow cost is 0, so both agree)."""
        if self._master_kind == "flextool":
            return self._last_trade_cost
        return sum(self._conn_cost[c] * C_by_conn[c] for c in C_by_conn)


# ---------------------------------------------------------------------------
# Region subproblem assembly.
# ---------------------------------------------------------------------------


@dataclass
class _Region:
    name: str
    wp: WarmProblem
    # forward arcs this region touches, with (region pin col-ids, master f
    # col-ids) aligned by (d,t).
    forward: list[tuple[_ArcMaster, np.ndarray, np.ndarray]]  # (arc, region_cols, master_cols)
    # reverse half-flow region col-ids to pin to 0.
    reverse_cols: np.ndarray


def _build_arcs(splits, warm) -> list[_ArcMaster]:
    """Discover the cross-region directed arcs + per-region pin columns."""
    couplings = _identify_coupling_cols(splits, warm)
    region_idx = {s.region: i for i, s in enumerate(splits)}
    arcs: list[_ArcMaster] = []
    for cpl in couplings:
        # Recover the (d,t) dim-tuples in the export region's column order.
        vf_e = warm[region_idx[cpl.export_region]]._p._vars["v_flow"]
        ehf_rows = vf_e.frame.filter(
            pl.col("col_id").is_in(cpl.export_cols)
        ).sort("d", "t")
        # Master arc dims use the ORIGINAL (p, source, sink) triple.
        p, s, k = cpl.pipeline_key
        dim_tuples = [
            (p, s, k, r["d"], r["t"]) for r in ehf_rows.iter_rows(named=True)
        ]
        arcs.append(
            _ArcMaster(
                key=cpl.pipeline_key,
                conn=p,
                export_region=cpl.export_region,
                import_region=cpl.import_region,
                dim_tuples=dim_tuples,
                f_col_ids=np.zeros(len(dim_tuples), dtype=np.int64),  # filled by master
                export_pin_cols=cpl.export_cols.astype(np.int64),
                import_pin_cols=cpl.import_cols.astype(np.int64),
            )
        )
    return arcs


def _reverse_cols(split, warm: WarmProblem) -> np.ndarray:
    """All half-flow v_flow columns in ``split`` whose virtual arc is a
    REVERSE cross-region direction (so we pin them to 0)."""
    vf = warm._p._vars["v_flow"]
    # A reverse half-flow is one whose (original_source, original_sink) is the
    # reverse of a forward coupling; but simpler: pin EVERY half-flow that is
    # not a forward-pinned one.  We compute forward virtual cols separately, so
    # here gather all half-flow virtual cols and let the caller subtract.
    cols = []
    for hf in split.half_flows:
        sub = vf.frame.filter(
            (pl.col("p") == hf.virtual_p)
            & (pl.col("source") == hf.virtual_arc_source)
            & (pl.col("sink") == hf.virtual_arc_sink)
        )
        cols.append(sub["col_id"].to_numpy().astype(np.int64))
    return np.concatenate(cols) if cols else np.zeros(0, dtype=np.int64)


# ---------------------------------------------------------------------------
# Driver.
# ---------------------------------------------------------------------------


def solve_benders(
    data: FlexData,
    regions: list[str],
    *,
    max_iters: int = 20,
    tol: float = 1e-4,
    monolith_objective: float | None = None,
    build_problem=None,
    master: str = "flextool",
    scale_the_objective: float = 1.0,
) -> BendersResult:
    """Run the multi-cut Benders loop on the regional decomposition of
    ``data``.

    Parameters
    ----------
    data
        The undecomposed :class:`FlexData`.
    regions
        Region names (the group entities the splitter partitions on).
    max_iters
        Iteration cap.
    tol
        Relative gap tolerance ``(best_UB − LB)/|best_UB|`` for convergence.
    monolith_objective
        If given (not ``None``), the loop asserts ``LB ≤ M·(1+1e-9)`` every
        iteration (an OPTIONAL test-time valid-lower-bound check against a
        known monolith optimum).  Pass ``None`` (the at-scale driver does) to
        skip it when no trustworthy/up-to-date M is available — the ALWAYS-ON
        ``LB ≤ best_UB`` sandwich guard is the monolith-free invalid-bound
        detector and needs no M.  ``M`` is interpreted in REAL units (same as
        the returned ``total_objective``/bounds), regardless of
        ``scale_the_objective``.
    build_problem
        Region builder.  Defaults to :func:`build_flextool` with the loop's
        ``scale_the_objective`` applied (so the regions build at the SAME
        scale as the master).  A custom builder is used as-is.
    master
        Master construction path: ``"flextool"`` (default, Phase 3a) builds
        the trade layer via ``build_flextool`` over the network-only reduced
        FlexData; ``"hand"`` (Phase 2) hand-builds it.  See
        :class:`_BendersMaster`.
    scale_the_objective
        Objective scale ``s`` applied to BOTH the master and the region
        subproblems (the solve's ``scale_the_objective``; ``s=1e-6`` for the
        real H2_trade solve, ``s=1.0`` for the small fixtures).  Because the
        cut coefficients are region OBJECTIVE duals, scaling both objectives by
        the same ``s`` scales the cuts consistently — an exact monotone
        re-expression that preserves cut validity and conditions the master LP
        (curing the at-scale warm-resolve kUnknown).  The loop's internal LB/UB
        arithmetic runs in scaled space; the returned ``total_objective``,
        ``lower_bound`` and ``upper_bound`` are UNSCALED back to real units
        (``÷s``).  ``s=1.0`` ⇒ byte-identical to the un-scaled path.

    Returns
    -------
    BendersResult
    """
    if build_problem is None:
        def build_problem(pb, d):
            _build_flextool(pb, d, scale_the_objective=scale_the_objective)

    _data_enums = getattr(data, "_axis_enums", None)
    _enums_token = None
    if _data_enums is not None and _data_enums != get_global_axis_enums():
        _enums_token = set_global_axis_enums(_data_enums)
    try:
        return _solve_benders_inner(
            data, regions, max_iters=max_iters, tol=tol,
            monolith_objective=monolith_objective, build_problem=build_problem,
            master=master, obj_scale=scale_the_objective,
        )
    finally:
        if _enums_token is not None:
            reset_global_axis_enums(_enums_token)


def _solve_benders_inner(data, regions, *, max_iters, tol, monolith_objective,
                         build_problem, master="flextool",
                         obj_scale: float = 1.0) -> BendersResult:
    # --- split with the cross-region half-flows UNCAPPED so the master pin is
    # feasible (Phase-2 splitter Benders mode).
    splits = _region_filter.split(
        data, regions=regions, benders_uncap_cross_region=True
    )
    subproblems = [Problem() for _ in splits]
    for s, pb in zip(splits, subproblems):
        build_problem(pb, s.data)
    warm = [WarmProblem(p) for p in subproblems]
    # Initial build of every region (fix_cols / col_dual need a built model).
    for w in warm:
        w.solve()

    arcs = _build_arcs(splits, warm)
    if not arcs:
        raise RuntimeError(
            "Benders: no cross-region coupling arcs found — nothing to "
            "decompose"
        )

    # Boundary-node penalty finiteness precondition (optimality-cuts-only).
    _assert_finite_boundary_penalties(data, arcs)

    region_idx = {s.region: i for i, s in enumerate(splits)}

    # NOTE: ``_build_arcs`` filled each arc's ``f_col_ids`` with zeros; the
    # master rewrites them to its live HiGHS f columns at build time.  The
    # region-meta below references the REGION pin columns (export/import
    # half-flows), which are already resolved, so it can be built before the
    # master; we wire the master f col-ids into the meta after the master build.
    regions_meta: list[_Region] = []
    for s in splits:
        w = warm[region_idx[s.region]]
        forward: list[tuple[_ArcMaster, np.ndarray, np.ndarray]] = []
        fwd_region_cols: set[int] = set()
        for a in arcs:
            if s.region == a.export_region:
                region_cols = a.export_pin_cols
            elif s.region == a.import_region:
                region_cols = a.import_pin_cols
            else:
                continue
            forward.append((a, region_cols, a.f_col_ids))  # f_col_ids: see below
            fwd_region_cols.update(int(c) for c in region_cols)
        all_hf_cols = _reverse_cols(s, w)
        reverse_cols = np.array(
            [int(c) for c in all_hf_cols if int(c) not in fwd_region_cols],
            dtype=np.int64,
        )
        regions_meta.append(
            _Region(name=s.region, wp=w, forward=forward, reverse_cols=reverse_cols)
        )

    # Pre-resolve dim-tuples for fix_cols (by region var frame, aligned to the
    # region pin col order).
    pin_dim_cache: dict[int, list[tuple]] = {}

    def _region_dim_tuples(w: WarmProblem, col_ids: np.ndarray) -> list[tuple]:
        vf = w._p._vars["v_flow"]
        fr = vf.frame.filter(pl.col("col_id").is_in(col_ids))
        # Order by the given col_ids.
        order = {int(c): i for i, c in enumerate(col_ids)}
        fr = fr.with_columns(
            pl.col("col_id").replace_strict(order, default=-1).alias("__o")
        ).sort("__o")
        return [tuple(r) for r in fr.select(*vf.dims).iter_rows()]

    def _pin_and_solve(rm: _Region, f_bar_local: dict[int, float]):
        """Pin region ``rm``'s forward half-flows to ``f_bar_local`` (reverse to
        0) and solve; return (cost_r, {master_f_col: slope})."""
        w = rm.wp
        for a, region_cols, master_cols in rm.forward:
            vals = np.array(
                [f_bar_local[int(mc)] for mc in master_cols], dtype=np.float64
            )
            dt = pin_dim_cache.setdefault(
                id(region_cols), _region_dim_tuples(w, region_cols)
            )
            w.fix_cols("v_flow", dt, vals)
        if rm.reverse_cols.size:
            dt_rev = pin_dim_cache.setdefault(
                id(rm.reverse_cols), _region_dim_tuples(w, rm.reverse_cols)
            )
            w.fix_cols(
                "v_flow", dt_rev, np.zeros(rm.reverse_cols.size, dtype=np.float64)
            )
        sol_r = w.solve()
        if not sol_r.optimal:
            raise RuntimeError(
                f"Benders region {rm.name!r} subproblem not optimal"
            )
        slopes: dict[int, float] = {}
        for a, region_cols, master_cols in rm.forward:
            rc = sol_r.col_dual[region_cols]
            for mc, g in zip(master_cols, rc):
                slopes[int(mc)] = slopes.get(int(mc), 0.0) + float(g)
        return float(sol_r.obj), slopes

    # --- Build the master with a PROVISIONAL η floor (refined after the
    # bootstrap once we know the real cost scale).  The master assigns the live
    # f column ids (rewriting each arc's ``f_col_ids``), which the cut keying
    # below depends on, so it must be built BEFORE the bootstrap region solve.
    master_kind = master
    # Provisional floor in SCALED space (×obj_scale): just keeps the cut-less
    # initial build bounded; replaced post-bootstrap by the tight
    # -1.1·max|cost^autarky| floor below.  No cuts exist yet, so no warm-append
    # range concern at build time.
    master = _BendersMaster(
        data, arcs, [s.region for s in splits],
        eta_floor=-_ETA_FLOOR_MULT * 1e9 * obj_scale,
        master=master_kind, obj_scale=obj_scale,
    )
    # Re-bind the region-meta forward tuples to the master-rewritten f col-ids.
    for rm in regions_meta:
        rm.forward = [
            (a, region_cols, a.f_col_ids) for (a, region_cols, _old) in rm.forward
        ]

    # --- f̄ state, keyed by the LIVE master f col-id.  Bootstrap f̄ = 0.
    f_bar: dict[int, float] = {int(cid): 0.0 for a in arcs for cid in a.f_col_ids}

    # --- BOOTSTRAP: solve regions autarkic (f̄=0) to (a) generate the first
    # cuts and (b) size the TIGHT η floor = -1.1·max_r|cost_r^autarky|.  Because
    # the regions are built at the same scale ``s``, ``cost_r`` is already in
    # scaled space, so the floor is automatically in the master's scaled space —
    # a provably valid global under-estimate (autarkic cost is the no-trade
    # |cost| extreme; trade only relaxes a region's balance, so the minimum
    # achievable cost_r is below it with margin) that is ~90× tighter than the
    # old -100·max|cost|, narrowing the floor/cut-coef dynamic range that drives
    # the warm post-append kUnknown.
    bootstrap_cuts: list[tuple[str, float, dict[int, float]]] = []
    for rm in regions_meta:
        cost_r, slopes = _pin_and_solve(rm, f_bar)
        bootstrap_cuts.append((rm.name, cost_r, slopes))
    cost_scale = max((abs(c) for _, c, _ in bootstrap_cuts), default=1.0)
    # Floor in scaled space; ``max(cost_scale, obj_scale)`` keeps it from
    # collapsing to ~0 in the degenerate all-zero-cost case (the unscaled guard
    # was ``max(cost_scale, 1.0)``; ×s keeps that 1-unit guard at scale).
    eta_floor = -_ETA_FLOOR_MULT * max(cost_scale, obj_scale)
    master.set_eta_floor(eta_floor)

    best_UB = float("inf")
    best_incumbent: dict | None = None
    LB = float("-inf")
    prev_LB = float("-inf")
    iterations = 0
    converged = False
    gap = float("inf")

    # ``pending_cuts`` are the cuts for the regions solved at the CURRENT f̄;
    # they are appended at the top of each iteration before the master solve.
    # Iter 0 uses the bootstrap cuts (regions at f̄=0).
    pending_cuts = bootstrap_cuts
    C_by_conn: dict[str, float] = {}

    for it in range(max_iters):
        iterations = it + 1

        # --- append the pending cuts and (warm) re-solve the master.  Each
        # region that contributes a cut has its η relaxed to free (-inf): the
        # cut now bounds η from below, so the bootstrap floor is no longer
        # needed and dropping it tightens the master + narrows the bound range.
        for region, cost_r, slopes in pending_cuts:
            master.add_cut(region, f_bar, cost_r, slopes)
            master.relax_eta_after_cut(region)
        msol = master.solve()
        prev_LB = LB
        LB = float(msol.obj)  # scaled space
        # LB monotone non-decreasing self-check (allow tiny numerical slack).
        if it > 0 and LB < prev_LB - 1e-6 * max(1.0, abs(prev_LB)):
            raise RuntimeError(
                f"Benders LB decreased {prev_LB:.10e} -> {LB:.10e} at iter "
                f"{iterations} — stale basis / wrong cut append"
            )
        # OPTIONAL test-time guard: LB ≤ M (M supplied in REAL units → compare
        # in scaled space against M·s).  Skipped when monolith_objective is None
        # (the at-scale driver passes None — no trustworthy/up-to-date M).
        if monolith_objective is not None:
            M_scaled = monolith_objective * obj_scale
            if LB > M_scaled * (1 + _LB_VALID_SLACK):
                raise RuntimeError(
                    f"Benders LB {LB / obj_scale:.10e} exceeds monolith M "
                    f"{monolith_objective:.10e} at iter {iterations} — "
                    f"INVALID lower bound (the bug this scheme fixes)"
                )

        new_f_bar, C_by_conn, eta_by_region = master.read_master(msol)
        _check_cuts_satisfied(pending_cuts, f_bar, new_f_bar, eta_by_region)
        # The master's chosen capacity must support its chosen flow (the
        # capacity coupling f ≤ C holds at the master optimum) — a cheap
        # feasibility self-check that the UB below is a valid primal point.
        for a in arcs:
            cap = C_by_conn.get(a.conn, 0.0)
            for cid in a.f_col_ids:
                if new_f_bar[int(cid)] > cap + 1e-6 * max(1.0, abs(cap)):
                    raise RuntimeError(
                        f"Benders master infeasible coupling: f={new_f_bar[int(cid)]} "
                        f"> C[{a.conn}]={cap}"
                    )

        # --- advance f̄ to the master optimum and solve the regions there to
        # (a) get this iteration's recourse cost (→ a VALID UB, since C ≥ f̄ at
        # the master optimum) and (b) produce the next iteration's cuts.
        f_bar = new_f_bar
        region_costs: dict[str, float] = {}
        next_cuts: list[tuple[str, float, dict[int, float]]] = []
        for rm in regions_meta:
            cost_r, slopes = _pin_and_solve(rm, f_bar)
            region_costs[rm.name] = cost_r
            next_cuts.append((rm.name, cost_r, slopes))

        # --- UB = master invest cost(C) + Σ cost_r(f̄) at the SAME (f̄, C).
        # All terms are in scaled space (region cost_r and master invest_cost
        # are both ×s), so UB and LB are directly comparable in scaled space.
        UB = master.invest_cost(C_by_conn) + sum(region_costs.values())
        if UB < best_UB:
            best_UB = UB
            best_incumbent = {
                "C": dict(C_by_conn),
                "f_bar": dict(f_bar),
                "region_costs": dict(region_costs),
            }

        # ALWAYS-ON monolith-free sandwich guard: LB ≤ optimum ≤ best_UB must
        # hold (best_UB is itself an upper bound on the optimum).  An
        # LB-exceeds-best_UB is the genuine invalid-lower-bound pathology (the
        # Lagrangian-style bug this scheme fixes) and needs no external M.
        if LB > best_UB * (1 + _LB_VALID_SLACK) + _LB_VALID_SLACK * max(1.0, abs(best_UB)):
            raise RuntimeError(
                f"Benders LB {LB / obj_scale:.10e} exceeds best UB "
                f"{best_UB / obj_scale:.10e} at iter {iterations} — "
                f"INVALID lower bound (the bug this scheme fixes)"
            )

        gap = (best_UB - LB) / max(1.0, abs(best_UB))
        _logger.info(
            "Benders iter %d: LB=%.6e UB=%.6e bestUB=%.6e gap=%.3e",
            iterations, LB, UB, best_UB, gap,
        )

        if gap <= tol:
            converged = True
            break

        pending_cuts = next_cuts

    # --- assemble result from the incumbent.  UNSCALE cost-valued outputs back
    # to real units (÷s): the loop's internal LB/UB/cost arithmetic ran in
    # scaled space (objectives built ×s), but callers/tests expect real-unit
    # costs.  ``invest`` (capacity C, MW) and ``trade_flow`` (MW) are NOT costs
    # and stay in their native (scale-invariant) units.  s=1.0 ⇒ no-op.
    inc = best_incumbent if best_incumbent is not None else {
        "C": C_by_conn, "f_bar": f_bar, "region_costs": region_costs,
    }
    trade_flow = _flow_frames(arcs, inc["f_bar"])
    inv_s = 1.0 / obj_scale
    return BendersResult(
        converged=converged,
        iterations=iterations,
        total_objective=best_UB * inv_s,
        lower_bound=LB * inv_s,
        upper_bound=best_UB * inv_s,
        gap=gap,
        region_costs={r: c * inv_s for r, c in inc["region_costs"].items()},
        invest=inc["C"],
        trade_flow=trade_flow,
    )


def _flow_frames(arcs: list[_ArcMaster], f_bar: dict[int, float]) -> dict[tuple, pl.DataFrame]:
    out: dict[tuple, pl.DataFrame] = {}
    for a in arcs:
        rows = []
        for dt, cid in zip(a.dim_tuples, a.f_col_ids):
            p, s, k, d, t = dt
            rows.append(
                {"p": p, "source": s, "sink": k, "d": d, "t": t,
                 "value": f_bar[int(cid)]}
            )
        out[a.key] = pl.DataFrame(rows)
    return out


def _check_cuts_satisfied(cuts, f_bar, new_f_bar, eta_by_region) -> None:
    """Mandatory self-check (critique Point 1): at the NEW master point each
    just-appended cut must be SATISFIED, i.e.

        eta_r  >=  cost_r(f̄) + Σ_cell slope[cell]·(f_master[cell] − f̄[cell])

    ``cuts`` carry ``(region, cost_r, {master_f_col: slope})`` evaluated at the
    OLD ``f_bar``; ``new_f_bar`` is the master's chosen flow.  We assert each
    eta_r is finite AND clears its own cut RHS (within a small relative
    tolerance) — a binding/active cut makes this an equality, a slack cut an
    inequality; either way a VIOLATION means a stale basis or a wrong append."""
    for region, cost_r, slopes in cuts:
        er = eta_by_region.get(region)
        if er is None or not np.isfinite(er):
            raise RuntimeError(
                f"Benders: recourse eta[{region!r}] not finite after master "
                f"solve ({er!r})"
            )
        rhs = cost_r + sum(
            g * (new_f_bar[c] - f_bar[c]) for c, g in slopes.items()
        )
        # eta_r >= rhs (up to LP optimality tolerance, scaled to the magnitude).
        tol_abs = 1e-5 * max(1.0, abs(rhs), abs(er))
        if er < rhs - tol_abs:
            raise RuntimeError(
                f"Benders cut for {region!r} VIOLATED at the new master point: "
                f"eta={er:.10e} < cut RHS={rhs:.10e} (cost_r={cost_r:.6e}) — "
                f"stale basis / wrong cut append"
            )


def _assert_finite_boundary_penalties(data: FlexData, arcs: list[_ArcMaster]) -> None:
    """Optimality-cuts-only feasibility precondition: every boundary node
    (source/sink of a cross-region arc) must carry FINITE up/down slack
    penalties, so the recourse is always feasible."""
    boundary_nodes = set()
    for a in arcs:
        _, s, k = a.key
        boundary_nodes.add(s)
        boundary_nodes.add(k)
    for pname in ("p_penalty_up", "p_penalty_down"):
        param = getattr(data, pname, None)
        if param is None:
            continue
        fr = param.frame
        if "n" not in fr.columns or "value" not in fr.columns:
            continue
        sub = fr.filter(pl.col("n").is_in(list(boundary_nodes)))
        if sub.height == 0:
            continue
        vals = sub["value"].to_numpy()
        if not np.all(np.isfinite(vals)):
            bad = sub.filter(~pl.col("value").is_finite())
            raise RuntimeError(
                f"Benders: non-finite {pname} on a boundary node — "
                f"optimality-cuts-only feasibility precondition violated:\n{bad}"
            )
