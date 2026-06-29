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
import os
from dataclasses import dataclass, field
from typing import Callable

import highspy
import numpy as np
import polars as pl

from polar_high import (
    Param,
    Problem,
    Sum,
    WarmProblem,
    resolve_worker_count,
    solve_indexed_parallel,
)

from flextool.engine_polars import _region_filter
from flextool.engine_polars import build_flextool as _build_flextool
from flextool.engine_polars._axis_enums import (
    get_global_axis_enums,
    reset_global_axis_enums,
    set_global_axis_enums,
)
from flextool.engine_polars._region_filter import HalfFlow, RegionSplit
from flextool.engine_polars.input import FlexData

_logger = logging.getLogger(__name__)


def _benders_quiet() -> bool:
    """Whether the per-solve HiGHS native log should be silenced.

    Benders solves a master plus N region subproblems (cold build + a warm
    re-solve per iteration, regions in parallel); leaving HiGHS verbose
    floods the console with many interleaved native logs.  By DEFAULT we
    mute every HiGHS solve and let the orchestrator's per-iteration
    LB/UB/gap log be the visible progress.  Set ``FLEXTOOL_BENDERS_VERBOSE``
    to restore the full native HiGHS output (the pre-silencing behaviour).
    """
    return not os.environ.get("FLEXTOOL_BENDERS_VERBOSE")


def _silence_if_quiet(wp: WarmProblem) -> None:
    """Mute the HiGHS native log on ``wp`` when Benders runs quietly.

    Output-flag persists on the WarmProblem's HiGHS handle across all of
    its subsequent cold / warm / retry solves, so a single call right after
    construction covers the whole lifetime.  Silencing changes no solve
    numerics — results are byte-identical with and without it.
    """
    if _benders_quiet():
        wp.set_output_flag(False)

# The four investment/divestment decision variables assembled into a
# whole-system handoff for the TIER-1 invest->dispatch chain.  Each is
# declared in ``model.py`` over a 2-tuple of dims whose FIRST element is
# the entity axis ("p" for process / connection vars, "n" for node vars)
# and whose second is the period axis "d".  ``Var.dims`` is read at
# runtime to recover the exact entity column name — we do not hard-code it.
_INVEST_VAR_NAMES = ("v_invest_p", "v_invest_n", "v_divest_p", "v_divest_n")

# Absolute tolerance below which a non-owner region's invest value for an
# entity is treated as a numerically-collapsed zero (the expected case for
# an out-of-region invest var).  A non-owner value above this triggers a
# (non-fatal) canary warning.
_NONOWNER_NONZERO_ABS_TOL = 1e-6

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

# Env override for the per-iteration region-solve worker count (machine-local;
# NO schema/DB knob — avoids another migration).  ``FLEXTOOL_BENDERS_WORKERS``
# pins the thread-pool size for the parallel region recourse pass; unset/<=0
# leaves the ``workers`` argument (default: auto = min(n_regions, cpu-1)) in
# charge.  ``workers=1`` (or 1 region) keeps the fully-sequential path.
_BENDERS_WORKERS_ENV = "FLEXTOOL_BENDERS_WORKERS"


def _resolve_benders_workers(n_regions: int, workers) -> int:
    """Resolve the effective region-solve worker count.

    Precedence: explicit ``FLEXTOOL_BENDERS_WORKERS`` env (machine-local) wins;
    otherwise the ``workers`` argument (``None`` ⇒ auto ``min(n, cpu-1)``).
    Clamped to ``[1, n_regions]`` by :func:`resolve_worker_count`.
    """
    env = os.environ.get(_BENDERS_WORKERS_ENV)
    if env:
        try:
            env_n = int(env)
        except ValueError:
            _logger.warning(
                "Benders: ignoring non-integer %s=%r", _BENDERS_WORKERS_ENV, env
            )
        else:
            if env_n > 0:
                workers = env_n
    return resolve_worker_count(n_regions, workers)


@dataclass
class Coupling:
    """One cross-region ``(p, source, sink)`` coupling pair.

    Pairs an export half-flow with the matching import half-flow and
    carries the per-region ``v_flow`` column ids for each cell of the
    arc.  This is the SHARED decomposition substrate consumed by
    :func:`_build_arcs` (and re-exported for tests).  The dual-subgradient
    ``lam`` multipliers that the old Lagrangian scheme stored here are NOT
    part of the Benders contract and have been dropped.
    """

    pipeline_key: tuple[str, str, str]
    export_region: str
    import_region: str
    export_cols: np.ndarray = field(
        default_factory=lambda: np.zeros(0, dtype=np.int64)
    )
    import_cols: np.ndarray = field(
        default_factory=lambda: np.zeros(0, dtype=np.int64)
    )


def _identify_coupling_cols(splits: list[RegionSplit],
                            warm: list[WarmProblem]) -> list[Coupling]:
    """Pair :class:`HalfFlow`s on ``(p, source, sink)`` and resolve the
    ``v_flow`` column ids per region.  Used by :func:`_build_arcs` and by
    the decomposition tests directly."""
    by_e: dict[tuple, tuple[str, list[HalfFlow]]] = {}
    by_i: dict[tuple, tuple[str, list[HalfFlow]]] = {}
    for s in splits:
        for hf in s.half_flows:
            key = (hf.original_p, hf.original_source, hf.original_sink)
            (by_e if hf.side == "export" else by_i).setdefault(
                key, (s.region, []))[1].append(hf)

    region_idx = {s.region: i for i, s in enumerate(splits)}
    out: list[Coupling] = []
    for key, (er, hfs_e) in by_e.items():
        if key not in by_i:
            continue
        ir, hfs_i = by_i[key]
        v_flow_e = warm[region_idx[er]]._p._vars["v_flow"]
        v_flow_i = warm[region_idx[ir]]._p._vars["v_flow"]
        ehf, ihf = hfs_e[0], hfs_i[0]

        def _cols(vf, hf):
            return (vf.frame.filter(
                (pl.col("p") == hf.virtual_p)
                & (pl.col("source") == hf.virtual_arc_source)
                & (pl.col("sink") == hf.virtual_arc_sink)
            ).sort("d", "t"))["col_id"].to_numpy().astype(np.int64)
        e_cols, i_cols = _cols(v_flow_e, ehf), _cols(v_flow_i, ihf)
        if e_cols.size == 0 or i_cols.size == 0:
            raise RuntimeError(
                f"Benders: empty coupling columns for arc {key!r} "
                f"(export={e_cols.size}, import={i_cols.size}).")
        if e_cols.size != i_cols.size:
            raise RuntimeError(
                f"Benders: pair size mismatch for {key!r}: "
                f"export={e_cols.size} vs import={i_cols.size}.")
        out.append(Coupling(
            pipeline_key=key, export_region=er, import_region=ir,
            export_cols=e_cols, import_cols=i_cols,
        ))
    return out


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
    # Whole-system, owner-de-duplicated invest/divest decision frames for the
    # TIER-1 invest->dispatch handoff.  Keys are a subset of
    # ``_INVEST_VAR_NAMES``; each value is a long-form frame whose columns
    # match ``polar_high.Solution.value(name)`` exactly (``(entity_col, "d",
    # "value")``), so a downstream ``SnapshotSolution`` can expose them to
    # ``build_handoff_from_solution``.  UNION of region in-region invest +
    # master trade-connection invest.  Empty dict when the model has no
    # investment.
    invest_solution_vars: dict[str, pl.DataFrame] = field(default_factory=dict)


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
        _silence_if_quiet(self._wp)
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
        # FlexData's ``pd_invest_set`` gives one invest period per
        # INVESTABLE pipe; we sum the master invest value over those periods
        # in ``read_master``.  A trade connection of FIXED EXISTING capacity
        # only (``existing>0`` + no ``invest_method``) carries NO
        # ``v_invest_p`` column — it contributes an empty ``_C_cols`` entry
        # (zero invested capacity, zero handoff invest), and its flow is
        # bounded natively by the FlexTool ``maxFlow`` row at
        # ``p_flow_upper_existing = existing/unitsize`` (model.py:2468-2473;
        # ``has_invest_p`` False ⇒ no ``-v_invest_p`` LHS term ⇒
        # ``v_flow ≤ existing/unitsize``).  Mixed masters (some pipes
        # investable, some existing-only) are handled in ONE master: the
        # FlexTool emit bounds each arc by ``existing + invested``, the
        # invest var existing only for the investable subset.
        conns = sorted({a.conn for a in self.arcs})
        invest_periods: dict[str, list[str]] = {}
        if reduced.pd_invest_set is not None:
            for r in reduced.pd_invest_set.iter_rows(named=True):
                invest_periods.setdefault(r["p"], []).append(r["d"])
        has_invest_var = "v_invest_p" in self._wp._p._vars
        self._C_cols: dict[str, list[int]] = {}
        for c in conns:
            periods = invest_periods.get(c, [])
            if not periods or not has_invest_var:
                # Existing-only trade connection: no invest var/column.  An
                # empty list ⇒ ``read_master`` reports invested capacity 0
                # for ``c`` and ``trade_invest_frame`` carries no invest row
                # for it.  The existing capacity term lives in the FlexTool
                # ``maxFlow`` RHS (``_existing_cap_by_col`` below), not in C.
                self._C_cols[c] = []
                continue
            self._C_cols[c] = [
                int(self._wp.col_id_of_var("v_invest_p", (c, d)))
                for d in periods
            ]

        # Per master f col-id, the EXISTING flow capacity ``existing/unitsize``
        # the FlexTool ``maxFlow`` RHS (``p_flow_upper_existing``) enforces.
        # The master-side capacity self-check (``solve_benders``) bounds each
        # cell's chosen flow by ``existing_cap + invested`` — so an
        # existing-only arc (C=0) is correctly allowed flow up to its existing
        # capacity, while a greenfield arc (existing=0) is unchanged.  The
        # reduced data's ``p_flow_upper_existing`` is keyed
        # ``(p, source, sink, d[, t])`` and already unitsize-normalised
        # (existing/unitsize), the SAME normalisation the master ``v_flow``
        # lives in (Phase-1 §A.5), so it drops in cell-for-cell.
        self._existing_cap_by_col: dict[int, float] = {}
        fue = getattr(reduced, "p_flow_upper_existing", None)
        ex_lookup: dict[tuple, float] = {}
        if fue is not None:
            fr = fue.frame
            has_t = "t" in fr.columns
            for r in fr.iter_rows(named=True):
                if has_t:
                    ex_lookup[(r["p"], r["source"], r["sink"],
                               r["d"], r["t"])] = float(r["value"])
                else:
                    ex_lookup[(r["p"], r["source"], r["sink"], r["d"])] = (
                        float(r["value"]))
        for a in self.arcs:
            for dt, cid in zip(a.dim_tuples, a.f_col_ids):
                p, s, k, d, t = dt
                cap = ex_lookup.get((p, s, k, d, t))
                if cap is None:
                    cap = ex_lookup.get((p, s, k, d), 0.0)
                self._existing_cap_by_col[int(cid)] = cap

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
        _silence_if_quiet(self._wp)
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
        # The hand master is GREENFIELD-only (Phase-2 prototype: existing=0 ⇒
        # all trade capacity is invested ``C``).  No existing-capacity term,
        # so the per-col existing cap is uniformly 0 — the capacity self-check
        # then reduces to ``f ≤ C`` exactly as before.
        self._existing_cap_by_col: dict[int, float] = {}
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

    def trade_invest_frame(self, sol) -> pl.DataFrame | None:
        """Return the master's trade-connection ``v_invest_p`` as a long-form
        ``(p, d, value)`` frame in the SAME ``Solution.value`` semantics /
        unitsize-normalisation FlexTool emits — for the TIER-1 invest
        handoff.  ``None`` on the hand master (no FlexTool ``v_invest_p``
        Var).

        The frame is built directly from ``Solution.value("v_invest_p")``
        (which indexes the master's ``v_invest_p`` Var frame by ``col_id``),
        so it is byte-identical in shape/units to what the monolith's
        ``v_invest_p`` value returns for the cross-region connections — the
        master is FlexTool-built over the network-only reduced data, so its
        ``v_invest_p`` is the very same normalised invest variable.  The
        cross-region connections are the ONLY entities the master invests in,
        so no extra filtering is needed (and they are disjoint from any
        region's in-region invest by construction)."""
        if self._master_kind != "flextool":
            return None
        if "v_invest_p" not in self._wp._p._vars:
            return None
        return sol.value("v_invest_p")

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


# ---------------------------------------------------------------------------
# TIER-1 whole-system invest assembly (region in-region + master trade).
# ---------------------------------------------------------------------------


def _resolve_entity_owner(
    region_membership: dict[str, dict[str, set[str]]],
    regions: list[str],
) -> dict[str, str]:
    """Build an ``entity -> owning-region`` map from region membership.

    Covers BOTH node entities (consumed by ``v_invest_n`` / ``v_divest_n``)
    and process/connection entities (consumed by ``v_invest_p`` /
    ``v_divest_p``).  A process is owned by the region that lists it in its
    ``"processes"`` set (the splitter assigns a process to a region via
    ``group_entity`` membership, i.e. the region containing its node(s)).

    *region_membership* is the EXCLUSIVE per-region membership returned by
    :func:`_region_filter.load_region_membership` — a region's OWN
    nodes/processes, NOT the shared set every region carries.  Ownership is
    therefore unambiguous for any entity claimed by exactly one region.

    An entity claimed by MORE than one region (shared, no unique owner) is
    assigned a deterministic owner: the first region in sorted region order
    that claims it (and a warning is emitted).  Iteration is over
    ``sorted(regions)`` so the tie-break is stable regardless of caller list
    order.

    Returns
    -------
    dict[str, str]
        ``{entity_name: region_name}`` for every node/process appearing in
        any region's membership.
    """
    owner: dict[str, str] = {}
    claims: dict[str, list[str]] = {}
    for region in sorted(regions):
        m = region_membership.get(region, {})
        for entity in m.get("nodes", set()) | m.get("processes", set()):
            claims.setdefault(entity, []).append(region)
    for entity, claiming in claims.items():
        owner[entity] = claiming[0]
        if len(claiming) > 1:
            _logger.warning(
                "Benders invest assembly: entity %r is shared across "
                "regions %r (no unique owner); assigning deterministic "
                "owner %r (first in sorted region order).  Shared "
                "invest-eligible entities are an untested edge case.",
                entity, claiming, owner[entity],
            )
    return owner


def _assemble_region_invest_vars(
    subproblems: list[Problem],
    subproblem_col_values: list[np.ndarray],
    owner_of_entity: Callable[[int, str], bool],
) -> dict[str, pl.DataFrame]:
    """Assemble whole-system invest/divest frames from the per-region
    recovered primal, keeping only owner-region rows.

    Parameters
    ----------
    subproblems
        Per-region :class:`polar_high.Problem` objects (region-index
        aligned).  Their ``_vars[name].frame`` carries ``(*dims, col_id)``
        and ``_vars[name].dims`` gives the natural dim order
        (``(entity_col, "d")`` for the invest/divest vars).
    subproblem_col_values
        Per-region recovered-primal ``col_value`` arrays
        (``Solution.col_value`` of each region's incumbent solve),
        region-index aligned with *subproblems*.  An empty / missing entry
        causes that region to be skipped.
    owner_of_entity
        Predicate ``(region_idx, entity) -> bool`` — ``True`` iff region
        ``region_idx`` OWNS ``entity``.  Only owned rows are kept, so the
        concatenated per-var frame has disjoint entity keys.

    Returns
    -------
    dict[str, pl.DataFrame]
        ``{name: frame}`` for each invest/divest var present in at least
        one region with >=1 owned row.  Each frame's columns exactly match
        ``polar_high.Solution.value(name)`` — ``(entity_col, "d",
        "value")`` — so a ``SnapshotSolution`` can serve them via
        ``.value(name)``.
    """
    out: dict[str, pl.DataFrame] = {}
    n_regions = len(subproblems)
    for name in _INVEST_VAR_NAMES:
        per_region_kept: list[pl.DataFrame] = []
        entity_col: str | None = None
        for i in range(n_regions):
            pb = subproblems[i]
            var = pb._vars.get(name)
            if var is None:
                continue
            if i >= len(subproblem_col_values):
                continue
            col_values = subproblem_col_values[i]
            if col_values is None or len(col_values) == 0:
                continue
            # Materialize this region's long-form frame exactly as
            # ``Solution.value(name)`` does: index the region's recovered
            # ``col_value`` by the Var's ``col_id`` and attach as "value".
            dims = tuple(var.dims)
            ent_col = dims[0]
            entity_col = ent_col
            frame = var.frame
            ids = frame["col_id"].to_numpy()
            vals = np.asarray(col_values)[ids]
            region_frame = frame.select(*dims).with_columns(
                value=pl.Series("value", vals)
            )
            # Owner-select: keep only rows whose entity is owned by this
            # region, so the concatenated frame has disjoint entity keys.
            entities = region_frame[ent_col].to_list()
            owned_mask = [bool(owner_of_entity(i, e)) for e in entities]
            # Canary: a NON-owner region carrying a materially non-zero
            # value violates the owner-selection assumption.  Warn, keep
            # only the owner's value.
            value_series = region_frame["value"].to_list()
            for e, owned, v in zip(entities, owned_mask, value_series):
                if (not owned) and v is not None and abs(v) > _NONOWNER_NONZERO_ABS_TOL:
                    _logger.warning(
                        "Benders invest assembly: non-owner region index "
                        "%d carries non-zero %s value %.6g for entity %r "
                        "(expected ~0 for an out-of-region invest var); "
                        "keeping only the owner's value.",
                        i, name, v, e,
                    )
            kept = region_frame.filter(pl.Series("__owned", owned_mask))
            if kept.height > 0:
                per_region_kept.append(kept)
        if per_region_kept:
            frame = pl.concat(per_region_kept, how="vertical")
            sort_cols = [c for c in (entity_col, "d") if c in frame.columns]
            if sort_cols:
                frame = frame.sort(sort_cols, maintain_order=True)
            out[name] = frame
    return out


def _assemble_benders_invest_vars(
    *,
    subproblems: list[Problem],
    region_of_index: list[str],
    region_membership: dict[str, dict[str, set[str]]],
    regions: list[str],
    region_col_values: list[np.ndarray] | None,
    master_trade_invest: pl.DataFrame | None,
    trade_conns: set[str],
) -> dict[str, pl.DataFrame]:
    """Assemble the whole-system TIER-1 invest handoff: the UNION of each
    region's owner-selected in-region invest/divest frames and the master's
    trade-connection ``v_invest_p``.

    The two contributions are DISJOINT by construction, but the partition is
    NOT pure region-membership ownership: a cross-region trade connection
    typically appears in BOTH regions' ``group_entity`` membership (it
    touches a node in each), so ``_resolve_entity_owner`` would otherwise
    hand it to one of them — clobbering the master's correct invested value
    with that region's pinned half-flow model's ZERO invest var.  The MASTER
    owns the trade-connection invest (it is the variable the capacity
    coupling acts on), so we EXCLUDE the trade connections from the region
    invest and take their value SOLELY from ``master_trade_invest``.  The
    result is the disjoint union: in-region entities from the regions, the
    cross-region pipes from the master.

    Returns the same-shaped dict the downstream ``SnapshotSolution`` /
    ``build_handoff_from_solution`` expects (each frame's columns match
    ``Solution.value(name)`` exactly: ``(entity_col, "d", "value")``).
    """
    # (a) region in-region invest, owner-de-duplicated, with the cross-region
    # trade connections EXCLUDED (they are the master's, see docstring).
    out: dict[str, pl.DataFrame] = {}
    if region_col_values is not None:
        owner_by_entity = _resolve_entity_owner(region_membership, regions)

        def _owner_of_entity(region_idx: int, entity: str) -> bool:
            if entity in trade_conns:
                return False  # master-owned; never claimed by a region
            return owner_by_entity.get(entity) == region_of_index[region_idx]

        out = _assemble_region_invest_vars(
            subproblems, region_col_values, _owner_of_entity
        )

    # (b) master trade-connection invest (``v_invest_p`` only — the master
    # invests in cross-region connections, never nodes).  Union into the
    # region ``v_invest_p`` frame.  The region part excluded these entities,
    # so the union is disjoint by construction; the defensive ``unique`` is a
    # belt-and-braces guard only.
    trade = master_trade_invest
    if trade is not None and trade.height > 0:
        existing = out.get("v_invest_p")
        if existing is not None:
            ent_col = trade.columns[0]
            trade = trade.select(existing.columns)
            merged = pl.concat([existing, trade], how="vertical")
            merged = merged.unique(
                subset=[ent_col, "d"], keep="first", maintain_order=True
            )
            sort_cols = [c for c in (ent_col, "d") if c in merged.columns]
            out["v_invest_p"] = (
                merged.sort(sort_cols, maintain_order=True)
                if sort_cols else merged
            )
        else:
            out["v_invest_p"] = trade
    return out


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
    progress_callback: Callable[[dict], None] | None = None,
    subsolve_callback: Callable[[dict], None] | None = None,
    workers: int | None = None,
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
    progress_callback
        Optional ``Callable[[dict], None]`` invoked ONCE per outer Benders
        iteration (after that iteration's master + region solves), so a
        caller (e.g. the orchestrator) can stream live per-iter lines.  The
        dict carries at least ``iter`` (1-based), ``lower_bound``,
        ``upper_bound`` (this iter's UB), ``best_upper_bound``, ``gap``,
        ``converged`` (bool), and ``region_costs`` — all bounds/costs in
        REAL units (÷s), matching the returned :class:`BendersResult`.
        No-op when ``None``; the loop behaviour is byte-identical when
        omitted.
    workers
        Worker-thread count for the per-iteration region recourse pass (the
        N independent region subproblem solves).  ``None`` (default)
        auto-resolves to ``min(n_regions, cpu_count - 1)``; ``<= 1`` (or a
        single region) keeps the fully-sequential path.  The machine-local
        env override ``FLEXTOOL_BENDERS_WORKERS`` takes precedence when set.
        The region solves are independent (each region owns its own HiGHS
        handle) and each solves single-threaded, so the parallel result is
        DETERMINISTIC — bit-identical ``(cost_r, cut slopes)`` and therefore
        identical LB/UB/iteration-count to ``workers=1`` (see the
        determinism gate test).

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
            progress_callback=progress_callback,
            subsolve_callback=subsolve_callback, workers=workers,
        )
    finally:
        if _enums_token is not None:
            reset_global_axis_enums(_enums_token)


def _solve_benders_inner(data, regions, *, max_iters, tol, monolith_objective,
                         build_problem, master="flextool",
                         obj_scale: float = 1.0,
                         progress_callback=None, subsolve_callback=None,
                         workers=None) -> BendersResult:
    # --- split with the cross-region half-flows UNCAPPED so the master pin is
    # feasible (Phase-2 splitter Benders mode).
    splits = _region_filter.split(
        data, regions=regions, benders_uncap_cross_region=True
    )
    subproblems = [Problem() for _ in splits]
    for s, pb in zip(splits, subproblems):
        build_problem(pb, s.data)
    warm = [WarmProblem(p) for p in subproblems]
    # Silence each region's per-solve HiGHS log by default (output_flag
    # persists across the cold build below and every warm parallel re-solve).
    for w in warm:
        _silence_if_quiet(w)
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
        0) and solve; return (cost_r, {master_f_col: slope}, region Solution).

        The returned :class:`polar_high.Solution` carries the region's
        recovered primal (incl. its in-region ``v_invest_p``/``v_invest_n``),
        used to assemble the whole-system TIER-1 invest handoff at the
        incumbent."""
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
        return float(sol_r.obj), slopes, sol_r

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

    # --- Parallel region recourse: the N region subproblems are independent
    # (each owns its own WarmProblem / HiGHS handle, and HiGHS run() releases
    # the GIL), so they fan out across a thread pool.  Every region's cold first
    # build already ran SEQUENTIALLY above (the ``for w in warm: w.solve()``
    # loop), so ``solve_indexed_parallel`` only parallelizes WARM re-solves and
    # the per-region solve is single-threaded + deterministic — the recovered
    # ``(cost_r, slopes)`` are bit-identical to the sequential path.  The
    # WarmProblem list is region-index aligned with ``regions_meta`` (both
    # iterate ``splits`` in order).
    region_warm = [rm.wp for rm in regions_meta]
    eff_workers = _resolve_benders_workers(len(regions_meta), workers)
    _logger.info(
        "Benders: region recourse pass over %d region(s) with %d worker(s)",
        len(regions_meta), eff_workers,
    )

    # Current outer-iteration index, surfaced to ``subsolve_callback`` so the
    # orchestrator can label each region-finish line (bootstrap = 0).
    _cur_iter = [0]

    def _solve_regions(f_bar_local):
        """Pin+solve every region at ``f_bar_local`` (parallel when
        ``eff_workers > 1``), returning a per-region-index list of
        ``(cost_r, slopes, sol_r)`` in deterministic order.  Fires
        ``subsolve_callback`` once per region as it FINISHES (from the worker
        thread; the callback must be thread-safe)."""
        def _fn(i):
            res = _pin_and_solve(regions_meta[i], f_bar_local)
            if subsolve_callback is not None:
                try:
                    subsolve_callback({
                        "iter": _cur_iter[0],
                        "region": regions_meta[i].name,
                        "obj": res[0] / obj_scale,  # cost_r → REAL units
                    })
                except Exception:  # noqa: BLE001 — observer must not break solve
                    pass
            return res
        return solve_indexed_parallel(region_warm, _fn, workers=eff_workers)

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
    for rm, (cost_r, slopes, _sol_r) in zip(regions_meta, _solve_regions(f_bar)):
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
    # Inverse objective scale: the loop's LB/UB arithmetic runs in scaled
    # space (objectives built ×s); callers/tests (and the progress callback)
    # expect REAL-unit costs (÷s).  s=1.0 ⇒ no-op.
    inv_s = 1.0 / obj_scale

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
        # FlexTool ``maxFlow`` row ``f ≤ existing_cap + Σ v_invest_p`` holds at
        # the master optimum) — a cheap feasibility self-check that the UB
        # below is a valid primal point.  For a GREENFIELD arc the existing
        # term is 0 (cap = invested C); for an EXISTING-only arc the invested
        # C is 0 (cap = existing/unitsize); for a BOTH arc both contribute.
        existing_cap_by_col = master._existing_cap_by_col
        # The solver returns a vertex within its feasibility tolerance of the
        # active rows, so the coupling row ``C − f ≥ 0`` can carry a tiny slack
        # on ``f ≤ cap``.  HiGHS enforces feasibility on the INTERNALLY-SCALED
        # problem, so the UNSCALED slack reported here can exceed the nominal
        # (scaled) tolerance — especially on a normalised coupling row with
        # small capacity.  Derive the self-check budget from the solver's own
        # achieved infeasibility (generic to whatever the solve produced) and
        # floor it with the sibling cut-satisfaction self-check's relative
        # tolerance so a near-zero reported value still admits the vertex slack.
        # A violation above this is a genuine read/stale-state bug, not solver
        # slack — the coupling row's slack is by construction ≤ this maximum.
        solver_feas = msol.max_primal_infeasibility
        for a in arcs:
            invested = C_by_conn.get(a.conn, 0.0)
            for cid in a.f_col_ids:
                cap = invested + existing_cap_by_col.get(int(cid), 0.0)
                f_val = new_f_bar[int(cid)]
                tol = max(1e-5 * max(1.0, abs(cap), abs(f_val)), solver_feas)
                if f_val > cap + tol:
                    raise RuntimeError(
                        f"Benders master infeasible coupling: "
                        f"f={f_val} > cap[{a.conn}]={cap} (slack {f_val - cap:.3e} "
                        f"> tol {tol:.3e}) "
                        f"(invested={invested}, "
                        f"existing={existing_cap_by_col.get(int(cid), 0.0)})"
                    )

        # --- advance f̄ to the master optimum and solve the regions there to
        # (a) get this iteration's recourse cost (→ a VALID UB, since C ≥ f̄ at
        # the master optimum) and (b) produce the next iteration's cuts.
        f_bar = new_f_bar
        _cur_iter[0] = iterations  # label this pass's region-finish lines
        region_costs: dict[str, float] = {}
        next_cuts: list[tuple[str, float, dict[int, float]]] = []
        # Per-region recovered primal at THIS f̄ (region-index aligned), for
        # the TIER-1 whole-system invest assembly when this iteration becomes
        # the incumbent.  Each entry is the region Solution's ``col_value``.
        region_col_values: list[np.ndarray] = [None] * len(regions_meta)
        for rm, (cost_r, slopes, sol_r) in zip(regions_meta, _solve_regions(f_bar)):
            region_costs[rm.name] = cost_r
            next_cuts.append((rm.name, cost_r, slopes))
            region_col_values[region_idx[rm.name]] = np.asarray(
                sol_r.col_value
            ).copy()

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
                # The master trade ``v_invest_p`` frame + per-region primal AT
                # this incumbent, for the TIER-1 invest handoff (master trade
                # ∪ region in-region invest, owner-de-duplicated).  The master
                # frame is MATERIALIZED here (a fresh DataFrame), not held as a
                # Solution reference: the warm-restart reuses the master's
                # ``col_value`` buffer across iterations, so a stashed Solution
                # would read a later iteration's values.  ``region_col_values``
                # are already per-region ``.copy()``-d above.
                "master_trade_invest": master.trade_invest_frame(msol),
                "region_col_values": region_col_values,
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
        # DEBUG (not info): the orchestrator streams the user-facing per-iter
        # line in REAL units via ``progress_callback``; this internal line
        # carries the raw SCALED values and would otherwise duplicate it.
        _logger.debug(
            "Benders iter %d: LB=%.6e UB=%.6e bestUB=%.6e gap=%.3e",
            iterations, LB, UB, best_UB, gap,
        )

        if progress_callback is not None:
            # Stream one live per-iteration summary.  Bounds are reported in
            # REAL units (÷s) so the orchestrator's lines match the returned
            # ``BendersResult`` fields regardless of ``scale_the_objective``.
            progress_callback({
                "iter": iterations,
                "lower_bound": LB * inv_s,
                "upper_bound": UB * inv_s,
                "best_upper_bound": best_UB * inv_s,
                "gap": gap,
                "converged": gap <= tol,
                "region_costs": {r: c * inv_s for r, c in region_costs.items()},
            })

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
        "master_trade_invest": (
            master.trade_invest_frame(msol) if iterations else None
        ),
        "region_col_values": region_col_values if iterations else None,
    }
    trade_flow = _flow_frames(arcs, inc["f_bar"])

    # --- TIER-1 whole-system invest handoff (GAP-a).  Assemble the same-shaped
    # ``{v_invest_p/v_invest_n/v_divest_p/v_divest_n -> (entity, d, value)}``
    # dict the downstream rolling-dispatch consumes, as the UNION of:
    #   (a) each REGION's in-region invest (owner-de-duplicated so each entity
    #       is claimed exactly once), AND
    #   (b) the MASTER's trade-connection ``v_invest_p`` (the cross-region
    #       pipes the master owns; disjoint from any region's in-region invest
    #       since the splitter never assigns a cross-region connection to a
    #       region's membership).
    # NORMALISATION: both the region subproblems and the master are FlexTool-
    # built (``build_flextool``), so their ``v_invest_p`` carry IDENTICAL
    # p_unitsize-normalised units — the same units ``Solution.value("v_invest_p")``
    # returns on the monolith.  The assembled frames therefore drop straight
    # into ``build_handoff_from_solution`` with no rescale.
    invest_solution_vars = _assemble_benders_invest_vars(
        subproblems=subproblems,
        region_of_index=[s.region for s in splits],
        region_membership=_region_filter.load_region_membership(data, regions),
        regions=regions,
        region_col_values=inc.get("region_col_values"),
        master_trade_invest=inc.get("master_trade_invest"),
        trade_conns={a.conn for a in arcs},
    )

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
        invest_solution_vars=invest_solution_vars,
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
