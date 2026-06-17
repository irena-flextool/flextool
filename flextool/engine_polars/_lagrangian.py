"""Lagrangian decomposition wrapper (flextool side).

Generic dual-subgradient algorithm lives in :mod:`polar_high.lagrangian`.
This module slices a whole-system :class:`flextool.input.FlexData` via
:mod:`flextool._region_filter`, builds per-region
:class:`polar_high.Problem`s, translates half-flow pair metadata into
:class:`polar_high.CouplingSpec`s, and delegates to
:class:`polar_high.LagrangianProblem`.
"""
from __future__ import annotations

import inspect
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import polars as pl

from polar_high import (CouplingEntry, CouplingSpec, LagrangianProblem,
                    Problem, WarmProblem)

from flextool.engine_polars import _region_filter
from flextool.engine_polars.input import FlexData
from flextool.engine_polars._region_filter import HalfFlow, RegionSplit
from flextool.engine_polars._axis_enums import (
    get_global_axis_enums,
    reset_global_axis_enums,
    set_global_axis_enums,
)


__all__ = ["Coupling", "LagrangianResult", "solve_lagrangian"]

_logger = logging.getLogger(__name__)

# The four investment/divestment decision variables assembled into a
# whole-system handoff.  Each is declared in ``model.py`` over a 2-tuple
# of dims whose FIRST element is the entity axis ("p" for process /
# connection vars, "n" for node vars) and whose second is the period
# axis "d".  ``Var.dims`` is read at runtime to recover the exact entity
# column name — we do not hard-code it here.
_INVEST_VAR_NAMES = ("v_invest_p", "v_invest_n", "v_divest_p", "v_divest_n")

# Absolute tolerance below which a non-owner region's invest value for an
# entity is considered a numerically-collapsed zero (the expected case;
# see spec §1c / Critique Claim 2b).  A non-owner value above this is a
# violated assumption and triggers a (non-fatal) canary warning.
_NONOWNER_NONZERO_ABS_TOL = 1e-6


@dataclass
class Coupling:
    """One cross-region (p, source, sink) coupling pair (back-compat
    surface; the live multipliers live in polar_high's resolved state)."""
    pipeline_key: tuple[str, str, str]
    export_region: str
    import_region: str
    export_cols: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))
    import_cols: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))
    lam_vec: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float64))

    @property
    def lam(self) -> float:
        return float(self.lam_vec.mean()) if self.lam_vec.size else 0.0


@dataclass
class LagrangianResult:
    converged: bool
    iterations: int
    total_objective: float
    region_objectives: dict[str, float]
    final_lambdas: dict[tuple[str, str, str], float]
    iteration_log: list[dict] = field(default_factory=list)
    couplings: list[Coupling] = field(default_factory=list)
    # ``best_dual_total`` is the tight lower bound (max Σ region obj across
    # iters) and equals ``total_objective``; ``recovered_total`` is the
    # tail-averaged fix-and-resolve primal.  Their relative gap is the
    # decomposition's optimality tradeoff — surfaced in the orchestrator's
    # per-solve summary.  Defaults cover the no-coupling trivial path.
    best_dual_total: float = 0.0
    recovered_total: float = 0.0
    # Whole-system, owner-de-duplicated invest/divest decision frames
    # assembled from the per-region recovered primal.  Keys are a subset
    # of ``_INVEST_VAR_NAMES``; each value is a long-form frame whose
    # columns match ``polar_high.Solution.value(name)`` exactly
    # (``(entity_col, "d", "value")``), so a downstream
    # ``SnapshotSolution`` can expose them via ``.value(name)`` to
    # ``build_handoff_from_solution``.  Empty dict when the model has no
    # investment (the var is absent from every region's ``Problem``).
    invest_solution_vars: dict[str, pl.DataFrame] = field(default_factory=dict)


def _identify_coupling_cols(splits: list[RegionSplit],
                            warm: list[WarmProblem]) -> list[Coupling]:
    """Pair :class:`HalfFlow`s on (p, source, sink) and resolve v_flow
    column ids per region.  Used by tests directly."""
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
                f"Lagrangian: empty coupling columns for arc {key!r} "
                f"(export={e_cols.size}, import={i_cols.size}).")
        if e_cols.size != i_cols.size:
            raise RuntimeError(
                f"Lagrangian: pair size mismatch for {key!r}: "
                f"export={e_cols.size} vs import={i_cols.size}.")
        out.append(Coupling(
            pipeline_key=key, export_region=er, import_region=ir,
            export_cols=e_cols, import_cols=i_cols,
            lam_vec=np.zeros(e_cols.size, dtype=np.float64),
        ))
    return out


def _build_coupling_specs(splits: list[RegionSplit], warm: list[WarmProblem],
                          couplings: list[Coupling]) -> list[CouplingSpec]:
    """Translate :class:`Coupling`s into 2-entry consensus
    :class:`polar_high.CouplingSpec`s (coefs +1 / -1, rhs 0)."""
    region_idx = {s.region: i for i, s in enumerate(splits)}
    specs: list[CouplingSpec] = []
    for cpl in couplings:
        v_flow_e = warm[region_idx[cpl.export_region]]._p._vars["v_flow"]
        v_flow_i = warm[region_idx[cpl.import_region]]._p._vars["v_flow"]

        def _tuples(vf, ids):
            rows = vf.frame.filter(pl.col("col_id").is_in(ids)).sort("d", "t")
            return [tuple(r) for r in rows.select(*vf.dims).iter_rows()]
        specs.append(CouplingSpec(
            entries=[
                CouplingEntry(region_idx[cpl.export_region], "v_flow",
                              _tuples(v_flow_e, cpl.export_cols), +1.0),
                CouplingEntry(region_idx[cpl.import_region], "v_flow",
                              _tuples(v_flow_i, cpl.import_cols), -1.0),
            ],
            rhs=0.0, key=cpl.pipeline_key,
        ))
    return specs


def _resolve_entity_owner(
    region_membership: dict[str, dict[str, set[str]]],
    regions: list[str],
) -> dict[str, str]:
    """Build an ``entity -> owning-region`` map from region membership.

    Covers BOTH node entities (consumed by ``v_invest_n`` / ``v_divest_n``)
    and process/connection entities (consumed by ``v_invest_p`` /
    ``v_divest_p``).  A process is owned by the region that lists it in
    its ``"processes"`` set (the splitter assigns a process to a region
    via ``group_entity`` membership, i.e. the region containing its
    node(s)).

    *region_membership* is the EXCLUSIVE per-region membership returned by
    :func:`_region_filter.load_region_membership` — a region's OWN
    nodes/processes, NOT the shared set every region carries in
    ``keep_nodes`` / ``keep_procs``.  Ownership is therefore unambiguous
    for any entity claimed by exactly one region.

    An entity claimed by MORE than one region (shared across regions —
    no unique owner) is assigned a deterministic owner: the first region
    in sorted region order that claims it.  This is an untested edge case
    (shared invest-eligible entities are unusual), so a warning is
    emitted.

    Region iteration order is ``sorted(regions)`` so the deterministic
    shared-owner tie-break is stable regardless of the caller's list
    order.

    Returns
    -------
    dict[str, str]
        ``{entity_name: region_name}`` for every node and process that
        appears in any region's membership.
    """
    owner: dict[str, str] = {}
    claims: dict[str, list[str]] = {}
    for region in sorted(regions):
        m = region_membership.get(region, {})
        for entity in m.get("nodes", set()) | m.get("processes", set()):
            claims.setdefault(entity, []).append(region)
    for entity, claiming in claims.items():
        # ``claiming`` is already in sorted region order (we iterate
        # ``sorted(regions)`` above), so ``claiming[0]`` is deterministic.
        owner[entity] = claiming[0]
        if len(claiming) > 1:
            _logger.warning(
                "Lagrangian invest assembly: entity %r is shared across "
                "regions %r (no unique owner); assigning deterministic "
                "owner %r (first in sorted region order).  Shared "
                "invest-eligible entities are an untested edge case.",
                entity, claiming, owner[entity],
            )
    return owner


def _assemble_invest_vars(
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
        aligned).  Their ``_vars[name].frame`` carries ``(*dims,
        col_id)`` and ``_vars[name].dims`` gives the natural dim order
        (``(entity_col, "d")`` for the invest/divest vars).
    subproblem_col_values
        Per-region recovered-primal ``col_value`` arrays from
        :attr:`polar_high.LagrangianSolution.subproblem_col_values`,
        region-index aligned with *subproblems*.  An empty / missing
        entry (e.g. an older polar_high that did not retain the field)
        causes that region to be skipped.
    owner_of_entity
        Predicate ``(region_idx, entity) -> bool`` — ``True`` iff region
        ``region_idx`` OWNS ``entity``.  Only owned rows are kept, so the
        concatenated per-var frame has disjoint entity keys by
        construction.

    Returns
    -------
    dict[str, pl.DataFrame]
        ``{name: frame}`` for each invest/divest var present in at least
        one region with ≥1 owned row.  Each frame's columns exactly match
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
            # region.  ``owner_of_entity`` is a Python predicate, so apply
            # it on the (small) distinct entity list and filter.
            entities = region_frame[ent_col].to_list()
            owned_mask = [bool(owner_of_entity(i, e)) for e in entities]
            # Canary: a NON-owner region carrying a materially non-zero
            # value violates the owner-selection assumption (spec §1c /
            # Critique 2b).  Warn but do not raise.
            value_series = region_frame["value"].to_list()
            for e, owned, v in zip(entities, owned_mask, value_series):
                if (not owned) and v is not None and abs(v) > _NONOWNER_NONZERO_ABS_TOL:
                    _logger.warning(
                        "Lagrangian invest assembly: non-owner region "
                        "index %d carries non-zero %s value %.6g for "
                        "entity %r (expected ~0 for an out-of-region "
                        "invest var); keeping only the owner's value.",
                        i, name, v, e,
                    )
            kept = region_frame.filter(pl.Series("__owned", owned_mask))
            if kept.height > 0:
                per_region_kept.append(kept)
        if per_region_kept:
            frame = pl.concat(per_region_kept, how="vertical")
            # Defensive ordering: stable sort by (entity, d) so the
            # assembled frame is deterministic across region order.
            sort_cols = [c for c in (entity_col, "d") if c in frame.columns]
            if sort_cols:
                frame = frame.sort(sort_cols, maintain_order=True)
            out[name] = frame
    return out


def solve_lagrangian(
    data: FlexData, *,
    work_dir: Path | str | None = None,
    regions: list[str] | None = None,
    alpha: float = 1e-3,
    max_iters: int = 200,
    tol: float = 1.0,
    primal_tail: int | None = None,
    build_problem: Callable[[Problem, FlexData], None] | None = None,
    decomposition_method: dict[str, str] | None = None,
    initial_lambda: float = 0.0,
    min_iters: int = 1,
    solver_config: "object | None" = None,
    progress_callback: Callable[[dict], None] | None = None,
) -> LagrangianResult:
    """Run Lagrangian decomposition on whole-system *data*.

    Parameters
    ----------
    solver_config
        Optional :class:`flextool.engine_polars._solve_config.SolverConfig`.
        Lagrangian decomposition is currently HiGHS-only (upstream
        polar-high gap: :class:`polar_high.LagrangianProblem` does not
        accept ``solver_name``).  When *solver_config* is provided and
        its ``name != "highs"``, this function raises
        :class:`flextool.engine_polars._solver_dispatch.FlexToolUserError`
        rather than silently running on HiGHS.  When omitted (the
        legacy code path), HiGHS is assumed without checking.
    """
    if solver_config is not None and getattr(solver_config, "name", "highs") != "highs":
        from flextool.engine_polars._solver_dispatch import FlexToolUserError

        raise FlexToolUserError(
            f"Lagrangian decomposition currently requires HiGHS (polar-high "
            f"upstream gap: LagrangianProblem.solve does not accept "
            f"solver_name).  Solve has solver={solver_config.name!r}; either "
            f"set ``solver = highs`` for this solve or remove "
            f"``decomposition_method = lagrangian_region`` from the group "
            f"to use the cascade dispatch."
        )
    if decomposition_method is None and work_dir is not None:
        decomposition_method = _region_filter.load_decomposition_method(work_dir)
    if decomposition_method is None:
        decomposition_method = {}
    lagr = sorted(g for g, m in decomposition_method.items()
                  if m == "lagrangian_region")
    if regions is None:
        regions = lagr
    if not regions or len(regions) < 2:
        raise ValueError(
            f"solve_lagrangian: need >=2 lagrangian_region groups; "
            f"got {regions!r}.  Use Problem().solve() for non-decomposed "
            f"scenarios.")

    if build_problem is None:
        from flextool.engine_polars.model import build_flextool as _bf
        build_problem = _bf

    # Activate ``data``'s own axis_enums snapshot for the duration of
    # the decomposition.  ``load_flextool`` leaves the live global set
    # to the most-recently-loaded data's vocabulary; in tests where a
    # sibling test loads a different DB between the lh2 fixture load
    # and this call, the live global no longer matches ``data``'s
    # frames, and downstream cast_dim / is_in operations fail with
    # ``conversion from str to enum failed`` or Enum-mismatch joins.
    # ``_region_filter.split`` widens this snapshot with virtual
    # half-flow tokens; the widened global persists across the
    # build_problem calls below thanks to the outer try/finally.
    _data_enums = getattr(data, "_axis_enums", None)
    _enums_token = None
    if _data_enums is not None and _data_enums != get_global_axis_enums():
        _enums_token = set_global_axis_enums(_data_enums)
    try:
        splits = _region_filter.split(data, regions=regions)
        subproblems = [Problem() for _ in splits]
        for s, pb in zip(splits, subproblems):
            build_problem(pb, s.data)

        # Resolve an ``entity -> owning-region`` map for invest/divest
        # assembly.  ``RegionSplit`` does not carry membership, so we
        # recompute it (cheap) from the EXCLUSIVE per-region membership.
        # The region order of ``splits`` matches ``regions`` (see
        # ``_region_filter.split``), so a region INDEX maps to a region
        # NAME via ``splits[i].region``.
        _region_membership = _region_filter.load_region_membership(
            data, regions)
        _owner_by_entity = _resolve_entity_owner(_region_membership, regions)
        _region_of_index = [s.region for s in splits]

        def _owner_of_entity(region_idx: int, entity: str) -> bool:
            return _owner_by_entity.get(entity) == _region_of_index[region_idx]

        warm_lookup = [WarmProblem(p) for p in subproblems]
        couplings = _identify_coupling_cols(splits, warm_lookup)
        if not couplings:
            region_objs: dict[str, float] = {}
            _trivial_col_values: list[np.ndarray] = []
            for s, pb in zip(splits, subproblems):
                sol = pb.solve()
                if not sol.optimal:
                    raise RuntimeError(
                        f"Lagrangian trivial solve {s.region!r} not optimal")
                region_objs[s.region] = sol.obj
                _trivial_col_values.append(np.asarray(sol.col_value).copy())
            _trivial_total = sum(region_objs.values())
            _trivial_invest = _assemble_invest_vars(
                subproblems, _trivial_col_values, _owner_of_entity)
            return LagrangianResult(
                converged=True, iterations=0,
                total_objective=_trivial_total,
                region_objectives=region_objs, final_lambdas={}, couplings=[],
                best_dual_total=_trivial_total,
                recovered_total=_trivial_total,
                invest_solution_vars=_trivial_invest)

        specs = _build_coupling_specs(splits, warm_lookup, couplings)
        _solve = LagrangianProblem(subproblems, specs).solve
        _solve_kwargs = dict(
            max_iters=max_iters, tol=tol, step=alpha,
            initial_lambda=initial_lambda, min_iters=min_iters,
            primal_tail=primal_tail)
        # ``progress_callback`` (live per-iteration streaming) landed in
        # polar-high 2.7.0.  Only forward it when the installed version
        # accepts it, so an older polar-high still runs the solve (the
        # live lines are simply absent; the returned result — and the
        # caller's final summary — are unchanged).
        if (
            progress_callback is not None
            and "progress_callback" in inspect.signature(_solve).parameters
        ):
            _solve_kwargs["progress_callback"] = progress_callback
        sol = _solve(**_solve_kwargs)
    finally:
        if _enums_token is not None:
            reset_global_axis_enums(_enums_token)

    # Assemble whole-system invest/divest frames from the per-region
    # recovered primal.  ``subproblems`` (the per-region ``_vars``) and
    # ``_owner_of_entity`` survive the ``finally`` (they are locals, not
    # deleted).  ``subproblem_col_values`` is the additive polar_high
    # retention field (>=2.8.0); an older polar_high leaves it empty and
    # the helper returns ``{}`` (TIER 1 silently disabled).
    _invest_vars = _assemble_invest_vars(
        subproblems,
        list(getattr(sol, "subproblem_col_values", []) or []),
        _owner_of_entity,
    )

    region_objs = {s.region: o for s, o in zip(splits, sol.subproblem_objectives)}
    final_lambdas: dict[tuple[str, str, str], float] = {}
    for cpl, lam in zip(couplings, sol.final_lambdas):
        cpl.lam_vec = lam.copy()
        final_lambdas[cpl.pipeline_key] = float(lam.mean()) if lam.size else 0.0

    domain_log: list[dict] = []
    for entry in sol.iteration_log:
        if entry.get("iter") == -1:
            domain_log.append(entry)
            continue
        domain_log.append({
            "iter": entry["iter"], "alpha_k": entry["alpha_k"],
            "max_abs_imbalance": entry["max_abs_residual"],
            "total_obj": entry["total_obj"],
            "lambdas_mean": {c.pipeline_key: c.lam for c in couplings},
            "imbalances_max_cell": {c.pipeline_key: entry["max_abs_residual"]
                                     for c in couplings},
        })

    return LagrangianResult(
        converged=sol.converged, iterations=sol.iterations,
        total_objective=sol.total_objective,
        region_objectives=region_objs,
        final_lambdas=final_lambdas,
        iteration_log=domain_log,
        couplings=couplings,
        best_dual_total=getattr(sol, "best_dual_total", sol.total_objective),
        recovered_total=getattr(sol, "recovered_total", sol.total_objective),
        invest_solution_vars=_invest_vars,
    )
