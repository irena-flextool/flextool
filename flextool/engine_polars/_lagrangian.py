"""Lagrangian decomposition wrapper (flextool side).

Generic dual-subgradient algorithm lives in :mod:`polar_high_opt.lagrangian`.
This module slices a whole-system :class:`flextool.input.FlexData` via
:mod:`flextool._region_filter`, builds per-region
:class:`polar_high_opt.Problem`s, translates half-flow pair metadata into
:class:`polar_high_opt.CouplingSpec`s, and delegates to
:class:`polar_high_opt.LagrangianProblem`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import polars as pl

from polar_high_opt import (CouplingEntry, CouplingSpec, LagrangianProblem,
                    Problem, WarmProblem)

from flextool.engine_polars import _region_filter
from flextool.engine_polars.input import FlexData
from flextool.engine_polars._region_filter import HalfFlow, RegionSplit


__all__ = ["Coupling", "LagrangianResult", "solve_lagrangian"]


@dataclass
class Coupling:
    """One cross-region (p, source, sink) coupling pair (back-compat
    surface; the live multipliers live in flexpy's resolved state)."""
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
    :class:`polar_high_opt.CouplingSpec`s (coefs +1 / -1, rhs 0)."""
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
) -> LagrangianResult:
    """Run Lagrangian decomposition on whole-system *data*."""
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
            f"solve_lagrangian: need ≥2 lagrangian_region groups; "
            f"got {regions!r}.  Use Problem().solve() for non-decomposed "
            f"scenarios.")

    if build_problem is None:
        from flextool.engine_polars.model import build_flextool as _bf
        build_problem = _bf

    splits = _region_filter.split(data, regions=regions)
    subproblems = [Problem() for _ in splits]
    for s, pb in zip(splits, subproblems):
        build_problem(pb, s.data)

    warm_lookup = [WarmProblem(p) for p in subproblems]
    couplings = _identify_coupling_cols(splits, warm_lookup)
    if not couplings:
        region_objs: dict[str, float] = {}
        for s, pb in zip(splits, subproblems):
            sol = pb.solve()
            if not sol.optimal:
                raise RuntimeError(
                    f"Lagrangian trivial solve {s.region!r} not optimal")
            region_objs[s.region] = sol.obj
        return LagrangianResult(
            converged=True, iterations=0,
            total_objective=sum(region_objs.values()),
            region_objectives=region_objs, final_lambdas={}, couplings=[])

    specs = _build_coupling_specs(splits, warm_lookup, couplings)
    sol = LagrangianProblem(subproblems, specs).solve(
        max_iters=max_iters, tol=tol, step=alpha,
        initial_lambda=initial_lambda, min_iters=min_iters,
        primal_tail=primal_tail)

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
    )
