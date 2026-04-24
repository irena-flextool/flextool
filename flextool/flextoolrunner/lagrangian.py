"""Spatial Lagrangian coordinator for decomposition-group-split models.

Scheme (Agent 3.2): dualise the commodity balance on each cross-region
pipeline.  Each region *r* owns its own LP; cross-region flows are
severed into **import/export half-flows** by the Agent 3.1 regional
filter, with a virtual ``commodity`` node at each half-flow terminus.
The coordinator prices those half-flows with a shared Lagrange
multiplier per pipeline::

    obj_r(λ) = (region-local costs) + λ_pipe * (exports_r from pipe)
                                    − λ_pipe * (imports_r from pipe)

In equilibrium the export from region *A* equals the import into
region *B* for every pipe; ``λ`` adjusts by a damped sub-gradient step
on the imbalance until the feasibility tolerance is met.

Convergence
-----------
For LP relaxations this scheme has a zero duality gap — the primal
objective sum ``Σ_r obj_r(λ*)`` equals the monolithic optimum once the
imbalance is zero (modulo the region-local cost accounting, which must
not double-count cross-region flows; half-flows carry no variable cost
in the filtered input, so that's automatic).  Non-zero duality gaps
appear only when a region's subproblem is inherently non-convex
(integer variables, UC).  The LH2 fixture is pure LP, so we expect a
gap below ~1 promille at tolerance 1e-3.

Entry point
-----------
:func:`run_lagrangian` — builds per-region work folders, drives the
outer subgradient loop, and returns the converged objective plus a
per-iteration log.
"""
from __future__ import annotations

import csv
import logging
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import highspy

from flextool.flextoolrunner.highs_handle import HighsModelHandle
from flextool.flextoolrunner import region_filter
from flextool.flextoolrunner.flextoolrunner import FlexToolRunner
from flextool.flextoolrunner.runner_state import FlexToolConfigError, FlexToolError


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass
class CouplingVar:
    """One coupling variable shared by two regions via one pipeline.

    *export_region* sends flow along the pipeline; *import_region*
    receives it.  The coupling's λ multiplies the export flow by ``+λ``
    and the import flow by ``−λ`` so that equilibrium
    (``export = import``) corresponds to an imbalance of zero.
    """

    pipeline: str
    export_region: str
    import_region: str
    # HiGHS column indices (populated per region after each solve).
    # ``export_cols[region] = [col_idx, ...]`` lists every ``v_flow``
    # column on the export side in the exporter's regional LP.
    export_cols: dict[str, list[int]] = field(default_factory=dict)
    import_cols: dict[str, list[int]] = field(default_factory=dict)
    # Current price.
    lam: float = 0.0
    # Last observed total primal (MWh over the horizon, scaled by
    # p_entity_unitsize if applicable — we use the raw column values
    # which is sufficient for imbalance-zero comparison because both
    # sides sit on the same process with the same unitsize).
    last_export: float = 0.0
    last_import: float = 0.0

    @property
    def imbalance(self) -> float:
        """Aggregate imbalance (export − import) across the horizon."""
        return self.last_export - self.last_import


@dataclass
class LagrangianResult:
    """Outcome of :func:`run_lagrangian`."""

    converged: bool
    iterations: int
    total_objective: float  # sum of regional primal costs at convergence
    region_objectives: dict[str, float]
    final_lambdas: dict[str, float]  # pipeline → λ
    iteration_log: list[dict] = field(default_factory=list)
    # Name of each region's work folder (for parquet output discovery).
    region_work_folders: dict[str, Path] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Region setup — produce input_region_<r>/ and prepare region work folders
# ---------------------------------------------------------------------------


def _prepare_region_workfolders(
    *,
    db_url: str,
    scenario: str,
    work_folder: Path,
    logger: logging.Logger,
    precision_digits: int = 0,
) -> tuple[list[str], dict[str, Path], dict[str, list[region_filter.HalfFlow]]]:
    """Create a per-region work folder with filtered ``input/`` for each
    decomposition region.

    Returns ``(regions, region_wf, half_flows_by_region)``.  Each
    ``region_wf[r]`` contains a complete input/ directory ready for
    GMPL invocation (the regional filter has been applied).

    This routine does NOT run glpsol — that happens inside the
    coordinator's initial-solve pass per region (see
    :func:`_glpsol_write_mps`).
    """
    work_folder = Path(work_folder)
    work_folder.mkdir(parents=True, exist_ok=True)

    regions = region_filter.discover_decomposition_regions_from_db(db_url)
    if not regions:
        raise FlexToolConfigError(
            "No decomposition regions found in the database.  Set "
            "``group.decomposition_method='lagrangian_region'`` on at "
            "least two groups (and populate group__node / group__connection "
            "membership) before invoking the Lagrangian coordinator."
        )
    if len(regions) < 2:
        raise FlexToolConfigError(
            f"Lagrangian decomposition requires at least two regions; "
            f"found only {regions!r}."
        )

    region_wf: dict[str, Path] = {}
    half_flows_by_region: dict[str, list[region_filter.HalfFlow]] = {}

    for region in regions:
        wf_r = work_folder / f"region_{region}"
        wf_r.mkdir(parents=True, exist_ok=True)
        (wf_r / "solve_data").mkdir(exist_ok=True)

        logger.info("Lagrangian: preparing region work folder %s", wf_r)

        # 1. Write monolithic input/ in the region work folder by running
        #    the normal write_input path.  This also produces the initial
        #    solve_data/ scaffolding.
        from flextool.flextoolrunner import input_writer as _input_writer
        _input_writer.write_input(
            db_url,
            scenario,
            logger,
            work_folder=wf_r,
            precision_digits=precision_digits,
        )

        # 2. Overwrite input/ with the filtered regional copy.
        input_region_dir = wf_r / f"input_region_{region}"
        result = region_filter.build_region_directory(
            input_dir=wf_r / "input",
            output_dir=input_region_dir,
            region=region,
            all_regions=regions,
        )
        # Swap: rename input/ → input_monolithic_backup/, rename
        # input_region_<r>/ → input/.  This keeps glpsol's hard-coded
        # ``input/`` reference working without modifying flextool.mod.
        monolithic_input = wf_r / "input"
        backup_dir = wf_r / "input_monolithic_backup"
        if backup_dir.exists():
            shutil.rmtree(backup_dir)
        monolithic_input.rename(backup_dir)
        input_region_dir.rename(monolithic_input)

        region_wf[region] = wf_r
        half_flows_by_region[region] = result["half_flows"]

    # Write a union coupling manifest into the coordinator's shared
    # work folder so downstream tooling (and follow-up agents) can see
    # every coupling at a glance.
    (work_folder / "solve_data").mkdir(exist_ok=True)
    region_filter.write_region_coupling_manifest(
        work_folder=work_folder,
        results=[
            {"region": r, "half_flows": hfs}
            for r, hfs in half_flows_by_region.items()
        ],
    )

    return regions, region_wf, half_flows_by_region


# ---------------------------------------------------------------------------
# Coupling classification — pair half-flows into shared-λ pipelines
# ---------------------------------------------------------------------------


def _build_couplings(
    half_flows_by_region: dict[str, list[region_filter.HalfFlow]],
) -> list[CouplingVar]:
    """Pair export and import half-flows across regions into couplings.

    Every cross-region pipeline appears as one export half-flow in
    exactly one region and one import half-flow in exactly one other
    region (the Agent 3.1 contract).  We group by ``original_connection``
    name.
    """
    by_pipe: dict[str, dict[str, list[region_filter.HalfFlow]]] = {}
    for region, hfs in half_flows_by_region.items():
        for hf in hfs:
            bucket = by_pipe.setdefault(hf.original_connection, {"export": [], "import": []})
            bucket[hf.side].append(hf)

    couplings: list[CouplingVar] = []
    for pipe, sides in by_pipe.items():
        if not sides["export"] or not sides["import"]:
            # A half-flow without a sibling means the counter-region
            # classifier missed it — should not happen given Agent 3.1
            # contract, but we skip defensively.
            continue
        # Assume one export and one import per pipe (the common case
        # for bilateral pipelines).  If multiple exports/imports exist
        # (e.g. a star topology), each pair is its own coupling — we
        # collapse to a single λ per pipe because that matches the
        # Lagrangian scheme (one dualised balance per pipe).
        export_hf = sides["export"][0]
        import_hf = sides["import"][0]
        couplings.append(CouplingVar(
            pipeline=pipe,
            export_region=export_hf.region,
            import_region=import_hf.region,
        ))
    return couplings


# ---------------------------------------------------------------------------
# Glpsol invocation — write per-region MPS
# ---------------------------------------------------------------------------


def _glpsol_binary(bin_dir: Path) -> str:
    """Return the platform-appropriate glpsol binary path."""
    if sys.platform.startswith("linux"):
        return str(bin_dir / "glpsol")
    if sys.platform.startswith("win32"):
        return str(bin_dir / "glpsol.exe")
    if sys.platform == "darwin":
        return str(bin_dir / "glpsol_macos15_arm64")
    return str(bin_dir / "glpsol")


def _glpsol_write_mps(
    *,
    region_wf: Path,
    flextool_dir: Path,
    bin_dir: Path,
    logger: logging.Logger,
) -> Path:
    """Invoke glpsol ``--check --wfreemps`` to produce ``flextool.mps``
    inside *region_wf*.

    Returns the MPS file path.  Raises :class:`FlexToolError` on glpsol
    failure.

    Note: this routine requires the region's ``solve_data/`` to have
    been populated by a prior orchestration pass.  Our typical flow is
    to run one full :meth:`FlexToolRunner.run_model` per region (which
    calls glpsol + HiGHS + writes solve_data), then re-load the
    produced MPS directly in later iterations.
    """
    glpsol = _glpsol_binary(bin_dir)
    mps_file = region_wf / "flextool.mps"
    mod_file = str(flextool_dir / "flextool.mod")
    dat_file = str(flextool_dir / "flextool_base.dat")

    # glpsol_phase.csv == 'read' — tells the model to do the pre-solve
    # printf pass that writes solve_data/p_flow_max.csv etc.
    (region_wf / "solve_data").mkdir(exist_ok=True)
    (region_wf / "solve_data" / "glpsol_phase.csv").write_text("phase\nread\n")

    cmd = [
        glpsol, "--check", "--model", mod_file,
        "-d", dat_file, "--wfreemps", str(mps_file),
    ]
    logger.info("Lagrangian: glpsol --wfreemps for region work folder %s", region_wf)
    completed = subprocess.run(cmd, cwd=str(region_wf), capture_output=True, text=True)
    if completed.returncode != 0:
        logger.error("glpsol failed for %s: rc=%s\nstdout:\n%s\nstderr:\n%s",
                     region_wf, completed.returncode,
                     completed.stdout[-4000:], completed.stderr[-4000:])
        raise FlexToolError(
            f"glpsol MPS generation failed for region work folder {region_wf}"
        )
    if not mps_file.exists():
        raise FlexToolError(
            f"glpsol returned success but did not produce MPS at {mps_file}"
        )
    return mps_file


# ---------------------------------------------------------------------------
# HighsModelHandle construction + coupling-column resolution
# ---------------------------------------------------------------------------


def _load_mps_into_handle(mps_file: Path) -> HighsModelHandle:
    """Load an MPS file into a fresh HiGHS instance wrapped in a handle."""
    h = highspy.Highs()
    # Turn down HiGHS log chatter; outer loop emits its own summary.
    h.setOptionValue("output_flag", False)
    status = h.readModel(str(mps_file))
    if status != highspy.HighsStatus.kOk:
        raise FlexToolError(f"HiGHS could not read MPS {mps_file}")
    handle = HighsModelHandle(h=h)
    handle.build_name_maps()
    return handle


def _resolve_coupling_columns(
    handle: HighsModelHandle,
    region: str,
    couplings: list[CouplingVar],
    half_flows: list[region_filter.HalfFlow],
) -> None:
    """For each coupling involving *region*, populate ``export_cols`` /
    ``import_cols`` by pattern-matching ``v_flow`` columns in the
    region's HiGHS instance.

    Half-flow v_flow columns are named
    ``v_flow[<virtual_connection>, <src>, <sink>, <period>, <t>]``
    where ``virtual_connection`` is ``<pipe>__<side>__<region>`` (see
    :mod:`region_filter`).  We glob on the first field only.
    """
    by_pipe: dict[str, region_filter.HalfFlow] = {hf.original_connection: hf for hf in half_flows}
    for cpl in couplings:
        hf = by_pipe.get(cpl.pipeline)
        if hf is None:
            continue  # this pipe doesn't touch this region
        pattern = f"v_flow[{hf.virtual_connection},*"
        cols = handle.cols_matching(pattern)
        if not cols:
            # Log-friendly diagnostic: most likely the virtual
            # connection got dropped by the regional filter, or the
            # column naming convention has drifted.
            raise FlexToolError(
                f"Lagrangian: no v_flow columns matched pattern {pattern!r} "
                f"in region {region}.  Check regional filter output and "
                f"confirm flextool.mod emits v_flow[virtual_connection,...]."
            )
        if hf.side == "export":
            cpl.export_cols[region] = cols
        else:
            cpl.import_cols[region] = cols


# ---------------------------------------------------------------------------
# Cost vector update + primal extraction
# ---------------------------------------------------------------------------


def _apply_lambda_costs(
    handle: HighsModelHandle,
    region: str,
    couplings: list[CouplingVar],
) -> None:
    """Set the regional objective coefficients on every coupling column
    to the appropriate ±λ.

    Export columns receive ``+λ`` (exporter pays λ per unit of export);
    import columns receive ``-λ`` (importer receives λ credit per unit
    of import).  All non-coupling columns keep their existing costs.
    """
    for cpl in couplings:
        if region == cpl.export_region and cpl.export_cols.get(region):
            handle.change_costs(cpl.export_cols[region], cpl.lam)
        if region == cpl.import_region and cpl.import_cols.get(region):
            handle.change_costs(cpl.import_cols[region], -cpl.lam)


def _measure_primal(
    handle: HighsModelHandle,
    region: str,
    couplings: list[CouplingVar],
) -> None:
    """After a solve, record ``last_export`` / ``last_import`` sums for
    every coupling this region participates in.
    """
    for cpl in couplings:
        if region == cpl.export_region and cpl.export_cols.get(region):
            vals = handle.primal(cpl.export_cols[region])
            cpl.last_export = float(sum(vals))
        if region == cpl.import_region and cpl.import_cols.get(region):
            vals = handle.primal(cpl.import_cols[region])
            cpl.last_import = float(sum(vals))


# ---------------------------------------------------------------------------
# Primary entry point
# ---------------------------------------------------------------------------


def run_lagrangian(
    db_url: str,
    scenario: str,
    solve: str | None = None,
    *,
    alpha: float = 0.5,
    max_iterations: int = 100,
    tolerance: float = 1e-3,
    work_folder: Path | None = None,
    flextool_dir: Path | None = None,
    bin_dir: Path | None = None,
    logger: logging.Logger | None = None,
    precision_digits: int = 0,
) -> LagrangianResult:
    """Run the spatial-Lagrangian outer loop on *scenario*.

    Parameters
    ----------
    db_url, scenario
        Input DB URL and scenario name — same semantics as
        :class:`FlexToolRunner`.
    solve
        Unused at present (accepted for API parity with the monolithic
        runner; decomposition regions are scenario-level, not
        solve-level in the current fixture).
    alpha
        Sub-gradient step size (damping factor).  Default 0.5 is a
        reasonable starting point for LPs; oscillation on ill-posed
        cases can be tamed by reducing to 0.1–0.25.
    max_iterations
        Hard cap on outer iterations.  Each iteration solves every
        region once and updates λ once.
    tolerance
        Max-absolute imbalance below which the solver is considered
        converged.  Units are the raw primal of the coupling flow
        (unitsize × MWh over the horizon).
    work_folder, flextool_dir, bin_dir
        Path overrides.  ``work_folder`` defaults to CWD (a per-region
        subfolder is created inside it); ``flextool_dir`` / ``bin_dir``
        default to the repo's shipped locations.
    logger
        Python logger; a default is created if omitted.
    precision_digits
        Forwarded to :func:`write_input`.

    Returns
    -------
    :class:`LagrangianResult`
        ``.converged`` indicates whether the final iteration's
        imbalance was below *tolerance*.  ``.total_objective`` sums
        each region's LP objective value at the final iterate — for
        LPs this should equal the monolithic optimum to within the
        duality gap (negligible when imbalance is zero).
    """
    if logger is None:
        logger = logging.getLogger("flextool.lagrangian")
        logger.setLevel(logging.INFO)

    wf = Path(work_folder) if work_folder is not None else Path.cwd()
    fd = (
        Path(flextool_dir) if flextool_dir is not None
        else Path(__file__).resolve().parent.parent
    )
    bd = Path(bin_dir) if bin_dir is not None else fd.parent / "bin"

    # ------------------------------------------------------------------
    # Setup: filter inputs + pre-solve glpsol pass per region.
    # ------------------------------------------------------------------
    regions, region_wf, half_flows_by_region = _prepare_region_workfolders(
        db_url=db_url, scenario=scenario, work_folder=wf, logger=logger,
        precision_digits=precision_digits,
    )
    couplings = _build_couplings(half_flows_by_region)
    if not couplings:
        raise FlexToolConfigError(
            "No cross-region couplings identified — every region's inputs "
            "are fully self-contained.  Lagrangian decomposition is a no-op."
        )
    logger.info(
        "Lagrangian: %d regions, %d cross-region couplings (%s)",
        len(regions), len(couplings), [c.pipeline for c in couplings],
    )

    # ------------------------------------------------------------------
    # Each region needs a populated solve_data/ before glpsol can
    # --wfreemps.  We run a full FlexToolRunner.run_model() per region
    # once with λ=0 — this writes solve_data/, invokes glpsol, and
    # does an initial HiGHS solve (which we discard and recompute in
    # the outer loop with explicit λ management).
    # ------------------------------------------------------------------
    for region in regions:
        logger.info(
            "Lagrangian: running initial FlexTool pass for region %s "
            "(writes solve_data/ + flextool.mps)", region,
        )
        _initial_solve_pass(
            db_url=db_url, scenario=scenario,
            region_wf=region_wf[region], flextool_dir=fd, bin_dir=bd,
            logger=logger,
        )

    # ------------------------------------------------------------------
    # Load per-region MPS into HighsModelHandle; resolve coupling cols.
    # ------------------------------------------------------------------
    handles: dict[str, HighsModelHandle] = {}
    for region in regions:
        mps = region_wf[region] / "flextool.mps"
        if not mps.exists():
            raise FlexToolError(
                f"Expected MPS file at {mps} — initial FlexTool pass "
                f"for region {region} did not produce it."
            )
        handle = _load_mps_into_handle(mps)
        _resolve_coupling_columns(
            handle=handle, region=region, couplings=couplings,
            half_flows=half_flows_by_region[region],
        )
        handles[region] = handle

    # ------------------------------------------------------------------
    # Outer loop.
    # ------------------------------------------------------------------
    iteration_log: list[dict] = []
    converged = False
    iter_obj: dict[str, float] = {r: float("nan") for r in regions}
    for it in range(1, max_iterations + 1):
        # 1. Update each region's objective with current λ.
        for region in regions:
            _apply_lambda_costs(handles[region], region, couplings)

        # 2. Solve each region.
        for region in regions:
            handles[region].solve()
            if not handles[region].is_optimal():
                raise FlexToolError(
                    f"Lagrangian iteration {it}: region {region} did not "
                    f"reach optimal (status={handles[region].h.getModelStatus()})"
                )
            iter_obj[region] = handles[region].objective()
            _measure_primal(handles[region], region, couplings)

        # 3. Imbalance + λ update.
        max_abs_imb = 0.0
        imbalances: dict[str, float] = {}
        for cpl in couplings:
            imb = cpl.imbalance
            imbalances[cpl.pipeline] = imb
            if abs(imb) > max_abs_imb:
                max_abs_imb = abs(imb)

        iteration_log.append({
            "iter": it,
            "lambdas": {c.pipeline: c.lam for c in couplings},
            "imbalances": dict(imbalances),
            "region_objectives": dict(iter_obj),
            "total_objective": sum(iter_obj.values()),
            "max_abs_imbalance": max_abs_imb,
        })
        logger.info(
            "Lagrangian iter %d: max|imb|=%.6g  Σ_obj=%.6g  λ=%s  imb=%s",
            it, max_abs_imb, sum(iter_obj.values()),
            {c.pipeline: round(c.lam, 6) for c in couplings},
            {k: round(v, 6) for k, v in imbalances.items()},
        )

        if max_abs_imb < tolerance:
            converged = True
            break

        # 4. Subgradient update on each coupling.
        for cpl in couplings:
            cpl.lam = cpl.lam + alpha * cpl.imbalance

    total = sum(iter_obj.values())
    return LagrangianResult(
        converged=converged,
        iterations=it,
        total_objective=total,
        region_objectives=dict(iter_obj),
        final_lambdas={c.pipeline: c.lam for c in couplings},
        iteration_log=iteration_log,
        region_work_folders=dict(region_wf),
    )


# ---------------------------------------------------------------------------
# Initial per-region FlexTool pass (writes solve_data + initial MPS).
# ---------------------------------------------------------------------------


def _initial_solve_pass(
    *,
    db_url: str,
    scenario: str,
    region_wf: Path,
    flextool_dir: Path,
    bin_dir: Path,
    logger: logging.Logger,
) -> None:
    """Drive one FlexToolRunner.run_model() inside *region_wf*.

    The runner is constructed with ``work_folder=region_wf`` so that all
    ``input/`` + ``solve_data/`` + ``output_raw/`` paths resolve inside
    the region's folder.  The input/ was already replaced with the
    filtered regional copy in :func:`_prepare_region_workfolders`.

    We chdir temporarily because the runner's downstream helpers use
    cwd-relative paths in some places (legacy glpsol invocation).
    """
    prev_cwd = os.getcwd()
    try:
        os.chdir(str(region_wf))
        runner = FlexToolRunner(
            input_db_url=db_url,
            scenario_name=scenario,
            flextool_dir=flextool_dir,
            bin_dir=bin_dir,
            root_dir=region_wf,
            work_folder=region_wf,
        )
        # DO NOT re-run write_input — the regional-filter swap already
        # gave us the input/ we want.  Running write_input again would
        # overwrite the filtered CSVs with monolithic ones.
        rc = runner.run_model()
        if rc != 0:
            raise FlexToolError(
                f"Initial FlexTool pass for {region_wf} returned rc={rc}"
            )
    finally:
        os.chdir(prev_cwd)


__all__ = [
    "CouplingVar",
    "LagrangianResult",
    "run_lagrangian",
]
