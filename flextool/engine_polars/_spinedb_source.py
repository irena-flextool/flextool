"""SpineDB-direct input source for flextool.

P1 of the CSV → DB-direct migration: a :class:`SpineDbSource` that
takes a SpineDB sqlite URL + scenario, runs flextool's preprocessing
pipeline, and exposes the resulting frames through the
:class:`flextool._input_source.FlexInputSource` Protocol.

Implementation notes — Option β (subprocess shim, in-process)
-------------------------------------------------------------
flextool's writers (``input_writer.write_input`` and the per-solve
preprocessing modules) emit CSVs through direct ``f.write(...)`` calls
on file handles, **not** through a DataFrame layer.  There is no
DataFrame boundary to monkey-patch without re-implementing the writers
(Option γ, ~2000 LOC, explicitly out of scope for P1).

The Phase-1 implementation therefore takes the subprocess-shim path
(Option β as documented in the dispatch):

1. Construct a :class:`flextool.flextoolrunner.flextoolrunner.FlexToolRunner`
   over a tempdir work folder.
2. Call :func:`flextool.engine_polars._native_input_writer.write_workdir_inputs`
   (Δ.20 — engine_polars-owned) — populates ``<tempdir>/input/`` and
   the L0-L9 batch ``<tempdir>/solve_data/*.csv``.  Replaces the legacy
   ``runner.write_input(...)`` call from earlier phases.
3. Drive ``orchestration.run_model(...)`` with a no-op solver so the
   per-solve preprocessing (timesets, scaling, period_first, …) writes
   the additional ``solve_data/*.csv`` files flexpy needs — without
   actually invoking glpsol/HiGHS.
4. Expose the populated tempdir's ``input/`` + ``solve_data/`` to
   downstream readers via the Protocol's directory properties.

This faithfully reproduces today's ``load_flextool_from_db`` behaviour
(P1 acceptance: parity with the CSV path on representative fixtures).
The CSV roundtrip-to-disk is preserved as a tempdir; future phases can
swap the implementation for true in-memory capture without touching
the consumer surface.

The actual driving of FlexToolRunner reuses the helpers already proven
in :func:`flextool.input.load_flextool_from_db` (single-solve branch
runs orchestration once with a no-op solver; multi-solve cascades use
flexpy as the inner solver and capture handoffs).  We share that code
path through ``_materialise_workdir`` rather than re-implementing it.
"""
from __future__ import annotations

import logging
import sys
import tempfile
from pathlib import Path

import polars as pl

from ._input_source import _read_csv_file


_REPO_ROOT = Path("/home/jkiviluo/sources/flextool")


def _ensure_flextool_importable() -> None:
    """Make flextool's runtime package (FlexToolRunner, orchestration)
    importable when the live checkout is co-located with flexpy_spike.

    Mirrors the path-shim used by ``load_flextool_from_db`` — we append
    rather than insert so that flexpy's local ``flextool/`` package
    keeps precedence as the importable name.
    """
    if str(_REPO_ROOT) not in sys.path:
        sys.path.append(str(_REPO_ROOT))


class SpineDbSource:
    """Read flextool input data directly from a SpineDB sqlite scenario.

    Parameters
    ----------
    db_url : str | Path
        Spine sqlite URL or path.  A bare path is upgraded to
        ``sqlite:///``.  Both ``Path`` and ``str`` are accepted.
    scenario : str | None
        Scenario filter to apply.  ``None`` picks the first scenario in
        the database (mirrors flextool's behaviour).
    solve : str | None
        Optional solve name filter (reserved for P2 / per-solve
        snapshots).  Not used in P1.
    flextool_dir, bin_dir : Path | str | None
        Override the flextool install location.  Defaults to
        ``/home/jkiviluo/sources/flextool/{flextool,bin}``.
    work_folder : Path | str | None
        Where to materialise the CSVs.  ``None`` (default) uses an
        auto-cleaned tempdir owned by this instance — the dir is
        deleted when the source is garbage collected.

    Lifecycle
    ---------
    Construction is cheap; the flextool pipeline is not run until
    :pyattr:`input_dir` / :pyattr:`solve_data_dir` are first accessed
    (or :meth:`build_frames` is called explicitly).  This matches the
    "lazy" semantics callers expect from a source — useful for tests
    that want to construct many sources up front and only materialise
    the ones they actually load.
    """

    def __init__(
        self,
        db_url: str | Path,
        scenario: str | None = None,
        solve: str | None = None,
        *,
        flextool_dir: Path | str | None = None,
        bin_dir: Path | str | None = None,
        work_folder: Path | str | None = None,
    ):
        url = str(db_url)
        if not url.startswith("sqlite:"):
            url = f"sqlite:///{url}"
        self._db_url = url
        self._scenario = scenario
        self._solve = solve
        self._flextool_dir = (
            Path(flextool_dir) if flextool_dir is not None
            else _REPO_ROOT / "flextool"
        )
        self._bin_dir = (
            Path(bin_dir) if bin_dir is not None
            else _REPO_ROOT / "bin"
        )
        # Tempdir lifecycle: when work_folder is None we own the dir and
        # auto-clean.  When the user supplies one, they own it.
        self._user_work_folder: Path | None
        if work_folder is None:
            self._tempdir = tempfile.TemporaryDirectory(prefix="flexpy_spinedb_")
            self._work_folder = Path(self._tempdir.name)
            self._user_work_folder = None
        else:
            self._tempdir = None
            self._work_folder = Path(work_folder)
            self._work_folder.mkdir(parents=True, exist_ok=True)
            self._user_work_folder = self._work_folder
        self._materialised = False

    # ------------------------------------------------------------------
    # Public attributes

    @property
    def db_url(self) -> str:
        return self._db_url

    @property
    def scenario(self) -> str | None:
        return self._scenario

    @property
    def work_folder(self) -> Path:
        return self._work_folder

    # ------------------------------------------------------------------
    # FlexInputSource protocol implementation

    @property
    def input_dir(self) -> Path:
        self._ensure_materialised()
        return self._work_folder / "input"

    @property
    def solve_data_dir(self) -> Path:
        self._ensure_materialised()
        return self._work_folder / "solve_data"

    def get(self, kind: str, name: str) -> pl.DataFrame | None:
        """Return the named frame.  ``kind`` is ``"input"`` or
        ``"solve_data"``; ``name`` is the CSV stem (with or without
        ``.csv``).  Triggers materialisation on first call."""
        if kind not in ("input", "solve_data"):
            raise ValueError(f"kind must be 'input' or 'solve_data', got {kind!r}")
        d = self.input_dir if kind == "input" else self.solve_data_dir
        fname = name if name.endswith(".csv") else f"{name}.csv"
        path = d / fname
        if not path.exists():
            return None
        return _read_csv_file(path)

    # ------------------------------------------------------------------
    # Materialisation

    def _ensure_materialised(self) -> None:
        if not self._materialised:
            self._materialise()
            self._materialised = True

    def build_frames(self) -> dict[tuple[str, str], pl.DataFrame]:
        """Force materialisation and return every CSV in the work dir as
        a ``{(kind, name): DataFrame}`` mapping (where ``kind`` is
        ``"input"`` or ``"solve_data"`` and ``name`` is the CSV stem).

        Useful for parity testing and inspection.  Note that this loads
        all frames eagerly — for routine use prefer :meth:`get` (lazy)
        or pass the source to :func:`flextool.load_flextool` (which
        only reads the frames it needs).
        """
        self._ensure_materialised()
        out: dict[tuple[str, str], pl.DataFrame] = {}
        for kind, d in (("input", self.input_dir),
                        ("solve_data", self.solve_data_dir)):
            if not d.exists():
                continue
            for f in sorted(d.glob("*.csv")):
                stem = f.stem
                try:
                    out[(kind, stem)] = _read_csv_file(f)
                except Exception:  # noqa: BLE001 — empty / malformed CSVs survive as missing
                    # Some flextool outputs are header-only and
                    # polars.read_csv accepts those; truly malformed
                    # files are extremely rare and would surface during
                    # downstream loading.  Skip-and-continue keeps
                    # build_frames usable for diagnostics.
                    continue
        return out

    # ------------------------------------------------------------------
    # Internal: drive flextool to populate the work folder

    def _materialise(self) -> None:
        """Run flextool's preprocessing pipeline into ``self._work_folder``.

        Single-solve scenarios: ``write_input`` then ``run_model`` with
        a no-op solver — populates ``input/`` plus the per-solve
        ``solve_data/`` files needed by flexpy's reader without
        invoking glpsol/HiGHS.

        Multi-solve cascades: same write_input then drive
        ``orchestration.run_model`` with a flexpy-as-inner-solver
        wrapper.  Each non-final solve is solved by flexpy on the
        snapshot, and a ``SolveHandoff`` is captured into
        ``runner.state.handoffs`` so the next iteration's
        preprocessing picks it up.  The final solve's preprocessing
        runs but the solve itself is skipped.

        This logic is identical to ``flextool.input.load_flextool_from_db``;
        we share it here rather than calling that function directly so
        SpineDbSource keeps its lifecycle ownership of the work folder
        (and so we can extend the materialisation step in P2 without
        touching the public ``load_flextool_from_db`` signature).
        """
        _ensure_flextool_importable()
        # Late imports — defer pulling flextool's runtime until needed.
        from flextool.flextoolrunner.flextoolrunner import FlexToolRunner
        from flextool.flextoolrunner import orchestration
        from flextool.flextoolrunner.solver_runner import SolverRunner

        runner = FlexToolRunner(
            input_db_url=self._db_url,
            scenario_name=self._scenario,
            flextool_dir=self._flextool_dir,
            bin_dir=self._bin_dir,
            work_folder=self._work_folder,
        )
        # Δ.20 — workdir CSV population is owned by engine_polars.  The
        # cascade no longer reaches into FlexToolRunner.write_input;
        # the native shim emits the same artefacts.
        from flextool.engine_polars._native_input_writer import (
            write_workdir_inputs,
        )

        write_workdir_inputs(
            self._db_url, self._scenario, self._work_folder,
            logger=runner.state.logger,
        )
        runner.state.logger.setLevel(logging.ERROR)

        solves = next(iter(runner.state.solve.model_solve.values()))
        total_solves = len(solves)

        if total_solves <= 1:
            class _NoOpSolver(SolverRunner):
                def run(self, complete_solve_name: str) -> int:  # noqa: ARG002
                    return 0
            orchestration.run_model(runner.state, _NoOpSolver(runner.state))
            return

        # Multi-solve cascade — drive flextool's loop with a wrapper
        # solver that runs flexpy on each non-final iteration and
        # captures the handoff.  Imports are local so the single-solve
        # path doesn't pay for them.
        from flextool.engine_polars.input import load_flextool, build_handoff_from_flexpy

        runner.state.handoffs = {}

        class _FlexpyCascadeSolver(SolverRunner):
            def __init__(self, runner_state, total_solves: int):
                super().__init__(runner_state)
                self._total = total_solves
                self._count = 0

            def run(self, complete_solve_name: str) -> int:
                self._count += 1
                if self._count == self._total:
                    return 0  # caller solves the last one
                # Local imports to avoid build-time cycles.
                from polar_high import Problem
                from flextool.engine_polars.model import build_flextool as _build
                from flextool.engine_polars._solver_dispatch import (
                    run_one_solve,
                )
                from flextool.engine_polars._solve_config import (
                    SolverConfig as _SolverConfig,
                )
                data = load_flextool(self.state.paths.work_folder)
                pb = Problem()
                _build(pb, data)
                # Phase 3 — route through ``run_one_solve``.  This is the
                # SpineDbSource fixture-test cascade path; HiGHS is the
                # default and ``run_one_solve`` short-circuits to
                # ``pb.solve(keep_solver=True)`` for that case (note: the
                # earlier call here used bare ``pb.solve()`` without
                # ``keep_solver``; ``run_one_solve`` always passes
                # ``keep_solver=True`` which is harmless on this path).
                solver_cfg = self.state.solve.solver_configs.get(
                    complete_solve_name, _SolverConfig()
                )
                sol = run_one_solve(pb, solver_cfg, logger=self.state.logger)
                if not sol.optimal:
                    self.state.logger.error(
                        f"flexpy non-optimal for {complete_solve_name}"
                    )
                    return 1
                prior = (
                    self.state.handoffs.get(self.state.last_captured_solve)
                    if self.state.last_captured_solve is not None else None
                )
                # Phase 4 (Gap F) — pass the in-memory FlexData + parent
                # handoff so the extractors skip the workdir CSV reads
                # where the same data is already in scope.
                parent_complete = getattr(
                    self.state, "current_parent_complete", None
                )
                parent_handoff = (
                    self.state.handoffs.get(parent_complete)
                    if parent_complete is not None else None
                )
                handoff = build_handoff_from_flexpy(
                    sol, self.state.paths.work_folder,
                    complete_solve_name, prior_handoff=prior,
                    flex_data=data,
                    parent_handoff=parent_handoff,
                )
                self.state.handoffs[complete_solve_name] = handoff
                return 0

        orchestration.run_model(
            runner.state, _FlexpyCascadeSolver(runner.state, total_solves),
        )

    # ------------------------------------------------------------------
    # Repr

    def __repr__(self) -> str:
        return (
            f"SpineDbSource(db_url={self._db_url!r}, "
            f"scenario={self._scenario!r}, work_folder={self._work_folder!s})"
        )
