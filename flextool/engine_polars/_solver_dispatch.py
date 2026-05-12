"""Multi-solver dispatch helpers (Phases 2 + 3 of the FlexTool multi-solver port).

This module owns:

* :func:`build_solver_options` — translate :class:`SolverConfig` into
  the option dict polar-high's ``solve()`` consumes (Phase 2).
* :func:`run_one_solve` — dispatch a single :class:`polar_high.Problem`
  to either ``Problem.solve()`` (HiGHS, default — preserves streaming
  + ``Solution.highs``) or :func:`polar_high.solvers.solve` (commercial
  solvers, normalised through
  :class:`flextool.engine_polars._solver_result_to_solution.LiteSolution`).
* :class:`FlexToolUserError` — surface user-facing errors from the
  commercial path with actionable hints.

See ``specs/flextool-multi-solver-handoff.md`` Step 3 for the design
rationale and the canonical _PARAM_MAP table.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from flextool.engine_polars._solve_config import SolverConfig

if TYPE_CHECKING:
    from polar_high import Problem


# ---------------------------------------------------------------------------
# User-facing error type
# ---------------------------------------------------------------------------


class FlexToolUserError(Exception):
    """Raised for user-actionable misconfiguration of the multi-solver
    dispatch.

    Carries a message intended for direct surfacing to the user — installer
    hint, license hint, or "switch the solver" hint.  ``__cause__`` carries
    the underlying polar-high exception for debugging.
    """


# Per-solver native parameter names for the three "convenience" knobs
# normalised by FlexTool.  Anything outside these three goes through
# untranslated via ``SolverConfig.options``.
#
# Source: ``specs/flextool-multi-solver-handoff.md`` lines 109-136.
_PARAM_MAP: dict[str, dict[str, str]] = {
    "highs":  {"time_limit": "time_limit", "mip_gap": "mip_rel_gap",          "threads": "threads"},
    "gurobi": {"time_limit": "TimeLimit",  "mip_gap": "MIPGap",                "threads": "Threads"},
    "cplex":  {"time_limit": "timelimit",  "mip_gap": "mip.tolerances.mipgap", "threads": "threads"},
    "xpress": {"time_limit": "maxtime",    "mip_gap": "miprelstop",            "threads": "threads"},
    "copt":   {"time_limit": "TimeLimit",  "mip_gap": "RelGap",                "threads": "Threads"},
}


def build_solver_options(solver_config: SolverConfig) -> dict[str, Any]:
    """Translate a :class:`SolverConfig` into the raw options dict that
    polar-high's ``solve()`` consumes.

    1. The three convenience knobs (``time_limit`` / ``mip_gap`` /
       ``threads``), when set on *solver_config*, are translated to the
       chosen solver's native parameter name via :data:`_PARAM_MAP`.
       ``None`` values are skipped (no override).
    2. The raw ``solver_config.options`` dict is merged on top of the
       translated knobs — **raw options win** on key collisions.  The
       user knows what they're doing; if they hand-write
       ``solver_options = {"TimeLimit": 30}`` and also set
       ``solver_time_limit = 60``, the raw value (30) wins.

    Parameters
    ----------
    solver_config
        Solve-level config built by
        :meth:`flextool.engine_polars._solve_config.SolveConfig.load_from_db`.

    Returns
    -------
    dict[str, Any]
        Option dict ready to ``**unpack`` into
        :func:`polar_high.solvers.solve`.  Empty dict when no
        convenience knobs are set and ``options`` is empty.

    Raises
    ------
    ValueError
        If ``solver_config.name`` is not in :data:`_PARAM_MAP` AND at
        least one convenience knob is set.  Raw-options-only with an
        unknown solver passes through silently so users can plug a
        future solver via ``solver_options`` before
        :data:`_PARAM_MAP` is updated.
    """
    has_convenience_knob = (
        solver_config.time_limit is not None
        or solver_config.mip_gap is not None
        or solver_config.threads is not None
    )
    mapping = _PARAM_MAP.get(solver_config.name)
    if mapping is None and has_convenience_knob:
        available = ", ".join(sorted(_PARAM_MAP.keys()))
        raise ValueError(
            f"unknown solver {solver_config.name!r}, expected one of: "
            f"{available}"
        )

    opts: dict[str, Any] = {}
    if mapping is not None:
        if solver_config.time_limit is not None:
            opts[mapping["time_limit"]] = solver_config.time_limit
        if solver_config.mip_gap is not None:
            opts[mapping["mip_gap"]] = solver_config.mip_gap
        if solver_config.threads is not None:
            opts[mapping["threads"]] = solver_config.threads

    # Raw options take precedence — see docstring.
    opts.update(solver_config.options)
    return opts


# ---------------------------------------------------------------------------
# Single-solve dispatch (Phase 3)
# ---------------------------------------------------------------------------


def run_one_solve(
    problem: "Problem",
    solver_config: SolverConfig,
    logger: logging.Logger | None = None,
):
    """Dispatch *problem* to the chosen solver.

    The default HiGHS path is byte-identical to the pre-Phase-3 code: a
    direct call to ``problem.solve(keep_solver=True)`` preserves
    streaming, the live ``Solution.highs`` (consumed by the output
    writer adapter), and the established option-resolution chain.

    The commercial path (gurobi / cplex / xpress / copt) routes through
    :func:`polar_high.solvers.solve`, then wraps the
    :class:`polar_high.solvers.SolverResult` into a
    :class:`flextool.engine_polars._solver_result_to_solution.LiteSolution`
    so downstream consumers (``input.py``,
    ``_writer_co2_accumulators.py``, ``process_outputs/read_parameters.py``)
    treat both shapes identically.

    Parameters
    ----------
    problem
        The :class:`polar_high.Problem` to solve.
    solver_config
        Resolved per-solve configuration (defaults to HiGHS/direct when
        no ``solver_*`` parameter is authored on the solve).
    logger
        Optional logger.  When provided, the commercial-path error
        messages are also logged at ERROR level before being raised.

    Returns
    -------
    polar_high.Solution | LiteSolution
        Either the native polar-high Solution (HiGHS path) or a
        LiteSolution wrapping the SolverResult (commercial path).  Both
        expose ``.value()`` / ``._vars`` / ``.obj`` / ``.optimal`` /
        ``.highs``.

    Raises
    ------
    FlexToolUserError
        If the requested solver's Python wrapper is not installed, the
        license check fails, or the solver returns a model-level error.
    """
    if solver_config.name == "highs":
        # Default path: keep ``Problem.solve()`` (preserves streaming +
        # ``Solution.highs`` for the output writer adapter).  Forward
        # ``solver_options`` and the convenience-knob translations so
        # HiGHS-side users get the same surface as commercial users —
        # ``problem.solve(options=...)`` accepts a dict and routes each
        # key to HiGHS via ``setOptionValue`` (polar-high engine.py).
        highs_options = build_solver_options(solver_config) or None
        return problem.solve(options=highs_options, keep_solver=True)

    # Commercial path.  Use polar-high's dispatch + normalise the result.
    from polar_high.solvers import (
        LicenseError,
        SolverError,
        SolverNotAvailableError,
    )
    from polar_high.solvers import solve as polar_solve

    options = build_solver_options(solver_config)

    try:
        result = polar_solve(
            problem,
            solver_name=solver_config.name,
            io_api=solver_config.io_api,
            **options,
        )
    except SolverNotAvailableError as e:
        from polar_high.solvers import available_solvers

        msg = (
            f"Solver {solver_config.name!r} is not installed on this "
            f"system.  Installed solvers: {available_solvers}.  See "
            f"docs/solvers/{solver_config.name}.md for installation "
            f"instructions."
        )
        if logger is not None:
            logger.error(msg)
        raise FlexToolUserError(msg) from e
    except LicenseError as e:
        msg = (
            f"Solver {solver_config.name!r} is installed but its license "
            f"check failed.  Details: {e}.  See "
            f"docs/solvers/{solver_config.name}.md#licensing for help."
        )
        if logger is not None:
            logger.error(msg)
        raise FlexToolUserError(msg) from e
    except SolverError as e:
        msg = (
            f"Solver {solver_config.name!r} returned an error: {e}.  This "
            f"is usually a model issue (numerics, scaling, infeasibility), "
            f"not a solver-install issue."
        )
        if logger is not None:
            logger.error(msg)
        raise FlexToolUserError(msg) from e

    # Normalise SolverResult → LiteSolution.  Local import keeps the
    # ``_solver_dispatch`` import surface narrow on the HiGHS path.
    from flextool.engine_polars._solver_result_to_solution import LiteSolution

    return LiteSolution.from_solver_result(result, problem)


_LICENSE_PROBE_CACHE: dict[str, str] | None = None


def probe_solver_licenses() -> dict[str, str]:
    """Return ``{solver_name: status}`` for every solver in
    ``polar_high.solvers.available_solvers``.

    Status values:

    - ``"licensed"`` — wrapper installed, license check passed, trivial
      solve completed.
    - ``"no-license"`` — wrapper installed but the solver refused on
      license grounds (commercial trial expired, no licence file, etc.).
    - ``"not-installed"`` — Python wrapper isn't on this system.
    - ``"probe-failed"`` — any other exception during the probe; the
      solver may or may not be functional on a real problem.

    The probe runs a 1-variable, 0-constraint LP through
    ``polar_high.solvers.solve(...)`` per solver.  Solver chatter on
    stdout is suppressed.  Result cached at module level so repeat
    cascade runs in the same Python process don't re-probe.

    Used by ``_orchestration.run_chain_from_db`` to print one INFO line
    per cascade, giving users a quick "is gurobi actually working on
    this machine" hint.
    """
    global _LICENSE_PROBE_CACHE
    if _LICENSE_PROBE_CACHE is not None:
        return _LICENSE_PROBE_CACHE
    import io
    import os
    from contextlib import redirect_stdout, redirect_stderr
    import polars as pl
    from polar_high import Problem
    from polar_high.solvers import (
        LicenseError,
        SolverError,
        SolverNotAvailableError,
        available_solvers,
    )
    from polar_high.solvers import solve as polar_solve

    statuses: dict[str, str] = {}
    # HiGHS / Xpress write probe chatter via C-level handles that
    # ``redirect_stdout`` alone can't catch.  Redirect the underlying
    # file descriptors for the duration of each probe so the startup
    # log stays clean.
    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    saved_stdout_fd = os.dup(1)
    saved_stderr_fd = os.dup(2)
    try:
        for solver_name in available_solvers:
            try:
                p = Problem()
                df = pl.DataFrame({"i": [0]})
                v = p.add_var(
                    "x", dims=("i",), index=df, lower=0.0, upper=10.0,
                )
                p.set_objective(v.to_expr())
                os.dup2(devnull_fd, 1)
                os.dup2(devnull_fd, 2)
                with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                    polar_solve(p, solver_name=solver_name)
                statuses[solver_name] = "licensed"
            except LicenseError:
                statuses[solver_name] = "no-license"
            except SolverNotAvailableError:
                statuses[solver_name] = "not-installed"
            except SolverError:
                statuses[solver_name] = "solver-error"
            except Exception:  # noqa: BLE001 — probe should never crash startup
                statuses[solver_name] = "probe-failed"
            finally:
                os.dup2(saved_stdout_fd, 1)
                os.dup2(saved_stderr_fd, 2)
    finally:
        os.close(devnull_fd)
        os.close(saved_stdout_fd)
        os.close(saved_stderr_fd)
    _LICENSE_PROBE_CACHE = statuses
    return statuses


__all__ = [
    "_PARAM_MAP",
    "FlexToolUserError",
    "build_solver_options",
    "probe_solver_licenses",
    "run_one_solve",
]
