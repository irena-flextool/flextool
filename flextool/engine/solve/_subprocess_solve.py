"""Subprocess solver driver — the only cold-solve path in FlexTool.

After the in-process cold-solve retirement, this module is the *single*
entry point for any solve that isn't going through the warm-active
HiGHS path:

* HiGHS with ``--save-memory`` (or with ``warm=False`` and the
  soft-promote rule fired in ``_orchestration.py``) — written via
  :meth:`polar_high.Problem.write_mps`, solved by
  :mod:`flextool.cli.cmd_solve_mps`.
* Non-HiGHS solvers (``gurobi`` / ``cplex`` / ``xpress`` / ``copt``) —
  written via the same :meth:`Problem.write_mps`, solved by the
  solver's CLI binary discovered on ``$PATH`` (with a small set of
  conventional install dirs as fallback).

The default HiGHS path in ``_orchestration.py`` (warm-active
:class:`polar_high.WarmProblem`) does **not** call this module.

**Non-HiGHS contract**: commercial solvers (Gurobi/CPLEX/Xpress/COPT)
*always* go through subprocess regardless of ``--save-memory``.  This
keeps every cold solve at the bound peak-memory footprint
:meth:`Problem.write_mps` was designed to deliver (~2-3 GB on a 9.9 M
row LP vs ~45 GB for HiGHS' own ``writeModel``).  The corollary: there
is no in-process Gurobi/CPLEX/Xpress/COPT dispatch in FlexTool at all.

The child process has a clean address space — none of FlexTool's
~7-11 GB of polars frames, no glibc fragmentation from upstream
preprocessing.  When it finishes, the parent reads the solution back:

* HiGHS path writes the MPS with GENERIC ``C…``/``R…`` names too and
  parses the subprocess-written .sol directly (:func:`_parse_highs_sol`):
  each primal / dual value goes to the col_id / row id its generic name
  encodes, every name and count is checked against the LP (a mismatch
  raises :class:`SubprocessSolutionMismatch`), and the real names come
  from the Problem (row names taken before the release, column names
  from the surviving ``_vars``).
* Commercial path writes the MPS with GENERIC ``C…``/``R…`` names
  (``emit_names=False``) so real entity names — which routinely contain
  spaces, illegal in whitespace-delimited free-format MPS — never
  corrupt the file.  It parses the solver-native .sol with the parsers
  imported from polar-high's ``_mps_fallback`` (copied verbatim so the
  parsers stay free of the ``LpView`` materialisation polar-high uses
  upstream of them), then maps the generic-named primal/dual back by
  *index* onto the real names rebuilt from the Problem's surviving
  ``_vars`` (:func:`_col_names_from_vars`) and pre-release ``_cstrs``
  (:func:`_row_names_from_cstrs`).

Neither path re-reads the MPS through :meth:`highspy.Highs.readModel`
in the parent: that used to spike tens of GB of RSS on large LPs, and
hard-failed on FlexTool's own free-format MPS whenever an entity name
contained a space.  Instead both wrap the parsed arrays in a
:class:`_SolHighsShim`, giving downstream output writers a uniform
``Solution.highs`` shape — they keep doing
``h.allVariableNames() + h.getSolution()`` regardless of which solver
produced the result.

Loses warm-LP reuse for the cascade — ``write_mps(release=True)``
puts the Problem in ``_released`` state and it can't be resolved.
Already documented on the ``--save-memory`` flag.
"""
from __future__ import annotations

import json
import logging
import math
import mmap
import os
import queue
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable, Mapping, NamedTuple, Sequence

import numpy as np

from flextool._child_python import child_env, python_module_argv
from flextool._solver_option_rules import native_opt_lines as _native_opt_lines
from flextool.engine.solve._acceptance import LIMIT_SHIM_STATUSES

if TYPE_CHECKING:
    from polar_high import Problem, Solution

    from flextool.run_options import RunOptions


# Warm-start first-transfer A/B margin.  A supplied basis disables HiGHS'
# presolve, so a *far* / stale seed can make the warm solve SLOWER than a
# cold (presolve-on) solve (Landmine B).  On a family's first warm
# transfer we time a cold reference run and only keep injecting the basis
# when the warm run's wall time is within ``(1 + _WARM_AB_MARGIN)`` of the
# cold time; otherwise the fingerprint is marked ``.nowarm`` and future
# solves go cold.  Env-tunable; a NEGATIVE value forces a regression
# verdict (used by tests to exercise the disable path deterministically).
_WARM_AB_MARGIN = float(os.environ.get("FLEXTOOL_WARM_AB_MARGIN", "0.10"))


def _warm_ab_margin() -> float:
    """Effective A/B margin, re-read from the environment at decision time.

    Falls back to the import-time :data:`_WARM_AB_MARGIN` default when the
    env var is unset or unparseable.  Read at runtime (not once at import)
    so ``FLEXTOOL_WARM_AB_MARGIN`` stays tunable within a live process —
    the gate fires at most once per fingerprint, so the cost is trivial.
    """
    try:
        return float(
            os.environ.get("FLEXTOOL_WARM_AB_MARGIN", str(_WARM_AB_MARGIN))
        )
    except (TypeError, ValueError):
        return _WARM_AB_MARGIN


def _touch_marker(path: Path, logger: logging.Logger | None) -> None:
    """Best-effort create an empty warm-start decision marker file.

    Markers (``<fp>.nowarm`` / ``<fp>.abtested``) live in the persistent
    basis cache dir and record the first-transfer A/B verdict.  A write
    failure is logged and non-fatal — the gate degrades to re-probing on
    the next solve, never breaking the solve itself.
    """
    try:
        path.write_text("")
    except OSError as exc:
        if logger is not None:
            logger.warning(
                "save_memory: warm-start marker write failed for %s (%s)",
                path, exc,
            )


def _run_cold_probe(
    *,
    mps_path: Path,
    options_json: Path,
    probe_sol: Path,
    probe_stats: Path,
    logger: logging.Logger | None,
    threads: int | None = None,
) -> tuple[float | None, bool]:
    """Spawn a throwaway COLD child solve purely to measure its wall time.

    Used by the first-transfer A/B gate (see :data:`_WARM_AB_MARGIN`).
    The child runs with the SAME ``--mps`` / ``--options-json`` but a throwaway
    ``--solution`` and ``--stats`` and NO ``--warm-basis`` — a pure cold
    reference solve.  Its solution is discarded; only ``run_time`` matters.

    Returns ``(run_time, ended)``: the cold ``run_time`` in seconds, or
    ``None`` when the probe could not run or produced no usable time; and
    whether the probe child ran to its end.  A probe that could not be
    started (including a stale result file that cannot be removed), was
    stopped by a signal, ended with any exit code other than 0 or 1, or
    ended with exit code 1 without writing its stats (an exception in the
    child, or a Windows ``taskkill /F``) is not a measurement: the caller
    records no verdict, so the family is probed again.  The probe's own files are removed first, so an earlier run's
    stats in the persistent work folder are never read as this probe's.
    Fully defensive — never raises.
    """
    try:
        _remove_old_results([probe_sol, probe_stats])
        probe_cmd = [
            *python_module_argv("flextool.cli.cmd_solve_mps"),
            "--mps", str(mps_path),
            "--solution", str(probe_sol),
            "--options-json", str(options_json),
            "--stats", str(probe_stats),
        ]
        if threads is not None:
            probe_cmd += ["--threads", str(threads)]
        cp = subprocess.run(probe_cmd, env=child_env())
        if cp.returncode not in (0, 1):
            if logger is not None:
                logger.warning(
                    "save_memory: the cold A/B probe %s",
                    _describe_child_exit(cp.returncode),
                )
            return None, False
        if not probe_stats.exists():
            # ``cmd_solve_mps`` writes the stats at the end of every normal run (exit 0 or 1).  Exit 1 without
            # them is an uncaught exception in the child or, on Windows, a ``taskkill /F`` / Task Manager
            # stop: not a measurement, so no verdict.
            if cp.returncode == 1:
                if logger is not None:
                    logger.warning(
                        "save_memory: the cold A/B probe ended without writing its stats "
                        "(it crashed or was stopped; exit code 1)"
                    )
                return None, False
            return None, True
        pstats = json.loads(probe_stats.read_text())
        rt = pstats.get("run_time")
        if rt is None:
            return None, True
        return float(rt), True
    except Exception as exc:  # noqa: BLE001 - probe is measurement-only
        if logger is not None:
            logger.warning("save_memory: the cold A/B probe could not run (%s)", exc)
        return None, False


def _remove_old_results(paths: Iterable[Path | None]) -> None:
    """Remove result files an earlier run may have left, just before a solver starts.

    The work folder (``<work>/solve_data/subprocess``) persists across runs and
    same-named solves.  A solver that ends without writing its result (killed,
    crashed, infeasible with no file, a wrapper that exits 0) must not have an
    earlier run's file read as this solve's.  A file that cannot be removed
    stops the solve: reading it afterwards would be exactly that mistake.
    """
    for path in paths:
        if path is None:
            continue
        try:
            path.unlink(missing_ok=True)
        except OSError as exc:
            raise RuntimeError(
                f"could not remove the result file {path} left by an earlier "
                f"run ({exc}); remove it, or close the program holding it, "
                f"and start the run again"
            ) from exc


def _describe_child_exit(returncode: int) -> str:
    """How a child process ended, for an error message: the signal that stopped it
    (POSIX: a negative exit code), a Windows status code in hex, else the exit code."""
    if returncode < 0:
        try:
            name = signal.Signals(-returncode).name
        except ValueError:
            name = f"signal {-returncode}"
        hint = " (often the out-of-memory killer)" if name == "SIGKILL" else ""
        return f"was stopped by {name}{hint}"
    if returncode > 0xFFFF:
        # A Windows status such as 0xC0000005 (access violation) or 0xC0000409 (abort).  Windows has no
        # out-of-memory killer: a child out of memory ends with 0xC0000017 (STATUS_NO_MEMORY) or exit 1.
        hint = " (out of memory)" if returncode == 0xC0000017 else ""
        return f"failed with exit code 0x{returncode:08X}{hint}"
    return f"failed with exit code {returncode}"


# ---------------------------------------------------------------------------
# HiGHS .opt formatting / .sol parsing helpers (existing)
# ---------------------------------------------------------------------------


def _format_opt_value(v: object) -> str:
    """Render *v* in HiGHS .opt file syntax (``key=value`` per line)."""
    if isinstance(v, bool):
        return "true" if v else "false"
    return str(v)


def _highs_option_pairs(options: Mapping[str, Any] | None) -> list[tuple[str, str]]:
    """The ``--save-memory`` HiGHS options as ``(name, value text)`` pairs in order (:func:`_format_opt_value`)."""
    return [(str(k), _format_opt_value(v)) for k, v in (options or {}).items()]


def _write_highs_option_files(pairs: list[tuple[str, str]], opts_path: Path, json_path: Path) -> None:
    """Write the HiGHS option *pairs* for the ``--save-memory`` child: ``<solve>.options.json`` (``[[name, text],
    ...]``, what the child applies, ``--options-json``) and ``<solve>.opt`` (``key=value`` lines: the record, and
    ``--options`` for a replay by hand)."""
    lines = [f"{k}={v}" for k, v in pairs]
    _check_command_lines("highs", "option-file line", lines, len(pairs))
    with open(opts_path, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(f"{line}\n")
    json_path.write_text(json.dumps([[k, v] for k, v in pairs]), encoding="utf-8")


def _set_objective_constant(sol: "Solution", obj_offset: float) -> None:
    """Record on *sol* the objective constant its ``obj`` includes and its
    ``highs`` shim's ``getObjectiveValue()`` leaves out (the MPS file has
    none), as polar-high's own ``Solution.obj_offset`` does."""
    sol.obj_offset = float(obj_offset)


def _read_objective_from_sol(sol_path: Path) -> float:
    """Parse the ``Objective <value>`` line from a HiGHS style=0 sol file.

    ``highspy.Highs.getObjectiveValue()`` only reflects the most recent
    ``run()`` and is zero after a bare ``readModel + readSolution``, so
    we read the value the subprocess wrote directly.  Returns ``0.0``
    when the line is absent (caller should treat as non-optimal).
    """
    try:
        with open(sol_path) as f:
            for line in f:
                if line.startswith("Objective "):
                    return float(line.split(None, 1)[1])
    except (OSError, ValueError):
        pass
    return 0.0


# ---------------------------------------------------------------------------
# HiGHS .sol direct parser + lightweight Highs-shim
# ---------------------------------------------------------------------------
#
# The cold (``--save-memory``) path used to re-load the *entire* MPS via
# ``highspy.Highs.readModel`` in the parent process — purely to satisfy
# downstream writers that call ``h.allVariableNames()`` / ``h.getSolution()``
# / ``h.getLp().row_names_``.  On large LPs (DES: ~10 M rows, 7 M cols)
# that ``readModel`` accounts for the +33 GB RSS spike at the per-solve
# ``Solver`` checkpoint — and immediately gets thrown away after the
# writers finish.  We sidestep it: parse the HiGHS style=0 .sol file
# directly (it carries primal + dual values under the generic ``C…`` /
# ``R…`` names of the MPS), place them by generic-name index into dense
# arrays, take the real names from the Problem, and wrap the arrays in a
# duck-typed shim that exposes exactly the API surface the writers use.
# No 33 GB sidecar Highs instance.

def _col_names_from_vars(problem: "Problem") -> list[str]:
    """Rebuild the dense, ``col_id``-indexed column-name list from a
    (possibly released) :class:`polar_high.Problem`.

    ``Problem.write_mps(release=True)`` keeps ``self._vars`` alive
    precisely so an external solver's solution can be mapped back to
    user-space names.  This helper reproduces the *exact* naming
    polar-high itself emits for ``Solution.col_names`` —
    ``"<family>[<d0>,<d1>,…]"`` for dimensioned variables, the bare
    family name for scalar ones — from those surviving
    ``Var.frame['col_id']`` columns (mirrors the canonical-matrix name
    pass in ``polar_high.engine.Problem._canonicalise``).

    Preferred over re-reading the MPS through
    ``highspy.Highs.readModel`` on the commercial-solver path: the result
    is **complete** (columns the MPS COLUMNS section elides because they
    carry no objective / matrix entry are still present here),
    ``col_id``-aligned by construction, and immune to the free-format-MPS
    whitespace fragility a text round-trip suffers when an entity name
    contains a space.
    """
    import polars as pl

    vars_map = getattr(problem, "_vars", {}) or {}
    # Every column is created by exactly one ``add_var`` and lands in that
    # variable's frame, so the max ``col_id`` across all frames + 1 is the
    # exact dense column count — no dependence on the internal counter
    # (which differs between ``Problem`` and ``WarmProblem``).
    max_cid = -1
    for v in vars_map.values():
        ids = v.frame["col_id"].to_numpy()
        if ids.size:
            max_cid = max(max_cid, int(ids.max()))
    n_cols = max_cid + 1

    col_names: list[str] = [""] * n_cols
    for v in vars_map.values():
        ids = v.frame["col_id"].to_numpy()
        if v.dims:
            tagged = v.frame.select(
                pl.format(
                    "{}[{}]",
                    pl.lit(v.name),
                    pl.concat_str(
                        [pl.col(d).cast(pl.String) for d in v.dims],
                        separator=",",
                    ),
                ).alias("__name")
            )["__name"].to_list()
            for cid, nm in zip(ids.tolist(), tagged):
                col_names[cid] = nm
        elif ids.size:
            col_names[int(ids[0])] = v.name
    return col_names


def _row_names_from_cstrs(problem: "Problem") -> list[str]:
    """Rebuild the ``row_id``-ordered constraint-name list from a Problem's
    ``_cstrs`` families — the row analogue of :func:`_col_names_from_vars`.

    Same ``"<family>[<d0>,<d1>,…]"`` format polar-high emits for
    ``Solution.row_names`` (bare family name for a scalar constraint),
    walking ``_cstrs`` in declaration order — which is the exact order the
    canonical matrix assigns row ids (mirrors
    ``polar_high.engine.Problem._canonicalise``'s row-name pass).

    Unlike ``_vars``, ``_cstrs`` is **dropped** by
    ``write_mps(release=True)``, so callers that need the real constraint
    names (to map a solver's ``.sol`` duals back by name) must call this
    *before* releasing.  Only the dual-returning commercial parsers
    (:data:`_DUAL_CAPABLE_SOLVERS`) need it; Gurobi / COPT / Xpress ``.sol``
    files carry no duals, so their path skips this entirely.
    """
    import polars as pl

    names: list[str] = []
    for cname, _proto, over in getattr(problem, "_cstrs", []) or []:
        if over is None:
            names.append(cname)
            continue
        axis_cols = list(over.columns)
        names.extend(
            over.select(
                pl.format(
                    "{}[{}]",
                    pl.lit(cname),
                    pl.concat_str(
                        [pl.col(d).cast(pl.String) for d in axis_cols],
                        separator=",",
                    ),
                ).alias("__rn")
            )["__rn"].to_list()
        )
    return names


# polar-high's ``write_mps(emit_names=False)`` emits generic, whitespace-safe
# column / row names in strict id order: column ``col_id=j`` → ``f"C{j+1:07d}"``
# and constraint ``row_id=i`` → ``f"R{i+2:07d}"`` (the objective row is the
# reserved name ``cost`` and never appears in a solver ``.sol``'s variable or
# dual list).  These regexes recover the id from such a name; the ``0*`` +
# open ``[0-9]+`` tolerates the field widening past 7 digits on >10 M-col LPs.
_GENERIC_COL_RE = re.compile(r"^C0*([0-9]+)$")
_GENERIC_ROW_RE = re.compile(r"^R0*([0-9]+)$")


def _generic_col_id(name: str) -> int | None:
    """``"C0000001"`` → ``0``; ``None`` when *name* isn't a generic col id."""
    m = _GENERIC_COL_RE.match(name)
    return int(m.group(1)) - 1 if m else None


def _generic_row_id(name: str) -> int | None:
    """``"R0000002"`` → ``0``; ``None`` when *name* isn't a generic row id."""
    m = _GENERIC_ROW_RE.match(name)
    return int(m.group(1)) - 2 if m else None


# Commercial ``.sol`` parsers that can return constraint duals.  Only these
# pay the cost of rebuilding real row names before ``write_mps`` releases;
# Gurobi / COPT / Xpress ResultFiles carry primal values only.
_DUAL_CAPABLE_SOLVERS = frozenset({"cplex"})


class SubprocessSolutionMismatch(RuntimeError):
    """The solver's ``.sol`` does not line up with the LP that was written.

    Raised when a HiGHS ``.sol`` read back on the subprocess path names a
    column or row that is not the generic name expected at that position,
    or has a different number of columns / rows than the LP.  Mapping such
    a file by position would put solution values on the wrong variables,
    so the solve stops instead.
    """


def _parse_highs_sol(
    sol_path: Path,
    n_cols: int,
    n_rows: int,
    *,
    require_primal: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Parse a HiGHS style=0 ``.sol`` of a generic-name MPS into dense arrays.

    The MPS was written by ``Problem.write_mps(emit_names=False)``, so
    column ``col_id=j`` is ``C{j+1:07d}`` and constraint row ``i`` is
    ``R{i+2:07d}``.  Returns ``(col_value, col_dual, row_dual)`` indexed by
    ``col_id`` / constraint row id — the same dense layout an in-process
    solve gives.  The real names are not in the file; the caller takes them
    from the Problem.

    Every name in the file is checked against the generic name expected at
    its position, and each section must hold exactly *n_cols* columns /
    *n_rows* rows.  Any difference raises :class:`SubprocessSolutionMismatch`
    (a column the reader dropped or moved would otherwise shift every later
    value onto the wrong variable).  A section HiGHS writes as ``None`` (no
    primal / dual solution, e.g. an infeasible or interrupted solve) leaves
    zeros; with *require_primal* (the child reported an optimal solve) a
    missing primal section raises instead.

    The HiGHS style=0 file format (see ``Highs::writeSolution``)::

        Model status
        <status>

        # Primal solution values
        Feasible | Infeasible | None
        Objective <value>
        # Columns <N>
        <name> <value>
        ...
        # Rows <M>
        <name> <value>
        ...

        # Dual solution values
        Feasible | Infeasible | None
        # Columns <N>
        <name> <dual>
        ...
        # Rows <M>
        <name> <dual>
        ...

        # Basis ...   (ignored)
    """
    col_value = np.zeros(n_cols, dtype=np.float64)
    col_dual = np.zeros(n_cols, dtype=np.float64)
    row_dual = np.zeros(n_rows, dtype=np.float64)
    row_value = np.zeros(n_rows, dtype=np.float64)  # parsed, checked, discarded
    targets = {
        ("primal", "col"): col_value,
        ("primal", "row"): row_value,
        ("dual", "col"): col_dual,
        ("dual", "row"): row_dual,
    }

    def _mismatch(detail: str) -> SubprocessSolutionMismatch:
        return SubprocessSolutionMismatch(
            f"HiGHS solution file {sol_path} does not match the LP written "
            f"for it ({n_cols} columns, {n_rows} rows): {detail}. The values "
            "cannot be mapped back to the model's variables safely, so the "
            "solve is stopped."
        )

    section: str | None = None
    bucket: str | None = None
    arr: np.ndarray | None = None
    pos = 0
    counts: dict[tuple[str, str], int] = {}

    def _close_bucket() -> None:
        if section is not None and bucket is not None:
            want = n_cols if bucket == "col" else n_rows
            if pos != want:
                raise _mismatch(
                    f"the {section} {bucket} section has {pos} entries, "
                    f"expected {want}"
                )
            counts[(section, bucket)] = pos

    with open(sol_path) as fh:
        for raw in fh:
            line = raw.strip()
            if not line:
                continue
            if line.startswith("#"):
                if line.startswith(("# Primal solution", "# Dual solution", "# Basis")):
                    _close_bucket()
                    bucket, arr = None, None
                    if line.startswith("# Basis"):
                        section = None
                        break
                    section = "primal" if line.startswith("# Primal") else "dual"
                    continue
                if line.startswith(("# Columns", "# Rows")) and section is not None:
                    _close_bucket()
                    bucket = "col" if line.startswith("# Columns") else "row"
                    pos = 0
                    arr = targets[(section, bucket)]
                    want = n_cols if bucket == "col" else n_rows
                    try:
                        declared = int(line.split()[2])
                    except (IndexError, ValueError):
                        declared = None
                    if declared != want:
                        raise _mismatch(
                            f"the {section} {bucket} section declares "
                            f"{line.split(None, 2)[-1]!r} entries, expected {want}"
                        )
                    continue
                continue
            if arr is None:
                # Status / ``Objective`` / ``Feasible`` / ``None`` lines.
                continue
            sp = line.split()
            if len(sp) != 2:
                raise _mismatch(f"unreadable line {line!r}")
            name, val_s = sp
            if bucket == "col":
                expected = f"C{pos + 1:07d}"
            else:
                expected = f"R{pos + 2:07d}"
            if name != expected:
                raise _mismatch(
                    f"{section} {bucket} entry {pos} is named {name!r}, "
                    f"expected {expected!r}"
                )
            if pos >= arr.shape[0]:
                raise _mismatch(f"more {section} {bucket} entries than expected")
            try:
                arr[pos] = float(val_s)
            except ValueError as exc:
                raise _mismatch(f"unreadable value in line {line!r}") from exc
            pos += 1
    _close_bucket()
    # HiGHS writes a section's column and row values together; one without
    # the other means a truncated file.
    for sec in ("primal", "dual"):
        has_col = (sec, "col") in counts
        has_row = (sec, "row") in counts
        if has_col != has_row:
            raise _mismatch(
                f"the {sec} section has only its "
                f"{'column' if has_col else 'row'} values"
            )
    if require_primal and ("primal", "col") not in counts:
        raise _mismatch("the solve reported optimal but the file has no primal values")
    return col_value, col_dual, row_dual


#: The model-status line of a HiGHS style=0 solution file (HiGHS 1.15) for a solve stopped by a limit or at an
#: objective target or bound, as FlexTool's status names.  ``Solution limit reached`` is HiGHS's one status for
#: its node, leaf, stall-node and improving-solution limits.
_HIGHS_SOL_STATUSES = {
    "Time limit reached": "TIME_LIMIT",
    "Solution limit reached": "SOLUTION_LIMIT",
    "Iteration limit reached": "ITERATION_LIMIT",
    "Target for objective reached": "OBJECTIVE_TARGET",
    "Bound on objective reached": "OBJECTIVE_BOUND",
    "Memory limit reached": "MEM_LIMIT",
}


def _highs_sol_status(sol_path: Path) -> tuple[str, bool]:
    """How a non-optimal HiGHS child solve ended, from its style=0 solution file: the status
    (:data:`_HIGHS_SOL_STATUSES`, else ``OTHER``) and whether the file's primal is ``Feasible``.  Reads the
    file's first lines only."""
    try:
        with open(sol_path) as fh:
            head = [fh.readline().strip() for _ in range(6)]
    except OSError:
        return "OTHER", False
    model_status = head[1] if len(head) > 1 and head[0] == "Model status" else ""
    primal_status = next(
        (head[i + 1] for i, line in enumerate(head[:-1]) if line == "# Primal solution values"), "",
    )
    return _HIGHS_SOL_STATUSES.get(model_status, "OTHER"), primal_status == "Feasible"


def _read_highs_mip_gap(gap_path: Path) -> float | None:
    """The relative MIP gap the HiGHS child wrote (``--mip-gap``), else ``None`` when the file is missing or
    unreadable.  A non-finite value (an LP, or no incumbent) is returned as is; the acceptance check treats it
    as "no gap"."""
    try:
        return float(gap_path.read_text().strip())
    except (OSError, ValueError):
        return None


class _SolHighsShim:
    """Duck-typed stand-in for ``highspy.Highs`` for the cold-path writers.

    The flextool writers under ``process_outputs/`` consume the solver
    instance via a small, stable surface:

    * ``allVariableNames()`` — list[str] of column names.
    * ``getSolution()`` — object with ``col_value`` / ``col_dual``
      / ``row_dual`` attributes (numpy arrays).
    * ``getLp().row_names_`` — list[str] of constraint names.
    * ``passColName(cid, name)`` — in-place rename of a column.

    This shim wraps the arrays parsed from the subprocess ``.sol`` file
    and exposes exactly those four entry points.  Sidesteps the parent-
    side ``highspy.Highs.readModel`` whose +33 GB RSS bump used to spike
    at the per-solve ``Solver`` checkpoint on large LPs.

    Memory cost: O(n_cols + n_rows) Python strings + the four numpy
    arrays — typically a few hundred MB on a 10 M-cell LP vs the tens
    of GB the full Highs sidecar took.
    """

    __slots__ = ("_col_names", "_row_names", "_solution", "_lp", "_obj")

    class _SolutionView:
        __slots__ = ("col_value", "col_dual", "row_dual")

        def __init__(
            self,
            col_value: np.ndarray,
            col_dual: np.ndarray,
            row_dual: np.ndarray,
        ):
            self.col_value = col_value
            self.col_dual = col_dual
            self.row_dual = row_dual

    class _LpView:
        __slots__ = ("row_names_",)

        def __init__(self, row_names: list[str]):
            self.row_names_ = row_names

    def __init__(
        self,
        *,
        col_names: list[str],
        row_names: list[str],
        col_value: np.ndarray,
        col_dual: np.ndarray,
        row_dual: np.ndarray,
        objective: float = 0.0,
    ):
        self._col_names = col_names
        self._row_names = row_names
        self._solution = self._SolutionView(col_value, col_dual, row_dual)
        self._lp = self._LpView(row_names)
        self._obj = objective

    # --- writer-facing API ------------------------------------------------

    def allVariableNames(self) -> list[str]:
        return self._col_names

    def getSolution(self) -> "_SolHighsShim._SolutionView":
        return self._solution

    def getLp(self) -> "_SolHighsShim._LpView":
        return self._lp

    def passColName(self, col_id: int, new_name: str) -> None:
        # Mirror highspy's in-place rename onto the shim's name array.
        self._col_names[col_id] = new_name

    def getNumCol(self) -> int:
        return len(self._col_names)

    def getNumRow(self) -> int:
        return len(self._row_names)

    def getObjectiveValue(self) -> float:
        # Used by ``write_v_obj`` (scaled value, the writer un-scales).
        # Backed by the value we parsed from the ``Objective`` line in
        # the .sol file: the MPS objective, without the objective constant
        # (``Solution.obj_offset``), as a live polar-high HiGHS handle.
        return self._obj

    # Silence / status helpers a few code paths invoke defensively.
    def silent(self) -> None:  # pragma: no cover - trivial
        return None


# ---------------------------------------------------------------------------
# Per-solver CLI dispatch (copied from polar-high's _mps_fallback.py)
# ---------------------------------------------------------------------------
# Source: ``polar-high-opt/src/polar_high/solvers/_mps_fallback.py``.
# We copy verbatim (with imports adapted) rather than reuse the upstream
# ``run_via_file`` because that entry point materialises an ``LpView`` —
# the memory-greedy path we spent days bounding via ``Problem.write_mps``.
# The duplication keeps both repos independent of each other on this
# memory-critical code path.

_BINARY_NAMES: dict[str, str] = {
    "gurobi": "gurobi_cl",
    "cplex": "cplex",
    "xpress": "optimizer",
    "copt": "copt_cmd",
}

_POSIX_INSTALL_DIRS: dict[str, list[str]] = {
    "gurobi": [
        "/opt/gurobi/bin",
        "/opt/gurobi/linux64/bin",
        "/Library/gurobi/bin",
        os.path.expanduser("~/gurobi/bin"),
    ],
    "cplex": [
        "/opt/ibm/ILOG/CPLEX_Studio/cplex/bin/x86-64_linux",
        "/opt/ibm/ILOG/CPLEX_Studio/cplex/bin",
        "/opt/cplex/bin",
        os.path.expanduser("~/cplex/bin"),
    ],
    "xpress": [
        "/opt/xpressmp/bin",
        "/opt/fico/xpress/bin",
        os.path.expanduser("~/xpressmp/bin"),
    ],
    "copt": [
        "/opt/copt/bin",
        "/opt/copt71/bin",
        os.path.expanduser("~/copt/bin"),
    ],
}


def _find_solver_binary(solver_name: str) -> Path | None:
    """Return the absolute path to the solver's CLI binary, or ``None``.

    Lookup order: the folders of ``$PATH``, then a small set of conventional POSIX install dirs.  Only absolute
    ``PATH`` folders are searched, and on Windows only ``<name>.exe`` and never the current folder (which
    :func:`shutil.which` searches first there): a run's current folder is the workspace, and a received workspace
    must not bring its own ``gurobi_cl.exe`` (item 240 B2; the path also goes into a solver wrapper's arguments,
    so it must not depend on the current folder).  Returns ``None`` on miss; the caller raises a user-actionable
    error.
    """
    bin_name = _BINARY_NAMES.get(solver_name)
    if bin_name is None:
        return None
    windows = os.name == "nt"
    file_name = f"{bin_name}.exe" if windows else bin_name
    try:
        cwd = os.path.normcase(os.path.realpath(os.getcwd()))
    except OSError:
        cwd = None
    for folder in os.environ.get("PATH", "").split(os.pathsep):
        if not folder or not os.path.isabs(folder):
            continue  # an empty or relative entry is the current folder, or depends on it
        if windows and cwd is not None and os.path.normcase(os.path.realpath(folder)) == cwd:
            continue
        candidate = Path(folder) / file_name
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return Path(os.path.abspath(candidate))
    if os.name == "posix":
        for d in _POSIX_INSTALL_DIRS.get(solver_name, []):
            candidate = Path(d) / bin_name
            if candidate.is_file() and os.access(candidate, os.X_OK):
                return candidate
    return None


#: The parameters handed to a commercial solver: ``(native name, value text)`` pairs in final order
#: (:func:`commercial_params`).
CommercialParams = tuple[tuple[str, str], ...]


def commercial_params(options: "Mapping[str, Any] | Sequence[tuple[str, Any]] | None") -> CommercialParams:
    """The resolved commercial *options* (name -> value, final order; or such pairs) as :data:`CommercialParams`:
    each value formatted by :func:`_format_native_opt_value`, ``None`` values left out."""
    pairs = options.items() if isinstance(options, Mapping) else (options or ())
    return tuple((str(name), _format_native_opt_value(value)) for name, value in pairs if value is not None)


def _check_params(solver: str, params: CommercialParams) -> None:
    """Last check of the pairs a script builder emits (item 240 B4): each is one parameter with one value, so it
    becomes exactly one argument or script line.  The options were checked when they were read
    (``_solver_dispatch.validate_solver_option``), so a failure here is an internal error; the solver is not
    started."""
    from flextool.engine.solve._solver_dispatch import FlexToolUserError

    for name, value in params:
        words = name.split(" ") if solver == "cplex" else [name]
        if value and not any(ch.isspace() for ch in value) and all(w and not any(ch.isspace() for ch in w)
                                                                    for w in words):
            continue
        raise FlexToolUserError(
            f"FlexTool did not start {solver}: the parameter {name[:80]!r} = {value[:80]!r} is not one name and one "
            "value. This is an internal check on the solver options; please report it with the solve's solver "
            "options."
        )


def _gurobi_script(
    binary: Path, mps_path: Path, sol_path: Path, params: CommercialParams = (), *, is_mip: bool = False,
) -> tuple[list[str], str | None]:
    """``gurobi_cl [<Name>=<value> ...] ResultFile=<sol> <mps>``.

    Each of *params* (:func:`commercial_params`) becomes one ``<Name>=<value>`` argument before the model file —
    the only way ``gurobi_cl`` takes parameters on its command line.  (It
    has no parameter-file argument: ``ReadParams=<file>`` is reported as
    ``Unknown parameter: 'ReadParams' (ignored)`` and the solve then runs
    at Gurobi's defaults.)  The same pairs are written next to the MPS as the
    record of what was passed (:func:`_write_commercial_opt_file`).  *is_mip* is not used: ``gurobi_cl`` solves
    an LP and a MIP alike.
    """
    _check_params("gurobi", params)
    argv: list[str] = [str(binary)]
    for name, value in params:
        argv.append(f"{name}={value}")
    argv.append(f"ResultFile={sol_path}")
    argv.append(str(mps_path))
    _check_command_lines("gurobi", "argument", argv, len(params) + 3)
    return argv, None


def _cplex_script(
    binary: Path, mps_path: Path, sol_path: Path, params: CommercialParams = (), *, is_mip: bool = False,
) -> tuple[list[str], str]:
    """CPLEX interactive optimizer: pipe commands on stdin.

    Each of *params* becomes ``set <name> <value>``, emitted *before* the ``read``/``optimize``
    pair so parameter values are in effect when the model loads.  *is_mip* is not used: ``optimize`` solves an
    LP and a MIP alike.
    """
    _check_params("cplex", params)
    pre = [f"set {name} {value}" for name, value in params]
    fixed = [
        f"read {mps_path}",
        "optimize",
        f"write {sol_path} sol",
        "quit",
    ]
    return [str(binary)], _join_script("cplex", pre, fixed)


#: Xpress controls a MIP's log needs, and the values that hide it: FlexTool reads how a MIP search ended from
#: the log (:func:`_xpress_mip_status`).  ``OUTPUTLOG`` other than 1 (0 no output, 2 / 3 only warnings and
#: errors) and ``MIPLOG 0`` leave out the ``*** Search completed ***``, ``Final MIP objective`` and ``Number of
#: solutions found`` lines (xpress 9.8.1 library); every other ``MIPLOG`` (1, 2, 3, a negative value) prints them.
_XPRESS_MIP_LOG_CONTROLS = (("OUTPUTLOG", lambda v: v != 1.0), ("MIPLOG", lambda v: v == 0.0))


def xpress_mip_log_overrides(params: CommercialParams) -> list[tuple[str, str, str]]:
    """The Xpress controls a MIP solve sets back to 1 after *params*, as ``(name, "1", the value it replaces)``:
    ``OUTPUTLOG`` when *params*' last one is not 1, ``MIPLOG`` when it is 0 (see
    :data:`_XPRESS_MIP_LOG_CONTROLS`).  A control *params* does not set keeps Xpress' default, which logs."""
    from flextool.engine.solve._solver_catalogue import commercial_option_key

    last: dict[str, str] = {}
    for name, value in params:
        last[commercial_option_key("xpress", name)] = value
    out: list[tuple[str, str, str]] = []
    for name, hides in _XPRESS_MIP_LOG_CONTROLS:
        value = last.get(name.casefold())
        if value is None:
            continue
        try:
            number = float(value)
        except ValueError:
            number = math.nan
        if math.isnan(number) or hides(number):
            out.append((name, "1", value))
    return out


def _xpress_script(
    binary: Path, mps_path: Path, sol_path: Path, params: CommercialParams = (), *, is_mip: bool = False,
) -> tuple[list[str], str]:
    """Xpress optimizer console script via stdin.

    Each of *params* becomes ``setControl <NAME> <value>``, emitted *before* ``readprob``/
    ``lpoptimize`` so controls are in effect for the solve.  A MIP (*is_mip*) is solved by ``mipoptimize``:
    ``lpoptimize`` would solve only its LP relaxation.  An LP keeps ``lpoptimize``.  A MIP whose *params* would
    hide the log FlexTool reads gets ``setControl OUTPUTLOG 1`` / ``setControl MIPLOG 1`` after them
    (:func:`xpress_mip_log_overrides`).
    """
    _check_params("xpress", params)
    pre = [f"setControl {name} {value}" for name, value in params]
    if is_mip:
        pre += [f"setControl {name} {value}" for name, value, _ in xpress_mip_log_overrides(params)]
    fixed = [
        f"readprob {mps_path}",
        "mipoptimize" if is_mip else "lpoptimize",
        # Force the MPS-like **SLX** solution (name-based, whitespace-
        # safe) for the primal — Xpress' default ``writesol`` emits an
        # index-based ``.asc``/``.hdr`` CSV pair with no usable
        # name→value mapping.  ``writeprtsol`` carries the objective
        # value SLX omits.  Xpress appends ``.slx`` / ``.prt`` to the
        # base path; :func:`_parse_xpress_sol` looks for those.
        f"writeslxsol {sol_path}",
        f"writeprtsol {sol_path}",
        "quit",
    ]
    return [str(binary)], _join_script("xpress", pre, fixed)


def _copt_script(
    binary: Path, mps_path: Path, sol_path: Path, params: CommercialParams = (), *, is_mip: bool = False,
) -> tuple[list[str], str]:
    """COPT's ``copt_cmd`` script via stdin.

    Each of *params* becomes ``set <ParamName> <value>``, emitted *before* ``read``/``optimize``.  *is_mip* is
    not used: ``optimize`` solves an LP and a MIP alike.
    """
    _check_params("copt", params)
    pre = [f"set {name} {value}" for name, value in params]
    fixed = [
        f"read {mps_path}",
        "optimize",
        f"write {sol_path}",
        "quit",
    ]
    return [str(binary)], _join_script("copt", pre, fixed)


#: Characters that end a line for a console or for :meth:`str.splitlines`.
_LINE_BREAKS = frozenset("\n\r\x0b\x0c\x1c\x1d\x1e\x85\u2028\u2029")


def _check_command_lines(solver: str, kind: str, lines: list[str], expected: int) -> None:
    """Last check before a solver starts (item 236 A4): *lines* (the command script's lines, or Gurobi's
    arguments) are one per parameter plus the fixed ones, and none holds a line break or ends with a backslash
    (a Tcl console joins such a line with the next).  The options were
    checked when they were read (``_solver_dispatch.validate_solver_option``), so a failure here is an
    internal error; the solver is not started."""
    from flextool.engine.solve._solver_dispatch import FlexToolUserError

    bad = next((line for line in lines if _LINE_BREAKS.intersection(line) or line.endswith("\\")), None)
    if bad is None and len(lines) == expected:
        return
    detail = (
        f"the {kind} {bad[:80]!r} holds a line break or ends with a backslash" if bad is not None
        else f"it has {len(lines)} {kind}s, not {expected}"
    )
    raise FlexToolUserError(
        f"FlexTool did not start {solver}: {detail}. This is an internal check on the solver options; "
        "please report it with the solve's solver options."
    )


def _join_script(solver: str, pre: list[str], fixed: list[str]) -> str:
    """The stdin script of a console solver: the parameter lines *pre*, then the *fixed* commands, each ending
    in a line break; checked by :func:`_check_command_lines`."""
    lines = pre + fixed
    _check_command_lines(solver, "line", lines, len(pre) + len(fixed))
    return "\n".join(lines + [""])


_SCRIPTS = {
    "gurobi": _gurobi_script,
    "cplex": _cplex_script,
    "xpress": _xpress_script,
    "copt": _copt_script,
}


# ---------------------------------------------------------------------------
# Per-solver option-file handling
# ---------------------------------------------------------------------------
# Each commercial solver gets a baseline ``solver_config/<solver>.opt``
# (shipped as a template, user-editable), written in the *solver's own*
# parameter syntax — see the comment header in each template.
# ``_solver_dispatch.resolve_commercial_options`` merges it with the solve's
# ``solver_mip_gap`` / ``solver_arguments`` and the operator's flags into one
# ordered parameter set; :func:`commercial_params` turns it into ``(name,
# value)`` pairs, which the per-solver script turns into the CLI's form
# (Gurobi: one ``Name=value`` argv token per parameter; CPLEX/Xpress/COPT:
# inline set/setControl commands on stdin), and :func:`_write_commercial_opt_file`
# writes next to the MPS as the record of what was passed (nothing reads it back).


def _parse_native_opt_file(path: Path) -> dict[str, str]:
    """Parse a native solver ``.opt`` file into a name → value dict.

    Format (shared across all four commercial baselines):

    * ``#`` introduces a comment line; blank lines are ignored.
    * Every other line is split on the *first* whitespace run into a
      ``(name, value)`` pair.  The name may contain a single internal
      space when needed (CPLEX uses dotted-or-spaced names like
      ``mip tolerances mipgap``) — the parser uses :func:`str.rsplit`
      with ``maxsplit=1`` so the **last** whitespace-separated token is
      the value and everything before is the name.

    Returns an empty dict when the file is missing or unreadable.
    """
    text = _read_native_opt_text(path)
    return {} if text is None else _parse_native_opt_text(text)


def _read_native_opt_text(path: Path) -> str | None:
    """The text of a native ``.opt`` file, read as UTF-8 without a leading BOM (as the run's solver_config
    folders are read); ``None`` when it is missing, unreadable or not UTF-8 text (with a warning)."""
    if not path.is_file():
        return None
    try:
        return path.read_text(encoding="utf-8-sig")
    except UnicodeDecodeError as exc:
        logging.getLogger(__name__).warning("Not reading %s: it is not UTF-8 text (%s).", path, exc)
        return None
    except OSError:
        return None


def _parse_native_opt_text(text: str) -> dict[str, str]:
    """The ``name value`` lines of a native ``.opt`` text (rules of :func:`_parse_native_opt_file`)."""
    return {name: value for _, name, value in _native_opt_lines(text)}


def _format_native_opt_value(v: object) -> str:
    """Render *v* for a native solver opt-file line.

    Booleans become ``1`` / ``0`` (Gurobi/COPT accept either word or
    int; the int form is portable across CPLEX/Xpress too).  A whole number
    is written without its ``.0`` (``Threads 1``, not ``Threads 1.0``), as a
    float or as the text of one (``solver_arguments`` values arrive as
    ``str(float)``): the database stores every number as a float, and an
    integer parameter may refuse ``1.0`` (the Xpress library does), while a
    float parameter reads ``1`` as 1.0.
    """
    if isinstance(v, bool):
        return "1" if v else "0"
    if isinstance(v, float) and v.is_integer() and abs(v) < 1e15:
        return str(int(v))
    if isinstance(v, str) and _WHOLE_NUMBER_TEXT.match(v):
        return v.strip().split(".", 1)[0]
    return str(v)


#: The text of a whole number written as a float (``1.0``, ``-3.00``), as ``str(float)`` gives it.
_WHOLE_NUMBER_TEXT = re.compile(r"^\s*-?\d{1,15}\.0+\s*$")


def _write_commercial_opt_file(
    options: "Mapping[str, Any] | Sequence[tuple[str, Any]] | None",
    out_path: Path,
) -> Path | None:
    """Write the resolved commercial *options* (name -> value, final order; or :data:`CommercialParams`) to
    *out_path*.

    One ``<name> <value>`` line per parameter, the layout of the baseline
    files: the record of what was passed (``<solve>.opt`` next to the MPS; the
    script builders take the pairs themselves).  Returns ``None`` when there is
    nothing to pass (no file is written).
    """
    lines = [f"{name} {value}\n" for name, value in commercial_params(options)]
    if not lines:
        return None
    out_path.write_text("".join(lines), encoding="utf-8")
    return out_path


class _ParsedSolution(NamedTuple):
    """What a commercial result file (and the solver's console output) says about a solve.

    ``primal`` is ``None`` when there is no solution; ``mip_gap`` is the
    relative gap the solver reported (a fraction), when known.
    """

    status: str
    objective: float | None
    primal: dict[str, float] | None
    dual: dict[str, float] | None
    mip_gap: float | None = None


#: Gurobi ``Status`` codes (``SolutionInfo.Status`` of a JSON result file).
_GUROBI_STATUS = {
    2: "OPTIMAL", 3: "INFEASIBLE", 4: "INF_OR_UNBD", 5: "UNBOUNDED", 6: "CUTOFF", 7: "ITERATION_LIMIT",
    8: "NODE_LIMIT",
    9: "TIME_LIMIT", 10: "SOLUTION_LIMIT", 11: "INTERRUPTED", 12: "NUMERIC", 13: "SUBOPTIMAL",
    15: "USER_OBJ_LIMIT", 16: "WORK_LIMIT", 17: "MEM_LIMIT",
}
_GUROBI_JSON_INFO = re.compile(rb'"SolutionInfo"\s*:\s*(\{[^{}]*\})')
_GUROBI_JSON_VAR = re.compile(rb'"VarName"\s*:\s*"([^"]*)"\s*,\s*"X"\s*:\s*([^,}\s]+)')
#: Console lines of a solve stopped by its time limit (Gurobi / COPT / Xpress).
_GUROBI_TIME_LIMIT = re.compile(r"^\s*Time limit reached\s*$", re.MULTILINE)
#: Console lines of a Gurobi solve stopped by another limit or an interrupt (the ``.sol`` text has no status).
_GUROBI_OTHER_STOP = re.compile(
    r"^\s*(Solution limit|Node limit|Iteration limit|Work limit|Memory limit|Interrupt request|Solve interrupted)"
    r"\b", re.MULTILINE | re.IGNORECASE,
)
_GUROBI_STOP_STATUS = {
    "solution limit": "SOLUTION_LIMIT", "node limit": "NODE_LIMIT", "iteration limit": "ITERATION_LIMIT",
    "work limit": "WORK_LIMIT", "memory limit": "MEM_LIMIT", "interrupt request": "INTERRUPTED",
    "solve interrupted": "INTERRUPTED",
}
#: Console lines of a Gurobi solve that finished: ``Optimal solution found (tolerance ...)`` (MIP),
#: ``Optimal objective  <v>`` (LP).
_GUROBI_OPTIMAL = re.compile(r"^\s*Optimal (solution found|objective)\b", re.MULTILINE)
_GUROBI_GAP = re.compile(r"^Best objective [^,]*, best bound [^,]*, gap ([-+\d.eE]+)%", re.MULTILINE)
#: COPT's summary of how a MIP ended (``MIP status      : solved`` / ``: stopped (time limit reached)`` /
#: ``: stopped (node limit reached)``) and of an LP (``Status: Optimal  Objective: ...``, ``Status: Timeout``,
#: ``Status: IterLimit``); FlexTool reads the last of each.
_COPT_MIP_STATUS = re.compile(r"^\s*MIP status\s*:\s*(.+?)\s*$", re.IGNORECASE | re.MULTILINE)
_COPT_LP_STATUS = re.compile(r"^\s*Status:\s*(\w+)", re.MULTILINE)
#: COPT LP ``Status:`` words, as FlexTool's status names (another word becomes its upper-case self).
_COPT_LP_STATUS_NAMES = {
    "optimal": "OPTIMAL", "infeasible": "INFEASIBLE", "unbounded": "UNBOUNDED", "infororunb": "INF_OR_UNBD",
    "timeout": "TIME_LIMIT", "iterlimit": "ITERATION_LIMIT", "interrupted": "INTERRUPTED",
}
_COPT_GAP = re.compile(r"^Best gap\s*:\s*([-+\d.eE]+)%", re.MULTILINE)
_CPLEX_GAP = re.compile(r"gap\s*=\s*[^,()]*,\s*([-+\d.eE]+)%")
_XPRESS_TIME_LIMIT = re.compile(r"MAXTIME limit reached|\*\*\* Time limit reached \*\*\*|time limit hit",
                                re.IGNORECASE)
#: How an Xpress MIP search ended, from its log.  The print solution cannot tell: after a search stopped by a
#: limit it still says ``Optimal solution found`` (that is the LP at the last node), and after a search stopped
#: before any integer solution the SLX file holds the LP relaxation.  Verified with the ``xpress`` 9.8.1 Python
#: package (the library's log, which the console prints); the console's own wording is not verified.
#: ``STOPPING - MAXNODE limit reached (MAXNODE=5  nodes=5).`` / ``... MAXMIPSOL ...`` / ``... MAXTIME ...``.
_XPRESS_MIP_STOP = re.compile(r"^\s*STOPPING - (MAXTIME|MAXNODE|MAXMIPSOL) limit reached", re.MULTILINE)
_XPRESS_MIP_STOP_STATUS = {"MAXTIME": "TIME_LIMIT", "MAXNODE": "NODE_LIMIT", "MAXMIPSOL": "SOLUTION_LIMIT"}
_XPRESS_SEARCH_COMPLETED = re.compile(r"^\s*\*\*\* Search completed \*\*\*", re.MULTILINE)
_XPRESS_SEARCH_UNFINISHED = re.compile(r"^\s*\*\*\* Search unfinished \*\*\*", re.MULTILINE)
_XPRESS_INTEGER_INFEASIBLE = re.compile(r"^\s*Problem is integer infeasible\b", re.MULTILINE)
#: ``Problem is unbounded`` after ``*** Search completed ***`` (mipstatus 7); the log may still count heuristic
#: solutions and print a ``Final MIP objective``.
_XPRESS_UNBOUNDED = re.compile(r"^\s*Problem is unbounded\b", re.MULTILINE)
#: ``  Number of solutions found / nodes   :        11 /      4693``: the integer solutions of the search.
_XPRESS_MIP_SOLUTIONS = re.compile(r"^\s*Number of solutions found / nodes\s*:\s*(\d+)\s*/", re.MULTILINE)
#: ``Final MIP objective                   : 2.300000000000000e+01`` (written only when a solution exists) and
#: ``Final MIP bound                       : 0.000000000000000e-01``.
_XPRESS_FINAL_OBJECTIVE = re.compile(r"^\s*Final MIP objective\s*:\s*(\S+)", re.MULTILINE)
_XPRESS_FINAL_BOUND = re.compile(r"^\s*Final MIP bound\s*:\s*(\S+)", re.MULTILINE)
#: ``gurobi_cl``'s message for a parameter name it does not know (it then solves on without it).
_GUROBI_UNKNOWN_PARAMETER = re.compile(r"Unknown parameter:?\s*'([^']+)'")


def _percent(match: "re.Match[str] | None") -> float | None:
    if match is None:
        return None
    try:
        return float(match.group(1)) / 100.0
    except ValueError:
        return None


def _parse_gurobi_json(path: Path) -> _ParsedSolution | None:
    """Parse a Gurobi JSON result file (``ResultFile=<x>.json``), else ``None`` when *path* is not one.

    ``SolutionInfo`` carries ``Status`` / ``ObjVal`` / ``MIPGap`` / ``SolCount``
    whatever the console logging; ``Vars`` lists the non-zero values only and
    is absent when there is no solution.  Read through ``mmap`` with a bytes
    regex, so a large file is not loaded as Python objects.
    """
    try:
        with path.open("rb") as fh:
            if not fh.read(64).lstrip().startswith(b"{"):
                return None
            fh.seek(0)
            try:
                data = mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ)
            except ValueError:  # empty file
                return None
            try:
                m = _GUROBI_JSON_INFO.search(data)
                info = json.loads(m.group(1)) if m else {}
                has_solution = data.find(b'"Vars"') >= 0
                primal = {
                    name.decode(): float(value)
                    for name, value in _GUROBI_JSON_VAR.findall(data)
                } if has_solution else None
            finally:
                data.close()
    except (OSError, ValueError):
        return None
    status = _GUROBI_STATUS.get(info.get("Status"), "OTHER")
    objective = info.get("ObjVal") if has_solution else None
    gap = info.get("MIPGap")
    return _ParsedSolution(
        status,
        float(objective) if isinstance(objective, (int, float)) else None,
        primal,
        None,
        float(gap) if isinstance(gap, (int, float)) and has_solution and abs(gap) < 1e99 else None,
    )


def _parse_gurobi_sol(path: Path, stdout: str = "", *, is_mip: bool = False) -> _ParsedSolution:
    """Parse a Gurobi result file: JSON (status, gap, non-zero values) first,
    else the ``.sol`` text format (values only).

    With the text format the status comes from the console output: a primal
    with ``Time limit reached`` in *stdout* is ``TIME_LIMIT``, one with another
    limit or an interrupt (``Node limit reached``, ...) carries that status (with
    the console's gap), one
    with ``Optimal solution found`` / ``Optimal objective`` is ``OPTIMAL``.  With
    none of these (console logging off, ``LogToConsole 0``) how the solve ended
    is unknown: ``NO_STATUS``, which the acceptance check rejects, since a solve
    stopped by a node limit or an interrupt would read the same.  (FlexTool asks
    for a JSON result file, which carries the status whatever the logging.)
    """
    parsed = _parse_gurobi_json(path) if path.is_file() else None
    if parsed is not None:
        if parsed.status in LIMIT_SHIM_STATUSES and parsed.mip_gap is None:
            parsed = parsed._replace(mip_gap=_percent(_GUROBI_GAP.search(stdout)))
        return parsed
    status, objective, primal = _parse_sol_text(path)
    if primal and _GUROBI_TIME_LIMIT.search(stdout):
        return _ParsedSolution("TIME_LIMIT", objective, primal, None, _percent(_GUROBI_GAP.search(stdout)))
    if primal:
        stop = _GUROBI_OTHER_STOP.search(stdout)
        if stop is not None:
            # Another limit (a MIP keeps its solution, the gap named) or an interrupt (rejected by name).
            return _ParsedSolution(
                _GUROBI_STOP_STATUS[stop.group(1).lower()], objective, primal, None,
                _percent(_GUROBI_GAP.search(stdout)),
            )
        if not _GUROBI_OPTIMAL.search(stdout):
            return _ParsedSolution("NO_STATUS", objective, primal, None)
    return _ParsedSolution(status, objective, primal, None)


def _parse_sol_text(
    path: Path,
) -> tuple[str, float | None, dict[str, float] | None]:
    """Parse a Gurobi / COPT ``.sol`` text file (``<name> <value>`` lines).

    Returns ``(status_str, objective, primal_dict)`` where ``status_str`` is
    ``"OPTIMAL"`` when there are values, else ``"OTHER"``.
    """
    if not path.is_file():
        return "OTHER", None, None
    objective: float | None = None
    primal: dict[str, float] = {}
    with path.open("r") as fh:
        for raw in fh:
            line = raw.strip()
            if not line:
                continue
            if line.startswith("#"):
                # Gurobi writes ``# Objective value = <v>``; COPT writes
                # ``# Objective value <v>`` (no ``=``) — accept either.
                m = re.search(r"[Oo]bjective\s+value\s*=?\s*([-\d.eE+inf]+)", line)
                if m:
                    try:
                        objective = float(m.group(1))
                    except ValueError:
                        objective = None
                continue
            parts = line.split()
            if len(parts) >= 2:
                try:
                    primal[parts[0]] = float(parts[1])
                except ValueError:
                    continue
    status = "OPTIMAL" if primal else "OTHER"
    return status, objective, (primal or None)


def _copt_status(stdout: str) -> str | None:
    """How a COPT solve ended, from its console log; ``None`` when the log does not say (logging off).

    A MIP's last ``MIP status`` line: ``solved`` is ``OPTIMAL``; ``stopped (time limit reached)`` is
    ``TIME_LIMIT``; ``stopped (node limit reached)`` ``NODE_LIMIT``; another ``stopped (...)`` is
    ``INTERRUPTED`` when it names an interrupt, else ``STOPPED``; ``infeasible`` / ``unbounded`` as named.  Else
    an LP's last ``Status:`` word (:data:`_COPT_LP_STATUS_NAMES`; another word, e.g. ``Numerical``, becomes
    ``NUMERICAL``).  Only ``OPTIMAL`` is optimal: a MIP stopped by its time or node limit keeps its solution (the
    D2 rule of :mod:`flextool.engine.solve._acceptance`, every search limit); an LP at a limit, an interrupt and
    an unnamed stop are rejected by name.
    """
    mip = _COPT_MIP_STATUS.findall(stdout)
    if mip:
        text = mip[-1].lower()
        if text.startswith("solved"):
            return "OPTIMAL"
        if text.startswith("stopped"):
            if "time limit" in text:
                return "TIME_LIMIT"
            if "node limit" in text:
                return "NODE_LIMIT"
            return "INTERRUPTED" if "interrupt" in text else "STOPPED"
        if "infeasible or unbounded" in text:
            return "INF_OR_UNBD"
        if "infeasible" in text:
            return "INFEASIBLE"
        if "unbounded" in text:
            return "UNBOUNDED"
        return "OTHER"
    lp = _COPT_LP_STATUS.findall(stdout)
    if lp:
        word = lp[-1]
        return _COPT_LP_STATUS_NAMES.get(word.lower(), word.upper())
    return None


def _parse_copt_sol(path: Path, stdout: str = "", *, is_mip: bool = False) -> _ParsedSolution:
    """COPT's ``.sol`` has Gurobi's text layout and no status; the status comes from the console
    (:func:`_copt_status`).

    A status is kept with or without a solution file (COPT writes none when it has no solution), so a run
    without one is told why (``stopped at its time limit before finding a solution``).  Values with no status
    in the log (``Logging 0`` / ``LogToConsole 0``) are ``NO_STATUS``: how the solve ended is unknown, so the
    acceptance check rejects them.
    """
    _, objective, primal = _parse_sol_text(path)
    status = _copt_status(stdout)
    if status is None:
        return _ParsedSolution("NO_STATUS" if primal else "OTHER", objective, primal, None)
    gap = _percent(_COPT_GAP.search(stdout)) if primal and status in LIMIT_SHIM_STATUSES else None
    return _ParsedSolution(status, objective if primal else None, primal, None, gap)


#: CPLEX ``solutionStatusValue`` codes (``CPX_STAT_*`` for LP, ``CPXMIP_*`` for MIP; the CPLEX reference manual,
#: "Solution status codes"), read first.  A limit or an abort is never optimal: a MIP with a solution at a time,
#: node, solution or iteration limit keeps it (the D2 rule of :mod:`flextool.engine.solve._acceptance`, every
#: search limit); an LP at a limit, an objective-limit abort (12 / 21 / 22, LP only) and an interrupt are rejected
#: by name.  (128 / 130 are the populate codes, unused by FlexTool and not verified here.)
#: "Optimal with unscaled infeasibilities" (5, 115) breaks the feasibility tolerances once unscaled, and an
#: "unbounded optimal face" (20, barrier without crossover) returns an interior point of an unbounded face, not a
#: vertex: neither is a certified optimum, and FlexTool has no diagnostics of a solver run as a program to verify
#: one, so both are rejected (as HiGHS' uncertified ``kUnknown`` is without diagnostics that show a feasible
#: primal).  A code not listed falls back to the phrase table.
_CPLEX_STATUS_CODES: Mapping[int, str] = {
    1: "OPTIMAL", 101: "OPTIMAL", 102: "OPTIMAL", 128: "OPTIMAL", 129: "OPTIMAL",
    2: "UNBOUNDED", 118: "UNBOUNDED",
    3: "INFEASIBLE", 103: "INFEASIBLE",
    4: "INF_OR_UNBD", 119: "INF_OR_UNBD",
    5: "OPTIMAL_INFEASIBLE", 115: "OPTIMAL_INFEASIBLE",
    20: "OPTIMAL_FACE_UNBOUNDED",
    6: "NUMERIC",
    10: "ITERATION_LIMIT",
    11: "TIME_LIMIT", 25: "TIME_LIMIT", 107: "TIME_LIMIT", 108: "TIME_LIMIT", 131: "TIME_LIMIT", 132: "TIME_LIMIT",
    12: "OBJECTIVE_LIMIT", 21: "OBJECTIVE_LIMIT", 22: "OBJECTIVE_LIMIT",
    13: "INTERRUPTED", 113: "INTERRUPTED", 114: "INTERRUPTED",
    104: "SOLUTION_LIMIT", 130: "SOLUTION_LIMIT",
    105: "NODE_LIMIT", 106: "NODE_LIMIT",
    111: "MEM_LIMIT", 112: "MEM_LIMIT",
}

#: CPLEX ``solutionStatusString`` phrases, checked in order (``CPXgetstatstring``), for a file without a known
#: ``solutionStatusValue``.  An unrecognised string stays ``OTHER`` (rejected, naming the solver status).
_CPLEX_STATUS_PHRASES: tuple[tuple[str, str], ...] = (
    ("non-optimal", "OTHER"),
    ("nonoptimal", "OTHER"),
    ("unscaled infeasibilities", "OPTIMAL_INFEASIBLE"),   # "[integer] optimal with unscaled infeasibilities"
    ("face unbounded", "OPTIMAL_FACE_UNBOUNDED"),         # "optimal face unbounded"
    ("time limit", "TIME_LIMIT"),          # also "deterministic time limit exceeded"
    ("node limit", "NODE_LIMIT"),
    ("solution limit", "SOLUTION_LIMIT"),
    ("iteration limit", "ITERATION_LIMIT"),
    ("memory limit", "MEM_LIMIT"),
    ("abort", "INTERRUPTED"),
    ("infeasible or unbounded", "INF_OR_UNBD"),
    ("optimal", "OPTIMAL"),                # "optimal", "integer optimal solution", "integer optimal, tolerance"
    ("infeasible", "INFEASIBLE"),
    ("unbounded", "UNBOUNDED"),
)

#: The interactive optimizer's line for a solve stopped by a time limit (``Time limit exceeded, no solution.``,
#: ``Deterministic time limit exceeded``): the reason when CPLEX wrote no solution file.
_CPLEX_TIME_LIMIT_TEXT = re.compile(r"time limit exceeded", re.IGNORECASE)


def _cplex_status(status_str: str, status_value: int | None = None) -> str:
    """The status of a CPLEX solution header: its ``solutionStatusValue`` when known
    (:data:`_CPLEX_STATUS_CODES`), else its ``solutionStatusString`` (lower case); ``OTHER`` when neither is
    recognised or both are empty."""
    if status_value is not None and status_value in _CPLEX_STATUS_CODES:
        return _CPLEX_STATUS_CODES[status_value]
    for phrase, status in _CPLEX_STATUS_PHRASES:
        if phrase in status_str:
            return status
    return "OTHER"


def _parse_cplex_sol(path: Path, stdout: str = "", *, is_mip: bool = False) -> _ParsedSolution:
    """Parse CPLEX XML ``.sol`` via :mod:`xml.etree.ElementTree`; the gap of a
    MIP stopped by a limit comes from the console (``gap = ..., X%``).

    CPLEX writes no file without a solution; a console that says ``time limit exceeded`` then gives
    ``TIME_LIMIT`` (no solution), so the run is told it stopped at its limit.
    """
    if not path.is_file():
        return _ParsedSolution("TIME_LIMIT" if _CPLEX_TIME_LIMIT_TEXT.search(stdout) else "OTHER", None, None, None)
    try:
        tree = ET.parse(path)
    except ET.ParseError:
        return _ParsedSolution("OTHER", None, None, None)
    root = tree.getroot()
    if root.tag == "CPLEXSolutions":
        children = list(root)
        if not children:
            return _ParsedSolution("OTHER", None, None, None)
        root = children[0]
    objective: float | None = None
    status = "OTHER"
    status_str = ""
    header = root.find("header")
    if header is not None:
        obj_str = header.get("objectiveValue")
        if obj_str is not None:
            try:
                objective = float(obj_str)
            except ValueError:
                pass
        status_str = (header.get("solutionStatusString") or "").strip().lower()
        try:
            status_value: int | None = int(header.get("solutionStatusValue") or "")
        except ValueError:
            status_value = None
        status = _cplex_status(status_str, status_value)
        if status_value is not None:
            status_str = status_str or str(status_value)
    primal: dict[str, float] = {}
    vars_el = root.find("variables")
    if vars_el is not None:
        for v in vars_el.findall("variable"):
            name = v.get("name")
            val_str = v.get("value")
            if name is None or val_str is None:
                continue
            try:
                primal[name] = float(val_str)
            except ValueError:
                continue
    dual: dict[str, float] = {}
    cons_el = root.find("linearConstraints")
    if cons_el is not None:
        for c in cons_el.findall("constraint"):
            name = c.get("name")
            dual_str = c.get("dual")
            if name is None or dual_str is None:
                continue
            try:
                dual[name] = float(dual_str)
            except ValueError:
                continue
    if status == "OTHER" and primal and not status_str:
        # A file without a status string: the values are all there is to go on.
        status = "OPTIMAL"
    gaps = _CPLEX_GAP.findall(stdout) if status in LIMIT_SHIM_STATUSES else []
    gap = None
    if gaps:
        try:
            gap = float(gaps[-1]) / 100.0
        except ValueError:
            gap = None
    return _ParsedSolution(status, objective, (primal or None), (dual or None), gap)


def _parse_xpress_sol(path: Path, stdout: str = "", *, is_mip: bool = False) -> _ParsedSolution:
    """Parse the Xpress solution written by :func:`_xpress_script`.

    We force Xpress to emit the MPS-like **SLX** solution (name-based,
    whitespace-safe) for the primal plus the **print** solution for the
    objective — rather than the default ``writesol`` output, which is an
    index-based ``.asc``/``.hdr`` CSV pair carrying no usable name→value
    mapping.  Xpress appends ``.slx`` / ``.prt`` to the base path passed
    to ``writeslxsol`` / ``writeprtsol``, so look for those next to
    *path* (with a couple of fallbacks for console-version differences).

    SLX body lines look like ``  C C0000001   10`` (indicator, name,
    value); the print solution carries one
    ``Objective function value is <v>`` line.

    A MIP (*is_mip*) is judged by its log (:func:`_xpress_mip_status`), never by the print solution: its values
    are used only when the log shows an integer solution.
    """
    def _first_existing(cands: list[Path]) -> Path | None:
        for c in cands:
            if c.is_file():
                return c
        return None

    primal: dict[str, float] = {}
    slx = _first_existing(
        [Path(str(path) + ".slx"), path, path.with_suffix(".slx")]
    )
    if slx is not None:
        for raw in slx.read_text().splitlines():
            parts = raw.split()
            # Data rows carry a trailing numeric value and a name just
            # before it; header lines (NAME/ENDATA/OBJSENSE) fail the
            # float() and are skipped.  ``_generic_col_id`` downstream
            # keeps only the ``C…`` column names.
            if len(parts) < 2:
                continue
            try:
                val = float(parts[-1])
            except ValueError:
                continue
            primal[parts[-2]] = val

    objective: float | None = None
    unfinished = False
    prt = _first_existing(
        [Path(str(path) + ".prt"), path.with_suffix(".prt")]
    )
    if prt is not None:
        prt_text = prt.read_text()
        # ``Problem is unfinished`` (the print solution's statistics) where a
        # finished solve says ``Optimal solution found``.
        unfinished = "Problem is unfinished" in prt_text
        m = re.search(
            r"[Oo]bjective\s+function\s+value\s+is\s+([-\d.eE+]+)",
            prt_text,
        )
        if m:
            try:
                objective = float(m.group(1))
            except ValueError:
                objective = None

    if is_mip:
        return _xpress_mip_result(stdout, primal, objective)
    if not primal:
        status = "OTHER"
    elif _XPRESS_TIME_LIMIT.search(stdout):
        status = "TIME_LIMIT"
    elif unfinished:
        status = "UNFINISHED"
    else:
        status = "OPTIMAL"
    return _ParsedSolution(status, objective, (primal or None), None)


def _last_float(pattern: "re.Pattern[str]", text: str) -> float | None:
    """The number of *pattern*'s last match in *text*, else ``None``."""
    found = pattern.findall(text)
    if not found:
        return None
    try:
        return float(found[-1])
    except ValueError:
        return None


def _xpress_mip_status(stdout: str) -> str:
    """How an Xpress MIP search ended, from its log (library-verified wording, see :data:`_XPRESS_MIP_STOP`).

    ``Problem is integer infeasible`` → ``INFEASIBLE``; ``Problem is unbounded`` → ``UNBOUNDED`` (the log also
    says ``*** Search completed ***``); a ``STOPPING - <limit> limit reached`` line → that limit's status; ``*** Search completed ***`` → ``OPTIMAL``; any other ``*** Search unfinished ***`` →
    ``UNFINISHED``; a log with none of these → ``OTHER``.
    """
    if _XPRESS_INTEGER_INFEASIBLE.search(stdout):
        return "INFEASIBLE"
    if _XPRESS_UNBOUNDED.search(stdout):
        return "UNBOUNDED"
    stops = _XPRESS_MIP_STOP.findall(stdout)
    if stops:
        return _XPRESS_MIP_STOP_STATUS[stops[-1]]
    if _XPRESS_TIME_LIMIT.search(stdout):
        return "TIME_LIMIT"
    if _XPRESS_SEARCH_UNFINISHED.search(stdout):
        return "UNFINISHED"
    if _XPRESS_SEARCH_COMPLETED.search(stdout):
        return "OPTIMAL"
    return "OTHER"


def _xpress_mip_result(stdout: str, primal: dict[str, float], prt_objective: float | None) -> _ParsedSolution:
    """The result of an Xpress MIP: the SLX values count only when the log shows an integer solution
    (``Number of solutions found`` above 0, or a ``Final MIP objective`` line).  Without one the SLX file holds
    the LP relaxation (or nothing), so there is no solution and the run stops.  The objective is the log's
    ``Final MIP objective`` (full precision), else the print solution's; the relative gap of a search stopped by
    a limit is ``|objective - bound| / max(|objective|, |bound|)`` (Xpress' ``MIPRELSTOP`` measure) from the
    log's final objective and bound."""
    status = _xpress_mip_status(stdout)
    final_objective = _last_float(_XPRESS_FINAL_OBJECTIVE, stdout)
    counts = _XPRESS_MIP_SOLUTIONS.findall(stdout)
    has_solution = (int(counts[-1]) > 0) if counts else final_objective is not None
    if status in ("INFEASIBLE", "UNBOUNDED") or not has_solution or not primal:
        if status in ("OPTIMAL", "OTHER"):
            status = "NO_INTEGER_SOLUTION"
        return _ParsedSolution(status, None, None, None)
    objective = final_objective if final_objective is not None else prt_objective
    gap = None
    bound = _last_float(_XPRESS_FINAL_BOUND, stdout)
    if status in LIMIT_SHIM_STATUSES and objective is not None and bound is not None:
        scale = max(abs(objective), abs(bound))
        gap = abs(objective - bound) / scale if scale > 0.0 else 0.0
    return _ParsedSolution(status, objective, primal, None, gap)


#: Each parser takes ``(path, stdout, *, is_mip)``; only Xpress' reads *is_mip* (its print solution cannot tell
#: how a MIP search ended).
_PARSERS = {
    "gurobi": _parse_gurobi_sol,
    "cplex": _parse_cplex_sol,
    "xpress": _parse_xpress_sol,
    "copt": _parse_copt_sol,
}


# Phrases of a licence *failure* in the solvers' output.  A bare mention of
# "license" is not one: every Gurobi run prints a banner such as ``Using
# license file /opt/gurobi/gurobi.lic`` or ``Restricted license - for
# non-production use only - expires 2027-11-24``, so an infeasible or
# otherwise failed solve would be misreported as a licensing problem.
# ``licen[cs]e`` accepts both spellings; matched case-insensitively.
_LICENSE_ERROR_PATTERNS = tuple(re.compile(p, re.IGNORECASE) for p in (
    # Gurobi "No Gurobi license found", CPLEX "no license found", Xpress/COPT "No license found"
    r"\bno\s+(?:valid\s+)?(?:\w+\s+)?licen[cs]e\s+(?:file\s+)?(?:was\s+)?found",
    # "license expired", "license has expired", "licence is expired" (not the banner's "expires <date>")
    r"\blicen[cs]e\s+(?:has\s+|is\s+)?expired",
    # Gurobi "License not valid", "Request denied: license not valid", "license is not valid for this host"
    r"\blicen[cs]e\b[^\n]{0,60}?\bnot\s+valid",
    r"\binvalid\s+licen[cs]e",
    # "Failed to obtain license", "Unable to check out license", "Failed to set up a license", "Fail to ..."
    r"\b(?:failed|fail|unable|cannot|could\s+not)\s+(?:to\s+)?(?:obtain|get|set\s+up|find|locate|open|read|"
    r"check\s*out|validate|verify|initiali[sz]e)\s+(?:a\s+|the\s+|any\s+)?(?:\w+\s+)?licen[cs]e",
    # "license error", "licensing error", "license check failed", "license not found", "licence unavailable"
    r"\blicen[cs](?:e|ing)\s+(?:error|failure|failed|check\s+failed|not\s+found|unavailable|denied|problem)",
    # Gurobi size-limited (pip) licence / CPLEX Community Edition limits
    r"\bsize-limited\s+licen[cs]e",
    r"\bpromotional\s+version",
    # Gurobi "HostID mismatch"
    r"\bhostid\s+mismatch",
    # Gurobi token server / Web License Service
    r"\bno\s+tokens?\s+(?:available|left)",
    r"\b(?:connect|contact)\w*\s+(?:to\s+)?(?:the\s+)?token\s+server",
    r"\btoken\s+server\b[^\n]{0,60}?\b(?:error|unavailable|refused|failed|not\s+respond)",
    r"\bweb\s+licen[cs]e\s+service\b[^\n]{0,60}?\b(?:error|failed|unavailable|denied|invalid|expired)",
))


def _looks_like_license_error(*texts: str) -> bool:
    """Whether the solver output reports a licence failure (not merely mentions a licence)."""
    blob = "\n".join(t for t in texts if t)
    return any(p.search(blob) for p in _LICENSE_ERROR_PATTERNS)


#: A result-file status without a solution, and how the error names it.
_NO_SOLUTION_REASONS = {
    "INFEASIBLE": "is infeasible",
    "UNBOUNDED": "is unbounded",
    "INF_OR_UNBD": "is infeasible or unbounded",
    "TIME_LIMIT": "stopped at its time limit before finding a solution",
    "ITERATION_LIMIT": "stopped at its iteration limit before finding a solution",
    "NODE_LIMIT": "stopped at its node limit before finding a solution",
    "SOLUTION_LIMIT": "stopped at its solution limit before finding a solution",
    "WORK_LIMIT": "stopped at its work limit before finding a solution",
    "USER_OBJ_LIMIT": "stopped at its objective limit (BestObjStop or BestBdStop) before finding a solution",
    "CUTOFF": "found no solution better than its objective cutoff (Cutoff)",
    "OBJECTIVE_LIMIT": "stopped at an objective limit before finding a solution",
    "MEM_LIMIT": "stopped at its memory limit before finding a solution",
    "INTERRUPTED": "was interrupted before finding a solution",
    "UNFINISHED": "stopped unfinished before finding a solution",
    "NO_INTEGER_SOLUTION": (
        "found no integer solution (its log shows none; FlexTool reads how a MIP search ended from the Xpress log, "
        "which needs the controls OUTPUTLOG 1 and MIPLOG other than 0)"
    ),
}


def _is_sparse_result(solver_name: str, sol_path: Path) -> bool:
    """Whether *sol_path* is a Gurobi JSON result file (only non-zero values listed)."""
    if solver_name != "gurobi" or not sol_path.is_file():
        return False
    with sol_path.open("rb") as fh:
        return fh.read(64).lstrip().startswith(b"{")


def _warn_unknown_gurobi_parameters(
    output: str,
    option_layers: Mapping[str, str] | None,
    solve_name: str,
    logger: logging.Logger | None,
) -> None:
    """One warning per ``Unknown parameter: '<name>' (ignored)`` line of ``gurobi_cl``, naming where the
    parameter was set.  Gurobi solves on without it.  The line is part of Gurobi's console log, so it is not
    seen when the log is off (``LogToConsole 0`` / ``OutputFlag 0``)."""
    names = dict.fromkeys(_GUROBI_UNKNOWN_PARAMETER.findall(output))
    if not names:
        return
    from flextool.engine.solve._solver_dispatch import commercial_option_key

    layers = {commercial_option_key("gurobi", k): v for k, v in (option_layers or {}).items()}
    log = logger or logging.getLogger(__name__)
    for name in names:
        where = layers.get(commercial_option_key("gurobi", name))
        log.warning(
            "Gurobi does not know the parameter %r%s and solved solve %s without it. Check the spelling "
            "against the Gurobi parameter list.",
            name, f" (set in {where})" if where else "", solve_name,
        )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def resolve_basis_cache_dir(work_folder: Path | str | None, run_options: "RunOptions | None" = None) -> Path:
    """The warm-start basis cache directory; created if missing.

    ``--basis-cache-dir`` / ``FLEXTOOL_BASIS_CACHE_DIR`` (``run_options``; ``None`` reads the environment) >
    ``<work_folder>/basis_cache`` > ``<tmp>/flextool_basis_cache``.  The in-process ``.nbasis`` cache and the
    subprocess ``.bas`` cache share it.
    """
    from flextool.run_options import RunOptions

    options = run_options if run_options is not None else RunOptions.from_env()
    configured = options.value("basis_cache_dir")
    if configured:
        cache_dir = Path(configured)
    elif work_folder is not None:
        cache_dir = Path(work_folder) / "basis_cache"
    else:
        cache_dir = Path(tempfile.gettempdir()) / "flextool_basis_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


# ---------------------------------------------------------------------------
# Commercial solver process: live log, watchdog, process-tree stop (item 238)
# ---------------------------------------------------------------------------

#: How the solver is named in the ``Running <Solver> (<program>)`` line that opens its log block (the FlexTool
#: window greys the block from that line: ``gui/execution_window._SOLVER_BLOCK_STARTS``, pinned by a test).
_DISPLAY_NAMES: dict[str, str] = {
    "gurobi": "Gurobi",
    "cplex": "CPLEX",
    "xpress": "Xpress",
    "copt": "COPT",
}

#: After the solver exits, how long its log readers may still take: a process the solver started (a licence
#: heartbeat) can hold the output pipes open for ever.
_READER_JOIN_S = 10.0
#: How often the solver's child processes are listed while it runs (for the stop of an orphan).
_CHILD_SNAPSHOT_S = 1.0
#: Lines of the solver log a failure message carries when the log was not shown (``--solver-log-level silent``).
_LOG_TAIL_LINES = 20


class _SolverRun(NamedTuple):
    """What the solver process gave: exit code (``None`` when FlexTool stopped it), its output streams (the same
    text ``capture_output`` gave), every line in arrival order, and whether the watchdog stopped it."""

    returncode: int | None
    stdout: str
    stderr: str
    lines: list[str]
    timed_out: bool


def _pipe_reader(name: str, pipe: Any, out: "queue.Queue[tuple[str, str | None]]") -> None:
    """Put each line of *pipe* on *out* as ``(name, line)``, then ``(name, None)``.  Nothing else: every write to
    the console happens on the main thread."""
    try:
        for line in iter(pipe.readline, ""):
            out.put((name, line))
    except (OSError, ValueError):
        pass
    finally:
        out.put((name, None))
        try:
            pipe.close()
        except OSError:
            pass


def _stdin_writer(pipe: Any, text: str) -> None:
    """Write the command script to the solver and close its stdin.  A solver that exits without reading it
    (``BrokenPipeError``) is told by its exit code."""
    try:
        pipe.write(text)
    except (OSError, ValueError):
        pass
    finally:
        try:
            pipe.close()
        except (OSError, ValueError):
            pass


def _run_solver_process(
    argv: list[str],
    *,
    stdin_text: str | None,
    cwd: Path,
    watchdog: float | None,
    echo: bool,
    log_path: Path | None,
    logger: logging.Logger | None = None,
    label: str = "solver",
) -> _SolverRun:
    """Run a commercial solver's program and show its log as it writes it.

    Two reader threads put the stdout and stderr lines on one queue; this (main) thread takes them in arrival
    order, writes them to ``sys.stdout`` when *echo* (one whole line per write, flushed, so the run log copy and
    the FlexTool window read whole lines), to *log_path* (UTF-8) and to the buffers the parsers read.  A last line
    without a newline (a solver prompt, a killed solver) gets one on the console and in the file; the buffers keep
    the raw text.  The pipes are decoded with the locale's encoding, as ``text=True`` did, and an undecodable byte
    becomes U+FFFD instead of an error.

    The solver stays in FlexTool's process group (no session of its own), so a signal to that group (a closed
    terminal, a dropped SSH session, ``kill -- -<pgid>``) ends it with FlexTool.  It is stopped with everything it
    started when the *watchdog* (seconds) runs out, when this thread is interrupted (Ctrl-C) or fails, and when,
    :data:`_READER_JOIN_S` after the solver exited, its output is still open (a process it started holds it).
    """
    import locale
    import threading
    import time

    import psutil

    from flextool._process_tree import kill_solver_tree, pipe_inode, processes_holding_pipes

    encoding = locale.getpreferredencoding(False)
    proc: subprocess.Popen | None = None
    readers: list[threading.Thread] = []
    inodes: tuple[int | None, ...] = ()
    lines_q: "queue.Queue[tuple[str, str | None]]" = queue.Queue()
    buffers: dict[str, list[str]] = {"stdout": [], "stderr": []}
    lines: list[str] = []
    out = sys.stdout if echo else None
    log_fh = None
    known_children: dict[int, psutil.Process] = {}
    timed_out = False
    finished = False

    def _take(item: tuple[str, str | None]) -> None:
        nonlocal out, log_fh
        name, line = item
        if line is None:
            return
        buffers[name].append(line)
        lines.append(line)
        shown = line if line.endswith("\n") else line + "\n"
        if log_fh is not None:
            try:
                log_fh.write(shown)
                log_fh.flush()
            except (OSError, ValueError):
                log_fh = None
        if out is not None:
            try:
                out.write(shown)
                out.flush()
            except (OSError, ValueError, AttributeError):
                # A closed console (the FlexTool window was closed, ``pythonw``): keep reading for the parsers.
                out = None

    def _snapshot_children() -> None:
        try:
            for child in psutil.Process(proc.pid).children(recursive=True):
                known_children.setdefault(child.pid, child)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass

    def _leftovers() -> list[psutil.Process]:
        """The solver's children seen while it ran, and whatever holds its output open now."""
        return [*known_children.values(), *processes_holding_pipes(inodes)]

    def _reading() -> bool:
        return any(th.is_alive() for th in readers)

    def _take_until(deadline_at: float) -> None:
        """Show the log while a reader still runs (its pipe is open), until *deadline_at*."""
        while _reading() and time.monotonic() < deadline_at:
            try:
                _take(lines_q.get(timeout=0.2))
            except queue.Empty:
                pass

    try:
        # Inside the ``try``: an interrupt from here on stops the solver.
        proc = subprocess.Popen(
            argv,
            stdin=subprocess.PIPE if stdin_text is not None else subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=cwd,
            text=True,
            encoding=encoding,
            errors="replace",
            bufsize=1,
        )
        inodes = (pipe_inode(proc.stdout), pipe_inode(proc.stderr))
        readers = [
            threading.Thread(target=_pipe_reader, args=("stdout", proc.stdout, lines_q), daemon=True),
            threading.Thread(target=_pipe_reader, args=("stderr", proc.stderr, lines_q), daemon=True),
        ]
        threads = list(readers)
        if stdin_text is not None:
            threads.append(threading.Thread(target=_stdin_writer, args=(proc.stdin, stdin_text), daemon=True))
        for th in threads:
            th.start()
        if log_path is not None:
            try:
                log_fh = open(log_path, "w", encoding="utf-8", newline="")
            except OSError as exc:
                if logger is not None:
                    logger.warning("Could not write the %s log to %s: %s", label, log_path, exc)
        start = time.monotonic()
        deadline = start + watchdog if watchdog is not None else None
        last_snapshot = start - _CHILD_SNAPSHOT_S

        # While the solver runs.
        while proc.poll() is None:
            try:
                _take(lines_q.get(timeout=0.2))
            except queue.Empty:
                pass
            now = time.monotonic()
            if now - last_snapshot >= _CHILD_SNAPSHOT_S:
                last_snapshot = now
                _snapshot_children()
            if deadline is not None and now > deadline and proc.poll() is None:
                timed_out = True
                kill_solver_tree(proc, known_children.values())
                break
        # After it exited (or was stopped): the readers end when nothing holds the output open any more.  The
        # bound is on them, not on the queue: a slow console still has lines to show of a closed output.
        _take_until(time.monotonic() + _READER_JOIN_S)
        if _reading():
            msg = (
                f"The output of {label} stayed open after it exited: a process it started is still running. "
                "FlexTool stopped it and stops reading the log there."
            )
            (logger or logging.getLogger(__name__)).warning(msg)
            kill_solver_tree(proc, _leftovers())
            _take_until(time.monotonic() + 2.0)
        # The rest of what was read, without a time limit (it is all on the queue).
        while True:
            try:
                _take(lines_q.get_nowait())
            except queue.Empty:
                break
        try:
            proc.wait(timeout=5.0)
        except subprocess.TimeoutExpired:
            pass
        finished = True
    finally:
        if not finished and proc is not None:
            # Interrupted (Ctrl-C) or failed here: never leave the solver running.
            kill_solver_tree(proc, _leftovers())
        if log_fh is not None:
            try:
                log_fh.close()
            except OSError:
                pass

    assert proc is not None
    return _SolverRun(
        returncode=None if timed_out else proc.returncode,
        stdout="".join(buffers["stdout"]),
        stderr="".join(buffers["stderr"]),
        lines=lines,
        timed_out=timed_out,
    )


def _solver_log_reference(run: _SolverRun, *, echoed: bool, log_path: Path | None) -> str:
    """The end of a failure message: where the solver's log is (shown above when it was echoed; else its last
    lines)."""
    where = f" and in {log_path}" if log_path is not None else ""
    if echoed:
        return f"Its log is above{where}." if run.lines else "It wrote no log."
    if not run.lines:
        return "It wrote no log."
    tail = "".join(run.lines[-_LOG_TAIL_LINES:])
    if not tail.endswith("\n"):
        tail += "\n"
    saved = f"\n(The whole log is in {log_path}.)" if log_path is not None else ""
    return f"The end of its log:\n{tail.rstrip(chr(10))}{saved}"


def solve_via_subprocess(
    problem: "Problem",
    solver_name: str,
    options: dict[str, Any] | None,
    *,
    solve_name: str,
    logger: logging.Logger | None = None,
    work_folder: Path | None = None,
    run_options: "RunOptions | None" = None,
    option_layers: Mapping[str, str] | None = None,
) -> "Solution":
    """Solve *problem* via the appropriate CLI subprocess; return a Solution.

    Parameters
    ----------
    problem
        The :class:`polar_high.Problem` to solve.  Its LP source is
        written to MPS via :meth:`Problem.write_mps(release=True)`,
        leaving the Problem in ``_released`` state.
    solver_name
        ``"highs"`` for HiGHS (routes to ``flextool.cli.cmd_solve_mps``)
        or one of ``"gurobi"`` / ``"cplex"`` / ``"xpress"`` / ``"copt"``
        for the commercial CLIs.
    options
        Effective solver options, objective-valued ones (targets, bounds,
        cutoffs, absolute gaps) in unscaled currency: they are mapped to the
        solver's value for *problem* before any file is written
        (:func:`flextool.engine.solve._objective_options.translate_for_solver`).
        For HiGHS, written as a HiGHS ``.opt``
        file ingested by the subprocess.  For commercial solvers, the
        already-resolved parameter set (name -> value, final order; see
        ``_solver_dispatch.resolve_commercial_options``), written next to
        the MPS and fed to the CLI via the per-solver mechanism.  Its
        native time limit also sets the subprocess watchdog
        (``2 x limit + 600 s``).
    solve_name
        Used to name the MPS / .opt / .sol files.
    logger
        Optional :class:`logging.Logger`.
    work_folder
        When given, intermediate files live under
        ``<work_folder>/solve_data/subprocess/`` for post-mortem.
        ``None`` uses a self-cleaning temp dir.
    run_options
        The run's operator options; the HiGHS path reads ``warm_start`` and
        ``basis_cache_dir`` from them, the commercial path
        ``solver_log_level`` (``silent`` hides the solver's log).  ``None``
        reads the environment.
    option_layers
        Commercial solvers: parameter name -> the layer that set it (option
        file path, ``solver_arguments of solve X``, a flag), named in the
        warning about a parameter the solver does not know.

    Returns
    -------
    polar_high.Solution
        Carries a :class:`_SolHighsShim` bound to the parsed
        primal/dual arrays and the recovered column/row names.
        Downstream writers see a uniform ``sol.highs`` shape regardless
        of which solver actually ran.
    """
    # Objective targets, bounds, cutoffs and absolute gaps are written in
    # unscaled currency; the solver reads them against the objective of the
    # MPS file (``scale_the_objective`` × the cost, without the objective
    # constant, and for HiGHS × ``2**user_objective_scale`` of these options).
    # Mapped here, once, before the option file is written; the child and the
    # CLIs apply the file as it is.
    if solver_name == "highs" or solver_name in _SCRIPTS:
        from flextool.engine.solve._objective_options import translate_for_solver

        options, _translated = translate_for_solver(
            problem, solver_name, "mps", options, solve_name=solve_name,
        )
    if solver_name == "highs":
        return _solve_highs_subprocess(
            problem, options, solve_name=solve_name, logger=logger,
            work_folder=work_folder, run_options=run_options,
        )
    if solver_name in _SCRIPTS:
        return _solve_commercial_subprocess(
            problem, solver_name, options, solve_name=solve_name,
            logger=logger, work_folder=work_folder, option_layers=option_layers,
            run_options=run_options,
        )
    raise ValueError(
        f"solve_via_subprocess: unknown solver_name={solver_name!r}; "
        f"expected 'highs' or one of {sorted(_SCRIPTS)}"
    )


# ---------------------------------------------------------------------------
# HiGHS subprocess path (the original ``solve_via_subprocess`` body)
# ---------------------------------------------------------------------------


def _solve_highs_subprocess(
    problem: "Problem",
    options: dict[str, Any] | None,
    *,
    solve_name: str,
    logger: logging.Logger | None,
    work_folder: Path | None,
    run_options: "RunOptions | None" = None,
) -> "Solution":
    """HiGHS-specific subprocess path (unchanged contract).

    Writes MPS via :meth:`Problem.write_mps` with GENERIC names
    (``emit_names=False``: column ``col_id=j`` is ``C{j+1:07d}``, constraint
    row ``i`` is ``R{i+2:07d}``), spawns :mod:`flextool.cli.cmd_solve_mps`,
    and parses the .sol back via :func:`_parse_highs_sol`, which places
    every value by its generic-name index into dense ``col_id`` / row-id
    arrays and raises :class:`SubprocessSolutionMismatch` when a name or a
    count does not line up.  The real constraint-row names come from the
    Problem's canonical matrix, taken before the release; the column names
    from the ``_vars`` that survive it (:func:`_col_names_from_vars`).  Real names never
    go through the whitespace-delimited MPS / .sol, so an entity name with
    a space (``Battery Farm``) cannot split into a different column.  The
    arrays are wrapped in :class:`_SolHighsShim` — no parent-side
    ``highspy.Highs.readModel`` (which used to dominate cold-path RSS).

    **Warm-start arm** (opt-in, active only when the save-memory
    subprocess path runs *and* ``--warm-start`` / ``FLEXTOOL_WARM_START=1`` is set): a native
    HiGHS ``.bas`` basis is cached in a persistent directory that
    survives the per-solve ``out_dir`` cleanup, keyed by the MPS
    structural fingerprint (``problem._last_mps_fingerprint``).  On a
    subsequent solve of the same structural model the cached basis is
    injected via the child's ``--warm-basis`` (readBasis); every
    warm-start run also captures a fresh basis (``--basis`` to a unique
    tmp, atomically published via ``os.replace`` on success) and a stats
    sidecar (``--stats``) for later measurement.  Every step is
    defensive: a missing fingerprint, a rejected/garbage basis, or any
    exception falls back to a correct cold solve.  Warm-start never
    breaks a solve.  This ``.bas`` cache is a separate branch from the
    in-memory ``WarmProblem`` reuse gate in ``_orchestration.py``.

    **First-transfer A/B gate** (Landmine B): a supplied basis disables
    HiGHS' presolve, so a stale/far seed can make the warm solve SLOWER
    than a cold (presolve-on) one.  On a fingerprint's FIRST warm
    transfer (cache hit, no verdict marker yet) we run one extra cold
    timing PROBE child (throwaway ``.probe.sol`` / ``.probe.stats.json``,
    no ``--warm-basis``) alongside the normal warm main run, then compare
    wall times (``run_time`` from each stats sidecar).  Two marker files
    in the cache dir record the verdict: ``<fp>.abtested`` (warm within
    ``(1 + _WARM_AB_MARGIN)`` of cold → keep injecting, never re-probe)
    or ``<fp>.nowarm`` (warm regressed → solve cold from now on, still
    refreshing the captured basis).  If the probe can't produce a time we
    write ``.abtested`` and proceed warm (no perpetual re-probing).  The
    main run's Solution is always returned unchanged — the probe only
    informs the decision, so the returned result is correct regardless of
    the timing verdict.  Marker writes are best-effort and non-fatal.
    """
    from polar_high import Solution

    from flextool.run_options import RunOptions

    if run_options is None:
        run_options = RunOptions.from_env()
    from flextool._thread_budget import run_budget

    child_threads = run_budget(run_options)[0]
    cleanup = work_folder is None
    if work_folder is not None:
        out_dir = Path(work_folder) / "solve_data" / "subprocess"
        out_dir.mkdir(parents=True, exist_ok=True)
    else:
        out_dir = Path(tempfile.mkdtemp(prefix="flextool_subprocess_"))

    # One file-name token strictly inside ``out_dir`` (no ``/``, ``\\`` or ``..`` escapes), as the commercial path.
    safe_name = re.sub(r"[^\w.\-]", "_", solve_name) or "solve"
    mps_path = out_dir / f"{safe_name}.mps"
    sol_path = out_dir / f"{safe_name}.sol"
    opts_path = out_dir / f"{safe_name}.opt"
    options_json = out_dir / f"{safe_name}.options.json"
    gap_path = out_dir / f"{safe_name}.mipgap"

    # Warm-start basis-cache slots (populated only with warm start on;
    # declared here so the ``finally`` cleanup can reference them safely).
    warm_cached: Path | None = None
    warm_tmp_bas: Path | None = None
    warm_stats_path: Path | None = None
    # First-transfer A/B gate slots.
    warm_probe_sol: Path | None = None
    warm_probe_stats: Path | None = None
    warm_nowarm_marker: Path | None = None
    warm_abtested_marker: Path | None = None
    warm_first_transfer = False
    warm_cold_time: float | None = None
    warm_probe_ended = True
    warm_run_time: float | None = None

    try:
        _write_highs_option_files(_highs_option_pairs(options), opts_path, options_json)

        if logger is not None:
            logger.info(
                "save_memory: building LP for %r, writing MPS to %s",
                solve_name, mps_path,
            )

        # The LP's column count and real constraint-row names, in the exact
        # order write_mps writes them (``canonicalise`` is cached, so
        # write_mps reuses this matrix).  The row names must be taken now:
        # the release drops ``_cstrs`` and the matrix.  They wait out the
        # child solve as one compact Arrow string column rather than a
        # Python list.  Column names survive the release in ``_vars`` and
        # are rebuilt after the solve.
        import polars as pl

        canonical = problem.canonicalise()
        n_lp_cols = int(canonical.n_cols)
        real_row_names = pl.Series("row_names", canonical.row_names, dtype=pl.String)
        del canonical
        # Generic, whitespace-safe names in the file (see the docstring);
        # the solution is mapped back by generic-name index below.
        # The MPS file carries no objective constant (``add_obj_constant``);
        # the solver's objective leaves it out and the Solution adds it back.
        obj_offset = float(getattr(problem, "_obj_offset", 0.0) or 0.0)
        problem.write_mps(str(mps_path), release=True, emit_names=False)

        cmd = [
            *python_module_argv("flextool.cli.cmd_solve_mps"),
            "--mps", str(mps_path),
            "--solution", str(sol_path),
            "--options-json", str(options_json),
            "--mip-gap", str(gap_path),
            # The run's thread budget sizes the child's polars pool (read before polars is imported); the
            # child's HiGHS threads come from the options.
            "--threads", str(child_threads),
        ]

        # --- Warm-start arm (opt-in) ------------------------------------
        # Keyed by the MPS structural fingerprint.  Everything here is
        # defensive: on any hiccup we drop the warm slots and solve cold.
        # The fingerprint is polar-high's hash of the REAL name-set (also
        # with ``emit_names=False``, polar-high >= 3.12.4).  The cached
        # ``.bas`` carries the MPS's generic names, which HiGHS' readBasis
        # matches by name, i.e. by col_id / row position: a hit needs the
        # same name-set built in the same order, which the deterministic
        # model build gives.  A basis saved by an older FlexTool (real
        # names, ``phbasis-v1`` key) misses once and is replaced.
        if run_options.value("warm_start"):
            try:
                fp = getattr(problem, "_last_mps_fingerprint", None)
                if fp is None:
                    if logger is not None:
                        logger.debug(
                            "save_memory: no MPS fingerprint — warm-start "
                            "skipped for %r", solve_name,
                        )
                else:
                    cache_dir = resolve_basis_cache_dir(work_folder, run_options)
                    warm_cached = cache_dir / f"{fp}.bas"
                    warm_nowarm_marker = cache_dir / f"{fp}.nowarm"
                    warm_abtested_marker = cache_dir / f"{fp}.abtested"
                    warm_stats_path = out_dir / f"{safe_name}.stats.json"

                    if warm_cached.exists():
                        if warm_nowarm_marker.exists():
                            # This family regressed on its first A/B —
                            # never inject the basis (solve cold); still
                            # refresh the captured basis below.
                            if logger is not None:
                                logger.info(
                                    "save_memory: warm-start disabled "
                                    "(.nowarm) for %s — solving cold", fp,
                                )
                        elif warm_abtested_marker.exists():
                            # Verdict already recorded as beneficial —
                            # trust it, inject warm, no A/B probe.
                            cmd += ["--warm-basis", str(warm_cached)]
                            if logger is not None:
                                logger.info(
                                    "save_memory: warm-basis cache hit %s "
                                    "(.abtested — trusting warm)", fp,
                                )
                        else:
                            # FIRST warm transfer for this fingerprint: run
                            # a cold timing probe now, then inject warm for
                            # the main run and decide the verdict afterwards.
                            warm_first_transfer = True
                            warm_probe_sol = (
                                out_dir / f"{safe_name}.probe.sol"
                            )
                            warm_probe_stats = (
                                out_dir / f"{safe_name}.probe.stats.json"
                            )
                            warm_cold_time, warm_probe_ended = _run_cold_probe(
                                mps_path=mps_path,
                                options_json=options_json,
                                probe_sol=warm_probe_sol,
                                probe_stats=warm_probe_stats,
                                logger=logger,
                                threads=child_threads,
                            )
                            cmd += ["--warm-basis", str(warm_cached)]
                            if logger is not None:
                                logger.info(
                                    "save_memory: warm-basis cache hit %s "
                                    "(first transfer — A/B probing, "
                                    "cold_time=%s)", fp, warm_cold_time,
                                )
                    # Always refresh the cache this run: capture a fresh
                    # basis to a per-process tmp (atomic publish on success),
                    # plus a stats sidecar for later measurement.
                    warm_tmp_bas = cache_dir / f"{fp}.bas.tmp.{os.getpid()}"
                    cmd += [
                        "--basis", str(warm_tmp_bas),
                        "--stats", str(warm_stats_path),
                    ]
            except Exception as exc:  # noqa: BLE001 - fail safe to cold
                if logger is not None:
                    logger.warning(
                        "save_memory: warm-start setup failed (%s) — "
                        "solving cold", exc,
                    )
                warm_cached = warm_tmp_bas = warm_stats_path = None
                warm_probe_sol = warm_probe_stats = None
                warm_first_transfer = False

        if logger is not None:
            logger.info(
                "save_memory: spawning subprocess HiGHS for %r", solve_name,
            )
        # The work folder persists across runs and same-named solves: a solution, gap or stats file left by an
        # earlier one must not be read as this solve's when the child writes none (killed, crashed; the gap
        # write is best-effort).
        _remove_old_results([sol_path, gap_path, warm_stats_path])
        cp = subprocess.run(cmd, env=child_env())
        # Exit 0 is optimal and 1 not optimal (``cmd_solve_mps``); any other code is a crash or a signal
        # (POSIX: negative).  Exit 1 is also an uncaught exception in the child, and on Windows a
        # ``taskkill /F`` / Task Manager stop: without a solution file it is a failure too.  Checked before
        # the warm-start steps, so a stopped run neither publishes a basis nor records an A/B verdict.
        if cp.returncode not in (0, 1):
            failure = _describe_child_exit(cp.returncode)
        elif not sol_path.is_file():
            failure = (
                f"ended without writing a solution (it crashed, for example out of memory, or was stopped; "
                f"exit code {cp.returncode})"
            )
        else:
            failure = None
        if failure is not None:
            if warm_tmp_bas is not None:
                try:
                    warm_tmp_bas.unlink(missing_ok=True)
                except OSError:
                    pass
            kept = (
                f"MPS+options preserved at {out_dir} for inspection" if not cleanup
                else "its temporary files are removed"
            )
            raise RuntimeError(
                f"the HiGHS child (--save-memory) for solve {solve_name!r} {failure}; {kept}"
            )
        optimal = cp.returncode == 0

        # --- Warm-start publish + measurement ---------------------------
        if warm_tmp_bas is not None:
            try:
                if cp.returncode == 0 and warm_tmp_bas.exists():
                    # Atomic publish (write-tmp-then-rename); tmp and cache
                    # share a directory so this is a same-FS rename.
                    os.replace(str(warm_tmp_bas), str(warm_cached))
                    if logger is not None:
                        logger.info(
                            "save_memory: published warm-basis cache %s",
                            warm_cached.name,
                        )
                else:
                    # Failed/partial solve — never publish; drop the tmp.
                    if warm_tmp_bas.exists():
                        warm_tmp_bas.unlink()
            except Exception as exc:  # noqa: BLE001 - non-fatal
                if logger is not None:
                    logger.warning(
                        "save_memory: warm-basis publish failed (%s) — "
                        "cache not updated", exc,
                    )
                try:
                    if warm_tmp_bas.exists():
                        warm_tmp_bas.unlink()
                except OSError:
                    pass
        if warm_stats_path is not None:
            try:
                if warm_stats_path.exists():
                    stats = json.loads(warm_stats_path.read_text())
                    rt = stats.get("run_time")
                    warm_run_time = float(rt) if rt is not None else None
                    if logger is not None:
                        logger.info(
                            "save_memory: warm-start stats for %r — "
                            "simplex_iters=%s warm_basis_used=%s run_time=%s",
                            solve_name,
                            stats.get("simplex_iteration_count"),
                            stats.get("warm_basis_used"),
                            warm_run_time,
                        )
            except Exception as exc:  # noqa: BLE001 - measurement only
                if logger is not None:
                    logger.warning(
                        "save_memory: warm-start stats parse failed (%s)",
                        exc,
                    )

        # --- First-transfer A/B verdict ---------------------------------
        # Decide whether warm-start helped this family.  A supplied basis
        # disables presolve, so compare wall times: keep injecting only
        # when warm is within (1 + margin) of cold, else mark ``.nowarm``.
        # No usable timing → trust warm and stop re-probing (``.abtested``).
        if warm_first_transfer and not warm_probe_ended:
            # The cold probe was stopped or crashed: no measurement, so no
            # verdict; the next solve of this family probes again.
            if logger is not None:
                logger.warning(
                    "save_memory: the cold A/B probe for %s did not finish — "
                    "no warm-start verdict recorded", fp,
                )
        elif warm_first_transfer:
            try:
                cold_t = warm_cold_time
                warm_t = warm_run_time
                if (
                    cold_t is not None and cold_t >= 0.0
                    and warm_t is not None and warm_t >= 0.0
                ):
                    if warm_t > cold_t * (1.0 + _warm_ab_margin()):
                        _touch_marker(warm_nowarm_marker, logger)
                        if logger is not None:
                            logger.info(
                                "warm-start regressed for %s "
                                "(warm=%ss vs cold=%ss) — disabling future "
                                "injection", fp, warm_t, cold_t,
                            )
                    else:
                        _touch_marker(warm_abtested_marker, logger)
                        if logger is not None:
                            logger.info(
                                "save_memory: warm-start confirmed "
                                "beneficial for %s (warm=%ss vs cold=%ss)",
                                fp, warm_t, cold_t,
                            )
                else:
                    # Probe gave no usable time — trust warm, don't re-probe.
                    _touch_marker(warm_abtested_marker, logger)
                    if logger is not None:
                        logger.debug(
                            "save_memory: A/B probe produced no usable time "
                            "for %s — trusting warm (.abtested)", fp,
                        )
            except Exception as exc:  # noqa: BLE001 - gating is best-effort
                if logger is not None:
                    logger.warning(
                        "save_memory: warm-start A/B gating failed (%s) — "
                        "no verdict recorded", exc,
                    )
        if logger is not None:
            logger.info(
                "save_memory: subprocess complete (exit=%d, optimal=%s); "
                "reading solution from %s",
                cp.returncode, optimal, sol_path,
            )

        # Parse the .sol directly — no parent-side ``Highs.readModel``.
        # On large LPs the old path's ``readModel(mps)`` spiked +33 GB
        # of RSS purely to satisfy ``allVariableNames()`` / ``getSolution()``
        # on downstream writers.  The .sol carries the primal + duals under
        # the generic names; each value goes to the col_id / row id its
        # name encodes, and any name or count that does not line up with
        # the LP raises (never a silent shift onto other variables).
        col_names = _col_names_from_vars(problem)
        if len(col_names) != n_lp_cols:
            raise SubprocessSolutionMismatch(
                f"solve {solve_name!r}: the variables name {len(col_names)} "
                f"columns but the LP written to {mps_path} has {n_lp_cols}"
            )
        row_names = real_row_names.to_list()
        del real_row_names
        col_value, col_dual, row_dual = _parse_highs_sol(
            sol_path, len(col_names), len(row_names), require_primal=optimal,
        )
        obj = _read_objective_from_sol(sol_path)
        h = _SolHighsShim(
            col_names=col_names,
            row_names=row_names,
            col_value=col_value,
            col_dual=col_dual,
            row_dual=row_dual,
            objective=obj,
        )

        sol = Solution(
            optimal=optimal,
            obj=obj + obj_offset,
            col_value=col_value,
            row_dual=row_dual,
            col_dual=col_dual,
            col_names=col_names,
            row_names=row_names,
            vars=dict(problem._vars),
            highs=h,
        )
        _set_objective_constant(sol, obj_offset)
        # The child exits 0 for kOptimal, 1 otherwise; a non-optimal solve
        # is told apart by the status and primal lines of its solution file
        # (``Solution limit reached`` + ``Feasible``: a MIP stopped at its
        # node limit with an incumbent).  The acceptance check judges a limit
        # as in-process HiGHS's and records the solve's model status.
        if optimal:
            sol.shim_status, sol.shim_primal_feasible = "OPTIMAL", True
        else:
            sol.shim_status, sol.shim_primal_feasible = _highs_sol_status(sol_path)
        sol.shim_solver = "highs"
        # The gap HiGHS stopped at, named in the limit warning.
        sol.shim_mip_gap = (
            _read_highs_mip_gap(gap_path) if sol.shim_status in LIMIT_SHIM_STATUSES else None
        )
        return sol
    finally:
        if cleanup:
            paths = [mps_path, sol_path, opts_path, options_json, gap_path]
            # The warm-start stats sidecar and A/B probe throwaways live
            # inside ``out_dir`` — sweep them too so the ``rmdir`` below
            # succeeds.  The cache itself (``warm_cached`` / ``warm_tmp_bas``)
            # and the verdict markers live OUTSIDE ``out_dir`` and survive.
            if warm_stats_path is not None:
                paths.append(warm_stats_path)
            if warm_probe_sol is not None:
                paths.append(warm_probe_sol)
            if warm_probe_stats is not None:
                paths.append(warm_probe_stats)
            for p in paths:
                try:
                    if p.exists():
                        p.unlink()
                except OSError:
                    pass
            try:
                out_dir.rmdir()
            except OSError:
                pass


# ---------------------------------------------------------------------------
# Commercial-solver subprocess path (new)
# ---------------------------------------------------------------------------


def _solve_commercial_subprocess(
    problem: "Problem",
    solver_name: str,
    options: dict[str, Any] | None,
    *,
    solve_name: str,
    logger: logging.Logger | None,
    work_folder: Path | None,
    option_layers: Mapping[str, str] | None = None,
    run_options: "RunOptions | None" = None,
) -> "Solution":
    """Solve via a commercial solver's CLI binary.

    1. ``problem.write_mps(release=True, emit_names=False)`` produces the
       MPS via the cheap polars writer (peak ~2-3 GB on a 9.9 M-row LP),
       using GENERIC ``C0000001`` / ``R0000002`` names.  Free-format MPS
       is whitespace-delimited with no portable quoting, so real
       FlexTool entity names — which routinely contain spaces (e.g.
       ``Battery Farm``) — cannot go in the file without corrupting it.
       We keep the real names in memory instead and map back by index.
    2. Locate ``gurobi_cl`` / ``cplex`` / ``optimizer`` / ``copt_cmd``
       via :func:`_find_solver_binary`.
    3. Spawn the binary in the output folder with the per-solver argv +
       stdin script and bare file names (no path, so no console quoting
       rule matters); the stdin script is also written there as
       ``<solve>.script`` for a replay by hand.  After a
       ``Running <Solver> (<program>)`` line, the solver's log streams to
       ``sys.stdout`` as it writes it and is saved as ``<solve>.log``
       (:func:`_run_solver_process`); ``--solver-log-level silent`` hides
       it (no line, no echo) but it is still read and saved.  A failure
       message points to the log instead of repeating it.
    4. Parse the .sol with the per-solver parser; map its generic-named
       primal (and duals, for :data:`_DUAL_CAPABLE_SOLVERS`) back by
       *index* onto the real ``col_id`` / ``row_id`` names rebuilt from
       ``problem._vars`` (:func:`_col_names_from_vars`) and the pre-release
       ``_cstrs`` snapshot (:func:`_row_names_from_cstrs`).
    5. Wrap as a :class:`polar_high.Solution` backed by a
       :class:`_SolHighsShim` so the existing output writer paths Just
       Work — no parent-side ``highspy.Highs.readModel`` (which used to
       OOM on large LPs and hard-fail on entity names containing spaces).

    ``options`` is the resolved parameter set
    (``_solver_dispatch.resolve_commercial_options``: ``<solver>.opt`` <
    ``solver_mip_gap`` < ``solver_arguments`` < flags), written next to the
    MPS as the record of what was passed.  Each per-solver script builder
    turns it into the CLI's form (Gurobi:
    ``gurobi_cl <Name>=<value> ... ResultFile=<x>.json <mps>``;
    CPLEX/Xpress/COPT: inline ``set``/``setControl`` lines piped on stdin
    before the ``read``/``optimize`` pair).  The native time limit among
    them sets the watchdog (``2 x limit + 600 s``): the solver enforces the
    limit itself, the watchdog only catches a hung process.

    The parsed status (``Solution.shim_status``) names a solve stopped by a
    limit with a solution (``TIME_LIMIT``, ``NODE_LIMIT``, ``SOLUTION_LIMIT``,
    ...: Gurobi from its JSON result file, else its console; COPT / Xpress
    from the console; CPLEX from its solution file); ``Solution.shim_mip_gap``
    carries the gap the solver reported, when known, and
    ``Solution.shim_solver`` the solver.  The acceptance check decides what
    that means (``_acceptance.classify_acceptance``).
    """
    from polar_high import Solution

    from flextool import run_record
    from flextool.engine.solve._solver_dispatch import FlexToolUserError, local_solver_wrapper

    # This computer's solver wrapper (item 240 B2), checked before anything is written; again just before start.
    wrapper = local_solver_wrapper(solver_name)
    binary = _find_solver_binary(solver_name)
    if binary is None:
        raise RuntimeError(
            f"{solver_name!r} CLI binary "
            f"({_BINARY_NAMES.get(solver_name)!r}) was not found on $PATH "
            f"or in the conventional install directories. Install the "
            f"solver and ensure its 'bin' directory is on $PATH."
        )
    via = f" (started through the solver wrapper {wrapper.program})" if wrapper is not None else ""

    cleanup = work_folder is None
    if work_folder is not None:
        out_dir = Path(work_folder) / "solve_data" / "subprocess"
        out_dir.mkdir(parents=True, exist_ok=True)
    else:
        out_dir = Path(tempfile.mkdtemp(prefix=f"flextool_{solver_name}_"))

    from flextool.engine.solve._solver_dispatch import (
        commercial_time_limit,
        commercial_watchdog_seconds,
    )

    # The solver runs in ``out_dir`` and gets bare file names: a folder name with a space, or a Windows path's
    # backslashes, then needs no quoting or escaping rule of any solver's console (none of which FlexTool can
    # check here).  The name keeps letters, digits, ``_``, ``.`` and ``-`` only: one plain token everywhere.
    safe_name = re.sub(r"[^\w.\-]", "_", solve_name) or "solve"
    mps_path = out_dir / f"{safe_name}.mps"
    # Gurobi writes a JSON result file for a ``.json`` name: it carries the
    # status, gap and solution count whatever the console logging.
    sol_path = out_dir / (f"{safe_name}.json" if solver_name == "gurobi" else f"{safe_name}.sol")
    opt_path = out_dir / f"{safe_name}.opt"

    time_limit = commercial_time_limit(solver_name, dict(options or {}))
    watchdog = commercial_watchdog_seconds(time_limit)

    try:
        if logger is not None:
            logger.info(
                "subprocess[%s]: building LP for %r, writing MPS to %s",
                solver_name, solve_name, mps_path,
            )
        # Capture the real constraint names BEFORE releasing.  Only the
        # dual-returning parsers need them (Gurobi/COPT/Xpress ResultFiles
        # carry primal values only), and ``write_mps(release=True)`` drops
        # ``_cstrs``.  Real *column* names survive release via ``_vars`` and
        # are rebuilt after the solve.
        real_row_names: list[str] = (
            _row_names_from_cstrs(problem)
            if solver_name in _DUAL_CAPABLE_SOLVERS else []
        )
        # Emit GENERIC, whitespace-safe names (``C0000001`` / ``R0000002`` …)
        # rather than the real entity-bearing names.  Free-format MPS is
        # whitespace-delimited with no portable quoting, so a real name
        # containing a space — common in FlexTool entity names, e.g. a node
        # inside ``v_flow[Battery Farm,…]`` — would split mid-token and the
        # solver silently mis-parses the column, returning a WRONG answer
        # (verified: obj changes, primal unrecoverable).  We don't need real
        # names in the file: the solution is mapped back by *index* below and
        # the real names come from the in-memory Problem.  This makes the
        # commercial path whitespace-agnostic, exactly like the in-process
        # HiGHS path.
        # The MPS file carries no objective constant (``add_obj_constant``);
        # the solver's objective leaves it out and the Solution adds it back.
        obj_offset = float(getattr(problem, "_obj_offset", 0.0) or 0.0)
        # Whether the problem has an integer column, taken before the release: Xpress needs ``mipoptimize`` for
        # a MIP, and its parser judges a MIP by the log.
        from flextool.engine.solve._solver_dispatch import problem_is_mip

        is_mip = problem_is_mip(getattr(problem, "_vars", None))
        problem.write_mps(str(mps_path), release=True, emit_names=False)

        # The resolved parameters as (name, value) pairs: the script builder emits one argument or line per
        # pair, and ``<solve>.opt`` records them next to the MPS (not written when there are none).
        params = commercial_params(options)
        _write_commercial_opt_file(params, opt_path)
        if solver_name == "xpress" and is_mip:
            from flextool.engine.solve._solver_dispatch import note_xpress_mip_log_overrides

            note_xpress_mip_log_overrides(xpress_mip_log_overrides(params), option_layers)

        argv, stdin_text = _SCRIPTS[solver_name](
            binary, Path(mps_path.name), Path(sol_path.name), params, is_mip=is_mip,
        )
        if stdin_text is not None:
            # The command script piped to the console (CPLEX / Xpress / COPT), next to the MPS: replay it by hand
            # with ``cd <this folder> && <solver program> < <solve>.script``.
            (out_dir / f"{safe_name}.script").write_text(stdin_text, encoding="utf-8")

        if logger is not None:
            logger.info(
                "subprocess[%s]: spawning %s for %r",
                solver_name, binary, solve_name,
            )
        from flextool.run_options import RunOptions

        opts = run_options if run_options is not None else RunOptions.from_env()
        echo = opts.value("solver_log_level") != "silent"
        log_path = None if cleanup else out_dir / f"{safe_name}.log"
        display = _DISPLAY_NAMES.get(solver_name, solver_name)
        running = f"Running {display} ({binary})"
        if wrapper is not None:
            # The files may have changed since the run's start: check the wrapper again (cheap), then start the
            # solver's program behind it.  The line still starts with ``Running <Solver>`` (the window's greying).
            wrapper = local_solver_wrapper(solver_name)
            if wrapper is None:
                raise FlexToolUserError(
                    f"The solver wrapper of {display} was removed during the run (solve {solve_name!r}); start the "
                    "run again.")
            argv = [*wrapper.argv, *argv]
            running += f" through the solver wrapper {wrapper.shown()} (from {wrapper.source})"
            run_record.active().note_solver_wrapper(solver_name, wrapper.record())
        if echo:
            print(running, flush=True)
        elif logger is not None:
            logger.info("%s", running)
        # An earlier run's result in the persistent work folder must not be read as this solve's when the
        # solver writes none (CPLEX or COPT without a solution, a crash or a wrapper that exits 0).  Every
        # name a parser probes: Xpress appends ``.slx`` / ``.prt`` to the base name, or replaces its suffix.
        _remove_old_results([
            sol_path,
            Path(str(sol_path) + ".slx"), Path(str(sol_path) + ".prt"),
            sol_path.with_suffix(".slx"), sol_path.with_suffix(".prt"),
        ])
        run = _run_solver_process(
            argv, stdin_text=stdin_text, cwd=out_dir, watchdog=watchdog, echo=echo,
            log_path=log_path, logger=logger, label=f"{display} ({binary}){via}",
        )
        log_ref = _solver_log_reference(run, echoed=echo, log_path=log_path)
        if run.timed_out:
            from flextool.engine.solve._solver_dispatch import _WATCHDOG_FACTOR, _WATCHDOG_GRACE_S

            raise RuntimeError(
                f"{solver_name!r}{via} did not stop at its own time limit of "
                f"{time_limit:g} s for solve {solve_name!r}: FlexTool stopped it "
                f"after {watchdog:g} s ({_WATCHDOG_FACTOR:g} x the limit + {_WATCHDOG_GRACE_S:g} s). {log_ref}"
            )

        stdout = run.stdout
        stderr = run.stderr
        if solver_name == "gurobi":
            _warn_unknown_gurobi_parameters(stdout + "\n" + stderr, option_layers, solve_name, logger)

        if run.returncode != 0:
            msg = (
                f"{solver_name!r} CLI for solve {solve_name!r}{via} exited with "
                f"returncode={run.returncode}. {log_ref}"
            )
            # MPS+sol preserved when work_folder was given; otherwise
            # the finally-block has already cleaned them up.
            if _looks_like_license_error(stdout, stderr):
                raise RuntimeError(f"LICENSE: {msg}")
            raise RuntimeError(msg)

        parsed = _PARSERS[solver_name](sol_path, stdout, is_mip=is_mip)
        status_str, objective, primal, dual = parsed.status, parsed.objective, parsed.primal, parsed.dual
        if primal is None:
            # A status the solver reported (infeasible, a limit) is the reason; the licence
            # check reads free log text, and a size-limited licence's banner (COPT: "No license
            # found. Starting COPT with size limitations") is not a failure.
            if status_str in _NO_SOLUTION_REASONS:
                raise RuntimeError(
                    f"{solver_name!r} reports that solve {solve_name!r} "
                    f"{_NO_SOLUTION_REASONS[status_str]}; there is no solution "
                    f"to use. {log_ref}"
                )
            if _looks_like_license_error(stdout, stderr):
                raise RuntimeError(
                    f"{solver_name!r} CLI for solve {solve_name!r}{via} returned "
                    f"no usable solution and its output reports a licence error. {log_ref}"
                )
            raise RuntimeError(
                f"{solver_name!r} CLI{via} produced no solution file we could "
                f"parse at {sol_path} for solve {solve_name!r}. {log_ref}"
            )

        optimal = status_str == "OPTIMAL"

        # Real, ``col_id``-indexed column names come from the surviving
        # ``_vars`` (never from the MPS — which now carries only generic
        # names).  The ``.sol`` primal is keyed by those generic names, so
        # we map each value back by *index*: ``C0000001`` → ``col_id`` 0.
        # No real (possibly space-bearing) name is ever round-tripped
        # through the whitespace-delimited MPS/.sol.
        col_names = _col_names_from_vars(problem)
        n_cols = len(col_names)
        col_value = np.zeros(n_cols, dtype=np.float64)
        primal_dict = primal or {}
        mapped = 0
        unrecognised = 0
        for gname, val in primal_dict.items():
            cid = _generic_col_id(gname)
            if cid is None or not (0 <= cid < n_cols):
                unrecognised += 1
                continue
            col_value[cid] = float(val)
            mapped += 1
        if unrecognised and logger is not None:
            logger.warning(
                "subprocess[%s]: %d/%d .sol entries had unrecognised "
                "(non-generic) column names for solve %r — ignored",
                solver_name, unrecognised, len(primal_dict), solve_name,
            )
        # A JSON result file lists only the non-zero values; a text one lists all.
        if mapped < n_cols and logger is not None and not _is_sparse_result(solver_name, sol_path):
            logger.warning(
                "subprocess[%s]: %d/%d columns absent from .sol for solve "
                "%r (defaulted to 0.0)",
                solver_name, n_cols - mapped, n_cols, solve_name,
            )

        # Duals: only the dual-capable parsers return a dict (Gurobi/COPT/
        # Xpress carry none).  Their ``.sol`` keys duals by the generic row
        # name (``R0000002`` → ``row_id`` 0); map onto the real constraint
        # names captured before release so downstream name-based dual
        # lookups keep working.
        row_names = real_row_names
        row_dual = np.zeros(len(row_names), dtype=np.float64)
        if dual and row_names:
            for gname, dv in dual.items():
                rid = _generic_row_id(gname)
                if rid is not None and 0 <= rid < len(row_names):
                    row_dual[rid] = float(dv)
        col_dual = np.zeros(n_cols, dtype=np.float64)

        # Wrap the parsed arrays in the same duck-typed shim the HiGHS
        # save-memory path uses.  Downstream writers consume
        # ``allVariableNames()`` / ``getSolution()`` / ``getLp().row_names_``
        # / ``passColName()`` identically to a live ``highspy.Highs`` — no
        # 33 GB sidecar ``readModel``, no MPS text round-trip.
        obj = objective if objective is not None else 0.0
        h = _SolHighsShim(
            col_names=col_names,
            row_names=row_names,
            col_value=col_value,
            col_dual=col_dual,
            row_dual=row_dual,
            objective=obj,
        )

        sol = Solution(
            optimal=optimal,
            obj=obj + obj_offset,
            col_value=col_value,
            row_dual=row_dual,
            col_dual=col_dual,
            col_names=col_names,
            row_names=row_names,
            vars=dict(problem._vars),
            highs=h,
        )
        _set_objective_constant(sol, obj_offset)
        # The status parsed from the solution file (OPTIMAL / INFEASIBLE /
        # UNBOUNDED / a limit / OTHER); the acceptance check records it
        # as the solve's model status (no HiGHS diagnostics exist here).
        # A primal is always there (a result without one is refused above).
        sol.shim_status = status_str
        sol.shim_primal_feasible = True
        sol.shim_solver = solver_name
        sol.shim_mip_gap = parsed.mip_gap
        return sol
    finally:
        if cleanup:
            # out_dir is a dedicated per-solve tempdir; remove it wholesale
            # so solver-written sidecars (Xpress' .slx/.prt, .hdr/.asc,
            # per-solver logs) don't leak or block an rmdir.
            shutil.rmtree(out_dir, ignore_errors=True)


__all__ = ["solve_via_subprocess"]
