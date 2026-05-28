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

* HiGHS path uses ``highspy.Highs.readSolution`` against the
  subprocess-written .sol.
* Commercial path parses the solver-native .sol with the parsers
  imported from polar-high's ``_mps_fallback`` (copied verbatim so the
  parsers stay free of the ``LpView`` materialisation polar-high uses
  upstream of them — we already wrote MPS via the cheap polars path).

In both cases the parent constructs a fresh, read-only
:class:`highspy.Highs` from the MPS and injects the parsed primal /
dual arrays via :meth:`setSolution`.  This gives downstream output
writers a uniform ``Solution.highs`` shape: they can keep doing
``h.allVariableNames() + h.getSolution()`` regardless of which solver
produced the result.

Loses warm-LP reuse for the cascade — ``write_mps(release=True)``
puts the Problem in ``_released`` state and it can't be resolved.
Already documented on the ``--save-memory`` flag.
"""
from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from polar_high import Problem, Solution


# ---------------------------------------------------------------------------
# HiGHS .opt formatting / .sol parsing helpers (existing)
# ---------------------------------------------------------------------------


def _format_opt_value(v: object) -> str:
    """Render *v* in HiGHS .opt file syntax (``key=value`` per line)."""
    if isinstance(v, bool):
        return "true" if v else "false"
    return str(v)


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
# directly (it already carries column / row names + primal + dual values)
# and wrap the arrays in a duck-typed shim that exposes exactly the API
# surface the writers use.  No 33 GB sidecar Highs instance.

def _parse_highs_sol(
    sol_path: Path,
) -> tuple[
    list[str], list[str],
    np.ndarray, np.ndarray, np.ndarray,
]:
    """Parse a HiGHS style=0 ``.sol`` file into the writer-facing arrays.

    Returns ``(col_names, row_names, col_value, col_dual, row_dual)``.
    All five live in-process at numpy / list scale — typically a few
    hundred MB even on the DES-scale LP, vs the +33 GB the equivalent
    ``Highs.readModel(mps)`` parent re-read used to cost.

    The HiGHS style=0 file format (see ``Highs::writeSolution``):

        Model status
        <status>

        # Primal solution values
        Feasible | Infeasible
        Objective <value>
        # Columns <N>
        <name> <value>
        ...
        # Rows <M>
        <name> <value>
        ...

        # Dual solution values
        Feasible | Infeasible
        # Columns <N>
        <name> <dual>
        ...
        # Rows <M>
        <name> <dual>
        ...

        # Basis ...   (ignored — basis statuses are not needed by the
                       parent-side output writers)

    Each ``<name> <value>`` line splits on whitespace; names with
    embedded spaces are not produced by HiGHS so the simple split is
    safe.
    """
    col_names: list[str] = []
    row_names: list[str] = []
    col_value_list: list[float] = []
    col_dual_list: list[float] = []
    row_value_list: list[float] = []   # not actually exposed; parse but discard
    row_dual_list: list[float] = []

    # Three-state parser:
    #   section   ∈ {"primal", "dual", None}
    #   bucket    ∈ {"col", "row", None}
    section: str | None = None
    bucket: str | None = None

    with open(sol_path) as fh:
        for raw in fh:
            line = raw.rstrip("\n")
            if not line.strip():
                continue
            if line.startswith("# Primal solution"):
                section, bucket = "primal", None
                continue
            if line.startswith("# Dual solution"):
                section, bucket = "dual", None
                continue
            if line.startswith("# Basis"):
                # Basis section ends the data we care about.
                break
            if line.startswith("# Columns"):
                bucket = "col"
                continue
            if line.startswith("# Rows"):
                bucket = "row"
                continue
            if line.startswith("#"):
                # Unrecognised comment header — keep current state.
                continue
            if (
                line.startswith("Model status")
                or line.startswith("Objective ")
                or line in ("Feasible", "Infeasible", "Unknown")
                or line.startswith("HiGHS_basis_file")
                or line in ("Valid", "None")
            ):
                continue
            # Data row: "<name> <value>"
            sp = line.rsplit(None, 1)
            if len(sp) != 2:
                continue
            name, val_s = sp
            try:
                val = float(val_s)
            except ValueError:
                continue
            if section == "primal" and bucket == "col":
                col_names.append(name)
                col_value_list.append(val)
            elif section == "primal" and bucket == "row":
                row_names.append(name)
                row_value_list.append(val)
            elif section == "dual" and bucket == "col":
                col_dual_list.append(val)
            elif section == "dual" and bucket == "row":
                row_dual_list.append(val)

    col_value = np.asarray(col_value_list, dtype=np.float64)
    col_dual = (
        np.asarray(col_dual_list, dtype=np.float64)
        if col_dual_list else np.zeros(len(col_value), dtype=np.float64)
    )
    row_dual = (
        np.asarray(row_dual_list, dtype=np.float64)
        if row_dual_list else np.zeros(len(row_names), dtype=np.float64)
    )
    return col_names, row_names, col_value, col_dual, row_dual


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
        # the .sol file.
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

    Lookup order: ``$PATH`` via :func:`shutil.which`, then a small set
    of conventional POSIX install dirs.  Returns ``None`` on miss; the
    caller raises a user-actionable error.
    """
    bin_name = _BINARY_NAMES.get(solver_name)
    if bin_name is None:
        return None
    found = shutil.which(bin_name)
    if found is not None:
        return Path(found)
    if os.name == "posix":
        for d in _POSIX_INSTALL_DIRS.get(solver_name, []):
            candidate = Path(d) / bin_name
            if candidate.is_file() and os.access(candidate, os.X_OK):
                return candidate
    return None


def _gurobi_script(
    binary: Path, mps_path: Path, sol_path: Path, opt_path: Path | None,
) -> tuple[list[str], str | None]:
    """``gurobi_cl [ReadParams=<opt>] ResultFile=<sol> <mps>``.

    When *opt_path* is given, Gurobi's native ``ReadParams=<file>`` slot
    loads our merged baseline+overlay parameters before the solve.
    """
    argv: list[str] = [str(binary)]
    if opt_path is not None:
        argv.append(f"ReadParams={opt_path}")
    argv.append(f"ResultFile={sol_path}")
    argv.append(str(mps_path))
    return argv, None


def _cplex_script(
    binary: Path, mps_path: Path, sol_path: Path, opt_path: Path | None,
) -> tuple[list[str], str]:
    """CPLEX interactive optimizer: pipe commands on stdin.

    When *opt_path* is given, each non-comment line is translated to
    ``set <name> <value>`` and emitted *before* the ``read``/``optimize``
    pair so parameter values are in effect when the model loads.
    """
    pre: list[str] = []
    if opt_path is not None:
        for name, value in _parse_native_opt_file(opt_path).items():
            pre.append(f"set {name} {value}")
    cmds = "\n".join(
        pre
        + [
            f"read {mps_path}",
            "optimize",
            f"write {sol_path} sol",
            "quit",
            "",
        ]
    )
    return [str(binary)], cmds


def _xpress_script(
    binary: Path, mps_path: Path, sol_path: Path, opt_path: Path | None,
) -> tuple[list[str], str]:
    """Xpress optimizer console script via stdin.

    When *opt_path* is given, each non-comment line is translated to
    ``setControl <NAME> <value>`` and emitted *before* ``readprob``/
    ``lpoptimize`` so controls are in effect for the solve.
    """
    pre: list[str] = []
    if opt_path is not None:
        for name, value in _parse_native_opt_file(opt_path).items():
            pre.append(f"setControl {name} {value}")
    cmds = "\n".join(
        pre
        + [
            f"readprob {mps_path}",
            "lpoptimize",
            f"writesol {sol_path}",
            "quit",
            "",
        ]
    )
    return [str(binary)], cmds


def _copt_script(
    binary: Path, mps_path: Path, sol_path: Path, opt_path: Path | None,
) -> tuple[list[str], str]:
    """COPT's ``copt_cmd`` script via stdin.

    When *opt_path* is given, each non-comment line is translated to
    ``set <ParamName> <value>`` and emitted *before* ``read``/``optimize``.
    """
    pre: list[str] = []
    if opt_path is not None:
        for name, value in _parse_native_opt_file(opt_path).items():
            pre.append(f"set {name} {value}")
    cmds = "\n".join(
        pre
        + [
            f"read {mps_path}",
            "optimize",
            f"write {sol_path}",
            "quit",
            "",
        ]
    )
    return [str(binary)], cmds


_SCRIPTS = {
    "gurobi": _gurobi_script,
    "cplex": _cplex_script,
    "xpress": _xpress_script,
    "copt": _copt_script,
}


# ---------------------------------------------------------------------------
# Per-solver baseline option-file handling
# ---------------------------------------------------------------------------
# Each commercial solver gets a baseline ``solver_config/<solver>.opt``
# (shipped with the FlexTool repo, user-editable).  The file is written
# in the *solver's own* parameter-file syntax — see the comment header
# in each baseline file for the exact format.  The scenario's
# ``solver_options`` dict (already keyed by native parameter names by
# ``build_solver_options`` in ``_solver_dispatch.py``) is then overlaid
# on top, taking precedence per key.  The merged dict is materialised
# to a temp file next to the MPS in the per-solver native format and
# fed to the CLI via the per-solver mechanism (Gurobi: ReadParams=
# argv slot; CPLEX/Xpress/COPT: inline set/setControl commands piped
# on stdin).  Format helpers below.


def _resolve_solver_config_dir() -> Path:
    """Resolve the directory holding ``<solver>.opt`` baseline files.

    Lookup order:

    1. ``$FLEXTOOL_SOLVER_CONFIG_DIR`` environment variable (matches the
       override hook used by the existing ``highs.opt`` resolution path
       in tests / CI).
    2. ``<cwd>/solver_config`` — the same default that
       :func:`flextool.engine_polars._orchestration.run_chain_from_db`
       uses for ``highs.opt``.

    Returns the resolved :class:`Path` regardless of whether it exists;
    callers check ``.is_file()`` on the specific solver file before
    parsing.
    """
    env = os.environ.get("FLEXTOOL_SOLVER_CONFIG_DIR")
    if env:
        return Path(env)
    return Path.cwd() / "solver_config"


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
    out: dict[str, str] = {}
    if not path.is_file():
        return out
    try:
        with path.open("r") as fh:
            for raw in fh:
                line = raw.strip()
                if not line or line.startswith("#"):
                    continue
                if " " not in line and "\t" not in line:
                    continue
                name, value = line.rsplit(maxsplit=1)
                out[name.strip()] = value.strip()
    except OSError:
        return {}
    return out


def _format_native_opt_value(v: object) -> str:
    """Render *v* for a native solver opt-file line.

    Booleans become ``1`` / ``0`` (Gurobi/COPT accept either word or
    int; the int form is portable across CPLEX/Xpress too).
    """
    if isinstance(v, bool):
        return "1" if v else "0"
    return str(v)


def _format_gurobi_opt(merged: dict[str, str]) -> str:
    """Format *merged* as a Gurobi ``.prm`` file body.

    One ``<ParamName> <value>`` per line.  Same syntax produced by
    ``gurobi_cl`` 's own ``-w`` parameter dump.
    """
    return "".join(
        f"{name} {_format_native_opt_value(value)}\n"
        for name, value in merged.items()
    )


def _format_cplex_opt(merged: dict[str, str]) -> str:
    """Format *merged* as our CPLEX baseline body.

    Same layout as the input file: ``<name> <value>`` per line, where
    ``<name>`` uses CPLEX' interactive ``set`` syntax (e.g.
    ``mip tolerances mipgap``).  The per-solver script builder translates
    each line into ``set <name> <value>`` for piping on stdin.
    """
    return "".join(
        f"{name} {_format_native_opt_value(value)}\n"
        for name, value in merged.items()
    )


def _format_xpress_opt(merged: dict[str, str]) -> str:
    """Format *merged* as our Xpress controls baseline body."""
    return "".join(
        f"{name} {_format_native_opt_value(value)}\n"
        for name, value in merged.items()
    )


def _format_copt_opt(merged: dict[str, str]) -> str:
    """Format *merged* as our COPT baseline body."""
    return "".join(
        f"{name} {_format_native_opt_value(value)}\n"
        for name, value in merged.items()
    )


_OPT_FORMATTERS = {
    "gurobi": _format_gurobi_opt,
    "cplex": _format_cplex_opt,
    "xpress": _format_xpress_opt,
    "copt": _format_copt_opt,
}


def _build_commercial_opt_file(
    solver_name: str,
    options: dict[str, Any] | None,
    out_path: Path,
    *,
    config_dir: Path | None = None,
) -> Path | None:
    """Merge baseline ``<solver>.opt`` with scenario *options* and write
    the result to *out_path*.

    Returns the path to the merged file when it has at least one entry,
    or ``None`` when both the baseline file is missing and the scenario
    *options* dict is empty (caller skips the per-solver opt-file slot
    entirely so the CLI doesn't get a ``ReadParams=`` to a no-op file).
    """
    config_dir = config_dir or _resolve_solver_config_dir()
    baseline_path = config_dir / f"{solver_name}.opt"
    merged: dict[str, str] = {
        k: str(v) for k, v in _parse_native_opt_file(baseline_path).items()
    }
    if options:
        for key, value in options.items():
            # Skip the FlexTool-internal ``time_limit`` knob — that's
            # forwarded as the subprocess timeout, not as a solver
            # parameter (Gurobi accepts ``TimeLimit`` which IS in the
            # native-name overlay; the friendly ``time_limit`` key never
            # appears here for the commercial path because
            # ``build_solver_options`` translates it first).
            if value is None:
                continue
            merged[str(key)] = _format_native_opt_value(value)
    if not merged:
        return None
    formatter = _OPT_FORMATTERS[solver_name]
    out_path.write_text(formatter(merged))
    return out_path


def _parse_gurobi_sol(
    path: Path,
) -> tuple[str, float | None, dict[str, float] | None, dict[str, float] | None]:
    """Parse a Gurobi ``.sol`` (key=value text).

    Returns ``(status_str, objective, primal_dict, dual_dict)`` where
    ``status_str`` is one of ``"OPTIMAL"`` / ``"OTHER"``.
    """
    if not path.is_file():
        return "OTHER", None, None, None
    objective: float | None = None
    primal: dict[str, float] = {}
    with path.open("r") as fh:
        for raw in fh:
            line = raw.strip()
            if not line:
                continue
            if line.startswith("#"):
                m = re.search(r"[Oo]bjective\s+value\s*=\s*([-\d.eE+inf]+)", line)
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
    return status, objective, (primal or None), None


def _parse_copt_sol(path: Path):
    """COPT's ``.sol`` matches Gurobi's key=value layout."""
    return _parse_gurobi_sol(path)


def _parse_cplex_sol(
    path: Path,
) -> tuple[str, float | None, dict[str, float] | None, dict[str, float] | None]:
    """Parse CPLEX XML ``.sol`` via :mod:`xml.etree.ElementTree`."""
    if not path.is_file():
        return "OTHER", None, None, None
    try:
        tree = ET.parse(path)
    except ET.ParseError:
        return "OTHER", None, None, None
    root = tree.getroot()
    if root.tag == "CPLEXSolutions":
        children = list(root)
        if not children:
            return "OTHER", None, None, None
        root = children[0]
    objective: float | None = None
    status = "OTHER"
    header = root.find("header")
    if header is not None:
        obj_str = header.get("objectiveValue")
        if obj_str is not None:
            try:
                objective = float(obj_str)
            except ValueError:
                pass
        status_str = (header.get("solutionStatusString") or "").lower()
        if "optimal" in status_str:
            status = "OPTIMAL"
        elif "infeasible" in status_str:
            status = "INFEASIBLE"
        elif "unbounded" in status_str:
            status = "UNBOUNDED"
        elif "time" in status_str:
            status = "TIME_LIMIT"
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
    if status == "OTHER" and primal:
        status = "OPTIMAL"
    return status, objective, (primal or None), (dual or None)


def _parse_xpress_sol(
    path: Path,
) -> tuple[str, float | None, dict[str, float] | None, dict[str, float] | None]:
    """Lenient Xpress tabular .sol parser."""
    if not path.is_file():
        return "OTHER", None, None, None
    objective: float | None = None
    primal: dict[str, float] = {}
    in_vars = False
    with path.open("r") as fh:
        for raw in fh:
            line = raw.strip()
            if not line:
                continue
            low = line.lower()
            if objective is None and "objective" in low:
                m = re.search(r"objective.*?[:=]?\s*([-\d.eE+]+)", line)
                if m:
                    try:
                        objective = float(m.group(1))
                    except ValueError:
                        pass
                continue
            if low.startswith("variables") or low.startswith("columns"):
                in_vars = True
                continue
            if low.startswith("rows") or low.startswith("constraints"):
                in_vars = False
                continue
            if not in_vars:
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            name = None
            rest: list[str] = []
            for i, tok in enumerate(parts):
                try:
                    float(tok)
                    continue
                except ValueError:
                    name = tok
                    rest = parts[i + 1:]
                    break
            if name is None:
                continue
            for tok in rest:
                try:
                    primal[name] = float(tok)
                    break
                except ValueError:
                    continue
    status = "OPTIMAL" if primal else "OTHER"
    return status, objective, (primal or None), None


_PARSERS = {
    "gurobi": _parse_gurobi_sol,
    "cplex": _parse_cplex_sol,
    "xpress": _parse_xpress_sol,
    "copt": _parse_copt_sol,
}


_LICENSE_HINTS = (
    "license",
    "licence",
    "no token",
    "token server",
    "wls",
)


def _looks_like_license_error(*texts: str) -> bool:
    blob = "\n".join(t for t in texts if t).lower()
    return any(h in blob for h in _LICENSE_HINTS)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def solve_via_subprocess(
    problem: "Problem",
    solver_name: str,
    options: dict[str, Any] | None,
    *,
    solve_name: str,
    logger: logging.Logger | None = None,
    work_folder: Path | None = None,
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
        Effective solver options.  For HiGHS, written as a HiGHS ``.opt``
        file ingested by the subprocess.  For commercial solvers,
        overlaid on top of the per-solver baseline at
        ``solver_config/<solver>.opt`` and fed to the CLI via the
        per-solver mechanism (see
        :func:`_build_commercial_opt_file`).  Raw options win on key
        collision.  ``time_limit`` is additionally honoured as the
        subprocess timeout.
    solve_name
        Used to name the MPS / .opt / .sol files.
    logger
        Optional :class:`logging.Logger`.
    work_folder
        When given, intermediate files live under
        ``<work_folder>/solve_data/subprocess/`` for post-mortem.
        ``None`` uses a self-cleaning temp dir.

    Returns
    -------
    polar_high.Solution
        Carries a live (read-only) ``highspy.Highs`` instance bound to
        the LP and to the parsed primal/dual values via
        :meth:`Highs.setSolution`.  Downstream writers see a uniform
        ``sol.highs`` shape regardless of which solver actually ran.
    """
    if solver_name == "highs":
        return _solve_highs_subprocess(
            problem, options, solve_name=solve_name, logger=logger,
            work_folder=work_folder,
        )
    if solver_name in _SCRIPTS:
        return _solve_commercial_subprocess(
            problem, solver_name, options, solve_name=solve_name,
            logger=logger, work_folder=work_folder,
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
) -> "Solution":
    """HiGHS-specific subprocess path (unchanged contract).

    Writes MPS via :meth:`Problem.write_mps`, spawns
    :mod:`flextool.cli.cmd_solve_mps`, parses the .sol back via
    :func:`_parse_highs_sol` + :class:`_SolHighsShim` — no parent-side
    ``highspy.Highs.readModel`` (which used to dominate cold-path RSS).
    """
    from polar_high import Solution

    cleanup = work_folder is None
    if work_folder is not None:
        out_dir = Path(work_folder) / "solve_data" / "subprocess"
        out_dir.mkdir(parents=True, exist_ok=True)
    else:
        out_dir = Path(tempfile.mkdtemp(prefix="flextool_subprocess_"))

    safe_name = solve_name.replace("/", "_").replace(" ", "_") or "solve"
    mps_path = out_dir / f"{safe_name}.mps"
    sol_path = out_dir / f"{safe_name}.sol"
    opts_path = out_dir / f"{safe_name}.opt"

    try:
        opts = options or {}
        with open(opts_path, "w") as f:
            for k, v in opts.items():
                f.write(f"{k}={_format_opt_value(v)}\n")

        if logger is not None:
            logger.info(
                "save_memory: building LP for %r, writing MPS to %s",
                solve_name, mps_path,
            )

        problem.write_mps(str(mps_path), release=True)

        cmd = [
            sys.executable, "-m", "flextool.cli.cmd_solve_mps",
            "--mps", str(mps_path),
            "--solution", str(sol_path),
            "--options", str(opts_path),
        ]
        if logger is not None:
            logger.info(
                "save_memory: spawning subprocess HiGHS for %r", solve_name,
            )
        cp = subprocess.run(cmd)
        optimal = cp.returncode == 0
        if cp.returncode > 1:
            raise RuntimeError(
                f"subprocess HiGHS for solve {solve_name!r} failed with "
                f"exit code {cp.returncode}; MPS+options preserved at "
                f"{out_dir} for inspection"
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
        # on downstream writers; the .sol file already carries names +
        # primal + duals, and the writers see the same shape via the
        # ``_SolHighsShim`` wrapper.
        col_names, row_names, col_value, col_dual, row_dual = (
            _parse_highs_sol(sol_path)
        )
        n_cols = len(col_value)
        if col_dual.size == 0:
            col_dual = np.zeros(n_cols, dtype=np.float64)
        obj = _read_objective_from_sol(sol_path)
        h = _SolHighsShim(
            col_names=col_names,
            row_names=row_names,
            col_value=col_value,
            col_dual=col_dual,
            row_dual=row_dual,
            objective=obj,
        )

        return Solution(
            optimal=optimal,
            obj=obj,
            col_value=col_value,
            row_dual=row_dual,
            col_dual=col_dual,
            col_names=col_names,
            row_names=row_names,
            vars=dict(problem._vars),
            highs=h,
        )
    finally:
        if cleanup:
            for p in (mps_path, sol_path, opts_path):
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
) -> "Solution":
    """Solve via a commercial solver's CLI binary.

    1. ``problem.write_mps(release=True)`` produces the MPS via the
       cheap polars writer (peak ~2-3 GB on a 9.9 M-row LP).
    2. Locate ``gurobi_cl`` / ``cplex`` / ``optimizer`` / ``copt_cmd``
       via :func:`_find_solver_binary`.
    3. Spawn the binary with the per-solver argv + stdin script
       (copied verbatim from polar-high's ``_mps_fallback``).
    4. Parse the .sol with the per-solver parser; map the primal dict
       (keyed by LP variable names) onto the HiGHS column order via a
       fresh read-only ``highspy.Highs`` constructed from the MPS.
    5. Wrap as a :class:`polar_high.Solution` with the populated HiGHS
       instance so the existing output writer paths Just Work.

    ``options`` is fed to the solver via a per-solver native opt-file
    materialised next to the MPS in the per-solve temp dir.  The file
    starts from the user-editable baseline at
    ``solver_config/<solver>.opt`` (shipped in the FlexTool repo) and
    overlays the scenario's *options* dict line-by-line (raw entries
    win on key collision — same semantics as
    :func:`build_solver_options`).  Each per-solver script builder then
    references the merged file in the cleanest native form
    (Gurobi: ``gurobi_cl ReadParams=<file> ...``; CPLEX/Xpress/COPT:
    inline ``set``/``setControl`` lines piped on stdin before the
    ``read``/``optimize`` pair).  ``time_limit`` is *also* honoured as
    the subprocess timeout when present in *options*.
    """
    import highspy
    from polar_high import Solution

    binary = _find_solver_binary(solver_name)
    if binary is None:
        raise RuntimeError(
            f"{solver_name!r} CLI binary "
            f"({_BINARY_NAMES.get(solver_name)!r}) was not found on $PATH "
            f"or in the conventional install directories. Install the "
            f"solver and ensure its 'bin' directory is on $PATH."
        )

    cleanup = work_folder is None
    if work_folder is not None:
        out_dir = Path(work_folder) / "solve_data" / "subprocess"
        out_dir.mkdir(parents=True, exist_ok=True)
    else:
        out_dir = Path(tempfile.mkdtemp(prefix=f"flextool_{solver_name}_"))

    safe_name = solve_name.replace("/", "_").replace(" ", "_") or "solve"
    mps_path = out_dir / f"{safe_name}.mps"
    sol_path = out_dir / f"{safe_name}.sol"
    opt_path = out_dir / f"{safe_name}.opt"

    # Pull a time_limit out of options if the caller forwarded one.
    time_limit: float | None = None
    if options:
        for key in ("time_limit", "TimeLimit", "timelimit", "maxtime"):
            v = options.get(key)
            if v is not None:
                try:
                    time_limit = float(v)
                    break
                except (TypeError, ValueError):
                    pass

    try:
        if logger is not None:
            logger.info(
                "subprocess[%s]: building LP for %r, writing MPS to %s",
                solver_name, solve_name, mps_path,
            )
        problem.write_mps(str(mps_path), release=True)

        # Merge baseline ``solver_config/<solver>.opt`` with the scenario
        # options dict (raw entries win) and write the result to
        # ``<solve>.opt`` for the per-solver CLI to ingest.  Returns
        # ``None`` when both sources are empty — the per-solver script
        # builders then skip the opt-file slot entirely.
        opt_arg = _build_commercial_opt_file(
            solver_name, options, opt_path,
        )

        argv, stdin_text = _SCRIPTS[solver_name](
            binary, mps_path, sol_path, opt_arg,
        )

        if logger is not None:
            logger.info(
                "subprocess[%s]: spawning %s for %r",
                solver_name, binary, solve_name,
            )
        try:
            cp = subprocess.run(
                argv,
                input=stdin_text,
                capture_output=True,
                text=True,
                check=False,
                timeout=time_limit,
            )
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(
                f"{solver_name!r} CLI exceeded time_limit={time_limit!r}s "
                f"for solve {solve_name!r}; captured stderr: {exc.stderr!r}"
            ) from exc

        stdout = cp.stdout or ""
        stderr = cp.stderr or ""

        if cp.returncode != 0:
            msg = (
                f"{solver_name!r} CLI for solve {solve_name!r} exited with "
                f"returncode={cp.returncode}.\n"
                f"--- stdout ---\n{stdout}\n--- stderr ---\n{stderr}"
            )
            # MPS+sol preserved when work_folder was given; otherwise
            # the finally-block has already cleaned them up.
            if _looks_like_license_error(stdout, stderr):
                raise RuntimeError(f"LICENSE: {msg}")
            raise RuntimeError(msg)

        status_str, objective, primal, dual = _PARSERS[solver_name](sol_path)
        if status_str == "OTHER" and not primal:
            if _looks_like_license_error(stdout, stderr):
                raise RuntimeError(
                    f"{solver_name!r} CLI for solve {solve_name!r} returned "
                    f"no usable solution and its output mentions licensing.\n"
                    f"--- stdout ---\n{stdout}\n--- stderr ---\n{stderr}"
                )
            raise RuntimeError(
                f"{solver_name!r} CLI produced no solution file we could "
                f"parse at {sol_path} for solve {solve_name!r}.\n"
                f"--- stdout ---\n{stdout}\n--- stderr ---\n{stderr}"
            )

        optimal = status_str == "OPTIMAL"

        # Read the MPS back into a fresh HiGHS so we can resolve LP
        # variable names → column indices and feed downstream writers
        # the same ``sol.highs`` shape as the HiGHS path.  This Highs
        # is read-only (no ``run()`` call here); peak RSS is just the
        # LP storage + the solution arrays we inject below.
        h = highspy.Highs()
        try:
            h.silent()
        except Exception:
            pass
        ok = (highspy.HighsStatus.kOk, highspy.HighsStatus.kWarning)
        if h.readModel(str(mps_path)) not in ok:
            raise RuntimeError(
                f"parent failed to read MPS back from {mps_path} after "
                f"{solver_name!r} subprocess solve",
            )

        col_names = list(h.allVariableNames())
        n_cols = len(col_names)
        col_value = np.zeros(n_cols, dtype=np.float64)
        primal_dict = primal or {}
        missing = 0
        for i, nm in enumerate(col_names):
            v = primal_dict.get(nm)
            if v is None:
                missing += 1
                continue
            col_value[i] = float(v)
        if missing and logger is not None:
            logger.warning(
                "subprocess[%s]: %d/%d primal values missing from .sol "
                "for solve %r (defaulted to 0.0)",
                solver_name, missing, n_cols, solve_name,
            )

        # Inject the recovered primal (and duals when available) into
        # the HiGHS instance so downstream writers' ``getSolution()``
        # calls see the values rather than zeros.  ``setSolution`` with
        # a ``HighsSolution`` is the documented HiGHS API for this.
        hs = highspy.HighsSolution()
        hs.col_value = col_value.tolist()
        hs.value_valid = True
        # row_dual ordering: HiGHS' internal row order, matched against
        # the parsed dual_dict by row name when we have one.
        n_rows = h.getNumRow()
        row_dual_arr = np.zeros(n_rows, dtype=np.float64)
        if dual:
            # HiGHS exposes row names via writeModel; but the cleanest
            # API here is ``getLp().row_names_``.  Fall back to
            # ``getRowByName`` per row if needed.
            try:
                lp = h.getLp()
                row_names = list(lp.row_names_)
            except Exception:
                row_names = []
            if len(row_names) == n_rows:
                for i, nm in enumerate(row_names):
                    v = dual.get(nm)
                    if v is not None:
                        row_dual_arr[i] = float(v)
                hs.row_dual = row_dual_arr.tolist()
                hs.dual_valid = True
        # col_dual: solver-specific (most commercial parsers don't read
        # reduced costs from .sol).  Leave zeros, ``Solution`` accepts
        # ``None`` and defaults itself.
        try:
            h.setSolution(hs)
        except Exception as exc:  # pragma: no cover — version-specific
            if logger is not None:
                logger.warning(
                    "subprocess[%s]: h.setSolution failed: %s — "
                    "downstream getSolution() will return zeros",
                    solver_name, exc,
                )

        return Solution(
            optimal=optimal,
            obj=objective if objective is not None else 0.0,
            col_value=col_value,
            row_dual=row_dual_arr,
            col_dual=np.zeros(n_cols, dtype=np.float64),
            col_names=col_names,
            row_names=[],
            vars=dict(problem._vars),
            highs=h,
        )
    finally:
        if cleanup:
            for p in (mps_path, sol_path, opt_path):
                try:
                    if p.exists():
                        p.unlink()
                except OSError:
                    pass
            try:
                out_dir.rmdir()
            except OSError:
                pass


__all__ = ["solve_via_subprocess"]
