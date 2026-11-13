"""Objective-valued solver options, written in unscaled currency.

Some solver options are objective values: a target or bound that stops the
solve (HiGHS ``objective_target`` / ``objective_bound``, Gurobi
``BestObjStop``), a cutoff, or an absolute gap (HiGHS ``mip_abs_gap``,
Gurobi ``MIPGapAbs``).  The user writes them in unscaled currency, the units
of the model's cost data.  The solver compares them with the objective it
works on: the cost multiplied by ``scale_the_objective`` (``build_flextool``
declares it with ``Problem.set_objective_scale``) and, for HiGHS, by
``2**user_objective_scale`` (autoscale Layer 3, or the user's own value).

polar-high (3.14.0 and later) owns the mapping and the list of options
(:mod:`polar_high.objective_options`); in-process HiGHS solves translate
inside polar-high.  This module wires the paths polar-high cannot see: the
option files of the ``--save-memory`` HiGHS child and of the commercial
CLIs (:func:`translate_for_solver`), the ``Solver args`` echo
(:func:`echo_values`) and the Benders check (:func:`check_benders_options`).

A polar-high without the mapping (an older editable checkout) still runs
every solve that sets none of these options; one that sets any is a
configuration error naming the polar-high version it needs, because passing
the currency value unchanged is the bug this module fixes.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import TYPE_CHECKING, Any

from flextool.engine.input._solve_state import FlexToolConfigError

if TYPE_CHECKING:  # pragma: no cover
    from polar_high import Problem

#: The polar-high release that maps objective-valued options.
POLAR_HIGH_OBJECTIVE_OPTIONS_VERSION = "3.14.0"

#: Units the user writes these options in (named in the echo and messages).
USER_UNITS = "unscaled currency"

# Every objective-valued option name of polar-high 3.14.0's table, folded by
# :func:`_legacy_fold`.  Used ONLY to recognise such an option when the
# installed polar-high has no mapping (to raise instead of passing a
# currency value unchanged); with polar-high >= 3.14.0 its own table decides.
_LEGACY_NAMES: tuple[str, ...] = (
    # HiGHS
    "objective_target", "objective_bound", "mip_abs_gap",
    # Gurobi
    "BestObjStop", "BestBdStop", "Cutoff", "MIPGapAbs", "PoolGapAbs",
    # CPLEX
    "mip tolerances lowercutoff", "mip tolerances uppercutoff",
    "mip limits lowerobjstop", "mip limits upperobjstop",
    "simplex limits lowerobj", "simplex limits upperobj",
    "mip tolerances absmipgap", "mip tolerances objdifference",
    "mip pool absgap", "mip polishafter absmipgap", "barrier limits objrange",
    # Xpress
    "MIPABSCUTOFF", "MIPADDCUTOFF", "MIPABSSTOP",
    # COPT
    "AbsGap",
)


def _legacy_fold(name: str) -> str:
    """A spelling-insensitive key: case-folded, one leading ``xprs_`` removed,
    every character that is not a letter or a digit dropped."""
    key = str(name).strip().casefold()
    if key.startswith("xprs_"):
        key = key[len("xprs_"):]
    return "".join(ch for ch in key if ch.isalnum())


_LEGACY_KEYS = frozenset(_legacy_fold(n) for n in _LEGACY_NAMES)


def _api():
    """:mod:`polar_high.objective_options`, or ``None`` when the installed
    polar-high predates the mapping."""
    try:
        from polar_high import Problem
        from polar_high import objective_options
    except ImportError:
        return None
    if not hasattr(Problem, "set_objective_scale") or not hasattr(Problem, "objective_transform"):
        return None
    return objective_options


def supported() -> bool:
    """Whether the installed polar-high maps objective-valued options."""
    return _api() is not None


def set_objective_scale(problem: "Problem", factor: float) -> None:
    """Declare on *problem* that its objective is *factor* × the cost in
    unscaled currency (no-op on a polar-high without the mapping)."""
    if hasattr(problem, "set_objective_scale"):
        problem.set_objective_scale(float(factor))


def _names(options: Mapping | Iterable | None) -> list[str]:
    if not options:
        return []
    if isinstance(options, Mapping):
        return [str(k) for k in options]
    return [str(item[0]) for item in options]


def objective_option_kind(solver: str, name: str) -> str | None:
    """``"value"`` (an objective level), ``"diff"`` (an objective difference)
    or ``None`` for option *name* of *solver*.  Without polar-high's mapping,
    any name in its 3.14.0 table reads as ``"value"`` (enough to refuse it)."""
    api = _api()
    if api is not None:
        if solver not in api.OBJECTIVE_OPTION_SOLVERS:
            return None
        return api.objective_option_kind(solver, name)
    return "value" if _legacy_fold(name) in _LEGACY_KEYS else None


def objective_valued(solver: str, options: Mapping | Iterable | None) -> list[str]:
    """The names in *options* that are objective-valued options of *solver*, in order."""
    return [n for n in _names(options) if objective_option_kind(solver, n) is not None]


def _unsupported_error(solver: str, names: list[str], solve_name: str | None) -> FlexToolConfigError:
    where = f"Solve '{solve_name}' sets" if solve_name else "The solve sets"
    return FlexToolConfigError(
        f"{where} the {solver} option(s) {', '.join(names)}, which are written in {USER_UNITS} and must be "
        f"scaled to the objective the solver sees. That needs polar-high {POLAR_HIGH_OBJECTIVE_OPTIONS_VERSION} "
        f"or later; the installed polar-high is older. Update polar-high, or remove these options."
    )


def require_supported(solver: str, options: Mapping | Iterable | None, *, solve_name: str | None = None) -> None:
    """Raise :class:`FlexToolConfigError` when *options* sets an objective-valued
    option of *solver* and the installed polar-high cannot map it.

    Called where a solve's HiGHS options are resolved
    (``_orchestration._solve_highs_options``), so the refusal holds on every
    in-process path, whatever prints or echoes follow; the option-file paths
    also meet it in :func:`translate_for_solver`."""
    if supported():
        return
    names = objective_valued(solver, options)
    if names:
        raise _unsupported_error(solver, names, solve_name)


def translate_for_solver(
    problem: "Problem",
    solver: str,
    io: str,
    options: Any,
    *,
    solve_name: str | None = None,
    factor_from_problem: bool = False,
) -> tuple[Any, list]:
    """*options* with the objective-valued options of *solver* mapped from
    unscaled currency to the solver's value for *problem*.

    *io*: ``"mps"`` for a solver that reads :meth:`Problem.write_mps`'s file
    (``--save-memory`` HiGHS, the commercial CLIs), ``"direct"`` for one that
    gets the model from polar-high (in-process HiGHS: polar-high translates
    itself, call this only for the echo).  For HiGHS the factor
    ``2**user_objective_scale`` comes from *options* itself (the dict that
    is applied) or, when *options* is ``None``, from the options stored on
    *problem*; *factor_from_problem* reads it from *problem* even when
    *options* is given (the echo, whose dict leaves out the engine and
    Layer-3 keys).  *options* is a dict or a list of ``(name, value, ...)``
    tuples; the input is never changed.  Returns ``(options, records)``,
    ``records`` the :class:`polar_high.objective_options.Translated` entries
    (empty when nothing changed).  Raises :class:`FlexToolConfigError` when
    an objective-valued option is set and the installed polar-high cannot
    map it.
    """
    if not objective_valued(solver, options):
        return options, []
    require_supported(solver, options, solve_name=solve_name)
    api = _api()
    n_options = options if (solver == "highs" and isinstance(options, Mapping) and not factor_from_problem) else None
    transform = problem.objective_transform(solver, io, n_options)
    return api.translate_objective_options(solver, options, transform)


def echo_values(records: Iterable) -> dict[str, str]:
    """``{name: text}`` for the ``Solver args`` line: the value the solver
    gets first (the line stays replicable), then the value the user wrote."""
    return {
        r.name: f"{r.solver_value:.12g} ({r.user_value:.12g} in {USER_UNITS})"
        for r in records
    }


def check_benders_options(
    options: Mapping[str, Any] | None,
    *,
    solve_name: str | None,
    layers: Mapping[str, str] | None = None,
) -> None:
    """Refuse a whole-model objective target / bound / cutoff on a Benders solve.

    The master and every region get the same HiGHS options, and an objective
    level of the whole model has no meaning for either part.  Absolute gaps
    (objective differences) are allowed: each Problem scales them to its
    own objective.  *layers* names where each option was set (the error
    says it).  Without polar-high's mapping any objective-valued option is
    refused (:func:`translate_for_solver`'s error).
    """
    names = objective_valued("highs", options)
    if not names:
        return
    require_supported("highs", options, solve_name=solve_name)
    levels = [n for n in names if objective_option_kind("highs", n) == "value"]
    if not levels:
        return
    layers = layers or {}
    parts = []
    for n in levels:
        layer = layers.get(n)
        parts.append(f"{n}={options[n]}" + (f" ({layer})" if layer else ""))
    one = len(parts) == 1
    what = "is a whole-model objective target or bound" if one else "are whole-model objective targets or bounds"
    subject = f"Solve '{solve_name}' uses decomposition=benders" if solve_name else "A Benders solve gets"
    alternative = f", or solve '{solve_name}' with decomposition=none" if solve_name else ""
    raise FlexToolConfigError(
        f"{subject}: {', '.join(parts)} {what}, which has no meaning for the Benders master or a region. "
        f"Remove {'it' if one else 'them'}{alternative}. Absolute gaps (mip_abs_gap) are allowed."
    )


__all__ = [
    "POLAR_HIGH_OBJECTIVE_OPTIONS_VERSION",
    "USER_UNITS",
    "check_benders_options",
    "echo_values",
    "objective_option_kind",
    "objective_valued",
    "require_supported",
    "set_objective_scale",
    "supported",
    "translate_for_solver",
]
