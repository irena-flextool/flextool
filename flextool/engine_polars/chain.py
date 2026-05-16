"""Thin compat shim around the native cascade — Δ.12e.

The legacy file-symlink driver (loading per-sub-solve
``solve_data_<sub>/`` snapshots directly from a work folder) retired
in Δ.12e once the native cascade in
:mod:`flextool.engine_polars._orchestration` reached feature parity:
warm-LP across structurally-compatible iterations (Δ.12d), in-memory
handoff carriers between sub-solves (Δ.11/Δ.12), full output writer
coverage (Δ.1, Δ.12c-fix), and override-chain authority for Direct /
Derived / Projection params (Δ.12-drop, Δ.12c, Δ.12c-fix2).

What's left here is a thin convenience wrapper that converts a
work-folder path (containing a ``tests.sqlite`` / ``input.sqlite``)
into the canonical native call
:func:`flextool.engine_polars._orchestration.run_chain_from_db` and
adapts the result to :class:`ChainStep` shape for backwards
compatibility with the few external callers that still import
``run_chain`` directly.

The warm-update primitives (structural fingerprint, Param classification,
:func:`_apply_warm_updates`) live in :mod:`flextool.engine_polars._warm`
and are re-exported here as module-level names for backwards
compatibility (e.g. ``from flextool.engine_polars.chain import
_MUTABLE_PARAMS`` in ``test_warm_param_autoupdate``).
"""
from __future__ import annotations

from pathlib import Path

from polar_high import Solution
from flextool.engine_polars._warm import (
    _STRUCTURAL_FIELDS,
    _WARM_PARAMS,
    _MUTABLE_PARAMS,
    _WARM_PARAMS_DEFERRED,
    _WARM_PARAMS_NO_OP,
    _WARM_PARAM_GATES,
    _IncompatibleUpdate,
    _fingerprint,
    _param_frame_equal,
    _param_values_position_equal,
    _gate_active,
    _apply_warm_updates,
    _build_warm_problem,
)


__all__ = ["run_chain", "ChainStep"]


class ChainStep:
    """Per-sub-solve result of :func:`run_chain`.

    Attributes
    ----------
    solve_name : str
        The sub-solve identifier (e.g. ``"y2025_5week"``).
    solution : flexpy.Solution
        The HiGHS solution for this sub-solve.
    handoff : flextool.SolveHandoff
        The flexpy-derived handoff carriers (see
        :func:`flextool.input.build_handoff_from_flexpy`).
    warm_used : bool
        True if this sub-solve was solved by warm-updating the prior
        sub-solve's :class:`WarmProblem` instance; False if it was a
        cold rebuild.  Always False for the first sub-solve and for
        ``warm=False`` runs.
    """

    __slots__ = ("solve_name", "solution", "handoff", "warm_used")

    def __init__(self, solve_name: str, solution: Solution, handoff,
                 warm_used: bool = False):
        self.solve_name = solve_name
        self.solution = solution
        self.handoff = handoff
        self.warm_used = warm_used

    def __repr__(self) -> str:  # pragma: no cover — debug-only
        return (f"ChainStep(solve_name={self.solve_name!r}, "
                f"obj={self.solution.obj!r}, "
                f"warm_used={self.warm_used}, "
                f"handoff_empty={self.handoff.is_empty()})")


# Same overrides used by the parity-sweep fixtures — see
# tests/engine_polars/test_solve_config_parity._discover_fixtures.  The
# keys are work_folder dirnames; values are the scenario names that
# produced those snapshots.
_NATIVE_SCENARIO_OVERRIDES: dict[str, str] = {
    "work_2day_stochastic_dispatch_full_storage": "2_day_stochastic_dispatch",
    "work_commodity_ladder_annual": "coal_ladder_annual",
    "work_commodity_ladder_cumulative": "coal_ladder_cumulative",
    "work_delay_source_coef": "water_pump_delayed",
    "work_inflation_check": "wind_battery_invest_lifetime_renew",
}


def _resolve_native_scenario(db_path: Path, work: Path) -> str | None:
    """Map ``work/`` dirname → scenario name using the same convention
    as the parity tests.  Returns ``None`` when no rule matches.
    """
    import re
    import spinedb_api as api

    if work.name in _NATIVE_SCENARIO_OVERRIDES:
        return _NATIVE_SCENARIO_OVERRIDES[work.name]

    scen_target = work.name.removeprefix("work_") if work.name.startswith("work_") else None
    if scen_target is None:
        return None

    try:
        with api.DatabaseMapping("sqlite:///" + str(db_path)) as db:
            scenarios = sorted(s.name for s in db.query(db.scenario_sq).all())
    except Exception:
        return None

    candidates = [scen_target]
    candidates.append(re.sub(r"(^|_)(\d+)([a-z])", r"\1\2_\3", scen_target))
    candidates.append(re.sub(r"(\d+)_([a-z])", r"\1\2", scen_target))
    if scen_target.endswith("_full_storage"):
        base = scen_target[: -len("_full_storage")]
        candidates.append(re.sub(r"(^|_)(\d+)([a-z])", r"\1\2_\3", base))
        candidates.append(base)
    for cand in candidates:
        if cand in scenarios:
            return cand
    return None


def run_chain(
    work_folder: Path | str,
    *,
    warm: bool = False,
    scenario: str | None = None,
) -> dict[str, ChainStep]:
    """Run a flextool multi-solve scenario end-to-end via the native
    cascade — convenience adapter that takes a work-folder path.

    Δ.12e — this function is now a thin shim that:

    1. Looks up ``tests.sqlite`` / ``input.sqlite`` under
       *work_folder*.
    2. Resolves the scenario name from the dir-name convention (see
       :data:`_NATIVE_SCENARIO_OVERRIDES`) or the explicit
       ``scenario`` kwarg / ``FLEXPY_NATIVE_SCENARIO`` env var.
    3. Delegates to
       :func:`flextool.engine_polars._orchestration.run_chain_from_db`.
    4. Adapts the result from
       :class:`flextool.engine_polars.OrchestrationStep` to
       :class:`ChainStep` shape (the legacy return type a few
       external callers still import).

    The previous file-symlink driver (loading
    ``solve_data_<sub>/`` snapshots directly from the work folder)
    and the ``use_handoff_overlay`` / ``native`` / ``chain`` kwargs
    retired in Δ.12e once the native cascade reached feature parity.
    Callers driving DB scenarios should prefer
    :func:`run_chain_from_db` directly.

    Parameters
    ----------
    work_folder : Path | str
        Directory containing a Spine SQLite (``tests.sqlite`` or
        ``input.sqlite``).  The native cascade re-runs flextool's
        preprocessing into a private tempdir; the work folder is
        consulted only for the DB.
    warm : bool, default False
        When True, attempt warm LP updates between consecutive
        structurally-compatible sub-solves using
        :class:`polar_high.WarmProblem`.  See
        :func:`flextool.engine_polars._orchestration.run_orchestration`
        for the per-iteration semantics.
    scenario : str | None, default None
        Explicit scenario name override.  ``None`` consults the
        ``FLEXPY_NATIVE_SCENARIO`` env var, then falls back to the
        ``work_<scenario>`` dir-name convention.

    Returns
    -------
    dict[str, ChainStep]
        Mapping ``solve_name → ChainStep(solve_name, solution, handoff,
        warm_used)``.

    Raises
    ------
    ValueError
        If no DB is found under *work_folder* or the scenario can't
        be determined.
    """
    import os
    work = Path(work_folder)
    db_path = None
    for cand in ("tests.sqlite", "input.sqlite"):
        p = work / cand
        if p.exists():
            db_path = p
            break
    if db_path is None:
        raise ValueError(
            f"run_chain: no DB found under {work} "
            f"(looked for tests.sqlite, input.sqlite).  The native "
            f"cascade requires a DB scenario."
        )

    # Late import to avoid a build-time cycle between chain and _orchestration.
    from flextool.engine_polars._orchestration import run_chain_from_db

    if scenario is None:
        scenario = os.environ.get("FLEXPY_NATIVE_SCENARIO") or None
    if scenario is None:
        scenario = _resolve_native_scenario(db_path, work)
    if scenario is None:
        raise ValueError(
            f"run_chain: cannot determine which scenario to run for "
            f"{work}.  Pass scenario= explicitly, set "
            f"FLEXPY_NATIVE_SCENARIO, or rename the work directory to "
            f"match the scenario name (work_<scenario>).  "
            f"work.name={work.name!r}"
        )

    # Phase C.5 — ``ChainStep`` keeps the per-step ``solution``
    # contract for legacy callers; opt into the full per-step state on
    # the underlying cascade.
    steps = run_chain_from_db(db_path, scenario, warm=warm, keep_solutions=True)
    out: dict[str, ChainStep] = {}
    for name, step in steps.items():
        out[name] = ChainStep(
            solve_name=name,
            solution=step.solution,
            handoff=step.handoff,
            warm_used=getattr(step, "warm_used", False),
        )
    return out
