"""End-to-end chain runner — cold rebuild + opt-in WarmProblem warm updates.

Drives a multi-solve scenario sub-solve by sub-solve in flexpy alone,
extracting an in-memory ``SolveHandoff`` after each sub-solve via
:func:`build_handoff_from_flexpy` and routing it into the next
sub-solve's ``FlexData`` via the loader's prior-handoff overlay.

Two execution modes:

* ``warm=False`` (default) — **cold rebuild** every sub-solve.  Each
  sub-solve gets a fresh :class:`polar_high.Problem` + HiGHS instance.
  Backward-compatible with the original ``run_chain`` behaviour.

* ``warm=True`` — **solve-type-aware** runner.  Builds a
  :class:`polar_high.WarmProblem` on the first sub-solve and, on each
  subsequent sub-solve, decides whether to warm-update the live LP
  (RHS / objective coefs only) or cold-rebuild from scratch.  The
  decision is made by comparing a structural fingerprint of each
  sub-solve's :class:`FlexData`: if two consecutive sub-solves share
  the same fingerprint AND the only Params that differ between them
  belong to the "clean-mapping" set (currently ``p_inflow``), the LP
  is warm-updated.  Any structural change OR any unmapped-Param diff
  triggers a cold rebuild and resets the warm state.

  This exploits flexpy's unique advantage over the GMPL pipeline in
  flextool — highspy's ``changeRowsBounds`` / ``changeColsCost`` lets
  us preserve the basis and the LP matrix across consecutive rolling
  sub-solves of, e.g., ``dispatch_fullYear_roll_roll_<i>``.

Today the per-sub-solve snapshots written by flextool's pre-run
(``solve_data_<sub>/``) already contain the prior solve's handoff
state baked into their CSVs (e.g. ``p_entity_previously_invested_capacity``,
``p_roll_continue_state``, ``fix_storage_quantity_<parent>.csv``).
``run_chain`` loads each snapshot AS IS — the snapshot is the source
of truth for handoff state.  The flexpy-derived handoff returned by
:func:`build_handoff_from_flexpy` is captured per sub-solve and
exposed via the returned dict for callers that want to compare
against flextool's writers, but it isn't used to override the
loader's read of the snapshot.

The warm-update primitives (structural fingerprint, Param classification,
:func:`_apply_warm_updates`) live in :mod:`flextool.engine_polars._warm`
so the native cascade in :mod:`flextool.engine_polars._orchestration`
can reuse them (Δ.12d).  This module re-exports them as module-level
names for backwards compatibility with existing callers (e.g.
``from flextool.engine_polars.chain import _MUTABLE_PARAMS``).
"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import polars as pl

from polar_high import Problem, Solution, WarmProblem
from flextool.engine_polars.input import load_flextool, build_handoff_from_flexpy
from flextool.engine_polars.model import build_flextool
from flextool.engine_polars._input_source import _read_csv_file
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

if TYPE_CHECKING:
    from flextool.engine_polars.input import FlexData


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


def _read_chain_order(work_folder: Path) -> list[str]:
    """Return the ordered list of sub-solve names from
    ``input/model__solve.csv``.  Order matches CSV row order.
    """
    msv = work_folder / "input" / "model__solve.csv"
    if not msv.exists():
        # Fall back to any solve_data_<name>/ dirs sorted alphabetically.
        dirs = sorted(
            d.name[len("solve_data_"):] for d in work_folder.iterdir()
            if d.is_dir() and d.name.startswith("solve_data_")
            and d.name != "solve_data"
        )
        return dirs
    df = _read_csv_file(msv)
    if "solve" in df.columns:
        return df["solve"].cast(pl.Utf8).to_list()
    # Schema fallback — flextool's column may be different in some fixtures.
    return [str(v) for v in df[df.columns[-1]].to_list()]


def _stage_subsolve_workdir(
    work_folder: Path, sub_solve: str, tmpdir: Path,
) -> Path:
    """Build a per-sub-solve view of ``work_folder`` that points
    ``solve_data/`` at the sub-solve's snapshot dir.

    Symlinks ``input/`` and ``output_raw/`` so flexpy's loader sees
    the same shared data the per-sub-solve tests use.
    """
    import os
    for child in ("input", "output_raw"):
        src = work_folder / child
        if src.exists() and not (tmpdir / child).exists():
            os.symlink(src, tmpdir / child)
    sub_dir = work_folder / f"solve_data_{sub_solve}"
    if not sub_dir.exists():
        # Handle the "single-solve" degenerate case — fall through to the
        # canonical solve_data/ directly.
        sub_dir = work_folder / "solve_data"
    if not (tmpdir / "solve_data").exists():
        os.symlink(sub_dir, tmpdir / "solve_data")
    return tmpdir


def _run_chain_native(
    work_folder: Path | str,
    *,
    chain: list[str] | None = None,
    scenario: str | None = None,
    warm: bool = False,
) -> dict[str, ChainStep]:
    """Native orchestrator backend for :func:`run_chain`.

    Discovers the scenario DB via:

    * an explicit ``tests.sqlite`` / ``input.sqlite`` under
      ``work_folder``.

    Picks the scenario via, in priority order:

    1. The explicit ``scenario`` kwarg (or the
       ``FLEXPY_NATIVE_SCENARIO`` env var as a fallback).
    2. The scenario whose name matches the directory's ``work_<S>``
       suffix (with a small set of legacy-naming overrides — see
       :data:`_NATIVE_SCENARIO_OVERRIDES`).
    3. Failing both, raises :class:`ValueError` instead of guessing —
       the native path is the consumer of a DB, not a snapshot tree;
       silently picking the first scenario alphabetically silently
       runs the wrong scenario in shared-DB fixtures.

    When the DB is found AND a unique scenario is determined,
    delegates to
    :func:`flextool.engine_polars._orchestration.run_chain_from_db`,
    re-runs flextool's preprocessing into the same work folder, and
    returns the result as ``dict[str, ChainStep]`` (mapping each solve
    to its Solution + handoff + ``warm_used``).
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
            f"_run_chain_native: no DB found under {work} "
            f"(looked for tests.sqlite, input.sqlite).  Native "
            f"orchestration requires a DB scenario; for the file-based "
            f"path call run_chain(..., native=False)."
        )

    # Late import to avoid a build-time cycle between chain and _orchestration.
    from flextool.engine_polars._orchestration import run_chain_from_db

    # Resolve the scenario.
    if scenario is None:
        scenario = os.environ.get("FLEXPY_NATIVE_SCENARIO") or None
    if scenario is None:
        scenario = _resolve_native_scenario(db_path, work)
    if scenario is None:
        raise ValueError(
            f"_run_chain_native: cannot determine which scenario to "
            f"run for {work}.  Pass scenario= explicitly, set "
            f"FLEXPY_NATIVE_SCENARIO, or rename the work directory to "
            f"match the scenario name (work_<scenario>).  "
            f"work.name={work.name!r}"
        )

    # Critical: do NOT pass ``work`` directly as ``work_folder`` to
    # ``run_chain_from_db``.  ``FlexToolRunner.write_input`` would
    # overwrite the fixture's ``input/`` directory (rewriting
    # ``input/model__solve.csv`` to whatever scenario we resolved,
    # potentially shrinking a 4-solve cascade fixture down to a single
    # solve).  Instead, run the orchestrator into a private tempdir;
    # the work folder is consulted only for its DB.
    steps = run_chain_from_db(
        db_path, scenario, warm=warm,
    )
    # Adapt to ChainStep shape so callers see the same return type.
    out: dict[str, ChainStep] = {}
    for name, step in steps.items():
        out[name] = ChainStep(
            solve_name=name,
            solution=step.solution,
            handoff=step.handoff,
            warm_used=getattr(step, "warm_used", False),
        )
    return out


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
    use_handoff_overlay: bool = False,
    warm: bool = False,
    chain: list[str] | None = None,
    native: bool | None = None,
) -> dict[str, ChainStep]:
    """Run a flextool multi-solve chain end-to-end in flexpy.

    Iterates the sub-solves listed in ``input/model__solve.csv`` (or, if
    that file is missing, any ``solve_data_<name>/`` dirs found under
    ``work_folder``).  Each sub-solve:

    1. Loads its per-sub-solve snapshot (``solve_data_<sub>/``).
    2. Builds (or warm-updates) a flexpy LP via :func:`build_flextool`.
    3. Solves with HiGHS.
    4. Captures an in-memory ``SolveHandoff`` via
       :func:`build_handoff_from_flexpy`, threaded forward as the
       ``prior_handoff`` for the next sub-solve.

    The returned dict preserves chain order via Python 3.7+ dict
    insertion semantics; iterate with ``run_chain(...).items()`` to
    walk the chain in sequence.

    Parameters
    ----------
    work_folder : Path | str
        Directory containing ``input/``, ``output_raw/`` and one
        ``solve_data_<sub>/`` per chained sub-solve.
    use_handoff_overlay : bool, default False
        When True, every sub-solve except the first passes the prior
        sub-solve's flexpy-extracted ``SolveHandoff`` into
        :func:`load_flextool` via the ``handoff=`` kwarg (Δ.11 — replaces
        the prior post-load ``apply_handoff`` overlay).  The carrier-
        derived FlexData fields are populated during the build directly
        from the in-memory carriers; flextool's per-sub-solve snapshots
        are needed only for STRUCTURE (entity sets, methods, profiles,
        …), and all multi-solve STATE flows in-memory between flexpy
        invocations.  Default ``False`` preserves the original
        behaviour: snapshot CSVs are the source of truth for handoff.
    warm : bool, default False
        When True, attempt warm LP updates between consecutive
        structurally-compatible sub-solves using
        :class:`polar_high.WarmProblem`.  Decisions are recorded per-step
        on :attr:`ChainStep.warm_used`.  Default ``False`` preserves
        the original cold-rebuild behaviour for full backward
        compatibility.
    chain : list[str] | None, default None
        Explicit sub-solve order; overrides the default
        ``input/model__solve.csv`` lookup.  Use to drive a
        ``solve_data_<sub>/`` enumeration that the CSV doesn't cover
        (e.g. rolling-horizon snapshots whose ``model__solve.csv``
        names a parent solve rather than the per-roll snapshot dirs).
    native : bool | None, default None
        Γ.8.D feature flag.  ``True`` delegates to the native
        orchestrator (``_orchestration.run_orchestration``) which
        re-runs flextool's per-solve preprocessing under
        ``work_folder`` and runs HiGHS for every solve in-process via
        the in-memory handoff path.  ``False`` (default) preserves the
        legacy file-symlink-based driver below — the behaviour every
        existing test exercises.  ``None`` (default-default) consults
        the ``FLEXPY_USE_NATIVE_ORCHESTRATION`` env var: ``"1"`` /
        ``"true"`` / ``"yes"`` enable the native path, anything else
        keeps legacy.  R-O7 mitigation: legacy path stays the default
        so the existing test surface stays green.

    Returns
    -------
    dict[str, ChainStep]
        Mapping ``solve_name → ChainStep(solve_name, solution, handoff,
        warm_used)``.
    """
    import os
    import tempfile

    # Feature-flag gate.  ``native=True`` always uses the new path;
    # ``native=False`` always uses legacy; ``native=None`` consults the
    # env var (default ``False``).  See Γ.8.D in
    # ``audit/solve_orchestration_plan.md``.
    if native is None:
        env_val = os.environ.get(
            "FLEXPY_USE_NATIVE_ORCHESTRATION", ""
        ).strip().lower()
        native = env_val in ("1", "true", "yes", "on")
    if native:
        return _run_chain_native(work_folder, chain=chain, warm=warm)

    work = Path(work_folder)
    if chain is None:
        chain = _read_chain_order(work)
    if not chain:
        raise ValueError(
            f"run_chain: no sub-solves found in {work} "
            f"(missing input/model__solve.csv and no solve_data_<sub>/ dirs)"
        )

    results: dict[str, ChainStep] = {}
    prior_handoff = None
    # Warm-mode state, used only when warm=True.
    warm_problem: WarmProblem | None = None
    prior_data: "FlexData | None" = None
    prior_fp: tuple | None = None

    for i, sub_solve in enumerate(chain):
        with tempfile.TemporaryDirectory() as tmp:
            td = Path(tmp)
            _stage_subsolve_workdir(work, sub_solve, td)
            # Δ.11 — construct-with-handoff: pass the in-memory
            # SolveHandoff into ``load_flextool`` so the carrier-derived
            # fields are populated during the build.  Replaces the
            # previous post-load ``apply_handoff`` overlay step.
            handoff_arg = (prior_handoff
                              if use_handoff_overlay and i > 0
                                 and prior_handoff is not None
                              else None)
            data = load_flextool(td, handoff=handoff_arg)

            warm_used = False
            if warm:
                fp = _fingerprint(data)
                tried_warm = (i > 0
                              and warm_problem is not None
                              and prior_data is not None
                              and prior_fp == fp)
                if tried_warm:
                    try:
                        _apply_warm_updates(warm_problem, prior_data, data)
                        warm_used = True
                    except _IncompatibleUpdate:
                        warm_problem = None
                if not warm_used:
                    warm_problem = _build_warm_problem(data)
                sol = warm_problem.solve()
                prior_data = data
                prior_fp = fp
            else:
                pb = Problem()
                build_flextool(pb, data)
                sol = pb.solve()

            handoff = build_handoff_from_flexpy(
                sol, td, sub_solve, prior_handoff=prior_handoff,
            )

        results[sub_solve] = ChainStep(sub_solve, sol, handoff,
                                       warm_used=warm_used)
        prior_handoff = handoff

    return results
