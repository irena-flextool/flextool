"""Loud-and-early guards of the solve chain: the invest-branching and dispatch-realization methods, the rolling
stochastic nesting errors, the realization solves of stochastic solves, and the Benders handoff guard."""
from __future__ import annotations

import re
from flextool.engine.input._solve_state import FlexToolConfigError


def _invest_branching_raw_guard(solve_cfg, base_name: str) -> None:
    """Refuse an unrecognised authored ``solve.invest_solve_branching_method``.

    Reads the RAW authored value: :meth:`SolveConfig.invest_branching_method_for` reads an unknown value (a typo)
    as ``operations_only``, so the check inspects the raw dict.  Blank / absent pass.  Where the value may be set
    (investment solves with ``stochastic_branches``) is checked by :meth:`SolveConfig.check_invest_branching`.
    """
    from flextool.engine.input._solve_config import INVEST_SOLVE_BRANCHING_METHODS

    raw = (getattr(solve_cfg, "invest_solve_branching_method", None) or {}).get(base_name)
    if raw is None:
        return
    val = str(raw).strip().lower()
    if val and val not in INVEST_SOLVE_BRANCHING_METHODS:
        raise FlexToolConfigError(
            f"Solve '{base_name}' sets invest_solve_branching_method='{raw}', an unrecognised value. Supported: "
            + ", ".join(f"'{m}'" for m in INVEST_SOLVE_BRANCHING_METHODS) + " (the first is the default).")


def _rolling_realization_fix(separate_dispatch: bool, stochastic: bool) -> str:
    """The way out of the nested / containing errors once the solves stay where they are: a ``separate_dispatch``
    solve needs its ``realized: yes`` branch (removing it stops the run for that), so it changes the method; a
    solve with a designation (no ``separate_dispatch``) can drop the designation."""
    if not separate_dispatch:
        return "remove its realized: yes branch"
    if stochastic:
        return ("set its dispatch_realization_method to deterministic (the solve then reads only its realized: yes "
                "branch)")
    return "set its dispatch_realization_method to deterministic"


def _rolling_containing_error(solve: str, *, separate_dispatch: bool = False,
                              stochastic: bool = True) -> FlexToolConfigError:
    """A rolling stochastic solve that contains other solves cannot have its jumps realized (4.7.0).
    *separate_dispatch*: the solve authors that method (the raw guard); *stochastic*: it has
    ``stochastic_branches``."""
    kind = "rolling stochastic solve" if stochastic else "rolling solve"
    method = " with dispatch_realization_method separate_dispatch" if separate_dispatch else ""
    why = (f"FlexTool 4.7.0 cannot yet realize the jumps of a {kind} that contains other solves." if stochastic else
           "FlexTool 4.7.0 cannot yet add the realization solve that separate_dispatch needs to a rolling solve "
           "that contains other solves.")
    return FlexToolConfigError(
        f"Solve '{solve}' is a {kind}{method} that contains other solves (contains_solves). {why} Move the solves it "
        f"contains out of its contains_solves, or {_rolling_realization_fix(separate_dispatch, stochastic)}."
    )


def _rolling_nested_error(solve: str, *, separate_dispatch: bool = False,
                          stochastic: bool = True) -> FlexToolConfigError:
    """A rolling stochastic solve nested in another solve cannot have its jumps realized (4.7.0).  Keywords as
    :func:`_rolling_containing_error`."""
    kind = "rolling stochastic solve" if stochastic else "rolling solve"
    method = " with dispatch_realization_method separate_dispatch" if separate_dispatch else ""
    why = (f"FlexTool 4.7.0 cannot yet realize the jumps of a nested {kind}." if stochastic else
           "FlexTool 4.7.0 cannot yet add the realization solve that separate_dispatch needs to a nested rolling "
           "solve.")
    return FlexToolConfigError(
        f"Solve '{solve}' is a {kind}{method} nested inside another solve's contains_solves. {why} Make it a "
        f"top-level solve, or {_rolling_realization_fix(separate_dispatch, stochastic)}."
    )


def _dispatch_realization_raw_guard(solve_cfg, base_name: str) -> None:
    """W4 Guard — validate a non-default ``dispatch_realization_method``.

    ``solve.dispatch_realization_method`` selects how realized operations
    are produced.  ``'deterministic'`` (the shipped default: a solve with
    ``stochastic_branches`` reads only its ``realized: yes`` branch,
    ``SolveConfig.apply_deterministic_realization``) and ``'within_solve'``
    are always allowed.  ``'separate_dispatch'`` (W4b) — the
    engine automatically registers a realization solve
    ``<solve>__realized_dispatch`` that takes the full
    invest→dispatch handoff + last-wins path — is accepted for a
    TOP-LEVEL ``single_solve`` or ``rolling_window`` solve that contains no
    solves and has a ``realized: yes`` branch.  A solve that itself
    ``contains_solves`` children, or one nested inside another's
    ``contains_solves``, is rejected; a rolling one with the 4.7.0 texts of
    a rolling stochastic solve whose jumps cannot be realized.  On an
    investment solve it is not read (``is_investment_solve``; warning
    W-0640), so it is not checked there.

    The guard reads the RAW authored value rather than the resolver's
    output: :meth:`SolveConfig.dispatch_realization_method_for` silently
    collapses any UNRECOGNISED value (a typo) to ``'deterministic'``, so an
    authoring-time check must inspect the raw dict.  Blank / absent
    tolerate-as-default.  No-op when the solve authors nothing.

    ``solve_modes`` / ``contains_solves`` are read with ``getattr`` +
    ``.get`` defaults so partial unit-test stubs (which may carry only
    ``model_solve`` + ``dispatch_realization_method``) don't raise.
    """
    raw = solve_cfg.dispatch_realization_method.get(base_name)
    if raw is None:
        return
    val = str(raw).strip().lower()
    if val in ("", "deterministic", "within_solve"):
        return
    if val != "separate_dispatch":
        # Typo / unknown value — unchanged hard-reject.
        raise FlexToolConfigError(
            f"Solve '{base_name}' sets dispatch_realization_method='{raw}' — "
            f"unrecognised value.  Supported: 'deterministic' (default), "
            f"'within_solve' and 'separate_dispatch'."
        )
    # separate_dispatch on an investment solve: not read (W-0640, SolveConfig.apply_deterministic_realization).
    from flextool.engine.input._solve_config import is_investment_solve

    if is_investment_solve(solve_cfg, base_name):
        return
    # separate_dispatch — eligible ONLY for a single top-level single_solve.
    solve_modes = getattr(solve_cfg, "solve_modes", {}) or {}
    model_solve = getattr(solve_cfg, "model_solve", {}) or {}
    contains_solves = getattr(solve_cfg, "contains_solves", {}) or {}
    top_level: set = set()
    for _solves in model_solve.values():
        top_level.update(_solves)
    nested: set = set()
    for _children in contains_solves.values():
        if isinstance(_children, (list, tuple, set)):
            nested.update(_children)
        elif _children is not None:
            nested.add(_children)
    mode = solve_modes.get(base_name, "single_solve")
    stochastic = bool((getattr(solve_cfg, "stochastic_branches", None) or {}).get(base_name))
    if mode == "rolling_window" and contains_solves.get(base_name):
        raise _rolling_containing_error(base_name, separate_dispatch=True, stochastic=stochastic)
    if mode == "rolling_window" and base_name in nested:
        raise _rolling_nested_error(base_name, separate_dispatch=True, stochastic=stochastic)
    reasons: list[str] = []
    if mode not in ("single_solve", "rolling_window"):
        reasons.append(
            f"it runs under solve_mode='{mode}' (separate_dispatch needs a "
            f"single_solve or rolling_window)"
        )
    if base_name not in top_level:
        reasons.append("it is not a top-level solve in model.solves")
    if contains_solves.get(base_name):
        reasons.append("it has contained solves (contains_solves)")
    if base_name in nested:
        reasons.append("it is nested inside another solve's contains_solves")
    if reasons:
        raise FlexToolConfigError(
            f"Solve '{base_name}' sets "
            f"dispatch_realization_method='separate_dispatch', which is only "
            f"supported for a top-level solve that contains no solves, but "
            + "; ".join(reasons)
            + "."
        )
    # The separate realized dispatch solve dispatches the realized branch: without a ``realized: yes`` row in
    # stochastic_branches there is nothing to realize.
    rows = (getattr(solve_cfg, "stochastic_branches", None) or {}).get(base_name) or []
    if not any(len(row) > 3 and str(row[3]).strip().lower() == "yes" for row in rows):
        raise FlexToolConfigError(
            f"Solve '{base_name}' sets dispatch_realization_method='separate_dispatch', which runs a separate "
            f"dispatch solve of the realized branch, but its stochastic_branches have no branch with realized: "
            f"yes. Mark the branch to realize with realized: yes."
        )


def _spawn_realized_dispatch_siblings(
    solve_cfg, cascade_db_reader, logger, timeline_cfg=None, cascade_solve_cfg=None,
) -> None:
    """Register the realization solve of every stochastic top-level solve
    that needs one (``solve._realization``).

    * a solve authoring ``dispatch_realization_method = separate_dispatch``
      gets an engine-automatic deterministic realized-dispatch sibling;
    * otherwise a solve followed in ``model.solves`` by an author-declared
      realization solve (``configured_realization_solve``) has that solve
      registered (``SolveConfig.register_realization_solve``);
    * otherwise a single_solve / rolling_window solve whose ``realized: yes``
      designation reaches its committed window
      (``designation_reaches_committed_window``) gets the sibling too, unless
      every forecast invests on its own there (``invest_solve_branching_method``
      ``investments_at_reveal_period``): under ``operations_only`` and
      ``investments_after_reveal_period`` the realized investments are one
      decision of every forecast, which the sibling keeps.  A designated
      solve that contains solves, or is nested in another solve, raises.

    ════════════════════════════════════════════════════════════════════
    REALIZED-DISPATCH FEATURE MAP (the single anchor for all touch-points)
    ════════════════════════════════════════════════════════════════════
    The ``separate_dispatch`` feature (spec ``stochastics_reassessment.md``
    §B/§H; slices W4a-W4d) is split across these pieces — THIS function is
    the entry point that ties them together:

    * schema (W4a): ``solve.dispatch_realization_method`` {within_solve,
      separate_dispatch, deterministic (v83, no realization solve: the
      solve reads its realized branch, ``apply_deterministic_realization``)} — migration ``_migrate_v75_...``, accessor
      ``SolveConfig.dispatch_realization_method_for``.
    * validation: ``_dispatch_realization_raw_guard`` (both wired sites) —
      accepts separate_dispatch for a top-level single or rolling solve that contains no solves.
    * registration (W4b): THIS function → for each eligible parent, a DUAL
      in-memory registration of the sibling ``<parent>__realized_dispatch``:
        - ``SpineDbReader.register_derived_solve`` — clones every RAW Spine
          param of the parent except
          ``SpineDbReader.REALIZATION_SOLVE_DROPPED_PARAMS``, so the sibling
          takes the full override path (``_solve_in_spine`` True);
        - ``SolveConfig.spawn_realized_dispatch_sibling`` — inserts the
          sibling after the parent in ``model_solve`` and derives its
          SolveConfig fields (``derive_realization_solve``: every field
          except ``_solve_config.REALIZATION_SOLVE_DROPS``); records the
          authoritative ``SolveConfig.realized_dispatch_parent[sibling] =
          parent``.
      ►► DUAL-DEFINITION INVARIANT: the reader's drop list and the
         SolveConfig drop list describe the SAME sibling config in two
         vocabularies. They MUST stay aligned — pinned by
         ``tests/engine/test_realized_dispatch_coherence.py``.
    * projection (W4c): ``_run_model`` step 3.6 reads
      ``realized_dispatch_parent`` and projects each stochastic param onto
      the parent's realized branch (``project_realized_branch_time_branch_rows``).
    * weighting (W4d): the parent's realized OPERATIONAL weight — keyed off
      the parent's ``dispatch_realization_method`` — is de-scenario'd in the
      ``pdt`` producers; reuse ``realized_dispatch_parent`` / the existing
      per-solve channel, do NOT add a parallel sibling-detection.
    ════════════════════════════════════════════════════════════════════

    Called ONCE from :func:`_drive_cascade` (after ``cascade_db_reader``
    is built, BEFORE ``native_run_model``).  For each eligible parent it
    performs a DUAL in-memory registration so the sibling takes the FULL
    override path — indistinguishable from W8.1's author-declared
    dispatch solve:

    1. ``cascade_db_reader.register_derived_solve(parent, sibling)`` — so
       the per-solve override chain treats the sibling as a REAL Spine
       solve (``_solve_in_spine`` → ``True``).  Without this the loader
       falls back to the synthetic ``<base>_<anchor>`` path, which
       silently drops reserves and other entity-level LP structure the
       realized dispatch must carry.
    2. ``solve_cfg.spawn_realized_dispatch_sibling(parent, sibling)`` —
       insert the sibling RIGHT AFTER its parent in ``model_solve`` (so
       it consumes the parent's committed capacity through the normal
       invest→dispatch handoff and last-wins hands it the realized
       window) and populate its per-solve config as a deterministic,
       fixed-capacity, monolithic single_solve.

    *solve_cfg* MUST be ``runner.state.solve`` — the config
    ``native_run_model`` actually drives.  *cascade_solve_cfg* is the outer
    cascade ``state.solve``, where the ``run()`` closure looks up each
    solve's solver: the sibling gets its parent's solver and solver
    settings and scaling there too (``SolveConfig.derive_realization_solve``), so a sibling
    of a Gurobi solve runs on Gurobi, not on HiGHS.  (The run()-site guard reads the
    SEPARATE outer cascade ``state.solve``, where the sibling is never
    injected, so it correctly no-ops for the sibling, which authors no
    ``dispatch_realization_method`` — no infinite re-spawn.)

    ``separate_dispatch`` eligibility was already validated by
    :func:`_dispatch_realization_raw_guard` at both wired sites; this
    re-selects the same eligible parents from the driving config so the
    two stay in lockstep.

    The sibling name is ``f"{parent}__realized_dispatch"`` — the DOUBLE
    underscore keeps it NON-period-like so the synthetic resolver's
    ``<base>_<anchor>`` split can never mistake the ``__realized_dispatch``
    tail for a real period anchor (correctness of the synthetic-resolver
    no-op relies on the anchor never being a real period).  A collision
    with an existing solve — in ``model_solve`` values,
    ``contains_solves`` values, or the reader's ``solve`` entities —
    raises :class:`FlexToolConfigError` naming the colliding owner.
    """
    model_solve = getattr(solve_cfg, "model_solve", {}) or {}
    top_level: set = set()
    for _solves in model_solve.values():
        top_level.update(_solves)

    # Collision sources (name → owner category).
    contains_solves = getattr(solve_cfg, "contains_solves", {}) or {}
    contained: set = set()
    for _children in contains_solves.values():
        if isinstance(_children, (list, tuple, set)):
            contained.update(_children)
        elif _children is not None:
            contained.add(_children)
    reader_solves: set = set()
    if cascade_db_reader is not None:
        try:
            _ents = cascade_db_reader.entities("solve")
            _name_col = "name" if "name" in _ents.columns else _ents.columns[0]
            reader_solves = set(_ents[_name_col].to_list())
        except Exception:  # noqa: BLE001
            reader_solves = set()

    from flextool.engine.solve import _realization, _realization_tail

    def _is_rolling(solve: str) -> bool:
        return (getattr(solve_cfg, "solve_modes", None) or {}).get(solve) == "rolling_window"

    def _add(solve: str, why: str) -> None:
        # W-0594 (the delayed units keep the added realization solve from looking ahead) is noted with the
        # realization notices, once the plans and the expansion have passed their checks
        # (_realization_tail.announce_realization_notices).
        eligible.append((solve, why))

    # Spawn list: every separate_dispatch solve (authored), then every
    # designated solve (a ``realized: yes`` branch) without a configured realization solve.  Each
    # entry carries the reason the error messages below name.
    eligible: list[tuple[str, str]] = []
    deterministic = getattr(solve_cfg, "deterministic_branches", None) or {}
    for p in sorted(top_level):
        if p in deterministic:
            # dispatch_realization_method deterministic: the solve reads its realized branch itself, no
            # realization solve (SolveConfig.apply_deterministic_realization).
            continue
        roll_method = _realization_tail.realization_roll_method(solve_cfg, p)
        if roll_method == "within_solve":
            # Each roll of a rolling within_solve solve is its own realization roll (``_realization_tail``): it
            # realizes its jump and looks ahead after it, so no realization solve is added.  (W-0635 / W-0636:
            # native_run_model, once the plans have passed their checks.)
            continue
        if solve_cfg.dispatch_realization_method_for(p) == "separate_dispatch":
            if not (getattr(solve_cfg, "realized_periods", None) or {}).get(p):
                # Nothing to realize (an invest-only solve): no realization solve.
                if logger is not None:
                    logger.info(
                        "solve '%s' has dispatch_realization_method = separate_dispatch but no "
                        "realized_periods: no realization solve is added (it would dispatch nothing)", p,
                    )
                continue
            _add(p, "dispatch_realization_method='separate_dispatch'")
            continue
        configured = _realization.configured_realization_solve(solve_cfg, p)
        if configured is not None:
            if contains_solves.get(p):
                # The solves it contains would take its pins (the PIN translation of a nested sub-solve), and the
                # realization solve would continue from the last of them, not from it.
                raise FlexToolConfigError(
                    f"Solve '{configured}' right after solve '{p}' in model.solves would realize the committed "
                    f"window of '{p}' (it is taken as a realization solve because it is deterministic, contains "
                    f"no solves, has the same solve_mode and rolling_times as '{p}', and realizes every period "
                    f"'{p}' commits inside its stochastic fan), but '{p}' contains other solves "
                    f"(contains_solves), so no solve can realize it.  Move the solves it contains out of its "
                    f"contains_solves, end its committed window before the reveal, or make '{configured}' differ "
                    f"from that rule if it is not meant as a realization solve."
                )
            solve_cfg.register_realization_solve(configured, p)
            if logger is not None:
                logger.info(
                    "registered author-declared realization solve '%s' for "
                    "stochastic solve '%s'", configured, p,
                )
            _line = _realization.lookahead_in_configured_realization(solve_cfg, configured, p)
            if _line is not None:
                # Printed only (solve-site console convention): the logger writes to the console too.
                print(f"Warning: {_line}", flush=True)
            # A realization solve takes its parent's detail levels (so that
            # its online variables and reserve rows match those of the solve
            # it realizes): name an authored
            # value that is overridden. Printed only (solve-site console
            # convention): the logger writes to the console too.
            for _line in solve_cfg.reduce_detail_overridden(configured):
                print(f"Warning: {_line}", flush=True)
            continue
        if not _realization.designation_reaches_committed_window(solve_cfg, p):
            continue
        if solve_cfg.invest_branching_method_for(p) == "investments_at_reveal_period":
            # Every forecast invests on its own from the reveal period on: no realized investment to fix (refusal 3
            # keeps the reveal period and later out of its realized periods, so a run never gets here).
            continue
        if contains_solves.get(p) and _is_rolling(p):
            raise _rolling_containing_error(p)
        if contains_solves.get(p):
            raise FlexToolConfigError(
                f"Solve '{p}' has a realized: yes designation in "
                f"stochastic_branches whose fan reaches its committed window, "
                f"but it contains other solves (contains_solves), so no "
                f"realization solve can realize it.  Move the solves it "
                f"contains out of its contains_solves, or remove the "
                f"designation."
            )
        _add(p, "a realized: yes designation in stochastic_branches")
    for c in sorted(contained):
        # A nested solve whose investments branch is refused later, with its own message.
        if (_realization.designation_reaches_committed_window(solve_cfg, c)
                and not solve_cfg.recourse_invest(c)):
            if _is_rolling(c):
                raise _rolling_nested_error(c)
            raise FlexToolConfigError(
                f"Solve '{c}' has a realized: yes designation in "
                f"stochastic_branches whose fan reaches its committed window, "
                f"but it is nested inside another solve's contains_solves, so "
                f"no realization solve can run right after it.  Make it a "
                f"top-level solve followed by a realization solve, or remove "
                f"the designation."
            )
    if eligible and cascade_db_reader is None:
        raise FlexToolConfigError(
            "A realization solve for "
            + ", ".join(f"'{p}' ({why})" for p, why in eligible)
            + " requires the DB-backed cascade reader (run via "
            "run_chain_from_db with db_url + scenario_name); none is "
            "available, so the realized-dispatch sibling cannot be registered."
        )

    for parent, why in eligible:
        # Positive storage-pinning guard (investigation outcome).
        # The sibling DROPS ``fix_storage_periods`` — the ONLY solve-level
        # storage mechanism it sheds (the node-level
        # ``storage_binding_method`` / ``storage_state_*`` are entity
        # parameters, preserved via the full override path, so storage
        # continuity WITHIN the monolithic single_solve is intact).  Any
        # authored ``fix_storage_periods`` would therefore be silently
        # re-floated by the sibling — giving a DIFFERENT storage trajectory
        # than the parent pinned.  Reject CONSERVATIVELY on ANY non-empty
        # ``fix_storage_periods`` (independent of realized-period count):
        # even a single pinned boundary is dropped, and gating on period
        # count would UNDER-fire (a single realized period with a 2-period
        # fix_storage authored still drops the pin).  No
        # ``fix_storage_periods`` → fine: the monolithic solve links storage
        # via its own balance constraints exactly as the parent would.
        _fix_storage = getattr(solve_cfg, "fix_storage_periods", {}) or {}
        if _fix_storage.get(parent):
            raise FlexToolConfigError(
                f"Solve '{parent}' needs an engine-added realization solve "
                f"({why}) but also authors fix_storage_periods: the "
                f"realization solve would re-float the pinned storage "
                f"boundaries.  Clear fix_storage_periods, or configure a "
                f"realization solve right after it in model.solves."
            )
        sibling = f"{parent}__realized_dispatch"
        if sibling in top_level or sibling in contained:
            raise FlexToolConfigError(
                f"Cannot auto-register realized-dispatch sibling "
                f"'{sibling}' for solve '{parent}': a solve named "
                f"'{sibling}' already exists in model.solves / "
                f"contains_solves."
            )
        if sibling in reader_solves:
            raise FlexToolConfigError(
                f"Cannot auto-register realized-dispatch sibling "
                f"'{sibling}' for solve '{parent}': a 'solve' entity named "
                f"'{sibling}' already exists in the database."
            )
        from flextool.engine.input._spinedb_reader import SpineDbReader
        # A single solve's realization solve reads its realized periods only
        # (the SolveConfig half: ``derive_realization_solve``).
        cascade_db_reader.register_derived_solve(
            parent, sibling, drop=SpineDbReader.REALIZATION_SOLVE_DROPPED_PARAMS,
            replace=solve_cfg.realization_reader_values(parent))
        solve_cfg.spawn_realized_dispatch_sibling(parent, sibling)
        if cascade_solve_cfg is not None and cascade_solve_cfg is not solve_cfg:
            cascade_solve_cfg.derive_realization_solve(parent, sibling)
        # The sibling reuses the parent's timesets, which a solve-level
        # new_stepduration has re-pointed to the aggregated timeline: it must
        # run at the same step (aggregated data, step lengths), not on the
        # aggregated steps with unaggregated data.  The reader half clones the
        # raw ``new_stepduration`` (it is not in the drop list).
        if timeline_cfg is not None:
            parent_step = timeline_cfg.new_step_durations.get(parent)
            if parent_step is not None:
                timeline_cfg.new_step_durations[sibling] = parent_step
        reader_solves.add(sibling)
        top_level.add(sibling)
        if logger is not None:
            logger.info(
                "registered engine-automatic realization solve '%s' for "
                "solve '%s' (%s)", sibling, parent, why,
            )
    if eligible:
        from flextool.engine.input._solve_config import SolveConfig
        siblings = [f"{parent}__realized_dispatch" for parent, _why in eligible]
        SolveConfig.check_derived_solves(solve_cfg, siblings)
        if cascade_solve_cfg is not None and cascade_solve_cfg is not solve_cfg:
            SolveConfig.check_derived_solves(cascade_solve_cfg, siblings)


def _benders_consume_guard_message(
    base_solve_name: str,
    prev_captured: str | None,
    benders_solve_names: "set[str]",
    benders_invest_handoff_names: "set[str]",
) -> str | None:
    """Loud guard for the cross-scheme handoff (v60/v62, narrowed for TIER 1).

    A Benders *investment* solve now deposits a real ``SolveHandoff``
    (``realized_cohort`` / ``realized_retire`` / ``divest_cumulative``)
    so a downstream rolling-dispatch solve can consume its invested
    capacity — that path is SUPPORTED and must NOT fire the guard.

    The guard now fires only when *base_solve_name* would consume a
    Benders predecessor that deposited **no usable investment handoff**
    — i.e. the predecessor is in *benders_solve_names* but NOT in
    *benders_invest_handoff_names* (its handoff carries no
    ``realized_cohort`` / ``divest_cumulative`` to hand forward, so the
    downstream solve would silently load nothing investy from it).

    Returns ``None`` when the situation is safe:

    * no predecessor, or the predecessor was not a Benders solve;
    * the predecessor is a roll of the *same* base solve (rolls of one
      Benders solve share its base name — not a cross-scheme consume);
    * the predecessor deposited a usable invest handoff (TIER 1 path).
    """
    if prev_captured is None or prev_captured not in benders_solve_names:
        return None
    if base_solve_name == re.sub(r"_roll_\d+$", "", prev_captured):
        return None
    if prev_captured in benders_invest_handoff_names:
        return None
    return (
        f"Solve '{base_solve_name}' follows '{prev_captured}', which ran "
        f"under decomposition=benders but produced NO invested/divested "
        f"capacity to hand forward (its handoff carries no realized_cohort "
        f"/ divest_cumulative). Consuming it would silently load nothing "
        f"from that solve. Make the downstream solve benders too, order "
        f"the chain so the Benders solve is terminal, or ensure the "
        f"Benders solve actually invests."
    )
