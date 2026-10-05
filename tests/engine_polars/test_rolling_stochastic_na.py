"""Coverage-first solving e2e for the rolling stochastic UC.

Fixture ``1_week_rolling_wind`` (scenario in ``tests/fixtures/stochastics.json``,
solve ``1week_rolling``): a 168-step week (timeset ``1week`` /
``timeset_duration = 168``) solved as ``solve_mode = rolling_window`` with
``rolling_solve_horizon = 96`` and ``rolling_solve_jump = 24``.  The jump of
24 over a 168-step horizon yields SEVEN rolls (roll starts t0001, t0025,
t0049, t0073, t0097, t0121, t0145 — matching the seven ``stochastic_branches``
reveal points).  Three equal-probability branches (``upper`` / ``lower`` /
``mid``) fan at every reveal alongside the ``realized`` anchor.

Why this test exists / what it protects
---------------------------------------
This file was written BEFORE W1 as a coverage-first guard.  Before W1 the
rolling solve CRASHED during E0 branch-reveal validation
(``flextool/engine_polars/_stochastic.py`` raised ``FlexToolConfigError`` —
"a stochastic branch must start at the FIRST step of its branching period").
This is the W1/W3 defect tracked in ``specs/stochastics_reassessment.md``
§A.1 / §A.2 / §E / §F.

W1 has now landed: the triggering-cohort validation replaces that first-step
guard, so the rolling solve runs to a strict optimum on every roll and
``test_rolling_stochastic_solve_completes`` asserts for real.

W3 HAS NOW LANDED too.  Root cause: the synthetic-solve early-return in
``_apply_db_overrides`` (``flextool/engine_polars/input.py``) fires for every
rolling sub-solve (``1week_rolling_roll_N`` is not declared in Spine), so
passes 3-10 — including ``apply_derived_g`` → ``apply_branch_cluster``, the
sole producer of ``period_branch_full`` / ``dt_non_anticipativity`` — never
ran per roll and both frames kept their deliberately-None seeds.  The fix (the
``NA-1`` re-wire, alongside the existing RESERVE-1/SECTION-1/MINLOAD-1/INFLOW-1
re-wires) populates ONLY those two frames on the synthetic path by replicating
``apply_branch_cluster``'s two frame assignments verbatim — it deliberately
does NOT call the full helper, which would overwrite the correctly-seeded
``period_in_use_set`` via an unsorted ``.unique()`` and reorder nested-invest
set outputs.  So the two NA-coupling tests below assert for real (xfail
decorators removed).

Deferred gap (NOT delivered by W3): absolute storage continuity — forecast
branches continuing from the committed roll-start state (spec §B / §D.6).  The
storage NA constraint ties NET CHARGE (per-(n, t) charge delta,
``model.py:219-234``), not absolute ``v_state``.  The realized trunk
``period1`` is start-anchored while the three forecast copies cyclic-self-link
their storage start, so absolute ``v_state`` carries a constant realized↔
forecast offset (~0.06 here) even though the net-charge deltas are tied to
~1e-16.  Closing that offset is a separate anchor-repointing slice.

The structural signal the frames test asserts is the exact precondition the
operational-NA gate (``model.py:4519-4522``) requires to emit the four
``non_anticipativity_*`` constraint families: a per-roll ``flex_data`` whose
``period_branch_full`` (with a non-empty ``d != b`` sub-cohort) and
``dt_non_anticipativity`` frames are both populated.
"""
from __future__ import annotations

import os

import polars as pl
import pytest

from flextool.engine_polars._orchestration import run_chain_from_db
from flextool.engine_polars._solve_state import FlexToolConfigError

SCENARIO = "1_week_rolling_wind"

# Deterministic roll count (see module docstring): 168-step horizon,
# rolling_solve_jump = 24 → 7 rolls.
EXPECTED_ROLLS = 7

# W1 HAS LANDED (triggering-cohort validation replaces the E0 first-step-only
# guard) — the rolling stochastic solve no longer crashes and every roll now
# reaches a strict optimum, so ``test_rolling_stochastic_solve_completes``
# asserts for real (module-level xfail removed).
#
# W3 HAS LANDED too: the ``NA-1`` synthetic-path re-wire in
# ``_apply_db_overrides`` now populates ``period_branch_full`` /
# ``dt_non_anticipativity`` per roll, so the operational non-anticipativity
# constraints fire in rolling and both NA-coupling tests below assert for real
# (their xfail decorators have been removed).
pytestmark = [pytest.mark.solver]


@pytest.fixture(scope="module")
def rolling_result(stochastic_db_url, test_solver_config_dir, tmp_path_factory):
    """Run the rolling stochastic scenario end-to-end ONCE for the module.

    Post-W1 the solve runs to completion, so this memoises the per-step dict.
    The ``except FlexToolConfigError`` is a defensive guard: if a future change
    re-breaks branch-reveal validation, the crash is memoised and re-raised in
    each test body (surfacing as a plain failure now that the module-level
    xfail is gone).  The ``run_chain_from_db`` call runs under
    ``FLEXTOOL_AUTOSCALE_STRICT=1`` so that any autoscale-registry gap in the NA
    path is loud rather than silently reverting to an un-scaled LP (invariant #1).
    """
    work_folder = tmp_path_factory.mktemp("rolling_stochastic_na")
    prev = os.environ.get("FLEXTOOL_AUTOSCALE_STRICT")
    os.environ["FLEXTOOL_AUTOSCALE_STRICT"] = "1"
    try:
        steps = run_chain_from_db(
            input_db_url=stochastic_db_url,
            scenario_name=SCENARIO,
            work_folder=work_folder,
            solver_config_dir=test_solver_config_dir,
            warm=False,
            keep_solutions=True,
        )
        return {"steps": steps, "error": None}
    except FlexToolConfigError as exc:
        return {"steps": None, "error": exc}
    finally:
        if prev is None:
            os.environ.pop("FLEXTOOL_AUTOSCALE_STRICT", None)
        else:
            os.environ["FLEXTOOL_AUTOSCALE_STRICT"] = prev


def _steps(rolling_result):
    """Return the per-step ``{solve_name: OrchestrationStep}`` dict.

    Re-raises the memoised ``FlexToolConfigError`` only in the defensive case
    where branch-reveal validation has re-broken (see ``rolling_result``); on
    the normal post-W1 path ``error`` is ``None`` and the dict is returned.
    """
    if rolling_result["error"] is not None:
        raise rolling_result["error"]
    return rolling_result["steps"]


def _pick(cols, *candidates):
    for c in candidates:
        if c in cols:
            return c
    return cols[0]


def test_rolling_stochastic_solve_completes(rolling_result):
    """Every roll of the rolling stochastic solve reaches a strict optimum."""
    steps = _steps(rolling_result)
    assert len(steps) == EXPECTED_ROLLS, (
        f"expected {EXPECTED_ROLLS} rolls (168-step horizon, "
        f"rolling_solve_jump=24), got {len(steps)}"
    )
    assert all(s.optimal for s in steps.values()), {
        name: s.optimal for name, s in steps.items()
    }


def test_rolling_na_frames_couple_branches_per_roll(rolling_result):
    """PRIMARY structural signal: each roll carries the NA precondition.

    The ``NA-1`` synthetic-path re-wire in ``_apply_db_overrides``
    (replicating ``apply_branch_cluster``'s two frame assignments) must
    populate, on every roll's ``flex_data``, both the ``period_branch_full``
    (d, b) anchor→sibling map — with a non-empty ``d != b`` sub-cohort — and
    the ``dt_non_anticipativity`` (d, t) tie set.  That is exactly the gate the
    operational-NA families check at ``model.py:4519-4522`` before emitting
    the ``non_anticipativity_*`` constraints.
    """
    steps = _steps(rolling_result)
    for name, s in steps.items():
        fd = s.flex_data
        assert fd is not None, f"roll {name!r} retained no flex_data"

        pbf = fd.period_branch_full
        assert pbf is not None and pbf.height > 0, (
            f"roll {name!r}: empty period_branch_full"
        )
        assert fd.dt_non_anticipativity is not None, (
            f"roll {name!r}: dt_non_anticipativity is None"
        )
        assert fd.dt_non_anticipativity.height > 0, (
            f"roll {name!r}: empty dt_non_anticipativity"
        )

        # The d != b sub-cohort is the real anchor→sibling tie set the NA
        # gate filters to.  Cast both axes to Utf8 before comparing — under
        # Phase 4 activation ``d`` and ``b`` are different Enum vocabularies
        # and polars refuses ``!=`` across them.
        off_diag = pbf.with_columns(
            pl.col("d").cast(pl.Utf8),
            pl.col("b").cast(pl.Utf8),
        ).filter(pl.col("d") != pl.col("b"))
        assert off_diag.height > 0, (
            f"roll {name!r}: period_branch_full has no d != b rows"
        )


def test_rolling_na_ties_storage_across_branches(rolling_result):
    """Storage NA ties NET CHARGE across branches at the tied (d, t) cells.

    This fixture authors NO unit commitment (no ``online_method`` /
    ``startup_method`` in ``stochastics.json``), so ``v_online_integer`` never
    exists — the variable NA actually ties here is ``v_state`` (storage params
    ARE present).  The storage non-anticipativity family pins the per-(n, t)
    NET CHARGE across siblings — the consecutive ``v_state`` delta
    ``v_state[n, d, t] - v_state[n, d, t_prev]`` (``model.py:219-234``) — NOT
    absolute ``v_state``.  This test therefore witnesses the net charge: for
    each branch-period ``d`` in the cohort (``v_state``'s ``d`` axis carries
    the realized trunk ``period1`` plus the three forecast copies
    ``period1_{lower,mid,upper}``) and node ``n``, it sorts by tied timestep
    ``t``, takes the consecutive ``v_state`` delta, and asserts that delta is
    equal across the cohort ``d`` at each matching ``(n, t)`` tied step.  The
    cohort includes the realized ``period1``, so this exercises realized↔
    forecast coupling, not merely forecast↔forecast.

    Absolute storage continuity is explicitly NOT delivered by W3 (spec §B /
    §D.6): the realized trunk ``period1`` is start-anchored while the forecast
    copies cyclic-self-link their storage start, so absolute ``v_state``
    carries a constant realized↔forecast offset (measured ~0.06 here) even
    though the net-charge deltas collapse to ~1e-16.  Closing that offset —
    forecast branches continuing from the committed roll-start state — is a
    separate anchor-repointing slice, not part of W3.

    Branches legitimately DIVERGE at non-tied steps, so the cross-branch
    net-charge equality is restricted to the tied timesteps carried by that
    roll's ``fd.dt_non_anticipativity``.  The test also asserts that a
    positive number of tied net-charge cells was actually checked so the
    witness can never be silently vacuous.
    """
    steps = _steps(rolling_result)
    tol = 1e-6
    checked = 0
    for name, s in steps.items():
        sol = s.solution
        assert sol is not None, f"roll {name!r} retained no solution"
        if "v_state" not in sol._vars:
            continue

        fd = s.flex_data
        assert fd is not None, f"roll {name!r} retained no flex_data"
        dtna = fd.dt_non_anticipativity
        if dtna is None or dtna.height == 0:
            continue

        vs = sol.value("v_state")
        cols = vs.columns
        ncol = _pick(cols, "n", "node")
        tcol = _pick(cols, "t", "time")
        dcol = _pick(cols, "d", "period")

        # Restrict to the tied timesteps (the (d, t) tie set's ``t`` axis).
        # ``v_state``'s ``d`` is the branch-period axis, so keeping every
        # branch at the tied timesteps is what lets the cross-branch net
        # charge be witnessed; cast every key to Utf8 so the join / grouping
        # composes across the Enum vocabularies.
        tied_t = (dtna.select(pl.col("t").cast(pl.Utf8).alias("__t"))
                      .unique())
        vs_tied = (vs.with_columns(
                        pl.col(ncol).cast(pl.Utf8).alias("__n"),
                        pl.col(dcol).cast(pl.Utf8).alias("__d"),
                        pl.col(tcol).cast(pl.Utf8).alias("__t"),
                   )
                     .join(tied_t, on="__t", how="inner"))
        if vs_tied.height == 0:
            continue

        # Net charge = consecutive v_state delta within each (n, d), over the
        # tied timesteps sorted by t.  The first tied step per (n, d) has no
        # predecessor → null delta, dropped.
        # NB: this takes the delta over consecutive TIED steps, which matches
        # the constraint's true-predecessor net charge only while the tied set
        # is contiguous in real time (it is here — each roll's jump is 24
        # contiguous steps).  A future non-contiguous dt_non_anticipativity
        # would need the delta taken over real predecessors instead.
        deltas = (vs_tied
                  .sort(["__n", "__d", "__t"])
                  .with_columns(
                      delta=(pl.col("value") - pl.col("value").shift(1))
                            .over(["__n", "__d"]))
                  .drop_nulls("delta"))

        # Spread of net charge across the branch-period axis at each (n, t).
        spread = (deltas
                  .group_by(["__n", "__t"])
                  .agg(
                      n_branches=pl.col("delta").len(),
                      dmax=pl.col("delta").max(),
                      dmin=pl.col("delta").min(),
                  )
                  .filter(pl.col("n_branches") > 1))
        for row in spread.iter_rows(named=True):
            assert row["dmax"] - row["dmin"] <= tol + 1e-6 * max(
                1.0, abs(row["dmax"]), abs(row["dmin"])
            ), (
                f"roll {name!r}: net-charge spread "
                f"{row['dmax'] - row['dmin']} at (n={row['__n']}, "
                f"t={row['__t']}) exceeds tol across branches"
            )
            checked += 1

    assert checked > 0, (
        "no tied (n, t) net-charge cells with >1 branch were checked — the "
        "storage NA witness would be vacuous"
    )
