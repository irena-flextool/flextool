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
``test_rolling_stochastic_solve_completes`` asserts for real.  W3 (per-roll NA
frames) has NOT landed — ``period_branch_full`` / ``dt_non_anticipativity``
are still ``None`` on every roll, so no non-anticipativity constraint fires in
rolling yet.  The two NA-coupling tests are ``xfail(strict=True,
raises=AssertionError)`` until W3 populates those frames; remove their
decorators (and reconcile the review advisories in their docstrings against the
real per-roll frames) once W3 merges.

The structural signal the tests assert post-flip is the exact precondition
the operational-NA gate (``model.py:4519-4522``) requires to emit the four
``non_anticipativity_*`` constraint families: a per-roll ``flex_data`` whose
``period_branch_full`` (with a non-empty ``d != b`` sub-cohort) and
``dt_non_anticipativity`` frames are both populated.  Those frames are
produced authoritatively by ``apply_branch_cluster`` within
``apply_derived_g`` (``flextool/engine_polars/input.py:3842-3845``) — the
``_derived_branch.py`` overlay is only a seed-preserving pass, not the
producer.
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
# W3 (per-roll NA frames) has NOT landed: ``apply_branch_cluster`` still does
# not populate ``period_branch_full`` / ``dt_non_anticipativity`` per roll
# (both come back ``None`` on every roll — confirmed empirically under the W1
# change), so NO non-anticipativity constraints fire in rolling yet.  The two
# NA-coupling assertions are therefore marked ``xfail(strict=True,
# raises=AssertionError)`` until W3 populates those frames.  When W3 lands the
# two decorators below come off (and the review advisories in each NA test's
# docstring must be reconciled against the real per-roll frames first).
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


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="W3: per-roll NA frames (period_branch_full / "
    "dt_non_anticipativity) not yet populated in rolling.",
)
def test_rolling_na_frames_couple_branches_per_roll(rolling_result):
    """PRIMARY structural signal: each roll carries the NA precondition.

    ``apply_branch_cluster`` (in ``apply_derived_g``,
    ``flextool/engine_polars/input.py:3842-3845``) must populate, on every
    roll's ``flex_data``, both the ``period_branch_full`` (d, b) anchor→sibling
    map — with a non-empty ``d != b`` sub-cohort — and the
    ``dt_non_anticipativity`` (d, t) tie set.  That is exactly the gate the
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


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="W3: per-roll NA frames (period_branch_full / "
    "dt_non_anticipativity) not yet populated in rolling.",
)
def test_rolling_na_ties_storage_across_branches(rolling_result):
    """Storage NA ties ``v_state`` across branches at the tied (d, t) cells.

    This fixture authors NO unit commitment (no ``online_method`` /
    ``startup_method`` in ``stochastics.json``), so ``v_online_integer`` never
    exists — the variable NA actually ties here is ``v_state`` (storage params
    ARE present).  The storage non-anticipativity family pins the per-(n, t)
    net charge across siblings, which is LP-equivalent to
    ``v_state[n, d, t] == v_state[n, b, t]`` at every tied timestep.

    Branches legitimately DIVERGE at non-tied steps, so the cross-branch
    equality is restricted to the tied timesteps carried by that roll's
    ``fd.dt_non_anticipativity``.  At those timesteps the spread of
    ``v_state`` across the branch-period axis must collapse to ~0; the test
    also asserts that a positive number of tied cells was actually checked so
    the witness can never be silently vacuous.
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

        # Restrict to the tied timesteps (the (d, t) tie set's ``t`` axis).
        # ``v_state``'s ``d`` is the branch-period axis, so keeping every
        # branch at the tied timesteps is what lets the cross-branch spread
        # be witnessed; cast ``t`` to Utf8 on both sides to join across the
        # Enum vocabularies.
        tied_t = (dtna.select(pl.col("t").cast(pl.Utf8).alias("__t"))
                      .unique())
        vs_tied = (vs.with_columns(pl.col(tcol).cast(pl.Utf8).alias("__t"))
                     .join(tied_t, on="__t", how="inner"))
        if vs_tied.height == 0:
            continue

        # Spread of v_state across the branch-period axis at each (n, t).
        spread = (vs_tied
                  .group_by([ncol, "__t"])
                  .agg(
                      n_branches=pl.col("value").len(),
                      vmax=pl.col("value").max(),
                      vmin=pl.col("value").min(),
                  )
                  .filter(pl.col("n_branches") > 1))
        for row in spread.iter_rows(named=True):
            assert row["vmax"] - row["vmin"] <= tol + 1e-6 * max(
                1.0, abs(row["vmax"]), abs(row["vmin"])
            ), (
                f"roll {name!r}: v_state spread "
                f"{row['vmax'] - row['vmin']} at (n={row[ncol]}, "
                f"t={row['__t']}) exceeds tol across branches"
            )
            checked += 1

    assert checked > 0, (
        "no tied (n, t) v_state cells with >1 branch were checked — the "
        "storage NA witness would be vacuous"
    )
