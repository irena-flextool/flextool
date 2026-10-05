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
Today the rolling solve CRASHES during E0 branch-reveal validation
(``flextool/engine_polars/_stochastic.py:466-476`` raises
``FlexToolConfigError`` — "a stochastic branch must start at the FIRST step
of its branching period").  This is the W1/W3 defect tracked in
``specs/stochastics_reassessment.md`` §A.1 / §A.2 / §E / §F.

This file lands BEFORE that fix.  It keeps the suite GREEN now by documenting
the crash as a strict xfail (``raises=FlexToolConfigError``) and flips to real
pass-assertions with a SINGLE one-line change (``W1_W3_LANDED = True``) once
W1 + W3 merge.  Until then every test memoises and re-raises the one
``FlexToolConfigError`` so pytest reports ``xfailed`` (never ``failed`` /
``xpassed``).

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

# >>> ONE-LINE FLIP: set True once W1 + W3 are merged. <<<
#
# Before flipping, reconcile the POST-FLIP assertions in
# ``test_rolling_na_ties_storage_across_branches`` against the real per-roll NA
# frames (they cannot be observed until W1/W3 exist — auto-memory
# "Rolling stochastic broken" records that the current stack drops all NA
# coupling in rolling).  Two review advisories:
#   (1) The storage-NA family ties per-(n,t) NET CHARGE (a delta), not absolute
#       ``v_state`` (model.py:4505).  Absolute-state equality holds only when the
#       tied window is a prefix from a shared initial state; a mid-horizon roll
#       with a per-branch-free initial state can satisfy the net-charge tie with
#       a constant ``v_state`` offset.  If so, assert on the tied net-charge LHS,
#       not absolute ``v_state``.
#   (2) This fixture reveals multiple times within one 96-step horizon, so tie
#       windows differ per reveal.  Restrict the cross-branch comparison to the
#       exact tied (d, t) PAIRS, not just the union of tied ``t`` across all
#       branch-periods, or a legitimately-diverged branch gets compared against a
#       still-tied one.
# The PRIMARY structural signal (test_rolling_na_frames_couple_branches_per_roll)
# is unaffected by both and is the load-bearing coverage assertion.
W1_W3_LANDED = False

pytestmark = [
    pytest.mark.solver,
    pytest.mark.xfail(
        not W1_W3_LANDED,
        strict=True,
        raises=FlexToolConfigError,
        reason="E0 validation crashes rolling stochastic UC "
        "(flextool/engine_polars/_stochastic.py:466-476); "
        "W1/W3 land after this coverage test — spec §A.1/§E/§F.",
    ),
]


@pytest.fixture(scope="module")
def rolling_result(stochastic_db_url, test_solver_config_dir, tmp_path_factory):
    """Run the rolling stochastic scenario end-to-end ONCE for the module.

    Memoises either the per-step dict (post-fix) or the
    ``FlexToolConfigError`` the E0 validation raises today, so each test can
    re-raise it in its own body and report a strict xfail.  The
    ``run_chain_from_db`` call runs under ``FLEXTOOL_AUTOSCALE_STRICT=1`` so
    that any future autoscale-registry gap in the NA path is loud rather than
    silently reverting to an un-scaled LP (invariant #1).
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
    """Return the per-step dict, or re-raise the memoised E0 crash.

    Pre-fix this re-raises ``FlexToolConfigError`` inside the test body, which
    the module-level strict xfail converts to ``xfailed``.  Post-fix it hands
    back the live ``{solve_name: OrchestrationStep}`` dict.
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
