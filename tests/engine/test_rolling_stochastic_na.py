"""Coverage-first solving e2e for the rolling stochastic UC.

Fixture ``1_week_rolling_wind`` (scenario in ``tests/fixtures/stochastics.json``,
solve ``1week_rolling``): a 168-step week (timeset ``1week`` /
``timeset_duration = 168``) solved as ``solve_mode = rolling_window`` with
``rolling_solve_horizon = 96`` and ``rolling_solve_jump = 24``.  The jump of
24 over a 168-step horizon yields SEVEN rolls (roll starts t0001, t0025,
t0049, t0073, t0097, t0121, t0145 — matching the seven ``stochastic_branches``
reveal points).  Three equal-probability branches (``upper`` / ``lower`` /
``mid``) fan at every reveal alongside the ``realized`` anchor.

**Since 4.7.0** (``dispatch_realization_method`` ``within_solve``, rolling) each
roll realizes its 24-step jump once under ``period1`` on the ``realized``
branch and looks ahead from the jump end with the forecasts made there, so no
roll ties anything across forecasts and no realization solve is added; the
history below describes the roll-start fans that 4.7.0 replaced.

Why this test exists / what it protects
---------------------------------------
This file was written BEFORE W1 as a coverage-first guard.  Before W1 the
rolling solve CRASHED during E0 branch-reveal validation
(``flextool/engine/solve/_stochastic.py`` raised ``FlexToolConfigError`` —
"a stochastic branch must start at the FIRST step of its branching period").
This is the W1/W3 defect tracked in ``specs/stochastics_reassessment.md``
§A.1 / §A.2 / §E / §F.

W1 has now landed: the triggering-cohort validation replaces that first-step
guard, so the rolling solve runs to a strict optimum on every roll and
``test_rolling_stochastic_solve_completes`` asserts for real.

W3 HAS NOW LANDED too.  Root cause: the synthetic-solve early-return in
``_apply_db_overrides`` (``flextool/engine/preprocess/flex_data/_db_overrides.py``) fires for every
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

§D.6 branch-point continuity (inter-branch) is now DELIVERED.  The storage NA
stack ties BOTH the per-(n, t) NET CHARGE (per-step charge delta,
``non_anticipativity_storage_use``) AND absolute ``v_state`` at the FIRST tied step of each
anchor period (``non_anticipativity_storage_state_first``,
both in ``_non_anticipativity.add_dispatch_constraints``).  Telescoping the per-step net-charge ties over the
contiguous jump off that absolute branch-point anchor closes the whole tied
range absolutely, so the constant realized↔forecast ``v_state`` offset
(previously ~0.06 here) collapses to ~0.  The xfail that once guarded this is
gone.

Remaining deferred item (NOT this slice): the HARD roll-to-roll carry-in — all
branches inheriting the previous roll's ACTUAL jump end-state rather than the
current soft ``roll_continue`` seed (spec W8b).  Because that carry-in is still
soft, the SHARED branch-point level is the soft carried value (~0.38 in
continuation rolls), not the nominal 0.5 — so the test below asserts the
branches equal EACH OTHER at the branch point, never equality to 0.5.

The structural signal the frames test asserts is the exact precondition the
operational-NA gate (``_non_anticipativity.add_dispatch_constraints``) requires to emit the four
``non_anticipativity_*`` constraint families: a per-roll ``flex_data`` whose
``period_branch_full`` (with a non-empty ``d != b`` sub-cohort) and
``dt_non_anticipativity`` frames are both populated.
"""
from __future__ import annotations

import os

import polars as pl
import pytest

from flextool.engine.solve._orchestration import run_chain_from_db
from flextool.engine.input._solve_state import FlexToolConfigError

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
    # The stochastic rolls; the fixture's realized: yes designation interleaves a realization roll after each.
    return {k: s for k, s in rolling_result["steps"].items() if "__realized_dispatch" not in k}


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


def test_rolling_rolls_tie_nothing_and_hold_their_jump(rolling_result):
    """Since 4.7.0 each roll of the rolling ``within_solve`` solve realizes its jump under the period's own name:
    ``period1`` holds the 24 jump steps t(24k+1)..t(24k+24) once, on the realized data, and the forecasts
    ``period1_{lower,mid,upper}`` (``period_branch_full`` rows ``period1 -> period1_<f>``) hold only the steps after
    it.  Nothing is decided per forecast over the jump, so nothing is tied: no ``non_anticipativity_*`` row (before:
    the storage use and the first state of the jump, tied across the forecasts) and no ``dt_non_anticipativity``.
    The last roll (t0145-t0168) is its jump only."""
    from tests.engine._unit_db import build

    steps = _steps(rolling_result)
    assert len(steps) == EXPECTED_ROLLS
    for k, (name, s) in enumerate(steps.items()):
        fd = s.flex_data
        assert fd is not None, f"roll {name!r} retained no flex_data"
        dt = fd.dt.select(pl.col("d").cast(pl.Utf8), pl.col("t").cast(pl.Utf8))
        jump = [f"t{_JUMP * k + h:04d}" for h in range(1, _JUMP + 1)]
        assert sorted(dt.filter(pl.col("d") == "period1")["t"].to_list()) == jump, name
        assert not [n for n in build(fd).cstr_names() if n.startswith("non_anticipativity")], name
        tails = dt.filter(pl.col("d") != "period1")
        if k == EXPECTED_ROLLS - 1:
            assert tails.height == 0, name
            continue
        assert fd.dt_non_anticipativity is None, name
        assert set(tails["d"].to_list()) == {"period1_lower", "period1_mid", "period1_upper"}, name
        assert tails["t"].min() == f"t{_JUMP * (k + 1) + 1:04d}", name
        pbf = fd.period_branch_full.with_columns(pl.col("d").cast(pl.Utf8), pl.col("b").cast(pl.Utf8))
        assert set(pbf.filter(pl.col("d") != pl.col("b")).rows()) == {
            ("period1", f"period1_{f}") for f in ("lower", "mid", "upper")}, name


def test_rolling_storage_of_the_jump_is_one_path(rolling_result):
    """The storage of the jump is decided once: ``hydro_reservoir``'s ``v_state`` has one row per jump step, on
    ``period1`` (before 4.7.0: one per forecast, tied across them by the net-charge and first-state rows), and each
    forecast's state starts after the jump, from the jump's last step (its first state change continues from
    ``(period1, t(24k+24))``, the seam link ``dtttdt``)."""
    steps = _steps(rolling_result)
    checked = 0
    for k, (name, s) in enumerate(list(steps.items())[:-1]):
        vs = s.solution.value("v_state")
        vs = vs.filter(pl.col("n").cast(pl.Utf8) == _STORAGE_NODE).select(
            pl.col("d").cast(pl.Utf8), pl.col("t").cast(pl.Utf8))
        jump = vs.filter(pl.col("d") == "period1")
        assert jump.height == _JUMP and jump["t"].n_unique() == _JUMP, name
        last = f"t{_JUMP * (k + 1):04d}"
        first = f"t{_JUMP * (k + 1) + 1:04d}"
        link = s.flex_data.dtttdt.select(*(pl.col(c).cast(pl.Utf8) for c in ("d", "t", "d_previous",
                                                                            "t_previous_within_solve")))
        for f in ("lower", "mid", "upper"):
            assert vs.filter(pl.col("d") == f"period1_{f}")["t"].min() == first, (name, f)
            assert link.filter((pl.col("d") == f"period1_{f}") & (pl.col("t") == first)).rows() == [
                (f"period1_{f}", first, "period1", last)], (name, f)
            checked += 1
    assert checked == 3 * (EXPECTED_ROLLS - 1)


# rolling_solve_jump for this fixture (see module docstring).
_JUMP = 24
_STORAGE_NODE = "hydro_reservoir"


def test_rolling_handoff_carries_trunk_committed_jump_end_storage(rolling_result):
    """PRODUCER-LEVEL guard for the H1 rolling-storage handoff fix.

    Since 4.7.0 each roll realizes its jump under the period ``period1`` itself, so it produces the carry: its
    own ``v_state`` at the COMMITTED JUMP-END (``period1``, t(24k+24)) × the state unitsize, consumed by the next
    roll as ``p_roll_continue_state``.  It must not carry a forecast branch's state at the discarded look-ahead
    HORIZON-END: the ``roll_end_state`` producer (``flex_data/_handoff.py``) reads the Provider anchor
    ``solve_data/realized_dispatch`` (the realized jump only), not the forecast-expanded
    ``flex_data.realized_dispatch``, whose lexically-last ``(d, t)`` is a forecast at the window horizon.

    Two independent defects are witnessed, so a future regression of EITHER surfaces here:

    * **wrong BRANCH** — the carry must come from ``period1``, not a forecast copy (HEAD once picked
      ``period1_upper``);
    * **wrong STEP** — the carry must be the committed jump-end step, not the discarded window-horizon step.

    Discriminator on ``1_week_rolling_wind``: on continuation rolls 0-5 the committed jump-end ``v_state``
    differs from the forecast window-horizon value (``period1_upper`` @ window end, a non-zero end-reference
    pin), so a carry of the horizon value is caught.  Roll 6 is the LAST roll: its jump reaches the window
    horizon and no next roll consumes the carry; it is excluded from the strict guard.
    """
    steps = _steps(rolling_result)
    names = list(steps)
    assert len(names) == EXPECTED_ROLLS, (
        f"expected {EXPECTED_ROLLS} rolls, got {len(names)}")
    assert list(rolling_result["steps"]) == names, "a realization solve was added"

    for i, name in enumerate(names):
        s = steps[name]
        res = s.handoff.roll_end_state
        assert res is not None, f"roll {name!r}: roll_end_state unexpectedly None"
        carried = {r["node"]: r["value"] for r in res.iter_rows(named=True)}
        assert _STORAGE_NODE in carried, (
            f"roll {name!r}: {_STORAGE_NODE} absent from roll_end_state "
            f"(got {sorted(carried)})")
        carried_v = carried[_STORAGE_NODE]

        sol = s.solution
        assert sol is not None, f"roll {name!r} retained no solution"
        vs = sol.value("v_state")
        cols = vs.columns
        ncol = _pick(cols, "n", "node")
        dcol = _pick(cols, "d", "period")
        tcol = _pick(cols, "t", "time")
        sub = (vs.filter(pl.col(ncol).cast(pl.Utf8) == _STORAGE_NODE)
                 .with_columns(pl.col(dcol).cast(pl.Utf8).alias("__d"),
                               pl.col(tcol).cast(pl.Utf8).alias("__t")))
        assert sub.height > 0, f"roll {name!r}: no {_STORAGE_NODE} v_state rows"
        horizon_t = sub["__t"].max()
        jump_end_t = f"t{_JUMP * (i + 1):04d}"

        def _vstate(d, t, _sub=sub):
            r = _sub.filter((pl.col("__d") == d) & (pl.col("__t") == t))
            return r["value"][0] if r.height else None

        # The period period1 dispatches the jump: its v_state at the jump end is the carry.
        jump_end_v = _vstate("period1", jump_end_t)
        assert jump_end_v is not None, f"roll {name!r}: no v_state at (period1, {jump_end_t})"
        su = (s.flex_data.p_state_unitsize.frame
              .filter(pl.col("n").cast(pl.Utf8) == _STORAGE_NODE)["value"][0])
        assert carried_v == pytest.approx(float(jump_end_v) * su, abs=1e-6), (name, carried_v)
        if i == EXPECTED_ROLLS - 1:
            assert jump_end_t == horizon_t, (
                f"last roll {name!r}: jump-end {jump_end_t} != horizon "
                f"{horizon_t} (the roll should realize to the window end)")
            continue
        fcast_horizon = _vstate("period1_upper", horizon_t)
        assert fcast_horizon is not None, (
            f"roll {name!r}: no forecast v_state at horizon "
            f"(period1_upper, {horizon_t})")

        # Continuation roll: the committed jump-end strictly precedes the discarded look-ahead window horizon
        # (the step distinction is real).
        assert jump_end_t < horizon_t, (
            f"roll {name!r}: jump-end {jump_end_t} not before horizon {horizon_t}")
        # (1) VALUE — the next roll starts from the carried jump-end state.
        nxt = steps[names[i + 1]].flex_data.p_roll_continue_state
        assert nxt is not None, f"roll {names[i + 1]!r}: no p_roll_continue_state"
        nxt_v = (nxt.frame.filter(pl.col("n").cast(pl.Utf8) == _STORAGE_NODE)["value"].to_list())
        assert nxt_v == [pytest.approx(carried_v, abs=1e-9)], (names[i + 1], nxt_v, carried_v)
        # (2) BRANCH and STEP identity — the FORECAST branch horizon value differs from the carried jump-end, so
        #     the carry is NOT a forecast-branch horizon value (guards the wrong-BRANCH and the wrong-STEP defects).
        #     This is the exact (period1_upper, horizon) cell HEAD once lexically selected.
        assert abs(fcast_horizon * su - carried_v) > 1e-6, (
            f"roll {name!r}: carried {carried_v} coincides with the forecast "
            f"horizon-end {fcast_horizon} — HEAD's wrong source")
