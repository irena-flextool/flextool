"""OC1 rolling + stochastic separate_dispatch — interleaved UC/dispatch rolls
with a two-source handoff, each dispatch roll scoped to its realized jump.

A ``dispatch_realization_method=separate_dispatch`` solve under
``solve_mode=rolling_window`` expands to the INTERLEAVED order
``[uc_roll_0, disp_roll_0, uc_roll_1, disp_roll_1, ...]`` where, per roll, a
stochastic UC solve (forecast fan) is followed by a deterministic
realized-dispatch sibling.  Two carrier flows are kept separate:

  * PIN: each ``disp_roll_k`` fixes its ``v_online`` to ``uc_roll_k``'s
    committed hedge (``online_fix_eq`` / ``fix_online``), consuming it from
    its sequential predecessor ``uc_roll_k``.
  * ROLL-FORWARD: the realized trajectory flows disp→disp — ``disp_roll_k``
    takes its roll-forward (``roll_end_online`` → ``p_roll_continue_online``,
    etc.) from ``disp_roll_{k-1}``, NOT from ``uc_roll_k``.

**SCOPE (the correctness invariant this test pins).** Each ``disp_roll_k``
solves ONLY its realized JUMP window (the committed section), NOT the UC
parent's full roll horizon (jump + look-ahead).  The UC parent legitimately
spans jump + look-ahead and fans/discards the look-ahead; the dispatch
sibling must realize only the jump — the look-ahead has no realized data and
no online pin, so solving it would force spurious unserved load (which it did
before the fix: the dispatch solved all 48 steps, the zero-wind look-ahead
half was unserved).  ``test_dispatch_rolls_scoped_to_jump`` asserts the
dispatch solves exactly ``JUMP_STEPS`` (24) steps, strictly fewer than the
UC's ``HORIZON_STEPS`` (48) — a solver-independent guard that FAILS on the
pre-fix behaviour (dispatch solved 48) and catches this whole bug class.

This is pure orchestration + handoff routing — NO new var/cstr/param family
(the pin family ``online_fix_eq`` is already registered), so CLAUDE.md
autoscale invariant #1 is not triggered.

Fixture (``rolling_separate_dispatch.json``, built from JSON per CLAUDE.md
invariant 3): 72 steps = 3 jump-24 rolls (horizon 48).  A committable BINARY
fleet (MIP): ``gas_plant`` (2 units × 250 MW, binary, min_load 0.4,
min_up/down 3) and ``coal_plant`` (3 units × 600 MW, binary, efficiency 0.4,
min_load 0.4, min_up/down 4, startup_cost 100) back ``wind`` (branch-varying
forecast; freely curtailable) + ``hydro`` (small reservoir water-loss penalty)
against 3000 MW demand; the ``rlz``/``upper``/``mid``/``lower`` fan re-reveals
at each roll boundary.  Operational only (no invest).

Owner-confirmed semantics (``==`` pin): the realized dispatch is LOCKED to the
plan's committed hedge (per-unit ``==``/``≤`` is a future setting).  The test
asserts ZERO unserved load (up-slack) in every realized-dispatch roll; the
``==`` lock may spill via the free curtailment / small water penalty.

NOTE on anchors: only the realized-DISPATCH objectives are pinned as
regression anchors (deterministic — with ``v_online`` pinned the dispatch is a
pure LP).  The UC-parent objectives are intentionally NOT anchored: they are
dominated by VOLL in the DISCARDED look-ahead half (the demo's low-wind /
energy-limited-hydro look-ahead is not fully adequate), which is pre-existing
rolling behaviour independent of this feature and would make a brittle,
misleading anchor.  Everything else is asserted as a structural invariant.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import polars as pl
import pytest

FLEXTOOL_ROOT = Path(__file__).resolve().parents[2]
FIXTURE = FLEXTOOL_ROOT / "tests" / "fixtures" / "rolling_separate_dispatch.json"

SCENARIO = "rolling_separate_dispatch"
UC = "1week_rolling_roll_{k}"
DISP = "1week_rolling__realized_dispatch_roll_{k}"
N_ROLLS = 3
GAS = "gas_plant"
COAL = "coal_plant"
REALIZED_PERIOD = "period1"            # the realized trunk the sibling owns
GAS_MIN_LOAD = 0.4                     # per-unit min-load fraction

# Rolling config from the fixture (``solve.jump`` / ``solve.horizon`` hours at
# 1 h/step): each roll optimises HORIZON steps, commits/realises JUMP steps.
JUMP_STEPS = 24
HORIZON_STEPS = 48

# Per-roll committed gas hedge (solver-determined tied forecast commitment,
# pinned verbatim into the dispatch sibling): gas is committed as ONE unit.
HEDGE = {0: 1.0, 1: 1.0, 2: 1.0}

# Realized-DISPATCH per-roll objectives (deterministic LP regression anchors;
# ``v_online`` pinned → the dispatch is a pure LP).  UC-parent objectives are
# deliberately NOT anchored (discarded-look-ahead VOLL — see module docstring).
DISP_OBJ = {
    "1week_rolling__realized_dispatch_roll_0": 34650718.860,
    "1week_rolling__realized_dispatch_roll_1": 32678805.286,
    "1week_rolling__realized_dispatch_roll_2": 31490226.310,
}


def _build_db(tmp_path: Path) -> str:
    if not FIXTURE.exists():
        pytest.skip(f"fixture JSON not present: {FIXTURE}")
    tests_dir = FLEXTOOL_ROOT / "tests"
    if str(tests_dir) not in sys.path:
        sys.path.insert(0, str(tests_dir))
    from db_utils import json_to_db  # noqa: E402
    from flextool.update_flextool.db_migration import migrate_database

    db_path = tmp_path / f"{FIXTURE.stem}.sqlite"
    url = json_to_db(FIXTURE, db_path)
    migrate_database(url)
    return url


def _run_chain(tmp_path: Path):
    from flextool.engine_polars import run_chain_from_db

    url = _build_db(tmp_path)
    work_folder = tmp_path / "work"
    work_folder.mkdir()
    prev = os.environ.get("FLEXTOOL_AUTOSCALE_STRICT")
    os.environ["FLEXTOOL_AUTOSCALE_STRICT"] = "1"
    try:
        steps = run_chain_from_db(
            url, SCENARIO, work_folder=work_folder,
            warm=False, keep_solutions=True,
        )
    finally:
        if prev is None:
            os.environ.pop("FLEXTOOL_AUTOSCALE_STRICT", None)
        else:
            os.environ["FLEXTOOL_AUTOSCALE_STRICT"] = prev
    assert steps, "run_chain_from_db produced no orchestration steps"
    return steps, work_folder


@pytest.fixture(scope="module")
def chain(tmp_path_factory):
    return _run_chain(tmp_path_factory.mktemp("rolling_separate_dispatch"))


def _online(step, p: str) -> pl.DataFrame:
    """``v_online`` for process ``p`` — integer (binary UC) or linear."""
    sol = step.solution
    for vn in ("v_online_integer", "v_online_linear"):
        if vn in sol._vars:
            f = sol.value(vn).filter(pl.col("p") == p)
            if f.height:
                return f
    return pl.DataFrame(schema={"p": pl.Utf8, "d": pl.Utf8,
                                "t": pl.Utf8, "value": pl.Float64})


def _n_time_steps(step) -> int:
    """Distinct time steps the solve actually spans (branch-independent)."""
    return int(step.solution.value("v_flow")["t"].n_unique())


@pytest.mark.solver
def test_is_mip_binary_commitment(chain):
    """The committable fleet is BINARY — gas and coal both carry an integer
    ``v_online`` (MIP), the realistic UC formulation."""
    steps, _ = chain
    uc0 = steps[UC.format(k=0)].solution
    assert "v_online_integer" in uc0._vars, "expected a binary (MIP) UC"
    procs = set(uc0.value("v_online_integer")["p"].unique().to_list())
    assert GAS in procs and COAL in procs, (
        f"both gas and coal should be binary-committable; got {procs}"
    )


@pytest.mark.solver
def test_interleaved_expansion_order(chain):
    """R1: the solve expands to the INTERLEAVED order
    ``[uc_0, disp_0, uc_1, disp_1, uc_2, disp_2]``."""
    steps, _ = chain
    expected = []
    for k in range(N_ROLLS):
        expected.append(UC.format(k=k))
        expected.append(DISP.format(k=k))
    assert list(steps) == expected


@pytest.mark.solver
def test_dispatch_rolls_scoped_to_jump(chain):
    """SCOPE INVARIANT (the correctness guard).  Each realized-dispatch roll
    solves EXACTLY its realized JUMP window (``JUMP_STEPS`` steps), strictly
    fewer than the UC parent's full HORIZON (``HORIZON_STEPS``).  The dispatch
    must realise only the committed jump — NOT the UC's look-ahead, which has
    no realized data or online pin and forced spurious unserved load before
    the fix.  Solver-independent (step counts from the fixture's rolling
    config); FAILS on the pre-fix behaviour where the dispatch solved all
    ``HORIZON_STEPS`` (48)."""
    steps, _ = chain
    assert JUMP_STEPS < HORIZON_STEPS   # sanity: look-ahead exists to exclude
    for k in range(N_ROLLS):
        uc_steps = _n_time_steps(steps[UC.format(k=k)])
        disp_steps = _n_time_steps(steps[DISP.format(k=k)])
        # UC parent spans the full horizon (roll 2 is the horizon-truncated
        # tail of the 72-step timeline, so it may be shorter — bound it).
        assert uc_steps > JUMP_STEPS or k == N_ROLLS - 1, (
            f"UC roll {k} spans {uc_steps} steps; expected > jump ({JUMP_STEPS})"
        )
        # The dispatch roll solves EXACTLY the jump — not the look-ahead.
        assert disp_steps == JUMP_STEPS, (
            f"dispatch roll {k} solves {disp_steps} steps, expected exactly "
            f"the jump ({JUMP_STEPS}); solving the look-ahead is the bug"
        )
        assert disp_steps < uc_steps or k == N_ROLLS - 1, (
            f"dispatch roll {k} ({disp_steps}) must be scoped tighter than its "
            f"UC parent ({uc_steps})"
        )


@pytest.mark.solver
def test_each_dispatch_roll_pinned_to_its_uc_commitment(chain):
    """R2 pin: each ``disp_roll_k`` gas ``v_online`` == ``uc_roll_k``'s
    committed hedge (the tied forecast commitment, excluding the weight-0
    realized trunk branch), non-degenerate."""
    steps, _ = chain
    for k in range(N_ROLLS):
        uc = _online(steps[UC.format(k=k)], GAS)
        disp = _online(steps[DISP.format(k=k)], GAS)
        uc_hedge = uc.filter(pl.col("d") != REALIZED_PERIOD)["value"].max()
        disp_commit = disp.filter(
            pl.col("d") == REALIZED_PERIOD)["value"].max()
        assert disp_commit == pytest.approx(uc_hedge, rel=1e-6), (
            f"roll {k}: disp commit {disp_commit} != uc hedge {uc_hedge}"
        )
        assert disp_commit == pytest.approx(HEDGE[k], rel=1e-6)
        assert disp_commit > 0.0, f"roll {k}: dispatch commitment is zero"


@pytest.mark.solver
def test_gas_dispatch_runs_above_min_load(chain):
    """Gas is a genuine marginal committable: committed as ONE unit yet its
    OUTPUT rises ABOVE the min-load floor in some hours (not pinned at the
    floor) — the realistic UC behaviour the 2×250 MW capacity limit forces.
    Checked on the realized-dispatch rolls (floor = ``min_load * v_online``)."""
    steps, _ = chain
    any_above = False
    for k in range(N_ROLLS):
        disp = steps[DISP.format(k=k)]
        on = _online(disp, GAS).select("d", "t", "value").rename(
            {"value": "online"})
        flow = (disp.solution.value("v_flow")
                .filter(pl.col("p") == GAS)
                .group_by(["d", "t"]).agg(pl.col("value").sum().alias("fl")))
        j = on.join(flow, on=["d", "t"], how="left").with_columns(
            floor=pl.col("online") * GAS_MIN_LOAD)
        if j.filter(pl.col("fl") > pl.col("floor") + 1e-3).height > 0:
            any_above = True
    assert any_above, (
        "gas never ran above its min-load floor in any dispatch roll — the "
        "fleet is not exercising genuine UC dispatch"
    )


@pytest.mark.solver
def test_coal_committed_and_used(chain):
    """Coal (the cheap baseload) is committed in the realized dispatch (the
    capacity-limited gas cannot cover demand alone)."""
    steps, _ = chain
    coal = _online(steps[DISP.format(k=0)], COAL)
    assert coal.height > 0, "coal has no online var"
    assert coal["value"].max() > 0.5, "coal is never committed in dispatch"


@pytest.mark.solver
def test_realized_trajectory_carries_dispatch_to_dispatch(chain):
    """R2 roll-forward: ``disp_roll_k`` rolls forward on ``disp_roll_{k-1}``
    — its consumed ``p_roll_continue_online`` equals the PREVIOUS dispatch
    roll's realized end commitment (``roll_end_online``), nonzero.  (The
    first dispatch roll has no carry.)"""
    steps, _ = chain

    def _reo(k):   # realized end commitment handed off by disp_roll_k
        h = steps[DISP.format(k=k)].handoff.roll_end_online
        g = h.filter(pl.col("process") == GAS) if h is not None else None
        return float(g["value"][0]) if g is not None and g.height else 0.0

    def _prc(k):   # roll-forward commitment consumed by disp_roll_k
        p = steps[DISP.format(k=k)].flex_data.p_roll_continue_online
        g = p.frame.filter(pl.col("p") == GAS) if p is not None else None
        return float(g["value"][0]) if g is not None and g.height else None

    assert _prc(0) is None          # first roll: no prior dispatch
    for k in range(1, N_ROLLS):
        carried = _prc(k)
        prev_end = _reo(k - 1)
        assert carried is not None, f"disp roll {k} consumed no roll-forward"
        assert carried == pytest.approx(prev_end, rel=1e-6), (
            f"roll {k}: consumed {carried} != prev dispatch end {prev_end}"
        )
        assert carried > 0.0, (
            f"roll {k}: realized carry is zero (weight-0 garbage, not a real "
            f"dispatch trajectory)"
        )


@pytest.mark.solver
def test_all_rolls_optimal_and_realized_demand_met(chain):
    """All six solves OPTIMAL, and every REALIZED DISPATCH roll serves all
    demand — ZERO unserved load (no up-slack VOLL).  This is meaningful ONLY
    because the dispatch is scoped to the jump (see
    ``test_dispatch_rolls_scoped_to_jump``): before the fix the dispatch
    solved the dataless look-ahead and this assertion failed."""
    steps, _ = chain
    for name, st in steps.items():
        assert st.optimal is True, f"{name} not optimal"
    for k in range(N_ROLLS):
        sol = steps[DISP.format(k=k)].solution
        if "vq_state_up" in sol._vars:
            up = sol.value("vq_state_up").filter(pl.col("value") > 1e-3)
            assert up.height == 0, (
                f"dispatch roll {k}: {up.height} unserved-load (up-slack) "
                f"rows — realized demand not met"
            )


@pytest.mark.solver
def test_dispatch_per_solve_objectives(chain):
    """Regression anchors for the realized-DISPATCH objectives (deterministic
    LP — ``v_online`` is pinned).  UC-parent objectives are deliberately not
    anchored (discarded-look-ahead VOLL; see module docstring)."""
    steps, _ = chain
    for name, expected in DISP_OBJ.items():
        assert name in steps, f"missing solve {name}"
        got = steps[name].solution.obj
        assert got == pytest.approx(expected, rel=1e-6), (
            f"{name}: objective {got} != {expected}"
        )
