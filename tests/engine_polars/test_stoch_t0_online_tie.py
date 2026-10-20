"""OC1 #1 — the ONLINE commitment tie over the JUMP + the separate_dispatch
sibling v_online PIN.

Unit commitment on the committed/realized section (the "jump") is here-and-
now: it is chosen before the scenario is known, so ``v_online`` is tied EQUAL
across the forecast branches over the whole jump.  The dispatch sibling then
FIXES that committed ``v_online`` over the jump while flows realize.

Mechanism (``model._add_non_anticipativity_constraints``): the online non-
anticipativity families join ``dtna_online`` = ``dt_non_anticipativity`` when
``has_dtna``, ELSE (a ``separate_dispatch`` PARENT hands its realized window
to the sibling, so its ``dt_non_anticipativity`` is empty) the FULL set of
online ``(d, t)`` steps for the anchor forecast periods — the whole horizon =
the jump.  So:

  * plain single-shot stochastic — ``dtna`` = its whole realized window →
    online already tied over that window (UNCHANGED, existing behaviour);
  * ``separate_dispatch`` parent — ``dtna`` empty → the ``jump_steps`` branch
    ties online over the whole horizon (the fix; was tying NOTHING);
  * rolling / mid-horizon / deterministic — ``dtna_online`` == ``dtna``
    (byte-parity).

Sibling pin (sub-slice b): a new ``fix_online`` SolveHandoff carrier ferries
the parent's committed ``v_online`` over the jump (read from a FORECAST branch
via the tie — the well-defined value, NOT the degenerate weight-0 realized
branch — re-keyed to the realized trunk period) into the sibling, where
``online_fix_eq`` pins the sibling's ``v_online`` to it.

Fixture (``stoch_t0_commit.json``, built from JSON per CLAUDE.md invariant 3):
a fan-at-first ``separate_dispatch`` stochastic solve on ``p2035`` (branches
``rlz`` w=0, ``high`` w=0.5, ``low`` w=0.5; reveal at t0001) over 3 steps.
A committable ``peaker`` (existing 100 MW, virtual_unitsize 50 -> 2 units,
min_load 0.5, startup_cost 1, other_operational_cost 10 CUR/MWh) backs up
``wind`` (existing 100 MW, upper_limit, CF rlz 0.5 / high 0.8 / low 0.0)
against 100 MW demand.

Hand-calc (H = 8760/3 = 2920 h/step):
  * PARENT ``stoch_rd``: ``low`` has no wind -> peaker covers the full 100 MW
    -> v_online = 2.0 every step.  The jump tie forces ``high`` to the same
    2.0 (hedge); min_load floors high's output at 0.5*2.0*50 = 50 MW.  The
    ``rlz`` branch carries weight 0 and is NOT tied (forecast-to-forecast) ->
    v_online 0.  Objective = Σ_branch w * 10 * 2920 * Σ_t output
      = 0.5*10*2920*(50*3)  [high]  = 2 190 000
      + 0.5*10*2920*(100*3) [low]   = 4 380 000  ->  6 570 000.
    (Commitment constant over the horizon -> cyclic startup 0.)
  * SIBLING ``stoch_rd__realized_dispatch``: dispatches the realized rlz wind
    0.5 -> wind 50, peaker 50; the pin fixes v_online = 2.0 (the parent's
    commitment) — binding (free dispatch would pick 1.0) but feasible (min_load
    floor 0.5*2.0*50 = 50 = the realized need, no VOLL).  Objective =
    50 MW * 10 * 2920 * 3 = 4 380 000.
"""
from __future__ import annotations

import base64
import json
import os
import sys
from pathlib import Path

import polars as pl
import pytest

FLEXTOOL_ROOT = Path(__file__).resolve().parents[2]
FIXTURE = FLEXTOOL_ROOT / "tests" / "fixtures" / "stoch_t0_commit.json"

SCENARIO = "stoch_t0_commit"
PARENT = "stoch_rd"
SIBLING = "stoch_rd__realized_dispatch"
REALIZED = "p2035"          # the realized / trunk period (weight-0 branch)
HIGH = "p2035_high"
LOW = "p2035_low"
STEPS = ("t0001", "t0002", "t0003")

TIED_COMMIT = 2.0           # low needs the full 2 units -> the tied value
PARENT_OBJECTIVE = 6_570_000.0
SIBLING_OBJECTIVE = 4_380_000.0


def _build_db(fixture_json: Path, tmp_path: Path) -> str:
    """Build-from-JSON + migrate (CLAUDE.md invariant 3; never read a
    checked-in .sqlite)."""
    if not fixture_json.exists():
        pytest.skip(f"fixture JSON not present: {fixture_json}")
    tests_dir = FLEXTOOL_ROOT / "tests"
    if str(tests_dir) not in sys.path:
        sys.path.insert(0, str(tests_dir))
    from db_utils import json_to_db  # noqa: E402
    from flextool.update_flextool.db_migration import migrate_database

    db_path = tmp_path / f"{fixture_json.stem}.sqlite"
    url = json_to_db(fixture_json, db_path)
    migrate_database(url)
    return url


def _run_chain(fixture_json: Path, tmp_path: Path):
    """Run the chain once under ``FLEXTOOL_AUTOSCALE_STRICT=1`` (warm=False,
    keep_solutions=True); return ``(steps, work_folder)``."""
    from flextool.engine_polars import run_chain_from_db

    url = _build_db(fixture_json, tmp_path)
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


def _write_variant(tmp_path: Path, name: str, mutate) -> Path:
    d = json.loads(FIXTURE.read_text())
    mutate(d)
    out = tmp_path / name
    out.write_text(json.dumps(d))
    return out


def _drop_separate_dispatch(d: dict) -> None:
    """Plain stochastic: remove the separate_dispatch designation (its
    ``dt_non_anticipativity`` is then the whole realized window — the
    existing whole-window tie path)."""
    d["parameter_values"] = [
        r for r in d["parameter_values"]
        if r[2] != "dispatch_realization_method"
    ]


def _to_integer_online(d: dict) -> None:
    """Integer online: ``startup_method=binary`` (-> METHOD_MIP ->
    v_online_integer) and ``high`` wind CF -> 1.0 so high's own optimum is 0
    units — the jump tie forces it UP to low's 1 unit (a genuine integer
    hedge)."""
    def _encs(x):
        return [base64.b64encode(json.dumps(x).encode()).decode(), "str"]
    for r in d["parameter_values"]:
        if r[1] == "peaker" and r[2] == "startup_method":
            r[3] = _encs("binary")
        if r[1] == "wprof" and r[2] == "profile":
            m = json.loads(base64.b64decode(r[3][0]).decode())
            for branch in m["data"]:
                if branch[0] == "high":
                    for ts in branch[1]["data"]:
                        for cell in ts[1]["data"]:
                            cell[1] = 1.0
            r[3] = [base64.b64encode(json.dumps(m).encode()).decode(), r[3][1]]


def _online(chain, solve: str, var: str) -> pl.DataFrame:
    steps, _ = chain
    assert solve in steps, f"no {solve!r} step in {list(steps)}"
    sol = steps[solve].solution
    assert var in sol._vars, f"{var} not in {solve} vars: {list(sol._vars)}"
    return sol.value(var)


def _at(fr: pl.DataFrame, d: str, t: str) -> float:
    row = fr.filter((pl.col("d") == d) & (pl.col("t") == t))
    assert row.height == 1, f"expected one ({d},{t}) row, got {row.height}"
    return float(row["value"][0])


def _obj(wf: Path, solve: str) -> float:
    return float(
        pl.read_parquet(wf / "output_raw" / f"v_obj__{solve}.parquet")
        ["objective"][0]
    )


@pytest.fixture(scope="module")
def sep_chain(tmp_path_factory):
    return _run_chain(FIXTURE, tmp_path_factory.mktemp("stoch_t0_commit_sep"))


@pytest.fixture(scope="module")
def plain_chain(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("stoch_t0_commit_plain")
    jf = _write_variant(tmp, "stoch_t0_commit_plain.json",
                        _drop_separate_dispatch)
    return _run_chain(jf, tmp)


@pytest.fixture(scope="module")
def integer_chain(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("stoch_t0_commit_int")
    jf = _write_variant(tmp, "stoch_t0_commit_integer.json", _to_integer_online)
    return _run_chain(jf, tmp)


# ---------------------------------------------------------------------------
# separate_dispatch PARENT — the jump tie (the broken-case fix)
# ---------------------------------------------------------------------------

@pytest.mark.solver
def test_parent_v_online_tied_over_jump(sep_chain):
    """The parent's committed v_online is EQUAL across the forecast branches
    over the WHOLE jump (all steps), driven to the non-degenerate hedge value
    (low's full 2 units) — NOT the degenerate realized branch."""
    fr = _online(sep_chain, PARENT, "v_online_linear")
    for t in STEPS:
        high = _at(fr, HIGH, t)
        low = _at(fr, LOW, t)
        assert high == pytest.approx(low, abs=1e-9), f"not tied at {t}"
        assert high == pytest.approx(TIED_COMMIT, abs=1e-9), (
            f"tied value at {t} = {high}, expected {TIED_COMMIT}"
        )


@pytest.mark.solver
def test_parent_realized_branch_excluded(sep_chain):
    """The weight-0 realized branch is NOT part of the forecast-to-forecast
    tie — it is free and parks at 0 (this is why the carried commitment must
    be read from a forecast branch, fixing the first prototype's VOLL blocker
    where pinning the sibling to the degenerate realized 0 forced VOLL)."""
    fr = _online(sep_chain, PARENT, "v_online_linear")
    for t in STEPS:
        assert _at(fr, REALIZED, t) == pytest.approx(0.0, abs=1e-9)


@pytest.mark.solver
def test_parent_objective(sep_chain):
    """Hand-calc parent objective (operational only; cyclic startup 0)."""
    _, wf = sep_chain
    assert _obj(wf, PARENT) == pytest.approx(PARENT_OBJECTIVE, rel=1e-6)


# ---------------------------------------------------------------------------
# separate_dispatch SIBLING — the v_online PIN (the #1 payoff)
# ---------------------------------------------------------------------------

@pytest.mark.solver
def test_sibling_v_online_pinned_to_parent_commitment(sep_chain):
    """The sibling's v_online is PINNED to the parent's committed schedule
    over the jump (the realized period ``p2035``): 2.0 every step — NOT the
    1.0 it would re-decide freely for the realized 50 MW residual."""
    parent = _online(sep_chain, PARENT, "v_online_linear")
    sib = _online(sep_chain, SIBLING, "v_online_linear")
    for t in STEPS:
        committed = _at(parent, HIGH, t)        # the tied parent commitment
        assert _at(sib, REALIZED, t) == pytest.approx(committed, abs=1e-9)
        assert _at(sib, REALIZED, t) == pytest.approx(TIED_COMMIT, abs=1e-9)


@pytest.mark.solver
def test_sibling_feasible_no_voll_and_objective(sep_chain):
    """The pin is feasible — the sibling stays OPTIMAL with ZERO balance
    slack (no VOLL) — and lands the realized-dispatch objective."""
    steps, wf = sep_chain
    sol = steps[SIBLING].solution
    for vq in ("vq_state_up", "vq_state_down"):
        if vq in sol._vars:
            nz = sol.value(vq).filter(pl.col("value") > 1e-6)
            assert nz.height == 0, f"sibling {vq} has {nz.height} nonzero rows"
    assert _obj(wf, SIBLING) == pytest.approx(SIBLING_OBJECTIVE, rel=1e-6)


# ---------------------------------------------------------------------------
# plain stochastic (no separate_dispatch) — pins the EXISTING whole-window tie
# ---------------------------------------------------------------------------

@pytest.mark.solver
def test_plain_stochastic_ties_online_over_realized_window(plain_chain):
    """Without separate_dispatch the solve's ``dt_non_anticipativity`` is its
    whole realized window, so online is already tied over it by the existing
    machinery (``dtna_online`` == ``dtna``) — the jump path leaves this
    UNCHANGED.  The parent's v_online is equal across forecasts every step."""
    fr = _online(plain_chain, PARENT, "v_online_linear")
    for t in STEPS:
        assert _at(fr, HIGH, t) == pytest.approx(_at(fr, LOW, t), abs=1e-9)
        assert _at(fr, HIGH, t) == pytest.approx(TIED_COMMIT, abs=1e-9)


# ---------------------------------------------------------------------------
# integer online (binary startup) — same dtna_online path, v_online_integer
# ---------------------------------------------------------------------------

@pytest.mark.solver
def test_integer_v_online_tied_over_jump(integer_chain):
    """The jump tie fires for the INTEGER online family too
    (non_anticipativity_online_integer over dtna_online): ``high`` (wind now
    fully covers demand, so its own optimum is 0 units) is forced UP to low's
    committed unit over the whole jump."""
    fr = _online(integer_chain, PARENT, "v_online_integer")
    for t in STEPS:
        assert _at(fr, HIGH, t) == pytest.approx(_at(fr, LOW, t), abs=1e-9)
        assert _at(fr, HIGH, t) == pytest.approx(1.0, abs=1e-9)
