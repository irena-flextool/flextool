"""Regression lock: rolling + stochastic + investment is a SAFE no-build.

This is the "#1" guard for the known, INTENDED behaviour that a flat
``rolling_window`` solve authored with investment intent does NOT build any
``v_invest`` variable — and does so safely (every roll solves, the W3
non-anticipativity guards stay inert, nothing raises).

Why this is intended, not a bug
-------------------------------
Investment in FlexTool is period-granular: ``v_invest`` is indexed ``(p, d)``
/ ``(n, d)`` with no timestep axis.  A ``rolling_window`` solve expands into
synthetic per-roll sub-solves named ``<base>_roll_<N>`` whose roll starts
advance PAST the beginning of each period.  ``_resolve_synthetic_solve``
(``flextool/engine/preprocess/params/_derived/_periods.py``) reads a ``roll_N`` token as an
invest-period anchor, so ``apply_synthetic_invest_sets`` leaves every
invest-set ``None`` — no ``v_invest`` is created and the authored investment
intent is silently dropped.

The DESIGN is to run a separate INVESTMENT solve BEFORE the rolling
(operational) solve and hand its realized capacity forward; rolling is for
operational modelling only.  This test therefore pins:

* the solve is SAFE — all rolls reach a strict optimum and the run raises no
  ``FlexToolConfigError``;
* the W3 per-roll activation of the ``period_branch_full`` consumers is
  CORRECT and inert here — ``is_genuinely_stochastic`` is ``True`` per roll and
  ``synthetic_branch_tokens`` is non-empty, yet nothing misfires;
* the KNOWN silent no-build — ``pd_invest_set is None`` on every roll.  A
  future change that starts building ``v_invest`` in a flat rolling solve will
  trip assertion (3) and force a conscious update of this test (and of the
  Task-1 warning below);
* the Task-1 warning — ``_recursive_solve`` emits exactly one WARNING per
  parent rolling solve explaining the no-build and pointing at a pre-rolling
  invest solve.

The no-build is pinned on the ``separate_dispatch`` method: its solve (1) of
each roll fans out at the roll's first step (the forecasts are made at every
roll start), so no roll holds a period label with time steps, and the
realization solve drops ``invest_periods``.  Under ``within_solve`` each roll is
a realization roll whose jump ends inside ``period1``: the label ``period1``
then dispatches the jump.  A roll builds no investment and therefore no capacity
margin either, so such a solve runs as any rolling solve with ``invest_periods``
(:func:`test_within_solve_with_invest_periods_runs_without_margin`).

The fixture is built by IN-TEST injection onto a tmp copy of
``tests/fixtures/stochastics.json`` (invariant #3: DB built from JSON/schema,
never a checked-in sqlite) so the committed fixture — and the existing
``1_week_rolling_wind`` tests that read it — stay untouched.
"""
from __future__ import annotations

import base64
import json
import logging
import os
from pathlib import Path

import pytest

from flextool.engine.solve._orchestration import run_chain_from_db
from flextool.engine.input._solve_state import FlexToolConfigError
from flextool.engine.preprocess._stochastic_detect import (
    is_genuinely_stochastic,
    synthetic_branch_tokens,
)
from tests.engine._unit_db import without_realization

pytestmark = [pytest.mark.solver]

SCENARIO = "1_week_rolling_wind"
FIXTURES_DIR = Path(__file__).resolve().parents[1] / "fixtures"

# 168-step horizon, rolling_solve_jump = 24 → 7 rolls (see
# tests/engine/test_rolling_stochastic_na.py docstring).
EXPECTED_ROLLS = 7


def _pack(value, value_type):
    """Spine parameter-value encoding: ``[base64(json), value_type]``."""
    return [
        base64.b64encode(json.dumps(value).encode()).decode("ascii"),
        value_type,
    ]


def _build_invest_spec(method: str | None = None, realization: str = "separate_dispatch",
                       margin: bool = False) -> dict:
    """Extend the rolling-stochastic fixture with investment intent.

    Mirrors the exploration setup: gas_plant becomes invest-eligible and the
    ``1week_rolling`` solve authors ``invest_periods`` + ``realized_invest_periods``
    over ``period1`` (and ``invest_solve_branching_method=<method>`` when *method*
    is given: refused on a rolling solve, see
    :func:`test_invest_branching_on_a_rolling_solve_is_refused`).  The
    invest params land on a NEW ``invest`` alternative wired into the end of
    the ``1_week_rolling_wind`` scenario chain, so the original fixture
    alternatives are not mutated.
    *realization* is the solve's ``dispatch_realization_method``; *margin* adds
    the group ``cm_g`` on ``demand_node`` with an absolute capacity margin.
    """
    src = json.loads((FIXTURES_DIR / "stochastics.json").read_text())

    src["alternatives"].append(["invest", "recourse invest on gas_plant"])

    # Rewire the scenario chain: ... -> 1week_rolling -> invest -> None.
    new_sa = []
    for sa in src["scenario_alternatives"]:
        if sa[0] == SCENARIO and sa[1] == "1week_rolling":
            new_sa.append([SCENARIO, "1week_rolling", "invest"])
        else:
            new_sa.append(sa)
    new_sa.append([SCENARIO, "invest", None])
    src["scenario_alternatives"] = new_sa

    src["parameter_values"].extend([
        # gas_plant becomes invest-eligible.
        ["unit", "gas_plant", "invest_method", _pack("invest_total", "str"),
         "invest"],
        ["unit", "gas_plant", "invest_cost", _pack(50.0, "float"), "invest"],
        ["unit", "gas_plant", "discount_rate", _pack(0.05, "float"), "invest"],
        ["unit", "gas_plant", "lifetime_technical", _pack(1.0, "float"), "invest"],
        ["unit", "gas_plant", "invest_max_total", _pack(5000.0, "float"),
         "invest"],
        ["unit", "gas_plant", "virtual_unitsize", _pack(100.0, "float"),
         "invest"],
        # Its 1000 MW as 10 units of 100 MW.
        ["unit", "gas_plant", "existing", _pack(10.0, "float"), "invest"],
        # Solve-level investment intent over the single period.
        ["solve", "1week_rolling", "invest_periods",
         _pack({"value_type": "str", "data": ["period1"]}, "array"),
         "invest"],
        ["solve", "1week_rolling", "realized_invest_periods",
         _pack({"value_type": "str", "data": ["period1"]}, "array"),
         "invest"],
        ["solve", "1week_rolling", "dispatch_realization_method",
         _pack(realization, "str"), "invest"],
    ])
    if method is not None:
        src["parameter_values"].append(["solve", "1week_rolling", "invest_solve_branching_method",
                                        _pack(method, "str"), "invest"])
    if margin:
        src["entities"] += [["group", "cm_g", None], ["group__node", ["cm_g", "demand_node"], None]]
        src["parameter_values"] += [
            ["group", "cm_g", "capacity_margin_method", _pack("absolute", "str"), "invest"],
            ["group", "cm_g", "capacity_margin_absolute", _pack(1.0e5, "float"), "invest"],
            ["group", "cm_g", "penalty_capacity_margin", _pack(1.0, "float"), "invest"]]
    return src


def _db_url(spec: dict, work: Path) -> str:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from db_utils import json_to_db
    from flextool.update_flextool.db_migration import migrate_database

    jpath = work / "rolling_stoch_invest.json"
    jpath.write_text(json.dumps(spec, indent=2) + "\n", encoding="utf-8")
    url = json_to_db(jpath, work / "rolling_stoch_invest.sqlite")
    migrate_database(url)
    return url


@pytest.fixture(scope="module")
def rolling_invest_db_url(tmp_path_factory) -> str:
    """Build the rolling+stochastic+invest DB from the injected JSON spec."""
    return _db_url(_build_invest_spec(), tmp_path_factory.mktemp("rolling_stoch_invest"))


class _WarningCollector(logging.Handler):
    """Collect formatted WARNING+ messages across the module-scoped run.

    ``caplog`` is function-scoped and cannot back a module-scoped fixture, so
    the Task-1 warning (emitted during the single memoised run) is captured
    with a dedicated handler attached to the ``flextool`` logger.
    """

    def __init__(self) -> None:
        super().__init__(level=logging.WARNING)
        self.messages: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.messages.append(record.getMessage())


def _run_collecting(url: str, work_folder: Path, solver_config_dir) -> dict:
    """Run *url* under ``FLEXTOOL_AUTOSCALE_STRICT=1`` (invariant #1: an autoscale gap is loud rather than silently
    reverting to an un-scaled LP), collecting the WARNING-level records of the whole run so the Task-1 warning can
    be asserted.  ``{"steps", "error", "warnings"}``; a ``FlexToolConfigError`` is kept in ``error``."""
    prev = os.environ.get("FLEXTOOL_AUTOSCALE_STRICT")
    os.environ["FLEXTOOL_AUTOSCALE_STRICT"] = "1"
    collector = _WarningCollector()
    flextool_logger = logging.getLogger("flextool")
    prev_level = flextool_logger.level
    if prev_level > logging.WARNING or prev_level == logging.NOTSET:
        flextool_logger.setLevel(logging.WARNING)
    flextool_logger.addHandler(collector)
    error = None
    steps = None
    try:
        steps = run_chain_from_db(
            input_db_url=url,
            scenario_name=SCENARIO,
            work_folder=work_folder,
            solver_config_dir=solver_config_dir,
            warm=False,
            keep_solutions=True,
        )
    except FlexToolConfigError as exc:
        error = exc
    finally:
        flextool_logger.removeHandler(collector)
        flextool_logger.setLevel(prev_level)
        if prev is None:
            os.environ.pop("FLEXTOOL_AUTOSCALE_STRICT", None)
        else:
            os.environ["FLEXTOOL_AUTOSCALE_STRICT"] = prev
    return {"steps": steps, "error": error, "warnings": collector.messages}


@pytest.fixture(scope="module")
def rolling_invest_result(
    rolling_invest_db_url, test_solver_config_dir, tmp_path_factory
):
    """Run the rolling+stochastic+invest scenario ONCE for the module (:func:`_run_collecting`)."""
    return _run_collecting(rolling_invest_db_url, tmp_path_factory.mktemp("rolling_stoch_invest_run"),
                           test_solver_config_dir)


def _steps(result):
    """The stochastic rolls (the realization rolls the designation adds are left out)."""
    if result["error"] is not None:
        raise result["error"]
    return without_realization(result["steps"])


@pytest.mark.parametrize("method", ["investments_after_reveal_period", "investments_at_reveal_period"])
def test_invest_branching_on_a_rolling_solve_is_refused(tmp_path_factory, test_solver_config_dir, method):
    """A rolling solve is a dispatch solve whatever its ``invest_periods``: ``invest_solve_branching_method`` applies
    to investment solves only, so the run stops before any roll and points to ``dispatch_realization_method``."""
    work = tmp_path_factory.mktemp("rolling_stoch_branching")
    url = _db_url(_build_invest_spec(method, realization="within_solve"), work)
    with pytest.raises(FlexToolConfigError) as err:
        run_chain_from_db(input_db_url=url, scenario_name=SCENARIO, work_folder=work / "w",
                          solver_config_dir=test_solver_config_dir, warm=False, keep_solutions=True)
    assert str(err.value) == (
        f"Solve '1week_rolling' sets invest_solve_branching_method {method}, which applies to investment solves "
        "only, but it is a rolling solve. Remove invest_solve_branching_method from '1week_rolling': a dispatch "
        "solve with stochastic_branches is set up with dispatch_realization_method.")


def _invest_warnings(result: dict) -> list[str]:
    return [w for w in result["warnings"]
            if "1week_rolling" in w and "investment is NOT built in a flat rolling_window solve" in w]


def test_within_solve_with_invest_periods_runs_without_margin(tmp_path_factory, test_solver_config_dir):
    """``within_solve``: roll 0 is t0001-t0096 with jump 24, so its jump ends at t0024 inside ``period1``
    (t0001-t0168), and the label ``period1`` dispatches t0001-t0024 while ``period1`` is in ``invest_periods``.  A
    roll builds no investment, so it builds no capacity margin: with the group ``cm_g`` (absolute margin 1e5 MW,
    far above any capacity) the run completes, no roll has a ``capacityMargin`` row or a margin slack, every roll
    equals the run without ``cm_g``, and the rolling-invest warning names the solve once."""
    from tests.engine._unit_db import build, cstr_over

    runs = {}
    for margin in (True, False):
        work = tmp_path_factory.mktemp(f"rolling_within_invest_{margin}")
        url = _db_url(_build_invest_spec(realization="within_solve", margin=margin), work)
        runs[margin] = _run_collecting(url, work / "w", test_solver_config_dir)
        assert runs[margin]["error"] is None, runs[margin]["error"]
    with_cm, without_cm = runs[True]["steps"], runs[False]["steps"]
    assert len(with_cm) == EXPECTED_ROLLS and list(with_cm) == list(without_cm)
    for name, s in with_cm.items():
        assert s.flex_data.groupCapacityMargin is not None and s.flex_data.groupCapacityMargin.height == 1, name
        assert s.flex_data.period_invest.height > 0, name
        over = cstr_over(build(s.flex_data), "capacityMargin")
        assert over is None or over.height == 0, name
        try:
            slack = s.solution.value("vq_capacity_margin")
        except KeyError:
            slack = None
        assert slack is None or slack.height == 0, name
        assert float(s.solution.obj) == pytest.approx(float(without_cm[name].solution.obj), rel=1e-9), name
    assert len(_invest_warnings(runs[True])) == 1


def test_rolling_invest_all_rolls_optimal(rolling_invest_result):
    """(1) The solve is SAFE: every roll reaches a strict optimum, no raise."""
    steps = _steps(rolling_invest_result)
    assert len(steps) == EXPECTED_ROLLS, (
        f"expected {EXPECTED_ROLLS} rolls, got {len(steps)}"
    )
    assert all(s.optimal for s in steps.values()), {
        name: s.optimal for name, s in steps.items()
    }


def test_rolling_invest_w3_guards_inert(rolling_invest_result):
    """(2) W3 per-roll stochastic guards are CORRECT and inert.

    Solve (1) of each of the 7 rolls fans out at the roll's first step (forecasts are made at every roll start,
    t0001, t0025, ..., t0145), so each has a genuinely stochastic ``flex_data`` with non-empty synthetic branch
    tokens, the last roll (t0145-t0168, its jump only) too, yet the whole run raised no ``FlexToolConfigError``:
    the W3 activation neither misfires nor crashes under investment intent.
    """
    # No FlexToolConfigError was raised by the run.
    assert rolling_invest_result["error"] is None, rolling_invest_result["error"]
    steps = _steps(rolling_invest_result)
    assert len(steps) == EXPECTED_ROLLS
    for name, s in steps.items():
        fd = s.flex_data
        assert fd is not None, f"roll {name!r} retained no flex_data"
        assert is_genuinely_stochastic(fd) is True, (
            f"roll {name!r}: is_genuinely_stochastic should be True"
        )
        tokens = synthetic_branch_tokens(fd.period_branch_full)
        assert tokens, (
            f"roll {name!r}: synthetic_branch_tokens must be non-empty"
        )


def test_rolling_invest_no_v_invest_built(rolling_invest_result):
    """(3) KNOWN silent no-build: ``pd_invest_set is None`` on every roll.

    Documents that a flat rolling solve builds NO investment.  If a future
    change starts building ``v_invest`` here this assertion trips, forcing a
    conscious review of this test and the Task-1 warning.
    """
    steps = _steps(rolling_invest_result)
    for name, s in steps.items():
        fd = s.flex_data
        assert fd is not None, f"roll {name!r} retained no flex_data"
        assert fd.pd_invest_set is None, (
            f"roll {name!r}: pd_invest_set is unexpectedly built "
            f"({fd.pd_invest_set!r}) — a flat rolling solve is not supposed "
            f"to build v_invest; review this test and the Task-1 warning"
        )


def test_rolling_invest_warns_once(rolling_invest_result):
    """(4) The Task-1 warning fires exactly once, naming the solve.

    Emitted once per PARENT rolling solve (not per roll) — explains the
    no-build and points at a pre-rolling investment solve.
    """
    _steps(rolling_invest_result)  # surface any memoised error first
    hits = [
        w for w in rolling_invest_result["warnings"]
        if "1week_rolling" in w
        and "investment is NOT built in a flat rolling_window solve" in w
    ]
    assert len(hits) == 1, (
        f"expected exactly one investment-in-rolling warning naming "
        f"'1week_rolling', got {len(hits)}: {hits}"
    )
