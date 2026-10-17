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
(``flextool/engine_polars/_derived_params.py``) reads a ``roll_N`` token as an
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

from flextool.engine_polars._orchestration import run_chain_from_db
from flextool.engine_polars._solve_state import FlexToolConfigError
from flextool.engine_polars._stochastic_detect import (
    is_genuinely_stochastic,
    synthetic_branch_tokens,
)

pytestmark = [pytest.mark.solver]

SCENARIO = "1_week_rolling_wind"
FIXTURES_DIR = Path(__file__).resolve().parents[1] / "fixtures"

# 168-step horizon, rolling_solve_jump = 24 → 7 rolls (see
# tests/engine_polars/test_rolling_stochastic_na.py docstring).
EXPECTED_ROLLS = 7


def _pack(value, value_type):
    """Spine parameter-value encoding: ``[base64(json), value_type]``."""
    return [
        base64.b64encode(json.dumps(value).encode()).decode("ascii"),
        value_type,
    ]


def _build_invest_spec() -> dict:
    """Extend the rolling-stochastic fixture with investment intent.

    Mirrors the exploration setup: gas_plant becomes invest-eligible and the
    ``1week_rolling`` solve authors ``stochastic_invest_method=recourse`` +
    ``invest_periods`` + ``realized_invest_periods`` over ``period1``.  The
    invest params land on a NEW ``invest`` alternative wired into the end of
    the ``1_week_rolling_wind`` scenario chain, so the original fixture
    alternatives are not mutated.
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
        ["unit", "gas_plant", "lifetime", _pack(1.0, "float"), "invest"],
        ["unit", "gas_plant", "invest_max_total", _pack(5000.0, "float"),
         "invest"],
        ["unit", "gas_plant", "virtual_unitsize", _pack(100.0, "float"),
         "invest"],
        # Solve-level investment intent (recourse over the single period).
        ["solve", "1week_rolling", "stochastic_invest_method",
         _pack("recourse", "str"), "invest"],
        ["solve", "1week_rolling", "invest_periods",
         _pack({"value_type": "str", "data": ["period1"]}, "array"),
         "invest"],
        ["solve", "1week_rolling", "realized_invest_periods",
         _pack({"value_type": "str", "data": ["period1"]}, "array"),
         "invest"],
    ])
    return src


@pytest.fixture(scope="module")
def rolling_invest_db_url(tmp_path_factory) -> str:
    """Build the rolling+stochastic+invest DB from the injected JSON spec."""
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from db_utils import json_to_db
    from flextool.update_flextool.db_migration import migrate_database

    spec = _build_invest_spec()
    work = tmp_path_factory.mktemp("rolling_stoch_invest")
    jpath = work / "rolling_stoch_invest.json"
    jpath.write_text(json.dumps(spec, indent=2) + "\n")
    dbpath = work / "rolling_stoch_invest.sqlite"
    url = json_to_db(jpath, dbpath)
    migrate_database(url)
    return url


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


@pytest.fixture(scope="module")
def rolling_invest_result(
    rolling_invest_db_url, test_solver_config_dir, tmp_path_factory
):
    """Run the rolling+stochastic+invest scenario ONCE for the module.

    Runs under ``FLEXTOOL_AUTOSCALE_STRICT=1`` (invariant #1) so an autoscale
    gap is loud rather than silently reverting to an un-scaled LP.  Captures
    WARNING-level records across the whole run so the Task-1 warning can be
    asserted.
    """
    work_folder = tmp_path_factory.mktemp("rolling_stoch_invest_run")
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
            input_db_url=rolling_invest_db_url,
            scenario_name=SCENARIO,
            work_folder=work_folder,
            solver_config_dir=test_solver_config_dir,
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


def _steps(result):
    if result["error"] is not None:
        raise result["error"]
    return result["steps"]


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

    Each roll's ``flex_data`` is genuinely stochastic and carries non-empty
    synthetic branch tokens, yet the whole run raised no ``FlexToolConfigError``
    — the W3 activation neither misfires nor crashes under investment intent.
    """
    # No FlexToolConfigError was raised by the run.
    assert rolling_invest_result["error"] is None, rolling_invest_result["error"]
    steps = _steps(rolling_invest_result)
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
