"""Invest-chain regression fixture — Pre-work 0b (R1 + R2).

Context
-------
The Phase E-h post-mortem (``specs/phase_e_h_post_mortem.md``) traced the
Rivendell ``B0_base_hourly_rp`` regression to a fixture-coverage gap:
**no fixture combined invest-chain + non-trivial
``p_entity_all_existing`` + end-to-end LP solve + value assertion** under
the (then-new) ``csv_emission_disabled`` gate.  19 commits stayed green
before a user surfaced the failure from the CLI.

The Step-2.5 / 2.6 FlexDataProvider migration consumed the
``csv_emission_disabled`` distinction (every cascade run now goes
through the in-memory Provider; there is no "gated" vs "default" mode
to switch).  But the underlying coverage requirement — assert
**invest-chain frame state** end-to-end at high precision against a
golden, not just objective parity at ``rel_tol=1e-6`` — still applies.

This module provides:

* **R1** — A native-cascade end-to-end run of
  ``5weeks_invest_fullYear_dispatch_coal_wind`` (the invest-chain
  fixture that the post-mortem flagged as canonical) asserted against
  ``golden_obj.json`` at ``rel_tol=1e-9``.  The fixture's two
  sub-solves (``invest_1year_5weeks`` then ``y2020_fullYear_dispatch``)
  exercise the lifetime / invest cumulative path that's
  structurally absent from ``work_base`` and ``work_fullYear_roll``.

* **R2** — Frame-level equality assertion on
  ``provider.get("p_entity_all_existing")`` against a checked-in
  expected snapshot.  This catches the kind of single-row shift that
  feasibly produces a different LP but masks behind objective parity
  at ``rel_tol=1e-6``.  Per the post-mortem, the original Rivendell
  bug shifted a single ``(entity, period)`` row 500× and still gave a
  feasible LP for most other scenarios.

The companion ``tests/test_scenarios.py::test_scenario`` already runs
golden-CSV parity at ``rel_tol=1e-4``; this test tightens the gate at
the in-memory frame layer, which is where the original regression
lived.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import polars as pl
import pytest

_TESTS_DIR = Path(__file__).resolve().parent.parent
if str(_TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(_TESTS_DIR))

from flextool.engine_polars import run_chain_from_db  # noqa: E402


SCENARIO = "5weeks_invest_fullYear_dispatch_coal_wind"

# Sub-solves the fixture's solve chain produces, in cascade order.
SUB_SOLVES = ("invest_1year_5weeks", "y2020_fullYear_dispatch")

# The post-mortem's R1 target: ``rel_tol=1e-9`` against v3.32.0 golden.
# The fixture's ``golden_obj.json`` carries the captured objective; we
# tighten from the file's recorded ``rel_tolerance=1e-6`` to ``1e-9``
# because the native cascade matches the golden to ~1e-15 today (see
# the file docstring's ``probe`` block in commit history).
OBJ_REL_TOL = 1e-9

_FIXTURE_DIR = (
    _TESTS_DIR
    / "engine_polars"
    / "data"
    / "work_5weeks_invest_fullYear_dispatch_coal_wind"
)

# R2 expected frame.  Hand-derived from the fixture spec:
#   - ``coal_market`` / ``west`` are markets / nodes — ``existing=0``.
#   - ``coal_plant`` ships with ``existing=500`` MW at ``p2020``.
#   - ``wind_plant`` ships with ``existing=1000`` MW at ``p2020``.
#
# Both sub-solves see the same frame because the invest chain in this
# fixture is single-period (invest in y2020, no chained carry-over).
# A future ``work_invest_chain_minimal`` fixture (post-mortem fallback
# option) would extend this to 2 periods with non-trivial ``p2030``
# carry-over.
_EXPECTED_PAE_ROWS = [
    {"entity": "coal_market", "period": "p2020", "value": "0.0"},
    {"entity": "west",        "period": "p2020", "value": "0.0"},
    {"entity": "coal_plant",  "period": "p2020", "value": "500.0"},
    {"entity": "wind_plant",  "period": "p2020", "value": "1000.0"},
]


pytestmark = pytest.mark.solver


@pytest.fixture(scope="module")
def invest_chain_steps(test_db_url, tmp_path_factory):  # noqa: ANN001
    """Run the invest-chain cascade ONCE per module and yield the steps.

    Both R1 and R2 hit the same cascade; sharing the run keeps this
    test file's per-CI cost at one ``run_chain_from_db`` invocation.
    Uses ``keep_solutions=True`` so each step retains its Provider for
    frame-equality checks (R2).
    """
    work = tmp_path_factory.mktemp("invest_chain_regression_work")
    steps = run_chain_from_db(
        test_db_url,
        SCENARIO,
        work_folder=work,
        keep_solutions=True,
    )
    assert steps, f"run_chain_from_db returned no steps for {SCENARIO!r}"
    return steps


def _golden_obj() -> float:
    with (_FIXTURE_DIR / "golden_obj.json").open() as fh:
        return float(json.load(fh)["obj"])


def test_invest_chain_objective_matches_golden(invest_chain_steps):
    """R1 — last-sub-solve objective matches v3.32.0 golden at 1e-9.

    The cascade's terminal sub-solve produces the full-horizon dispatch
    objective; the invest sub-solve's objective is a different number
    (the investment + first-week dispatch) and is covered by the
    per-sub-solve parquet test in
    ``test_flex_5weeks_invest_fullYear_dispatch_coal_wind_per_sub_solve.py``.
    """
    last = next(reversed(invest_chain_steps.values()))
    assert last.solution is not None and last.solution.optimal, (
        f"Last sub-solve of {SCENARIO!r} did not produce an optimal LP"
    )
    golden = _golden_obj()
    # ``step.obj`` is the un-scaled (model-cost) objective populated by
    # ``OrchestrationStep`` — ``step.solution.obj`` is the raw LP
    # solver value, which may be scaled by the ``scale_the_objective``
    # heuristic and is several orders of magnitude smaller.  Match
    # against the un-scaled value, which is what the golden was
    # captured from.
    assert last.obj is not None, (
        f"Last sub-solve of {SCENARIO!r} has no unscaled objective"
    )
    obj = float(last.obj)
    rel = abs(obj - golden) / max(1.0, abs(golden))
    assert rel < OBJ_REL_TOL, (
        f"objective parity broken on {SCENARIO!r}: "
        f"native={obj!r} golden={golden!r} rel={rel!r} tol={OBJ_REL_TOL}"
    )


@pytest.mark.parametrize("sub_solve", SUB_SOLVES)
def test_invest_chain_pae_frame_equality(invest_chain_steps, sub_solve):
    """R2 — Provider's ``p_entity_all_existing`` matches expected exactly.

    Frame-level equality (not just objective parity) catches single-row
    shifts in the invest chain that the Phase E-h post-mortem
    identified as the most insidious regression class — they still
    produce a feasible LP and pass loose objective gates.
    """
    step = invest_chain_steps[sub_solve]
    provider = step.flex_data_provider
    assert provider is not None, (
        f"sub-solve {sub_solve!r}: flex_data_provider unexpectedly None "
        f"(keep_solutions=True should retain it)"
    )
    frame = provider.get("p_entity_all_existing")
    assert frame is not None, (
        f"sub-solve {sub_solve!r}: provider has no "
        f"'p_entity_all_existing' key — invest-chain cascade did not "
        f"populate it"
    )

    expected = pl.DataFrame(
        _EXPECTED_PAE_ROWS,
        schema={"entity": pl.Utf8, "period": pl.Utf8, "value": pl.Utf8},
    )

    # Sort both sides on (entity, period) to make the comparison
    # insertion-order-independent — the Provider's writer ordering is
    # an internal detail; what matters is the *set* of (entity,
    # period, value) tuples.
    actual_sorted = frame.sort(["entity", "period"])
    expected_sorted = expected.sort(["entity", "period"])

    assert actual_sorted.equals(expected_sorted), (
        f"sub-solve {sub_solve!r}: p_entity_all_existing frame drift\n"
        f"--- actual ---\n{actual_sorted}\n"
        f"--- expected ---\n{expected_sorted}"
    )
