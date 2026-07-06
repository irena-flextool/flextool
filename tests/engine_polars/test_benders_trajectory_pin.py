"""C5a — pin today's λ=0 Benders trajectory as the C5b refactor gate.

Runs the unmodified HEAD ``solve_benders`` loop on the
``lh2_three_region_trade_invest`` fixture (JSON-fixture-built DB,
``db_fixture="lh2_trade_invest"`` — CLAUDE.md invariant #3) with
``in_out_weight=0`` (λ=0 exact Benders), ``workers=1`` and
``scale_the_objective=1.0``, and compares the per-iteration
``progress_callback`` payloads against literals captured from this exact
loop.  This is the in-repo byte-parity gate for commit C5b (the rewire of
``_benders.py`` onto the ``polar_high.benders`` coordinator) — see
``specs/benders_master_hosted_nodes_plan.md`` §5 C5a and risk R7.

SCOPE CAVEAT (read before "fixing" a failure of this test): the literals
are a **same-machine / same-HiGHS-build refactor gate**.  LP vertex
trajectories are not portable across solver builds or hardware, so this
test must NOT be treated as a permanent cross-machine CI pin.  It exists
to hold C5b to byte-parity; after C5b lands it is acceptable to loosen
the tolerances or retire the test (orchestrator's call).

Tolerances: iteration count, convergence flags and region-cost key sets
are compared exactly; every numeric field with ``rtol=1e-12`` (atol=0).
The trajectory was captured 13 times in separate processes (random and
pinned ``PYTHONHASHSEED`` 0–9, clean and deliberately polluted env): the
iteration structure and bounds are stable, with only ULP-level float
wobble (worst observed relative deviation ~7e-15, hash-seed-dependent
dict-summation order), three orders of magnitude inside ``rtol=1e-12``.

Environment pinning: the transcript is env-sensitive (the loop
env-resolves the in-out weight, workers, region-duals mode, cut
compaction/window/policy, max-stall, and the region autoscale gating via
``FLEXTOOL_SCALING``), so every ``FLEXTOOL_BENDERS_*`` knob plus
``FLEXTOOL_SCALING`` is deleted for the whole module — the recorded and
replayed trajectories cannot be skewed by the invoking shell.
"""
from __future__ import annotations

import math
import os

import pytest

from flextool.engine_polars import load_flextool
from flextool.engine_polars._benders import solve_benders

_REGIONS = ["region_A", "region_B", "region_C"]

# Every env knob the Benders loop (or its region autoscale) resolves.
# Grep anchor: ``FLEXTOOL_BENDERS_`` in flextool/engine_polars/_benders.py.
_PINNED_ENV = [
    "FLEXTOOL_BENDERS_IN_OUT_WEIGHT",
    "FLEXTOOL_BENDERS_WORKERS",
    "FLEXTOOL_BENDERS_REGION_DUALS",
    "FLEXTOOL_BENDERS_MAX_STALL",
    "FLEXTOOL_BENDERS_CUT_COMPACT_AT",
    "FLEXTOOL_BENDERS_CUT_WINDOW",
    "FLEXTOOL_BENDERS_CUT_POLICY",
    "FLEXTOOL_BENDERS_VERBOSE",
    "FLEXTOOL_SCALING",
]

# --- The frozen reference trajectory (captured at HEAD, pre-C5b). ----------
# Per-iteration ``progress_callback`` payloads: iter and converged are exact;
# lower_bound / upper_bound / best_upper_bound / gap / region_costs values
# are rtol=1e-12; region_costs key sets are exact.
_EXPECTED_TRAJECTORY = [
    {
        "iter": 1,
        "lower_bound": -14949901007.79792,
        "upper_bound": 453590520713.20526,
        "best_upper_bound": 453590520713.20526,
        "gap": 1.0329590243294577,
        "converged": False,
        "region_costs": {
            "region_A": 228220318274.38412,
            "region_B": 7896339713.459629,
            "region_C": 217466767479.6316,
        },
    },
    {
        "iter": 2,
        "lower_bound": 8509274940.821064,
        "upper_bound": 8573360797.927618,
        "best_upper_bound": 8573360797.927618,
        "gap": 0.007474998266962599,
        "converged": False,
        "region_costs": {
            "region_A": 2941915941.4270535,
            "region_B": 1908512458.796694,
            "region_C": 3722767479.6315784,
        },
    },
    {
        "iter": 3,
        "lower_bound": 8533293529.396137,
        "upper_bound": 8549489173.323434,
        "best_upper_bound": 8549489173.323434,
        "gap": 0.0018943405388278799,
        "converged": False,
        "region_costs": {
            "region_A": 2882596183.676689,
            "region_B": 1943969394.0643466,
            "region_C": 3722767479.6315784,
        },
    },
    {
        "iter": 4,
        "lower_bound": 8544247283.473894,
        "upper_bound": 8544247283.473894,
        "best_upper_bound": 8544247283.473894,
        "gap": 0.0,
        "converged": True,
        "region_costs": {
            "region_A": 2882646182.3895144,
            "region_B": 1925088261.0731094,
            "region_C": 3736358027.465967,
        },
    },
]

# Final ``BendersResult`` fields (real units, s=1.0).
_EXPECTED_ITERATIONS = 4
_EXPECTED_LOWER_BOUND = 8544247283.473894
_EXPECTED_UPPER_BOUND = 8544247283.473894
_EXPECTED_TOTAL_OBJECTIVE = 8544247283.473894
_EXPECTED_GAP = 0.0

_RTOL = 1e-12


@pytest.fixture(scope="module", autouse=True)
def _pinned_environment():
    """Delete every Benders/scaling env knob for the whole module.

    ``monkeypatch`` is function-scoped, so use a manual
    :class:`pytest.MonkeyPatch` — the fixture-build (cascade) AND the
    pinned solve both run under the cleaned environment, whatever the
    invoking shell exports.
    """
    mp = pytest.MonkeyPatch()
    for var in _PINNED_ENV:
        mp.delenv(var, raising=False)
    yield
    mp.undo()


@pytest.fixture(scope="module")
def ti_data(scenario_workdir):
    work = scenario_workdir(
        "lh2_three_region_trade_invest", db_fixture="lh2_trade_invest"
    )
    return load_flextool(work)


def _assert_close(actual: float, expected: float, what: str) -> None:
    assert math.isclose(actual, expected, rel_tol=_RTOL, abs_tol=0.0), (
        f"{what}: {actual!r} != pinned {expected!r} (rtol={_RTOL})"
    )


def test_lambda0_trajectory_matches_pinned_literals(ti_data) -> None:
    # Belt-and-braces: the module fixture must have scrubbed the knobs.
    for var in _PINNED_ENV:
        assert var not in os.environ, f"env pin failed for {var}"

    payloads: list[dict] = []
    res = solve_benders(
        ti_data,
        _REGIONS,
        max_iters=20,
        tol=1e-4,
        in_out_weight=0.0,
        workers=1,
        progress_callback=payloads.append,
    )

    # Iteration count / structure: exact.
    assert len(payloads) == len(_EXPECTED_TRAJECTORY), (
        f"iteration count drifted: {len(payloads)} payloads != "
        f"{len(_EXPECTED_TRAJECTORY)} pinned"
    )
    for got, exp in zip(payloads, _EXPECTED_TRAJECTORY):
        it = exp["iter"]
        assert got["iter"] == it
        assert got["converged"] == exp["converged"], f"iter {it}: converged flag"
        assert set(got["region_costs"]) == set(exp["region_costs"]), (
            f"iter {it}: region_costs key set"
        )
        for key in ("lower_bound", "upper_bound", "best_upper_bound", "gap"):
            _assert_close(got[key], exp[key], f"iter {it}: {key}")
        for region, cost in exp["region_costs"].items():
            _assert_close(
                got["region_costs"][region], cost,
                f"iter {it}: region_costs[{region}]",
            )

    # Final result: exact structure, rtol=1e-12 numerics.
    assert res.converged is True
    assert res.iterations == _EXPECTED_ITERATIONS
    _assert_close(res.lower_bound, _EXPECTED_LOWER_BOUND, "final lower_bound")
    _assert_close(res.upper_bound, _EXPECTED_UPPER_BOUND, "final upper_bound")
    _assert_close(
        res.total_objective, _EXPECTED_TOTAL_OBJECTIVE, "final total_objective"
    )
    assert math.isclose(res.gap, _EXPECTED_GAP, rel_tol=_RTOL, abs_tol=0.0), (
        f"final gap {res.gap!r} != pinned {_EXPECTED_GAP!r}"
    )
