"""Tests for the Lagrangian decomposition driver (gap A4).

Exercises ``flextool._lagrangian.solve_lagrangian`` on:

* the LH2 three-region fixture (parity vs monolithic);
* the coupling-column identification machinery;
* convergence behavior (max-iters bound, trivial single-region case);
* error reporting (non-decomposed scenario).
"""
from __future__ import annotations

import inspect
import threading

import pytest

from polar_high import Problem, WarmProblem

import flextool.engine_polars._lagrangian as _lagr_mod
from flextool.engine_polars import build_flextool, load_flextool
from flextool.engine_polars._lagrangian import (
    solve_lagrangian,
    _identify_coupling_cols,
)
from flextool.engine_polars._orchestration import (
    _format_subsolve_line,
    _resolve_lagrangian_workers,
)
from flextool.engine_polars._region_filter import split as region_split


# ---------------------------------------------------------------------------
# Coupling column identification
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def lh2_workdir(scenario_workdir):
    return scenario_workdir("lh2_three_region", db_fixture="lh2")


@pytest.fixture(scope="module")
def lh2_data(lh2_workdir):
    return load_flextool(lh2_workdir)


@pytest.fixture(scope="module")
def lh2_warmproblems(lh2_data):
    splits = region_split(lh2_data, regions=["region_A", "region_B", "region_C"])
    warms = []
    for s in splits:
        pb = Problem()
        build_flextool(pb, s.data)
        wp = WarmProblem(pb)
        wp.solve()
        warms.append(wp)
    return splits, warms


class TestCouplingIdentification:
    def test_lh2_coupling_count(self, lh2_warmproblems) -> None:
        splits, warms = lh2_warmproblems
        couplings = _identify_coupling_cols(splits, warms)
        # pipe_AB has two directions × pipe_BC has two directions = 4
        # cross-region arcs = 4 couplings.
        assert len(couplings) == 4
        keys = {c.pipeline_key for c in couplings}
        assert keys == {
            ("pipe_AB", "lh2_A", "lh2_B"),
            ("pipe_AB", "lh2_B", "lh2_A"),
            ("pipe_BC", "lh2_B", "lh2_C"),
            ("pipe_BC", "lh2_C", "lh2_B"),
        }

    def test_lh2_coupling_columns_sized_correctly(self, lh2_warmproblems) -> None:
        splits, warms = lh2_warmproblems
        couplings = _identify_coupling_cols(splits, warms)
        # Each (d, t) of the original arc has its own coupling cell;
        # the LH2 fixture has 168 timesteps in 1 period.
        for cpl in couplings:
            assert cpl.export_cols.size == 168
            assert cpl.import_cols.size == 168

    def test_lh2_export_import_regions_correct(self, lh2_warmproblems) -> None:
        splits, warms = lh2_warmproblems
        couplings = _identify_coupling_cols(splits, warms)
        by_key = {c.pipeline_key: c for c in couplings}
        # pipe_AB(A→B): A exports, B imports.
        assert by_key[("pipe_AB", "lh2_A", "lh2_B")].export_region == "region_A"
        assert by_key[("pipe_AB", "lh2_A", "lh2_B")].import_region == "region_B"
        # pipe_AB(B→A): B exports, A imports.
        assert by_key[("pipe_AB", "lh2_B", "lh2_A")].export_region == "region_B"
        assert by_key[("pipe_AB", "lh2_B", "lh2_A")].import_region == "region_A"
        # pipe_BC(C→B): C exports, B imports.
        assert by_key[("pipe_BC", "lh2_C", "lh2_B")].export_region == "region_C"
        assert by_key[("pipe_BC", "lh2_C", "lh2_B")].import_region == "region_B"


# ---------------------------------------------------------------------------
# LH2 parity test — flagship integration
# ---------------------------------------------------------------------------


# Documented Lagrangian gap on this fixture: subgradient (without
# bundle / ADMM extension) hits a ~0.1% gap floor due to bang-bang LP
# response on the cross-region pipeline flows.  flextool's reference
# test allows 2%.  We assert <0.5% — comfortably over our observed
# 0.108% but tight enough to catch regressions.
LH2_GAP_TOLERANCE = 0.005


@pytest.fixture(scope="module")
def lh2_monolithic_obj(lh2_data):
    pb = Problem()
    build_flextool(pb, lh2_data)
    sol = pb.solve()
    return sol.obj


def test_lh2_lagrangian_converges_to_monolithic(
    lh2_data, lh2_workdir, lh2_monolithic_obj
) -> None:
    """Running solve_lagrangian on LH2 gets within 0.5% of the
    monolithic optimum after 100 iters."""
    result = solve_lagrangian(
        lh2_data,
        work_dir=lh2_workdir,
        alpha=10.0,
        max_iters=100,
        tol=0.5,
        initial_lambda=0.0,
        min_iters=20,
    )
    assert result.iterations > 0
    # Per-region objs add up to total_objective.
    assert isinstance(result.total_objective, float)
    # Coupling pairs identified.
    assert set(result.final_lambdas.keys()) == {
        ("pipe_AB", "lh2_A", "lh2_B"),
        ("pipe_AB", "lh2_B", "lh2_A"),
        ("pipe_BC", "lh2_B", "lh2_C"),
        ("pipe_BC", "lh2_C", "lh2_B"),
    }
    rel_gap = abs(result.total_objective - lh2_monolithic_obj) / abs(lh2_monolithic_obj)
    assert rel_gap <= LH2_GAP_TOLERANCE, (
        f"LH2 Lagrangian gap {rel_gap*100:.4f}% exceeds tolerance "
        f"{LH2_GAP_TOLERANCE*100:.2f}%; "
        f"reported total={result.total_objective:.6e}, "
        f"monolithic={lh2_monolithic_obj:.6e}"
    )


def test_lh2_three_regions_solved(lh2_data, lh2_workdir) -> None:
    """All three regions should be solved and reported in
    region_objectives."""
    result = solve_lagrangian(
        lh2_data, work_dir=lh2_workdir, alpha=1.0, max_iters=20, tol=1.0,
    )
    assert set(result.region_objectives.keys()) == {
        "region_A", "region_B", "region_C"
    }
    for r, obj in result.region_objectives.items():
        assert obj > 0, f"{r} obj={obj} non-positive — region LP failed?"


def test_lh2_iteration_log_populated(lh2_data, lh2_workdir) -> None:
    result = solve_lagrangian(
        lh2_data, work_dir=lh2_workdir, alpha=1.0, max_iters=10, tol=1.0,
        min_iters=10,
    )
    # 10 iterations + 1 trailing "report kind" log entry
    assert len(result.iteration_log) >= 10
    # Each iter entry has the right shape
    for log in result.iteration_log[:10]:
        assert "iter" in log
        assert "max_abs_imbalance" in log
        assert "total_obj" in log
        assert "imbalances_max_cell" in log


# ---------------------------------------------------------------------------
# Convergence behavior tests
# ---------------------------------------------------------------------------


class TestConvergenceBehavior:
    def test_max_iters_returns_unconverged(self, lh2_data, lh2_workdir) -> None:
        """When max_iters is too small to converge, return with
        converged=False."""
        result = solve_lagrangian(
            lh2_data, work_dir=lh2_workdir, alpha=0.001, max_iters=3, tol=1e-9,
            min_iters=3,
        )
        # tolerance is so tight (1e-9) that 3 iters can't possibly
        # converge — we should NOT raise, we should return
        # converged=False.
        assert result.converged is False
        assert result.iterations == 3

    def test_min_iters_floor_enforced(self, lh2_data, lh2_workdir) -> None:
        """Setting min_iters > 1 prevents trivial early-termination at
        iteration 1 even when imbalance is already < tolerance."""
        result = solve_lagrangian(
            lh2_data, work_dir=lh2_workdir, alpha=1.0, max_iters=10,
            tol=1e10,  # huge tol → would converge on iter 1
            min_iters=5,
        )
        assert result.iterations >= 5


# ---------------------------------------------------------------------------
# Error / negative tests
# ---------------------------------------------------------------------------


class TestErrorPaths:
    def test_solve_lagrangian_requires_two_regions(self, lh2_data) -> None:
        with pytest.raises(ValueError, match=">=2 lagrangian_region groups"):
            solve_lagrangian(
                lh2_data,
                regions=["region_A"],
                decomposition_method={"region_A": "lagrangian_region"},
            )

    def test_solve_lagrangian_no_regions_errors(self, lh2_data) -> None:
        with pytest.raises(ValueError, match=">=2 lagrangian_region groups"):
            solve_lagrangian(
                lh2_data,
                regions=[],
                decomposition_method={},
            )

    def test_non_lagrangian_scenario_errors(self, scenario_workdir) -> None:
        """A scenario without decomposition_method should raise."""
        # work_base has no decomposition_method param.
        work_base = scenario_workdir("base")
        data = load_flextool(work_base)
        with pytest.raises(ValueError, match=">=2 lagrangian_region groups"):
            solve_lagrangian(data, work_dir=work_base)


# ---------------------------------------------------------------------------
# Primal recovery sanity test
# ---------------------------------------------------------------------------


def test_primal_recovery_total_recorded(lh2_data, lh2_workdir) -> None:
    """The recovery solve appends a 'best_dual_total' / 'recovered_total'
    log entry so callers can see both bounds."""
    result = solve_lagrangian(
        lh2_data, work_dir=lh2_workdir, alpha=1.0, max_iters=30, tol=1.0,
        min_iters=30,
    )
    # Last log entry is the report-kind summary.
    last = result.iteration_log[-1]
    assert "best_dual_total" in last
    assert "recovered_total" in last
    assert "report_kind" in last
    assert last["report_kind"] in ("best_dual", "recovered_primal")
    # The reported total equals one of the two bounds.
    assert (result.total_objective == last["best_dual_total"]
            or result.total_objective == last["recovered_total"])


# ---------------------------------------------------------------------------
# Synthetic 2-region LP smoke test
# ---------------------------------------------------------------------------


def test_lagrangian_smoke_lh2_subset(lh2_data) -> None:
    """Decompose only into 2 regions (A + B); pipe_BC's lh2_C terminal
    is "shared" so pipe_BC isn't a coupling.  Only pipe_AB's two
    directions are couplings (2 total)."""
    result = solve_lagrangian(
        lh2_data,
        regions=["region_A", "region_B"],
        decomposition_method={
            "region_A": "lagrangian_region",
            "region_B": "lagrangian_region",
        },
        alpha=1.0, max_iters=20, tol=1.0, min_iters=10,
    )
    # 2 couplings (pipe_AB × 2 directions).  pipe_BC stays whole in
    # whatever region's frame it lives in (B's, since lh2_B is in B).
    assert len(result.final_lambdas) == 2
    assert ("pipe_AB", "lh2_A", "lh2_B") in result.final_lambdas
    assert ("pipe_AB", "lh2_B", "lh2_A") in result.final_lambdas


# ---------------------------------------------------------------------------
# A. Parallel-worker resolution (pure helper)
# ---------------------------------------------------------------------------


class TestResolveLagrangianWorkers:
    @pytest.mark.parametrize(
        "setting, cpu, expected",
        [
            (0, 8, 7),     # auto => cpu-1
            (0, 1, 1),     # auto on single core floored at 1
            (4, 8, 4),     # explicit, under cap
            (100, 8, 8),   # explicit, capped at cpu
            (1, 1, 1),     # explicit single worker
            (-5, 8, 7),    # negative => auto (cpu-1)
        ],
    )
    def test_resolve(self, setting, cpu, expected) -> None:
        assert _resolve_lagrangian_workers(setting, cpu) == expected


# ---------------------------------------------------------------------------
# B. Cross-repo subsolve callback contract (the ONLY guard against
#    event/phase schema drift between FlexTool and polar-high).
# ---------------------------------------------------------------------------


class TestFormatSubsolveLine:
    REGIONS = ["north", "south"]

    def test_start_event(self) -> None:
        line = _format_subsolve_line(
            {"event": "start", "iter": 0, "subproblem": 0, "phase": "initial"},
            self.REGIONS, "[lagrangian s]", 100,
        )
        assert line is not None
        assert "solving north" in line

    def test_finish_event_optimal(self) -> None:
        line = _format_subsolve_line(
            {"event": "finish", "iter": 3, "subproblem": 1,
             "phase": "iterate", "obj": 1234.5},
            self.REGIONS, "[lagrangian s]", 100,
        )
        assert line is not None
        assert "done south" in line
        assert "obj=1234.5" in line

    def test_finish_event_non_optimal(self) -> None:
        # recovery phase, finish event, no obj -> "(non-optimal)".
        line = _format_subsolve_line(
            {"event": "finish", "iter": -1, "subproblem": 0, "phase": "recovery"},
            self.REGIONS, "[lagrangian s]", 100,
        )
        assert line is not None
        assert "done north" in line
        assert "(non-optimal)" in line

    def test_unknown_event_returns_none(self) -> None:
        assert _format_subsolve_line(
            {"event": "bogus", "iter": 0, "subproblem": 0, "phase": "iterate"},
            self.REGIONS, "[lagrangian s]", 100,
        ) is None
        # Missing event key entirely -> also None.
        assert _format_subsolve_line(
            {"iter": 0, "subproblem": 0},
            self.REGIONS, "[lagrangian s]", 100,
        ) is None


# ---------------------------------------------------------------------------
# C. Version-gated forwarding of workers->max_workers + subsolve_callback.
# ---------------------------------------------------------------------------


class _StubCoupling:
    """A coupling sentinel exposing the attributes ``solve_lagrangian``
    reads when assembling its result (``pipeline_key`` + ``lam_vec``)."""
    def __init__(self, key):
        self.pipeline_key = key
        self.lam_vec = None
        self.lam = 0.0


def _make_recording_result(n_regions: int, n_couplings: int):
    """A stand-in for polar-high's LagrangianResult exposing the fields
    ``solve_lagrangian`` reads after the solve."""
    import numpy as np

    class _RecordingResult:
        optimal = True
        obj = 0.0
        converged = True
        iterations = 1
        total_objective = 0.0
        subproblem_objectives = [0.0] * n_regions
        final_lambdas = [np.zeros(1) for _ in range(n_couplings)]
        iteration_log: list = []
        best_dual_total = 0.0
        recovered_total = 0.0
        subproblem_col_values: list = []
    return _RecordingResult()


def _install_stub_lagrangian(monkeypatch, *, accepts: bool, recorder: dict):
    """Monkeypatch LagrangianProblem with a stub whose ``.solve`` either
    DOES or DOES NOT accept ``max_workers`` / ``subsolve_callback``, and
    records the kwargs it received.  Also stubs the coupling machinery so
    the solve reaches the ``LagrangianProblem(...).solve(...)`` call with
    a non-trivial coupling set.  Returns the coupling list used so callers
    can size the result."""
    couplings = [_StubCoupling(("pipe_AB", "lh2_A", "lh2_B"))]

    if accepts:
        def _solve(self, *, max_iters, tol, step, initial_lambda,
                   min_iters, primal_tail, progress_callback=None,
                   max_workers=None, subsolve_callback=None):
            recorder["kwargs"] = {
                "max_workers": max_workers,
                "subsolve_callback": subsolve_callback,
            }
            return _make_recording_result(len(self._splits), len(couplings))
    else:
        def _solve(self, *, max_iters, tol, step, initial_lambda,
                   min_iters, primal_tail, progress_callback=None):
            recorder["kwargs"] = {}
            return _make_recording_result(len(self._splits), len(couplings))

    class _StubLagrangian:
        # Capture the number of subproblems so the stub result can size
        # its ``subproblem_objectives`` to match ``splits``.
        def __init__(self, subproblems, specs):
            self._splits = subproblems
        solve = _solve

    monkeypatch.setattr(_lagr_mod, "LagrangianProblem", _StubLagrangian)
    # Force a non-trivial coupling path so solve_lagrangian builds the
    # LagrangianProblem (rather than taking the trivial early-return).
    monkeypatch.setattr(
        _lagr_mod, "_identify_coupling_cols",
        lambda splits, warms: couplings,
    )
    monkeypatch.setattr(
        _lagr_mod, "_build_coupling_specs",
        lambda splits, warms, couplings: ["__spec__"],
    )
    # The post-solve invest assembly reads ``subproblem_col_values``; with
    # the empty list above it returns ``{}`` (TIER 1 disabled) — fine.


def test_forwarding_passes_when_signature_accepts(lh2_data, lh2_workdir, monkeypatch) -> None:
    rec: dict = {}
    _install_stub_lagrangian(monkeypatch, accepts=True, recorder=rec)
    sentinel_cb = lambda entry: None  # noqa: E731
    result = solve_lagrangian(
        lh2_data, work_dir=lh2_workdir, alpha=1.0, max_iters=5, tol=1.0,
        workers=3, subsolve_callback=sentinel_cb,
    )
    assert result is not None
    assert rec["kwargs"]["max_workers"] == 3            # workers -> max_workers
    assert rec["kwargs"]["subsolve_callback"] is sentinel_cb


def test_forwarding_omitted_when_signature_lacks(lh2_data, lh2_workdir, monkeypatch) -> None:
    rec: dict = {}
    _install_stub_lagrangian(monkeypatch, accepts=False, recorder=rec)
    # The stub ``.solve`` would raise TypeError if max_workers/subsolve
    # were forwarded; reaching a result proves they were gated out.
    result = solve_lagrangian(
        lh2_data, work_dir=lh2_workdir, alpha=1.0, max_iters=5, tol=1.0,
        workers=3, subsolve_callback=lambda entry: None,
    )
    assert result is not None
    assert rec["kwargs"] == {}


# ---------------------------------------------------------------------------
# D. END-TO-END against the live editable polar-high branch: workers + the
#    subsolve callback actually fire through the real LagrangianProblem.
# ---------------------------------------------------------------------------


def test_end_to_end_subsolve_callback_fires(lh2_data, lh2_workdir) -> None:
    """Real solve_lagrangian -> real polar-high LagrangianProblem.solve
    with workers=2 and a thread-safe collector.  Proves the FlexTool ->
    polar-high wiring works and the pinned callback schema holds."""
    # If the installed polar-high predates the parallel-subsolve API the
    # callback simply never fires; skip rather than silently pass.
    if "subsolve_callback" not in inspect.signature(
        __import__("polar_high").LagrangianProblem.solve
    ).parameters:
        pytest.skip("installed polar-high lacks subsolve_callback support")

    lock = threading.Lock()
    collected: list[dict] = []

    def _collector(entry: dict) -> None:
        with lock:
            collected.append(dict(entry))

    regions = ["region_A", "region_B", "region_C"]
    result = solve_lagrangian(
        lh2_data,
        work_dir=lh2_workdir,
        regions=regions,
        alpha=1.0,
        max_iters=10,
        tol=1.0,
        min_iters=3,
        workers=2,
        subsolve_callback=_collector,
    )
    assert result.iterations > 0

    assert collected, "subsolve_callback never fired end-to-end"
    events = {e.get("event") for e in collected}
    assert "start" in events and "finish" in events, (
        f"expected both start+finish events, got {events}"
    )
    for e in collected:
        # event discriminator
        assert e["event"] in ("start", "finish"), e
        # phase label is never a finish/done sentinel
        assert e["phase"] in ("initial", "iterate", "recovery"), e
        # subproblem is a 0-based region index in range
        assert isinstance(e["subproblem"], int)
        assert 0 <= e["subproblem"] < len(regions), e
        # finish-optimal entries carry a float obj
        if e["event"] == "finish" and "obj" in e:
            assert isinstance(e["obj"], float), e
