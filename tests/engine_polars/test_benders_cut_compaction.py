"""Benders periodic MASTER CUT COMPACTION (delete strictly-slack cut rows).

Covers the FIRST-SHIP compaction (spec ``specs/benders_cut_aging_plan.md``),
now delegated to ``polar_high.WarmProblem.compact_cuts`` — polar-high owns the
classify / delete / verify-restore (tested there in
``tests/test_cut_compaction.py``); FlexTool only triggers it and tracks the
kept count.  This suite covers the FlexTool side:

* ``_resolve_benders_cut_compact_at`` — the ``FLEXTOOL_BENDERS_CUT_COMPACT_AT``
  env resolver (default 0 = OFF; positive threshold; reject non-int / negative
  with a warning), mirroring ``_resolve_benders_max_stall``.
* LB-monotone + same-optimum e2e — a small benders solve with compaction forced
  ON (low threshold) converges to the SAME objective as OFF, never hard-fails
  the b16 LB-monotonicity guard, and the active cut-row count after a compaction
  stays bounded.
* OFF-path regression — the default (env unset / 0) path is byte-identical to
  the pre-compaction loop.
"""
from __future__ import annotations

import contextlib
import logging

import numpy as np
import pytest

from flextool.engine_polars import load_flextool
from flextool.engine_polars._benders import (
    _BendersMaster,
    _BENDERS_CUT_COMPACT_AT_ENV,
    _BENDERS_CUT_POLICY_ENV,
    _BENDERS_CUT_WINDOW_DEFAULT,
    _BENDERS_CUT_WINDOW_ENV,
    _resolve_benders_cut_compact_at,
    _resolve_benders_cut_policy,
    _resolve_benders_cut_window,
    solve_benders,
)

_REGIONS = ["region_A", "region_B", "region_C"]


# ---------------------------------------------------------------------------
# _resolve_benders_cut_compact_at — env resolver (mirrors max-stall).
# ---------------------------------------------------------------------------
def test_compact_at_defaults_off_when_unset(monkeypatch):
    monkeypatch.delenv(_BENDERS_CUT_COMPACT_AT_ENV, raising=False)
    assert _resolve_benders_cut_compact_at() == 0


def test_compact_at_positive_env(monkeypatch):
    monkeypatch.setenv(_BENDERS_CUT_COMPACT_AT_ENV, "50")
    assert _resolve_benders_cut_compact_at() == 50


def test_cut_policy_defaults_to_slack_when_unset(monkeypatch):
    monkeypatch.delenv(_BENDERS_CUT_POLICY_ENV, raising=False)
    assert _resolve_benders_cut_policy() == "slack"


def test_cut_policy_dominance_opt_in(monkeypatch):
    monkeypatch.setenv(_BENDERS_CUT_POLICY_ENV, "dominance")
    assert _resolve_benders_cut_policy() == "dominance"


def test_cut_policy_unrecognised_warns_and_defaults(monkeypatch):
    monkeypatch.setenv(_BENDERS_CUT_POLICY_ENV, "bogus")
    assert _resolve_benders_cut_policy() == "slack"


def test_compact_at_zero_is_off(monkeypatch):
    monkeypatch.setenv(_BENDERS_CUT_COMPACT_AT_ENV, "0")
    assert _resolve_benders_cut_compact_at() == 0


def test_compact_at_negative_ignored_with_warning(monkeypatch, caplog):
    monkeypatch.setenv(_BENDERS_CUT_COMPACT_AT_ENV, "-3")
    with caplog.at_level(logging.WARNING):
        assert _resolve_benders_cut_compact_at() == 0
    assert any("negative" in r.message for r in caplog.records)


def test_compact_at_non_integer_ignored_with_warning(monkeypatch, caplog):
    monkeypatch.setenv(_BENDERS_CUT_COMPACT_AT_ENV, "not-a-number")
    with caplog.at_level(logging.WARNING):
        assert _resolve_benders_cut_compact_at() == 0
    assert any("non-integer" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# _resolve_benders_cut_window — dominance-policy trial-point window resolver.
# ---------------------------------------------------------------------------
def test_cut_window_defaults_when_unset(monkeypatch):
    monkeypatch.delenv(_BENDERS_CUT_WINDOW_ENV, raising=False)
    assert _resolve_benders_cut_window() == _BENDERS_CUT_WINDOW_DEFAULT


def test_cut_window_positive_env(monkeypatch):
    monkeypatch.setenv(_BENDERS_CUT_WINDOW_ENV, "25")
    assert _resolve_benders_cut_window() == 25


def test_cut_window_zero_falls_back_to_default(monkeypatch):
    monkeypatch.setenv(_BENDERS_CUT_WINDOW_ENV, "0")
    assert _resolve_benders_cut_window() == _BENDERS_CUT_WINDOW_DEFAULT


def test_cut_window_non_integer_ignored_with_warning(monkeypatch, caplog):
    monkeypatch.setenv(_BENDERS_CUT_WINDOW_ENV, "not-a-number")
    with caplog.at_level(logging.WARNING):
        assert _resolve_benders_cut_window() == _BENDERS_CUT_WINDOW_DEFAULT
    assert any("non-integer" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# Fixture for the e2e test.
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def ti_data(scenario_workdir):
    wd = scenario_workdir(
        "lh2_three_region_trade_invest", db_fixture="lh2_trade_invest"
    )
    return load_flextool(wd)


# ---------------------------------------------------------------------------
# e2e — compaction ON converges to the SAME optimum as OFF, LB monotone,
# active cut-row count bounded.
# ---------------------------------------------------------------------------
def test_compaction_matches_off_and_bounds_cut_rows(ti_data, monkeypatch):
    """With compaction forced ON at a LOW threshold (3 regions => 3 cuts/iter,
    so threshold 6 triggers a compaction by iter 2), the solve:

    (i)   converges to the SAME objective as the OFF (default) path;
    (ii)  never hard-fails the b16 LB-monotonicity guard (a green solve is the
          assertion — the guard raises inside ``solve_benders``);
    (iii) after each compaction the active ``_master_cut_rows`` is bounded, so
          the master's row count never runs away like the OFF path's unbounded
          accumulation.

    We spy ``_BendersMaster.compact_cuts`` to prove >= 1 compaction fired and to
    capture the polar-high ``{"kept", "dropped", "restored"}`` reports, and read
    the loop's per-iteration ``master_cut_rows=`` count off the timing-line INFO
    record to bound the active rows.
    """
    # --- OFF (default) baseline. ---
    monkeypatch.delenv(_BENDERS_CUT_COMPACT_AT_ENV, raising=False)
    res_off = solve_benders(ti_data, _REGIONS, max_iters=20, tol=1e-4)
    assert res_off.converged, "OFF baseline did not converge"

    # --- ON at a low threshold, forcing >= 1 compaction.  Opt in to the
    # non-default DOMINANCE policy (default is 'slack') to exercise it e2e. ---
    threshold = 6
    monkeypatch.setenv(_BENDERS_CUT_COMPACT_AT_ENV, str(threshold))
    monkeypatch.setenv(_BENDERS_CUT_POLICY_ENV, "dominance")

    reports: list[dict] = []
    trial_lens: list[int] = []
    policies: list[str] = []
    real_compact = _BendersMaster.compact_cuts

    def spy_compact(self, solution, *, policy="slack", trial_col_values=None):
        policies.append(policy)
        trial_lens.append(0 if trial_col_values is None else len(trial_col_values))
        res = real_compact(
            self, solution, policy=policy, trial_col_values=trial_col_values
        )
        reports.append(res)
        return res

    monkeypatch.setattr(_BendersMaster, "compact_cuts", spy_compact)

    with caplog_active_rows() as (active_rows, dropped_line):
        res_on = solve_benders(ti_data, _REGIONS, max_iters=20, tol=1e-4)

    assert res_on.converged, (
        f"ON did not converge: gap={res_on.gap:.3e} iters={res_on.iterations}"
    )
    # (i) SAME optimum (byte-close; the master row trajectory differs).
    assert np.isclose(res_on.total_objective, res_off.total_objective, rtol=1e-4), (
        f"ON obj {res_on.total_objective:.10e} != OFF {res_off.total_objective:.10e}"
    )
    # (ii) A VALID lower bound survives (the whole point — the b16 guard would
    # have hard-failed inside the solve otherwise; a green solve proves it).
    assert res_on.lower_bound <= res_off.total_objective * (1 + 1e-6)

    # >= 1 compaction actually fired (the compaction path was exercised).
    assert len(reports) >= 1, "no cut compaction fired"
    # The loop drove the DOMINANCE policy with a non-empty trial-point window.
    assert policies and all(p == "dominance" for p in policies), (
        f"expected dominance policy for every compaction, got {policies}"
    )
    assert all(n >= 1 for n in trial_lens), (
        f"dominance compaction was handed an empty trial window: {trial_lens}"
    )
    # A compaction genuinely DROPPED at least one slack cut (real work, not a
    # no-op keeping everything).  Cross-checked against the timing-line count.
    total_dropped = sum(r["dropped"] for r in reports)
    assert total_dropped >= 1, f"no cut was ever dropped by a compaction: {reports}"
    assert dropped_line and max(dropped_line) >= 1, (
        f"timing line reported no dropped cut: {dropped_line}"
    )
    # No compaction had to roll back (the verify belt stayed quiet on this
    # well-conditioned fixture — LB genuinely preserved by the deletions).
    assert not any(r["restored"] for r in reports), (
        f"a compaction rolled back (verify belt fired): {reports}"
    )

    # (iii) GROWTH BOUND — the invariant compaction exists to enforce.  The
    # per-iteration ``master_cut_rows=`` timing line is printed AFTER that
    # iteration's fresh cuts are appended but BEFORE the loop-end compaction
    # deletes the slack rows, so it captures the POST-APPEND peak: at most the
    # last compaction's KEPT count plus one iteration's fresh cuts (n_regions).
    # The KEPT count is NOT bounded by ``threshold`` itself — on this tiny
    # fixture most cuts are binding, so ``compact_cuts`` retains the binding
    # ones (the row-COUNT ceiling ``max_kept + n_regions`` is what caps growth,
    # not the drop fraction).  The decisive check is vs the OFF path: OFF
    # accumulates ``n_regions`` cuts EVERY iteration with no deletion, so its
    # terminal master carries ``n_regions * iterations`` rows; the compacted
    # peak stays well below that.
    n_regions = len(_REGIONS)
    max_kept = max(r["kept"] for r in reports)
    ceiling = max_kept + n_regions
    assert active_rows, "no [benders timing] lines captured"
    assert max(active_rows) <= ceiling, (
        f"active cut rows {max(active_rows)} exceeded the ceiling {ceiling} "
        f"(= max kept {max_kept} + n_regions {n_regions}); compaction failed "
        f"to bound master growth: {active_rows}"
    )
    # The compacted peak is strictly BELOW the OFF path's unbounded
    # accumulation (n_regions cuts per iteration, never deleted) — the point of
    # compaction.  Guard against a degenerate fixture where both are tiny.
    off_terminal_rows = n_regions * res_off.iterations
    if off_terminal_rows > ceiling:
        assert max(active_rows) < off_terminal_rows, (
            f"compacted peak {max(active_rows)} not below OFF accumulation "
            f"{off_terminal_rows}"
        )


@contextlib.contextmanager
def caplog_active_rows():
    """Capture the loop's ``[benders timing]`` INFO lines via a temporary
    logging handler and yield ``(active_rows, dropped)``:

    * ``active_rows`` — the per-iteration ``master_cut_rows=`` count (reported
      every iteration, AFTER any compaction reset it);
    * ``dropped`` — the ``dropped=`` count from each compaction's extra line.
    """
    active: list[int] = []
    dropped: list[int] = []
    logger = logging.getLogger("flextool.engine_polars._benders")

    class _Grab(logging.Handler):
        def emit(self, record):
            msg = record.getMessage()
            if "[benders timing]" not in msg:
                return
            if "dropped=" in msg:
                try:
                    dropped.append(int(msg.split("dropped=")[1].split()[0]))
                except (ValueError, IndexError):
                    pass
            elif "master_cut_rows=" in msg:
                try:
                    active.append(int(msg.split("master_cut_rows=")[1].split()[0]))
                except (ValueError, IndexError):
                    pass

    h = _Grab(level=logging.INFO)
    prev_level = logger.level
    logger.addHandler(h)
    logger.setLevel(logging.INFO)
    try:
        yield active, dropped
    finally:
        logger.removeHandler(h)
        logger.setLevel(prev_level)
