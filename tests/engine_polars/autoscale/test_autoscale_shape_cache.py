"""Regression test for the per-rolling-group autoscale DECISION cache.

Commit 8464dc76 (``fix(autoscale): cache scaling decision per rolling
group; skip per-roll range traversal``) added a per-structural-fingerprint
cache on the cascade solver:

* :class:`flextool.engine_polars._orchestration._AutoscaleShapeCacheEntry`
  stores the Layer-2 exponents + Layer-3 plan + pre/post
  :class:`RangeReport`s the autoscaler decided on the first solve of a
  given LP shape.
* On a same-shape roll the decision is replayed WITHOUT any range
  traversal — Layer 2 via
  :func:`flextool.engine_polars.autoscale._layer2.apply_layer2_with_exponents`
  (family layout only, no coefficient walk) and Layer 3 by re-applying the
  cached plan.  The per-roll :func:`detect_ranges` /
  ``_ranges_via_streaming`` Problem walk (the multi-GB ``priv_dirty``
  spike) is skipped.
* ``FLEXTOOL_DISABLE_AUTOSCALE_CACHE=1`` forces the OLD per-roll-traversal
  behaviour (always recompute).

This test runs the SAME small deterministic within-period rolling-dispatch
scenario through the real cascade solver twice — cache ON and cache OFF —
and asserts BOTH:

1. **Mechanism**: with the cache ON the autoscaler performs strictly fewer
   range-detection Problem traversals than with the cache OFF.  We count
   by wrapping :func:`_orchestration._autoscale_compute_ranges` (the import
   alias for :func:`polar_high.autoscale.detect_ranges`, the symbol used
   for every per-roll Layer-2 pre-solve, Layer-3 pre-solve, and Layer-1
   post-solve range read).  We separate the EXPENSIVE Problem-path walk
   (``streamed_lp_ranges`` absent → ``_ranges_via_streaming``) from the
   cheap post-solve Solution-path read, and assert the expensive walk
   count drops; the total count must also drop.

2. **Correctness**: the per-solve objective values are BYTE-IDENTICAL
   between the cache-ON and cache-OFF runs.  The fix is memory-only, so
   every solve's objective must match exactly (``==``, not approximate).

The scenario genuinely triggers the cache: ``coal_cum_within_period`` is a
4-roll within-period rolling solve (2 periods × 2 rolls of a 2-day
timeset) whose rolls share one structural fingerprint, and the autoscaler
actively engages (Layer 2 + Layer 3 fire — verified: a non-trivial
``user_bound_scale`` and non-empty Layer-2 exponents on the first solve).
That makes assertion (1) non-vacuous: the cached decision is a real
non-empty decision, not a stored ``None``.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd
import pytest

TEST_DIR = Path(__file__).resolve().parents[2]

if str(TEST_DIR) not in sys.path:
    sys.path.insert(0, str(TEST_DIR))

from db_utils import json_to_db  # noqa: E402

import test_commodity_ladder_rolling as _ladder  # noqa: E402
from flextool.engine_polars import (  # noqa: E402
    _orchestration,
    _warm,
    run_chain_from_db,
)
from flextool.update_flextool.db_migration import migrate_database  # noqa: E402

SCENARIO = "coal_cum_within_period"


@pytest.fixture(scope="module")
def shape_cache_db_url(tmp_path_factory: pytest.TempPathFactory) -> str:
    """Within-period CUMULATIVE rolling DB built from JSON/schema.

    Reuses the ladder test's scenario builder so we exercise the exact
    same 4-roll within-period rolling solve.  Built from the JSON fixture
    via :func:`json_to_db` + :func:`migrate_database` (never a checked-in
    ``.sqlite``, per the repo invariant).
    """
    db_path = tmp_path_factory.mktemp("db_autoscale_cache") / "tests.sqlite"
    url = json_to_db(TEST_DIR / "fixtures" / "tests.json", db_path)
    migrate_database(url, up_to=40)
    _ladder._add_within_period_rolling_scenarios(
        url,
        ladder_method="price_ladder_cumulative",
        tier1_quantity_mwh=1e9,
        tier2_price=1000.0,
        scenario_name=SCENARIO,
        alternative_name="ladder_cum_on",
    )
    return url


def _read_per_solve_objectives(workdir: Path) -> dict[str, float]:
    """Map each solve's parquet file → its objective value.

    The cascade writes one ``output_raw/v_obj__<solve>.parquet`` per
    realized roll; each has columns ``['objective', 'solve']``.  Keyed by
    file name so per-solve identity is preserved even when two rolls share
    an objective value.
    """
    raw = workdir / "output_raw"
    matches = sorted(raw.glob("v_obj__*.parquet"))
    assert matches, f"No v_obj parquet under {raw}"
    out: dict[str, float] = {}
    for pq in matches:
        df = pd.read_parquet(pq)
        assert not df.empty, f"Empty objective parquet: {pq.name}"
        out[pq.name] = float(df["objective"].iloc[-1])
    return out


class _RangeWalkCounter:
    """Wrap ``_orchestration._autoscale_compute_ranges`` with a call counter.

    ``_autoscale_compute_ranges`` is the orchestration import alias for
    :func:`polar_high.autoscale.detect_ranges`.  It is called for the
    Layer-2 pre-solve walk, the Layer-3 pre-solve walk, and the Layer-1
    post-solve read.  The first two take a polar-high ``Problem`` and run
    the expensive ``_ranges_via_streaming`` traversal; the last takes a
    ``Solution`` carrying ``streamed_lp_ranges`` and is cheap.  We count
    both, separating the expensive Problem-path walk (the thing the cache
    eliminates) from the total.  The wrapper still calls through, so
    behaviour is unchanged.
    """

    def __init__(self) -> None:
        self.total = 0
        self.problem_walk = 0
        self._orig = _orchestration._autoscale_compute_ranges

    def __enter__(self) -> "_RangeWalkCounter":
        orig = self._orig

        def wrapped(arg, cfg):  # type: ignore[no-untyped-def]
            self.total += 1
            streamed = getattr(arg, "streamed_lp_ranges", None)
            if not isinstance(streamed, dict):
                # Pre-solve Problem path → the expensive range traversal.
                self.problem_walk += 1
            return orig(arg, cfg)

        _orchestration._autoscale_compute_ranges = wrapped
        return self

    def __exit__(self, *exc: object) -> None:
        _orchestration._autoscale_compute_ranges = self._orig


def _run_scenario(
    db_url: str, workdir: Path,
) -> tuple[dict[str, float], _RangeWalkCounter]:
    """Run the scenario once through a FRESH cascade solver.

    ``run_chain_from_db`` constructs a new orchestrator (and thus a fresh,
    empty ``_autoscale_shape_cache``) per call, so the two runs cannot
    leak cache state into each other.  Returns the per-solve objectives
    and the populated range-walk counter.
    """
    os.chdir(workdir)
    counter = _RangeWalkCounter()
    with counter:
        steps = run_chain_from_db(
            db_url, SCENARIO, work_folder=workdir,
            warm=True, keep_solutions=True,
        )
    last_step = next(reversed(list(steps.values())))
    assert last_step.optimal, (
        f"Scenario '{SCENARIO}' did not solve to optimality"
    )
    return _read_per_solve_objectives(workdir), counter


def test_shape_cache_skips_range_walks_and_preserves_objectives(
    shape_cache_db_url: str,
    tmp_path_factory: pytest.TempPathFactory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cache ON does strictly fewer range walks than cache OFF, with
    byte-identical per-solve objectives."""
    # ---- Run with the cache OFF (forced per-roll traversal) ----
    monkeypatch.setenv("FLEXTOOL_DISABLE_AUTOSCALE_CACHE", "1")
    off_dir = tmp_path_factory.mktemp("autoscale_cache_off")
    off_objs, off_counter = _run_scenario(shape_cache_db_url, off_dir)
    monkeypatch.delenv("FLEXTOOL_DISABLE_AUTOSCALE_CACHE", raising=False)

    # ---- Run with the cache ON (default) ----
    monkeypatch.delenv("FLEXTOOL_DISABLE_AUTOSCALE_CACHE", raising=False)
    on_dir = tmp_path_factory.mktemp("autoscale_cache_on")
    on_objs, on_counter = _run_scenario(shape_cache_db_url, on_dir)

    # ---- Guard: the scenario must actually be a multi-roll solve so the
    # cache has same-shape rolls to hit, otherwise assertion (1) is
    # vacuous. ----
    assert len(off_objs) >= 2, (
        f"Expected >= 2 solves to exercise the cache, got {len(off_objs)}: "
        f"{sorted(off_objs)}"
    )

    # ---- Guard: autoscale must engage (the expensive walk must run at
    # least once with the cache off), else the cache stores nothing to
    # replay and assertion (1) is vacuous. ----
    assert off_counter.problem_walk > 0, (
        "autoscale never ran a range traversal with the cache off — "
        "Layer 1/2 did not engage on this scenario, so the cache test is "
        "vacuous"
    )

    # ---- (1) MECHANISM: cache ON performs strictly fewer range walks. ----
    assert on_counter.problem_walk < off_counter.problem_walk, (
        "autoscale shape cache did not reduce the expensive Problem-path "
        f"range traversals: cache-off={off_counter.problem_walk}, "
        f"cache-on={on_counter.problem_walk}"
    )
    assert on_counter.total < off_counter.total, (
        "autoscale shape cache did not reduce the total detect_ranges "
        f"calls: cache-off={off_counter.total}, cache-on={on_counter.total}"
    )

    # ---- (2) CORRECTNESS: per-solve objectives byte-identical. ----
    assert set(on_objs) == set(off_objs), (
        "cache-on and cache-off produced different solve sets: "
        f"on={sorted(on_objs)} off={sorted(off_objs)}"
    )
    for key in off_objs:
        assert on_objs[key] == off_objs[key], (
            f"objective for solve {key!r} changed with the cache: "
            f"cache-off={off_objs[key]!r}, cache-on={on_objs[key]!r} "
            "(the fix is memory-only; results must be byte-identical)"
        )


class _FingerprintPerturber:
    """Force ``_fingerprint(data)`` to return a UNIQUE value per solve.

    Reproduces the real DES condition that motivated commit ``b7751db5``:
    on a rolling-dispatch cascade the structural data fingerprint
    (:func:`flextool.engine_polars._warm._fingerprint`) *slides* every
    roll — a windowed period/dt field's height tracks the rolling
    horizon — even though every roll builds an LP of *identical shape*.

    The orchestrator imports ``_fingerprint`` from ``_warm`` *inside* its
    ``run`` method (a runtime ``from ... import _fingerprint``) and calls it
    exactly ONCE per solve iteration (``fp = _fingerprint(data)``).  Because
    that binding is re-resolved from the ``_warm`` module on every ``run``
    call, the perturbation must replace the attribute on the SOURCE module
    (:mod:`flextool.engine_polars._warm`), not on ``_orchestration``.
    Returning a fresh incrementing tuple on every call
    therefore yields a distinct fingerprint per solve — exactly the DES
    behaviour.  Because the value differs from the previous solve's, the
    warm-reuse predicate ``self._prior_fp == fp`` is never satisfied, so
    every roll cold-rebuilds the warm problem (``tried_warm`` False) and
    re-enters the per-shape autoscale block.  That is the correct DES
    analogue: the OLD ``_fingerprint``-keyed decision cache would MISS
    every roll, while the NEW LP-shape-keyed cache must still HIT.

    A per-call counter (rather than keying on ``id(data)``) is sufficient
    and faithful here precisely because ``_fingerprint`` is invoked once
    per solve; there is no intra-solve second call that would need a
    stable value.
    """

    def __init__(self) -> None:
        self.calls = 0
        self._orig = _warm._fingerprint

    def __enter__(self) -> "_FingerprintPerturber":
        def perturbed(data):  # type: ignore[no-untyped-def]
            self.calls += 1
            # Unique per call ⇒ unique per solve ⇒ slides like DES.
            return ("perturbed_unique_fp", self.calls)

        _warm._fingerprint = perturbed
        return self

    def __exit__(self, *exc: object) -> None:
        _warm._fingerprint = self._orig


def _run_scenario_perturbed(
    db_url: str, workdir: Path,
) -> tuple[dict[str, float], _RangeWalkCounter, _FingerprintPerturber]:
    """Run the scenario with ``_fingerprint`` perturbed to slide per solve.

    Mirrors :func:`_run_scenario` but additionally installs
    :class:`_FingerprintPerturber` for the duration of the cascade so the
    structural fingerprint differs on every roll — disabling warm reuse and
    making the autoscale decision cache the ONLY mechanism that can spare a
    same-shape roll its range traversal.
    """
    os.chdir(workdir)
    counter = _RangeWalkCounter()
    perturber = _FingerprintPerturber()
    with perturber, counter:
        steps = run_chain_from_db(
            db_url, SCENARIO, work_folder=workdir,
            warm=True, keep_solutions=True,
        )
    last_step = next(reversed(list(steps.values())))
    assert last_step.optimal, (
        f"Scenario '{SCENARIO}' did not solve to optimality"
    )
    return _read_per_solve_objectives(workdir), counter, perturber


def test_shape_cache_hits_when_fingerprint_slides_per_solve(
    shape_cache_db_url: str,
    tmp_path_factory: pytest.TempPathFactory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Re-key regression (commit ``b7751db5``): with a per-solve-sliding
    ``_fingerprint``, the LP-SHAPE-keyed decision cache still hits.

    The DES condition is "fingerprint differs per solve, but the LP shape
    is identical".  We reproduce it on the fast ``coal_cum_within_period``
    rolling scenario by perturbing ``_orchestration._fingerprint`` to
    return a unique value per call.  Under that perturbation:

    * The OLD ``_fingerprint``-keyed decision cache (pre-``b7751db5``)
      would MISS every roll — the cache key slides with the fingerprint —
      so cache-ON would walk exactly as much as cache-OFF.
    * The NEW :func:`_autoscale_lp_shape_signature`-keyed cache keys on the
      built LP's structural signature, which is invariant across the rolls,
      so it MUST still HIT on same-shape rolls.

    Both runs apply the SAME ``_fingerprint`` perturbation, so the only
    variable is the ``FLEXTOOL_DISABLE_AUTOSCALE_CACHE`` gate.  We assert
    that with the perturbation active cache-ON does STRICTLY FEWER
    expensive Problem-path range walks than cache-OFF, and that per-solve
    objectives are byte-identical.  This FAILS on pre-``b7751db5`` code
    (the fp-keyed cache never hits under the perturbation) and PASSES now.
    """
    # ---- Run with the cache OFF, fingerprint sliding per solve. ----
    monkeypatch.setenv("FLEXTOOL_DISABLE_AUTOSCALE_CACHE", "1")
    off_dir = tmp_path_factory.mktemp("autoscale_pert_off")
    off_objs, off_counter, off_pert = _run_scenario_perturbed(
        shape_cache_db_url, off_dir,
    )
    monkeypatch.delenv("FLEXTOOL_DISABLE_AUTOSCALE_CACHE", raising=False)

    # ---- Run with the cache ON, same fingerprint perturbation. ----
    on_dir = tmp_path_factory.mktemp("autoscale_pert_on")
    on_objs, on_counter, on_pert = _run_scenario_perturbed(
        shape_cache_db_url, on_dir,
    )

    # ---- Guard: the perturbation actually took effect — a unique
    # fingerprint was produced for every solve, on BOTH runs. ----
    assert off_pert.calls >= 2 and on_pert.calls >= 2, (
        "perturbed _fingerprint was not invoked once per solve as expected: "
        f"cache-off calls={off_pert.calls}, cache-on calls={on_pert.calls}"
    )
    assert off_pert.calls == on_pert.calls, (
        "the two runs took different solve paths under the perturbation: "
        f"cache-off fingerprint calls={off_pert.calls}, "
        f"cache-on={on_pert.calls}"
    )

    # ---- Guard: multi-roll, and autoscale actually engaged with the
    # cache off, else assertion (1) is vacuous. ----
    assert len(off_objs) >= 2, (
        f"Expected >= 2 solves to exercise the cache, got {len(off_objs)}: "
        f"{sorted(off_objs)}"
    )
    assert off_counter.problem_walk > 0, (
        "autoscale never ran a range traversal with the cache off under the "
        "perturbation — Layer 1/2 did not engage, so this test is vacuous"
    )

    # ---- (1) MECHANISM: even though the fingerprint slides per solve, the
    # LP-shape-keyed cache hits on same-shape rolls and skips their
    # expensive Problem-path walks.  The OLD fp-keyed cache could not, so
    # this strict inequality is the actual re-key guard. ----
    assert on_counter.problem_walk < off_counter.problem_walk, (
        "with a per-solve-sliding _fingerprint the LP-shape autoscale cache "
        "did not reduce the expensive Problem-path range traversals — the "
        "decision cache is keyed on the fingerprint, not the LP shape "
        "(pre-b7751db5 behaviour): "
        f"cache-off={off_counter.problem_walk}, "
        f"cache-on={on_counter.problem_walk}"
    )
    assert on_counter.total < off_counter.total, (
        "with a per-solve-sliding _fingerprint the LP-shape autoscale cache "
        "did not reduce the total detect_ranges calls: "
        f"cache-off={off_counter.total}, cache-on={on_counter.total}"
    )

    # ---- (2) CORRECTNESS: per-solve objectives byte-identical. ----
    assert set(on_objs) == set(off_objs), (
        "cache-on and cache-off produced different solve sets under the "
        f"perturbation: on={sorted(on_objs)} off={sorted(off_objs)}"
    )
    for key in off_objs:
        assert on_objs[key] == off_objs[key], (
            f"objective for solve {key!r} changed with the cache under the "
            f"perturbation: cache-off={off_objs[key]!r}, "
            f"cache-on={on_objs[key]!r} (decision cache must be memory-only)"
        )
