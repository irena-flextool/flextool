"""Cross-level HiGHS-retention regression.

A prior solve-LEVEL's live ``Solution.highs`` (and ``flex_data_provider``)
must be released BEFORE the next level builds its FlexData + LP — otherwise
the two levels' footprints coexist (the DES storage+dispatch ~2x peak that
drove the 7/9 near-OOM).

This runs a tiny 2-level (investment -> dispatch) cascade with the env-gated
cross-level audit on and asserts that no EXHAUSTED solve-level is still
holding a live ``.highs`` at the instant a later-level solve builds.  The
synthetic LP is trivial (sub-second per solve); the bug is structural, not
size-dependent, so we assert the release invariant directly rather than
measuring GB.

Pre-fix: FAILS — the investment level's ``Solution.highs`` is only nulled by
the dispatch solve's POST-solve slim, i.e. after the dispatch LP already
built on top of it.
"""
from __future__ import annotations

import pytest

from flextool.engine_polars import _orchestration, run_chain_from_db

pytestmark = pytest.mark.solver

SCEN = "5weeks_invest_fullYear_dispatch_coal_wind"


def _step_obj(step) -> float:
    """Objective for a step, robust to keep_solutions slimming (the scalar
    ``obj`` survives the slim even when ``solution`` is dropped)."""
    for o in (getattr(step, "obj", None),
              getattr(getattr(step, "solution", None), "obj", None)):
        if o is not None:
            return float(o)
    raise AssertionError("no objective on step")


def test_exhausted_level_highs_released_before_next_build(
    scenario_workdir, monkeypatch,
) -> None:
    work = scenario_workdir(SCEN)
    db_path = work / "tests.sqlite"

    # Reference: keep_solutions=True retains everything (release gated off),
    # so its per-step objectives are the ground truth.
    ref = run_chain_from_db(
        db_path, scenario_name=SCEN, warm=True, keep_solutions=True,
    )
    ref_obj = {k: _step_obj(s) for k, s in ref.items()}

    # Subject: keep_solutions=False runs the cross-level release + provider
    # eviction.  Audit must show zero violators AND objectives must match the
    # reference bit-for-bit (the release/eviction is memory-only).
    monkeypatch.setenv("FLEXTOOL_LEVEL_RELEASE_AUDIT", "1")
    _orchestration._LEVEL_RELEASE_AUDIT.clear()
    sub = run_chain_from_db(
        db_path, scenario_name=SCEN, warm=True, keep_solutions=False,
    )

    audit = _orchestration._LEVEL_RELEASE_AUDIT
    assert audit, (
        "cross-level audit recorded nothing — hook not firing or cascade had "
        "fewer than one solve"
    )
    violations = [r for r in audit if r.get("violators")]
    assert not violations, (
        "an EXHAUSTED solve-level still held a live Solution.highs at the "
        "moment a later-level solve built its LP (cross-level retention bug — "
        f"two level footprints coexist). Violation records: {violations}"
    )

    # Correctness: releasing highs + evicting the exhausted-level provider
    # must not change any solve's result.
    sub_obj = {k: _step_obj(s) for k, s in sub.items()}
    assert sub_obj.keys() == ref_obj.keys()
    for k in ref_obj:
        assert sub_obj[k] == pytest.approx(ref_obj[k], rel=1e-9, abs=1e-6), (
            f"objective for {k} changed under keep_solutions=False "
            f"(release/eviction altered results): {sub_obj[k]} vs {ref_obj[k]}"
        )
