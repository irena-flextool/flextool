"""Per-roll parity gate for the pdtProfile full_dt conversion (R4).

``pdtProfile`` is a BRANCH-DEPENDENT ``(d,t)`` family on the "solve as
picker" registry (specs §9/§10): it is computed ONCE over the
BRANCH-EXPANDED complete-solve timeline per ``complete_solve`` and sliced to
each roll's window via
:func:`flextool.engine_polars._full_dt.slice_full_dt`, rather than
re-derived per roll.

For EVERY roll/solve on BOTH a rolling non-stochastic fixture
(``fullYear_roll``) AND a stochastic fixture (``2_day_stochastic_dispatch``,
which carries the branch-periods ``period1_lower`` / ``period1_mid`` /
``period1_upper`` and exercises the stochastic / parent-period profile
fold-in), this test asserts the sliced-from-full frame that
``emit_pdtProfile`` writes is byte-identical (schema + every Utf8 value
cell, INCLUDING the branch-period rows and their DISTINCT per-branch values)
to the legacy per-roll :func:`derive_pdtProfile` recomputed from the same
roll's ``steps_in_use`` window + per-roll branch maps.

It also asserts the full_dt path was actually exercised (the cache frame was
non-None) and that the frame is non-empty — otherwise the gate would be
vacuous — and, on the stochastic fixture, that branch-period rows are
present (R4's whole point).
"""
from __future__ import annotations

from pathlib import Path
import tempfile

import pytest

from flextool.engine_polars import run_chain_from_db
from flextool.engine_polars import _emit_period_params
from flextool.engine_polars._emit_provider_io import _provider_key


# (scenario, db_fixture, expect_branch_periods)
_CASES = [
    ("fullYear_roll", "main", False),
    ("2_day_stochastic_dispatch", "stochastic", True),
]


@pytest.mark.parametrize("scenario,db_fixture,expect_branch_periods", _CASES)
def test_pdtProfile_sliced_from_full_matches_legacy_every_roll(
    scenario, db_fixture, expect_branch_periods,
    scenario_workdir, monkeypatch,
) -> None:
    db = scenario_workdir(scenario, db_fixture=db_fixture) / "tests.sqlite"

    rolls_checked: list[dict] = []
    real_emit = _emit_period_params.emit_pdtProfile

    def _spy_emit(input_dir, solve_data_dir, *, provider,
                  full_dt_frame=None):
        real_emit(
            input_dir, solve_data_dir, provider=provider,
            full_dt_frame=full_dt_frame,
        )
        emitted = provider.get(
            _provider_key(Path(solve_data_dir) / "pdtProfile.csv")
        )
        legacy = _emit_period_params.derive_pdtProfile(
            Path(input_dir), Path(solve_data_dir), provider=provider,
        )
        rolls_checked.append(
            {
                "took_full_path": full_dt_frame is not None,
                "emitted": emitted,
                "legacy": legacy,
            }
        )

    monkeypatch.setattr(
        _emit_period_params, "emit_pdtProfile", _spy_emit,
    )

    with tempfile.TemporaryDirectory() as t:
        wf = Path(t) / "default"
        run_chain_from_db(
            db, scenario_name=scenario, work_folder=wf,
            keep_solutions=True,
        )

    assert rolls_checked, (
        "emit_pdtProfile was never called — no rolls run"
    )

    # The full_dt compute-once path must have been exercised for every roll.
    assert all(r["took_full_path"] for r in rolls_checked), (
        "some rolls fell back to the legacy per-roll derivation — the "
        "full_dt cache frame was absent"
    )

    # Non-vacuous: the family has rows.
    assert any(r["emitted"].height > 0 for r in rolls_checked), (
        f"pdtProfile was empty for every roll on {scenario}"
    )

    # On the stochastic fixture, branch-period rows must be present
    # (the R4 invariant) — otherwise the gate would not exercise the fix.
    if expect_branch_periods:
        seen_branch = any(
            r["emitted"].height > 0
            and any(
                "_" in p
                for p in r["emitted"]["period"].unique().to_list()
            )
            for r in rolls_checked
        )
        assert seen_branch, (
            f"{scenario}: no branch-period rows (e.g. 'period1_lower') in "
            f"emitted pdtProfile — branch expansion did not reach it"
        )

    # Per-roll byte-identical parity: sliced-from-full == legacy per-roll.
    mismatches: list[str] = []
    for idx, r in enumerate(rolls_checked):
        emitted = r["emitted"]
        legacy = r["legacy"]
        if emitted.schema != legacy.schema:
            mismatches.append(
                f"roll {idx}: schema {emitted.schema} != {legacy.schema}"
            )
            continue
        if not emitted.equals(legacy):
            ep = set(emitted["period"].unique().to_list()) if emitted.height else set()
            lp = set(legacy["period"].unique().to_list()) if legacy.height else set()
            mismatches.append(
                f"roll {idx}: frame differs (shape {emitted.shape} vs "
                f"{legacy.shape}; emitted-only periods "
                f"{sorted(ep - lp)}, legacy-only periods {sorted(lp - ep)})"
            )

    assert not mismatches, (
        f"{scenario}: pdtProfile sliced-from-full != legacy per-roll "
        f"on {len(mismatches)}/{len(rolls_checked)} rolls:\n  "
        + "\n  ".join(mismatches)
    )
