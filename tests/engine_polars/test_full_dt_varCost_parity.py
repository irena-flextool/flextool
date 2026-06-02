"""Per-roll parity gate for the varCost full_dt conversion (R6b).

The varCost chain joins the "solve as picker" full_dt registry (specs
§9/§10) as a CACHE-READ chain: each family is computed ONCE over the
BRANCH-EXPANDED complete-solve timeline per ``complete_solve`` — reading its
UPSTREAM families' cached complete-timeline frames from the in-progress
per-``cs`` cache (R6b) and RECONSTRUCTING its ``(p, src, snk)`` domains
in-place from solve-invariant topology inputs (R6a) — then sliced to each
roll's window via
:func:`flextool.engine_polars._full_dt.slice_full_dt`, rather than
re-derived per roll.

Two producer families::

    pdtProcess__source__sink__dt_varCost
    pdtProcess__source__sink__dt_varCost_alwaysProcess

and four downstream filter families::

    pssdt_varCost_noEff
    pssdt_varCost_eff_unit_source
    pssdt_varCost_eff_unit_sink
    pssdt_varCost_eff_connection

For EVERY roll/solve on BOTH a rolling non-stochastic fixture
(``fullYear_roll``) AND a stochastic fixture (``2_day_stochastic_dispatch``,
which carries the branch-periods), this test asserts the sliced-from-full
frame each emit writes is byte-identical (schema + every Utf8 value cell,
INCLUDING the branch-period rows) to the legacy per-roll derivation
recomputed from the same roll's ``steps_in_use`` window + the per-roll
``pdtProcess*`` / ``process_source_sink*`` frames carried in this Provider.

It also asserts the full_dt path was actually exercised (the cache dict was
present) and that the producer is non-empty — otherwise the gate would be
vacuous.
"""
from __future__ import annotations

from pathlib import Path
import tempfile

import pytest

from flextool.engine_polars import run_chain_from_db
from flextool.engine_polars import _emit_period_params as _period
from flextool.engine_polars._emit_provider_io import _provider_key


# (scenario, db_fixture, expect_branch_periods)
_CASES = [
    ("fullYear_roll", "main", False),
    ("2_day_stochastic_dispatch", "stochastic", True),
]

# Emitted Provider key (basename) ↔ legacy per-roll deriver.
_VARCOST_KEYS = {
    "pdtProcess__source__sink__dt_varCost":
        _period.derive_pdtProcess__source__sink__dt_varCost,
    "pdtProcess__source__sink__dt_varCost_alwaysProcess":
        _period.derive_pdtProcess__source__sink__dt_varCost_alwaysProcess,
}
_PSSDT_KEYS = {
    "pssdt_varCost_noEff":
        _period.derive_pssdt_varCost_noEff,
    "pssdt_varCost_eff_unit_source":
        _period.derive_pssdt_varCost_eff_unit_source,
    "pssdt_varCost_eff_unit_sink":
        _period.derive_pssdt_varCost_eff_unit_sink,
    "pssdt_varCost_eff_connection":
        _period.derive_pssdt_varCost_eff_connection,
}


@pytest.mark.parametrize("scenario,db_fixture,expect_branch_periods", _CASES)
def test_varCost_sliced_from_full_matches_legacy_every_roll(
    scenario, db_fixture, expect_branch_periods,
    scenario_workdir, monkeypatch,
) -> None:
    db = scenario_workdir(scenario, db_fixture=db_fixture) / "tests.sqlite"

    rolls_checked: list[dict] = []

    real_varcost = _period.emit_pdtProcess__source__sink__dt_varCost_pair
    real_pssdt = _period.emit_pssdt_varCost_filters

    def _spy_varcost(input_dir, solve_data_dir, *, provider,
                     full_dt_frames=None):
        real_varcost(
            input_dir, solve_data_dir, provider=provider,
            full_dt_frames=full_dt_frames,
        )
        rec = {"took_full_path": full_dt_frames is not None, "pairs": {}}
        for basename, legacy_fn in _VARCOST_KEYS.items():
            emitted = provider.get(
                _provider_key(Path(solve_data_dir) / f"{basename}.csv")
            )
            legacy = legacy_fn(
                Path(input_dir), Path(solve_data_dir), provider=provider,
            )
            rec["pairs"][basename] = (emitted, legacy)
        rolls_checked.append(rec)

    def _spy_pssdt(input_dir, solve_data_dir, *, provider,
                   full_dt_frames=None):
        real_pssdt(
            input_dir, solve_data_dir, provider=provider,
            full_dt_frames=full_dt_frames,
        )
        rec = {"took_full_path": full_dt_frames is not None, "pairs": {}}
        for basename, legacy_fn in _PSSDT_KEYS.items():
            emitted = provider.get(
                _provider_key(Path(solve_data_dir) / f"{basename}.csv")
            )
            legacy = legacy_fn(
                Path(input_dir), Path(solve_data_dir), provider=provider,
            )
            rec["pairs"][basename] = (emitted, legacy)
        rolls_checked.append(rec)

    monkeypatch.setattr(
        _period, "emit_pdtProcess__source__sink__dt_varCost_pair", _spy_varcost,
    )
    monkeypatch.setattr(
        _period, "emit_pssdt_varCost_filters", _spy_pssdt,
    )

    with tempfile.TemporaryDirectory() as t:
        wf = Path(t) / "default"
        run_chain_from_db(
            db, scenario_name=scenario, work_folder=wf,
            keep_solutions=True,
        )

    assert rolls_checked, "varCost emits were never called — no rolls run"

    # The full_dt compute-once path must have been exercised for every roll.
    assert all(r["took_full_path"] for r in rolls_checked), (
        "some rolls fell back to the legacy per-roll derivation — the "
        "full_dt cache was absent"
    )

    # Non-vacuous: the basic producer has rows on some roll.
    saw_rows = any(
        r["pairs"].get("pdtProcess__source__sink__dt_varCost",
                       (None, None))[0] is not None
        and r["pairs"]["pdtProcess__source__sink__dt_varCost"][0].height > 0
        for r in rolls_checked
        if "pdtProcess__source__sink__dt_varCost" in r["pairs"]
    )
    assert saw_rows, (
        f"pdtProcess__source__sink__dt_varCost was empty for every roll on "
        f"{scenario}"
    )

    # On the stochastic fixture, branch-period rows must reach the producer.
    if expect_branch_periods:
        seen_branch = any(
            emitted is not None and emitted.height > 0
            and any(
                "_" in p
                for p in emitted["period"].unique().to_list()
            )
            for r in rolls_checked
            for basename, (emitted, _legacy) in r["pairs"].items()
            if basename == "pdtProcess__source__sink__dt_varCost"
        )
        assert seen_branch, (
            f"{scenario}: no branch-period rows in emitted varCost — branch "
            f"expansion did not reach the varCost chain"
        )

    # Per-roll byte-identical parity: sliced-from-full == legacy per-roll.
    mismatches: list[str] = []
    for idx, r in enumerate(rolls_checked):
        for basename, (emitted, legacy) in r["pairs"].items():
            if emitted is None:
                mismatches.append(f"roll {idx} {basename}: emitted is None")
                continue
            if emitted.schema != legacy.schema:
                mismatches.append(
                    f"roll {idx} {basename}: schema {emitted.schema} != "
                    f"{legacy.schema}"
                )
                continue
            if not emitted.equals(legacy):
                mismatches.append(
                    f"roll {idx} {basename}: frame differs "
                    f"(shape {emitted.shape} vs {legacy.shape})"
                )

    assert not mismatches, (
        f"{scenario}: varCost sliced-from-full != legacy per-roll on "
        f"{len(mismatches)} (roll, family) pairs:\n  "
        + "\n  ".join(mismatches)
    )
