"""Per-roll parity gate for the pdtCommodity full_dt conversion.

Proof-of-concept of the "solve as picker" architecture (specs §9/§10):
``pdtCommodity`` is computed ONCE over the complete-solve timeline per
``complete_solve`` and sliced to each roll's window via
:func:`flextool.engine_polars._full_dt.slice_full_dt`, rather than
re-derived per roll.

This test drives the real 72-roll ``fullYear_roll`` cascade and, for EVERY
roll, asserts the sliced-from-full frame that ``emit_pdtCommodity`` writes
into the Provider is byte-identical (schema + every Utf8 value cell) to the
legacy per-roll :func:`derive_pdtCommodity` recomputed from the same roll's
``steps_in_use`` window.  It also asserts the full_dt path was actually
exercised (the cache frame was non-None) and that ``pdtCommodity`` is
non-empty on this fixture — otherwise the gate would be vacuous.

``fullYear_roll`` is backed by ``tests/fixtures/tests.json``, which defines
the commodities ``coal`` and ``anti_energy`` each carrying a ``price``
parameter, so the derived frame has rows.
"""
from __future__ import annotations

from pathlib import Path
import tempfile

from flextool.engine_polars import run_chain_from_db
from flextool.engine_polars import _emit_period_params
from flextool.engine_polars._emit_provider_io import _provider_key


_SCEN = "fullYear_roll"


def test_pdtCommodity_sliced_from_full_matches_legacy_every_roll(
    scenario_workdir, monkeypatch,
) -> None:
    db = scenario_workdir(_SCEN) / "tests.sqlite"

    # Per-roll diagnostics captured inside the patched emit.
    rolls_checked: list[dict] = []
    real_emit = _emit_period_params.emit_pdtCommodity

    def _spy_emit(input_dir, solve_data_dir, *, provider,
                  full_dt_frame=None):
        # Run the real (sliced-from-full when cached) emit first.
        real_emit(
            input_dir, solve_data_dir, provider=provider,
            full_dt_frame=full_dt_frame,
        )
        emitted = provider.get(
            _provider_key(Path(solve_data_dir) / "pdtCommodity.csv")
        )
        # Legacy per-roll recompute from the SAME roll window
        # (``steps_in_use``) carried in this Provider.
        legacy = _emit_period_params.derive_pdtCommodity(
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
        _emit_period_params, "emit_pdtCommodity", _spy_emit,
    )

    with tempfile.TemporaryDirectory() as t:
        wf = Path(t) / "default"
        run_chain_from_db(
            db, scenario_name=_SCEN, work_folder=wf,
            keep_solutions=True,
        )

    assert rolls_checked, "emit_pdtCommodity was never called — no rolls run"

    # The full_dt compute-once path must have been exercised for every roll
    # (the cache frame is computed once per complete_solve and reused).
    assert all(r["took_full_path"] for r in rolls_checked), (
        "some rolls fell back to the legacy per-roll derivation — the "
        "full_dt cache frame was absent"
    )

    # The gate must be non-vacuous: pdtCommodity has rows on this fixture.
    assert any(r["emitted"].height > 0 for r in rolls_checked), (
        "pdtCommodity was empty for every roll on fullYear_roll — parity "
        "gate would be vacuous; pick a commodity-priced fixture"
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
            mismatches.append(
                f"roll {idx}: frame differs (shape "
                f"{emitted.shape} vs {legacy.shape})"
            )

    assert not mismatches, (
        "pdtCommodity sliced-from-full != legacy per-roll on "
        f"{len(mismatches)}/{len(rolls_checked)} rolls:\n  "
        + "\n  ".join(mismatches)
    )
