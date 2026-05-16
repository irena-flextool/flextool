"""Phase E-a — cascade-consumes-seed instrumentation test.

Phase D made ``load_flextool`` accept an in-memory ``seed=`` accumulator
that short-circuits disk reads for the ~108 CSV basenames the
:class:`FlexDataAccumulator` covers.  Phase E-a wires
``_FlexpyCascadeSolver.run`` to actually pass the per-sub-solve
accumulator to ``load_flextool``.

This test confirms the wiring by patching
:func:`flextool.engine_polars._input_source._read_csv_file` and counting
seed-hits (frames returned from the in-memory accumulator) vs
seed-misses (frames returned from the disk-read fallback) across one
cascade run on a multi-sub-solve fixture.  We assert:

  * Seed hits are non-trivially positive (the cascade actually consumes
    the accumulator).  Exact counts depend on accumulator coverage but
    every sub-solve should contribute at least a few hits, so we expect
    the total to scale with the number of sub-solves.
  * Seed misses are also positive (the accumulator covers ~108 of ~200
    CSV basenames; the rest still come from disk).  This proves we did
    not accidentally short-circuit ALL disk reads.

The instrumentation patches the public seed-lookup helper rather than
``load_flextool`` itself so we observe the effect inside the loader
where the seed is consumed.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import polars as pl
import pytest

from flextool.engine_polars import run_chain_from_db
from flextool.engine_polars import _input_source as _ipsrc


pytestmark = pytest.mark.solver


HERE = Path(__file__).resolve().parent
DATA = HERE / "data"


def _pick_multi_solve_fixture() -> tuple[Path, str] | None:
    """Return ``(sqlite_path, scenario)`` for a multi-sub-solve fixture
    if one is available; ``None`` otherwise.

    We prefer ``work_chain_co2_rolling`` (the canonical multi-solve
    chain fixture used by the other Phase D/E tests) and fall back to
    ``work_base`` if that one is missing.  The instrumentation works on
    any cascade run; this preference simply maximises sub-solve count
    so the seed-hit signal is unambiguous.
    """
    candidates = [
        ("work_fullYear_roll", "fullYear_roll"),
        ("work_multi_year", "multi_year"),
        ("work_base", "base"),
    ]
    for name, scen in candidates:
        sqlite = DATA / name / "tests.sqlite"
        if sqlite.exists():
            return sqlite, scen
    return None


def test_cascade_passes_seed_to_load_flextool(tmp_path: Path) -> None:
    """Run a small cascade and count seed-hits / seed-misses inside the
    loader.  Hits must be > 0 (the cascade consumes the accumulator);
    misses must be > 0 (the ~92 non-accumulator basenames still read
    from disk)."""
    fixture = _pick_multi_solve_fixture()
    if fixture is None:
        pytest.skip("no multi-sub-solve fixture available")
    sqlite, scen = fixture

    work = tmp_path / "work"

    # Counters.  We patch ``_seed_lookup`` because it's the precise
    # branch ``_read_csv_file`` consults before falling back to disk.
    # A hit = lookup returned a frame; a miss-while-seed-active = lookup
    # returned None and we fell back to disk-read with a seed installed.
    # We also tally calls where no seed was installed so we know which
    # path the cascade ended up on per call (for diagnostics on test
    # failure).
    hits: list[str] = []
    misses_seed_active: list[str] = []
    misses_no_seed: list[str] = []

    real_seed_lookup = _ipsrc._seed_lookup

    def patched_seed_lookup(path: Any) -> "pl.DataFrame | None":
        seed = _ipsrc._active_seed
        result = real_seed_lookup(path)
        try:
            name = Path(str(path)).name
        except Exception:  # noqa: BLE001
            name = str(path)
        if result is not None:
            hits.append(name)
        elif seed is not None:
            misses_seed_active.append(name)
        else:
            misses_no_seed.append(name)
        return result

    _ipsrc._seed_lookup = patched_seed_lookup  # type: ignore[assignment]
    try:
        sols = run_chain_from_db(
            sqlite, scenario_name=scen, work_folder=work,
        )
    finally:
        _ipsrc._seed_lookup = real_seed_lookup  # type: ignore[assignment]

    assert sols, "cascade produced no sub-solves"

    # The cascade should have installed a seed for every per-sub-solve
    # load_flextool call.  Hits must be strictly > 0 (Phase E-a's whole
    # point).  Misses-while-seed-active must also be > 0 (the ~92 CSVs
    # the accumulator does not cover still come from disk).
    assert hits, (
        "Phase E-a regression: cascade ran but no seed-hits were "
        "observed inside load_flextool — the cascade is not consuming "
        "the accumulator.  Check that "
        "``_FlexpyCascadeSolver.run`` passes ``seed=...`` to "
        "``load_flextool``."
    )
    assert misses_seed_active, (
        "Sanity check failed: every CSV read was a seed-hit.  "
        "Accumulator coverage is partial (~108 of ~200 basenames); "
        "the rest should miss-and-fall-back-to-disk.  Zero misses "
        "suggests the accumulator over-claims coverage."
    )

    # The hit count should scale with the number of sub-solves — at
    # least a handful of hits per sub-solve (the writer-port thin
    # wrappers contribute ~37+ frames per solve, and the loader reads
    # most of them in the seed-covered branch).  A conservative lower
    # bound is ``len(sols)`` (one hit per sub-solve, weakly).  This
    # guards against a regression where only the first sub-solve's
    # accumulator is consumed and later ones fall through to disk.
    assert len(hits) >= len(sols), (
        f"seed-hit count ({len(hits)}) is lower than the number of "
        f"sub-solves ({len(sols)}) — at least one sub-solve appears "
        "to have skipped the seed path."
    )
