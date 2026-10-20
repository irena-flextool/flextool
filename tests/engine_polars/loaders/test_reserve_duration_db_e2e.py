"""End-to-end DB-reader coverage for ``reserve_duration`` (#322).

Builds a fresh SQLite from the JSON fixture (``json_to_db`` → tmp DB;
invariant #3 — never a checked-in ``.sqlite``), writes a
``reserve_duration`` value onto the ``(primary, up, electricity)`` reserve
group under the existing ``reserve`` alternative, then reads it back
through the REAL :class:`SpineDbReader` cascade.  This exercises the
SpineDB ``parameter_explicit`` path that the in-memory loader unit tests
(``test_direct_params_delta4.py``) cannot.

Note: the ``network_coal_wind_reserve`` scenario's reserve providers are
generators (coal / wind), NOT storage-backed, so the new coupling
constraints are (correctly) inert there — the storage energy-adequacy
constraints and their solved values are proven in the synthetic engine
suite ``tests/engine_polars/synthetic/test_flex_toy_reserve_storage.py``.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from flextool.engine_polars._spinedb_reader import SpineDbReader
from flextool.engine_polars import _direct_params as dp
from polar_high import Param

TEST_DIR = Path(__file__).resolve().parents[2]
FIXTURES_DIR = TEST_DIR / "fixtures"


@pytest.mark.skipif(
    __import__("sys").platform == "win32",
    reason="sqlite handle can't be unlinked while open on Windows",
)
def test_reserve_duration_reads_from_real_spinedb(tmp_path: Path):
    from db_utils import json_to_db
    from spinedb_api import DatabaseMapping, import_data

    db_path = tmp_path / "reserve_duration.sqlite"
    url = json_to_db(FIXTURES_DIR / "tests.json", db_path)

    # Append the reserve_duration value under the pre-existing ``reserve``
    # alternative (already ranked into the network_coal_wind_reserve
    # scenario).  Append-only: no existing value is touched.
    with DatabaseMapping(url) as dbm:
        count, errors = import_data(
            dbm,
            # tests.json now ships at v71 WITH the reserve_duration
            # parameter_definition (regenerated in commit 30286318), so this
            # re-declaration is a harmless idempotent upsert — kept only so
            # the value import still resolves if the fixture is ever rebuilt
            # from an older export.
            parameter_definitions=[[
                "reserve__upDown__group", "reserve_duration"]],
            parameter_values=[[
                "reserve__upDown__group",
                ["primary", "up", "electricity"],
                "reserve_duration",
                0.75,
                "reserve",
            ]],
        )
        assert not errors, errors
        dbm.commit_session("add reserve_duration (#322 e2e)")

    reader = SpineDbReader(url, "network_coal_wind_reserve")
    p = dp.p_reserve_upDown_group_reserve_duration_from_source(reader)
    assert isinstance(p, Param)
    assert p.dims == ("r", "ud", "g")
    rows = p.frame.to_dicts()
    match = [r for r in rows
             if r["r"] == "primary" and r["ud"] == "up" and r["g"] == "electricity"]
    assert len(match) == 1, rows
    assert match[0]["value"] == pytest.approx(0.75)


@pytest.mark.skipif(
    __import__("sys").platform == "win32",
    reason="sqlite handle can't be unlinked while open on Windows",
)
def test_reserve_duration_absent_from_real_spinedb(tmp_path: Path):
    """Without any reserve_duration value the reader yields None (byte-
    identical LP — no storage coupling)."""
    from db_utils import json_to_db

    db_path = tmp_path / "no_reserve_duration.sqlite"
    url = json_to_db(FIXTURES_DIR / "tests.json", db_path)
    reader = SpineDbReader(url, "network_coal_wind_reserve")
    assert dp.p_reserve_upDown_group_reserve_duration_from_source(reader) is None
