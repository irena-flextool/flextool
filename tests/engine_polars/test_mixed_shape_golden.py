"""Permanent golden snapshot (spec fix_map_reading.md §4).

Guards the "unchanged by construction" claim of the per-row period/time
placement: for every parameter OUTSIDE ``PERIOD_TIME_PARAMS``, plus every
``PERIOD_TIME_PARAMS`` parameter whose rows are all one shape, the
``parameter()`` / ``parameter_explicit()`` frames must be byte-identical
whether or not the new placement path runs.

The baseline is produced by the ``_PERIOD_TIME_PLACEMENT_ENABLED`` test
seam (toggled to ``False`` the legacy positional unroll runs).  Comparing
the new path against the legacy path over the real fixture corpus (built
from JSON per invariant #3 — no checked-in ``.sqlite``) proves the hot
path is unchanged for single-shape / non-registry parameters.
"""
from __future__ import annotations

import pytest

import flextool.engine_polars._spinedb_reader as sr
from flextool.engine_polars import SpineDbReader

# Corpus entries: (db_url fixture name, scenario).  Built from the shared
# session JSON-backed DB fixtures in tests/conftest.py.
_CORPUS = [
    ("test_db_url", "base"),
    ("test_db_url", "coal_co2_price"),
    ("test_db_url", "fullYear"),
    ("test_db_url", "multi_year"),
    ("lh2_db_url", "lh2_three_region"),
]


def _enumerate_params(reader: SpineDbReader):
    """Yield every (entity_class, parameter_name) the reader knows."""
    for (cls_id, pname) in reader._pdef_by_class_name:
        cls_name = reader._class_id_to_name.get(cls_id)
        if cls_name is None:
            continue
        yield cls_name, pname


def _capture(reader, cls_name, pname):
    return (
        reader.parameter(cls_name, pname),
        reader.parameter_explicit(cls_name, pname),
    )


@pytest.mark.parametrize("db_fixture_name,scenario", _CORPUS)
def test_golden_single_shape_unchanged(request, db_fixture_name, scenario):
    url = request.getfixturevalue(db_fixture_name)
    # Build WITHOUT enums so we compare the raw columnar unroll output of
    # the new placement path against the legacy positional path, isolating
    # the _unroll_rows change from the cast-on-emit layer.
    reader = SpineDbReader(url, scenario)

    compared = 0
    skipped_mixed: list[tuple[str, str]] = []
    original = sr._PERIOD_TIME_PLACEMENT_ENABLED
    try:
        for cls_name, pname in _enumerate_params(reader):
            # Eligibility: non-PERIOD_TIME (variants == empty set) or a
            # PERIOD_TIME param with at most one classified shape variant.
            try:
                variants = reader.parameter_shape_variants(cls_name, pname)
            except KeyError:
                continue
            if len(variants) > 1:
                skipped_mixed.append((cls_name, pname))
                continue

            sr._PERIOD_TIME_PLACEMENT_ENABLED = True
            new_p, new_pe = _capture(reader, cls_name, pname)
            sr._PERIOD_TIME_PLACEMENT_ENABLED = False
            old_p, old_pe = _capture(reader, cls_name, pname)

            assert new_p.equals(old_p), (
                f"parameter({cls_name!r}, {pname!r}) changed by per-row "
                f"placement (byte-parity violation)")
            assert new_pe.equals(old_pe), (
                f"parameter_explicit({cls_name!r}, {pname!r}) changed by "
                f"per-row placement (byte-parity violation)")
            compared += 1
    finally:
        sr._PERIOD_TIME_PLACEMENT_ENABLED = original

    # Sanity: the corpus actually exercised the comparison.
    assert compared > 0
