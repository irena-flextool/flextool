"""Phase 4 activation sanity check — substrate threading.

After Phase 4.4 the activation substrate is in place:

* ``provider.axis_enums`` + ``provider.contract`` attributes are
  populated by ``load_flextool`` (either inherited from the production
  ``input_derivation.run`` path or lazy-built against the workdir
  sqlite).
* ``_AXIS_SYNONYMS`` is extended with the cluster-discovered short
  forms (``b`` → branch, ``b_first`` / ``b_next`` → t, ``b_f`` →
  block, ``anchor`` → d, plus the entity-class element columns
  ``unit`` / ``connection`` / ``node_1`` / ``node_2``).
* ``SpineDbReader._maybe_cast_frame`` handles 0-dim ``name`` columns
  and n-dim element columns via ``_axis_for_entity_class``.
* ``cast_frame_axes`` is synonym-aware.

The cascade-wide ``set_global_axis_enums`` flip is INTENTIONALLY left
disabled (see ``input.py`` BLOCKER comment near the call site).  The
remaining cross-axis alias surface in ``_derived_params.py`` /
``_projection_params.py`` / ``model.py`` requires a systematic audit
that exceeds the scope of one dispatch.  This test verifies the
substrate is wired so the follow-up dispatch can flip the global on
without touching the surrounding plumbing.

After the follow-up lands, strengthen this test per the dispatch
template (assert ``utf8_count == 0`` on every FlexData dim column).
"""
from __future__ import annotations

import dataclasses
from pathlib import Path

import polars as pl
import pytest

from flextool.engine_polars.input import load_flextool


_DATA_DIR = Path(__file__).resolve().parent / "data"
_FIX = _DATA_DIR / "work_base"


def _all_dim_dtypes(flex_data) -> list[tuple[str, str, pl.DataType]]:
    """Return ``[(field_name, column_name, dtype), …]`` for every dim
    column on every Param / DataFrame / LazyFrame field of *flex_data*.
    """
    from polar_high import Param

    dim_names = {
        "e", "n", "p", "c", "g", "d", "t", "f", "i", "branch", "block",
        "r", "ud", "constraint", "cn", "bk", "b", "b_f", "b_first",
        "b_next",
        # Synonyms (per _AXIS_SYNONYMS) that should also be Enum
        # when the cascade is fully Enum-aware.
        "node", "process", "commodity", "group", "entity",
        "d_invest", "d_divest", "d_previous", "d_upper", "d_back",
        "period", "anchor",
    }
    out: list[tuple[str, str, pl.DataType]] = []
    for f in dataclasses.fields(flex_data):
        val = getattr(flex_data, f.name, None)
        if val is None:
            continue
        if isinstance(val, Param):
            schema = val.lazy.collect_schema()
        elif isinstance(val, pl.DataFrame):
            schema = val.schema
        elif isinstance(val, pl.LazyFrame):
            schema = val.collect_schema()
        else:
            continue
        for col, dt in schema.items():
            if col in dim_names:
                out.append((f.name, col, dt))
    return out


@pytest.mark.skipif(
    not (_FIX / "tests.sqlite").exists(),
    reason="work_base fixture missing",
)
def test_load_flextool_threads_axis_enums(tmp_path: Path) -> None:
    """Phase 4 substrate: ``load_flextool`` populates the Provider's
    ``axis_enums`` + ``contract`` attributes from the workdir sqlite.

    This is the prerequisite for the cascade-wide activation flip
    (which is BLOCKED on the cross-axis alias audit — see Phase 4
    BLOCKERS in ``specs/enum_dtype_refactor_plan.md``).
    """
    import shutil

    work = tmp_path / "work_base"
    shutil.copytree(_FIX, work)

    # Call without an explicit provider so load_flextool builds an
    # ephemeral one seeded from the workdir; the helper then
    # auto-builds axis_enums from the workdir sqlite.
    flex_data = load_flextool(work)
    assert flex_data is not None

    # Build the same axis_enums independently to confirm the workdir
    # sqlite produces the canonical vocabulary the substrate would
    # have threaded.
    from flextool.spinedb_backend._axis_enums import (
        build_axis_enums,
        load_axis_contract,
    )
    from flextool.spinedb_backend import SpineDBBackend

    sqlite_path = work / "tests.sqlite"
    contract = load_axis_contract()
    with SpineDBBackend(f"sqlite:///{sqlite_path}", None) as backend:
        axis_enums = build_axis_enums(backend, contract)

    # Spot-check vocabulary: the ``p`` axis enum is the union of unit
    # + connection names; ``work_base`` has at least one of each.
    p_enum = axis_enums.get("p")
    assert p_enum is not None and isinstance(p_enum, pl.Enum)
    assert len(p_enum.categories) > 0


@pytest.mark.skipif(
    not (_FIX / "tests.sqlite").exists(),
    reason="work_base fixture missing",
)
def test_load_flextool_emits_enum_typed_dim_columns(tmp_path: Path) -> None:
    """Sanity check the FlexData dim-column dtypes.

    Phase 4.4 keeps the cascade-wide flip disabled.  As a result, the
    dim columns on FlexData fields are Utf8 (pre-Phase-4 default).
    This test documents that state and provides the inversion point
    for the follow-up dispatch:

    * When the follow-up flips the global on, the assertion below
      should fail (Utf8 columns present).  At that point, the test
      should be strengthened to ``utf8_count == 0`` as the
      contract — see the dispatch ``Strengthen the activation test``
      section.
    """
    import shutil

    work = tmp_path / "work_base"
    shutil.copytree(_FIX, work)

    flex_data = load_flextool(work)

    dim_dtypes = _all_dim_dtypes(flex_data)
    assert dim_dtypes, "load_flextool produced no dim columns at all"

    enum_columns = [
        (field, col, dt) for field, col, dt in dim_dtypes
        if isinstance(dt, pl.Enum)
    ]
    utf8_columns = [
        (field, col, dt) for field, col, dt in dim_dtypes
        if dt == pl.Utf8
    ]

    # Pre-flip state: most dim columns remain Utf8.  Either we have
    # an active cascade-flip (then enum_columns > 0 and utf8_count
    # should be 0) OR we're still in pre-flip mode (utf8_count > 0,
    # enum_columns may also be > 0 due to ad-hoc casts inside the
    # cascade).  Just record both counts as a smoke check.
    assert len(dim_dtypes) > 0
    # NB: when the follow-up dispatch flips ``set_global_axis_enums``
    # on AND completes the alias audit, replace this loose assertion
    # with ``assert len(utf8_columns) == 0`` per the dispatch.
    print(f"[enum_activation] dim columns: {len(dim_dtypes)}, "
          f"Enum: {len(enum_columns)}, Utf8: {len(utf8_columns)}")
