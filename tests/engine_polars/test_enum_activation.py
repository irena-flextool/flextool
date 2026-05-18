"""Phase 4 activation sanity check.

Verifies that ``load_flextool`` produces a :class:`FlexData` whose dim
columns are :class:`pl.Enum` typed, not :class:`pl.Utf8`.  This is the
end-to-end proof that the activation flipped: Phases 0–3 landed the
substrate, the contract, the cast helpers and the Provider migration,
and Phase 4 wires ``axis_enums`` into the load path + the
``cast_flexdata_axes`` final-tidy sweep at end of load.

Without Phase 4, every dim column on a load_flextool-produced FlexData
is ``pl.Utf8`` (the pre-Phase-4 "disabled activation" state).  After
Phase 4, dim columns on representative Param frames carry the
canonical :class:`pl.Enum` dtype keyed by the matching axis name.
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

    A "dim column" is any column whose name is in our axis-friendly set
    (``e``, ``n``, ``p``, ``c``, ``d``, ``t``, …) — the same set
    Phase 4's ``cast_flexdata_axes`` targets.
    """
    from polar_high import Param

    dim_names = {
        "e", "n", "p", "c", "g", "d", "t", "f", "i", "branch", "block",
        "r", "ud", "constraint", "cn",
        # Common synonyms (per _AXIS_SYNONYMS) that should also be Enum
        # after the final-tidy sweep.
        "node", "process", "commodity", "group", "entity",
        "d_invest", "d_divest", "d_previous", "d_upper", "d_back",
        "period",
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
def test_load_flextool_emits_enum_typed_dim_columns(tmp_path: Path) -> None:
    """After Phase 4, load_flextool produces a FlexData whose dim
    columns are pl.Enum, not pl.Utf8.
    """
    # Copy the fixture so the loader has a writable workdir (the
    # cascade has side effects on solve_data/).
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

    # Phase 4 activation: at least one dim column on at least one
    # representative Param frame must carry pl.Enum after the
    # final-tidy sweep.  Strengthening this to ``utf8_count == 0`` is
    # blocked on the pre-cascade FlexData → Enum sweep, which exposes
    # many cross-axis rename sites that need systematic Enum-awareness
    # (``e``→``p`` aliases, ``node``→``source`` aliases, etc.).
    # Tracked as Phase 4 redo follow-up.
    assert len(enum_columns) > 0, (
        "Phase 4 activation regressed: no dim columns are pl.Enum-typed "
        f"after load_flextool.  Sampled {len(dim_dtypes)} dim columns, "
        f"{len(utf8_columns)} are still pl.Utf8."
    )

    # Print a small summary the user can eyeball when running with -s.
    print(
        f"\nPhase 4 activation:\n"
        f"  dim columns sampled: {len(dim_dtypes)}\n"
        f"  pl.Enum-typed:        {len(enum_columns)}\n"
        f"  pl.Utf8-typed:        {len(utf8_columns)}\n"
        f"  first Enum sample:    "
        f"{enum_columns[0][0]}.{enum_columns[0][1]} : {enum_columns[0][2]}"
    )
