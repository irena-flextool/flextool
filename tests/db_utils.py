"""Utilities for managing the FlexTool test database fixture.

Developer workflow
------------------
Edit the Spine DB in Spine Toolbox, then commit changes to git:

    python test/db_utils.py export test/tests.sqlite test/fixtures/tests.json

To (re-)create the SQLite DB from the JSON fixture (e.g. after pulling changes):

    python test/db_utils.py import test/fixtures/tests.json test/tests.sqlite

The SQLite DB is NOT committed to git. The JSON fixture is.
The test suite calls json_to_db() automatically at session start via conftest.py.
"""
from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Any

import pandas as pd
from spinedb_api import DatabaseMapping, export_data, import_data


def _pack_value(raw_bytes: bytes | None, value_type: str | None) -> list[Any]:
    """Serialize a raw DB value to a JSON-safe [base64_string, type] pair.

    spinedb_api stores values as (bytes, type_str). The bytes are typically
    UTF-8 JSON but may be binary for some types, so base64 is used for safety.
    raw_bytes is None for parameter definitions with no default value.
    """
    if raw_bytes is None:
        return [None, value_type]
    return [base64.b64encode(raw_bytes).decode("ascii"), value_type]


def _unpack_value(packed: Any) -> tuple[bytes | None, str | None]:
    """Reconstruct raw DB value from a [base64_string, type] pair."""
    b64, value_type = packed
    if b64 is None:
        return None, value_type
    return base64.b64decode(b64), value_type


def db_to_json(db_path: Path, json_path: Path) -> None:
    """Export a Spine SQLite DB to a JSON file for version control."""
    url = f"sqlite:///{db_path.resolve()}"
    with DatabaseMapping(url) as db_map:
        data = export_data(db_map, parse_value=_pack_value)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(json_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Exported {len(data.get('parameter_values', []))} parameter values → {json_path}")


def json_to_db(json_path: Path, db_path: Path) -> str:
    """Import a JSON fixture into a new SQLite DB. Returns the sqlite:/// URL."""
    with open(json_path) as f:
        data = json.load(f)
    url = f"sqlite:///{db_path.resolve()}"
    with DatabaseMapping(url, create=True) as db_map:
        count, errors = import_data(db_map, unparse_value=_unpack_value, **data)
        if errors:
            raise RuntimeError(f"Import errors: {errors[:5]}")
        db_map.commit_session("Import test fixture")
    print(f"Imported {count} items → {db_path}")
    return url


def round_for_comparison(df: pd.DataFrame) -> pd.DataFrame:
    """Round numeric columns for stable comparison across solver versions.

    Rounding to 4 decimals (absolute ±5e-5) is layered with the caller's
    ``pd.testing.assert_frame_equal(..., rtol=1e-4)``.  The combined
    tolerance is comfortably looser than HiGHS / glpsol solver noise on
    the v3.32 frozen goldens (e.g. the regen agent observed one cell at
    0.0002% relative drift, well within ±5e-5 absolute + rtol=1e-4).

    TODO: tighten once the v3.32.0 gate is at 65/65.  The current setting
    can hide a real engine bug that perturbs a small-value column by
    ~0.001 (round(4) absorbs anything below 5e-5), and at large costs
    the rtol=1e-4 gate is ±100 EUR on a 1M EUR figure.  Once cascade ≡
    v3.32.0 is established, drop the round(4) and tighten rtol to 1e-7
    or 1e-8 (HiGHS feasibility tolerance is 1e-8).  This requires
    regenerating goldens against current HiGHS via
    ``pytest tests/test_scenarios.py --regenerate <scenario>`` so the
    new tighter baseline absorbs solver-version-noise without masking
    real engine bugs.  See project memory
    ``feedback_tighten_test_tolerance_after_parity.md``.

    The previous docstring claimed "one order of magnitude coarser than
    1e-8" — that's mathematically wrong (1e-8 vs 1e-4 is four orders
    coarser).  Corrected here.
    """
    numeric_cols = df.select_dtypes(include="number").columns
    df = df.copy()
    df[numeric_cols] = df[numeric_cols].round(4)
    return df


if __name__ == "__main__":
    import sys

    usage = "Usage: python db_utils.py export <db.sqlite> <out.json>\n" \
            "       python db_utils.py import <in.json> <db.sqlite>"

    if len(sys.argv) != 4:
        print(usage)
        sys.exit(1)

    command, arg1, arg2 = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3])
    if command == "export":
        db_to_json(arg1, arg2)
    elif command == "import":
        json_to_db(arg1, arg2)
    else:
        print(usage)
        sys.exit(1)
