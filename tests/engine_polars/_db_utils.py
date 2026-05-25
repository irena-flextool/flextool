"""Vendored subset of ``flextool/tests/db_utils.py``.

Only ``json_to_db`` (and its private dependency ``_unpack_value``) is
vendored — the rest of the upstream module (``db_to_json``,
``round_for_comparison``, the CLI) is unused by polar_high_spike and was
intentionally not copied.

Source of truth (upstream): ``flextool/tests/db_utils.py``.  Re-vendor
when that file's ``json_to_db`` / ``_unpack_value`` change.
"""
from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Any

from spinedb_api import DatabaseMapping, import_data


def _unpack_value(packed: Any) -> tuple[bytes | None, str | None]:
    """Reconstruct raw DB value from a [base64_string, type] pair."""
    b64, value_type = packed
    if b64 is None:
        return None, value_type
    return base64.b64decode(b64), value_type


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
