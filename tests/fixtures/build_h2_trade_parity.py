"""Extract the ``scenario_test_6h_no_carrier_storage`` slice of the H2 trade
workshop database into a small JSON fixture suitable for committing.

The H2 trade workshop database (`projects/h2-imo/input_sources/H2_trade.sqlite`)
is the user's live working copy — regenerated from code each iteration and
~14 MB on disk, dominated by full-year hourly profile data for ~17
scenarios.  The python-preprocessing migration only needs the parity
baseline scenario ``scenario_test_6h_no_carrier_storage`` (24 hours, one
period, exercises the commodity ladder + indirect conversion).

Filtering rules
---------------
1. Only the alternative stack of the target scenario is kept:
   ``base`` + ``shared`` + ``solve_select_test_6h``.
2. All other alternatives, scenarios, and parameter values keyed on
   them are dropped.
3. ``timestep_duration`` of ``tl_plexos`` and every ``profile`` Map keyed
   by timestamp are truncated to the 24 timestamps actually used by
   ``ts_test_6h`` (``2050-01-01T00:00:00`` .. ``2050-01-01T23:00:00``).
   Other Map kinds (``period`` / ``constraint`` / ``tier`` indexes) are
   passed through untouched — they are already small.

The truncation of profile data is the load-bearing size reduction:
40 profiles × 8760 entries → 40 profiles × 24 entries shrinks the JSON
fixture from ~18 MB to well under 1 MB.

Schema integrity
----------------
``import_data`` from spinedb_api recreates entity classes, parameter
definitions, value lists, parameter types, parameter groups, entities,
entity_alternatives, parameter_values, alternatives, scenarios and
scenario_alternatives.  We export everything (Asterisk) for the schema-
shaped tables and only filter the alternative-keyed and scenario-keyed
tables — that way the rebuilt SQLite has exactly the same schema as the
source DB but only the data needed for the parity scenario.

Usage
-----
::

    python tests/fixtures/build_h2_trade_parity.py \
        [--source /path/to/H2_trade.sqlite] \
        [--out tests/fixtures/h2_trade_parity.json]

Defaults: source = ``projects/h2-imo/input_sources/H2_trade.sqlite``,
out = ``tests/fixtures/h2_trade_parity.json``.

To rebuild a SQLite DB from the JSON fixture::

    python tests/db_utils.py import \
        tests/fixtures/h2_trade_parity.json \
        /tmp/h2_trade_parity.sqlite
"""
from __future__ import annotations

import argparse
import base64
import json
import sys
from pathlib import Path
from typing import Any

from spinedb_api import DatabaseMapping, Map, export_data, from_database, to_database


HERE = Path(__file__).parent
REPO_ROOT = HERE.parent.parent
DEFAULT_SOURCE = REPO_ROOT / "projects" / "h2-imo" / "input_sources" / "H2_trade.sqlite"
DEFAULT_OUT = HERE / "h2_trade_parity.json"

SCENARIO = "scenario_test_6h_no_carrier_storage"

# The alternative stack of ``scenario_test_6h_no_carrier_storage``.
# ``base`` carries no parameter values in this DB but is included for
# parity with how the live DB is structured (every scenario implicitly
# layers above ``base``).
KEEP_ALTERNATIVES: frozenset[str] = frozenset({"base", "shared", "solve_select_test_6h"})

# The 24 timestamps activated by timeset ``ts_test_6h`` (start
# ``2050-01-01T00:00:00`` for 24h, hourly steps).  Profile and timeline
# Map values keyed on timestamps outside this window are stripped.
TS_START = "2050-01-01T00:00:00"
N_HOURS_KEPT = 24


def _pack_value(raw_bytes: bytes | None, value_type: str | None) -> list[Any]:
    """Match ``tests.db_utils._pack_value`` — base64 + type tag."""
    if raw_bytes is None:
        return [None, value_type]
    return [base64.b64encode(raw_bytes).decode("ascii"), value_type]


def _truncate_time_map(
    raw_bytes: bytes | None, value_type: str | None, kept_timestamps: list[str]
) -> tuple[bytes | None, str | None]:
    """If the value is a time-indexed Map, truncate it to ``kept_timestamps``.

    Returns ``(new_raw_bytes, new_type)``.  Non-Map / non-time-indexed
    values pass through unchanged.
    """
    if raw_bytes is None or value_type != "map":
        return raw_bytes, value_type
    parsed = from_database(raw_bytes, value_type)
    if not isinstance(parsed, Map):
        return raw_bytes, value_type
    if parsed.index_name != "time":
        return raw_bytes, value_type

    indexes = list(parsed.indexes)
    values = list(parsed.values)
    keep = set(kept_timestamps)
    new_indexes: list[str] = []
    new_values: list[Any] = []
    for idx, val in zip(indexes, values):
        if str(idx) in keep:
            new_indexes.append(str(idx))
            new_values.append(val)
    if not new_indexes:
        # Nothing to keep → leave the raw bytes alone (defensive — the 24h
        # window contains every data-bearing timestamp on this DB).
        return raw_bytes, value_type
    truncated = Map(
        indexes=new_indexes,
        values=new_values,
        index_type=parsed.index_type,
        index_name=parsed.index_name,
    )
    new_raw, new_type = to_database(truncated)
    return new_raw, new_type


def _resolve_kept_timestamps(db: DatabaseMapping) -> list[str]:
    """Pull the 24-h window from ``tl_plexos.timestep_duration``.

    The window is the first ``N_HOURS_KEPT`` timestamps starting at
    ``TS_START``.  All ``profile`` and ``timeline`` Map entries keyed
    on timestamps outside this window are dropped during export.
    """
    pvs = list(
        db.find_parameter_values(
            entity_class_name="timeline",
            entity_byname=("tl_plexos",),
            parameter_definition_name="timestep_duration",
        )
    )
    if not pvs:
        raise RuntimeError("tl_plexos timestep_duration not found in source DB")
    pv = next((p for p in pvs if p["alternative_name"] in KEEP_ALTERNATIVES), pvs[0])
    parsed = from_database(pv["value"], pv["type"])
    if not isinstance(parsed, Map):
        raise RuntimeError("tl_plexos timestep_duration is not a Map")
    indexes = [str(i) for i in parsed.indexes]
    if TS_START not in indexes:
        raise RuntimeError(f"timestamp {TS_START!r} not found in tl_plexos timeline")
    start_i = indexes.index(TS_START)
    return indexes[start_i : start_i + N_HOURS_KEPT]


def build_payload(source_url: str) -> dict[str, list]:
    """Export the filtered scenario slice as an ``import_data``-compatible
    dict.

    Strategy: use ``export_data`` with ``Asterisk`` for every ID list
    EXCEPT the alternative-keyed / scenario-keyed ones, which we pin to
    just the data needed for ``scenario_test_6h_no_carrier_storage``.
    """
    with DatabaseMapping(source_url) as db:
        kept_ts = _resolve_kept_timestamps(db)

        keep_alt_ids = [
            a["id"] for a in db.find_alternatives() if a["name"] in KEEP_ALTERNATIVES
        ]
        keep_scenario_ids = [
            s["id"] for s in db.find_scenarios() if s["name"] == SCENARIO
        ]
        keep_scenario_alt_ids = [
            sa["id"]
            for sa in db.find_scenario_alternatives()
            if sa["scenario_name"] == SCENARIO
        ]
        keep_pv_ids = [
            p["id"]
            for p in db.find_parameter_values()
            if p["alternative_name"] in KEEP_ALTERNATIVES
        ]
        keep_ea_ids = [
            e["id"]
            for e in db.find_entity_alternatives()
            if e["alternative_name"] in KEEP_ALTERNATIVES
        ]

        def packer(raw_bytes: bytes | None, value_type: str | None) -> list[Any]:
            truncated_bytes, truncated_type = _truncate_time_map(
                raw_bytes, value_type, kept_ts
            )
            return _pack_value(truncated_bytes, truncated_type)

        data = export_data(
            db,
            alternative_ids=keep_alt_ids,
            scenario_ids=keep_scenario_ids,
            scenario_alternative_ids=keep_scenario_alt_ids,
            parameter_value_ids=keep_pv_ids,
            entity_alternative_ids=keep_ea_ids,
            parse_value=packer,
        )

    # Remove unused entities and entity_alternatives that reference
    # alternatives we are no longer carrying — export_data by default
    # leaves all entities (which is what we want, since dropping a
    # solve entity that was only referenced by an unused
    # solve_select_* alt would still result in the entity being
    # required by FlexTool's schema-level checks elsewhere).  But
    # entity_alternatives is already pre-filtered above by ID.
    return data


def _make_unpacker():
    """Build the ``unparse_value`` callable that ``import_data`` consumes."""

    def _unpack_value(packed: Any) -> tuple[bytes | None, str | None]:
        b64, value_type = packed
        if b64 is None:
            return None, value_type
        return base64.b64decode(b64), value_type

    return _unpack_value


def write_json_fixture(source_url: str, out_path: Path) -> int:
    """Write the JSON fixture and return the byte size on disk."""
    data = build_payload(source_url)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as fh:
        # Stable indent + sort_keys=False (preserve list order so
        # entities and pvs ordering matches export_data's insertion
        # order; the resulting JSON is still byte-stable across runs).
        json.dump(data, fh, indent=2)
    return out_path.stat().st_size


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Extract scenario_test_6h_no_carrier_storage from H2_trade.sqlite "
            "into a JSON fixture for parity testing."
        )
    )
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args(argv)

    if not args.source.exists():
        print(f"ERROR: source DB not found: {args.source}", file=sys.stderr)
        return 2
    source_url = f"sqlite:///{args.source.resolve()}"
    size = write_json_fixture(source_url, args.out)
    print(f"Wrote {args.out}  ({size:,} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
