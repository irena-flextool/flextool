"""One-shot pre-pass for Batch F Option A: re-add is_enabled on three classes.

Third-attempt counterpart to ``_batch_f_prepass.py`` (which used the
legacy parameter name ``is_active`` and was abandoned after spinedb_api's
compatibility shim was discovered) and ``_batch_f_prepass_is_enabled.py``
(which used ``is_enabled`` but lacked the engine-side wire-up that
replaces spinedb_api's scenario_filter for the three affected classes).
This Option A variant ships the same JSON-side rewrites — the rewrites
themselves are independent of the engine wire-up; only the engine code
needed an additional change.

The Spine migration loop in :mod:`flextool.update_flextool.db_migration`
runs ``next_version <= new_version``; if a JSON is already at the target
version (v56) the loop is a no-op.  Every in-repo canonical/test-fixture
JSON sits at v56, so we need a one-shot pre-pass that applies the
equivalent of :func:`_migrate_v56_reactivate_is_enabled_parameter`
directly to the JSON shape.

This file is throwaway: it lives under ``_audit_reports/`` (already
git-ignored as a working dir) and is invoked once from this commit's
shell session.  After verify-trio is green it serves no further purpose.

JSON shape recap (per the export_database/db_to_json format):
- ``entity_classes``: list of ``[name, dimension_names, description, ?, active_by_default]``
- ``entities``: list of ``[class_name, byname, description]``
- ``entity_alternatives``: list of ``[class_name, byname_list, alt_name, active_bool]``
- ``parameter_definitions``: list of ``[class_name, name, default_value, value_list, description, group]``
- ``parameter_types``: list of ``[class_name, name, type_str, rank]``
- ``parameter_values``: list of ``[class_name, byname_list_or_str, param_name, [b64, type], alt_name]``
- ``parameter_value_lists``: list of ``[list_name, value]``

The canonical_databases JSONs use UTF-8 ``parameter_values`` (the
non-base64 form via ``initialize_database`` / ``export_database``); the
``tests/fixtures/*.json`` files use base64-packed values via
``tests.db_utils.db_to_json``.  Both forms share the same
``["bytes", "type"]`` tuple structure, only the encoding of ``bytes``
differs — so the same code that drops/adds parameter_values works for
both as long as we use the format that the surrounding file already
uses.
"""

from __future__ import annotations

import base64
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

CANONICAL_DIR = REPO_ROOT / "flextool" / "schemas" / "canonical_databases"
FIXTURES_DIR = REPO_ROOT / "tests" / "fixtures"

# Class-specific migration logic (mirrors the helper in db_migration.py).
CLASSES_WITH_FLIP = (
    "reserve__upDown__unit__node",
    "reserve__upDown__connection__node",
)
CLASS_NO_FLIP = "constraint"
AFFECTED_CLASSES = (*CLASSES_WITH_FLIP, CLASS_NO_FLIP)
GROUP_BY_CLASS = {
    "constraint": "constraint",
    "reserve__upDown__unit__node": "reserve",
    "reserve__upDown__connection__node": "reserve",
}
DESCRIPTION = (
    "Whether the entity is enabled. Set to 'no' to disable "
    "without deleting the entity. Constant."
)

CANONICAL_JSONS = sorted(CANONICAL_DIR.glob("*.json"))
FIXTURE_JSONS = sorted(FIXTURES_DIR.glob("*.json"))


def _encode_str(s: str, use_base64: bool) -> str:
    """Encode a JSON-encoded scalar string for storage."""
    raw = json.dumps(s)
    if use_base64:
        return base64.b64encode(raw.encode("utf-8")).decode("ascii")
    return raw


def _detect_base64_encoding(d: dict) -> bool:
    """Detect whether parameter_values use base64-packed bytes."""
    for pv in d.get("parameter_values", []):
        val = pv[3] if len(pv) > 3 else None
        if (
            isinstance(val, list)
            and len(val) == 2
            and val[1] == "str"
            and isinstance(val[0], str)
            and val[0]
        ):
            return not val[0].startswith('"')
    for pd in d.get("parameter_definitions", []):
        default = pd[2]
        if (
            isinstance(default, list)
            and len(default) == 2
            and default[1] == "str"
            and isinstance(default[0], str)
            and default[0]
        ):
            return not default[0].startswith('"')
    return False


def _sort_parameter_definitions(pds: list) -> list:
    return sorted(pds, key=lambda r: (r[0], r[1]))


def _sort_parameter_types(pts: list) -> list:
    return sorted(pts, key=lambda r: (r[0], r[1], r[2], r[3]))


def _sort_parameter_value_lists(pvls: list) -> list:
    return sorted(pvls, key=lambda r: (r[0], r[1]))


def process_file(path: Path) -> tuple[int, int, int, int, int, int]:
    """Apply the Batch F pre-pass to one JSON file.

    Returns ``(ea_dropped, pv_added, base_backfilled, ec_flipped,
    pd_dropped_stale, pd_added)``.
    """
    with open(path) as f:
        d = json.load(f)

    use_b64 = _detect_base64_encoding(d)
    yes_enc = _encode_str("yes", use_b64)
    no_enc = _encode_str("no", use_b64)

    # parameter_value rows use a STRING byname for 1-D classes
    # (constraint) and a LIST byname for multi-D classes (reserve).
    # entity_alternative rows always use the list form, so we collapse
    # 1-element bynames back to string before writing parameter_values
    # for ``constraint``.
    def _byname_for_pv(cls: str, byname_list: list) -> object:
        if cls == "constraint" and len(byname_list) == 1:
            return byname_list[0]
        return list(byname_list)

    eas = d.get("entity_alternatives", [])
    new_eas = []
    base_seen: set[tuple[str, tuple[str, ...]]] = set()
    pv_rows_to_add: list[list] = []
    ea_dropped = 0
    for ea in eas:
        cls = ea[0]
        if cls not in AFFECTED_CLASSES:
            new_eas.append(ea)
            continue
        byname_list = list(ea[1])
        alt = ea[2]
        active = bool(ea[3])
        pv_rows_to_add.append([
            cls,
            _byname_for_pv(cls, byname_list),
            "is_enabled",
            [yes_enc if active else no_enc, "str"],
            alt,
        ])
        if cls in CLASSES_WITH_FLIP and alt == "Base":
            base_seen.add((cls, tuple(byname_list)))
        ea_dropped += 1
    d["entity_alternatives"] = new_eas

    # Backfill Base is_enabled="no" for two reserve classes' entities
    # lacking a Base entity_alternative entry.
    backfilled = 0
    for ent in d.get("entities", []):
        cls = ent[0]
        if cls not in CLASSES_WITH_FLIP:
            continue
        name = ent[1]
        if isinstance(name, list):
            byname_list = list(name)
        else:
            byname_list = [name]
        if (cls, tuple(byname_list)) in base_seen:
            continue
        pv_rows_to_add.append([
            cls,
            _byname_for_pv(cls, byname_list),
            "is_enabled",
            [no_enc, "str"],
            "Base",
        ])
        backfilled += 1

    # Append new parameter_values; leave original order in place
    # (export pipelines preserve spinedb_api insertion order).
    pvs = d.get("parameter_values", [])
    pvs.extend(pv_rows_to_add)
    d["parameter_values"] = pvs
    pv_added = len(pv_rows_to_add)

    # Flip active_by_default on the two reserve classes.
    ec_flipped = 0
    ecs = d.get("entity_classes", [])
    for ec in ecs:
        if ec[0] in CLASSES_WITH_FLIP and len(ec) >= 5 and ec[4] is False:
            ec[4] = True
            ec_flipped += 1

    # Drop the orphan ``is_active`` value list entry.
    pvls = d.get("parameter_value_lists", [])
    new_pvls = [r for r in pvls if r[0] != "is_active"]
    d["parameter_value_lists"] = _sort_parameter_value_lists(new_pvls)

    # Drop any stale ``is_active`` or ``is_enabled`` parameter_definitions
    # (idempotency: a re-run on already-migrated JSONs should re-emit
    # the canonical form), then add the canonical new ones for the
    # three affected classes.
    pds = d.get("parameter_definitions", [])
    pds_kept = [
        r for r in pds
        if r[1] not in ("is_active", "is_enabled")
    ]
    pd_dropped_stale = len(pds) - len(pds_kept)

    # Detect default_value packing format by scanning sibling
    # str-typed defaults in the file.
    pd_default_form = "plain"
    for r in pds_kept:
        default = r[2]
        if (
            isinstance(default, list)
            and len(default) == 2
            and default[1] == "str"
            and isinstance(default[0], str)
        ):
            if default[0].startswith('"'):
                pd_default_form = "json_string"
            else:
                pd_default_form = "base64"
            break
    if pd_default_form == "base64":
        yes_default = [
            base64.b64encode(b'"yes"').decode("ascii"),
            "str",
        ]
    elif pd_default_form == "json_string":
        yes_default = ['"yes"', "str"]
    else:
        yes_default = "yes"
    new_pds = []
    for cls in AFFECTED_CLASSES:
        new_pds.append([
            cls,
            "is_enabled",
            yes_default,
            "yes_no",
            DESCRIPTION,
            GROUP_BY_CLASS[cls],
        ])
    pds_kept.extend(new_pds)
    d["parameter_definitions"] = _sort_parameter_definitions(pds_kept)
    pd_added = len(new_pds)

    # Drop stale ``is_active``/``is_enabled`` parameter_types entries
    # (idempotency), then add canonical str-scalar rows for each
    # affected class.
    pts = d.get("parameter_types", [])
    pts = [r for r in pts if r[1] not in ("is_active", "is_enabled")]
    for cls in AFFECTED_CLASSES:
        pts.append([cls, "is_enabled", "str", 0])
    d["parameter_types"] = _sort_parameter_types(pts)

    is_canonical = (CANONICAL_DIR in path.parents)
    if is_canonical:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(d, f, sort_keys=True, indent=2, ensure_ascii=False)
            f.write("\n")
    else:
        with open(path, "w") as f:
            json.dump(d, f, indent=2)

    return (
        ea_dropped, pv_added, backfilled, ec_flipped,
        pd_dropped_stale, pd_added,
    )


def main() -> None:
    for jsons, label in (
        (CANONICAL_JSONS, "canonical"),
        (FIXTURE_JSONS, "fixture"),
    ):
        for p in jsons:
            stats = process_file(p)
            print(f"[{label}] {p.name}: "
                  f"ea_dropped={stats[0]} pv_added={stats[1]} "
                  f"base_backfilled={stats[2]} ec_flipped={stats[3]} "
                  f"pd_stale_dropped={stats[4]} pd_added={stats[5]}")


if __name__ == "__main__":
    main()
