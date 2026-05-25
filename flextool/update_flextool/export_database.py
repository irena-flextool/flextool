"""Export a FlexTool Spine DB to deterministic JSON.

Companion to :func:`flextool.update_flextool.initialize_database`.  The
round-trip ``export_database -> initialize_database -> export_database``
is bit-identical, which lets canonical example/template databases live
in git as JSON instead of binary ``.sqlite`` files.

Two callables sit at the heart of the round-trip:

``keep_serialized_parse``
    Passed as ``parse_value`` to :func:`spinedb_api.export_data`.  Instead
    of returning a parsed Python object (``Array``, ``Map``, ``DateTime``,
    ...), it returns the raw ``(value_json_str, type_str)`` pair that
    ``spinedb_api`` already stores in the DB.  That keeps every shape
    JSON-serialisable and avoids lossy round-trips through parsed
    object identity (e.g. ``Map`` repr ordering, float printing).

``keep_serialized_unparse``
    Passed as ``unparse_value`` to :func:`spinedb_api.import_data` by
    :func:`flextool.update_flextool.initialize_database` when the JSON
    came from this exporter.  Accepts either the ``(str, type)`` pair
    (preserved from JSON) or a raw scalar (master-template style); in
    the former case it bypasses :func:`spinedb_api.to_database` and
    writes the bytes straight back.
"""

import argparse
import json
from typing import Any

from spinedb_api import DatabaseMapping, export_data
from spinedb_api.parameter_value import to_database


def keep_serialized_parse(value: bytes | None, type_: str | None) -> Any:
    """``parse_value`` callback that preserves the raw DB form.

    Returns ``None`` for ``None`` and for the JSON-``null`` blob (those
    are equivalent on the import side and collapsing them keeps the
    exported JSON tidy and round-trip-stable).  Otherwise returns a
    ``[json_string, type_string]`` two-list — a list, not a tuple,
    so it survives ``json.dumps``/``json.loads`` unchanged.
    """
    if value is None:
        return None
    if value == b"null" and type_ is None:
        return None
    return [value.decode("utf-8"), type_]


def keep_serialized_unparse(value: Any) -> tuple[bytes | None, str | None]:
    """``unparse_value`` callback paired with :func:`keep_serialized_parse`.

    Recognises the ``[json_string, type_string]`` shape produced by this
    exporter and routes it back to the DB without re-parsing.  Falls
    back to the default :func:`spinedb_api.to_database` for any other
    shape, so master templates that author raw scalars (``"no_method"``,
    numbers, ``None``) keep working.
    """
    if (
        isinstance(value, list)
        and len(value) == 2
        and (value[1] is None or isinstance(value[1], str))
    ):
        val_str, type_str = value
        if val_str is None:
            return None, None
        return val_str.encode("utf-8"), type_str
    return to_database(value)


def export_database(database_path: str, json_path: str) -> None:
    """Export ``database_path`` to a deterministic JSON file.

    The output is ``json.dumps``-ed with ``sort_keys=True`` and
    ``indent=2`` so diffs are reviewable and ordering is stable.  Lists
    inside the dump preserve ``spinedb_api`` insertion order — that is
    already deterministic per DB, and forcing a sort would invalidate
    ordered structures such as ``parameter_value_lists``.
    """
    if database_path.startswith("sqlite://"):
        url = database_path
    elif database_path.endswith(".sqlite"):
        url = "sqlite:///" + database_path
    else:
        raise ValueError(
            f"Expected a .sqlite path or sqlite:// URL, got {database_path!r}"
        )

    with DatabaseMapping(url, create=False, upgrade=True) as db:
        data = export_data(db, parse_value=keep_serialized_parse)

    with open(json_path, "w", encoding="utf-8") as out:
        json.dump(data, out, sort_keys=True, indent=2, ensure_ascii=False)
        out.write("\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("database_path", help="Source .sqlite path")
    parser.add_argument("json_path", help="Destination JSON path")
    args = parser.parse_args()
    export_database(args.database_path, args.json_path)
