"""Audit defaults declared in the FlexTool DB schema vs. the .mod file.

Phase 1 of the migration moves all ``default X`` clauses from
``flextool.mod`` to the DB schema (``flextool_template_master.json``).
This script catalogs:

- Every ``parameter_definition`` in the master template, with its current
  ``default_value`` (may be None).
- Every ``param X{...} ... default Y;`` clause in flextool.mod.

Output: ``migration/schema_defaults.csv`` with columns:

  source           : 'schema' or 'mod'
  entity_class     : for schema rows; empty for mod (mod params don't carry class)
  param_name       : the parameter name as it appears in the source
  default_value    : raw default — None / empty if absent
  default_type     : schema-side type tag (str, float, etc.)
  parameter_value_list : the value-list name (enum) if any
  mod_line         : line number in flextool.mod (mod rows only)
  description      : schema description (schema rows only)

The PHASE 1 task per parameter is then:
1. Find a 'mod' row with a default but no matching 'schema' row — schema
   migration needed (add the default).
2. Find a 'schema' row with a default whose 'mod' counterpart STILL has
   ``default Y;`` — drop the .mod default (Python preprocessing /
   ``input_writer.py`` already materializes the schema default into the
   CSV).
3. Find a 'mod' row whose 'schema' counterpart has DIFFERENT default —
   investigate and reconcile.

Cross-referencing schema rows to mod rows is intentionally NOT
automated here: it depends on the per-param-table convention used by
input_writer.py (e.g., the schema's ``connection.availability`` feeds
into mod's ``pdtProcess[p, 'availability', d, t]``). The agent
performing each Phase 1 step does the lookup by reading the relevant
input_writer code path.

CLI:
    python -m migration.schema_defaults_audit
        [--template version/flextool_template_master.json]
        [--mod flextool/flextool.mod]
        [--out migration/schema_defaults.csv]
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path


# ``param X{...} default Y;`` — extract name and default literal.
# The default literal stops at ``;`` or the next ``,`` clause inside a
# multi-clause header. We stop at whitespace+``;``.
_PARAM_DEFAULT = re.compile(
    r"""^\s*param\s+
        (?P<name>[A-Za-z_][A-Za-z_0-9]*)
        (?P<rest>[\s\S]*?)
        \bdefault\s+(?P<default>[^\s;]+)
    """,
    re.VERBOSE,
)


def _decode_default(raw: bytes | str | None) -> str | None:
    """Schema defaults are either bytes (typed via to_database) or None."""
    if raw is None:
        return None
    if isinstance(raw, str):
        # JSON serialization stores bytes as base64; but spinedb
        # round-tripped JSON typically gives readable strings for simple
        # primitives. Accept whatever is here.
        return raw
    if isinstance(raw, bytes):
        try:
            return raw.decode("utf-8")
        except UnicodeDecodeError:
            return raw.hex()
    return repr(raw)


def schema_rows(template_path: Path) -> list[dict]:
    data = json.loads(template_path.read_text())
    rows: list[dict] = []
    for pd in data.get("parameter_definitions", []):
        # Layout: [class, name, default_value, default_type, description, value_list]
        cls = pd[0] if len(pd) > 0 else ""
        name = pd[1] if len(pd) > 1 else ""
        default = pd[2] if len(pd) > 2 else None
        dtype = pd[3] if len(pd) > 3 else None
        desc = pd[4] if len(pd) > 4 else ""
        vlist = pd[5] if len(pd) > 5 else None
        rows.append({
            "source": "schema",
            "entity_class": cls,
            "param_name": name,
            "default_value": _decode_default(default) if default is not None else "",
            "default_type": dtype or "",
            "parameter_value_list": vlist or "",
            "mod_line": "",
            "description": desc or "",
        })
    return rows


def mod_default_rows(mod_path: Path) -> list[dict]:
    """Find every ``param X ... default Y;`` in flextool.mod.

    Single-pass: split on top-level ``;`` and regex-match each statement.
    """
    src = mod_path.read_text()
    # Strip line comments to avoid commented-out defaults
    src_no_comments = re.sub(r"#[^\n]*", "", src)
    rows: list[dict] = []
    line_offsets = [0]
    for ch in src_no_comments:
        if ch == "\n":
            line_offsets.append(line_offsets[-1] + 1)
    # Use simple scan: iterate through ``param`` declarations
    pos = 0
    while True:
        idx = src_no_comments.find("\nparam ", pos)
        # Also handle param at file start
        if idx < 0:
            if pos == 0:
                idx = -1 if not src_no_comments.startswith("param ") else 0
            else:
                break
        if idx < 0:
            break
        # Find the terminating ; (top-level)
        end = src_no_comments.find(";", idx)
        if end < 0:
            break
        stmt = src_no_comments[idx:end]
        m = _PARAM_DEFAULT.match(stmt)
        if m:
            # 1-based line number of the start of the statement
            line_no = src_no_comments.count("\n", 0, idx) + 1
            rows.append({
                "source": "mod",
                "entity_class": "",
                "param_name": m.group("name"),
                "default_value": m.group("default"),
                "default_type": "",
                "parameter_value_list": "",
                "mod_line": str(line_no),
                "description": "",
            })
        pos = end + 1
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--template", type=Path,
                        default=Path("version/flextool_template_master.json"))
    parser.add_argument("--mod", type=Path,
                        default=Path("flextool/flextool.mod"))
    parser.add_argument("--out", type=Path,
                        default=Path("migration/schema_defaults.csv"))
    args = parser.parse_args(argv)

    schema = schema_rows(args.template)
    mod = mod_default_rows(args.mod)

    rows = sorted(schema + mod, key=lambda r: (r["source"], r["entity_class"], r["param_name"]))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    n_schema = len(schema)
    n_schema_with_default = sum(1 for r in schema if r["default_value"])
    n_mod = len(mod)
    print(f"Wrote {args.out}")
    print(f"  schema parameter_definitions: {n_schema}")
    print(f"    with default_value:         {n_schema_with_default}")
    print(f"    without default_value:      {n_schema - n_schema_with_default}")
    print(f"  flextool.mod ``default`` clauses: {n_mod}")
    print()
    print("Phase 1 cross-ref convention: agents look up which schema rows")
    print("feed which mod params via input_writer.py's CSV→param wiring.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
