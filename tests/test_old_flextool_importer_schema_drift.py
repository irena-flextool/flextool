"""Guard: the FlexTool-2 importer must only write parameters its template defines.

The ``.xlsm`` importer (``write_old_flextool_to_db``) hand-writes parameter
values against a **frozen** schema version — the v56 import template
(``schemas/old_flextool_import_template.json``) it initialises the DB from;
``migrate_database`` then carries the result up to the live schema.  The
importer is deliberately pinned to that template and is NOT updated for later
renames — migration owns those.

The hazard this guards is ``_add_param`` **silently logging and skipping** an
unknown parameter rather than failing: if the importer writes a name the
frozen template does not define, the value is dropped with no error and the
imported model is quietly wrong (a fuel plant that cannot generate, an
investment with no discount rate, ...).  That can happen if someone edits the
importer to a name from a *later* schema version, or re-snapshots the template
inconsistently.

This is the CLAUDE.md invariant-#4 hazard for the importer.  Rather than rely
on someone re-running a model and noticing missing output, this test pins the
contract statically: every ``(entity_class, parameter)`` the importer emits
via ``_add_param`` / ``_add_param_if_set`` with literal names must exist in
the frozen import template.  A drifted name fails here immediately.

Real regressions this would have caught:
- ``unit/connection/node . interest_rate`` (renamed to ``discount_rate`` at
  migration v28) — silently dropped, leaving investments with no discount
  rate.
"""
from __future__ import annotations

import ast
import json
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
_IMPORTER = _REPO / "flextool" / "process_inputs" / "write_old_flextool_to_db.py"
# The importer targets its FROZEN template, not the live schema — migration
# bridges the two, so checking against the live schema would wrongly fail the
# moment a future rename lands.
_SCHEMA = _REPO / "flextool" / "schemas" / "old_flextool_import_template.json"

_ADDERS = {"_add_param", "_add_param_if_set"}


def _importer_param_pairs() -> set[tuple[str, str]]:
    """Extract every literal ``(class_name, param_name)`` the importer writes.

    Signature is ``_add_param(db, class_name, entity_byname, param_name, ...)``
    (and the same first four positions for ``_add_param_if_set``), so the
    class is ``args[1]`` and the parameter name is ``args[3]``.  Calls whose
    class or name is a variable (the lone one being ``_add_param_if_set``'s
    internal delegation to ``_add_param``) are skipped — they are exercised
    through their concrete literal callers.
    """
    tree = ast.parse(_IMPORTER.read_text())
    pairs: set[tuple[str, str]] = set()
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id in _ADDERS
            and len(node.args) >= 4
        ):
            cls, pname = node.args[1], node.args[3]
            if isinstance(cls, ast.Constant) and isinstance(pname, ast.Constant):
                pairs.add((cls.value, pname.value))
    return pairs


def _schema_param_pairs() -> set[tuple[str, str]]:
    schema = json.loads(_SCHEMA.read_text())
    return {(pd[0], pd[1]) for pd in schema.get("parameter_definitions", [])}


def test_importer_params_exist_in_schema() -> None:
    written = _importer_param_pairs()
    defined = _schema_param_pairs()

    # Sanity: the extractor must actually find the importer's writes; a parse
    # change that silently returns nothing would make this test vacuous.
    assert len(written) > 50, (
        f"extractor found only {len(written)} parameter writes — the "
        "_add_param call shape likely changed; update the extractor."
    )

    missing = sorted(p for p in written if p not in defined)
    assert not missing, (
        "FlexTool-2 importer writes parameters absent from "
        "schemas/spinedb_schema.json (a schema rename was not propagated to "
        "write_old_flextool_to_db.py; _add_param drops these silently):\n"
        + "\n".join(f"  {cls}.{name}" for cls, name in missing)
    )
