"""Registry coverage: every parameter / variable / constraint named in
FlexTool must have a Layer-2 quantity-type / family registration.

This test is the future-proofing layer for the autoscaler.  Phase 0c
populated :data:`flextool.engine_polars.autoscale._quantity_types.
PARAMETER_TYPES` from the parameter_definition descriptions in
``templates/input_data_template.sqlite``; phase 1c populated
:data:`VARIABLE_FAMILIES` and :data:`CONSTRAINT_FAMILIES` from every
``add_var`` / ``add_cstr`` call site in ``flextool/engine_polars/``.

What this test pins:

* **Parameter coverage** — for every ``(parameter_name, entity_class)``
  pair the template DB declares, ``PARAMETER_TYPES`` must have an
  entry.  A new parameter added by a model author MUST also be
  registered here, otherwise Layer 2 would silently skip its scaling
  contribution.  Phase 0c was exhaustive; this test guards against
  drift.
* **Variable coverage** — every string literal passed to ``add_var``
  in non-test, non-autoscale engine sources must appear in
  ``VARIABLE_FAMILIES``.  A new variable that the autoscaler does not
  know about would raise :class:`KeyError` in Layer 2 at solve time;
  surfacing it here gives a clear failure message instead.
* **Constraint coverage** — every string literal passed to
  ``add_cstr`` must resolve via :func:`lookup_cstr` (which knows the
  ``_p`` / ``_n`` / ``_linear`` / ``_integer`` suffix-strip rules).
  Dynamic constraint names (``f"maxOnline_{suffix}"``) are out of
  scope for the static grep — the prefix entry in
  ``CONSTRAINT_FAMILIES`` covers them transparently.

We deliberately use a regex scan rather than AST parsing: the
``add_var`` / ``add_cstr`` call shape is unambiguous (the name is
always a leading string literal), the regex catches multi-line calls,
and we avoid the overhead of building a CPython AST for every engine
module.  The cost is that genuinely dynamic names ("f-strings") are
silently skipped — which is the correct behaviour here (no static
name to register).

The template DB lives at the absolute path below; it is read-only
(opened via ``mode=ro`` in the sqlite URI) so this test cannot
mutate it.  We do NOT copy to /tmp because we only read parameter
metadata, never solve.
"""
from __future__ import annotations

import re
import sqlite3
from pathlib import Path

import pytest


_TEMPLATE_DB = Path(
    "/home/jkiviluo/sources/flextool-engine/templates/input_data_template.sqlite"
)


# add_var("name", ...)  /  add_cstr("name", ...).  Both names are
# always the first positional argument as a string literal.  The
# pattern is permissive about whitespace + newlines so the multi-line
# calls in ``_dc_power_flow.py`` (``m.add_var(\n    "v_angle", ...)``)
# are matched too.
_ADD_VAR_PATT = re.compile(r"add_var\(\s*[\"']([a-zA-Z_][a-zA-Z0-9_]*)[\"']")
_ADD_CSTR_PATT = re.compile(r"add_cstr\(\s*[\"']([a-zA-Z_][a-zA-Z0-9_]*)[\"']")


# The solver-license probe in ``_solver_dispatch.py`` builds a
# throwaway Problem with a one-var LP called ``"x"`` purely to check
# whether the solver is licensed.  That variable never appears in a
# FlexTool LP — it lives entirely inside the probe — so excluding it
# from the registry is correct.
_PROBE_ONLY_VARS = {"x"}


def _engine_sources() -> list[Path]:
    """Engine_polars source files, excluding tests and the autoscale
    package (which has no ``add_var`` calls of its own)."""
    engine_dir = Path(__file__).resolve().parents[3] / "flextool" / "engine_polars"
    assert engine_dir.is_dir(), f"engine_polars not found at {engine_dir}"
    return [
        f for f in engine_dir.rglob("*.py")
        if "test" not in f.parts and "autoscale" not in f.parts
    ]


def test_parameter_registry_covers_template_db() -> None:
    """Every (parameter, entity_class) pair in the template has a
    :class:`QuantityType` registration.

    If this fails the diagnostic message lists the missing pairs —
    those parameters need a registry entry in
    ``flextool/engine_polars/autoscale/_quantity_types.py`` with a
    quantity tag derived from the parameter's ``description`` field.
    """
    if not _TEMPLATE_DB.exists():
        pytest.skip(f"template DB not present at {_TEMPLATE_DB}")
    from flextool.engine_polars.autoscale._quantity_types import PARAMETER_TYPES

    # Read-only URI so a buggy test cannot mutate the canonical
    # template DB even by accident.
    conn = sqlite3.connect(f"file:{_TEMPLATE_DB}?mode=ro", uri=True)
    try:
        rows = conn.execute(
            "SELECT pd.name, ec.name "
            "FROM parameter_definition pd "
            "JOIN entity_class ec ON pd.entity_class_id = ec.id"
        ).fetchall()
    finally:
        conn.close()

    pairs = [(p, ec) for p, ec in rows]
    missing = [p for p in pairs if p not in PARAMETER_TYPES]
    assert not missing, (
        f"{len(missing)} (parameter, entity_class) pairs missing from "
        f"PARAMETER_TYPES (out of {len(pairs)} total in template):\n"
        + "\n".join(f"  {p!r}" for p in missing[:50])
        + ("\n  ..." if len(missing) > 50 else "")
        + "\n\nAdd each pair to "
        "flextool/engine_polars/autoscale/_quantity_types.py with the "
        "QuantityType matching the parameter's description unit."
    )


def test_variable_registry_covers_engine_sources() -> None:
    """Every static ``add_var("name", ...)`` literal has a
    :class:`VarFamily` registration in :data:`VARIABLE_FAMILIES`.

    A missing entry would cause Layer 2 to raise :class:`KeyError` at
    solve time.  Surfacing it here gives the developer a clean failure
    list instead of an opaque solve-time crash.
    """
    from flextool.engine_polars.autoscale._layer2_types import VARIABLE_FAMILIES

    found: dict[str, list[Path]] = {}
    for src_path in _engine_sources():
        text = src_path.read_text(encoding="utf-8")
        for m in _ADD_VAR_PATT.finditer(text):
            found.setdefault(m.group(1), []).append(src_path)

    # Exclude the solver-probe variable (see _PROBE_ONLY_VARS).
    for name in _PROBE_ONLY_VARS:
        found.pop(name, None)

    missing = sorted(n for n in found if n not in VARIABLE_FAMILIES)
    assert not missing, (
        "add_var names without a VARIABLE_FAMILIES entry:\n"
        + "\n".join(
            f"  {n!r} (in {', '.join(sorted(set(str(p) for p in found[n])))})"
            for n in missing
        )
        + "\n\nAdd each to flextool/engine_polars/autoscale/_layer2_types.py "
        "with the QuantityType matching its column meaning."
    )


def test_constraint_registry_covers_engine_sources() -> None:
    """Every static ``add_cstr("name", ...)`` literal resolves via
    :func:`lookup_cstr` (exact match or known suffix strip).

    Dynamic constraint names (``f"maxOnline_{sfx}"``) are matched via
    prefix entries in the registry — those don't appear as literals in
    the grep, and the registry covers them transparently via the
    prefix dispatch in :func:`lookup_cstr`.
    """
    from flextool.engine_polars.autoscale._layer2_types import lookup_cstr

    found: dict[str, list[Path]] = {}
    for src_path in _engine_sources():
        text = src_path.read_text(encoding="utf-8")
        for m in _ADD_CSTR_PATT.finditer(text):
            found.setdefault(m.group(1), []).append(src_path)

    missing: list[str] = []
    for name in sorted(found):
        try:
            lookup_cstr(name)
        except KeyError:
            missing.append(name)

    assert not missing, (
        "add_cstr names without a CONSTRAINT_FAMILIES entry "
        "(neither exact match nor known suffix strip resolves them):\n"
        + "\n".join(
            f"  {n!r} (in {', '.join(sorted(set(str(p) for p in found[n])))})"
            for n in missing
        )
        + "\n\nAdd each to flextool/engine_polars/autoscale/_layer2_types.py "
        "with the QuantityType matching its row-bound (RHS) meaning, or "
        "``CstrFamily(None)`` if the row carries no per-row scaling."
    )
