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


def test_parameter_registry_covers_schema(schema_db_url: str) -> None:
    """Every (parameter, entity_class) pair the schema declares has a
    :class:`QuantityType` registration.

    Reads from a DB built off ``schemas/spinedb_schema.json`` at session
    start (the ``schema_db_url`` fixture) rather than a checked-in
    template SQLite.  The template lags the schema between regenerations
    and silently under-covers — the v56 ``is_enabled`` add was present in
    the schema but missing from ``templates/input_data_template.sqlite``,
    so the old template-DB form of this test stayed green while
    ``PARAMETER_TYPES`` was incomplete.  Building from the schema also
    exercises the real ``import_data`` path.

    If this fails the diagnostic lists the missing pairs — add each to
    ``flextool/engine_polars/autoscale/_quantity_types.py`` with the
    quantity tag from the parameter's description (flags / yes_no gates →
    DIMENSIONLESS).
    """
    from flextool.engine_polars.autoscale._quantity_types import PARAMETER_TYPES

    db_path = schema_db_url.replace("sqlite:///", "")
    # Read-only URI so a buggy test cannot mutate the built DB by accident.
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
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
        f"{len(missing)} (parameter, entity_class) pairs declared in "
        f"spinedb_schema.json are missing from PARAMETER_TYPES (out of "
        f"{len(pairs)} total):\n"
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


# Dynamic (f-string) constraint names that the literal grep above CANNOT
# see — ``add_cstr(f"...")`` calls.  These resolve only through
# ``lookup_cstr``'s prefix / suffix dispatch, so the grep test's claim
# that "the registry covers them transparently" is unverified by it.
# The ramp family proved that gap can bite: the ``"ramp"`` prefix entry
# existed but prefix dispatch was unimplemented, so
# ``ramp_sink_up_constraint`` raised KeyError and silently degraded the
# whole solve to an un-scaled LP.  Pin every dynamic family behaviorally
# here.  When a new ``add_cstr(f"...")`` family is added in the engine,
# add a representative instantiation to this list.
_DYNAMIC_CONSTRAINT_SAMPLES = [
    # ramp — model.py:2328  f"ramp_{side}_{dir_}_constraint"
    "ramp_sink_up_constraint", "ramp_sink_down_constraint",
    "ramp_source_up_constraint", "ramp_source_down_constraint",
    # unit commitment — model.py:3909+  f"...{sfx}", sfx ∈ {_linear,_integer}
    "maxOnline_linear", "maxOnline_integer",
    "maxStartup_linear", "maxStartup_integer",
    "maxShutdown_linear", "maxShutdown_integer",
    "online__startup_linear", "online__startup_integer",
    "online__shutdown_linear", "online__shutdown_integer",
    "maxToSink_online_linear", "maxToSink_online_integer",
    "minToSink_minload_linear", "minToSink_minload_integer",
    "minimum_uptime_linear", "minimum_uptime_integer",
    "minimum_downtime_linear", "minimum_downtime_integer",
]


@pytest.mark.parametrize("name", _DYNAMIC_CONSTRAINT_SAMPLES)
def test_dynamic_constraint_names_resolve(name: str) -> None:
    """Every dynamic (f-string) constraint family resolves via
    :func:`lookup_cstr` and :func:`resolve_cstr_rhs_type`.

    Behavioral complement to the literal-grep test above: actually
    instantiating and resolving the dynamic names is the only thing that
    catches a registry entry whose dispatch is broken (the ramp bug) or a
    new dynamic family with no entry at all.
    """
    from flextool.engine_polars.autoscale._layer2_types import (
        lookup_cstr,
        resolve_cstr_rhs_type,
    )

    assert lookup_cstr(name) is not None  # raises KeyError if unresolved
    # The path Layer 2 actually calls; must not raise for these samples
    # (none carry the _p/_n suffix the group_capacity resolver requires).
    resolve_cstr_rhs_type(name)
