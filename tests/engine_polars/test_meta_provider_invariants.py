"""Meta-test enforcing post-Step-2 cascade invariants.

After Step 2 of the FlexDataProvider migration, the cascade has ONE
data pathway:

    Source → Provider → loaders / writers / LP / post-processing

Cascade modules consume data through ``provider.get(name)`` (and
parent-qualified variants).  Direct CSV reads from disk in cascade
code re-introduce the deleted seed-funnel class of bug: a frame that
the live Provider already carries gets read back from a stale (or
empty) on-disk artefact instead.  See:

* ``specs/flex_data_provider_migration_handoff.md`` — guardrails.
* ``specs/architecture_provider_and_future_duckdb.md`` — the
  "CSV dump is a snapshot, not a re-creation" principle.

This test scans ``flextool/engine_polars/*.py`` and fails if a new
cascade module introduces any of the three rule violations below
without an explicit exemption.

Rule 1 — No disk CSV reads in cascade code
==========================================

Forbidden call shapes in cascade modules:

* ``pl.read_csv(...)`` / ``pl.scan_csv(...)``
* ``_read_csv_file(...)`` (the residual Provider-fallback helper)
* ``csv.reader(<fh>)`` where ``<fh>`` is the result of a bare
  ``path.open()`` (i.e. NOT a Provider-served buffer)

Files that legitimately read raw inputs are listed in
``RAW_INPUT_FALLBACK_ALLOWLIST`` below, with a per-file justification.
Files that ARE the Provider's own I/O implementation are listed in
``PROVIDER_IMPL_ALLOWLIST``.

Rule 2 — No "photocopier" patterns
==================================

A "photocopier" is anything that takes a frame off the Provider and
stores it where Step 3's eviction cannot see it:

* ``@functools.lru_cache`` / ``@cache`` on functions that call
  ``provider.get(...)``
* Module-level globals typed ``pl.DataFrame`` (or ``dict[str,
  pl.DataFrame]``) outside the Provider implementation file

Rule 3 — ``_provider_or_*`` / ``_provider_has_or_*`` shims require justification
================================================================================

The transitional ``_provider_or_exists`` / ``_provider_or_read_csv`` /
``_provider_has_or_exists`` shim pattern wraps ``provider.get`` with a
disk fallback for raw inputs not yet in the Provider.  These ARE
allowed where documented, but every shim DEFINITION must carry a
``# CASCADE INVARIANT EXEMPT: <reason>`` comment on (or immediately
above) its ``def`` line, or the marker phrase inside its docstring.

When this test fires
====================

The failure message names the file, line, and offending pattern, and
points the developer at the fix:

* "Route the read through ``provider.get(name)``"
* "If this file legitimately falls outside the cascade, add it to
  the allowlist with a per-file justification"
* "If the ``_provider_or_*`` shim is genuinely transitional, add the
  ``# CASCADE INVARIANT EXEMPT: <reason>`` comment"
"""
from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path

import pytest


CASCADE_ROOT = Path(__file__).resolve().parents[2] / "flextool" / "engine_polars"


# ---------------------------------------------------------------------------
# Allowlists (with per-file justifications)
# ---------------------------------------------------------------------------


# Files that ARE the Provider's own I/O / snapshot implementation.  The
# rules don't apply here by construction.
PROVIDER_IMPL_ALLOWLIST: dict[str, str] = {
    "_flex_data_provider.py": (
        "Provider implementation: defines snapshot/dump I/O."
    ),
    "_input_source.py": (
        "Defines _read_csv_file itself — the canonical residual reader."
    ),
    "_writer_provider_io.py": (
        "Off-cascade writer-port test-harness shim "
        "(documented at module top): Provider-first with disk fallback "
        "used exclusively by test_writer_port_phase1.  In-cascade the "
        "Provider always carries the frame."
    ),
    "_dump_csvs.py": (
        "Snapshot-writer subsystem (--csv-dump). "
        "pl.read_csv here serves the merge-write path that preserves "
        "existing on-disk slices when dumping new rows; this is a "
        "snapshot I/O concern, not a cascade read."
    ),
}


# Cascade files that legitimately fall back to disk for raw fixture
# inputs (``input/*.csv``) and legacy synthetic-seed inputs not yet
# carried by the Provider.  Each entry MUST cite why the file's reads
# are out of scope for the current migration step.  When the input_writer
# preprocessing layer lands in the Provider, these entries should be
# removed and the corresponding files retro-migrated.
RAW_INPUT_FALLBACK_ALLOWLIST: dict[str, str] = {
    # --- Raw input fixture readers (input/*.csv not in Provider) ---
    "_solve_handoff.py": (
        "Reads workdir solve-data CSV at Provider-less handoff "
        "boundary (post-Step-2 the Provider is threaded through, "
        "but the handoff retains a single disk read for back-compat)."
    ),
    "_solve_context.py": (
        "Reads workdir solve_data/p_model.csv and similar context "
        "files at solve-context build time (pre-Provider)."
    ),
    # Note: `_pdt_lookup.py` and the writer-port modules
    # (`_writer_arc_unions.py`, `_writer_chain_params.py`,
    # `_writer_dispatchers.py`, `_writer_pdt_params.py`,
    # `_writer_period_params.py`, `_writer_solve_writers.py`) previously
    # appeared here because the Rule 1 AST detector couldn't distinguish
    # `csv.reader(fh)` on a `_provider_open()` in-memory buffer from
    # `csv.reader(open(path))` on disk.  The detector now whitelists
    # `csv.reader` sites whose enclosing function calls `_provider_open`,
    # so those files no longer need allowlist entries.
}


# Union — files exempt from Rule 1.
RULE_1_ALLOWLIST: dict[str, str] = {
    **PROVIDER_IMPL_ALLOWLIST,
    **RAW_INPUT_FALLBACK_ALLOWLIST,
}


# ---------------------------------------------------------------------------
# Violation reporting
# ---------------------------------------------------------------------------


@dataclass
class Violation:
    file: str
    line: int
    pattern: str
    hint: str

    def format(self) -> str:
        return f"  {self.file}:{self.line}  {self.pattern}  — {self.hint}"


# ---------------------------------------------------------------------------
# AST detectors
# ---------------------------------------------------------------------------


def _iter_cascade_files():
    for p in sorted(CASCADE_ROOT.glob("*.py")):
        if p.name == "__init__.py":
            continue
        yield p


def _is_pl_call(call: ast.Call, attr: str) -> bool:
    f = call.func
    return (
        isinstance(f, ast.Attribute)
        and f.attr == attr
        and isinstance(f.value, ast.Name)
        and f.value.id == "pl"
    )


def _is_read_csv_file_call(call: ast.Call) -> bool:
    f = call.func
    if isinstance(f, ast.Name):
        return f.id == "_read_csv_file"
    if isinstance(f, ast.Attribute):
        return f.attr == "_read_csv_file"
    return False


def _is_csv_reader_call(call: ast.Call) -> bool:
    f = call.func
    return (
        isinstance(f, ast.Attribute)
        and f.attr == "reader"
        and isinstance(f.value, ast.Name)
        and f.value.id in ("csv", "_csv")
    )


def _function_uses_provider_open(func_node: ast.AST) -> bool:
    """Return True iff *func_node* contains a call to ``_provider_open``.

    Used to whitelist ``csv.reader(fh)`` sites where ``fh`` is produced
    by ``_provider_open(provider, ...)`` — i.e. served from the
    in-memory Provider, never from disk.
    """
    for sub in ast.walk(func_node):
        if not isinstance(sub, ast.Call):
            continue
        f = sub.func
        if isinstance(f, ast.Name) and f.id == "_provider_open":
            return True
        if isinstance(f, ast.Attribute) and f.attr == "_provider_open":
            return True
    return False


def _detect_rule1_violations(path: Path, tree: ast.AST) -> list[Violation]:
    out: list[Violation] = []
    fname = path.name
    # Map each ast node to its enclosing function (None for module-level).
    parent_func: dict[int, ast.AST] = {}

    def _walk(node: ast.AST, current_func: ast.AST | None) -> None:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            current_func = node
        if isinstance(node, ast.Call):
            parent_func[id(node)] = current_func  # type: ignore[assignment]
        for child in ast.iter_child_nodes(node):
            _walk(child, current_func)

    _walk(tree, None)

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if _is_pl_call(node, "read_csv"):
            out.append(Violation(
                fname, node.lineno, "pl.read_csv(...)",
                "Route the read through provider.get(name) instead.",
            ))
        elif _is_pl_call(node, "scan_csv"):
            out.append(Violation(
                fname, node.lineno, "pl.scan_csv(...)",
                "Route the read through provider.get(name) instead.",
            ))
        elif _is_read_csv_file_call(node):
            out.append(Violation(
                fname, node.lineno, "_read_csv_file(...)",
                "Use provider.get(name); _read_csv_file is the "
                "Provider-fallback helper, not a cascade entry point.",
            ))
        elif _is_csv_reader_call(node):
            # Whitelist: csv.reader on a handle produced by
            # _provider_open(...) inside the same function.  The
            # Provider serves the bytes from memory in this case.
            enclosing = parent_func.get(id(node))
            if enclosing is not None and _function_uses_provider_open(
                enclosing
            ):
                continue
            out.append(Violation(
                fname, node.lineno, "csv.reader(<fh>)",
                "Cascade code must obtain the file handle from "
                "_provider_open(...) so the Provider serves the bytes "
                "when available.  Direct csv.reader on a disk fh "
                "reintroduces the seed-funnel bug class.",
            ))
    return out


def _detect_rule2_lru_cache(tree: ast.AST, fname: str) -> list[Violation]:
    """Detect @functools.lru_cache / @cache / @functools.cache on a
    function whose body references ``provider.get(...)``.
    """
    out: list[Violation] = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        cache_decorator = None
        for dec in node.decorator_list:
            target = dec.func if isinstance(dec, ast.Call) else dec
            if isinstance(target, ast.Attribute) and target.attr in (
                "lru_cache", "cache",
            ):
                cache_decorator = target.attr
                break
            if isinstance(target, ast.Name) and target.id in (
                "lru_cache", "cache",
            ):
                cache_decorator = target.id
                break
        if cache_decorator is None:
            continue
        # Check function body for provider.get call.
        uses_provider = False
        for sub in ast.walk(node):
            if not isinstance(sub, ast.Call):
                continue
            f = sub.func
            if (
                isinstance(f, ast.Attribute)
                and f.attr == "get"
                and isinstance(f.value, ast.Name)
                and f.value.id == "provider"
            ):
                uses_provider = True
                break
        if uses_provider:
            out.append(Violation(
                fname, node.lineno,
                f"@{cache_decorator} on '{node.name}' (calls provider.get)",
                "Caching Provider results across calls hides "
                "Provider references from Step 3's eviction.  Use "
                "the Provider's own cache or restructure the call.",
            ))
    return out


def _detect_rule2_module_globals(tree: ast.AST, fname: str) -> list[Violation]:
    """Detect module-level globals annotated as pl.DataFrame or
    dict[str, pl.DataFrame], populated at import time.

    Only the Provider implementation file may hold module-level frame
    dicts.
    """
    out: list[Violation] = []
    if fname == "_flex_data_provider.py":
        return out

    def _annotation_mentions_dataframe(ann: ast.AST) -> bool:
        for n in ast.walk(ann):
            if (
                isinstance(n, ast.Attribute)
                and n.attr == "DataFrame"
                and isinstance(n.value, ast.Name)
                and n.value.id == "pl"
            ):
                return True
        return False

    if not isinstance(tree, ast.Module):
        return out
    for stmt in tree.body:
        if isinstance(stmt, ast.AnnAssign) and stmt.annotation is not None:
            if _annotation_mentions_dataframe(stmt.annotation):
                # Only flag if the value is not None — None-initialised
                # placeholders are fine (e.g. lazy init).
                if stmt.value is not None and not (
                    isinstance(stmt.value, ast.Constant)
                    and stmt.value.value is None
                ):
                    target = (
                        stmt.target.id
                        if isinstance(stmt.target, ast.Name)
                        else "<global>"
                    )
                    out.append(Violation(
                        fname, stmt.lineno,
                        f"module-level '{target}: pl.DataFrame' "
                        f"populated at import",
                        "Module-level frame globals hide Provider "
                        "references.  Pass the frame through "
                        "explicit arguments or hold it on a per-"
                        "sub-solve object instead.",
                    ))
    return out


_EXEMPT_RE = "CASCADE INVARIANT EXEMPT"


def _detect_rule3_shim_definitions(
    path: Path, tree: ast.AST, source: str,
) -> list[Violation]:
    """Flag any ``def _provider_or_*`` whose definition or nearby
    comments don't carry ``# CASCADE INVARIANT EXEMPT: <reason>``.

    "Nearby" = the def line itself, the line above, or the function's
    docstring text.
    """
    out: list[Violation] = []
    fname = path.name
    if fname in PROVIDER_IMPL_ALLOWLIST:
        return out
    lines = source.splitlines()
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        # Detect both ``_provider_or_*`` and ``_provider_has_or_*``
        # shim shapes — same architectural pattern.
        if not (
            node.name.startswith("_provider_or_")
            or node.name.startswith("_provider_has_or_")
        ):
            continue
        # Inspect: def line, line above, and the function's first
        # docstring node.
        def_line_idx = node.lineno - 1
        ctx_lines: list[str] = []
        if def_line_idx >= 0:
            ctx_lines.append(lines[def_line_idx])
        if def_line_idx - 1 >= 0:
            ctx_lines.append(lines[def_line_idx - 1])
        docstring = ast.get_docstring(node) or ""
        if _EXEMPT_RE in docstring:
            continue
        if any(_EXEMPT_RE in l for l in ctx_lines):
            continue
        out.append(Violation(
            fname, node.lineno,
            f"def {node.name}(...) without CASCADE INVARIANT EXEMPT",
            "_provider_or_* shims are transitional Provider-first "
            "wrappers with disk fallback.  Add a "
            "'# CASCADE INVARIANT EXEMPT: <reason>' comment on the "
            "def line (or in the docstring) explaining why the disk "
            "arm is still needed.",
        ))
    return out


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def _read_tree(path: Path) -> tuple[str, ast.AST]:
    text = path.read_text()
    return text, ast.parse(text, filename=str(path))


def test_no_disk_csv_reads_in_cascade() -> None:
    """Rule 1: no bare disk CSV reads in cascade modules."""
    violations: list[Violation] = []
    for path in _iter_cascade_files():
        if path.name in RULE_1_ALLOWLIST:
            continue
        _, tree = _read_tree(path)
        violations.extend(_detect_rule1_violations(path, tree))
    if violations:
        msg = [
            "Cascade invariant violation: direct disk CSV reads found "
            "in cascade code.",
            "",
            *(v.format() for v in violations),
            "",
            "FIX OPTIONS:",
            "  (a) Route the read through provider.get(name) — the "
            "canonical cascade pathway after Step 2.",
            "  (b) If this file legitimately runs OUTSIDE the cascade "
            "(Provider impl, off-cascade test harness, --csv-dump "
            "snapshot, raw input loader before Provider is populated), "
            "add it to PROVIDER_IMPL_ALLOWLIST or "
            "RAW_INPUT_FALLBACK_ALLOWLIST in this file WITH a per-file "
            "justification explaining why disk reads remain.",
            "",
            "See specs/flex_data_provider_migration_handoff.md for the "
            "post-Step-2 cascade contract.",
        ]
        pytest.fail("\n".join(msg))


def test_no_photocopier_lru_cache() -> None:
    """Rule 2a: no @lru_cache / @cache wrapping provider.get callers."""
    violations: list[Violation] = []
    for path in _iter_cascade_files():
        if path.name in PROVIDER_IMPL_ALLOWLIST:
            continue
        _, tree = _read_tree(path)
        violations.extend(_detect_rule2_lru_cache(tree, path.name))
    if violations:
        pytest.fail(
            "Photocopier pattern detected: @lru_cache / @cache wraps a "
            "function that calls provider.get(...).  Cached Provider "
            "results survive past the Provider's intended lifetime and "
            "hide references from Step 3's eviction pass.\n\n"
            + "\n".join(v.format() for v in violations)
        )


def test_no_module_level_frame_globals() -> None:
    """Rule 2b: no module-level pl.DataFrame globals in cascade modules.

    Conservative: only flags ``<name>: pl.DataFrame = <non-None>``.  Type
    aliases and ``None``-initialised placeholders are allowed.
    """
    violations: list[Violation] = []
    for path in _iter_cascade_files():
        if path.name in PROVIDER_IMPL_ALLOWLIST:
            continue
        _, tree = _read_tree(path)
        violations.extend(_detect_rule2_module_globals(tree, path.name))
    if violations:
        pytest.fail(
            "Photocopier pattern detected: module-level pl.DataFrame "
            "global populated at import.\n\n"
            + "\n".join(v.format() for v in violations)
            + "\n\nHold per-sub-solve frames on the Provider or on an "
            "explicit per-call object; never on a module-level global."
        )


def test_provider_or_shims_have_justification() -> None:
    """Rule 3: every ``_provider_or_*`` def carries a
    ``# CASCADE INVARIANT EXEMPT: <reason>`` comment or docstring tag.
    """
    violations: list[Violation] = []
    for path in _iter_cascade_files():
        source, tree = _read_tree(path)
        violations.extend(_detect_rule3_shim_definitions(path, tree, source))
    if violations:
        pytest.fail(
            "Transitional _provider_or_* shim defined without a "
            "'# CASCADE INVARIANT EXEMPT: <reason>' justification.\n\n"
            + "\n".join(v.format() for v in violations)
            + "\n\nAdd the exemption comment on the def line (or "
            "inside the docstring) explaining why the disk-fallback "
            "arm is needed (typically: raw input fixtures not yet in "
            "the Provider; input_writer.py preprocessing pending "
            "migration)."
        )


def test_allowlists_cover_only_existing_files() -> None:
    """Sanity: every allowlisted filename must exist on disk.

    Catches stale allowlist entries left behind after refactors.
    """
    on_disk = {p.name for p in _iter_cascade_files()}
    stale = [
        name for name in (PROVIDER_IMPL_ALLOWLIST | RAW_INPUT_FALLBACK_ALLOWLIST)
        if name not in on_disk
    ]
    if stale:
        pytest.fail(
            "Stale allowlist entries (file no longer exists in "
            f"flextool/engine_polars/): {stale}.  Remove from "
            "PROVIDER_IMPL_ALLOWLIST / RAW_INPUT_FALLBACK_ALLOWLIST."
        )
