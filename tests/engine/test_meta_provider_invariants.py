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

This test scans every ``*.py`` under ``flextool/engine/`` recursively
(except the package root ``__init__.py`` and the ``autoscale/``
subpackage) and fails if a new cascade module introduces any of the
four rule violations below without an explicit exemption.  Allowlists
and violation reports key files by their POSIX path relative to
``flextool/engine/`` (e.g. ``_dump_csvs.py`` or ``output/_dump_csvs.py``).

Rule 1 — No disk CSV reads in cascade code
==========================================

Forbidden call shapes in cascade modules:

* ``pl.read_csv(...)`` / ``pl.scan_csv(...)``
* ``_read_csv_file(...)`` (the plain read behind ``CsvSource.get``)
* ``csv.reader(...)`` (cascade code reads frames from the Provider, never
  CSV text from a file handle)

Files that ARE the Provider's own I/O implementation (or a dump writer)
are listed in ``PROVIDER_IMPL_ALLOWLIST``.  ``RAW_INPUT_FALLBACK_ALLOWLIST``
(cascade files that read raw inputs from disk) is empty: since c24 stage 3
the loader reads only the Provider.

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

Rule 4 — The loader and the model have no disk arms
===================================================

The main loader is provider-only (c24 stage 3): nothing in a run reads
``input/`` or ``solve_data/`` back from the work folder, so a dump or
stale CSVs in a reused folder cannot change a run.

* (a) Anywhere in cascade code: no reference (name, attribute, import,
  definition) to the deleted disk helpers in ``RULE_4_BANNED_NAMES``.
* (b) In ``preprocess/`` and ``model/``: no file-system probe or read —
  ``.exists()`` / ``.is_file()`` / ``.is_dir()`` / ``.is_symlink()``
  (also ``os.path.exists/isfile/isdir``), ``open(...)`` / ``.open(...)``
  in a read mode (mode absent, not a literal, or containing ``r`` or
  ``+``), ``.read_text()`` / ``.read_bytes()``, directory listings, and
  the polars / json file readers.  Write and append ``open`` is allowed.
  ``RULE_4_READ_ALLOWLIST`` names the few legitimate reads by file and
  literal path (the ``/proc/self/status`` RSS probes).

``RULE_4_EXEMPT_FILES`` lists files the rule does not apply to, each
with a reason; a test checks that every exemption is still needed.

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


CASCADE_ROOT = Path(__file__).resolve().parents[2] / "flextool" / "engine"

# Subfolders of CASCADE_ROOT that are not cascade code.  ``autoscale/``
# is the LP-scaling layer (no Provider access).  ``_csv_reload/`` is the
# unsupported, opt-in CSV-dump reload: it reads dump CSVs from disk by
# design and nothing in the cascade imports it (see
# ``test_csv_reload_is_a_leaf`` below).
EXCLUDED_SUBDIRS: frozenset[str] = frozenset({"autoscale", "_csv_reload"})

# Relative paths of files skipped outright: the package root
# ``__init__.py`` only re-exports the public API.
EXCLUDED_FILES: frozenset[str] = frozenset({"__init__.py"})

# The scan must never silently cover nothing (e.g. after files move
# into subfolders and the walk stops finding them).
MIN_SCANNED_FILES = 80


# ---------------------------------------------------------------------------
# Allowlists (with per-file justifications)
# ---------------------------------------------------------------------------


# Keys of every allowlist below are POSIX paths relative to CASCADE_ROOT.

# The Provider implementation module — the only file allowed to hold
# module-level frame globals (Rule 2b).
PROVIDER_IMPL_FILE = "preprocess/_flex_data_provider.py"


# Files that ARE the Provider's own I/O / snapshot implementation.  The
# rules don't apply here by construction.
PROVIDER_IMPL_ALLOWLIST: dict[str, str] = {
    "preprocess/_flex_data_provider.py": (
        "Provider implementation: defines snapshot/dump I/O."
    ),
    "input/_input_source.py": (
        "Defines _read_csv_file (the plain read behind CsvSource.get); "
        "no main-path caller."
    ),
    "output/_dump_csvs.py": (
        "Snapshot-writer subsystem (--csv-dump). "
        "pl.read_csv here serves the merge-write path that preserves "
        "existing on-disk slices when dumping new rows; this is a "
        "snapshot I/O concern, not a cascade read."
    ),
}


# Cascade files that read raw inputs from disk.  Empty since c24 stage 3
# (the loader, SolveContext and the handoff read only the Provider).  A
# new entry needs a per-file justification of why the Provider cannot
# carry the frame.
RAW_INPUT_FALLBACK_ALLOWLIST: dict[str, str] = {}


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


def _rel(path: Path) -> str:
    """POSIX path of *path* relative to CASCADE_ROOT (allowlist key)."""
    return path.relative_to(CASCADE_ROOT).as_posix()


def _iter_cascade_files():
    for p in sorted(CASCADE_ROOT.rglob("*.py")):
        rel = p.relative_to(CASCADE_ROOT)
        if rel.parts[0] in EXCLUDED_SUBDIRS:
            continue
        if rel.as_posix() in EXCLUDED_FILES:
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


def _detect_rule1_violations(path: Path, tree: ast.AST) -> list[Violation]:
    out: list[Violation] = []
    fname = _rel(path)
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
            out.append(Violation(
                fname, node.lineno, "csv.reader(...)",
                "Read the frame through provider.get(name); csv.reader "
                "on a file handle reads the work folder and reintroduces "
                "the seed-funnel bug class.",
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
    if fname == PROVIDER_IMPL_FILE:
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
    fname = _rel(path)
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
        if any(_EXEMPT_RE in line for line in ctx_lines):
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
    text = path.read_text(encoding="utf-8")  # source is UTF-8; Windows defaults to cp1252
    return text, ast.parse(text, filename=str(path))


def test_no_disk_csv_reads_in_cascade() -> None:
    """Rule 1: no bare disk CSV reads in cascade modules."""
    violations: list[Violation] = []
    for path in _iter_cascade_files():
        if _rel(path) in RULE_1_ALLOWLIST:
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
            "(Provider impl, --csv-dump snapshot writer), "
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
        if _rel(path) in PROVIDER_IMPL_ALLOWLIST:
            continue
        _, tree = _read_tree(path)
        violations.extend(_detect_rule2_lru_cache(tree, _rel(path)))
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
        if _rel(path) in PROVIDER_IMPL_ALLOWLIST:
            continue
        _, tree = _read_tree(path)
        violations.extend(_detect_rule2_module_globals(tree, _rel(path)))
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


def test_scan_covers_the_engine_package() -> None:
    """Sanity: the recursive walk finds the cascade modules.

    Guards against the scan silently covering nothing — every rule test
    above passes vacuously on an empty file list.
    """
    scanned = list(_iter_cascade_files())
    assert len(scanned) > MIN_SCANNED_FILES, (
        f"only {len(scanned)} cascade files scanned under {CASCADE_ROOT} "
        f"(floor {MIN_SCANNED_FILES}); _iter_cascade_files is broken or "
        "the package moved."
    )


def test_allowlists_cover_only_existing_files() -> None:
    """Sanity: every allowlisted filename must exist on disk.

    Catches stale allowlist entries left behind after refactors.
    """
    on_disk = {_rel(p) for p in _iter_cascade_files()}
    stale = [
        name for name in (
            [PROVIDER_IMPL_FILE]
            + list(PROVIDER_IMPL_ALLOWLIST | RAW_INPUT_FALLBACK_ALLOWLIST)
        )
        if name not in on_disk
    ]
    if stale:
        pytest.fail(
            "Stale allowlist entries (file no longer exists in "
            f"flextool/engine/): {stale}.  Remove or update the "
            "relative path in "
            "PROVIDER_IMPL_FILE / PROVIDER_IMPL_ALLOWLIST / "
            "RAW_INPUT_FALLBACK_ALLOWLIST."
        )


# ---------------------------------------------------------------------------
# Rule 4 — the loader and the model have no disk arms (c24 stage 3)
# ---------------------------------------------------------------------------

# (a) Deleted disk helpers: a reference anywhere in cascade code brings a work-folder read back.
RULE_4_BANNED_NAMES: frozenset[str] = frozenset({
    "read_csv_fallback",          # provider-miss → read the CSV at the path
    "seed_provider_from_dir",     # fill a provider from the work folder's CSVs
    "workdir_provider_for_paths",  # provider built from the work folder for p_profile_value_lf
    "_read_csv_file",             # the plain CSV read behind CsvSource.get
    "_install_csv_cache",         # SolveContext's per-load CSV cache
    "_find_scenario",             # scenario discovery from a tests.sqlite in the work folder
    "_FIND_SCENARIO_OVERRIDES",
    "_auto_construct_db_reader",  # SpineDbReader built from a tests.sqlite in the work folder
    "_provider_open",             # provider-first, disk-second file handle
    "_recourse_usable_ctx",       # snapshot-reload guard of the recourse walk
})

# Top-level folders of CASCADE_ROOT that (b) applies to.
RULE_4_DISK_SCOPE: frozenset[str] = frozenset({"preprocess", "model"})

# Files Rule 4 does not apply to (POSIX path relative to CASCADE_ROOT → reason).
# ``test_rule_4_exemptions_are_needed`` fails when an entry no longer has a hit.
RULE_4_EXEMPT_FILES: dict[str, str] = {
    "input/_input_source.py": (
        "(a) defines _read_csv_file, the plain read behind CsvSource.get; no main-path caller."
    ),
    "preprocess/_dump_stamp.py": (
        "(b) reads only the dump stamp _dump_meta.json: write_dump_stamp merges an existing stamp, and "
        "_csv_reload checks it before a reload. Never a model input."
    ),
}

# (b) Legitimate reads: (file, literal path of the open() call) → reason.  A literal path, not a line number,
# so the entry survives edits; ``test_rule_4_read_allowlist_is_used`` fails on an unused entry.
RULE_4_READ_ALLOWLIST: dict[tuple[str, str], str] = {
    ("preprocess/flex_data.py", "/proc/self/status"): (
        "_loadflex_prof: RSS probe of the running process, only with FLEXTOOL_LOADFLEX_PROFILE=1; "
        "not the work folder."
    ),
    ("model/core.py", "/proc/self/status"): (
        "_build_prof: RSS probe of the running process, only with FLEXTOOL_BUILD_PROFILE=1; "
        "not the work folder."
    ),
}

# Method calls that probe or read the file system (Path.exists, os.path.isfile, Path.read_text, Path.glob, ...).
_DISK_PROBE_ATTRS: frozenset[str] = frozenset({
    "exists", "is_file", "is_dir", "is_symlink", "isfile", "isdir",
    "read_text", "read_bytes", "iterdir", "glob", "rglob",
})
# ``<module>.<function>`` file readers, directory listings and stat probes.  Polars is matched under both of its
# import names; pandas file readers (every ``read_*``) are matched separately below.
_POLARS_FILE_READERS: frozenset[str] = frozenset({
    "read_parquet", "scan_parquet", "read_ipc", "scan_ipc", "read_json", "read_ndjson", "scan_ndjson",
    "read_csv", "scan_csv", "read_csv_batched", "read_excel",
})
_DISK_READ_FUNCS: frozenset[tuple[str, str]] = frozenset(
    {(module, func) for module in ("pl", "polars") for func in _POLARS_FILE_READERS}
    | {
        ("json", "load"), ("pickle", "load"),
        ("os", "listdir"), ("os", "scandir"), ("os", "walk"), ("os", "stat"), ("os", "lstat"),
    }
)
_PANDAS_MODULE_NAMES: frozenset[str] = frozenset({"pd", "pandas"})
# Modules whose ``open(path, mode)`` takes the path first, like the builtin.
_PATH_FIRST_OPEN_MODULES: frozenset[str] = frozenset({"builtins", "io", "codecs", "gzip", "bz2", "lzma"})


def _rule_4_banned_references(tree: ast.AST, fname: str) -> list[Violation]:
    """Rule 4(a): every name, attribute, import or definition of a deleted disk helper."""
    hint = ("deleted disk helper (c24 stage 3): read the frame from the sub-solve's FlexDataProvider; "
            "tests use tests/_support/provider.py")
    out: list[Violation] = []
    for node in ast.walk(tree):
        hits: list[str] = []
        if isinstance(node, ast.Name):
            hits = [node.id]
        elif isinstance(node, ast.Attribute):
            hits = [node.attr]
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            hits = [alias.name.rsplit(".", 1)[-1] for alias in node.names]
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            hits = [node.name]
        out.extend(Violation(fname, node.lineno, f"reference to {name}", hint)
                   for name in hits if name in RULE_4_BANNED_NAMES)
    return out


def _open_mode_and_path(call: ast.Call) -> tuple[bool, str | None] | None:
    """For an ``open`` call: (is it a read, literal path or None); None when *call* is not an ``open``."""
    func = call.func
    if isinstance(func, ast.Name) and func.id == "open":
        path_first = True
    elif isinstance(func, ast.Attribute) and func.attr == "open":
        path_first = isinstance(func.value, ast.Name) and func.value.id in _PATH_FIRST_OPEN_MODULES
    else:
        return None
    mode_pos = 1 if path_first else 0
    mode_node = next((kw.value for kw in call.keywords if kw.arg == "mode"), None)
    if mode_node is None and len(call.args) > mode_pos:
        mode_node = call.args[mode_pos]
    if mode_node is None:
        is_read = True  # default mode "r"
    elif isinstance(mode_node, ast.Constant) and isinstance(mode_node.value, str):
        is_read = "r" in mode_node.value or "+" in mode_node.value
    else:
        is_read = True  # a computed mode cannot be shown to be a write
    path = None
    if path_first and call.args and isinstance(call.args[0], ast.Constant) and isinstance(call.args[0].value, str):
        path = call.args[0].value
    return is_read, path


def _rule_4_disk_calls(tree: ast.AST, fname: str) -> tuple[list[Violation], set[str]]:
    """Rule 4(b): file-system probes and reads; also the allow-listed literal paths that were seen."""
    hint = ("the loader and the model read only the FlexDataProvider; if this read is not of the work folder, "
            "add (file, literal path) to RULE_4_READ_ALLOWLIST with a reason")
    out: list[Violation] = []
    allowed_seen: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        opened = _open_mode_and_path(node)
        if opened is not None:
            is_read, path = opened
            if not is_read:
                continue
            if path is not None and (fname, path) in RULE_4_READ_ALLOWLIST:
                allowed_seen.add(path)
                continue
            shown = f"open({path!r})" if path is not None else "open(...)"
            out.append(Violation(fname, node.lineno, f"{shown} in a read mode", hint))
        elif isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name) and (
            (func.value.id, func.attr) in _DISK_READ_FUNCS
            or (func.value.id in _PANDAS_MODULE_NAMES and func.attr.startswith("read_"))
        ):
            out.append(Violation(fname, node.lineno, f"{func.value.id}.{func.attr}(...)", hint))
        elif isinstance(func, ast.Attribute) and func.attr in _DISK_PROBE_ATTRS:
            out.append(Violation(fname, node.lineno, f".{func.attr}(...)", hint))
    return out, allowed_seen


def _rule_4_violations(source: str, fname: str) -> tuple[list[Violation], set[str]]:
    """Both parts of Rule 4 for one file, ignoring RULE_4_EXEMPT_FILES (the callers apply it)."""
    tree = ast.parse(source)
    violations = _rule_4_banned_references(tree, fname)
    allowed_seen: set[str] = set()
    if fname.split("/", 1)[0] in RULE_4_DISK_SCOPE:
        disk, allowed_seen = _rule_4_disk_calls(tree, fname)
        violations.extend(disk)
    return violations, allowed_seen


def test_loader_and_model_have_no_disk_arms() -> None:
    """Rule 4: no deleted disk helper anywhere in cascade code, and no disk probe or read in ``preprocess/`` and
    ``model/`` (the main loader is provider-only; c24 stage 3)."""
    violations: list[Violation] = []
    for path in _iter_cascade_files():
        fname = _rel(path)
        if fname in RULE_4_EXEMPT_FILES:
            continue
        source, _ = _read_tree(path)
        violations.extend(_rule_4_violations(source, fname)[0])
    if violations:
        pytest.fail(
            "Disk arm in the loader or the model: nothing in a run may read input/ or solve_data/ back from the "
            "work folder (a dump or stale CSVs in a reused folder would change the run).\n\n"
            + "\n".join(v.format() for v in violations)
        )


def test_rule_4_exemptions_are_needed() -> None:
    """Every RULE_4_EXEMPT_FILES entry exists and still has a Rule 4 hit (so the exemption is not stale)."""
    files = {_rel(p): p for p in _iter_cascade_files()}
    stale = []
    for fname in RULE_4_EXEMPT_FILES:
        if fname not in files:
            stale.append(f"{fname} (no such file)")
        elif not _rule_4_violations(_read_tree(files[fname])[0], fname)[0]:
            stale.append(f"{fname} (no Rule 4 hit left)")
    assert not stale, f"stale RULE_4_EXEMPT_FILES entries: {stale}"


def test_rule_4_read_allowlist_is_used() -> None:
    """Every RULE_4_READ_ALLOWLIST (file, path) entry matches an ``open`` call in that file."""
    files = {_rel(p): p for p in _iter_cascade_files()}
    seen: set[tuple[str, str]] = set()
    for fname in {f for f, _ in RULE_4_READ_ALLOWLIST}:
        if fname in files:
            seen |= {(fname, p) for p in _rule_4_violations(_read_tree(files[fname])[0], fname)[1]}
    unused = sorted(set(RULE_4_READ_ALLOWLIST) - seen)
    assert not unused, f"RULE_4_READ_ALLOWLIST entries with no matching open() call: {unused}"


@pytest.mark.parametrize("source,fname", [
    # (a) anywhere in cascade code
    ("from flextool.engine.input._input_source import read_csv_fallback", "output/_writer.py"),
    ("from flextool.engine.input import _input_source\n_input_source.seed_provider_from_dir(p)", "solve/_x.py"),
    ("workdir_provider_for_paths(paths)", "preprocess/params/_profiles.py"),
    ("import flextool.engine.input._input_source._read_csv_file", "solve/_x.py"),
    ("def _install_csv_cache(cache):\n    pass", "preprocess/_solve_context.py"),
    ("x = _find_scenario", "preprocess/flex_data.py"),
    # (b) in preprocess/ and model/
    ("p.exists()", "preprocess/flex_data.py"),
    ("p.is_file()", "model/_reserve.py"),
    ("p.is_dir()", "preprocess/emit/_blocks.py"),
    ("p.is_symlink()", "model/core.py"),
    ("os.path.exists(p)", "preprocess/params/_derived.py"),
    ("os.path.isfile(p)", "model/core.py"),
    ("os.path.isdir(p)", "preprocess/_solve_context.py"),
    ("open(p)", "preprocess/flex_data.py"),
    ("open(p, 'r')", "model/core.py"),
    ("open(p, mode='rb')", "preprocess/flex_data.py"),
    ("open(p, 'r+')", "preprocess/flex_data.py"),
    ("open(p, m)", "preprocess/flex_data.py"),
    ("p.open()", "preprocess/emit/_blocks.py"),
    ("p.open('r', encoding='utf-8')", "preprocess/emit/_blocks.py"),
    ("gzip.open(p, 'rt')", "model/core.py"),
    ("p.read_text()", "model/core.py"),
    ("p.read_bytes()", "preprocess/flex_data.py"),
    ("pl.read_parquet(p)", "preprocess/flex_data.py"),
    ("pl.scan_parquet(p)", "model/core.py"),
    ("pl.read_excel(p)", "preprocess/flex_data.py"),
    ("pl.read_csv(p)", "model/core.py"),
    ("polars.read_parquet(p)", "preprocess/flex_data.py"),
    ("polars.scan_csv(p)", "model/core.py"),
    ("pd.read_csv(p)", "preprocess/emit/_blocks.py"),
    ("pd.read_excel(p)", "model/core.py"),
    ("pandas.read_parquet(p)", "preprocess/flex_data.py"),
    ("os.stat(p)", "preprocess/flex_data.py"),
    ("os.lstat(p)", "model/core.py"),
    ("json.load(fh)", "preprocess/flex_data.py"),
    ("os.listdir(d)", "preprocess/flex_data.py"),
    ("d.glob('*.csv')", "preprocess/flex_data.py"),
    ("d.iterdir()", "model/core.py"),
    # an allow-listed literal path in another file, or another path in an allow-listed file
    ("open('/proc/self/status')", "model/_reserve.py"),
    ("open('/proc/self/maps')", "model/core.py"),
])
def test_rule_4_detector_flags_every_pattern(source: str, fname: str) -> None:
    """Self-check: Rule 4 flags each banned name and each disk-read form exactly once."""
    assert len(_rule_4_violations(source, fname)[0]) == 1, source


@pytest.mark.parametrize("source,fname", [
    ("open('/proc/self/status')", "model/core.py"),                 # RULE_4_READ_ALLOWLIST
    ("open('/proc/self/status')", "preprocess/flex_data.py"),       # RULE_4_READ_ALLOWLIST
    ("p.open('a', encoding='utf-8')", "preprocess/_provider_translators.py"),  # append
    ("open(p, 'w')", "preprocess/flex_data.py"),
    ("open(p, mode='wb')", "model/core.py"),
    ("p.write_text('x')", "preprocess/flex_data.py"),
    ("p.mkdir(parents=True, exist_ok=True)", "preprocess/flex_data.py"),
    ("p.exists()", "solve/_orchestration.py"),                       # (b) covers preprocess/ and model/ only
    ("json.loads(s)", "preprocess/flex_data.py"),
    ("pd.DataFrame(rows)", "preprocess/flex_data.py"),            # pandas, but not a file read
    ("pl.read_database(q, conn)", "preprocess/flex_data.py"),    # a DB query, not the work folder
    ("frame.stat()", "model/core.py"),                            # only os.stat / os.lstat are stat probes
    ("provider.get('solve_data/process_exact')", "preprocess/flex_data.py"),
    ('"""read_csv_fallback was deleted"""', "preprocess/flex_data.py"),  # docstrings are not references
])
def test_rule_4_detector_ignores_allowed_forms(source: str, fname: str) -> None:
    """Self-check: allow-listed reads, writes, other folders, string mentions and provider reads are not flagged."""
    assert _rule_4_violations(source, fname)[0] == [], source


# ---------------------------------------------------------------------------
# Leaf rule — nothing outside ``flextool/engine/_csv_reload/`` imports it
# ---------------------------------------------------------------------------
#
# The CSV reload package is unsupported and not kept in sync with the
# engine.  It may import the engine; the engine, the tools and the default
# test suite must not import it (a normal test importing it would make the
# default gate depend on unsupported code).  Its own tests live in
# ``tests/csv_reload/`` and are the only exemption.

REPO_ROOT = Path(__file__).resolve().parents[2]
CSV_RELOAD_MODULE = "flextool.engine._csv_reload"

# (scan root, sub-folder of it that is exempt, minimum files the walk must see)
LEAF_SCAN_ROOTS: tuple[tuple[str, str, int], ...] = (
    ("flextool", "flextool/engine/_csv_reload", 250),
    ("tools", "", 1),
    ("tests", "tests/csv_reload", 400),
)


def _module_package(path: Path) -> list[str]:
    """Package parts of the module at *path* (repo-relative), for resolving relative imports."""
    parts = list(path.relative_to(REPO_ROOT).with_suffix("").parts)
    return parts[:-1]  # ``pkg/__init__.py`` and ``pkg/mod.py`` both resolve ``.`` to ``pkg``


def _names_csv_reload(module: str) -> bool:
    return module == CSV_RELOAD_MODULE or module.startswith(CSV_RELOAD_MODULE + ".")


def _csv_reload_imports(source: str, package: list[str]) -> list[tuple[int, str]]:
    """(line, statement) of every import of the CSV reload package in *source*.

    Detects ``import flextool.engine._csv_reload[...]``, ``from
    flextool.engine._csv_reload[...] import ...``, ``from flextool.engine
    import _csv_reload``, relative ``from`` imports resolved against
    *package*, and string module names passed to ``importlib.import_module``
    / ``__import__`` (``import_module`` with ``package=`` resolves a
    relative name).
    """
    hits: list[tuple[int, str]] = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if _names_csv_reload(alias.name):
                    hits.append((node.lineno, f"import {alias.name}"))
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                base = package[: len(package) - (node.level - 1)] if node.level - 1 <= len(package) else []
                module = ".".join(base + ([node.module] if node.module else []))
            else:
                module = node.module or ""
            if _names_csv_reload(module) or any(
                _names_csv_reload(f"{module}.{alias.name}") for alias in node.names
            ):
                hits.append((node.lineno, f"from {'.' * node.level}{node.module or ''} import ..."))
        elif isinstance(node, ast.Call):
            func = node.func
            name = func.attr if isinstance(func, ast.Attribute) else func.id if isinstance(func, ast.Name) else None
            if name not in ("import_module", "__import__") or not node.args:
                continue
            arg = node.args[0]
            if not (isinstance(arg, ast.Constant) and isinstance(arg.value, str)):
                continue
            target = arg.value
            if target.startswith("."):
                pkg = next((kw.value.value for kw in node.keywords
                            if kw.arg == "package" and isinstance(kw.value, ast.Constant)
                            and isinstance(kw.value.value, str)), None)
                if pkg is None:
                    continue
                dots = len(target) - len(target.lstrip("."))
                base = pkg.split(".")
                base = base[: len(base) - (dots - 1)]
                target = ".".join(base + [target.lstrip(".")])
            if _names_csv_reload(target):
                hits.append((node.lineno, f"{name}({arg.value!r})"))
    return hits


def _iter_leaf_scan_files():
    for root, exempt, _minimum in LEAF_SCAN_ROOTS:
        exempt_dir = (REPO_ROOT / exempt) if exempt else None
        for p in sorted((REPO_ROOT / root).rglob("*.py")):
            if exempt_dir is not None and exempt_dir in p.parents:
                continue
            yield root, p


def test_csv_reload_is_a_leaf() -> None:
    """No module outside ``flextool/engine/_csv_reload/`` (and ``tests/csv_reload/``) imports the package.

    Scans ``flextool/``, ``tools/`` and ``tests/``.  Every file must parse,
    and each root must yield at least its minimum file count, so a broken
    walk cannot pass silently.
    """
    counts = {root: 0 for root, _exempt, _minimum in LEAF_SCAN_ROOTS}
    violations: list[str] = []
    for root, path in _iter_leaf_scan_files():
        counts[root] += 1
        source = path.read_text(encoding="utf-8")
        for line, stmt in _csv_reload_imports(source, _module_package(path)):
            violations.append(f"{path.relative_to(REPO_ROOT).as_posix()}:{line}: {stmt}")
    too_few = {root: counts[root] for root, _exempt, minimum in LEAF_SCAN_ROOTS if counts[root] < minimum}
    assert not too_few, f"leaf scan saw too few files (the walk is broken or the tree moved): {too_few}"
    assert not violations, (
        "flextool.engine._csv_reload (unsupported CSV reload) is imported from outside the package; only the "
        "package itself and tests/csv_reload/ may import it:\n" + "\n".join(violations)
    )


@pytest.mark.parametrize("source,package", [
    ("import flextool.engine._csv_reload", ["flextool", "engine"]),
    ("import flextool.engine._csv_reload.load as l", ["tests"]),
    ("from flextool.engine._csv_reload import load_from_dump", ["tools"]),
    ("from flextool.engine._csv_reload.load import load_from_dump", ["tests", "engine"]),
    ("from flextool.engine import _csv_reload", ["flextool", "output"]),
    ("from . import _csv_reload", ["flextool", "engine"]),
    ("from ._csv_reload import load_from_dump", ["flextool", "engine"]),
    ("from .._csv_reload.load import load_from_dump", ["flextool", "engine", "output"]),
    ("from ... import _csv_reload", ["flextool", "engine", "preprocess", "emit"]),
    ("def f():\n    from flextool.engine._csv_reload import check_dump", ["flextool"]),
    ("import importlib\nimportlib.import_module('flextool.engine._csv_reload')", ["tests"]),
    ("from importlib import import_module\nimport_module('flextool.engine._csv_reload._stamp')", ["tools"]),
    ("import importlib\nimportlib.import_module('._csv_reload', package='flextool.engine')", ["tests"]),
    ("__import__('flextool.engine._csv_reload.load')", ["flextool"]),
])
def test_csv_reload_leaf_detector_flags_every_import_form(source: str, package: list[str]) -> None:
    """Self-check: the leaf detector flags each import form (so a broken detector cannot pass silently)."""
    assert len(_csv_reload_imports(source, package)) == 1, source


@pytest.mark.parametrize("source,package", [
    ("from flextool.engine import _dump_csvs", ["flextool", "engine"]),
    ("import flextool.engine._csv_reloader", ["tests"]),
    ("from . import _csv_reload", ["flextool", "process_outputs"]),
    ('"""See flextool.engine._csv_reload for the reload."""', ["flextool"]),
    ("x = 'flextool.engine._csv_reload'", ["tests"]),
])
def test_csv_reload_leaf_detector_ignores_non_imports(source: str, package: list[str]) -> None:
    """Self-check: docstrings, other modules and same-named siblings elsewhere are not flagged."""
    assert _csv_reload_imports(source, package) == [], source


# ---------------------------------------------------------------------------
# Rule 5 — the output writers read no work-folder inputs (item 163, 163-6)
# ---------------------------------------------------------------------------
#
# design_163.md §6: after item 163, these six files read ONLY the sub-solve's FlexData /
# FlexDataProvider / OutputWriterState — never ``input/`` or ``solve_data/`` back from the work
# folder. Two of them (``process_outputs/*``) live outside ``flextool/engine/`` (CASCADE_ROOT
# above), so Rule 5 uses its own file list, given as paths relative to the REPO root.
#
# (a) No call of ``pd.read_csv``, ``pl.read_csv`` or ``pl.scan_csv`` anywhere in these files.
# (b)/(c) No banned literal path segment (``"input"``, ``"solve_data"``,
#     ``"group_entity_invest.csv"``, ``"solve__p_entity_pre_existing.csv"``, ``"v_flow.csv"``,
#     ``"timeline_breaks.csv"``) is used as an argument of a ``/`` path join, ``Path(...)`` or
#     ``os.path.join(...)`` in a function that ALSO contains a disk-read call (``.exists()``,
#     ``.is_file()``, ``.read_text()``, ``open(`` in a read mode, or a read_csv/scan_csv call).
#
# design_163.md §6's own text offers a "simpler" shortcut for (b) — "no such constants anywhere
# in the file" — but that is NOT what's implemented here: ``read_highs_solution.py`` and
# ``write_outputs.py`` legitimately keep bare ``"solve_data"`` / ``"input"`` literals as
# PROVIDER-KEY SELECTORS (``_provider_lookup``, ``_provider_lookup_df`` — e.g.
# ``Path(work_folder) / "solve_data" / "enable_optional_outputs.csv"`` passed to
# ``_provider_lookup``, never opened), landed correctly in 163-1..163-3. A blanket per-file ban on
# the bare literal would false-positive on that already-correct code. The co-occurrence form
# below is the rule's actual intent — catch a real disk READ, not a key-selector literal — and it
# passes cleanly on the current tree with zero exemptions needed.

RULE_5_ROOT = Path(__file__).resolve().parents[2]

RULE_5_FILES: tuple[str, ...] = (
    "flextool/process_outputs/read_highs_solution.py",
    "flextool/process_outputs/handoff_writers.py",
    "flextool/process_outputs/read_variables.py",
    "flextool/process_outputs/solve_order.py",
    "flextool/process_outputs/union_realized_slice.py",
    "flextool/process_outputs/write_outputs.py",
    "flextool/engine/output/_writer.py",
)

RULE_5_BANNED_SEGMENTS: frozenset[str] = frozenset({
    "input", "solve_data", "group_entity_invest.csv",
    "solve__p_entity_pre_existing.csv", "v_flow.csv", "timeline_breaks.csv",
})

_RULE_5_READ_CSV_FUNCS: frozenset[tuple[str, str]] = frozenset({
    ("pd", "read_csv"), ("pl", "read_csv"), ("pl", "scan_csv"),
})


def _rule_5_read_csv_calls(tree: ast.AST) -> list[Violation]:
    """(a): every ``pd.read_csv`` / ``pl.read_csv`` / ``pl.scan_csv`` call, anywhere."""
    out: list[Violation] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        f = node.func
        if (
            isinstance(f, ast.Attribute) and isinstance(f.value, ast.Name)
            and (f.value.id, f.attr) in _RULE_5_READ_CSV_FUNCS
        ):
            out.append(Violation(
                "", node.lineno, f"{f.value.id}.{f.attr}(...)",
                "the output writers read only flex_data / provider / OutputWriterState; "
                "route this read through provider.get(...) or delete the dead arm",
            ))
    return out


def _rule_5_is_read_csv_call(node: ast.AST) -> bool:
    if not isinstance(node, ast.Call):
        return False
    f = node.func
    return (
        isinstance(f, ast.Attribute) and isinstance(f.value, ast.Name)
        and (f.value.id, f.attr) in _RULE_5_READ_CSV_FUNCS
    )


def _rule_5_is_disk_read_call(node: ast.AST) -> bool:
    """``.exists()``, ``.is_file()``, ``.read_text()``, ``open(`` in a read mode, or read_csv."""
    if not isinstance(node, ast.Call):
        return False
    if _rule_5_is_read_csv_call(node):
        return True
    f = node.func
    if isinstance(f, ast.Attribute) and f.attr in ("exists", "is_file", "read_text"):
        return True
    opened = _open_mode_and_path(node)
    return opened is not None and opened[0]


def _rule_5_banned_segments_in_path_context(node: ast.AST) -> list[str]:
    """A banned literal used as a ``/`` operand, or a ``Path(...)`` / ``os.path.join(...)`` arg."""

    def _segment(n: ast.AST) -> str | None:
        if (
            isinstance(n, ast.Constant) and isinstance(n.value, str)
            and n.value in RULE_5_BANNED_SEGMENTS
        ):
            return n.value
        return None

    hits: list[str] = []
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
        for side in (node.left, node.right):
            seg = _segment(side)
            if seg is not None:
                hits.append(seg)
    elif isinstance(node, ast.Call):
        f = node.func
        is_path_call = isinstance(f, ast.Name) and f.id == "Path"
        is_join_call = (
            isinstance(f, ast.Attribute) and f.attr == "join"
            and isinstance(f.value, ast.Attribute) and f.value.attr == "path"
        )
        if is_path_call or is_join_call:
            for arg in node.args:
                seg = _segment(arg)
                if seg is not None:
                    hits.append(seg)
    return hits


def _rule_5_function_violations(fn: "ast.FunctionDef | ast.AsyncFunctionDef", fname: str) -> list[Violation]:
    """(b)/(c): banned path segment co-occurring with a disk-read call, within one function."""
    segments: list[str] = []
    has_read = False
    for node in ast.walk(fn):
        segments.extend(_rule_5_banned_segments_in_path_context(node))
        if _rule_5_is_disk_read_call(node):
            has_read = True
    if segments and has_read:
        return [Violation(
            fname, fn.lineno,
            f"{fn.name}(...): path segment(s) {sorted(set(segments))} built alongside a disk-read call",
            "the output writers read only flex_data / provider / OutputWriterState; a bare path "
            "segment used only as a provider-key selector is fine, but not next to an actual "
            "disk-read call in the same function",
        )]
    return []


def _rule_5_violations(source: str, fname: str) -> list[Violation]:
    """Both parts of Rule 5 for one file's source."""
    tree = ast.parse(source)
    violations = [
        Violation(fname, v.line, v.pattern, v.hint) for v in _rule_5_read_csv_calls(tree)
    ]
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            violations.extend(_rule_5_function_violations(node, fname))
    return violations


def test_output_writers_read_no_work_folder_inputs() -> None:
    """Rule 5 (item 163, 163-6): the output-writer modules read no work-folder inputs.

    design_163.md §6: after item 163 these six files source everything from the sub-solve's
    FlexData / FlexDataProvider / OutputWriterState; a dump or stale CSVs in a reused work folder
    cannot change their output.
    """
    violations: list[Violation] = []
    for rel in RULE_5_FILES:
        path = RULE_5_ROOT / rel
        assert path.is_file(), f"Rule 5 scope file missing: {rel}"
        violations.extend(_rule_5_violations(path.read_text(encoding="utf-8"), rel))
    if violations:
        pytest.fail(
            "Output writer reads a work-folder input (item 163): outputs must read only "
            "flex_data / provider / OutputWriterState, never input/ or solve_data/ back from "
            "disk.\n\n" + "\n".join(v.format() for v in violations)
            + "\n\nSee specs/bugfix/c24/design_163.md §3.3/§3.4/§6 (Rule 5)."
        )


@pytest.mark.parametrize("source,fname", [
    # (a) an absolute ban, anywhere
    ("pd.read_csv(p)", "flextool/process_outputs/handoff_writers.py"),
    ("pl.read_csv(p)", "flextool/process_outputs/read_highs_solution.py"),
    ("pl.scan_csv(p)", "flextool/process_outputs/read_variables.py"),
    # (b)/(c) banned segment co-occurring with a disk read, in one function
    (
        "def f(wf):\n"
        "    p = Path(wf) / 'solve_data' / 'x.csv'\n"
        "    if p.exists():\n"
        "        return p.read_text()\n",
        "flextool/process_outputs/read_highs_solution.py",
    ),
    (
        "def g(wf):\n"
        "    p = os.path.join(wf, 'input', 'process_unit.csv')\n"
        "    with open(p) as f:\n"
        "        return f.read()\n",
        "flextool/process_outputs/handoff_writers.py",
    ),
    (
        "def h(wf):\n"
        "    p = Path(wf) / 'timeline_breaks.csv'\n"
        "    return p.exists()\n",
        "flextool/process_outputs/write_outputs.py",
    ),
])
def test_rule_5_detector_flags_every_pattern(source: str, fname: str) -> None:
    """Self-check: Rule 5 flags each absolute-ban call and each segment+read co-occurrence."""
    assert len(_rule_5_violations(source, fname)) == 1, source


@pytest.mark.parametrize("source,fname", [
    # a bare path-segment literal used ONLY as a provider-key selector (no disk read nearby) —
    # exactly the real ``output_horizon_enabled`` / ``_load_pre_existing`` shape.
    (
        "def f(wf, *, provider=None):\n"
        "    path = Path(wf) / 'solve_data' / 'enable_optional_outputs.csv'\n"
        "    return _provider_lookup(provider, path)\n",
        "flextool/process_outputs/read_highs_solution.py",
    ),
    # a disk read elsewhere in the file for something unrelated (not input/ or solve_data/)
    (
        "def f(p):\n"
        "    with open(p, 'r', encoding='utf-8') as fh:\n"
        "        return fh.read()\n",
        "flextool/process_outputs/write_outputs.py",
    ),
    # a write, not a read — the handoff mirrors' own out_path pattern
    (
        "def f(wf):\n"
        "    out_path = wf / 'solve_data' / 'p_entity_divested.csv'\n"
        "    with open(out_path, 'w', encoding='utf-8') as fh:\n"
        "        fh.write('x')\n",
        "flextool/process_outputs/handoff_writers.py",
    ),
    # the banned string as a plain call argument, not a /, Path(...) or os.path.join(...) operand
    (
        "def f(provider):\n"
        "    df = _provider_lookup_df(provider, 'solve_data', 'group_entity')\n"
        "    return df.exists()\n",
        "flextool/process_outputs/write_outputs.py",
    ),
    # a docstring / comment mention, not code
    (
        '"""Reads input/group_entity_invest.csv from disk — no, it does not."""\n'
        "def f(p):\n"
        "    return p.exists()\n",
        "flextool/process_outputs/read_variables.py",
    ),
])
def test_rule_5_detector_ignores_allowed_forms(source: str, fname: str) -> None:
    """Self-check: key-selector-only literals, unrelated reads, writes, plain call args and
    docstring mentions are not flagged."""
    assert _rule_5_violations(source, fname) == [], source


def test_rule_5_scope_files_exist() -> None:
    """Sanity: every Rule 5 scope file exists (catches a stale path after a rename/move)."""
    missing = [rel for rel in RULE_5_FILES if not (RULE_5_ROOT / rel).is_file()]
    assert not missing, f"Rule 5 scope files missing on disk: {missing}"
