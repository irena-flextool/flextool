"""Format stamp for CSV dump folders (``<folder>/_dump_meta.json``).

The two CSV dump writers, :func:`flextool.engine.output._dump_csvs.dump_csvs` (behind ``FlexData.dump_csvs``) and
:meth:`FlexDataProvider.snapshot_processed_inputs`, call :func:`write_dump_stamp` on the folder they wrote.  The
stamp records which dump layout and which engine wrote the folder, so a reader can refuse or warn on a dump it was not
written for.  The writers are supported; reading a dump back is not (that lives in ``flextool.engine._csv_reload``,
which this module must not import).

Stamp fields (``json.dumps(..., indent=1, sort_keys=True)``; no timestamp, so dump folders stay deterministic):

* ``dump_format`` — :data:`DUMP_FORMAT`.
* ``flextool`` — the installed FlexTool version (``"unknown"`` when it cannot be read).
* ``schema_version`` — ``FLEXTOOL_DB_VERSION`` of the running engine.
* ``git`` — the checkout's commit (``git describe --always --dirty``, 8+ hex digits, ``-dirty`` when the tree has
  uncommitted changes), or ``null`` when FlexTool does not run from a git checkout (a wheel install, a git worktree)
  or git fails.
* ``writers`` — sorted names of the writers that wrote into the folder (``"dump_csvs"``,
  ``"snapshot_processed_inputs"``).
* ``previous`` — only when the folder already held a stamp from another engine (any of ``dump_format``,
  ``flextool``, ``schema_version``, ``git`` differs): that older stamp, without its own ``previous``.  The folder then
  also holds files of the older dump; nothing cleans it.

Merge rule: an existing stamp from the same engine (the four identity fields equal) keeps its ``previous`` and gets
the new writer added to ``writers``; any other existing stamp is replaced and kept under ``previous``.  An existing
stamp that cannot be parsed is overwritten.

:data:`DUMP_FORMAT` changes only when the dump **layout convention** changes (folder names, key-to-file naming, this
stamp's schema), not when a FlexData field or a provider key is added.

The stamp is debug metadata: a failure to write it is logged at WARNING and never raises.
"""
from __future__ import annotations

import functools
import json
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

DUMP_FORMAT: int = 1
STAMP_FILE: str = "_dump_meta.json"

_IDENTITY_FIELDS: tuple[str, ...] = ("dump_format", "flextool", "schema_version", "git")


@functools.cache
def _engine_identity() -> dict[str, object]:
    """``{dump_format, flextool, schema_version, git}`` of the running engine, computed once per process.

    The writers run once per test snapshot and once per roll under ``--csv-dump``, so the git subprocess must not run
    on every call.
    """
    from flextool.update_flextool import FLEXTOOL_DB_VERSION
    from flextool.update_flextool.install_info import current_commit, flextool_version, git_checkout_root

    root = git_checkout_root()
    return {
        "dump_format": DUMP_FORMAT,
        "flextool": flextool_version(),
        "schema_version": FLEXTOOL_DB_VERSION,
        "git": current_commit(root) if root is not None else None,
    }


def read_dump_stamp(folder: Path | str) -> dict | None:
    """The stamp in ``folder``, or ``None`` when it is absent, unreadable or not a JSON object."""
    path = Path(folder) / STAMP_FILE
    try:
        stamp = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return stamp if isinstance(stamp, dict) else None


def _write_stamp_file(path: Path, text: str) -> None:
    """Write ``text`` to ``path`` through a sibling temp file, so a reader never sees a half-written stamp."""
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8", newline="\n")
    os.replace(tmp, path)


def write_dump_stamp(folder: Path | str, *, writer: str) -> Path:
    """Write or merge ``folder/_dump_meta.json`` for ``writer`` (see the module docstring); return its path.

    Never raises: a failure is logged at WARNING (the dump itself is debug output and is already on disk).
    """
    path = Path(folder) / STAMP_FILE
    try:
        identity = dict(_engine_identity())
        old = read_dump_stamp(folder)
        if old is not None and all(old.get(k) == identity[k] for k in _IDENTITY_FIELDS):
            writers = old.get("writers")
            writers = set(writers) if isinstance(writers, list) else set()
            stamp = {**identity, "writers": sorted(writers | {writer})}
            if "previous" in old:
                stamp["previous"] = old["previous"]
        else:
            stamp = {**identity, "writers": [writer]}
            if old is not None:
                stamp["previous"] = {k: v for k, v in old.items() if k != "previous"}
        _write_stamp_file(path, json.dumps(stamp, indent=1, sort_keys=True) + "\n")
    except Exception as exc:  # noqa: BLE001 — debug metadata must never fail the dump
        logger.warning("Could not write the CSV dump stamp %s (%s): %s", path, writer, exc)
    return path


__all__ = ["DUMP_FORMAT", "STAMP_FILE", "read_dump_stamp", "write_dump_stamp"]
