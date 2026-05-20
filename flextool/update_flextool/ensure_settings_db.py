"""Self-healing creation of lightweight settings SQLite databases.

FlexTool accepts three small, user-editable SQLite databases as
command-line inputs: ``output_info.sqlite`` (output-location mapping),
``output_settings.sqlite`` (per-scenario output configuration), and
``comparison_settings.sqlite`` (scenario-comparison configuration).
These were previously created only by ``flextool-update``, which users
had to remember to run at least once.  On a fresh clone, forgetting
this leads to opaque "file not found" errors from ``spinedb_api``.

This module lets the main runtime paths seed them on demand from their
tracked JSON templates under ``version/``.  The logic is intentionally
conservative:

* Only file basenames in :data:`SETTINGS_TEMPLATES` are auto-seeded.
  Other ``*.sqlite`` paths the user passes are left alone.
* Existing files are never overwritten — user edits always persist.
* Parent directories are created as needed.

For heavier templates (``examples.sqlite``, ``results_template.sqlite``,
``example_input.xlsx``) the canonical path remains ``flextool-update``;
those require migrations that aren't appropriate to run on every solve.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

from .initialize_database import initialize_database


_LOGGER = logging.getLogger(__name__)


SETTINGS_TEMPLATES: dict[str, str] = {
    "output_info.sqlite": "output_info_template.json",
    "output_settings.sqlite": "output_settings_template.json",
    "comparison_settings.sqlite": "comparison_settings_template.json",
}
"""Mapping from settings-DB basename to the JSON template that seeds it.

Templates are looked up under ``flextool/version/`` via
``importlib.resources`` (see :func:`ensure_settings_db`).  Keys are
compared case-sensitively against ``Path.name`` of the target path, not
full paths — so a user-custom name like ``my_config.sqlite`` is
intentionally not auto-seeded.
"""


def _sqlite_url_to_path(sqlite_url: str, cwd: Optional[Path] = None) -> Optional[Path]:
    """Return the filesystem path for a ``sqlite://...`` URL, or ``None``
    if the input isn't a plain SQLite URL.

    Accepts both ``sqlite:///relative/path.sqlite`` and
    ``sqlite:////absolute/path.sqlite`` forms.  Unusual URL schemes
    (in-memory, mysql, remote) are not handled here.
    """
    if sqlite_url is None:
        return None
    parsed = urlparse(sqlite_url)
    if parsed.scheme and parsed.scheme != "sqlite":
        return None
    if parsed.scheme == "sqlite":
        # ``sqlite:///relative/path``  → parsed.path == '/relative/path'
        #                                (one leading slash from URL syntax;
        #                                 actual path is relative)
        # ``sqlite:////abs/path``      → parsed.path == '//abs/path'
        #                                (two leading slashes; path is absolute)
        raw = parsed.path or parsed.netloc
        if raw.startswith("//"):
            fs_path = Path(raw[1:])  # ``//abs/path`` → ``/abs/path``
        elif raw.startswith("/"):
            fs_path = Path(raw[1:])  # ``/relative/path`` → ``relative/path``
        else:
            fs_path = Path(raw)
    else:
        # Bare filesystem path (no scheme). Use as-is; do NOT strip a
        # leading slash — on POSIX that makes absolute paths relative.
        fs_path = Path(sqlite_url)
    if not fs_path.is_absolute() and cwd is not None:
        fs_path = cwd / fs_path
    return fs_path


def ensure_settings_db(
    target: str | Path,
    repo_root: Path | None = None,
    *,
    logger: Optional[logging.Logger] = None,
) -> Optional[Path]:
    """If ``target`` is a known settings DB (by basename) and does not
    yet exist, create it from the corresponding JSON template.

    ``target`` may be a filesystem path or an ``sqlite://...`` URL.
    The JSON template ships inside the ``flextool`` package
    (``flextool/version/<template>``); ``repo_root`` is accepted for
    backward compatibility but ignored — templates are always read from
    package data via :mod:`importlib.resources`.
    """
    log = logger or _LOGGER
    if target is None:
        return None
    if isinstance(target, str):
        path = _sqlite_url_to_path(target)
        if path is None:
            return None
    else:
        path = Path(target)

    template_name = SETTINGS_TEMPLATES.get(path.name)
    if template_name is None:
        return None
    if path.exists():
        return None

    from flextool._resources import package_data_path
    template_path = package_data_path(f"version/{template_name}")
    if not template_path.is_file():
        log.warning(
            "Cannot auto-seed %s: template %s is missing.",
            path, template_path,
        )
        return None

    path.parent.mkdir(parents=True, exist_ok=True)
    initialize_database(str(template_path), str(path))
    log.info("Seeded %s from %s", path, template_path)
    return path


__all__ = ["SETTINGS_TEMPLATES", "ensure_settings_db"]
