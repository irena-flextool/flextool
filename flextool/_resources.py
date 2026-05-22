"""Locate bundled FlexTool package data files.

Static FlexTool resources — schema JSONs, YAML/text templates, and
canonical-database sources (all under ``flextool/schemas/``), plus the HiGHS
options template (``flextool/bin/``) — must be reached through
:mod:`importlib.resources` rather than ``Path(__file__).resolve()``
walks.  After ``pip install flextool`` the package is the only thing
on ``sys.path``; the historical layout where these dirs sat at the
*repo root* alongside the package no longer exists.

Two helpers are exposed:

``package_data_path(relative)``
    Returns a :class:`pathlib.Path` to a resource inside the
    ``flextool`` package, e.g.
    ``package_data_path("schemas/default_plots.yaml")``.
    Editable installs return the real on-disk path; wheel installs
    return the unpacked site-packages path (importlib.resources
    materialises zipped data on demand, but we never ship zipped so
    in practice this is always a stable filesystem path).

``package_data_text(relative)``
    Convenience wrapper that returns the file contents as a string —
    handy when the caller doesn't actually need a path (YAML/JSON
    parsing).
"""

from __future__ import annotations

from importlib import resources
from pathlib import Path

import flextool


def package_data_path(relative: str) -> Path:
    """Path to ``flextool/<relative>`` inside the installed package."""
    return Path(resources.files(flextool).joinpath(relative))


def package_data_text(relative: str) -> str:
    """Text contents of ``flextool/<relative>``."""
    return resources.files(flextool).joinpath(relative).read_text(encoding="utf-8")
