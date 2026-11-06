"""mkdocs hook (``mkdocs.yml`` ``hooks:``): add the pages generated from the code to the web docs.

``on_files`` renders the formulation page and the results catalogue with ``flextool.docs_build`` (the same
generator as ``python -m flextool.docs_build``) and adds them as generated files: they are rendered, indexed
and in the nav, and nothing is written into ``docs/``.  The pages describe the code of this checkout: its root
goes first on ``sys.path``, ahead of any other installed FlexTool.  When the local private-names list exists, a
page that matches it fails the build.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

from mkdocs.structure.files import File, Files

log = logging.getLogger("mkdocs.hooks.flextool")
REPO_ROOT = Path(__file__).resolve().parents[1]


def on_files(files: Files, config) -> Files:
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    from flextool.docs_build import check_documentation, default_private_names_path, render_pages
    from flextool.docs_build.catalogue import unmatched_bullets

    docs_dir = Path(config["docs_dir"])
    pages = render_pages(docs_dir)
    check_documentation(docs_dir, pages, default_private_names_path())
    for bullet in unmatched_bullets(docs_dir):
        log.warning("results.md (%s): names no result table: %s", bullet.topic, bullet.text)
    for name, content in pages.items():
        files.append(File.generated(config, name, content=content))
    return files
