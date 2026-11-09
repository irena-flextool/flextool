"""``python -m flextool.docs_build --out DIR``: write the generated documentation pages into ``DIR``."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from flextool.docs_build import (
    PrivateNameError,
    default_private_names_path,
    write_pages,
)
from flextool.docs_build.catalogue import unmatched_bullets

PROG = "python -m flextool.docs_build"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog=PROG,
        description="Write the formulation page and the results catalogue, generated from the installed code.",
    )
    parser.add_argument("--out", type=Path, required=True, help="folder to write the pages into")
    parser.add_argument("--docs-dir", type=Path, default=None,
                        help="the Markdown documentation they are built with (default: docs/ of this clone, else "
                             "the copy in the installed package)")
    parser.add_argument("--private-names", type=Path, default=None,
                        help="list of private names to check the pages against (default: .private_names at the "
                             "repository root, skipped when absent)")
    args = parser.parse_args(argv)
    docs_dir = args.docs_dir
    if docs_dir is None:
        from flextool.assistant.docs_store import markdown_root

        docs_dir = markdown_root()
        if docs_dir is None:
            print(f"{PROG}: this installation has no documentation; give its folder with --docs-dir",
                  file=sys.stderr)
            return 1
    if not docs_dir.is_dir():
        print(f"{PROG}: the documentation folder {docs_dir} does not exist", file=sys.stderr)
        return 1
    private_names = args.private_names if args.private_names is not None else default_private_names_path()
    try:
        written = write_pages(args.out, docs_dir, private_names)
        unmatched = unmatched_bullets(docs_dir)
    except PrivateNameError as exc:
        print(exc, file=sys.stderr)
        return 1
    except (OSError, ValueError) as exc:        # a missing or unreadable page; HeaderGrammarError is a ValueError
        print(f"{PROG}: {exc}", file=sys.stderr)
        return 1
    for bullet in unmatched:
        print(f"results.md ({bullet.topic}): names no result table: {bullet.text}", file=sys.stderr)
    for path in written:
        print(path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
