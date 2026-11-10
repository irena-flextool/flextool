"""``python -m flextool.docs_build --out DIR`` writes the generated documentation pages into ``DIR``;
``--xlsx FILE [--input-db DB [--scenario NAME]]`` writes the parameter catalogue as an Excel data dictionary."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from flextool.docs_build import (
    PrivateNameError,
    default_private_names_path,
    write_pages,
)

PROG = "python -m flextool.docs_build"


def _write_pages(args: argparse.Namespace) -> int:
    from flextool.docs_build.catalogue import unmatched_bullets

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
    except OSError as exc:                      # a missing or unreadable page
        # The path as it is, not as Python quotes it (doubled backslashes on Windows).
        reason = f"{exc.strerror}: {exc.filename}" if exc.strerror and exc.filename else str(exc)
        print(f"{PROG}: {reason}", file=sys.stderr)
        return 1
    except ValueError as exc:                   # HeaderGrammarError is a ValueError
        print(f"{PROG}: {exc}", file=sys.stderr)
        return 1
    for bullet in unmatched:
        print(f"results.md ({bullet.topic}): names no result table: {bullet.text}", file=sys.stderr)
    for path in written:
        print(path)
    return 0


def _write_xlsx(args: argparse.Namespace) -> int:
    from flextool.docs_build.parameters import CatalogueInputError, database_usage, version_mismatch, write_xlsx

    usage = None
    if args.input_db is not None:
        try:
            usage = database_usage(args.input_db, args.scenario)
        except CatalogueInputError as exc:
            print(f"{PROG}: {exc}", file=sys.stderr)
            return 1
        note = version_mismatch(usage.version)
        if note:
            print(f"{PROG}: {note}", file=sys.stderr)
    try:
        path = write_xlsx(args.xlsx, usage, input_db=args.input_db, scenario=args.scenario)
    except OSError as exc:
        print(f"{PROG}: cannot write {args.xlsx}: {exc}", file=sys.stderr)
        return 1
    print(path)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog=PROG,
        description="Write the generated documentation pages (the formulation page, the results catalogue and the "
                    "parameter catalogue) and/or the parameter catalogue as an Excel data dictionary, from the "
                    "installed code.",
    )
    parser.add_argument("--out", type=Path, default=None, help="folder to write the pages into")
    parser.add_argument("--docs-dir", type=Path, default=None,
                        help="the Markdown documentation they are built with (default: docs/ of this clone, else "
                             "the copy in the installed package)")
    parser.add_argument("--private-names", type=Path, default=None,
                        help="list of private names to check the pages against (default: .private_names at the "
                             "repository root, skipped when absent)")
    parser.add_argument("--xlsx", type=Path, default=None,
                        help="write the parameter catalogue (data dictionary) to this Excel file")
    parser.add_argument("--input-db", default=None,
                        help="with --xlsx: an input database, as a file path (input_data.sqlite) or a URL "
                             "(sqlite:///input_data.sqlite); the parameters it gives values to are marked 'used in "
                             "DB'. The file is opened read-only")
    parser.add_argument("--scenario", default=None,
                        help="with --input-db: count only the alternatives of this scenario")
    args = parser.parse_args(argv)
    if args.out is None and args.xlsx is None:
        parser.error("give --out, --xlsx or both")
    if args.input_db is not None and args.xlsx is None:
        parser.error("--input-db needs --xlsx")
    if args.scenario is not None and args.input_db is None:
        parser.error("--scenario needs --input-db")
    if args.out is not None:
        code = _write_pages(args)
        if code:
            return code
    if args.xlsx is not None:
        return _write_xlsx(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
