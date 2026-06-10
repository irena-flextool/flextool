"""Standalone HiGHS-from-MPS subprocess solver.

Invoked by FlexTool's ``--save-memory`` path to keep HiGHS' active-solve
memory footprint isolated from the parent Python process.  Reads an MPS
file written by ``polar_high.Problem.build_only``, runs HiGHS with the
caller's options, writes a solution file the parent reads back.

Intentionally imports nothing from the rest of FlexTool so the child
process's Python footprint stays small.  Only depends on ``highspy``.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from flextool.cli._console import run_tool


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Solve an MPS file with HiGHS in a subprocess.",
    )
    parser.add_argument("--mps", type=Path, required=True, help="Input MPS file")
    parser.add_argument(
        "--solution", type=Path, required=True,
        help="Output solution file (write_solution_style=0, pretty/sparse)",
    )
    parser.add_argument(
        "--basis", type=Path, default=None,
        help="Optional output basis file",
    )
    parser.add_argument(
        "--options", type=Path, default=None,
        help="Optional HiGHS options file (key=value per line)",
    )
    args = parser.parse_args()

    import highspy

    # ``kWarning`` is non-fatal — HiGHS issues it for things like
    # writeSolution-on-an-LP-without-ranging, options-file lines it
    # didn't recognise, etc.  Treat the API call as successful in that
    # case but surface the warning to stderr.
    def _ok(status: object, what: str) -> bool:
        if status == highspy.HighsStatus.kOk:
            return True
        if status == highspy.HighsStatus.kWarning:
            print(
                f"NOTICE: HiGHS {what} returned kWarning (continuing)",
                file=sys.stderr,
            )
            return True
        return False

    h = highspy.Highs()

    # Apply options BEFORE readModel so presolve / solver / scaling
    # decisions take effect on the read.  HiGHS' own .opt file format
    # is ``key=value`` per line; we use that as the IPC mechanism so
    # FlexTool doesn't need to keep its option-dict translation aligned
    # with this CLI separately.
    if args.options is not None:
        if not _ok(h.readOptions(str(args.options)), "readOptions"):
            print(
                f"ERROR: HiGHS rejected options file {args.options}",
                file=sys.stderr,
            )
            return 2

    if not _ok(h.readModel(str(args.mps)), "readModel"):
        print(f"ERROR: HiGHS failed to read MPS {args.mps}", file=sys.stderr)
        return 3

    if not _ok(h.run(), "run"):
        print("ERROR: HiGHS run() returned non-OK status", file=sys.stderr)
        return 4

    model_status = h.getModelStatus()
    # Write the solution unconditionally so the parent can inspect
    # whatever HiGHS produced (useful for debugging non-optimal runs).
    # Pretty/sparse format (style=0) — matches what the parent's
    # h.readSolution expects.
    if not _ok(h.writeSolution(str(args.solution), 0), "writeSolution"):
        print(
            f"ERROR: HiGHS failed to write solution {args.solution}",
            file=sys.stderr,
        )
        return 5

    if args.basis is not None:
        # Best-effort: basis write isn't critical for the parent's
        # output path, and small models may not have a basis on
        # interior-point solves.  Warn but don't fail.
        try:
            _ok(h.writeBasis(str(args.basis)), "writeBasis")
        except Exception as exc:
            print(f"WARNING: writeBasis raised: {exc}", file=sys.stderr)

    # Exit code carries the model status so the parent can branch on
    # optimality without parsing the solution file first.
    if model_status == highspy.HighsModelStatus.kOptimal:
        return 0
    print(
        f"NOTICE: HiGHS finished with non-optimal status={model_status}",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    run_tool(main)
