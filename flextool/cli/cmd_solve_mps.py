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
import json
import sys
from pathlib import Path

from flextool.cli._console import run_tool


def apply_option_pairs(h: object, path: Path, ok) -> str | None:
    """Set the HiGHS options of the JSON file *path* (``[[name, value text], ...]``) on *h* in order; ``None`` when
    all were taken, else what was refused (the first option HiGHS rejects stops the rest).

    The same outcome as ``readOptions`` on the ``key=value`` file of the same pairs, which this replaces for
    FlexTool's ``--save-memory`` solve (item 240 B4): name and value are trimmed as that reader does, and an
    empty value is refused as that reader refuses it (``setOptionValue`` would take an empty text option).
    *ok(status, what)* treats ``kWarning`` as taken, as for the other HiGHS calls here.
    """
    try:
        pairs = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return f"it cannot be read ({exc})"
    if not isinstance(pairs, list) or not all(
        isinstance(p, list) and len(p) == 2 and all(isinstance(x, str) for x in p) for p in pairs
    ):
        return "it is not a list of [name, value] pairs"
    for raw_name, raw_value in pairs:
        name, value = raw_name.strip(), raw_value.strip()
        if not value:
            return f"option {name!r} has no value"
        if not ok(h.setOptionValue(name, value), f"setOptionValue({name})"):
            return f"option {name!r} = {value!r}"
    return None


#: HiGHS model statuses of a solve stopped by a limit or at an objective target or bound: the ones whose
#: primal feasibility the parent's acceptance check reads (it keeps such a MIP when its incumbent is feasible).
_LIMIT_MODEL_STATUS_NAMES = frozenset(
    {"kTimeLimit", "kSolutionLimit", "kIterationLimit", "kObjectiveTarget", "kObjectiveBound"}
)


def _option_value(h: object, name: str, default: object) -> object:
    """The value of HiGHS option *name* on *h* (``default`` when the HiGHS build does not know it)."""
    try:
        got = h.getOptionValue(name)
    except Exception:  # an older HiGHS without the option
        return default
    return got[1] if isinstance(got, tuple) else got


def solved_space_primal_feasible(h: object, highspy: object) -> bool | None:
    """Whether the primal solution on *h* is feasible in the user-scaled problem HiGHS solved: no column or row
    is further than the primal feasibility tolerance outside its bounds there.  ``None`` when HiGHS applied no
    user scaling (its own verdict is then in that space already) or there is no primal solution.

    HiGHS's ``user_bound_scale = N`` multiplies the bounds and values of the continuous columns and of every
    row by ``2**N``; after the solve it divides them back and judges feasibility again with the same tolerance,
    so with ``N < 0`` a residual within the tolerance in the solved problem can read as an infeasibility.
    Multiplying by a power of two is exact, so the solved-space residuals are rebuilt exactly from the
    unscaled model and solution, and tested as HiGHS counts an infeasibility (its ``infeasibility()``: outside
    the bound by more than the tolerance).  The in-process counterpart is polar-high's ``user_scaled_kkt_failures``; this
    child imports only highspy and numpy.
    """
    import numpy as np

    bound_scale = int(_option_value(h, "user_bound_scale", 0) or 0)
    objective_scale = int(_option_value(h, "user_objective_scale", 0) or 0)
    if bound_scale == 0 and objective_scale == 0:
        return None
    solution = h.getSolution()
    if not solution.value_valid:
        return None
    tolerance = float(_option_value(h, "primal_feasibility_tolerance", 1e-7))
    kkt_tolerance = float(_option_value(h, "kkt_tolerance", 1e-7))
    if kkt_tolerance != 1e-7:  # HiGHS: a non-default kkt_tolerance replaces the feasibility tolerances
        tolerance = kkt_tolerance
    lp = h.getLp()
    factor = 2.0**bound_scale
    col_factor = np.full(int(lp.num_col_), factor)
    if len(lp.integrality_):
        kinds = lp.integrality_
        semi = (highspy.HighsVarType.kSemiContinuous, highspy.HighsVarType.kSemiInteger)
        if any(t in semi for t in kinds):
            return None  # HiGHS judges semi-variables by their own rule; keep its verdict
        continuous = np.array([t == highspy.HighsVarType.kContinuous for t in kinds], dtype=bool)
        col_factor = np.where(continuous, factor, 1.0)
    for lower, upper, value, scale in (
        (lp.col_lower_, lp.col_upper_, solution.col_value, col_factor),
        (lp.row_lower_, lp.row_upper_, solution.row_value, factor),
    ):
        lo = np.asarray(lower, dtype=np.float64) * scale
        up = np.asarray(upper, dtype=np.float64) * scale
        val = np.asarray(value, dtype=np.float64) * scale
        outside = (val < lo - tolerance) | (val > up + tolerance)
        residual = np.where(val > up, val - up, np.where(val < lo, lo - val, 0.0))
        if np.any(outside & (residual > tolerance)):
            return False
    return True


def main(argv: list[str] | None = None) -> int:
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
        "--warm-basis", type=Path, default=None,
        help="Optional input basis file to warm-start from",
    )
    parser.add_argument(
        "--stats", type=Path, default=None,
        help="Optional output JSON file with solve measurement fields",
    )
    parser.add_argument(
        "--mip-gap", type=Path, default=None,
        help="Optional output text file with HiGHS' relative MIP gap (info.mip_gap; inf for an LP)",
    )
    parser.add_argument(
        "--primal-feasible", type=Path, default=None,
        help="Optional output text file: 1 or 0, whether a solve stopped by a limit has a primal solution "
             "feasible in the user-scaled problem HiGHS solved; written only when user scaling was applied",
    )
    options = parser.add_mutually_exclusive_group()
    options.add_argument(
        "--options", type=Path, default=None,
        help="Optional HiGHS options file (key=value per line)",
    )
    options.add_argument(
        "--options-json", type=Path, default=None,
        help="Optional HiGHS options as a JSON list of [name, value text] pairs, applied in order "
             "(what FlexTool's --save-memory solve passes)",
    )
    parser.add_argument(
        "--threads", type=int, default=None, metavar="N",
        help="Thread budget of the parent run; read before polars starts (flextool/_thread_budget.py).  "
             "HiGHS' own threads come from --options / --options-json",
    )
    args = parser.parse_args(argv)

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
    # decisions take effect on the read.  FlexTool passes the options as
    # (name, value text) pairs (``--options-json``), set one by one; a HiGHS
    # option file (``--options``, ``key=value`` per line) is read by HiGHS.
    if args.options is not None:
        if not _ok(h.readOptions(str(args.options)), "readOptions"):
            print(
                f"ERROR: HiGHS rejected options file {args.options}",
                file=sys.stderr,
            )
            return 2
    if args.options_json is not None:
        problem = apply_option_pairs(h, args.options_json, _ok)
        if problem is not None:
            print(f"ERROR: HiGHS rejected options file {args.options_json}: {problem}", file=sys.stderr)
            return 2

    if not _ok(h.readModel(str(args.mps)), "readModel"):
        print(f"ERROR: HiGHS failed to read MPS {args.mps}", file=sys.stderr)
        return 3

    # Warm-start injection.  A rejected / missing / corrupt basis is a
    # safe silent-cold fallback: HiGHS' run() is correct after a failed
    # readBasis, so we log and continue rather than abort the solve.
    warm_basis_used = False
    if args.warm_basis is not None:
        try:
            status = h.readBasis(str(args.warm_basis))
            if status == highspy.HighsStatus.kOk:
                warm_basis_used = True
                print(
                    f"INFO: warm-basis accepted from {args.warm_basis}",
                    file=sys.stderr,
                )
            else:
                print(
                    f"WARNING: warm-basis rejected ({status}) — solving cold",
                    file=sys.stderr,
                )
        except Exception as exc:
            print(
                f"WARNING: warm-basis read failed ({exc}) — solving cold",
                file=sys.stderr,
            )

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

    if args.mip_gap is not None:
        # The relative MIP gap the solve stopped at, for the parent's time-limit warning.  Best-effort: the
        # parent falls back to pointing at the solver log when the file is missing.
        try:
            args.mip_gap.write_text(repr(float(h.getInfo().mip_gap)))
        except Exception as exc:
            print(f"WARNING: mip-gap write failed: {exc}", file=sys.stderr)

    if args.primal_feasible is not None and model_status.name in _LIMIT_MODEL_STATUS_NAMES:
        # The solution file's Feasible / Infeasible line is HiGHS's verdict after removing user scaling;
        # the parent keeps a limit-stopped MIP on the verdict in the space HiGHS solved.  Best-effort: without
        # the file the parent uses the solution file's line.
        try:
            feasible = solved_space_primal_feasible(h, highspy)
            if feasible is not None:
                args.primal_feasible.write_text("1" if feasible else "0")
        except Exception as exc:
            print(f"WARNING: primal-feasible write failed: {exc}", file=sys.stderr)

    if args.basis is not None:
        # Best-effort: basis write isn't critical for the parent's
        # output path, and small models may not have a basis on
        # interior-point solves.  Warn but don't fail.
        try:
            _ok(h.writeBasis(str(args.basis)), "writeBasis")
        except Exception as exc:
            print(f"WARNING: writeBasis raised: {exc}", file=sys.stderr)

    if args.stats is not None:
        # Best-effort measurement sidecar for the parent's A/B gating.
        # A stats-write failure must never fail the solve.
        try:
            info = h.getInfo()
            # Wall-clock solve time drives the parent's A/B warm-vs-cold
            # gate.  Best-effort: a highspy build without ``getRunTime``
            # (or one that returns junk on a tiny model) falls back to
            # ``-1.0``, which the parent treats as "no timing → skip gate".
            try:
                run_time = float(h.getRunTime())
            except Exception:
                run_time = -1.0
            stats = {
                "simplex_iteration_count": int(info.simplex_iteration_count),
                "model_status": str(model_status),
                # The MPS objective: the file carries no objective constant
                # (``Problem.write_mps``); the parent's Solution adds
                # ``Problem._obj_offset`` back (``_set_objective_constant``).
                "objective": float(h.getObjectiveValue()),
                "warm_basis_used": warm_basis_used,
                "run_time": run_time,
            }
            args.stats.write_text(json.dumps(stats))
        except Exception as exc:
            print(f"WARNING: stats write failed: {exc}", file=sys.stderr)

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
