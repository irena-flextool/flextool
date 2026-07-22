"""Command-line front end for the adequacy-margin calibrator (C2).

Wires the argument surface onto :class:`~flextool.calibrate._loop.CalibConfig`,
runs :func:`~flextool.calibrate._loop.run_calibration`, then renders the
report CSVs and prints the human-readable summary.  Exposed as
``python -m flextool.calibrate`` via :mod:`flextool.calibrate.__main__`.

The calibrator solves ``--iterations + 1`` times (iteration 0 is the
baseline) and, for every shedding node, sizes an ``energy_margin_adder``
increment that injects that node's residual unserved energy back as demand,
damped and guarded against over-build.  This module is the operator's entry
point to that loop.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from flextool.calibrate._loop import (
    CalibConfig,
    CalibError,
    CalibResult,
    run_calibration,
)
from flextool.calibrate._report import format_summary, write_report

DEFAULT_WORK_DIR = Path("calib_work")
DEFAULT_OUT_ROOT = Path("calib_out")
# Small positive default: convergence when total residual unserved energy is
# at/below this many MWh (a single-MWh floor absorbs solver noise).
DEFAULT_SLACK_THRESHOLD_MWH = 1.0


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser for the calibrator CLI."""
    parser = argparse.ArgumentParser(
        prog="flextool.calibrate",
        description=(
            "Calibrate per-node energy-margin adders until every node's "
            "residual unserved energy falls to/under the slack threshold, "
            "freezing resource-capped nodes rather than over-building demand."
        ),
    )
    parser.add_argument(
        "db",
        help="SpineDB URL (or path) of the model to calibrate.",
    )
    parser.add_argument(
        "scenario",
        help="Name of the scenario (model instance) to calibrate.",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        required=True,
        help=(
            "Number of ADJUSTMENT iterations after the baseline; the loop "
            "runs iterations+1 solves (iteration 0 is the baseline)."
        ),
    )
    parser.add_argument(
        "--slack-threshold",
        type=float,
        default=DEFAULT_SLACK_THRESHOLD_MWH,
        dest="slack_threshold_mwh",
        help=(
            "Convergence threshold in MWh: stop once total residual unserved "
            "energy is at/under this value (default: "
            f"{DEFAULT_SLACK_THRESHOLD_MWH})."
        ),
    )
    parser.add_argument(
        "--stall-fraction",
        type=float,
        default=0.05,
        dest="stall_fraction",
        help=(
            "Over-build guard STALL fraction [0..1]. A shedding node whose "
            "residual unserved energy drops by LESS than this fraction of its "
            "prior gap in response to its own bump is frozen as resource-capped "
            "(margin buys it no adequacy) and reported as needing firm "
            "capacity / imports / storage. HIGHER freezes a stalled node "
            "SOONER; 0.0 disables the guard. Default: 0.05."
        ),
    )
    parser.add_argument(
        "--over-build-tightness",
        type=float,
        default=0.05,
        dest="over_build_tightness",
        help=(
            "DEPRECATED / no-op: retained for compatibility only. The "
            "over-build guard now freezes on residual STALL (see "
            "--stall-fraction), not curtailment efficiency, so this value is "
            "not consulted. Default: 0.05."
        ),
    )
    parser.add_argument(
        "--damping-first-iteration",
        type=float,
        default=1.0,
        dest="damping_first",
        help=(
            "Damping factor lambda applied to the FIRST correction "
            "(increment = lambda * residual / W). 1.0 = full undamped step. "
            "Default: 1.0."
        ),
    )
    parser.add_argument(
        "--damping-remaining-iterations",
        type=float,
        default=0.5,
        dest="damping_remaining",
        help=(
            "Damping factor lambda applied to every correction AFTER the "
            "first; lower damps oscillation as slack nears zero. Default: 0.5."
        ),
    )
    parser.add_argument(
        "--overshoot",
        type=float,
        default=1.0,
        dest="overshoot",
        help=(
            "Planning-margin SAFETY multiplier on the sized margin (default "
            "1.0 = off). A single-year (or single-year representative-period) "
            "model under-estimates true multi-year severity, so a value >1 "
            "deliberately over-provisions: overshoot=1.2 builds ~20%% beyond "
            "the measured slack. Higher builds more headroom for unmodeled "
            "multi-year risk; the right value is MODEL-DEPENDENT."
        ),
    )
    parser.add_argument(
        "--sizing",
        choices=("uniform", "timed"),
        default="uniform",
        dest="sizing",
        help=(
            "Adder placement mode. 'uniform' (default): a constant "
            "per-timestep margin sized lambda*residual/W. 'timed': the SAME "
            "total energy placed per-cell at the low-VRE stress hours, folded "
            "from node_slack_up_dt_e onto the representative timeline."
        ),
    )
    parser.add_argument(
        "--warm-start-cache-dir",
        type=Path,
        default=None,
        dest="warm_start_cache_dir",
        help=(
            "Directory for the warm-start basis cache shared across "
            "iterations (default: <work-dir>/warm_start_cache)."
        ),
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=DEFAULT_WORK_DIR,
        dest="work_dir",
        help=(
            "Subprocess working directory for the per-iteration solves "
            f"(default: {DEFAULT_WORK_DIR}/)."
        ),
    )
    parser.add_argument(
        "--output-location",
        type=Path,
        default=DEFAULT_OUT_ROOT,
        dest="out_root",
        help=(
            "Output-location root; solve outputs land under "
            "<output-location>/output_parquet/<scenario>/ and the report "
            f"CSVs under <output-location>/report/ (default: {DEFAULT_OUT_ROOT}/)."
        ),
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help=(
            "Archive each iteration's outputs to out_iter_<k>/ instead of "
            "clearing them between iterations."
        ),
    )
    return parser


def _config_from_args(args: argparse.Namespace) -> CalibConfig:
    """Translate parsed args into a :class:`CalibConfig`.

    Fills the warm-start cache default relative to the work dir when the
    operator did not pin one.
    """
    warm_start_cache_dir = args.warm_start_cache_dir
    if warm_start_cache_dir is None:
        warm_start_cache_dir = Path(args.work_dir) / "warm_start_cache"
    return CalibConfig(
        iterations=args.iterations,
        slack_threshold_mwh=args.slack_threshold_mwh,
        damping_first=args.damping_first,
        damping_remaining=args.damping_remaining,
        over_build_tightness=args.over_build_tightness,
        warm_start_cache_dir=Path(warm_start_cache_dir),
        work_dir=Path(args.work_dir),
        out_root=Path(args.out_root),
        debug=args.debug,
        sizing=args.sizing,
        overshoot=args.overshoot,
        stall_fraction=args.stall_fraction,
    )


def main(argv: list[str] | None = None) -> int:
    """Run the calibrator from the command line.

    Parses *argv* (or ``sys.argv``), runs the calibration, writes the report
    CSVs under ``<output-location>/report/`` and prints the summary to
    stdout.  Returns 0 on success; on a fail-closed solve
    (:class:`CalibError`) prints the reason to stderr and returns 1.
    """
    parser = build_parser()
    args = parser.parse_args(argv)
    config = _config_from_args(args)

    try:
        result: CalibResult = run_calibration(args.db, args.scenario, config)
    except CalibError as exc:
        print(f"calibration failed: {exc}", file=sys.stderr)
        return 1

    report_dir = Path(config.out_root) / "report"
    write_report(result, out_dir=report_dir)
    print(format_summary(result))
    print(f"\nReport CSVs written under {report_dir}/")
    return 0


__all__ = ["build_parser", "main"]
