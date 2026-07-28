"""Single-source argv builders for the RP-preprocess and calibrate CLIs.

These pure functions are the ONE renderer per command. Both the
ExecutionManager auxiliary-job worker (which runs the subprocess with the
raw argv list) and the dialog's "Copy CLI command" button (which renders
the same argv via :func:`command_to_display_string`) go through here, so
the copied text can never drift from what is actually executed.

Targets:
  * ``python -m flextool.representative_periods.preprocess <db_url>
    <scenario> <n_rp> <period_length> [flags]``
    (see ``flextool/representative_periods/preprocess.py::main``)
  * ``python -m flextool.calibrate <db> <scenario> [flags]``
    (see ``flextool/calibrate/_cli.py::build_parser``)

Region-scope RP args are intentionally NOT emitted here (out of GUI scope).
Calibrate's ``--slack-threshold`` is intentionally omitted so the
calibrator uses its own default.
"""
from __future__ import annotations

import shlex
from collections.abc import Sequence

RP_MODULE = "flextool.representative_periods.preprocess"
CALIBRATE_MODULE = "flextool.calibrate"


def overshoot_pct_to_multiplier(pct: float) -> float:
    """Convert a planning-safety-margin PERCENT into the CLI ``--overshoot``
    multiplier.

    The dialog stores ``calib_overshoot_pct`` (0 = off); the calibrator CLI
    consumes a multiplier where ``1.0`` means off. ``20`` percent → ``1.2``.
    """
    return 1.0 + pct / 100.0


def build_rp_command(
    python_exe: str,
    db_url: str,
    scenario: str,
    *,
    n_rp: int,
    period_length: int,
    force_sustained: bool,
    force_peak: bool,
    force_window: int | None,
    count_mode: str,
    solves: Sequence[str],
    alternative_name: str | None,
) -> list[str]:
    """Build the argv for the representative-periods preprocess CLI.

    ``force_sustained`` maps to ``--force-highest-net-load`` ("sustained net
    load") and ``force_peak`` to ``--force-peak-load`` ("instantaneous
    peak"). Boolean flags are emitted only when True; ``--solves`` only when
    the list is non-empty (comma-joined); ``--alternative-name`` only when
    truthy.
    """
    argv: list[str] = [
        python_exe,
        "-m",
        RP_MODULE,
        db_url,
        scenario,
        str(n_rp),
        str(period_length),
    ]
    if force_sustained:
        argv.append("--force-highest-net-load")
    if force_peak:
        argv.append("--force-peak-load")
    if force_window is not None:
        argv += ["--force-window", str(force_window)]
    if count_mode:
        argv += ["--force-count-mode", str(count_mode)]
    if solves:
        argv += ["--solves", ",".join(solves)]
    if alternative_name:
        argv += ["--alternative-name", str(alternative_name)]
    return argv


def build_calibrate_command(
    python_exe: str,
    db_url: str,
    scenario: str,
    *,
    iterations: int,
    sizing: str,
    overshoot: float,
    damping_first: float,
    damping_remaining: float,
    stall_fraction: float,
    warm_start_cache_dir: str | None,
    work_dir: str | None,
    output_location: str | None,
    debug: bool,
) -> list[str]:
    """Build the argv for the calibrate CLI.

    ``overshoot`` is the CLI MULTIPLIER (not the dialog percent); the caller
    converts via :func:`overshoot_pct_to_multiplier`. ``--overshoot`` is
    emitted only when the multiplier differs from ``1.0`` (off). ``--debug``
    is emitted only when True.
    """
    argv: list[str] = [
        python_exe,
        "-m",
        CALIBRATE_MODULE,
        db_url,
        scenario,
        "--iterations",
        str(iterations),
    ]
    if sizing:
        argv += ["--sizing", str(sizing)]
    if overshoot != 1.0:
        argv += ["--overshoot", str(overshoot)]
    argv += ["--damping-first-iteration", str(damping_first)]
    argv += ["--damping-remaining-iterations", str(damping_remaining)]
    argv += ["--stall-fraction", str(stall_fraction)]
    if warm_start_cache_dir:
        argv += ["--warm-start-cache-dir", str(warm_start_cache_dir)]
    if work_dir:
        argv += ["--work-dir", str(work_dir)]
    if output_location:
        argv += ["--output-location", str(output_location)]
    if debug:
        argv.append("--debug")
    return argv


def command_to_display_string(argv: Sequence[str]) -> str:
    """Render *argv* as a single copy-pasteable POSIX shell line.

    Uses :func:`shlex.join`, so ``shlex.split`` on the result reproduces the
    original argv exactly — the copied text and the run argv are the same
    command.
    """
    return shlex.join(argv)
