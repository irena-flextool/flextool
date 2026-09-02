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


def final_write_methods_from_settings(settings) -> list[str]:
    """Map the project's "File outputs" choices to the calibrator's final
    output formats.

    The calibration loop always leaves a ``parquet`` result tree, so parquet is
    NOT a choice here; the calibrator regenerates the remaining formats from
    that parquet after the loop (no re-solve).  This mirrors the regular-run
    write-method assembly in
    :meth:`flextool.gui.execution_manager.ExecutionManager._build_run_command`
    (same ``auto_generate_*`` flags, same order) minus parquet, so a
    calibration's outputs match exactly what a normal run of the same project
    would produce.  An empty list means the operator unchecked every file
    output — the caller emits ``--skip-final-outputs`` to honour that.
    """
    methods: list[str] = []
    if settings.auto_generate_scen_plots:
        methods.append("plot")
    if settings.auto_generate_scen_excels:
        methods.append("excel")
    if settings.auto_generate_scen_csvs:
        methods.append("csv")
    if settings.auto_generate_comp_spinedb:
        methods.append("spinedb")
    return methods


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
    solves: Sequence[str],
    alternative_name: str | None,
    alternative_description: str | None = None,
) -> list[str]:
    """Build the argv for the representative-periods preprocess CLI.

    ``force_sustained`` maps to ``--force-highest-net-load`` ("sustained net
    load") and ``force_peak`` to ``--force-peak-load`` ("instantaneous
    peak"). Boolean flags are emitted only when True; ``--solves`` only when
    the list is non-empty (comma-joined); ``--alternative-name`` only when
    truthy. ``--alternative-description`` is emitted only when a description
    is supplied (single argv element, so ``shlex`` quotes the spaces).
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
    if solves:
        argv += ["--solves", ",".join(solves)]
    if alternative_name:
        argv += ["--alternative-name", str(alternative_name)]
    if alternative_description:
        argv += ["--alternative-description", str(alternative_description)]
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
    final_write_methods: list[str] | None = None,
) -> list[str]:
    """Build the argv for the calibrate CLI.

    ``overshoot`` is the CLI MULTIPLIER (not the dialog percent); the caller
    converts via :func:`overshoot_pct_to_multiplier`. ``--overshoot`` is
    emitted only when the multiplier differs from ``1.0`` (off). ``--debug``
    is emitted only when True.

    ``final_write_methods`` selects which formats the calibrator regenerates
    from the final parquet after the loop (see
    :func:`final_write_methods_from_settings`):

    * ``None`` — emit no final-output flag; the CLI keeps its own default (csv).
    * ``[]`` — emit ``--skip-final-outputs`` (the operator unchecked every File
      output, so leave the results parquet-only).
    * non-empty — emit ``--final-write-methods <methods…>``.
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
    if final_write_methods is not None:
        if final_write_methods:
            argv += ["--final-write-methods", *[str(m) for m in final_write_methods]]
        else:
            argv.append("--skip-final-outputs")
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
