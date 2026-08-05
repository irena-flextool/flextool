"""Unit tests for the shared RP-preprocess / calibrate argv builders.

Pure argv assertions — no GUI, no DB, no subprocess. These pin the single
renderer per command so the copied "Copy CLI command" text can never drift
from the argv the ExecutionManager actually runs.
"""
from __future__ import annotations

import shlex

from flextool.gui.calibrate_commands import (
    build_calibrate_command,
    build_rp_command,
    command_to_display_string,
    overshoot_pct_to_multiplier,
)


def test_rp_command_full_options_exact() -> None:
    argv = build_rp_command(
        "/usr/bin/python",
        "sqlite:///m.sqlite",
        "base",
        n_rp=12,
        period_length=24,
        force_sustained=True,
        force_peak=True,
        force_window=48,
        solves=["invest_solve", "y2050"],
        alternative_name="rp12_alt",
    )
    assert argv == [
        "/usr/bin/python",
        "-m",
        "flextool.representative_periods.preprocess",
        "sqlite:///m.sqlite",
        "base",
        "12",
        "24",
        "--force-highest-net-load",
        "--force-peak-load",
        "--force-window",
        "48",
        "--solves",
        "invest_solve,y2050",
        "--alternative-name",
        "rp12_alt",
    ]


def test_rp_command_booleans_omitted_when_false() -> None:
    argv = build_rp_command(
        "python",
        "db",
        "sc",
        n_rp=4,
        period_length=168,
        force_sustained=False,
        force_peak=False,
        force_window=None,
        solves=[],
        alternative_name=None,
    )
    assert argv == [
        "python",
        "-m",
        "flextool.representative_periods.preprocess",
        "db",
        "sc",
        "4",
        "168",
    ]
    assert "--force-highest-net-load" not in argv
    assert "--force-peak-load" not in argv
    assert "--force-window" not in argv


def test_rp_solves_empty_omitted_and_joined_otherwise() -> None:
    empty = build_rp_command(
        "python", "db", "sc",
        n_rp=4, period_length=24,
        force_sustained=False, force_peak=False,
        force_window=None,
        solves=[], alternative_name=None,
    )
    assert "--solves" not in empty

    joined = build_rp_command(
        "python", "db", "sc",
        n_rp=4, period_length=24,
        force_sustained=False, force_peak=False,
        force_window=None,
        solves=["a", "b", "c"], alternative_name=None,
    )
    assert joined[joined.index("--solves") + 1] == "a,b,c"


def test_rp_alternative_name_omitted_when_falsy() -> None:
    for name in (None, ""):
        argv = build_rp_command(
            "python", "db", "sc",
            n_rp=4, period_length=24,
            force_sustained=False, force_peak=False,
            force_window=None,
            solves=[], alternative_name=name,
        )
        assert "--alternative-name" not in argv


def test_calibrate_command_full_options_exact() -> None:
    argv = build_calibrate_command(
        "/usr/bin/python",
        "sqlite:///m.sqlite",
        "base",
        iterations=6,
        sizing="timed",
        overshoot=1.2,
        damping_first=1.0,
        damping_remaining=0.5,
        stall_fraction=0.05,
        warm_start_cache_dir="/tmp/ws",
        work_dir="/tmp/work",
        output_location="/tmp/out",
        debug=True,
    )
    assert argv == [
        "/usr/bin/python",
        "-m",
        "flextool.calibrate",
        "sqlite:///m.sqlite",
        "base",
        "--iterations",
        "6",
        "--sizing",
        "timed",
        "--overshoot",
        "1.2",
        "--damping-first-iteration",
        "1.0",
        "--damping-remaining-iterations",
        "0.5",
        "--stall-fraction",
        "0.05",
        "--warm-start-cache-dir",
        "/tmp/ws",
        "--work-dir",
        "/tmp/work",
        "--output-location",
        "/tmp/out",
        "--debug",
    ]


def test_calibrate_debug_omitted_when_false() -> None:
    argv = build_calibrate_command(
        "python", "db", "sc",
        iterations=3, sizing="uniform", overshoot=1.0,
        damping_first=1.0, damping_remaining=0.5, stall_fraction=0.05,
        warm_start_cache_dir=None, work_dir=None, output_location=None,
        debug=False,
    )
    assert "--debug" not in argv


def test_calibrate_overshoot_gated_on_multiplier() -> None:
    off = build_calibrate_command(
        "python", "db", "sc",
        iterations=3, sizing="uniform", overshoot=1.0,
        damping_first=1.0, damping_remaining=0.5, stall_fraction=0.05,
        warm_start_cache_dir=None, work_dir=None, output_location=None,
        debug=False,
    )
    assert "--overshoot" not in off

    on = build_calibrate_command(
        "python", "db", "sc",
        iterations=3, sizing="uniform", overshoot=1.35,
        damping_first=1.0, damping_remaining=0.5, stall_fraction=0.05,
        warm_start_cache_dir=None, work_dir=None, output_location=None,
        debug=False,
    )
    assert on[on.index("--overshoot") + 1] == "1.35"


def test_calibrate_optional_dirs_omitted_when_none() -> None:
    argv = build_calibrate_command(
        "python", "db", "sc",
        iterations=3, sizing="uniform", overshoot=1.0,
        damping_first=1.0, damping_remaining=0.5, stall_fraction=0.05,
        warm_start_cache_dir=None, work_dir=None, output_location=None,
        debug=False,
    )
    assert "--warm-start-cache-dir" not in argv
    assert "--work-dir" not in argv
    assert "--output-location" not in argv


def test_overshoot_pct_to_multiplier() -> None:
    assert overshoot_pct_to_multiplier(0) == 1.0
    assert overshoot_pct_to_multiplier(20) == 1.2


def test_display_string_round_trips_to_argv() -> None:
    argv = build_calibrate_command(
        "/usr/bin/python",
        "sqlite:///m with space.sqlite",
        "base scenario",
        iterations=6,
        sizing="timed",
        overshoot=1.2,
        damping_first=1.0,
        damping_remaining=0.5,
        stall_fraction=0.05,
        warm_start_cache_dir="/tmp/ws dir",
        work_dir="/tmp/work",
        output_location="/tmp/out",
        debug=True,
    )
    display = command_to_display_string(argv)
    assert isinstance(display, str)
    assert shlex.split(display) == argv


def test_display_string_round_trips_rp_argv() -> None:
    argv = build_rp_command(
        "/usr/bin/python",
        "sqlite:///m.sqlite",
        "base",
        n_rp=12,
        period_length=24,
        force_sustained=True,
        force_peak=False,
        force_window=48,
        solves=["invest_solve", "y2050"],
        alternative_name="rp12_alt",
    )
    assert shlex.split(command_to_display_string(argv)) == argv
