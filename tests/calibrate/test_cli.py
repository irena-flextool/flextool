"""Unit tests for the calibrator CLI (C2) — no solver.

The solve is stubbed: :func:`run_calibration` is patched to a spy that
returns a canned :class:`CalibResult`, so these tests exercise the argument
surface, the config translation, report writing and summary printing without
launching any subprocess.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from flextool.calibrate import _cli
from flextool.calibrate._loop import CalibConfig, CalibError, CalibResult, IterRecord
from flextool.calibrate._report import LONG_FILENAME, SUMMARY_FILENAME


def _canned_result() -> CalibResult:
    rec = IterRecord(
        iteration=0,
        adders={},
        residual={"west": 0.0},
        curtailment={"west": 0.0},
        penalty_total=0.0,
        penalty_by_node={"west": 0.0},
    )
    return CalibResult(
        converged=True,
        iterations_run=1,
        final_adders={},
        trajectory=[rec],
        guard_flagged_nodes=[],
        stop_reason="converged",
    )


def test_build_parser_defaults():
    parser = _cli.build_parser()
    args = parser.parse_args(["mydb.sqlite", "scenA", "--iterations", "3"])
    assert args.db == "mydb.sqlite"
    assert args.scenario == "scenA"
    assert args.iterations == 3
    assert args.slack_threshold_mwh == _cli.DEFAULT_SLACK_THRESHOLD_MWH
    assert args.over_build_tightness == 0.05
    assert args.damping_first == 1.0
    assert args.damping_remaining == 0.5
    assert args.warm_start_cache_dir is None
    assert args.work_dir == _cli.DEFAULT_WORK_DIR
    assert args.out_root == _cli.DEFAULT_OUT_ROOT
    assert args.debug is False


def test_build_parser_all_flags():
    parser = _cli.build_parser()
    args = parser.parse_args(
        [
            "db.sqlite",
            "scen",
            "--iterations",
            "5",
            "--slack-threshold",
            "2.5",
            "--over-build-tightness",
            "0.2",
            "--damping-first-iteration",
            "0.8",
            "--damping-remaining-iterations",
            "0.3",
            "--warm-start-cache-dir",
            "/tmp/wc",
            "--work-dir",
            "/tmp/wk",
            "--output-location",
            "/tmp/out",
            "--debug",
        ]
    )
    assert args.iterations == 5
    assert args.slack_threshold_mwh == 2.5
    assert args.over_build_tightness == 0.2
    assert args.damping_first == 0.8
    assert args.damping_remaining == 0.3
    assert args.warm_start_cache_dir == Path("/tmp/wc")
    assert args.work_dir == Path("/tmp/wk")
    assert args.out_root == Path("/tmp/out")
    assert args.debug is True


def test_iterations_required():
    parser = _cli.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["db.sqlite", "scen"])


def test_config_from_args_warm_start_default():
    parser = _cli.build_parser()
    args = parser.parse_args(
        ["db.sqlite", "scen", "--iterations", "1", "--work-dir", "/tmp/wk"]
    )
    config = _cli._config_from_args(args)
    assert isinstance(config, CalibConfig)
    # Warm-start cache defaults relative to the work dir.
    assert config.warm_start_cache_dir == Path("/tmp/wk") / "warm_start_cache"


def test_main_success_writes_report_and_prints(tmp_path, monkeypatch, capsys):
    captured = {}

    def _stub(url, scenario, config):
        captured["url"] = url
        captured["scenario"] = scenario
        captured["config"] = config
        return _canned_result()

    monkeypatch.setattr(_cli, "run_calibration", _stub)

    out_root = tmp_path / "out"
    rc = _cli.main(
        [
            "db.sqlite",
            "scenA",
            "--iterations",
            "2",
            "--slack-threshold",
            "3.0",
            "--over-build-tightness",
            "0.1",
            "--work-dir",
            str(tmp_path / "wk"),
            "--output-location",
            str(out_root),
        ]
    )

    assert rc == 0
    # The config passed to the (stubbed) solve carries the parsed flags.
    cfg = captured["config"]
    assert captured["url"] == "db.sqlite"
    assert captured["scenario"] == "scenA"
    assert cfg.iterations == 2
    assert cfg.slack_threshold_mwh == 3.0
    assert cfg.over_build_tightness == 0.1
    assert cfg.out_root == out_root
    assert cfg.warm_start_cache_dir == (tmp_path / "wk") / "warm_start_cache"

    # Report CSVs were written under <out_root>/report/.
    report_dir = out_root / "report"
    assert (report_dir / LONG_FILENAME).exists()
    assert (report_dir / SUMMARY_FILENAME).exists()

    # Summary printed to stdout.
    out = capsys.readouterr().out
    assert "CONVERGED" in out


def test_main_calib_error_returns_nonzero(tmp_path, monkeypatch, capsys):
    def _boom(url, scenario, config):
        raise CalibError("iteration 0 solve not trusted: missing output")

    monkeypatch.setattr(_cli, "run_calibration", _boom)

    rc = _cli.main(
        [
            "db.sqlite",
            "scenA",
            "--iterations",
            "1",
            "--output-location",
            str(tmp_path / "out"),
        ]
    )
    assert rc == 1
    err = capsys.readouterr().err
    assert "calibration failed" in err
    assert "missing output" in err
