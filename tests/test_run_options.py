"""The run-option resolver: flag > environment variable > database > default, one registry, no env writes."""
from __future__ import annotations

import itertools
import logging
import os
from types import SimpleNamespace

import pytest

from flextool.run_options import OPTIONS, RunOptions, resolve

# Per option: (flag value, environment text, value that text means, database value).  The flag and database
# values differ from the default and from each other so the winning tier is visible in the value.
SAMPLES: dict[str, tuple[object, str, object, object]] = {
    "threads": (6, "4", 4, 3),
    "highs_threads": (6, "4", 4, 3),
    "presolve": ("off", "choose", "choose", "on"),
    "solver_log_level": ("verbose", "silent", "silent", "verbose"),
    "solver_time_limit": (30.0, "120", 120.0, 60.0),
    "solver_mip_gap": (0.01, "0.0005", 0.0005, 0.002),
    "highs_time_limit": (30.0, "120", 120.0, 60.0),
    "highs_mip_gap": (0.01, "0.0005", 0.0005, 0.002),
    "scaling": ("off", "basic", "basic", "solver_only"),
    "user_bound_scale": (-4, "-2", -2, -6),
    "save_memory": (True, "1", True, True),
    "warm_start": (True, "1", True, True),
    "basis_cache_dir": ("/flag/cache", "/env/cache", "/env/cache", "/db/cache"),
    "solver_config_dir": (("/flag/cfg",), "/env/cfg", ("/env/cfg",), ("/db/cfg",)),
    "benders_workers": (3, "5", 5, 2),
    "benders_in_out_weight": (0.5, "0.3", 0.3, 0.7),
}


def test_samples_cover_the_registry():
    assert set(SAMPLES) == set(OPTIONS)


TIERS = ("cli", "env", "db")


@pytest.mark.parametrize("key", sorted(OPTIONS))
@pytest.mark.parametrize(
    "present", [c for n in range(4) for c in itertools.combinations(TIERS, n)], ids=lambda c: "+".join(c) or "none",
)
def test_precedence_table(key, present):
    spec = OPTIONS[key]
    flag_value, env_text, env_value, db_value = SAMPLES[key]
    environ = {spec.env: env_text} if "env" in present and spec.env else {}
    got = resolve(
        spec,
        cli=flag_value if "cli" in present else None,
        environ=environ,
        db=db_value if "db" in present else None,
    )
    if "cli" in present:
        expected = (flag_value, "cli")
    elif "env" in present and spec.env:
        expected = (env_value, "env")
    elif "db" in present:
        expected = (db_value, "db")
    else:
        expected = (spec.default, "default")
    assert (got.value, got.source) == expected


@pytest.mark.parametrize(
    ("key", "bad"),
    [
        ("threads", "four"),
        ("threads", "0"),
        ("highs_threads", "four"),
        ("highs_threads", "0"),
        ("presolve", "maybe"),
        ("solver_time_limit", "-5"),
        ("scaling", "fulll"),
        ("save_memory", "yes"),
        ("benders_in_out_weight", "1.0"),
    ],
)
def test_malformed_env_warns_once_and_falls_through(key, bad, caplog):
    spec = OPTIONS[key]
    db_value = SAMPLES[key][3]
    with caplog.at_level(logging.WARNING, logger="flextool.run_options"):
        got = resolve(spec, environ={spec.env: bad}, db=db_value)
    assert (got.value, got.source) == (db_value, "db")
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert spec.env in warnings[0].getMessage()


def test_empty_env_counts_as_unset(caplog):
    spec = OPTIONS["highs_threads"]
    with caplog.at_level(logging.WARNING, logger="flextool.run_options"):
        got = resolve(spec, environ={spec.env: "  "})
    assert (got.value, got.source) == (1, "default")
    assert not caplog.records


def test_benders_workers_zero_means_automatic_without_warning(caplog):
    spec = OPTIONS["benders_workers"]
    with caplog.at_level(logging.WARNING, logger="flextool.run_options"):
        got = resolve(spec, environ={spec.env: "0"})
    assert (got.value, got.source) == (None, "default")
    assert not caplog.records


@pytest.mark.parametrize(
    ("key", "bad", "bad_text"),
    [
        ("solver_time_limit", 0.0, "0"),
        ("solver_time_limit", -5.0, "-5"),
        ("solver_mip_gap", -0.1, "-0.1"),
        ("highs_threads", 0, "0"),
        ("threads", 0, "0"),
    ],
)
def test_out_of_range_flag_is_ignored_like_its_environment_variable(key, bad, bad_text, caplog):
    """A flag value outside the option's range warns once and falls through, exactly as the same text in the
    environment variable does: both forms give the database value."""
    spec = OPTIONS[key]
    db_value = SAMPLES[key][3]
    with caplog.at_level(logging.WARNING, logger="flextool.run_options"):
        from_flag = resolve(spec, cli=bad, environ={}, db=db_value)
    flag_warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="flextool.run_options"):
        from_env = resolve(spec, environ={spec.env: bad_text}, db=db_value)
    env_warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert (from_flag.value, from_flag.source) == (from_env.value, from_env.source) == (db_value, "db")
    assert len(flag_warnings) == len(env_warnings) == 1
    assert spec.flag in flag_warnings[0].getMessage()


def test_out_of_range_flag_lets_the_environment_variable_through():
    from flextool.cli.cmd_run_flextool import build_parser

    args = build_parser().parse_args(["sqlite:///x.sqlite", "--solver-time-limit", "0"])
    opts = RunOptions.from_cli(args, environ={"FLEXTOOL_SOLVER_TIME_LIMIT": "120"})
    assert (opts.value("solver_time_limit"), opts.source("solver_time_limit")) == (120.0, "env")


def test_benders_workers_flag_zero_is_an_explicit_automatic(caplog):
    """``--benders-workers 0`` is the documented automatic setting: kept as the flag's choice, no warning."""
    spec = OPTIONS["benders_workers"]
    with caplog.at_level(logging.WARNING, logger="flextool.run_options"):
        got = resolve(spec, cli=0, environ={spec.env: "5"})
    assert (got.value, got.source) == (0, "cli")
    assert not caplog.records


def _flags(**given):
    """A parsed command line: every registry flag unset (None) except *given*."""
    ns = {spec.dest: None for spec in OPTIONS.values() if spec.flag}
    ns.update(given)
    return SimpleNamespace(**ns)


def test_from_cli_flag_over_env_and_env_over_default():
    environ = {"FLEXTOOL_HIGHS_THREADS": "4", "FLEXTOOL_SCALING": "basic"}
    opts = RunOptions.from_cli(_flags(scaling="off"), environ=environ)
    assert (opts.value("scaling"), opts.source("scaling")) == ("off", "cli")
    assert (opts.value("highs_threads"), opts.source("highs_threads")) == (4, "env")
    assert (opts.value("presolve"), opts.source("presolve")) == (None, "default")


def test_with_db_sits_between_env_and_default():
    opts = RunOptions.from_env(environ={})
    assert opts.with_db("scaling", "basic").source == "db"
    assert opts.with_db("scaling", None).source == "default"
    env_opts = RunOptions.from_env(environ={"FLEXTOOL_SCALING": "off"})
    assert (env_opts.with_db("scaling", "basic").value, env_opts.with_db("scaling", "basic").source) == ("off", "env")


def test_resolving_never_writes_the_environment(monkeypatch):
    for spec in OPTIONS.values():
        if spec.env:
            monkeypatch.delenv(spec.env, raising=False)
    monkeypatch.setenv("FLEXTOOL_HIGHS_THREADS", "4")
    before = dict(os.environ)
    RunOptions.from_cli(_flags(scaling="off", save_memory=True, highs_threads=2))
    RunOptions.from_env()
    for spec in OPTIONS.values():
        resolve(spec, cli=SAMPLES[spec.key][0])
    assert dict(os.environ) == before


def test_summary_lines_name_the_source():
    opts = RunOptions.from_cli(_flags(scaling="basic"), environ={"FLEXTOOL_HIGHS_THREADS": "4"})
    assert opts.summary_lines() == ["highs_threads=4 (env FLEXTOOL_HIGHS_THREADS)", "scaling=basic (flag)"]


def test_to_argv_forwards_only_flag_given_options():
    opts = RunOptions.from_cli(
        _flags(scaling="off", save_memory=True, solver_mip_gap=0.0005),
        environ={"FLEXTOOL_HIGHS_THREADS": "4", "FLEXTOOL_THREADS": "2"},
    )
    assert opts.to_argv() == ["--solver-mip-gap", "0.0005", "--scaling", "off", "--save-memory"]
    flagged = RunOptions.from_cli(_flags(threads=3), environ={"FLEXTOOL_THREADS": "2"})
    assert flagged.to_argv() == ["--threads", "3"]


def test_to_argv_round_trips_through_the_run_parser():
    from flextool.cli.cmd_run_flextool import build_parser

    given = {
        "threads": 5, "highs_threads": 3, "presolve": "off", "solver_log_level": "silent", "solver_time_limit": 12.5,
        "solver_mip_gap": 0.0005, "scaling": "basic", "user_bound_scale": -3,
        "save_memory": True, "warm_start": True, "basis_cache_dir": "/tmp/cache", "benders_workers": 2,
        "solver_config_dir": ("/tmp/project/solver_config", "/tmp/solver_config"),
    }
    flagged = {key for key, spec in OPTIONS.items() if spec.flag}
    assert set(given) == flagged
    parent = RunOptions.from_cli(_flags(**{OPTIONS[k].dest: v for k, v in given.items()}), environ={})
    args = build_parser().parse_args(["sqlite:///x.sqlite", *parent.to_argv()])
    child = RunOptions.from_cli(args, environ={})
    assert {k: (r.value, r.source) for k, r in child.resolved.items() if r.source == "cli"} == {
        k: (v, "cli") for k, v in given.items()
    }


def test_registry_flags_default_to_unset_on_the_run_parser():
    from flextool.cli.cmd_run_flextool import build_parser

    args = build_parser().parse_args(["sqlite:///x.sqlite"])
    for spec in OPTIONS.values():
        if spec.flag:
            assert getattr(args, spec.dest) is None, spec.flag


def test_solver_config_dir_lists_several_folders():
    """The environment twin separates folders with the platform's path separator (empty parts dropped); the
    flag is repeated, one folder per flag, and a child process gets the same repeated flags."""
    from flextool.cli.cmd_run_flextool import build_parser

    env = RunOptions.from_env({"FLEXTOOL_SOLVER_CONFIG_DIR": os.pathsep.join(["/p/solver_config", "", "/ws/cfg"])})
    assert (env.value("solver_config_dir"), env.source("solver_config_dir")) == (("/p/solver_config", "/ws/cfg"), "env")
    args = build_parser().parse_args(["x.sqlite", "--solver-config-dir", "/a", "--solver-config-dir", "/b"])
    flagged = RunOptions.from_cli(args, environ={"FLEXTOOL_SOLVER_CONFIG_DIR": "/env"})
    assert flagged.value("solver_config_dir") == ("/a", "/b")
    assert flagged.to_argv() == ["--solver-config-dir", "/a", "--solver-config-dir", "/b"]
