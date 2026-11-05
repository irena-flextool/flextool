"""Runtime options: one resolver per option, one registry for all of them.

FlexTool reads its settings from three places, and each has its own job:

* the **database** holds what changes the model or its answer (a solve's ``scaling``, ``solver_mip_gap``,
  ``benders_in_out_weight``, ...);
* a **command-line flag** is a choice for one run (threads, time limit, log level, ``--save-memory``, ...);
* an **environment variable** is a process-level setting that must exist before import (allocator, polars build)
  or a developer / CI knob, and the twin of a flag for callers that have no command line (Spine Toolbox tools,
  a Python console, tests).

An option that has more than one of these forms is resolved here, always in the order
**flag > environment variable > database > default**.  The rules:

1. Flags of registry options default to ``None`` on the command line, never to a concrete value, so an unset
   flag lets the environment variable through.  A ``store_true`` flag gives ``True`` or ``None``.
2. Resolving never writes ``os.environ``.  ``cmd_run_flextool`` builds :meth:`RunOptions.from_cli` and passes
   it on; a library caller that passes nothing gets :meth:`RunOptions.from_env`.
3. A child FlexTool process gets the parent's explicit choices as flags (:meth:`RunOptions.to_argv`) and
   inherits the parent's environment untouched, so a variable set in the shell reaches the child as set.
4. Process-level variables (``MALLOC_ARENA_MAX``, ``_RJEM_MALLOC_CONF``, ``POLARS_*``) keep ``setdefault``
   semantics and are listed in :data:`ENV_KNOBS`, not in :data:`OPTIONS`.
5. HiGHS keys keep their own stack (baseline < ``highs.opt`` < ``solve.solver_mip_gap`` <
   ``solve.solver_arguments`` < operator layer); this registry supplies only the operator layer.

The database tier is not read here: the caller passes the database value (``db=``) because only it knows
which solve is active.

This module imports nothing from the engine, so the CLI, the GUI and the calibrator can use it.
"""
from __future__ import annotations

import logging
import math
import os
from dataclasses import dataclass, field
from typing import Any, Callable, Generic, Literal, Mapping, TypeVar

T = TypeVar("T")

Source = Literal["cli", "env", "db", "default"]
Audience = Literal["user", "dev", "test"]

_logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Parsers: text (an environment variable) -> value.  Raise ValueError on bad text; return None for a documented
# "not set" sentinel so the next tier applies without a warning.
# ---------------------------------------------------------------------------


def _choice(*choices: str) -> Callable[[str], str]:
    def parse(text: str) -> str:
        value = text.strip().lower()
        if value not in choices:
            raise ValueError(f"must be one of {', '.join(choices)}")
        return value
    return parse


def _positive_int(text: str) -> int:
    value = int(text.strip())
    if value < 1:
        raise ValueError("must be a whole number of at least 1")
    return value


def _int(text: str) -> int:
    return int(text.strip())


def _positive_float(text: str) -> float:
    value = float(text.strip())
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError("must be a positive number")
    return value


def _non_negative_float(text: str) -> float:
    value = float(text.strip())
    if not math.isfinite(value) or value < 0.0:
        raise ValueError("must be a number of at least 0")
    return value


def _switch(text: str) -> bool:
    value = text.strip()
    if value == "1":
        return True
    if value == "0":
        return False
    raise ValueError("must be 1 (on) or 0 (off)")


def _path_text(text: str) -> str:
    return text.strip()


def _workers(text: str) -> int | None:
    value = int(text.strip())
    # 0 or a negative count is the documented "automatic" setting.
    return value if value > 0 else None


def _in_out_weight(text: str) -> float:
    value = float(text.strip())
    if not 0.0 <= value < 1.0:
        raise ValueError("must be in [0, 1): 1 or more never queries the master, below 0 is meaningless")
    return value


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class OptionSpec(Generic[T]):
    """One operator option that has more than one form (flag / environment variable / database)."""

    key: str
    flag: str | None
    env: str | None
    db: str | None  # the database parameter, for documentation; the caller passes its value
    parse: Callable[[str], T | None]
    default: T | None
    audience: Audience
    help: str

    @property
    def dest(self) -> str:
        """The ``argparse`` attribute name of :attr:`flag`."""
        if self.flag is None:
            return self.key
        return self.flag.lstrip("-").replace("-", "_")


@dataclass(frozen=True)
class Resolved(Generic[T]):
    value: T | None
    source: Source


def _spec(**kwargs: Any) -> OptionSpec:
    return OptionSpec(**kwargs)


OPTIONS: dict[str, OptionSpec] = {
    spec.key: spec
    for spec in (
        _spec(
            key="highs_threads", flag="--highs-threads", env="FLEXTOOL_HIGHS_THREADS", db=None,
            parse=_positive_int, default=1, audience="user",
            help="HiGHS solver threads; more than 1 turns on HiGHS parallel mode",
        ),
        _spec(
            key="presolve", flag="--presolve", env="FLEXTOOL_HIGHS_PRESOLVE", db=None,
            parse=_choice("on", "off", "choose"), default=None, audience="user",
            help="HiGHS presolve (on / off / choose); unset keeps the baseline on",
        ),
        _spec(
            key="solver_log_level", flag="--solver-log-level", env="FLEXTOOL_SOLVER_LOG_LEVEL", db=None,
            parse=_choice("silent", "normal", "verbose"), default="normal", audience="user",
            help="HiGHS log verbosity (silent / normal / verbose)",
        ),
        _spec(
            key="solver_time_limit", flag="--solver-time-limit", env="FLEXTOOL_HIGHS_TIME_LIMIT", db=None,
            parse=_positive_float, default=None, audience="user",
            help="HiGHS wall-clock time limit in seconds; unset means no limit",
        ),
        _spec(
            key="solver_mip_gap", flag="--solver-mip-gap", env="FLEXTOOL_HIGHS_MIP_GAP", db="solve.solver_mip_gap",
            parse=_non_negative_float, default=None, audience="user",
            help="HiGHS MIP relative gap for every solve; unset keeps the solve's solver_mip_gap, else highs.opt",
        ),
        _spec(
            key="scaling", flag="--scaling", env="FLEXTOOL_SCALING", db="solve.scaling",
            parse=_choice("off", "solver_only", "basic", "full"), default="full", audience="user",
            help="autoscaler strategy (off / solver_only / basic / full)",
        ),
        _spec(
            key="user_bound_scale", flag="--user-bound-scale", env="FLEXTOOL_USER_BOUND_SCALE", db=None,
            parse=_int, default=None, audience="user",
            help="HiGHS user_bound_scale exponent; unset lets the autoscaler choose",
        ),
        _spec(
            key="save_memory", flag="--save-memory", env="FLEXTOOL_SAVE_MEMORY", db=None,
            parse=_switch, default=False, audience="user",
            help="solve each LP in a separate process to lower peak memory",
        ),
        _spec(
            key="warm_start", flag="--warm-start", env="FLEXTOOL_WARM_START", db=None,
            parse=_switch, default=False, audience="user",
            help="reuse a cached HiGHS basis across solves of the same model structure",
        ),
        _spec(
            key="basis_cache_dir", flag="--basis-cache-dir", env="FLEXTOOL_BASIS_CACHE_DIR", db=None,
            parse=_path_text, default=None, audience="user",
            help="folder for the warm-start basis cache; unset uses <work folder>/basis_cache",
        ),
        _spec(
            key="benders_workers", flag="--benders-workers", env="FLEXTOOL_BENDERS_WORKERS", db=None,
            parse=_workers, default=None, audience="user",
            help="threads for the Benders region solves; unset or 0 uses min(regions, cores - 1)",
        ),
        _spec(
            key="benders_in_out_weight", flag=None, env="FLEXTOOL_BENDERS_IN_OUT_WEIGHT",
            db="solve.benders_in_out_weight", parse=_in_out_weight, default=0.0, audience="dev",
            help="developer override of the solve's benders_in_out_weight, in [0, 1)",
        ),
    )
}


@dataclass(frozen=True)
class EnvKnob:
    """An environment variable that is not the twin of a flag: process-level, developer or test-only."""

    name: str
    audience: Audience
    page: str  # the documentation page that describes it
    help: str


_USER_PAGE = "docs/env_vars.md"

ENV_KNOBS: dict[str, EnvKnob] = {
    knob.name: knob
    for knob in (
        EnvKnob("FLEXTOOL_SOLVER_CONFIG_DIR", "user", _USER_PAGE,
                "folder whose solver option files (highs.opt, ...) seed a run's work folder"),
        EnvKnob("FLEXTOOL_NO_ENV_CHECK", "user", _USER_PAGE, "skip the start-up check of the native libraries"),
        EnvKnob("FLEXTOOL_NO_UPDATE_CHECK", "user", _USER_PAGE, "the GUI does not look for a newer FlexTool"),
        EnvKnob("FLEXTOOL_DPI", "user", _USER_PAGE, "GUI screen resolution override"),
        EnvKnob("FLEXTOOL_SLICE", "user", _USER_PAGE, "systemd slice for GUI-started runs (Linux)"),
        EnvKnob("MALLOC_ARENA_MAX", "user", _USER_PAGE, "glibc malloc arena cap; a value you set is kept"),
        EnvKnob("POLARS_PREFER_PKG", "user", _USER_PAGE, "which polars build loads; a value you set is kept"),
        EnvKnob("POLARS_FORCE_PKG", "user", _USER_PAGE, "force one polars build"),
        EnvKnob("POLARS_MAX_THREADS", "user", _USER_PAGE, "polars thread pool size; printed in the run header"),
        EnvKnob("POLARS_ENGINE_AFFINITY", "dev", "docs/dev/env_vars.md",
                "polars engine choice; only the in-memory engine is verified"),
        EnvKnob("_RJEM_MALLOC_CONF", "dev", "docs/dev/env_vars.md", "jemalloc settings of polars' allocator"),
    )
}


# ---------------------------------------------------------------------------
# Resolver
# ---------------------------------------------------------------------------


def resolve(
    spec: OptionSpec[T],
    *,
    cli: T | None = None,
    environ: Mapping[str, str] | None = None,
    db: T | None = None,
) -> Resolved[T]:
    """Return the first set tier of *spec*: flag, environment variable, database, default.

    An empty environment variable counts as unset.  A malformed one logs one warning naming the variable and
    falls through to the next tier.  Never writes the environment.
    """
    if cli is not None:
        return Resolved(cli, "cli")
    env = os.environ if environ is None else environ
    if spec.env is not None:
        raw = env.get(spec.env)
        if raw is not None and raw.strip():
            try:
                value = spec.parse(raw)
            except ValueError as exc:
                _logger.warning("Ignoring %s=%r (%s).", spec.env, raw, exc)
            else:
                if value is not None:
                    return Resolved(value, "env")
    if db is not None:
        return Resolved(db, "db")
    return Resolved(spec.default, "default")


@dataclass(frozen=True)
class RunOptions:
    """The operator options of one run, each with the tier it came from.

    Holds the flag and environment tiers; the database tier is per solve and added with :meth:`with_db`.
    """

    resolved: Mapping[str, Resolved] = field(default_factory=dict)

    @classmethod
    def from_cli(cls, args: Any, environ: Mapping[str, str] | None = None) -> "RunOptions":
        """Options of a ``cmd_run_flextool`` run: the parsed flags of *args* over the environment."""
        env = os.environ if environ is None else environ
        resolved = {
            key: resolve(spec, cli=getattr(args, spec.dest, None) if spec.flag else None, environ=env)
            for key, spec in OPTIONS.items()
        }
        return cls(resolved=resolved)

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> "RunOptions":
        """Options of a library call (Spine Toolbox tool, Python console, test): the environment only."""
        resolved = {key: resolve(spec, environ=environ) for key, spec in OPTIONS.items()}
        return cls(resolved=resolved)

    def get(self, key: str) -> Resolved:
        try:
            return self.resolved[key]
        except KeyError:
            raise KeyError(f"{key!r} is not a registered run option") from None

    def value(self, key: str) -> Any:
        return self.get(key).value

    def source(self, key: str) -> Source:
        return self.get(key).source

    def with_db(self, key: str, db_value: Any) -> Resolved:
        """Resolve *key* with a solve's database value: flag and environment variable still win over it."""
        current = self.get(key)
        if current.source in ("cli", "env") or db_value is None:
            return current
        return Resolved(db_value, "db")

    def to_argv(self) -> list[str]:
        """The flags that reproduce this run's flag-given options in a child FlexTool process.

        Only flag-sourced options are forwarded; the child inherits the environment, so an environment-sourced
        option reaches it unchanged and is not turned into a flag.
        """
        argv: list[str] = []
        for key, spec in OPTIONS.items():
            r = self.resolved.get(key)
            if r is None or r.source != "cli" or spec.flag is None:
                continue
            if isinstance(r.value, bool):
                if r.value:
                    argv.append(spec.flag)
                continue
            argv.extend([spec.flag, _format_value(r.value)])
        return argv

    def summary_lines(self) -> list[str]:
        """``key=value (source)`` for each option set by a flag or an environment variable."""
        lines = []
        for key, spec in OPTIONS.items():
            r = self.resolved.get(key)
            if r is None or r.source not in ("cli", "env"):
                continue
            origin = "flag" if r.source == "cli" else f"env {spec.env}"
            lines.append(f"{key}={_format_value(r.value)} ({origin})")
        return lines


def _format_value(value: Any) -> str:
    # ``repr`` of a float is the shortest text that parses back to the same float.
    if isinstance(value, float):
        return repr(value)
    return str(value)


__all__ = [
    "ENV_KNOBS",
    "EnvKnob",
    "OPTIONS",
    "OptionSpec",
    "Resolved",
    "RunOptions",
    "Source",
    "resolve",
]
