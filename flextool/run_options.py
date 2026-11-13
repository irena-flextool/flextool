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
5. Solver options keep their own stack, the same for every solver (HiGHS: baseline < ``highs.opt``; a
   commercial solver: ``<solver>.opt``; then ``solve.solver_mip_gap`` < ``solve.solver_arguments`` < operator
   layer); this registry supplies only the operator layer.

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


def split_path_list(text: str) -> tuple[str, ...]:
    """The folders of an environment value that lists several (``FLEXTOOL_SOLVER_CONFIG_DIR``): split at
    :data:`os.pathsep` (``:`` on Linux and macOS, ``;`` on Windows, as ``PATH``), each stripped, empty parts
    dropped."""
    return tuple(part.strip() for part in text.split(os.pathsep) if part.strip())


def _path_list(text: str) -> tuple[str, ...] | None:
    return split_path_list(text) or None


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
            key="threads", flag="--threads", env="FLEXTOOL_THREADS", db=None,
            parse=_positive_int, default=None, audience="user",
            help="threads for one FlexTool run: FlexTool's data processing (at most 8 unless POLARS_MAX_THREADS) "
                 "and the solver (HiGHS and the commercial solvers' thread count); unset uses the available cores "
                 "and leaves the solvers' thread settings as they are (a Spine Toolbox run: 1 thread for all)",
        ),
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
            help="solver log on the screen: silent hides every solver's log (a commercial solver's is still saved as "
                 "solve_data/subprocess/<solve>.log), normal shows it, verbose adds HiGHS' developer log (HiGHS only)",
        ),
        _spec(
            key="solver_time_limit", flag="--solver-time-limit", env="FLEXTOOL_SOLVER_TIME_LIMIT", db=None,
            parse=_positive_float, default=None, audience="user",
            help="time limit of each solve in seconds, on every solver (HiGHS time_limit, Gurobi TimeLimit, CPLEX "
                 "timelimit, Xpress MAXTIME, COPT TimeLimit); unset means no limit",
        ),
        _spec(
            key="solver_mip_gap", flag="--solver-mip-gap", env="FLEXTOOL_SOLVER_MIP_GAP", db="solve.solver_mip_gap",
            parse=_non_negative_float, default=None, audience="user",
            help="relative MIP gap of every solve, on every solver (HiGHS mip_rel_gap, Gurobi MIPGap, CPLEX mip "
                 "tolerances mipgap, Xpress MIPRELSTOP, COPT RelGap); unset keeps the solve's solver_mip_gap, else "
                 "the solver's option file",
        ),
        # The former environment names of the two options above, kept for HiGHS only: a shell or CI setting made
        # for HiGHS must not start limiting a commercial solver.  The flag and the FLEXTOOL_SOLVER_* twin win.
        _spec(
            key="highs_time_limit", flag=None, env="FLEXTOOL_HIGHS_TIME_LIMIT", db=None,
            parse=_positive_float, default=None, audience="user",
            help="HiGHS-only time limit of each solve in seconds; --solver-time-limit and "
                 "FLEXTOOL_SOLVER_TIME_LIMIT win over it",
        ),
        _spec(
            key="highs_mip_gap", flag=None, env="FLEXTOOL_HIGHS_MIP_GAP", db=None,
            parse=_non_negative_float, default=None, audience="user",
            help="HiGHS-only relative MIP gap of every solve; --solver-mip-gap and FLEXTOOL_SOLVER_MIP_GAP win "
                 "over it",
        ),
        _spec(
            key="solver_config_dir", flag="--solver-config-dir", env="FLEXTOOL_SOLVER_CONFIG_DIR", db=None,
            parse=_path_list, default=None, audience="user",
            help="folders of the solver option files (highs.opt, gurobi.opt, ...), highest first: each solver's "
                 "file comes from the first folder that has it, else FlexTool's default; repeat the flag for "
                 "several folders (the environment variable separates them with os.pathsep, ':' or on Windows "
                 "';'); unset looks next to the input database, in its project and workspace, the Spine Toolbox "
                 "project folder and the current folder",
        ),
        _spec(
            key="workspace_solver_config_dir", flag="--workspace-solver-config-dir", env=None, db=None,
            parse=_path_list, default=None, audience="user",
            help="solver option folders the launcher found, highest first (the FlexTool window passes its "
                 "project's solver_config, then its workspace's; repeat the flag for several); the files of each "
                 "are trusted only when its workspace or project is one this computer set up; above the "
                 "--solver-config-dir folders when both are given",
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
    """An environment variable that is not the twin of a flag: process-level, developer or test-only.

    *read_by* names who reads it: ``"flextool"`` (this package, or for a test knob the test suite) or the
    library that does (``"polars"``) when FlexTool only documents it.  ``tests/test_env_knob_registry.py`` checks
    that every variable FlexTool reads is registered here or in :data:`OPTIONS`, that each is on its page, and
    that no page lists one that is no longer read.
    """

    name: str
    audience: Audience
    page: str  # the documentation page that describes it
    help: str
    read_by: str = "flextool"


USER_PAGE = "docs/env_vars.md"
DEV_PAGE = "docs/dev/env_vars.md"
TEST_PAGE = "docs/dev/testing.md"

#: The page of each audience; :data:`OPTIONS` entries are documented on the page of their audience.
AUDIENCE_PAGES: dict[str, str] = {"user": USER_PAGE, "dev": DEV_PAGE, "test": TEST_PAGE}


def _user(name: str, help: str, read_by: str = "flextool") -> EnvKnob:
    return EnvKnob(name, "user", USER_PAGE, help, read_by)


def _dev(name: str, help: str) -> EnvKnob:
    return EnvKnob(name, "dev", DEV_PAGE, help)


def _test(name: str, help: str) -> EnvKnob:
    return EnvKnob(name, "test", TEST_PAGE, help)


# Developer and test variables keep their names (no ``FLEXTOOL_DEV_*`` prefix); the audience here and the page
# they are on mark them as developer-only.
ENV_KNOBS: dict[str, EnvKnob] = {
    knob.name: knob
    for knob in (
        # --- user: process-level and GUI settings ---------------------------------------------------------
        _user("FLEXTOOL_NO_ENV_CHECK", "skip the start-up check of the native libraries (polars, highspy)"),
        _user("FLEXTOOL_NO_UPDATE_CHECK", "the GUI does not look for a newer FlexTool at start"),
        _user("FLEXTOOL_DPI", "GUI screen resolution override"),
        _user("FLEXTOOL_SLICE", "systemd slice for GUI-started runs (Linux)"),
        _user("MALLOC_ARENA_MAX", "glibc malloc arena cap; run_flextool sets 4 unless you set it"),
        _user("POLARS_PREFER_PKG", "which polars build loads first; a value you set is kept"),
        _user("POLARS_FORCE_PKG", "force one polars build; also skips FlexTool's processor check"),
        _user("FLEXTOOL_RUN_LOG_MAX_BYTES", "bytes of a run's _run.log kept from its end (4 MiB, at most 256 MiB); "
              "the first 64 KiB are kept too"),
        _user("FLEXTOOL_ALLOW_UNLISTED_SOLVER_PARAMETERS", "1 takes solver parameter names that are not in "
              "FlexTool's lists from input databases (default off; also solver_settings.json in the user config "
              "folder)"),
        *(_user(name, f"the program (and its arguments, as a JSON list) that starts in front of {label}'s "
                "program; empty or [] means none (also solver_settings.json)")
          for name, label in (("FLEXTOOL_SOLVER_WRAPPER_GUROBI", "Gurobi"), ("FLEXTOOL_SOLVER_WRAPPER_CPLEX", "CPLEX"),
                              ("FLEXTOOL_SOLVER_WRAPPER_XPRESS", "Xpress"), ("FLEXTOOL_SOLVER_WRAPPER_COPT", "COPT"))),
        _user("FLEXTOOL_MIGRATION_LOCK_TIMEOUT", "Windows: seconds a migration waits for the lock of another "
              "migration of the same database (600); not read on Linux or macOS"),
        _user("POLARS_MAX_THREADS", "polars thread pool size, in place of FlexTool's limit of 8 but at most the "
              "run's --threads; the run header prints the effective count", read_by="polars"),
        # --- dev: solve acceptance ------------------------------------------------------------------------
        _dev("FLEXTOOL_ACCEPT_NEAR_OPTIMAL", "0 rejects every solver status other than optimal, a MIP stopped by a "
             "limit included (default on)"),
        _dev("FLEXTOOL_ACCEPT_PD_GAP", "largest primal-dual gap of an accepted near-optimal solve (1e-2)"),
        _dev("FLEXTOOL_ACCEPT_PRIMAL_REL", "relative primal infeasibility of an accepted near-optimal solve; "
             "replaces the default max(1e-6, 10 x primal_feasibility_tolerance)"),
        _dev("FLEXTOOL_ACCEPT_PRIMAL_ABS_MARGIN",
             "absolute primal infeasibility allowed, as a multiple of HiGHS' feasibility tolerance (10)"),
        # --- dev: Benders ---------------------------------------------------------------------------------
        _dev("FLEXTOOL_BENDERS_CUT_POLICY", "master cut compaction policy: slack (default) or dominance"),
        _dev("FLEXTOOL_BENDERS_CUT_WINDOW", "trial-point window of the dominance cut policy (10)"),
        _dev("FLEXTOOL_BENDERS_CUT_COMPACT_AT", "master cut count that triggers compaction; 0 / unset is off"),
        _dev("FLEXTOOL_BENDERS_MAX_STALL", "iterations without progress before the stall guard stops (8)"),
        _dev("FLEXTOOL_BENDERS_TRUST_REGION_RADIUS", "trust-region half-width on the coupling flows; unset is off"),
        _dev("FLEXTOOL_BENDERS_REGION_DUALS", "region dual for the cuts: basic (default) or interior"),
        _dev("FLEXTOOL_BENDERS_PIN_DIAGNOSTIC",
             "parquet of monolith coupling flows: evaluate the decomposition there once instead of iterating"),
        _dev("FLEXTOOL_BENDERS_PIN_MONOLITH_OBJ", "monolith objective for the pin diagnostic's exactness check"),
        _dev("FLEXTOOL_BENDERS_PIN_BLOWUP_MULT", "pin diagnostic: region cost multiple that counts as blow-up (100)"),
        _dev("FLEXTOOL_BENDERS_VERBOSE", "show the HiGHS log of every Benders master and region solve"),
        # --- dev: checks and modes ------------------------------------------------------------------------
        _dev("FLEXTOOL_AUTOSCALE_STRICT", "a missing autoscale registry entry raises instead of solving unscaled"),
        _dev("FLEXTOOL_AXIS_ENUMS", "experimental axis-Enum mode for a whole run"),
        _dev("FLEXTOOL_FORCE_ROW_SCALING", "turn on the legacy node / group balance row scaling"),
        _dev("FLEXTOOL_SKIP_SOLVER_PROBE", "skip the start-up probe of commercial solver licences"),
        _dev("FLEXTOOL_NATIVE_SCENARIO", "scenario name for run_chain when the work folder name does not give it"),
        _dev("FLEXTOOL_AUDIT_SOURCES", "log every externally injected Provider entry to audit_sources.log"),
        _dev("FLEXTOOL_LEVEL_RELEASE_AUDIT", "record solves whose HiGHS instance outlives its level"),
        _dev("FLEXTOOL_DUMP_CSVS", "with --csv-dump, also write the eight largest debug CSVs"),
        _dev("FLEXTOOL_COLD_KEEP_PROVIDER", "under --save-memory keep the data provider across cold solves"),
        _dev("FLEXTOOL_RSS_BUDGET_MB", "RSS budget above which the data provider evicts cached frames"),
        _dev("FLEXTOOL_WARM_AB_MARGIN", "warm start is kept when within this fraction of the cold time (0.10)"),
        _dev("FLEXTOOL_DISABLE_AUTOSCALE_CACHE", "recompute the autoscale plan in every roll"),
        _dev("FLEXTOOL_DISABLE_OUTPUT_HOIST", "fetch solution arrays per output instead of once per solve"),
        # --- dev: profiling -------------------------------------------------------------------------------
        _dev("FLEXTOOL_MEMORY_DIAGNOSTICS", "memory recorder with tracemalloc and memory_diagnostics.csv"),
        _dev("FLEXTOOL_MEMORY_VERBOSE", "print every phase of the memory table"),
        _dev("FLEXTOOL_MEM_SAMPLER", "background RSS sampler"),
        _dev("FLEXTOOL_MEM_SAMPLER_LOG", "RSS sampler log path"),
        _dev("FLEXTOOL_MEM_SAMPLER_INTERVAL_MS", "RSS sampler interval in milliseconds (100)"),
        _dev("FLEXTOOL_PHASE_TIMING", "per-iteration rows in timings.csv"),
        _dev("FLEXTOOL_PHASE_PROFILE", "RSS line per orchestration phase on stderr"),
        _dev("FLEXTOOL_PYRAMID_PROFILE", "RSS line per preprocessing emit on stderr"),
        _dev("FLEXTOOL_AUTOSCALE_PROFILE", "RSS markers around the autoscale steps"),
        _dev("FLEXTOOL_BUILD_PROFILE", "RSS line per model-build step on stderr"),
        _dev("FLEXTOOL_LOADFLEX_PROFILE", "RSS line per data-load step on stderr"),
        _dev("POLAR_HIGH_LAYER2_PROFILE", "profile line per term of the autoscale Layer 2 walk"),
        _dev("POLAR_HIGH_RANGES_MAX_FAMILY_ROWS",
             "row cap per constraint family of Layer 2 bucketing with an old polar-high (1000000)"),
        # --- dev: run record ------------------------------------------------------------------------------
        _dev("FLEXTOOL_RUN_LOG_OWNER", "set by the GUI / flextool-run for the engine it starts: they write _run.log "
             "from its output pipe; the engine removes it"),
        _dev("FLEXTOOL_RUN_STOP_FILE", "set by the GUI for a calibration / representative-period job and inherited "
             "by its flextool-run supervisors: the file whose appearance stops their engine runs (recorded as "
             "killed); the supervisor leaves it out of the engine's environment"),
        # --- dev: user folders ----------------------------------------------------------------------------
        _dev("FLEXTOOL_USER_HOME", "folder in place of FlexTool's user config and cache folders (config/, cache/); "
             "the test suite sets it for every test"),
        # --- dev: process level ---------------------------------------------------------------------------
        _dev("POLARS_ENGINE_AFFINITY", "polars engine; FlexTool sets in-memory, the only engine it verifies"),
        _dev("_RJEM_MALLOC_CONF", "jemalloc settings of polars' allocator; FlexTool sets page decay"),
        # --- test suite -----------------------------------------------------------------------------------
        _test("FLEXTOOL_NATIVE_PARITY_SWEEP", "run the slow parity sweep over every fixture"),
        _test("FLEXTOOL_WRITE_SOLVER_HANDOFF_GOLDEN", "1 rewrites the solver hand-off golden record"),
        _test("FLEXTOOL_TEST_RSS_BUDGET_MB", "RSS budget of the input-derivation memory test (1500)"),
        _test("FLEXTOOL_TEST_KEYRING_LOG", "set by the root conftest: the file the session's keyring violations go to"),
        _test("FLEXTOOL_TEST_SANDBOX", "set by the root conftest: the session's temp folder that HOME, XDG_* and "
              "APPDATA/LOCALAPPDATA point into"),
        _test("FLEXTOOL_TEST_REAL_HOME", "set by the root conftest: the real home folder, which no test may write under"),
        _test("FLEXTOOL_TEST_REAL_USER_DIRS", "set by the root conftest: the real FlexTool user folders, which no test "
              "may reach"),
        _test("FLEXTOOL_SHIM_RECORD", "set by the commercial_shim tests: the file each solver shim appends its "
              "call record to"),
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
    falls through to the next tier.  A numeric flag value is held to the same rule as its environment variable
    (``--solver-time-limit 0`` is ignored like ``FLEXTOOL_HIGHS_TIME_LIMIT=0``).  Never writes the environment.
    """
    if cli is not None:
        if _cli_value_valid(spec, cli):
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


def _cli_value_valid(spec: OptionSpec, cli: Any) -> bool:
    """False, with one warning, when a numeric flag value fails the rule of :attr:`OptionSpec.parse`.

    argparse already converts the text to a number (``type=float`` / ``int``) and checks ``choices``; the range
    rule (positive, at least 0, ...) lives in ``parse``, so the flag is checked with it here.
    """
    if isinstance(cli, bool) or not isinstance(cli, (int, float)) or spec.flag is None:
        return True
    try:
        spec.parse(_format_value(cli))
    except ValueError as exc:
        _logger.warning("Ignoring %s %s (%s).", spec.flag, _format_value(cli), exc)
        return False
    return True


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
            key: resolve(spec, cli=_cli_value(spec, getattr(args, spec.dest, None)) if spec.flag else None, environ=env)
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
            if isinstance(r.value, (tuple, list)):  # a repeatable flag: once per value
                for item in r.value:
                    argv.extend([spec.flag, _format_value(item)])
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


def _cli_value(spec: OptionSpec, value: Any) -> Any:
    """A parsed flag value as the registry holds it: a repeatable flag's list (``action="append"``) as a tuple, an
    empty one as unset; one folder of a folder-list option as a 1-tuple."""
    if isinstance(value, list):
        return tuple(str(v) for v in value) or None
    if spec.parse is _path_list and isinstance(value, (str, os.PathLike)):
        return (str(value),) if str(value).strip() else None
    return value


def _format_value(value: Any) -> str:
    # ``repr`` of a float is the shortest text that parses back to the same float.
    if isinstance(value, float):
        return repr(value)
    if isinstance(value, (tuple, list)):  # several folders: shown as an environment value lists them
        return os.pathsep.join(str(item) for item in value)
    return str(value)


__all__ = [
    "AUDIENCE_PAGES",
    "ENV_KNOBS",
    "EnvKnob",
    "OPTIONS",
    "OptionSpec",
    "Resolved",
    "RunOptions",
    "Source",
    "resolve",
    "split_path_list",
]
