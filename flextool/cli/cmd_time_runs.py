"""``flextool-time-runs``: run scenarios one after another and show where each run's time goes.

Meant for trainers: check that the exercises run fast enough on the computers of a course.  By default it runs
the scenarios of the example and how-to databases that come with FlexTool (built from their JSON files into a
temporary folder); ``--db PATH --scenario S ...`` runs the scenarios of your own project database instead.

Each run is the command the FlexTool window starts (``python -m flextool.cli.cmd_run_flextool ...``) with plots
off unless ``--plots`` says otherwise.  The time of each phase is read from the run's ``timings.csv``:

* **start-up**: the run's wall time minus the engine's ``total`` row (Python start and the imports);
* **read input**: ``read_input`` (reading the database, preparing the input data, setting up the run);
* **preprocess**: ``preprocessing_global`` + ``roll_setup`` + the per-solve ``preprocessing`` and ``lp_build``;
* **solve**: the per-solve ``solve`` rows;
* **outputs**: the per-solve ``handoff`` rows (writing each solve's results) + ``write_outputs/total`` minus the
  viewer and plot rows;
* **viewer**: ``Computed plot plans`` + ``Wrote dispatch metadata`` (what the result viewer needs; these run
  also with plots off);
* **plots**: ``Plotted figures``;
* **untimed**: what is left of the wall time.  A solve through Benders decomposition writes no per-solve rows,
  so its time lands here.

Only the rows named above are read.  ``per_iter/warm_used`` is a flag (1 = a warm start was used), not a time,
and the ``handoff_part`` rows are parts of ``handoff``: neither is added.  The ``cumulative_s`` column is not
used either: two recorders write the file, each from its own start.
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import fnmatch
import io
import json
import os
import platform
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

from flextool._resources import package_data_path

#: Example databases left out of the default set: a settings stub, not an exercise.
_EXCLUDED_SOURCES = ("templates_time_settings_only.json",)

#: Table columns after the scenario: (key, header).
PHASES: tuple[tuple[str, str], ...] = (
    ("startup", "start-up"),
    ("read_input", "read input"),
    ("preprocess", "preprocess"),
    ("solve", "solve"),
    ("outputs", "outputs"),
    ("viewer", "viewer"),
    ("plots", "plots"),
    ("untimed", "untimed"),
)

_PREPROCESS_PHASES = ("preprocessing_global", "roll_setup")
_PREPROCESS_PER_ITER = ("preprocessing", "lp_build")
_VIEWER_STEPS = ("Computed plot plans", "Wrote dispatch metadata")
_PLOT_STEPS = ("Plotted figures",)


@dataclass
class Run:
    """One scenario to run."""
    source: str  # label of the database (the JSON or .sqlite file name)
    db_url: str
    scenario: str
    plots: bool


@dataclass
class RunResult:
    """What one run took."""
    run: Run
    wall_s: float
    ok: bool
    verdict: str
    phases: dict[str, float | None] = field(default_factory=dict)
    log_path: Path | None = None


def canonical_sources() -> list[Path]:
    """The example and how-to JSON databases shipped with FlexTool (the default sources)."""
    folder = package_data_path("schemas/canonical_databases")
    return [p for p in sorted(folder.glob("*.json")) if p.name not in _EXCLUDED_SOURCES]


def _resolve_source(name: str) -> Path:
    """A JSON file path, or the name of a shipped example database (with or without ``.json``)."""
    path = Path(name)
    if path.is_file():
        return path
    folder = package_data_path("schemas/canonical_databases")
    for candidate in (folder / name, folder / f"{name}.json"):
        if candidate.is_file():
            return candidate
    shipped = ", ".join(p.name for p in canonical_sources())
    raise SystemExit(f"No JSON database '{name}'. Shipped ones: {shipped}")


def json_scenarios(json_path: Path) -> list[str]:
    """Scenario names of a FlexTool JSON database, in file order."""
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)
    return [row[0] for row in data.get("scenarios", [])]


def db_scenarios(db_path: Path) -> list[str]:
    """Scenario names of a .sqlite input database, read without writing to it."""
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        return [row[0] for row in con.execute("SELECT name FROM scenario ORDER BY name")]
    finally:
        con.close()


def _select(scenarios: list[str], patterns: list[str] | None) -> list[str]:
    if not patterns:
        return list(scenarios)
    return [s for s in scenarios if any(fnmatch.fnmatchcase(s, p) for p in patterns)]


def _build_sqlite(json_path: Path, folder: Path) -> str:
    """Build a .sqlite from a JSON database (quietly) and return its URL."""
    from flextool.update_flextool.initialize_database import initialize_database

    folder.mkdir(parents=True, exist_ok=True)
    target = folder / f"{json_path.stem}.sqlite"
    with contextlib.redirect_stdout(io.StringIO()):
        initialize_database(str(json_path), str(target))
    return f"sqlite:///{target}"


def phase_seconds(rows: list[dict[str, str]], wall_s: float) -> dict[str, float | None]:
    """Split a run's wall time into the table's phases from its ``timings.csv`` rows.

    Reads the ``seconds`` column of the rows named in the module docstring only (never ``cumulative_s``, never
    ``per_iter/warm_used`` or ``handoff_part``).  Without a ``total`` row (the run stopped early) start-up and
    untimed are ``None``; the other phases are the rows that are there.
    """
    def total(pred) -> float:
        return sum(float(r["seconds"]) for r in rows if pred(r.get("phase", ""), r.get("subphase", "")))

    def has(pred) -> bool:
        return any(pred(r.get("phase", ""), r.get("subphase", "")) for r in rows)

    viewer = total(lambda p, s: p == "write_outputs" and s in _VIEWER_STEPS)
    plots = total(lambda p, s: p == "write_outputs" and s in _PLOT_STEPS)
    write_outputs_total = total(lambda p, s: p == "write_outputs" and s == "total")
    phases: dict[str, float | None] = {
        "read_input": (total(lambda p, s: p == "read_input")
                       if has(lambda p, s: p == "read_input") else None),
        "preprocess": total(lambda p, s: p in _PREPROCESS_PHASES
                            or (p == "per_iter" and s in _PREPROCESS_PER_ITER)),
        "solve": total(lambda p, s: p == "per_iter" and s == "solve"),
        "outputs": (total(lambda p, s: p == "per_iter" and s == "handoff")
                    + max(0.0, write_outputs_total - viewer - plots)),
        "viewer": viewer,
        "plots": plots,
    }
    if has(lambda p, s: p == "total"):
        phases["startup"] = max(0.0, wall_s - total(lambda p, s: p == "total"))
        known = sum(v for v in phases.values() if v is not None)
        phases["untimed"] = wall_s - known
    else:
        phases["startup"] = None
        phases["untimed"] = None
    return phases


def read_timings(path: Path) -> list[dict[str, str]]:
    """The rows of a ``timings.csv``; empty when the file is missing."""
    if not path.is_file():
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _verdict(output_location: Path, subdir: str) -> str:
    status = output_location / "output_parquet" / subdir / "_run_status.json"
    try:
        data = json.loads(status.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return ""
    return str(data.get("verdict") or data.get("state") or "")


def run_one(run: Run, folder: Path) -> RunResult:
    """Run one scenario in ``folder`` (work folder, outputs and log) and time it."""
    work = folder / "work"
    out = folder / "out"
    folder.mkdir(parents=True, exist_ok=True)
    write_methods = ["parquet", "plot"] if run.plots else ["parquet"]
    cmd = [
        sys.executable, "-m", "flextool.cli.cmd_run_flextool", run.db_url,
        "--scenario-name", run.scenario,
        "--work-folder", str(work),
        "--output-location", str(out),
        "--output-subdir", run.scenario,
        "--write-methods", *write_methods,
        "--run-launcher", "time-runs",
    ]
    env = dict(os.environ, FLEXTOOL_PHASE_TIMING="1")
    log_path = folder / "run.log"
    with open(log_path, "w", encoding="utf-8", errors="replace") as log:
        t0 = time.perf_counter()
        proc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, env=env, check=False)
        wall_s = time.perf_counter() - t0
    rows = read_timings(out / "output_csv" / run.scenario / "timings.csv") or read_timings(
        work / "solve_data" / "timings.csv")
    verdict = _verdict(out, run.scenario) or ("failed" if proc.returncode else "")
    return RunResult(run=run, wall_s=wall_s, ok=proc.returncode == 0, verdict=verdict,
                     phases=phase_seconds(rows, wall_s), log_path=log_path)


def machine_lines() -> list[str]:
    """The computer the runs are timed on: processor, cores, memory, system, versions."""
    cpu = platform.processor() or platform.machine()
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.is_file():
        for line in cpuinfo.read_text(errors="replace").splitlines():
            if line.startswith("model name"):
                cpu = line.split(":", 1)[1].strip()
                break
    try:
        import psutil

        physical = psutil.cpu_count(logical=False)
        ram_gb = psutil.virtual_memory().total / 1024 ** 3
        free_gb = psutil.virtual_memory().available / 1024 ** 3
        memory = f"{ram_gb:.1f} GB RAM ({free_gb:.1f} GB free)"
    except Exception:
        physical = None
        memory = "RAM unknown"
    logical = os.cpu_count()
    cores = f"{physical} cores, {logical} threads" if physical else f"{logical} threads"
    try:
        from importlib import metadata

        version = metadata.version("flextool")
    except Exception:
        version = "unknown"
    return [
        f"Machine: {cpu}, {cores}, {memory}",
        f"System:  {platform.system()} {platform.release()}, Python {platform.python_version()}, "
        f"FlexTool {version}",
    ]


def _fmt(value: float | None) -> str:
    return "" if value is None else f"{value:.1f}"


def format_table(results: list[RunResult], target_s: float) -> list[str]:
    """A fixed-width table, slowest run first; ``!`` marks a run over ``target_s``."""
    labels = [_label(r.run) for r in results]
    width = max([len("scenario"), *(len(x) for x in labels)])
    src_width = max([len("source"), *(len(r.run.source) for r in results)])
    head = (f"  {'scenario':<{width}}  {'source':<{src_width}}  {'wall':>6}  "
            + "  ".join(f"{h:>{len(h)}}" for _, h in PHASES) + "  status")
    lines = [head, "-" * len(head)]
    for result, label in sorted(zip(results, labels), key=lambda x: -x[0].wall_s):
        flag = "!" if result.wall_s > target_s else " "
        status = result.verdict if result.ok else f"FAILED ({result.verdict})" if result.verdict else "FAILED"
        lines.append(
            f"{flag} {label:<{width}}  {result.run.source:<{src_width}}  {result.wall_s:>6.1f}  "
            + "  ".join(f"{_fmt(result.phases.get(key)):>{len(h)}}" for key, h in PHASES)
            + f"  {status}")
    return lines


def _label(run: Run) -> str:
    return f"{run.scenario} (plots)" if run.plots else run.scenario


def write_csv(path: Path, results: list[RunResult], target_s: float) -> None:
    """The table as a CSV file (seconds, empty where unknown)."""
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["source", "scenario", "plots", "wall_s", *(f"{k}_s" for k, _ in PHASES),
                         "over_target", "ok", "verdict"])
        for r in sorted(results, key=lambda x: -x.wall_s):
            writer.writerow([r.run.source, r.run.scenario, int(r.run.plots), f"{r.wall_s:.3f}",
                             *("" if r.phases.get(k) is None else f"{r.phases[k]:.3f}" for k, _ in PHASES),
                             int(r.wall_s > target_s), int(r.ok), r.verdict])


def _print_log_tail(log_path: Path | None, *, keep: bool, lines: int = 15) -> None:
    """The end of a failed run's log (the log itself is removed at the end unless ``--keep``)."""
    if log_path is None or not log_path.is_file():
        return
    tail = log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-lines:]
    for line in tail:
        print(f"    | {line}")
    if keep:
        print(f"    full log: {log_path}")


def build_parser() -> argparse.ArgumentParser:
    """The ``flextool-time-runs`` command line."""
    parser = argparse.ArgumentParser(
        prog="flextool-time-runs",
        description="Run scenarios one after another and show where each run's time goes (start-up, reading "
                    "the input, preprocessing, solving, outputs, viewer data, plots), so trainers can check "
                    "that exercises run fast enough. Default: every scenario of the example and how-to "
                    "databases that come with FlexTool. A Benders solve writes no per-solve timing rows: its "
                    "time is shown as untimed.",
    )
    parser.add_argument("--db", metavar="PATH", action="append", default=None,
                        help="A .sqlite input database (a path or a sqlite:/// URL) to run instead of the "
                             "examples; can be given several times. It is read, never changed; the results "
                             "go to a temporary folder.")
    parser.add_argument("--source", metavar="JSON", action="append", default=None,
                        help="A JSON database to build and run: a file, or the name of a shipped one (e.g. "
                             "templates_examples or howto_demand). Default, without --db: all shipped "
                             "examples and how-tos.")
    parser.add_argument("--scenario", metavar="NAME", nargs="+", default=None,
                        help="Scenarios to run: names or patterns (* and ?), e.g. coal 'network_*'. "
                             "Default: every scenario of each database.")
    parser.add_argument("--plots", choices=("off", "on", "both"), default="off",
                        help="Write the scenario plots (png) in the runs: off (default, as for exercises "
                             "with plots switched off), on, or both (each scenario twice).")
    parser.add_argument("--target", type=float, default=60.0, metavar="SECONDS",
                        help="Mark runs that take longer than this (default 60 s).")
    parser.add_argument("--strict", action="store_true",
                        help="Exit with code 2 when a run took longer than --target (a failed run always "
                             "gives exit code 1).")
    parser.add_argument("--keep", metavar="DIR", default=None,
                        help="Keep the runs' work folders, results and logs in DIR (default: a temporary "
                             "folder that is removed at the end).")
    parser.add_argument("--csv", metavar="FILE", default=None, help="Also write the table to a CSV file.")
    return parser


def collect_runs(args: argparse.Namespace, build_folder: Path) -> list[Run]:
    """The runs the command line asks for, with the JSON sources built into ``build_folder``."""
    plot_modes = {"off": (False,), "on": (True,), "both": (False, True)}[args.plots]
    sources: list[tuple[str, str, list[str]]] = []  # (label, url, scenarios)
    for db in args.db or []:
        from flextool.update_flextool.frozen import sqlite_path_from

        path = sqlite_path_from(db)
        if path is None or not path.is_file():
            raise SystemExit(f"No .sqlite file at {db}")
        path = path.resolve()
        sources.append((path.name, f"sqlite:///{path}", db_scenarios(path)))
    json_paths = ([_resolve_source(s) for s in args.source] if args.source
                  else ([] if args.db else canonical_sources()))
    for json_path in json_paths:
        scenarios = _select(json_scenarios(json_path), args.scenario)
        if scenarios:
            sources.append((json_path.stem, _build_sqlite(json_path, build_folder), scenarios))
    runs = [Run(label, url, scenario, plots)
            for label, url, scenarios in sources
            for scenario in _select(scenarios, args.scenario)
            for plots in plot_modes]
    unmatched = [p for p in args.scenario or []
                 if not any(fnmatch.fnmatchcase(r.scenario, p) for r in runs)]
    if unmatched:
        raise SystemExit(f"No scenario matches: {', '.join(unmatched)}")
    return runs


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.keep:
        base = Path(args.keep).resolve()
        base.mkdir(parents=True, exist_ok=True)
    else:
        base = Path(tempfile.mkdtemp(prefix="flextool_time_runs_"))
    results: list[RunResult] = []
    interrupted = False
    try:
        for line in machine_lines():
            print(line)
        runs = collect_runs(args, base / "input")
        if not runs:
            print("No scenarios to run.")
            return 1
        print(f"{len(runs)} runs, one after another (plots {args.plots}, target {args.target:g} s)\n",
              flush=True)
        try:
            for i, run in enumerate(runs, start=1):
                folder = base / "runs" / f"{i:03d}_{run.scenario}{'_plots' if run.plots else ''}"
                result = run_one(run, folder)
                results.append(result)
                state = "ok" if result.ok else "FAILED"
                print(f"[{i}/{len(runs)}] {_label(run)} ({run.source}): {result.wall_s:.1f} s, {state}",
                      flush=True)
                if not result.ok:
                    _print_log_tail(result.log_path, keep=bool(args.keep))
        except KeyboardInterrupt:
            interrupted = True
            print("\nStopped; the table shows the runs that finished.")
    finally:
        if not args.keep:
            shutil.rmtree(base, ignore_errors=True)
    if not results:
        return 130 if interrupted else 1
    print()
    for line in machine_lines()[:1]:
        print(line)
    print("Seconds per phase; ! = over the target.")
    for line in format_table(results, args.target):
        print(line)
    failed = [r for r in results if not r.ok]
    slow = [r for r in results if r.wall_s > args.target]
    print(f"\n{len(results)} runs, {len(slow)} over {args.target:g} s (marked !), {len(failed)} failed.")
    if args.csv:
        write_csv(Path(args.csv), results, args.target)
        print(f"Table written to {args.csv}")
    if args.keep:
        print(f"Runs kept in {base}")
    if interrupted:
        return 130
    if failed:
        return 1
    if args.strict and slow:
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
