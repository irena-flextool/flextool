"""Output snapshot harness — item 163 (c24).

Mirrors ``tools/lp_snapshot.py``'s harness, but for the output-writer pathway instead of the LP: run
each fixture the way the CLI does (``run_chain_from_db`` -> ``write_outputs``), snapshot its output
set, and compare two snapshots with the F7 canonical compare
(:func:`tests._support.output_compare.diff_output_trees`).

Reuses ``tools/lp_snapshot.py``'s ``Fixture`` / ``_build_fixture_db`` / ``_fixture_by_name`` (which
in turn hold its ``FIXTURES`` / ``PATCHES`` tables) so the fixture JSON + patch table is defined
exactly once.  A fixture name not in that table falls back
to a bare ``tests/fixtures/tests.json`` scenario of the same name (unpatched) — covers GATE15 names
such as ``multi_year_one_solve`` / ``wind_battery_invest`` that ``lp_snapshot`` does not carry.

Usage::

    # write OUT/<fixture>/{output_raw,out/output_parquet/<scenario>,solve_data/<mirror>} per fixture
    python tools/output_snapshot.py snapshot OUT [--fixtures a,b] [--csv-dump] [-j N]

    # --dirty-with: run DIRTIER first in the same work/ (csv_dump=True + a snapshot), then the target
    python tools/output_snapshot.py snapshot OUT --fixtures base --dirty-with co2_period_rolling

    # compare two snapshots (F7 canonical compare; prints the differing fixtures + files)
    python tools/output_snapshot.py compare A B [--fixtures a,b]

    python tools/output_snapshot.py list   # the GATE15 fixture names
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import os
import secrets
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
TESTS_DIR = REPO / "tests"
TOOLS_DIR = Path(__file__).resolve().parent

sys.path.insert(0, str(TOOLS_DIR))
from lp_snapshot import (  # noqa: E402
    Fixture,
    _build_fixture_db,
    _fixture_by_name as _lp_fixture_by_name,
)

OUTPUT_CONFIG = REPO / "templates" / "default_plots.yaml"

# GATE15 (design_163.md §6): the output gate's fixture set for the 163 commits.
GATE15: tuple[str, ...] = (
    "base", "multi_year", "multi_year_one_solve",
    "5weeks_invest_fullYear_dispatch_coal_wind", "co2_period_rolling",
    "rolling_separate_dispatch", "nested_fix_quantity", "nested_rolling_storage_uc",
    "storage_sequence_start_end", "wind_battery_invest",
    "wind_battery_invest_lifetime_renew_4solve", "stoch_recourse",
    "stochastic_1_week_rolling_wind", "invest_limits", "conn_exact_flow",
)

# Mirrors the ``handoff_writers.MIRROR_FILES`` tuple design_163.md §3.4 adds in commit 163-4 — the
# six solve_data/ write-only handoff-mirror CSVs a run leaves behind.  Hardcoded here (163-0 runs
# before 163-4 exists); keep in sync with that tuple once it lands.
MIRROR_FILES: tuple[str, ...] = (
    "p_entity_divested.csv",
    "fix_storage_quantity.csv",
    "fix_storage_usage.csv",
    "p_roll_continue_state.csv",
    "period_capacity.csv",
    "p_entity_period_existing_capacity.csv",
)


def _fixture_by_name(name: str) -> Fixture:
    """``name`` resolved against ``lp_snapshot.FIXTURES``, falling back to a bare
    ``tests/fixtures/tests.json`` scenario of the same name (unpatched) when absent there."""
    by_name = _lp_fixture_by_name()
    if name in by_name:
        return by_name[name]
    return Fixture(name, "tests/fixtures/tests.json", name, patch="none")


# ---------------------------------------------------------------------------
# Worker (runs inside the per-fixture subprocess)
# ---------------------------------------------------------------------------

def _run_cli_like(db_url: str, scenario: str, work: Path, solver_cfg: Path, *,
                  csv_dump: bool = False):
    """Run one fixture the way the CLI does: the cascade, then ``write_outputs`` with the CLI's
    call shape (``cmd_run_flextool.py`` l.966-985).  Returns the last :class:`OrchestrationStep`."""
    from flextool.engine import run_chain_from_db
    from flextool.process_outputs.write_outputs import write_outputs

    steps = run_chain_from_db(
        input_db_url=db_url,
        scenario_name=scenario,
        work_folder=work,
        solver_config_dir=solver_cfg,
        warm=False,
        csv_dump=csv_dump,
    )
    if not steps:
        raise RuntimeError(f"{scenario}: run_chain_from_db returned no steps")
    last = next(reversed(list(steps.values())))
    if csv_dump and last.flex_data_provider is not None:
        last.flex_data_provider.snapshot_processed_inputs(work)
    write_outputs(
        scenario,
        output_location=str(work / "out"),
        subdir=scenario,
        output_config_path=str(OUTPUT_CONFIG),
        write_methods=["parquet"],
        raw_output_dir=str(work / "output_raw"),
        flex_data=last.flex_data,
        solution=last.solution,
        solve_name=last.solve_name,
        flex_data_provider=last.flex_data_provider,
    )
    return last


def _copy_output_set(work: Path, out_dir: Path, scenario: str) -> None:
    """Copy this run's output set (``output_raw/``, ``out/output_parquet/<scenario>/``, the
    ``solve_data/`` handoff mirrors) into *out_dir* for later comparison."""
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_src = work / "output_raw"
    if raw_src.is_dir():
        shutil.copytree(raw_src, out_dir / "output_raw")
    parquet_src = work / "out" / "output_parquet" / scenario
    if parquet_src.is_dir():
        shutil.copytree(parquet_src, out_dir / "out" / "output_parquet" / scenario)
    sd_src = work / "solve_data"
    for name in MIRROR_FILES:
        p = sd_src / name
        if p.is_file():
            dst = out_dir / "solve_data"
            dst.mkdir(parents=True, exist_ok=True)
            shutil.copy(p, dst / name)


def _worker_main(json_name: str, scenario: str, patch: str, out_dir: Path, *,
                 csv_dump: bool = False, dirty_with: str = "") -> int:
    os.environ.setdefault("FLEXTOOL_SKIP_SOLVER_PROBE", "1")
    sys.path.insert(0, str(TESTS_DIR))

    target = Fixture("target", json_name, scenario, patch)
    out_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="output_snapshot_") as tmp:
        tmp_path = Path(tmp)
        url, _ = _build_fixture_db(target, tmp_path, "fixture")
        solver_cfg = tmp_path / "solver_config"
        solver_cfg.mkdir()
        shutil.copy(TESTS_DIR / "highs.opt", solver_cfg / "highs.opt")
        work = tmp_path / "work"
        work.mkdir()
        if dirty_with:
            dirtier = _fixture_by_name(dirty_with)
            d_url, _ = _build_fixture_db(dirtier, tmp_path, "dirtier")
            try:
                _run_cli_like(d_url, dirtier.scenario, work, solver_cfg, csv_dump=True)
            except Exception as exc:
                # The dirtier itself may legitimately crash the target below (F6) — record it as
                # the dirt, don't abort: the whole point of --dirty-with is to leave the mess behind.
                (out_dir / "_dirtier_error.txt").write_text(f"{type(exc).__name__}: {exc}\n")
        try:
            _run_cli_like(url, scenario, work, solver_cfg, csv_dump=csv_dump)
        except Exception as exc:
            (out_dir / "_error.txt").write_text(f"{type(exc).__name__}: {exc}\n")
            return 2
        _copy_output_set(work, out_dir, scenario)
    return 0


def cmd_worker(args: argparse.Namespace) -> int:
    return _worker_main(
        args.json, args.scenario, args.patch, Path(args.out),
        csv_dump=args.csv_dump, dirty_with=args.dirty_with,
    )


# ---------------------------------------------------------------------------
# snapshot / compare / list
# ---------------------------------------------------------------------------

def _run_fixture(name: str, out_root: Path, *, csv_dump: bool, dirty_with: str) -> tuple[str, int, float, str]:
    fx = _fixture_by_name(name)
    out_dir = out_root / name
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    env = dict(os.environ)
    env["PYTHONHASHSEED"] = str(secrets.randbelow(2**32 - 1) + 1)
    env["FLEXTOOL_SKIP_SOLVER_PROBE"] = "1"
    cmd = [sys.executable, str(Path(__file__).resolve()), "_worker",
           "--json", fx.json, "--scenario", fx.scenario, "--patch", fx.patch,
           "--out", str(out_dir)]
    if csv_dump:
        cmd.append("--csv-dump")
    if dirty_with:
        cmd += ["--dirty-with", dirty_with]
    t0 = time.monotonic()
    proc = subprocess.run(cmd, env=env, cwd=str(REPO), capture_output=True, text=True)
    dt = time.monotonic() - t0
    (out_dir / "_run.log").write_text(
        f"PYTHONHASHSEED={env['PYTHONHASHSEED']}\n"
        f"csv_dump={csv_dump} dirty_with={dirty_with!r}\n"
        f"exit={proc.returncode}\nseconds={dt:.1f}\n"
        f"--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}\n"
    )
    tail = "" if proc.returncode == 0 else proc.stderr.strip()[-2000:]
    return name, proc.returncode, dt, tail


def _resolve_names(args: argparse.Namespace) -> list[str]:
    if args.fixtures:
        return [n.strip() for n in args.fixtures.split(",") if n.strip()]
    return list(GATE15)


def cmd_snapshot(args: argparse.Namespace) -> int:
    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    names = _resolve_names(args)
    print(f"mode: csv_dump={args.csv_dump} dirty_with={args.dirty_with!r}", flush=True)
    t0 = time.monotonic()
    failed = []
    with cf.ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futs = [pool.submit(_run_fixture, name, out_root,
                            csv_dump=args.csv_dump, dirty_with=args.dirty_with)
                for name in names]
        for fut in cf.as_completed(futs):
            name, rc, dt, tail = fut.result()
            status = "ok" if rc == 0 else f"FAILED (exit {rc})"
            print(f"{name:45s} {status:16s} {dt:6.1f}s", flush=True)
            if rc != 0:
                failed.append(name)
                print(f"    {tail}", flush=True)
    print(f"\n{len(names)} fixtures, {time.monotonic() - t0:.1f}s wall")
    if failed:
        print(f"FAILED: {', '.join(sorted(failed))}", file=sys.stderr)
        return 1
    return 0


def cmd_compare(args: argparse.Namespace) -> int:
    sys.path.insert(0, str(TESTS_DIR))
    from _support.output_compare import diff_output_trees

    a_root, b_root = Path(args.a), Path(args.b)
    names = _resolve_names(args)
    any_diff = False
    for name in names:
        fx = _fixture_by_name(name)
        a_dir, b_dir = a_root / name, b_root / name
        a_err = (a_dir / "_error.txt")
        b_err = (b_dir / "_error.txt")
        if a_err.is_file() or b_err.is_file():
            a_txt = a_err.read_text().strip() if a_err.is_file() else "ok"
            b_txt = b_err.read_text().strip() if b_err.is_file() else "ok"
            if a_txt != b_txt:
                any_diff = True
                print(f"{name}: run outcome differs — A: {a_txt!r}  B: {b_txt!r}")
            else:
                print(f"{name}: both runs failed identically ({a_txt!r}) — no output to diff")
            continue
        diffs = diff_output_trees(a_dir, b_dir, fx.scenario)
        # The six handoff mirrors are byte-compared directly (not under output_raw/out/).
        for mirror in MIRROR_FILES:
            ap, bp = a_dir / "solve_data" / mirror, b_dir / "solve_data" / mirror
            if ap.is_file() != bp.is_file():
                diffs.append(f"solve_data/{mirror}: present in only one of the two trees")
            elif ap.is_file() and ap.read_bytes() != bp.read_bytes():
                diffs.append(f"solve_data/{mirror}: differs")
        if diffs:
            any_diff = True
            print(f"{name}: {len(diffs)} differing path(s)")
            for d in diffs:
                print(f"    {d}")
        else:
            print(f"{name}: identical")
    return 1 if any_diff else 0


def cmd_list(_args: argparse.Namespace) -> int:
    for name in GATE15:
        print(name)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_snap = sub.add_parser("snapshot")
    p_snap.add_argument("out_dir")
    p_snap.add_argument("--fixtures", default="")
    p_snap.add_argument("--csv-dump", action="store_true")
    p_snap.add_argument("--dirty-with", default="")
    p_snap.add_argument("-j", "--jobs", type=int, default=4, dest="jobs")
    p_snap.set_defaults(func=cmd_snapshot)

    p_cmp = sub.add_parser("compare")
    p_cmp.add_argument("a")
    p_cmp.add_argument("b")
    p_cmp.add_argument("--fixtures", default="")
    p_cmp.set_defaults(func=cmd_compare)

    p_list = sub.add_parser("list")
    p_list.set_defaults(func=cmd_list)

    p_worker = sub.add_parser("_worker")
    p_worker.add_argument("--json", required=True)
    p_worker.add_argument("--scenario", required=True)
    p_worker.add_argument("--patch", default="none")
    p_worker.add_argument("--out", required=True)
    p_worker.add_argument("--csv-dump", action="store_true")
    p_worker.add_argument("--dirty-with", default="")
    p_worker.set_defaults(func=cmd_worker)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
