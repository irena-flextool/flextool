"""``flextool.engine._csv_reload.offline_outputs``: offline outputs from MPS + solution + dump (c24 stage 2, S2-4).

Relational test, no oracle: run a scenario's cascade in ``tmp_path``; snapshot the last sub-solve's provider into a
separate stamped dump (as ``--csv-dump`` does).  The dump's ``scale_the_objective.csv`` must equal the one the solve
wrote to ``work/solve_data/``.  Rebuild the last LP from the dump (``load_from_dump`` + ``build_flextool`` with that
scale), write it as a named MPS, solve it with the test ``highs.opt``, write the solution, then run
``write_outputs_from_files`` on those files and the dump, without passing the scale.  Every parquet it writes must
equal the production ``output_raw/`` parquet of that solve (rtol 1e-6), and it must write every variable parquet
production wrote for that solve.

Precondition (checked first): the rebuilt LP's objective equals the production objective.  A dump has no
solve-to-solve carry, so a scenario whose last solve depends on carry fails here with that message instead of with
many parquet differences.  Scenarios:

* ``coal`` — single solve, no investment.
* ``multi_year`` — four rolls; the carry is empty because nothing invests (the precondition pins that).
* ``wind_battery_invest`` — single solve with a nonzero investment (asserted), so the ``v_invest`` rename and the
  invest duals are compared on real values.
"""
from __future__ import annotations

import shutil
import subprocess
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path

import highspy
import pandas as pd
import pytest

from polar_high import Problem
from flextool.engine import build_flextool, run_chain_from_db
from flextool.engine._csv_reload import CsvReloadWarning, load_from_dump
from flextool.engine._csv_reload.load import load_from_dump_with_provider
from flextool.engine._csv_reload.offline_outputs import read_scale_the_objective, write_outputs_from_files
from flextool.engine.input._solve_state import FlexToolConfigError
from flextool.lean_parquet import read_lean_parquet
from flextool.process_outputs.read_highs_solution import VARIABLE_SPECS, _actual_solve_name

_SCENARIOS = ("coal", "multi_year", "wind_battery_invest")
_UNSUPPORTED = "CSV reload is unsupported"
# Output stems of ``write_all_variables``: one per VARIABLE_SPECS entry plus its custom writers.
_VARIABLE_WRITER_STEMS = frozenset(spec.output_name or spec.name for spec in VARIABLE_SPECS) | {
    "v_obj", "v_dual_invest_unit", "v_dual_invest_connection", "v_dual_invest_node", "v_dual_node_balance",
    "v_self_discharge", "v_dual_reserve__upDown__group__period__t", "v_invest_fan",
}


@dataclass(frozen=True)
class _Offline:
    """One scenario: production run (``work``), its dump, the rebuilt files and the offline outputs."""

    scenario: str
    db_url: str
    work: Path
    dump: Path
    solve_name: str
    roll: str
    last_obj: float
    scale: float
    mps: Path
    sol: Path
    out: Path
    paths: tuple[Path, ...]


def _rebuild_and_solve(dump: Path, db_url: str, scenario: str, scale: float, tmp: Path,
                       solver_config_dir: Path) -> tuple[Path, Path, float]:
    """Named MPS + style-0 solution of the last sub-solve rebuilt from the dump; return them and the objective."""
    with pytest.warns(CsvReloadWarning, match=_UNSUPPORTED):
        fd = load_from_dump(dump, db_url=db_url, scenario=scenario)
    pb = Problem()
    build_flextool(pb, fd, scale_the_objective=scale)
    mps = tmp / "model.mps"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pb.write_mps(mps, emit_names=True)
    h = highspy.Highs()
    h.setOptionValue("output_flag", False)
    assert h.readOptions(str(solver_config_dir / "highs.opt")) == highspy.HighsStatus.kOk
    h.setOptionValue("output_flag", False)
    assert h.readModel(str(mps)) == highspy.HighsStatus.kOk
    h.run()
    assert h.getModelStatus() == highspy.HighsModelStatus.kOptimal
    sol = tmp / "model.sol"
    assert h.writeSolution(str(sol), 0) == highspy.HighsStatus.kOk
    return mps, sol, h.getInfo().objective_function_value / scale


@pytest.fixture(scope="module")
def offline(test_db_url, tmp_path_factory, test_solver_config_dir):
    """Factory ``offline(scenario) -> _Offline`` (one cascade, rebuild and offline write per scenario per module)."""
    cache: dict[str, _Offline] = {}

    def get(scenario: str) -> _Offline:
        if scenario in cache:
            return cache[scenario]
        tmp = tmp_path_factory.mktemp(f"offline_{scenario}")
        work = tmp / f"work_{scenario}"
        steps = run_chain_from_db(
            input_db_url=test_db_url, scenario_name=scenario, work_folder=work,
            solver_config_dir=test_solver_config_dir, csv_dump=False, keep_solutions=True,
        )
        solve_name, last = list(steps.items())[-1]
        dump = tmp / "dump"
        last.flex_data_provider.snapshot_processed_inputs(dump)
        # The provider snapshot carries the scale the solve wrote (item 165: it used to be header-only).
        scale = read_scale_the_objective(dump)
        assert scale == read_scale_the_objective(work)
        mps, sol, obj = _rebuild_and_solve(dump, test_db_url, scenario, scale, tmp, test_solver_config_dir)
        assert obj == pytest.approx(last.obj, rel=1e-9), (
            f"{scenario}: the LP rebuilt from the dump has objective {obj!r}, production {last.obj!r}; "
            f"the dump has no solve-to-solve carry, so this scenario's last solve cannot be rebuilt from it"
        )
        out = tmp / "offline_output_raw"
        with pytest.warns(CsvReloadWarning, match=_UNSUPPORTED):
            paths = write_outputs_from_files(
                mps, sol, solve_name, out, dump_dir=dump, db_url=test_db_url, scenario=scenario,
            )
        # Item 163: ``_actual_solve_name`` is provider-only (no work-folder read); load the dump's
        # provider the same way ``write_outputs_from_files`` does internally.
        with pytest.warns(CsvReloadWarning, match=_UNSUPPORTED):
            _, dump_provider = load_from_dump_with_provider(dump, db_url=test_db_url, scenario=scenario)
        cache[scenario] = _Offline(
            scenario=scenario, db_url=test_db_url, work=work, dump=dump, solve_name=solve_name,
            roll=_actual_solve_name(solve_name, provider=dump_provider), last_obj=last.obj, scale=scale,
            mps=mps, sol=sol, out=out, paths=tuple(Path(x) for x in paths),
        )
        return cache[scenario]

    return get


@pytest.mark.parametrize("scenario", _SCENARIOS)
def test_offline_parquets_equal_production(offline, scenario):
    """Every offline parquet equals production's; every production variable parquet of the solve is written."""
    o = offline(scenario)
    prod = o.work / "output_raw"
    suffix = f"__{o.roll}.parquet"
    written = sorted(p.name for p in o.paths)
    assert written, f"{scenario}: no offline parquets written"
    assert written == sorted(p.name for p in o.out.glob("*.parquet"))
    assert all(name.endswith(suffix) for name in written), written
    missing_in_prod = [name for name in written if not (prod / name).exists()]
    assert not missing_in_prod, f"{scenario}: offline parquets production did not write: {missing_in_prod}"
    # Production's output_raw also holds the parameter / set parquets of other writers; compare the files of the
    # writers ``write_all_variables`` runs (its VARIABLE_SPECS and its custom writers).
    prod_vars = sorted(p.name for p in prod.glob(f"*{suffix}") if p.name[: -len(suffix)] in _VARIABLE_WRITER_STEMS)
    assert prod_vars == written, (
        f"{scenario}: production variable parquets of the solve not written offline: "
        f"{sorted(set(prod_vars) - set(written))}"
    )
    diffs = []
    for name in written:
        a, b = read_lean_parquet(o.out / name), read_lean_parquet(prod / name)
        try:
            pd.testing.assert_frame_equal(a, b, check_exact=False, rtol=1e-6, atol=1e-6, check_dtype=False)
        except AssertionError as exc:
            diffs.append(f"{name}: {str(exc).splitlines()[0]}")
    assert not diffs, f"{scenario}: {len(diffs)}/{len(written)} parquets differ:\n" + "\n".join(diffs)


@pytest.mark.parametrize("scenario", _SCENARIOS)
def test_v_obj_and_v_invest_are_restored(offline, scenario):
    """The two outputs the old offline path got wrong: ``v_obj`` (was 0) and ``v_invest`` (was empty)."""
    o = offline(scenario)
    v_obj = read_lean_parquet(o.out / f"v_obj__{o.roll}.parquet")
    assert float(v_obj["objective"].iloc[0]) == pytest.approx(o.last_obj, rel=1e-9)
    name = f"v_invest__{o.roll}.parquet"
    prod = o.work / "output_raw" / name
    assert prod.exists() == (o.out / name).exists()
    if prod.exists():
        assert read_lean_parquet(o.out / name).size == read_lean_parquet(prod).size


def test_invest_scenario_invests(offline):
    """``wind_battery_invest`` invests in production, so the parity above covers a nonzero ``v_invest``."""
    o = offline("wind_battery_invest")
    prod = read_lean_parquet(o.work / "output_raw" / f"v_invest__{o.roll}.parquet")
    assert float(prod.to_numpy().max()) > 0


def test_objective_scale_from_the_dump(offline, tmp_path):
    """Without ``scale_the_objective=`` the scale is read from the dump; missing or header-only → error.

    The provider snapshot holds the scale the solve wrote (one row); a dump whose file is header-only or missing is
    refused, never read as a silent 1e-6.
    """
    o = offline("coal")
    assert pd.read_csv(o.dump / "solve_data" / "scale_the_objective.csv").shape[0] == 1
    assert read_scale_the_objective(o.dump) == o.scale == read_scale_the_objective(o.work)
    dump = tmp_path / "dump"
    shutil.copytree(o.dump, dump)
    (dump / "solve_data" / "scale_the_objective.csv").write_text("key,value\n")
    with pytest.raises(FlexToolConfigError, match="no 'value' row.*pass scale_the_objective"):
        read_scale_the_objective(dump)
    with pytest.raises(FlexToolConfigError, match="pass scale_the_objective"):
        write_outputs_from_files(o.mps, o.sol, o.solve_name, tmp_path / "out", dump_dir=dump)
    assert not (tmp_path / "out").exists()
    empty = tmp_path / "empty"
    (empty / "solve_data").mkdir(parents=True)
    with pytest.raises(FlexToolConfigError, match="is missing.*pass scale_the_objective"):
        read_scale_the_objective(empty)


def test_cli_help_exits_zero():
    """``python -m flextool.engine._csv_reload.offline_outputs --help`` exits 0 and documents ``--dump``."""
    res = subprocess.run(
        [sys.executable, "-m", "flextool.engine._csv_reload.offline_outputs", "--help"],
        capture_output=True, text=True, timeout=120,
    )
    assert res.returncode == 0, res.stderr
    assert "--dump" in res.stdout and "--scale" in res.stdout
