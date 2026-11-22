"""The scaling range guard in real runs (``run_chain_from_db``), on synthetic patches of ``tests.json``.

Each case appends one alternative to scenario ``coal_co2_limit`` (one solve, ``y2020_2day_dispatch``,
period p2020, 48 hourly steps; ``tests/highs.opt`` sets the primal tolerance to 1e-8, so the window is
[1e-7, 1e10] and 17 decades fit):

* ``co2_max_period`` of group ``co2_limit`` set to a finite placeholder (1e18 …).  The smallest
  meaningful limit is ``maxFlow`` of coal_plant (existing 1 × availability 1 = 1).
* the rolling cases split the solve into two 24 h rolls and give coal_plant an availability of 1 in roll 1
  and ``a`` in roll 2, so roll 2's smallest limit is ``a``.

Hand-derived expectations (``--scaling basic``: no Layer 2, model values = what HiGHS gets):

* 1 … 1e18: 18.0 decades > 17 → error; fitting the large end, N = floor(log2(1e10/1e18)) = -27, allows a
  violation of 1e-8·2^27 = 1.34, 1.34x the smallest limit.
* rolling, co2 1e12, a = 1e-3: roll 1 N = -19 (smallest-side, raised to 100·tol); in roll 2,
  1e-3·2^-19 < 1e-7, so roll 2 gets N = -9 and warns (15 decades).
* rolling, co2 1e14, a = 1e-4: roll 2 spans 18 decades → error naming roll 2 (roll 1 solves).
"""
from __future__ import annotations

import base64
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

TEST_DIR = Path(__file__).resolve().parents[2]
if str(TEST_DIR) not in sys.path:
    sys.path.insert(0, str(TEST_DIR))

from db_utils import json_to_db  # noqa: E402

from flextool.engine import _behaviour_changes, run_chain_from_db  # noqa: E402
from flextool.engine.autoscale import _guard  # noqa: E402
from flextool.engine.input._solve_state import ScalingRangeError  # noqa: E402
from flextool.engine.solve import _orchestration  # noqa: E402
from flextool.update_flextool.db_migration import migrate_database  # noqa: E402

BASE = "coal_co2_limit"
SOLVE = "y2020_2day_dispatch"


def _pack(value) -> list:
    if isinstance(value, (int, float)):
        payload, kind = float(value), "float"
    elif isinstance(value, str):
        payload, kind = value, "str"
    else:
        payload, kind = value, "map"
    return [base64.b64encode(json.dumps(payload).encode()).decode(), kind]


def _map(index_name: str, items: dict) -> dict:
    return {"index_type": "str", "index_name": index_name, "data": [[k, v] for k, v in items.items()]}


def _hours(values: list[float]) -> dict:
    return _map("time", {f"t{i + 1:04d}": v for i, v in enumerate(values)})


def _scenario_db(root: Path, name: str, values: list) -> str:
    """``tests.json`` plus alternative ``name`` (``values``: ``[class, entity, parameter, value]``) on
    top of ``coal_co2_limit``, as scenario ``name``; built from JSON and migrated."""
    data = json.loads((TEST_DIR / "fixtures" / "tests.json").read_text())
    rows = [r for r in data["scenario_alternatives"] if r[0] == BASE]
    nxt = {a: b for _, a, b in rows}
    alt = next(a for a in nxt if a not in set(nxt.values()))
    chain = []
    while alt is not None:
        chain.append(alt)
        alt = nxt.get(alt)
    chain.append(name)
    data["alternatives"].append([name, ""])
    for cls, ent, param, val in values:
        data["parameter_values"].append([cls, ent, param, _pack(val), name])
    data["scenarios"].append([name, False, ""])
    for a, b in zip(chain, chain[1:] + [None]):
        data["scenario_alternatives"].append([name, a, b])
    src = root / f"{name}.json"
    src.write_text(json.dumps(data))
    url = json_to_db(src, root / f"{name}.sqlite")
    migrate_database(url)
    return url


def _co2(value: float) -> list:
    return ["group", "co2_limit", "co2_max_period", _map("period", {"p2020": value})]


def _rolling(roll2_availability: float) -> list:
    return [
        ["solve", SOLVE, "solve_mode", "rolling_window"],
        ["solve", SOLVE, "rolling_solve_horizon", 24.0],
        ["solve", SOLVE, "rolling_solve_jump", 24.0],
        ["unit", "coal_plant", "availability", _hours([1.0] * 24 + [roll2_availability] * 24)],
    ]


@pytest.fixture(scope="module")
def dbs(tmp_path_factory) -> dict[str, str]:
    root = tmp_path_factory.mktemp("scaling_guard_dbs")
    return {
        "placeholder_1e18": _scenario_db(root, "placeholder_1e18", [_co2(1e18)]),
        "gurobi_1e18": _scenario_db(root, "gurobi_1e18", [_co2(1e18), ["solve", SOLVE, "solver", "gurobi"]]),
        "placeholder_1e24": _scenario_db(root, "placeholder_1e24", [_co2(1e24)]),
        "roll_adjust": _scenario_db(root, "roll_adjust", [_co2(1e12), *_rolling(1e-3)]),
        "roll_error": _scenario_db(root, "roll_error", [_co2(1e14), *_rolling(1e-4)]),
        "roll_placeholder": _scenario_db(root, "roll_placeholder", [_co2(1e18), *_rolling(1.0)]),
    }


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    for key in ("FLEXTOOL_SCALING", "FLEXTOOL_SCALING_GUARD", "FLEXTOOL_SAVE_MEMORY",
                "FLEXTOOL_DISABLE_AUTOSCALE_CACHE"):
        monkeypatch.delenv(key, raising=False)
    _behaviour_changes.start_run(None)
    yield
    _behaviour_changes.start_run(None)


def _solver_cfg(tmp_path: Path) -> Path:
    cfg = tmp_path / "solver_config"
    cfg.mkdir(parents=True, exist_ok=True)
    shutil.copy(TEST_DIR / "highs.opt", cfg / "highs.opt")
    return cfg


def _run(url: str, scenario: str, tmp_path: Path, **kwargs):
    return run_chain_from_db(input_db_url=url, scenario_name=scenario, work_folder=tmp_path / "work",
                             solver_config_dir=_solver_cfg(tmp_path), **kwargs)


class _Count:
    """Wrap ``module.name`` and count its calls (the wrapped function still runs)."""

    def __init__(self, monkeypatch, module, name: str) -> None:
        self.calls: list = []
        real = getattr(module, name)

        def wrapped(*a, **k):
            self.calls.append((a, k))
            return real(*a, **k)

        monkeypatch.setattr(module, name, wrapped)


def _count_highs_runs(monkeypatch) -> list:
    """Count ``highspy.Highs.run`` calls (each HiGHS solve, warm or cold, in this process)."""
    import highspy

    calls: list = []
    real = highspy.Highs.run

    def run(self, *a, **k):
        calls.append(1)
        return real(self, *a, **k)

    monkeypatch.setattr(highspy.Highs, "run", run)
    return calls


def _yaml(tmp_path: Path) -> dict:
    import yaml

    return yaml.safe_load((tmp_path / "work" / "solve_data" / f"autoscale_{SOLVE}.yaml").read_text())


# --------------------------------------------------------------------- error / warn / off


def test_error_stops_before_any_solver_runs(dbs, tmp_path, monkeypatch):
    monkeypatch.setenv("FLEXTOOL_SCALING", "basic")
    runs = _count_highs_runs(monkeypatch)
    with pytest.raises(ScalingRangeError) as exc:
        _run(dbs["placeholder_1e18"], "placeholder_1e18", tmp_path)
    assert runs == []
    text = str(exc.value)
    for part in (f"Solve {SOLVE} cannot be scaled", "span 18.0 decades", "(1 … 1e+18)", "at most 17 fit",
                 "primal feasibility tolerance 1e-08, solver options",
                 "Largest:  1e+18 (model units) in co2_max_period (constraint right-hand side) at "
                 "g=co2_limit, d=p2020", 'a stand-in for "no limit"? Use inf',
                 "Smallest: 1 (model units) in maxFlow (constraint right-hand side)",
                 "p=coal_plant, source=coal_market, sink=west, d=p2020",
                 "the smallest limit could be violated by up to 1.34 (model units), 1.34x its size",
                 "--scaling-guard warn (FLEXTOOL_SCALING_GUARD=warn; in the GUI: Solver options, Scaling guard: Warn)"):
        assert part in text, part
    # The pre-solve report is written before the stop.
    tree = _yaml(tmp_path)
    assert tree["guard"]["verdict"] == "error"
    assert tree["guard"]["findings"]["bound_rhs"]["entries"]["max"][0]["family"] == "co2_max_period"


def test_guard_that_cannot_run_says_so_once(dbs, tmp_path, monkeypatch, coded_log):
    """A failing range readout (no verdict possible) is announced once and the run goes on (strict
    mode re-raises it instead, so it is off here)."""
    monkeypatch.delenv("FLEXTOOL_AUTOSCALE_STRICT", raising=False)
    monkeypatch.setenv("FLEXTOOL_SCALING", "basic")

    def broken(*a, **k):
        raise RuntimeError("readout failed")

    monkeypatch.setattr(_orchestration, "_autoscale_run_guard", broken)
    _run(dbs["roll_adjust"], "roll_adjust", tmp_path)
    assert coded_log().count("Warning: scaling guard could not run") == 1


def test_error_is_not_swallowed_when_layer3_apply_fails(dbs, tmp_path, monkeypatch):
    monkeypatch.setenv("FLEXTOOL_SCALING", "basic")

    def broken(*a, **k):
        raise RuntimeError("apply failed")

    monkeypatch.setattr(_orchestration, "_autoscale_apply_scaling", broken)
    with pytest.raises(ScalingRangeError, match="cannot be scaled"):
        _run(dbs["placeholder_1e18"], "placeholder_1e18", tmp_path)


def test_cli_prints_the_error_without_a_traceback(dbs, tmp_path):
    env = {**os.environ, "FLEXTOOL_SCALING": "basic", "FLEXTOOL_SKIP_SOLVER_PROBE": "1",
           "FLEXTOOL_THREADS": "1"}
    env.pop("FLEXTOOL_SCALING_GUARD", None)
    proc = subprocess.run(
        [sys.executable, "-m", "flextool.cli.cmd_run_flextool", dbs["placeholder_1e18"],
         "--scenario-name", "placeholder_1e18", "--work-folder", str(tmp_path / "work"),
         "--solver-config-dir", str(_solver_cfg(tmp_path))],
        env=env, capture_output=True, text=True, timeout=900,
    )
    out = proc.stdout + proc.stderr
    assert proc.returncode != 0
    assert f"Solve {SOLVE} cannot be scaled" in out
    assert "Traceback" not in out


def test_warning_prints_once_across_rolls_without_naming_walks_on_hits(
        dbs, tmp_path, monkeypatch, coded_log):
    """Full mode: each 24 h roll carries half of the 48 h period's CO2 cap, 5e17 (model units), next to
    maxFlow 1: 1 … 5e17 = log10(5e17) = 17.7 decades before Layer 2 (> 17, so --scaling basic would stop,
    and the warning says so).  Layer 2 brings the pool to 0.5 … 7.6e12 (13.2 decades): a warning, and both
    rolls solve.  The warm path's post-solve YAML keeps the Layer 2 plan."""
    located = _Count(monkeypatch, _guard, "locate_extremes")
    runs = _count_highs_runs(monkeypatch)
    steps = _run(dbs["roll_placeholder"], "roll_placeholder", tmp_path)
    assert steps and len(runs) == 2  # both rolls solved
    printed = coded_log()
    assert printed.count(f"Solve {SOLVE}: limits and bounds span") == 1
    assert "Layer 2 narrowed the span from 17.7 decades; under --scaling basic this would stop." in printed
    tree = _yaml(tmp_path)
    assert tree["layer2"] and tree["guard"]["verdict"] == "warn"
    # Named once, on the first roll (both ends); the cache hit of roll 2 does no naming walk.
    assert len(located.calls) == 2


def test_off_checks_nothing_but_layer3_still_clips(dbs, tmp_path, monkeypatch, coded_log):
    monkeypatch.setenv("FLEXTOOL_SCALING", "basic")
    monkeypatch.setenv("FLEXTOOL_SCALING_GUARD", "off")
    planned = _Count(monkeypatch, _orchestration, "_autoscale_emit_layer1")
    _run(dbs["roll_adjust"], "roll_adjust", tmp_path)
    assert "limits and bounds span" not in coded_log()
    exps = [k["layer3_plan"].user_bound_scale for _a, k in planned.calls]
    assert exps == [-19, -9]
    assert _yaml(tmp_path)["guard"]["verdict"] == "off"


# --------------------------------------------------------------------- per-roll recheck


@pytest.mark.parametrize("warm", [True, False], ids=["warm_first_build", "cold_save_memory"])
def test_roll_recheck_moves_roll_2_only(dbs, tmp_path, monkeypatch, coded_log, warm):
    monkeypatch.setenv("FLEXTOOL_SCALING", "basic")
    planned = _Count(monkeypatch, _orchestration, "_autoscale_emit_layer1")
    _run(dbs["roll_adjust"], "roll_adjust", tmp_path, warm=warm)
    assert [k["layer3_plan"].user_bound_scale for _a, k in planned.calls] == [-19, -9]
    printed = coded_log()
    assert f"Solve {SOLVE}, roll 2: limits and bounds span 15.0 decades" in printed
    assert "This roll uses user_bound_scale -9 instead of -19" in printed
    guard = _yaml(tmp_path)["guard"]
    assert guard["roll_user_bound_scale"] == -9 and guard["verdict"] == "warn"


def test_roll_recheck_on_the_save_memory_path(dbs, tmp_path, monkeypatch):
    """--save-memory: the recheck runs in write_mps and roll 2's exponent reaches the child's options."""
    from flextool.engine.solve import _subprocess_solve

    monkeypatch.setenv("FLEXTOOL_SCALING", "basic")
    monkeypatch.setenv("FLEXTOOL_SAVE_MEMORY", "1")
    written = _Count(monkeypatch, _subprocess_solve, "_write_highs_option_files")
    _run(dbs["roll_adjust"], "roll_adjust", tmp_path, warm=False)
    exps = [dict(a[0]).get("user_bound_scale") for a, _k in written.calls]
    assert exps == ["-19", "-9"]


@pytest.mark.parametrize("warm", [True, False], ids=["warm_first_build", "cold_save_memory"])
def test_later_roll_error_names_the_roll(dbs, tmp_path, monkeypatch, warm):
    monkeypatch.setenv("FLEXTOOL_SCALING", "basic")
    from flextool.engine.solve import _subprocess_solve

    runs = _count_highs_runs(monkeypatch)
    children = _Count(monkeypatch, _subprocess_solve, "_write_highs_option_files")
    with pytest.raises(ScalingRangeError) as exc:
        _run(dbs["roll_error"], "roll_error", tmp_path, warm=warm)
    # Roll 1 solved; roll 2 stopped before HiGHS ran (warm: in this process; warm=False routes HiGHS
    # through the --save-memory child, which roll 2 never got: the check ran inside write_mps).
    assert (len(runs), len(children.calls)) == ((1, 0) if warm else (0, 1))
    assert f"Solve {SOLVE}, roll 2 cannot be scaled" in str(exc.value)
    assert "span 18.0 decades" in str(exc.value)
    # The YAML holds roll 2's own verdict (written before the roll it was roll 1's warning: 1 … 1e14).
    guard = _yaml(tmp_path)["guard"]
    assert guard["verdict"] == "error"
    assert guard["findings"]["bound_rhs"]["numbers"]["b_lo"] == pytest.approx(1e-4)


# --------------------------------------------------------------------- strict mode


def test_strict_raises_for_a_layer2_caused_error_only(dbs, tmp_path, monkeypatch, coded_log):
    """1e24 is beyond HiGHS' infinite bound (no limit) until Layer 2 scales it into range (1.5e19):
    Layer 2 caused that error.  1e18 under --scaling basic is the model's own: strict leaves it to warn."""
    monkeypatch.setenv("FLEXTOOL_SCALING_GUARD", "warn")
    monkeypatch.setenv("FLEXTOOL_AUTOSCALE_STRICT", "1")
    with pytest.raises(ScalingRangeError, match="FLEXTOOL_AUTOSCALE_STRICT=1"):
        _run(dbs["placeholder_1e24"], "placeholder_1e24", tmp_path / "full")
    from flextool.engine.input._solve_state import FlexToolSolveError

    monkeypatch.setenv("FLEXTOOL_SCALING", "basic")
    try:
        _run(dbs["placeholder_1e18"], "placeholder_1e18", tmp_path / "basic")
    except FlexToolSolveError:
        pass  # not stopped by the guard; the solver may fail on this LP
    assert f"Solve {SOLVE} cannot be scaled" in coded_log()


def test_layer2_widening_is_named_in_the_warning(dbs, tmp_path, monkeypatch, coded_log):
    """Full mode, 1e24: before Layer 2 the value is a solver infinity; the message says Layer 2 made it a
    limit and how to avoid it.  Strict mode would raise for this Layer-2-caused error, so it is off."""
    from flextool.engine.input._solve_state import FlexToolSolveError

    monkeypatch.delenv("FLEXTOOL_AUTOSCALE_STRICT", raising=False)
    monkeypatch.setenv("FLEXTOOL_SCALING_GUARD", "warn")
    try:
        _run(dbs["placeholder_1e24"], "placeholder_1e24", tmp_path)
    except FlexToolSolveError:
        pass  # solved anyway, as asked: the solver may well fail on this LP; the warning came first
    printed = coded_log()
    assert "Layer 2 widened the span from" in printed and "--scaling basic avoids that" in printed
    assert _yaml(tmp_path)["guard"]["findings"]["bound_rhs"]["caused_by_layer2"] is True


# --------------------------------------------------------------------- commercial solver


def test_commercial_solver_only_warns(dbs, tmp_path, monkeypatch, coded_log):
    """Gurobi gets no FlexTool scaling: the 1e18 placeholder is judged at N = 0 and Gurobi's default
    feasibility tolerance 1e-6 (window [1e-5, 1e12]); it would stop a HiGHS solve, here it warns and the
    run reaches the solver call (stopped there by the test: no licence needed)."""
    from flextool.engine.solve import _solver_dispatch

    class _Reached(Exception):
        pass

    def stop(*a, **k):
        raise _Reached

    monkeypatch.setenv("FLEXTOOL_SCALING", "basic")
    monkeypatch.setattr(_solver_dispatch, "run_one_solve", stop)
    with pytest.raises(_Reached):
        _run(dbs["gurobi_1e18"], "gurobi_1e18", tmp_path, warm=False)
    printed = coded_log()
    assert f"Solve {SOLVE} (gurobi) cannot be scaled" in printed
    assert "gurobi gets no FlexTool scaling, so this is a warning only" in printed
    guard = _yaml(tmp_path)["guard"]
    assert guard["solver"] == "gurobi" and guard["verdict"] == "warn"
    assert guard["user_bound_scale"] == 0 and guard["window"]["hi"] == pytest.approx(1e12)


# --------------------------------------------------------------------- h2_trade_parity


def test_h2_trade_parity_warns_that_layer2_widened_the_span(h2_trade_parity_db_url, tmp_path, coded_log):
    """The one fixture of the decision snapshot that warns: Layer 2 widens its limits from 10.8 to 13.8
    decades (2.14e-5 … 1.36e9 after it).  Layer 3 (tol 1e-8, smallest-side tier) aims the small end at
    100·tol: N = ceil(log2(1e-6 / 2.14e-5)) = -4 (the clamp-large tier value
    floor(log2(1e6 / 1.36e9)) = -11 is raised to it); the large end fits (N_max = floor(log2(1e10 /
    1.36e9)) = 2).  The verdict reports the hard window: n_min = ceil(log2(1e-7 / 2.14e-5)) = -7."""
    work = tmp_path / "work"
    run_chain_from_db(input_db_url=h2_trade_parity_db_url, scenario_name="scenario_test_6h_no_carrier_storage",
                      work_folder=work, solver_config_dir=_solver_cfg(tmp_path))
    printed = coded_log()
    assert "limits and bounds span 13.8 decades" in printed
    assert "Layer 2 widened the span from 10.8 decades: --scaling basic avoids that." in printed
    import yaml

    tree = yaml.safe_load((work / "solve_data" / "autoscale_solve_test_6h.yaml").read_text())
    assert tree["layer3"]["user_bound_scale"] == -4
    assert tree["layer3"]["bound_tag"] == "smallest-side+raised"
    b = tree["guard"]["findings"]["bound_rhs"]
    assert tree["guard"]["verdict"] == "warn" and b["caused_by_layer2"] is True
    assert b["numbers"]["n_min"] == -7 and b["numbers"]["n_max"] == 2
