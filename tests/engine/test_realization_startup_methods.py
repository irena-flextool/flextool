"""What a single stochastic solve's realization keeps of the units online, for each ``startup_method`` with online
variables.

``linear_fix_in_uc`` / ``integer_fix_in_uc``: the realization solve pins the units online over the realized periods
to the stochastic solve's reference forecast (``online_fix_eq_linear`` / ``online_fix_eq_integer``).  ``linear`` /
``integer``: no pin, the realization decides them again with the realized data.

Toy: ``stoch_t0_commit.json`` (built from JSON under the pytest tmp dir, CLAUDE.md invariant 3), one single solve
``stoch_rd`` over ``p2035`` (3 steps, each weighted 8760 / 3 = 2920 h), fan at t0001: forecasts ``high`` (wind CF 0.8)
and ``low`` (CF 0.0), w 0.5 each, and ``rlz`` the ``realized: yes`` branch.  Demand 100 MW, penalties 10 000 €/MWh.
``peaker``: 2 existing units of 50 MW, 10 €/MWh, start-up cost 1, min up / down 1 h.  Patched here: peaker
``min_load`` 1.0 (a unit online runs at full output, so output = 50 · v_online and the units online have one best
value) and ``rlz`` CF 0.8 (the realized wind can cover 80 MW).

Hand derivation (oracle; not read from a run):
* Stochastic solve: in ``low`` the peaker must make 100 MW, so v_online = 2.  Where the units online are tied across
  the forecasts over the committed window (``within_solve``, and a ``_fix_in_uc`` peaker under
  ``separate_dispatch``), ``high`` (the reference forecast, first in sorted order) also has 2 at t0001-t0003 and
  curtails its wind to 0.  Solve (1) of ``separate_dispatch`` does not tie a plain ``linear`` / ``integer`` peaker
  (its realization decides it again): ``high`` (wind CF 0.8, the same as ``rlz``) then takes the re-decided values
  below, v_online 0.4 (linear) / 1 (integer).
* Realization pinned (``_fix_in_uc``): v_online = 2 at every step, peaker 100 MW, wind 0: objective
  100 · 10 · 2920 · 3 = 8 760 000.
* Realization ``linear``: peaker 50 · v, wind 100 − 50 · v ≤ 80, so v ≥ 0.4; the cost 10 · 50 · v rises with v, so
  v = 0.4 (20 MW), wind 80: objective 20 · 10 · 2920 · 3 = 1 752 000.  No start-up: t0001 is a free start and v is
  constant.
* Realization ``integer``: v ∈ {0, 1, 2}; v = 0 leaves 20 MW unserved at 10 000 €/MWh; v = 1 gives 50 MW, wind 50:
  objective 50 · 10 · 2920 · 3 = 4 380 000.
So the re-decided objective is below the pinned one for both classes.  Both ``separate_dispatch`` (a rolling solve
with one roll) and ``within_solve`` (a single solve keeps its separate realization) give these values.

Connection variant (the pin of a connection, ``separate_dispatch``): ``peaker`` is replaced by the source unit
``gen`` (1000 MW, 10 €/MWh) on node ``ext`` and the ``unidirectional`` connection ``link`` ext -> elec (2 existing
units of 50 MW, lossless, no cost of its own; one direction, so no flow can circulate).  Connections have no min
load, so the user constraint ``c_full`` (50 · v_online(link) − flow(link) ≤ 0) does the work of ``min_load`` 1.0:
with flow ≤ 50 · v_online the link's flow is exactly 50 · v_online.  Every number above then holds with ``link`` for ``peaker``: the parent's ``low`` needs 100 MW over
the link (v_online 2, tied to ``high``; an untied ``integer`` link: ``high`` has v_online 1); ``integer_fix_in_uc``
pins v_online 2, objective 8 760 000; ``integer`` re-decides v_online 1, objective 4 380 000.
"""
from __future__ import annotations

import base64
import json
import os
import sys
from pathlib import Path

import polars as pl
import pytest

from tests.engine._lp_rows import row_terms
from tests.engine._unit_db import build, utf8

pytestmark = pytest.mark.solver

FLEXTOOL_ROOT = Path(__file__).resolve().parents[2]
FIXTURE = FLEXTOOL_ROOT / "tests" / "fixtures" / "stoch_t0_commit.json"
SCENARIO = "stoch_t0_commit"
PARENT = "stoch_rd"
REALIZATION = "stoch_rd__realized_dispatch"
STEPS = ("t0001", "t0002", "t0003")

PINNED_ONLINE = 2.0
PINNED_OBJECTIVE = 8_760_000.0
REDECIDED = {"linear": (0.4, 1_752_000.0), "integer": (1.0, 4_380_000.0)}


def _enc(value) -> list:
    return [base64.b64encode(json.dumps(value).encode()).decode(), "str" if isinstance(value, str) else "float"]


def _variant(startup: str, method: str):
    def patch(d: dict) -> None:
        for r in d["parameter_values"]:
            if r[1] == "peaker" and r[2] == "startup_method":
                r[3] = _enc(startup)
            elif r[1] == "peaker" and r[2] == "min_load":
                r[3] = _enc(1.0)
            elif r[2] == "dispatch_realization_method":
                r[3] = _enc(method)
            elif r[1] == "wprof" and r[2] == "profile":
                m = json.loads(base64.b64decode(r[3][0]).decode())
                for branch in m["data"]:
                    if branch[0] == "rlz":
                        for cell in branch[1]["data"][0][1]["data"]:
                            cell[1] = 0.8
                r[3] = [base64.b64encode(json.dumps(m).encode()).decode(), r[3][1]]
    return patch


def _connection_variant(startup: str):
    """``_variant`` with ``peaker`` replaced by ``gen`` -> ``ext`` -> ``link`` -> ``elec`` (module docstring)."""
    def patch(d: dict) -> None:
        _variant("linear", "separate_dispatch")(d)

        def keep(name) -> bool:
            return name != "peaker" and not (isinstance(name, list) and "peaker" in name)
        d["entities"] = [e for e in d["entities"] if keep(e[1])]
        d["entity_alternatives"] = [e for e in d["entity_alternatives"] if keep(e[1][0])]
        d["parameter_values"] = [r for r in d["parameter_values"] if keep(r[1])]
        d["entities"] += [["node", "ext", None], ["unit", "gen", None], ["unit__outputNode", ["gen", "ext"], None],
                          ["connection", "link", None], ["connection__node__node", ["link", "ext", "elec"], None],
                          ["connection__node", ["link", "elec"], None], ["constraint", "c_full", None]]
        d["entity_alternatives"] += [["node", ["ext"], "init", True], ["unit", ["gen"], "init", True],
                                     ["connection", ["link"], "init", True]]
        cmap = {"type": "map", "index_type": "str", "index_name": "constraint", "data": [["c_full", None]]}
        d["parameter_values"] += [
            ["node", "ext", "node_type", _enc("balance"), "base"],
            ["unit", "gen", "efficiency", _enc(1.0), "base"],
            ["unit", "gen", "existing", _enc(1000.0), "base"],
            ["unit__outputNode", ["gen", "ext"], "other_operational_cost", _enc(10.0), "base"],
            ["connection", "link", "transfer_method", _enc("unidirectional"), "base"],
            ["connection", "link", "existing", _enc(2.0), "base"],
            ["connection", "link", "virtual_unitsize", _enc(50.0), "base"],
            ["connection", "link", "startup_method", _enc(startup), "base"],
            ["connection", "link", "constraint_online_coeff", _map(cmap, 50.0), "base"],
            ["connection__node", ["link", "elec"], "constraint_flow_coeff", _map(cmap, -1.0), "base"],
            ["constraint", "c_full", "sense", _enc("less_than"), "base"],
            ["constraint", "c_full", "constant", _enc(0.0), "base"]]
    return patch


def _map(template: dict, coeff: float) -> list:
    value = {**template, "data": [[template["data"][0][0], coeff]]}
    return [base64.b64encode(json.dumps(value).encode()).decode(), "map"]


def _run(tmp_path: Path, patch) -> tuple[dict, Path]:
    from flextool.engine import run_chain_from_db
    from flextool.update_flextool.db_migration import migrate_database

    tests_dir = FLEXTOOL_ROOT / "tests"
    if str(tests_dir) not in sys.path:
        sys.path.insert(0, str(tests_dir))
    from db_utils import json_to_db  # noqa: E402

    data = json.loads(FIXTURE.read_text(encoding="utf-8"))
    patch(data)
    fixture = tmp_path / "fixture.json"
    fixture.write_text(json.dumps(data), encoding="utf-8")
    url = json_to_db(fixture, tmp_path / "db.sqlite")
    migrate_database(url)
    work = tmp_path / "work"
    work.mkdir()
    prev = os.environ.get("FLEXTOOL_AUTOSCALE_STRICT")
    os.environ["FLEXTOOL_AUTOSCALE_STRICT"] = "1"
    try:
        steps = run_chain_from_db(url, SCENARIO, work_folder=work, warm=False, keep_solutions=True)
    finally:
        if prev is None:
            os.environ.pop("FLEXTOOL_AUTOSCALE_STRICT", None)
        else:
            os.environ["FLEXTOOL_AUTOSCALE_STRICT"] = prev
    return steps, work


def _objective(work: Path, solve: str) -> float:
    return float(pl.read_parquet(work / "output_raw" / f"v_obj__{solve}.parquet")["objective"][0])


def _online(step, kind: str, d: str, process: str = "peaker") -> list[float]:
    v = utf8(step.solution.value(f"v_online_{kind}")).filter((pl.col("p") == process) & (pl.col("d") == d))
    return [float(x) for x in v.sort("t")["value"]]


@pytest.mark.parametrize("method", ["separate_dispatch", "within_solve"])
@pytest.mark.parametrize("startup", ["linear", "linear_fix_in_uc", "integer", "integer_fix_in_uc"])
def test_single_solve_realization_by_startup_method(tmp_path_factory, startup, method) -> None:
    """``_fix_in_uc``: the realization's units online equal the reference forecast's over the realized period
    (one ``online_fix_eq_*`` row per step).  Plain: no pin row, the units online and the objective are the
    re-decided optimum of the module docstring, below the pinned objective.  The stochastic solve's ``high`` is tied
    to ``low``'s 2 units except for a plain peaker under ``separate_dispatch`` (module docstring)."""
    kind = "linear" if startup.startswith("linear") else "integer"
    steps, work = _run(tmp_path_factory.mktemp(f"{startup}_{method}"), _variant(startup, method))
    assert all(s.solution.optimal for s in steps.values())
    parent, real = steps[PARENT], steps[REALIZATION]
    tied = method == "within_solve" or startup.endswith("_fix_in_uc")
    high = PINNED_ONLINE if tied else REDECIDED[kind][0]
    assert _online(parent, kind, "p2035_low") == pytest.approx([PINNED_ONLINE] * 3, abs=1e-9)
    assert _online(parent, kind, "p2035_high") == pytest.approx([high] * 3, abs=1e-6)
    rows = row_terms(build(real.flex_data), f"online_fix_eq_{kind}")
    if startup.endswith("_fix_in_uc"):
        assert sorted(rows) == [f"online_fix_eq_{kind}[peaker,p2035,{t}]" for t in STEPS]
        assert _online(real, kind, "p2035") == pytest.approx(_online(parent, kind, "p2035_high"), abs=1e-9)
        assert _objective(work, REALIZATION) == pytest.approx(PINNED_OBJECTIVE, rel=1e-6)
    else:
        online, objective = REDECIDED[kind]
        assert not rows and real.flex_data.p_fix_online is None
        assert _online(real, kind, "p2035") == pytest.approx([online] * 3, abs=1e-6)
        assert _objective(work, REALIZATION) == pytest.approx(objective, rel=1e-6)
        assert objective < PINNED_OBJECTIVE


@pytest.mark.parametrize("startup", ["integer", "integer_fix_in_uc"])
def test_connection_realization_by_startup_method(tmp_path_factory, startup) -> None:
    """A connection's units online: ``integer_fix_in_uc`` pins them in the realization to the reference
    forecast's (one ``online_fix_eq_integer`` row per step); ``integer`` decides them again (module docstring),
    and is not tied in the stochastic solve (``high`` 1 unit, ``low`` 2)."""
    steps, work = _run(tmp_path_factory.mktemp(f"link_{startup}"), _connection_variant(startup))
    assert all(s.solution.optimal for s in steps.values())
    parent, real = steps[PARENT], steps[REALIZATION]
    high = PINNED_ONLINE if startup == "integer_fix_in_uc" else REDECIDED["integer"][0]
    assert _online(parent, "integer", "p2035_low", "link") == pytest.approx([PINNED_ONLINE] * 3, abs=1e-9)
    assert _online(parent, "integer", "p2035_high", "link") == pytest.approx([high] * 3, abs=1e-6)
    rows = row_terms(build(real.flex_data), "online_fix_eq_integer")
    if startup == "integer_fix_in_uc":
        assert sorted(rows) == [f"online_fix_eq_integer[link,p2035,{t}]" for t in STEPS]
        assert _online(real, "integer", "p2035", "link") == pytest.approx([PINNED_ONLINE] * 3, abs=1e-9)
        assert _objective(work, REALIZATION) == pytest.approx(PINNED_OBJECTIVE, rel=1e-6)
    else:
        online, objective = REDECIDED["integer"]
        assert not rows and real.flex_data.p_fix_online is None
        assert _online(real, "integer", "p2035", "link") == pytest.approx([online] * 3, abs=1e-6)
        assert _objective(work, REALIZATION) == pytest.approx(objective, rel=1e-6)
