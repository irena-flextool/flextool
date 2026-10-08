"""Regression: the loose fuel-arc bound of a multi-output indirect unit must
never be the binding limit on the unit's operation.

For an indirect (``nvar``) unit the capacity is enforced per OUTPUT arc
(``maxToSink``: ``out_k ≤ capacity · capacity_max_coeff_k``).  The input
(fuel) arcs carry only a structural "loose" bound from ``p_flow_upper``
that exists to give the solver a bounded region.  The fuel is tied to the
outputs by ``conversion_indirect``::

    Σ_s src_conv_s · in_s = slope · Σ_k sink_conv_k · out_k

so the largest fuel any single input arc can ever be asked for is::

    in_s ≤ capacity · slope · Σ_k (sink_conv_k · capacity_max_coeff_k)
                                                          / src_conv_s

The bound used to be ``slope · capacity`` (enough fuel for ONE output at
full capacity).  With several outputs — or with a ``conversion_flow_coeff``
above 1 on an output — the loose bound became the binding constraint and
silently capped the unit below its declared capacity.

The fixture uses the ``coal_chp`` unit from ``tests.json`` (existing
1000 MW, efficiency 0.9, outputs ``west`` + ``heat``, input
``coal_market``), switches its ratio-fixing user constraint off, sets
coefficients for which Σ conv·capcoef / src_conv = (2.0 + 0.5) / 0.8 =
3.125, and sets both demands to 1000 MW so the optimum runs BOTH outputs
at full capacity in every timestep.  Before the fix the fuel arc capped
the unit (≤ 1000/0.9 MW of fuel instead of 3125/0.9) → unserved demand.
"""
from __future__ import annotations

import base64
import json
import sys
from pathlib import Path

import polars as pl
import pytest

FLEXTOOL_ROOT = Path(__file__).resolve().parents[2]
TESTS_DIR = FLEXTOOL_ROOT / "tests"
BASE_FIXTURE_JSON = TESTS_DIR / "fixtures" / "tests.json"

UNIT = "coal_chp"
FUEL_NODE = "coal_market"
CAPACITY = 1000.0
EFFICIENCY = 0.9
SINK_CONV = {"west": 2.0, "heat": 0.5}
SRC_CONV = 0.8
# Σ_k sink_conv_k · capacity_max_coeff_k / src_conv  (capacity coefs = 1)
FUEL_FACTOR = sum(SINK_CONV.values()) / SRC_CONV

_ALT = "indirect_fuel_bound"
_SCENARIO = "indirect_fuel_bound_test"
_CHAIN = ["init", "west", "coal_chp", "heat", _ALT]


def _b64(x, kind: str) -> list:
    """Spine import-format packed scalar value."""
    return [base64.b64encode(json.dumps(x).encode()).decode(), kind]


@pytest.fixture(scope="module")
def fuel_bound_db(tmp_path_factory: pytest.TempPathFactory) -> str:
    """``tests.json`` + an override alternative making the coal_chp unit
    a 2-output indirect unit that must run both outputs at full capacity.
    """
    if str(TESTS_DIR) not in sys.path:
        sys.path.insert(0, str(TESTS_DIR))
    from db_utils import json_to_db  # noqa: E402
    from flextool.update_flextool.db_migration import migrate_database

    data = json.loads(BASE_FIXTURE_JSON.read_text())
    data["alternatives"].append([_ALT, ""])
    pv = data["parameter_values"]
    # Free the two outputs from the ratio-fixing user constraint.
    pv.append(["constraint", "coal_chp_fix", "is_enabled",
               _b64("no", "str"), _ALT])
    # Both demands equal the unit capacity (constant over the horizon).
    pv.append(["node", "west", "inflow", _b64(-CAPACITY, "float"), _ALT])
    pv.append(["node", "heat", "inflow", _b64(-CAPACITY, "float"), _ALT])
    for sink, conv in SINK_CONV.items():
        pv.append(["unit__outputNode", [UNIT, sink], "conversion_flow_coeff",
                   _b64(conv, "float"), _ALT])
    pv.append(["unit__inputNode", [UNIT, FUEL_NODE], "conversion_flow_coeff",
               _b64(SRC_CONV, "float"), _ALT])
    data["scenarios"].append([_SCENARIO, False, ""])
    for alt, before in zip(_CHAIN, _CHAIN[1:] + [None]):
        data["scenario_alternatives"].append([_SCENARIO, alt, before])

    work = tmp_path_factory.mktemp("indirect_fuel_bound_db")
    json_path = work / "fuel_bound.json"
    json_path.write_text(json.dumps(data))
    url = json_to_db(json_path, work / "fuel_bound.sqlite")
    migrate_database(url)
    return url


@pytest.fixture(scope="module")
def solved_step(fuel_bound_db: str, tmp_path_factory: pytest.TempPathFactory):
    from flextool.engine_polars import run_chain_from_db

    work_folder = tmp_path_factory.mktemp("indirect_fuel_bound_work")
    steps = run_chain_from_db(
        fuel_bound_db, _SCENARIO, work_folder=work_folder,
        keep_solutions=True,
    )
    assert steps, "run_chain_from_db produced no orchestration steps"
    step = list(steps.values())[-1]
    assert step.solution is not None, "solve produced no solution"
    return step


def _unitsize(fd) -> float:
    return float(fd.p_unitsize.frame
                 .filter(pl.col("p") == UNIT)["value"][0])


def test_unit_is_indirect(solved_step) -> None:
    """Guard: the scenario really exercises the indirect (nvar) path."""
    fd = solved_step.flex_data
    assert fd.process_indirect is not None
    assert UNIT in fd.process_indirect["p"].cast(pl.Utf8).to_list()


def test_fuel_arc_bound_covers_conversion_equation(solved_step) -> None:
    """``p_flow_upper`` on the fuel arc equals the conversion-implied
    maximum ``slope · cap · Σ(sink_conv·capcoef) / src_conv`` (per unit
    of unitsize) — not ``slope · cap`` and not ``slope · cap · n_outputs``.
    """
    fd = solved_step.flex_data
    us = _unitsize(fd)
    rows = (fd.p_flow_upper.frame
            .with_columns(pl.col(c).cast(pl.Utf8)
                          for c in ("p", "source", "sink"))
            .filter((pl.col("p") == UNIT) & (pl.col("source") == FUEL_NODE)
                    & (pl.col("sink") == UNIT)))
    assert rows.height > 0, "no p_flow_upper rows for the fuel arc"
    expected = CAPACITY / us / EFFICIENCY * FUEL_FACTOR
    assert rows["value"].to_list() == pytest.approx(
        [expected] * rows.height, rel=1e-9)


def test_both_outputs_reach_full_capacity(solved_step) -> None:
    """With demand = capacity on both outputs, the optimum runs both at
    full capacity in every timestep — the fuel arc must not bind."""
    fd = solved_step.flex_data
    us = _unitsize(fd)
    flow = (solved_step.solution.value("v_flow")
            .with_columns(pl.col(c).cast(pl.Utf8)
                          for c in ("p", "source", "sink")))
    for sink in SINK_CONV:
        out = flow.filter((pl.col("p") == UNIT) & (pl.col("source") == UNIT)
                          & (pl.col("sink") == sink))
        assert out.height > 0, f"no v_flow rows for {UNIT}->{sink}"
        assert (out["value"] * us).to_list() == pytest.approx(
            [CAPACITY] * out.height, rel=1e-6), (
            f"{UNIT}->{sink} did not reach its capacity: the fuel arc "
            f"bound is constraining the unit")
    fuel = flow.filter((pl.col("p") == UNIT) & (pl.col("source") == FUEL_NODE)
                       & (pl.col("sink") == UNIT))
    expected_fuel = CAPACITY / EFFICIENCY * FUEL_FACTOR
    assert (fuel["value"] * us).to_list() == pytest.approx(
        [expected_fuel] * fuel.height, rel=1e-6)


# ── Uncapped output → unbounded fuel need ───────────────────────────────
_ALT_UNCAPPED = "indirect_fuel_uncapped"
_SCENARIO_UNCAPPED = "indirect_fuel_uncapped_test"


def test_uncapped_output_leaves_fuel_arc_unconstrained(
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    """An output with ``capacity_max_coeff = 0`` has no cap (coeff_zero),
    so the fuel it can demand is unbounded — the fuel arc must take the
    unconstrained value rather than a capacity-derived one."""
    if str(TESTS_DIR) not in sys.path:
        sys.path.insert(0, str(TESTS_DIR))
    from db_utils import json_to_db  # noqa: E402
    from flextool.engine_polars import run_chain_from_db
    from flextool.update_flextool.db_migration import migrate_database

    data = json.loads(BASE_FIXTURE_JSON.read_text())
    data["alternatives"].append([_ALT_UNCAPPED, ""])
    pv = data["parameter_values"]
    pv.append(["constraint", "coal_chp_fix", "is_enabled",
               _b64("no", "str"), _ALT_UNCAPPED])
    pv.append(["unit__outputNode", [UNIT, "heat"], "capacity_max_coeff",
               _b64(0.0, "float"), _ALT_UNCAPPED])
    chain = ["init", "west", "coal_chp", "heat", _ALT_UNCAPPED]
    data["scenarios"].append([_SCENARIO_UNCAPPED, False, ""])
    for alt, before in zip(chain, chain[1:] + [None]):
        data["scenario_alternatives"].append(
            [_SCENARIO_UNCAPPED, alt, before])
    work = tmp_path_factory.mktemp("indirect_fuel_uncapped")
    json_path = work / "uncapped.json"
    json_path.write_text(json.dumps(data))
    url = json_to_db(json_path, work / "uncapped.sqlite")
    migrate_database(url)

    steps = run_chain_from_db(url, _SCENARIO_UNCAPPED,
                              work_folder=work, keep_solutions=True)
    fd = list(steps.values())[-1].flex_data
    rows = (fd.p_flow_upper.frame
            .with_columns(pl.col(c).cast(pl.Utf8)
                          for c in ("p", "source", "sink"))
            .filter((pl.col("p") == UNIT) & (pl.col("source") == FUEL_NODE)
                    & (pl.col("sink") == UNIT)))
    assert rows.height > 0
    assert rows["value"].min() >= 1_000_000.0


# ── Delay: one input step can feed a heavier-weighted output step ───────
class _StubSource:
    """Minimal InputSource exposing only ``unit.delay``."""

    def __init__(self, delay: pl.DataFrame) -> None:
        self._delay = delay

    def parameter(self, entity_class: str, parameter_name: str):
        if (entity_class, parameter_name) == ("unit", "delay"):
            return self._delay
        raise KeyError((entity_class, parameter_name))


def test_delay_fuel_factor() -> None:
    """``delay_factor = 1 / max(weight)``: a 0.25/0.75 split lets one input
    step carry 1/0.75 of the steady-state fuel; a scalar delay (single
    duration, weight 1) leaves the bound unchanged."""
    from flextool.engine_polars._derived_params import _delay_fuel_factor_lf

    weighted = _StubSource(pl.DataFrame({
        "name": ["u_split", "u_split"],
        "delay_duration": ["1.0", "2.0"],
        "value": [0.25, 0.75],
    }))
    got = dict(_delay_fuel_factor_lf(weighted)
               .with_columns(pl.col("p").cast(pl.Utf8)).collect().iter_rows())
    assert got == pytest.approx({"u_split": 1.0 / 0.75})

    single = _StubSource(pl.DataFrame({"name": ["u_single"], "value": [3.0]}))
    got = dict(_delay_fuel_factor_lf(single)
               .with_columns(pl.col("p").cast(pl.Utf8)).collect().iter_rows())
    assert got == pytest.approx({"u_single": 1.0})


# ── Indirect output arcs honour capacity_max_coeff ──────────────────────
def test_indirect_output_capacity_max_coeff_is_enforced(
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    """``coal_chp_extraction`` sets ``capacity_max_coeff = 0.8`` on the
    electricity output (west), whose demand peaks at 870 MW.  The output
    must stop at 0.8 · 1000 = 800 MW (maxToSink) — and must actually reach
    it, i.e. neither the per-output coefficient may be dropped nor may the
    fuel arc cap the unit below it (it used to pin west at 470 MW)."""
    if str(TESTS_DIR) not in sys.path:
        sys.path.insert(0, str(TESTS_DIR))
    from db_utils import json_to_db  # noqa: E402
    from flextool.engine_polars import run_chain_from_db
    from flextool.update_flextool.db_migration import migrate_database

    work = tmp_path_factory.mktemp("indirect_output_capcoef")
    url = json_to_db(BASE_FIXTURE_JSON, work / "tests.sqlite")
    migrate_database(url)
    steps = run_chain_from_db(url, "coal_chp_extraction",
                              work_folder=work, keep_solutions=True)
    step = list(steps.values())[-1]
    us = _unitsize(step.flex_data)
    flow = (step.solution.value("v_flow")
            .with_columns(pl.col(c).cast(pl.Utf8)
                          for c in ("p", "source", "sink")))
    west = flow.filter((pl.col("p") == UNIT) & (pl.col("source") == UNIT)
                       & (pl.col("sink") == "west"))
    assert west.height > 0
    assert float(west["value"].max()) * us == pytest.approx(0.8 * CAPACITY,
                                                            rel=1e-6)
