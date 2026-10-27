"""Δ.10 — Cluster F (scalar arithmetic) parity tests.

Per-fixture parity sweep + targeted tests for the cluster F helpers in
:mod:`flextool.engine.preprocess.params._arithmetic` and the existing
cluster F helpers in :mod:`flextool.engine.preprocess.params._derived`
(``p_slope``, ``p_section``, ``p_flow_upper_existing``, ``p_state_upper``,
``p_process_existing_count``).

Per-fixture check: each cluster F field of a rebuild from the run's post-solve provider (the
production loader call, ``tests._support.load.load_scenario``) equals the field of the FlexData the
cascade itself built and solved (``run.last_flex_data``).  A difference means the rebuild path and the
cascade build disagree on that field for that fixture.  (These tests used to compare a CSV load with a
DB-direct load of the same snapshot; once the snapshot became DB-derived both sides took the same path
and the comparison could no longer fail.)

Cluster F coverage:

* ``p_unitsize``                — per-process unitsize cascade.
* ``p_state_unitsize``          — per-node unitsize cascade.
* ``p_penalty_up`` / ``p_penalty_down`` — node penalty broadcast.
* ``p_process_source_conversion_flow_coeff`` /
  ``p_process_sink_conversion_flow_coeff`` — indirect conversion-flow
  coefficient.
* ``p_slope``                   — efficiency-curve slope (verified).
* ``p_section``                 — min_load_efficiency y-intercept.
* ``p_flow_upper_existing``     — existing/unitsize per arc.
* ``p_state_upper``             — capacity / unitsize per (n, d).
* ``p_process_existing_count``  — existing / unitsize per (p, d).
"""
from __future__ import annotations

import polars as pl
import pytest

from flextool.engine import InMemoryReader
from flextool.engine.preprocess.params import _arithmetic as ar
from tests._support.fields import field_diff
from tests._support.load import load_scenario


# Phase 3d: curated parity sweep — replaces the legacy disk-discovery
# function.  See ``tests/engine/conftest.py::PARITY_SWEEP_CASES``.
from _parity_sweep import PARITY_SWEEP_CASES  # noqa: E402

PARITY_CASES = [(legacy, scen, dbf) for legacy, scen, dbf in PARITY_SWEEP_CASES]


# ---------------------------------------------------------------------------
# Per-fixture rebuild check
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def rebuilt(scenario_run):
    """``(scenario, db_fixture) -> (rebuilt FlexData, cascade FlexData)``, one rebuild per scenario.

    The rebuild is the production loader call on the run's post-solve provider (``load_scenario``); the
    reference is the FlexData the cascade itself built and solved for its last step
    (``run.last_flex_data``, the step's own ``flex_data`` object, not a second load).
    """
    cache: dict[tuple[str, str], tuple[object, object]] = {}

    def get(scenario: str, db_fixture: str) -> tuple[object, object]:
        key = (scenario, db_fixture)
        if key not in cache:
            run = scenario_run(scenario, db_fixture=db_fixture)
            assert run.last_flex_data is not None, f"{scenario}: the cascade kept no FlexData"
            cache[key] = (load_scenario(run), run.last_flex_data)
        return cache[key]

    return get


def _check_rebuild(rebuilt, work_name: str, scenario: str, db_fixture: str, field: str) -> None:
    """``field`` of the rebuild from the post-solve provider == the cascade's own FlexData field."""
    data, cascade = rebuilt(scenario, db_fixture)
    diff = field_diff(getattr(data, field), getattr(cascade, field))
    assert diff is None, f"{field}: rebuild != cascade on {work_name}: {diff}"


@pytest.mark.parametrize(
    "work_name,scenario,db_fixture", PARITY_CASES,
    ids=[c[0] for c in PARITY_CASES],
)
def test_p_unitsize_parity(work_name: str, scenario: str, db_fixture: str, rebuilt) -> None:
    """``p_unitsize``: rebuild from the post-solve provider == the cascade's FlexData (cluster F)."""
    _check_rebuild(rebuilt, work_name, scenario, db_fixture, "p_unitsize")


@pytest.mark.parametrize(
    "work_name,scenario,db_fixture", PARITY_CASES,
    ids=[c[0] for c in PARITY_CASES],
)
def test_p_state_unitsize_parity(work_name: str, scenario: str, db_fixture: str, rebuilt) -> None:
    """``p_state_unitsize``: rebuild from the post-solve provider == the cascade's FlexData (cluster F)."""
    _check_rebuild(rebuilt, work_name, scenario, db_fixture, "p_state_unitsize")


@pytest.mark.parametrize(
    "work_name,scenario,db_fixture", PARITY_CASES,
    ids=[c[0] for c in PARITY_CASES],
)
def test_p_penalty_up_parity(work_name: str, scenario: str, db_fixture: str, rebuilt) -> None:
    """``p_penalty_up``: rebuild from the post-solve provider == the cascade's FlexData (cluster F)."""
    _check_rebuild(rebuilt, work_name, scenario, db_fixture, "p_penalty_up")


@pytest.mark.parametrize(
    "work_name,scenario,db_fixture", PARITY_CASES,
    ids=[c[0] for c in PARITY_CASES],
)
def test_p_penalty_down_parity(work_name: str, scenario: str, db_fixture: str, rebuilt) -> None:
    """``p_penalty_down``: rebuild from the post-solve provider == the cascade's FlexData (cluster F)."""
    _check_rebuild(rebuilt, work_name, scenario, db_fixture, "p_penalty_down")


@pytest.mark.parametrize(
    "work_name,scenario,db_fixture", PARITY_CASES,
    ids=[c[0] for c in PARITY_CASES],
)
def test_p_process_source_conversion_flow_coeff_parity(work_name: str, scenario: str, db_fixture: str, rebuilt) -> None:
    """``p_process_source_conversion_flow_coeff``: rebuild from the post-solve provider == the cascade's FlexData
    (cluster F)."""
    _check_rebuild(rebuilt, work_name, scenario, db_fixture, "p_process_source_conversion_flow_coeff")


@pytest.mark.parametrize(
    "work_name,scenario,db_fixture", PARITY_CASES,
    ids=[c[0] for c in PARITY_CASES],
)
def test_p_process_sink_conversion_flow_coeff_parity(work_name: str, scenario: str, db_fixture: str, rebuilt) -> None:
    """``p_process_sink_conversion_flow_coeff``: rebuild from the post-solve provider == the cascade's FlexData
    (cluster F)."""
    _check_rebuild(rebuilt, work_name, scenario, db_fixture, "p_process_sink_conversion_flow_coeff")


@pytest.mark.parametrize(
    "work_name,scenario,db_fixture", PARITY_CASES,
    ids=[c[0] for c in PARITY_CASES],
)
def test_p_slope_full_sweep_parity(work_name: str, scenario: str, db_fixture: str, rebuilt) -> None:
    """``p_slope``: rebuild from the post-solve provider == the cascade's FlexData (cluster F)."""
    _check_rebuild(rebuilt, work_name, scenario, db_fixture, "p_slope")


@pytest.mark.parametrize(
    "work_name,scenario,db_fixture", PARITY_CASES,
    ids=[c[0] for c in PARITY_CASES],
)
def test_p_section_full_sweep_parity(work_name: str, scenario: str, db_fixture: str, rebuilt) -> None:
    """``p_section``: rebuild from the post-solve provider == the cascade's FlexData (cluster F)."""
    _check_rebuild(rebuilt, work_name, scenario, db_fixture, "p_section")


@pytest.mark.parametrize(
    "work_name,scenario,db_fixture", PARITY_CASES,
    ids=[c[0] for c in PARITY_CASES],
)
def test_p_flow_upper_existing_full_sweep_parity(work_name: str, scenario: str, db_fixture: str, rebuilt) -> None:
    """``p_flow_upper_existing``: rebuild from the post-solve provider == the cascade's FlexData (cluster F)."""
    _check_rebuild(rebuilt, work_name, scenario, db_fixture, "p_flow_upper_existing")


@pytest.mark.parametrize(
    "work_name,scenario,db_fixture", PARITY_CASES,
    ids=[c[0] for c in PARITY_CASES],
)
def test_p_state_upper_full_sweep_parity(work_name: str, scenario: str, db_fixture: str, rebuilt) -> None:
    """``p_state_upper``: rebuild from the post-solve provider == the cascade's FlexData (cluster F)."""
    _check_rebuild(rebuilt, work_name, scenario, db_fixture, "p_state_upper")


@pytest.mark.parametrize(
    "work_name,scenario,db_fixture", PARITY_CASES,
    ids=[c[0] for c in PARITY_CASES],
)
def test_p_process_existing_count_full_sweep_parity(work_name: str, scenario: str, db_fixture: str, rebuilt) -> None:
    """``p_process_existing_count``: rebuild from the post-solve provider == the cascade's FlexData (cluster F)."""
    _check_rebuild(rebuilt, work_name, scenario, db_fixture, "p_process_existing_count")


# ---------------------------------------------------------------------------
# In-memory unit tests for the new helpers
# ---------------------------------------------------------------------------


def test_p_unitsize_inmemory_default_cascade():
    """Three units: one with virtual_unitsize set, one with existing
    only, one with neither — cascade returns 50, 200, 1000 respectively
    (when each unit appears in pss).
    """
    src = InMemoryReader(
        entities={
            "unit": pl.DataFrame({"name": ["u_vu", "u_ex", "u_def"]}),
            "node": pl.DataFrame({"name": ["n_x"]}),
            "connection": pl.DataFrame(schema={"name": pl.Utf8}),
            "unit__inputNode": pl.DataFrame(schema={
                "unit": pl.Utf8, "node": pl.Utf8,
            }),
            "unit__outputNode": pl.DataFrame(schema={
                "unit": pl.Utf8, "node": pl.Utf8,
            }),
        },
        parameters={
            ("unit", "virtual_unitsize"): pl.DataFrame({
                "name": ["u_vu"], "value": [50.0],
            }),
            ("unit", "existing"): pl.DataFrame({
                "name": ["u_ex"], "value": [200.0],
            }),
        },
    )
    pss = pl.DataFrame({
        "p": ["u_vu", "u_ex", "u_def"],
        "source": ["n_x", "n_x", "n_x"],
        "sink": ["u_vu", "u_ex", "u_def"],
    })
    p = ar.p_unitsize_from_source(src, pss)
    assert p is not None
    rows = dict(p.frame.sort("p").iter_rows())
    assert rows == {"u_def": 1000.0, "u_ex": 200.0, "u_vu": 50.0}


def test_p_unitsize_inmemory_pss_filter():
    """Process unitsize is filtered to processes appearing in *pss*.
    A unit not in pss is dropped from the output.
    """
    src = InMemoryReader(
        entities={
            "unit": pl.DataFrame({"name": ["u_in_pss", "u_ghost"]}),
            "node": pl.DataFrame({"name": ["n_x"]}),
            "connection": pl.DataFrame(schema={"name": pl.Utf8}),
        },
        parameters={
            ("unit", "virtual_unitsize"): pl.DataFrame({
                "name": ["u_in_pss", "u_ghost"], "value": [10.0, 20.0],
            }),
        },
    )
    pss = pl.DataFrame({
        "p": ["u_in_pss"], "source": ["n_x"], "sink": ["u_in_pss"],
    })
    p = ar.p_unitsize_from_source(src, pss)
    assert p is not None
    assert p.frame["p"].to_list() == ["u_in_pss"]
    assert p.frame["value"][0] == 10.0


def test_p_unitsize_inmemory_empty_pss_returns_none():
    """No processes in pss → helper returns None."""
    src = InMemoryReader(
        entities={
            "unit": pl.DataFrame({"name": ["u"]}),
            "node": pl.DataFrame({"name": ["n"]}),
            "connection": pl.DataFrame(schema={"name": pl.Utf8}),
        },
        parameters={
            ("unit", "virtual_unitsize"): pl.DataFrame({
                "name": ["u"], "value": [1.0],
            }),
        },
    )
    empty_pss = pl.DataFrame(schema={
        "p": pl.Utf8, "source": pl.Utf8, "sink": pl.Utf8,
    })
    p = ar.p_unitsize_from_source(src, empty_pss)
    assert p is None


def test_p_state_unitsize_inmemory_filters_to_node_state():
    """State unitsize: cascade restricted to nodes in nodeState_df."""
    src = InMemoryReader(
        entities={
            "node": pl.DataFrame({"name": ["n_state", "n_other"]}),
            "unit": pl.DataFrame(schema={"name": pl.Utf8}),
            "connection": pl.DataFrame(schema={"name": pl.Utf8}),
        },
        parameters={
            ("node", "virtual_unitsize"): pl.DataFrame({
                "name": ["n_state"], "value": [42.0],
            }),
        },
    )
    nodeState = pl.DataFrame({"n": ["n_state"]})
    p = ar.p_state_unitsize_from_source(src, nodeState)
    assert p is not None
    rows = dict(p.frame.iter_rows())
    assert rows == {"n_state": 42.0}


def test_p_state_unitsize_inmemory_empty_nodestate_returns_none():
    """No nodeState entries → helper returns None."""
    src = InMemoryReader(
        entities={
            "node": pl.DataFrame({"name": ["n"]}),
            "unit": pl.DataFrame(schema={"name": pl.Utf8}),
            "connection": pl.DataFrame(schema={"name": pl.Utf8}),
        },
        parameters={
            ("node", "virtual_unitsize"): pl.DataFrame({
                "name": ["n"], "value": [1.0],
            }),
        },
    )
    empty_ns = pl.DataFrame(schema={"n": pl.Utf8})
    p = ar.p_state_unitsize_from_source(src, empty_ns)
    assert p is None


def test_p_unitsize_inmemory_process_absent_from_cascade_defaults(caplog):
    """Regression — a process present in *pss* but absent from the
    entity-unitsize cascade (the structural set / entity-vocabulary
    divergence that crashed ``model/core.py`` with ``Expr * NoneType``) must
    NOT empty the projection.  It is covered by the canonical 1000.0
    default so ``p_unitsize`` is complete and never None for a non-empty
    pss, and a warning is emitted.

    Here ``u_orphan`` appears in pss (so the model would build a v_flow
    for it) but is NOT in the ``unit`` entity set, so the cascade omits
    it.  The old inner-join dropped it; with only orphans it returned
    None → build crash.
    """
    src = InMemoryReader(
        entities={
            "unit": pl.DataFrame({"name": ["u_real"]}),
            "node": pl.DataFrame({"name": ["n_x"]}),
            "connection": pl.DataFrame(schema={"name": pl.Utf8}),
        },
        parameters={
            ("unit", "virtual_unitsize"): pl.DataFrame({
                "name": ["u_real"], "value": [75.0],
            }),
        },
    )
    # pss carries an orphan process not backed by a unit entity.
    pss = pl.DataFrame({
        "p": ["u_real", "u_orphan"],
        "source": ["n_x", "n_x"],
        "sink": ["u_real", "u_orphan"],
    })
    import logging
    with caplog.at_level(logging.WARNING):
        p = ar.p_unitsize_from_source(src, pss)
    assert p is not None
    rows = dict(p.frame.sort("p").iter_rows())
    assert rows == {"u_real": 75.0, "u_orphan": 1000.0}
    assert any("u_orphan" in r.message and "p_unitsize" in r.message
               for r in caplog.records), "expected a divergence warning"


def test_p_unitsize_inmemory_all_processes_orphan_not_none(caplog):
    """Even when EVERY pss process is absent from the cascade, the helper
    returns a complete Param (all defaulted) — never None — so the build
    cannot crash on ``v_flow * None``.
    """
    src = InMemoryReader(
        entities={
            "unit": pl.DataFrame(schema={"name": pl.Utf8}),
            "node": pl.DataFrame({"name": ["n_x"]}),
            "connection": pl.DataFrame(schema={"name": pl.Utf8}),
        },
        parameters={},
    )
    pss = pl.DataFrame({
        "p": ["u_a", "u_b"], "source": ["n_x", "n_x"], "sink": ["u_a", "u_b"],
    })
    p = ar.p_unitsize_from_source(src, pss)
    assert p is not None
    assert dict(p.frame.sort("p").iter_rows()) == {
        "u_a": 1000.0, "u_b": 1000.0}


def test_p_state_unitsize_inmemory_node_absent_from_cascade_defaults():
    """Per-node mirror: a nodeState node absent from the node cascade is
    covered by the canonical default rather than dropped/None.
    """
    src = InMemoryReader(
        entities={
            "node": pl.DataFrame({"name": ["n_real"]}),
            "unit": pl.DataFrame(schema={"name": pl.Utf8}),
            "connection": pl.DataFrame(schema={"name": pl.Utf8}),
        },
        parameters={
            ("node", "virtual_unitsize"): pl.DataFrame({
                "name": ["n_real"], "value": [42.0],
            }),
        },
    )
    nodeState = pl.DataFrame({"n": ["n_real", "n_orphan"]})
    p = ar.p_state_unitsize_from_source(src, nodeState)
    assert p is not None
    assert dict(p.frame.sort("n").iter_rows()) == {
        "n_real": 42.0, "n_orphan": 1000.0}


def test_p_penalty_up_inmemory_scalar_broadcast():
    """Per-node scalar penalty broadcasts over (n, d, t) restricted to
    nodeBalance nodes.
    """
    src = InMemoryReader(
        entities={"node": pl.DataFrame({"name": ["n_a", "n_b"]})},
        parameters={
            ("node", "penalty_up"): pl.DataFrame({
                "name": ["n_a", "n_b"], "value": [100.0, 200.0],
            }),
        },
    )
    nb = pl.DataFrame({"n": ["n_a"]})  # n_b excluded
    dt = pl.DataFrame({
        "d": ["d1", "d1", "d2"], "t": ["t1", "t2", "t1"],
    })
    p = ar.p_penalty_up_from_source(src, nb, dt)
    assert p is not None
    f = p.frame.sort("n", "d", "t")
    assert f["n"].to_list() == ["n_a"] * 3
    assert f["value"].to_list() == [100.0] * 3


def test_p_penalty_up_inmemory_no_param_returns_none():
    """No penalty_up param → helper returns None."""
    src = InMemoryReader(
        entities={"node": pl.DataFrame({"name": ["n"]})},
        parameters={},
    )
    nb = pl.DataFrame({"n": ["n"]})
    dt = pl.DataFrame({"d": ["d"], "t": ["t"]})
    assert ar.p_penalty_up_from_source(src, nb, dt) is None


def _penalty_dt() -> pl.DataFrame:
    return pl.DataFrame({"d": ["d1", "d1", "d2", "d2"], "t": ["t1", "t2", "t3", "t4"]})


def test_p_penalty_up_mixed_shapes_keep_scalars_and_default():
    """A period map on one node next to a scalar on another and no value on a third (#62): the scalar
    stays on every (d, t), the map's node gets the default in the period its map leaves out, and the
    node without a value gets the schema default."""
    src = InMemoryReader(
        entities={"node": pl.DataFrame({"name": ["n_a", "n_b", "n_c"]})},
        parameters={
            ("node", "penalty_up"): pl.DataFrame({
                "name": ["n_a", "n_b"], "period": [None, "d1"], "value": [100.0, 200.0],
            }),
        },
        defaults={("node", "penalty_up"): 10000.0},
    )
    nb = pl.DataFrame({"n": ["n_a", "n_b", "n_c"]})
    p = ar.p_penalty_up_from_source(src, nb, _penalty_dt())
    assert p is not None and p.dims == ("n", "d", "t")
    got = {(n, d, t): v for n, d, t, v in p.frame.select(
        [pl.col(c).cast(pl.Utf8) for c in ("n", "d", "t")] + [pl.col("value")]).iter_rows()}
    want = {}
    for d, t in _penalty_dt().iter_rows():
        want[("n_a", d, t)] = 100.0
        want[("n_b", d, t)] = 200.0 if d == "d1" else 10000.0
        want[("n_c", d, t)] = 10000.0
    assert got == want


def _stochastic_penalty_source() -> InMemoryReader:
    """``n_a`` authors a rank-3 (branch, analysis_time, time) penalty, ``n_b`` a period map, ``n_c`` nothing."""
    frame = pl.concat([
        pl.DataFrame({"name": ["n_a"], "branch": ["b1"], "analysis_time": ["t1"], "t": ["t1"], "value": [5.0]}),
        pl.DataFrame({"name": ["n_b"], "period": ["d1"], "value": [200.0]}),
    ], how="diagonal_relaxed")
    return InMemoryReader(
        entities={"node": pl.DataFrame({"name": ["n_a", "n_b", "n_c"]})},
        parameters={("node", "penalty_up"): frame},
        defaults={("node", "penalty_up"): 10000.0},
    )


def test_p_penalty_up_stochastic_time_needs_solve_emit():
    """A rank-3 penalty is read from the solve's emit fold; outside a load (no emit) that is an error, not a
    silent fallback."""
    from flextool.engine.input._solve_state import FlexToolConfigError

    nb = pl.DataFrame({"n": ["n_a", "n_b", "n_c"]})
    with pytest.raises(FlexToolConfigError, match="only be read during load_flextool"):
        ar.p_penalty_up_from_source(_stochastic_penalty_source(), nb, _penalty_dt())


def test_p_penalty_up_stochastic_time_merges_fold():
    """With the solve's emit active: the rank-3 node gets its folded values, the other nodes their own value or
    the default (before the fix the seed was kept for every node, hiding ``n_b``'s map)."""
    from flextool.engine.input._param_shapes import (
        activate_stochastic_time_source,
        deactivate_stochastic_time_source,
    )

    class _Provider:
        def __init__(self, frames):
            self.frames = frames

        def has(self, name):
            return name in self.frames

        def get(self, name):
            return self.frames[name]

    folded = {(d, t): 6000.0 + i for i, (d, t) in enumerate(_penalty_dt().iter_rows())}
    pdt_node = pl.DataFrame([("n_a", "penalty_up", d, t, repr(v)) for (d, t), v in folded.items()],
                            schema=["node", "param", "period", "time", "value"], orient="row")
    nb = pl.DataFrame({"n": ["n_a", "n_b", "n_c"]})
    token = activate_stochastic_time_source(_Provider({"solve_data/pdtNode": pdt_node}))
    try:
        p = ar.p_penalty_up_from_source(_stochastic_penalty_source(), nb, _penalty_dt())
    finally:
        deactivate_stochastic_time_source(token)
    got = {(n, d, t): v for n, d, t, v in p.frame.select(
        [pl.col(c).cast(pl.Utf8) for c in ("n", "d", "t")] + [pl.col("value")]).iter_rows()}
    want = {}
    for d, t in _penalty_dt().iter_rows():
        want[("n_a", d, t)] = folded[(d, t)]
        want[("n_b", d, t)] = 200.0 if d == "d1" else 10000.0
        want[("n_c", d, t)] = 10000.0
    assert got == want


_GRID = pl.DataFrame({"n": ["a", "a", "a", "b", "b", "b"],
                      "d": ["d1", "d1", "d2", "d1", "d1", "d2"],
                      "t": ["t1", "t2", "t3", "t1", "t2", "t3"]})
_ENTITIES = _GRID.select("n")
_STEPS = _GRID.select("d", "t")


@pytest.mark.parametrize("dims, rows, want_a", [
    (("n",), {"n": ["a"], "value": [1.0]}, [1.0, 1.0, 1.0]),
    (("n", "d"), {"n": ["a"], "d": ["d2"], "value": [2.0]}, [9.0, 9.0, 2.0]),
    (("n", "t"), {"n": ["a"], "t": ["t2"], "value": [3.0]}, [9.0, 3.0, 9.0]),
    (("n", "d", "t"), {"n": ["a"], "d": ["d1"], "t": ["t1"], "value": [4.0]}, [4.0, 9.0, 9.0]),
], ids=["scalar", "period", "time", "period_time"])
def test_fill_default_over_grid_shapes(dims, rows, want_a):
    from polar_high import Param

    from flextool.engine.input._param_shapes import fill_default_over_grid

    out = fill_default_over_grid(Param(dims, pl.DataFrame(rows)), _ENTITIES, _STEPS, 9.0, name="x")
    assert out.dims == ("n", "d", "t")
    assert isinstance(out.frame, pl.DataFrame)
    f = out.frame
    assert f.select("n", "d", "t").equals(_GRID.sort("n", "d", "t"))
    assert f.filter(pl.col("n") == "a")["value"].to_list() == want_a
    assert f.filter(pl.col("n") == "b")["value"].to_list() == [9.0, 9.0, 9.0]


def test_fill_default_over_grid_without_default_or_values():
    from polar_high import Param

    from flextool.engine.input._param_shapes import fill_default_over_grid

    authored = Param(("n", "d"), pl.DataFrame({"n": ["a"], "d": ["d2"], "value": [2.0]}))
    inner = fill_default_over_grid(authored, _ENTITIES, _STEPS, None, name="x")
    assert inner.frame.rows() == [("a", "d2", "t3", 2.0)]
    assert fill_default_over_grid(None, _ENTITIES, _STEPS, None, name="x") is None
    everywhere = fill_default_over_grid(None, _ENTITIES, _STEPS, 7.0, name="x")
    assert everywhere.frame.height == _GRID.height and set(everywhere.frame["value"]) == {7.0}


def test_fill_default_over_grid_order():
    """Unsorted factors with repeated rows: the grid is unique sorted entities × unique sorted steps (the
    solve's pairs, not d × t), already in key order, with no duplicate cells."""
    from polar_high import Param

    from flextool.engine.input._param_shapes import fill_default_over_grid

    entities = pl.DataFrame({"n": ["b", "a", "b", "a"]})
    steps = pl.DataFrame({"d": ["d2", "d1", "d1", "d2", "d1"], "t": ["t3", "t2", "t1", "t3", "t2"]})
    authored = Param(("n", "t"), pl.DataFrame({"n": ["b"], "t": ["t2"], "value": [5.0]}))
    out = fill_default_over_grid(authored, entities, steps, 9.0, name="x").frame
    assert out.rows() == [
        ("a", "d1", "t1", 9.0), ("a", "d1", "t2", 9.0), ("a", "d2", "t3", 9.0),
        ("b", "d1", "t1", 9.0), ("b", "d1", "t2", 5.0), ("b", "d2", "t3", 9.0)]
    assert out.equals(out.sort("n", "d", "t"))
    group = fill_default_over_grid(None, entities.rename({"n": "g"}), steps.select("d"), 5000.0, name="g")
    assert group.dims == ("g", "d")
    assert group.frame.rows() == [("a", "d1", 5000.0), ("a", "d2", 5000.0), ("b", "d1", 5000.0),
                                  ("b", "d2", 5000.0)]


def test_fill_default_over_grid_rejects_duplicate_keys():
    """Duplicate authored keys would duplicate grid rows (terms counted twice): raise instead."""
    from polar_high import Param

    from flextool.engine.input._param_shapes import fill_default_over_grid
    from flextool.spinedb_backend._axis_enums import FlexDataIntegrityError

    dup = Param(("n", "d"), pl.DataFrame({"n": ["a", "a"], "d": ["d1", "d1"], "value": [1.0, 2.0]}))
    with pytest.raises(FlexDataIntegrityError, match="node.penalty_up.*share their key"):
        fill_default_over_grid(dup, _ENTITIES, _STEPS, 9.0, name="node.penalty_up")


def test_p_process_source_conversion_flow_coeff_inmemory_zero_drop():
    """Zero-coef row is reported in zero_pairs; non-default non-zero
    triggers a Param keyed on the surviving (p, source) set.
    """
    src = InMemoryReader(
        entities={
            "unit": pl.DataFrame({"name": ["u_chp"]}),
            "node": pl.DataFrame({"name": ["n_fuel", "n_zero"]}),
            "unit__inputNode": pl.DataFrame({
                "unit": ["u_chp", "u_chp"],
                "node": ["n_fuel", "n_zero"],
            }),
        },
        parameters={
            ("unit__inputNode", "conversion_flow_coeff"): pl.DataFrame({
                "unit": ["u_chp", "u_chp"],
                "node": ["n_fuel", "n_zero"],
                "value": [2.0, 0.0],
            }),
        },
    )
    indirect_pairs = pl.DataFrame({
        "p": ["u_chp", "u_chp"],
        "source": ["n_fuel", "n_zero"],
        "sink": ["u_chp", "u_chp"],
    })
    z, p = ar.p_process_source_conversion_flow_coeff_from_source(src, indirect_pairs)
    assert z is not None
    assert dict(z.iter_rows()) == {"u_chp": "n_zero"}
    assert p is not None
    # surviving pair gets the non-default coefficient.  Zero rows are NOT
    # in the Param frame because we anti-join inputs first.
    f = p.frame.sort("p", "source")
    # Note the helper covers ALL surviving pairs in indirect_pairs,
    # including the zero one — the caller is expected to anti-join
    # zero_pairs before re-keying.  Here we passed both pairs in.
    assert f.height == 2
    rows = {(r["p"], r["source"]): r["value"]
              for r in f.iter_rows(named=True)}
    assert rows[("u_chp", "n_fuel")] == 2.0
    assert rows[("u_chp", "n_zero")] == 0.0  # the zero is the actual coef


def test_p_process_source_conversion_flow_coeff_inmemory_all_default_returns_none():
    """When every coefficient equals 1.0, the helper emits no Param —
    model/core.py's gate falls through.
    """
    src = InMemoryReader(
        entities={
            "unit": pl.DataFrame({"name": ["u"]}),
            "node": pl.DataFrame({"name": ["n_x"]}),
            "unit__inputNode": pl.DataFrame({
                "unit": ["u"], "node": ["n_x"],
            }),
        },
        parameters={
            ("unit__inputNode", "conversion_flow_coeff"): pl.DataFrame({
                "unit": ["u"], "node": ["n_x"], "value": [1.0],
            }),
        },
    )
    indirect_pairs = pl.DataFrame({
        "p": ["u"], "source": ["n_x"], "sink": ["u"],
    })
    z, p = ar.p_process_source_conversion_flow_coeff_from_source(src, indirect_pairs)
    assert z is None
    assert p is None


def test_p_process_sink_conversion_flow_coeff_inmemory_smoke():
    """``p_process_sink_conversion_flow_coeff`` is symmetric to source —
    same contract on ``unit__outputNode``.
    """
    src = InMemoryReader(
        entities={
            "unit": pl.DataFrame({"name": ["u"]}),
            "node": pl.DataFrame({"name": ["heat", "elec"]}),
            "unit__outputNode": pl.DataFrame({
                "unit": ["u", "u"], "node": ["heat", "elec"],
            }),
        },
        parameters={
            ("unit__outputNode", "conversion_flow_coeff"): pl.DataFrame({
                "unit": ["u", "u"],
                "node": ["heat", "elec"],
                "value": [0.2, 2.0],
            }),
        },
    )
    indirect_pairs = pl.DataFrame({
        "p": ["u", "u"], "source": ["u", "u"], "sink": ["heat", "elec"],
    })
    z, p = ar.p_process_sink_conversion_flow_coeff_from_source(src, indirect_pairs)
    assert z is None  # neither coef is zero
    assert p is not None
    rows = {(r["p"], r["sink"]): r["value"]
              for r in p.frame.iter_rows(named=True)}
    assert rows == {("u", "heat"): 0.2, ("u", "elec"): 2.0}
