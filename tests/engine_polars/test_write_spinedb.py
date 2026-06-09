"""Round-trip tests for the ``spinedb`` output write-method.

These build a tiny synthetic ``results`` dict + a faithful stub ``s`` (no
full solve) and assert the SpineDB round-trips entities, alternatives, and
nested Map parameter values per the schema.
"""

import logging
import multiprocessing as mp

import pandas as pd
import pytest
from spinedb_api import DatabaseMapping, from_database

from flextool.process_outputs.spinedb_replay import (
    _CONNECTION_PREFIX,
    _collect_process_names,
    _process_node_map,
    build_replay_s,
)
from flextool.process_outputs.write_spinedb import (
    ensure_results_db,
    write_spinedb,
)


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

class StubS:
    """Minimal faithful stand-in for the engine ``s`` (sets) namespace.

    Exposes only the attributes the writer reads: ``solve_period_time``,
    ``solve_period``, ``process_unit``, ``process_connection`` and
    ``process_source_sink`` (connection topology).
    """

    def __init__(self):
        # One solve "s1" with one period "p2020" and two timesteps.
        self.solve_period_time = pd.MultiIndex.from_tuples(
            [("s1", "p2020", "t01"), ("s1", "p2020", "t02")],
            names=["solve", "period", "time"],
        )
        self.solve_period = pd.MultiIndex.from_tuples(
            [("s1", "p2020")], names=["solve", "period"],
        )
        self.process_unit = pd.Index(["u1"], name="process")
        self.process_connection = pd.Index(["c1"], name="process")
        # connection topology: c1 connects n1 (left) and n2 (right)
        self.process_source_sink = pd.MultiIndex.from_tuples(
            [("c1", "n1", "n2")], names=["process", "source", "sink"],
        )


def _dt_index():
    return pd.MultiIndex.from_tuples(
        [("p2020", "t01"), ("p2020", "t02")], names=["period", "time"],
    )


def _flow_df(value_t01=42.0, value_t02=7.0):
    """unit_outputNode_dt_ee: (period, time) index, (unit, node) cols."""
    cols = pd.MultiIndex.from_tuples([("u1", "n1")], names=["unit", "node"])
    return pd.DataFrame(
        {("u1", "n1"): [value_t01, value_t02]}, index=_dt_index(), columns=cols,
    )


def _capacity_df():
    """unit_capacity_ed_p: (unit, period) index, capacity-category cols."""
    idx = pd.MultiIndex.from_tuples([("u1", "p2020")], names=["unit", "period"])
    df = pd.DataFrame(
        {
            "existing": [100.0],
            "invested": [float("nan")],
            "divested": [10.0],
            "total": [90.0],
        },
        index=idx,
    )
    df.columns.name = "parameter"
    return df


def _read_value(db, ec, byname, param, alternative):
    pvs = db.get_parameter_value_items(
        entity_class_name=ec, alternative_name=alternative,
    )
    for pv in pvs:
        if pv["entity_byname"] == byname and pv["parameter_definition_name"] == param:
            return from_database(pv["value"], pv["type"])
    return None


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_ensure_results_db_creates_schema(tmp_path):
    url = "sqlite:///" + str(tmp_path / "results.sqlite")
    ensure_results_db(url)
    with DatabaseMapping(url) as db:
        assert len(db.get_entity_class_items()) == 12
        assert len(db.get_parameter_definition_items()) == 52
        assert [a["name"] for a in db.get_alternative_items()] == ["Base"]
    # idempotent: second call is a no-op and does not raise
    ensure_results_db(url)
    with DatabaseMapping(url) as db:
        assert len(db.get_entity_class_items()) == 12


def test_round_trip_scalar_and_map(tmp_path):
    url = "sqlite:///" + str(tmp_path / "results.sqlite")
    s = StubS()
    results = {
        "unit_outputNode_dt_ee": _flow_df(),
        "unit_capacity_ed_p": _capacity_df(),
    }
    count = write_spinedb(results, s, url, "scenA", "s1")
    assert count > 0

    with DatabaseMapping(url) as db:
        # alternative present
        alt_names = {a["name"] for a in db.get_alternative_items()}
        assert "scenA" in alt_names

        # relationship + member entities present
        ents = {
            (e["entity_class_name"], e["entity_byname"])
            for e in db.get_entity_items()
        }
        assert ("unit__node", ("u1", "n1")) in ents
        assert ("unit", ("u1",)) in ents
        assert ("node", ("n1",)) in ents

        # flow_t nested Map solve -> period -> time
        flow = _read_value(db, "unit__node", ("u1", "n1"), "flow_t", "scenA")
        assert flow is not None
        assert flow.get_value("s1").get_value("p2020").get_value("t01") == 42.0
        assert flow.get_value("s1").get_value("p2020").get_value("t02") == 7.0

        # capacity Map has total + divested keys, NaN invested dropped
        cap = _read_value(db, "unit", ("u1",), "capacity", "scenA")
        assert cap is not None
        cap_cats = set(cap.indexes)
        assert "total" in cap_cats
        assert "divested" in cap_cats
        assert "existing" in cap_cats
        assert "invested" not in cap_cats  # all-NaN dropped
        assert cap.get_value("total").get_value("s1").get_value("p2020") == 90.0


def test_two_runs_two_alternatives(tmp_path):
    url = "sqlite:///" + str(tmp_path / "results.sqlite")
    s = StubS()
    write_spinedb({"unit_outputNode_dt_ee": _flow_df(42.0, 7.0)}, s, url,
                  "scenA", "s1")
    write_spinedb({"unit_outputNode_dt_ee": _flow_df(99.0, 1.0)}, s, url,
                  "scenB", "s1")

    with DatabaseMapping(url) as db:
        alt_names = {a["name"] for a in db.get_alternative_items()}
        assert {"scenA", "scenB"} <= alt_names

        a = _read_value(db, "unit__node", ("u1", "n1"), "flow_t", "scenA")
        b = _read_value(db, "unit__node", ("u1", "n1"), "flow_t", "scenB")
        assert a.get_value("s1").get_value("p2020").get_value("t01") == 42.0
        assert b.get_value("s1").get_value("p2020").get_value("t01") == 99.0


def test_rerun_purges_alternative(tmp_path):
    url = "sqlite:///" + str(tmp_path / "results.sqlite")
    s = StubS()
    write_spinedb({"unit_outputNode_dt_ee": _flow_df(42.0, 7.0)}, s, url,
                  "scenA", "s1")
    write_spinedb({"unit_outputNode_dt_ee": _flow_df(55.0, 7.0)}, s, url,
                  "scenA", "s1")

    with DatabaseMapping(url) as db:
        pvs = [
            pv for pv in db.get_parameter_value_items(
                entity_class_name="unit__node", alternative_name="scenA")
            if pv["parameter_definition_name"] == "flow_t"
            and pv["entity_byname"] == ("u1", "n1")
        ]
        # no duplicate rows
        assert len(pvs) == 1
        val = from_database(pvs[0]["value"], pvs[0]["type"])
        # second value wins
        assert val.get_value("s1").get_value("p2020").get_value("t01") == 55.0


def test_unknown_table_ignored(tmp_path):
    url = "sqlite:///" + str(tmp_path / "results.sqlite")
    s = StubS()
    results = {
        "unit_outputNode_dt_ee": _flow_df(),
        "some_unknown_table_xyz": pd.DataFrame(
            {"a": [1, 2]}, index=pd.Index([0, 1], name="x")),
    }
    # must not crash on the non-whitelisted table
    count = write_spinedb(results, s, url, "scenA", "s1")
    assert count > 0

    with DatabaseMapping(url) as db:
        # only the whitelisted table produced params
        flow = _read_value(db, "unit__node", ("u1", "n1"), "flow_t", "scenA")
        assert flow is not None
        # no entity class / param exists for the unknown table
        ec_names = {e["name"] for e in db.get_entity_class_items()}
        assert "some_unknown_table_xyz" not in ec_names


# ---------------------------------------------------------------------------
# Replay shim: squeezed single-column Series handling (#5)
# ---------------------------------------------------------------------------

def test_collect_process_names_handles_squeezed_series():
    """A connection frame squeezed to a single-column ``Series`` (label in
    ``.name``, no ``.columns``) must still contribute its process — otherwise
    the process set silently under-covers."""
    # connection_dt_eee squeezed to a Series: name = bare connection.
    squeezed_bare = pd.Series(
        [1.0, 2.0], index=_dt_index(), name="conn_only_via_series",
    )
    # connection_leftward_dt_eee squeezed: name = (process, node) tuple.
    squeezed_tuple = pd.Series(
        [3.0, 4.0], index=_dt_index(), name=("conn_via_tuple", "nodeA"),
    )
    results = {
        "connection_dt_eee": squeezed_bare,
        "connection_leftward_dt_eee": squeezed_tuple,
    }
    names = _collect_process_names(results, _CONNECTION_PREFIX)
    assert set(names) == {"conn_only_via_series", "conn_via_tuple"}, (
        "squeezed Series frames dropped from the process set"
    )


def test_process_node_map_handles_squeezed_series():
    """A squeezed ``(process, node)`` directional Series must still map its
    node (label in ``.name``)."""
    squeezed = pd.Series(
        [5.0, 6.0], index=_dt_index(), name=("cX", "west"),
    )
    out = _process_node_map({"connection_leftward_dt_eee": squeezed},
                            "leftward")
    assert out == {"cX": "west"}


# ---------------------------------------------------------------------------
# Replay shim: 2-way (bidirectional) connection detection / warning (#7)
# ---------------------------------------------------------------------------

def _conn_dir_df(direction_node):
    """connection_<direction>_dt_eee with one (process, node) column."""
    cols = pd.MultiIndex.from_tuples([direction_node], names=["process", "node"])
    return pd.DataFrame(
        {direction_node: [1.0, 1.0]}, index=_dt_index(), columns=cols,
    )


def _net_flow_df(connection, values):
    """connection_dt_eee net-flow frame: column = bare connection name."""
    cols = pd.Index([connection], name="process")
    return pd.DataFrame(
        {connection: values}, index=_dt_index(), columns=cols,
    )


def test_two_way_connection_warns(caplog):
    """A connection whose NET flow takes both signs is bidirectional; the
    replay shim must warn and name it."""
    results = {
        "connection_leftward_dt_eee": _conn_dir_df(("cBi", "west")),
        "connection_rightward_dt_eee": _conn_dir_df(("cBi", "battery")),
        # net flow flips sign across the two timesteps -> 2-way.
        "connection_dt_eee": _net_flow_df("cBi", [5.0, -3.0]),
    }
    with caplog.at_level(
        logging.WARNING, logger="flextool.process_outputs.spinedb_replay"
    ):
        build_replay_s(results)
    msgs = [r.getMessage() for r in caplog.records
            if "bidirectional" in r.getMessage()]
    assert msgs, "no 2-way warning emitted for a bidirectional connection"
    assert any("cBi" in m for m in msgs), f"warning did not name cBi: {msgs}"


def test_one_way_connection_does_not_warn(caplog):
    """A connection whose net flow is single-signed is 1-way; NO warning."""
    results = {
        "connection_leftward_dt_eee": _conn_dir_df(("cUni", "west")),
        "connection_rightward_dt_eee": _conn_dir_df(("cUni", "east")),
        # net flow strictly positive -> 1-way.
        "connection_dt_eee": _net_flow_df("cUni", [5.0, 3.0]),
    }
    with caplog.at_level(
        logging.WARNING, logger="flextool.process_outputs.spinedb_replay"
    ):
        s = build_replay_s(results)
    msgs = [r.getMessage() for r in caplog.records
            if "bidirectional" in r.getMessage()]
    assert not msgs, f"unexpected 2-way warning for a 1-way connection: {msgs}"
    # 1-way triple round-trips exactly (source=left, sink=right).
    assert list(s.process_source_sink) == [("cUni", "west", "east")]


# ---------------------------------------------------------------------------
# Concurrent cold-start safety on a shared results DB (#4)
# ---------------------------------------------------------------------------

def _ensure_worker(url: str) -> str:
    """Subprocess entry point: ensure the schema DB, return 'ok' or the error
    repr.  Module-level so it is picklable under the 'spawn' start method."""
    try:
        ensure_results_db(url)
        return "ok"
    except Exception as exc:  # noqa: BLE001
        return f"ERR:{type(exc).__name__}:{exc}"


def test_concurrent_cold_start_creates_one_valid_db(tmp_path):
    """Many processes cold-starting the SAME shared results DB must converge
    on ONE valid schema DB (the lock serializes the create; no temp clobbers a
    DB another process is using)."""
    url = "sqlite:///" + str(tmp_path / "shared_results.sqlite")
    ctx = mp.get_context("spawn")
    n = 6
    with ctx.Pool(n) as pool:
        outcomes = pool.map(_ensure_worker, [url] * n)
    assert all(o == "ok" for o in outcomes), f"worker failures: {outcomes}"

    # Exactly one DB file, valid schema, no leftover temp / lock churn that
    # would indicate a clobber.
    path = tmp_path / "shared_results.sqlite"
    assert path.exists()
    leftovers = list(tmp_path.glob(".results-*.sqlite.tmp"))
    assert not leftovers, f"leftover temp DBs: {leftovers}"
    with DatabaseMapping(url) as db:
        assert len(db.get_entity_class_items()) == 12
        assert len(db.get_parameter_definition_items()) == 52

    # The DB stays usable for an append after the concurrent create.
    write_spinedb({"unit_outputNode_dt_ee": _flow_df()}, StubS(), url,
                  "scenA", "s1")
    with DatabaseMapping(url) as db:
        flow = _read_value(db, "unit__node", ("u1", "n1"), "flow_t", "scenA")
        assert flow is not None


if __name__ == "__main__":
    pytest.main([__file__, "-q"])
