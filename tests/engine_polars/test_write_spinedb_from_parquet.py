"""Round-trip test for the ``spinedb`` parquet-REPLAY path.

This exercises the full chain end-to-end on a small native scenario
(``network_coal_wind`` — units + connections, single solve):

1. Run the engine cascade via ``run_chain_from_db`` (DB built from the JSON
   fixture, never a checked-in ``.sqlite`` — CLAUDE.md invariant #3).
2. NATIVE path: ``write_outputs(write_methods=['parquet', 'spinedb'])`` →
   reference ``results.sqlite`` + the processed ``output_parquet`` bundle.
3. REPLAY path: ``write_outputs(read_parquet_dir=True,
   write_methods=['spinedb'])`` reading the SAME parquet bundle, building the
   ``s`` shim purely from the processed results (``build_replay_s``) and
   writing a SECOND results DB.
4. Diff the two DBs.

Assertions:
  (a) every non-discount parameter value (+ its solve/period/time Map keys)
      matches the native DB byte-for-byte;
  (b) the two discount/inflation params (#33/#34) are ABSENT in the replay DB
      (the accepted, documented loss on the parquet path — ``par=None``);
  (c) the connection-triple node ordering (source / sink) in the replay
      matches native (the one caveat flagged in the scoping doc).

Reuses the same harness as ``tests/test_scenarios.py`` (``run_chain_from_db``
→ ``write_outputs``) and the session-scoped DB fixtures from
``tests/conftest.py``.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import pytest
from spinedb_api import DatabaseMapping, from_database

TEST_DIR = Path(__file__).resolve().parent.parent  # tests/
REPO_ROOT = TEST_DIR.parent
OUTPUT_CONFIG = str(REPO_ROOT / "templates" / "default_plots.yaml")

if str(TEST_DIR) not in sys.path:
    sys.path.insert(0, str(TEST_DIR))

from flextool.engine_polars import run_chain_from_db  # noqa: E402
from flextool.process_outputs.write_outputs import write_outputs  # noqa: E402

pytestmark = pytest.mark.solver

# Single-solve scenario carrying both units and (unidirectional) connections,
# so the connection-triple (source, sink) ordering and the unit/connection
# partition are both exercised.  Small enough for a unit test (~1-2 s solve).
#
# UNIDIRECTIONAL connections are used deliberately: for a 1-way arc the
# native ``s.process_source_sink`` has a single row whose (source, sink)
# equals (node_1, node_2) = (leftward_node, rightward_node) — exactly what
# the replay reconstructs from the ``connection_leftward_*`` /
# ``connection_rightward_*`` parquet columns, so the triple round-trips
# byte-identically.
#
# NOTE — 2-way connections are a KNOWN, UNRECOVERABLE limitation of the
# parquet-replay path (see module docstring of ``spinedb_replay`` and the
# scoping doc's G3 caveat): native ``process_source_sink`` carries BOTH arcs
# ``(a, b)`` and ``(b, a)`` sorted, and ``_connection_triple_lookup`` keeps
# the lexicographically-first; the processed parquet retains only the
# (node_1, node_2) geometry and no signal for the arc direction / 2-way-ness,
# so the (source, sink) ORDER cannot be reproduced for 2-way connections.
SCENARIO = "unidirectional_connection"  # uses test_db_url (tests.json)

# MULTI-SOLVE round-trip scenario: a nested cascade combining the
# ``invest_solveSequence_5weeks`` invest sequence (4 distinct period-solves)
# with the ``dispatch_fullYear_roll`` rolling dispatch.  This is the only
# tests.json scenario that simultaneously
#   * runs MANY cascade sub-solves (rolling rolls + an invest sequence), so the
#     ``solve`` Map axis carries multiple labels and the shim's per-period
#     last-solve winner (``solve_period``) and per-(period,time) ``keep='last'``
#     dedup (``solve_period_time``) are BOTH exercised with real duplicates
#     (the realized timeline has 100+ overlapping (period,time) across rolls);
#   * carries a connection — and that connection (``battery_inverter``,
#     ``transfer_method = regular``) is BIDIRECTIONAL (2-way), so it also
#     exercises the replay-path 2-way detection / warning.
#
# Single-solve scenarios (e.g. ``unidirectional_connection`` above) leave the
# rolling logic with ZERO duplicates, so the dedup / winner code is never
# validated there — this scenario is what makes the multi-solve labels real.
MULTISOLVE_SCENARIO = "multi_fullYear_battery_nested_multi_invest"

# The bidirectional connection in the multi-solve scenario.  Its (source, sink)
# node ordering is the KNOWN-UNRECOVERABLE 2-way limitation of the parquet
# replay path, so the multi-solve test asserts the WARNING fires for it rather
# than triple equality (covered exactly for 1-way by the single-solve test).
_MULTISOLVE_TWO_WAY_CONN = "battery_inverter"

# Discount-factor params recoverable ONLY from ``par`` (debug-only on disk):
# intentionally dropped on the replay path.
_DISCOUNT_PARAMS = {
    "investments discount factor",
    "operations discount factor",
}


def _dump_param_values(db_url: str, alternative: str) -> dict:
    """Return ``{(ec, byname, param): rendered_value}`` for one alternative.

    Map values are rendered to a comparable nested-dict / scalar form so the
    two DBs can be diffed structurally (the serialized blobs themselves may
    differ in irrelevant metadata)."""
    out: dict = {}
    with DatabaseMapping(db_url) as db:
        for pv in db.get_parameter_value_items(alternative_name=alternative):
            key = (
                pv["entity_class_name"],
                pv["entity_byname"],
                pv["parameter_definition_name"],
            )
            value = from_database(pv["value"], pv["type"])
            out[key] = _render(value)
    return out


def _render(value):
    """Render a spinedb_api value (Map or scalar) into a comparable form."""
    indexes = getattr(value, "indexes", None)
    if indexes is None:
        # scalar
        return value
    return {
        str(idx): _render(value.get_value(idx))
        for idx in indexes
    }


def _entities(db_url: str) -> set:
    with DatabaseMapping(db_url) as db:
        return {
            (e["entity_class_name"], tuple(e["entity_byname"]))
            for e in db.get_entity_items()
        }


def test_spinedb_parquet_replay_matches_native(
    test_db_url: str,
    test_solver_config_dir: Path,
    tmp_path: Path,
) -> None:
    # --- 1. run the cascade -------------------------------------------------
    steps = run_chain_from_db(
        input_db_url=test_db_url,
        scenario_name=SCENARIO,
        work_folder=tmp_path,
        solver_config_dir=test_solver_config_dir,
        warm=True,
        keep_solutions=True,
    )
    assert steps, f"no steps for scenario {SCENARIO!r}"
    last_step = next(reversed(steps.values()))
    assert last_step.solution is not None and last_step.solution.optimal

    native_db = "sqlite:///" + str(tmp_path / "results_native.sqlite")
    replay_db = "sqlite:///" + str(tmp_path / "results_replay.sqlite")

    common = dict(
        scenario_name=SCENARIO,
        output_location=str(tmp_path),
        subdir=SCENARIO,
        output_config_path=OUTPUT_CONFIG,
        fallback_output_location=str(tmp_path),
        raw_output_dir=str(tmp_path / "output_raw"),
    )

    # --- 2. native: parquet + spinedb --------------------------------------
    write_outputs(
        **common,
        write_methods=["parquet", "spinedb"],
        results_db_url=native_db,
        solution=last_step.solution,
        solve_name=last_step.solve_name,
        solve_steps=[
            (s.solve_name, s.flex_data, s.effective_solution)
            for s in steps.values()
        ],
        flex_data_provider=last_step.flex_data_provider,
    )

    # --- 3. replay: spinedb from the processed parquet ---------------------
    write_outputs(
        **common,
        read_parquet_dir=True,
        write_methods=["spinedb"],
        results_db_url=replay_db,
    )

    alternative = SCENARIO  # derive_alternative_name == scenario_name

    native_vals = _dump_param_values(native_db, alternative)
    replay_vals = _dump_param_values(replay_db, alternative)

    assert native_vals, "native DB wrote no parameter values"
    assert replay_vals, "replay DB wrote no parameter values"

    # (b) discount params present natively only when par carries them; on the
    # replay path they must be ABSENT regardless.
    replay_discount = {
        k for k in replay_vals if k[2] in _DISCOUNT_PARAMS
    }
    assert not replay_discount, (
        f"replay DB unexpectedly contains discount params: {replay_discount}"
    )

    # (a) every NON-discount native param value must match the replay value
    # exactly (same keys, same solve/period/time Map structure & numbers).
    native_non_discount = {
        k: v for k, v in native_vals.items() if k[2] not in _DISCOUNT_PARAMS
    }
    assert native_non_discount, "native run produced no non-discount params"

    missing = sorted(set(native_non_discount) - set(replay_vals))
    assert not missing, (
        f"{len(missing)} non-discount params present natively but missing "
        f"in replay: {missing[:10]}"
    )

    extra = sorted(set(replay_vals) - set(native_vals))
    assert not extra, (
        f"{len(extra)} params present in replay but not native: {extra[:10]}"
    )

    mismatches = []
    for key, native_v in native_non_discount.items():
        if replay_vals[key] != native_v:
            mismatches.append(key)
    assert not mismatches, (
        f"{len(mismatches)} non-discount param VALUES differ between native "
        f"and replay: {mismatches[:10]}"
    )

    # (c) connection-triple node ordering: the connection__node__node
    # entities (which encode (connection, source, sink)) must be identical.
    native_conn = {
        (ec, byname) for ec, byname in _entities(native_db)
        if ec == "connection__node__node"
    }
    replay_conn = {
        (ec, byname) for ec, byname in _entities(replay_db)
        if ec == "connection__node__node"
    }
    assert native_conn, "no connection__node__node entities in native DB"
    assert native_conn == replay_conn, (
        "connection (source, sink) node ordering differs between native and "
        f"replay.\n native-only: {sorted(native_conn - replay_conn)[:10]}\n "
        f"replay-only: {sorted(replay_conn - native_conn)[:10]}"
    )


def test_spinedb_parquet_replay_matches_native_multisolve(
    test_db_url: str,
    test_solver_config_dir: Path,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """MULTI-SOLVE round-trip: validate that the per-solve ``solve`` Map keys,
    the per-period last-solve winner, and the per-(period,time) ``keep='last'``
    dedup are reconstructed identically to a native solve.

    The single-solve test above exercises the shim with zero overlapping
    (period,time), so the rolling/dedup/winner logic is never validated there;
    this test drives a nested multi-solve cascade (invest sequence + rolling
    dispatch) whose realized timeline DOES overlap across rolls, then asserts:

      (a) every non-discount parameter value matches native byte-for-byte,
          INCLUDING the full Map key structure — i.e. the per-solve ``solve``
          keys, which only appear once the run is genuinely multi-solve;
      (b) discount params (#33/#34) are absent on the replay path;
      (c) for the BIDIRECTIONAL connection the replay path emits a 2-way
          warning naming it (its (source, sink) order is unrecoverable from
          processed parquet); the 1-way exact-ordering case is covered by the
          single-solve test.
    """
    # --- 1. run the (multi-solve) cascade ----------------------------------
    steps = run_chain_from_db(
        input_db_url=test_db_url,
        scenario_name=MULTISOLVE_SCENARIO,
        work_folder=tmp_path,
        solver_config_dir=test_solver_config_dir,
        warm=True,
        keep_solutions=True,
    )
    assert steps, f"no steps for scenario {MULTISOLVE_SCENARIO!r}"
    # This MUST be genuinely multi-solve or the test is vacuous.
    assert len(steps) > 1, (
        f"{MULTISOLVE_SCENARIO!r} ran as a single solve "
        f"({len(steps)} step) — multi-solve logic would not be exercised"
    )
    last_step = next(reversed(steps.values()))
    assert last_step.solution is not None and last_step.solution.optimal

    native_db = "sqlite:///" + str(tmp_path / "results_native.sqlite")
    replay_db = "sqlite:///" + str(tmp_path / "results_replay.sqlite")

    common = dict(
        scenario_name=MULTISOLVE_SCENARIO,
        output_location=str(tmp_path),
        subdir=MULTISOLVE_SCENARIO,
        output_config_path=OUTPUT_CONFIG,
        fallback_output_location=str(tmp_path),
        raw_output_dir=str(tmp_path / "output_raw"),
    )

    # --- 2. native: parquet + spinedb --------------------------------------
    write_outputs(
        **common,
        write_methods=["parquet", "spinedb"],
        results_db_url=native_db,
        solution=last_step.solution,
        solve_name=last_step.solve_name,
        solve_steps=[
            (s.solve_name, s.flex_data, s.effective_solution)
            for s in steps.values()
        ],
        flex_data_provider=last_step.flex_data_provider,
    )

    # --- 3. replay: spinedb from the processed parquet (capture warnings) ---
    with caplog.at_level(
        logging.WARNING, logger="flextool.process_outputs.spinedb_replay"
    ):
        write_outputs(
            **common,
            read_parquet_dir=True,
            write_methods=["spinedb"],
            results_db_url=replay_db,
        )

    alternative = MULTISOLVE_SCENARIO  # derive_alternative_name == scenario

    native_vals = _dump_param_values(native_db, alternative)
    replay_vals = _dump_param_values(replay_db, alternative)
    assert native_vals, "native DB wrote no parameter values"
    assert replay_vals, "replay DB wrote no parameter values"

    # Sanity: the run is multi-solve, so at least one rendered Map must carry
    # more than one distinct ``solve`` key path.  We detect this by finding a
    # _dt param whose value Map nests period->time and confirming that across
    # all params at least two distinct solve labels were emitted natively.
    native_solve_labels = _collect_solve_labels(native_db, alternative)
    replay_solve_labels = _collect_solve_labels(replay_db, alternative)
    assert len(native_solve_labels) >= 2, (
        "native multi-solve run emitted <2 distinct solve labels; the "
        f"per-solve Map keys would be untested. labels={native_solve_labels}"
    )
    assert native_solve_labels == replay_solve_labels, (
        "per-solve Map labels differ between native and replay.\n"
        f" native-only: {sorted(native_solve_labels - replay_solve_labels)}\n"
        f" replay-only: {sorted(replay_solve_labels - native_solve_labels)}"
    )

    # (b) discount params absent on the replay path.
    replay_discount = {k for k in replay_vals if k[2] in _DISCOUNT_PARAMS}
    assert not replay_discount, (
        f"replay DB unexpectedly contains discount params: {replay_discount}"
    )

    # (a) every non-discount native param value matches replay exactly — same
    # keys, same per-solve/period/time Map structure & numbers.  The 2-way
    # connection's triple-ordering is the one documented exception: its
    # ``connection__node__node`` byname (source/sink) may be swapped, so we
    # exclude only that entity class from the value diff and check it
    # separately via the warning below.
    def _is_two_way_conn(key) -> bool:
        ec, byname, _param = key
        return ec == "connection__node__node" and (
            _MULTISOLVE_TWO_WAY_CONN in tuple(byname)
        )

    native_cmp = {
        k: v for k, v in native_vals.items()
        if k[2] not in _DISCOUNT_PARAMS and not _is_two_way_conn(k)
    }
    replay_cmp = {
        k: v for k, v in replay_vals.items()
        if k[2] not in _DISCOUNT_PARAMS and not _is_two_way_conn(k)
    }
    assert native_cmp, "native run produced no comparable params"

    missing = sorted(set(native_cmp) - set(replay_cmp))
    assert not missing, (
        f"{len(missing)} non-discount params present natively but missing "
        f"in replay: {missing[:10]}"
    )
    extra = sorted(set(replay_cmp) - set(native_cmp))
    assert not extra, (
        f"{len(extra)} params present in replay but not native: {extra[:10]}"
    )
    mismatches = [
        k for k, v in native_cmp.items() if replay_cmp[k] != v
    ]
    assert not mismatches, (
        f"{len(mismatches)} non-discount param VALUES (incl. per-solve Map "
        f"keys) differ between native and replay: {mismatches[:10]}"
    )

    # (c) the bidirectional connection must trigger the 2-way replay warning,
    # naming it.  (For a 1-way connection no such warning fires — verified by
    # the single-solve test, whose connection is unidirectional and exact.)
    two_way_warnings = [
        r.getMessage() for r in caplog.records
        if r.levelno >= logging.WARNING
        and "bidirectional" in r.getMessage()
    ]
    assert two_way_warnings, (
        "expected a 2-way connection warning on the replay path for "
        f"{_MULTISOLVE_TWO_WAY_CONN!r}, got none"
    )
    assert any(
        _MULTISOLVE_TWO_WAY_CONN in m for m in two_way_warnings
    ), (
        f"2-way warning did not name {_MULTISOLVE_TWO_WAY_CONN!r}: "
        f"{two_way_warnings}"
    )

    # The 2-way connection IS present in both DBs (only its node ORDER may
    # differ) — confirm the connection itself round-tripped, byname order
    # aside.
    def _two_way_conns(db_url):
        return {
            tuple(byname)[0]
            for ec, byname in _entities(db_url)
            if ec == "connection__node__node"
            and _MULTISOLVE_TWO_WAY_CONN in tuple(byname)
        }
    assert _two_way_conns(native_db) == _two_way_conns(replay_db) == {
        _MULTISOLVE_TWO_WAY_CONN
    }


# Parameters whose OUTERMOST Map key is the reconstructed ``solve`` axis
# (``Map[solve]->period[->time]``) — these are the per-row solve-tagged
# timestep frames, so their outer keys are exactly the solve labels.  Cost /
# category params (e.g. ``balance_t``) are excluded because their outer keys
# are categories, not solves.
_SOLVE_AXIS_PARAMS = {
    ("node", "price_t"),
    ("node", "state_t"),
    ("unit__node", "flow_t"),
}


def _collect_solve_labels(db_url: str, alternative: str) -> set:
    """Return the set of distinct ``solve`` Map labels emitted for the
    solve-axis parameters in ``alternative``.

    The solve-axis ``_dt`` frames wrap values under ``Map[solve]->period->
    time``, so the outermost Map keys of :data:`_SOLVE_AXIS_PARAMS` are exactly
    the per-solve labels.  A genuinely multi-solve run yields >= 2 distinct
    labels; native and replay must agree on the exact set (the precise
    per-solve dedup / last-solve-winner check)."""
    labels: set = set()
    with DatabaseMapping(db_url) as db:
        for pv in db.get_parameter_value_items(alternative_name=alternative):
            if (pv["entity_class_name"],
                    pv["parameter_definition_name"]) not in _SOLVE_AXIS_PARAMS:
                continue
            value = from_database(pv["value"], pv["type"])
            indexes = getattr(value, "indexes", None)
            if indexes is None:
                continue
            for idx in indexes:
                labels.add(str(idx))
    return labels


if __name__ == "__main__":
    pytest.main([__file__, "-q"])
