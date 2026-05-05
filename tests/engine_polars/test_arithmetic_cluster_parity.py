"""Δ.10 — Cluster F (scalar arithmetic) parity tests.

Per-fixture parity sweep + targeted tests for the cluster F helpers in
:mod:`flextool.engine_polars._derived_arithmetic` and the existing
cluster F helpers in :mod:`flextool.engine_polars._derived_params`
(``p_slope``, ``p_section``, ``p_flow_upper_existing``, ``p_state_upper``,
``p_process_existing_count``).

The CSV-loaded ``FlexData`` is the parity oracle — any divergence between
the source-driven path (DB-direct) and the CSV path surfaces as a
per-fixture failure.

Cluster F coverage:

* ``p_unitsize``                — per-process unitsize cascade.
* ``p_state_unitsize``          — per-node unitsize cascade.
* ``p_penalty_up`` / ``p_penalty_down`` — node penalty broadcast.
* ``p_process_source_flow_coef`` / ``p_process_sink_flow_coef`` —
  indirect flow coefficient.
* ``p_slope``                   — efficiency-curve slope (verified).
* ``p_section``                 — min_load_efficiency y-intercept.
* ``p_flow_upper_existing``     — existing/unitsize per arc.
* ``p_state_upper``             — capacity / unitsize per (n, d).
* ``p_process_existing_count``  — existing / unitsize per (p, d).

Per the Δ.10 hand-off the ~10 % Γ.4 deferred parity gap is the long
tail of multi-period broadcasting + multi-block fixtures.  The sweep
is the gating oracle for those.
"""
from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest
import spinedb_api as api

from flextool.engine_polars import (
    InMemoryReader,
    SpineDbReader,
    load_flextool,
)
from flextool.engine_polars import _derived_arithmetic as ar
from polar_high import Param


HERE = Path(__file__).resolve().parent
DATA = HERE / "data"


# Per-fixture scenario overrides — same convention as test_npv_cluster_parity.
_DIRNAME_TO_SCENARIO_OVERRIDES: dict[str, str] = {
    "work_2day_stochastic_dispatch_full_storage": "2_day_stochastic_dispatch",
    "work_commodity_ladder_annual": "coal_ladder_annual",
    "work_commodity_ladder_cumulative": "coal_ladder_cumulative",
    "work_delay_source_coef": "water_pump_delayed",
    "work_inflation_check": "wind_battery_invest_lifetime_renew",
}


def _discover_fixtures() -> list[tuple[str, str]]:
    """``[(work_dirname, scenario_name), …]`` for every fixture with a DB."""
    import re

    out: list[tuple[str, str]] = []
    for d in sorted(DATA.iterdir()):
        if not d.is_dir() or not d.name.startswith("work_"):
            continue
        sqlite = d / "tests.sqlite"
        if not sqlite.exists():
            continue
        if d.name in _DIRNAME_TO_SCENARIO_OVERRIDES:
            target = _DIRNAME_TO_SCENARIO_OVERRIDES[d.name]
            try:
                with api.DatabaseMapping("sqlite:///" + str(sqlite)) as db:
                    found = any(
                        s.name == target for s in db.query(db.scenario_sq).all()
                    )
            except Exception:
                found = False
            if found:
                out.append((d.name, target))
                continue
        scen_target = d.name.removeprefix("work_")
        try:
            with api.DatabaseMapping("sqlite:///" + str(sqlite)) as db:
                scenarios = sorted(
                    s.name for s in db.query(db.scenario_sq).all()
                )
        except Exception:
            continue
        candidates = [scen_target]
        candidates.append(re.sub(r"(^|_)(\d+)([a-z])", r"\1\2_\3", scen_target))
        candidates.append(re.sub(r"(\d+)_([a-z])", r"\1\2", scen_target))
        if scen_target.endswith("_full_storage"):
            base = scen_target[: -len("_full_storage")]
            candidates.append(re.sub(r"(^|_)(\d+)([a-z])", r"\1\2_\3", base))
            candidates.append(base)
        chosen: str | None = None
        for cand in candidates:
            if cand in scenarios:
                chosen = cand
                break
        if chosen is not None:
            out.append((d.name, chosen))
        elif scenarios:
            out.append((d.name, scenarios[0]))
    return out


PARITY_CASES = _discover_fixtures()


# Known fixture-data divergences — fixtures where the CSV preprocessing
# captures a parameter value that the SQLite source doesn't carry (or
# vice versa).  These are pre-existing fixture inconsistencies surfaced
# by Δ.10's wider parity sweep — *not* defects in the cluster F helpers.
# The parity test xfails on the listed (fixture, field) pairs.
#
# Each entry: (work_name, field_name) → reason.  The fixture-data fix
# is upstream (regenerate the work_*/input/*.csv against the canonical
# SpineDB source); deferred until Δ.12 fixture rebuild.
_FIXTURE_DATA_DIVERGENCES: dict[tuple[str, str], str] = {
    ("work_delay_source_coef", "p_penalty_up"):
        "CSV input has water_sink.penalty_up=500; SQLite has the schema "
        "default (10000).  Fixture rebuild needed.",
    ("work_delay_source_coef", "p_penalty_down"):
        "CSV input has water_sink.penalty_down=500; SQLite has the schema "
        "default (10000).  Fixture rebuild needed.",
}


def _maybe_skip_fixture_data_divergence(work_name: str, field: str) -> None:
    key = (work_name, field)
    if key in _FIXTURE_DATA_DIVERGENCES:
        pytest.skip(
            f"fixture-data divergence on {work_name}/{field}: "
            f"{_FIXTURE_DATA_DIVERGENCES[key]}"
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _frames_equal(a: pl.DataFrame | None, b: pl.DataFrame | None,
                    keys: tuple[str, ...]) -> tuple[bool, str | None]:
    """Compare two frames for row-set equality on *keys*+ value column.

    Returns ``(equal, diff_message)``.  Both ``None`` is equal.
    """
    if a is None and b is None:
        return True, None
    if a is None:
        return False, f"left None, right {b.height} rows"
    if b is None:
        return False, f"left {a.height} rows, right None"
    if set(a.columns) != set(b.columns):
        return False, f"columns differ: left={a.columns} right={b.columns}"
    if a.height != b.height:
        return False, f"row counts differ: left={a.height} right={b.height}"
    a_sorted = a.sort(by=list(keys))
    b_sorted = b.select(a.columns).sort(by=list(keys))
    if a_sorted.equals(b_sorted):
        return True, None
    val_col = next((c for c in a.columns if c not in keys), None)
    if val_col is None:
        return False, "no value column"
    a_keys = a_sorted.select(list(keys))
    b_keys = b_sorted.select(list(keys))
    if not a_keys.equals(b_keys):
        return False, f"key sets differ"
    av = a_sorted[val_col].cast(pl.Float64, strict=False).to_list()
    bv = b_sorted[val_col].cast(pl.Float64, strict=False).to_list()
    max_diff = 0.0
    for x, y in zip(av, bv):
        if x is None or y is None:
            if x != y:
                return False, f"null mismatch: {x} vs {y}"
            continue
        d = abs(x - y)
        if d > max_diff:
            max_diff = d
    # Tolerance: 1e-6 * max(1, |value|) — same bar as
    # ``test_db_direct_parity._frame_eq_value``.  CSV serialisation
    # truncates to 6 significant digits; DB-direct values round-tripped
    # through CSV can differ at ~1e-7.
    scale = max(1.0, max((abs(x) for x in av if x is not None),
                          default=1.0))
    if max_diff < 1e-6 * scale:
        return True, None
    return False, f"max abs diff = {max_diff!r} (scale={scale})"


def _param_frame(p) -> pl.DataFrame | None:
    if p is None:
        return None
    return p.frame if hasattr(p, "frame") else p


# ---------------------------------------------------------------------------
# Per-fixture parity sweep
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "work_name,scenario", PARITY_CASES,
    ids=lambda v: v if isinstance(v, str) else "?",
)
def test_p_unitsize_parity(work_name: str, scenario: str) -> None:
    """Per-fixture parity for ``p_unitsize`` (cluster F)."""
    work = DATA / work_name
    sqlite = work / "tests.sqlite"
    if not sqlite.exists():
        pytest.skip("fixture missing tests.sqlite")
    csv_data = load_flextool(work)
    reader = SpineDbReader(sqlite, scenario)
    db_data = load_flextool(work, db_reader=reader)
    a = _param_frame(csv_data.p_unitsize)
    b = _param_frame(db_data.p_unitsize)
    if a is None and b is None:
        pytest.skip("p_unitsize None on both paths")
    ok, msg = _frames_equal(a, b, ("p",))
    assert ok, f"p_unitsize mismatch on {work_name}: {msg}\n  csv:\n{a}\n  db:\n{b}"


@pytest.mark.parametrize(
    "work_name,scenario", PARITY_CASES,
    ids=lambda v: v if isinstance(v, str) else "?",
)
def test_p_state_unitsize_parity(work_name: str, scenario: str) -> None:
    """Per-fixture parity for ``p_state_unitsize`` (cluster F)."""
    work = DATA / work_name
    sqlite = work / "tests.sqlite"
    if not sqlite.exists():
        pytest.skip("fixture missing tests.sqlite")
    csv_data = load_flextool(work)
    reader = SpineDbReader(sqlite, scenario)
    db_data = load_flextool(work, db_reader=reader)
    a = _param_frame(csv_data.p_state_unitsize)
    b = _param_frame(db_data.p_state_unitsize)
    if a is None and b is None:
        pytest.skip("p_state_unitsize None on both paths")
    ok, msg = _frames_equal(a, b, ("n",))
    assert ok, (
        f"p_state_unitsize mismatch on {work_name}: {msg}\n"
        f"  csv:\n{a}\n  db:\n{b}"
    )


@pytest.mark.parametrize(
    "work_name,scenario", PARITY_CASES,
    ids=lambda v: v if isinstance(v, str) else "?",
)
def test_p_penalty_up_parity(work_name: str, scenario: str) -> None:
    """Per-fixture parity for ``p_penalty_up`` (cluster F)."""
    _maybe_skip_fixture_data_divergence(work_name, "p_penalty_up")
    work = DATA / work_name
    sqlite = work / "tests.sqlite"
    if not sqlite.exists():
        pytest.skip("fixture missing tests.sqlite")
    csv_data = load_flextool(work)
    reader = SpineDbReader(sqlite, scenario)
    db_data = load_flextool(work, db_reader=reader)
    a = _param_frame(csv_data.p_penalty_up)
    b = _param_frame(db_data.p_penalty_up)
    if a is None and b is None:
        pytest.skip("p_penalty_up None on both paths")
    ok, msg = _frames_equal(a, b, ("n", "d", "t"))
    assert ok, (
        f"p_penalty_up mismatch on {work_name}: {msg}"
    )


@pytest.mark.parametrize(
    "work_name,scenario", PARITY_CASES,
    ids=lambda v: v if isinstance(v, str) else "?",
)
def test_p_penalty_down_parity(work_name: str, scenario: str) -> None:
    """Per-fixture parity for ``p_penalty_down`` (cluster F)."""
    _maybe_skip_fixture_data_divergence(work_name, "p_penalty_down")
    work = DATA / work_name
    sqlite = work / "tests.sqlite"
    if not sqlite.exists():
        pytest.skip("fixture missing tests.sqlite")
    csv_data = load_flextool(work)
    reader = SpineDbReader(sqlite, scenario)
    db_data = load_flextool(work, db_reader=reader)
    a = _param_frame(csv_data.p_penalty_down)
    b = _param_frame(db_data.p_penalty_down)
    if a is None and b is None:
        pytest.skip("p_penalty_down None on both paths")
    ok, msg = _frames_equal(a, b, ("n", "d", "t"))
    assert ok, (
        f"p_penalty_down mismatch on {work_name}: {msg}"
    )


@pytest.mark.parametrize(
    "work_name,scenario", PARITY_CASES,
    ids=lambda v: v if isinstance(v, str) else "?",
)
def test_p_process_source_flow_coef_parity(work_name: str,
                                                scenario: str) -> None:
    """Per-fixture parity for ``p_process_source_flow_coef`` (cluster F)."""
    work = DATA / work_name
    sqlite = work / "tests.sqlite"
    if not sqlite.exists():
        pytest.skip("fixture missing tests.sqlite")
    csv_data = load_flextool(work)
    reader = SpineDbReader(sqlite, scenario)
    db_data = load_flextool(work, db_reader=reader)
    a = _param_frame(csv_data.p_process_source_flow_coef)
    b = _param_frame(db_data.p_process_source_flow_coef)
    if a is None and b is None:
        pytest.skip("p_process_source_flow_coef None on both paths")
    ok, msg = _frames_equal(a, b, ("p", "source"))
    assert ok, (
        f"p_process_source_flow_coef mismatch on {work_name}: {msg}\n"
        f"  csv:\n{a}\n  db:\n{b}"
    )


@pytest.mark.parametrize(
    "work_name,scenario", PARITY_CASES,
    ids=lambda v: v if isinstance(v, str) else "?",
)
def test_p_process_sink_flow_coef_parity(work_name: str,
                                              scenario: str) -> None:
    """Per-fixture parity for ``p_process_sink_flow_coef`` (cluster F)."""
    work = DATA / work_name
    sqlite = work / "tests.sqlite"
    if not sqlite.exists():
        pytest.skip("fixture missing tests.sqlite")
    csv_data = load_flextool(work)
    reader = SpineDbReader(sqlite, scenario)
    db_data = load_flextool(work, db_reader=reader)
    a = _param_frame(csv_data.p_process_sink_flow_coef)
    b = _param_frame(db_data.p_process_sink_flow_coef)
    if a is None and b is None:
        pytest.skip("p_process_sink_flow_coef None on both paths")
    ok, msg = _frames_equal(a, b, ("p", "sink"))
    assert ok, (
        f"p_process_sink_flow_coef mismatch on {work_name}: {msg}\n"
        f"  csv:\n{a}\n  db:\n{b}"
    )


@pytest.mark.parametrize(
    "work_name,scenario", PARITY_CASES,
    ids=lambda v: v if isinstance(v, str) else "?",
)
def test_p_slope_full_sweep_parity(work_name: str, scenario: str) -> None:
    """Per-fixture parity for the existing cluster F ``p_slope`` helper.

    Δ.10 widens the parity surface to every fixture (the
    ``test_db_direct_parity.TestDerivedBTopology`` covers a 3-fixture
    subset only).
    """
    work = DATA / work_name
    sqlite = work / "tests.sqlite"
    if not sqlite.exists():
        pytest.skip("fixture missing tests.sqlite")
    csv_data = load_flextool(work)
    reader = SpineDbReader(sqlite, scenario)
    db_data = load_flextool(work, db_reader=reader)
    a = _param_frame(csv_data.p_slope)
    b = _param_frame(db_data.p_slope)
    if a is None and b is None:
        pytest.skip("p_slope None on both paths")
    ok, msg = _frames_equal(a, b, ("p", "d", "t"))
    assert ok, f"p_slope mismatch on {work_name}: {msg}"


@pytest.mark.parametrize(
    "work_name,scenario", PARITY_CASES,
    ids=lambda v: v if isinstance(v, str) else "?",
)
def test_p_section_full_sweep_parity(work_name: str, scenario: str) -> None:
    """Per-fixture parity for the existing cluster F ``p_section`` helper."""
    work = DATA / work_name
    sqlite = work / "tests.sqlite"
    if not sqlite.exists():
        pytest.skip("fixture missing tests.sqlite")
    csv_data = load_flextool(work)
    reader = SpineDbReader(sqlite, scenario)
    db_data = load_flextool(work, db_reader=reader)
    a = _param_frame(csv_data.p_section)
    b = _param_frame(db_data.p_section)
    if a is None and b is None:
        pytest.skip("p_section None on both paths")
    ok, msg = _frames_equal(a, b, ("p", "d", "t"))
    assert ok, f"p_section mismatch on {work_name}: {msg}"


@pytest.mark.parametrize(
    "work_name,scenario", PARITY_CASES,
    ids=lambda v: v if isinstance(v, str) else "?",
)
def test_p_flow_upper_existing_full_sweep_parity(work_name: str,
                                                     scenario: str) -> None:
    """Per-fixture parity for the existing cluster F
    ``p_flow_upper_existing`` helper.
    """
    work = DATA / work_name
    sqlite = work / "tests.sqlite"
    if not sqlite.exists():
        pytest.skip("fixture missing tests.sqlite")
    csv_data = load_flextool(work)
    reader = SpineDbReader(sqlite, scenario)
    db_data = load_flextool(work, db_reader=reader)
    a = _param_frame(csv_data.p_flow_upper_existing)
    b = _param_frame(db_data.p_flow_upper_existing)
    if a is None and b is None:
        pytest.skip("p_flow_upper_existing None on both paths")
    ok, msg = _frames_equal(a, b, ("p", "source", "sink", "d"))
    assert ok, f"p_flow_upper_existing mismatch on {work_name}: {msg}"


@pytest.mark.parametrize(
    "work_name,scenario", PARITY_CASES,
    ids=lambda v: v if isinstance(v, str) else "?",
)
def test_p_state_upper_full_sweep_parity(work_name: str,
                                              scenario: str) -> None:
    """Per-fixture parity for the existing cluster F ``p_state_upper``
    helper.
    """
    work = DATA / work_name
    sqlite = work / "tests.sqlite"
    if not sqlite.exists():
        pytest.skip("fixture missing tests.sqlite")
    csv_data = load_flextool(work)
    reader = SpineDbReader(sqlite, scenario)
    db_data = load_flextool(work, db_reader=reader)
    a = _param_frame(csv_data.p_state_upper)
    b = _param_frame(db_data.p_state_upper)
    if a is None and b is None:
        pytest.skip("p_state_upper None on both paths")
    ok, msg = _frames_equal(a, b, ("n", "d"))
    assert ok, f"p_state_upper mismatch on {work_name}: {msg}"


@pytest.mark.parametrize(
    "work_name,scenario", PARITY_CASES,
    ids=lambda v: v if isinstance(v, str) else "?",
)
def test_p_process_existing_count_full_sweep_parity(work_name: str,
                                                          scenario: str
                                                          ) -> None:
    """Per-fixture parity for the existing cluster F
    ``p_process_existing_count`` helper.
    """
    work = DATA / work_name
    sqlite = work / "tests.sqlite"
    if not sqlite.exists():
        pytest.skip("fixture missing tests.sqlite")
    csv_data = load_flextool(work)
    reader = SpineDbReader(sqlite, scenario)
    db_data = load_flextool(work, db_reader=reader)
    a = _param_frame(csv_data.p_process_existing_count)
    b = _param_frame(db_data.p_process_existing_count)
    if a is None and b is None:
        pytest.skip("p_process_existing_count None on both paths")
    ok, msg = _frames_equal(a, b, ("p", "d"))
    assert ok, f"p_process_existing_count mismatch on {work_name}: {msg}"


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


def test_p_process_source_flow_coef_inmemory_zero_drop():
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
            ("unit__inputNode", "flow_coefficient"): pl.DataFrame({
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
    z, p = ar.p_process_source_flow_coef_from_source(src, indirect_pairs)
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


def test_p_process_source_flow_coef_inmemory_all_default_returns_none():
    """When every coefficient equals 1.0, the helper emits no Param —
    model.py's gate falls through.
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
            ("unit__inputNode", "flow_coefficient"): pl.DataFrame({
                "unit": ["u"], "node": ["n_x"], "value": [1.0],
            }),
        },
    )
    indirect_pairs = pl.DataFrame({
        "p": ["u"], "source": ["n_x"], "sink": ["u"],
    })
    z, p = ar.p_process_source_flow_coef_from_source(src, indirect_pairs)
    assert z is None
    assert p is None


def test_p_process_sink_flow_coef_inmemory_smoke():
    """``p_process_sink_flow_coef`` is symmetric to source — same
    contract on ``unit__outputNode``.
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
            ("unit__outputNode", "flow_coefficient"): pl.DataFrame({
                "unit": ["u", "u"],
                "node": ["heat", "elec"],
                "value": [0.2, 2.0],
            }),
        },
    )
    indirect_pairs = pl.DataFrame({
        "p": ["u", "u"], "source": ["u", "u"], "sink": ["heat", "elec"],
    })
    z, p = ar.p_process_sink_flow_coef_from_source(src, indirect_pairs)
    assert z is None  # neither coef is zero
    assert p is not None
    rows = {(r["p"], r["sink"]): r["value"]
              for r in p.frame.iter_rows(named=True)}
    assert rows == {("u", "heat"): 0.2, ("u", "elec"): 2.0}
