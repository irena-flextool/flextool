"""Round-trip GO test for a Map-valued ``energy_margin_adder``.

Proves the full DB → ingestion → emit path for a period-time (2d) and a
period (1d) Map ``energy_margin_adder``, WITHOUT a HiGHS solve.  This is
the fast, airtight proof of the fix: before it, a Map written to the DB
was silently mangled at ingestion (the scalar spec's 2-col header
truncated the flattened map row in ``_rows_to_frame``, dropping the time
axis and value) so the adder did nothing.

The fix has two halves, both exercised here against a REAL
``SpineDBBackend`` reading a REAL SQLite built from the JSON fixture:

1. **Ingestion (``_specs.py``).**  The scalar spec now carries
   ``filter_in_type=["float"]`` so it stops eating maps; two new specs
   (``filter_in_type=["1d_map"]`` / ``["2d_map"]``) route the Map shapes
   into their own ``pd_``/``pdt_`` frames with the value + index axes
   intact.  We call ``SpineDBBackend.parameter_values`` for all three
   specs and assert the map lands in the ``pdt_``/``pd_`` frame as
   ``[node, period, time, value]`` / ``[node, period, value]`` and is
   ABSENT from the scalar frame (no longer mangled).

2. **Emit (``_emit_energy_margin_adder.py``).**  The emitter now reads all
   three authored files and unions them.  We feed the REAL ingested frame
   into a ``FlexDataProvider`` and run ``emit_energy_margin_adder`` over an
   invest ``(d, t)`` grid, asserting the demand at the map's authored
   cells is DEEPENED by exactly the map value (sign: demand is negative
   inflow, so adding demand SUBTRACTS), and cells the map does NOT author
   are byte-identical.

The DB is built from the JSON fixture under a tmp dir (never a checked-in
``.sqlite``; CLAUDE.md invariant #3); the Map is set via the spinedb
``import_data`` API (no JSON edit).
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import polars as pl

_TESTS_DIR = Path(__file__).resolve().parent.parent
if str(_TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(_TESTS_DIR))

from db_utils import json_to_db  # noqa: E402

from flextool.engine_polars._emit_energy_margin_adder import (  # noqa: E402
    emit_energy_margin_adder,
)
from flextool.engine_polars._flex_data_provider import (  # noqa: E402
    FlexDataProvider,
)
from flextool.spinedb_backend import SpineDBBackend  # noqa: E402

FIXTURE_JSON = _TESTS_DIR / "fixtures" / "tests.json"
NODE = "west"  # a real node present in the fixture

# The 2d map: period -> time -> MWh.  LARGE only at (p1, t2) and (p2, t1);
# zero at the other two authored cells; UNAUTHORED elsewhere.
MAP_2D = {
    ("p1", "t1"): 0.0,
    ("p1", "t2"): 9000.0,
    ("p2", "t1"): 7000.0,
    ("p2", "t2"): 0.0,
}
# The 1d map: period -> MWh.
MAP_1D = {"p1": 1500.0, "p2": 2500.0}


# ---------------------------------------------------------------------------
# DB builders (build from JSON; set the Map via import_data — no JSON edit)
# ---------------------------------------------------------------------------

def _build_db_with_map(tmp_path: Path, *, dim: int) -> str:
    """Fresh SQLite from the JSON fixture with a ``dim``-d Map
    ``energy_margin_adder`` (+ ``inflow_adder`` method) set on ``NODE``."""
    from spinedb_api import DatabaseMapping, Map, import_data

    from flextool.update_flextool.db_migration import migrate_database

    url = json_to_db(FIXTURE_JSON, tmp_path / f"map{dim}d.sqlite")
    migrate_database(url)

    if dim == 2:
        periods = ["p1", "p2"]
        times = ["t1", "t2"]
        inner = [
            Map(times, [MAP_2D[(p, t)] for t in times], index_name="time")
            for p in periods
        ]
        value = Map(periods, inner, index_name="period")
    elif dim == 1:
        periods = list(MAP_1D)
        value = Map(periods, [MAP_1D[p] for p in periods], index_name="period")
    else:  # pragma: no cover - guard
        raise ValueError(dim)

    with DatabaseMapping(url) as db:
        _count, errors = import_data(
            db,
            parameter_values=[
                ("node", NODE, "energy_margin_adder", value),
                ("node", NODE, "energy_margin_method", "inflow_adder"),
            ],
        )
        assert not errors, f"import_data errors: {errors}"
        db.commit_session(f"energy_margin_adder {dim}d map")
    return url


def _ingest(url: str) -> dict[str, pl.DataFrame]:
    """Run the REAL three energy_margin_adder specs through
    ``SpineDBBackend.parameter_values`` (the DB→frame ingestion path that
    ``input_derivation`` drives) and return the three frames by shape."""
    backend = SpineDBBackend(url)
    try:
        scalar = backend.parameter_values(
            cl_pars=[("node", "energy_margin_adder")],
            header="node,energy_margin_adder",
            filter_in_type=["float"],
        )
        pd_map = backend.parameter_values(
            cl_pars=[("node", "energy_margin_adder")],
            header="node,period,energy_margin_adder",
            filter_in_type=["1d_map"],
        )
        pdt_map = backend.parameter_values(
            cl_pars=[("node", "energy_margin_adder")],
            header="node,period,time,energy_margin_adder",
            filter_in_type=["2d_map"],
        )
    finally:
        backend.close()
    return {"scalar": scalar, "pd": pd_map, "pdt": pdt_map}


# ---------------------------------------------------------------------------
# Emit driver — feed a REAL ingested frame through the emitter
# ---------------------------------------------------------------------------

def _utf8(cols: dict[str, list[str]]) -> pl.DataFrame:
    return pl.DataFrame(cols, schema={c: pl.Utf8 for c in cols})


def _emit_over_grid(
    ingested: dict[str, pl.DataFrame],
    *,
    grid: list[tuple[str, str]],
    existing_inflow: pl.DataFrame,
) -> pl.DataFrame:
    """Register the ingested frames under their canonical Provider keys
    (exactly the ``input/<stem>`` keys ``input_derivation`` uses) and run
    the invest-solve emitter over ``grid``; return the emitted
    ``pdtNodeInflow`` frame."""
    prov = FlexDataProvider()
    prov.put("solve_data/pdtNodeInflow", existing_inflow)
    prov.put(
        "solve_data/steps_in_use",
        _utf8({"period": [d for d, _t in grid], "time": [t for _d, t in grid]}),
    )
    prov.put(
        "input/node__energy_margin_method",
        _utf8({"node": [NODE], "energy_margin_method": ["inflow_adder"]}),
    )
    # The three ingested frames land under the SAME keys input_derivation
    # registers them under (_provider_key of each spec's filename).
    prov.put("input/energy_margin_adder", ingested["scalar"])
    prov.put("input/pd_energy_margin_adder", ingested["pd"])
    prov.put("input/pdt_energy_margin_adder", ingested["pdt"])

    state = SimpleNamespace(
        solve=SimpleNamespace(invest_periods={"S": [(d,) for d, _t in grid]}),
    )
    emit_energy_margin_adder(
        state, "S", Path("input"), Path("solve_data"), provider=prov,
    )
    return prov.get("solve_data/pdtNodeInflow")


# ---------------------------------------------------------------------------
# Test 1 — 2d (period-time) Map round-trips to the authored (d, t) cells
# ---------------------------------------------------------------------------

def test_2d_map_adder_reaches_authored_cells(tmp_path) -> None:
    url = _build_db_with_map(tmp_path, dim=2)
    ingested = _ingest(url)

    # --- Ingestion half: the map lands in the pdt_ frame intact and is
    #     ABSENT from the scalar frame (no longer mangled). ---
    pdt = ingested["pdt"]
    assert pdt.columns == ["node", "period", "time", "energy_margin_adder"]
    got = {
        (r["period"], r["time"]): float(r["energy_margin_adder"])
        for r in pdt.iter_rows(named=True)
        if r["node"] == NODE
    }
    assert got == MAP_2D, f"2d map mis-ingested: {got!r} != {MAP_2D!r}"
    assert ingested["scalar"].filter(pl.col("node") == NODE).height == 0, (
        "scalar spec ingested the map — the ['float'] filter is not gating"
    )
    assert ingested["pd"].filter(pl.col("node") == NODE).height == 0, (
        "1d spec matched a 2d map"
    )

    # --- Emit half: the invest grid covers the four AUTHORED cells plus an
    #     UNAUTHORED cell (p1, t3) the map never touches. ---
    grid = [("p1", "t1"), ("p1", "t2"), ("p1", "t3"),
            ("p2", "t1"), ("p2", "t2")]
    existing = _utf8({
        "node": [NODE] * len(grid),
        "period": [d for d, _t in grid],
        "time": [t for _d, t in grid],
        "value": ["-100.0", "-200.0", "-300.0", "-400.0", "-500.0"],
    })
    out = _emit_over_grid(ingested, grid=grid, existing_inflow=existing).sort(
        ["period", "time"],
    )
    vals = {
        (r["period"], r["time"]): float(r["value"])
        for r in out.iter_rows(named=True)
    }
    # Authored cells: value_new = value_old - map_value (demand deepened).
    assert vals[("p1", "t1")] == -100.0 - 0.0      # authored 0 → no-op
    assert vals[("p1", "t2")] == -200.0 - 9000.0   # -9200.0
    assert vals[("p2", "t1")] == -400.0 - 7000.0   # -7400.0
    assert vals[("p2", "t2")] == -500.0 - 0.0      # authored 0 → no-op
    # Unauthored cell: byte-identical (the map does not reach it).
    unauth = out.filter(
        (pl.col("period") == "p1") & (pl.col("time") == "t3"),
    )
    assert unauth["value"].to_list() == ["-300.0"], (
        "unauthored (p1, t3) cell was touched — the 2d map over-broadcast"
    )
    # The load-bearing proof: the LARGE authored value reached its cell.
    assert vals[("p1", "t2")] == -9200.0
    assert vals[("p2", "t1")] == -7400.0


# ---------------------------------------------------------------------------
# Test 2 — 1d (period) Map round-trips, broadcast across each period's times
# ---------------------------------------------------------------------------

def test_1d_map_adder_broadcasts_per_period(tmp_path) -> None:
    url = _build_db_with_map(tmp_path, dim=1)
    ingested = _ingest(url)

    pd_map = ingested["pd"]
    assert pd_map.columns == ["node", "period", "energy_margin_adder"]
    got = {
        r["period"]: float(r["energy_margin_adder"])
        for r in pd_map.iter_rows(named=True)
        if r["node"] == NODE
    }
    assert got == MAP_1D, f"1d map mis-ingested: {got!r} != {MAP_1D!r}"
    assert ingested["scalar"].filter(pl.col("node") == NODE).height == 0
    assert ingested["pdt"].filter(pl.col("node") == NODE).height == 0

    # Two times per period → the period value must deepen BOTH times.
    grid = [("p1", "t1"), ("p1", "t2"), ("p2", "t1"), ("p2", "t2")]
    existing = _utf8({
        "node": [NODE] * len(grid),
        "period": [d for d, _t in grid],
        "time": [t for _d, t in grid],
        "value": ["-10.0", "-20.0", "-30.0", "-40.0"],
    })
    out = _emit_over_grid(ingested, grid=grid, existing_inflow=existing).sort(
        ["period", "time"],
    )
    vals = out["value"].to_list()
    # p1 rows deepened by 1500, p2 rows by 2500 (every time in the period).
    assert vals == [
        repr(-10.0 - 1500.0), repr(-20.0 - 1500.0),   # p1: both times
        repr(-30.0 - 2500.0), repr(-40.0 - 2500.0),   # p2: both times
    ], f"1d period map did not broadcast per-period: {vals!r}"
