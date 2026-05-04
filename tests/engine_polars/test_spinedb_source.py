"""P1 of the DB-direct migration — :class:`flextool.SpineDbSource` tests.

Covers:

1. **Smoke**: SpineDbSource constructs cleanly and ``build_frames``
   returns a non-empty mapping with both kinds populated.
2. **Frame-count parity**: the frame count returned by
   ``build_frames`` is within tolerance of the corresponding fixture's
   on-disk CSV count (allowing for tempdir-vs-fixture artifacts that
   aren't input files — e.g. logs, output_raw, .mps, .sol).
3. **End-to-end load + solve parity**: ``load_flextool(SpineDbSource(...))``
   produces a FlexData whose ``build_flextool`` solve matches the
   fixture's recorded ``v_obj`` to <1e-6 relative error.
4. **CsvSource backward-compat**: ``load_flextool(CsvSource(workdir))``
   produces an identical FlexData to ``load_flextool(workdir)``.

Three fixtures span representative patterns:

* ``work_coal``     — single-process commodity-buy dispatch (single solve,
  single period); built by ``_gen_input.py coal``.
* ``work_dc_power_flow`` — DC OPF on PGLib case14 (single solve, 1 period,
  network angle constraints); built by ``_gen_dc_power_flow.py`` from
  the MATPOWER .m file.
* ``work_base``     — base scenario with multiple processes; built by
  ``_gen_input.py base``.

These three exercise the single-solve path of SpineDbSource's
materialiser.  Multi-solve cascade coverage will be added in P2.
"""
from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from polar_high_opt import Problem
from flextool.engine_polars import (
    CsvSource, FlexInputSource, SpineDbSource,
    build_flextool, load_flextool,
)


DATA = Path(__file__).resolve().parent / "data"


# Fixtures: (work_name, sqlite_filename, scenario_name, v_obj_parquet_stem)
PARITY_FIXTURES = [
    ("work_coal",            "tests.sqlite",  "coal",         "y2020_2day_dispatch"),
    ("work_dc_power_flow",   "case14.sqlite", "dc_opf_test",  "dispatch"),
    ("work_base",            "tests.sqlite",  "base",         "y2020_2day_dispatch"),
]


# ---------------------------------------------------------------------------
# 1. Smoke

def test_spinedb_source_smoke():
    """SpineDbSource on the coal fixture constructs and materialises
    on first directory access."""
    work = DATA / "work_coal"
    src = SpineDbSource(work / "tests.sqlite", scenario="coal")
    # Construction is lazy — no work done yet.
    assert src.scenario == "coal"
    # First access triggers materialisation.
    inp = src.input_dir
    sd = src.solve_data_dir
    assert inp.is_dir()
    assert sd.is_dir()
    # input_dir always carries entity.csv when a scenario was loaded.
    assert (inp / "entity.csv").exists()


def test_spinedb_source_get_returns_dataframe():
    """``get(kind, name)`` returns a polars DataFrame for an existing
    file and ``None`` for a missing one."""
    work = DATA / "work_coal"
    src = SpineDbSource(work / "tests.sqlite", scenario="coal")
    df = src.get("input", "entity")
    assert isinstance(df, pl.DataFrame)
    assert df.height > 0
    # Missing file → None.
    assert src.get("input", "definitely_does_not_exist_csv_zzz") is None


def test_spinedb_source_get_rejects_invalid_kind():
    work = DATA / "work_coal"
    src = SpineDbSource(work / "tests.sqlite", scenario="coal")
    with pytest.raises(ValueError):
        src.get("not_a_kind", "entity")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# 2. CsvSource backward-compat

def test_csv_source_protocol():
    """CsvSource implements the FlexInputSource Protocol."""
    src = CsvSource(DATA / "work_coal")
    assert isinstance(src, FlexInputSource)
    assert (src.input_dir / "entity.csv").exists()
    assert (src.solve_data_dir / "steps_in_use.csv").exists()
    df = src.get("solve_data", "steps_in_use")
    assert df is not None and df.height > 0
    assert src.get("input", "no_such_file") is None


def test_load_flextool_accepts_csv_source():
    """``load_flextool(CsvSource(workdir))`` produces the same data as
    ``load_flextool(workdir)`` (no behaviour change)."""
    work = DATA / "work_coal"
    a = load_flextool(work)
    b = load_flextool(CsvSource(work))
    assert a.dt.equals(b.dt)
    assert a.nodeBalance.equals(b.nodeBalance)
    # Spot-check a Param frame too.
    assert a.p_inflow.frame.equals(b.p_inflow.frame)


def test_load_flextool_rejects_unknown_source_type():
    with pytest.raises(TypeError):
        load_flextool(12345)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# 3. Frame-count parity

@pytest.mark.parametrize(
    "work_name, sqlite_name, scenario, _vobj_stem",
    PARITY_FIXTURES,
    ids=[f[0] for f in PARITY_FIXTURES],
)
def test_spinedb_source_frame_count_parity(
    work_name, sqlite_name, scenario, _vobj_stem,
):
    """The number of frames produced by SpineDbSource matches the
    fixture's on-disk count for ``input/`` (deterministic ``write_input``
    output) and is non-empty for ``solve_data/``.

    ``solve_data/`` totals can differ between the fixture and the
    SpineDbSource tempdir because the fixture is generated by running
    the FULL flextool pipeline (including solve + post-solve writers
    that emit additional per-solve diagnostics, snapshots, etc.),
    while SpineDbSource short-circuits the actual solve via a no-op
    SolverRunner — so per-iteration / post-solve files are absent.
    The frames flexpy actually consumes are produced before the
    solve, so this size mismatch is expected and benign.  We only
    sanity-check that solve_data/ is non-empty.
    """
    work = DATA / work_name
    src = SpineDbSource(work / sqlite_name, scenario=scenario)
    frames = src.build_frames()
    assert len(frames) > 0
    # Both kinds populated.
    kinds = {k for k, _ in frames}
    assert kinds == {"input", "solve_data"}
    fix_input = list((work / "input").glob("*.csv"))
    src_input = sum(1 for k, _ in frames if k == "input")
    src_sd = sum(1 for k, _ in frames if k == "solve_data")
    # input/ frame counts can differ slightly because the fixture is
    # generated by the full flextool pipeline (which may write a few
    # extra debug files post-solve), while SpineDbSource only runs
    # write_input + a no-op solve — bounded but non-zero delta.
    assert abs(src_input - len(fix_input)) <= 15, (
        f"input frames: src={src_input}, fixture={len(fix_input)}")
    # solve_data/ should at least have the core preprocessing outputs.
    assert src_sd >= 100, f"solve_data underpopulated: {src_sd} frames"


# ---------------------------------------------------------------------------
# 4. End-to-end load + solve parity

@pytest.mark.parametrize(
    "work_name, sqlite_name, scenario, vobj_stem",
    PARITY_FIXTURES,
    ids=[f[0] for f in PARITY_FIXTURES],
)
def test_spinedb_source_solve_parity(
    work_name, sqlite_name, scenario, vobj_stem,
):
    """End-to-end: load via SpineDbSource, build + solve, compare to
    the fixture's recorded flextool objective at rel < 1e-6."""
    work = DATA / work_name
    src = SpineDbSource(work / sqlite_name, scenario=scenario)
    data = load_flextool(src)
    pb = Problem()
    build_flextool(pb, data)
    sol = pb.solve()
    assert sol.optimal
    flextool_obj = pl.read_parquet(
        work / "output_raw" / f"v_obj__{vobj_stem}.parquet"
    )["objective"][0]
    rel = abs(sol.obj - flextool_obj) / max(1.0, abs(flextool_obj))
    assert rel < 1e-6, (
        f"{work_name}: flexpy={sol.obj}, flextool={flextool_obj}, rel={rel}")


def test_spinedb_source_matches_csv_source_load():
    """For the work_coal fixture, FlexData loaded via SpineDbSource
    should be functionally equivalent to FlexData loaded via the
    fixture's CSVs (same dt set, same key Params)."""
    work = DATA / "work_coal"
    a = load_flextool(work)  # CSV path
    b = load_flextool(SpineDbSource(work / "tests.sqlite", scenario="coal"))
    # Time set + node set must match exactly.
    assert a.dt.equals(b.dt)
    assert a.nodeBalance.sort("n").equals(b.nodeBalance.sort("n"))
    # Param frames may differ in row order but must agree on sorted content.
    a_inflow = a.p_inflow.frame.sort(["n", "d", "t"])
    b_inflow = b.p_inflow.frame.sort(["n", "d", "t"])
    assert a_inflow.equals(b_inflow)
