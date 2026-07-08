"""Regression: representative-period label columns must survive the
``load_flextool`` axis-enum sweep (the CSV / provider round-trip).

Root cause (fixed in ``_axis_enums._FLEXDATA_SWEEP_SKIP_COLUMNS``): the RP
base-period / rep-start relations name their label columns ``"b"`` (base
period start) and ``"r"`` (rep-block start) so ``model.py``'s inter-period
closure wiring can join on the ``.mod`` symbols — but those labels are
TIMESTEP tokens (``t0001`` …), NOT branch / rep-axis tokens.  The
end-of-load ``cast_flexdata_axes`` sweep resolves ``"b" → branch`` and
treats ``"r"`` as the (rep) axis via the synonym table, so a blind cast
nulled every value (branch vocab is e.g. ``{eff, noEff}``; the rep axis
enum is empty for a non-stochastic solve).  A null ``rp_base_period_set.b``
collapses ``v_state_inter``'s index → the seasonal-closure constraints
(``rp_inter_period_cyclic`` / ``_balance``) are silently dropped → the LP
solves an under-constrained, too-cheap problem.

This test pins the contract that ``load_flextool`` reproduces the emitted
RP labels faithfully AND that the seasonal-closure constraint therefore
fires with a non-empty index.  It exercises BOTH the disk-CSV reload
(bare-Path ``load_flextool``) and — mirroring production
(``_orchestration.py`` ``load_flextool(..., provider=...)``) — the
in-memory Provider reload; both must produce identical, non-null RP frames
and the same closure row count.
"""
from __future__ import annotations

import shutil
from pathlib import Path
from urllib.parse import urlparse

import polars as pl
import pytest

from polar_high import Problem

from flextool.engine_polars import build_flextool, load_flextool


@pytest.fixture(scope="module")
def rp_build(tmp_path_factory, lh2_rp_invest_db_url, test_solver_config_dir):
    """Run the full cascade for the RP-invest fixture and return
    ``(workdir, provider)`` — the workdir carries the emitted CSVs on
    disk and ``provider`` is the live in-memory sub-solve Provider
    (kept via ``keep_solutions=True``), so the test can compare the two
    reload paths."""
    from flextool.engine_polars._orchestration import run_chain_from_db

    parent = tmp_path_factory.mktemp("_root_rp_roundtrip")
    wf = parent / "work_lh2_three_region_rp_invest"
    wf.mkdir()
    steps = run_chain_from_db(
        input_db_url=lh2_rp_invest_db_url,
        scenario_name="lh2_three_region_rp_invest",
        work_folder=wf,
        solver_config_dir=test_solver_config_dir,
        csv_dump=True,
        keep_solutions=True,
    )
    shutil.copy(urlparse(lh2_rp_invest_db_url).path, wf / "tests.sqlite")
    last_step = next(reversed(list(steps.values())))
    provider = getattr(last_step, "flex_data_provider", None)
    assert provider is not None
    provider.snapshot_processed_inputs(wf)
    return wf, provider


def _read_csv_labels(sd: Path, name: str, col: str) -> set[str]:
    df = pl.read_csv(sd / f"{name}.csv")
    return set(df[col].cast(pl.Utf8).to_list())


def _closure_rows(data) -> int:
    pb = Problem()
    build_flextool(pb, data)
    names = pb.cstr_names()
    total = 0
    for n in names:
        if n.startswith("rp_inter_period_cyclic") or n.startswith(
            "rp_inter_period_balance"
        ):
            total += pb.cstr_row_count(n)
    return total


@pytest.mark.parametrize("via_provider", [False, True])
def test_rp_labels_survive_load(rp_build, via_provider):
    wf, provider = rp_build
    sd = wf / "solve_data"

    # The emitted CSVs on disk DO carry the labels (so any nulling is a
    # read/cast bug, not an emit bug).
    emitted_base = _read_csv_labels(sd, "rp_base_first", "base_start")
    emitted_rep = _read_csv_labels(sd, "rp_weights", "rep_start")
    assert emitted_base and all(b is not None for b in emitted_base)
    assert emitted_rep and all(r is not None for r in emitted_rep)

    data = load_flextool(wf, provider=provider if via_provider else None)

    # --- The RP label columns round-trip NON-NULL and equal to the emit.
    def labels(field, col):
        v = getattr(data, field)
        frame = getattr(v, "frame", v)
        assert frame is not None and frame.height > 0, f"{field} empty"
        s = frame[col]
        assert s.null_count() == 0, (
            f"{field}.{col} has {s.null_count()} null label(s) after load "
            f"(the branch/rep-axis cast collision regressed)"
        )
        return set(s.cast(pl.Utf8).to_list())

    assert labels("rp_base_first", "b") == emitted_base
    assert labels("rp_base_last", "b") == emitted_base
    assert labels("rp_base_period_set", "b")  # non-null, non-empty
    rep_labels = labels("rp_base__rep", "r")
    assert rep_labels == emitted_rep
    assert labels("rp_base__rep", "b") == emitted_base
    assert labels("p_rp_last_step", "r") == emitted_rep

    # --- Because the labels survive, the seasonal-closure constraint
    # fires with a non-empty index (it would be dropped entirely if any
    # of the tightly-coupled RP frames nulled out).
    assert _closure_rows(data) > 0, (
        "rp_inter_period seasonal-closure constraint did not fire — a "
        "nulled RP label collapsed v_state_inter's index"
    )


def test_disk_and_provider_reload_agree(rp_build):
    """The disk-CSV reload and the in-memory Provider reload (the
    production path) produce byte-identical RP label frames and the same
    closure row count."""
    wf, provider = rp_build
    disk = load_flextool(wf)
    mem = load_flextool(wf, provider=provider)

    for field, cols in (
        ("rp_base_first", ("b", "d")),
        ("rp_base_last", ("b", "d")),
        ("rp_base_period_set", ("b",)),
        ("rp_base__rep", ("b", "r", "value")),
        ("p_rp_last_step", ("r", "last_step")),
    ):
        dv = getattr(disk, field)
        mv = getattr(mem, field)
        df = getattr(dv, "frame", dv).select(cols).sort(cols)
        mf = getattr(mv, "frame", mv).select(cols).sort(cols)
        assert df.equals(mf), f"{field} differs between disk and provider reload"

    assert _closure_rows(disk) == _closure_rows(mem) > 0
