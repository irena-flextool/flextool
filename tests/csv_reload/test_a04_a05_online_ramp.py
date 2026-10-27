"""Surface A.4 (Online / Unit-Commitment) and A.5 (Ramp-Rate) loader tests on reload-only seeds.

Moved from ``tests/engine/loaders/test_a04_a05_online_ramp.py`` (c24 S1-8).
They overlay CSVs on a copied work folder and load it reader-less
(``_csv_reload.load_from_dump(workdir)``, no ``db_url``):

* the online tests: ``process_online`` / ``_linear`` / ``_integer`` /
  ``process_minload`` are ``_AUTHORITATIVE_SETS`` and
  ``process_min_load_eff`` a ``SIMPLE_PROJECTIONS`` set, re-derived from
  the DB on every production load.  On the provider path a header-only
  ``process_online`` overlay still gives ``[coal_plant]``, a dropped
  ``process_online_integer`` gives None (not the empty frame), and the
  ``process__ct_method`` overlays (input/ or solve_data/ ``method``
  alias) leave ``process_min_load_eff == [coal_plant]``;
* the all-empty / ramp-speed parts of the ramp blank-path test: the DB
  re-derives ``sink_up`` (still ``[coal_plant]`` under header-only
  overlays) and ``apply_direct_params`` fills ``p_ramp_speed_up_sink``
  (0.001), so "all four None" holds only for a reader-less load.
"""
from __future__ import annotations

import shutil
from pathlib import Path

import polars as pl
import pytest

from flextool.engine._csv_reload import load_from_dump


@pytest.fixture(scope="function")
def online_workdir(tmp_path: Path, scenario_workdir) -> Path:
    """Copy of the ``coal_min_load`` workdir (1 online process w/
    min_load_eff); the reader-less ``load_from_dump`` honours overlays of
    ``solve_data/process_*.csv``.
    """
    target = tmp_path / "work_coal_min_load"
    shutil.copytree(scenario_workdir("coal_min_load"), target)
    return target


@pytest.fixture(scope="function")
def ramp_workdir(tmp_path: Path, scenario_workdir) -> Path:
    """Copy of the ``coal_ramp_limit`` workdir —
    the seed has exactly one ``sink_up`` row and three header-only
    direction files, ideal for the all-four-direction-independence
    assertion.
    """
    target = tmp_path / "work_coal_ramp_limit"
    shutil.copytree(scenario_workdir("coal_ramp_limit"), target)
    return target


def _write(workdir: Path, rel: str, content: str) -> None:
    p = workdir / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content)


# --- A.4 ------------------------------------------------------------

def test_online_empty_and_partition_and_missing_int_file(online_workdir):
    """Covers A4-process_online_empty_returns_blank (direct via empty
    overlay) + A4-process_online_linear_partition (consolidated via the
    seed's online_linear=[coal_plant], integer absent) +
    A4-process_online_integer_missing_file (direct).

    Three sub-asserts on one workdir state: (1) write a header-only
    ``process_online.csv`` so the empty-gate fires and EVERY online key
    returns None, (2) restore the seed and confirm the linear partition
    is preserved verbatim, (3) delete ``process_online_integer.csv``
    and confirm the missing-file fallback substitutes an empty
    ``{p:Utf8}`` frame (NOT None).

    Reload-only seed (c24 S1-7): see the module docstring; moved to tests/csv_reload/ by c24 S1-8.
    """
    # (1) Empty-gate — overwrite to header-only.
    _write(online_workdir, "solve_data/process_online.csv", "process\n")
    data = load_from_dump(online_workdir)
    # Hand-calc: empty process_online.csv -> every online key None.
    for key in ("process_online", "process_online_linear",
                "process_online_integer", "process_minload",
                "process_min_load_eff", "p_online_dt",
                "pdt_online_linear", "pdt_online_integer"):
        assert getattr(data, key) is None, f"{key} not None on empty gate"

    # (2) Restore seed online + delete integer file.  The seed's
    # process_online_linear.csv carries [coal_plant]; integer.csv is
    # already header-only, so we delete it outright to hit the
    # missing-file branch.
    _write(online_workdir, "solve_data/process_online.csv",
           "process\ncoal_plant\n")
    (online_workdir / "solve_data" / "process_online_integer.csv").unlink()
    data = load_from_dump(online_workdir)
    # Hand-calc: linear set is the verbatim 1-row frame from CSV.
    assert data.process_online_linear["p"].to_list() == ["coal_plant"]
    # Hand-calc: missing integer file -> empty {p:Utf8} frame, not None.
    assert data.process_online_integer is not None
    assert data.process_online_integer.height == 0
    assert data.process_online_integer.schema == {"p": pl.Utf8}


def test_minload_set_and_min_load_eff_filter(online_workdir):
    """Covers A4-process_minload_set_load (direct) +
    A4-process_min_load_eff_filter (direct).

    Mutate ``process__ct_method.csv`` to a 2-row mix of methods so the
    filter on ``ct_method == "min_load_efficiency"`` collapses to a
    1-row p-set; ``process_minload`` is loaded as-is from the seed.
    Inserts a second process ``p2`` with method ``linear`` — the seed
    already has ``coal_plant -> min_load_efficiency``.

    Reload-only seed (c24 S1-7): see the module docstring; moved to tests/csv_reload/ by c24 S1-8.
    """
    # Both files exist in seed.  ct_method (input/) wins per precedence.
    _write(online_workdir, "input/process__ct_method.csv",
           "process,ct_method\n"
           "coal_plant,min_load_efficiency\n"
           "p2,linear\n")
    data = load_from_dump(online_workdir)
    # Hand-calc: process_minload set is loaded verbatim from seed
    # (single coal_plant row).
    assert data.process_minload["p"].to_list() == ["coal_plant"]
    # Hand-calc: filter keeps only the min_load_efficiency row -> p=coal_plant.
    assert data.process_min_load_eff["p"].to_list() == ["coal_plant"]
    assert data.process_min_load_eff.height == 1


def test_ct_method_alias_and_inp_over_sd_precedence(online_workdir):
    """Covers A4-ct_method_column_alias_method (direct) +
    A4-ct_method_path_precedence_input_over_solve_data (direct).

    Two mutually-exclusive overlays: the ``input/`` file uses the
    canonical column name ``ct_method`` and lists ``coal_plant``; the
    ``solve_data/`` file uses the .mod's debug-export alias ``method``
    and lists ``other_p``.  The loader must (a) prefer ``input/`` and
    (b) successfully read whichever-named column.  Then we delete the
    ``input/`` file and assert the ``solve_data/`` ``method`` alias is
    honoured.

    Reload-only seed (c24 S1-7): see the module docstring; moved to tests/csv_reload/ by c24 S1-8.
    """
    # input/ has ct_method column, solve_data/ has method alias with
    # different membership.
    _write(online_workdir, "input/process__ct_method.csv",
           "process,ct_method\ncoal_plant,min_load_efficiency\n")
    _write(online_workdir, "solve_data/process__ct_method.csv",
           "process,method\nother_p,min_load_efficiency\n")
    data = load_from_dump(online_workdir)
    # Hand-calc: input/ wins -> only coal_plant retained.
    assert data.process_min_load_eff["p"].to_list() == ["coal_plant"]

    # Now drop input/ -> sd/ method-alias is read.
    (online_workdir / "input" / "process__ct_method.csv").unlink()
    data = load_from_dump(online_workdir)
    # Hand-calc: sd/ method alias detected -> only other_p retained.
    assert data.process_min_load_eff["p"].to_list() == ["other_p"]


# --- A.5 ------------------------------------------------------------

def test_ramp_blank_paths_pss_none_and_all_empty_and_speed_none(
        tiny_workdir, ramp_workdir):
    """Covers A5-empty_when_pss_none (direct on tiny_workdir, where
    ``process_source_sink.csv`` is header-only -> pss=None) +
    A5-empty_files_collapse_to_blank (direct via header-only overlays
    on the ramp workdir) + A5-ramp_speed_params_always_none (direct,
    Δ.12 contract: even with populated direction sets the loader must
    leave all four ``p_ramp_speed_*`` Params as ``None`` — they're
    filled later by ``apply_direct_params``).

    Reload-only seed (c24 S1-7) for parts (2) and (3): see the module
    docstring; moved to tests/csv_reload/ by c24 S1-8.  Part (1) runs on the
    production path in :func:`test_ramp_blank_when_pss_none`.
    """
    # (1) pss=None branch: tiny seed has header-only pss CSV.
    data = load_from_dump(tiny_workdir)
    # Hand-calc: pss None -> blank dict, every ramp key None.
    for key in ("process_source_sink_ramp_limit_sink_up",
                "process_source_sink_ramp_limit_sink_down",
                "process_source_sink_ramp_limit_source_up",
                "process_source_sink_ramp_limit_source_down",
                "p_ramp_speed_up_sink", "p_ramp_speed_down_sink",
                "p_ramp_speed_up_source", "p_ramp_speed_down_source"):
        assert getattr(data, key) is None, f"{key} not None on tiny seed"

    # (2) all-empty collapse: overwrite all four direction CSVs to
    # header-only on the ramp workdir.
    for dirn in ("sink_up", "sink_down", "source_up", "source_down"):
        _write(ramp_workdir,
               f"solve_data/process_source_sink_ramp_limit_{dirn}.csv",
               "process,source,sink\n")
    data = load_from_dump(ramp_workdir)
    # Hand-calc: every _read_set returns None -> any() False -> blank.
    for key in ("process_source_sink_ramp_limit_sink_up",
                "process_source_sink_ramp_limit_sink_down",
                "process_source_sink_ramp_limit_source_up",
                "process_source_sink_ramp_limit_source_down"):
        assert getattr(data, key) is None, f"{key} not None on all-empty"

    # (3) ramp_speed contract: restore one populated direction and
    # confirm Params remain None per Δ.12 drop.
    _write(ramp_workdir,
           "solve_data/process_source_sink_ramp_limit_sink_up.csv",
           "process,source,sink\ncoal_plant,coal_market,west\n")
    data = load_from_dump(ramp_workdir)
    # Hand-calc: set populated -> not the blank branch; per Δ.12 drop
    # all four ramp_speed Params still None.
    assert data.process_source_sink_ramp_limit_sink_up.height == 1
    assert data.p_ramp_speed_up_sink is None
    assert data.p_ramp_speed_down_sink is None
    assert data.p_ramp_speed_up_source is None
    assert data.p_ramp_speed_down_source is None
