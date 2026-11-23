"""K-1 on a CSV reload: a one-variable connection's efficiency that is not a unit conversion is ignored there too.

The model is ``lossless_2way`` of ``tests.json`` with efficiency 0.98 on ``ll`` and no units on its nodes
(``tests/engine/test_one_variable_connection.py`` has the solve-side tests).
"""
from __future__ import annotations

import polars as pl
import pytest

from flextool.engine._csv_reload import copy_runner_meta, load_from_dump
from tests._support.load import run_scenario
from tests.engine._unit_db import b64, make_db

pytestmark = pytest.mark.solver


def _ll_slope(flex_data) -> pl.DataFrame:
    frame = flex_data.p_slope.frame
    frame = frame.collect() if isinstance(frame, pl.LazyFrame) else frame
    return frame.filter(pl.col("p").cast(pl.Utf8) == "ll")


@pytest.mark.parametrize("how", ["work_folder", "dump_light", "dump_heavy", "with_db"])
def test_a_csv_reload_ignores_the_efficiency(tmp_path_factory, test_solver_config_dir, how: str) -> None:
    """Slope 1 on ll after a reload of the run's provider snapshot, of a ``dump_csvs`` with or without the heavy
    files (without them the slope comes from the snapshot) and of a reload that reads the database too."""
    url, scen = make_db(tmp_path_factory, alt=f"k_reload_{how}",
                        values=[["connection", "ll", "efficiency", b64(0.98, "float")]],
                        base_chain=["Base", "lossless_2way"])
    work = tmp_path_factory.mktemp(f"k_reload_{how}") / "work"
    run = run_scenario(url, scen, work, solver_config_dir=test_solver_config_dir)
    ran = _ll_slope(run.last_flex_data)
    assert ran.height > 0 and (ran.get_column("value") == 1.0).all(), ran
    dump = work
    if how.startswith("dump"):
        dump = work.parent / "dump"
        run.last_flex_data.dump_csvs(dump, include_heavy=how == "dump_heavy")
        copy_runner_meta(work, dump)
    with pytest.warns(Warning):
        fd = load_from_dump(dump, db_url=url, scenario=scen) if how == "with_db" else load_from_dump(dump)
    reloaded = _ll_slope(fd)
    assert reloaded.height > 0 and (reloaded.get_column("value") == 1.0).all(), reloaded
