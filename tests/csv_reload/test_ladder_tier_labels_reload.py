"""CSV reload of a price ladder with text tier labels (``01/02/03``): the reload keeps the labels as written."""
from __future__ import annotations

import json

import pytest

from tests._support.fields import field_diff
from tests.db_utils import json_to_db
from tests.engine.test_ladder_tier_labels import ANN_TIERS, _data, _ladder


def test_csv_reload_keeps_text_labels(tmp_path_factory, test_solver_config_dir) -> None:
    """A dump of a ``01/02/03`` ladder reloads with the same tier labels in the ladder sets (the reload reads
    ``tier`` columns as text, so ``01`` does not become ``1``)."""
    from flextool.engine._csv_reload import copy_runner_meta, load_from_dump
    from flextool.update_flextool.db_migration import migrate_database
    from tests._support.load import run_scenario

    work = tmp_path_factory.mktemp("lbl_reload")
    (work / "f.json").write_text(json.dumps(_data("annual", "lbl_reload", _ladder(["01", "02", "03"], ANN_TIERS))))
    url = json_to_db(work / "f.json", work / "f.sqlite")
    migrate_database(url)
    run = run_scenario(url, "lbl_reload", work / "work_lbl_reload", solver_config_dir=test_solver_config_dir)
    dump = work / "dump"
    run.last_flex_data.dump_csvs(dump, include_heavy=True)
    assert copy_runner_meta(run.work_folder, dump) > 0
    with pytest.warns(Warning):
        reloaded = load_from_dump(dump)
    for field in ("commodity__tier_ann", "cndi_ladder_ann"):
        got = getattr(reloaded, field)
        assert got is not None, field
        assert field_diff(got, getattr(run.last_flex_data, field)) is None, field
