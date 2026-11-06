"""``SpineDBBackend.parameter_values(filter_in_index=...)`` keeps only the explicitly-labelled axis (#76).

``filter_in_index`` is the mirror of ``filter_out_index``: a map row is kept only when its outer
``index_name`` equals the filter.  It backs the ``pd_process`` fix for #76 (a period name that is also a
timestep name): without this option a wrong filter would silently drop data with no test to catch it.
"""
from __future__ import annotations

from pathlib import Path

import pytest


def _build_db(sqlite_path: Path) -> str:
    """One class (``widget``), one parameter (``coef``), four entities:

    * ``explicit_period`` — a Map explicitly labelled ``index_name="period"``.
    * ``explicit_time``   — a Map explicitly labelled ``index_name="time"``.
    * ``silent``           — a Map with Spine's silent default index name (``"x"``).
    * ``scalar``            — a plain float (not a map at all).
    """
    from spinedb_api import DatabaseMapping
    from spinedb_api.parameter_value import Map, to_database

    url = f"sqlite:///{sqlite_path}"
    rows = {
        "explicit_period": Map(["p1", "p2"], [1.0, 2.0], index_name="period"),
        "explicit_time": Map(["t1", "t2"], [3.0, 4.0], index_name="time"),
        "silent": Map(["p1", "p2"], [5.0, 6.0]),
        "scalar": 7.0,
    }
    with DatabaseMapping(url, create=True) as db:
        db.add_scenario_item(name="Base")
        db.add_scenario_alternative_item(scenario_name="Base", alternative_name="Base", rank=1)
        db.add_entity_class_item(name="widget")
        db.add_parameter_definition_item(name="coef", entity_class_name="widget")
        for name in rows:
            db.add_entity_item(entity_class_name="widget", name=name)
        for name, value in rows.items():
            db_val, db_type = to_database(value)
            db.add_parameter_value_item(
                entity_class_name="widget", parameter_definition_name="coef",
                entity_byname=(name,), alternative_name="Base", value=db_val, type=db_type)
        db.commit_session("seed")
    return url


@pytest.fixture
def backend(tmp_path: Path):
    from flextool.spinedb_backend import SpineDBBackend

    url = _build_db(tmp_path / "widget.sqlite")
    b = SpineDBBackend(url)
    yield b
    b.close()


def test_filter_in_index_keeps_only_the_explicit_axis(backend) -> None:
    frame = backend.parameter_values(
        cl_pars=[("widget", "coef")], header="widget,idx,coef",
        filter_in_type=["1d_map"], filter_in_index="period")
    assert sorted(frame["widget"].to_list()) == ["explicit_period", "explicit_period"]
    assert sorted(float(v) for v in frame["coef"]) == [1.0, 2.0]


def test_filter_in_index_drops_silent_and_other_axis_maps(backend) -> None:
    frame = backend.parameter_values(
        cl_pars=[("widget", "coef")], header="widget,idx,coef",
        filter_in_type=["1d_map"], filter_in_index="period")
    assert "silent" not in frame["widget"].to_list()
    assert "explicit_time" not in frame["widget"].to_list()


def test_filter_in_index_leaves_scalars_and_other_types_unaffected(backend) -> None:
    """A scalar row has no ``index_name`` to check — ``filter_in_index`` must not drop it."""
    frame = backend.parameter_values(
        cl_pars=[("widget", "coef")], header="widget,coef",
        filter_in_type=["float"], filter_in_index="period")
    assert frame["widget"].to_list() == ["scalar"]


def test_no_filter_in_index_keeps_every_map(backend) -> None:
    frame = backend.parameter_values(
        cl_pars=[("widget", "coef")], header="widget,idx,coef",
        filter_in_type=["1d_map"])
    assert sorted(frame["widget"].unique().to_list()) == ["explicit_period", "explicit_time", "silent"]
