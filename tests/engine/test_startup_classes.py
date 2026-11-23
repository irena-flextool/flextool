"""The ``startup_method`` members of data version 83 read as the engine's startup classes (plan §2 "Engine").

``linear`` and ``linear_fix_in_uc`` are the class ``linear`` (continuous units online), ``integer`` and
``integer_fix_in_uc`` the class ``binary`` (whole units, a MIP); ``fix_in_uc`` marks the ``_fix_in_uc`` members.  So a
model with ``linear_fix_in_uc`` and ``integer`` gets the same online sets, internal methods and
``input/process__startup_method`` classes as the same model with ``linear`` and ``integer_fix_in_uc`` (4.6.1
``linear`` / ``binary``).  ``process_online_fix_in_uc`` holds the ``_fix_in_uc`` processes; the realization pins (``fix_online``) and
W-0588 read it.

The model is the ``test_online_connections`` "mixed" chain on ``tests.json`` (built from JSON under the pytest tmp
dir): ``coal_plant`` online by its ``minimum_time_method`` (class linear, no startup_method), the CHP unit
``coal_chp``, the regular connection ``west_east``, the one-way ``west_north`` and the one-variable ``east_north``.
"""
from __future__ import annotations

import logging
import shutil
from pathlib import Path

import polars as pl
import pytest
from spinedb_api import DatabaseMapping, to_database

from flextool.engine.preprocess.params._projection import process_online_fix_in_uc
from flextool.input_derivation._method_constants import startup_class
from flextool.input_derivation._process_method import derive_process_method
from tests._support.coded_warnings import capture_coded
from tests.engine._unit_db import b64, make_db, run, utf8

pytestmark = pytest.mark.solver

CHAIN = ["init", "west", "coal", "heat", "coal_chp", "coal_min_updowntime", "network"]
ENTITIES = [["connection", "en_exact", None], ["connection__node__node", ["en_exact", "east", "north"], None]]
SHAPES = [["connection", "en_exact", "transfer_method", b64("exact", "str")],
          ["connection", "en_exact", "existing", b64(50.0, "float")],
          ["connection", "west_north", "transfer_method", b64("unidirectional", "str")],
          ["connection", "east_north", "transfer_method", b64("no_losses_no_variable_cost", "str")]]
#: process -> (the 4.6.1 member, the v83 member of the same class).
MEMBERS = {"coal_chp": ("integer_fix_in_uc", "integer"), "west_east": ("linear", "linear_fix_in_uc"),
           "west_north": ("integer_fix_in_uc", "integer_fix_in_uc"), "east_north": ("linear", "linear")}
LINEAR = {"coal_plant", "west_east", "east_north"}
INTEGER = {"coal_chp", "west_north"}


def test_startup_class_of_each_member() -> None:
    """Plan §2: the class and the fix_in_uc flag of each member; ``binary`` (before data version 83) reads as
    ``integer_fix_in_uc``; an unknown value is its own class."""
    assert {raw: startup_class(raw) for raw in ("no_startup", "linear", "linear_fix_in_uc", "integer",
                                                "integer_fix_in_uc", "binary", "other")} == {
        "no_startup": ("no_startup", False), "linear": ("linear", False), "linear_fix_in_uc": ("linear", True),
        "integer": ("binary", False), "integer_fix_in_uc": ("binary", True), "binary": ("binary", True),
        "other": ("other", False)}


def _values(member: int) -> list:
    out = list(SHAPES)
    for p, methods in MEMBERS.items():
        cls = "unit" if p == "coal_chp" else "connection"
        out += [[cls, p, "startup_method", b64(methods[member], "str")], [cls, p, "startup_cost", b64(3.0, "float")]]
    return out


@pytest.fixture(scope="module")
def runs(tmp_path_factory) -> dict:
    out = {}
    for member, alt in ((0, "sc_461"), (1, "sc_v83")):
        url, scen = make_db(tmp_path_factory, alt=alt, values=_values(member), base_chain=CHAIN, entities=ENTITIES)
        out[alt] = (url, scen, list(run(tmp_path_factory, url, scen).values())[-1])
    return out


def _p(frame) -> set[str]:
    return set() if frame is None else set(utf8(frame.select(frame.columns[0]))[frame.columns[0]].to_list())


def test_new_members_give_the_same_sets_and_methods(runs) -> None:
    for alt in ("sc_461", "sc_v83"):
        _url, _scen, step = runs[alt]
        fd, prov = step.flex_data, step.flex_data_provider
        assert _p(fd.process_online_linear) == LINEAR, alt
        assert _p(fd.process_online_integer) == INTEGER, alt
        assert _p(fd.process_online) == LINEAR | INTEGER, alt
        assert _p(prov.get("solve_data/process_online_integer")) == INTEGER, alt  # the _calc_params MIP set
        assert _p(prov.get("solve_data/process_online_linear")) == LINEAR, alt
        sm = dict(utf8(prov.get("input/process__startup_method")).select("process", "startup_method").iter_rows())
        assert {p: sm[p] for p in MEMBERS} == {"coal_chp": "binary", "west_east": "linear", "west_north": "binary",
                                               "east_north": "linear"}, alt
    methods = {alt: utf8(runs[alt][2].flex_data_provider.get("input/process_method")).sort("process")
               for alt in runs}
    assert methods["sc_461"].equals(methods["sc_v83"])


def test_process_online_fix_in_uc(runs) -> None:
    """The ``_fix_in_uc`` processes: west_north in both runs, coal_chp only with 4.6.1's member, west_east only with
    v83's; coal_plant (online by its minimum_time_method) never."""
    from flextool.engine.input._spinedb_reader import SpineDbReader

    want = {"sc_461": {"coal_chp", "west_north"}, "sc_v83": {"west_east", "west_north"}}
    for alt, (url, scen, _step) in runs.items():
        assert _p(process_online_fix_in_uc(SpineDbReader(url, scen))) == want[alt], alt


def test_w0228_names_the_stored_member(tmp_path: Path, schema_db_url: str) -> None:
    """A regular connection with two arcs (fork_yes) and startup_method integer has no method (W-0228); the note
    names the stored member, not its class."""
    path = tmp_path / "w0228.sqlite"
    shutil.copy(schema_db_url.removeprefix("sqlite:///"), path)
    with DatabaseMapping(f"sqlite:///{path}") as db:
        for cls, byname in [("node", ("a",)), ("node", ("b",)), ("node", ("c",)), ("connection", ("x",)),
                            ("connection__node__node", ("x", "a", "b")),
                            ("connection__node__node", ("x", "a", "c"))]:
            db.add_entity(entity_class_name=cls, entity_byname=byname)
        val, typ = to_database("integer")
        db.add_parameter_value(entity_class_name="connection", entity_byname=("x",),
                               parameter_definition_name="startup_method", alternative_name="base", value=val,
                               type=typ)
        db.commit_session("w0228")

    class _Provider:
        def __init__(self) -> None:
            self.frames = {"input/process__startup_method": pl.DataFrame(
                {"process": ["x"], "startup_method": ["binary"]})}

        def has(self, key: str) -> bool:
            return key in self.frames

        def get(self, key: str) -> pl.DataFrame:
            return self.frames[key]

        def put(self, key: str, frame: pl.DataFrame) -> None:
            self.frames[key] = frame

    with capture_coded() as got, DatabaseMapping(f"sqlite:///{path}") as db:
        derive_process_method(db, _Provider(), logging.getLogger(__name__), ct_method_overrides={})
    assert [(c.code, n, c.details.get(n)) for c in got.calls for n in c.names if c.code == "W-0228"] == [
        ("W-0228", "x", "ct_method regular, startup_method integer, fork_method fork_yes")]
