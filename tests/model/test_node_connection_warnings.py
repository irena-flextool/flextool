"""``warn_node_connections`` (#276): a node whose balance has nothing to work with is named once per run.

Each case is a database built from ``schemas/spinedb_schema.json`` (its defaults: ``node_type`` balance,
``inflow_method`` use_original, ``transfer_method`` regular) with the entities of the case active in the alternative
``base``, read through the scenario filter of scenario ``s`` = [base] (as ``input_derivation.run`` reads it).
"""
from __future__ import annotations

import contextlib
import io
import shutil
from pathlib import Path

import spinedb_api as api

from flextool.input_derivation._validators import warn_node_connections
from tests.engine._unit_db import PLANT_CHAIN, make_db, run

EMPTY_INFLOW = ("Warning: node '{n}' (alternative '{a}') has no units or connections; its inflow can only be met by "
                "the penalty slack (penalty_up / penalty_down).")
EMPTY = "Warning: node '{n}' has no units or connections and no inflow; it does nothing in this scenario."
ONLY_DRAINED = ("Warning: node '{n}' (node_type {k}) has no positive inflow and its units / connections only take "
                "energy out of it; nothing brings energy in, so those flows stay at 0, or only the penalty slack "
                "(penalty_up) can supply them. Add a unit or connection that supplies it, give it a positive inflow, "
                "or make it a commodity node.")
ONLY_SUPPLIED = ("Warning: node '{n}' (node_type {k}) has no negative inflow (demand) and its units / connections "
                 "only bring energy into it; nothing takes energy out, so those flows stay at 0, or only the penalty "
                 "slack (penalty_down) can absorb them. Add a unit or connection that takes from it, or give it a "
                 "negative inflow.")


def _warnings(tmp_path: Path, schema_db_url: str, entities: list, values: list = (),
              inactive: tuple[str, ...] = (), name: str = "db") -> list[str]:
    """The printed lines of ``warn_node_connections`` on a database holding ``entities`` (``[class, byname]``;
    1-D ones active in ``base`` unless named in ``inactive``) and ``values`` (``[class, entity, parameter,
    value]``, in ``base``)."""
    sqlite = tmp_path / f"{name}.sqlite"
    shutil.copy(schema_db_url.removeprefix("sqlite:///"), sqlite)
    url = f"sqlite:///{sqlite}"
    with api.DatabaseMapping(url) as db:
        api.import_functions.import_data(
            db,
            alternatives=[["base", ""]],
            scenarios=[["s", False, ""]],
            scenario_alternatives=[["s", "base", None]],
            entities=[[cls, byname, None] for cls, byname in entities],
            entity_alternatives=[[cls, [byname], "base", byname not in inactive]
                                 for cls, byname in entities if isinstance(byname, str)],
            parameter_values=[[cls, ent, par, val, "base"] for cls, ent, par, val in values],
        )
        db.commit_session("node connection warnings fixture")
    with api.DatabaseMapping(url) as db:
        db.fetch_all("entity")
        db.fetch_all("parameter_value")
        api.filters.scenario_filter.scenario_filter_from_dict(
            db, api.filters.scenario_filter.scenario_filter_config("s"))
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            warn_node_connections(db)
    return out.getvalue().splitlines()


def _ts_val(values: list[float]) -> api.Map:
    """A time map over t0001, t0002, ..."""
    return api.Map([f"t{i + 1:04d}" for i in range(len(values))], values, index_name="time")


PLANT = [["node", "fuel"], ["node", "elec"], ["unit", "plant"],
         ["unit__inputNode", ["plant", "fuel"]], ["unit__outputNode", ["plant", "elec"]]]


def test_empty_node_with_inflow(tmp_path, schema_db_url) -> None:
    """(a) No unit, no connection, a demand: only the slack meets it; the inflow's alternative is named."""
    lines = _warnings(tmp_path, schema_db_url, [["node", "west"]],
                      [["node", "west", "inflow", _ts_val([0.0, -10.0, -20.0])]])
    assert lines == [EMPTY_INFLOW.format(n="west", a="base")]


def test_empty_node_without_inflow(tmp_path, schema_db_url) -> None:
    """(b) No unit, no connection, no inflow; an all-zero inflow is no inflow."""
    lines = _warnings(tmp_path, schema_db_url, [["node", "a"], ["node", "b"]],
                      [["node", "b", "inflow", _ts_val([0.0, 0.0])]])
    assert lines == [EMPTY.format(n="a"), EMPTY.format(n="b")]


def test_no_inflow_method_ignores_the_inflow(tmp_path, schema_db_url) -> None:
    """(c) inflow_method no_inflow: the inflow is ignored, so the node has none."""
    lines = _warnings(tmp_path, schema_db_url, [["node", "west"]],
                      [["node", "west", "inflow", -5.0], ["node", "west", "inflow_method", "no_inflow"]])
    assert lines == [EMPTY.format(n="west")]


def test_commodity_and_storage_nodes(tmp_path, schema_db_url) -> None:
    """(d) A commodity node has no balance: silent, empty or one-sided.  (e) A storage node is exempt from the
    one-sided check (its state is the other side), but an empty storage node is named."""
    entities = [["node", "coal"], ["node", "bare_coal"], ["node", "store"], ["node", "bare_store"],
                ["unit", "charger"], ["unit__outputNode", ["charger", "store"]],
                ["unit__inputNode", ["charger", "coal"]]]
    lines = _warnings(tmp_path, schema_db_url, entities, [
        ["node", "coal", "node_type", "commodity"], ["node", "bare_coal", "node_type", "commodity"],
        ["node", "store", "node_type", "storage"], ["node", "bare_store", "node_type", "storage"]])
    assert lines == [EMPTY.format(n="bare_store")]


def test_one_sided_balance_nodes(tmp_path, schema_db_url) -> None:
    """(f) A balance fuel node only drained by a plant, and the plant's output node only supplied, both without
    inflow: named.  A demand on elec (inflow) or a supply on fuel silences that node."""
    lines = _warnings(tmp_path, schema_db_url, PLANT)
    assert lines == [ONLY_SUPPLIED.format(n="elec", k="balance"), ONLY_DRAINED.format(n="fuel", k="balance")]
    lines = _warnings(tmp_path, schema_db_url, PLANT, [["node", "elec", "inflow", -100.0],
                                                       ["node", "fuel", "inflow", _ts_val([0.0, 50.0])]],
                      name="with_inflow")
    assert lines == []


def test_balance_within_period_is_checked(tmp_path, schema_db_url) -> None:
    """(g) balance_within_period is balanced too (over the period): one-sided is named, with its node_type."""
    lines = _warnings(tmp_path, schema_db_url, PLANT, [["node", "elec", "inflow", -100.0],
                                                       ["node", "fuel", "node_type", "balance_within_period"]])
    assert lines == [ONLY_DRAINED.format(n="fuel", k="balance_within_period")]


def test_connection_direction(tmp_path, schema_db_url) -> None:
    """(h) A two-way connection counts as both sides at both nodes (silent); a unidirectional one drains its source
    and supplies its sink only."""
    entities = [["node", "n1"], ["node", "n2"], ["connection", "line"],
                ["connection__node__node", ["line", "n1", "n2"]]]
    assert _warnings(tmp_path, schema_db_url, entities) == []
    lines = _warnings(tmp_path, schema_db_url, entities, [["connection", "line", "transfer_method", "unidirectional"]],
                      name="unidirectional")
    assert lines == [ONLY_DRAINED.format(n="n1", k="balance"), ONLY_SUPPLIED.format(n="n2", k="balance")]


def test_inflow_of_the_wrong_sign_does_not_balance(tmp_path, schema_db_url) -> None:
    """``inflow`` is energy into the node (negative for a demand): only a positive inflow supplies a node whose
    links only drain it, only a negative one drains a node whose links only supply it.  A demand on fuel (only
    drained) and a positive inflow on elec (only supplied) leave both one-sided."""
    lines = _warnings(tmp_path, schema_db_url, PLANT, [["node", "elec", "inflow", 100.0],
                                                       ["node", "fuel", "inflow", _ts_val([0.0, -50.0])]])
    assert lines == [ONLY_SUPPLIED.format(n="elec", k="balance"), ONLY_DRAINED.format(n="fuel", k="balance")]
    lines = _warnings(tmp_path, schema_db_url, PLANT, [["node", "elec", "inflow", _ts_val([10.0, -100.0])],
                                                       ["node", "fuel", "inflow", _ts_val([-5.0, 50.0])]],
                      name="both_signs")
    assert lines == []


def test_inflow_adder_margin_drains(tmp_path, schema_db_url) -> None:
    """An energy_margin_method inflow_adder adds demand in the investment solve: elec is not one-sided."""
    lines = _warnings(tmp_path, schema_db_url, PLANT, [["node", "elec", "energy_margin_method", "inflow_adder"],
                                                       ["node", "fuel", "inflow", 50.0]])
    assert lines == []


def test_group_transfer_method_overrides_the_connection(tmp_path, schema_db_url) -> None:
    """A group holding both nodes overrides the connection's transfer_method: a unidirectional group makes a
    regular line one-way, a regular group makes a unidirectional line two-way; a group holding one node only, or
    use_connection_transfer_methods, changes nothing."""
    line = [["node", "n1"], ["node", "n2"], ["node", "n3"], ["connection", "line"],
            ["connection__node__node", ["line", "n1", "n2"]], ["group", "g"]]
    both = [*line, ["group__node", ["g", "n1"]], ["group__node", ["g", "n2"]]]
    one_way = [ONLY_DRAINED.format(n="n1", k="balance"), ONLY_SUPPLIED.format(n="n2", k="balance"),
               EMPTY.format(n="n3")]
    lines = _warnings(tmp_path, schema_db_url, both, [["group", "g", "transfer_method", "unidirectional"]],
                      name="g_uni")
    assert lines == one_way
    lines = _warnings(tmp_path, schema_db_url, both, [["group", "g", "transfer_method", "regular"],
                                                      ["connection", "line", "transfer_method", "unidirectional"]],
                      name="g_regular")
    assert lines == [EMPTY.format(n="n3")]
    lines = _warnings(tmp_path, schema_db_url, [*line, ["group__node", ["g", "n1"]], ["group__node", ["g", "n3"]]],
                      [["group", "g", "transfer_method", "unidirectional"]], name="g_one_end")
    assert lines == [EMPTY.format(n="n3")]
    lines = _warnings(tmp_path, schema_db_url, both,
                      [["group", "g", "transfer_method", "use_connection_transfer_methods"],
                       ["connection", "line", "transfer_method", "unidirectional"]], name="g_use_own")
    assert lines == one_way


def test_anti_energy_unit_counts_both_ways(tmp_path, schema_db_url) -> None:
    """A unit with a negative existing (anti-energy: its flows run the other way) counts on both sides of its
    nodes, so neither fuel nor elec is one-sided; a positive existing keeps the plain direction."""
    lines = _warnings(tmp_path, schema_db_url, PLANT, [["unit", "plant", "existing", -50.0]])
    assert lines == []
    lines = _warnings(tmp_path, schema_db_url, PLANT, [["unit", "plant", "existing", 50.0]], name="positive")
    assert lines == [ONLY_SUPPLIED.format(n="elec", k="balance"), ONLY_DRAINED.format(n="fuel", k="balance")]


def test_inactive_unit_leaves_the_node_empty(tmp_path, schema_db_url) -> None:
    """(i) The scenario filter drops the links of a unit inactive in the scenario: its nodes are empty."""
    lines = _warnings(tmp_path, schema_db_url, PLANT, [["node", "elec", "inflow", -100.0]],
                      inactive=("plant",))
    assert lines == [EMPTY_INFLOW.format(n="elec", a="base"), EMPTY.format(n="fuel")]


def test_printed_once_per_run(tmp_path_factory) -> None:
    """A full run (tests.json init + west + coal and a bare node 'lonely') prints the warning once."""
    url, scenario = make_db(tmp_path_factory, alt="lonely_node", values=[], base_chain=PLANT_CHAIN,
                            entities=[["node", "lonely", None]])
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        run(tmp_path_factory, url, scenario)
    assert out.getvalue().count(EMPTY.format(n="lonely")) == 1, out.getvalue()
    assert "node 'west'" not in out.getvalue()

