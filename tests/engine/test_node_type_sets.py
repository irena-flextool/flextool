"""The node sets of the database path read a node without ``node_type`` as its default, ``balance`` when that is empty.

``params/_projection.node_type_effective`` feeds ``nodeBalance`` (balance and storage nodes), ``nodeState``
(storage nodes) and ``nodeBalancePeriod`` (balance_within_period nodes).  Nodes ``b`` (balance), ``s`` (storage),
``w`` (balance_within_period) and ``c`` (commodity) have a value; ``x`` has none.
"""
from __future__ import annotations

import polars as pl
import pytest

from flextool.engine.input._inmemory_reader import InMemoryReader
from flextool.engine.preprocess.params._projection import nodeBalance, nodeBalancePeriod, nodeState


def _source(default=None, typed: bool = True) -> InMemoryReader:
    parameters = {("node", "node_type"): pl.DataFrame({"name": ["b", "c", "s", "w"],
                                                       "value": ["balance", "commodity", "storage",
                                                                 "balance_within_period"]})} if typed else {}
    defaults = {("node", "node_type"): default} if default is not None else None
    return InMemoryReader(entities={"node": pl.DataFrame({"name": ["b", "c", "s", "w", "x"]})},
                          parameters=parameters, defaults=defaults)


def _names(frame: pl.DataFrame) -> list[str]:
    return frame["n"].cast(pl.Utf8).to_list()


@pytest.mark.parametrize(("default", "balance", "state", "period"), [
    (None, ["b", "s", "x"], ["s"], ["w"]),            # empty default: x is a balance node
    ("commodity", ["b", "s"], ["s"], ["w"]),          # x takes the default: in no set
    ("storage", ["b", "s", "x"], ["s", "x"], ["w"]),  # x takes the default: a storage node
])
def test_a_node_without_node_type_takes_the_default_or_balance(default, balance, state, period) -> None:
    source = _source(default)
    assert (_names(nodeBalance(source)), _names(nodeState(source)), _names(nodeBalancePeriod(source))) == (
        balance, state, period)


def test_without_any_node_type_value_every_node_is_a_balance_node() -> None:
    source = _source(typed=False)
    assert _names(nodeBalance(source)) == ["b", "c", "s", "w", "x"]
    assert _names(nodeState(source)) == [] and _names(nodeBalancePeriod(source)) == []
