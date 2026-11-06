"""Group ``invest_method`` values (``group_invest_methods`` list, database version 77).

``not_active`` is the group's "no limits" value (and its default, which the
reader spreads to every group without a value): such a group gets no
``group_invest`` / ``group_divest`` row, while a group with an investment or
retirement method does.  The old name ``not_allowed`` means the database was
not upgraded and is an error.
"""
from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from flextool.engine.input._solve_state import FlexToolConfigError
from flextool.engine.preprocess.emit._leaf_sets import derive_group_divest, derive_group_invest

INPUT = "input/group__invest_method.csv"


class _Provider:
    """A minimal FlexDataProvider: frames keyed by their canonical provider key."""

    def __init__(self, frames: dict[str, pl.DataFrame]):
        from flextool.engine.preprocess._provider_io import _provider_key

        self._frames = {_provider_key(path): frame for path, frame in frames.items()}

    def has(self, key: str) -> bool:
        return key in self._frames

    def get(self, key: str) -> pl.DataFrame:
        return self._frames[key]


def _groups(rows: list[tuple[str, str]], derive) -> list[str]:
    frame = pl.DataFrame({"group": [g for g, _ in rows], "invest_method": [m for _, m in rows]})
    out = derive(Path("input"), provider=_Provider({INPUT: frame}))
    return out.get_column("group").cast(pl.Utf8).to_list()


ROWS = [
    ("g_default", "not_active"),     # the spread default
    ("g_explicit", "not_active"),
    ("g_invest", "invest_total"),
    ("g_retire", "retire_period"),
    ("g_both", "invest_retire_period_total"),
]


def test_not_active_groups_do_not_invest() -> None:
    assert _groups(ROWS, derive_group_invest) == ["g_invest", "g_both"]


def test_not_active_groups_do_not_divest() -> None:
    assert _groups(ROWS, derive_group_divest) == ["g_retire", "g_both"]


@pytest.mark.parametrize("derive", [derive_group_invest, derive_group_divest])
def test_old_not_allowed_value_is_an_error(derive) -> None:
    with pytest.raises(FlexToolConfigError, match="renamed 'not_active' in database version 77.*g_old"):
        _groups([("g_old", "not_allowed"), ("g_invest", "invest_total")], derive)
