"""A ``bind_intraperiod_blocks`` node in a solve without timeset blocks is refused (item 113).

Before, ``_eq_block_coupling`` returned silently when the block frames were missing, so the node had no energy
balance at all (``nodeBalance_eq`` skips ``nodeStateBlock`` nodes).
"""
from __future__ import annotations

import pytest
from polar_high import Problem

from flextool.engine import build_flextool
from flextool.engine.input._solve_state import FlexToolConfigError


def test_block_node_without_blocks_raises(toy_storage_blocks) -> None:
    data = toy_storage_blocks
    data.period_block = None
    data.period_block_succ = None
    data.period_block_time = None
    with pytest.raises(FlexToolConfigError, match="s.*no timeset blocks"):
        build_flextool(Problem(), data)
