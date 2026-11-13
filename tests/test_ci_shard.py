"""``--shard K/N`` (tests/_ci_shard.py) partitions the selection exactly.

The PR Tests workflow runs the suite as N shard jobs per OS/Python cell; a test
in no shard would silently never run in PR CI, one in two would run twice.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from _ci_shard import assign_modules, parse_shard, shard_nodeids

REPO = Path(__file__).resolve().parent.parent


def _ids() -> list[str]:
    ids = [f"tests/m{i}.py::test_{j}" for i in range(12) for j in range(i + 1)]
    return ids + ["tests/new.py::test_a", "tests/new.py::test_b[x]"]


@pytest.mark.parametrize("n", [1, 2, 4, 7])
def test_shards_partition_the_ids_module_atomically(n):
    ids = _ids()
    durations = {f"tests/m{i}.py": float(100 - 7 * i) for i in range(12)}  # tests/new.py unknown
    shards = [shard_nodeids(ids, k, n, durations) for k in range(1, n + 1)]
    assert set().union(*shards) == set(ids)
    assert sum(len(s) for s in shards) == len(ids)
    for s in shards:
        for nid in s:
            same_module = {x for x in ids if x.split("::")[0] == nid.split("::")[0]}
            assert same_module <= s


def test_lpt_balances_and_is_order_independent():
    counts = {f"tests/m{i}.py": 1 for i in range(6)}
    durations = {f"tests/m{i}.py": w for i, w in enumerate([50.0, 40.0, 30.0, 30.0, 20.0, 10.0])}
    a = assign_modules(counts, 2, durations)
    b = assign_modules(dict(reversed(list(counts.items()))), 2, durations)
    assert a == b
    loads = [sum(durations[m] for m, s in a.items() if s == i) for i in range(2)]
    assert loads == [90.0, 90.0]


@pytest.mark.parametrize("bad", ["0/4", "5/4", "2", "a/b", "1/0"])
def test_parse_shard_rejects_bad_values(bad):
    with pytest.raises(ValueError):
        parse_shard(bad)


def _collect(*args: str) -> list[str]:
    done = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/calibrate", "--collect-only", "-q", "-n", "0",
         "-p", "no:cacheprovider", "-m", "not nightly_heavy", *args],
        cwd=REPO, capture_output=True, text=True, encoding="utf-8", errors="replace",
    )
    # The child's own report (a collection error, the session's user-folder check) is the failure message.
    assert done.returncode == 0, f"exit {done.returncode}\n{done.stdout[-3000:]}\n{done.stderr[-3000:]}"
    return [line for line in done.stdout.splitlines() if "::" in line]


def test_the_shard_option_partitions_a_real_marker_filtered_collection():
    full = _collect()
    shards = [_collect("--shard", f"{k}/3") for k in (1, 2, 3)]
    assert full and all(shards)
    assert sorted(x for s in shards for x in s) == sorted(full)
