"""``--shard K/N``: run one of N time-balanced, module-atomic slices of the suite.

The PR "Tests" workflow (``.github/workflows/tests.yml``) runs every OS/Python
cell as N matrix jobs, each passing ``--shard K/N``.  The split:

* is **module-atomic** — every test of a file lands in the same shard, so a
  module-scoped fixture is built once (the same reason the suite runs with
  ``--dist loadfile``; within a shard xdist still distributes by file);
* is **time-balanced** — modules are weighted by their recorded CI worker
  seconds (``tests/ci_durations.json``) and placed greedily, heaviest first,
  into the currently lightest shard (LPT).  A module missing from the file
  (a new one) weighs ``len(its selected tests) * DEFAULT_SECONDS_PER_TEST``;
* is **deterministic** — it depends only on the selected test ids and the
  committed durations file (ties broken by module path, then shard index), so
  every xdist worker and every shard job computes the same assignment;
* runs **after** ``-m`` / ``-k`` / ``--regenerate`` deselection (``trylast``),
  so the N shards partition exactly the set the same command would run
  unsharded (checked by ``tests/test_ci_shard.py``).

Refresh the durations file from complete CI job logs with
``python tools/ci_durations.py <job-log>...`` (see CONTRIBUTING "Running Tests").
"""
from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from pathlib import Path

import pytest

DURATIONS_FILE = Path(__file__).parent / "ci_durations.json"
# Mean worker seconds per test over the recorded Linux CI runs (~368 worker-min
# over ~6400 tests); the weight of one test of a module not in the file.
DEFAULT_SECONDS_PER_TEST = 3.5


def parse_shard(value: str) -> tuple[int, int]:
    """``"K/N"`` -> ``(K, N)`` with ``1 <= K <= N``; raises ``ValueError`` otherwise."""
    try:
        k_str, n_str = value.split("/")
        k, n = int(k_str), int(n_str)
    except ValueError:
        raise ValueError(f"--shard expects K/N (e.g. 2/4), got {value!r}") from None
    if not 1 <= k <= n:
        raise ValueError(f"--shard {value!r}: need 1 <= K <= N")
    return k, n


def load_durations(path: Path = DURATIONS_FILE) -> dict[str, float]:
    """Module path (``tests/...py``) -> recorded worker seconds; empty when missing."""
    try:
        return {str(k): float(v) for k, v in json.loads(path.read_text(encoding="utf-8")).items()}
    except FileNotFoundError:
        return {}


def module_of(nodeid: str) -> str:
    return nodeid.split("::", 1)[0]


def assign_modules(
    module_counts: Mapping[str, int], n: int, durations: Mapping[str, float]
) -> dict[str, int]:
    """Module -> shard index (0-based) by LPT over the modules' weights."""
    weights = {
        m: durations.get(m, count * DEFAULT_SECONDS_PER_TEST) for m, count in module_counts.items()
    }
    loads = [0.0] * n
    out: dict[str, int] = {}
    for m in sorted(weights, key=lambda m: (-weights[m], m)):
        i = min(range(n), key=lambda j: (loads[j], j))
        out[m] = i
        loads[i] += weights[m]
    return out


def shard_nodeids(nodeids: Iterable[str], k: int, n: int, durations: Mapping[str, float]) -> set[str]:
    """The node ids (of ``nodeids``) that shard ``k`` of ``n`` (1-based) runs."""
    ids = list(nodeids)
    counts: dict[str, int] = {}
    for nid in ids:
        counts[module_of(nid)] = counts.get(module_of(nid), 0) + 1
    where = assign_modules(counts, n, durations)
    return {nid for nid in ids if where[module_of(nid)] == k - 1}


class ShardPlugin:
    def __init__(self, k: int, n: int) -> None:
        self.k, self.n = k, n

    @pytest.hookimpl(trylast=True)
    def pytest_collection_modifyitems(self, config: pytest.Config, items: list[pytest.Item]) -> None:
        keep = shard_nodeids((it.nodeid for it in items), self.k, self.n, load_durations())
        selected = [it for it in items if it.nodeid in keep]
        deselected = [it for it in items if it.nodeid not in keep]
        if deselected:
            config.hook.pytest_deselected(items=deselected)
        items[:] = selected

    def pytest_report_header(self, config: pytest.Config) -> str:
        return f"shard: {self.k}/{self.n} (module-atomic, time-balanced; tests/_ci_shard.py)"
