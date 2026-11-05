"""Rebuild ``tests/ci_durations.json`` (the ``--shard`` weights) from CI job logs.

Usage::

    gh api repos/:owner/:repo/actions/jobs/<job-id>/logs > job.log   # a few complete Linux jobs
    python tools/ci_durations.py job1.log job2.log [--exclude-ids ids.txt] [-o tests/ci_durations.json]

Each log must come from a ``pytest -v`` xdist run (lines ``<ts>Z [gwN] [ P%] PASSED <nodeid>``).
A test's worker seconds are the gap to the previous result line of the same
worker, so module-scoped fixture setup is charged to the module's first test.
Module weight = the mean, over the logs that ran the module, of its summed seconds
(so the logs of the N shard jobs of one PR run combine),
counting only tests not listed in ``--exclude-ids`` (one node id per line, e.g.
the ``pytest --collect-only -q -m nightly_heavy`` output, so the weights match
the PR selection).  Logs of PR shard jobs already lack those tests.  Only the
relative weights matter; Linux logs are the reference.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import re
from collections import defaultdict
from pathlib import Path

_LINE = re.compile(r"(\S+)Z \[(gw\d+)\] \[ *\d+%\] \w+ (\S+)")


def per_test_seconds(log: Path) -> dict[str, float]:
    last: dict[str, dt.datetime] = {}
    out: dict[str, float] = {}
    for line in log.read_text(encoding="utf-8", errors="replace").splitlines():
        m = _LINE.match(line)
        if not m:
            continue
        t = dt.datetime.fromisoformat(m[1][:26])
        out[m[3]] = (t - last[m[2]]).total_seconds() if m[2] in last else 0.0
        last[m[2]] = t
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("logs", nargs="+", type=Path)
    ap.add_argument("--exclude-ids", type=Path, default=None)
    ap.add_argument("-o", "--output", type=Path, default=Path("tests/ci_durations.json"))
    args = ap.parse_args()
    exclude: set[str] = set()
    if args.exclude_ids:
        exclude = {s.strip() for s in args.exclude_ids.read_text(encoding="utf-8").splitlines() if "::" in s}
    per_log: dict[str, list[float]] = defaultdict(list)
    for log in args.logs:
        totals: dict[str, float] = defaultdict(float)
        for nodeid, sec in per_test_seconds(log).items():
            if nodeid not in exclude:
                totals[nodeid.split("::", 1)[0]] += sec
        for m, v in totals.items():
            per_log[m].append(v)
    weights = {m: round(sum(v) / len(v), 1) for m, v in sorted(per_log.items())}
    args.output.write_text(json.dumps(weights, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(f"{args.output}: {len(weights)} modules, {sum(weights.values()) / 60:.0f} worker-min")


if __name__ == "__main__":
    main()
