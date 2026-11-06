"""Run a command and report the runner's cores and the run's peak memory.

Usage: ``python .github/scripts/peak_mem.py pytest tests/ ...``

The CI test jobs run pytest with one xdist worker per logical core
(``-n logical``); this wrapper prints the runner's logical / physical core
count and total RAM before the run, and after it the peak system-wide used
memory, the peak summed RSS of the command's process tree (pytest + workers)
and the largest single process RSS, sampled every 2 s.  The summary also goes
to the job summary (``$GITHUB_STEP_SUMMARY``).  It exits with the command's
exit code.  Needs ``psutil`` (a FlexTool dependency).
"""
from __future__ import annotations

import os
import subprocess
import sys
import time

import psutil

GIB = 1024**3


def main(argv: list[str]) -> int:
    vm = psutil.virtual_memory()
    head = (
        f"runner: {psutil.cpu_count(logical=True)} logical / "
        f"{psutil.cpu_count(logical=False)} physical cores, {vm.total / GIB:.1f} GiB RAM, "
        f"{vm.available / GIB:.1f} GiB available"
    )
    print(head, flush=True)
    proc = subprocess.Popen(argv)
    root = psutil.Process(proc.pid)
    peak_used = vm.total - vm.available
    peak_tree = peak_one = 0
    while proc.poll() is None:
        rss = []
        try:
            members = [root, *root.children(recursive=True)]
        except psutil.Error:
            members = []
        for p in members:
            try:
                rss.append(p.memory_info().rss)
            except psutil.Error:
                pass
        peak_tree = max(peak_tree, sum(rss))
        peak_one = max(peak_one, max(rss, default=0))
        vm = psutil.virtual_memory()
        peak_used = max(peak_used, vm.total - vm.available)
        time.sleep(2)
    tail = (
        f"peak memory: system used {peak_used / GIB:.1f} GiB of {vm.total / GIB:.1f}; "
        f"pytest tree {peak_tree / GIB:.1f} GiB; largest process {peak_one / GIB:.1f} GiB"
    )
    print(tail, flush=True)
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a", encoding="utf-8") as fh:
            fh.write(f"- {head}\n- {tail}\n")
    return proc.returncode


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
