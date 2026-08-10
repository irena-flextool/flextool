"""Switch S — per-region weight matrices are cross-process deterministic.

``compute_region_weight_maps`` must return byte-stable weights across
separate Python processes (differing ``PYTHONHASHSEED``), per the repo's
LP-determinism / canonical-sort convention.  Dict-iteration order over
profiles / inflows must not leak into the numeric weights.

Runs the computation in two subprocesses with different hash seeds and
compares the serialized (region, base, rep, weight) triples at round(4).
"""
from __future__ import annotations

import json
import subprocess
import sys

_WORKER = r"""
import json
from flextool.representative_periods.preprocess import compute_region_weight_maps

period_length = 2
timestep_keys = [f"t{i}" for i in range(10)]

def series(vals):
    return [(f"t{i}", float(v)) for i, v in enumerate(vals)]

# Two regions with deliberately different net-load shapes so their base
# periods project onto the shared rep anchors at different weights.
profiles = {
    "wind_ie":   series([0.05, 0.95, 0.50, 0.55, 0.10, 0.20, 0.90, 0.85, 0.45, 0.40]),
    "wind_scot": series([0.90, 0.10, 0.20, 0.85, 0.80, 0.30, 0.15, 0.60, 0.55, 0.25]),
}
inflows = {
    "elec_ie":   series([2, 9, 5, 6, 1, 3, 8, 7, 4, 5]),
    "elec_scot": series([9, 1, 2, 8, 7, 3, 2, 6, 5, 4]),
}
region_profiles = {"IE": ["wind_ie"], "SCOT": ["wind_scot"]}
region_nodes = {"IE": ["elec_ie"], "SCOT": ["elec_scot"]}
rep_indices = [0, 2, 4]

maps = compute_region_weight_maps(
    profiles, inflows, region_profiles, region_nodes,
    rep_indices, timestep_keys, period_length,
)

def flatten(m):
    out = []
    for i, base in enumerate(m.indexes):
        inner = m.values[i]
        for j, rep in enumerate(inner.indexes):
            out.append([str(base), str(rep), round(float(inner.values[j]), 4)])
    return sorted(out)

result = {r: flatten(m) for r, m in maps.items()}
print(json.dumps(result, sort_keys=True))
"""


def _run(seed: str) -> dict:
    proc = subprocess.run(
        [sys.executable, "-c", _WORKER],
        capture_output=True, text=True,
        env={"PYTHONHASHSEED": seed, "PATH": "/usr/bin:/bin"},
    )
    assert proc.returncode == 0, (
        f"worker (seed={seed}) failed:\n{proc.stderr}"
    )
    return json.loads(proc.stdout.strip().splitlines()[-1])


def test_region_weight_maps_cross_process_deterministic() -> None:
    a = _run("0")
    b = _run("1")
    assert a == b, (
        "per-region weight maps diverged across PYTHONHASHSEED — "
        f"seed0={a}\nseed1={b}"
    )
    # Sanity: both regions produced a matrix and they are not identical
    # (distinct region signals => distinct projections).
    assert set(a) == {"IE", "SCOT"}
    assert a["IE"] != a["SCOT"], (
        "the two regions collapsed to identical weight matrices — the "
        "region filtering is not distinguishing them"
    )
