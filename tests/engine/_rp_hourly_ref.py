"""Overlays of ``lh2_three_region_rp_invest.json`` whose representative-period model has an exact hourly twin.

Each base day is an exact copy of the representative day it maps to (weight 1), so the hourly model of the
same days is the reference the RP model approximates (critique of cycle c22, §3.2).  Two experiments:

exp1 (distinct base names): one 336 h timeline; y2030 = days 1-7 (pattern A B A B A B A, A/B = fixture days
  1/2), y2040 = days 8-14 (pattern C D C D C D C, C/D = fixture days 3/4).  RP: y2030 reps t0001/t0025,
  y2040 reps t0169/t0193.
exp2 (shared base names, as rp_multi_base): one 168 h timeline, days pattern A B A B A B A (days 3/4 = A/B
  copies), both periods' bases t0001..t0145; y2030 reps t0001/t0025, y2040 reps t0049/t0073; lh2 demand made
  equal in both periods (days 1/2 values) so the two periods see the same data.

Batteries: bind_within_timeblock (daily cycle in both RP and hourly, the hourly timesets have 24 h blocks).
lh2 nodes: storage_start_end_method fix_nothing everywhere (removes the fix_start confound).
Every builder returns ``patch(data) -> data`` appending the override alternative and scenario ``custom``.
"""
import base64
import importlib.util
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def _lp_snapshot():
    """``tools/lp_snapshot.py`` (fixture overlay helpers), loaded by path."""
    name = "_lp_snapshot_for_tests"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, REPO / "tools" / "lp_snapshot.py")
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
    return sys.modules[name]


L = _lp_snapshot()


SFX = "__lh2_three_region_rp_invest"
CHAIN = ["lh2_three_region", "lh2_three_region_rp_invest"]
LH2 = ("lh2_A", "lh2_B", "lh2_C")
BAT = ("battery_A", "battery_B", "battery_C")


def _t(day: int, hour: int = 0) -> str:
    return f"t{1 + 24 * (day - 1) + hour:04d}"


def _effective_time_maps(data):
    """(cls, ent, param) -> {step: value} for every time-indexed map in the chain (last alternative wins)."""
    eff = {}
    for alt in CHAIN:
        for cls, ent, param, packed, a in data["parameter_values"]:
            if a != alt or packed[1] != "map":
                continue
            j = json.loads(base64.b64decode(packed[0]))
            if j.get("index_name") != "time" or cls in ("timeline", "timeset"):
                continue
            eff[(cls, json.dumps(ent), param)] = {str(k): v for k, v in j["data"]}
    return eff


def _remap(data, day_source: dict[int, int], *, equalise_lh2_to: dict[int, int] | None = None):
    """Values that rebuild every time map so that target day d copies source day day_source[d]."""
    vals = []
    for (cls, ent_s, param), m in _effective_time_maps(data).items():
        ent = json.loads(ent_s)
        src_map = day_source
        if equalise_lh2_to is not None and cls == "node" and ent in LH2:
            src_map = {d: equalise_lh2_to[s] for d, s in day_source.items()}
        new = {}
        for d_tgt, d_src in sorted(src_map.items()):
            for h in range(24):
                k = _t(d_src, h)
                if k in m:
                    new[_t(d_tgt, h)] = m[k]
        vals.append([cls, ent, param, L._pmap(new, "time")])
    return vals


def _weights(bases, rep_of_base):
    data = []
    for b in bases:
        r = rep_of_base[b]
        data.append([b, {"type": "map", "index_type": "str", "index_name": "representative_period", "rank": 1,
                         "data": [[r, 1.0]]}])
    return {"type": "map", "index_type": "str", "index_name": "base_period", "rank": 1, "data": data}


def _common_storage(lh2_method):
    v = [["node", n, "storage_binding_method", lh2_method] for n in LH2]
    v += [["node", n, "storage_start_end_method", "fix_nothing"] for n in LH2]
    v += [["node", n, "storage_binding_method", "bind_within_timeblock"] for n in BAT]
    v += [["node", n, "storage_start_end_method", "fix_nothing"] for n in BAT]
    return v


def _timeline(n_days):
    return [["timeline", "y2030_168h", "timestep_duration",
             L._pmap({f"t{i:04d}": 1.0 for i in range(1, 24 * n_days + 1)}, "time")]]


def _hourly_ts(_name, days):
    return L._pmap({_t(d): 24.0 for d in days}, "time")


SOLVE = "lh2_rp_invest" + SFX
PATTERN = [0, 1, 0, 1, 0, 1, 0]


def exp1(kind: str, method: str):
    """kind = 'rp' or 'hourly'; method = lh2 binding method (blended for rp, plain for hourly)."""
    def f(data):
        src = {}
        for k, p in enumerate(PATTERN):
            src[1 + k] = 1 + p          # days 1-7  <- fixture days 1/2
            src[8 + k] = 3 + p          # days 8-14 <- fixture days 3/4
        vals = _timeline(14) + _remap(data, src) + _common_storage(method)
        ents = []
        if kind == "rp":
            b30 = [_t(d) for d in range(1, 8)]
            b40 = [_t(d) for d in range(8, 15)]
            vals += [
                ["timeset", "rp_y2030" + SFX, "representative_period_weights",
                 _weights(b30, {b: (_t(1) if p == 0 else _t(2)) for b, p in zip(b30, PATTERN)})],
                ["timeset", "rp_y2040" + SFX, "representative_period_weights",
                 _weights(b40, {b: (_t(8) if p == 0 else _t(9)) for b, p in zip(b40, PATTERN)})],
                ["timeset", "rp_y2040" + SFX, "timeset_duration", L._pmap({_t(8): 24.0, _t(9): 24.0}, "time")],
            ]
        else:
            ents = [["timeset", "h2030"], ["timeset", "h2040"]]
            vals += [
                ["timeset", "h2030", "timeline", "y2030_168h"],
                ["timeset", "h2040", "timeline", "y2030_168h"],
                ["timeset", "h2030", "timeset_duration", _hourly_ts("h2030", range(1, 8))],
                ["timeset", "h2040", "timeset_duration", _hourly_ts("h2040", range(8, 15))],
                ["solve", SOLVE, "period_timeset", L._pmap({"y2030": "h2030", "y2040": "h2040"})],
            ]
        return L._overlay(data, name="custom", base="lh2_three_region_rp_invest", values=vals, entities=ents)
    return f


def exp2(kind: str, method: str):
    def f(data):
        src = {1 + k: 1 + p for k, p in enumerate(PATTERN)}     # days 1-7 <- fixture days 1/2 (A B A B ...)
        # lh2 demand: the same in both periods (fixture days 1/2 values; days 3/4 differ in the fixture)
        vals = _timeline(7) + _remap(data, src, equalise_lh2_to={1: 1, 2: 2}) + _common_storage(method)
        ents = []
        bases = [_t(d) for d in range(1, 8)]
        if kind == "rp":
            vals += [
                ["timeset", "rp_y2030" + SFX, "representative_period_weights",
                 _weights(bases, {b: (_t(1) if p == 0 else _t(2)) for b, p in zip(bases, PATTERN)})],
                ["timeset", "rp_y2040" + SFX, "representative_period_weights",
                 _weights(bases, {b: (_t(3) if p == 0 else _t(4)) for b, p in zip(bases, PATTERN)})],
            ]
        else:
            ents = [["timeset", "h168"]]
            vals += [
                ["timeset", "h168", "timeline", "y2030_168h"],
                ["timeset", "h168", "timeset_duration", _hourly_ts("h168", range(1, 8))],
                ["solve", SOLVE, "period_timeset", L._pmap({"y2030": "h168", "y2040": "h168"})],
            ]
        return L._overlay(data, name="custom", base="lh2_three_region_rp_invest", values=vals, entities=ents)
    return f


BLENDED = {"wp": "bind_within_period_blended_weights", "ws": "bind_within_solve_blended_weights",
           "fo": "bind_forward_only_blended_weights"}
PLAIN = {"wp": "bind_within_period", "ws": "bind_within_solve", "fo": "bind_forward_only"}
EXPERIMENTS = {"e1": exp1, "e2": exp2}


def one_block(fn, kind, method):
    """Hourly variant with ONE 168 h block per period: bind_within_timeblock then cycles within the period,
    the same as bind_within_period."""
    base = fn(kind, method)

    def f(data):
        data = base(data)
        for r in data["parameter_values"]:
            if (r[4] == "custom" and r[0] == "timeset" and r[2] == "timeset_duration"
                    and r[1] in ("h2030", "h2040", "h168")):
                first = {"h2030": 1, "h2040": 8, "h168": 1}[r[1]]
                r[3] = L._pack(L._pmap({_t(first): 168.0}, "time"))
        return data
    return f
