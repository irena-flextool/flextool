"""The temporary strict check of the c24 stage-4 seed prune (``FLEXTOOL_SEED_STRICT=1``).

``flextool.engine.preprocess._seed_strict`` poisons the seeds of the prune candidates before the DB overlay and
fails the load if one was read or survived.  These tests check the mechanism itself:

* (a) the poison raises on use, records every use, and leaves ``isinstance`` alone;
* (b) a load whose DB overlay does nothing fails ``check`` (the seeds survive): the fail-before of the mechanism;
* (c) a touch that an ``except Exception`` swallows still fails ``check``;
* (d) under ``_cascade_gate()`` nothing is armed;
* (e) a strict load gives the same FlexData as a plain load, and the env var unset arms nothing.

The scenario is one where at least 3 candidates are seeded on the last solve (from the S4-1b armed log).
"""
from __future__ import annotations

import dataclasses

import polars as pl
import pytest

from flextool.engine.preprocess import _seed_strict
from flextool.engine.preprocess import flex_data as flex_data_mod
from flextool.engine.preprocess._seed_strict import SEED_PRUNE_CANDIDATES, SeedPoisonError, _Poison
from flextool.engine.preprocess.flex_data import FlexData, _cascade_gate
from tests._support.fields import field_diff
from tests._support.load import load_scenario

pytestmark = pytest.mark.solver

# test_a_lot: the last solve seeds p_profile_value and the three delay fields of the candidates left after S4-4
# (multi_year, used until S4-3, seeds only p_profile_value of them).
_SCENARIO = "test_a_lot"


@pytest.fixture
def strict(monkeypatch):
    monkeypatch.setenv("FLEXTOOL_SEED_STRICT", "1")
    monkeypatch.delenv("FLEXTOOL_SEED_STRICT_LOG", raising=False)


def _armed_in_load(monkeypatch) -> list[dict]:
    """Record what ``arm`` returns during the next loads."""
    seen: list[dict] = []
    real_arm = _seed_strict.arm

    def spy(flex_data, solve):
        armed = real_arm(flex_data, solve)
        seen.append(dict(armed))
        return armed

    monkeypatch.setattr(_seed_strict, "arm", spy)
    return seen


def test_poison_raises_records_and_keeps_isinstance_safe(monkeypatch):
    monkeypatch.delenv("FLEXTOOL_SEED_STRICT_LOG", raising=False)
    p = _Poison("p_profile_value", "s1")
    uses = [
        ("height", lambda: p.height),
        ("frame", lambda: p.frame),
        ("bool", lambda: bool(p)),
        ("len", lambda: len(p)),
        ("iter", lambda: list(p)),
        ("eq", lambda: p == 1),
        ("mul", lambda: p * 2.0),
        ("hasattr", lambda: hasattr(p, "frame")),
    ]
    for label, use in uses:
        with pytest.raises(SeedPoisonError, match="p_profile_value"):
            use()
    assert isinstance(p, pl.DataFrame) is False
    assert type(p) is _Poison
    assert len(p._touches) == len(uses)
    assert [w for w, _ in p._touches][:2] == [".height", ".frame"]


def test_check_raises_when_the_overlay_does_not_replace_the_seeds(scenario_run, strict, monkeypatch):
    run = scenario_run(_SCENARIO)
    seen = _armed_in_load(monkeypatch)
    monkeypatch.setattr(flex_data_mod, "_apply_db_overrides", lambda *a, **k: None)
    with pytest.raises(SeedPoisonError) as exc:
        load_scenario(run)
    assert len(seen) == 1 and len(seen[0]) >= 3, seen
    msg = str(exc.value)
    for name in seen[0]:
        assert f"not replaced: {name}" in msg, msg


def test_check_raises_on_a_swallowed_touch(scenario_run, strict, monkeypatch):
    run = scenario_run(_SCENARIO)
    seen = _armed_in_load(monkeypatch)
    real = flex_data_mod._apply_db_overrides
    swallowed: list[str] = []

    def touching_overrides(flex_data, *args, **kwargs):
        name = next(iter(seen[-1]))
        try:
            getattr(flex_data, name).height
        except Exception:
            swallowed.append(name)
        return real(flex_data, *args, **kwargs)

    monkeypatch.setattr(flex_data_mod, "_apply_db_overrides", touching_overrides)
    with pytest.raises(SeedPoisonError) as exc:
        load_scenario(run)
    assert len(swallowed) == 1
    msg = str(exc.value)
    assert f"touched {swallowed[0]} (.height)" in msg, msg
    assert "  not replaced: " not in msg, msg


def test_cascade_gate_arms_nothing(scenario_run, strict, monkeypatch):
    run = scenario_run(_SCENARIO)
    seen = _armed_in_load(monkeypatch)
    with _cascade_gate():
        data = load_scenario(run)
        seeded = dataclasses.replace(data, **{n: pl.DataFrame({"x": [1]}) for n in SEED_PRUNE_CANDIDATES})
        assert _seed_strict.arm(seeded, "s") == {}
    assert seen == [{}, {}]  # the load, then the direct call
    assert not any(isinstance(v, _Poison) for v in vars(data).values())
    assert all(isinstance(getattr(seeded, n), pl.DataFrame) for n in SEED_PRUNE_CANDIDATES)


def test_strict_load_equals_plain_load(scenario_run, monkeypatch):
    run = scenario_run(_SCENARIO)
    monkeypatch.delenv("FLEXTOOL_SEED_STRICT", raising=False)
    calls: list[str] = []
    real_arm = _seed_strict.arm
    monkeypatch.setattr(_seed_strict, "arm", lambda fd, s: calls.append(s) or real_arm(fd, s))
    plain = load_scenario(run)
    assert calls == [], "arm ran with FLEXTOOL_SEED_STRICT unset"

    monkeypatch.setenv("FLEXTOOL_SEED_STRICT", "1")
    seen = _armed_in_load(monkeypatch)
    strict_data = load_scenario(run)
    assert len(seen) == 1 and len(seen[0]) >= 3, seen
    diffs = {f.name: d for f in dataclasses.fields(FlexData)
             if (d := field_diff(getattr(strict_data, f.name), getattr(plain, f.name))) is not None}
    assert diffs == {}
