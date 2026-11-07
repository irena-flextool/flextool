"""Token budgets by level and window (P3 §8 test 3; §4).

Expected numbers are worked by hand from P3 §3.3, §4.1 and §4.3 (written out next to each assertion), not read
from the code: the window classes give the result caps and the compaction rhythm, the small-window caps follow
the §4.3 formula, a turn at those caps stays within 95 % of a 16 384-token window on float-heavy results, the
output limit is at most a quarter of a known window, the free plan is automatic only for OpenRouter ``:free``
models, and the compaction settings read ``null`` and P1's stored 6 / 2 as automatic.
"""
from __future__ import annotations

import json
import math
import random
from pathlib import Path

import pytest

from flextool.assistant import budget, caps, settings
from flextool.assistant.budget import LEVEL_BUDGETS, approx_tokens, fit_caps, resolve_budget

SMALL = caps.preset("small")        # 50 rows, 6 000 characters, 5 results, 100 items
MEDIUM = caps.preset("medium")      # 200, 20 000, 8, 300
LARGE = caps.preset("large")        # 1 000, 60 000, 10, 1 000


def _plain(c: caps.Caps) -> tuple[int, int, int, int]:
    return c.max_rows, c.max_chars, c.max_results, c.max_items


def test_the_small_window_caps_worked_check():
    """P3 §4.3 with the shipped basic budget (prefix 4 500, output 4 096, three planned calls) at 16 384 tokens,
    keep 0: 16 384 − 4 500 − 4 096 − 1 638.4 − min(4 000, 2 457.6) = 3 692; per call 3 692 / 1 / 3 = 1 230.7
    tokens; × 1.5 = 1 846 characters, raised to the 2 000 floor; rows max(10, round(50 × 2 000 / 6 000 = 16.7))
    = 17; results max(3, round(1.67)) = 3; items max(3, round(33.3)) = 33."""
    assert _plain(fit_caps(16_384, LEVEL_BUDGETS["small"], 4_096, 0)) == (17, 2_000, 3, 33)
    # 30 000 tokens: 30 000 − 4 500 − 4 096 − 3 000 − 4 000 = 14 404; / 3 = 4 801.3; × 1.5 = 7 202 → the small
    # preset's 6 000 characters; everything at the small preset.
    assert _plain(fit_caps(30_000, LEVEL_BUDGETS["small"], 4_096, 0)) == (50, 6_000, 5, 100)
    # 24 000 tokens, keep 0: 24 000 − 4 500 − 4 096 − 2 400 − 3 600 = 9 404; / 3 = 3 134.7; × 1.5 = 4 702;
    # rows round(50 × 0.78367) = 39; results round(3.92) = 4; items round(78.4) = 78.
    assert _plain(fit_caps(24_000, LEVEL_BUDGETS["small"], 4_096, 0)) == (39, 4_702, 4, 78)


def _float_rows(n: int, seed: int) -> dict:
    rng = random.Random(seed)
    rows = [[node, "p2020", f"t{i:04d}", float(f"{rng.uniform(-900, 900):.10g}")]
            for i in range(1, n + 1) for node in ("west", "east")]
    return {"run": "coal_1", "table": "node_dt_e", "columns": ["node", "period", "time", "value"], "rows": rows,
            "unit": {"value": "MW"}, "total_rows": len(rows), "notes": []}


def test_a_turn_at_the_caps_fits_a_16k_window():
    """A basic turn at 16 384 tokens: the prefix budget, three results cut to the caps (float-heavy rows, as
    query_result sends them), the answer and the text history stay within 95 % of the window."""
    b = resolve_budget("small", 16_384, "loaded", local=True)
    results = [caps.fit_json(_float_rows(400, seed), b.result_caps.max_chars, lambda s: s, rows_key="rows")
               for seed in (1, 2, 3)]
    assert all(len(r) <= 2_000 for r in results)
    history = min(4_000, 0.15 * 16_384)
    total = LEVEL_BUDGETS["small"].prefix_tokens + sum(approx_tokens(r, "json") for r in results) + \
        b.max_output_tokens + history
    assert total <= 0.95 * 16_384


@pytest.mark.parametrize(("tier", "window", "caps_expected", "every", "keep"), [
    # 12 288 – 32 767: the fitted caps (at most small), every question starts with earlier results shortened.
    ("small", 16_384, (17, 2_000, 3, 33), 1, 0),
    # 32 768 – 65 535: small, 4 / 2.
    ("small", 40_000, _plain(SMALL), 4, 2),
    ("medium", 40_000, _plain(SMALL), 4, 2),
    ("large", 40_000, _plain(SMALL), 4, 2),
    # 65 536 – 199 999: the level's preset, at most medium, 6 / 2.
    ("small", 128_000, _plain(SMALL), 6, 2),
    ("medium", 128_000, _plain(MEDIUM), 6, 2),
    ("large", 128_000, _plain(MEDIUM), 6, 2),
    # 200 000 and more: the level's preset.
    ("small", 400_000, _plain(SMALL), 6, 2),
    ("medium", 400_000, _plain(MEDIUM), 6, 2),
    ("large", 400_000, _plain(LARGE), 6, 2),
])
def test_window_classes(tier, window, caps_expected, every, keep):
    b = resolve_budget(tier, window, "model", local=False)
    assert _plain(b.result_caps) == caps_expected
    assert b.result_caps.ceilings is not None and _plain(b.result_caps.ceilings) == caps_expected
    assert (b.compaction_every, b.keep_recent_turns) == (every, keep)
    assert b.fit


def test_below_the_minimum_window_there_is_no_budget():
    with pytest.raises(ValueError):
        resolve_budget("small", 12_287, "loaded", local=True)


@pytest.mark.parametrize(("tier", "window", "expected"), [
    ("small", 16_384, 4_096),       # min(4 096, 4 096)
    ("small", 12_288, 3_072),       # 12 288 // 4
    ("medium", 32_768, 8_000),      # min(8 000, 8 192)
    ("medium", 20_000, 5_000),
    ("large", 65_536, 16_000),      # min(16 000, 16 384)
    ("large", 40_000, 10_000),
])
def test_the_output_limit_is_at_most_a_quarter_of_a_known_window(tier, window, expected):
    assert resolve_budget(tier, window, "model", local=False).max_output_tokens == expected


def test_an_assumed_window():
    # A remote model whose window nobody reports: 128 000 assumed, no pre-send fit, the level's output limit.
    remote = resolve_budget("large", None, None, local=False)
    assert (remote.window, remote.window_source, remote.fit, remote.max_output_tokens) == (
        128_000, "assumed", False, 16_000)
    assert _plain(remote.result_caps) == _plain(MEDIUM)
    # A local one: 16 384 assumed, and the fit runs (Ollama drops the start of an over-long prompt).
    local = resolve_budget("small", None, None, local=True)
    assert (local.window, local.fit, local.compaction_every) == (16_384, True, 1)


def test_result_cap_override_stays_under_the_ceiling():
    b = resolve_budget("large", 400_000, "model", local=False, result_cap="small")
    assert _plain(b.result_caps) == _plain(SMALL) and _plain(b.result_caps.ceilings) == _plain(LARGE)
    b = resolve_budget("small", 40_000, "model", local=False, result_cap="large")
    assert _plain(b.result_caps) == _plain(SMALL)


def test_a_call_is_clamped_to_the_context_ceiling():
    rule = caps.CapArg("max_rows", maximum=None)
    ceiling = resolve_budget("small", 16_384, "loaded", local=True).result_caps
    assert rule.resolve(5_000, ceiling) == 17           # the context's 17 rows, below the global 5 000
    assert rule.resolve(5_000, SMALL) == 5_000           # without a context ceiling: the global ceiling
    assert rule.resolve(None, ceiling) == 17


def test_free_plan():
    # Automatic: on only for an OpenRouter model whose id ends ":free" (P3-O2).
    assert budget.free_tier_on("openrouter", "meta-llama/llama-3.3-70b-instruct:free", None)
    assert not budget.free_tier_on("openrouter", "meta-llama/llama-3.3-70b-instruct", None)
    assert not budget.free_tier_on("gemini", "gemini-2.5-flash", None)
    assert not budget.free_tier_on("openrouter", "x:free", False)
    assert budget.free_tier_on("gemini", "gemini-2.5-flash", True)
    # Its numbers (§4.1 brackets): rounds 3 / 4 / 6, calls per question 3 / 8 / 12, per step 3 / 4 / 6; result caps
    # at most medium.
    for tier, numbers in (("small", (3, 3, 3)), ("medium", (4, 8, 4)), ("large", (6, 12, 6))):
        b = resolve_budget(tier, 400_000, "model", local=False, free_tier=True)
        assert (b.level.max_rounds, b.level.max_calls_per_turn, b.level.max_calls_per_round) == numbers
        assert b.free_tier
    assert _plain(resolve_budget("large", 400_000, "model", local=False, free_tier=True).result_caps) == \
        _plain(MEDIUM)


def test_level_numbers():
    """P3 §4.1: rounds 3 / 8 / 12, calls per question 3 / 16 / 30, per step 3 / 8 / 8, output 4 096 / 8 000 /
    16 000, planned calls 3 / 6 / 8, smallest window 12 288 / 32 768 / 65 536."""
    got = {t: (b.max_rounds, b.max_calls_per_turn, b.max_calls_per_round, b.max_output_tokens, b.planned_calls,
               b.min_window) for t, b in LEVEL_BUDGETS.items()}
    assert got == {"small": (3, 3, 3, 4_096, 3, 12_288), "medium": (8, 16, 8, 8_000, 6, 32_768),
                   "large": (12, 30, 8, 16_000, 8, 65_536)}


def test_approx_tokens():
    # 10 digits + ceil(0 / 3.2) + 4; letters: ceil(32 / 3.2) + 4 = 14; JSON: ceil(22 / 2.2) + 4 = 14.
    assert approx_tokens("0123456789") == 14
    assert approx_tokens("a" * 32) == 14
    assert approx_tokens("{" * 22, "json") == 14
    # Non-ASCII characters one token each: "Määrä" = 3 + ceil(2 / 3.2) + 4 = 8.
    assert approx_tokens("Määrä") == 8


def test_token_correction_keeps_the_running_maximum():
    corr = budget.TokenCorrection()
    corr.update(None, 1_000)
    assert corr.factor == 1.0
    corr.update(800, 1_000)                  # below the estimate: never lowers it
    assert corr.factor == 1.0
    corr.update(1_300, 1_000)
    assert corr.factor == pytest.approx(1.3)
    corr.update(1_100, 1_000)
    assert corr.factor == pytest.approx(1.3)


def test_compaction_settings_are_automatic(tmp_path: Path):
    root = tmp_path / "config"
    root.mkdir()
    path = root / "settings.json"
    for stored, expected in (({}, (None, None)), ({"compaction_every": None, "keep_recent_turns": None}, (None, None)),
                             ({"compaction_every": 6, "keep_recent_turns": 2}, (None, None)),     # P1's /save
                             ({"compaction_every": 3, "keep_recent_turns": 1}, (3, 1)),
                             ({"compaction_every": 3, "keep_recent_turns": 2}, (3, None))):
        path.write_text(json.dumps({"format": 1, **stored}), encoding="utf-8")
        s = settings.load_settings(root)
        assert (s.compaction_every, s.keep_recent_turns) == expected, stored
    path.write_text(json.dumps({"format": 1, "compaction_every": 0, "free_tier": "yes"}), encoding="utf-8")
    s = settings.load_settings(root)
    assert (s.compaction_every, s.free_tier) == (None, None) and len(s.notes) == 2
    # An override wins over the window's rhythm; null leaves it to the window.
    b = resolve_budget("small", 16_384, "loaded", local=True, compaction_every=3, keep_recent_turns=1)
    assert (b.compaction_every, b.keep_recent_turns) == (3, 1)
    s = settings.Settings()
    assert s.to_json()["compaction_every"] is None and s.to_json()["free_tier"] is None


def test_the_free_plan_hint_is_remembered_once_per_route(tmp_path: Path):
    root = tmp_path / "config"
    assert settings.mark_free_tier_suggested("gemini|https://example.test", root) is True
    assert settings.mark_free_tier_suggested("gemini|https://example.test", root) is False
    assert settings.mark_free_tier_suggested("openrouter|https://example.test", root) is True
    s = settings.load_settings(root)
    assert s.free_tier_suggested == ["gemini|https://example.test", "openrouter|https://example.test"]
    # Saving settings read before a hint keeps the hint shown.
    older = settings.Settings()
    settings.save_settings(older, root)
    assert settings.load_settings(root).free_tier_suggested == s.free_tier_suggested
    assert math.isclose(budget.FIT_SHARE, 0.95)
