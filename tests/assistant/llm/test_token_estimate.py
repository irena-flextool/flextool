"""The digit-aware token estimate stays above real tokenizers on result JSON (P3 §8 test 11; critique B2).

``data/result_floats.json`` is a committed query result (two nodes × time steps, values with 10 significant
digits, as query_result rounds them), 5 995 characters.  Its token counts were computed once, offline, by an
independent agent with the real tokenizers — Qwen2.5-14B-Instruct's ``tokenizer.json`` and o200k (gpt-oss-20b's
``tokenizer.json``, cross-checked with tiktoken's ``o200k_base``), ``add_special_tokens=False`` — and are stored
here as constants: no tokenizer is a dependency.  If the fixture changes, its counts must be computed again the
same way.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

from flextool.assistant.budget import approx_tokens

FIXTURE = Path(__file__).parent / "data" / "result_floats.json"
FIXTURE_SHA256 = "32626ce8d537451166d2cc83745793c9e05b3ba64a075f01e09f7e79f989a214"
QWEN25_TOKENS = 4_208
O200K_TOKENS = 2_671


def test_the_fixture_is_the_counted_one():
    assert hashlib.sha256(FIXTURE.read_bytes()).hexdigest() == FIXTURE_SHA256


def test_the_estimate_is_not_below_the_real_tokenizers():
    text = FIXTURE.read_text(encoding="utf-8")
    estimate = approx_tokens(text, "json")
    assert estimate >= QWEN25_TOKENS
    assert estimate >= O200K_TOKENS
    # …and does not over-estimate o200k by more than the critique's measured 1.8×.
    assert estimate <= 1.8 * O200K_TOKENS
