"""Every text the chat shows is plain English (P1 §15 test 6, the presets' and messages' part).

No abbreviations a FlexTool user may not know, no web links except the presets' own addresses (D11), and no
promise about how a service handles the data.
"""
from __future__ import annotations

import re

import pytest

from flextool.assistant import llm_errors, net
from flextool.assistant.providers import PRESETS

ABBREVIATIONS = ("LLM", "ZDR", "DB", "LP", "MCP")
#: Promises about the data no text makes ("safety" as a refusal reason is not a promise).
PROMISES = (r"\bnot used for training", r"\bprivate\b", r"\bsafe\b", r"\bguarantee")
#: The one place an abbreviation may appear: the literal field a knob report quotes.
LITERAL_FIELDS = re.compile(r"`[^`]*`")
URL = re.compile(r"https?://[\w.\[\]:-]+[^\s)'\",]*")


def _texts() -> list[tuple[str, str]]:
    out = [(f"MESSAGES[{k}]", v) for k, v in llm_errors.MESSAGES.items()]
    out += [(f"refusal category {k}", v) for k, v in llm_errors.REFUSAL_CATEGORIES.items()]
    out.append(("refusal other", llm_errors.REFUSAL_OTHER))
    out += [("base URL credentials", net.BASE_URL_CREDENTIALS), ("base URL scheme", net.BASE_URL_SCHEME)]
    for preset in PRESETS.values():
        out.append((f"{preset.id} label", preset.label))
        out += [(f"{preset.id} account setting", t) for t in preset.account_knobs]
        for knob in preset.request_knobs:
            for name in ("label", "report", "note", "relaxed"):
                text = getattr(knob, name)
                if text:
                    out.append((f"{preset.id}.{knob.id} {name}", text))
    return out


@pytest.mark.parametrize("where,text", _texts(), ids=[w for w, _ in _texts()])
def test_plain_english(where, text):
    prose = LITERAL_FIELDS.sub("", text)
    for word in ABBREVIATIONS:
        assert not re.search(rf"\b{word}\b", prose, re.IGNORECASE), f"{where}: {word}"
    for promise in PROMISES:
        assert not re.search(promise, text, re.IGNORECASE), f"{where}: {promise}"
    base_urls = {p.base_url for p in PRESETS.values() if p.base_url}
    for url in URL.findall(text):
        assert url in base_urls or url.startswith("http://localhost"), f"{where}: link {url}"


def test_the_zdr_report_is_the_literal_field():
    report = PRESETS["openrouter"].knob("zdr").report
    assert "`provider.zdr: true`" in report
    assert PRESETS["openrouter"].knob("zdr").note == (
        "Only use providers that keep no copy of your requests. This narrows the list of models you can use, free "
        "models especially.")
