"""call_tool: caps clamped before validation, schema problems and unknown tools as messages, a fault without its
exception text, and results cut to max_chars with a truncated note."""
from __future__ import annotations

import json

from flextool.assistant import caps, registry
from flextool.assistant.caps import CapArg
from flextool.assistant.registry import Fit, call_tool, make_tool


def _tool(handler, **kw):
    return make_tool(name="t", title="t", description="A test tool.", properties={
        "n": {"type": "integer"}, "max_rows": {"type": "integer"}, "max_chars": {"type": "integer"}},
        handler=handler, area="inputs", small_tier=False, **kw)


def _with(monkeypatch, tool) -> None:
    monkeypatch.setattr(registry, "_tools", {tool.name: tool})


def test_caps_are_clamped_and_defaulted(ctx, monkeypatch) -> None:
    seen = {}
    _with(monkeypatch, _tool(lambda c, a: seen.update(a) or {}, caps={"max_rows": CapArg("max_rows")}))
    assert call_tool(ctx, "t", {"max_rows": 10 ** 9}).ok
    assert seen["max_rows"] == caps.CEILINGS.max_rows
    assert call_tool(ctx, "t", {}).ok
    assert seen["max_rows"] == caps.PRESETS["medium"].max_rows


def test_bad_arguments_and_unknown_tools_are_messages(ctx, monkeypatch) -> None:
    _with(monkeypatch, _tool(lambda c, a: {}))
    out = call_tool(ctx, "t", {"n": "three"})
    assert not out.ok and out.message.startswith("The arguments of t are not valid: n: ")
    out = call_tool(ctx, "t", {"other": 1})
    assert not out.ok and "other" in out.message
    out = call_tool(ctx, "nope", {})
    assert not out.ok and out.message.startswith("There is no tool named 'nope'.")


def test_a_fault_hides_the_exception_text(ctx, monkeypatch, capsys) -> None:
    def broken(c, a):
        raise ValueError(f"cannot read {ctx.project_path}/secret.txt")
    _with(monkeypatch, _tool(broken))
    out = call_tool(ctx, "t", {})
    assert not out.ok
    assert out.message == ("FlexTool could not answer this request (ValueError). This is a fault in the assistant "
                           "tools; please report it.")
    assert "secret.txt" in capsys.readouterr().err


def test_long_results_drop_whole_rows(ctx, monkeypatch) -> None:
    rows = [{"i": i, "text": "x" * 50} for i in range(500)]
    _with(monkeypatch, _tool(lambda c, a: Fit({"rows": rows}, rows_key="rows"), caps={"max_chars": CapArg("max_chars")}))
    out = call_tool(ctx, "t", {"max_chars": 2000})
    assert out.ok and len(out.text) <= 2000
    data = json.loads(out.text)
    k = len(data["rows"])
    assert data["rows"] == rows[:k] and 0 < k < 500
    assert data["truncated"] == f"showing {k} of 500 rows — add filters, group_by or top_n"
