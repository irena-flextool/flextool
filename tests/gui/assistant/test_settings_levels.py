"""The level and the free plan in the AI assistant's settings (P3 §8 test 9; §3.3, §4.6, §5).

* The level list offers only the levels the model allows, with the reason; without facts from an earlier check
  of the model it adds the "may be lowered" line, which goes away once FlexTool has seen the model's context.
* The free plan sits in the level's rows and has three states: "Automatic (on / off for this model)" — on only for
  an OpenRouter model whose name ends ``:free`` — On and Off; each is saved as ``free_tier`` null, true or false
  and shown again when the dialog opens.

Expected texts are copied from the design (P3 §3.3, §5), never read from ``strings.py``.

Run headless:  xvfb-run -a python -m pytest tests/gui/assistant --run-gui
"""
from __future__ import annotations

import pytest

from flextool.assistant import probes
from flextool.assistant.probes import ModelFacts
from flextool.assistant.providers import PRESETS
from tests.gui.assistant import test_assistant_settings_dialog as base
from tests.gui.assistant.tk_helpers import pump

pytestmark = pytest.mark.gui

# The dialog fixtures of the settings tests (the keys of the environment and earlier probes are not seen).
open_dialog = base.open_dialog
no_key_variables = base.no_key_variables
no_recent_probes = base.no_recent_probes
_exists = base._exists
_settings_file = base._settings_file

HAIKU = "anthropic/claude-haiku-4-5"
HAIKU_PLACED = ("This model is placed at the basic level from its name (a small model of its family (haiku)). "
                "Higher levels are not available for it.")
MAY_BE_LOWERED = "The level comes from the model name; it may be lowered when FlexTool sees the model's context size."
FREE_MODEL = "meta-llama/llama-3.3-70b-instruct:free"


def _show_model(dialog, host, model: str) -> None:
    dialog.model_var.set(model)
    pump(host, timeout=0.2)


def test_only_the_allowed_levels_with_the_reason_and_the_may_be_lowered_line(open_dialog, host):
    dialog = open_dialog()
    dialog.select_preset("openrouter")
    _show_model(dialog, host, HAIKU)
    assert list(dialog.level_box["values"]) == ["Automatic (basic for this model)", "basic"]
    assert dialog.level_reason.cget("text") == HAIKU_PLACED + "\n" + MAY_BE_LOWERED

    probes.remember_facts("openrouter", PRESETS["openrouter"].base_url, HAIKU,
                          ModelFacts(context_window=200_000, source="openrouter"))
    _show_model(dialog, host, "")
    _show_model(dialog, host, HAIKU)
    assert list(dialog.level_box["values"]) == ["Automatic (basic for this model)", "basic"]
    assert dialog.level_reason.cget("text") == HAIKU_PLACED


def test_the_free_plan_has_three_states_and_is_saved(open_dialog, host, user_config_in_tmp):
    dialog = open_dialog()
    dialog.select_preset("openrouter")
    assert dialog.free_box.winfo_parent() in {str(w) for w in dialog.level_frame.winfo_children()}
    assert dialog.free_label.cget("text") == (
        "Free plan: fewer steps for each question, to stay within the provider's request limits")
    _show_model(dialog, host, FREE_MODEL)
    assert list(dialog.free_box["values"]) == ["Automatic (on for this model)", "On", "Off"]
    assert dialog.free_var.get() == "Automatic (on for this model)" and dialog.chosen_free_tier() is None
    _show_model(dialog, host, "x-ai/grok-4")
    assert list(dialog.free_box["values"]) == ["Automatic (off for this model)", "On", "Off"]

    for label, saved in (("Off", False), ("On", True), ("Automatic (off for this model)", None)):
        dialog.free_var.set(label)
        dialog.free_box.event_generate("<<ComboboxSelected>>")
        dialog.save()
        pump(host, lambda d=dialog: not _exists(d))
        assert _settings_file(user_config_in_tmp)["free_tier"] is saved
        dialog = open_dialog()
        pump(host, timeout=0.2)
        assert dialog.free_var.get() == label
        assert dialog.chosen_free_tier() is saved


def test_another_provider_keeps_the_free_plan_choice(open_dialog, host):
    dialog = open_dialog()
    dialog.select_preset("openrouter")
    _show_model(dialog, host, FREE_MODEL)
    dialog.free_var.set("Off")
    dialog.free_box.event_generate("<<ComboboxSelected>>")
    dialog.select_preset("gemini")
    pump(host, timeout=0.2)
    assert dialog.free_var.get() == "Off"
    assert list(dialog.free_box["values"]) == ["Automatic", "On", "Off"]
