"""The AI assistant's settings dialog (P2 §13.2 test 9; §8.4, §9.3; P3-O7/O12 level list; P1-O10 hint).

* A provider change updates the address (fixed or editable), the privacy check boxes and the account list; no
  label shows a web address (only the address field does).
* OpenRouter's "data collection" is on by default and its "keep no copy" setting off, with its note; turning the
  first off asks, "Keep on" checks it again, "Turn off" keeps it off and Save writes it.
* The level list offers "Automatic" and only the levels at or below the model's, the higher ones too (marked) for
  a name FlexTool does not recognise; the chosen level and the parallel questions are saved.
* Keeping conversations for 0 days asks first; a wrong number is refused.
* The model list comes from the backend (a fake here), a failure shows P1's text, a missing key says so.
* The key status names the environment variable; the key file needs its question; the keyring stores directly.
* A background result that arrives after the dialog closed, or after the provider changed, is dropped (m16).
* Save writes P1's ``settings.json`` (keeping unknown keys) and calls ``on_saved``.

Expected texts are copied from the design (P2 §8.4, §9.3, P3 §3.3, P1-O10), never read from ``strings.py``.

Run headless:  xvfb-run -a python -m pytest tests/gui/assistant --run-gui
"""
from __future__ import annotations

import json
import queue
import threading
import tkinter as tk
from pathlib import Path
from tkinter import ttk

import pytest

from flextool.assistant import keys, probes
from flextool.assistant.llm_adapter import ModelEntry
from flextool.assistant.llm_errors import ErrorContext, TimedOut, make_error
from flextool.assistant.probes import ModelFacts
from flextool.assistant.providers import PRESETS
from flextool.assistant.tiers import NAME_ONLY
from flextool.gui.assistant.settings_dialog import AssistantSettingsDialog
from tests.assistant.llm.fake_backend import FakeBackend
from tests.gui.assistant.tk_helpers import pump, widgets

pytestmark = pytest.mark.gui


@pytest.fixture(autouse=True)
def no_key_variables(monkeypatch: pytest.MonkeyPatch) -> None:
    """The keys of the developer's environment are not seen."""
    for preset in PRESETS.values():
        if preset.key_env:
            monkeypatch.delenv(preset.key_env, raising=False)


@pytest.fixture(autouse=True)
def no_recent_probes(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(probes, "_RECENT", {})


class MainQueue:
    """The main window's ``post_to_main``: callbacks queued from any thread, run when the test drains them."""

    def __init__(self) -> None:
        self.calls: queue.SimpleQueue = queue.SimpleQueue()

    def post(self, fn, *args) -> None:  # noqa: ANN001 — the main window's signature
        self.calls.put((fn, args))

    def drain(self) -> int:
        n = 0
        while True:
            try:
                fn, args = self.calls.get_nowait()
            except queue.Empty:
                return n
            fn(*args)
            n += 1


@pytest.fixture
def open_dialog(host: tk.Toplevel, user_config_in_tmp: Path):
    made: list[AssistantSettingsDialog] = []

    def make(**kwargs) -> AssistantSettingsDialog:  # noqa: ANN003
        kwargs.setdefault("config_root", user_config_in_tmp)
        post = kwargs.pop("post_to_main", None)
        on_saved = kwargs.pop("on_saved", None)
        dialog = AssistantSettingsDialog(host, post, on_saved, **kwargs)
        made.append(dialog)
        pump(host, lambda: dialog.winfo_viewable())
        return dialog

    yield make
    for dialog in made:
        for child in list(dialog.winfo_children()) if _exists(dialog) else []:
            if isinstance(child, tk.Toplevel):
                child.destroy()
        if _exists(dialog):
            dialog.alive = True
            dialog.saving = False
            dialog.close()


def _exists(widget: tk.Misc) -> bool:
    try:
        return bool(widget.winfo_exists())
    except tk.TclError:
        return False


def _texts(dialog: tk.Misc) -> list[str]:
    out = []
    for widget in widgets(dialog):
        if isinstance(widget, (ttk.Label, ttk.Checkbutton, ttk.Radiobutton, ttk.Button, ttk.Labelframe)):
            out.append(str(widget.cget("text")))
    return out


def _key_status(dialog: AssistantSettingsDialog, host: tk.Misc) -> str:
    pump(host, lambda: dialog.key_status.cget("text") not in ("", "Checking the key…"))
    return str(dialog.key_status.cget("text"))


def _write_settings(config: Path, data: dict) -> None:
    config.mkdir(parents=True, exist_ok=True)
    (config / "settings.json").write_text(json.dumps(data), encoding="utf-8")


def _settings_file(config: Path) -> dict:
    return json.loads((config / "settings.json").read_text(encoding="utf-8"))


def _open_confirm(dialog: AssistantSettingsDialog, host: tk.Misc):
    pump(host, lambda: dialog.confirm is not None and dialog.confirm.winfo_viewable())
    return dialog.confirm


# --- the provider ---------------------------------------------------------------------------------------------
def test_provider_change_updates_address_knobs_and_accounts(open_dialog, host):
    dialog = open_dialog()
    dialog.select_preset("openai")
    pump(host, timeout=0.2)
    assert dialog.address_var.get() == "https://api.openai.com/v1"
    assert "readonly" in dialog.address_entry.state()
    assert dialog.address_fixed_label.winfo_ismapped()
    assert dialog.address_fixed_label.cget("text") == "(fixed)"
    assert set(dialog.knob_vars) == {"store"}
    assert dialog.knob_vars["store"].get() is True
    assert "Set in your provider console:" in _texts(dialog)
    assert "• zero data retention, which needs approval from OpenAI" in str(dialog.account_label.cget("text"))
    assert dialog.indicator_label.cget("text") == "Data goes to: OpenAI (api.openai.com)."

    dialog.select_preset("ollama")
    pump(host, timeout=0.2)
    assert "readonly" not in dialog.address_entry.state()
    assert not dialog.address_fixed_label.winfo_ismapped()
    assert dialog.address_var.get() == "http://localhost:11434/v1"
    assert dialog.knob_vars == {}
    texts = _texts(dialog)
    assert "Nothing to set: the endpoint is on this computer." in texts
    assert "Checked again when you send: a cloud model sends data off this computer." in texts
    assert "Set in your provider console:" not in texts

    dialog.select_preset("openai_compatible")
    pump(host, timeout=0.2)
    assert dialog.address_var.get() == ""
    assert dialog.indicator_label.cget("text") == "Enter the address of the endpoint."
    dialog.address_var.set("https://models.example.test/v1")
    pump(host, timeout=0.2)
    assert dialog.indicator_label.cget("text") == "Data goes to: Custom endpoint (models.example.test)."

    for preset_id in PRESETS:
        dialog.select_preset(preset_id)
        pump(host, timeout=0.2)
        for text in _texts(dialog):
            assert "http" not in text and "www." not in text, (preset_id, text)


def test_openrouter_privacy_defaults_and_keep_on(open_dialog, host, user_config_in_tmp):
    dialog = open_dialog()
    dialog.select_preset("openrouter")
    pump(host, timeout=0.2)
    assert dialog.knob_vars["data_collection"].get() is True
    assert dialog.knob_vars["zdr"].get() is False
    assert ("Only use providers that keep no copy of your requests. This narrows the list of models you can use, "
            "free models especially.") in _texts(dialog)

    dialog.knob_buttons["data_collection"].invoke()
    confirm = _open_confirm(dialog, host)
    assert confirm.message_label.cget("text") == (
        "Turn off “Use only providers that do not collect the data of your requests”? Requests will then be sent "
        "without this setting, and the chat window will show that the privacy setting is relaxed.")
    assert confirm.cancel_button.cget("text") == "Keep on"
    assert confirm.action_button.cget("text") == "Turn off"
    confirm.cancel_button.invoke()
    pump(host, timeout=0.2)
    assert dialog.knob_vars["data_collection"].get() is True

    dialog.knob_buttons["data_collection"].invoke()
    _open_confirm(dialog, host).action_button.invoke()
    pump(host, timeout=0.2)
    assert dialog.knob_vars["data_collection"].get() is False
    dialog.knob_buttons["zdr"].invoke()           # turning one on asks nothing
    pump(host, timeout=0.2)
    assert dialog.confirm is None
    dialog.model_var.set("vendor/some-model")
    dialog.save()
    pump(host, lambda: not _exists(dialog))
    knobs = _settings_file(user_config_in_tmp)["provider"]["knobs"]
    assert knobs["openrouter.data_collection"] == "allow"
    assert knobs["openrouter.zdr"] is True


# --- the level --------------------------------------------------------------------------------------------------
def test_level_list_is_capped_and_level_and_parallel_are_saved(open_dialog, host, user_config_in_tmp):
    dialog = open_dialog()
    dialog.select_preset("openai")
    dialog.model_var.set("gpt-5-mini")
    pump(host, timeout=0.2)
    assert list(dialog.level_box["values"]) == ["Automatic (basic for this model)", "basic"]
    assert dialog.level_reason.cget("text") == (
        "This model is placed at the basic level from its name (a small model of its family (mini)). Higher "
        "levels are not available for it.\n" + NAME_ONLY)

    dialog.model_var.set("my-finetune")
    pump(host, timeout=0.2)
    assert list(dialog.level_box["values"]) == ["Automatic (standard for this model)", "basic", "standard",
                                                "advanced (at your own risk)"]
    assert dialog.level_reason.cget("text") == (
        "FlexTool does not recognise this model, so its level is a guess. A higher level may give unreliable "
        "answers.\n" + NAME_ONLY)

    dialog.model_var.set("gpt-5.1")
    pump(host, timeout=0.2)
    assert list(dialog.level_box["values"]) == ["Automatic (advanced for this model)", "basic", "standard",
                                                "advanced"]
    dialog.level_var.set("standard")
    dialog.level_box.event_generate("<<ComboboxSelected>>")
    dialog.parallel_var.set("3")
    dialog.save()
    pump(host, lambda: not _exists(dialog))
    data = _settings_file(user_config_in_tmp)
    assert data["provider"]["tier"] == "standard"
    assert data["provider"]["model"] == "gpt-5.1"
    assert data["parallel_requests"] == 3


def test_a_saved_level_above_the_model_is_lowered(open_dialog, host, user_config_in_tmp):
    _write_settings(user_config_in_tmp,
                    {"format": 1, "provider": {"preset": "openai", "model": "gpt-5-mini", "tier": "advanced"}})
    dialog = open_dialog()
    pump(host, timeout=0.2)
    assert dialog.level_var.get() == "Automatic (basic for this model)"
    assert ("The level chosen before (advanced) is above this model's level, so the automatic level is used."
            in dialog.level_reason.cget("text"))
    assert dialog.chosen_level() is None


def test_the_context_flextool_last_saw_lowers_the_level(open_dialog, host, user_config_in_tmp):
    """P3 §3.3: the settings never probe; the facts of the last probe (kept in the user cache) set the ceiling."""
    probes.remember_facts("ollama", "http://localhost:11434/v1", "example:70b",
                          ModelFacts(context_window=131_072, loaded_context=16_384, source="ollama"))
    dialog = open_dialog()
    dialog.select_preset("ollama")
    dialog.model_var.set("example:70b")
    pump(host, timeout=0.2)
    assert list(dialog.level_box["values"]) == ["Automatic (basic for this model)", "basic"]
    assert dialog.level_reason.cget("text") == (
        "This model is placed at the standard level from its size (a model of about 70 billion parameters).\n"
        "This model runs with a 16384-token context, so the highest level is basic. A larger context allows a "
        "higher level. Set OLLAMA_CONTEXT_LENGTH or the model's context length in its load settings.")


def test_parallel_out_of_range_is_refused(open_dialog, host, user_config_in_tmp):
    dialog = open_dialog()
    dialog.parallel_var.set("5")
    dialog.save()
    pump(host, timeout=0.2)
    assert dialog.error_label.cget("text") == "Enter a whole number from 1 to 4."
    assert not (user_config_in_tmp / "settings.json").exists()


# --- conversations ----------------------------------------------------------------------------------------------
def test_retention_zero_asks_first(open_dialog, host, user_config_in_tmp):
    saved: list[int] = []
    dialog = open_dialog(on_saved=lambda: saved.append(1))
    dialog.retention_var.set("abc")
    dialog.save()
    pump(host, timeout=0.2)
    assert dialog.error_label.cget("text") == "Enter a whole number of days from 0 to 3650."

    dialog.retention_var.set("0")
    dialog.save()
    confirm = _open_confirm(dialog, host)
    assert confirm.message_label.cget("text") == (
        "No chats will be saved, and saved ones will be deleted when FlexTool closes.")
    confirm.cancel_button.invoke()
    pump(host, timeout=0.2)
    assert not (user_config_in_tmp / "settings.json").exists()
    assert saved == []

    dialog.save()
    _open_confirm(dialog, host).action_button.invoke()
    pump(host, lambda: saved == [1])
    assert _settings_file(user_config_in_tmp)["conversation_expiry_days"] == 0
    assert not _exists(dialog)


def test_fewer_days_name_the_number(open_dialog, host):
    dialog = open_dialog()
    dialog.retention_var.set("7")
    dialog.save()
    assert _open_confirm(dialog, host).message_label.cget("text") == (
        "Saved chats older than 7 days will be deleted when FlexTool closes.")


# --- the model list ---------------------------------------------------------------------------------------------
def test_model_list_from_the_backend_and_its_failures(open_dialog, host, monkeypatch, fake_backends):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-environment")
    backend = FakeBackend([], models=[ModelEntry("model-a"), ModelEntry("model-b", "a note")])
    fake_backends.append(backend)
    made = []

    def factory(preset, conn):  # noqa: ANN001
        made.append(conn)
        return backend

    dialog = open_dialog(backend_factory=factory)
    dialog.select_preset("openai")
    assert _key_status(dialog, host) == (
        "Using the key from the environment variable OPENAI_API_KEY. It is used instead of a stored key.")
    dialog.refresh_models()
    pump(host, lambda: dialog.models_label.cget("text") == "2 models")
    assert list(dialog.model_box["values"]) == ["model-a", "model-b"]
    assert backend.closed
    assert made[0].api_key == "sk-test-environment"
    pump(host, lambda: "disabled" not in dialog.refresh_button.state())

    def failing(preset, conn):  # noqa: ANN001
        ctx = ErrorContext(provider="OpenAI", read_timeout=30.0)

        class Broken(FakeBackend):
            def list_models(self):
                raise make_error(TimedOut, "timeout", ctx)

        fake = Broken([])
        fake_backends.append(fake)
        return fake

    dialog._backend_factory = failing
    dialog.refresh_models()
    pump(host, lambda: dialog.models_label.cget("text") != "Fetching the model list…")
    assert dialog.models_label.cget("text") == (
        "The model list could not be fetched: OpenAI did not answer within 30 seconds. Type the model name instead.")

    monkeypatch.delenv("OPENAI_API_KEY")
    dialog.select_preset("anthropic")
    pump(host, lambda: "disabled" not in dialog.refresh_button.state())
    dialog.refresh_models()
    pump(host, lambda: dialog.models_label.cget("text") != "Fetching the model list…")
    assert dialog.models_label.cget("text") == "Store a key first to see the model list."


def test_the_model_list_is_kept_per_provider_and_address(open_dialog, host, monkeypatch, fake_backends,
                                                         user_config_in_tmp, user_cache_in_tmp):
    """The last fetched list shows at once in the next dialog, with its date; the saved model is the selected
    one before any fetch; the list is fetched off the Tk thread; the saved file holds no key."""
    import datetime

    from flextool.assistant.settings import ProviderSettings, Settings, save_settings

    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-environment")
    backend = FakeBackend([], models=[ModelEntry("model-a"), ModelEntry("model-b")])
    fake_backends.append(backend)
    threads: list[threading.Thread] = []

    def factory(preset, conn):  # noqa: ANN001
        threads.append(threading.current_thread())
        return backend

    dialog = open_dialog(backend_factory=factory)
    dialog.select_preset("openai")
    assert list(dialog.model_box["values"]) == [] and dialog.models_label.cget("text") == ""
    dialog.refresh_models()
    pump(host, lambda: dialog.models_label.cget("text") == "2 models")
    assert threads and threads[0] is not threading.main_thread()
    saved_file = user_cache_in_tmp / "assistant" / "model_lists.json"
    raw = saved_file.read_text(encoding="utf-8")
    assert "sk-test-environment" not in raw
    data = json.loads(raw)
    assert list(data["lists"]) == ["openai|https://api.openai.com/v1"]
    assert data["lists"]["openai|https://api.openai.com/v1"]["ids"] == ["model-a", "model-b"]
    dialog.close()

    save_settings(Settings(provider=ProviderSettings(preset="openai", model="model-b")), user_config_in_tmp)
    again = open_dialog(backend_factory=factory)
    today = datetime.date.today()
    assert list(again.model_box["values"]) == ["model-a", "model-b"]
    assert again.models_label.cget("text") == (
        f"2 models (list from {today.day}.{today.month}.{today.year}; Refresh to update)")
    assert again.model_box.get() == "model-b" and again.model_box.current() == 1
    assert len(threads) == 1                                   # nothing fetched to show it
    # Another provider, or another address of the same one, has its own list (none saved here).
    again.select_preset("ollama")
    assert list(again.model_box["values"]) == [] and again.models_label.cget("text") == ""
    again.select_preset("openai")
    assert list(again.model_box["values"]) == ["model-a", "model-b"]


def test_the_list_shows_the_latest_versions_and_keeps_the_chosen_older_one(open_dialog, host, monkeypatch,
                                                                           fake_backends, user_config_in_tmp,
                                                                           user_cache_in_tmp):
    """A remote service's list shows the latest version of each model; the chosen model, when older, stays in the
    list marked "(older version)" and is not replaced; "Show all models" shows the whole list, which is what is
    saved.  A server on this computer lists all its models (the box is off)."""
    from flextool.assistant.settings import ProviderSettings, Settings, save_settings

    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    backend = FakeBackend([], models=[ModelEntry("gemini-2.5-flash"), ModelEntry("gemini-2.5-pro"),
                                      ModelEntry("gemini-3-flash")])
    fake_backends.append(backend)
    save_settings(Settings(provider=ProviderSettings(preset="gemini", model="gemini-2.5-flash")), user_config_in_tmp)
    dialog = open_dialog(backend_factory=lambda preset, conn: backend)
    assert dialog.show_all_button.cget("text") == "Show all models"
    assert not dialog.show_all_var.get() and "disabled" not in dialog.show_all_button.state()
    dialog.refresh_models()
    pump(host, lambda: dialog.models_label.cget("text") == "2 models")
    assert list(dialog.model_box["values"]) == ["gemini-2.5-flash (older version)", "gemini-2.5-pro",
                                                "gemini-3-flash"]
    assert dialog.model_box.get() == "gemini-2.5-flash"
    dialog.model_box.current(0)                                  # picking the marked entry gives the bare name
    assert dialog.model_var.get() == "gemini-2.5-flash"
    data = json.loads((user_cache_in_tmp / "assistant" / "model_lists.json").read_text(encoding="utf-8"))
    assert data["lists"]["gemini|https://generativelanguage.googleapis.com/v1beta/openai"]["ids"] == [
        "gemini-2.5-flash", "gemini-2.5-pro", "gemini-3-flash"]
    dialog.show_all_button.invoke()
    assert list(dialog.model_box["values"]) == ["gemini-2.5-flash", "gemini-2.5-pro", "gemini-3-flash"]
    assert dialog.models_label.cget("text") == "3 models"
    dialog.select_preset("ollama")
    assert "disabled" in dialog.show_all_button.state()


def test_partly_on_the_processor_hint(open_dialog, host):
    probes.remember_facts("ollama", "http://localhost:11434/v1", "big-model:70b",
                          ModelFacts(loaded_context=32768, partly_on_cpu=True))
    dialog = open_dialog()
    dialog.select_preset("ollama")
    dialog.model_var.set("small-model:7b")
    pump(host, timeout=0.2)
    assert dialog.cpu_hint.cget("text") == ""
    dialog.model_var.set("big-model:70b")
    pump(host, timeout=0.2)
    assert dialog.cpu_hint.cget("text") == "This model doesn't fit fully in graphics memory and will be slower."


# --- the key ----------------------------------------------------------------------------------------------------
def test_keyring_stores_directly_and_the_file_needs_its_question(open_dialog, host, fake_keyring,
                                                                 user_config_in_tmp):
    dialog = open_dialog()
    dialog.select_preset("mistral")
    assert _key_status(dialog, host) == "No key stored."
    dialog.storage_var.set("keyring")
    dialog.key_entry.insert(0, "mistral-key-1")
    dialog.store_key()
    pump(host, lambda: dialog.key_status.cget("text") == "Using the key stored in the system keyring.")
    assert fake_keyring.passwords[("flextool-assistant", "mistral")] == "mistral-key-1"
    assert dialog.key_entry.get() == ""

    key_file = user_config_in_tmp / "keys.json"
    dialog.storage_var.set("file")
    dialog.key_entry.insert(0, "mistral-key-2")
    dialog.store_key()
    confirm = _open_confirm(dialog, host)
    assert confirm.message_label.cget("text") == (
        f"The key will be stored unencrypted at {key_file}. Anyone who can read your files can read it. Store it "
        "there?")
    assert confirm.action_button.cget("text") == "Store in the file"
    confirm.cancel_button.invoke()
    pump(host, timeout=0.2)
    assert not key_file.exists()
    dialog.store_key()
    _open_confirm(dialog, host).action_button.invoke()
    pump(host, lambda: dialog.key_status.cget("text") == f"Using the key stored unencrypted in {key_file}.")
    assert json.loads(key_file.read_text(encoding="utf-8"))["keys"]["mistral"] == "mistral-key-2"
    assert ("flextool-assistant", "mistral") not in fake_keyring.passwords

    dialog.forget_key()
    pump(host, lambda: dialog.key_status.cget("text") == "No key stored.")
    assert not key_file.exists()


def test_an_unusable_keyring_is_greyed_with_the_reason(open_dialog, host, monkeypatch):
    monkeypatch.setattr(keys, "keyring_status",
                        lambda timeout=None: (False, keys.KEYRING_NONE.format(backend="keyring.backends.fail")))
    dialog = open_dialog()
    pump(host, lambda: "disabled" in dialog.keyring_radio.state())
    assert dialog.storage_var.get() == "file"
    note = dialog.keyring_note.cget("text")
    assert note.startswith("There is no usable system keyring on this computer (keyring.backends.fail).")
    assert "--key-file" not in note
    assert "choose a file on this computer" in note


def test_an_optional_key_says_it_is_not_needed(open_dialog, host):
    dialog = open_dialog()
    dialog.select_preset("lmstudio")
    assert _key_status(dialog, host) == "No key is needed for this endpoint unless it asks for one."


# --- results after the dialog closed or the provider changed -------------------------------------------------------
def test_results_after_close_or_provider_change_are_dropped(open_dialog, host, monkeypatch, fake_backends):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    main = MainQueue()
    release = threading.Event()
    started = threading.Event()

    class Slow(FakeBackend):
        def list_models(self):
            started.set()
            release.wait(10)
            return [ModelEntry("late-model")]

    def factory(preset, conn):  # noqa: ANN001
        fake = Slow([])
        fake_backends.append(fake)
        return fake

    dialog = open_dialog(backend_factory=factory, post_to_main=main.post)
    dialog.select_preset("openai")
    pump(host, lambda: main.drain() >= 0 and dialog.key_status.cget("text") not in ("", "Checking the key…"))
    dialog.refresh_models()
    assert started.wait(5)
    dialog.select_preset("gemini")                  # the provider changed while the list was fetched
    release.set()
    pump(host, lambda: main.drain() >= 0 and dialog.stale_results >= 1)
    assert list(dialog.model_box["values"]) == []
    assert dialog.models_label.cget("text") == ""

    release.clear()
    started.clear()
    dialog.select_preset("openai")
    dialog.refresh_models()
    assert started.wait(5)
    before = dialog.stale_results
    dialog.cancel()
    assert not _exists(dialog)
    release.set()
    pump(host, lambda: main.drain() >= 0 and dialog.stale_results > before)


# --- save ---------------------------------------------------------------------------------------------------------
def test_save_writes_the_settings_and_calls_on_saved(open_dialog, host, user_config_in_tmp):
    _write_settings(user_config_in_tmp,
                    {"format": 1, "provider": {"preset": "openai", "model": "gpt-4o"}, "a_later_setting": {"x": 1}})
    saved: list[int] = []
    dialog = open_dialog(on_saved=lambda: saved.append(1))
    assert dialog.provider_var.get() == "OpenAI"
    assert dialog.model_var.get() == "gpt-4o"
    dialog.select_preset("gemini")
    assert dialog.model_var.get() == ""
    dialog.model_var.set("gemini-model-x")
    dialog.select_preset("ollama")
    dialog.address_var.set("http://127.0.0.1:11500/v1")
    dialog.select_preset("gemini")
    assert dialog.model_var.get() == "gemini-model-x"
    dialog.save()
    pump(host, lambda: saved == [1])
    assert not _exists(dialog)
    data = _settings_file(user_config_in_tmp)
    assert data["provider"]["preset"] == "gemini"
    assert data["provider"]["model"] == "gemini-model-x"
    assert data["provider"]["base_url"] is None
    assert data["provider"]["tier"] is None
    assert data["conversation_expiry_days"] == 30
    assert data["parallel_requests"] == 1
    assert data["a_later_setting"] == {"x": 1}


def test_an_edited_address_is_saved_and_a_wrong_one_refused(open_dialog, host, user_config_in_tmp):
    dialog = open_dialog()
    dialog.select_preset("ollama")
    dialog.address_var.set("http://user:secret@localhost:11434/v1")
    dialog.save()
    pump(host, timeout=0.2)
    assert dialog.error_label.cget("text") != ""
    assert not (user_config_in_tmp / "settings.json").exists()
    dialog.address_var.set("http://127.0.0.1:11500/v1")
    dialog.model_var.set("qwen2.5:14b")
    dialog.save()
    pump(host, lambda: not _exists(dialog))
    data = _settings_file(user_config_in_tmp)
    assert data["provider"]["base_url"] == "http://127.0.0.1:11500/v1"


def test_cancel_discards(open_dialog, host, user_config_in_tmp):
    saved: list[int] = []
    dialog = open_dialog(on_saved=lambda: saved.append(1))
    dialog.select_preset("gemini")
    dialog.model_var.set("gemini-model-x")
    dialog.cancel_button.invoke()
    pump(host, timeout=0.2)
    assert not _exists(dialog)
    assert not (user_config_in_tmp / "settings.json").exists()
    assert saved == []
