"""The AI assistant's settings dialog (P2 §8.4): provider, address, key, model, level, parallel questions, data
use and how long conversations are kept.

```
Provider [Google Gemini ▾]   Address [https://…] (fixed)
Key      Using the key from the environment variable GEMINI_API_KEY. …   [•••••••] [Store key] [Forget key]
         Store in: (o) system keyring ( ) a file on this computer (not encrypted)
Model    [gemini-… ▾] [Refresh list] 34 models
Level    [Automatic (standard for this model) ▾]   the reason
         Free plan: fewer steps for each question, … [Automatic (off for this model) ▾]
Questions answered at the same time [1] (more chats wait their turn)
── Data use ──  Data goes to: … · Sent with every request: [x] {knob} {note} · Set in your provider console: • …
── Conversations ──  Keep conversations for [30] days. 0 means conversations are not saved.
                                                                                         [Cancel] [Save]
```

The dialog never asks a server, the keyring or the store on the Tk thread: the key status, the keyring check, the
model list (:meth:`Backend.list_models`, the backend closed afterwards), storing or forgetting a key and the save
run on short daemon threads.  Their results come back through *post_to_main* (the main window's, or the dialog's
own queue when none is given), each with the dialog's generation number, and are dropped when the dialog is closed
or the provider or address changed since (critique m16).  The saved settings file is read once at open (P2-D2).

* **Level** (P3-O7, P3-O12): the list offers "Automatic (…)" and only the levels at or below the model's automatic
  level (:func:`~flextool.assistant.tiers.resolve_tier`: its name or size, and the context FlexTool last saw it
  run with); for a model FlexTool does not recognise, the higher levels too, marked "(at your own risk)".  The
  lines below say why, and that the level may be lowered when no context is known yet.
* **Free plan** (P3 §4.6), in the level's rows: "Automatic (on / off for this model)" — on only for an OpenRouter
  model whose name ends ``:free`` — then On and Off.  Saved with the rest (``free_tier``: null, true or false).
* **Data use**: one check box per privacy setting the requests carry (unchecking one that is on by default asks
  first; "Keep on" checks it again), the settings only the account can change as a list (no links).
* **Save** writes P1's ``settings.json`` (keeping keys this FlexTool does not know), then calls *on_saved()* —
  the chat window's :meth:`~flextool.gui.assistant.chat_window.ChatWindow.settings_changed`.  Cancel discards
  everything except a key already stored or forgotten.
"""
from __future__ import annotations

import copy
import logging
import queue
import threading
import tkinter as tk
from collections.abc import Callable
from pathlib import Path
from tkinter import ttk
from typing import TYPE_CHECKING, Any

from flextool.gui.assistant import strings
from flextool.gui.assistant.notices import ConfirmDialog, _AssistantDialog

if TYPE_CHECKING:
    from flextool.assistant.providers import Knob, Preset

logger = logging.getLogger(__name__)

PostToMain = Callable[..., None]
#: The dialog's own queue is drained this often when no *post_to_main* is given.
POLL_MS = 50
MAX_RETENTION_DAYS = 3650
#: Internal level name -> the name users see.
_USER_LEVEL = {"small": "basic", "medium": "standard", "large": "advanced"}


def _default_backend_factory(preset: Preset, conn: Any) -> Any:
    from flextool.assistant.llm_adapter import make_backend

    return make_backend(preset, conn)


def _knob_value(knob: Knob, on: bool) -> object:
    """How the settings keep *knob* on or off: its sent text value where it has one (``"deny"``), else a flag."""
    value = knob.on_value if on else knob.off_value
    return value if isinstance(value, str) else on


class AssistantSettingsDialog(_AssistantDialog):
    """The settings dialog over *parent*.  *post_to_main(fn, *args)* runs ``fn(*args)`` on the Tk thread (from
    any thread); *on_saved()* is called on the Tk thread after a successful Save.  *backend_factory(preset,
    connection)* builds the backend that lists the models; *config_root* is the assistant's settings folder
    (``None``: the user's)."""

    def __init__(self, parent: tk.Misc, post_to_main: PostToMain | None, on_saved: Callable[[], None] | None, *,
                 backend_factory: Callable[[Preset, Any], Any] | None = None,
                 config_root: Path | None = None) -> None:
        from flextool.assistant.providers import PRESETS
        from flextool.assistant.settings import Settings, load_settings

        super().__init__(parent, strings.SETTINGS_TITLE)
        self._own_queue: queue.SimpleQueue | None = None
        self._poll_id: str | None = None
        if post_to_main is None:
            own: queue.SimpleQueue = queue.SimpleQueue()
            self._own_queue = own

            def post_to_main(fn: Callable[..., None], *args: Any) -> None:
                own.put((fn, args))
        self._post_to_main = post_to_main
        self._on_saved = on_saved
        self._backend_factory = backend_factory or _default_backend_factory
        self._config_root = config_root
        self.alive = True
        self.generation = 0
        #: Background results dropped because the dialog closed or the provider changed (for the tests).
        self.stale_results = 0
        self.confirm: ConfirmDialog | None = None
        self.saving = False
        try:
            self._settings = load_settings(config_root)
        except Exception:  # noqa: BLE001 — an unreadable settings file reads as the defaults
            logger.exception("The AI assistant settings could not be read")
            self._settings = Settings()
        provider = self._settings.provider
        self._presets = list(PRESETS.values())
        self._knobs: dict[str, object] = dict(provider.knobs)
        self._models: dict[str, str] = {provider.preset: provider.model or ""} if provider.preset else {}
        self._addresses: dict[str, str] = {}
        if provider.preset and provider.base_url:
            self._addresses[provider.preset] = provider.base_url
        self._level_choice: str | None = provider.tier
        self._level_values: list[str | None] = [None]
        self._free_choice: bool | None = self._settings.free_tier
        self._has_key: bool | None = None
        self._preset: Preset | None = None
        self._knob_frame_children: list[tk.Misc] = []
        self.knob_vars: dict[str, tk.BooleanVar] = {}
        self.knob_buttons: dict[str, ttk.Checkbutton] = {}
        self._build()
        if self._own_queue is not None:
            self._poll_id = self.after(POLL_MS, self._poll)
        self.bind("<Destroy>", self._on_destroy, add="+")
        self._check_keyring()
        self.select_preset(provider.preset if provider.preset in PRESETS else None)
        self.show()

    # --- layout -----------------------------------------------------------------------------------------------
    def _build(self) -> None:
        cw, lh = self.cw, self.lh
        notes = list(self._settings.notes)
        if self._settings.read_only:
            notes.append(strings.SETTINGS_NEWER)
        self.notes_label = ttk.Label(self.body, text="\n".join(notes), wraplength=self.wrap, justify="left")
        if notes:
            self.notes_label.pack(anchor="w", fill="x", pady=(0, lh // 2))
        form = ttk.Frame(self.body)
        form.pack(fill="x")
        form.columnconfigure(1, weight=1)

        def label(row: int, text: str) -> None:
            ttk.Label(form, text=text).grid(row=row, column=0, sticky="nw", padx=(0, cw), pady=(lh // 3, 0))

        # Provider and address.
        label(0, strings.SET_PROVIDER)
        row = ttk.Frame(form)
        row.grid(row=0, column=1, sticky="ew", pady=(lh // 3, 0))
        self.provider_var = tk.StringVar(self)
        self.provider_box = ttk.Combobox(row, textvariable=self.provider_var, state="readonly",
                                         values=[p.label for p in self._presets], width=20)
        self.provider_box.pack(side="left")
        self.provider_box.bind("<<ComboboxSelected>>", self._on_provider_selected)
        ttk.Label(row, text=strings.SET_ADDRESS).pack(side="left", padx=(cw * 2, cw))
        self.address_var = tk.StringVar(self)
        self.address_entry = ttk.Entry(row, textvariable=self.address_var, width=44)
        self.address_entry.pack(side="left", fill="x", expand=True)
        self.address_entry.bind("<FocusOut>", lambda _e: self._address_committed())
        self.address_entry.bind("<Return>", lambda _e: (self._address_committed(), "break")[1])
        self.address_fixed_label = ttk.Label(row, text=strings.SET_ADDRESS_FIXED)
        self.address_var.trace_add("write", lambda *_a: self._update_route())

        # Key.
        label(1, strings.SET_KEY)
        key = ttk.Frame(form)
        key.grid(row=1, column=1, sticky="ew", pady=(lh // 3, 0))
        self.key_status = ttk.Label(key, text="", wraplength=self.wrap, justify="left")
        self.key_status.pack(anchor="w", fill="x")
        entry_row = ttk.Frame(key)
        entry_row.pack(anchor="w", fill="x")
        self.key_entry = ttk.Entry(entry_row, show="•", width=30)
        self.key_entry.pack(side="left")
        self.store_button = ttk.Button(entry_row, text=strings.SET_STORE_KEY, command=self.store_key)
        self.store_button.pack(side="left", padx=(cw, 0))
        self.forget_button = ttk.Button(entry_row, text=strings.SET_FORGET_KEY, command=self.forget_key)
        self.forget_button.pack(side="left", padx=(cw // 2, 0))
        storage_row = ttk.Frame(key)
        storage_row.pack(anchor="w", fill="x")
        ttk.Label(storage_row, text=strings.SET_STORE_IN).pack(side="left")
        self.storage_var = tk.StringVar(self, value=self._settings.provider.key_storage)
        self.keyring_radio = ttk.Radiobutton(storage_row, text=strings.SET_STORE_KEYRING, value="keyring",
                                             variable=self.storage_var)
        self.keyring_radio.pack(side="left", padx=(cw, 0))
        self.file_radio = ttk.Radiobutton(storage_row, text=strings.SET_STORE_FILE, value="file",
                                          variable=self.storage_var)
        self.file_radio.pack(side="left", padx=(cw, 0))
        self.keyring_note = ttk.Label(key, text="", wraplength=self.wrap, justify="left")
        self.keyring_note.pack(anchor="w", fill="x")

        # Model and level.
        label(2, strings.SET_MODEL)
        model = ttk.Frame(form)
        model.grid(row=2, column=1, sticky="ew", pady=(lh // 3, 0))
        model_row = ttk.Frame(model)
        model_row.pack(anchor="w", fill="x")
        self.model_var = tk.StringVar(self)
        self.model_box = ttk.Combobox(model_row, textvariable=self.model_var, width=36)
        self.model_box.pack(side="left")
        self.refresh_button = ttk.Button(model_row, text=strings.SET_REFRESH, command=self.refresh_models)
        self.refresh_button.pack(side="left", padx=(cw, 0))
        self.models_label = ttk.Label(model_row, text="", wraplength=cw * 36, justify="left")
        self.models_label.pack(side="left", padx=(cw, 0))
        self.cpu_hint = ttk.Label(model, text="", wraplength=self.wrap, justify="left")
        self.cpu_hint.pack(anchor="w", fill="x")
        self.model_var.trace_add("write", lambda *_a: self._model_changed())

        label(3, strings.SET_LEVEL)
        level = ttk.Frame(form)
        level.grid(row=3, column=1, sticky="ew", pady=(lh // 3, 0))
        self.level_var = tk.StringVar(self)
        self.level_box = ttk.Combobox(level, textvariable=self.level_var, state="readonly", width=36)
        self.level_box.pack(anchor="w")
        self.level_box.bind("<<ComboboxSelected>>", self._on_level_selected)
        self.level_reason = ttk.Label(level, text="", wraplength=self.wrap, justify="left")
        self.level_reason.pack(anchor="w", fill="x")
        #: The level rows: the list, the reason and the free plan (P3 §5).
        self.level_frame = level
        free = ttk.Frame(level)
        free.pack(anchor="w", fill="x", pady=(lh // 3, 0))
        self.free_label = ttk.Label(free, text=strings.SET_FREE_PLAN, wraplength=self.wrap, justify="left")
        self.free_label.pack(anchor="w", fill="x")
        self.free_var = tk.StringVar(self)
        self.free_box = ttk.Combobox(free, textvariable=self.free_var, state="readonly", width=36)
        self.free_box.pack(anchor="w")
        self.free_box.bind("<<ComboboxSelected>>", self._on_free_selected)

        parallel = ttk.Frame(form)
        parallel.grid(row=4, column=0, columnspan=2, sticky="ew", pady=(lh // 3, 0))
        ttk.Label(parallel, text=strings.SET_PARALLEL).pack(side="left", padx=(0, cw))
        self.parallel_var = tk.StringVar(self, value=str(self._settings.parallel_requests))
        self.parallel_box = ttk.Spinbox(parallel, from_=1, to=4, increment=1, width=4,
                                        textvariable=self.parallel_var)
        self.parallel_box.pack(side="left")
        ttk.Label(parallel, text=strings.SET_PARALLEL_NOTE).pack(side="left", padx=(cw, 0))

        # Data use.
        data = ttk.Labelframe(self.body, text=strings.SET_DATA_USE, padding=(cw, lh // 3))
        data.pack(fill="x", pady=(lh, 0))
        self.indicator_label = ttk.Label(data, text="", wraplength=self.wrap, justify="left")
        self.indicator_label.pack(anchor="w", fill="x")
        self.ollama_note = ttk.Label(data, text=strings.SET_OLLAMA_CHECK, wraplength=self.wrap, justify="left")
        self.knob_frame = ttk.Frame(data)
        self.knob_frame.pack(anchor="w", fill="x")

        # Conversations.
        conv = ttk.Labelframe(self.body, text=strings.SET_CONVERSATIONS, padding=(cw, lh // 3))
        conv.pack(fill="x", pady=(lh // 2, 0))
        keep = ttk.Frame(conv)
        keep.pack(anchor="w", fill="x")
        ttk.Label(keep, text=strings.SET_KEEP).pack(side="left", padx=(0, cw))
        self.retention_var = tk.StringVar(self, value=str(self._settings.conversation_expiry_days))
        self.retention_box = ttk.Spinbox(keep, from_=0, to=MAX_RETENTION_DAYS, increment=1, width=6,
                                         textvariable=self.retention_var)
        self.retention_box.pack(side="left")
        ttk.Label(keep, text=strings.SET_KEEP_DAYS).pack(side="left", padx=(cw, 0))
        ttk.Label(conv, text=strings.SET_KEEP_NOTE, wraplength=self.wrap, justify="left").pack(anchor="w")

        self.error_label = ttk.Label(self.body, text="", wraplength=self.wrap, justify="left")
        self.error_label.pack(anchor="w", fill="x", pady=(lh // 2, 0))
        buttons = self.buttons()
        self.save_button = ttk.Button(buttons, text=strings.SAVE, command=self.save)
        self.save_button.pack(side="right")
        self.cancel_button = ttk.Button(buttons, text=strings.CANCEL, command=self.cancel)
        self.cancel_button.pack(side="right", padx=(0, cw))
        if self._settings.read_only:
            self.save_button.state(["disabled"])

    def initial_focus(self) -> tk.Misc | None:
        return self.provider_box

    # --- background work and its results ------------------------------------------------------------------------
    def _run(self, fn: Callable[[], None], name: str) -> threading.Thread:
        thread = threading.Thread(target=fn, name=name, daemon=True)
        thread.start()
        return thread

    def _post(self, generation: int | None, fn: Callable[..., None], *args: Any) -> None:
        """From a background thread: run ``fn(*args)`` on the Tk thread if the dialog is still open and (for a
        *generation*) the provider and address are still the ones the work started with."""
        self._post_to_main(self._deliver, generation, fn, args)

    def _deliver(self, generation: int | None, fn: Callable[..., None], args: tuple) -> None:
        try:
            exists = self.alive and bool(self.winfo_exists())
        except tk.TclError:
            exists = False
        if not exists or (generation is not None and generation != self.generation):
            self.stale_results += 1
            return
        fn(*args)

    def _poll(self) -> None:
        self._poll_id = None
        if not self.alive or self._own_queue is None:
            return
        while True:
            try:
                fn, args = self._own_queue.get_nowait()
            except queue.Empty:
                break
            try:
                fn(*args)
            except Exception:  # noqa: BLE001 — one failed update must not stop the dialog
                logger.exception("An assistant settings update failed")
        if self.alive:
            self._poll_id = self.after(POLL_MS, self._poll)

    # --- the provider ---------------------------------------------------------------------------------------------
    def _on_provider_selected(self, _event: object = None) -> None:
        label = self.provider_var.get()
        preset = next((p for p in self._presets if p.label == label), None)
        self.select_preset(preset.id if preset is not None else None)

    def select_preset(self, preset_id: str | None) -> None:
        """Show the settings of the preset *preset_id* (``None``: no provider chosen yet)."""
        from flextool.assistant.providers import PRESETS

        old = self._preset
        if old is not None:
            self._models[old.id] = self.model_var.get().strip()
            if self._address_editable(old):
                self._addresses[old.id] = self.address_var.get().strip()
        preset = PRESETS.get(preset_id or "")
        self._preset = preset
        self.generation += 1
        self._has_key = None
        self.provider_var.set(preset.label if preset is not None else "")
        if preset is not None and self._address_editable(preset):
            self.address_entry.state(["!disabled", "!readonly"])
            self.address_fixed_label.pack_forget()
            self.address_var.set(self._addresses.get(preset.id, preset.base_url or ""))
        else:
            self.address_var.set((preset.base_url or "") if preset is not None else "")
            self.address_entry.state(["readonly"])
            if preset is not None:
                self.address_fixed_label.pack(side="left", padx=(self.cw // 2, 0))
            else:
                self.address_fixed_label.pack_forget()
        enabled = ["!disabled"] if preset is not None else ["disabled"]
        for widget in (self.key_entry, self.store_button, self.forget_button, self.model_box, self.refresh_button,
                       self.level_box):
            widget.state(enabled)
        if preset is not None:
            self.level_box.state(["readonly"])
        self.model_box["values"] = ()
        self.models_label.configure(text="")
        self.model_var.set(self._models.get(preset.id, "") if preset is not None else "")
        self._build_knobs()
        self._update_route()
        self._check_key()
        self._model_changed()

    @staticmethod
    def _address_editable(preset: Preset) -> bool:
        return preset.id in ("ollama", "lmstudio", "openai_compatible")

    def _address(self) -> str:
        """The address as typed (the preset's own when the field is fixed)."""
        return self.address_var.get().strip()

    def _checked_address(self) -> str | None:
        from flextool.assistant.net import BaseUrlRefused, check_base_url

        try:
            return check_base_url(self._address()) if self._address() else None
        except BaseUrlRefused:
            return None

    def _key_endpoint(self) -> str | None:
        preset = self._preset
        if preset is None or not preset.key_optional:
            return None
        return self._address() or preset.base_url

    def _address_committed(self) -> None:
        """The address was changed and left: the key and the model list belong to the new address."""
        preset = self._preset
        if preset is None or not self._address_editable(preset):
            return
        if self._addresses.get(preset.id) == self._address():
            return
        self._addresses[preset.id] = self._address()
        self.generation += 1
        self.models_label.configure(text="")
        self._check_key()
        self._model_changed()

    def _update_route(self) -> None:
        """The "Data goes to" line of what the dialog shows now (no server is asked)."""
        from flextool.assistant import endpoint

        preset = self._preset
        url = self._checked_address()
        text = ""
        if preset is not None and url is not None:
            try:
                text = endpoint.indicator_text(endpoint.route_for(preset, url, model=self.model_var.get().strip()))
            except Exception:  # noqa: BLE001 — a proxy setting that cannot be read: no line
                logger.debug("The route of the settings could not be worked out", exc_info=True)
        elif preset is not None and self._address_editable(preset) and not self._address():
            text = strings.SET_ADDRESS_NEEDED
        self.indicator_label.configure(text=text)
        if preset is not None and preset.id == "ollama":
            self.ollama_note.pack(anchor="w", fill="x", after=self.indicator_label)
        else:
            self.ollama_note.pack_forget()

    # --- privacy settings -------------------------------------------------------------------------------------
    def _build_knobs(self) -> None:
        from flextool.assistant.providers import knob_on

        for child in self._knob_frame_children:
            child.destroy()
        self._knob_frame_children = []
        self.knob_vars = {}
        self.knob_buttons = {}
        self.account_label = None
        preset = self._preset
        if preset is None:
            return
        frame = self.knob_frame

        def add(widget: tk.Misc, **pack: Any) -> None:
            widget.pack(anchor="w", fill="x", **pack)
            self._knob_frame_children.append(widget)

        if preset.request_knobs:
            add(ttk.Label(frame, text=strings.SET_SENT_WITH), pady=(self.lh // 3, 0))
        for knob in preset.request_knobs:
            try:
                on = knob_on(preset, knob, self._knobs)
            except ValueError:
                on = knob.default_on
            var = tk.BooleanVar(self, value=on)
            button = ttk.Checkbutton(frame, text=knob.label, variable=var,
                                     command=lambda k=knob: self._knob_toggled(k))
            add(button, padx=(self.cw, 0))
            self.knob_vars[knob.id] = var
            self.knob_buttons[knob.id] = button
            if knob.note:
                add(ttk.Label(frame, text=knob.note, wraplength=self.wrap - self.cw * 4, justify="left"),
                    padx=(self.cw * 4, 0))
        bullets = "\n".join(f"• {text}" for text in preset.account_knobs)
        if preset.local:
            add(ttk.Label(frame, text=strings.SET_NOTHING_LOCAL, wraplength=self.wrap, justify="left"),
                pady=(self.lh // 3, 0))
            if bullets:
                self.account_label = ttk.Label(frame, text=bullets, wraplength=self.wrap, justify="left")
                add(self.account_label)
        elif bullets:
            add(ttk.Label(frame, text=strings.SET_CONSOLE), pady=(self.lh // 3, 0))
            self.account_label = ttk.Label(frame, text=bullets, wraplength=self.wrap - self.cw, justify="left")
            add(self.account_label, padx=(self.cw, 0))

    def _knob_toggled(self, knob: Knob) -> None:
        preset = self._preset
        if preset is None:
            return
        from flextool.assistant.providers import knob_key

        var = self.knob_vars[knob.id]
        on = bool(var.get())
        if on or not knob.default_on:
            self._knobs[knob_key(preset, knob)] = _knob_value(knob, on)
            return

        def answered(turn_off: bool) -> None:
            self.confirm = None
            if turn_off:
                self._knobs[knob_key(preset, knob)] = _knob_value(knob, False)
            else:
                var.set(True)
                self._knobs[knob_key(preset, knob)] = _knob_value(knob, True)

        self.confirm = ConfirmDialog(self, strings.RELAX_TITLE, strings.RELAX_CONFIRM.format(label=knob.label),
                                     strings.RELAX_OFF, answered, cancel_text=strings.RELAX_KEEP)

    # --- the key --------------------------------------------------------------------------------------------------
    def _check_keyring(self) -> None:
        """Whether the system keyring can keep keys here (in the background)."""
        def run() -> None:
            from flextool.assistant import keys

            try:
                usable, text = keys.keyring_status()
            except Exception as exc:  # noqa: BLE001 — a keyring that cannot be asked is not usable
                usable, text = False, keys.KEYRING_NONE.format(backend=type(exc).__name__)
            self._post(None, self._show_keyring, usable, text)

        self._run(run, "flextool-assistant-keyring-status")

    def _show_keyring(self, usable: bool, text: str) -> None:
        from flextool.assistant import keys

        if usable:
            self.keyring_radio.state(["!disabled"])
            self.keyring_note.configure(text="")
            return
        self.keyring_radio.state(["disabled"])
        if self.storage_var.get() == "keyring":
            self.storage_var.set("file")
        self.keyring_note.configure(text=keys.for_window(text))

    def _check_key(self) -> None:
        """Where the key of the shown provider comes from (in the background)."""
        preset = self._preset
        if preset is None:
            self.key_status.configure(text="")
            return
        self.key_status.configure(text=strings.SET_KEY_CHECKING)
        generation = self.generation
        endpoint = self._key_endpoint()
        config_root = self._config_root

        def run() -> None:
            from flextool.assistant import keys

            try:
                key, source, note = keys.resolve_key(preset, endpoint, config_root=config_root)
            except Exception as exc:  # noqa: BLE001 — shown, nothing else depends on it
                self._post(generation, self._show_key_text, strings.KEY_PROBLEM.format(reason=exc), None)
                return
            self._post(generation, self._show_key, source.name, note, bool(key))

        self._run(run, "flextool-assistant-key-status")

    def _show_key(self, source: str, note: str | None, has_key: bool) -> None:
        from flextool.assistant import keys

        preset = self._preset
        if preset is None:
            return
        if source == "ENV":
            text = strings.KEY_FROM_ENV.format(name=preset.key_env)
        elif source == "KEYRING":
            text = strings.KEY_FROM_KEYRING
        elif source == "FILE":
            text = strings.KEY_FROM_FILE.format(path=keys.key_file_path(self._config_root))
        elif preset.key_optional:
            text = strings.KEY_OPTIONAL
        else:
            text = strings.KEY_NONE
        if note and not has_key:
            text = f"{text} {keys.for_window(note)}"
        self._show_key_text(text, has_key)

    def _show_key_text(self, text: str, has_key: bool | None) -> None:
        self._has_key = has_key
        self.key_status.configure(text=text)

    def store_key(self) -> None:
        """[Store key]: the typed key goes to the keyring, or to the key file after its question."""
        from flextool.assistant import keys

        preset = self._preset
        if preset is None:
            return
        typed = self.key_entry.get().strip()
        if not typed:
            self.key_status.configure(text=strings.SET_KEY_EMPTY)
            return
        to_file = self.storage_var.get() == "file"
        if not to_file:
            self._store(preset, typed, False)
            return

        def answered(store: bool) -> None:
            self.confirm = None
            if store:
                self._store(preset, typed, True)

        self.confirm = ConfirmDialog(
            self, strings.KEY_FILE_TITLE,
            strings.KEY_FILE_CONFIRM.format(path=keys.key_file_path(self._config_root)),
            strings.KEY_FILE_BUTTON, answered)

    def _store(self, preset: Preset, key: str, to_file: bool) -> None:
        generation = self.generation
        endpoint = self._key_endpoint()
        config_root = self._config_root
        self.key_status.configure(text=strings.SET_KEY_CHECKING)

        def run() -> None:
            from flextool.assistant import keys

            try:
                keys.store_key(preset, endpoint, key, to_file=to_file, config_root=config_root)
            except Exception as exc:  # noqa: BLE001 — KeyStoreError, a locked or newer key file
                self._post(generation, self._show_key_text, strings.KEY_STORE_FAILED.format(reason=exc),
                           self._has_key)
                return
            self._post(generation, self._key_stored)

        self._run(run, "flextool-assistant-key-store")

    def _key_stored(self) -> None:
        self.key_entry.delete(0, "end")
        self._check_key()

    def forget_key(self) -> None:
        """[Forget key]: remove the stored key from the keyring and the key file."""
        preset = self._preset
        if preset is None:
            return
        generation = self.generation
        endpoint = self._key_endpoint()
        config_root = self._config_root
        self.key_status.configure(text=strings.SET_KEY_CHECKING)

        def run() -> None:
            from flextool.assistant import keys

            try:
                note = keys.forget_key(preset, endpoint, config_root=config_root)
            except Exception as exc:  # noqa: BLE001 — a locked or newer key file
                self._post(generation, self._show_key_text, strings.KEY_PROBLEM.format(reason=exc), self._has_key)
                return
            self._post(generation, self._key_forgotten, note)

        self._run(run, "flextool-assistant-key-forget")

    def _key_forgotten(self, note: str | None) -> None:
        self._check_key()
        if note:
            from flextool.assistant import keys

            self.keyring_note.configure(text=keys.for_window(note))

    # --- the model and its level ----------------------------------------------------------------------------------
    def refresh_models(self) -> None:
        """[Refresh list]: the provider's live model list (sends the key, no case data)."""
        from flextool.assistant import endpoint
        from flextool.assistant.net import BaseUrlRefused, check_base_url

        preset = self._preset
        if preset is None:
            return
        try:
            base_url = check_base_url(self._address())
        except BaseUrlRefused as exc:
            self.models_label.configure(text=str(exc) if self._address() else strings.SET_ADDRESS_NEEDED)
            return
        self.models_label.configure(text=strings.MODELS_LOADING)
        self.refresh_button.state(["disabled"])
        generation = self.generation
        knobs = dict(self._knobs)
        config_root = self._config_root
        endpoint_for_key = self._key_endpoint()
        factory = self._backend_factory

        def run() -> None:
            from flextool.assistant import keys
            from flextool.assistant.llm_adapter import Connection
            from flextool.assistant.llm_errors import LLMError, message_for, render

            try:
                key, source, _note = keys.resolve_key(preset, endpoint_for_key, config_root=config_root)
                if preset.key_required and not key:
                    self._post(generation, self._show_models_text, strings.MODELS_NEED_KEY)
                    return
                route = endpoint.route_for(preset, base_url)
                conn = Connection(base_url=base_url, model="", api_key=key or "",
                                  key_source=source.describe(preset) if key else "", knobs=knobs, route=route)
                backend = factory(preset, conn)
                try:
                    entries = backend.list_models()
                finally:
                    backend.close()
            except LLMError as err:
                text = render("model_list", reason=message_for(err, "gui").rstrip("."))
                self._post(generation, self._show_models_text, text)
                return
            except Exception as exc:  # noqa: BLE001 — a missing library, an unexpected answer: shown, logged
                logger.warning("The model list could not be fetched", exc_info=True)
                self._post(generation, self._show_models_text,
                           strings.MODELS_FAILED_INTERNAL.format(type=type(exc).__name__))
                return
            self._post(generation, self._show_models, [entry.id for entry in entries])

        self._run(run, "flextool-assistant-model-list")
        self._watch_refresh(generation)

    def _watch_refresh(self, generation: int) -> None:
        """The button comes back when a result arrives or the provider changes."""
        def again() -> None:
            try:
                if not self.alive or not self.winfo_exists():
                    return
            except tk.TclError:
                return
            if generation != self.generation or self.models_label.cget("text") != strings.MODELS_LOADING:
                if self._preset is not None:
                    self.refresh_button.state(["!disabled"])
                return
            self.after(100, again)

        self.after(100, again)

    def _show_models(self, ids: list[str]) -> None:
        preset = self._preset
        self.model_box["values"] = ids
        if not ids:
            self.models_label.configure(text=strings.MODELS_NONE.format(provider=preset.label if preset else ""))
        elif len(ids) == 1:
            self.models_label.configure(text=strings.MODELS_ONE)
        else:
            self.models_label.configure(text=strings.MODELS_COUNT.format(n=len(ids)))

    def _show_models_text(self, text: str) -> None:
        self.models_label.configure(text=text)

    def _model_changed(self) -> None:
        self._update_level()
        self._update_free_plan()
        self._update_route()
        self._update_hint()

    def _update_hint(self) -> None:
        """The probe's hint for this model, when one ran in this FlexTool (P1-O10)."""
        from flextool.assistant import probes

        preset = self._preset
        url = self._checked_address()
        model = self.model_var.get().strip()
        facts = probes.recent_facts(preset.id, url, model) if preset is not None and url and model else None
        self.cpu_hint.configure(text=probes.PARTLY_ON_CPU if facts is not None and facts.partly_on_cpu else "")

    def _update_level(self) -> None:
        """The level list of the model (P3 §3.3): "Automatic", then only the levels the model may be used at, from
        the facts FlexTool last found for it (the settings never ask a server) or its name alone."""
        from flextool.assistant import endpoint, probes, tiers

        preset = self._preset
        model = self.model_var.get().strip()
        if preset is None or not model:
            self._level_values = [None]
            self.level_box["values"] = [strings.LEVEL_AUTOMATIC_NO_MODEL]
            self.level_var.set(strings.LEVEL_AUTOMATIC_NO_MODEL)
            self.level_reason.configure(text="")
            return
        url = self._checked_address()
        facts = probes.cached_facts(preset.id, url, model) if url else None
        route = endpoint.route_for(preset, url, facts=facts, model=model) if url else None
        choice = self._level_choice
        try:
            tier = tiers.resolve_tier(preset, model, facts, choice, route)
            too_small = None
        except tiers.WindowTooSmall as exc:
            tier = tiers.resolve_tier(preset, model, None, choice, route)
            too_small = strings.LEVEL_WINDOW_TOO_SMALL.format(window=exc.window)
        automatic = _USER_LEVEL[tier.automatic]
        labels = [strings.LEVEL_AUTOMATIC.format(level=automatic)]
        values: list[str | None] = [None]
        for level in tier.allowed:
            name = _USER_LEVEL[level]
            above = tiers.LEVEL_ORDER.index(level) > tiers.LEVEL_ORDER.index(tier.automatic)
            labels.append(strings.LEVEL_RISK.format(level=name) if above else name)
            values.append(name)
        self._level_values = values
        self.level_box["values"] = labels
        if choice is not None and choice not in values:
            choice = None
        self.level_var.set(labels[values.index(choice)])
        lines = tier.lines()
        if too_small:
            lines.append(too_small)
        self.level_reason.configure(text="\n".join(lines))

    def _on_level_selected(self, _event: object = None) -> None:
        labels = list(self.level_box["values"])
        try:
            index = labels.index(self.level_var.get())
        except ValueError:
            return
        self._level_choice = self._level_values[index]

    def chosen_level(self) -> str | None:
        """The level the list shows: ``None`` for automatic, else basic / standard / advanced."""
        labels = list(self.level_box["values"])
        try:
            return self._level_values[labels.index(self.level_var.get())]
        except (ValueError, IndexError):
            return None

    #: The free plan's list after its first entry: the saved values.
    _FREE_VALUES: tuple[bool | None, ...] = (None, True, False)

    def _update_free_plan(self) -> None:
        """The free plan's list: "Automatic (…)" says what automatic gives for the model shown (P3-O2)."""
        from flextool.assistant.budget import free_tier_on

        preset = self._preset
        model = self.model_var.get().strip()
        if preset is None or not model:
            automatic = strings.FREE_PLAN_AUTOMATIC_NO_MODEL
        else:
            on = free_tier_on(preset.id, model, None)
            automatic = strings.FREE_PLAN_AUTOMATIC.format(
                state=strings.FREE_PLAN_STATE_ON if on else strings.FREE_PLAN_STATE_OFF)
        labels = [automatic, strings.FREE_PLAN_ON, strings.FREE_PLAN_OFF]
        self.free_box["values"] = labels
        self.free_var.set(labels[self._FREE_VALUES.index(self._free_choice)])

    def _on_free_selected(self, _event: object = None) -> None:
        self._free_choice = self.chosen_free_tier()

    def chosen_free_tier(self) -> bool | None:
        """The free plan the list shows: ``None`` for automatic, else on (``True``) or off (``False``)."""
        labels = list(self.free_box["values"])
        try:
            return self._FREE_VALUES[labels.index(self.free_var.get())]
        except (ValueError, IndexError):
            return None

    # --- save and cancel ------------------------------------------------------------------------------------
    def _error(self, text: str) -> None:
        self.error_label.configure(text=text)

    @staticmethod
    def _whole(text: str, low: int, high: int) -> int | None:
        text = text.strip()
        if not text.isdigit():
            return None
        value = int(text)
        return value if low <= value <= high else None

    def save(self) -> None:
        """[Save]: check the values, ask about fewer kept days, then write the settings in the background."""
        from flextool.assistant.net import BaseUrlRefused, check_base_url

        if self.saving:
            return
        self._error("")
        preset = self._preset
        base_url: str | None = None
        if preset is not None and self._address_editable(preset):
            address = self._address()
            if not address and preset.base_url is None:
                self._error(strings.SET_ADDRESS_NEEDED)
                self.address_entry.focus_set()
                return
            try:
                base_url = check_base_url(address) if address else None
            except BaseUrlRefused as exc:
                self._error(str(exc))
                self.address_entry.focus_set()
                return
            if base_url is not None and base_url == preset.base_url:
                base_url = None
        parallel = self._whole(self.parallel_var.get(), 1, 4)
        if parallel is None:
            self._error(strings.PARALLEL_INVALID)
            self.parallel_box.focus_set()
            return
        days = self._whole(self.retention_var.get(), 0, MAX_RETENTION_DAYS)
        if days is None:
            self._error(strings.RETENTION_INVALID)
            self.retention_box.focus_set()
            return
        new = copy.deepcopy(self._settings)
        new.notes = []
        provider = new.provider
        if preset is not None:
            provider.preset = preset.id
            provider.base_url = base_url
            provider.model = self.model_var.get().strip() or None
            provider.tier = self.chosen_level()
        provider.knobs = dict(self._knobs)
        provider.key_storage = "file" if self.storage_var.get() == "file" else "keyring"
        new.parallel_requests = parallel
        new.free_tier = self.chosen_free_tier()
        new.conversation_expiry_days = days
        if days < self._settings.conversation_expiry_days:
            text = strings.RETENTION_CONFIRM_ZERO if days == 0 else strings.RETENTION_CONFIRM.format(n=days)

            def answered(ok: bool) -> None:
                self.confirm = None
                if ok:
                    self._write(new)

            self.confirm = ConfirmDialog(self, strings.RETENTION_TITLE, text, strings.OK, answered)
            return
        self._write(new)

    def _write(self, new: Any) -> None:
        self.saving = True
        self.save_button.state(["disabled"])
        config_root = self._config_root

        def run() -> None:
            from flextool.assistant.settings import save_settings

            try:
                save_settings(new, config_root)
            except Exception as exc:  # noqa: BLE001 — NewerFormat, LockTimeout, OSError: shown, nothing changed
                logger.warning("The AI assistant settings could not be saved", exc_info=True)
                self._post(None, self._save_failed, str(exc) or type(exc).__name__)
                return
            self._post_to_main(self._saved)

        self._run(run, "flextool-assistant-settings-save")

    def _save_failed(self, reason: str) -> None:
        self.saving = False
        self.save_button.state(["!disabled"])
        self._error(strings.SAVE_FAILED.format(reason=reason))

    def _saved(self) -> None:
        """The settings are on disk: the chat window hears of it even when the dialog was closed meanwhile."""
        self.saving = False
        try:
            if self._on_saved is not None:
                self._on_saved()
        finally:
            if self.alive:
                self.close()

    def cancel(self) -> None:
        if self.saving:
            return                          # the save ends in a moment and closes the dialog itself
        self.close()

    def close(self) -> None:
        self.alive = False
        if self._poll_id is not None:
            try:
                self.after_cancel(self._poll_id)
            except tk.TclError:
                pass
            self._poll_id = None
        super().close()

    def _on_destroy(self, event: tk.Event) -> None:  # type: ignore[type-arg]
        if event.widget is self:
            self.alive = False
