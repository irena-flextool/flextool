"""Every text the AI assistant shows in the FlexTool window (P2 §9), in plain English: no abbreviations, no links.

The terminal's texts stay in :mod:`flextool.assistant.chat_cli`; the provider and session messages come from P1
(:func:`flextool.assistant.llm_errors.message_for` with ``"gui"``).  Standard library only.
"""
from __future__ import annotations

# --- the side menu (§9.1) -----------------------------------------------------------------------------------------
ENTRY_BUTTON = "AI assistant…"
SETTINGS_SYMBOL = "⚙"
SETTINGS_SYMBOL_FALLBACK = "…"
ENTRY_SETTINGS_TOOLTIP = "AI assistant settings: provider, key, model and data use."
ENTRY_TOOLTIP = ("Ask an AI assistant about this project's inputs, results and run logs. It reads your files with "
                 "read-only tools and cannot change your model. Nothing is sent until you ask a question.")
UNAVAILABLE = ("The AI assistant's packages are missing or broken in this installation. Update FlexTool (Update "
               "FlexTool… or pip install --upgrade flextool) to install them.")

# --- the chat window (§9.2) ---------------------------------------------------------------------------------------
WINDOW_TITLE = "AI assistant — {project}"
CHATS = "Chats"
CHATS_MORE = "Chats ({n} more)"
CHATS_ARROW = " ▾"
CHATS_ATTENTION = "• "
SETTINGS_BUTTON = "Settings {symbol}"
HOW_IT_WORKS = "How this works"
NEW_CHAT = "New chat"
OPEN_CHATS = "Open chats"
REOPEN = "Reopen a saved chat"
DELETE_SAVED = "Delete a saved chat"
DELETE_ALL_CLOSED = "Delete all closed chats…"
RENAME_CHAT = "Rename chat…"
CLOSE_CHAT = "Close chat"
CLOSE_OTHER_CHATS = "Close other chats"
SEND = "Send"
STOP = "Stop"
YOU = "You"
ASSISTANT = "Assistant"
THIS_CHAT = "This chat:"
OPEN_MARK = "(open)"
MENU_ENTRY_PARTS_SEPARATOR = " · "
NO_SAVED_CHATS = "No saved chats"
NEW_CHAT_ACCELERATOR = "Ctrl+T"
CLOSE_CHAT_ACCELERATOR = "Ctrl+W"
NEW_CHAT_ACCELERATOR_MAC = "Command+T"
CLOSE_CHAT_ACCELERATOR_MAC = "Command+W"

RENAME_TITLE = "Rename chat"
RENAME_LABEL = "Name:"
RENAME_HINT = "Leave empty to show the chat's number."
OK = "OK"
CANCEL = "Cancel"
CLOSE = "Close"

#: Shown in an empty input box while the chat has no messages (P2-O5, exact).
PLACEHOLDER = "Ask here — Enter: send, Ctrl+Enter: new line"
#: The transcript of an empty "New chat".
EXAMPLE = ("Ask about this project's inputs, results or run logs, for example: “Why is there loss of load in my last "
           "run?” The assistant also sees what is selected in the FlexTool window and the result viewer.")

CONFIRM_DELETE = "Delete this chat from this computer? This cannot be undone."
CONFIRM_DELETE_OPEN = "This chat is open. Deleting it also closes its tab. This cannot be undone."
CONFIRM_DELETE_ALL = "Delete every saved chat that is not open in a tab? This cannot be undone."
CONFIRM_CLOSE_UNSAVED = "This chat isn't saved. Close it anyway?"
CONFIRM_CLOSE_UNSAVED_MANY = "{n} chats aren't saved. Close them anyway?"
DELETE_TITLE = "Delete chat"
DELETE_BUTTON = "Delete"
PROJECT_DELETE_NOTE = "Its saved AI assistant chats are deleted too."

NOT_SAVED = "Chats are not saved (see Settings)."
WAITING_CLOSED_CHAT = "Waiting for the closed chat to finish…"
STILL_FINISHING = "This chat is still finishing a tool call. Try again in a moment."
NOT_DELETED_LOCKED = "This chat is open in another FlexTool window or terminal, so it was not deleted."
NOT_RESTORED = "{n} chats are open in another FlexTool window or terminal and were not restored."
NOT_RESTORED_ONE = "1 chat is open in another FlexTool window or terminal and was not restored."
OPEN_ELSEWHERE = "This chat is open in another FlexTool window or terminal."
OPEN_MISSING = "This chat is no longer saved on this computer."
SEND_WHEN_DONE = "You can send when the current answer is done."
STOPPED = "Stopped."
NOT_SENT = "Not sent."
NOTHING_SENT = "Nothing was sent."
CONTEXT_SENT = "▸ FlexTool context sent with this question"
ARGUMENTS = "Arguments:"
SENT_TO = "Sent to {provider} (first {n} characters):"
COMPACTED = "removed to keep the conversation short"
RELAXED = "Privacy setting relaxed: {report}."
TOOL_DONE = "done"
TOOL_ERROR = "error"
TOOL_NOT_RUN = "not run"

NOTICE_TITLE = "Before sending to {provider}"
NOTICE_CHAT = "Chat {number}"
NOTICE_SEND = "Send to {provider}"
PROJECT_STILL_FINISHING = "The AI assistant is still finishing a tool call. Try again in a moment."
PROJECT_WAITING = "Waiting for the AI assistant to finish…"

# --- the tabs: state prefixes (§4.3) ------------------------------------------------------------------------------
PREFIX_WORKING = "… "
PREFIX_WAITING = "‖ "
PREFIX_NOTICE = "? "
PREFIX_ERROR = "! "
PREFIX_UNSEEN = "• "
TAB_NAME_MAX = 16
MENU_FIRST_WORDS_MAX = 40

# --- status of a chat (§4.6) --------------------------------------------------------------------------------------
STATUS_PREPARING = "Preparing…"
STATUS_LOADING = "Loading the model on this computer ({s} s)…"
STATUS_THINKING = "Thinking…"
STATUS_WAITING_MODEL = "Waiting for the model ({s} s). Stop cancels the question."
STATUS_RUNNING_TOOL = "Running {tool}…"
STATUS_GATE_BUSY = "Waiting: another chat is using {provider}."
STATUS_GATE_COOL_DOWN = "Waiting: {provider} asked to slow down ({s} s)."
STATUS_NOTICE_OPEN = "Waiting for your answer to the data notice."
STATUS_NOTICE_QUEUED = "Waiting: the data notice of another chat is open."
STATUS_TOOL_WAITING = "Waiting: another chat is running a tool."

# --- the totals and the footer ------------------------------------------------------------------------------------
TOTALS = "This chat: {input} in · {output} out"
FOOTER = "{provider} this month: {input} in · {output} out"
FOOTER_REJECTED = " · rate-limited attempts: {n}"
FOOTER_NONE = "{provider} this month: nothing used yet"

# --- the error actions (§5.6) -------------------------------------------------------------------------------------
ACTION_OPEN_SETTINGS = "Open settings"
ACTION_NEW_CHAT = "New chat"
ACTION_NEW_CHAT_SUMMARY = "New chat with summary"

# --- setup problems (§9.4) ----------------------------------------------------------------------------------------
CHOOSE_PROVIDER = "Choose an AI provider in the settings first."
CHOOSE_MODEL = "No model is chosen for {provider}. Pick one in the assistant settings."
NEEDS_ADDRESS = "{provider} needs the address of the endpoint. Enter it in the assistant settings."
UNKNOWN_PROVIDER = "The saved provider {preset} is not known to this FlexTool; choose one in the assistant settings."
CONSENT_NOT_KEPT = "The acceptance of the data notice could not be kept ({reason}); it is asked again next time."

# --- the settings (§8.4, §9.3) -----------------------------------------------------------------------------------
SETTINGS_TITLE = "AI assistant settings"
SET_PROVIDER = "Provider"
SET_CHOOSE_PROVIDER = "Choose a provider"
SET_ADDRESS = "Address"
SET_ADDRESS_FIXED = "(fixed)"
SET_ADDRESS_NEEDED = "Enter the address of the endpoint."
SET_KEY = "Key"
SET_STORE_KEY = "Store key"
SET_FORGET_KEY = "Forget key"
SET_STORE_IN = "Store in:"
SET_STORE_KEYRING = "system keyring"
SET_STORE_FILE = "a file on this computer (not encrypted)"
SET_KEY_CHECKING = "Checking the key…"
SET_KEY_EMPTY = "Paste a key in the box first."
KEY_FROM_ENV = "Using the key from the environment variable {name}. It is used instead of a stored key."
KEY_FROM_KEYRING = "Using the key stored in the system keyring."
KEY_FROM_FILE = "Using the key stored unencrypted in {path}."
KEY_NONE = "No key stored."
KEY_OPTIONAL = "No key is needed for this endpoint unless it asks for one."
KEY_PROBLEM = "The key could not be read: {reason}"
KEY_STORE_FAILED = "The key was not stored: {reason}"
SET_MODEL = "Model"
SET_REFRESH = "Refresh list"
MODELS_LOADING = "Fetching the model list…"
MODELS_COUNT = "{n} models"
MODELS_ONE = "1 model"
MODELS_NONE = "{provider} lists no models."
MODELS_NEED_KEY = "Store a key first to see the model list."
MODELS_FAILED_INTERNAL = "The model list could not be fetched ({type}). Type the model name instead."
SET_LEVEL = "Level"
LEVEL_AUTOMATIC = "Automatic ({level} for this model)"
LEVEL_AUTOMATIC_NO_MODEL = "Automatic"
LEVEL_RISK = "{level} (at your own risk)"
LEVEL_WINDOW_TOO_SMALL = ("FlexTool last saw this model run with a {window}-token context, too small for the "
                          "assistant.")
SET_FREE_PLAN = "Free plan: fewer steps for each question, to stay within the provider's request limits"
FREE_PLAN_AUTOMATIC = "Automatic ({state} for this model)"
FREE_PLAN_AUTOMATIC_NO_MODEL = "Automatic"
FREE_PLAN_ON = "On"
FREE_PLAN_OFF = "Off"
FREE_PLAN_STATE_ON = "on"
FREE_PLAN_STATE_OFF = "off"
SET_PARALLEL = "Questions answered at the same time"
SET_PARALLEL_NOTE = "(more chats wait their turn)"
PARALLEL_INVALID = "Enter a whole number from 1 to 4."
SET_DATA_USE = "Data use"
SET_SENT_WITH = "Sent with every request:"
SET_CONSOLE = "Set in your provider console:"
SET_NOTHING_LOCAL = "Nothing to set: the endpoint is on this computer."
SET_OLLAMA_CHECK = "Checked again when you send: a cloud model sends data off this computer."
SET_CONVERSATIONS = "Conversations"
SET_KEEP = "Keep conversations for"
SET_KEEP_DAYS = "days. 0 means conversations are not saved."
SET_KEEP_NOTE = "Older conversations are deleted when FlexTool closes."
RETENTION_INVALID = "Enter a whole number of days from 0 to 3650."
SAVE = "Save"
SAVE_FAILED = "The settings were not saved: {reason}"
SETTINGS_NEWER = "These settings were written by a newer FlexTool, so this one does not change them."
KEY_FILE_TITLE = "Store the key in a file"
KEY_FILE_CONFIRM = ("The key will be stored unencrypted at {path}. Anyone who can read your files can read it. Store "
                    "it there?")
KEY_FILE_BUTTON = "Store in the file"
RELAX_TITLE = "Privacy setting"
RELAX_CONFIRM = ("Turn off “{label}”? Requests will then be sent without this setting, and the chat window will show "
                 "that the privacy setting is relaxed.")
RELAX_OFF = "Turn off"
RELAX_KEEP = "Keep on"
RETENTION_TITLE = "Keep conversations"
RETENTION_CONFIRM = "Saved chats older than {n} days will be deleted when FlexTool closes."
RETENTION_CONFIRM_ZERO = "No chats will be saved, and saved ones will be deleted when FlexTool closes."

# --- the welcome (§8.2, owner-approved text, P2-O6) ---------------------------------------------------------------
WELCOME_TITLE = "How the AI assistant works"
#: (heading, body) in order; "⚙" is the settings symbol of this platform (replaced where the font lacks it).
WELCOME_SECTIONS: tuple[tuple[str, str], ...] = (
    ("Nothing is sent until you ask",
     "Opening FlexTool or this window sends nothing to any AI service. Your data leaves this computer only when you "
     "send a question in the chat."),
    ("What the assistant does",
     "It reads what it needs from your input data, results and run logs with read-only tools and gives advice. It "
     "cannot change your model."),
    ("What you need",
     "An account with an AI provider and its key, which you paste in Settings (⚙), or an AI model (LLM) running on "
     "this computer. Free plans work, but they limit how many questions you can send and often let the provider use "
     "your data for training. Paid plans have higher limits and may more often let you limit how your data is used. "
     "Check the provider's current policies yourself. A model running on this computer keeps your data here, but it "
     "needs a powerful computer. Some models offered through programs on this computer still run elsewhere, such as "
     "Ollama's cloud models; FlexTool tells you when that happens."),
    ("Assistant level",
     "FlexTool picks a level (basic, standard or advanced) from the model name and its context size. Smaller models "
     "get simpler tasks, such as finding and explaining values; larger models are also asked to work through "
     "problems in several steps, such as why a run has slack. You can choose a lower level in Settings."),
)
WELCOME_OPEN_SETTINGS = "Open settings"
WELCOME_OPEN_CHAT = "Open chat"
#: The tooltip of [Open chat] while no provider has been saved in the settings.
WELCOME_CHAT_NEEDS_SETTINGS = "Check the settings first."


def welcome_text(symbol: str = SETTINGS_SYMBOL) -> str:
    """The welcome as one text: the title, then each heading and its body (the dialog shows the same parts)."""
    parts = [WELCOME_TITLE]
    for heading, body in WELCOME_SECTIONS:
        parts += ["", heading, body.replace(SETTINGS_SYMBOL, symbol)]
    return "\n".join(parts)


def settings_button_text(symbol: str = SETTINGS_SYMBOL) -> str:
    return SETTINGS_BUTTON.format(symbol=symbol)
