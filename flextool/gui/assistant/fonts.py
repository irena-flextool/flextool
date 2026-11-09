"""The fonts of the chat window and the assistant's dialogs (owner decisions 2026-11-08): one named base font and
the fonts derived from it.

:data:`CHAT_FONT` is FlexTool's body text font (``TkTextFont``, ``ui_metrics.setup_fonts``) :data:`CHAT_STEP_PT`
point larger: 11 pt at the default 10 pt, and the font size setting still scales it.  It is the font of the
transcript, the input box ("Ask here") and the text of the welcome, the data notice, the rename dialog and the
questions.  Bold and italic are its family at its size; inline code, code blocks, tables and the details of a tool
call the fixed-width family at its point size (Tk tells no x-height, and the ascent misleads: Noto Sans reports a
taller ascent than DejaVu Sans Mono, whose letters are not smaller); the headings larger in proportion
(:data:`HEADING_SCALE`).  Buttons, tabs and the status lines keep the theme's font, as in the rest of FlexTool.

:func:`sync_fonts` sets them all from the body font as it is now; the transcript calls it at every answer and on a
theme change, so a changed font size reaches them.  It configures a font only when a value differs (a configure
re-lays out every text widget of the application).  The fonts are shared by every chat and dialog of the process
and never deleted (a named font deleted while a widget uses it falls back to the default font).  Tk thread only.
"""
from __future__ import annotations

import tkinter as tk
import tkinter.font as tkfont

#: The font size setting's body font, which the chat font follows.
BODY_FONT = "TkTextFont"
#: The chat's base font, and the fonts derived from it.
CHAT_FONT = "FlexToolChat"
FONTS = {"bold": "FlexToolChatBold", "italic": "FlexToolChatItalic", "code": "FlexToolChatCode",
         "h1": "FlexToolChatHeading1", "h2": "FlexToolChatHeading2"}
#: How many points the chat font is above the body font (11 pt at the default 10 pt).
CHAT_STEP_PT = 1
#: How much larger the headings are than the chat text (a Markdown ``###`` heading, and a section heading of a
#: dialog, is bold at the same size; the welcome's title is the ``##`` heading font).
HEADING_SCALE = {"h1": 1.3, "h2": 1.15}


def _named(widget: tk.Misc, name: str, **options: object) -> tkfont.Font:
    try:
        font = tkfont.Font(root=widget, name=name, exists=True)
        # Only what differs: Tk re-lays out every widget using the font at any configure, even with the same values
        # (review G19, C7-m8), and this runs at every answer.
        current = font.configure()
        changed = {key: value for key, value in options.items() if str(current.get(key)) != str(value)}
        if changed:
            font.configure(**changed)
    except tk.TclError:
        font = tkfont.Font(root=widget, name=name, **options)
        # Kept for the process: a Font object left to the garbage collector would delete the named font (on
        # whichever thread collects it) under the widgets that use it.
        font.delete_font = False
    return font


def _scaled(size: int, factor: float) -> int:
    """*size* (points) times *factor*, at least 1."""
    return max(1, round(size * factor))


def chat_size(widget: tk.Misc) -> int:
    """The chat font's size in points: the body font's (in points also when it is set in pixels) plus
    :data:`CHAT_STEP_PT`."""
    body = tkfont.nametofont(BODY_FONT, root=widget)
    size = int(body.cget("size"))
    if size <= 0:                       # a size in pixels (or 0): the size in points Tk shows it at
        size = int(body.actual("size")) or 10
    return size + CHAT_STEP_PT


def sync_fonts(widget: tk.Misc) -> dict[str, tkfont.Font]:
    """Set :data:`CHAT_FONT` and the fonts derived from it (:data:`FONTS`) from the body font as it is now; returns
    them by role (``"base"`` and the keys of :data:`FONTS`)."""
    body = tkfont.nametofont(BODY_FONT, root=widget)
    family = body.actual("family")
    fixed = tkfont.nametofont("TkFixedFont", root=widget).actual("family")
    size = chat_size(widget)
    return {
        "base": _named(widget, CHAT_FONT, family=family, size=size, weight="normal", slant="roman"),
        "bold": _named(widget, FONTS["bold"], family=family, size=size, weight="bold", slant="roman"),
        "italic": _named(widget, FONTS["italic"], family=family, size=size, weight="normal", slant="italic"),
        "code": _named(widget, FONTS["code"], family=fixed, size=size, weight="normal", slant="roman"),
        "h1": _named(widget, FONTS["h1"], family=family, size=_scaled(size, HEADING_SCALE["h1"]), weight="bold",
                     slant="roman"),
        "h2": _named(widget, FONTS["h2"], family=family, size=_scaled(size, HEADING_SCALE["h2"]), weight="bold",
                     slant="roman"),
    }
