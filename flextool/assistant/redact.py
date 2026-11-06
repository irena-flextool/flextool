"""Take local paths and machine names out of every text the tools send (P0-D13).

``fit_json`` applies :meth:`Redactor.redact` to every string of every payload, and ``call_tool`` to every
message: the project folder becomes ``<project>``, the home folder ``~``, the host name ``<host>``, the user name
``<user>``, and any other absolute path its last part.  Paths are matched in their plain, forward-slash,
backslash and percent-encoded spellings (a database URI percent-encodes its path).
"""
from __future__ import annotations

import getpass
import re
import socket
from pathlib import Path
from urllib.parse import quote

# An absolute path: POSIX "/a/b" (at least two parts, not inside a word or a URL's "//"), or a Windows
# "C:\a" / "C:/a" / "\\server\share".  A part runs to whitespace or a quote.
_PART = r"[^\s/\\'\"<>|]+"
_POSIX_PATH = re.compile(rf"(?<![\w.:/~<>-])/(?:{_PART}/)+{_PART}/?")
_WINDOWS_PATH = re.compile(rf"(?<![\w])(?:[A-Za-z]:|\\\\{_PART})[\\/](?:{_PART}[\\/])*{_PART}[\\/]?")


def _spellings(path: Path) -> list[str]:
    """The ways *path* can appear in a text, longest first."""
    text = str(path)
    forms = {text, path.as_posix(), text.replace("/", "\\"), quote(path.as_posix(), safe="/:"),
             quote(path.as_posix(), safe="/")}
    return sorted((f for f in forms if len(f) > 1), key=len, reverse=True)


def _basename(match: re.Match) -> str:
    text = match.group(0).rstrip("/\\")
    return re.split(r"[\\/]", text)[-1] or text


class Redactor:
    """The redaction for one project on this machine."""

    def __init__(self, project_root: Path | None, *, home: Path | None = None, host: str | None = None,
                 user: str | None = None) -> None:
        self.replacements: list[tuple[str, str]] = []
        if project_root is not None:
            for p in {Path(project_root), Path(project_root).resolve()}:
                self.replacements += [(s, "<project>") for s in _spellings(p)]
        home_dir = Path.home() if home is None else home
        for p in {home_dir, home_dir.resolve()}:
            self.replacements += [(s, "~") for s in _spellings(p)]
        self.replacements.sort(key=lambda item: len(item[0]), reverse=True)
        self.words: list[tuple[re.Pattern, str]] = []
        host_name = socket.gethostname() if host is None else host
        try:
            user_name = getpass.getuser() if user is None else user
        except (OSError, KeyError):
            user_name = ""
        for word, token in ((host_name, "<host>"), (host_name.split(".")[0], "<host>"), (user_name, "<user>")):
            if word and len(word) >= 3:
                self.words.append((re.compile(rf"(?<![\w]){re.escape(word)}(?![\w])", re.IGNORECASE), token))

    def redact(self, text: str) -> str:
        if not text:
            return text
        for needle, token in self.replacements:
            if needle in text:
                text = text.replace(needle, token)
        text = _WINDOWS_PATH.sub(_basename, text)
        text = _POSIX_PATH.sub(_basename, text)
        for pattern, token in self.words:
            text = pattern.sub(token, text)
        return text
