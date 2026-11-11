"""Take local paths and machine names out of every text the tools send (P0-D13, review G7).

Every text that is sent is redacted exactly once (``fit_json`` for payloads, ``errors.render`` for messages):

1. Exact replacements, longest first: the project folder becomes ``<project>``, a known input source (one added
   from outside the project) its file name and its folder ``<source folder>``, the home folder ``~``.  Each is
   matched in its plain, forward-slash, backslash and percent-encoded spellings, and only as a whole folder name
   (``/home/al`` does not match inside ``/home/alice``).
2. Any other path becomes its last part: a quoted path (``'/srv/Acme Ltd/My Model.sqlite'`` → ``'My Model.sqlite'``,
   spaces allowed inside the quotes); a URL with a path (``file:``, ``sqlite:///`` …, not ``http(s)``/``ftp``),
   percent-decoded and without its query; a Windows path (``C:\\``, ``C:/``, ``\\\\server\\share``,
   ``//server/share``, ``\\folder\\file`` on the current drive); a POSIX absolute path; a relative path that climbs
   (``../../x``).  A folder name in the middle of an unquoted path may hold up to four words (``Acme Corp``); the
   last part may not (a sentence goes on after it), so an unquoted path that ends in a folder with a space keeps
   that folder's last word — rare, since Python quotes the paths in its error messages (G7-D2).
3. The host name becomes ``<host>`` and the user name ``<user>``.

Paths under the home folder follow one switch, :data:`KEEP_HOME_FOLDERS` (decision D3), in quoted and unquoted
text alike.

The result is a fixed point for text quoted after it (``json.dumps``): redacting the quoted text again changes
nothing, so a name can never lose its quotes to a second pass (``tests/assistant/test_redaction.py``).
"""
from __future__ import annotations

import getpass
import re
import socket
from functools import lru_cache
from pathlib import Path
from urllib.parse import quote, unquote

#: D3 (owner decision 2026-11-09, review G7): what happens to a path under the home folder.  ``False`` (in use)
#: shrinks it to its file name, as any other path outside the project: ``/home/alice/clients/Acme Corp/m.sqlite``
#: → ``m.sqlite``.  ``True`` would keep the folders below it (``~/clients/Acme Corp/m.sqlite``).  The one switch:
#: quoted and unquoted paths, and every spelling of the home folder, follow it alike.
KEEP_HOME_FOLDERS = False

#: The path patterns run until nothing changes, at most this often (each pass that changes shortens the text).
_MAX_PASSES = 8

PROJECT_TOKEN = "<project>"
SOURCE_FOLDER_TOKEN = "<source folder>"
HOME_TOKEN = "~"

_SEP = r"[\\/]"
_PART = r"[^\s/\\'\"<>|]+"                          # a part of a path: runs to white space, a quote or a separator
#: A folder in the middle of a path may hold up to four words with single spaces ("Acme Corp", "Program Files
#: (x86)"), but a sentence does not continue one: no space after ``. , ; :`` or a backquote, and no backquote,
#: ``;`` or square bracket in a later word ("see [reference](../x.md)" is two things, not a folder).
_SPACE = r"(?<![.,;:`]) "
_CONT = r"[^\s/\\'\"<>|`;\[\]]+"
_MID = rf"{_PART}(?:{_SPACE}{_CONT}){{0,3}}"
_LAST = rf"(?:{_MID}{_SEP})*{_PART}{_SEP}?"         # the folders, then the last part
_FIRST = rf"[^\s/\\'\"<>|{{}}]+(?:{_SPACE}[^\s/\\'\"<>|{{}}`;\[\]]+){{0,3}}"  # a server or top folder (no braces: TeX)
_UPART = r"[^\s/\\'\"<>|?#]+"                        # a part of a URL's path
_UMID = rf"{_UPART}(?:{_SPACE}[^\s/\\'\"<>|?#`;\[\]]+){{0,3}}"
_SCHEME = r"(?:file:(?=/)|(?!(?:https?|ftp|ftps|flextool-docs):)[A-Za-z][A-Za-z0-9+.-]*://)"
_URL = rf"{_SCHEME}/*(?:[A-Za-z]:[\\/])?(?:{_UMID}[\\/])*{_UPART}[\\/]?(?:[?#][^\s'\"<>]*)?"
#: Where a backslash path may start: not inside a word, after another backslash or a dot (``..\\x`` climbs), or
#: after a TeX ``$``.
_NOT_AFTER_BACKSLASH_ROOT = r"(?<![\w\\.$])"


def _quoted_roots(keep_home: bool) -> str:
    roots = [r"/", r"\\", r"[A-Za-z]:[\\/]", r"\.\.[\\/]", _SCHEME]
    if not keep_home:
        roots.append(r"~[\\/]")
    return "(?:" + "|".join(roots) + ")"


@lru_cache(maxsize=2)
def _patterns(keep_home: bool) -> tuple[re.Pattern, ...]:
    """The path patterns, in the order they apply (see the module text)."""
    # A quoted path holds no quote of either kind, and an escaped quote (\\") neither opens nor closes one: the
    # text may already be quoted JSON.
    quoted = re.compile(rf"(?<![\w\\])(['\"])({_quoted_roots(keep_home)}[^'\"\n]{{0,1024}}?)(?<!\\)\1")
    url = re.compile(rf"(?<![\w.+-]){_URL}")
    windows = re.compile(
        rf"(?<![\w])[A-Za-z]:[\\/]{_LAST}"                                   # C:\a  C:/a
        rf"|{_NOT_AFTER_BACKSLASH_ROOT}\\\\?{_FIRST}{_SEP}{_LAST}"            # \\server\share  \folder\file
        rf"|(?<![\w:/]){'//'}{_FIRST}/{_LAST}")                               # //server/share (not https://)
    posix = re.compile(rf"(?<![\w.:/~<>\\$)\]}}-])/{_MID}/{_LAST}")     # not after a closing bracket: (a+b)/2
    dotted = re.compile(rf"(?<![\w.])(?:\.\.[\\/])+{_LAST}")
    out = [quoted, url, dotted, windows, posix]
    if not keep_home:
        out.append(re.compile(rf"(?<![\w.:/~<>\\-])~[\\/]{_LAST}"))
    return tuple(out)


def _last_part(path: str) -> str:
    """The last part of *path* (a URL's query and fragment dropped and its path percent-decoded); *path* itself
    when it has none."""
    text = path
    is_url = re.match(_SCHEME, text) is not None
    if is_url:
        text = re.split(r"[?#]", text, maxsplit=1)[0]
    text = text.rstrip("/\\")
    last = re.split(r"[\\/]", text)[-1]
    if is_url:
        last = unquote(last)
    return last or path


def _shrink(match: re.Match) -> str:
    return _last_part(match.group(0))


#: The paths a quoted text may be (any part may hold spaces): the shapes of the unquoted patterns, so that text
#: quoted after redaction, whose backslashes are doubled, is never read as a path it was not.
_Q = r"[^/\\]+"
_Q_FIRST = r"[^/\\{}]+"
_Q_TAIL = rf"(?:{_Q}[\\/])*{_Q}[\\/]?"


@lru_cache(maxsize=2)
def _quoted_shapes(keep_home: bool) -> re.Pattern:
    shapes = [rf"[A-Za-z]:[\\/]{_Q_TAIL}", rf"\\\\?{_Q_FIRST}[\\/]{_Q_TAIL}", rf"//{_Q_FIRST}/{_Q_TAIL}",
              rf"/{_Q}/{_Q_TAIL}", rf"(?:\.\.[\\/])+{_Q_TAIL}",
              rf"{_SCHEME}/*(?:[A-Za-z]:[\\/])?(?:[^/\\?#]+[\\/])*[^/\\?#]+[\\/]?(?:[?#].*)?"]
    if not keep_home:
        shapes.append(rf"~[\\/]{_Q_TAIL}")
    return re.compile("|".join(f"(?:{s})" for s in shapes))


def _shrink_quoted(match: re.Match, keep_home: bool) -> str:
    quote_char, inner = match.group(1), match.group(2)
    if not _quoted_shapes(keep_home).fullmatch(inner):
        return match.group(0)
    last = _last_part(inner)
    return f"{quote_char}{last}{quote_char}" if last != inner else match.group(0)


def _spellings(path: Path) -> list[str]:
    """The ways *path* can appear in a text, longest first."""
    text = str(path)
    forms = {text, path.as_posix(), text.replace("/", "\\"), quote(path.as_posix(), safe="/:"),
             quote(path.as_posix(), safe="/")}
    return sorted((f for f in forms if len(f) > 1), key=len, reverse=True)


def _forms(path: Path) -> set[Path]:
    """*path* as given and resolved (a symlinked folder has both spellings in texts)."""
    out = {Path(path)}
    try:
        out.add(Path(path).resolve())
    except (OSError, RuntimeError):
        pass
    return out


class Redactor:
    """The redaction for one project on this machine."""

    def __init__(self, project_root: Path | None, *, home: Path | None = None, host: str | None = None,
                 user: str | None = None, keep_home_folders: bool | None = None) -> None:
        self.keep_home_folders = KEEP_HOME_FOLDERS if keep_home_folders is None else keep_home_folders
        self._roots: list[Path] = []
        self._exact: dict[str, str] = {}             # spelling → token; the first registered wins
        if project_root is not None:
            for p in _forms(Path(project_root)):
                self._roots.append(p)
                self._add(p, PROJECT_TOKEN)
        home_dir = Path.home() if home is None else Path(home)
        self._homes = sorted(_forms(home_dir))
        self._known: set[Path] = set()
        self._home_spellings = [s for p in self._homes for s in _spellings(p)]
        self._build()
        self.words: list[tuple[re.Pattern, str]] = []
        host_name = socket.gethostname() if host is None else host
        try:
            user_name = getpass.getuser() if user is None else user
        except (OSError, KeyError):
            user_name = ""
        for word, token in ((host_name, "<host>"), (host_name.split(".")[0], "<host>"), (user_name, "<user>")):
            if word and len(word) >= 3:
                self.words.append((re.compile(rf"(?<![\w<]){re.escape(word)}(?![\w>])", re.IGNORECASE), token))

    # --- the exact replacements --------------------------------------------------------------------------------
    def _add(self, path: Path, token: str) -> None:
        for spelling in _spellings(path):
            self._exact.setdefault(spelling, token)

    def _build(self) -> None:
        table = dict(self._exact)
        for spelling in self._home_spellings:
            table.setdefault(spelling, HOME_TOKEN)
        self.replacements: list[tuple[str, str]] = sorted(table.items(), key=lambda item: len(item[0]),
                                                          reverse=True)
        self._table = table
        self._exact_pattern = (re.compile("(?:" + "|".join(re.escape(s) for s, _ in self.replacements)
                                          + r")(?![\w-])") if self.replacements else None)

    def _near_root(self, folder: Path) -> bool:
        """Whether *folder* is the project or the home folder, holds one of them, or lies in the project: its
        spelling must not be replaced by ``<source folder>``."""
        for root in self._roots + self._homes:
            if folder == root or folder in root.parents:
                return True
        return len(folder.parts) <= 1 or any(root in folder.parents for root in self._roots)

    def add_known(self, path: Path) -> None:
        """Register an input source from outside the project: its path (as stored and resolved) becomes its file
        name, its folder ``<source folder>`` — before the home folder takes its prefix (longest first)."""
        path = Path(path)
        if path in self._known:
            return
        self._known.add(path)
        for p in _forms(path):
            if not p.name:
                continue
            self._add(p, p.name)
            if not self._near_root(p.parent):
                self._add(p.parent, SOURCE_FOLDER_TOKEN)
        self._build()

    def known_paths(self) -> list[str]:
        """The input sources from outside the project registered with :meth:`add_known`, sorted (a checkpoint
        keeps them, so a resumed calibration cell redacts them before a tool learns them again)."""
        return sorted(str(p) for p in self._known)

    # --- redaction ---------------------------------------------------------------------------------------------
    def redact(self, text: str) -> str:
        if not text:
            return text
        quoted, url, *paths = _patterns(self.keep_home_folders)
        for _ in range(_MAX_PASSES):     # a last part and the text after it can form a new path: until none is left
            before = text
            if self._exact_pattern is not None:
                text = self._exact_pattern.sub(lambda m: self._table[m.group(0)], text)
            text = quoted.sub(lambda m: _shrink_quoted(m, self.keep_home_folders), text)
            text = url.sub(_shrink, text)
            for pattern in paths:
                text = pattern.sub(_shrink, text)
            if text == before:
                break
        for pattern, token in self.words:
            text = pattern.sub(token, text)
        return text
