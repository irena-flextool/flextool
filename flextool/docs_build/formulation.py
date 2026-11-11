"""The model formulation page, generated from the ``EQ.`` / ``VAR.`` / ``OBJ.`` headers of the installed code.

The header grammar is declared in ``docs/dev/engine.md`` "Equation headers"; :func:`parse_headers` is its
one reader.  A header that breaks the grammar raises :class:`HeaderGrammarError` naming its file and line.
"""
from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from pathlib import Path

from flextool.docs_build import scope

PACKAGE_DIR = Path(__file__).resolve().parents[1]          # .../flextool
MODEL_DIR = PACKAGE_DIR / "engine" / "model"
BENDERS_FILE = PACKAGE_DIR / "engine" / "solve" / "_benders.py"
BENDERS_TITLE = "Benders decomposition master"

KINDS = ("EQ", "VAR", "OBJ")
# `{sfx}` is the one declared brace placeholder: the unit-commitment suffix of the online families.
PLACEHOLDERS: dict[str, tuple[str, ...]] = {"sfx": ("_linear", "_integer")}

_HEADER_RE = re.compile(r"^(?P<indent>[ \t]*)# ─── (?P<kind>[A-Z]+)\. ─── (?P<text>.*?)\s*$")
_ANY_RULE_HEADER_RE = re.compile(r"^[ \t]*# ─── ")
_CONTINUATION_RE = re.compile(r"^[ \t]*# {13}(?P<text>\S.*?)\s*$")
_NAMES_LINE_RE = re.compile(r"^names:\s*(?P<names>.+?)\s*$")
_IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_SEPARATOR_RE = re.compile(r"\s*;\s*|,\s+|\s+/\s+")
# A prose line starts, unindented, with a capitalised word ("Forces …", "A coarse row …", "Row rule …").
_PROSE_START_RE = re.compile(r"^[A-Z](?:[a-z']| [a-z])")


class HeaderGrammarError(ValueError):
    """A header that does not follow the declared grammar (docs/dev/engine.md "Equation headers")."""


@dataclass(frozen=True)
class Header:
    """One ``EQ.`` / ``VAR.`` / ``OBJ.`` header."""

    kind: str                   # "EQ", "VAR" or "OBJ"
    title: str                  # the names text as written (qualifiers included)
    names: tuple[str, ...]      # the expanded names
    formula: str                # the formula lines (the leading run of the body)
    notes: str                  # the prose lines after the formula
    file: str                   # "flextool/engine/model/core.py"
    line: int                   # 1-based line of the header in the installed code
    function: str | None        # the enclosing function
    module: str                 # "core", "_online", …, "_benders"


# ---------------------------------------------------------------------------
# Names


def _strip_qualifiers(text: str, where: str) -> str:
    """Drop every parenthesised qualifier ``(…)``, nested ones included."""
    out: list[str] = []
    depth = 0
    for ch in text:
        if ch == "(":
            depth += 1
        elif ch == ")":
            if depth == 0:
                raise HeaderGrammarError(f"{where}: unbalanced ')' in the header names {text!r}")
            depth -= 1
        elif depth == 0:
            out.append(ch)
    if depth:
        raise HeaderGrammarError(f"{where}: unbalanced '(' in the header names {text!r}")
    return "".join(out)


def _expand_one(pattern: str, where: str) -> list[str]:
    """Expand ``{a,b}`` alternatives, ``[_x]`` optional parts and the ``{sfx}`` placeholder of one name."""
    m = re.search(r"[{\[]", pattern)
    if m is None:
        if any(c in pattern for c in "}]"):
            raise HeaderGrammarError(f"{where}: unbalanced bracket in the header name {pattern!r}")
        return [pattern]
    start = m.start()
    close = "}" if pattern[start] == "{" else "]"
    end = pattern.find(close, start)
    if end < 0:
        raise HeaderGrammarError(f"{where}: unbalanced bracket in the header name {pattern!r}")
    inner = pattern[start + 1:end]
    if any(c in inner for c in "{[]}"):
        raise HeaderGrammarError(f"{where}: nested brackets in the header name {pattern!r}")
    if close == "]":
        options = ["", inner]
    elif "," in inner:
        options = [o.strip() for o in inner.split(",")]
    elif inner in PLACEHOLDERS:
        options = list(PLACEHOLDERS[inner])
    else:
        raise HeaderGrammarError(
            f"{where}: {{{inner}}} in {pattern!r} is neither a list {{a,b}} nor a declared placeholder "
            f"({', '.join('{' + k + '}' for k in PLACEHOLDERS)})")
    head = pattern[:start]
    return [head + option + rest for option in options for rest in _expand_one(pattern[end + 1:], where)]


def expand_names(text: str, where: str = "header") -> tuple[str, ...]:
    """The names a header's names text stands for, in order, without duplicates.

    ``(…)`` qualifiers are dropped; names are separated by `` / ``, ``, `` or ``;``; a ``label:`` before a
    name is dropped (``<family>_coarse: maxFlow_coarse``); ``{a,b}`` and ``[_x]`` expand, ``{sfx}`` is
    ``_linear`` / ``_integer``.  Every result must be an identifier.
    """
    stripped = _strip_qualifiers(text, where)
    names: list[str] = []
    for part in _SEPARATOR_RE.split(stripped):
        part = part.strip()
        if ":" in part:
            part = part.rsplit(":", 1)[1].strip()
        if not part:
            continue
        for name in _expand_one(part, where):
            if not _IDENT_RE.match(name):
                raise HeaderGrammarError(
                    f"{where}: {name!r} is not a name; a free-text header needs a '# names: a, b' line")
            if name not in names:
                names.append(name)
    if not names:
        raise HeaderGrammarError(f"{where}: the header names no family")
    return tuple(names)


# ---------------------------------------------------------------------------
# Parsing


def _comment_text(line: str) -> str:
    """The text of a comment line: indentation, ``#`` and one following space removed."""
    text = line.lstrip()[1:]
    return text[1:] if text.startswith(" ") else text


def _function_spans(source: str) -> list[tuple[int, int, str]]:
    tree = ast.parse(source)
    return [(n.lineno, n.end_lineno or n.lineno, n.name) for n in ast.walk(tree)
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]


def _enclosing_function(spans: list[tuple[int, int, str]], line: int) -> str | None:
    best: tuple[int, int, str] | None = None
    for span in spans:
        if span[0] <= line <= span[1] and (best is None or span[0] > best[0]):
            best = span
    return best[2] if best else None


def split_body(body: list[str]) -> tuple[str, str]:
    """Split a header body into the formula (its leading run) and the prose (from the first prose line)."""
    for i, text in enumerate(body):
        if _PROSE_START_RE.match(text):
            return "\n".join(body[:i]).strip("\n"), "\n".join(body[i:]).strip("\n")
    return "\n".join(body).strip("\n"), ""


def parse_headers(source: str, file: str, module: str) -> list[Header]:
    """Every header of one source file, in file order."""
    lines = source.splitlines()
    spans = _function_spans(source)
    headers: list[Header] = []
    i = 0
    while i < len(lines):
        m = _HEADER_RE.match(lines[i])
        if m is None:
            i += 1
            continue
        where = f"{file}:{i + 1}"
        kind = m.group("kind")
        if kind not in KINDS:
            raise HeaderGrammarError(f"{where}: unknown header kind {kind}. (one of {', '.join(KINDS)})")
        start = i
        parts = [m.group("text")]
        while not parts[-1].endswith("─"):
            i += 1
            c = _CONTINUATION_RE.match(lines[i]) if i < len(lines) else None
            if c is None:
                raise HeaderGrammarError(
                    f"{where}: the header names do not end in '─' (continuation lines are '#' + 13 spaces)")
            parts.append(c.group("text"))
        title = " ".join(parts).rstrip("─").strip()
        i += 1
        body: list[str] = []
        names_line: str | None = None
        while i < len(lines) and lines[i].lstrip().startswith("#") and not _ANY_RULE_HEADER_RE.match(lines[i]):
            text = _comment_text(lines[i])
            if not body and names_line is None and (n := _NAMES_LINE_RE.match(text)):
                names_line = n.group("names")
            elif text.strip() and set(text.strip()) == {"─"}:
                pass                                   # the rule under the header
            else:
                body.append(text.rstrip())
            i += 1
        names = expand_names(names_line if names_line is not None else title, where)
        formula, notes = split_body(body)
        headers.append(Header(kind=kind, title=title, names=names, formula=formula, notes=notes, file=file,
                              line=start + 1, function=_enclosing_function(spans, start + 1), module=module))
    return headers


# ---------------------------------------------------------------------------
# Sources


def _rel(path: Path) -> str:
    return path.resolve().relative_to(PACKAGE_DIR.parent).as_posix()


def module_order() -> list[str]:
    """Model modules in the ``build_flextool`` table-of-contents order (core first), then the modules with
    headers that the table does not name, alphabetically."""
    tree = ast.parse((MODEL_DIR / "core.py").read_text(encoding="utf-8"))
    order = ["core"]
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "build_flextool":
            for sub in ast.walk(node):
                if isinstance(sub, ast.For) and isinstance(sub.iter, ast.Tuple):
                    for elt in sub.iter.elts:
                        if (isinstance(elt, ast.Attribute) and isinstance(elt.value, ast.Name)
                                and elt.value.id not in order):
                            order.append(elt.value.id)
    rest = sorted(p.stem for p in MODEL_DIR.glob("*.py") if p.stem not in order and p.stem != "__init__")
    return order + rest


def collect_headers() -> dict[str, list[Header]]:
    """Headers by module, in page order; modules without a header are left out."""
    out: dict[str, list[Header]] = {}
    for module in module_order():
        path = MODEL_DIR / f"{module}.py"
        hs = parse_headers(path.read_text(encoding="utf-8"), _rel(path), module)
        if hs:
            out[module] = hs
    out["_benders"] = parse_headers(BENDERS_FILE.read_text(encoding="utf-8"), _rel(BENDERS_FILE), "_benders")
    return out


def all_headers() -> list[Header]:
    return [h for hs in collect_headers().values() for h in hs]


# ---------------------------------------------------------------------------
# Rendering


def _md_escape(text: str) -> str:
    return re.sub(r"([\\`*_\[\]<>|#])", r"\\\1", text)


def _module_title(module: str) -> tuple[str, str]:
    """(section title, the module docstring's first line)."""
    if module == "_benders":
        path = BENDERS_FILE
    else:
        path = MODEL_DIR / f"{module}.py"
    doc = ast.get_docstring(ast.parse(path.read_text(encoding="utf-8"))) or ""
    first = doc.strip().splitlines()[0] if doc.strip() else ""
    if module == "core":
        return "Core model", first
    if module == "_benders":
        return BENDERS_TITLE, first
    title = first.split(":", 1)[0] if ":" in first else first.rstrip(".")
    return title.replace("``", ""), first


def _render_notes(notes: str) -> list[str]:
    """Prose as paragraphs; an indented line in it (a formula fragment) as a code block."""
    out: list[str] = []
    para: list[str] = []
    code: list[str] = []

    def flush_para() -> None:
        if para:
            out.extend([_md_escape(" ".join(s.strip() for s in para)), ""])
            para.clear()

    def flush_code() -> None:
        if code:
            out.extend(["```text", *code, "```", ""])
            code.clear()

    for line in notes.splitlines():
        if not line.strip():
            flush_para()
            flush_code()
        elif line.startswith("  "):
            flush_para()
            code.append(line)
        else:
            flush_code()
            para.append(line)
    flush_para()
    flush_code()
    return out


def section_text(markdown: str, title: str) -> str:
    """The body of the ``#``–``###`` section titled ``title`` (fence-aware), without its heading."""
    lines = markdown.splitlines()
    fence = False
    start: int | None = None
    for i, line in enumerate(lines):
        if line.lstrip().startswith("```"):
            fence = not fence
            continue
        if fence:
            continue
        h = re.match(r"^(#{1,3}) (.*?)\s*$", line)
        if h is None:
            continue
        if start is not None:
            return "\n".join(lines[start:i]).strip("\n")
        if h.group(2) == title:
            start = i + 1
    if start is None:
        raise ValueError(f"no section {title!r}")
    return "\n".join(lines[start:]).strip("\n")


def render_formulation(markdown_root: Path) -> str:
    """The formulation page (Markdown)."""
    engine_md = (markdown_root / scope.ENGINE_DOC).read_text(encoding="utf-8")
    preamble = section_text(engine_md, scope.FORMULATION_PREAMBLE_SECTION)
    out = [
        "# Model formulation",
        "",
        "Every constraint, variable and objective term of the FlexTool LP, taken from the `EQ.` / `VAR.` /",
        "`OBJ.` headers in the model code. Each entry names the source file and line it was generated from.",
        "Names in braces expand: `maxOnline_{linear,integer}` is `maxOnline_linear` and `maxOnline_integer`,",
        "`{sfx}` is `_linear` / `_integer`, and a part in square brackets is optional (`co2_max_total[_path]`).",
        "Capacities are in MW (MWh for storage) and costs in CUR, the default units of FlexTool",
        "(see [Model parameters](reference.md)).",
        "",
        f"## {scope.FORMULATION_PREAMBLE_SECTION}",
        "",
        preamble,
        "",
    ]
    for module, headers in collect_headers().items():
        title, first = _module_title(module)
        out += [f"## {title}", ""]
        if first:
            out += [_md_escape(first.replace("``", "")), ""]
        for h in headers:
            out += [f"### {h.kind}. `{h.title}`", ""]
            if h.names != (h.title,):
                out += ["Names: " + ", ".join(f"`{n}`" for n in h.names), ""]
            if h.formula:
                out += ["```text", h.formula, "```", ""]
            out += _render_notes(h.notes)
            where = f"`{h.file}:{h.line}`"
            if h.function:
                where += f" (`{h.function}`)"
            out += [f"Source: {where}", ""]
    return "\n".join(out).rstrip("\n") + "\n"
