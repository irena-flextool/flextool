"""The assistant's documentation index: the documents it reads, their search chunks and the generated records.

``generate(markdown_root, out_dir)`` writes, as UTF-8 with LF line ends:

- ``formulation.md`` and ``results_catalogue.md`` — the generated pages (``render_pages``), from the
  installed code, so their source lines match it;
- ``docs_index.json`` — the search chunks of every shipped document, with their BM25 term counts;
- ``formulations.json`` — one record per model header (``{name, names, kind, module, file, line, formula,
  notes, function}``);
- ``result_catalog.json`` — one record per result table (``catalogue.records``);
- ``manifest.json`` last, when one is given (``flextool.assistant.docs_store`` passes its cache key).

Documents (``documents``): the user pages of the scope, the two generated pages, the developer pages, and of
``dev/engine.md`` only the sections ``scope.ENGINE_SECTIONS`` (with their subsections).  Chunks start at every
``#``–``###`` heading outside code fences; a chunk longer than :data:`CHUNK_MAX` characters is split into its
paragraphs, and a paragraph longer than that into its list items (``_split_long``).  Anchors are the heading ids of the web docs (python-markdown ``toc``: slug, ``_1``, ``_2``… for
repeats, counted over the whole page).  Heading terms weigh double in BM25, and so do the names that a
definition bullet introduces (``defined_names``).
"""
from __future__ import annotations

import html
import json
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path

from flextool.assistant.textsearch import term_counts
from flextool.docs_build import render_pages, scope, write_text_atomic

CHUNK_MAX = 6000
CHUNK_LEVELS = 3                # headings of this level and above start a chunk
INDEX_FILE = "docs_index.json"
FORMULATIONS_FILE = "formulations.json"
CATALOGUE_FILE = "result_catalog.json"
MANIFEST_FILE = "manifest.json"
INDEX_FORMAT = 1                # bump when the layout of the JSON files changes

_ATX_RE = re.compile(r"^(?P<hashes>#{1,6})(?P<text>.*?)#*\s*$")
# A fence line: three or more backticks or tildes; an opening backtick fence's info string has no backtick
# (a line such as ```code``` is inline code, not a fence).
_FENCE_RE = re.compile(r"^\s*(?P<fence>`{3,}(?=[^`]*$)|~{3,})")
_IDCOUNT_RE = re.compile(r"^(.*)_([0-9]+)$")
_ITEM_RE = re.compile(r"^(?:[-*+]|[0-9]+[.)])[ \t]")
_DEFINITION_RE = re.compile(r"^[ \t]*[-*][ \t]+(?P<names>`[^`\n]+`(?:[ \t]*[/,][ \t]*`[^`\n]+`)*)[ \t]+[-–—:]",
                            re.MULTILINE)


class _Fences:
    """Tracks whether a line is inside a fenced code block (a fence closes with the same character, at least
    as long, and nothing after it)."""

    def __init__(self) -> None:
        self.open: str | None = None

    def step(self, line: str) -> bool:
        """Feed the next line; True when it is a fence line or inside a fence."""
        m = _FENCE_RE.match(line)
        if self.open is None:
            if m:
                self.open = m.group("fence")
                return True
            return False
        if m and m.group("fence")[0] == self.open[0] and len(m.group("fence")) >= len(self.open) \
                and not line.strip().lstrip(self.open[0]):
            self.open = None
        return True


@dataclass(frozen=True)
class Heading:
    line: int            # 0-based line of the heading
    level: int
    title: str           # the heading as the reader sees it (inline Markdown removed)
    anchor: str


# ---------------------------------------------------------------------------
# Headings and anchors


def heading_text(raw: str) -> str:
    """The text of a heading's inline Markdown as rendered: code spans kept literally, links reduced to their
    text, images dropped, emphasis markers, escapes and HTML tags removed, entities decoded."""
    out: list[str] = []
    pos = 0
    for m in re.finditer(r"(`+)(.+?)\1", raw):
        out.append(_strip_inline(raw[pos:m.start()]))
        out.append(m.group(2).strip())
        pos = m.end()
    out.append(_strip_inline(raw[pos:]))
    return html.unescape("".join(out)).strip()


def _strip_inline(text: str) -> str:
    text = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", text)
    text = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", text)
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"(\*{1,3})(\S(?:.*?\S)?)\1", r"\2", text)
    text = re.sub(r"(?<!\w)(_{1,3})(\S(?:.*?\S)?)\1(?!\w)", r"\2", text)
    return re.sub(r"\\([\\`*_{}\[\]()#+\-.!])", r"\1", text)


def slugify(title: str) -> str:
    """python-markdown ``toc.slugify`` with its default separator."""
    value = unicodedata.normalize("NFKD", title).encode("ascii", "ignore").decode("ascii")
    value = re.sub(r"[^\w\s-]", "", value).strip().lower()
    return re.sub(r"[-\s]+", "-", value)


def _unique(anchor: str, used: set[str]) -> str:
    while anchor in used or not anchor:
        m = _IDCOUNT_RE.match(anchor)
        anchor = f"{m.group(1)}_{int(m.group(2)) + 1}" if m else f"{anchor}_1"
    used.add(anchor)
    return anchor


def headings(markdown: str) -> list[Heading]:
    """Every ATX heading outside code fences, with its web-docs anchor."""
    out: list[Heading] = []
    used: set[str] = set()
    fences = _Fences()
    for i, line in enumerate(markdown.splitlines()):
        if fences.step(line):
            continue
        m = _ATX_RE.match(line)
        if m is None:
            continue
        title = heading_text(m.group("text").strip())
        out.append(Heading(i, len(m.group("hashes")), title, _unique(slugify(title), used)))
    return out


# ---------------------------------------------------------------------------
# Documents


def section_spans(markdown: str, titles: tuple[str, ...]) -> list[tuple[int, int]]:
    """``[(first line, end line)]`` of the sections titled ``titles`` (each with its subsections), in page
    order.  A title the page lacks raises ``ValueError``."""
    hs = headings(markdown)
    n_lines = len(markdown.splitlines())
    spans: list[tuple[int, int]] = []
    found: set[str] = set()
    for k, h in enumerate(hs):
        if h.title not in titles:
            continue
        found.add(h.title)
        end = next((later.line for later in hs[k + 1:] if later.level <= h.level), n_lines)
        spans.append((h.line, end))
    missing = [t for t in titles if t not in found]
    if missing:
        raise ValueError(f"sections missing from the page: {', '.join(missing)}")
    return spans


def _engine_text(markdown: str) -> str:
    lines = markdown.splitlines()
    parts = ["\n".join(lines[a:b]).strip("\n") for a, b in section_spans(markdown, scope.ENGINE_SECTIONS)]
    return "\n\n".join(parts) + "\n"


def documents(markdown_root: Path, pages: dict[str, str]) -> dict[str, str]:
    """``{document name: Markdown}`` of everything the assistant reads, in reading order: the user pages, the
    generated pages, the developer pages; ``dev/engine.md`` holds only its shipped sections."""
    out: dict[str, str] = {}
    for rel in scope.USER_DOCS:
        out[rel] = (markdown_root / rel).read_text(encoding="utf-8")
    for name in scope.GENERATED_PAGES:
        out[name] = pages[name]
    for rel in scope.DEV_DOCS:
        out[rel] = (markdown_root / rel).read_text(encoding="utf-8")
    out[scope.ENGINE_DOC] = _engine_text((markdown_root / scope.ENGINE_DOC).read_text(encoding="utf-8"))
    return out


# ---------------------------------------------------------------------------
# Chunks


def _size(lines: list[str]) -> int:
    return sum(len(x) + 1 for x in lines)


def _split_long(lines: list[str]) -> list[list[str]]:
    """A section body of at most :data:`CHUNK_MAX` characters is one part.  A longer one is split into its
    paragraphs (at blank lines outside code fences), and a paragraph longer than that into its top-level
    list items (the parameter lists of reference.md: one definition per part)."""
    if _size(lines) <= CHUNK_MAX:
        return [lines]
    paragraphs: list[list[str]] = [[]]
    fences = _Fences()
    for line in lines:
        fence = fences.step(line)
        if not fence and not line.strip() and paragraphs[-1]:
            paragraphs.append([])
        paragraphs[-1].append(line)
    parts: list[list[str]] = []
    for para in paragraphs:
        parts.extend(_list_items(para) if _size(para) > CHUNK_MAX else [para])
    return [p for p in parts if "".join(p).strip()]


def _list_items(paragraph: list[str]) -> list[list[str]]:
    """The paragraph split before every top-level list item outside code fences (the lines before the first
    item stay together)."""
    items: list[list[str]] = [[]]
    fences = _Fences()
    for line in paragraph:
        fence = fences.step(line)
        if not fence and _ITEM_RE.match(line) and items[-1]:
            items.append([])
        items[-1].append(line)
    return items


def defined_names(text: str) -> list[str]:
    """The names that definition bullets of ``text`` introduce: ``- `penalty_up` - …`` or ``- `a` / `b` - …``
    (the parameter lists of reference.md).  They weigh like heading terms: the passage that defines a name
    outranks a longer one that merely uses it."""
    out: list[str] = []
    for m in _DEFINITION_RE.finditer(text):
        out.extend(re.findall(r"`([^`]+)`", m.group("names")))
    return out


def chunks(doc: str, markdown: str) -> list[dict]:
    """The search chunks of one document: ``{doc, section, anchor, level, part, text, terms}``."""
    lines = markdown.splitlines()
    starts = [h for h in headings(markdown) if h.level <= CHUNK_LEVELS]
    bounds: list[tuple[Heading | None, int, int]] = []
    first = starts[0].line if starts else len(lines)
    if "\n".join(lines[:first]).strip():
        bounds.append((None, 0, first))
    for k, h in enumerate(starts):
        end = starts[k + 1].line if k + 1 < len(starts) else len(lines)
        bounds.append((h, h.line, end))
    out: list[dict] = []
    for h, a, b in bounds:
        body = lines[a + 1:b] if h is not None else lines[a:b]
        section = h.title if h is not None else ""
        for part, piece in enumerate(_split_long(body) if body else [[]]):
            text = "\n".join(piece).strip("\n")
            if not text and h is None:
                continue
            out.append({
                "doc": doc,
                "section": section,
                "anchor": h.anchor if h is not None else "",
                "level": h.level if h is not None else 0,
                "part": part,
                "text": text,
                "terms": term_counts(text, " ".join([section, *defined_names(text)])),
            })
    return out


def build_index(docs: dict[str, str]) -> dict:
    return {"format": INDEX_FORMAT, "chunks": [c for doc, md in docs.items() for c in chunks(doc, md)]}


# ---------------------------------------------------------------------------
# Records


def formulation_records() -> list[dict]:
    """One record per ``EQ.`` / ``VAR.`` / ``OBJ.`` header of the installed model code, in page order."""
    from flextool.docs_build.formulation import all_headers

    return [{
        "name": h.title,
        "names": list(h.names),
        "kind": h.kind,
        "module": h.module,
        "file": h.file,
        "line": h.line,
        "formula": h.formula,
        "notes": h.notes,
        "function": h.function,
        "terms": term_counts(" ".join([h.formula, h.notes, h.function or ""]), " ".join([h.title, *h.names])),
    } for h in all_headers()]


def _dump(data: object) -> str:
    return json.dumps(data, ensure_ascii=False, separators=(",", ":")) + "\n"


def generate(markdown_root: Path, out_dir: Path, manifest: dict | None = None) -> list[Path]:
    """Render the generated pages and write them, the index and the records into ``out_dir`` (created);
    ``manifest.json`` goes last, so a folder with a manifest is complete.  Return the written paths."""
    from flextool.docs_build.catalogue import records

    pages = render_pages(markdown_root)
    files: dict[str, str] = dict(pages)
    files[INDEX_FILE] = _dump(build_index(documents(markdown_root, pages)))
    files[FORMULATIONS_FILE] = _dump({"format": INDEX_FORMAT, "records": formulation_records()})
    files[CATALOGUE_FILE] = _dump({"format": INDEX_FORMAT, "records": records(markdown_root)})
    if manifest is not None:
        files[MANIFEST_FILE] = _dump({**manifest, "format": INDEX_FORMAT, "files": sorted(files)})
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for name, text in files.items():
        path = out_dir / name
        write_text_atomic(path, text)
        written.append(path)
    return written
