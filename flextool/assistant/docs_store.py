"""The documentation the assistant reads and searches, generated on first use and cached per user.

Markdown root (:func:`markdown_root`): a ``flextool`` package that sits in a clone (``mkdocs.yml`` and
``docs/`` next to it: an editable install, also the Toolbox bundle) reads the clone's ``docs/``; any other
install reads the copy the build step put in the package (``flextool/assistant/_docs/md``, from
``_flextool_build.py``).  Neither → :data:`NOT_INCLUDED`.

First use (:func:`load`): the cache key is the FlexTool version and a hash of every file the index is made
from (:func:`source_files`).  The cache folder ``<user cache dir>/assistant_docs/<version>-<hash12>`` is read
when it exists; otherwise ``flextool.docs_build.index.generate`` writes the generated pages, the search index
and the records into a temporary folder that is then renamed into place (a process that loses the race uses
the winner's folder).  The newest :data:`KEEP` folders are kept.  A failure writes its traceback to stderr and
every later call answers :data:`BUILD_FAILED` until the process restarts; nothing else is affected.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import sys
import tempfile
import threading
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path

from flextool.docs_build import scope
from flextool.docs_build import index as docs_index
from flextool.assistant.textsearch import BM25, tokens

PACKAGE_DIR = Path(__file__).resolve().parents[1]          # .../flextool
PACKAGED_MARKDOWN = Path("assistant") / "_docs" / "md"      # under PACKAGE_DIR (_flextool_build.PACKAGED_DOCS)
CACHE_SUBDIR = "assistant_docs"
KEEP = 3
STALE_BUILD_SECONDS = 3600      # a temporary build folder older than this belongs to a process that died
SNIPPET_MAX = 300

NOT_INCLUDED = "The FlexTool documentation is not included in this installation."
BUILD_FAILED = "The documentation index could not be built: {reason}."

KINDS = ("docs", "formulations", "all")


class DocsUnavailable(RuntimeError):
    """The documentation cannot be served; ``str(exc)`` is the message for the user."""


# ---------------------------------------------------------------------------
# Where the sources are, and the cache key


def markdown_root(package_dir: Path = PACKAGE_DIR) -> Path | None:
    """The folder the curated Markdown is read from, or ``None`` when the installation has none."""
    clone = package_dir.parent
    if (clone / "mkdocs.yml").is_file() and (clone / "docs").is_dir():
        return clone / "docs"
    packaged = package_dir / PACKAGED_MARKDOWN
    if packaged.is_dir():
        return packaged
    return None


def source_files(markdown_root: Path, package_dir: Path = PACKAGE_DIR) -> list[tuple[str, Path]]:
    """``[(label, path)]`` of every file the generated documentation depends on, sorted by label: the curated
    Markdown, the model code and the Benders master (formulation), the output metadata, the results-database
    spec and the default output settings (catalogue), and the generator and tokenizer themselves."""
    out: list[tuple[str, Path]] = []
    for rel in scope.CURATED_DOCS:
        path = markdown_root / rel
        if path.is_file():
            out.append((f"md/{rel}", path))
    patterns = (
        "engine/model/*.py",
        "engine/solve/_benders.py",
        "process_outputs/_output_meta.py",
        "process_outputs/write_spinedb.py",
        "schemas/default_plots.yaml",
        "docs_build/*.py",
        "assistant/textsearch.py",
    )
    for pattern in patterns:
        for path in sorted(package_dir.glob(pattern)):
            out.append((f"pkg/{path.relative_to(package_dir).as_posix()}", path))
    return sorted(out)


def source_hash(markdown_root: Path, package_dir: Path = PACKAGE_DIR) -> str:
    """sha256 over the labels and bytes of :func:`source_files`."""
    digest = hashlib.sha256()
    for label, path in source_files(markdown_root, package_dir):
        data = path.read_bytes()
        digest.update(label.encode("utf-8") + b"\0" + str(len(data)).encode("ascii") + b"\0")
        digest.update(data)
    return digest.hexdigest()


def flextool_version() -> str:
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version("flextool")
    except PackageNotFoundError:
        return "unknown"


def cache_key(markdown_root: Path, package_dir: Path = PACKAGE_DIR, version: str | None = None) -> str:
    """``<FlexTool version>-<first 12 characters of the source hash>``."""
    v = re.sub(r"[^\w.+-]", "_", version if version is not None else flextool_version())
    return f"{v}-{source_hash(markdown_root, package_dir)[:12]}"


def cache_root() -> Path:
    """``<user cache dir of flextool>/assistant_docs``."""
    from flextool.user_dirs import user_cache_dir

    return user_cache_dir() / CACHE_SUBDIR


# ---------------------------------------------------------------------------
# The loaded documentation


@dataclass
class DocsStore:
    """Everything the docs tools serve, in memory: documents, search chunks, formulation and result records."""

    key: str
    documents: dict[str, str]                 # document name → Markdown, in reading order
    chunks: list[dict]
    formulations: list[dict]
    catalogue: list[dict]
    _docs_bm25: BM25 = field(init=False, repr=False)
    _formulation_bm25: BM25 = field(init=False, repr=False)
    _by_name: dict[str, dict] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._docs_bm25 = BM25(c["terms"] for c in self.chunks)
        self._formulation_bm25 = BM25(r["terms"] for r in self.formulations)
        self._by_name = {}
        for r in self.formulations:
            for name in (r["name"], *r["names"]):
                self._by_name.setdefault(name, r)

    def doc_names(self) -> list[str]:
        return list(self.documents)

    def formulation(self, name: str) -> dict | None:
        """The formulation record whose header or expanded name is ``name``."""
        return self._by_name.get(name)

    def result_table(self, stem: str) -> dict | None:
        return next((r for r in self.catalogue if r["stem"] == stem), None)

    def search(self, query: str, kind: str = "all", max_results: int = 5) -> list[dict]:
        """The best matches, best first: documentation chunks ``{kind: "docs", doc, section, anchor, snippet}``
        and model headers ``{kind: "formulations", name, module, line, formula_first_line}``."""
        if kind not in KINDS:
            raise ValueError(f"kind must be one of {', '.join(KINDS)}")
        scored: list[tuple[float, int, dict]] = []
        if kind in ("docs", "all"):
            for i, score in self._docs_bm25.search(query, max_results):
                scored.append((score, 0, self._docs_hit(self.chunks[i], query)))
        if kind in ("formulations", "all"):
            for i, score in self._formulation_bm25.search(query, max_results):
                scored.append((score, 1, self._formulation_hit(self.formulations[i])))
        scored.sort(key=lambda item: (-item[0], item[1]))
        return [hit for _, _, hit in scored[:max_results]]

    @staticmethod
    def _docs_hit(chunk: dict, query: str) -> dict:
        return {"kind": "docs", "doc": chunk["doc"], "section": chunk["section"], "anchor": chunk["anchor"],
                "snippet": snippet(chunk["text"], query)}

    @staticmethod
    def _formulation_hit(record: dict) -> dict:
        first = record["formula"].splitlines()[0].strip() if record["formula"] else ""
        return {"kind": "formulations", "name": record["name"], "module": record["module"],
                "line": record["line"], "formula_first_line": first}


def snippet(text: str, query: str, limit: int = SNIPPET_MAX) -> str:
    """At most ``limit`` characters of ``text`` (whitespace collapsed) around the first query term it
    contains; the start of the text when it contains none."""
    flat = " ".join(text.split())
    if len(flat) <= limit:
        return flat
    lower = flat.lower()
    hits = [lower.find(t) for t in sorted(set(tokens(query)), key=len, reverse=True)]
    hits = [h for h in hits if h >= 0]
    start = max(0, min(hits) - limit // 5) if hits else 0
    end = min(len(flat), start + limit - 2)
    start = max(0, min(start, end - (limit - 2)))
    out = flat[start:end]
    return ("…" if start > 0 else "") + out + ("…" if end < len(flat) else "")


# ---------------------------------------------------------------------------
# Loading and the cache


def _read_json(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or data.get("format") != docs_index.INDEX_FORMAT:
        raise ValueError(f"{path.name}: not an index of this FlexTool")
    return data


def read_cache(folder: Path, key: str, root: Path) -> DocsStore:
    """The store from a complete cache folder; ``ValueError`` / ``OSError`` when it is not usable."""
    manifest = _read_json(folder / docs_index.MANIFEST_FILE)
    if manifest.get("key") != key:
        raise ValueError("manifest.json: another cache key")
    pages = {name: (folder / name).read_text(encoding="utf-8") for name in scope.GENERATED_PAGES}
    return DocsStore(
        key=key,
        documents=docs_index.documents(root, pages),
        chunks=_read_json(folder / docs_index.INDEX_FILE)["chunks"],
        formulations=_read_json(folder / docs_index.FORMULATIONS_FILE)["records"],
        catalogue=_read_json(folder / docs_index.CATALOGUE_FILE)["records"],
    )


def build_cache(root: Path, cache: Path, key: str, version: str) -> Path:
    """Generate the cache folder ``cache/key`` through a temporary sibling renamed into place; return it."""
    cache.mkdir(parents=True, exist_ok=True)
    target = cache / key
    tmp = Path(tempfile.mkdtemp(prefix=".building-", dir=cache))
    try:
        docs_index.generate(root, tmp, manifest={"key": key, "flextool_version": version})
        try:
            os.replace(tmp, target)
        except OSError:
            if not (target / docs_index.MANIFEST_FILE).is_file():
                raise
            # Another process built the same key first: use its folder.
    finally:
        if tmp.exists():
            shutil.rmtree(tmp, ignore_errors=True)
    return target


def prune_cache(cache: Path, keep_folder: Path, keep: int = KEEP) -> None:
    """Keep ``keep_folder`` and the newest others up to ``keep`` folders in all; remove temporary build
    folders that a process left behind."""
    now = time.time()
    folders: list[tuple[float, Path]] = []
    for path in cache.iterdir():
        if not path.is_dir():
            continue
        try:
            mtime = path.stat().st_mtime
        except OSError:
            continue
        if path.name.startswith(".building-"):
            if now - mtime > STALE_BUILD_SECONDS:
                shutil.rmtree(path, ignore_errors=True)
        elif path != keep_folder:
            folders.append((mtime, path))
    folders.sort(key=lambda item: item[0], reverse=True)
    for _, path in folders[max(keep - 1, 0):]:
        shutil.rmtree(path, ignore_errors=True)


def open_store(root: Path, cache: Path, package_dir: Path = PACKAGE_DIR) -> DocsStore:
    """The store for ``root``: from the cache folder of its key, else generated into it first."""
    version = flextool_version()
    key = cache_key(root, package_dir, version)
    folder = cache / key
    if (folder / docs_index.MANIFEST_FILE).is_file():
        try:
            return read_cache(folder, key, root)
        except (OSError, ValueError, KeyError):
            shutil.rmtree(folder, ignore_errors=True)       # damaged by hand: build it again
    started = time.perf_counter()
    folder = build_cache(root, cache, key, version)
    prune_cache(cache, folder)
    store = read_cache(folder, key, root)
    print(f"FlexTool built its documentation index in {time.perf_counter() - started:.1f} s.",
          file=sys.stderr, flush=True)
    return store


_lock = threading.Lock()
_store: DocsStore | None = None
_failure: str | None = None


def load() -> DocsStore:
    """The documentation for this process, loaded on the first call.  Raises :class:`DocsUnavailable` with
    the message for the user when the installation has no documentation or the index could not be built."""
    global _store, _failure
    with _lock:
        if _store is not None:
            return _store
        if _failure is not None:
            raise DocsUnavailable(_failure)
        root = markdown_root()
        if root is None:
            raise DocsUnavailable(NOT_INCLUDED)
        try:
            _store = open_store(root, cache_root())
        except Exception as exc:  # noqa: BLE001 — any failure only disables the docs tools
            traceback.print_exc(file=sys.stderr)
            _failure = BUILD_FAILED.format(reason=str(exc).strip().splitlines()[0] if str(exc).strip()
                                           else type(exc).__name__)
            raise DocsUnavailable(_failure) from None
        return _store


def reset() -> None:
    """Forget the loaded documentation and an earlier failure (as a restart does)."""
    global _store, _failure
    with _lock:
        _store = None
        _failure = None
