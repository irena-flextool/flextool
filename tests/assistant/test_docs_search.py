"""The assistant's documentation: search, the Markdown root of each kind of install, and the first-use cache.

The index is generated from this checkout's docs and code into a tmp cache folder (the user cache folder is
redirected by ``conftest.py``).
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from flextool.assistant import docs_store
from flextool.docs_build import index as docs_index
from flextool.docs_build import scope

REPO = Path(__file__).resolve().parents[2]
DOCS = REPO / "docs"


@pytest.fixture(scope="module")
def store(tmp_path_factory: pytest.TempPathFactory) -> docs_store.DocsStore:
    return docs_store.open_store(DOCS, tmp_path_factory.mktemp("docs_cache"))


def test_a_parameter_name_finds_its_reference_entry(store: docs_store.DocsStore) -> None:
    hits = store.search("penalty_up", kind="docs", max_results=3)
    assert [h["kind"] for h in hits] == ["docs"] * 3
    ref = [h for h in hits if h["doc"] == "reference/nodes.md"]
    assert ref, hits
    assert "`penalty_up`" in ref[0]["snippet"] and len(ref[0]["snippet"]) <= docs_store.SNIPPET_MAX
    assert ref[0]["anchor"] == "data-for-nodes"


def test_formulation_search_finds_an_expanded_name(store: docs_store.DocsStore) -> None:
    hits = store.search("maxOnline_linear", kind="formulations")
    assert hits[0]["kind"] == "formulations"
    assert hits[0]["name"] == "maxOnline_{linear,integer}"
    assert hits[0]["module"] == "_online"
    record = store.formulation("maxOnline_linear")
    assert record is store.formulation("maxOnline_{linear,integer}")
    assert record["names"] == ["maxOnline_linear", "maxOnline_integer"]
    assert record["file"] == "flextool/engine/model/_online.py"
    source = (REPO / record["file"]).read_text(encoding="utf-8").splitlines()
    assert "maxOnline_{linear,integer}" in source[record["line"] - 1]
    assert store.search("maxOnline_linear", kind="all")[0]["kind"] in ("docs", "formulations")
    with pytest.raises(ValueError):
        store.search("x", kind="tables")


def test_documents_are_the_scope_and_engine_md_only_its_shipped_sections(store: docs_store.DocsStore) -> None:
    assert store.doc_names() == [*scope.USER_DOCS, *scope.GENERATED_PAGES, *scope.DEV_DOCS, scope.ENGINE_DOC]
    titles = [h.title for h in docs_index.headings(store.documents[scope.ENGINE_DOC]) if h.level == 2]
    assert titles == list(scope.ENGINE_SECTIONS)
    engine_sections = {c["section"] for c in store.chunks if c["doc"] == scope.ENGINE_DOC}
    assert "Equation headers" not in engine_sections and "Fix-storage semantics (Deferred-B wiring)" in \
        engine_sections
    assert store.result_table("unit_outputNode_dt_ee")["topic"] == "Energy flows"
    assert max(len(c["text"]) for c in store.chunks if c["doc"].startswith("reference/")) <= 2 * docs_index.CHUNK_MAX


def test_anchors_are_those_of_the_web_docs() -> None:
    """Every heading id equals python-markdown's toc id under the web docs' extensions (repeats get _1, _2)."""
    markdown = pytest.importorskip("markdown")
    pytest.importorskip("pymdownx")
    texts = {rel: (DOCS / rel).read_text(encoding="utf-8") for rel in scope.CURATED_DOCS}
    texts.update(docs_index.render_pages(DOCS))
    for rel, text in texts.items():
        md = markdown.Markdown(extensions=["toc", "extra", "tables", "fenced_code", "admonition",
                                           "pymdownx.details", "pymdownx.superfences"])
        md.convert(text)

        def flat(tokens: list[dict]) -> list[tuple[int, str]]:
            return [x for t in tokens for x in [(t["level"], t["id"]), *flat(t["children"])]]

        assert [(h.level, h.anchor) for h in docs_index.headings(text)] == flat(md.toc_tokens), rel


def test_markdown_root_of_a_clone_and_of_a_packaged_install(tmp_path: Path) -> None:
    """A clone, or an unpacked sdist (the scope's Markdown, no mkdocs.yml), is found by the scope file
    ``docs/dev/engine.md``; a ``docs/`` folder without it is not a clone (review C2-6)."""
    sdist = tmp_path / "flextool-4.2.2"
    (sdist / "flextool").mkdir(parents=True)
    (sdist / "docs" / "dev").mkdir(parents=True)
    (sdist / "docs" / scope.ENGINE_DOC).write_text("# The engine\n", encoding="utf-8")
    assert not (sdist / "mkdocs.yml").exists()
    assert docs_store.markdown_root(sdist / "flextool") == sdist / "docs"
    other = tmp_path / "other"
    (other / "flextool").mkdir(parents=True)
    (other / "docs").mkdir()
    (other / "mkdocs.yml").write_text("site_name: x\n", encoding="utf-8")
    assert docs_store.markdown_root(other / "flextool") is None
    site = tmp_path / "site-packages"
    packaged = site / "flextool" / "assistant" / "_docs" / "md"
    packaged.mkdir(parents=True)
    assert docs_store.markdown_root(site / "flextool") == packaged
    bare = tmp_path / "bare" / "flextool"
    bare.mkdir(parents=True)
    assert docs_store.markdown_root(bare) is None


def test_no_documentation_in_the_installation(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(docs_store, "markdown_root", lambda package_dir=docs_store.PACKAGE_DIR: None)
    with pytest.raises(docs_store.DocsUnavailable, match="^The FlexTool documentation is not included in this "
                                                         "installation.$"):
        docs_store.load()


def test_first_use_builds_the_cache_and_later_loads_read_it(user_cache_in_tmp: Path,
                                                            monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[Path] = []
    real_generate = docs_index.generate

    def counting_generate(root: Path, out_dir: Path, manifest: dict | None = None) -> list[Path]:
        calls.append(out_dir)
        return real_generate(root, out_dir, manifest)

    monkeypatch.setattr(docs_index, "generate", counting_generate)
    first = docs_store.load()
    assert docs_store.load() is first
    folder = user_cache_in_tmp / docs_store.CACHE_SUBDIR / first.key
    assert first.key == docs_store.cache_key(docs_store.markdown_root())
    assert first.key.startswith(f"{docs_store.flextool_version()}-")
    names = sorted(p.name for p in folder.iterdir())
    assert names == sorted([*scope.GENERATED_PAGES, docs_index.INDEX_FILE, docs_index.FORMULATIONS_FILE,
                            docs_index.CATALOGUE_FILE, docs_index.MANIFEST_FILE])
    for path in folder.iterdir():
        raw = path.read_bytes()
        assert b"\r" not in raw
        raw.decode("utf-8")
    assert [p.name for p in folder.parent.iterdir()] == [first.key]      # the temporary folder is gone
    docs_store.reset()                                                   # as a restart
    second = docs_store.load()
    assert len(calls) == 1 and second.key == first.key
    assert second.search("penalty_up", "docs", 3) == first.search("penalty_up", "docs", 3)


def test_the_cache_key_follows_the_model_code(tmp_path: Path) -> None:
    package = tmp_path / "flextool"
    model = package / "engine" / "model"
    model.mkdir(parents=True)
    core = model / "core.py"
    core.write_text("x = 1\n", encoding="utf-8")
    key = docs_store.cache_key(DOCS, package, version="9.9")
    assert key == docs_store.cache_key(DOCS, package, version="9.9")
    core.write_text("x = 2\n", encoding="utf-8")
    changed = docs_store.cache_key(DOCS, package, version="9.9")
    assert changed != key and changed.startswith("9.9-")
    assert docs_store.cache_key(DOCS, package, version="10.0") != changed
    labels = [label for label, _ in docs_store.source_files(DOCS)]
    for expected in ("md/reference/nodes.md", "pkg/engine/model/core.py", "pkg/engine/solve/_benders.py",
                     "pkg/process_outputs/_output_meta.py", "pkg/process_outputs/write_spinedb.py",
                     "pkg/docs_build/index.py", "pkg/assistant/textsearch.py"):
        assert expected in labels


def test_a_generator_failure_answers_the_message_until_restart(user_cache_in_tmp: Path,
                                                                monkeypatch: pytest.MonkeyPatch,
                                                                capsys: pytest.CaptureFixture[str]) -> None:
    calls: list[Path] = []

    def failing_generate(root: Path, out_dir: Path, manifest: dict | None = None) -> list[Path]:
        calls.append(out_dir)
        raise RuntimeError("results.md cannot be read")

    monkeypatch.setattr(docs_index, "generate", failing_generate)
    message = "The documentation index could not be built: results.md cannot be read."
    for _ in range(2):
        with pytest.raises(docs_store.DocsUnavailable) as err:
            docs_store.load()
        assert str(err.value) == message
    assert len(calls) == 1
    assert "Traceback" in capsys.readouterr().err
    cache = user_cache_in_tmp / docs_store.CACHE_SUBDIR
    assert list(cache.iterdir()) == []                                   # no half-built folder


def test_a_concurrent_build_uses_the_winners_folder_and_old_folders_are_pruned(tmp_path: Path) -> None:
    cache = tmp_path / "cache"
    winner = cache / "v-key"
    winner.mkdir(parents=True)
    (winner / docs_index.MANIFEST_FILE).write_text("{}\n", encoding="utf-8")
    assert docs_store.build_cache(DOCS, cache, "v-key", "v") == winner
    assert (winner / docs_index.MANIFEST_FILE).read_text(encoding="utf-8") == "{}\n"     # not replaced
    assert [p.name for p in cache.iterdir()] == ["v-key"]
    for i, name in enumerate(["a", "b", "c", "d"]):
        (cache / name).mkdir()
        os.utime(cache / name, (1_000_000 + i, 1_000_000 + i))
    stale = cache / ".building-x"
    stale.mkdir()
    os.utime(stale, (1_000_000, 1_000_000))
    docs_store.prune_cache(cache, winner)
    assert sorted(p.name for p in cache.iterdir()) == ["c", "d", "v-key"]
