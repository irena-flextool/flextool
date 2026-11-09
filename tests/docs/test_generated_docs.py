"""The generated documentation pages: the formulation page and the results catalogue.

They render from the installed code and are written as UTF-8 with LF line ends whatever the locale; the
private-names check fails the build with a list file and is silent without one; results.md names a result
table in every result bullet; the mkdocs hook adds the pages without writing into docs/; the build step copies
exactly the curated Markdown of the scope into the package, and the sdist and package data carry it.
"""
from __future__ import annotations

import fnmatch
import os
import subprocess
import tomllib
import sys
from pathlib import Path

import pytest
import yaml

from flextool.assistant import docs_store
from flextool.docs_build import (
    PrivateNameError,
    default_private_names_path,
    render_pages,
    scope,
    write_pages,
)
from flextool.docs_build.__main__ import main as docs_build_main
from flextool.docs_build.catalogue import records, stems, unmatched_bullets

REPO = Path(__file__).resolve().parents[2]
DOCS = REPO / "docs"


def test_pages_render_and_are_written_utf8_lf(tmp_path: Path) -> None:
    written = write_pages(tmp_path, DOCS, private_names=tmp_path / "absent.txt")
    assert sorted(p.name for p in written) == sorted(scope.GENERATED_PAGES)
    for path in written:
        raw = path.read_bytes()
        assert b"\r" not in raw
        text = raw.decode("utf-8")
        assert text.endswith("\n") and not text.endswith("\n\n")
    formulation = (tmp_path / scope.FORMULATION_PAGE).read_text(encoding="utf-8")
    assert "### EQ. `maxOnline_{linear,integer}`" in formulation
    assert "Names: `maxOnline_linear`, `maxOnline_integer`" in formulation
    catalogue = (tmp_path / scope.RESULTS_CATALOGUE_PAGE).read_text(encoding="utf-8")
    assert "### `unit_outputNode_dt_ee`" in catalogue
    assert "- Results database: `unit__node` entity `flow_t` parameter" in catalogue
    assert "- Counterpart at the other time resolution: `unit_outputNode_d_ee`" in catalogue
    assert ("Membership sets are flat tables whose first column is\n`scenario`, followed by the listed row levels; "
            "`timeline_breaks` is a flat `(period, time)` table.") in catalogue


def test_pages_do_not_depend_on_the_locale_encoding(tmp_path: Path) -> None:
    """Written under an ASCII locale (as a cp1252 Windows locale would mangle the formulas' ∀, Σ, ─ instead of
    failing) the pages are byte-identical to those of this process."""
    here = tmp_path / "here"
    write_pages(here, DOCS, private_names=None)
    env = {**os.environ, "LC_ALL": "C", "LANG": "C", "PYTHONCOERCECLOCALE": "0", "PYTHONUTF8": "0",
           "PYTHONIOENCODING": "utf-8"}
    ascii_dir = tmp_path / "ascii"
    code = ("import locale, sys; from pathlib import Path; from flextool.docs_build import write_pages; "
            "print(locale.getpreferredencoding(False)); "
            "write_pages(Path(sys.argv[1]), Path(sys.argv[2]), private_names=None)")
    out = subprocess.run([sys.executable, "-X", "utf8=0", "-c", code, str(ascii_dir), str(DOCS)], env=env,
                         cwd=REPO, capture_output=True, text=True, encoding="utf-8", timeout=300)
    assert out.returncode == 0, out.stderr[-2000:]
    assert out.stdout.splitlines()[-1].lower() not in ("utf-8", "utf8")
    for name in scope.GENERATED_PAGES:
        assert (ascii_dir / name).read_bytes() == (here / name).read_bytes(), name


def test_private_names_fail_with_a_list_and_are_skipped_without(tmp_path: Path) -> None:
    names = tmp_path / ".private_names"
    names.write_text("# a comment line\nRESULTS CATALOGUE\n", encoding="utf-8")
    with pytest.raises(PrivateNameError) as err:
        write_pages(tmp_path / "out", DOCS, private_names=names)
    message = str(err.value)
    assert f"{scope.RESULTS_CATALOGUE_PAGE}:1 (private-names pattern 1)" in message
    assert "results catalogue" not in message.lower()       # the hit is named by its pattern number only
    assert not (tmp_path / "out").exists()
    write_pages(tmp_path / "out", DOCS, private_names=tmp_path / "no_such_file")
    assert (tmp_path / "out" / scope.RESULTS_CATALOGUE_PAGE).exists()


def test_private_names_also_check_the_shipped_markdown(tmp_path: Path) -> None:
    names = tmp_path / "names"
    names.write_text("Choosing an interface\n", encoding="utf-8")       # a heading of a shipped user page
    with pytest.raises(PrivateNameError, match=r"interface_overview\.md:\d+"):
        write_pages(tmp_path / "out", DOCS, private_names=names)


def test_the_private_names_list_of_a_worktree_is_the_main_checkouts(tmp_path: Path) -> None:
    main = tmp_path / "main"
    gitdir = main / ".git" / "worktrees" / "wt"
    gitdir.mkdir(parents=True)
    (gitdir / "commondir").write_text("../..\n", encoding="utf-8")
    worktree = tmp_path / "wt"
    worktree.mkdir()
    (worktree / ".git").write_text(f"gitdir: {gitdir}\n", encoding="utf-8")
    assert default_private_names_path(worktree) == main / scope.PRIVATE_NAMES_FILE
    (worktree / scope.PRIVATE_NAMES_FILE).write_text("x\n", encoding="utf-8")
    assert default_private_names_path(worktree) == worktree / scope.PRIVATE_NAMES_FILE


def test_results_md_names_a_table_in_every_result_bullet() -> None:
    """Every results.md bullet that names a result names a results-database parameter of the spec, a table or a
    CSV file of one."""
    assert [b.text for b in unmatched_bullets(DOCS)] == []


def test_the_catalogue_covers_every_table() -> None:
    recs = records(DOCS)
    assert [r["stem"] for r in sorted(recs, key=lambda r: r["stem"])] == stems()
    by_stem = {r["stem"]: r for r in recs}
    assert by_stem["unit_outputNode_dt_ee"]["topic"] == "Energy flows"
    assert by_stem["flowGroup_gd_p"]["topic"] == "Flow group results"
    assert by_stem["reserve_prices_dt_ppg"]["topic"] == "Prices"
    assert by_stem["costs_dt_p"]["csv"] == "costs__dt.csv"
    assert by_stem["group_node"]["membership_set"] is True


def test_the_mkdocs_hook_adds_the_pages_and_writes_nothing_to_docs() -> None:
    """The hook as mkdocs.yml loads it: run through the configured plugins' on_files event."""
    pytest.importorskip("mkdocs")
    from mkdocs.config import load_config
    from mkdocs.structure.files import Files

    before = sorted(p.relative_to(DOCS) for p in DOCS.rglob("*"))
    config = load_config(str(REPO / "mkdocs.yml"))
    files = config.plugins.on_files(Files([]), config=config)
    added = {f.src_uri: f for f in files if f.generated_by is not None}
    assert sorted(added) == sorted(scope.GENERATED_PAGES)
    pages = render_pages(DOCS)
    for name, f in added.items():
        assert f.content_string == pages[name]
    assert sorted(p.relative_to(DOCS) for p in DOCS.rglob("*")) == before


def test_the_build_step_copies_exactly_the_scope(tmp_path: Path) -> None:
    """``BuildPy.run`` on a tmp build folder writes every ``CURATED_DOCS`` file, byte-identical, and nothing
    else under ``flextool/assistant/_docs/md``; a stale file of an earlier build is removed."""
    from setuptools.dist import Distribution

    import _flextool_build

    stale = tmp_path / _flextool_build.PACKAGED_DOCS / "dropped_from_scope.md"
    stale.parent.mkdir(parents=True)
    stale.write_text("old\n", encoding="utf-8")
    cmd = _flextool_build.BuildPy(Distribution({"name": "flextool-docs-copy-test", "packages": []}))
    cmd.build_lib = str(tmp_path)
    cmd.ensure_finalized()
    cmd.run()
    target = tmp_path / _flextool_build.PACKAGED_DOCS
    copied = sorted(p.relative_to(target).as_posix() for p in target.rglob("*") if p.is_file())
    assert copied == sorted(scope.CURATED_DOCS)
    for rel in copied:
        assert (target / rel).read_bytes() == (DOCS / rel).read_bytes(), rel
    outputs = {Path(p).resolve() for p in cmd.get_outputs()}
    assert {(target / rel).resolve() for rel in copied} <= outputs


def test_the_build_step_fails_on_a_scope_file_missing_from_docs(tmp_path: Path) -> None:
    import _flextool_build

    project = tmp_path / "project"
    (project / _flextool_build.SCOPE_FILE).parent.mkdir(parents=True)
    (project / _flextool_build.SCOPE_FILE).write_bytes((REPO / _flextool_build.SCOPE_FILE).read_bytes())
    (project / "docs").mkdir()
    (project / "docs" / "index.md").write_text("# FlexTool\n", encoding="utf-8")
    with pytest.raises(FileNotFoundError, match=r"reference\.md"):
        _flextool_build.copy_docs(tmp_path / "build", project_root=project)


def test_the_sdist_and_the_package_data_carry_the_scope() -> None:
    """MANIFEST.in includes the build step and exactly the scope's docs (an sdist without them cannot build a
    wheel); the package-data globs of pyproject.toml match every copied file."""
    manifest = [line.split() for line in (REPO / "MANIFEST.in").read_text(encoding="utf-8").splitlines()
                if line.strip() and not line.lstrip().startswith("#")]
    assert all(len(words) == 2 and words[0] == "include" for words in manifest), manifest
    included = [words[1] for words in manifest]
    assert sorted(included) == sorted(["_flextool_build.py"] + [f"docs/{rel}" for rel in scope.CURATED_DOCS])
    config = tomllib.loads((REPO / "pyproject.toml").read_text(encoding="utf-8"))
    assert config["tool"]["setuptools"]["cmdclass"] == {"build_py": "_flextool_build.BuildPy"}
    globs = [g for g in config["tool"]["setuptools"]["package-data"]["flextool"] if g.startswith("assistant/")]
    for rel in scope.CURATED_DOCS:
        path = f"assistant/_docs/md/{rel}"
        # setuptools globs do not cross folders: "*" must not match a "/".
        assert any(fnmatch.fnmatch(path, g) and path.count("/") == g.count("/") for g in globs), path


def _nav_pages(entries: list) -> list[str]:
    out: list[str] = []
    for entry in entries:
        value = next(iter(entry.values())) if isinstance(entry, dict) else entry
        out += _nav_pages(value) if isinstance(value, list) else [value]
    return out


def test_every_user_page_of_the_nav_is_in_the_scope() -> None:
    """The user sections of the mkdocs nav (all but the developer guide, the changelog and the archive) are
    USER_DOCS plus the generated pages, in the same order (review C2-10: the AI assistant page was missing)."""

    class _Loader(yaml.SafeLoader):
        pass

    _Loader.add_multi_constructor("tag:yaml.org,2002:python/", lambda loader, suffix, node: None)
    nav = yaml.load((REPO / "mkdocs.yml").read_text(encoding="utf-8"), Loader=_Loader)["nav"]
    skipped = {"Developer guide", "Changelog", "Archived"}
    user = _nav_pages([e for e in nav if not (isinstance(e, dict) and next(iter(e)) in skipped)])
    assert [p for p in user if p not in scope.GENERATED_PAGES] == list(scope.USER_DOCS)


def test_the_command_defaults_to_the_installed_documentation(tmp_path: Path, monkeypatch, capsys) -> None:
    """Without --docs-dir the command reads ``docs_store.markdown_root()`` (a wheel's packaged copy), not a docs/
    next to the package; a folder it cannot build from gives one line on stderr and exit 1 (review C2-2)."""
    # A backslash in the path (the separator on Windows, a plain character elsewhere): the message gives the path
    # as it is, not with the doubled backslashes of Python's quoting (CI of PR #362, Windows).
    packaged = tmp_path / "docs\\md"
    (packaged / "dev").mkdir(parents=True)          # no dev/engine.md: the build cannot read its preamble
    monkeypatch.setattr(docs_store, "markdown_root", lambda package_dir=docs_store.PACKAGE_DIR: packaged)
    assert docs_build_main(["--out", str(tmp_path / "out"), "--private-names", str(tmp_path / "absent")]) == 1
    err = capsys.readouterr().err.strip()
    assert len(err.splitlines()) == 1 and "Traceback" not in err
    assert err == f"python -m flextool.docs_build: No such file or directory: {packaged / 'dev' / 'engine.md'}"

    monkeypatch.setattr(docs_store, "markdown_root", lambda package_dir=docs_store.PACKAGE_DIR: None)
    assert docs_build_main(["--out", str(tmp_path / "out")]) == 1
    err = capsys.readouterr().err.strip()
    assert err == ("python -m flextool.docs_build: this installation has no documentation; give its folder "
                   "with --docs-dir")

    missing = tmp_path / "no_docs"
    assert docs_build_main(["--out", str(tmp_path / "out"), "--docs-dir", str(missing)]) == 1
    assert capsys.readouterr().err.strip() == (
        f"python -m flextool.docs_build: the documentation folder {missing} does not exist")
    assert not (tmp_path / "out").exists()
