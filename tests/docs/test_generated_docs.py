"""The generated documentation pages: the formulation page and the results catalogue.

They render from the installed code and are written as UTF-8 with LF line ends whatever the locale; the
private-names check fails the build with a list file and is silent without one; results.md names a result
table in every result bullet; the mkdocs hook adds the pages without writing into docs/.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from flextool.docs_build import (
    PrivateNameError,
    default_private_names_path,
    render_pages,
    scope,
    write_pages,
)
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
