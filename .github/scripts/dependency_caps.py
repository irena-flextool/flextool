"""Dependency-cap helper for the weekly canary workflow (.github/workflows/canary.yml).

FlexTool caps some dependencies below their next untested major/minor
(``pyproject.toml`` ``[project] dependencies``).  The canary installs the
newest versions with those caps lifted, runs the test suite, and reports
which installed packages are past their caps.

Subcommands
-----------
lift
    Rewrite ``pyproject.toml`` in place so every capped dependency keeps only
    its lower bound (``<``/``<=`` dropped, ``==X``/``~=X`` turned into
    ``>=X``), and save the original requirement strings to a JSON file.
report
    Print a markdown table: each capped package, its original spec, the
    installed version and whether the installed version exceeds the cap.
issue
    Open, update or close the canary GitHub issue (``gh`` CLI, uses
    ``GH_TOKEN``).  ``--dry-run`` prints what it would do.

Only ``packaging`` is needed besides the standard library.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tomllib
from importlib import metadata
from pathlib import Path

from packaging.requirements import Requirement
from packaging.specifiers import Specifier, SpecifierSet
from packaging.version import Version

ISSUE_TITLE = "canary: dependency drift"
ISSUE_LABEL = "canary"

_UPPER_OPS = {"<", "<="}


def _is_capped(req: Requirement) -> bool:
    return any(s.operator in _UPPER_OPS | {"==", "~=", "==="} for s in req.specifier)


def _lifted_spec(req: Requirement) -> SpecifierSet:
    """The requirement's specifier with every upper bound removed."""
    kept: list[str] = []
    for spec in req.specifier:
        if spec.operator in _UPPER_OPS:
            continue
        if spec.operator in {"==", "~=", "==="}:
            version = spec.version.removesuffix(".*")
            kept.append(f">={version}")
        else:
            kept.append(str(spec))
    return SpecifierSet(",".join(kept))


def _spec_text(spec: SpecifierSet) -> str:
    """Specifier text with lower bounds first (``>=1.14.0,<1.15``)."""
    return ",".join(sorted((str(s) for s in spec), key=lambda t: (t[0] != ">", t)))


def _requirement_text(req: Requirement, spec: SpecifierSet) -> str:
    text = req.name
    if req.extras:
        text += "[" + ",".join(sorted(req.extras)) + "]"
    text += _spec_text(spec)
    if req.marker is not None:
        text += f"; {req.marker}"
    return text


def lift(pyproject: Path, caps_out: Path) -> dict[str, str]:
    """Drop upper bounds in ``pyproject``; return {name: original spec}."""
    data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    text = pyproject.read_text(encoding="utf-8")
    caps: dict[str, str] = {}
    for raw in data["project"]["dependencies"]:
        req = Requirement(raw)
        if not _is_capped(req):
            continue
        new = _requirement_text(req, _lifted_spec(req))
        pattern = re.compile(r'(["\'])' + re.escape(raw) + r"\1")
        text, count = pattern.subn(lambda m: m.group(1) + new + m.group(1), text)
        if count != 1:
            raise SystemExit(
                f"could not locate dependency {raw!r} exactly once in {pyproject}"
            )
        caps[req.name] = _spec_text(req.specifier)
        print(f"lifted: {raw!r} -> {new!r}")
    pyproject.write_text(text, encoding="utf-8")
    caps_out.write_text(json.dumps(caps, indent=2) + "\n", encoding="utf-8")
    return caps


def _installed(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def _exceeds(spec: SpecifierSet, version: str) -> bool:
    """True when ``version`` is above an upper bound of ``spec``."""
    v = Version(version)
    for s in spec:
        if s.operator not in _UPPER_OPS | {"==", "~=", "==="}:
            continue
        outside = not Specifier(str(s), prereleases=True).contains(v)
        if outside and v >= Version(s.version.removesuffix(".*")):
            return True
    return False


def drift_rows(caps: dict[str, str]) -> list[tuple[str, str, str, bool]]:
    rows = []
    for name, spec in sorted(caps.items()):
        version = _installed(name)
        over = version is not None and _exceeds(SpecifierSet(spec), version)
        rows.append((name, spec, version or "not installed", over))
    return rows


def report_markdown(rows: list[tuple[str, str, str, bool]]) -> str:
    lines = [
        "| package | FlexTool requirement | newest installed | past cap |",
        "|---|---|---|---|",
    ]
    for name, spec, version, over in rows:
        lines.append(f"| {name} | `{spec}` | {version} | {'**YES**' if over else 'no'} |")
    return "\n".join(lines)


def _failed_tests(pytest_log: Path, limit: int = 60) -> str:
    if not pytest_log.is_file():
        return "(no pytest log — the tests did not run)"
    lines = pytest_log.read_text(encoding="utf-8", errors="replace").splitlines()
    failed = [ln for ln in lines if ln.startswith(("FAILED ", "ERROR "))]
    summary = [ln for ln in lines if re.match(r"^=+ .*(passed|failed|error).* =+$", ln)]
    out = failed[:limit]
    if len(failed) > limit:
        out.append(f"... and {len(failed) - limit} more")
    out += summary[-1:]
    return "\n".join(out) if out else "(no failures listed)"


def _gh(args: list[str], dry_run: bool) -> str:
    if dry_run:
        print("DRY RUN: gh " + " ".join(args))
        return ""
    return subprocess.run(
        ["gh", *args], check=True, capture_output=True, text=True
    ).stdout.strip()


def _open_issue_number(dry_run: bool) -> str | None:
    if dry_run:
        return None
    out = _gh(
        ["issue", "list", "--state", "open", "--label", ISSUE_LABEL,
         "--json", "number,title", "--limit", "100"],
        dry_run,
    )
    for item in json.loads(out or "[]"):
        if item["title"] == ISSUE_TITLE:
            return str(item["number"])
    return None


def issue(caps: dict[str, str], install_ok: bool, tests_ok: bool,
          pytest_log: Path, run_url: str, dry_run: bool) -> None:
    rows = drift_rows(caps)
    drifted = [r[0] for r in rows if r[3]]
    healthy = install_ok and tests_ok and not drifted
    number = _open_issue_number(dry_run)

    if healthy:
        if number:
            _gh(["issue", "close", number, "--comment",
                 f"Canary run {run_url}: tests pass with the newest dependencies "
                 "and no installed package is past FlexTool's caps. Closing."],
                dry_run)
        print("canary healthy; " + (f"closed #{number}" if number else "no open issue"))
        return

    if not install_ok:
        status = "**Install failed** with the caps lifted — see the run log."
    elif not tests_ok:
        status = "**Tests failed** with the newest dependencies."
    else:
        status = "Tests pass with the newest dependencies."
    if drifted:
        status += (
            "\n\nNewer releases than FlexTool allows are available for: "
            + ", ".join(f"`{d}`" for d in drifted)
            + ". If the tests pass, the cap can be raised in `pyproject.toml`."
        )
    body = (
        "Opened and updated by the weekly dependency canary "
        "(`.github/workflows/canary.yml`), which installs the newest versions "
        "of FlexTool's capped dependencies and runs the test suite.\n\n"
        f"Latest run: {run_url}\n\n{status}\n\n"
        f"{report_markdown(rows)}\n\n"
        "<details><summary>Failing tests</summary>\n\n```\n"
        f"{_failed_tests(pytest_log) if install_ok else '(not run)'}\n```\n</details>\n"
    )
    body_file = Path("canary_issue_body.md")
    body_file.write_text(body, encoding="utf-8")
    if dry_run:
        print(body)
    if number:
        _gh(["issue", "edit", number, "--body-file", str(body_file)], dry_run)
        _gh(["issue", "comment", number, "--body", f"Still open after canary run {run_url}."],
            dry_run)
        print(f"updated #{number}")
    else:
        _gh(["label", "create", ISSUE_LABEL, "--force", "--color", "fbca04",
             "--description", "Weekly dependency canary"], dry_run)
        _gh(["issue", "create", "--title", ISSUE_TITLE, "--label", ISSUE_LABEL,
             "--body-file", str(body_file)], dry_run)
        print("opened a new canary issue")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    p_lift = sub.add_parser("lift")
    p_lift.add_argument("--pyproject", type=Path, default=Path("pyproject.toml"))
    p_lift.add_argument("--caps-out", type=Path, default=Path("caps.json"))
    p_report = sub.add_parser("report")
    p_report.add_argument("--caps", type=Path, default=Path("caps.json"))
    p_issue = sub.add_parser("issue")
    p_issue.add_argument("--caps", type=Path, default=Path("caps.json"))
    p_issue.add_argument("--install-outcome", required=True)
    p_issue.add_argument("--tests-outcome", required=True)
    p_issue.add_argument("--pytest-log", type=Path, default=Path("pytest.log"))
    p_issue.add_argument("--run-url", default="(local run)")
    p_issue.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    if args.cmd == "lift":
        lift(args.pyproject, args.caps_out)
        return 0
    caps = json.loads(args.caps.read_text(encoding="utf-8"))
    if args.cmd == "report":
        print(report_markdown(drift_rows(caps)))
        return 0
    issue(caps, args.install_outcome == "success", args.tests_outcome == "success",
          args.pytest_log, args.run_url, args.dry_run)
    return 0


if __name__ == "__main__":
    sys.exit(main())
