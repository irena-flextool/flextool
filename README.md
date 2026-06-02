[![Documentation Status](https://img.shields.io/badge/Documentation-passing-brightgreen)](https://irena-flextool.github.io/flextool/)
[![Python](https://img.shields.io/badge/python-3.11%20|%203.12-blue.svg)](https://www.python.org/downloads/release/python-3120/)

![IRENA FlexTool logo](./docs/img/flextool_logo.png)

IRENA FlexTool is an energy and power systems model for understanding the role of variable power generation in future energy systems. It performs capacity expansion planning as well as operational planning.

> [!NOTE]
> **This is the `main` branch — the current, actively developed FlexTool.** The
> previous `master` branch is deprecated and no longer updated.
>
> `main` is not the default branch yet because it has so far been **tested well
> on Linux only**; Windows and macOS are not yet fully validated. On Linux,
> using `main` now is encouraged. On other systems, expect rough edges and please
> report issues in the [tracker](https://github.com/irena-flextool/flextool/issues).
>
> If you already run FlexTool from `master`, the recommended way to try `main`
> is a **fresh parallel install** (separate directory and venv) — see
> [Installation](#installation) — so your existing setup is left untouched.

## What's new

- **A lot faster.** The model build and pre-processing now run as pure Python
  instead of the previous GLPSOL-based text pre-processing and AMPL-style model
  translation. There is no LP file written to disk and re-read between build and
  solve. (Solver time is unchanged — that is still HiGHS.)
- **A new, easy-to-use interface.** The **FlexTool GUI** is a standalone
  application (`python -m flextool.gui`) that manages projects and input
  sources, runs scenarios, and lets you **browse the results** — a result viewer
  reads outputs directly and supports single-scenario, scenario-comparison, and
  network-graph views with keyboard navigation. See the
  [FlexTool GUI guide](https://irena-flextool.github.io/flextool/flextool_gui_interface/).
- **Spine Toolbox is still a parallel interface.** Advanced users who build
  multi-tool workflows or integrate FlexTool with other models can keep using
  the [Spine Toolbox workflow](https://irena-flextool.github.io/flextool/spine_toolbox/).
- **The same model and data.** Existing input databases continue to work; the
  change is in how the matrix is generated and solved, not in the model
  formulation.

This is IRENA FlexTool v3.x.x (see current version from RELEASE.md) in beta testing. Report any bugs or difficulties in the [issue tracker](https://github.com/irena-flextool/flextool/issues). 
The previous version of IRENA FlexTool can be found in https://www.irena.org/energytransition/Energy-System-Models-and-Data/IRENA-FlexTool.

## Under the hood: pure-Python matrix generation

FlexTool reads its input data, builds the optimisation matrix in
[polars](https://pola.rs/) DataFrames via
[`polar-high`](https://github.com/nodal-tools/polar-high), and solves it with
[HiGHS](https://highs.dev/). Variables and parameters are polars frames;
multiplications are joins and aggregations are group-bys, so coefficient work
happens inside polars rather than as per-coefficient Python objects, and the
matrix is passed to HiGHS through `highspy` with no intermediate LP/MPS file.
`polar-high` is a general-purpose, domain-free modelling layer published
separately (Apache-2.0, `pip install polar-high`); its
[benchmark](https://nodal-tools.fi/polar-high/compare/benchmark/) compares it
against linopy and Pyomo on the same HiGHS solver.

## Installation

FlexTool requires **Python 3.11+**. The recommended way to start — and to try
`main` without disturbing an existing `master` setup — is a fresh clone in its
own directory and virtual environment.

**Windows (PowerShell):**

```powershell
git clone https://github.com/irena-flextool/flextool.git flextool_main
cd flextool_main
git checkout main
py -m venv .venv            # Could also be 'python' instead of 'py'
.\.venv\Scripts\Activate.ps1
# If PowerShell blocks the activate script, run once per user:
#   Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
python -m pip install --upgrade pip
pip install -e .            # resolves the polar-high pin from PyPI
python -m update-flextool --skip-git  # Creates template and example input databases
```

**Linux / macOS** — identical, with two exceptions:

- create the venv with `python3 -m venv .venv`
- activate with `. .venv/bin/activate`

(On Windows `cmd.exe` rather than PowerShell, activate with
`.venv\Scripts\activate.bat`.)

Update later with `git pull origin main` — the editable install picks up source
changes automatically; only re-run `pip install -e .` if a dependency pin in
`pyproject.toml` changed.

Then launch the FlexTool GUI:

```
python -m flextool.gui
```

## [Documentation](https://irena-flextool.github.io/flextool/) and [installation](https://irena-flextool.github.io/flextool/install_toolbox/)

> [!IMPORTANT]
> Installation, user guide and ***documentation*** can be found at: https://irena-flextool.github.io/flextool/.
