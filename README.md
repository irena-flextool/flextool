[![PyPI version](https://img.shields.io/pypi/v/flextool.svg)](https://pypi.org/project/flextool/)
[![Documentation Status](https://img.shields.io/badge/Documentation-passing-brightgreen)](https://irena-flextool.github.io/flextool/)
[![Python](https://img.shields.io/badge/python-3.11%20|%203.12-blue.svg)](https://www.python.org/downloads/release/python-3120/)

![IRENA FlexTool logo](./docs/img/flextool_logo.png)

IRENA FlexTool is an energy and power systems model for understanding the role of variable power generation in future energy systems. It performs capacity expansion planning as well as operational planning.

> [!NOTE]
> The previous `master` branch is deprecated and no longer updated — `main` is
> now FlexTool. If you still run FlexTool from `master`, migrate with a fresh
> parallel install (separate directory and venv); see
> [Installation](#installation) so your existing setup is left untouched.

## What's new

- **A lot faster.** The model build and pre-processing now run as pure Python
  instead of the previous GLPSOL-based text pre-processing and AMPL-style model
  translation. There is no LP file written to disk and re-read between build and
  solve.
- **Automatic problem scaling.** The `polar-high` matrix layer automatically
  scales the optimisation problem before it reaches the solver. Better numerical
  conditioning often speeds up the HiGHS solve considerably — sometimes
  dramatically — so total run time usually improves well beyond the faster build
  alone. The solver is still HiGHS; the gain comes from handing it a
  better-conditioned problem.
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

This is IRENA FlexTool v4 (see the current version in CHANGELOG.md). Report any bugs or difficulties in the [issue tracker](https://github.com/irena-flextool/flextool/issues). 
The previous version of IRENA FlexTool can be found in https://www.irena.org/energytransition/Energy-System-Models-and-Data/IRENA-FlexTool.

## Under the hood: Rust-based matrix generation

FlexTool is a thin Python layer over a Rust engine. It reads its input data and
builds the optimisation matrix as [polars](https://pola.rs/) DataFrames via
[`polar-high`](https://github.com/nodal-tools/polar-high), then solves with
[HiGHS](https://highs.dev/). Variables and parameters are polars frames, so
multiplications become joins and aggregations become group-bys — the heavy
coefficient work runs inside polars' Rust core rather than as per-coefficient
Python objects, and the matrix goes straight to HiGHS through `highspy` with no
intermediate LP/MPS file. `polar-high` is a general-purpose, domain-free
modelling layer published separately (Apache-2.0, `pip install polar-high`); it
also applies the automatic problem scaling noted above, and its
[benchmark](https://nodal-tools.fi/polar-high/compare/benchmark/) compares it
against linopy and Pyomo on the same HiGHS solver.

## Installation

### Check your Python (and possibly install it)

FlexTool requires **Python 3.11+**.

Open a terminal (Windows: Command Prompt or PowerShell; macOS: Terminal; Linux: your shell)
and check which Python you have:

| Platform | Command |
|---|---|
| Windows | `py --version` (falls back to `python --version`) |
| macOS / Linux | `python3 --version` |

The output looks like `Python 3.12.4`. If the command is not found, prints
something below 3.11, or (on Windows) opens the Microsoft Store, install Python
from https://www.python.org/downloads/ — on Windows tick
**"Add python.exe to PATH"** in the installer. Then close and reopen the
terminal so the new PATH takes effect.

> On Windows, use `py` in place of `python` in the commands below if `python`
> is not recognised. If several versions are installed, `py -3.11 --version`
> (or `-3.12`, …) picks a specific one, and `py -0` lists them all.
> On macOS/Linux, `python` often means Python 2 or does not exist at all, so
> use `python3`. Inside an activated virtual environment, plain `python` always
> means that environment's Python on every platform.


### Install from PyPI (recommended)

Install FlexTool into a virtual environment to keep its packages separate from
other Python applications. If you want to install into the base Python (perhaps
because you do not otherwise use Python), just run the last command from below,
`pip install flextool`.

```
python -m venv .venv        # Windows: py -m venv .venv
                            # macOS/Linux: python3 -m venv .venv
# activate — Windows PowerShell:  .\.venv\Scripts\Activate.ps1
#            Windows CMD:         .\.venv\Scripts\activate.bat
#            Linux / macOS:       . .venv/bin/activate
python -m pip install --upgrade pip
pip install flextool
```

Your prompt should now show `(.venv)`. Verify with `python --version` — it
should report 3.11 or newer.

(If PowerShell blocks the activate script, run once per user:
`Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`.)

Launch the FlexTool GUI:

```
flextool
```

— equivalently `python -m flextool.gui`. The GUI creates and manages your
projects and input databases; to seed the bundled template and example
databases (handy for the CLI or Spine Toolbox workflows) run `flextool-update`.

### Install from source (developers / latest `main`)

To participate in FlexTool development — or to run the not-yet-released `main`
— clone the repo and install it editable:

```powershell
git clone https://github.com/irena-flextool/flextool.git
cd flextool
python -m venv .venv
.\.venv\Scripts\Activate.ps1     # Linux / macOS:  . .venv/bin/activate
python -m pip install --upgrade pip
pip install -e .
```

The editable install picks up source changes on `git pull`; re-run
`pip install -e .` only when a dependency pin in `pyproject.toml` changes.

### Update FlexTool:

```
# If using virtual environment (venv), then activate venv first:
#   Windows PowerShell:  .\.venv\Scripts\Activate.ps1
#   Windows CMD:         .\.venv\Scripts\activate.bat
#   Linux / macOS:       . .venv/bin/activate
pip install --upgrade flextool
```

## [Documentation](https://irena-flextool.github.io/flextool/) and [installation](https://irena-flextool.github.io/flextool/install_toolbox/)

> [!IMPORTANT]
> Installation, user guide and ***documentation*** can be found at: https://irena-flextool.github.io/flextool/.
