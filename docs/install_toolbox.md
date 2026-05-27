# Install FlexTool with Spine Toolbox

This page describes the **Spine Toolbox** installation path for IRENA FlexTool. Spine Toolbox is the heavier, multi-tool integration path: it wraps FlexTool as one node in a directed-acyclic-graph (DAG) workflow alongside other tools and data stores.

Most users do not need this path. If your work is centred on FlexTool itself — building scenarios, running them, inspecting results — follow [Install with FlexTool GUI](install_gui.md) instead. The Spine Toolbox path is intended for users who already use Spine Toolbox or who combine FlexTool with other Spine-based models. See [Choosing an interface](interface_overview.md) for the side-by-side comparison.

!!! note "v4 is in alpha"
    FlexTool v4 is currently in alpha (`4.0.0a1`, released 2026-05-26). The release is published as a PEP 440 pre-release on PyPI, so pip needs the `--pre` flag to resolve it — e.g. `pip install --pre "flextool[toolbox]"`. Parity with the 3.47.x line is expected; reports against scenarios that worked on 3.47.0 but misbehave on 4.0.0a1 are especially welcome.

## Requirements

- **Python ≥ 3.11** (see `requires-python` in `pyproject.toml` for up-to-date requirement)
- **Git** (optional, but recommended for easy updates)

## Installation Steps

### 1. Install Python

Download and install Python from [python.org](https://www.python.org/downloads/). Make sure to check **"Add Python to PATH"** during installation on Windows.

> **Note:** If you already have Python in your PATH (check with `python --version` in a terminal) and it is older than 3.9, you need to install a supported version. On Windows, you may need to use the full path to the new Python (e.g. `C:\Python312\python.exe`) unless you update your PATH to point to it.

### 2. Install Git (optional)

Git makes it easy to update FlexTool in the future. The alternative is to download zip files manually.

- Download and install from [https://git-scm.com/downloads](https://git-scm.com/downloads) or use your system's package manager.
- **Windows tips:** During git installation, the defaults are fine for most questions. Exceptions:
  - Text editor: switch from VIM to Nano (scroll upward) — unless you know and like VIM.
  - Use Windows' default console window (unless you know what you are doing).

### 3. Get FlexTool

Either clone with Git (recommended):
```bash
git clone https://github.com/irena-flextool/flextool.git
```

Or download and unzip from [GitHub](https://github.com/irena-flextool/flextool/archive/refs/heads/main.zip).

### 4. Enter the FlexTool directory

All the remaining steps must be run from inside the FlexTool directory:

```bash
cd flextool
```

(Substitute the extracted folder name if you used the zip download.)

### 5. Create a Virtual Environment

**Linux (Debian / Ubuntu / Mint):** The system Python may not include `venv` with `pip` by default. Install it first:

```bash
sudo apt install python3-venv
```

Create a virtual environment inside the FlexTool directory:

```bash
python -m venv .venv
```

Activate the virtual environment:

Linux / macOS:
```bash
source .venv/bin/activate
```

Windows:
```
.venv\Scripts\activate
```

### 6. Install FlexTool with Spine Toolbox

Install FlexTool and the `toolbox` extra. This pulls in Spine Toolbox alongside FlexTool's core dependencies (declared in `pyproject.toml` — `requirements.txt` is no longer used):

```bash
pip install ".[toolbox]"
```

The extra brings in Spine Toolbox itself, Spine DB Editor, Spine Items, Spine Engine, PySide6, and the rest of Spine Toolbox's runtime stack. On a slow link you may want `--timeout=10000`.

If you would rather pull from PyPI than work from a clone, the same extra works on the pre-release wheel:

```bash
pip install --pre "flextool[toolbox]"
```

### 7. Initialize FlexTool

Create the necessary database files and templates (skip the `--skip-git` flag if you cloned with git and want `update_flextool.py` to fetch upstream changes too):

```bash
python update_flextool.py --skip-git
```

### 8. Verify the install

Confirm the FlexTool CLI is wired up:

```bash
flextool-run --help
```

This should print the FlexTool CLI usage. FlexTool runs independently of Spine Toolbox, so this check passes even before Toolbox is launched — useful for isolating environment issues from Toolbox UI issues.

### 9. Launch Spine Toolbox

```bash
spinetoolbox
```

Then open the FlexTool project from the menu: File... Open project... and choose the **FlexTool directory** (not a file inside it).

## Updating FlexTool

The update flow mirrors [Install with FlexTool GUI](install_gui.md#updating-flextool): pull the latest FlexTool, then migrate the input databases.

If you installed with Git:

```bash
cd flextool
source .venv/bin/activate   # Linux / macOS
.venv\Scripts\activate      # Windows
python update_flextool.py
```

If you installed from a zip file, download the latest zip, extract it over the existing directory, and run:

```bash
python update_flextool.py --skip-git
```

`update_flextool.py` migrates the input databases to the current schema version without destroying data (a backup is still good practice). The databases that get migrated are:

- the database currently selected as input data in the project,
- `init.sqlite`,
- `input_data_template.sqlite`,
- `time_settings_only.sqlite`,
- the how-to example databases.

To migrate any other database separately:

```bash
python migrate_database.py path/to/database.sqlite
```

Spine Toolbox itself updates separately. With the FlexTool venv active:

```bash
pip install --upgrade spinetoolbox
```

### Update troubleshooting

- If git reports merge conflicts, it is almost always because the template files were modified locally. Restore the tree with `git restore .` and re-run `git pull`, then `python update_flextool.py`.
- `Results.sqlite` only ever receives additive changes (to protect data), which leaves old parameter definitions lingering. To clear them, replace the database with a copy of the up-to-date `Results_template.sqlite`.

## Troubleshooting

### "command not found: spinetoolbox" / "flextool-run"

Make sure your virtual environment is activated. Spine Toolbox and the `flextool-*` console scripts are only on PATH when the venv where FlexTool was installed is active.

### Linux / macOS notes

FlexTool calls HiGHS through the `highspy` Python bindings, which install automatically as a dependency, so no separate solver binary is needed on x64 Linux or Intel macOS. On other Linux architectures or Apple Silicon, a prebuilt `highspy` wheel may not be available; in that case you will need to build HiGHS / highspy from source for your platform.

Spine Toolbox runs on Linux and macOS as well, with occasional minor graphical glitches on macOS.

### Import errors or missing modules

Try reinstalling:

```bash
pip install ".[toolbox]" --force-reinstall
```

### Solver issues

FlexTool uses **HiGHS** by default — installed automatically via the `highspy` Python bindings, no additional setup needed.

It is also possible to wire in commercial solvers (e.g. CPLEX) for licensed users; see the documentation for details.
