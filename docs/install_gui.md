# Install FlexTool with the FlexTool GUI

This page describes how to install IRENA FlexTool using the built-in FlexTool GUI. This is the simplest installation path — it does not require Spine Toolbox (though you can optionally add it).

For the Spine Toolbox installation path, see [Install with Spine Toolbox](install_toolbox.md).

## Requirements

- **Python 3.9–3.13** 
- **Git** (optional, but recommended for easy updates)

## Installation Steps

### 1. Install Python

Download and install Python from [python.org](https://www.python.org/downloads/). Make sure to check **"Add Python to PATH"** during installation on Windows.

> **Note:** If you already have Python in your PATH (check with `python --version` in a terminal) and it is not a supported version (3.9–3.13), you need to install a supported version. On Windows, you may need to use the full path to the new Python (e.g. `C:\Python312\python.exe`) unless you update your PATH to point to it.

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
cd flextool
```

Or download and unzip from [GitHub](https://github.com/irena-flextool/flextool/archive/refs/heads/master.zip), then navigate to the extracted directory.

### 4. Create a Virtual Environment

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

### 5. Install FlexTool

**Recommended** — install FlexTool with Spine Toolbox (includes Spine DB Editor for editing input databases):

```bash
pip install ".[toolbox]"
```

This installs FlexTool with the built-in GUI, HiGHS solver (via highspy), and Spine Toolbox with its dependencies (Spine DB Editor, Spine Items, Spine Engine, PySide6, etc.).

#### Lighter weight install (without Spine Toolbox)

If you prefer a leaner installation or have issues installing Spine Toolbox dependencies:

```bash
pip install .
```

This installs FlexTool and the GUI, but without Spine DB Editor. You can run scenarios and view results, but editing `.sqlite` input sources from the GUI requires Spine DB Editor. You can upgrade to the full install at any time by running the recommended command above.

### 6. Initialize FlexTool

Create the necessary database files and templates:

```bash
python update_flextool.py --skip-git
```

### 7. Launch FlexTool

Start the FlexTool GUI:

```bash
python -m flextool.gui
```

If you also installed Spine Toolbox, you can launch it with:

```bash
spinetoolbox
```

Then open FlexTool project from the menu: File... Open project... Choose FlexTool folder.

## Updating FlexTool

FlexTool update process will get the latest FlexTool version and then run a migration script that updates the databases to the latest version. 

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

## Troubleshooting

### "command not found: flextool-gui"

Make sure your virtual environment is activated. The `flextool-gui` command is only available when the virtual environment where FlexTool was installed is active.

### "No module named '_tkinter'" (macOS / Linux)

The FlexTool GUI uses `tkinter`, which is part of Python's standard library but depends on Tcl/Tk system libraries. Some platforms do not bundle these by default.

**macOS (Homebrew Python):** Install the Tcl/Tk binding for your Python version:

```bash
brew install python-tk@3.12
```

Replace `3.12` with your Python version (check with `python3 --version`).

**Linux (Debian / Ubuntu):** Install the system package:

```bash
sudo apt install python3-tk
```

### Import errors or missing modules

Try reinstalling:
```bash
pip install . --force-reinstall
```

### Solver issues

FlexTool includes two open-source solvers by default:

- **HiGHS** — installed automatically via `highspy`. No additional setup needed.
- **GLPK (glpsol)** — a pre-built binary is included in the `bin/` directory for Linux and Windows. For macOS, the binary `bin/glpsol_macos15_arm64` is provided for Apple Silicon Macs. If you encounter issues, check that the binary has execute permissions: `chmod +x bin/glpsol*`.

It is also possible to add commercial solvers, further instructions in the documentation.
