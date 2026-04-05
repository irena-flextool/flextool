# Install FlexTool with the FlexTool GUI

This page describes how to install IRENA FlexTool using the built-in FlexTool GUI. This is the simplest installation path — it does not require Spine Toolbox (though you can optionally add it).

For the Spine Toolbox installation path, see [Install with Spine Toolbox](install_toolbox.md).

## Requirements

- **Python 3.9–3.13** 
- **Git** (optional, but recommended for easy updates)

## Installation Steps

### 1. Install Python

Download and install Python from [python.org](https://www.python.org/downloads/). Make sure to check **"Add Python to PATH"** during installation on Windows.

### 2. Create a Virtual Environment

Open a terminal and navigate to where you want to install FlexTool:

```bash
python -m venv flextool-venv
```

Activate the virtual environment:

Linux / macOS:
```bash
source flextool-venv/bin/activate
```

Windows:
```
flextool-venv\Scripts\activate
```

### 3. Get FlexTool

Either clone with Git (recommended):
```bash
git clone https://github.com/irena-flextool/flextool.git
cd flextool
```

Or download and unzip from [GitHub](https://github.com/irena-flextool/flextool/archive/refs/heads/master.zip), then navigate to the extracted directory.

### 4. Install FlexTool

**Recommended** — install FlexTool with Spine Toolbox (includes Spine DB Editor for editing input databases):

```bash
pip install -e ".[toolbox]"
```

This installs FlexTool with the built-in GUI, HiGHS solver (via highspy), and Spine Toolbox with its dependencies (Spine DB Editor, Spine Items, Spine Engine, PySide6, etc.).

#### Lightweight install (without Spine Toolbox)

If you prefer a leaner installation or have issues installing Spine Toolbox dependencies:

```bash
pip install -e .
```

This installs FlexTool and the GUI, but without Spine DB Editor. You can run scenarios and view results, but editing `.sqlite` input sources from the GUI requires Spine DB Editor. You can upgrade to the full install at any time by running the recommended command above.

### 5. Initialize FlexTool

Create the necessary database files and templates:

```bash
python update_flextool.py --skip-git
```

### 6. Launch FlexTool

Start the FlexTool GUI:

```bash
python -m flextool.gui
```

If you also installed Spine Toolbox, you can launch it with:

```bash
spinetoolbox
```

## Updating FlexTool

FlexTool update process will get the latest FlexTool version and then run a migration script that updates the databases to the latest version. 

If you installed with Git:

```bash
cd flextool
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
pip install -e . --force-reinstall
```

### Solver issues

FlexTool includes two open-source solvers by default:

- **HiGHS** — installed automatically via `highspy`. No additional setup needed.
- **GLPK (glpsol)** — a pre-built binary is included in the `bin/` directory for Linux and Windows. For macOS, the binary `bin/glpsol_macos15_arm64` is provided for Apple Silicon Macs. If you encounter issues, check that the binary has execute permissions: `chmod +x bin/glpsol*`.

It is also possible to add commercial solvers, further instructions in the documentation.
