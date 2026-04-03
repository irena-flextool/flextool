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

Install FlexTool and all its dependencies:

```bash
pip install -e .
```

This installs FlexTool with the built-in GUI, HiGHS solver (via highspy), and all required libraries.

#### Optional: Add Spine Toolbox

If you also want to use Spine Toolbox as an alternative interface (useful especially if you want to build your own workflows for data processing and to link more tools):

```bash
pip install -e ".[toolbox]"
```

This installs everything above plus Spine Toolbox and its dependencies (Spine Items, Spine Engine, PySide6, etc.).

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

### "No module named 'tkinter'" (Linux)

On most Linux distributions, `tkinter` is included with Python. However, some distributions (notably Debian and Ubuntu) package it separately. If you see this error, install it with:

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
