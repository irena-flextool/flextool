[![Documentation Status](https://img.shields.io/badge/Documentation-passing-brightgreen)](https://irena-flextool.github.io/flextool/)
[![Python](https://img.shields.io/badge/python-3.9%20|%203.10%20|%203.11%20|%203.12-blue.svg)](https://www.python.org/downloads/release/python-3120/)

![IRENA FlexTool logo](./docs/flextool_logo.png)

> [!IMPORTANT]
> **This `master` branch is being deprecated. Active development has moved to `main`.**
>
> `main` is a lot faster, ships an easy-to-use new interface (the FlexTool GUI)
> for building models and browsing results, and runs on a pure-Python core that
> generates the optimisation matrix with
> [polar-high](https://github.com/nodal-tools/polar-high) and solves it with
> HiGHS. `master` will stop receiving updates.
>
> `main` is not the default branch yet because it has so far been **tested well
> on Linux only** — Windows and macOS are not yet fully validated. If you are on
> Linux, switching now is encouraged; on other systems, expect rough edges and
> please report them.
>
> **Try `main` as a fresh parallel install** (separate directory and venv —
> leaves this `master` setup untouched). Requires Python 3.11+. On Windows
> (PowerShell):
> ```powershell
> git clone https://github.com/irena-flextool/flextool.git flextool_main
> cd flextool_main
> git checkout main
> py -m venv .venv
> .\.venv\Scripts\Activate.ps1
> python -m pip install --upgrade pip
> pip install -e .                       # pulls polar-high from PyPI
> ```
> On Linux/macOS the only differences are `python3 -m venv .venv` and
> `. .venv/bin/activate`. Update later with `git pull origin main` (the editable
> install picks up source changes automatically). Then run `python -m flextool.gui`.
> Full instructions and a smoke test:
> [`main` README](https://github.com/irena-flextool/flextool/blob/main/README.md).

IRENA FlexTool is an energy and power systems model for understanding the role of variable power generation in future energy systems. It performs capacity expansion planning as well as operational planning.

This is IRENA FlexTool v3.x.x (see current version from RELEASE.md) in beta testing. Report any bugs or difficulties in the [issue tracker](https://github.com/irena-flextool/flextool/issues). 
The previous version of IRENA FlexTool can be found in https://www.irena.org/energytransition/Energy-System-Models-and-Data/IRENA-FlexTool.

## [Documentation](https://irena-flextool.github.io/flextool/) and [installation](https://irena-flextool.github.io/flextool/install_toolbox/)

> [!IMPORTANT]
> Installation, user guide and ***documentation*** can be found at: https://irena-flextool.github.io/flextool/.

The documenation has a [section on installation](https://irena-flextool.github.io/flextool/install_toolbox/) where one installs Python and git and then uses virtual environment (venv) to create and install FlexTool. This allows to update both FlexTool and its graphical interface (Spine Toolbox) with new versions with relative ease.

There is also an experimental [zip file](https://github.com/irena-flextool/flextool/releases/download/v3.9.0/Flextool-Toolbox.zip) containing both Flextool and Spine Toolbox (Windows 10 and 11 only). This requires only unzipping to a user-controlled folder (e.g. 'users/user_name/', do not use 'Program Files'). However, it does not allow direct updates (you need to download a new zip file to update).

> [!NOTE]  
> Spine Toolbox has received a major upgrade 29th of April 2024. Next time you update FlexTool, [update Spine Toolbox first](https://github.com/spine-tools/Spine-Toolbox#installation). Follow the upgrade instructions of your Toolbox installation method.
