![IRENA FlexTool logo](./docs/flextool_logo.png)

IRENA FlexTool is an energy and power systems model for understanding the role of variable power generation in future energy systems. It performs capacity expansion planning as well as operational planning.

This is IRENA FlexTool v3.x.x (see current version from RELEASE.md) in beta testing. Report any bugs or difficulties in the [issue tracker](https://github.com/irena-flextool/flextool/issues). 
The previous version of IRENA FlexTool can be found in https://www.irena.org/energytransition/Energy-System-Models-and-Data/IRENA-FlexTool.

User guide and documentation are under development, but the current version can be found at https://irena-flextool.github.io/flextool/.

# Getting started

## Main alternatives to use IRENA FlexTool

- [Use a browser](#connecting-to-irena-flextool-server): IRENA FlexTool can be accessed with a web browser if you have an account for an IRENA FlexTool server. However, no public servers available at the moment. The browser interface is shown [below](#browser-interface-in-brief).
- Local server: It is possible to setup a local server and then use a browser to access that server. See https://github.com/irena-flextool/flextool-web-interface
- [Install a front-end](#installing-spine-toolbox-and-irena-flextool-on-a-local-computer): Install Spine Toolbox and run IRENA FlexTool as a Spine Toolbox project. This gives you the graphical user interface of Spine Toolbox. https://github.com/Spine-project/Spine-Toolbox
<!---
- [Use Excel](#using-excel-as-an-interface): It is also possible to define all the data in Excel and execute IRENA FlexTool workflows that takes the data and scenarios from Excel and returns results in another Excel file. This functionality is still under development.
--->

### Connecting to IRENA FlexTool server

Instruction will be added later. The interface is shown [below](#browser-interface-in-brief).

<!---
### Setting up a local server

See https://github.com/irena-flextool/flextool-web-interface#installation
--->

### Installing Spine Toolbox and IRENA FlexTool on a local computer

- Install [Miniconda](https://docs.conda.io/en/latest/miniconda.html) (or Anaconda)  [Can be ignored if already installed]
- Start Anaconda prompt
- Create new Python environment [Also possible to use existing, up-to-date, Spine Toolbox environment]
  ```shell
  conda create -n flextool python=3.8
  ```
- Activate the environment
  ```shell
  conda activate flextool
  ```
- Install Git to the environment [Also possible to use existing Git installation]
  ```shell
  conda install git
  ```
- `cd` to a directory into which both FlexTool and Spine Toolbox will make their own folders
- Clone the FlexTool Git repository
  ```shell
  git clone https://github.com/irena-flextool/flextool
  ```
- Install Spine Toolbox [Can be skipped if using existing Toolbox environment]
  - Clone the Toolbox repository
    ```shell
    git clone https://github.com/Spine-project/Spine-Toolbox.git
    ```
  - cd to the freshly created folder
    ```shell
    cd Spine-Toolbox
    ```
  - Make sure Pip is up-to-date
    ```shell
    python -m pip install --upgrade pip
    ```
  - Install packages required by Toolbox
    ```shell
    python -m pip install -r requirements.txt
    ```
- Launch Spine Toolbox
  ```shell
  python -m spinetoolbox
  ```
- Open FlexTool3 project in Spine Toolbox (Choose the flextool *folder* from File > Open project dialog)

In case of problems when installing Spine Toolbox, more instructions are available at: https://github.com/Spine-project/Spine-Toolbox#installation

### Run
1. Open a conda prompt.
2. Activate the environment
  ```shell
  conda activate flextool
  ```
3. Launch Spine Toolbox
  ```shell
  python -m spinetoolbox
  ```
4. Open FlexTool3 project in Spine Toolbox (Choose the flextool *folder* from File > Open project dialog)

<!---
### Using Excel as an interface

Functionality yet not available.
--->

## Next steps

Learn about the basic data structure (important for understanding the model): [Spine database](https://irena-flextool.github.io/flextool/spine_database).

If using FlexTool with Spine Toolbox, learn how the Spine Toolbox workflow functions: [Spine Toolbox workflow](https://irena-flextool.github.io/flextool/spine_toolbox).

If using FlexTool with a web-browser, read how it works: [Browser interface](https://irena-flextool.github.io/flextool/browser_interface).

Finally, go to the documentation of the model itself: [FlexTool tutorial and documentation](https://irena-flextool.github.io/flextool/).


## Updating IRENA FlexTool

Update of IRENA FlexTool to the latest version is done as follows
- Start anaconda prompt
- `conda activate flextool` (or whatever is your conda environment name for IRENA FlexTool)
- cd to the FlexTool directory
- `git restore .` (THIS WILL DELETE YOUR LOCAL CHANGES TO THE FILES IN THE WORKFLOW. This will be improved in the future. Currently you can work around this by making your own input files (Excel or SQLite) and pointing the workflow items (Excel_input_data or Input_Data) to your own files instead of the input_data.sqlite or FlexTool_import_template.xlsx.) 
- `git pull`
