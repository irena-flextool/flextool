![IRENA FlexTool logo](./docs/flextool_logo.png)

IRENA FlexTool is an energy and power systems model for understanding the role of variable power generation in future energy systems. It performs capacity expansion planning as well as operational planning.

This is IRENA FlexTool v3.0 in beta testing. Report any bugs or difficulties in the [issue tracker](https://github.com/irena-flextool/flextool/issues). Previous version of IRENA FlexTool can be found in https://www.irena.org/energytransition/Energy-System-Models-and-Data/IRENA-FlexTool.

User guide and documentation are under development, but the current version can be found [here](https://irena-flextool.github.io/flextool/).

# Getting started

## Main alternatives to use IRENA FlexTool

- [Use a browser](#connecting-to-irena-flextool-server): IRENA FlexTool can be accessed with a web browser if you have an account for an IRENA FlexTool server. No public servers available at the moment.
- Local server: It is possible to setup a local server and then use a browser to access that server. See https://github.com/irena-flextool/flextool-web-interface
- [Install a front-end](#installing-spine-toolbox-and-irena-flextool-on-a-local-computer): Install Spine Toolbox and run IRENA FlexTool as a Spine Toolbox project. This gives you the graphical user interface of Spine Toolbox. https://github.com/Spine-project/Spine-Toolbox
<!---
- [Use Excel](#using-excel-as-an-interface): It is also possible to define all the data in Excel and execute IRENA FlexTool workflows that takes the data and scenarios from Excel and returns results in another Excel file. This functionality is still under development.
--->

### Connecting to IRENA FlexTool server

Instruction will be added later

<!---
### Setting up a local server

See https://github.com/irena-flextool/flextool-web-interface#installation
--->

### Installing Spine Toolbox and IRENA FlexTool on a local computer

- Install [Miniconda](https://docs.conda.io/en/latest/miniconda.html) (or Anaconda)  [Can be ignored if already installed]
- Start anaconda prompt
- `conda create -n flextool python=3.8`  [Also possible to use existing, up-to-date, Spine Toolbox environment]
- `conda activate flextool`
- `conda install git`
- cd to a directory into which both FlexTool and SpineToolbox will make their own folders
- `git clone https://github.com/irena-flextool/flextool`
- `git clone https://github.com/Spine-project/Spine-Toolbox.git`
- `cd SpineToolbox`
- `pip install --upgrade pip`
- `pip install -r requirements.txt`
- `python -m spinetoolbox`
- Open FlexTool3 project in Spine Toolbox (Choose FlexTool folder)

In case of problems when installing Spine Toolbox, more instructions are available at: https://github.com/Spine-project/Spine-Toolbox#installation

<!---
### Using Excel as an interface

Functionality yet not available.
--->

## IRENA FlexTool workflow explained

IRENA FlexTool workflow is a Spine Toolbox workflow that can be modified by the user. The workflow provided in the repository is a template project that can be **either** copied for local changes **or** the workflow data input data files can be switched to local files. It is also possible to work directly with the template, but then one needs to be careful when updating IRENA FlexTool (the input data file contents need to be copied to safety before updating). 

![IRENA FlexTool workflow](./docs/flextool_workflow.png)

The panel on the right shows the different `scenarios` that are available in the database. The user can choose which scenarios will be processed by the workflow (until item `Results`, which combines the results into one database). Spine Toolbox can execute scenarios in parallel (as long as using 'work directories' defined in `FlexTool` item.

`Input_data` workflow item points to a sqlite file that needs to have IRENA FlexTool data format (that uses Spine Toolbox database definition). The template file has the right format and contains empty object classes corresponding to FlexTool data structure as well as parameters available in each object class. Double clicking the Input_data workflow item will open the database editor. Just selecting the Input_data workflow item allows one to change the file (make a copy of the existing Input_data.sqlite and point to the copy).

`Init` workflow item points to a sqlite file with predefined data that showcases IRENA FlexTool functionality. Some of the scenarios from there are used in the user guide. `Initialize` copies the contents of the Init database to the Input_data database.

`Export_to_csv` workflow item is a Spine Toolbox exporter that has been set to write csv files that IRENA FlexTool model code will read.

`FlexTool` workflow item contains a Python script that calls FlexTool model code for each solve and passes data between these solves. FlexTool model is written in MathProg and it calls HiGHS solver by default to solve the model. The outputs are csv files.

`Import_results` is a Spine Toolbox importer that takes the output csv files and write them in the Results database.

`Excel_input_data` and `Import_from_Excel` allow users to use Excel as an interface for the input data. They are optional parts of the workflow.

`To_Excel` worfklow item will export most scenario results to a simple Excel file. One way to utilize is this by creating another Excel file that draws figures from the result Excel file that is then updated by the workflow.

The browser interface of FlexTool also runs part of this same workflow (`Export_to_csv` --> `FlexTool` --> `Import_results`). The server takes a copy of the workflow (inside the user_projects) folder and uses Spine Toolbox to execute the scenarios.

## Database editor in brief

Spine Toolbox database editor can be used to modify data and build scenarios. The figure below shows an example where parameter data from two `alternatives` have been selected for display (in the data table). The object tree on the left selects two `nodes` ('coal_market' and 'west') as well as one `unit` ('coal_plant'). These are visualized in the graph on top. The mouse pointer is showing a relationship entity that connects the 'coal_plant' and its output `node` 'west'. The relationship entity is defined in a relationship tree, which is not visible here.

The `scenario` tree (on the right, below the `alternative` tree) shows that the 'coal' `scenario` is formed by taking all data from the 'init' `alternative` and then all data from the 'coal' `alternative`. If there would be same parameter defined for both `scenarios`, then the latter `alternative` would overwrite the first `alternative`.

![Database editor](./docs/database_editor.png)

## Updating IRENA FlexTool

Update of IRENA FlexTool to the latest version is done as follows
- Start anaconda prompt
- `conda activate flextool` (or whatever is your conda environment name for IRENA FlexTool)
- cd to the FlexTool directory
- `git restore .` (THIS WILL DELETE YOUR LOCAL CHANGES TO THE FILES IN THE WORKFLOW. This will be improved in the future. Currently you can work around this by making your own input files (Excel or SQLite) and pointing the workflow items (Excel_input_data or Input_Data) to your own files instead of the input_data.sqlite or FlexTool_import_template.xlsx. Whenever you update IRENA FlexTool you need to update the file links again.) 
- `git pull`
