### Installing Spine Toolbox and IRENA FlexTool on a local computer

Follow video tutorial for installation here: [Link to YouTube](https://youtu.be/N3qB0rzxPYw).

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

Install Spine Toolbox [Can be skipped if using existing Toolbox environment]

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

5. In the flextool folder, make a copy the Results_template.sqlite and rename it Results.sqlite (Results.sqlite is not part of the repository to avoid accidental overwrites of your results in the future) 

## Updating IRENA FlexTool

### New update version (release 3.1.4 or later):
Update of IRENA FlexTool to the latest version is done as follows:
- Start anaconda prompt
- `conda activate flextool` (or whatever is your conda environment name for IRENA FlexTool)
- cd to the FlexTool directory
- `python update_flextool.py`
- This will pull the new version of the tool as well as migrating the input databases to the new version without destroying the data. Making a backup copy of the input data is still a good practice.
    - The updated databases are: 
        - The database chosen as the input data !!!
        - init.sqlite
        - input_data_template.sqlite
        - how to example databases

- One can also migrate other input databases to the new version by calling:
    - `python migrate_database.py *path_to_database*`

### Old version (release 3.1.3 or earlier):
Update of IRENA FlexTool to the latest version is done as follows:

- Start anaconda prompt
- `conda activate flextool` (or whatever is your conda environment name for IRENA FlexTool)
- cd to the FlexTool directory
- `git restore .` (THIS WILL DELETE YOUR LOCAL CHANGES TO THE FILES IN THE WORKFLOW. This will be improved in the future. Currently you can work around this by making your own input files (Excel or SQLite) and pointing the workflow items (Excel_input_data or Input_Data) to your own files instead of the input_data.sqlite or FlexTool_import_template.xlsx.) 
- `git pull`
