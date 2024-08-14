## Installing Spine Toolbox and IRENA FlexTool on a local computer

Follow video tutorial for installation here: [Link to YouTube](https://youtu.be/N3qB0rzxPYw).
Currently python versions 3.9 or higher are supported.

- Install [Miniconda](https://docs.conda.io/en/latest/miniconda.html) (or Anaconda)  [Can be ignored if already installed]
- Start Anaconda prompt
- Create new Python environment [Also possible to use existing, up-to-date, Spine Toolbox environment]

```shell
conda create -n flextool python=3.11
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

Install Spine Toolbox [Can be skipped if using existing Toolbox installation using any of the methods presented in https://github.com/Spine-project/Spine-Toolbox#installation]

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
In case of problems when installing Spine Toolbox, more instructions are available at: https://github.com/Spine-project/Spine-Toolbox#installation

Generate databases from templates:

- `cd` back to the flextool repository folder. Most likely:

```shell
cd ../flextool
```
- Run update_flextool.py script. This will generate the Input_data.sqlite and Results.sqlite from their templates. These are not directly in the repository to avoid future overwrites. Update_flextool.py can also be used later when one wants to update the tool.  

```shell
python update_flextool.py
```

## Starting IRENA FlexTool in Spine Toolbox

1. Open a conda prompt.
2. Activate the environment

    ```
    conda activate flextool
    ```

3. Launch Spine Toolbox

    ```
    python -m spinetoolbox
    ```

4. Open FlexTool3 project in Spine Toolbox (Choose the flextool *folder* from File > Open project dialog)

## Updating IRENA FlexTool

### Updates for version 3.1.4 or later:

- Start anaconda/miniconda prompt
- `conda activate flextool` (or whatever is your conda environment name for IRENA FlexTool)

If using Spine Toolbox, start by updating Spine Toolbox:

- cd to the repository folder Spine-Toolbox, where you cloned it. 
For example `cd C:\Users\YourUser\Documents\Spine-Toolbox`
- `git pull`
- `python -m pip install -U -r requirements.txt`
- cd back to FlexTool directory, where you cloned it. 
For example `cd C:\Users\YourUser\Documents\flextool`

Update IRENA FlexTool:

- cd to the FlexTool directory
- `python update_flextool.py`
- This will pull the new version of the tool as well as migrating the input databases to the new version without destroying the data. Making a backup copy of the input data is still a good practice. The input_data_template.sqlite should not be used directly but by making a copy of it. 
    - The updated databases are: 
        - The database chosen as the input data in the tool!!!
        - init.sqlite
        - input_data_template.sqlite
        - time_settings_only.sqlite
        - how to example databases
- If the git complains about merge conflicts, it is probably due to you modifying the template files. Use `git restore .`  and `git pull `. This will restore ALL the files downloaded from the repository to their original states. Then repeat `python update_flextool.py`
- The Results.sqlite will only get additive changes, as we do not want to destroy your data. This causes old parameter definitions to linger in the database. You can remove them by replacing the database with a copy of the Results_template.sqlite that is kept up to date.

- One can also migrate other input databases to the new version by calling:
    - `python migrate_database.py *absolute_path_to_database*` or
    - `python migrate_database.py database_name.sqlite` if in the main flextool folder

### Old version (release 3.1.3 or earlier):
Update of IRENA FlexTool to the latest version is done as follows:

- Start anaconda prompt
- `conda activate flextool` (or whatever is your conda environment name for IRENA FlexTool)
- cd to the FlexTool directory
- `git restore .` (THIS WILL DELETE YOUR LOCAL CHANGES TO THE FILES IN THE WORKFLOW. This will be improved in the future. Currently you can work around this by making your own input files (Excel or SQLite) and pointing the workflow items (Excel_input_data or Input_Data) to your own files instead of the input_data.sqlite or FlexTool_import_template.xlsx.) 
- `git pull`
Then do the update_flextool discribed above to migrate the databases:
- `python update_flextool.py`
