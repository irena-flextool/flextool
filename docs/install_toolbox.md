## Installing Spine Toolbox and IRENA FlexTool on a local computer

Follow video tutorial for installation here: [Link to YouTube](https://youtu.be/N3qB0rzxPYw). This is bit old, but should still work. The instructions below are somewhat simpler. These instructions are for Windows, see Linux and Mac at the bottom.

- Install Python (3.9 - 3.12 supported as of 15.11.2024) from the Python website or from your favourite app manager (e.g. Windows store). Or use existing installation if available.

> [! NOTE]
> If you have existing Python in the PATH (try `python --version` in a terminal) and it is not a supported version, then you need to install another Python and call it with a full path (unless you replace the old Python with the new in the PATH).

- Start a terminal (e.g. type `cmd` in Windows search)
- 'cd' to a directory where you want to have FlexTool's virtual environment directory (Python packages FlexTool needs). If you are not familiar with terminals, just `cd full_path_to_the_directory`, where that full path can be copied from a file manager.
- Make a virtual environment for FlexTool by

```
python -m venv flextool-venv
```

- Activate the newly created environment (and remember to activate it always when starting FlexTool)

```
flextool-venv\Scripts\activate
```

- Get Flextool:
    - Option 1, easy but harder to keep up-to-date:
        - Download a [zip file](https://github.com/irena-flextool/flextool/archive/refs/heads/master.zip).
        - Unzip to a location of your choice.
    - Option 2, install using git (and use git later when keeping FlexTool up-to-date):
        - Download and install from [https://git-scm.com/downloads](https://git-scm.com/downloads) or using app manager (not available in Windows Store)
        - 'cd' to a directory where you want Flextool's directory to be located (this could be the same location as for flextool-venv).
        - Lastly

```
git clone https://github.com/irena-flextool/flextool.git
```

- 'cd' to the FlexTool root (main) directory (unless there already)
- Install requirements by

```
python -m pip install -r requirements.txt --timeout=10000
```

- Please note, the `timeout` argument is necessary only if you are on a slow internet connection
- Create basic files that FlexTool needs (`skip-git` argument is needed if you did not install git)

```
python update_flextool-py --skip-git
```

- Start Spine Toolbox (can take a small while)

```
spinetoolbox
```

## Starting IRENA FlexTool

- Start a terminal (e.g. type 'cmd' in Windows search)
- 'cd' to directory where the virtual environment directory for flextool is (assumed below to be 'flextool-venv', but modify as needed)
- Activate the environment

```
flextool-venv\Scripts\activate
```

- Launch Spine Toolbox

```
python -m spinetoolbox
```

- Open FlexTool3 project in Spine Toolbox (Choose the flextool *directory* from File > Open project dialog by navigating to the directory where FlexTool itself is installed)

## Updating IRENA FlexTool

### Updates when using venv (virtual environment, as instructed above since 15.11.2024)

- Start by updating Spine Toolbox
    - Activate environment as [above](#starting-irena-flextool)
    - Then upgrade Spine Toolbox

```
python -m pip install --upgrade spinetoolbox
```


- Then, to update FlexTool
- 'cd' to flextool directory

```
python update_flextool.py
```

- This will not work if you don't have git installed. In that case, you need to download the latest [zip file](https://github.com/irena-flextool/flextool/archive/refs/heads/master.zip), unzip it to the FlexTool directory (overwriting existing files) and then
  
```
python update_flextool.py --skip-git
```

- If this fails, see [below](#upgrade-troubleshooting)
- This update will pull the new version of the tool as well as migrating the input databases to the new version without destroying the data. Making a backup copy of the input data is still a good practice. The input_data_template.sqlite should not be used directly but by making a copy of it. 
    - The updated databases are: 
        - The database chosen as the input data in the tool!! But no other databases you might have - those can be updated separately, see below.
        - init.sqlite
        - input_data_template.sqlite
        - time_settings_only.sqlite
        - how to example databases


### Updates when using anaconda/miniconda (this was the installation instruction until 15.11.2024):

- Start anaconda/miniconda prompt
- `conda activate flextool` (or whatever is your conda environment name for IRENA FlexTool)

If using Spine Toolbox, start by updating Spine Toolbox:

- cd to the directory of Spine-Toolbox, where you cloned it. 
For example `cd C:\Users\YourUser\Documents\Spine-Toolbox`
- `git pull`
- `python -m pip install -U -r requirements.txt`
- cd back to FlexTool directory, where you cloned it. 
For example `cd C:\Users\YourUser\Documents\flextool`

Update IRENA FlexTool:

- cd to the FlexTool directory
- `python update_flextool.py`
- If this fails, see [below](#upgrade-troubleshooting)
- This update will pull the new version of the tool as well as migrating the input databases to the new version without destroying the data. Making a backup copy of the input data is still a good practice. The input_data_template.sqlite should not be used directly but by making a copy of it. 
    - The updated databases are: 
        - The database chosen as the input data in the tool!! But no other databases you might have - those can be updated separately, see below.
        - init.sqlite
        - input_data_template.sqlite
        - time_settings_only.sqlite
        - how to example databases

### Old version (release 3.1.3 or earlier):
Update of IRENA FlexTool to the latest version is done as follows:

- Start anaconda prompt
- `conda activate flextool` (or whatever is your conda environment name for IRENA FlexTool)
- cd to the FlexTool directory
- `git restore .` (THIS WILL DELETE YOUR LOCAL CHANGES TO THE FILES IN THE WORKFLOW. This will be improved in the future. Currently you can work around this by making your own input files (Excel or SQLite) and pointing the workflow items (Excel_input_data or Input_Data) to your own files instead of the input_data.sqlite or FlexTool_import_template.xlsx.) 
- `git pull`
Then do the update_flextool discribed above to migrate the databases:
- `python update_flextool.py`

### Upgrade troubleshooting

- If the git complains about merge conflicts, it is probably due to you modifying the template files. Use `git restore .`  and `git pull `. This will restore ALL the files downloaded from the repository to their original states. Then repeat `python update_flextool.py`
- The Results.sqlite will only get additive changes, as we do not want to destroy your data. This causes old parameter definitions to linger in the database. You can remove them by replacing the database with a copy of the Results_template.sqlite that is kept up to date.

- One can also migrate other input databases to the new version by calling:
    - `python migrate_database.py path_to_database/database_name.sqlite` or
    - `python migrate_database.py database_name.sqlite` if in the main flextool directory

## Installing for Linux or Mac

FlexTool repository contains executables for highs and glpsol also for x64 Linux and it should work. Install Toolbox first, maybe by creating a new venv for the Toolbox and FlexTool - you can follow the spirit of the instructions above even though commands will be somewhat different. If your Linux runs on another architecture or you have a Mac, then we haven't tested those. 

Other Linux architecture's could work, but get correct binaries for highs from https://github.com/JuliaBinaryWrappers/HiGHSstatic_jll.jl/releases and compile glpsol from https://github.com/mingodad/GLPK/ (FlexTool uses some of the improvements made to GLPK in this fork). Then replace the binaries in the FlexTool root directory. This will cause conflicts when next time updating FlexTool using the instructions above. You need to resolve those conflicts, maybe by using `git restore .` before the update and re-copying the binaries after the update. Mac version might require small changes to the code (especially flextoolrunner.py) in addition to the correct binaries. Spine Toolbox works also on Mac, but apparently has some graphical glitches (21st Sep. 2024 - hopefully will be fixed at some point).
