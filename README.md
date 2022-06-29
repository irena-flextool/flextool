# IRENA FlexTool

IRENA FlexTool is an energy and power systems model for understanding the role of variable power generation in future energy systems. It performs capacity expansion planning as well as operational planning.

## Main alternatives to use IRENA FlexTool

- The easiest way is through a webpage when someone is running a server with IRENA FlexTool. You need an account from the manager of the server.
- You can also setup a local server yourself and then use the IRENAL FlexTool browser interface to access your own server. See https://github.com/irena-flextool/flextool-web-interface
- Install Spine Toolbox and run IRENA FlexTool as a Spine Toolbox project. This gives you the graphical user interface of Spine Toolbox. https://github.com/Spine-project/Spine-Toolbox
- It is also possible to define all the data in Excel and execute IRENA FlexTool workflows that takes the data from Excel and returns results in another Excel file. This functionality is still under development.

## Connecting to IRENA FlexTool server

Instruction will be added later

## Setting up your own server

https://github.com/irena-flextool/flextool-web-interface#installation

## Installing Spine Toolbox and IRENA FlexTool on local computer

- Install Miniconda (or Anaconda)  [Can be ignored if already installed]
- Start anaconda prompt
- `conda create -n flextool python=3.8`  [Also possible to use existing, up-to-date, Spine Toolbox environment]
- `conda activate flextool`
- `conda install git`
- cd to a directory into which both FlexTool and SpineToolbox will make their own folders
- `git clone https://gitlab.vtt.fi/FlexTool/flextool3.git`
- `git clone https://github.com/Spine-project/Spine-Toolbox.git`
- `cd SpineToolbox`
- `pip install --upgrade pip`
- `pip install -r requirements.txt`
- `python -m spinetoolbox`
- Open FlexTool3 project in Spine Toolbox (Choose FlexTool folder)

In case of problems when installing Spine Toolbox, more instructions are available at: https://github.com/Spine-project/Spine-Toolbox#installation

## Using Excel as an interface

asdf
