## Database editor in brief

Spine Toolbox database editor can be used to modify data and to build scenarios. 
The figure below shows an example where parameter data from two `alternatives` 
have been selected for display (in the data table). The object tree on the left 
selects two `nodes` ('coal_market' and 'west') as well as one `unit` ('coal_plant'). 
These are visualized in the graph on top. The mouse pointer is showing a relationship 
entity that connects the 'coal_plant' and its output `node` 'west'. The relationship 
entity is defined in a relationship tree, which is not visible here.

The `scenario` tree (on the right, below the `alternative` tree) shows that 
the 'coal' `scenario` is formed by taking all data from the 'init' `alternative` 
and then all data from the 'coal' `alternative`. If there would be same parameter 
defined for both `scenarios`, then the latter `alternative` would overwrite 
the first `alternative`.

![Database editor](./database_editor.png)

More on Spine Database editor in https://spine-toolbox.readthedocs.io/en/latest/spine_db_editor/index.html.
