![IRENA FlexTool logo](./irena_flextool_logo.png)

IRENA FlexTool is an energy systems optimisation model developed for power and energy systems with high shares of wind and solar power. It can be used to find cost-effective sources of flexibility across the energy system to mitigate the increasing variability arising from the power systems. It can perform multi-year capacity expansion as well as unit commitment and economic dispatch in a user-defined sequence of solves. The aim has been to make it fast to learn and easy to use while including lot of functionality especially in the time scales relevant for investment planning and operational scheduling of energy systems.

# Documentation structure

- Install IRENA FlexTool following the [installation instructions](install_toolbox.md). Follow the video tutorial for installation with Spine Toolbox here: [Link to YouTube](https://youtu.be/N3qB0rzxPYw)
- The easiest way to use FlexTool is through the [FlexTool GUI](flextool_gui_interface.md) -- start it with `python -m flextool.gui`.
- For advanced data management, use the [Spine Toolbox workflow](spine_toolbox.md).
- For scripting and automation, use the [terminal workflow](terminal_workflow.md).
- See the [overview of interfaces](interface_overview.md) for a comparison of the available options.

- The tutorial is recommended for the new users of FlexTool: [Tutorial](tutorial.md)
- How-to section has examples on how to add specific features to a model: [How to](how_to.md)
- More advanced users can find the model parameter descriptions useful: [Model parameters](reference.md)
- Finally, result parameters are documented here: [Model results](results.md)

# Monthly user support telcos

The monthly user support telco is held on the last Monday of each month at 12-13 UTC (skipping December and July). (Notice that time is according to UTC and in places where day-light saving time is applied, the time of the meeting may change between winter/summer) Each 1 h session starts with a ~15 min presentation on simple IRENA FlexTool demos or tutorials, followed by 45 min Q&A session.

[Teams link](https://teams.microsoft.com/l/meetup-join/19%3ameeting_MmRkYzAyNzktOTVhZS00NzAyLWI5OTItZTg4ZjhlM2I3NDc5%40thread.v2/0?context=%7b%22Tid%22%3a%2268d6b592-5008-43b5-9b04-23bec4e86cf7%22%2c%22Oid%22%3a%225138c2b5-7b5a-472e-9793-addd3b524ae7%22%7d)

Please contact anni.niemi@vtt.fi for an Outlook calendar invitation.

Recordings and presentations of the past support calls can be found from [here](https://drive.google.com/drive/folders/1cqEqCRpAEjZ24by3BjiWSxPv6cXed7Ib).



# Background for the FlexTool modelling approach

The theory slides below give some background how FlexTool is formulated. There are also examples that show some ways how FlexTool can be used (including examples from other similar models). The slides were made for training in the OASES project (funded by LEAP-RE, project no: 963530, co-funding from European Commission and national funding agencies). The files can also be found in the folder docs/theory_slides.

[1: Energy planning and types of modelling approaches](./theory_slides/Session1_Energy_planning_and_types_of_modelling_approaches.pdf)

[2: Modelling tools process and IRENA Flextool approach](./theory_slides/Session2_Modelling_tools_process_and_IRENA_FlexTool_approach.pdf)

[3: IRENA Flextool in practice](./theory_slides/Session3_IRENA_FlexTool_in_practice.pdf)

[4: Examples of studies done with IRENA Flextool approach](./theory_slides/Examples_of_studies_done_with_IRENA_FlexTool_approach.pdf)
