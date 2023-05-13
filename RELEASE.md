** Release 3.1.0

Bug fixes
- Division by zero capacity for units with cf
- Output fixed costs for existing units
- Fix output calculation for annualized fixed costs
- Fix error in importing CO2_methods for groups
- Add other_variable_cost for flows from unit to sink
- 


New features
- New investment method with a limited horizon (model used to assume infinite horizon)
  - *Braking change*: new parameter 'years_represented', which replaces 'discount_years'. It is a map to indicate how many years the period represents before the next period in the solve. Used for discounting. Can be below one (multiple periods in one year). Index: period, value: years.
- Added sidebar to documentation (gh-pages branch)
- Default solver changed from GLPSOL to HiGHS
- Documentation improvements
- Scaling of the model resulting in very big performance improvement for linear problems
- Improved plotting for the web browser version
- In Spine Toolbox, plotting is based on the specification from the browser version
- More results: group flows over time
- More results: costs for whole model run
- More result plots




** Release 3.0.1

Bug fixes
- CO2_method for groups was not imported correctly from Excel inputs
- Fix other_variable_cost in units for flows from unit to sink (source to unit was working)

New features
- Output fixed costs for existing units
- Added outputs to show average capacity factors for periods

** Release 3.0.0

All planned features implemented


----
TEMPLATE
** Release 

Bug fixes
- 

New features
- 
