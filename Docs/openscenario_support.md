# OpenScenario Support

The scenario_runner provides support for the upcoming [OpenScenario](http://www.openscenario.org/) standard.
The current implementation covers initial support for maneuver Actions, Conditions, Stories and the Storyboard.
If you would like to use evaluation criteria for a scenario to evaluate pass/fail results, these can be implemented
as EndConditions. However, not all features for these elements are yet available. If in doubt, please see the
module documentation in srunner/tools/openscenario_paser.py

An example for a supported scenario based on OpenScenario is available [here](../srunner/configs/FollowLeadingVehicle.xosc) 