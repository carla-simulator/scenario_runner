# OpenSCENARIO Support

The scenario_runner provides support for the upcoming [OpenSCENARIO](http://www.openscenario.org/) standard.
The current implementation covers initial support for maneuver Actions, Conditions, Stories and the Storyboard.
If you would like to use evaluation criteria for a scenario to evaluate pass/fail results, these can be implemented
as EndConditions. However, not all features for these elements are yet available. If in doubt, please see the
module documentation in srunner/tools/openscenario_parser.py

An example for a supported scenario based on OpenSCENARIO is available [here](../srunner/examples/FollowLeadingVehicle.xosc)


## Overview of available features of OpenSCENARIO v0.9:
- [ ] Catalogs
- [ ] Use of parameter
- [ ] RoadNetwork:
    * [x] Logics (OpenDrive): Specifying the OpenDrive file is supported
    * [ ] OpenSceneGraph:
    * [ ] Signals
- [x] Entities: Defining different entities is supported, with a limitation on the position definition
- [x] Positions: Can only be defined w.r.t the world frame (i.e. global coordinates)
- [ ] Controllers: Use of different (vehicle) controllers is not available
- [x] Storyboard: The Storyboard with repeated sequences is supported
- [ ] Maneuver actions:
    * [ ] Longitudinal:
       * [x] Speed
       * [ ] Distance 
    * [ ] Lateral
    * [ ] Visibility
    * [ ] Meeting
    * [x] Autonomous
    * [ ] Controller
    * [x] Position
    * [ ] Routing
       * [x] FollowRoute
       * [ ] FollowTrajectory
       * [ ] AquirePosition
    * [x] Command (Support for command 'Idle')
    * [ ] Script
    * [ ] SetEnvironment (Weather is supported)
    * [ ] Entity
    * [ ] Parameter
    * [ ] Traffic
    * [ ] Infrastructure
- [ ] Conditions
    * [ ] EndOfRoad 
    * [x] Collision
    * [ ] Offroad
    * [ ] TimeHeadway
    * [x] TimeToCollision
    * [ ] Acceleration 
    * [x] StandStill
    * [x] Speed
    * [ ] RelativeSpeed
    * [x] TraveledDistance
    * [x] ReachPosition
    * [x] Distance (not all comparisions are supported)
    * [x] RelativeDistance 
    * [ ] AtStart 
    * [ ] AfterTermination
    * [ ] Command
    * [ ] Signal
    * [ ] Controller
    * [x] Parameter
    * [ ] TimeOfDay
    * [x] SimulationTime

## Roadmap of planned extensions

- Maneuver actions:
  * Lane change (lateral action): September 2019
- Conditions:
  * Complete distance checks: August/September 2019
  * AtStart: August/September 2019
  * AfterTermination: August/September 2019