# OpenScenario Support

The scenario_runner provides support for the upcoming [OpenScenario](http://www.openscenario.org/) standard.
The current implementation covers initial support for maneuver Actions, Conditions, Stories and the Storyboard.
If you would like to use evaluation criteria for a scenario to evaluate pass/fail results, these can be implemented
as EndConditions. However, not all features for these elements are yet available. If in doubt, please see the
module documentation in srunner/tools/openscenario_paser.py

An example for a supported scenario based on OpenScenario is available [here](../srunner/configs/FollowLeadingVehicle.xosc) 


## Overview of available features of OpenScenario v0.9:
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
    * [ ] SetEnvironment
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
    * [x] Distance
    * [x] RelativeDistance 
    * [ ] AtStart 
    * [ ] AfterTermination
    * [ ] Command
    * [ ] Signal
    * [ ] Controller
    * [x] Parameter
    * [ ] TimeOfDay
    * [x] SimulationTime
